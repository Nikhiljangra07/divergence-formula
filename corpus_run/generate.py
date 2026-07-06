"""
generate.py — build the three-corpus environment. THEME-CONDITIONED, strict DAV gate, isolated + safe.

  python generate.py set_a   --target 300   # WITH consequence  (DAV_full)
  python generate.py set_b   --target 300   # WITHOUT consequence (DAV_lite)
  python generate.py aggregate              # build hard_negatives/ + MANIFEST.json from set_a + set_b

Outputs (corpus/):
  set_a_with_consequence/    passers.jsonl (UNLABELED) · passers.md · rejects.jsonl (FULL TEXT) · near_miss.jsonl · summary.json
  set_b_without_consequence/ (same shape)
  hard_negatives/            negatives.jsonl · near_miss_premium.jsonl · summary.json
  MANIFEST.json              top-level counts / config / thresholds / spend

Safety: reads only OPENROUTER_API_KEY (gen) + OPENAI_API_KEY (embeddings); writes only under corpus/.
Hard money cap with a live ledger; save-as-you-go; theme round-robin guarantees landscape spread.
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from collections import Counter
from pathlib import Path
import httpx
from dotenv import load_dotenv

import config as C
import dav

HERE = Path(__file__).resolve().parent
CORP = HERE / "corpus"
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]
OAI_KEY = os.environ["OPENAI_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

SETTINGS = {
    "set_a": {"dir": "set_a_with_consequence",    "worker": C.WORKER_A, "consequence_gate": True,  "yield": C.YIELD_A},
    "set_b": {"dir": "set_b_without_consequence", "worker": C.WORKER_B, "consequence_gate": False, "yield": C.YIELD_B},
}

HARD_USD_CAP = float(os.environ.get("CORPUS_USD_CAP", "8.0"))   # surfaced; override via env if needed


class Ledger:
    def __init__(self, cap): self.cap, self.spent, self.calls = cap, 0.0, 0; self.lock = asyncio.Lock()
    async def charge(self, u):
        c = u.get("prompt_tokens", 0) * C.PRICE_IN + u.get("completion_tokens", 0) * C.PRICE_OUT
        async with self.lock: self.spent += c; self.calls += 1
    def over(self): return self.spent >= self.cap


LED = Ledger(HARD_USD_CAP); SEM = asyncio.Semaphore(C.CONCURRENCY)
CAND_SEM = asyncio.Semaphore(C.CAND_CONCURRENCY)  # bounds candidates in flight (anti-starvation)


async def dsv4(client, prompt, max_tokens, temp):
    if LED.over(): return None
    body = {"model": C.MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    async with SEM:
        for a in range(C.RETRIES):
            try:
                r = await client.post(OR_URL, headers=headers, json=body, timeout=C.TIMEOUT)
                r.raise_for_status(); d = r.json()
                await LED.charge(d.get("usage", {}))
                msg = (d["choices"][0]["message"]["content"] or "").strip()
                if msg: return msg
            except Exception as e:
                if a == C.RETRIES - 1: print(f"  [dsv4] fail: {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(2 * (a + 1))
    return None


def parse_json(txt):
    if not txt: return None
    try: return json.loads(txt)
    except Exception: pass
    i = txt.find("{")
    if i < 0: return None
    depth = 0
    for j in range(i, len(txt)):
        if txt[j] == "{": depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(txt[i:j+1])
                except Exception: return None
    return None


async def one_thread(client, worker, problem, facets_str, angle):
    for _ in range(C.THREAD_REGENS + 1):
        t = await dsv4(client, worker.format(problem=problem, facets=facets_str, angle=angle),
                       C.WORKER_MAXTOK, 0.75)
        if t and t.strip():
            return t.strip()
    return None


async def gen_one(client, worker, src, theme):
    refr = await dsv4(client, C.REFRACTOR.format(brief=C.SOURCES[src], theme=theme), C.REFRACTOR_MAXTOK, 0.95)
    spec = parse_json(refr)
    if not spec or not all(k in spec for k in ("problem", "facets", "angles")): return None
    angles = spec["angles"][:C.K]
    if len(angles) < C.K: return None
    facets_str = "; ".join(spec["facets"][:3])
    threads = await asyncio.gather(*[one_thread(client, worker, spec["problem"], facets_str, a) for a in angles])
    if any(t is None for t in threads): return None
    return {"source": src, "theme": theme, "problem": spec["problem"],
            "facets": spec["facets"][:3], "threads": [t for t in threads]}


def balanced_plan(n, start):
    """n (source, theme) pairs cycling sources every step, advancing theme each full cycle."""
    srcs = list(C.SOURCES); plan = []
    for i in range(n):
        s = srcs[(start + i) % len(srcs)]
        th = C.THEMES[s]; plan.append((s, th[((start + i) // len(srcs)) % len(th)]))
    return plan


def score_and_gate(cands, consequence_gate):
    texts, spans = [], []
    for c in cands:
        s = {"p": len(texts)}; texts.append(c["problem"])
        s["f"] = len(texts); texts += c["facets"]
        s["t"] = len(texts); texts += c["threads"]; spans.append(s)
    V = dav.embed(texts, OAI_KEY)
    for c, s in zip(cands, spans):
        p = V[s["p"]]; f = V[s["f"]:s["f"]+3]; t = V[s["t"]:s["t"]+len(c["threads"])]
        c["metrics"] = dav.score(p, f, t)
        c["fails"] = dav.gate_fails(c["metrics"], c["threads"], consequence_gate)
        c["verdict"] = dav.verdict(c["fails"])
    return cands


async def run_set(key, target):
    cfg = SETTINGS[key]
    odir = CORP / cfg["dir"]; odir.mkdir(parents=True, exist_ok=True)
    raw_path = odir / "candidates_raw.jsonl"; raw_path.write_text("")
    print("=" * 70)
    print(f"{key}  ({cfg['dir']})  target={target} passers | consequence_gate={cfg['consequence_gate']} "
          f"| cap ${HARD_USD_CAP:.2f}")
    print("=" * 70)

    passers, near, rejects = [], [], []
    cursor = 0; t0 = time.time()
    async with httpx.AsyncClient() as client:
        while len(passers) < target and not LED.over():
            deficit = target - len(passers)
            batch_n = max(8, int(deficit / cfg["yield"] * 1.15))
            plan = balanced_plan(batch_n, cursor); cursor += batch_n

            async def bounded(s, th):
                async with CAND_SEM:          # only CAND_CONCURRENCY candidates active at once
                    try:                       # timeout starts HERE (after acquiring the slot), not at creation
                        return await asyncio.wait_for(
                            gen_one(client, cfg["worker"], s, th), timeout=C.CANDIDATE_TIMEOUT)
                    except Exception:
                        return None            # genuinely stuck — abandon, never block the batch

            cands = []
            tasks = [asyncio.ensure_future(bounded(s, th)) for s, th in plan]
            with raw_path.open("a") as f:
                for fut in asyncio.as_completed(tasks):     # flush each candidate AS IT COMPLETES
                    c = await fut
                    if c:
                        f.write(json.dumps(c) + "\n"); f.flush(); cands.append(c)
            if not cands:
                print("  batch produced 0 candidates — stopping."); break
            score_and_gate(cands, cfg["consequence_gate"])
            for c in cands:
                (passers if c["verdict"] == "pass" else near if c["verdict"] == "near_miss" else rejects).append(c)
            print(f"  batch {batch_n:>3} -> +{sum(c['verdict']=='pass' for c in cands)} pass "
                  f"| total passers {len(passers)}/{target} | calls={LED.calls} spend=${LED.spent:.2f}", flush=True)
            if batch_n and not cands: break

    passers.sort(key=lambda c: c["metrics"]["pairdist"], reverse=True)
    _write_set(odir, key, cfg, passers, near, rejects, target, time.time() - t0)
    return passers, near, rejects


def _md(c):
    return c["metrics"]


def _write_set(odir, key, cfg, passers, near, rejects, target, secs):
    # passers — UNLABELED training file
    with (odir / "passers.jsonl").open("w") as f:
        for c in passers:
            f.write(json.dumps({"problem": c["problem"], "facets": c["facets"], "threads": c["threads"]}) + "\n")
    # rejects + near-miss — FULL TEXT + diagnosis (the hard-negative source)
    def dump(path, items):
        with (odir / path).open("w") as f:
            for c in items:
                f.write(json.dumps({"source": c["source"], "theme": c["theme"], "problem": c["problem"],
                                    "facets": c["facets"], "threads": c["threads"],
                                    "metrics": {k: round(v, 3) for k, v in c["metrics"].items()},
                                    "fails": [[g, round(m, 3), d] for g, m, d in c["fails"]]}) + "\n")
    dump("rejects.jsonl", rejects)
    dump("near_miss.jsonl", near)
    # human QA markdown (labeled)
    with (odir / "passers.md").open("w") as f:
        f.write(f"# {len(passers)} passers — {cfg['dir']} (unlabeled in jsonl; source/theme shown for QA)\n\n")
        for i, c in enumerate(passers, 1):
            m = _md(c)
            f.write(f"## {i}. [{c['source']}] _{c['theme'][:48]}_  vol={m['volume']:.2f} pd={m['pairdist']:.2f} "
                    f"grd={m['ground']:.2f} whl={m['whole']:.2f}\n\n**{c['problem']}**\n\n")
            for t in c["threads"]: f.write(f"- {t}\n")
            f.write("\n")
    # summary
    src_mix = Counter(c["source"] for c in passers)
    theme_mix = Counter(c["theme"][:40] for c in passers)
    import numpy as np
    pm = {k: round(float(np.mean([c["metrics"][k] for c in passers])), 3) for k in
          ("volume", "ground", "whole", "pairdist")} if passers else {}
    summary = {
        "setting": key, "dir": cfg["dir"], "consequence_gate": cfg["consequence_gate"],
        "target": target, "passers": len(passers), "near_miss": len(near), "rejects": len(rejects),
        "passer_means": pm,
        "source_mix": dict(src_mix),
        "themes_covered": len(theme_mix),
        "reject_reasons": dict(Counter(g for c in rejects for g, _, _ in c["fails"])),
        "spend_usd": round(LED.spent, 3), "dsv4_calls": LED.calls, "seconds": round(secs),
    }
    (odir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  passers {len(passers)} | near {len(near)} | reject {len(rejects)} "
          f"| themes {len(theme_mix)} | src {dict(src_mix)}")
    print(f"  means {pm} | spend ${LED.spent:.2f} | {round(secs)}s")
    print(f"  read -> {odir/'passers.md'}")


def aggregate():
    """Build hard_negatives/ from both sets' rejects + near-miss, plus a top-level MANIFEST."""
    hn = CORP / "hard_negatives"; hn.mkdir(parents=True, exist_ok=True)
    negs, premium = [], []
    per_set = {}
    for key, cfg in SETTINGS.items():
        odir = CORP / cfg["dir"]
        rj = [json.loads(l) for l in (odir / "rejects.jsonl").open()] if (odir / "rejects.jsonl").exists() else []
        nm = [json.loads(l) for l in (odir / "near_miss.jsonl").open()] if (odir / "near_miss.jsonl").exists() else []
        for c in rj: c["origin"] = key; negs.append(c)
        for c in nm: c["origin"] = key; premium.append(c); negs.append(c)
        per_set[key] = {"rejects": len(rj), "near_miss": len(nm)}
    with (hn / "negatives.jsonl").open("w") as f:
        for c in negs: f.write(json.dumps(c) + "\n")
    with (hn / "near_miss_premium.jsonl").open("w") as f:
        for c in premium: f.write(json.dumps(c) + "\n")
    (hn / "summary.json").write_text(json.dumps({
        "total_negatives": len(negs), "premium_near_miss": len(premium), "per_set": per_set,
        "fail_tally": dict(Counter(g for c in negs for g, *_ in c.get("fails", []))),
    }, indent=2))
    # manifest
    def load(p):
        return json.loads((CORP / p / "summary.json").read_text()) if (CORP / p / "summary.json").exists() else None
    manifest = {
        "set_a_with_consequence": load("set_a_with_consequence"),
        "set_b_without_consequence": load("set_b_without_consequence"),
        "hard_negatives": json.loads((hn / "summary.json").read_text()),
        "thresholds": {"VOL_GATE": C.VOL_GATE, "EPS_G": C.EPS_G, "EPS_W": C.EPS_W,
                       "PD_FLOOR": C.PD_FLOOR, "MIN_WORDS": C.MIN_WORDS},
        "model": C.MODEL,
    }
    (CORP / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"hard_negatives: {len(negs)} (premium {len(premium)})  ->  {hn}")
    print(f"MANIFEST -> {CORP/'MANIFEST.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["set_a", "set_b", "aggregate"])
    ap.add_argument("--target", type=int, default=300)
    args = ap.parse_args()
    if args.mode == "aggregate":
        aggregate()
    else:
        asyncio.run(run_set(args.mode, args.target))


if __name__ == "__main__":
    main()
