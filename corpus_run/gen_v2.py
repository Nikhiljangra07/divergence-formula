"""
gen_v2.py — v2 corpus generator (consequence-only) with the probe's fixes baked in.

FIXES vs v1 (diagnosed from the granite probe):
  1. REFRACTOR: the four angles must span FOUR DIFFERENT FAMILIES of strategic move (not four
     rewordings of one) — this is the divergence-ceiling fix. Angles now carry a `family` label
     AND are PERSISTED (v1 dropped them; no Haiku reconstruction needed this time).
  2. WORKER: keep the consequence (= depth / decidable value), but (a) demand mechanical specificity
     + a non-obvious insight (kill 'surface' threads), and (b) VARY the consequence grammar so it
     stops hardening into one 'costs X gains Y' skeleton that compresses the embeddings.
  No without-consequence variant — single positive corpus + hard negatives.

Pilot: generate TARGET examples, DAV-gate, and report angle-pairdist (NEW) + thread-pairdist vs v1.
Safe: bounded candidate concurrency + per-candidate flush + 60s call timeout (no hangs). Money cap.
Run under caffeinate. Output: corpus_v2/.
"""
from __future__ import annotations
import argparse, asyncio, json, os, time
from pathlib import Path
import numpy as np, httpx
from dotenv import load_dotenv

import config as C
import dav

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_v2"; OUT.mkdir(exist_ok=True)
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]; OAI_KEY = os.environ["OPENAI_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

HARD_USD_CAP = float(os.environ.get("V2_USD_CAP", "1.5"))
CONCURRENCY, CAND_CONCURRENCY = 12, 16
TIMEOUT, CAND_TIMEOUT, RETRIES, THREAD_REGENS = 60.0, 200.0, 4, 2
REFRACTOR_MAXTOK, WORKER_MAXTOK = 3000, 1600
K = 4

REFRACTOR = (
    "You are preparing a decision dilemma for divergent analysis.\n\nSOURCE: {brief}\n\n"
    "DILEMMA TYPE for THIS example (instantiate exactly this, do not drift to another): {theme}\n\n"
    "Produce ONE concrete, faithful dilemma of EXACTLY this type that a figure in this world genuinely "
    "faces — fresh, not the single most famous instance. Then three key FACETS, then FOUR strategic ANGLES.\n\n"
    "CRITICAL — DIVERGENCE BY MOVE-TYPE: the four angles must each be a categorically DIFFERENT KIND of "
    "strategic move, not four versions of one. Assign each angle a DISTINCT family chosen from:\n"
    "  CONFRONT/ELIMINATE · EVADE/DEFER · CO-OPT/INTEGRATE · TRANSFORM/REFRAME · DELEGATE/EXTERNALIZE · ENDURE/SACRIFICE\n"
    "All four families must differ. Before finalizing, check: if two angles are the same underlying move "
    "(e.g. both 'scapegoat a proxy', both 'kill the threat'), REPLACE one until all four families are distinct "
    "and lead to genuinely different actions.\n\n"
    "Return STRICT JSON only, no prose:\n"
    '{{"problem":"<1-2 sentence dilemma>","facets":["<f1>","<f2>","<f3>"],'
    '"angles":[{{"family":"<FAMILY>","directive":"<angle directive>"}},'
    '{{"family":"<FAMILY>","directive":"<angle>"}},{{"family":"<FAMILY>","directive":"<angle>"}},'
    '{{"family":"<FAMILY>","directive":"<angle>"}}]}}'
)

WORKER = (
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; you "
    "are blind to the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE [{family}]: {angle}\n\n"
    "Write a single thread (two COMPLETE sentences, 40-60 words, ending in a full stop) that COMMITS to "
    "this angle as a concrete, mechanically specific strategy for the WHOLE problem — name the actual move "
    "(who, what, how), never a generic gesture — and ENDS by projecting its concrete consequence: what it "
    "COSTS and what it GAINS.\n"
    "DEPTH: every thread must carry real strategic substance and a non-obvious insight; no surface "
    "restatement of the angle.\n"
    "VARY the grammar of the consequence — do NOT reuse a fixed 'costs X but gains Y' skeleton; sometimes "
    "lead with the gain, sometimes fuse cost and gain into the move itself.\n"
    "VOICE: cold, analytical, decisive. No hedging, no therapy language, no lists. Output ONLY the thread."
)


class Ledger:
    def __init__(self, cap): self.cap, self.spent, self.calls = cap, 0.0, 0; self.lock = asyncio.Lock()
    async def charge(self, u):
        async with self.lock:
            self.spent += u.get("prompt_tokens", 0) * C.PRICE_IN + u.get("completion_tokens", 0) * C.PRICE_OUT
            self.calls += 1
    def over(self): return self.spent >= self.cap


LED = Ledger(HARD_USD_CAP); SEM = asyncio.Semaphore(CONCURRENCY); CAND_SEM = asyncio.Semaphore(CAND_CONCURRENCY)


async def dsv4(client, prompt, max_tokens, temp):
    if LED.over(): return None
    body = {"model": C.MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    async with SEM:
        for a in range(RETRIES):
            try:
                r = await client.post(OR_URL, headers=headers, json=body, timeout=TIMEOUT)
                r.raise_for_status(); d = r.json()
                await LED.charge(d.get("usage", {}))
                msg = (d["choices"][0]["message"]["content"] or "").strip()
                if msg: return msg
            except Exception as e:
                if a == RETRIES - 1: print(f"  [dsv4] {type(e).__name__}", flush=True)
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


async def one_thread(client, problem, facets_str, ang):
    for _ in range(THREAD_REGENS + 1):
        t = await dsv4(client, WORKER.format(problem=problem, facets=facets_str,
                                             family=ang["family"], angle=ang["directive"]), WORKER_MAXTOK, 0.75)
        if t and t.strip(): return t.strip()
    return None


async def gen_one(client, src, theme):
    refr = await dsv4(client, REFRACTOR.format(brief=C.SOURCES[src], theme=theme), REFRACTOR_MAXTOK, 0.95)
    spec = parse_json(refr)
    if not spec or not all(k in spec for k in ("problem", "facets", "angles")): return None
    # tolerant angle normalization — recover dicts missing 'family' and bare-string angles (yield)
    angles = []
    for a in spec["angles"][:K]:
        if isinstance(a, dict) and a.get("directive"):
            angles.append({"family": str(a.get("family", "UNSPECIFIED")).strip(),
                           "directive": str(a["directive"]).strip()})
        elif isinstance(a, str) and a.strip():
            angles.append({"family": "UNSPECIFIED", "directive": a.strip()})
    if len(angles) < K: return None
    facets_str = "; ".join(spec["facets"][:3])
    threads = await asyncio.gather(*[one_thread(client, spec["problem"], facets_str, a) for a in angles])
    if any(t is None for t in threads): return None
    return {"source": src, "theme": theme, "problem": spec["problem"], "facets": spec["facets"][:3],
            "angles": angles, "threads": [t for t in threads]}


def balanced_plan(n):
    srcs = list(C.SOURCES); plan = []
    for i in range(n):
        s = srcs[i % len(srcs)]; th = C.THEMES[s]; plan.append((s, th[(i // len(srcs)) % len(th)]))
    return plan


def embed(texts):
    out = []
    for i in range(0, len(texts), 256):
        r = httpx.post("https://api.openai.com/v1/embeddings", headers={"Authorization": f"Bearer {OAI_KEY}"},
                       json={"model": "text-embedding-3-small", "input": texts[i:i+256]}, timeout=60)
        r.raise_for_status(); out += [d["embedding"] for d in r.json()["data"]]
    V = np.array(out, float); return V / np.linalg.norm(V, axis=1, keepdims=True)


def unit(V): return V / np.linalg.norm(V, axis=1, keepdims=True)


def score(cands):
    texts, spans = [], []
    for c in cands:
        s = {"p": len(texts)}; texts.append(c["problem"])
        s["f"] = len(texts); texts += c["facets"]
        s["a"] = len(texts); texts += [a["directive"] for a in c["angles"]]
        s["t"] = len(texts); texts += c["threads"]; spans.append(s)
    V = embed(texts)
    for c, s in zip(cands, spans):
        p = V[s["p"]]; f = V[s["f"]:s["f"]+3]; ang = V[s["a"]:s["a"]+4]; t = V[s["t"]:s["t"]+4]
        c["m"] = {"volume": dav.volume(t), "ground": float((t @ p).min()),
                  "whole": float((t @ f.T).mean(1).min()), "pairdist": dav.pairdist(t),
                  "angle_pairdist": dav.pairdist(ang)}
        fams = {a["family"].upper().split("/")[0].strip() for a in c["angles"]}
        c["m"]["distinct_families"] = len(fams)
        c["cons_ok"] = dav.consequence_ok(c["threads"]) and all(t.rstrip().endswith(dav.TERMINAL) for t in c["threads"])
    return cands


def v1_angle_pairdist():
    """Baseline: angle-pairdist of the v1 (A) corpus, from its reconstructed angles."""
    p = HERE / "corpus/set_a_with_consequence/decomposer.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.open()][:60]
    texts, spans = [], []
    for r in rows:
        spans.append(len(texts)); texts += r["angles"][:4]
    V = embed(texts)
    pds = [dav.pairdist(V[s:s+4]) for s in spans if len(V[s:s+4]) == 4]
    return float(np.mean(pds))


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, default=40); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    print(f"v2 pilot — target {args.target} | cap ${HARD_USD_CAP}", flush=True)
    raw = OUT / "candidates_raw.jsonl"; raw.write_text("")
    cands = []
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        if args.smoke:
            c = await gen_one(client, "RI", C.THEMES["RI"][2])
            print(json.dumps(c, indent=2)[:1500] if c else "SMOKE FAILED"); return
        plan = balanced_plan(int(args.target / 0.78 * 1.2))
        async def bounded(s, th):
            async with CAND_SEM:
                try: return await asyncio.wait_for(gen_one(client, s, th), timeout=CAND_TIMEOUT)
                except Exception: return None
        tasks = [asyncio.ensure_future(bounded(s, th)) for s, th in plan]
        with raw.open("a") as f:
            for fut in asyncio.as_completed(tasks):
                c = await fut
                if c:
                    f.write(json.dumps(c) + "\n"); f.flush(); cands.append(c)
                    if len(cands) >= int(args.target / 0.78 * 1.2): pass
    print(f"generated {len(cands)} candidates in {time.time()-t0:.0f}s | spend ${LED.spent:.2f}", flush=True)
    score(cands)
    passers = [c for c in cands if c["m"]["volume"] > C.VOL_GATE and c["m"]["ground"] >= C.EPS_G
               and c["m"]["whole"] >= C.EPS_W and c["m"]["pairdist"] >= C.PD_FLOOR and c["cons_ok"]]
    passers.sort(key=lambda c: c["m"]["pairdist"], reverse=True)
    passers = passers[:args.target]
    with (OUT / "passers.jsonl").open("w") as f:
        for c in passers:
            f.write(json.dumps({"problem": c["problem"], "facets": c["facets"], "angles": c["angles"],
                                "threads": c["threads"]}) + "\n")
    with (OUT / "passers.md").open("w") as f:
        for i, c in enumerate(passers, 1):
            m = c["m"]
            f.write(f"## {i}. [{c['source']}] pd={m['pairdist']:.3f} angle_pd={m['angle_pairdist']:.3f} "
                    f"fam={m['distinct_families']}\n**{c['problem']}**\n")
            for a, t in zip(c["angles"], c["threads"]): f.write(f"- [{a['family']}] {t}\n")
            f.write("\n")

    mean = lambda k: round(float(np.mean([c["m"][k] for c in passers])), 4)
    v1_apd = v1_angle_pairdist()
    summary = {
        "target": args.target, "generated": len(cands), "passers": len(passers),
        "pass_rate": round(len(passers)/max(len(cands),1), 3),
        "v2_means": {k: mean(k) for k in ("pairdist","angle_pairdist","ground","whole","volume","distinct_families")},
        "v1_thread_pairdist_A": 0.385, "v1_angle_pairdist_A": round(v1_apd,4) if v1_apd else None,
        "spend_usd": round(LED.spent, 3),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== V2 PILOT RESULT ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
