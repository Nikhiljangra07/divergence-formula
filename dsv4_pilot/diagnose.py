"""
diagnose.py — review the candidates that did NOT survive the DAV gate, and explain WHY.

Reconstructs every generated candidate from candidates_raw.jsonl, re-scores each with the
EXACT gate logic imported from generate.py (single source of truth), and for every non-passer
prints which gate(s) failed, by how much, and — for consequence failures — which specific thread
broke and why (too short, or no consequence projected). Also flags mid-sentence truncation, a
quality defect the DAV gate does not catch.

Isolated: reads only OPENAI_API_KEY (embeddings) + the local out/ files. Writes out/rejects_review.md.
Run: python3 diagnose.py
"""
import json, os, importlib.util
from pathlib import Path
import numpy as np
import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
KEY = os.environ["OPENAI_API_KEY"]

# import gate constants + functions from generate.py (main() is guarded, so nothing runs)
spec = importlib.util.spec_from_file_location("g", HERE / "generate.py")
g = importlib.util.module_from_spec(spec)
exec(compile(open(HERE / "generate.py").read(), "generate.py", "exec"), g.__dict__)

TERMINAL = tuple('.!?"”’\')')  # a complete thread should end on one of these


def embed(texts):
    out = []
    for i in range(0, len(texts), 256):
        r = httpx.post("https://api.openai.com/v1/embeddings",
                       headers={"Authorization": f"Bearer {KEY}"},
                       json={"model": "text-embedding-3-small", "input": texts[i:i+256]}, timeout=60)
        r.raise_for_status(); out += [d["embedding"] for d in r.json()["data"]]
    V = np.array(out, float); return V / np.linalg.norm(V, axis=1, keepdims=True)


def gate_fails(vol, grd, whl, pd, threads):
    """Return list of (gate, margin, detail) for every floor this candidate misses."""
    fails = []
    if vol <= g.VOL_GATE:
        fails.append(("volume", g.VOL_GATE - vol, "threads collapsed onto one direction"))
    if grd < g.EPS_G:
        fails.append(("ground", g.EPS_G - grd, "a thread drifts off the actual problem"))
    if whl < g.EPS_W:
        fails.append(("whole", g.EPS_W - whl, "a thread ignores the key facets"))
    if pd < g.PD_FLOOR:
        fails.append(("pairdist", g.PD_FLOOR - pd, "two threads are near-duplicates"))
    if not g.consequence_ok(threads):
        # pinpoint the offending thread(s)
        bad = []
        for i, t in enumerate(threads):
            w = len(t.split()); hit = bool(g.CONS_RE.search(t))
            if w < 18 or not hit:
                why = []
                if w < 18: why.append(f"only {w} words (<18)")
                if not hit: why.append("no cost/gain/trade language")
                bad.append(f"thread {i+1}: {'; '.join(why)}")
        fails.append(("consequence", 1.0, " | ".join(bad)))
    return fails


def main():
    raw = [json.loads(l) for l in open(HERE / "out/candidates_raw.jsonl")]
    passer_probs = {json.loads(l)["problem"] for l in open(HERE / "out/passers.jsonl")}

    # embed everything once
    texts, spans = [], []
    for c in raw:
        s = {"p": len(texts)}; texts.append(c["problem"])
        s["f"] = len(texts); texts += c["facets"]
        s["t"] = len(texts); texts += c["threads"]; spans.append(s)
    V = embed(texts)

    losers = []
    for c, s in zip(raw, spans):
        p = V[s["p"]]; f = V[s["f"]:s["f"]+3]; t = V[s["t"]:s["t"]+len(c["threads"])]
        c["vol"] = g.volume(t); c["grd"] = float((t @ p).min())
        c["whl"] = float((t @ f.T).mean(1).min()); c["pd"] = g.pairdist(t)
        c["fails"] = gate_fails(c["vol"], c["grd"], c["whl"], c["pd"], c["threads"])
        c["truncated"] = [i+1 for i, th in enumerate(c["threads"]) if not th.rstrip().endswith(TERMINAL)]
        is_passer = c["problem"] in passer_probs
        if not is_passer:
            losers.append(c)

    # classify: a non-passer with NO failing gate actually PASSED the gate and was only
    # cut by the KEEP=50 cap (rank 51+). Don't slander those as rejects.
    def kind(c):
        fl = c["fails"]
        if not fl:
            return "cut_by_cap"   # gate-passed; lost only to the top-50 cap
        if len(fl) == 1 and fl[0][0] != "consequence" and fl[0][1] <= g.NEAR_MARGIN:
            return "near_miss"
        return "reject"
    for c in losers:
        c["kind"] = kind(c)

    # ---- console summary ----
    from collections import Counter
    cnt = Counter(f[0] for c in losers for f in c["fails"])
    gate_passed_total = len(raw) - sum(c["kind"] in ("reject", "near_miss") for c in losers)
    print(f"generated {len(raw)} | gate-PASSED {gate_passed_total} "
          f"(kept 50 + {sum(c['kind']=='cut_by_cap' for c in losers)} cut by the top-50 cap) "
          f"| gate-FAILED {sum(c['kind'] in ('reject','near_miss') for c in losers)} "
          f"(reject {sum(c['kind']=='reject' for c in losers)}, near-miss {sum(c['kind']=='near_miss' for c in losers)})")
    print("failing-gate tally (a loser can miss several):", dict(cnt))
    trunc = [c for c in raw if c["truncated"]]
    print(f"mid-sentence truncations across ALL {len(raw)} generated: {len(trunc)} candidates")
    print()
    for c in sorted(losers, key=lambda c: (c["kind"], -max((f[1] for f in c["fails"]), default=0))):
        tag = ",".join(f"{f[0]}(+{f[1]:.2f})" if f[0] != "consequence" else "consequence" for f in c["fails"])
        print(f"[{c['kind']:9}] {c['source']}  vol={c['vol']:.2f} grd={c['grd']:.2f} whl={c['whl']:.2f} pd={c['pd']:.2f}  FAILS: {tag}")
        print(f"   Q: {c['problem'][:95]}")
        for gate, margin, detail in c["fails"]:
            print(f"      └─ {gate}: {detail}")

    # ---- full-text review file ----
    md = HERE / "out/rejects_review.md"
    with md.open("w") as out:
        out.write(f"# DAV losers — {len(losers)} of {len(raw)} generated (full text + diagnosis)\n\n")
        for n, c in enumerate(sorted(losers, key=lambda c: c["kind"]), 1):
            out.write(f"## {n}. [{c['source']}] {c['kind'].upper()}  "
                      f"vol={c['vol']:.2f} grd={c['grd']:.2f} whl={c['whl']:.2f} pd={c['pd']:.2f}\n\n")
            out.write(f"**{c['problem']}**\n\n")
            out.write("**Why it lost:**\n")
            for gate, margin, detail in c["fails"]:
                out.write(f"- `{gate}` — {detail}\n")
            if c["truncated"]:
                out.write(f"- `truncated` — thread(s) {c['truncated']} cut off mid-sentence\n")
            out.write("\n_facets:_ " + " · ".join(c["facets"]) + "\n\n")
            for i, t in enumerate(c["threads"], 1):
                mark = " ⟶ TRUNCATED" if i in c["truncated"] else ""
                out.write(f"{i}. {t}{mark}\n")
            out.write("\n")
    print(f"\nfull-text review → {md}")


if __name__ == "__main__":
    main()
