"""
regate_v34.py — re-gate the v3/v4 corpora (402 rows) through the v5 6-dim judge for the H-Small blend run.

Gate = SHARPNESS ONLY: distinctness>=4 AND concreteness>=4 (Nikhil's call: v3/v4's amoral sources are kept
for their edge; viability is SCORED and recorded for the record but NOT gated — the blend-run hypothesis is
that a 9B-active model can absorb the sharpness without the 3.4B's viability trade).

Judge = JUDGE_V5 verbatim from gen_v5.py (6-dim, Haiku via Anthropic-direct — no OpenRouter).
Leak guard: asserts no v3/v4 problem collides with the v5 held-out 20 or the 48 OOD benchmark problems.
Output: corpus_v34_regated/passers.jsonl with `corpus` provenance ("v3"/"v4") + `judge6` scores.

  python regate_v34.py            # ~402 Haiku calls, ~$1.5
"""
from __future__ import annotations
import asyncio, json
from pathlib import Path
import numpy as np, httpx
import gen_v3 as g
import gen_v5 as v5

HERE = Path(__file__).resolve().parent
SOURCES = {"v3": HERE / "corpus_v3_train" / "passers.jsonl", "v4": HERE / "corpus_v4_train" / "passers.jsonl"}
OUT = HERE / "corpus_v34_regated"; OUT.mkdir(exist_ok=True)
HELD = HERE.parent / "round2_kit" / "data_v5" / "eval_problems.jsonl"
BENCH = HERE / "benchmark" / "problems.jsonl"


def gate_sharp(j):
    return j["distinctness"] >= 4 and j["concreteness"] >= 4


async def main():
    rows = []
    for tag, path in SOURCES.items():
        for l in path.open():
            if l.strip():
                r = json.loads(l); r["corpus"] = tag; rows.append(r)
    # leak guard vs v5 held-out + OOD bench
    protected = {json.loads(l)["problem"].strip() for l in HELD.open()} | \
                {json.loads(l)["problem"].strip() for l in BENCH.open()}
    collisions = [r for r in rows if r["problem"].strip() in protected]
    assert not collisions, f"LEAK: {len(collisions)} v3/v4 problems collide with eval sets"
    print(f"re-gating {len(rows)} rows (v3 {sum(r['corpus']=='v3' for r in rows)}, "
          f"v4 {sum(r['corpus']=='v4' for r in rows)}) | leak check clean | judge=Haiku(direct) 6-dim", flush=True)

    async with httpx.AsyncClient() as client:
        async def judged(r):
            j = await v5.judge_v5(client, r["problem"], r["angles"], r["threads"])
            return (r, j)
        results = await asyncio.gather(*[judged(r) for r in rows])

    kept, dropped, nojudge = [], [], 0
    for r, j in results:
        if j is None:
            nojudge += 1; continue
        r["judge6"] = j
        (kept if gate_sharp(j) else dropped).append(r)

    with (OUT / "passers.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    def means(rs):
        return {d: round(float(np.mean([r["judge6"][d] for r in rs])), 2) for d in v5.DIMS} if rs else {}
    from collections import Counter
    summary = {
        "input": len(rows), "judge_failures": nojudge, "kept": len(kept), "dropped": len(dropped),
        "survival_pct": round(100 * len(kept) / max(len(rows) - nojudge, 1), 1),
        "kept_by_corpus": dict(Counter(r["corpus"] for r in kept)),
        "kept_by_source": dict(Counter(r["source"] for r in kept)),
        "kept_means": means(kept), "dropped_means": means(dropped),
        "gate": "distinct>=4 & concrete>=4 (sharpness only; viability recorded NOT gated)",
        "spend_usd": round(g.LED.spent, 2),
    }
    (OUT / "regate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
