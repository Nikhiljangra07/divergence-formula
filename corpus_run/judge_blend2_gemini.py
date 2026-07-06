"""
judge_blend2_gemini.py — Gemini 2.5 Pro verdict on the round-8 BLEND2 bench-48 threads.
Judge prompt VERBATIM from head2head_v5.py (== gen_v5 JUDGE_V5) so numbers are comparable with the
banked round-6/7 table (3.4B sft, hs_base, hs_v5, hs_blend, haiku — same judge, same 48 problems).
"""
from __future__ import annotations
import asyncio, json
from pathlib import Path
import numpy as np, httpx
import head2head_v5 as H

BENCH = Path(__file__).resolve().parent / "benchmark"
THREADS = BENCH / "eval_hs_blend2_bench_v5_threads.jsonl"


async def main():
    cat = {json.loads(l)["problem"].strip(): json.loads(l).get("category", "?") for l in H.PROBLEMS.open()}
    rows = [json.loads(l) for l in THREADS.open()]
    print(f"judging {len(rows)} blend2 bench threads with {H.GEMINI_ID}", flush=True)
    out = []
    async with httpx.AsyncClient() as client:
        async def one(r):
            j = await H.judge(client, r["problem"], r["angles"], r["threads"])
            out.append({"problem": r["problem"], "category": cat.get(r["problem"].strip(), "?"), "judge": j})
        await asyncio.gather(*[one(r) for r in rows])
    ok = [r for r in out if r["judge"]]
    agg = {d: round(float(np.mean([r["judge"][d] for r in ok])), 2) for d in H.DIMS}
    agg["overall"] = round(float(np.mean([r["judge"]["mean"] for r in ok])), 2)
    agg["dist>=4%"] = round(100 * float(np.mean([r["judge"]["distinctness"] >= 4 for r in ok])), 1)
    agg["viab>=4%"] = round(100 * float(np.mean([r["judge"]["viability"] >= 4 for r in ok])), 1)
    agg["n"] = len(ok)
    from collections import defaultdict
    bycat = defaultdict(list)
    for r in ok: bycat[r["category"]].append(r["judge"]["mean"])
    result = {"label": "hs_blend2 (Gemini bench-48)", "agg": agg,
              "by_category": {c: round(float(np.mean(v)), 2) for c, v in sorted(bycat.items())}}
    (BENCH / "blend2_gemini_summary.json").write_text(json.dumps(result, indent=2))
    (BENCH / "blend2_gemini_rows.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
