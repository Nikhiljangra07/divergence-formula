"""
rejudge_all_gemini.py — SAME-SESSION Gemini re-judge of all shipped/baseline models on the 48-problem
bench, to remove the cross-session-judging caveat from the round-8 verdict. One judging session, one
judge (gemini-2.5-pro, temp 0), identical prompt: 3.4B-sft, hs_blend (r7 ship), hs_blend2 (r8 ship),
and Haiku 4.5 regenerated fresh (its generation temp is 0.0 in head2head_v5, so threads reproduce).
"""
from __future__ import annotations
import asyncio, json
from pathlib import Path
import numpy as np, httpx
import head2head_v5 as H

BENCH = Path(__file__).resolve().parent / "benchmark"
FILES = {
    "3.4B-sft": BENCH / "eval_bench_sft_v5_threads.jsonl",
    "hs_blend(r7)": BENCH / "eval_hs_blend_bench_v5_threads.jsonl",
    "hs_blend2(r8)": BENCH / "eval_hs_blend2_bench_v5_threads.jsonl",
}


def agg(rows):
    ok = [r for r in rows if r.get("judge")]
    a = {d: round(float(np.mean([r["judge"][d] for r in ok])), 2) for d in H.DIMS}
    a["overall"] = round(float(np.mean([r["judge"]["mean"] for r in ok])), 2)
    a["dist>=4%"] = round(100 * float(np.mean([r["judge"]["distinctness"] >= 4 for r in ok])), 1)
    a["n"] = len(ok)
    return a


async def main():
    probs = [json.loads(l)["problem"] for l in H.PROBLEMS.open()]
    out = {}
    async with httpx.AsyncClient() as client:
        # regenerate Haiku threads (temp 0.0 -> reproducible)
        haiku = {}
        async def gen_h(p):
            haiku[p] = await H.haiku_refract(client, p)
        await asyncio.gather(*[gen_h(p) for p in probs])
        print(f"haiku threads regenerated: {sum(1 for v in haiku.values() if v)}/48", flush=True)

        async def judge_set(label, rows):
            res = []
            async def one(r):
                j = await H.judge(client, r["problem"], r["angles"], r["threads"])
                res.append({"problem": r["problem"], "judge": j})
            await asyncio.gather(*[one(r) for r in rows])
            out[label] = agg(res)
            print(f"{label}: {json.dumps(out[label])}", flush=True)

        for label, f in FILES.items():
            await judge_set(label, [json.loads(l) for l in f.open()])
        await judge_set("haiku-4.5(fresh)", [{"problem": p, **haiku[p]} for p in probs if haiku[p]])

    (BENCH / "rejudge_all_summary.json").write_text(json.dumps(out, indent=2))
    print("\n=== SAME-SESSION RE-JUDGE (gemini-2.5-pro, one session) ===")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
