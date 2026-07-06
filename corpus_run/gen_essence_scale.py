"""
gen_essence_scale.py — ESSENCE ROUND scale run: DeepSeek V4 Pro generator (the A/B winner — tied
Sonnet 5 on quality at 1/6.6 the price, and BEAT it on distinctness 5.0 vs 4.67), judged/gated by
Gemini 2.5 Pro exactly like the pilot (same judge = no calibration drift between pilot and scale).

24 influence archetypes (12 pilot-validated + 12 scale additions), modern settings only, quota per
theme so coverage is even. Resumable (re-run continues from passers.jsonl). SFT positives only — no
DPO negatives (DPO was refuted in round-6; essence rows join the blend as SFT).

Gate = Gemini-calibrated: viability>=4 & distinctness>=4 & concreteness>=4 (pilot yield: 91.7%).

  python gen_essence_scale.py --target 360
"""
from __future__ import annotations
import argparse, asyncio, json, time
from collections import Counter
from pathlib import Path
import numpy as np
import httpx
import gen_essence_ab as AB
import config_essence as CE

HERE = Path(__file__).resolve().parent
# 12 candidate-workers against SEM(12) call slots: with judge calls at 30-90s the effective concurrent
# DSV4 load stays well under the ~16-candidate timeout cliff (config.py note) while doubling throughput.
WORKERS = 12


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=360)
    ap.add_argument("--out-dir", default="corpus_essence")
    args = ap.parse_args()
    out = HERE / args.out_dir; out.mkdir(exist_ok=True)
    passers = out / "passers.jsonl"; manifest = out / "manifest.json"

    pairs = [(s, th) for s in CE.SOURCES for th in CE.THEMES[s]]   # 24 (source, theme)
    per = max(1, args.target // len(pairs))                        # quota per theme
    quota = {th: per for _, th in pairs}
    src_of = {th: s for s, th in pairs}
    counts = {th: 0 for _, th in pairs}
    if passers.exists():
        for line in passers.open():
            try: r = json.loads(line)
            except Exception: continue
            if r.get("theme") in counts and isinstance(r.get("pos_judge"), dict) and AB.gate(r["pos_judge"]):
                counts[r["theme"]] += 1
    goal = sum(quota.values())
    inflight = {th: 0 for th in quota}
    attempts = 0; MAX_ATTEMPTS = goal * 3
    print(f"ESSENCE SCALE — goal {goal} ({len(pairs)} themes x {per}) | gen={AB.DSV4} | "
          f"judge={AB.GEMINI_ID} gate=viab&dist&conc>=4 | cap ${AB.HARD_USD_CAP} | "
          f"resume {sum(counts.values())}", flush=True)
    if sum(counts.values()) >= goal:
        print("already complete"); return

    t0 = time.time(); lock = asyncio.Lock(); f = passers.open("a")
    async with httpx.AsyncClient() as client:
        async def worker():
            nonlocal attempts
            while True:
                async with lock:
                    if AB.LED.over() or attempts >= MAX_ATTEMPTS: return
                    th = next((t for t in quota if counts[t] + inflight[t] < quota[t]), None)
                    if th is None: return
                    inflight[th] += 1; attempts += 1
                try:
                    # 600s: the remaining juicy themes run 120-180s per DSV4 call; at shared concurrency a
                    # full candidate (refractor + 4 workers + judge) legitimately exceeds the old 300s.
                    c = await asyncio.wait_for(AB.gen_one(client, AB.DSV4, src_of[th], th), timeout=600)
                except Exception as e:
                    c = {"fail": f"exc:{type(e).__name__}"}
                async with lock:
                    inflight[th] -= 1
                    if c and "pos_judge" in c and AB.gate(c["pos_judge"]) and counts[th] < quota[th]:
                        f.write(json.dumps(c) + "\n"); f.flush(); counts[th] += 1
                        tot = sum(counts.values())
                        print(f"  PASS {tot}/{goal} | attempts {attempts} | ${AB.LED.spent:.2f} | "
                              f"{time.time()-t0:.0f}s", flush=True)
                    else:
                        # every non-pass is logged with its stage — a stalled run must be VISIBLE
                        why = "gate" if (c and "pos_judge" in c) else (c or {}).get("fail", "none")
                        print(f"  fail[{why}] attempts={attempts} | ${AB.LED.spent:.2f} | "
                              f"{time.time()-t0:.0f}s", flush=True)
        await asyncio.gather(*[worker() for _ in range(WORKERS)])
    f.close()

    rows = [json.loads(l) for l in passers.open() if l.strip()]
    gp = [r for r in rows if isinstance(r.get("pos_judge"), dict) and AB.gate(r["pos_judge"])]
    means = {d: round(float(np.mean([r["pos_judge"][d] for r in gp])), 3) for d in AB.DIMS} if gp else {}
    manifest.write_text(json.dumps({
        "goal": goal, "generated": len(gp), "by_source": dict(Counter(r["source"] for r in gp)),
        "by_theme": dict(Counter(r["theme"][:60] for r in gp)),
        "generator": AB.DSV4, "judge": AB.GEMINI_ID, "gate": "gemini viab>=4 & dist>=4 & conc>=4",
        "attempts": attempts, "spend_usd": round(AB.LED.spent, 3),
        "spend_by_model": {m: round(v, 3) for m, v in AB.LED.by_model.items()},
        "elapsed_s": round(time.time() - t0), "complete": len(gp) >= goal,
        "positive_judge_means": means,
    }, indent=2))
    print(f"\n=== {'COMPLETE' if len(gp) >= goal else 'PARTIAL'} ({len(gp)}/{goal}) === "
          f"means={means} | ${AB.LED.spent:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
