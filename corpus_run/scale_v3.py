"""
scale_v3.py — generate the v3 TRAINING corpus: 201 judge-gated, source-balanced foresight examples.

Matches round-2 size (201 problems) for a clean corpus-quality comparison. Single foresight arm
(no B/no-consequence control — foresight IS consequence-projection in v3).

Reuses the VALIDATED building blocks from gen_v3 (REFRACTOR_V3, WORKER_V3 [reverted to original],
gen_one, judge, Ledger). This file only adds the scale concerns the pilot lacked:
  - RESUMABLE: appends to passers.jsonl, never truncates. Re-run after any crash/lid-close and it
    continues from the per-source counts already on disk (the pilot wiped its file on start — the exact
    failure mode that killed the round-2 B-gen at 34/186).
  - JUDGE-GATED: only min(judge) >= GATE_MIN enters the corpus (drops foresight-collapse examples).
  - QUOTA-BALANCED: fills each source to its quota (TH 51, others 50 = 201), no overshoot.
  - DOUBLE BACKSTOP: money cap + global attempt cap, both logged if hit.

Run:  caffeinate -i python3 scale_v3.py > scale_v3.log 2>&1 &
"""
from __future__ import annotations
import asyncio, json, os, time
from pathlib import Path
import httpx
import gen_v3 as g
import config as C

OUT = g.HERE / "corpus_v3_train"; OUT.mkdir(exist_ok=True)
PASSERS = OUT / "passers.jsonl"
MANIFEST = OUT / "manifest.json"

QUOTA = {"TH": 51, "PR": 50, "JPM": 50, "RI": 50}   # 201 total, balanced across the 4 v3 sources
GATE_MIN = 3                                          # keep only judge-min >= 3 (drops foresight-collapse / any weak dim)
CAP = float(os.environ.get("V3_SCALE_CAP", "8.0"))   # money backstop; expected ~$4-5
MAX_ATTEMPTS = 1200                                   # attempt backstop (expected ~450); guards a pathological gate-reject loop

g.LED = g.Ledger(CAP)                                 # fresh ledger at the scale cap (pilot ledger is replaced)


def load_existing():
    """Per-source counts already on disk — the basis for resumability. Skips malformed lines."""
    counts = {s: 0 for s in QUOTA}
    if PASSERS.exists():
        for line in PASSERS.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = r.get("source")
            if s in counts and isinstance(r.get("judge"), dict) and r["judge"].get("min", 0) >= GATE_MIN:
                counts[s] += 1
    return counts


async def main():
    counts = load_existing()
    inflight = {s: 0 for s in QUOTA}
    theme_idx = {s: counts[s] for s in QUOTA}   # continue theme rotation from where we left off
    attempts = 0
    t0 = time.time()
    total_goal = sum(QUOTA.values())
    print(f"scale_v3 — goal {total_goal} | resume from {counts} ({sum(counts.values())} on disk) | "
          f"gate min>={GATE_MIN} | cap ${CAP} | judge={g.JUDGE_MODEL}", flush=True)
    if sum(counts.values()) >= total_goal:
        print("already complete — nothing to do", flush=True)
        return

    lock = asyncio.Lock()
    f = PASSERS.open("a")  # APPEND — resumable; never truncate

    async with httpx.AsyncClient() as client:
        async def worker():
            nonlocal attempts
            while True:
                async with lock:
                    if g.LED.over():
                        return
                    if attempts >= MAX_ATTEMPTS:
                        return
                    src = next((s for s in QUOTA if counts[s] + inflight[s] < QUOTA[s]), None)
                    if src is None:
                        return
                    inflight[src] += 1
                    ti = theme_idx[src]; theme_idx[src] += 1
                    attempts += 1
                theme = C.THEMES[src][ti % len(C.THEMES[src])]
                try:
                    c = await asyncio.wait_for(g.gen_one(client, src, theme), timeout=g.CAND_TIMEOUT)
                except Exception:
                    c = None
                async with lock:
                    inflight[src] -= 1
                    if c and c["judge"]["min"] >= GATE_MIN and counts[src] < QUOTA[src]:
                        f.write(json.dumps(c) + "\n"); f.flush()
                        counts[src] += 1
                        tot = sum(counts.values())
                        if tot % 10 == 0 or tot == total_goal:
                            print(f"  {tot}/{total_goal} | {counts} | attempts {attempts} | "
                                  f"${g.LED.spent:.2f} | {time.time()-t0:.0f}s", flush=True)

        await asyncio.gather(*[worker() for _ in range(g.CAND_CONCURRENCY)])

    f.close()
    final = load_existing()
    done = sum(final.values())
    manifest = {
        "goal": total_goal, "generated": done, "by_source": final,
        "gate": f"judge.min>={GATE_MIN}", "attempts": attempts,
        "spend_usd": round(g.LED.spent, 3), "elapsed_s": round(time.time() - t0),
        "complete": done >= total_goal,
        "worker_prompt": "original (reverted)", "sources": list(QUOTA), "judge": g.JUDGE_MODEL,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    status = "COMPLETE" if done >= total_goal else f"PARTIAL ({done}/{total_goal}) — re-run to resume"
    print(f"\n=== {status} ===\n{json.dumps(manifest, indent=2)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
