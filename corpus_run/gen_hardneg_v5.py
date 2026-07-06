"""
gen_hardneg_v5.py — HARD-negative DPO pairs targeting VIABILITY (the SFT weak spot).

The first DPO used EASY negatives (illegal/fantasy/baroque) -> ~2.1pt pos-neg judge gap -> near-zero useful
gradient -> regressed. This regenerates negatives that are JUST AS distinct/concrete/decisive as the positive
but SUBTLY LESS VIABLE (assume cooperation/resources/timing a real actor can't count on; ignore a hard
constraint) -> DPO's gradient targets viability specifically, holding the other dims ~equal.

Reuses the 480 v5 positives (corpus_v5_train/passers.jsonl); ONLY the negatives are new.
Chosen = existing pos_thread, Rejected = new hard-neg. Prompt = prep_v5 worker contract (train==eval==dpo).
Generator = DeepSeek V4 Pro (via OpenRouter — no direct key exists). Pilot judge = Haiku (Anthropic-direct).

  pilot:  python gen_hardneg_v5.py --pilot 12   # verify: viability gap LARGE, distinctness gap SMALL
  scale:  python gen_hardneg_v5.py               # -> ../round2_kit/data_v5/dpo_pairs_hard.jsonl
"""
from __future__ import annotations
import argparse, asyncio, difflib, json
from pathlib import Path
import numpy as np, httpx
import gen_v3 as g
import config_v5 as C5
import gen_v5 as v5  # reuse judge_v5 for pilot verification

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus_v5_train" / "passers.jsonl"
OUT = HERE.parent / "round2_kit" / "data_v5" / "dpo_pairs_hard.jsonl"
g.SEM = asyncio.Semaphore(24)  # single-call gen -> 24 in-flight is safe (throttle appeared at 144-in-flight full attempts)

# worker contract — BYTE-IDENTICAL to prep_v5.py / dav_eval_v5.py (train == eval == dpo prompt)
WRK_SYS = "You write one precise, decisive, realistic reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two or "
            "three sentences, cold and analytical) that COMMITS to THIS angle as a concrete, realistic, VIABLE "
            "strategy that resolves the whole problem in a distinct way — name the actual first move (who does "
            "what, to whom, by when) and the one most likely downstream consequence it is betting on. It must be "
            "lawful, executable, and unmistakably a different KIND of move than the other families would choose.")

# REWRITE the positive with ONE SUBSTANTIVE viability over-reach -> keeps the move family/distinctness,
# degrades real-world executability. NOT "minimal edit" (that produced identical/trivial negs); the plan must
# GENUINELY DEPEND on the false assumption.
HARDNEG = (
    "PROBLEM: {problem}\n\nHere is a STRONG, viable reasoning thread for this decision:\n\"{pos}\"\n\n"
    "Rewrite it into a HARD NEGATIVE for preference training. KEEP the same strategic FAMILY and general "
    "direction of the move (so it stays the same distinct KIND of move), and keep it sharp, specific, and "
    "decisive. But make it CLEARLY LESS VIABLE by injecting ONE concrete OVER-REACH — pick whichever fits: "
    "commit to an unrealistically fast timeline; assume a budget, headcount, or resource that isn't established; "
    "assume a rival, partner, regulator, or team will cooperate when they have no reason to; or quietly ignore a "
    "hard constraint the situation imposes. The over-reach must be SUBSTANTIVE — change the specifics as needed "
    "so the plan GENUINELY DEPENDS on the false assumption, not just a reworded sentence. It must still read as "
    "confident and concrete (a casual reader wouldn't flag it) and must NOT be illegal, fantastical, or baroque — "
    "the SOLE defect is that it would realistically fail or be far harder to execute than the original. Two or "
    "three sentences. Output ONLY the rewritten thread."
)


async def one_neg(client, problem, pos_thread):
    # Reject rewrites that come back identical/trivial (the prior batch's failure) -> force a real perturbation.
    pos = pos_thread.strip()
    for _ in range(g.THREAD_REGENS + 3):
        t = await g.call(client, HARDNEG.format(problem=problem, pos=pos_thread), C5.MODEL, g.WORKER_MAXTOK, 0.85)
        if t and t.strip():
            t = t.strip()
            if t != pos and (1 - difflib.SequenceMatcher(None, pos, t).ratio()) >= 0.08:
                return t
    return None  # give up -> pair dropped (better a smaller clean set than trivial pairs)


def dpo_row(problem, facets_str, ang, chosen, rejected):
    u = WRK_USER.format(problem=problem, facets=facets_str, angle=(f"[{ang['family']}] {ang['directive']}"))
    return {"prompt": [{"role": "system", "content": WRK_SYS}, {"role": "user", "content": u}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}]}


async def gen_for_row(client, r):
    angles, pos = r["angles"], r["pos_threads"]
    facets_str = "; ".join(r["facets"][:3])
    negs = await asyncio.gather(*[one_neg(client, r["problem"], p) for p in pos])
    if any(n is None for n in negs):
        return None
    return {"row": r, "facets_str": facets_str, "negs": [n.strip() for n in negs]}


async def run_pilot(n):
    rows = [json.loads(l) for l in CORPUS.open() if l.strip()][:n]
    print(f"HARD-NEG pilot — {len(rows)} problems | verify viability gap LARGE, distinctness gap SMALL", flush=True)
    async with httpx.AsyncClient() as client:
        gens = await asyncio.gather(*[gen_for_row(client, r) for r in rows])
        gens = [x for x in gens if x]
        # MAGNITUDE check (the thing the last pilot missed): how perturbed are the accepted negs?
        cd = [1 - difflib.SequenceMatcher(None, p.strip(), n.strip()).ratio()
              for x in gens for p, n in zip(x["row"]["pos_threads"], x["negs"])]
        drop = round(100 * (len(rows) - len(gens)) / max(len(rows), 1), 1)
        print(f"rows kept {len(gens)}/{len(rows)} (drop {drop}%) | char-diff of negs: "
              f"median {np.median(cd)*100:.0f}% p25 {np.percentile(cd,25)*100:.0f}% p75 {np.percentile(cd,75)*100:.0f}% "
              f"(want median >=12%)", flush=True)
        # judge pos-set and neg-set per problem (Haiku, 6-dim)
        async def judged(x):
            pj = await v5.judge_v5(client, x["row"]["problem"], x["row"]["angles"], x["row"]["pos_threads"])
            nj = await v5.judge_v5(client, x["row"]["problem"], x["row"]["angles"], x["negs"])
            return pj, nj
        pairs = await asyncio.gather(*[judged(x) for x in gens])
    pairs = [(p, n) for p, n in pairs if p and n]
    def mean(side, dim): return round(float(np.mean([s[dim] for s in side])), 2)
    pos = [p for p, n in pairs]; neg = [n for p, n in pairs]
    print(f"\n=== HARD-NEG PILOT ({len(pairs)} judged) — Haiku 6-dim ===")
    print(f"{'dim':14} {'POS':>5} {'NEG':>5} {'gap':>6}")
    for d in v5.DIMS:
        pm, nm = mean(pos, d), mean(neg, d)
        print(f"{d:14} {pm:>5} {nm:>5} {pm-nm:>+6.2f}")
    vg = mean(pos, "viability") - mean(neg, "viability")
    dg = mean(pos, "distinctness") - mean(neg, "distinctness")
    print(f"\nVERDICT: viability gap {vg:+.2f} (want LARGE, >=0.6), distinctness gap {dg:+.2f} (want SMALL, <=0.4)")
    print("  -> GOOD hard negatives = big viability gap + small distinctness gap (contrastive on viability only)")
    print(f"spend ${g.LED.spent:.2f}", flush=True)


async def run_scale():
    rows = [json.loads(l) for l in CORPUS.open() if l.strip()]
    # LEAK GUARD: exclude the 20 held-out eval problems (same ones dav_eval_v5/head2head judge on).
    evalf = OUT.parent / "eval_problems.jsonl"
    held = {json.loads(l)["problem"].strip() for l in evalf.open()} if evalf.exists() else set()
    before = len(rows); rows = [r for r in rows if r["problem"].strip() not in held]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"HARD-NEG scale — {len(rows)} train problems ({before-len(rows)} held-out excluded) x 4 = "
          f"{len(rows)*4} pairs | generator={C5.MODEL}", flush=True)
    written = 0
    async with httpx.AsyncClient() as client:
        with OUT.open("w") as f:
            # process in chunks so we flush progress and don't hold everything in memory
            CH = 40
            for i in range(0, len(rows), CH):
                chunk = rows[i:i+CH]
                gens = await asyncio.gather(*[gen_for_row(client, r) for r in chunk])
                for x in gens:
                    if not x: continue
                    for ang, ch, ng in zip(x["row"]["angles"], x["row"]["pos_threads"], x["negs"]):
                        f.write(json.dumps(dpo_row(x["row"]["problem"], x["facets_str"], ang, ch.strip(), ng)) + "\n")
                        written += 1
                    f.flush()
                print(f"  {min(i+CH,len(rows))}/{len(rows)} problems | {written} pairs | ${g.LED.spent:.2f}", flush=True)
    print(f"\n=== DONE: {written} hard-neg DPO pairs -> {OUT} | spend ${g.LED.spent:.2f} ===", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    args = ap.parse_args()
    if args.pilot:
        await run_pilot(args.pilot)
    else:
        await run_scale()


if __name__ == "__main__":
    asyncio.run(main())
