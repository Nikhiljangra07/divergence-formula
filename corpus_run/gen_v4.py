"""
gen_v4.py — PRECISION corpus generator (the "next lemon").

v3 §9 proved foresight is a 3.4B CAPACITY WALL (trained foresight flat at base, model learns the FORM
"the bet is X" but fills X with fantasy). So v4 STOPS spending model capacity on foresight and focuses on
what DID transfer + has headroom: PRECISE dismantling — sharp, mechanically-specific moves, maximally
DISTINCT across the four families (trained distinctness was only 2.53 vs corpus 4.84 — big room).

Changes from v3 (everything else identical — same 4 sources, same DSV4 generator, same Haiku judge):
  - REFRACTOR_V4: bias toward a CLEAR underlying structure that splits crisply into 4 categorically
    different families (pattern-recognition), NOT a foresight dilemma.
  - WORKER_V4: TWO sentences, no foresight clause — a precise, concrete, decisive move + an explicit
    why-this-is-a-different-KIND-of-move-than-the-others beat. Capacity goes to precision + distinctness.
  - GATE: distinctness>=4 AND concreteness>=4 (the learnable, valuable dims). foresight is still SCORED
    by the same judge but kept only as a LOGGED DIAGNOSTIC — never gated (don't reject good precise
    examples for failing a skill we've abandoned).

Reuses gen_v3 plumbing (Ledger, call, parse_json, judge, geom). Modes:
  pilot:  python gen_v4.py --pilot --target 25                 # ungated, inspect judge means + gate yield
  scale:  python gen_v4.py --target 201 --weights "TH:51,PR:50,JPM:50,RI:50"   # gated, balanced, RESUMABLE
"""
from __future__ import annotations
import argparse, asyncio, json, os, time
from pathlib import Path
import numpy as np, httpx
import gen_v3 as g
import config as C

OUT_DEFAULT = "corpus_v4_train"

REFRACTOR_V4 = (
    "You are preparing a STRATEGIC decision dilemma for divergent analysis.\n\nSOURCE: {brief}\n\n"
    "DILEMMA TYPE for THIS example (instantiate exactly this, do not drift): {theme}\n\n"
    "Produce ONE concrete dilemma a figure in this world genuinely faces — a hard choice with a CLEAR "
    "underlying structure that GENUINELY admits FOUR categorically DIFFERENT strategic responses. The four "
    "must be real alternatives leading to DIFFERENT actions (not rephrasings of one move). If the problem "
    "has one obvious answer, redesign it until it truly forks four ways.\n\n"
    "Then three key FACETS, then FOUR strategic ANGLES.\n"
    "CRITICAL — each angle a categorically DIFFERENT KIND of move, a distinct family from:\n"
    "  CONFRONT/ELIMINATE · EVADE/DEFER · CO-OPT/INTEGRATE · TRANSFORM/REFRAME · DELEGATE/EXTERNALIZE · ENDURE/SACRIFICE\n"
    "All four families must differ; if two collapse to the same underlying move, replace one. Each angle "
    "opens a genuinely different line of attack on the WHOLE problem.\n\n"
    "Return STRICT JSON only:\n"
    '{{"problem":"<1-2 sentence dilemma>","facets":["<f1>","<f2>","<f3>"],'
    '"angles":[{{"family":"<FAMILY>","directive":"<angle>"}},'
    '{{"family":"<FAMILY>","directive":"<angle>"}},{{"family":"<FAMILY>","directive":"<angle>"}},'
    '{{"family":"<FAMILY>","directive":"<angle>"}}]}}'
)

WORKER_V4 = (
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; you are "
    "blind to the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE [{family}]: {angle}\n\n"
    "Write a single thread (TWO sentences, cold and analytical) that COMMITS to this angle as a concrete, "
    "mechanically SPECIFIC strategy for the WHOLE problem — name the actual move (who, what, how), precisely. "
    "It must be unmistakably a DIFFERENT KIND of move than the other families would choose — sharp and "
    "non-obvious, no vague gesture, no surface restatement of the angle.\n"
    "VOICE: cold, analytical, decisive. No hedging, no lists, no foresight padding. Output ONLY the thread."
)


async def one_thread_v4(client, problem, facets_str, ang):
    for _ in range(g.THREAD_REGENS + 1):
        t = await g.call(client, WORKER_V4.format(problem=problem, facets=facets_str, family=ang["family"],
                                                  angle=ang["directive"]), C.MODEL, g.WORKER_MAXTOK, 0.75)
        if t and t.strip():
            return t.strip()
    return None


async def gen_one_v4(client, src, theme):
    refr = await g.call(client, REFRACTOR_V4.format(brief=C.SOURCES[src], theme=theme), C.MODEL, g.REFRACTOR_MAXTOK, 0.95)
    spec = g.parse_json(refr)
    if not spec or not all(k in spec for k in ("problem", "facets", "angles")):
        return None
    angles = []
    for a in spec["angles"][:g.K]:
        if isinstance(a, dict) and a.get("directive"):
            angles.append({"family": str(a.get("family", "UNSPECIFIED")).strip(), "directive": str(a["directive"]).strip()})
        elif isinstance(a, str) and a.strip():
            angles.append({"family": "UNSPECIFIED", "directive": a.strip()})
    if len(angles) < g.K:
        return None
    facets_str = "; ".join(spec["facets"][:3])
    threads = await asyncio.gather(*[one_thread_v4(client, spec["problem"], facets_str, a) for a in angles])
    if any(t is None for t in threads):
        return None
    j = await g.judge(client, spec["problem"], angles, threads)
    if j is None:
        return None
    return {"source": src, "theme": theme, "problem": spec["problem"], "facets": spec["facets"][:3],
            "angles": angles, "threads": list(threads), "judge": j}


def gate_v4(j):
    """The precision gate: distinct AND concrete. foresight deliberately NOT gated (capacity wall)."""
    return j["distinctness"] >= 4 and j["concreteness"] >= 4


def parse_weights(s):
    return {kv.split(":")[0].strip(): int(kv.split(":")[1]) for kv in s.split(",")} if s else None


def load_counts(passers, quota):
    counts = {s: 0 for s in quota}
    if passers.exists():
        for line in passers.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            s = r.get("source")
            if s in counts and isinstance(r.get("judge"), dict) and gate_v4(r["judge"]):
                counts[s] += 1
    return counts


async def run_pilot(target, weights):
    """Ungated: generate ~target candidates, save all, report judge means + gate yield + best/worst."""
    out = g.HERE / "corpus_v4_pilot"; out.mkdir(exist_ok=True)
    raw = out / "candidates.jsonl"; raw.write_text("")
    w = weights or {s: 1 for s in C.SOURCES}
    N = int(target / 0.5 * 1.2)
    plan = g.weighted_plan(N, w) if weights else g.balanced_plan(N)
    print(f"v4 PRECISION pilot — target {target} | gate=distinct>=4 & concrete>=4 (logged, not enforced in pilot) | cap ${g.HARD_USD_CAP}", flush=True)
    cands, t0, lock = [], time.time(), asyncio.Lock()
    async with httpx.AsyncClient() as client:
        with raw.open("a") as f:
            async def bounded(s, th):
                async with g.CAND_SEM:
                    try:
                        c = await asyncio.wait_for(gen_one_v4(client, s, th), timeout=g.CAND_TIMEOUT)
                    except Exception:
                        c = None
                if c:
                    async with lock:
                        f.write(json.dumps(c) + "\n"); f.flush()
                    cands.append(c)
            await asyncio.gather(*[bounded(s, th) for s, th in plan])
    if not cands:
        print("PILOT EMPTY"); return
    dims = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight")
    means = {d: round(float(np.mean([c["judge"][d] for c in cands])), 2) for d in dims}
    yield_rate = round(100 * float(np.mean([gate_v4(c["judge"]) for c in cands])), 1)
    summary = {"n": len(cands), "spend_usd": round(g.LED.spent, 3), "judge_means": means,
               "gate_yield(distinct>=4 & concrete>=4)": yield_rate,
               "by_source_distinct": {s: round(float(np.mean([c["judge"]["distinctness"] for c in cands if c["source"] == s])), 2)
                                      for s in sorted({c["source"] for c in cands})}}
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== V4 PILOT SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


async def run_scale(target, weights, out_dir):
    """Gated (distinct>=4 & concrete>=4), quota-balanced, RESUMABLE (append; re-run to continue)."""
    out = g.HERE / out_dir; out.mkdir(exist_ok=True)
    passers = out / "passers.jsonl"; manifest = out / "manifest.json"
    quota = weights or {s: target // len(C.SOURCES) for s in C.SOURCES}
    counts = load_counts(passers, quota)
    inflight = {s: 0 for s in quota}; theme_idx = {s: counts[s] for s in quota}
    attempts = 0; t0 = time.time(); goal = sum(quota.values())
    MAX_ATTEMPTS = goal * 8
    print(f"v4 PRECISION scale — goal {goal} | resume {counts} | gate=distinct>=4 & concrete>=4 | cap ${g.HARD_USD_CAP}", flush=True)
    if sum(counts.values()) >= goal:
        print("already complete"); return
    lock = asyncio.Lock(); f = passers.open("a")
    async with httpx.AsyncClient() as client:
        async def worker():
            nonlocal attempts
            while True:
                async with lock:
                    if g.LED.over() or attempts >= MAX_ATTEMPTS:
                        return
                    src = next((s for s in quota if counts[s] + inflight[s] < quota[s]), None)
                    if src is None:
                        return
                    inflight[src] += 1; ti = theme_idx[src]; theme_idx[src] += 1; attempts += 1
                theme = C.THEMES[src][ti % len(C.THEMES[src])]
                try:
                    c = await asyncio.wait_for(gen_one_v4(client, src, theme), timeout=g.CAND_TIMEOUT)
                except Exception:
                    c = None
                async with lock:
                    inflight[src] -= 1
                    if c and gate_v4(c["judge"]) and counts[src] < quota[src]:
                        f.write(json.dumps(c) + "\n"); f.flush(); counts[src] += 1
                        tot = sum(counts.values())
                        if tot % 10 == 0 or tot == goal:
                            print(f"  {tot}/{goal} | {counts} | attempts {attempts} | ${g.LED.spent:.2f} | {time.time()-t0:.0f}s", flush=True)
        await asyncio.gather(*[worker() for _ in range(g.CAND_CONCURRENCY)])
    f.close()
    final = load_counts(passers, quota); done = sum(final.values())
    dims = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight")
    rows = [json.loads(l) for l in passers.open() if l.strip()]
    means = {d: round(float(np.mean([r["judge"][d] for r in rows])), 3) for d in dims} if rows else {}
    manifest.write_text(json.dumps({
        "goal": goal, "generated": done, "by_source": final, "gate": "distinct>=4 & concrete>=4",
        "attempts": attempts, "spend_usd": round(g.LED.spent, 3), "elapsed_s": round(time.time() - t0),
        "complete": done >= goal, "judge_means": means, "worker": "v4 precision (2-sentence, no foresight)",
    }, indent=2))
    status = "COMPLETE" if done >= goal else f"PARTIAL ({done}/{goal}) — re-run to resume"
    print(f"\n=== {status} === judge_means={means}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=201)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--weights", default="")
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    args = ap.parse_args()
    w = parse_weights(args.weights)
    if args.pilot:
        await run_pilot(args.target, w)
    else:
        await run_scale(args.target, w, args.out_dir)


if __name__ == "__main__":
    asyncio.run(main())
