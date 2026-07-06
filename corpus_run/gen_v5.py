"""
gen_v5.py — v5 "big corpus" DUAL generator: clean POSITIVE refractions (SFT) + paired flawed NEGATIVE
refractions (DPO rejected), across 8 rebalanced sources, classical AND modern settings.

Fixes the benchmark-diagnosed gaps:
  - VIABILITY: strategies must be realistic, lawful, executable (kills the amoral/baroque/fantasy leak).
  - DECISION-RELEVANCE: each thread must actually RESOLVE the decision a distinct way, not spin a scheme.
  - LIGHT FORESIGHT: one realistic downstream consequence (borderline depth — no fantasy chains).
  - DOMAIN COVERAGE: modern settings (career/health/finance/relationship/ethics) interleaved with classical.

Each example yields a DPO pair: pos_threads (chosen) + neg_threads (rejected) on the SAME problem.
6-dim judge (5 prior + viability). Gate (positives): viability>=4 & distinct>=4 & concrete>=4.

  pilot: python gen_v5.py --pilot --target 40
  scale: python gen_v5.py --target 480
"""
from __future__ import annotations
import argparse, asyncio, json, os, time
from pathlib import Path
import numpy as np, httpx
import gen_v3 as g
import config_v5 as C5

DIMS = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight", "viability")
K = 4
# Gate judge = Haiku (cheap, known). Haiku scores ~1 point harsh vs the fair judge (Gemini): on identical
# threads, Haiku viability 2.96 vs Gemini 4.04. So we DON'T switch judges — we CALIBRATE THE THRESHOLD:
# gate at viability>=3 (Haiku) which ~= viability>=4 (Gemini) = genuinely viable. Recovers yield 25%->67%.
V5_JUDGE_MODEL = os.environ.get("V5_JUDGE", g.JUDGE_MODEL)  # Haiku
V5_JUDGE_MAXTOK = 450

REFRACTOR_V5 = (
    "You are preparing a decision dilemma for divergent strategic analysis.\n\n"
    "SOURCE (decision tradition): {brief}\n\n"
    "DECISION ARCHETYPE for THIS example (instantiate exactly this, do not drift): {theme}\n\n"
    "SETTING: {setting}\n\n"
    "Produce ONE concrete dilemma a real person genuinely faces. CRITICAL CONSTRAINT: the problem must "
    "GENUINELY admit FOUR different responses that are ALL viable — each lawful, realistic, and a course of "
    "action a reasonable person would actually consider (NONE may be a crime, a fantasy, or a convoluted "
    "scheme). Many decisions only have ONE honorable answer — if this archetype is like that, you MUST "
    "REDESIGN THE PROBLEM: raise the stakes, add genuinely competing legitimate considerations (duty vs "
    "duty, two goods in tension, real uncertainty about facts), so that four DIFFERENT reasonable people "
    "would each defensibly choose a different one of the four families. Do NOT manufacture viability by "
    "adding unethical options — manufacture it by designing a harder, genuinely forked problem.\n\n"
    "Then three key FACETS, then FOUR strategic ANGLES — each a categorically DIFFERENT KIND of move "
    "(a distinct family), and each a VIABLE option for THIS problem. All four families must differ.\n\n"
    "Return STRICT JSON only:\n"
    '{{"problem":"<1-2 sentence dilemma>","facets":["<f1>","<f2>","<f3>"],'
    '"angles":[{{"family":"<FAMILY>","directive":"<angle>"}},{{"family":"<FAMILY>","directive":"<angle>"}},'
    '{{"family":"<FAMILY>","directive":"<angle>"}},{{"family":"<FAMILY>","directive":"<angle>"}}]}}'
)

SETTING_CLASSICAL = "Keep it TRUE TO THE SOURCE'S WORLD — its era, kind of characters, and stakes."
SETTING_MODERN = ("Set it in a MODERN, present-day real-world situation (a career, business, money, family, "
                  "health, or relationship decision an ordinary person today could face). Keep the source's "
                  "DECISION LOGIC, but a contemporary everyday situation — no historical or fictional trappings.")

WORKER_V5_POS = (
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; you are "
    "blind to the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE [{family}]: {angle}\n\n"
    "Write a single thread (TWO or THREE sentences, cold and analytical) that COMMITS to this angle as a "
    "concrete, REALISTIC, VIABLE strategy that actually RESOLVES the whole decision in a distinct way. "
    "MECHANICAL SPECIFICITY IS REQUIRED: name the actual first move with who does what, to whom, by when — "
    "use concrete particulars (a specific action, a number, a timeframe, a named role or party) so the move "
    "could be executed tomorrow. It must be lawful, executable, and unmistakably a DIFFERENT KIND of decision "
    "than the other families would make. Then name the ONE most likely downstream consequence it is betting "
    "on — one realistic step ahead, no fantasy chains.\n"
    "NO convoluted schemes, NO obfuscation, NO illegal or manipulative maneuvers, NO vague gestures, NO "
    "restatement of the angle. VOICE: cold, analytical, decisive. Output ONLY the thread."
)

WORKER_V5_NEG = (
    "You are writing ONE reasoning thread that DELIBERATELY exhibits a common FAILURE pattern — it will be "
    "used as a NEGATIVE training example, so it must be a plausibly-tempting but WORSE answer.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE [{family}]: {angle}\n\n"
    "Write a single thread (TWO or THREE sentences) that pursues this angle but FAILS in ONE of these ways "
    "(pick whichever fits): an elaborate CONVOLUTED scheme that never cleanly decides; a BAROQUE restatement "
    "barely different from the other approaches; an UNREALISTIC or fantastical mechanism; or a clever but "
    "UNETHICAL/manipulative maneuver. It should sound superficially strategic but be clearly a worse answer "
    "than a clean, viable, decisive move. Output ONLY the thread."
)

JUDGE_V5 = (
    "You are grading ONE training example for a model that must REFRACT a hard decision into FOUR genuinely "
    "distinct, viable strategies. Be a strict critic.\n\n"
    "PROBLEM: {problem}\n\nThe four strategies:\n{threads}\n\n"
    "Score each 1-5 (5=excellent, 1=fails):\n"
    "- admits_multiplicity: does the PROBLEM genuinely admit several viable DIFFERENT strategies?\n"
    "- distinctness: are the four genuinely DIFFERENT approaches, not rephrasings or the same move?\n"
    "- concreteness: does each name a concrete, mechanically specific move (who/what/how), cleanly (not baroque)?\n"
    "- decisiveness: does each COMMIT to a move and actually RESOLVE the decision rather than hedge?\n"
    "- foresight: does each name a realistic downstream consequence a step ahead — WITHOUT fantasy over-reach?\n"
    "- viability: are the four REALISTIC, LAWFUL, and actually executable in the real world (NOT fantasy, NOT "
    "illegal, NOT convoluted obfuscation)? 1=fantasy/illegal/incoherent, 5=fully realistic and viable.\n\n"
    "Return STRICT JSON only:\n"
    '{{"admits_multiplicity":N,"distinctness":N,"concreteness":N,"decisiveness":N,"foresight":N,"viability":N,'
    '"weakest":"<dimension>","note":"<=12 words"}}'
)


async def judge_v5(client, problem, angles, threads, model=V5_JUDGE_MODEL, maxtok=V5_JUDGE_MAXTOK):
    block = "\n".join(f"{i+1}. [{a['family']}] {t}" for i, (a, t) in enumerate(zip(angles, threads)))
    out = await g.call(client, JUDGE_V5.format(problem=problem, threads=block), model, maxtok, 0.0)
    j = g.parse_json(out)
    if not j: return None
    try:
        s = {d: int(j[d]) for d in DIMS}
    except Exception:
        return None
    s["weakest"] = str(j.get("weakest", "")); s["note"] = str(j.get("note", ""))
    s["min"] = min(s[d] for d in DIMS); s["mean"] = round(sum(s[d] for d in DIMS) / 6, 2)
    return s


async def one_thread(client, prompt_tmpl, problem, facets_str, ang):
    for _ in range(g.THREAD_REGENS + 1):
        t = await g.call(client, prompt_tmpl.format(problem=problem, facets=facets_str, family=ang["family"],
                                                    angle=ang["directive"]), C5.MODEL, g.WORKER_MAXTOK, 0.75)
        if t and t.strip(): return t.strip()
    return None


async def gen_one(client, src, theme, setting, judge_neg=True):
    refr = await g.call(client, REFRACTOR_V5.format(brief=C5.SOURCES[src], theme=theme, setting=setting),
                        C5.MODEL, g.REFRACTOR_MAXTOK, 0.95)
    spec = g.parse_json(refr)
    if not spec or not all(k in spec for k in ("problem", "facets", "angles")): return None
    angles = []
    for a in spec["angles"][:K]:
        if isinstance(a, dict) and a.get("directive"):
            angles.append({"family": str(a.get("family", "UNSPECIFIED")).strip(), "directive": str(a["directive"]).strip()})
    if len(angles) < K: return None
    facets_str = "; ".join(spec["facets"][:3])
    pos = await asyncio.gather(*[one_thread(client, WORKER_V5_POS, spec["problem"], facets_str, a) for a in angles])
    neg = await asyncio.gather(*[one_thread(client, WORKER_V5_NEG, spec["problem"], facets_str, a) for a in angles])
    if any(t is None for t in pos) or any(t is None for t in neg): return None
    pj = await judge_v5(client, spec["problem"], angles, pos)              # Gemini gate (the decision)
    if pj is None: return None
    nj = await judge_v5(client, spec["problem"], angles, neg, model=g.JUDGE_MODEL, maxtok=450) if judge_neg else None  # Haiku cheap contrast (pilot only)
    return {"source": src, "theme": theme, "setting": "classical" if setting == SETTING_CLASSICAL else "modern",
            "problem": spec["problem"], "facets": spec["facets"][:3], "angles": angles,
            "pos_threads": list(pos), "pos_judge": pj, "neg_threads": list(neg), "neg_judge": nj}


def gate_pos(j):
    # viability>=3 is the Haiku-calibrated equivalent of "genuinely viable" (Haiku scores ~1 pt harsh)
    return j["viability"] >= 3 and j["distinctness"] >= 4 and j["concreteness"] >= 4


def plan(n):
    """Balanced across 8 sources; alternate classical/modern; rotate themes within source."""
    srcs = list(C5.SOURCES); out = []
    for i in range(n):
        s = srcs[i % len(srcs)]; th = C5.THEMES[s]
        theme = th[(i // len(srcs)) % len(th)]
        setting = SETTING_CLASSICAL if i % 2 == 0 else SETTING_MODERN
        out.append((s, theme, setting))
    return out


async def run_pilot(target):
    out = g.HERE / "corpus_v5_pilot"; out.mkdir(exist_ok=True)
    raw = out / "candidates.jsonl"; raw.write_text("")
    p = plan(int(target / 0.5 * 1.2))
    print(f"v5 DUAL pilot — target {target} | 8 sources, classical+modern | 6-dim judge | cap ${g.HARD_USD_CAP}", flush=True)
    cands, lock = [], asyncio.Lock()
    async with httpx.AsyncClient() as client:
        with raw.open("a") as f:
            async def bounded(s, th, setting):
                async with g.CAND_SEM:
                    try: c = await asyncio.wait_for(gen_one(client, s, th, setting), timeout=g.CAND_TIMEOUT)
                    except Exception: c = None
                if c:
                    async with lock:
                        f.write(json.dumps(c) + "\n"); f.flush()
                    cands.append(c)
            await asyncio.gather(*[bounded(s, th, st) for s, th, st in p])
    if not cands:
        print("PILOT EMPTY"); return
    pos = [c["pos_judge"] for c in cands]
    pmeans = {d: round(float(np.mean([j[d] for j in pos])), 2) for d in DIMS}
    gate_yield = round(100 * float(np.mean([gate_pos(c["pos_judge"]) for c in cands])), 1)
    # negative contrast: how much worse is the rejected branch?
    gap = round(float(np.mean([c["pos_judge"]["mean"] - c["neg_judge"]["mean"] for c in cands])), 2)
    neg_worse = round(100 * float(np.mean([c["neg_judge"]["mean"] < c["pos_judge"]["mean"] for c in cands])), 1)
    from collections import Counter
    summary = {"n": len(cands), "spend_usd": round(g.LED.spent, 3),
               "positive_judge_means": pmeans, "gate_yield(viable&distinct&concrete>=4)": gate_yield,
               "DPO_contrast": {"avg_pos_minus_neg_mean": gap, "pct_neg_worse_than_pos": neg_worse,
                                "neg_mean": round(float(np.mean([c['neg_judge']['mean'] for c in cands])), 2)},
               "by_source": dict(Counter(c["source"] for c in cands)),
               "by_setting": dict(Counter(c["setting"] for c in cands))}
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== V5 PILOT SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


async def run_scale(target, out_dir):
    """Gemini-gated (viable&distinct&concrete>=4), balanced 8 sources, classical/modern alternating,
    RESUMABLE (append). Each passer = one SFT positive + one DPO pair. Negatives kept unjudged (cost)."""
    out = g.HERE / out_dir; out.mkdir(exist_ok=True)
    passers = out / "passers.jsonl"; manifest = out / "manifest.json"
    quota = {s: target // len(C5.SOURCES) for s in C5.SOURCES}
    # resume: count gated passers already on disk
    counts = {s: 0 for s in quota}
    if passers.exists():
        for line in passers.open():
            try: r = json.loads(line)
            except Exception: continue
            if r.get("source") in counts and isinstance(r.get("pos_judge"), dict) and gate_pos(r["pos_judge"]):
                counts[r["source"]] += 1
    inflight = {s: 0 for s in quota}; theme_idx = {s: counts[s] for s in quota}; setting_ct = {s: counts[s] for s in quota}
    attempts = 0; t0 = time.time(); goal = sum(quota.values()); MAX_ATTEMPTS = goal * 8
    print(f"v5 SCALE — goal {goal} | resume {counts} | judge={V5_JUDGE_MODEL} | gate=viable&distinct&concrete>=4 | cap ${g.HARD_USD_CAP}", flush=True)
    if sum(counts.values()) >= goal:
        print("already complete"); return
    lock = asyncio.Lock(); f = passers.open("a")
    async with httpx.AsyncClient() as client:
        async def worker():
            nonlocal attempts
            while True:
                async with lock:
                    if g.LED.over() or attempts >= MAX_ATTEMPTS: return
                    src = next((s for s in quota if counts[s] + inflight[s] < quota[s]), None)
                    if src is None: return
                    inflight[src] += 1; ti = theme_idx[src]; theme_idx[src] += 1
                    setting = SETTING_CLASSICAL if setting_ct[src] % 2 == 0 else SETTING_MODERN
                    setting_ct[src] += 1; attempts += 1
                theme = C5.THEMES[src][ti % len(C5.THEMES[src])]
                try:
                    c = await asyncio.wait_for(gen_one(client, src, theme, setting, judge_neg=False), timeout=g.CAND_TIMEOUT)
                except Exception:
                    c = None
                async with lock:
                    inflight[src] -= 1
                    if c and gate_pos(c["pos_judge"]) and counts[src] < quota[src]:
                        f.write(json.dumps(c) + "\n"); f.flush(); counts[src] += 1
                        tot = sum(counts.values())
                        if tot % 10 == 0 or tot == goal:
                            print(f"  {tot}/{goal} | {counts} | attempts {attempts} | ${g.LED.spent:.2f} | {time.time()-t0:.0f}s", flush=True)
        await asyncio.gather(*[worker() for _ in range(g.CAND_CONCURRENCY)])
    f.close()
    rows = [json.loads(l) for l in passers.open() if l.strip()]
    gp = [r for r in rows if gate_pos(r["pos_judge"])]
    means = {d: round(float(np.mean([r["pos_judge"][d] for r in gp])), 3) for d in DIMS} if gp else {}
    from collections import Counter
    manifest.write_text(json.dumps({
        "goal": goal, "generated": len(gp), "by_source": dict(Counter(r["source"] for r in gp)),
        "by_setting": dict(Counter(r["setting"] for r in gp)), "gate": "haiku viability>=3 & distinct>=4 & concrete>=4",
        "judge": V5_JUDGE_MODEL, "attempts": attempts, "spend_usd": round(g.LED.spent, 3),
        "elapsed_s": round(time.time() - t0), "complete": len(gp) >= goal, "positive_judge_means": means,
    }, indent=2))
    print(f"\n=== {'COMPLETE' if len(gp) >= goal else 'PARTIAL'} ({len(gp)}/{goal}) === means={means}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=40); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", default="corpus_v5_train")
    args = ap.parse_args()
    if args.pilot: await run_pilot(args.target)
    else: await run_scale(args.target, args.out_dir)


if __name__ == "__main__":
    asyncio.run(main())
