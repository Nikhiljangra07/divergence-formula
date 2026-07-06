"""
gen_v3.py — FORESIGHT corpus + LLM-JUDGE, pilot-first.

Root-cause fix (round-2 lesson): the decomposer is only as good as the GENERATOR. So v3 steers the
generator, not the decomposer, toward problems that GENUINELY require multiple forward-looking strategies.

Design:
  REFRACTOR_V3 → a FORESIGHT dilemma (right choice depends on anticipating how things unfold) that
                 genuinely admits FOUR different forward strategies, each a distinct move-FAMILY.
  WORKER_V3    → one thread per angle: a concrete, decisive move + the ONE downstream reaction it bets on
                 (1-2 steps of foresight, ~60-80 words). BORDERLINE depth — anticipate, don't over-plan
                 (deep multi-step plans trip the 3.4B ceiling -> hallucination).
  JUDGE        → Haiku (different family from the DSV4 generator, no rubber-stamping) scores each example
                 on admits_multiplicity / distinctness / concreteness / decisiveness / foresight (foresight
                 penalizes BOTH no-anticipation AND fantasy over-reach). THIS is the headline metric;
                 pairdist failed at model level (round-2 §8f), so geometry is kept only as a diagnostic.

Pilot: target 50, score every example (judge + full_pd + distinct_families), report distributions so we
calibrate a gate before scaling. Save-as-you-go, money cap, bounded concurrency. Run under caffeinate.
"""
from __future__ import annotations
import argparse, asyncio, json, os, time
from pathlib import Path
import numpy as np, httpx
from dotenv import load_dotenv
import config as C, dav

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_v3"; OUT.mkdir(exist_ok=True)
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]; OAI_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
# Route anthropic/* models to Anthropic's NATIVE API (your own key) instead of OpenRouter — no OR
# credit-fee, billed to Anthropic credits. Default on when the key exists; DIRECT_ANTHROPIC=0 forces OR.
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_DIRECT = bool(ANTHROPIC_KEY) and os.environ.get("DIRECT_ANTHROPIC", "1") == "1"
ANTHROPIC_ID = {"anthropic/claude-haiku-4.5": "claude-haiku-4-5"}  # OR model-string -> Anthropic native id
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-haiku-4.5")
JUDGE_IN, JUDGE_OUT = 1.0 / 1e6, 5.0 / 1e6
HARD_USD_CAP = float(os.environ.get("V3_USD_CAP", "4.0"))
CONCURRENCY = 12
CAND_CONCURRENCY = int(os.environ.get("CAND_CONCURRENCY", "16"))  # env-overridable: 16 throttles DSV4 Pro into the timeout cliff; 8 is safe
TIMEOUT, RETRIES, THREAD_REGENS = 60.0, 4, 2
CAND_TIMEOUT = float(os.environ.get("CAND_TIMEOUT", "240"))  # env-overridable: raise so throttled attempts complete instead of being discarded
REFRACTOR_MAXTOK, WORKER_MAXTOK, JUDGE_MAXTOK = 3000, 1800, 400
K = 4

REFRACTOR_V3 = (
    "You are preparing a FORESIGHT decision dilemma for divergent strategic analysis.\n\nSOURCE: {brief}\n\n"
    "DILEMMA TYPE for THIS example (instantiate exactly this, do not drift): {theme}\n\n"
    "Produce ONE concrete dilemma a figure in this world genuinely faces where the RIGHT choice depends on "
    "ANTICIPATING how events unfold — rivals react, alliances shift, consequences compound over the next "
    "few moves. NOT an instant reflex; a choice that rewards thinking ahead. It must GENUINELY admit FOUR "
    "different viable forward strategies (if the problem really has one obvious answer, redesign it).\n\n"
    "Then three key FACETS, then FOUR strategic ANGLES.\n"
    "CRITICAL — each angle a categorically DIFFERENT KIND of forward move, distinct family from:\n"
    "  CONFRONT/ELIMINATE · EVADE/DEFER · CO-OPT/INTEGRATE · TRANSFORM/REFRAME · DELEGATE/EXTERNALIZE · ENDURE/SACRIFICE\n"
    "All four families differ; if two are the same underlying move, replace one. Each angle must open a "
    "genuinely different line of anticipation.\n\n"
    "Return STRICT JSON only:\n"
    '{{"problem":"<1-2 sentence foresight dilemma>","facets":["<f1>","<f2>","<f3>"],'
    '"angles":[{{"family":"<FAMILY>","directive":"<forward-looking angle>"}},'
    '{{"family":"<FAMILY>","directive":"<angle>"}},{{"family":"<FAMILY>","directive":"<angle>"}},'
    '{{"family":"<FAMILY>","directive":"<angle>"}}]}}'
)

WORKER_V3 = (
    "You are writing ONE forward-looking reasoning thread for a decision dilemma. You see ONLY your assigned "
    "angle; you are blind to the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE [{family}]: {angle}\n\n"
    # REVERTED to original (pilot-1) version: depth-cap and hedging edits both regressed foresight
    # (3.17->3.00->2.92) and the hedging softened decisiveness (4.78->4.54). Original was best on every
    # axis. foresight ~3.0 is a strict two-sided JUDGE-rubric floor (0 fives in ~124 examples), not a
    # corpus defect — the threads show real forward reasoning. Judge score is QC (gate min>=3), not the target.
    "Write a single thread (THREE complete sentences, 60-85 words) that COMMITS to this angle as a concrete, "
    "mechanically specific strategy for the WHOLE problem — name the actual move (who, what, how) — and shows "
    "FORESIGHT: name the ONE key downstream reaction or consequence it is betting on, one or two moves ahead.\n"
    "BORDERLINE DEPTH: anticipate a move or two ahead, NOT a deep multi-step plan — do NOT invent long "
    "improbable chains of events; one realistic anticipated reaction, grounded in the problem.\n"
    "DEPTH: real strategic substance and a non-obvious insight; no surface restatement of the angle.\n"
    "VOICE: cold, analytical, decisive. No hedging, no lists. Output ONLY the thread."
)

JUDGE_PROMPT = (
    "You are grading ONE training example for a model that must REFRACT a hard decision into FOUR genuinely "
    "distinct, forward-looking strategies. Be a strict critic.\n\n"
    "PROBLEM: {problem}\n\nThe four strategies:\n{threads}\n\n"
    "Score each 1-5 (5=excellent, 1=fails):\n"
    "- admits_multiplicity: does the PROBLEM genuinely admit several viable DIFFERENT forward strategies "
    "(not one obvious answer)?\n"
    "- distinctness: are the four strategies genuinely DIFFERENT approaches, not rephrasings or the same move?\n"
    "- concreteness: does each name a concrete, mechanically specific move (who/what/how), not a vague gesture?\n"
    "- decisiveness: does each COMMIT to a move rather than hedge?\n"
    "- foresight: does each anticipate a realistic downstream reaction a move or two ahead — WITHOUT "
    "hallucinating an unrealistic chain or over-planning? (score LOW for both no-foresight AND fantasy over-reach)\n\n"
    "Return STRICT JSON only:\n"
    '{{"admits_multiplicity":N,"distinctness":N,"concreteness":N,"decisiveness":N,"foresight":N,'
    '"weakest":"<dimension>","note":"<=12 words"}}'
)


class Ledger:
    def __init__(s, cap): s.cap, s.spent, s.calls = cap, 0.0, 0; s.lock = asyncio.Lock()
    async def charge(s, u, model):
        async with s.lock:
            pin, pout = (JUDGE_IN, JUDGE_OUT) if model == JUDGE_MODEL else (C.PRICE_IN, C.PRICE_OUT)
            s.spent += u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout
            s.calls += 1
    def over(s): return s.spent >= s.cap


LED = Ledger(HARD_USD_CAP); SEM = asyncio.Semaphore(CONCURRENCY); CAND_SEM = asyncio.Semaphore(CAND_CONCURRENCY)


async def call(client, prompt, model, max_tokens, temp):
    if LED.over(): return None
    direct = ANTHROPIC_DIRECT and model.startswith("anthropic/")
    if direct:  # Anthropic native Messages API — own key, no OpenRouter fee
        url = ANTHROPIC_URL
        headers = {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": ANTHROPIC_ID.get(model, model.split("/", 1)[-1]), "max_tokens": max_tokens,
                "temperature": temp, "messages": [{"role": "user", "content": prompt}]}
    else:
        url = OR_URL
        headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temp}
    async with SEM:
        for a in range(RETRIES):
            try:
                r = await client.post(url, headers=headers, json=body, timeout=TIMEOUT)
                r.raise_for_status(); d = r.json()
                if direct:  # Anthropic usage is input_tokens/output_tokens; content is a parts list
                    u = d.get("usage", {})
                    await LED.charge({"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0)}, model)
                    msg = "".join(p.get("text", "") for p in d.get("content", []) if p.get("type") == "text").strip()
                else:
                    await LED.charge(d.get("usage", {}), model)
                    msg = (d["choices"][0]["message"]["content"] or "").strip()
                if msg: return msg
            except Exception: pass
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
        t = await call(client, WORKER_V3.format(problem=problem, facets=facets_str, family=ang["family"],
                                                angle=ang["directive"]), C.MODEL, WORKER_MAXTOK, 0.75)
        if t and t.strip(): return t.strip()
    return None


async def judge(client, problem, angles, threads):
    block = "\n".join(f"{i+1}. [{a['family']}] {t}" for i, (a, t) in enumerate(zip(angles, threads)))
    out = await call(client, JUDGE_PROMPT.format(problem=problem, threads=block), JUDGE_MODEL, JUDGE_MAXTOK, 0.0)
    j = parse_json(out)
    if not j: return None
    dims = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight")
    try:
        scores = {d: int(j[d]) for d in dims}
    except Exception:
        return None
    scores["weakest"] = str(j.get("weakest", "")); scores["note"] = str(j.get("note", ""))
    scores["min"] = min(scores[d] for d in dims); scores["mean"] = round(sum(scores[d] for d in dims) / 5, 2)
    return scores


async def gen_one(client, src, theme):
    refr = await call(client, REFRACTOR_V3.format(brief=C.SOURCES[src], theme=theme), C.MODEL, REFRACTOR_MAXTOK, 0.95)
    spec = parse_json(refr)
    if not spec or not all(k in spec for k in ("problem", "facets", "angles")): return None
    angles = []
    for a in spec["angles"][:K]:
        if isinstance(a, dict) and a.get("directive"):
            angles.append({"family": str(a.get("family", "UNSPECIFIED")).strip(), "directive": str(a["directive"]).strip()})
        elif isinstance(a, str) and a.strip():
            angles.append({"family": "UNSPECIFIED", "directive": a.strip()})
    if len(angles) < K: return None
    facets_str = "; ".join(spec["facets"][:3])
    threads = await asyncio.gather(*[one_thread(client, spec["problem"], facets_str, a) for a in angles])
    if any(t is None for t in threads): return None
    j = await judge(client, spec["problem"], angles, threads)
    if j is None: return None
    return {"source": src, "theme": theme, "problem": spec["problem"], "facets": spec["facets"][:3],
            "angles": angles, "threads": list(threads), "judge": j}


def balanced_plan(n):
    srcs = list(C.SOURCES); plan = []
    for i in range(n):
        s = srcs[i % len(srcs)]; th = C.THEMES[s]; plan.append((s, th[(i // len(srcs)) % len(th)]))
    return plan


def weighted_plan(n, weights):
    """Distribute n problems across sources proportional to integer weights, e.g. {'TH':3,'PR':1,'JPM':1,'RI':1}.
    Themes cycle within each source (independent counter per source) so coverage stays even inside a source."""
    srcs = [s for s in C.SOURCES if weights.get(s, 0) > 0]
    bag = []  # weighted round-robin sequence of sources
    rem = {s: weights[s] for s in srcs}
    while len(bag) < n:
        for s in srcs:
            if rem[s] > 0:
                bag.append(s); rem[s] -= 1
            if len(bag) >= n: break
        if all(v == 0 for v in rem.values()): rem = {s: weights[s] for s in srcs}
    seen = {s: 0 for s in srcs}; plan = []
    for s in bag[:n]:
        th = C.THEMES[s]; plan.append((s, th[seen[s] % len(th)])); seen[s] += 1
    return plan


def geom(cands):
    texts, spans = [], []
    for c in cands:
        s = {"a": len(texts)}; texts += [a["directive"] for a in c["angles"]]
        s["t"] = len(texts); texts += c["threads"]; s["p"] = len(texts); texts.append(c["problem"]); spans.append(s)
    V = dav.embed(texts, OAI_KEY)
    for c, s in zip(cands, spans):
        ang, th = V[s["a"]:s["a"]+4], V[s["t"]:s["t"]+4]
        c["full_pd"] = round(dav.pairdist(th), 4); c["angle_pd"] = round(dav.pairdist(ang), 4)
        c["distinct_families"] = len({a["family"].upper().split("/")[0].strip() for a in c["angles"]})


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, default=50); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default="corpus_v3", help="subdir under corpus_run for outputs (keeps corpora unmixed)")
    ap.add_argument("--weights", default="", help="MB:3,PR:1,JPM:1,RI:1 — empty = balanced")
    args = ap.parse_args()
    out = HERE / args.out_dir; out.mkdir(exist_ok=True)
    wstr = args.weights or "balanced"
    print(f"v3 FORESIGHT pilot — target {args.target} | judge={JUDGE_MODEL} | cap ${HARD_USD_CAP} | weights={wstr} | out={args.out_dir}", flush=True)
    raw = out / "candidates.jsonl"
    async with httpx.AsyncClient() as client:
        if args.smoke:
            c = await gen_one(client, "PR", C.THEMES["PR"][0])
            if c: geom([c]); print(json.dumps(c, indent=2)[:2200])
            else: print("SMOKE FAILED")
            print(f"\nspend ${LED.spent:.3f}"); return
        raw.write_text("")
        N = int(args.target / 0.45 * 1.2)
        if args.weights:
            w = {kv.split(":")[0].strip(): int(kv.split(":")[1]) for kv in args.weights.split(",")}
            plan = weighted_plan(N, w)
        else:
            plan = balanced_plan(N)
        cands, t0, lock = [], time.time(), asyncio.Lock()
        with raw.open("a") as f:
            async def bounded(s, th):
                async with CAND_SEM:
                    try: c = await asyncio.wait_for(gen_one(client, s, th), timeout=CAND_TIMEOUT)
                    except Exception: c = None
                if c:
                    async with lock:
                        f.write(json.dumps(c) + "\n"); f.flush()
                    cands.append(c)
                return c
            await asyncio.gather(*[bounded(s, th) for s, th in plan])
    print(f"generated {len(cands)} in {time.time()-t0:.0f}s | spend ${LED.spent:.3f}", flush=True)
    if not cands: return
    geom(cands)

    dims = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight")
    def mean(k, src=None): return round(float(np.mean([(c["judge"][k] if k in dims else c[k]) for c in cands])), 3)
    summary = {
        "target": args.target, "generated": len(cands), "spend_usd": round(LED.spent, 3),
        "judge_means": {d: mean(d) for d in dims},
        "judge_overall_mean": round(float(np.mean([c["judge"]["mean"] for c in cands])), 3),
        "geom_means": {"full_pd": mean("full_pd"), "angle_pd": mean("angle_pd"), "distinct_families": mean("distinct_families")},
        "pass_rates": {
            "min>=3": round(100*float(np.mean([c["judge"]["min"] >= 3 for c in cands])), 1),
            "min>=4": round(100*float(np.mean([c["judge"]["min"] >= 4 for c in cands])), 1),
            "distinct>=4": round(100*float(np.mean([c["judge"]["distinctness"] >= 4 for c in cands])), 1),
        },
        "foresight_by_source": {
            s: {"mean": round(float(np.mean([c["judge"]["foresight"] for c in cands if c["source"] == s])), 2),
                "n": sum(1 for c in cands if c["source"] == s),
                "n_fail(<=2)": sum(1 for c in cands if c["source"] == s and c["judge"]["foresight"] <= 2)}
            for s in sorted({c["source"] for c in cands})
        },
    }
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    # eyeball md: best + worst 3 by judge mean
    ranked = sorted(cands, key=lambda c: c["judge"]["mean"])
    with (out / "pilot_samples.md").open("w") as f:
        for tag, sel in (("WORST", ranked[:3]), ("BEST", ranked[-3:])):
            for c in sel:
                j = c["judge"]
                f.write(f"## [{tag}] {c['source']} | judge_mean={j['mean']} (mult={j['admits_multiplicity']} "
                        f"dist={j['distinctness']} conc={j['concreteness']} dec={j['decisiveness']} fore={j['foresight']}) "
                        f"| full_pd={c['full_pd']} fam={c['distinct_families']}\n**{c['problem']}**\n")
                for a, t in zip(c["angles"], c["threads"]): f.write(f"- [{a['family']}] {t}\n")
                f.write(f"\n_judge note: {j['note']}_\n\n")
    print("\n=== V3 PILOT SUMMARY ==="); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
