"""
head2head_v5.py — neutral 3-way benchmark on 48 OOD problems: BASE vs v5-SFT vs Haiku 4.5.

Judge = Gemini 2.5 Pro (neutral: not Anthropic, not our pipeline), 6-dim v5 JUDGE. All provider-NATIVE keys
(no OpenRouter): Gemini via GEMINI_API_KEY (generativelanguage API), Haiku via ANTHROPIC_API_KEY (messages API).

Inputs:
  benchmark/problems.jsonl                          — 48 OOD problems {problem, category}
  benchmark/eval_bench_base_v5_threads.jsonl        — base granite threads  (pulled from pod dav_eval_v5)
  benchmark/eval_bench_sft_v5_threads.jsonl         — v5-SFT threads         (pulled from pod dav_eval_v5)
Haiku competitor threads are generated here (API). Gemini judges all three with the identical prompt.

  python head2head_v5.py
"""
from __future__ import annotations
import asyncio, json, os, re
from pathlib import Path
import numpy as np, httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
HAIKU_ID = "claude-haiku-4-5"
GEMINI_ID = "gemini-2.5-pro"
DIMS = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight", "viability")
BENCH = HERE / "benchmark"
PROBLEMS = BENCH / "problems.jsonl"
SEM = asyncio.Semaphore(6)
SPENT_NOTE = {"gemini_calls": 0, "haiku_calls": 0}

# ---- prompts: BYTE-IDENTICAL to prep_v5.py / dav_eval_v5.py (train == eval == bench) ----
DEC_SYS = "You refract a hard decision problem into four categorically distinct strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
            "categorically different KIND of move (a distinct family) leading to a different action — sharp "
            "alternatives, not rephrasings.\nFormat exactly:\n"
            "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
            "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
WRK_SYS = "You write one precise, decisive, realistic reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two or "
            "three sentences, cold and analytical) that COMMITS to THIS angle as a concrete, realistic, VIABLE "
            "strategy that resolves the whole problem in a distinct way — name the actual first move (who does "
            "what, to whom, by when) and the one most likely downstream consequence it is betting on. It must be "
            "lawful, executable, and unmistakably a different KIND of move than the other families would choose.")

# ---- judge: VERBATIM from gen_v5.py JUDGE_V5 (6-dim) ----
JUDGE_PROMPT = (
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


def fam_of(a):
    m = re.match(r"\s*\[([^\]]+)\]", a); return m.group(1).strip() if m else "UNSPECIFIED"


def parse_decomp(text):
    facets, angles = [], []
    m = re.search(r"FACETS:\s*(.+)", text)
    if m:
        facets = [x.strip() for x in re.split(r"\||;", m.group(1).splitlines()[0]) if x.strip()][:3]
    for n in range(1, 5):
        a = re.search(rf"{n}\)\s*(.+)", text)
        if a: angles.append(a.group(1).strip())
    return facets, angles[:4]


async def anthropic_call(client, system, user, max_tokens):
    body = {"model": HAIKU_ID, "max_tokens": max_tokens, "temperature": 0.0,
            "system": system, "messages": [{"role": "user", "content": user}]}
    async with SEM:
        for a in range(4):
            try:
                r = await client.post("https://api.anthropic.com/v1/messages",
                                      headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                                               "content-type": "application/json"}, json=body, timeout=90)
                r.raise_for_status(); SPENT_NOTE["haiku_calls"] += 1
                return "".join(p.get("text", "") for p in r.json().get("content", []) if p.get("type") == "text").strip()
            except Exception:
                await asyncio.sleep(2 * (a + 1))
    return None


async def gemini_call(client, prompt, max_tokens=4000):
    # Gemini 2.5 Pro is a THINKING model — needs generous maxOutputTokens (thinking tokens count toward it).
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_ID}:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens}}
    async with SEM:
        for a in range(4):
            try:
                r = await client.post(url, json=body, timeout=120)
                r.raise_for_status(); d = r.json(); SPENT_NOTE["gemini_calls"] += 1
                cand = (d.get("candidates") or [{}])[0]
                parts = (cand.get("content") or {}).get("parts") or []
                return "".join(p.get("text", "") for p in parts if "text" in p).strip()
            except Exception:
                await asyncio.sleep(2 * (a + 1))
    return None


async def haiku_refract(client, problem):
    dtext = await anthropic_call(client, DEC_SYS, DEC_USER.format(problem=problem), 800)
    if not dtext: return None
    facets, angles = parse_decomp(dtext)
    if len(angles) < 4 or len(facets) < 1: return None
    threads = await asyncio.gather(*[anthropic_call(client, WRK_SYS,
        WRK_USER.format(problem=problem, facets=" | ".join(facets), angle=a), 300) for a in angles])
    if any(t is None for t in threads): return None
    return {"angles": angles, "threads": [t.strip() for t in threads]}


async def judge(client, problem, angles, threads):
    block = "\n".join(f"{i+1}. [{fam_of(a)}] {t}" for i, (a, t) in enumerate(zip(angles, threads)))
    out = await gemini_call(client, JUDGE_PROMPT.format(problem=problem, threads=block))
    j = parse_json(out)
    if not j: return None
    try:
        s = {d: int(j[d]) for d in DIMS}
    except Exception:
        return None
    s["mean"] = round(sum(s[d] for d in DIMS) / 6, 2)
    return s


def load_threads(path):
    out = {}
    if Path(path).exists():
        for l in Path(path).open():
            r = json.loads(l); out[r["problem"].strip()] = r
    return out


async def main():
    cat = {json.loads(l)["problem"].strip(): json.loads(l).get("category", "?") for l in PROBLEMS.open()}
    probs = list(cat.keys())
    base = load_threads(BENCH / "eval_bench_base_v5_threads.jsonl")
    sft = load_threads(BENCH / "eval_bench_sft_v5_threads.jsonl")
    print(f"head2head_v5 — {len(probs)} OOD problems | judge={GEMINI_ID} (direct) | competitor=Haiku (direct)")
    print(f"  loaded base threads: {len(base)} | sft threads: {len(sft)}", flush=True)
    if not sft:
        print("FATAL: no SFT threads — run dav_eval_v5 on the 48 problems on the pod first."); raise SystemExit(1)

    async with httpx.AsyncClient() as client:
        haiku = {}
        async def gen_haiku(p):
            haiku[p] = await haiku_refract(client, p)
        await asyncio.gather(*[gen_haiku(p) for p in probs])

        async def judge_all(p):
            res = {"problem": p, "category": cat[p]}
            for lbl, src in (("base", base.get(p)), ("sft", sft.get(p)), ("haiku", haiku.get(p))):
                res[lbl] = await judge(client, p, src["angles"], src["threads"]) if src else None
            return res
        rows = await asyncio.gather(*[judge_all(p) for p in probs])

    def agg(key):
        v = [r[key] for r in rows if r.get(key)]
        if not v: return {}
        return {d: round(float(np.mean([x[d] for x in v])), 2) for d in DIMS} | {
            "overall": round(float(np.mean([x["mean"] for x in v])), 2), "n": len(v),
            "dist>=4%": round(100 * float(np.mean([x["distinctness"] >= 4 for x in v])), 1)}
    summary = {"judge": GEMINI_ID, "n_problems": len(probs),
               "base": agg("base"), "sft": agg("sft"), "haiku": agg("haiku"),
               "api_calls": SPENT_NOTE}
    from collections import defaultdict
    bycat = defaultdict(list)
    for r in rows: bycat[r["category"]].append(r)
    summary["by_category"] = {c: {lbl: round(float(np.mean([r[lbl]["mean"] for r in rs if r.get(lbl)])), 2)
                                  for lbl in ("base", "sft", "haiku") if any(r.get(lbl) for r in rs)}
                              for c, rs in sorted(bycat.items())}
    (BENCH / "head2head_v5_summary.json").write_text(json.dumps(summary, indent=2))
    (BENCH / "head2head_v5_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("\n=== HEAD-TO-HEAD v5 (Gemini 2.5 Pro, 6-dim) ===")
    print(json.dumps({k: summary[k] for k in ("base", "sft", "haiku")}, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
