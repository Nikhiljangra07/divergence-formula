"""
head2head.py — trained model (v4@r64) vs Haiku 4.5 at refraction, judged by Gemini 2.5 Pro (neutral).

Fair setup:
  - Haiku runs the SAME refraction harness (same DEC + WRK prompts as dav_eval_v4) on the 48 OOD problems.
  - Our trained model's outputs are ALREADY saved (eval_bench_trained_v4_threads.jsonl) — reused, not re-run.
  - Gemini 2.5 Pro judges BOTH sides with the identical JUDGE_PROMPT (neutral: not Anthropic, not our pipeline).
All API — no pod. Prompts copied verbatim from dav_eval_v4.py / gen_v3.py.

  python head2head.py
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, time
from pathlib import Path
import numpy as np, httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1/chat/completions"
HAIKU = "anthropic/claude-haiku-4.5"
JUDGE = os.environ.get("H2H_JUDGE", "google/gemini-2.5-pro")
DIMS = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight")
CAP = float(os.environ.get("H2H_CAP", "6.0"))

PROBLEMS = HERE / "benchmark" / "problems.jsonl"
TRAINED = Path("/Users/nikhil/Desktop/divergent-model-backups/benchmark_run/eval_bench_trained_v4_threads.jsonl")
OUT = HERE / "benchmark"

# ---- prompts: verbatim from dav_eval_v4.py ----
DEC_SYS = "You refract a hard decision problem into four categorically distinct strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
            "categorically different KIND of move (a distinct family) leading to a different action — sharp "
            "alternatives, not rephrasings.\nFormat exactly:\n"
            "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
            "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
WRK_SYS = "You write one precise, decisive reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two "
            "sentences, cold and analytical) that COMMITS to THIS angle as a concrete, mechanically specific "
            "strategy for the whole problem — a sharp, non-obvious move that is unmistakably a different KIND "
            "of move than the other families would choose.")

# ---- judge: verbatim from gen_v3.py ----
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

SPENT = {"v": 0.0}
SEM = asyncio.Semaphore(8)


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


def parse_decomp(text):
    facets, angles = [], []
    m = re.search(r"FACETS:\s*(.+)", text)
    if m:
        facets = [x.strip() for x in re.split(r"\||;", m.group(1).splitlines()[0]) if x.strip()][:3]
    for n in range(1, 5):
        a = re.search(rf"{n}\)\s*(.+)", text)
        if a: angles.append(a.group(1).strip())
    return facets, angles[:4]


def fam_of(a):
    m = re.match(r"\s*\[([^\]]+)\]", a); return m.group(1).strip() if m else "UNSPECIFIED"


async def call(client, model, system, user, max_tokens, temp):
    if SPENT["v"] >= CAP: return None
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temp}
    async with SEM:
        for a in range(4):
            try:
                r = await client.post(URL, headers={"Authorization": f"Bearer {OR}"}, json=body, timeout=90)
                r.raise_for_status(); d = r.json()
                u = d.get("usage", {}); SPENT["v"] += (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)) * 3e-6
                msg = (d["choices"][0]["message"]["content"] or "").strip()
                if msg: return msg
            except Exception:
                await asyncio.sleep(2 * (a + 1))
    return None


async def haiku_refract(client, problem):
    dtext = await call(client, HAIKU, DEC_SYS, DEC_USER.format(problem=problem), 800, 0.0)
    if not dtext: return None
    facets, angles = parse_decomp(dtext)
    if len(angles) < 4 or len(facets) < 1: return None
    threads = await asyncio.gather(*[call(client, HAIKU, WRK_SYS,
                WRK_USER.format(problem=problem, facets=" | ".join(facets), angle=a), 300, 0.0) for a in angles])
    if any(t is None for t in threads): return None
    return {"angles": angles, "threads": [t.strip() for t in threads]}


async def judge(client, problem, angles, threads):
    block = "\n".join(f"{i+1}. [{fam_of(a)}] {t}" for i, (a, t) in enumerate(zip(angles, threads)))
    out = await call(client, JUDGE, None, JUDGE_PROMPT.format(problem=problem, threads=block), 3000, 0.0)  # Gemini 2.5 Pro is a thinking model — needs headroom for reasoning + JSON
    j = parse_json(out)
    if not j: return None
    try:
        s = {d: int(j[d]) for d in DIMS}
    except Exception:
        return None
    s["mean"] = round(sum(s[d] for d in DIMS) / 5, 2)
    return s


async def main():
    cat = {json.loads(l)["problem"].strip(): json.loads(l)["category"] for l in PROBLEMS.open()}
    trained = {json.loads(l)["problem"].strip(): json.loads(l) for l in TRAINED.open()}
    probs = list(cat.keys())
    print(f"head2head — {len(probs)} problems | competitor=Haiku 4.5 | judge={JUDGE} | cap ${CAP}", flush=True)

    # reuse saved Haiku refractions if present (deterministic; avoids re-spending) — only judging failed before
    saved = {}
    rows_path = OUT / "head2head_rows.jsonl"
    if rows_path.exists():
        for l in rows_path.open():
            r = json.loads(l)
            if r.get("haiku_out"): saved[r["problem"].strip()] = r["haiku_out"]
        print(f"reusing {len(saved)} saved Haiku refractions (re-judge only)", flush=True)

    async with httpx.AsyncClient() as client:
        # 1. Haiku refracts (same harness) — or reuse saved
        haiku_out = [saved.get(p) or await haiku_refract(client, p) for p in probs] if saved else \
                    await asyncio.gather(*[haiku_refract(client, p) for p in probs])
        # 2. judge both sides with Gemini
        async def judge_pair(p, ho):
            tr = trained.get(p)
            tj = await judge(client, p, tr["angles"], tr["threads"]) if tr else None
            hj = await judge(client, p, ho["angles"], ho["threads"]) if ho else None
            return {"problem": p, "category": cat[p], "trained_judge": tj, "haiku_judge": hj,
                    "haiku_out": ho}
        rows = await asyncio.gather(*[judge_pair(p, ho) for p, ho in zip(probs, haiku_out)])

    # aggregate
    def agg(rows, key):
        v = [r[key] for r in rows if r[key]]
        return {d: round(float(np.mean([x[d] for x in v])), 2) for d in DIMS} | {
            "overall": round(float(np.mean([x["mean"] for x in v])), 2), "n": len(v)}
    summary = {"judge": JUDGE, "overall_trained": agg(rows, "trained_judge"),
               "overall_haiku": agg(rows, "haiku_judge"), "spend_usd": round(SPENT["v"], 3)}
    # per category
    from collections import defaultdict
    bycat = defaultdict(list)
    for r in rows: bycat[r["category"]].append(r)
    summary["by_category"] = {}
    for c, rs in sorted(bycat.items()):
        t = [r["trained_judge"] for r in rs if r["trained_judge"]]
        h = [r["haiku_judge"] for r in rs if r["haiku_judge"]]
        summary["by_category"][c] = {
            "trained_overall": round(float(np.mean([x["mean"] for x in t])), 2) if t else None,
            "haiku_overall": round(float(np.mean([x["mean"] for x in h])), 2) if h else None,
            "trained_distinct": round(float(np.mean([x["distinctness"] for x in t])), 2) if t else None,
            "haiku_distinct": round(float(np.mean([x["distinctness"] for x in h])), 2) if h else None,
        }
    (OUT / "head2head_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "head2head_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("\n=== HEAD-TO-HEAD (Gemini 2.5 Pro judge) ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
