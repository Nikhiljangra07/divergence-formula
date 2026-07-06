"""
gen_essence_ab.py — ESSENCE ROUND generator A/B: DeepSeek V4 Pro vs Claude Sonnet 5, paired on the
SAME 12 influence archetypes, judged by Gemini 2.5 Pro (neutral family — no generator judges itself,
no Anthropic model judges an Anthropic generator).

The question this answers before the ~250-example scale run: does a frontier-tier teacher (Sonnet 5,
~7-8x the price at intro rates) produce essence-round examples that are enough sharper AND more viable
to justify the premium over the proven DeepSeek pipeline?

Self-contained on purpose: its own per-model price ledger (gen_v3's Ledger prices every non-judge call
at DeepSeek rates — would understate Sonnet ~8x), its own hard cap ($4), save-as-you-go.

Gate = Gemini-calibrated: viability>=4 & distinctness>=4 & concreteness>=4 (Gemini is the fair judge;
Haiku's >=3 gate elsewhere is the harsh-judge equivalent of this).

  python gen_essence_ab.py            # 12 themes x 2 generators = 24 candidates
"""
from __future__ import annotations
import asyncio, json, os, time
from collections import Counter
from pathlib import Path
import httpx
import numpy as np
from dotenv import load_dotenv
import config_essence as CE

HERE = Path(__file__).resolve().parent
OUT = HERE / "essence_ab"
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]

DSV4 = "deepseek/deepseek-v4-pro"          # via OpenRouter (no native key)
SONNET5 = "claude-sonnet-5"                 # via Anthropic NATIVE (own key, intro pricing thru 2026-08-31)
GEMINI_ID = "gemini-2.5-pro"                # judge, Gemini NATIVE
GENERATORS = (DSV4, SONNET5)

# $/token — the whole point of a per-model table (Sonnet intro $2/$10; Gemini 2.5 Pro $1.25/$10)
PRICE = {DSV4: (0.43 / 1e6, 0.87 / 1e6), SONNET5: (2.00 / 1e6, 10.00 / 1e6), GEMINI_ID: (1.25 / 1e6, 10.00 / 1e6)}
HARD_USD_CAP = float(os.environ.get("ESSENCE_USD_CAP", "4.0"))

DIMS = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight", "viability")
K = 4
# Refractor 5000 not 3000: DSV4 reasoning eats the budget on hard archetypes and truncates mid-JSON
# (7/45 attempts died this way at 3000 in the scale run). Ceiling, not cost — only generated tokens bill.
REFRACTOR_MAXTOK, WORKER_MAXTOK, JUDGE_MAXTOK = 5000, 1600, 4000
# 180s not 90: juicy themes (whisper-campaign, mediating-rivals) push DSV4 reasoning to ~2.3k tokens
# and past 90s — every call was being killed mid-generation and retried forever (59-attempt fail loop).
TIMEOUT = float(os.environ.get("ESSENCE_TIMEOUT", "180"))
RETRIES, THREAD_REGENS = 4, 2
# 12 not 6: Gemini judge calls are 30-90s (thinking) and hog the slots; 6 choked the scale run to 20/hr.
SEM = asyncio.Semaphore(int(os.environ.get("ESSENCE_CONC", "12")))

# judge prompt VERBATIM from gen_v5.py JUDGE_V5 — scores stay comparable to every prior number
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


class Ledger:
    def __init__(s, cap):
        s.cap, s.spent, s.calls = cap, 0.0, 0
        s.by_model = Counter()
        s.lock = asyncio.Lock()
    async def charge(s, model, tin, tout):
        async with s.lock:
            pin, pout = PRICE[model]
            usd = tin * pin + tout * pout
            s.spent += usd; s.by_model[model] += usd; s.calls += 1
    def over(s): return s.spent >= s.cap


LED = Ledger(HARD_USD_CAP)


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


async def call_openrouter(client, prompt, model, max_tokens, temp):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                          headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"},
                          json=body, timeout=TIMEOUT)
    r.raise_for_status(); d = r.json()
    u = d.get("usage", {})
    await LED.charge(model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    return (d["choices"][0]["message"]["content"] or "").strip()


async def call_anthropic(client, prompt, model, max_tokens, temp):
    # Sonnet 5 REJECTS `temperature` with a 400 ("deprecated for this model") — the model samples on
    # its own. Discovered live: all 12 refractor calls failed on this before the fix.
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    r = await client.post("https://api.anthropic.com/v1/messages",
                          headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"}, json=body, timeout=TIMEOUT)
    r.raise_for_status(); d = r.json()
    u = d.get("usage", {})
    await LED.charge(model, u.get("input_tokens", 0), u.get("output_tokens", 0))
    return "".join(p.get("text", "") for p in d.get("content", []) if p.get("type") == "text").strip()


async def call_gemini(client, prompt, max_tokens=JUDGE_MAXTOK):
    # Gemini 2.5 Pro is a thinking model — thinking tokens count toward maxOutputTokens; usageMetadata
    # exposes them as thoughtsTokenCount (billed at output rate), so charge candidates+thoughts.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_ID}:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens}}
    r = await client.post(url, json=body, timeout=120)
    r.raise_for_status(); d = r.json()
    u = d.get("usageMetadata", {})
    await LED.charge(GEMINI_ID, u.get("promptTokenCount", 0),
                     u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0))
    cand = (d.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if "text" in p).strip()


async def gen_call(client, prompt, model, max_tokens, temp):
    """Route to the generator's own provider; retry with backoff; None on exhaustion/cap."""
    if LED.over(): return None
    fn = call_anthropic if model == SONNET5 else call_openrouter
    async with SEM:
        for a in range(RETRIES):
            try:
                msg = await fn(client, prompt, model, max_tokens, temp)
                if msg: return msg
            except Exception:
                pass
            await asyncio.sleep(2 * (a + 1))
    return None


async def judge(client, problem, angles, threads):
    block = "\n".join(f"{i+1}. [{a['family']}] {t}" for i, (a, t) in enumerate(zip(angles, threads)))
    async with SEM:
        for a in range(RETRIES):
            try:
                out = await call_gemini(client, JUDGE_PROMPT.format(problem=problem, threads=block))
                j = parse_json(out)
                if j:
                    s = {d: int(j[d]) for d in DIMS}
                    s["weakest"] = str(j.get("weakest", "")); s["note"] = str(j.get("note", ""))
                    s["mean"] = round(sum(s[d] for d in DIMS) / 6, 2)
                    return s
            except Exception:
                pass
            await asyncio.sleep(2 * (a + 1))
    return None


async def one_thread(client, model, problem, facets_str, ang):
    for _ in range(THREAD_REGENS + 1):
        t = await gen_call(client, CE.WORKER_ESSENCE.format(
            problem=problem, facets=facets_str, family=ang["family"], angle=ang["directive"]),
            model, WORKER_MAXTOK, 0.75)
        if t and t.strip(): return t.strip()
    return None


async def gen_one(client, model, src, theme):
    refr = await gen_call(client, CE.REFRACTOR_ESSENCE.format(
        brief=CE.SOURCES[src], theme=theme, setting=CE.SETTING_MODERN), model, REFRACTOR_MAXTOK, 0.95)
    spec = parse_json(refr)
    if not spec or not all(k in spec for k in ("problem", "facets", "angles")):
        return {"generator": model, "source": src, "theme": theme, "fail": "refractor"}
    angles = []
    for a in spec["angles"][:K]:
        if isinstance(a, dict) and a.get("directive"):
            angles.append({"family": str(a.get("family", "UNSPECIFIED")).strip(),
                           "directive": str(a["directive"]).strip()})
    if len(angles) < K:
        return {"generator": model, "source": src, "theme": theme, "fail": "angles"}
    facets_str = "; ".join(spec["facets"][:3])
    threads = await asyncio.gather(*[one_thread(client, model, spec["problem"], facets_str, a) for a in angles])
    if any(t is None for t in threads):
        return {"generator": model, "source": src, "theme": theme, "fail": "worker"}
    j = await judge(client, spec["problem"], angles, threads)
    if j is None:
        return {"generator": model, "source": src, "theme": theme, "fail": "judge"}
    return {"generator": model, "source": src, "theme": theme, "setting": "modern",
            "problem": spec["problem"], "facets": spec["facets"][:3], "angles": angles,
            "pos_threads": list(threads), "pos_judge": j}


def gate(j):  # Gemini fair-judge calibration
    return j["viability"] >= 4 and j["distinctness"] >= 4 and j["concreteness"] >= 4


async def main():
    OUT.mkdir(exist_ok=True)
    raw = OUT / "candidates.jsonl"
    pairs = [(s, th) for s in CE.SOURCES for th in CE.THEMES[s]]  # 12 themes
    # RESUMABLE: keep prior successes (rows with pos_judge), retry only missing/failed combos —
    # so the temperature-bug rerun keeps the already-judged DeepSeek half instead of re-paying for it.
    results = []
    if raw.exists():
        for line in raw.open():
            try: c = json.loads(line)
            except Exception: continue
            if "pos_judge" in c: results.append(c)
    done_keys = {(c["generator"], c["theme"]) for c in results}
    todo = [(m, s, th) for m in GENERATORS for s, th in pairs if (m, th) not in done_keys]
    raw.write_text("".join(json.dumps(c) + "\n" for c in results))
    print(f"ESSENCE A/B — {len(pairs)} themes x 2 generators | judge={GEMINI_ID} (neutral) | "
          f"gate=viab&dist&conc>=4 (Gemini) | cap ${HARD_USD_CAP} | resume: {len(results)} kept, "
          f"{len(todo)} to run", flush=True)
    t0 = time.time(); lock = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        async def run(model, src, theme):
            c = await gen_one(client, model, src, theme)
            async with lock:
                with raw.open("a") as f:
                    f.write(json.dumps(c) + "\n")
                results.append(c)
                done = len(results)
                if done % 6 == 0:
                    print(f"  {done}/{len(pairs)*2} | ${LED.spent:.2f}", flush=True)
        await asyncio.gather(*[run(m, s, th) for m, s, th in todo])

    summary = {"judge": GEMINI_ID, "themes": len(pairs), "elapsed_s": round(time.time() - t0),
               "spend_usd_total": round(LED.spent, 3),
               "spend_usd_by_model": {m: round(v, 3) for m, v in LED.by_model.items()}}
    for m in GENERATORS:
        ok = [c for c in results if c["generator"] == m and "pos_judge" in c]
        fails = Counter(c["fail"] for c in results if c["generator"] == m and "fail" in c)
        entry = {"n_ok": len(ok), "gen_failures": dict(fails)}
        if ok:
            entry["judge_means"] = {d: round(float(np.mean([c["pos_judge"][d] for c in ok])), 2) for d in DIMS}
            entry["overall"] = round(float(np.mean([c["pos_judge"]["mean"] for c in ok])), 2)
            entry["gate_yield_pct"] = round(100 * float(np.mean([gate(c["pos_judge"]) for c in ok])), 1)
        summary[m] = entry
    (OUT / "ab_summary.json").write_text(json.dumps(summary, indent=2))

    # side-by-side eyeball file, paired by theme
    with (OUT / "ab_samples.md").open("w") as f:
        for s, th in pairs:
            f.write(f"# [{s}] {th[:100]}\n\n")
            for m in GENERATORS:
                c = next((c for c in results if c["generator"] == m and c["theme"] == th), None)
                tag = "DSV4" if m == DSV4 else "SONNET5"
                if not c or "pos_judge" not in c:
                    f.write(f"## {tag} — FAILED ({(c or {}).get('fail','missing')})\n\n"); continue
                j = c["pos_judge"]
                f.write(f"## {tag} | mean={j['mean']} viab={j['viability']} dist={j['distinctness']} "
                        f"conc={j['concreteness']} decis={j['decisiveness']} fore={j['foresight']} "
                        f"| gate={'PASS' if gate(j) else 'fail'}\n**{c['problem']}**\n")
                for a, t in zip(c["angles"], c["pos_threads"]):
                    f.write(f"- [{a['family']}] {t}\n")
                f.write(f"\n_judge: weakest={j['weakest']} — {j['note']}_\n\n")
    print("\n=== ESSENCE A/B SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
