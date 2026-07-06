"""
generate.py — DeepSeek V4 Pro generates divergent examples across 4 sources; DAV gates them.

ISOLATED + SAFE. Reads only OPENROUTER_API_KEY (generation) and OPENAI_API_KEY (embeddings)
from the reasoningEngine .env. Writes only into ./out/. Hard money cap with a live ledger.
Refractor + ISOLATED workers (never one-shot). Passers are written UNLABELED.
"""
from __future__ import annotations
import asyncio, json, os, re, time
from pathlib import Path
import numpy as np
import httpx
from dotenv import load_dotenv

load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]
OAI_KEY = os.environ["OPENAI_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OAI_URL = "https://api.openai.com/v1/embeddings"
MODEL = "deepseek/deepseek-v4-pro"

# --- knobs / safety ---------------------------------------------------------------
HARD_USD_CAP = 3.00
PRICE_IN, PRICE_OUT = 0.43 / 1e6, 0.87 / 1e6
CONCURRENCY = 10
TIMEOUT = 180.0
RETRIES = 4
K = 4                        # threads per example
TARGET = 80                  # candidates (20/source) → gate down to ≤50 with margin
KEEP = 50
# deepseek-v4-pro is a REASONING model: reasoning tokens count against max_tokens.
# Give generous breathing room so content emits instead of starving mid-reasoning (empties).
REFRACTOR_MAXTOK = 2000
WORKER_MAXTOK = 1000
THREAD_REGENS = 2            # regenerate a single failed thread before dropping the candidate
LAM = 1e-3
# DAV thresholds, calibrated from the 40-example study (embedder = text-embedding-3-small)
VOL_GATE = -8.0              # genuine ≈ -6.0..-6.2 ; collapse ≈ -12 → -8 cleanly separates
EPS_G = 0.27                 # grounding floor
EPS_W = 0.18                 # wholeness floor (compressed embedder scale)
PD_FLOOR = 0.30             # pairwise-distance floor (catch near-collapse); also the ranker
NEAR_MARGIN = 0.06          # fail one gate within this margin → near-miss (hard negative)

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)

SOURCES = {
    "RI": "Reverend Insanity — a Chinese cultivation novel of ruthless, amoral strategy. Its "
          "protagonist values only himself and schemes coldly: concealment, betrayal timing, "
          "resource gambles, alliances of convenience, sacrificing others as tools.",
    "JPM": "Jin Ping Mei — a Chinese domestic novel of a corrupt merchant household. Its "
           "dilemmas are social and economic: rival wives and concubines, bribery of officials, "
           "managing servants, reputation, favor and money as currency, inheritance intrigue.",
    "PR": "The Prince (Machiavelli) — a European treatise on acquiring and holding power. Its "
          "dilemmas are statecraft: holding conquered provinces, cruelty vs mercy, feared vs "
          "loved, mercenaries vs citizen arms, treating nobles and deposed heirs, fortune vs caution.",
    "MB": "The Mahabharata — an Indian epic of dharmic dilemmas: duty vs kinship, vows that bind "
          "to unjust causes, the ethics of war and deception, sacrifice for a greater cause, "
          "where honor, consequence, and kinship pull against each other.",
}

REFRACTOR = (
    "You are preparing a decision dilemma from a source for divergent analysis.\n\nSOURCE: {brief}\n\n"
    "Produce ONE concrete, faithful decision dilemma a figure in this world genuinely faces — a hard "
    "choice with several defensible directions. Then give three key FACETS (sub-aspects any complete "
    "answer must engage) and FOUR genuinely DISTINCT angles for approaching the WHOLE problem. The four "
    "angles must lead to different actions — real alternatives, not rephrasings of one another.\n\n"
    "Avoid the single most obvious textbook example; vary it.\n\n"
    "Return STRICT JSON only, no prose:\n"
    '{{"problem": "<1-2 sentence dilemma>", "facets": ["<f1>","<f2>","<f3>"], '
    '"angles": ["<angle1 directive>","<angle2>","<angle3>","<angle4>"]}}'
)

WORKER = (
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; "
    "you cannot see the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE: {angle}\n\n"
    "Write a single thread (~2 sentences, 35-55 words) that pursues THIS angle as a strategy for the "
    "WHOLE problem and ends with a concrete projected consequence — what it costs and what it gains. "
    "Voice: cold, analytical, decisive. No hedging, no therapy language, no lists. Vary your phrasing; "
    "do NOT reuse a fixed 'the cost is X but you gain Y' template. Output ONLY the thread text."
)


class Ledger:
    def __init__(self, cap): self.cap, self.spent, self.calls = cap, 0.0, 0; self.lock = asyncio.Lock()
    async def charge(self, u):
        c = u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT
        async with self.lock: self.spent += c; self.calls += 1
        return c
    def over(self): return self.spent >= self.cap


LED = Ledger(HARD_USD_CAP); SEM = asyncio.Semaphore(CONCURRENCY)


async def dsv4(client, prompt, max_tokens, temp):
    if LED.over(): return None
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    async with SEM:
        for a in range(RETRIES):
            try:
                r = await client.post(OR_URL, headers=headers, json=body, timeout=TIMEOUT)
                r.raise_for_status(); d = r.json()
                await LED.charge(d.get("usage", {}))
                msg = (d["choices"][0]["message"]["content"] or "").strip()
                if msg: return msg
                # empty content = reasoning ate the whole budget; back off and retry with room
                if a == RETRIES - 1: print(f"  [dsv4] empty content after {RETRIES} tries", flush=True)
            except Exception as e:
                if a == RETRIES - 1: print(f"  [dsv4] fail: {type(e).__name__}: {e}", flush=True)
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


async def one_thread(client, problem, facets_str, angle):
    """One ISOLATED worker thread; regenerate a few times before giving up (resilience)."""
    for _ in range(THREAD_REGENS + 1):
        t = await dsv4(client, WORKER.format(problem=problem, facets=facets_str, angle=angle),
                       WORKER_MAXTOK, 0.75)
        if t and t.strip():
            return t.strip()
    return None


async def gen_one(client, src):
    refr = await dsv4(client, REFRACTOR.format(brief=SOURCES[src]), REFRACTOR_MAXTOK, 0.95)
    spec = parse_json(refr)
    if not spec or not all(k in spec for k in ("problem", "facets", "angles")): return None
    angles = spec["angles"][:K]
    if len(angles) < K: return None
    # ISOLATED workers — each its own call, blind to the others
    facets_str = "; ".join(spec["facets"][:3])
    threads = await asyncio.gather(*[
        one_thread(client, spec["problem"], facets_str, a) for a in angles
    ])
    if any(t is None for t in threads): return None
    return {"source": src, "problem": spec["problem"], "facets": spec["facets"][:3],
            "threads": [t for t in threads]}


# --- embeddings + DAV --------------------------------------------------------------
def embed(texts):
    out = []
    for i in range(0, len(texts), 256):
        r = httpx.post(OAI_URL, headers={"Authorization": f"Bearer {OAI_KEY}"},
                       json={"model": "text-embedding-3-small", "input": texts[i:i+256]}, timeout=60)
        r.raise_for_status(); out += [d["embedding"] for d in r.json()["data"]]
    V = np.array(out, float); return V / np.linalg.norm(V, axis=1, keepdims=True)


def volume(T):
    c = T - T.mean(0); n = np.linalg.norm(c, axis=1, keepdims=True)
    if (n < 1e-6).any(): return float("-inf")
    c = c / n; K_ = c @ c.T
    return float(np.linalg.slogdet(K_ + LAM * np.eye(len(T)))[1])


def pairdist(T):
    S = T @ T.T; iu = np.triu_indices(len(T), 1); return float((1 - S[iu]).mean())


# A thread "projects a consequence" if it names a cost / gain / trade-off. The worker prompt
# deliberately VARIES phrasing (no fixed "cost is X but Y" template), so the gate must recognise
# the full consequence/trade vocabulary — not just a handful of keywords — or it false-rejects
# genuinely consequence-bearing threads (verified on the first run: 16/16 rejects were false).
CONS_RE = re.compile(
    r"\b(cost|costs|price|gain|gains|lose|loses|lost|losing|forfeit|forfeits|"
    r"surrender|surrenders|sacrifice|sacrifices|trade|trades|trading|exchange|"
    r"buy|buys|buying|bought|purchase|purchases|purchasing|convert|converts|converting|"
    r"transform|transforms|secure|secures|preserve|preserves|ensure|ensures|"
    r"yield|yields|risk|risks|spend|spends|spent|but|yet|though|however|whereas|while)\b|—",
    re.I,
)
def consequence_ok(threads):
    return all(len(t.split()) >= 18 and CONS_RE.search(t) for t in threads)


def main():
    print("=" * 64)
    print(f"DSV4 pilot — target {TARGET} candidates → ≤{KEEP} passers | cap ${HARD_USD_CAP:.2f}")
    print("=" * 64)

    # PHASE 1 — generate (balanced round-robin across sources), save-as-you-go
    order = [list(SOURCES)[i % len(SOURCES)] for i in range(TARGET)]
    raw_path = OUT / "candidates_raw.jsonl"; raw_path.write_text("")

    async def run():
        async with httpx.AsyncClient() as client:
            done = 0
            results = await asyncio.gather(*[gen_one(client, s) for s in order])
            with raw_path.open("a") as f:
                for c in results:
                    if c:
                        f.write(json.dumps(c) + "\n"); done += 1
            return [c for c in results if c], done

    t0 = time.time()
    cands, ok = asyncio.run(run())
    print(f"generated {ok}/{TARGET} candidates in {time.time()-t0:.0f}s | "
          f"calls={LED.calls} spend=${LED.spent:.3f}/{HARD_USD_CAP:.0f}", flush=True)
    if not cands:
        print("no candidates — aborting."); return

    # PHASE 2 — embed + DAV score
    texts, spans = [], []
    for c in cands:
        s = {"p": len(texts)}; texts.append(c["problem"])
        s["f"] = len(texts); texts += c["facets"]
        s["t"] = len(texts); texts += c["threads"]
        spans.append(s)
    V = embed(texts)
    for c, s in zip(cands, spans):
        p = V[s["p"]]; f = V[s["f"]:s["f"]+3]; t = V[s["t"]:s["t"]+len(c["threads"])]
        c["volume"] = volume(t); c["ground"] = float((t @ p).min())
        c["whole"] = float((t @ f.T).mean(1).min()); c["pairdist"] = pairdist(t)
        c["cons_ok"] = consequence_ok(c["threads"])

    # PHASE 3 — gate
    def verdict(c):
        fails = []
        if c["volume"] <= VOL_GATE: fails.append(("volume", VOL_GATE - c["volume"]))
        if c["ground"] < EPS_G: fails.append(("ground", EPS_G - c["ground"]))
        if c["whole"] < EPS_W: fails.append(("whole", EPS_W - c["whole"]))
        if c["pairdist"] < PD_FLOOR: fails.append(("pairdist", PD_FLOOR - c["pairdist"]))
        if not c["cons_ok"]: fails.append(("consequence", 1.0))
        if not fails: return "pass", fails
        if len(fails) == 1 and fails[0][0] != "consequence" and fails[0][1] <= NEAR_MARGIN:
            return "near_miss", fails
        return "reject", fails

    passers, near, rejects = [], [], []
    for c in cands:
        v, fails = verdict(c); c["_fails"] = fails
        (passers if v == "pass" else near if v == "near_miss" else rejects).append(c)
    passers.sort(key=lambda c: c["pairdist"], reverse=True)  # rank by divergence
    passers = passers[:KEEP]

    # PHASE 4 — write (passers UNLABELED), diagnostics labeled
    with (OUT / "passers.jsonl").open("w") as f:
        for c in passers:
            f.write(json.dumps({"problem": c["problem"], "facets": c["facets"], "threads": c["threads"]}) + "\n")
    with (OUT / "near_miss.jsonl").open("w") as f:
        for c in near: f.write(json.dumps({k: c[k] for k in ("problem", "facets", "threads", "_fails")}) + "\n")
    with (OUT / "rejects.jsonl").open("w") as f:
        for c in rejects: f.write(json.dumps({"source": c["source"], "_fails": c["_fails"],
                                              "metrics": {m: round(c[m], 3) for m in ("volume","ground","whole","pairdist")}}) + "\n")
    with (OUT / "passers.md").open("w") as f:
        f.write(f"# {len(passers)} passers (unlabeled in jsonl; source shown here for your QA)\n\n")
        for i, c in enumerate(passers, 1):
            f.write(f"## {i}. [{c['source']}]  vol={c['volume']:.2f} pd={c['pairdist']:.2f} "
                    f"grd={c['ground']:.2f} whl={c['whole']:.2f}\n\n**{c['problem']}**\n\n")
            for t in c["threads"]: f.write(f"- {t}\n")
            f.write("\n")

    # diagnostics
    from collections import Counter
    mix = Counter(c["source"] for c in passers)
    arr = lambda k, L: np.array([c[k] for c in L]) if L else np.array([0.0])
    summary = {
        "generated": ok, "target": TARGET,
        "passers": len(passers), "near_miss": len(near), "rejects": len(rejects),
        "pass_rate": round(len(passers) / max(ok, 1), 3),
        "emergent_source_mix": dict(mix),
        "passer_means": {k: round(float(arr(k, passers).mean()), 3) for k in ("volume","ground","whole","pairdist")},
        "reject_reasons": dict(Counter(f[0] for c in rejects for f in c["_fails"])),
        "spend_usd": round(LED.spent, 3), "dsv4_calls": LED.calls,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- RESULTS ---")
    print(f"passers {len(passers)} | near-miss {len(near)} | rejects {len(rejects)} "
          f"| pass-rate {summary['pass_rate']:.0%}")
    print(f"emergent source mix (passers): {dict(mix)}")
    print(f"passer means: {summary['passer_means']}")
    print(f"reject reasons: {summary['reject_reasons']}")
    print(f"spend: ${LED.spent:.3f} / ${HARD_USD_CAP:.0f}")
    print(f"\nread → {OUT/'passers.md'}")


if __name__ == "__main__":
    main()
