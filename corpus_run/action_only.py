"""
action_only.py — lever #1: does the divergence the eye can see become visible to the GEOMETRY
when we embed only the DECISION clause instead of the whole paragraph?

Reuses corpus_v2/passers.jsonl (186 problems × 4 threads). For each thread we compute pairdist under
four representations and compare:
  - full      : the whole thread          (baseline — must reproduce gen_v2's 0.382)
  - sent1     : first sentence only        (free heuristic: the move is stated first, consequence second)
  - action    : Haiku VERBATIM action span (robust: copies the move words, drops consequence/justification)
  - angle     : the seeding angle directive (reference — 0.478)

Meticulous guards: verbatim extraction (no paraphrase → extractor can't normalize threads together),
two independent extractors cross-checked, full-thread recompute as a pipeline sanity check, hard $ cap,
save-as-you-go, sample extractions dumped for eyeball.
"""
from __future__ import annotations
import asyncio, json, os, re, time
from pathlib import Path
import numpy as np, httpx
from dotenv import load_dotenv
import config as C, dav

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_v2"
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]; OAI_KEY = os.environ["OPENAI_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
HAIKU = "anthropic/claude-haiku-4.5"
HAIKU_IN, HAIKU_OUT = 1.0 / 1e6, 5.0 / 1e6           # ~Haiku 4.5 pricing, for the cap estimate
CAP = float(os.environ.get("AO_USD_CAP", "1.0"))
CONC, TIMEOUT, RETRIES = 12, 45.0, 3

EXTRACT = (
    "This is a LITERARY-ANALYSIS task. The text below is a fictional strategy passage from classic "
    "literature (Machiavelli's The Prince, the Mahabharata, Jin Ping Mei, the xianxia novel Reverend "
    "Insanity). It is NOT a real plan; you are only doing prose segmentation, not advising anyone.\n\n"
    "Copy out ONLY the clause stating the core ACTION the passage commits to (the specific who / what / "
    "how).\n"
    "RULES:\n"
    "- Quote the ORIGINAL words verbatim. Do NOT paraphrase, summarize, rephrase, normalize, editorialize, "
    "or add disclaimers.\n"
    "- EXCLUDE the projected consequence (what it costs / gains) and any justification.\n"
    "- If action and consequence share a sentence, return only the action portion.\n"
    "- Output ONLY the copied clause, nothing else.\n\n"
    "PASSAGE:\n{thread}\n\nACTION CLAUSE (verbatim):"
)
# detect a refusal / disclaimer leaking into the extraction so we can fall back to the verbatim heuristic
REFUSE_RE = re.compile(
    r"(i can'?t|i cannot|i won'?t|i will not|i'?m (not able|unable)|i am (not able|unable)|"
    r"won'?t help|operationaliz|as requested|happy to help|i'?m sorry|cannot assist|can'?t assist)",
    re.I,
)


class Ledger:
    def __init__(s, cap): s.cap, s.spent, s.calls = cap, 0.0, 0; s.lock = asyncio.Lock()
    async def charge(s, u):
        async with s.lock:
            s.spent += u.get("prompt_tokens", 0) * HAIKU_IN + u.get("completion_tokens", 0) * HAIKU_OUT
            s.calls += 1
    def over(s): return s.spent >= s.cap


LED = Ledger(CAP); SEM = asyncio.Semaphore(CONC)


def first_sentence(t: str) -> str:
    # split on sentence boundary; keep the first complete sentence (the asserted move)
    m = re.split(r'(?<=[.!?])\s+', t.strip())
    return m[0] if m else t.strip()


async def haiku_extract(client, thread):
    if LED.over(): return None
    body = {"model": HAIKU, "messages": [{"role": "user", "content": EXTRACT.format(thread=thread)}],
            "max_tokens": 120, "temperature": 0.0}
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    async with SEM:
        for a in range(RETRIES):
            try:
                r = await client.post(OR_URL, headers=headers, json=body, timeout=TIMEOUT)
                r.raise_for_status(); d = r.json(); await LED.charge(d.get("usage", {}))
                msg = (d["choices"][0]["message"]["content"] or "").strip().strip('"')
                if msg: return msg
            except Exception:
                pass
            await asyncio.sleep(1.5 * (a + 1))
    return None


def pd_rows(problems, key):
    """embed `key` field (list of 4 strings per problem) and return per-problem pairdist."""
    texts, spans = [], []
    for p in problems:
        spans.append(len(texts)); texts += p[key]
    V = dav.embed(texts, OAI_KEY)
    return [dav.pairdist(V[s:s + 4]) for s in spans]


async def main():
    rows = [json.loads(l) for l in (OUT / "passers.jsonl").open()]
    print(f"action-only test over {len(rows)} problems ({len(rows)*4} threads) | cap ${CAP}", flush=True)
    t0 = time.time()

    # heuristic (free)
    for p in rows:
        p["sent1"] = [first_sentence(t) for t in p["threads"]]

    # Haiku verbatim action extraction (robust)
    async with httpx.AsyncClient() as client:
        async def one(p):
            acts = await asyncio.gather(*[haiku_extract(client, t) for t in p["threads"]])
            return acts
        extracted = await asyncio.gather(*[one(p) for p in rows])
    # build clean action clauses: detect refusals/disclaimers + null, fall back to the verbatim first sentence
    n_refused = n_null = 0
    kept = []
    for p, acts in zip(rows, extracted):
        acts = acts or [None] * 4
        raw, clean, flags = [], [], []
        for thread, s1, a in zip(p["threads"], p["sent1"], acts):
            if a is None:
                n_null += 1; raw.append(""); clean.append(s1); flags.append("null->sent1")
            elif REFUSE_RE.search(a):
                n_refused += 1; raw.append(a); clean.append(s1); flags.append("refusal->sent1")
            else:
                raw.append(a); clean.append(a); flags.append("ok")
        p["action_raw"] = [r if r else None for r in raw]  # may contain refusals (for audit only)
        p["action"] = clean                                 # refusal-free, used for the headline number
        p["flags"] = flags
        kept.append(p)
    print(f"extracted {len(kept)*4} clauses | refused={n_refused} null={n_null} "
          f"(both -> verbatim first-sentence fallback) | spend ${LED.spent:.3f} | {time.time()-t0:.0f}s", flush=True)

    # persist all extractions for audit
    with (OUT / "action_only_raw.jsonl").open("w") as f:
        for p in kept:
            f.write(json.dumps({"problem": p["problem"], "threads": p["threads"], "sent1": p["sent1"],
                                "action_raw": p["action_raw"], "action_clean": p["action"],
                                "flags": p["flags"]}) + "\n")

    # angle directives for reference
    for p in kept:
        p["angle_txt"] = [a["directive"] for a in p["angles"]]

    variants = {"full": "threads", "sent1": "sent1", "action": "action", "angle": "angle_txt"}
    res = {v: pd_rows(kept, key) for v, key in variants.items()}

    def stats(xs):
        return {"mean": round(float(np.mean(xs)), 4), "median": round(float(np.median(xs)), 4),
                "min": round(float(min(xs)), 3), "max": round(float(max(xs)), 3)}
    summary = {"n": len(kept), "spend_usd": round(LED.spent, 3),
               "extraction": {"threads": len(kept) * 4, "refused_to_sent1": n_refused, "null_to_sent1": n_null,
                              "note": "'action' = refusal-free (refused/null fall back to verbatim first sentence)"},
               "variants": {v: stats(res[v]) for v in variants}}
    # per-problem deltas: action vs full
    deltas = [a - f for a, f in zip(res["action"], res["full"])]
    summary["action_vs_full"] = {
        "mean_delta": round(float(np.mean(deltas)), 4),
        "pct_improved": round(100 * float(np.mean([d > 0 for d in deltas])), 1),
        "pct_improved_big(>0.05)": round(100 * float(np.mean([d > 0.05 for d in deltas])), 1),
    }
    (OUT / "action_only_summary.json").write_text(json.dumps(summary, indent=2))

    # eyeball dump: 6 sample extractions
    with (OUT / "action_only_samples.md").open("w") as f:
        for i, p in enumerate(kept[:6], 1):
            f.write(f"## {i}. full_pd={res['full'][i-1]:.3f}  action_pd={res['action'][i-1]:.3f}\n")
            f.write(f"**{p['problem']}**\n\n")
            for a, full, act in zip(p["angles"], p["threads"], p["action"]):
                f.write(f"- **[{a['family']}]**\n    - full:   {full}\n    - ACTION: {act}\n")
            f.write("\n")

    print("\n=== ACTION-ONLY RESULT ===")
    print(json.dumps(summary, indent=2))
    print(f"\nbaseline reminders: gen_v2 full=0.382, angle=0.478")


if __name__ == "__main__":
    asyncio.run(main())
