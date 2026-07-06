"""
trained_action_check.py — close the loop: does the TRAINED 3.4B model's OWN output (not the DSV4 corpus)
show metric-visible divergence under action-only embedding?

Parses qual.txt (20 held-out problems × {BASE, A, B} × 4 threads) from the pod backup, then for each
label computes full-thread pairdist (sanity: must reproduce the pairdist recorded in qual.txt and the
eval means 0.233/0.308/0.358) vs action-only pairdist (same Haiku verbatim extractor + sent1 fallback
as action_only.py). If A/B rise the way the corpus did, the trained model refracts measurably too.
"""
from __future__ import annotations
import asyncio, json, os, re, sys, time
from pathlib import Path
import numpy as np, httpx
from dotenv import load_dotenv
import config as C, dav

QUAL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/private/tmp/claude-501/-Users-nikhil-Desktop-lora-v1-frontend/"
    "a63325bf-c4b0-41ac-a940-a570409d6407/scratchpad/trained_check/div/out/qual.txt")
OUTDIR = Path(__file__).resolve().parent / "corpus_v2"
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]; OAI_KEY = os.environ["OPENAI_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
HAIKU = "anthropic/claude-haiku-4.5"
HAIKU_IN, HAIKU_OUT = 1.0 / 1e6, 5.0 / 1e6
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
    "- Output ONLY the copied clause, nothing else.\n\n"
    "PASSAGE:\n{thread}\n\nACTION CLAUSE (verbatim):"
)
REFUSE_RE = re.compile(
    r"(i can'?t|i cannot|i won'?t|i will not|i'?m (not able|unable)|i am (not able|unable)|"
    r"won'?t help|operationaliz|as requested|happy to help|i'?m sorry|cannot assist)", re.I)
LABELS = {"BASE": "base", "A (with consequence)": "A", "B (without consequence)": "B"}


def first_sentence(t):
    m = re.split(r'(?<=[.!?])\s+', t.strip())
    return m[0] if m else t.strip()


def parse_qual(text):
    """-> list of {problem_idx, label, reported_pd, threads[<=4]}"""
    # split into problem blocks
    probs = re.split(r'#{30,}\s*\nPROBLEM\s+(\d+)\s*\[[AB]\]:', text)
    records = []
    # probs = [pre, idx1, body1, idx2, body2, ...]
    for i in range(1, len(probs), 2):
        idx = int(probs[i]); body = probs[i + 1]
        # split body into label sections
        parts = re.split(r'====\s*(BASE|A \(with consequence\)|B \(without consequence\))\s*====\s+pairdist=([\d.eE+-]+)', body)
        for j in range(1, len(parts), 3):
            label_raw, pd, block = parts[j], float(parts[j + 1]), parts[j + 2]
            threads = re.findall(r'THREAD:\s*(.*?)(?=\n\s*ANGLE\b|\n=+|\n#{3,}|\nPROBLEM\b|\Z)', block, re.S)
            threads = [re.sub(r'\s+', ' ', t).strip() for t in threads if t.strip()]
            if len(threads) >= 4:
                records.append({"idx": idx, "label": LABELS[label_raw], "reported_pd": pd, "threads": threads[:4]})
    return records


class Ledger:
    def __init__(s, cap): s.cap, s.spent = cap, 0.0; s.lock = asyncio.Lock()
    async def charge(s, u):
        async with s.lock: s.spent += u.get("prompt_tokens", 0) * HAIKU_IN + u.get("completion_tokens", 0) * HAIKU_OUT
    def over(s): return s.spent >= s.cap


LED = Ledger(CAP); SEM = asyncio.Semaphore(CONC)


async def extract(client, thread):
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
            except Exception: pass
            await asyncio.sleep(1.5 * (a + 1))
    return None


def pd_mean(groups, key):
    texts, spans = [], []
    for g in groups:
        spans.append(len(texts)); texts += g[key]
    V = dav.embed(texts, OAI_KEY)
    pds = [dav.pairdist(V[s:s + 4]) for s in spans]
    return float(np.mean(pds)), pds


async def main():
    recs = parse_qual(QUAL.read_text())
    by = {"base": [], "A": [], "B": []}
    for r in recs: by[r["label"]].append(r)
    print(f"parsed {len(recs)} label-blocks: base={len(by['base'])} A={len(by['A'])} B={len(by['B'])}", flush=True)

    for r in recs:
        r["sent1"] = [first_sentence(t) for t in r["threads"]]
    async with httpx.AsyncClient() as client:
        async def one(r):
            return await asyncio.gather(*[extract(client, t) for t in r["threads"]])
        ex = await asyncio.gather(*[one(r) for r in recs])
    n_ref = n_null = 0
    for r, acts in zip(recs, ex):
        acts = acts or [None] * 4; clean = []
        for s1, a in zip(r["sent1"], acts):
            if a is None: n_null += 1; clean.append(s1)
            elif REFUSE_RE.search(a): n_ref += 1; clean.append(s1)
            else: clean.append(a)
        r["action"] = clean
    print(f"extracted | refused={n_ref} null={n_null} -> sent1 fallback | spend ${LED.spent:.3f}", flush=True)

    out = {"n_per_label": {k: len(v) for k, v in by.items()}, "spend_usd": round(LED.spent, 3),
           "refused": n_ref, "null": n_null, "labels": {}}
    for lab in ("base", "A", "B"):
        g = by[lab]
        full_m, full_pds = pd_mean(g, "threads")
        sent_m, _ = pd_mean(g, "sent1")
        act_m, _ = pd_mean(g, "action")
        reported = float(np.mean([r["reported_pd"] for r in g]))
        out["labels"][lab] = {
            "n": len(g),
            "full_recomputed": round(full_m, 4),
            "full_reported_in_qual": round(reported, 4),     # sanity: should ~match full_recomputed
            "sent1": round(sent_m, 4),
            "action": round(act_m, 4),
            "action_minus_full": round(act_m - full_m, 4),
        }
    (OUTDIR / "trained_action_check.json").write_text(json.dumps(out, indent=2))
    print("\n=== TRAINED-MODEL ACTION-ONLY CHECK ===")
    print(json.dumps(out, indent=2))
    print("\nreference — eval_*.json means: base 0.233 / A 0.308 / B 0.358 ; corpus full 0.382 -> action 0.484")


if __name__ == "__main__":
    asyncio.run(main())
