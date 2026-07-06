"""
gen_v2_b.py — paired B-worker pass: reuse v2-A's problems + angles, generate the NO-CONSEQUENCE
threads (v2-fixed worker), then report the clean ablation v2-A vs v2-B (same problems/angles, threads
differ only by consequence) plus v1 baselines.

WORKER_B v2 fix (from the probe diagnosis): B drifted into rationale/justification, hurting grounding
and decisiveness. Now it must DECIDE not justify — assert the move, one compact reason, no outcome,
no mechanics-explaining — while keeping depth.

Run under caffeinate after gen_v2.py finishes. Reads corpus_v2/passers.jsonl.
"""
from __future__ import annotations
import asyncio, json, os, time
from pathlib import Path
import numpy as np, httpx
from dotenv import load_dotenv
import config as C, dav

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_v2"
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]; OAI_KEY = os.environ["OPENAI_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY, CAND_CONCURRENCY = 12, 16
TIMEOUT, CAND_TIMEOUT, RETRIES, THREAD_REGENS, WORKER_MAXTOK = 60.0, 200.0, 4, 2, 1600
HARD_USD_CAP = float(os.environ.get("V2B_USD_CAP", "1.0"))

WORKER_B = (
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; you "
    "are blind to the other threads.\n\n"
    "PROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE [{family}]: {angle}\n\n"
    "Write a single thread (two COMPLETE sentences, 40-60 words, ending in a full stop) that COMMITS to "
    "this angle as a concrete, mechanically specific strategy for the WHOLE problem — name the actual move "
    "(who, what, how). State the move ASSERTIVELY in the first sentence; the second gives at most ONE "
    "compact reason it fits.\n"
    "HARD RULE: do NOT state any outcome, consequence, cost, gain, or trade-off, and do NOT explain the "
    "mechanics or enumerate why-it-works — DECIDE, do not justify.\n"
    "DEPTH: real strategic substance and a non-obvious insight; no surface restatement of the angle.\n"
    "VOICE: cold, analytical, decisive. Output ONLY the thread."
)


class Ledger:
    def __init__(s, cap): s.cap, s.spent, s.calls = cap, 0.0, 0; s.lock = asyncio.Lock()
    async def charge(s, u):
        async with s.lock:
            s.spent += u.get("prompt_tokens", 0) * C.PRICE_IN + u.get("completion_tokens", 0) * C.PRICE_OUT
            s.calls += 1
    def over(s): return s.spent >= s.cap


LED = Ledger(HARD_USD_CAP); SEM = asyncio.Semaphore(CONCURRENCY); CAND_SEM = asyncio.Semaphore(CAND_CONCURRENCY)


async def dsv4(client, prompt):
    if LED.over(): return None
    body = {"model": C.MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": WORKER_MAXTOK, "temperature": 0.75}
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    async with SEM:
        for a in range(RETRIES):
            try:
                r = await client.post(OR_URL, headers=headers, json=body, timeout=TIMEOUT)
                r.raise_for_status(); d = r.json(); await LED.charge(d.get("usage", {}))
                msg = (d["choices"][0]["message"]["content"] or "").strip()
                if msg: return msg
            except Exception:
                pass
            await asyncio.sleep(2 * (a + 1))
    return None


async def thread_b(client, problem, facets_str, ang):
    for _ in range(THREAD_REGENS + 1):
        t = await dsv4(client, WORKER_B.format(problem=problem, facets=facets_str, family=ang["family"], angle=ang["directive"]))
        if t and t.strip(): return t.strip()
    return None


async def gen_b(client, ex):
    fs = "; ".join(ex["facets"][:3])
    threads = await asyncio.gather(*[thread_b(client, ex["problem"], fs, a) for a in ex["angles"]])
    if any(t is None for t in threads): return None
    return {**ex, "threads_b": [t.strip() for t in threads]}


def embed(texts):
    out = []
    for i in range(0, len(texts), 256):
        r = httpx.post("https://api.openai.com/v1/embeddings", headers={"Authorization": f"Bearer {OAI_KEY}"},
                       json={"model": "text-embedding-3-small", "input": texts[i:i+256]}, timeout=60)
        r.raise_for_status(); out += [d["embedding"] for d in r.json()["data"]]
    V = np.array(out, float); return V / np.linalg.norm(V, axis=1, keepdims=True)


def metrics(problem_v, facet_v, thread_v):
    return {"pairdist": dav.pairdist(thread_v), "ground": float((thread_v @ problem_v).min()),
            "whole": float((thread_v @ facet_v.T).mean(1).min()), "volume": dav.volume(thread_v)}


async def main():
    A = [json.loads(l) for l in (OUT / "passers.jsonl").open()]
    print(f"v2-B paired pass over {len(A)} v2-A examples | cap ${HARD_USD_CAP}", flush=True)
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        async def bounded(ex):
            async with CAND_SEM:
                try: return await asyncio.wait_for(gen_b(client, ex), timeout=CAND_TIMEOUT)
                except Exception: return None
        res = await asyncio.gather(*[bounded(e) for e in A])
    paired = [r for r in res if r]
    print(f"generated B-threads for {len(paired)}/{len(A)} in {time.time()-t0:.0f}s | spend ${LED.spent:.2f}", flush=True)

    # embed everything: per example -> problem, 3 facets, 4 A-threads, 4 B-threads, 4 angles
    rows_a, rows_b, rows_ang = [], [], []
    for c in paired:
        V = embed([c["problem"]] + c["facets"][:3] + c["threads"] + c["threads_b"] + [a["directive"] for a in c["angles"]])
        p = V[0]; f = V[1:4]; ta = V[4:8]; tb = V[8:12]; ang = V[12:16]
        rows_a.append(metrics(p, f, ta)); rows_b.append(metrics(p, f, tb))
        rows_ang.append(dav.pairdist(ang))

    def mean(rows, k): return round(float(np.mean([r[k] for r in rows])), 4)
    out = {
        "n_paired": len(paired),
        "angle_pairdist_v2": round(float(np.mean(rows_ang)), 4),
        "v2_A_with_consequence": {k: mean(rows_a, k) for k in ("pairdist", "ground", "whole", "volume")},
        "v2_B_without_consequence": {k: mean(rows_b, k) for k in ("pairdist", "ground", "whole", "volume")},
        "v1_reference": {"A_pairdist": 0.385, "B_pairdist": 0.404, "A_ground": 0.531, "B_ground": 0.500},
        "spend_usd": round(LED.spent, 3),
    }
    (OUT / "ab_compare.json").write_text(json.dumps(out, indent=2))
    # write paired md for eyeball
    with (OUT / "ab_pairs.md").open("w") as fmd:
        for i, c in enumerate(paired[:12], 1):
            fmd.write(f"## {i}. [{c['source']}]\n**{c['problem']}**\n")
            for a, ta, tb in zip(c["angles"], c["threads"], c["threads_b"]):
                fmd.write(f"- [{a['family']}]\n  - A (consequence): {ta}\n  - B (bare): {tb}\n")
            fmd.write("\n")
    print("\n=== V2 A/B COMPARISON ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
