"""
gen_regated_b.py — revive the worker-B (NO-consequence) arm for round 2, PAIRED to the re-gated A corpus.

Reads passers_regated.jsonl (the 201 with-consequence problems) and, reusing the SAME problems + angles,
generates the no-consequence B threads (assert the move + one compact reason, NO outcome). Output is a
clean paired corpus passers_regated_b.jsonl with identical {problem, facets, angles} and B threads.

NO-MIXUP CONTRACT:
  - A corpus = passers_regated.jsonl     (threads = with consequence)   [input, untouched]
  - B corpus = passers_regated_b.jsonl   (threads = without consequence) [output]
  Both share the SAME 201 problems + angles, so the ONLY difference at train time is the worker target.
  The old gen_v2_b ab_compare.json (n=34, sleep-failed) is superseded and must not be reused.

SAFE: save-as-you-go (append+flush per problem) + RESUMABLE (skips problems already in the output) so a
sleep/stall cannot wipe progress (the bug that killed the last B pass). caffeinate + money cap + per-thread
completeness regen. Keep the lid OPEN while it runs (~15 min); caffeinate -i blocks idle sleep, not lid-close.
"""
from __future__ import annotations
import asyncio, json, os, time
from pathlib import Path
import httpx
from dotenv import load_dotenv
import config as C, dav

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_v2"
SRC = OUT / "passers_regated.jsonl"          # A corpus (input)
DST = OUT / "passers_regated_b.jsonl"        # B corpus (output)
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
CAP = float(os.environ.get("BREVIVE_USD_CAP", "1.5"))
CONC, CAND_CONC = 12, 16
TIMEOUT, CAND_TIMEOUT, RETRIES, REGENS, MAXTOK = 60.0, 200.0, 4, 2, 1600

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
    def __init__(s, cap): s.cap, s.spent = cap, 0.0; s.lock = asyncio.Lock()
    async def charge(s, u):
        async with s.lock: s.spent += u.get("prompt_tokens", 0)*C.PRICE_IN + u.get("completion_tokens", 0)*C.PRICE_OUT
    def over(s): return s.spent >= s.cap


LED = Ledger(CAP); SEM = asyncio.Semaphore(CONC); CAND_SEM = asyncio.Semaphore(CAND_CONC)


def directive(a): return a["directive"] if isinstance(a, dict) else a
def family(a): return a.get("family", "UNSPECIFIED") if isinstance(a, dict) else "UNSPECIFIED"
def complete(t): return t and len(t.split()) >= C.MIN_WORDS and t.rstrip().endswith(dav.TERMINAL)


async def dsv4(client, prompt):
    if LED.over(): return None
    body = {"model": C.MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK, "temperature": 0.75}
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    async with SEM:
        for a in range(RETRIES):
            try:
                r = await client.post(OR_URL, headers=headers, json=body, timeout=TIMEOUT)
                r.raise_for_status(); d = r.json(); await LED.charge(d.get("usage", {}))
                msg = (d["choices"][0]["message"]["content"] or "").strip()
                if msg: return msg
            except Exception: pass
            await asyncio.sleep(2 * (a + 1))
    return None


async def thread_b(client, problem, facets_str, ang):
    for _ in range(REGENS + 1):
        t = await dsv4(client, WORKER_B.format(problem=problem, facets=facets_str,
                                               family=family(ang), angle=directive(ang)))
        if complete(t): return t.strip()
    return t.strip() if t else None     # keep best effort; completeness reported in summary


async def gen_b(client, ex):
    fs = "; ".join(ex["facets"][:3])
    threads = await asyncio.gather(*[thread_b(client, ex["problem"], fs, a) for a in ex["angles"]])
    if any(t is None for t in threads): return None
    return {"problem": ex["problem"], "facets": ex["facets"][:3], "angles": ex["angles"], "threads": threads}


async def main():
    src = [json.loads(l) for l in SRC.open()]
    done = {json.loads(l)["problem"] for l in DST.open()} if DST.exists() else set()
    todo = [e for e in src if e["problem"] not in done]
    print(f"B revival (paired to re-gated A): {len(src)} total, {len(done)} already done, {len(todo)} to do | cap ${CAP}", flush=True)
    t0 = time.time()
    lock = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        with DST.open("a") as fo:
            async def bounded(ex):
                async with CAND_SEM:
                    try: r = await asyncio.wait_for(gen_b(client, ex), timeout=CAND_TIMEOUT)
                    except Exception: r = None
                if r:
                    async with lock:
                        fo.write(json.dumps(r) + "\n"); fo.flush()   # save-as-you-go
                return r
            res = await asyncio.gather(*[bounded(e) for e in todo])
    ok = sum(1 for r in res if r)
    # completeness audit over the full output
    allb = [json.loads(l) for l in DST.open()]
    incomplete = sum(1 for e in allb for t in e["threads"] if not complete(t))
    print(f"\ngenerated {ok}/{len(todo)} this run | total B corpus {len(allb)}/{len(src)} | "
          f"incomplete threads {incomplete}/{len(allb)*4} | spend ${LED.spent:.3f} | {time.time()-t0:.0f}s", flush=True)
    if len(allb) < len(src):
        print(f"  NOTE: {len(src)-len(allb)} problems still missing — RE-RUN to resume (skips the done ones).", flush=True)
    else:
        print("  COMPLETE: all 201 problems have paired B threads.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
