"""
gen_b_stragglers.py — finish the last few B problems gen_regated_b couldn't.

Targets ONLY the A problems missing from the B corpus, with generous timeouts, more regens, low
concurrency, and VERBOSE per-angle diagnostics (so we learn if a straggler is slow vs a hard DSV4
failure). Appends successes to passers_regated_b.jsonl (resumable, no dupes). Same WORKER_B prompt.
"""
from __future__ import annotations
import asyncio, json, os, time
from pathlib import Path
import httpx
from dotenv import load_dotenv
import config as C, dav

HERE = Path(__file__).resolve().parent; OUT = HERE / "corpus_v2"
SRC, DST = OUT / "passers_regated.jsonl", OUT / "passers_regated_b.jsonl"
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]; OR_URL = "https://openrouter.ai/api/v1/chat/completions"
CAP = float(os.environ.get("BREVIVE_USD_CAP", "3.0"))
TIMEOUT, REGENS, MAXTOK = 120.0, 5, 1600          # generous: longer call timeout + more regens
SEM = asyncio.Semaphore(8)

WORKER_B = (
    "You are writing ONE reasoning thread for a decision dilemma. You see ONLY your assigned angle; you "
    "are blind to the other threads.\n\nPROBLEM: {problem}\nKEY FACETS: {facets}\nYOUR ANGLE [{family}]: {angle}\n\n"
    "Write a single thread (two COMPLETE sentences, 40-60 words, ending in a full stop) that COMMITS to "
    "this angle as a concrete, mechanically specific strategy for the WHOLE problem — name the actual move "
    "(who, what, how). State the move ASSERTIVELY in the first sentence; the second gives at most ONE "
    "compact reason it fits.\nHARD RULE: do NOT state any outcome, consequence, cost, gain, or trade-off, "
    "and do NOT explain the mechanics — DECIDE, do not justify.\nDEPTH: real strategic substance and a "
    "non-obvious insight; no surface restatement.\nVOICE: cold, analytical, decisive. Output ONLY the thread."
)
spent = 0.0
def directive(a): return a["directive"] if isinstance(a, dict) else a
def family(a): return a.get("family", "UNSPECIFIED") if isinstance(a, dict) else "UNSPECIFIED"
def complete(t): return t and len(t.split()) >= C.MIN_WORDS and t.rstrip().endswith(dav.TERMINAL)


async def call(client, prompt):
    global spent
    body = {"model": C.MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK, "temperature": 0.8}
    h = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    try:
        r = await client.post(OR_URL, headers=h, json=body, timeout=TIMEOUT)
        r.raise_for_status(); d = r.json(); u = d.get("usage", {})
        spent += u.get("prompt_tokens", 0)*C.PRICE_IN + u.get("completion_tokens", 0)*C.PRICE_OUT
        return (d["choices"][0]["message"]["content"] or "").strip(), None
    except Exception as e:
        return None, f"{type(e).__name__}"


async def one_thread(client, problem, fs, ang, idx):
    for attempt in range(REGENS + 1):
        async with SEM:
            t, err = await call(client, WORKER_B.format(problem=problem, facets=fs, family=family(ang), angle=directive(ang)))
        if complete(t):
            return t.strip()
        why = err or ("incomplete" if t else "empty")
        print(f"      angle{idx} attempt{attempt+1}: {why}", flush=True)
        await asyncio.sleep(1.0)
    return t.strip() if t else None


async def main():
    A = [json.loads(l) for l in SRC.open()]
    done = {json.loads(l)["problem"] for l in DST.open()} if DST.exists() else set()
    missing = [e for e in A if e["problem"] not in done]
    print(f"stragglers: {len(missing)} missing of {len(A)} | cap ${CAP}\n", flush=True)
    if not missing:
        print("nothing missing — B corpus already complete."); return
    got = 0
    async with httpx.AsyncClient() as client:
        with DST.open("a") as fo:
            for k, e in enumerate(missing, 1):
                if spent >= CAP: print("cap hit, stopping."); break
                print(f"[{k}/{len(missing)}] {e['problem'][:90]}", flush=True)
                fs = "; ".join(e["facets"][:3])
                threads = await asyncio.gather(*[one_thread(client, e["problem"], fs, a, i+1)
                                                 for i, a in enumerate(e["angles"])])
                if all(complete(t) for t in threads):
                    fo.write(json.dumps({"problem": e["problem"], "facets": e["facets"][:3],
                                         "angles": e["angles"], "threads": threads}) + "\n"); fo.flush()
                    got += 1; print(f"    -> OK", flush=True)
                else:
                    fail = [i+1 for i, t in enumerate(threads) if not complete(t)]
                    print(f"    -> still failing angles {fail}", flush=True)
    total = sum(1 for _ in DST.open())
    print(f"\nrecovered {got}/{len(missing)} | B total {total}/{len(A)} | spend ${spent:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
