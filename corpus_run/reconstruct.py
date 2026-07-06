"""
reconstruct.py — recover the refractor's ANGLES from the threads, so the corpus can train the
harness seats (decomposer + worker), not just end-to-end.

Each passing example was generated as: refractor -> 4 distinct angles -> 4 isolated workers -> 4 threads.
The angles were used transiently and not saved. This reads {problem, facets, threads} and, with one
cheap LLM call per example, labels the strategic angle each thread pursues (in thread order), then emits:

  <set>/decomposer.jsonl   {problem, facets, angles}                  -> trains the refractor (crown jewel)
  <set>/worker.jsonl       {problem, facets, angle, thread} x4        -> trains the worker
  <set>/passers_angled.jsonl  {problem, facets, angles, threads}      -> full record

Isolated + safe: reads only OPENROUTER_API_KEY; writes only under corpus/. Ledger cap; save-as-you-go.
Run: python reconstruct.py [set_a|set_b|both]   (default both)
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys
from pathlib import Path
import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
CORP = HERE / "corpus"
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("ANGLE_MODEL", "anthropic/claude-haiku-4.5")  # Haiku 4.5 — fast, cheap extraction
CONCURRENCY = 10
TIMEOUT = 60.0
RETRIES = 4
HARD_USD_CAP = float(os.environ.get("ANGLE_USD_CAP", "2.0"))
PRICE_IN, PRICE_OUT = 0.80 / 1e6, 4.0 / 1e6  # claude-3.5-haiku approx; ledger is the real guard

SETS = {"set_a": "set_a_with_consequence", "set_b": "set_b_without_consequence"}

PROMPT = (
    "Below is a decision PROBLEM and FOUR reasoning threads. Each thread was written by pursuing one "
    "distinct strategic ANGLE on the WHOLE problem. Recover the angle each thread pursues.\n\n"
    "PROBLEM: {problem}\n\nTHREADS:\n{threads}\n\n"
    "For each thread, in order, state its ANGLE as a short imperative directive (5-12 words) naming the "
    "strategic approach that would have seeded it — distinct from the others, no outcome language, just "
    "the lens/move. Return STRICT JSON only:\n"
    '{{"angles": ["<angle for thread 1>","<thread 2>","<thread 3>","<thread 4>"]}}'
)


class Ledger:
    def __init__(self, cap): self.cap, self.spent, self.calls = cap, 0.0, 0; self.lock = asyncio.Lock()
    async def charge(self, u):
        async with self.lock:
            self.spent += u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT
            self.calls += 1
    def over(self): return self.spent >= self.cap


LED = Ledger(HARD_USD_CAP); SEM = asyncio.Semaphore(CONCURRENCY)


async def call(client, prompt):
    if LED.over(): return None
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0.2}
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    async with SEM:
        for a in range(RETRIES):
            try:
                r = await client.post(OR_URL, headers=headers, json=body, timeout=TIMEOUT)
                r.raise_for_status(); d = r.json()
                await LED.charge(d.get("usage", {}))
                msg = (d["choices"][0]["message"]["content"] or "").strip()
                if msg: return msg
            except Exception as e:
                if a == RETRIES - 1: print(f"  [angle] fail: {type(e).__name__}: {e}", flush=True)
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


async def angle_for(client, ex):
    threads = "\n".join(f"{i+1}. {t}" for i, t in enumerate(ex["threads"]))
    out = parse_json(await call(client, PROMPT.format(problem=ex["problem"], threads=threads)))
    if not out or "angles" not in out or len(out["angles"]) != len(ex["threads"]):
        return None
    return [a.strip() for a in out["angles"]]


async def run_set(key, limit=None):
    sdir = CORP / SETS[key]
    exs = [json.loads(l) for l in (sdir / "passers.jsonl").open()]
    if limit: exs = exs[:limit]
    print(f"=== {key}: reconstructing angles for {len(exs)} examples (model={MODEL}) ===", flush=True)
    angled_p = (sdir / "passers_angled.jsonl").open("w")
    dec_p = (sdir / "decomposer.jsonl").open("w")
    wrk_p = (sdir / "worker.jsonl").open("w")
    done = fail = 0
    async with httpx.AsyncClient() as client:
        async def one(ex):
            nonlocal done, fail
            angles = await angle_for(client, ex)
            if not angles:
                fail += 1; return
            rec = {"problem": ex["problem"], "facets": ex["facets"], "angles": angles, "threads": ex["threads"]}
            angled_p.write(json.dumps(rec) + "\n"); angled_p.flush()
            dec_p.write(json.dumps({"problem": ex["problem"], "facets": ex["facets"], "angles": angles}) + "\n")
            for ang, th in zip(angles, ex["threads"]):
                wrk_p.write(json.dumps({"problem": ex["problem"], "facets": ex["facets"],
                                        "angle": ang, "thread": th}) + "\n")
            done += 1
        await asyncio.gather(*[one(e) for e in exs])
    for f in (angled_p, dec_p, wrk_p): f.close()
    print(f"  done {done} | fail {fail} | calls {LED.calls} | spend ${LED.spent:.3f}", flush=True)
    print(f"  -> {sdir/'decomposer.jsonl'} ({done}) , {sdir/'worker.jsonl'} ({done*4} rows)")
    return done, fail


async def smoke(key, n):
    sdir = CORP / SETS[key]
    exs = [json.loads(l) for l in (sdir / "passers.jsonl").open()][:n]
    async with httpx.AsyncClient() as client:
        for ex in exs:
            angles = await angle_for(client, ex)
            print("PROBLEM:", ex["problem"][:120])
            if not angles: print("  -> FAILED"); continue
            for a, t in zip(angles, ex["threads"]):
                print(f"  ANGLE: {a}")
                print(f"    thread: {t[:130]}")
            print()
    print(f"smoke spend ${LED.spent:.4f} calls {LED.calls}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="both", choices=["set_a", "set_b", "both", "smoke"])
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "smoke":
        asyncio.run(smoke("set_a", args.n))
    elif args.mode == "both":
        async def go():
            a = await run_set("set_a"); b = await run_set("set_b")
            print(f"\nTOTAL angled: A={a[0]} B={b[0]} | spend ${LED.spent:.3f}")
        asyncio.run(go())
    else:
        asyncio.run(run_set(args.mode))


if __name__ == "__main__":
    main()
