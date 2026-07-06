"""
regate.py — adopt the action-only divergence metric (§8c) and RE-GATE the v2 candidate pool with it.

Two phases, separated so the expensive part runs once:
  SCORE (cached -> corpus_v2/regate_scored.jsonl): for all 267 raw candidates, extract the verbatim
    action clause per thread (Haiku, literary framing + refusal/null -> first-sentence fallback), embed
    problem/facets/angles/threads/actions, and record every metric (full_pd, action_pd, angle_pd, ground,
    whole, volume, cons_ok, complete).
  SELECT (cheap, re-runnable at any floor): keep the SAME quality gates (ground/whole/volume/complete/
    consequence) but swap the divergence criterion from full-prose PD_FLOOR=0.30 to an ACTION-space floor.
    Prints a calibration table (how many pass + how many are RECOVERED false-negatives at several floors),
    then writes passers_regated.jsonl (sorted by action_pd) + a fresh summary.

Why this is the right metric for selection but NOT the training target: we still train workers on the
FULL threads (that is what the model must produce); action_pd is only the ruler that decides which
candidate sets are genuinely divergent enough to keep. Run under caffeinate. Hard $ cap.
"""
from __future__ import annotations
import asyncio, json, os, re, time
from pathlib import Path
import numpy as np, httpx
from dotenv import load_dotenv
import config as C, dav

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus_v2"
SCORED = OUT / "regate_scored.jsonl"
load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
OR_KEY = os.environ["OPENROUTER_API_KEY"]; OAI_KEY = os.environ["OPENAI_API_KEY"]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
HAIKU = "anthropic/claude-haiku-4.5"
HAIKU_IN, HAIKU_OUT = 1.0 / 1e6, 5.0 / 1e6
CAP = float(os.environ.get("REGATE_USD_CAP", "1.5"))
ACTION_FLOOR = float(os.environ.get("ACTION_FLOOR", "0.38"))
CONC, TIMEOUT, RETRIES = 12, 45.0, 3

EXTRACT = (
    "This is a LITERARY-ANALYSIS task. The text below is a fictional strategy passage from classic "
    "literature (Machiavelli's The Prince, the Mahabharata, Jin Ping Mei, the xianxia novel Reverend "
    "Insanity). It is NOT a real plan; you are only doing prose segmentation, not advising anyone.\n\n"
    "Copy out ONLY the clause stating the core ACTION the passage commits to (the specific who / what / "
    "how).\n"
    "RULES:\n- Quote the ORIGINAL words verbatim. Do NOT paraphrase, summarize, normalize, editorialize, "
    "or add disclaimers.\n- EXCLUDE the projected consequence and any justification.\n"
    "- Output ONLY the copied clause, nothing else.\n\nPASSAGE:\n{thread}\n\nACTION CLAUSE (verbatim):"
)
REFUSE_RE = re.compile(r"(i can'?t|i cannot|i won'?t|i will not|i'?m (not able|unable)|won'?t help|"
                       r"operationaliz|as requested|happy to help|i'?m sorry|cannot assist)", re.I)


def first_sentence(t):
    m = re.split(r'(?<=[.!?])\s+', t.strip()); return m[0] if m else t.strip()


class Ledger:
    def __init__(s, cap): s.cap, s.spent = cap, 0.0; s.lock = asyncio.Lock()
    async def charge(s, u):
        async with s.lock: s.spent += u.get("prompt_tokens", 0)*HAIKU_IN + u.get("completion_tokens", 0)*HAIKU_OUT
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


async def score_phase():
    cands = [json.loads(l) for l in (OUT / "candidates_raw.jsonl").open()]
    print(f"SCORE: {len(cands)} candidates ({len(cands)*4} threads) | cap ${CAP}", flush=True)
    t0 = time.time()
    # action extraction with refusal/null -> first-sentence fallback
    async with httpx.AsyncClient() as client:
        async def one(c):
            return await asyncio.gather(*[extract(client, t) for t in c["threads"]])
        ex = await asyncio.gather(*[one(c) for c in cands])
    n_ref = n_null = 0
    for c, acts in zip(cands, ex):
        acts = acts or [None]*4; clean = []
        for s1, a in zip([first_sentence(t) for t in c["threads"]], acts):
            if a is None: n_null += 1; clean.append(s1)
            elif REFUSE_RE.search(a): n_ref += 1; clean.append(s1)
            else: clean.append(a)
        c["actions"] = clean
    print(f"  extracted | refused={n_ref} null={n_null} -> sent1 fallback | ${LED.spent:.3f} | {time.time()-t0:.0f}s", flush=True)

    # embed + metrics
    for c in cands:
        V = dav.embed([c["problem"]] + c["facets"][:3] + [a["directive"] for a in c["angles"]]
                      + c["threads"] + c["actions"], OAI_KEY)
        p, f, ang, th, act = V[0], V[1:4], V[4:8], V[8:12], V[12:16]
        c["m"] = {
            "full_pd": dav.pairdist(th), "action_pd": dav.pairdist(act), "angle_pd": dav.pairdist(ang),
            "ground": float((th @ p).min()), "whole": float((th @ f.T).mean(1).min()), "volume": dav.volume(th),
        }
        c["cons_ok"] = dav.consequence_ok(c["threads"]) and all(t.rstrip().endswith(dav.TERMINAL) for t in c["threads"])
    with SCORED.open("w") as fo:
        for c in cands: fo.write(json.dumps(c) + "\n")
    print(f"  cached -> {SCORED.name}", flush=True)
    return cands


def quality_ok(c):
    """the on-topic / completeness gates — unchanged from gen_v2 (divergence handled separately)."""
    m = c["m"]
    return (m["volume"] > C.VOL_GATE and m["ground"] >= C.EPS_G and m["whole"] >= C.EPS_W and c["cons_ok"])


def select_phase(cands):
    qual = [c for c in cands if quality_ok(c)]
    old_pass = [c for c in qual if c["m"]["full_pd"] >= C.PD_FLOOR]      # what the OLD gate kept
    old_keys = {c["problem"] for c in old_pass}
    print(f"\nSELECT: {len(cands)} scored | quality-gate pass {len(qual)} | OLD full-prose gate pass {len(old_pass)}")
    print(f"\ncalibration — ACTION floor sweep (among the {len(qual)} quality-passers):")
    print(f"  {'floor':>6} {'pass':>5} {'recovered*':>11} {'dropped**':>10}")
    for fl in (0.30, 0.34, 0.36, 0.38, 0.40, 0.42):
        keep = [c for c in qual if c["m"]["action_pd"] >= fl]
        recovered = sum(c["problem"] not in old_keys for c in keep)   # passed action, failed old full-prose
        dropped = sum(c["problem"] in old_keys for c in old_pass if c["m"]["action_pd"] < fl)  # old-pass now cut
        print(f"  {fl:>6.2f} {len(keep):>5} {recovered:>11} {dropped:>10}")
    print("  *recovered = action-divergent sets the OLD full-prose gate wrongly rejected")
    print("  **dropped  = old-gate passers whose ACTIONS are near-duplicate (correctly cut)")

    passers = sorted([c for c in qual if c["m"]["action_pd"] >= ACTION_FLOOR],
                     key=lambda c: c["m"]["action_pd"], reverse=True)
    with (OUT / "passers_regated.jsonl").open("w") as fo:
        for c in passers:
            fo.write(json.dumps({"problem": c["problem"], "facets": c["facets"][:3], "angles": c["angles"],
                                 "threads": c["threads"]}) + "\n")
    mean = lambda k: round(float(np.mean([c["m"][k] for c in passers])), 4)
    recovered_n = sum(c["problem"] not in old_keys for c in passers)
    summary = {
        "action_floor": ACTION_FLOOR, "quality_passers": len(qual), "old_full_prose_passers": len(old_pass),
        "regated_passers": len(passers), "recovered_false_negatives": recovered_n,
        "means_regated": {k: mean(k) for k in ("action_pd", "full_pd", "angle_pd", "ground", "whole", "volume")},
        "note": "train workers on FULL threads; action_pd is the selection ruler only.",
    }
    (OUT / "regate_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== REGATE SUMMARY (floor=%.2f) ===" % ACTION_FLOOR)
    print(json.dumps(summary, indent=2))


async def main():
    if SCORED.exists():
        print(f"using cached {SCORED.name} (delete to re-score)")
        cands = [json.loads(l) for l in SCORED.open()]
    else:
        cands = await score_phase()
    select_phase(cands)


if __name__ == "__main__":
    asyncio.run(main())
