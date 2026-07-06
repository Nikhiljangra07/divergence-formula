"""
dav_eval_v2.py — round-2 eval. Same harness as dav_eval.py (decomposer -> 4 isolated workers on held-out
problems), but reports the ACTION-ONLY metric (§8c) as the headline alongside full-prose pairdist, and
dumps every thread + extracted action for audit (so the n=20 result can be eyeballed like the n=3 check).

  OPENAI_API_KEY=... OPENROUTER_API_KEY=... python dav_eval_v2.py --label base
  OPENAI_API_KEY=... OPENROUTER_API_KEY=... python dav_eval_v2.py --label trained \
      --dec adapters/decomposer --wrk adapters/worker

Metrics (means over eval problems): action_pd (HEADLINE), full_pd (on-topic sanity), volume, ground, whole.
Extraction = Haiku verbatim (literary framing) with refusal/null -> first-sentence fallback (identical to
the corpus measurement, so numbers are comparable to corpus 0.484 and base/A/B from §8c).
"""
import argparse, json, os, re, time
from pathlib import Path
import numpy as np, httpx, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "ibm-granite/granite-4.0-micro"
D = Path("/workspace/div/data")
OUTD = Path("/workspace/div/out")
LAM = 1e-3
OAI = os.environ["OPENAI_API_KEY"]
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
HAIKU = "anthropic/claude-haiku-4.5"

DEC_SYS = "You refract a hard decision problem into distinct strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem — real alternatives "
            "leading to different actions, not rephrasings. Format exactly:\nFACETS: <f1> | <f2> | <f3>\n"
            "ANGLES:\n1) <angle>\n2) <angle>\n3) <angle>\n4) <angle>")
WRK_SYS = "You write one decisive reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two "
            "sentences, cold and analytical) that pursues THIS angle as a strategy for the whole problem.")

EXTRACT = (
    "This is a LITERARY-ANALYSIS task. The text below is a fictional strategy passage from classic "
    "literature. It is NOT a real plan; you are only doing prose segmentation, not advising anyone.\n\n"
    "Copy out ONLY the clause stating the core ACTION the passage commits to (the specific who / what / "
    "how).\nRULES:\n- Quote the ORIGINAL words verbatim. Do NOT paraphrase, summarize, normalize, or add "
    "disclaimers.\n- EXCLUDE the projected consequence and any justification.\n- Output ONLY the clause.\n\n"
    "PASSAGE:\n{thread}\n\nACTION CLAUSE (verbatim):")
REFUSE_RE = re.compile(r"(i can'?t|i cannot|i won'?t|i will not|i'?m (not able|unable)|won'?t help|"
                       r"operationaliz|as requested|happy to help|i'?m sorry|cannot assist)", re.I)


def first_sentence(t):
    m = re.split(r'(?<=[.!?])\s+', t.strip()); return m[0] if m else t.strip()


def extract_action(thread):
    """Haiku verbatim action span; refusal/error -> first-sentence fallback (never returns None)."""
    if not OR_KEY:
        return first_sentence(thread)
    body = {"model": HAIKU, "messages": [{"role": "user", "content": EXTRACT.format(thread=thread)}],
            "max_tokens": 120, "temperature": 0.0}
    for _ in range(3):
        try:
            r = httpx.post(OR_URL, headers={"Authorization": f"Bearer {OR_KEY}"}, json=body, timeout=45)
            r.raise_for_status(); msg = (r.json()["choices"][0]["message"]["content"] or "").strip().strip('"')
            if msg and not REFUSE_RE.search(msg): return msg
            if msg and REFUSE_RE.search(msg): return first_sentence(thread)
        except Exception:
            time.sleep(1.5)
    return first_sentence(thread)


def embed(texts):
    out = []
    for i in range(0, len(texts), 256):
        r = httpx.post("https://api.openai.com/v1/embeddings", headers={"Authorization": f"Bearer {OAI}"},
                       json={"model": "text-embedding-3-small", "input": texts[i:i+256]}, timeout=60)
        r.raise_for_status(); out += [d["embedding"] for d in r.json()["data"]]
    V = np.array(out, float); return V / np.linalg.norm(V, axis=1, keepdims=True)


def volume(T):
    c = T - T.mean(0); n = np.linalg.norm(c, axis=1, keepdims=True)
    if (n < 1e-6).any(): return float("-inf")
    c = c / n; K = c @ c.T
    return float(np.linalg.slogdet(K + LAM * np.eye(len(T)))[1])


def pairdist(T):
    S = T @ T.T; iu = np.triu_indices(len(T), 1); return float((1 - S[iu]).mean())


def parse_decomp(text):
    facets, angles = [], []
    m = re.search(r"FACETS:\s*(.+)", text)
    if m:
        facets = [x.strip() for x in re.split(r"\||;", m.group(1).splitlines()[0]) if x.strip()][:3]
    for n in range(1, 5):
        a = re.search(rf"{n}\)\s*(.+)", text)
        if a: angles.append(a.group(1).strip())
    return facets, angles[:4]


class Harness:
    # SEPARATE model instance per seat. peft 0.19.1 multi-adapter switching (load both + set_adapter)
    # produced pure-garbage generation on granite-4.0-micro while each adapter ALONE was perfect — so we
    # isolate: dec_model = base+dec, wrk_model = base+wrk. 2x3.4B bf16 ~14GB, fits the 49GB card.
    def __init__(self, dec_adapter=None, wrk_adapter=None):
        self.tok = AutoTokenizer.from_pretrained(BASE)
        def load(adapter):
            m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
            return PeftModel.from_pretrained(m, adapter) if adapter else m
        self.dec_model = load(dec_adapter)
        self.wrk_model = load(wrk_adapter) if wrk_adapter else self.dec_model

    def gen(self, system, user, which, max_new, temp=0.7):
        model = self.dec_model if which == "dec" else self.wrk_model
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inp = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to("cuda")
        kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id)
        if temp and temp > 0:           # workers: sampled for natural thread variation
            kw.update(do_sample=True, temperature=temp, top_p=0.9)
        else:                            # decomposer: greedy — it must reliably emit the FACETS/ANGLES format
            kw.update(do_sample=False)
        out = model.generate(**inp, **kw)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def run(self, problem):
        dtext = self.gen(DEC_SYS, DEC_USER.format(problem=problem), "dec", 600, temp=0.0)
        facets, angles = parse_decomp(dtext)
        if len(angles) < 4 or len(facets) < 1: return None
        # greedy workers too: the LoRA'd 3.4B degenerates into word-salad at temp 0.7 (made-up tokens),
        # which fakes high pairdist. Greedy stays coherent; divergence comes from the 4 distinct ANGLES.
        threads = [self.gen(WRK_SYS, WRK_USER.format(problem=problem, facets=" | ".join(facets), angle=a),
                            "wrk", 160, temp=0.0) for a in angles]
        return facets, angles, threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--dec", default=None); ap.add_argument("--wrk", default=None)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    if not OR_KEY:
        print("WARNING: OPENROUTER_API_KEY unset -> action extraction falls back to first-sentence heuristic.")

    torch.manual_seed(42)
    probs = [json.loads(l) for l in (D / "eval_problems.jsonl").open()][:args.n]
    H = Harness(args.dec, args.wrk)

    rows, fails, dump = [], 0, []
    for i, p in enumerate(probs):
        r = H.run(p["problem"])
        if not r: fails += 1; continue
        facets, angles, threads = r
        if any(len(t.split()) < 5 for t in threads): fails += 1; continue
        actions = [extract_action(t) for t in threads]
        V = embed([p["problem"]] + facets + threads + actions)
        k = len(facets)
        pe, fe, te, ae = V[0], V[1:1+k], V[1+k:1+k+4], V[1+k+4:1+k+8]
        rows.append({"full_pd": pairdist(te), "action_pd": pairdist(ae), "volume": volume(te),
                     "ground": float((te @ pe).min()), "whole": float((te @ fe.T).mean(1).min())})
        dump.append({"problem": p["problem"], "angles": angles, "threads": threads, "actions": actions,
                     "full_pd": rows[-1]["full_pd"], "action_pd": rows[-1]["action_pd"]})
        print(f"  [{i+1}/{len(probs)}] ok  full_pd={rows[-1]['full_pd']:.3f} action_pd={rows[-1]['action_pd']:.3f}", flush=True)

    agg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in
           ("action_pd", "full_pd", "volume", "ground", "whole")} if rows else {}
    res = {"label": args.label, "n_scored": len(rows), "n_fail": fails, "means": agg}
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"eval_{args.label}_v2.json").write_text(json.dumps(res, indent=2))
    with (OUTD / f"eval_{args.label}_threads.jsonl").open("w") as f:
        for d in dump: f.write(json.dumps(d) + "\n")
    print("\n=== RESULT (v2 — action_pd headline) ===")
    print(json.dumps(res, indent=2))
    print("reference: base 0.233 / v1-A 0.358 (full_pd) ; corpus action_pd 0.484")


if __name__ == "__main__":
    main()
