"""
dav_eval_v3.py — v3 eval: base-vs-trained refraction scored by the JUDGE (not action_pd).

Round-2 lesson twice over: pairdist/action_pd reward lexical spread, NOT the product goal. v3's headline
is the 5-dim foresight JUDGE (the same grader that gated the corpus), so train/eval/gate all agree.

Harness is the round-2 one (decomposer -> 4 isolated workers, SEPARATE model per seat, greedy both seats —
peft 0.19.1 multi-adapter switching garbles granite, and temp>0 word-salads the LoRA'd 3.4B). Prompts are
BYTE-IDENTICAL to prep_v3.py (train == eval) and the JUDGE_PROMPT is verbatim from gen_v3.py (scores
comparable to the corpus). full_pd kept only as a diagnostic.

  OPENROUTER_API_KEY=... python dav_eval_v3.py --label base
  OPENROUTER_API_KEY=... python dav_eval_v3.py --label trained --dec adapters/decomposer --wrk adapters/worker
Run base then trained; compare the judge_means blocks (foresight, distinctness, ...).
"""
import argparse, json, os, re, time
from pathlib import Path
import numpy as np, httpx, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "ibm-granite/granite-4.0-micro"
DATA = Path(os.environ.get("V3_DATA", "/workspace/div/data_v3"))
OUTD = Path(os.environ.get("V3_OUT", "/workspace/div/out"))
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OAI = os.environ.get("OPENAI_API_KEY", "")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-haiku-4.5")
DIMS = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight")

# ---- prompts: BYTE-IDENTICAL to prep_v3.py ----
DEC_SYS = "You refract a hard decision problem into four distinct, forward-looking strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
            "categorically different KIND of forward move (a distinct family) leading to a different action — "
            "real alternatives, not rephrasings.\nFormat exactly:\n"
            "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
            "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
WRK_SYS = "You write one decisive, forward-looking reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (three "
            "sentences, cold and analytical) that COMMITS to THIS angle as a concrete strategy for the whole "
            "problem and names the ONE key downstream reaction it is betting on, one or two moves ahead.")

# ---- judge: VERBATIM from gen_v3.py ----
JUDGE_PROMPT = (
    "You are grading ONE training example for a model that must REFRACT a hard decision into FOUR genuinely "
    "distinct, forward-looking strategies. Be a strict critic.\n\n"
    "PROBLEM: {problem}\n\nThe four strategies:\n{threads}\n\n"
    "Score each 1-5 (5=excellent, 1=fails):\n"
    "- admits_multiplicity: does the PROBLEM genuinely admit several viable DIFFERENT forward strategies "
    "(not one obvious answer)?\n"
    "- distinctness: are the four strategies genuinely DIFFERENT approaches, not rephrasings or the same move?\n"
    "- concreteness: does each name a concrete, mechanically specific move (who/what/how), not a vague gesture?\n"
    "- decisiveness: does each COMMIT to a move rather than hedge?\n"
    "- foresight: does each anticipate a realistic downstream reaction a move or two ahead — WITHOUT "
    "hallucinating an unrealistic chain or over-planning? (score LOW for both no-foresight AND fantasy over-reach)\n\n"
    "Return STRICT JSON only:\n"
    '{{"admits_multiplicity":N,"distinctness":N,"concreteness":N,"decisiveness":N,"foresight":N,'
    '"weakest":"<dimension>","note":"<=12 words"}}'
)


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


def or_call(prompt, max_tokens, temp=0.0):
    if not OR_KEY: return None
    body = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    for _ in range(3):
        try:
            r = httpx.post(OR_URL, headers={"Authorization": f"Bearer {OR_KEY}"}, json=body, timeout=60)
            r.raise_for_status(); return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            time.sleep(1.5)
    return None


def fam_of(angle):
    m = re.match(r"\s*\[([^\]]+)\]", angle); return m.group(1).strip() if m else "UNSPECIFIED"


def judge(problem, angles, threads):
    block = "\n".join(f"{i+1}. [{fam_of(a)}] {t}" for i, (a, t) in enumerate(zip(angles, threads)))
    out = or_call(JUDGE_PROMPT.format(problem=problem, threads=block), 400, 0.0)
    j = parse_json(out)
    if not j: return None
    try:
        s = {d: int(j[d]) for d in DIMS}
    except Exception:
        return None
    s["weakest"] = str(j.get("weakest", "")); s["note"] = str(j.get("note", ""))
    s["min"] = min(s[d] for d in DIMS); s["mean"] = round(sum(s[d] for d in DIMS) / 5, 2)
    return s


def embed(texts):
    if not OAI: return None
    out = []
    for i in range(0, len(texts), 256):
        r = httpx.post("https://api.openai.com/v1/embeddings", headers={"Authorization": f"Bearer {OAI}"},
                       json={"model": "text-embedding-3-small", "input": texts[i:i+256]}, timeout=60)
        r.raise_for_status(); out += [d["embedding"] for d in r.json()["data"]]
    V = np.array(out, float); return V / np.linalg.norm(V, axis=1, keepdims=True)


def pairdist(T):
    S = T @ T.T; iu = np.triu_indices(len(T), 1); return float((1 - S[iu]).mean())


def parse_decomp(text):
    facets, angles = [], []
    m = re.search(r"FACETS:\s*(.+)", text)
    if m:
        facets = [x.strip() for x in re.split(r"\||;", m.group(1).splitlines()[0]) if x.strip()][:3]
    for n in range(1, 5):
        a = re.search(rf"{n}\)\s*(.+)", text)
        if a: angles.append(a.group(1).strip())   # keeps the leading [FAMILY] tag -> fed to worker verbatim
    return facets, angles[:4]


class Harness:
    # SEPARATE model instance per seat (peft 0.19.1 multi-adapter switching garbles granite-4.0-micro).
    def __init__(self, dec_adapter=None, wrk_adapter=None):
        self.tok = AutoTokenizer.from_pretrained(BASE)
        def load(adapter):
            m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
            return PeftModel.from_pretrained(m, adapter) if adapter else m
        self.dec_model = load(dec_adapter)
        self.wrk_model = load(wrk_adapter) if wrk_adapter else self.dec_model

    def gen(self, system, user, which, max_new):
        model = self.dec_model if which == "dec" else self.wrk_model
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inp = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to("cuda")
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,  # greedy both seats
                             pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def run(self, problem):
        dtext = self.gen(DEC_SYS, DEC_USER.format(problem=problem), "dec", 600)
        facets, angles = parse_decomp(dtext)
        if len(angles) < 4 or len(facets) < 1: return None
        threads = [self.gen(WRK_SYS, WRK_USER.format(problem=problem, facets=" | ".join(facets), angle=a),
                            "wrk", 220) for a in angles]
        return facets, angles, threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--dec", default=None); ap.add_argument("--wrk", default=None)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    if not OR_KEY:
        print("FATAL: OPENROUTER_API_KEY unset — judge cannot run."); raise SystemExit(1)

    torch.manual_seed(42)
    probs = [json.loads(l) for l in (DATA / "eval_problems.jsonl").open()][:args.n]
    H = Harness(args.dec, args.wrk)

    scored, fails, dump = [], 0, []
    for i, p in enumerate(probs):
        r = H.run(p["problem"])
        if not r:
            fails += 1; print(f"  [{i+1}/{len(probs)}] FAIL (decomp/format)", flush=True); continue
        facets, angles, threads = r
        if any(len(t.split()) < 8 for t in threads):
            fails += 1; print(f"  [{i+1}/{len(probs)}] FAIL (empty thread)", flush=True); continue
        j = judge(p["problem"], angles, threads)
        if not j:
            fails += 1; print(f"  [{i+1}/{len(probs)}] FAIL (judge)", flush=True); continue
        V = embed(threads)
        full_pd = round(pairdist(V), 4) if V is not None else None
        scored.append({"judge": j, "full_pd": full_pd})
        dump.append({"problem": p["problem"], "angles": angles, "threads": threads, "judge": j, "full_pd": full_pd})
        print(f"  [{i+1}/{len(probs)}] mean={j['mean']} fore={j['foresight']} dist={j['distinctness']} "
              f"min={j['min']} full_pd={full_pd}", flush=True)

    jm = {d: round(float(np.mean([s['judge'][d] for s in scored])), 3) for d in DIMS} if scored else {}
    res = {
        "label": args.label, "n_scored": len(scored), "n_fail": fails,
        "judge_means": jm,
        "judge_overall_mean": round(float(np.mean([s['judge']['mean'] for s in scored])), 3) if scored else None,
        "foresight_pass(>=3)": round(100*float(np.mean([s['judge']['foresight'] >= 3 for s in scored])), 1) if scored else None,
        "full_pd_diag": round(float(np.mean([s['full_pd'] for s in scored if s['full_pd'] is not None])), 4)
                        if any(s['full_pd'] is not None for s in scored) else None,
    }
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"eval_{args.label}_v3.json").write_text(json.dumps(res, indent=2))
    with (OUTD / f"eval_{args.label}_v3_threads.jsonl").open("w") as f:
        for d in dump: f.write(json.dumps(d) + "\n")
    print(f"\n=== RESULT v3 (JUDGE headline) — {args.label} ===")
    print(json.dumps(res, indent=2))
    print("corpus reference (gated): foresight 3.30 | distinct 4.84 | decisive 4.87 | overall 4.44")


if __name__ == "__main__":
    main()
