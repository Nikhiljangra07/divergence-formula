"""
dav_eval_v5.py — v5 eval: base-vs-trained refraction scored by the 6-dim JUDGE (adds viability).

Same harness as v4 (decomposer -> 4 isolated workers, SEPARATE model per seat, greedy both seats —
peft 0.19.1 multi-adapter switching garbles granite-4.0-micro, temp>0 word-salads the 3.4B). Prompts are
BYTE-IDENTICAL to prep_v5.py (train == eval) and JUDGE_V5 is VERBATIM from gen_v5.py (scores comparable to
the corpus gate). Judge routes DIRECT to Anthropic when ANTHROPIC_API_KEY is present (own key, no OpenRouter
fee — DIRECT_ANTHROPIC=0 forces OpenRouter).

  ANTHROPIC_API_KEY=... python dav_eval_v5.py --label base
  ANTHROPIC_API_KEY=... python dav_eval_v5.py --label sft   --dec adapters/decomposer_v5 --wrk adapters/worker_v5
  ANTHROPIC_API_KEY=... python dav_eval_v5.py --label dpo   --dec adapters/decomposer_v5 --wrk adapters/worker_v5_dpo
Compare judge_means (viability, distinctness, ...) across base / sft / dpo.
"""
import argparse, json, os, re, time
from pathlib import Path
import numpy as np, httpx, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = os.environ.get("LORA_BASE", "ibm-granite/granite-4.0-micro")  # set to granite-4.0-h-small for the H-Small runs
DATA = Path(os.environ.get("V5_DATA", "/workspace/div/data_v5"))
OUTD = Path(os.environ.get("V5_OUT", "/workspace/div/out"))
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OAI = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-haiku-4.5")
ANTHROPIC_DIRECT = bool(ANTHROPIC_KEY) and os.environ.get("DIRECT_ANTHROPIC", "1") == "1"
ANTHROPIC_ID = {"anthropic/claude-haiku-4.5": "claude-haiku-4-5"}
DIMS = ("admits_multiplicity", "distinctness", "concreteness", "decisiveness", "foresight", "viability")

# ---- prompts: BYTE-IDENTICAL to prep_v5.py ----
DEC_SYS = "You refract a hard decision problem into four categorically distinct strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
            "categorically different KIND of move (a distinct family) leading to a different action — sharp "
            "alternatives, not rephrasings.\nFormat exactly:\n"
            "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
            "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
WRK_SYS = "You write one precise, decisive, realistic reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two or "
            "three sentences, cold and analytical) that COMMITS to THIS angle as a concrete, realistic, VIABLE "
            "strategy that resolves the whole problem in a distinct way — name the actual first move (who does "
            "what, to whom, by when) and the one most likely downstream consequence it is betting on. It must be "
            "lawful, executable, and unmistakably a different KIND of move than the other families would choose.")

# ---- judge: VERBATIM from gen_v5.py (JUDGE_V5, 6-dim) ----
JUDGE_PROMPT = (
    "You are grading ONE training example for a model that must REFRACT a hard decision into FOUR genuinely "
    "distinct, viable strategies. Be a strict critic.\n\n"
    "PROBLEM: {problem}\n\nThe four strategies:\n{threads}\n\n"
    "Score each 1-5 (5=excellent, 1=fails):\n"
    "- admits_multiplicity: does the PROBLEM genuinely admit several viable DIFFERENT strategies?\n"
    "- distinctness: are the four genuinely DIFFERENT approaches, not rephrasings or the same move?\n"
    "- concreteness: does each name a concrete, mechanically specific move (who/what/how), cleanly (not baroque)?\n"
    "- decisiveness: does each COMMIT to a move and actually RESOLVE the decision rather than hedge?\n"
    "- foresight: does each name a realistic downstream consequence a step ahead — WITHOUT fantasy over-reach?\n"
    "- viability: are the four REALISTIC, LAWFUL, and actually executable in the real world (NOT fantasy, NOT "
    "illegal, NOT convoluted obfuscation)? 1=fantasy/illegal/incoherent, 5=fully realistic and viable.\n\n"
    "Return STRICT JSON only:\n"
    '{{"admits_multiplicity":N,"distinctness":N,"concreteness":N,"decisiveness":N,"foresight":N,"viability":N,'
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


def judge_call(prompt, max_tokens, temp=0.0):
    """Judge via Anthropic native API (own key) when available, else OpenRouter."""
    direct = ANTHROPIC_DIRECT and JUDGE_MODEL.startswith("anthropic/")
    for _ in range(3):
        try:
            if direct:
                r = httpx.post("https://api.anthropic.com/v1/messages",
                               headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                                        "content-type": "application/json"},
                               json={"model": ANTHROPIC_ID.get(JUDGE_MODEL, JUDGE_MODEL.split("/", 1)[-1]),
                                     "max_tokens": max_tokens, "temperature": temp,
                                     "messages": [{"role": "user", "content": prompt}]}, timeout=60)
                r.raise_for_status()
                return "".join(p.get("text", "") for p in r.json().get("content", []) if p.get("type") == "text").strip()
            if not OR_KEY: return None
            r = httpx.post(OR_URL, headers={"Authorization": f"Bearer {OR_KEY}"},
                           json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}],
                                 "max_tokens": max_tokens, "temperature": temp}, timeout=60)
            r.raise_for_status(); return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            time.sleep(1.5)
    return None


def fam_of(angle):
    m = re.match(r"\s*\[([^\]]+)\]", angle); return m.group(1).strip() if m else "UNSPECIFIED"


def judge(problem, angles, threads):
    block = "\n".join(f"{i+1}. [{fam_of(a)}] {t}" for i, (a, t) in enumerate(zip(angles, threads)))
    out = judge_call(JUDGE_PROMPT.format(problem=problem, threads=block), 450, 0.0)
    j = parse_json(out)
    if not j: return None
    try:
        s = {d: int(j[d]) for d in DIMS}
    except Exception:
        return None
    s["weakest"] = str(j.get("weakest", "")); s["note"] = str(j.get("note", ""))
    s["min"] = min(s[d] for d in DIMS); s["mean"] = round(sum(s[d] for d in DIMS) / 6, 2)
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
        if a: angles.append(a.group(1).strip())   # keep leading [FAMILY] tag -> fed to worker verbatim
    return facets, angles[:4]


SEQ = os.environ.get("V5_SEQ", "0") == "1"  # sequential seats: REQUIRED for H-Small (2x64GB > 80GB VRAM)


def load_model(adapter=None):
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    return PeftModel.from_pretrained(m, adapter) if adapter else m


def gen_once(model, tok, system, user, max_new):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to("cuda")
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_all_sequential(probs, dec_adapter, wrk_adapter):
    """One seat in VRAM at a time: decompose ALL problems, free, then thread ALL. Same prompts, greedy."""
    import gc as _gc
    tok = AutoTokenizer.from_pretrained(BASE)
    model = load_model(dec_adapter)
    decomps = []
    for i, p in enumerate(probs):
        decomps.append(parse_decomp(gen_once(model, tok, DEC_SYS, DEC_USER.format(problem=p["problem"]), 600)))
        print(f"  [dec {i+1}/{len(probs)}]", flush=True)
    del model; _gc.collect(); torch.cuda.empty_cache()
    model = load_model(wrk_adapter)
    out = []
    for i, (p, (facets, angles)) in enumerate(zip(probs, decomps)):
        if len(angles) < 4 or len(facets) < 1:
            out.append(None); print(f"  [wrk {i+1}/{len(probs)}] SKIP (bad decomp)", flush=True); continue
        threads = [gen_once(model, tok, WRK_SYS, WRK_USER.format(
            problem=p["problem"], facets=" | ".join(facets), angle=a), 256) for a in angles]
        out.append((facets, angles, threads))
        print(f"  [wrk {i+1}/{len(probs)}]", flush=True)
    del model; _gc.collect(); torch.cuda.empty_cache()
    return out


class Harness:
    # SEPARATE model instance per seat (peft 0.19.1 multi-adapter switching garbles granite-4.0-micro).
    def __init__(self, dec_adapter=None, wrk_adapter=None):
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.dec_model = load_model(dec_adapter)
        self.wrk_model = load_model(wrk_adapter) if wrk_adapter else self.dec_model

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
                            "wrk", 256) for a in angles]  # v5 threads run 2-3 sentences -> a touch more room than v4
        return facets, angles, threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--dec", default=None); ap.add_argument("--wrk", default=None)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    if not (ANTHROPIC_DIRECT or OR_KEY):
        print("FATAL: no judge key (set ANTHROPIC_API_KEY or OPENROUTER_API_KEY)."); raise SystemExit(1)

    torch.manual_seed(42)
    probs = [json.loads(l) for l in (DATA / "eval_problems.jsonl").open()][:args.n]
    print(f"judge={JUDGE_MODEL} | route={'anthropic-direct' if (ANTHROPIC_DIRECT and JUDGE_MODEL.startswith('anthropic/')) else 'openrouter'}"
          f" | base={BASE} | seq={SEQ}", flush=True)
    if SEQ and args.dec and args.wrk:
        runs = run_all_sequential(probs, args.dec, args.wrk)
    else:
        H = Harness(args.dec, args.wrk)
        runs = [H.run(p["problem"]) for p in probs]

    scored, fails, dump = [], 0, []
    for i, (p, r) in enumerate(zip(probs, runs)):
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
        print(f"  [{i+1}/{len(probs)}] mean={j['mean']} viab={j['viability']} dist={j['distinctness']} "
              f"conc={j['concreteness']} min={j['min']}", flush=True)

    jm = {d: round(float(np.mean([s['judge'][d] for s in scored])), 3) for d in DIMS} if scored else {}
    res = {
        "label": args.label, "n_scored": len(scored), "n_fail": fails, "judge_means": jm,
        "judge_overall_mean": round(float(np.mean([s['judge']['mean'] for s in scored])), 3) if scored else None,
        "viability_pass(>=3)": round(100*float(np.mean([s['judge']['viability'] >= 3 for s in scored])), 1) if scored else None,
        "distinct_pass(>=4)": round(100*float(np.mean([s['judge']['distinctness'] >= 4 for s in scored])), 1) if scored else None,
        "full_pd_diag": round(float(np.mean([s['full_pd'] for s in scored if s['full_pd'] is not None])), 4)
                        if any(s['full_pd'] is not None for s in scored) else None,
    }
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"eval_{args.label}_v5.json").write_text(json.dumps(res, indent=2))
    with (OUTD / f"eval_{args.label}_v5_threads.jsonl").open("w") as f:
        for d in dump: f.write(json.dumps(d) + "\n")
    print(f"\n=== RESULT v5 (6-dim JUDGE) — {args.label} ===")
    print(json.dumps(res, indent=2))
    print("corpus reference (v5 gated, Haiku): distinct 4.64 | concrete 4.04 | decisive 4.83 | viability 3.38 | foresight 3.75")


if __name__ == "__main__":
    main()
