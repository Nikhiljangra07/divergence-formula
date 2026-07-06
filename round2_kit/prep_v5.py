"""
prep_v5.py — prep the v5 BIG corpus (8 sources, dual pos/neg) into SFT files + DPO pairs.

Input:  corpus_v5_train/passers.jsonl  (480 gated; angles {family,directive}; pos_threads[4] + neg_threads[4])
Output (--out-dir, default data_v5/):
  decomposer_train.jsonl  — SFT: problem -> facets + 4 angles
  worker_train.jsonl      — SFT: (problem, angle) -> positive thread   (4 per problem)
  dpo_pairs.jsonl         — DPO: (problem, angle) prompt; chosen=pos thread, rejected=neg thread  (4 per problem)
  eval_problems.jsonl     — held-out (default last 20), never in any train file

The DEC contract is byte-identical to prep_v4 (decomposition task unchanged). The WRK contract distills
WORKER_V5_POS (viable + mechanically specific + one downstream consequence) into the eval/train form — the
SAME prompt the future dav_eval_v5 must send, AND the same prompt used to build the DPO pairs (so SFT and DPO
share one worker contract). Corpora kept unmixed: v3/v4 prep untouched.

  python prep_v5.py --corpus ../corpus_run/corpus_v5_train/passers.jsonl --out-dir data_v5 --holdout 20
"""
import argparse, json
from pathlib import Path

DEC_SYS = "You refract a hard decision problem into four categorically distinct strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
            "categorically different KIND of move (a distinct family) leading to a different action — sharp "
            "alternatives, not rephrasings.\nFormat exactly:\n"
            "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
            "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
DEC_ASST = "FACETS: {facets}\nANGLES:\n1) {a1}\n2) {a2}\n3) {a3}\n4) {a4}"

# v5 worker contract — distilled from WORKER_V5_POS (viable, mechanically specific, one consequence).
WRK_SYS = "You write one precise, decisive, realistic reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two or "
            "three sentences, cold and analytical) that COMMITS to THIS angle as a concrete, realistic, VIABLE "
            "strategy that resolves the whole problem in a distinct way — name the actual first move (who does "
            "what, to whom, by when) and the one most likely downstream consequence it is betting on. It must be "
            "lawful, executable, and unmistakably a different KIND of move than the other families would choose.")


def chat(s, u, a):
    return {"messages": [{"role": "system", "content": s}, {"role": "user", "content": u},
                         {"role": "assistant", "content": a}]}


def fam(a):
    if isinstance(a, dict):
        f = str(a.get("family", "")).strip(); d = str(a.get("directive", "")).strip()
        return f"[{f}] {d}" if f else d
    return str(a)


def wellformed(r):
    return len(r.get("angles", [])) == 4 and len(r.get("pos_threads", [])) == 4 \
        and len(r.get("neg_threads", [])) == 4 and r.get("facets") \
        and all(t and len(str(t).split()) >= 10 for t in r["pos_threads"]) \
        and all(t and len(str(t).split()) >= 5 for t in r["neg_threads"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../corpus_run/corpus_v5_train/passers.jsonl")
    ap.add_argument("--out-dir", default="data_v5")
    ap.add_argument("--holdout", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = sum(1 for _ in Path(args.corpus).open())
    rows = [r for r in (json.loads(l) for l in Path(args.corpus).open()) if wellformed(r)]
    held_rows = rows[-args.holdout:]
    held = {r["problem"] for r in held_rows}
    train = [r for r in rows if r["problem"] not in held]
    assert not (held & {r["problem"] for r in train}), "TRAIN/EVAL LEAK"

    with (out / "eval_problems.jsonl").open("w") as f:
        for r in held_rows:
            f.write(json.dumps({"problem": r["problem"], "facets": r["facets"][:3], "angles": r["angles"],
                                "threads": r["pos_threads"], "source": r.get("source"),
                                "setting": r.get("setting"), "judge": r.get("judge")}) + "\n")

    dn = 0
    with (out / "decomposer_train.jsonl").open("w") as f:
        for r in train:
            a = [fam(x) for x in r["angles"]]
            f.write(json.dumps(chat(DEC_SYS, DEC_USER.format(problem=r["problem"]),
                    DEC_ASST.format(facets=" | ".join(r["facets"][:3]), a1=a[0], a2=a[1], a3=a[2], a4=a[3]))) + "\n")
            dn += 1

    wn = 0
    with (out / "worker_train.jsonl").open("w") as f:
        for r in train:
            for a, t in zip(r["angles"], r["pos_threads"]):
                f.write(json.dumps(chat(WRK_SYS, WRK_USER.format(
                    problem=r["problem"], facets=" | ".join(r["facets"][:3]), angle=fam(a)), str(t).strip())) + "\n")
                wn += 1

    # DPO: conversational format — prompt (system+user) with chosen/rejected assistant turns.
    pn = 0
    with (out / "dpo_pairs.jsonl").open("w") as f:
        for r in train:
            for a, pt, nt in zip(r["angles"], r["pos_threads"], r["neg_threads"]):
                u = WRK_USER.format(problem=r["problem"], facets=" | ".join(r["facets"][:3]), angle=fam(a))
                f.write(json.dumps({
                    "prompt": [{"role": "system", "content": WRK_SYS}, {"role": "user", "content": u}],
                    "chosen": [{"role": "assistant", "content": str(pt).strip()}],
                    "rejected": [{"role": "assistant", "content": str(nt).strip()}],
                }) + "\n")
                pn += 1

    print(f"corpus={len(rows)} (dropped {raw-len(rows)} malformed) | train_problems={len(train)} | held_out={len(held)}")
    print(f"decomposer_train={dn} | worker_train={wn} | dpo_pairs={pn} | eval_problems={len(held_rows)} -> {out}/")


if __name__ == "__main__":
    main()
