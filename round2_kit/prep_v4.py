"""
prep_v4.py — prep the v4 PRECISION corpus into chat-format SFT files (single arm).

Same mechanics as prep_v3.py; the ONLY differences are the PROMPTS, which match v4's precision design
(drop foresight, sharpen distinctness). These prompts are the CONTRACT — dav_eval_v4.py must send
byte-identical prompts (train == eval). v3 prep/eval are left untouched (corpora kept unmixed).

Input:  corpus_v4_train/passers.jsonl  (201 gated distinct>=4 & concrete>=4; angles {family,directive}; 2-sentence threads)
Output (--out-dir, default data_v4/): decomposer_train / worker_train / eval_problems
Held-out = last 20 (matches v3 -> 181 train problems, 724 worker rows). No train/test leak (asserted).

  python prep_v4.py --corpus ../corpus_run/corpus_v4_train/passers.jsonl --out-dir data_v4 --holdout 20
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

WRK_SYS = "You write one precise, decisive reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two "
            "sentences, cold and analytical) that COMMITS to THIS angle as a concrete, mechanically specific "
            "strategy for the whole problem — a sharp, non-obvious move that is unmistakably a different KIND "
            "of move than the other families would choose.")


def msgs(s, u, a):
    return {"messages": [{"role": "system", "content": s}, {"role": "user", "content": u},
                         {"role": "assistant", "content": a}]}


def fam(a):
    if isinstance(a, dict):
        f = str(a.get("family", "")).strip(); d = str(a.get("directive", "")).strip()
        return f"[{f}] {d}" if f else d
    return str(a)


def wellformed(r):
    return len(r.get("angles", [])) == 4 and len(r.get("threads", [])) == 4 and r.get("facets") \
        and all(t and len(str(t).split()) >= 10 for t in r["threads"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../corpus_run/corpus_v4_train/passers.jsonl")
    ap.add_argument("--out-dir", default="data_v4")
    ap.add_argument("--holdout", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = sum(1 for _ in Path(args.corpus).open())
    rows = [r for r in (json.loads(l) for l in Path(args.corpus).open()) if wellformed(r)]
    held = {r["problem"] for r in rows[-args.holdout:]}
    train = [r for r in rows if r["problem"] not in held]
    assert not (held & {r["problem"] for r in train}), "TRAIN/EVAL LEAK"

    with (out / "eval_problems.jsonl").open("w") as f:
        for r in rows[-args.holdout:]:
            f.write(json.dumps({"problem": r["problem"], "facets": r["facets"][:3], "angles": r["angles"],
                                "threads": r["threads"], "source": r.get("source"), "judge": r.get("judge")}) + "\n")

    dn = 0
    with (out / "decomposer_train.jsonl").open("w") as f:
        for r in train:
            a = [fam(x) for x in r["angles"]]
            f.write(json.dumps(msgs(DEC_SYS, DEC_USER.format(problem=r["problem"]),
                    DEC_ASST.format(facets=" | ".join(r["facets"][:3]), a1=a[0], a2=a[1], a3=a[2], a4=a[3]))) + "\n")
            dn += 1

    wn = 0
    with (out / "worker_train.jsonl").open("w") as f:
        for r in train:
            for a, t in zip(r["angles"], r["threads"]):
                f.write(json.dumps(msgs(WRK_SYS, WRK_USER.format(
                    problem=r["problem"], facets=" | ".join(r["facets"][:3]), angle=fam(a)), str(t).strip())) + "\n")
                wn += 1

    print(f"corpus={len(rows)} (dropped {raw-len(rows)} malformed) | train_problems={len(train)} | held_out={len(held)}")
    print(f"decomposer_train={dn} | worker_train={wn} | eval_problems={len(held)} -> {out}/")


if __name__ == "__main__":
    main()
