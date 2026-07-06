"""
prep_v2.py — round-2 prep from the RE-GATED paired corpora. THREE adapters (parity with round 1):
decomposer (shared) + worker_a (with consequence) + worker_b (without consequence).

NO-MIXUP CONTRACT — the two corpora share the SAME problems + angles; only the worker target differs:
  --corpus-a  passers_regated.jsonl    (threads = WITH consequence)    -> worker_a_train
  --corpus-b  passers_regated_b.jsonl  (threads = WITHOUT consequence) -> worker_b_train
  decomposer_train is built from the A corpus angles (angles are pre-consequence, shared by both arms).
  Held-out eval = the SAME last N problems for both arms (matched by problem text) -> no train/test leak,
  and A/B are evaluated on identical problems. If --corpus-b is omitted, builds A-only (2 adapters).

PROMPTS COPIED VERBATIM FROM dav_eval_v2.py — train on exactly the prompts you eval with.

  python prep_v2.py --corpus-a data/passers_regated.jsonl --corpus-b data/passers_regated_b.jsonl \
                    --out-dir data --holdout 20
Outputs: decomposer_train.jsonl, worker_a_train.jsonl, worker_b_train.jsonl, eval_problems.jsonl
"""
import argparse, json
from pathlib import Path

DEC_SYS = "You refract a hard decision problem into distinct strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem — real alternatives "
            "leading to different actions, not rephrasings. Format exactly:\nFACETS: <f1> | <f2> | <f3>\n"
            "ANGLES:\n1) <angle>\n2) <angle>\n3) <angle>\n4) <angle>")
DEC_ASST = "FACETS: {facets}\nANGLES:\n1) {a1}\n2) {a2}\n3) {a3}\n4) {a4}"
WRK_SYS = "You write one decisive reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two "
            "sentences, cold and analytical) that pursues THIS angle as a strategy for the whole problem.")


def msgs(s, u, a): return {"messages": [{"role": "system", "content": s}, {"role": "user", "content": u},
                                        {"role": "assistant", "content": a}]}
def directive(a): return a["directive"] if isinstance(a, dict) else a
def wellformed(r): return len(r.get("angles", [])) == 4 and len(r.get("threads", [])) == 4 and r.get("facets")


def write_worker(rows, held, path):
    n = 0
    with open(path, "w") as f:
        for r in rows:
            if r["problem"] in held: continue
            for a, t in zip(r["angles"], r["threads"]):
                f.write(json.dumps(msgs(WRK_SYS, WRK_USER.format(problem=r["problem"],
                        facets=" | ".join(r["facets"][:3]), angle=directive(a)), t)) + "\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-a", default="data/passers_regated.jsonl")
    ap.add_argument("--corpus-b", default=None)
    ap.add_argument("--out-dir", default="data"); ap.add_argument("--holdout", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    A = [r for r in (json.loads(l) for l in Path(args.corpus_a).open()) if wellformed(r)]
    held = {r["problem"] for r in A[-args.holdout:]}          # last N problems of the A corpus
    a_train = [r for r in A if r["problem"] not in held]

    # eval set (shared problems; gold A threads kept for reference; eval generates fresh)
    with (out / "eval_problems.jsonl").open("w") as f:
        for r in A[-args.holdout:]:
            f.write(json.dumps({"problem": r["problem"], "facets": r["facets"][:3],
                                "angles": [directive(a) for a in r["angles"]], "threads": r["threads"]}) + "\n")

    # decomposer (shared; from A angles)
    dn = 0
    with (out / "decomposer_train.jsonl").open("w") as f:
        for r in a_train:
            a = [directive(x) for x in r["angles"]]
            f.write(json.dumps(msgs(DEC_SYS, DEC_USER.format(problem=r["problem"]),
                    DEC_ASST.format(facets=" | ".join(r["facets"][:3]), a1=a[0], a2=a[1], a3=a[2], a4=a[3]))) + "\n")
            dn += 1

    wa = write_worker(A, held, out / "worker_a_train.jsonl")
    print(f"corpus_a={len(A)} train_problems={len(a_train)} held_out={len(held)}")
    print(f"decomposer_train={dn} | worker_a_train={wa} | eval_problems={len(held)}")

    if args.corpus_b:
        B = [r for r in (json.loads(l) for l in Path(args.corpus_b).open()) if wellformed(r)]
        a_probs = {r["problem"] for r in A}
        paired = [r for r in B if r["problem"] in a_probs]    # strict pairing guard
        if len(paired) != len(B):
            print(f"  WARNING: {len(B)-len(paired)} B rows have no A pair (dropped — corpus mixup guard)")
        wb = write_worker(paired, held, out / "worker_b_train.jsonl")
        print(f"corpus_b={len(B)} paired={len(paired)} | worker_b_train={wb} (held-out matched to A)")


if __name__ == "__main__":
    main()
