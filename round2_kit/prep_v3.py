"""
prep_v3.py — prep the v3 FORESIGHT corpus (single arm) into chat-format SFT files for LoRA.

Input:  corpus_v3_train/passers.jsonl   (201 gated, balanced; angles carry {family, directive}; threads = 3-sentence foresight)
Output (--out-dir, default data_v3/):
  decomposer_train.jsonl   problem            -> FACETS + 4 family-tagged ANGLES
  worker_train.jsonl       (problem,facets,angle) -> foresight thread   (~4x rows)
  eval_problems.jsonl      held-out last N problems (gold kept for reference; eval regenerates fresh)

SINGLE ARM: v3 has no no-consequence "B" control — foresight IS consequence-projection, so the B arm is
moot. (Round 1/2 had A/B; v3 deliberately drops it.)

THE PROMPTS BELOW ARE THE CONTRACT. dav_eval_v3.py MUST send byte-identical prompts — you train on exactly
what you eval with (round-2 lesson). They are v3-shaped: three-sentence foresight threads + family-tagged
angles, NOT round-2's two-sentence / family-less format.

Held-out = last N problems (no train/test leak). N=20 matches round-2 -> 181 train problems, 724 worker rows.

  python prep_v3.py --corpus ../corpus_run/corpus_v3_train/passers.jsonl --out-dir data_v3 --holdout 20
"""
import argparse, json
from pathlib import Path

DEC_SYS = "You refract a hard decision problem into four distinct, forward-looking strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
            "categorically different KIND of forward move (a distinct family) leading to a different action — "
            "real alternatives, not rephrasings.\nFormat exactly:\n"
            "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
            "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
DEC_ASST = "FACETS: {facets}\nANGLES:\n1) {a1}\n2) {a2}\n3) {a3}\n4) {a4}"

WRK_SYS = "You write one decisive, forward-looking reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (three "
            "sentences, cold and analytical) that COMMITS to THIS angle as a concrete strategy for the whole "
            "problem and names the ONE key downstream reaction it is betting on, one or two moves ahead.")


def msgs(s, u, a):
    return {"messages": [{"role": "system", "content": s}, {"role": "user", "content": u},
                         {"role": "assistant", "content": a}]}


def fam(a):  # "[CONFRONT/ELIMINATE] <directive>" — family preserved end-to-end (decomposer out == worker in)
    if isinstance(a, dict):
        f = str(a.get("family", "")).strip()
        d = str(a.get("directive", "")).strip()
        return f"[{f}] {d}" if f else d
    return str(a)


def wellformed(r):
    return len(r.get("angles", [])) == 4 and len(r.get("threads", [])) == 4 and r.get("facets") \
        and all(t and len(str(t).split()) >= 10 for t in r["threads"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../corpus_run/corpus_v3_train/passers.jsonl")
    ap.add_argument("--out-dir", default="data_v3")
    ap.add_argument("--holdout", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in (json.loads(l) for l in Path(args.corpus).open()) if wellformed(r)]
    n_drop = 0
    raw = sum(1 for _ in Path(args.corpus).open())
    n_drop = raw - len(rows)
    held = {r["problem"] for r in rows[-args.holdout:]}          # last N problems held out
    train = [r for r in rows if r["problem"] not in held]

    # leak guard: no held-out problem may appear in train
    assert not (held & {r["problem"] for r in train}), "TRAIN/EVAL LEAK"

    # eval set (held-out; gold kept for reference, eval regenerates fresh)
    with (out / "eval_problems.jsonl").open("w") as f:
        for r in rows[-args.holdout:]:
            f.write(json.dumps({"problem": r["problem"], "facets": r["facets"][:3],
                                "angles": r["angles"], "threads": r["threads"],
                                "source": r.get("source"), "judge": r.get("judge")}) + "\n")

    # decomposer (problem -> facets + 4 family-tagged angles)
    dn = 0
    with (out / "decomposer_train.jsonl").open("w") as f:
        for r in train:
            a = [fam(x) for x in r["angles"]]
            f.write(json.dumps(msgs(DEC_SYS, DEC_USER.format(problem=r["problem"]),
                    DEC_ASST.format(facets=" | ".join(r["facets"][:3]),
                                    a1=a[0], a2=a[1], a3=a[2], a4=a[3]))) + "\n")
            dn += 1

    # worker (problem, facets, family-tagged angle -> foresight thread)
    wn = 0
    with (out / "worker_train.jsonl").open("w") as f:
        for r in train:
            for a, t in zip(r["angles"], r["threads"]):
                f.write(json.dumps(msgs(WRK_SYS, WRK_USER.format(
                    problem=r["problem"], facets=" | ".join(r["facets"][:3]), angle=fam(a)), str(t).strip())) + "\n")
                wn += 1

    print(f"corpus={len(rows)} (dropped {n_drop} malformed) | train_problems={len(train)} | held_out={len(held)}")
    print(f"decomposer_train={dn} | worker_train={wn} (= {len(train)}x4) | eval_problems={len(held)}")
    print(f"out -> {out}/")


if __name__ == "__main__":
    main()
