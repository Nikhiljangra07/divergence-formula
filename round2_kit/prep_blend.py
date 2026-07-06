"""
prep_blend.py — build the v3+v4+v5 BLEND training set for the H-Small run #2 (Nikhil's sharpness hypothesis).

Inputs:
  data_v5/decomposer_train.jsonl + worker_train.jsonl   — the v5 set EXACTLY as run #1 uses it (457 + 1828)
  corpus_v34_regated/passers.jsonl                       — v3/v4 rows that survived the sharpness re-gate
Output (data_v5blend/): decomposer_train / worker_train / eval_problems (copied from data_v5 — SAME eval).

Contract policy: every row trains under its ORIGINAL corpus contract (v3 foresight prompts from prep_v3,
v4 precision prompts from prep_v4, v5 rows reused verbatim from data_v5). Eval stays the v5 contract
(dav_eval_v5) — identical to run #1, so run#1-vs-run#2 isolates the DATA variable. The known confound
(mixed output styles vs a v5-contract eval) is accepted and documented in §11 discussion.

Leak guard: no v5 held-out problem and no OOD bench problem may appear in any train row.

  python prep_blend.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5DATA = HERE / "data_v5"
REGATED = HERE.parent / "corpus_run" / "corpus_v34_regated" / "passers.jsonl"
BENCH = HERE.parent / "corpus_run" / "benchmark" / "problems.jsonl"
OUT = HERE / "data_v5blend"; OUT.mkdir(exist_ok=True)

# ---- v3 contract (BYTE-IDENTICAL to prep_v3.py) ----
V3_DEC_SYS = "You refract a hard decision problem into four distinct, forward-looking strategic angles."
V3_DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
               "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
               "categorically different KIND of forward move (a distinct family) leading to a different action — "
               "real alternatives, not rephrasings.\nFormat exactly:\n"
               "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
               "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
V3_WRK_SYS = "You write one decisive, forward-looking reasoning thread pursuing a given strategic angle."
V3_WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (three "
               "sentences, cold and analytical) that COMMITS to THIS angle as a concrete strategy for the whole "
               "problem and names the ONE key downstream reaction it is betting on, one or two moves ahead.")

# ---- v4 contract (BYTE-IDENTICAL to prep_v4.py) ----
V4_DEC_SYS = "You refract a hard decision problem into four categorically distinct strategic angles."
V4_DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
               "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
               "categorically different KIND of move (a distinct family) leading to a different action — sharp "
               "alternatives, not rephrasings.\nFormat exactly:\n"
               "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
               "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
V4_WRK_SYS = "You write one precise, decisive reasoning thread pursuing a given strategic angle."
V4_WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two "
               "sentences, cold and analytical) that COMMITS to THIS angle as a concrete, mechanically specific "
               "strategy for the whole problem — a sharp, non-obvious move that is unmistakably a different KIND "
               "of move than the other families would choose.")
DEC_ASST = "FACETS: {facets}\nANGLES:\n1) {a1}\n2) {a2}\n3) {a3}\n4) {a4}"

CONTRACTS = {"v3": (V3_DEC_SYS, V3_DEC_USER, V3_WRK_SYS, V3_WRK_USER),
             "v4": (V4_DEC_SYS, V4_DEC_USER, V4_WRK_SYS, V4_WRK_USER)}


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
    # protected problems: v5 held-out + OOD bench
    protected = {json.loads(l)["problem"].strip() for l in (V5DATA / "eval_problems.jsonl").open()} | \
                {json.loads(l)["problem"].strip() for l in BENCH.open()}

    # start from the v5 training rows verbatim (identical to run #1)
    dec_rows = [json.loads(l) for l in (V5DATA / "decomposer_train.jsonl").open()]
    wrk_rows = [json.loads(l) for l in (V5DATA / "worker_train.jsonl").open()]
    n_v5_dec, n_v5_wrk = len(dec_rows), len(wrk_rows)

    # add regated v3/v4 rows under their original contracts
    regated = [json.loads(l) for l in REGATED.open() if l.strip()]
    kept, skipped_leak, skipped_malformed = 0, 0, 0
    by_corpus = {"v3": 0, "v4": 0}
    for r in regated:
        if not wellformed(r):
            skipped_malformed += 1; continue
        if r["problem"].strip() in protected:
            skipped_leak += 1; continue
        dsys, duser, wsys, wuser = CONTRACTS[r["corpus"]]
        a = [fam(x) for x in r["angles"]]
        dec_rows.append(msgs(dsys, duser.format(problem=r["problem"]),
                             DEC_ASST.format(facets=" | ".join(r["facets"][:3]), a1=a[0], a2=a[1], a3=a[2], a4=a[3])))
        for ang, t in zip(r["angles"], r["threads"]):
            wrk_rows.append(msgs(wsys, wuser.format(problem=r["problem"], facets=" | ".join(r["facets"][:3]),
                                                    angle=fam(ang)), str(t).strip()))
        kept += 1; by_corpus[r["corpus"]] += 1

    with (OUT / "decomposer_train.jsonl").open("w") as f:
        for r in dec_rows: f.write(json.dumps(r) + "\n")
    with (OUT / "worker_train.jsonl").open("w") as f:
        for r in wrk_rows: f.write(json.dumps(r) + "\n")
    # eval identity with run #1
    (OUT / "eval_problems.jsonl").write_text((V5DATA / "eval_problems.jsonl").read_text())

    print(f"v5 base: dec {n_v5_dec} / wrk {n_v5_wrk}")
    print(f"regated added: {kept} problems ({by_corpus}) | skipped: leak {skipped_leak}, malformed {skipped_malformed}")
    print(f"BLEND totals: decomposer {len(dec_rows)} | worker {len(wrk_rows)} | eval = v5 held-out 20 (identical to run #1)")
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main()
