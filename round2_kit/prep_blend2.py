"""
prep_blend2.py — build the BLEND2 training set for the H-Small round-8 retrain:

  data_v5blend (verbatim)             — v5 480-corpus rows + 49 regated v3/v4 rows, exactly as round-7
                                        run#2 trained (dec 506 / wrk 2024). Untouched.
  corpus_v5_topup/passers.jsonl       — 484 fresh v5-style balanced examples (8 sources, Haiku gate,
                                        same calibration as the original 480). V5 contract.
  corpus_essence/passers.jsonl        — 360 essence-round examples (influence craft, 24 archetypes,
                                        Gemini gate viab&dist&conc>=4). V5 contract: the essence worker
                                        prompt was designed to distill into the same eval form (viable,
                                        mechanically specific first move, one downstream consequence).

Output data_blend2/: decomposer_train.jsonl / worker_train.jsonl / eval_problems.jsonl (copied from
data_v5 — the SAME 20 held-out problems as rounds 5-7, so results stay comparable across rounds).

Guards:
  - leak: no v5 held-out problem, no OOD bench problem in any train row (assert, not just skip-count)
  - dedup: exact-problem dedup across v5+regated+topup+essence (two topup writers ran concurrently)
  - wellformed: 4 angles, 4 pos threads >=10 words each, facets present

  python prep_blend2.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5DATA = HERE / "data_v5"
BLEND = HERE / "data_v5blend"
CORPUS = HERE.parent / "corpus_run"
TOPUP = CORPUS / "corpus_v5_topup" / "passers.jsonl"
ESSENCE = CORPUS / "corpus_essence" / "passers.jsonl"
V5RAW = CORPUS / "corpus_v5_train" / "passers.jsonl"
REGATED = CORPUS / "corpus_v34_regated" / "passers.jsonl"
BENCH = CORPUS / "benchmark" / "problems.jsonl"
OUT = HERE / "data_blend2"; OUT.mkdir(exist_ok=True)

# ---- v5 contract (BYTE-IDENTICAL to prep_v5.py — train == eval == bench) ----
DEC_SYS = "You refract a hard decision problem into four categorically distinct strategic angles."
DEC_USER = ("PROBLEM: {problem}\n\nGive the three key FACETS any complete answer must engage, then FOUR "
            "genuinely distinct strategic ANGLES for approaching the whole problem. Each angle must be a "
            "categorically different KIND of move (a distinct family) leading to a different action — sharp "
            "alternatives, not rephrasings.\nFormat exactly:\n"
            "FACETS: <f1> | <f2> | <f3>\nANGLES:\n1) [FAMILY] <angle>\n2) [FAMILY] <angle>\n"
            "3) [FAMILY] <angle>\n4) [FAMILY] <angle>")
DEC_ASST = "FACETS: {facets}\nANGLES:\n1) {a1}\n2) {a2}\n3) {a3}\n4) {a4}"
WRK_SYS = "You write one precise, decisive, realistic reasoning thread pursuing a given strategic angle."
WRK_USER = ("PROBLEM: {problem}\nFACETS: {facets}\nANGLE: {angle}\n\nWrite a single reasoning thread (two or "
            "three sentences, cold and analytical) that COMMITS to THIS angle as a concrete, realistic, VIABLE "
            "strategy that resolves the whole problem in a distinct way — name the actual first move (who does "
            "what, to whom, by when) and the one most likely downstream consequence it is betting on. It must be "
            "lawful, executable, and unmistakably a different KIND of move than the other families would choose.")


def msgs(s, u, a):
    return {"messages": [{"role": "system", "content": s}, {"role": "user", "content": u},
                         {"role": "assistant", "content": a}]}


def fam(a):
    if isinstance(a, dict):
        f = str(a.get("family", "")).strip(); d = str(a.get("directive", "")).strip()
        return f"[{f}] {d}" if f else d
    return str(a)


def wellformed(r):
    return len(r.get("angles", [])) == 4 and len(r.get("pos_threads", [])) == 4 and r.get("facets") \
        and all(t and len(str(t).split()) >= 10 for t in r["pos_threads"])


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def main():
    protected = {json.loads(l)["problem"].strip() for l in (V5DATA / "eval_problems.jsonl").open()} | \
                {json.loads(l)["problem"].strip() for l in BENCH.open()}

    # problems already trained on (v5 corpus + regated) — new rows must not duplicate them
    seen = {r["problem"].strip() for r in load_jsonl(V5RAW)} | \
           {r["problem"].strip() for r in load_jsonl(REGATED)}

    # start from the round-7 blend verbatim
    dec_rows = load_jsonl(BLEND / "decomposer_train.jsonl")
    wrk_rows = load_jsonl(BLEND / "worker_train.jsonl")
    n_blend_dec, n_blend_wrk = len(dec_rows), len(wrk_rows)

    stats = {}
    for tag, path in (("topup", TOPUP), ("essence", ESSENCE)):
        kept = skipped_leak = skipped_dup = skipped_malformed = 0
        for r in load_jsonl(path):
            if not wellformed(r):
                skipped_malformed += 1; continue
            p = r["problem"].strip()
            if p in protected:
                skipped_leak += 1; continue
            if p in seen:
                skipped_dup += 1; continue
            seen.add(p)
            a = [fam(x) for x in r["angles"]]
            dec_rows.append(msgs(DEC_SYS, DEC_USER.format(problem=p),
                                 DEC_ASST.format(facets=" | ".join(r["facets"][:3]),
                                                 a1=a[0], a2=a[1], a3=a[2], a4=a[3])))
            for ang, t in zip(r["angles"], r["pos_threads"]):
                wrk_rows.append(msgs(WRK_SYS, WRK_USER.format(problem=p, facets=" | ".join(r["facets"][:3]),
                                                              angle=fam(ang)), str(t).strip()))
            kept += 1
        stats[tag] = {"kept": kept, "leak": skipped_leak, "dup": skipped_dup, "malformed": skipped_malformed}

    # hard leak assert over the FINAL files, not just the additions
    for rows in (dec_rows, wrk_rows):
        for r in rows:
            u = r["messages"][1]["content"]
            prob = u.split("PROBLEM: ", 1)[1].split("\n", 1)[0].strip()
            assert prob not in protected, f"LEAK IN FINAL TRAIN SET: {prob[:80]}"

    with (OUT / "decomposer_train.jsonl").open("w") as f:
        for r in dec_rows: f.write(json.dumps(r) + "\n")
    with (OUT / "worker_train.jsonl").open("w") as f:
        for r in wrk_rows: f.write(json.dumps(r) + "\n")
    (OUT / "eval_problems.jsonl").write_text((V5DATA / "eval_problems.jsonl").read_text())

    total_dec, total_wrk = len(dec_rows), len(wrk_rows)
    ess_share = round(100 * stats["essence"]["kept"] * 5 / (total_dec + total_wrk), 1)
    print(f"blend base (round-7 run#2): dec {n_blend_dec} / wrk {n_blend_wrk}")
    for tag, s in stats.items(): print(f"{tag}: {s}")
    print(f"BLEND2 totals: decomposer {total_dec} | worker {total_wrk} | rows {total_dec + total_wrk} "
          f"| essence share ~{ess_share}% | eval = v5 held-out 20 (identical to rounds 5-7)")
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main()
