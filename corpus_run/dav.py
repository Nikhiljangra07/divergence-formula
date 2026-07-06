"""
dav.py — DAV scoring + the gate, with the consequence floor as a SWITCH (DAV_full vs DAV_lite).

Same geometry as the pilot (proven on the 50-passer run): centered+renormalized cosine Gram,
log-det volume, min-grounding, min-wholeness, mean pairwise distance. Adds a completeness check
(reject mid-sentence truncation) that applies to BOTH sets. consequence_gate toggles the single
ablation variable.
"""
import re
import numpy as np
import httpx
import config as C

# consequence/trade vocabulary — broad on purpose (the worker varies phrasing); used ONLY for Set A.
CONS_RE = re.compile(
    r"\b(cost|costs|price|gain|gains|lose|loses|lost|losing|forfeit|forfeits|"
    r"surrender|surrenders|sacrifice|sacrifices|trade|trades|trading|exchange|"
    r"buy|buys|buying|bought|purchase|purchases|purchasing|convert|converts|converting|"
    r"transform|transforms|secure|secures|preserve|preserves|ensure|ensures|"
    r"yield|yields|risk|risks|spend|spends|spent|but|yet|though|however|whereas|while)\b|—",
    re.I,
)
TERMINAL = ('.', '!', '?', '"', '”', '’', "'", ')')


def embed(texts, key):
    out = []
    for i in range(0, len(texts), 256):
        r = httpx.post("https://api.openai.com/v1/embeddings",
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": "text-embedding-3-small", "input": texts[i:i+256]}, timeout=60)
        r.raise_for_status(); out += [d["embedding"] for d in r.json()["data"]]
    V = np.array(out, float); return V / np.linalg.norm(V, axis=1, keepdims=True)


def volume(T):
    c = T - T.mean(0); n = np.linalg.norm(c, axis=1, keepdims=True)
    if (n < 1e-6).any(): return float("-inf")
    c = c / n; K = c @ c.T
    return float(np.linalg.slogdet(K + C.LAM * np.eye(len(T)))[1])


def pairdist(T):
    S = T @ T.T; iu = np.triu_indices(len(T), 1); return float((1 - S[iu]).mean())


def consequence_ok(threads):
    """Each thread is >= MIN_WORDS and carries consequence/trade language (the regex proxy for val(Cᵢ))."""
    return all(len(t.split()) >= C.MIN_WORDS and CONS_RE.search(t) for t in threads)


def score(problem_v, facet_v, thread_v):
    return {
        "volume": volume(thread_v),
        "ground": float((thread_v @ problem_v).min()),
        "whole": float((thread_v @ facet_v.T).mean(1).min()),
        "pairdist": pairdist(thread_v),
    }


def gate_fails(m, threads, consequence_gate):
    """Return [(gate, margin, detail)] for every floor missed. consequence floor only if requested."""
    fails = []
    if m["volume"] <= C.VOL_GATE:
        fails.append(("volume", C.VOL_GATE - m["volume"], "threads collapsed onto one direction"))
    if m["ground"] < C.EPS_G:
        fails.append(("ground", C.EPS_G - m["ground"], "a thread drifts off the actual problem"))
    if m["whole"] < C.EPS_W:
        fails.append(("whole", C.EPS_W - m["whole"], "a thread ignores the key facets"))
    if m["pairdist"] < C.PD_FLOOR:
        fails.append(("pairdist", C.PD_FLOOR - m["pairdist"], "two threads are near-duplicates"))
    # length + completeness apply to BOTH sets
    short = [i + 1 for i, t in enumerate(threads) if len(t.split()) < C.MIN_WORDS]
    if short:
        fails.append(("length", 1.0, f"thread(s) {short} under {C.MIN_WORDS} words"))
    incomplete = [i + 1 for i, t in enumerate(threads) if not t.rstrip().endswith(TERMINAL)]
    if incomplete:
        fails.append(("incomplete", 1.0, f"thread(s) {incomplete} cut off mid-sentence"))
    # the single ablation variable
    if consequence_gate:
        flat = [i + 1 for i, t in enumerate(threads) if not CONS_RE.search(t)]
        if flat:
            fails.append(("consequence", 1.0, f"thread(s) {flat} project no cost/gain"))
    return fails


def verdict(fails):
    """pass / near_miss / reject. near_miss = a single GEOMETRY gate missed within NEAR_MARGIN."""
    if not fails:
        return "pass"
    if len(fails) == 1:
        g, margin, _ = fails[0]
        if g in ("volume", "ground", "whole", "pairdist") and margin <= C.NEAR_MARGIN:
            return "near_miss"
    return "reject"
