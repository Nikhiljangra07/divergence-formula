"""
compare_sources.py — score RI vs Jin Ping Mei example sets with the same DAV and compare.

Same geometry as dav.py (real OpenAI embeddings, spec §11). Format is held constant across
both sources, so the only variable is the source DOMAIN (strategic-survival vs domestic-social).
Isolated: reads only the OpenAI key; writes nothing.
"""
import json, os
from pathlib import Path
import numpy as np
import httpx
from dotenv import load_dotenv

load_dotenv(Path.home() / "Desktop" / "reasoningEngine" / ".env")
KEY = os.environ["OPENAI_API_KEY"]
LAM = 1e-3
HERE = Path(__file__).resolve().parent
SOURCES = [("RI  (Reverend Insanity)", "examples.json"),
           ("JPM (Jin Ping Mei)", "examples_jpm.json"),
           ("PR  (The Prince)", "examples_prince.json"),
           ("MB  (Mahabharata)", "examples_mahabharata.json")]


def embed(texts):
    out = []
    for i in range(0, len(texts), 256):
        r = httpx.post("https://api.openai.com/v1/embeddings",
                       headers={"Authorization": f"Bearer {KEY}"},
                       json={"model": "text-embedding-3-small", "input": texts[i:i+256]},
                       timeout=60)
        r.raise_for_status()
        out += [d["embedding"] for d in r.json()["data"]]
    V = np.array(out, dtype=float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def volume(T):
    c = T - T.mean(0)
    n = np.linalg.norm(c, axis=1, keepdims=True)
    if (n < 1e-6).any():
        return float("-inf")
    c = c / n
    K = c @ c.T
    return float(np.linalg.slogdet(K + LAM * np.eye(len(T)))[1])


def mean_pair_dist(T):
    """mean pairwise cosine DISTANCE among threads — a finer-grained divergence ranker."""
    S = T @ T.T
    k = len(T)
    iu = np.triu_indices(k, 1)
    return float((1 - S[iu]).mean())


def score_file(path):
    EX = json.load(open(HERE / path))
    texts, spans = [], []
    for e in EX:
        s = {}
        s["p"] = len(texts); texts.append(e["problem"])
        s["f0"] = len(texts); texts += e["facets"]; s["fn"] = len(e["facets"])
        s["t0"] = len(texts); texts += [t["text"] for t in e["threads"]]; s["tn"] = len(e["threads"])
        spans.append((e["id"], s))
    V = embed(texts)
    rows = []
    for rid, s in spans:
        p = V[s["p"]]; f = V[s["f0"]:s["f0"]+s["fn"]]; t = V[s["t0"]:s["t0"]+s["tn"]]
        rows.append((rid, volume(t), float((t @ p).min()), float((t @ f.T).mean(1).min()), mean_pair_dist(t)))
    return rows


def stats(rows, col):
    a = np.array([r[col] for r in rows])
    return a.mean(), a.min(), a.max(), a.std()


print(f"{'':<26}{'volume':>9}{'pairDist':>10}{'ground.min':>12}{'whole.min':>11}")
allrows = {}
for label, path in SOURCES:
    rows = score_file(path)
    allrows[label] = rows
    print(f"\n== {label} ==")
    for rid, vol, g, w, pd in rows:
        print(f"  {rid:<24}{vol:>9.3f}{pd:>10.3f}{g:>12.3f}{w:>11.3f}")
    vm = stats(rows, 1); pm = stats(rows, 4); gm = stats(rows, 2); wm = stats(rows, 3)
    print(f"  {'MEAN':<24}{vm[0]:>9.3f}{pm[0]:>10.3f}{gm[0]:>12.3f}{wm[0]:>11.3f}")
    print(f"  {'STD':<24}{vm[3]:>9.3f}{pm[3]:>10.3f}{gm[3]:>12.3f}{wm[3]:>11.3f}")

print("\n" + "=" * 64)
print("SIDE BY SIDE (means across all sources)")
labels = [s[0] for s in SOURCES]
short = [lab.split()[0] for lab in labels]
print(f"{'metric':<12}" + "".join(f"{s:>11}" for s in short))
for name, col in [("volume", 1), ("pairDist", 4), ("ground.min", 2), ("whole.min", 3)]:
    vals = [stats(allrows[lab], col)[0] for lab in labels]
    print(f"{name:<12}" + "".join(f"{v:>11.3f}" for v in vals))
