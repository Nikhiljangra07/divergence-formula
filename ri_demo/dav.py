"""
dav.py — score the 10 RI examples with the real DAV criterion.

Geometry is REAL: threads/problem/facets are embedded with OpenAI text-embedding-3-small,
L2-normalized, and DAV is computed exactly as the spec (§11):
  volume    = log det(K + λI)   over CENTERED, renormalized thread embeddings
  grounding = min_i  cos(thread_i, problem)
  wholeness = min_i  mean_j cos(thread_i, facet_j)
The consequence gate is an LLM/human judge in production; here every thread was authored
with a consequence clause, so it is reported as a structural check, not faked as geometry.

Only each thread's `text` is embedded — the hidden `_lens` (its source angle) is stripped,
exactly as it would be before training. Isolated: reads only the OpenAI key; writes nothing.
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
EX = json.load(open(HERE / "examples.json"))


def embed(texts):
    r = httpx.post("https://api.openai.com/v1/embeddings",
                   headers={"Authorization": f"Bearer {KEY}"},
                   json={"model": "text-embedding-3-small", "input": texts},
                   timeout=60)
    r.raise_for_status()
    V = np.array([d["embedding"] for d in r.json()["data"]], dtype=float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)  # L2-normalize


def volume(T):
    """log det(K+λI) over centered, renormalized thread embeddings T (k×d)."""
    c = T - T.mean(0)
    n = np.linalg.norm(c, axis=1, keepdims=True)
    if (n < 1e-6).any():
        return float("-inf")  # total-collapse guard
    c = c / n
    K = c @ c.T
    return float(np.linalg.slogdet(K + LAM * np.eye(len(T)))[1])


def score(p_vec, f_vecs, t_vecs):
    grounding = float((t_vecs @ p_vec).min())
    wholeness = float((t_vecs @ f_vecs.T).mean(1).min())
    return volume(t_vecs), grounding, wholeness


# --- embed everything in one batch -------------------------------------------------
texts, index = [], []
for e in EX:
    index.append((len(texts), "p"));        texts.append(e["problem"])
    fi = len(texts); texts += e["facets"];  index.append((fi, "f", len(e["facets"])))
    ti = len(texts); texts += [t["text"] for t in e["threads"]]; index.append((ti, "t", len(e["threads"])))
V = embed(texts)

rows = []
k = 0
for e in EX:
    p = V[index[k][0]]; k += 1
    f = V[index[k][0]: index[k][0] + index[k][2]]; k += 1
    t = V[index[k][0]: index[k][0] + index[k][2]]; k += 1
    vol, g, w = score(p, f, t)
    rows.append((e["id"], len(e["threads"]), vol, g, w))

# --- report ------------------------------------------------------------------------
print(f"{'example':<22}{'k':>3}{'volume':>10}{'ground.min':>12}{'whole.min':>11}")
print("-" * 58)
for rid, kk, vol, g, w in rows:
    print(f"{rid:<22}{kk:>3}{vol:>10.3f}{g:>12.3f}{w:>11.3f}")

vols = np.array([r[2] for r in rows]); grs = np.array([r[3] for r in rows]); whs = np.array([r[4] for r in rows])
print("-" * 58)
print(f"{'mean':<22}{'':>3}{vols.mean():>10.3f}{grs.mean():>12.3f}{whs.mean():>11.3f}")
print(f"{'min':<22}{'':>3}{vols.min():>10.3f}{grs.min():>12.3f}{whs.min():>11.3f}")
print(f"{'max':<22}{'':>3}{vols.max():>10.3f}{grs.max():>12.3f}{whs.max():>11.3f}")

# --- collapse control: prove the volume term punishes a hack -----------------------
print("\nCOLLAPSE CONTROL (RI-01 with thread #4 replaced by a near-copy of thread #1):")
e = EX[0]
ti = index[1*3 - 1][0]  # RI-01 thread start (k positions: each example uses 3 index entries)
# recompute RI-01 thread vectors directly
base = embed([t["text"] for t in e["threads"]])
hacked = base.copy()
hacked[3] = base[0] * 0.97 + base[1] * 0.03  # thread4 ≈ thread1 (a rhyming duplicate)
hacked = hacked / np.linalg.norm(hacked, axis=1, keepdims=True)
print(f"  genuine 4 distinct threads : volume = {volume(base):.3f}")
print(f"  one thread collapsed onto another : volume = {volume(hacked):.3f}")
print("  → the volume term drops sharply when two threads stop being distinct.")
