# Round 2 — runbook (re-gated corpus + action metric, A & B arms)

Pre-built/tested off-pod so paid GPU time is just train + eval. **Three adapters** (parity with round 1):
decomposer (shared) + worker_a (with consequence) + worker_b (without consequence). Eval on `action_pd`.

## NO-MIXUP CONTRACT
- `passers_regated.jsonl`   = **A corpus** (201, threads WITH consequence)
- `passers_regated_b.jsonl` = **B corpus** (201, threads WITHOUT consequence) — SAME problems + angles
- A and B differ ONLY in the worker target. Held-out eval = same 20 problems for both.
- The old `corpus_v2/ab_compare.json` (n=34, sleep-failed) is **superseded — do not reuse.**

## Kit
| file | what | status |
|---|---|---|
| `passers_regated.jsonl` | A corpus (201) | ready |
| `passers_regated_b.jsonl` | B corpus (201, paired) | generated off-pod |
| `prep_v2.py` | both corpora → 3 train files + 20 held-out (strict A/B pairing + leak guard) | tested |
| `train_lora.py` | LoRA SFT (r=32 α=64 3ep) | from v1 |
| `dav_eval_v2.py` | harness eval, **action_pd headline** + full_pd + thread dump | ready |

## On the pod (after upload to `/workspace/div/`)
```bash
# 0. fresh pod only: pip install transformers==5.12 trl peft datasets accelerate bitsandbytes
mkdir -p /workspace/div/data /workspace/div/adapters /workspace/div/out
mv passers_regated.jsonl passers_regated_b.jsonl /workspace/div/data/

# 1. build train splits + shared held-out eval (3 train files)
python prep_v2.py --corpus-a /workspace/div/data/passers_regated.jsonl \
                  --corpus-b /workspace/div/data/passers_regated_b.jsonl \
                  --out-dir /workspace/div/data --holdout 20

# 2. train THREE adapters (~20-40 min each on RTX 6000 Ada)
python train_lora.py --data /workspace/div/data/decomposer_train.jsonl --out /workspace/div/adapters/decomposer
python train_lora.py --data /workspace/div/data/worker_a_train.jsonl   --out /workspace/div/adapters/worker_a
python train_lora.py --data /workspace/div/data/worker_b_train.jsonl   --out /workspace/div/adapters/worker_b

# 3. eval base vs A vs B, all on action_pd (n=20 held-out)
export OPENAI_API_KEY=...  OPENROUTER_API_KEY=...
python dav_eval_v2.py --label base
python dav_eval_v2.py --label A --dec /workspace/div/adapters/decomposer --wrk /workspace/div/adapters/worker_a
python dav_eval_v2.py --label B --dec /workspace/div/adapters/decomposer --wrk /workspace/div/adapters/worker_b
```

## Success criterion
Trained `action_pd` (n=20 held-out) clears base, ideally near corpus 0.49. Expect **B action_pd ≥ A**
(B is the more divergent arm) and **A ground ≥ B** (consequence anchors). `full_pd` is the blinded ruler —
expect it to stay ~0.38; that's fine.

## After
`tar` + sha the 3 new adapters → mirror `divergent-model-backups/`, **then stop/terminate the pod.**
Log the base/A/B action_pd result in WORKING_PAPER §8.
```
