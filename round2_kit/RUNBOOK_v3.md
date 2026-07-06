# RUNBOOK v3 — train + judge-eval the FORESIGHT corpus

**Goal:** does the v3 corpus (Thucydides-swapped, foresight, judge-gated) move the trained model where round-2 did not? Single arm: **2 adapters** (decomposer + worker), then **base vs trained** scored by the JUDGE.

Matched to round-2 size: 181 train problems, 724 worker rows, 20 held-out eval.

## 0. Pod
RTX 6000 Ada 49GB (~$0.79/hr). Total wall: ~1.5–2 hr (2 trains ~20–30 min each + 2 evals). Budget ~$2–3.

## 1. Setup (on pod)
```bash
pip install --break-system-packages -U "transformers==5.12.1" "peft==0.19.1" "trl==1.7.0" \
    "torch==2.8.0" datasets accelerate httpx numpy
mkdir -p /workspace/div/data_v3 /workspace/div/out /workspace/div/adapters
```

## 2. Upload (from laptop — round2_kit/)
```bash
# scripts
scp dav_eval_v3.py train_lora.py <pod>:/workspace/div/
# data
scp data_v3/decomposer_train.jsonl data_v3/worker_train.jsonl data_v3/eval_problems.jsonl <pod>:/workspace/div/data_v3/
```

## 3. Train 2 adapters
```bash
cd /workspace/div
python train_lora.py --data data_v3/decomposer_train.jsonl --out adapters/decomposer
python train_lora.py --data data_v3/worker_train.jsonl     --out adapters/worker
```
(LoRA r=32/α=64/all-linear/3ep/lr2e-4/bf16; `router_aux_loss_coef=0.0` already set — TRL mis-IDs granite as MoE.)

## 4. Eval — base then trained (judge headline)
Keys: pass `OPENROUTER_API_KEY` (judge) and `OPENAI_API_KEY` (full_pd diagnostic) **without echoing them**.
```bash
export OPENROUTER_API_KEY=...   # from ~/Desktop/reasoningEngine/.env, do NOT print
export OPENAI_API_KEY=...
cd /workspace/div
python dav_eval_v3.py --label base
python dav_eval_v3.py --label trained --dec adapters/decomposer --wrk adapters/worker
```

## 5. Compare
```bash
echo '--- base ---';    cat out/eval_base_v3.json
echo '--- trained ---'; cat out/eval_trained_v3.json
```
**Read:** `judge_means.foresight`, `.distinctness`, `.decisiveness`, `judge_overall_mean`, `n_fail`.
- **Win** = trained foresight/distinctness/overall **> base**, with low `n_fail`.
- Corpus reference (gated): foresight 3.30 · distinct 4.84 · decisive 4.87 · overall ~4.44.
- The teacher (DSV4) set that bar; the trained 3.4B won't match it, but should beat its own base.

## 6. Pull artifacts back
```bash
scp <pod>:/workspace/div/out/eval_*_v3.json out/
scp <pod>:/workspace/div/out/eval_*_v3_threads.jsonl out/
# adapters (optional, ~250MB each)
scp -r <pod>:/workspace/div/adapters/decomposer <pod>:/workspace/div/adapters/worker ../divergent-model-backups/
```

## Notes / guards
- Greedy both seats (temp=0) — the LoRA'd 3.4B word-salads at temp>0; the round-2 "0.62 action_pd" was that gibberish faking divergence. Don't re-enable sampling.
- Separate model instance per seat — peft 0.19.1 multi-adapter switching garbles granite.
- Eval prompts are byte-identical to prep_v3.py; JUDGE_PROMPT byte-identical to gen_v3.py (verified).
- Eval is non-destructive + cheap (~$0.20 of judge calls per label) — safe to re-run.
