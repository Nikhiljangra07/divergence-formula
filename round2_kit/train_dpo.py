"""
train_dpo.py — LoRA DPO of granite-4.0-micro on conversational preference pairs (prompt/chosen/rejected).

Recipe (standard SFT->DPO): the SFT worker adapter is MERGED into the base (that merged model becomes the
DPO reference, used adapter-disabled), then a FRESH LoRA is trained with DPO on the 1828 (clean vs flawed)
worker pairs. DPO is the step SFT structurally cannot do — it pushes probability mass toward the clean,
viable, distinct thread and AWAY from the baroque / unethical / duplicate one. This targets the
distinctness/viability ceiling SFT alone plateaued at (~2.75).

  python train_dpo.py --data data_v5/dpo_pairs.jsonl --sft adapters/worker_v5 --out adapters/worker_v5_dpo
  python train_dpo.py ... --max_steps 5    # GPU dry-run (catch the granite aux-loss landmine before the full run)
"""
import argparse, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

BASE = "ibm-granite/granite-4.0-micro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sft", required=True, help="SFT worker adapter to start from (merged in = DPO reference)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-6)        # DPO uses a much lower LR than SFT (2e-4)
    ap.add_argument("--beta", type=float, default=0.1)        # KL strength to the reference
    ap.add_argument("--bs", type=int, default=2)             # DPO holds chosen+rejected -> ~2x memory
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=-1)     # >0 = sanity dry-run (overrides epochs)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    # Bake the SFT worker into the weights -> this merged model is the DPO reference (adapter-disabled forward).
    model = PeftModel.from_pretrained(base, args.sft).merge_and_unload()

    # granite-4.0-micro carries MoE-ish config flags TRL's aux-loss path trips on (cf. train_lora's
    # router_aux_loss_coef=0.0). Defuse at the MODEL level so DPO's forward never emits/charges router aux loss.
    for attr, val in (("output_router_logits", False), ("router_aux_loss_coef", 0.0)):
        if hasattr(model.config, attr):
            setattr(model.config, attr, val)

    ds = load_dataset("json", data_files=args.data, split="train")

    peft_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules="all-linear")

    cfg = DPOConfig(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1, beta=args.beta,
        logging_steps=10, save_strategy="no", bf16=True, max_length=1024,  # TRL 1.7 DPOConfig: only max_length (no max_prompt_length)
        report_to="none", seed=42,
    )

    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"SAVED DPO adapter -> {args.out}")


if __name__ == "__main__":
    main()
