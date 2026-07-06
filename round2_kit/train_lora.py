"""
train_lora.py — LoRA SFT of granite-4.0-micro on a chat-format jsonl (messages column).

  python train_lora.py --data data/decomposer_train.jsonl --out adapters/decomposer
  python train_lora.py --data data/worker_a_train.jsonl   --out adapters/worker_a
  python train_lora.py --data data/worker_b_train.jsonl   --out adapters/worker_b
"""
import argparse, os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# LORA_BASE env overrides for the H-Small runs (ibm-granite/granite-4.0-h-small); default stays Micro.
BASE = os.environ.get("LORA_BASE", "ibm-granite/granite-4.0-micro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--gc", action="store_true", help="gradient checkpointing (needed for 32B on 80GB)")
    # Micro: TRL 1.7 mis-IDs granite as MoE -> 0.0 disables the crashing aux-loss path (the §5 fix).
    # H-Small (REAL MoE): pass the model's own value (e.g. --router_aux -1 keeps model default) so expert
    # load-balancing stays ON. -1 = don't override.
    ap.add_argument("--router_aux", type=float, default=0.0)
    ap.add_argument("--optim", default="adamw_torch")  # paged_adamw_8bit for H-Small: fp32 AdamW states on ~600M LoRA params OOM the 80GB card
    ap.add_argument("--max_steps", type=int, default=-1)  # >0 = sanity dry-run (overrides epochs)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")

    ds = load_dataset("json", data_files=args.data, split="train")

    peft_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules="all-linear")

    cfg_kw = dict(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, save_strategy="no", bf16=True, max_length=1024,
        packing=False, report_to="none", seed=42, gradient_checkpointing=args.gc, optim=args.optim,
    )
    if args.router_aux >= 0:
        cfg_kw["router_aux_loss_coef"] = args.router_aux  # 0.0 = the Micro fix; -1 skips (H-Small keeps model default)
    cfg = SFTConfig(**cfg_kw)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"SAVED adapter -> {args.out}")


if __name__ == "__main__":
    main()
