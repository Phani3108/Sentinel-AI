"""
Sentinel AI — LoRA Fine-Tuning Script (Phase 5)
Fine-tune Florence-2 or LLaVA on custom VQA data using PEFT (LoRA).

Usage:
    python finetune/train_lora.py \
        --model florence2 \
        --train_data finetune/data/florence_train.jsonl \
        --val_data finetune/data/florence_val.jsonl \
        --output_dir finetune/checkpoints/florence2-lora \
        --epochs 3 \
        --batch_size 4 \
        --lr 2e-4

NOTE: Requires GPU (CUDA or Apple MPS). Training on CPU is very slow.
"""
import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Argument Parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sentinel AI LoRA Fine-Tuner")
    p.add_argument("--model", choices=["florence2", "llava", "phi3v"], default="florence2",
                   help="Base model to fine-tune")
    p.add_argument("--train_data", default="finetune/data/florence_train.jsonl")
    p.add_argument("--val_data",   default="finetune/data/florence_val.jsonl")
    p.add_argument("--output_dir", default="finetune/checkpoints/model-lora")
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--lora_r",     type=int, default=8, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA 4-bit quantization")
    p.add_argument("--mlflow_experiment", default="sentinel-ai-finetune")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_IDS = {
    "florence2": "microsoft/Florence-2-base-ft",
    "llava":     "llava-hf/llava-1.5-7b-hf",
    "phi3v":     "microsoft/Phi-3-vision-128k-instruct",
}

LORA_TARGET_MODULES = {
    "florence2": ["q_proj", "v_proj", "k_proj", "out_proj"],
    "llava":     ["q_proj", "v_proj", "k_proj", "o_proj"],
    "phi3v":     ["qkv_proj", "o_proj"],
}


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Run: pip install transformers peft datasets")
        raise

    # MLflow tracking
    try:
        from finetune.mlflow_tracking import start_run, log_params, log_metrics, end_run
        run = start_run(args.mlflow_experiment, args.model)
        log_params({
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "load_in_4bit": args.load_in_4bit,
        })
        mlflow_enabled = True
    except Exception:
        mlflow_enabled = False
        logger.warning("MLflow not available — training without experiment tracking")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Training on: {device}")

    model_id = MODEL_IDS[args.model]
    logger.info(f"Loading base model: {model_id}")

    # Load model
    load_kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        **load_kwargs,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES[args.model],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=(device == "cuda"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=args.seed,
        report_to=["mlflow"] if mlflow_enabled else [],
    )

    # Dataset
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_data, "val": args.val_data},
    )

    def preprocess(batch):
        inputs = processor(
            text=batch["question"],
            images=None,  # handled separately in collator
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs["labels"] = processor.tokenizer(
            batch["answer"], return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        return inputs

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    model.save_pretrained(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    logger.info(f"Model saved → {output_dir}/final")

    # Log final metrics
    if mlflow_enabled:
        eval_result = trainer.evaluate()
        log_metrics(eval_result)
        end_run()

    return str(output_dir / "final")


if __name__ == "__main__":
    args = parse_args()
    train(args)
