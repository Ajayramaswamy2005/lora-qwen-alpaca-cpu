import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"
OUTPUT_DIR = "qwen-lora-alpaca"
FINAL_DIR = "qwen-lora-alpaca-final"

MAX_TRAIN_SAMPLES = 2000
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 1
LEARNING_RATE = 2e-4
EVAL_SAMPLES = 50
GEN_SAMPLES = 3
GEN_MAX_NEW_TOKENS = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_prompt(example):
    instruction = example["instruction"].strip()
    context = (example.get("input") or "").strip()
    output = example["output"].strip()
    if context:
        user = f"{instruction}\n\n{context}"
    else:
        user = instruction
    return user, output


def format_dataset(tokenizer):
    def fn(example):
        return tokenize_example(tokenizer, example)
    return fn


def tokenize_example(tokenizer, example):
    user, assistant = build_prompt(example)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = prompt + assistant + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    if len(full_ids) > MAX_LEN:
        full_ids = full_ids[:MAX_LEN]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[: len(full_ids)]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    print(f"[+] Loading tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"[+] Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")
    if MAX_TRAIN_SAMPLES and MAX_TRAIN_SAMPLES < len(ds):
        ds = ds.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
    ds = ds.map(format_dataset(tokenizer), remove_columns=ds.column_names)
    print(f"[+] Training samples: {len(ds)}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        use_cpu=True,
        fp16=False,
        bf16=False,
        dataloader_num_workers=4,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    print("[+] Starting training (CPU)...")
    trainer.train()

    print(f"[+] Saving LoRA adapter to {FINAL_DIR}")
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print("[+] Done.")
    compare_with_base(tokenizer)


def load_held_out(loss_n, gen_n):
    full = load_dataset(DATASET_NAME, split="train").shuffle(seed=42)
    start = MAX_TRAIN_SAMPLES or 0
    start = min(start, len(full) - loss_n - gen_n)
    loss_ds = full.select(range(start, start + loss_n))
    gen_ds = full.select(range(start + loss_n, start + loss_n + gen_n))
    return loss_ds, gen_ds


def compute_eval_loss(model, tokenizer, examples):
    model.eval()
    collator = PadCollator(tokenizer.pad_token_id)
    feats = [tokenize_example(tokenizer, ex) for ex in examples]
    total_loss, total_tokens = 0.0, 0
    bs = 4
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            batch = collator(feats[i:i + bs])
            labels = batch["labels"]
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
            n_valid = (labels != -100).sum().item()
            total_loss += out.loss.item() * n_valid
            total_tokens += n_valid
    return total_loss / max(total_tokens, 1)


def generate(model, tokenizer, user_text):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=False,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def compare_with_base(tokenizer):
    from peft import PeftModel

    print("\n" + "=" * 70)
    print("[COMPARISON] base model vs fine-tuned (LoRA) on held-out alpaca")
    print("=" * 70)

    loss_ds, gen_ds = load_held_out(EVAL_SAMPLES, GEN_SAMPLES)

    print(f"[+] Loading base model for comparison: {MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base.config.use_cache = True

    base_loss = compute_eval_loss(base, tokenizer, list(loss_ds))
    print(f"[+] Held-out loss  | base: {base_loss:.4f}")

    ft = PeftModel.from_pretrained(base, FINAL_DIR)
    ft_loss = compute_eval_loss(ft, tokenizer, list(loss_ds))
    print(f"[+] Held-out loss  | fine-tuned: {ft_loss:.4f}")
    delta = base_loss - ft_loss
    print(f"[+] Loss reduction: {delta:.4f}  ({delta / base_loss * 100:.1f}% lower)")

    print("\n" + "-" * 70)
    print(f"[+] Side-by-side generations on {GEN_SAMPLES} held-out examples")
    print("-" * 70)
    base.config.use_cache = True
    for ex in gen_ds:
        user, reference = build_prompt(ex)
        base_out = generate(base, tokenizer, user)
        ft_out = generate(ft, tokenizer, user)
        print("\n### INSTRUCTION:")
        print(user)
        print("\n### REFERENCE (alpaca gold):")
        print(reference)
        print("\n### BASE MODEL:")
        print(base_out)
        print("\n### FINE-TUNED:")
        print(ft_out)
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
