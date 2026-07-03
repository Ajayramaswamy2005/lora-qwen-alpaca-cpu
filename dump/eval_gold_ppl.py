import math
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FINAL_DIR = "qwen-lora-alpaca-final"
DATASET_NAME = "yahma/alpaca-cleaned"
TRAIN_SKIP = 2000
N = 20
PLOT_FILE = "gold_ppl_base_vs_tuned.png"

BASE_COLOR = "#888888"
TUNED_COLOR = "#4C72B0"


def build_prompt(example):
    instruction = example["instruction"].strip()
    context = (example.get("input") or "").strip()
    user = f"{instruction}\n\n{context}" if context else instruction
    return user, example["output"].strip()


def load_held_out():
    ds = load_dataset(DATASET_NAME, split="train").shuffle(seed=42)
    start = min(TRAIN_SKIP, len(ds) - N)
    return ds.select(range(start, start + N))


def gold_ppl(model, tokenizer, examples):
    model.eval()
    ppls = []
    with torch.no_grad():
        for ex in examples:
            user, gold = build_prompt(ex)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            full = prompt + gold + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            seq = torch.tensor([full_ids])
            logits = model(seq).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            targets = seq[0, 1:]
            token_logps = log_probs[0, :-1, :].gather(1, targets.unsqueeze(1)).squeeze(1)
            gold_logps = token_logps[len(prompt_ids) - 1:]
            mean_logp = gold_logps.mean().item()
            ppls.append(math.exp(-mean_logp) if mean_logp > -20 else float("inf"))
    return ppls


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples = list(load_held_out())
    print(f"[+] Held-out alpaca examples: {len(examples)}")

    print(f"[+] Loading base: {MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base.config.use_cache = True
    base_ppls = gold_ppl(base, tok, examples)

    print(f"[+] Wrapping tuned: {FINAL_DIR}")
    tuned = PeftModel.from_pretrained(base, FINAL_DIR)
    tuned_ppls = gold_ppl(tuned, tok, examples)

    print("\n" + "=" * 60)
    print("GOLD-ANSWER PERPLEXITY (lower = fits alpaca better)")
    print("=" * 60)
    print(f"{'#':>3} {'base':>8} {'tuned':>8} {'delta':>8}")
    for i in range(N):
        d = tuned_ppls[i] - base_ppls[i]
        print(f"{i+1:>3} {base_ppls[i]:>8.2f} {tuned_ppls[i]:>8.2f} {d:>+8.2f}")
    ab, at = avg(base_ppls), avg(tuned_ppls)
    print(f"\nAVG  base {ab:.3f}  | tuned {at:.3f}  "
          f"-> {(at-ab)/ab*100:+.1f}%  (negative = tuned better fit)")

    x = torch.arange(N)
    w = 0.4
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, [min(p, 50) for p in base_ppls], w, label="base", color=BASE_COLOR)
    ax.bar(x + w/2, [min(p, 50) for p in tuned_ppls], w, label="fine-tuned", color=TUNED_COLOR)
    ax.set_title("Perplexity on gold alpaca answer (lower = better fit to dataset)")
    ax.set_ylabel("perplexity")
    ax.set_xlabel("example #")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(N)])
    ax.axhline(ab, color=BASE_COLOR, linestyle="--", alpha=0.6)
    ax.axhline(at, color=TUNED_COLOR, linestyle="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)
    print(f"[+] Plot saved to {PLOT_FILE}")


if __name__ == "__main__":
    main()
