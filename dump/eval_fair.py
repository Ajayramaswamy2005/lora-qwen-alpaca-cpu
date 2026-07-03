import json
import math
import time
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
MAX_NEW_TOKENS = 150
WARMUP_PROMPT = "What is 2+2? Answer briefly."
TUNED_FILE = "tuned_results.json"
BASE_FILE = "base_results.json"
PLOT_FILE = "fair_compare.png"

BASE_COLOR = "#888888"
TUNED_COLOR = "#4C72B0"


def build_prompt(ex):
    instruction = ex["instruction"].strip()
    context = (ex.get("input") or "").strip()
    user = f"{instruction}\n\n{context}" if context else instruction
    return user, ex["output"].strip()


def load_questions():
    ds = load_dataset(DATASET_NAME, split="train").shuffle(seed=42)
    start = min(TRAIN_SKIP, len(ds) - N)
    return [ds[i] for i in range(start, start + N)]


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def warmup(model, tokenizer):
    inputs = tokenizer(WARMUP_PROMPT, return_tensors="pt")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)


def run(model, tokenizer, examples, label):
    print(f"\n=== {label} ===")
    print("[+] warmup generation (steady thermal state)...")
    warmup(model, tokenizer)
    results = []
    for i, ex in enumerate(examples, 1):
        user, gold = build_prompt(ex)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen_time = time.time() - t0
        gen_ids = out[0, prompt_len:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        n_tok = int(len(gen_ids))
        results.append({
            "user": user, "gold": gold, "answer": answer,
            "n_tokens": n_tok, "gen_time": gen_time,
            "tok_per_sec": n_tok / gen_time if gen_time > 0 else 0,
        })
        print(f"  [{i:>2}/{N}] {n_tok:>3} tok | {gen_time:>5.1f}s | "
              f"{n_tok/gen_time:>4.1f} tok/s")
    return results


def save(res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[+] Saved {path}")


def load_results(path):
    with open(path) as f:
        return json.load(f)


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def plot(base_res, tuned_res):
    n = N
    x = torch.arange(n)
    w = 0.4
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    def grouped(ax, bvals, tvals, title, ylabel, fmt):
        ax.bar(x - w/2, bvals, w, label="base", color=BASE_COLOR)
        ax.bar(x + w/2, tvals, w, label="fine-tuned", color=TUNED_COLOR)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i+1) for i in range(n)])
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        b, t = avg(bvals), avg(tvals)
        ax.axhline(b, color=BASE_COLOR, linestyle="--", alpha=0.6)
        ax.axhline(t, color=TUNED_COLOR, linestyle="--", alpha=0.6)
        ax.text(n - 0.5, max(b, t), f"avg base={fmt(b)}  tuned={fmt(t)}",
                ha="right", va="bottom", fontsize=8)

    grouped(axes[0],
            [r["gen_time"] for r in base_res],
            [r["gen_time"] for r in tuned_res],
            "Inference time per example (s)", "seconds", lambda v: f"{v:.1f}s")
    grouped(axes[1],
            [r["n_tokens"] for r in base_res],
            [r["n_tokens"] for r in tuned_res],
            "Response length (tokens)", "tokens", lambda v: f"{v:.0f}")
    grouped(axes[2],
            [r["tok_per_sec"] for r in base_res],
            [r["tok_per_sec"] for r in tuned_res],
            "Throughput (tokens/sec) — higher = faster", "tokens/sec",
            lambda v: f"{v:.1f}")
    axes[2].set_xlabel("example #")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)
    print(f"[+] Plot saved to {PLOT_FILE}")


def phase_tuned():
    tok = load_tokenizer()
    examples = load_questions()
    with open("questions.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"[+] {len(examples)} in-distribution questions saved to questions.json")
    print(f"[+] Loading base + wrapping LoRA: {FINAL_DIR}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base.config.use_cache = True
    tuned = PeftModel.from_pretrained(base, FINAL_DIR)
    tuned.eval()
    res = run(tuned, tok, examples, "FINE-TUNED model")
    save(res, TUNED_FILE)
    print("\n[+] Phase 1 (tuned) done. Let the CPU cool, then run:")
    print("    uv run python eval_fair.py base")


def phase_base():
    tok = load_tokenizer()
    examples = load_results("questions.json")
    print(f"[+] Loading base: {MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base.config.use_cache = True
    base.eval()
    res = run(base, tok, examples, "BASE model")
    save(res, BASE_FILE)
    compare()


def compare():
    base_res = load_results(BASE_FILE)
    tuned_res = load_results(TUNED_FILE)
    print("\n" + "=" * 72)
    print("FAIR COMPARISON  (both ran on cool CPU with warmup)")
    print("=" * 72)
    print(f"{'#':>3} {'time_b':>7} {'time_t':>7} {'tps_b':>6} {'tps_t':>6} "
          f"{'tok_b':>6} {'tok_t':>6}")
    for i in range(N):
        b, t = base_res[i], tuned_res[i]
        print(f"{i+1:>3} {b['gen_time']:>7.1f} {t['gen_time']:>7.1f} "
              f"{b['tok_per_sec']:>6.1f} {t['tok_per_sec']:>6.1f} "
              f"{b['n_tokens']:>6} {t['n_tokens']:>6}")
    bt = avg([r["gen_time"] for r in base_res])
    tt = avg([r["gen_time"] for r in tuned_res])
    btps = avg([r["tok_per_sec"] for r in base_res])
    ttps = avg([r["tok_per_sec"] for r in tuned_res])
    print("\nAVERAGES")
    print(f"  avg inference time: base {bt:.2f}s | tuned {tt:.2f}s "
          f"-> {(tt-bt)/bt*100:+.1f}%")
    print(f"  avg throughput     : base {btps:.2f} tok/s | tuned {ttps:.2f} tok/s "
          f"-> {(ttps-btps)/btps*100:+.1f}%")
    plot(base_res, tuned_res)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in ("tuned", "base"):
        print("Usage: python eval_fair.py tuned   # run this first")
        print("       python eval_fair.py base    # run after CPU cools")
        sys.exit(1)
    if sys.argv[1] == "tuned":
        phase_tuned()
    else:
        phase_base()
