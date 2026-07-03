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
N_QUESTIONS = 20
MAX_NEW_TOKENS = 150
ANSWERS_FILE = "answers_alpaca.txt"
PLOT_FILE = "compare_alpaca_in_dist.png"

BASE_COLOR = "#888888"
TUNED_COLOR = "#4C72B0"


def build_prompt(example):
    instruction = example["instruction"].strip()
    context = (example.get("input") or "").strip()
    if context:
        return f"{instruction}\n\n{context}", example["output"].strip()
    return instruction, example["output"].strip()


def load_held_out():
    ds = load_dataset(DATASET_NAME, split="train").shuffle(seed=42)
    start = min(TRAIN_SKIP, len(ds) - N_QUESTIONS)
    return ds.select(range(start, start + N_QUESTIONS))


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base():
    print(f"[+] Loading base model: {MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    base.config.use_cache = True
    base.eval()
    return base


def wrap_tuned(base):
    print(f"[+] Wrapping with LoRA adapter: {FINAL_DIR}")
    tuned = PeftModel.from_pretrained(base, FINAL_DIR)
    tuned.eval()
    return tuned


def rouge1(gen, ref):
    g = set(gen.lower().split())
    r = set(ref.lower().split())
    if not g or not r:
        return 0.0
    common = g & r
    if not common:
        return 0.0
    p = len(common) / len(g)
    rec = len(common) / len(r)
    return 2 * p * rec / (p + rec)


def run_model(model, tokenizer, examples, label):
    print(f"\n=== Running {label} ===")
    results = []
    for i, ex in enumerate(examples, 1):
        user, gold = build_prompt(ex)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen_time = time.time() - t0

        gen_ids = out[0, prompt_len:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        seq = out[0]
        with torch.no_grad():
            logits = model(seq.unsqueeze(0)).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = seq[1:]
        token_logps = log_probs[0, :-1, :].gather(1, targets.unsqueeze(1)).squeeze(1)
        gen_token_logps = token_logps[prompt_len - 1:]
        mean_logp = gen_token_logps.mean().item()
        ppl = math.exp(-mean_logp) if mean_logp > -20 else float("inf")
        r1 = rouge1(answer, gold)

        results.append({
            "answer": answer,
            "gold": gold,
            "user": user,
            "n_tokens": int(len(gen_ids)),
            "gen_time": gen_time,
            "ppl": ppl,
            "rouge1": r1,
        })
        print(f"  [{i:>2}/{N_QUESTIONS}] {results[-1]['n_tokens']:>3} tok | "
              f"{results[-1]['gen_time']:>5.1f}s | ppl={results[-1]['ppl']:.2f} | "
              f"rouge1={r1:.2f}")
    return results


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def save_answers(base_res, tuned_res, path):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(N_QUESTIONS):
            f.write(f"Q{i+1}: {base_res[i]['user']}\n\n")
            f.write(f"--- GOLD (alpaca) ---\n{base_res[i]['gold']}\n\n")
            f.write(f"--- BASE MODEL ---\n{base_res[i]['answer']}\n\n")
            f.write(f"--- FINE-TUNED ---\n{tuned_res[i]['answer']}\n\n")
            f.write("=" * 70 + "\n\n")
    print(f"[+] Answers saved to {path}")


def plot(base_res, tuned_res, path):
    n = N_QUESTIONS
    x = torch.arange(n)
    w = 0.4
    fig, axes = plt.subplots(4, 1, figsize=(14, 13))

    def grouped(ax, bvals, tvals, title, ylabel, fmt):
        ax.bar(x - w / 2, bvals, w, label="base", color=BASE_COLOR)
        ax.bar(x + w / 2, tvals, w, label="fine-tuned", color=TUNED_COLOR)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(n)])
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        b, t = avg(bvals), avg(tvals)
        ax.axhline(b, color=BASE_COLOR, linestyle="--", alpha=0.6, linewidth=1)
        ax.axhline(t, color=TUNED_COLOR, linestyle="--", alpha=0.6, linewidth=1)
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
            [min(r["ppl"], 50) for r in base_res],
            [min(r["ppl"], 50) for r in tuned_res],
            "Perplexity on own output (lower = more confident)",
            "perplexity", lambda v: f"{v:.2f}")
    grouped(axes[3],
            [r["rouge1"] for r in base_res],
            [r["rouge1"] for r in tuned_res],
            "ROUGE-1 vs gold alpaca answer (higher = closer to reference)",
            "rouge-1 F1", lambda v: f"{v:.2f}")
    axes[3].set_xlabel("example #")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[+] Plot saved to {path}")


def main():
    tokenizer = load_tokenizer()
    examples = list(load_held_out())
    print(f"[+] Held-out alpaca examples (in-distribution, unseen): {len(examples)}")

    base = load_base()
    base_res = run_model(base, tokenizer, examples, "BASE model")
    tuned = wrap_tuned(base)
    tuned_res = run_model(tuned, tokenizer, examples, "FINE-TUNED model")

    save_answers(base_res, tuned_res, ANSWERS_FILE)

    print("\n" + "=" * 74)
    print("SUMMARY (base vs fine-tuned) on in-distribution alpaca")
    print("=" * 74)
    print(f"{'#':>3} {'time_b':>7} {'time_t':>7} {'ppl_b':>6} {'ppl_t':>6} "
          f"{'r1_b':>6} {'r1_t':>6}")
    for i in range(N_QUESTIONS):
        b, t = base_res[i], tuned_res[i]
        print(f"{i+1:>3} {b['gen_time']:>7.1f} {t['gen_time']:>7.1f} "
              f"{b['ppl']:>6.2f} {t['ppl']:>6.2f} {b['rouge1']:>6.2f} {t['rouge1']:>6.2f}")

    bt = avg([r["gen_time"] for r in base_res])
    tt = avg([r["gen_time"] for r in tuned_res])
    bp = avg([r["ppl"] for r in base_res])
    tp = avg([r["ppl"] for r in tuned_res])
    br = avg([r["rouge1"] for r in base_res])
    tr = avg([r["rouge1"] for r in tuned_res])

    print("\nAVERAGES")
    print(f"  avg inference time : base {bt:.2f}s  | tuned {tt:.2f}s  "
          f"-> {(tt-bt)/bt*100:+.1f}%")
    print(f"  avg perplexity     : base {bp:.2f}  | tuned {tp:.2f}  "
          f"-> {(tp-bp)/bp*100:+.1f}%  (negative = tuned more confident)")
    print(f"  avg ROUGE-1 vs gold: base {br:.2f}  | tuned {tr:.2f}  "
          f"-> {(tr-br)/br*100:+.1f}%  (positive = tuned closer to reference)")

    plot(base_res, tuned_res, PLOT_FILE)


if __name__ == "__main__":
    main()
