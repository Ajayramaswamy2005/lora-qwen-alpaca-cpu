import math
import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FINAL_DIR = "qwen-lora-alpaca-final"
MAX_NEW_TOKENS = 150
ANSWERS_FILE = "answers.txt"
PLOT_FILE = "compare_base_vs_tuned.png"

QUESTIONS = [
    "If a train leaves station A at 60 mph and another leaves station B at 40 mph, 200 miles apart, heading toward each other, when and where do they meet?",
    "Three guests pay $30 for a $25 room. The bellhop keeps $2 and returns $1 to each guest. Each paid $9 = $27, plus $2 = $29. Where is the missing dollar? Explain the flaw.",
    "Write a Python function that returns the nth Fibonacci number using memoization and state its time complexity.",
    "You have 3 apples and take away 2. How many apples do you have? Explain your reasoning.",
    "Explain why the sky is blue at noon and red/orange at sunset, in terms of light scattering.",
    "A trolley is heading toward 5 people; you can pull a lever to divert it to kill 1. Should you? Compare utilitarian and deontological reasoning.",
    "What is the difference between TCP and UDP? Give two realistic use cases for each.",
    "Find the derivative of f(x) = x^3 * ln(x). Show every step.",
    "A man pushes his car to a hotel and tells the owner he is bankrupt. Why?",
    "Write one sentence that contains exactly three words starting with 'b' and also explains photosynthesis.",
    "Explain what a closure is in JavaScript with a minimal code example.",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets? Explain.",
    "Explain the difference between mitosis and meiosis and why meiosis matters for sexual reproduction.",
    "Explain inflation and why central banks often target around 2% rather than 0%.",
    "What were the main long-term and short-term causes of World War I?",
    "Evaluate 8 / 2(2+2). Explain the ambiguity and the two common interpretations.",
    "All cats are animals. Some animals are pets. Can we conclude that some cats are pets? Why or why not?",
    "Explain the difference between a process and a thread, including memory isolation.",
    "Write a 4-line rhyming poem about entropy.",
    "Why does ice float on liquid water? Relate it to hydrogen bonding and density.",
]

BASE_COLOR = "#888888"
TUNED_COLOR = "#4C72B0"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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


def run_model(model, tokenizer, label):
    print(f"\n=== Running {label} ===")
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
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

        results.append({
            "answer": answer,
            "n_tokens": int(len(gen_ids)),
            "gen_time": gen_time,
            "mean_logp": mean_logp,
            "ppl": ppl,
        })
        print(f"  [{i:>2}/{len(QUESTIONS)}] {r_str(results[-1])}")
    return results


def r_str(r):
    return f"{r['n_tokens']:>3} tok | {r['gen_time']:>5.1f}s | ppl={r['ppl']:.2f}"


def save_answers(base_res, tuned_res, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, q in enumerate(QUESTIONS, 1):
            f.write(f"Q{i}: {q}\n\n")
            f.write(f"--- BASE MODEL ---\n{base_res[i-1]['answer']}\n\n")
            f.write(f"--- FINE-TUNED ---\n{tuned_res[i-1]['answer']}\n\n")
            f.write("=" * 70 + "\n\n")
    print(f"[+] Answers saved to {path}")


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def plot(base_res, tuned_res, path):
    n = len(QUESTIONS)
    x = torch.arange(n)
    w = 0.4

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    def grouped(ax, base_vals, tuned_vals, title, ylabel, fmt):
        ax.bar(x - w/2, base_vals, w, label="base", color=BASE_COLOR)
        ax.bar(x + w/2, tuned_vals, w, label="fine-tuned", color=TUNED_COLOR)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(n)])
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        b, t = avg(base_vals), avg(tuned_vals)
        ax.axhline(b, color=BASE_COLOR, linestyle="--", alpha=0.6, linewidth=1)
        ax.axhline(t, color=TUNED_COLOR, linestyle="--", alpha=0.6, linewidth=1)
        ax.text(n - 0.5, max(b, t), f"avg base={fmt(b)}  tuned={fmt(t)}",
                ha="right", va="bottom", fontsize=8)

    grouped(axes[0],
            [r["gen_time"] for r in base_res],
            [r["gen_time"] for r in tuned_res],
            "Inference time per question (s) — lower is better",
            "seconds", lambda v: f"{v:.1f}s")
    grouped(axes[1],
            [r["n_tokens"] for r in base_res],
            [r["n_tokens"] for r in tuned_res],
            "Response length per question (tokens)",
            "tokens", lambda v: f"{v:.0f}")
    grouped(axes[2],
            [min(r["ppl"], 50) for r in base_res],
            [min(r["ppl"], 50) for r in tuned_res],
            "Perplexity on own output — lower = more confident",
            "perplexity", lambda v: f"{v:.2f}")
    axes[2].set_xlabel("question #")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[+] Plot saved to {path}")


def main():
    tokenizer = load_tokenizer()
    base = load_base()

    base_res = run_model(base, tokenizer, "BASE model")
    tuned = wrap_tuned(base)
    tuned_res = run_model(tuned, tokenizer, "FINE-TUNED model")

    save_answers(base_res, tuned_res, ANSWERS_FILE)

    print("\n" + "=" * 70)
    print("SUMMARY  (base  vs  fine-tuned)")
    print("=" * 70)
    print(f"{'#':>3} {'time_b':>8} {'time_t':>8} {'tok_b':>6} {'tok_t':>6} {'ppl_b':>7} {'ppl_t':>7}")
    for i in range(len(QUESTIONS)):
        b, t = base_res[i], tuned_res[i]
        print(f"{i+1:>3} {b['gen_time']:>8.1f} {t['gen_time']:>8.1f} "
              f"{b['n_tokens']:>6} {t['n_tokens']:>6} {b['ppl']:>7.2f} {t['ppl']:>7.2f}")

    bt = avg([r["gen_time"] for r in base_res])
    tt = avg([r["gen_time"] for r in tuned_res])
    bp = avg([r["ppl"] for r in base_res])
    tp = avg([r["ppl"] for r in tuned_res])
    btk = avg([r["n_tokens"] for r in base_res])
    ttk = avg([r["n_tokens"] for r in tuned_res])

    print("\nAVERAGES")
    print(f"  avg inference time : base {bt:.2f}s  | tuned {tt:.2f}s  "
          f"-> {(tt-bt)/bt*100:+.1f}%")
    print(f"  avg response tokens: base {btk:.1f}  | tuned {ttk:.1f}  "
          f"-> {(ttk-btk)/btk*100:+.1f}%")
    print(f"  avg perplexity     : base {bp:.2f}  | tuned {tp:.2f}  "
          f"-> {(tp-bp)/bp*100:+.1f}%")

    plot(base_res, tuned_res, PLOT_FILE)


if __name__ == "__main__":
    main()
