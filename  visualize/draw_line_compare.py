import json
import matplotlib.pyplot as plt
import numpy as np

BASE_FILE = "base_200.json"
TUNED_FILE = "tuned_200.json"
OUT_FILE = "line_compare_200.png"


def load(path):
    with open(path) as f:
        return json.load(f)


def avg(xs):
    return sum(xs) / len(xs)


def main():
    base = load(BASE_FILE)
    tuned = load(TUNED_FILE)
    n = len(base)
    x = np.arange(1, n + 1)

    base_time = [r["gen_time"] for r in base]
    tuned_time = [r["gen_time"] for r in tuned]
    base_tok = [r["output_tokens"] for r in base]
    tuned_tok = [r["output_tokens"] for r in tuned]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(x, base_time, label="base", color="#888888", linewidth=1.2, alpha=0.8)
    axes[0].plot(x, tuned_time, label="tuned (merged)", color="#4C72B0", linewidth=1.2, alpha=0.8)
    axes[0].axhline(avg(base_time), color="#888888", linestyle="--", alpha=0.6,
                    label=f"base avg={avg(base_time):.2f}s")
    axes[0].axhline(avg(tuned_time), color="#4C72B0", linestyle="--", alpha=0.6,
                    label=f"tuned avg={avg(tuned_time):.2f}s")
    axes[0].fill_between(x, base_time, tuned_time, where=[b >= t for b, t in zip(base_time, tuned_time)],
                         color="#4C72B0", alpha=0.15, label="tuned faster")
    axes[0].set_title("Inference time per question (s) — lower is better")
    axes[0].set_ylabel("seconds")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(x, base_tok, label="base", color="#888888", linewidth=1.2, alpha=0.8)
    axes[1].plot(x, tuned_tok, label="tuned (merged)", color="#4C72B0", linewidth=1.2, alpha=0.8)
    axes[1].axhline(avg(base_tok), color="#888888", linestyle="--", alpha=0.6,
                    label=f"base avg={avg(base_tok):.1f}")
    axes[1].axhline(avg(tuned_tok), color="#4C72B0", linestyle="--", alpha=0.6,
                    label=f"tuned avg={avg(tuned_tok):.1f}")
    axes[1].fill_between(x, base_tok, tuned_tok, where=[b >= t for b, t in zip(base_tok, tuned_tok)],
                         color="#4C72B0", alpha=0.15, label="tuned shorter")
    axes[1].set_title("Output tokens per question — shorter is faster")
    axes[1].set_ylabel("tokens")
    axes[1].set_xlabel("question #")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Base vs fine-tuned (merged) on 200 held-out alpaca questions", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"[+] Line comparison graph saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
