import json
import matplotlib.pyplot as plt
import numpy as np

BASE_FILE = "base_200.json"
TUNED_FILE = "tuned_200.json"
OUT_FILE = "scatter_compare_200.png"


def load(path):
    with open(path) as f:
        return json.load(f)


def avg(xs):
    return sum(xs) / len(xs)


def main():
    base = load(BASE_FILE)
    tuned = load(TUNED_FILE)

    base_time = [r["gen_time"] for r in base]
    tuned_time = [r["gen_time"] for r in tuned]
    base_tok = [r["output_tokens"] for r in base]
    tuned_tok = [r["output_tokens"] for r in tuned]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(base_time, tuned_time, alpha=0.5, s=25, color="#4C72B0", edgecolors="none")
    lo, hi = min(min(base_time), min(tuned_time)), max(max(base_time), max(tuned_time))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x (same speed)")
    ax.axvline(avg(base_time), color="#888888", linestyle=":", alpha=0.7)
    ax.axhline(avg(tuned_time), color="#4C72B0", linestyle=":", alpha=0.7)
    ax.set_xlabel("base time (s)")
    ax.set_ylabel("tuned time (s)")
    ax.set_title("Inference time per question")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(base_tok, tuned_tok, alpha=0.5, s=25, color="#55A868", edgecolors="none")
    lo, hi = min(min(base_tok), min(tuned_tok)), max(max(base_tok), max(tuned_tok))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x (same length)")
    ax.axvline(avg(base_tok), color="#888888", linestyle=":", alpha=0.7)
    ax.axhline(avg(tuned_tok), color="#55A868", linestyle=":", alpha=0.7)
    ax.set_xlabel("base output tokens")
    ax.set_ylabel("tuned output tokens")
    ax.set_title("Output tokens per question")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Scatter comparison: base vs fine-tuned (merged) — 200 held-out questions",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"[+] Scatter comparison graph saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
