import json
import matplotlib.pyplot as plt

BASE_FILE = "base_200.json"
TUNED_FILE = "tuned_200.json"
OUT_FILE = "summary_compare_200.png"


def load(path):
    with open(path) as f:
        return json.load(f)


def avg(xs):
    return sum(xs) / len(xs)


def main():
    base = load(BASE_FILE)
    tuned = load(TUNED_FILE)

    labels = ["Avg time (s)", "Avg output tokens", "Avg throughput\n(tok/s)"]
    base_vals = [
        avg([r["gen_time"] for r in base]),
        avg([r["output_tokens"] for r in base]),
        avg([r["tok_per_sec"] for r in base]),
    ]
    tuned_vals = [
        avg([r["gen_time"] for r in tuned]),
        avg([r["output_tokens"] for r in tuned]),
        avg([r["tok_per_sec"] for r in tuned]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = ["#888888", "#4C72B0"]
    for i, (label, bv, tv) in enumerate(zip(labels, base_vals, tuned_vals)):
        bars = axes[i].bar(["base", "tuned(merged)"], [bv, tv], color=colors)
        axes[i].set_title(label)
        axes[i].set_ylabel(label.split("(")[0].strip().lower())
        axes[i].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, [bv, tv]):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Base vs fine-tuned (merged) — 200 held-out alpaca questions", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"[+] Summary graph saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
