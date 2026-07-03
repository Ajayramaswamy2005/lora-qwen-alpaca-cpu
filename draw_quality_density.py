import json
import numpy as np
import matplotlib.pyplot as plt

BASE_FILE = "results/base_200.json"
TUNED_FILE = "results/tuned_200.json"
OUT_FILE = "results/quality_density.png"


def rouge1(hyp, ref):
    h = set(hyp.lower().split())
    r = set(ref.lower().split())
    if not h or not r:
        return 0.0
    common = h & r
    p = len(common) / len(h)
    rec = len(common) / len(r)
    if p + rec == 0:
        return 0.0
    return 2 * p * rec / (p + rec)


def lcs_len(a, b):
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def rouge_l(hyp, ref):
    h = hyp.lower().split()
    r = ref.lower().split()
    ll = lcs_len(h, r)
    if not h or not r or ll == 0:
        return 0.0
    p = ll / len(h)
    rec = ll / len(r)
    if p + rec == 0:
        return 0.0
    return 2 * p * rec / (p + rec)


def kde(data, x_grid, bandwidth=0.05):
    data = np.array(data)
    n = len(data)
    contributions = np.exp(-0.5 * ((x_grid[:, None] - data[None, :]) / bandwidth) ** 2)
    return contributions.sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))


def avg(xs):
    return sum(xs) / len(xs)


def main():
    base = json.load(open(BASE_FILE))
    tuned = json.load(open(TUNED_FILE))

    base_r1 = np.array([rouge1(b["answer"], b["gold"]) for b in base])
    tuned_r1 = np.array([rouge1(t["answer"], t["gold"]) for t in tuned])
    base_rl = np.array([rouge_l(b["answer"], b["gold"]) for b in base])
    tuned_rl = np.array([rouge_l(t["answer"], t["gold"]) for t in tuned])

    x = np.linspace(0, 1, 500)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def plot_density(ax, bvals, tvals, title):
        ax.fill_between(x, kde(bvals, x), alpha=0.3, color="#888888", label="base")
        ax.fill_between(x, kde(tvals, x), alpha=0.3, color="#4C72B0", label="tuned")
        ax.plot(x, kde(bvals, x), color="#888888", linewidth=2)
        ax.plot(x, kde(tvals, x), color="#4C72B0", linewidth=2)
        ax.axvline(avg(bvals), color="#888888", linestyle="--", linewidth=1.5,
                   label=f"base avg={avg(bvals):.3f}")
        ax.axvline(avg(tvals), color="#4C72B0", linestyle="--", linewidth=1.5,
                   label=f"tuned avg={avg(tvals):.3f}")
        ax.set_title(title)
        ax.set_xlabel("ROUGE score")
        ax.set_ylabel("density (relative likelihood)")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plot_density(axes[0], base_r1, tuned_r1, "ROUGE-1 score likelihood")
    plot_density(axes[1], base_rl, tuned_rl, "ROUGE-L score likelihood")

    fig.suptitle("Quality score density: base vs fine-tuned (200 held-out questions)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"[+] Quality density graph saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
