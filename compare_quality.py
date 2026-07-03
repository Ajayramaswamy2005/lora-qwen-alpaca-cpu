import json

BASE_FILE = "results/base_200.json"
TUNED_FILE = "results/tuned_200.json"
OUT_FILE = "results/per_question_quality.txt"


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


def trunc(s, n=250):
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def main():
    base = json.load(open(BASE_FILE))
    tuned = json.load(open(TUNED_FILE))
    n = len(base)

    rows = []
    for i in range(n):
        b, t = base[i], tuned[i]
        gold = b["gold"]
        b_r1, b_rl = rouge1(b["answer"], gold), rouge_l(b["answer"], gold)
        t_r1, t_rl = rouge1(t["answer"], gold), rouge_l(t["answer"], gold)
        rows.append({
            "idx": i + 1,
            "user": b["user"],
            "gold": gold,
            "base_ans": b["answer"],
            "tuned_ans": t["answer"],
            "base_r1": b_r1,
            "tuned_r1": t_r1,
            "base_rl": b_rl,
            "tuned_rl": t_rl,
            "r1_winner": "tuned" if t_r1 > b_r1 else ("base" if b_r1 > t_r1 else "tie"),
            "rl_winner": "tuned" if t_rl > b_rl else ("base" if b_rl > t_rl else "tie"),
        })

    r1_tuned_wins = sum(1 for r in rows if r["r1_winner"] == "tuned")
    r1_base_wins = sum(1 for r in rows if r["r1_winner"] == "base")
    r1_ties = sum(1 for r in rows if r["r1_winner"] == "tie")
    rl_tuned_wins = sum(1 for r in rows if r["rl_winner"] == "tuned")
    rl_base_wins = sum(1 for r in rows if r["rl_winner"] == "base")
    rl_ties = sum(1 for r in rows if r["rl_winner"] == "tie")

    avg_b_r1 = sum(r["base_r1"] for r in rows) / n
    avg_t_r1 = sum(r["tuned_r1"] for r in rows) / n
    avg_b_rl = sum(r["base_rl"] for r in rows) / n
    avg_t_rl = sum(r["tuned_rl"] for r in rows) / n

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("PER-QUESTION QUALITY COMPARISON: base vs fine-tuned (merged)\n")
        f.write("Metric: ROUGE-1 / ROUGE-L vs gold alpaca answer (higher = better match)\n")
        f.write("=" * 90 + "\n\n")

        f.write("OVERALL STATS\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total questions: {n}\n\n")
        f.write(f"ROUGE-1 wins:  tuned={r1_tuned_wins}  base={r1_base_wins}  ties={r1_ties}\n")
        f.write(f"ROUGE-L wins:  tuned={rl_tuned_wins}  base={rl_base_wins}  ties={rl_ties}\n\n")
        f.write(f"Avg ROUGE-1:  base={avg_b_r1:.3f}  tuned={avg_t_r1:.3f}  "
                f"delta={avg_t_r1-avg_b_r1:+.3f}\n")
        f.write(f"Avg ROUGE-L:  base={avg_b_rl:.3f}  tuned={avg_t_rl:.3f}  "
                f"delta={avg_t_rl-avg_b_rl:+.3f}\n\n")

        # Top examples where tuned improved the most on ROUGE-L
        top_improved = sorted(rows, key=lambda r: r["tuned_rl"] - r["base_rl"], reverse=True)[:10]
        f.write("TOP 10 QUESTIONS WHERE TUNED IMPROVED MOST (ROUGE-L)\n")
        f.write("-" * 90 + "\n")
        for r in top_improved:
            f.write(f"Q{r['idx']:03d}: base_rl={r['base_rl']:.3f} -> tuned_rl={r['tuned_rl']:.3f}  "
                    f"(+{r['tuned_rl']-r['base_rl']:.3f})\n")
            f.write(f"  Instruction: {trunc(r['user'], 200)}\n")
            f.write(f"  Gold:        {trunc(r['gold'], 200)}\n")
            f.write(f"  Base:        {trunc(r['base_ans'], 200)}\n")
            f.write(f"  Tuned:       {trunc(r['tuned_ans'], 200)}\n\n")

        # Top examples where tuned degraded the most
        top_degraded = sorted(rows, key=lambda r: r["tuned_rl"] - r["base_rl"])[:10]
        f.write("\nTOP 10 QUESTIONS WHERE TUNED DEGRADED MOST (ROUGE-L)\n")
        f.write("-" * 90 + "\n")
        for r in top_degraded:
            f.write(f"Q{r['idx']:03d}: base_rl={r['base_rl']:.3f} -> tuned_rl={r['tuned_rl']:.3f}  "
                    f"({r['tuned_rl']-r['base_rl']:+.3f})\n")
            f.write(f"  Instruction: {trunc(r['user'], 200)}\n")
            f.write(f"  Gold:        {trunc(r['gold'], 200)}\n")
            f.write(f"  Base:        {trunc(r['base_ans'], 200)}\n")
            f.write(f"  Tuned:       {trunc(r['tuned_ans'], 200)}\n\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("ALL QUESTIONS (summary line per question)\n")
        f.write("=" * 90 + "\n\n")
        for r in rows:
            f.write(f"Q{r['idx']:03d} | R1 base={r['base_r1']:.3f} tuned={r['tuned_r1']:.3f} "
                    f"winner={r['r1_winner']:5s} | "
                    f"RL base={r['base_rl']:.3f} tuned={r['tuned_rl']:.3f} "
                    f"winner={r['rl_winner']:5s}\n")
            f.write(f"      Instruction: {trunc(r['user'], 180)}\n\n")

    print(f"[+] Quality comparison saved to {OUT_FILE}")
    print("\nOVERALL QUALITY STATS (vs gold alpaca answer):")
    print(f"  ROUGE-1 wins: tuned {r1_tuned_wins} | base {r1_base_wins} | ties {r1_ties}")
    print(f"  ROUGE-L wins: tuned {rl_tuned_wins} | base {rl_base_wins} | ties {rl_ties}")
    print(f"  Avg ROUGE-1: base={avg_b_r1:.3f} tuned={avg_t_r1:.3f} delta={avg_t_r1-avg_b_r1:+.3f}")
    print(f"  Avg ROUGE-L: base={avg_b_rl:.3f} tuned={avg_t_rl:.3f} delta={avg_t_rl-avg_b_rl:+.3f}")


if __name__ == "__main__":
    main()
