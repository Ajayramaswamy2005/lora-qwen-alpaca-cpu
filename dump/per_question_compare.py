import json

BASE_FILE = "results/base_200.json"
TUNED_FILE = "results/tuned_200.json"
OUT_FILE = "results/per_question_compare.txt"


def trunc(s, n=200):
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def pct_change(new, old):
    return (new - old) / old * 100 if old != 0 else 0.0


def main():
    base = json.load(open(BASE_FILE))
    tuned = json.load(open(TUNED_FILE))
    assert len(base) == len(tuned)
    n = len(base)

    rows = []
    for i in range(n):
        b, t = base[i], tuned[i]
        time_saved = b["gen_time"] - t["gen_time"]
        tok_saved = b["output_tokens"] - t["output_tokens"]
        rows.append({
            "idx": i + 1,
            "user": b["user"],
            "base_time": b["gen_time"],
            "tuned_time": t["gen_time"],
            "time_saved": time_saved,
            "time_pct": pct_change(t["gen_time"], b["gen_time"]),
            "base_tok": b["output_tokens"],
            "tuned_tok": t["output_tokens"],
            "tok_saved": tok_saved,
            "tok_pct": pct_change(t["output_tokens"], b["output_tokens"]),
            "base_tps": b["tok_per_sec"],
            "tuned_tps": t["tok_per_sec"],
            "base_ans": b["answer"],
            "tuned_ans": t["answer"],
        })

    faster = sum(1 for r in rows if r["time_saved"] > 0)
    shorter = sum(1 for r in rows if r["tok_saved"] > 0)
    avg_time_saved = sum(r["time_saved"] for r in rows) / n
    avg_tok_saved = sum(r["tok_saved"] for r in rows) / n
    max_time_saved = max(r["time_saved"] for r in rows)
    max_tok_saved = max(r["tok_saved"] for r in rows)
    max_time_lost = min(r["time_saved"] for r in rows)
    max_tok_lost = min(r["tok_saved"] for r in rows)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("PER-QUESTION COMPARISON: base vs fine-tuned (merged)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Total questions: {n}\n")
        f.write(f"Tuned faster: {faster}/{n} ({faster/n*100:.1f}%)\n")
        f.write(f"Tuned shorter: {shorter}/{n} ({shorter/n*100:.1f}%)\n")
        f.write(f"Avg time saved: {avg_time_saved:.2f}s\n")
        f.write(f"Avg tokens saved: {avg_tok_saved:.1f}\n")
        f.write(f"Max time saved: {max_time_saved:.2f}s\n")
        f.write(f"Max time lost: {max_time_lost:.2f}s\n")
        f.write(f"Max tokens saved: {max_tok_saved}\n")
        f.write(f"Max tokens added: {max_tok_lost}\n")
        f.write("\n" + "=" * 90 + "\n\n")

        for r in rows:
            f.write(f"Q{r['idx']:03d}\n")
            f.write(f"Instruction: {trunc(r['user'], 200)}\n")
            f.write(f"Base   — time={r['base_time']:>6.2f}s  out_tok={r['base_tok']:>4}  tps={r['base_tps']:>5.1f}\n")
            f.write(f"Tuned  — time={r['tuned_time']:>6.2f}s  out_tok={r['tuned_tok']:>4}  tps={r['tuned_tps']:>5.1f}\n")
            f.write(f"Delta  — time={r['time_saved']:>+6.2f}s ({r['time_pct']:>+5.1f}%)  "
                    f"tok={r['tok_saved']:>+4} ({r['tok_pct']:>+5.1f}%)\n")
            f.write(f"Base answer:   {trunc(r['base_ans'], 250)}\n")
            f.write(f"Tuned answer:  {trunc(r['tuned_ans'], 250)}\n")
            f.write("-" * 90 + "\n\n")

    print(f"[+] Per-question comparison saved to {OUT_FILE}")
    print(f"\nSummary ({n} questions):")
    print(f"  Tuned faster : {faster}/{n} ({faster/n*100:.1f}%)")
    print(f"  Tuned shorter: {shorter}/{n} ({shorter/n*100:.1f}%)")
    print(f"  Avg time saved : {avg_time_saved:.2f}s")
    print(f"  Avg tokens saved: {avg_tok_saved:.1f}")
    print(f"  Best speedup   : Q{[r['idx'] for r in rows if r['time_saved']==max_time_saved][0]:03d} saved {max_time_saved:.2f}s")
    print(f"  Worst slowdown : Q{[r['idx'] for r in rows if r['time_saved']==max_time_lost][0]:03d} lost {-max_time_lost:.2f}s")

    # Print top 10 fastest-improved questions
    print("\nTop 10 questions where tuned saved the most time:")
    top = sorted(rows, key=lambda r: -r["time_saved"])[:10]
    for r in top:
        print(f"  Q{r['idx']:03d}: saved {r['time_saved']:.2f}s  "
              f"({r['base_time']:.1f}s -> {r['tuned_time']:.1f}s)  "
              f"tok {r['base_tok']} -> {r['tuned_tok']}")


if __name__ == "__main__":
    main()
