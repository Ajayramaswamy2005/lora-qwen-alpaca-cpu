import json
import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FINAL_DIR = "qwen-lora-alpaca-final"
MERGED_DIR = "model_tuned"
N = 20
MAX_NEW_TOKENS = 150
WARMUP_PROMPT = "What is 2+2? Answer briefly."
QUESTIONS_FILE = "questions.json"
TUNED_FILE = "merged_tuned_results.json"
BASE_FILE = "merged_base_results.json"
PLOT_FILE = "merged_compare.png"

BASE_COLOR = "#888888"
TUNED_COLOR = "#4C72B0"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_prompt(q):
    instruction = q["instruction"].strip()
    context = (q.get("input") or "").strip()
    user = f"{instruction}\n\n{context}" if context else instruction
    return user, q["output"].strip()


def warmup(model, tokenizer):
    inputs = tokenizer(WARMUP_PROMPT, return_tensors="pt")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)


def run(model, tokenizer, questions, label):
    print(f"\n=== {label} ===")
    print("[+] warmup generation (steady thermal state)...")
    warmup(model, tokenizer)
    results = []
    for i, q in enumerate(questions, 1):
        user, gold = build_prompt(q)
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


def load_json(path):
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
        ax.bar(x + w/2, tvals, w, label="tuned (merged)", color=TUNED_COLOR)
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


def compare():
    base_res = load_json(BASE_FILE)
    tuned_res = load_json(TUNED_FILE)
    print("\n" + "=" * 74)
    print("FAIR COMPARISON  base  vs  tuned-MERGED (no adapter overhead)")
    print("=" * 74)
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
    print(f"  avg inference time: base {bt:.2f}s | tuned(merged) {tt:.2f}s "
          f"-> {(tt-bt)/bt*100:+.1f}%")
    print(f"  avg throughput     : base {btps:.2f} tok/s | tuned(merged) {ttps:.2f} tok/s "
          f"-> {(ttps-btps)/btps*100:+.1f}%")
    plot(base_res, tuned_res)


def phase_merge():
    tok = load_tokenizer()
    print(f"[+] Loading base + LoRA adapter: {FINAL_DIR}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, FINAL_DIR)
    print("[+] Merging LoRA weights into base model...")
    merged = model.merge_and_unload()
    print(f"[+] Saving merged model to {MERGED_DIR}/")
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    print("[+] Merge complete. model_tuned/ is a standalone full model.")


def phase_tuned():
    tok = load_tokenizer()
    questions = load_json(QUESTIONS_FILE)
    print(f"[+] Loading base + LoRA: {FINAL_DIR}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(base, FINAL_DIR)
    print("[+] Merging LoRA into base (in-memory, no disk save)...")
    model = peft_model.merge_and_unload()
    model.config.use_cache = True
    model.eval()
    print("[+] Merge complete. Architecture is now identical to base (no adapter).")
    res = run(model, tok, questions, "TUNED (merged) model")
    save(res, TUNED_FILE)
    print("\n[+] Phase tuned done. Let CPU cool, then run:")
    print("    uv run python eval_merged.py base")


def phase_base():
    tok = load_tokenizer()
    questions = load_json(QUESTIONS_FILE)
    print(f"[+] Loading base: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.config.use_cache = True
    model.eval()
    res = run(model, tok, questions, "BASE model")
    save(res, BASE_FILE)
    compare()


if __name__ == "__main__":
    import sys
    cmds = {"tuned": phase_tuned, "base": phase_base}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("Usage:")
        print("  python eval_merged.py tuned   # merge LoRA in-memory + benchmark")
        print("  python eval_merged.py base    # benchmark base + compare")
        sys.exit(1)
    cmds[sys.argv[1]]()
