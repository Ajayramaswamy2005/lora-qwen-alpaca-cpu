import json
import time
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FINAL_DIR = "qwen-lora-alpaca-final"
DATASET_NAME = "yahma/alpaca-cleaned"
TRAIN_SAMPLES = 2000
N = 200
MAX_NEW_TOKENS = 150
WARMUP_PROMPT = "What is 2+2? Answer briefly."
QUESTIONS_FILE = "test_200.json"
TUNED_FILE = "tuned_200.json"
BASE_FILE = "base_200.json"
PLOT_FILE = "compare_200.png"

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


def sample_test_questions():
    ds = load_dataset(DATASET_NAME, split="train").shuffle(seed=42)
    remaining = ds.select(range(TRAIN_SAMPLES, len(ds)))
    sampled = remaining.shuffle(seed=123).select(range(N))
    return [dict(sampled[i]) for i in range(N)]


def warmup(model, tokenizer):
    inputs = tokenizer(WARMUP_PROMPT, return_tensors="pt")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)


def run(model, tokenizer, questions, label, out_file):
    print(f"\n=== {label} === ({len(questions)} questions)")
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
        n_out = int(len(gen_ids))
        rec = {
            "user": user, "gold": gold, "answer": answer,
            "input_tokens": prompt_len,
            "output_tokens": n_out,
            "gen_time": gen_time,
            "tok_per_sec": n_out / gen_time if gen_time > 0 else 0,
        }
        results.append(rec)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        if i % 10 == 0 or i == len(questions):
            print(f"  [{i:>3}/{len(questions)}] avg_time={sum(r['gen_time'] for r in results)/len(results):.1f}s "
                  f"avg_out_tok={sum(r['output_tokens'] for r in results)/len(results):.0f} "
                  f"avg_tps={sum(r['tok_per_sec'] for r in results)/len(results):.1f}")
    return results


def load_json(path):
    with open(path) as f:
        return json.load(f)


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def compare():
    base_res = load_json(BASE_FILE)
    tuned_res = load_json(TUNED_FILE)
    print("\n" + "=" * 80)
    print(f"COMPARISON  base vs tuned(merged)  —  {len(base_res)} held-out questions")
    print("=" * 80)

    bt = avg([r["gen_time"] for r in base_res])
    tt = avg([r["gen_time"] for r in tuned_res])
    bi = avg([r["input_tokens"] for r in base_res])
    ti = avg([r["input_tokens"] for r in tuned_res])
    bo = avg([r["output_tokens"] for r in base_res])
    to = avg([r["output_tokens"] for r in tuned_res])
    btps = avg([r["tok_per_sec"] for r in base_res])
    ttps = avg([r["tok_per_sec"] for r in tuned_res])

    print(f"\n{'metric':<28} {'base':>12} {'tuned(merged)':>14} {'delta':>10}")
    print("-" * 66)
    print(f"{'avg time (s)':<28} {bt:>12.2f} {tt:>14.2f} {(tt-bt)/bt*100:>+9.1f}%")
    print(f"{'avg input tokens':<28} {bi:>12.1f} {ti:>14.1f} {(ti-bi)/bi*100:>+9.1f}%")
    print(f"{'avg output tokens':<28} {bo:>12.1f} {to:>14.1f} {(to-bo)/bo*100:>+9.1f}%")
    print(f"{'avg throughput (tok/s)':<28} {btps:>12.2f} {ttps:>14.2f} {(ttps-btps)/btps*100:>+9.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = [
        ("gen_time", "Inference time (s)", axes[0, 0]),
        ("tok_per_sec", "Throughput (tokens/sec)", axes[0, 1]),
        ("output_tokens", "Output tokens", axes[1, 0]),
        ("input_tokens", "Input tokens", axes[1, 1]),
    ]
    for key, title, ax in metrics:
        bvals = [r[key] for r in base_res]
        tvals = [r[key] for r in tuned_res]
        lo = min(min(bvals), min(tvals))
        hi = max(max(bvals), max(tvals))
        bins = 30
        ax.hist(bvals, bins=bins, alpha=0.6, color=BASE_COLOR, label="base")
        ax.hist(tvals, bins=bins, alpha=0.6, color=TUNED_COLOR, label="tuned (merged)")
        ax.axvline(avg(bvals), color=BASE_COLOR, linestyle="--", linewidth=1.5)
        ax.axvline(avg(tvals), color=TUNED_COLOR, linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)
    print(f"\n[+] Plot saved to {PLOT_FILE}")


def phase_tuned():
    tok = load_tokenizer()
    questions = sample_test_questions()
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"[+] {len(questions)} held-out questions saved to {QUESTIONS_FILE}")
    print(f"[+] Loading base + LoRA: {FINAL_DIR}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(base, FINAL_DIR)
    print("[+] Merging LoRA into base (in-memory)...")
    model = peft_model.merge_and_unload()
    model.config.use_cache = True
    model.eval()
    print("[+] Merge complete. No adapter overhead.")
    run(model, tok, questions, "TUNED (merged)", TUNED_FILE)
    print(f"\n[+] Phase tuned done. Results in {TUNED_FILE}.")
    print("[+] Let the CPU cool, then run:")
    print("    uv run python eval_200.py base")


def phase_base():
    tok = load_tokenizer()
    questions = load_json(QUESTIONS_FILE)
    print(f"[+] Loading base: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.config.use_cache = True
    model.eval()
    run(model, tok, questions, "BASE", BASE_FILE)
    compare()


if __name__ == "__main__":
    import sys
    cmds = {"tuned": phase_tuned, "base": phase_base}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("Usage:")
        print("  python eval_200.py tuned   # merge + benchmark 200 questions")
        print("  python eval_200.py base    # benchmark base + compare")
        sys.exit(1)
    cmds[sys.argv[1]]()
