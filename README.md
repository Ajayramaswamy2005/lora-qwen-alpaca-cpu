# LoRA Fine-Tuning on CPU: Qwen2.5-0.5B-Instruct with alpaca-cleaned

This project fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` on the `yahma/alpaca-cleaned` instruction-following dataset using LoRA, entirely on a CPU. It includes scripts to train the adapter, merge it into the base model, and benchmark the merged model against the original base on held-out data.

## What was done

1. **LoRA fine-tuning** (`finetune.py`)  
   Fine-tuned the base instruct model on 2,000 samples from `yahma/alpaca-cleaned` using LoRA. Only the attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) were adapted. Trainable parameters: ~1.08M out of 495M (0.22%).

2. **Adapter merge for fair inference** (`eval_200.py`)  
   The LoRA adapter was merged into the base weights (`merge_and_unload`) before benchmarking. This removes the adapter overhead and lets the fine-tuned model run at the same architecture/speed as the base model during inference.

3. **200-question held-out benchmark** (`eval_200.py`)  
   200 random examples were sampled from the alpaca dataset **after removing the first 2,000 rows used for training** (same shuffle seed as training, then a separate random sample from the remainder). Each question was run through both the base model and the merged fine-tuned model. Metrics logged per question: input tokens, output tokens, inference time, throughput.

4. **Answer quality comparison** (`compare_quality.py`)  
   Each generated answer was compared to the gold alpaca answer using ROUGE-1 and ROUGE-L to see which model produced better-matched responses.

## Scripts

| Script | Purpose |
|---|---|
| `finetune.py` | Train the LoRA adapter on alpaca-cleaned |
| `eval_200.py` | Run 200-question benchmark: `python eval_200.py tuned` then `python eval_200.py base` |
| `draw_line_compare.py` | Generate line-plot comparison |
| `draw_scatter_compare.py` | Generate scatter-plot comparison |
| `compare_quality.py` | Compare answer quality vs gold answers (ROUGE-1 / ROUGE-L) |

## Key results (200 held-out questions)

| Metric | Base model | Fine-tuned (merged) | Change |
|---|---|---|---|
| Avg inference time | 6.62 s | **5.19 s** | **−21.6%** |
| Avg input tokens | 46.9 | 46.9 | 0.0% |
| Avg output tokens | 112.6 | **94.5** | **−16.1%** |
| Avg throughput | 17.20 tok/s | **18.28 tok/s** | **+6.2%** |

The merged fine-tuned model is **21.6% faster end-to-end**. The speedup comes mainly from generating **shorter, more concise responses** (−16.1% output tokens), with a small additional gain in per-token throughput (+6.2%).

## Answer quality (200 held-out questions vs gold alpaca answers)

| Metric | Base model | Fine-tuned (merged) | Change |
|---|---|---|---|
| ROUGE-1 wins | 71 | **117** | tuned wins 58.5% |
| ROUGE-L wins | 65 | **120** | tuned wins 60.0% |
| Avg ROUGE-1 | 0.319 | **0.363** | **+13.8%** |
| Avg ROUGE-L | 0.228 | **0.269** | **+18.0%** |

The fine-tuned model produces answers that match the gold alpaca reference style **significantly better**: it wins on ~60% of questions and improves average ROUGE-L by 18.0% and ROUGE-1 by 13.8%.

## Combined takeaway

On 200 held-out alpaca questions:

- **Faster:** 21.6% lower inference time, driven by 16.1% shorter outputs.
- **Better quality:** ~60% win rate vs the base model on ROUGE-L, with ~14–18% higher average scores.

So the fine-tuned model is both faster and better-aligned with the target instruction-following style.

## Additional files

- `qwen-lora-alpaca-final/` — saved LoRA adapter
- `results/test_200.json` — the 200 held-out questions
- `results/base_200.json` / `results/tuned_200.json` — per-question results
- `results/compare_200.png` / `results/summary_compare_200.png` / `results/line_compare_200.png` / `results/scatter_compare_200.png` — comparison graphs
- `results/per_question_quality.txt` — per-question quality comparison (ROUGE-1 / ROUGE-L)
- `results/per_question_compare.txt` — per-question time/token comparison

## How to reproduce

Install dependencies (this repo uses `uv`):

```bash
uv sync
```

Train the adapter (optional if adapter already saved):

```bash
uv run python finetune.py
```

Run the 200-question benchmark:

```bash
# Phase 1: merged fine-tuned model
uv run python eval_200.py tuned

# Let CPU cool, then run base model comparison
uv run python eval_200.py base
```

Generate additional comparison graphs and quality report:

```bash
uv run python draw_line_compare.py
uv run python draw_scatter_compare.py
uv run python compare_quality.py
```

## Environment

- CPU: Intel Core Ultra 5 125H (18 logical cores)
- RAM: 30 GB
- Python 3.12
- PyTorch, Transformers, PEFT, Datasets, Accelerate, Matplotlib
- All training and inference ran on CPU only.
