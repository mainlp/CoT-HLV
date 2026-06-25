# Decoupling Chain-of-Thought Reasoning for Human Label Variation

This repository contains reusable code for the experiments in **"Decoupling the
Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective"**.
The project studies whether long Chain-of-Thought (CoT) reasoning helps large
language models approximate human label distributions on ambiguous NLI examples.

The code supports three workflows:

1. Generate CoT traces for ChaosNLI examples.
2. Split each CoT into cumulative prefixes and inject those prefixes into other
   models for Cross-CoT / step-wise logit tracing.
3. Evaluate first-token option logits against human judgment distributions with
   accuracy, Jensen-Shannon distance, Spearman correlation, and additive ANOVA.

No data, model checkpoints, or generated traces are included in this repository.
See [docs/data.md](docs/data.md) for dataset sources and expected file names.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install PyTorch according to your CUDA environment before running large-model
experiments if the default wheel is not suitable.

## Data

Download ChaosNLI from the official release:

https://github.com/easonnie/ChaosNLI

Place the files in a local directory such as:

```text
data/
  chaosNLI_mnli_m.jsonl
  chaosNLI_snli.jsonl
  chaosNLI_alphanli.jsonl
```

The code intentionally uses command-line paths; no machine-specific paths are
hard-coded.

## Quick Check

Evaluate the tiny synthetic example:

```bash
python scripts/evaluate_trace.py \
  --input examples/tiny_trace.jsonl \
  --aggregate
```

## Reproduction Workflow

### 1. Generate CoT traces

```bash
python scripts/generate_cot.py \
  --dataset snli \
  --input data/chaosNLI_snli.jsonl \
  --output outputs/snli/CoT_gpt.jsonl \
  --model openai/gpt-oss-20b \
  --cache-dir .cache/huggingface
```

For local checkpoints, pass the checkpoint directory to `--model`.

### 2. Create cumulative CoT prefixes

```bash
python scripts/make_chunks.py \
  --input outputs/snli/CoT_gpt.jsonl \
  --output outputs/snli/CoT_chunks_gpt.jsonl \
  --num-chunks 10 \
  --method token
```

This produces 11 prefixes: 0%, 10%, ..., 100%.

### 3. Run Cross-CoT logit tracing

```bash
python scripts/run_logit_trace.py \
  --input outputs/snli/CoT_chunks_gpt.jsonl \
  --output outputs/snli/r1-qwen_logits_trace_for_CoT_gpt.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --options A B C \
  --template think \
  --cache-dir .cache/huggingface
```

For alphaNLI, use `--options A B`. If a model tokenizer does not map these
letters to single tokens, pass explicit IDs with `--option-token-ids`.

Use `--template gpt-oss` for GPT-OSS style analysis/final channel delimiters and
`--template think` for models that use `<think>...</think>` conventions.

### 4. Evaluate traces

```bash
python scripts/evaluate_trace.py \
  --input outputs/snli/r1-qwen_logits_trace_for_CoT_gpt.jsonl \
  --output outputs/snli/r1-qwen_for_gpt_metrics.csv
```

The default probability conversion is the linear normalization used in the
paper. For a robustness check, use `--probability-mode softmax`.

### 5. Estimate model-vs-CoT contributions

After concatenating metric CSVs with `model` and `cot_source` columns:

```bash
python scripts/anova_contributions.py \
  --input outputs/snli/final_step_metrics.csv \
  --score jsd
```

This fits an additive two-way ANOVA model:

```text
score ~ C(model) + C(cot_source)
```

and reports each factor's sum-of-squares contribution percentage.

## Repository Layout

```text
configs/        Model and dataset metadata templates
docs/           Data provenance and usage notes
examples/       Tiny synthetic examples for smoke checks
scripts/        Command-line entry points
src/cot_hlv/    Reusable Python package
```

## Citation

If you use this code, please cite:

```bibtex
@article{DBLP:journals/corr/abs-2601-03154,
  author       = {Beiduo Chen and Tiancheng Hu and Caiqi Zhang and Robert Litschko and Anna Korhonen and Barbara Plank},
  title        = {Decoupling the Effect of Chain-of-Thought Reasoning: {A} Human Label Variation Perspective},
  journal      = {CoRR},
  volume       = {abs/2601.03154},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2601.03154},
  doi          = {10.48550/ARXIV.2601.03154},
  eprinttype   = {arXiv},
  eprint       = {2601.03154}
}
```

## Notes

- Large-model inference is compute intensive. The paper used multi-GPU inference
  for the largest models.
- Generated CoT traces may contain model-specific template markers. The helper
  functions cover common `<think>` and GPT-OSS channel formats, but you may need
  to adjust `extract_reasoning` for new models.
- This repository omits private paths, local caches, raw data, and intermediate
  outputs by design.
