# Data

This repository does not redistribute experimental data or model outputs.

The experiments in the paper use ChaosNLI, a benchmark with dense human label
distributions for examples from SNLI, MNLI, and alphaNLI. Please download the
dataset from the official ChaosNLI release and follow its license and citation
requirements:

- Paper: Nie et al., 2020, "Adversarial NLI: A New Benchmark for Natural
  Language Understanding"
- Dataset page: https://github.com/easonnie/ChaosNLI

Expected local layout:

```text
data/
  chaosNLI_mnli_m.jsonl
  chaosNLI_snli.jsonl
  chaosNLI_alphanli.jsonl
```

Each JSONL record is expected to include a human label distribution under
`label_dist`. The original ChaosNLI examples store task-specific fields under
`example`; the code also accepts flattened records with equivalent fields.

Generated artifacts, such as CoT traces, chunked traces, and logit traces, should
be written under `outputs/` or another local directory outside version control.
