"""Shared constants for the CoT-HLV experiments."""

MODEL_ALIASES = {
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "qwen",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "r1-llama",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "r1-qwen",
    "allenai/Olmo-3-32B-Think": "olmo",
    "zai-org/GLM-Z1-32B-0414": "glm",
    "ByteDance-Seed/Seed-OSS-36B-Instruct": "seed",
    "openai/gpt-oss-20b": "gpt",
}

DATASET_FILES = {
    "mnli": "chaosNLI_mnli_m.jsonl",
    "snli": "chaosNLI_snli.jsonl",
    "anli": "chaosNLI_alphanli.jsonl",
}

NLI_LABELS = ["entailment", "neutral", "contradiction"]
ANLI_LABELS = ["hypothesis_a", "hypothesis_b"]
