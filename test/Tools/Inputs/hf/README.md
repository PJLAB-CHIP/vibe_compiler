# HuggingFace test input snapshots

`tiny-random-llama-config.json` is a stable test snapshot of the HuggingFace
config for `peft-internal-testing/tiny-random-LlamaForCausalLM`. The lit tests do
not fetch from HuggingFace at runtime; the PyTorch/XLA capture helper uses this
config to build a deterministic Llama decoder block for Wafer frontend/SPMD
compiler gates.
