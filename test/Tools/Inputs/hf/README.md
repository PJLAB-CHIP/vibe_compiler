# HuggingFace test input snapshots

`tiny-random-llama-config.json` is a stable test snapshot of the HuggingFace
config for `peft-internal-testing/tiny-random-LlamaForCausalLM`. The lit tests do
not fetch from HuggingFace at runtime; the PyTorch/XLA capture helper uses this
config plus a fixed upper-triangular causal attention mask to build a
deterministic Llama decoder block for Wafer frontend/SPMD compiler gates. The
vertical corpus records the effective config, model semantics and their SHA-256
source revision; this file does not claim an unrecorded upstream HuggingFace Git
revision.
