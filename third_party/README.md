# Third-Party Dependency Root

This directory is the default local root for pinned third-party dependencies
managed by `tools/bootstrap_deps.py`.

Expected local layout:

```text
third_party/
  llvm/<version>/        # pinned LLVM/MLIR prebuilt package
  stablehlo/             # StableHLO git submodule pinned by superproject
  shardy/                # Shardy/SPMD git submodule pinned by superproject
  xla/                   # OpenXLA/XLA git submodule for future GSPMD partitioner work
  pytorch-xla/           # PyTorch/XLA git submodule for torch.export StableHLO path
  torch-mlir/            # torch-mlir git submodule for future Torch importer adapters
  googletest/            # googletest git submodule for unit-test fallback
  python/                # pinned Python tooling venv
  python-importer/       # pinned torch/torchvision/torch_xla importer venv
  downloads/             # resumable downloaded archives
```

Public source dependencies are maintained as git submodules at the first level of
`third_party/`. Large generated or downloaded contents are intentionally ignored
by git. Version pins and CMake discovery live in `cmake/third_party/`.

Dependency classes:

| class | dependency | management |
| --- | --- | --- |
| core compiler | LLVM/MLIR | pinned prebuilt under `third_party/llvm/<version>` or explicit `MLIR_DIR`/`LLVM_DIR` override |
| input dialect / SPMD | StableHLO, Shardy, OpenXLA/XLA | git submodules under `third_party/stablehlo`, `third_party/shardy`, and `third_party/xla` |
| frontend importer | PyTorch/XLA, torch-mlir, torch/torchvision/torch_xla wheels | source submodules plus `requirements-importer.txt` |
| future runtime / driver | HPGR SDK, KMD/UAPI headers, legacy Tsm/VS SDK | not vendored yet; CMake exposes explicit opt-in roots |
| test tooling | lit, FileCheck, GTest | Python venv / LLVM tools / `third_party/googletest` |
