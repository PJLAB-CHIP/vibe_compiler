# Third-Party Dependency Root

This directory is the default local root for pinned third-party dependencies
managed by `tools/bootstrap_deps.py`.

Expected local layout:

```text
third_party/
  llvm-project/          # LLVM/MLIR git submodule pinned to the OpenXLA/XLA workspace stack
  stablehlo/             # StableHLO git submodule pinned to the OpenXLA/XLA workspace stack
  shardy/                # Shardy/SPMD git submodule pinned to the OpenXLA/XLA workspace stack
  xla/                   # OpenXLA/XLA git submodule; source of truth for the C++/MLIR dependency stack
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
| core compiler | LLVM/MLIR | git submodule under `third_party/llvm-project`; build/install it and pass `MLIR_DIR`/`LLVM_DIR` or use `WAFER_LLVM_INSTALL_DIR` |
| input dialect / SPMD | StableHLO, Shardy, OpenXLA/XLA | git submodules under `third_party/stablehlo`, `third_party/shardy`, and `third_party/xla`; LLVM/StableHLO pins match XLA, and Shardy must contain XLA's Shardy base pin while using the same lower stack |
| frontend importer | torch/torchvision/torch_xla wheels | Python tooling pinned by `requirements-importer.txt`; PyTorch/XLA and torch-mlir source trees are not Wafer C++ dependencies because they vendor separate XLA/LLVM stacks |
| future runtime / driver | HPGR SDK, KMD/UAPI headers, legacy Tsm/VS SDK | not vendored yet; CMake exposes explicit opt-in roots |
| test tooling | lit, FileCheck, GTest | Python venv / LLVM tools / `third_party/googletest` |

OpenXLA/XLA and Shardy may apply their own `third_party/stablehlo/temporary.patch`
inside their Bazel workspaces. Those patch files are treated as upstream
workspace inputs, not as additional Wafer StableHLO source trees. Wafer keeps one
top-level StableHLO checkout and checks that the XLA/Shardy StableHLO patch
inputs remain identical.
