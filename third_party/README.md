# Third-Party Dependency Root

This directory is the default local root for third-party dependencies whose
versions are fixed by `utils/deps/bootstrap_deps.py` and
`cmake/third_party/WaferDependencyVersions.cmake`.

Expected local layout:

```text
third_party/
  llvm-project/          # LLVM/MLIR git submodule fixed to the OpenXLA/XLA workspace commit
  stablehlo/             # StableHLO git submodule fixed to the OpenXLA/XLA workspace commit
  shardy/                # Shardy/SPMD git submodule fixed to a commit on the OpenXLA/XLA stack
  xla/                   # OpenXLA/XLA git submodule selected by PyTorch/XLA WORKSPACE xla_hash
  pytorch-xla/           # PyTorch/XLA git submodule; source of truth for the frontend XLA version
  googletest/            # googletest git submodule for unit-test fallback
  minimalloc/            # audited source-derived C++17 MiniMalloc core port
  egg/                   # pinned upstream Rust e-graph implementation
  rust-vendor/           # bootstrap-managed Cargo directory source for locked Rust dependencies
  tx8_deps/              # vendored TX8 headers, static libraries, RTOS evidence, and Xuantie RISC-V GCC toolchain
  numeric-model/         # managed SoftFloat/TestFloat/m4/GMP/MPFR source, build, install, and conformance record
  onednn/                 # managed oneDNN source, build, install, and conformance record
  systemc-model/         # managed Accellera SystemC source, build, install, and conformance record
  python/                # fixed-version Python test-tool venv
  python-importer/       # canonical Python 3.11 importer env with torch and source-built torch_xla
  downloads/             # resumable downloaded archives
```

Public source dependencies are normally maintained as git submodules at the
first level of `third_party/`. The curated MiniMalloc core is the explicit small
source-derived-port exception; its provenance and modifications are recorded beside
the source. Large generated or downloaded contents are intentionally ignored by
git. Exact versions and CMake discovery live in `cmake/third_party/`.

“Fixed version” here means an exact git commit or Python package version. It
does not mean the dependency is already used by Wafer compiler code. In the
current tree, PyTorch/XLA selects the XLA version and is also the source tree
from which a local `torch_xla` Python runtime can be built and installed for
frontend capture. Wafer does not yet call `torch_xla` APIs from core compiler
targets.

Dependency classes:

| class | dependency | management |
| --- | --- | --- |
| core compiler | LLVM/MLIR | git submodule under `third_party/llvm-project`; build/install it and pass `MLIR_DIR`/`LLVM_DIR` or use `WAFER_LLVM_INSTALL_DIR` |
| static memory packing | MiniMalloc | audited Apache-2.0 source-derived C++17 core port under `third_party/minimalloc`; fixed-capacity solver only, built offline as a hidden non-exported target |
| structured graph normalization | egg | exact git submodule under `third_party/egg`; the Wafer-owned adapter uses a committed Cargo lock and bootstrap-managed directory source, and CMake builds it locked/offline as a hidden static library |
| input dialect / SPMD | StableHLO, Shardy, OpenXLA/XLA | git submodules under `third_party/stablehlo`, `third_party/shardy`, and `third_party/xla`; XLA is selected by the PyTorch/XLA `WORKSPACE` `xla_hash`; LLVM/StableHLO versions match XLA, and Shardy must contain XLA's Shardy base commit while using the same lower stack |
| frontend importer | PyTorch/XLA source, torch/torchvision, `torch_xla` runtime | `third_party/pytorch-xla` fixes the framework importer source version and may be built/installed to produce the importable `torch_xla` package plus `_XLAC` extension; the current `_XLAC` contract pins Python 3.11 and `requirements-importer.txt` fixes the matching Python packages; current Wafer core code does not call `torch_xla` |
| device code link | TX8 deps, Xuantie RISC-V GCC, Wafer CRT source | `third_party/tx8_deps` is vendored and is the default root for TX8 headers, `libcommon_util.a`, `libinstr_tx81.a`, `liblibc_stub.a`, sysroot, `riscv64-unknown-elf-gcc`, `riscv64-unknown-elf-objcopy`, and `riscv64-unknown-elf-nm`; Wafer-owned CRT symbols are built from `runtime/crt` during device linking |
| functional numeric model | SoftFloat/TestFloat, GNU m4, GMP, MPFR | `utils/deps/bootstrap_deps.py --numeric-model-deps` owns `third_party/numeric-model`; CMake consumes only its validated canonical record and imported targets |
| target numeric backend | oneDNN | `utils/deps/bootstrap_deps.py --onednn-deps` owns `third_party/onednn`; `WaferOneDNNBackend` consumes only its validated canonical record and `WaferOneDNN::oneDNN` |
| functional event model | Accellera SystemC | `utils/deps/bootstrap_deps.py --systemc-model-deps` owns `third_party/systemc-model`; CMake consumes only its validated canonical record and official `SystemC::systemc` |
| future runtime / driver | HPGR SDK, KMD/UAPI headers, legacy Tsm/VS SDK | not vendored yet; CMake exposes explicit opt-in roots |
| test tools | lit, FileCheck, GTest | Python venv / LLVM tools / `third_party/googletest` |

OpenXLA/XLA and Shardy may apply their own `third_party/stablehlo/temporary.patch`
inside their Bazel workspaces. Those patch files are treated as upstream
workspace inputs, not as additional Wafer StableHLO source trees. Wafer keeps one
top-level StableHLO checkout and checks that the XLA/Shardy StableHLO patch
inputs remain identical.

Wafer's dependency compile check does not run Shardy's standalone Bazel
workspace. Shardy is compiled through
`cmake/third_party/WaferShardyCMake.cmake`, which reuses the top-level fixed
LLVM/MLIR and embedded StableHLO targets and builds
`wafer-shardy-cmake-gate` / `shardy-sdy-opt` from the single source stack.
