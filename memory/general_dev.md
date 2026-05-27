## Wafer compiler local build harness

- 第三方依赖的固定版本集中在 `cmake/third_party/WaferDependencyVersions.cmake`；不要把 LLVM、StableHLO、
  Shardy、OpenXLA/XLA、PyTorch/XLA、torch-mlir、lit 或 gtest 版本散落到源码里。
- 用 `python3 tools/bootstrap_deps.py --python` 把固定版本 Python 测试工具安装到
  `third_party/python`。
- 用 `python3 tools/bootstrap_deps.py --importer-sources` shallow fetch 固定版本 PyTorch/XLA、StableHLO、
  Shardy 和 OpenXLA/XLA source submodules 到 `third_party/<name>`；PyTorch/XLA 的 `WORKSPACE`
  `xla_hash` 决定 frontend 要匹配的 XLA 版本，顶层 `third_party/xla` 必须与它一致；不要 full clone
  上游历史作为默认 bootstrap。
- `third_party/pytorch-xla` 是 PyTorch/XLA 源码事实源；P2.F1 的可运行 `torch_xla` 必须由该 checkout
  编译/安装得到，并让 importer Python 环境通过 `import torch_xla` 和顶层 `import _XLAC`。
  prebuilt `torch_xla` wheel 不能作为 P2.F1 完成证明。
- `python3 tools/bootstrap_deps.py --importer-python` 只准备 PyTorch/XLA 源码构建需要的 importer
  Python packages；`torch_xla` runtime 必须随后从 `third_party/pytorch-xla` 源码用该 Python 编译/安装。
- `tools/build_pytorch_xla_runtime.py --jobs 8` 是当前 `torch_xla` 源码构建入口；它调用
  `third_party/pytorch-xla`，并用 Bazel override 固定到本仓库 `third_party/xla` /
  `third_party/llvm-project` 和 importer Python 的 `torch` headers/libs。这个步骤可以生成本地
  editable install，但不能替换成 prebuilt `torch_xla` wheel。
- 用 `python3 tools/bootstrap_deps.py --llvm` 下载固定版本 LLVM/MLIR 预编译包；脚本会检查远端
  Content-Length，并把未完成下载保存在 `.part` 后续续传，避免把半包当成可解包 archive。
- 在固定版本 LLVM/MLIR 预编译包下载完成前，本地 bring-up 可以显式 override：
  `cmake -S . -B build/p0 -GNinja -DMLIR_DIR=<mlir-cmake-dir> -DLLVM_DIR=<llvm-cmake-dir>
  -DPython3_EXECUTABLE=$PWD/third_party/python/bin/python -DWAFER_ALLOW_UNPINNED_LLVM=ON`。
- 当前统一依赖 smoke 验证使用固定版本 LLVM/MLIR install 配置：
  `cmake -S . -B build/r0-deps-pytorch-xla -GNinja -DMLIR_DIR=$PWD/build/third_party/llvm-install/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/lib/cmake/mlir -DLLVM_DIR=$PWD/build/third_party/llvm-install/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/lib/cmake/llvm -DWAFER_ENABLE_IMPORTER_DEPS=ON -DWAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS=ON -DWAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON -DWAFER_IMPORTER_PYTHON_EXECUTABLE=$PWD/third_party/python-importer-py311/bin/python`，
  然后跑 `cmake --build build/r0-deps-pytorch-xla --target check-wafer -- -j128` 和
  `ctest --test-dir build/r0-deps-pytorch-xla --output-on-failure`。
- Shardy 不用 standalone Bazel workspace 作为 Wafer dependency 编译验证；`WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON`
  会通过 `cmake/third_party/WaferShardyCMake.cmake` 编译 `wafer-shardy-cmake-gate` / `shardy-sdy-opt`，
  复用同一套固定版本 LLVM/MLIR 和 embedded StableHLO。
- `WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时，`wafer-opt` / `wafer-import-model` 会注册 SDY dialect；
  新增 SDY artifact gate 要用 `REQUIRES: shardy`，避免关闭 Shardy 时让后端 textual tests 硬依赖
  `sdy`。
- frontend artifact verifier 入口是
  `wafer-import-model --verify-import-result <mlir>`；PyTorch/XLA capture 主链路用
  `wafer-import-model --verify-stablehlo-bundle <bundle-dir>` 校验 `functions/forward.mlir`、
  `functions/forward.meta` 和 `data/<parameter>`，不要再为同一关系生成 Wafer 私有伴随 JSON。
- P2.S1 sharded 4096 matmul artifact 入口是
  `third_party/python-importer-py311/bin/python tools/wafer_pytorch_xla_capture.py --emit-p2s1-sharded-matmul --output-root <dir> --global-rank <rank>`。
  该入口在 frontend 模型中用 `torch.ops.xla.dynamo_mark_sharding` 覆盖 data/batch、column
  parallel、row/contracting、2D output、2D contracting+output 和 partial replication；输出仍是
  PyTorch/XLA bundle，不生成 `wafer_sharding.json`。每个 strategy 用
  `wafer-import-model --verify-spmd-bundle <dir>/<strategy>` 校验，再用
  `build/r0-deps-pytorch-xla/shardy-sdy-opt <dir>/<strategy>/functions/forward.mlir --sdy-propagation-pipeline`
  检查 SDY propagation 可消费这些 sharding facts。
- 依赖一致性检查入口是 `tools/check_deps.py`；默认检查固定版本、importer registration hook、
  public source submodule checkout HEAD、importer Python package pin 和 core/frontend/runtime/test
  tool dependency layering。
- Wafer IR 文件组织检查入口是 `tools/check_ir_organization.py --root .`；它检查 `WaferOps.td` 只作为
  TableGen 聚合入口、op family ODS/verifier 文件存在，以及 `test/Dialect/Wafer` 按 family 分目录。
- Stage-connection 测试放在 `test/StageConnections`；入口是
  `tools/check_stage_connection_tests.py --root .`，用于防止这类 gate 退回到
  `unrealized_conversion_cast` cast-only 用例。
- 任务支持范围按硬件能力、runtime/ABI 证据和当前 IR contract 判断，不能按“当前下游 pass 尚未
  实现”反向裁剪上游语义。若 frontend/SPMD/planner 产出合法且硬件可表达的事实，而 IR/lowering
  还没覆盖，应补 IR contract、verifier 或下游恢复任务；不能把实现缺口写成上游不支持。
- ODS op 如果引入 `RecursiveMemoryEffects`、`ReturnLike` 等 interface trait，公开 dialect 头要
  include 对应 C++ interface header，`WaferIR` 也要显式 link 对应 MLIR interface target。
- Dialect 增加 TypeDef 后，base dialect td 需要启用 `useDefaultTypePrinterParser = 1`，否则即使
  `addTypes` 已注册，文本 IR 仍会报 “provides no type parsing hook”。
- `add_mlir_library` 会生成静态库 target 和 `obj.<target>` object target；源文件需要的 compile
  definition 要加到 `obj.<target>`，只加到静态库 target 不会影响实际编译命令。
- MLIR pass 如果会创建其它 dialect 的 op，必须在 `getDependentDialects` 中显式声明对应 dialect；
  只在 driver registry 里注册还不保证 pass 运行时 context 已加载该 dialect。
