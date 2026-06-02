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
- 当前统一依赖验证使用固定版本 LLVM/MLIR install 配置：
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
- P2.S1 负责所有 sharding 相关策略。graph 中存在任意用户 sharding seed 时（函数边界或中间
  `sdy.sharding` / `sdy.sharding_constraint` / `sdy.reshard` / manual sharding），默认 policy
  必须跳过，让 Shardy propagation 推完整图。完全没有用户 seed 时，P2.S1 在 SPMD 层补默认
  single-card function-input sharding seed：默认 `tile-count=16`，调试 `tile-count=1`；找不到合适
  输入切分维度时生成 16-tile replicated seed。不要把这个默认策略放到 placement/group 后段实现。
- P2.S1 不能用手写 `sdy.sharding`、`wafer.spmd.*` attr、私有 JSON 或名字约定冒充 partitioned
  artifact。正确主链是：frontend Python 只通过 `torch_xla.distributed.spmd.mark_sharding`
  标记 4096 matmul 图并导出带 `mhlo.sharding` 的 PyTorch/XLA StableHLO bundle；随后由
  `wafer-import-model --propagate-stablehlo-sharding` / `wafer-propagate-stablehlo-sharding`
  接管 default input seed + Shardy propagation；再由后续 Wafer-owned SPMD partition artifact
  stage 调 XLA SPMD partitioner 并导出 partitioned StableHLO bundle。旧的私有 sharding attr
  emitter、sidecar JSON、单独 `wafer-import-model --verify-spmd-bundle` 和 Python post-SPMD
  路线已移除；不要恢复只生成私有 attrs/sidecar、只跑 SDY propagation 冒充完成，或把 Python
  test helper 写成 SPMD / 用户编译入口。
- P2.S1 当前测试 artifact 入口：
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-reference-bundle` 默认生成 4096 reference
  bundle；需要把真实 PyTorch/XLA export artifact 接到本地 compile gate 时，可以用 `--size <n>` 生成
  小尺寸同构图，避免让 single-tile bring-up 被 4096 工作集容量卡住。
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-sharded-bundle --sharding-strategy=<name>` 生成
  pre-partition mark artifact。post-SPMD partitioned artifact 只能由 P2.S2 的 Wafer-owned SPMD
  partition stage 产生。
- P2.S2 pinned-XLA helper 构建入口是 `tools/build_xla_spmd_partitioner_helper.py`；它在
  `build/xla-spmd-helper/workspace` 生成围绕 `third_party/xla` 的 Bazel overlay，默认用 clang 构建
  `//xla/wafer_tools:wafer_xla_spmd_partitioner`，产物复制到
  `build/xla-spmd-helper/wafer_xla_spmd_partitioner`。本地把 helper 接进 lit：
  `cmake -S . -B build/r0-deps-pytorch-xla -DWAFER_XLA_SPMD_PARTITIONER_HELPER=$PWD/build/xla-spmd-helper/wafer_xla_spmd_partitioner`。
  之后 `cmake --build build/r0-deps-pytorch-xla --target check-wafer-lit` 会运行真实 P2.S2 partition
  artifact gate；没有配置 helper 时该 gate 通过 `REQUIRES: xla-spmd-helper` 自动 unsupported。
- PyTorch/XLA StableHLO bundle 的 `data/<parameter>` 由 upstream exporter 用 `np.save` 写入，因此
  P2.S2 helper 需要解析 `.npy` header 才能切片输入参数；P2.S2 输出的 rank-local shard payload 沿用
  NPY stream，路径为 `parameter_shards/<parameter>/rank_XXXXX.npy`。形状和 dtype 由 NPY header 与
  `forward.parameter_shards.json` 共同校验；不要把 NumPy 文件格式升级成 Wafer package/runtime ABI。
- Wafer-owned Shardy / SPMD 源码放在 `lib/Wafer/Transforms/SPMD/`。只依赖 MLIR / StableHLO /
  Shardy CMake target 的 pass 编进 `WaferTransforms`；需要直接依赖 XLA HLO service /
  `spmd_partitioner` / generated proto / TSL 的入口也放在同一 Wafer 源码目录，但通过
  `tools/build_xla_spmd_partitioner_helper.py` symlink 到 pinned XLA Bazel overlay 编译。不要把这类
  pipeline stage 源码放进 `tools/` 或 `third_party/xla`。
- `test/Spmd` 目前只覆盖 P2.S1 default input seed 和 SDY/Shardy artifact parse/verify，不覆盖
  XLA SPMD partitioner，也不输出 rank-local StableHLO。`test/Frontend` 覆盖 StableHLO/Linalg local
  compute normalization；softmax、RMSNorm、LayerNorm 输入是 fine-grained StableHLO staged graph
  （reduce、broadcast、elementwise、shape ops），不是 `stablehlo.softmax` / `stablehlo.norm`
  或 Wafer 私有 high-level op。`check-wafer` 的大量 lit case 主要来自 Dialect/Transforms/Frontend/
  Pipelines/Integration/Tools，不代表旧 Python post-SPMD oracle 仍存在。
- post-SPMD collective 先进入 Wafer LinalgExt-style tensor collective handoff，和 `linalg` 一起进入
  group/tiling；`wafer.comm` 只能在 `wafer.tile_region` / SPM tile buffers / placement 明确后
  materialize。StableHLO collective 直降 `wafer.comm` 且靠 `unrealized_conversion_cast` 桥 tensor
  和 tile_buffer 的 pass/test 已移除；不要在 group 输入侧恢复这种入口。
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
- 用户级 compiler target 名称统一为 `wafer`，Wafer IR target attr 的唯一主线 spelling 是
  `#wafer.target<wafer>`。`tx8` / `tx81` 只保留在硬件、依赖逆向和外部历史命名事实里，不能作为
  compiler driver target、pipeline 名称或测试 fixture 的主线命名。
- 非小修主线任务动实现前必须先写清楚 pipeline contract：upstream artifact / IR、current stage
  responsibility、output artifact / IR、downstream consumer、user-level driver / named pipeline、
  explicit non-goals 和 completion gate。只说明某个 pass / tool / test 的局部功能不够；完成证明
  必须重放已完成上游 artifact chain，并证明当前 stage 输出会被下游边界直接消费。
- 主链路 compile gate 用 `WaferPipelines` 中注册的 named pipeline，不在 Integration 里手动拼 pass
  串。当前用户级 compile 入口是 `wafer-import-model --compile-stablehlo-bundle-to-cabi`；
  sharding propagation 阶段检查入口是 `wafer-import-model --propagate-stablehlo-sharding`。
  `wafer-opt` named pipeline 是
  `wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、
  `wafer-lower-linalg-to-cabi`、`wafer-lower-stablehlo-to-cabi` 和
  `wafer-lower-tile-communication-to-cabi`。`wafer-propagate-stablehlo-sharding` 只做 default input
  seed + Shardy propagation，不冒充 XLA SPMD partitioner；当前没有 Python post-SPMD 路线。
  P2.S2 的 Wafer-owned partition artifact stage 由 `wafer-import-model --partition-stablehlo-bundle`
  消费 propagated StableHLO/SDY 并调用 pinned-XLA helper，不能塞进 frontend Python 或
  `wafer-lower-stablehlo-to-linalg`。`wafer-lower-stablehlo-to-cabi` 必须组合
  StableHLO->Linalg 与 Linalg->C ABI body，不能另起一套 parallel lowering。`target=wafer` 会
  materialize/校验 `#wafer.target<wafer>`，非 `wafer` target 必须诊断。compile driver 必须拒绝带
  pre-SPMD sharding seed 但没有 post-SPMD marker 的 bundle。单 pass flags 只用于
  `test/Transforms`、`test/StageConnections` 等 unit/debug 覆盖。
- ODS op 如果引入 `RecursiveMemoryEffects`、`ReturnLike` 等 interface trait，公开 dialect 头要
  include 对应 C++ interface header，`WaferIR` 也要显式 link 对应 MLIR interface target。
- Dialect 增加 TypeDef 后，base dialect td 需要启用 `useDefaultTypePrinterParser = 1`，否则即使
  `addTypes` 已注册，文本 IR 仍会报 “provides no type parsing hook”。
- `add_mlir_library` 会生成静态库 target 和 `obj.<target>` object target；源文件需要的 compile
  definition 要加到 `obj.<target>`，只加到静态库 target 不会影响实际编译命令。
- MLIR pass 如果会创建其它 dialect 的 op，必须在 `getDependentDialects` 中显式声明对应 dialect；
  只在 driver registry 里注册还不保证 pass 运行时 context 已加载该 dialect。
- 不要恢复 `wafer-check-softmax-schedule`、`wafer-check-norm-schedule`、
  `wafer-check-projection-residual-schedule` 或 `wafer-check-mlp-schedule` 这类 case-specific
  transformer acceptance pass。StableHLO->Linalg 只证明 structured tensor lowering；softmax/norm/MLP
  的真实完成证明应来自通用 group formation、tile/materialization、resource verifier 和下游消费。
