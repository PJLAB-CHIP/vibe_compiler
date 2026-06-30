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
  `cmake -S . -B build/wafer-bootstrap -GNinja -DMLIR_DIR=<mlir-cmake-dir> -DLLVM_DIR=<llvm-cmake-dir>
  -DPython3_EXECUTABLE=$PWD/third_party/python/bin/python -DWAFER_ALLOW_UNPINNED_LLVM=ON`。
- 当前统一依赖验证使用固定版本 LLVM/MLIR install 配置，默认 build dir 使用中性的
  `build/wafer-dev`，不要把阶段名、任务号或某个 frontend 依赖名写进长期 build 目录约定：
  `cmake -S . -B build/wafer-dev -GNinja -DMLIR_DIR=$PWD/build/third_party/llvm-install/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/lib/cmake/mlir -DLLVM_DIR=$PWD/build/third_party/llvm-install/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/lib/cmake/llvm -DWAFER_ENABLE_IMPORTER_DEPS=ON -DWAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS=ON -DWAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON -DWAFER_IMPORTER_PYTHON_EXECUTABLE=$PWD/third_party/python-importer-py311/bin/python`，
  然后跑 `cmake --build build/wafer-dev --target check-wafer -- -j128` 和
  `ctest --test-dir build/wafer-dev --output-on-failure`。
- `ctest` 通过不等于关键 program / E2E gate 被执行。涉及 frontend/SPMD/program pipeline 或声称
  主链路跑通时，还要跑
  `/root/miniconda3/bin/lit -sv --show-unsupported build/wafer-dev/test`，确认相关 `REQUIRES` 测试没有
  被 `unsupported` 跳过；必要时查 `build/wafer-dev/CMakeCache.txt` 中对应 feature/helper 是否为空。
  对当前 SPMD program gate，`wafer-opt-spmd-partition.test` 和 `wafer-opt-spmd-to-group.test` 必须在
  配置了 `WAFER_XLA_SPMD_PARTITIONER_HELPER` 后实际执行，不能用 `ctest passed` 代替。
- 不要并发运行两个会写同一个 lit output tree 的验证命令，例如同时跑 `ctest --test-dir
  build/wafer-dev` 和 `/root/miniconda3/bin/lit ... build/wafer-dev/test`。部分 `test/Tools` 用固定
  `%t` output 路径，两个 lit 实例会互相清理目录，导致假失败；需要顺序跑。
- Runtime adapter 测试分层：package metadata/exporter 这类 compiler artifact golden 用 lit；no-card
  adapter contract、`fake-tx` test backend call sequence 和 runtime library discovery diagnostics 用 Python
  unittest / ctest；真实板端 launch/completion/error propagation 必须 gated 到有卡环境，不能塞进默认 lit。
- C++ `wafer-run` no-card gate 不只做 `dlopen` / symbol check：它必须从 package metadata 构造
  RuntimeSession binding/module/launch/completion plan，验证 `binding_order`，并在 symbol gate 前拒绝
  descriptor-only BPM、tx-host 不支持的 completion source 和非 tx-host runtime mode。
- Shardy 不用 standalone Bazel workspace 作为 Wafer dependency 编译验证；`WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON`
  会通过 `cmake/third_party/WaferShardyCMake.cmake` 编译 `wafer-shardy-cmake-gate` / `shardy-sdy-opt`，
  复用同一套固定版本 LLVM/MLIR 和 embedded StableHLO。
- `WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时，`wafer-opt` 和 `wafer-compile-stablehlo`
  会注册 SDY dialect；新增 SDY program gate 要用 `REQUIRES: shardy`，
  避免关闭 Shardy 时让后端 textual tests 硬依赖 `sdy`。
- frontend program verifier 入口是
  `wafer-compile-stablehlo --verify-frontend-program <mlir>`；PyTorch/XLA capture 主链路用
  `wafer-compile-stablehlo --verify-stablehlo-program <program-dir>` 校验 `functions/forward.mlir`、
  `functions/forward.meta` 和 `data/<parameter>`，不要再为同一关系生成 Wafer 私有伴随 JSON。
- P2.S1 负责所有 sharding 相关策略。graph 中存在任意用户 sharding seed 时（函数边界或中间
  `sdy.sharding` / `sdy.sharding_constraint` / `sdy.reshard` / manual sharding），默认 policy
  必须跳过，让 Shardy propagation 推完整图。完全没有用户 seed 时，P2.S1 在 SPMD 层补默认
  function-input sharding seed：rank count / axes 来自 SPMD 前选出的 `wafer.execution.mesh`；单卡默认
  topology 配置是 4x4 / 16 tile。`wafer.execution.mesh` 默认使用 `all_available` policy，用满
  topology 中所有 available endpoints 且不保存 endpoint section；1-rank 或少 tile mesh 只能作为显式
  debug/bring-up/资源隔离 override 进入。找不到合适输入切分维度时生成同一 mesh 上的 replicated
  seed。不要把这个默认策略放到 group 后段实现。
- P2.S1 不能用手写 `sdy.sharding`、`wafer.spmd.*` attr、私有 JSON 或名字约定冒充 partitioned
  program。正确主链是：frontend Python 只通过 `torch_xla.distributed.spmd.mark_sharding`
  标记 4096 matmul 图并导出带 `mhlo.sharding` 的 PyTorch/XLA StableHLO program directory；随后由
  `wafer-opt --program-pipeline=stablehlo-spmd`、
  `wafer-opt --program-pipeline=stablehlo-spmd-to-linalg` 或
  `wafer-opt --program-pipeline=stablehlo-spmd-to-group` 在 Wafer compiler 层接管 default input
  seed、Shardy propagation 和 XLA SPMD partition，并导出 partitioned StableHLO program，或继续
  写回 post-linalg / group Wafer program。旧的私有 sharding attr
  emitter、sidecar JSON、单独旧 SPMD verify flag 和 Python post-SPMD
  路线已移除；不要恢复只生成私有 attrs/sidecar、只跑 SDY propagation 冒充完成，或把 Python
  test helper 写成 SPMD / 用户编译入口。
- P2.S1 当前测试 program 入口：
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-reference-program` 默认生成 4096 reference
  program directory；需要把真实 PyTorch/XLA export program 接到本地 compile gate 时，可以用 `--size <n>` 生成
  小尺寸同构图，避免让 single-tile bring-up 被 4096 工作集容量卡住。
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-sharded-program --sharding-strategy=<name>` 生成
  pre-partition mark program。post-SPMD partitioned program 只能由 P2.S2 的 Wafer-owned SPMD
  partition stage 产生。
- P2.S2 pinned-XLA helper 构建入口是 `tools/build_xla_spmd_partitioner_helper.py`；它在
  `build/xla-spmd-helper/workspace` 生成围绕 `third_party/xla` 的 Bazel overlay，默认用 clang 构建
  `//xla/wafer_tools:wafer_xla_spmd_partitioner`，产物复制到
  `build/xla-spmd-helper/wafer_xla_spmd_partitioner`。本地把 helper 接进 `wafer-opt` build / lit：
  `cmake -S . -B build/wafer-dev -DWAFER_XLA_SPMD_PARTITIONER_HELPER=$PWD/build/xla-spmd-helper/wafer_xla_spmd_partitioner`。
  之后用户级 `wafer-opt --program-pipeline=stablehlo-spmd*` 命令不再传 helper 路径；`wafer-opt`
  从 build-time `WAFER_XLA_SPMD_PARTITIONER_HELPER` 解析 helper。`cmake --build
  build/wafer-dev --target check-wafer-lit` 会运行真实 P2.S2 partition program gate；没有配置
  helper 时该 gate 通过 `REQUIRES: xla-spmd-helper` 自动 unsupported。
- PyTorch/XLA StableHLO program directory 的 `data/<parameter>` 由 upstream exporter 用 `np.save` 写入，因此
  P2.S2 helper 需要解析 `.npy` header 才能切片输入参数；P2.S2 输出的 rank-local shard payload 沿用
  NPY stream，路径为 `parameter_shards/<parameter>/rank_XXXXX.npy`。形状和 dtype 由 NPY header 与
  `forward.parameter_shards.json` 共同校验；不要把 NumPy 文件格式升级成 Wafer package/runtime ABI。
- PyTorch/XLA transformer / RoPE 这类真实图会把 scalar 或 tensor captured constants 放进
  StableHLO function arguments，并在 metadata 中标成 `input_locations` 的 `type_ = "constant"`、
  payload `constants/<position>`。Wafer frontend verifier 要校验这些 NPY payload；pinned-XLA
  helper 输出 post-SPMD program 时也必须复制 constants 目录，否则 `stablehlo-spmd-to-group`
  会在 program directory verifier 阶段被正确拒绝。
- Wafer-owned Shardy / SPMD 源码放在 `lib/Wafer/Transforms/SPMD/`。只依赖 MLIR / StableHLO /
  Shardy CMake target 的 pass 编进 `WaferTransforms`；需要直接依赖 XLA HLO service /
  `spmd_partitioner` / generated proto / TSL 的入口也放在同一 Wafer 源码目录，但通过
  `tools/build_xla_spmd_partitioner_helper.py` symlink 到 pinned XLA Bazel overlay 编译。不要把这类
  pipeline stage 源码放进 `tools/` 或 `third_party/xla`。
- `test/Spmd` 目前只覆盖 P2.S1 default input seed 和 SDY/Shardy program parse/verify，不覆盖
  XLA SPMD partitioner，也不输出 rank-local StableHLO。`test/Frontend` 覆盖 StableHLO/Linalg local
  compute normalization；softmax、RMSNorm、LayerNorm 输入是 fine-grained StableHLO staged graph
  （reduce、broadcast、elementwise、shape ops），不是 `stablehlo.softmax` / `stablehlo.norm`
  或 Wafer 私有 high-level op。`check-wafer` 的大量 lit case 主要来自 Dialect/Transforms/Frontend/
  Pipelines/Integration/Tools，不代表旧 Python post-SPMD helper 仍存在。
- post-SPMD collective 先进入 Wafer LinalgExt-style tensor collective handoff，和 `linalg` 一起进入
  group/tiling；`wafer.tile.*` communication 只能在 `wafer.tile.region` / SPM storage values / endpoint resource facts 明确后
  materialize。StableHLO collective 直降 `wafer.tile.*` communication 且靠 `unrealized_conversion_cast` 桥 tensor
  和 storage 的 pass/test 已移除；不要在 group 输入侧恢复这种入口。
- R2.4 linalg extension collective handoff 的主线验证入口是同一个 Wafer program pipeline：
  `wafer-opt --program-pipeline=stablehlo-spmd-to-linalg ...` 必须从真实 PyTorch/XLA sharded
  program 产出含 `wafer.linalg_ext.collective.*` 的 `functions/forward.mlir`，并保留
  `forward.parameter_shards.json` 与 rank-local NPY payload。局部 `test/Frontend` fixture 可以覆盖
  `all_reduce` / `reduce_scatter` / `all_to_all` / `collective_permute`，但不能替代这个 program
  handoff gate。
- R2.4 local compute 主线不要恢复本地 `wafer-lower-stablehlo-{dot,elementwise,reduce,shape}` 或
  `wafer-normalize-constants` 窄子集；这些旧 pass 入口已经删除。`wafer-lower-stablehlo-to-linalg`
  的主线 body 先运行 Wafer collective handoff，再调用当前 StableHLO pin 的官方
  `stablehlo-legalize-to-linalg`；Wafer collective handoff 不能证明时要 `signalPassFailure`，
  不能静默把 raw StableHLO 留给 R3 group。
- XLA SPMD partitioner 输出的 rank/mask helper 可能以 residual `stablehlo.partition_id` /
  `stablehlo.replica_id`、静态 tensor view 常量链和 all-constant integer `linalg.generic`
  形式出现。R2.4 cleanup 的职责是在 official StableHLO-to-Linalg 前后把这类可静态证明的常量
  折掉，确保 group 输入没有 raw StableHLO residual；不要把这扩成运行时 shape 计算或 Wafer 私有
  compute lowering。
- R2.4 `wafer.linalg_ext.collective.*` 不是只靠 op 名字或 pass switch 的 skeleton；五类 collective
  必须实现 `DestinationStyleOpInterface`、MLIR `TilingInterface`、`WaferTilingInterface` 和
  `WaferLinalgExtCollectiveOpInterface`。slot-crossing 或动态不可证明的 collective-axis tile 应由
  `TilingInterface` 返回 failure，等待 group planner 拆 slot-aligned tile 或 R6 materialization。
- StableHLO `replica_groups` 有多个 row 时不要压成一个 `rank_group`。`wafer.linalg_ext.collective.*`
  现在用互斥的 `rank_group` / `rank_groups` 表达单组或多组 logical ranks；rank-specialized
  tile-region materialization 按当前 logical rank 选择所在 row。这里仍然只保存 logical rank，不保存
  physical endpoint 或 communication algorithm。
- R3.2a/R3.2b 是 analysis-only 阶段：`GroupTilingDemand` 和 `GroupLayoutPlan` 可以用
  `--wafer-dump-group-tiling-demand` / `--wafer-dump-group-layout-plan` dump，但不能把 tile demand、
  layout assignment 或 materialization cut 写成 `wafer.group` attr，也不能在这两步生成
  `wafer.tile.region`。主线 completion gate 要在真实 `stablehlo-spmd-to-group` 输出上重放这些 dump。
- 依赖一致性检查入口是 `tools/check_deps.py`；默认检查固定版本、importer registration hook、
  public source submodule checkout HEAD、importer Python package pin 和 core/frontend/runtime/test
  tool dependency layering。
- Device-code local link gate 默认不再读取外部 machine-local TX8 deps root。TX8 headers、
  libs、sysroot 和 Xuantie `riscv64-unknown-elf-gcc` 来自 repo-vendored
  `third_party/tx8_deps`；Wafer CRT 默认从 `third_party/wafer_crt/lib` 查找 `libvr.a`。当前 vendored
  Xuantie toolchain 的可用 64-bit double-float multilib 是 `rv64imafdc/lp64d`。
  LLVM 21 生成的 RISC-V object 在进入 Xuantie GNU ld 2.35 前需要用 vendored
  `riscv64-unknown-elf-objcopy -R .riscv.attributes` 做 metadata normalization；repo-local
  `libvr.a` 是 debug-stripped archive，避免旧 linker 读取 LLVM RISC-V debug relocations。
  设备链接不再默认编译或链接 capture shim；LLVM object 之外的 target CRT symbol 必须来自
  repo-local TX81/Wafer CRT 或合法 runtime/loader 外部依赖。
- target LLVM lowering 输出给 `mlir-translate --mlir-to-llvmir` 前不能残留任何 Wafer op。target
  topology / execution mesh 在 target LLVM lowering 前是 fact source；lowering 完成后这些 metadata
  应被消费或剥离，并在发现其它 `wafer.*` op 残留时报错。pipeline 测试应把 target/mesh 放进输入，
  防止只检查函数体而漏掉 module-level metadata。
- Wafer IR 文件组织检查入口是 `tools/check_ir_organization.py --root .`；它检查 `WaferOps.td` 只作为
  TableGen 聚合入口、ODS/verifier/test 按 `Tensor`、`Tile`、`Resource`、`Instr`、`Runtime`
  和 `Common` IR 层组织，并检查 `Conversion` 不再被 `WaferTransforms` 直接 owning。
- Wafer transform pass API 的主入口是 `include/Wafer/Transforms/Passes.td` 生成的
  `WaferPasses.h.inc`；新增非可选 pass 应先在 `Passes.td` 声明 argument、summary 和
  dependent dialects，再让实现继承 generated base。只读 dump pass 结束前要
  `markAllAnalysesPreserved()`。
- Region op 的 verifier 要按 MLIR 阶段拆：boundary / operand / result invariant 放普通
  `verify()`，body argument、terminator 和 region body legality 放 `verifyRegions()`。父 region op
  只解释自己 body 的直接 op，不递归解释子 op 内部 region。
- 长期 op/type 协议优先放 ODS type constraints 和 verifier，不靠手写字符串诊断补类型合法性；
  `!wafer.storage`、ranked tensor boundary 和 async token 这类类型要在 ODS 里约束，并在公开
  dialect header / CMake link 中显式包含对应 MLIR type 依赖。
- `wafer-opt` 需要显式注册要暴露的 MLIR pass families；如果测试或用户入口依赖 canonicalizer/CSE
  这类标准 pass，注册 `mlir::registerTransformsPasses()` 并链接 `MLIRTransforms`，不要假设
  `MlirOptMain` 会自动注册。
- 历史 stage-connection 测试和 `tools/check_stage_connection_tests.py` 已删除；后续 group/tile/storage
  连接必须由真实 frontend/SPMD program chain 和 R3/R6/R7 contract 恢复，不能重建手写 fixture 链来冒充主线。
- 任务支持范围按硬件能力、runtime/ABI 证据和当前 IR contract 判断，不能按“当前下游 pass 尚未
  实现”反向裁剪上游语义。若 frontend/SPMD/planner 产出合法且硬件可表达的事实，而 IR/lowering
  还没覆盖，应补 IR contract、verifier 或下游恢复任务；不能把实现缺口写成上游不支持。
- `wafer.target.topology` 的稳定 V0 表示是规则拓扑加例外：`card_grid`、`card_interconnect`、
  `tile_grid` 和 `unavailable_tiles`。不要恢复成展开的 `tile_ids` / `tile_coords` / `links` graph，
  不要把 bad/PG-disabled 分成多套不可用集合，也不要通过 `id_encoding` 把 endpoint 编码规则变成
  IR 合同。单卡 tile 邻接和跨卡 C2C 邻接从规则 grid / mesh-or-torus kind 派生。公开 pass /
  pipeline 选项用 `card-y`、`card-x`、`tile-y`、`tile-x` 表达 grid 规模，不恢复冗余的
  `*-count` spelling。
- `Cx/NCx` layout 规则容易误用，必须按硬件文档的 `get_CxC0` /
  `common_tensor_info_generate_i64` 口径理解：对齐的是 logical last dimension `C`，不是 flatten
  后的任意元素流；INT8/UINT8 full block 是 128，其它 dtype full block 是 64；tail 小于等于半块时
  保留为按 `4/8/16/32/64` 级别对齐的 `C0`，大于半块时 fold 到下一 full block；C alignment 后还要
  计入 256B bank alignment。`Cx` 通常用于 2D，`NCx` 用于 rank > 2，但 `NCx` 的 `N` 只是历史外层
  slice 命名，不等于 semantic batch。full-block 物理顺序是 channel-block major：`Cx` 是
  `[CxBlock][outer][lane]`，`NCx` 是 `[N][CxBlock][HW][lane]`；`aligned_C` 只用于 footprint /
  batch size，不是 logical row stride。不要用 `ceil(C/64)*64`、layout marker 名字或
  `physicalBytes` 单点事实替代完整 physical mapping。
- 判断 `wafer.tile.reshape` 是否需要 instruction movement 时，先保留 StableHLO/tensor reshape
  的 logical 语义：source/result 的 canonical linear element number 对齐，result multi-index
  按新 shape 解释。movement 触发条件是同一 linear element 在 source/result 中的 physical byte
  offset 映射变化，或目标 physical footprint/descriptor 需要 materialized buffer；不是“看见
  reshape”或“看见 cx/ncx”。compact `tensor/ntensor` 可用标准 memref view；`Cx/NCx` reshape
  要先用统一 physical layout calculator 比较 source/result mapping，只有排布变化才发
  `wafer.instr.gather_scatter`。如果统一 helper 还不能表达真实 `C0` tail/fold 和 bank padding，
  先补 helper，不要在 lowering 里临时重写一份局部 layout 解释。
- movement descriptor lowering（`extract_slice/insert_slice/broadcast/transpose`）要从 op
  自身的 logical index relation 出发，枚举静态 iteration domain，再调用同一个
  `computeWaferPhysicalElementByteOffset` 得到 source/dest byte offset 并 coalesce 相邻段；
  compact、`Cx`、`NCx` 都走这条路径。coalesce 后要尽量把规则段打包进 TDMA 三层
  source/dest stride/iteration descriptor，不能退回“每个 coalesced segment 一条 instruction”的长期
  lowering。`insert_slice` 不是只写 slice：它返回 updated dest buffer，所以 lowering 必须先把旧
  dest payload copy 到新 result，再把 source slice overlay 到 result。RDMA/WDMA lowering 要消费
  DDR 侧 `memref.subview` / strided memref layout：整块 compact DDR boundary 生成 contiguous
  descriptor，静态 strided tile view 生成三层 byte stride/iteration descriptor；动态 view、负
  stride、bit-packed element 或超过三层的 descriptor 不能靠名字/shape 猜测，必须 structured
  failure 或等上游补显式 boundary facts。candidate tile-view materialization 已覆盖当前 IR 中 explicit static boundary
  `tensor.extract_slice`、direct output `tensor.insert_slice` storeback，以及 candidate output
  tile offsets/sizes candidate evaluation lowering 的 DDR `memref.subview` producer；instruction lowering
  仍不能根据 whole-boundary shape 自己恢复 subview，closed-loop traversal / tile-shape search 归
  candidate-selection。
- SPM offset assignment 的 planned offset fact 不属于 `#wafer.memory<spm, layout>` 本身。
  planning fact 当前挂在 SPM `memref.alloc` 的 `wafer.spm.offset`
  attr 上，值为 `#wafer.spm_offset<offset>`；arena 作用域是单个 `wafer.tile.region`，不同 tile-region
  可以复用相同 offset。`size`、alignment 和 bank span 必须由
  `computeWaferPhysicalTensorInfo(memrefType)`、target policy 和 offset 重算，所以 `Cx/NCx` padding、
  C0 tail/fold 和 256B bank alignment 都进入 footprint。当前 V0 对 instruction-level IR 建立可重算的
  region-aware lifetime dataflow：base/view-like alias 共享 root ref，`scf.if` 用互斥 path condition
  判断 branch reuse，`scf.for` 对 iter_args/yield/backedge 延伸 loop-carried lifetime，async issue
  的 SPM operand 通过 `!async.token` 延伸到 wait/fence use；本地 compute/movement SPM write 在
  `wafer.instr.local_fence` 前不能被复用，DTE send/recv source/destination 则由 `dte_wait` token
  收口。offset 搜索使用
  pressure-weighted offline packing：physical size 大、conflict pressure 高、lifetime span 长的 demand
  先放置，再在合法 gap 中选最低 offset；搜索 trace、priority weight 和未接受 offset 不写进 IR。
- DDR offset assignment 不是 external DMA validation 的别名。compiler-managed DDR demand 由 DDR
  `memref.alloc` 本身表达；accepted fact 写回同一个 alloc 的 `wafer.ddr.offset =
  #wafer.ddr_offset<offset>`。external function argument 不分配 offset，也不写 access summary attr；
  launch/resource binding requirement 由 ABI/package/runtime 使用点从 committed instruction IR、
  accepted offset facts、topology/execution-mesh contract、program parameter shard metadata 和薄 launch/block binding 重算；runtime allocation/import/query 属于
  runtime adapter。DDR planner
  复用 SPM 同类 structured lifetime dataflow：view-like alias、tile-region
  boundary arg、`scf.if` path condition、`scf.for` iter_args/yield/backedge 和 async token 都从 IR
  结构重算；offset 搜索在 default DDR arena 中做 pressure-weighted first-fit，只有 lifetime 证明不重叠
  时才复用。runtime allocation object、physical address、packet/ABI 字段仍属于 runtime/ABI boundary，
  不能塞进 DDR planning attr。
- group-to-tile-region 的 buffer-level collective materialization 是 rank-specialized lowering：局部
  pass / dump 入口用 `logical-rank` 选项表示当前 logical rank，并只用它在 collective `rank_group`
  内计算 group-local `local_rank`。输出 `wafer.tile.all_gather` / `reduce_scatter` / `all_reduce`
  显式携带 SPM buffer、`rank_group`、`local_rank`、`group_size` 和 byte count；不在这一步选择 p2p
  schedule、endpoint 或 DTE packet。
- group formation 在 `outs` 固定后会吸收 group 内部 static support producers：`arith.constant`、
  `tensor.empty`、static `tensor.extract_slice` / `tensor.insert_slice`、`tensor.expand_shape` 和
  `tensor.collapse_shape`。这用于避免 XLA/HF 产生的 static `insert_slice` collective input 被错误
  作为 group 外部 DDR boundary；不能因此跨 side-effect、memref/runtime 或 raw StableHLO op。
- floating `arith.select` 在 tile-region-to-instr 中不能继续生成 `wafer.instr.elementwise <select>`；
  现在会 lower 成 false-copy `gather_scatter` + `wafer.instr.bit2fp` + `wafer.instr.mask_move`。
  这条序列按 Triton/TX81 的 `bit2fp` / `mask_move` 证据对齐 target wrapper 粒度；integer/select
  泛化仍要等 target wrapper 证据补齐。
- top-level single-result `wafer.linalg_ext.collective.collective_permute` 现在直接 materialize 成
  `wafer.instr.dte_send` / `dte_recv` / `dte_wait`、local copy 或 zero-fill。
- top-level single-result `wafer.linalg_ext.collective.all_to_all` 的 V0 materialization 要求
  `split_count == rank_group.size()`，把每个 split slot 先 extract 成连续 SPM comm buffer，DTE 只收发
  连续 buffer，recv 后再 insert 到 concat result slot；当前没有 ring/blocked schedule selector、
  raw non-unicast DTE 或 cross-card route binding。
- tile-region-to-instr 的 communication schedule selector 是 pass-level rewrite policy，不进入 IR：
  `all-gather-schedule=auto|ring|direct` 默认 `auto=ring`，`all-reduce-schedule=auto|ring|tree`
  默认 `auto=ring`，`reduce-scatter-schedule=auto|direct` 默认 `auto=direct`。展开后只保留
  `wafer.instr.dte_*` / local compute body，不保存 algorithm attr。
- tile-region-to-instr 的 V0 all-gather lowering 从 `wafer.tile.all_gather` 的 compact `tensor/ntensor`
  local/gather SPM buffer shape 推导唯一 gather axis。`ring` 先把 local chunk 写入本 rank slot，
  插入 `wafer.instr.local_fence` 后沿 ring forward slot view；`direct` 每个 phase 发送 local slot
  给 `(rank+d)`，同时接收 `(rank-d)` 的 chunk 到对应 slot。DTE peer 仍是 logical rank，SPM offset、
  physical endpoint、DTE id 和 packet field 留给后续 planning / ABI 边界。
- tile-region-to-instr 的 V0 all-reduce lowering 支持 full-buffer ring reduce 和 binomial tree。
  `ring` 先把 input copy 到 accumulator 和 forward staging buffer，`local_fence` 后每步 DTE send
  forward buffer、recv 到 staging buffer、wait token，再用 `wafer.instr.elementwise` 做 sum/max/min
  accumulation；下一步 forward 的是刚收到的 partial，不是 accumulator。`tree` 先 reduce 到
  group-local root 0，再 reverse broadcast final accumulator；accumulator 被 DTE 读取前需要 local fence。
- tile-region-to-instr 的 V0 reduce-scatter lowering 使用 full input + local slot result 表示：
  group-to-tile-region 不再预切当前 rank slot，`wafer.tile.reduce_scatter` 显式携带 scatter `axis`，
  instruction lowering 从 full input 派生 per-target slot `memref.subview`，按 phase-ordered
  all-to-owner unicast 生成 `wafer.instr.dte_send` / `dte_recv` / `dte_wait`，wait 后用
  `wafer.instr.elementwise` 把 recv contribution 累计到 local accumulator。
- 用户级 compiler target 名称统一为 `wafer`，Wafer IR target attr 的唯一主线 spelling 是
  `#wafer.target<wafer>`。`tx8` / `tx81` 只保留在硬件、依赖逆向和外部历史命名事实里，不能作为
  compiler target、pipeline 名称或测试 fixture 的主线命名。
- 非小修主线任务动实现前必须先写清楚 pipeline contract：upstream program / IR、current stage
  responsibility、output program / IR、downstream consumer、user-level driver / named pipeline、
  explicit non-goals 和 completion gate。只说明某个 pass / tool / test 的局部功能不够；完成证明
  必须重放已完成上游 program chain，并证明当前 stage 输出会被下游边界直接消费。
- 主链路 gate 用 `wafer-opt` program pipeline 重放已完成上游链路，不在 Integration
  里手动拼 pass 串。当前 frontend verifier 入口是
  `wafer-compile-stablehlo --verify-stablehlo-program`；用户级 `wafer-opt` program
  pipeline 入口是 `--program-pipeline=stablehlo-spmd`、
  `--program-pipeline=stablehlo-spmd-to-linalg` 和
  `--program-pipeline=stablehlo-spmd-to-group`；`wafer-compile-stablehlo --propagate-stablehlo-sharding`、
  `wafer-compile-stablehlo --partition-stablehlo-program` 已删除，因为 Shardy/SPMD 不属于 frontend
	  verifier tool；旧 C ABI compile 入口也已删除。当前稳定后端主线由
	  `wafer-opt --program-pipeline=stablehlo-spmd-to-group` 后接
	  `wafer-lower-groups-to-ddr-memory-planned-instr` 负责，target LLVM/package auto-export 是下一 gate。
	  旧显式 target CRT issue-op、ring collective、SPM/DDR debug path 和 single-tile
	  materialization pass 链已删除；不要恢复成用户级 compile flow。当前 HF transformer no-card gate
	  已覆盖真实 frontend/SPMD program 到 memory-planned instruction IR；真实 target LLVM/package、
	  board allocation/launch/completion、数值 correctness 和 HF selected-candidate closed-loop path 仍是后续 gate。
- ODS op 如果引入 `RecursiveMemoryEffects`、`ReturnLike` 等 interface trait，公开 dialect 头要
  include 对应 C++ interface header，`WaferIR` 也要显式 link 对应 MLIR interface target。
- ODS op 如果直接使用 MLIR `TilingInterface` 这类 upstream op interface，避免让 TableGen 在
  Wafer namespace 下生成未限定的 `SmallVector` / `OpBuilder` / `ArrayRef` 方法声明；可用 interface
  trait 加 `extraClassDeclaration` 写全限定 C++ 签名，或确保公开 dialect 头有明确且局部的别名。
- Dialect 增加 TypeDef 后，base dialect td 需要启用 `useDefaultTypePrinterParser = 1`，否则即使
  `addTypes` 已注册，文本 IR 仍会报 “provides no type parsing hook”。
- `add_mlir_library` 会生成静态库 target 和 `obj.<target>` object target；源文件需要的 compile
  definition 要加到 `obj.<target>`，只加到静态库 target 不会影响实际编译命令。
- MLIR pass 如果会创建其它 dialect 的 op，必须在 `getDependentDialects` 中显式声明对应 dialect；
  只在 driver registry 里注册还不保证 pass 运行时 context 已加载该 dialect。
- 不要恢复 `wafer-check-softmax-schedule`、`wafer-check-norm-schedule`、
  `wafer-check-linear-residual-schedule` 或 `wafer-check-mlp-schedule` 这类 case-specific
  transformer acceptance pass。StableHLO->Linalg 只证明 structured tensor lowering；softmax/norm/MLP
  的真实完成证明应来自通用 group formation、tile/materialization、resource verifier 和下游消费。
- 不要恢复 `tools/wafer_package_metadata.py --emit-*` 这类 fixed package emitter，也不要把
  `wafer-compile-stablehlo --emit-static-reference-program` 这类 synthetic program emitter 作为 importer
  或 package 主线。Package metadata validator / C stub generator 只能消费显式 package metadata
  测试输入做 tool-unit 覆盖；主线 package metadata 必须由当前 IR / named pipeline 自动导出。
