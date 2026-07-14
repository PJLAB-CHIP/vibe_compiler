## Accepted executable的reference测试

- reference数值测试应从group/tensor IR调用Q16 `buildExecutableBundle`，让candidate selection、bufferization、
  instruction lowering和memory planning真实执行；手写selected instruction IR只适合负例补充。
- executor可以在单次invocation内把accepted IR构造成immutable execution projection，但必须all-and-only逐op/
  control-edge投影、执行前完成capability preflight、不序列化、不进入bundle/package，也不复制candidate、memory或
  collective schedule；否则会退化成长期shadow plan。
- prepared reference program用独立value id和复制后的typed command field执行；只保留shared MLIRContext为不可变
  `MemRefType`/layout保活，不保留`Operation`或`Value`。测试应在prepare后修改source op并确认prepared结果不变，
  同时确认重新prepare能看到mutation；unsupported capability必须先于input import和arena allocation失败。
- accepted executable不能用“module里只有一个func.func”代替entry合同。共享closure检查应从结构选择唯一public entry
  （仅sole private function作兼容），要求其余helper已定义、private、从entry可达、direct且non-recursive；Q17的program
  output/workspace ABI只改entry。当前target arena base不跨helper隐式传播，所以private helper不能拥有compiler-managed
  DDR root，只能通过DDR memref参数消费entry-owned storage。
- exporter source里的private helper可能被pinned XLA/SPMD helper内联，因此“source里写了`func.call`且完整driver成功”
  不能证明Q16/Q17接收了call closure。direct-call artifact gate必须检查post-helper/accepted rank仍含call，或像reference
  integration test一样从真实group lowering构造保留helper的`ExecutableBundle`；target conversion另用callee-only instruction
  和alias forwarding fixture证明lowering关系。
- 数值differential的非零payload本身不足以证明覆盖。MLP这类组合case应在独立CPU loop oracle之外增加敏感性检查：逐个
  屏蔽hidden channel、逐层清零bias都必须改变完整expected output；否则零bias、零权重或只连接部分channel会让executor
  漏算仍然通过。固定payload使用测试侧明确的整数PRNG和可精确表示的缩放，避免host distribution差异。
- `ReferenceTensor`是compact row-major program-boundary payload，executor内部用shared DDR/SPM arena和accepted offset
  表达physical storage，再通过`computeWaferPhysicalElementByteOffset`访问logical element。descriptor movement必须先
  完整读取payload再写回，才能在source/dest alias时保持确定语义。
- production physical-layout helper保持唯一实现事实源；独立slow coordinate mapper只放测试中作property/differential
  oracle，不得复用production `WaferPhysicalTensorInfo`或offset helper，也不被production代码或artifact消费。矩阵应逐
  logical坐标比较footprint/offset/唯一性/in-range，并覆盖compact stride、Cx/NCx、dtype/rank、tail对齐台阶、channel
  block边界和每一维negative/one-past；这样既能发现同源bug，也不形成第二协议。
- instruction kind决定的type pair和parameter policy属于IR语义，应由dialect typed helper唯一拥有，verifier、projector
  和其它需要该关系的consumer共同消费；executor只从type派生自己的numeric representation。完整enum capability gate
  可遍历TableGen生成的`symbolize*` domain并实际prepare/execute，不能再抄一份kind字符串或case表作为expected能力。
- 硬件wrapper的参数名、函数签名和packet字段写入只能证明编码路径，不能自动证明数值公式。reference语义审计要继续检查
  direct source、object disassembly、public header/register和state入口。zero-point只有`src1`而没有公式时，对全部typed kind
  执行前fail closed。stochastic只有mode枚举而没有hardware seed/state合同时，若产品明确需要common reference policy，必须
  把它和硬件等价性分开：显式execution seed、固定PRNG算法、明确逐dynamic element推进规则和相邻值距离概率，并穷举所有
  rounding kind；未来硬件证据不能静默改写该可重放policy。
- reference executor增长后按`immutable program definition <- projection`和`numeric/storage <- interpreter`拆内部文件，
  public translation unit只保留rank选择、owner lifetime和prepare/execute orchestration。projection是唯一允许读取accepted
  MLIR的模块，interpreter只能消费immutable graph；这种源码拆分不能新增serializable plan、第二份schedule或公共artifact。
- 若reference multi-rank需要的DTE被当前ExecutableBundle fail closed，先补memory-planning后的physical transport
  acceptance和target consumer，再解锁multi-rank executor；不能用手写DTE module绕过bundle gate。
- internal bundle builder把shared MLIR context转移给返回bundle；测试要在bundle存活期间销毁source module。
- source-backed纵向reference应作为同一production driver发布package后的下游gate：重新读取package中的grouped
  artifact构造accepted bundle，用frontend唯一NPY parser加载typed global input和package-relative parameter/constant，
  再按`RankProgramBinding`的global/local shape与slice构造rank invocation。reference mismatch返回非零但不删除已经
  验证的package；CLI的index/tolerance语法错误则必须在编译和发布前拒绝。
- source-backed implementation-readiness census必须先新鲜执行formal `wafer-compile` gate，再从该次正式producer的
  grouped/accepted artifact派生all-rank IR-local replay；手写fixture只能补负例，debug dump也不能冒充package成员。
  规模估算要逐rank记录真实op kind/init/dim/shape/layout/extent，并把compile expansion budget和SPM high-water分别用
  checked arithmetic核算，避免用一个代表rank或理论shape替代实际corpus。
- multi-rank是执行域，不等于transport种类。bundle-level executor先证明all-rank transport合同同质；`None`在每rank
  独立arena执行并用typed output slice重组，`DirectDTE`才使用deterministic event scheduler。replicated output还必须
  跨rank byte-identical，不能为了复用DTE路径而伪造通信。

## Wafer compiler local build harness

- `tasks/progress.md` 是任务队列，不是设计合同。确定下一步时先定位队列项，再读该项指向的编号
  设计文档；不要从旧 progress 叙事、单个工具现状或历史 memory 反推出当前架构边界。若
  `memory/` 与编号设计文档或任务队列冲突，同步修 memory。
- 当实施路线在一个umbrella任务中穿插shared foundation、owner bootstrap、正式producer和下游consumer时，
  每个可独立调度边界都要有queue row和显式`blocked by`；不能把整个umbrella标成单一`next`，再让执行者
  从长计划正文猜状态迁移。需要下游直接消费才能证明完成的producer，拆成implementation row、consumer row和
  integrated completion row，避免隐式循环或提前`done`。
- `docs/`中的hardware/register/reverse-engineering资料只拥有source-backed evidence。production IR、ABI、transport、
  package/runtime policy只由对应编号设计文档拥有；编号合同更新后要搜索supporting docs中的“负责”“主目标”
  “后续自定义ABI”“当前已覆盖/仍待”等规范性或动态措辞，防止形成第二事实源。
- `tasks/plans/`中只有被`tasks/progress.md`当前实施计划索引引用的计划是执行入口；已完成且被新路线
  替代的计划移入`tasks/archive/`并在文件顶部标成historical/superseded，保留审计过程但不得据此恢复
  旧owner或旁路协议。实施计划不能依赖某个agent skill或工具目录才能解释和执行。
- 第三方依赖的固定版本集中在 `cmake/third_party/WaferDependencyVersions.cmake`；不要把 LLVM、StableHLO、
  Shardy、OpenXLA/XLA、PyTorch/XLA、torch-mlir、lit 或 gtest 版本散落到源码里。
- 新numeric/SystemC依赖进入production前分层取证：先枚举checkout/system package，再检查binary architecture和link
  closure，最后只在transaction-local目录用官方candidate源码做configure/build/run probe。每项至少记录exact version/
  commit或digest、license入口、导出的唯一CMake target、thread/TLS状态和最小运行结果；probe通过只形成candidate，不能
  替代统一版本文件、受管bootstrap、上游self-test和项目gate。configure始终显式指定source/build目录和工作目录，避免
  把上游临时文件写入repo root。
- 用 `python3 tools/bootstrap_deps.py --python` 把固定版本 Python 测试工具安装到
  `third_party/python`。
- 用 `python3 tools/bootstrap_deps.py --importer-sources` shallow fetch 固定版本 PyTorch/XLA、StableHLO、
  Shardy 和 OpenXLA/XLA source submodules 到 `third_party/<name>`；PyTorch/XLA 的 `WORKSPACE`
  `xla_hash` 决定 frontend 要匹配的 XLA 版本，顶层 `third_party/xla` 必须与它一致；不要 full clone
  上游历史作为默认 bootstrap。
- `third_party/pytorch-xla` 是PyTorch/XLA源码事实源；framework importer使用的可运行`torch_xla`必须由该checkout
  编译/安装得到，并让 importer Python 环境通过 `import torch_xla` 和顶层 `import _XLAC`。
  prebuilt `torch_xla` wheel不能作为source-build完成证明。
- `python3 tools/bootstrap_deps.py --importer-python` 只准备 PyTorch/XLA 源码构建需要的 importer
  Python packages；`torch_xla` runtime 必须随后从 `third_party/pytorch-xla` 源码用该 Python 编译/安装。
- `tools/build_pytorch_xla_runtime.py --python third_party/python-importer/bin/python --jobs 8`是当前
  `torch_xla`源码构建入口。`bootstrap_deps.py --importer-python`创建`third_party/python-importer`，而build
  script仍有`python-importer-py311`历史默认值，所以在Q13.W收口前必须显式传`--python`。该入口调用
  `third_party/pytorch-xla`，并用 Bazel override 固定到本仓库 `third_party/xla` /
  `third_party/llvm-project` 和 importer Python 的 `torch` headers/libs。这个步骤可以生成本地
  editable install，但不能替换成 prebuilt `torch_xla` wheel。
- 当前依赖栈没有pinned LLVM/MLIR prebuilt URL。用`python3 tools/bootstrap_deps.py --llvm-source`
  同步`WaferDependencyVersions.cmake`固定的llvm-project commit，再独立build/install；当前不要使用
  `--llvm`，也不能把它写成可用的prebuilt bootstrap入口。
- 本地bring-up只有在明确接受非固定工具链风险时才可显式override：
  `cmake -S . -B build/wafer-bootstrap -GNinja -DMLIR_DIR=<mlir-cmake-dir> -DLLVM_DIR=<llvm-cmake-dir>
  -DPython3_EXECUTABLE=$PWD/third_party/python/bin/python -DWAFER_ALLOW_UNPINNED_LLVM=ON`。
- 当前统一依赖验证使用上述source-built pinned LLVM/MLIR install。默认build dir使用中性的
  `build/wafer-dev`，不要把阶段名、任务号或某个frontend依赖名写进长期build目录约定：
  `cmake -S . -B build/wafer-dev -GNinja -DMLIR_DIR=<pinned-llvm-install>/lib/cmake/mlir -DLLVM_DIR=<pinned-llvm-install>/lib/cmake/llvm -DWAFER_ENABLE_IMPORTER_DEPS=ON -DWAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS=ON -DWAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON -DWAFER_IMPORTER_PYTHON_EXECUTABLE=$PWD/third_party/python-importer-py311/bin/python`，
  然后跑 `cmake --build build/wafer-dev --target check-wafer -- -j128` 和
  `ctest --test-dir build/wafer-dev --output-on-failure`。
- `ctest`通过不等于关键program/E2E gate被执行。涉及frontend/SPMD/program pipeline或声称
  主链路跑通时，先跑`cmake --build build/wafer-dev --target check-wafer-lit -- -j128`；需要审计
  unsupported清单时，使用`build/wafer-dev/CMakeCache.txt`中配置的`LLVM_EXTERNAL_LIT`执行
  `-sv --show-unsupported build/wafer-dev/test`，不要硬编码开发机Python路径。
  grouped-program production gate必须执行统一`wafer-compile`到verified grouped program；沿用历史文件名的
  `wafer-opt-spmd-partition.test`和`wafer-opt-spmd-to-group.test`已经改为调用该driver，必须在配置了
  `WAFER_XLA_SPMD_PARTITIONER_HELPER`后实际执行，不能只用`ctest passed`宣称完成。
- 不要并发运行两个会写同一个lit output tree的验证命令，例如同时跑`ctest --test-dir
  build/wafer-dev`和configured lit的`... build/wafer-dev/test`。部分`test/Tools`用固定
  `%t` output 路径，两个 lit 实例会互相清理目录，导致假失败；需要顺序跑。
- Runtime adapter 测试分层：package metadata/exporter 这类 compiler artifact golden 用 lit；no-card
  adapter contract、`fake-tx` test backend call sequence 和 runtime library discovery diagnostics 用 Python
  unittest / ctest；真实板端 launch/completion/error propagation 必须 gated 到有卡环境，不能塞进默认 lit。
- C++ `wafer-run`当前只是独立prototype，不能冒充typed manifest/no-card gate。正式no-card preflight必须从
  verified manifest构造binding/module/launch/completion plan，验证ABI slot与resource的一一对应，并在任何
  allocation/load side effect前拒绝unsupported completion或runtime mode；不能从旧自由字符串或instruction文本恢复。
- Shardy 不用 standalone Bazel workspace 作为 Wafer dependency 编译验证；`WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON`
  会通过 `cmake/third_party/WaferShardyCMake.cmake` 编译 `wafer-shardy-cmake-gate` / `shardy-sdy-opt`，
  复用同一套固定版本 LLVM/MLIR 和 embedded StableHLO。
- `WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时，`wafer-compile`、`wafer-opt` 和
  `wafer-compile-stablehlo` 会注册 SDY dialect；新增 SDY program gate 要用 `REQUIRES: shardy`，
  避免关闭 Shardy 时让后端 textual tests 硬依赖 `sdy`。
- frontend program verifier 入口是
  `wafer-compile-stablehlo --verify-frontend-program <mlir>`；PyTorch/XLA capture 主链路用
  `wafer-compile-stablehlo --verify-stablehlo-program <program-dir>` 校验 `functions/forward.mlir`、
  `functions/forward.meta` 和 `data/<parameter>`，不要再为同一关系生成 Wafer 私有伴随 JSON。
- production SPMD由pinned XLA helper内部完成Shardy/XLA propagation；`wafer-compile`不能先把
  `sdy.constant`、`sdy.reshard`或其它SDY中间op写给只接StableHLO的helper。graph已有用户`mhlo.sharding`
  时由helper消费；无用户seed时当前采用replicated correctness基线。基于execution mesh自动补split seed的
  named pipeline只作IR-local调试，等有完整SDY→StableHLO bridge后才能进入production。单卡execution rank
  没有默认值，用户必须显式选择1或16；target profile同样没有默认值。
- default-sharding/SPMD chain不能用手写`sdy.sharding`、`wafer.spmd.*` attr、私有JSON或名字约定冒充partitioned
  program。正确主链是：frontend Python 只通过 `torch_xla.distributed.spmd.mark_sharding`
  标记 4096 matmul 图并导出带 `mhlo.sharding` 的 PyTorch/XLA StableHLO program directory；随后由
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`
  在 Wafer compiler
  层接管 target/mesh，并由pinned helper内部完成Shardy/XLA SPMD partition，再执行local normalization
  和 logical group formation。当前typed grouped-program driver的输出只到重新读取并验证过的
  grouped program directory；不公开
  program stage selector 或 stop-stage。旧的私有 sharding attr
  emitter、sidecar JSON、单独旧 SPMD verify flag 和 Python post-SPMD
  路线已移除；不要恢复只生成私有 attrs/sidecar、只跑 SDY propagation 冒充完成，或把 Python
  test helper 写成 SPMD / 用户编译入口。
- default-sharding当前测试program入口：
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-reference-program` 默认生成 4096 reference
  program directory；需要把真实 PyTorch/XLA export program 接到本地 compile gate 时，可以用 `--size <n>` 生成
  小尺寸同构图，避免让 single-tile bring-up 被 4096 工作集容量卡住。
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py --emit-sharded-program --sharding-strategy=<name>` 生成
  pre-partition mark program。post-SPMD partitioned program只能由Wafer-owned SPMD
  partition stage 产生。
- pinned-XLA SPMD partition helper构建入口是`tools/build_xla_spmd_partitioner_helper.py`；它在
  `build/xla-spmd-helper/workspace` 生成围绕 `third_party/xla` 的 Bazel overlay，默认用 clang 构建
  `//xla/wafer_tools:wafer_xla_spmd_partitioner`，产物复制到
  `build/xla-spmd-helper/wafer_xla_spmd_partitioner`。本地把 helper 接进 compiler build / lit：
  `cmake -S . -B build/wafer-dev -DWAFER_XLA_SPMD_PARTITIONER_HELPER=$PWD/build/xla-spmd-helper/wafer_xla_spmd_partitioner`。
  production `wafer-compile` 不接收 helper 路径；driver 从 build-time
  `WAFER_XLA_SPMD_PARTITIONER_HELPER` 解析 helper。`wafer-opt` 只处理显式 MLIR 的IR-local debug/test，
  不拥有 program-directory orchestration。`cmake --build
  build/wafer-dev --target check-wafer-lit`会运行真实SPMD-to-group driver gate；没有配置
  helper 时该 gate 通过 `REQUIRES: xla-spmd-helper` 自动 unsupported。只有启用unit tests的build可用显式
  `WAFER_TEST_XLA_SPMD_PARTITIONER_HELPER`做failure-injection override；production binary忽略该环境变量，
  不能把ambient runtime environment变成helper选择协议。
- PyTorch/XLA StableHLO program directory 的 `data/<parameter>` 由 upstream exporter 用 `np.save` 写入，因此
  partition helper需要解析`.npy` header才能切片输入参数；其输出的rank-local shard payload沿用
  NPY stream，路径为 `parameter_shards/<parameter>/rank_XXXXX.npy`。形状和 dtype 由 NPY header 与
  `forward.parameter_shards.json` 共同校验；不要把 NumPy 文件格式升级成 Wafer package/runtime ABI。
- `forward.parameter_shards.json` schema v3 的 parameter binding 必须显式写
  `distribution = replicated | partitioned`。partitioned slices 做无重叠精确覆盖检查；replicated slices
  必须覆盖完整 global tensor、replica-id domain 完整且 NPY byte-identical。当前 helper 遇到 partial
  replication 直接失败；不要用重复 offset 或 `replica_id = 0` 暗示复制关系。
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
- `test/Spmd`目前只覆盖default input seed和SDY/Shardy program parse/verify，不覆盖
  XLA SPMD partitioner，也不输出 rank-local StableHLO。`test/Frontend` 覆盖 StableHLO/Linalg local
  compute normalization；softmax、RMSNorm、LayerNorm 输入是 fine-grained StableHLO staged graph
  （reduce、broadcast、elementwise、shape ops），不是 `stablehlo.softmax` / `stablehlo.norm`
  或 Wafer 私有 high-level op。`check-wafer` 的大量 lit case 主要来自 Dialect/Transforms/Frontend/
  Pipelines/Integration/Tools，不代表旧 Python post-SPMD helper 仍存在。
- post-SPMD collective 先进入 Wafer LinalgExt-style tensor collective handoff，和 `linalg` 一起进入
  group/tiling；`wafer.tile.*` communication 只能在 `wafer.tile.region` / SPM storage values / endpoint resource facts 明确后
  materialize。StableHLO collective 直降 `wafer.tile.*` communication 且靠 `unrealized_conversion_cast` 桥 tensor
  和 storage 的 pass/test 已移除；不要在 group 输入侧恢复这种入口。
- linalg extension collective handoff的主线验证入口是同一个`wafer-compile` driver：它必须从真实
  PyTorch/XLA sharded program 产出含 `wafer.linalg_ext.collective.*` 的 `functions/forward.mlir`，并保留
  `forward.parameter_shards.json` 与 rank-local NPY payload。局部 `test/Frontend` fixture 可以覆盖
  `all_reduce` / `reduce_scatter` / `all_to_all` / `collective_permute`，但不能替代这个 program
  handoff gate。
- local compute normalization主线不要恢复本地`wafer-lower-stablehlo-{dot,elementwise,reduce,shape}`或
  `wafer-normalize-constants` 窄子集；这些旧 pass 入口已经删除。`wafer-lower-stablehlo-to-linalg`
  的主线 body 先运行 Wafer collective handoff，再调用当前 StableHLO pin 的官方
  `stablehlo-legalize-to-linalg`；Wafer collective handoff 不能证明时要 `signalPassFailure`，
  不能静默把raw StableHLO留给logical group formation。
- XLA SPMD partitioner 输出的 rank/mask helper 可能以 residual `stablehlo.partition_id` /
  `stablehlo.replica_id`、静态 tensor view 常量链和 all-constant integer `linalg.generic`
  形式出现。local compute cleanup的职责是在official StableHLO-to-Linalg前后把这类可静态证明的常量
  折掉，确保 group 输入没有 raw StableHLO residual；不要把这扩成运行时 shape 计算或 Wafer 私有
  compute lowering。
- `wafer.linalg_ext.collective.*`不是只靠op名字或pass switch的skeleton；五类collective
  必须实现 `DestinationStyleOpInterface`、MLIR `TilingInterface`、`WaferTilingInterface` 和
  `WaferLinalgExtCollectiveOpInterface`。slot-crossing 或动态不可证明的 collective-axis tile 应由
  `TilingInterface`返回failure，等待group planner拆slot-aligned tile或tile communication materialization。
- StableHLO `replica_groups` 有多个 row 时不要压成一个 `rank_group`。`wafer.linalg_ext.collective.*`
  现在用互斥的 `rank_group` / `rank_groups` 表达单组或多组 logical ranks；rank-specialized
  tile-region materialization 按当前 logical rank 选择所在 row。这里仍然只保存 logical rank，不保存
  physical endpoint 或 communication algorithm。
- group tiling-demand和layout-plan是analysis-only边界：`GroupTilingDemand`和`GroupLayoutPlan`可以用
  `--wafer-dump-group-tiling-demand` / `--wafer-dump-group-layout-plan` dump，但不能把 tile demand、
  layout assignment 或 materialization cut 写成 `wafer.group` attr，也不能在这两步生成
  `wafer.tile.region`。per-rank bundle integrated gate要从typed driver产生并重新验证的grouped program输入
  重放这些analysis。
- 依赖一致性检查入口是 `tools/check_deps.py`；默认检查固定版本、importer registration hook、
  public source submodule checkout HEAD、importer Python package pin 和 core/frontend/runtime/test
  tool dependency layering。
- Device-code local link gate 默认不再读取外部 machine-local TX8 deps root。TX8 headers、
  libs、sysroot 和 Xuantie `riscv64-unknown-elf-gcc` 来自 repo-vendored
  `third_party/tx8_deps`；Wafer-owned `wafer_tx81_*`边界由target LLVM编号设计、repo-local public CRT header
  和source/object共同约束；不能从旧`libvr.a`archive或TX81`__*`
  symbol 反推出 compiler target CRT closure。当前 vendored Xuantie toolchain 的可用 64-bit
  double-float multilib 是 `rv64imafdc/lp64d`。
  compiler-generated RISC-V object 在进入 Xuantie GNU ld 2.35 前需要用 vendored
  `riscv64-unknown-elf-objcopy -R .riscv.attributes` 做 metadata normalization；repo-local
  Wafer CRT source/object 和 target object 一起进入 link gate。设备链接不再默认编译或链接
  capture shim；LLVM object 之外的 target CRT symbol 必须来自 repo-local Wafer CRT source/object
  或明确合法的 runtime/loader 外部依赖。`tools/wafer_device_link.py` 能执行
  `.ll -> .o -> kcore .so` 不等于主线 gate 完成；required-symbol 检查必须拒绝未解释的
  `wafer_tx81_*` undefined symbol。
- `tools/check_target_crt_symbols.py`从target lowering的literal symbol family和`WaferAttrs.td` enum
  spelling推导production surface，再与CRT header/source和编译对象`nm`做exact closure；
  `check_target_crt_conformance.py`从instruction verifier、target address lowering和CRT实现交叉证明关系。
  两者都不能解析`tasks/`或supporting Markdown marker作为expected ABI事实源。
- target LLVM call emission 输出给 `mlir-translate --mlir-to-llvmir` 前不能残留任何 Wafer op。target
  topology / execution mesh 在 target LLVM call emission 前是 fact source；lowering 完成后这些 metadata
  应被消费或剥离，并在发现其它 `wafer.*` op 残留时报错。pipeline 测试应把 target/mesh 放进输入，
  防止只检查函数体而漏掉 module-level metadata。
- Wafer IR 文件组织检查入口是 `tools/check_ir_organization.py --root .`；它检查 `WaferOps.td` 只作为
  TableGen 聚合入口、ODS/verifier/test 按 `Tensor`、`Tile`、`Resource`、`Instr`、`Runtime`
  和 `Common` IR 层组织，并检查 `Conversion` 不再被 `WaferTransforms` 直接 owning。
- Wafer transform pass API 的主入口是 `include/Wafer/Transforms/Passes.td` 生成的
  `WaferPasses.h.inc`；新增非可选 pass 应先在 `Passes.td` 声明 argument、summary 和
  dependent dialects，再让实现继承 generated base。pass即使只间接创建某个dialect的type（例如collective
  materialization创建`async.token`）也必须声明该dialect；组合pipeline中的后置pass可能偶然预加载dialect，因此还要有
  只运行该named pipeline的回归。只读 dump pass 结束前要
  `markAllAnalysesPreserved()`。
- Region op 的 verifier 要按 MLIR 阶段拆：boundary / operand / result invariant 放普通
  `verify()`，body argument、terminator 和 region body legality 放 `verifyRegions()`。父 region op
  只解释自己 body 的直接 op，不递归解释子 op 内部 region。
- 长期 op/type 协议优先放 ODS type constraints 和 verifier，不靠手写字符串诊断补类型合法性；
  `!wafer.storage`、ranked tensor boundary 和 async token 这类类型要在 ODS 里约束，并在公开
  dialect header / CMake link 中显式包含对应 MLIR type 依赖。
- `wafer-opt` 需要显式注册要暴露的 MLIR pass families；如果显式IR debug/test依赖 canonicalizer/CSE
  这类标准 pass，注册 `mlir::registerTransformsPasses()` 并链接 `MLIRTransforms`，不要假设
  `MlirOptMain` 会自动注册。
- 历史 stage-connection 测试和 `tools/check_stage_connection_tests.py` 已删除；后续 group/tile/storage
  连接必须由真实frontend/SPMD program chain和当前group/tile/storage编号合同恢复，不能重建手写fixture链来冒充主线。
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
- SPM/DDR accepted offsets不属于layout本身。`wafer.tile.region`是`IsolatedFromAbove`且verifier禁止SPM
  buffer跨边界，因此当前SPM correctness是逐region从IR重算alias、branch、loop-carried和async completion
  lifetime，并在每条region exit证明pending set为空；whole-variant clone只汇总all-and-only regions并保证
  atomic acceptance。跨region whole-entry allocator是peak/fragmentation优化，不是当前correctness缺口；若未来
  允许SPM跨边界，必须先扩IR/SSA/verifier合同。DDR `wafer.ddr.offset`始终是arena-relative fact，没有typed
  arena base binding时target不得把它当absolute address。physical size、alignment和bank span统一从shared
  geometry helper推导；runtime object、physical address和packet字段不得写回planning IR。
- reduction语义恢复不能只看yielded op class。使用`mlir::matchReduction`或等价结构匹配，证明单一combiner
  的operands精确连接reduced value与accumulator。显式reduction op本身允许implementation-defined binary
  tree，因此floating add不额外要求`fastmath<reassoc>`；但当前kind不能保真的`maxnum/minnum` NaN语义、
  unsigned min/max和integer overflow flags仍必须fail closed。
- whole-op fast path必须证明整个payload可被删除：passthrough/concat/reduction以及named
  fill/matmul/batch_matmul都要检查exact SSA wiring、允许op集合和effect；只匹配yield、shape或op class会
  静默擦除side effect或改写数值语义。Group verifier应递归检查nested body dialect/type。
- destination-style tensor仍遵守functional SSA：fill写fresh result而不覆盖旧init；insert_slice在旧dest仍有
  observable use时构造fresh result并延后boundary store。只有旧dest其余use都被证明是unread DPS-init时才可
  direct tile store。`ins + outs` exact SSA必须唯一；不同SSA的physical no-alias由后续typed driver/ABI闭合。
- variadic custom assembly要覆盖空列表round-trip。`wafer.group`允许零个`ins`，assembly中的operand/type组必须
  optional并有parse→print→reparse gate；不能让printer生成`ins( : )`。
- async completion按path和engine分别建模：local compute/movement的全部SPM read/write只由覆盖同一路径的
  local fence收口，DTE send/recv只由matching token/wait收口；两者不能互相消费。zero-trip loop、分支join和
  loop-carried token没有精确proof时拒绝，region/function terminal boundary不得隐式清空pending状态。
- target undefined-symbol gate使用代码拥有的exact allowlist，并检查全部undefined symbols，而不只检查
  `wafer_*`前缀；prefix/substring命中不能替代精确成员关系。allowlist通过只证明loader ABI surface，不证明
  packet、transport、completion或board正确性。
- all-rank target publication直接消费Q16 `ExecutableBundle`。每rank先把returned compiler-managed DDR root
  重定向到append-only output ABI slot，再从剩余DDR alloc重算workspace high-water/alignment并追加唯一i64 arena
  base argument；target pass只有收到显式argument index才把`wafer.ddr.offset`lower成`base + offset`。lowered entry
  必须是与typed `KernelABISlot[]`一一对应的fixed `void(i64...)`。LLVM IR、object/CRT和`.so`只写Q17 transaction
  staging；all-rank entry/RISC-V64 ELF/symbol/digest readback后才发布`TargetArtifactBundle`，不能用单文件atomic
  replace冒充多rank原子性。ABI slot和workspace的最低DDR alignment来自生成memory plan的同一target policy，
  workspace再与alloc显式alignment取最大值；不得在artifact层另造更小默认值。
- ABI narrowing必须在compiler verifier/target preflight中完成：地址使用uint64，count/stride/iteration/enum/
  mask等普通字段适配uint32，`Data_Shape`维度适配底层uint16；CRT header/source和compiler call保持同一typed
  signature，不用宽形参加wrapper内部cast隐藏截断。
- candidate provenance必须穿过唯一accepted-artifact handoff：selector在transformation-local clone完成完整
  traversal和legality，rejected clone整体丢弃；debug replay消费同一accepted artifact，不重新运行另一套
  direct lowering。target conversion同样在module clone上运行，full success才替换source；多rank staging由
  外层transaction一次发布，单module成功不等于bundle原子性。
- 完整traversal的静态展开必须用checked ceil-div/product并设置显式编译资源预算。当前4096个
  output-tile/reduction-chunk materialization实例上限只防止
  unrolled IR导致编译时间/内存失控，不能写成硬件容量、IR/workload legality或16-tile topology限制；长期应
  用compact loop表示替代静态materialization，而不是把预算扩成架构常量。
- executable/resource handoff必须来自accepted IR和typed C++ bundle，不从raw instruction文本、文件名或参数名
  重建。当前没有executable dialect或独立resource-view协议；resource/entry/completion事实必须从accepted IR
  use-def、type、effect和offset直接校验后进入bundle，不能成为side table或package旁路。
- group-to-tile-region 的buffer-level collective materialization必须从enclosing typed distributed
  instance/candidate entry取得partition和replica coordinates。局部pass选项只可用于明确的replay测试，
  production driver不得使用default rank 0或CLI option承载rank语义。这个边界仍只产生logical buffer
  schedule；每个send/recv的跨rank静态匹配身份必须由protocol phase和logical payload slice显式进入typed IR，
  structured loop/branch中的动态实例再由control-flow instance区分，不能靠op/scheduler顺序或名字恢复。
  endpoint/channel/FSM由post-memory transport acceptance在exact topology/mesh上处理，
  当前不引入pinned/relocatable runtime remapping。
- Direct DTE public helper的`direct_sync_wait`、`direct_fsm_monitor_receive`与`direct_dte_wait_done`都是无timeout参数的
  blocking wait；后者可报告本地DTE错误。compiler/CRT不能伪造device timeout能力：本地status由target ABI写回，
  timeout必须由launch watchdog观察，跨rank peer failure由runtime completion DAG合成。
- selected instruction handoff固定先由selector在tensor函数clone中完成完整traversal、instruction和SPM/DDR
  planning，再复用同一份function-boundary OneShot Bufferization配置消除tensor signature与
  `bufferization.to_memref/to_tensor` wrapper。target named replay直接消费该bufferized accepted artifact；
  不得重新串direct group-to-tile/instr或memory planning。所有C++ builder和group-to-tile public API都显式
  接收logical rank；只有标明debug的named replay可在注册处显式构造rank 0。
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
- 通用 compiler target 名称统一为 `wafer`，Wafer IR target attr 的唯一主线 spelling 是
  `#wafer.target<wafer>`。裸的 `tx8` / `tx81` 不能作为 dialect、pipeline、pass、fixture 或可推断字段的
  主线命名；硬件/依赖逆向事实和tasks/14 closed registry中的opaque canonical profile key例外。例如
  `wafer-tx81-single-card-kernel-v1` 只能整体解析为typed `TargetProfileId`，不得拆字符串恢复target、
  revision、ABI或numeric policy。
- 非小修主线任务动实现前必须先写清楚 pipeline contract：upstream artifact / IR、current stage
  responsibility、output artifact / IR、downstream consumer、user-level driver / named pipeline、
  explicit non-goals 和 completion gate。只说明某个 pass / tool / test 的局部功能不够；完成证明
  必须重放已完成上游 program chain，并证明当前 stage 输出会被下游边界直接消费。
- 主链路gate应由独立`wafer-compile` owner-aware program driver重放已完成上游链路，不在Integration
  里手动拼 pass 串。当前 frontend verifier 入口是
  `wafer-compile-stablehlo --verify-stablehlo-program`；production compile入口统一为
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`。
  typed grouped-program boundary从frontend admission推进到重新读取并验证过的grouped program directory；
  同一production transaction随后把frontend verifier返回的typed boundary/shard facts和grouped module直接交给
  per-rank bundle boundary，不暴露stop-stage。
  `wafer-compile-stablehlo --propagate-stablehlo-sharding`、
  `wafer-compile-stablehlo --partition-stablehlo-program` 已删除，因为 Shardy/SPMD 不属于 frontend
  verifier tool；旧 C ABI compile 入口也已删除。`wafer-opt`和现有named MLIR pipelines只处理显式IR，
  用于IR-local debug/regression，不拥有program-directory I/O，也不构成用户可选stage。当前bundle boundary对
  rank-count 1/16实际创建all-and-only isolated clones，经selector、function bufferization、whole-rank SPM/DDR和
  terminal legality后形成move-only `RankExecutable[]`/context-owning `ExecutableBundle`；rank-15 late failure仍在
  同一transaction内，因此不会先发布grouped checkpoint。无DTE时transport contract为`None`；Q16.T已在完整rank
  domain的post-memory acceptance后形成`DirectDTE`；rank module分拆使sender无法本地重算remote receiver offset，
  因而该cross-rank accepted start必须进入typed binding，不能假设各rank allocation同址。target把async token降成
  CRT返回的opaque i64 event；recv issue先初始化FSM并post ready，send实际attach/send延迟到wait，避免所有rank
  在本地recv ready之前同时阻塞于sender wait。entry status只表达pending/success/local transport error，timeout由
  manifest声明的host watchdog负责，peer failure由runtime合成。
  旧显式 target CRT issue-op、ring collective、SPM/DDR debug path 和 single-tile
  materialization pass 链已删除；不要恢复成用户级 compile flow。当前HF/Llama-style真实program可重放到
  verified logical group staging；HF compute coverage、Direct DTE target/status ABI、runtime、board execution和
  数值correctness仍是后续独立gate。
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
- 不要恢复 `tools/wafer_package_metadata.py --emit-*` 这类fixed package emitter，也不要把
  `wafer-compile-stablehlo --emit-static-reference-program` 这类 synthetic program emitter 作为 importer
  或package主线。主线typed manifest必须只由accepted executable bundle和verified target modules构造，
  不能从raw IR、单个module、printer text或旁路resource view恢复。
- 近期package wire使用唯一C++ typed model的canonical JSON，不预设另一套schema/registry基础设施。Python只可作
  薄CLI或显式legacy converter；唯一semantic verifier在C++，package不复制instruction schedule。
- Kernel ABI近期由typed slot/resource双射和rank/module/entry digest表达；ELF note或跨进程descriptor等真实
  loader/cache consumer出现后再扩展，不能先建设global identity registry。
- 完整compiler driver不能伪装成纯`OpPassManager` named pipeline。`wafer-compile`拥有source、explicit rank
  clones、target staging和atomic publication；named MLIR pipeline只保留IR-local transform。
- package parsing/semantic verification、pure RuntimeSession preflight和provider execution是三层边界。no-card
  preflight不分配、不加载、不发命令；fake/board provider实际调用必须分别记录failure suppression和cleanup。
- 当前typed package入口是`Wafer/Runtime/PackageManifest.h`：compiler只从Q16 `ExecutableBundle`和Q17
  `TargetArtifactBundle`构造manifest，在私有staging内复制payload、核对digest、canonical serialize/parse readback、
  fsync后no-replace发布。`wafer-run --package-dir <root> --entry-id <id> --no-card`是唯一runtime inspection入口；
  Direct DTE package还必须显式声明兼容environment：`--direct-dte-status-abi wafer-direct-dte-status-v1
  --supports-host-watchdog`，缺失时应fail closed；这些选项只形成preflight facts，不代表provider执行。
  Python adapter只启动该二进制。排查package时先跑`PackageManifestTest.*`和`test/Runtime/wafer-run.test`，不要恢复
  已删除的Python schema/exporter或C++ `HostRuntime` acceptance。
- 纵向source corpus不要用`torch.empty()`、framework默认初始化或提交生成物固定输入。当前最小corpus spec在
  `test/Tools/Inputs/workloads/single-card-vertical-v1.json`：整数序列加二进制可精确表示的f32缩放生成
  input/parameter，独立NumPy实现生成完整CPU expected，再与同payload的framework CPU module按tolerance
  交叉检查；真实PyTorch/XLA export后反读parameter NPY逐元素核对，并用`forward.mlir`、canonical meta和
  typed payload构造canonical program digest做重复export证明。CPU-only入口是
  `wafer_pytorch_xla_capture.py --emit-cpu-reference`，真实admission入口是`--emit-workload-corpus
  --verify-corpus-reproducibility`；两者都只证明corpus/frontend admission，不证明compiler、runtime或board。
- reference executor的dtype convert不能依赖C++ cast或host rounding environment。projection从typed
  `InstrConvertKind`复制source/destination format和verified parameter policy，把RND_MODE 0..3显式映射到APFloat
  rounding；执行先为全部logical element生成APInt bits，全部成功后再写destination。浮点到整数的NaN/Inf/越界是
  hard failure。mode4 common reference policy要求显式seed，以固定SplitMix64逐dynamic convert element推进并按上下相邻
  可表示值距离概率选择；它不代表hardware RNG。zero-point数学公式缺少证据时仍必须在input import/storage allocation前
  拒绝。
- reference control-flow projection不能依赖MLIR block存储顺序或隐式fallthrough。先为entry CFG全部block argument和
  顶层SSA result分配value-id，再逐block复制terminator successor/operand；执行branch时先snapshot incoming runtime
  values，再绑定successor arguments。结构化loop保留lb/ub/step、IV和iter_args/yield backedge；当前无环CFG在preflight
  做cycle check，循环继续由`scf.for`表达，避免执行后才发现无法证明终止的CFG cycle。
- reference multi-rank Direct DTE不需要host thread或wall-clock timeout。先投影all-rank immutable programs，再按
  logical rank顺序从inputs重放到未完成wait；send snapshot和matched recv payload由typed message + structured
  control instance索引，下轮重放时注入新建rank-local arena。这保留现有recursive SCF/CFG/call interpreter，
  同时让每轮no-progress直接变成可重放的deadlock诊断；禁止用op访问次序充当dynamic message identity。
