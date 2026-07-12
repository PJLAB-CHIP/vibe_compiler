# Wafer Compiler 架构事实重基线审计

日期：2026-07-12

性质：本文件记录一次基于当前代码、构建和测试的独立审计。它只保存证据、风险和重构依据，
不是新的总体架构合同。审计后采用的边界必须回写对应编号设计文档，执行状态以
`tasks/progress.md` 为准。

## 1. 审计目标和方法

本轮不把 2026-07-10 形成的长期设计和实施计划当作事实源，而是从当前可运行 pipeline 重新回答：

- 当前 source program 实际经过哪些 IR / artifact；
- 每层由谁解释、验证和消费；
- 哪些已声明对象没有生产实现或 consumer；
- 哪些路径会静默误编译、发布 partial artifact 或恢复错误 ABI；
- 单 tile、单卡 16-tile 和 tiny Llama 纵向闭环真正需要哪些最小边界。

证据来自当前 production code、ODS、CMake、lit/CTest、定向失败复现和 Git 历史。历史审计和计划只用于
寻找待验证假设，不用于证明结论。

### Pipeline position

```text
Pipeline position:
- Upstream artifact / IR:
  当前 verified StableHLO program directory、现有 Wafer IR、target CRT/device-link、JSON package/runtime 工具。
- Current stage responsibility:
  重新建立实现事实矩阵，区分必须保留、必须重构、删除和延后的边界，并给出近期单卡纵向实现顺序。
- Output artifact / IR:
  风险分级、事实矩阵、设计收缩结论、旧任务映射和可验证的新任务 DAG。
- Downstream consumer:
  tasks/01-16 对应设计修订、single-card vertical-slice 实施计划和后续代码重构。
- User-level driver / named pipeline:
  审计对象包括当前 wafer-opt program mode、IR-local named pipelines、wafer-compile-stablehlo、
  device-link、package/runtime 和 check-wafer；审计本身没有用户编译入口。
- Explicit non-goals:
  不把审计建议当作已实现功能；不在本文件定义第二份总体架构；不以 no-card/reference 证据冒充板端证据。
- Completion gate:
  所有 P0/P1 结论有代码或新鲜复现证据；采用的边界同步编号设计和 progress；旧长计划退出 active DAG。
```

## 2. 基线事实

### 2.1 文档与实现规模

- `tasks/01-16` 在 2026-07-10 从 10,276 行增长到 14,234 行，增加 3,958 行。
- 同日新增 7 份 active plan，共 13,304 行、68 个 Task；对应提交没有 production compiler/runtime 代码推进。
- 7 份计划引用的 565 个可解析 literal path 中，475 个（84.1%）当前不存在。
- `ProgramOutputTransaction`、`TargetArtifactSet`、`PackageManifest`、`RankClassId`、`ProjectionSetId`、
  `WCRE` 和 `KernelAbiDescriptor` 在 production code 中均无实现。
- 当前真实实现为 47 个 Wafer ODS op、15 个 pass、9 个 IR-local named pipeline；不存在
  `wafer.model.*`、`wafer.distributed.*`、`wafer.parallel.*`、`wafer.executable.*` 或
  `wafer.target.environment`。

这说明长期对象只能作为待验证方向，不能继续阻塞当前 correctness 或被写成已收敛实现合同。

### 2.2 当前真实 pipeline

```text
exported/pre-exported StableHLO program directory
  -> frontend metadata/payload verification
  -> target topology + execution mesh materialization
  -> Shardy propagation + external XLA SPMD helper
  -> StableHLO collective normalization + StableHLO-to-Linalg
  -> wafer.group formation
  -> group-to-tile-region + one-shot bufferization
  -> tile-region-to-instr
  -> SPM / DDR planning
  -> target LLVM CRT calls
  -> device object + repo-local CRT + kcore shared object
```

`wafer-opt --program-pipeline` 当前只支持 `stablehlo-spmd`、`stablehlo-spmd-to-linalg` 和
`stablehlo-spmd-to-group`。真实模型测试在 program mode 后再次手工调用 IR-local pass pipeline；
`wafer-compile-stablehlo` 只做验证。不存在 source-to-executable 的单一用户入口。

package/runtime 是另一条旁路：Python 从打印的 instruction/LLVM 文本恢复 schema-v2 JSON，Python 和 C++
分别解释部分 schema，`wafer-run` 只生成 launch plan、检查动态库 symbol 并明确不执行 board launch。

### 2.3 新鲜测试基线

- `ctest --test-dir build/wafer-dev --output-on-failure`：3/3 通过。
- lit：205 discovered，204 passed，1 unsupported；unsupported 是 enabled build 下的
  `wafer-compile-stablehlo-disabled.test` feature-inverse case。
- `check-wafer` 会运行 lit 并构建 `WaferUnitTests`，但不会执行 C++ gtest；真正执行只注册在 CTest。

测试通过只能证明现有局部合同，没有覆盖下面的静默误编译和 publication 问题。

## 3. 风险分级

### P0-1 Candidate commit 只提交首个代表 tile

`SelectGroupTile.cpp` 只构造 first/tail representative 做 candidate 验证，最终仅保留第一个 representative
module 并替换原 group；`WaferGroupToTileRegion.cpp` 也只 materialize 调用方给出的单组 offsets/sizes。

定向复现把 multi-output traversal 放大后，selector 选择小于完整 traversal 的 tile，但提交 IR 只包含
第一个 subview/WDMA，剩余输出保持旧值。这是 silent partial output，不是搜索质量问题。

临时边界：正式完整 traversal materialization 落地前，mandatory selector 只接受 tile size 等于完整
traversal shape；candidate 抽样只能用于早期拒绝，不能成为 commit artifact。

正式门槛：commit 必须 materialize 全部静态 traversal（包括 tail），随后在完整 entry 上重跑
instruction、SPM、DDR 和 verifier；reference executor 证明无 gap/overlap。

### P0-2 Target LLVM flatten 结构化控制流和 call

`LowerInstrToTargetLLVM.cpp` 递归 walk function 内全部 instruction，按 walk 顺序在单一 LLVM entry block
发 call，最后生成一个 return 并删除原 function。`scf.if` false branch、不同 loop trip count、CFG 和
callee-only instruction 都会被无条件线性执行。

临时边界：在任何 mutation 前拒绝 multi-block function、call 和除单 block `wafer.tile.region` 外的
nested region；整个 module 在 clone 中转换，失败时原 module byte-identical。

正式门槛：使用 structure-preserving dialect conversion 在原 SCF/CF/function 位置 lowering leaf op，
覆盖 false branch、0/2 trip loop、diamond CFG 和 call。

### P0-3 Async WDMA source 可在完成前复用

WDMA 的 resource effect 是 SPM read + DDR write + movement issue；当前 SPM planner 只延长 SPM write 的
pending lifetime，忽略 async issue 的 SPM read。新鲜复现出现 WDMA source offset 被紧随其后的 compute
destination 立即复用，中间没有 completion/fence。

门槛：所有 async issue 的读写资源都必须活到对应 completion；无 token/fence/engine completion proof
时不得复用。测试覆盖 fence 前不可复用、fence 后可复用，以及分支/循环和跨 tile-region。

### P0-4 Instruction geometry 和 ABI narrowing 不闭合

- RDMA/WDMA 未在 op verifier 同时证明 descriptor payload、DDR/SPM 两端 physical range。
- DTE bytes 未证明小于 buffer physical bytes。
- convert 未证明 source/dest element count 相等。
- conv/pool/unpool shape attrs 未与 memref shape 和算子关系闭合。
- target lowering 将多个 i64 attr 直接构造成 i32；CRT 又将 shape 收窄到 uint16。

门槛：建立共享 physical geometry verifier，由 op verifier、memory planning 和 pre-target legality 共用；
覆盖 payload equality、两端范围、shape relation、element count 和所有 uint32/uint16 narrowing。

### P0-5 Package ABI slot 不是双射

当前 exporter 从 LLVM 参数数量猜 workspace，却用只包含 input/output 的自由字符串 `binding_order`；
Python validator 只检查名字存在，不检查重复、完整覆盖和 signature 双射。Python/C++ runtime 都按该列表
组装实参。重复 `lhs` 的 package 会被接受并构造两个 lhs 参数。

门槛：production package 在 typed slot/resource 双射落地前拒绝 parameter/workspace；最终以 C++ typed
`SlotId -> ResourceId` 唯一表示 ABI，不再接受自由字符串顺序。

### P0-6 Device link 在 late validation 前发布 final output

`wafer_device_link.py` 直接向调用者给定的 object/CRT/final `.so` 路径写入，完成 link 后才检查
undefined Wafer symbol。缺失 symbol 会返回失败，但 final `.so` 和中间 object 已可见。

门槛：所有输出写入 transaction-owned staging root；symbol、ELF、digest 和 manifest 全部验证后单次
rename/CAS 发布。late failure 后 final root 不存在或旧版本 byte-identical。

## 4. P1 架构缺口

1. rank identity 由 group lowering API/pass 的默认 0 承担；mandatory selector没有 rank 输入，16-rank
   program 会 materialize rank-0 peer/slice。近期采用显式 per-rank static clone，删除生产默认值。
2. frontend parameter shard verifier只检查单 shard 范围和 rank 数量，不证明 partition coverage、gap、
   overlap 或与 sharding relation 一致。
3. Python/C++ 各自拥有 package schema、executor、RuntimeSession 和 required-symbol 规则；C++ 接受未知
   schema version、bogus ABI/format 和重复 slot。
4. exporter通过正则解析 instruction/LLVM printer文本并把 instruction schedule复制进 package，形成
   shadow schedule 和旁路 legality。
5. 当前 runtime只做 metadata plan/symbol discovery，不执行 allocation/load/copy/launch/completion。
6. 当前没有真实 source-to-target 用户 driver，现有集成测试长期依赖手工拼 program mode 和 pass pipeline。
7. `check-wafer` 名称暗示完整 gate，但不执行 C++ unit tests。

## 5. 保留、重构、删除和延后

### 保留

- verified program directory 与 framework/exporter metadata/payload 校验；
- `wafer.target.topology`、`wafer.execution.mesh`；
- LinalgExt collective、`wafer.group`、`wafer.tile.region`；
- Wafer memref memory/layout、SPM/DDR accepted offsets；
- tile compute/movement、instruction、Direct DTE、local fence；
- proposal/accepted 分离、显式 rank、失败不污染主 IR；
- package不复制 instruction schedule、runtime不重新 planning；
- no-card、reference executor、board evidence三种证据严格分开。

### 近期重构

- target structure-preserving conversion、完整 traversal commit、shared geometry、async lifetime；
- typed compile request/result 与 per-rank static executable bundle；
- 单一 C++ typed manifest + canonical JSON verifier/serializer；
- transaction-owned target link和atomic bundle publication；
- C++ reference executor和实际执行的 test gates。

### 从当前主线删除

- 把不存在的 `stablehlo-to-executable`、Protobuf manifest、WCRE或各种 registry写成当前事实；
- default rank 0、pass option和名字承担生产 identity；
- Python/C++ 多份 schema、instruction text semantic recovery和package shadow schedule；
- 用局部 FileCheck、JSON roundtrip或symbol closure宣称纵向完成。

### 延后

- 新的 model/distributed/parallel/executable dialect家族；
- hybrid rank-class dedup、MPMD、跨卡 coherent variant；
- WCRE、全局 semantic ID registry、Protobuf admission framework；
- capability lease/generation/trust lattice、cross-model migration、共享 weight/cache service；
- MoE、70B/100GB、10万/100万record stress和cost calibration；
- 完整ELF ABI-note/fingerprint体系。近期只保留必要module/rank/entry/digest绑定。

## 6. 采用的近期边界

近期同时证明 rank-count=1 和单卡 rank-count=16，二者走同一 compile API。V0 使用 per-rank static
executable，不引入新的 executable dialect：

```text
CompilationRequest
  -> verified source + target config + explicit execution ranks
  -> isolated per-rank module clones
  -> complete traversal/candidate/instr/SPM/DDR/target verification
  -> RankExecutable[]
  -> atomic ExecutableBundle
  -> typed C++ PackageManifest + canonical JSON
  -> no-card RuntimeSession + reference executor
```

首个 gate 使用真实 exporter 产生的 linear-residual/MLP；第二个 gate 使用 tiny Llama decoder block。
没有稳定 board 时，reference executor证明中层语义，target artifact只证明结构/ABI，board numeric和真实
transport/completion保持 external gate。

## 7. 旧任务映射

- 原 Q0.C/Q0.1/Q0.L 合并到近期 `target-correctness`，不再被 Proto/WCRE/transaction registry 阻塞。
- 原 Q0.F/Q0.O/Q0.2/Q0.3/Q0.4 收缩为 typed compile request、显式 per-rank clone、bundle commit；
  未实现的 registry/lease/rank-class不进入近期任务。
- 原 Q0.A 收缩为 transaction-owned module link、必要 digest和atomic artifact bundle。
- 原 Q4/Q6.N 收缩为 typed C++ manifest、canonical JSON、no-card runtime；复杂 shared service延后。
- 原 Q5/Q7 收缩为 linear/MLP和tiny Llama纵向gate。
- 原 Q6.B/Q8/Q9保留为外部board/长期扩展，不作为近期 compiler correctness前置。

## 8. 审计结论

当前最需要的不是实现 2026-07-10 长计划中的基础设施，而是把 P0 correctness 从这些未验证对象的
依赖后面移到当前 instruction/target pipeline 前端，并建立一个最小、原子、可解释的单卡纵向切片。

现有 group/tile/instr/memory/topology 分层大体可保留；必须重构的是完整 traversal、control-flow、
physical geometry、async completion、rank identity、package ABI单源和artifact publication。代码拆文件应在
这些语义边界稳定后进行，不能用目录整理代替正确性修复。
