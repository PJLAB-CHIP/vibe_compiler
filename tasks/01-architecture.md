# Wafer AI Compiler Architecture Design

状态：本轮长期架构边界合同已收敛；子设计边界索引见第8节，实现状态以`tasks/progress.md`任务队列为准。


说明：本文使用 **Wafer** 作为目标硬件和软件栈名称。底层公开文档、依赖和已有后端中仍可能出现 TX8/TX81 等历史命名，本文只在引用事实时保留这些名称。

本文只定义 Wafer compiler 的 IR 分层、阶段责任和子设计边界。专题细节以第 8 节列出的
子设计文档为准；本文不复制 layout/SPM/DDR/compute/communication 的算法细节，也不把
历史 backend、旧 CRT、runtime 路径或单个 workload 写成架构合同。

本文的一级依据是：

- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/README.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`

## 1. 背景和目标

目标是构建一套面向 Wafer 硬件的 AI compiler/runtime 栈，使上游模型可以通过标准图编译路径运行到
Wafer 多 tile / 多卡系统上。PyTorch 是重要入口之一，但不是 Wafer 后端的唯一或长期 IR 边界。

主路径选择 verified Wafer program、Shardy/SPMD 和 Wafer IR contract，而不是从历史
backend、某个 importer 或 runtime wrapper 反推整套架构：

```text
source model / exported program / pre-exported StableHLO
  -> frontend importer adapter
  -> verified program: StableHLO + typed IO/parameter/persistent-state ABI + symbolic bounds
  -> target environment / topology / execution-mesh selection
  -> Shardy/SDY propagation + SPMD/MPMD distributed program
  -> component-local Linalg/Tensor/SCF compute + tensor collectives
  -> logical group and bounded candidate planning
  -> complete traversal/layout/instruction/SPM/DDR/transport/event proposals in a variant clone
  -> whole-variant legality gate across every component and logical rank
  -> atomic commit to `wafer.executable` / static `wafer.executable.variant`
  -> rank-class mapping + static rank `func.func` programs + typed resources/transport/completion
  -> target LLVM + compiler-generated Kernel ABI descriptors
  -> device-code compile/link + ELF descriptor/module digest/environment and artifact fingerprints
  -> Protobuf PackageManifest assembly
  -> RuntimeSession validates, selects one coherent variant, binds resources/endpoints, launches and completes
```

核心判断：

- Shardy/GSPMD 负责全局张量的逻辑切分和 collective 插入。
- `wafer.target.environment` 拥有 target legality capability；`wafer.target.topology` / execution mesh
  拥有部署 rank domain。calibration profile 只影响 cost 和 candidate 选择，不是程序语义。
- distributed program 显式保存 component/stage、partition/replica coordinate、`dp/tp/pp/ep` axes、rank
  group 和 shard relation。flat logical rank 只是 mesh coordinate 的可派生 ordinal，不能代替这些身份。
- 长期交付采用 hybrid executable set：上层保留 rank-parametric distributed semantics，commit 后产生
  static rank program；只有 local IR、ABI、memory plan 和 transport template 等价时多个 ranks 才共享
  rank class。首版可以保守退化为每 rank 一个 class，不能默认 rank 0。
- SPMD 后的 StableHLO collective 先规整成 `wafer.linalg_ext.collective.*` tensor-level
  collective，和 local compute 一起进入 group/tiling；到 `wafer.tile.region` / SPM buffer
  materialize 之后，再 lowering 到 buffer-level `wafer.tile.*` collective 和 instruction-level
  Direct DTE/FSM/SPM-sync 协议。
- `wafer.group`、layout、memory 和 transport 结果在 candidate clone 中只是 proposal。只有完整 static
  distributed variant 的 traversal、layout、SPM/DDR、transport、completion 和 target ABI 同时通过，
  才原子 commit；`DirectFullShape` 是普通 candidate policy，不是 production bypass。
- dynamic batch/sequence、KV/persistent state、target variant 和 rank class 在上层/executable 层表达；
  target instruction program 可以继续静态。所有 ranks 必须选择同一个 coherent distributed variant。
- PackageManifest 是 committed executable 的不可变交付表示，不复制 instruction schedule；RuntimeSession
  只做 capability 校验、variant/template 选择、typed binding、launch、completion 和 error aggregation，
  不重新做 sharding、placement、memory 或 transport planning。
- Wafer 后端不以 LLVM target intrinsic 为核心抽象；V0/V1 通过 target CRT 调用 public Tsm wrapper / Kcore runtime，下发 NE/CT/LSU/DTE 等硬件任务。
- 硬件事实可以作为 pass 的 legality/cost input，但必须在合适 IR 层级 materialize，不能污染上层语义 IR。

### 1.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verified program、typed model/resource ABI、symbolic shape bounds、target environment、topology snapshot
  和 deployment policy。
- Current stage responsibility:
  依次形成 distributed program，执行 bounded candidate planning，在完整 distributed variant clone 上验证
  traversal/layout/instruction/SPM/DDR/transport/event/target ABI，并原子 commit executable set；随后只从
  committed facts 生成 target modules、Kernel ABI descriptors 和 PackageManifest。
- Output artifact / IR:
  `wafer.executable` set、static `wafer.executable.variant`、rank-class/static-rank entrypoints、typed
  resources/transport/completion、target modules、Kernel ABI descriptors 和 Protobuf PackageManifest。
- Downstream consumer:
  RuntimeSession materialization、module/weight cache、persistent-state registry、typed launch、completion
  和 error aggregation。
- User-level driver / named pipeline:
  当前 program pipeline 是分阶段调试入口；主线完成前必须提供单一
  `wafer-opt --program-pipeline=stablehlo-to-executable` 或等价 compile driver，内部调用同一 named
  pipelines，不能要求用户手工拼 pass。
- Explicit non-goals:
  runtime 不重新做 sharding、rank placement、candidate/layout/memory/transport planning；PackageManifest
  不复制 instruction schedule；低层 instruction 不承担任意 dynamic-shape dispatch。
- Completion gate:
  真实 exported model 至少覆盖 coherent multi-rank variant、两组 shape guards、persistent state 和
  target/package ABI；输出经过 mandatory atomic commit、target LLVM、实际 device link、manifest
  semantic validation 和 no-card session materialization。板端 gate 再证明 numeric、completion 和
  failure aggregation。
```

## 2. IR 分层总原则

Wafer 是基于 MLIR 的编译器。每一层 IR 只携带自己能稳定解释、变换和验证的信息。

| 阶段 | 主体 Dialect / IR | 允许表达 | 不应提前表达 |
| --- | --- | --- | --- |
| Verified program | StableHLO, func, tensor, typed program metadata | 模型语义、symbolic bounds、typed IO、immutable parameter、persistent mutable state/alias | rank placement、SPM/DDR offset、runtime handle |
| Target environment / mesh | `wafer.target.environment`, `wafer.target.topology`, `wafer.execution.mesh` | target capability、topology snapshot、logical rank domain、prevalidated endpoint policy | tensor sharding、candidate plan、物理地址 |
| Distributed program | StableHLO + Shardy/SDY/MPMD | component/stage、partition/replica coordinate、`dp/tp/pp/ep` axes、collective/shard relation | physical transport、SPM buffer、target packet |
| Local tensor program | Linalg, Tensor, SCF, Arith, Math, `wafer.linalg_ext.collective.*` | component/rank-local structured compute、state use-def、tensor collective | physical layout、endpoint/channel、package ABI |
| Candidate planning | logical/candidate `wafer.group`, `wafer.tile.region`, target-abstract Wafer ops | bounded traversal/layout/instruction/memory/transport/event proposals | accepted executable/package事实 |
| Executable composition | `wafer.executable`, `wafer.executable.variant`, static rank `func.func`, typed resources | coherent variant guard、rank class/entry、accepted layout/offset/transport/completion、完整 traversal | planner trace、rejected candidate、runtime handle |
| Target code | `wafer.instr.*`, LLVM dialect, Kernel ABI descriptor, ELF module | 静态 rank program、target CRT calls、module ABI/fingerprint | sharding/search、package shadow schedule |
| Package/runtime | Protobuf PackageManifest + RuntimeSession | immutable artifacts/entry graph/resources/completion template；actual handles/selected variant | 重新规划 placement/memory/transport、物理地址序列化 |

IREE 的可借鉴点是层级纪律，不是 dialect taxonomy。Wafer 不复制 Flow/Stream/HAL/VM 分层，也不把
IREE 的 experimental op 当成硬件无关答案；但采用三个原则：

- Flow-like 原则：dispatch / group formation 只形成可验证执行单元，不提前决定 runtime buffer、
  physical device handle 或 final launch packet。
- Stream-like 原则：只有在 async scheduling、resource lifetime、range access 和 wait/completion
  需要跨 stage 保留时，才引入显式 resource / token / effect / range 表示；planner 搜索过程和
  resource estimate 不落 IR。
- HAL-like 原则：device binding、allocation/import/query、fence/completion、package metadata 是
  runtime boundary 的 late materialization；上层只保留 target environment、distributed identity、
  committed executable/resource/transport/event 和 accepted offset 这些不可从 local IR 重算的事实。

两条 layout 线必须分开：

- `layout` 是 tensor semantic layout，只说明维度业务含义和 op 如何解释 shape，例如 feature `NHWC`、conv weight `HWOI/HWIO`、普通 GEMM matrix。它可以出现在 tensor/group 层。
- physical layout marker 说明 SPM/DDR 中真实组织形式，例如 `Tensor`、`NTensor`、`Cx`、`NCx`。它只能在 `wafer.tile.region` / SPM bufferization 之后出现在 Wafer-tagged memref 上。Cx/NCx 的 `C0`、storage bytes 和 256B padding 是从 shape、dtype 和 target policy 推导出的 layout info，不写进 marker 本身。

Wafer buffer value 使用 MLIR `memref`，但 memref memory-space slot 中放 Wafer target attr：

```mlir
memref<64x256xf16, #wafer.memory<spm, tensor>>
memref<64x256xf16, #wafer.memory<spm, cx>>
memref<64x256xf16, #wafer.memory<ddr, tensor>>
```

`#wafer.memory<space, layout>` 是统一的 addressable storage space + physical layout marker。
V0 至少区分：

- `#wafer.memory<spm, *>`：tile-local SRAM，由 SPM bufferization / allocator 负责容量、
  lifetime、range 和 reuse。
- `#wafer.memory<ddr, *>`：device/global DDR 的目标侧地址空间，由 DDR memory planning
  验证 external view/descriptor demand，并为 compiler-managed DDR `memref.alloc` 写入 accepted
  offset facts；launch/runtime/package 负责把已接受的 demand 映射到运行时分配对象。

SPM allocator 只分配 `spm` buffer，不代表 IR 里没有 `ddr`。RDMA/WDMA、host-visible input/output、
constant load 和 runtime allocation object 都应通过同一套 memory-space / effect / verifier 体系表达
source/destination address space；差别在于 owner、lifetime 和 lowering 层级不同。DDR 的 default
arena、external view/descriptor validation、compiler-managed/resident accepted offset facts、constant
residency 和 bandwidth 设计见 `tasks/12-ddr-memory-planning.md`。

`Tensor_Fmt` 不能作为 compiler layout 模型。layout conversion 是真实 data movement，不是
metadata reshape；它先在合适层级表达成 `wafer.tile.materialize_layout` 或等价 movement op，再由
lowering 选择 `ChannelNorm`、`DechannelNorm`、`GatherScatter`、TDMA 或 wrapper path。默认策略是把
`Cx/NCx` materialization 延后到 aligned-only 指令边界；如果 planner 为了复用或减少重复转换选择
更早 materialize，必须显式承担 SPM footprint 和额外搬运代价。layout materialization 的
算法、boundary contract 和 cost model 见
`tasks/08-layout-materialization.md`。

constant 语义同样分层：

- Frontend / StableHLO 阶段允许 `stablehlo.constant` 或 exporter-native program directory 中的 weight data。
- 进入 Linalg / Wafer planning 前，常量统一成 `arith.constant` 或其它 `ConstantLike` tensor op；
  大 tensor 可以使用 resource-backed elements attr。
- Wafer 不定义私有 tensor constant op。constant 不是 group external input，但 tile execution 中仍要
  通过 `wafer.tile.load` 从 device-addressable storage 读入；constant storage transform / load
  lowering 直接改写 backing data/resource 或生成 packed storage，并把 read-only DDR demand 交给
  DDR memory planner。
- immutable parameter 在 executable/package 层拥有稳定 resource id、content digest、shard relation、
  storage encoding/quant descriptor 和 residency scope；文件路径和 SSA 名只用于序列化/诊断。
- 能被目标 op 合法 fold 成 immediate、attribute 或 fill pattern 的 scalar/splat/small constants
  不产生 DDR demand；需要作为 tensor tile data 读取的 constants 才进入 load/DDR 路径。
- Weight 切分由 consumer op 的 tiling/indexing relation 推出；constant storage transform 可以选择
  whole backing、chunked backing 或 streaming，但不能在 constant/DDR 层隐式改变 compute tile 或
  reduction split。

## 3. 阶段和 Dialect Ownership

### 3.1 Frontend Program Stage

主体 IR / Dialect：

- `stablehlo`
- `func`
- `tensor`
- `arith`
- optional exporter-native weight metadata

输入：

```text
source model / exported program / pre-exported StableHLO
  -> frontend importer adapter
  -> StableHLO / MLIR module
```

职责：

- 保留模型语义、dtype、rank、symbolic shape 和 verified bounds。
- 为 user IO、immutable parameter、persistent mutable state 建立稳定 value/resource identity；显式保存
  access、alias/update relation 和跨 invocation 语义。paged state 是通用 persistent-state descriptor，
  KV cache 只是其中一个 consumer。
- 保留或导入用户/框架侧 sharding 标记。
- 决定 weight 表达方式和 content identity：StableHLO constant 或 exporter-native program
  metadata/payload；shard、packing/quant 和 residency 在后续层决定。
- freeze weights 的逻辑保持硬件无关，命名和实现不绑定 Wafer。

不负责：

- 不表达 Wafer tile endpoint mapping。
- 不表达 Wafer memory attr、physical layout marker、DTE、NCC queue、worker、fence/wait。
- 不把 runtime launch 或 package ABI 写进模型 IR。
- 不从参数名、文件名或模型层名识别 state/KV/weight role。

V0 策略：

- 稳定 compiler 入口是 verified Wafer program。
- 主链路完成证明优先来自真实 framework/exporter 产生的实际图 program，例如 PyTorch/XLA、
  JAX 或其它 exporter 导出的 StableHLO / MLIR。手写 StableHLO 只保留为 pre-exported program
  测试输入、verifier negative test 或局部 lowering bring-up，不能证明 framework-specific capture 已完成。
- 具体 importer API 不是 Wafer 后端合同；后端只消费 verified program 和已 materialize 到 IR 的
  importer facts。
- 先接受静态或 bounded dynamic shape；bounds进入 executable variant guard，不直接变成任意动态
  target instruction descriptor。任意 PyTorch eager 动态行为不是 compiler core 合同。
- 支持范围由 exporter program 的合法语义、Wafer 硬件能力和当前 IR contract 决定；当前某个后续
  lowering / endpoint / runtime 边界存在实现缺口，不能反向成为 frontend、SPMD 或 planner 的不支持
  理由。若硬件可表达但 IR/lowering 未覆盖，必须补 IR contract 或下游恢复任务。

Frontend program 的模型导入、第三方依赖组织、constant/weight、sharding annotation 和验证合同见
`tasks/02-frontend-stablehlo-program.md`。

### 3.2 Target Environment / Topology / Execution Mesh Stage

主体 IR / Dialect：

- `wafer.target.environment`
- `wafer.target.topology`
- `wafer.execution.mesh`
- MLIR DLTI 可稳定承载的通用 target/data-layout facts

输入：

```text
verified program requirements
  + compiler target descriptor / runtime capability snapshot / board profile
  + optional deployment isolation policy
```

职责：

- 在 SPMD 前 materialize target legality environment：revision、triple/ABI、memory/engine/DTE limits、
  dtype/layout/packet capability、errata 和 compatibility fingerprint。
- 保存部署 topology snapshot、unavailable endpoint 和 topology digest；从中选择带 `dp/tp/pp/ep`
  axes 的 logical execution mesh/rank domain。
- 区分 capability fingerprint、topology snapshot digest 和 calibration provenance。前两者影响 legality、
  deployment variant 和 cache identity；calibration 只影响 cost/selection。
- 为后续 pinned projection 或 compiler-generated relocatable projection template提供唯一 topology事实源。

不负责：

- 不做 tensor sharding、rank-local program selection、candidate/layout/SPM/DDR/transport planning。
- 不在 pre-SPMD stage 保存最终 per-rank entrypoint、channel/FSM 或 runtime handle。
- 不把硬件 profile估算值写成 IR 语义。

post-SPMD launch/transport projection 属于 executable composition：它把已经存在的 component/rank
coordinate绑定到 accepted rank class、entrypoint、endpoint和transport resource。`pinned` variant编码
完整 projection并绑定 topology digest；`relocatable` variant只能选择 compiler生成的有限 templates或
确定性 control-table binding，runtime不能重新搜索 placement/route。

Topology / target environment / execution mesh 的详细合同见
`tasks/04-topology-execution-mesh.md`。

### 3.3 Sharding, SPMD and MPMD Distributed Program Stage

主体 IR / Dialect：

- `stablehlo`
- Shardy / SDY / MPMD component representation
- collective ops in partitioned StableHLO

输入：

```text
verified StableHLO + typed resource/state ABI + sharding seeds
  + wafer.target.environment
  + wafer.execution.mesh
```

处理：

```text
sharding import / seed
  -> Shardy/SDY propagation
  -> SPMD partition and MPMD component formation
  -> verified distributed program
```

输出必须显式表示：

- MPMD component/stage identity 和 stage/value dependency。
- partition coordinate、replica coordinate、`dp/tp/pp/ep` mesh coordinate 和可派生 flat rank。
- component/rank-local StableHLO body、logical rank groups、collectives 和 parameter/state shard relation。
- bounded dynamic shape symbols，作为后续 coherent distributed variant guard输入。

长期优先复用 Shardy MPMD/SDY 的稳定表示；当前 pin 不具备所需接口时先升级依赖或使用薄 adapter，
不建立平行的 Wafer 私有 MPMD 事实源。

hybrid rank合同：rank-parametric distributed semantics可以保留 partition/replica SSA；per-rank constant
specialization只能由显式 specialization record产生。后续 rank class共享需要 executable verifier证明
local IR、static shape、ABI、memory plan和transport template等价；不允许 pass option或默认 0承担身份。

所有 ranks必须选择同一个 coherent shape/target distributed variant；不能由每个 rank在 runtime独立
判断 guard。该层不选择 physical endpoint、DTE algorithm、FSM/channel、SPM sync slot或target packet。

Shardy / distributed program 的详细合同见 `tasks/03-shardy-spmd.md`。

### 3.4 Local Compute Normalization Stage

主体 IR / Dialect：

- `linalg`
- `tensor`
- `scf`
- `arith`
- `math`
- `wafer.linalg_ext.collective.*` ops

输入：

```text
distributed component/rank-local StableHLO compute ops
  + logical collective ops
  + typed parameter/state boundary
```

输出：

```text
tensor-level Linalg/Tensor/SCF/Arith/Math local shard program
  + tensor-level collective ops
  + explicit parameter/state SSA use-def and alias/update relation
```

职责：

- 把 StableHLO compute lowering 到 structured compute。
- 把 StableHLO logical collective 规整成 `wafer.linalg_ext.collective.*` op，使 collective
  仍保持 tensor semantics、DPS / tiling interface 和 combiner region，而不是提前进入 tile-local
  communication IR。
- 把 `stablehlo.constant` 统一成后续 pipeline 可处理的 `arith.constant` 或其它 `ConstantLike`
  tensor op；大 tensor / weight 可以使用 resource-backed elements attr，但不引入 Wafer 私有
  tensor constant op。
- 保留 indexing map、iterator type、DPS operand/result 绑定关系。
- 保留 persistent state 的 SSA read/update/alias语义和 component boundary；不把 state降成普通无身份
  temporary，也不从 buffer名恢复 role。
- 为 tiling、fusion、bufferization 提供可分析结构。
- 维护 semantic layout、dtype、rank、shape，但不引入 Wafer memory attr 或 physical layout marker。

不负责：

- 不表达 NE/CT/LSU/DTE。
- 不表达 SPM offset、bank/color、worker、queue、packet field。
- 不生成 `wafer.tile.*` communication、SPM communication buffer 或 Direct DTE token。
- 不生成 target CRT call。
- 不生成 transformed constant storage；constant storage transform 是后续 layout/load lowering 阶段职责。

设计原则：

- Linalg 是结构化计算表达，不是最终硬件计划。
- 历史 backend 观察只能作为实现证据或反例，不决定当前 IR 边界。
- StableHLO collective 不在本阶段直接 lower 成 `wafer.tile.*` communication。需要跨 tile 的通信语义先作为
  tensor collective 进入 group/tiling；physical communication 在 storage 和 endpoint facts 明确后
  materialize。

Local compute normalization 的 StableHLO-to-structured-IR 合同、transformer block 所需
dot/broadcast/reduce/softmax/norm/RoPE 表达和 verifier 见
`tasks/05-local-compute-normalization.md`。

### 3.5 Group Scheduling Stage

主体 IR / Dialect：

- `wafer.group`
- `scf`
- `tensor`
- optional transform dialect schedule dump/replay

职责：

- logical `wafer.group` 表达 fusion boundary和structured tiling demand；scheduled form只允许存在于
  candidate/template clone，表达待验证 traversal proposal，不是 accepted stage。
- 把 producer/consumer 拉进同一个 tile schedule，而不是逐 op materialize full tensor。
- 通过 IR 结构、SSA use-def 和必要的 op/effect 表达 schedule 后仍需要保留的约束。
- 为后续 `wafer.tile.region` 构造提供 tile-local storage、communication staging need、sync need、
  compute/DMA/communication pressure 等 analysis 输入。
- `DirectFullShape`、tiled traversal和split方案属于同一个 bounded candidate frontier，必须走相同
  layout/instruction/SPM/DDR/transport/event gates。

不负责：

- 不表达 SPM physical address。
- 不表达 `Cx/NCx` storage。
- 不表达 NCC queue、worker id、DTE node id、FSM id、packet id。
- 不表达 target CRT call。
- 不单独 commit representative tile、per-group memory plan或scheduled group；只有完整 distributed
  variant通过后才有 accepted executable。

`wafer.group` 的详细设计见 `tasks/06-group.md`。该文档是 group 语义和
tile-and-fuse 的主文档，本架构文档只规定它在全 pipeline 中的位置。

### 3.6 Tile Region, Bufferization and Layout Materialization Stage

主体 IR / Dialect：

- `wafer.tile.region`
- target-abstract `wafer.tile.*` compute / movement / boundary ops
- Wafer-tagged `memref`
- `memref`
- `scf`
- `wafer.tile.materialize_layout` 和下游 movement ops

职责：

- 在 candidate clone 的完整 traversal 中引入 `wafer.tile.region` 作为一次 tile-local execution scope；
  它不是独立 executable，也不能单独提交 representative tile。
- 把 candidate tiled SSA graph materialize 成 tile-local storage、movement、layout conversion、
  target-abstract compute、communication 和 sync/effect op。
- 在 planner candidate evaluation 中根据 candidate tile shape 把 DDR boundary materialize 成显式
  `memref.subview` tile view，让 RDMA/WDMA descriptor 从 IR view 推出。
- 对 target-abstract op 先做 Wafer instruction legalization / selection，在 Wafer-tagged
  memref graph 上产出 instruction-level `wafer.instr.*` IR；SPM/DDR memory
  planner 只消费 instruction-level IR 的 memref use-def、effect 和 lifetime，
  不从高层 op 名或单个 case 猜 demand。
- group-level layout/SPM/DDR结果只是 feasibility proposal；最终 layout/materialization、SPM/DDR offsets
  和跨 group lifetime必须在完整 static rank entry 上重算并验证，成功后才进入 committed variant。

`wafer.tile.region` 不重新做 group formation、root tile search 或 traversal selection，也不表达
host launch/package ABI；candidate region不能被 package/target pipeline直接消费。详细合同见
`tasks/07-tile-region.md`。
layout materialization 见 `tasks/08-layout-materialization.md`；SPM
bufferization 见 `tasks/09-spm-memory-planning.md`；DDR memory planning 见
`tasks/12-ddr-memory-planning.md`。

### 3.7 Wafer Instruction Legalization / Selection Stage

主体 IR / Dialect：

- target-abstract `wafer.tile.*` compute / movement / boundary ops
- `wafer.instr.*`
- `wafer.instr.local_fence`

职责：

- 输入是 target-abstract `wafer.tile.*` compute / movement / boundary op；这些 op 已经在 layout materialization
  前进入 IR，并提供 layout contract。
- 如果 RDMA/WDMA 读写的是 DDR tile slice，输入 IR 必须已经通过 `memref.subview` / strided
  memref view 表达该 slice；instruction legalization 只消费 view，不从 tile shape 自己恢复 view。
- 将 target-abstract op lower 到复用 Wafer-tagged memref 的 instruction-level
  `wafer.instr.*` IR，再由 SPM memory planning 在同一 IR 上填入 offset/range/bank；
  这一层不直接手写 raw packet bitfield。
- 覆盖 CT、NE、RDMA、WDMA、TDMA 和 DTE 的 hardware invocation family、issue/fence/wait
  抽象和 memref read/write/effect。
- issue op通过 `async.token` / typed effect暴露 completion domain；wait/fence不能混淆 local NCC、DTE、
  host command和multi-rank barrier。hardware busytable只有被 target environment/runtime验证后才能作为
  legality input。
- shared geometry verifier从 memref/layout推导 physical interval，证明 byte/count/iteration、shape、
  canonical GEMM mapping和所有 target integer narrowing；lowering不能补猜未验证字段。
- 为 verifier 提供明确的 legality target。

这里产出的 instruction IR在 candidate clone中仍是 proposal。只有完整 traversal、whole-entry
memory/event和variant-set transport/ABI全部通过后，static rank program才成为 committed instruction
artifact。

`wafer.tile.*` compute / movement 的 IR 生命周期、op family、layout/resource interface、issue/fence
模型和 lowering 合同见
`tasks/10-compute-movement.md`。该子设计固定的是 target-abstract
compute/movement 层的 verifier 和 lowering 边界，不把某个 wrapper 名、示例 tile shape 或 raw
packet 字段写成上层 IR 语义。

committed static rank instruction program到target LLVM call emission、target CRT wrapper和golden packet的合同见
`tasks/14-target-llvm-golden-packet.md`。

### 3.8 Wafer Communication Stage

主体 IR / Dialect：

- `wafer.tile.*` collective ops
- `wafer.instr.dte_send` / `wafer.instr.dte_recv` / `wafer.instr.dte_wait`

职责：

- 把 tile_region / SPM materialization 之后的 tiled tensor collective lowering 到
  buffer-level `wafer.tile.*` collective。
- 在 candidate p2p schedule lowering 阶段选择 Direct DTE unicast protocol、ring/tree collective、
  sync/effect 边界，并把 p2p body materialize 成 instruction-level `wafer.instr.dte_*` ops；这些 logical
  communication facts和staging demand先由whole-entry SPM/DDR/event planning消费。
- accepted SPM/DDR offsets形成后、atomic commit前，physical transport acceptance再固定 logical peer对应的
  endpoint、channel/FSM、receive buffer exact range、phase和completion。该结果在candidate clone中只是
  stage-accepted fact，只有variant-set cross-rank verifier和global gate通过后才成为committed transport。
- 在 target LLVM前形成launch projection并验证全部component/rank/resource/transport coverage。
  `pinned` projection直接绑定 topology digest；relocatable template只保留 compiler预验证的 control-table
  slots和有限投影选择，不把 route planning交给 runtime。
- 区分 compiler inline Direct DTE path 与 host runtime dyn TLV D2D/P2P path。

V0 主路径：

- fixed-size unicast Direct DTE helper。
- single-card cluster。
- ring all-gather。
- ring reduce-scatter/all-reduce。
- all-to-all 或其它 collective 若由上游合法产出，先保留 logical collective / shard metadata，后续
  `wafer.tile.*` collective 和 `wafer.instr.dte_*` unicast p2p schedule 可组合实现；缺少专用
  lowering 不是上游不支持理由。

Direct DTE/FSM 相关 API：

- `direct_dte_send_async`
- `direct_dte_send_sync`
- `direct_dte_wait_done`
- `direct_fsm_monitor_init`
- `direct_fsm_monitor_receive`
- `direct_sync_post`
- `direct_sync_wait`

边界：

- `DirectDTESendInfo` 只有单个 `dst_addr`、`dst_tile`、`remote_fsm_id`。不能把 raw DTE register 层的 broadcast/shuffle/scatter 字段直接当作当前 helper 的多目的地发射能力。
- single-card collective 默认用 unicast ring/tree，不使用 raw broadcast/shuffle。
- raw DTE non-unicast ABI 需要独立子设计和板端验证后才能进入 V1。

Stream/mailbox 不是 compiler data-plane 主路径。如果 runtime adapter 保留或调用 legacy
Stream/mailbox 兼容路径，则需要把它作为 control/compatibility plane 显式建模，且不得
把它作为 correctness fence。

`wafer.tile.*` communication 的 collective-level op、point-to-point unicast op、token/effect、Direct DTE V0
contract、collective lowering 和 verifier 见
`tasks/13-communication.md`。上游 StableHLO/Shardy collective
只提供 logical collective 语义；endpoint projection、p2p schedule、DTE resource 和 runtime completion
分别在各自 IR 层级 materialize。

### 3.9 Executable Composition and Whole-Variant Atomic Commit Stage

主体 IR / Dialect：

- `wafer.executable`
- `wafer.executable.variant`
- `wafer.executable.resource`
- `wafer.executable.rank` / `wafer.executable.entry`
- `wafer.executable.transport`
- static rank `func.func` + committed `wafer.instr.*`

职责：

- 一个 executable拥有完整 model/program executable set；variant显式携带`ShapeGuardRef`并引用正交的
  `TargetVariantId`和`ProjectionSetId`。shape guard、target artifact compatibility和rank mapping不能压成
  一个自由字符串variant key。
- rank mapping显式保存 `component + dp/tp/pp/ep coordinate -> rank class + entrypoint + resource/transport
  binding`。多个 ranks只有在 local IR、ABI、memory plan和transport template完全等价时才共享class。
- typed resources区分 external IO、immutable weight、persistent state、transient workspace和control
  table；entry ABI slots与 `func.func` signature精确双射。
- shape guard在全distributed program上选择一次；target兼容性先按environment/fingerprint过滤；rank mapping
  是commit后固定事实，不是每个rank独立选择的第三套runtime guard。target/shape/rank-class维度正交。
- 整个 variant clone同时通过完整domain coverage、layout、whole-entry SPM/DDR、transport/event closure、
  instruction/target ABI和cross-rank protocol验证后，才原子替换主 IR。

最小 typed executable 合同：

| 对象 | 必须拥有的字段/关系 | 不拥有 |
| --- | --- | --- |
| `wafer.executable` | program semantic digest、model interface ref、resource/variant symbols、typed target-variant records和accepted projection-set records | runtime session、selected actual shape、provider handle |
| `wafer.executable.resource` | stable `ResourceId`、typed role、dtype/layout/shape bounds/capacity、access、alignment、`DdrArenaId`/placement domain、lifetime scope、alias/update relation；immutable resource含content digest，persistent state的consistency enum只能是`atomic_version`或`in_place_poison_on_failure` | physical address、allocator handle、名字推断role |
| `wafer.executable.variant` | `ExecutableVariantId`、structured `ShapeGuardRef`、`TargetVariantId`、`ProjectionSetId`、完整rank coverage、entry graph、typed completion nodes/edges和terminal policy | rejected candidate、planner trace、per-rank guard |
| `wafer.executable.rank` | canonical component/partition/replica/`dp/tp/pp/ep` coordinate、committed `RankClassId`、entry/resource/transport refs | default rank、文件名约定、重新分片 |
| `wafer.executable.entry` | stable `EntryId`、static function symbol/semantic digest、ordered `SlotId -> ResourceId` bindings、stage/component refs、canonical execution-instance coverage、entry dependency和`CompletionExportId -> completion node` refs | LLVM文本解析结果、自由`binding_order`、内部instruction list |
| `wafer.executable.transport` | accepted transport/projection ref、issue/wait/status/error/completion refs和control resource slots | duplicated p2p algorithm body、runtime route search |

`ShapeGuardRef` 是 typed AST，只能引用已声明 `DimId`、state-capacity或显式invocation-policy fields并使用
有界比较和布尔组合；不能执行任意脚本。completion DAG由variant拥有typed node/edge records，entry/
transport只导出或引用node，因此package不是entry graph或completion语义的首个事实源。

`TargetVariantId`由`wafer.executable`内唯一typed target-requirement record拥有，记录target ABI、required
capabilities和environment compatibility；device-code gate生成的完整module set按该ID发布。
`ProjectionSetId`由`tasks/04-topology-execution-mesh.md`定义的accepted launch projection set拥有并随variant
原子提交。`wafer.executable.variant`只引用这两个owner，不复制target/projection字段。

candidate clone在任何rank-specialized lowering前先建立typed、uncommitted `wafer.executable.rank/entry`
records，逐个引用distributed canonical coordinate、component、execution mesh和static local function；此时
每个rank可保守视为独立provisional class，尚无final `RankClassId`。production passes只从这些SymbolRef取
rank identity。atomic commit验证完整program/ABI/memory/transport等价后才写入final `RankClassId`并提升同一
records；CLI rank option不能参与production identity。

本文把这个clone-local生命周期称为`CandidateExecutionEntry`；它不是新增op或可序列化artifact，而是
uncommitted `wafer.executable.rank/entry` record的状态。target/package pipeline必须拒绝candidate record。
atomic commit保持`ExecutionInstanceId`不变并把同一record提升为`ExecutableEntry`，同时补final
`RankClassId`；失败clone中的identity不能泄漏到cache/manifest。

pre-commit `ExecutableResourceView`是从candidate static entries、frontend resource declarations、accepted
SPM/DDR offsets、arena/placement、transport和projection重算的transformation-local analysis。executable
composer在atomic commit中把它materialize为`wafer.executable.resource`和entry `SlotId -> ResourceId`
bindings；失败则整个candidate丢弃。commit后target只能从这些typed owners和memref/view关系派生具体
address/range/descriptor，PackageManifest只序列化runtime-observable fields，RuntimeSession只实例化typed
bindings；三者都不得重新恢复resource role、scope、alias/update或lifetime。

Executable verifier至少检查：所有ID唯一且引用闭合；resource use-def、slot/signature、access、alias/update、
lifetime和state consistency合法；shape guards priority/fallback确定；每个canonical execution instance恰好覆盖；
entry graph和completion DAG无非法cycle且每个user-visible output可达terminal success；transport/projection引用
完整；最终rank class内local program、typed ABI、final memory plan、transport template和completion exports等价。
distributed rank class只提供必要条件，commit只能继续拆分，不能合并上游已判不等价的instances。

不负责：

- 不保存 planner trace、rejected candidate、duplicated schedule、runtime handle或物理地址。
- 不允许 logical/scheduled group、representative-only tile、pending event或unbound transport进入 committed
  executable。
- 不把 package schema当作 executable事实源。

### 3.10 Target LLVM, Kernel ABI, Package and Runtime Stage

主体 IR / Dialect：

- LLVM dialect
- concrete target CRT call
- compiler-generated Kernel ABI descriptor + ELF module
- Protobuf PackageManifest
- RuntimeSession

职责：

- 从 `wafer.instr.*` 生成 LLVM dialect / LLVM IR 中的 Wafer-owned target CRT call
  declarations/calls；leaf instruction在原SCF/CF/function结构位置转换，unsupported container在mutation
  前失败，不能recursive walk后平铺。
- 在 device-code gate 中用 LLVM clang 和 repo-vendored TX8 deps 生成 RISC-V kcore device `.so`，
  并通过 required-symbol 检查证明 `wafer_tx81_*` 由 repo-local Wafer CRT source/object 或明确合法外部依赖解析。
- 从 `wafer.executable.entry` 和 lowered function生成 ordered typed `KernelAbiDescriptor`；descriptor hash
  同时进入ELF note/export和manifest，package不解析LLVM文本猜ABI。
- 从committed executable和complete TargetArtifactSet派生immutable PackageManifest：environment/artifact
  fingerprints、artifact digest、orthogonal target/shape selection、committed rank graph、typed resources、
  entry graph、endpoint template和completion DAG；不复制instruction
  list。
- RuntimeSession校验manifest/module/target/topology，选择一个coherent variant，绑定module/weight/state/
  workspace/endpoint，执行entry graph并聚合completion/error；不序列化地址，不重新规划。

PackageManifest 不替代 committed executable。前者是不可变交付表示；实际 allocation/module/stream/
event/state handles只属于RuntimeSession。旧package v2只允许由离线converter进入新semantic verifier，
不能让runtime长期维护双合同。

Runtime/package 的 package 内容、Tx runtime provider / KMD / legacy `TsmRun` 分层、runtime allocation/import mapping、
legacy bootparam/TLV 和 completion/stub shielding 合同见
`tasks/15-launch-runtime-package.md`。target CRT 到 wrapper/golden packet 的细节见
`tasks/14-target-llvm-golden-packet.md`。

## 4. Milestone 路线

本节按语义能力描述路线。历史 milestone 代号只作为外部记录索引，不能进入代码、pass、IR、
测试或任务命名；后续实现应按 IR 层、通信能力和验证合同推进。下面的 single-tile、p2p 和 static
transformer 都只是递增验证切片，不能定义长期架构或单独证明 compiler/runtime 完成。长期 gate 必须
最终覆盖 bounded dynamic variants、persistent state、组合并行、多卡 transport 和真实 package/runtime。

### Long-Horizon LLM Completion Boundary

最终系统完成证明至少包含：

- 同一 verified program 形成两个以上 coherent static shape variants，越界 shape 在 launch 前拒绝。
- prefill 与连续 decode共享 immutable weights和persistent paged state，workspace按 invocation隔离；
  state失败遵守 `atomic_version` 或 `in_place_poison_on_failure` 策略。
- TP+DP、PP+DP和EP/MoE distributed program保留component/coordinate/ragged route语义，并原子commit
  全部ranks；单rank失败能聚合并阻止错误state继续使用。
- pinned和relocatable endpoint variant都经过target/topology/transport验证，runtime只选择预编译
  projection/template。
- INT8/FP8或其它mixed-precision variant显式携带quant/storage descriptor和target capability guard。
- 真实model chain经过mandatory atomic commit、target LLVM、actual device link、Kernel ABI/ELF校验、
  Protobuf manifest和no-card/board RuntimeSession；不靠手写fixture或shadow schedule。

### Single-Tile Compute

目标：

- 单 tile 上跑通基础 compute。
- 验证 StableHLO/Linalg -> `wafer.group` -> `wafer.tile.region` / SPM bufferization
  -> `wafer.tile.*` compute -> target CRT 的最小链路。
- 验证 kcore `.so`、SPM allocation、load/compute/store。

范围：

- matmul 或 vector add/reduce 中的一个最小闭环。
- 不做多 tile 通信。

验收标准：

- 至少覆盖 RDMA/WDMA/CT/NE/TDMA 中一个完整 load/compute/store 闭环，并为涉及 wrapper 生成 golden packet。
- 验证 `TsmExecute` 0..4 分派和 `TsmWaitfinish` local drain；不能用 per-op hard wait 掩盖 issue/drain 语义。
- 验证 tx runtime module/kernel path 或 legacy `TsmRun` bootparam path；不能用 `TsmLaunch` / `DeviceSynchronize` 当通过标准。
- Runtime 初始化显式写 `serial_mode=0` 或读回确认。

### Multi-Tile Data Parallel, No Communication

目标：

```text
WaferRuntimeAdapter cluster launch
  -> 多 tile 同时运行同一个 kcore so
  -> 每个 tile 获取自己的 encoded tile endpoint / block id
  -> 每个 tile 处理 input batch slice
  -> 每个 tile 写回 output slice
```

设计重点：

- Endpoint/resource metadata。
- `N` 维切分。
- weights replicated。
- input/output slice metadata。
- block id 和 local slice binding。
- host-side output 拼接或按分片读取。

验收标准：

- package metadata 中的 tile endpoint、block id、local slice metadata 和 available/excluded tile
  信息来自 `wafer.target.topology`、`wafer.execution.mesh`、program parameter shard metadata/resource
  view 和薄 launch/block binding，不维护第二份 endpoint mapping 事实源。
- 不依赖 `TsmGetDeviceNum/List/Properties` 这类 discovery stub 得到 capability。
- completion 来自 tx runtime command/module/stream completion、legacy `TsmRun` synchronous completion，或 kcore 内显式 CSR local drain 加 host runtime completion。

### Direct DTE Point-To-Point

目标：

- 单卡 2 tile 固定包长 SPM-to-SPM。
- 支持 one-way 和 ping-pong。
- 验证 `wafer.tile.*` communication -> Direct DTE helper lowering。

范围：

- `collective_permute` 的最小语义。
- 不混合复杂 compute。
- 只验证 fixed-size unicast Direct DTE helper，不启用 raw broadcast/shuffle/scatter。

验收标准：

- 覆盖 `direct_dte_send_async`、FSM monitor receive、DTE wait/status/error 和 packet counter update。
- DTE resource allocator 管理 high-performance node、normal node、FSM id、packet id、stream id、remote tile 和 release。
- Direct DTE wait 与 local fence / local wait 明确分离。

### Direct DTE Collective Library

目标：

- 单卡 4/8/16 tile collective 原型。
- 先 ring all-gather，再 ring reduce-scatter/all-reduce。

范围：

- `all_gather`
- `reduce_scatter`
- `all_reduce`

验收标准：

- `all_gather`、`reduce_scatter`、`all_reduce` 的每个 step 都能追溯到
  `wafer.instr.dte_send` / `dte_recv` / `dte_wait`。
- DTE buffer、FSM id、packet id、stream id 和 SPM sync slot 没有跨 step 冲突。
- raw DTE non-unicast ABI 只作为 V1/HardwareVerify 入口记录，不进入 single-card collective
  correctness path。

### Partitioned StableHLO Collective Handoff

目标：

- 把 SPMD partition 后的 StableHLO collective 规整成 `wafer.linalg_ext.collective.*`，并在
  group/tiling、tile_region/SPM materialization 后 lowering 到 buffer-level `wafer.tile.*`
  collective，再展开成 instruction-level Direct DTE p2p schedule。
- 对接 group、endpoint projection 和 comm planner。

范围：

- 主路径从真实frontend形成的verified distributed program和static rank candidates开始。
- 手写 partitioned StableHLO 只作为 collective lowering 的局部 verifier / pattern 测试输入；它不能替代
  importer/Shardy 自动导出的 program chain。

验收标准：

- StableHLO collective handoff 后保留 logical mesh、rank group、combiner / slice relation 和 local
  shard shape；tensor collective op 不携带 SPM buffer、DTE token 或 runtime handle。
- 在 tile_region / SPM materialization 后生成的 collective IR 明确区分 compiler inline Direct DTE
  protocol 与 host runtime dyn TLV D2D/P2P path。
- committed executable 保留每个 collective 对应的 accepted transport/resource/event；PackageManifest只
  引用entry、resource和跨entry completion边界，不复制device communication schedule。

### Compute And Communication Mixed Scheduling

目标：

- 支持真实 tensor parallel 子图。
- 将 local compute 和 collective 通信放入同一个 kcore 程序和 runtime package。

验收标准：

- overlap 基于 issue/fence/wait、DTE wait、group barrier、SPM bank/page coloring 和 PMU/profiling，不是简单把 op 放进同一个 kcore function。
- Scheduler 维护 estimated in-flight SPM bank/page/color set、RDMA/WDMA DDR range、DTE resource set 和 NCC queue state。
- PMU case 覆盖 serial mode、parallel mode、64KB page coloring、256B compact layout 的 blocking/exe time 对比。

### Static Transformer Regression Slice

目标：

- 以静态 shape、无 persistent state 的 transformer block 验证 local shard / single-card cluster
  基础能力；它是 regression slice，不是长期完成边界。
- 覆盖 norm、QKV linear matmul、RoPE、attention score、mask/scale、softmax、attention value、
  output linear matmul、residual 和 MLP。

范围：

- regression 可以从单 batch、固定 sequence/head/hidden shape 开始，但不得把这些参数固化到协议。
- 如果 tensor parallel 需要 collective，必须先满足 p2p、single-card collective 和 partitioned
  StableHLO collective handoff 的 communication gate。
- 当前 no-card compile gate 已用 HuggingFace Llama tiny config + PyTorch/XLA `mark_sharding`
  + 单卡 16-rank Megatron-style tensor parallel 覆盖到
  `stablehlo-spmd-to-group` 和 memory-planned instruction IR。该 gate 的 Megatron
  contracting-dimension sharding 保留并消费 `all_reduce` collective，basic `arith.select`
  在 instruction lowering 中改写成 `gather_scatter` + `bit2fp` + `mask_move` target sequence。
  该历史 case 只作为 importer/SPMD/group回归。mandatory commit、target LLVM、Kernel ABI/manifest、
  RuntimeSession、板端launch和数值正确性必须由后续真实chain gate证明。

验收标准：

- Transformer block 的 StableHLO local shard 能 normalized 到 structured tensor IR，不依赖名字识别。
- softmax、RMSNorm/LayerNorm、RoPE 和 MLP activation 都展开成可验证 staged tensor IR。
- group planner 能对 staged reduction/softmax 给出合法 group split 或明确拒绝原因。
- layout/SPM/DDR feasibility 对当前 no-card compile gate 中 accepted groups 通过；constants/weights
  的长期 residency 仍应通过 `ConstantLike` + `wafer.tile.load` / constant storage transform 路径进入
  device storage，并在 board/resource gate 验证。
- 所有启用的 compute/movement/communication ABI family 有 verifier 和必要 golden packet 覆盖。
- Runtime completion 仍使用可信 fence，不使用旧 launch/sync stub。

### Stateful Prefill / Decode Variants

目标：

- 同一 model/program 形成至少一个 prefill variant 和两个不同 sequence/batch guard 的 decode variants。
- immutable weights由session cache复用；persistent paged state在连续invocations间保持identity和update/
  alias；workspace只活到当前invocation结束。

验收标准：

- frontend/distributed program不从 `kv` 名字恢复state；resource role、bounds、page geometry和update relation
  均为typed事实。
- 所有ranks对actual dimensions选择同一个distributed variant；不存在per-rank独立guard。
- no-card gate证明weight/state handle复用和workspace隔离；board numeric gate证明prefill后连续decode读取并
  更新同一state。
- timeout/partial-rank failure按`atomic_version`保留旧版本，或按`in_place_poison_on_failure`持久标记原state
  不可复用；不能出现第三种隐式policy。

### Composite Parallel and MoE Variants

目标：

- 覆盖TP+DP、PP+DP和EP/MoE component graph；PP micro-batch和stage dependency来自entry graph
  template，EP token route/all-to-all-v保留ragged runtime data语义。

验收标准：

- `dp/tp/pp/ep` coordinate、component/stage、partition/replica和rank groups可分别验证；flat rank不承担
  多重语义。
- rank class共享由local IR/ABI/memory/transport等价证明；uneven shard或异构expert自动拆class。
- executable-set cross-rank verifier证明send/recv phase、bytes、buffer和transport assignment匹配。
- RuntimeSession只绑定manifest中的rank/entry/projection，不生成pipeline schedule或collective route。

## 5. 测试和 Verifier 合同

测试和 verifier 跟 IR 分层一样，也按 IR stage 增量建立。不是每个子设计进入实现前都要
把 register-level spec、runtime shielding、PMU/cost-model 全部做完；只要求该子设计引入
的新 IR 语义、lowering contract 或 runtime boundary 有对应验证。

建议按 IR stage 和语义能力取用：

| 范围 | 阶段内验证 |
| --- | --- |
| Verified program | program parse/roundtrip、symbolic bounds、typed IO/parameter/state、alias/update、payload/content identity |
| Target environment / topology / mesh | capability/ABI、topology digest、unavailable endpoints、logical axes/rank domain、pinned/relocatable policy |
| Distributed program | Shardy/SDY/MPMD propagation、component/stage、partition/replica/`dp/tp/pp/ep` coordinate、collective/shard relation |
| Local compute normalization | StableHLO dot/broadcast/reduce/shape/state use-def 到 structured tensor IR，softmax/norm/RoPE staged form |
| Candidate group / tile | group formation、bounded traversal proposal、完整domain coverage、candidate-only tile-region、failure attribution |
| Layout materialization | proposal与accepted assignment分离、physical layout、aligned-only legality、constant/weight storage encoding |
| Whole-entry SPM / DDR | physical range、cross-group lifetime/reuse、persistent/transient demand、capacity、pending event和accepted offsets |
| `wafer.tile.*` compute | 对已支持 op 建 wrapper golden packet，例如 CT unary/binary、NE GEMM、RDMA/WDMA contiguous end-address、DMA stride byte-unit 和 `iteration - 1` |
| Communication / transport | logical collective、Direct DTE send/recv/wait、endpoint/channel/FSM/recv buffer、cross-rank match、completion/error |
| Atomic executable commit | global guard、rank coverage/class equivalence、whole-entry memory/event、all-rank rollback、无logical group/pending event/unbound transport |
| Target/Kernel ABI | structure-preserving conversion、geometry/narrowing、ordered slots、ELF descriptor/hash、module digest/fingerprint |
| Manifest/RuntimeSession | Protobuf semantic verifier、orthogonal target/shape selection、committed rank mapping、typed bindings、weight/state/workspace lifecycle、completion DAG、failure aggregation |
| Scheduler / PMU | 只在进入 overlap/cost-model milestone 后添加：serial/parallel mode、SPM bank/page-color conflict、DDR overlap、PMU `exe_time` / `blocking_time` case |

每个能力阶段的测试面只覆盖该阶段实际启用的 dialect op 和 lowering path。例如 single-tile
compute 只需要单 tile compute 闭环和涉及 op 的 verifier/golden packet；DTE、runtime TLV 和
PMU microbench 不应成为 single-tile compute 前置条件。

## 6. 当前非目标

以下内容不作为 v0 目标：

- 任意 PyTorch 模型无约束 seamless 运行。
- unbounded dynamic shape、运行时JIT低层instruction或任意eager控制流全覆盖；bounded shape guard和
  static variants仍是core合同。
- `all_to_all` 高性能 ring/blocked schedule；logical program 和 direct p2p correctness path 仍应保留。
- 完整 vLLM/SGLang请求调度、prefix cache和调度策略集成；persistent paged state、prefill/decode
  variants及typed runtime ABI仍属于compiler/runtime core合同。
- 自定义 LLVM 后端或真正 ISA intrinsic lowering。
- 依赖旧 Stream/Score data-plane 作为主通信路径。
- 直接以历史 backend 为唯一架构来源。
- raw DTE broadcast/shuffle/scatter 作为 V0 collective 主路径。
- KMD compute fence、旧 `DeviceSynchronize` 或 launch stub 作为 correctness completion。
- Conv optional/fused 特性；V0 只保留基础规则和后续验证入口。
- runtime临时重做sharding、candidate/layout/memory、rank placement或transport route。

## 7. MLIR 工程组织

工程上先采用少量 Wafer-owned op family，而不是一开始拆成很多完全独立 dialect。早期
接口还会快速变化，拆太散会让 type/attr/conversion plumbing 过重。推荐做法是保留核心
`wafer` dialect namespace，同时把 post-SPMD tensor collective handoff 放在
`wafer.linalg_ext.collective.*` op prefix 下，按 op family 和文件组织分层：

```mlir
wafer.group
wafer.linalg_ext.collective.all_reduce
wafer.tile.region
wafer.tile.materialize_layout
memref.alloc : memref<..., #wafer.memory<spm, *>>
wafer.tile.gemm
wafer.instr.dte_send
wafer.instr.local_fence
runtime package metadata
```

建议目录：

```text
include/Wafer/
  Frontend/
  IR/
    WaferBase.td
    WaferDialect.td
    WaferOps.td
    LinalgExt/
      CollectiveOps.td
    Tensor/
      GroupOps.td
    Tile/
      TileRegionOps.td
      LayoutOps.td
      ComputeOps.td
      MoveOps.td
      ViewOps.td
      CommOps.td
    Resource/
      SPMOps.td
    Instr/
      SyncOps.td
  Analysis/
  Transforms/
    Passes.td
  Conversion/

lib/Wafer/
  Frontend/
    ModelImport/
    Program/
    ThirdPartyAdapters/
  IR/
    WaferDialect.cpp
    WaferTypes.cpp
    WaferAttrs.cpp
    Tensor/
    Tile/
    Resource/
    Instr/
    Runtime/
    Common/
  Analysis/
    Group/
  Transforms/
    GroupFormation/
    GroupScheduling/
    TileRegionMaterialization/
    SPMBufferize/
    LayoutMaterialization/
    LaunchOutlining/
  Conversion/
    StableHLOToLinalg/
    WaferGroupToTileRegion/
    WaferInstructionLegalization/
    WaferSPMMemoryPlanning/
    WaferToLLVM/

tools/
  wafer-opt/
  wafer-compile-stablehlo/

cmake/
  third_party/
```

`WaferBase.td` 放共享定义：

- `WaferSemanticLayoutAttr`
- `WaferMemoryAttr`
- `WaferTargetAttr`
- `WaferTileMappingAttr`
- common op interfaces，例如 `WaferTilingInterface`、`WaferLayoutOpInterface`、
  `WaferLayoutMaterializationOpInterface`、`WaferResourceEffectInterface`

`include/Wafer/Transforms/Passes.td` 是 Wafer transform pass API 的声明源。非可选 pass 的 command
line argument、summary、dependent dialects 和 factory declaration 由 TableGen 生成；C++ 实现只定义
pass body 并继承 generated base。可选依赖 pass 可以保留条件编译的手写注册，但不能让普通 pass
继续散落手写 argument / description / dependent dialects。

Region op 的 verifier 按 MLIR 约定拆分：不依赖 region body 的 boundary invariant 放普通
`verify()`；body argument、terminator、nested body legality 和 region-specific semantic attrs 放
`verifyRegions()`。父 region op 只解释自己 body 的直接 op；子 op 内部 region 由子 op 自己的
verifier 负责。

Pass pipeline 建议：

任何新增主线 stage 或重写现有 stage 前，任务文档必须先写清楚 pipeline contract，而不是只描述
某个 pass / tool 的局部功能：

- upstream program / IR。
- current stage responsibility。
- output program / IR。
- downstream consumer。
- user-level driver / named pipeline。
- explicit non-goals。
- completion gate。

pass 名、tool flag、test 名和任务号只作为实现索引；架构边界仍由 IR / program contract 和
verifier/lowering 责任定义。主线 gate 必须通过 `wafer-opt` program pipeline 重放已完成上游链路；
Wafer named MLIR pipeline 只作为内部构件或局部 debug/unit 覆盖。不能依赖 integration test
手动拼 pass、Python helper 或手写测试输入来表示长期 compile flow。

当前用户级 / Integration 调试入口不直接暴露下面这些单 pass。2026-06-02 后由 `WaferPipelines`
注册按 IR 边界命名且真实成立的内部 pipeline；用户级 program pipeline 是
`stablehlo-spmd`、`stablehlo-spmd-to-linalg` 和 `stablehlo-spmd-to-group`。
内部 `wafer-propagate-stablehlo-sharding` 和 `wafer-lower-stablehlo-to-linalg`
只作为 program pipeline 的构件或局部验证入口。
`wafer-propagate-stablehlo-sharding` 组合 Wafer default input seed 和 Shardy propagation，但不冒充
XLA SPMD partitioner；partitioned / replicated-local StableHLO program directory 必须由Wafer-owned
SPMD/MPMD distributed-program stage消费sharding propagation输出的StableHLO/SDY IR后产出。
`wafer-compile-stablehlo` 只保留 frontend / StableHLO program verifier；program-level
入口统一在 `wafer-opt --program-pipeline=stablehlo-spmd`、
`wafer-opt --program-pipeline=stablehlo-spmd-to-linalg` 和
`wafer-opt --program-pipeline=stablehlo-spmd-to-group` 下。旧 C ABI lowering/compile 入口已删除；
旧 single-tile/target CRT/ring/SPM/DDR unit/debug pass 链也已删除。`tx8` 只保留为底层硬件/
依赖事实名，不作为 compiler target。上述入口尚未形成完整production compiler driver；长期必须新增
单一 `stablehlo-to-executable` program pipeline或等价driver。下面列表描述该长期边界，不是要求用户
手动串 pass。

```text
ModelImport/FrontendProgram
  -> verified program + typed resources/state + bounds
  -> target environment/topology + valid execution mesh
  -> Shardy/SDY SPMD/MPMD distributed program
  -> component/rank-local structured tensor program
  -> logical group + bounded candidate frontier
  -> complete variant clones with traversal/layout/instruction/SPM/DDR/transport/event proposals
  -> whole-variant atomic commit
  -> executable set + static variants/rank programs/resources/transport/completion
  -> structure-preserving target LLVM + Kernel ABI descriptors
  -> device modules + ELF descriptors/digests/fingerprint
  -> Protobuf PackageManifest
  -> RuntimeSession materialization / no-card or board execution
```

当前 HF transformer no-card gate 走 `stablehlo-spmd-to-group` 后的 direct group -> instruction 路径，
只证明历史 regression slice；它没有经过 mandatory whole-variant commit，不能作为production主线或
长期完成证明。`DirectFullShape`在新合同中必须变成同一candidate driver内的普通policy并通过全部gate。

工程边界：

第三方工程依赖的 ownership：

| 依赖族 | owner | 可见范围 |
| --- | --- | --- |
| LLVM / MLIR | core compiler build + IR/pass implementation | 全编译器工程，但不作为 Wafer 语义名词 |
| StableHLO / Shardy / SDY | frontend、SPMD、conversion pipeline | frontend 到 local compute normalization |
| model importer dependencies | frontend adapter、import tools | importer tool 和 import 最小验证 |
| runtime / driver headers | launch/runtime adapter、target CRT layer | launch/package、runtime allocation/import、board bring-up |
| test tooling | test harness / CI | tests、golden packet、lit/FileCheck、unit tests |

- LLVM、MLIR、StableHLO、Shardy / SDY、model importer、runtime / driver headers 和测试工具按层
  组织依赖。core compiler deps 可以被 IR / pass implementation 使用；importer-only deps 只在
  frontend adapter 和 importer tools 中出现；runtime / driver deps 只进入 launch / target CRT /
  runtime adapter。
- 第三方依赖版本由 repo-level manifest、lock file、submodule pin 或等价 build 配置固定。具体
  checkout path、CMake target 名、第三方 C++ API 细节不是 Wafer IR contract。
- 所有使用第三方 dialect 的 tool / pass 显式注册 dependent dialects，不能依赖全局 MLIR context
  中“刚好已经加载”的状态。
- `wafer.group.*` ops 使用 tensor types，不接受 SPM memref。
- `wafer.tile.region` 是 `wafer.group` 之后的 bufferized tile-local execution boundary，
  可以包含 memory/movement/compute/comm/sync op，但不重新承载 tensor fusion 决策。
- `select-wafer-compute-impl` 把 tiled `linalg` / tensor compute 绑定到正式的 target-abstract
  `wafer.tile.*` compute / movement op；这些 op 是 layout/materialization 和
  verifier 依赖的 IR contract。
- layout materialization ops 是真实 data movement，负责表达 physical layout conversion，
  不作为 metadata cast。
- instruction legalization / selection 必须发生在 SPM memory planning 之前；target-abstract
  `wafer.tile.*` compute/movement/collective op 只表达 target-abstract semantic，instruction-level
  IR 才表达 concrete storage values、temp/psum/staging、instruction family 和 async lifetime。
- Wafer memory attr 只在 `wafer.tile.region` / SPM bufferization 层出现，不进入
  tensor-level `wafer.group`。
- 没有独立 placed-memref sidecar阶段。committed static rank program通过operands、memref views、accepted
  SPM/DDR offsets、typed resources和transport refs携带不可重算事实；size/range/stride仍从当前IR重算。
- instruction-level compute/communication lowering只消费committed rank program，不再做fusion、rank
  placement、layout/memory或transport选择。
- `async.token` + producing/wait op interface表达device completion；host/stage/rank aggregation由executable
  completion template表达。local fence、DTE wait和multi-rank barrier不能互相替代。
- `wafer.executable`是compiler内唯一committed composition边界；PackageManifest是其不可变派生产物，
  RuntimeSession是非序列化实例。三者不能互相复制schedule或承担对方的规划责任。
- target LLVM只消费committed static rank function、entry ABI、accepted offsets/transport和target
  environment，不回头改schedule/layout/memory。Kernel ABI descriptor由compiler结构化生成而非文本解析。

稳定 op family包括：`wafer.group`、`wafer.linalg_ext.collective.*`、`wafer.tile.*`、`wafer.instr.*`、
`wafer.target.*` / `wafer.execution.mesh`，以及最小 `wafer.executable*` composition ops。新增对象必须
直接服务legality、lowering、diagnostic或删除旧side channel。

## 8. 子设计边界索引

本节是设计文档状态的唯一索引。其它文档只说明自身边界，不维护第二份状态清单。

| 范围 | 主文档 | 状态 | 只负责 | 不负责 |
| --- | --- | --- | --- | --- |
| Frontend / verified program | `tasks/02-frontend-stablehlo-program.md` | 草案 | model semantics、symbolic bounds、typed IO/parameter/persistent state、alias/update、payload identity | rank placement、SPM/DTE、runtime handle |
| Target environment / topology / mesh | `tasks/04-topology-execution-mesh.md` | 草案 | capability/ABI fingerprint、topology snapshot、logical mesh、pinned/relocatable projection policy | sharding、candidate/memory/route search |
| Distributed program | `tasks/03-shardy-spmd.md` | 草案 | Shardy/SDY/MPMD、component/stage、partition/replica/parallel coordinates、collective/shard relation | physical transport、SPM、default rank |
| Local compute normalization | `tasks/05-local-compute-normalization.md` | 草案 | component/rank-local structured tensor IR、state use-def、collective handoff | group/candidate、physical layout、target/runtime |
| Candidate group planning | `tasks/06-group.md` | 草案 | logical group、bounded candidate schedule/direct-full-shape policy、failure attribution | accepted executable、SPM offset、package |
| Candidate tile scope | `tasks/07-tile-region.md` | 草案 | complete traversal内tile-local scope、buffer/effect/movement/compute/sync proposal | 独立commit、host package |
| Layout materialization | `tasks/08-layout-materialization.md` | 草案 | proposal/accepted assignment边界、physical layout、weight storage encoding/materialization | runtime placement、packet field |
| SPM memory planning | `tasks/09-spm-memory-planning.md` | 草案 | whole-rank-entry liveness/event、range/alignment、accepted offsets | DDR、runtime handle、collective algorithm |
| Compute / movement | `tasks/10-compute-movement.md` | 草案 | target-abstract ops、layout/resource/effect、issue/token/fence/wait、shared legality | fusion、sharding、package schema |
| Static rank instruction program | `tasks/11-instruction-ir.md` | 草案 | complete `wafer.instr.*` rank function、geometry/range/narrowing、event closure | runtime mapping、raw packet、shadow schedule |
| DDR memory planning | `tasks/12-ddr-memory-planning.md` | 草案 | whole-entry external/persistent/transient demand、cross-group lifetime、accepted offsets/capacity | runtime handles、packet bits |
| Communication / transport | `tasks/13-communication.md` | 草案 | logical collective到accepted endpoint/channel/FSM/buffer/token/error transport | tensor sharding、runtime route planning |
| Executable composition | 本文 + `tasks/06-group.md` | 草案 | global guards、rank classes/entries、typed resources/transport/completion、whole-variant atomic commit | rejected trace、runtime handle、schedule duplication |
| target LLVM / Kernel ABI | `tasks/14-target-llvm-golden-packet.md` | 实现中 | structure-preserving conversion、Kernel ABI descriptor、ELF/module fingerprint、CRT/golden | candidate search、package text inference |
| Manifest / RuntimeSession | `tasks/15-launch-runtime-package.md` | 草案 | Protobuf manifest、typed bindings/state/workspace、entry/completion graph、session materialization | compiler replanning、instruction schedule |
| Verification plan | `tasks/16-verification-plan.md` | 草案 | stage diagnostics、roundtrip、golden packet、runtime shielding、PMU/cost-model gate | 替代各 dialect 语义设计 |
| Serving product integration | 外部后续设计 | 延后 | request scheduling、prefix cache、admission/batching policy | core persistent-state/variant/resource ABI |

跨文档判断规则：如果某个事实不能由当前 IR 层的 op/type/region/effect/verifier 稳定解释，它只能
作为 analysis/cost input 引用，不能提前写成该层 IR 语义。

Transformer block 落地时的文档阅读顺序是：

```text
Frontend program
  -> Target environment / topology / execution mesh
  -> Shardy / SPMD / MPMD distributed program
  -> Local compute normalization + `wafer.linalg_ext.collective.*` handoff
  -> logical group / candidate tile region
  -> Layout / Instruction / SPM / DDR / Transport proposals
  -> Whole-variant atomic commit / executable set
  -> target LLVM / Kernel ABI / ELF module
  -> PackageManifest / RuntimeSession
  -> Verification plan
```

完整 serving request scheduler不属于core compiler；bounded prefill/decode variants、persistent paged state、
typed resource ABI和RuntimeSession lifecycle属于core compiler/runtime合同。

## 9. 参考材料

本设计基于当前仓库中的以下材料整理。优先级以整理后的 Wafer 主文档和
reverse-engineering 合同为准，旧实现和历史材料只作为这些文档中的证据来源，不作为当前
架构主线或 IR 命名来源：

- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/README.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`
