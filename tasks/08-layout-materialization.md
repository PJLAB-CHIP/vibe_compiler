# Wafer Layout Materialization Design

状态：本轮长期边界合同已收敛；实现状态以`tasks/progress.md`为准。范围：memref-backed Wafer memory attr、candidate planning 和 instruction-level pipeline。

本文定义 Wafer 后端的 physical layout planning 和 layout materialization 边界。它服务于
`wafer.group` candidate/template 的 legality search，也服务于完整 static rank variant set 中
`wafer.tile.region` / SPM bufferization 的真实 lowering。

本文只负责：

- 从 target-abstract `wafer.tile.*` compute / movement / communication op 的 layout interface 和
  boundary contract 推导 physical layout assignment。
- 在 whole-variant candidate clone 上插入 `wafer.tile.materialize_layout` 或选择 compile-time constant
  storage transform，并由 committed memref types 与 explicit movement 唯一拥有最终 layout 真值。
- 做跨完整 static rank entry 的 bounded boundary co-planning 和 materialization cleanup。

本文不负责 group formation、tile shape search、SPM offset allocation、DDR memory planning、
compute/comm op 语义、launch/runtime package 或 raw instruction packet。layout planner 的中间
constraint graph、early layout proposals、layout alternatives、cost breakdown 和失败原因都是 analysis；
只有 whole-variant gates 接受后 committed memref type、explicit movement 和可 lower 的 constant storage
选择进入主 IR。early group layout 不是第二份最终 layout 真值。

## 1. 核心结论

layout materialization 是从硬约束出发的局部 dataflow planning：

```text
candidate/template scheduled tile tensor IR
  -> pass-local LayoutVariable / LayoutEdge analysis
  -> early group-local layout proposal
  -> full-entry boundary/lifetime/resource co-planning in a whole-variant clone
  -> final tile-local memref IR with Wafer physical layout marker
  -> materialization cleanup / canonicalization
  -> runtime/target-codegen address derivation from accepted facts
  -> concrete movement / compute emission
```

`ChannelNorm` / `DechannelNorm` / `GatherScatter` 是真实 data movement，不是 metadata reshape。
它们会增加 buffer、liveness、movement latency、SPM pressure 和可能的 DDR load/store pressure，
所以只能在 layout domain 冲突、且 SPM / DDR feasibility 接受时插入。

## 2. 借鉴点

只吸收这些已有系统的基本方法，不照搬实现：

- MLIR Bufferization：先 analysis，再 rewrite；通过 op interface 查询局部语义。
- TVM ConvertLayout：从 layout-sensitive op 发起 conversion，而不是全图盲插。
- XLA BufferAssignment / TFLM arena planner：layout 选择必须和 buffer size、lifetime、reuse 一起验证。

对应到 Wafer：

- layout planner 是 analysis + rewrite，不是 IR attr plan。
- layout-sensitive op 通过 interface 给出 hard domain 和 preference。
- layout assignment 必须调用 SPM allocation 和 DDR memory planning；只算 layout conversion 次数不够。

## 3. 表示生命周期

layout materialization 文档内部只使用一条表示生命周期，避免把 analysis、accepted IR 和
lowering IR 写成三套互不相干的东西：

| 层次 | 表示对象 | 是否进入 IR | 责任 |
| --- | --- | --- | --- |
| scheduled candidate/template IR | candidate clone 中 `wafer.group` scheduled body 的 tensor SSA value | 是，仅候选 | 表达 tile-local tensor dataflow、完整 traversal、boundary slice；不表达最终 physical layout |
| target-abstract tile-region IR | `wafer.tile.region` 中的 `wafer.tile.*` compute / boundary / data movement op | 是 | 承载选中的 Wafer implementation 和 `WaferLayoutOpInterface`，但尚未绑定 concrete instruction/runtime descriptor |
| layout analysis | `LayoutVariable` / `LayoutEdge` / layout assignment alternatives | 否 | 从 SSA use-def、op interface 和 boundary contract 推导 layout domain、preference、materialization cut |
| early boundary proposal analysis | group-local/相邻 boundary 的可行 layout summary 和 proposal | 否 | 约束候选搜索；不是最终 owner，不得跨 pass 作为已接受事实消费 |
| whole-entry layout finalization | complete static rank variant clone 上的 layout/resource/lifetime co-planning | analysis 只改 candidate clone；atomic commit 后才进入主 IR | 同时验证所有 group boundary、SPM/DDR footprint、movement 和完整 traversal lifetime |
| committed tile-local memref IR | `memref<shape x dtype, #wafer.memory<space, layout>>` 和 `wafer.tile.materialize_layout` | 是 | 最终 layout 的单一 owner；记录已接受的 address space/physical layout marker 和真实 movement edge |
| materialization cleanup | canonicalization pattern 和 layout-aware rewrite | 是，通过 rewrite 当前 IR | 删除冗余 materialization；不保存搜索过程 |
| runtime/target-codegen address derivation | target call/codegen 参数或 lower-level emission metadata | 是，仅作为 very-late derived form | 从 committed Wafer-tagged memref、accepted offset facts、view relation 和 layout helper 派生目标指令需要的 address/range/stride representation；不形成新的主线 IR 事实源 |
| lowered movement IR | `ChannelNorm` / `DechannelNorm` / `GatherScatter` / TDMA / lower-level effects | 是 | 把 abstract materialization 展开为目标相关 movement、sync 和 byte/range 约束 |

constant storage transform 与这条主线并行：它只处理 compile-time `ConstantLike` value 的
backing data/resource 如何按目标 layout 存放，并由 `wafer.tile.load` / storage lowering 直接消费。
它不引入新的 tensor constant 语义，也不把 package format 当成 layout IR 边界。

### 3.1 输入边界

layout planning 有两个恢复层次：

- R3.2b 在 logical `wafer.group` 层运行。它消费 R3.2a `GroupTilingDemand` facts 和当前
  group SSA use-def，构造 transformation-local `LayoutVariable` / `LayoutEdge` graph、op
  layout constraints、layout assignment alternatives、materialization cut 和 materialization
  buffer demand。该层只产出 early proposal analysis result 和 debug dump，不 rewrite `wafer.group`，
  不写 layout attr，也不生成 `wafer.tile.region`；其 proposal 在任何 candidate rewrite 后都可失效、可重算。
- committed complete static rank program 中的 `wafer.tile.region` / instruction-level IR 已经包含
  whole-variant gates 接受的
  Wafer-tagged memref value 和 `wafer.tile.materialize_layout` op。后续 topology/execution-mesh、
  target LLVM call emission 和 package metadata 只从这些 IR facts 派生 lower-level 参数，
  不再重新 materialize layout assignment 或 materialization cut。

完整 layout materialization 的输入来自 whole-variant candidate clone 中的 target-abstract tile-region IR。
它由 candidate/template scheduled `wafer.group` 的完整 traversal lowering 而来，但 layout-sensitive tiled op 已经被绑定为正式的
`wafer.tile.*` compute / boundary / data movement op。layout materialization pass 不从裸
`linalg.*` 名字推断硬件 layout，而是通过这些 Wafer op 的 `WaferLayoutOpInterface`
查询 hard constraint 和 preference。

输入信息只有这些：

- scheduled body 中的 SSA use-def 和 `scf` loop/control-flow。
- 每个 tile-local tensor value 的 rank、shape、dtype。
- `tensor.extract_slice` / `tensor.insert_slice` 表达的 boundary slice。
- target-abstract `wafer.tile.*` compute / boundary / data movement op 通过 layout interface 暴露的
  operand/result layout constraint。
- group planner 在当前 transformation 内部提供的 tile shape、operand demand、temporary/workspace/
  accumulator demand。

这些信息不作为 `wafer.group` attribute 保存；layout planner 直接从 scheduled body 和当前
transformation-local analysis 读取。R3.2b 的 logical-group implementation 在 target-abstract
compute op 尚未 materialize 前，只能使用 `GroupTilingDemand`、structured op semantics、tensor
collective interface 和 target policy 构造保守 layout constraints；不能把单个 workload 或 op
名字序列写成长期协议。

#### 3.1.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verifier-legal tensor-level logical `wafer.group` op，以及 `GroupTilingDemand` analysis result。
- Current stage responsibility:
  从 `GroupTilingDemand` facts、group body SSA use-def、DPS ties、structured op
  semantics 和 tensor collective facts 构造 transformation-local layout planning graph；
  为 tile values / uses / boundary 生成 layout constraints、layout assignment alternatives、
  materialization cuts 和 materialization buffer demand。
- Output artifact / IR:
  transformation-local early `GroupLayoutPlan` proposal；debug dump pass 可以打印同一结构。
  本阶段不修改 `wafer.group`，不生成 `wafer.tile.region`，不写 layout attr。
- Downstream consumer:
  group-to-tile-region lowering、Wafer instruction legalization / selection、
  candidate DDR tile-view materialization、SPM memory planning、
  DDR memory planning + compute/movement legality analysis，
  以及 closed-loop candidate driver。
- User-level driver / named pipeline:
  production 主线由 `wafer-opt --program-pipeline=stablehlo-to-executable` 或等价 driver 调用 early
  layout analysis；`stablehlo-spmd-to-group` 和 `--wafer-dump-group-layout-plan` 只用于 stage replay / debug。
- Explicit non-goals:
  不选择最终 closed-loop tile shape、不 select/reject/split group、不分配 SPM、不判断
  DDR view/range/resource、不 materialize compute/movement/comm op、不生成 package/ABI；不把
  `GroupLayoutPlan`、boundary summary 或 group-local passing result 当作最终 layout owner。
- Completion gate:
  FileCheck 覆盖 linalg matmul/broadcast/elementwise、multi-group、tensor collective、
  materialization demand 和 unsupported tiling failure；program pipeline gate 能在真实
  `stablehlo-spmd-to-group` 输出上重放 layout-plan dump。该 gate 只证明 early proposal 可重算，
  不证明 executable layout 已完成；最终 completion 见下节 whole-variant finalization contract。
```

R3.2b 已落地：按上述 logical-group analysis 边界完成。当前
`GroupLayoutPlan` 消费 R3.2a `GroupTilingDemand` facts 和 group SSA use-def，输出
boundary layout、per-op layout constraints/assignment、materialization cut、group-result
materialization demand 和 failure forwarding；它不 rewrite `wafer.group`，不写 layout attr，
不生成 `wafer.tile.region`。completion gate 覆盖手写 group 测试输入和真实
`stablehlo-spmd-to-group` program 输出。

#### 3.1.2 Whole-Variant Layout Finalization Contract

```text
Pipeline position:
- Upstream artifact / IR:
  已物化所有 static rank entries、所有 candidate groups 完整 traversal 的 whole-variant clone；
  clone 中包含 target-abstract tile-region/instruction IR、early layout proposals 和真实 memref use-def。
- Current stage responsibility:
  从当前 clone 重算 layout graph，在完整 rank-entry lifetime、所有 group boundary、SPM/DDR demand、
  movement、constant storage 和 target instruction constraints 下选择唯一最终 assignment；把 assignment
  写成 memref type 与显式 materialization/movement，并运行 cleanup 后重新验证资源和 lifetime。
- Output artifact / IR:
  只存在于 passing clone 中的 final Wafer-tagged memref IR；whole-variant atomic commit 后它成为
  committed layout 的单一 owner。失败时只返回结构化原因，不更新主 IR。
- Downstream consumer:
  whole-entry SPM/DDR planning、complete instruction/event/transport verification、launch projection、
  whole-variant atomic commit；commit 后才由 target lowering 和 package derivation 消费。
- User-level driver / named pipeline:
  `stablehlo-to-executable` production driver 内的 whole-variant candidate-selection/commit pipeline；局部
  layout pass/dump 只用于实现验证，不能单独提交。
- Explicit non-goals:
  不保存 early proposal/constraint graph，不按 group 分批提交，不让 package/runtime 重选 layout，
  不用 representative tile/rank 或 `DirectFullShape` 特判绕过完整 traversal/resource gates；不因
  distributed rank equivalence 相同就提前共享 final layout 或创建 executable rank class。
- Completion gate:
  每个 static rank entry 的所有 tile-local value 和跨 group boundary 都有唯一可 lower layout；
  full traversal 下的 materialization lifetime 已进入 SPM/DDR plan；shared physical geometry/range/narrowing
  verifier 证明 physical footprint、descriptor range 和 ABI width；任一 rank/group 失败则整个 clone
  不提交。layout 只为 whole-variant commit 决定最终 executable rank class 提供验证事实，不拥有该 class。
```

### 3.2 从 Tensor Value 到 Buffer Value

layout planner 先把 target-abstract tile-region 中的 tiled SSA value 映射成 layout variable，
接受 assignment 后再生成带 physical layout 的 tile-local storage。映射关系来自 SSA use-def，
不是名字约定：

| target-abstract tile-region value | layout planning 中的角色 | accepted `wafer.tile.region` 表达 |
| --- | --- | --- |
| group block argument / external input slice | `#wafer.memory<ddr, tensor>` compact external boundary + load demand | `wafer.tile.load` 产生 Wafer-tagged memref |
| tiled op operand/result | `LayoutVariable` + op layout constraints | `memref<..., #wafer.memory<space, layout>>` |
| producer-consumer edge | `LayoutEdge`，可能允许 materialization | same-layout use 或 `wafer.tile.materialize_layout` |
| group output slice | writeback boundary | `wafer.tile.store` 或后续 writeback movement |
| loop-carried accumulator / partial result | loop-carried layout variable | entry/yield layout 一致，或 loop 内显式 materialization |

因此 layout 文档中的 Wafer-tagged memref 不是另一套上游 IR；它是 target-abstract tile-region
中的 tiled SSA value 在 accepted layout 层的 bufferized 表达。

### 3.3 贯穿本文的 Case Fragment

以下片段来自 group 文档中的 two-output case，只保留从 scheduled group 到 target-abstract
tile-region 的局部结构。第一段是 group scheduling 后的 tensor-level tiled IR：

```mlir
%a_tile = tensor.extract_slice %ga[%m0, 0] [64, 256] [1, 1]
          : tensor<128x256xf16> to tensor<64x256xf16>
%b_tile = tensor.extract_slice %gb[0, %n0] [256, 64] [1, 1]
          : tensor<256x128xf16> to tensor<256x64xf16>
%matmul_tile = linalg.matmul
               ins(%a_tile, %b_tile
                   : tensor<64x256xf16>, tensor<256x64xf16>)
               outs(%tile_zero : tensor<64x64xf16>)
               -> tensor<64x64xf16>
%relu = ...
%r_partial = ...
```

layout materialization 前，layout-sensitive op 已经被选成正式的 target-abstract Wafer op：

```mlir
%a_tile2 = tensor.extract_slice %ga[%m0, 0] [64, 256] [1, 1]
           : tensor<128x256xf16> to tensor<64x256xf16>
%b_tile2 = tensor.extract_slice %gb[0, %n0] [256, 64] [1, 1]
           : tensor<256x128xf16> to tensor<256x64xf16>
%mm = wafer.tile.gemm %a_tile2, %b_tile2
    : (tensor<64x256xf16>, tensor<256x64xf16>) -> tensor<64x64xf16>
%relu = wafer.tile.elementwise %mm
    : tensor<64x64xf16> -> tensor<64x64xf16>
%r_partial = wafer.tile.reduce %relu
    : tensor<64x64xf16> -> tensor<64xf32>
```

layout planning 对 target-abstract Wafer IR 的理解是：

- `%a_tile` / `%b_tile` 来自 group external inputs，默认 external layout 是 compact。
- `wafer.tile.gemm` 通过 layout interface 要求参与 NE/GEMM 的
  operands/results 使用 aligned family；这里 2D tile 对应 `Cx` family。
- `%relu` 如果 lower 到 flexible CT / elementwise，可以接受 producer 的 selected layout，避免
  多余 conversion。
- `%r_partial` 的 reduce 若走 native reduce，input/output 也要满足 aligned family；若走其它
  implementation，则由该 implementation 的 interface 给出自己的 allowed layouts。
- `%r_tile` 这类 loop-carried partial result 必须在 loop entry/yield 上保持一致 physical layout，
  除非 loop body 内有明确 materialization。

这一层的 analysis 可以抽象成下面的 constraint graph。注意这不是 IR，也不是 attr plan：

```text
Var(%a_tile): #wafer.memory<ddr, tensor> compact external boundary, materializable to aligned for consumer
Var(%b_tile): #wafer.memory<ddr, tensor> compact external boundary, materializable to aligned for consumer
Var(%matmul_tile): constrained by selected matmul implementation
Var(%relu): flexible, preferably inherit producer layout
Var(%r_partial): constrained by selected reduce implementation and loop-carried tie

Edge(%a_tile -> matmul operand 0): materializable
Edge(%b_tile -> matmul operand 1): materializable
Edge(%matmul_tile -> relu operand 0): same-layout preferred
Edge(%relu -> reduce operand 0): materializable only if selected reduce requires different layout
```

被接受后，同一个 value graph 在 `wafer.tile.region` 层可以长成这样：

```mlir
wafer.tile.region {
  %a_t = wafer.tile.load %ga_tile
      : memref<64x256xf16, #wafer.memory<ddr, tensor>>
     -> memref<64x256xf16, #wafer.memory<spm, tensor>>
  %b_t = wafer.tile.load %gb_tile
      : memref<256x64xf16, #wafer.memory<ddr, tensor>>
     -> memref<256x64xf16, #wafer.memory<spm, tensor>>

  %a_cx = wafer.tile.materialize_layout %a_t
      : memref<64x256xf16, #wafer.memory<spm, tensor>>
     -> memref<64x256xf16, #wafer.memory<spm, cx>>
  %b_cx = wafer.tile.materialize_layout %b_t
      : memref<256x64xf16, #wafer.memory<spm, tensor>>
     -> memref<256x64xf16, #wafer.memory<spm, cx>>

  %mm = wafer.tile.gemm %a_cx, %b_cx
      : (memref<64x256xf16, #wafer.memory<spm, cx>>,
         memref<256x64xf16, #wafer.memory<spm, cx>>)
     -> memref<64x64xf16, #wafer.memory<spm, cx>>

  %relu = wafer.tile.elementwise %mm
      : memref<64x64xf16, #wafer.memory<spm, cx>>
     -> memref<64x64xf16, #wafer.memory<spm, cx>>

  %r_partial = wafer.tile.reduce %relu
      : memref<64x64xf16, #wafer.memory<spm, cx>>
     -> memref<64xf32, #wafer.memory<spm, tensor>>

  %relu_tensor = wafer.tile.materialize_layout %relu
      : memref<64x64xf16, #wafer.memory<spm, cx>>
     -> memref<64x64xf16, #wafer.memory<spm, tensor>>

  wafer.tile.store %relu_tensor, %go0_tile
      : memref<64x64xf16, #wafer.memory<spm, tensor>>
     -> memref<64x64xf16, #wafer.memory<ddr, tensor>>
  wafer.tile.store %r_partial, %go1_tile
      : memref<64xf32, #wafer.memory<spm, tensor>>
     -> memref<64xf32, #wafer.memory<ddr, tensor>>
}
```

这只是 layout 层的示例形态：具体 compute op、load/store op 名和最终 movement lowering 属于
`wafer.tile.region` / compute / memory dialect 设计。本文固定的是 value mapping、layout
assignment、materialization edge 和 verifier 责任。示例里两个 output 的 shape 不同，layout
planner 只按各自 SSA value、boundary contract 和 consumer/producer constraint 处理；不要求它们
共享一个 root domain，也不编造二者之间的 shape relation。示例里的 tile shape、op 名和某个
implementation 选择都只是为了说明 IR 如何流动，不是 layout 架构边界。示例里的
`%ga_tile` / `%gb_tile` / `%go*_tile` 是已经由 group/tile boundary lowering 切分出的 DDR
tile memref，不表示 layout planner 自己负责 global tensor slicing。

## 4. Layout 分类

layout 和 storage encoding 概念必须分开：

- semantic layout：tensor 维度业务含义，例如 NHWC、HWOI/HWIO、GEMM matrix。
- physical layout marker：device buffer 的实际组织，例如 compact `Tensor/NTensor`、
  aligned `Cx/NCx`。
- constant storage encoding：compile-time constant 的 backing data/resource 在 storage lowering
  或最终 package 中的字节排布。它可以保持原始 compact 排布，也可以由显式 constant storage
  transform pass 生成目标相关排布。
- external layout contract：host/runtime ABI 上 visible 的 tensor layout。

V0 规则：

- host-visible dynamic input/output 默认 `#wafer.memory<ddr, tensor>` compact external tensor layout。
- compile-time constants 可以通过显式 constant storage transform 直接改写 backing data/resource
  或替换为等价 storage-level constant；这不是隐式 layout fact，也不是 weight 专属规则。
- device-side intermediate 可以选择 `Tensor/NTensor` 或 `Cx/NCx` physical layout family。
- `Tensor_Fmt` 不能作为 compiler layout 模型。

## 5. Op Layout Contract

每个 layout-sensitive op 提供一个最小 interface：

```text
getAllowedLayouts(operand/result, tileShape, target)
getPreferredLayouts(operand/result, tileShape, target)
verifyLayoutCombination(operands, results)
getMaterializationCost(srcLayout, dstLayout, shape, dtype)
```

V0 只需要三类 op：

1. aligned-only op

   NE、Reduce、Pool、UnPool 这类 op 的 input/output 默认要求 aligned physical layout。2D 通常是
   `Cx`，rank 大于 2 通常是 `NCx`。GEMM 按矩阵最后一维做 Cx-style align。

2. flexible op

   CT / DataMove / DMA / 部分 elementwise 可以接受 compact `Tensor/NTensor` 或 aligned `Cx/NCx`，
   但仍受 dtype、stride、bool bitpack、range-end 和 wrapper 限制。

3. materialization op

   layout conversion 先用统一的 `wafer.tile.materialize_layout` 表示，再 lower 到一条或多条
   `GatherScatter`、TDMA 或 wrapper path。same-layout materialization 必须 canonicalize 掉。
   对 `C > block` 且保留 `C0` tail 的 `Cx/NCx` 转换，full C blocks 和 compact tail block 的
   inner width / stride 不同，通常需要至少两段 GatherScatter：full-block 段和 tail-C0 段。

compute/movement op 的具体 interface、op family 和 issue/fence/wait 边界见
`tasks/10-compute-movement.md`。communication op 不改变 tensor semantic
layout，p2p transfer 默认是 byte-preserving；若 collective/p2p schedule 消费或产生 storage，
它必须按 `tasks/13-communication.md` 暴露 buffer、layout relation、
token/effect 和 staging demand。layout planner 只消费这些接口事实，不复制 compute/comm 的
lowering 计划。

### 5.1 V0 IR Contract

本节只定义第 3 节生命周期中的 accepted tile-local storage IR 和 constant storage transform 合同。
`LayoutVariable` / `LayoutEdge` 不属于本节 IR；lowered `ChannelNorm` / `GatherScatter` / TDMA
也不属于本节 IR。语法是设计草图，不要求现在就完全等同 ODS/parser 打印格式；真正要固定的是
语义归属、verifier 责任和 lowering 边界。

#### 5.1.1 Attributes / Types

V0 使用 MLIR `memref` 作为 buffer value。Wafer 的 address space 和 physical layout marker
统一放在 memref memory-space attr 中：

```mlir
memref<64x64xf16, #wafer.memory<spm, tensor>>
memref<64x64xf16, #wafer.memory<spm, cx>>
memref<64x64xf16, #wafer.memory<ddr, tensor>>
```

`#wafer.memory<space, layout>` 的 `space` 至少包含 `spm`、`ddr`；`layout` 至少包含
`tensor`、`ntensor`、`cx`、`ncx`。它只能出现在 `wafer.tile.region` 及其下游 buffer /
instruction IR 上，不进入 `wafer.group`。

语义约定：

- `spm` 是 tile-local SRAM，容量、lifetime、range 和 reuse 由 SPM bufferization 负责。
- `ddr` 是 device/global DDR 的目标侧 address space；external view/descriptor validation、
  compiler-managed DDR accepted offset facts 和 declared arena/placement-domain constraints 由 DDR memory planning
  负责，package/runtime allocation mapping 由 launch/runtime/package 层负责。
- `layout` 是 Wafer physical layout marker。DDR buffer 可以是 compact，也可以在 constant storage
  transform 或 device-side materialization 后带目标相关 marker；SPM buffer 同样通过这个 marker
  表达 compact 或 aligned family。
- compile-time constant 在 semantic tensor IR 中仍是 `ConstantLike` value。只有在
  `wafer.tile.load` / lowering 需要 addressable device storage 时，才产生明确 memref、derived
  address/range 参数或 runtime/package metadata。

DDR planned ranges、runtime allocation mapping、resident constant、capacity 和 bandwidth 不属于
`#wafer.memory<space, layout>`，见
`tasks/12-ddr-memory-planning.md`。

V0 不把以下派生结果写进 `#wafer.memory<space, layout>`：

- aligned axis。
- C block size。
- `Cx` / `C0`。
- `aligned_C`。
- batch memory size。
- storage bytes。
- bank/page/color。

原因是这些信息都能从当前 IR 和 target policy 推出，写进 attr 会形成重复事实源。统一
calculator / verifier 负责从以下输入计算 layout storage facts：

```text
WaferPhysicalTensorInfo {
  memory_space        // spm / ddr
  layout_marker       // Tensor / NTensor / Cx / NCx
  semantic_layout
  logical_shape
  dtype
  target_policy

  // derived, not stored in IR
  block_size
  aligned_c
  cx
  c0
  batch_mem_elems
  storage_bytes
  bank_padding
}
```

`Cx/NCx` 的 C alignment 总是针对 logical last dimension，不是 attr 中任选的 axis。`Cx` 用于
2D operand，`NCx` 用于 rank > 2 operand；`NCx` 名字里的 `N` 只是历史命名的外层 slice，
不等价于 semantic batch。Conv weight 的 `HWOI/HWIO` 这类 rank > 2 operand 也可以使用
`NCx` physical layout family。full-block 物理顺序是 channel-block major：`Cx` 为
`[CxBlock][outer][lane]`，`NCx` 为 `[N][CxBlock][HW][lane]`。`aligned_C` 只参与
footprint / batch size 计算，不能当成 logical outer/HW row 的 dense stride。

```text
full block, c = cb * B + lane:
  Cx  offset = cb * outer * B + outer_idx * B + lane
  NCx offset = n * batch_mem_elems + cb * hw * B + hw_idx * B + lane
```

`Cx/C0/aligned_C/batch_mem_elems/storage_bytes` 按硬件文档中的 `get_CxC0` /
`common_tensor_info_generate_i64` 口径计算：INT8/UINT8 full block 是 128，其它 dtype full
block 是 64；tail 是否保留由半块阈值决定；随后继续计入 256B bank padding。对于 `C > block`
且保留 `C0` tail 的 layout conversion，lowering 必须把 full-block span 和 tail span 分开建模；
如果它们不能分别映射到 V0 GatherScatter 的三层 stride/iteration descriptor，就结构化失败。

`Cx/NCx` 不放进 memref layout slot，也不实现为 `MemRefLayoutAttrInterface`。普通 compact
tensor、static subview 和 affine/strided view 可以继续使用标准 memref layout 机制；Wafer
`Cx/NCx` 是依赖 dtype、tail/fold 和 256B padding 的 target storage rule，不能伪装成单个
affine map。所有 pass 通过统一 helper 解析：

```text
computeWaferPhysicalTensorInfo(memrefType)
```

`computeWaferPhysicalTensorInfo` 返回 address space、layout marker、bool bitpack、block size、
`Cx/C0`、`aligned_C`、footprint、range-end、descriptor stride 和 legality；
`computeWaferPhysicalElementByteOffset` 返回 logical index 到 physical byte offset 的映射。禁止每个
pass 自己根据字符串或局部约定解释 `cx/ncx`。

Wafer-tagged memref 在 runtime/target-codegen 派生之前仍是 logical-shape memref：shape 是 logical
shape，element type 是 logical dtype；它不能被 generic memref-to-LLVM lowering 当成
`product(shape) * elemBytes` 的真实 footprint。runtime/target-codegen 负责从 accepted offset facts、
memref view relation 和 layout helper 派生目标指令需要的 address/range/stride/layout 参数；
这些参数不作为新的主线 IR 事实源。

V0 不定义额外的 constant storage encoding attr。constant 的 storage encoding 不作为独立
layout attr 在 pipeline 中传播；它由当前 `ConstantLike` value、consumer `wafer.tile.load`
result layout、storage lowering 规则和最终 package emission 共同决定。若后续确实需要把
encoding 固化进 IR，也应落在 storage-level constant/load op 的类型或 verifier 可检查的
attribute 上。

#### 5.1.2 `wafer.tile.materialize_layout`

`wafer.tile.materialize_layout` 是 device-side real data movement 的抽象 op。它只表示从一个
physical layout 生成另一个 physical layout；不改变 tensor 数学语义。

```mlir
%dst = wafer.tile.materialize_layout %src
    : memref<64x64xf16, #wafer.memory<spm, tensor>>
   -> memref<64x64xf16, #wafer.memory<spm, cx>>
```

ODS 草图：

```tablegen
def Wafer_LayoutMaterializeOp : Wafer_Op<"tile.materialize_layout", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<WaferLayoutMaterializationOpInterface>
  ]> {
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);
}
```

Verifier 至少检查：

- source/result rank、shape、element type 的 semantic extent 一致。
- source/result memory space 合法；V0 只允许 tile-local device buffer。
- source/result `#wafer.memory<space, layout>` 的 layout marker 不同；相同 layout 的 materialize
  必须 canonicalize 掉。
- 对应 conversion path 存在，且由 shape/dtype/semantic layout/target policy 推出的
  `WaferPhysicalTensorInfo` 合法。
- op effects 正确表达 read source / write result；若 lowering 变成 async movement，后续 sync
  verifier 必须能看到 fence/wait。

`wafer.tile.materialize_layout` 不保存 cost、失败原因、planner 选择理由或备选方案。debug dump 可以
打印这些信息，但不能让下游依赖。

#### 5.1.3 Constant Value and Storage Transform

Wafer 不定义新的 tensor constant op。constant 的数学值复用已有 MLIR / upstream dialect：

- StableHLO 输入阶段可以保留 `stablehlo.constant`。
- 进入 Linalg / Wafer planning 前必须做 constant normalization，把 `stablehlo.constant` 转成
  后续 pipeline 可统一处理的 `arith.constant` 或其它实现 `ConstantLike` 语义的 tensor op。
- 文档中的通用示例使用 `arith.constant`；合同写成 `ConstantLike tensor op`。
- 大 tensor / weight 不适合 inline dense attr 时，可以使用 resource-backed elements attr
  表示 constant data；这仍是 constant value 的存储方式，不是 Wafer 新 constant 语义。

示例：

```mlir
%w = arith.constant dense_resource<W_raw> : tensor<256x128xf16>
```

compile-time constant 如果需要目标相关排布，不生成新的 Wafer tensor constant op。相关 pass
直接读取 `ConstantLike` value 的 elements/resource，
按 consumer 需要的 layout marker 生成新的 backing data/resource，或在 `wafer.tile.load` lowering
时直接生成对应 storage。

constant 不是 `wafer.group` 的 external input，但它仍然是 tile execution 的 source。进入
`wafer.tile.region` 后，constant use 必须变成显式 `wafer.tile.load` 或等价 load source：

- tile shape 和 consumer indexing 决定 constant 的 tile slice / access range。
- `wafer.tile.load` result type 决定 load 后 memref 的 address space 和 layout marker。
- DDR memory planner 必须看到 constant 的 read-only source demand，包括 resident 或 streaming
  策略、storage size、alignment、lifetime、bandwidth 和 per-tile access range。
- 若原始 compact backing data 不能直接满足 result layout，planner 可以选择 compile-time storage
  transform，也可以保留 raw load 并在 consumer edge 插入 `wafer.tile.materialize_layout`。选择必须重新跑
  SPM 和 DDR feasibility。

并非所有 constant 都需要 DDR residency：

- scalar、splat、small attribute-like constant 如果能被 compute op 作为 immediate、attribute、fill
  pattern 或 folded value 消费，不生成 `wafer.tile.load`，也不产生 DDR demand。
- 一旦 constant 以 tensor data source 参与 tile compute，或者需要按 tile slice 读入 SPM，它就必须
  通过 `wafer.tile.load` / equivalent load source 表达，不能靠隐式 package side channel。
- 判断 immediate/fill/load 是 op lowering 的 legality 选择，必须由 op verifier 或 lowering pattern
  明确支持；不能靠 constant 名字、大小阈值的 ad hoc matcher。

weight 切分不是新的 Wafer constant 语义。它只是某个 `ConstantLike` tensor 被 tiled consumer
按 logical slice 使用：

- op tiling / `WaferTilingInterface` 先把 consumer tile 映射到 operand tile slice。GEMM weight、
  conv weight、scale/bias 这类常量都走同一条规则，不能靠名字或 “weight” 特判。
- `wafer.tile.load` 必须显式携带这个 source slice 或等价 index operands；slice 可以包含 reduction
  dimension、channel/block dimension，也可以是 op interface 推出的 hidden/internal dimension。
- storage transform 可以选择 whole-constant backing，也可以选择 chunked backing。chunk key 来自
  `(constant SSA value, logical slice/chunk shape, result layout marker, target policy)`，不是新的 IR 名词。
- chunked backing 只能覆盖已有 `wafer.tile.load` slice 的 union/coalescing；不能为了 packing 引入
  额外 compute tiling 或 reduction split。若需要改变 K/internal split，必须回到 group/op tiling planner。
- 如果 chunk 数量、package size、DDR residency 或 bandwidth 失控，应回退到 raw backing +
  materialization，或返回 group planner 调整 tile shape / group boundary。

如果输入来自 compile-time constant，并且 accepted assignment 选择 constant storage transform，
`wafer.tile.load` 仍然消费当前 IR 里的 constant value；它的 result type 表达下游看到的
physical layout：

```mlir
%w = arith.constant dense_resource<W_cx> : tensor<256x128xf16>
%w_ddr = bufferization.to_memref %w read_only
    : memref<256x128xf16, #wafer.memory<ddr, tensor>>

%wt_tensor = wafer.tile.load %w_ddr
    : memref<256x128xf16, #wafer.memory<ddr, tensor>>
   -> memref<256x128xf16, #wafer.memory<spm, tensor>>
%wt = wafer.tile.materialize_layout %wt_tensor
    : memref<256x128xf16, #wafer.memory<spm, tensor>>
   -> memref<256x128xf16, #wafer.memory<spm, cx>>
```

`wafer.tile.load` 的边界仍是 DDR memref；是否把 constant backing data 预先放成硬件友好的物理布局，
属于后续 constant/storage lowering 的事实。当前 layout planner 只固定 tile-local consumer 需要的
SPM layout，并在必要 edge 上插入 `wafer.tile.materialize_layout`，不改变 constant 的数学语义。

注意：在 semantic tensor IR 层，不能在保持同一个 logical tensor contract 的同时无标记地把
payload 改成另一种 physical byte order。packing 必须发生在 layout 已经被 `wafer.tile.load`
result type、storage-level type 或等价 verifier contract 约束之后；否则就是把数学值和存储表示混在一起。

### 5.2 Interfaces

V0 需要一个 op layout 接口；constant storage transform 是 pass 行为，不单独定义 Wafer op
interface。

R1.2 先落地 accepted-layout 层的可查询合同：`WaferLayoutOpInterface` 由 layout-sensitive
compute/movement op 实现，返回当前 IR 类型已经表达的 exact operand/result layout requirement，
并提供 verifier 可调用的组合检查。`wafer.tile.materialize_layout` 不实现该接口，它实现单独的
`WaferLayoutMaterializationOpInterface`，因为它是 explicit conversion edge，不是普通
layout-sensitive compute op。

```c++
void collectWaferLayoutRequirements(
    SmallVectorImpl<WaferLayoutRequirement> &requirements);

LogicalResult verifyWaferLayoutContract();

void collectWaferMaterializationLayouts(
    SmallVectorImpl<WaferLayoutRequirement> &requirements);

LogicalResult verifyWaferLayoutMaterializationContract();
```

后续 layout assignment planner 仍需要在 target-abstract / pre-assignment 阶段扩展更丰富的
domain/cost 查询。该扩展应保持同一接口边界，不能退回 pass side table：

```c++
LayoutDomain getAllowedMemLayouts(Value operandOrResult,
                                  TileShape tileShape,
                                  TargetAttr target);

SmallVector<LayoutPreference> getPreferredMemLayouts(Value operandOrResult,
                                                     TileShape tileShape,
                                                     TargetAttr target);

LogicalResult verifyMemLayoutCombination(Operation *op,
                                         ValueRange operands,
                                         ValueRange results);

SmallVector<MaterializationOption>
getMaterializationOptions(MemLayoutAttr src,
                          MemLayoutAttr dst,
                          ShapedType semanticType,
                          TargetAttr target);
```

约定：

- allowed layout 是 hard legality。
- preferred layout 只参与 cost/tie-break。
- accepted-layout requirement 是当前 IR 已经承诺的事实；它不能反向伪装成 planner 的搜索空间。
- op 如果不关心 physical layout，可以不实现该接口，由通用 passthrough/flexible 规则处理。
- shape-changing、rank-changing、aligned-only、layout-sensitive movement op 必须实现接口或提供
  verifier 可调用的规则。

V0 implementer 范围：

| op / provider | 是否实现 | 说明 |
| --- | --- | --- |
| `wafer.tile.load` / 等价 boundary load op | 必须 | 暴露 `#wafer.memory<ddr, tensor>` compact external boundary、constant source 和 result memref 的 allowed/preferred layout |
| `wafer.tile.store` / 等价 boundary store op | 必须 | 暴露 `#wafer.memory<ddr, tensor>` host-visible compact writeback、device-side group boundary 和 store input layout 约束 |
| `wafer.tile.gemm` / NE-style matmul op | 必须 | 通常是 aligned-only consumer/producer；决定 operand/result 是否必须是 `Cx/NCx` family |
| `wafer.tile.reduce`、pool、unpool | 必须 | 这些 op 有硬件 layout legality，不能靠通用 passthrough 规则猜 |
| `wafer.tile.elementwise` / CT-style flexible op | 必须或提供默认 flexible trait | 若 op 只是 shape-preserving passthrough，可复用默认 flexible 规则；若受 dtype/range/wrapper 限制，必须实现接口 |
| `wafer.tile.*` target-abstract movement op | 必须 | load/store、local movement 如果限制 physical layout、range 或 stride，需要把限制暴露给 planner/verifier；不新增单独 movement op namespace |
| `wafer.tile.*` communication ops 中消费/产生 storage 的 op | 必须或提供等价 relation | p2p comm 默认 byte-preserving，但仍要暴露 source/destination buffer、byte count、token/effect 和 staging demand；collective-level op 展开前只表达 semantic，展开后由 p2p op 验证 |
| `wafer.tile.materialize_layout` | 不实现这个接口；实现 `WaferLayoutMaterializationOpInterface` | 它表示 layout conversion edge，本身由 source/result type 和 materialization interface 验证 |
| `wafer.group`、`wafer.tile.region`、`scf.*` | 不实现 | 它们提供 region/control-flow/边界结构；layout 约束来自 region 内 value 和 op interface |
| `linalg.*` / upstream compute op | 不直接实现 | 进入 Wafer planning 后由选中的 Wafer lowerable implementation 或 adapter 提供 layout contract，不修改 upstream dialect |
| `arith.constant` / `stablehlo.constant` / generic `ConstantLike` op | 不实现 | constant 不消费 storage；transform pass 读取其 value/resource 并在 storage lowering 附近改写 |

constant storage transform 的 implementer 是 pass / pattern，不是 op interface：

- 输入必须是已经 normalized 的 `ConstantLike` tensor value；若来自 `stablehlo.constant`，必须先经过
  constant normalization。
- transform 从当前 IR 的 use-def、accepted tile shape、op tiling interface 推出的 operand slice、
  `wafer.tile.load` tile slice、result layout marker、target policy 和 `computeWaferPhysicalTensorInfo`
  推导目标 storage order。
- transform 直接替换 constant backing data/resource，或在 lowering `wafer.tile.load` 时生成
  packed storage；不创建新的旁路 IR 事实源。
- transform 不处理已经被合法 fold 成 immediate / attribute / fill pattern 的 constants；这类
  constants 不进入 DDR demand。
- dynamic host input/output 不能走这条路径；它们仍受 `#wafer.memory<ddr, tensor>` compact external ABI 和 runtime allocation
  约束。

### 5.3 Pass Contract

pass 名是实现组织，不是架构边界；边界仍以 IR contract 和 verifier 为准。V0 可以按以下 pass
组织：

1. `wafer-layout-materialize`

   运行位置：whole-variant candidate clone 的完整 traversal 已经形成、tile shape 仍是候选选择、
   layout-sensitive tiled op 已经被选成 target-abstract `wafer.tile.*` compute / boundary / data movement
   op、下游 SPM allocator 可用、最终硬件 movement 还没 lower 之前。

   输入：

   - target-abstract `wafer.tile.region` SSA body。
   - op 的 `WaferLayoutOpInterface`。
   - `wafer.tile.*` communication p2p op 的 byte-preserving layout relation、token/effect 和 staging demand。
   - `#wafer.memory<ddr, tensor>` compact external boundary。
   - normalized `ConstantLike` tensor value 及其 backing data/resource。
   - target policy、SPM allocator 和 DDR memory planner。

   行为：

   - 从当前 IR 构造 `LayoutVariable` / `LayoutEdge` analysis；analysis 不写入 IR。
   - 生成 group-local layout alternatives 和 early boundary proposals；这些结果不拥有最终 layout。
   - 对 complete rank entry 的 device-side group-to-group edge 做 bounded co-planning，并把当前
     boundary proposal 作为 clone 内可失效、可重算的 hard constraint 传回本地 planner。
   - 选择最终 physical layout assignment 和 materialization edge。
   - 对候选 assignment 调用 SPM allocator 和 DDR memory planner；改变 boundary
     layout 或 materialization cut 后必须重新验证。
   - layout proposal 通过当前 whole-entry layout/resource checks 后，把 final candidate assignment
     rewrite 到 clone：生成或更新 tile-local memref 的 `#wafer.memory<space, layout>`，插入
     `wafer.tile.materialize_layout`，并把 compile-time constant 的 storage transform 机会保留在
     `wafer.tile.load` use 上。
     这个 rewrite 是后续 instruction/SPM/DDR gates 的输入，不是主 IR commit；只有全部 whole-variant
     gates 通过后，clone 中的 memref type / explicit movement 才随 atomic commit 成为最终 owner。
   - 对 rejected layout alternative 只发 diagnostic；任一 group/rank 失败都丢弃 clone，不部分更新主 IR。

   输出：

   - 带 `#wafer.memory<space, layout>` 的 tile-local memref values。
   - 明确的 `wafer.tile.materialize_layout` op。
   - `wafer.tile.load` 对 constant source 的明确 use-def 关系，供后续 constant resource transform /
     lowering 使用。

2. `wafer-layout-materialize-cleanup`

   运行位置：candidate clone 中 final layout IR 已经形成、lower-level movement 尚未生成且 atomic commit
   尚未发生之前。它也可以实现为
   `wafer-layout-materialize` 末尾的一组 canonicalization pattern；是否拆 pass 不影响 IR contract。

   行为：

   - 删除 same-layout、dead、inverse-pair materialization。
   - 对 flexible op 做 layout-aware rewrite，减少两侧 conversion。
   - 合并重复 conversion 或移动 materialization cut 前，重新运行 SPM allocation。
   - 不读取旧 planner side table，不创建新的 layout plan attr。

3. Constant storage/resource transform stage

   运行位置：whole-variant layout assignment 已接受之后，`wafer.tile.load` lowering 或 package emission 之前。

   输入：

   - normalized `ConstantLike` tensor value。
   - `wafer.tile.load` use、result layout marker、tile slice 和从 op tiling interface 推出的 operand
     slice relation。
   - 原始 constant data。
   - `computeWaferPhysicalTensorInfo` calculator 和 target policy。

   行为：

   - 若 constant 的 consumer layout 可以在编译期满足，按 accepted tile shape / tile slice 生成
     对应 storage order 的 backing data/resource，或把 `wafer.tile.load` lower 成使用 packed
     storage 的形式。
   - 对 weight / large constant，允许 whole-constant transform、按 tile-slice chunk transform、
     或按多个 load slice 的 union/coalesced chunk transform；chunk 只服务已有 `wafer.tile.load`
     use，不改变 compute tile。
   - 为 constant source 生成 read-only DDR demand：resident constant、streaming load 或 staging
     由 DDR planner 根据 capacity、range、alignment、bandwidth 和 reuse 决定。
   - 若同一个 constant 被多个 incompatible consumers 使用，可以 clone / specialize constant use，
     或保留 device-side `wafer.tile.materialize_layout`；选择由 cost、SPM 和 DDR feasibility 决定。
   - 最终 package emission 只序列化当前 IR 已经选定的 constant storage，不创建新的 IR 事实源。

   不负责：

   - 不选择 tile shape。
   - 不决定 package/load-time storage materialization cut。
   - 不修改 tensor semantic layout。

4. Runtime / ABI address derivation

   运行位置：accepted layout IR 和 cleanup 完成之后，lower-level movement/compute emission 需要
   address/range/stride 参数之前。它不是独立主线 IR 阶段；边界是从 committed IR 和 accepted facts
   派生 very-late emission 参数。

   输入：

   - tile-local memref 的 logical shape、dtype 和 `#wafer.memory<space, layout>`。
   - `computeWaferPhysicalTensorInfo` calculator。
   - SPM allocation / range / lifetime result。
   - target descriptor / emission policy。

   行为：

   - compact layout 的 address/range 由 committed memref、accepted offset fact 和 compact footprint 派生。
   - `Cx/NCx` 的 physical extent、padding、flat storage span 和 packet fields 由
     `computeWaferPhysicalTensorInfo` 派生。
   - 给 lower-level movement/compute emission 生成 address/range/stride/layout 参数。
   - 不在主线 IR 中删除或替换 Wafer-tagged memref 以制造第二份 storage 事实源。

   不负责：

   - 不重新选择 physical layout。
   - 不重新插入或移动 `wafer.tile.materialize_layout`。
   - 不绕过标准 memref/LLVM lowering；能复用标准 conversion 的 storage 必须复用。

5. `wafer-lower-layout-materialize`

   运行位置：committed instruction IR 已经带有 accepted layout/materialization facts 之后，
   lower-level movement/compute emission 之前或过程中。

   输入：

   - `wafer.tile.materialize_layout` op。
   - source/result physical layout、shape、dtype、memory space。
   - SPM allocation / range-end 信息。
   - target lowering table。

   行为：

   - 选择 `ChannelNorm`、`DechannelNorm`、`GatherScatter`、TDMA 或 wrapper path。
   - 生成 lower-level movement op 和必要 fence/wait effect。
   - 删除已经 lower 的 `wafer.tile.materialize_layout`。

   不负责：

   - 不重新规划 layout assignment。
   - 不移动 materialization cut。
   - 不重新打包 compile-time constant。

## 6. Planning Model

layout planner 在 transformation 内部构造第 3 节生命周期第二层的局部 constraint graph，不把它
写进 IR。它必须能从当前 scheduled tile tensor IR 重算；一旦 rewrite 改变 use-def、loop-carried
关系或 op interface 结果，旧 analysis 即失效：

```text
LayoutVariable {
  value
  domain
  hard_constraints
  preferences
}

LayoutEdge {
  producer_value
  consumer_operand
  relation
  materializable
}
```

变量来源：

- tile-local SSA value。
- DPS / in-place tie 形成的 alias group。
- loop-carried entry / yield value。
- group input/output 和 host-visible writeback boundary。
- materialization location alternative 产生的新 value。

约束来源：

- op layout interface 的 allowed / preferred layouts。
- dtype、rank、shape、bool bitpack、C0 tail/fold、padding、range-end 规则。
- `#wafer.memory<ddr, tensor>` compact external ABI。
- compile-time constant source 的 current backing data/resource，以及 `wafer.tile.load` result layout。
- producer-consumer edge 是否允许插入 real movement。

V0 只需要区分 hard constraint 和 preference：

- hard constraint 失败就是 layout infeasible，不能靠 cost model 覆盖。
- preference 只影响 assignment 和 tie-break，不是 verifier 合同。
- producer 和 consumer 的 selected layout 冲突时，只有 materializable edge 可以插入
  `wafer.tile.materialize_layout`；否则必须回到 group planner 拆 group、换 tile 或请求 op-local
  implementation。

final candidate assignment 只通过 clone 中 rewrite 后的 tile-local memref type、
`wafer.tile.materialize_layout`、`wafer.tile.load` 的 use-def 和 result type 体现。domain frontier、
cost breakdown、备选 cut、失败原因都属于 diagnostic / debug dump，不能成为下游 pass 依赖的
IR 事实；atomic commit 后同一组 memref types / explicit movement 成为 committed layout 的单一 owner。

### 6.1 Materialization Cut Location Algorithm

materialization cut 的核心问题是同时决定两件事：

```text
layout assignment:
  每个 tile-local SSA value / alias group 选择一个 physical layout label

cut location:
  如果 producer value 和 consumer operand 的 selected layout 不一致，
  在哪条 materializable edge 上插入 wafer.tile.materialize_layout
```

V0 不把这个问题写成“遇到某类 op 就插 conversion”。通用算法是 hard constraint propagation、
小规模 graph labeling、局部 cut 优化、SPM allocation 和 DDR memory planning 的组合。

#### 6.1.1 Hard Constraint Propagation

先从当前 IR、type、op interface 和 boundary contract 推导 hard domain：

- aligned-only op 收窄 operand/result domain，例如 NE/GEMM、native reduce、pool/unpool。
- host-visible dynamic input/output 固定为 `#wafer.memory<ddr, tensor>` compact external layout。
- loop-carried entry/yield value 必须 layout 一致，除非 loop body 内有显式 materialization。
- DPS / in-place / must-alias value 必须共享 compatible layout。
- non-materializable producer-consumer edge 两端必须同 layout。
- communication p2p edge 默认 byte-preserving，source/destination physical layout relation 必须明确。
- compile-time constant 可以作为 specializable source；它的 storage transform 只在 load/storage
  lowering 附近发生，不改变 tensor value 的数学语义。

传播后如果某个 `LayoutVariable.domain` 为空，当前 tile plan 直接 layout-infeasible，返回 group
planner 调整 tile shape、internal split 或 group boundary。

#### 6.1.2 Graph Labeling Model

hard domain 非空后，剩余选择可以看成小规模 multi-label graph labeling：

```text
label(value) ∈ allowed_layouts(value)

node_cost(value, label):
  storage_size(label)
  + preferred_layout_penalty
  + lifetime_pressure(label)

edge_cost(producer, consumer, src_label, dst_label):
  0, if src_label == dst_label
  materialization_cost(src_label -> dst_label), if edge materializable
  infinite, otherwise
```

op 不是简单二元边时，使用 op interface 给出的 combination verifier 处理 hyperedge：

- 如果 operand/result layout 组合不合法，该组合 cost 是 infinite。
- 如果只是 preference 低，记入 penalty，不变成 hard failure。
- loop body 内 edge 的 materialization cost 应乘以 loop trip count 或 loop-depth weight。
- large tensor / high fanout value 的 repeated conversion penalty 要高于 small temporary。

这只是 analysis model，不进入 IR。

#### 6.1.3 V0 Greedy Assignment

V0 不默认上 ILP。先用 deterministic greedy 生成一个初始 assignment：

1. 先满足所有 hard constrained op。
2. 对 flexible chain，默认继承 producer layout，避免无意义 conversion。
3. 对 high-fanout value，优先选择能被最多 hard consumers 接受的 layout。
4. 对 large tensor，避免为多个 consumers 重复 materialize。
5. 对 host-visible output，只在最终 writeback boundary 前 materialize 回 compact。
6. 对 compile-time constant，把 compile-time storage transform 作为候选之一；只有 consumer layout、
   package size、SPM allocation 和 DDR memory planning 都合适时才接受。

这个 assignment 只产生 layout alternative，不代表已经可 lower。

#### 6.1.4 Cut Location Rules

给定 selected layout 后，只在 layout 不一致的 materializable edge 上放 cut。局部规则：

- 单 consumer 需要不同 layout：cut 放在该 consumer edge 上。
- 多个 consumers 都需要同一 destination layout：尝试 hoist 到共同支配点，但必须检查 lifetime
  是否显著变长。
- high-fanout value 中只有少数 small consumers 需要不同 layout：保持 producer layout，在少数
  consumer edge 上 materialize。
- flexible op 两侧都有 conversion：尝试改写 flexible op 的 operand/result layout，让它直接接受
  producer layout 或直接产生 consumer layout。
- loop 内 conversion：优先把 cut 移到 loop 外，除非 layout legality 或 lifetime 明确不允许。
- host-visible boundary：不能把 compact writeback requirement 移出 ABI boundary。
- communication p2p boundary：若 producer/consumer 都能接受同一 physical layout，优先保持
  byte-preserving transfer；否则在明确 edge 上插 `wafer.tile.materialize_layout`。

任何 cut 移动只是在 layout alternative 上发生。被接受前不能写入 `wafer.group` 或全局 attr。

#### 6.1.5 Bounded Alternatives and Repair

初始 assignment 失败时，只对少数高影响 conflict 保留 bounded frontier，不做全图搜索：

- materialize 在 producer edge。
- materialize 在 consumer chain 入口。
- hoist 到共同支配点。
- sink 到唯一 hard consumer。
- 让 flexible op 继承另一侧 layout。
- 对 compile-time constant 改用 specialized storage backing。
- 标记 conflict edge，建议 group planner 拆 group 或缩 tile。

frontier 大小由 target policy 控制，必须是小常数。每个 alternative 都要重新构造真实
materialization demand 并运行 SPM allocation。

#### 6.1.6 Resource Acceptance

最终接受条件不是 graph cost 最低，而是：

- layout combination verifier 通过。
- `wafer.tile.materialize_layout` 都有 lowerable conversion path。
- SPM allocation 通过，包含 materialization temp、communication staging、loop-carried value、
  async lifetime 和 range/end-address validation。
- DDR external view/descriptor demand、compiler-managed DDR `memref.alloc`、resident constant demand 和
  bandwidth summary 可由 DDR memory planner 接受。
- cleanup 后仍能通过 layout verifier、SPM allocation 和 DDR memory planning。

如果所有 bounded alternatives 都失败，layout planner 不生成“等待下游修复”的 IR，而是返回
structured failure，让 group planner 调整 tile shape、internal split、output coverage 或 group
boundary。

#### 6.1.7 Min-Cut / ILP 的位置

二元 min-cut 只适合受限子问题：

- 只有两种 layout，例如 compact vs aligned。
- 所有约束都能表示成 unary / pairwise cost。
- 不考虑 SPM peak、lifetime、loop-carried alias 和 communication boundary。
- pairwise cost 满足可被 min-cut 处理的形式。

Wafer 的常见 case 是 multi-label、op hyperedge、loop-carried value、SPM peak 和 boundary
co-planning 耦合，所以 V0 不把 min-cut 当主算法。后续如果 profiling 证明某个局部区域的
conversion cost 是主瓶颈，可以在 bounded frontier 的局部子图上引入 ILP / beam search；结果仍必须
通过同一套 verifier、SPM allocation 和 DDR memory planning。

## 7. Boundary Contract

dynamic host input/output：

- 固定为 `#wafer.memory<ddr, tensor>` compact external layout。
- 不要求用户提供 `Cx/NCx`。

compile-time constants：

- 默认保留原始 compact backing data/resource。
- scalar、splat 或 small constants 可以由 op lowering fold 成 immediate / attribute / fill，不产生
  DDR demand；前提是目标 op verifier 明确允许。
- 若 layout/materialization planning 选择 compile-time storage transform，必须由显式 transform
  pass 直接改写 constant backing data/resource，或在 lowering `wafer.tile.load` 时生成对应 storage。
- constant 虽然不是 group external input，但每个 tile-region use 都必须通过显式 load source
  表达，并参与 DDR demand / tiling / bandwidth 计算。
- 若同一个 constant 被多个 incompatible consumers 共享，V0 可以 clone / specialize constant use，
  也可以在 consumer edge 做 device-side materialization；选择由 cost、package size、SPM allocation
  和 DDR memory planning 决定。
- constant storage transform 只是把 conversion 提前到编译期执行，不是新的上层 IR 语义。

load/store：

- load/store 根据 source/destination memory attr 和 layout assignment 选择 movement lowering。
  典型 dynamic boundary 是 `#wafer.memory<ddr, tensor>` compact buffer 与
  `#wafer.memory<spm, *>` tile-local memref 之间的 RDMA/WDMA。
- 它们不是 layout 的根本来源。

group output：

- device-side group-to-group value 可以在 candidate clone 中保持 selected physical layout；atomic commit
  后由其 memref type / explicit movement 成为 final layout fact。
- host-visible output 在 writeback 前必须回到 `#wafer.memory<ddr, tensor>` compact external layout。

## 8. 跨 Group Boundary Co-Planning

跨 group co-planning 要解决的是这种模式：

```text
group A produces value in layout X
  -> materialize X -> compact at A output
  -> store/load device-side boundary
  -> materialize compact -> Y at group B input
```

如果这个 boundary 不是 host-visible、不逃逸到未知 runtime ABI、且下游 verifier 能用 boundary
value 的 type/effect 检查 `#wafer.memory<space, layout>`，那它可以保持 `X` 或 `Y`，不必强制回到
compact。这个选择仍然不能写成 `wafer.group` 上的全局 layout plan；它只能作为当前 planning 的
analysis，先通过 candidate clone 中 boundary buffer value 的 type 和必要的
`wafer.tile.materialize_layout` 供全 entry gates 验证，只有 atomic commit 后才进入主 IR。

V0 的优化搜索只做 bounded adjacent co-planning，不追求 full-program 最优；但最终 legality 必须在
完整 static rank entry/variant-set 上验证：

1. 本地 summary

   对每个 logical group，layout planner 在 rewrite 前生成 boundary summary：

   ```text
   BoundaryLayoutSummary {
     boundary_value
     direction        // producer result or consumer operand
     allowed_layouts
     preferred_layouts
     local_cost(layout)
     spm_allocation(layout)
     ddr_memory_plan(layout)
   }
   ```

   summary 是 analysis，不进入 IR。`local_cost` 必须包含 group 内 materialization bytes、peak SPM
   变化、compiler-managed DDR allocation demand / bandwidth pressure 和 writeback/load movement；`spm_allocation` 必须来自
   同一个 SPM allocator，`ddr_memory_plan` 必须来自 DDR memory planner。

2. 建 boundary graph

   只为 device-side group-to-group SSA edge 建 graph。host-visible output、unknown alias、runtime
   escape、必须按 ABI compact 的 edge 不进入 co-planning，仍按 boundary contract materialize。

3. 选 boundary layout

   对相邻 producer/consumer edge，候选 layout 来自：

   ```text
   producer.allowed_output_layouts
     ∩ consumer.allowed_input_layouts
     ∩ boundary_storage_allowed_layouts
   ```

   `boundary_storage_allowed_layouts` 由 boundary 的 memory space、load/store/movement interface 和
   target policy 推出，不是额外 attr。

   若交集非空，选择使

   ```text
   producer_local_cost(layout)
   + consumer_local_cost(layout)
   + boundary_storage_cost(layout)
   + remaining_materialization_cost
   ```

   最小的 layout。fanout boundary 使用小 frontier：优先选择能被最多 hard consumers 接受、并减少
   large tensor repeated conversion 的 layout；如果 fanout consumers 的 hard layout 冲突严重，
   保留 producer layout，并在少数 consumer edge 上 materialize，或者返回 group planner 建议拆分。

4. 带 boundary constraint 重新本地规划

   selected candidate boundary layout 作为当前 clone 的 hard boundary constraint 传回 producer 和
   consumer 的本地 planner。若任一 group 的 SPM allocation 或 DDR memory planning 失败，回退到下一个
   boundary alternative；
   frontier 耗尽时，退回本地规划并保留显式 boundary materialization。

5. Rewrite

   只有 accepted candidate result 写入 clone：producer boundary value、consumer boundary argument/load
   value 的 physical layout 一致，或者 boundary edge 上有明确 `wafer.tile.materialize_layout`。
   co-planning summary、备选 layout 和失败原因不写入 IR；clone 必须再通过完整 traversal 的
   SPM/DDR/lifetime/instruction/event/transport/target gates，才能原子写入主 IR。

这个机制的关键不是引入全局最优，而是允许相邻 group 在 device-side boundary 上共享一个 verifier
可见的 physical layout。它覆盖常见 repeated materialization 成本；group-local proposal 可独立重算，
但 final layout、SPM/DDR lifetime 和 completion legality 必须在完整 rank entry/variant-set 上共同验证。

## 9. Materialization Cleanup

初始 layout materialization 可能保守插入多余 conversion。accepted IR 形成后，必须运行
layout-aware cleanup；它是普通 IR rewrite / canonicalization，不是重新解释旧 planner state。

可以无条件做的局部 fold：

- `materialize A -> A` 直接删除。
- `materialize(A -> B)` 的结果无 use 时删除。
- `materialize(A -> B -> A)` 且中间 `B` value 没有其它 use 时，把最终 uses 改回原始 `A` value。
- 同一 source、同一 destination layout、同一 shape/dtype 的 materialization，如果共享结果不会引入
  新的 lifetime 冲突，可以合并；否则不能只按文本相同做 CSE。

需要重新验证 layout interface、SPM allocation 和 DDR memory planning 的 rewrite：

- sink：把 conversion 从 producer 后推到真正需要该 layout 的 consumer 前，避免 fanout 上所有 use
  都承担 conversion。
- hoist：多个 consumers 都需要同一 destination layout 时，把重复 conversion 合并到共同支配点。
- through flexible op：若 flexible op 可以接受 source layout 或直接产生 destination layout，改写
  op 的 operand/result layout，删除两侧 materialization。
- boundary pair elimination：producer group output 先 materialize 到 compact，consumer group input
  又 materialize 回 aligned 时，如果 boundary co-planning 证明 device-side boundary 可保持 aligned，
  删除这一对 boundary conversion，并把 boundary value type 改成 accepted layout。

cleanup 的限制：

- 不能跨 host-visible boundary、unknown alias、runtime escape 或 verifier 无法表达 layout 的 edge。
- 不能改变 tensor semantic layout、shape、dtype 或 memory space。
- 任何会延长 buffer lifetime、改变 materialization cut 或合并多个 conversion result 的 rewrite，
  都必须重新运行 SPM allocation 和 DDR memory planning；只删除 dead/same-layout
  conversion 且缩短 lifetime 的 fold 可以直接应用。
- cleanup 后仍由 verifier 检查 op layout contract、loop-carried entry/yield layout、
  `wafer.tile.load` result layout、constant source 可 lower 性和 lowerable conversion path。

## 10. V0 Algorithm

V0 不做全局最优，但不能只做一次贪心选择。主路径是 deterministic greedy assignment，加一个
很小的 bounded alternative frontier，用来处理 layout 和 SPM 强耦合的失败 case。

1. 建 constraint graph

   为每个 tile-local value / alias group 建 `LayoutVariable`。host input/output 固定 compact；
   compile-time constant 通过 `wafer.tile.load` use 参与图，tile slice 由 accepted tile shape、
   consumer indexing 和 op tiling interface 推导；它使用当前 backing data/resource，并可带有显式
   storage transform pass 提供的 whole/chunked backing 候选。aligned-only op 收窄自己的 operand/result；
   flexible op 保留 compact 和 aligned 候选。

2. 传播 hard constraint

   沿 SSA use-def、DPS/in-place tie、loop-carried value 和 group boundary 传播。若某个 value 的
   domain 变空，直接向 group planner 返回 layout infeasible。

3. 选 assignment

   使用第 6.1 节的 greedy graph labeling 生成初始 assignment：

   - 先满足 aligned-only op。
   - 对 flexible chain，继承 producer 或 dominant consumer 的 layout。
   - 对高 fanout / large tensor，尽量避免多次 conversion。
   - 对 compile-time constant，在成本合适时优先把 conversion 放到 compile-time storage transform，而不是
     runtime materialize。
   - 对 host-visible output，只在最终 writeback boundary 前 materialize 回 compact。

4. 插 materialization location alternative

   只在 producer/consumer selected layout 不一致的 edge 上插 alternative。cut location 使用第
   6.1.4 节的规则。多 use producer 的常见策略：

   - 若 consumers 都能接受 aligned，保持 aligned。
   - 若只有一个小 consumer 需要不同 layout，在该 edge 上 materialize。
   - 若多个 large consumers 需要 incompatible hard layout，返回 group planner，倾向拆 group 或重选 tile。

5. 保留 bounded alternatives

   对少数高影响 conflict edge 保留第 6.1.5 节的候选，不做全图搜索：

   - materialize 在 producer edge。
   - materialize 在 consumer chain 入口。
   - hoist 到共同支配点。
   - sink 到唯一 hard consumer。
   - 对 compile-time constant 改用 specialized storage backing。
   - 让 flexible op 继承另一个已有 layout。
   - 把 conflict edge 标记为建议 group split。

   frontier 大小应是 target policy 控制的小常量；它是编译期 search 策略，不进入 IR。

6. 生成 whole-entry resource-planned alternatives

   对初始 assignment 和 bounded alternatives 构造真实 materialization demand，运行 SPM allocation
   和 DDR memory planning，形成 group-local early alternative / boundary proposal，并立即放入完整
   rank-entry lifetime/resource context 验证。SPM 或 DDR
   失败时，layout planner 只做有限调整：

   - 移动 materialization cut。
   - 改用 specialized constant storage backing。
   - 让 flexible op 接受另一个已有 layout。

   如果仍没有本地可行 alternative，返回 group planner 调整 tile shape、internal split 或 group
   boundary。

7. 完整 rank entry 的跨 group boundary co-planning

   对 device-side group-to-group edge，收集相邻 group 的 boundary proposals，在 bounded frontier 内选择
   final boundary layout，并把它作为当前 clone 的 hard constraint 传回相关 group 的本地 planner。
   必须在完整 traversal、全 entry SPM/DDR lifetime 下重新验证；如果 co-planning 失败，可以在 clone
   中尝试显式 boundary materialization，仍失败则丢弃整个 clone，不提交已通过的 group。

8. Rewrite candidate clone

   对选中的 final candidate assignment 改写 clone：`wafer.tile.region` 中出现 Wafer-tagged memref 和
   `wafer.tile.materialize_layout`。具体 `ChannelNorm` / `GatherScatter` / TDMA 等 lower-level movement
   由后续 lowering pass 生成。该 provisional rewrite 供完整 instruction/SPM/DDR/event/transport/target
   gates 消费；只有这些 gates 全部通过，clone 才能原子提交。

9. Cleanup

   在 candidate clone 上运行 layout materialization cleanup。无条件 fold 直接删除冗余 op；改变 lifetime
   或 boundary layout 的 rewrite 必须重新通过 SPM allocation 和 DDR memory planning。

10. Resource / endpoint / launch / ABI handoff

   cleanup和memory/event gates通过后，pre-commit `ExecutableResourceView`、physical transport、launch
   projection和target-entry ABI preflight在同一candidate clone中闭合，再atomic commit typed executable。
   post-commit target从committed memref、typed entry/resources、accepted offsets、view relation、projection和
   layout helper派生address/range/stride；package只序列化typed owners。layout planner不直接生成LLVM ABI，
   但必须保证accepted layout都能被这个派生过程合法实现。

## 11. Cost Model

V0 cost 保持简单：

```text
cost =
  materialization_bytes
  + added_peak_spm_bytes * spm_pressure_coeff
  + extra_writeback_or_load_bytes
  - package_time_transform_benefit
```

tie-breaker：

- 少插 materialization。
- 少延长 large buffer lifetime。
- 少在 loop 内做 conversion。
- 优先把 conversion 放在 small tensor 或 boundary 上。
- 对 device-side group boundary，优先复用 producer/consumer 都能验证的 physical layout，避免成对
  materialize。

bank/page coloring、PMU latency、overlap blocking 暂时只作为 SPM/scheduler 的 cost input，不在
layout planner 里单独做复杂模型。

## 12. IR 表达

`wafer.group` 不携带 physical layout assignment。

layout 相关事实按第 3 节生命周期分层表达：

- candidate/template scheduled tile tensor IR 表达 tensor dataflow，不表达 final physical layout。
- `LayoutVariable` / `LayoutEdge` 是 pass-local analysis，不表达成 op/type/attr。
- group boundary proposal / co-planning frontier 是 pass-local analysis，不表达成 op/type/attr，也不是最终 owner。
- committed complete static rank program 的 `wafer.tile.region` / SPM bufferization 层是 final layout 单一 owner，表达：
  - tile-local memref 的 address space 和 physical layout marker。
  - `wafer.tile.materialize_layout` 或等价 explicit data movement op。
  - `wafer.tile.load` 对 `ConstantLike` source 的 use-def，以及 result layout marker。
  - op verifier 可检查的 layout contract。
- runtime/target-codegen 派生层从 committed Wafer-tagged memref 和 accepted offset facts 得到
  address/range/stride 参数。compact layout 应尽量复用标准 memref lowering；Cx/NCx 的 lower-level
  storage facts 只作为 very-late emission 参数出现，不作为独立 access descriptor IR 传递。
- lower-level movement IR 表达具体 instruction/wrapper path、sync/effect 和 byte/range 约束。

不要维护全局 `layout_plan` attr，也不要把 analysis graph 序列化成 side table 给后续 pass 使用。
失败的 layout alternative、cost breakdown、materialization 尝试都只是 analysis。

## 13. Verifier

layout verifier 至少检查：

- `#wafer.memory<space, layout>` 的 layout marker 只包含 `Tensor/NTensor/Cx/NCx` family，不包含
  `axis/C0/aligned_C/storage_bytes` 这类可推导字段。
- aligned-only op 的 operand/result physical layout 合法。
- 通过统一 calculator 推出的 physical tensor info 与 dtype、rank、shape、C0 tail/fold、bool
  bitpack 一致。
- materialization source/destination layout 不同，且 conversion path 可 lower。
- host-visible dynamic input/output 满足 `#wafer.memory<ddr, tensor>` compact external layout contract。
- compile-time constant source 可以被 `wafer.tile.load` lowering 成 result layout 要求的 storage。
- loop-carried value 的 entry/yield layout 一致，除非 loop body 中有明确 materialization。
- device-side group boundary 的 producer value 和 consumer value physical layout 一致；如果不一致，
  boundary edge 上必须有明确且可 lower 的 `wafer.tile.materialize_layout`。
- 所有 physical footprint、padding、view/root range、descriptor fields 和 ABI width narrowing 必须调用
  shared physical geometry/range/narrowing verifier；禁止各 consumer 各算一套或静默把宽整数截成窄字段。
- final layout verification 在 complete rank entry/full traversal 上运行；early proposal、单 group 或
  representative tile 通过不构成 completion，任一失败必须阻止 whole-variant commit。

lowered movement verifier 另行检查具体 movement 的 byte size、stride、range-end、sync/effect 与
physical layout 一致；这些事实不反写进 `wafer.tile.materialize_layout`。

`wafer.tile.*` communication verifier 另行检查 peer、route、token/wait、DTE/FSM resource 和 fixed-size unicast
约束。layout verifier 只要求 communication source/destination physical layout relation 明确：
如果传输前后 layout 相同，可以直接 byte-preserving transfer；如果 layout 不同，必须有显式
`wafer.tile.materialize_layout` 或其它可验证 movement edge。

## 14. 后续扩展

这些机制有价值，但不进入 V0 主路径。进入条件必须明确，不能因为“看起来更强”就提前实现：

- beam search：当局部贪心在 LLM 主线 case 上频繁把 large tensor 反复 materialize，且换 cut 可明显
  降低 SPM peak 或 movement bytes 时引入。
- ILP / 全局最优 assignment：只在小型子图离线 tuning 或 debug mode 中考虑，不作为默认 compiler
  path。
- 跨多个 group 的全局 layout optimization：当 bounded adjacent co-planning 仍无法消除主要 repeated
  materialization，且 profile 显示跨长链 layout 决策成为主成本时引入。
- 多版本 device-side materialization cache：只有同一 value 在多个 incompatible consumers 间反复转换，
  且 SPM/DDR tradeoff 明确优于 recompute / split group 时引入。
- PMU conversion latency model：当 board profiling 能稳定区分 `GatherScatter` / TDMA / wrapper path
  的 latency 后替换 V0 byte-based cost。
- non-compact host tensor ABI：只有 runtime/package contract 明确暴露 layout，并且用户侧 framework
  能安全交付这种 layout 时引入。

## 15. 参考材料

- MLIR Bufferization / One-Shot Bufferize：<https://mlir.llvm.org/docs/Bufferization/>
- MLIR Data Layout：<https://mlir.llvm.org/docs/DataLayout/>
- TVM ConvertLayout：<https://daobook.github.io/tvm/docs/arch/convert_layout.html>
- XLA BufferAssignment：<https://openxla.org/xla/hlo_to_thunks>
- TFLM Memory Planner：<https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf>
