# Wafer Layout Materialization Design

日期：2026-05-21

状态：设计草案；2026-05-25 边界收口；2026-06-04 对齐 instruction-level Wafer IR 先于
SPM placement

本文定义 Wafer 后端的 physical layout planning 和 layout materialization 边界。它服务于
`wafer.group` 的 legality search，也服务于 `wafer.tile_region` / SPM bufferization 的真实
lowering。

本文只负责：

- 从 target-abstract `wafer.compute` / movement / communication op 的 layout interface 和
  boundary contract 推导 physical layout assignment。
- 在 accepted `wafer.tile_region` 上插入 `wafer.layout.materialize` 或选择 compile-time constant
  storage transform。
- 做 bounded group-to-group boundary co-planning 和 materialization cleanup。

本文不负责 group formation、tile shape search、SPM offset allocation、DDR BO allocation、
compute/comm op 语义、launch/runtime package 或 raw instruction packet。layout planner 的中间
图、candidate、cost trace 和失败原因都是 analysis；只有 accepted buffer layout、explicit
movement 和可 lower 的 constant storage 选择进入 IR 或 lowering 输入。

## 1. 核心结论

layout materialization 是从硬约束出发的局部 dataflow planning：

```text
scheduled tile tensor IR
  -> pass-local LayoutVariable / LayoutEdge analysis
  -> bounded group-to-group boundary co-planning
  -> accepted tile-local buffer IR with physical layout
  -> materialization cleanup / canonicalization
  -> tile buffer storage realization
  -> concrete movement / compute lowering
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
- layout assignment 必须调用 SPM allocation 和 DDR resource planning；只算 layout conversion 次数不够。

## 3. 表示生命周期

layout materialization 文档内部只使用一条表示生命周期，避免把 analysis、accepted IR 和
lowering IR 写成三套互不相干的东西：

| 层次 | 表示对象 | 是否进入 IR | 责任 |
| --- | --- | --- | --- |
| scheduled tile tensor IR | `wafer.group` scheduled body 中的 tensor SSA value | 是 | 表达 tile-local tensor dataflow、loop/control-flow、boundary slice；不表达 physical layout |
| target-abstract tile-region IR | `wafer.tile_region` 中的 `wafer.compute` / boundary / data movement op | 是 | 承载选中的 Wafer implementation 和 `WaferLayoutOpInterface`，但尚未绑定 concrete instruction/runtime descriptor |
| layout analysis | `LayoutVariable` / `LayoutEdge` / candidate assignment | 否 | 从 SSA use-def、op interface 和 boundary contract 推导 layout domain、preference、materialization cut |
| boundary co-planning analysis | 相邻 device-side group boundary 的可行 layout summary 和 selected boundary layout | 否 | 在不引入全局 plan attr 的前提下，减少 producer/consumer 边界上的重复 materialization |
| accepted tile-local buffer IR | `!wafer.tile_buffer<shape, mem_layout, memory_space>` 和 `wafer.layout.materialize` | 是 | 记录已接受的 physical layout，以及真实 layout movement 的抽象边 |
| materialization cleanup | canonicalization pattern 和 layout-aware rewrite | 是，通过 rewrite 当前 IR | 删除冗余 materialization；不保存搜索过程 |
| tile buffer storage realization | physical `memref` / flat storage / Wafer descriptor | 是 | 把 `!wafer.tile_buffer` 降成标准 lowering 和目标指令能消费的 storage representation |
| lowered movement IR | `ChannelNorm` / `DechannelNorm` / `GatherScatter` / TDMA / lower-level effects | 是 | 把 abstract materialization 展开为目标相关 movement、sync 和 byte/range 约束 |

constant storage transform 与这条主线并行：它只处理 compile-time `ConstantLike` value 的
backing data/resource 如何按目标 layout 存放，并由 `wafer.load_tile` / storage lowering 直接消费。
它不引入新的 tensor constant 语义，也不把 package format 当成 layout IR 边界。

### 3.1 输入边界

layout planning 有两个恢复层次：

- R3.2b 在 logical `wafer.group` 层运行。它消费 R3.2a `GroupTilingDemand` facts 和当前
  group SSA use-def，构造 transformation-local `LayoutVariable` / `LayoutEdge` graph、op
  layout constraints、layout assignment candidate、materialization cut 和 materialization
  buffer demand。该层只产出 analysis result 和 debug dump，不 rewrite `wafer.group`，不写
  layout attr，也不生成 `wafer.tile_region`。
- R3.4 在 accepted `wafer.tile_region` / instruction-level IR 层运行。它消费 R3.2g accepted plan，把 layout
  assignment 和 materialization cut materialize 成带 `mem_layout` 的 tile buffer type 和
  `wafer.layout.materialize` op。

完整 layout materialization 的输入来自 target-abstract tile-region IR。它由 scheduled
`wafer.group` lowering 而来，但 layout-sensitive tiled op 已经被绑定为正式的
`wafer.compute` / boundary / data movement op。layout materialization pass 不从裸
`linalg.*` 名字推断硬件 layout，而是通过这些 Wafer op 的 `WaferLayoutOpInterface`
查询 hard constraint 和 preference。

输入信息只有这些：

- scheduled body 中的 SSA use-def 和 `scf` loop/control-flow。
- 每个 tile-local tensor value 的 rank、shape、dtype。
- `tensor.extract_slice` / `tensor.insert_slice` 表达的 boundary slice。
- target-abstract `wafer.compute` / boundary / data movement op 通过 layout interface 暴露的
  operand/result layout constraint。
- group planner 在当前 transformation 内部提供的 tile shape、operand demand、temporary/scratch/
  accumulator demand。

这些信息不作为 `wafer.group` attribute 保存；layout planner 直接从 scheduled body 和当前
transformation-local analysis 读取。R3.2b 的 logical-group implementation 在 target-abstract
compute op 尚未 materialize 前，只能使用 `GroupTilingDemand`、structured op semantics、tensor
collective interface 和 target policy 构造保守 layout constraints；不能把单个 workload 或 op
名字序列写成长期协议。

#### 3.1.1 R3.2b Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.1 verifier-legal tensor-level `wafer.group` candidate，以及 R3.2a
  `GroupTilingDemand` analysis result。
- Current stage responsibility:
  从 `GroupTilingDemand` facts、group body SSA use-def、DPS ties、structured op
  semantics 和 tensor collective facts 构造 transformation-local layout planning graph；
  为 tile values / uses / boundary 生成 layout constraints、candidate layout assignment、
  materialization cuts 和 materialization buffer demand。
- Output artifact / IR:
  transformation-local `GroupLayoutPlan` analysis result；debug dump pass 可以打印同一结构。
  本阶段不修改 `wafer.group`，不生成 `wafer.tile_region`，不写 layout attr。
- Downstream consumer:
  R3.2c provisional tile-region candidate lowering、R3.2d Wafer instruction legalization / selection、
  R3.2e SPM placement、R3.2f DDR/resource planning + compute/movement legality analysis，
  以及 R3.2g closed-loop planner。
- User-level driver / named pipeline:
  主线仍由 `wafer-opt --program-pipeline=stablehlo-spmd-to-group` 产生 R3.1 group；
  R3.2b 的局部验证入口是 `wafer-opt --wafer-dump-group-layout-plan`，用于在
  group IR 上 dump analysis 输出。
- Explicit non-goals:
  不选择最终 closed-loop tile shape、不 accept/reject/split group、不分配 SPM、不判断
  DDR pool/range/bandwidth、不 materialize compute/movement/comm op、不生成 package/ABI。
- Completion gate:
  FileCheck 覆盖 linalg matmul/broadcast/elementwise、multi-group、tensor collective、
  materialization demand 和 unsupported tiling failure；program pipeline gate 能在真实
  `stablehlo-spmd-to-group` 输出上重放 layout-plan dump。
```

2026-06-03 收口状态：R3.2b 已按上述 logical-group analysis 边界完成。当前
`GroupLayoutPlan` 消费 R3.2a `GroupTilingDemand` facts 和 group SSA use-def，输出
boundary layout、per-op layout constraints/assignment、materialization cut、group-result
materialization demand 和 failure forwarding；它不 rewrite `wafer.group`，不写 layout attr，
不生成 `wafer.tile_region`。completion gate 覆盖手写 group fixture 和真实
`stablehlo-spmd-to-group` program 输出。

### 3.2 从 Tensor Value 到 Buffer Value

layout planner 先把 target-abstract tile-region 中的 tiled SSA value 映射成 layout variable，
接受 assignment 后再生成带 physical layout 的 tile-local buffer。映射关系来自 SSA use-def，
不是名字约定：

| target-abstract tile-region value | layout planning 中的角色 | accepted `wafer.tile_region` 表达 |
| --- | --- | --- |
| group block argument / external input slice | `#ddr` compact external boundary + load demand | `wafer.load_tile` 产生 tile buffer |
| tiled op operand/result | `LayoutVariable` + op layout constraints | 带 `mem_layout` family 的 `!wafer.tile_buffer` |
| producer-consumer edge | `LayoutEdge`，可能允许 materialization | same-layout use 或 `wafer.layout.materialize` |
| group output slice | writeback boundary | `wafer.store_tile` 或后续 writeback movement |
| loop-carried accumulator / partial result | loop-carried layout variable | entry/yield layout 一致，或 loop 内显式 materialization |

因此 layout 文档中的 `!wafer.tile_buffer` 不是另一套上游 IR；它是 target-abstract tile-region
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
%mm = wafer.compute.gemm %a_tile2, %b_tile2
    : (tensor<64x256xf16>, tensor<256x64xf16>) -> tensor<64x64xf16>
%relu = wafer.compute.elementwise %mm
    : tensor<64x64xf16> -> tensor<64x64xf16>
%r_partial = wafer.compute.reduce %relu
    : tensor<64x64xf16> -> tensor<64xf32>
```

layout planning 对 target-abstract Wafer IR 的理解是：

- `%a_tile` / `%b_tile` 来自 group external inputs，默认 external layout 是 compact。
- `wafer.compute.gemm` 通过 layout interface 要求参与 NE/GEMM 的
  operands/results 使用 aligned family；这里 2D tile 对应 `Cx` family。
- `%relu` 如果 lower 到 flexible CT / elementwise，可以接受 producer 的 selected layout，避免
  多余 conversion。
- `%r_partial` 的 reduce 若走 native reduce，input/output 也要满足 aligned family；若走其它
  implementation，则由该 implementation 的 interface 给出自己的 allowed layouts。
- `%r_tile` 这类 loop-carried partial result 必须在 loop entry/yield 上保持一致 physical layout，
  除非 loop body 内有明确 materialization。

这一层的 analysis 可以抽象成下面的 constraint graph。注意这不是 IR，也不是 attr plan：

```text
Var(%a_tile): #ddr compact external boundary, materializable to aligned for consumer
Var(%b_tile): #ddr compact external boundary, materializable to aligned for consumer
Var(%matmul_tile): constrained by selected matmul implementation
Var(%relu): flexible, preferably inherit producer layout
Var(%r_partial): constrained by selected reduce implementation and loop-carried tie

Edge(%a_tile -> matmul operand 0): materializable
Edge(%b_tile -> matmul operand 1): materializable
Edge(%matmul_tile -> relu operand 0): same-layout preferred
Edge(%relu -> reduce operand 0): materializable only if selected reduce requires different layout
```

被接受后，同一个 value graph 在 `wafer.tile_region` 层可以长成这样：

```mlir
wafer.tile_region {
  %a_t = wafer.load_tile %ga[%m0, 0]
      : tensor<64x256xf16> -> !wafer.tile_buffer<64x256xf16, #tensor, #spm>
  %b_t = wafer.load_tile %gb[0, %n0]
      : tensor<256x64xf16> -> !wafer.tile_buffer<256x64xf16, #tensor, #spm>

  %a_cx = wafer.layout.materialize %a_t
      : !wafer.tile_buffer<64x256xf16, #tensor, #spm>
     -> !wafer.tile_buffer<64x256xf16, #cx, #spm>
  %b_cx = wafer.layout.materialize %b_t
      : !wafer.tile_buffer<256x64xf16, #tensor, #spm>
     -> !wafer.tile_buffer<256x64xf16, #cx, #spm>

  %mm = wafer.compute.gemm %a_cx, %b_cx
      : (!wafer.tile_buffer<64x256xf16, #cx, #spm>,
         !wafer.tile_buffer<256x64xf16, #cx, #spm>)
     -> !wafer.tile_buffer<64x64xf16, #cx, #spm>

  %relu = wafer.compute.elementwise %mm
      : !wafer.tile_buffer<64x64xf16, #cx, #spm>
     -> !wafer.tile_buffer<64x64xf16, #cx, #spm>

  %r_partial = wafer.compute.reduce %relu
      : !wafer.tile_buffer<64x64xf16, #cx, #spm>
     -> !wafer.tile_buffer<64xf32, #tensor, #spm>

  %relu_tensor = wafer.layout.materialize %relu
      : !wafer.tile_buffer<64x64xf16, #cx, #spm>
     -> !wafer.tile_buffer<64x64xf16, #tensor, #spm>

  wafer.store_tile %relu_tensor, %go0[%m0, %n0]
      : !wafer.tile_buffer<64x64xf16, #tensor, #spm> -> tensor<128x128xf16>
  wafer.store_tile %r_partial, %go1[%m0]
      : !wafer.tile_buffer<64xf32, #tensor, #spm> -> tensor<128xf32>
}
```

这只是 layout 层的示例形态：具体 compute op、load/store op 名和最终 movement lowering 属于
`wafer.tile_region` / compute / memory dialect 设计。本文固定的是 value mapping、layout
assignment、materialization edge 和 verifier 责任。示例里两个 output 的 shape 不同，layout
planner 只按各自 SSA value、boundary contract 和 consumer/producer constraint 处理；不要求它们
共享一个 root domain，也不编造二者之间的 shape relation。示例里的 tile shape、op 名和某个
implementation 选择都只是为了说明 IR 如何流动，不是 layout 架构边界。

## 4. Layout 分类

layout 和 storage encoding 概念必须分开：

- semantic layout：tensor 维度业务含义，例如 NHWC、HWOI/HWIO、GEMM matrix。
- physical layout / `mem_layout`：device buffer 的实际组织，例如 compact `Tensor/NTensor`、
  aligned `Cx/NCx`。
- constant storage encoding：compile-time constant 的 backing data/resource 在 storage lowering
  或最终 package 中的字节排布。它可以保持原始 compact 排布，也可以由显式 constant storage
  transform pass 生成目标相关排布。
- external layout contract：host/runtime ABI 上 visible 的 tensor layout。

V0 规则：

- host-visible dynamic input/output 默认 `#ddr` compact external tensor layout。
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

   layout conversion 先用统一的 `wafer.layout.materialize` 表示，再 lower 到 `GatherScatter`、
   TDMA 或 wrapper path。same-layout materialization 必须 canonicalize 掉。

compute/movement op 的具体 interface、op family 和 issue/drain 边界见
`tasks/2026-05-25-wafer-compute-dialect-design.md`。communication op 不改变 tensor semantic
layout，p2p transfer 默认是 byte-preserving；若 collective/p2p schedule 消费或产生 tile buffer，
它必须按 `tasks/2026-05-25-wafer-communication-dialect-design.md` 暴露 buffer、layout relation、
token/effect 和 staging demand。layout planner 只消费这些接口事实，不复制 compute/comm 的
lowering 计划。

### 5.1 V0 IR Contract

本节只定义第 3 节生命周期中的 accepted tile-local buffer IR 和 constant storage transform 合同。
`LayoutVariable` / `LayoutEdge` 不属于本节 IR；lowered `ChannelNorm` / `GatherScatter` / TDMA
也不属于本节 IR。语法是设计草图，不要求现在就完全等同 ODS/parser 打印格式；真正要固定的是
语义归属、verifier 责任和 lowering 边界。

#### 5.1.1 Attributes / Types

`WaferMemLayoutAttr` 只表达 device buffer 的 physical layout family。它只能出现在
`wafer.tile_region` 及其下游 buffer / memref / descriptor 上，不进入 `wafer.group`。

`WaferMemorySpaceAttr` 统一表达 addressable storage space。不要把 SPM、DDR、host-visible DDR
或 package/runtime storage 处理成几套互不相干的 memory-space 语义。V0 至少需要：

```mlir
#tensor = #wafer.mem_layout<tensor>
#ntensor = #wafer.mem_layout<ntensor>
#cx = #wafer.mem_layout<cx>
#ncx = #wafer.mem_layout<ncx>
#spm = #wafer.memory_space<spm>
#ddr = #wafer.memory_space<ddr>
```

语义约定：

- `#spm` 是 tile-local SRAM，容量、lifetime、range 和 reuse 由 SPM bufferization 负责。
- `#ddr` 是 device/global DDR 或 host-visible device buffer 的目标侧 address space，allocation /
  BO pool / visibility / package ownership 由 DDR resource planning 和 launch/runtime/package 层负责。
- `mem_layout` 和 `memory_space` 正交。DDR buffer 可以是 compact，也可以在 constant storage
  transform 或 device-side materialization 后带目标相关 physical layout；SPM buffer 同样通过
  `mem_layout` 表达 compact 或 aligned family。
- compile-time constant 在 semantic tensor IR 中仍是 `ConstantLike` value。只有在
  `wafer.load_tile` / storage lowering 需要 addressable device storage 时，才产生明确
  memory space、descriptor 或 runtime/package metadata。

DDR 的 pool/domain、workspace BO、external binding、constant residency、capacity 和 bandwidth
不属于 `WaferMemLayoutAttr`，见
`tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`。

V0 不把以下派生结果写进 `WaferMemLayoutAttr`：

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
PhysicalLayoutInfo {
  mem_layout_family   // Tensor / NTensor / Cx / NCx
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
`NCx` physical layout family。

`Cx/C0/aligned_C/batch_mem_elems/storage_bytes` 按硬件文档中的 `get_CxC0` /
`common_tensor_info_generate_i64` 口径计算：INT8/UINT8 full block 是 128，其它 dtype full
block 是 64；tail 是否保留由半块阈值决定；随后继续计入 256B bank padding。

V0 推荐引入 Wafer layout-aware buffer type，或者等价的 target buffer abstraction。不要把
`Cx/NCx` 直接伪装成 MLIR affine memref layout encoding；它不是简单 affine map，而是依赖 dtype、
tail/fold 和 256B padding 的 target storage rule。

```mlir
// 设计草图：具体 printer 可按 MLIR type/parser 约束调整。
!wafer.tile_buffer<64x64xf16, #tensor, #spm>
!wafer.tile_buffer<64x64xf16, #cx, #spm>
!wafer.tile_buffer<64x64xf16, #tensor, #ddr>
```

现有示例大多使用 `#spm`，因为 tile-local compute operand 默认落在 SPM。DDR 不是另一套语义：
如果 DDR buffer 在 `wafer.tile_region` 中以同一 buffer abstraction 出现，就用 `#ddr`；
如果它已经 placed 成 memref/descriptor，也必须携带同一 `WaferMemorySpaceAttr`。

如果某个 lowering 阶段必须落到 `memref`，那里的 memref layout / offset / stride / byte range
应由 `PhysicalLayoutInfo` 展开得到，属于更低层 lowering 结果，不是 V0 layout planning 的
canonical IR contract。

`!wafer.tile_buffer` 不是替代 `memref` 的长期底座，也不应保留到 LLVM lowering 前才处理。它在
accepted layout IR 和 SPM bufferization 之间提供一个 layout-aware buffer abstraction；随后必须由
tile buffer storage realization 转成标准 lowering 能消费的 storage：

- compact `Tensor/NTensor`：优先 lower 成对应 memory space 的 strided/identity `memref`，复用
  MLIR memref/LLVM lowering。`#spm` 由 SPM allocator 给 range；`#ddr` 由 DDR resource plan /
  runtime BO binding 给 addressability 和 ownership。
- aligned `Cx/NCx`：由 `PhysicalLayoutInfo` 展开成 physical-shape `memref`、flat storage `memref`，
  或 descriptor + backing storage；descriptor 只承载目标指令需要的 address/range/stride/layout
  facts。
- 任何 descriptor 最终都必须 lower 成 LLVM dialect 可表达的 struct / pointer / integer operands；
  不允许让下游靠 `tile_buffer` 类型本身解释 storage。

V0 不定义额外的 constant storage encoding attr。constant 的 storage encoding 不作为独立
layout attr 在 pipeline 中传播；它由当前 `ConstantLike` value、consumer `wafer.load_tile`
result layout、storage lowering 规则和最终 package emission 共同决定。若后续确实需要把
encoding 固化进 IR，也应落在 storage-level constant/load op 的类型或 verifier 可检查的
attribute 上。

#### 5.1.2 `wafer.layout.materialize`

`wafer.layout.materialize` 是 device-side real data movement 的抽象 op。它只表示从一个
physical layout 生成另一个 physical layout；不改变 tensor 数学语义。

```mlir
%dst = wafer.layout.materialize %src
    : !wafer.tile_buffer<64x64xf16, #tensor, #spm>
   -> !wafer.tile_buffer<64x64xf16, #cx, #spm>
```

ODS 草图：

```tablegen
def Wafer_LayoutMaterializeOp : Wafer_Op<"layout.materialize", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<WaferLayoutMaterializationOpInterface>
  ]> {
  let arguments = (ins Wafer_TileBufferType:$source);
  let results = (outs Wafer_TileBufferType:$result);
}
```

Verifier 至少检查：

- source/result rank、shape、element type 的 semantic extent 一致。
- source/result memory space 合法；V0 只允许 tile-local device buffer。
- source/result `mem_layout` 不同；相同 layout 的 materialize 必须 canonicalize 掉。
- 对应 conversion path 存在，且由 shape/dtype/semantic layout/target policy 推出的
  `PhysicalLayoutInfo` 合法。
- op effects 正确表达 read source / write result；若 lowering 变成 async movement，后续 sync
  verifier 必须能看到 wait/drain。

`wafer.layout.materialize` 不保存 cost、失败原因、planner 选择理由或备选方案。debug dump 可以
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
按 consumer 需要的 `mem_layout` 生成新的 backing data/resource，或在 `wafer.load_tile` lowering
时直接生成对应 storage。

constant 不是 `wafer.group` 的 external input，但它仍然是 tile execution 的 source。进入
`wafer.tile_region` 后，constant use 必须变成显式 `wafer.load_tile` 或等价 load source：

- tile shape 和 consumer indexing 决定 constant 的 tile slice / access range。
- `wafer.load_tile` result type 决定 load 后 tile buffer 的 `mem_layout` 和 memory space。
- DDR resource planner 必须看到 constant 的 read-only source demand，包括 resident 或 streaming
  策略、storage size、alignment、lifetime、bandwidth 和 per-tile access range。
- 若原始 compact backing data 不能直接满足 result layout，planner 可以选择 compile-time storage
  transform，也可以保留 raw load 并在 consumer edge 插入 `wafer.layout.materialize`。选择必须重新跑
  SPM 和 DDR feasibility。

并非所有 constant 都需要 DDR residency：

- scalar、splat、small attribute-like constant 如果能被 compute op 作为 immediate、attribute、fill
  pattern 或 folded value 消费，不生成 `wafer.load_tile`，也不产生 DDR demand。
- 一旦 constant 以 tensor data source 参与 tile compute，或者需要按 tile slice 读入 SPM，它就必须
  通过 `wafer.load_tile` / equivalent load source 表达，不能靠隐式 package side channel。
- 判断 immediate/fill/load 是 op lowering 的 legality 选择，必须由 op verifier 或 lowering pattern
  明确支持；不能靠 constant 名字、大小阈值的 ad hoc matcher。

weight 切分不是新的 Wafer constant 语义。它只是某个 `ConstantLike` tensor 被 tiled consumer
按 logical slice 使用：

- op tiling / `WaferTilingInterface` 先把 consumer tile 映射到 operand tile slice。GEMM weight、
  conv weight、scale/bias 这类常量都走同一条规则，不能靠名字或 “weight” 特判。
- `wafer.load_tile` 必须显式携带这个 source slice 或等价 index operands；slice 可以包含 reduction
  dimension、channel/block dimension，也可以是 op interface 推出的 hidden/internal dimension。
- storage transform 可以选择 whole-constant backing，也可以选择 chunked backing。chunk key 来自
  `(constant SSA value, logical slice/chunk shape, result mem_layout, target policy)`，不是新的 IR 名词。
- chunked backing 只能覆盖已有 `wafer.load_tile` slice 的 union/coalescing；不能为了 packing 引入
  额外 compute tiling 或 reduction split。若需要改变 K/internal split，必须回到 group/op tiling planner。
- 如果 chunk 数量、package size、DDR residency 或 bandwidth 失控，应回退到 raw backing +
  materialization，或返回 group planner 调整 tile shape / group boundary。

如果输入来自 compile-time constant，并且 accepted assignment 选择 constant storage transform，
`wafer.load_tile` 仍然消费当前 IR 里的 constant value；它的 result type 表达下游看到的
physical layout：

```mlir
%w = arith.constant dense_resource<W_cx> : tensor<256x128xf16>

%wt = wafer.load_tile %w[%tile]
    : tensor<256x128xf16> -> !wafer.tile_buffer<256x64xf16, #cx, #spm>
```

如果 planner 没有选择 constant storage transform，`wafer.load_tile` 从 normalized
`ConstantLike` value 的原始 compact backing data 读取，并在必要 edge 上插入
`wafer.layout.materialize`。这个选择是 load/storage lowering 的事实，不改变 constant 的数学语义。

注意：在 semantic tensor IR 层，不能在保持同一个 logical tensor contract 的同时无标记地把
payload 改成另一种 physical byte order。packing 必须发生在 layout 已经被 `wafer.load_tile`
result type、storage-level type 或等价 verifier contract 约束之后；否则就是把数学值和存储表示混在一起。

### 5.2 Interfaces

V0 需要一个 op layout 接口；constant storage transform 是 pass 行为，不单独定义 Wafer op
interface。

R1.2 先落地 accepted-layout 层的可查询合同：`WaferLayoutOpInterface` 由 layout-sensitive
compute/movement op 实现，返回当前 IR 类型已经表达的 exact operand/result layout requirement，
并提供 verifier 可调用的组合检查。`wafer.layout.materialize` 不实现该接口，它实现单独的
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
| `wafer.load_tile` / 等价 boundary load op | 必须 | 暴露 `#ddr` compact external boundary、constant source 和 tile buffer result 的 allowed/preferred layout |
| `wafer.store_tile` / 等价 boundary store op | 必须 | 暴露 `#ddr` host-visible compact writeback、device-side group boundary 和 store input layout 约束 |
| `wafer.compute.gemm` / NE-style matmul op | 必须 | 通常是 aligned-only consumer/producer；决定 operand/result 是否必须是 `Cx/NCx` family |
| `wafer.compute.reduce`、pool、unpool | 必须 | 这些 op 有硬件 layout legality，不能靠通用 passthrough 规则猜 |
| `wafer.compute.elementwise` / CT-style flexible op | 必须或提供默认 flexible trait | 若 op 只是 shape-preserving passthrough，可复用默认 flexible 规则；若受 dtype/range/wrapper 限制，必须实现接口 |
| `wafer.data_move.*` / target-abstract movement op | 必须 | DMA、load/store、local movement 如果限制 physical layout、range 或 stride，需要把限制暴露给 planner/verifier |
| `wafer.comm.*` 中消费/产生 tile buffer 的 op | 必须或提供等价 relation | p2p comm 默认 byte-preserving，但仍要暴露 source/destination buffer、byte count、token/effect 和 staging demand；collective-level op 展开前只表达 semantic，展开后由 p2p op 验证 |
| `wafer.layout.materialize` | 不实现这个接口；实现 `WaferLayoutMaterializationOpInterface` | 它表示 layout conversion edge，本身由 source/result type 和 materialization interface 验证 |
| `wafer.group`、`wafer.tile_region`、`scf.*` | 不实现 | 它们提供 region/control-flow/边界结构；layout 约束来自 region 内 value 和 op interface |
| `linalg.*` / upstream compute op | 不直接实现 | 进入 Wafer planning 后由选中的 Wafer lowerable implementation 或 adapter 提供 layout contract，不修改 upstream dialect |
| `arith.constant` / `stablehlo.constant` / generic `ConstantLike` op | 不实现 | constant 不消费 tile buffer；transform pass 读取其 value/resource 并在 storage lowering 附近改写 |

constant storage transform 的 implementer 是 pass / pattern，不是 op interface：

- 输入必须是已经 normalized 的 `ConstantLike` tensor value；若来自 `stablehlo.constant`，必须先经过
  constant normalization。
- transform 从当前 IR 的 use-def、accepted tile shape、op tiling interface 推出的 operand slice、
  `wafer.load_tile` tile slice、result `mem_layout`、target policy 和 `PhysicalLayoutInfo`
  推导目标 storage order。
- transform 直接替换 constant backing data/resource，或在 lowering `wafer.load_tile` 时生成
  packed storage；不创建新的旁路 IR 事实源。
- transform 不处理已经被合法 fold 成 immediate / attribute / fill pattern 的 constants；这类
  constants 不进入 DDR demand。
- dynamic host input/output 不能走这条路径；它们仍受 `#ddr` compact external ABI 和 runtime binding
  约束。

### 5.3 Pass Contract

pass 名是实现组织，不是架构边界；边界仍以 IR contract 和 verifier 为准。V0 可以按以下 pass
组织：

1. `wafer-layout-materialize`

   运行位置：scheduled tile tensor IR 已经形成、tile shape 已被当前 group candidate 接受、
   layout-sensitive tiled op 已经被选成 target-abstract `wafer.compute` / boundary / data movement
   op、下游 SPM allocator 可用、最终硬件 movement 还没 lower 之前。

   输入：

   - target-abstract `wafer.tile_region` SSA body。
   - op 的 `WaferLayoutOpInterface`。
   - `wafer.comm` p2p op 的 byte-preserving layout relation、token/effect 和 staging demand。
   - `#ddr` compact external boundary。
   - normalized `ConstantLike` tensor value 及其 backing data/resource。
   - target policy、SPM allocator 和 DDR resource planner。

   行为：

   - 从当前 IR 构造 `LayoutVariable` / `LayoutEdge` analysis；analysis 不写入 IR。
   - 生成 group-local layout candidates 和 boundary summary。
   - 对 device-side group-to-group edge 做 bounded co-planning，并把 accepted boundary layout 作为
     当前 transformation 的 hard constraint 传回本地 planner。
   - 选择最终 physical layout assignment 和 materialization edge。
   - 对候选 assignment 调用 SPM allocator 和 DDR resource planner；改变 boundary
     layout 或 materialization cut 后必须重新验证。
   - 对 accepted assignment rewrite IR：生成或更新 tile-local buffer type 的 `mem_layout`，插入
     `wafer.layout.materialize`，并把 compile-time constant 的 storage transform 机会保留在
     `wafer.load_tile` use 上。
   - 对 rejected candidate 只发 diagnostic，不写入 IR。

   输出：

   - 带 `mem_layout` 的 tile-local buffer values。
   - 明确的 `wafer.layout.materialize` op。
   - `wafer.load_tile` 对 constant source 的明确 use-def 关系，供后续 storage transform / lowering 使用。

2. `wafer-layout-materialize-cleanup`

   运行位置：accepted layout IR 已经形成、lower-level movement 尚未生成之前。它也可以实现为
   `wafer-layout-materialize` 末尾的一组 canonicalization pattern；是否拆 pass 不影响 IR contract。

   行为：

   - 删除 same-layout、dead、inverse-pair materialization。
   - 对 flexible op 做 layout-aware rewrite，减少两侧 conversion。
   - 合并重复 conversion 或移动 materialization cut 前，重新运行 SPM allocation。
   - 不读取旧 planner side table，不创建新的 layout plan attr。

3. `wafer-constant-storage-transform`

   运行位置：layout assignment 已接受之后，`wafer.load_tile` lowering 或 package emission 之前。

   输入：

   - normalized `ConstantLike` tensor value。
   - `wafer.load_tile` use、result `mem_layout`、tile slice 和从 op tiling interface 推出的 operand
     slice relation。
   - 原始 constant data。
   - `PhysicalLayoutInfo` calculator 和 target policy。

   行为：

   - 若 constant 的 consumer layout 可以在编译期满足，按 accepted tile shape / tile slice 生成
     对应 storage order 的 backing data/resource，或把 `wafer.load_tile` lower 成使用 packed
     storage 的形式。
   - 对 weight / large constant，允许 whole-constant transform、按 tile-slice chunk transform、
     或按多个 load slice 的 union/coalesced chunk transform；chunk 只服务已有 `wafer.load_tile`
     use，不改变 compute tile。
   - 为 constant source 生成 read-only DDR demand：resident constant、streaming load 或 staging
     由 DDR planner 根据 capacity、range、alignment、bandwidth 和 reuse 决定。
   - 若同一个 constant 被多个 incompatible consumers 使用，可以 clone / specialize constant use，
     或保留 runtime `wafer.layout.materialize`；选择由 cost、SPM 和 DDR feasibility 决定。
   - 最终 package emission 只序列化当前 IR 已经选定的 constant storage，不创建新的 IR 事实源。

   不负责：

   - 不选择 tile shape。
   - 不决定 runtime materialization cut。
   - 不修改 tensor semantic layout。

4. `wafer-realize-tile-buffer-storage`

   运行位置：accepted layout IR 和 cleanup 完成之后，lower-level movement/compute op 需要真实 storage
   之前。它可以作为 `wafer-spm-bufferize` 的后半段，也可以拆成独立 conversion pass；边界是
   `!wafer.tile_buffer` 在这里消失。

   输入：

   - tile-local buffer 的 logical shape、dtype、memory space 和 `mem_layout`。
   - `PhysicalLayoutInfo` calculator。
   - SPM allocation / range / lifetime result。
   - target descriptor policy。

   行为：

   - compact layout 转成普通 SPM `memref`。
   - `Cx/NCx` 转成 physical-shape `memref`、flat storage `memref` 或 explicit Wafer descriptor。
   - 给 lower-level movement/compute op 绑定 address/range/stride/layout descriptor。
   - 删除或替换所有 `!wafer.tile_buffer` typed values。

   不负责：

   - 不重新选择 physical layout。
   - 不重新插入或移动 `wafer.layout.materialize`。
   - 不绕过标准 memref/LLVM lowering；能复用标准 conversion 的 storage 必须复用。

5. `wafer-lower-layout-materialize`

   运行位置：tile buffer storage realization 已经给出真实 storage / descriptor 之后，lower-level
   movement/compute dialect 之前或过程中。

   输入：

   - `wafer.layout.materialize` op。
   - source/result physical layout、shape、dtype、memory space。
   - SPM allocation / range-end 信息。
   - target lowering table。

   行为：

   - 选择 `ChannelNorm`、`DechannelNorm`、`GatherScatter`、TDMA 或 wrapper path。
   - 生成 lower-level movement op 和必要 wait/drain effect。
   - 删除已经 lower 的 `wafer.layout.materialize`。

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
- materialization candidate 产生的新 value。

约束来源：

- op layout interface 的 allowed / preferred layouts。
- dtype、rank、shape、bool bitpack、C0 tail/fold、padding、range-end 规则。
- `#ddr` compact external ABI。
- compile-time constant source 的 current backing data/resource，以及 `wafer.load_tile` result layout。
- producer-consumer edge 是否允许插入 real movement。

V0 只需要区分 hard constraint 和 preference：

- hard constraint 失败就是 layout infeasible，不能靠 cost model 覆盖。
- preference 只影响 assignment 和 tie-break，不是 verifier 合同。
- producer 和 consumer 的 selected layout 冲突时，只有 materializable edge 可以插入
  `wafer.layout.materialize`；否则必须回到 group planner 拆 group、换 tile 或请求 op-local
  implementation。

accepted assignment 只通过 rewrite 后的 tile-local buffer type、`wafer.layout.materialize`、
`wafer.load_tile` 的 use-def 和 result type 体现。candidate domain、cost trace、备选 cut、失败原因
都属于 diagnostic / debug dump，不能成为下游 pass 依赖的 IR 事实。

### 6.1 Materialization Cut Placement Algorithm

materialization cut 的核心问题是同时决定两件事：

```text
layout assignment:
  每个 tile-local SSA value / alias group 选择一个 physical layout label

cut placement:
  如果 producer value 和 consumer operand 的 selected layout 不一致，
  在哪条 materializable edge 上插入 wafer.layout.materialize
```

V0 不把这个问题写成“遇到某类 op 就插 conversion”。通用算法是 hard constraint propagation、
小规模 graph labeling、局部 cut 优化、SPM allocation 和 DDR resource acceptance 的组合。

#### 6.1.1 Hard Constraint Propagation

先从当前 IR、type、op interface 和 boundary contract 推导 hard domain：

- aligned-only op 收窄 operand/result domain，例如 NE/GEMM、native reduce、pool/unpool。
- host-visible dynamic input/output 固定为 `#ddr` compact external layout。
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
   package size、SPM allocation 和 DDR resource planning 都合适时才接受。

这个 assignment 只产生 candidate，不代表已经可 lower。

#### 6.1.4 Cut Placement Rules

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
  byte-preserving transfer；否则在明确 edge 上插 `wafer.layout.materialize`。

任何 cut 移动只是在 candidate 上发生。被接受前不能写入 `wafer.group` 或全局 attr。

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
- `wafer.layout.materialize` 都有 lowerable conversion path。
- SPM allocation 通过，包含 materialization temp、communication staging、loop-carried value、
  async lifetime 和 range/end-address validation。
- DDR demand / workspace / external binding / bandwidth summary 可由 DDR resource planner 接受。
- cleanup 后仍能通过 layout verifier、SPM allocation 和 DDR resource planning。

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
通过同一套 verifier、SPM allocation 和 DDR resource acceptance。

## 7. Boundary Contract

dynamic host input/output：

- 固定为 `#ddr` compact external layout。
- 不要求用户提供 `Cx/NCx`。

compile-time constants：

- 默认保留原始 compact backing data/resource。
- scalar、splat 或 small constants 可以由 op lowering fold 成 immediate / attribute / fill，不产生
  DDR demand；前提是目标 op verifier 明确允许。
- 若 layout/materialization planning 选择 compile-time storage transform，必须由显式 transform
  pass 直接改写 constant backing data/resource，或在 lowering `wafer.load_tile` 时生成对应 storage。
- constant 虽然不是 group external input，但每个 tile-region use 都必须通过显式 load source
  表达，并参与 DDR demand / tiling / bandwidth 计算。
- 若同一个 constant 被多个 incompatible consumers 共享，V0 可以 clone / specialize constant use，
  也可以在 consumer edge 做 runtime materialization；选择由 cost、package size、SPM allocation
  和 DDR resource planning 决定。
- constant storage transform 只是把 conversion 提前到编译期执行，不是新的上层 IR 语义。

load/store：

- load/store 根据 source/destination memory space 和 layout assignment 选择 movement lowering。
  典型 dynamic boundary 是 `#ddr` compact buffer 与 `#spm` tile-local buffer 之间的 RDMA/WDMA。
- 它们不是 layout 的根本来源。

group output：

- device-side group-to-group value 可以保持 selected physical layout。
- host-visible output 在 writeback 前必须回到 `#ddr` compact external layout。

## 8. 跨 Group Boundary Co-Planning

跨 group co-planning 要解决的是这种模式：

```text
group A produces value in layout X
  -> materialize X -> compact at A output
  -> store/load device-side boundary
  -> materialize compact -> Y at group B input
```

如果这个 boundary 不是 host-visible、不逃逸到未知 runtime ABI、且下游 verifier 能用 boundary
value 的 type/effect 检查 memory space 和 `mem_layout`，那它可以保持 `X` 或 `Y`，不必强制回到
compact。这个选择仍然不能写成 `wafer.group` 上的全局 layout plan；它只能作为当前 planning 的
analysis，最终通过 accepted boundary buffer value 的 type 和必要的 `wafer.layout.materialize`
进入 IR。

V0 做 bounded adjacent co-planning，不做 full-program layout solve：

1. 本地 summary

   对每个 group candidate，layout planner 在 rewrite 前生成 boundary summary：

   ```text
   BoundaryLayoutSummary {
     boundary_value
     direction        // producer result or consumer operand
     allowed_layouts
     preferred_layouts
     local_cost(layout)
     spm_allocation(layout)
     ddr_resource_plan(layout)
   }
   ```

   summary 是 analysis，不进入 IR。`local_cost` 必须包含 group 内 materialization bytes、peak SPM
   变化、DDR workspace / bandwidth pressure 和 writeback/load movement；`spm_allocation` 必须来自
   同一个 SPM allocator，`ddr_resource_plan` 必须来自 DDR resource planner。

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

   selected boundary layout 作为当前 transformation 的 hard boundary constraint 传回 producer 和
   consumer 的本地 planner。若任一 group 的 SPM allocation 或 DDR resource planning 失败，回退到下一个
   boundary candidate；
   frontier 耗尽时，退回本地规划并保留显式 boundary materialization。

5. Rewrite

   只有 accepted result 写入 IR：producer boundary value、consumer boundary argument/load value
   的 physical layout 一致，或者 boundary edge 上有明确 `wafer.layout.materialize`。co-planning
   summary、备选 layout 和失败原因不写入 IR。

这个机制的关键不是引入全局最优，而是允许相邻 group 在 device-side boundary 上共享一个 verifier
可见的 physical layout。它覆盖常见 repeated materialization 成本，同时保持每个 group 的 local
legality、SPM allocation 和 DDR resource planning 可独立重算。

## 9. Materialization Cleanup

初始 layout materialization 可能保守插入多余 conversion。accepted IR 形成后，必须运行
layout-aware cleanup；它是普通 IR rewrite / canonicalization，不是重新解释旧 planner state。

可以无条件做的局部 fold：

- `materialize A -> A` 直接删除。
- `materialize(A -> B)` 的结果无 use 时删除。
- `materialize(A -> B -> A)` 且中间 `B` value 没有其它 use 时，把最终 uses 改回原始 `A` value。
- 同一 source、同一 destination layout、同一 shape/dtype 的 materialization，如果共享结果不会引入
  新的 lifetime 冲突，可以合并；否则不能只按文本相同做 CSE。

需要重新验证 layout interface、SPM allocation 和 DDR resource planning 的 rewrite：

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
  都必须重新运行 SPM allocation 和 DDR resource planning；只删除 dead/same-layout
  conversion 且缩短 lifetime 的 fold 可以直接应用。
- cleanup 后仍由 verifier 检查 op layout contract、loop-carried entry/yield layout、
  `wafer.load_tile` result layout、constant source 可 lower 性和 lowerable conversion path。

## 10. V0 Algorithm

V0 不做全局最优，但不能只做一次贪心选择。主路径是 deterministic greedy assignment，加一个
很小的 bounded alternative frontier，用来处理 layout 和 SPM 强耦合的失败 case。

1. 建 constraint graph

   为每个 tile-local value / alias group 建 `LayoutVariable`。host input/output 固定 compact；
   compile-time constant 通过 `wafer.load_tile` use 参与图，tile slice 由 accepted tile shape、
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

4. 插 materialization candidate

   只在 producer/consumer selected layout 不一致的 edge 上插 candidate。cut placement 使用第
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

6. 生成 resource-planned local candidates

   对初始 assignment 和 bounded alternatives 构造真实 materialization demand，运行 SPM allocation
   和 DDR resource planning，形成 group-local candidate / boundary summary。SPM 或 DDR
   失败时，layout planner 只做有限调整：

   - 移动 materialization cut。
   - 改用 specialized constant storage backing。
   - 让 flexible op 接受另一个已有 layout。

   如果仍没有本地可行 candidate，返回 group planner 调整 tile shape、internal split 或 group
   boundary。

7. 跨 group boundary co-planning

   对 device-side group-to-group edge，收集相邻 group 的 boundary summary，在 bounded frontier 内选择
   boundary layout，并把 accepted boundary layout 作为 hard constraint 传回相关 group 的本地 planner。
   如果 co-planning 失败，保留显式 boundary materialization，不把失败计划写入 IR。

8. Rewrite

   只对 accepted assignment 写 IR：`wafer.tile_region` 中出现 physical `mem_layout` 和
   `wafer.layout.materialize`。具体 `ChannelNorm` / `GatherScatter` / TDMA 等 lower-level movement
   由后续 lowering pass 生成。

9. Cleanup

   在 accepted IR 上运行 layout materialization cleanup。无条件 fold 直接删除冗余 op；改变 lifetime
   或 boundary layout 的 rewrite 必须重新通过 SPM allocation 和 DDR resource planning。

10. Storage realization handoff

   cleanup 后交给 `wafer-realize-tile-buffer-storage` / `wafer-spm-bufferize` 后半段，把
   `!wafer.tile_buffer` 转成 physical `memref`、flat storage 或 descriptor。layout planner 不直接
   生成 LLVM ABI，但必须保证 accepted layout 都能被这个 conversion 合法实现。

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

- scheduled tile tensor IR 表达 tensor dataflow，不表达 physical layout。
- `LayoutVariable` / `LayoutEdge` 是 pass-local analysis，不表达成 op/type/attr。
- group boundary summary / co-planning frontier 是 pass-local analysis，不表达成 op/type/attr。
- accepted `wafer.tile_region` / SPM bufferization 层表达：
  - tile-local buffer 的 memory space 和 `mem_layout`。
  - `wafer.layout.materialize` 或等价 explicit data movement op。
  - `wafer.load_tile` 对 `ConstantLike` source 的 use-def，以及 result `mem_layout`。
  - op verifier 可检查的 layout contract。
- tile buffer storage realization 层删除 `!wafer.tile_buffer`，把它转换成 physical `memref`、flat
  storage `memref` 或 explicit descriptor。compact layout 应尽量复用标准 memref lowering；
  Cx/NCx 的 descriptor 只承载 lower-level instruction 必需的 storage facts。
- lower-level movement IR 表达具体 instruction/wrapper path、sync/effect 和 byte/range 约束。

不要维护全局 `layout_plan` attr，也不要把 analysis graph 序列化成 side table 给后续 pass 使用。
失败的 layout candidate、cost trace、materialization 尝试都只是 analysis。

## 13. Verifier

layout verifier 至少检查：

- `WaferMemLayoutAttr` 只包含 `Tensor/NTensor/Cx/NCx` family，不包含 `axis/C0/aligned_C/storage_bytes`
  这类可推导字段。
- aligned-only op 的 operand/result physical layout 合法。
- 通过统一 calculator 推出的 `PhysicalLayoutInfo` 与 dtype、rank、shape、C0 tail/fold、bool
  bitpack 一致。
- materialization source/destination layout 不同，且 conversion path 可 lower。
- host-visible dynamic input/output 满足 `#ddr` compact external layout contract。
- compile-time constant source 可以被 `wafer.load_tile` lowering 成 result layout 要求的 storage。
- loop-carried value 的 entry/yield layout 一致，除非 loop body 中有明确 materialization。
- device-side group boundary 的 producer value 和 consumer value physical layout 一致；如果不一致，
  boundary edge 上必须有明确且可 lower 的 `wafer.layout.materialize`。

lowered movement verifier 另行检查具体 movement 的 byte size、stride、range-end、sync/effect 与
physical layout 一致；这些事实不反写进 `wafer.layout.materialize`。

`wafer.comm` verifier 另行检查 peer、route、token/wait、DTE/FSM resource 和 fixed-size unicast
约束。layout verifier 只要求 communication source/destination physical layout relation 明确：
如果传输前后 layout 相同，可以直接 byte-preserving transfer；如果 layout 不同，必须有显式
`wafer.layout.materialize` 或其它可验证 movement edge。

## 14. 后续扩展

这些机制有价值，但不进入 V0 主路径。进入条件必须明确，不能因为“看起来更强”就提前实现：

- beam search：当局部贪心在 LLM 主线 case 上频繁把 large tensor 反复 materialize，且换 cut 可明显
  降低 SPM peak 或 movement bytes 时引入。
- ILP / 全局最优 assignment：只在小型子图离线 tuning 或 debug mode 中考虑，不作为默认 compiler
  path。
- 跨多个 group 的全局 layout optimization：当 bounded adjacent co-planning 仍无法消除主要 repeated
  materialization，且 profile 显示跨长链 layout 决策成为主成本时引入。
- 多版本 runtime materialization cache：只有同一 value 在多个 incompatible consumers 间反复转换，
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
