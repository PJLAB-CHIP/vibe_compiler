# Wafer Compute and Movement Dialect Design

日期：2026-05-25

状态：设计草案；2026-05-25 边界收口；2026-06-04 对齐 instruction-level Wafer IR 先于 SPM placement；
2026-06-05 对齐 memref-backed Wafer memory attr 合同

本文定义 Wafer 后端中 target-abstract compute / movement IR 的边界。它连接
`wafer.group` 产生的 tile-local tensor program、layout materialization / SPM bufferization，
以及后续 instruction / C ABI lowering。

本文中的 `wafer.tile.*` compute 是正式 IR contract。它表达“这个 tile-local op 已经选择了某类
Wafer 目标实现族，并能提供 layout、effect 和 instruction family legality”。它仍然不表达 raw packet
bitfield、SPM physical offset、worker window、runtime launch 或 host ABI。最终 SPM memref demand
不是 target-abstract op 自身的属性，而是 R3.2d 产出的 instruction-level `wafer.instr.*`
over unplaced Wafer-tagged memref IR 的结果。
instruction-level IR 的具体 op/type/interface 合同见
`tasks/2026-06-05-wafer-instruction-ir-design.md`；本文不重复维护 `wafer.instr` op 列表。

本文只负责 target-abstract compute/movement op 的语义、interface、effect、issue/drain 和
lowering legality。它不重新做 group formation、tile search、layout assignment、SPM/DDR
allocation、communication collective lowering 或 launch/package emission。

## 1. 设计目标

目标：

- 给 layout planner 一个稳定查询入口：每个 op 明确 operand/result 允许的 physical layout、
  preferred layout、materialization cost 和组合合法性。
- 给 instruction legalization / selection 一个稳定入口：每个 op 能提供可验证的 CT/NE/TDMA/RDMA/WDMA
  instruction family legality；R3.2d 再生成 instruction-level IR，并显式报告
  input/output/temp/scratch/accumulator/psum memref demand，以及 effect / async lifetime 对 buffer reuse
  的约束。
- 给 hardware lowering 一个稳定 legality target：CT、NE、native reduce、RDMA、WDMA、TDMA
  等 target family 的合法性先在 `wafer.tile.*` compute / movement 层被验证，再进入更低层发射。
- 保留 issue/drain 优化空间：IR 不在每个 compute/movement op 后隐式插入 wait。

非目标：

- 不重新做 group formation、fusion、traversal schedule 或 tile shape search。
- 不把 `linalg` op 名字、某个 workload、某个 internal split 或某个 C ABI helper 固化成架构边界。
- 不在本层表达 Direct DTE / collective；跨 tile data plane 属于 `wafer.tile.*` communication。
- 不直接生成裸寄存器 packet；raw packet/debug dialect 只属于更低层验证或调试路径。

## 2. IR 生命周期

`wafer.tile.*` compute op 不是 pipeline 末端才突然出现的 C ABI call。它应在 layout materialization
之前进入 IR，然后随类型和 storage 表示逐步 lower：

```text
scheduled wafer.group tensor body
  -> target-abstract tile_region IR with wafer.tile.* compute / movement ops
  -> layout materialization and accepted Wafer-tagged memref values
  -> instruction-level wafer.instr.* IR over unplaced Wafer-tagged memref values
  -> same instruction-level IR after SPM placement
  -> codegen emission to Wafer C ABI / packet / package metadata
```

各层表示：

| 层次 | op 形态 | value 形态 | 责任 |
| --- | --- | --- | --- |
| scheduled group | `linalg.*` / `tensor.*` / `scf.*` | tensor SSA value | 表达数学语义、tile-local dataflow 和 traversal，不选硬件实现 |
| target-abstract compute | `wafer.tile.*` compute ops 和 target-abstract movement op | tensor SSA value 或 Wafer-tagged memref | 选择目标实现族，提供 layout/resource/lowering interface，不绑定具体 storage placement |
| accepted layout | 同一类 compute/movement op | `memref<..., #wafer.memory<space, layout>>` | 验证 address space 和 physical layout marker，显式插入 `wafer.tile.materialize_layout` |
| instruction-level | `wafer.instr.*` | unplaced Wafer-tagged memref SSA value | 选择 CT/NE/TDMA/RDMA/WDMA 指令形态，列出 queue、temp/psum/staging memref values、alias、effect 和 descriptor attrs，不含 SPM offset；DTE 属于 `wafer.tile.*` communication / communication lowering |
| placed instruction-level | 同一 `wafer.instr.*` | placed memref、flat storage value 或 access descriptor | 具备 SPM offset/range/bank、stride/descriptor，可进入 codegen emission |
| launch/ABI emission | LLVM / C call / package metadata | concrete ABI arg | 调用 Wafer C ABI、发 package metadata、连接 host runtime；不作为主线 IR 层 |

因此，`wafer.tile.gemm` 这类 op 在不同阶段可以被 type conversion 改写 operand/result type，
但它的 semantic contract 仍是同一个：本 tile 内的 GEMM target implementation。若某个阶段需要
的信息无法由当前 IR、type、interface 或 verifier 推出，应扩 op/type/interface，而不是在 pass
side table 中保留影子计划。

### 2.1 R3.2d Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2c `wafer.tile.region` IR，内部包含 accepted layout 的
  Wafer-tagged memref、`wafer.tile.*` compute ops、`wafer.tile.*` movement ops、`wafer.tile.materialize_layout`、
  load/store boundary 和 view/alias relation。
- Current stage responsibility:
  对 target-abstract compute/movement/layout/load/store op 做 Wafer instruction legalization /
  selection，改写或构造 instruction-level `wafer.instr.*`，复用现有 Wafer-tagged memref
  SSA graph，并显式生成 queue/effect、temp/psum/staging、alias/view 和 descriptor attrs。
- Output artifact / IR:
  instruction-level Wafer IR over unplaced Wafer-tagged memref，或结构化 failure reason。
- Downstream consumer:
  R3.2e SPM placement、R3.2f DDR/resource legality、R3.2g closed-loop planner，以及 R3.6
  codegen emission。
- User-level driver / named pipeline:
  主线仍从 `wafer-opt --program-pipeline=stablehlo-spmd-to-group` 进入 R3.1/R3.2；
  R3.2d 可提供局部 dump / lit gate，但不能成为用户级 compile flow。
- Explicit non-goals:
  不决定 group boundary、tile shape、layout assignment、SPM offset、DDR BO allocation policy、ABI call
  symbol 或 packet field。
- Completion gate:
  对 R3.2c 已支持的 compute/movement/view family 生成 verifier-legal instruction-level IR；
  unsupported hardware instruction form 必须结构化失败，不能让 SPM placement 从 target-abstract op
  猜 memref demand。
```

## 3. Op 家族

V0 先覆盖能形成单 tile compute 闭环和后续 collective 原型所需的最小集合。

| 家族 | 建议 op | 语义 | 目标实现族 |
| --- | --- | --- | --- |
| matrix contraction | `wafer.tile.gemm` | tile-local matrix multiply / contraction；M/K/N、transpose、batch 语义来自 op contract 和 operand/result type | NE GEMM |
| elementwise / relation / logic / activation | `wafer.tile.elementwise` | 同 shape 或 verifier 可证明的 broadcast / scalar form；具体 kind 是语义 enum，不用名字匹配 | CT family |
| dtype conversion | `wafer.tile.convert` 或 `elementwise` convert kind | 明确 src/dst dtype pair、rounding mode 或 zero-point 语义 | CT convert |
| local reduction | `wafer.tile.reduce` | tile-local reduce；reduce dimensions 是语义字段，因为仅靠 input/output shape 可能无法唯一恢复 | native reduce 或 fallback compute sequence |
| local fill/copy/move | target-abstract movement op | SPM 内 copy/fill、DDR<->SPM tile load/store、strided movement、layout materialization support | RDMA / WDMA / TDMA / CT peripheral |
| conv / pool / unpool | 后续可引入 `wafer.tile.conv`、`pool`、`unpool` | 只有当前端 lowering 和 verifier 能稳定表达 semantic layout、pad/stride/dilation 等字段时启用 | NE / CT reduce-like family |

`wafer.tile.*` compute 不需要为每个底层 wrapper 造一个一一对应 op。op 的粒度应对应稳定的 compiler
语义和 verifier 合同；wrapper / C ABI 名字是 lowering 选择。比如 CT 加法、比较、激活可以由
同一个 elementwise op 通过受控 enum 表达，也可以在实现中拆成多个 op，只要 parser/printer、
verifier 和 lowering contract 一致即可。

### 3.1 GEMM

`wafer.tile.gemm` 表达本 tile 内的矩阵乘或批量矩阵乘。它不表达 group 的 internal reduction
split 决策；内部 reduction 是否需要进一步切分是 op tiling / SPM allocation search 的结果，
不能写成固定架构规则。

最小合同：

- operand/result 的 rank、shape、dtype 必须能推出 M/K/N 和输出 tile shape。
- transpose、batch、accumulator 或 psum 语义必须由 operand/result/use-def 或明确字段表达，
  不能靠变量名或示例参数顺序恢复。
- 如果存在累加输入，它应是 SSA operand；如果结果需要被后续累加，使用 SSA result 或
  loop-carried value 表达，不把 psum 生命周期复制成全局计划 attr。
- layout interface 给出 aligned-only 约束。2D 矩阵通常映射到 `Cx` family；具体 C0、padding 和
  descriptor 由 `computeWaferPhysicalTensorInfo`、SPM placement 和 placement realization 计算。

V0 不把 bias、scale、sparse、INT8 quant、fused activation 作为默认合同。若后续引入 fused form，
它们应是可验证 operand/attr，并能 canonicalize 回非 fused form 或明确 lower 到目标 wrapper。

### 3.2 Elementwise / Convert

`wafer.tile.elementwise` 表达 CT family 中的 arithmetic、relation、logic、activation 和
transcendental 子集。它应满足：

- op kind 使用受控 enum 或拆分 op，不通过字符串名字匹配。
- dtype 组合由 verifier 检查；普通 elementwise 默认 input/output dtype 一致，convert 明确记录
  src/dst dtype pair 和 rounding / zero-point 语义。
- bool/i1 使用 logical element count，storage bytes 和 bitpack 由 placement realization / lower-level
  verifier 负责。
- layout preference 通常是 flexible：若 producer 已经是 aligned layout，elementwise 可以继承以避免
  materialization；若 consumer 更偏好 compact，也可以在 cut edge 上 materialize。
- transformer block 需要的 elementwise 子集必须作为明确 kind 或拆分 op 表达，至少包括
  add、sub、mul、div、max、min、neg、recip、sqrt、rsqrt、exp、compare/select 或等价 mask-add。
  SiLU / GELU 可以先作为 staged decomposition，使用 sigmoid/tanh/erf/exp 中已经被 verifier
  支持的子集；没有被支持的 transcendental 不能靠名字 fallback。

当前落地的 V0 子集约束在 accepted-layout Wafer-tagged memref 形式：operands/result 必须是
`#wafer.memory<spm, tensor>`。普通 arithmetic / activation / transcendental 要求 operand/result element type
一致；relation kind 要求 operand element type 彼此一致、result element type 为 `i1`。op 由
`#wafer.elementwise_kind<...>` 记录 add/sub/mul/div/max/min/neg/recip/sqrt/rsqrt/exp/tanh 和
eq/ne/lt/le/gt/ge。`wafer.tile.elementwise` 可以不带
`indexing_maps`，此时要求所有 operand/result 逻辑 tensor type 完全一致；也可以携带和
`linalg.elementwise` 对齐的 projected-permutation `indexing_maps`，此时 result map 必须是 identity，
input map 的每个维度必须映射到 result 的一个维度，静态维度必须一致。这个合同覆盖当前
same-shape、row/head/vector broadcast 子集；更复杂 broadcast、scalar immediate、dynamic shape、
logic、select/mask 和 convert 仍按后续 gate 推进。

### 3.3 Reduce

`wafer.tile.reduce` 表达本 tile 内的 local reduce，不表达跨 tile collective reduce。跨 tile
reduce-scatter / all-reduce 由 `wafer.tile.*` communication 组合 local compute 和 communication。

当前 ring reduce collective lowering 使用 `wafer.tile.elementwise` 的 add/max/min 作为同形状
recv chunk 与 accumulator 的本地累计步骤；`wafer.tile.reduce` 仍只表示 tile 内按维度 reduce，
不被复用来伪装跨 tile collective reduction。

最小合同：

- reduce dimensions 是 op 语义的一部分。若从 `linalg.reduce` lowering 而来，维度来自 structured
  op；进入 `wafer.tile.reduce` 后仍应能被 verifier 和 printer 明确看到。
- V0 native reduce 只承诺 `sum`、`avg`、`max`、`min`。其它 reduction 可以在上游保持 structured
  loop，或 lower 成多个 supported compute op。
- native reduce 属于 aligned-only op，verifier 要求 rank <= 2 使用 `Cx`，rank > 2 使用 `NCx`；
  这来自硬件指令集的 Reduce operand/result physical layout 约束，不是 planner 偏好。
- output dtype、init value 和 NaN/overflow 等细节如果会影响语义，应保留在 op contract 中，而不是
  留给 wrapper 默认值。当前 tile-region IR lowering 从 scalar-constant `linalg.fill` out
  恢复 `init_value` attr；若 init 是 group boundary scalar，则作为 `wafer.tile.reduce` 的
  scalar init operand 保留 SSA 关系。R3.6 codegen emission 如果目标 wrapper 仍只接受 issue-time
  `init_value` 参数，必须把动态 init 明确拆成 native reduce + supported scalar combine，或扩展
  wrapper contract，不能在 R3.2c 丢失语义。

### 3.4 Movement Ops

load/store、SPM local copy、strided movement 和 layout materialization support 与 compute 紧密相邻，
但它们不等同于 tensor semantic compute。本文把它们称作 target-abstract movement op；最终可按工程
需要组织为 `wafer.tile.*` movement ops，并通过 memref type 上的 `#wafer.memory<space, layout>`
表达 memory space / physical layout；不新增单独 memory op namespace。

movement op 的合同：

- `wafer.tile.load` / `wafer.tile.store` 连接 `#wafer.memory<ddr, tensor>` compact external tensor boundary、
  DDR runtime allocation / resident constant source 和 `#wafer.memory<spm, *>` tile-local memref。host-visible dynamic input/output 默认
  compact；constant source 由 `ConstantLike` value、constant storage transform 和 load op contract 表达，
  DDR allocation policy / compiler-managed allocation / pool 由 DDR resource 文档定义。
- 当 `wafer.tile.load` 的 source 是 `ConstantLike` 时，load op 仍必须表达 logical slice / index
  operands。Weight chunking 是 storage/lowering 策略；compute op 只消费 load 后的 storage，
  不依赖旁路 metadata 或名字约定。
- scalar/splat/small constants 可以在 op lowering 中变成 immediate、attribute 或 fill pattern；
  只有需要作为 tensor tile data 读取的 constant 才生成 `wafer.tile.load` 和 DDR demand。
- `wafer.tile.materialize_layout` 是真实 data movement，不是 cast。它由 layout 文档定义，compute/movement
  lowering 负责把它展开成可执行的 GatherScatter、TDMA 或其它 path。
- `wafer.tile.extract_slice` / `wafer.tile.insert_slice` 表达 static offsets/sizes/strides 的
  tile-local slice movement。它们读写 SPM，并通过 verifier 检查 full slice shape、MLIR 合法的
  rank reduction、slice range、layout 和 memory-space；不能用 tensor name 或 side table 恢复
  slice。
- `wafer.tile.broadcast` / `wafer.tile.transpose` / `wafer.tile.copy` 表达 R2.4 当前以 passthrough
  `linalg.generic` 形式产出的 movement。它们是 TDMA/DataMove 候选，不是 compute elementwise。
- `wafer.tile.reshape` 只表达 static element-count-preserving shape view；它无 SPM write effect。
  如果 reshape 需要 physical layout change，必须使用 explicit materialization/movement op。
- RDMA 方向是 `#wafer.memory<ddr, *> -> #wafer.memory<spm, *>`，WDMA 方向是
  `#wafer.memory<spm, *> -> #wafer.memory<ddr, *>`。TDMA / local movement 只在 tile-local
  memory 或 verifier 允许的 address domain 内工作。
- stride 和 byte count 的单位在 placed instruction-level IR / access descriptor 层必须明确。上层 tensor stride 是 element stride，
  lower 到 DMA/TDMA/DTE descriptor 前必须转换成 byte stride。

## 4. Interfaces

长期合同应通过 MLIR op interface、type、effect 和 verifier 表达，而不是 pass 间 side table。

### 4.1 `WaferComputeOpInterface`

建议每个 `wafer.tile.*` compute ops 实现：

```text
getComputeKind()
verifySemanticOperandsAndResults()
getLoweringFamilies(target)
getInstructionFamilies(tileShape, layoutAssignment, target)
verifyInstructionLegality(instructionFamily, operands, results, target)
getAsyncLoweringPolicy(target)
```

它回答“这个 op 作为 tile-local compute 是什么，以及有哪些可验证硬件 instruction family”。它不回答
“最终 packet 每个 bit 怎么写”，也不直接替 SPM allocator 给出唯一 memref demand；memref demand
属于 R3.2d 生成的 instruction-level IR。

### 4.2 `WaferLayoutOpInterface`

当前 ODS / verifier 原型先覆盖 accepted-layout 层：layout-sensitive compute/movement op 通过
`collectWaferLayoutRequirements` 暴露 operand/result 当前承诺的 address space 和 layout marker，
并通过 `verifyWaferLayoutContract` 做 verifier 可调用检查。R3.2c 已把同一接口迁移到
`memref<..., #wafer.memory<space, layout>>`；旧 storage / split memory attr 路径不再是主线 IR
合同。
pre-assignment planner 需要的 allowed/preferred layout domain 仍是同一接口边界上的后续扩展：

```text
getAllowedLayouts(operand_or_result, tileShape, dtype, target)
getPreferredLayouts(operand_or_result, tileShape, dtype, target)
verifyLayoutCombination(operands, results)
getMaterializationCost(srcLayout, dstLayout, shape, dtype, target)
```

NE GEMM、native reduce、pool/unpool 是 aligned-only；多数 CT elementwise、DMA/TDMA movement 是
flexible，但仍可因 dtype、stride、range 或 bitpack 约束拒绝某些组合。

### 4.3 Effects and Resources

compute/movement op 应实现或组合 MLIR memory effect / resource effect：

- read effects：input storage、constant load source、DDR source。
- write effects：output storage、store destination、temporary/scratch。
- resource effects：CT/NE/RDMA/WDMA/TDMA issue family、worker resource、SPM bank/page/color class。
- async policy：op 是否可 lower 成 issue-only，以及哪些 buffer lifetime 必须延伸到 drain/wait。

这些 effect 用于 liveness、SPM reuse、scheduler 和 verifier。它们不等于保存一份全局 issue plan。

R1.2 的具体接口是 `collectWaferResourceEffects` / `verifyWaferResourceEffectContract`。它返回结构化
`WaferResourceEffect`，区分 SPM、DDR、movement、compute、communication 和 sync，以及 read/write/
issue/wait/drain。关键 movement/compute/comm op 同时接入 MLIR `MemoryEffectOpInterface` 的
Wafer resource，供通用 effect 分析查询。

## 5. Verifier and Legality

Target-abstract verifier：

- operand/result type、rank、shape、dtype 与 op semantic fields 一致。
- reduce dimensions、GEMM M/K/N、broadcast 或 scalar form 可由 IR 明确证明。
- op 不携带 raw packet field、worker id、SPM offset、DTE resource id 或 C ABI symbol。
- layout-sensitive op 必须实现 `WaferLayoutOpInterface`。
- 不允许通过名字匹配恢复 operand role。

Accepted layout verifier：

- Wafer-tagged memref 的 layout marker 满足 op hard constraint。
- `wafer.tile.materialize_layout` 的 source/destination layout family 合法，且 materialization op 有真实 movement
  lowering。
- loop-carried buffer 的 entry/yield layout 一致，除非 loop body 内有显式 materialization。
- boundary load/store 的 external layout contract 与 host/runtime 或 package metadata 一致。

Placed instruction-level verifier：

- CT、NE、TDMA operand 是 SPM address 或 descriptor；RDMA source 是 DDR、destination 是 SPM；
  WDMA source 是 SPM、destination 是 DDR。
- memory-space verifier 必须用统一的 `#wafer.memory<space, layout>` 检查这些 address domains；不能把
  external boundary、DDR descriptor 和 SPM storage 当成几套不相干的空间语义。
- DDR access descriptor 的 pool/domain、compiler-managed slice、external allocation、capacity 和 bandwidth 不是本层
  op attr；本层只通过 memory effects、range、byte count 和 direction contract 把需求暴露给
  DDR resource planner。
- stride、iteration、byte count、range end、bool bitpack 和 alignment 规则已完成转换和检查。
- 普通 `TsmExecute` 路径只覆盖 CT、NE、RDMA、WDMA、TDMA；SCALAR、DTE、CSR 不走该 path。
- local drain 只出现在 Kcore 可见性、host-visible boundary、DTE/stream protocol、group barrier 或 task end
  等需要完成证明的位置。

## 6. Lowering Passes

建议 pass 边界按 IR contract 命名，pass 名称可调整：

| 阶段 | 输入 | 输出 | 责任 |
| --- | --- | --- | --- |
| select Wafer compute implementation | tiled `linalg` / tensor / SCF | target-abstract `wafer.tile.*` compute / movement op | 选择本 tile 实现族，保留数学语义，建立 layout/resource interface |
| layout materialization | target-abstract Wafer op | accepted Wafer-tagged memref + materialization edge | 基于 op interface 做 layout assignment 和真实 movement cut |
| instruction legalization / selection | accepted layout IR | instruction-level `wafer.instr.*` over unplaced Wafer-tagged memref | 将 target-abstract op 改写成 CT/NE/TDMA/RDMA/WDMA 指令形态，列出 queue、effects、temp/psum/staging memref values、alias 和 descriptor attrs；DTE communication 不进入普通 `wafer.instr` path |
| SPM placement | instruction-level IR with unplaced Wafer-tagged memref | same instruction-level IR with placed SPM memref values | 从 memref use-def 和 instruction effects 收集 demand、liveness，分配 offset/range/bank |
| placement realization | placed instruction-level IR | placed `memref` / flat storage / access descriptor | 复用标准 memref lowering 或生成目标 access descriptor |
| codegen emission | placed instruction-level IR | LLVM call / C ABI call / package metadata | 生成具体 ABI call 或 packet emission，不回头修改 schedule/layout |

如果一个 pass 创建 `wafer.tile.*` compute、movement、layout、SPM 或 sync op，应声明 dependent dialects。pass
pipeline 只表达 transformation 顺序，不承载隐藏语义。

## 7. Issue / Drain Model

硬件支持 CT、NE、RDMA、WDMA、TDMA 独立提交和依赖检测。编译器 IR 不应继承“每个 helper 后立刻
wait”的保守 CRT 风格。

V0 模型：

- target-abstract compute/movement op 从 SSA 语义看是顺序 op；lowering 可以把它拆成 issue op 和
  later drain/wait op。
- instruction legalization / selection 决定哪些 issue / drain / wait event 参与 storage lifetime；
  SPM placement 通过 instruction effect event 扩展 async op 的 source/destination lifetime。
- local drain 是显式 sync op，例如 `wafer.instr.local_drain` 或等价 IR；它不是 compute op 的默认后缀。
- DTE wait、stream wait、group barrier 属于 `wafer.tile.*` communication / `wafer.instr.local_drain` 和后续 sync boundary 的完成边界，不能用 local
  NCC wait 代替。

这样做允许 single-tile local compute 先走 correctness-first 同步路径，也允许后续逐步打开 overlap，而不改变上层
compute op 语义。

## 8. V0 Coverage

V0 推荐实现顺序：

1. `wafer.tile.elementwise`：覆盖一个 unary、一个 binary、一个 convert 或 relation。
2. target-abstract load/store 和 RDMA/WDMA contiguous movement。
3. `wafer.tile.gemm`：覆盖基础 NE GEMM，不带 fused bias/activation/quant。
4. `wafer.tile.reduce`：覆盖 `sum/max/min/avg` 中至少一个。
5. `wafer.tile.materialize_layout` 到 GatherScatter / TDMA 的最小闭环。

当前旧原型已经先覆盖了 accepted-layout `wafer.tile.gemm`、load/store、layout materialize，
并补入 same-shape identity 与 projected-permutation limited broadcast elementwise 到
`wafer.tile.elementwise` 的 target-abstract path；随后补入 sum/max/min
local reduce 到 `wafer.tile.reduce` 的 path，保留 reduce dimensions
和 scalar init value。P5.8 后续又补入 attention QK^T / AV 的 rank-4 contraction physical
slice：只接受可由 `linalg.generic` indexing maps、parallel/reduction iterator types、mul-add
body 和静态 shape relation 验证的 batch/head 形态，materialize 为带显式 batch/head/m/k/n 维度
attrs 的 `wafer.tile.gemm`。后续 R3.2d/R3.6 必须把它 lower 成带 `batch_count` 和 M/K/N 的
instruction-level GEMM 以及对应 C ABI emission。历史 transformer fixed package fixture 已删除；compiler-managed/resident constant metadata、
resource summary 一致性验证和 full block package manifest 必须由后续 IR-derived package gate
恢复。当前覆盖仍不是通用 elementwise/reduce/GEMM coverage；更复杂 broadcast、relation/logic、convert、多输入/非
constant-init reduce 和 mask/select 仍按后续泛化 gate 推进。当前 coverage 不能被解释成
Wafer compute 语义上不支持这些结构；只要硬件 wrapper / structured lowering 能表达，就应补
compute op、verifier、instruction lowering、ABI emission 或 resource gate。

V1 或后续扩展：

- conv / pool / unpool 的完整 semantic layout 和 verifier。
- fused GEMM / conv epilogue。
- 更复杂 broadcast、masked op、dynamic shape。
- raw packet builder 和 wrapper-golden 双路径测试。
- PMU/cost-model 驱动的 issue overlap。

### 8.1 Transformer Block Minimum Coverage

不能只因为 GEMM、一个 elementwise 和一个 reduce 能跑，就声称 transformer block 支持完成。
在 transformer block vertical slice 之前，compute/movement 层至少要覆盖：

- `wafer.tile.gemm` 的 batch/head 维和 transpose relation，用于 QKV projection、QK^T、
  attention value、output projection 和 MLP。
- `wafer.tile.reduce` 的 `max` 和 `sum`，用于 softmax；`sum` 或 `avg`，用于 RMSNorm /
  LayerNorm。
- elementwise `add/sub/mul/div/max/min/neg/recip/sqrt/rsqrt/exp`。
- limited broadcast：scalar、vector、head_dim 或 row-wise broadcast 必须能由 type/indexing map
  验证。
- compare/select 或 mask-add path，用于 causal / padding mask。若直接使用 large negative
  add-mask，constant 必须走普通 `ConstantLike` / immediate / load 规则。
- load/store 对 sin/cos RoPE table、norm scale/bias、projection weights 和 MLP weights 的
  constant slice 关系。

不在第一版 transformer block gate 内：

- dropout/random mask。
- dynamic sequence length 的通用 runtime specialization。
- paged KV cache 和 serving prefill/decode 调度。
- fused GEMM epilogue 或 fused softmax op；这些可以后续作为优化，但不是语义前提。

## 9. Case Fragment

下面的 case 只展示 IR 如何流经 compute 层，不定义架构边界。tile shape、layout choice 和 op
implementation 都是 planner 的候选结果。

Target-abstract tile-region：

```mlir
%a_tile = tensor.extract_slice %a[%m0, 0] [64, 256] [1, 1]
    : tensor<128x256xf16> to tensor<64x256xf16>
%b_tile = tensor.extract_slice %b[0, %n0] [256, 64] [1, 1]
    : tensor<256x128xf16> to tensor<256x64xf16>

%mm = wafer.tile.gemm %a_tile, %b_tile
    : (tensor<64x256xf16>, tensor<256x64xf16>) -> tensor<64x64xf16>
%act = wafer.tile.elementwise %mm {kind = #wafer.elementwise_kind<relu>}
    : tensor<64x64xf16> -> tensor<64x64xf16>
%row_sum = wafer.tile.reduce #wafer.reduce_kind<sum> %act
    {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
    : tensor<64x64xf16> -> tensor<64xf32>
```

Accepted layout 后：

```mlir
%a_spm = wafer.tile.load %a[%m0, 0]
    : tensor<64x256xf16> -> memref<64x256xf16, #wafer.memory<spm, tensor>>
%a_cx = wafer.tile.materialize_layout %a_spm
    : memref<64x256xf16, #wafer.memory<spm, tensor>>
   -> memref<64x256xf16, #wafer.memory<spm, cx>>

%b_spm = wafer.tile.load %b[0, %n0]
    : tensor<256x64xf16> -> memref<256x64xf16, #wafer.memory<spm, tensor>>
%b_cx = wafer.tile.materialize_layout %b_spm
    : memref<256x64xf16, #wafer.memory<spm, tensor>>
   -> memref<256x64xf16, #wafer.memory<spm, cx>>

%mm = wafer.tile.gemm %a_cx, %b_cx
    : (memref<64x256xf16, #wafer.memory<spm, cx>>,
       memref<256x64xf16, #wafer.memory<spm, cx>>)
   -> memref<64x64xf16, #wafer.memory<spm, cx>>

%act = wafer.tile.elementwise %mm {kind = #wafer.elementwise_kind<relu>}
    : memref<64x64xf16, #wafer.memory<spm, cx>>
   -> memref<64x64xf16, #wafer.memory<spm, cx>>

%row_sum = wafer.tile.reduce #wafer.reduce_kind<sum> %act
    {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
    : memref<64x64xf16, #wafer.memory<spm, cx>>
   -> memref<64xf32, #wafer.memory<spm, tensor>>
```

这个例子里 `wafer.tile.gemm` 需要 aligned layout，elementwise 继承 producer layout，reduce
根据自己的 implementation 给出 hard constraint。是否把某个 internal reduction dimension 再切分、
是否 materialize output 为 compact、是否启用 double buffer，都由 layout/SPM/scheduler analysis
闭环决定，不是 `wafer.tile.*` compute op 自己保存的计划。

## 10. 与其它文档的关系

全局文档边界见 `tasks/2026-05-11-wafer-ai-compiler-architecture.md` 第 8 节。本文只维护
target-abstract compute/movement op 的语义、interface 和 lowering legality；group formation、
layout assignment、SPM/DDR allocation、communication 和 launch/runtime 不在本文重复定义。
register-level wrapper / packet 约束只在 placed instruction-level lowering / codegen emission 后消费。
