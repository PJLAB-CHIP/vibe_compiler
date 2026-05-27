# Wafer Compute and Movement Dialect Design

日期：2026-05-25

状态：设计草案；2026-05-25 边界收口

本文定义 Wafer 后端中 target-abstract compute / movement IR 的边界。它连接
`wafer.group` 产生的 tile-local tensor program、layout materialization / SPM bufferization，
以及后续 instruction / C ABI lowering。

本文中的 `wafer.compute` 是正式 IR contract。它表达“这个 tile-local op 已经选择了某类
Wafer 目标实现，并能提供 layout、buffer demand、effect 和 lowering legality”。它仍然不表达
raw packet bitfield、SPM physical offset、worker window、runtime launch 或 host ABI。

本文只负责 target-abstract compute/movement op 的语义、interface、effect、issue/drain 和
lowering legality。它不重新做 group formation、tile search、layout assignment、SPM/DDR
allocation、communication collective lowering 或 launch/package emission。

## 1. 设计目标

目标：

- 给 layout planner 一个稳定查询入口：每个 op 明确 operand/result 允许的 physical layout、
  preferred layout、materialization cost 和组合合法性。
- 给 SPM oracle 一个稳定输入：每个 op 能报告 input/output/temp/scratch/accumulator demand，
  以及 effect / async lifetime 对 buffer reuse 的约束。
- 给 hardware lowering 一个稳定 legality target：CT、NE、native reduce、RDMA、WDMA、TDMA
  等 target family 的合法性先在 `wafer.compute` / movement 层被验证，再进入更低层发射。
- 保留 issue/drain 优化空间：IR 不在每个 compute/movement op 后隐式插入 wait。

非目标：

- 不重新做 group formation、fusion、traversal schedule 或 tile shape search。
- 不把 `linalg` op 名字、某个 workload、某个 internal split 或某个 C ABI helper 固化成架构边界。
- 不在本层表达 Direct DTE / collective；跨 tile data plane 属于 `wafer.comm`。
- 不直接生成裸寄存器 packet；raw packet/debug dialect 只属于更低层验证或调试路径。

## 2. IR 生命周期

`wafer.compute` op 不是 pipeline 末端才突然出现的 C ABI call。它应在 layout materialization
之前进入 IR，然后随类型和 storage 表示逐步 lower：

```text
scheduled wafer.group tensor body
  -> target-abstract tile_region IR with wafer.compute / movement ops
  -> layout materialization and accepted !wafer.tile_buffer values
  -> SPM bufferization and tile buffer storage realization
  -> lower-level instruction/runtime-form Wafer ops
  -> LLVM call to Wafer C ABI or package/runtime emission
```

各层表示：

| 层次 | op 形态 | value 形态 | 责任 |
| --- | --- | --- | --- |
| scheduled group | `linalg.*` / `tensor.*` / `scf.*` | tensor SSA value | 表达数学语义、tile-local dataflow 和 traversal，不选硬件实现 |
| target-abstract compute | `wafer.compute.*` 和 target-abstract movement op | tensor SSA value | 选择目标实现族，提供 layout/resource/lowering interface，不绑定 storage |
| accepted layout | 同一类 compute/movement op | `!wafer.tile_buffer<shape,dtype,mem_layout,space>` | 验证 physical layout，显式插入 `wafer.layout.materialize` |
| storage-realized | lower-level Wafer op | `memref`、flat storage value 或 descriptor | 具备 address/range/stride/descriptor，可进入 instruction/runtime lowering |
| launch/ABI | LLVM / `wafer.launch` | concrete ABI arg | 调用 Wafer C ABI、发 package metadata、连接 host runtime |

因此，`wafer.compute.gemm` 这类 op 在不同阶段可以被 type conversion 改写 operand/result type，
但它的 semantic contract 仍是同一个：本 tile 内的 GEMM target implementation。若某个阶段需要
的信息无法由当前 IR、type、interface 或 verifier 推出，应扩 op/type/interface，而不是在 pass
side table 中保留影子计划。

## 3. Op 家族

V0 先覆盖能形成单 tile compute 闭环和后续 collective 原型所需的最小集合。

| 家族 | 建议 op | 语义 | 目标实现族 |
| --- | --- | --- | --- |
| matrix contraction | `wafer.compute.gemm` | tile-local matrix multiply / contraction；M/K/N、transpose、batch 语义来自 op contract 和 operand/result type | NE GEMM |
| elementwise / relation / logic / activation | `wafer.compute.elementwise` | 同 shape 或 verifier 可证明的 broadcast / scalar form；具体 kind 是语义 enum，不用名字匹配 | CT family |
| dtype conversion | `wafer.compute.convert` 或 `elementwise` convert kind | 明确 src/dst dtype pair、rounding mode 或 zero-point 语义 | CT convert |
| local reduction | `wafer.compute.reduce` | tile-local reduce；reduce dimensions 是语义字段，因为仅靠 input/output shape 可能无法唯一恢复 | native reduce 或 fallback compute sequence |
| local fill/copy/move | target-abstract movement op | SPM 内 copy/fill、DDR<->SPM tile load/store、strided movement、layout materialization support | RDMA / WDMA / TDMA / CT peripheral |
| conv / pool / unpool | 后续可引入 `wafer.compute.conv`、`pool`、`unpool` | 只有当前端 lowering 和 verifier 能稳定表达 semantic layout、pad/stride/dilation 等字段时启用 | NE / CT reduce-like family |

`wafer.compute` 不需要为每个底层 wrapper 造一个一一对应 op。op 的粒度应对应稳定的 compiler
语义和 verifier 合同；wrapper / C ABI 名字是 lowering 选择。比如 CT 加法、比较、激活可以由
同一个 elementwise op 通过受控 enum 表达，也可以在实现中拆成多个 op，只要 parser/printer、
verifier 和 lowering contract 一致即可。

### 3.1 GEMM

`wafer.compute.gemm` 表达本 tile 内的矩阵乘或批量矩阵乘。它不表达 group 的 internal reduction
split 决策；内部 reduction 是否需要进一步切分是 op tiling / SPM feasibility search 的结果，
不能写成固定架构规则。

最小合同：

- operand/result 的 rank、shape、dtype 必须能推出 M/K/N 和输出 tile shape。
- transpose、batch、accumulator 或 psum 语义必须由 operand/result/use-def 或明确字段表达，
  不能靠变量名或示例参数顺序恢复。
- 如果存在累加输入，它应是 SSA operand；如果结果需要被后续累加，使用 SSA result 或
  loop-carried value 表达，不把 psum 生命周期复制成全局计划 attr。
- layout interface 给出 aligned-only 约束。2D 矩阵通常映射到 `Cx` family；具体 C0、padding 和
  descriptor 由 layout/SPM/storage realization 计算。

V0 不把 bias、scale、sparse、INT8 quant、fused activation 作为默认合同。若后续引入 fused form，
它们应是可验证 operand/attr，并能 canonicalize 回非 fused form 或明确 lower 到目标 wrapper。

### 3.2 Elementwise / Convert

`wafer.compute.elementwise` 表达 CT family 中的 arithmetic、relation、logic、activation 和
transcendental 子集。它应满足：

- op kind 使用受控 enum 或拆分 op，不通过字符串名字匹配。
- dtype 组合由 verifier 检查；普通 elementwise 默认 input/output dtype 一致，convert 明确记录
  src/dst dtype pair 和 rounding / zero-point 语义。
- bool/i1 使用 logical element count，storage bytes 和 bitpack 由 storage realization / lower-level
  verifier 负责。
- layout preference 通常是 flexible：若 producer 已经是 aligned layout，elementwise 可以继承以避免
  materialization；若 consumer 更偏好 compact，也可以在 cut edge 上 materialize。
- transformer block 需要的 elementwise 子集必须作为明确 kind 或拆分 op 表达，至少包括
  add、sub、mul、div、max、min、neg、recip、sqrt、rsqrt、exp、compare/select 或等价 mask-add。
  SiLU / GELU 可以先作为 staged decomposition，使用 sigmoid/tanh/erf/exp 中已经被 verifier
  支持的子集；没有被支持的 transcendental 不能靠名字 fallback。

当前落地的 V0 子集约束在 accepted-layout `!wafer.tile_buffer` 形式：operands/result 必须是
SPM + tensor layout、element type 一致，并由 `#wafer.elementwise_kind<...>` 记录
add/sub/mul/div/max/min/neg/recip/sqrt/rsqrt/exp/tanh。`wafer.compute.elementwise` 可以不带
`indexing_maps`，此时要求所有 operand/result 逻辑 tensor type 完全一致；也可以携带和
`linalg.elementwise` 对齐的 projected-permutation `indexing_maps`，此时 result map 必须是 identity，
input map 的每个维度必须映射到 result 的一个维度，静态维度必须一致。这个合同覆盖当前
same-shape、row/head/vector broadcast 子集；更复杂 broadcast、scalar immediate、dynamic shape、
relation/logic 和 convert 仍按后续 gate 推进。

### 3.3 Reduce

`wafer.compute.reduce` 表达本 tile 内的 local reduce，不表达跨 tile collective reduce。跨 tile
reduce-scatter / all-reduce 由 `wafer.comm` 组合 local compute 和 communication。

当前 ring reduce collective lowering 使用 `wafer.compute.elementwise` 的 add/max/min 作为同形状
recv chunk 与 accumulator 的本地累计步骤；`wafer.compute.reduce` 仍只表示 tile 内按维度 reduce，
不被复用来伪装跨 tile collective reduction。

最小合同：

- reduce dimensions 是 op 语义的一部分。若从 `linalg.reduce` lowering 而来，维度来自 structured
  op；进入 `wafer.compute.reduce` 后仍应能被 verifier 和 printer 明确看到。
- V0 native reduce 只承诺 `sum`、`avg`、`max`、`min`。其它 reduction 可以在上游保持 structured
  loop，或 lower 成多个 supported compute op。
- native reduce 属于 aligned-only op，layout planner 必须看到 input/output hard constraint。
- output dtype、init value 和 NaN/overflow 等细节如果会影响语义，应保留在 op contract 中，而不是
  留给 wrapper 默认值。当前 local skeleton 从 scalar-constant `linalg.fill` out 恢复
  `init_value`，并把 `dimensions` / `init_value` 一起带到 `wafer.abi.reduce`。

### 3.4 Movement Ops

load/store、SPM local copy、strided movement 和 layout materialization support 与 compute 紧密相邻，
但它们不等同于 tensor semantic compute。本文把它们称作 target-abstract movement op；最终可按工程
需要组织为 `wafer.mem.*`、`wafer.move.*` 或同一 Wafer namespace 下的 op。

movement op 的合同：

- `wafer.load_tile` / `wafer.store_tile` 连接 `#ddr` compact external tensor boundary、DDR workspace /
  resident constant descriptor 和 `#spm` tile-local buffer。host-visible dynamic input/output 默认
  compact；constant source 由 `ConstantLike` value、constant storage transform 和 load op contract 表达，
  DDR binding / workspace / pool 由 DDR resource 文档定义。
- 当 `wafer.load_tile` 的 source 是 `ConstantLike` 时，load op 仍必须表达 logical slice / index
  operands。Weight chunking 是 storage/lowering 策略；compute op 只消费 load 后的 tile buffer，
  不依赖旁路 metadata 或名字约定。
- scalar/splat/small constants 可以在 op lowering 中变成 immediate、attribute 或 fill pattern；
  只有需要作为 tensor tile data 读取的 constant 才生成 `wafer.load_tile` 和 DDR demand。
- `wafer.layout.materialize` 是真实 data movement，不是 cast。它由 layout 文档定义，compute/movement
  lowering 负责把它展开成可执行的 GatherScatter、TDMA 或其它 path。
- RDMA 方向是 `#ddr -> #spm`，WDMA 方向是 `#spm -> #ddr`。TDMA / local movement 只在 tile-local
  memory 或 verifier 允许的 address domain 内工作。
- stride 和 byte count 的单位在 storage-realized 层必须明确。上层 tensor stride 是 element stride，
  lower 到 DMA/TDMA/DTE descriptor 前必须转换成 byte stride。

## 4. Interfaces

长期合同应通过 MLIR op interface、type、effect 和 verifier 表达，而不是 pass 间 side table。

### 4.1 `WaferComputeOpInterface`

建议每个 `wafer.compute.*` 实现：

```text
getComputeKind()
verifySemanticOperandsAndResults()
getLoweringFamilies(target)
collectTileBufferDemand(tileShape, layoutAssignment, target)
getAsyncLoweringPolicy(target)
```

它回答“这个 op 作为 tile-local compute 是什么”，不回答“最终 packet 每个 bit 怎么写”。

### 4.2 `WaferLayoutOpInterface`

R1.2 当前实现先覆盖 accepted-layout 层：所有 layout-sensitive compute/movement op 通过
`collectWaferLayoutRequirements` 暴露当前 IR 中 operand/result `!wafer.tile_buffer` 已经承诺的
`mem_layout` 和 `memory_space`，并通过 `verifyWaferLayoutContract` 做 verifier 可调用检查。
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

- read effects：input tile buffer、constant load source、DDR source。
- write effects：output tile buffer、store destination、temporary/scratch。
- resource effects：CT/NE/RDMA/WDMA/TDMA queue family、worker resource、SPM bank/page/color class。
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

- `!wafer.tile_buffer` 的 `mem_layout` 满足 op hard constraint。
- `wafer.layout.materialize` 的 source/destination layout family 合法，且 materialization op 有真实 movement
  lowering。
- loop-carried buffer 的 entry/yield layout 一致，除非 loop body 内有显式 materialization。
- boundary load/store 的 external layout contract 与 host/runtime 或 package metadata 一致。

Storage-realized / instruction-form verifier：

- CT、NE、TDMA operand 是 SPM address 或 descriptor；RDMA source 是 DDR、destination 是 SPM；
  WDMA source 是 SPM、destination 是 DDR。
- memory-space verifier 必须用统一的 `WaferMemorySpaceAttr` 检查这些 address domains；不能把
  external boundary、DDR descriptor 和 SPM tile buffer 当成几套不相干的空间语义。
- DDR descriptor 的 pool/domain、workspace slice、external binding、capacity 和 bandwidth 不是本层
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
| select Wafer compute implementation | tiled `linalg` / tensor / SCF | target-abstract `wafer.compute` / movement op | 选择本 tile 实现族，保留数学语义，建立 layout/resource interface |
| layout materialization | target-abstract Wafer op | accepted `!wafer.tile_buffer` + materialization edge | 基于 op interface 做 layout assignment 和真实 movement cut |
| SPM bufferization | accepted tile buffer IR | allocation-ready tile-region IR | 收集 buffer demand、liveness、effects、SPM feasibility |
| tile buffer storage realization | `!wafer.tile_buffer` | `memref` / flat storage / descriptor | 复用标准 memref lowering 或生成目标 descriptor |
| lower compute/movement to instruction form | storage-realized Wafer op | lower-level Wafer instruction/runtime op | 选择 CT/NE/RDMA/WDMA/TDMA family 和 Wafer C ABI shape |
| Wafer to LLVM C ABI | instruction/runtime op | LLVM call / package metadata | 生成具体 ABI call，不回头修改 schedule/layout |

如果一个 pass 创建 `wafer.compute`、movement、layout、SPM 或 sync op，应声明 dependent dialects。pass
pipeline 只表达 transformation 顺序，不承载隐藏语义。

## 7. Issue / Drain Model

硬件支持 CT、NE、RDMA、WDMA、TDMA 独立提交和依赖检测。编译器 IR 不应继承“每个 helper 后立刻
wait”的保守 CRT 风格。

V0 模型：

- target-abstract compute/movement op 从 SSA 语义看是顺序 op；lowering 可以把它拆成 issue op 和
  later drain/wait op。
- SPM oracle 通过 effect event 扩展 async op 的 source/destination lifetime。
- local drain 是显式 sync op，例如 `wafer.sync.local_wait` 或等价 IR；它不是 compute op 的默认后缀。
- DTE wait、stream wait、group barrier 属于 `wafer.comm` / `wafer.sync` 的完成边界，不能用 local
  NCC wait 代替。

这样做允许 M0 先走 correctness-first 同步路径，也允许 M5 以后逐步打开 overlap，而不改变上层
compute op 语义。

## 8. V0 Coverage

V0 推荐实现顺序：

1. `wafer.compute.elementwise`：覆盖一个 unary、一个 binary、一个 convert 或 relation。
2. target-abstract load/store 和 RDMA/WDMA contiguous movement。
3. `wafer.compute.gemm`：覆盖基础 NE GEMM，不带 fused bias/activation/quant。
4. `wafer.compute.reduce`：覆盖 `sum/max/min/avg` 中至少一个。
5. `wafer.layout.materialize` 到 GatherScatter / TDMA 的最小闭环。

当前实现顺序已经先覆盖了 accepted-layout `wafer.compute.gemm`、load/store、layout materialize，
并补入 same-shape identity 与 projected-permutation limited broadcast elementwise 到
`wafer.compute.elementwise` / `wafer.abi.elementwise` 的 local skeleton；随后补入 sum/max/min
local reduce 到 `wafer.compute.reduce` / `wafer.abi.reduce` 的 skeleton，保留 reduce dimensions
和 scalar init value。P5.8 后续又补入 attention QK^T / AV 的 rank-4 contraction physical
slice：只接受可由 `linalg.generic` indexing maps、parallel/reduction iterator types、mul-add
body 和静态 shape relation 验证的 batch/head 形态，materialize 为带显式 batch/head/m/k/n 维度
attrs 的 `wafer.compute.gemm`，并 lower 到带 `batch_count` 和 M/K/N 的 `wafer.abi.gemm`
skeleton。M6 package gate 随后补入 workspace/resident constant metadata、resource summary
一致性验证和当前 full block lowering 输出的完整 ABI issue 序列。它仍不是通用
elementwise/reduce/GEMM coverage；更复杂 broadcast、relation/logic、convert、多输入/非
constant-init reduce 和 mask/select 仍按后续泛化 gate 推进。当前 coverage 不能被解释成
Wafer compute 语义上不支持这些结构；只要硬件 wrapper / structured lowering 能表达，就应补
compute op、verifier、ABI 或 resource gate。

V1 或后续扩展：

- conv / pool / unpool 的完整 semantic layout 和 verifier。
- fused GEMM / conv epilogue。
- 更复杂 broadcast、masked op、dynamic shape。
- raw packet builder 和 wrapper-golden 双路径测试。
- PMU/cost-model 驱动的 issue overlap。

### 8.1 Transformer Block Minimum Coverage

不能只因为 GEMM、一个 elementwise 和一个 reduce 能跑，就声称 transformer block 支持完成。
在 M6 transformer block vertical slice 之前，compute/movement 层至少要覆盖：

- `wafer.compute.gemm` 的 batch/head 维和 transpose relation，用于 QKV projection、QK^T、
  attention value、output projection 和 MLP。
- `wafer.compute.reduce` 的 `max` 和 `sum`，用于 softmax；`sum` 或 `avg`，用于 RMSNorm /
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

%mm = wafer.compute.gemm %a_tile, %b_tile
    : (tensor<64x256xf16>, tensor<256x64xf16>) -> tensor<64x64xf16>
%act = wafer.compute.elementwise %mm {kind = #wafer.compute_kind<relu>}
    : tensor<64x64xf16> -> tensor<64x64xf16>
%row_sum = wafer.compute.reduce #wafer.reduce_kind<sum> %act
    {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
    : tensor<64x64xf16> -> tensor<64xf32>
```

Accepted layout 后：

```mlir
%a_spm = wafer.load_tile %a[%m0, 0]
    : tensor<64x256xf16> -> !wafer.tile_buffer<64x256xf16, #tensor, #spm>
%a_cx = wafer.layout.materialize %a_spm
    : !wafer.tile_buffer<64x256xf16, #tensor, #spm>
   -> !wafer.tile_buffer<64x256xf16, #cx, #spm>

%b_spm = wafer.load_tile %b[0, %n0]
    : tensor<256x64xf16> -> !wafer.tile_buffer<256x64xf16, #tensor, #spm>
%b_cx = wafer.layout.materialize %b_spm
    : !wafer.tile_buffer<256x64xf16, #tensor, #spm>
   -> !wafer.tile_buffer<256x64xf16, #cx, #spm>

%mm = wafer.compute.gemm %a_cx, %b_cx
    : (!wafer.tile_buffer<64x256xf16, #cx, #spm>,
       !wafer.tile_buffer<256x64xf16, #cx, #spm>)
   -> !wafer.tile_buffer<64x64xf16, #cx, #spm>

%act = wafer.compute.elementwise %mm {kind = #wafer.compute_kind<relu>}
    : !wafer.tile_buffer<64x64xf16, #cx, #spm>
   -> !wafer.tile_buffer<64x64xf16, #cx, #spm>

%row_sum = wafer.compute.reduce #wafer.reduce_kind<sum> %act
    {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
    : !wafer.tile_buffer<64x64xf16, #cx, #spm>
   -> !wafer.tile_buffer<64xf32, #tensor, #spm>
```

这个例子里 `wafer.compute.gemm` 需要 aligned layout，elementwise 继承 producer layout，reduce
根据自己的 implementation 给出 hard constraint。是否把某个 internal reduction dimension 再切分、
是否 materialize output 为 compact、是否启用 double buffer，都由 layout/SPM/scheduler analysis
闭环决定，不是 `wafer.compute` op 自己保存的计划。

## 10. 与其它文档的关系

全局文档边界见 `tasks/2026-05-11-wafer-ai-compiler-architecture.md` 第 8 节。本文只维护
target-abstract compute/movement op 的语义、interface 和 lowering legality；group formation、
layout assignment、SPM/DDR allocation、communication 和 launch/runtime 不在本文重复定义。
register-level wrapper / packet 约束只在 storage-realized lowering 后消费。
