# Wafer Local Compute Normalization Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口；2026-05-27 补 post-SPMD Wafer LinalgExt-style
tensor collective handoff；2026-06-01 明确 softmax/norm 是 fine-grained StableHLO staged graph，
本层不承载 SPMD partition

本文定义 SPMD 产出的 StableHLO local program 到 `wafer.group` 之前的 local tensor normalization
边界。输入可以是用户 sharding 经过 partitioner 后的本地 shard 程序，也可以是 no-user-sharding
默认 policy 产生的 replicated / single-tile local body。该阶段负责把本地 compute 规整到可 tile、
可 fuse、可验证的 `linalg` / `tensor` / `scf` / `arith` / `math` IR 子集；如果输入包含 post-SPMD
StableHLO collective，则同时把它规整成 Wafer LinalgExt-style tensor collective ops，使 collective
能和 local compute 一起进入 group/tiling。它不引入 Wafer physical layout、SPM/DDR allocation、
DTE、C ABI 或 runtime package。

本文依赖：

- `tasks/2026-05-25-wafer-frontend-stablehlo-artifact-design.md`
- `tasks/2026-05-25-wafer-shardy-spmd-design.md`
- `tasks/2026-05-12-wafer-group-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- MLIR Linalg / Bufferization / Dialect Conversion 官方文档。

## 1. 目标和非目标

目标：

- 把 StableHLO local shard compute lowering 到结构化 tensor IR。
- 把 StableHLO local shard collective lowering 到 Wafer LinalgExt-style tensor collective IR。
- 保留 op 的 indexing map、iterator type、DPS operand/result 关系、shape、dtype 和
  broadcast / reduction 语义。
- 把 transformer block 所需的 dot、batch matmul、elementwise、broadcast、reduce、reshape、
  transpose、slice、concat、softmax、RMSNorm / LayerNorm、RoPE 和 MLP 激活表达成通用
  IR 结构，而不是 Wafer 私有高层 op。
- 明确 softmax、RMSNorm、LayerNorm 在本层输入里通常已经由 frontend / StableHLO 表示为
  `stablehlo.reduce`、`stablehlo.broadcast_in_dim`、elementwise、shape op 等细粒度 staged graph；
  本层不寻找 `stablehlo.softmax`、`stablehlo.norm` 或 `wafer.softmax` / `wafer.norm` 这种高层 op。
- 为 `wafer.group` planner 和 per-op tiling interface 提供稳定输入。

非目标：

- 不决定 group boundary、traversal tile shape 或 internal split。
- 不表达 physical `mem_layout`、SPM offset、DDR buffer object、DTE resource、packet field、
  worker id 或 C ABI call。
- 不引入 `wafer.softmax`、`wafer.layer_norm`、`wafer.rope` 这类普通 tensor 语义 op 作为长期
  架构边界。需要 pattern 时使用 rewrite / canonicalization，把它们展开到结构化 IR。
- 不把 transformer block 的某个 shape、head 数或隐藏维度写成协议。
- 不把 sharding 表示成 `wafer.spmd.*`，也不把 tensor-level collective 提前 lower 成
  `wafer.comm` / `tile_buffer` / DTE op。

## 2. 输入和输出

输入：

```text
StableHLO local program
  = partitioned StableHLO local shard
    or replicated / single-tile local body from default SPMD policy
  + ConstantLike tensor values
  + optional StableHLO logical collective ops produced by SPMD partitioning
```

输出：

```text
func + tensor + linalg + scf + arith + math
  + Wafer LinalgExt-style tensor collective ops
  + explicit shape/indexing/broadcast/reduction structure
  + ConstantLike tensor values
```

输出 IR 应该只包含 tensor-level 数学语义、structured compute 语义和 logical collective 语义。
Wafer target facts 只能作为后续 legality / cost input。

no-user-sharding 的默认 seed policy 属于 SPMD 阶段，不属于本阶段。本阶段只消费 SPMD 之后的
local body；如果默认 policy 选择 `tile-count=1` 或 replicated fallback，本阶段看到的可能是
whole-shape local body 且没有 collective。缺少 collective 不能作为拒绝 local compute lowering 的理由。

本阶段的主 pipeline 是 `wafer-lower-stablehlo-to-linalg`。它只负责 local compute normalization：
`stablehlo.reduce` -> `linalg.reduce`，StableHLO elementwise -> `linalg.generic` + `arith` /
`math`，shape-only ops -> `tensor` / `linalg` shape ops，`stablehlo.constant` -> `arith.constant`。
它不执行 Shardy propagation，不调用 XLA SPMD partitioner，不写 per-rank parameter shard binding，
也不决定 group / tile / SPM / DDR / C ABI。

## 3. Transformer Block Coverage

跑通一个静态 transformer block 至少需要下面这些 local compute 形态。这里列的是
normalization 输出应能表达的 IR 结构，不是 group 边界或硬件实现承诺。

| 子结构 | Normalized IR | 后续主要 owner |
| --- | --- | --- |
| QKV / output projection / MLP GEMM | `linalg.matmul`、`linalg.batch_matmul` 或等价 structured generic | group + compute GEMM |
| QK^T / attention value matmul | batch/head 维保留在 type/indexing map 中，transpose 是显式 indexing / shape-only 关系 | group + compute GEMM |
| bias / residual / scale | `linalg.generic` + `arith`，带可证明 broadcast | group + compute elementwise |
| RMSNorm | square、reduce sum/mean、rsqrt、mul、scale 的 staged tensor IR | group + compute reduce/elementwise |
| LayerNorm | reduce mean、sub、square、variance、rsqrt、scale/bias 的 staged tensor IR | group + compute reduce/elementwise |
| softmax | row max、subtract、exp、row sum、divide 的 staged tensor IR | group staged schedule + compute reduce/elementwise |
| causal / padding mask | compare/select 或 mask add 的 explicit tensor IR；large negative constant 是 ordinary constant | compute elementwise / mask verifier |
| RoPE | split/slice/concat/neg/mul/add，sin/cos table 作为 ConstantLike 或 explicit `math.sin/cos` source | tensor canonicalization + elementwise |
| SiLU / GELU | decomposition using sigmoid/tanh/erf/exp 或 accepted approximation | compute elementwise subset |
| shape views | `tensor.expand_shape`、`tensor.collapse_shape`、`tensor.extract_slice`、`tensor.insert_slice`、transpose-like indexing | group tiling + layout later |

如果某个 frontend pattern 只能靠 op 名、参数名或模型层名字识别，不能进入长期 lowering。应通过
StableHLO op semantics、types、indexing maps、SSA use-def 和 verifier 可证明的 relation 恢复。

## 4. StableHLO 到 Structured IR 合同

### 4.1 Dot and Batch Matmul

`stablehlo.dot_general` lowering 必须显式保留：

- contracting dimensions。
- batch dimensions。
- lhs/rhs transpose 或 permutation relation。
- output shape and dtype。
- accumulator / result dtype policy if it affects semantics。

简单 2D dot 可以 lowering 到 `linalg.matmul`。带 batch/head 维的 dot 可以 lowering 到
`linalg.batch_matmul` 或 `linalg.generic`，只要 indexing map 和 iterator type 可由 verifier
检查。QK^T 不应该靠 `rhs` 名字识别 transpose；transpose relation 来自 dot dimension numbers
或显式 `transpose` / indexing map。

当前 P5.3 实现扩展 `--wafer-lower-stablehlo-dot`，支持 attention score 的 QK^T 形态：

```text
query : tensor<BxHxQxD>
key   : tensor<BxHxKxD>
score : tensor<BxHxQxK>
```

该 lowering 只接受 StableHLO `dot_general` 中可验证的 dimension numbers：`lhs/rhs` batch
dimensions 都是 `[0, 1]`，contracting dimension 都是最后一维 `[3]`，并检查静态 shape 和
输出 shape 一致。输出使用 `linalg.generic` contraction：

- `query` map: `(b, h, q, k, d) -> (b, h, q, d)`。
- `key` map: `(b, h, q, k, d) -> (b, h, k, d)`。
- `score` map: `(b, h, q, k, d) -> (b, h, q, k)`。
- iterator types: `b/h/q/k` 是 parallel，`d` 是 reduction。

这里的 transpose relation 完全来自 `dot_general` dimension numbers 和 indexing map，不依赖
参数名、函数名或模型层名字。更宽泛的 batch matmul 形态应继续按同一原则扩展，不把某个
head 数、sequence length 或隐藏维写成协议。

当前 P5.4 实现继续扩展 `--wafer-lower-stablehlo-dot`，支持 attention value 的 AV 形态：

```text
prob  : tensor<BxHxQxK>
value : tensor<BxHxKxD>
out   : tensor<BxHxQxD>
```

该 lowering 接受 `lhs/rhs` batch dimensions `[0, 1]`、`lhs` contracting dimension `[3]` 和
`rhs` contracting dimension `[2]`，并检查输出 shape 与 `B/H/Q/D` 对齐。输出使用
`linalg.generic` contraction：

- `prob` map: `(b, h, q, d, k) -> (b, h, q, k)`。
- `value` map: `(b, h, q, d, k) -> (b, h, k, d)`。
- `out` map: `(b, h, q, d, k) -> (b, h, q, d)`。
- iterator types: `b/h/q/d` 是 parallel，`k` 是 reduction。

配套测试覆盖 softmax staged output 作为 AV lhs operand 的 SSA use-def 链。这里验证的是
local structured tensor IR 中的 dataflow 和结果 lifetime 边界；SPM residency、workspace
materialization 和 physical buffer lifetime 仍由后续 group/resource planner 在 Wafer IR 层表达。

### 4.2 Elementwise and Broadcast

elementwise lowering 使用 `linalg.generic` + `arith` / `math`。Broadcast 必须由 indexing map
或 `tensor.expand_shape` / `stablehlo.broadcast_in_dim` 的 normalized relation 表达。

Transformer block 第一阶段需要的 elementwise kind 至少包括：

- add、sub、mul、div。
- max、min、neg、recip、sqrt、rsqrt。
- exp。
- compare/select 或可验证 mask-add 形式。
- sigmoid 或 tanh，如果 SiLU / GELU 选择该 decomposition。

这些 op 在 local tensor IR 中仍是普通 `arith` / `math` / `linalg` 语义；是否能 lower 到 CT
wrapper、是否需要拆成多个 target op，由 `wafer.compute` 负责。

历史 P5.5 曾用 output projection + residual vertical slice 的 acceptance checker 证明以下 SSA 链
可由 structured tensor IR 表达：

- rank-2 `linalg.matmul` 作为 projection。
- 消费 projection 结果的 rank-2 / rank-1 broadcast add 作为 bias add。
- 消费 bias add 结果的 rank-2 / rank-2 add 作为 residual add。

该 checker 已删除；当前只保留 frontend lowering fixture 覆盖 StableHLO 2D projection dot、bias
broadcast 和 residual add 进入 `linalg.matmul` / `linalg.generic`。不引入 `wafer.projection`
或 fused residual op，也不把 bias/residual tensor 名写成语义来源。

历史 P5.6 曾用 MLP vertical slice 的 acceptance checker 证明以下 SSA 链可由 structured tensor
IR 表达：

- rank-2 projection `linalg.matmul` 的结果进入 `linalg.elementwise<tanh>` activation。
- activation 结果与另一个 rank-2 projection matmul 结果进入 `linalg.elementwise<mul>` gate。
- gated activation 结果进入 rank-2 down projection `linalg.matmul`。

该 checker 已删除；当前只保留 frontend lowering fixture 覆盖已有 tanh activation 子集。GELU /
SwiGLU 的其它 decomposition 需要通过通用 structured tensor lowering 和后续 group/materialization
验证扩展，不能再新增 case-specific schedule checker。当前不引入 `wafer.mlp`、fused activation op
或名字约定，也不把中间 tile/group split 写成 IR attr。

当前保留 full local transformer block structured IR fixture。该 integration test 在单个函数中
串联：

- rank-4 RMSNorm staged form。
- QK^T attention score、softmax 和 AV attention value。
- rank-4 到 rank-2 的 shape-only collapse，用于 output projection 和 MLP。
- output projection + bias + residual。
- MLP gate/up/down projection。

该 fixture 只验证这些 fine-grained StableHLO dataflow 经过 local tensor normalization 后仍能由
`linalg` / `tensor` / `arith` / `math` structured IR 表达；不运行 transformer-specific schedule
acceptance pass，也不证明 group schedule、SPM residency、workspace 或 package completion。为支持
该 fixture，`--wafer-lower-stablehlo-shape` 现在支持静态连续维度 reassociation 的
expand/collapse reshape，例如
`tensor<BxHxQxD> -> tensor<(BHQ)xD>`。该批次仍不声称 physical layout、SPM residency、
workspace 或 DDR binding 已完成；这些事实必须在后续 Wafer group/resource lowering 和
transformer local compile gate 中 materialize。

### 4.3 Reduction

reduction 必须保留：

- reduction dimensions。
- init value。
- reduction kind。
- output dtype。
- NaN / overflow / approximate math policy if frontend semantics requires it。

RMSNorm、LayerNorm 和 softmax 都不能被 normalization 压成 opaque high-level op。它们应展开成
reduce + elementwise 的 staged tensor IR，使 group planner 可以决定是否放在一个 group 中、
是否拆成多个 groups、以及 reduction axis 是否需要 internal split。
这里的“展开”指输入 StableHLO graph 已经是细粒度 op 链，或由 frontend canonicalization 变成
细粒度 op 链；Wafer 当前 lowering 只是把这些细粒度 StableHLO op 转成结构化 tensor IR，不新增
`wafer.softmax`、`wafer.norm` 或模型层级语义。

### 4.4 Shape-only Ops

reshape、transpose、slice、concat、split、expand/collapse 这类 op 在本阶段优先保持为
shape/indexing relation。只有当后续 layout / memory / hardware lowering 需要真实 movement 时，
才在 `wafer.tile_region` / layout materialization 阶段生成 movement op。

Normalization 不能因为目标硬件偏好提前插入 ChannelNorm、DechannelNorm、GatherScatter、
RDMA/WDMA 或 TDMA。

### 4.5 Tensor Collective Normalization

StableHLO collective 不适合直接混在 Linalg tiling 主链路中，也不应在本阶段直接 lower 到
`wafer.comm`。本阶段新增一层 Wafer LinalgExt-style tensor collective IR：它不是 sharding
表示，不是 `wafer.spmd`，也不是 tile-local communication op；它是 post-SPMD partitioned
StableHLO collective 的 tensor-level handoff。

该层采用类似 IREE `LinalgExt` 的工程模式：定义 Wafer 自己的 tensor collective ops，并由 op
interface 暴露统一的 collective facts 和 tiling contract。R2.4 的落地边界是
destination-style tensor op + MLIR `TilingInterface` + `WaferTilingInterface` +
`WaferTensorCollectiveOpInterface`；它必须让 group / tiling 边界能直接发现 collective 的 tensor
operand、destination、result、rank group、axis/slot、combiner 和 communication effect。后续
R3/R6 在 tile shape、SPM buffer 和 placement 明确后，再把 tiled tensor collective materialize
到 `wafer.comm` 或 explicit p2p schedule。

R2.4 V0 interface contract：

- `DestinationStyleOpInterface`：使用 `ins(...) outs(...)` 表达输入和 destination-style 输出，
  结果 tensor 与 outs 一一对应。
- `WaferTilingInterface`：暴露 input / output / result tiling demand，供 `wafer.group` 和后续 group
  planner 统一消费；不得返回 planner-local side table。
- MLIR `TilingInterface`：暴露 result-space iteration domain、parallel iterator 类型、tiled
  implementation 和 result tile position。`all_reduce` / `collective_permute` 支持 shape-preserving
  tile clone；`all_gather` / `reduce_scatter` / `all_to_all` 只在当前 IR 能证明 collective 轴完整覆盖时
  生成 tiled op，slot-crossing 或动态不可证明的 tile 返回 failure，要求 planner 拆成 slot-aligned
  tile 或延后到 R3/R6。
- `WaferTensorCollectiveOpInterface`：暴露 collective kind、rank group、source-target pairs、
  axis/split/concat/split_count、channel、reduction combiner 和 communication effect。
- verifier：检查 shape、rank、dtype、axis、rank group、slot mapping、reduction body 和禁止
  physical tile / SPM / DTE / runtime metadata。

概念层面的 ops 包括：

- `wafer.tensor_collective.all_reduce`
- `wafer.tensor_collective.all_gather`
- `wafer.tensor_collective.reduce_scatter`
- `wafer.tensor_collective.all_to_all`
- `wafer.tensor_collective.collective_permute`

R2.4 ODS 落地命名采用 `wafer.tensor_collective.*` op family。长期合同是
LinalgExt-style tensor collective 层；`tensor_collective` 是 Wafer dialect 内的 op-family
前缀，不是新的 sharding dialect、physical communication dialect 或 runtime protocol。

各 collective 的 tile 关系：

- `all_reduce`：input tile 和 result tile 同 shape、同 offset；combiner region 保留在 tensor
  collective 层，后续 lowering 再判断可否映射到 `wafer.compute`。
- `all_gather`：沿 gather dimension 按 rank slot concat。result tile 不能隐式跨 slot；若 tile
  覆盖多个 rank slot，tiling 必须拆成多个 slot-aligned tiled collective 或由 planner 选择
  slot-aligned tile。
- `reduce_scatter`：result tile 对应当前 logical rank 的 scatter slot；input demand 由 scatter
  dimension、rank group 和 result tile 反推。
- `all_to_all`：同时表达 split dimension 和 concat dimension 的 slot 映射；每个 slot 的
  source/destination slice 必须可验证。
- `collective_permute`：tensor slice shape 不变；source-target pair 是 communication effect 和
  placement input，不在本层选择 physical peer。

该层输出仍是 tensor IR，可以和 `linalg.matmul`、`linalg.generic`、`linalg.reduce` 等一起进入
`wafer.group`。例如：

```mlir
%mm = linalg.matmul ins(%a, %b : tensor<...>, tensor<...>)
      outs(%init : tensor<...>) -> tensor<...>
%red = wafer.tensor_collective.all_reduce ins(%mm : tensor<...>)
       outs(%init : tensor<...>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    wafer.tensor_collective.yield %sum : f32
} -> tensor<...>
%y = linalg.generic ... ins(%red : tensor<...>) ...
```

这里的 `wafer.tensor_collective.*` 不拥有 physical placement、SPM buffer 或 DTE resource。到
`wafer.tile_region` / SPM materialization 之后，tiled tensor collective 才会变成 `wafer.comm.*`
或 explicit p2p schedule。

R2.4 pipeline position：

```text
Pipeline position:
- Upstream artifact / IR: P2.S2 partitioned / replicated-local StableHLO bundle，含 rank-local function signature、StableHLO collective op、replica_groups、channel metadata 和 rank-local parameter payload。
- Current stage responsibility: 把 post-SPMD StableHLO logical collective normalize 成 Wafer-owned destination-style tensor collective op，并保留 group/tiling 可验证的 tensor-level collective facts。
- Output artifact / IR: `linalg` / `tensor` / `scf` local compute IR 加 `wafer.tensor_collective.*` ops；不含 `wafer.comm`、SPM tile buffer、DTE token 或 runtime handle。
- Downstream consumer: R3 group candidate / tiling，以及 R6 tiled tensor collective -> `wafer.comm` materialization。
- User-level driver / named pipeline: `wafer-lower-stablehlo-to-linalg` 消费 P2.S2 输出的 rank-local StableHLO module；collective normalization 是该 named pipeline 的一部分，不要求用户手动拼 pass。
- Explicit non-goals: 不恢复 `wafer.spmd.*` 私有协议，不把 StableHLO collective 直接 lower 到 `wafer.comm`，不在 R2.4 选择 physical peer、ring schedule、SPM/DDR buffer 或 packet/runtime ABI。
- Completion gate: 真实 P2.S2 artifact 中的 StableHLO collective 经 named pipeline 变成 verifier-legal `wafer.tensor_collective.*` op；这些 op 实现 `DestinationStyleOpInterface`、MLIR `TilingInterface`、`WaferTilingInterface` 和 `WaferTensorCollectiveOpInterface`，输出可被 group 边界作为 tensor-level IR 消费；fixture/FileCheck/gtest 只做补充覆盖。
```

## 5. Softmax and Norm Staged Form

本节描述的是 structured tensor dataflow，不是 StableHLO dialect 中存在一个高层 softmax/norm op。
当前测试里的输入 StableHLO 已经是 `stablehlo.reduce`、`stablehlo.subtract`、`stablehlo.exponential`、
`stablehlo.divide`、`stablehlo.rsqrt`、`stablehlo.broadcast_in_dim` 等细粒度算子。Wafer lowering
把这些 op 分别转成 `linalg.reduce`、`linalg.generic`、`arith`、`math` 和 `tensor`/`linalg` shape
ops。后续 softmax/norm 是否能进入合法 tile/group schedule，必须由通用 group/resource planner 和
materialization verifier 从 structured IR 与 SSA use-def 关系重算，不能再由 case-specific
acceptance checker 给出结论。

Softmax 的 normalized form 至少是：

```text
scores = matmul(Q, K^T) * scale + mask
row_max = reduce_max(scores, key_dim)
shifted = scores - row_max
exp_scores = exp(shifted)
row_sum = reduce_sum(exp_scores, key_dim)
prob = exp_scores / row_sum
```

这只是数学/dataflow 结构，不是要求一个 `wafer.group` 覆盖整条链。若 key dimension 无法在一个
tile schedule 内合法覆盖，group planner 必须拆成多阶段 schedule，例如 row max stage、row sum
stage、normalize/value stage，并通过 DDR workspace 或 tile-local loop-carried state 明确表达
中间结果。

RMSNorm / LayerNorm 类似：

```text
rms = rsqrt(mean(x * x, hidden_dim) + eps)
y = x * rms * weight
```

或：

```text
mean = reduce_mean(x, hidden_dim)
var = reduce_mean((x - mean) * (x - mean), hidden_dim)
y = (x - mean) * rsqrt(var + eps) * weight + bias
```

epsilon、scale、bias 都是普通 constant / input value。它们不通过名字或 layer type 特判。

历史 P5.1 曾用 norm vertical slice 的 schedule acceptance checker 从 structured tensor IR 重算
以下事实：

- 是否存在沿最后一维的 `linalg.reduce`。
- 是否存在 `rsqrt` elementwise stage。
- 是否存在 rank-N / rank-(N-1) broadcast multiply stage。

该 checker 已删除；当前 coverage 只来自 StableHLO -> Linalg lowering fixture。真正 tile shape、
SPM residency 和 group split 仍归后续 group/resource planner，不能再靠 case-specific checker
冒充 schedule 完成。

历史 P5.2 曾用 softmax vertical slice 的 schedule acceptance checker 从 structured tensor IR 和
SSA use-def 重算以下事实：

- 是否存在沿最后一维的 reduce max，作为 row/key 维最大值阶段。
- 是否存在 `scores - row_max` 的 broadcast subtract。
- 是否存在消费 subtract 结果的 `exp` elementwise stage。
- 是否存在消费 exp 结果、沿同一最后一维的 reduce sum。
- 是否存在 `exp_scores / row_sum` 的 broadcast divide normalize。

该 checker 已删除。当前不引入 `wafer.softmax`，不写入 schedule attr，也不把 group split 或
workspace 选择固化成 IR 合同。若后续需要 multi-stage softmax，row max、row sum、normalize/value
的分割应由 group/resource planner 通过显式 IR 边界和 workspace demand materialize。

## 6. Pass 合同

实现上可以拆成这些职责：

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| StableHLO legalize to structured tensor | partitioned local shard or replicated/single-tile local body | `linalg` / `tensor` / `arith` / `math` / `scf` |
| StableHLO collective normalization | optional partitioned StableHLO collectives | Wafer LinalgExt-style tensor collective ops |
| dot_general normalization | StableHLO dot | matmul / batch_matmul / structured generic |
| broadcast/reduction normalization | StableHLO broadcast/reduce | explicit indexing maps and reduce dims |
| transformer pattern canonicalization | softmax/norm/RoPE/MLP activation patterns | staged structured tensor IR |
| shape-view cleanup | reshape/transpose/slice chains | canonical shape/indexing relation |

这些 pass 可以使用 pattern rewrite、canonicalization 和 dialect conversion，但 pass pipeline 不承载
隐藏语义。若一个 op 不能被 normalized 到 verifier 可解释的结构，应保留在上游 dialect 并报
明确 diagnostic，不能通过名字 fallback。这里的 diagnostic 只说明当前 normalization 边界还没有
可验证表示；如果该语义能由 StableHLO / structured tensor IR 和 Wafer 硬件能力表达，后续任务应
扩展 normalized IR、verifier 或 lowering，而不是把当前 pattern 覆盖范围写成长期不支持。

## 7. Verifier

Normalization 后必须能检查：

- function boundary shape、rank、dtype 和 bounded dynamic shape。
- dot / batch matmul 的 contracting、batch、transpose relation。
- broadcast indexing relation。
- reduction dimensions、init value 和 output dtype。
- softmax/norm staged IR 中的 reduction axis 和 elementwise consumer 关系。
- shape-only ops 没有提前变成 target movement。
- StableHLO collectives 已经规整成 tensor-level collective ops，且这些 ops 不含 `wafer.spmd.*`、
  `wafer.comm`、`tile_buffer`、DTE 或 runtime metadata。
- IR 中没有 Wafer physical memory、layout materialization、DTE、packet 或 runtime launch 事实。

### 7.1 R2.3 覆盖状态口径

2026-05-26 R2.3 已把当前 local compute normalization 覆盖状态改写为 structured tensor IR 证据，
不再把 acceptance pass 视为 schedule completion。当前可引用的 evidence 如下：

| 子结构 | 当前证据 | 结论边界 |
| --- | --- | --- |
| dot / 2D GEMM | `test/Frontend/lower-stablehlo-dot-to-linalg.mlir`、`stablehlo-dot-artifact.mlir`、`linalg-gemm-artifact.mlir` | 证明 2D dot 可进入 structured matmul，不证明 tile shape / GEMM packet |
| attention QK^T / AV | `lower-stablehlo-attention-score.mlir`、`lower-stablehlo-attention-value.mlir`、`lower-stablehlo-attention-softmax-value.mlir` | rank-4 attention dot 当前不再特判 lowering；测试确认它保留为 StableHLO `dot_general`，后续需要通用 batched contraction / tensor collective handoff 合同 |
| elementwise / broadcast | `lower-stablehlo-elementwise.mlir`、`lower-stablehlo-projection-residual.mlir` | 证明当前 add/sub/mul/div/tanh/exp/broadcast 子集的 SSA dataflow；mask/select 和 complex broadcast 未闭环 |
| reduce | `lower-stablehlo-reduce.mlir`、norm/softmax staged tests | 证明细粒度 StableHLO reduce 到 `linalg.reduce` 的 constant-init 子集；non-constant-init reduce 和 numeric policy 未闭环 |
| softmax | `lower-stablehlo-softmax-staged.mlir` | 证明 fine-grained StableHLO softmax dataflow 可变成 `linalg.reduce` / `linalg.generic` staged IR；不证明 multi-stage workspace 或 group schedule |
| norm | `lower-stablehlo-norm-staged.mlir` | 证明 fine-grained RMSNorm/LayerNorm dataflow 中 last-dim reduce / rsqrt / broadcast multiply gate；不证明完整 LayerNorm/RMSNorm family |
| RoPE | `lower-stablehlo-rope-mlp-staged.mlir` | 证明当前 RoPE slice/shape/elementwise staged pattern；sin/cos table storage slicing 未闭环 |
| MLP | `lower-stablehlo-mlp.mlir`、`lower-stablehlo-local-transformer-block.mlir` | 证明 tanh-gated MLP dataflow fixture 和 full local transformer structured fixture；GELU/SwiGLU/package consistency 未闭环 |
| shape views | `lower-stablehlo-shape.mlir`、`lower-stablehlo-local-transformer-block.mlir` | 证明 static expand/collapse shape-only relation；dynamic shape view 和 layout materialization 未闭环 |
| tensor collective handoff | R2.4 已恢复 | `wafer-lower-stablehlo-to-linalg` 已把 StableHLO logical collective normalize 成 `wafer.tensor_collective.*`；旧的 StableHLO -> `wafer.comm` bridge 已移除，不能作为 group/tiling 输入 |

## 8. 与其它文档的关系

- Frontend 文档负责 artifact 和 constant normalization 的入口。
- Shardy / SPMD 文档负责用户 sharding 和 no-user-sharding 默认 policy 生成的 SDY seed、
  partitioned / replicated-local StableHLO 和 StableHLO logical collective。
- 本文负责 SPMD 后 local body 内的 structured tensor IR，以及 StableHLO collective 到 Wafer
  LinalgExt-style tensor collective IR 的 handoff。
- Group 文档负责 tile-local residency、traversal schedule 和 closed-loop resource search。
- Compute 文档负责把 selected tensor op lower 到 target-abstract `wafer.compute`。
- Layout / SPM / DDR 文档负责 physical layout、bufferization 和 resource legality。

参考 MLIR 官方方向：Linalg structured ops 提供 tiling/fusion 所需的 indexing/interface 基础；
Bufferization 负责从 tensor 语义到 memref 语义的转换；Dialect Conversion 提供合法性驱动的
op conversion 框架。Wafer 复用这些机制，但目标硬件资源和 layout 约束由自己的下游 IR 表达。

## 9. 参考材料

- MLIR Linalg dialect: <https://mlir.llvm.org/docs/Dialects/Linalg/>
- MLIR TilingInterface: <https://llvm.googlesource.com/llvm-project/mlir/+/main/include/mlir/Interfaces/TilingInterface.td>
- MLIR DestinationStyleOpInterface: <https://llvm.googlesource.com/llvm-project/mlir/+/main/include/mlir/Interfaces/DestinationStyleOpInterface.td>
- IREE LinalgExt reference: <https://iree.dev/reference/mlir-dialects/LinalgExt/>
- MLIR Bufferization: <https://mlir.llvm.org/docs/Bufferization/>
- MLIR Dialect Conversion: <https://mlir.llvm.org/docs/DialectConversion/>
