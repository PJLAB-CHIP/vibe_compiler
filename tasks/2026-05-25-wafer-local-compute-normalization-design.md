# Wafer Local Compute Normalization Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口

本文定义 SPMD partition 之后、`wafer.group` 之前的 local tensor compute normalization
边界。该阶段负责把 partitioned StableHLO 的本地 shard 程序规整到可 tile、可 fuse、可
验证的 `linalg` / `tensor` / `scf` / `arith` / `math` IR 子集。它不引入 Wafer physical
layout、SPM/DDR allocation、DTE、C ABI 或 runtime package。

本文依赖：

- `tasks/2026-05-25-wafer-frontend-stablehlo-artifact-design.md`
- `tasks/2026-05-25-wafer-shardy-spmd-design.md`
- `tasks/2026-05-12-wafer-group-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- MLIR Linalg / Bufferization / Dialect Conversion 官方文档。

## 1. 目标和非目标

目标：

- 把 StableHLO local shard compute lowering 到结构化 tensor IR。
- 保留 op 的 indexing map、iterator type、DPS operand/result 关系、shape、dtype 和
  broadcast / reduction 语义。
- 把 transformer block 所需的 dot、batch matmul、elementwise、broadcast、reduce、reshape、
  transpose、slice、concat、softmax、RMSNorm / LayerNorm、RoPE 和 MLP 激活表达成通用
  IR 结构，而不是 Wafer 私有高层 op。
- 为 `wafer.group` planner 和 per-op tiling interface 提供稳定输入。

非目标：

- 不决定 group boundary、traversal tile shape 或 internal split。
- 不表达 physical `mem_layout`、SPM offset、DDR buffer object、DTE resource、packet field、
  worker id 或 C ABI call。
- 不引入 `wafer.softmax`、`wafer.layer_norm`、`wafer.rope` 这类普通 tensor 语义 op 作为长期
  架构边界。需要 pattern 时使用 rewrite / canonicalization，把它们展开到结构化 IR。
- 不把 transformer block 的某个 shape、head 数或隐藏维度写成协议。

## 2. 输入和输出

输入：

```text
partitioned StableHLO local shard
  + ConstantLike tensor values
  + logical collective boundary already kept as StableHLO collective or later wafer.comm source
```

输出：

```text
func + tensor + linalg + scf + arith + math
  + explicit shape/indexing/broadcast/reduction structure
  + ConstantLike tensor values
```

输出 IR 应该只包含上游 tensor dialect 能解释的数学语义。Wafer target facts 只能作为后续
legality / cost input。

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

当前 P5.5 实现用 `--wafer-check-projection-residual-schedule` 作为 output projection + residual
vertical slice 的 acceptance gate。该 pass 只从当前 structured tensor IR 重算以下 SSA 链：

- rank-2 `linalg.matmul` 作为 projection。
- 消费 projection 结果的 rank-2 / rank-1 broadcast add 作为 bias add。
- 消费 bias add 结果的 rank-2 / rank-2 add 作为 residual add。

它不引入 `wafer.projection` 或 fused residual op，也不把 bias/residual tensor 名写成语义来源。
frontend integration test 覆盖 StableHLO 2D projection dot、bias broadcast 和 residual add 到该
acceptance gate 的完整 lowering。

当前 P5.6 实现用 `--wafer-check-mlp-schedule` 作为 MLP vertical slice 的 acceptance gate。
该 pass 只从当前 structured tensor IR 重算以下 SSA 链：

- rank-2 projection `linalg.matmul` 的结果进入 `linalg.elementwise<tanh>` activation。
- activation 结果与另一个 rank-2 projection matmul 结果进入 `linalg.elementwise<mul>` gate。
- gated activation 结果进入 rank-2 down projection `linalg.matmul`。

当前实现覆盖已有 frontend 支持的 tanh activation 子集；GELU/SwiGLU 的其它 decomposition 可在
同一 use-def 检查上扩展。该 gate 不引入 `wafer.mlp`、fused activation op 或名字约定，也不把
中间 tile/group split 写成 IR attr。

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

### 4.4 Shape-only Ops

reshape、transpose、slice、concat、split、expand/collapse 这类 op 在本阶段优先保持为
shape/indexing relation。只有当后续 layout / memory / hardware lowering 需要真实 movement 时，
才在 `wafer.tile_region` / layout materialization 阶段生成 movement op。

Normalization 不能因为目标硬件偏好提前插入 ChannelNorm、DechannelNorm、GatherScatter、
RDMA/WDMA 或 TDMA。

## 5. Softmax and Norm Staged Form

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

当前 P5.1 实现用 `--wafer-check-norm-schedule` 作为 norm vertical slice 的 schedule acceptance
gate。该 pass 只从当前 structured tensor IR 重算以下事实：

- 是否存在沿最后一维的 `linalg.reduce`。
- 是否存在 `rsqrt` elementwise stage。
- 是否存在 rank-2 / rank-1 broadcast multiply stage。

它不写入 group attr、side table、planner trace 或 cost 分数。失败时诊断指向缺失的 staged
结构；真正 tile shape、SPM residency 和 group split 仍归后续 group/resource planner。

当前 P5.2 实现用 `--wafer-check-softmax-schedule` 作为 softmax vertical slice 的 schedule
acceptance gate。该 pass 只从当前 structured tensor IR 和 SSA use-def 重算以下事实：

- 是否存在沿最后一维的 reduce max，作为 row/key 维最大值阶段。
- 是否存在 `scores - row_max` 的 broadcast subtract。
- 是否存在消费 subtract 结果的 `exp` elementwise stage。
- 是否存在消费 exp 结果、沿同一最后一维的 reduce sum。
- 是否存在 `exp_scores / row_sum` 的 broadcast divide normalize。

它不引入 `wafer.softmax`，不写入 schedule attr，也不把 group split 或 workspace 选择固化成
IR 合同。若后续需要 multi-stage softmax，row max、row sum、normalize/value 的分割应由
group/resource planner 通过显式 IR 边界和 workspace demand materialize。

## 6. Pass 合同

实现上可以拆成这些职责：

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| StableHLO legalize to structured tensor | partitioned StableHLO | `linalg` / `tensor` / `arith` / `math` / `scf` |
| dot_general normalization | StableHLO dot | matmul / batch_matmul / structured generic |
| broadcast/reduction normalization | StableHLO broadcast/reduce | explicit indexing maps and reduce dims |
| transformer pattern canonicalization | softmax/norm/RoPE/MLP activation patterns | staged structured tensor IR |
| shape-view cleanup | reshape/transpose/slice chains | canonical shape/indexing relation |

这些 pass 可以使用 pattern rewrite、canonicalization 和 dialect conversion，但 pass pipeline 不承载
隐藏语义。若一个 op 不能被 normalized 到 verifier 可解释的结构，应保留在上游 dialect 并报
明确 unsupported diagnostic，不能通过名字 fallback。

## 7. Verifier

Normalization 后必须能检查：

- function boundary shape、rank、dtype 和 bounded dynamic shape。
- dot / batch matmul 的 contracting、batch、transpose relation。
- broadcast indexing relation。
- reduction dimensions、init value 和 output dtype。
- softmax/norm staged IR 中的 reduction axis 和 elementwise consumer 关系。
- shape-only ops 没有提前变成 target movement。
- IR 中没有 Wafer physical memory、layout materialization、DTE、packet 或 runtime launch 事实。

## 8. 与其它文档的关系

- Frontend 文档负责 artifact 和 constant normalization 的入口。
- Shardy / SPMD 文档负责 global sharding 和 logical collective。
- 本文负责 local shard 内的 structured tensor IR。
- Group 文档负责 tile-local residency、traversal schedule 和 closed-loop resource search。
- Compute 文档负责把 selected tensor op lower 到 target-abstract `wafer.compute`。
- Layout / SPM / DDR 文档负责 physical layout、bufferization 和 resource legality。

参考 MLIR 官方方向：Linalg structured ops 提供 tiling/fusion 所需的 indexing/interface 基础；
Bufferization 负责从 tensor 语义到 memref 语义的转换；Dialect Conversion 提供合法性驱动的
op conversion 框架。Wafer 复用这些机制，但目标硬件资源和 layout 约束由自己的下游 IR 表达。

## 9. 参考材料

- MLIR Linalg dialect: <https://mlir.llvm.org/docs/Dialects/Linalg/>
- MLIR Bufferization: <https://mlir.llvm.org/docs/Bufferization/>
- MLIR Dialect Conversion: <https://mlir.llvm.org/docs/DialectConversion/>
