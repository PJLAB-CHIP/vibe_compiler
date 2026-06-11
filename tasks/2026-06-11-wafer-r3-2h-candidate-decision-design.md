# Wafer R3.2h Candidate Decision Design

本文以 R3.2h 任务索引记录 candidate-selection 边界。candidate-selection 不是新的 memory
allocator，也不是把失败计划写进 IR 等后段修复的阶段；它负责枚举 tile/lowering 候选，逐个
重放 candidate DDR tile-view materialization、instruction lowering、SPM offset assignment、
DDR offset assignment 和 verifier gates，只把已经通过全部 gates 的 candidate artifact 交给 R3.3。

## 1. Goal and Non-Goals

目标：

- 根据实际 traversal shape 枚举 bounded tile candidate。
- 把 bounded traversal tile shape、可证明同 traversal domain 的 output coverage，以及当前支持的
  reduction/internal split 转成 candidate evaluation IR；layout choice 只在对应 interface 能表达多个
  合法候选后进入 search space。
- 对每个 candidate 重放 candidate DDR tile-view materialization、instruction lowering、SPM offset
  assignment、DDR offset assignment 和 verifier。
- 默认选择第一个合法 candidate；可通过 `tile-search` 选项改为估算时间最小的合法 candidate。
- 在失败时返回结构化 no-candidate / split-needed reason，不把失败 candidate plan 写回主 IR。

非目标：

- 不发明新的 SPM/DDR allocation 算法。SPM/DDR 容量、range、alignment、bandwidth 等约束属于
  SPM/DDR planning config 和对应 gate。
- 不把 SPM/DDR limit、arena、capacity 或 bandwidth 建模成 candidate 字段。
- 不生成 ABI call、packet、physical address、runtime handle 或 package metadata。
- 不把 search order、rejected candidate、cost breakdown、lifetime trace 或 planner 中间状态写入 IR。
- 不在 candidate-selection 自己恢复 DDR subview、SPM footprint 或 instruction legality；这些必须由
  candidate gates 的当前 IR 和 verifier 重算。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.1 logical wafer.group；R3.2a tiling demand analysis；R3.2b layout planning analysis；
  candidate DDR tile-view producer；instruction lowering；SPM offset assignment；DDR offset assignment。
- Current stage responsibility:
  枚举 candidate spec；为每个 candidate 构造 evaluation clone；按 candidate DDR tile-view
  materialization -> instruction lowering -> SPM offset assignment -> DDR offset assignment -> verifier
  的顺序运行 gates。只接受通过全部 gates 的 candidate。`tile-search=first-legal`
  选择第一个合法 candidate；`tile-search=min-estimated-time` 在合法 candidate 中用粗估时间选最小。
- Output artifact / IR:
  passing candidate artifact，包含 actual DDR tile views、instruction-level IR、accepted SPM offset
  facts 和 accepted DDR offset facts；或 no-candidate / split-needed failure reason。
- Downstream consumer:
  R3.3 只 commit candidate-selection 选出的 passing candidate artifact；
  R3.4/R3.5 消费 accepted SPM/DDR facts 和当前 IR 可重算的 descriptor/view/allocation demand，
  不重新枚举 candidate 或重做 memory planning。
- User-level driver / named pipeline:
  `wafer-select-group-tile` pass 和 `wafer-lower-groups-to-selected-instr` named pipeline；
  driver 内部重放 candidate gates，用户不应手工拼接 tile-view materialization、instruction lowering、
  SPM planning 和 DDR planning pass 作为主线
  compile flow。
- Explicit non-goals:
  不生成 placed memref、runtime allocation/import/query、ABI call、packet 或 physical address；
  不把失败 candidate 的 transient plan 落入 committed main IR。
- Completion gate:
  simple matmul/elementwise group 能由 candidate-selection driver 自动生成候选、运行 candidate gates、按 `tile-search`
  选择 passing candidate；capacity/layout/view/memory failure 能驱动 retry 或给出 no-candidate /
  split-needed reason；completion proof 不依赖手工 fixture 串 pass。
```

## 3. Candidate Model

Candidate 只表达程序如何被切分和 lowering，不表达资源上限。

```text
CandidateSpec:
  traversal_tile_shape
  layout_choice
  internal_split
  output_coverage
```

- `traversal_tile_shape`：当前 group 的输出 traversal domain 上每维 tile size。
- `layout_choice`：来自 layout analysis / layout interface 的候选。当前实现使用 layout analysis /
  lowering 默认给出的 layout 顺序，不在 candidate-selection 中重新发明 `tensor`、`cx`、`ncx` 规则；
  未来如果 layout interface 暴露多个可选 materialization cut，candidate-selection 只把它们作为候选维度。
- `internal_split`：matmul/reduction 这类 op 的内部 reduction split，例如 matmul 的 `K` split 或
  `linalg.generic` reduction iterator split。它不是 traversal domain。
- `output_coverage`：candidate 如何覆盖 result。当前实现支持单输出，以及多个静态 ranked result
  共享同一个 traversal shape、每个 yielded root 都能映射到对应 output boundary 的 multi-output
  coverage。非 reduction root 要求 DPS init 是对应 direct output boundary；reduction root 可以使用
  tile-local fill/init，并由 candidate DDR tile-view materialization 显式生成到对应 output boundary 的
  `tensor.insert_slice` storeback。
  不同 output domain、partial writeback、复杂 scatter coverage 返回结构化 failure；这些需要先扩
  group output coverage interface 和 candidate tile-view materializer，不能用名字或 side table 伪支持。

下面这些不属于 candidate：

- SPM base/limit、reserved range、alignment、bank bound。
- DDR capacity、largest-contiguous、bandwidth、arena 或 runtime binding path。
- SPM/DDR planning 的 allocation result、offset、lifetime、failure trace。
- cost breakdown 或 rejected candidate list。

这些资源约束通过 planner config 传给对应 gate：

```text
SPMPlanningConfig:
  base
  limit
  alignment
  reserved_ranges

DDRPlanningConfig:
  capacity_bytes
  largest_contiguous_bytes
  bandwidth_limit_bytes
  alignment_bytes
```

candidate-selection 不解释这些字段，只在 gate failure 时 reject 当前 candidate 并继续枚举下一个 candidate。

## 4. Search Space

### 4.1 Traversal Domain

Traversal domain 是 outer tile loop 实际遍历的输出坐标空间：

- elementwise `Y[M, N] = f(X[M, N])`：traversal domain 是 `Y[M, N]`。
- matmul `C[M, N] = A[M, K] * B[K, N]`：traversal domain 是 `C[M, N]`；`K` 是 internal split。
- reduce `R[M] = reduce(Y[M, N], axis=N)`：traversal domain 是 `R[M]`；被 reduce 的 `N`
  是内部 reduction 维。

### 4.2 Tile Size Candidate

Tile size 从实际 shape 生成，不能只套固定表。对每个 traversal 维，候选来源：

- full dimension。
- common divisors。
- target preferred tile sizes 中不超过当前维度的值。
- tail-aware size，避免产生特别小的 tail。

例如某维长度是 `1000`，候选可以包括：

```text
1000, 500, 250, 200, 125, 100, 128, 64
```

每维先排序并截断到 bounded 数量。排序原则是确定性的：优先覆盖更大 tile、无 tail 或小 tail
风险低、接近 target preferred size、不会明显增加 candidate 数量。组合后先做 cheap bound
过滤，再运行完整 candidate gates。

### 4.3 Representative Tile Classes

一个 tile shape 不能只验证 `[0, 0, ...]`。candidate-selection 至少为每个 shape 生成代表性 tile class：

- interior tile。
- 每个维度的 tail tile。
- 多维 corner tail tile。
- 如果整除，也至少验证 first tile 和 last/max-offset tile。

只有所有代表 tile class 都通过 candidate gates，这个 candidate 才算合法。否则当前
candidate reject，不能把某个局部 tile 通过当成全 shape 通过。

### 4.4 Internal Split

Internal split 由 op tiling interface 提供。V0 策略：

- matmul 默认先试 no split；只有 no-split traversal candidates 都没有通过时，才枚举更小的 `K`
  tile。
- matmul K split 的 candidate evaluation IR 生成多个 partial GEMM，并用 tile-local elementwise
  add 累加 partial results，最后只 storeback 一次 output tile。
- `linalg.generic` reduction split 使用同一个机制：candidate-selection 从 reduction iterator 的静态 range
  枚举 split size；candidate DDR tile-view materialization 在 candidate evaluation IR 中为每个 reduction chunk 生成 partial reduce，
  再用 tile-local elementwise combine 合并 partial results，最后只 storeback 一次 output tile。
- reduction split 只接受 reducer body 可识别、且 partial init / combine 语义能从当前 IR 验证的
  reduction。当前实现覆盖 sum/max/min 形态，其中 split evaluation 需要能复用合法的 tile-local
  reduction init；不能在 IR 外记一个 loop-carried accumulator 让后段猜语义。
- 在 `tile-search=min-estimated-time` 下，合法的 internal split 也参与估算时间比较。

以 matmul 为例，`K` split 的 footprint 粗估来自：

```text
A tile bytes = tm * k_tile * sizeof(dtype)
B tile bytes = k_tile * tn * sizeof(dtype)
C tile bytes = tm * tn * sizeof(dtype or accumulator dtype)
```

真实合法性仍以 candidate gates 生成和验证后的 IR 为准。

## 5. Gate Runner

每个 candidate 在独立 evaluation clone 上运行：

```text
CandidateSpec
  -> materialize candidate DDR tile views
       fail: reject candidate
  -> lower tile region to instruction IR
       fail: reject candidate
  -> assign SPM offsets
       fail: reject candidate
  -> assign DDR offsets
       fail: reject candidate
  -> verifier
       fail: reject candidate
  -> PassingCandidate
```

失败时立即停止当前 candidate 的后续 gates。SPM 失败不能继续假装 DDR planning 还能补救；DDR
失败也不能回头改 instruction semantics。candidate-selection 只能选择下一个 candidate 或返回 no-candidate /
split-needed reason。

Passing candidate artifact 是 transformation-local artifact。只有 R3.3 会把被选中的 artifact
commit 到主 IR。Rejected candidate IR 必须丢弃。

## 6. `tile-search` Modes

candidate-selection 提供用户可配置的 tile search 模式。命令行 pass option 使用 MLIR 风格拼写
`tile-search`。

```text
--tile-search=first-legal
--tile-search=min-estimated-time
```

### 6.1 `first-legal`

默认模式。按确定性顺序枚举 candidate，遇到第一个完整通过 candidate gates 的 candidate 就接受。
该模式不运行 cost comparison。

排序原则：

- layout choice 使用 layout analysis 提供的顺序。
- tile shape 优先较大覆盖、较少 tail、较少 internal split。
- no split 优先于 split。
- 单输出优先；同 traversal 多输出必须满足 coverage legality。

### 6.2 `min-estimated-time`

估算时间模式。candidate-selection 继续使用同一个 bounded search space，但不会在第一个合法 candidate 停止。
它收集所有通过 candidate gates 的 candidate，对每个合法 candidate 计算粗估时间，选择
`estimated_cycles` 最小者。

cost model 只在合法 candidate 之间排序，不参与 legality，也不能接受一个 gate 失败的 candidate。
cost breakdown 是 diagnostic，不写入 committed IR。

## 7. Rough Time Estimation

V0 cost model 使用硬件参数、计算量、DDR 访存量、SPM/local movement 量和 instruction issue
数量估算时间。参数来自 target policy / pass option；没有板端校准时使用保守默认值。

```text
TileTimeConfig:
  compute_ops_per_cycle
  ddr_bytes_per_cycle
  spm_bytes_per_cycle
  instr_issue_cycles
  assume_ddr_compute_overlap
```

基础估算：

```text
compute_cycles = compute_ops / compute_ops_per_cycle
ddr_cycles     = ddr_bytes / ddr_bytes_per_cycle
spm_cycles     = spm_bytes / spm_bytes_per_cycle
issue_cycles   = instr_count * instr_issue_cycles

if assume_ddr_compute_overlap:
  estimated_cycles = max(compute_cycles, ddr_cycles) + spm_cycles + issue_cycles
else:
  estimated_cycles = compute_cycles + ddr_cycles + spm_cycles + issue_cycles
```

V0 默认不假设 DDR/compute overlap，避免在没有 PMU 校准前过度乐观。后续 PMU/profiling 工作
只校准 latency、blocking time、overlap 和 conflict cost，不改变 candidate-selection 的 IR 合同。

统计来源必须来自 passing candidate 的 lowered IR：

- `ddr_bytes`：从 RDMA/WDMA descriptor、DDR `memref.subview` / strided view 和 WDMA writeback
  统计。
- `spm_bytes`：从 TDMA、gather-scatter、layout materialization、local copy 和 SPM movement
  instruction 统计。
- `compute_ops`：从 `wafer.instr.gemm`、`reduce`、`elementwise`、`fill/convert` 等 op 的 tile shape
  和 op kind 估算。
- `instr_count`：从 instruction-level IR 统计 issue op 数量。

SPM/DDR offset facts 可以用于诊断 high-water、reuse 和容量压力，但不是 candidate 字段，也不是
cost model 接受 candidate 的依据。

## 8. Failure and Diagnostics

candidate-selection 的失败原因分三类：

- `retryable_candidate_failure`：当前 traversal tile / matmul `K` split candidate 失败，可以换下一个
  candidate。
- `split_needed`：当前 group 需要拆分，例如多输出 coverage 不兼容、required movement 还不能表达、
  或所有 bounded tile 都被 memory/movement gate 拒绝。
- `no_candidate`：bounded search space 内没有 passing candidate，且没有更细的 split policy 可用。

diagnostic 应记录：

- 失败发生在哪个 gate。
- candidate spec 的 traversal tile shape、reduction/internal split 和 output coverage；未来 layout
  候选需要等对应 interface 落地后再进入 diagnostic。
- diagnostic gate prefix 使用稳定语义边界名：`tile-region`、`instr-lowering`、`spm-offsets`、
  `ddr-offsets`、`verifier`；不把 `R3.2*` 任务号打印成用户可见诊断。
- SPM/DDR offset assignment gate 的结构化 reason，例如 capacity、largest-contiguous、
  bandwidth、alignment、descriptor/view mismatch。
- `min-estimated-time` 模式下 winning candidate 的 cost breakdown。

diagnostic 不成为 IR 合同；R3.3 只消费被选中的 passing candidate artifact。

## 9. Verification

candidate-selection completion proof 至少覆盖：

- `first-legal`：两个合法 candidate 中选择排序最靠前的 passing candidate。
- `min-estimated-time`：两个合法 candidate 中选择 `estimated_cycles` 更低的 candidate。
- gate early-exit：SPM offset gate failure 后不继续把当前 candidate 当作 passing candidate，也不写入
  DDR offset fact。
- representative tile classes：tail / corner tail 失败时整个 tile shape 被 reject。
- simple matmul：traversal domain 是 `C[M, N]`，`K` split 作为 internal split；合法 candidate
  的 DDR tile views 来自 tile-region materialization，不由 instruction lowering、SPM offset
  assignment 或 DDR offset assignment 猜。
- simple elementwise：tile size 来自实际 output shape，external input/output DDR subview 和
  RDMA/WDMA descriptor 匹配。
- same-domain multi-output：多个输出共享同一个 traversal tile；每个 root 的 output map 都通过
  tile-region materialization 生成对应 DDR tile view，instruction lowering、SPM offset assignment
  和 DDR offset assignment 对同一 candidate artifact 统一验证。
- generic reduction split：`linalg.generic` reduction root 的 reduction iterator split 生成多个
  partial reduce 和 tile-local combine；只有所有 representative tiles 的完整 candidate gates 通过才接受。
- 文本一致性：candidate-selection 文档和 progress 不再把资源上限建模成 candidate 字段，也不把 DDR access
  summary/range attr 当成 committed IR fact。

## 10. Implementation Status

2026-06-11 当前实现：

- `wafer-select-group-tile`：module pass，扫描 `wafer.group`，为每个 group 生成独立 passing
  candidate artifact。
- `wafer-lower-groups-to-selected-instr`：named pipeline，作为 candidate-selection 用户级 replay 入口。
- candidate 生成：从 static ranked result 的实际 traversal shape 生成 bounded tile sizes；同一个
  group 的多个 result 必须共享同一 traversal shape 才进入 multi-output candidate。matmul root 和
  supported `linalg.generic` reduction root 额外生成 bounded reduction split。候选顺序是全部 no-split
  traversal candidates 先行，然后再进入 split candidates。
- representative coverage：每个 tile shape 至少验证 first / last / tail / corner tail 的代表
  tile classes；所有代表都通过 candidate gates 才接受该 candidate。
- gates：每个 candidate evaluation clone 按 candidate DDR tile-view materialization、instruction
  lowering、SPM offset assignment、DDR offset assignment、verifier 顺序执行；gate failure 早停，
  不继续跑后续 gate。
- `tile-search=first-legal`：默认模式，返回第一个 passing candidate。
- `tile-search=min-estimated-time`：只在 passing candidates 之间用 lowered instruction IR 的
  compute/DDR/SPM/issue 粗估时间排序。
- diagnostics：`print-candidate-summary` 输出 selected tile、split、estimated cycles、visited /
  rejected candidate 数和 representative 数；这些是诊断，不写入 committed IR。

当前显式限制：

- multi-output coverage 当前只覆盖所有 result 具有相同 static traversal shape，且每个 yielded value
  是可独立 materialize 的 destination-style linalg root。非 reduction root 的 DPS init 必须是对应
  group output boundary；reduction root 可以使用 tile-local fill/init，但 storeback 必须由 candidate
  tile-view materialization 显式写到对应 output boundary。不同 output domain、partial scatter coverage、跨 output 依赖或需要
  recompute/cut 的复杂 coverage 继续返回 structured `no_candidate` / gate failure。
- general reduction split 当前只覆盖 instruction lowering 可处理的 `linalg.generic` reduction 形态，并要求
  partial reduce 的 init/combine 语义能从 IR 中验证。softmax/scan、非结构化 loop-carried accumulator、
  dynamic reduction range 和需要跨 tile 状态的 reduction 不在当前 V0 范围。
- layout 候选仍消费 layout analysis / lowering 的默认 layout 决策；candidate-selection 不在本层硬编码 `tensor`、`cx`、
  `ncx` 的替代 layout 枚举。

当前测试入口：

- `test/Transforms/select-group-tile.mlir`：elementwise、static slice、large K=1000 matmul、reduce、
  `scf.if`、`scf.for`、tail/corner representative coverage。
- `test/Transforms/select-group-tile-internal-split.mlir`：SPM gate 迫使 matmul 使用 `K` split，
  并检查 partial GEMM + elementwise add + single WDMA 形态。
- `test/Transforms/select-group-tile-multi-output.mlir`：同 traversal domain 的多输出 candidate 覆盖。
- `test/Transforms/select-group-tile-reduction-split.mlir`：SPM gate 迫使 `linalg.generic` reduction
  使用 internal split，并检查 partial reduce + elementwise combine + single WDMA 形态。
- `test/Transforms/select-group-tile-min-estimated.mlir`：`tile-search=min-estimated-time`。
- `test/Transforms/select-group-tile-failure.mlir`：invalid mode 和 gate early-exit。
- `test/Pipelines/lower-groups-to-selected-instr.mlir`：named pipeline replay。
