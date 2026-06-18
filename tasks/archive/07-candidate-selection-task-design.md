# Wafer R3.2h Candidate Decision Design

> 归档记录：本文保留历史审计、恢复或任务级背景；不作为当前主线架构合同。当前入口见 `tasks/README.md` 和编号设计文档。

本文以 R3.2h 任务索引记录 candidate-selection 边界。candidate-selection 不是新的 memory
allocator，也不是把失败计划写进 IR 等后段修复的阶段；它负责枚举 tile/lowering 候选，逐个
重放 candidate DDR tile-view materialization、instruction lowering、SPM offset assignment、
DDR offset assignment 和 verifier gates。失败 candidate 只存在于 transformation-local clone；
selected candidate 由 R3.3 commit step 写回主 IR。

## 1. Goal and Non-Goals

目标：

- 根据实际 traversal shape 生成每维候选 tile size 序列，从 full traversal tile 开始做 lazy search
  frontier，而不是预先展开固定候选表。
- 把当前 candidate 的 traversal tile shape、可证明同 traversal domain 的 output coverage，以及当前支持的
  reduction/internal split 转成 candidate evaluation IR；layout choice 只在对应 interface 能表达多个
  合法候选后进入 search frontier。
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
  从 full traversal tile/no-split candidate 开始，按 gate failure 和当前 IR 推导的容量压力逐步细化
  traversal tile 或 reduction split；为每个 visited candidate 构造 evaluation clone；按 candidate DDR tile-view
  materialization -> instruction lowering -> SPM offset assignment -> DDR offset assignment -> verifier
  的顺序运行 gates。只接受通过全部 gates 的 candidate。`tile-search=first-legal`
  选择第一个合法 candidate；`tile-search=min-estimated-time` 在合法 candidate 中用粗估时间选最小。
- Output artifact / IR:
  committed main IR，包含 selected candidate 的 actual DDR tile views、instruction-level IR、
  accepted SPM offset facts 和 accepted DDR offset facts；或 no-candidate / split-needed failure
  reason。
- Downstream consumer:
  topology/device-mesh/shard-binding contract、R3.6 ABI/LLVM lowering、R3.7 package manifest 和 R3.8 runtime adapter 消费
  committed main IR 中的 accepted SPM/DDR facts 和当前 IR 可重算的 descriptor/view/allocation demand，
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
  simple matmul/elementwise group 能由 candidate-selection driver 自动生成 shape-driven refinement frontier、
  运行 candidate gates、按 `tile-search`
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

### 4.2 Tile Size Refinement

Tile size 从实际 shape 出发逐步细化，不能只套固定表，也不能预先把所有维度做笛卡尔积展开。
对每个 traversal 维和 reduction 维，candidate-selection 构造一组由 shape 驱动的候选 size：

- 第一个值永远是当前维度的 full size。
- 下一步优先靠近当前 size 的一半，避免一开始就跳到过小 tile。
- 如果实际 shape 有合适 divisor，优先选能减少 tail 的 divisor。
- target preferred tile sizes 只作为 tie-break / hint；它们不能替代实际 shape，也不能成为固定候选表。
- `max-candidates-per-dim` 是搜索空间控制项；正数限制每个维度最多保留多少个候选 size，`0`
  表示保留该维度的全部 shape-driven refinement。默认不能小到阻止搜索找到第一个合法 candidate。

例如某维长度是 `1000`，在常见 preferred hint 下，候选 size 序列可能是：

```text
1000 -> 500 -> 250 -> 125 -> 64 -> ...
```

这里的 `64` 不是“固定表里拿来枚举”的协议，而是在逐步减半、divisor 和 preferred hint 共同排序后
进入 bounded size options 的一个 refinement point。不同 shape 会得到不同候选序列。

候选搜索从 full traversal tile 和 no split 开始。某个 candidate 被 gate 拒绝后，planner 根据
当前 IR root 的容量压力对 refinement 维度排序：

- matmul 的 `M` 压力来自 `A[M,K]` 和 `C[M,N]`，`N` 压力来自 `B[K,N]` 和 `C[M,N]`，reduction
  压力来自 `A[M,K]` 和 `B[K,N]`。
- generic / elementwise root 的压力按当前 traversal tile 的输出 footprint 分摊到 traversal 维。
- 多输出 group 对每个 yielded root 分别累计压力，但是否能复用 buffer 仍由后续 SPM offset
  planner 判定。

search frontier 每次加入高压力维度的一步 refinement，并额外加入少量 top-pressure 组合邻居，用于处理
footprint 来自多个维度乘积的情况。这样 search space 的上界仍由每维候选 size 数限定，但默认路径不需要
预先展开 `D^R` 个候选。

frontier 还有显式搜索空间控制项。硬件 SPM/DDR 容量、alignment 和粗估吞吐是 target policy
事实，不作为 `wafer-select-group-tile` 的用户 option；搜索空间本身可以配置：

- `max-search-candidates`：candidate 总访问数的软上限，只在已经存在 passing candidate 后启用。
  它不能阻止 search 找到第一个合法 candidate；如果还没有 passing candidate，search 必须继续遍历
  完整 refinement frontier，直到找到合法 candidate 或 frontier 本身耗尽。并行模式可能完成当前
  in-flight batch 后再检查上限。`0` 表示找到第一个 passing candidate 后不继续做优化探索。
- `search-beam-width`：找到第一个 passing candidate 后，每个 rejected/passing candidate 最多向队列
  加入多少个 refinement neighbor。它只约束优化 frontier，不能约束第一个合法 candidate 的发现；
  `0` 表示不限制 beam。
- `candidate-parallelism`：`tile-search=min-estimated-time` 下每批最多并行评估多少个 candidate。
  并行 worker 只处理 transformation-local standalone group 文本和 worker-local MLIRContext，不直接
  修改主 IR；最终 selected candidate 的 commit artifact 仍由主线程在原 context 中重新 materialize。
  它只在 `min-estimated-time` 下校验和消费，`first-legal` 不读取这个参数。
- `tile-search-effort=quick|default|deep` 提供 search-space preset；显式设置的
  `max-candidates-per-dim`、`preferred-tile-sizes`、`max-search-candidates` 和 `search-beam-width`
  覆盖 preset。

quick SPM bound 只能做必然失败剪枝：它使用当前 candidate 的最小必要 live footprint 下界，例如 matmul 的
`A tile + B tile + output tile` 裸字节；多输出取各 root 下界的最大值而不是相加。它不能使用 layout
padding、临时 buffer 或 aligned-family multiplier 的上界来 reject candidate。真实 SPM 峰值、复用、
layout padding 和 lifetime overlap 必须由 SPM offset assignment gate 在 lowered IR 上验证。

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

- 初始 candidate 使用 no split；当 gate failure 的压力主要来自 reduction 维，search frontier 会把
  reduction split 作为 refinement neighbor 加入队列，不需要先跑完整个 no-split traversal 笛卡尔积。
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
  -> quick SPM lower-bound check
       fail only if minimum required bytes exceed planning window
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

quick SPM lower-bound check 只能拒绝必然超过 planning window 的 candidate；它不能因为 conservative
upper-bound estimate 失败就跳过完整 gates。失败时立即停止当前 candidate 的后续 gates。SPM 失败不能继续假装 DDR planning 还能补救；DDR
失败也不能回头改 instruction semantics。candidate-selection 只能选择下一个 candidate 或返回 no-candidate /
split-needed reason。

Passing candidate artifact 是 transformation-local artifact。R3.3 commit step 只把被选中的
artifact inline 回原 `wafer.group` 所在位置，并用 selected return values 替换 group results。
Rejected candidate IR 必须丢弃。

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
- search 从 full traversal tile/no split 开始。
- gate failure 后按当前 IR 推导的容量压力选择下一批 traversal/reduction refinement neighbor。
- 同一层 refinement 中，压力更高的维度先尝试；压力相同时 reduction refinement 优先，因为它通常
  能降低 matmul/reduction 的输入 live footprint 而不增加输出 traversal tile 数。
- 单输出优先；同 traversal 多输出必须满足 coverage legality。

### 6.2 `min-estimated-time`

估算时间模式。candidate-selection 继续使用同一个 shape-driven refinement frontier，但不会在第一个
合法 candidate 停止。每个 passing candidate 也会继续扩展 bounded refinement neighbor，直到 frontier
耗尽或在已经找到 passing candidate 后达到 `max-search-candidates` 软上限。它收集已访问范围内所有通过
candidate gates 的 candidate，对每个合法 candidate 计算粗估时间，选择 `estimated_cycles` 最小者。
当前实现是 first-legal discovery + budgeted optimization frontier，不承诺全局最优。

当 `candidate-parallelism > 1` 时，candidate evaluation 按队列顺序组成 batch 并行运行；主线程按
原队列顺序合并结果、更新 rejected count、扩展 frontier 和选择 best candidate。因此并行只改变
evaluation wall time，不改变确定性选择规则。

cost model 只在合法 candidate 之间排序，不参与 legality，也不能接受一个 gate 失败的 candidate。
cost breakdown 是 diagnostic，不写入 committed IR。

## 7. Rough Time Estimation

V0 cost model 使用 target policy 中的硬件参数、计算量、DDR 访存量、SPM/local movement 量和
instruction issue 数量估算时间。硬件吞吐、alignment 和 memory range 不作为
`wafer-select-group-tile` 的 public option；没有板端校准时使用 target policy 的保守默认值。

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

- `retryable_candidate_failure`：当前 traversal tile / reduction split candidate 失败，可以换下一个
  candidate。
- `split_needed`：当前 group 需要拆分，例如多输出 coverage 不兼容、required movement 还不能表达、
  或 search frontier 内所有 candidate 都被 memory/movement gate 拒绝。
- `no_candidate`：bounded refinement frontier 本身没有 passing candidate，且没有更细的 split policy
  可用。search budget / beam 不能单独制造 legality failure。

diagnostic 应记录：

- 失败发生在哪个 gate。
- candidate spec 的 traversal tile shape、reduction/internal split 和 output coverage；未来 layout
  候选需要等对应 interface 落地后再进入 diagnostic。
- diagnostic gate prefix 使用稳定语义边界名：`tile-region`、`instr-lowering`、`spm-offsets`、
  `ddr-offsets`、`verifier`；不把 `R3.2*` 任务号打印成用户可见诊断。
- SPM/DDR offset assignment gate 的结构化 reason，例如 capacity、largest-contiguous、
  bandwidth、alignment、descriptor/view mismatch。
- `min-estimated-time` 模式下 winning candidate 的 cost breakdown。

diagnostic 不成为 IR 合同；R3.3 commit step 只把被选中的 passing candidate 写回原 group 位置。

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
- large control-flow composition：full-tile candidate 下覆盖 `scf.if` 中的大 shape matmul + elementwise
  chain，以及 `scf.for` 中的大 shape elementwise accumulate；输出必须 commit 回原函数，不生成
  selected side artifact。
- composed multi-output selection：`scf.if` 分支内的 same-domain multi-output group 同时包含
  `linalg.matmul` 和独立 elementwise root。普通 lit 覆盖固定 target SPM range 下的 K split；
  专用 heavy runner 覆盖 large-shape traversal tiling、K split 和三输出多 op 组合。
- tiled control-flow root gap：当大 shape `scf.if` root 需要 smaller traversal tile 才能满足 SPM
  gate 时，当前 candidate tile-view materializer 仍要求 yielded root 是 linalg root，并返回结构化
  `no_candidate`；这需要后续 output coverage / control-flow tiling interface 扩展，不能用名字或
  side table 伪支持。
- 文本一致性：candidate-selection 文档和 progress 不再把资源上限建模成 candidate 字段，也不把 DDR access
  summary/range attr 当成 committed IR fact。

## 10. 实现边界

已落地：

- `wafer-select-group-tile`：module pass，扫描 `wafer.group`，为每个 group 生成独立 passing
  candidate artifact，并将 selected candidate commit 回原 group 位置。
- `wafer-lower-groups-to-selected-instr`：named pipeline，作为 candidate-selection 用户级 replay 入口。
- candidate 生成：从 static ranked result 的实际 traversal shape 生成每维候选 tile size 序列；同一个
  group 的多个 result 必须共享同一 traversal shape 才进入 multi-output candidate。matmul root 和
  supported `linalg.generic` reduction root 额外生成 reduction split size options。search 从 full traversal
  tile/no split 开始，gate failure 后按当前 root 的容量压力扩展 traversal 或 reduction refinement
  neighbor；不预先展开固定候选表。
- quick SPM bound：只使用最小必要 live footprint 下界做必然失败剪枝；layout padding、临时 buffer、
  aligned-family multiplier 和真实 lifetime overlap 由 SPM offset assignment gate 判定。
- representative coverage：每个 tile shape 至少验证 first / last / tail / corner tail 的代表
  tile classes；所有代表都通过 candidate gates 才接受该 candidate。
- gates：每个 candidate evaluation clone 按 candidate DDR tile-view materialization、instruction
  lowering、SPM offset assignment、DDR offset assignment、verifier 顺序执行；gate failure 早停，
  不继续跑后续 gate。
- `tile-search=first-legal`：默认模式，返回第一个 passing candidate。
- `tile-search=min-estimated-time`：先保证找到第一个 passing candidate，再遍历 budgeted bounded
  optimization frontier，在已访问的 passing candidates 之间用 lowered instruction IR 的
  compute/DDR/SPM/issue 粗估时间排序；当前不承诺全局最优。
- fixed target policy：SPM range、SPM alignment、DDR planning range/alignment 和粗估 timing
  参数由 target policy 提供，不作为 `wafer-select-group-tile` 的 public option。select pass 的 CLI
  只控制搜索空间和诊断。
- search controls：`tile-search-effort=quick|default|deep` 提供默认 search-space policy；
  `max-candidates-per-dim` 默认跟随 policy，当前 policy 使用 `0` 表示不裁剪每维 shape-driven
  refinement；`preferred-tile-sizes` 默认使用 policy hint；`max-search-candidates` 和
  `search-beam-width` 默认跟随 policy。显式传入这些 option 会覆盖 policy。budget/beam 只约束已经有
  passing candidate 之后的优化探索；`candidate-parallelism` 只在 `tile-search=min-estimated-time`
  下校验和消费，`first-legal` 不读取该参数。并行模式只在 `min-estimated-time` 下启用，worker 使用
  独立 MLIRContext 评估 candidate，主线程重新 materialize selected commit artifact。
- diagnostics：`print-candidate-summary` 输出 selected tile、split、estimated cycles、visited /
  rejected candidate 数和 representative 数；这些是诊断，不写入 committed IR。
- commit：selected clone 的 lowered function body 按 group inputs/outs 映射 inline 回原
  `wafer.group` 前，返回值替换 group results；原 parent function、无关函数和同函数其它 ops 保留；
  输出不生成 `*_selected_group_*` 旁路函数。

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
  `scf.if`、`scf.for`、representative coverage。
- `test/Transforms/select-group-tile-multi-output.mlir`：同 traversal domain 的多输出 candidate 覆盖。
- `test/Transforms/select-group-tile-reduction-split.mlir`：SPM gate 迫使 `linalg.generic` reduction
  使用 internal split，并检查 partial reduce + elementwise combine + single WDMA 形态。
- `test/Transforms/select-group-tile-min-estimated.mlir`：`tile-search=min-estimated-time`。
- `test/Transforms/select-group-tile-complex-control-flow.mlir`：大 shape `scf.if` + matmul + elementwise
  chain、大 shape `scf.for` + elementwise accumulate 的 commit。
- `test/Transforms/select-group-tile-tiled-control-flow-gap.mlir`：固定 target SPM range 下需要 tiled
  control-flow root materialization 时的 structured `no_candidate`。
- `test/Transforms/select-group-tile-huge-composed.mlir`：`scf.if` 分支内的 same-domain multi-output
  group 覆盖固定 target SPM range 下的 `matmul + elementwise` 组合和 matmul `K` split。
- `test/Transforms/select-group-tile-public-options.test`：`wafer-select-group-tile` 的 public option
  边界；硬件/吞吐 target facts 不作为 option，搜索空间参数保持可配置。
- `tools/run_heavy_candidate_selection_tests.py`：专用手动 heavy runner，不接入 lit/ctest；使用默认
  SPM planning range 覆盖 `257x4096 * 4096x257 -> 257x257` 的 multi-output traversal tiling、
  `257x8192 * 8192x257 -> 257x257` 的 multi-output K split，以及三输出 `matmul + add + mul`
  组合。
- `test/Transforms/select-group-tile-failure.mlir`：invalid mode 和 gate early-exit。
- `test/Pipelines/lower-groups-to-selected-instr.mlir`：named pipeline replay；覆盖原函数名保留、
  unrelated function 保留、同函数多 group commit 和禁止 `selected_group` 旁路函数。
