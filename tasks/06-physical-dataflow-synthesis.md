# Wafer Whole-DAG Multi-Tile 时空综合

状态：2026-08-13 作为Q49、Q50.A–Q50.K及Q51–Q53 current whole-card synthesis主线的唯一设计 owner。旧 whole-rank 实现已经证明
relation、actual clone、SPM/DDR packing、NoC lowering、completion 和 package mechanics 可以工作，但它在
GSPMD 后把 logical rank 直接绑定到单卡 16 个物理 Tile，因而丢失了 Tile 级 spatial mapping、不同 op 并行和
spatial/temporal/fusion 的联合搜索空间。本设计替换该决策合同；实现状态只看 `tasks/progress.md`，施工顺序只看
`tasks/plans/whole-card-tile-dataflow-synthesis.md`。

本文不要求保留当前搜索器、cost 比较或 rank-oriented orchestration。现有代码只有在能落入本设计的 IR 与
owner 边界时才复用；为旧架构或旧测试服务的实现直接改写或删除。Llama、Attention 和 decode 只作为通用 DAG、
state 与规模验证，不定义任何 shape、mask、参数顺序或专用 shortcut。

## 1. 核心结论

编译器优化对象是一个 card-local structured DAG 的完整时空执行，而不是单个 op、单条 consumer chain、预先
切好的 16 个 rank 或所有 op 共用的 tile shape：

```text
global iteration domain
  -> card_id
  -> physical tile_id
  -> temporal wave
  -> tile-local iteration
```

同一个候选必须共同决定：

1. 单个 op 如何拆给多个物理 Tile；
2. 多个不同 op、独立分支或不同 wave 如何同时占用不同 Tile 集合；
3. implementation、temporal tile、loop order 与 tail；
4. 同 Tile producer/consumer 的 coupled traversal、独立 traversal 和 SPM residency；
5. spatial mapping 改变时的 NoC redistribution、multicast、gather 和 reduction；
6. selective spill/reload、recompute、region cut 和 release；
7. compute、DDR、NoC 与显式 SPM movement 的 buffered overlap；
8. 整卡所有 observable output/effect 完成时的 makespan。

GSPMD 只拥有 card 级 global-to-local tensor partition。单卡内部 4×4 Tile 的工作划分由本文 stage 完成；
`num_partitions` 保留为 card 级数量，不再表示 Tile 数。所有 available Tile 必须进入候选空间。当存在 ready work，
且把它放到空闲 Tile 不增加通信、容量或关键路径时，继续空闲的状态被直接支配；少用 Tile 只有在理论模型预测
更低 makespan 时才能成为 winner。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  GSPMD完成card级分区、target-independent canonicalization完成后的完整card-local structured tensor DAG；
  current SSA、structured op semantics、IndexRelation、effect、type、shape和dtype均可验证，尚未绑定物理Tile；
  函数边界是frontend验证后的functional input/result ABI，argument只表示真实input/parameter/constant，result表示
  observable output，不存在隐式trailing output参数。
- Current stage responsibility:
  对整张DAG联合选择physical tile_id、op-wave时间顺序、spatial work domain、temporal tile、implementation、
  region/residency、physical representation、DDR/NoC movement、buffering和可证明overlap；只对shortlist物化actual IR。
- Output artifact / IR:
  一个wafer.card.program，内部包含按physical tile_id区分的wafer.tile.program；Tile程序用实际loop、SSA、
  wafer.tile.region、movement、send/recv/wait和event表达selected MPMD执行。
- Downstream consumer:
  Tile到Instr conversion、fresh completion reconstruction、fixed-capacity SPM/DDR planning、transport/resource/ABI
  verification、target conversion、package publication和runtime launch。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；public optimization policy仍只有`search`与`none`，不新增逐机制开关。
- Explicit non-goals:
  不重做跨card GSPMD；不建设cycle-exact simulator；不从名字、shape、workload或文件恢复语义；
  不让allocator、completion、communication或cost analysis生成repair；不发布search plan或sidecar。
- Completion gate:
  通用single-op、chain、branch、fanin/fanout、reduction、layout-changing和stateful DAG均由同一source-to-package
  pipeline产生MPMD；HF prefill/decode与Llama block完成FP16/BF16 fresh package/no-card；真实板端matched A/B
  证明至少一个Llama代表和一个prefill/decode代表相对同源baseline获得可重复性能改善。
```

## 3. 稳定 IR 对象与所有权

### 3.1 `wafer.card.program`

`wafer.card.program` 是一个 `card_id` 的完整 MPMD 验证范围。它拥有：

- card-local observable inputs、outputs 和 shared DDR boundary；
- all-and-only physical Tile programs；
- 跨 Tile message matching、coverage 和 completion verification 范围。

它不保存 frontier、score、schedule list、route side table 或 target calibration。若底层 ABI 仍要求扁平 launch slot，
只在 target lowering 中把 `(card_id, tile_id)` 编码为 slot；该编码不反向进入 structured IR。

### 3.2 `wafer.tile.program`

`wafer.tile.program` 绑定唯一 physical `tile_id`。不同 Tile 可以拥有完全不同的 op、loop、tile shape、worker
候选和执行长度。实际程序顺序、并发和依赖由 body 中的 loop、SSA、send、recv、wait 与 event 表达，不另存
全局 schedule attr。

verifier 要求 Tile 属于 current topology 的 available endpoint，card 内 `tile_id` 唯一；任何 SPM memref root 或
alias 不能跨 `tile.program` SSA 传递。

### 3.3 `wafer.tile.region`

`wafer.tile.region` 继续只表示一个物理 Tile 内的一段 SPM residency domain，不表示硬件 Tile，也不要求 region
内所有 op 使用相同 tile shape。同一 region 可以包含：

- consumer-driven coupled traversal；
- 多个不同 tile shape 的独立 loop；
- local physical conversion；
- selective spill 某个 root 而其它 roots 继续 resident。

跨 region shaped data 必须显式 materialize；跨 Tile data 由 source SPM、NoC send、destination SPM staging 和
matching recv 表达。跨 Tile edge 可以属于同一 card-level operator pipeline，但不是一个跨 Tile 共享的 SPM root。

### 3.4 Whole-card verifier

verifier 从 current IR 重建并证明：

- `card_id`、`tile_id` 唯一、可用且属于 current topology；
- 各 Tile loop/subview 的 work domain 合并后 all-and-only 覆盖原 structured semantics；
- 除显式 reduction 与 selected recompute 外不存在重叠执行；
- send/recv 的 source、destination、message identity、domain、encoding、physical bytes 和 completion 匹配；
- fanout source lifetime 覆盖最后一个 consumer；fanin/reduction 的输入和等待完整；
- SPM roots、DDR ownership、observable effects 和 output completion 闭合。

## 4. Whole-DAG 符号调度模型

### 4.1 Op-wave DAG

完整 static/dynamic temporal 实例不在搜索中逐个展开。每个 structured loop 只形成有限的 prologue、steady-state
pattern 和必要 tail classes。edge readiness 细化到 producer 的某个数据 window，因此 consumer 可在所需 window
ready 后启动，而不必等待整个 producer op 完成。

consumer-driven tile propagation 仍负责根据 indexing semantics 与 `IndexRelation` 计算 consumer 需要 producer 的
exact domain；它不再拥有全局遍历顺序。唯一全局 owner 是 whole-DAG event-driven scheduler。

可观察独立分支首先按 current structured SSA 的无向依赖连通分量证明：从每个函数 result 穿过无副作用 support
op 找到最近 structured roots，再把 producer/consumer edge、共享 structured producer 和同一 result 汇合的 roots
合并。只有所有相关 structured op 都是无 memory effect 的 tensor SSA、每个 node 都归属于至少一个 observable
result，且不存在跨分量 effect / memref alias 时，分量才可绑定 disjoint Tile 集合。证明失败只删除该并行候选；
不会靠 op 名或 shape 猜测独立，也不会把联合 baseline 判成非法。

### 4.2 Query-local state

```text
DAG:
  ready / running / completed op-wave classes

Per physical Tile:
  scheduled/running op-wave
  local ready and finish time
  selected spatial iterator domain, temporal traversal and implementation
  resident values: producer-consumer edge, finite wave class, domain, MemLayout/physical encoding and footprint
  implementation temporaries and explicit layout-conversion buffers
  selected buffer count, rotating slot and send/receive staging buffers
  compute and explicit SPM-movement resource calendars

Communication:
  per-directed-link NoC and card-shared DDR resource calendars
  pending movement, data-ready and completion events

Global:
  shared DDR/NoC work, observable obligations and current makespan
```

这些对象只能存在于一次 candidate-selection invocation，允许失效和重算；它们不能跨 pass、序列化、发布或
成为 late verifier 输入。winner 的全部执行事实必须物化为 actual IR，随后销毁 search state。

### 4.3 Event-driven transition

每次推进到下一个 compute、movement 或 data-ready event，然后：

1. 形成当前 ready op-wave 集合；
2. 选择一组无依赖冲突且可同时运行的 op-wave；
3. 为每个 op-wave 选择 physical Tile 集合与每 Tile work domain；
4. 选择 implementation、temporal tile、loop order、每个 value 的 MemLayout/physical encoding 和 buffer count/rotation；
5. 选择 local fusion/residency、local conversion、NoC、DDR、spill 或 recompute；
6. 在同一 transition 中预留 per-Tile compute/SPM movement、per-link NoC 和 card-shared DDR 时间窗口；
7. 更新 per-Tile SPM live set、rotating slot、message、event、resource work 和 finish time；
8. 释放已完成最后 consumer 及异步访问的 roots，并产生新的 ready work。

这统一覆盖 intra-op data/tensor parallelism、独立 branch parallelism、fanout/fanin、dependent wave pipeline、
不同 op 的 spatial pipeline 和 compute/movement overlap。

## 5. Spatial、Temporal、Fusion 与 Communication 联合域

### 5.1 Spatial mapping

空间候选从 structured iterator、factor、reduction semantics、tail、implementation geometry、physical encoding、
producer location 和 4×4 topology 共同派生。固定 `{2,4,8,16}` 可以由 provider 贡献普通 prior，但不构成完整空间。

placement 与 tile/fusion 同时决定：

- 相同 Tile 且 producer/consumer shard 通过structured indexing map落在同一iterator/domain时，直接local SPM reuse；
  仅Tile集合相同、`shardDimension`数字相同不构成local证明，transpose等轴置换必须形成exact redistribution；
- Tile 集合部分重叠时，重叠部分本地消费，只传缺失 domain；
- mapping 改变时显式形成 scatter、gather、broadcast、reduction 或一般 redistribution；
- fanout 可比较共置一个分支、multicast、分别发送和 recompute；
- fanin 可比较 gather/reduction tree 和 consumer remapping。

result shape 不需要相同。selected mapping 为每个 observable result 指定自身的 shard dimension 和 physical Tile
集合；同一 dependency component 的 results 使用同一 Tile group，独立 components 才能使用 disjoint groups。
actual CardProgram materialization在每个Tile只保留该Tile拥有的result roots及其current-SSA producer closure；其它
results 保持完整card-shared ABI但不产生store。该query-local mapping在actual IR生成后销毁，winner的区别只由各
`tile.program` body中的真实op、subview、load/store和SSA表达。

TileRegion物化需要可写result destination时，只在isolated candidate clone中按result显式追加private scheduling
destination；该区间以转换当次记录的source argument count验证并在CardProgram输出前全部消费。不得按参数位置、
数量、shape/type相同关系猜测source argument是output，也不得把private destination发布成CardProgram、TargetABI或
package接口。

独立 work 使用 deterministic minimum-cost assignment；存在 coupled edge、broadcast 或 reduction 时，对 4×4 固定
规模使用 topology-symmetry-reduced branch-and-bound。placement 不能在 temporal tiling 或 fusion 提交后事后补齐。
node placement 的合法性不依赖是否生成 peer transfer：same-group local residency、独立分支的 disjoint group 与
确实需要传输的 partial/disjoint coupled group 都进入同一有界 frontier；peer action 为空只是该 placement 的数据
无需跨 Tile 搬运，不能作为删除候选的条件。

对发生 redistribution 的 direct current-SSA edge，query-local actual 输入同时携带 consumer result shard、由
structured indexing relation 推出的 producer demand、same-Tile local fragment 和 cross-Tile remote fragment。
actual materializer 必须相对该 producer demand 证明所有 fragment all-and-only 覆盖、互不重叠，再按稳定 domain
顺序用 `tensor.insert_slice` 形成一个 SPM staging SSA 链；local fragment 从 exact producer tile 取得，remote
fragment 由显式 receive/wait 填充。`tensor.empty` 只提供 typed destination，任何 hole 都不能作为有效数据。
同一 Tile 可以同时拥有 observable output、send 和 receive；这些是 body 中可组合的 edge action，不是互斥角色。
partial-overlap 的 LiveSPM 只计实际 local intersection，不能把未物化的完整 producer shard 记为 resident。

### 5.2 Finite temporal tile domain

每个维度只枚举会改变硬件 work 的 breakpoint：

- `ceilDiv(extent, spatial_factor * tile_size)` 与 wave 数变化；
- native block、physical issued work、padding 或 footprint 变化；
- DDR/SPM/NoC transaction 与 descriptor 数变化；
- implementation geometry、instruction field 和 SPM feasibility 边界。

每个scheduled structured op的temporal state是覆盖其全部iterator的一个完整向量，不另设reduction-axis旁路字段。
理论驻留按当前operand indexing map对应的实际tile window和per-buffer alignment计算；无法从IR与已知hardware参数证明的项
直接省略，不能用完整iteration-domain体积、`Unknown`状态或猜测系数代替。无`reassoc`的浮点reduction只允许沿保持源
lexicographic顺序的adjacent breakpoint推进：后一个reduction轴必须等前轴unit-tiled后才可split。

禁止late allocator在已选actual candidate上反复`tile_size / 2`并把修补结果冒充原winner；但capacity refinement本身属于
whole-DAG搜索。对其它轴固定的一个candidate branch，搜索从当前spatial mapping产生的完整per-Tile local extent开始；
若已证明的working-set lower bound或一次受管actual packing失败，则在同一frontier中生成二分子候选，直到形成
infeasible/feasible区间。随后补入该区间内会改变tail/divisibility、native issued work、layout padding、transaction、
buffer-count feasibility或wave数的非二次幂breakpoint。每个子候选消耗统一ledger、重新参与dominance和exact gate，
不得由lowering或allocator私下修复。小 tile 只有在换来更长residency、更多spatial/pipeline parallelism、更少外部
movement或更短makespan时才可能保留；若仅减少footprint却增加wave、message和instruction，则被支配。

### 5.3 Attention/decode 算法资格与物化边界

Attention算法资格必须在physical search前从current structured SSA的typed语义证明，不得从op名、tensor名、单一shape、
`Q length == 1`或外部prefill/decode mode恢复。普通Attention语义可以产生online softmax recurrence候选；只有进一步证明
functional K/V cache append/update、更新结果被当前Attention消费且append/query/cache domain一致时，才允许产生
split-K/V decoding候选。符合decode语义的图仍可同时产生online候选；前端图决定候选算法集合，不提前替性能搜索选winner。

每个算法点必须先在isolated candidate中通过等价于`MaterializeFlashAttention`或`MaterializeFlashDecoding`的actual
transformation改写成显式recurrence/partition/merge DAG，随后whole-DAG scheduler才在该真实DAG上联合搜索spatial
mapping、进一步temporal tiling、fusion/residency、layout、buffering和resource overlap。算法名、provider key或参数向量
不得作为未物化旁路直接进入physical legality或cost。

`keyValueTileSize`表示每次online recurrence处理的K/V reduction window，`splitCount`表示split-K/V并行partition数；
二者都不是KV cache总长度、SPM容量或已完成的物理分配。完整reduction domain、divisor/tail、native geometry、transaction与
capacity breakpoint及actual失败反馈共同产生候选。`{128,64,32,...}`等target-generic seed最多调整枚举顺序，不得成为
搜索空间边界、固定默认值或SPM合法性依据；例如`KV=128`只有作为一个尚待验证的候选才有意义。cheap live-byte estimate
同样只能排序或做已证明的下界剪枝，最终合法性必须由materialized actual IR的lifetime、buffer、layout和fixed-capacity
SPM packing决定。

Attention的K/V refinement必须直接接入上述统一candidate transition。每个spatial factor/Tile placement先从其完整local
KV extent尝试，而不是从global KV长度或固定`128`开始；容量不足产生local KV二分子状态，容量允许时完整KV与较小KV仍可
并存，以比较single-buffer full traversal、较小tile的multi-buffer overlap、更大Q tile、跨算子fusion和split-K/V并行。
只有在其它候选轴固定且footprint对K/V tile单调时，二分结果才可用于区间剪枝；改变layout、buffer count、Q tile、fusion
或placement后必须按新状态重新判断，不能跨状态复用“最大可放下KV”作为全局事实。

### 5.4 Fusion and residency actions

每条 producer/consumer edge 至少有以下通用 action：

1. coupled local fusion；
2. same-region independent traversal with retention；
3. local physical conversion；
4. peer NoC transfer and destination retention；
5. selective spill/reload；
6. pure producer recompute；
7. region cut and release。

搜索总是保留 maximal feasible local-residency chain，但同时保留把相邻 op 放在不同 Tile 组、通过不同 wave
形成 operator pipeline 的候选。fusion 不按 op 数量奖励；只由减少的 DDR/reload/conversion、增加的 NoC、SPM
pressure、parallelism 和最终 makespan 决定。

DPS init-operand 边（如 `linalg.fill` 直接供给 consumer 的 `outs`）是初始化状态，不是 spatial data edge：
它不携带上述任何 edge action，初始化状态由 consumer 的 typed lowering 管辖（producer closure 在 consumer
traversal 内融合）。whole-DAG scheduler 仍通过同一 DAG 边做 wave readiness 排序，只是 edge strategy plan
与 card spatial mapping 不为它生成 action；card-program materialization verifier 同样只要求 direct
structured data-input 边有 selected treatment。

### 5.5 Physical representation、buffering 与 resource timeline

layout 分配不是 lowering 的默认决定或 materialization 后的 repair。对每个 scheduled value，候选状态显式选择
consumer/implementation 可接受的 `MemLayout` 与 physical encoding；producer 和 consumer 表示不一致时，同一 edge
transition 必须选择 local conversion、NoC 传输中的表示转换、spill/reload 或其它已定义 action，并把转换
buffer、movement 和 lifetime 计入同一状态。winner 必须将选择直接物化为 typed IR，lowering 只验证和消费，
不再自行选 layout。

single/double/triple buffering 也是每个 pipeline edge/op-wave 的候选维度。状态同时跟踪 slot 轮转、所有者、最后
async consumer 和 all-buffer aligned footprint；只有 buffer 彼此独立且当前 event/resource 约束证明重叠可实现时，
才生成 overlap transition。compute、explicit SPM movement、每条 directed NoC link 和 card-shared DDR 各自使用有界的
resource calendar；transition 预留实际时间窗口并生成 data-ready/completion event，不能在最后对已选顺序仅用
`max(compute, movement)` 估算重叠。layout、buffer 数、worker/order 或 resource 时间窗口的任何 exact failure 都回到
同一未放置 parent，不调用独立 allocator/scheduler 修复。

## 6. SPM Search Bound 与 Late Exact Packing

每个 Tile 的 search-time working set 包含：

```text
live inputs + resident intermediates + outputs
+ implementation temporaries + layout conversion buffers
+ NoC staging + rotating pipeline buffers + physical padding
```

搜索从 footprint、alignment、definite-live、may-overlap、remaining consumer 和 async completion 构造临时 conflict
关系：已知同时 live 集合或 weighted clique 超过 usable SPM capacity 时立即拒绝；保守 aligned placement 能放下时
提前确认存在可行排列；其它状态保留到 shortlist。

完整 MPMD 完成 Tile→Instr、selected worker/order物化和 fresh completion 后，才从 actual roots、control flow、lifetime 和
coexistence 形成 fixed allocation problem 并调用 MiniMalloc。allocator 只返回 validated offsets 或失败：不得改变
tile、spill、region、order、worker 或 completion。失败交回同一个 scheduler，从未放置 parent 选择其它状态。

## 7. 理论 Cost Model

### 7.1 边界

复用现有 work collectors、理论公式和 target 参数来源；不新增第二套硬件模型，也不建设精细 bank/port/arbiter
simulator。hard legality/capacity 与性能 estimate 完全分离。

性能 cost 删除 `Unknown`、`Indeterminate`、Unknown propagation、不可比较状态、`ProvenBenefit` / `EstimatedBenefit`
分流及依赖它们的 promotion/fallback。所有 hard-legal candidates 必须得到数值 makespan。

参数只按以下规则处理：

1. 有 matching target/profile 实际值时使用实际值；
2. 否则使用现有明确的理论参数；
3. 完全不知道的项不计算。

未知项不按零、不设无穷大、不参与 Pareto，也不阻塞比较。一次 comparison cohort 统一确定 enabled terms；某项
不可用就从该 cohort 的全部 candidates 删除，不能 candidate-local 缺项。参数来源可作为 diagnostic，不进入选择语义。

### 7.2 Formula

```text
T_compute = max_tile(physical issued work / compute throughput)
T_ddr     = card DDR bytes / DDR bandwidth
T_noc     = max_directed_link(NoC bytes / link bandwidth)
T_spm     = max_tile(explicit local movement bytes / SPM bandwidth)
T_control = exact command count * configured command cost
```

startup/issue 只有参数存在时才加入；未知 bank conflict、outstanding、endpoint arbitration 等不计算。exact raw work
始终保留用于 dominance 与诊断：per-Tile work、wave/native block、DDR/SPM/NoC bytes、link load、message、descriptor、
instruction、wait、SPM high-water 和 Tile idle time。

### 7.3 Branch、Pipeline 与 Movement Overlap

独立 branch 或不同 Tile 组同时执行时，compute phase 取并发组完成时间最大值。候选 actual/dataflow 明确存在独立
buffer 与 double/triple buffering 时，steady-state：

```text
II = max(enabled compute, DDR, NoC, explicit SPM movement per-wave time)
makespan = prologue + (waves - 1) * II + epilogue + enabled control overhead
```

有 data/effect dependency 的 phases 顺序相加；无证明的 overlap 不假定存在。不得对整程序无条件全相加或全取 max。
理论 cost 用于 best-first/beam ordering 和 final numeric selection，不宣称 cycle accuracy。

final selector只消费accepted Instr IR形成的`StaticSchedulePlan`。CardProgram私有candidate clone以typed、query-local
structured node lineage跟踪source operation到最终operation；lineage不进入公开IR schema，并在winner进入下游前从全部
location中剥离。cost collector仍遍历完整entry control flow，只按最终operation集合过滤，因此`scf` trip count、call
closure和NoC topology不会因切片丢失。被合法消除的内部source node允许形成空slice，但其每条observable successor path
必须由accepted IR中存活的下游lineage覆盖；多lineage融合只归属一次，并且必须存在由DAG依赖证明的唯一下游owner。
每个node slice和明确的boundary/communication/control residual slice必须彼此不重叠，并对所有可加raw metrics、per-Tile
allocation count及high-water组合与whole-program cost精确守恒；observable terminal lineage丢失、foreign lineage、无唯一
下游owner的融合或不守恒直接删除该候选，不允许按比例拆分整程序cost。

当前可证明overlap只作用于physical Tile集合不相交的node local compute/SPM phases；card-shared DDR和NoC仍各以完整
accepted whole-card work形成共享资源phase。temporal loop的真实重复次数保留在node slice中，不把同一actual cost复制到
symbolic prologue/steady/tail。只有actual IR进一步给出buffer/resource独立性证明时，才允许把共享movement与compute拆成
更细的pipeline overlap。

## 8. Search Algorithm 与 Bounded Materialization

- linear/tree 子图可以用 boundary-compatible DP 压缩，但结果必须回到同一个 whole-DAG scheduler；
- 一般 DAG 先使用complete lazy generation、event-driven best-first与Pareto ordering；bounded beam、候选cap或随机
  启发式不是预设合同，只能在Q52对实际负载完成热点、质量和最优性损失分析后作为显式trade-off启用；
- bounded小图保留不裁剪完整枚举oracle，`search`策略必须持续与其比较winner与stable digest；
- topology symmetry、exact coverage、SPM lower bound 和 enabled raw-work dominance 在 clone 前剪枝；
- stable global work ledger 管理 expanded states、actual materializations 与 exact gates；
- host worker 只并行独立 state evaluation，frontier insertion 与 tie-break 按 stable key；
- 只有 shortlist materialize whole-card MPMD actual clone；任意时刻最多一个 live actual candidate；
- exact failure 消耗明确 work unit 后销毁 clone并继续 frontier，不建立 repair selector；
- exact/correctness模式的work budget按deterministic units记录，不用wall-clock timeout悄悄删除合法状态；`search`
  trade-off的时间或work budget只能由Q52基于实际profiling配置，并必须保留合法baseline与质量诊断；wall/RSS同时作为
  scalability回归证据。

winner 必须通过同一 Tile→Instr、fresh completion、SPM、DDR、communication/resource、ABI 和 final recost gates。
`none` baseline 也走相同下游，不调用 legacy compatibility path；“相同下游”不包含search的candidate frontier、
allocation-feedback分支或edge-action transition。

### 8.1 Deterministic baseline controller

`none`在whole-DAG search controller之外执行一个单状态legality recurrence：每个Linalg op确定性切分到完整可用
16-Tile参与集合，op间值统一经compiler-owned DDR发布和重新读取，所有op独立执行且不融合，buffer count固定为1。
本地producer/consumer shard以DDR RegionCut隔离；确定性spatial shard集合不一致时按typed indexing relation物化唯一
required peer fragments，并在consumer端DDR assembly后进入独立op stage。这是correctness communication，不是baseline
搜索坐标；可选edge action、route和通信优化仍属于后续统一搜索。每个Tile上的每个op从其spatial shard的完整iterator
box开始；actual SPM allocator失败时，controller只缩小冲突lineage对应op的temporal iterator shape，并要求DDR
load/store随当前temporal window流动。

同一allocator conflict set中多个不同op可以在一个确定性轮次各缩一个或多个有限breakpoint；这是单一program state的
legality推进，不创建子candidate，也不按cost选择winner。baseline controller不得改变spatial assignment、edge action、
fusion、layout、NoC、buffering、worker或overlap。最小temporal endpoint仍不能通过actual allocator时，baseline明确失败。
accepted baseline与`search` winner从CardProgram materialization开始共享完全相同的projection、Instr、completion、
SPM/DDR、admission和package链路。

## 9. Existing Mechanics 的处置

### 9.1 可保留的语义能力

- structured tiling/reduction interfaces、DPS 与 `IndexRelation`；
- typed physical representation、movement 和 numeric legality；
- `tile.region` local residency、MiniMalloc、DDR planner；
- peer/collective IR、Direct-DTE lowering；
- typed completion、worker、loop-carried slot和instruction-order表达能力；旧独立candidate API不保留；
- global ledger、delayed clone、deterministic parallel evaluation、wall/RSS/work counters；
- current package/runtime 对不同 physical endpoint program 的发布能力。
- typed Attention/functional decode SSA语义证明、online recurrence与split-K/V actual materialization及其正负测试；其接口和
  owner可以改造，但不能退化为shape/name shortcut，也不能只留下算法枚举而删除真实图改写。

这些能力允许改接口、owner 和代码组织；默认处置是把已有算法、actual transformation、typed verifier和有效测试改造到
current whole-DAG owner，而不是随旧接口一起删除。“保留能力”不要求永久保留旧search selector，但在current
候选transition、selected actual IR和替代测试逐项证明等价前，旧实现与覆盖必须保留或恢复。特别是layout solver、
ready-order、fixed-slot software pipeline、NCC worker placement、peer/collective lowering和可泛化的专项tiling/reduction
实现，不能退化成固定默认值、只保留字段，或仅凭新主线能编译就宣称已迁移。

### 9.2 必须重写或删除

- GSPMD 后按单卡 16 Tile 建 logical ranks；
- logical rank 到 physical Tile 的直接映射、rank0 clone N 和 rank-local winner；
- common function-result tile vector、linear connection DP 与 fixed partition count 完整域；
- 不含 ready/running op-wave 和 per-Tile LiveSPM 的旧 frontier，以及只由 unit test 手工增减、未接入
  search-policy event transition 的 residency API；
- late NoC profitability selector 与全程序 sequential/pipelined 二态 overlap；
- performance `Unknown`/`Indeterminate` comparison；
- 未经typed SSA语义证明的Attention、decode、mask、shape或参数名shortcut；
- 只为上述旧合同服务、且其承载能力已经在current pipeline由等价或更强gate接替的tests、docs、diagnostics和
  compatibility branches。旧owner过时本身不是删除算法实现、verifier或测试的充分条件。

## 10. Failure、Determinism 与 Diagnostics

- semantic、effect、alias、coverage、encoding 或 hard capacity 无法证明时，拒绝对应 candidate；
- 性能参数缺失只删除统一 cost term，不成为 legality failure；
- baseline 始终存在且经过相同 exact gates；
- serial/parallel generation 必须产生相同 admitted frontier、winner 和 package；
- diagnostic 报告 enabled cost terms、parameter source、per-Tile work/SPM、parallel op-waves、DDR/NoC/SPM work、
  estimated phase/II/makespan、expanded states、actual clone 和 exact gate counts；
- 不把 search state、历史失败、accepted offset 或 board sample 写入 selected IR。

## 11. Completion Gate

本设计按独立队列项交付：Q49闭合current同路径baseline；Q50.A–Q50.K分别闭合edge correctness、semantic proof、
online DAG、partitioned-K/V DAG、layout、residency、communication、ready-order、multi-buffer、worker/completion和
fusion traversal机制；Q51闭合统一搜索正确性与search-policy cutover，Q52只基于实际负载优化scalability，Q53形成
fresh package/no-card与board证据。任一前项的阶段性
编译或测试通过都不能代替后项完成；具体依赖、状态与提交门禁由`tasks/progress.md`和唯一实施计划管理。

### IR / verifier

- distinct `tile.program` 可含不同 op、loop、tile shape 和长度；
- duplicate/unavailable tile、coverage hole/overlap、cross-Tile SPM alias、mismatched send/recv、premature release fail；
- same-region different temporal tiles、multi-region cut、selective spill、recompute和NoC staging可验证。

### Scheduler

- single-op multi-Tile、independent ops、chain wave pipeline、diamond/fanout parallel、fanin gather/reduction、
  partial co-location与mixed NoC mapping均由同一算法产生；
- ready work 存在时无理由 idle Tile 被 dominance 删除；
- maximal local fusion 与 cross-Tile operator pipeline 对立候选同时存在；
- small-tile wave/message/instruction代价、double-buffer SPM footprint和overlap进入选择；
- 任意未配置性能参数删除后，cohort仍有确定数值排序且无performance Unknown。

### End to end

- generic GEMM、elementwise、reduction、conv/mixed DAG、official HF prefill、functional KV-cache decode、Llama block
  FP16/BF16从真实source生成完整package并fresh no-card；frontend保持原始模型语义，无专用mask或`-inf`改写；
- host evaluation使用可用逻辑CPU并行，actual clone、packing、peak live clone有确定上界；wall/RSS处于fresh baseline
  可解释范围，不设任意60秒门槛；
- 无卡阶段达到board-ready后，真实设备只串行执行current matched cases；Llama及一个prefill/decode代表必须相对
  同源baseline获得可重复性能改善，才能把任务标记done。

## 12. 参考算法原则

本设计吸收而不复制外部实现：consumer-driven exact demand propagation、factorized spatial/temporal mapping、
resident/refetch/recompute frontier、topology-aware placement、boundary-compatible Pareto composition和whole-DAG
event scheduling。任何具体论文或仓库的shape、算子集合、solver依赖和搜索常数都不成为本项目协议。
