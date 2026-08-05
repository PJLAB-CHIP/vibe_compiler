# Wafer Whole-Rank Tile Dataflow Synthesis

状态：本文是 structured tensor IR 到 selected tile-dataflow IR 之间的唯一联合决策设计 owner。
Q32 已闭合 implementation、relation、layout、storage、order 和 communication 等独立机制；Q49 负责把这些机制从
“按 task 分别枚举、过早 lowering、再拼 complete variant”的实现路径，收敛为一次 complete-rank、consumer-driven、
resource-aware 的联合综合。region partition、traversal fusion/materialization、tile size和resident/spill不是先后阶段：它们与loop order、
layout/version、movement、communication和completion consequence在同一搜索中共同选择。`tile.region` op的语义固定为
SPM residency domain，但一个rank program如何划分为这些domains是搜索变量。实现状态只看 `tasks/progress.md`，施工 checkpoint 只看
`tasks/plans/whole-rank-tile-dataflow-synthesis.md`。

本文不新增第二套 layout、candidate IR、调度 plan 或 runtime 协议。当前 MLIR operation、region、SSA、type、typed
attribute、effect 和 target capability 是唯一语义事实源；candidate 只有 isolated actual IR clone。Llama block 用于暴露和
验证通用问题，不定义任何 shape、op 顺序或模型专用规则。

术语固定为：**whole-rank**表示一次优化覆盖rank function完整范围；**complete-rank actual clone**表示确实包含该rank完整
program的候选artifact；**all-rank/whole-variant**只表示把all-and-only logical ranks组成一个原子tuple并执行跨rank gate。

## 1. 核心结论

当前优化能力并不缺少单点机制，主要缺口是决策顺序和物化边界：producer/consumer 在仍可联合 tiling、共享、驻留和重排时，
已经按 task 各自变成完整 Tile/Instr 程序；每个局部结果随后分别插入 DDR round-trip、layout movement 和 completion join。
后续 pass 即使能删除局部冗余，也无法可靠恢复被过早丢失的完整 SSA、tile relation、fanout 和 lifetime 选择空间。

终态采用以下原则：

1. **先形成 complete-rank structured SSA，再做whole-rank联合综合。** post-SPMD 每个 rank 的完整函数是 rank-local 决策范围；
   跨 rank collective、peer edge 和资源对应关系由同一 all-rank transaction 验证，不把 rank-local 分析伪装成 card-wide 图。
2. **consumer-driven tile propagation 是主算法。** 从 external output、observable effect 和其它真实edge-legality cut 向 producer
   反推所需 tile；只有 exact relation、numeric legality 和 target capability 均成立时才跨 op 传播。
3. **联合选择residency partition、fusion、tiling与physical dataflow。** `tile.region`表示一组可共同驻留、可由SSA/effect
   和liveness解释的physical tile dataflow；一个complete static rank entry可以有一个或多个non-nested regions。搜索同时决定
   merge/split、共同或独立traversal、tile shape/loop order、physical representation、storage/transport和execution/completion：
   single-region resident候选可能迫使tile变小；multi-region materialized候选增加DDR，却可能让两侧tile变大、缩短共同live roots并
   减少loop-expanded work。task、source scope、collective、loop或单个spill都不自动成为cut；selected cut必须在actual IR中
   显式结束跨界SPM residency并materialize所有跨界data。两者没有先验胜负。
4. **延迟不可逆 lowering，不延迟候选事实。** 不因为 task、source scope、loop boundary或旧 `tile.region` 自动创建 RDMA、WDMA、GS 或
   `NCCJoin`；任何带region/traversal/tile/layout/residency/order决定的state在进入frontier或cost/gate前，都先把selected
   regions、SCF loops、view、movement和structured event/effect obligation物化进自己的actual clone。resident edge、local movement、
   recompute和DDR materialization均由SSA/effects显式表达，region结构不反向产生movement或join。只有terminal survivors
   执行不可逆Tile→Instr和exact late gates。
5. **legality owner 独立，decision owner 唯一但分层执行。** structured 层决定 tile/loop/layout/resident edge，Instr 层才派生
   worker/slot/completion siblings；relation、physical realizability、SPM、DDR、communication、instruction、
   ABI 各自只回答自己的可验证问题；只有本文 owner 生成邻居、排序候选和决定 fallback。
6. **capacity failure 反馈给搜索，不让 allocator 修 IR。** allocator 返回 typed failure；搜索 owner 从未放置 parent 建立
   有界 structured sibling，例如改变region partition、traversal fusion/materialization与两侧tile、缩短 lifetime、改变 physical version或spill某条edge；terminal
   Instr层另行派生worker/slot/latest-necessary-completion variants。final bufferization后从complete current IR的roots、control flow
   和coexistence/conflict关系形成fixed SPM allocation problems，并由MiniMalloc和独立validator覆盖all-and-only roots；actual high-water只报告
   capacity/headroom，bank phase至多进入allocator既有单次搜索的deterministic offset ordering，作为
   hard-valid choices间的末级soft preference；不改变hard feasible set，也不反向改变spill、region、DDR movement、worker/order或join。
7. **completion 在最终执行结构上重建。** source-observable ordering 只在 schedule/completion 维度形成不可跨越的约束；
   它不自动切断 tile propagation、physical representation 或 storage 选择。compiler-derived join 只在 worker、slot、
   range reuse、communication 和 external drain 已确定后，从 current Instr effects/event SSA fresh 构造。

这不是“把更多 fusion pattern 加到现有 pipeline”。它改变的是联合决策的 IR 范围、候选生命周期和 lowering 时机；
已有 canonicalization 仍保留，但不再承担恢复全局结构的职责。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  verifier-legal post-SPMD rank-local structured tensor IR；每个 logical rank 保留完整 function/region/SSA、
  DPS destination、indexing map、iterator/reduction、structured control flow、typed collective/peer relation、
  dtype/numeric semantics 和 external effects。target capability、execution mesh 与 memory arena 是只读输入。
- Current stage responsibility:
  在 isolated complete-rank clones 上发现可联合综合的 connected components与逐edge正交legality；通过 TilingInterface、
  IndexRelation、physical encoding/transfer interfaces 和 current effects 从 consumer 向 producer 传播 tile demand；
  在all-rank transaction内共享一个deterministic work budget和coordinated frontier，联合枚举region merge/split/cut、
  traversal fusion/materialization、implementation、
  tile shape/loop order、share/recompute、physical version/layout、resident/spill、local/peer movement和loop reuse choices；
  每次merge/split/fuse/separate都同时重算两侧tile schedule、region boundary interface、overlapping SPM-root lifetime/live bound与
  loop-expanded DDR/GS/compute/completion work；
  用有界resource-aware search保留deterministic Pareto frontier；只对terminal all-rank Tile variant中的complete-rank Tile clones按
  typed collective/peer algorithm参数逐点lower actual canonical Instr sibling，再派生worker/fixed-slot/ready-order、删除全部
  compiler-derived join并fresh重建completion；从每个terminal rank-entry Instr variant的current roots/lifetime/conflict派生
  all-and-only fixed SPM allocation problems并原子验证，对每个complete all-rank variant执行whole-variant DDR、post-memory
  Direct-DTE/resource、ABI和final recost exact gate；DDR gate按current explicit arenas/placement domains构造问题并原子汇总，
  再由当前校准的hardware cost model选择并原子提交all-rank winner。
- Output artifact / IR:
  原子accepted `ExecutableBundle`：包含all-and-only logical ranks的final verifier-legal Instr artifact、accepted SPM/DDR
  placement、transport binding与current target-legality records。terminal Tile clone只是本stage内部handoff；
  不发布all-rank correspondence、component graph、搜索state、candidate ordinal、layout proposal、repair recipe或proof sidecar。
- Downstream consumer:
  target module/package publication、runtime loader与board launch。Tile→Instr、worker/fixed-slot/ready-order、fresh completion、
  whole-rank SPM、whole-variant DDR、post-memory Direct-DTE binding和target ABI均是本stage内部terminal gates。
- User-level driver / named pipeline:
  wafer-compile source-to-bundle production pipeline；wafer-opt 只复用相同 interfaces/conversions 做局部测试，
  不提供需要用户手工拼 pass 的第二条生产路径。
- Explicit non-goals:
  不建立 workload/op-name/shape matcher，不维护 IR 外长期调度计划，不让 runtime 重新选择 physical realization，
  不把 allocator 变成 optimizer，不以未校准模型替代 exact legality，不在本层声称板端性能。
- Completion gate:
  complete-rank chain、diamond、fanin/fanout、multi-root、loop-carried reduction、collective 和 mixed-effect source
  均经默认 driver 产生 selected actual IR；同一source至少覆盖single-region fused-small-tile、multi-region
  separated-large-tile、same-region separated traversal和selective spill等对立candidate，单条spill不迫使无关resident root结束。
  task/source-scope/loop边界不自动生成region；每个region的data inputs/results为variadic DDR边界且SPM operand/result为零，
  selected cross-region edge有显式store/completion/load。region exit不因容器本身自动形成join，只证明仍访问其SPM roots的work已完成；
  entry terminal证明all-and-only pending external effects已完成。所有terminal candidates在selection前通过
  fresh complete traversal、memory、descriptor、completion、all-rank、ABI 和 source-to-package/no-card gates；
  短序列、read-only cached-attention与长prefill cases完成host/model/package/no-card而达到board-ready；fresh板端只执行
  当前Llama `(16, 16)` matched A/B与一个read-only cached-attention representative，correctness/performance通过后任务才可done。
```

## 3. 稳定对象与所有权

### 3.1 唯一长期事实源

跨 stage 保留的事实必须存在于 current IR：

- structured op、region、SSA use-def、DPS destination、indexing map、iterator 和 reduction semantics；
- `memref<..., #wafer.memory<space, layout>>`、allocation root、standard/typed view 和唯一 `MemLayout`；
- selected `wafer.tile.*` compute/movement/event、structured traversal 和 loop-carried state；
- `MemoryEffectOpInterface`、必要 custom resource、async token、wait/fence 和 external observability；
- accepted SPM/DDR offset、worker/slot、Direct-DTE binding 和 target-owned typed fields。

以下对象只可存在于一次 transformation/query 内，返回即销毁：

- connected-component walk、IndexRelation composition、alias/effect/liveness和loop-expanded/bounded-work estimate；
- partial search state、Pareto/cost summary、failed placement、lower bound 和 rejection reason；
- current operation/value 的非拥有引用。

它们不能序列化为 attr、side table、artifact 或下游输入。rewrite 改变 op、region、SSA、type、view、effect 或 control flow
后，旧 analysis 和 cost 全部失效并从 current clone 重算。

### 3.2 Candidate 生命周期

“candidate”只指 isolated、可验证的 actual MLIR clone：

```text
verified post-SPMD snapshot
  -> clone complete rank
  -> apply one bounded set of structured/physical decisions to actual IR
  -> canonicalize + verify current layer + keep in bounded structured frontier or destroy
  -> shortlist terminal Tile clones
  -> for each typed collective/peer algorithm parameter, lower complete rank to an actual canonical Instr sibling
  -> worker / fixed-slot / ready-order siblings
  -> erase and fresh-rebuild completion
  -> exact SPM / DDR / post-memory communication / ABI gates and final recost
  -> atomically commit one complete all-rank tuple
```

不携带决策的 relation/cost proposal 可以避免为明显 infeasible 的前缀复制 module；proposal不是候选、不能跨pass、不能成为
另一份execution graph。任何被frontier保留且携带tile/layout/residency/order决定的state都必须拥有actual unplaced rank clone；
PBQP或其它factor proposal一旦被保留就立即replay到clone并销毁。进入exact gate前不存在只靠decision list解释的survivor。
失败clone不修改source、其它clone或accepted module。

不再使用 task candidate、standalone task module、task alternative ordinal、physical artifact kind 或先拼 ordinal 再做
complete-rank import 的生产合同。稳定排序只由 IR-derived key、typed decision 和 deterministic traversal 给出；ordinal 只能是
容器内部的临时迭代位置，不能参与语义对应。

### 3.3 独立 legality owners

| Owner | 只回答 | 不回答 |
| --- | --- | --- |
| structured interfaces / IndexRelation | tile/index relation 能否 exact 表达与组合 | 是否值得 fusion 或 spill |
| physical realization | view、encoding、DMA/GS/peer transfer 是否可实现 | 选择哪个全局 layout |
| instruction conversion | selected Tile IR 能否无损变成 typed Instr IR | 失败时换实现 |
| SPM planner | current complete-rank clone 是否能在固定 arena 合法放置 | 应缩哪一个 tile |
| DDR planner | current whole-variant ranges/lifetimes/offsets 是否合法 | 是否应保留某条 resident edge |
| communication / completion | typed issue、wait、participant、resource 和 drain 是否闭合 | 通过插入保守 task join 修复上游结构 |
| target / ABI | current typed commands 与地址是否可编码 | 反向限制上层为历史 ABI 形态 |

decision owner 只消费上述 typed success/failure 和 current-IR facts，不能复制其 verifier 或维护第二份 capability table。

## 4. 五个联合决策维度

五个维度相互影响，但没有任何一个边界自动推出另一个边界。

### 4.1 Region partition、traversal/materialization 与 tile schedule

`tile.region`的固定语义是SPM residency domain；rank entry如何partition成一个或多个non-nested regions由搜索决定。
同一region可以包含一个或多个traversal domains、tile shapes和loop nests，只要selected resident relations、root lifetime和
working set能由current IR验证；“同region”不等于“同一个mega-tile”或“全部root同时live”。不同region之间不得传递SPM
root/alias，所有data edge必须由显式DDR store、可信completion和matching load连接。typed control/event只有在不携带SPM
alias且自身verifier允许时才能穿过边界。

每条producer/consumer edge的联合action同时决定：两侧位于同一还是不同region、使用耦合还是独立traversal、各自tile/loop，
以及连接edge采用resident SSA、local movement、DDR materialization、streaming或recompute。single-region fused-resident候选可能
节省DDR却迫使共同tile变小；multi-region materialized候选可让两侧独立retile、缩短共同root lifetime与SPM live bound，却增加
DDR、descriptor和completion；same-region separated traversal与selective spill也都是独立候选。schedule separation、per-value
spill和region cut相关但不互相蕴含，每个action必须把partition、两侧tile、lifetime、movement与loop-expanded work共同物化和计价。

task、source scope、旧region、普通effect、collective、layout conversion、retile或单个spill都不是cut witness。selected cut必须
结束切口上的全部SPM residency，并在actual IR中显式表达跨界materialization；没有真实residency终止、materialization或external
ownership含义的空壳boundary由verifier拒绝或canonicalization移除。opaque clobber无法由typed effect解释时fail closed，不能靠
多包几个region伪装成支持。region数量只作结构诊断，不能进入dominance或代替DDR、GS、NCC/completion和critical-path成本。

### 4.2 Tile connection

决定 consumer tile 如何映射到 producer tile、是否跨 op 传播、是否共享或重算，以及 traversal/reduction 如何组织。
依据是 Tiling/DPS semantics、IndexRelation、SSA、control flow、effects 和 numeric contract。

### 4.3 Physical representation

决定每个 logical value 需要哪些 physical versions、每个 version 的唯一 `MemLayout`、view relation、valid-lane/padding 和
target implementation orientation。fanout可以共享一个version，也可以在收益明确时形成有界的K个显式SSA versions；K只由
search budget控制，不是IR语义上限，所有consumer仍须逐edge证明relation与materialization。

### 4.4 Storage and transport

决定 value/tile 在 SPM、DDR、peer/NoC 或可重算状态中的位置，以及 direct view、mapped DMA、local GS、staged movement、
Direct-DTE、typed collective、spill/reload 等 edge realization。collective 是通信/effect 语义，不自动是 DDR 边界。

### 4.5 Execution and completion

决定 dependency、worker/slot、engine overlap、buffer reuse、wait/fence 和 terminal drain。task、component、traversal、loop、
任意builder调用、static op顺序或`tile.region` boundary都不是completion cut。region exit只需证明仍访问将被释放SPM roots的
pending work已经完成；与这些roots无关的typed work可继续，entry terminal才闭合all-and-only observable pending effects。
verifier只检查final facts，不因看见region op自动插join。

例如 producer 与 consumer 共享同一resident SSA version时，其root lifetime由实际use-def覆盖；它们处于同一region仍可能因
exact relation不可表达而需要显式GS，也可能只对某个value做internal DDR spill。反过来，没有spill也不要求两个traversal
合成一个loop nest：耦合schedule可能扩大同时live working set、迫使tile缩小并放大loop/GS/descriptor/completion work。
collective后可以保留SPM resident result；跨worker edge也可能只需最小participant join而不需要DDR round-trip。

## 5. Component 与逐 Edge Legality

综合从 observable roots 逆向遍历 SSA。component 是一次分析中得到的connected search subproblem，不是IR op、region、fusion
group、SPM arena或artifact。一个region可以含多个analysis components，每个component也可以包含多个traversal domain；component
边界只停止局部propagation/组合，本身不创建region、DDR movement或completion；decision owner仍可在有typed legality与完整
materialization action的edge frontier上提出region merge/split sibling。

每条SSA edge分别求以下legality向量，不能压成一个“可融合/不可融合”布尔值：

| 维度 | 问题 | 典型结果 |
| --- | --- | --- |
| region/traversal/tile schedule | 两侧是否同一residency region；能否形成耦合structured traversal，或物化两套独立loop并分别retile | same-region耦合/独立loop、cross-region materialized cut或仅停止此action |
| tile propagation | producer/consumer domain能否由finite `IndexRelation` exact组合 | direct tile、显式movement或仅停止此维传播 |
| numeric reassociation | reduction/recompute/order变化是否满足typed numeric contract或query-local proof | 允许特定actual sibling或保持原顺序 |
| storage/materialization | view、layout、valid domain和target transfer是否可实现 | resident/direct、GS/staged、DDR boundary或unsupported |
| schedule/completion | effect、async dependency、reuse和observer要求何种顺序 | issue order、wait/join、terminal drain或barrier |
| rank coupling | collective/peer tile/payload是否允许同一组terminal algorithm参数，展开后的chunk/message能否all-rank对应 | 参数只生成actual Instr sibling，随后逐tuple重证或拒绝 |

unknown/external mutation、无法表示的control flow、缺失typed target capability和source-observable publication可以阻断对应维度；
只有所有可用实现都无法跨越时才停止该component内的联合传播。collective通常允许tile传播，却要求completion和all-rank coupling；
output publication要求store/drain，却不禁止producer tile直接写output view；layout不兼容要求某种materialization，但不自动决定region cut。

任何维度的分析失败都不自动插入 DDR、GS、`NCCJoin`或新region。selected action可以在同一region内物化resident、GS、
recompute、streaming或selective spill/reload，也可以物化结束切口全部SPM residency的cross-region store/completion/load；
storage/transport/completion始终由各自typed facts决定。partition由搜索显式选择并立即落入actual IR，不从analysis failure、
source scope或旧结构机械恢复。
如果 relation 或 capability 是硬件/IR 可表达但当前尚未实现，先扩对应 interface/op/verifier；不得用 op 名、buffer 名、参数位置、
固定 shape 或模型角色恢复语义。

## 6. Consumer-Driven Tile Propagation

### 6.1 基本算法

对每个 external output、observable store、collective result 或 component root：

1. 从 result domain 生成由 target capability 与 resource bounds准入的有限 tile domains和consumer traversal seed；
   full-domain conservative baseline永远保留。
2. 通过 `TilingInterface`/DPS/indexing maps 把 consumer tile 精确映射到每个 input/init tile。
3. 用 `IndexRelation` composition 穿过 view、slice、broadcast、permutation 和 reshape；physical-isomorphic relation形成 view，
   否则只生成显式 movement sibling。
4. 对 pure/speculatable producer，比较共享、consumer-local recompute、loop-invariant hoist 和 materialize once；fanout 的所有
   consumers 在同一 component 内联合考虑。
5. 对每条producer edge原子生成有限联合action proposal：same-region或cross-region partition、耦合traversal与兼容tile/loop，
   或两套独立traversal与各自tile/loop；每种组合再与resident/local movement/DDR streaming或spill/reload/recompute的合法
   storage realization组合。cross-region action必须覆盖all crossing values并显式物化边界。
   schedule separation与DDR materialization互不蕴含，不能先选局部tile winner再补storage/lifetime。
6. proposal只在clone前使用SSA relation、typed legality、target capability和footprint lower bound作便宜rejection。对通过预筛并要进入
   global frontier的action，立即clone complete-rank current IR，同时物化region partition、loop/tile/physical version/storage/transport决定，
   重算SPM live-working-set bound、loop-expanded DDR/GS/compute work和completion obligation，然后销毁proposal。
   从这一点开始，frontier、cost、gate和后续action只消费actual IR clone。
7. 到另一root或已覆盖frontier后，只按future live interface做DP/beam合并；component或旧scope identity不进入语义key。

传播必须证明 output all-and-only coverage、tail union、reduction order和每个 fanout consumer 的 version relation。不能用
representative tile、单个 loop body dump 或 static op 数量代替 complete traversal proof。

### 6.2 通用数据流形态

- **chain**：比较single-region耦合/独立tile schedule、multi-region显式materialization与same-region selective spill；SPM resident
  direct SSA只是一个candidate，不能压过“多一次DDR但tile更大、共同lifetime更短、loop/GS更少”的合法sibling。
- **diamond / fanout**：联合比较共享一个 resident version、共享多个 physical versions、分支局部转换和纯 producer recompute；
  不按某一个 branch 先决定 layout。
- **fanin / shared input**：同一 input 的 load、layout conversion 和 loop placement共同计价；例如多个 contraction 的 LHS 可以在
  外层 tile loop 只 load/convert 一次，而不是在每个 N tile 内重复。
- **loop-carried reduction**：state、partial、accumulator、valid domain 和 completion显式 loop-carried；不同 reduction strategy
  只是 actual IR siblings。
- **collective-connected flow**：logical collective/peer relation作为 typed edge，前后 compute tile 可以在合法时保持 resident；
  all-rank matching仍在完整 tuple 上原子验证。
- **selective spill / streaming**：一个region内可让A长期resident，同时B store→completion→reload，或让Q/state长期resident而
  K/V逐tile从DDR streaming。B/K/V的movement只结束对应root，不能把A/Q/state一起变成region boundary。

### 6.3 Reduction 与 softmax

reduction strategy 不由固定序列长度决定，而由 selected tile、SPM peak、typed instruction capability、numeric contract 和 cost共同决定：

- reduction domain 能在一个 legal tile 内完整覆盖时，优先生成 native one-pass reduction；
- 需要跨 tile 时生成 explicit partial reduction/merge；
- 对 softmax 或 attention，只有在 current typed IR 能表达 max、normalizer 和可选 value accumulator 的 loop-carried state，且
  numeric verifier接受时，才生成 online sibling。

online reduction不是consumer-driven tiling自然推导出的普通rewrite，而是独立typed semantic candidate；Q49的最小online纵向是
由现有SCF、tile reduce/elementwise/GEMM表达的fused attention `(m,l,o)` capability，不新增Instr op或ABI字段。candidate只在
current SSA证明score/softmax中间值除matching weighted-value contraction外没有其它observable use、dropout为静态identity，且
scale、bias、mask的broadcast/indexing、dtype、应用顺序与source语义均可typed重放时生成；名字、shape或参数位置不能代替证明。

对每个query row，先按source顺序应用`z = scale * score + bias + mask`；boolean-excluded lane不参与max/sum，
additive mask则作为typed bias参与计算；具体boolean/additive mask、
NaN/Inf/signed-zero和fully-masked-row结果必须来自source numeric contract。无法区分时先扩semantic op/verifier，本candidate fail closed。
running state还携带由valid-domain派生的`has_value`控制谓词，数值初值与合并为：

```text
init: has_value = false, m = -inf, l = 0, o = 0
tile(z_t, v_t): m_t = max(valid z_t)
                l_t = sum(exp(z_t - m_t))
                o_t = sum(exp(z_t - m_t) * v_t)
merge(a, b): if empty(a) return b; if empty(b) return a
             m = max(m_a, m_b)
             l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
             o = o_a * exp(m_a - m) + o_b * exp(m_b - m)
finalize fused attention: y = o / l when has_value
```

empty tile/state在进入`max`或指数差前由`has_value`分支处理，不能计算`-inf - -inf`；fully-masked row的finalize必须逐项匹配
source定义，不能擅自选择zero或NaN。`m/l/o`的accumulator dtype、rounding、exp实现、tail mask、underflow/overflow和reduction
order全部进入typed numeric contract。

standalone softmax若要输出完整probability tensor，`(m,l)`本身不够：actual sibling必须显式物化第一遍统计与第二遍
`p = exp(z - m) / l`全域写回，第二遍的re-read/recompute/resident storage及cost都留在IR；或者显式保存并按后续global-state
merge重标定每个partial。Q49不要求实现第二条online纵向，未实现时保留native/partial/two-pass baseline。fused candidate遇到
softmax probability额外use、非identity dropout、无法重放的mask/bias、dynamic RNG或未定义fully-masked policy时一律拒绝。
small/native、partial和online仍进入同一通用候选域，不建立softmax专用pipeline。Q48后续可自动发现和证明更广semantic variants，
但只消费Q49 final Instr/TargetCall合同，不能反向成为本条纵向的完成前提。

## 7. 有界联合搜索

### 7.1 State 与 cost

搜索采用 live-frontier constrained pebbling：只保留已经处理的 component frontier 以及仍影响未来决策的事实。structured阶段的
每个survivor拥有actual unplaced Tile IR clone；worker、slot和concrete completion只在terminal Tile clone lower成Instr后派生，
不能在pre-Instr frontier中虚构pending NCC或worker schedule。一个structured state至少包含：

- 当前selected region membership/boundary interface、traversal coupling/materialization、covered domains、live tile relations
  和fanout obligations；
- 每个 live value 的 physical versions、`MemLayout`、location 和 valid domain；
- 从structured IR派生的per-region及cross-region coexistence SPM lifetime/peak bounds、DDR spill/reload、loop reuse和
  recompute work；exact placement后置；
- 尚未兑现的typed async/effect/observer obligation和range reuse requirement；跨rank匹配只从actual IR临时分桶并逐tuple重证；
- static site、static-trip loop-expanded execution work和conditional-path lower/upper bound；运行时实测计数仍由Q9 profiler拥有；
- cost vector 与 deterministic tie-break key。

cost vector至少分别记录：

- DDR read/write bytes 与 call/descriptor count；
- local GS/pack/unpack bytes 与 call/descriptor count；
- pre-Instr可证明的collective/peer bytes、NoC link/endpoint work、effect/observer obligations与各engine work lower bound；
- compute/recompute的static-trip expanded work、selected tile utilization与dependency lower bound；worker、Direct-DTE和`NCCJoin` exact work
  在terminal Instr完成worker/order/completion后才Known；
- SPM live-footprint bound、exact SPM movement bytes、DDR peak、buffer count 和 descriptor/resource pressure。SPM bytes保持
  独立可审计work；SPM peak/high-water是hard capacity与headroom事实，不作为普通execution Pareto维度，也不用历史
  SPM0/RAM_ACC `128 GB/s` flat rate换算duration；bank phase不形成candidate cost。

pre-Instr的bytes只有在physical coverage已exact时才可`Known`；RDMA、WDMA、GS及pack/unpack的最终call/descriptor count会受
Tile→Instr descriptor splitting/coalescing影响，没有同源exact proof时只能记safe bound或`Unknown`，不能参与exact dominance。
terminal final-IR recost才产生这些exact counts并用于最终winner selection。这些量不能未校准地相加成伪时间，`Unknown`不能
当作零、不能dominates任何对应Known值，也不能因插入顺序拒绝later candidate。terminal resource scopes固定为：

- DDR：all-rank aggregate read/write bytes与RDMA/WDMA executions为card-shared主压力，另报max-rank issue work；
- GS/layout：max-rank GS bytes/executions为tile-local主压力，aggregate只作总工作审计；
- completion：max-rank steady-state、nonterminal和total participant waits及其critical-path位置为主，join op数只作次级审计。

直接dominance要求所有selection-sensitive维度都可比较且不差，并至少一项严格更好：除上述DDR、GS和completion外，还包括
compute/recompute、tile utilization、NoC/link/endpoint work、Instr work、descriptor/resource pressure、SPM movement、
dependency/critical-path bound和all-rank coupling；SPM high-water只参与capacity/headroom，不作为execution cost。任一对应维度为`Unknown`且不能证明
两边具有同一unknown disposition时，两个state不可比较。DDR下降而GS、completion或任何其它维度回退时同样保持不可比较，不能由
`ExternalMovementFirst`无条件覆盖baseline。terminal recost逐rank报告上述工作及aggregate shared-resource demand；不同维度的最大值
不能伪装成同一个真实critical rank。Pareto frontier固定保留baseline及关键资源代表，当前校准的hardware cost model只排序已经通过
hard legality与terminal exact gates的states。它以fresh final IR的exact work和qualified point/bound parameters计算nominal/bounded
makespan与promotion margin；不能计算或不能清除baseline margin时保留baseline。stable semantic order只在完整target
selection tuple相等时作最后tie-break，不能代替cost从不可比Pareto states中任意选winner。
loop 中一条 instruction 必须按 exact static trip count或保守 symbolic multiplicity计入，不能把 static call site 当作 dynamic count。

### 7.2 求解策略

- chain/tree component仅在有限枚举的tile/action domain内，且未处理IR可观察的live-frontier facts、physical versions、loop reuse和
  async/effect obligations完全相同时使用exact dynamic programming；这里的“exact”只对该有限domain和等价类成立；
- 有限 fanout DAG 使用 deterministic Pareto beam，按 live frontier state等价合并；
- 大component优先在articulation与有限future-live interface处分解，再对interface做有界组合；这种analysis分解本身不改变
  region，但selected merge/split action可以改变actual partition，并显式物化DDR/completion；
- layout PBQP只作为同一component/analysis epoch内的局部factor reducer。它只可删除已证非法项，或在future live
  interface、physical versions、SPM lifetime/capacity disposition，以及完整selection vector（DDR all-rank aggregate/
  max-rank issue、GS/local max-rank/aggregate、completion/wait/critical path、NoC/link/endpoint、SPM movement、
  compute/recompute、tile utilization、Instr、descriptor/resource、all-rank coupling）和所有`Unknown` disposition
  上完全等价的项；不得以local conversion cost产生winner。proposal通过预筛后立即materialize到actual clone并销毁；
- small component只用仓库内exhaustive/property test检查有限domain的最优性、dominance和
  baseline retention；production不依赖外部solver；
- all-rank transaction内的complete-rank structured与terminal alternatives共享一个global deterministic work ledger；
  coordinator在generation前为conservative baseline的全部mandatory terminal exact gates预留credits，并保留有限repair reserve。
  非baseline proposal只有在可按deterministic upper bound预留其所需Tile→Instr、worker/order/completion、SPM、DDR、transport和ABI
  evaluation credits后才能进入frontier；generation不能消耗已预留credits。actual evaluation按stable action order消费或释放reservation，
  late failure只有在剩余global repair credits内才能回到unplaced parent生成neighbor。component、rank、PBQP、layout、worker或artifact
  不得各有独立allowance后再做Cartesian product。允许从actual clone即时派生携带region boundary、collective/peer接口、
  additive与max-reduction cost contribution及`Unknown` disposition的transient rank/component factor summary，用于同一coordinator
  的factorized DP/beam；summary不能发布、独立选winner/commit或形成`N^R` tuple。budget exhaustion只能停止
  未准入扩展，不能把未证明infeasible伪装成capacity/legality failure，也不能让已准入terminal candidate因前置搜索耗尽预算。

用于预筛的transient hash必须从actual IR即时派生，命中后仍逐项重证future-interface等价；它不能成为正确性key、IR/interface或schema。
dominance key不得含 task ID、op/value 名、candidate ordinal、模型角色或路径。相同 source、target facts 和 options 必须得到相同
frontier 和 winner；并行评估只改变吞吐，不改变接受顺序。

### 7.3 分层求解器边界

whole-rank综合不是一个固定变量集的单次packing问题。traversal fusion/materialization、tile、share/recompute、layout、resident/spill、
movement、loop order和worker/completion的选择会真实改写op、SSA、effect、lifetime和conflict graph；把它们
一次性编码进全局ILP/SMT/CP-SAT会复制dialect/interface/verifier语义，并在每次actual rewrite后立即失效。
生产只保留下列分层求解：

1. structured层由consumer-driven propagation生成有限typed actions，用component-local DP、Pareto beam和
   interface decomposition保留actual Tile clones；这些action原子包含region merge/split、traversal coupling、两侧tile schedule和physical realization，
   structured层不发布winner；
2. terminal Instr层在worker/slot/order确定后fresh重建completion，由current IR得到固定lifetime与
   pairwise conflict relation；
3. packing层从complete current Instr IR的SPM roots、control-flow coexistence与pairwise conflict形成all-and-only fixed
   SPM allocation problems，再从current explicit DDR arenas/placement domains形成DDR problems；用受管MiniMalloc
   fixed-capacity search和独立validator求解，problem/query数量是budget diagnostic，
   不是region或entry固定语义；packing层
   不选tile、不插spill/join、不返回repair recipe；
4. accepted offsets回到同一actual clone后重跑无搜索的range/descriptor/completion-consistency/ABI validator和final recost，
   不再插入或移动join，也不存在solver sidecar。dynamic view/index range必须在实际use位置解释包围它的typed structured
   branch predicate；SPM provenance/high-water consumer必须覆盖planner已接受的同一`scf.if`/select/loop-carried alias形式，
   无法证明的path或origin保持typed `Unknown`/fail closed。

MiniMalloc是当前fixed-lifetime/fixed-capacity合同的唯一production backend；这是专用搜索、确定性、
三态failure和轻量集成上的工程选择，不声称它对所有图都有通用运行时最优性。不建立常驻
ILP/CP-SAT oracle或完成门禁：小图正确性由仓库内exhaustive tests证明；只当真实workload持续
出现`ResourceExhausted`或可量化的packing质量问题时，才可导出该份actual
`StaticPackingProblem`做一次性外部诊断；诊断结果不回写IR、不成为production输入。Q48的query-local
SMT只证明semantic rewrite等价，与本层packing求解无关。

### 7.4 Capacity-aware siblings

structured frontier只消费从current Tile IR派生的capacity lower bound或`Unknown`；Instr-level SPM/DDR planner只对terminal
siblings执行并保持pure fixed-capacity gate。terminal evaluation期间，actual Tile parent一直作为拥有IR的parent clone存在，不是
decision list或sidecar。placement失败后，decision owner根据typed failure class从该无offset actual parent生成有限sibling。
structured parent可生成：

- merge/split相邻residency regions，或在same-region内耦合/分离相邻traversal，并同时改变相关tile shape/loop order，
  或只缩放一个合法tile dimension；
- 把某条 live edge 从 resident 改为 spill/reload，或从共享改为 pure recompute；
- 改变 local physical version/encoding 或选择另一个可实现 transfer；
- 缩短buffer lifetime、调整loop nesting或structured traversal order；
- 在capability允许时使用partial reduction或double buffer。

terminal Tile clone lower成unplaced Instr后，另一个有界层才可派生worker/slot/order/completion siblings；它们不反向成为pre-Instr
shadow schedule。若这些选择改变lifetime或packing feasibility，重新从相应unplaced parent materialize并执行fresh gates。

每个alternative都从未写offset的parent clone产生。structured alternative只重跑本层relation、materialization、coverage/verifier和safe bounds；
只有terminal survivor才执行bufferization、completion、lifetime、packing与全部exact late gates。Instr variant从canonical unplaced
Instr parent重跑completion、lifetime、packing和后续gate。allocator不返回repair recipe，不修改traversal/tile/layout/residency，
也不把rejected offset带入sibling。SPM allocator必须保留capacity/range/alignment/lifetime-valid baseline；actual high-water
只用于capacity/headroom。MiniMalloc的既有单次搜索可用accepted-offset-derived bank phase作
deterministic offset ordering的末级soft preference；不得为此改变hard feasible set、执行额外query、建立独立
candidate/relocation frontier，也不产生新的region、spill或join。

## 8. Layout、Movement 与 Residency 的共同选择

项目只保留 `MemLayout` 一套 physical representation。layout assignment 不能在 per-task 图上先独立选完：

1. consumer-driven propagation共同提出traversal coupling/materialization、候选tile relation/loop schedule和fanout closure；
2. component-local layout reducer计算可兼容的 physical versions和转换代价；它只删除已证非法或对future interface与
   全部选择敏感cost维度完全等价的proposal；
3. unified state共同选择 direct view、producer-native layout、consumer-local version、mapped DMA、GS 或 staged movement；
4. SPM capacity/lifetime bound以及DDR、GS、completion、compute/recompute、NoC、descriptor/resource cost和`Unknown`
   disposition全部反馈到同一frontier；SPM high-water只是headroom，不是winner cost；
5. selected clone用 memref encoding、view和explicit movement表达结果，proposal对象立即销毁。

relation composition、metadata-view folding、exact same-root/view transfer elimination、loop-invariant movement和通用GS descriptor
canonicalization仍作为policy-free canonicalization运行。它们只删除current IR上已经可证明的冗余，不保留all-or-nothing
full-buffer decision owner，不枚举全局选择，也不作为主融合完成证明。

权重/常量仍从typed、effect-proven read-only external root和consumer indexing/layout推导。当前通用路径优先把pure transpose/view与
contraction indexing合成oriented GEMM，再用exact composed relation把原始逻辑shape的weight从Tensor DDR mapped transfer到所需
physical version；它不假定DMA能执行任意transpose，也不要求package-time prepack。未来若package拥有typed persistent prepacked resource，才可让同一证明直接消费；
在此之前IR必须保留不能被oriented consumer吸收的真实conversion。不能每次invocation默认先读compact DDR、GS transpose、
再写回DDR，也不能用参数名推断预打包。

## 9. Completion 与 `NCCJoin` 重建

selected Tile IR 以 SSA dependency、effect 和 async event 表达未完成工作；region/task materializer不得在每个局部 return 插入
terminal join。`wafer.instr.ncc_join`只表示compiler派生的ordinary NCC participant completion；current source-observable ordering
由structured control-flow、SSA event/token和typed effects表达，DTE wait与group barrier各有独立typed op/event。若未来source
fence无法由这些事实typed区分，先扩独立op/verifier再进入本stage，绝不借用join provenance/name保留。terminal
Tile→canonical Instr与worker/slot/ready-order完成后，对complete-rank Instr IR
执行一次fresh completion reconstruction：

1. 识别 source-observable ordering/observer、external write/publication、typed collective/DTE protocol step和所有pending effects；
2. 从 worker order、event token、read/write range、alias、reuse和control-flow exits重算最晚必要 completion；
3. reconstruction入口先删除clone中全部`wafer.instr.ncc_join`；不靠名字、provenance字符串或“保守join”标记区分来源；
4. 在固定worker/order/effect frontier上，仅在跨worker/engine dependency、unsafe buffer reuse、协议要求或terminal external drain处
   构造latest-necessary participant completion，并合并所有可安全coalesce的join；
5. 验证每条 exit path已 drain all-and-only observable/pending effects，且无 join被当作 DTE wait或group barrier；
6. fresh重算全部roots的lifetime/coexistence/conflict，形成all-and-only fixed SPM allocation problems并执行3 MiB
   capacity placement与independent physical-alias validator；不做capacity tightening、quality probe或post-solve relocation；packing不得制造新的
   lifetime overlap。失败时销毁clone：worker/slot/ready-order/completion变化只能从canonical unplaced Instr parent生成并完整重跑；
   spill、retile、layout/physical-version或其它structured变化必须回到拥有完整语义的actual Tile parent生成新sibling，再重新lower。

同一 worker 上已有严格 issue order且无 range reuse/跨 engine可见性需求时，不额外插 join。collective completion是 execution cut，
不自动物化 store/reload。最终 join 数量由当前 effects和execution mapping决定，不能由 `tile.region`、task 数量或历史 builder site决定。

## 10. All-Rank Coordination 与原子性

all-rank coordinator和唯一tuple-level work ledger在第一个generation action前就建立，并持续拥有frontier、预算与actual parents，
直到terminal exact evaluation、winner selection和atomic commit结束；terminal evaluator只是服务，不接管frontier或发布局部winner。
rank-local analysis可以从actual clone派生transient factor summary，携带region boundary interface、collective/peer参数、
additive DDR contribution、GS/completion/critical-path max contribution、NoC link/endpoint contribution及`Unknown` disposition；
summary只供同一coordinator的factorized DP/beam使用，不形成rank-local winner、commit、独立budget或可发布artifact。含collective/peer
的组合逐项fresh重证，不能把summaries做成`N^R` Cartesian product。
worker/fixed-slot/ready-order也作为coordinator一次性作用于complete all-rank tuple的terminal action；每个action只生成匹配的actual
rank siblings并原子评估，不先为每rank建立`W^R`组合。typed exact failure返回同一coordinator；只有它能在剩余reserved repair
credits内从对应unplaced parent生成structured或terminal neighbor。
coordinator可以用current typed IR即时派生transient hash做保守预筛，但每个tuple必须重新读取actual IR验证：

- logical rank domain完整；collective algorithm、group、step、tile/segment 和 bytes在参与 rank 间一致；
- peer send/recv、message instance、physical range、wait、FSM/resource 和 failure path all-and-only匹配；
- cross-rank physical payload的 layout/footprint/valid domain一致或存在显式 pack/unpack；
- whole-variant DDR placement、external binding、target narrowing、ABI 和 package records原子通过。

coordinator不输出或保存semantic key/correspondence sidecar，也不使用task/layout/artifact ordinal恢复对应关系。任一rank或
late gate失败都销毁整个 tuple，不发布 partial offsets、bindings、module、bundle 或 package。reserved baseline也必须经过相同 exact gates。
每个terminal rank-entry Instr variant的全部SPM roots必须被fresh派生的fixed allocation problems all-and-only覆盖；每个complete
all-rank variant的全部DDR arenas/placement domains必须原子验证和汇总。planner problem/query数量进入统一work budget与diagnostic，
但不与entry、rank或region数量绑定为IR语义。后续winner selection/commit只重跑无搜索validator，不重新物化候选或调用planner。

## 11. Materialization 与 Pipeline 顺序

生产顺序收敛为：

```text
post-SPMD complete-rank structured IR
  -> structured normalization and exact relation analyses
  -> coordinated all-rank frontier + one deterministic work budget
  -> complete-rank consumer-driven component synthesis in actual clones
  -> terminal all-rank Tile variants
  -> per-rank complete-entry Tile-to-Instr conversion and bufferization
  -> worker / fixed-slot / ready-order Instr variants
  -> fresh dependency-driven completion reconstruction
  -> derive fixed SPM allocation problems from all roots / lifetimes / coexistence and validate all-and-only coverage
  -> derive DDR placement problems from explicit arenas/domains and atomically evaluate the complete all-rank variant
  -> post-memory Direct-DTE binding / all-rank message-resource / descriptor gates
  -> target ABI gates
  -> exact recost and atomic winner commit
  -> target module / package publication
```

如果 physical or completion choices 会改变 lifetime，必须在 SPM placement前完成；placement后的physical-alias、descriptor和
communication range只做fresh接受/拒绝并recost，不能反向修改completion或allocator。任何late rewrite改变root/view/effect/lifetime，都使旧offset、descriptor、
completion 和 cost失效并重跑对应 owner。

禁止的生产顺序包括：

- 单独 lower 每个 task 到完整 Instr 后再拼 complete rank；
- 在 task/region return处插 terminal join，再依赖 normalizer保留它；
- 先生成 spill/resident/layout 等 artifact 笛卡尔积，再按 ordinal拼 rank tuple；
- 先让每个局部候选通过 SPM/DDR并写 offset，再合并函数；
- 把 failed packing 的 repair 决策放进 allocator或 lowering；
- 在 target/ABI stage恢复上游 logical role或选择另一个实现。

## 12. 通用案例

### 12.1 Shared-input contractions

一个 producer同时供给三个 contraction。算法从三个 consumer tile共同反推 producer domain，比较一个共享 resident version、
若干 consumer-native versions和pure recompute。若 contraction 的 N loop 不改变 LHS tile，LHS load/layout conversion可位于 N loop外；
这来自 loop-invariance、SSA和relation，不来自“QKV”或“gate/up”名字。

### 12.2 Layout-changing chain

`producer -> permutation/view -> consumer` 中，exact composed relation可以形成 metadata view或让 producer直接生成consumer layout；
不可 exact 表达时生成显式 GS/staged sibling。layout reducer只删除已证非法或完整selection vector等价的proposal；
hard-capacity通过后由统一frontier按DDR、GS、completion、compute、tile utilization等完整cost保留siblings，SPM high-water只报headroom。
local cleanup只删除winner中的残余no-op movement。

### 12.3 Collective-connected chain

rank-local compute tile进入 all-reduce，再被下一段 compute消费。logical collective和participant completion保留；如果 selected
physical payload可被 collective与consumer共同接受，result可跨 collective保持 SPM resident。collective前后没有因 region boundary
自动出现 WDMA/RDMA，terminal drain只在真实 external output或resource reuse处建立。

### 12.4 Long reduction / attention

当 reduction domain无法完整放入一个 tile，consumer-driven traversal显式携带 partial或online state；score tile是否写DDR、
softmax是否 native/partial/online、V tile是否复用在同一搜索中选择。算法只读 structured reduction、matmul/index relation、dtype、
capacity和typed capability，不识别“attention”名字。

### 12.5 Fusion/tiling 对立候选与 selective spill

对同一个`producer -> consumer` chain，搜索至少形成三类可比actual candidates：single-region resident handoff配合耦合但可能
更小的tile；two-region显式DDR store/completion/load，允许producer/consumer分别选择更大的tile和loop order并缩短共同SPM
root lifetime；same-region的两套独立loop nests但保持可证明resident relation。前者节省DDR，第二类可能减少loop-expanded GS、
descriptor、completion和低利用率compute，第三类验证traversal separation不等于region cut；共同terminal cost决定winner。

对`A`长期live而`B`需要spill的graph，第三类candidate把二者保留在同一region，只对`B`显式store/completion/reload；
`A`继续使用同一SPM root。这个case证明`spill(edge) !=> differentRegion`，也证明同一region不等于所有edge都resident。

## 13. Llama Workload 证据与非规范示例

exact workload matrix、shape、oracle、收益阈值和board subset只有tasks/16拥有；本节不另设验收事实源，只保存设计输入证据并说明
不同执行形态为什么需要同一算法覆盖。当前 Llama-2 7B TP16、hidden 4096、intermediate 11008、FP16 block的fresh pre-Q49 IR
audit显示：

- `(S_q, S_kv) = (16, 16)` pre-Q49 fixture含52个按旧structured scope物化的static `tile.region`，
  且region输出普遍形成DDR边界；这是过度切分与高DDR/join的基线证据，不因为“52”这个数量本身非法。新搜索必须逐cut证明
  merge后的resident收益或保留split后的retile/容量收益。
- 一般rank的exact static-trip loop-expanded work约为137 RDMA、86 WDMA、205 GS和54个`NCCJoin`，特殊collective rank约56个join；
- per-rank DDR movement约82.4 MB，其中 runtime full-weight transpose 与 RHS physical conversion占主要 GS/DDR bytes；
- 当前 optimized/unoptimized路径的 DDR bytes相同，说明现有局部优化没有改变最关键的storage boundary；
- SPM high-water约97%，因此“全部 activation 常驻”不是合法默认答案，必须通过joint tiling/lifetime/packing选择。

这些数字是实现前的问题定位证据，不是完成时必须硬编码达到的常数，也不能替代tasks/16要求的C0 fresh baseline。已有IR机会分析投影：若oriented consumer、
effect-proven read-only weight的exact composed mapped transfer、共享LHS loop hoist、跨compute/collective residency与fresh completion
同时被选中，残余量级约为25.9 MB DDR、85次GS和3/5个join。它只是候选空间的opportunity estimate；只有fresh actual IR、
exact counters和matched execution才能证明最终收益，未达投影也必须按current typed capability解释。

tasks/16的矩阵分别覆盖短序列block、read-only cached-attention与prefill：前者暴露shared input、weight physical version、
native reduction、fusion后tile缩小与internal DDR materialization后独立retile的取舍，以及join重建；cached-attention暴露小query/长KV
traversal、Q/state resident与K/V internal streaming、partial/online state及bounded SPM；
prefill暴露score domain不能整体驻留时的full-matrix round-trip风险。这些只是通用SSA/reduction/storage关系的不同压力点，
不形成算法阈值或matcher。read-only cached-attention只验证一次invocation内对显式K/V的读取；真正decode还需要KV cache写入、
position和跨invocation persistent-state合同，当前tasks/12/15尚未提供，必须在这些前置闭合后另行纳入gate。

## 14. 迁移与删除边界

### 14.1 保留并复用

- structured op interfaces、`TilingInterface`、DPS、`IndexRelation`、physical relation/transfer realizability；
- 唯一 `MemLayout`、Q46 relation fixed-point 和 layout cost primitives；
- actual-clone transaction、cost/Pareto/all-rank atomic coordinator；
- typed Tile/Instr conversion、fixed-capacity SPM/DDR planners、worker/fixed-slot、collective/peer expansion、post-memory Direct-DTE
  binder和typed ABI；
- source numeric algebraic/recompute、canonicalization 和 exact verification infrastructure。

### 14.2 重新定位

- layout PBQP从 per-task独立 winner变为 component-local factor reducer；
- worker、fixed-slot和ready-order从已经lowered task的后处理变为terminal Instr siblings；collective/peer tile、payload、layout与
  residency在structured frontier联合选择，Direct/Ring/Tree参数在terminal conversion逐点生成actual Instr sibling，post-memory
  Direct-DTE binder只验证和绑定，不重新决策；
- relation/view motion、LICM、exact same-root transfer elimination和通用GS descriptor cleanup只做policy-free canonicalization；
- completion normalizer从“保留已有 terminal joins”变为在final complete-rank effects上fresh reconstruction。

### 14.3 Production cutover 后删除

- per-task `CandidateEvaluation -> Instr -> SPM/DDR`生产路径；
- lowered task import/commit和task-return terminal join；
- spill/resident/ready artifact Cartesian product、physical artifact kind和全局 task/layout ordinals；
- all-or-nothing full-buffer handoff decision owner与其它独立scope/layout/communication selectors；
- 任何通过名字、shape、operand顺序或Llama角色恢复dataflow的 matcher；
- 任何与 actual IR并行保存的 component、schedule、repair 或 proof sidecar。

删除必须发生在新的默认 driver 已覆盖同等 conservative baseline、全部 current typed capabilities和late exact gates之后；不能只把旧路径
关闭但继续维护。archive保留历史证据，不作为当前协议。

## 15. Failure、Determinism 与 Diagnostics

failure分层：

- `InvalidIR`：source/current clone违反自身verifier，终止对应candidate或invocation，不能伪装成target不支持；
- `UnsupportedRepresentationOrTarget`：只停止对应edge维度/transition；只有所有跨越实现均不支持时才形成component separator；
- `ProvenInfeasible`：numeric、relation、capacity、alignment、range、descriptor或completion已给出typed反证，拒绝clone；
- `ResourceExhausted`：solver/packer自身资源耗尽且未证明infeasible，保留baseline并单独诊断；
- materialization/conversion/verifier失败按上述真实类别归因并销毁actual clone；decision owner只从unplaced parent生成有界sibling；
- all-rank message/resource/ABI不匹配：拒绝完整tuple；
- `BudgetExhausted`：停止非baseline扩展，不能报告为legality/capacity失败。

diagnostic只描述 current IR、typed decision和owner failure class；不含任务号、candidate ordinal、模型专用角色或临时构建路径。
在相同输入与target facts下，component traversal、neighbor order、Pareto tie-break、all-rank组合和winner digest必须确定；并行执行不能改变
结果。每个 bounded search始终保留同一 conservative baseline，若 baseline本身不合法则明确失败，不偷偷回退旧pipeline。

## 16. Completion Gate

本文终态同时满足：

1. 默认 `wafer-compile` 在 complete-rank structured IR 上进行consumer-driven联合综合；没有production per-task先lower再拼接路径。
2. task、source scope、component、traversal和local builder边界不自动产生region、DDR movement、SPM arena释放或terminal join；
   region partition与traversal fusion/materialization、tile/loop、layout/version和resident/spill/recompute联合选择，并物化为一个或
   多个non-nested `tile.region`。SPM value/root/alias不跨region；region内允许逐value streaming/selective spill，region exit只
   验证仍访问其roots的pending completion，entry terminal闭合observable work，不因结构边界插join。
3. chain、diamond、fanin/fanout、multi-root、loop-carried reduction、structured control flow和collective都以通用interface覆盖；
   source正负例经完整driver，不靠名字/shape matcher。
4. implementation、tile、layout/physical version、share/recompute和residency/transport进入structured frontier；worker/slot/
   completion在terminal Instr variant层由同一decision owner协调，PBQP/allocator/lowering保持各自边界且不存在跨层shadow plan。
5. candidate只有actual clones；accepted IR不依赖task/artifact/layout ordinal、shadow plan、side table或proposal attr。
6. cheap proposal只在clone前作relation/type/effect/footprint rejection；每个frontier survivor的决定已物化进actual clone并
   重跑其层内verifier。terminal variant从fresh current IR派生SPM/DDR allocation problems，all-and-only覆盖roots/domains并
   原子执行communication/ABI exact evaluation与final recost；problem/query数量只受统一预算约束，不是region/entry固定语义。
7. completion从final current Instr effects/event/ranges fresh重建；无task-return join泄漏，join、DTE wait和group barrier语义不混用。
8. work/cost分别记录static site、可证明static-trip loop-expanded exact work、conditional-path lower/upper bound和
   symbolic/`Unknown`；Q9独占runtime-measured count。DDR、GS、NCC/join、Instr、SPM peak和critical path分别可审计，
   不以static site count冒充执行work，也不把runtime dynamic count写回compile-time cost。
9. Llama短序列、read-only cached-attention和长prefill矩阵完成同源baseline/winner IR dump、structural counter、CPU/model oracle、
   complete package和fresh no-card；unsupported/skipped不算通过。
10. compiler-side gate通过后达到`board-ready`；真实设备按FP16/BF16、单进程、逐case、bounded timeout完成exact output/guard和
    matched baseline/winner性能，才可标`done`。
11. 旧per-task evaluation/import、artifact Cartesian product、task-return join和并行decision owners已删除，而不是仅禁用。
12. 同一通用source形成并比较single-region fused-small-tile、multi-region separated-large-tile、same-region separated traversal和
    selective spill等actual candidates；region partition、独立遍历与DDR materialization分别作为选择，不以region数量替代
    DDR/GS/completion cost，也不以单条spill结束无关root lifetime。
13. all-rank coordinator和唯一work budget在generation前建立；允许不可发布、不可独立选winner的rank/component factor summary，
    但不存在rank-local commit、`N^R` tuple或C3/C4重复
    materialization/exact planning。final winner由当前校准的hardware cost model决定，stable semantic order只是等cost末级tie-break。

## 17. 参考机制

这些工作只提供算法/工程启发，不是可复制协议：

- [Welder](https://www.usenix.org/conference/osdi23/presentation/shi)：tile graph、memory traffic和fusion联合搜索，
  说明layout/tiling/fusion不能各自贪心。
- [Unity](https://www.usenix.org/conference/osdi22/presentation/unger)：联合优化代数变换与并行化，说明相互影响的选择不应由
  独立贪心stage提前固定。
- [Rammer](https://www.usenix.org/conference/osdi20/presentation/ma)：hardware-aware task orchestration及inter/intra-op
  scheduling共同影响利用率。
- [AStitch](https://doi.org/10.1145/3503222.3507723)：以hierarchical data reuse、consumer-driven thread-mapping propagation
  和resource-aware scope扩大memory-intensive fusion，启发fanout与资源共同决策。
- [Mirage](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi)：以kernel/thread-block/thread多层实际程序表示、
  abstraction pruning和带理论保证的probabilistic equivalence verification自动搜索kernel实现。
- [SpaceFusion](https://doi.org/10.1145/3689031.3696087)：用Space-Mapping Graph联合表达算子内/算子间空间依赖，并按硬件
  resource configuration自动生成fusion schedule，启发tile、fusion和mapping共同建模。
- [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)与
  [Bufferization](https://mlir.llvm.org/docs/Bufferization/)：structured semantics、destination style、tile接口和one-shot analysis边界。
- [IREE Stream](https://iree.dev/reference/mlir-dialects/Stream/)：显式resource lifetime与execution scheduling分层。

当前设计仍以本仓库typed target、约3 MiB可用SPM、16-rank execution/communication、现有Instr/ABI和exact verifiers为准；
外部方案不能替代当前 capability、numeric、memory 或 board证据。
