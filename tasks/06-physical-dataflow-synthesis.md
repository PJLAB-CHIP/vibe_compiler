# Wafer Physical-Dataflow Synthesis 与 Candidate Selection

状态：2026-07-20已同步Q32.V typed target-capability vertical；mapped DMA、physical-footprint fill和
versioned oriented GEMM已形成typed compiler/formal/SystemC闭环，后续由Q32.M接入共同candidate owner。
实现状态以`tasks/progress.md`为准。

本文是 rank-local physical-dataflow candidate generation、candidate acceptance 和 all-rank atomic commit 的唯一设计
owner。它不定义第二套 semantic IR、provider registry、shadow schedule、序列化 frontier 或 qualification 协议。07、08、
10-13分别拥有 selected tile/dataflow IR、physical realization、compute/movement、instruction、memory 和 communication
合同；它们向本文提供 IR、interface、analysis、rewrite、conversion 和 exact verifier，不再发布另一份 planning protocol。

当前 production 是 Q29 已完成的 bounded tile-dataflow scheduler。Q32 从这条链路增量演进：先用一条真实、可验证的
dependent-tiling/resident rewrite打通MLIR-native路径，再补齐implementation、tile、encoding、storage realization、route、
residency、buffering/order和communication的有界联合选择，最后原子替换旧decision owner。施工顺序从窄纵向开始，不表示
终态功能只剩两条rewrite；没有实际rewrite、下游消费和完整gate的接口或框架不进入production。

## 1. 核心结论

Wafer 需要联合评估 implementation、tile、physical encoding、storage realization、transfer route、residency、
buffering/order和communication对最终program的影响，但不需要一个平行于MLIR的巨型联合求解协议。主线固定为：

```text
verified structured MLIR
  -> op/type/attr interfaces + recomputable analyses
  -> bounded composition of policy-free rewrites on isolated clones
  -> typed tile/instruction IR through dialect conversion
  -> existing exact SPM/DDR/event/transport/instruction/ABI gates
  -> static comparison
  -> existing all-rank coordinator and atomic ExecutableBundle commit
```

设计原则：

1. **当前 MLIR IR 是唯一语义事实源。** iterator、indexing、DPS tie、scalar body、effect、layout、message 和 completion
   不复制进 detached descriptor、query record 或 compatibility record。
2. **interface 表达局部能力，analysis 表达可重算关系。** 单 op/type/attr 的行为放 interface；跨 value/op 的关系由
   当前 IR 派生的 analysis/helper 计算；最终选择进入 typed IR。
3. **候选是 IR clone，不是 shadow plan。** rewrite 被接受后立即修改隔离 clone；后续 legality、resource 和 cost 只读取
   该 clone。rewrite 前的 proposal 不得与 materialized IR 并列成为事实源。
4. **机制先于搜索。** 每种relation/view normalization、tiling/fusion、implementation absorption、encoding/view选择、
   route materialization、physical-version reuse、movement/resident-cut elimination、buffering/order或collective expansion
   必须先作为独立rewrite在真实source上通过correctness和完整downstream gate；搜索只组合已验证rewrite及其typed参数。
5. **联合评估不等于一次性联合求解。** 各选择轴可以按稳定顺序分层展开，约束也可以在每次clone rewrite后fresh传播；但
   Q32完成时必须实际覆盖全部当前支持轴，而不是只把两个固定rewrite包装成planner。
6. **合法 baseline 永远保留。** 优化预算只限制新增候选，不得把合法输入变成编译失败；baseline 自身失败仍按真实
   pipeline failure 报告。
7. **不伪造硬件性能。** 板端校准前只比较final IR可证明的静态量；先做exact Pareto pruning，再由target profile显式拥有的
   static selection policy处理已知tradeoff。该policy是编译器取舍，不称为硬件时间；缺少policy或关键量Unknown时回到baseline。
   Q9才拥有hardware-calibrated ranking。
8. **目标能力与运行资格分离。** compilation 只接收明确的 `TargetProfileId`/target capabilities。model 或 board
   qualification 是 downstream admission/evidence，不进入候选生成、过滤或排序。

### 1.1 功能目标保留矩阵

删除平行协议不能删除优化能力。Q32终态至少保留下列目标，并用MLIR-native owner实现：

| 功能目标 | MLIR-native实现 | selected / completion evidence |
| --- | --- | --- |
| parameterized implementation选择 | source OpInterface；上游op用external model；选择后立即materialize为typed `wafer.tile.*` | 至少两个真实implementation参数点进入complete clone，非baseline实现被真实source选中 |
| dependent tiling、fusion和relation propagation | `TilingInterface`、DPS、Affine/Presburger/ValueBounds `IndexRelation`、PatternRewriter；traversal capability由具体op的tile mapping与legality决定，不按dialect/op family一刀切 | dependent region、当前production启用的terminal shape-preserving all-reduce、tail和all-and-only coverage可重证，最终IR实际删除重复whole-tensor work/movement |
| physical encoding与metadata view | encoding attr/type interface、标准view op和显式Wafer physical op | 至少两个current target encoding/view/materialization alternative形成不同IR并通过physical verifier |
| transfer route与storage realization | 跨两端buffer的analysis/helper；zero-copy/direct/current GS/staged alternative立即变成view/movement/temp/event IR | route不是sidecar；final descriptor、range、valid-lane和completion从当前IR重证 |
| residency与physical-version reuse | SSA root/view、alias/effect/lifetime分析和typed rewrite | chain、fanout及partial-compatible use能有界复用；spill/reload cut在winner中真实消失 |
| buffering和resource-aware DAG-legal issue order | 实际buffer SSA、loop-carried value、async token/wait/fence和resource effect；从current SSA/effect DAG生成少量ready-order alternative | current Q29能力不回退；至少一个非source-order alternative改变真实buffer/token/order IR、通过lifetime/resource gate并成为winner |
| communication algorithm选择 | collective OpInterface、topology helper，以及Direct、Ring和保持`rank_group`中序的ordered-Tree complete-clone rewrite | alternative进入同一rank frontier；all-rank coordinator只从最终p2p/message/completion IR重证，floating collective的ordered Tree与会置换归约leaf的Ring分别做数值合法性判断 |
| resource-aware candidate generation | 从当前clone重算liveness、capacity lower bound、descriptor/resource pressure；邻居仍必须物化完整IR | resource事实能产生有界tile/residency/buffering/order邻居，09/12/Q34继续作唯一exact placement gate |
| bounded joint search和selection | baseline slot、actual IR clones、有界worklist/frontier、exact cost和existing all-rank transaction | 全部producer受生成/materialization/whole-variant hard cap；预算耗尽保留合法baseline |
| target能力纵向 | Q32.V已在source/ODS/type/interface、Instr、TargetCall、ABI和SystemC中闭合mapped DMA、physical fill、oriented GEMM | Q32.M起由共同owner消费这些typed能力；external model/board admission仍不参与compile-time choice |
| whole-tensor share-vs-recompute | SSA use-def、`TilingInterface`、IndexRelation及effect/speculation proof分别物化共享version和按consumer dependent region重算的clone | share与recompute各有production winner；compute work、movement和live-range/high-water从各自final IR比较 |
| static loop-invariant hoist | `LoopLikeOpInterface`、dominance、SSA、effect/completion proof和PatternRewriter直接移动真实op/value | 至少一个hoist winner从loop外dominant SSA取值并重跑lifetime/placement；不可移动或资源更差时保留baseline |
| fixed Cx/NCx encoding absorption | existing typed encoding、IndexRelation、physical-map/valid-lane proof让已有target contract接受Cx/NCx的family直接消费physical version；current限GEMM/batched GEMM | Tensor↔Cx/NCx `materialize_layout`、GS或等价pack/unpack movement在winner中真实消失；其它compute family不自动获得该能力，也不新增虚构的vector-width/packing参数 |
| integer-domain exact/modular-proof-gated algebraic variants | 从current integer IR及其overflow/wrap语义证明reassociation、显式rank-local reduction tree和algebraic distribution/factorization的exact/modular子集 | 每个current variant各有独立production正例/winner和无proof负例；tree/order必须是actual SSA/SCF；所有floating rank-local algebraic rewrite及`contract`-based FMA均不属于current |

`IndexRelation`、候选worklist、cost和资源摘要可以是transformation-local C++ analysis/state，但不是新的IR、wire schema或
跨stage事实源。目标是保留上述能力闭包，同时删除descriptor/provider/query/key/registry/telemetry等重复语义。

### 1.2 功能进入production的交付合同

功能写进表、interface存在、pass已注册或局部测试能调用都不算交付。上表每个功能必须沿同一条主线形成可审计证据：

```text
Q15 verified source
  -> the production candidate owner discovers the opportunity
  -> PatternRewriter/DialectConversion mutates an actual isolated clone
  -> typed tile/instruction/memory/effect/token IR carries the choice
  -> rank-local and all-rank exact consumers accept that clone
  -> the clone enters the common frontier and final-IR selection
  -> final-IR policy selects it for a production source and atomically commits ExecutableBundle/package
  -> bundle/readback/SystemC evidence is derived from that winner IR
```

Q32.M证明每个producer真实产生并被完整下游接受；Q32.S证明每个producer进入同一bounded frontier，并让每个可选择功能至少在一个
非workload特化的production source上成为winner；Q32.G证明这些winner由默认`wafer-compile`路径产生并提交，而不是只存在于
Q32.B test seam、`wafer-opt`手工pass、隐藏feature flag或测试专用callback中。required normalization/closure没有独立选择分支时，
其mutation效果必须保留在committed winner中。

“capability-conditioned”只表示一次应用必须满足source numeric/effect、target typed capability和resource proof，不表示可以省略
实现、production接入或正负测试。本文列为Q32功能的conditional row必须至少有一个predicate-positive production winner；若当前
target纵向无法表达，必须在Q32完成前把它移动到有明确前置的Later row，不能留下已注册但永远不被production消费的死功能。

floating rank-local algebraic reassociation/reduction-tree rewrite、generic online reduction、non-GEMM FMA contraction以及超出
current integer-domain exact/modular子集的algebraic distribution/factorization不属于已完成Q32合同；当前Q32.N
`numeric-algebraic-extension`按`tasks/plans/numeric-algebraic-extension.md`补齐production source permission、
typed selected/target consumer和低精度纵向。任一feature在对应consumer闭合前不得注册production candidate，也不能拿
target固定FMA语义或手写`wafer-opt`测试冒充采用。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Shardy/XLA SPMD 之后按 logical rank 静态 specialize、通过当前 verifier 的
  Linalg/Tensor/SCF/Arith/Math structured tensor program；logical collective、native MLIR numeric/effect/control semantics、
  ExecutionConfig 和 TargetProfileId 均已显式。
- Current stage responsibility:
  直接通过structured op interfaces、SSA use-def、type/shape、indexing map、effect、native numeric permissions/semantics及
  transformation-local integer-domain exact/modular proof识别有界analysis scope；从当前IR重算IndexRelation、alias/root、liveness和resource
  facts；在隔离complete-rank clone
  上分层枚举并立即物化implementation、tile、encoding、storage realization、route、residency、buffering/order和
  communication alternatives。每次mutation后fresh重算analysis，再lower到typed tile/instruction IR并复用现有
  rank-local finalization、SPM、instruction/geometry/completion gate；现有all-rank coordinator再有界组合rank
  survivors，对每个complete variant运行whole-variant DDR、post-memory transport binding、all-rank resource、
  target ABI和package-eligibility pure gates，最后按final facts做resource-aware Pareto/static-policy selection并原子提交。
- Output artifact / IR:
  atomic ExecutableBundle；每个 logical rank 恰有一个覆盖完整静态 traversal、已完成 placement/binding、
  不含 Tensor/Linalg/search residue 的 instruction/memory/completion program。candidate clone、analysis、rewrite
  proposal、cost trace 和调试统计不跨越该边界。
- Downstream consumer:
  14直接从 winner instruction IR 做 target conversion/module publication；15形成 package；17的
  SystemC/CModel消费同一 target program。07-13是本stage调用的 IR/lowering/verification services，不是并列
  accepted artifact producer。
- User-level driver / named pipeline:
  wafer-compile source-to-bundle named production pipeline。wafer-opt只可调用同一底层 rewrite/conversion 做局部
  调试，不拥有另一套 candidate selector 或 accepted artifact。
- Explicit non-goals:
  不创建detached semantic descriptor、provider/query/key/registry、canonical frontier/candidate serializer、版本化
  诊断统计协议或Transform control plane；不做whole-model equality saturation、全维Cartesian product、名字驱动优化，
  不让allocator反向修改candidate，也不让model/board qualification参与compile-time choice；不承诺dynamic shape、
  多实例dynamic loop、未经event证明的ping-pong、persistent prepack、cycle timing或board性能。mapped DMA、physical fill、
  oriented GEMM由Q32.V独立typed target纵向实现，winner capability projection仅在真实
  package/runtime consumer需要时由14-17从final Instr/TargetCall派生；二者都不是planner-side语义协议。
- Completion gate:
  功能目标保留矩阵中的current-target选择轴全部进入同一bounded candidate owner：真实source至少选择一个非baseline
  implementation、一个改变encoding/view/materialization或route的alternative、tiling/resident与multi-use reuse/movement
  elimination机制、share-vs-recompute、static loop-invariant hoist、fixed Cx/NCx encoding absorption、各自integer-domain
  exact/modular-proof-gated algebraic variant、resource-aware static buffering/ready-order及direct/ring/tree communication
  alternative；Q32.V
  mapped/physical-fill/oriented纵向独立闭合后由同一owner消费。每次rewrite后derived analyses均fresh重建；baseline和optimized
  clone经过相同exact gates；每个choice producer完成§1.2的discover→materialize→accept→select→commit链，required closure的
  mutation保留在committed winner；resource-aware selection实际读取validated placement/high-water和final metrics；rank-count=1/16、
  通用拓扑、7B scale和完整PyTorch/SystemC数值验证通过；
  production只保留一个decision owner，旧scope/layout/maximal-resident、communication selector和未校准scalar-time旁路删除。
```

## 3. 稳定职责和对象

### 3.1 Structured source interfaces

source op 的语义首先来自 upstream MLIR interface：

- `linalg::LinalgOp`/structured indexing maps；
- `DestinationStyleOpInterface`；
- `TilingInterface`和subset/view语义；
- `MemoryEffectOpInterface`及必要的MLIR `SideEffects::Resource`；
- SSA、region/control flow、type/rank/shape/dtype和显式numeric attrs。

Wafer 需要暴露“该 source op 在当前 target 上有哪些有限实现形态”时，在 source op 上定义一个窄的
`WaferTargetImplementationOpInterface`。Linalg/Tensor op通过 MLIR external model实现，无需修改上游 dialect。它可以接收
immutable target capabilities和当前已知约束，返回少量 typed `ImplementationCandidate`：

```text
ImplementationCandidate {
  implementation_kind
  typed_parameters
  operand/result encoding constraints
  tile constraints
  numeric preconditions
}
```

该对象是一次 rewrite 调用的普通 C++ value：不版本化、不序列化、不含 op/value 名字、不带 materializer key、registry
digest 或 cache identity。候选被选择后立即由对应 rewrite 构造 typed `wafer.tile.*`；selected op 的合法性由
typed op/attrs、ODS verifier、适用的标准MLIR interface和conversion legality验证。

当前 target profile 是 closed registry。没有第二个 target 实现时，普通 typed helper也足够；只有出现多个 dialect/target
实现且确有共同 consumer 时，才评估 `DialectInterface`。不得为了未来插件化先建立动态 provider registry。

Q32.I已完成首轮native-interface reuse audit：只重新枚举DPS inputs/outs/results的
`WaferTilingInterface`已迁移到`TilingInterface`、`DestinationStyleOpInterface`和typed operands/results后删除；
实现与consumer盘点证据见`tasks/archive/mlir-native-implementation-relation-foundation.md`。
Q32.M已完成后续native-interface reuse closure：layout requirement由typed memref encoding、标准view/subset
语义和op verifier表达，effect由`MemoryEffectOpInterface`及MLIR `SideEffects::Resource`表达，bytes/footprint
从current IR重算；原`WaferLayoutOpInterface`、`WaferLayoutMaterializationOpInterface`与
`WaferResourceEffectInterface`及其重复record已删除。证据见
`tasks/archive/physical-mechanism-choice-closure.md`。

structured traversal root的切块能力同样必须消费current op的`TilingInterface`，不能把全部logical collective长期
归为whole-tensor-only。shape-preserving `all_reduce`和`collective_permute`允许沿任意result tensor维切块；
`all_gather`、`reduce_scatter`和`all_to_all`只允许接口证明不跨越其gather/scatter/split/concat受限轴的tile。
候选物化必须从terminal collective result tile反向融合同尺寸producer slice，并把每个tile的compute、communication和
writeback保留在同一complete compact traversal中；接口拒绝的tile只拒绝该候选，不能靠op名放宽，也不能把可切root
降级成一次full-buffer尝试。

Q35 production enablement只开放verifier已证明shape-preserving、单输入单输出的`all_reduce`作为terminal tiled root；
其它collective虽然已有接口级tile mapping，仍保持`FullTraversalOnly`，直到各自的producer fusion、受限轴、control-instance
和下游transport gate有独立正负例。接口“能描述某个tile”不等于该op已经进入production candidate search。

`WaferTargetImplementationOpInterface`已由Q32.I用真实reciprocal/division actual-clone贯通；之所以允许保留，
是因为标准MLIR interface不表达Wafer target implementation
参数枚举；它不得重新发布tiling、DPS、layout或effect语义。任何其它Wafer-specific interface必须列出标准接口缺口、
至少两个真实op family和generic consumer；否则使用typed pattern/helper，不新增动态语义层。

Q29迁移审计曾使用的`StructuredSchedulingTilingDemand`和`StructuredSchedulingLayoutPlan`已随旧decision owner删除。
Q32从current op/interface/IndexRelation/encoding直接重算所需analysis；普通局部analysis可以保留，但不能复制一份
selected tile/layout事实。

### 3.2 IndexRelationAnalysis

`IndexRelation`描述 consumer logical index 到 producer logical index 的可组合关系，用于 dependent tiling、view legality、
producer fusion、transfer cover和rewrite coverage proof。它是从当前 IR 派生、可失效、可重算的 analysis value，不是 op、attr、
sidecar或跨pass协议。

实现优先复用：

- structured indexing `AffineMap`；
- MLIR Presburger集合/关系；
- `ValueBoundsOpInterface`和相关 bounds utilities；
- `TilingInterface`、subset/view/reassociation语义。

Q32按真实consumer分批实现、但完成时必须覆盖当前优化所需的关系闭包：identity、projected permutation、broadcast、
static slice、reshape reassociation、concat/分段view，以及这些关系的composition。至少提供dependent-region preimage、
image、functional/injective/bijective分类、exact equivalence/implication和与valid domain的交；pointwise propagation、
view folding、transfer cover和multi-use reuse都必须成为实际consumer。piecewise关系优先直接使用MLIR Presburger relation/set
及其整数约束表达，不能再定义带version/digest/parser/printer的私有relation wire language。

施工仍遵守consumer-driven：Q32.I/R只实现下一批mechanism直接需要的MLIR-backed子集；每增加一种表达必须同时增加真实
consumer、composition/property test和unsupported path。这个顺序限制首批代码量，不降低Q32 integrated completion的关系覆盖。
无法精确表示时返回unsupported并保留显式movement/spill baseline。

analysis API必须区分：

- exact relation/proof；
- conservative bound，只可用于安全剪枝或诊断；
- unsupported representation；
- invalid IR；
- subsystem-owned resource exhaustion。

exact rewrite只能消费 exact result。任何 applied rewrite 都结束当前 IR epoch：旧 `Operation*`/`Value` binding、relation、alias、
effect、lifetime和cost全部销毁，从修改后的 clone fresh重建。不同 clone之间只能共享不含 IR引用的 immutable target facts。
relation expression/node/piece和composition work使用analysis owner的本地hard cap；cap耗尽只拒绝对应optimized expansion，不能
缓存partial relation或把合法baseline判成unsupported。

### 3.3 Physical Encoding 与 Transfer Realizability

selected physical encoding是 type/attr语义。footprint、alignment、valid domain和 logical-index-to-physical-offset 行为属于
physical encoding attr/type或其 interface；implementation interface只声明它能接受的 encoding constraints。

Cx/NCx具有 tail/padding和可能非 affine 的 physical mapping，不能伪装成通用 `MemRefLayoutAttrInterface` 能解释的 affine
layout。标准 memref view只用于其语义确实成立的 encoding；其余 reorder/materialization必须由显式 Wafer op表示并由08验证。

transfer route依赖 source root/view、destination root/view、两端 encoding、IndexRelation、alias/effect和target DMA/GS限制，
不是单 op 行为。08提供普通`TransferRealizabilityAnalysis`/typed helper，返回少量即时alternatives。选中后直接在clone中物化：

- zero-copy view；
- direct load/store或mapped movement；
- explicit temporary、layout materialization、DMA/GS和event/completion。

物化后的 IR 是唯一事实源；route proposal随后销毁。不存在 route query schema、route id sidecar或resolved schedule record。

Q32.R已把`StorageLoadOp`迁移为显式DDR source + 已创建SPM destination、无隐式allocation/result；candidate
materializer先创建allocation/view再发load，tile-to-instruction conversion直接消费这两个typed operands。
同批`TransferRealizability`从两端memref、encoding和`IndexRelation`现场证明current metadata view、compact DMA、
GS与staged route；relation-backed full-buffer rewrite把producer SPM root通过tile-region SSA交给consumer并删除真实
WDMA/RDMA。它们已贯通非7B source和标准7B exact gate，但尚未替代Q32.B之后的共同candidate owner与frontier。

### 3.4 Communication planning

logical collective语义只由当前 collective op/interface表达。topology/target helper可以枚举少量 typed algorithm parameters，
例如 direct/ring/tree 及必要 chunk 参数；每个 alternative必须直接在 complete-rank clones 中展开为真实 p2p、local compute、
staging、token和wait IR。

all-rank correctness继续由现有 coordinator从当前 instruction IR收集 message、buffer range、completion和binding并重算。允许用
一个从 typed collective/message IR 派生的轻量 grouping key减少不可能组合，但它不承担 correctness，也不复制完整 message
claims。不存在 CommunicationScheduleSkeleton、ResolvedCommunicationSchedule 或 transport signature协议。

### 3.5 Candidate 与 Evaluation Clone 生命周期

candidate的语义主体始终是actual IR clone，但生成与exact evaluation必须分开：

```text
generation worklist entry
  = unplaced actual complete-rank clone + semantic generation ordinal + next-decision cursor

rank frontier entry
  = separately evaluated actual rank clone + semantic generation ordinal + physical artifact kind + rank-local exact facts/safe bounds

whole-variant evaluation
  = disposable clone of one generation- and artifact-kind-compatible complete rank tuple + whole-variant exact facts
```

generation clone不带owner-produced SPM/DDR offset、Direct DTE binding或ABI/package派生事实。rank-local
conversion/SPM/geometry/completion在它的独立evaluation clone上执行；通过后该evaluation clone才进入rank frontier，且绝不返回
generation worklist。whole-variant DDR、transport binding和最终gate再在rank tuple的独立variant clone上执行。

`next-decision cursor`只是在稳定IR遍历中定位下一处尚未展开的选择，不描述已选语义；它在clone mutation后重新定位，不能保存
跨rewrite的`Operation*`/`Value*`。生成中可以暂存一个typed rewrite参数点，但参数一旦applied，后续选择只读修改后的generation
clone。任何work item/frontier entry不得携带另一份implementations、physical versions、schedule DAG、allocation map、message
claims或canonical serialized IR；evaluation facts只是invocation-local analysis/cost，不是shadow semantics。

bounded worklist/frontier是普通invocation-local容器，不是跨stage对象。它可以同时保存多个实际clone，以便组合
implementation、tile、encoding、route、residency、buffering/order和communication；不能以“禁止shadow plan”为由只保留
两条固定rewrite，也不能为这些clone另造语义摘要来替代IR。

determinism来自稳定 IR traversal、固定 proposal insertion order和明确的比较规则，不通过重新序列化完整 IR建立 identity。
memoization只有在 profiling证明必要，且 key能由局部 immutable value安全构造时才加入；不得缓存带 IR pointer 的 derived analysis。

task recipe只消费稳定passing顺序中的`taskAlternativeOrdinal`，因此收集到
`taskAlternativeOrdinal + 1`个通过全部rank-local gate的candidate后必须停止继续展开；并行执行可以让已经提交的固定有界batch
完成，但不得据线程完成顺序提前选择或继续消费后续queue。并行evaluation由一次rank-frontier invocation内的有界worker
executor承载：每个worker独占并复用一个完整注册的`MLIRContext`，并在输入文本未变化时复用只读standalone task；
candidate仍在该context内独立materialize。通过gate的
完整module用仅供本次owner handoff的MLIR文本导入owner context，owner fresh verify并从导入IR重算cost；该临时传输既不建立
candidate identity/缓存/schema，也不得触发第二次Tile→Instr→SPM→DDR lowering。serial与parallel必须产生相同selected module；
fixed batch只允许evaluation count有界overshoot，不改变queue order、passing ordinal或winner。

### 3.6 Selected IR

跨stage保留的事实必须由 typed IR表达：

- memref root/view、memory space和accepted encoding；
- selected target-abstract compute、movement和collective op；
- resident SSA handoff、spill/reload和immutable input；
- async issue、producer completion、join、reuse和terminal wait；
- complete static traversal、tail、reduction chain、SPM/DDR offsets和transport binding。

fusion label、group、scope-prefix、candidate score、planner choice attr、route id、analysis digest和shadow schedule不得进入selected IR。

## 4. Transformation 和 Conversion 边界

### 4.1 同层 transformation

structured tiling、producer fusion、subset/view folding、pointwise propagation和resident handoff使用 `RewriterBase`/
`PatternRewriter`及 upstream Tiling/structured transformation utilities。所有 mutation必须通过 rewriter，使 listener、tracking和
诊断看到真实改写。

每个 rewrite应满足：

- 只读取当前 IR、immutable target facts和本次调用局部 analysis；
- precondition由 interface/type/effect/numeric/relation证明，不按名称、模型或operand位置猜；
- applied后直接得到 verifier-legal的同层 IR；
- not-applicable不修改 clone；
- 失败不串接隐式 fallback，baseline由外层 candidate generation保留；
- 不读取全局candidate score、beam状态或历史尝试。

不建立 `MechanismSpec`、mechanism key/registry、qualification set或版本化 outcome协议。rewrite进入主线的资格由代码调用和测试
证明：production candidate generator真实调用、至少一个通用source发生改写、完整 exact gates接受，并有negative coverage。

Q32.M的mandatory mechanism closure如下；它描述必须交付的功能，不要求为mechanism建立registry：

| mechanism / closure | IR cut | 最低功能域 |
| --- | --- | --- |
| relation/view normalization | structured tensor | identity、permutation、broadcast、slice、reshape、concat/分段view的composition与exact cover |
| dependent producer tiling/fusion | structured tensor | DPS pure producer→consumer、static tile、tail和effect/numeric barrier |
| pointwise relation propagation | structured tensor | 多operand exact relation对齐，不能靠same-shape猜测 |
| implementation materialization/absorption | structured→selected tile | 至少一个非baseline current implementation；已闭合Q32.V typed GEMM orientation；current Cx/NCx packing identity只留在encoding |
| encoding/view/materialization choice | selected physical payload | current Tensor/Cx/NCx及合法metadata view和显式reorder/materialization |
| boundary transfer choice | selected physical payload | zero-copy、current compact DMA、GS/staged；Q32.V后增加mapped DMA |
| multi-use physical-version reuse | selected physical payload | fanout、partial-compatible maximal subset和immutable input reuse，受hard cap而非`2^fanout`枚举 |
| movement与resident-cut elimination | selected physical payload | 删除可证明冗余local movement及compiler-managed spill/store/reload cut |
| whole-tensor share-vs-recompute | structured tensor→selected physical payload | pure/speculatable producer、exact dependent region和effect proof；共享version与按consumer重算都形成actual clone |
| static loop-invariant hoist | structured tensor / selected physical payload | loop-invariant operand、dominance、effect/completion可移动；hoist后延长的lifetime重新placement |
| fixed Cx/NCx encoding absorption | structured→selected physical payload | 仅已有typed verifier/target contract接受Cx/NCx的compute family可直接消费；current限GEMM/batched GEMM，exact physical-map/valid-lane proof后删除显式layout/GS movement；无packing side attr |
| static buffering/resource-aware order | selected physical payload / instruction | 显式buffer slot、token、wait/fence和DAG-legal ready-order alternative；unknown completion保守串行 |
| collective expansion alternative | structured/tile→instruction | current direct/ring/tree参数实际展开为p2p、staging、local work和completion IR |
| integer-domain exact/modular-proof-gated algebraic rewrite | structured tensor | current只开放可从integer IR及overflow/wrap语义证明exact/modular的reassociation、显式rank-local reduction tree和distribution/factorization；无proof保持barrier，全部floating rank-local algebraic rewrite与`contract`-based FMA归Q32.N |

每一row至少有通用positive/negative source、一次真实production candidate mutation和完整exact-gate消费，并按§1.2继续进入
共同frontier、selection和默认production commit。current numeric子项分别验收，不能用一个reassociation正例代表tree或
distribution/factorization。
某row的target参数尚未由typed target纵向支持时先由Q32.V补齐，不得用硬件flag、op名字或CModel能力绕过IR合同。

### 4.2 层间 conversion

从 structured tensor program 到 selected `wafer.tile`，以及从 `wafer.tile` 到 `wafer.instr`，使用明确的
DialectConversion边界：

- `ConversionTarget`声明legal/illegal dialect、op和动态 legality；
- conversion patterns只消费已经显式选择的 typed op/attrs；
- 需要改变type时使用 `TypeConverter`和受验证的materialization；
- conversion完成后禁止前一层非法 residue；
- target op verifier重新证明geometry、encoding、descriptor、effect和completion合同。

candidate generation不得用 side table告诉conversion“本来选了什么”。conversion无法从当前 IR读取的选择必须先成为typed
op/type/attr或SSA关系。

### 4.3 Numeric 和 effect barrier

Q32.N让现有rank-local reassociation、reduction切分及algebraic distribution/factorization直接覆盖支持的
integer和floating op family。float candidate不要求额外fast-math标注；integer仍由current IR的
overflow/wrap语义证明modular合法性，带no-wrap promise的改写保持barrier。Tree和Ring collective同样接受
支持的floating element type，但rank group、topology、chunk、completion和target encoding检查不变。
unknown effect、不可解释DPS tie、control-flow join、collective wait和observable store仍形成rewrite boundary。

## 5. Bounded MLIR-Native Joint Candidate Generation

Q32不实现独立约束语言或一次性巨型solver，但必须有界组合全部当前支持的选择轴。实现建立在Q29 transaction上：worklist里的
每个state都拥有actual isolated MLIR clone，尚未展开的选择从该clone fresh发现，已选事实只存在于clone IR。

稳定流程如下：

1. **建立reserved baseline。** 用当前production规则形成每rank unplaced generation clone，并在独立evaluation clones上经过
   与optimized candidate相同的rank-local gates。all-baseline complete tuple必须第一个组合，并用独立于optimization limits的
   allowance完成whole-variant DDR、post-memory transport、ABI/package及final cost gate；它失败时报告真实pipeline failure，
   不能靠optimized candidate或预算耗尽掩盖。
2. **发现下一处decision site。** 按stable IR preorder、result/edge ordinal和typed interface顺序，从current clone发现
   implementation、tile/relation、encoding/view及fixed Cx/NCx absorption、storage/route/residency、share-vs-recompute、loop-invariant
   hoist、numeric variant、buffering/ready-order及collective expansion机会；不建立detached semantic graph。
3. **局部收紧可行域。** source/interface、IndexRelation、encoding/transfer、alias/effect/liveness和target facts只返回当前
   site的typed alternatives或proof；sound bound只做安全剪枝，不能代替exact materialization。
4. **clone并立即改写。** 每个参数点在parent clone副本上用PatternRewriter/DialectConversion实际改IR；not-applicable不入队，
   applied后销毁所有旧analysis并从新clone重新发现后续选择。不能只改C++decision record。
5. **组合mechanisms。** tiling/fusion、relation propagation、implementation absorption、encoding/view及fixed Cx/NCx absorption、
   route、reuse、movement/resident-cut、share-vs-recompute、loop-invariant hoist、current numeric variants、buffering/ready-order和communication
   按稳定cut顺序组合；每个producer、每site和每rank都有generation hard cap，禁止展开全Cartesian product或`2^fanout`。
6. **rank-local exact gate。** 从unplaced generation parent另建evaluation clone；便宜source/selected verifier和resource
   lower bound可早拒绝，随后完成tile/instruction conversion、whole-rank SPM、descriptor/geometry以及rank-local
   effect/completion gate。whole-variant DDR、post-memory communication binding、all-rank resource、ABI和package eligibility
   只能在complete rank tuple形成后运行。任何失败只销毁对应evaluation clone或variant，不把已写offset带回parent。
7. **resource-aware expansion。** evaluation产生的capacity、descriptor、temporary和event pressure只作为invocation-local
   selection hint，最多用于优先展开当前IR已有的typed tile/residency/buffering/order alternatives。邻居必须从对应unplaced
   generation parent重新clone并回到步骤4实际改写；不得改写或复用已带offset/binding的evaluation clone。09/12/Q34 allocator
   只回答placement，不返回repair或替planner改IR。
8. **维护rank frontier。** 从rank-local finalized current IR读取已知exact facts和safe bounds，只做不会误删潜在
   whole-variant winner的剪枝，并按semantic generation ordinal与physical artifact kind保留有界rank frontier。budget耗尽不淘汰reserved baseline。
9. **all-rank join与最终选择。** existing coordinator先用reserved allowance评估all-baseline tuple；baseline variant ready后
   才按optimization hard cap组合其它complete rank tuples；每个tuple的所有rank必须同时匹配semantic generation与
   spill/spill-ready/resident/resident-ready artifact kind，communication compatibility仅作可重算预筛。每个variant在独立
   clone上依次运行whole-variant DDR placement、post-memory Direct DTE matching/binding、all-rank
   message/range/completion/resource、accepted-rank/resource projection并读取validated high-water与final metrics。reserved
   baseline随后完整运行target ABI/LLVM gate；optimized tuple先用与正式Pareto插入完全相同的Unknown/equivalent-dataflow/
   static-order/frontier-cap规则，相对已经完整target-gated的optimized frontier判断是否会被保留，只有会保留者才运行target
   ABI/LLVM。target失败不淘汰既有frontier并继续后续tuple；只有完整gate通过的variant能进入frontier、static-policy winner和
   最终一次性提交的bundle。

条件性producer产生passing clone后仍必须进入相同rank frontier和whole-variant selection；producer内部不得自行把它标成accepted，
也不得因为局部cost看似更好而绕过baseline、exact Pareto或typed static policy。

Q32.B按上述边界收口首条production-shaped seam：每个scope policy完成task commit和canonicalization后，先删除task-local
evaluation留下的SPM/DDR placement，得到不带owner-produced offset/binding的generation parent；spill与full-buffer-resident分别从
该parent独立clone并重跑rank-local verifier、SPM placement、descriptor/completion和fresh cost。只有conservative policy的spill
artifact标记为invocation-local reserved baseline，且每rank必须恰有一个；该标记只在Scheduling→Compiler私有move-only结构间传递，
不进入IR、序列化artifact或public driver mode。resident与spill都作为actual placed IR进入rank frontier，不再由producer局部二选一。

Compiler finalization在function-boundary bufferization后再次从每个candidate的current IR重跑SPM并fresh recost，但不提前写DDR；
all-rank coordinator先在独立allowance内clone并评估唯一all-baseline tuple，然后才消费optimized visit budget。每次tuple evaluation在
disposable clones上运行whole-variant DDR placement、Direct DTE、all-rank resource/message/completion并形成pre-target
variant。baseline必须立即通过完整target ABI/LLVM gate，否则是pipeline failure；optimized pre-target variant只有按正式插入规则会
留在当前fully-gated optimized Pareto frontier时才运行同一target gate，late failure只丢弃该tuple且不能先淘汰已有survivor。
characterization先按最终Instr IR phase过滤，再在该域内执行相同retain→target-gate→insert顺序。winner比较先用全部Known的
whole-card exact dimensions做strict Pareto，tradeoff再用既有typed target static cost；未完整target-gated的对象不能成为frontier
member或winner，缺失、相等或不能证明更优时保留baseline。validated SPM high-water参与capacity gate和后续
resource-aware expansion，但当前target没有“地址高水位越低性能越好”的typed policy，因此不把它伪装成performance Pareto轴。
Q32.B已开放这条compiler-private seam与resident bring-up alternative，并由默认`wafer-compile`调用现有Scheduling/
Compiler owner完成纵向取证；它不新增public mode，也不把后续Q32.M producers或Q32.S完整joint search切换为默认
production owner。实现与fresh gate归档在`tasks/archive/physical-dataflow-test-seam-vertical.md`。

施工最先打通dependent tiling + resident handoff，随后补relation/view、fanout reuse、movement elimination、implementation、
encoding/route、buffering/order和current communication alternatives。前两条rewrite只是bring-up checkpoint；只有§1.1和§4.1
功能矩阵全部进入上述流程，Q32.S才可收口。

bounded vector、Pareto frontier或beam只是容器策略：固定小vector若能覆盖全部当前decision producers可以成为终态；若实际
candidate增长要求frontier/beam，则在相同actual-clone语义上增加。是否需要beam不影响必须支持的优化轴。

## 6. Tile、Resident Dataflow 和 Movement

tile domain由op interface、static shape、target geometry和现有policy提供的少量候选构成。第一版不枚举全部因子，也不把每维
tile笛卡尔积交给通用solver。consumer tile通过exact IndexRelation求producer dependent region；无法精确反推时保留原边界。

resident dataflow是selected IR的数据流结果，不是预先枚举的fusion partition：

- producer result由consumer通过同一SPM root/view和SSA use-def直接消费时，形成resident edge；
- required movement、observable store、effect/completion或lifetime conflict切断resident edge；
- fanout按exact relation、encoding、effect和lifetime形成stable maximal-compatible subsets；在hard cap内物化少量partial-reuse
  clones，而不是一个不兼容use使全部use回退，也不枚举所有subset；
- collective是completion/transport boundary，不自动成为DDR boundary；
- buffering slot、ready order和overlap必须由实际buffer SSA、loop-carried value、issue token、wait/fence与resource effect表达；
  unknown completion保守串行。dynamic multi-instance ping-pong仍是later，但current static buffering/order必须进入Q32非回退和
  candidate gate。

physical encoding或route选择不能作为事后layout修补。rewrite在clone中建立所需typed encoding/view/movement，08与11从IR重新
证明physical footprint、valid lanes、descriptor cover和instruction geometry。

## 7. Resource、Packing 和 Bounded Work

SPM/DDR legality继续由09/12和Q34的fixed-capacity packing owner负责。Q32保留memory-aware synthesis功能，但不让allocator
成为decision owner：

- 每个完整candidate调用正常placement/validation路径，accepted placement的实际high-water必须进入静态cost；
- generation worklist只保存无owner-produced placement/binding的actual clones；rank-local SPM和whole-variant DDR分别在
  独立evaluation clones中运行，任何evaluated clone都不重新进入rewrite worklist；
- lower bound只可做明确安全的early rejection，不能替代owner placement；
- packing resource exhaustion按owner既有合同处理，不被改写成semantic unsupported；
- capacity、lifetime conflict、temporary、descriptor和event pressure可由06从current IR解释，并生成少量tile/residency/
  buffering/order邻居；allocator只评估这些已物化邻居，不返回repair recipe；
- 对selection-sensitive shortlist，Q32.S可以在独立optimization budget内重复调用同一owner-private fixed-capacity primitive，
  收紧“已知可行high-water / 已知不可行capacity”区间；结果只活在本次candidate evaluation，不定义`PackingEnvelope`、
  versioned proof、跨candidate cache或artifact。probe placement只有在fresh evaluation clone中原子apply，并重新运行全部
  offset-dependent descriptor/range、post-memory transport、ABI/package gate后，才能作为actual high-water/final cost；否则
  只能作safe bound或剪枝。预算耗尽保留已验证placement和quality gap，不能改写成capacity failure；
- allocator不改变implementation、tile、encoding、residency、route、buffering或执行顺序。

第一版总控只保留少量limits：

```text
PlannerLimits {
  max_decision_sites_per_scope
  max_rewrite_alternatives_per_root
  max_materialized_candidates_per_rank
  max_rank_frontier
  max_whole_variant_attempts
  max_resource_neighbors_per_candidate
  max_optional_packing_queries
}
```

IndexRelation、packing等子系统的内部复杂度预算继续由各owner管理。每rank baseline不可淘汰，all-baseline tuple先用独立
allowance完成全部whole-variant gates；只有baseline variant ready后才消耗optimization limits。并行度只是执行hint；candidate
insertion order、comparison和winner不依赖线程完成顺序。正常运行不以wall deadline决定winner；driver cancellation中止整个
transaction且不提交partial artifact。

## 8. Cost 和 Selection

legality先于cost。candidate只有完成lowering、placement和所有当前scope exact gates后才可比较。复用并扩展现有
`InstructionProgramCost` typed C++结构，从final IR至少收集：validated SPM/DDR high-water、DDR read/write、SPM movement、
transport bytes/messages、compute logical work、descriptor/command/issue、temporary、event/fence/wait、padding work和immutable
payload。Known zero与Unknown分开；overflow或无法扫描为Unknown，不能伪造为0。

communication的whole-variant metric还必须消费current `wafer.target.topology`、`wafer.execution.mesh`和final
`dte_send.peer`。payload injected bytes保持独立维度；另以graph shortest-path distance计算
`sum(payload bytes * minimum hops)`。该值是规则topology上的minimum link-byte demand，不是route、contention或time；
没有typed route policy时directional/per-link load保持Unknown。rank-local cost不能因缺少canonical source rank而猜测
physical endpoint，这一维只能在完整rank domain的late analysis中重算。

这些字段不需要17维registry、canonical wire vector或完整IR signature。字段方向、聚合和static preference由closed typed
target profile/C++ policy定义；新增维度必须有final-IR collector和测试，而不是只增加诊断名。

具体变换的收益和代价只能从其final IR反映：recompute必须增加`compute logical work`并同时反映减少的movement/live range；
loop hoist必须反映动态执行multiplicity变化以及延长后的lifetime/high-water；fixed Cx/NCx absorption必须反映消失的layout/GS
movement以及保留的padding、descriptor/issue变化；ready-order必须反映validated high-water、movement和event/fence变化。
numeric variant先过独立numeric gate，
不能因op count减少就推断合法或更快。若现有typed cost无法表达会影响选择的事实，先补final-IR collector和policy，再允许该producer
进入Q32.S；不能用mechanism-applied计数代替选择依据。

板端校准前的production规则：

1. baseline非法时按真实编译错误处理，不能由优化候选掩盖pipeline invariant；
2. 先在observable semantics、transport domain和exact-gate scope一致的候选间做componentwise Pareto dominance；
3. 对Pareto-incomparable survivors，只有当前target profile显式提供static priority/lexicographic policy且所需维度全部Known时
   才排序；该policy只表达编译器静态取舍，不叫estimated hardware time；
4. policy缺失、关键维度Unknown或checked aggregation overflow时保留baseline；完全相同的metrics优先baseline，再按stable
   generation ordinal确定诊断顺序；
5. all-rank cost从各rank final IR按字段语义checked sum/max或identity去重，不能用代表rank标量代替whole variant；
6. `estimatedTimePs`、discovery order、thread completion order和模型名不得决定winner。

Q9获得validated PMU/timing evidence后，可以新增calibrated ranking policy，但只能重排已经通过同一exact legality的候选，不能
改变semantic、numeric、capacity、descriptor或ABI合法性。

当前closed target profile的未校准static policy按resource class显式排序：DDR read/write、NoC payload aggregate/
collective与minimum-hop demand、SPM movement、instruction/event、compute logical work、最后是
data-dependency depth/ready-order inversion。每一class内部只接受
componentwise strict reduction；同一class出现双向tradeoff、任何所需量Unknown或overflow时保持当前winner，最终无法证明优于
reserved baseline时返回baseline。validated SPM high-water只作capacity/resource事实；它可以参与exact resource Pareto，但不被解释成
hardware time。policy在完整Pareto frontier上逐个比较survivor，不能只找第一个优于baseline的candidate，也不能用candidate
discovery ordinal打破语义选择。

Q32.S已按actual producer增长固定当前上界：source clone 16、recipe 12、optimized rank evaluation 64、optimized rank frontier
256、whole best-first tuple 64、coordinated tuple 64和whole Pareto 16；唯一conservative spill baseline拥有独立allowance。source、
recipe与scope policy先分别入列，再优先加入source×recipe、source×policy及all-applicable联合状态，剩余预算按稳定笛卡尔顺序
materialize。implementation与direct-mapped route等同一task内的可组合参数必须形成同一actual recipe；只生成两个单点不能冒充
联合覆盖。实现与winner证据见`tasks/archive/bounded-joint-physical-dataflow-selection.md`。

### 8.1 Production optimizer 同源成对资格边界

production package单独通过数值板测，只能证明当前winner在该输入和环境中可执行，不能证明某项优化确实进入最终目标程序，
也不能证明它相对保守候选更优。compiler-wide资格因此复用coordinator已经保留并通过完整whole-variant gate的唯一
reserved baseline，与默认production winner形成同源成对输入：

- baseline与winner必须来自同一verified source snapshot、ExecutionConfig、TargetProfileId和host-visible ABI；二者分别从各自
  accepted Instr IR继续经过target translation、device link、manifest publication和readback，不能通过改source、跳pass或编译
  两个不同版本伪造对照；
- compiler-private characterization seam只允许测试入口提交已经被同一late gate接受的all-baseline tuple。它不序列化frontier，
  不新增公开的强制tile/layout/collective/ordinal选项，也不改变正常source-to-package driver始终提交默认winner的行为；
- 资格按最终可观察机制合并implementation/tile/physical route、resident/share/recompute、numeric DAG、ready-order和
  collective algorithm；canonicalization、alias proof、packing reject和verifier negative继续由host exact gate闭合，不按
  pass数量消耗板卡。当前LICM缺少真实公开source producer，明确保留host gate；production winner
  qualification没有AG结构不同的默认winner，但独立hardware characterization允许通过test-only typed selector从
  actual accepted Instr phase选择AG Direct/Ring、RS Direct/Ring和AR Ring/Tree；selector要求每个execution rank只含
  请求的phase family，sidecar保留逐message tuple供跨rank matching和cycle/tree graph重放，不以相同package、
  独立field集合或算法名字冒充A/B；
- 对照首先检查host-visible manifest boundary一致，再从最终linked ELF按case读取实际可证明的静态callsite种类/数量、
  straight-line scheduler顺序、workspace或scheduler-body hash；没有CFG/peer解析的case不得声称动态顺序或完整transport
  graph。candidate统计、pre-lowering IR标签或producer计数不能替代最终结构证据。随后两份package都必须对同一CPU
  expected完成全输出、output canary/status和lifecycle验证。

该资格链属于Q37的pre-board/board evidence，不重复已有raw case，也不修改Q32完成结论。已有保守fallback只关闭对应
legality/correctness风险，不表示hardware behavior或cost surface完备；新的区分case由统一硬件校准台账管理。
其成对样本在没有可信device measurement basis时只保留为原始observation；host进程wall time、单次样本、untimed model或仅有
正确性差异都不能改写本节static policy。只有Q9在PMU counter unit、clear/wrap、workload correlation、重复稳定性和held-out
均闭合后，才能把与同一候选对齐的device观测发布为calibrated ranking profile；该profile仍只能重排已通过全部exact gate的
候选。

Q9 profiler foundation只消费本节已经保留的同源reserved baseline和默认winner，不增加新的candidate selector或用户输入。
`wafer-compile --profile`仍提交逐字节不变的普通winner package，并为两份已通过同一late gate的variant形成未插桩execution
binding及内部summary/count/trace逻辑capture binding；若两份variant artifact alias，则精确复用同三个物理capture package。
采集、时钟资格、统计和report均是downstream diagnostic artifact，
不进入accepted IR，也不在foundation阶段反馈本节Pareto/static policy。当前没有合格clock mapping时保留16条entry-local
timeline而不做跨tile对齐。TSM trace的事件单位是最终CRT实际调用的`TsmExecute`，不能用planner中的ready-order、token、
fence或wait代替实际调用，更不能把submit call返回跨度解释成engine完成时间。

## 9. All-Rank Coordination 和 Atomicity

all-rank coordination保留现有 compiler-level owner，不放入function pass，也不建立跨rank shadow program。每个rank candidate
是同一MLIRContext中的完整module clone；coordinator负责：

- rank并行lowering使用独立MLIRContext时，worker只在本次compiler transaction内返回actual finalized module的文本所有权转移
  载体及既有`(semantic generation ordinal, physical artifact kind, reserved baseline)`元数据；收齐完整rank domain后先仅用
  元数据精确重放既有reserved allowance、bounded Cartesian positions和coordinated correspondence顺序，始终保留原frontier
  slot、顺序、重复key及每rank全部reserved-baseline marker，只把会通过correspondence预检的实际attempt所引用module导入
  bundle-owner context。其余slot只在本次transaction内保留nullable module和原metadata，使attempt index set、budget及winner
  与eager import严格相同。该预筛不承担legality、cost或winner判断，也不进入ExecutableBundle/package；rank1退化为导入全部
  实际attempt candidate，任何baseline或结构错误仍交给相同coordinator contract拒绝；
- 按轻量、IR-derived communication compatibility facts剪掉明显不可能组合；
- 从每个完整variant的typed instruction IR重新收集message、range、completion和resource，并用canonical rank
  order及current topology/mesh重算minimum-hop demand；
- 先运行whole-variant DDR placement，再运行post-memory Direct DTE matching/binding、whole-variant resource和
  accepted-rank/resource projection，形成带validated placement/high-water和final metrics的invocation-local pre-target variant；
- all-baseline tuple拥有独立reserved evaluation allowance并先完成target ABI/LLVM gate；
- optimized tuple只相对已经target-gated的optimized frontier精确模拟正式Pareto insertion；不会被保留者跳过target gate，会被保留者
  通过target ABI/LLVM后才实际插入。characterization在该判断前先按current Instr IR匹配自己的phase域；
- 任一target late failure丢弃整个当前variant，不改变既有frontier并继续稳定attempt sequence；未fully gated对象不能参与winner；
- 只有winning variant全部rank通过后才提交offset、binding、source replacement和ExecutableBundle。

不得把 `PlacedRankCompatibilityClaims`、transport signature或canonical candidate bytes作为正确性输入。若为了性能缓存
compatibility facts，它们必须能从current IR重算、只活在coordinator invocation内，并由最终exact gate重新证明。

## 10. Target Capability 和 Downstream Admission

编译阶段只接收明确解析的 `TargetProfileId`及其immutable target capabilities，例如engine、dtype、tile geometry、encoding、
DMA/GS、SPM和event限制。它们可以由closed target registry或小型typed target model提供，不携带
`ModelProfileId`、`BoardEnvironmentId`或qualification floor。

`statically representable`、`compiler emittable`、`model executable`和`board supported`是不同边界：

- Q32要求所选op对当前compiler target profile可表示、可lower，并通过repo-owned TargetCall/formal/SystemC纵向；
- external functional model admission由17的model profile拥有；
- board admission由configured board environment拥有；
- 板端缺失不阻塞compiler结构优化，也不能被compiler结果冒充为board证据。

Q32.V已完成独立typed target纵向：mapped DMA/WDMA两端offset与descriptor、physical-footprint fill/invalid-lane初始化、
oriented GEMM的source/ODS/interface/verifier、Instr/TargetCall、v2 ABI revision和SystemC/formal consumer均已闭合。
Q32.M起由§5同一通用candidate owner消费，不新增feature-specific planner。

Count仍由独立Q3.6拥有，因为当前缺少稳定source predicate和model证据。若Q32.V扩展command的package/runtime consumer需要逐row
preflight，`RequiredCapabilitySet`和必要package revision只能从winner实际Instr/TargetCall rows在post-selection阶段派生；它们
不是planner输入，也不能携带implementation、encoding、route、residency或candidate history。

## 11. Failure、Diagnostics 和 Instrumentation

公共控制流优先使用 MLIR `LogicalResult`、`FailureOr<T>`、diagnostic和conversion legality。内部需要区分时可用普通typed enum，
至少区分not-applicable、unsupported representation、invalid IR和resource exhaustion，但不形成跨stage schema。

第一版诊断统计可以是invocation-local C++ counters或MLIR pass statistics：

```text
generated_candidates
materialized_candidates
rejected_by_conversion
rejected_by_spm
rejected_by_ddr
rejected_by_transport
accepted_candidates
used_baseline
```

统计不进入IR、cache key、bundle、package或完成语义，不固定phase矩阵和provider status rows。测试可以读取统计证明rewrite实际发生、
hard cap生效和baseline被保留，但统计格式本身不是ABI。

## 12. 通用案例和 Scale Evidence

### 12.1 Dependent tile + resident handoff

示例形态是一个structured producer后接pure pointwise consumer。协议事实是：

- producer和consumer由Tiling/DPS/indexing/effect interface解释；
- exact IndexRelation把consumer tile反推到producer region；
- rewrite通过PatternRewriter在clone中构造dependent tiles和SSA handoff；
- physical realization决定能否共享SPM root/view或需要movement；
- final instruction IR重新证明coverage、lifetime、completion和resource；
- 若删除整tensor store/reload后static cost严格占优，candidate才可胜baseline。

具体op可能是activation、multiply、convert或其它elementwise structured op；具体shape、模型名和tile值只是测试参数，不进入
interface或rewrite matcher。

### 12.2 Shared input / projected view reuse

示例形态是pure producer或immutable input有两个以上consumers。只有SSA root、IndexRelation、alias/effect和lifetime证明允许时，
rewrite才复用physical version或吸收projected view；某一use需要不同encoding、observable order或更长lifetime时保留独立movement。

该机制服务chain、diamond、fanout/fanin和multi-root图，不按QKV、gate/up等角色识别。

### 12.3 Llama-2 7B单block

标准Llama-2 7B单block、batch 1、sequence 16、TP16、FP16 storage只作规模与回归证据。它验证通用rewrite在大量contraction、
pointwise、reduction、residual和collective上不会依赖名称，且rank-count=16的resource、transport、package和完整输出数值不回退。

当前Q29/Q28/Q30/Q31数据只是fresh baseline；旧tile、rank-0 high-water或最终resident cut都不是新合同。收益必须从新winner final
IR重新统计，算法接入本身不算性能收益。

## 13. 实施顺序

已完成checkpoint与integrated checklist见`tasks/archive/physical-dataflow-synthesis.md`；最终fresh证据见
`tasks/archive/physical-dataflow-synthesis-completion-audit.md`。施工顺序为：

1. fresh重放当前baseline，确认现有candidate clone、finalization、all-rank coordinator和exact gates；
2. Q32.I增加source OpInterface/external models、至少一个真实implementation alternative及第一批MLIR-backed IndexRelation；
3. Q32.R补齐relation/view/physical-map/route proof基础并实现dependent-tiling/resident handoff；
4. Q32.B把baseline和第一批optimized clones接入现有rank frontier/all-rank exact gate，完成production-shaped test seam；
5. Q32.V独立闭合mapped DMA、physical fill和oriented GEMM typed纵向，只有完成的row才进入candidate domain；
6. Q32.M完成§4.1 mandatory mechanism closure，并把current implementation、encoding/view/materialization、route、
   share-vs-recompute、loop-invariant hoist、fixed Cx/NCx absorption、各current numeric variant、buffering/ready-order及direct/ring/tree
   communication alternatives全部接入同一actual-clone路径；
7. Q32.S有界组合全部producer，逐项证明它们进入共同frontier并有production winner，加入resource-aware neighbors、validated
   high-water/cost、Pareto/static-policy选择；根据完整producer的actual growth决定fixed-capacity candidate vector/frontier/beam，
   而不是只根据前两条
   rewrite决定；
8. Q32.G已让默认`wafer-compile`重放逐功能winner/commit证据，并删除旧scope-prefix、layout/materialization、maximal-resident、
   communication selector和scalar-time旁路；all-rank candidate还必须同时匹配semantic generation与physical artifact kind；
9. 重放rank-count=1/16、通用property/topology、Q20/Q21、7B PyTorch/SystemC及全部exact gates，完成原子audit。

上述顺序不插入provider registry、query codec、版本化诊断schema、shadow frontier或Transform extension。target capability纵向、
memory quality和communication choice仍然实现，只是分别由typed target IR、existing packing owner和actual collective rewrite承担。

## 14. 完成定义

Q32完成必须同时满足：

1. 所有production source compute root通过OpInterface/external model进入有界implementation choice；至少一个真实source的非baseline
   implementation被materialize、选择并进入winner；
2. matcher只依赖structured semantics、SSA、type/indexing/effect、native numeric permission/semantics、局部proof和target facts，无模型名、buffer名、固定shape或
   operand-position语义恢复；
3. `IndexRelation`优先复用MLIR Affine/Presburger/ValueBounds，覆盖§3.2 relation closure并由tiling、view、propagation、transfer和
   reuse真实消费；无私有wire relation IR；
4. §4.1 mandatory mechanisms全部被production candidate generator真实调用，至少各有一个通用source发生mutation并通过完整gate；
   share-vs-recompute、static loop-invariant hoist、fixed Cx/NCx absorption以及reassociation、显式rank-local reduction tree、algebraic
   distribution/factorization的integer-domain exact/modular variants各自有predicate-positive mutation/acceptance和
   predicate-negative baseline证据；
5. current implementation、encoding/view/materialization、route、residency、buffering/ready-order和direct/ring/tree communication
   alternatives形成不同actual clones；Q32.V完成的mapped/physical-fill/oriented rows也由相同owner消费；每个会产生选择分支的功能
   至少有一个非workload特化source进入rank/whole-variant selection并成为committed winner，required closure的mutation保留在winner；
6. 上述证据全部来自默认`wafer-compile` source-to-bundle路径；只在Q32.B seam、`wafer-opt`手工pass、隐藏flag或测试callback中
   被调用不算production接入，bundle/readback必须从winner final IR重证相应变化；
7. applied rewrite后所有派生analysis fresh重算，selected clone是唯一事实源；candidate生成、materialization、rank frontier、
   all-rank attempt和subsystem work均有hard cap，resource exhaustion保留baseline；
8. resource-aware generation和selection实际读取Q34 validated placement/high-water、DDR/SPM movement、transport、compute、
   descriptor/instruction/event等final-IR facts；allocator不改变candidate；
9. optimized candidate和baseline运行同一DialectConversion、SPM/DDR/event/transport/instruction及最终ABI/package gate；baseline
   先完整通过，optimized只有在pre-target Pareto retention证明会保留后才延迟执行ABI/LLVM，失败不修改fully-gated frontier，
   all-rank失败不提交partial mutation；
10. rank-count=1/16、chain/diamond/partial-fanout/fanin/reduction/broadcast、current communication topology、Q20/Q21、7B scale和
   完整PyTorch/SystemC数值gate通过；
11. production只保留一个decision owner，旧layout planner、scope-prefix/maximal-resident、communication selector和未校准scalar
    winner路径删除；candidate、analysis、diagnostic和统计不进入bundle/package；
12. model/board admission不参与candidate选择，不把compiler静态收益宣传成board/timing收益，Q9 calibration边界保持关闭。

只完成interface、只生成proposal、只打通两条rewrite、只通过局部FileCheck、passing candidate从未进入frontier/winner、只在
单个7B case减少一个op、只跑单rank，或新旧decision owner并存，均不构成完成。

## 15. 参考机制

- MLIR Interfaces和external models：<https://mlir.llvm.org/docs/Interfaces/>
- MLIR Pattern Rewriter：<https://mlir.llvm.org/docs/PatternRewriter/>
- MLIR Dialect Conversion：<https://mlir.llvm.org/docs/DialectConversion/>
- MLIR Pass/analysis invalidation和instrumentation：<https://mlir.llvm.org/docs/PassManagement/>
- MLIR Linalg structured transformations：<https://mlir.llvm.org/docs/Dialects/Linalg/>
- MLIR Affine与Presburger：<https://mlir.llvm.org/docs/Dialects/Affine/>

这些机制决定实现形态；外部优化论文只能提供算法启发，不能引入与Wafer typed IR并列的长期协议。
