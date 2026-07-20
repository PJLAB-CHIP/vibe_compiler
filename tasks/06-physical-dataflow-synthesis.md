# Wafer Physical-Dataflow Synthesis 与 Candidate Selection

状态：2026-07-20按 MLIR interface、analysis、rewrite、conversion 和 pass 边界重写。实现状态以
`tasks/progress.md`为准。

本文是 rank-local physical-dataflow candidate generation、candidate acceptance 和 all-rank atomic commit 的唯一设计
owner。它不定义第二套 semantic IR、provider registry、shadow schedule、序列化 frontier 或 qualification 协议。07、08、
10-13分别拥有 selected tile/dataflow IR、physical realization、compute/movement、instruction、memory 和 communication
合同；它们向本文提供 IR、interface、analysis、rewrite、conversion 和 exact verifier，不再发布另一份 planning protocol。

当前 production 是 Q29 已完成的 bounded tile-dataflow scheduler。Q32 从这条链路演进：先增加一条真实、可验证的
dependent-tiling/resident rewrite，再扩展有限 candidate generation，最后原子替换旧 decision owner。没有实际 rewrite、
下游消费和完整 gate 的接口或框架不进入 production。

## 1. 核心结论

Wafer 需要联合评估 implementation、tile、physical encoding、storage realization、residency 和 communication 对最终
program 的影响，但不需要一个平行于 MLIR 的巨型联合求解协议。主线固定为：

```text
verified structured MLIR
  -> op/type/attr interfaces + recomputable analyses
  -> a small number of policy-free rewrites on isolated clones
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
4. **机制先于搜索。** 每种 tiling、fusion、view folding、resident handoff 或 movement elimination 必须先作为独立
   rewrite 在真实 source 上通过 correctness 和完整 downstream gate；搜索只决定调用哪些已验证 rewrite 及其少量参数。
5. **联合评估不等于一次性联合求解。** implementation、encoding、route 和 residency 可以分阶段产生有限候选，但所有候选
   最终必须在同一份完整 IR 上重算 exact gates 和 cost。
6. **合法 baseline 永远保留。** 优化预算只限制新增候选，不得把合法输入变成编译失败；baseline 自身失败仍按真实
   pipeline failure 报告。
7. **不伪造硬件性能。** 板端校准前只比较 final IR 可证明的静态量；存在真实 tradeoff 时保留 baseline。Q9 才拥有
   hardware-calibrated ranking。
8. **目标能力与运行资格分离。** compilation 只接收明确的 `TargetProfileId`/target capabilities。model 或 board
   qualification 是 downstream admission/evidence，不进入每个候选查询。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Shardy/XLA SPMD 之后按 logical rank 静态 specialize、通过当前 verifier 的
  Linalg/Tensor/SCF/Arith/Math structured tensor program；logical collective、numeric policy、
  ExecutionConfig 和 TargetProfileId 均已显式。
- Current stage responsibility:
  直接通过 structured op interfaces、SSA use-def、type/shape、indexing map、effect 和 numeric policy识别
  有界 rewrite scope；从当前 IR 重算 IndexRelation、alias/root 和 effect facts；在隔离 complete-rank clone
  上应用少量 policy-free rewrites，lower 到 typed tile/instruction IR，并复用现有 finalization、SPM、DDR、
  event、transport、instruction、geometry、target ABI 和 package-eligibility pure gates。对 accepted clones
  使用未校准静态比较，交给现有 all-rank coordinator 原子选择。
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
  诊断统计协议或 Transform control plane；不做 whole-model equality saturation、全维 Cartesian product、
  minimum-high-water packing optimization、board/model qualification、mapped DMA、oriented GEMM、package schema upgrade或
  RequiredCapabilitySet扩展；不承诺动态shape、多实例动态loop、cycle timing或board性能。
- Completion gate:
  至少两类非workload特化的真实 rewrite 经 source-to-bundle 主线实际发生并被下游直接消费；每次rewrite后
  derived analyses均fresh重建；baseline和optimized clone经过相同exact gates；rank-count=1/16、通用拓扑、
  7B scale和完整PyTorch/SystemC数值验证通过；production只保留一个decision owner，旧scope/layout/
  maximal-resident旁路和未校准scalar-time winner规则删除。
```

## 3. 稳定职责和对象

### 3.1 Structured source interfaces

source op 的语义首先来自 upstream MLIR interface：

- `linalg::LinalgOp`/structured indexing maps；
- `DestinationStyleOpInterface`；
- `TilingInterface`和subset/view语义；
- `MemoryEffectOpInterface`及Wafer resource effects；
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
`WaferComputeOpInterface`、ODS verifier和conversion legality验证。

当前 target profile 是 closed registry。没有第二个 target 实现时，普通 typed helper也足够；只有出现多个 dialect/target
实现且确有共同 consumer 时，才评估 `DialectInterface`。不得为了未来插件化先建立动态 provider registry。

### 3.2 IndexRelationAnalysis

`IndexRelation`描述 consumer logical index 到 producer logical index 的可组合关系，用于 dependent tiling、view legality、
producer fusion、transfer cover和rewrite coverage proof。它是从当前 IR 派生、可失效、可重算的 analysis value，不是 op、attr、
sidecar或跨pass协议。

实现优先复用：

- structured indexing `AffineMap`；
- MLIR Presburger集合/关系；
- `ValueBoundsOpInterface`和相关 bounds utilities；
- `TilingInterface`、subset/view/reassociation语义。

第一版只覆盖真实 rewrite 需要的最小闭包：identity、projected permutation、broadcast、static slice和已能用现有 MLIR
表示/证明的 reshape reassociation。每增加一种表达必须同时增加真实 consumer、composition/property test和 unsupported path。
无法精确表示时返回 unsupported并保留 movement baseline，不构造私有 piecewise wire language。

analysis API必须区分：

- exact relation/proof；
- conservative bound，只可用于安全剪枝或诊断；
- unsupported representation；
- invalid IR；
- subsystem-owned resource exhaustion。

exact rewrite只能消费 exact result。任何 applied rewrite 都结束当前 IR epoch：旧 `Operation*`/`Value` binding、relation、alias、
effect、lifetime和cost全部销毁，从修改后的 clone fresh重建。不同 clone之间只能共享不含 IR引用的 immutable target facts。

### 3.3 Physical encoding and transfer planning

selected physical encoding是 type/attr语义。footprint、alignment、valid domain和 logical-index-to-physical-offset 行为属于
physical encoding attr/type或其 interface；implementation interface只声明它能接受的 encoding constraints。

Cx/NCx具有 tail/padding和可能非 affine 的 physical mapping，不能伪装成通用 `MemRefLayoutAttrInterface` 能解释的 affine
layout。标准 memref view只用于其语义确实成立的 encoding；其余 reorder/materialization必须由显式 Wafer op表示并由08验证。

transfer route依赖 source root/view、destination root/view、两端 encoding、IndexRelation、alias/effect和target DMA/GS限制，
不是单 op 行为。08提供普通 `TransferPlanningAnalysis`/typed helper，返回少量即时 alternatives。选中后直接在 clone 中物化：

- zero-copy view；
- direct load/store或mapped movement；
- explicit temporary、layout materialization、DMA/GS和event/completion。

物化后的 IR 是唯一事实源；route proposal随后销毁。不存在 route query schema、route id sidecar或resolved schedule record。

### 3.4 Communication planning

logical collective语义只由当前 collective op/interface表达。topology/target helper可以枚举少量 typed algorithm parameters，
例如 direct/ring/tree 及必要 chunk 参数；每个 alternative必须直接在 complete-rank clones 中展开为真实 p2p、local compute、
staging、token和wait IR。

all-rank correctness继续由现有 coordinator从当前 instruction IR收集 message、buffer range、completion和binding并重算。允许用
一个从 typed collective/message IR 派生的轻量 grouping key减少不可能组合，但它不承担 correctness，也不复制完整 message
claims。不存在 CommunicationScheduleSkeleton、ResolvedCommunicationSchedule 或 transport signature协议。

### 3.5 Candidate

candidate的最小长期 C++ 形态是：

```text
Candidate {
  complete_rank_module_clone
  stable_generation_ordinal
  final_instruction_program_cost
  optional diagnostic rejection reason
}
```

生成中可以暂存一个 typed rewrite参数点，但参数一旦 applied，后续只读 clone。candidate不得携带另一份 implementations、
physical versions、schedule DAG、allocation map、message claims或canonical serialized IR。

determinism来自稳定 IR traversal、固定 proposal insertion order和明确的比较规则，不通过重新序列化完整 IR建立 identity。
memoization只有在 profiling证明必要，且 key能由局部 immutable value安全构造时才加入；不得缓存带 IR pointer 的 derived analysis。

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

默认只做保持原 evaluation order 和 numeric contract 的结构变换。浮点重结合、reduction tree改变、NaN/Inf/-0、overflow或
rounding变化必须有显式 numeric policy和独立 correctness evidence；没有证据时形成barrier。unknown effect、不可解释DPS tie、
control-flow join、collective wait和observable store同样形成rewrite boundary。

## 5. 第一版 Candidate Generation

第一版不实现通用约束求解器。它在现有 Q29 transactional skeleton 上加入有限 alternatives：

1. **建立 baseline。** 用当前production规则生成一个完整 rank clone，经过现有lowering/finalization/exact gates；baseline slot
   不占优化candidate预算。
2. **稳定遍历rewrite roots。** 按IR preorder和structured result顺序访问有界root，构造当前epoch的relation/effect facts。
3. **枚举少量typed参数。** 从source op external interface、TilingInterface和target capabilities获得有限参数点；每个root和
   mechanism均有小hard cap。
4. **立即物化clone。** 为每个参数点clone baseline或最近accepted parent，通过PatternRewriter应用一次明确rewrite；applied后
   销毁旧analysis并fresh验证同层IR。
5. **lower并运行exact gates。** 通过现有tile/instruction conversion、SPM/DDR planning、completion、descriptor、transport、
   geometry、ABI和package eligibility。任何gate失败只拒绝该clone。
6. **读取final cost。** 只从finalized IR计算现有 `InstructionProgramCost`；不使用proposal估计替代exact result。
7. **形成小rank frontier。** baseline加有限strictly-better alternatives进入现有 rank candidate frontier。
8. **复用all-rank coordinator。** coordinator按现有hard cap尝试完整组合，在current bound IR上重跑transport/resource/ABI gate，
   只有全部rank成功才原子形成bundle。

首批通用rewrite固定围绕已观察到、但不绑定workload名称的两类数据流：

1. contraction/structured producer 的 dependent tile直接交给其 pointwise或elementwise consumer，并在合法时保留SPM resident
   handoff，删除中间整tensor store/reload；
2. 多consumer共享同一pure producer/input时，在root、relation、effect、lifetime均可证明的范围内复用一个physical version，
   或吸收一个projected-permutation/view而不引入额外movement。

第二类只有第一类完整纵向稳定后才进入主线。它们必须从iterator/indexing/SSA/effect识别，不能使用QKV、gate/up、SiLU、
模型名或buffer名作为协议。

## 6. Tile、Resident Dataflow 和 Movement

tile domain由op interface、static shape、target geometry和现有policy提供的少量候选构成。第一版不枚举全部因子，也不把每维
tile笛卡尔积交给通用solver。consumer tile通过exact IndexRelation求producer dependent region；无法精确反推时保留原边界。

resident dataflow是selected IR的数据流结果，不是预先枚举的fusion partition：

- producer result由consumer通过同一SPM root/view和SSA use-def直接消费时，形成resident edge；
- required movement、observable store、effect/completion或lifetime conflict切断resident edge；
- fanout只有在所有uses和lifetime可证明时共享physical version，否则局部spill；
- collective是completion/transport boundary，不自动成为DDR boundary；
- ping-pong和overlap只有在event、buffer slot和reuse safety全部显式时才允许。

physical encoding或route选择不能作为事后layout修补。rewrite在clone中建立所需typed encoding/view/movement，08与11从IR重新
证明physical footprint、valid lanes、descriptor cover和instruction geometry。

## 7. Resource、Packing 和 Bounded Work

SPM/DDR legality继续由09/12和Q34的fixed-capacity packing owner负责。Q32不增加minimum-high-water binary refinement、packing
proof envelope或shared packing objective：

- 每个完整candidate调用一次正常placement/validation路径；
- accepted placement的实际high-water可以进入诊断或静态cost；
- lower bound只可做明确安全的early rejection，不能替代owner placement；
- packing resource exhaustion按owner既有合同处理，不被改写成semantic unsupported；
- allocator不改变tile、residency、route或执行顺序。

第一版总控只保留少量limits：

```text
PlannerLimits {
  max_rewrite_alternatives_per_root
  max_materialized_candidates_per_rank
  max_rank_frontier
  max_whole_variant_attempts
}
```

IndexRelation、packing等子系统的内部复杂度预算继续由各owner管理。baseline完成后才消耗optimization limits。并行度只是执行
hint；candidate insertion order、comparison和winner不依赖线程完成顺序。正常运行不以wall deadline决定winner；driver
cancellation中止整个transaction且不提交partial artifact。

## 8. Cost 和 Selection

legality先于cost。candidate只有完成lowering、placement和所有当前scope exact gates后才可比较。第一版复用
`InstructionProgramCost`的Known/Unknown维度，不新增17维registry、canonical vector schema或完整IR signature。

板端校准前的production规则：

1. baseline非法时按真实编译错误处理，不能由优化候选掩盖pipeline invariant；
2. optimized candidate只有在所有共同Known、方向明确的静态维度上不差，且至少一维严格更好时，才能淘汰baseline；
3. 任一关键维度Unknown，或DDR/SPM/compute/transport之间存在不可校准tradeoff时，保留baseline；
4. 多个strictly-dominating candidate之间按稳定生成ordinal选择，或保留有限frontier交给all-rank legality；
5. 不再把coarse `estimatedTimePs`称为硬件时间，也不让它决定新路径production winner。

该规则有意保守：Q32首先证明真实movement/residency改写能贯通，而不是在无板端证据时发明偏好。Q9获得validated PMU/timing
evidence后，可以替换ranking policy，但不能改变本任务的语义和resource legality。

## 9. All-Rank Coordination 和 Atomicity

all-rank coordination保留现有 compiler-level owner，不放入function pass，也不建立跨rank shadow program。每个rank candidate
是同一MLIRContext中的完整module clone；coordinator负责：

- 按轻量、IR-derived communication compatibility facts剪掉明显不可能组合；
- 从每个完整variant的typed instruction IR重新收集message、range、completion和resource；
- 运行Direct DTE matching/binding、whole-variant resource和ABI/package eligibility；
- 失败时丢弃整个variant；
- 只有winning variant全部rank通过后才提交offset、binding、source replacement和ExecutableBundle。

不得把 `PlacedRankCompatibilityClaims`、transport signature或canonical candidate bytes作为正确性输入。若为了性能缓存
compatibility facts，它们必须能从current IR重算、只活在coordinator invocation内，并由最终exact gate重新证明。

## 10. Target Capability 和 Downstream Admission

编译阶段只接收明确解析的 `TargetProfileId`及其immutable target capabilities，例如engine、dtype、tile geometry、encoding、
DMA/GS、SPM和event限制。它们可以由closed target registry或小型typed target model提供，不携带
`ModelProfileId`、`BoardEnvironmentId`或qualification floor。

`statically representable`、`compiler emittable`、`model qualified`和`board supported`是不同边界：

- Q32只要求所选op对当前compiler target profile可表示、可lower、可被当前package/model主线消费；
- functional model admission由17的model profile拥有；
- board admission由configured board environment拥有；
- 板端缺失不阻塞compiler结构优化，也不能被compiler结果冒充为board证据。

oriented GEMM、mapped DMA、Count、schema升级和 `RequiredCapabilitySet`若有独立downstream consumer，应作为各自target/ABI/package
任务实现。`RequiredCapabilitySet`只能从winner实际Instr/TargetCall rows在post-selection阶段派生，不反向参与Q32搜索，也不作为
Q32第一版前置。

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

详细checkpoint见`tasks/plans/physical-dataflow-synthesis.md`。顺序固定为：

1. fresh重放当前baseline，确认现有candidate clone、finalization、all-rank coordinator和exact gates；
2. 增加source external interface及第一条rewrite需要的最小IndexRelation子集；
3. 实现dependent-tiling/resident handoff，在真实通用source上删除movement并走完整source-to-bundle纵向；
4. 接入现有rank frontier，形成baseline加少量optimized clones，使用strict-dominance规则；
5. 增加第二条shared-input reuse或projected-view absorption机制；
6. 用实际candidate count和compile wall数据决定是否需要小型frontier/beam；无增长证据则不实现通用solver；
7. production切换后删除旧scope-prefix、layout/materialization、maximal-resident和scalar-time decision旁路；
8. 重放rank-count=1/16、通用property/topology、7B PyTorch/SystemC和全部exact gates，完成原子audit。

明确不在上述顺序中插入provider registry、query codec、版本化诊断schema、Transform extension、minimum-high-water objective或target
ABI capability扩展。

## 14. 完成定义

Q32完成必须同时满足：

1. 至少两类通用rewrite被production candidate generator真实调用，并在accepted candidate中发生；
2. matcher只依赖structured semantics、SSA、type/indexing/effect/numeric policy和target facts，无模型名、buffer名、固定shape或
   operand-position语义恢复；
3. `IndexRelation`优先复用MLIR Affine/Presburger/ValueBounds并只覆盖真实consumer；无私有wire relation IR；
4. applied rewrite后所有派生analysis fresh重算，selected clone是唯一事实源；
5. baseline、hard caps、resource exhaustion和candidate rejection有正负测试；
6. optimized candidate和baseline运行同一DialectConversion、SPM/DDR/event/transport/instruction/ABI gate；
7. rank-count=1/16、chain/diamond/fanout/fanin/reduction/broadcast、7B scale和完整PyTorch/SystemC数值gate通过；
8. production只保留一个decision owner，旧layout planner、scope-prefix/maximal-resident选择器和未校准scalar winner路径删除；
9. candidate、analysis、diagnostic和统计不进入bundle/package；all-rank失败不提交partial mutation；
10. 不把compiler静态收益宣传成board/timing收益，later calibration边界保持关闭。

只完成interface、只生成proposal、只通过局部FileCheck、只在单个7B case减少一个op、只跑单rank，或新旧decision owner并存，均不
构成完成。

## 15. 参考机制

- MLIR Interfaces和external models：<https://mlir.llvm.org/docs/Interfaces/>
- MLIR Pattern Rewriter：<https://mlir.llvm.org/docs/PatternRewriter/>
- MLIR Dialect Conversion：<https://mlir.llvm.org/docs/DialectConversion/>
- MLIR Pass/analysis invalidation和instrumentation：<https://mlir.llvm.org/docs/PassManagement/>
- MLIR Linalg structured transformations：<https://mlir.llvm.org/docs/Dialects/Linalg/>
- MLIR Affine与Presburger：<https://mlir.llvm.org/docs/Dialects/Affine/>

这些机制决定实现形态；外部优化论文只能提供算法启发，不能引入与Wafer typed IR并列的长期协议。
