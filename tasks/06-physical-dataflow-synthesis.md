# Wafer Physical-Dataflow Synthesis 与 Candidate Selection

状态：2026-07-18已收敛终态设计；实现由`tasks/progress.md`中的
`physical-dataflow-synthesis`任务跟踪。本文不定义或恢复`wafer.group`。

本文是 rank-local **physical-dataflow synthesis** 的唯一设计 owner。它联合选择等价计算形式、target
implementation、tile、physical encoding、storage realization、residency、buffering 和合法执行顺序，再把唯一选中方案
在transaction-local clone中物化、lower并通过exact gates，最后只提交final placed instruction/completion program。07、08、10
分别拥有这一stage内部使用的 tile-dataflow IR、物理编码/传输路线和 target implementation family；09、11-13拥有对应
memory/instruction/communication服务。它们不维护第二个 planner，也不另行发布一份accepted中间program。

当前 production 仍是 Q29 已完成的 bounded tile-dataflow scheduler：它使用确定性 source order，并在完整 rank clone
上比较 spill baseline 与 maximal full-buffer-resident alternative。该实现是迁移起点，不是本文终态算法。迁移期间不得
发布第二条 production pipeline；新 planner 必须先证明同一合法 baseline，再原子替换旧决策权，完成时删除 scope-prefix、
二选一 resident 后处理和 layout 独立决策旁路。

## 1. 核心结论

Wafer 的优化对象不是“把多少 graph op 塞进一个 group”，而是：

```text
rank-local structured semantics
  + numeric/effect policy
  + target capability families
  -> bounded physical-dataflow synthesis
  -> one explicit typed tile/dataflow program
  -> exact SPM/DDR/event/transport/instruction gates
  -> all-rank atomic executable bundle
```

关键原则如下：

1. **融合是结果，不是搜索变量。** producer/consumer 通过同一 SPM physical version 相连时即形成 resident
   dataflow；显式 store/reload、movement、effect 或数值 barrier 才切断它。IR 不增加 fused-kernel 名称或 group ID。
2. **layout 不是事后修补。** implementation、tile、encoding 和 transfer route 在同一个约束问题中选择；不先固定
   compute op，再插入一串 layout transform。
3. **目标相关、workload 无关。** planner 可以认识 Wafer engine、descriptor、SPM、Cx/NCx 和 event 能力，但不能认识
   Llama、Q/K/V、gate/up、参数名、文件名或固定 shape。
4. **搜索必须有合法 baseline 和硬上界。** 预算耗尽只降低优化质量，不能把合法输入误报为语义非法；所有候选均受相同
   verifier 和 exact resource gate。
5. **搜索状态不进入长期 IR。** relation、domain、Pareto frontier 和 cost trace 都是当前 IR 上可失效、可重算的 analysis；
   跨 stage 只保留选中方案的 typed compute、view、movement、buffer、event 和 offset。
6. **Transform Dialect 只作可选控制面。** production named pipeline 与可选 transform extension 调用同一 C++ planner/
   materializer；Transform IR 不承载 solver 内部状态，也不成为执行 artifact。
7. **机制先于策略。** relation normalization、tiling、producer fusion、view/subset folding、implementation absorption、
   physical-version reuse和movement elimination先作为带precondition/proof的typed atomic mechanisms独立验证；搜索只组合、
   排序和接受这些机制，不能在policy分支中临时发明rewrite。
8. **packing既是exact gate，也是候选质量oracle。** materialization前只用可证明的SPM lower bound做安全剪枝；完整
   instruction candidate形成后，由shared `MemoryPlanning` fixed-capacity packing primitive先证明硬件容量合法性，再由本stage对
   selection-sensitive shortlist组合有界capacity queries，得到最小high-water的可证明上下界。优化预算耗尽只留下
   bounded quality gap，不得改写成capacity failure，也不得让allocator自行改变tile、residency或执行顺序。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Shardy/XLA SPMD 之后、按 logical rank 静态 specialize并通过当前verifier的
  Linalg/Tensor/SCF/Arith/Math structured tensor program；logical collective
  已由 typed interface 表达。输入同时携带 validated
  ExecutionConfig、TargetProfileId、numeric policy 和可查询的 target capability provider。
- Current stage responsibility:
  从 structured iterator/indexing map、SSA use-def、shape/dtype、view relation、effect、control flow 和
  numeric policy 构造有限 semantic islands；通过已独立验证的policy-free typed mechanisms和target implementation family惰性地产生
  decision domains，联合选择 implementation、tile、physical encoding、storage realization、residency、
  buffering 和 DAG-legal issue order；在 transformation-local complete rank clones 上物化候选并运行 exact
  instruction、SPM、DDR、completion、transport、geometry和14 registry/versioned-signature artifact-eligibility
  preflight（纯函数、不生成module/artifact）；对selection-sensitive survivors组合shared static-packing fixed-capacity
  queries，得到validated placement与high-water上下界；对surviving candidates做有界
  Pareto selection，并原子提交一个 all-rank variant。
- Output artifact / IR:
  profile-bearing atomic `ExecutableBundle`：每个rank恰有一个覆盖完整静态traversal、final placed且不含
  tensor/Linalg/search/group残留的instruction/memory/completion program；tasks/14 registry从这些winner instruction rows
  投影rank-local capability keys并形成all-rank canonical `RequiredCapabilitySet`。tile/dataflow、unplaced instruction、packing
  problem和candidate clone都只是本stage transaction-local中间artifact，不与bundle并列发布；required set也不是search state。
- Downstream consumer:
  14直接消费bundle的final instruction rows做target conversion/module publication，15形成package，17的SystemC/CModel消费
  同一target program；16横切验证上述边界。07-13是本stage内部materialization/lowering/exact-gate services，不是另一组
  accepted-artifact下游。任何consumer都不读取rejected candidate、search trace、Transform IR、历史group边界或shadow plan。
- User-level driver / named pipeline:
  production 只经 wafer-compile 的 source-to-bundle named pipeline。可选 Wafer Transform extension 只能调用
  同一 planner/materializer library，用于受控实验、复现和诊断；不得形成不同语义或第二种 accepted artifact。
- Explicit non-goals:
  本 stage 不定义 framework 数学语义、SPMD partition、runtime manifest、raw packet/CRT ABI 或 cycle-accurate
  timing；不做 whole-model/unbounded equality saturation、任意 dynamic graph scheduling、名字驱动优化、模型专用
  pass、未获 numeric policy 许可的重结合，也不把 board 性能估计当 legality。首轮不承诺 dynamic-shape、动态
  multi-instance loop、未经 event 证明的 ping-pong 或全局 persistent weight cache。
- Completion gate:
  production planner 只消费 structured semantics/effect/numeric policy 和 target provider；parameterized family
  惰性实例化、constraint propagation、state canonicalization、hard budgets、baseline fallback 与 candidate-growth
  telemetry 均实际生效；每个候选rewrite来自已注册atomic mechanism并在改写后fresh重算relation/effect/alias/resource；
  hardware-capacity legality与packing objective budget正交，interval-aware dominance和deterministic shared query fuel实际生效；
  selected proposal 只以 typed payload IR 跨阶段，旧 scope-prefix/layout-planner/
  maximal-resident 决策旁路清零。property、拓扑、多 dtype、多 workload、rank-count=1/16、7B scale、完整
  PyTorch/SystemC numerical、exact SPM/DDR/event/transport/ABI 和 atomic commit gates 全部通过。
```

## 3. 设计边界和稳定对象

### 3.1 Semantic island

`SemanticIsland` 是 planner 内部的有界 analysis scope，不是 IR op。它是 effect/control/numeric barrier 之间一段能从当前
IR 完整解释的 pure structured dataflow：

- roots、DPS inputs/inits/results 和 nested scalar regions 都可追溯；
- iterator domain、indexing relation、dtype、shape、broadcast/reduction 和 side effect 均已知；
- 外部 inputs/outputs、collective wait、observable store、unknown call 和 control-flow join 形成显式 boundary；
- scope 大小受 op/value/relation 数硬限制，超限时沿合法 boundary 切分并保留 baseline。

island 不是 fusion partition。planner 不枚举所有 partition；它先建立最大可分析 island，再由选中的 storage/movement/
effect edges 自然导出 resident components。

### 3.2 IndexRelation

`IndexRelation`描述consumer logical index到producer logical index的关系，是从structured IR重算的受限piecewise
quasi-affine functional map：

```text
IndexRelation {
  relation_version
  consumer_domain
  producer_domain
  pieces[]
}

RelationPiece {
  half_open_consumer_box
  normalized_output_exprs[]
}
```

expression只允许dimension、constant、checked add、constant multiply、positive-constant floor-div/mod，以及用这些节点规范表示的
linearize/delinearize。这个closed algebra覆盖identity、projected permutation、broadcast、static slice、reshape
reassociation和concat edge piece；动态或无法表示的关系保留显式movement/spill baseline，不用value/op名恢复。

canonical form必须验证rank/static extent、每个piece box都包含于`consumer_domain`、piece两两互斥，并证明所有非空piece的
union **exact覆盖** `consumer_domain`；空consumer domain只能对应空piece集合。每个piece的output expression数必须等于
producer rank，且在该piece全域上的image必须落入`producer_domain`。缺piece、重叠piece、越界image或只形成sound
over-approximation都不是合法relation。规范化折叠`+0`/`*1`等唯一形式，删除empty piece，按box和expression canonical bytes
排序，并只在union仍是rectangle且expression一致时合并piece。digest不得依赖MLIR context、pointer或printer order。

所有构造与查询使用typed outcome：

```text
RelationOutcome<T> {
  status: Success | UnsupportedRepresentation | ResourceExhausted | InvalidInput
  value?
  reason?
  work_summary
}

RelationProperty = ProvenYes | ProvenNo | Unknown
```

至少提供`compose`、injective/surjective/bijective分类、`tryInverse`、exact image/preimage、tagged sound bounding-box cover及
logical point evaluation。physical point/bit/byte offset不属于`IndexRelation`，必须由08把logical result与
`PhysicalEncoding`组合后计算。exact result与sound over-approximation必须类型区分；all-and-only rewrite、descriptor proof和
semantic equivalence只能消费exact结果。非bijective是`ProvenNo`而非invalid；fuel耗尽不返回partial relation，也不缓存为
semantic truth。

```text
RelationWorkPolicy {
  rewrite_steps
  expression_nodes
  pieces
  box_fragments
  division_depth
}
```

work按canonical piece/node order消费；所有字段versioned并有hard cap。

关系组合只服务：

- 判断 view 是否零拷贝；
- 把 consumer tile 反推为 producer dependent region；
- 验证多 operand pointwise 的 index 一致性；
- 计算 transfer descriptor cover；
- 证明等价改写覆盖 all-and-only logical elements。

`IndexRelation` 不复制 source IR 的长期事实，不写入 candidate attrs；08 负责把选中的关系解释为 physical map、view 或
movement。

### 3.3 ImplementationFamily

`ImplementationFamily` 是 10 中 target capability provider 暴露的参数化实现集合，不是按 op/dtype/layout 枚举的 case 表。
06作为planner consumer统一拥有implementation、physical encoding、transfer route及communication capability query的顶层outcome协议；各provider
不得自定义第五种状态或把空domain混入`Invalid`：

```text
ProviderQueryStatus = Available | Unsupported | ResourceExhausted | Invalid

ProviderUnsupportedReason =
    UnsupportedRepresentation
  | UnsupportedCapability
  | InfeasibleConstraints

ProviderKindV1 = Implementation | PhysicalEncoding | TransferRoute | Communication
```

`Available`才可携带family domain/baseline或exact proof value；`Unsupported`必须携带上述typed reason；
`ResourceExhausted`不携带partial result且不得缓存为capability truth；`Invalid`表示query、descriptor或registry合同错误并fail
closed。`InfeasibleConstraints`专指表示和资格能力均受支持、但normalized constraints/domain或selected exact参数无解，不是独立
顶层状态。provider可增加诊断detail，但planner控制流只分这套共享status/reason，不按implementation/route/communication分叉。

四类provider共享唯一cache identity：`(ProviderKindV1, full canonical QueryKey bytes)`；比较和命中必须使用完整bytes，digest只作索引加速。
每类QueryKey由owner all-and-only编码其query全部字段以及provider schema/registry revision；06不能拿一份“通用字段子集”替代。
特别是communication key必须覆盖collective semantic/rank-group/topology、available resource domain、schedule constraints、staging
bounds和context/work policy。`ResourceExhausted`和`Invalid`不缓存，Unsupported只能缓存其完整typed reason与key，任何registry/
policy/context变化自然miss。

每个 family 声明：

- semantic kind 和 numeric contract；
- engine/instruction form 与 read/write/async effects；
- operand/result `IndexRelation` 约束；
- dtype、accumulator、rounding 和 exception domain；
- accepted physical encoding domain；
- tile geometry、alignment、tail、invalid-lane precondition和result transfer function；
- temporary/accumulator/resource requirements；
- parameter schema，例如 orientation、batch、vector width 或 algorithm variant；
- materializer 和 exact legality hook。

planner 先根据语义和粗约束获得 family descriptor，再在 frontier 中按需缩小参数 domain。它不得预先展开
`op × dtype × layout × tile × route × order` 的笛卡尔积。无法被 provider 资格化的能力 fail closed。

### 3.4 PhysicalEncoding

`PhysicalEncoding` 的详细合同由 08 拥有。本文只依赖其纯函数语义：

```text
phi(encoding, logical shape, dtype, logical index)
  -> physical bit offset + valid/padding domain
```

encoding 必须能回答 storage extent、alignment、block/tail、padding domain、view compatibility 和 transfer descriptor
cover；byte-addressable target field只能在bit offset可整除8后做checked byte投影，不能让bitpacked BOOL丢失byte内位置。
`Tensor`、`Cx`、`NCx` 是 target provider 的具体 encoding，不是 planner 内散落的字符串分支。

### 3.5 StorageRealization 和 PhysicalVersion

每条 logical value edge 最终选择一个 `StorageRealization`：

- `View`：相同 storage root 上的零拷贝 typed view；
- `ComputeAbsorbed`：index/layout relation 被选中 implementation 参数吸收；
- `ResidentPhysicalVersion`：producer/consumer 共享一个 SPM physical version；
- `BoundaryTransfer`：外部 DDR 与 SPM 间由 RDMA/WDMA/GS 等直接映射；
- `LocalMovement`：SPM 内显式 layout/repack；
- `StagedMovement`：直接路线不可覆盖时的有限 typed staging；
- `SpillReload`：producer 显式写 DDR、consumer 显式重载；
- `ImmutablePrepackedResource`：有稳定 owner、identity 和发布生命周期的 immutable physical payload。

这些名称是 planner 内部分类；accepted IR 只保留具体 memref/view/movement/compute/effect。每个 live physical version 由
`(storage root, logical tile region, encoding, valid domain, InvalidLaneState)`标识；其provider completion、consumer set和
last-use event可从selected IR重算。

`InvalidLaneState`是transformation-local有限小格，只区分`NoInvalidLanes`、`Unknown`和
`KnownSplat(typed raw value)`；若encoding存在固有piecewise padding class，也只能按hard-capped canonical segment class保存。
zero是`KnownSplat(0)`，某值是否neutral必须再结合implementation family、operand role和numeric policy判断，不能把“zero”
普遍等同“neutral”。bitpacked尾部未用bits同样属于invalid lanes。implementation family声明每个operand的
`requiredInvalidLanePredicate(role, state)`、access policy和result transfer function；route以`outState = inState`、
`Unknown`或`KnownSplat(v)`等有限transfer function声明写后状态。masked/unobserved属于consumer access proof，不进入buffer
内容state；incoming state不同的version不得canonical merge。

这个analysis状态不作为shadow attr进入accepted IR。selected candidate必须用显式fill、segmented valid-tail command、mask、
typed family valid-lane mode或完整physical write使verifier能从IR重建pre/post condition；例如只写valid segments的mapped load
产生`Unknown` padding，除非另有显式fill。无法把required neutral state物化为这些事实时，该choice不合法。

temporary compiler-owned layout tensor 不能伪装成 immutable resource。只有 package/artifact owner、logical identity、
encoding identity、read-only lifetime 和 invalidation 都明确时，才能选择 prepack。

### 3.6 Candidate

candidate 是一个 transformation-local complete proposal，至少包含：

- island 中各 root 的 implementation family 参数；
- collective roots的selected `CommunicationScheduleFamily`参数和all-rank compatibility constraints；
- shared tile-domain variables 和 dependent-region relations；
- values 的 selected physical encoding 与 storage realization；
- buffering slot 和 DAG-legal issue/order decisions；
- exact-materializable rank clone；
- legality status、resource vector、cost vector 和 deterministic tie-break signature。

candidate ID、score、beam parent 和 search trace 不写入 payload IR。accepted IR 是唯一事实源。

## 4. Typed Transformation Mechanism 合同

机制库位于固定structured optimization与search policy之间。它可以复用MLIR upstream的`TilingInterface`、destination-style
helpers、tensor subset/view folding、Linalg tiling/fusion和canonicalization patterns，但必须用Wafer typed precondition约束
适用域，并返回`applied / not-applicable / proof-failed`及改写后的实际IR。linked、registered或能在`wafer-opt`里手工运行不算
planner采用；只有候选生成代码真实调用、至少一个测试发生改写且完整exact gates接受，才可进入choice domain。

每个mechanism是policy-free原子操作：不读取model name、candidate score、beam width或全局搜索历史，不自行选择另一个
implementation/route，也不在失败时串接fallback。应用发生在isolated complete-rank clone上；任何use-def、alias、effect、
lifetime或allocation root变化都使旧analysis失效，必须从改写后的IR fresh重建semantic descriptor、IndexRelation、SPM/DDR
lifetime、completion和cost。greedy convergence、pattern application order及generic canonicalizer结果都不能成为正确性前提。

```text
PhysicalMechanismCutPointV1 = StructuredTensorPrePhysical | SelectedPhysicalPayload

MechanismSpec {
  mechanism_key
  ir_cut_point
  parameter_schema
  precondition_contract
  numeric_effect_contract
  rewrite_work_policy
  proof_obligations
}
```

这两个值分别一对一映射05 compiler-wide `OptimizationCutPointV1::{StructuredTensorModule,
SelectedPhysicalPayloadModule}`，不是另一套可扩展cut-point registry。registry按`MechanismKey`唯一，并验证05 spec绑定的cut与这里
映射相同；同一utility若服务两层必须使用两个key/spec。mechanism只能接收该cut的typed root，错层调用返回`Invalid`且不改clone。
`StructuredTensorPrePhysical`仍是Linalg/Tensor/SCF pure tensor program，`SelectedPhysicalPayload`已经含memref allocation
root/encoding/movement/effect/completion，两者不能共用normal-form entry point。

统一返回：

```text
MechanismOutcomeV1 = Applied | NotApplicable | ProofRejected | ResourceExhausted | Invalid

MechanismResult {
  status: MechanismOutcomeV1
  changed_roots
  work_summary
}
```

除`Applied`外不得修改clone；`Applied`后descriptor、relation、alias/effect、SPM/DDR、completion和cost全部fresh重算。
`changed_roots`只用于诊断和限定fresh扫描入口，不能充当“其余analysis仍有效”的许可，因此不另设可造成选择性复用的
`invalidated_analysis_classes`字段。
Q32.M的mandatory closure不是“尽量接入”，而是下面的最低通用支持面；每一row至少有一个非workload特化的真实source发生
改写并通过完整exact gate：

| mandatory mechanism / closure | IR cut point | 最低完成域 |
| --- | --- | --- |
| relation/view normalization | StructuredTensorPrePhysical | identity、permutation、slice、reshape、concat piece |
| pure structured producer tiling/fusion | StructuredTensorPrePhysical | DPS pure producer→consumer，static tile与tail |
| pointwise relation propagation | StructuredTensorPrePhysical | bijective relation及显式证明的scalar/broadcast |
| implementation absorption | StructuredTensorPrePhysical | permutation进入已资格化oriented contraction |
| shared physical-version reuse | SelectedPhysicalPayload | 通用2+ fanout/multi-root，不识别QKV等角色 |
| mapped boundary folding | SelectedPhysicalPayload | compact DDR view直接进入qualified SPM encoding |
| movement elimination | SelectedPhysicalPayload | physical-isomorphic、no-op及冗余materialization |
| resident cut elimination | SelectedPhysicalPayload | producer/consumer共享version并删除显式spill/reload |
| structured-tensor post-mechanism closure | StructuredTensorPrePhysical | 运行05 required tensor normalizer及该cut已资格化fixed utility；不生成search choice |
| selected-payload post-mechanism closure | SelectedPhysicalPayload | 只运行07 selected-payload normalizer及该cut proof-preserving cleanup；不调用05 normalizer、不生成search choice |

whole-tensor share-vs-recompute、static loop-invariant hoist、loop multi-instance/ping-pong，以及numeric reassociation、reduction
tree和distribution是capability-conditioned：只有其typed precondition/provider row闭合才进入domain。immutable prepack、
dynamic-shape specialization、paged/serving与segmented MoE仍deferred，不用mandatory case冒充。

两条post-mechanism closure按`MechanismSpec.ir_cut_point` dispatch，并在对应cut的每次`Applied`后始终运行；它们不产生“开/关”
candidate、不参与beam或收益排序。fuel耗尽或postcondition失败直接丢弃clone；成功后仍按上面的统一规则fresh重算。一个
physical-payload mechanism不得先跑05 tensor normalizer，一个tensor mechanism也不得提前套07 root/layout/effect合同。

机制按以下语义类别组织；类别不是固定执行阶段，也不是op-pair case表：

### Bitwise index/view normalization

- identity view、连续 reshape reassociation、互逆 transpose/permutation、full static slice 等可证明关系的组合和消除；
- 不改变 logical iteration、运算次序、dtype 或 storage value；
- 只canonicalize logical relation；已物化storage间的movement仍须由08证明physical-isomorphism后才能删除；
- 使用唯一normal form和单调下降measure，无法证明下降时消耗明确rewrite fuel后停止，避免循环或无界表达式增长。

### Pointwise relation propagation

- pure pointwise op 可把同一 bijective relation 推过 result；
- 所有非 scalar operands 必须在 logical index 上一致，broadcast 必须显式证明；
- scalar body、rounding、NaN/Inf 和 conversion 顺序保持不变；
- padding lane 不属于 logical domain，除非 08/10 共同证明 selected implementation 的 padding invariant。

### Target-qualified implementation absorption

- transpose、orientation、vector packing 或 boundary mapping 只有在 family 明确提供参数并由 instruction/CRT/CModel 纵向
  资格化时才可吸收；
- 例如 GEMM transpose flag 不能因为底层 packet 疑似存在就进入 accepted candidate，必须由 10/11/14/17 的 typed
  capability 闭环支持；
- 吸收后的 logical result 和 numeric contract 必须与 source op 相同。

### Multi-root shared-input physical-version reuse

- 多个独立 contraction/compute roots 若共享相同 input region、numeric/effect 独立、tile domains 相容，可形成一个
  shared-load hyperedge；
- 该规则只共享 physical version、movement 和局部调度，不合并数学 result，也不要求固定 root 数量；
- fanout、不同 result shape 和不同 implementation family 通过 relation/constraint 处理，不匹配 QKV 或 gate/up 名称。

### Pure structured producer tiling/fusion

- mandatory domain只处理实现`DestinationStyleOpInterface`/`TilingInterface`的pure structured producer与consumer，且
  dependent region由exact `IndexRelation`推出；unknown effect、control/collective barrier和不可解释DPS tie均拒绝；
- 只切parallel iterator并保持scalar region evaluation、reduction iterator次序、init tie和每个logical producer point的执行
  multiplicity。会重复producer计算、多次观察exception/status或改变reduction order的proposal，只有独立numeric/effect contract
  明确许可时才进入capability-conditioned domain；
- static full/tail tile都必须all-and-only覆盖原iteration domain，multi-use producer不得只改写部分use后留下语义不同的共享值。

### Mapped boundary folding、movement与resident-cut elimination

- mapped folding只在planner已选择08返回的direct route alternative后，把compact DDR typed view与qualified SPM version物化为
  destination-style load/store；exact cover失败返回`ProofRejected`，不在mechanism内部改选staged route；
- movement elimination必须证明两端logical relation、physical bit map、root-relative range、alias/effect、completion和
  `InvalidLaneState`等价。只比较logical shape、footprint byte数或layout label不足以删除movement；
- resident-cut elimination只删除compiler-managed、不可观察的spill/store/reload cut，并要求同一logical region与physical
  version、producer completion支配全部consumer、last-use/reuse关系可重建。external/ABI store、unknown effect、collective
  completion或尚未通过whole-rank capacity/liveness的proposal都不得删除；
- 上述mechanism只改写当前clone中的typed op/SSA/effect。direct route形成load/store，staged route保持显式temp、DMA、GS和event，
  不产生route attr、descriptor sidecar或lowering-time fallback。

### Numeric-policy-gated algebraic transformation

- reassociation、distribution、reduction tree、online reduction 或 FMA contraction 默认关闭；
- 只有 source IR 明确给出足够的 fast-math/overflow/identity 事实，且 target family 和 numerical verification 支持时才可
  进入 domain；
- 缺少许可时保持 source evaluation order 或 baseline，不用性能目标放宽正确性。

所有规则都必须能独立做 property/differential test。未注册的模型 pattern、固定 shape rewrite、operand-position 猜测和
字符串 role matcher 永远不进入 planner。

## 5. Tile、resident dataflow 与执行结构

### 5.1 空间和时间切分

空间 partition 已由 topology、execution mesh 和 Shardy/XLA SPMD 决定。planner 只处理一个显式 logical rank 的 local
program；Cx/NCx 是 rank 内 encoding，不是 mesh partition。全部 rank 独立 synthesis 和验证，即使代码相同也保留 typed
rank identity。

时间切分由以下变量共同决定：

- parallel iterator tile；
- 获 numeric policy 许可的 reduction chunk；
- producer/consumer 间的 dependent tile region；
- invariant physical version 的 reuse window；
- buffer slot、issue order 和 completion edge。

一个 task instance 由 source semantic root、logical tile coordinate、可选 reduction step 和 selected implementation
确定。主 traversal 使用 compact structured loops 和有限 static tail classes，不按 element 或所有静态 instance eager
展开。

### 5.2 Tile domain 生成

tile domain 只从通用事实产生：

- static extent 的 divisors、full extent 和 legal tail；
- target family 的 vector/block/geometry/alignment domain；
- dependent-region relation 的 compatibility；
- SPM lower bound、accumulator 和 staging threshold；
- descriptor count/cover threshold。

domain 以区间、divisibility、有限枚举和 relation constraints 保存。只有进入 frontier 的选择才惰性实例化具体整数；
不得生成从 1 到 extent 的全量 tile 表，也不得对每个 op 独立取 Cartesian product。

### 5.3 Resident dataflow 和 fusion result

selected IR 中：

```text
producer -> same SPM physical version -> consumer
```

表示 resident dataflow。producer 和 consumer 可以是不同 engine、不同 typed task、甚至位于不同
`wafer.tile.region`；region 边界不得自动 store/reload。以下情况形成 cut：

- observable external/ABI boundary；
- explicit spill/reload 或 staged movement；
- unknown/ordered effect 与 control-flow barrier；
- collective/provider completion 尚未闭合；
- relation、numeric、capacity、descriptor 或 instruction legality 不满足。

因此“最终融合组”只是从 selected physical versions 和 cut edges 派生的诊断视图，不是 allocator scope、candidate
identity 或 lowering protocol。

### 5.4 Dynamic loop 和 ping-pong

当前只接受可静态证明的 loop-carried version/lifetime。double buffering 必须物化两个不同 SPM slots，并明确
ready/free event、parity selection 和 last-use；若 dynamic multi-instance、nested tile-region capture 或 loop alias 不能从
IR 证明，planner 保留 single-buffer baseline 或 fail closed。不得以全局 summary 或 shadow loop schedule 掩盖缺失语义。

## 6. 有界联合搜索算法

### 6.1 总体流程

对每个 rank：

1. **Normalize relations**：只调用有fuel和单调measure的bitwise IndexRelation normalizer，建立structured semantic graph、
   IndexRelation和effect/numeric barrier；这不是运行generic MLIR canonicalizer。
2. **Island formation**：建立最大有界 semantic islands；超限沿合法 boundary 切分。
3. **Capability query / skeleton closure**：按 semantic root 查询 parameterized implementation、physical encoding、transfer route和communication schedule
   families，不展开组合；communication只返回13的bounded logical schedule skeleton，新增staging/local-compute node和edge重新进入
   同一implementation→encoding→route domain propagation，直到所有skeleton node都有typed domain；它不能在materializer中隐藏选择。provider
   按canonical semantic signature排序，禁止依赖pointer、registration、DenseMap或parallel completion顺序。
4. **Backward region propagation**：从 outputs/roots 向 inputs 传播 dependent tile region，合并 shared-input constraints。
5. **Constraint propagation**：在`constraint_propagation_work_cap`、`communication_skeleton_nodes_cap`及
   `communication_skeleton_edges_cap`内反复收紧 tile、family params、encoding、route、valid-domain 和 resource lower-bound domains；
   只有能从当前frontier facts证明的bound才可因超过硬容量剪枝，unknown或启发式estimate只能影响访问顺序。空domain立即剪枝。
6. **Reserved baseline**：每个production semantic family必须提供一个可判定的baseline implementation、canonical accepted
   encoding和由target geometry/capacity公式驱动的有界legality-directed safe-tile refinement；若provider没有baseline，该语义
   就不在该profile的production支持面，optimized family不得成为唯一正确性路径。为每rank baseline和all-rank baseline bundle
   保留不可被beam/top-K/optimization deadline淘汰的slot；先物化保持source order、external/boundary Tensor、family
   canonical encoding（如NE GEMM的Cx/NCx）、显式Tensor↔Cx/NCx materialization/spill的baseline，再完成分层exact gate。
7. **Pareto beam exploration**：baseline通过后，只对能消除movement、降低live bytes或启用合法implementation的frontier
   decision惰性分裂。
8. **Per-rank materialization/gates**：将bounded proposals写入隔离complete-rank clones，先完成offset-independent
   instruction/encoding/effect/completion legality，再从unplaced current IR纯evaluate SPM full-arena feasibility并保留
   validated incumbent；只有仍可能改变Pareto/最终选择的shortlist再共享optimization fuel收紧packing lower/upper。
   选定best placement后只apply一次offset，并从该placed current IR重跑SPM owner range/resource、descriptor/address/narrowing、
   DTE receiver/local-offset、per-rank DDR view/range、`PlacedRankCompatibilityClaimsV1`和全部offset-dependent verifier/gate，再fresh recost/
   分桶。旧placement对应的gate或signature一律失效。
9. **Lazy variant coordination**：先传播shared transport variables，再baseline-first、best-first/factorized地惰性join各rank桶；
   不预先形成`K^R`。每个完整variant先运行whole-variant DDR planning，再做post-memory physical
   transport binding并从bound IR形成`AcceptedVariantTransportSignatureV1`，最后运行package eligibility和14
   registry/versioned-signature artifact-eligibility pure preflight；preflight可临时投影prospective capability keys，但不进入
   frontier、accepted IR或artifact，且不生成module。
10. **Selection/commit**：只在相同`PlacedRankCompatibilityClaimsV1` bytes内做local Pareto dominance；从全部passing variants确定性选择winner，
    从winner final bound instruction IR fresh recost，重算每rank capability projection与all-rank set/digest，并将
    final rank programs、`RequiredCapabilitySet`和`ExecutableBundle`在同一Q16 transaction原子发布。只有该commit
    成功后，下游14 target publication才对winner正式lower一次；不是commit后才形成capability set。

reserved baseline不是“三个provider各挑第一个”的约定。06唯一拥有`CanonicalBaselineComposerV1`，它消费各provider的
`canonical_baseline` recipe并形成一条无搜索分支的正确性路径：

```text
CanonicalBaselineComposeRequestV1 {
  normalized_rank_semantics
  target_capability_context
  deterministic_work_policy
  implementation_provider_and_registry_snapshot
  physical_encoding_provider_and_registry_snapshot
  transfer_route_provider_and_registry_snapshot
  communication_provider_and_registry_snapshot
}

CanonicalBaselineComposeResultV1 {
  status: Success | Unsupported | ResourceExhausted | Invalid
  reserved_unfinalized_seed?
  canonical_composition_key?
  reason?
  work_summary
}
```

四个snapshot都是invocation-local只读typed provider interface + immutable registry revision，不是预查询结果、side table或
artifact；caller不能提前拼四套结果成为第二个composer。composer按typed structured traversal得到semantic root ordinal，再按
`(root ordinal, operand/result edge ordinal)`和collective ordinal访问。对每个root依次发出implementation query、实例化其
baseline requirement并取得implementation domain中的canonical initial tile。随后在**该tile**上发出encoding query并实例化唯一
canonical accepted encoding；对每条boundary/dataflow edge以已选tile/两端encoding发出route query；最后以同一tile/slice发出
communication query并合并rank-group、message、resource与completion约束。每个recipe必须是一个canonical parameter point加一个有单调measure的bounded materialization rule，不能返回
“任选一个safe rule”字符串。每步只做domain intersection和typed materialization，不回溯、不尝试optimized family。provider
返回`Unsupported`或`ResourceExhausted`时原样传播其typed reason/status；声称`Available`却缺recipe、含多个canonical point或
point不属于domain是registry合同`Invalid`。只有各合法baseline recipe的constraint intersection为空才返回
`Unsupported(InfeasibleConstraints)`，说明该semantic/profile不在production支持面。

communication baseline若返回staging/local-compute skeleton，composer按skeleton node/edge canonical ordinal为每个新增语义node
顺序取得implementation baseline，再取得encoding baseline，并为新增edge取得route baseline；不允许communication materializer
直接挑选这些sub-family。每插入/解析一个node或edge都先消费对应reserved skeleton cap及provider/constraint work；每解析一个
node都使unresolved node数减一，skeleton禁止
嵌套产生collective；任一sub-baseline无唯一compatible recipe按上述typed status失败。tile下降时这些sub-query/result也全部失效并
随主edge一起fresh重发。

若任一tile-dependent provider返回`Unsupported(InfeasibleConstraints)`或geometry/descriptor gate证明当前tile不合法，composer
按下述规则下降tile；表示/能力Unsupported直接透传。每次下降都丢弃旧encoding/route/communication result、descriptor、
materialization及gate，在新tile上按encoding→route→communication顺序fresh query和重物化，且每个query/rewrite在执行前计费；
旧query key只留在本次ordered telemetry/composition trace，不能当新tile事实。

同一composer运行baseline-only safe-tile refinement。`bounded_materialization_rule_key`必须解析到immutable、versioned
`BaselineSafeTileRuleV1` registry row；它消费当前query形成的`CanonicalFiniteTileDomainV1`、可选strict upper bound和
`baseline_safe_tile_rule_work_cap`，返回唯一point或typed failure。domain只允许§5.2的有界区间、congruence、有限枚举、
geometry和relation constraint algebra，并携带canonical bytes/digest；rule不能读取op名字、容器遍历顺序或provider side state。
总序固定为`(checked_sum(tile_extents), tile_extents in iterator order)`的lexicographic升序。初始请求求整个domain的exact
`argmax`；下降请求求满足componentwise不大于当前tile且在上述总序严格更小的子域exact `argmax`。每个Success都必须携带
可独立验证的domain-membership证明，以及“满足请求边界且比返回point更大的domain为空”的exact maximality证明；实现可使用
bounded Presburger emptiness、congruence jumping或已验证的separable closed form，但没有separability proof时不得逐维greedy，
也不得枚举`1..extent`或完整tile Cartesian product。rule work在求解/proof前checked计费；耗尽返回`ResourceExhausted`且不
返回point或partial proof。改变domain algebra、total order、proof codec或work unit必须升级rule/policy schema。

每次refinement只能componentwise不增并至少把一个正extent减小；canonical measure就是上述总序的严格下降，dimension tie按
iterator ordinal。geometry/descriptor constraint通过同一rule的strict-upper-bound请求得到下一个exact最大合法点；pressure
proof选中dimension时，该请求把此dimension约束为严格小于current、其它dimension约束为不大于current，再在所得子域取exact
argmax。只有完整baseline materialization得到validated
`ProvenInfeasible` capacity proof时，才按proof的validated pressure view选择最早可减pressure iterator；proof不携带pressure
view（例如capacity cut）时，不恢复demand集合，而按iterator order选择其next legal point能使physical-version checked footprint
bound严格下降的首个dimension，仍无此类dimension时选择首个大于一的extent。每次下降前消费reserved
`baseline_tile_refinements_cap`，optimized neighbor只能消费独立`optimized_tile_refinements_cap`，两者不得互借。
`ResourceExhausted`不移动tile，也不伪装成infeasible；当前point确证不合法且不存在更小domain point时返回
`Unsupported(InfeasibleConstraints)`。所有query、refinement和materialization都在发生
前消费§6.6的provider及baseline work allowance，所以循环必然终止。

成功结果恰有一个`is_reserved_baseline=true` seed；`canonical_composition_key`由schema version，以及按root/edge/collective
ordinal和refinement-attempt ordinal记录的四类**ordered full query-key/result-status/reason sequences**、selected
recipe/parameter canonical bytes、最终baseline tile/encoding/route/communication decision bytes做
length-prefix连接。它只用于seed order、重放和telemetry，不是accepted attr或final candidate signature。各rank composer完成后，
all-rank coordinator只接受唯一baseline组合中shared communication constraints可exact匹配的bundle；不匹配即profile
`Unsupported`，不能回退到optimized family。由此将全部optimized生成预算设为零仍能得到同一bounded baseline纵向。

唯一per-rank reserved baseline按logical-rank canonical order直接组装一次all-rank baseline，不进入optimized join queue，也不
消费任何`optimized_*` counter；它的complementary transport constraint match、whole-variant DDR、
post-memory binding、package/ABI gate使用baseline
reserved exact-gate/work allowance并先于optimized coordinator完成。per-rank/all-rank survivor容器在policy cap之外各保留一个
不可淘汰baseline slot；只有`baseline_status=Ready`后才启用optimized finalized/join/frontier/combinations caps和optimization
deadline。因此任意合法policy（包括全部optimized cap为零）不会在baseline bundle形成前被optimized work耗尽。

Q32.B到Q32.G的迁移只在compiler owner-private bundle transaction保留一个invocation-local producer seam，不新增CLI、pass、
driver mode或第二条pipeline：

```text
RankPlanningRequest {
  source_rank_module: read-only
  logical_rank
  target_capability_context
  deterministic_work_policy
}

RankPlanningResult {
  status: Success | Unsupported | ResourceExhausted | Invalid | Cancelled
  unfinalized_proposal_seeds[]
  telemetry
}

UnfinalizedRankProposalSeed {
  move-only complete-traversal proposal module
  canonical_seed_order_key
  is_reserved_baseline
}
```

这个seam精确位于rank proposal producer与compiler-owned finalization之间。seed的module已覆盖完整rank traversal，但仍是
**unfinalized** proposal；function-boundary bufferization、physical-memory replanning、best placement apply、offset-dependent
verifier和fresh recost都只在seam之后执行。因此seed不得携带`PlacedRankCompatibilityClaimsV1`、
`AcceptedVariantTransportSignatureV1`、`exact_static_cost_vector`或
`canonical_candidate_signature`；placed claims和rank-local signature只有从finalized/placed/unbound current rank IR重算后成立，
accepted transport/final candidate signature还必须等待all-rank binding后从bound IR fresh构造。`canonical_seed_order_key`
只决定finalization的确定性访问顺序，不能跨finalization充当candidate signature或semantic fact。

`Success`要求非空seed且恰有一个reserved baseline。optimization budget或baseline-ready后才arm的optimization deadline耗尽时仍返回
`Success`和baseline，并在telemetry记录原因；只有baseline seed/proof在其reserved allowance内也无法完成时返回
`ResourceExhausted`。`Unsupported`表示target/workload没有canonical baseline，`Invalid`表示request、registry或内部合同错误；
`Cancelled`只表示driver/process在任意时点显式取消整次transaction；四种非success状态都不返回partial seed或artifact。

Q32.G前production factory固定使用legacy `RankFrontierProducer`；`Compiler/Testing.h`中的临时test-only注入让新旧producer都
只产生上述seed，并经过完全相同的finalization、all-rank coordinator、package、TargetCall和SystemC链。seed在isolated
context/clone中构造并在同一context完成finalization；只有final survivor才按canonical rank order导回bundle owner context。
seam不保存plan或SSA pointer。Q32.G把production固定到新producer并同时删除legacy producer与临时planner selector，不能让它
演化成长期双入口。

### 6.2 Canonical frontier state

search约束、placed rank claims与all-rank accepted binding使用三个不同typed对象，禁止用一个“compatibility signature”跨越
materialization和binding：

```text
TransportInstanceRefV1 {
  instance_ordinal: U32
  rank_group_ordinal: U32
}

TransportMessageIdentityV1 {
  instance: TransportInstanceRefV1
  phase_ordinal: U32
  message_ordinal: U32
  sender_logical_rank: U32
  receiver_logical_rank: U32
  segment_ordinal: U32
  logical_byte_offset: U64
  byte_count: U64
}

TransportProgramPointV1 {
  canonical_operation_path: Sequence<U32>
}

TransportMessageRoleV1 = Send | Receive
TransportCompletionKindV1 = DirectDTEWait
TransportResourceKindV1 = AllocationProfile | ReceiverFSM | CompletionProfile
ConstraintDomainKindV1 =
    RankGroupTopology | FamilyParameter | MessageIdentity
  | Segmentation | PeerOrderCompletion

CanonicalConstraintDomainV1 {
  domain_kind: ConstraintDomainKindV1
  registry_revision: U16
  registry_digest: Digest32
  canonical_normalized_domain_bytes: Bytes
}

TransportMessageDomainV1 {
  instance: TransportInstanceRefV1
  message_ordinal: U32
  identity_domain: CanonicalConstraintDomainV1
  segmentation_domain: CanonicalConstraintDomainV1
}

TransportResourceAssignmentKeyV1 {
  instance: TransportInstanceRefV1
  resource_kind: TransportResourceKindV1
  resource_ordinal: U32
}

TransportResourceDomainClaimV1 {
  requesting_logical_rank: U32
  assignment_key: TransportResourceAssignmentKeyV1
  resource_registry_revision: U16
  resource_registry_digest: Digest32
  allowed_resource_ids: Sequence<U32>
}

PlacedTransportMessageClaimV1 {
  identity: TransportMessageIdentityV1
  local_logical_rank: U32
  local_role: TransportMessageRoleV1
  local_buffer_root_ordinal: U32
  local_root_relative_offset: U64
  local_span_bytes: U64
  required_assignment_keys: Sequence<TransportResourceAssignmentKeyV1>
}

TransportCompletionClaimV1 {
  identity: TransportMessageIdentityV1
  local_logical_rank: U32
  local_role: TransportMessageRoleV1
  completion_kind: TransportCompletionKindV1
  issue_point: TransportProgramPointV1
  wait_point: TransportProgramPointV1
  ordered_predecessor_messages: Sequence<TransportMessageIdentityV1>
  wait_dominates_terminal_paths: Sequence<TransportProgramPointV1>
}

BoundTransportResourceAssignmentV1 {
  assignment_key: TransportResourceAssignmentKeyV1
  resource_registry_revision: U16
  resource_registry_digest: Digest32
  selected_resource_id: U32
}

BoundTransportMessageClaimV1 {
  identity: TransportMessageIdentityV1
  sender_buffer_root_ordinal: U32
  sender_root_relative_offset: U64
  sender_span_bytes: U64
  receiver_buffer_root_ordinal: U32
  receiver_root_relative_offset: U64
  receiver_span_bytes: U64
  required_assignment_keys: Sequence<TransportResourceAssignmentKeyV1>
}

CompatibilityConstraintKeyV1 {
  schema_version: U16 = 1
  transport_instances: Sequence<TransportInstanceRefV1>
  rank_group_topology_domain: CanonicalConstraintDomainV1
  family_and_parameter_domain: CanonicalConstraintDomainV1
  message_identity_segmentation_domains: Sequence<TransportMessageDomainV1>
  peer_order_completion_domain: CanonicalConstraintDomainV1
  shared_resource_domains: Sequence<TransportResourceDomainClaimV1>
  target_capability_context_canonical_bytes: Bytes
  target_capability_context_digest_for_index: Digest32
}

PlacedRankCompatibilityClaimsV1 {
  schema_version: U16 = 1
  logical_rank: U32
  canonical_message_claims: Sequence<PlacedTransportMessageClaimV1>
  requested_shared_resource_domains: Sequence<TransportResourceDomainClaimV1>
  terminal_completion_claims: Sequence<TransportCompletionClaimV1>
}

AcceptedVariantTransportSignatureV1 {
  schema_version: U16 = 1
  canonical_rank_ordered_bound_message_claims: Sequence<BoundTransportMessageClaimV1>
  canonical_shared_resource_assignments: Sequence<BoundTransportResourceAssignmentV1>
  terminal_completion_claims: Sequence<TransportCompletionClaimV1>
}
```

本节上述compatibility records/nested records，以及后文明示“使用§6.2统一canonical codec”的06-owned records共用下列codec；
已经单独冻结专用domain/field encoding的既有record保持其专用合同，不被本段覆盖。统一codec的domain prefix固定为
`"wafer.physical-dataflow." || exact case-sensitive record type spelling || "\0"`；随后按声明顺序把field id从1连续编码为
`u16be(field-id) + u8(type-tag) + u32be(payload-size) + payload`。type-tag固定为
`Bool=0x01, U16=0x02, U32=0x03, U64=0x04, ClosedEnum=0x05, Digest32=0x06, Bytes=0x07,
Record=0x08, Sequence=0x09, Optional=0x0a`；integer big-endian，Digest32恰32 bytes，Bytes/Record length由外层field
给出，Sequence为`u32be(count)`后连接每个`u32be(element-size) + element-payload`。`Optional<T>` payload为
`0x00`或`0x01 + u8(inner-type-tag) + u32be(inner-size) + inner-payload`，禁止nested optional。closed enum按本文
每个声明中从左到右的顺序固定为从0开始的u32 ordinal；改变顺序、字段、variant或element schema必须新增版本。
引用05/08/13/14 owner的nested record时，Record payload直接使用owner定义的完整canonical bytes并length-prefix，06不复制其
字段。所有Sequence若下文称canonical set，仍使用Sequence wire type，但必须按完整element bytes排序去重。

`TransportProgramPointV1.canonical_operation_path`从当前rank entry起按
`function/region/block/operation` ordinal交替编码且非空；它只由current structured IR重算，不使用symbol或SSA打印名。
message identity按
`(instance, phase, message, sender, receiver, segment, logical_byte_offset, byte_count)`排序且`byte_count>0`；同一
instance内`(phase,message,segment)`不得映射到两个不同identity。resource assignment key按
`(instance, resource_kind, resource_ordinal)`排序；allowed resource ID严格递增、非空且属于record绑定的registry。
placed message按`(identity, local_role)`排序，completion按`(identity, local_logical_rank, local_role)`排序，bound message按
`(sender_logical_rank, receiver_logical_rank, identity)`排序，assignment按key排序，重复一律拒绝。每个
`CanonicalConstraintDomainV1`的kind必须匹配其所在field，registry digest必须验证其canonical normalized bytes；
`target_capability_context_digest_for_index`必须等于完整context bytes按§9 context codec重算的digest，不能单独信任。

`CompatibilityConstraintKeyV1`由current structured semantics和四类provider domain/formula规范化得到，只用于frontier merge、
constraint propagation和lazy join访问顺序；它允许domain而不是假装已经做出binding，materialization/rewrite后必须重算，不能进入
seed、IR、artifact或final gate。context相等/排序比较完整canonical bytes，digest只作索引，不能以hash碰撞合并state。
`PlacedRankCompatibilityClaimsV1`只能从**finalized、placed但尚未做all-rank transport binding**的
current instruction IR fresh构造：message claim逐项包含typed instance identity、sender/receiver rank、peer约束、byte/segment
范围、accepted local offset/range、issue/wait/order/completion relation，以及当前Direct DTE ABI可兑现的allocation-profile、
receiver-FSM和completion resource domain；exact endpoint只由topology/logical peer派生，不请求不存在的static channel ID；
它不能伪造尚未分配的actual binding，也不读取search key、provider side object、SSA打印名或discovery order。

同rank候选只有placed claims完整bytes相等时才能做local dominance和分桶。每个placed record必须all-and-only覆盖该rank current
instruction IR中的send/receive message、所需shared resource domain和completion：`local_logical_rank`必须等于outer rank，role必须
与identity sender/receiver一致，local span必须等于identity byte count且落在accepted SPM root range；每条message恰有一个同role
completion claim，issue支配wait、wait支配列出的all-and-only terminal exit path，predecessor只引用同rank可达message。缺项、
多项、交叉role、unknown path或domain均拒绝。

all-rank coordinator按typed message identity匹配恰一sender和恰一receiver complementary claim；两者identity、segment、bytes和
required assignment key set必须相同。每个assignment key汇集all-and-only requesting-rank domain claim并求typed ID交集；空交集
拒绝。13在对应baseline/optimized exact-gate deterministic allowance内，对这些有限、registry-bounded domains求全局无冲突
assignment，并在所有合法assignment中取按
`(assignment key, selected resource ID)`展开后的lexicographic最小解；不能逐key greedy后因可避免的冲突拒绝，也不能按线程完成
顺序选解；allowance耗尽返回ResourceExhausted而不伪装成domain不相容。随后把唯一allocation profile、receiver FSM和completion profile通过
`DirectDTEBindingAttr`等typed binding写回全部rank clone；endpoint从topology派生，当前ABI不伪造channel/event ID。一个bound
message只引用外层assignment表中的key，不复制selected ID；一个assignment key恰有一个
`BoundTransportResourceAssignmentV1`，且selected ID属于每个对应domain。sender/receiver root、offset/span从各自bound current IR
fresh重算并与原placed claim逐字段相等。

绑定后从**bound all-rank current IR** fresh构造`AcceptedVariantTransportSignatureV1`，验证message集合等于所有rank placed
message identity的canonical union，每个identity恰一bound message、两个role completion claim及all-and-only assignment coverage；
任一未使用assignment、缺binding、resource冲突、completion path不闭合或terminal仍pending都拒绝。只有这份signature通过的variant
可进入final recost/selection/commit。
三类对象都以versioned field-order/length-prefix canonical bytes比较，digest只用于cache/telemetry而不替代bytes或进入bundle。
placement使placed claims失效；binding、instruction或event变化使accepted signature及canonical candidate signature一并失效。

beam state 只保留继续求解所需的最小规范形式：

```text
graph frontier
+ live physical versions:
    (storage root, tile region, encoding, valid domain, invalid-lane state)
+ constrained implementation/tile/route domains
+ SPM live signature and sound lower bound or unknown
+ engine/effect/completion frontier
+ CompatibilityConstraintKeyV1 / remaining domain
+ static resource/cost vector
```

这里的frontier lower bound只来自尚未物化阶段可证明的physical-version size、同时存活和target geometry事实；不能把
representative tile estimate伪装成exact packing结果。已经完成且不再影响 future liveness/constraints 的历史决策不进入 key。等价状态按 relation、domain 和 live signature
canonicalize 后合并。这里的cache是planner-owned `CanonicalFrontierStateKeyV1` memo，不是provider query cache；它使用完整规范化
state bytes：semantic/index relation、shape/dtype、tile/valid domain、source/
destination memory space与view offset、normalized alias/effect signature（same-root relative overlap、proven-disjoint或
unknown/may-alias）、family/route/encoding参数、按全部读写operand角色记录的incoming source/destination invalid-lane states
和完整`target_capability_context`（target/profile/ABI、model provider/profile或board environment/allowlist identity、revision、
qualification floor）的完整canonical bytes及deterministic work-policy完整bytes；unknown alias只可复用保守
fail-closed结果，不含op/value/
模型名。search-time compatibility key至少约束collective/message bytes与segmentation、peer/order/completion和shared remote
buffer/resource requirements；不同constraint key/domain不得canonical merge。即使search key相同，per-rank local
dominance也只比较whole-variant聚合函数已声明为componentwise monotone（例如sum/max）的exact dimensions；非单调或跨rank
目标进入constraint domain或延迟到finalized all-rank matching后比较。frontier key比较完整versioned bytes，digest只作memo索引；
provider cache仍只能使用§3.3的`(ProviderKindV1, owner full QueryKey bytes)`，不得拿本段flattened state字段命中provider结果。

### 6.3 惰性 decision 和候选顺序

planner 不枚举所有 fusion partitions、resident subsets、tile integers、physical versions 或 topological orders。decision
按预期收益/约束强度排序：

1. 能 canonicalize 或 compute-absorb 的 relation；
2. 大 boundary movement 的 direct route / resident opportunity；
3. shared-input reuse hyperedge；同relation/domain的全部compatible roots先形成deterministic maximal hyperedge，冲突只允许
   hard-capped domain split，不枚举`2^fanout` compatible subsets；
4. SPM peak 或 descriptor legality 所迫使的 tile refinement；
5. 少量能改变 reuse distance 的 ready-task alternatives；
6. optional buffering。

上述顺序不是“实现自己估一个收益”。所有待访问decision都先形成唯一work key：

```text
CanonicalDecisionWorkKeyV1 {
  schema_version = 1
  canonical_frontier_state_bytes
  category: RelationAbsorption | BoundaryResidentRoute | SharedInputReuse |
            LegalityTileRefinement | ReadyOrderAlternative | OptionalBuffering
  legality_forced
  proven_constraint_domain_reduction?
  proven_min_movement_bytes_eliminated?
  proven_min_live_bytes_reduced?
  structured_root_ordinal
  edge_or_decision_ordinal
  mechanism_key?
  family_and_parameter_canonical_bytes
}
```

只有先按§6.2证明等价并merge的frontier state可共享state bytes；非等价state即使对同root提出同一种decision也不得dedup。
category按上列ordinal升序；`legality_forced=true`排在false前。三个optional evidence只有从current frontier exact proof得到时才
present，present排在absent前，数值按descending比较（canonical bytes中编码为`u64be(~value)`），unknown固定在最后且不能用
estimate填充。其余字段按canonical structured ordinal和length-prefixed typed bytes升序；optional字段有显式presence bit。
ready queue、budget分配和dedup只比较完整versioned bytes，同bytes work合并，digest不决定相等/顺序。并行生成或provider完成
只能把work放入该queue，不能按discovery/future completion顺序抢占hard cap。改变category/evidence/comparator必须升级key version。

task scheduling 使用 deterministic resource-aware list scheduling。默认 tie-break 来自稳定 IR order；只有 ready set 中
resource/reuse signature 不同且仍有预算时，才分裂有限 alternative。绝不枚举 `N!` topological orders。

### 6.4 两级 static-memory packing oracle

packing oracle服务两个不同抽象层，二者不能混用：

1. **frontier bound**：在instruction IR尚未完整物化时，从live physical version、tile domain、encoding extent、
   required temporary和已证明的同时存活关系计算`SoundLowerBound | Unknown`。只有sound bound超过可用arena时才能剪枝；
   sound partial bound即使尚未包含未来positive temporary仍可安全剪枝，因为它只会低估；只有size/必然存在/同时存活关系
   未证明的representative或heuristic estimate才只能排序。
2. **materialized packing envelope**：完整rank candidate已经lower成unplaced instruction memref后，从当前clone重算
   `LifetimeDemand`、pairwise conflict和absolute alignment，形成shared `MemoryPlanning`的owner-independent static packing problem。
   fixed-capacity primitive仍是唯一capacity legality owner；本stage只组合它的typed query结果，不能直接调用third-party类型，
   也不能从semantic/op名称补demand。

materialized query返回纯analysis结果，概念合同为：

```text
PackingEnvelope {
  legality: Feasible | ProvenInfeasible | ResourceExhausted | Invalid
  optimality: NotRun | ProvenOptimal | Bounded
  lower_bound_bytes?
  upper_bound_bytes?          // validated best placement相对arena.begin的high-water
  best_placements?            // 返回前已从canonical identity remap为当前call的demand index
  optimality_gap_bytes?
  lower_bound_proof?          // tagged trivial-zero、individual、clique或fixed-capacity cut
  fixed_capacity_queries
  search_nodes
  budget_reason?
}
```

`legality`和`optimality`正交：完整硬件arena上的`ProvenInfeasible`才拒绝candidate；若搜索已有一份通过独立validator的
placement，则minimum-height refinement耗尽只能得到`Feasible + Bounded`，不能变成`packing_search_exhausted`。
`ResourceExhausted`且没有任何合法placement时才是compiler resource failure。`Invalid`继续fail closed。上下界统一使用
`max(offset + size) - arena.begin`的byte span；absolute alignment仍在每次fixed-capacity query中以absolute address验证。

typed result只允许以下组合，不能用默认0补不存在的事实：

- `Invalid`：`NotRun`，lower/upper/placement/gap/proof等可选facts均为空；
- full-arena `ProvenInfeasible`或无incumbent的`ResourceExhausted`：`NotRun`，upper/placement/gap为空；只有确有
  proof时lower可存在；
- `Feasible + NotRun`：validated incumbent与upper必有，lower可选，gap为空；
- `Feasible + Bounded`：validated incumbent、lower、upper、gap和stop reason必有，且
  `0 <= lower <= upper`、`gap = upper - lower`；
- `Feasible + ProvenOptimal`：validated incumbent、lower和upper必有且相等，gap为0。

任何存在的lower都必须有bound相等的tagged proof，不能让数值与proof分别更新。任一非空placement都必须通过同一validator。
empty-demand problem不是zero-byte demand：它的empty placement天然合法；若只跑
legality则返回`Feasible + NotRun`及upper 0，若请求objective则返回`Feasible + ProvenOptimal`、lower/upper/gap均为0，且不发起
追加capacity query；lower由`TrivialZero` proof支持。

lower bound按由弱到强、始终可验证的方式建立：

- 每个demand从`arena.begin`出发的最早absolute-aligned end；
- adapter已验证activity clique中`sum(size)`的最大值；任何`ConflictClique` proof成员必须在原conflict graph中两两有edge；
- fixed-capacity query在某个span完整证明不可行后，最小可行span的lower bound提升到该span的下一byte。

多个proof给出同一bound时按tag和stable identities的固定总序选择primary proof，不能依赖container或并行发现顺序。

`LowerBoundProof`是唯一proof事实，不另存一份可能失配的witness：`TrivialZero`支持empty或通用非负下界；
`SingleDemand`和`ConflictClique`携带stable demand
identity及各自`witness_bound_bytes`，可由06按需投影为pressure view；`ProvenInfeasibleCut`只携带problem digest和完整证明的
capacity cut，不伪造demand集合。这不是IIS接口；若capacity cut把全局lower bound推得高于clique bound，clique proof仍只
声明自己的bound，不能冒充最终lower bound的完整解释；返回的primary proof切换为capacity cut且pressure view为空，
不用朴素逐buffer重求解minimal unsat core。
首版也不另求NP-hard最优edge-clique cover或maximum-weight clique；更强lower-bound实现只有保持同一proof validator、
三态capacity语义和global optimization budget时才可替换当前bound producer。

minimum-height refinement不恢复MiniMalloc的一体化minimize模式，而是组合同一三态fixed-capacity primitive：

1. 先在完整硬件arena运行现有legality query，取得`ProvenInfeasible`，或取得一份validated incumbent并以实际high-water
   建立upper bound；first-fit fallback若成功只提供合法upper bound，不提供不可行证明。
2. 若lower bound等于upper bound，直接以`lower-bound proof + validated placement`证明最优，不再调用solver。
3. 否则在整数byte span区间`[lower, upper)`内按稳定policy选择midpoint做byte-capacity binary refinement；不默认先查询通常最难的lower-bound
   tight point。`Feasible`用返回placement的实际high-water降低upper bound，
   `ProvenInfeasible`用checked `probe + 1`提高lower bound，`ResourceExhausted`不移动任何bound。对同一probe可按几何增长的node slice重试，
   但所有retry和candidate共享本轮optimization fuel。
4. 上下界相等时返回`ProvenOptimal`；query/node/candidate预算结束时返回`Bounded`和当前最优validated placement。
   每个结果离开oracle前仍运行shared `MemoryPlanning` placement validator，随后由09/12运行owner-specific range/resource gate。

搜索质量预算与capacity legality预算分开。baseline及每个已进入exact gate的candidate先获得完整arena legality allowance；
minimum-height只消费candidate-selection独立的global deterministic node/query fuel，不设置会改变选择语义的短
wall-clock timeout。refinement开始前从**finalized但unplaced**的current instruction IR构造：

```text
CanonicalShortlistWorkKey {
  schema_version
  canonical_unplaced_instruction_ir_bytes
  prepared_packing_problem_digest
  target_capability_context_digest
  deterministic_work_policy_digest
}
```

IR bytes按typed op/type/attr、root/view ordinal、SSA use-def/effect/completion canonical序列化，但明确排除accepted offset、
incumbent placement、probe、retry、solver/cache trace和wall timing。key使用length-prefixed full bytes做相等/排序，digest只作加速；
它只服务本次transformation的shortlist dedup与fuel order，不是final candidate/artifact signature。best placement apply并fresh
offset-dependent gates后，才按§7生成`canonical_candidate_signature`。

共享fuel按`CanonicalShortlistWorkKey`、probe span和retry level形成稳定round-robin work order，
不能按并行future完成顺序先到先得；相同signature先canonical dedup，仍需保留的同key query在调度前coalesce成一个有稳定epoch的
work item。baseline-ready后的optimization deadline按6.6丢弃optimized states并返回reserved baseline；driver/process cancellation
则中止整个transaction且不返回artifact。

prepared problem digest绑定完整硬件arena begin/end、每个canonical demand的stable identity/size/alignment、规范化conflict、
fixed/reserved constraints和adapter/policy version；probe span不改prepared digest，只作为query key第二部分。同一次transformation
可按`(prepared problem digest, queried span)`缓存validated feasible placement和完整证明的infeasible cut；placement在cache中
只按stable/canonical identity保存，命中后remap到当前demand index并重新validate。`ResourceExhausted`只能累计telemetry，不能
缓存为query truth。每次实际solver invocation（包括同span retry）计一次query并累计全部重复DFS nodes；cache hit不消耗solver
query/node但单独计数。cache不跨pass、不写IR，且final winner必须从当前IR重建
digest并重新验证placement。首版不依赖solver warm-start hint；validated incumbent只作为upper bound，因为当前受管core
没有消费hint的合同。

不是所有passing candidate都做高度二分。完整arena gate后先用其它exact cost维度和packing interval维持Pareto frontier；
只有bound overlap仍可能改变dominance或最终tie-break的shortlist进入refinement。对SPM维度，只有
`A.upper_bound <= B.lower_bound`时才能仅凭packing证明A不差于B；区间重叠时保持二者，或继续查询。全rank SPM envelope按
`max(per-rank bound)`聚合。预算结束仍有重叠时，winner用其实际validated upper bound进入06 cost vector，再按稳定policy/
signature确定性选择，并把optimality gap作为telemetry而非artifact事实。

shared capacity primitive对owner-independent problem通用，但Q32.S首版objective只激活这里定义的per-rank SPM维度；DDR保持
whole-variant full-arena legality与accepted high-water。启用DDR refinement前必须先在本设计增加明确cost维度、all-rank聚合/
shortlist位置、budget和verification合同，不能由12或shared API自行扩展。

`LowerBoundProof`的可选pressure view只给上游搜索排序，不是allocator repair命令。planner通过当前clone的demand `RootRef`映射回
`PhysicalVersion`和已有tile/residency/buffering/order decision，优先探索能减小proof内demand size或缩短其conflict的已资格化
mechanism；它不能按buffer名识别模型角色、临时发明rewrite或直接修改已提交IR。降低SPM同时增加DDR movement的方案仍保留
为普通Pareto tradeoff，不把“更低high-water”硬编码成无条件更优。

该`RootRef -> PhysicalVersion/decision`关系只来自本次candidate materializer的invocation-local `IRMapping`和canonical
decision key；不得以`Value*`/`Operation*`或raw demand index缓存，不得跨clone、rewrite或analysis invalidation继续使用，也不
进入下游gate/artifact。任一neighbor materialize后从新current IR和其本次mapping重建关系；无法稳定映射时proof仍可用于
capacity诊断，但不产生candidate priority。

### 6.5 Hard legality gates

cheap constraints 只能提前剪枝，不能替代 materialized exact gate。per-rank gate为：

1. source semantic、numeric/equivalence、shape 和 dtype；
2. implementation family、encoding、tail 和 padding；
3. transfer relation、descriptor cover、offset/range/alignment；
4. whole-rank SPM alias/liveness/peak；
5. local provider completion、event join、buffer reuse和terminal drain；
6. per-rank instruction geometry、narrowing及可独立证明的DDR view/range。

只有完整all-rank variant才能继续验证：

7. whole-variant DDR demand/range/package；
8. all-rank collective/transport matching和shared resource closure；
9. versioned target ABI、artifact和package preflight。

任何未知能力或不完整 proof 均拒绝该优化 choice，而不是降低 verifier。baseline 若也失败，才报告 workload/target
不支持，并给出最早失败的 typed diagnostic。
capacity refinement只替换placement也会使所有读取offset/address/range、receiver local offset或placed compatibility claims的
旧结果失效；final gate必须在best placement apply后从current clone重跑，不能把“storage graph未变”当作复用理由。

### 6.6 Hard budgets 和 fallback

06唯一拥有下面的预算schema；planner、production driver、Transform adapter和qualification harness只能逐字段消费它，不能
各自定义“差不多”的timeout、默认值或私有budget。所有数值字段都是checked unsigned 64-bit integer，单位是字段名声明的
确定性work/count/bytes，不是host cycle或wall time：

```text
DeterministicWorkPolicyV1 {
  schema_version = 1

  island_ops_cap
  island_values_cap
  island_relation_nodes_cap
  family_instantiations_per_root_cap
  optimized_generated_states_cap
  optimized_frontier_insertions_cap
  optimized_mechanism_applications_cap
  optimized_mechanism_rewrite_attempts_cap
  implementation_provider_queries_cap
  implementation_provider_work_cap
  physical_encoding_provider_queries_cap
  physical_encoding_provider_work_cap
  transfer_route_provider_queries_cap
  transfer_route_provider_work_cap
  communication_provider_queries_cap
  communication_provider_work_cap
  communication_skeleton_nodes_cap
  communication_skeleton_edges_cap
  constraint_propagation_work_cap

  relation_rewrite_steps_cap
  relation_expression_nodes_cap
  relation_pieces_cap
  relation_box_fragments_cap
  relation_division_depth_cap
  dependent_region_fragments_cap
  descriptor_segments_per_transfer_cap
  descriptor_split_work_cap
  hyperedge_conflict_splits_cap

  optimized_tile_refinements_cap
  optimized_ready_set_alternatives_cap
  optimized_beam_width_cap
  optimized_exact_top_k_cap
  optimized_finalized_rank_candidates_cap
  optimized_all_rank_combinations_cap
  optimized_all_rank_join_nodes_cap
  optimized_all_rank_join_frontier_cap
  optimized_materialization_work_cap
  optimized_exact_gate_work_cap

  packing_legality_queries_per_candidate_cap
  packing_legality_nodes_per_query_cap
  packing_objective_queries_cap
  packing_objective_nodes_cap
  packing_result_cache_bytes_cap
  analysis_cache_bytes_cap

  baseline_materialization_work_cap
  baseline_tile_refinements_cap
  baseline_safe_tile_rule_work_cap
  baseline_exact_gate_work_cap
  baseline_packing_legality_queries_cap
  baseline_packing_legality_nodes_cap
  baseline_analysis_bytes_cap
}
```

relation五字段一对一构造§3.2的`RelationWorkPolicy`；其它字段也只在名字所指的phase消费。`*_work_cap`按该phase
canonical traversal中、rewrite/materialize/gate发生前累计的versioned work unit计数；新增或改变一个work unit的含义必须升级
schema。optimized generated state、frontier insertion、mechanism application/rewrite attempt、四类provider query/work和每次constraint
domain收紧尝试均在创建state、
尝试rewrite或发出query**之前**checked计数；provider返回的`work_summary`必须等于对应provider work counter增量。这样beam/
top-K只限制survivor而不会掩盖进入frontier前的无界生成，communication parameter/segmentation也受communication provider
query/work与family-instantiation双重上界；logical skeleton的每个node和edge分别在插入前消费对应cap，任一耗尽均不保留
partial skeleton。baseline safe-tile rule的每个canonical solver/proof work unit在执行前消费
`baseline_safe_tile_rule_work_cap`。lazy coordinator在enqueue每个partial或complete join node前消费
`optimized_all_rank_join_nodes_cap`，在插入surviving partial frontier前消费`optimized_all_rank_join_frontier_cap`；
`optimized_all_rank_combinations_cap`只计进入whole-variant exact gate的完整optimized组合，因此首个optimized完整组合前也有硬上界。
`packing_*_nodes_cap`累计MiniMalloc/shared fixed-capacity primitive报告的实际solver node；两个cache byte cap都按versioned logical
charge计算：canonical serialized key bytes + canonical serialized value bytes + schema固定的per-entry/per-table overhead常数，并在
insert前checked。host object layout、container capacity growth、allocator metadata或`malloc_usable_size`不得参与budget或选择；真实
allocation failure仍单独返回`ResourceExhausted`。legality query/node allowance按candidate预留，
objective query/node/cache allowance由shortlist全局共享，二者不得互借。baseline七字段只服务唯一reserved baseline，不能被
optimized work消费；普通allocation failure仍是`ResourceExhausted`，不靠虚构host-memory estimate冒充deterministic cap。

island/relation/family/provider/constraint/descriptor/hyperedge/skeleton node/skeleton edge等required结构字段、两个packing-legality字段和七个baseline字段
必须大于零，否则policy无效。所有`optimized_*`字段以及packing-objective query/node字段允许为零，零表示不生成相应optional
work并直接保留baseline；
`packing_result_cache_bytes_cap=0`或`analysis_cache_bytes_cap=0`表示禁用对应cache而不改变语义。任何计数加法溢出都按本policy
`ResourceExhausted`，不wrap或saturate。为避免含糊，validator还要求
`optimized_exact_top_k_cap <= optimized_finalized_rank_candidates_cap`；任一侧为零时不形成optimized shortlist。每个可进入
objective packing的candidate必须已经先取得独立非零legality allowance。

production named pipeline从compiler invocation选择一个registered、immutable policy row；没有显式选择时使用仓库注册表中唯一
标记为production-default的完整row，而不是逐字段默认。focused test可直接构造完整row；Q32.T必须显式给出完整attr，不能使用
production default。policy不进入IR或bundle identity，但其digest进入invocation telemetry、cache/work key和qualification
observation。canonical bytes固定为domain separator
`"wafer.deterministic-work-policy\0"`、`u16be(schema_version)`，再按上面声明顺序连接全部`u64be`字段；digest为这些bytes的
SHA-256。registry拒绝unknown version、缺字段、额外字段、非法零值、关系不满足或非canonical encoding。

`RankLocalCandidateOrderV1`是无配置的closed singleton policy，不带字段、权重或hidden default；canonical bytes固定为
`"wafer.rank-local-candidate-order\0" || u16be(1)`。其全序冻结为：先丢弃exact gate失败者并保留reserved baseline；仅在相同
`PlacedRankCompatibilityClaimsV1`完整bytes内做strict exact Pareto；若baseline vector完整，任何含unknown dimension的optimized
candidate排在所有complete survivor之后，若baseline自身在任一rank-local-eligible component Unknown/Unavailable则直接选择baseline
并报告incomparable；其余survivor依次比较`RankLocalStaticMetricEvidenceV1`中按registry canonical key展开的全部eligible Known
projection、length-prefixed `PlacedRankCompatibilityClaimsV1` bytes、`baseline_tie_bit`（baseline为0）和完整
`RankLocalPlacedPayloadSignatureV1` bytes。metric direction/short-span penalty严格复用§7 registry，但不生成whole-variant aggregate或
immutable scalar。每一步都是lexicographic ascending；
digest只加速比较，不能代替bytes。改变任一步或字段顺序必须新增policy version，不能给同名attr加可选参数。

baseline先于optimized exploration执行并使用上述独立materialization/gate/packing/cache allowance；optimization budget不得淘汰
它，但baseline allowance本身也有确定性硬上界。若连baseline proof都在该allowance内无法完成，返回明确
`compiler_resource_exhausted`，不能标成target/workload illegal。baseline gate完成后，独立optimization预算无条件生效，
包括“已有候选仍在并行评估”的情况。生产可复现选择只由这些deterministic caps决定。policy外的wall机制分为两种且都不进入
digest或candidate访问顺序：optimization deadline只能在baseline完成全部reserved exact gates后arm；触发时丢弃所有optimized
survivor、返回reserved baseline并报告`optimization_deadline_exceeded`，不按恰好完成的并行任务选择。driver/process
cancellation可以在任意时点触发，必须abort整个clone/bundle transaction并返回`Cancelled`、无artifact，不能伪造尚未完成的
baseline。跨线程determinism只在相同policy且两类外部信号均未触发时验证。
optimization预算耗尽时：

- 停止新增 optimized states；
- 完成已 materialize 的 bounded exact gates；
- 若 baseline 合法则返回 baseline，并发出结构化 budget diagnostic；
- 若 baseline 不合法则返回真实 legality failure，不伪装成 timeout。

planner诊断只使用一个versioned schema，其它文档与harness引用该类型而不复制字段名：

```text
PhysicalDataflowPhaseV1 =
    RelationNormalization | IslandFormation | CapabilityQuery
  | RegionPropagation | ConstraintPropagation | MechanismExpansion
  | Materialization | PerRankExactGates | RankCoordination
  | WholeVariantExactGates | SelectionCommit

PhaseBudgetStatusV1 = NotEntered | Completed | Exhausted | Cancelled
PhysicalDataflowStopReasonV1 =
    Completed | OptimizationBudgetExhausted | BaselineAllowanceExhausted
  | OptimizationDeadlineFallback | DriverCancelled | NoLegalCandidate
  | InvalidRequestOrInvariant
BaselineStatusV1 = NotStarted | Ready | Unsupported | ResourceExhausted | Invalid | Cancelled
BaselineFailureReasonV1 =
    UnsupportedRepresentation | UnsupportedCapability | InfeasibleConstraints
  | DeterministicAllowanceExhausted | HostResourceExhausted
  | InvalidRequest | InvalidRegistry | InternalInvariant
PackingStopReasonV1 =
    NotRun | LegalityOnly | Exact | ObjectiveBudgetExhausted | ResourceExhausted

PhysicalDataflowTelemetryV1 {
  schema_version: U16 = 1
  input_payload_digest: Digest32
  target_capability_context_digest: Digest32
  deterministic_work_policy_digest: Digest32
  phases: Sequence<Record(PhaseTelemetryV1)>
  provider_queries_by_kind_status_reason: Sequence<Record(ProviderQueryTelemetryV1)>
  packing: Record(PackingTelemetryV1)
  baseline_status_and_failure: Record(BaselineTelemetryV1)
  selected_candidate_signature: Optional<Record(CanonicalCandidateSignatureV1)>
  selected_exact_cost_vector: Optional<Record(ExactStaticCostVectorV1)>
  stop_reason: ClosedEnum(PhysicalDataflowStopReasonV1)
}

PhaseTelemetryV1 {
  schema_version: U16 = 1
  phase: ClosedEnum(PhysicalDataflowPhaseV1)
  input_states: U64
  generated_states: U64
  domain_reductions: U64
  constraint_pruned: U64
  canonical_merged: U64
  dominance_pruned: U64
  materialized: U64
  exact_gate_attempts: U64
  passing: U64
  peak_frontier: U64
  deterministic_work_units: U64
  wall_ns_diagnostic_only: U64
  budget_status: ClosedEnum(PhaseBudgetStatusV1)
}

ProviderQueryTelemetryV1 {
  schema_version: U16 = 1
  provider_kind: ClosedEnum(ProviderKindV1)
  status: ClosedEnum(ProviderQueryStatus)
  unsupported_reason: Optional<ClosedEnum(ProviderUnsupportedReason)>
  query_count: U64
  deterministic_work_units: U64
}

PackingTelemetryV1 {
  schema_version: U16 = 1
  legality_queries: U64
  objective_queries: U64
  cache_hits: U64
  solver_nodes: U64
  validated_lower: Optional<U64>
  validated_upper: Optional<U64>
  selected_high_water: Optional<U64>
  gap: Optional<U64>
  stop_reason: ClosedEnum(PackingStopReasonV1)
}

BaselineTelemetryV1 {
  schema_version: U16 = 1
  status: ClosedEnum(BaselineStatusV1)
  failure_reason: Optional<ClosedEnum(BaselineFailureReasonV1)>
}
```

`phases`按`PhysicalDataflowPhaseV1` ordinal all-and-only恰有11行；未进入phase的全部counter为0且
`budget_status=NotEntered`，进入后的不适用counter写typed zero。provider表按
`(ProviderKindV1, ProviderQueryStatus, unsupported_reason)`排序并all-and-only覆盖四个provider的
`Available/ResourceExhausted/Invalid`无reason行和`Unsupported`的三个typed reason行，即固定24行；只有Unsupported行reason
present，其它status必须absent，零次组合保留typed zero row。所有counter checked，phase/provider重复、缺项、额外row或sum与
planner work counter不闭合均拒绝。

packing的lower/upper/gap三者必须同时present或同时absent；present时`lower <= upper`且`gap = upper - lower`使用checked
arithmetic。`NotRun`要求全部counter和四个optional均为zero/absent；`Exact`要求interval present且lower=upper；其它stop可保留
此前已验证的interval，但`ResourceExhausted`不得凭未完成probe移动bound。`selected_high_water`只表示最终实际validated
placement，不得用upper代填。baseline `Ready`要求failure absent；`Unsupported/ResourceExhausted/Invalid`要求与status匹配的
typed failure present：Unsupported只接受前三个reason，ResourceExhausted只接受
`DeterministicAllowanceExhausted | HostResourceExhausted`，Invalid只接受后三个reason；`NotStarted/Cancelled`要求failure absent。

outer stop中`Completed | OptimizationBudgetExhausted | OptimizationDeadlineFallback`要求baseline Ready、selected signature与
vector恰好present并从最终selected bound IR fresh重算，且packing `selected_high_water` present；若packing stop为Exact，
该值还必须等于相同的lower/upper。`BaselineAllowanceExhausted`要求baseline ResourceExhausted，
`NoLegalCandidate`要求baseline Unsupported，`InvalidRequestOrInvariant`要求baseline Invalid，后三者selected facts恰好absent。
全部失败stop及`DriverCancelled`的packing `selected_high_water`必须absent；`DriverCancelled`允许取消前已形成任一baseline
status，但selected facts必须absent且整个transaction不发布artifact。只有一个
selected fact present、失败stop携带selected fact或成功stop缺fact都是schema failure。所有record使用§6.2统一canonical codec；
wall只作观察，不能影响未取消执行的winner。该record是invocation-local diagnostic，可由qualification保存，但不是IR、cache
key、bundle或下游协议。

### 6.7 原子提交

candidate materialization、07 required payload normalization、qualified optional cleanup、bufferization、physical-memory
replanning和recost全部在隔离clone上进行。required normalization始终开启；generic canonicalization只清理已证明不影响
postcondition的冗余，不能决定candidate合法性。
一个rank可按placed compatibility claims保留有限surviving frontier；all-rank coordinator先传播shared constraints，再用
baseline-first lazy/factorized join访问hard-capped完整组合，绝不先构造rank-frontier笛卡尔积。每个完整variant重放
whole-variant DDR resource gate，再执行post-memory transport binding并验证
`AcceptedVariantTransportSignatureV1`，最后做package/ABI pure preflight。只有winning variant的全部ranks同时通过后，
才由tasks/14 registry从各rank final bound instruction
rows派生canonical `RequiredCapabilitySet`、验证all-and-only union/digest，并与source program替换及ExecutableBundle一起原子提交。
Q22.L target conversion从实际发射的TargetCall rows重算rank-local投影，Q17 readback并携带global set，Q18 schema-v4只序列化该set；
这些不是planner choice，也不能反向参与搜索。

winning rank clone提交前必须从其final instruction IR fresh重建packing problem。若此后IR未变化且规范化problem digest与
oracle输入完全一致，可以复用best placement，但仍须再次运行owner-independent validator和09/12 owner range/resource gate；
digest不一致则必须重新运行完整arena legality query，不能沿用stale incumbent。只有这一步接受的offset进入IR。

局部 tile、单 op、单 rank、代表 rank 或部分 offsets 永不提交。rejected clone 不得泄漏 memref type、offset、event 或
resource mutation。

## 7. Cost、排序和板端校准

legality 与 cost 完全分离。板端校准前使用可从 selected IR 精确统计的 Pareto vector：

```text
(
  peak_spm_bytes,
  ddr_read_bytes,
  ddr_write_bytes,
  spm_movement_bytes,
  transport_bytes_by_route,
  transport_message_counts_by_route,
  compute_class_counts,
  compute_logical_work_by_class,
  transfer_descriptor_counts_by_engine,
  inner_contiguous_bytes_distribution_by_engine,
  route_temporary_peak_bytes,
  engine_command_counts,
  instruction_issue_counts,
  fence_wait_event_counts,
  padding_work,
  immutable_payload_bytes
)
```

telemetry和qualification引用的完整variant对象固定为：

```text
ExactStaticMetricComponentV1 {
  metric_id: u32
  status: Known | Unknown
  scalar_value?: u64
  ordered_vector_value?: u64[]
}

ExactStaticCostVectorV1 {
  schema_version = 1
  static_metric_registry_version_and_digest
  ordered_components: ExactStaticMetricComponentV1[]
}

BoundRankInstructionSignatureV1 {
  logical_rank: u32
  canonical_bound_instruction_ir_bytes: bytes
}

CanonicalCandidateSignatureV1 {
  schema_version = 1
  ordered_rank_signatures: BoundRankInstructionSignatureV1[]
}
```

metric registry为每个`metric_id`固定`ScalarU64 | OrderedU64Vector`值类型及vector的完整key/bucket顺序；
`Known`恰好存在一个与registry匹配的value，`Unknown`两个value都absent，wrong type、两值同时present、
vector缺项/多项均拒绝。components按metric ID的registry canonical order all-and-only存在，已证明不出现的
row是Known zero而非absent。candidate signature按logical rank严格递增且all-and-only覆盖execution mesh。
两类record的canonical bytes使用u16 schema、u32 length prefix、big-endian integer及分别的domain separator
`"wafer.exact-static-cost-vector\0"`、`"wafer.canonical-candidate-signature\0"`；unknown/missing/duplicate rank或metric、
registry digest不匹配及非final bound IR来源均是validation failure。

map/histogram维度由唯一versioned static-metric registry解释。每个row除metric key/unit/direction外还固定
`all_rank_aggregation = CheckedSum | CheckedMax | CanonicalBucketSum | IdentitySetUnionThenSize`与
`rank_local_dominance_eligible`；registry固定route key、engine key、compute-class key和inner-span bucket阈值的canonical顺序。
只有aggregation对每rankcomponentwise monotone且不依赖跨rank identity/join时eligibility才可为true。immutable payload在rank
阶段保留canonical `(PayloadIdentity, bytes)` set，row固定`IdentitySetUnionThenSize`且eligibility=false；只有完整variant去重后才
产生`immutable_payload_bytes`标量，不能用某rank局部标量剪枝。新增key、改变aggregation/eligibility或bucket必须升级policy
version，不能依赖enum registration或容器迭代顺序。
某个已注册component在当前IR中经完整扫描证明不存在时其值是known zero；无法扫描、provider未给出exact metric或算术溢出时是
typed unknown，二者不能合并。

Pareto比较中的所有component都规范成“smaller is better”。raw
`inner_contiguous_bytes_distribution_by_engine`先按fixed bucket聚合，再转成每个engine的cumulative short-span penalty（inner span
低于每个阈值的`command_count`与`transferred_bytes`二元组，按阈值升序展开）；只比较该penalty，不对raw histogram做方向不明的
componentwise dominance。raw histogram继续作为exact diagnostic保留。

`transport_*`同时覆盖NoC/DTE/collective和其它已注册transport route；08返回的physical read/write、descriptor、inner-span、
temporary、engine command和event metrics按上述同名维度聚合，09/12返回peak与DDR/SPM事实，13返回transport
bytes/message事实。unknown dimension 不得伪造为 0。先做 Pareto dominance，再用稳定 profile policy 作 deterministic
beam/tie-break；文档和diagnostic必须同时保留向量，不能把未校准 scalar estimate 宣称为硬件时间。

板端校准前唯一选择policy固定为`UncalibratedStaticOrderV1`。all-rank聚合中peak与temporary peak取checked `max`；traffic、
work、descriptor、command、issue与event取checked `sum`；histogram按canonical bucket逐项sum；immutable payload按typed payload
identity去重后sum。所有加法（包括tuple首项的DDR read+write）都使用checked wide integer，不做saturation/wrap；overflow形成
typed incomplete metric。exact Pareto之后的完整variant按下列tuple升序确定唯一winner：

```text
(
  ddr_read_bytes + ddr_write_bytes,
  transport_total_bytes,
  transport_total_messages,
  spm_movement_bytes,
  padding_work,
  total_compute_logical_work,
  transfer_descriptor_count,
  engine_command_count,
  instruction_issue_count,
  fence_wait_event_count,
  peak_spm_bytes,
  route_temporary_peak_bytes,
  immutable_payload_bytes,
  canonical route/engine/compute-class sub-vectors,
  baseline_tie_bit,
  canonical_candidate_signature
)
```

其中canonical sub-vectors按registry key升序、字段顺序固定展开：每个route为`(bytes, messages)`；每个engine为
`(descriptor_count, cumulative_short_span_penalties..., command_count, issue_count)`；每个compute class为
`(invocation_count, logical_work)`。数组采用versioned、length-prefixed canonical bytes，不能省略known-zero row或按出现顺序压缩。

rank-local placed但未做all-rank binding的research path使用独立名字，不能借用production final signature/vector：

```text
RankLocalVerificationScopeV1 = PlacedRankUnbound
RankLocalMetricStatusV1 = Known | Unknown | Unavailable
RankLocalUnavailableReasonV1 = WholeVariantOnly | AllRankDDRPlanningNotRun

RankLocalPlacedPayloadSignatureV1 {
  schema_version: U16 = 1
  scope: ClosedEnum(RankLocalVerificationScopeV1) = PlacedRankUnbound
  canonical_placed_unbound_instruction_ir_bytes: Bytes
  placed_compatibility_claims: Record(PlacedRankCompatibilityClaimsV1)
}

RankLocalMetricComponentV1 {
  metric_id: U32
  status: ClosedEnum(RankLocalMetricStatusV1)
  scalar_value: Optional<U64>
  ordered_vector_value: Optional<Sequence<U64, element_wire=U64>>
  unavailable_reason: Optional<ClosedEnum(RankLocalUnavailableReasonV1)>
}

ImmutablePayloadEvidenceV1 {
  canonical_payload_identity_bytes: Bytes
  byte_count: U64
}

RankLocalStaticMetricEvidenceV1 {
  schema_version: U16 = 1
  scope: ClosedEnum(RankLocalVerificationScopeV1) = PlacedRankUnbound
  static_metric_registry_version: U16
  static_metric_registry_digest: Digest32
  ordered_components: Sequence<Record(RankLocalMetricComponentV1)>
  immutable_payload_identity_and_bytes_set:
      Sequence<Record(ImmutablePayloadEvidenceV1)>  // canonical set keyed by full identity bytes
}

RankLocalScopedScalarEvidenceV1 {
  status: ClosedEnum(RankLocalMetricStatusV1)
  scalar_value: Optional<U64>
  unavailable_reason: Optional<ClosedEnum(RankLocalUnavailableReasonV1)>
}
```

signature从finalized、SPM-placed、unbound current rank IR fresh结构序列化，内嵌同一份typed DTE unbound claims而不另存第二份
claims bytes；任何IR/offset/claim变化都使整个signature失效。component按metric registry canonical ordinal all-and-only排序。
`Known`要求unavailable reason absent，并按registry row的`ScalarU64 | OrderedU64Vector`值类型恰有一个匹配value；`Unknown`要求
两个value和reason全部absent；`Unavailable`要求两个value absent且reason恰好present。vector长度/各位置含义必须与同一registry
row匹配，不能因rank-local scope把vector压成scalar；只有registry明确标记`rank_local_dominance_eligible`且Known的projection可
比较。immutable payload字段刻意沿用§6.2唯一`Sequence` wire tag，而不是新增`SortedSet` type-tag；其set语义冻结为按完整
`canonical_payload_identity_bytes`升序、identity恰好唯一、byte count checked，非该顺序或重复identity均拒绝。它保留identity
set而不压成局部标量。
`RankLocalScopedScalarEvidenceV1`复用同一presence规则：Known恰有value、Unknown二者皆absent、Unavailable只带reason；Q32.T的
`whole_variant_ddr_high_water_bytes`字段在本scope必须固定为
`Unavailable(AllRankDDRPlanningNotRun)`。transport binding及model/board资格不是metric component，分别由下文typed
verification disposition表达；inspect不得把它们或DDR high-water用rank-local heuristic冒充。上述record只供Q32.T/diagnostic，
不进入production IR、bundle、package或与final winner比较。
它们使用§6.2统一canonical codec；scope、metric status和unavailable reason按上述声明顺序固定ordinal，unknown/missing/extra
component、registry mismatch、错误presence或重复payload identity均拒绝。

每个rank的`canonical_candidate_signature`从**finalized、placed且已完成all-rank transport binding的当前rank instruction IR** fresh构造：按canonical structured
traversal序列化typed op kind、type/attrs、root/view canonical ordinal、SSA use-def/effect/completion和实际指令字段；whole-variant
signature再按logical rank把这些length-prefixed bytes连接。二者都排除SSA打印名、location、pointer、symbol展示名、discovery order、
beam trace与telemetry。最终tie-break比较完整canonical bytes；digest只用于cache/diagnostic，不能单独用可能碰撞的hash决定winner。

该tuple是compiler policy而非hardware time；`baseline_tie_bit`固定reserved baseline为0、其它candidate为1，只在全部静态
维度相同时偏向baseline。unknown维度不
参与dominance，也不能按0；baseline vector完整时，不完整optimized candidate不能获胜；baseline自身不完整时强制返回baseline
并报告`uncalibrated_incomparable`。Q32当前static支持面的normal winner与baseline必须全部policy dimensions已知。Q9未来可
注册calibrated policy，但只能重排已经通过exact legality的候选。

`peak_spm_bytes`永远是winner当前validated placement的实际high-water，不是lower bound或二分probe容量。packing
lower/upper/gap只在candidate-local interval dominance、追加查询优先级和telemetry中存在；gap未闭合不把该维度伪造为
unknown或0。更低SPM但更多DDR/SPM movement、descriptor或issue的方案仍是普通tradeoff，只有完整vector满足严格关系时才做
dominance。

只有 IR/event 能证明 overlap，且对应 engine/queue/resource 能力已资格化时，resource DAG scheduler 才允许重叠；否则
保守串行。板端 Q9 只用 PMU/带宽/latency 数据校准合法候选的排序权重和 overlap 模型，不改变 semantic、numeric、
descriptor、capacity 或 ABI legality。

对外发布性能结论时分三层：

- 静态 IR 事实：bytes、peak、descriptor、issue、event；
- CModel wall time：host functional execution 性能，不代表板端；
- 板端实测：仅在配置、版本、PMU 和统计方法完整时报告。

## 8. Transform Dialect 与 Compiler-Wide Control Plane

```text
Pipeline position:
- Upstream artifact / IR: explicit rank-local post-legalization structured module or materialized rank-local candidate handle.
- Current stage responsibility: adapt Transform handles/params to the same Q32 rank-frontier/mechanism/materializer libraries and fresh diagnostics.
- Output artifact / IR: modified structured IR, one explicitly nonproduction rank-local candidate, or ephemeral Transform params; never an accepted execution artifact.
- Downstream consumer: wafer-opt research/debug composition, then all per-rank exact gates derivable from payload IR; binding-dependent and all-rank gates remain explicitly unverified.
- User-level driver / named pipeline: wafer-opt Transform interpreter only; wafer-compile production remains the named pipeline.
- Explicit non-goals: all-rank coordination/binding proof, production-winner comparison, alternate planner, solver state, package Transform IR or workload-specific schedule ops.
- Completion gate: adapter/library `RankLocalPlacedPayloadSignatureV1` equality under `RankLocalCandidateOrderV1`, handle/effect/failure/atomicity gates, explicit unverified marker and zero Transform facts downstream.
```

MLIR Transform Dialect 对本设计有帮助，但它只解决 orchestration、匹配、参数传递和可复现控制，不替代atomic mechanism、
联合solver或exact gate。全compiler按四层组织：

1. production default policy：固定named pipeline，只调用已资格化的required/fixed机制；
2. atomic typed mechanisms：普通C++ utility/pattern/interface，独立声明precondition、effect和proof obligation；
3. candidate-local policy/search：在隔离clone中组合mechanisms和target choices，失败不污染主IR；
4. optional Transform/外部tuner：在稳定handle、param和diagnostic边界上编排同一机制或调用粗粒度solver。

推荐结构：

```text
production named pipeline ─────────────┐
optional Transform / external tuner ───┼─> shared typed mechanisms
                                       │          |
                                       └─> physical-dataflow policy/search
                                                  |
                                                  v
                                      typed payload IR + exact gates
```

普通Transform interpreter只有单棵payload root，而production whole-card coordinator消费rank module vector、frontend program
facts与execution config。因此Q32.T只能是**rank-local、可选、非production driver**；在没有typed all-rank IR container前，
不得通过TransformState side table、wrapper IR或外部metadata伪造whole-card synthesis。最小op集合固定为：

1. `transform.wafer.optimize_structured`：消费singleton post-legalization、pre-physical-planning rank-local `builtin.module`
   handle及singleton `StructuredOptimizationPolicyV1` param，调用显式rank-local rewrite utility并返回fresh root handle与
   singleton `MechanismReportV1`；
2. `transform.wafer.materialize_physical_dataflow_candidate`：消费singleton pre-scheduling rank module，在clone上调用
   Q32同一rank frontier producer、materializer与payload可重算的per-rank exact gates，再用singleton required
   `TargetProfileId`、logical-rank、`RankLocalCandidateOrderV1`及`DeterministicWorkPolicyV1` params物化一个研究/调试candidate，
   返回fresh handle与singleton `CandidateRunReportV1`；
3. `transform.wafer.inspect_physical_dataflow_candidate`：只读placed-unbound candidate module，fresh重算
   `RankLocalPlacedPayloadSignatureV1`、SPM high-water和`RankLocalStaticMetricEvidenceV1`；whole-variant DDR high-water返回显式
   typed Unavailable，binding及model/board field保持typed Unverified，输出singleton `CandidateInspectionV1` param。

profile canonical attr必须唯一解析为tasks/14的`TargetProfileId`并按§9固定构造
`CompilerEmission + CompilerEmittable` context；Q32.T不接受model/board context param。budget每个字段一对一映射本节
`RankPlanningRequest::deterministic_work_policy`，extension不得另建default或schema。每个input handle/param映射和每个result
mapping都必须exact one；empty/multi-target由下述silenceable failure处理。

在当前pinned MLIR中`TransformParamTypeInterface`由param **SSA type**实现，TransformState再把该value映射到Attribute payload；
extension不能声称attr本身实现该interface。exact映射冻结为：

```text
!transform.wafer_structured_policy_v1       -> StructuredOptimizationPolicyV1Attr
!transform.wafer_target_profile_v1          -> TargetProfileIdAttr
!transform.wafer_logical_rank_v1            -> I64Attr (checked nonnegative)
!transform.wafer_rank_local_order_v1        -> RankLocalCandidateOrderV1Attr
!transform.wafer_work_policy_v1             -> DeterministicWorkPolicyV1Attr
!transform.wafer_mechanism_report_v1        -> MechanismReportV1Attr
!transform.wafer_candidate_run_report_v1    -> CandidateRunReportV1Attr
!transform.wafer_candidate_inspection_v1    -> CandidateInspectionV1Attr
```

每个type实现interface并在mapping时验证exact attr class/version/schema；generic DictionaryAttr、StringAttr和类型兼容fallback拒绝。
structured policy唯一schema为：

```text
StructuredOptimizationPolicyV1 {
  schema_version: U16 = 1
  enabled_mechanism_keys: Sequence<Record(MechanismKeyV1)>
}
```

mechanism keys必须sorted unique且适用于当前IR cut；unknown、duplicate或wrong-cut在mutation前拒绝。canonical bytes固定为domain
`"wafer.structured-optimization-policy\0"`、u16 schema和length-prefixed key sequence，无extension私有default。
三个result param的exact record由06唯一冻结：

```text
MechanismOutcomeCountV1 {
  outcome: ClosedEnum(MechanismInvocationOutcomeV1)
  count: U64
}

MechanismReportEntryV1 {
  mechanism_key: Record(MechanismKeyV1)
  outcome_counts: Sequence<Record(MechanismOutcomeCountV1)>
  invocation_count: U64
  rewrite_count: U64
  work_summary: Record(WorkSummaryV1)
}

MechanismReportV1 {
  schema_version: U16 = 1
  structured_optimization_policy_digest: Digest32
  entries: Sequence<Record(MechanismReportEntryV1)>
}

RankLocalCandidateStopReasonV1 = Completed | OptimizationBudgetFallback | UncalibratedIncomparable
RankLocalVerificationDispositionV1 = Unverified

CandidateRunReportV1 {
  schema_version: U16 = 1
  scope: ClosedEnum(RankLocalVerificationScopeV1) = PlacedRankUnbound
  all_rank_and_binding_verification: ClosedEnum(RankLocalVerificationDispositionV1) = Unverified
  model_and_board_qualification: ClosedEnum(RankLocalVerificationDispositionV1) = Unverified
  selected_is_reserved_baseline: Bool
  payload_signature: Record(RankLocalPlacedPayloadSignatureV1)
  static_metric_evidence: Record(RankLocalStaticMetricEvidenceV1)
  whole_variant_ddr_high_water_bytes: Record(RankLocalScopedScalarEvidenceV1)
  work_summary: Record(WorkSummaryV1)
  budget_stop: ClosedEnum(RankLocalCandidateStopReasonV1)
  packing: Record(PackingTelemetryV1)
}

CandidateInspectionV1 {
  schema_version: U16 = 1
  scope: ClosedEnum(RankLocalVerificationScopeV1) = PlacedRankUnbound
  all_rank_and_binding_verification: ClosedEnum(RankLocalVerificationDispositionV1) = Unverified
  model_and_board_qualification: ClosedEnum(RankLocalVerificationDispositionV1) = Unverified
  payload_signature: Record(RankLocalPlacedPayloadSignatureV1)
  static_metric_evidence: Record(RankLocalStaticMetricEvidenceV1)
  whole_variant_ddr_high_water_bytes: Record(RankLocalScopedScalarEvidenceV1)
}
```

`WorkSummaryV1`和`MechanismKeyV1`直接嵌入05 owner的canonical record。每个mechanism entry的outcome rows按enum ordinal
all-and-only恰有7行，`invocation_count`等于七行count checked sum；`rewrite_count`只统计Applied invocation实际commit的
rewrite，不能大于对应mechanism contract允许的rewrite count。entries按完整MechanismKey bytes排序且all-and-only覆盖本次policy
实际调用的required/fixed/cleanup key；重复、未调用key、missing key、qualified-set或structured-policy digest不匹配拒绝。

`CandidateRunReportV1`只在Transform op成功返回fresh candidate handle时产生，因此`budget_stop=Completed`、在reserved
baseline合法时为`OptimizationBudgetFallback`，或因baseline自身rank-local metric含Unknown/Unavailable而按§7返回baseline时为
`UncalibratedIncomparable`；baseline allowance耗尽、Invalid、Unsupported或driver cancellation按下文failure
合同不产生result mapping。payload signature已经内嵌唯一typed placed claims，report不得再保存第二份claims字段；scope和两项
verification disposition必须是上面的singleton值。signature、metric scope/registry必须逐字段一致，packing必须有
`selected_high_water`且等于metric registry中的SPM high-water Known scalar；budget fallback要求
`selected_is_reserved_baseline=true`，uncalibrated incomparable也要求它为true。run report和inspection的
`whole_variant_ddr_high_water_bytes`必须byte-identical且固定为`Unavailable(AllRankDDRPlanningNotRun)`。inspection不含work、budget或
packing字段，因为这些不能从payload IR fresh恢复；它只从current IR重建与run report byte-identical的signature及同registry
metric evidence，并按scope构造上述固定DDR证据。

上述record按声明顺序把field id固定为1..N并使用§6.2统一canonical codec；每个Optional都以上述
`Optional<U64>`、`Optional<Sequence<...>>`、`Optional<ClosedEnum(...)>`或`Optional<Record(...)>`声明的inner wire type编码，
不得按当前value猜inner tag。改变字段、presence、enum或nested type需要新版本。`MechanismReportV1Attr`、`CandidateRunReportV1Attr`和
`CandidateInspectionV1Attr`各自只能承载同名record的canonical bytes，generic dictionary/string attr、重复placed claims或
sentinel不可作为兼容输入。

candidate op只接受closed target profile、explicit logical rank、`RankLocalCandidateOrderV1`和versioned deterministic work
budget；并行度只是execution hint，不改变结果。local order先保留baseline并做per-rank exact Pareto，再按complete exact static
rank-local metric evidence、placed compatibility claims、baseline tie与`RankLocalPlacedPayloadSignatureV1`确定一个candidate；
eligible component含Unknown/Unavailable的optimized evidence不能胜
complete baseline。该顺序只为wafer-opt研究重放提供确定性，不是production policy或hardware time。

op不接受frontend program metadata、rank module vector、execution bundle、beam/frontier/placement或physical-version参数；依赖
frontend binding的per-rank gate不伪造输入而保持unverified。返回的`CandidateRunReportV1`显式标记
`all_rank_and_binding_verification=Unverified`及`model_and_board_qualification=Unverified`，并包含
`RankLocalPlacedPayloadSignatureV1`（其中内嵌唯一`PlacedRankCompatibilityClaimsV1`）、
`RankLocalStaticMetricEvidenceV1`、work summary、budget stop与typed `PackingTelemetryV1`；不重复保存claims。
payload不得进入package/RuntimeSession；不能与production winner比较或声称等价。inspect只重算IR-derived metrics，不得从
candidate IR猜search telemetry。

cut verifier只读payload IR，不靠marker attr：`optimize_structured`输入必须是singleton rank-local `builtin.module`的05
structured tensor envelope，含typed rank entry且无Wafer physical memref、tile/instr op、accepted offset或transport binding；
输出必须通过当前structured IR verifier。`materialize_physical_dataflow_candidate`输入还必须携带exact topology/execution mesh，
显式logical rank属于mesh，且零physical/instruction residue；输出必须无
Tensor/Linalg/search residue，是finalized、SPM-placed但unbound的rank instruction payload，并通过全部从payload可重算的per-rank
instruction/descriptor/SPM/local-completion/range gates。`inspect`只接受这份`PlacedRankUnbound` output cut；它不得把缺失的
whole-variant DDR plan或transport binding补成known。

failure映射冻结为：empty/multi-target、wrong cut、owner `UnsupportedSemantic/UnsupportedRepresentation/UnsupportedCapability`
或无合法rank-local candidate在clone mutation前返回silenceable failure；invalid IR/profile/policy/topology/rank、registry合同错误、
`InternalInvariant`和reserved baseline `ResourceExhausted`返回definite failure。optimized budget耗尽且baseline已通过则success并
返回baseline/report；driver cancellation按§6.6 definite abort且无result mapping。所有silenceable/definite failure必须证明
transaction clone未提交且payload byte-identical。

两个mutating op使用upstream operation handle的consume-old/produce-new形式：对target调用`consumesHandle + modifiesPayload`，
只读policy/profile/rank/budget param，并对fresh handle/report调用`producesHandle`；inspect对target调用`onlyReadsHandle`、只产生
inspection param且不写payload。empty/multi-target、stage不适用或无合法rank-local candidate在mutation前返回
silenceable failure且payload byte-identical；其它status严格按上一段映射。所有正常失败都发生在clone上；用户显式
Transform sequence的silenceable failure可由upstream `transform.alternatives`编排，但它不承担cost search。

明确禁止：

- 为每个 dtype/layout/model/op pair 建 transform op；
- 用 transform handle/param/attr 表示 beam、SPM liveness、physical versions 或 cost frontier；
- 用 `transform.alternatives` 代替 top-K/Pareto 搜索——它是 first-success，不是 cost selector；
- 声称单payload Transform op等价于production all-rank bundle coordination；
- 把materialized rank-local candidate称为winner，或省略
  `all_rank_and_binding_verification=Unverified` / `model_and_board_qualification=Unverified`；
- 把 TransformState 或 transform module 保存为 package/accepted artifact；
- production 与 Transform extension 各维护一套 matcher、legality 或 materializer。

Transform op的粒度按稳定compiler capability划分，而不是按每个upstream pass、dtype、layout或workload展开。三者均实现
`TransformOpInterface`和`MemoryEffectsOpInterface`，沿用upstream handle invalidation；debug/test默认开启expensive invalidation
check。payload IR始终是唯一事实源；若未来autotuner产生schedule，只能选择这些显式机制和参数，不能绕过numeric/effect/
resource gate或保存shadow candidate plan。Transform IR/param不进入accepted module、bundle或package。

`WaferTransformDialectExtension`只依赖MLIR Transform dialect/interfaces与WaferTransforms/Target libraries，并声明可能创建的
Wafer（含DTE ops）、func、linalg、tensor、scf、memref、bufferization、async、arith、math payload dialect；candidate中的
`async.token`/ops和bufferization临时adapter必须在extension registry显式加载，不能靠wafer-opt偶然预注册。Transform interpreter pass只由`wafer-opt`另行链接与
注册。Transform extension只有在核心planner、typed IR和named production pipeline稳定后实施；它不是主线完成的前置。
registration test必须在只注册Transform core + Wafer extension的fresh context解析/应用含async token、bufferization adapter和
DTE candidate的sequence，并验证missing dependent dialect不会在运行期才崩溃。

## 9. Target capability 与硬件约束

planning query使用06唯一owner的typed context；14只拥有其中ID、profile registry与qualification row事实：

```text
TargetCapabilityContext {
  target_profile: TargetProfileId
  use: CompilerEmission
     | ModelExecution(ModelProfileId)
     | BoardExecution(BoardEnvironmentId)
  qualification_floor: CompilerEmittable | ModelQualified | BoardSupported
  capability_registry_revision
}
```

合法组合固定为`CompilerEmission + CompilerEmittable`、`ModelExecution(id) + ModelQualified`或
`BoardExecution(id) + BoardSupported`；缺id、floor/use不匹配、unknown revision或profile无法由14唯一解析到target identity、
Kernel Runtime ABI与module format均是`Invalid`，不存在default/fallback。canonical resolved bytes包含上述字段及14映射出的
typed identity/ABI/format与registry digest，使用versioned length-prefix生成`target_capability_context_digest`；不能只hash
profile spelling，也不把派生identity复制回IR。

`wafer-compile`普通source-to-package request显式构造`CompilerEmission`；target-model named qualification从17的显式
`ModelProfileId`构造`ModelExecution`；board path只能从configured board environment构造`BoardExecution`。Q32.T没有
frontend/provider environment，固定构造`CompilerEmission + CompilerEmittable`，并在report标记model/board qualification及
binding/all-rank均未验证；Transform script不能传一个profile就隐式获得model/board资格。

planner不直接散落硬件 magic number，而从 resolved `TargetCapabilityContext` 对应 provider 查询。当前 Wafer profile至少应暴露：

- 每 tile 可用 SPM window `[0x10000, 0x2F0000)`，即 3,014,656 bytes，以及 alignment/reserved ranges；
- Tensor/Cx/NCx 的 block、tail、padding 和 physical map；当前非 int8 channel block 为 64，int8 为 128；
- RDMA source-strided/destination-sequential、WDMA inverse、GS 双侧映射及最多三层 outer stride/iteration 的 descriptor
  能力；
- compute engine 的 dtype、tile geometry、accumulator、layout/orientation 和 padding-lane contract；
- queue/resource/effect、provider completion、fence/wait 和 transport能力；
- instruction/CRT/CModel 已纵向资格化的字段。

每个 capability row 分开记录 `statically_representable`、`compiler_emittable`、`model_qualified` 和
`board_supported`，不能压成一个 `supported`。context的qualification floor决定可进入domain的最低证据级别：model-only
source-to-CModel vertical只能使用对应`ModelProfileId`已资格的`model_qualified` row；真实board
publication必须进一步要求同target/environment的`board_supported` row。板端不可用不阻塞本任务的model-only实现闭环，
但也不能把model结果宣传成硬件已校准。

当前 packet/wrapper 虽存在 GEMM transpose flags，但 live Instr op、target call 和 CRT 默认路径尚未形成完整 typed
transposition capability；当前 load/store 也并非所有 layout/local-offset route 都可表达。因此这些只能作为实现
checkpoint，不能被 planner 假设为已支持。任何新 family 必须纵向补齐 10/11/14/17 后才进入 domain。

padding correctness 是 hard legality：若 physical lanes 超出 logical valid domain，compute 只有在所选 family 证明这些
lanes不被读取、使用 neutral value 或在结果前被 mask 时才可运行 full physical extent。`exp(0)=1`、scalar add 等会破坏
zero padding invariant；此时必须使用 segmented valid tail、mask-capable family 或 materialize Tensor，不能靠 case 经验。

## 10. Selected IR 和下游责任

选中 proposal 通过 07 物化后只表达以下长期事实：

- typed memref memory space 与 accepted physical encoding；
- `memref.subview`/reassociation 等可验证 view；
- target-abstract compute/movement/collective；
- explicit resident SSA、spill/reload 和 immutable resource；
- async issue、provider completion、join、reuse 和 terminal wait；
- structured static traversal、tail 和合法 reduction chain。

下游责任：

- 07：从 proposal 原子构造完整 tile-dataflow IR，并验证 coverage/use-def；
- 08：physical map、valid domain、descriptor cover、view/route/materialization；
- 09：whole-rank exact SPM lifetime、alias、offset 和 peak；
- 10：implementation family 与 target-abstract compute/movement ops；
- 11：typed instruction field、geometry、descriptor 和 narrowing legality；
- 12：whole-variant DDR demand、lifetime、offset 和 package range；
- 13：collective/DTE completion 和 all-rank matching；
- 14/15：从final instruction/TargetCall rows派生、readback并在schema-v4投影canonical `RequiredCapabilitySet`；
- 16/17：source oracle、SystemC/CModel、multi-dtype 与 board 分层 evidence。

analysis cache、choice domain、candidate score、fusion label、layout label graph、group、scope-prefix 和 shadow schedule
不得跨越该边界。

## 11. 通用示例和 7B scale case

### 11.1 示例参数与协议事实

通用 contraction chain、diamond、fanout/fanin、broadcast/reduce、multi-root shared-input、attention-like dataflow 和
branched expert-like dataflow 都由相同对象处理：

- iterator/indexing maps 产生 IndexRelation；
- target provider 产生 parameterized families；
- backward propagation 连接 dependent tiles；
- selected physical versions 决定 resident dataflow；
- effect/numeric/resource/descriptor gates 决定 cut；
- cost vector 和 hard budget 决定有限候选顺序。

这些是协议。某个模型的 op 名、root 数、shape、tile 数和最终融合视图只是输入实例。

### 11.2 Llama-2 7B 单 block 作为规模压力

标准 Llama-2 7B 单 block、batch 1、sequence 16、TP16、FP16 storage 只作 scale/evidence case：每 rank 的
attention projections、MLP contractions、normalization、RoPE、softmax、residual 和 collectives 必须走同一通用算法。

它用于验证：

- shared-input 多 root 不按 QKV/gate-up 名称识别；
- projection orientation/layout 可在 capability 闭合后由 implementation 吸收，而不是先写整块 DDR 转置；
- 很小的 pointwise/reduce task 能通过 resident physical version 与相邻 compute 连通，而非每 op 一个 DDR group；
- tile 缩小带来的 SPM headroom、重复 invariant load、descriptor/issue 增长全部进入同一 cost vector；
- collective 是 completion barrier，不自动成为 DDR barrier；
- rank-count=16 的 all-rank transport、package 和 complete output 数值不回退。

当前 Q29/Q28/Q30/Q31 evidence 只作为迁移 baseline：rank-0 high-water 曾为
2,725,568 / 3,014,656 bytes（90.411%），现有 layout/weight movement 和碎片化小 task 暴露了联合优化空间。该数字、
当前 tile shape 和最终 resident components 不进入规则；新 planner 必须重新从最终 selected IR 统计，并用完整 PyTorch
expected 与 SystemC/CModel 比较数值。

## 12. 实施边界和完成定义

实施计划位于 `tasks/plans/physical-dataflow-synthesis.md`。施工必须纵向推进，而不是先造一个脱离 accepted IR 的 solver：

1. 冻结当前production structured IR、shared capability与typed IR合同，建立canonical semantic descriptor与current-v1 provider；
2. 实现IndexRelation、PhysicalEncoding、valid-domain和descriptor-cover property core；
3. 让保守baseline经新family/provider物化并通过全部exact gates；
4. 补齐implementation/transfer family到Instr/CRT/SystemC的最小纵向能力；
5. 独立实现并验证policy-free bitwise relation、pointwise propagation、implementation absorption和shared-input reuse；
   代数变换只在numeric policy完整且有直接correctness测试时注册；
6. search只组合上述mechanisms，加入惰性domain、constraint propagation、canonical frontier、Pareto beam、两级
   static-memory oracle、interval-aware dominance、hard budget和telemetry；
7. 切换production，并删除旧layout planner、implicit per-use materialization、scope-prefix/maximal-resident决策旁路、
   `estimatedTimePs`/`estimateScheduledRankProgramTimePs`/scalar-combination ranking/discovery-order fallback，以及13列出的
   collective schedule enums/options/default selector；
9. 可选地接入compiler-wide Transform control plane，共用同一C++ implementation。

完成必须同时满足：

1. 无 workload 名、固定 shape、parameter position 或 op-pair case table；
2. 无 fusion partition、tile integer、task permutation、resident subset 或 family Cartesian-product 全枚举；
3. hard budget、baseline fallback、canonical dedup 和 candidate-growth telemetry 有正负测试；
4. selected state 只以 typed payload IR 跨阶段，旧决策接口和重复事实源清零；
5. property tests 覆盖 dtype/block/tail/relation/descriptor/padding；拓扑覆盖 chain、diamond、fanout、fanin、
   multi-root、reduction、broadcast；workload 覆盖 GEMM/MLP、convolution/contraction、attention 和 branched/MoE-like；
6. rank-count=1/16、7B scale 和完整 PyTorch/SystemC numerical gate 不回退；
7. exact SPM/DDR/event/transport/instruction/ABI 和 atomic candidate/rank commit 不回退；
8. 预算耗尽仍返回合法 baseline 与结构化 diagnostic；
9. 板端 performance/calibration 仍由 later gate 拥有，不成为本任务虚假完成证据。

其中memory-oracle完成还要求：已知packing最优值、alignment hole、disconnected component、zero-byte和first-fit反例均能
闭合lower/upper；`ResourceExhausted`不推进bound且有incumbent时保持candidate合法；selection-sensitive shortlist才发生
追加capacity query；相同deterministic policy下query/node/selected signature可复现。7B迁移case必须报告当前selected
high-water、proof/gap和查询增量；若仍达到既有clique lower bound，应明确证明“零gap但未降低”，不能把算法接入本身
宣传成SPM容量收益。

局部 FileCheck、只生成候选、只减少 layout op、只在 7B case 生效、单 rank 通过或旧 planner 与新 planner 并存，都不构成
完成。

## 13. 参考材料

以下工作只提供算法启发，不改变本文的 Wafer IR/target 合同：

- MLIR Canonicalization是best-effort且不能承担pipeline correctness；Linalg参数化tiling/fusion建立在structured interfaces上。
  <https://mlir.llvm.org/docs/Canonicalization/>
  <https://mlir.llvm.org/docs/Dialects/Linalg/>
- VTC：在fusion后graph上用virtual tensor/index mapping消除kernel间显式data movement，并全局选择virtual-tensor
  creation strategy；它明确与layout optimization/operator fusion互补。本文借鉴“跨kernel物理关系联合选择”，不复制其IR。
  <https://www.usenix.org/conference/osdi26/presentation/hu-muyan>
- Welder：从 consumer tile 反向传播 dependent region，并联合考虑 memory traffic；本文采用有界 region propagation。
  <https://www.usenix.org/conference/osdi23/presentation/shi>
- SmartMem：跨算子选择 layout 并减少 layout conversion；本文把 route/encoding 纳入 physical-dataflow synthesis。
  <https://arxiv.org/abs/2404.13528>
- TASO使用verified substitution与cost-based backtracking，TENSAT使用e-graph/equality saturation探索通用graph rewrite；
  本文只在小semantic island中使用有证明、hard-capped alternatives，不做whole-model equality saturation。
  <https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf>
  <https://proceedings.mlsys.org/paper_files/paper/2021/hash/cc427d934a7f6c0663e5923f49eba531-Abstract.html>
- ALT联合graph data layout与operator loop tuning；PBQP显式计入format-conversion cost做primitive/layout selection。
  本文借鉴参数域和约束传播，不引入第二份assignment IR。
  <https://arxiv.org/abs/2210.12415>
  <https://arxiv.org/abs/1710.01079>
- GraphTurbo展示DSA上的coarse subgraph partition/order与cross-layer scheduling；COSMA联合scratchpad accelerator的operator
  scheduling、memory allocation与tensor replacement；Korch在GPU上做operator fission与BLP kernel orchestration。本文只借鉴
  它们对计算/数据移动/资源联合决策的不同拆法，并用typed effects、exact resource gates与bounded frontier约束适用范围。
  <https://www.usenix.org/conference/osdi23/presentation/zhao>
  <https://arxiv.org/abs/2311.18246>
  <https://arxiv.org/abs/2406.09465>
- MLIR Transform Dialect：作为 transformation orchestration/control plane，而不是 solver state 或执行 artifact。
  <https://mlir.llvm.org/docs/Dialects/Transform/>
  <https://mlir.llvm.org/docs/Tutorials/transform/Ch4/>
- Google MiniMalloc提供fixed-capacity canonical search、section inference和dominance pruning；Wafer保留其固定容量核心，
  minimum-height由本stage用三态capacity query和独立validator组合，避免把solver exhaustion当成不可行。
  <https://research.google/pubs/minimalloc-a-lightweight-memory-allocator-for-hardware-accelerated-machine-learning/>
  <https://github.com/google/minimalloc/blob/9f5cf810fec4494df473c23cffd0567989e81b69/src/solver.cc>
- Bounded Memory Scheduling使用heuristic feasible solution与lower bound共同缩小exact search；本文只借鉴
  incumbent/lower-bound envelope，不复制其schedule表示或全局求解器。
  <https://www.cs.rice.edu/~zoran/Publications_files/PACT2014-BMS.pdf>
- OR-Tools CP-SAT区分feasible、infeasible、optimal和unknown；本文同样保持legality、objective gap与resource exhaustion
  正交，但不引入CP-SAT作为compiler依赖。
  <https://developers.google.com/optimization/cp/cp_solver>
