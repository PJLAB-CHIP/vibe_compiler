# Wafer Local Structured Tensor Normalization 与 Optimization 设计

状态：2026-07-19按已完成Q33 required tensor normalization、adoption与qualification实现同步。本文拥有post-SPMD StableHLO local compute与logical collective到
optimizer-ready structured tensor IR的normalization、required normal form和target-independent固定优化合同；不拥有SPMD、
task/dataflow candidate、target-aware choice、memory、target或runtime。实现状态看
`tasks/progress.md`。

本文定义收敛后的稳定边界。当前production在collective/residual normalization与official legalization之间保留两个按cut point
登记的required canonicalization，再运行explicit required normal form；探索性机制已关闭为空proposal，不额外启用optional generic
cleanup。Equivalent-IR构造、typed qualification/archive、完整隔离runner、active-ref production消费和mandatory vertical均已闭合；
当前active set及完成证据见§8。不能把schema或局部测试单独解释为未来机制已经资格化。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q15 helper输出并重新通过program verifier的post-SPMD StableHLO local program；可能包含parameter shard
  payload、StableHLO logical collectives，或无用户sharding时的replicated local body。
- Current stage responsibility:
  将StableHLO compute/data movement/constants通过pinned官方StableHLO-to-Linalg conversion规整成
  `linalg`/`tensor`/`scf`/`arith`/`math`；先把supported StableHLO collectives转换为
  `wafer.linalg_ext.collective.*`destination-style tensor ops；清理可静态证明的SPMD residual；以显式、
  可验证的rewrite建立required structured normal form，并运行已资格化且对所有输入固定启用的target-independent优化。
- Output artifact / IR:
  optimizer-ready target-independent structured tensor program，包含local compute、ConstantLike values和logical collective；
  不残留raw StableHLO或SDY语义，也不要求下游依赖某个偶然producer形状或optional generic cleanup的worklist收敛结果。
- Downstream consumer:
  physical-dataflow synthesis从该artifact建立rank-local semantic descriptor、candidate与whole-variant commit；当前Q29 scheduler
  是迁移baseline，不是长期consumer合同。
- User-level driver / named pipeline:
  production只经`wafer-compile`并继续到verified rank-local structured tensor program。
  `wafer-lower-stablehlo-to-linalg`是显式IR
  debug/test pipeline，不是program-directory入口或用户stop-stage。
- Explicit non-goals:
  不运行Shardy/XLA SPMD，不写parameter shard metadata，不决定task/tile、logical rank specialization、
  physical layout、SPM/DDR、DTE、target CRT、manifest或runtime binding；不在固定pipeline中做target-aware fusion、
  implementation/layout/residency/search，也不把任意pass顺序开放成production tuning surface。
- Completion gate:
  Q15真实helper输出经同一normalization后不残留StableHLO/SDY，supported collectives成为verifier-legal
  LinalgExt ops；required normal form不依赖best-effort canonicalization；baseline及经CSE、standard view/DPS等价改写的
  结构都能被同一下游直接消费。每个宣称采用的upstream机制必须在production或candidate调用点发生真实改写，并通过
  semantic、alias/effect、memory、target与数值gate；unsupported semantic fail closed而不是留给下游猜测。
```

## 2. 稳定边界

normalization输入是一个post-SPMD local tensor graph。用户没有`mark_sharding`时，当前helper correctness基线可以
产生replicated graph；rank-count=1也可能得到whole-shape graph。是否存在collective不决定local compute是否合法。

本stage只保留数学与structured tensor事实：

```text
func + tensor + linalg + scf + arith + math
  + wafer.linalg_ext.collective.*
```

shape、dtype、indexing maps、iterator types、DPS ties、reduction region和SSA use-def必须保持可验证。target facts
只能在下游作为legality/cost input，不能提前变成layout strings、memory attrs、physical endpoint或packet fields。

当前frontend program没有typed mutable-state/model graph合同；本stage也不虚构state/resource owner。parameter在
program directory和function argument上的绑定由tasks/02 verifier拥有，normalization只保持IR value/type关系。

## 3. Compute Normalization

Wafer复用当前pinned StableHLO官方Linalg legalization，而不是维护另一套按op名分发的窄conversion。official
conversion通过Dialect Conversion和legality target将可表达的StableHLO转换成structured IR；无法转换的raw
StableHLO使pipeline失败。

主要family：

| StableHLO语义 | normalized form | 本stage保留的事实 |
| --- | --- | --- |
| constant | `arith.constant`或其它ConstantLike | exact element type/shape/value |
| pointwise | `linalg.generic` + scalar arith/math | broadcast/indexing与dtype |
| broadcast/reshape/transpose/slice/concat | linalg/tensor/scf data-movement或view结构 | dimension mapping、static slice/view relation |
| `dot_general`/dot | `linalg.matmul`、batch matmul或structured generic | batch/contracting dims、indexing maps、accumulator/result type |
| reduce/reduce-window supported subset | linalg reduction/pooling-like structure | reduction dims、init与combiner region |
| staged softmax/norm/RoPE/MLP graph | 细粒度reduce/pointwise/shape ops | 原SSA dataflow，不引入Wafer高层model op |

`dot_general`是否最终能映射Wafer GEMM由Q16 instruction/geometry gates决定；normalization只证明structured
tensor semantics。QKᵀ、attention-value、batched contraction等不能靠operand名识别，必须由dimension numbers与
indexing maps推出。

softmax、RMSNorm、LayerNorm和RoPE在当前input中是fine-grained StableHLO graph。长期不引入
`wafer.softmax`、`wafer.norm`或`wafer.rope`来隐藏数学语义。若其multi-stage schedule需要额外temporary或
DDR/SPM residency，由tile-dataflow candidate与memory层通过显式IR建立。

## 4. Logical Collective Handoff

StableHLO collective在official compute conversion前先进入Wafer-owned tensor handoff：

- `stablehlo.all_gather` → `wafer.linalg_ext.collective.all_gather`；
- `stablehlo.all_reduce` → `wafer.linalg_ext.collective.all_reduce`；
- `stablehlo.reduce_scatter` → `wafer.linalg_ext.collective.reduce_scatter`；
- `stablehlo.all_to_all` → `wafer.linalg_ext.collective.all_to_all`；
- `stablehlo.collective_permute` → `wafer.linalg_ext.collective.collective_permute`。

这些ops是destination-style tensor ops，并实现MLIR`TilingInterface`、Wafer tiling interface和collective info
interface。它们保留：

- input/init/result tensor type和DPS tie；
- collective axis或split/concat dimension；
- single `rank_group`或同shape的`rank_groups`，二者互斥；
- all-reduce/reduce-scatter的exact scalar combiner region；
- collective-permute的logical source-target pairs。

verifier用execution mesh检查logical ranks范围，并用selected rank-group size检查gather/scatter/all-to-all shape
relation。rank group只含logical rank，不含physical endpoint、DTE channel、route、SPM buffer或runtime resource。

collective在rank-local structured tensor program中仍是tensor semantics；Q29 task materialization才产生
buffer-level `wafer.tile.*` collective，后续communication/instruction阶段再选择transport。normalization不得
直接跳到DTE或把algorithm/peer assignment塞进LinalgExt attrs。

当前没有`segmented_all_to_all`、MPMD component edge或MoE count/capacity合同；这些需要真实frontend表示与下游
consumer后另行设计，不能作为当前completion gate。

## 5. Static Residual Cleanup

XLA SPMD output可能带有可以从constants与static tensor views完全求值的rank/mask helper结构。Wafer在official
conversion前后运行窄的residual cleanup，覆盖：

- `stablehlo.partition_id`/`stablehlo.replica_id`形成的static helper；
- static `tensor.extract_slice`、collapse/expand-shape和`tensor.extract`常量链；
- all-constant integer passthrough/add/compare `linalg.generic`。

cleanup只在DenseElementsAttr、static type/offset/shape和op semantics能完整证明结果时折叠。它不是第二套
StableHLO lowering，也不能扩成运行时shape evaluator或按rank名字matcher。最终raw StableHLO residual仍存在时
pipeline fail closed。

其中post-legalization与structured-tensor两个位置的canonicalization是required form的一部分，分别以
`PostLegalizationCanonicalization`和`StructuredTensorCanonicalization`进入invocation inventory。它们保持既有顺序，负责把
静态rank/mask helper化成下游可验证的常量offset；任一位置关闭都会改变scheduler legality，因此不属于可资格化的optional
cleanup。proposal中的`StablehloCleanup`/`StructuredTensorCleanup`只表示required form之后的额外尝试，当前空proposal不会启用。

具体实现中，post-legalization cleanup从标量`tensor.extract`反向证明常量来源：只跟踪
`arith.constant` DenseElementsAttr、offset/stride为字面量或可由常量整数SSA链精确证明的
`tensor.extract_slice`，以及静态元素数保持的
collapse/expand-shape，并按canonical row-major element order计算唯一标量结果。任一dynamic
index/shape/offset/stride无法由常量链证明、越界或非常量来源都保留原IR交给后续legality gate；本步不物化
shaped result，不保存旁路常量表。折叠pattern本身负责建立正确result/use关系；无用view/constant链可以再由
best-effort canonicalization清理，但下游legality和正确性不能依赖该清理是否发生。

SDY op/type/attr不属于post-SPMD local program。Q15在normalization前已有零SDY gate，本stage不能把residual SDY
静默当unknown dialect保留。

## 6. Structured Optimization 与 Upstream Adoption

本stage把“优化”分为三个边界，避免把固定optimization、搜索choice和target lowering混成任意pass串：

1. **required normalization**：为下游接口建立确定语义形态的显式rewrite/verifier，例如DPS init、view/reshape relation和
   reduction source的稳定恢复。它是correctness合同，不能委托给optional generic cleanup的greedy收敛。
2. **fixed target-independent optimization**：对所有输入采用同一已资格化policy，只允许保持source numeric/effect/alias语义的
   upstream pass或pattern。只消除scalar/shape/identity scaffolding且不改变tensor sharing/lifetime的窄CSE subset可以评估为
   fixed；whole-tensor CSE不能默认进入本层。无实际改写或无consumer收益的pass不因“常用”而加入production。
3. **candidate-local mechanism**：tiling、producer fusion、whole-tensor CSE、unit-dim/view propagation、empty-tensor elimination和
   loop hoist等会改变tile、sharing、lifetime、residency或target机会的mechanism由06在隔离candidate clone中选择；05只提供
   共享typed utility，不在固定pipeline提前决定physical dataflow。

### 6.1 Required structured normal form

本节只拥有**pre-bufferization structured tensor normal form**。required tensor normalizer与postcondition
verifier是两个独立入口：normalizer只做建立合同所必需的确定性rewrite，verifier只从当前IR检查
结果，不以“再跑一次canonicalizer”修复输入。transaction root固定为拥有rank entry及其private callee的rank-local
`builtin.module`：先clone整module，在clone上运行normalizer，通过纯读verifier后才以整module原子替换；任何non-success均使
原module byte-identical，不能只clone entry func而修改共享callee。normalizer必须幂等并按canonical region/block/op order消费versioned work policy。normal form不写
marker attr、side table、descriptor或producer名字；打开或关闭optional generic cleanup不得改变同一IR的支持性。两个位置固定的
required canonicalization另按各自cut point和顺序接受完整主线验证，不能与optional cleanup混为同一开关。

入口返回typed outcome，不将unsupported、非法IR与resource exhaustion折叠成一个`failure()`：

```text
TensorNormalizationOutcome {
  status: Success | UnsupportedSemantic | InvalidIR | ResourceExhausted | InternalInvariant
  changed
  diagnostic?: {family, reason, canonical_operation_path}
  work_summary
}
```

variant invariant固定为：`work_summary`在所有status都required并包含截至退出点的checked counters；`Success`唯一允许
`changed=true`且禁止diagnostic，`changed`必须等于输入/输出`CanonicalIRSnapshotV1(SemanticStructure)`是否不同；四个
non-success都要求`changed=false`和完整diagnostic，不返回output module。`UnsupportedSemantic`只用于closed表示域外，
`InvalidIR`只用于输入违反当前IR verifier/typed合同，`ResourceExhausted`只用于上述work policy耗尽或checked counter overflow，
`InternalInvariant`表示compiler bug并作为definite failure；reason enum必须与status匹配。任何partial clone、partial diagnostic
列表或“changed但未提交”组合均非法。

本文和07共享唯一结构快照编码器，只用于幂等/atomicity，不把printer或MLIR bytecode偶然稳定性当合同：

```text
CanonicalIRSnapshotV1 {
  schema_version = 1
  mode: SemanticStructure | MutationGuard
  root_operation_kind
  structural_bytes
  sha256_digest
}
```

编码器按region/block/op source order递归；op使用registered typed op identity，result/type、inherent properties和discardable
attrs分别用其registered canonical codec，dictionary attr按identifier bytes排序，block arguments/results用结构ordinal，operand/successor用definition/block path编码，region/
block数量和顺序显式length-prefix。SSA打印名、pretty-printer whitespace和context pointer永不编码；symbol/name属性作为真实attr
照常编码，不能因“名字不参与语义恢复”而删除。`SemanticStructure`排除location，`MutationGuard`再编码完整location tree及其
typed payload；两种mode都拒绝没有registered canonical codec的unknown type/attr/property。digest只校验完整bytes，不代替byte equality。
normalizer once/twice与`changed`比较`SemanticStructure`；non-success source atomicity比较前后`MutationGuard`。encoder schema、
type/attr codec revision和MLIR dialect registry digest进入bytes header；不hash host object layout，也不调用generic printer/bytecode。

`family` 只取下表七类，`reason`是owner-defined closed enum，`canonical_operation_path`由region/block/op
ordinal构成且只用于diagnostic，不进入IR或digest。多个错误同时存在时只报canonical order下首个；相同
input/work policy必须返回相同status、reason和canonical operation path。除`Success`外丢弃完整clone，source transaction root
保持byte-identical。

`schedulable root`是frontend typed program binding指定的rank-local entry function；`schedulable path`是该entry中的
所有nested region，以及经direct `func.call`可静态解析到defined private function的transitive closure。函数按
symbol-table source order、region/block/op按IR source order遍历，同一private callee只验证一次；recursive call cycle与
indirect call返回`UnsupportedSemantic`。external/unknown call保持conservative all-resource barrier且normalizer不跨越；
如果其tensor result/operand进入structured scheduling且没有typed semantic/effect summary，该path返回
`UnsupportedSemantic`，不从symbol名字猜语义。

`RequiredTensorNormalizationWorkPolicyV1`以transaction初始schedulable closure的op数`O`、SSA operand与successor-operand边数
`E`、static rank与reassociation index总数`D`固定`fuel = 1024 + 32*O + 8*E + 8*D`。op/region visit、rewrite
attempt、committed rewrite、new op和relation node分别消耗1/2/8/8/2单位fuel；所有加乘checked overflow。耗尽返回
`ResourceExhausted`并丢弃clone。调整系数必须新增work-policy version并重跑qualification，不允许根据wall time
在运行中扩容。

当前static-ranked支持域的逐family postcondition固定为：

| family | required postcondition |
| --- | --- |
| program envelope | schedulable path无StableHLO、SDY和`unrealized_conversion_cast`；tensor均为当前支持的static ranked type |
| structured compute DPS | local compute op实现`DestinationStyleOpInterface`并保持pure tensor semantics；每个result与唯一init tie及exact type一致；named和generic form都合法 |
| DPS init / reduction | init读取性、来源和combiner可从DPS tie、SSA use-def、static view及scalar region证明，不要求init由直接`linalg.fill`定义 |
| scalar region | block argument、yield、ConstantLike、captured或inline scalar、typed raw constant、evaluation order及fast-math/numeric policy均可解释 |
| tensor relation | reshape、permutation、slice/subset关系可从typed view/structured semantics查询；只有真正identity的full view被显式删除 |
| structured control | static zero/one-trip `scf.for`由required rewrite显式折叠；其它loop-carried value/effect由block argument与yield完整表达 |
| logical collective | DPS、rank group、mesh、shape及combiner合同完整，并具有generic pass可见的Communication write barrier |

DPS init用两个正交typed事实恢复：

```text
InitReadState = Unread | Read
InitOrigin = Undefined | ExactSplat(typed raw value) | ExistingValue
```

`Read + Undefined`非法；共享/非共享empty、fill、ConstantLike及其static tensor view链必须得到相同结论。
`InitReadState`和`InitOrigin`是从当前IR重算的invocation-local analysis value，不存入attr、side table或artifact。
`ExactSplat`只陈述init payload，不自动证明reduction identity或padding neutral；后者仍由numeric policy和
selected implementation证明。

pre-bufferization full `tensor.extract_slice`只有同时满足零offset、单位stride、完整size、无rank
reduction时才是identity。source/result ranked-tensor type完全相同时可直接替换；否则只能在MLIR
`tensor.cast`合法且element type、rank、static shape facts与encoding都不丢失时显式保留cast。其它tensor
ViewLike必须保留并向下游提供exact relation；不能因printer形态像full slice就折叠。

post-bufferization selected payload的`memref.subview`、allocation root、memory space和layout合同由07单独拥有；
它必须使用独立entry point、postcondition、fuel和adoption cut point，不调用本节的tensor verifier。两层只共享
纯读`IdentityViewProof`和static trip-count proof helper；helper返回`ProvenIdentity | ProvenNonIdentity |
Unsupported | ResourceExhausted`及当前IR中的proof facts，不执行rewrite、不保存root/layout也不形成跨stage
normal-form合同。

### 6.2 Adoption maturity 与 fixed-policy seam

底层record使用五个正交维度，不把“依赖在source tree中”、“已链接”、“debug可重放”和“production
已调用”压成一个availability enum：

```text
ProviderOrigin = Internal | Upstream | Vendored | Toolchain
Availability = Absent | PresentUnresolved | Resolved
ExposureSet ⊆ {DebugRegistered, LibraryCallable, BackendExecutable}
AdoptionMode = None | RequiredNormalization | FixedOptimization | BestEffortCleanup |
               CandidateLocal | TargetSpecific | TargetBackend
EvidenceKindV1 = Rewrite | BackendAction | InvocationOnly
QualificationStatus = Unassessed | NoOpObserved | DownstreamBlocked | Qualified | Rejected
OptimizationBatchStatusV1 = Qualified | Rejected
```

上述closed enum以及本节后续在同一代码块中以`A | B | ...`声明的closed enum，wire ordinal均按从左到右从0开始冻结；
set按ordinal或完整element bytes排序。ordinal不从C++ declaration/registration order推导，新增variant只能追加并提升包含它的
record schema；已有ordinal不得重排或复用。

`MechanismKey`的唯一typed定义也在本文，避免registry、disable-one、telemetry和Transform attr各自使用pass名或自由字符串：

```text
MechanismKeyV1 {
  schema_version = 1
  semantic_mechanism_id: u32
}
```

`semantic_mechanism_id`来自compiler library内唯一declarative registry；每个row固定一个永不复用的非零u32和仅用于展示的
label，注册时拒绝重复ID、unknown ID和零。key标识一种可独立声明precondition/effect/proof/work-policy的**语义机制**，不编码
pass/tool/task名、dtype、shape、workload、cut point、target profile、候选实例或registration order；这些分别属于spec字段或
invocation observation。新增机制只追加registry row，不改变既有ID；改变既有机制的semantic/effect contract必须分配新ID。
canonical bytes固定为`"wafer.mechanism-key\0" || u16be(1) || u32be(semantic_mechanism_id)`；display label不进入bytes、
相等性或排序。`MechanismKeyAttr`只封装这份typed key并在parse/verify时查同一registry，不接受字符串fallback。

compiler-wide cut point和operation domain同样由本文唯一typed定义；它们描述稳定artifact边界，不描述pass调用序号：

```text
OptimizationCutPointV1 =
    FrontendProgramImport
  | PreSPMDStableHLOModule
  | SPMDPartitionTransaction
  | PostSPMDStableHLOModule
  | StructuredTensorModule
  | SelectedPhysicalPayloadModule
  | FinalInstructionModule
  | TargetLLVMModule
  | DevicePublicationTransaction

OperationDomainV1 = IRDomainV1 | ArtifactActionDomainV1

IRDomainV1 {
  semantic_families: sorted-set<SemanticOperationFamilyV1>
  required_interfaces: sorted-set<OperationInterfaceKeyV1>
  allowed_dialects: sorted-set<DialectKeyV1>
}

ArtifactActionDomainV1 {
  executor_kinds: sorted-set<ActionExecutorKindV1>
  action_kinds: sorted-set<ArtifactActionKindV1>
  input_artifact_kinds: sorted-set<ArtifactKindV1>
  output_artifact_kinds: sorted-set<ArtifactKindV1>
}
```

这些key/enum都来自compiler library的唯一declarative registry并使用不复用的u32 ordinal；semantic family表达view、elementwise、
contraction、reduction、control、bufferization、movement、communication、instruction或target-call等通用职责，
不是op spelling、dtype/shape或workload case。IR operation只有dialect属于allowed set、semantic family命中且实现全部required
interfaces时才进入domain；mechanism自己的precondition再做更细typed判断。每个set按ordinal排序去重，semantic/allowed set非空，
unknown ordinal、重复、printer name、trait字符串或空泛`all`拒绝。artifact action覆盖typed frontend import、compiler-owned
artifact transaction及target/device publication：以executor kind（internal adapter/frontend helper/target compiler/linker/
publication helper）、action kind（import/partition/compile/optimize/link/verify/publish）和typed input/output artifact kind表达，
不伪造MLIR dialect/interface；四个set均非空。canonical bytes为
`"wafer.operation-domain\0" || u16be(1) || u8(discriminant)`后连接对应variant的
`u32be(count) + u32be(ordinal)...`列表；cut point编码为
`"wafer.optimization-cut-point\0" || u16be(1) || u32be(ordinal)`。

这里区分**standalone owner bytes**与外层TLV：`OptimizationCutPointV1`在`AdoptionSpec`和
`InvocationEvidenceV1`中直接以`ClosedEnum`的u32 ordinal作为field payload；只有把cut point单独用作canonical key时才使用上述
`wafer.optimization-cut-point` envelope。`OperationDomainV1`作为`Record`嵌入时，其payload恰为上述完整
`wafer.operation-domain` bytes（含schema/discriminant和唯一variant body），外层只再写一次Record type-tag与payload length，不递归
重编码variant set。`MechanismKeyV1`同理：`Record(MechanismKeyV1)` payload恰为§6.2已经冻结的完整
`wafer.mechanism-key` bytes。其它没有独立owner envelope的nested record才按下文field-TLV递归编码。任何实现同时省略或重复
owner domain separator/schema都不是canonical form。

前三个cut分别表示typed frontend program/import transaction、进入SPMD前的StableHLO module和产生all-and-only specialized rank
modules的SPMD partition transaction；因此inventory能覆盖frontend/helper/SPMD，而不是从post-SPMD才开始。conversion mechanism
按其输入artifact所属cut登记，输出合同由spec的typed contract refs约束，不为每个pass新增cut值。

一个`MechanismKeyV1`恰好绑定一个cut point和一份operation domain；同一upstream utility若在不同artifact层调用，必须注册不同
semantic mechanism key/spec，provider identity可以相同。这样`DisableOne(key)`、telemetry和qualification都只有一个位置含义，
也无需把cut point字符串塞进key。

Q33 adoption record的唯一typed schema也由本节拥有；16与实施计划只引用，不得复制字段子集：

```text
AdoptionSpec {
  schema_version
  mechanism_key: MechanismKey
  provider_origin
  provider_identity_and_revision
  optimization_cut_point: OptimizationCutPointV1
  operation_domain: OperationDomainV1
  availability
  exposure_set
  adoption_mode
  precondition_contract
  numeric_contract
  effect_contract
  work_policy_id
  owner
  evidence_kind
}

QualificationObservation {
  schema_version
  mechanism_key
  spec_digest
  build_toolchain_host_corpus_feature_config_digests
  qualification_policy_id
  ordered_outcome_counts_and_skip_failure_reasons
  invocation_and_rewrite_counts_by_cut_point_and_case
  mechanism_gate_results
  compile_work_and_fuel_summary
  exact_static_cost_vector_schema_and_values
  ordered_abba_wall_rss_samples
  qualification_status
  closed_reason
  optimization_proposal_digest
  qualification_run_digest
  optimization_publication_attempt_digest
}

OptimizationBatchObservationV1 {
  schema_version
  proposal_digest
  qualification_identity
  qualification_policy
  global_static_comparisons
  global_abba_samples
  equivalent_ir_2x2_gate_results
  production_all_on_gate_results
  batch_status
  closed_reason
  qualification_run_digest
  optimization_publication_attempt_digest
}
```

顶层field number和type固定如下，不能由C++成员顺序或JSON key推断：

| `AdoptionSpec` field | type |
| --- | --- |
| 1 `schema_version` | `U16` |
| 2 `mechanism_key` | `Record(MechanismKeyV1)` |
| 3 `provider_origin` | `ClosedEnum` |
| 4 `provider_identity_and_revision` | `Record(ProviderIdentityV1)` |
| 5 `optimization_cut_point` | `ClosedEnum(OptimizationCutPointV1)` |
| 6 `operation_domain` | `Record(OperationDomainV1)` |
| 7 `availability` | `ClosedEnum` |
| 8 `exposure_set` | `SortedSet<ClosedEnum>` |
| 9 `adoption_mode` | `ClosedEnum` |
| 10 `precondition_contract` | `Record(ContractRefV1)` |
| 11 `numeric_contract` | `Record(ContractRefV1)` |
| 12 `effect_contract` | `Record(ContractRefV1)` |
| 13 `work_policy_id` | `Record(WorkPolicyRefV1)` |
| 14 `owner` | `Record(OwnerKeyV1)` |
| 15 `evidence_kind` | `ClosedEnum(EvidenceKindV1)` |

| `QualificationObservation` field | type |
| --- | --- |
| 1 `schema_version` | `U16` |
| 2 `mechanism_key` | `Record(MechanismKeyV1)` |
| 3 `spec_digest` | `Digest32` |
| 4 `build_toolchain_host_corpus_feature_config_digests` | `Record(QualificationIdentityV1)` |
| 5 `qualification_policy_id` | `Record(QualificationPolicyRefV1)` |
| 6 `ordered_outcome_counts_and_skip_failure_reasons` | `Sequence<OutcomeCountV1>` |
| 7 `invocation_and_rewrite_counts_by_cut_point_and_case` | `Sequence<InvocationEvidenceV1>` |
| 8 `mechanism_gate_results` | `Record(GateEvidenceBundleV1)` |
| 9 `compile_work_and_fuel_summary` | `Record(WorkSummaryV1)` |
| 10 `exact_static_cost_vector_schema_and_values` | `Sequence<StaticComparisonEvidenceV1>` |
| 11 `ordered_abba_wall_rss_samples` | `Sequence<ABBASampleV1>` |
| 12 `qualification_status` | `ClosedEnum` |
| 13 `closed_reason` | `Optional<Record(ClosedReasonV1)>` |
| 14 `optimization_proposal_digest` | `Optional<Digest32>` |
| 15 `qualification_run_digest` | `Digest32` |
| 16 `optimization_publication_attempt_digest` | `Optional<Digest32>` |

`OptimizationBatchObservationV1`使用同一TLV/type-tag规则，field number按上面声明顺序1..12；字段类型依次为`U16、Digest32、
Record(QualificationIdentityV1)、Record(QualificationPolicyRefV1)、Sequence<StaticComparisonEvidenceV1>、
Sequence<ABBASampleV1>、Record(GateEvidenceBundleV1)、Record(GateEvidenceBundleV1)、ClosedEnum、
Optional<Record(ClosedReasonV1)>、Digest32、Digest32`，unknown/missing/extra同样拒绝。

nested record也不是opaque bytes；下面声明顺序就是各record从1开始的field number，全部字段required，只有显式`Optional<T>`可
absent：

```text
EquivalentInputVariantV1 = Original | Metamorphic
OptimizationComparisonKindV1 = GlobalAllOffVsAllOn | FixedDisableOneVsAllOn |
                           CleanupDisableOneVsAllOn
StaticComparisonResultV1 = BStrictlyBetter | BEqual | BRegressed | IncomparableUnknown
GateStatusV1 = Passed | Failed | Skipped | Unsupported | Timeout | Cancelled
ABBASamplePositionV1 = ALeft | BLeft | BRight | ARight
ProcessStatusV1 = Success
BackendActionStatusV1 = Success | Failed | Timeout | Cancelled
MechanismInvocationOutcomeV1 = Applied | NoChange | NotApplicable | Unsupported |
                               ResourceExhausted | Invalid | Cancelled
OptimizationGroupSelectionV1 = AllOn | AllOff | DisableOne

ProviderIdentityV1 { provider_id: TypedId, revision: Bytes, content_digest: Optional<Digest32> }
ContractRefV1 { contract_kind: ClosedEnum, contract_id: U32, contract_schema: U16,
                canonical_definition_digest: Digest32 }
WorkPolicyRefV1 { policy_kind: ClosedEnum, policy_schema: U16, canonical_policy_digest: Digest32 }
OwnerKeyV1 { owner_id: U32 }
QualificationIdentityV1 { build: Digest32, toolchain: Digest32, host_kernel_affinity_governor: Digest32,
                          corpus: Digest32, feature_config: Digest32 }
QualificationPolicyRefV1 { policy_id: U32, policy_schema: U16, canonical_policy_digest: Digest32 }
OutcomeCountV1 { outcome: MechanismInvocationOutcomeV1,
                 reason: Optional<Record(ClosedReasonV1)>, count: U64 }
RegistryRefV1 { id: U32, registry_schema: U16, registry_digest: Digest32 }
BackendActionEvidenceV1 { invocation_id: Digest32, action_ordinal: U32,
                          argv: Sequence<Bytes>, observed_tool_digest: Digest32,
                          observed_output_digest: Digest32,
                          terminal_status: BackendActionStatusV1 }
InvocationEvidenceV1 { invocation_site: RegistryRefV1, cut_point: ClosedEnum(OptimizationCutPointV1),
                       case_key: QualificationCaseKeyV1,
                       invocation_count: U64, rewrite_count: U64,
                       backend_actions: Sequence<BackendActionEvidenceV1> }
QualificationCaseKeyV1 { corpus: RegistryRefV1, rank_count: U32,
                         input_variant: EquivalentInputVariantV1 }
OptimizationComparisonKeyV1 { case_key: QualificationCaseKeyV1,
                         comparison_kind: OptimizationComparisonKindV1,
                         disabled_mechanism_key: Optional<Record(MechanismKeyV1)>,
                         configuration_a: OptimizationConfigurationV1, configuration_b: OptimizationConfigurationV1 }
GateEvidenceV1 { gate: RegistryRefV1, case_key: QualificationCaseKeyV1,
                 configuration: OptimizationConfigurationV1, status: GateStatusV1,
                 evidence_digest: Digest32,
                 terminal_reason: Optional<Record(ClosedReasonV1)> }
GateEvidenceBundleV1 { ordered_gate_results: Sequence<GateEvidenceV1> }
WorkCounterV1 { counter_id: U32, value: U64 }
WorkSummaryV1 { work_policy_digest: Digest32, ordered_counters: Sequence<WorkCounterV1> }
MetricEvidenceV1 { metric_id: U32, value: Optional<U64> }
ExactStaticVectorEvidenceV1 { registry_schema: U16, registry_digest: Digest32,
                              ordered_metrics: Sequence<MetricEvidenceV1> }
OptimizationConfigurationV1 { fixed_selection: OptimizationGroupSelectionV1,
                      fixed_disabled_key: Optional<Record(MechanismKeyV1)>,
                      cleanup_selection: OptimizationGroupSelectionV1,
                      cleanup_disabled_key: Optional<Record(MechanismKeyV1)> }
StaticComparisonEvidenceV1 { proposal_digest: Digest32, comparison_key: OptimizationComparisonKeyV1,
                             vector_a: ExactStaticVectorEvidenceV1,
                             vector_b: ExactStaticVectorEvidenceV1,
                             comparison_result: StaticComparisonResultV1 }
ABBASampleV1 { proposal_digest: Digest32, comparison_key: OptimizationComparisonKeyV1,
               block_index: U32, sequence_position: ABBASamplePositionV1,
               wall_ns: U64, peak_rss_bytes: U64,
               process_status: ProcessStatusV1 }
ClosedReasonV1 { reason: Record(RegistryRefV1), detail_digest: Optional<Digest32> }
```

`contract_id`、owner、gate、counter和metric ID均来自各owner的closed declarative registry；invocation site与corpus通过
`RegistryRefV1`绑定closed registry revision/digest，不能用裸ordinal跨archive解释。definition/policy/registry digest把ID
绑定到实际typed内容，不能让自由文本成为合同。human-readable diagnostic可作为由`detail_digest`引用的附件，但不参与控制流。
`ClosedReasonV1.reason`必须引用05拥有的全局closed-reason registry row；row定义稳定的reason category与允许出现的owner/outcome
集合，新增reason只能追加ID并提升registry schema，不能让不同owner各自解释同一ordinal。
type-tag数值冻结为`Bool=0x01, U16=0x02, U32=0x03, U64=0x04, ClosedEnum=0x05,
TypedId=0x06, Digest32=0x07, Bytes=0x08, Record=0x09, Sequence=0x0a, SortedSet=0x0b,
Optional=0x0c`；`Digest32`必须恰32 bytes，`TypedId/Bytes`先以u32 length prefix编码，sequence保留顺序，set按完整element bytes
排序去重。`TypedId`限制为1..128 bytes的canonical ASCII `[a-z0-9][a-z0-9._-]*`，不接受Unicode normalization、大小写
alias或NUL；`Bytes`是u32长度内的原始bytes，具体record policy可再施加更小上限但不能改编码。`Optional<T>` payload absent固定为单byte`0x00`；present固定为`0x01 || u8(inner-type-tag) ||
u32be(inner-size) || inner-payload`，禁止其它presence值、nested optional或absent后残留bytes。metric unknown用absent optional，
不能同时携带value或用0冒充known zero。record字段仍使用下述TLV；nested field unknown/missing/type mismatch同样拒绝。

`work_policy_id`绑定normalizer/fixed mechanism实际使用的versioned fuel；改变系数而不改变policy id是schema violation。
`AdoptionMode=FixedOptimization|BestEffortCleanup`的observation要求field 14、16均present，分别等于本批
`OptimizationQualificationProposalV1`和下述`OptimizationSetPublicationAttemptV1` digest；其它mode两字段必须absent。field 15对所有mode都required，
引用独立的`AdoptionQualificationRunV1`。static comparison和ABBA每条内嵌digest必须与field 14相同。
`OutcomeCountV1`的`Applied/NoChange/NotApplicable`要求reason absent；`Unsupported/ResourceExhausted/Invalid/Cancelled`
要求reason恰好present，同一outcome+reason只能一行。gate的evidence digest始终required并绑定完整terminal gate record：
`Passed`要求terminal reason absent，`Failed/Skipped/Unsupported/Timeout/Cancelled`要求reason present；Qualified evidence中只允许
Passed。`BackendActionEvidenceV1`按`(invocation_id, action_ordinal)`排序；invocation ID必须all-and-only引用同一aggregate bucket
实际包含的terminal record，同一invocation的action从0连续编号，argv非空。tool和output digest对任何terminal status均required，
空输出也hash成确定Digest32；没有backend action时sequence恰为空，不允许flattened argv/tool/output sentinel。
所有qualification status先满足下文gateway定义的kind结构规则；在此基础上，`Qualified`的`Rewrite`要求aggregate非零rewrite，
`BackendAction`要求至少一条action且全部`Success`，`InvocationOnly`要求非零invocation。`NoOpObserved/DownstreamBlocked/Rejected`
可以没有Qualified所需的nonzero/success证据，但不能违反kind结构或混入另一kind evidence。ExposureSet只表示可调用性，不参与
选择这三种evidence invariant。

`OptimizationConfigurationV1`中AllOn/AllOff要求所属disabled key absent，DisableOne要求key present且属于proposal对应group；wrong-group、
nonmember、同key双disable或其它presence组合拒绝。qualification evidence只接受本文定义的global
`B=(AllOn,AllOn)`/`A=(AllOff,AllOff)`，或恰一所属组DisableOne且另一组AllOn的单key边际pair；global pair只允许出现在
`OptimizationBatchObservationV1`，fixed/cleanup per-key observation只允许其自身key的marginal pair。2×2与production-AllOn gate也只
属于batch；RequiredNormalization/其它mode则按其spec声明的key-local gate解释field 8。任意其它configuration pair只能作为debug sample，
不能进入`Qualified` evidence。

`QualificationCaseKeyV1.rank_count`只接受qualification policy声明的closed rank domain且非零。global comparison固定
`comparison_kind=GlobalAllOffVsAllOn`、disabled mechanism absent、A=两组AllOff、B=两组AllOn；fixed marginal固定
`FixedDisableOneVsAllOn`、disabled mechanism为fixed member、A仅fixed DisableOne(key)且cleanup AllOn、B两组AllOn；cleanup marginal对称。
其它kind/disabled-mechanism/configuration presence拒绝。`StaticComparisonResultV1`精确表示逐component比较B相对A：全部known且至少一项更小为
`BStrictlyBetter`，全部known且全等为`BEqual`，任一known component更大为`BRegressed`，无回退但存在unknown为
`IncomparableUnknown`；validator必须从同registry的vector fresh重算，Qualified只接受前两者。

map-like sequence的canonical key固定为：outcome count按`(outcome, reason optional bytes)`，invocation evidence按
`(invocation-site RegistryRef bytes, cut-point bytes, QualificationCaseKeyV1 bytes)`，work counter按counter registry ordinal，static
comparison按`(proposal digest, OptimizationComparisonKeyV1 bytes)`，gate按`(gate RegistryRef bytes, case key bytes, configuration bytes)`；
这些sequence按key排序且重复key拒绝。backend action与ABBA sample是真sequence，分别保留canonical invocation/action和执行顺序。
validator从`QualificationPolicyRefV1`绑定的mandatory corpus×rank domain fresh生成expected rows：proposal union非空时，batch对每个
`(corpus, rank, Original)`恰一global static/ABBA pair；union为空时该domain为空。每个proposal key的observation对每个相同case恰一自身marginal pair，
all-and-only无缺失/多余。Equivalent-IR gate另对`Original/Metamorphic × cleanup AllOff/AllOn`四个case/configuration生成expected gate key，
fixed恒AllOff；production gate对mandatory original case生成两组AllOn key。gate registry/policy还可声明required-normalizer的
key-local expected rows，但不能用自由附加row补缺。

每个proposal/corpus/rank/comparison的ABBA evidence恰含block index 0..4；每block严格按sequence
`ALeft, BLeft, BRight, ARight`四条，同一`OptimizationComparisonKeyV1`决定A/B configuration，sample不得再复制configuration；
`process_status=Success`。A/B warmup另存runner附件但不进入sample sequence；missing/duplicate/wrong position或多余sample使
evidence invalid；子进程失败/timeout/cancel不进入`ABBASampleV1`，而是直接形成带reason的generic run
`Invalid`/`ResourceExhausted`/`Cancelled` terminal并禁止发布observation。batch proposal/identity/policy必须与全部per-key
observation逐字段相同，qualified set validator再核对其batch/key digest all-and-only closure。

terminal status invariant固定为：`Qualified`要求`closed_reason` optional absent及完整Qualified invariant；`NoOpObserved`要求Resolved、
complete passing evidence但既无strict downstream benefit也无显著host benefit；`DownstreamBlocked`要求明确非本mechanism语义失败的
downstream gate及evidence digest；`Rejected`要求terminal semantic/resource/host/contract failure；`Unassessed`只允许尚未形成
terminal evidence且会阻塞Q33 closure。status/reason不匹配、Qualified带reason或非Qualified缺typed reason均拒绝。
`OptimizationBatchStatusV1=Qualified`要求`closed_reason` optional absent，且global static/ABBA、Equivalent-IR 2×2与
production-AllOn证据按qualification policy的mandatory corpus/rank domain all-and-only完整、全部通过；
`Rejected`要求optional present且含一个typed terminal reason与对应evidence digest，不得被
`QualifiedOptimizationSetV1`引用。
batch status/reason的其它组合、或把per-key marginal evidence写入batch，均是schema validation failure。
`qualification_status`只存在于observation，spec不能预填decision。observation按`MechanismKey` canonical排序且保存实际backend
argv/tool identity。canonical encoding固定为`u16be(field-number) + u8(type-tag) + u32be(payload-size) + payload`，field按number
升序且unknown/missing field拒绝；integer一律big-endian，bool为单byte0/1，closed enum为固定`u32be` ordinal，typed ID使用上文
canonical ASCII grammar，nested record/list递归length-prefix。`ExposureSet`按完整enum bytes排序去重；按key组织的record先按完整
`MechanismKey` canonical bytes排序；ABBA samples、argv与gate execution是sequence，必须保留记录顺序而不能当set排序。

digest固定为：

```text
spec_digest = SHA-256(
  "wafer.optimization-adoption-spec\0" || u16be(schema_version) ||
  u32be(spec_bytes.size) || spec_bytes)

observation_digest = SHA-256(
  "wafer.optimization-qualification-observation\0" || u16be(schema_version) ||
  u32be(observation_bytes.size) || observation_bytes)

optimization_batch_digest = SHA-256(
  "wafer.optimization-batch-observation\0" || u16be(schema_version) ||
  u32be(batch_bytes.size) || batch_bytes)
```

字符串中的NUL是实际domain-separator byte。archive同时保存canonical spec/observation bytes和digest；readback先按schema重编码并
做byte equality，再核对digest，不能hash C++ layout、JSON/printer order、展示字符串或registration ordinal。编码变化必须提升
schema version并更换domain input中的version。

`ProviderOrigin`只陈述来源；`Resolved`只表示当前configured build能解析其library/tool identity；
`ExposureSet`是可同时拥有多项的typed set。`LibraryCallable`只陈述utility对compiler library可见，
`BackendExecutable`只陈述可启动受控toolchain command，二者都不证明production已调用。
以下表是五个底层字段的derived maturity view，不是第二套可写状态机：

| 状态 | 含义 | 能否宣称已采用 |
| --- | --- | --- |
| resolved | `Availability=Resolved` | 否 |
| registered/replayable | `DebugRegistered ∈ ExposureSet` | 否 |
| library/backend-callable | `LibraryCallable ∈ ExposureSet` 或 `BackendExecutable ∈ ExposureSet` | 否 |
| production-consumed | named production policy选择exact spec且有live telemetry；fixed/cleanup还必须由active qualified set引用 | 仍需效果证据 |
| qualified | 满足下述完整`Qualified` invariant | 是 |

`QualificationStatus=Qualified`是derived decision，不是单独一个足够条件。validator必须同时证明：spec
`Availability=Resolved`、`AdoptionMode!=None`；observation的spec/work/qualification-policy digest全部匹配并满足下文run/result
manifest closure；该key在其唯一
cut/domain有非零live invocation；所有mandatory corpus/gate有terminal pass且无missing/skipped/unsupported/timeout；
所有status先通过EvidenceKind结构validator，Qualified的`Rewrite`有非零rewrite，`BackendAction`有非零成功action及output evidence，
`InvocationOnly`有非零invocation；required/fixed/cleanup机制满足各自exact static/host规则；
outcome/work counters与实际invocation闭合。任一条件不满足只能是`Unassessed/NoOpObserved/DownstreamBlocked/Rejected`之一，
不能被手写`Qualified`绕过。production fixed/cleanup set只消费通过这套validator的immutable observation。

gateway的exactly-once observation seam使用下列唯一terminal record；它是invocation-local/qualification archive value，不是IR
attr、artifact sidecar或跨pass analysis：

```text
InvocationScopeKindV1 = ProductionCompile | DebugReplay | QualificationRun

InvocationIdentityV1 {
  scope_kind: InvocationScopeKindV1
  scope_digest: Digest32
  mechanism_key: Record(MechanismKeyV1)
  invocation_site: Record(RegistryRefV1)
  cut_point: ClosedEnum(OptimizationCutPointV1)
  invocation_ordinal: U64
}

InvocationTelemetryV1 {
  schema_version: U16 = 1
  identity: Record(InvocationIdentityV1)
  qualification_case: Optional<Record(QualificationCaseKeyV1)>
  spec_digest: Digest32
  input_snapshot_digest: Digest32
  outcome: MechanismInvocationOutcomeV1
  terminal_reason: Optional<Record(ClosedReasonV1)>
  rewrite_count: U64
  work_summary: Record(WorkSummaryV1)
  backend_actions: Sequence<BackendActionEvidenceV1>
}
```

两个record按声明顺序使用本节TLV规则；canonical invocation ID为
`SHA-256("wafer.optimization-invocation\0" || u16be(1) || u32be(identity-bytes.size) || identity-bytes)`。
terminal digest为
`SHA-256("wafer.optimization-invocation-terminal\0" || u16be(1) || u32be(telemetry-bytes.size) || telemetry-bytes)`，其中
telemetry bytes是完整`InvocationTelemetryV1` canonical record；result manifest必须同时核对invocation ID与terminal digest。
`QualificationRun`要求qualification case present且scope digest等于下述`AdoptionQualificationRunV1` digest；其它scope要求case absent。
caller在并行dispatch前按current artifact canonical traversal预留连续invocation ordinal，不能按线程完成顺序分配。同一scope中
identity bytes/ID必须唯一。

gateway先验证spec/domain并建立input snapshot，然后调用owner；begin状态不对archive可见。owner正常返回、typed failure、
subprocess failure或cancellation后，gateway构造恰一个terminal record，在单次append/commit中发布；相同invocation ID的第二次terminal
commit、无begin的terminal或terminal后继续追加action/work一律拒绝。`Applied/NoChange/NotApplicable`要求terminal reason absent，
其它outcome要求present；backend action的invocation ID必须等于outer identity bytes按上述公式得到的digest。进程在terminal commit前崩溃只留下
不完整run，不得由聚合器猜测outcome或发布observation。

gateway从匹配spec读取`EvidenceKindV1`并对**所有status**执行结构规则：`Rewrite`要求action sequence为空，且`Applied`当且仅当
rewrite count非零；`BackendAction`要求rewrite count为零；`InvocationOnly`要求rewrite count为零、action sequence为空且outcome
不得为`Applied`。这些规则也从terminal fresh聚合到observation，非Qualified状态不能携带跨kind证据；Qualified再叠加下文的
nonzero/success门槛。

`QualificationObservation`的outcome、`InvocationEvidenceV1`、work和backend action只能从同一run全部committed terminal records
按canonical identity归并：invocation count为record数、rewrite/work为checked sum、backend actions按identity/action连接；任何expected
invocation缺terminal、重复ID、spec/case/cut不匹配或aggregate与records不等都使run不完整。这样aggregate不是第二个可独立写入的事实源。

所有纳入inventory的production/debug机制调用必须经过`OptimizationInvocationGatewayV1`：gateway接收registered
`MechanismKeyV1`、当前typed cut/root和policy ref，校验spec domain后才调用owner adapter，并原子发出outcome/work/action
telemetry；backend argv也只能由对应backend adapter在gateway内启动。为防“绕过gateway所以telemetry看不见”，build-time
checker从同一registry生成known upstream pass factory/utility/backend launcher symbol表和唯一allowed adapter symbol表，检查
production/debug pipeline translation units的AST/call graph及link dependency：known symbol在adapter外有direct call、adapter调用
未带key、live gateway key无spec、active spec无invocation site都会失败。新增第三方机制时必须先扩registry/checker rule；source
path只可出现在审计diagnostic，不成为runtime key。runtime bidirectional telemetry closure与该static direct-call closure必须同时
通过，不能用其中一个替代另一个。

`wafer-opt`应注册用于诊断的Linalg、Tensor、SCF、Bufferization和generic Transform pass families，但执行入口必须是带
`MechanismKey`的gateway-wrapped adapter/pass instrumentation，不能让raw upstream pass绕过inventory telemetry；注册只扩大debug
replay能力，不改变production policy。固定pipeline只列入qualified机制；candidate-local调用必须复用upstream interface/
pattern utility并叠加Wafer precondition和exact gates，不复制op-pair matcher。SCCP、LICM或其它pass若在当前artifact上无改写，
保留为debug能力即可。

generic CSE可以合法地共享`tensor.empty`、fill、DPS init或常量producer，也可能延长whole-tensor lifetime并改变SPM/spill
选择；因此它当前首先是等价IR压力与candidate-local share-vs-recompute mechanism，不预设为fixed whole pass。下游semantic recovery、bufferization与SPM/DDR
lifetime必须从type、DPS tie、SSA use-def、ViewLike/structured semantics重算，不能要求“init恰好由某个直接fill producer定义”。
standard fold/specialization产生的新合法structured form若尚无consumer，应由required normalizer转回受支持的语义形态或在
adoption gate明确拒绝该pass；不能静默依赖原始producer拓扑。

production/qualification使用一个冻结成员集，不能在运行时把“当前registry里碰巧注册的机制”解释成`AllOn`：

```text
MechanismSpecBindingV1 {
  mechanism_key: Record(MechanismKeyV1)
  spec_digest: Digest32
}

MechanismObservationBindingV1 {
  mechanism_key: Record(MechanismKeyV1)
  observation_digest: Digest32
}

OptimizationQualificationProposalV1 {
  schema_version: U16 = 1
  fixed_bindings: SortedSet<Record(MechanismSpecBindingV1)>
  cleanup_bindings: SortedSet<Record(MechanismSpecBindingV1)>
}

QualifiedOptimizationSetV1 {
  schema_version: U16 = 1
  proposal_digest: Digest32
  batch_observation_digest: Digest32
  observation_bindings: SortedSet<Record(MechanismObservationBindingV1)>
}
```

四个record的field number均按声明顺序从1连续编号，并使用本节同一TLV/type-tag/Optional/SortedSet规则；binding record所有字段
required，binding set按完整`MechanismKeyV1` owner bytes排序且重复key拒绝。proposal两组各自可空，union也可空，且两组key互斥；
空union表示探索性证据已在发布前关闭全部可选机制，是合法且唯一的空proposal编码。
前者每row的spec必须`AdoptionMode=FixedOptimization`，后者必须`AdoptionMode=BestEffortCleanup`。每个spec digest必须从同一archive
canonical spec bytes fresh重算，不能只信binding。

proposal canonical body是三个field的TLV bytes，standalone bytes固定为
`"wafer.optimization-qualification-proposal\0" || u16be(1) || u32be(body-size) || body`；proposal digest是该standalone bytes的SHA-256。
qualified set body同理，standalone bytes固定为
`"wafer.qualified-optimization-set\0" || u16be(1) || u32be(body-size) || body`并取SHA-256。SortedSet payload统一为
`u32be(count)`后逐项连接`u32be(record-size) || record-canonical-bytes`，不使用host tuple layout；empty set只能编码count=0。

qualified set的observation bindings必须all-and-only覆盖proposal两个group的canonical union，每个observation都绑定同一proposal
digest、qualification run和publication attempt并通过完整`Qualified` invariant；batch observation唯一且也绑定同一
proposal/run/attempt。production policy、
telemetry和qualification configuration均携带相应digest，production只能消费已由下述active reference原子选择的qualified set。
当前inventory中的canonicalizer callsite按真实责任分组：为静态offset legality建立required form的两个固定cut属于
`RequiredNormalization=AlwaysOn`，required form之后的额外尝试才属于cleanup组（当前数量只写入observation，不是schema常量）。
每个cut point使用独立key/spec，不与fixed组或其它required normalizer重复计数。

qualification只允许内部构建器使用下面的正交配置，不形成CLI、公开pass option或第二条production pipeline：

```text
RequiredNormalization = AlwaysOn
FixedOptimization = AllOn | AllOff | DisableOne(FixedMechanismKey)
BestEffortCleanup = AllOn | AllOff | DisableOne(CleanupMechanismKey)
```

通用qualification run、只属于fixed/cleanup optimization的publication attempt，以及production active ref分别由以下typed records
唯一表达。通用run不携带proposal或active-set CAS事实，因此RequiredNormalization、CandidateLocal、TargetBackend和inventory
observation可以独立重放；只有fixed/cleanup batch才建立publication attempt：

```text
QualificationInputCaseV1 {
  case_key: Record(QualificationCaseKeyV1)
  input_snapshot_digest: Digest32
}

InvocationTerminalBindingV1 {
  invocation_id: Digest32
  terminal_digest: Digest32
}

AdoptionQualificationInputV1 {
  schema_version: U16 = 1
  spec_bindings: SortedSet<Record(MechanismSpecBindingV1)>
  qualification_cases: SortedSet<Record(QualificationInputCaseV1)>
  optimization_proposal_digest: Optional<Digest32>
}

AdoptionQualificationRunV1 {
  schema_version: U16 = 1
  qualification_input_digest: Digest32
  qualification_identity: Record(QualificationIdentityV1)
  qualification_policy: Record(QualificationPolicyRefV1)
  run_series_ordinal: U64
  attempt_ordinal: U32
}

AdoptionQualificationRunOutcomeV1 = CompletedEvidence | Cancelled |
                                    HostEnvironmentInvalidated | ResourceExhausted | Invalid

AdoptionQualificationRunTerminalV1 {
  schema_version: U16 = 1
  qualification_run_digest: Digest32
  outcome: AdoptionQualificationRunOutcomeV1
  result_manifest_digest: Optional<Digest32>
  closed_reason: Optional<Record(ClosedReasonV1)>
}

AdoptionQualificationResultManifestV1 {
  schema_version: U16 = 1
  qualification_run_digest: Digest32
  invocation_terminals: SortedSet<Record(InvocationTerminalBindingV1)>
  observation_bindings: SortedSet<Record(MechanismObservationBindingV1)>
  optimization_batch_observation_digest: Optional<Digest32>
}

OptimizationSetPublicationAttemptV1 {
  schema_version: U16 = 1
  qualification_run_digest: Digest32
  proposal_digest: Digest32
  expected_active_ref_digest: Optional<Digest32>
}

OptimizationSetPublicationOutcomeV1 = QualifiedPublished | RejectedEvidence | PublicationConflict |
                              PublicationFailed

OptimizationSetPublicationTerminalV1 {
  schema_version: U16 = 1
  optimization_publication_attempt_digest: Digest32
  outcome: OptimizationSetPublicationOutcomeV1
  batch_observation_digest: Optional<Digest32>
  candidate_set_digest: Optional<Digest32>
  closed_reason: Optional<Record(ClosedReasonV1)>
}

ActiveQualifiedOptimizationSetRefV1 {
  schema_version: U16 = 1
  generation: U64
  parent_set_digest: Optional<Digest32>
  set_digest: Digest32
  qualification_run_digest: Digest32
  adoption_qualification_run_terminal_digest: Digest32
  optimization_publication_attempt_digest: Digest32
  optimization_publication_terminal_digest: Digest32
}
```

record按声明顺序使用本节TLV规则。七类standalone domain分别为
`wafer.adoption-qualification-input\0`、`wafer.adoption-qualification-run\0`、
`wafer.adoption-qualification-run-terminal\0`、`wafer.adoption-qualification-result-manifest\0`、
`wafer.optimization-publication-attempt\0`、`wafer.optimization-publication-terminal\0`和
`wafer.active-qualified-optimization-set-ref\0`；canonical bytes统一为
`domain || u16be(1) || u32be(body-size) || body`，digest均为这些完整bytes的SHA-256。
input的spec/case set均非空，分别按完整mechanism key与case key bytes排序且重复拒绝；每个case的`input_snapshot_digest`绑定
该case all-and-only typed inputs/payload/config，不能用路径或case名恢复。spec bindings必须all-and-only覆盖该run按named
qualification pipeline实际会调用的全部mechanism，包括always-on RequiredNormalization、下游gate/backend及受控fixed/cleanup。
optimization proposal absent表示通用独立run；present时proposal union必须恰等于input中`AdoptionMode=FixedOptimization |
BestEffortCleanup`的subset并逐spec digest相同，input中的required/其它mode bindings仍保留且不进入proposal/qualified set。
qualification cases必须恰为policy mandatory corpus×rank×input-variant域。`qualification_input_digest`只等于这份完整input
standalone bytes的SHA-256，不等于proposal digest。

每个`QualificationObservation.qualification_run_digest`和batch field 11必须引用对应run；observation field 4/5及batch field 3/4
必须分别与run的qualification identity/policy逐字段相同，observation case/invocation/static/ABBA rows必须all-and-only来自input
cases。fixed/cleanup observation的field 14必须等于input的present proposal digest，field 16及batch field 12还必须引用同一
publication attempt；attempt的run/proposal也必须逐项相同。同一optimization run中的其它adoption mode observation field 14/16必须
absent，即使input自身含proposal；它们仍通过同一run/result manifest封存，不进入proposal或qualified set。只有input proposal
absent的通用run才完全不创建publication attempt。

journal对同一input/identity/policy以原子checked counter分配从0连续递增的`run_series_ordinal`；同一
series内首次attempt为0，只有host 环境失效重试递增`attempt_ordinal`。因此显式重放得到新series，host 环境失效重试仍可被budget精确计数，二者不会
产生run ID碰撞；不得按wall clock、PID或随机nonce恢复身份。result manifest的invocation bindings按invocation ID排序唯一，
terminal digest必须fresh匹配对应`InvocationTelemetryV1`；observation bindings按mechanism key排序唯一并all-and-only覆盖input
spec bindings；每个terminal的qualification case必须命中input唯一case且`input_snapshot_digest`相同，observation digest必须fresh
匹配同run/identity/policy。optimization input要求batch digest present且匹配同run/attempt，
非optimization input要求absent。expected invocation identity由input/policy生成，manifest必须all-and-only覆盖，不能删掉失败terminal。

每个run恰有一个可达terminal：`CompletedEvidence`要求result manifest present、reason absent；
`Cancelled/HostEnvironmentInvalidated/ResourceExhausted/Invalid`要求manifest absent、reason present且不得产生observation、batch、
candidate或publication terminal。run terminal与result manifest在同一archive transaction提交并永久seal该run scope；seal后任何
invocation terminal、observation、batch或第二个run terminal append一律拒绝。只有`CompletedEvidence` run可以继续publication，
并且optimization input必须恰有一个同run/proposal的publication attempt。publication attempt bytes/digest可以在runner内预计算供
observation引用，但仅随CompletedEvidence archive transaction变为可达；失败run不会留下一个无terminal的可达publication attempt。

active generation从1开始；generation=1要求parent absent，generation>1要求parent present且等于CAS前active ref中的set digest。
初次publication attempt要求expected active ref absent；已有active时该field required并等于其完整ref standalone bytes digest。
`QualifiedPublished`要求batch/candidate present、reason absent，batch status Qualified且candidate digest指向完整验证的
`QualifiedOptimizationSetV1`；`RejectedEvidence`要求batch present、candidate absent、reason present且batch status Rejected；
`PublicationConflict`要求已完成的Qualified batch与candidate digest均present且reason present；`PublicationFailed`要求batch
present、reason present，且candidate仅在batch Qualified时present。每个可达publication attempt恰有一个可达terminal，同attempt
第二个terminal或terminal缺失均使archive无效。其它presence组合拒绝。active ref的
四个run/publication digest必须形成同一条CompletedEvidence→QualifiedPublished链；它不能引用Rejected或Conflict terminal。

production固定`RequiredNormalization=AlwaysOn`、两组`AllOn`；cleanup只能改变非必要清理结果。
Equivalent-IR 2×2的两轴固定为`EquivalentInput={Original, Metamorphic}` ×
`BestEffortCleanup={AllOff, AllOn}`，
四条路径固定`FixedOptimization=AllOff`，都先运行required normalizer及其postcondition verifier，cleanup只能在该verifier通过后运行，
不得修复或掩盖required-form缺口；另比较normalizer运行一次与两次后的canonical bytes、outcome和work
summary：两次后canonical bytes必须相同，第二次必须是`Success(changed=false)`，两次各自的work
summary在重放时确定，但不要求第一/二次计数相等。proposal union非空时，rank-count=1/16和冻结7B corpus
单独比较两组同时AllOn与同时AllOff；union为空时两种configuration语义相同，global static/ABBA comparison
domain固定为空，禁止为identity comparison制造样本；Equivalent-IR 2×2、normalizer once/twice和最终production gate仍完整执行。
逐mechanism marginal comparison只在其所属组使用`DisableOne(k)`，另一组保持AllOn；最终production set还需以两组AllOn重放完整downstream
gates，不能用2×2替代production组合验证。

`QualificationPolicyRefV1`所绑定的首个fixed/cleanup policy definition至少冻结下列字段；数值cap属于canonical policy bytes，
不由runner补default，当前implementation row在注册时选择宽松但有限的checked值：

```text
OptimizationSetQualificationPolicyV1 {
  schema_version: U16 = 1
  mandatory_corpus_registry: Record(RegistryRefV1)
  mandatory_rank_counts: SortedSet<U32>
  static_metric_registry: Record(RegistryRefV1)
  abba_block_count: U32 = 5
  wall_guard_fraction_numerator: U32 = 1
  wall_guard_fraction_denominator: U32 = 20
  mad_multiplier: U32 = 3
  host_environment_retry_cap: U32
  total_process_launch_cap: U64
  total_gateway_invocation_cap: U64
}
```

rank set非空且值非零，fraction denominator非零，两个total cap必须足以覆盖host 环境稳定时的 mandatory run并使用checked算术验证；否则
policy registry拒绝。host environment retry cap计**首次attempt之后**允许的新attempt数，所有attempt的process launch/gateway
invocation共同消费两个total cap，不能每次污染后重置。改变corpus/rank/metric registry、sample数、阈值或任一cap都产生新policy
digest。definition按声明顺序使用本节TLV规则，standalone bytes为
`"wafer.optimization-set-qualification-policy\0" || u16be(1) || u32be(body-size) || body`；
`QualificationPolicyRefV1.canonical_policy_digest`必须对这些bytes取SHA-256并readback一致。

proposal union非空时，`OptimizationSetQualificationPolicyV1`以`B=(Fixed AllOn, Cleanup AllOn)`、全局比较
`A=(Fixed AllOff, Cleanup AllOff)`；union为空时没有global comparison。机制`k`的marginal comparison只在所属组
`DisableOne(k)`。Release样本要求同机、同compiler/toolchain identity、build/config/corpus和thread count；runner记录
host/toolchain/kernel、实际CPU affinity、governor可观察状态及并发负载检测结果并纳入observation identity，但qualification
不要求管理员权限去修改governor、取得独占机器或预先存在专用runner。A/B block期间任一identity事实改变、affinity漂移或检测到
其它compiler/model竞争负载时，runner为当前run原子写入
`AdoptionQualificationRunTerminalV1(HostEnvironmentInvalidated)`并整组重跑，
不形成result manifest/`QualificationObservation`/batch/candidate set，不能删除单点或扩大guard。每次重跑使用递增attempt ordinal并保持其它run
字段不变，并fresh读取active ref形成新的publication attempt；超过 host environment retry cap或任一total work cap时写入
`AdoptionQualificationRunTerminalV1(ResourceExhausted)`并停止，active set保持byte-identical。
driver cancellation写入Cancelled且不自动重试；同一稳定状态下的
普通非特权runner即可形成有效内部证据。每个样本用新子进程执行；
wall使用monotonic clock，peak RSS取该子进程的OS high-water。A/B先各warm-up一次且不计样本，再运行
固定5个`A,B,B,A`
block。每个block对指标`x`计算`a_i=median(A_left,A_right)`、`b_i=median(B_left,B_right)`和
`d_i=b_i-a_i`；`m_A=median(a_i)`、`m_d=median(d_i)`、`MAD=median(abs(d_i-m_d))`，固定
`guard=max(0.05*m_A, 3*MAD)`。wall和peak RSS分别要求`m_d <= guard`；任一超限即拒绝。显著host
收益定义为至少一项`m_d < -guard`且另一项不回退。所有median对偶数样本取中间两值的checked
arithmetic mean；实现全程使用canonical reduced exact rational（arbitrary-precision signed numerator、positive denominator），
偶数median精确为`(x+y)/2`，不做integer或floating rounding。`0.05*m_A`按`m_A/20`、`3*MAD`按exact integer multiply，
比较用符号正确的cross multiplication；只在human diagnostic展示时round，展示值不回流decision。`MAD=0`不增加额外容差。

每个fixed/cleanup mechanism必须有按`MechanismKey`归属的非零invocation，并按spec的`EvidenceKindV1`分别满足非零rewrite、
backend-action的零rewrite与非零成功action/output，或invocation-only的零rewrite/空action。所有机制都要求完整纵向等价和exact static
DDR/SPM/movement/command vector逐分量不恶化：全局要求`B <= global A`，机制`k`要求
`B <= DisableOne(k)`，比较时unknown不当作0或passing。`DeterministicDownstreamBenefitV1(k)`精确定义为所有mandatory
sample的每个registered component均known且满足上述不恶化，并至少一个sample/component严格更小；只有全相等时不能称为下游收益。
required-normalization与拟标记`Qualified`的fixed/cleanup row
在其declared operation domain的mandatory corpus中任一unsupported、skipped、timeout或缺样本都使qualification失败；
production required及proposal内全部fixed/cleanup rows的mandatory纵向任一上述结果使Q33 closure失败。明确标记`Rejected`或
`DownstreamBlocked`的非production row可以保留closed reason并完成inventory，但不能宣称采用。没有确定性下游
收益且没有上述显著host
收益的机制固定记录为`QualificationStatus=NoOpObserved`；被评估spec仍保留原`AdoptionMode`和完整正交ExposureSet，不回写
immutable spec digest。`DebugRegistered ∈ ExposureSet`且该spec/key不属于任何active `QualifiedOptimizationSetV1`共同形成
derived debug-only maturity；LibraryCallable/BackendExecutable等真实exposure不能为此删除。它不进入production set。

资格发布是set级原子transaction：先冻结`OptimizationQualificationProposalV1`及全部spec digest，并读取当前
`ActiveQualifiedOptimizationSetRefV1`（首次为absent），构造含proposal、proposal fixed/cleanup subset、named qualification pipeline
会调用的其它all-and-only spec bindings及mandatory case/input snapshots的`AdoptionQualificationInputV1`，再以其digest构造通用run及引用该run/proposal/expected-active基线的publication
attempt。global AllOn/AllOff、Equivalent-IR 2×2和
最终production-AllOn gates只写入唯一`OptimizationBatchObservationV1`；每个key的`QualificationObservation`只保存该key invocation、
DisableOne marginal static/ABBA和per-key gate，不能复制或任选一个key承载global证据。完成batch、逐key和mandatory vertical后，
先原子发布all-and-only result manifest与`AdoptionQualificationRunTerminalV1(CompletedEvidence)`并seal run；batch Rejected时只发布
`OptimizationSetPublicationTerminalV1(RejectedEvidence)`，不得构造candidate或执行CAS。batch Qualified时，在不可见staging中写入并
重新parse/canonicalize全部spec、terminal invocation records、observations、batch及候选`QualifiedOptimizationSetV1`，逐digest/
all-and-only readback后发布immutable set directory；随后在同一no-replace/CAS transaction中准备
`OptimizationSetPublicationTerminalV1(QualifiedPublished)`和同时引用run/run-terminal/attempt/set/publication-terminal digest的
generation+1 active ref，对attempt绑定的expected active-ref digest做单次CAS。该一次active-ref替换使set与QualifiedPublished
terminal同时可达；CAS不匹配时隐藏terminal和prepared ref均不可达并被丢弃，只发布恰一个`PublicationConflict` terminal，保留
当前active ref，不得覆盖另一并发run。batch完成后的验证/写入失败形成恰一个`PublicationFailed` terminal；run未形成完整batch前
失败则只形成generic Cancelled/ResourceExhausted/Invalid terminal。上一份qualified production set始终byte-identical。新增、移除、
改变spec或把row在fixed/cleanup间移动都会产生新set digest并要求整批重放，不能在一次失败后静默删掉该key继续沿用其它旧
observation。

active ref永远只指向已发布且full readback通过的immutable set。crash发生在CAS前时active ref仍指向旧set，startup按run journal清理
不可达staging；CAS原子替换成功后set已完整存在。publication和audit路径必须重读全部spec、observation、manifest binding与terminal pack。
production路径读取完整ref、small run ownership records，以及set目录内由digest闭合的proposal/batch/publication terminal和proposal
成员spec/observation；其工作量只随active proposal大小增长，不随历史invocation terminal数量增长。terminal pack不携带production
membership，且在CAS前已经full readback，因此不进入每个compiler进程的policy selection。任一operational member缺失、stale、非Qualified
或digest不匹配立即fail closed，也不扫描目录寻找“最新”set。旧generation只有不再被任一active reader引用后才可best-effort GC，
GC失败不影响active语义。

## 7. 当前支持面与限制

当前evidence覆盖：

- 2D dot/matmul与rank-4 attention contractions；
- elementwise、broadcast、static shape views与concatenate；
- basic reductions及staged softmax/norm/RoPE/MLP graphs；
-上述五类StableHLO logical collective；
- real PyTorch/XLA data/column/row sharding helper输出进入structured tensor program。

这些证据只证明local structured IR和logical collective handoff，不证明：

- 任意StableHLO family都可lower；
- dynamic shape candidate已闭合；
- transformer所有activation/quantization变体；
- tile shape、physical layout、SPM/DDR或instruction legality；
- physical collective transport、target artifact、runtime或numeric output。

遇到硬件/ABI本可表达但当前official conversion或Wafer interface缺失的semantic，应扩本stage表示与verifier或记录
后续任务，不能把下游缺口反写成frontend长期不支持。

## 8. 实现与调试入口

当前production driver实际调用：

```text
buildStablehloToLinalgPipeline
  = StableHLO collective normalization [0]
  + official StableHLO legalize-to-Linalg
  + StableHLO collective normalization [1]
  + required post-legalization canonicalization
  + StableHLO collective normalization [2]
  + required structured-tensor canonicalization
  + explicit required structured normalization
  + qualified fixed target-independent optimization
```

两个canonicalization分别绑定`PostLegalizationCanonicalization`和`StructuredTensorCanonicalization`，属于required invocation
全集而不是best-effort cleanup。当前qualified optional proposal为空，因此builder不会追加`StablehloCleanup`或
`StructuredTensorCleanup`。production从`qualification/optimization-adoption/active-ref.bin`选择generation 1的immutable set；
最终run digest为`d09b6c3cc9631b7ca1abdfa0bbe9f7e787b631ff65c5dd22fdc2cf8e100d0e88`，set digest为
`161242b7ce46a3590122fb067c3bc3cfd3ea0bfefa8577c591b6d23f26befde2`。该run以21个隔离子进程覆盖source rank1/rank16和冻结7B
rank16的Equivalent-IR 2×2、normalizer once/twice及production-AllOn，共聚合745141个canonical terminal；source输出exact，7B
16-rank PyTorch differential最大绝对误差`0.0029296875`，且SystemC执行19696个target transaction。完整archive readback通过后才发布active ref。

registered `wafer-lower-stablehlo-to-linalg`只为显式MLIR replay和unit tests提供相同body。helper与program
directory orchestration由`wafer-compile`负责，用户不选择该stage或手工续接调度passes。

production与debug入口必须调用同一pipeline builder；debug driver额外注册的upstream pass families不能被用户拼成第二条
production pipeline。新增fixed pass前要先补全其producer/alias/effect等价合同和纵向gate；新增candidate mechanism则进入
06共享utility/provider，不塞入本pipeline尾部。

实现入口可以拆pattern/pass，但长期合同是输入/输出IR与legality，不是pass名。创建
`wafer.linalg_ext.collective.*`的pass必须声明dependent dialect；official conversion pin变化时要重跑coverage，
不能依赖进程中偶然注册的dialect。

## 9. 验证

必须覆盖：

- official conversion后raw StableHLO为零；
- dot/batch/contracting/indexing、broadcast与reduction关系；
- static shape-view与constant residual cleanup positive/negative；
- 五类collective的shape、axis、DPS ties、rank group/rank groups与combiner verifier；
- logical rank越mesh范围、invalid replica groups、shape mismatch与unsupported collective fail closed；
- output不含SDY、physical layout/memory、DTE、packet或runtime facts；
- `wafer-compile`从真实post-SPMD program继续形成并重新verify structured tensor program。
- metamorphic equivalent-IR corpus至少覆盖共享/非共享`tensor.empty`与fill、DPS init、named/generic structured op、
  collapse/expand/transpose/extract-slice view链及合法specialization；各形态进入下游后语义、alias/effect和完整输出一致；
- 每对equivalent input都按`Original/Metamorphic × BestEffortCleanup AllOff/AllOn`四路重放，
  `FixedOptimization=AllOff`且required normalizer始终开启；global、per-key与production-AllOn证据分别只进入
  batch/per-key observation规定的owner；
  once/twice后canonical bytes相同，第二次`Success(changed=false)`，每次work summary重放稳定；
- `UnsupportedSemantic`、`InvalidIR`、`ResourceExhausted`和`InternalInvariant`按canonical first diagnostic分类，
  任一失败都不修改source transaction root；work-policy checked overflow和fuel exhaustion有独立negative；
- 每个qualified upstream mechanism都必须证明pass实际执行且至少一个case发生预期改写；只注册、只统计op数量或只有
  isolated FileCheck不算production采用；
- fixed pipeline改动必须重放rank-count=1/16通用source和7B source-to-package/SystemC/PyTorch gate，并报告编译资源/性能
  按`OptimizationSetQualificationPolicyV1`非回退；production subset的mandatory case不允许
  unsupported/skipped/timeout/缺样本。
  candidate-local机制的scale gate由06/16拥有。

显式IR FileCheck证明local conversion；只有Q15 unified driver消费真实program directory/helper output并发布verified
structured tensor program，才能证明本stage接入主线。06拥有终态candidate/bundle scheduling，当前Q29只作迁移baseline；Q20/Q21拥有固定
CPU expected corpus和纵向
workload completion，不能由本stage测试代替。

## 10. 参考材料

- MLIR Canonicalization：<https://mlir.llvm.org/docs/Canonicalization/>；canonicalizer是best-effort，pipeline不能依赖它保证正确性。
- MLIR Linalg Dialect：<https://mlir.llvm.org/docs/Dialects/Linalg/>；structured interface支持参数化tiling与producer-consumer fusion。
- MLIR Transform Dialect：<https://mlir.llvm.org/docs/Dialects/Transform/>；用于细粒度编排，不替代pass/pattern infrastructure。
