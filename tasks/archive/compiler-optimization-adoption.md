# Compiler Optimization Adoption 实施计划

状态：completed historical plan；Q33 `compiler-optimization-adoption`已于2026-07-19完成。
本文只拆施工checkpoint、验证和退出门槛；
长期pipeline/优化分层由01、05-13、16和18拥有，动态状态只看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  当前verified post-SPMD StableHLO与official legalization产生的rank-local Linalg/Tensor/SCF/Arith/Math program，
  以及现有candidate/finalization、target backend和rank-count=1/16 source/model baselines。
- Current stage responsibility:
  建立全compiler upstream pass/utility adoption inventory；修复structured semantic recovery、DPS init、view/alias/effect/
  lifetime对偶然producer拓扑和optional generic cleanup的正确性依赖；资格化required normal form、fixed target-independent
  optimization和candidate-local mechanism边界，并把通过验证的fixed subset接入现有named production pipeline。
- Output artifact / IR:
  optimizer-ready rank-local structured tensor program、显式required-normalization postcondition、可重放的adoption records，
  以及供后续physical-dataflow planner复用的typed upstream mechanism边界。adoption record不是IR/package sidecar。
- Downstream consumer:
  Q32.I SemanticOpDescriptor/TargetImplementationProvider、Q32.M policy-free rewrite mechanisms、Q32.S bounded search，
  以及09/12/16 exact resource和equivalent-IR gates。
- User-level driver / named pipeline:
  production仍只经wafer-compile；wafer-opt注册完整相关upstream debug pass families用于replay，但不能由用户手拼成
  第二条production pipeline。
- Explicit non-goals:
  不建立Wafer -O3 pass soup，不向pinned XLA helper加入general HLO optimizer，不把whole-tensor CSE/fusion/pack/vector/
  loop scheduling无条件放入fixed pipeline，不改变physical layout/implementation/residency，不实现Transform control plane，
  不因upstream存在某个pass就承诺采用。
- Completion gate:
  adoption inventory覆盖全部compile cut points；required tensor normalizer的支持性不依赖optional generic cleanup；mandatory equivalent-IR
  families进入同一production consumer并通过complete rank/all-rank、SPM/DDR/completion/ABI及source-to-SystemC/PyTorch gate；
  每个qualified fixed mechanism在真实corpus按typed EvidenceKind产生所需rewrite/action/invocation证据且资源/host性能不回退。
  无效、no-op、blocked或rejected机制有
  明确结论，不能伪装成已采用。
```

## 2. 施工原则

1. 优化单位是IR cut point、pre/postcondition和consumer gate，不是pass名字。
2. provider origin、availability、exposure、adoption mode和qualification按05的五轴正交记录；
   resolved/registered/debug-replayable/library-callable/backend-executable都不等于production采用。
3. required tensor normalizer用确定性rewrite与postcondition verifier拥有自身正确性；两个位置固定的canonicalization分别登记为
   required mechanism，其余generic canonicalizer调用只能作为optional cleanup。
4. whole pass只有全部rewrite都满足fixed合同才可进入默认pipeline；否则抽取安全pattern subset，或作为candidate-local mechanism。
5. 任何改变SSA sharing、alias、effect、allocation root或lifetime的改写都使旧analysis失效，并从当前IR fresh重算。
6. 每个机制先证明实际发生调用；按spec的`EvidenceKindV1`分别证明非零改写、零rewrite且非零成功backend action/output，或
   invocation-only的零rewrite/空action，随后再证明
   下游接受和数值正确；只跑pass exit 0、op count或手写fixture不算采用。

## 3. Checkpoints

### Checkpoint A：全 pipeline adoption inventory 与 fresh baseline

审计frontend/SPMD、StableHLO-to-Linalg、structured optimization、candidate materialization、function-boundary finalization、
instruction/target lowering和device publication的全部pass、pattern和library utility。每条record至少包含：

`AdoptionSpec`与`QualificationObservation`的all-and-only schema、五轴enum、work-policy binding和canonical serialization只由
05 §6.2拥有；本计划实现并验证该schema，不复制字段子集。schema validator必须拒绝unknown/missing field、spec/observation
key或digest不匹配及work-policy id漂移。

状态组合由schema validator固定：`Availability!=Resolved`时不得产生`Qualified`；`AdoptionMode=None`
时不得宣称production-consumed；`Qualified`必须有与spec digest一致的live invocation和全部mandatory gate。
所有status先复用05统一EvidenceKind结构validator：Rewrite无backend action，BackendAction零rewrite，InvocationOnly零rewrite/
空action/零Applied；`Qualified`再要求Rewrite在mandatory corpus有非零预期rewrite、BackendAction有成功action/output evidence，
InvocationOnly有非零invocation。不能从adoption mode或ExposureSet猜evidence类别。
`NoOpObserved/DownstreamBlocked/Rejected`不得进入production fixed subset。qualification只存在
observation，`AdoptionSpec`不复制一份decision。

代码typed registry是唯一**audit spec/index**，不是live implementation或已采用状态的替代事实源；production不读取
registry row，调用点与registry只共用单点`MechanismKey`定义并发出telemetry。source/build checker必须双向验证：

- 每个live invocation key都恰有一个spec，cut point/domain一致；
- 每个active spec都有实际invocation observation，或按05唯一状态机闭合为
  `Rejected/DownstreamBlocked + ClosedReasonV1`；不另造`Unavailable`状态；
- 同key不能覆盖多个cut point，spec变更必须改变spec digest；
- backend row从实际执行argv/tool identity生成observation，不用registry重写一份flag。

harness按`MechanismKey`生成canonical sorted observation，record不进入production IR/artifact。Q33完成归档保存
canonical observation本身、spec/build/corpus digest和replay command；不只保存无法审阅样本的单一digest。首轮必须
逐cut point登记official StableHLO-to-Linalg、OneShotBufferize、`TilingInterface`、`linalg::makeTiledShapes`、
`scf::tileAndFuseProducerOfSlice`、CSE/SCCP、Linalg/Tensor/SCF/Bufferization/Arith families、全部canonicalizer
调用点、instruction/target conversion，以及device publication backend。当前baseline应观察到6个canonicalizer cut point，
以及实际argv中的clang `-O2`、CRT GCC `-O2`和`--gc-sections`；数量和flag是observation，不是稳定schema常量。
`wafer-opt`补齐相关upstream pass-family debug registration；registration不改变production pipeline。Q33 scope内无
`Unassessed` observation才算inventory closure。
所有production/debug执行均经`OptimizationInvocationGatewayV1` wrapped adapter/pass instrumentation；从同一
registry生成的AST/call-graph/link checker必须拒绝adapter外直接调用known upstream pass factory/utility或
backend launcher、adapter缺key及active spec无invocation site。registration不能成为绕过telemetry的入口。

Gate：从现有named production入口fresh记录source vertical的rank-count=1/16，以及冻结TP16 7B的
source-to-bundle/SystemC/PyTorch、unsupported/skipped、
compile work和host wall；inventory能通过正交exposure/adoption/qualification区分library-callable、production-consumed、
derived debug-only（`DebugRegistered ∈ ExposureSet`且exact spec不被active production policy选中；spec可以本来就是Mode None，
也可以是评估后未进入active set的fixed/cleanup）以及`NoOpObserved`、`DownstreamBlocked`、`Rejected`
qualification status；gateway live telemetry闭包与static raw-call/link
bypass checker同时通过。

### Checkpoint B：Equivalent-IR stability 与 deterministic required normalization

先修语义恢复再采用新优化：

- reduction/DPS init从当前SSA、DPS tie、ConstantLike/fill semantics证明，不依赖direct `linalg.fill` defining op或emitter访问顺序；
- current structured-IR postcondition verifier接受named/generic、共享/非共享producer、scalar capture/inline和
  unit-extent等价形态；Q32.I才建立的`SemanticOpDescriptor`/family domain不是Q33 completion前置；
- standard collapse/expand/transpose/extract-slice、`to_tensor`/`to_memref` alias由ViewLike、type、reassociation和SSA重算；
- 分别把tensor one-trip/full-slice与selected-payload one-trip/full-subview所必需的折叠改成各自显式bounded
  rewrite与postcondition verifier；optional generic cleanup开关或worklist顺序不改变支持性；
- 审计Wafer op memory effects、completion和observable store，确保DCE/CSE/LICM不能删除或越过issue/wait/fence/collective；
- 每次改写后fresh重跑semantic、alias/effect、SPM/DDR lifetime和exact cost，不复用旧analysis。

05 tensor normal form必须逐family验证program envelope、structured DPS tie、`InitReadState × InitOrigin`、scalar
region、tensor relation、zero/one-trip structured control和logical collective communication barrier。normalizer/verifier分离、
幂等、transaction-local且使用`RequiredTensorNormalizationWorkPolicyV1`；full tensor slice按05的tensor type/cast规则删除。
07 selected-payload normalizer是独立cut point，另行检查memref root/layout/effect/traversal；两者只共享纯读identity-view和
trip-count proof helper。standard `MemoryEffectOpInterface`与detailed `WaferResourceEffectInterface`分别服务generic
legality与bytes/role/resource lifecycle marker，并由统一coverage verifier检查；completion identity仍由SSA token、wait/fence
和path verifier证明。

mandatory metamorphic families由16 §5.1拥有。测试从同一verified module构造等价transaction-local clones，二者进入同一
production scheduling/finalization consumer，不形成用户可见第二pipeline。

Gate：所有等价形态都有合法reserved baseline；05逐family postcondition、canonical tensor IR与numeric/effect语义
一致，真实sharing机会可导致不同cost但不能导致当前consumer visitation失败。normalizer一次/两次后
canonical bytes相同，第二次是`Success(changed=false)`且work summary可重现；typed failure不改变source transaction
root。complete rank/all-rank gates和完整source expected differential通过。

### Checkpoint C：fixed optimization qualification 与 post-adoption baseline

按05合同逐项判定：

- scalar/shape/identity scaffolding的窄CSE或其它cleanup只有在不改变tensor sharing/lifetime时才可评fixed；
- whole-tensor CSE、elementwise/producer fusion、unit-dim/view propagation、inline scalar、empty-tensor elimination和loop
  transforms默认交给Q32.M candidate-local机制；
- pack/vector/convert-to-loops等会抢占physical owner或销毁structured semantics的pass保持rejected/deferred；
- mandatory corpus零改写或无下游收益的SCCP等机制记录`QualificationStatus=NoOpObserved`，不象征性加入production；
- qualified fixed/cleanup set接入05同一pipeline builder，debug和production复用body；每项按typed `EvidenceKindV1`产生对应
  rewrite、backend action/output或invocation-only telemetry，且每项都有关闭配置比较。

Gate：两组AllOn/AllOff都通过Equivalent-IR和完整纵向；AllOn成员按typed `EvidenceKindV1`满足rewrite、backend action/output或
invocation-only invariant；compile work/static movement/
host wall无不可解释回退。重新冻结post-adoption rank-count=1/16与7B baseline，作为Q32.I及后续physical-dataflow比较起点。

内部qualification seam先冻结`OptimizationQualificationProposalV1`的fixed/cleanup两个互斥key set及spec digests；
required normalization永远开启，`FixedOptimization`和`BestEffortCleanup`各自只允许
`AllOn | AllOff | DisableOne(own-group key)`。production无CLI/pass option并固定两组AllOn。Equivalent-IR按05固定的
`Original/Metamorphic × BestEffortCleanup AllOff/AllOn` 2×2，四路的`FixedOptimization=AllOff`，required
normalizer均开启，并另做once/twice幂等。mandatory case domain显式固定为source vertical × rank1、source vertical ×
rank16、冻结7B × rank16；冻结7B source config的execution mesh是16，不能与rank集合做笛卡尔积或伪造7B × rank1。
proposal union非空时，scale corpus以两组同时`AllOn`对同时`AllOff`；当前空proposal不生成语义相同configuration之间的
global static/ABBA样本，但仍完整执行Equivalent-IR 2×2、normalizer幂等和production gate。
每个key的marginal comparison只在所属组`DisableOne(k)`、另一组AllOn；每项按其typed `EvidenceKindV1`满足rewrite、backend action或
invocation-only invariant，static vector逐分量不恶化且work在
声明fuel内。wall/peak RSS严格使用05的`OptimizationSetQualificationPolicyV1` ABBA/median/MAD公式，不得
在看到样本后调整guard。required normalizer和proposal内拟`Qualified`的全部fixed/cleanup rows在declared domain的mandatory corpus
任一unsupported/skipped/timeout/缺样本都使qualification失败；production required及完整proposal的mandatory纵向
任一上述结果使Q33 closure失败。非production row可以closed `Rejected/DownstreamBlocked` reason完成inventory，
但不能宣称采用。超界拒绝；无确定性下游收益且无显著host收益时固定为`QualificationStatus=NoOpObserved`；被评估spec保持
immutable，以`DebugRegistered ∈ ExposureSet`且不属于active qualified set表达derived debug-only。global
AllOn/AllOff、2×2和production-AllOn只进入唯一`OptimizationBatchObservationV1`，per-key observation只保存
它自身marginal证据；全部all-and-only通过后才原子发布绑定proposal、batch及全部key observation digest的
`QualifiedOptimizationSetV1`。实现还必须按05唯一`AdoptionQualificationInputV1`、通用
`AdoptionQualificationRunV1`、`AdoptionQualificationResultManifestV1`与run terminal，以及optimization-only
`OptimizationSetPublicationAttemptV1`/terminal、host 环境失效重试总budget和`ActiveQualifiedOptimizationSetRefV1` staging-readback-CAS合同发布；
production只消费active ref指向的immutable qualified set。

mandatory original case的探索性运行最初观察到`StablehloCleanup`与`CandidateCommitCleanup`发生改写，但进一步隔离表明
这些改写掩盖了required-form缺口：scalar/shape scaffolding已收归required tensor normalization，one-trip view与常量offset
已收归selected-payload normalization及对应legality。post-legalization与structured-tensor两个历史canonicalization cut会把
静态rank/mask索引链恢复成下游可验证的常量offset；禁用任一位置都会改变legality，因此现分别登记为
`PostLegalizationCanonicalization`与`StructuredTensorCanonicalization` required mechanism，并保持原pipeline顺序。
修正required边界后，optional `StablehloCleanup`无独立下游收益，以`NoOpObserved`关闭；`CandidateCommitCleanup`增加static
movement与instruction count，以`Rejected`关闭；其余四个optional cleanup只有`NoChange`，同样以`NoOpObserved`关闭。
因此当前冻结proposal是合法空集，production不启用optional generic cleanup；required canonicalization仍经过gateway并属于
required invocation全集。
该判定绑定上述mandatory case domain，不把单个workload现象提升成canonicalizer的一般能力结论。

当前static metric registry四维的精确定义依次为：全部rank已接受DDR allocation end的最大值、单rank SPM high-water的
最大值、全部rank SPM movement bytes总和、全部rank instruction command count总和。四项都从本次纵向已经接受且将被
下游直接消费的`ExecutableBundle`计算，不另跑pipeline，也不从日志或名称恢复。

## 4. Completion Audit

最终build-bound资格化发布结果：

- immutable run digest：`d09b6c3cc9631b7ca1abdfa0bbe9f7e787b631ff65c5dd22fdc2cf8e100d0e88`；
- active qualified set digest：`161242b7ce46a3590122fb067c3bc3cfd3ea0bfefa8577c591b6d23f26befde2`，generation 1；
- 3个mandatory original case分别覆盖source rank1、source rank16和冻结7B rank16；每个case执行4路Equivalent-IR、
  normalizer once/twice和production-AllOn，共21个隔离子进程、745141个canonical invocation terminal；
- source rank1/rank16完整输出与CPU expected逐元素exact；7B 16 ranks完整PyTorch differential的最大绝对误差为
  `0.0029296875`，在冻结`atol=0.004`、`rtol=0.002`内；最终SystemC执行19696个target transaction；
- full archive readback验证全部manifest binding和压缩terminal pack；production通过bounded publication capsule消费同一active set，
  不随证据terminal数量线性增加启动成本。

- 01/05-13/16/18、README和progress与live调用点一致；
- current correctness不再依赖optional generic cleanup或direct producer形态；两个required canonicalization的cut顺序由主线gate锁定；
- adoption records覆盖已链接/注册/实际调用的上游机制，并记录qualified/no-op/blocked/rejected结论；
- fixed pipeline只含qualified target-independent机制，candidate-only清单直接交给Q32.M；
- 双配置build、lit/unit/CTest、dependency/organization checks和mandatory source/model vertical有fresh结果；
- 稳定经验同步memory，计划完成后移入archive并提交。

上述各项均已满足，因此Q32.I解除前置阻塞；未来重放仍不得以“pass可运行”或“标准MLIR通常支持”替代该gate。
