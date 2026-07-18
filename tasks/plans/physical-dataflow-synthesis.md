# Physical-Dataflow Synthesis 实施计划

状态：queued umbrella plan。Q33完成前首个Q32 row Q32.I保持`blocked`；Q32.I/R/B/V/M/S/G及最终Q32 completion的
blocked-by关系只看`tasks/progress.md`，不能从本计划checkpoint标题推断动态状态。

本计划只拆施工顺序、artifact checkpoint、删除门槛和验证范围，不复制总体设计。长期合同由：

- 01：production pipeline 与 artifact 总览；
- 05：optimizer-ready structured IR、required normal form与fixed/candidate optimization分界；
- 06：联合 synthesis、搜索、排序和 atomic commit；
- 07：selected tile-dataflow IR 物化；
- 08：physical encoding、valid domain、view/route/descriptor cover；
- 09、12：exact SPM/DDR gate；
- 10：`TargetImplementationProvider`、ImplementationFamily和target-abstract compute/movement；
- 11：selected movement/compute到exact instruction legality；
- 13：communication family、all-rank compatibility、transport acceptance和transport metrics；
- 14、15：versioned target-call ABI、artifact/package identity；
- 16、17：验证、SystemC/CModel 与板端分层；

共同拥有。源码 ownership 仍按 18，不能以本任务号命名 pass、pipeline、CLI、target、CMake target、IR 或 diagnostic。

## 1. Pipeline Position

```text
Pipeline position:
- Upstream artifact / IR:
  Q33已经完成required normal form、Equivalent-IR Stability和qualified fixed hygiene的optimizer-ready rank-local structured
  tensor program、numeric/effect policy、ExecutionConfig、TargetProfileId和typed target capability provider；Q33重新冻结
  当前production、rank-count=1/16、7B数值与host性能baseline。
- Current stage responsibility:
  实现06定义的有界physical-dataflow synthesis；补齐所需target capability纵向；把唯一selected proposal物化为
  07/08/10定义的typed payload IR；由09/12/11/13和14 pure ABI/artifact-eligibility preflight做exact legality，并原子形成all-rank bundle。
- Output artifact / IR:
  不含search/group/shadow-plan状态的accepted tile/dataflow、instruction、memory和completion IR，以及
  从winner final instruction rows派生canonical `RequiredCapabilitySet`的profile-bearing atomic ExecutableBundle，
  以及结构化planner diagnostics。
- Downstream consumer:
  target LLVM/artifact/package、no-card runtime、17的target-call/SystemC CModel，以及later board numeric/cost/timing
  gates。
- User-level driver / named pipeline:
  wafer-compile现有source-to-bundle named production pipeline；可选Transform extension只调用同一C++ library。
- Explicit non-goals:
  本计划不完成board性能校准、cycle accuracy、dynamic-shape/online scheduling、全局persistent weight cache、
  immutable prepack package publication、whole-model equality saturation或模型专用优化；Transform Dialect控制面不是
  主线completion前置。prepack family只有02/15先补齐typed source/package owner后才可另行启用。
- Completion gate:
  policy-free typed rewrite mechanisms先独立闭合，再由新planner组合；新planner成为唯一production决策路径；旧scope-prefix、独立layout assignment、implicit per-use materialization和
  maximal-resident策略旁路删除；通用property/topology/workload、rank-count=1/16、7B PyTorch/SystemC、exact
  SPM/DDR/event/transport/instruction/ABI、reserved baseline/resource-budget和双配置全量gate全部fresh通过。显式v1重放旧
  profile；显式v2重放rank-count=1/16及mapped/oriented通用source vertical，并在7B实际选择新family时重放Q28 fixed seed与
  Q31 held-out multi-seed comparator/统计；若某能力未被7B触发，用独立通用source case补，不改admission或加入Llama特化。
```

## 2. 施工原则

1. 每个checkpoint先形成可被下一层直接消费的typed artifact，不提交只会打印candidate的旁路工具。
2. analysis只从当前IR与target provider重算；search state不写入attr、side table、TransformState或package。
3. 每增加一个优化choice，先证明保守baseline仍能走同一family/materializer/exact gates，再启用搜索。
4. unknown target capability fail closed；不以底层bit存在、历史代码或CModel可模拟替代typed ABI和板端资格分层。
5. 先保正确性与有界性，再比较静态cost；没有板端校准时不发布时间收益。
6. 切换production时删除旧decision owner，不能长期双轨运行。

## 3. 外部前置：Q33 Compiler Optimization Adoption

Q32不再承担generic canonicalizer正确性债务、direct producer形态恢复或上游pass inventory。开工前必须由
`tasks/plans/compiler-optimization-adoption.md`闭合：

- explicit required normalization和Equivalent-IR Stability；
- fixed hygiene、candidate-local和target-specific三层边界；
- upstream mechanism的availability/adoption/qualification records；
- post-adoption rank-count=1/16与7B source/model/resource baseline。

Q33未完成时不得先构造依赖direct fill、named-op形态或canonicalizer worklist的SemanticOpDescriptor/ImplementationFamily，
否则会把已知表示债务固化进provider。

## 4. Checkpoints

### Checkpoint A（Q32.I）：post-adoption合同冻结和 fresh baseline

输入：Q33 optimizer-ready structured IR和post-adoption baseline；当前Q29 scheduler仍只作迁移实现对照。

施工：

- 将活跃编号文档的历史错位文件名收敛到稳定semantic owner，重写01为当前pipeline/artifact spine；旧basename、错误章节号
  交叉引用和“文件名仅作兼容”话术在current docs中清零；archive保留当时basename/line-range快照，并由README明确其
  non-current-path语义；
- 对06-18与live code做字段/consumer矩阵，确认每个selected decision最终落到哪个typed op/type/attr；
- 固定post-adoption rank-count=1/16 corpus、7B source/config/expected、package readback和unsupported feature清单；
- 为planner telemetry定义结构化但非artifact的schema：generated、constraint-pruned、canonical-merged、
  dominance-pruned、exact-materialized、passing、frontier peak、budget reason和各cost vector；
- 锁定旧planner/layout/movement决策入口和删除清单。

产出：可复现的source-to-bundle/CModel baseline与consumer矩阵。

Gate：active owner导航与本地引用无断链、旧basename零残留、01当前/目标事实与live public artifact types一致；development和
target-model两种配置均有fresh build/lit/unit/CTest、完整unsupported审计、rank-count=1/16与7B结果；本checkpoint不以
历史数字代替重放。

不算完成：只保存IR dump、只跑单rank或只统计layout op数量。

### Checkpoint B（Q32.I）：current-v1 implementation provider

输入：structured semantic root、current v1 Tile/Instr/TargetCall/CRT/SystemC合同和硬件事实。

施工：

- 从structured semantics/indexing/numeric/effect归一`SemanticOpDescriptor`，建立独立parameterized
  `TargetImplementationProvider`；未选family前不创建`wafer.tile.*`；
- 先注册现有v1可发射family与每个production semantic family的canonical baseline implementation/safe-tile公式，声明
  semantic/numeric、dtype/accumulator、encoding、geometry/tail、invalid-lane predicate/transfer、resource/effect和exact legality；
- accepted `WaferComputeOpInterface`/`WaferLayoutOpInterface`只验证selected精确合同，不返回domain、route或cost；
- provider输出按canonical semantic signature排序，统一physical size/resource metrics，不依赖名字、pointer、registration或线程顺序。

产出：从normalized structured descriptor到current-v1 parameterized family/baseline的planning capability；未资格化family不进入domain。

Gate：每个family、有限enum全覆盖、参数约束边界类和property-generated shapes验证domain/verifier/materializer/current-v1
vertical；不同线程数/registration顺序返回同一canonical family signature。

不算完成：把family枚举放回accepted op interface、缺少production baseline family，或让旧planner补选layout/route。

### Checkpoint C（Q32.R）：relation、encoding、route 和 descriptor proof core

输入：Linalg indexing maps、view ops、static shapes/dtypes和target encoding/transfer provider。

施工：

- 实现`IndexRelation`的identity、permutation、broadcast、static slice、reshape reassociation、concat piece和组合；
- 实现`PhysicalEncoding`的logical-to-physical map、storage extent、valid/padding domain、block/tail和view compatibility；
- 在08建立唯一`TransferRouteFamily` provider，从relation与physical map计算direct view、RDMA、WDMA、GS或staged
  movement的descriptor cover、metrics和invalid-lane state transfer；
- 实现有限`InvalidLaneState = NoInvalidLanes | Unknown | KnownSplat(typed raw value)`及family predicate/semantic transfer；
  fill、valid-only movement、segmented/full-physical执行分别验证，不能统一假设zero padding稳定；
- cache key使用与06一致的完整规范化query signature：semantic/index relation、shape/dtype、tile/valid domain、source/
  destination memory space/view/offset、normalized alias/effect signature、family/route/encoding参数、按全部读写operand角色记录的
  incoming source/destination invalid-lane states和target profile；unknown/may-alias只复用保守fail-closed结果；
- 给relation normalization/compose、image/preimage piece、dependent-region fragment和descriptor split设置deterministic fuel/cap。

产出：纯analysis/proof library和target-independent property generators；不产生accepted search attrs。

Gate：随机小shape穷举logical/physical index differential，覆盖multi-dtype、block/tail、padding、bitpacked tail、offset、
三层descriptor、broadcast重复source read、destination valid-domain exact write、fill+segmented ordering、空/越界/不可覆盖负例；
同一proof由planner和materializer/verifier共用。

不算完成：对固定4096/64 shape hard-code，或只验证descriptor count而不验证all-and-only地址集合。

### Checkpoint D（Q32.B）：新provider上的合法baseline纵向

输入：现有rank-local structured program与Checkpoint B/C能力。

施工：

- 用保守source order、provider给出的有界full/safe tile、external/boundary Tensor、每个family canonical accepted encoding
  （例如NE GEMM Cx/NCx）、显式Tensor↔Cx/NCx materialization和spill/reload构造baseline；
- 经同一`ImplementationFamily`、07 materializer、08 route和10 compute/movement产生完整rank clone；
- 先运行per-rank instruction/descriptor/SPM/local-completion gate并形成compatibility signature；再用baseline bundle运行
  whole-variant DDR/package/transport和14 pure target ABI/artifact-eligibility preflight，从最终IR重算exact vector；
- transaction-local构造/验证全部RankExecutable与ExecutableBundle后才原子提交；
- search budget设为零或只允许baseline时仍能走production vertical。

产出：不依赖旧layout planner/scope-prefix策略生成的合法typed baseline，但production尚不切换。

Gate：现有Q20/Q21、rank-count=1/16、7B package、完整PyTorch expected/SystemC CModel和late-rank atomic failure均与当前
baseline不回退。

不算完成：新planner调用旧planner补layout/residency，或失败后绕回另一条production pipeline。

### Checkpoint E（Q32.V）：mapped/oriented/invalid-lane typed vertical

输入：已通过Checkpoint D的current-v1 baseline、Checkpoint C proof core和硬件/ABI事实。

施工：

- 将GEMM orientation作为同一family的typed参数，不复制四套op；Tile、Instr、TargetCall、transaction key、digest、
  formal/oneDNN qualification和SystemC均显式消费；
- 新增closed `wafer-tx81-kernel-v2`/profile与versioned GEMM symbol；不改变v1 signature、identity或qualification；
- 增加mapped transfer的typed segment和SPM local offset；RDMA仅DDR source strided、SPM destination sequential，WDMA反向；
- 对required known padding物化带closed fill domain的Tile→Instr physical-footprint fill + valid segments，或使用可验证
  segmented/masked route；checked elem-count与scalar→canonical-raw semantics闭合到TargetCall/SystemC，CModel只消费最终命令；
- capability分别记录statically-representable、compiler-emittable、model-qualified和board-supported predicates；model-only
  admission不等待board，真实board provider按environment allowlist拒绝未资格row。
- 从winner final instruction rows经14 registry派生rank-local capability keys和all-rank canonical sorted unique
  `RequiredCapabilitySet`，与Q16 bundle原子提交；target conversion从实际TargetCall rows重算rank-local投影，Q17/module
  metadata逐项readback并验证union，Q18 schema-v4按14的versioned canonical bytes/SHA-256协议序列化keys+digest。
  model provider以显式`ModelProfileId`、board provider以显式environment在任何effect前逐key匹配。set只含TargetCall/ABI
  observable capability，不含implementation/encoding/route/invalid-lane planner choice，也不是command list、lease或sidecar；
  Q32 cutover对v1/v2都生产v4并明确拒绝无set的v3输入。

产出：mapped transfer、NN/NT/TN/TT和invalid-lane命令从selected IR到versioned TargetCall/SystemC的model-qualified纵向。

Gate：每个family、orientation有限enum、参数边界类和property-generated shapes覆盖parser/printer/verifier、address/range/canary、
digest、formal/SystemC differential；v1/v2混用、unknown padding、错误tail/local offset、required key missing/extra/tamper/
late-rank union均在effect前失败。板端数值资格仍是later row，但board allowlist preflight合同在本row闭合。

不算完成：只在packet builder写flag、让CModel读planner trace/猜transpose/padding，或同一ABI identity静默扩签名。

### Checkpoint F（Q32.M）：policy-free physical-dataflow mechanisms

输入：Q33的typed upstream mechanism边界、Checkpoint C relation/proof core、Checkpoint D baseline和Checkpoint E已资格化target
capabilities。此checkpoint不依赖beam、score或candidate ranking。

施工：

- 每个atomic mechanism声明输入IR cut、参数、precondition、numeric/effect contract、rewrite fuel和proof result，只做一次有界
  transformation，不读取model name、candidate score、search history或fallback policy；
- 复用upstream `TilingInterface`、`linalg::makeTiledShapes`、`scf::tileAndFuseProducerOfSlice`及可适用的Linalg/Tensor/SCF
  pattern utility；禁止把已链接whole pass复制成Wafer matcher；
- 建立bitwise IndexRelation normalization、bijective pointwise propagation、target-qualified implementation absorption、
  multi-root shared physical-version reuse和movement elimination；
- 将whole-tensor CSE/share-vs-recompute、elementwise/producer fusion、unit-dim/view propagation、inline scalar、empty-tensor
  elimination和合法loop hoist按能力逐项接入；不要求所有机制都qualified，无稳定precondition的保持blocked/rejected；
- 任一改写改变SSA sharing、alias、effect、root、lifetime或event后立即失效旧analysis，从改写后的complete clone重建descriptor、
  relation、SPM/DDR/completion和cost，再运行相同exact gates；每个机制保留显式not-applied baseline；
- numeric-policy-gated reassociation/reduction mechanism默认关闭，只有source与target family共同许可才注册。

产出：可由production planner、focused tests和later Transform control plane共同调用的typed mechanism library；它不产生搜索
frontier、fusion group、layout plan或accepted sidecar。

Gate：property/metamorphic测试覆盖chain、diamond、fanout/fanin、multi-root、broadcast、reshape/transpose、reduce、collective
barrier、多dtype/block/tail和unknown effect/numeric negative；每个qualified mechanism至少一个真实source case发生改写，
改写前后完整output满足source comparator且fresh exact gates通过；rank-count=1/16代表性source及7B compile-scale replay证明
work/fuel有界。按Llama role、参数顺序、固定root数或固定shape注册rewrite不算完成。

### Checkpoint G（Q32.S）：有界constraint search 与 resident dataflow policy

输入：Q32.B可物化reserved baseline、Q32.V typed extension、Q32.M policy-free mechanisms、parameterized domains和proof core。

施工：

- 在`MemoryPlanning` owner-private层建立prepared static problem和capacity analysis：canonical demand/conflict/component/
  activity-clique/digest只构造一次；复用现有三态fixed-capacity MiniMalloc、first-fit资源耗尽fallback和独立validator，返回
  hardware feasibility、validated incumbent、lower/high-water、objective state、tagged lower-bound proof和work summary；不修改
  third-party API，不恢复其minimum-capacity/IIS路径；
- 将SPM owner从“分析、求解、直接写offset”的单体入口拆成`current IR -> planning problem`、`problem -> typed evaluation`、
  `validated placement -> atomic apply`。现有IR replay/named pipeline默认仍只运行完整arena legality，保持用户入口和
  accepted `wafer.spm.offset` schema不变；
- 实现backward dependent-region propagation、shared tile-domain约束和domain narrowing；
- 实现canonical frontier state：graph frontier、带InvalidLaneState的live physical versions、constrained domains、SPM
  sound lower bound或unknown、engine/effect frontier、all-rank compatibility signature和06统一cost vector；partial
  frontier的estimate只有可证明时才能按capacity剪枝；
- 只组合Q32.M已资格化mechanisms，惰性实例化family/tile/route/residency/buffering choice；只在同compatibility signature内
  canonical merge，并只对componentwise-monotone exact dimensions做local Pareto dominance；
- 使用deterministic resource-aware list scheduler，仅对ready-set中reuse/resource signature不同的少量choice分裂；
- shared-input使用deterministic maximal hyperedge和hard-capped conflict split，不枚举`2^fanout`；
- 对island/relation pieces、mechanism applications、family instantiation、tile refinement、ready alternatives、beam、top-K、cache
  和rank frontier设置deterministic hard caps；wall clock只作外部取消，触发时丢弃optimized states并返回reserved baseline；
- 每个materialized candidate先获得独立reserved full-arena legality allowance；忽略SPM interval后仍可能改变Pareto/最终选择的
  canonical shortlist才进入capacity refinement。用single-demand absolute-alignment与validated activity-clique建立lower，
  用best placement实际high-water建立upper，在global node/query fuel下二分fixed-capacity；Feasible降低upper、
  ProvenInfeasible提高lower、ResourceExhausted不移动bound；
- shared objective fuel按`candidate signature / probe / retry level`稳定round-robin分配，不依赖并行完成顺序；cache只保存
  validated feasible placement和proved-infeasible cut。区间重叠不按SPM维度剪枝，只有`A.upper <= B.lower`才可证明A在该维
  不差于B；预算结束以实际validated upper进入cost，gap/stop reason只作telemetry；
- lower-bound proof只允许empty/nonnegative的trivial-zero、single demand、经原conflict graph复验的clique或不带demand集合的
  capacity cut；single/clique可投影
  pressure view，06只用它排序已有tile/encoding/residency/order/buffering neighbor。每个neighbor重新materialize和fresh
  planning，allocator不返回repair recipe，也不另存一份witness；proof到decision的映射只使用本次materializer
  invocation-local `IRMapping`/canonical decision key，clone或rewrite后重建；
- best incumbent apply后，旧offset-dependent range/descriptor/address/narrowing、DTE local-offset、compatibility signature和cost
  全部失效并从placed current clone重跑；不能只fresh recost，也不能沿用full-arena初始placement的gate结果；
- capacity interval只接入新canonical frontier，不给Q29旧matmul/scope pressure heuristic继续增加shape/op case；旧排序与
  spill-vs-maximal-resident decision owner在Checkpoint H一并删除；
- rank proposals只跑per-rank exact gates并按compatibility分桶；coordinator先传播shared transport constraints，再baseline-first
  best-first/factorized lazy join，不构造`K^R`；完整variant才跑DDR/package/transport/ABI eligibility gate；
- selected physical versions和explicit cuts自然导出resident dataflow，不产生fused-kernel/group协议。

产出：有合法baseline、确定性tie-break、结构化budget diagnostic和candidate-growth telemetry的production-grade solver。

Gate：构造会触发partition `2^N`、tile Cartesian product、rank `K^R`和task `N!`风险的压力图，证明访问数不越cap；
固定work policy且无外部取消时不同线程数/重复运行selected signature一致；optimization预算/外部deadline返回baseline，
baseline allowance耗尽返回compiler-resource-exhausted；任一rejected clone不污染accepted IR。同一机器、Release build和冻结7B
corpus相对Checkpoint A post-adoption baseline记录完整source-to-bundle wall、planner各phase wall/work、exact-materialized/top-K/
frontier peak与cap命中；rank-count=1/16、7B完整source/SystemC differential、通用topology和多dtype/tail全部通过。重复运行不得
出现无界增长，外部deadline必须返回reserved baseline。阈值与budget只属于profile/resource policy和实施证据，不写成架构
常量或workload legality。

packing专项还须以独立小图穷举minimum height对照覆盖empty/zero-byte、nonzero base、absolute alignment、disconnected component、
path conflict和first-fit反例；验证lower/high-water收敛、`ResourceExhausted`不推进bound、有incumbent时objective耗尽仍合法、
输入排列/线程数不改变结果。通用chain/diamond/fanout/branch/loop case至少有一例因capacity interval改变候选选择；7B记录
full-arena solve、objective query/node/cache、lower/high-water/gap及最终offset-derived peak，不能只打印solver调用成功。

不算完成：在search policy里临时实现rewrite、只限制最终top-K而允许前面无界生成，或用全局mutable side table保存跨pass决策。

### Checkpoint H（Q32.G）：production切换与旧路径删除

输入：Checkpoint D-G全部fresh通过的新纵向。

施工：

- 让wafer-compile现有named source-to-bundle pipeline只调用新planner/materializer；
- 删除旧scope-prefix policy、spill-vs-maximal-resident finalizer、独立layout label/greedy assignment、per-use implicit
  materialization和只服务旧决策面的options/tests；
- 删除重复的layout/cost/resource facts，consumer只读取typed selected IR；
- 更新source organization、dependency registration和测试镜像；
- 全仓搜索历史pass/pipeline/CLI/diagnostic残留，保留archive叙述但不保留compatibility执行路径。

产出：单一production semantics和可从accepted IR重算的下游facts。

Gate：双配置fresh build、lit/unit/CTest、unsupported清单、dependency/IR/source organization/CRT checks；显式v1重放
Q20/Q21、rank-count=1/16和7B旧profile；显式v2重放rank-count=1/16、mapped/oriented通用source及实际选择新family的
Q28 fixed seed/Q31 held-out multi-seed source-package-SystemC-PyTorch，未由7B触发的能力用独立通用case补；atomic failure
全部fresh通过且不改变admission。最终payload无planner/group/search attrs和旧decision interfaces。

不算完成：新旧planner由flag长期并存、旧planner作为silent fallback、或只从默认pipeline移除但仍保留公开入口。

## 5. 未排期可选后续：Compiler Transform control plane（Q32.T）

该项不是Q32 checkpoint，Q32 completion audit不检查其实现；只有Q33 fixed pipeline与Q32 atomic mechanisms、planner/
materializer、production named pipeline和diagnostic均稳定时才实施：

- 少量coarse-grained Wafer transform ops编排Q33/Q32同一typed mechanisms；完整rank-local static root的physical synthesis仍
  只调用同一planner library；
- 可选只读collect/report helper；
- extension source/registration按18组织；fixed policy重放与production等价，candidate policy在相同profile/budget下产生相同
  selected payload signature；
- 不用TransformState承载beam、SPM、cost、physical version，不把transform module保存进artifact。

若不实施，Q32以06中的optional-control-plane non-goal收口，不留第二条未完成主线。

## 6. Integrated Completion Audit（Q32）

标记Q32完成前逐项确认：

- 05-18合同与live code一致，01/README/progress导航同步；Q32.T不在完成门槛内；
- planner只消费structured semantics/effect/numeric policy/target provider，无模型名、shape表、operand-position matcher；
- Q32.M mechanisms先于Q32.S policy实现；planner和later Transform只调用同一typed implementation，不在policy中复制rewrite；
- hard caps覆盖生成过程而非只覆盖结果，baseline fallback和telemetry有fresh测试；
- 同机Release/frozen 7B相对Checkpoint A post-adoption baseline有source-to-bundle与planner phase wall/work记录，
  exact-materialized/top-K/frontier不越policy cap，重复运行无无界增长且external deadline返回baseline；
- selected proposal只以typed payload IR跨stage，search state、shadow plan、重复layout assignment不存在；
- old decision paths和兼容入口已删除，不能以“默认不用”代替清理；
- exact resource/completion/ABI和Q28 fixed-seed/Q31 held-out multi-seed full numerical gates实际执行，不是unsupported/skipped；
- 只报告static movement/SPM/search或CModel host wall；板端数值/带宽/overlap/timing仍由later gates；
- `memory/bugs.md`只记录真实根因/防复发，`memory/general_dev.md`只记录稳定workflow；
- 计划移入`tasks/archive/physical-dataflow-synthesis.md`，progress更新done index，相关改动提交。

任一项未满足时保持任务未完成，并在progress/本计划写明准确边界；不另建worklog或第二份总体总结。
