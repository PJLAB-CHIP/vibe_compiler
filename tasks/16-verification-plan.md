# Wafer Verification Plan Design

状态：长期验证合同已收敛；当前实现只通过部分局部 gate，不能据此宣称 executable、package 或复杂
大模型主线完成。

本文把 verified program、target environment、distributed program、candidate planning、whole-variant
atomic commit、static rank program、target module、PackageManifest 和 RuntimeSession 串成一条可执行的
验证链。各 dialect/op 的局部语义仍由对应编号设计文档定义；本文只定义跨阶段证据如何组成完成证明。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  tasks/02-15 定义的 verified program、target environment/topology/mesh、distributed program、candidate
  IR、committed executable set、target module/Kernel ABI descriptor 和 PackageManifest。
- Current stage responsibility:
  为每个 artifact 边界定义 parser/verifier、conversion、planning、atomic commit、target/package/runtime
  和真实 program-chain gates；证明失败不会生成部分 accepted artifact或污染persistent state。
- Output artifact / IR:
  stage-local diagnostics、variant/commit reports、module/manifest validation result、RuntimeSession
  completion/error result、numeric comparison和可复现的CI/board evidence。
- Downstream consumer:
  tasks/progress.md 状态推进、release/package acceptance、board deployment和性能校准。
- User-level driver / named pipeline:
  局部named pipelines只作构件；主线完成证明必须通过
  `wafer-opt --program-pipeline=stablehlo-to-executable`选择的direct owner-aware driver重放完整上游链路，再进入
  device link、manifest和RuntimeSession gate。
- Explicit non-goals:
  不用手写fixture、单pass FileCheck、schema roundtrip、symbol count或最小静态block代替主线；不把
  unsupported/skipped test算作通过；无板环境不声明board numeric/completion成立。
- Completion gate:
  真实exported model覆盖至少两个shape variants、persistent-state prefill/decode、TP+DP、PP+DP或EP
  distributed graph中的两个组合并行case；全部经过mandatory atomic commit、target LLVM、actual device
  link、ELF/Kernel ABI/manifest验证和RuntimeSession。板端gate进一步证明numeric、multi-rank completion、
  timeout/error aggregation和state consistency。
```

## 2. 验证原则

- 每层只验证自己拥有的事实；下游实现缺口不能反向删除合法上游语义。
- 影响codegen的事实必须存在于当前IR/type/op/region/effect/symbol中，或由它们确定性重算。
- `ShapeGuardRef`与`TargetVariantId`/`ProjectionSetId`正交；`RankClassId` mapping由commit固定，不是第三个
  runtime guard。所有ranks对一次invocation使用同一coherent target/shape tuple。
- candidate可局部失败并反馈planner；只有完整variant-set全部通过才允许原子commit。
- `DirectFullShape`与tiled方案走同一candidate和global legality gates，不存在production bypass。
- RuntimeSession只验证、选择预编译variant/projection、绑定、launch和complete，不重新规划。
- package/module/ABI只有一个typed事实源；JSON debug view、文件名、参数名和LLVM文本不是协议。
- local regression通过只说明该边界可继续推进，不等于整个compiler/runtime完成。
- 性能profile只校准cost；legality、numeric和failure语义不依赖profile结果。

## 3. Stage Gates

| Gate | 输入 | 必须证明 | 主要 negative evidence |
| --- | --- | --- | --- |
| Verified program | imported StableHLO/program payload | model semantics、dtype/rank、symbolic bounds、typed IO/immutable parameter/persistent state、alias/update、content identity；rename-stable structural ResourceId/DimId | unbounded dimension、payload mismatch、非法alias、从名字恢复state、rename改变ID |
| Target environment | target descriptor/runtime capability/board profile | revision、triple/ABI、memory/engine/DTE/dtype/layout/packet limits、exact DTE block/channel map、nonblocking status/finite wait、errata、capability fingerprint | missing capability、ABI mismatch、unknown required feature、无证据却启用inline DTE |
| Topology/mesh | target environment + deployment snapshot | topology digest、available endpoints、logical axes/shape、rank domain、connectivity、projection policy | duplicate/unavailable/disconnected endpoint、axis product mismatch |
| Distributed program | verified program + execution mesh | component/stage、partition/replica、`dp/tp/pp/ep` coordinate、rank groups、collectives、parameter/state shard relation | default rank 0、rank coverage hole、per-rank independent guard、name-derived stage |
| Local tensor program | component/rank-local StableHLO | structured compute、DPS/indexing、state SSA、tensor collective、bounded shape symbols | raw StableHLO残留、state identity丢失、lower-level memory/transport泄漏 |
| Logical group | local tensor program | fusion boundary、tensor-level body、recursive region legality、tiling/resource demand | nested lower-level op、unsupported region、名字matcher |
| Candidate traversal | logical groups + bounded policy frontier | complete domain/output/reduction coverage；`DirectFullShape`和tiled方案同等进入 | representative-only tile、tail/reduction缺口、hidden fallback |
| Candidate tile/layout | complete traversal clone | tile-local scope、layout proposal、explicit movement、constant/weight storage proposal | metadata-only layout cast、elementwise枚举爆炸、重复accepted assignment |
| Instruction legality | candidate target-abstract IR | op family、canonical mapping、physical range、byte/count/iteration relation、target integer width | OOB、convert count mismatch、GEMM mapping丢失、overflow/非整除 |
| Whole-entry SPM | complete static rank entry | cross-group/event-aware lifetime、range/alignment/reserved area、pending completion closure | region-local reuse越界、未证明busytable、pending write at exit |
| Whole-entry DDR | complete static rank entry + typed resources | external/persistent/transient demand、cross-group lifetime、capacity/alignment/bandwidth、accepted offsets | single-group plan冒充global、resident/state identity丢失、view OOB |
| Transport | all memory-planned candidate rank entries + topology/mesh | send/recv peer、bytes、phase、endpoint/exact DTE block/channel/FSM/stream/exact recv range、action/member ID、token/status一一匹配，输出 stage-accepted transport | unbound logical peer、phase/bytes mismatch、offset缺失、duplicate member、DTE wait/status缺失 |
| Launch projection | distributed coverage + candidate entries/resources + stage-accepted transport | 每个 component/partition/replica 映射唯一；pinned fingerprint 或有限 relocatable template 完整 | coverage hole、stale topology、resource/transport collision、runtime有搜索自由度 |
| Executable variant | all candidate facts | typed ShapeGuardRef、target/projection refs、CandidateExecutionEntry到ExecutableEntry identity保持、rank coverage、final class为单一distributed prerequisite class的partition refinement、entry/resource/transport refs、whole-variant atomicity | partial commit、跨prerequisite class合并、candidate entry泄漏、per-rank guard、logical group/pending event残留 |
| Target LLVM | committed static rank functions | structure-preserving SCF/CF/function conversion、typed CRT calls、无非法Wafer op | recursive flatten、mutation后失败、undefined target op |
| Kernel ABI/ELF | one target variant的全部modules + executable entries | ordered slots、type/role/access/address-space/alignment、WCRE/domain-separated descriptor digest、`.note.wafer.abi` bytes/type/alignment、ELF target/exports/module content digest；`TargetArtifactSetId`即set-root semantic digest并绑定source executable和EntryId/function-digest/member map，全部rank-class modules验证后原子发布 | slot漏/重/错序、non-canonical identity record、note缺失/重复/错type、manifest/ELF digest mismatch、wrong ISA/ABI、跨executable同ABI错换module、单module link失败后partial set可见 |
| PackageManifest | committed executable + target artifacts | Protobuf schema、exact blob digest + WCRE semantic manifest ID、orthogonal target/shape selection、committed rank graph、typed resources、entry/completion graph、artifact digests | 把deterministic Protobuf当canonical ID、unknown identity field、shadow instruction list、LLVM regex ABI、unknown module、guard overlap/hole |
| RuntimeSession | manifest + verified actual target/topology + invocation | recomputed environment fingerprint、coherent target/shape tuple、rank projection、每个resource slot唯一axis-covered realization、immutable/state/workspace lifecycle、`DdrArenaId + scope` allocation/base-range gate、exact binding、event-driven completion/error aggregation | forged environment wrapper、target mismatch、resource realization overlap/hole、endpoint unavailable、arena/scope mismatch、base+span overflow、stale state publish、wait-before-peer-issue、local drain冒充global completion |
| Board/numeric | RuntimeSession + actual hardware | result correctness、multi-rank progress、timeout/status、state update/poison policy、cleanup | silent hang、partial output accepted、failed state reused |

任何gate通过只证明其输出可被下一层消费。不能把target symbol closure、package parse或module load分别
提升成端到端完成。

## 4. Test Taxonomy

### 4.1 IR and Symbol Tests

- parser/printer/bytecode roundtrip覆盖target environment、distributed/executable/resource/entry/transport对象。
- verifier negative tests覆盖symbol refs、region/body、rank coordinate、guard、resource alias和entry slots。
- recursive region tests覆盖`scf.if`、`scf.for`、loop-carried state和nested collective。
- canonicalization只能删除可重算冗余，不得删除resource/state/rank/transport identity。
- frontend admission gate覆盖10k functions/resources/dims、high-fanout program graph、deep/large textual/bytecode input、
  metadata/NPY header/output bombs和parse worker timeout/RSS；充分limits identity不变，limit failure kill/reap且无partial
  module/output。VerifiedProgramSource始终以root capability+expected refs工作且无public acquire/open/read；frontend与
  artifact-materialization private access分别联合source/stage/outer预算签发read/work lease。反复取同一ref、cross-stage
  lease、异常/取消/提前release不能突破reader/FD/worker/byte/buffer上界。

### 4.2 Conversion and Candidate Tests

- 每个conversion先有unsupported/failure test，再增加positive pattern。
- candidate测试包含full-shape、tiled、tail、reduction split和多output完整coverage。
- 任一candidate失败时主IR byte-for-byte保持未commit状态；diagnostic定位到owner和candidate事实。
- full variant clone至少注入一个rank-specific failure，证明不会留下其它ranks的部分commit。
- parallel/SPMD/distributed limits gate覆盖10k members/edges/instances、coordinate product overflow、clone/analysis bytes、
  helper hang/output-op bomb/stdout/RSS/cancel；checked preflight在展开前失败。充分limits与worker数变化不改变component/
  distributed IDs或partition，超限无partial IR/artifact。topology/projection同样验证large endpoint/template limits且不取prefix。
- candidate solver scale gate用1000 entries和稀疏alternatives证明fragment constraint propagation/merge join/nogood不
  materializenaive product，full clones与backtracking受limits控制；充分limits、parallelism和memo命中不改变winner，
  compose后仍全量reverify。
- compilation-input lifetime gate要求public production driver move-consume`CompilationRequest`并在首个clone/select stage前一次
  seal成move-only `ExecutableCompilationInput`，后续clone/select/materialize/commit stage只接受该owner；它同时持有frontend module、
  `VerifiedProgramSource` coordinator、limits、canonical owner、exact context registry和target materialization plan。registry/
  immutable mesh projection/policy的missing/extra/cross-pair先在IR mutation为0时失败；direct driver随后通过private access在owned
  module materialize target/mesh并从IR重验。第一次candidate clone后销毁原request frame，后续immutable
  materialization与outer commit仍成功。raw `ModuleOp` adapter只能存在于test target且无法materialize/publish。production
  `runStablehloToExecutableCompilation`驱动move-only owners和transaction；IR-only `OpPassManager` pipeline不得取得payload source、
  attach或commit权限，用户也不能手工拼pass绕过driver gate。
- two-target calibration gate用`VerifiedCalibrationProfileSet`验证exact environment lookup、duplicate/wrong/stale
  binding拒绝、missing target analytic fallback和single-profile normalization；cost query必须携带同一
  VerifiedTargetCompilationContext，CompilationRequest拒绝null且empty set是唯一absence；profile/worker order不改变
  legality或充分search下的deterministic winner。

### 4.3 Resource, Geometry and Event Tests

- SPM/DDR按完整rank entry测试cross-group overlap、branch/loop lifetime、alias/view root和persistent资源。
- DMA/compute descriptor测试physical end、element-size整除、iteration product和所有ABI narrowing上界。
- issue/event测试local compute、RDMA/WDMA/TDMA、DTE分别等待；hardware busytable只有在target/runtime
  capability gate确认后才能缩短显式lifetime。
- function return前仍有影响visible resource的pending event必须失败或导出明确entry completion。

### 4.4 Target and Artifact Tests

- target lowering语义测试覆盖false branch、不同loop trip count、nested branch和function call，不能只数calls。
- actual positive链路必须执行`committed executable -> target LLVM -> LLVM IR -> target object -> CRT object
  -> final kcore .so`；`--print-commands`不算执行。
- final ELF检查machine/ISA/MABI/attributes/exports，undefined symbols只允许正式loader ABI allowlist。
- Kernel ABI descriptor的typed fields在compiler object、`.note.wafer.abi`和manifest三方一致；
  loader从descriptor delivery bytes重建WCRE V1并验证domain-separated SHA-256，不对raw
  Protobuf bytes声称canonical。
- WCRE unit tests覆盖每个value tag、big-endian width、field ordering、set sorting、domain separation、
  unknown-field rejection和cross-process golden bytes；content digest单独覆盖final ELF/package blob/payload exact bytes。
- WCRE scale gate用10万/100万record及large canonical sets验证measure+stream两遍编码、完整element-byte external
  merge sort、shared/spill backing和collision compare；instrumented peak heap/FD/scratch受
  `CanonicalEncodingLimits`约束且不随canonical bytes多份复制。不同inline/spill/run/worker配置在充分limits下产生
  byte-identical stream/digest，limit-1失败而不截断；public API不存在unbounded canonical-byte SmallVector返回。
- model identity两阶段测试要求record-11 preimage只含local keys、record-13/14/15都含typed model owner；两个
  local entry/resource/dim结构完全相同但program或payload不同的models必须产生不同ModelInterfaceSemanticId、
  ModelEntrypointId、ResourceId和DimId，cross-model cache/state attach拒绝。rename/path-only保持不变；权重/程序semantic
  change有意换owner，旧state只有显式typed migration后才能进入新registry snapshot。
- artifact loader limits测试逐一覆盖delivery/KAD bytes、recursion、locator/string、set/member/blob/KAD/slot/export、
  ELF header/note/profile、single/concurrent module bytes和checked total overflow；每个limit-1通过、limit+1在
  reserve/map/open/publish前以typed owner diagnostic失败。`TargetArtifactBuildLimits`覆盖closure/unit/module totals、
  prelink/final/staging/memory/FD/workers；不同足够大的limits得到相同canonical partition/modules/set identity，超限
  不产生staging root、subprocess或partial set。
- target subprocess envelope测试用hang、fork child、stdout/stderr bomb、RSS/CPU excess和caller cancel fixtures验证
  per-process/total wall/CPU/RSS/process/output/kill-reap limits；必须kill整个process group、reap、清partial staging且零
  published root。充分limits下sandbox/worker/resource limits不改变tool args、module bytes或fingerprint。
- target-set delivery trust测试只允许
  `TrustedTargetArtifactSetDeliveryRef + ArtifactMetadataVerificationSession & -> LoadedTargetArtifactSetMetadata` production
  metadata API；raw path、独立`ArtifactAdmissionLimits + CanonicalEncodingContext`和default/unbounded session overload不存在。
  metadata load不得打开实际ELF；standalone、migration transform和package-backed actual module load都必须在exact-domain bind后
  共享对应RuntimeArtifactVerificationSession的reader/worker/bytes/FD预算。unreferenced
  private/orphan root、whole-root self-consistent replacement、wrong external
  delivery digest/size/set ID和debug accept-any result均不能产出package/runtime可消费的verified set。
- module partition scale测试覆盖数千entry共享proved-pure helper/immutable non-address-significant global，要求按policy
  拆module且每module按structural digest只链接一份clone；effect/address-taken/mutable global/entry-to-entry edge改为
  non-clonable connectivity，oversize fail-closed。兼容plain/INT8/MXFP units允许合并，final module从actual refs求profile
  union再算fingerprint；worker/enumeration变化不改变canonical packing。
- clone-object link gate检查prelink hidden linkonce_odr/COMDAT digest signature、same-signature definitions byte-equal、
  per-final-module exactly-one definition和core ref closure；same digest/unequal COMDAT、cross-module private undefined失败。
  final ELF readelf/object verifier要求helper hidden/local且不在dynamic exports，只有typed entry exports可见。

### 4.5 Manifest and Runtime Tests

- Protobufgenerated C++/Python类型来自同一schema；semantic verifier只有一份C++实现。
- 同一validated manifest的pinned C++ producer重复输出byte-identical deterministic blob；不同
  Protobuf serialization bytes不能代替WCRE semantic ID。runtime先验blob digest，再parse/
  semantic-verify/重算manifest ID。
- JSON只测试从canonical object导出/导回debug view，不单独定义字段合法性。
- entry slot完整、唯一、顺序、resource kind/access/alias与function/ELF ABI严格双射。
- `CompletionExportId`与executable DAG leaf一一绑定；缺失、重复或绑定到错误entry/rank必须失败。
- guard测试覆盖priority、互斥/重叠、coverage、fallback、all-rank coherent selection。
- model invocation测试覆盖同一interface的prefill/decode/score等至少两个typed `ModelEntrypointId`、各自IO/state/
  graph root/terminal mapping和跨variant一致性；unknown ID、用static `EntryId`冒充、alias歧义、错误IO/state contract及
  symbol rename/permutation在open/allocate/load/submit为0时失败。CLI diagnostic alias必须先唯一解析为typed ID，
  runtime request不得携带raw function/module name。
- invocation-policy测试覆盖typed field declaration/ID、required/default/bounds、request unknown/duplicate/missing/
  out-of-range，以及BoundedCountExpr只引用声明ID；所有失败在provider side effect计数为0时发生。
- no-card gate至少创建两个共享同一service context的sequential decode `RuntimeSession`s，每个只执行一个preflighted
  invocation：resident weight lease复用、streamed weight使用bounded staging，persistent state group snapshot复用，workspace/
  control与capacity token互相隔离。
- state并发测试覆盖group base-version CAS、stale loser零member publish、full-copy snapshot-ready、page-COW
  eager update footprint/refcount cleanup、in-place exclusive group epoch、整组poison/reset/attach线性化；payload/module
  只有在selected source的同一opened lease完成digest/copy后才可发布cache/module lease。
- state durability测试覆盖StateNamespaceId version/length/all-zero、registry restart后StateGroupVersionId和
  ResourceVersionId单调继续、完整member map恢复、stale fencing token/ABA拒绝、counter exhaustion不wrap，以及
  两个namespace exact-byte隔离；durable backing refs在新provider session重新import/attest而不复用transient handle，
  missing/generation/integrity mismatch转requires-reset或poison。atomic candidate crash保持old current并journal回收；
  in-place first-write前dirty marker、crash/lease loss recovery整组poison和successful terminal清dirty均有故障注入。
- persistent scope测试覆盖model-shared/component/execution-instance `PersistentStateScopeKeyV1` restart roundtrip与
  exact isolation；invocation/entry/iteration-scoped persistent state在compile/package verify时失败，transient
  ScopeInstanceId永不进入registry codec。
- cross-model state migration gate覆盖record-21 typed old/new owner/group/member/scope complete mapping、full-copy、fenced
  page-COW snapshot和bounded canonical `VerifiedMigrationArtifactBundle`。full-copy/COW拒绝group/member/scope任一非双射
  split/merge；trusted transform positive包含跨heterogeneous target artifact sets的TP 2-shard -> 4-shard与KV reshard的
  explicit N:M components/scope/slot relation。rename/name-similarity、partial/overlapping或
  implicit split/merge map、missing/duplicate transform output、descriptor/scope/provider
  mismatch、transform写old、stale old version和existing new group拒绝。至少一个plan同时迁移两个groups，success通过一次
  batch CAS发布全部new groups且old全部保留；在第二组copy/transform/attestation及publish前后注入failure/cancel/crash，
  都必须保持全部old current、零partial new group/member，并按整份migration transaction回收journal/backing。逐组CAS实现
  必须被测试拒绝。无plan attach固定
  `state_model_mismatch`，不能靠namespace/alias绕过。
  record-21只编码`StateMigrationScopeSelectorV1` shared IDs，不出现runtime `PersistentStateScopeKeyV1` bytes；runtime join
  plan+manifests后才形成owner-bound `VerifiedStateMigrationScopeRelation`。unknown/duplicate selector、component/execution-instance
  owner错配、selector与manifest group scope不符、full-copy/COW missing/duplicate/ordinal-zipped scope pair、transform slot未显式
  覆盖N:M selector relation及caller直接提交durable key均在registry访问为0时失败。
- migration-plan delivery trust gate只接受
  `TrustedStateMigrationPlanRef + StateMigrationAdmissionLimits + ArtifactMetadataVerificationSession &`；raw path、独立/default
  canonical context、artifact limits/reader和inspection-result conversion均不存在。所有transform set metadata必须来自同一verified
  delivery并由一个metadata session all-required/no-extra加载；跨delivery拼接、missing/extra set/member失败。wrong metadata owner、
  read/work lease超限或计划bytes/ID swap在registry snapshot/backing open/provider side effect前失败；publisher/compiler的显式encoding
  context不能被runtime loader复用为绕过session的overload。
- migration bootstrap gate先从old/new package metadata、record-21、artifact-set metadata、typed old/new
  `(TargetVariantId, ExecutableVariantId, ProjectionSetId)`选择、namespace和validated domain policy构造
  `StateMigrationPlacementRequest`及owner-bound scope relation；request一次性move-own runtime/migration policy、capacity mode和control，
  后续不能替换。invalid selector/endpoint/policy在deployment registry query为0时失败。
  `WaferRuntimeServiceRegistry`随后move-consume request并只读签发all-and-only `VerifiedStatePlacementInventory`，由inventory继续
  唯一拥有request/scope relation并覆盖每个old resolved key的catalog
  owner/epoch、opaque placement identity、provider generation、current domain和reuse/reimport/transfer约束，但无backing ref/handle/
  open/lease/authority。pure placement再用该inventory、verified environment inventory、control/policy/limits形成
  `StateMigrationPlacementPlan`，并证明此时capacity authority、exact context、execution-registry snapshot、blob/backing open及provider
  side-effect计数全0；placement-inventory query恰好一次，同一request不能重复签发或与另一inventory重新配对。plan必须给出exact
  old/new/transform domain set；随后才创建exact context、
  all-or-none `RuntimeBoundMigrationArtifactBundle`、execution registry snapshot及`StateMigrationSessionPlan`。最终snapshot重验catalog/
  placement/provider generation和backing portability/reimport关系；stale在任何backing/provider side effect前失败。任一步owner/domain/
  generation swap都全量失败，不能先拿complete-domain context或execution snapshot再反推domains。
- migration scale gate使用100GB-class sparse/fake backing和multi-component/multi-set/multi-scope plan，覆盖artifact-set/member、
  old/new scope-selector/resolved-key、group/member/page/chunk/logical/read/write/scratch/FD/buffer/worker/time每个limit的limit-1/limit+1；
  `StateMigrationPlacementRequest`不存在caller context/backend/inventory proof/snapshot、单一scope key、caller scope map或raw
  provider/device ID；domain policy只接受authenticated deployment constraint catalog签发的duplicate-free prioritized-allowed refs，第二份
  allowed set/priority map、missing/duplicate/cross-owner ref均失败。bootstrap返回共享owner/generation的move-only inventory和
  retained immutable constraint catalog；inventory move进service bootstrap后catalog refs仍有效，typed ID lookup、stale catalog/
  environment generation和provider factory伪造均覆盖。两轮pure preflight期间
  execution-lease/open/allocate/journal计数均为0，execution
  一次取得全部fenced old leases后才允许副作用。full-copy/COW/transform保持bounded peak host memory，不按group分批publish；
  `RuntimeAdmissionPolicy`、migration limits、provider hard limits、host ledger和capacity reservation的重叠维度逐字段取最严格值。
  full-copy/COW的module demand为0；trusted-transform覆盖两个verification/residency轴的四种组合，充分limits/worker/mode变化得到
  相同new state relation，取消/超限保持全部old current和零partial new。
- migration snapshot gate要求只有repo final registry能签发`VerifiedStateMigrationRegistrySnapshot`；caller expected-version/
  new-absent claims、cross-namespace/plan reuse和stale epoch/fence失败。snapshot签发/preflight的lease/open/journal计数为0，
  issuer move-consume `RuntimeBoundStateMigrationPlacement`并由snapshot保留plan/context owner，同一placement不能签两次；execution
  必须把token交回同registry并一次取得全部old-group leases后才允许副作用。
- migration result gate要求`StateMigrationResult`的success arm只携带canonical typed
  `PublishedStateGroupVersion {StateGroupKey, StateGroupVersionId}` records，failure/cancel/deadline没有publish summary；parallel key/version
  arrays、duplicate/reordered/missing key、outcome/payload矛盾和caller `oldPreserved` claim均不可构造。failure使用独立
  `StateMigrationFailure`/`BoundedStateMigrationFailureReport`和closed phase/kind + typed plan/old-new key/scope/artifact/domain；复用
  `InvocationFailure`、component ordinal/name或按diagnostic字符串分支均失败。
- bundle scalability测试在低`RLIMIT_NOFILE`下装载含数千module/payload locators的bundle，证明unselected blob不
  open、active handle数有界；覆盖capability-relative beneath/no-follow open、symlink/rename/mutation race和eviction后重验。
- executable resource relational-join gate使用至少100,000 resources、10,000 ranks和多个target/shape/projection records，
  证明coverage verifier按canonical composite key做bounded sort/merge/set sweep，不分配四轴Cartesian product；漏项、重复和
  extra relation仍给出typed owner diagnostic，充分limits/worker变化不改变结果。
- geometry authority gate分别从candidate transaction内部fresh ResourceView和sealed committed conversion request构造
  `VerifiedPhysicalAllocationView`，要求相同root/view/access/range事实通过同一core；raw range/callback、stale generation、
  swapped operand或cross-owner view失败。candidate preflight不得产LLVM/KAD/staged blob；commit后篡改任一capacity/offset/
  prototype必须在artifact conversion重放时失败，不能复用pre-commit passing proof。
- target conversion lifetime gate销毁caller/coverage frame后继续完成converted owner的KAD和scoped object emission，证明
  entry-boundary/clone-dependency private storage保留committed MLIR-context lifetime share且owned module先析构；cross-context share、
  仅move `OwningOpRef`或独立caller `MLIRContext`参数在mutation/tool launch前失败。
- bundle trust测试只从deployment-owned `VerifiedPackageBundleReference`取得index/manifest expected size、content
  digest和PackageManifestId；whole-root替换、内部自洽的old/new bundle swap、tenant expected-ID swap、bundle index
  自报另一组expected values均在其它blob open/allocate/load/submit前失败。debug accept-any结果不能构造RuntimeSession。
- parser backing测试要求KAD/set/manifest/ELF note proof只保留owner-backed `abi::ImmutableByteBackingRef`；销毁caller
  raw buffer后proof仍安全，从mutable/错digest/错size owner、超`maxInlineRecordBytes` test adapter或悬空ArrayRef构造失败。
- bootstrap/session测试先用Delivery session验证commit record，再用Artifact metadata session形成`LoadedPackageMetadata`，证明此时
  module/blob open、provider authority/acquire/load/submit均为0；pure preflight选择exact domains后才创建service context，one-way
  bind形成`RuntimeBoundPackage`。先销毁两个metadata sessions及其栈frame，owner-backed delivery/package/target metadata仍能pure
  preflight和bind；任何parse lease都不能逃逸。metadata/domain/context/session任一swap、preflight后改invocation、反向unbind或直接从metadata source
  open都失败。`InvocationControl`和reservation mode必须在preflight时move入plan，create API不能替换；preflight后cancel/expiry在
  context/reservation/open边界可观察。同一immutable metadata storage并发创建两个invocation sessions不重复parse、不复制open
  authority且共享host/service ledgers。创建loader的栈frame退出后分别执行eager和at-first-use lazy ELF验证，二者使用bound move-owned owner token/limits/
  canonical scratch并得到相同结果。Delivery/Artifact/runtime三类typed session不可互转但共享一个
  `HostVerificationRegistry` physical parent ledger；每个service owner/generation另恰好一个artifact child ledger。两个同context
  sessions共享parent/child预算，不能通过新建session、package、invocation或切换admission mode重置。依赖检查证明
  VerificationSupport不依赖semantic层、Delivery不依赖Artifact/Runtime且WaferArtifact不依赖Runtime。
- runtime lazy source预算测试要求metadata只产生无任何acquire/open/read能力的`UnboundBlobSourceDescriptor`/
  `UnboundTargetModuleSourceDescriptor`，one-way bind后才产生`BoundBlobSource`/`BoundTargetModuleSource`；bound types也无public
  acquire/open/read，只有同owner
  `ArtifactVerificationReadLease`可形成opened-object lease，且reader/FD计数直到opened lease/backing销毁才归还。
  `ArtifactVerificationWorkLease`覆盖worker/verified-byte工作期。cross-session/source lease、复制/拆分/提前release、异常/取消
  路径泄漏和从`RuntimeBoundPackage`/`RuntimeBoundTargetArtifactSets`反复open绕过预算都必须失败；compiler/diagnostic source不能转换为runtime source。
  joint-accounting负例让host parent、service child或invocation capacity任一侧在prepare后失败，要求按固定owner顺序
  try/rollback、等待时零partial reservation、无deadlock且任一瞬间各层usage均不超限；并行delivery/package metadata parse不能
  双花host FD/worker。metadata parse计对应typed child+host parent，actual module/blob验证恰好计runtime child+host parent与invocation
  capacity各一次。
- publication/parse/admission测试分别覆盖package blob/module/payload/staging/reader/FD/buffer预算，delivery byte/
  recursion、各repeated record、AST depth/node，以及selected/peak module code/verification/mapping bytes、expanded
  graph/resource/host metadata/arena/concurrent host-window transfer bytes和checked product overflow limits；失败发生在
  output creation/open/allocate/load/submit计数为0，调高deployment
  policy可接纳同一合法manifest但不能改变graph/variant/microbatch/window。
- outer publication测试覆盖两个target sets中第二个失败、package assembly最后失败、immutable roots写完但trusted
  commit record前crash、commit record CAS失败和multi-directory配置；所有case都无可见program/package alias/index，
  loader不能接受orphan root。成功只有一次unified-root rename或单一`ProgramDeliveryCommitRecord`原子可见，包含当前
  completion scope要求的全部roots/reference。
- delivery transaction测试要求staged executable/target/package tokens和`PackageAssemblyInputView`均
  non-aggregate/private-construction；stable canonical owner generation与attachment-table snapshot generation严格分离。
  `ExecutableCompilationInput`、winner、`CommittedExecutableProgram`和lower commitment必须沿同一owner链move-own all-and-only
  canonical `VerifiedTargetCompilationContextRegistry`，attach API不得另接context vector；销毁
  `CompilationRequest`后Target build仍能按TargetVariantId解析完整context。missing/extra/duplicate context、cross-context/token swap、
  stale generation和caller重新注入context都在tool launch计数为0时失败。frontend owner必须同时retain module所依赖的
  `MLIRContext`；销毁caller parse frame/context handle后late clone、target materialization和failure cleanup仍安全，production
  materializer/direct driver不存在独立`MLIRContext &`参数。
  连续attach至少两个targets时首个staged token保持有效，旧view失效，最新view覆盖全部targets；cross-transaction、混用两种
  generation、stale view、caller target subset/reorder/extra attachment及post-attach mutation失败。commit-record deterministic codec覆盖unknown field/version、unsafe
  locator、scope coverage、record/ref limits及external trusted digest。`ProgramOutputLimits`测试让multi-target/package各自
  fit但aggregate logical/physical/root/FD/metadata超限，要求all-or-none reserve；exact dedup只减physical计数。
  crash orphan GC不得删除active/retained record可达root，unreferenced root永远不能派生trusted child refs。
- blob attachment scale测试用100GB-class sparse/fake-CAS accounting证明same-store CAS ref/reflink/protected-hardlink不会
  全量复制且取得独立destination lifetime；普通writable hardlink、seal/generation mutation、refcount/GC race失败。
  cross-filesystem走bounded same-open-handle stream-copy并复核chunks/digest。logical/physical byte limits分别计费，
  attachment mode变化不改变ArtifactRef/manifest identity或bundle-relative locator contract。
- transaction factory测试要求completion scope、VerifiedProgramOutputSink capability/generation和ProgramOutputLimits
  只在`create`时move-own绑定，`commit()`无参数；staging后scope downgrade、sink/root swap、limit replacement API在
  compile-time不存在。任一child failure永久uncommittable，即使较小scope已满足也不能提交。
- module verification/residency测试正交覆盖四种
  `eager_active_set|at_first_use` x `eager_active_set|graph_liveness`组合。selected unconditional module数大于provider
  同时load上限时eager residency拒绝，graph liveness按first-use/typed last-command-completion运行；dynamic false expert在
  两个eager模式下仍保持blob open/verify/load/copy/submit全0，true wave在首个dependent side effect前完成所选eager gate。
  另覆盖simultaneous-live peak超限preflight、shared cache exact-key
  reuse、concurrent single-flight、last completion前禁止unload和load/resolve failure零published lease。cache只保存
  intrinsic `ProviderCodeLease`；同module bytes被两个packages引用时，每次都重新join current member/EntryId/KAD/class/
  completion并产生新的ModuleUseLease/VerifiedEntrySubmission，wrong-package relation在submit=0时失败。
  CLI只接受独立`--module-verification`/`--module-residency` closed values；旧`--module-admission`、缺值/unknown值和运行中
  policy replacement均在metadata source open/provider side effect为0时失败。
- symbolic release-rule测试覆盖same-iteration+delta、domain-final-tail和invocation-terminal-set，small graph与full expansion
  逐instance对照；百万iteration保持bounded cursor/RSS。iteration i terminal不能释放仍覆盖i+1 future demand的aggregate
  module/resource lease，final-count边界、delta overflow、missing/duplicate/early terminal和跨domain错误relation均在issue前失败。
- runtime capacity reservation测试让多个sessions各自preflight通过但合计超过arena/module-code/commands/events/
  copies/windows/blob-lease/FD/host-transfer limits，要求provider/device/generation manager做多维all-or-none reserve，
  任一时刻总usage不超限且失败无partial token。覆盖cache hit marginal accounting、single-flight miss不double count、
  DAG last-completion增量释放、failure/cancellation release、FIFO bounded wait/no starvation、nonblocking overload以及
  provider generation变化使旧tokens/cache ownership失效并重新reserve。
- multi-domain reservation gate用2/4 devices及TP/PP/EP rank projection验证snapshot的canonical domain set和每rank exact
  generation binding；一个InvocationResourcePlan分解后由coordinator按domain ID ordered prepare/commit。任一domain
  capacity不足、prepare failure、restart、deadline或cancel都rollback全部domains且零partial counters/leases；成功token
  覆盖完整nonempty domain set，释放/失效保持per-domain cache隔离并终止受影响的整次invocation。
- reservation authority gate模拟两个process共享device：exclusive-local只有持deployment lease/fencing generation者可
  reserve/submit，stale/expired lease失败；central/provider mode跨process prepare/commit保持总量上限且failure rollback。
  inventory声明shared但两种authority均无时session creation structured unsupported，不能静默使用process-local counter。
- provider authority gate要求production registry只move-own一个sealed `VerifiedRuntimeServiceBootstrap`；其factory move-consume
  canonical `RuntimeProviderDomainCapabilitySet` vector、独立authenticated deployment inventory、reservation authority、optional
  durable-state service和runtime host capability并验证完整cross-service relation。provider adapter不能mint其它trust input，registry
  create不能接原始capability或独立relation proof，proof后swap和同ID self-report均失败；host capability只由lower layer验证自身
  provenance/liveness并计physical budget，不与provider owner作semantic join。每个domain set内部
  environment/module/memory/transport/command/completion/cancel能力绑定同一provider owner/generation，heterogeneous domains合法。
  deployment authority与durable-state service可有独立verified owner，但cross-service device/namespace/fencing relation必须闭合。
  transport capability缺失、被ordinary command capability冒充、request member/relocation被替换或raw receipt/handle来自另一domain/
  generation均在cursor advancement前失败；copy/readback success无completion receipt、range/source swap和cross-domain receipt也失败。
  provider receipt不能直接构造semantic completion proof。
  `RuntimeSession::create`、executor和state migration execute的可调用API均不存在
  `RuntimeBackend &`或零散manager/callback参数。caller fake即使声明相同identity也不能构造context/proof或submit；unit fake只能实现
  low-level capability并经registry/final managers重放全部join、reservation和fencing。
- streamed-weight测试使用大于resident arena capacity的immutable backing，证明canonical windows精确覆盖consumer
  slices、chunk digest/range合法、copy completion支配consumer、double-buffer reuse等待last consumer、device allocation
  受staging上界约束且host不缓存全payload。hidden full backing、window hole/overlap、early reuse和copy failure均拒绝。
- immutable artifact materialization测试从structured source tensor/shard经过accepted layout/quant/storage/residency
  生成exact resident和streamed blobs/digests/chunk tables；packing/coverage/digest可重放且峰值host memory有界。
  不同IO buffer/worker count必须得到相同bytes/digest及固定4 MiB `ArtifactChunkingPolicyV1` table，window overlap只
  复用chunk ref而不重切。source verify后替换、symlink、truncate/grow、concurrent mutation和reopen-by-path均失败；
  同一`SourceArtifactLease`完成parse/read/digest/stat。candidate rejection、publication failure、missing/extra/wrong
  blob/chunk和descriptor mismatch均证明IR与shared store原子不变，package不能回原NPY/path重建。
- target compiler staging测试要求`StagedObjectRef`无public acquire/open/read且不常驻每object FD；只有同owner live
  `TargetArtifactBuildSession`签发的`TargetBuildReadLease`可形成same-handle opened lease并联合stage/outer reader/FD/buffer
  预算。cross-session/generation、重复open超限、tool临时path reopen、异常/取消和提前release均不得绕过账本。
- candidate artifact-cost测试构造多个structurally passing/不同cost candidates和large sparse source，要求先形成
  exact `ImmutableArtifactMaterializationPlan`，只对deterministic winner读取/pack一次；相同recipe跨candidate只产生一份
  staged blob/chunk proof。winner-specific encoding failure按canonical order和attempt/byte limits尝试下一项，source-global
  digest/IO failure立即终止；budget failure不发布未materialized candidate。
- runtime environment测试从provider side-effect-free facts query + trusted inventory构造verified snapshot，覆盖forged
  caller digest、stale generation、provider/inventory endpoint/arena/mode mismatch，且失败前allocation/load/submit计数为0。
- provider-generation测试在相同device/environment fingerprint下模拟provider restart/reload，要求module/weight cache
  miss或隔离旧lease、旧handle submit失败且新generation重新load/copy；generation不能只用于diagnostic。
- module/entry resolution测试要求selected `OpenedBlobLease`经shared ELF verifier与manifest member join生成
  `VerifiedLoadableModule/Entry`后才load，`proof -> ModuleLease -> ExecutableHandle`逐项绑定EntryId/symbol/KAD/class/
  completion coverage；proof types不可forge，wrong/missing function或KAD在submit计数为0时失败。
- endpoint测试覆盖pinned strict match和relocatable预编译union；negative cases覆盖environment/mesh/
  `ProjectionSetId` digest mismatch、relocation slot type/count/owner错误、allowed-binding/member digest错误，
  runtime不得生成新route/binding。
- segmented exchange测试覆盖count phase支配data phase、per-peer counts/displacements/capacity、send/recv总量、
  overflow和partial-peer failure；equal-split all-to-all不能替代该gate。
- completion DAG测试覆盖host command、local drain、DTE/collective wait、stage barrier、timeout和rank failure join；
  reciprocal two-rank和pipeline stage case必须先issue全部ready participant再poll/wait，证明executor不因稳定排序死锁。
- invocation control测试覆盖pre-issue cancellation、caller steady-clock deadline、mid-PP、mid-stream-copy、
  mid-atomic-state-write和mid-in-place-write；未issue descendants抑制、issued cancel/safe-wait、cleanup ordering、
  atomic candidate discard及in-place durable poison必须与failure合同一致。pre-issue side-effect计数为0，取消一个
  InvocationId不影响并发invocation，cleanup failure不能覆盖cancel/deadline或更早provider failure。
- PP iteration template测试覆盖bounded count expression、overflow/max、zero-delta cycle rejection、positive-progress
  iteration edge、u64 intermediate但u32 max/actual/iteration narrow boundary（UINT32_MAX与+1）、至少两个microbatches
  的stage overlap、iteration-scoped workspace和中途failure后的remaining-node抑制。
- rolling graph cursor gate用百万iteration/长decode证明`VerifiedGraphInstancePlan`只保存template/count/total/peak/
  max-live-span proof，`VerifiedGraphInstanceCursor` host memory保持在active frontier上界；small graph与full expansion逐
  EntryInstanceId/edge/order/result differential一致。unbounded live span、frontier limit、trace sink limit、midstream cancel/
  failure和iteration resource release均覆盖；dry-run不能积累全trace。
- sparse MoE activation测试让count phase产生typed bounded per-expert counts，覆盖0/nonzero predicates、finite waves、
  false-path skipped-success、conditional join、copy/module/entry/data dominance和staging reuse。总expert weights大于
  resident arena而每wave envelope合法；zero-count expert的blob-open/load/copy/submit均为0，active experts数值与combine
  正确。count overflow、producer未完成、predicate错owner、active超envelope、conditional state write、missing skip join、
  provider不支持和mid-wave failure均结构化失败，runtime不得少选expert或退回load-all。

### 4.6 Real Program and Board Tests

- importer/framework gate必须使用真实或忠实exported model，不用手写`wafer.group`替代。
- static tiny block保留为regression，不作为LLM completion证明。
- board numeric使用独立reference，比较prefill及连续decode；只检查shape或module load不算numeric。
- multi-rank board gate注入单rank timeout/error，确认其它rank停止、资源清理和state consistency策略。
- performance/profile只在correctness gate之后运行，并记录target/profile provenance。
- calibration profile gate验证versioned generated schema、unknown field/version rejection、exact environment/protocol/
  correctness provenance、integer observation integrity和deterministic loader；planner只接收immutable verified view，
  report JSON/path没有import路径，missing/out-of-domain profile回退analytic estimate且不改变legality。million-sample/
  large-evidence fixture覆盖delivery/observation/sample/string/scratch/buffer/worker limits、owner-backed input lifetime及bounded
  peak memory；producer先measure再stream，parser不接raw ArrayRef，充分limits/worker变化保持exact bytes/content digest一致。

## 5. Mandatory Vertical Gates

### 5.1 Static Compiler Regression

输入可以是小型static transformer block，但必须经过真实frontend/distributed program和mandatory atomic
commit，到target LLVM、actual device link和manifest validation。它只证明基础编译链，不证明state、
dynamic variants或multi-card。

### 5.2 Bounded Dynamic Variant Gate

- 同一verified program含至少两个bounded dynamic dimensions和两个static executable variants。
- actual dimensions命中确定variant；所有ranks选择一致；越界/无覆盖shape在allocation前拒绝。
- target instruction和memory plan保持static，不把dynamic fallback推给runtime。

### 5.3 Stateful Prefill / Decode Gate

- immutable weight、persistent paged state、workspace和user IO使用不同resource kinds。
- prefill后连续两次decode读取并更新同一state；weight handle复用，workspace不复用到下一invocation。
- KV backing、page table/control等必须共同publish的state属于同一`StateConsistencyGroupId`；entry slots显式选择
  current/candidate/in-place version role。
- `atomic_version`分别覆盖full-copy和page-COW snapshot；任一member失败都不切换group version，CAS loser不发布
  partial member。`in_place_poison_on_failure`失败后session拒绝继续使用整组state。
- 销毁并重建RuntimeSession后，PersistentStateRegistry仍保留current `StateGroupVersionId`、完整member map或poison状态；
  不能通过新session绕过失败。

### 5.4 Composite Parallel Gate

- 至少覆盖TP+DP和PP+DP；PP+DP使用至少两个microbatches并检查stage overlap/iteration edge/failure；EP/MoE gate进一步覆盖ragged token route、all-to-all-v等价语义和expert
  weight placement。
- EP/MoE negative gate覆盖count/displacement越界、peer总量不匹配、count phase未完成即issue data和
  expert destination capacity overflow。
- component/stage、partition/replica、`dp/tp/pp/ep` coordinate分别验证；flat rank只作ordinal。
- rank class共享和拆分均有case；uneven shard或不同transport计划不得错误共享。
- RankClass V1 resource等价gate要求TP instances若ordered slot的shard/content ArtifactRef、storage/capacity/arena/
  scope或state version role任一不同必须拆class；DP replicas仅在这些facts逐项相同时可共享。相同static EntryId/module
  可以跨拆分后的classes复用，不能为共享代码错误合并resource lookup class。

### 5.5 Multi-Card Transport and Completion Gate

- endpoint/channel/FSM/recv-buffer assignment在candidate中完成验证，并与rank entries/projection随
  executable原子提交；RuntimeSession只消费committed结果。
- pinned projection严格匹配topology digest；relocatable只从manifest的`ConcreteRecordSet`或
  `FiniteTemplateSet.allowed_bindings`按确定性优先级选择。
- projection negative gate覆盖environment/mesh/projection/member digests和relocation slot type/count/owner；
  segmented transport逐peer匹配count/data phase和completion/error。
- board报告host completion、device local drain、DTE/collective wait和rank aggregation，不用单一字符串代替。

### 5.6 Quantized / Mixed-Precision Gate

- source-backed或忠实StableHLO input使用registered affine quantized/FP8 types/ops；shared
  `QuantizationDescriptor`与`StorageEncodingDescriptor`分别拥有数学语义和physical packing，scale/zero-point/
  block/tail、accumulator/result、rounding/saturation及target capability/profile均typed。
- affine tests覆盖per-tensor/per-axis/per-group descriptor positive/negative；首个TX81 native numeric gate使用
  capability允许的signed INT8 GEMM subset，覆盖负值、nonzero left/right zero point、q0/q1 shift、scale mode和
  saturation boundary；mathematical zp `0/127`为边界positive，`-1/128/255`在首个signed profile下拒绝且不能
  static-cast。reference按q1 first shift、scale、q0 second shift和INT8 result计算；需要f16时验证后续显式
  dequant/convert，不能把i32 internal accumulator或FP16 AddOutput假定成native quant result。unsupported
  granularity/rounding/output relation在target mutation前拒绝。首个native profile禁用axis scale；positive/negative
  scale只有exact formula/indexing/table dtype的packet+board evidence后才可新增profile，改变fixed rounding/saturation但
  packet不变的case必须拒绝。
- FP8 tests覆盖E4M3/E4M3FN/E5M2的zero/subnormal/normal/Inf/NaN policy、E8M0 block scale、32-value block、tail/
  padding和malformed packed size；TX81必须选择explicit decode到BF16/FP16再GEMM，不能生成native FP8 format。
- target gate验证两个fixed command struct的size/alignment/offset、exact CRT fields、required-symbol closure、golden
  packet/composite issue以及decode completion支配GEMM/scratch reuse。capability/profile fingerprint mismatch在load/
  submit计数为0时失败；board reference分别覆盖integer和FP8 composite numeric。

### 5.7 Oversubscribed Immutable Weight Gate

- 至少一个真实或忠实exported workload的immutable payload大于selected device resident arena capacity，但每个
  compiler-declared staging window及同时live windows之和满足capacity/alignment/bandwidth约束。
- compiler在candidate IR中显式提交source slice、`StreamWindowId`、staging resource/range、copy completion、consumer
  slots和reuse edges；若需要K/reduction或expert partition，必须先由typed group/tile/instruction IR表达，runtime不发明。
- package/runtime在低FD限制下只打开selected blob，按同一opened lease分块校验并异步copy；不创建全量device
  WeightLease、不把全payload缓存在host内存。no-card验证graph/allocation，board验证copy-compute overlap、数值和cleanup。

## 6. Failure and State Consistency

- verifier/conversion失败在mutation前结束；若conversion需要clone，失败只删除clone。
- local candidate failure可继续搜索；complete executable variant candidate失败不能保留其它rank的accepted facts。
- package/runtime校验失败发生在module load、allocation或launch之前；部分已创建handle按反序清理。
- launch后timeout/error停止后继entry，等待可安全回收的events，聚合rank/stage状态。
- `atomic_version` state group只在snapshot-ready及全部writer completion成功后一次CAS发布；旧group snapshot在失败时
  继续有效，不能partial member publish。
- `in_place_poison_on_failure` state group发生任何不可回滚member写入后若失败，必须poison整组并拒绝后续invocation。
- diagnostics包含artifact/variant/component/rank/entry/resource稳定ID；名字只用于显示。

## 7. CI and Reproducibility Gate

- 默认CI实际运行lit、C++ unit、Python/generated-binding tests和semantic verifier，不只构建test target。
- CI报告unsupported/skipped清单；mainline依赖不能被静默skip。
- dependency payload使用commit/hash lock；target toolchain profile、CRT ABI和symbol-set version进入fingerprint。
- device positive/negative link在干净环境执行；final ELF保留或等价验证RISC-V attributes。
- PackageManifest、ELF modules、weights和其它payload都有digest；tx8依赖有license/SBOM/digest manifest。
- board环境和无板环境分开报告，不能用no-card结果替代board numeric/completion。

## 8. 当前证据和限制

当前lit/ctest、IR organization、dependency consistency、target CRT conformance和105-symbol closure是有效
局部证据；actual hand-written positive device link也证明工具链可链接。但以下事实使新主线尚未完成：

- target LLVM仍需修复structured-control flattening。
- rank identity、whole-variant commit、whole-entry memory/event和physical transport尚未实现。
- Kernel ABI descriptor、Protobuf PackageManifest和新的RuntimeSession尚未实现。
- 当前HF case是static custom block，未覆盖stateful decode、real MPMD、manifest或board numeric。

因此，设计文档收敛不改变上述实现状态。只有本文件相应vertical gate得到新鲜执行证据后，任务队列
才能把实现项标记为`done`。
