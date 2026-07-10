# Wafer Launch and Runtime Package Design

状态：本轮长期边界合同已收敛；实现状态以`tasks/progress.md`为准。范围：Protobuf `PackageManifest`、非序列化 `RuntimeSession`、host runtime adapter、
entry/completion graph 和 completion/error contract。

本文定义 model-level typed runtime package、host runtime session / adapter 和 completion contract。该边界
消费 whole-variant atomic commit 后的 `wafer.executable`、其 static rank functions、typed resources、
accepted transport/completion template、target LLVM / `KernelAbiDescriptor` 产物和 kcore executable modules，
负责把模型接口、resource metadata、endpoint policy、DDR binding、constant/weight artifact refs、runtime
requirements、orthogonal shape/target variants、committed rank/stage entry graph 和 completion DAG 组织成
可执行 package。

长期 wire contract 是单一 Protobuf `PackageManifest`。C++ compiler/package/runtime 共享生成的 message
types 和一份 semantic verifier；Python 工具只使用生成 binding 或调用 C++ validator。package identity
分为exact delivery blob digest和从已验证typed object计算的WCRE semantic ID；不假定
不同语言或版本重序列化得到 canonical Protobuf bytes。JSON 只由同一 typed object生成用于诊断/调试，不是第二份
schema或可独立编辑的生产输入。当前 schema v2 JSON只允许经显式 converter迁移到 typed manifest，并在完整
semantic validation后使用；runtime不能同时维护 v2 和新 manifest两套 launch合同。

`txLaunchKernel` / `txLaunchClusterKernel` 只是 TX backend 的 kernel-level entrypoint materialization。
它们不能作为 Wafer package 的主语义。模型级 package 的稳定语义是“executable model +
typed resources + runtime session state + entry/completion graph”；kernelArg、BPM table、graph
directory、legacy bootparam 都是 package entrypoint 的具体承载方式。

本文依赖：

- `tasks/04-topology-execution-mesh.md`
- `tasks/12-ddr-memory-planning.md`
- `tasks/14-target-llvm-golden-packet.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/tx8-api-struct-contract-annex.md`
- Protobuf proto3 language guide: https://protobuf.dev/programming-guides/proto3/
- Protobuf schema best practices: https://protobuf.dev/best-practices/dos-donts/
- Protobuf serialization is not canonical: https://protobuf.dev/programming-guides/serialization-not-canonical/

## 1. 目标和非目标

目标：

- 组织 RISC-V kcore device modules、model interface、typed resource declarations、endpoint projection、
  shape guards、target artifact sets、committed rank/stage entry graph、completion DAG和 optional
  profiling/control metadata。
- 区分 immutable package facts与 invocation/session facts：`PackageManifest`不保存 provider handle或物理
  地址，`RuntimeSession`不重新决定 compiler resource/layout/transport plan。
- 保持 `wafer.executable` 是 compiler内唯一 committed composition事实源；manifest是不可变派生产物，
  不能反向成为 compiler scheduling或variant事实源。
- 为 persistent paged state、immutable resident weights、ephemeral workspace、external IO和 transport /
  completion control resources建立明确 role、lifetime、alias、capacity、placement和 ABI slot合同。
- 明确 `WaferRuntimeAdapter`、registry-owned `RuntimeProviderDomainCapabilitySet`、Tx/KMD 事实来源和 legacy `TsmRun` fallback
  的职责分层；HPGR / `libhpgr.so` 是当前 `tx_runtime` provider 证据，不作为 Wafer 主抽象名。
- 给 runtime allocation failure、stub shielding、completion DAG、timeout/error propagation 和 status/profiling 建立可验证
  合同。

非目标：

- 不决定 group boundary、tile shape、layout cut、SPM allocation algorithm 或 DTE collective
  algorithm。
- 不引入新的 tensor constant 语义。constant/weight 在这里表现为compiler生成的read-only artifact refs
  和对应DDR demand；大payload bytes不进入Protobuf message。
- 不把 legacy `Tsm*` stub 当作 correctness path。
- 不把 `PackageManifest` 反向写回上层 tensor/group IR。
- 不把 `txModuleLoad` / `txLaunchKernel` / kernelArg 作为模型级 package 主合同。
- 不把 `instructions` 数组、collective step list或 planner trace序列化成 package shadow schedule；device
  instruction和通信顺序仍由 committed IR/module表达。
- 不以 flat binding list、自由字符串 lifecycle、单一 `binding_order` 或单 scalar completion source作为
  长期 ABI；不在 JSON和 C++ parser中分别实现语义验证。

## 2. Launch Boundary

当前不引入单独的 launch IR op。一次编译后的 model invocation 由 typed `PackageManifest`、
非序列化 `RuntimeSession`、entry graph和 target LLVM/device module artifact共同表达。
`PackageManifest`是committed executable中全部runtime-observable facts的lossless delivery projection，再
附加原子发布的target artifact identity；它不从多个旁路重新恢复语义，也不复制function body/instruction
schedule。下面字段都必须追溯到唯一committed owner：

- model interface：user-visible inputs/outputs、symbolic shape/bounds、dtype、external layout、alias policy和
  persistent state relation。
- endpoint view：committed `ProjectionSetId`和`wafer.executable.transport` records；topology/mesh只用于
  verifier重放，不在package阶段重新选择mapping。
- resource requirements：committed `wafer.executable.resource`和entry slot bindings。`KernelAbiDescriptor`
  只交叉验证ABI shape，不首次定义external IO/weight/state/workspace/control role。
- modules：kcore `.so`、graph directory、未来 BPM/control descriptor 等可执行或控制对象。
- executable variants：每个`ExecutableVariantId`直接序列化committed `ShapeGuardRef`、`TargetVariantId`、
  `ProjectionSetId`和entry/completion graph引用。shape guard只读actual dimensions/state capacity/policy；
  target requirement/artifact set只读capability/fingerprint/topology compatibility，二者不能混成同一guard。
- rank classes / entry graph：只序列化executable composer已提交的`RankClassId`及其rank mapping、entry node、
  module/entrypoint和host-visible dependency；package阶段不得重新归并rank，也不复制module内部schedule。
- runtime requirements：device selection、PG tile selection policy、stream/event policy、allocation /
  copy policy、completion timeout/error/profiling policy。runtime handle 和 physical address 只属于
  runtime session，不写回 package canonical facts。

`PackageManifest` 不包含 tensor-level fusion plan、tile-local memory effects或 p2p schedule；这些属于
`wafer.group`、`wafer.tile.region` 和 committed communication IR/module。manifest只组织 runtime必须观察的
resource、entry和 completion边界。

### 2.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  whole-variant atomic commit生成的完整`wafer.executable`，以及`tasks/14`按其program semantic digest和
  `TargetVariantId`原子发布的complete `TargetArtifactSet`。不接受单个KAD/module或resource/launch side view。
- Current stage responsibility:
  校验complete target artifact set identity，并从committed executable生成deterministic Protobuf
  `PackageManifest`：typed resources、executable variants、target artifact sets、原有rank classes、entry graph、
  endpoint projection和completion DAG。delivery/artifact metadata阶段先产生owner-backed proof和不可open source；pure preflight
  得到exact domains后才取得sealed service context并one-way bind。C++ runtime再创建非序列化`RuntimeSession`：先过滤
  target/projection compatibility，再对整个invocation求值一次shape guard，随后
  使用committed rank mapping完成state/weight/workspace binding、endpoint/stream materialization、
  launch和 completion/error处理。跨model state migration另按semantic request -> placement inventory -> exact-domain plan ->
  context/artifact bind -> execution snapshot -> whole-plan batch CAS推进。
- Output artifact / IR:
  validated Protobuf `PackageManifest`、exact blob digest、WCRE semantic manifest ID、可选JSON debug
  projection，以及引用/封装complete
  `TargetArtifactSet`与weight/constant payloads的runtime-loadable bundle root/index。device object/shared
  object仍是TargetArtifactSet成员，不由本stage重新产出。进程内另产生`LoadedPackageMetadata`、`RuntimeSessionPlan`、
  `RuntimeBoundPackage`、`StateMigrationPlacementPlan`和closed `StateMigrationResult`；它们只在
  进程内存在，不序列化 runtime handle、physical address、stream/event或 provider-private object id。
- Downstream consumer:
  repo-owned runtime executor经registry-owned Tx/KMD或restricted legacy low-level provider capabilities执行、board correctness gate 和
  profiling/error propagation gate。
- User-level driver / named pipeline:
  package emission由`wafer-opt --program-pipeline=stablehlo-to-executable`选择的direct production driver生成committed
  executable和target artifacts驱动；package assembly/device-link tools是该driver内部或下游构件，不以
  hand-written manifest/JSON、instruction text或C stub table作为production入口。
- Explicit non-goals:
  不重新做 tile search、layout、SPM/DDR planning、communication schedule或 target LLVM call emission；
  只序列化`tasks/04`已提交的pinned/relocatable `ProjectionSetId`和transport refs，不在package阶段决定
  projection mode或从名字恢复；不把
  legacy bootparam/TLV字段反向提升为 compiler IR语义，不保存 shadow instruction schedule；manifest不替代
  `wafer.executable`，RuntimeSession不生成新 variant、rank class、entry graph或transport route。
- Completion gate:
  device-code gate产生descriptor semantic digest、`TargetEnvironmentFingerprint`、
  `TargetArtifactFingerprint`和module digest一致的ELF；真实program
  chain生成 deterministic `PackageManifest`并由唯一 C++ semantic verifier接受。no-card gate覆盖 resource
  slot双射、shape/target axis正交、committed rank/stage graph、pinned/relocatable projection、persistent state / weight /
  workspace lifecycle、metadata/runtime one-way bootstrap、shared capacity/provider authority、state migration atomicity和completion
  DAG负例；board gate证明 host command、device drain、DTE/collective
  completion和 error/timeout按 DAG传播。手写 JSON、v2 roundtrip、symbol discovery或单 scalar completion
  都不能单独满足本 gate。
```

## 3. Runtime Package Contents

Runtime package 是交付给 runtime adapter 的 immutable编译产物集合。长期 `PackageManifest` 包含：

| 部分 | 内容 | 来源 |
| --- | --- | --- |
| model interface | ordered typed `ModelEntrypointId` records及各自inputs/outputs/state contract、symbolic dimensions/bounds、dtype/layout、alias和typed bounded invocation-policy integer declarations | committed `wafer.executable.model_interface_ref`与`wafer.executable.invocation` mapping |
| typed resources | external IO、immutable parameter、persistent state、workspace、staging、control/status resource及其typed realizations | committed `wafer.executable.resource` |
| modules | final module content digest、format、descriptor semantic digest、artifact fingerprint | atomically published `TargetArtifactSet` validated against executable entries |
| executable variants | `ExecutableVariantId`、typed `ShapeGuardRef`、`TargetVariantId`、`ProjectionSetId`和graph refs | committed `wafer.executable.variant`；shape/target/rank axes保持正交 |
| target artifact sets | imported locator-free `TargetArtifactSetIdentity`、stable `TargetArtifactSetId`（即set-root semantic digest）、source executable digest、target requirement/fingerprints、member map和compatible projection class | target LLVM/device-code gate keyed by committed `TargetVariantId`；不嵌target-set delivery root |
| rank classes / entry graph | committed `RankClassId`、canonical rank mapping、`SlotId -> (ResourceId, StateSlotVersionRole)` bindings、host-visible dependencies | committed `wafer.executable.rank/entry`；KAD只交叉验证，package不重新归并 |
| endpoint projection | pinned projection fingerprint或 relocatable placement/control requirements | committed projection set + `wafer.executable.transport` |
| SPM summary | per-tile SPM peak、reserved range、allocation summary | 从committed offsets派生的non-semantic diagnostic/capacity summary |
| constant/weight artifacts | chunked `ArtifactRef`：digest、byte size、encoding、object/chunk offset、shard/packing metadata | committed immutable resources + artifact bundle assembly |
| completion DAG | host command/event、copy、device drain、DTE/collective wait、stage barrier、status/error edge | committed variant/entry/transport completion records |
| runtime requirements | capability、allocation/copy、timeout/error/profiling policy | committed target requirement/resource/completion policy；provider只验证支持性 |

constant/weight artifact不是新的IR constant op。`PackageManifest`只保存`ArtifactRef`，实际bytes属于
artifact bundle/object payload；每个ref必须能追溯到原始constant/parameter、slice/shard relation、dtype、
shape、packing/quant descriptor和selected storage layout。若schema提供small-inline优化，必须受固定
schema上限约束并由semantic verifier检查；immutable weights和超过上限的payload必须chunked external，
不能把大模型权重塞进Protobuf。

`PackageManifest`不能直接嵌入`target_artifact_set.proto`中带build-output locator的delivery root，也不能让
record-10 WCRE递归编码delivery-only fields。它直接import并嵌入该schema唯一的locator-free
`TargetArtifactSetVerifiedRecord`，其中复用`semantic_identity.proto::TargetArtifactSetIdentity`、claimed
`TargetArtifactSetId`、typed `TargetArtifactMemberKey`和验证evidence；package不得复制这些semantic/member records。
manifest-owned cross-relations只能按member key引用。package assembly通过outer `PackageAssemblyInputView`消费同一
owner-backed verified set record/module views并写该message，不调用standalone target-root loader。bundle内module/payload的relative locator只由
package-owned `PackageBundleIndex`保存一次并明确排除在record-10 identity之外；manifest中的ArtifactRef用typed blob
key/content digest引用该index，不再保存第二个locator。原target-set build locator不进入package，也不能在bundle移动后
继续生效。

production bundle load的trust anchor不能来自bundle自身。deployment/control plane先提供nonsemantic
`TrustedProgramDeliveryRef`；`VerifiedPackageBundleReference`只能由成功验证该outer delivery的type-specific child-ref adapter
构造，至少绑定bundle root locator/capability acquisition、expected bundle-index exact size/content digest、manifest blob exact
size/content digest以及expected `PackageManifestId`。bundle内index、manifest、文件名或独立trusted inventory都不能补默认/
覆盖这些expected values或绕过outer completion scope。该value不进入PackageManifest identity，也不因换部署路径改变semantic ID。

artifact验证分为两个单向阶段，不能让尚未选择runtime capacity domain的metadata bootstrap反向依赖
`WaferRuntimeServiceContext`。

Delivery metadata、Artifact metadata和runtime artifact虽然语义权限隔离，却会争用同一host FD/reader/worker/memory。
更低层runtime-neutral `WaferVerificationSupport`拥有process/deployment-scoped `HostVerificationRegistry`和唯一物理资源
ledger；`TrustedDeploymentBootstrapAccess`只能从`TrustedHostVerificationInventory`的owner/generation/hard maxima签发唯一
`HostVerificationLedgerCapability`，`HostVerificationRegistry::create`要求requested limits逐项不超过hard maxima。production caller
不能自造issuer/capability，test issuer只存在unittest target。Support只做checked operational reservation，不include、forward-
declare或friend任何Delivery/Artifact/Runtime semantic type。registry分别签发不可互转的
`ProgramDeliveryHostBudgetCapability`、`ArtifactMetadataHostBudgetCapability`和`RuntimeArtifactHostBudgetCapability`，三类reservation
都计入同一parent ledger且各自仍可有更小typed child ledger。Delivery、Artifact和Runtime只能消费自己的capability；不能相互转换、
升级session或用新建typed registry重置parent总量。三种capability只提供typed pure-operational
`tryReserve(HostVerificationDemand) -> HostVerificationLease`/child-ledger机制，其private constructors只friend
`HostVerificationRegistry`，lower layer不知道任何upper owner。

`WaferArtifact`先拥有process/deployment-scoped、runtime-neutral的
`ArtifactMetadataVerificationRegistry`。registry唯一签发move-only、non-aggregate、non-forgeable
`ArtifactMetadataVerificationBudgetCapability`，其底层绑定`ArtifactMetadataHostBudgetCapability`且所有capability/session共享同一
metadata child ledger；新建session不能重置parent或child的
active reader、open FD、verification worker或simultaneously verified metadata bytes计数。唯一factory
`createArtifactMetadataVerificationSession(ArtifactMetadataVerificationBudgetCapability,
abi::ArtifactAdmissionLimits, abi::CanonicalEncodingContext)`按值move-own全部参数，生成
`ArtifactMetadataVerificationSession`。该session只授权delivery/index/manifest、locator-free target records、KAD metadata和
migration-plan metadata的bounded parse/hash/semantic verification；它不能打开实际module/weight/state backing、调用provider或
构造runtime execution proof。

production package metadata入口固定为
`loadPackageBundleMetadata(VerifiedPackageBundleReference, PackageParseLimits,
ArtifactMetadataVerificationSession &) -> LoadedPackageMetadata`。standalone target入口固定为
`loadAndVerifyTargetArtifactSetMetadata(TrustedTargetArtifactSetDeliveryRef,
ArtifactMetadataVerificationSession &) -> LoadedTargetArtifactSetMetadata`。二者返回owner-backed immutable metadata、
semantic proofs和尚未绑定runtime的lazy source；没有raw root/path、独立/default canonical context、内部自报expected值或
inspection-result conversion。package metadata loader验证index/manifest和内嵌locator-free target records，但不得打开任何
model blob或ELF module。migration plan也走同一个metadata阶段；其actual transform module只在后续runtime bind后打开。

runtime-safe owner固定为`include/Wafer/Artifact/RuntimeArtifactVerificationSession.h`中的
`wafer::artifact::RuntimeArtifactVerificationSession`。`WaferArtifact`同时拥有runtime-neutral、move-only、non-aggregate且
non-forgeable的`ArtifactVerificationBudgetCapability`；它只能对active readers、concurrent verification workers、
simultaneously verified module bytes和open file descriptors做全或无的operational reservation，不能构造semantic
identity/proof、改写limit或代替canonical verifier。唯一session factory是
`createRuntimeArtifactVerificationSession(ArtifactVerificationBudgetCapability,
abi::ArtifactAdmissionLimits, abi::CanonicalEncodingContext)`；三个参数都按值move-own，失败不返回partial owner。
每个`WaferRuntimeServiceContext` owner/generation恰好拥有一个绑定`RuntimeArtifactHostBudgetCapability`的runtime artifact child
ledger；该context签发的全部capabilities/sessions共享此ledger并与host parent及session-local positive limits取逐项minimum，不能通过新建session、package或invocation
重置全局预算。session私有保留limits、canonical scratch/collision state、独立session owner token以及不可伪造的service-owner/
generation binding；没有default/implicit/unbounded factory，也没有public context、capability或owner-token accessor。
`WaferArtifact`不include/link `WaferRuntime`，也不知道`WaferRuntimeServiceContext`类型。

runtime session private access只签发move-only、non-forgeable `ArtifactVerificationReadLease`和
`ArtifactVerificationWorkLease`：read lease联合预留active reader/open FD并由最终opened-object lease持有；work lease联合预留
verification worker/simultaneously verified bytes并覆盖hash/ELF/KAD工作区lifetime。两者都绑定session与service
owner/generation，不能公开构造、复制、拆分、提前release或跨source/session使用。runtime-private joint reservation还必须把
相同reader/FD/worker/bytes demand原子计入host parent、service child和当前`RuntimeCapacityReservation`：按稳定ledger owner顺序执行
nonblocking prepare，后一侧失败就立即rollback此前全部，任何等待前均不得持有partial reservation。metadata child与runtime child
权限彼此独立但共享host parent；metadata parse只计对应metadata child+parent，actual blob/module验证计runtime child+parent及
invocation capacity各一次，既不能漏计也不能重复计费。

metadata loader先从reference取得root capability，按external expected size/digest bounded创建owner-backed
`abi::ImmutableByteBackingRef`并解析`PackageBundleIndex`，再要求
index中的唯一manifest locator及blob facts与reference逐项一致；随后用同一root做beneath/no-follow open，验证manifest
exact size/digest、parse/semantic-verify并比对expected `PackageManifestId`。这些步骤通过前不得打开其它blob、分配/
load module或发布loaded bundle。把整个root/index/manifest替换成另一份内部自洽的合法bundle、old/new version swap或
跨tenant expected-ID swap都必须失败。显式diagnostic/debug `loadAnyBundleForInspection`若保留，返回类型不得转换为
production `LoadedPackageMetadata`，也不能创建RuntimeSession。

manifest semantic verifier对每个内嵌locator-free `TargetArtifactSetVerifiedRecord`调用WaferABI唯一record verifier，生成
owner-backed `VerifiedTargetArtifactModuleView`s；每个view闭合一个ModuleEvidence及parent all-and-only members。随后按module
content digest把view与bundle index派生的`UnboundBlobSourceDescriptor`一一join。它不构造`TrustedTargetArtifactSetDeliveryRef`，不读取
原target-root locator，也不接受caller member array。actual ELF验证从view构造`VerifiedTargetElfContract`并从同一package
runtime bind后生成的`BoundBlobSource`读取；standalone target metadata使用`UnboundTargetModuleSourceDescriptor`，bind后生成
`BoundTargetModuleSource`，两种physical source adapter共享同一ELF core。unbound descriptor只保存root owner、safe locator、
exact size/digest/chunk facts且没有任何acquire/open/read能力。one-way bind重验metadata/context/session owners后才创建
`BoundBlobSource`；runtime-private blob access必须move-consume一个同owner `ArtifactVerificationReadLease`才能返回
`OpenedBlobLease`，后者持有该预算lease直到FD/backing关闭。任何测试、manager、cache或inspection helper都不能从
`LoadedPackageMetadata`取得source后绕开runtime artifact session反复open；compiler `BoundArtifactBlobSource`与diagnostic source使用分离类型和
各自预算owner。

`LoadedPackageMetadata`是immutable owner-backed handle，其shared const storage拥有manifest/index backing、root capability、metadata
provenance token和unbound lazy sources；复制handle只延长同一storage lifetime，不复制lease、ledger或open authority。它不借用
metadata session或其lease。session只约束本次parse I/O，返回后可销毁，同一metadata可供多个并发invocation pure preflight/
one-way bind而不重复parse；provenance token只用于证明factory/registry generation，不能重新open。runtime先只用
它、side-effect-free `VerifiedRuntimeEnvironmentInventory`、typed invocation、admission policy、by-value
`InvocationControl`和closed `RuntimeCapacityReservationMode`运行纯
`preflightRuntimeSession`，得到唯一target/projection/shape/rank tuple、exact capacity-domain set和move-only
`RuntimeSessionPlan`；plan move-own control/mode并在每个后续边界重验deadline/cancellation，不能在create时替换。preflight不得取得
device authority、打开blob、allocate、load或submit。随后registry才为该exact domain set取得
`WaferRuntimeServiceContext`和reservation authority，由context-private adapter签发runtime artifact capability并构造verification
session。唯一one-way bind是runtime-private
`detail::bindPackageForRuntime(LoadedPackageMetadata, RuntimeSessionPlan,
std::shared_ptr<WaferRuntimeServiceContext>, RuntimeArtifactVerificationSession) -> RuntimeBoundPackage`，全部参数按值
保留/consume并重验metadata/plan/domain/context/session owner-generation；plan/context/session是move-only，metadata handle复制仍指向
同一immutable storage。不存在反向unbind、换context/session或从metadata对象直接open。
`RuntimeBoundPackage`共同拥有metadata、plan、shared context和verification session。production
`RuntimeSession::create(RuntimeBoundPackage)`不再接受caller backend、context、manager、artifact session或limits。
standalone/migration target metadata使用等价的all-or-none `bindTargetArtifactSetsForRuntime`，可一次绑定多个heterogeneous sets，
不能逐set成功后留下partial runtime input。`eager_active_set`或`at_first_use` actual ELF验证都只能借bound session；owner mismatch、
预算不足或session失效在module load前失败。

package assembly只接受同一outer `ProgramOutputTransaction`内sealed、已重验但尚不可见的committed executable/
`ArtifactRef` attachment及private content-addressed staging。compiler-private
`detail::PackageProgramOutputAccess::beginPackageArtifactBuild(ProgramOutputTransaction &, PackageAssemblyInputView,
PackagePublicationLimits)`是唯一factory，返回move-only、non-aggregate `PackageArtifactBuildSession`。session一次绑定stable
canonical owner token、input view的exact `AttachmentTableGeneration`、受限canonical encoding session、immutable backing/
attachment factory、verified package staging area、outer/stage budgets和cancellation；assemble/stage及V2 converter都只消费
该session或其result token，不能分别接raw `CanonicalEncodingContext`、backing factory、root/path或callback。

session在同一transaction attach bundle/index/reference，不能要求或触发whole-stage提前publication。每个ref必须绑定
`ResourceRealizationRecordKey`、storage/quant profile、logical coverage、exact size/content digest和canonical chunk table；
resident ref覆盖完整backing，streamed refs与每个`StreamWindowId` source range/chunks闭合。不存在installed/public
`ArtifactBlobReader` interface。session/view内部只持private-construction `BoundArtifactBlobSource`；writer只能经friend按完整
canonical attachment view遍历，并从source取得same-handle opened lease后stat/read/hash/stat，不能访问frontend NPY path、
重新packing、按path reopen、根据resource name生成payload或注入custom reader。missing/extra/wrong digest/chunk/descriptor、
stale attachment generation或跨transaction source使整个package build session失败且不发布partial bundle。

canonical chunk table唯一使用`ArtifactChunkingPolicyV1`：每个content blob从byte offset 0开始按固定
`4 * 1024 * 1024` bytes切分，除最后tail外每段长度必须等于该值；record按offset递增，记录checked offset/size和
exact chunk SHA-256，要求无hole/overlap且总和等于blob size。policy version和algorithm enum随ArtifactRef序列化；
V1不允许caller/profile/window改变chunk size。每个stream window引用与其source range相交的完整canonical chunk
ordinal set；同一acquisition内重叠window可按`(ArtifactRef, chunk ordinal)`去重校验，但不能合并/重切chunk或改变
window语义。materializer/runtime可以用更小bounded IO buffer跨多个read增量hash一个chunk，IO buffer、worker count和
window overlap都不得改变blob digest或chunk table。

package publication不能无条件把outer transaction中已验证的module/payload再次全量复制。identity-neutral
`VerifiedBlobAttachment`使用closed `BlobAttachmentModeV1 = cas_reference | reflink | protected_hardlink |
bounded_stream_copy`：assembler根据source/destination backend的verified immutable capabilities选择；选择不进入
PackageManifest/ArtifactRef identity，最终PackageBundleIndex仍只保存bundle-relative locator、exact size/digest。
`cas_reference`必须取得destination/root独立lifetime refcount；`reflink`必须证明COW independence；
`protected_hardlink`只允许source inode/object由write-once content store或fs-verity/等价seal证明任何actor均不可修改，
并为destination取得独立GC lease。chmod/read-only path不算seal。无法证明same-store attachment时使用bounded
stream-copy，从同一opened verified source lease增量复核digest/chunks并写private destination，不能按path重开。

attachment proof绑定source content owner/generation、exact digest/size、backend attestation、destination root/relative
locator和lifetime token；private constructor factory在attach后重验destination stat/seal，outer commit前再验证token。
mutation/generation/GC/refcount mismatch使transaction uncommittable。`PackagePublicationLimits`和
`ProgramOutputLimits`分别记录logical output bytes与actual newly copied/allocated physical bytes；dedup/reference不能把
logical member count/size隐藏掉，stream fallback也不能超physical/staging/FD/buffer预算。

bundle loader不能把“曾经校验过的path”交给RuntimeSession，也不能在variant选择前为每个module/shard常驻一个
open handle。manifest exact bytes由loaded bundle不可变持有；loaded bundle持有bundle/content-store root的
capability directory handle，runtime-safe `BoundBlobSource`只保存该non-forgeable root owner、validated relative
locator、expected content digest/size和chunk map。locator拒绝absolute、parent traversal、symlink和non-regular object；
未选择的blob保持unopened。

selected module/payload首次使用时，runtime-private access必须先取得同owner read lease，再通过root capability做
beneath/no-follow relative open，检查同一opened object的type/stat/declared size并返回move-only `OpenedBlobLease`。
消费方另取得work lease并从这个opened lease的同一handle流式读取，在
H2D/module load过程中复算chunk和exact-byte digest；校验后不能按path重新打开。bounded handle cache只能缓存
仍持有相同verified lease的selected object，eviction后的下一次acquire必须重新capability-open并在新handle上完整
重验。全部required chunks、copy和provider verification成功前不得发布module handle或resident cache lease；若读取
期间size/content变化、提前EOF或digest不符，当前acquisition失败并逆序回收。这样同时关闭check/use TOCTOU、把FD
占用限制在active acquisition集合，并且不要求大模型权重整体驻留host内存。

### 3.1 Outer Program Delivery Transaction

compiler、target builder和package assembler共享唯一outer `ProgramOutputTransaction`。其C++ owner至少提供：

```text
ProgramOutputTransaction::create(ProgramDeliveryCompletionScope,
                                 VerifiedProgramOutputSink,
                                 ProgramOutputLimits,
                                 CanonicalEncodingContext)
  -> ProgramOutputTransaction
detail::ExecutableProgramOutputAccess::attachCommittedExecutable(
  ProgramOutputTransaction &, CommittedExecutableProgram)
  -> compiler::StagedExecutableToken
detail::TargetProgramOutputAccess::attachTargetArtifactSet(
  ProgramOutputTransaction &, compiler::StagedExecutableToken, staged verified set)
  -> compiler::StagedTargetArtifactSetToken
detail::PackageProgramOutputAccess::buildPackageAssemblyInputView(
  ProgramOutputTransaction &, compiler::StagedExecutableToken)
  -> compiler::PackageAssemblyInputView
detail::PackageProgramOutputAccess::attachPackageBundle(
  ProgramOutputTransaction &, compiler::PackageAssemblyInputView, staged bundle)
  -> compiler::StagedPackageBundleToken
ProgramOutputTransaction::commit() -> PublishedProgramDelivery
```

scope、sink capability、limits和compiler canonical encoding context在唯一factory中move-own绑定，之后没有setter/
overload/default。所有child preflight和
attachment都读取同一retained values；不能在staging后换output root、提高budget或把`package_bundle`降级成
`executable_only`。`VerifiedProgramOutputSink`只能由trusted filesystem/deployment backend factory创建并绑定root/
namespace generation。任一child/verification/reservation failure永久把transaction标为uncommittable；`commit()`无参数，
只检查factory声明的原scope完整coverage并执行已绑定sink操作。

compiler-only `CanonicalEncodingSession`由transaction私有拥有，包含context和non-forgeable
`CanonicalEncodingOwnerToken`。该owner generation从factory成功到commit/abort保持稳定；增加第二、第三个target attachment
不能使它或已有staged identity token失效。
`ProgramOutputTransaction`没有public session/context/token accessor。只有compiler-private、按owner拆分且被transaction显式
friend的`detail::FrontendProgramOutputAccess`、`detail::ExecutableProgramOutputAccess`、
`detail::TargetProgramOutputAccess`和`detail::PackageProgramOutputAccess`可以在各自stage lifetime内借出受限non-const
session ref或比较owner token；adapter不能被generic caller构造，也不能把context/token返回到public API。record factories
只通过对应adapter调用session的private `context()`。frontend/materialized program、candidate、staged target/package
attachments都保留同一owner token，下一stage先比对owner/generation；不能把由另一scratch/collision registry生成的proof attach
到当前transaction。token不序列化、不进入ID，commit/abort后失效且context不可取走。

`ProgramOutputTransaction`的installed header不声明接受`CommittedExecutableProgram`、`StagedTargetArtifactSet`、
`StagedPackageBundle`或返回upper-layer view的public attach方法。它只include向下的完整Delivery类型，公开`create/commit`和
由本层完整定义的move-only staged token；`llvm::Expected`/`FailureOr`不得实例化incomplete upper type。各stage adapter在
各自上层header/source消费完整upper object，构造不同的non-installed typed attachment commitment并调用transaction private
core。commitment不是generic blob bag或opaque semantic sidecar：executable/target/package三类storage、typed IDs、delivery
commitment和access class互不转换，semantic verifier仍在各自owner层；transaction只统一lifetime、预算、attachment
generation、scope completeness和visibility。Package view由package adapter从lower attachment snapshot及各owner的sealed
typed access形成，不是transaction header依赖Package类型。standalone header compile与dependency gate必须证明
Compiler/Delivery不include/link Whole、Target或Package上层库。

`ExecutableCompilationInput`从创建起move-own all-and-only、按`TargetVariantId` canonical索引的
`VerifiedTargetCompilationContextRegistry`；winner transaction和`commitExecutable`把同一个registry随owner链移入
`CommittedExecutableProgram`，attach API不能在commit后另接vector重新配对。lower executable typed commitment再同时move-own sealed
executable与该exact registry；不能只保存owner token、profile ID或可从已销毁`CompilationRequest`借用的view。每项context封闭
toolchain/capability/calibration及generation facts，并与executable target requirement逐项重验。它是后续编译stage的typed
operational input，不序列化进artifact/package，也不是generic sidecar。`StagedExecutableToken`绑定这份storage；
`TargetProgramOutputAccess`只能以`(transaction, token, TargetVariantId)`解析对应live context，不能让caller重新传入或替换。
因此原`CompilationRequest`销毁后仍可继续target build；missing/extra/duplicate context、cross-transaction token、context generation
swap或target ID不一致在启动toolchain前失败。

`compiler::StagedExecutableToken`和每个`compiler::StagedTargetArtifactSetToken`是同transaction owner下稳定、
private-construction、move-only/
non-aggregate的identity attachment proof；对应attachment存续期间不因其它target attach失效。transaction另维护单调递增的
`AttachmentTableGeneration`，只为enumeration/snapshot view做失效检测。所有token/view都不能跨transaction或在publication后
使用。`PackageAssemblyInputView`绑定创建时的attachment-table generation，由transaction扫描staged executable的全部typed target
requirements，要求每个required target attachment恰好一次、无extra/duplicate，并只读暴露locator-free verified set
records和capability-bound module/payload sources；caller不能传target数组、选择子集或重排。assembler不能取得mutable
`ExecutableOp`或raw staging path。每次target/package attach只递增attachment-table generation并使旧view失效；它不改变
canonical owner generation或已有staged tokens。至少两个target连续attach后必须能从最新generation构造完整view；任何实现
若混用两种generation都结构化失败。

`PackageArtifactBuildSession`必须保留创建它的attachment-table generation直到attach bundle；期间新增/替换target attachment
会使session及其result token失效，caller只能从最新完整`PackageAssemblyInputView`重新开始。它不能通过刷新单个member把旧/
新snapshot混装；canonical owner token保持同一个，但两种generation不得互相替代。

`ProgramDeliveryCompletionScope`是closed enum：`executable_only`要求一个sealed executable及其全部ArtifactRefs/blobs；
`target_artifacts`在此基础上要求每个committed TargetVariant requirement的完整set；`package_bundle`再要求唯一manifest、
bundle index和external-trust commitments。scope只决定用户明确请求的交付边界，不允许target/package builder缩小coverage。
production full driver使用`package_bundle`或明确的target scope，不能先commit较小scope再“升级”同一delivery。

`ProgramOutputLimits`是transaction factory提供并永久绑定的validated nonidentity预算，所有字段positive checked且0不表示unbounded；
至少限制total logical/physical staged bytes、unique blobs/content roots/inodes、staged metadata/index/commit-record bytes、
open FDs、simultaneous child workers/read-write buffers和retained orphan bytes/age。每个child attach/build先向outer owner做
多维all-or-none reserve；content digest+exact size相同的dedup physical bytes只计一次，logical refs仍全部计数；collision
hard fail。各stage自己的build/publication limits与outer剩余额度取minimum。任一stage失败把transaction标为
uncommittable并释放可回收reservation，不能通过改变partition/chunk/member selection适配预算。

delivery wire由`schema/wafer/program_delivery.proto`唯一拥有。versioned `ProgramDeliveryCommitRecord`是nonsemantic
delivery record，至少包含：completion scope；executable semantic digest与program/module/index exact content
digest/size/safe relative locator；canonical target attachments的TargetVariantId、TargetArtifactSetId、delivery index
digest/size/locator；optional package bundle-index digest/size/locator、manifest blob digest/size和PackageManifestId；以及
所有referenced immutable root IDs。它使用pinned deterministic Protobuf bytes并计算exact
`ProgramDeliveryCommitRecordDigest`，但不分配WCRE record type、不改变model/package semantic identity。unknown fields、
unsafe locator、duplicate/missing scope member和unsupported version失败。`ProgramDeliveryParseLimits`限制record bytes、
recursion、locator/string、target/root/ref counts及totals，parser在reserve前checked累计。

`VerifiedProgramOutputSink`只有两种verified backend语义：同一filesystem的统一temporary delivery root经files/fsync+directory
fsync后单次rename；或先写immutable content roots，再在一个trusted deployment namespace对
`TrustedProgramDeliveryRef(root capability acquisition, commit-record locator, expected exact size/digest, expected
completion scope)`做单一atomic create/CAS。多个独立output directories不能声称multi-rename原子。commit成功返回
non-forgeable `PublishedProgramDelivery`/trusted ref；失败或crash只留下unreferenced immutable orphan，loader绝不接受。

`WaferDelivery`拥有process/deployment-scoped、runtime-neutral的`ProgramDeliveryVerificationRegistry`；其
`ProgramDeliveryVerificationCapability`底层绑定不可转换的`ProgramDeliveryHostBudgetCapability`，全部sessions共享一个typed
child ledger且每项reservation同时计入`HostVerificationRegistry` parent，不能以新建session/registry重置任一层。唯一factory
`createProgramDeliveryVerificationSession(ProgramDeliveryVerificationCapability,
ProgramDeliveryParseLimits, CanonicalEncodingContext)`按值move-own三项，生成non-aggregate、non-forgeable
`ProgramDeliveryVerificationSession`。production
`loadAndVerifyProgramDelivery(const TrustedProgramDeliveryRef &, ProgramDeliveryVerificationSession &)`先验证external expected
commit bytes，再bounded解析/验证scope和所有content commitments，返回move-owned immutable backing/root capability的
`VerifiedProgramDelivery`。它不借用session或parse lease，session栈销毁后仍可派生trusted child refs。Delivery层不依赖
Artifact或Runtime；该session不能打开target/module/package payload。只有verified delivery能派生
`TrustedTargetArtifactSetDeliveryRef`、`VerifiedPackageBundleReference`和migration refs，再交给artifact metadata session。
子root不能自证。GC只回收不被current/retained verified commit records或active leases可达的roots，使用generation/refcount/
fencing避免与reader竞争。debug accept-any record不能派生production refs。

### 3.2 Protobuf Schema and Single Semantic Verifier

Protobuf IDL 是 package wire shape、field number、enum value和 compatibility rule的唯一声明。生成的 C++
类型被 compiler/package assembly和 runtime共同使用；生成的 Python binding仅用于调试工具。结构验证和
跨字段 semantic validation由一份 C++ library实现，至少检查 resource id/use-def、variant guard、entry
graph、ABI slot、endpoint projection、artifact identity和 completion DAG。Python/C++不能各自维护一套
allowed enum、默认字段或跨字段规则。

交付字节由compiler/package writer使用pinned C++ Protobuf producer的deterministic serialization生成，
用于同一toolchain内的reproducible build；但Protobuf deterministic bytes不是canonical semantic
encoding。Package同时拥有两个不同identity：

- `PackageBlobDigest = SHA-256(exact delivered Protobuf bytes)`，用于bundle/content integrity；
- `PackageManifestId`由唯一C++ semantic verifier对typed message构造`tasks/14` WCRE V1 record，
  用`wafer.package-manifest.v1` domain-separated SHA-256计算，用于semantic identity。

runtime必须先对收到的exact bytes检查blob digest，再parse、拒绝当前identity schema version
不识别的unknown fields、执行唯一semantic verifier并重算manifest ID；它不通过跨语言
重新序列化判断identity。schema evolution禁止复用field number，删除字段必须
`reserved`，enum保留zero-valued unspecified，避免required/`Any`承载核心合同。与identity
对应的repeated set/member必须在semantic verifier中按stable raw ID规范化，proto `map`、float、
unknown field和JSON object order不进入WCRE。
JSON printer/parser仅用于debug projection和人工诊断：JSON输出必须从已验证message生成，JSON输入若
保留只允许进入debug/converter入口，不能绕过C++ semantic verifier成为production package。schema v2
JSON的迁移路径固定为：

```text
schema v2 JSON
  -> explicit v2 converter
  -> typed PackageManifest
  -> current C++ semantic verifier
  -> WCRE semantic identity + deterministic Protobuf delivery bytes
```

Converter必须拒绝无法无歧义恢复的 binding order、workspace、completion和 module identity；不能用默认
值把旧 package伪装成新合同。converter存在不代表 runtime长期支持两套 schema。

### 3.2.1 Parse and Admission Limits

`PackagePublicationLimits`、`PackageParseLimits`和`RuntimeAdmissionPolicy`是caller/deployment提供的validated
operational inputs，不属于manifest semantic identity、artifact fingerprint或compiler schedule。它们的字段是
positive checked integer，不能用0表示unbounded，也不能由manifest覆盖。

`PackagePublicationLimits`约束package assembly/build service，至少限制blob count、单个及total module bytes、
单个及total payload bytes、manifest/index bytes、staging/output disk bytes、simultaneous source readers、open FDs、
worker/read buffers和concurrent copied bytes。assembler从已验证TargetArtifactSet和committed ArtifactRef先用checked
arithmetic完成纯preflight；manifest/index用bounded counting writer取得exact upper/exact size。通过前不得创建bundle
output、staging file、打开source blob或启动worker。materialization期间实际计数只能等于preflight结果；超限或磁盘/
reader故障回滚整个`ProgramOutputTransaction`，不发布partial bundle。该limits不得丢member、改变
`ArtifactChunkingPolicyV1`、合并blob或重写compiled schedule。

`PackageParseLimits`至少限制exact delivery bytes、Protobuf recursion depth、locator/string bytes以及resources、
realizations、state groups/members、target sets/members、modules/payloads/chunks/windows、variants/ranks/entries/slots、
expression AST nodes/depth、iteration domains和template graph nodes/edges的单项与总数。loader在Protobuf
`CodedInputStream`或等价parser层先设置byte/recursion limits，每读一个repeated/nested record先checked累加再
reserve/construct；不能先按wire count分配vector再做semantic verifier。unknown fields仍按schema规则拒绝，不因
limit存在而跳过。

`RuntimeAdmissionPolicy`至少限制selected invocation的template nodes/edges、checked total instantiated work、
max live cursor frontier/iteration span、resource/scope instances、host
metadata bytes、per-arena及total requested bytes、simultaneous staging windows/copy/events/commands、active blob
leases/FD、loaded module count、selected module content bytes、simultaneously verified module bytes、peak mapped/
resident module code bytes以及concurrent immutable-window host transfer buffers/bytes。`preflightRuntimeSession`在typed
model entrypoint、actual shape/policy/count expression选择和求值后，以`VerifiedRuntimeEnvironmentInventory`运行纯
size/liveness/domain-selection preflight，所有sum/product/ceildiv用checked arithmetic；初始effective hard limit是deployment
policy与inventory provider-declared capability的逐项minimum。该阶段只有side-effect-free facts，在通过前不得取得authority、
open blob、allocate、load module或submit command。exact-domain context形成后必须用其
`VerifiedRuntimeEnvironmentSnapshot`重放plan/domain generation和limits；任何收紧或变化在reservation/open前失败。

limit failure返回typed `parse_limit_exceeded`、`admission_limit_exceeded`或`size_overflow`并指出stable owner ID与
limit kind，source manifest、registry/cache和provider side-effect counters保持不变。limits只决定当前deployment是否
接纳已编译语义，不允许截断graph、选择canonical prefix、减少microbatch、合并window或改变variant/schedule。

单invocation preflight通过不等于共享provider容量已取得。`VerifiedRuntimeEnvironmentSnapshot`拥有canonical
`RuntimeCapacityDomain` set；每个domain绑定provider identity、device identity、epoch/generation、arenas/queues和
projection endpoints，每个ExecutionInstanceId/rank必须映射到exact domain。RuntimeSession把preflight产出的immutable
`InvocationResourcePlan`按这些domains分解，交给`RuntimeCapacityReservationCoordinator`，在任何blob open、
allocation、module load或command submit前对全部domains做一次跨域、多维all-or-none reservation。plan至少覆盖per-arena bytes/
allocation count、module slots和mapped/resident code bytes、commands/events/copies/windows、active blob leases/FD、
host metadata、verification/read/transfer buffers及provider声明的其它finite queues。reservation使用同一
snapshot domain generations和effective hard limits；逐字段先checked add当前usage。V1进程内managers按canonical
domain ID获取locks/prepare tokens，全部domain验证成功后一起commit counters；provider-backed reservation若存在则按同序
prepare，任一不足/generation change/cancellation都反序rollback全部prepared domains。失败不占用任何domain字段并返回
typed `runtime_overloaded`/domain/capacity kind。禁止逐rank先成功后失败留下partial ownership。

cache accounting必须计算marginal capacity：已存在且compatible的provider code/resident weight cache lease由cache
owner持有intrinsic capacity，invocation只取得引用和自己的command/window等额度；cache miss先预留prospective
insertion，single-flight waiters共享一个insertion reservation，失败/loser不能double count或发布partial cache entry。
arena/session/invocation allocations按declared scope计费。reservation token按compiler DAG/liveness的last terminal
completion增量释放window/event/command/transient bytes，session/cache-scoped部分在对应lease close/evict后释放；
failure/cancellation cleanup也走相同release edges，不能提前释放仍可能被issued command访问的容量。

coordinator为等待模式分配snapshot-local monotonic FIFO ticket；head request在全部required domains/维度可用时原子取得，或被其
caller deadline/cancellation移除，后续request不能永久越过它。显式nonblocking模式可立即返回overload但不入队；两种
模式都不能重排manifest DAG或降低request语义。任一participating provider/device epoch/generation变化原子invalidate
整个invocation token，同时各domain manager只失效自己的cache-capacity ownership，阻止旧handle submit；新snapshot
重新做全域preflight/reserve。并发sessions的测试必须证明每域及合计
从不越provider hard limit、无部分reservation泄漏、bounded-wait request不饥饿。

跨process共享device时不能把进程内counter当全局事实。`VerifiedRuntimeEnvironmentSnapshot`必须包含closed
`ReservationAuthorityMode`：`exclusive_local`要求deployment authority为完整capacity-domain set签发带expiry和
monotonic fencing generation的exclusive lease，process-local coordinator每次reserve/submit都验证；
`provider_or_central_atomic`要求provider/central reservation service对上述multi-domain prepare/commit/rollback提供
cross-process原子/fenced API。trusted inventory明确device是否可能共享。共享deployment既无exclusive lease也无central
authority时RuntimeSession creation返回structured unsupported，在任何side effect前失败；不能假定当前进程独占。
lease expiry、stale fence或authority generation change使整session停止新issue并按failure cleanup，旧process无法release/
publish新generation ownership。

### 3.3 Typed Resource Contract

每个logical resource使用稳定`ResourceId`，记录role、semantic dtype/symbolic shape bounds、access、alias set、
lifetime和artifact/source relation；其typed realization records显式引用`TargetVariantId`、
`ExecutableVariantId`、`RankClassId` coverage和必要`ProjectionSetId`，并记录storage layout/capacity、alignment、
`DdrArenaId`/placement domain、scope、shard/packing和accepted range relation。`DdrArenaId`必须引用selected
target environment/variant中fingerprinted arena declaration；manifest verifier按
`(ResourceId, TargetVariantId, ExecutableVariantId, ProjectionSetId)`建立canonical sorted index，由entry slots导出
required RankClass sets并用coverage set sweep/merge join证明每个实际使用tuple恰好唯一命中。禁止物化自由axis
笛卡尔积，也不能按缺字段/wildcard/default补realization；state/window/transport用typed key join同样闭合。
role enum与`tasks/01`唯一集合一致：

- `external_input` / `external_output`：invocation-scoped host/runtime binding；actual shape必须满足 variant
  guard和 declared bounds。
- `immutable_parameter`：content digest、shard mapping、packing/quant descriptor、read-only access和typed
  `resident | streamed_windows` residency。resident可跨invocation复用完整allocation；streamed只能使用manifest
  已声明的window、staging resource、copy/consumer completion关系，不能创建全量weight lease或改变bytes/layout/shard。
- `persistent_state`：跨 invocation mutable state；记录 create/attach/reset/update/close语义、alias/update
  relation、capacity和唯一`StateConsistencyGroupId` ref；failure/snapshot policy只由selected executable
  `StateGroupRealization`拥有。paged state额外记录 page geometry、page-table/control resource、logical
  length和maximum capacity；KV cache只是该通用合同的一个实例。不能用free-form repair/default policy
  让partial completion静默成为可复用 state。
- `workspace`：variant/rank/stage-scoped ephemeral allocation；大小是 verified expression或上界，不能跨
  invocation保留 identity，也不能遗漏在 kernel ABI slot sequence中。
- `staging`：仅用于launch-visible、跨entry或resident transfer root；entry内部可从instruction IR重算的SPM
  temporary/staging不进入manifest。
- `control` / `status`：relocatable endpoint table、channel/FSM assignment、page table、completion/error和
  profiling buffer；它们不是普通 tensor input。

manifest必须逐项序列化committed executable state groups。每个group record拥有canonical nonempty member set和
axis-covered `StateGroupRealization`；selected tuple只能是`atomic_version + (full_copy | page_cow)`或
`in_place_poison_on_failure`。atomic realization还拥有typed per-member bounded update footprint、snapshot-ready和
group-publish completion refs；in-place realization拥有exclusive group-epoch acquire和group poison/reset refs。
member resource不能复制一份可独立变化的consistency policy。

low-precision resource保留shared `QuantizationDescriptor`、per-realization `StorageEncodingDescriptor`和selected
artifact member实际使用的typed `QuantStorageAbiProfileId` refs/profile records；member fingerprint的
`usedQuantStorageProfiles`必须从这些refs重算。manifest verifier交叉检查launch-visible packed/scale/zp/staging或
cross-entry scratch resources、slot
shape/capacity/alignment、target capability/profile/fingerprint和KAD geometry；runtime不解释量化公式、不重新pack或
选择native/composite路径，只绑定compiler已提交的ordinary resources和执行entry DAG。unsupported/mismatched
profile必须在blob open、allocation、module load或submit前失败。

一个entrypoint按`KernelAbiDescriptor`的ordered `SlotId` sequence调用；committed executable entry和manifest
entry record提供唯一`SlotId -> (ResourceId, StateSlotVersionRole)` binding。manifest不另存自由`binding_order`；semantic verifier
要求descriptor slots、resource declarations和entry bindings一一对应、无重复、无遗漏。

### 3.4 Model Invocation, Executable Variants, Rank Classes, and Entry Graph

`PackageManifest`逐项保留model interface的ordered `ModelEntrypointId` records，并对每个executable variant
序列化typed invocation mapping：model entrypoint ID、initial entry/completion template roots、required external
IO/state-group contract和terminal output/state/completion roots。mapping只能引用同variant已有graph records；
`EntryId`仍只标识static kernel entry，不能充当用户调用API。每个model entrypoint在所有宣称支持它的variant中都必须
保持同一model IO/state contract，缺root/terminal、跨variant错误contract或悬空graph ref失败。

Runtime选择顺序固定为：先验证request的typed `ModelEntrypointId`并限定支持该API的variant，再用actual target environment/topology筛选兼容`TargetVariantId`和
`ProjectionSetId`，再对剩余`ExecutableVariantId`的typed `ShapeGuardRef`求值一次，最后使用该variant已提交
的rank mapping。shape guard只能读取actual dimensions、state capacity和显式invocation policy；target
predicate只能读取typed capability/fingerprint/topology class。两者都不能执行任意脚本或匹配名字。
verifier检查target filter和shape guard各自priority/fallback确定，并要求得到唯一compiler-validated tuple。

`RankClassId`只由whole-variant executable commit产生。manifest Entry逐项序列化nonempty canonical
`RankClassId` coverage和canonical execution-instance coverage、
 module/`KernelAbiDescriptor`/`SlotId -> (ResourceId, StateSlotVersionRole)` bindings/transport/completion refs，并验证同一class记录一致；它不根据这些
字段重新归并或拆分rank。entry graph node记录stage、committed rank-class coverage、module/static `EntryId`、argument
slots和host-visible dependency；edge只能表达跨module/stage/runtime可观察的launch依赖，不能复制module
内部compute或p2p instruction schedule。

manifest还逐项序列化`tasks/01`的finite `EntryGraphIterationDomain`、bounded count expression、max count、
node domain binding和edge `iteration_delta`。semantic verifier重放expression bound/overflow、zero-delta DAG、
positive-progress cycle、expanded terminal coverage和resource-scope relation。RuntimeSession在variant选择后只求值
一次checked-u64 expression，要求final actual count通过u32 narrow proof，再由rolling cursor机械生成
`(node, uint32 iteration)`；它不接受caller提交展开schedule，也不根据provider吞吐改变
microbatch count/order。

“机械实例化”不要求把全部iterations存成host vector。preflight创建non-forgeable
`VerifiedGraphInstancePlan`，保留verified template、actual domain counts、checked total work、resource/module/window
symbolic liveness、每类peak和compiler/verifier证明的`max_live_iteration_span`；它用symbolic/merge-join算法验证terminal
coverage和admission totals，不分配O(total iterations) objects。若edge/lifetime使live span无法有限证明，compile/package
verify失败。

每个runtime capacity/module/resource demand还必须携带typed `SymbolicTerminalInstanceRule`，不能只保存terminal template
node key。V1 closed union至少表达：`same_iteration_plus_fixed_delta`（typed iteration-domain tuple和checked delta）、
`domain_final_iteration_minus_fixed_tail`以及`invocation_terminal_set`；规则同时绑定demand owner/domain、terminal node/domain、
release kind和最后安全use/completion relation。verifier对每个symbolic demand证明恰好一个可达terminal instance、terminal不早于
任一consumer/issued command、delta/final-boundary不越界，并区分per-instance lease与覆盖未来iterations的aggregate lease。
cursor只在demand进入live horizon或final boundary已知时实例化对应terminal key；它不能因iteration i结束就释放仍覆盖
iteration i+1的module/resource，也不能为百万iteration预建release vector。

executor从该plan创建private `VerifiedGraphInstanceCursor`，按canonical template-node/domain/iteration order只物化
bounded ready/issued/live frontier。V1 edge delta为0/+1，cursor维护per-template next iteration、bounded dependency state、
completed watermarks和failure summary；过了所有resource/module/window last-use的iteration立即释放instance state，不能
因profiling/debug保留整图。frontier capacity由manifest proof、RuntimeAdmissionPolicy和capacity reservation共同限制，
不足时在issue前失败，不能改变order/count。cursor生成的`EntryInstanceId/ScopeInstanceId`与full expansion公式一致；
small graph differential test必须逐node/edge/result相同。dry-run/profiling只向bounded streaming sink输出trace，trace
limit超限截断diagnostic并标记，不影响执行/semantic result。

manifest也逐项序列化finite activation predicates/expert waves：typed `ActivationPredicateId`、
`bounded_count_nonzero`的count ResourceId/expert-member ordinal/offset/type/capacity/count-complete ref、每个node的
always/true/false activation、predicate-evaluate/conditional-join refs、wave member set、resource/module/window first-use/
last-use、最大active count和capacity envelope。semantic verifier证明count producer支配evaluate、active/skip两条path都
到达join、copy/module/entry/data仅在true path、combine/reuse被join支配、conditional region不写persistent state/
user terminal，并由typed model-member/component/entry关系闭合；runtime不能添加predicate或wave。

RuntimeSession在count-complete后从declared control instance exact range读取并检查bound，一次求值predicate。false path
把受控nodes标为`skipped_success`并直接完成skip join，不open expert blob、load module、copy/submit；true path从initial
`RuntimeCapacityReservation`已经保留的compiler wave envelope取得actual member sublease，再机械执行原DAG。若actual
active count超过declared max/envelope则failure，不可少选expert。expert cache只按exact member/source/storage/provider
key命中，eviction是capacity实现且不得改变binding/wave/order。provider不支持conditional issue时在首次side effect前
拒绝该manifest，不退回always-load-all。

Pinned projection把rank-to-endpoint mapping和projection fingerprint绑定到`ProjectionSetId`，pinned module
再通过`TargetArtifactFingerprint`记录该projection dependency；`TargetEnvironmentFingerprint`保持独立；
RuntimeSession只验证并使用。Relocatable projection严格序列化`tasks/04`的typed union：有序有限
`ConcreteRecordSet`，或template加有限`allowed_bindings`的`FiniteTemplateSet`，以及endpoint/channel/FSM/
control resource slots。manifest verifier复核每个member的compiler proof/digest；RuntimeSession只按确定性
优先级选择compatible member并机械填表，不能生成新binding或搜索placement/route。同一node不能同时携带
pinned mapping和relocatable slots。

### 3.5 Device Code Compile/Link Gate

Device-code gate消费target LLVM call-emission生成的LLVM IR、`KernelAbiDescriptor`和
`TargetEnvironmentFingerprint`，不消费`wafer.instr.*`、`func.call` ABI IR或package test input。它负责把device kernel
编译成 TX8 runtime 可以装载的 kcore shared object，写入/验证`tasks/14`固定的
`.note.wafer.abi` mandatory descriptor note，计算 final
module digest，并把final module content digest、descriptor semantic digest、environment/artifact fingerprints和
artifact locator交给package
assembly。该 module 可以被
`tx.module` / `tx.cluster` entrypoint 直接使用，也可以作为未来 `tx.model` / `tx.graph`
materialization 的组成部分；module 本身不决定 package 的模型级执行语义。

Package assembly只消费`tasks/14`原子发布的完整`TargetArtifactSet`。它不能扫描staging目录或把部分
rank-class modules拼成target variant；set中的`TargetVariantId`、final module content digests、descriptor semantic digests、projection
dependencies和digests必须与committed executable coverage一一闭合。

当前必须区分两个事实：`tools/wafer_device_link.py` 已能对已有 / compiler-generated LLVM IR 执行
`.ll -> .o -> kcore .so` 的本地 compile/link；但主线 device-code gate 还没有完成，直到
required-symbol 检查证明 `wafer_tx81_*` 这类 Wafer-owned target CRT symbol 由 repo-local
Wafer CRT source/object 或明确合法外部依赖解析。单纯因为 `--allow-shlib-undefined` 生成 `.so`
不能作为完成证明；在 descriptor/hash/fingerprint/digest gate落地前也不能作为 typed module完成证明。

V0 采用已经可运行的 TX8 RISC-V compile/link 工具链 profile，并在 final link 前做 object
metadata normalization：

```text
LLVM IR (.ll)
  -> LLVM clang++ RISC-V compile
       kernel.o
  -> Xuantie GNU ld compatibility normalization
       kernel.o without incompatible .riscv.attributes metadata
  -> repo-vendored tx8_deps riscv64-unknown-elf-gcc link
       kernel.so
```

第一段用 LLVM `clang++` 从 `.ll` 生成 RISC-V relocatable object：

```sh
clang++ kernel.ll -O2 -c -fPIC \
  --target=riscv64-unknown-elf \
  -march=rv64imafdc \
  -o kernel.o
```

进入 Xuantie GNU ld final link 前，gate 会对 LLVM `clang++` 生成的 object 移除
`.riscv.attributes`：

```sh
riscv64-unknown-elf-objcopy -R .riscv.attributes kernel.o
```

这是 object 兼容性 normalization，不是 IR 语义。pinned LLVM工具链生成的object可包含把
`rv64imafdc`展开为`zaamo`/`zalrsc`的split-extension attribute，而vendored Xuantie GNU ld 2.35
不能解析该attribute字符串。代码段仍按`-march=rv64imafdc -mabi=lp64d`生成，final link继续由
Xuantie GCC driver选择对应multilib；LLVM版本事实只从集中dependency pin读取，不在本合同重复固定。

最后用 repo-vendored `third_party/tx8_deps` 中的 RISC-V GCC 链接 kcore shared object：

```sh
riscv64-unknown-elf-gcc -shared -march=rv64imafdc -O2 \
  -nostartfiles -Wl,--allow-shlib-undefined \
  -mabi=lp64d -Wl,--no-dynamic-linker \
  kernel.o \
  runtime/wafer_crt/wafer_tx81_crt.o \
  -Lthird_party/tx8_deps/<tx8-toolchain>/riscv64-unknown-elf/lib/rv64imafdc/lp64d \
  -Lthird_party/tx8_deps/<tx8-toolchain>/lib/gcc/riscv64-unknown-elf/10.4.0/rv64imafdc/lp64d \
  -Lthird_party/tx8_deps/lib \
  -Wl,--start-group \
  -lcommon_util -linstr_tx81 -llibc_stub \
  -Wl,--end-group \
  -lm -Wl,--gc-sections -Wl,--unique=.rodata.name \
  -lc -lgcc \
  -o kernel.so
```

这里 `.ll -> .o` 不能交给 GCC；GCC 只负责 final link。`libcommon_util.a`、
`libinstr_tx81.a` 和 `liblibc_stub.a` 来自 repo-vendored `third_party/tx8_deps/lib`；Wafer-owned
`wafer_tx81_*` symbol closure 来自 repo-local Wafer CRT source/object，而不是旧 `libvr.a` archive
或 TX81/Triton `__*` symbol。若某个 public wrapper 仍需要
lower-level archive 作为实现依赖，必须作为显式底层依赖进入 link，但不能成为 Wafer compiler
target CRT ABI 的事实源。V0 profile 固定为 `rv64imafdc/lp64d`，因为这是当前 vendored Xuantie
toolchain 实际提供的 64-bit double-float multilib；`-mcpu=c908` 或其它 Xuantie multilib profile
需要单独的 object 兼容性和板端验证后再升级成新 profile。

`-Wl,--allow-shlib-undefined` 只允许 kcore shared object 保留由 target/runtime ABI version明确列出的
外部符号；它不是
证明缺失 target CRT symbol 可以被忽略的信号。当前 device-code gate 不再默认编译或链接
capture shim；LLVM IR 中出现的 target CRT symbol 必须来自 repo-local Wafer CRT source/object
或明确由 runtime/loader 解析。仍可能存在的 unresolved symbol 必须来自合法外部依赖，不能来自
已删除的 helper ABI 层。required-symbol gate 必须把未解释的 `wafer_tx81_*` undefined symbol 当作
失败，而不是把链接器成功返回当作 target support 完成。

Final link还必须写入/验证`KernelAbiDescriptor` mandatory note，检查ELF machine、ISA/MABI和environment/
artifact fingerprints，再计算覆盖final ELF的module digest。Package assembly只能引用该descriptor semantic digest、
两类fingerprints和digest；artifact locator改变不改变identity，文件内容改变必须导致digest mismatch。

## 4. Runtime Layering

稳定主路径分层：

```text
WaferRuntimeAdapter
  -> WaferRuntimeServiceRegistry
       -> VerifiedRuntimeEnvironmentInventory
       -> exact-domain WaferRuntimeServiceContext
            -> repo-owned final managers/executor
            -> RuntimeProviderCapabilitySet
                 -> dry-run/fake low-level provider
                 -> tx_runtime/KMD low-level provider
                 -> restricted legacy low-level provider
```

已知事实：

- `tx_runtime.h` 是当前主 host runtime ABI，覆盖 device、memory、stream、event、module、kernel、
  model、graph、rank、tile、P2P。`libhpgr.so` 是当前可见 provider 名称，不进入 Wafer package
  schema 或 adapter 类名。
- KMD/UAPI 负责 `/dev/accel/dev-N`、runtime allocation object、jobs、NPU tile memory、C2C、log、device
  info、topology、driver-level DTE ioctl、BAR/ATU 和 firmware loading。
- Legacy `Tsm*` / VS runtime 是兼容和证据层。

`RuntimeProviderCapabilitySet`是低层机制边界，不是可替换semantic backend。它是canonical bounded
`RuntimeProviderDomainCapabilitySet` vector；每个domain set内部的side-effect-free environment query、module load/unload/resolve、
allocation/import/copy/readback、transport、command submit和event/completion poll/wait/status/cancel capabilities绑定同一provider
owner token/generation。deployment reservation-authority factory与transactional durable-state service可以有各自verified owner/
generation，但registry必须显式验证它们与domain sets的device/namespace/fencing relation。该结构允许heterogeneous providers，
不能把不同domain错误压成单owner。任何子capability都不能构造
verified environment/module/allocation/state/execution proof或选择semantic key。Tx、dry-run/fake和legacy只实现这组低层能力；
production caller不能传`RuntimeBackend &`、manager callback或同identity的自造fake越过registry。
transport是独立低层capability：它只接收repo-owned executor从已验证`TransportBindingMemberId`/relocation facts构造的typed
mechanism request，并返回绑定同一domain/provider generation的raw submission receipt/completion handle；不能选择endpoint/member、
构造verified submission或自行宣告completion。ordinary command capability不能暗中代替缺失transport capability。
copy/readback capability同样只从selected resource/window/range facts接typed request并返回raw completion receipt；仅返回success
而没有可归属、可等待的completion handle不能实现async copy node，也不能被final executor接受。

repo-owned final executor以entrypoint graph消费model package，不以某个TX API名字作为主合同：

```text
PackageManifest
  -> RuntimeSession
       -> target/projection compatibility filter
       -> one global shape-guard evaluation / executable-variant selection
       -> committed rank mapping materialization
       -> device selection / PG tile selection
       -> weight cache / persistent state registry / workspace allocation
       -> pinned projection validation or relocatable endpoint materialization
       -> rank/stage entry graph materialization
            -> tx.model        # model-level path, requires BPM descriptor/table
            -> tx.graph        # graph artifact descriptor
            -> tx.module       # kernel-level bring-up/debug entrypoint
            -> tx.cluster      # cluster-kernel bring-up/debug entrypoint
            -> legacy.tsm      # restricted fallback
       -> completion DAG execution / timeout / error aggregation
```

`txLaunchModel` / `txLaunchModelSync` 是 TX runtime 的模型级执行面，但当前 compiler package 还没有
完整 BPM table schema。V0 只能把 `tx.model` 表达为 descriptor/materialization contract；如果
`PackageManifest` 声明的 BPM descriptor 仍是 `descriptor_only`，runtime adapter 必须结构化拒绝真实 launch，
不能隐式退化到 `txLaunchKernel`。

`tx.module` / `tx.cluster` entrypoint 需要显式标记为 `debug=true`。它们可以用于
no-card command construction、device-code smoke、HF transformer compile/package intake gate 和板端
bring-up，但不能作为模型级 board launch / correctness 完成证明。

Provider API由 C++ backend将 typed entry/completion nodes映射到具体调用。provider可以优化同一 DAG，
但不能改变 resource alias、rank/stage dependency、completion/error scope或把selected target/shape tuple
拆成rank-local选择。symbol存在只证明 capability discovery，不证明 entry graph已经执行。

Legacy fallback：

- `TsmRun` 可以作为 bring-up fallback：bootparam device pointer 经 runtime physicalization 后
  调 `txLaunchModelSync`。
- `TsmLaunch/TsmLaunchPg`、`TsmAsyncRun`、`TsmDeviceSynchronize` 当前不能作为 correctness fence。
- `TsmGetDeviceNum/List/Properties` 当前不能作为 capability discovery。
- `TsmMemcpyOffsetH2D/D2H` 不能作为 offset copy correctness path。
- `TsmMemcpyD2D`、`TsmSend`、`TsmRecv` 是 host runtime dyn TLV + Kcore DTE path，不等价于
  compiler inline Direct DTE。

## 5. Runtime Allocation and DDR Binding

`PackageManifest` 不静态保存最终 physical address；它保存 typed allocation / binding requirements，
`RuntimeSession` 在 package load / invocation / entry graph materialization 时执行：

- allocate / import runtime allocation object。
- query physical address、size 和 runtime capability/resource metadata。
- validate external input/output binding。
- attach/create/reset persistent state group，并验证member set、page geometry、logical length、capacity和snapshot policy。
- 对resident immutable weight取得完整verified cache lease；对streamed weight只保留selected `BoundBlobSource`，
  按compiler-declared window向bounded staging allocation提交异步copy。
- 为selected executable variant中已提交rank/stage mapping分配隔离workspace。
- materialize relocatable endpoint/control/status resources或验证 pinned projection。

Typed resource contract 必须区分：

- user/runtime external input/output binding。
- persistent mutable state group、member snapshots/page table和state control metadata。
- immutable weight/resident constant及其content identity，以及streamed source/window/staging relation。
- compiler workspace runtime allocation object demand。
- executable/BPM/graph/log/control metadata allocation。它们是 runtime/package 内部对象，不是 generic tensor
  DDR planning arena。

KMD/UAPI的低层分配类别只作为runtime mapping evidence使用；DDR memory planning产出accepted ranges，
pre-commit `ExecutableResourceView`已将resource semantics materialize为committed executable objects。
target LLVM只派生address/range参数，manifest只序列化这些objects，runtime adapter只实例化
external binding、persistent state、workspace、immutable weight和control/status requirements。runtime adapter在
package load / launch 时执行 allocate/import/query/bind，并报告
runtime allocation failure；不能在 runtime/package 层重新决定 DDR range plan。

### 5.1 Non-serialized RuntimeSession Contract

`RuntimeSession` 是 C++ host adapter 从已验证 `PackageManifest`和一个已preflight invocation派生的进程内对象，不是 compiler IR、
wire format或可缓存 sidecar。每个session执行exactly one typed invocation；连续prefill/decode或并发请求创建多个session，
通过同一`WaferRuntimeServiceContext`共享cache/state/capacity而不复用invocation-local plan/control。它拥有真实 provider handles和 invocation状态，销毁时负责有序释放；这些
facts不能回写 manifest或用于下一次编译恢复语义。

RuntimeSession 至少包含：

- package/target identity：已验证的manifest digest、selected `TargetArtifactSetId`、source
  executable digest，以及selected member map中的module digest/descriptor semantic digest；不能用
  单个module代表整个set。
- invocation facts：request-selected typed `ModelEntrypointId`、actual dimensions、selected
  `TargetVariantId`/`ProjectionSetId`/`ExecutableVariantId`、
  request IO、typed `InvocationPolicyFieldId -> uint64` values和persistent state attach/create/reset policy。
  `InvocationRequest`只接受typed model entrypoint ID；unknown ID、与该API不匹配的IO/state contract、required field
  缺失、duplicate/unknown/out-of-range policy value都在任何provider side effect前拒绝，optional fields才使用manifest
  default。CLI/API alias若保留，只能先通过已验证manifest中nonidentity且唯一的diagnostic alias table解析为ID；
  不能把`--entrypoint forward`字符串、LLVM/function symbol或static `EntryId`直接交给RuntimeSession。actual shape/policy只实例化symbolic bounds/expression，不修改
  manifest type事实；rank class不在session中重新选择。
- resource instances：external allocations、resident immutable weight cache handles或streamed immutable source +
  bounded staging allocations、persistent state group snapshot/page-table handles、rank/stage workspace和
  control/status/profiling buffers。RuntimeSession先按已选target/projection/shape和committed
  rank class形成singular
  `ResourceRealizationLookupKey = (TargetVariantId, ExecutableVariantId, RankClassId, ProjectionSetId)`并取得唯一
  typed realization，再形成instance key
  `(ResourceId, selected realization tuple, ScopeInstanceId)`：executable/session-shared scope使用canonical shared key，rank/stage/entry/
  invocation scope分别加入`ExecutionInstanceId`或`EntryInstanceId`，不能用单一`ResourceId -> handle`覆盖
  多个workspace instances。lookup key是runtime query，不包含coverage set，也不能替代serialized
  `ResourceRealizationRecordKey`；semantic verifier先证明singular tuple恰好命中一个record，再返回immutable view。
- arena instances：`(DdrArenaId, ScopeInstanceId) -> runtime allocation object/base/capacity`；每个resource
  instance必须引用compatible arena instance，RuntimeSession在launch前执行actual-base alignment/range/address-
  width gate，不能把不同placement domains折叠到一个隐式default arena。
- placement instances：device/PG、pinned projection validation结果或 relocatable rank-to-endpoint、channel/FSM /
  stream/event assignment。
- entry graph instances：`EntryId`、canonical `ExecutionInstanceId` coverage、committed `RankClassId`、
  module/function/provider executable，以及按`KernelAbiDescriptor.SlotId -> (executable ResourceId,
  StateSlotVersionRole) -> scoped
  resource instance`生成的launch arguments；iterated node同时携带verified
  `(IterationDomainId, iteration)`。禁止按名字、vector顺序或参数数量猜测。
- completion instances：manifest completion DAG节点对应的 provider command/event、device drain、DTE /
  collective wait、timeout、status/error和 rank/stage failure aggregation。

`preflightRuntimeSession`按值接收并把nonserialized `InvocationControl` move入`RuntimeSessionPlan`：可选
`std::chrono::steady_clock::time_point` absolute deadline和thread-safe cancellation token。bind/session保留同一owner，create或
execute不能替换。它不是manifest字段、semantic identity、shape/policy input或schedule选择器；取消只会
终止当前`InvocationId`，不能缩短microbatch/window graph、改变variant或影响其它invocation。executor必须在首次
issue前、每个ready issue wave前后、poll completion边界和进入blocking wait前观察control；provider `waitAny`使用
`min(node timeout deadline, caller deadline)`，不能因长node timeout忽略caller deadline。

deadline/cancellation触发与failure相同的typed cleanup路径：不再issue non-cleanup descendants，抑制未issue nodes，
对已issue command执行provider cancel或安全terminal wait，再按DAG释放资源。尚未publish的`atomic_version`整组
candidate必须discard；只要`in_place_poison_on_failure`的任一write可能已issue，就必须先durably poison整组再返回。
首次被观察的provider/device failure优先于其后的caller cancellation，cancellation/deadline优先于cleanup failure；
result分别保留typed `cancelled`或`deadline_exceeded`。pre-issue cancellation必须保持open/allocate/load/submit为0。

deployment/process-scoped repo-owned `WaferRuntimeServiceRegistry`先move-own单一move-only、non-aggregate
`VerifiedRuntimeServiceBootstrap`。它的唯一factory分别从provider adapter取得完整canonical
`RuntimeProviderCapabilitySet`，从互不相同且deployment-authenticated的bootstrap access取得`TrustedDeploymentInventory`、
deployment reservation-authority capability和optional durable-state service capability，再move-consume不可转换的
`RuntimeArtifactHostBudgetCapability`；provider adapter不能签发inventory/authority/state/host trust。factory在seal前逐domain验证
provider owner/generation、inventory facts、authority coverage/fence和durable namespace/fencing relation，bundle内部唯一拥有全部输入。
host capability只由lower runtime-neutral layer自身验证liveness/provenance并独立执行physical budget；它不与provider deployment
owner做semantic equality、不授权provider，也不需要Runtime friend/forward declaration。registry factory只消费该bundle，不接受原始
capabilities、独立cross-service proof或可在proof后替换的参数。
host capability本身不能创建runtime artifact session，只有exact-domain context把它投影到service owner/generation后才可签发
`ArtifactVerificationBudgetCapability`。每组provider capability绑定opaque provider owner token和
epoch/generation，不能单独替换。registry先只通过environment-query capability与trusted inventory形成side-effect-free、
owner-backed `VerifiedRuntimeEnvironmentInventory`，供package metadata preflight选择exact target/projection和capacity-domain set；
inventory不能allocate/load/submit，也不授予device authority。

只有move-only `RuntimeSessionPlan`确定canonical完整domain set后，registry才能为该plan内部取得
`ReservationAuthority`并调用`getOrCreateContext(RuntimeSessionPlan)`。返回的shared、non-aggregate
`WaferRuntimeServiceContext`按exact domain set、provider owner/epoch/generation和authority fencing identity single-flight；同key的
所有RuntimeSessions取得同一个context。`exclusive_local` authority lease只能被一个live context消费；central/provider authority
也由registry复用。context move-own该domain set需要的完整module/memory/command/completion/state capability projections，不从caller
接收`RuntimeBackend &`或零散provider对象。generation失效后拒绝新session/issue并创建新key，旧context只能cleanup；不能通过
直接构造第二个context绕过exclusive fence或全局counter。

`WaferRuntimeServiceContext`唯一拥有共享`RuntimeCapacityReservationCoordinator`、恰好一个runtime artifact operational ledger以及提供semantic authorization的repo-owned
concrete final managers：
`WaferModuleResidencyManager`、`WaferResidentWeightCache`和`WaferPersistentStateRegistry`。production `RuntimeSession`
factory/executor不接受`RuntimeBackend &`、`ModuleResidencyManager`/`WeightCache`/`PersistentStateRegistry` abstract base、derived
override或callback。context内低层capabilities只做load/unload/query、allocate/copy/import/readback、submit/poll/wait/cancel及
fenced journal/batch-CAS I/O，不能构造verified module/allocation/state lease、决定semantic key或跳过manifest join。只有repo-owned
final managers是proof/lease constructor friends。RuntimeSession只retain由`RuntimeBoundPackage`带入的同一个shared context并申请
session/invocation leases，不自行new coordinator/cache/registry。unit-test fake只能实现完整低层capability contract；显式test-only
registry位于unittest target且不能链接production runtime，不能直接返回semantic proof。这样provider/store可替换而语义验证、
cross-session capacity、artifact accounting和atomicity owner唯一。

context另外通过`runtime::detail::RuntimeArtifactBudgetAccess`的私有adapter签发
`artifact::ArtifactVerificationBudgetCapability`；该adapter绑定context的service owner token、generation和唯一artifact ledger，
但只暴露上述四类operational reservation。所有由同context签发的capability/session共享ledger。runtime-private joint access把
artifact lease demand和当前invocation capacity token按稳定owner顺序nonblocking prepare/commit，失败全量rollback，绝不在持有
一侧reservation时等待另一侧。`RuntimeBoundPackage`构造时将verification session的service-owner/generation与context和plan
比较；不匹配、已失效或已move必须在blob open和provider side effect前失败。`WaferRuntimeServiceContext`/该private adapter是
production中唯一issuer；调用方不能自己构造capability，`WaferArtifact`也不反向调用Runtime factory。

Instance/version identity owner固定如下：

- `ExecutionInstanceId`来自committed executable并序列化进manifest；runtime不能重编号。
- `InvocationId`由RuntimeSession创建且只在一次invocation内有效。`EntryInstanceId`由
  `(InvocationId, EntryId, ExecutionInstanceId, graph iteration/microbatch key)`确定性构造；
  `ScopeInstanceId`由declared resource scope和上述IDs构造。它们都不序列化回manifest。
- persistent state create/attach/reset request必须携带typed caller-owned `StateNamespaceId`；它区分tenant、
  conversation/request lineage或其它调用方state domain，不从用户名、package path或provider handle推断，也不
  写回manifest。registry group key固定为`(StateNamespaceId, ModelInterfaceSemanticId,
  StateConsistencyGroupId, PersistentStateScopeKey)`；group record内部再以tagged model `ResourceId`索引
  canonical member map。attach必须先验证完整member set，再逐项验证state type/capacity/page geometry/storage/
  alias-update和selected group realization兼容，不能分别attach出跨version的member组合。`PackageManifestId`只作
  本次session provenance，不能作为registry namespace。
- `PersistentStateScopeKeyV1`是durable closed union：`model_shared`无payload，`component`携带
  `(DistributedProgramSemanticId, ComponentId)`，`execution_instance`携带完整`ExecutionInstanceId`。它从manifest
  declared group scope和selected distributed mapping确定，不含`InvocationId`、`EntryInstanceId`、graph iteration、
  provider handle或transient `ScopeInstanceId`。persistent state声明invocation/entry/iteration scope非法；需要其它
  durable scope必须提升codec version。`ScopeInstanceId`仍只用于当前session resource instance，不写registry key。
- `StateNamespaceIdV1`的durable representation固定为version `uint16=1`加32 caller-owned opaque bytes；all-zero、
  unknown version和错误长度非法。它不是用户名/字符串/semantic digest，runtime只做exact byte identity，registry
  canonical key codec使用big-endian fixed widths和tagged ResourceId/PersistentStateScopeKey fields。
- `PersistentStateRegistry`为每个group key持久化current `StateGroupVersionId`、canonical
  `ResourceId -> (ResourceVersionId, DurableStateBackingRef)` map、`valid | poisoned | requires_reset` status和
  fencing epoch。`DurableStateBackingRef`是registry/backing-service版本化typed reference，记录足以重新import并
  验证的backend owner、object/version、size/page geometry和integrity/fencing facts；绝不持久化进程内
  `AllocationHandle`、page-table pointer、physical address或provider-private transient ID。
  `StateGroupVersionIdV1`与`ResourceVersionIdV1`都是不同strong type的
  `(uint64 registry_epoch, uint64 monotonic_counter)` fixed big-endian durable value；两字段非零，各自在group/member
  owner下原子递增且永不复用。restart从持久值继续，counter耗尽hard fail且不wrap。
- `beginAtomicUpdateGroup`以expected base `StateGroupVersionId`做线性化检查，原子捕获完整base member map并为每个
  member创建未发布`ResourceVersionId`。`full_copy`在任何entry launch前分配并复制全部member capacity；`page_cow`
  只允许manifest已声明paged geometry的member，clone page table、以refcount共享只读base pages，并在launch前对
  compiler-declared bounded update footprint涉及的pages完成eager allocation/copy。不能依赖provider page fault或
  runtime猜测write set。candidate backing refs/refcounts和owner transaction写入durable recovery journal；全部member
  准备成功后才完成`state_snapshot_ready`，所有candidate slot都由该node支配。
- current readers持有base group snapshot lease直到其terminal node；candidate readers/writers按
  `StateSlotVersionRole`看到完整logical snapshot。group publish以base `StateGroupVersionId`做一次CAS，原子替换整个
  durable member map并产生新group version；stale loser不切换任何member，释放其private pages/backing refs/
  refcounts。crash recovery保留旧current并按journal回收未发布atomic candidates；partial member publish不存在。
- `in_place_poison_on_failure`在issue同组任一mutation前取得registry-owned exclusive group-epoch lease；lease覆盖
  全部member write、terminal completion和publish/poison。在first write issue前必须durably记录`dirty + fencing`
  marker，成功terminal/publish后才原子清除；process crash、lease/fencing loss或recovery发现dirty一律把整组转为
  `poisoned`或显式`requires_reset`，不能假设write未发生。同group的attach/update/reset与poison在线性化点串行；
  任一已issue write的failure poison整个group，reset也只能原子建立新的完整group snapshot。poison/version/fencing
  token持久化，新RuntimeSession不能绕过，stale process/session不能publish或清poison，从而避免ABA。
- attach通过`PersistentStateBackingStore`用current verified provider/device/generation重新import每个
  `DurableStateBackingRef`，返回新的typed `RealizedAllocation`/page-table handle和attestation，再与manifest、version、
  size/geometry/integrity逐项验证。同一group/resource version可在不同session获得不同transient handles；handle不是
  version identity。backing丢失、generation不兼容、attestation/integrity失败或page refcount损坏时不返回partial group，
  registry持久转为`requires_reset`（若可能发生in-place partial write则poison）。
- resident executable/session-shared immutable resources可在registry/cache中规范化到shared instance；weight cache key
  必须同时覆盖content digest、storage/shard/packing descriptor、`TargetEnvironmentFingerprint`、`DdrArenaId`、
  placement `ScopeInstanceId`、verified provider identity/device和provider epoch/generation，不能跨不兼容device/arena
  或provider restart复用allocation handle。
  workspace等invocation/rank/stage scope始终由ScopeInstanceId隔离。resident cache acquisition只有在selected
  `BoundBlobSource`产生的同一`OpenedBlobLease`完成流式digest、完整provider copy和readback/size gate后才原子
  publish lease；失败或并发loser不得暴露partial handle。streamed realization不进入full-weight cache：session只
  保留bound source和bounded staging allocations，按manifest window从同一opened lease校验required chunk后调用
  asynchronous copy；window failure按DAG取消consumer并回收staging，不得退回同步全量初始化。

### 5.1.1 Cross-Model State Migration

由于V1 `ResourceId`由完整`ModelInterfaceSemanticId` owner隔离，old model state绝不能直接attach到new model。
`schema/wafer/state_migration.proto`和WCRE record 21定义immutable `StateMigrationPlanV1`：typed old/new
ModelInterfaceSemanticId和canonical migration components。每个component显式携带nonempty old/new
StateConsistencyGroupId sets、每组complete ResourceId members、每个old/new member完整canonical typed state descriptor及
exact slot/shape/shard/storage relation（semantic type/shape bounds/access/alias-update/capacity/page geometry/storage/scope）、
migration mode、bounded canonical nonempty old/new `StateMigrationScopeSelectorV1` sets、必要target artifact refs和
`StateMigrationPlanId`。selector是ABI层closed union：`model_shared`、携带shared typed
`(DistributedProgramSemanticId, ComponentId)`的`component`，或携带shared typed `ExecutionInstanceId`的
`execution_instance`；优先复用semantic-identity schema的scope fact，不编码runtime durable key、namespace或transient
`ScopeInstanceId`。plan要求所有声明参与的old/new groups/members/selectors恰好属于
一个component且new model要求迁移的state coverage完整；mapping按IDs/typed roles给出，禁止name/ordinal similarity、partial/
overlapping member map、隐式split/merge或caller callback。

selector sets本身不隐含按排序位置配对。`full_copy`/`page_cow_snapshot` component必须保存canonical
`StateMigrationScopePairV1` records，逐项显式引用一个old和一个new selector并all-and-only覆盖两侧；trusted transform则让每个
KAD input/output slot显式引用对应selector/component role，形成bounded N:M relation。缺pair、重复selector、仅靠ordinal zip或
无法从slot relation重建完整N:M coverage都失败。

runtime semantic verifier先由唯一factory构造move-only、non-aggregate、nonserialized
`StateMigrationPlacementRequest`。request显式move-own old/new metadata、verified plan、完整migration artifact metadata bundle、
唯一`StateNamespaceId`，以及old/new端点各自的typed `(TargetVariantId, ExecutableVariantId, ProjectionSetId)`选择和validated
`StateMigrationDomainSelectionPolicy`；transform components另保存plan-owned artifact target选择。domain policy只能给出canonical
duplicate-free prioritized-allowed constraint vector、topology locality和是否允许verified cross-provider transfer等请求约束；
每个constraint都是由authenticated deployment inventory bootstrap所产catalog签发的owner-backed
`TrustedRuntimeDomainConstraintRef`，vector包含全部allowed constraints恰好一次且顺序就是唯一priority，不能再带第二份allowed set/
priority map。authenticated deployment bootstrap一次返回move-only `TrustedDeploymentInventory`和immutable/copyable
`TrustedRuntimeDomainConstraintCatalog`，二者共享owner/generation；inventory随后move入`VerifiedRuntimeServiceBootstrap`，trusted host
仍可用catalog按typed `RuntimeDomainConstraintId`解析refs。catalog不暴露raw provider/device ID，provider adapter不能签发它；
placement必须把ref/catalog generation与registry environment inventory重验。policy不能声称provider、backing、
capacity或兼容性事实。request还一次性move-own validated `RuntimeAdmissionPolicy`、`StateMigrationAdmissionLimits`、
`RuntimeCapacityReservationMode`和`InvocationControl`；catalog query前后、placement/bind/snapshot/execution只能重验同一control和policy，
不能在后续替换。factory把plan selectors逐项join到old/new manifests的declared scope、distributed mapping和selected
realization，生成nonserialized、owner-bound `VerifiedStateMigrationScopeRelation`，其中才包含resolved old/new
`PersistentStateScopeKeyV1` sets。所有resolved keys位于同一namespace，namespace不能映射或重命名；unknown/duplicate/
noncanonical selector、manifest无法唯一解析、typed端点不属于对应manifest、selector与group scope不符或policy本身不合法均在
deployment registry访问前失败；policy中的typed allowed-domain constraint是否存在且兼容留给后续两份verified inventory join。
caller不能传durable key、scope map、raw provider/device ID或隐式要求runtime替它选择old/new
target/projection。

V1 mode是closed union：

- `full_copy`：component必须同时满足old/new group、member、explicit selector-pair及resolved scope-key bijection，logical state value/storage/page semantics
  exact compatible；为每个new scoped member创建独立backing并复制；
- `page_cow_snapshot`：同样要求group/member/scope三者bijection；除上述兼容外要求verified backing store支持fenced COW/refcount，new snapshot初始共享只读pages，
  任一old/new后续write各自COW，绝不让两个mutable groups写同一page；
- `trusted_transform`：引用完整verified migration TargetArtifactSet/entry/KAD和ordinary old-current read-only/new-candidate
  slots；允许component把N个old groups/members/scopes完整变换为M个new groups/members/scopes，用于TP/EP degree变化、KV reshard或显式
  state format升级。KAD/plan逐项描述all input/output slot、partition/concat/split/index relation、type/shape/storage、scratch/
  completion和numeric/integrity verification；每个new byte/value必须有verified producer且所有old input只读。它不能写old
  backing、遗漏/重复new output或调用host callback。

trusted-transform artifact输入不是单个可选set。`VerifiedMigrationArtifactBundle`是bounded canonical、move-only、
non-aggregate all-required/no-extra set，完整覆盖plan引用的每个`(TargetVariantId, TargetArtifactSetId)`；每个component再用typed
`TargetArtifactMemberKey`/EntryId/KAD slot引用bundle member。不同components可以使用heterogeneous target sets/domains，但所有sets
必须从同一个`VerifiedProgramDelivery`派生的trusted refs，在同一个`ArtifactMetadataVerificationSession`中完成metadata load，
随后通过同一个exact-domain context/runtime artifact session一次all-or-none bind。逐set独立bind、跨delivery拼接、missing/extra
set/member或把content相同member换到另一set均失败。

production migration-plan metadata loader唯一形态是
`loadAndVerifyStateMigrationPlan(const TrustedStateMigrationPlanRef &, const StateMigrationAdmissionLimits &,
ArtifactMetadataVerificationSession &)`。publisher/compiler构造delivery bytes时仍显式消费自己的
`CanonicalEncodingContext`，但runtime loader不能另传/新建context、artifact limits或source reader；它通过session private
access取得同owner read/work leases并复用canonical context，结合migration limits取逐维更严格预算。返回proof绑定metadata
owner generation；standalone inspection使用不互转的diagnostic ref/result。raw path、default/unbounded context或把
inspection proof转成runtime migration input都不存在。

migration必须先打破exact-domain bootstrap环，但不能假定old durable backing天然portable。process/deployment-owned
`WaferRuntimeServiceRegistry`以其已经验证的durable-state service relation执行
`snapshotStatePlacementInventory(StateMigrationPlacementRequest)`，move-consume request并只通过private access读取其中已经sealed的
完整old resolved-key query，不解析plan/manifests、重做selector join或生成scope relation。它做一次只读catalog查询并签发
move-only、owner-backed `VerifiedStatePlacementInventory`；inventory继续唯一拥有原request和scope relation，不能从同一request重复
签发或在后续重新配对。inventory all-and-only覆盖scope relation中的old current keys，保存state-catalog
owner/epoch、每项opaque placement-record identity、provider owner/generation、current capacity domains、storage/backing class及verified
same-provider reuse、cross-domain reimport/copy和cross-provider transfer约束。它没有`DurableStateBackingRef`、path/FD/provider handle，
不能open/read/import backing、取得execution/capacity lease、签发最终registry snapshot或调用provider；extra/missing key、跨namespace/
request owner或catalog/provider generation不闭合均失败。

随后纯
`preflightStateMigrationPlacement(VerifiedStatePlacementInventory,
VerifiedRuntimeEnvironmentInventory) -> StateMigrationPlacementPlan`只做typed joins和checked arithmetic。它验证old
source domains确由placement inventory证明，按new/transform selected target/projection和domain policy从environment inventory确定唯一
canonical domain assignment，并验证每条same-provider/cross-domain/cross-provider transfer都由两份inventory共同允许；无法唯一选择、
nonportable/non-reimportable backing或policy冲突结构化失败。plan move-own placement inventory（其中保留request/scope relation）、
environment inventory、all-required
set/member keys、old/new/transform exact capacity-domain set和保守资源上界。它不取得reservation authority、exact-domain context、
execution-registry snapshot，不能bind runtime source、open backing/blob或调用provider。只有此move-only plan完成后，
`WaferRuntimeServiceRegistry`才能按exact domains取得context/runtime capacity authority，并用同context的artifact session all-or-none
形成独立`RuntimeBoundMigrationArtifactBundle`；full-copy/COW要求显式empty bound bundle，trusted-transform要求exact all-required bundle。
metadata bundle、placement plan、placement inventory、context和bound bundle不能跨owner重新配对。

取得exact-domain context后，先以one-way
`bindStateMigrationPlacementForRuntime(StateMigrationPlacementPlan, WaferRuntimeServiceContext) ->
RuntimeBoundStateMigrationPlacement`绑定plan/context owner；再以同一bound placement完成migration artifact bundle bind。其repo-owned
final `WaferPersistentStateRegistry`才是execution registry precondition的唯一owner。它以一次只读一致事务签发move-only、
non-aggregate `VerifiedStateMigrationRegistrySnapshot`，绑定exact StateNamespaceId/PlanId、verified scope relation的全部
component old/new resolved scope-key sets及
group keys、完整old group/member/version/status maps、每个new key的absent或expected-reset predicate、registry epoch和fencing
generation；`issueStateMigrationRegistrySnapshot(RuntimeBoundStateMigrationPlacement)` move-consume bound placement并让snapshot
唯一保留其plan/context owner，保证同一placement不能重复签snapshot。snapshot同时重验placement inventory的state-catalog
owner/epoch、每项placement-record identity、provider generation、current
backing placement以及计划依赖的reuse/reimport/transfer能力。任一事实变化返回stale且发生在backing/blob open、capacity/execution lease和
provider side effect之前。caller不能提交`expectedOldVersions` map、布尔`newAbsent`或构造snapshot；snapshot签发不取得execution lease、
不open backing且不写journal。任何registry mutation/epoch/fence变化使其stale。snapshot只可用于对应plan/manifests的一次
preflight/execution，不能裁剪groups或跨namespace复用。

state migration另接收validated nonidentity `StateMigrationAdmissionLimits`，至少限制component/artifact-set/artifact-member/
old-new scope-selector/resolved-scope-key/group/member/page/chunk counts、single/
total logical state bytes、single/total bytes read/copied/transformed/written、COW metadata/refcount operations、transform scratch/
output bytes、simultaneous source/destination leases、open FDs、buffers/buffer bytes、workers/provider commands/events和wall/CPU
deadline。所有字段positive checked且0不表示unbounded；limits不能删member、拆plan、改变mode/chunking或把atomic migration
降级为逐组。它与`RuntimeAdmissionPolicy`、verified environment/provider hard limits、host ledger和capacity reservation重叠的
operational维度逐字段取最严格值，不存在优先覆盖或两套独立预算。full-copy/COW的module demand恒为0；trusted-transform
模块严格复用`ModuleVerificationMode`和`ModuleResidencyMode`：`eager_active_set`在对应验证/驻留轴覆盖全部selected transform
active set，`at_first_use`只可在首条依赖command之前验证，`graph_liveness`只按verified transform DAG terminal释放，任何组合都
不能改变transform组件、slot relation、copy/command order或batch-CAS语义。第二阶段
`preflightStateMigrationExecution`显式move-consume保留bound placement的上述verified snapshot和同owner runtime-bound artifact bundle，
用checked arithmetic形成绑定context/snapshot owner token的完整`StateMigrationSessionPlan`；
在此阶段仍不取得state execution lease、不open backing、不allocate/copy/transform或写journal。100GB-class
full-copy必须用bounded same-handle chunks流式read/copy/hash/attest，不能把完整old/new state放入host memory；COW/transform也
受相同metadata/scratch/concurrency预算。不同充分limits/worker数产生相同new versions/content relation。

唯一host request是上述`StateMigrationPlacementRequest`；它没有caller context/backend/inventory proof/snapshot、单一
`PersistentStateScopeKey`字段或scope map。orchestrator按request semantic join -> placement inventory -> pure placement plan -> exact
context -> runtime artifact bind -> execution registry snapshot -> pure session plan的固定顺序工作；不能跳步或替换owner。
`preflightStateMigrationExecution`显式move-consume保留bound placement的`VerifiedStateMigrationRegistrySnapshot`和独立
`RuntimeBoundMigrationArtifactBundle`，不能把bundle或snapshot藏进caller可替换的backend/context wrapper。semantic verifier把plan内每个typed descriptor逐字段join到
old/new manifests，不接受generic descriptor fingerprint。execution把snapshot/session-plan token交回同一个registry，在任何
backing/provider副作用前按canonical key一次性取得全部component/
scope的old-current complete-group
fenced read leases并重验所有versions、new preconditions、registry/provider generation；任一失败释放全部leases且副作用为0。随后为plan内
所有new scope/group/member sets创建一个migration-transaction-scoped unpublished candidate/journal。copy/COW/transform和
attestation全部成功后，registry执行一次batch compare-and-swap：原子比较全部old leased
versions、全部new-key preconditions和provider fencing generation，并一次发布全部new `StateGroupVersionId/member map`。
V1不允许逐组CAS或“已发布组补偿回滚”。journal/recovery、reservation和cleanup都以整份migration transaction为单位；
failure/cancel/crash保留全部old current、回收全部new candidates且零partial new group/member publish。publish new不自动
删除old，retirement另需caller显式操作。stale old version/new existing group/provider generation mismatch都在mutation前失败。
`executeStateMigration(StateMigrationSessionPlan) -> StateMigrationResult`返回private-construction、non-aggregate closed result：success
arm只保存由同一batch-CAS receipt产生的canonical duplicate-free `PublishedStateGroupVersion {StateGroupKey,
StateGroupVersionId}` records；failed/cancelled/deadline arms保存独立runtime-only `StateMigrationFailure`和
`BoundedStateMigrationFailureReport`且没有publish summary，不能复用带entry/iteration语义的`InvocationFailure`。failure使用closed
kind/phase、plan ID、optional typed old/new `StateGroupKey`/`PersistentStateScopeKey`、artifact member和capacity domain定位事实；不使用
component ordinal/name或自由字符串做programmatic分支，bounded diagnostic只作解释。key和version不能作为parallel arrays靠ordinal
配对，old state preserved也不是caller可填布尔字段，而是每个result arm都必须满足的execution invariant。

没有verified migration plan时V1行为是structured `state_model_mismatch`并要求fresh create/reset；普通attach、alias、
StateNamespaceId复用或diagnostic alias不能触发迁移。migration plan/build/runtime实现是独立implementation task，但本文
固定其长期ABI、atomicity和默认fail-closed边界。

provider allocation/import返回typed `RealizedAllocation`，至少包含opaque move-only handle、provider device、
actual base、capacity、alignment、arena和scope；module path返回绑定`VerifiedLoadableModule`、content digest和
provider handle的package-scoped typed `ModuleUseLease`。RuntimeSession只通过这些返回字段执行address-width、alignment、range和KAD address construction，
不窥视opaque handle，也不从另一个query side table补事实。

Dry-run low-level provider配合repo-owned executor打印 RuntimeSession 的 variant/resource/placement/entry/completion projection，作为
无卡 contract test。`fake-tx` 是 no-card test provider capability implementation，消费同一`VerifiedGraphInstancePlan`和rolling
`VerifiedGraphInstanceCursor`，按issue/completion顺序流式记录bounded trace；它不得构造完整展开图，也不单独重建
resource slots或completion order，更不能直接签发verified proof。该 Python 工具只用于 package/no-card 调试，不作为真实 host runtime implementation。

### 5.2 C++ Host Runtime Boundary

真实 host runtime 主路径落在 C/C++，不是 Python 脚本。C++ runtime 的稳定职责是：

- parse exact Protobuf delivery bytes、验证blob digest并立即调用唯一 semantic verifier；
  schema/version/target/artifact/resource /
  variant/graph/completion任一验证失败都不能创建 RuntimeSession。
- 先以runtime target/topology筛选唯一compatible target/projection候选，再对actual dimensions求值一次
  shape guard并选择一个`ExecutableVariantId`；随后使用其committed rank mapping，所有rank/stage一致。
- materialize typed resources、placement、entry graph和 completion DAG，并逐 module验证 ELF descriptor
  hash、environment/artifact fingerprints和module digest。
- 动态发现 HPGR / `tx_runtime` provider。默认 build不硬链接板端库；no-card环境可以做 capability /
  symbol discovery，但必须明确停止在未执行状态，不能把 symbol存在解释成 launch/completion。
- 按 entry/completion node需要验证 provider capability；缺少 event、timeout、status/error或 executor API
  时结构化失败，不使用无关 sync API替代。

`VerifiedRuntimeEnvironmentInventory`/`VerifiedRuntimeEnvironmentSnapshot`不能由caller填fingerprint构造。
`WaferRuntimeServiceRegistry`先通过其move-owned完整provider capability set的side-effect-free environment query取得canonical
provider/device/capacity-domain identities、各自ABI/mode/arena capability/epoch-generation、topology/endpoints和
rank/projection-to-domain mapping，再与registry持有的`TrustedDeploymentInventory`逐项比对并重算environment/topology digests；
只有registry factory可构造verified inventory，并只在`RuntimeSessionPlan`选出exact domains后派生对应snapshot/context。该query不allocate/load/submit。RuntimeSession在第一次provider side effect前
再次检查所有domain identity/generation未变化；forged caller facts、stale generation、inventory/provider endpoint、
arena或runtime mode不一致都在allocation前失败，不触发fallback replanning。no-card recording backend提供同一
typed low-level query接口，不能绕过registry直接注入expected digest。

Board gate实现真实 set-device、state/weight/workspace allocation/binding、H2D/D2H、module load/function
lookup、rank/stage graph launch、completion DAG和 error propagation。Python工具只打印由 C++ typed object /
validator产生的 debug projection，不再承载 provider抽象或独立 package semantics。

module residency由provider/device-scoped `ModuleResidencyManager`拥有，而不是每个entry临时load/unload。cache key至少
是`(final module ContentDigest, TargetArtifactFingerprint, TargetEnvironmentFingerprint, verified provider identity,
provider device identity, provider epoch/generation)`；
intrinsic `ProviderCodeCache` value只能以move-only `ProviderCodeLease`暴露，持有provider `ModuleHandle`以及从同一
ELF bytes验证的machine/ISA/MABI、exports、KAD note set、fingerprints和content digest；它不保存任一package的
TargetArtifactMemberKey、EntryId/rank-class/completion coverage或“已允许本次调用”的proof。

每次current package使用module时，无论cache hit/miss，都必须把其locator-free verified target-set member、EntryId/
static digest/KAD/class/completion relation与`ProviderCodeLease` intrinsic facts重新join，返回新的package/session-scoped
`ModuleUseLease`；随后typed function resolution再产生绑定exact EntryId/KAD/slot schema/provider generation的
`ExecutableHandle`/`VerifiedEntrySubmission`。wrong package member不能因module content相同或cache hit跳过关系验证，
package lease也不能写回global cache。entry graph从每个
module的first-use node到所有using commands的last terminal completion导出liveness；module在last command completion
前不能unload，不能把submit-return当成last use。

`RuntimeAdmissionPolicy`把两个正交轴分别固定为
`ModuleVerificationMode = eager_active_set | at_first_use`和
`ModuleResidencyMode = eager_active_set | graph_liveness`。verification mode决定actual bytes/ELF/KAD验证时机，residency mode
决定provider load/resolve与release时机；任何组合都必须先验证再load，不能用一个enum隐式耦合。`eager_active_set`在session
create时只处理unconditional selected modules；对dynamic activation wave，必须等trusted predicate=true后再在该wave首个
dependent side effect前验证或load完整active set。predicate=false member的blob open、verify、load、copy、submit始终为0。
`at_first_use`/`graph_liveness`分别按manifest DAG的deterministic first-use验证/load，并按typed symbolic
last-completion rule释放；两轴都不得为满足provider limit串行化或重排entry。无predicate时`eager_active_set`等价于处理全部
selected modules。preflight从verified graph template、count bounds、activation envelopes、periodic delta和release frontier证明exact peak，或使用经证明
不小于actual的conservative peak；不得构造actual expanded graph，无法在admission limits内证明峰值就拒绝。证明出的
simultaneously-live peak与provider max-loaded-modules和manager当前可原子reservation容量比较；peak超限或reservation
失败在任何load前结束。shared cache只复用exact key，concurrent
single-flight load loser取得同一lease；zero-lease entry才可按manager的typed bounded policy eviction。
每次lease use/submit前验证snapshot generation仍一致；provider restart/reload即使device/environment fingerprint相同也
必须cache miss并evict或隔离旧transient handles，不能把generation只作diagnostic。

`wafer-run`若暴露CLI policy，只能使用两个独立closed controls
`--module-verification=eager_active_set|at_first_use`和
`--module-residency=eager_active_set|graph_liveness`，解析后进入validated `RuntimeAdmissionPolicy`并随plan封闭；旧combined
`--module-admission`、自由字符串、运行中切换或provider默认覆盖都不存在。

provider adapter的module/entry边界必须typed且non-forgeable。runtime-private access用selected `BoundBlobSource`与同owner
read/work leases形成同一`OpenedBlobLease`/backing；`WaferArtifact::verifyTargetElf`在该handle上聚合复核content digest、ELF/KAD note、target/artifact
fingerprints和manifest member coverage，private join factory生成`VerifiedLoadableModule`及其
`VerifiedLoadableEntry` views。它不要求RuntimeSession凭空构造compiler-side `VerifiedTargetArtifactMember`。
`loadModule`只接收`VerifiedLoadableModule`并在成功后publish `ModuleLease`；load/verify failure不发布handle。
`resolveExecutable(ModuleLease, VerifiedLoadableEntry)`在adapter内部使用proof-owned entry symbol查找provider function，
逐项复核EntryId、KAD function type/digest和RankClass/completion coverage，返回move-only `ExecutableHandle`。
`submitEntry`只接收该handle和按KAD SlotId构造的arguments，不接受symbol string或任意ModuleHandle。provider-specific
string lookup不能越过adapter或成为cache identity；missing/wrong symbol/KAD/member在任何submit前失败。

### 5.3 Current v2 Tools and Migration Boundary

当前 `tools/wafer_package_metadata.py`、`tools/wafer_export_package_metadata.py` 和
`tools/wafer_runtime_adapter.py` 只保留为 schema v2审计、negative fixture和迁移输入。v2的 flat model、
path-only module、free-form lifecycle、`binding_order`、instruction list和 scalar completion都不是长期
`PackageManifest`合同。特别禁止：

- 从 LLVM文本、`i64`参数数量、参数名、module文件名或 instruction文本窗口恢复 ABI/resource role。
- 把 package中的 `instructions`数组当 runtime schedule或 target support证明。
- 让 Python validator和 C++ loader分别决定 accepted enum、默认值和跨字段 legality。
- 在缺descriptor semantic digest、environment/artifact fingerprints或module/set-root digest时用
  path/symbol discovery补齐identity。

`tools/wafer_device_link.py`仍是device-code local gate，但长期输出必须进入`tasks/14`定义的完整、原子发布
`TargetArtifactSet`；它不生成manifest，也不代表board completion。Package assembly只消费committed
`wafer.executable`和对应complete `TargetArtifactSet`，不另接verified-program metadata/resource-view/
accepted-transport旁路，也不解析LLVM或instruction text。

No-card adapter只消费已经由 C++ semantic verifier接受的 typed manifest，验证 variant/resource /
placement/entry/completion materialization和 provider capability。真实执行仍由 C++ runtime承担；dry-run或
fake provider输出不能成为新的语义源。

Endpoint projection是committed executable facts的交付投影，不是第二份topology。Pinned section记录
topology/mesh/availability/projection fingerprint和per-rank endpoint；relocatable section严格记录
`ConcreteRecordSet`或`FiniteTemplateSet.allowed_bindings`及endpoint/channel/FSM/control slots。semantic
verifier要求其与committed transport、rank mapping、`TargetEnvironmentFingerprint`及对应
`TargetArtifactFingerprint`一致；名称只用于诊断，不作为绑定协议。

## 6. Legacy Bootparam and Dyn TLV

如果选择 legacy fallback，package 需要能序列化 legacy bootparam / dyn TLV：

- `D_BootParamHead` size 56。
- `D_BootParamDyninfo` size 72。
- dyninfo 从 `head + 0x38` 开始，顺序是 inputs、outputs、params。
- dyn TLV header 是 `{ uint32_t type; uint32_t len; }`。
- 已知 dyn TLV type 包括 final、cfg PMU、kcore cfg、export SPM、disable calc、profiling config、
  dynlib load/run/unload、memcpy D2D、P2P send/recv、group data dump。

这些结构只属于 legacy runtime delivery。它们不改变 compiler IR contract，也不能被上游
group/layout/SPM 文档当成语义对象。

## 7. Completion and Status

每个 runtime path 必须声明 typed completion DAG。DAG node至少覆盖需要的：

- host command submission / command-object completion。
- stream/event wait、H2D/D2H copy completion。
- device local drain；它只证明本地 NCC可见性，不能单独完成 invocation。
- DTE/FSM、collective和 stage barrier completion。
- legacy synchronous model completion，仅限显式 legacy entry node。
- status/error observe、timeout和 cleanup/release node。

Node记录 stable kind、rank/stage/request scope、timeout policy、status source和 failure domain；edge表达必须先
完成的关系。RuntimeSession将 node映射到 provider command/event/stream或 device status，但 provider API名
不进入 manifest主合同。Global invocation completion必须可追溯到所有输出相关 rank/stage的 host command、
device/DTE completion和 copy visibility；不能只选一个 scalar字符串。

executor必须event-driven：每轮先耗尽所有ready receiver-ready、copy issue、module/entry submit等nonblocking
issue node，再poll/await provider events和device/status completion；blocking wait不能因stable node排序抢在尚未
issue的peer/rank/stage之前。collective/DTE verifier还要证明每个participant的issue frontier都能在任何participant
wait变为阻塞依赖前到达。node timeout、caller deadline/cancellation或failure触发同failure domain中未issue node抑制、已issue command的provider
cancel或安全terminal wait，再按DAG执行cleanup；原始失败优先于cleanup failure。

streamed immutable copy也是typed DAG语义：`resource_copy_issue`只消费compiler-declared
`StreamWindowId/source range/staging range`，provider copy completion支配所有consumer entry；最后一个consumer
completion支配同一staging range的下一次reuse。RuntimeSession不能把resident初始化API用于window copy，不能在
copy失败后提交consumer，也不能因provider stream数变化重排manifest dependency。

Persistent state的publish/poison也是typed DAG终点：

- `atomic_version`：`state_snapshot_ready`支配同组全部candidate slot consumer；writer nodes只更新未发布的member
  versions，唯一`state_group_publish`必须被全部相关rank/stage writer completion和status-success支配。任一
  failure/timeout/cancellation不执行group CAS，旧group snapshot保持有效，cleanup回收整个未发布candidate而非部分member。
- `in_place_poison_on_failure`：任一可能已经issue同组任一member write的path发生failure/timeout/cancellation，必须先持久完成
  `state_group_poison`再结束失败cleanup；后续attach/read/update整组拒绝，直到显式`reset`建立新的完整valid snapshot。

Runtime不能推断第三种repair/default policy，也不能让cleanup覆盖原始failure。

Schema v2 的 `runtime_stream_wait`、`runtime_command_completion`、`kcore_local_drain` 和
`legacy_model_sync` 只作为 converter输入。converter只有在能构造无歧义 DAG时才能迁移；尤其
`kcore_local_drain`必须与可信 host completion组合，不能单独转换为 terminal node。

不能作为 correctness fence：

- old `TsmDeviceSynchronize` stub。
- `TsmLaunch/TsmLaunchPg` stub success。
- KMD compute fence that only signals after MHU doorbell submission。

Runtime adapter 必须把 stub shielding 做成显式 validation。不能把 “API 返回 success” 当成模型已经
执行完成。任一 rank/stage timeout、transport error、device exception或 provider failure必须进入 DAG
error edge和聚合状态；cleanup不能覆盖原始失败。

## 8. Verifier and Tests

验证分层：

- Schema/codegen：Protobuf field number/enum/version compatibility、deterministic serialization、generated
  C++/Python type roundtrip和 JSON debug projection来自同一 typed object。v2 converter覆盖可迁移和必须拒绝
  的歧义 case；普通 runtime入口拒绝 v2 JSON。
- Semantic verifier：resource id唯一且 use-def闭合；actual shape/bounds、persistent state alias/page capacity、
  state-group member/axis/snapshot/slot-version coverage、immutable weight digest/shards、resident或streamed-window
  coverage、staging reuse和copy dominance、workspace size/scope和control/status slots合法；每个DDR resource的
  `DdrArenaId`存在且placement domain受target declaration允许，arena/scope instance mapping无歧义。
- Kernel ABI/artifact：每个 entry node的 slots与 `KernelAbiDescriptor`一一对应；
  `.note.wafer.abi` descriptor digest、target
  fingerprint和 module digest匹配；`EntryId`/function semantic digest到module/entry symbol/KAD/rank coverage
  映射闭合；每个`CompletionExportId`恰好绑定一个executable DAG node。缺失/重复 slot/export、错误scope
  instance、错误code digest/module或target mismatch均失败。
- Variant/entry graph：target/projection filter和shape guard各自priority/fallback确定，选择唯一tuple；
  `RankClassId`与committed mapping一致且覆盖launch ranks，stage/replica映射唯一；graph无非法cycle或悬空
  dependency，node不包含instruction shadow schedule。
- Endpoint projection：pinned mapping可从 accepted topology/mesh/transport重算且 fingerprint匹配；relocatable
  mapping的 endpoint/channel/FSM/receiver/control requirements完整；两种模式互斥。
- Completion DAG：所有 output相关 node可达 terminal success；host command、device drain、DTE/collective wait和
  copy visibility按 edge组合；local drain不能单独 terminal；timeout/error/peer failure有状态源、聚合和 cleanup。
- Bundle scalability：在低`RLIMIT_NOFILE`和数千unselected module/payload locators下load/selection不耗尽FD；
  symlink/rename/mutation race、opened-handle eviction/reacquire、early EOF和chunk digest failure均不能发布handle/lease。
- No-card RuntimeSession：覆盖 variant选择、state group create/attach/reset、resident weight handle复用、streamed
  weight bounded staging/copy/reuse、并发 invocation
  workspace隔离、entry argument construction、`DdrArenaId + ScopeInstanceId` allocation/base/range/address-width
  validation、provider capability rejection和明确 `not executed`状态；销毁并重建RuntimeSession后，
  `PersistentStateRegistry`仍保留current `StateGroupVersionId`、完整member version map或group poison状态；两个session
  基于同一group version并发atomic update时只有一个group CAS成功，stale loser不发布任何member，full-copy/page-COW
  cleanup和in-place group update/poison/reset race保持线性化。
- Event-driven executor：覆盖two-rank reciprocal DTE/collective、pipeline多stage先issue后wait、单rank timeout时
  其它pending/issued command的抑制、cancel或安全terminal wait，证明stable ordering不会制造host-side deadlock。
- Board：连续 prefill/decode或等价两次 invocation证明 persistent state group更新和读取；resident权重跨invocation
  复用，streamed权重按bounded window执行且无全量device allocation，
  multi-rank/stage graph按 DAG完成，并验证单 rank failure、timeout、transport error、atomic-version
  publish或 state poison、数值结果和资源释放。
- Legacy：bootparam/dyn TLV serialization只在 explicit legacy node中验证 size/offset/header；不能让 legacy
  scalar completion或 stub path满足 typed package完成门槛。

主线完成证明必须从真实 framework program重放 compiler、target LLVM、device link、manifest assembly、
C++ verifier和 RuntimeSession。手写 JSON、schema roundtrip、fake command trace、shared-library symbol存在或
host API success只能作为局部覆盖，不能替代真实 artifact identity和 board completion。
