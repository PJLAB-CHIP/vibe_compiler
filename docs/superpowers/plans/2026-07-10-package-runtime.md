# Package and Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 committed `wafer.executable` 和 complete `TargetArtifactSet` 生成唯一 typed Protobuf package，把它作为同一 `compiler::ProgramOutputTransaction` 的私有 root 附着并通过单一 program delivery commit 可见，再由单一 C++ semantic verifier 和非序列化 `RuntimeSession` 完成 no-card 选择、资源实例化、状态一致性与 completion DAG 执行。

**Architecture:** prerequisite `WaferProgramDeliveryFormat` 是package/target/MLIR无关的outer commit wire、trust与limited reader边界；compiler-side `compiler::ProgramOutputTransaction`绑定scope/sink/limits并拥有唯一visibility transition。`WaferPackageFormat` 是无MLIR的compiler/package/runtime共享manifest wire、semantic validation、WCRE manifest identity、exact blob integrity和bundle reader边界；`WaferPackageCompiler`单独拥有transaction-view assembly、bundle staging/attach和v2 converter。package root、executable root与全部target-set roots只在同一 transaction 内私有构造，最终由统一delivery-root rename或可信namespace中的单一 `ProgramDeliveryCommitRecord` 原子可见；production bundle loader只接受从verified delivery派生的deployment-owned `VerifiedPackageBundleReference`，不能由bundle自认证。`WaferRuntime`链接`WaferProgramDeliveryFormat`、`WaferPackageFormat`和runtime-safe `WaferArtifact`，先从immutable metadata与registry-owned environment inventory选择唯一target/projection/shape/rank tuple，再取得exact-domain context、one-way bind、重验selected actual ELF/KAD/fingerprints，并由repo-owned managers通过registry-owned provider capability set实例化arena/resource/state/entry/completion；Python generated binding仅输出debug projection，不拥有schema、legality、selection或launch语义。

**Tech Stack:** C++17、MLIR/LLVM 20、Wafer executable ODS、Protobuf 3.21.9 generated C++/Python bindings、`WaferProtoSupport`、`WaferABI` WCRE V1/domain-separated SHA-256、RISC-V ELF/KAD、LLVM `Expected`、GTest、lit、CTest。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  whole-variant atomic commit 产生的完整 committed `wafer.executable`，以及 tasks/14 对每个 referenced
  `TargetVariantId` 原子发布并重新验证的 complete `TargetArtifactSet`；不接受 candidate、单 module/KAD、
  手写 manifest、JSON 或其它 resource/entry/completion side view。
- Current stage responsibility:
  compiler-only package assembly 把 committed executable 与 complete artifact sets 无损投影成唯一 typed
  `PackageManifest`，在同一 `compiler::ProgramOutputTransaction` 内stage并attach capability-bound private bundle；outer driver
  把 executable、全部target sets和package作为一个completion scope原子提交；runtime-safe loader/semantic verifier
  从deployment-owned reference固定identity与inert source descriptors；pure preflight固定exact domains，one-way runtime bind才生成private sources；RuntimeSession 对一次 invocation 固定
  environment/target/projection/shape/rank/iteration/resource/state/
  transport/entry/completion 关系，重验 selected actual ELF，并按 event-driven DAG 实例化和执行。
- Output artifact / IR:
  deterministic Protobuf delivery bytes、exact `PackageBlobDigest`、WCRE `PackageManifestId`、transaction-private bundle
  root和最终单一 `ProgramDeliveryCommitRecord`/unified delivery root，以及进程内 non-serializable
  `RuntimeSessionPlan`/`RuntimeSession`、typed provider handles 和 invocation result。
- Downstream consumer:
  no-card dry-run/recording gate、动态 TX provider capability adapter、后续真实模型/板端 numeric、completion 与性能 gate。
- User-level driver / named pipeline:
  `wafer-opt --program-pipeline=stablehlo-to-executable` 按声明的full-program completion scope在同一output transaction
  中构造executable、target sets和package并执行一次outer commit；`wafer-run --program-delivery=<trusted-ref>
  --parse-limit=<kind=value>... --admission-limit=<kind=value>...
  --module-verification=eager_active_set|at_first_use
  --module-residency=eager_active_set|graph_liveness --backend=dry-run|fake-tx|tx`是standalone运行入口；kind来自closed
  C++ enums且unknown/duplicate/missing失败。embedded/daemon入口直接传入同样validated policy对象。
- Explicit non-goals:
  不重做 planning、variant/rank/route/window 选择，不从名字/路径/参数顺序恢复语义，不序列化 runtime handle、
  state version 或 provider facts，不把 no-card orchestration 当成 numeric/board completion 证明。
- Completion gate:
  真实 program chain 经过 committed executable、complete artifact set、private package staging、单一outer commit、
  deployment-owned bundle reference、package verifier、lazy bundle、trusted environment acquisition、actual
  ELF/KAD/fingerprint 重验、typed resource/state materialization和 event-driven
  completion；static 与 count>=2 的 PP no-card gates 实际执行，所有 parse/admission/selection/ELF/resource/
  state/capability 负例在规定的首个 side-effect 边界前 fail closed。
```

## Global Constraints

- 设计 owner：`tasks/01-architecture.md`、`tasks/04-topology-execution-mesh.md`、`tasks/12-ddr-memory-planning.md`、`tasks/14-target-llvm-golden-packet.md`、`tasks/15-launch-runtime-package.md`、`tasks/16-verification-plan.md`。
- Task 0只在target artifact shared Tasks 1-5完成后执行，并且只消费其`WaferProtoSupport`、`WaferABI`、
  shared semantic/content digest value、bounded canonical encoding context和host/deployment基础；它不消费
  `WaferTargetArtifactSetProto`、target-set loader、committed executable或任何post-commit artifact proof。
- 除Task 0外，本计划其余package/runtime任务的前置接口由target artifact post-commit Tasks 6-14提供：
  `WaferTargetArtifactSetProto`、non-interchangeable `PackageManifestId`/`ModelInterfaceSemanticId`/
  `TopologySnapshotId`/artifact/KAD/target ID wrappers、`wafer::abi::ContentDigest`、owner-backed
  `abi::ImmutableByteBackingRef`/`abi::VerifiedTargetArtifactModuleView`、validated
  `abi::ArtifactAdmissionLimits`、内部canonical identity core、`wafer::abi::computeContentDigest`、
  `wafer::artifact::TrustedTargetArtifactSetDeliveryRef`、runtime-neutral
  `artifact::ArtifactMetadataVerificationRegistry`/`artifact::ArtifactMetadataVerificationSession`、
  `artifact::loadAndVerifyTargetArtifactSetMetadata(...)`、runtime-neutral non-forgeable
  `artifact::ArtifactVerificationBudgetCapability`、`artifact::RuntimeArtifactVerificationSession`唯一
  three-move-owned-input factory、runtime-private all-or-none `bindTargetArtifactSetsForRuntime`和compiler-only
  executable relation verifier。Standalone metadata只返回owner-backed record、KAD facts和不可open的
  `UnboundTargetModuleSourceDescriptor`；exact-domain bind后才产生`BoundTargetModuleSource`，其same-handle lease
  封成ABI backing后调用shared streaming verifier。package不含/不伪造target delivery root，而从embedded
  locator-free record取得exact module view并在runtime bind后从bundle root的`BoundBlobSource`取得bytes。
  `artifact::buildLoadedTargetElfContract(moduleView)`不接delivery/member array/context，
  `artifact::verifyTargetElf(backing, contract, limits, encoding)`绑定exact backing。
  `schema/wafer/target_artifact_set.proto`分别拥有locator-free `TargetArtifactSetVerifiedRecord`与delivery-only
  `TargetArtifactSetDeliveryRoot`；PackageManifest只能复用前者，bundle-local locator只属于`PackageBundleIndex`。
  trusted target delivery ref只来自verified ProgramDelivery standalone target root，不能由package index/module source
  或待验证root自声明。本计划不重复实现、重声明或重新解释这些对象，也不在public API暴露generic
  `SemanticDigest`转换；`WaferArtifact`不得include/link `WaferRuntime`。
- `WaferABI`是compiler/package/runtime共享、可序列化structural ID C++ value type和generated-message conversion的唯一owner。Package必须复用其tagged `abi::ResourceId` union、`ModelEntrypointId = (ModelInterfaceSemanticId, nonzero api ordinal)`及distributed/target/executable scoped IDs，不得声明同名raw bytes/string/integer wrapper。Runtime-only `InvocationId`/`EntryInstanceId`/transient `ScopeInstanceId`/durable `PersistentStateScopeKey`/state version/namespace strong types固定由`include/Wafer/Runtime/RuntimeIdentity.h`拥有且没有Protobuf conversion。
- Protobuf 版本固定为 3.21.9；package schema必须消费共享codegen target并验证版本一致，不新增第二份dependency pin、protoc发现路径或Python schema。
- `schema/wafer/package_manifest.proto`是package field number和shared WCRE field-option的唯一声明；实现/测试不得另存一张手写编号表或record-specific排序表。
- `PackageBlobDigest`固定为`SHA-256(exact delivered Protobuf bytes)`；`PackageManifestId`固定为WCRE record type 10和`wafer.package-manifest.v1` domain。deterministic Protobuf bytes用于同一pinned producer的reproducible delivery，不能充当semantic identity。
- 所有record-10和embedded record-9 canonical encoding必须共享一个move-only `abi::CanonicalEncodingContext`，由validated `CanonicalEncodingLimits`和unique `CanonicalScratchStore`通过唯一factory构造。没有default/unbounded context、隐式内存scratch或另传limits的top-level identity/verifier overload；scratch backing只经`CanonicalRecordBackingRef`共享，不复制完整canonical bytes。
- runtime顺序固定为：检查expected blob digest、parse、递归拒绝unknown fields、single C++ semantic verify、重算manifest ID；禁止通过重新序列化比较identity。
- package assembly唯一语义输入是同一outer transaction派生的`PackageArtifactBuildSession`：它封装committed `wafer.executable`、覆盖其全部target variants的complete `TargetArtifactSet`集合和capability-bound exact-byte sources。不存在public/custom artifact reader；bound source只能交付已提交content digest对应的same-handle bytes，不能提供resource、rank、slot、projection或completion语义。
- `WaferProgramDeliveryFormat`不得链接WaferArtifact、WaferPackageFormat、Wafer IR或MLIR；WaferArtifact和WaferPackageFormat分别通过type-specific verified-delivery access消费distinct Delivery-owned commitment view，target不能形成target->package依赖。`WaferPackageFormat`只允许单向依赖runtime-neutral `WaferArtifact` session/read-work lease边界以实现production bundle reader；`WaferArtifact`绝不能反向include/link WaferPackageFormat或WaferRuntime。`WaferPackageFormat`仍不得链接Wafer IR、WaferTargetArtifacts、WaferPackageCompiler或MLIR。`WaferRuntime`链接两者以重验actual ELF，但不得包含Wafer IR/MLIR header或链接`WaferIR`、`WaferTargetArtifacts`、`WaferPackageCompiler`、任何MLIR library。只有`WaferPackageCompiler`通过`compiler::PackageAssemblyInputView`的sealed relation proof消费只读committed projection；无public raw `ExecutableOp` relation/assembly入口。
- logical `Resource`只拥有role/type/access/lifetime/alias/state语义；storage/capacity/arena/scope/shard/packing/accepted range只来自manifest中的repeated typed `ResourceRealization`。发布package只接受commit后规范化的`RankClassId` coverage；candidate `ExecutionInstanceId` coverage不进入package。Runtime不得从flat resource、selected module或名字重新计算realization。
- low-precision facts复用shared `QuantizationDescriptor`、`StorageEncodingDescriptor`、`QuantStorageAbiProfileV1`和strong `QuantStorageAbiProfileId`。Manifest保留logical quant descriptor、per-realization storage encoding、selected profile refs/records及artifact member实际`usedQuantStorageProfiles`闭包；runtime只验证和绑定ordinary resources，不解释quant公式、不repack、不选择native/composite fallback。
- immutable payload refs只接受upstream committed `ArtifactChunkingPolicyV1` canonical table：fixed 4 MiB chunks except the final tail，window overlap引用同一chunk records而不重切。Package验证/copy exact bytes和table，不从frontend path/NPY重建、不按worker/buffer/runtime window重新chunk。
- package bundle root不是独立publication unit。writer只能向同一个 `compiler::ProgramOutputTransaction` stage/attach immutable root和package reference；只有outer driver在completion scope中的executable、全部target sets和package都成功后，才通过一次unified-root rename或可信namespace中的单一 `ProgramDeliveryCommitRecord` 建立可见性。多目录/object-store root在commit前即使完整也只是可GC orphan，不能被loader接受。
- production load只接受deployment/control-plane factory从可信program delivery commit构造的 `VerifiedPackageBundleReference`，其中外部绑定root capability acquisition、bundle-index exact size/digest、manifest exact size/digest和expected `PackageManifestId`。bundle index、manifest、路径或文件名不能提供/覆盖expected值；diagnostic accept-any结果与production loaded type不互转，也不能创建RuntimeSession。
- `LoadedPackageMetadata`不可变持有manifest/index exact backings、metadata owner proof、root capability和不可open的`UnboundBlobSourceDescriptor`；metadata阶段没有`BoundBlobSource`。只有exact-domain one-way bind重验metadata/plan/context/runtime-session owner后才产生`RuntimeBoundPackage`及private `BoundBlobSource`。runtime-private access先联合取得同owner `ArtifactVerificationReadLease`再beneath/no-follow打开`OpenedBlobLease`，验证工作另持`ArtifactVerificationWorkLease`；未选择blob不占object FD，也不能把checked path交给provider。
- package不含`instructions` shadow schedule、自由`binding_order`、scalar completion source、LLVM regex ABI、module path identity、runtime handle或physical address。
- `RuntimeSessionPlan`和`RuntimeSession`是进程内派生对象，不序列化、不写回manifest、不生成variant/rank class/route/binding。
- state consistency只允许`atomic_version`和`in_place_poison_on_failure`；cleanup failure不得覆盖原始launch/timeout/status failure。
- Cross-model state attach默认失败：只有WCRE record 21 `StateMigrationPlanV1`/non-interchangeable `StateMigrationPlanId`的trusted verified plan可把old完整group迁移为new unpublished candidate；record直接含完整canonical old/new typed descriptor relations并逐字段join manifests，不使用名字、ordinal、generic descriptor fingerprint或caller callback。
- 默认no-card build不得硬链接TX runtime/KMD/legacy库；provider symbol存在只证明capability discovery，不证明launch或completion。
- 实施发现需要新增semantic field、enum policy、runtime selection自由度或package side channel时，先修改对应编号设计文档；本计划不能自行扩ABI。

---

## File Ownership Map

| Path | Responsibility |
| --- | --- |
| `schema/wafer/package_manifest.proto` | 唯一package wire shape、field number、enum和oneof声明；同时声明delivery-only bundle index |
| `schema/wafer/program_delivery.proto` | nonsemantic outer delivery commit wire、completion scope、child root commitments和exact digest/size/locator |
| `include/Wafer/Verification/HostVerificationRegistry.h` | runtime-neutral physical host ledger、trusted inventory bootstrap、三种不可互转operational child capability/lease |
| `include/Wafer/Delivery/ProgramDeliveryVerificationSession.h` | Delivery metadata typed child registry/session；不取得Artifact或device authority |
| `include/Wafer/Package/PackageManifest.h` | runtime-safe validated manifest、parse/serialize result和single verifier public API |
| `lib/Wafer/Package/PackageManifestVerifier.cpp` | resource/variant/entry/projection/artifact/completion跨字段semantic legality |
| `lib/Wafer/Package/PackageManifestIdentity.cpp` | validated message到WCRE record type 10和`PackageManifestId`的唯一投影 |
| `lib/Wafer/Package/PackageManifestIO.cpp` | exact blob digest、Protobuf parse、unknown-field rejection、deterministic writer和debug JSON output |
| `include/Wafer/Package/PackageAssembly.h` | compiler-only committed executable + complete artifact set assembly API |
| `lib/Wafer/Package/PackageAssembly.cpp` | lossless executable/artifact projection；不读取打印IR或名字恢复语义 |
| `include/Wafer/Package/PackageBundle.h` | runtime-safe `LoadedPackageMetadata`、inert `UnboundBlobSourceDescriptor`和delivery index metadata reader API |
| `include/Wafer/Delivery/ProgramDelivery.h` | package-independent trusted outer commit reference、limited verifier和private child commitment proof |
| `lib/Wafer/Package/PackageBundleReader.cpp` | 无MLIR的index/manifest校验和immutable blob handle绑定 |
| `lib/Wafer/Delivery/ProgramDelivery.cpp` | external commit-byte gate、scope/root commitment verifier和immutable verified delivery owner |
| `lib/Wafer/Package/PackageBundleWriter.cpp` | compiler-only private staging、typed artifact-set consumption和outer transaction attach；不独立publication |
| `include/Wafer/Package/V2PackageConverter.h` | 显式schema v2迁移入口；普通loader不可见 |
| `lib/Wafer/Package/V2PackageConverter.cpp` | compiler-only legacy JSON审计与typed事实交叉验证；歧义即拒绝 |
| `include/Wafer/Runtime/RuntimeSession.h` | verified environment、invocation/preflight/session typed public API |
| `include/Wafer/Runtime/RuntimeIdentity.h` | Invocation/EntryInstance/transient Scope、durable PersistentStateScope、Version/StateNamespace runtime-only strong values；无Protobuf conversion |
| `include/Wafer/Runtime/RuntimeProviderCapabilities.h` | complete per-domain low-level provider capability set；不构造semantic proof |
| `include/Wafer/Runtime/RuntimeServiceBootstrap.h` | authenticated provider/inventory/authority/state/host capability one-way bootstrap bundle |
| `include/Wafer/Runtime/RuntimePackageBinding.h` | metadata/plan/context/runtime-session one-way bind、`RuntimeBoundPackage`和private runtime blob sources |
| `include/Wafer/Runtime/RuntimeModules.h` | actual-ELF proof、verified entry view和ExecutableHandle边界 |
| `include/Wafer/Runtime/RuntimeCapacityReservation.h` | service-context-owned多域/跨进程fenced原子reservation、cache marginal accounting和FIFO wait |
| `include/Wafer/Runtime/ModuleResidencyManager.h` | repo-owned final provider/device-scoped module reservation/cache/liveness和ModuleLease owner |
| `include/Wafer/Runtime/PersistentStateRegistry.h` | repo-owned final session外ResourceVersionId/current/poison owner |
| `lib/Wafer/Runtime/RuntimeEnvironment.cpp` | runtime-safe typed target/topology facts验证和fingerprint重算 |
| `lib/Wafer/Runtime/RuntimePreflight.cpp` | target/projection/shape选择、scope key和exact slot binding；无provider调用 |
| `lib/Wafer/Runtime/RuntimeModules.cpp` | 从selected same-handle `ImmutableByteBackingRef`流式重验actual ELF/KAD/module view并解析typed executable handle |
| `lib/Wafer/Runtime/RuntimeCapacityReservation.cpp` | all-or-none reserve、DAG-terminal增量释放、single-flight cache ownership和generation invalidation |
| `lib/Wafer/Runtime/ModuleResidencyManager.cpp` | eager-active-set或graph-liveness reservation、single-flight load、lease/release/eviction |
| `lib/Wafer/Runtime/RuntimeResources.cpp` | arena/resource/weight/workspace/control实例化和逆序释放 |
| `lib/Wafer/Runtime/PersistentStateRegistry.cpp` | create/attach/reset/begin-update/publish/poison状态机 |
| `lib/Wafer/Runtime/CompletionExecutor.cpp` | entry/completion DAG、timeout/failure join、state publish/poison和cleanup |
| `lib/Wafer/Runtime/DryRunProviderCapabilities.cpp` | registry-owned no-card capability adapter，明确`not executed` |
| `lib/Wafer/Runtime/RecordingProviderCapabilities.cpp` | registry-owned no-card call recording与deterministic failure injection，不拥有语义 |
| `lib/Wafer/Runtime/TxProviderCapabilities.cpp` | 动态provider capability adapter和typed API mapping；板端执行由后续board计划验收 |

### Task 0: Outer Program Delivery Foundation

> Execute this task after target-artifact shared Tasks 1-5 and before whole-variant Task 1. It is a prerequisite for
> whole-variant work, target-artifact post-commit Tasks 6-14, package-manifest work and runtime work; it is not a
> prerequisite of target-artifact shared Tasks 1-5. Later tasks consume this owner and must not create a partial
> `compiler::ProgramOutputTransaction` or duplicate delivery codec.

**Files:**
- Create: `schema/wafer/program_delivery.proto`
- Modify: `schema/CMakeLists.txt`
- Create: `include/Wafer/Verification/HostVerificationRegistry.h`
- Create: `lib/Wafer/Verification/TrustedHostVerificationBootstrap.h`
- Create: `lib/Wafer/Verification/HostVerificationRegistry.cpp`
- Create: `lib/Wafer/Verification/TrustedHostVerificationBootstrap.cpp`
- Create: `lib/Wafer/Verification/CMakeLists.txt`
- Create: `include/Wafer/Delivery/ProgramDelivery.h`
- Create: `include/Wafer/Delivery/ProgramDeliveryVerificationSession.h`
- Create: `lib/Wafer/Delivery/ProgramDelivery.cpp`
- Create: `lib/Wafer/Delivery/ProgramDeliveryVerificationSession.cpp`
- Create: `include/Wafer/Compiler/ProgramOutputTransaction.h`
- Create: `lib/Wafer/Compiler/ProgramOutputTransaction.cpp`
- Create: `lib/Wafer/Compiler/ProgramOutputTransactionInternal.h`
- Modify: `lib/Wafer/CMakeLists.txt`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Create: `lib/Wafer/Delivery/CMakeLists.txt`
- Create: `unittests/Delivery/ProgramDeliverySchemaTest.cpp`
- Create: `unittests/Delivery/ProgramDeliveryTest.cpp`
- Create: `unittests/Delivery/ProgramDeliveryVerificationSessionTest.cpp`
- Create: `unittests/Verification/HostVerificationRegistryTest.cpp`
- Create: `unittests/Verification/TrustedHostVerificationBootstrapTest.cpp`
- Create: `unittests/Compiler/ProgramOutputTransactionTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: shared semantic/content digest value types; deployment-created verified sink capability; installed/authenticated deployment host inventory and OS hard-limit probe available only to the lower Verification bootstrap; positive requested `ProgramOutputLimits` and host/delivery operational limits; move-only `CanonicalEncodingContext`s built from validated limits and bounded scratch; no executable/target/package semantics yet.
- Produces: lower runtime-neutral `WaferVerificationSupport` with one physical `HostVerificationRegistry` and three noninterchangeable typed sub-capabilities for Delivery metadata, Artifact metadata and Runtime artifact work; the sole delivery wire/limited verifier and independent `ProgramDeliveryVerificationRegistry`/session; trusted outer reference, `VerifiedProgramDelivery`, complete outer transaction state machine, closed completion scope, stable attachment-identity tokens plus attachment-table-generation snapshot views, and the only atomic visibility transition used by later stages.

This combined excerpt shows both headers for proximity. In the actual files, `ProgramOutputTransaction.h` includes the complete lower-layer `ProgramDelivery.h` before declaring any `llvm::Expected<delivery::PublishedProgramDelivery>`; it never relies on an incomplete Delivery type.

```cpp
namespace wafer::verification {
namespace detail {
struct HostVerificationRegistryStorage;
struct TrustedHostVerificationInventoryStorage;
struct HostVerificationLedgerCapabilityStorage;
struct ProgramDeliveryHostBudgetStorage;
struct ArtifactMetadataHostBudgetStorage;
struct RuntimeArtifactHostBudgetStorage;
struct ProgramDeliveryHostLeaseStorage;
struct ArtifactMetadataHostLeaseStorage;
struct RuntimeArtifactHostLeaseStorage;
} // namespace detail

class TrustedDeploymentBootstrapAccess;
class ProgramDeliveryHostBudgetCapability;
class ArtifactMetadataHostBudgetCapability;
class RuntimeArtifactHostBudgetCapability;

struct HostVerificationLimitValues {
  // Positive checked process/deployment physical maxima.
  std::uint64_t maxActiveReaders;
  std::uint64_t maxOpenFileDescriptors;
  std::uint64_t maxConcurrentWorkers;
  std::uint64_t maxSimultaneouslyVerifiedBytes;
  std::uint64_t maxWorkspaceBytes;
};

struct HostVerificationDemand {
  std::uint64_t activeReaders;
  std::uint64_t openFileDescriptors;
  std::uint64_t concurrentWorkers;
  std::uint64_t simultaneouslyVerifiedBytes;
  std::uint64_t workspaceBytes;
};

class TrustedHostVerificationInventory final {
public:
  TrustedHostVerificationInventory(TrustedHostVerificationInventory &&) noexcept;
  TrustedHostVerificationInventory(
      const TrustedHostVerificationInventory &) = delete;
  ~TrustedHostVerificationInventory();
private:
  friend class TrustedDeploymentBootstrapAccess;
  explicit TrustedHostVerificationInventory(
      std::unique_ptr<detail::TrustedHostVerificationInventoryStorage> storage);
  std::unique_ptr<detail::TrustedHostVerificationInventoryStorage> storage_;
};

class HostVerificationLedgerCapability final {
public:
  HostVerificationLedgerCapability(HostVerificationLedgerCapability &&) noexcept;
  HostVerificationLedgerCapability(const HostVerificationLedgerCapability &) = delete;
  ~HostVerificationLedgerCapability();
private:
  friend class TrustedDeploymentBootstrapAccess;
  explicit HostVerificationLedgerCapability(
      std::unique_ptr<detail::HostVerificationLedgerCapabilityStorage> storage);
  std::unique_ptr<detail::HostVerificationLedgerCapabilityStorage> storage_;
};

class ProgramDeliveryHostLease final {
public:
  ProgramDeliveryHostLease(ProgramDeliveryHostLease &&) noexcept;
  ProgramDeliveryHostLease(const ProgramDeliveryHostLease &) = delete;
  ~ProgramDeliveryHostLease();
private:
  friend class ProgramDeliveryHostBudgetCapability;
  explicit ProgramDeliveryHostLease(
      std::unique_ptr<detail::ProgramDeliveryHostLeaseStorage> storage);
  std::unique_ptr<detail::ProgramDeliveryHostLeaseStorage> storage_;
};

class ArtifactMetadataHostLease final {
public:
  ArtifactMetadataHostLease(ArtifactMetadataHostLease &&) noexcept;
  ArtifactMetadataHostLease(const ArtifactMetadataHostLease &) = delete;
  ~ArtifactMetadataHostLease();
private:
  friend class ArtifactMetadataHostBudgetCapability;
  explicit ArtifactMetadataHostLease(
      std::unique_ptr<detail::ArtifactMetadataHostLeaseStorage> storage);
  std::unique_ptr<detail::ArtifactMetadataHostLeaseStorage> storage_;
};

class RuntimeArtifactHostLease final {
public:
  RuntimeArtifactHostLease(RuntimeArtifactHostLease &&) noexcept;
  RuntimeArtifactHostLease(const RuntimeArtifactHostLease &) = delete;
  ~RuntimeArtifactHostLease();
private:
  friend class RuntimeArtifactHostBudgetCapability;
  explicit RuntimeArtifactHostLease(
      std::unique_ptr<detail::RuntimeArtifactHostLeaseStorage> storage);
  std::unique_ptr<detail::RuntimeArtifactHostLeaseStorage> storage_;
};

class ProgramDeliveryHostBudgetCapability final {
public:
  ProgramDeliveryHostBudgetCapability(
      ProgramDeliveryHostBudgetCapability &&) noexcept;
  ProgramDeliveryHostBudgetCapability(
      const ProgramDeliveryHostBudgetCapability &) = delete;
  ~ProgramDeliveryHostBudgetCapability();
  llvm::Expected<ProgramDeliveryHostLease>
  tryReserve(HostVerificationDemand demand);
private:
  friend class HostVerificationRegistry;
  explicit ProgramDeliveryHostBudgetCapability(
      std::unique_ptr<detail::ProgramDeliveryHostBudgetStorage> storage);
  std::unique_ptr<detail::ProgramDeliveryHostBudgetStorage> storage_;
};

class ArtifactMetadataHostBudgetCapability final {
public:
  ArtifactMetadataHostBudgetCapability(
      ArtifactMetadataHostBudgetCapability &&) noexcept;
  ArtifactMetadataHostBudgetCapability(
      const ArtifactMetadataHostBudgetCapability &) = delete;
  ~ArtifactMetadataHostBudgetCapability();
  llvm::Expected<ArtifactMetadataHostLease>
  tryReserve(HostVerificationDemand demand);
private:
  friend class HostVerificationRegistry;
  explicit ArtifactMetadataHostBudgetCapability(
      std::unique_ptr<detail::ArtifactMetadataHostBudgetStorage> storage);
  std::unique_ptr<detail::ArtifactMetadataHostBudgetStorage> storage_;
};

class RuntimeArtifactHostBudgetCapability final {
public:
  RuntimeArtifactHostBudgetCapability(
      RuntimeArtifactHostBudgetCapability &&) noexcept;
  RuntimeArtifactHostBudgetCapability(
      const RuntimeArtifactHostBudgetCapability &) = delete;
  ~RuntimeArtifactHostBudgetCapability();
  llvm::Expected<RuntimeArtifactHostLease>
  tryReserve(HostVerificationDemand demand);
private:
  friend class HostVerificationRegistry;
  explicit RuntimeArtifactHostBudgetCapability(
      std::unique_ptr<detail::RuntimeArtifactHostBudgetStorage> storage);
  std::unique_ptr<detail::RuntimeArtifactHostBudgetStorage> storage_;
};

class TrustedDeploymentBootstrapAccess final {
public:
  static llvm::Expected<TrustedHostVerificationInventory>
  loadInstalledHostVerificationInventory();
  static llvm::Expected<HostVerificationLedgerCapability>
  issueHostVerificationLedger(TrustedHostVerificationInventory inventory);
};

class HostVerificationRegistry final {
public:
  static llvm::Expected<HostVerificationRegistry> create(
      HostVerificationLedgerCapability capability,
      HostVerificationLimitValues limits);
  HostVerificationRegistry(HostVerificationRegistry &&) noexcept;
  HostVerificationRegistry(const HostVerificationRegistry &) = delete;
  ~HostVerificationRegistry();
  llvm::Expected<ProgramDeliveryHostBudgetCapability>
  issueProgramDeliveryBudget();
  llvm::Expected<ArtifactMetadataHostBudgetCapability>
  issueArtifactMetadataBudget();
  llvm::Expected<RuntimeArtifactHostBudgetCapability>
  issueRuntimeArtifactBudget();
private:
  explicit HostVerificationRegistry(
      std::unique_ptr<detail::HostVerificationRegistryStorage> storage);
  std::unique_ptr<detail::HostVerificationRegistryStorage> storage_;
};
} // namespace wafer::verification

namespace wafer::artifact::detail {
class ProgramDeliveryTargetAccess;
} // namespace wafer::artifact::detail

namespace wafer::package::detail {
class ProgramDeliveryPackageAccess;
} // namespace wafer::package::detail

namespace wafer::delivery {
namespace detail {
class ProgramDeliveryVerificationSessionAccess;
struct ProgramDeliveryVerificationRegistryStorage;
struct ProgramDeliveryVerificationCapabilityStorage;
struct ProgramDeliveryVerificationSessionStorage;
} // namespace detail

class ProgramDeliveryParseLimits;
class ProgramDeliveryVerificationSession;

class ProgramDeliveryVerificationCapability final {
public:
  ProgramDeliveryVerificationCapability(
      ProgramDeliveryVerificationCapability &&) noexcept;
  ProgramDeliveryVerificationCapability(
      const ProgramDeliveryVerificationCapability &) = delete;
  ~ProgramDeliveryVerificationCapability();
private:
  friend class ProgramDeliveryVerificationRegistry;
  friend class detail::ProgramDeliveryVerificationSessionAccess;
  explicit ProgramDeliveryVerificationCapability(
      std::unique_ptr<detail::ProgramDeliveryVerificationCapabilityStorage> storage);
  std::unique_ptr<detail::ProgramDeliveryVerificationCapabilityStorage> storage_;
};

struct ProgramDeliveryVerificationLimitValues {
  // Positive checked values; zero never means unbounded.
  std::uint64_t maxActiveReaders;
  std::uint64_t maxOpenFileDescriptors;
  std::uint64_t maxConcurrentWorkers;
  std::uint64_t maxSimultaneouslyVerifiedBytes;
  std::uint64_t maxWorkspaceBytes;
};

class ProgramDeliveryVerificationSession final {
public:
  ProgramDeliveryVerificationSession(
      ProgramDeliveryVerificationSession &&) noexcept;
  ProgramDeliveryVerificationSession(
      const ProgramDeliveryVerificationSession &) = delete;
  ~ProgramDeliveryVerificationSession();
private:
  friend class detail::ProgramDeliveryVerificationSessionAccess;
  explicit ProgramDeliveryVerificationSession(
      std::unique_ptr<detail::ProgramDeliveryVerificationSessionStorage> storage);
  std::unique_ptr<detail::ProgramDeliveryVerificationSessionStorage> storage_;
};

class ProgramDeliveryVerificationRegistry final {
public:
  static llvm::Expected<ProgramDeliveryVerificationRegistry> create(
      verification::ProgramDeliveryHostBudgetCapability capability,
      ProgramDeliveryVerificationLimitValues limits);
  ProgramDeliveryVerificationRegistry(
      ProgramDeliveryVerificationRegistry &&) noexcept;
  ProgramDeliveryVerificationRegistry(
      const ProgramDeliveryVerificationRegistry &) = delete;
  ~ProgramDeliveryVerificationRegistry();
  llvm::Expected<ProgramDeliveryVerificationCapability>
  issueCapability();
private:
  explicit ProgramDeliveryVerificationRegistry(
      std::unique_ptr<detail::ProgramDeliveryVerificationRegistryStorage> storage);
  std::unique_ptr<detail::ProgramDeliveryVerificationRegistryStorage> storage_;
};

llvm::Expected<ProgramDeliveryVerificationSession>
createProgramDeliveryVerificationSession(
    ProgramDeliveryVerificationCapability capability,
    ProgramDeliveryParseLimits limits,
    abi::CanonicalEncodingContext encoding);
} // namespace wafer::delivery

namespace wafer::compiler {
namespace detail {
class FrontendProgramOutputAccess;
class ExecutableProgramOutputAccess;
class TargetProgramOutputAccess;
class PackageProgramOutputAccess;
} // namespace detail

class CanonicalEncodingOwnerToken {
public:
  CanonicalEncodingOwnerToken(const CanonicalEncodingOwnerToken &) = default;
  CanonicalEncodingOwnerToken &operator=(
      const CanonicalEncodingOwnerToken &) = default;
  bool sameLiveOwner(const CanonicalEncodingOwnerToken &) const;
private:
  friend class CanonicalEncodingSession;
  class Storage;
  explicit CanonicalEncodingOwnerToken(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class CanonicalEncodingSession {
private:
  friend class ProgramOutputTransaction;
  friend class detail::FrontendProgramOutputAccess;
  friend class detail::ExecutableProgramOutputAccess;
  friend class detail::TargetProgramOutputAccess;
  friend class detail::PackageProgramOutputAccess;
  abi::CanonicalEncodingContext &context();
  const CanonicalEncodingOwnerToken &ownerToken() const;
  class Storage;
  explicit CanonicalEncodingSession(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class StagedExecutableToken final {
public:
  StagedExecutableToken(StagedExecutableToken &&) noexcept;
  StagedExecutableToken(const StagedExecutableToken &) = delete;
  ~StagedExecutableToken();
private:
  friend class detail::ExecutableProgramOutputAccess;
  friend class detail::TargetProgramOutputAccess;
  friend class ProgramOutputTransaction;
  class Storage;
  explicit StagedExecutableToken(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class StagedTargetArtifactSetToken final {
public:
  StagedTargetArtifactSetToken(StagedTargetArtifactSetToken &&) noexcept;
  StagedTargetArtifactSetToken(const StagedTargetArtifactSetToken &) = delete;
  ~StagedTargetArtifactSetToken();
private:
  friend class detail::TargetProgramOutputAccess;
  friend class ProgramOutputTransaction;
  class Storage;
  explicit StagedTargetArtifactSetToken(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class PackageAssemblyInputView final {
public:
  PackageAssemblyInputView(PackageAssemblyInputView &&) noexcept;
  PackageAssemblyInputView(const PackageAssemblyInputView &) = delete;
  ~PackageAssemblyInputView();
private:
  friend class detail::PackageProgramOutputAccess;
  friend class ProgramOutputTransaction;
  class Storage;
  explicit PackageAssemblyInputView(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class StagedPackageBundleToken final {
public:
  StagedPackageBundleToken(StagedPackageBundleToken &&) noexcept;
  StagedPackageBundleToken(const StagedPackageBundleToken &) = delete;
  ~StagedPackageBundleToken();
private:
  friend class detail::PackageProgramOutputAccess;
  friend class ProgramOutputTransaction;
  class Storage;
  explicit StagedPackageBundleToken(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class ProgramOutputTransaction {
public:
  static llvm::Expected<ProgramOutputTransaction> create(
      delivery::ProgramDeliveryCompletionScope completionScope,
      delivery::VerifiedProgramOutputSink sink,
      delivery::ProgramOutputLimits limits,
      abi::CanonicalEncodingContext canonicalEncoding);
  ProgramOutputTransaction(ProgramOutputTransaction &&) noexcept;
  ProgramOutputTransaction &operator=(ProgramOutputTransaction &&) noexcept;
  ~ProgramOutputTransaction();

  llvm::Expected<delivery::PublishedProgramDelivery> commit();
private:
  friend class detail::FrontendProgramOutputAccess;
  friend class detail::ExecutableProgramOutputAccess;
  friend class detail::TargetProgramOutputAccess;
  friend class detail::PackageProgramOutputAccess;
  CanonicalEncodingSession &encodingSessionForStage();
  class Storage;
  explicit ProgramOutputTransaction(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};
} // namespace wafer::compiler

namespace wafer::delivery {
enum class ProgramDeliveryCompletionScope : std::uint8_t {
  ExecutableOnly,
  TargetArtifacts,
  PackageBundle
};

struct ProgramOutputLimitValues {
  std::uint64_t maxLogicalStagedBytes;
  std::uint64_t maxPhysicalStagedBytes;
  std::uint64_t maxUniqueBlobs;
  std::uint64_t maxContentRoots;
  std::uint64_t maxInodes;
  std::uint64_t maxMetadataIndexCommitBytes;
  std::uint64_t maxOpenFileDescriptors;
  std::uint64_t maxChildWorkers;
  std::uint64_t maxReadWriteBufferBytes;
  std::uint64_t maxRetainedOrphanBytes;
  std::uint64_t maxRetainedOrphanAgeSeconds;
};

class ProgramOutputLimits {
public:
  static llvm::Expected<ProgramOutputLimits>
  create(ProgramOutputLimitValues values);
  const ProgramOutputLimitValues &values() const;
private:
  explicit ProgramOutputLimits(ProgramOutputLimitValues values);
  ProgramOutputLimitValues values_;
};

class VerifiedProgramOutputSink {
public:
  VerifiedProgramOutputSink(VerifiedProgramOutputSink &&) noexcept;
  VerifiedProgramOutputSink &operator=(VerifiedProgramOutputSink &&) noexcept;
  ~VerifiedProgramOutputSink();
private:
  friend class compiler::ProgramOutputTransaction;
  class Storage;
  explicit VerifiedProgramOutputSink(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class OpenDirectoryCapability;
class AtomicCommitNamespaceCapability;
class VerifiedFilesystemDurabilityContract;
class VerifiedAtomicCasContract;
class DeploymentOutputNamespaceId;

class UnifiedFilesystemOutputRoot {
private:
  friend llvm::Expected<UnifiedFilesystemOutputRoot>
  verifyUnifiedFilesystemOutputRoot(
      OpenDirectoryCapability, DeploymentOutputNamespaceId, std::uint64_t,
      VerifiedFilesystemDurabilityContract);
  class Storage;
  explicit UnifiedFilesystemOutputRoot(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class TrustedCommitNamespace {
private:
  friend llvm::Expected<TrustedCommitNamespace>
  verifyTrustedCommitNamespace(
      AtomicCommitNamespaceCapability, DeploymentOutputNamespaceId,
      std::uint64_t, VerifiedAtomicCasContract);
  class Storage;
  explicit TrustedCommitNamespace(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

llvm::Expected<UnifiedFilesystemOutputRoot>
verifyUnifiedFilesystemOutputRoot(
    OpenDirectoryCapability alreadyOpenRoot,
    DeploymentOutputNamespaceId expectedIdentity,
    std::uint64_t expectedGeneration,
    VerifiedFilesystemDurabilityContract durability);

llvm::Expected<TrustedCommitNamespace>
verifyTrustedCommitNamespace(
    AtomicCommitNamespaceCapability capability,
    DeploymentOutputNamespaceId expectedIdentity,
    std::uint64_t expectedGeneration,
    VerifiedAtomicCasContract atomicCas);

llvm::Expected<VerifiedProgramOutputSink>
createUnifiedFilesystemProgramOutputSink(
    const UnifiedFilesystemOutputRoot &root,
    llvm::StringRef safeRelativeDeliveryLocator,
    std::uint64_t expectedNamespaceGeneration);

llvm::Expected<VerifiedProgramOutputSink>
createTrustedNamespaceProgramOutputSink(
    const TrustedCommitNamespace &nameSpace,
    llvm::StringRef commitKey,
    std::uint64_t expectedNamespaceGeneration);

class TrustedProgramDeliveryRef {
private:
  friend class PublishedProgramDelivery;
  friend llvm::Expected<TrustedProgramDeliveryRef>
  acquireTrustedProgramDeliveryRef(const TrustedCommitNamespace &,
                                   llvm::StringRef,
                                   std::uint64_t,
                                   std::uint64_t,
                                   const abi::ContentDigest &,
                                   ProgramDeliveryCompletionScope);
  friend llvm::Expected<TrustedProgramDeliveryRef>
  acquireTrustedProgramDeliveryRef(const UnifiedFilesystemOutputRoot &,
                                   llvm::StringRef,
                                   std::uint64_t,
                                   std::uint64_t,
                                   const abi::ContentDigest &,
                                   ProgramDeliveryCompletionScope);
  class Storage;
  explicit TrustedProgramDeliveryRef(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class PublishedProgramDelivery {
public:
  const TrustedProgramDeliveryRef &trustedReference() const;
  ProgramDeliveryCompletionScope completionScope() const;
  std::uint64_t exactCommitRecordSize() const;
  const abi::ContentDigest &exactCommitRecordDigest() const;
private:
  friend class compiler::ProgramOutputTransaction;
  class Storage;
  explicit PublishedProgramDelivery(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

llvm::Expected<TrustedProgramDeliveryRef>
acquireTrustedProgramDeliveryRef(
    const TrustedCommitNamespace &nameSpace,
    llvm::StringRef commitKey,
    std::uint64_t expectedNamespaceGeneration,
    std::uint64_t expectedExactSize,
    const abi::ContentDigest &expectedExactDigest,
    ProgramDeliveryCompletionScope expectedScope);

llvm::Expected<TrustedProgramDeliveryRef>
acquireTrustedProgramDeliveryRef(
    const UnifiedFilesystemOutputRoot &root,
    llvm::StringRef safeRelativeCommitRecordLocator,
    std::uint64_t expectedNamespaceGeneration,
    std::uint64_t expectedExactSize,
    const abi::ContentDigest &expectedExactDigest,
    ProgramDeliveryCompletionScope expectedScope);

struct ProgramDeliveryParseLimitValues {
  // Every value is positive and checked; zero never means unbounded.
  std::uint64_t maxRecordBytes;
  std::uint32_t maxRecursionDepth;
  std::uint64_t maxLocatorBytes;
  std::uint64_t maxStringBytes;
  std::uint64_t maxTotalLocatorAndStringBytes;
  std::uint64_t maxTargetRecords;
  std::uint64_t maxRootRecords;
  std::uint64_t maxReferenceRecords;
  std::uint64_t maxTotalRecords;
};

class ProgramDeliveryParseLimits {
public:
  static llvm::Expected<ProgramDeliveryParseLimits> create(
      ProgramDeliveryParseLimitValues values);
  const ProgramDeliveryParseLimitValues &values() const;
private:
  explicit ProgramDeliveryParseLimits(ProgramDeliveryParseLimitValues values);
  ProgramDeliveryParseLimitValues values_;
};

class VerifiedTargetDeliveryCommitmentView final {
public:
  VerifiedTargetDeliveryCommitmentView(
      VerifiedTargetDeliveryCommitmentView &&) noexcept;
  VerifiedTargetDeliveryCommitmentView(const VerifiedTargetDeliveryCommitmentView &) = delete;
  ~VerifiedTargetDeliveryCommitmentView();
private:
  friend class VerifiedProgramDelivery;
  friend class artifact::detail::ProgramDeliveryTargetAccess;
  class Storage;
  explicit VerifiedTargetDeliveryCommitmentView(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class VerifiedPackageDeliveryCommitmentView final {
public:
  VerifiedPackageDeliveryCommitmentView(
      VerifiedPackageDeliveryCommitmentView &&) noexcept;
  VerifiedPackageDeliveryCommitmentView(const VerifiedPackageDeliveryCommitmentView &) = delete;
  ~VerifiedPackageDeliveryCommitmentView();
private:
  friend class VerifiedProgramDelivery;
  friend class package::detail::ProgramDeliveryPackageAccess;
  class Storage;
  explicit VerifiedPackageDeliveryCommitmentView(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class VerifiedProgramDelivery {
public:
  ProgramDeliveryCompletionScope completionScope() const;
  const abi::ExecutableSemanticDigest &executable() const;
private:
  friend class artifact::detail::ProgramDeliveryTargetAccess;
  friend class package::detail::ProgramDeliveryPackageAccess;
  llvm::Expected<VerifiedTargetDeliveryCommitmentView>
  targetCommitment(const abi::TargetVariantId &) const;
  llvm::Expected<VerifiedPackageDeliveryCommitmentView>
  packageCommitment() const;
  friend llvm::Expected<VerifiedProgramDelivery>
  loadAndVerifyProgramDelivery(const TrustedProgramDeliveryRef &,
                               ProgramDeliveryVerificationSession &);
  class Storage;
  explicit VerifiedProgramDelivery(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

llvm::Expected<VerifiedProgramDelivery>
loadAndVerifyProgramDelivery(const TrustedProgramDeliveryRef &reference,
                             ProgramDeliveryVerificationSession &verification);
} // namespace wafer::delivery
```

The sealed producer values above are opaque move-owned pimpl boundaries with private friends; their semantic factories land in later tasks. `ProgramOutputTransaction.h` names only its complete compiler-owned token/view types and the complete downward Delivery types. It never names whole-program `CommittedExecutableProgram`, target builder/result, package bundle/result or any `mlir::FailureOr<upper-type>`. `ProgramOutputTransactionInternal.h` defines non-installed owner-specific commitment primitives plus test access; each upper stage's access adapter consumes its own complete object in that upper library, translates it once into the matching lower primitive and calls the transaction pimpl. `ExecutableCompilationInput` must initially move-own the canonical all-and-only `VerifiedTargetCompilationContextRegistry`; candidate selection transfers that same owner into the winner and then into `CommittedExecutableProgram`. Whole-program `detail::ExecutableProgramOutputAccess::attachCommittedExecutable(output, committed)` accepts only that one complete upper object and moves both executable proof and its already-bound registry into the distinct lower commitment. There is no overload or second parameter that re-pairs a context vector after commit. The lower commitment rechecks exact executable target-requirement coverage and lets `TargetProgramOutputAccess` resolve a context only by `(transaction, staged executable token, TargetVariantId)`. It never stores a borrowed view, context digest/profile surrogate or generic sidecar, and caller target code cannot resupply/replace a context. The compiler canonical encoding owner token and each `compiler::StagedExecutableToken`/`compiler::StagedTargetArtifactSetToken`/`compiler::StagedPackageBundleToken` remain stable for the transaction/attachment lifetime; they do not use the mutable attachment-table generation. A separate private `compiler::AttachmentTableGeneration` increments on each attach and is captured only by `compiler::PackageAssemblyInputView`/enumeration snapshots, so a stale view fails while the same executable token can attach two or more targets. These facts are exact typed commitments, not a generic blob bag/installed ABI/side channel. The two Delivery-owned commitment views are complete, move-only and non-forgeable, retain the exact verified root capability/owner and type-specific expected facts, and expose no public raw locator/digest/root accessor. `ProgramDelivery.h` forward-declares only the two upper non-installed access classes; it never names an upper result type or instantiates `llvm::Expected<T>` for an incomplete WaferArtifact/WaferPackageFormat type. Later target/package adapters consume only their distinct view and cannot reinterpret one as the other or enumerate a generic child bag.

`TrustedDeploymentBootstrapAccess::loadInstalledHostVerificationInventory()` is the sole production trust-root factory. It reads the installed authenticated deployment inventory and OS/process hard-limit probe through the lower Verification implementation, accepts no caller path or caller-provided maxima, and seals a move-only `TrustedHostVerificationInventory` containing deployment scope owner, generation and every physical hard maximum. `issueHostVerificationLedger` may consume that inventory exactly once and binds those same facts into `HostVerificationLedgerCapability`; the only fake issuer is linked into `WaferUnitTests`, never an installed or production target. `HostVerificationRegistry::create` move-consumes the capability, rejects zero/overflow and any requested dimension above the sealed hard maximum, and retains the owner/generation. A second registry cannot be issued for the same inventory generation.

`HostVerificationRegistry` is therefore created once per trusted process/deployment host scope. Its typed Delivery-metadata, Artifact-metadata and Runtime-artifact sub-capabilities are noninterchangeable and cannot reserve independently of the same parent reader/FD/worker/verified-byte/workspace counters, so separate semantic session owners cannot double-spend host resources. The lower header has no Delivery/Artifact/Runtime forward declaration, friend or semantic method: each typed capability exposes only its distinct `tryReserve(HostVerificationDemand)` and move-only RAII lease, while only `HostVerificationRegistry` can construct the capabilities. `ProgramDeliveryVerificationRegistry` move-consumes only the Delivery sub-capability and invokes that operational API; creating more Delivery sessions never resets either its typed subledger or the parent ledger. It is runtime-neutral and has no provider, device-domain, reservation-authority or Runtime dependency. Each move-only session owns one canonical context and a stable registry/host owner-generation binding. Non-installed session access composes the lower lease into read/work reservations: a read demand reserves reader+FD before the trusted reference performs same-handle open, and an independent work demand reserves worker+verified bytes/workspace through wire/hash/parse verification. They cannot be constructed, copied, split, released early or paired with another session/reference. Partial acquisition uses a fixed nonblocking order and reverse rollback; no wait occurs while holding a partial tuple. WaferArtifact later move-owns its distinct typed sub-capability and creates independent metadata/runtime sessions, never an upgraded Delivery session. None of these host-verification operations consumes device authority.

`UnifiedFilesystemOutputRoot` and `TrustedCommitNamespace` are non-forgeable deployment capability sources with retained namespace identity/generation and lifetime. Their sole verification factories consume an already-open directory or atomic namespace capability plus externally expected strong namespace identity/generation and a verified closed durability/atomic-CAS contract. They validate beneath/no-follow access, same-filesystem single rename or atomic create/CAS, fsync/directory durability and fencing before a sink factory can run; no raw path/global namespace/callback constructor exists. `PublishedProgramDelivery` returns the trusted reference from the successful sink. After restart or in another process, deployment inventory can use the backend-specific `acquireTrustedProgramDeliveryRef` overload for either verified root/namespace with external generation, locator/key, exact size/digest and scope; none is read from the record being opened.

- [ ] **Step 1: Write schema, state-machine, and trust failures**

Cover unsupported/unknown delivery version/field/scope, unsafe locator, duplicate/missing target/root, incorrect canonical order, wrong external commit size/digest/scope, record/ref/recursion/aggregate overflow, and deterministic exact-byte roundtrip. For every wire count/depth/string/locator limit, exercise limit, limit+1 and checked overflow while instrumenting the generated-message arena/allocator; no generated message or attacker-sized side vector may allocate before a successful `preflightProtoWire`, and post-parse reflection mismatch must expose no proof. Test trusted host bootstrap authentication/probe failure, zero/overflow hard maxima, requested-equals-hard success, requested-hard+1 failure, stale owner/generation, unique issuance and second-registry rejection. The production header has no caller-maxima issuer and no test fake; the fake inventory/issuer compiles only into `WaferUnitTests`. Test every lower typed capability/lease for move-only/non-aggregate/private construction and reject cross-kind use at compile time. Then test registry/session zero/overflow limits, stale/cross-registry owner, concurrent Delivery/Artifact-metadata/Runtime-artifact reservations exceeding their shared physical reader/FD/worker/byte/workspace parent, cancellation at every lease boundary and reverse rollback with exact counter restoration. Multiple independently sufficient local parse limits cannot overspend the one shared Delivery ledger, and no session creation can reset it. Sink/reference factories reject raw paths, unsafe keys, wrong namespace generation, missing fsync/atomic-rename/CAS/fencing capability and expired root lifetime; successful `PublishedProgramDelivery::trustedReference()` remains usable while its capability owner lives. Require `ProgramOutputTransaction::create` to reject zero/overflowing output/canonical limits, missing scratch and unverified sink. Assert completion scope, move-owned sink, output limits and encoding session/context have no default/setter/replacement/take or public accessor path. Only the four compiler owner-specific detail access classes can borrow the same transaction-lifetime session/compare its token; frontend/candidate/target/package proofs must retain it, cross-owner/generation backing/view use fails, every retained token/view becomes stale after commit/abort, and `commit()` accepts no arguments. Compile-time dependency tests must compile the lower Verification header with no Delivery/Artifact/Runtime declaration, compile `ProgramDelivery.h` without defining any WaferArtifact/WaferPackageFormat result type and compile `ProgramOutputTransaction.h` without whole/target/package/MLIR headers. They reject any lower friend/accessor naming an upper semantic owner, installed signature naming upper staged objects or `FailureOr<upper-type>`, public construction/copy/raw-field access on both commitment views and cross-use of the target/package access classes.

Require compiler staged tokens and `compiler::PackageAssemblyInputView` to be move-only, non-aggregate and private-construction. Reject cross-transaction token/view, stale attachment-table view, post-finalization use, caller target subset/reorder/extra, target/package inconsistency, second executable/package, incomplete commit, second commit and scope downgrade. Executable input/winner/commit/attachment tests reject empty/missing/extra/duplicate/noncanonical target-context registries, context/requirement ID or generation mismatch, lost owner transfer and borrowed context lifetime. The attach API must have exactly one upper argument and no caller vector/context parameter. After destroying the original compilation request, target access must resolve every exact retained `VerifiedTargetCompilationContext` by ID and reject caller replacement or unknown IDs before toolchain work. Through the upper non-installed access adapters, positively attach at least two target sets sequentially with the same stable `compiler::StagedExecutableToken`; then let `compiler::detail::PackageProgramOutputAccess` build a fresh view containing both exactly once, and prove an earlier view is stale while neither staged token nor compiler canonical owner token changed. Mixing canonical owner/session generation with attachment-table snapshot generation is a compile-time/API error. Any child error permanently marks the transaction uncommittable.

Exercise `ProgramOutputLimits` with child operations that individually fit but aggregate logical bytes, newly allocated physical bytes, unique blobs/roots/inodes, metadata/index/commit bytes, FDs, workers, buffers or retained-orphan budgets exceed the retained limit. Reservation is multidimensional all-or-none; exact `(digest,size)` dedup reduces only physical accounting, never logical refs. A content collision hard-fails. No test can change limits/sink/scope after staging to make a failed transaction committable.

For the two verified sink backends, inject file/fsync/directory-fsync/rename, immutable-root write, trusted-namespace create/CAS and crash-before-commit failures. A success has exactly one unified-root rename or one trusted commit-record CAS. Every failure leaves no accepted child alias/reference; unreferenced immutable roots are GC candidates. GC retains every root reachable from current/retained verified records or active leases, uses generation/refcount/fencing against readers, and never lets an orphan derive a production child reference. Diagnostic accept-any delivery output cannot construct `VerifiedProgramDelivery`.

- [ ] **Step 2: Implement the nonsemantic wire and limited verifier**

Generate `WaferProgramDeliveryProto` from a versioned non-WCRE `ProgramDeliveryCommitRecord` containing the closed completion scope, one executable exact-content commitment, canonical target commitments, optional package bundle-index/manifest commitments and referenced immutable-root IDs. Every child carries exact size/digest and safe relative locator; the record has no self digest, runtime handle, policy or semantic identity annotation. Pin deterministic Protobuf 3.21.9 output and compute external `ProgramDeliveryCommitRecordDigest` over exact bytes. Implement `ProgramDeliveryParseLimits` so a delivery-specific fixed plan/budget delegates to WaferABI `detail::preflightProtoWire` over owner-backed exact bytes before generated parse or attacker-derived allocation; `verifyParsedProtoMatchesPreflight` must match tag/wire/length/unknown/depth/string/locator/target/root/ref counts and spans before sealing the proof. `ProgramDeliveryVerificationRegistry` issues a move-only `ProgramDeliveryVerificationCapability` backed by its shared typed child ledger; the sole `createProgramDeliveryVerificationSession(std::move(capability), std::move(limits), std::move(encoding))` factory retains all three owners in the session. The only production loader is `loadAndVerifyProgramDelivery(reference, ProgramDeliveryVerificationSession &)`: it obtains same-owner read/work reservations before opening/hash/preflight/parse, verifies external expected bytes first, rejects scope incompleteness and uses the session-retained limits/context for every embedded typed semantic ID/child commitment verification. There is no raw path, per-call replacement limits, independent context, one-off/unbounded session or hidden process ledger. `artifact::detail::ProgramDeliveryTargetAccess` and `package::detail::ProgramDeliveryPackageAccess` alone may ask the proof for their distinct Delivery-owned commitment view and derive the corresponding upper reference. A child root/index cannot self-authenticate, neither adapter sees a generic child collection, and no default context overload exists.

Build `WaferProgramDeliveryFormat` from `lib/Wafer/Delivery/ProgramDelivery.cpp`; it links only `WaferABI`, `WaferProgramDeliveryProto`, `WaferProtoSupport`, LLVM Support and system capability-open/hash dependencies. It has no PackageManifest, WaferArtifact, WaferIR, target-builder or MLIR edge. The compiler-side transaction/writer target links this format library and MLIR/compiler owners in the forward direction. Later WaferArtifact and WaferPackageFormat non-installed access classes depend downward on Delivery, move the corresponding sealed view into their own child reference and never require Delivery to include an upper header or instantiate an upper template type. Extend dependency tests to reject target-to-package, delivery-to-target/package, delivery-to-MLIR and any public generic commitment accessor.

- [ ] **Step 3: Implement the bound transaction and sink backends**

`create(scope, verifiedSink, outputLimits, canonicalEncoding)` move-owns all four inputs and wraps the context in one compiler-only session with a stable non-forgeable owner token. The private pimpl protocol implements reserve/attach/completeness once. Each upper stage's non-installed access adapter receives its own complete staged object, a restricted transaction/session borrow and the matching lower token; it validates semantics in the upper owner, constructs one owner-specific lower commitment and retains the canonical owner token. No upper object, view factory or generic attachment function enters the installed transaction public API: it exposes only `create`, `commit` and the complete lower-owned token types. Attach requires `sameLiveOwner`, which compares private owner/session generation and irreversible alive state without raw access. No stage can replace/move context or pair another scratch registry. Child failures are terminal. Each successful attach advances only `compiler::AttachmentTableGeneration`; compiler staged identity/session tokens remain valid, while older package views fail. Package access privately builds `PackageAssemblyInputView` from the lower attachment snapshot plus each owner-specific sealed typed access and captures the current table generation only after exact coverage. `commit()` builds one record, revalidates roots/sink fence and invokes the sink once. Commit/abort finalizes shared owner-token storage, making every retained token/view stale; scratch dies after dependent proof cleanup. No replacement parameters or CAS-to-multi-rename fallback exists.

- [ ] **Step 4: Run the foundation gates**

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
build/wafer-dev/bin/WaferUnitTests \
  --gtest_filter='TrustedHostVerificationBootstrapTest.*:HostVerificationRegistryTest.*:ProgramDeliverySchemaTest.*:ProgramDeliveryTest.*:ProgramDeliveryVerificationSessionTest.*:ProgramOutputTransactionTest.*'
python3 tools/check_deps.py
```

Expected: the exact codec/trust negatives and aggregate output-budget/transaction/sink/GC failures pass; no target/package source is needed and no child root can be accepted without one outer commit.

- [ ] **Step 5: Commit**

```bash
git add schema/wafer/program_delivery.proto schema/CMakeLists.txt \
  include/Wafer/Verification/HostVerificationRegistry.h \
  lib/Wafer/Verification/TrustedHostVerificationBootstrap.h \
  lib/Wafer/Verification/HostVerificationRegistry.cpp \
  lib/Wafer/Verification/TrustedHostVerificationBootstrap.cpp \
  lib/Wafer/Verification/CMakeLists.txt \
  include/Wafer/Delivery/ProgramDelivery.h \
  include/Wafer/Delivery/ProgramDeliveryVerificationSession.h \
  lib/Wafer/Delivery/ProgramDelivery.cpp \
  lib/Wafer/Delivery/ProgramDeliveryVerificationSession.cpp \
  include/Wafer/Compiler/ProgramOutputTransaction.h \
  lib/Wafer/Compiler/ProgramOutputTransaction.cpp \
  lib/Wafer/Compiler/ProgramOutputTransactionInternal.h \
  lib/Wafer/CMakeLists.txt lib/Wafer/Compiler/CMakeLists.txt \
  lib/Wafer/Delivery/CMakeLists.txt unittests/Delivery/ProgramDeliverySchemaTest.cpp \
  unittests/Delivery/ProgramDeliveryTest.cpp \
  unittests/Delivery/ProgramDeliveryVerificationSessionTest.cpp \
  unittests/Verification/HostVerificationRegistryTest.cpp \
  unittests/Verification/TrustedHostVerificationBootstrapTest.cpp \
  unittests/Compiler/ProgramOutputTransactionTest.cpp unittests/CMakeLists.txt
git commit -m "Add outer program delivery foundation"
```

### Task 1: PackageManifest Schema and Generated Bindings

**Files:**
- Create: `schema/wafer/package_manifest.proto`
- Modify: `schema/CMakeLists.txt`
- Modify: `lib/Wafer/CMakeLists.txt`
- Create: `include/Wafer/Package/PackageManifest.h`
- Create: `lib/Wafer/Package/CMakeLists.txt`
- Create: `lib/Wafer/Package/PackageManifest.cpp`
- Create: `unittests/Package/PackageManifestSchemaTest.cpp`
- Create: `test/Tools/package-manifest-generated-binding.test`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: shared `WaferProtoSupport`; generated identity/KAD messages; `schema/wafer/target_artifact_set.proto`生成的locator-free `TargetArtifactSetVerifiedRecord`和`WaferTargetArtifactSetProto`; tasks/15 typed package contents. Delivery root/locator不是manifest schema input。
- Produces: generated target `WaferPackageProto`; `wafer::package::proto::PackageManifest`; `wafer::package::proto::PackageBundleIndex`; C++ schema-version constants exposed by `PackageManifest.h`. Task 0 already owns program-delivery codegen and trust types.

- [ ] **Step 1: Write schema-shape tests before the proto exists**

Add a GTest that uses generated descriptors and requires these top-level facts:

```cpp
TEST(PackageManifestSchemaTest, ContainsOnlyTypedLongTermSections) {
  const auto *manifest =
      wafer::package::proto::PackageManifest::descriptor();
  for (llvm::StringRef name : {"model_interface", "resources",
                               "state_groups",
                               "executable_variants", "target_artifact_sets",
                               "rank_classes", "entries", "projection_sets",
                               "transports",
                               "completion_nodes", "completion_edges",
                               "runtime_requirements"})
    EXPECT_NE(manifest->FindFieldByName(name.str()), nullptr) << name.str();
  EXPECT_EQ(manifest->FindFieldByName("instructions"), nullptr);
  EXPECT_EQ(manifest->FindFieldByName("binding_order"), nullptr);
  EXPECT_EQ(manifest->FindFieldByName("completion_source"), nullptr);
}

TEST(PackageManifestSchemaTest, UsesNoMapOrFloatingIdentityFields) {
  assertNoMapOrFloatingFieldReachableFrom(
      wafer::package::proto::PackageManifest::descriptor());
}
```

Add descriptor assertions that `ModelInterface` has ordered typed `ModelEntrypointId` records, every `ExecutableVariant` has repeated typed `ModelInvocationMapping`, and each mapping owns initial roots, required IO/state contract and terminal roots without a symbol/static `EntryId` user API field. Every resource reference imports `wafer.identity.proto.ResourceIdentityRef` with its tagged oneof, `Resource.realizations` is repeated `ResourceRealization`, realization coverage is a non-empty committed `RankClassId` set, logical low-precision resources use the imported shared `QuantizationDescriptor`, and each realization uses the imported `StorageEncodingDescriptor` plus typed `QuantStorageAbiProfileId` refs. Every `RankClass` carries canonical typed `ExecutionInstance` mapping, every executable variant can own repeated finite `EntryGraphIterationDomain`, finite typed `ActivationPredicate`/`ExpertWave` records and explicit node activation (`always | predicate true | predicate false`); each predicate uses imported `ActivationPredicateId` and closed `bounded_count_nonzero`, never a script/name/provider condition. Every entry/completion node has an explicit singleton/domain binding, every graph edge carries `iteration_delta`, every `Transport` has repeated complete `TransportBindingMember`, and relocation slots reference an action-local `TransportBindingMemberId`. Assert `PackageManifest.target_artifact_sets` directly uses imported locator-free, nested-record-annotated `TargetArtifactSetVerifiedRecord` whose descriptor file is `wafer/target_artifact_set.proto`; its verified build/member/KAD/used-profile evidence is the sole delivered source for those records and can participate in record-10 WCRE. Recursively reject `TargetArtifactSetDeliveryRoot`, blob locator fields, unannotated delivery evidence and any package-local `TargetArtifactSet`/`TargetArtifactMember`/quant-profile lookalike from PackageManifest. Implement `assertNoMapOrFloatingFieldReachableFrom(const google::protobuf::Descriptor *)` in the same test as a recursive descriptor walk with a visited set. Also assert every enum has a zero-valued `*_UNSPECIFIED`, projection/count expression/node-domain binding/activation are typed `oneof`s, resource role/state policy/completion/predicate kinds are enums, runtime-only IDs are absent recursively, and forbidden legacy names are reserved in their owning messages. Numeric reservations become mandatory whenever a future schema version removes a published field.

Add a lit test that imports only the generated Python binding and checks the same source descriptor:

```text
# RUN: env PYTHONPATH=%wafer_obj_root/schema/python %python -c \
# RUN:   'from wafer import package_manifest_pb2 as p; d=p.PackageManifest.DESCRIPTOR; assert d.fields_by_name["resources"].number > 0; assert "instructions" not in d.fields_by_name'
```

This Python test may inspect and print generated messages; it must not implement a package validator.

- [ ] **Step 2: Run the test and verify codegen is absent**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
```

Expected before implementation: build fails because `WaferPackageProto` and generated `package_manifest.pb.h` do not exist.

- [ ] **Step 3: Define the complete schema projection**

Use `syntax = "proto3"`, package `wafer.package.proto`, import `wafer/kernel_abi.proto`, `wafer/semantic_identity.proto` and `wafer/target_artifact_set.proto`. Reuse only the shared message options `wafer.wcre_record_type` / `wafer.wcre_schema_version` and field options `wafer.wcre_value_kind` / `wafer.wcre_collection` / `wafer.wcre_identity` / `wafer.wcre_union_discriminant`; package code must not declare another extension. Any unspecified WCRE option is a schema/codegen failure.

Annotate `PackageManifest` as WCRE record type 10, schema version 1. Mark semantic sequences `ORDERED`, semantic sets `SET`, normal fields `INCLUDED`, human diagnostics `EXCLUDED_DIAGNOSTIC`, and delivery locators `EXCLUDED_LOCATOR`. Use the shared `BOOL/U64/I64/BYTES/ASCII/RECORD/UNION` value kinds; do not create a package-specific encoder convention.

Import and reuse `semantic_identity.proto` messages and `WaferABI` decoders for every shared manifest identity; `package_manifest.proto` declares no structurally similar ID message. Import and reuse only locator-free `TargetArtifactSetVerifiedRecord` and its verified member/KAD/build/profile evidence from `target_artifact_set.proto`; package-specific records may reference members by `TargetArtifactMemberKey` and add package-owned resource/entry/completion cross-relations, but cannot copy that field registry or embed `TargetArtifactSetDeliveryRoot`. `ResourceId` fields use the closed tagged `ResourceIdentityRef = ModelBoundaryResourceId | ModelInternalResourceId | ExecutableResourceId`, with the oneof discriminant included in identity. `StateConsistencyGroupId` preserves `(ModelInterfaceSemanticId, nonzero uint32)`, `ExecutionInstanceId` preserves `(distributed program semantic ID, component, partition coordinate, replica coordinate)`, `IterationDomainId` preserves `(ExecutableVariantId, nonzero uint32)`, `ResourceRealizationRecordKey` preserves `(ResourceId, TargetVariantId, ExecutableVariantId, canonical nonempty RankClassId coverage set, canonical nonempty ProjectionSetId coverage set)`, `StreamWindowId` preserves `(ResourceRealizationRecordKey, nonzero uint32)`, `TransportActionId` preserves `(EntryId, entry-local uint64)`, and `TransportBindingMemberId` preserves `(TransportActionId, action-local uint32)`. Reuse KAD-owned `SlotId`/`CompletionExportId` and shared digest/descriptor messages. `StateNamespaceId`, `ScopeInstanceId`, `PersistentStateScopeKey`, `ResourceVersionId` and `StateGroupVersionId` are runtime-only values from `Wafer/Runtime/RuntimeIdentity.h`, have no generated-message conversion and must be absent from PackageManifest recursively; manifest carries only the declared group scope facts from which runtime derives the durable key. A target artifact member is referenced only by the tasks/14 canonical composite key `EntryId + static-function semantic digest + final module content digest`; there is no independent module identifier or other member identifier.

Define messages for:

```text
PackageManifest
  schema_version, identity_schema_version
  model_interface, resources, state_groups
  executable_variants, target_artifact_sets
  rank_classes, entries, projection_sets, transports
  completion_nodes, completion_edges, runtime_requirements

ModelInterface
  compiler-owned model-interface semantic ID
  ordered ModelEntrypointId records with canonical root, IO and state-group API contract
  optional unique diagnostic aliases excluded from identity/selection
  ordered external ports, typed shape bounds, aliases and state relations
  declared invocation-policy integer fields with finite bounds

PackageBundleIndex
  schema_version, manifest_locator
  expected_package_blob_digest, expected_package_manifest_id
  expected_target_artifact_set_ids, module_blobs, payload_blobs

ArtifactRef
  ResourceRealizationRecordKey, exact logical coverage and content size/digest
  StorageEncodingDescriptor, QuantStorageAbiProfileId refs
  ArtifactChunkingPolicyV1 + canonical fixed-4-MiB chunk table
  resident covers complete backing; streamed ranges close over declared StreamWindowId chunks

Imported TargetArtifactSetVerifiedRecord
  PackageManifest embeds the complete locator-free verified record, never TargetArtifactSetDeliveryRoot
  package-owned entry/resource/completion records reference its canonical TargetArtifactMemberKey relation
  no package-local TargetArtifactSet/TargetArtifactMember/KAD/build-profile field copy
  verified build evidence supplies the sole delivered used QuantStorageAbiProfileV1 records
  each member retains canonical usedQuantStorageProfiles from actual closure/KAD refs

Resource
  tagged ResourceIdentityRef, typed role/semantic type/access/lifetime/alias/update relation
  optional shared QuantizationDescriptor for logical low-precision semantics
  immutable ArtifactRef or persistent StateDescriptor + StateConsistencyGroupId ref
  repeated ResourceRealization

ResourceRealization
  shared ResourceRealizationRecordKey with target/executable/projection coverage sets
  committed RankClassId coverage set; no runtime singular lookup key on wire
  storage descriptor, static capacity, alignment, DdrArenaId/placement domain
  declared scope, shard/packing, shared StorageEncodingDescriptor,
    QuantStorageAbiProfileId refs and accepted range relation
  immutable oneof resident | streamed_windows residency

StreamWindow
  realization-scoped nonzero uint32 StreamWindowId
  source artifact byte range and exact chunk digest coverage
  logical tensor slice, staging ResourceId + destination range
  consumer EntryId/SlotId set and copy-completion node

StateGroup
  model-interface-scoped StateConsistencyGroupId
  canonical nonempty persistent ResourceId member set and group scope
  repeated axis-covered StateGroupRealization

StateGroupRealization
  TargetVariantId, ExecutableVariantId, RankClassId coverage, ProjectionSetId
  exact member ResourceRealization refs and oneof policy
  atomic_version + oneof full_copy | page_cow, bounded member footprints,
    snapshot-ready and group-publish completion refs
  in_place_poison_on_failure + exclusive epoch acquire,
    group publish/poison/reset completion refs

Imported StateSlotVersionRole
  `kernel_abi.proto` owns enum values none | current | candidate | in_place
  PackageManifest only references it and declares no package-local lookalike

ExecutableVariant
  ExecutableVariantId, typed ShapeGuardRef, TargetVariantId, ProjectionSetId
  typed model invocation mappings, rank/entry/completion references, EntryGraphIterationDomains
  deterministic priority/fallback

ModelInvocationMapping
  ModelEntrypointId
  initial entry/completion template roots
  required external IO and state-group contract
  terminal output/state/completion roots

EntryGraphIterationDomain
  variant-local nonzero uint32 IterationDomainId, typed kind
  recursive BoundedCountExpr and compile-time uint32 max_count

ActivationPredicate
  imported variant-scoped ActivationPredicateId
  closed bounded_count_nonzero(count ResourceId, expert ModelProgramMemberId/component relation,
    checked element offset/type/capacity/max count, count-phase-complete node)
  predicate-evaluate and conditional-join completion refs

GraphNodeActivation
  oneof always | predicate_true(ActivationPredicateId) |
    predicate_false(ActivationPredicateId)

ExpertWave
  canonical finite predicate/member/entry/module/window set
  per-domain compiler-planned capacity envelope and max simultaneously active count
  typed first-use/last-use, active-terminal, skipped-success and conditional-join refs

BoundedCountExpr
  oneof nonnegative constant | DimId actual | declared invocation-policy integer
  | typed binary add/mul/ceildiv/min/max operands

RankClass
  committed RankClassId and non-empty canonical ExecutionInstance coverage
  prerequisite-class refs and shared entry/module/KAD/resource/transport/completion refs

ExecutionInstance
  canonical ExecutionInstanceId and component/stage identity
  typed partition and replica coordinates plus dp/tp/pp/ep axis-role coordinates
  execution-mesh reference; any flat rank is excluded or diagnostic-only

Entry
  EntryId, nonempty canonical RankClassId coverage set
  canonical target artifact member composite key per final module relation
  ordered SlotBinding = SlotId -> (ResourceId, StateSlotVersionRole)
  explicit singleton/IterationDomainId node binding
  entry dependencies with iteration_delta 0/+1 and CompletionExportBinding

ProjectionSet
  oneof pinned_projection | relocatable_projection
  relocatable oneof ConcreteRecordSet | FiniteTemplateSet

CompletionNode / CompletionEdge
  node singleton/IterationDomainId binding
  typed kind/scope/status/failure-domain/timeout/state transition
  edge iteration_delta restricted to 0/+1

Transport
  EntryId + entry-local uint64 TransportActionId, rank/ProjectionSetId, logical peer/buffer/range
  repeated complete TransportBindingMember and optional RelocationSchema
  typed issue/receiver-ready/wait/status/error/release completion refs
  fixed-size transfer or segmented count/data phase with per-peer bounds

TransportBindingMember
  action-local uint32 TransportBindingMemberId, local/remote endpoint, DTE channel
  local/remote FSM, packet/stream resource class
  receiver ResourceId/realization, offset/capacity/alignment/ownership/lifetime
  status/error source and release relation

RelocationSchema
  typed slots with field kind/width, owner action/member and finite allowed values
```

Use repeated entry messages instead of proto `map`; ordered ABI slots/model ports preserve order, while set/member collections carry stable IDs and are normalized by the semantic verifier. Reserve the legacy names `instructions`, `binding_order` and `completion_source` in the corresponding messages. Do not add provider API names, file-system paths to semantic messages, free-form lifecycle strings, arbitrary scripts, floats, `Any` or lite-runtime codegen.

`TargetArtifactSetId` already is the WCRE set-root semantic digest. Manifest and delivery index carry that one typed value; they do not add a second root-digest field. The filesystem root locator remains delivery-only and identity-excluded.

- [ ] **Step 4: Add codegen and library targets**

In `schema/CMakeLists.txt` use the shared helper owned by `WaferProtoSupport`:

```cmake
wafer_add_proto_library(
  TARGET WaferPackageProto
  PROTOS schema/wafer/package_manifest.proto
  IMPORT_DIRS "${CMAKE_SOURCE_DIR}/schema"
  PYTHON_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/python"
)
target_link_libraries(WaferPackageProto PUBLIC WaferTargetArtifactSetProto)
```

Create runtime-safe `WaferPackageFormat` with an initially empty public wrapper and require exact dependency compatibility:

```cmake
add_library(WaferPackageFormat
  PackageManifest.cpp
)
target_link_libraries(WaferPackageFormat
  PUBLIC WaferABI WaferPackageProto WaferTargetArtifactSetProto
         WaferProtoSupport LLVMSupport)
wafer_require_protobuf_version("3.21.9")
```

`wafer_require_protobuf_version` and the exact helper signature above are exported by `cmake/WaferProtoSupport.cmake`; the CMake configure step must fail if its pinned protoc and runtime report any other version. Dependency checks require `WaferPackageProto -> WaferTargetArtifactSetProto`, proving imported set/member/profile messages have one generated owner. `WaferPackageFormat` must compile with no Wafer IR or MLIR include/link dependency.

- [ ] **Step 5: Build and run focused schema tests**

Run:

```bash
cmake --build build/wafer-dev --target WaferPackageProto WaferUnitTests -- -j128
build/wafer-dev/bin/WaferUnitTests \
  --gtest_filter='PackageManifestSchemaTest.*'
<configured-lit> -sv \
  build/wafer-dev/test/Tools/package-manifest-generated-binding.test
```

Expected: generated C++ and Python bindings come from the same proto graph; target-set/member descriptors resolve to `target_artifact_set.proto` rather than package-local copies; all schema-shape tests pass; no semantic verifier is claimed yet.

- [ ] **Step 6: Commit**

```bash
git add schema/wafer/package_manifest.proto schema/CMakeLists.txt \
  include/Wafer/Package/PackageManifest.h lib/Wafer/CMakeLists.txt \
  lib/Wafer/Package unittests/Package/PackageManifestSchemaTest.cpp \
  unittests/CMakeLists.txt test/Tools/package-manifest-generated-binding.test
git commit -m "Define typed package manifest schema"
```

### Task 2: Single C++ Semantic Verifier and Dual Package Identity

**Files:**
- Modify: `include/Wafer/Package/PackageManifest.h`
- Create: `lib/Wafer/Package/PackageManifestVerifier.cpp`
- Create: `lib/Wafer/Package/PackageManifestIdentity.cpp`
- Create: `lib/Wafer/Package/PackageManifestIO.cpp`
- Modify: `lib/Wafer/Package/CMakeLists.txt`
- Create: `unittests/Package/PackageManifestVerifierTest.cpp`
- Create: `unittests/Package/PackageManifestIdentityTest.cpp`
- Create: `unittests/Package/PackageManifestTestBuilder.h`
- Modify: `unittests/CMakeLists.txt`
- Modify: `tools/check_deps.py`
- Modify: `test/Tools/check-deps.test`

**Interfaces:**
- Consumes: `wafer::package::proto::PackageManifest`; WaferABI implementation-private descriptor-driven canonical core; shared record-specific semantic/content digest types. No public generic encoder/digest entry is added.
- Produces:

```cpp
namespace wafer::package {
namespace detail {
class PackageManifestSerializationAccess;
} // namespace detail

class PackageParseLimits;

namespace detail { class VerifiedPackageManifestStorage; }

class VerifiedPackageManifest {
public:
  const proto::PackageManifest &message() const;
  const abi::PackageManifestId &id() const;
  const abi::ModelInterfaceSemanticId &modelInterfaceId() const;
  llvm::ArrayRef<abi::VerifiedTargetArtifactSetRecord>
  targetArtifactSets() const;
private:
  friend llvm::Expected<VerifiedPackageManifest>
  verifyPackageManifest(std::unique_ptr<proto::PackageManifest>,
                        const PackageParseLimits &,
                        const abi::ArtifactAdmissionLimits &,
                        abi::CanonicalEncodingContext &);
  explicit VerifiedPackageManifest(
      std::shared_ptr<const detail::VerifiedPackageManifestStorage> storage);
  std::shared_ptr<const detail::VerifiedPackageManifestStorage> storage_;
};

enum class PackageRecordKind : std::uint8_t {
  ModelEntrypoint,
  ModelPort,
  ShapeBound,
  Resource,
  ResourceRealization,
  QuantStorageProfile,
  ArtifactRef,
  ArtifactChunk,
  StateGroup,
  StateGroupMember,
  StateGroupRealization,
  TargetArtifactSet,
  TargetArtifactMember,
  ModuleBlob,
  PayloadBlob,
  PayloadChunk,
  StreamWindow,
  ExecutableVariant,
  ModelInvocationMapping,
  RankClass,
  ExecutionInstance,
  Entry,
  SlotBinding,
  CompletionExportBinding,
  ProjectionSet,
  ProjectionRecord,
  ProjectionTemplate,
  CountExpressionNode,
  IterationDomain,
  ActivationPredicate,
  ExpertWave,
  TemplateGraphNode,
  TemplateGraphEdge,
  CompletionNode,
  CompletionEdge,
  Transport,
  TransportBindingMember,
  RelocationSlot,
  RuntimeRequirement,
  Count
};

enum class PackageParseScalarLimitKind : std::uint8_t {
  DeliveryBytes,
  RecursionDepth,
  ExpressionDepth,
  StringBytes,
  TotalStringBytes,
  LocatorBytes,
  TotalLocatorBytes,
  TotalRecords
};

struct PackageParseLimitValues {
  // Every value is positive; zero never means unbounded.
  std::uint64_t maxDeliveryBytes;
  std::uint32_t maxRecursionDepth;
  std::uint32_t maxExpressionDepth;
  std::uint64_t maxStringBytes;
  std::uint64_t maxTotalStringBytes;
  std::uint64_t maxLocatorBytes;
  std::uint64_t maxTotalLocatorBytes;
  std::uint64_t maxTotalRecords;
  std::array<std::uint64_t,
             static_cast<std::size_t>(PackageRecordKind::Count)>
      maxRecordsByKind;
};

class PackageParseLimits {
public:
  static llvm::Expected<PackageParseLimits>
  create(PackageParseLimitValues values);
  std::uint64_t maxDeliveryBytes() const;
  std::uint32_t maxRecursionDepth() const;
  std::uint64_t limitFor(PackageRecordKind kind) const;
  const PackageParseLimitValues &values() const;
private:
  explicit PackageParseLimits(PackageParseLimitValues values);
  PackageParseLimitValues values_;
};

class SerializedPackageManifest {
public:
  const abi::ImmutableByteBackingRef &backing() const;
  std::uint64_t byteSize() const;
  const abi::ContentDigest &blobDigest() const;
  const abi::PackageManifestId &manifestId() const;
private:
  friend class detail::PackageManifestSerializationAccess;
  SerializedPackageManifest(
      abi::ImmutableByteBackingRef backing,
      abi::ContentDigest blobDigest,
      abi::PackageManifestId manifestId);
  abi::ImmutableByteBackingRef backing_;
  abi::ContentDigest blobDigest_;
  abi::PackageManifestId manifestId_;
};

llvm::Expected<VerifiedPackageManifest>
verifyPackageManifest(std::unique_ptr<proto::PackageManifest> manifest,
                      const PackageParseLimits &limits,
                      const abi::ArtifactAdmissionLimits &artifactLimits,
                      abi::CanonicalEncodingContext &canonicalEncoding);

llvm::Expected<VerifiedPackageManifest>
parsePackageManifest(const abi::ImmutableByteBackingRef &exactBytes,
                     const abi::ContentDigest &expectedBlobDigest,
                     const abi::PackageManifestId &expectedManifestId,
                     const PackageParseLimits &limits,
                     const abi::ArtifactAdmissionLimits &artifactLimits,
                     abi::CanonicalEncodingContext &canonicalEncoding);
} // namespace wafer::package
```

- [ ] **Step 1: Add failing verifier and identity tests**

Build a valid typed message with `PackageManifestTestBuilder`, then mutate one relation per test. Required failures:

```text
duplicate/empty ResourceId
duplicate/unknown ModelEntrypointId or model-interface-local ordinal zero
invocation mapping missing/extra initial root, terminal, external IO or state-group contract
mapping root outside its executable variant or same model entrypoint contract differs across variants
static EntryId/symbol/diagnostic alias used as semantic invocation identity
logical Resource carrying flat storage/arena/capacity fields
ResourceRealization missing target/executable/projection coverage
ResourceRealization missing committed RankClassId coverage
realization tuple overlap, hole or cross-prerequisite-class sharing
selected realization missing DdrArenaId or placement-domain mismatch
role-specific immutable/state/workspace realization mismatch
missing/duplicate/unknown QuantStorageAbiProfileId or profile record/ID mismatch
logical QuantizationDescriptor and per-realization StorageEncodingDescriptor incompatibility
packed/scale/zero-point/staging/cross-entry scratch resource or slot geometry mismatch
artifact member usedQuantStorageProfiles differs from refs recomputed from its resources/KAD closure
profile target capability/environment/artifact fingerprint mismatch
RankClass with empty/duplicate ExecutionInstance coverage or inconsistent component/coordinates
opaque/default flat rank used instead of canonical component/partition/replica/axis-role mapping
invalid paged-state geometry, page-table/control owner or capacity relation
alias/update target missing or illegal
immutable realization missing/both resident and streamed_windows policy
StreamWindowId duplicate, source/chunk/logical/staging range hole or overlap
ArtifactChunkingPolicy version/4-MiB boundary/tail/digest table mismatch or window rechunk
stream window staging role/realization/consumer SlotId/copy completion mismatch
stream copy does not dominate every consumer or last consumer does not dominate reuse
state group empty/overlapping member set or persistent resource missing/multiple group refs
StateGroupRealization axis/member-realization coverage hole or mixed group scope
atomic group missing full_copy/page_cow, bounded footprint, snapshot-ready or group publish
in-place group missing exclusive epoch acquire or group poison/reset terminal
StateSlotVersionRole inconsistent across group members, policy or slot access
unspecified/unknown ResourceIdentityRef discriminant or wrong tagged ResourceId alternative
target/shape priority overlap or uncovered fallback
rank coverage hole and cross-class entry reference
duplicate singular Entry records for one EntryId instead of one coverage set
Entry RankClassId coverage overlap/hole or artifact-member coverage mismatch
zero/duplicate IterationDomainId or missing node-domain reference
BoundedCountExpr unknown operand/op, ceildiv by zero or unprovable intermediate overflow
count bounds not provably within [1,max_count] from shape-guard/policy bounds
iteration_delta outside 0/+1, zero-delta cycle or cycle without positive progress
symbolic output/state terminal coverage or iteration-scoped resource relation hole
ActivationPredicateId wrong owner/duplicate or bounded_count_nonzero has unknown/unbounded count scalar/range
count producer completion does not dominate predicate evaluate or expert/member relation is unresolved
predicate true/false paths do not both reach one conditional join
expert copy/module/entry/data node is always-active or appears on false path
combine/resource reuse is not dominated by conditional join
conditional region writes persistent state or owns a user-visible terminal
expert wave member/first-last-use/max-active/capacity envelope is incomplete or exceeded symbolically
missing/duplicate SlotId and SlotId -> (ResourceId, StateSlotVersionRole) mismatch against KAD
unknown canonical artifact member composite or descriptor semantic digest mismatch
pinned/relocatable projection both present or incomplete finite member set
incomplete TransportBindingMember endpoint/channel/FSM/receiver/status/release fields
duplicate/unknown TransportBindingMemberId
relocation slot kind/width/owner/allowed-value mismatch
projection member vector referencing an unverified transport binding member
segmented transport count/data phase, displacement, peer-total or capacity mismatch
completion cycle, unreachable output, local-drain-only terminal
missing/duplicate CompletionExportId binding
group atomic publish not dominated by all member/rank/iteration writer success
in-place group failure path without durable group poison
unknown field at any nested message
```

The builder header owns these test-only helpers so later tests use one typed fixture source:

```cpp
wafer::package::proto::PackageManifest buildValidPackageManifest();
wafer::package::VerifiedPackageManifest
buildVerifiedManifestWithResourceOrder(llvm::ArrayRef<abi::ResourceId> ids);
std::pair<llvm::SmallVector<std::uint8_t>,
          llvm::SmallVector<std::uint8_t>>
serializeKnownFieldsInDifferentRepeatedOrder();
wafer::abi::PackageManifestId
parseAndComputeManifestId(llvm::ArrayRef<std::uint8_t> bytes);
```

Identity tests must establish the distinction:

```cpp
TEST(PackageManifestIdentityTest, RepeatedSetOrderDoesNotChangeSemanticId) {
  auto [resourceA, resourceB] = buildTwoVerifiedResourceIds();
  auto left = buildVerifiedManifestWithResourceOrder({resourceA, resourceB});
  auto right = buildVerifiedManifestWithResourceOrder({resourceB, resourceA});
  EXPECT_EQ(left.id(), right.id());
}

TEST(PackageManifestIdentityTest, ExactBytesOwnIndependentBlobDigest) {
  auto [bytesA, bytesB] = serializeKnownFieldsInDifferentRepeatedOrder();
  EXPECT_NE(wafer::abi::computeContentDigest(bytesA),
            wafer::abi::computeContentDigest(bytesB));
  EXPECT_EQ(parseAndComputeManifestId(bytesA), parseAndComputeManifestId(bytesB));
}
```

Add a test that passes the digest of reserialized bytes as the expected blob digest and confirms it is rejected before parse/semantic validation.

Add a recursive descriptor-coverage test that starts at `PackageManifest`, visits every reachable repeated message field and requires exactly one `PackageRecordKind` category before the verifier may reserve it. It rejects an unclassified field, two categories for one descriptor, a category whose descriptor is unreachable, and any schema addition without an explicit limit category; activation predicates, expert waves, model entrypoints/invocation mappings and state-group realizations are mandatory coverage cases. Scalar totals remain an aggregate backstop, not a substitute for owner-qualified per-kind accounting. For every category plus depth/string/AST/total limits, instrument the generated-message arena and require limit+1/overflow failure with zero generated-message allocation before preflight; after an accepted preflight, mutate the parser/reflection result and require exact post-parse mismatch with no immutable owner or semantic index published.

Add a large-manifest stress fixture with hundreds of thousands of typed records. Through non-installed test access, instrument the sealed ABI-backing factory, allocator and identity encoder: deterministic serialization creates one shared `abi::ImmutableByteBackingRef`, re-reads exact size/digest before sealing the result, package identity uses counting plus bounded streaming hash, publisher streams that same backing, and loader retains one backing while Protobuf owns only its typed object graph. Peak extra raw-byte/WCRE memory must stay within the configured worker window rather than scaling as another 2-4 full manifest copies; changing the inline/sealed-file/CAS threshold cannot change exact bytes or IDs. Installed headers expose no output writer/factory callback.

Add an orthogonal large-rank/resource fixture whose theoretical axis product is enormous but whose declared coverage/ref sets are sparse. Instrument join comparisons and allocations; the verifier must use canonical merge joins proportional to records plus actual references and never allocate/iterate the full Cartesian product.

Add compile-time assertions that `abi::PackageManifestId`, `abi::ModelInterfaceSemanticId`, `abi::TopologySnapshotId` and `abi::KernelAbiSemanticDigest` are pairwise non-constructible, non-assignable, non-comparable and non-convertible. Also assert `PackageParseLimits`, `SerializedPackageManifest` and `abi::VerifiedTargetArtifactSetRecord` are non-aggregate/not publicly constructible, and that serialized bytes/digests/artifact records cannot be mixed by caller mutation. Add header/dependency checks that no public generic semantic-digest value type or raw-digest constructor exists. Add runtime negatives for a model-interface identity carrying the wrong record/domain, target verified record claimed-ID/member/KAD/profile mismatch, embedded delivery root/locator and a bundle index whose typed package ID bytes do not match the verified record.

- [ ] **Step 2: Run tests and verify the public API is missing**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
```

Expected before implementation: compile fails on `verifyPackageManifest`, the non-installed `detail::PackageManifestSerializationAccess` and `parsePackageManifest`.

- [ ] **Step 3: Implement recursive unknown-field rejection and semantic validation**

Before invoking generated Protobuf parsing or constructing any semantic side vector, build the fixed package descriptor/category plan and validated budget, then call WaferABI's sole non-installed schema-aware wire engine over `abi::ImmutableByteBackingRef::readAt`. That engine resolves every tag through the pinned descriptor, verifies wire type and bounded length-delimited submessage/string/bytes framing, rejects unknown/reserved fields, tracks recursion/expression depth, and checked-counts every reachable repeated-message `PackageRecordKind`, string/locator byte total, AST node and total record before any attacker-derived reserve/allocation. Packed scalar payloads are length/wire checked without materializing them. Only its backing/schema-bound `VerifiedProtoWirePreflight` may enter `CodedInputStream`; `verifyParsedProtoMatchesPreflight` requires exact per-kind/scalar/depth/span agreement before the immutable proto owner or semantic indexes are sealed. Mismatch is an internal/parser contract failure, never acceptance. Validate all IDs/use-def relations with owner-typed lookup keys, including model-interface-scoped `ModelEntrypointId`, variant-local iteration IDs, `EntryId + uint64` transport action keys and action-local member ordinals. Require every model entrypoint record to have canonical ordered IO/state/root semantics and every supporting executable variant to have exactly one mapping whose initial roots, required contract and terminal roots close within that variant; static `EntryId`, symbol and diagnostic alias never become invocation identity. The same model entrypoint's model IO/state contract must be identical across variants. First validate each committed `RankClassId` against its complete canonical `ExecutionInstance` mapping; package code never regroups instances. Require each `EntryId` to appear once with one nonempty canonical rank-class coverage set, prove coverage against variant rank mapping, and require exact equality with target artifact member coverage; do not split one entry into class-salted records. For every `EntryGraphIterationDomain`, iteratively validate `BoundedCountExpr`, use checked nonnegative arithmetic plus shape-guard/invocation-policy bounds to prove every intermediate and final count, require final `[1,max_count]`, then symbolically prove the template: delta-zero subgraph acyclic, every template cycle has positive total delta, boundary equations make every required output/state terminal iteration reachable, and iteration-scoped resource reuse/stage edges have a finite live span. Do not recurse on attacker-controlled AST depth or serialize/cache an expanded graph.

For each finite `ActivationPredicate`, decode the imported owner-scoped ID and require its canonical ordinal tuple to match the closed `bounded_count_nonzero` record. Prove the typed count resource/expert-member relation, checked element offset/type/capacity/max and count-complete owner. Count completion must dominate evaluate. Every controlled copy/module-first-use/entry/data node is on the true branch; the false branch reaches an explicit `skipped_success` terminal; both reach exactly one conditional join that dominates combine and staging/resource reuse. Conditional regions cannot write persistent state or produce a user-visible terminal. Validate canonical expert waves, member coverage, first/last use, maximum simultaneously active members and per-domain capacity envelopes without assuming all experts active or letting runtime choose a subset.

Build limit-checked canonical indexes before semantic joins. Target/executable/projection IDs come from one bound executable-variant key; for each entry slot, sweep its sorted required `RankClassId` set against sorted `ResourceRealizationRecordKey` coverage intervals/sets and require an exact nonoverlapping partition with one realization per required tuple. Never materialize a `TargetVariantId x ExecutableVariantId x RankClassId x ProjectionSetId` Cartesian product. Verify each matched realization's storage/capacity/arena/scope/shard/packing/range fields against logical resource constraints without synthesizing defaults. Large-rank/resource tests assert work and temporary allocations remain proportional to declared records plus actual refs. For each low-precision resource call the shared descriptor/profile verifiers, require its logical `QuantizationDescriptor`, selected realization `StorageEncodingDescriptor`, ordinary packed/scale/zero-point/staging/scratch resource relations and ordered KAD slot geometry/capacity/alignment to close exactly. Recompute every artifact member's duplicate-free `usedQuantStorageProfiles` from its actual resource/KAD refs and require equality with the imported member plus target capability/environment/artifact fingerprint. Runtime/package code does not evaluate quant formulas or invent repacking/native fallback. For immutable resources, require exactly one resident/streamed policy; each `ArtifactRef` must bind the exact `ResourceRealizationRecordKey`, encoding/profile refs, logical coverage, size/digest and canonical chunks, while streamed windows must have exact source/chunk/logical coverage, bounded staging destinations and graph dominance proving copy-before-consumer plus last-consumer-before-reuse. Validate each state group member set and axis-covered realization, exact member realization refs, snapshot/epoch policy, bounded update footprints and `StateSlotVersionRole` compatibility; group publish/poison terminals must cover every member writer as one consistency unit. Validate complete transport binding members, relocation slot ownership/finite allowed values and projection-member references before completion relations. Validate role, quant/storage profile, alias, target/shape, rank/entry/iteration template, KAD slot, canonical artifact member composite and completion relations in separate file-local functions; return the first stable owner-qualified diagnostic.

The sole production raw-message semantic entry is move-owning `verifyPackageManifest(std::unique_ptr<proto::PackageManifest>, const PackageParseLimits &, const abi::ArtifactAdmissionLimits &, abi::CanonicalEncodingContext &)`. It rejects null and never exposes the object after transfer. Adopt through `abi::ImmutableProtoMessageOwner::adopt(std::unique_ptr<google::protobuf::Message>)`; `VerifiedPackageManifestStorage` retains that owner, so assembler/parser never keep both a parsed/local manifest and copied graph. For each embedded locator-free target record, build a bounded `ProtoSubmessagePathStep::singular(fieldNo)`/`repeated(fieldNo,index)` path from descriptors, then call `resolveTargetArtifactSetVerifiedRecord(owner, path, artifactLimits)` to obtain typed `ImmutableTargetArtifactSetRecordRef`; there is no raw pointer plus arbitrary owner overload. Call `verifyTargetArtifactSetVerifiedRecordView(ref, artifactLimits, canonicalEncoding)`. The proof retains shared owner/path, bounded typed indexes and `CanonicalRecordBackingRef`, not another generated record or deterministic-byte copy. Standalone target parsing uses the same owner/root-path model. The package semantic core uses const views for cross-relations and never reconstructs fields or depends on WaferArtifact; Task 3's production reader adds only the separate operational session/lease dependency. The same required encoding context verifies embedded record 9 views and record 10, enforcing one limit/scratch budget. It constructs `VerifiedPackageManifest` only after every check passes and keeps its constructor private; no production const-reference verifier, raw owner pairing, default context or generic digest API exists. A clearly named test-only helper may explicitly clone a fixture before move-adoption.

Add `PackageManifestVerifier.cpp`, `PackageManifestIdentity.cpp` and `PackageManifestIO.cpp` only to `WaferPackageFormat`. At this pure manifest milestone its transitive links remain `WaferABI`, generated Protobuf, `WaferProtoSupport` and LLVM Support; Task 3 later adds the one allowed runtime-neutral `WaferArtifact` dependency when the production session-bound bundle reader lands. A CMake test must always reject `WaferIR`, `WaferTargetArtifacts`, `WaferPackageCompiler` or any `MLIR*` entry in this target's link interface and, before Task 3, also proves that the semantic core itself does not require WaferArtifact.

- [ ] **Step 4: Implement WCRE record type 10 projection**

Call the shared descriptor-driven encoder with `WCRERecordType::PackageManifest`; field options classify semantic sequences/sets and diagnostic/delivery exclusions. Ordered model entrypoints/invocation mappings, resource/quant/storage/artifact/state facts, finite iteration domains/count ASTs/node bindings/edge deltas, activation predicates/branch annotations/expert waves/capacity envelopes, transport members, relocation values and projection sets are included typed semantic collections, not derived runtime data. Diagnostic aliases remain excluded. The shared encoder reuses field numbers, encodes present optionals only, preserves semantic order, sorts sets by element WCRE bytes and rejects unknown/unregistered fields.

Compute exactly:

```cpp
auto measured = abi::detail::measureIdentityMessage(
    canonicalEncoding, manifest, abi::WCRERecordType::PackageManifest);
if (!measured)
  return measured.takeError();
auto hasher = detail::PackageManifestIdentityAccess::beginVerifiedRecord(
    *measured, abi::WCREDomain::PackageManifest);
if (auto error = abi::detail::streamIdentityMessage(
        canonicalEncoding, manifest, abi::WCRERecordType::PackageManifest,
        [&](llvm::ArrayRef<std::uint8_t> bytes) {
          return hasher.update(bytes);
        }))
  return std::move(error);
return hasher.finishPackageManifestId();
```

`abi::detail::measureIdentityMessage`/`streamIdentityMessage`和`detail::PackageManifestIdentityAccess`只通过WaferABI private implementation对`PackageManifestIdentity.cpp`可见。Public `include/Wafer/ABI/CanonicalEncoding.h` exposes only move-only `CanonicalEncodingContext` and its validated factory. The first pass computes checked length; the second streams through context-owned bounded scratch/backing into the fixed-domain hasher, so a large package never materializes another full record. The context checks measured/emitted bytes and cumulative nested record/scratch budgets; `CanonicalRecordBackingRef` keeps any collision evidence in the immutable scratch store. `verifyPackageManifest` directly returns strong `PackageManifestId`; generic digest, caller domain, separate scratch/limit or unbounded/default context cannot cross the boundary. Exclude diagnostic aliases, labels, locators, exact blob digest and claimed manifest ID. Reject non-ASCII identity text and unimplemented schema versions.

- [ ] **Step 5: Implement exact-byte parse and deterministic writer**

`parsePackageManifest` must execute in this order:

```cpp
if (exactBytes.exactSize() > limits.maxDeliveryBytes())
  return makePackageError("parse_limit_exceeded", "delivery_bytes");
auto actualBlobDigest = computeBackingContentDigest(exactBytes);
if (!actualBlobDigest || *actualBlobDigest != expectedBlobDigest)
  return makePackageError("package_blob_digest_mismatch");
auto wireLedger = preflightPackageManifestWire(exactBytes, limits);
if (!wireLedger)
  return wireLedger.takeError();
auto parsed = parseBackingWithCodedInputLimits(exactBytes, limits, *wireLedger);
if (!parsed)
  return parsed.takeError(); // preserves parse_limit_exceeded vs malformed wire
if (auto error = abi::detail::verifyParsedProtoMatchesPreflight(
        **parsed, *wireLedger))
  return std::move(error);
auto verified = verifyPackageManifest(std::move(*parsed), limits,
                                      artifactLimits, canonicalEncoding);
if (!verified)
  return verified.takeError();
if (verified->id() != expectedManifestId)
  return makePackageError("package_manifest_id_mismatch");
return verified;
```

`preflightPackageManifestWire` and `parseBackingWithCodedInputLimits` read only through the owner-backed `abi::ImmutableByteBackingRef` without copying or retaining caller raw views. The package wrapper constructs one fixed `abi::detail::ProtoWireAdmissionPlan` from the generated root descriptor plus the exact `PackageRecordKind` registry and one matching `ProtoWireAdmissionBudget` from validated package/artifact limits, then delegates the wire walk solely to non-installed WaferABI `detail::preflightProtoWire`. It does not implement another tag parser. The returned `VerifiedProtoWirePreflight` binds backing digest/size, schema/plan identity, counts and spans. Only then may `CodedInputStream` allocate within the accepted ledger and return `std::unique_ptr<proto::PackageManifest>`; `detail::verifyParsedProtoMatchesPreflight` must match the generated message exactly before the immutable proto owner or any semantic index is sealed. Size/count overflow has its own `size_overflow` error and is never folded into ordinary parse failure. Production file/capability sources create the ref only after bounded same-handle stat/read/hash/stat; the only inline unit/debug adapter is capped by `ArtifactAdmissionLimits.maxInlineRecordBytes`. The runtime-private verification-session adapter lends one retained canonical context to parse and every nested verifier. The unique message is moved exactly once into immutable owner/storage; after the move the parser retains no second typed graph or raw byte pointer. Deterministic serialization first counts exact bytes, then non-installed `detail::PackageManifestSerializationAccess` asks the `PackageArtifactBuildSession`'s sealed output-backing factory for exactly that size and streams Protobuf bytes plus `PackageBlobDigest` through one bounded buffer. `finish` must return an `abi::ImmutableByteBackingRef` for the same output object; the access layer rechecks exact size/content digest by `readAt`/`writeTo` before privately constructing `SerializedPackageManifest`. No public virtual writer/factory or callback can mint the result. The compiler factory uses bounded inline storage only below its publication limit and a private sealed file/CAS backing above it; bundle loading holds root-bound ABI backings for index/manifest. `PackageAssemblyResult`, stager and loaded package share immutable owners rather than copying bytes. Callers cannot pair bytes/IDs/record backing from different contexts. Debug JSON is emitted only from `VerifiedPackageManifest`; no JSON parse branch is added.

- [ ] **Step 6: Run focused verifier/identity tests**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
build/wafer-dev/bin/WaferUnitTests \
  --gtest_filter='PackageManifestVerifierTest.*:PackageManifestIdentityTest.*'
python3 tools/check_deps.py
```

Expected: all positive/negative tests pass; different exact bytes can share one semantic ID, while any wrong exact blob digest fails before semantic acceptance.

- [ ] **Step 7: Commit**

```bash
git add include/Wafer/Package/PackageManifest.h lib/Wafer/Package \
  unittests/Package unittests/CMakeLists.txt tools/check_deps.py \
  test/Tools/check-deps.test
git commit -m "Add single package semantic verifier"
```

### Task 3: Compiler-Only Assembly, Transactional Bundle Staging, and Trusted Loading

**Files:**
- Create: `include/Wafer/Package/PackageAssembly.h`
- Create: `include/Wafer/Package/PackageBundle.h`
- Create: `lib/Wafer/Package/PackageAssembly.cpp`
- Create: `lib/Wafer/Package/PackageBundleReader.cpp`
- Create: `lib/Wafer/Package/PackageBundleWriter.cpp`
- Create: `lib/Wafer/Package/PackageProgramOutputAdapter.cpp`
- Modify: `lib/Wafer/Package/CMakeLists.txt`
- Create: `unittests/Package/PackageAssemblyTest.cpp`
- Create: `unittests/Package/PackageBundleTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Modify: `tools/check_deps.py`
- Modify: `test/Tools/check-deps.test`

**Interfaces:**
- Consumes: one outer `compiler::ProgramOutputTransaction` created in Task 0 with retained `PackageBundle` completion scope, verified sink and `ProgramOutputLimits`, and already owning exactly one `compiler::StagedExecutableToken` plus the complete canonical `compiler::StagedTargetArtifactSetToken` set; the latest non-forgeable `compiler::PackageAssemblyInputView` returned only by `compiler::detail::PackageProgramOutputAccess::buildPackageAssemblyInputView`; validated build-service `PackagePublicationLimits`/`PackageParseLimits`/`ArtifactAdmissionLimits`. The sole compiler-private factory `compiler::detail::PackageProgramOutputAccess::beginPackageArtifactBuild` binds that view's exact attachment-table generation, stable transaction owner/session, staging/backing/attachment capabilities, budgets and cancellation into one move-only `package::PackageArtifactBuildSession`. Assembly, staging and V2 conversion accept only this session/result path; no caller-owned raw `ExecutableOp`, target array, canonical context, backing/root/path, callback/custom reader or replacement outer limit/sink/scope is accepted. Production metadata loading consumes target-plan-owned runtime-neutral `artifact::ArtifactMetadataVerificationSession`, created from the HostVerificationRegistry's distinct Artifact-metadata sub-capability plus validated artifact limits/context. It parses only index/manifest/embedded target records and constructs inert source descriptors; it cannot accept or synthesize `RuntimeArtifactVerificationSession`, device authority or an openable blob source.
- Produces:

```cpp
namespace wafer::runtime::detail {
class RuntimePackageBindingAccess;
} // namespace wafer::runtime::detail

namespace wafer::package {
struct PackagePublicationLimitValues {
  std::uint64_t maxBlobCount;
  std::uint64_t maxModuleBytes;
  std::uint64_t maxTotalModuleBytes;
  std::uint64_t maxPayloadBytes;
  std::uint64_t maxTotalPayloadBytes;
  std::uint64_t maxChunkCount;
  std::uint64_t maxManifestBytes;
  std::uint64_t maxIndexBytes;
  std::uint64_t maxLogicalOutputBytes;
  std::uint64_t maxPhysicalAllocatedOrCopiedBytes;
  std::uint64_t maxStagingOutputBytes;
  std::uint64_t maxSimultaneousSourceReaders;
  std::uint64_t maxOpenFileDescriptors;
  std::uint64_t maxWorkerReadBufferBytes;
  std::uint64_t maxConcurrentCopiedBytes;
};

class PackagePublicationLimits {
public:
  static llvm::Expected<PackagePublicationLimits>
  create(PackagePublicationLimitValues values);
  const PackagePublicationLimitValues &values() const;
private:
  explicit PackagePublicationLimits(PackagePublicationLimitValues values);
  PackagePublicationLimitValues values_;
};

struct PackagePublicationSummary {
  std::uint64_t blobCount;
  std::uint64_t largestModuleBytes;
  std::uint64_t totalModuleBytes;
  std::uint64_t largestPayloadBytes;
  std::uint64_t totalPayloadBytes;
  std::uint64_t chunkCount;
  std::uint64_t manifestBytes;
  std::uint64_t indexBytes;
  std::uint64_t logicalOutputBytes;
  std::uint64_t physicalAllocatedOrCopiedBytes;
  std::uint64_t stagingOutputBytes;
  std::uint64_t simultaneousSourceReaders;
  std::uint64_t openFileDescriptors;
  std::uint64_t workerReadBufferBytes;
  std::uint64_t concurrentCopiedBytes;
};

namespace detail {
class PackageArtifactBuildSessionStorage;
class PackageAssemblyResultStorage;
class BoundArtifactBlobSource;
} // namespace detail

class PackageAssemblyResult;

class PackageArtifactBuildSession {
public:
  PackageArtifactBuildSession(PackageArtifactBuildSession &&) noexcept;
  PackageArtifactBuildSession &operator=(
      PackageArtifactBuildSession &&) noexcept;
  ~PackageArtifactBuildSession();
private:
  friend class compiler::detail::PackageProgramOutputAccess;
  friend mlir::FailureOr<PackageAssemblyResult>
  assemblePackage(PackageArtifactBuildSession &,
                  const PackageParseLimits &,
                  const abi::ArtifactAdmissionLimits &);
  friend mlir::FailureOr<compiler::StagedPackageBundleToken>
  stagePackageBundle(PackageArtifactBuildSession &&,
                     PackageAssemblyResult &&);
  explicit PackageArtifactBuildSession(
      std::unique_ptr<detail::PackageArtifactBuildSessionStorage> storage);
  std::unique_ptr<detail::PackageArtifactBuildSessionStorage> storage_;
};

} // namespace wafer::package

namespace wafer::compiler::detail {
class PackageProgramOutputAccess final {
public:
  static llvm::Expected<PackageAssemblyInputView>
  buildPackageAssemblyInputView(
      ProgramOutputTransaction &output,
      const StagedExecutableToken &executable);
  static llvm::Expected<package::PackageArtifactBuildSession>
  beginPackageArtifactBuild(ProgramOutputTransaction &output,
                            PackageAssemblyInputView input,
                            package::PackagePublicationLimits limits);
private:
  PackageProgramOutputAccess() = delete;
};
} // namespace wafer::compiler::detail

namespace wafer::package {

enum class BlobAttachmentModeV1 : std::uint8_t {
  CasReference,
  Reflink,
  ProtectedHardlink,
  BoundedStreamCopy
};

class VerifiedBlobAttachment {
public:
  BlobAttachmentModeV1 mode() const;
  const abi::ContentDigest &contentDigest() const;
  std::uint64_t byteSize() const;
  llvm::StringRef bundleRelativeLocator() const;
private:
  friend class compiler::ProgramOutputTransaction;
  class Storage;
  explicit VerifiedBlobAttachment(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class PackageAssemblyResult {
public:
  PackageAssemblyResult(PackageAssemblyResult &&) noexcept;
  PackageAssemblyResult &operator=(PackageAssemblyResult &&) noexcept;
  ~PackageAssemblyResult();
  const VerifiedPackageManifest &manifest() const;
  const SerializedPackageManifest &serializedManifest() const;
  const proto::PackageBundleIndex &bundleIndex() const;
  const PackagePublicationSummary &publicationSummary() const;
private:
  friend mlir::FailureOr<PackageAssemblyResult>
  assemblePackage(PackageArtifactBuildSession &,
                  const PackageParseLimits &,
                  const abi::ArtifactAdmissionLimits &);
  friend mlir::FailureOr<compiler::StagedPackageBundleToken>
  stagePackageBundle(PackageArtifactBuildSession &&,
                     PackageAssemblyResult &&);
  explicit PackageAssemblyResult(
      std::unique_ptr<detail::PackageAssemblyResultStorage> storage);
  std::unique_ptr<detail::PackageAssemblyResultStorage> storage_;
};

struct BlobChunk {
  std::uint64_t offset;
  std::uint64_t size;
  abi::ContentDigest digest;
};

struct BlobRange {
  std::uint64_t offset;
  std::uint64_t size;
};

namespace detail {
class BundleRootCapability;
class UnboundBlobSourceStorage;
} // namespace detail

class LoadedPackageMetadata;
class VerifiedPackageBundleReference;

class UnboundBlobSourceDescriptor final {
public:
  const abi::ContentDigest &expectedDigest() const;
  std::uint64_t expectedSize() const;
  llvm::ArrayRef<BlobChunk> chunks() const;
private:
  friend llvm::Expected<LoadedPackageMetadata>
  loadPackageBundleMetadata(const VerifiedPackageBundleReference &,
                      const PackageParseLimits &,
                      artifact::ArtifactMetadataVerificationSession &);
  friend class runtime::detail::RuntimePackageBindingAccess;
  explicit UnboundBlobSourceDescriptor(
      std::shared_ptr<const detail::UnboundBlobSourceStorage> storage);
  std::shared_ptr<const detail::UnboundBlobSourceStorage> storage_;
};

class VerifiedPayloadBlob {
public:
  const abi::ContentDigest &digest() const;
  std::uint64_t byteSize() const;
  const UnboundBlobSourceDescriptor &source() const;
private:
  friend class LoadedPackageMetadata;
  friend llvm::Expected<LoadedPackageMetadata>
  loadPackageBundleMetadata(const VerifiedPackageBundleReference &,
                      const PackageParseLimits &,
                      artifact::ArtifactMetadataVerificationSession &);
  explicit VerifiedPayloadBlob(
      std::shared_ptr<const UnboundBlobSourceDescriptor> source);
  std::shared_ptr<const UnboundBlobSourceDescriptor> source_;
};

mlir::FailureOr<PackageAssemblyResult>
assemblePackage(PackageArtifactBuildSession &session,
                const PackageParseLimits &buildServiceLimits,
                const abi::ArtifactAdmissionLimits &artifactLimits);

mlir::FailureOr<compiler::StagedPackageBundleToken> stagePackageBundle(
    PackageArtifactBuildSession &&session,
    PackageAssemblyResult &&package);

class VerifiedModuleBlob {
public:
  const abi::ContentDigest &contentDigest() const;
  llvm::ArrayRef<abi::TargetArtifactMemberKey> memberKeys() const;
  const abi::VerifiedTargetArtifactModuleView &targetModuleView() const;
  std::uint64_t byteSize() const;
  const UnboundBlobSourceDescriptor &source() const;
private:
  friend class LoadedPackageMetadata;
  friend llvm::Expected<LoadedPackageMetadata>
  loadPackageBundleMetadata(const VerifiedPackageBundleReference &,
                      const PackageParseLimits &,
                      artifact::ArtifactMetadataVerificationSession &);
  VerifiedModuleBlob(llvm::SmallVector<abi::TargetArtifactMemberKey> memberKeys,
                     std::shared_ptr<const UnboundBlobSourceDescriptor> source);
  llvm::SmallVector<abi::TargetArtifactMemberKey> memberKeys_;
  std::shared_ptr<const UnboundBlobSourceDescriptor> source_;
};

class LoadedPackageMetadata final {
public:
  LoadedPackageMetadata(const LoadedPackageMetadata &) noexcept = default;
  LoadedPackageMetadata &operator=(
      const LoadedPackageMetadata &) noexcept = default;
  LoadedPackageMetadata(LoadedPackageMetadata &&) noexcept;
  LoadedPackageMetadata &operator=(LoadedPackageMetadata &&) noexcept;
  ~LoadedPackageMetadata();
  const abi::ImmutableByteBackingRef &exactIndexBacking() const;
  const abi::ImmutableByteBackingRef &exactManifestBacking() const;
  const VerifiedPackageManifest &manifest() const;
  llvm::ArrayRef<VerifiedModuleBlob> modules() const;
  llvm::ArrayRef<VerifiedPayloadBlob> payloads() const;
private:
  friend llvm::Expected<LoadedPackageMetadata>
  loadPackageBundleMetadata(const VerifiedPackageBundleReference &,
                      const PackageParseLimits &,
                      artifact::ArtifactMetadataVerificationSession &);
  friend class runtime::detail::RuntimePackageBindingAccess;
  class Storage;
  explicit LoadedPackageMetadata(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

llvm::Expected<LoadedPackageMetadata>
loadPackageBundleMetadata(
    const VerifiedPackageBundleReference &reference,
    const PackageParseLimits &limits,
    artifact::ArtifactMetadataVerificationSession &verification);
} // namespace wafer::package

```

`compiler::detail::PackageProgramOutputAccess::buildPackageAssemblyInputView(output, executable)` scans the lower attachment snapshot and joins the sealed executable's complete typed target requirements through each owner-specific non-installed access, requires every required `compiler::StagedTargetArtifactSetToken` exactly once and rejects extra/duplicate attachments. `ProgramOutputTransaction` itself has no public upper-view factory and its installed header never names package types. The returned move-only/non-aggregate `compiler::PackageAssemblyInputView` binds transaction identity and exact attachment-table generation and exposes only read-only committed projection, locator-free verified records and capability-bound module/payload sources through private assembler access. It has no mutable `ExecutableOp`, raw staging path, target-array constructor or reorder/subset operation. `compiler::detail::PackageProgramOutputAccess::beginPackageArtifactBuild(output, std::move(latestView), publicationLimits)` is the only way to turn that proof into a `package::PackageArtifactBuildSession`; it privately borrows the transaction canonical session, binds the same owner token/generation and installs the verified staging, immutable-backing and attachment factories. Any target/package attach advances the table generation and invalidates an older view, live package session and its result; cross-transaction token/view/session use, post-attach mutation and use after commit are rejected before package projection/source open.

The session/view internally owns private-construction `detail::BoundArtifactBlobSource` values for every committed payload, each binding the exact source root capability/generation, content digest/size and canonical chunk table. Only package-writer friend access can enumerate them and acquire a move-only same-handle lease; there is no installed/public `ArtifactBlobReader`, arbitrary callback or source factory. `PackageAssemblyResult` retains the stable canonical owner token and exact attachment generation from its session, so `stagePackageBundle(std::move(session), std::move(result))` cannot pair output built by another transaction, context, backing store or snapshot.

`stagePackageBundle` lets `compiler::detail::PackageProgramOutputAccess` convert the complete package-owned staged result into its distinct lower attachment commitment and invoke the transaction private core, returning only `compiler::StagedPackageBundleToken`. There is no public `ProgramOutputTransaction::attachPackageBundle` method. Task 0 already bound `ProgramDeliveryCompletionScope::PackageBundle`, the move-owned `VerifiedProgramOutputSink` and `ProgramOutputLimits` at transaction creation; final `commit()` accepts no replacement parameters and returns `PublishedProgramDelivery`. Production full-package flow therefore cannot publish a smaller scope first or downgrade after a package/target failure. The verified sink admits only one-filesystem unified-root rename or trusted-namespace single create/CAS backends. `TrustedProgramDeliveryRef` has no public raw constructor and externally binds root capability acquisition, commit-record locator, exact commit bytes and expected scope. `loadAndVerifyProgramDelivery` applies byte/recursion/string/locator/target/root/total limits before reserve, rejects unknown fields/version/unsafe or incomplete scope, and returns the only verified proof from which the two type-specific private access classes can derive their noninterchangeable child references. Diagnostic accept-any delivery records cannot derive either child type.

`TargetArtifactMemberKey` is the typed composite `(EntryId, static-function semantic digest, final module content digest)` from tasks/14. `VerifiedModuleBlob` is module-level: all member keys sharing one final digest reference one source and load once, so a multi-entry/shared module is not duplicated per entry. `contentDigest()` returns the source's expected digest; every member key must carry that same final digest. The verifier rejects disagreement rather than storing independent identity fields. There is no independent module identifier.

`VerifiedPackageBundleReference` has no public raw path/digest constructor. The type-specific ProgramDelivery adapter constructs it only after validating the outer commit and binds root capability acquisition, externally expected bundle-index exact size/digest, manifest exact size/digest and `PackageManifestId`; none of those values is accepted from the root being opened. `LoadedPackageMetadata`, `UnboundBlobSourceDescriptor`, `VerifiedModuleBlob` and `VerifiedPayloadBlob` have no public unchecked constructor, mutator, open/read/acquire method or conversion to a runtime source. `loadPackageBundleMetadata(reference, parseLimits, artifactMetadataSession)` uses only WaferArtifact's runtime-neutral metadata session: its typed sub-capability and local limits jointly reserve against the shared `HostVerificationRegistry`, then same-handle open/hash/preflight/parse only the externally committed index and manifest. It creates owner-backed `abi::ImmutableByteBackingRef`s for those exact bytes and seals immutable metadata/host provenance plus the reference-derived root capability in shared const storage. A copied handle shares those already verified bytes/descriptors without copying a live session, lease, ledger counter or open authority; destroying the metadata session after load does not invalidate metadata. It never creates a service context, requests device authority, opens a module/payload or constructs `RuntimeArtifactVerificationSession`.

Each `UnboundBlobSourceDescriptor` is inert metadata: expected digest/size/chunks plus a private validated root-relative locator token tied to the loaded metadata owner. It neither carries an open authority nor can form an `OpenedBlobLease`; even a caller holding a valid `LoadedPackageMetadata` cannot spend an FD. Task 5 may inspect these facts for pure selection/liveness only. After exact domains are proved, Task 6's sole runtime binding access retains one by-value metadata handle, revalidates index/manifest backing provenance and joins every descriptor to a separately created exact-domain runtime artifact session before producing `RuntimeBoundPackage` and private `BoundBlobSource`s. Multiple invocation plans may concurrently bind distinct runtime sessions from copies of the same immutable metadata without reparsing or sharing runtime leases. Each bind is one-way and has no unbind/downgrade path. Compiler `detail::BoundArtifactBlobSource` and diagnostic inspection sources/sessions remain disjoint. Any optional accept-any inspector lives in a non-installed tool-only header and returns `InspectedPackageBundle` with no conversion, accessor or friendship that can create metadata/runtime production types.

- [ ] **Step 1: Write transaction and source-of-truth failures**

Parse committed executable fixtures and inject these failures:

```text
candidate commit_state reaches assembly
raw/caller-mutated ExecutableOp or caller target-set array has no assembly overload
cross-transaction or stale compiler::PackageAssemblyInputView reaches assembly/staging
cross-transaction/stale PackageArtifactBuildSession or result-token pairing
post-view target/package attachment or sealed executable generation mismatch
artifact set source executable digest differs
artifact set omits one target variant/rank-class member
one EntryId is duplicated into singular-class records or its coverage differs from artifact member
entry function digest/module/KAD relation differs
SlotId -> (ResourceId, StateSlotVersionRole) is missing, duplicated or reordered
CompletionExportId has no executable DAG owner
resource/projection/transport reference is unresolved
state group/member realization/version-role relation is unresolved
stream window source/staging/consumer/completion relation is unresolved
ArtifactRef misses/widens its ResourceRealizationRecordKey, encoding/profile or logical coverage
low-precision resource/profile/KAD refs disagree with artifact member usedQuantStorageProfiles
caller callback/custom blob reader is offered as an assembly or staging source
bound payload source attempts to repack or substitute equivalent logical values with different exact bytes
payload bytes differ from committed content digest
bound source lease emits more/fewer than exact declared bytes or violates canonical chunk coverage
same source bytes under different reader buffer/worker count change chunk table or output digest
one member copy fails after earlier members entered staging
same-store CAS reference lacks independent destination lifetime/refcount
reflink lacks COW independence or destination read-back attestation
ordinary writable hardlink/read-only chmod is presented as protected_hardlink
protected hardlink seal/source generation changes before outer commit
cross-filesystem attachment attempts anything except bounded same-lease stream copy
100GB-class logical payload exceeds logical limit despite zero-copy physical dedup
stream fallback physical bytes/staging/FD/buffer exceeds physical publication or outer limit
attachment mode changes ArtifactRef/chunk table/manifest ID or bundle locator semantics
GC/refcount race drops a source/destination lifetime still reachable from staged/verified delivery
second target-set staging fails after the first private root is complete
package attach fails after every target-set attachment is complete
crash after immutable roots/fsync but before the outer commit record
outer commit-record CAS loses or its completion scope omits one required root
whole bundle root is replaced with another internally self-consistent bundle
old/new bundle or cross-tenant expected PackageManifestId is swapped
bundle index or manifest self-reports expected values different from the deployment reference
diagnostic accept-any result is passed to production preflight/session construction
zero/overflowing PackagePublicationLimits or manifest attempt to override them
blob/chunk/single/total module/payload/manifest/index/staging byte preflight exceeds publication limit
logical output or actual newly allocated/copied physical byte preflight exceeds its distinct limit
source-reader/FD/worker-buffer/concurrent-copy peak exceeds publication limit
actual publication count/size differs from the checked preflight summary
locator replacement after load attempts to redirect a verified module/payload
opened payload is truncated, mutated or returns early EOF while streaming
thousands of unselected blob locators under low RLIMIT_NOFILE
delivery byte/recursion/string/locator/per-record/total-count limit exceeded
deep expression/graph nesting and checked aggregate counter overflow
absolute/parent/symlink/non-regular locator or beneath-root escape
concurrent first acquire, two positioned readers, partial-reader cancel and bounded-cache eviction
```

For every publication-limit/preflight failure assert the input executable generic text is unchanged, no package child root is created, no source blob opens and no worker starts. Later staging/attach failures remove or orphan only transaction-private immutable roots, mark the outer transaction uncommittable and expose no program/package alias, index or reference. A failure in a second target set, package-last staging, fsync, outer commit or commit-record CAS cannot leave an accepted partial delivery. A crash before the single outer commit may leave GC-able immutable roots but the production loader must reject them as unreachable. For metadata loader/reference/descriptor failures assert no non-index/manifest blob is opened and no runtime source/module/payload lease is created. Add compile-time assertions that `PackageArtifactBuildSession`, `PackagePublicationLimits`, `PackageAssemblyResult`, `VerifiedPackageBundleReference`, `UnboundBlobSourceDescriptor` and tool-only diagnostic `InspectedPackageBundle` are not publicly constructible or aggregate and are not mutually convertible; `LoadedPackageMetadata` alone is an immutable copyable owner-backed handle with private construction. Installed package headers expose no `ArtifactBlobReader`, source acquire/open/read/stream method or custom source callback. Repeatedly retrieving or copying descriptors/metadata must spend zero additional FD/reader budget. Destroy the creating metadata session, then concurrently run two pure plans and two exact-domain binds from copies of the same metadata; assert no reparse, no shared runtime lease/session owner and independent cleanup. Concurrent package/target/migration metadata loads with individually sufficient typed limits must still fit the common parent `HostVerificationRegistry`; cross-session owner use, cancellation and parse failure restore the Artifact-metadata typed subledger and parent ledger exactly once. Compiler and inspection descriptors/sessions cannot convert to production metadata/runtime sources. The stager must reject any owner/generation or internal manifest/serialized/index/summary identity disagreement instead of trusting a tuple. Add a positive test that changes debug names and artifact locators while keeping typed semantic facts/content fixed; `PackageManifestId` must remain equal. Show that raising valid publication/parse limits accepts the same complete deterministic output/semantic ID, while the manifest cannot override a lower limit and no limit may omit members or change chunking. Under a low FD limit, load metadata for a bundle with thousands of blobs and assert only the committed root plus bounded index/manifest handles are active and every module/payload remains inert. Whole-root substitution, internally valid old/new replacement and tenant swap must fail against external index/manifest commitments before descriptors are sealed; the tool-only inspector may describe them but its result cannot enter production preflight. Symlink/replace/truncate/mutation and concurrent same-handle runtime acquisition move to Task 6 after one-way binding, where actual open authority first exists.

Add a second positive with two `TargetVariantId`s and two complete artifact sets; assembly must preserve the orthogonal target axis and reject either one missing, duplicated or associated with the wrong source executable digest. Add a third positive where one `EntryId` covers two committed rank classes and one artifact member relation; package assembly must emit one entry record with the canonical two-class set and preserve both execution-instance mappings.

Add a 100GB-class sparse/fake-CAS fixture that measures logical bytes without allocating them. Same-store `CasReference`, verified reflink and sealed protected-hardlink cases must avoid full read/write amplification, obtain a destination/root lifetime independent of the source transaction and remain valid across source GC. A writable hardlink, chmod-only pseudo-seal, source generation mutation, missing destination attestation or refcount/GC race fails and makes the outer transaction uncommittable. Cross-filesystem/untrusted stores must choose `BoundedStreamCopy`, read from one already-open verified source lease, recheck canonical chunks/full digest and stay within physical/staging/FD/buffer limits. All four modes produce identical manifest ID, `ArtifactRef`, chunk table and bundle-relative locator contract. Assert `PackagePublicationSummary` and retained `ProgramOutputLimits` account every logical ref while charging physical bytes only for newly allocated/copied storage; a mode cannot be selected to hide either budget.

- [ ] **Step 2: Run tests and verify assembly is unavailable**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
```

Expected before implementation: compile fails because `assemblePackage`, `stagePackageBundle` and trusted-reference loading are undefined.

- [ ] **Step 3: Project committed executable without recovery**

Walk only the verified projection sealed into `package::PackageArtifactBuildSession` from its exact `compiler::PackageAssemblyInputView`: committed invocation/variant/resource/state/rank/entry/transport/activation owners and their model interface, shape, finite iteration, projection and completion records. Copy each ordered typed model entrypoint/mapping and typed resource/graph facts: quant/storage/profile refs, exact artifact/chunk/window records, state groups/version roles, bounded count ASTs, node-domain deltas, `ActivationPredicateId` bounded count records, node true/false activation, conditional joins and finite expert-wave envelopes. Require fixed 4 MiB chunks; never recover an entrypoint/expert from name/alias/static ID or pre-expand instances. Copy each session-owned locator-free target record through the generated type, then cross-check profiles/member keys against committed entry/resource/KAD/activation closure. Never copy delivery roots; module locators are rebuilt only in the bundle index. The view already proves complete target coverage, so there is no caller target array. Do not inspect LLVM text, names, filenames, parameter counts, instruction bodies or staging paths, and do not derive/repack bytes.

Insert every semantic set in canonical raw-ID/WCRE order and every semantic sequence in its committed order so the pinned C++ producer emits reproducible deterministic bytes without claiming those bytes are a cross-producer canonical encoding.

Target build/readback attachment and `compiler::detail::PackageProgramOutputAccess::buildPackageAssemblyInputView` have already used `TargetArtifactBuildSession` plus the compiler-only executable relation verifier while the sealed executable proof was available; they retain one non-forgeable exact relation proof per target set. Package assembly only consumes those same-transaction proofs and never receives a raw `ExecutableOp`, delivered root or re-runs a relation API that needs one. Separately, standalone runtime/transform tooling may obtain `artifact::TrustedTargetArtifactSetDeliveryRef` only from a verified ProgramDelivery target root and call `loadAndVerifyTargetArtifactSetMetadata(delivery, artifactMetadataSession)`. That path verifies only delivery/index/locator-free records/KAD metadata and returns immutable `LoadedTargetArtifactSetMetadata` plus inert `UnboundTargetModuleSourceDescriptor`; it cannot open actual modules, attach into a compiler transaction or construct MLIR/runtime proofs. After an exact-domain plan/context exists, `RuntimeTargetArtifactSetsBinder` atomically binds the canonical all-required metadata vector with one context-bound runtime artifact session into `RuntimeBoundTargetArtifactSets`; there is no public single-set binder or partial result. A package index/module source can never construct the target ref; package runtime instead uses its embedded owner-backed module view plus package-root descriptor. There is no raw path/digest/size or independent limits/context loader overload, and the root under verification cannot supply expected facts or policy. Never translate a failed metadata load/bind into an empty set.

Move the complete generated message into `verifyPackageManifest(std::make_unique<proto::PackageManifest>(std::move(message)), ..., sessionCanonicalEncoding)` with validated build-service `PackageParseLimits` and `ArtifactAdmissionLimits`; the package access adapter privately borrows the exact context retained by the output transaction and assembly has no context parameter or second graph/context. Then use checked arithmetic over verified target modules and committed `ArtifactRef`s plus bounded counting writers for exact manifest/index bytes to compute `PackagePublicationSummary`. Compare blob/chunk counts, single/total module/payload bytes, manifest/index/logical/physical/staging bytes and reader/FD/buffer/copy peaks with the session-retained positive `PackagePublicationLimits` before constructing `PackageAssemblyResult`. If any step fails, return without creating an output path/backing, opening a source blob or starting a worker. After the pure gate, non-installed `detail::PackageManifestSerializationAccess` streams deterministic bytes through the session's sealed output-backing factory, obtains one `abi::ImmutableByteBackingRef`, and rechecks exact size/content digest before sealing `SerializedPackageManifest`; it chooses bounded inline or private sealed-file/CAS storage within the retained summary. The result records the session owner token and attachment generation. Operational/canonical limits do not enter identity; different sufficient policies cannot change members/chunks/bytes/ID.

- [ ] **Step 4: Stage the bundle and attach it to the outer transaction**

Create `PackageBundleIndex` with a locator plus expected content digest for `manifest.pb`, each final ELF and each weight/constant payload; store the WCRE `PackageManifestId` separately from `PackageBlobDigest`. The index is a delivery locator object and cannot add semantic facts absent from the manifest.

`stagePackageBundle` first requires the move-owned session and result to carry the same live outer owner and exact current attachment-table generation. It recomputes the pure publication summary from the non-forgeable result and session-owned artifact/payload declarations, requires exact equality with the retained summary and rechecks every field against the session-retained `PackagePublicationLimits`. Through `PackageProgramOutputAccess` it asks that same `ProgramOutputTransaction` for an all-or-none child reservation using the per-field minimum of stage limits and retained outer remaining limits; logical bytes, prospective physical bytes, roots/inodes/FDs/workers/buffers are distinct dimensions. Only then may it create a child private immutable root. It rechecks manifest/serialized bytes/blob digest/manifest ID/index equality plus exact coverage and identity equality with the retained view proof.

For each unique module/payload, the transaction-owned attachment factory selects exactly one `BlobAttachmentModeV1` from verified source/destination capabilities. `CasReference` acquires an independent destination/root refcount; `Reflink` proves COW independence; `ProtectedHardlink` requires a write-once store, fs-verity or equivalent seal plus a separate destination GC lease, never chmod alone. Each produces private `VerifiedBlobAttachment` binding source owner/generation, exact digest/size, backend attestation, destination root/relative locator and lifetime token; destination stat/seal is checked after attach and again by outer `commit()`. These modes charge full logical bytes but only actual new physical allocation and must not stream the entire blob.

If no trusted same-store mode is provable, `BoundedStreamCopy` is mandatory. Modules use only the distinct compiler-private `BoundTargetArtifactAttachmentSource` sealed into `PackageAssemblyInputView` and rebound to the live `PackageArtifactBuildSession`; package access move-consumes a build-session read/buffer lease before same-handle open/stream/hash/stat and retains it through destination write verification. It never calls or converts the standalone runtime `BoundTargetModuleSource`, and no `ArtifactVerificationReadLease`/`RuntimeArtifactVerificationSession` enters compiler staging. Payloads analogously use the session-owned compiler-private `BoundArtifactBlobSource` and its build-budget lease. Neither source exposes public `acquire/read/finish`, receives a locator to reopen or accepts a caller callback. Before writing each bounded chunk it checked-subtracts remaining logical/physical/staging/concurrent-copy budgets; end-of-stream requires zero remaining bytes, canonical chunk boundaries/digests and exact full digest. Actual readers/FDs/buffers/new physical bytes must remain below both summaries; mismatch is failure, not rechunk/retry. The source cannot receive logical tensor values, quant parameters or destination encoding and cannot repack.

Write/fsync deterministic `manifest.pb` and bundle index, retain every attachment lifetime token, then let the package access adapter translate the complete staged bundle into the distinct package attachment commitment and call the transaction's private core with the retained view proof, exact index/manifest commitments and expected `PackageManifestId`. It returns only `compiler::StagedPackageBundleToken`; no installed signature names `StagedPackageBundle`, and it performs no final rename, trusted alias/index update or standalone root acceptance. Any owner/table-generation/attachment attestation/refcount/mutation/GC mismatch permanently makes the outer transaction uncommittable.

`PackageProgramOutputAdapter.cpp` defines the Task 0-declared typed package member adapter. It is the only friend that converts sealed `StagedPackageBundle` into the private package commitment and invokes the core attach operation; it cannot construct executable/target commitments, alter retained scope/sink/limits or access a generic attachment API.

The outer driver remains the only visibility owner. After the declared completion scope's executable root, every target-set root and package attachment succeed, it publishes exactly one unified delivery-root rename or atomically creates/CASes one trusted `ProgramDeliveryCommitRecord` binding all those roots and the package bundle reference. On any package source/write/fsync/attach failure the outer transaction becomes uncommittable; already complete immutable roots remain unreachable and may be reclaimed by transaction cleanup or GC. A multi-directory/object-store deployment never treats a child directory rename as package publication.

`PackageBundleWriter.cpp` belongs only to `WaferPackageCompiler`; it consumes `package::PackageArtifactBuildSession`, never a caller target array/source interface, and uses its exact locator-free verified records plus capability-bound module/private payload sources. It chooses/validates `VerifiedBlobAttachment`s as above and places only bundle-relative locators plus exact size/digest in `PackageBundleIndex`; attachment mode/backend attestation never enters the index or manifest identity. It never calls a `moduleLocator()` path API or reopens target output by pathname. It streams the assembly result's shared immutable manifest backing directly to the transaction, without a second vector. `PackageBundleReader.cpp` belongs only to `WaferPackageFormat`; `loadPackageBundleMetadata` consumes `VerifiedPackageBundleReference` plus `artifact::ArtifactMetadataVerificationSession`, acquires the bound root and bounded-reads the bundle index into an `abi::ImmutableByteBackingRef` against external expected size/digest before trusting its manifest locator. An index-specific fixed descriptor/category plan and validated budget delegate to WaferABI `detail::preflightProtoWire`; generated parse is allowed only after that proof and `verifyParsedProtoMatchesPreflight` must match before index acceptance. It requires the accepted index's unique manifest facts to equal the external reference, then uses the same root for beneath/no-follow open, checks external manifest size/digest, creates a second immutable root-bound ABI backing and delegates to `parsePackageManifest`, whose shared-engine preflight precedes generated parse and semantic verification. Both use the metadata session's retained artifact limits/canonical context and typed host sub-capability. These gates complete before any other blob is opened. `LoadedPackageMetadata` shares those backings and owner-backed `VerifiedTargetArtifactSetRecord` views and retains only inert `UnboundBlobSourceDescriptor`s joined by canonical member/content key. It never sees attachment mode, parses `TargetArtifactSetDeliveryRoot`, calls the standalone target loader or constructs a runtime artifact source/session. Any accept-any inspection entry lives in a non-installed diagnostic tool, uses a disjoint inspection session/source and returns only `InspectedPackageBundle`; production code has no path overload.

In `lib/Wafer/Package/CMakeLists.txt`, add `PackageBundleReader.cpp` to `WaferPackageFormat` and create:

```cmake
target_link_libraries(WaferPackageFormat PUBLIC WaferArtifact)

add_library(WaferPackageCompiler
  PackageAssembly.cpp
  PackageBundleWriter.cpp
  PackageProgramOutputAdapter.cpp
)
target_link_libraries(WaferPackageCompiler PUBLIC
  WaferProgramDeliveryFormat WaferPackageFormat WaferArtifact WaferIR
  WaferTargetArtifacts MLIRIR MLIRParser)
```

`WaferRuntime` links `WaferProgramDeliveryFormat`, `WaferPackageFormat` and runtime-safe `WaferArtifact`. `wafer-opt` and `wafer-package-convert` link `WaferPackageCompiler`; no runtime target may link compiler-only targets transitively.

Extend `check_deps.py`/`check-deps.test` to require `WaferProgramDeliveryFormat -> WaferABI + WaferProgramDeliveryProto` while rejecting its target/package/compiler/MLIR edges; require the sole upper operational edge `WaferPackageFormat -> WaferArtifact` for session-bound readers while rejecting `WaferIR`, `WaferTargetArtifacts`, `WaferPackageCompiler` or `MLIR*` from that target; reject any reverse `WaferArtifact -> WaferPackageFormat` or `WaferArtifact -> WaferRuntime` include/link edge; require `WaferRuntime -> WaferProgramDeliveryFormat + WaferPackageFormat + WaferArtifact` while rejecting its `WaferIR`, `WaferTargetArtifacts`, `WaferPackageCompiler` and `MLIR*` edges; and require `WaferPackageCompiler -> WaferProgramDeliveryFormat + WaferPackageFormat + WaferArtifact + WaferIR + WaferTargetArtifacts`.

- [ ] **Step 5: Run focused assembly tests**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
build/wafer-dev/bin/WaferUnitTests \
  --gtest_filter='PackageAssemblyTest.*:PackageBundleTest.*'
python3 tools/check_deps.py
```

Expected: all source-of-truth and operational-quota negatives fail at their declared pre-output boundary; positive output contains one validated manifest, complete artifact members, fixed chunk tables and digest-checked payloads.

- [ ] **Step 6: Commit**

```bash
git add include/Wafer/Package/PackageAssembly.h \
  include/Wafer/Package/PackageBundle.h lib/Wafer/Package \
  unittests/Package unittests/CMakeLists.txt tools/check_deps.py \
  test/Tools/check-deps.test
git commit -m "Assemble package from committed executable"
```

### Task 4: Explicit Schema V2 Converter Boundary

**Files:**
- Create: `include/Wafer/Package/V2PackageConverter.h`
- Create: `lib/Wafer/Package/V2PackageConverter.cpp`
- Modify: `lib/Wafer/Package/CMakeLists.txt`
- Create: `tools/wafer-package-convert/CMakeLists.txt`
- Create: `tools/wafer-package-convert/wafer-package-convert.cpp`
- Modify: `CMakeLists.txt`
- Create: `unittests/Package/V2PackageConverterTest.cpp`
- Create: `test/Tools/wafer-package-convert.test`
- Modify: `test/Tools/wafer-package-metadata.test`
- Modify: `test/Tools/wafer-export-package-metadata.test`

**Interfaces:**
- Consumes: schema v2 JSON strictly as migration input; one `package::PackageArtifactBuildSession` produced from the latest same-transaction `compiler::PackageAssemblyInputView` by a fresh production compile/target flow. It has no raw committed-MLIR, caller target-array, context, backing factory or blob-reader import.
- Produces:

```cpp
mlir::FailureOr<PackageAssemblyResult>
convertV2Package(llvm::StringRef legacyJson,
                 PackageArtifactBuildSession &session,
                 const PackageParseLimits &buildServiceLimits,
                 const abi::ArtifactAdmissionLimits &artifactLimits);
```

- [ ] **Step 1: Add ambiguous-v2 failure matrix**

The converter tests must reject:

```text
workspace-bearing ABI whose binding_order omits workspace
duplicate or incomplete binding_order
module identified only by path with no matching complete artifact member key
LLVM argument count/name used to fill a slot
instructions array used as schedule or legality evidence
scalar kcore_local_drain used as terminal completion
completion source with no unique committed DAG projection
free-form entrypoint/symbol with no unique committed ModelEntrypointId mapping
persistent resources without exact state group/member set, realization,
snapshot/epoch policy and slot version roles
resource/module/entry fact that differs from the committed executable
raw committed MLIR path or caller target-ref array is offered as authority
PackageArtifactBuildSession is stale, cross-transaction or from a non-package scope
custom canonical context/backing/root/path/blob callback is offered to converter
```

Add one positive migration fixture whose every entrypoint, resource, slot, module and completion relation can be proven equal to the fresh production session's sealed `compiler::PackageAssemblyInputView`. Assert the output contains no instruction list/free binding order and passes the move-owning verifier with the converter transaction's package/artifact/canonical limits. The returned result must stage only with that same live session and exact attachment-table generation.

- [ ] **Step 2: Run focused tests and confirm no converter exists**

Run:

```bash
cmake --build build/wafer-dev --target wafer-package-convert WaferUnitTests -- -j128
```

Expected before implementation: target/source is missing.

- [ ] **Step 3: Implement audit-only conversion**

Check legacy input bytes and JSON nesting/counts against validated `PackageParseLimits`, then parse v2 with LLVM JSON. Treat every field as a claim, never authority. Derive the new manifest through `assemblePackage(session, buildServiceLimits, artifactLimits)`; compare v2 model/entrypoint/resource/module/entry/completion claims against the session's sealed transaction view and reject unprovable/contradictory claims. Return `mlir::FailureOr`, anchor diagnostics through the session's sealed executable owner and propagate failure directly. `instructions` only triggers an unsupported-schedule diagnostic and is never copied. No overload accepts a raw canonical context, backing factory, staging root or source callback.

Add `V2PackageConverter.cpp` to `WaferPackageCompiler`. The CLI first runs the normal source-backed named production compiler in-process, using one Task 0 `compiler::ProgramOutputTransaction` already bound to package scope/sink/output limits/canonical context, and builds every required target attachment. `compiler::detail::PackageProgramOutputAccess::buildPackageAssemblyInputView` creates the latest view, which is immediately moved through `beginPackageArtifactBuild`; only the resulting `package::PackageArtifactBuildSession` reaches the converter. It stages the returned result by move-consuming that same session. It emits a semantically validated package and structured rejections. It does not link into `WaferRuntime`, and ordinary package loading has no JSON branch.

The CLI is fixed to:

```text
wafer-package-convert \
  --legacy-json=<schema-v2.json> \
  --input-program-dir=<source-backed-program> \
  --artifact-limit=<kind=value>... \
  --canonical-limit=<kind=value>... \
  --publication-limit=<kind=value>... \
  --output-program-delivery=<trusted-commit-destination>
```

The converter validates closed artifact/canonical/publication/output limit profiles before creating the transaction. Source tensors/payloads flow through the normal program importer/materializer into transaction-owned content storage; JSON cannot supply them or override semantics. The production compiler attaches the sealed executable and every target, builds the view, runs conversion/assembly, stages the bundle and calls parameterless outer `commit()`. There is no `--committed-executable`, raw target root/ref array or path-based proof shortcut. It never emits an independently accepted package directory; compile/target/package/commit failure leaves child roots unreachable.

- [ ] **Step 4: Reclassify old tool tests as migration coverage**

Change old schema-v2 tests so success means either explicit converter acceptance with typed compiler inputs or a named ambiguity rejection. Remove assertions that validate instruction families, free lifecycle strings or scalar completion as current package semantics.

- [ ] **Step 5: Run focused converter tests**

Run:

```bash
cmake --build build/wafer-dev --target wafer-package-convert WaferUnitTests -- -j128
build/wafer-dev/bin/WaferUnitTests --gtest_filter='V2PackageConverterTest.*'
<configured-lit> -sv \
  build/wafer-dev/test/Tools/wafer-package-convert.test \
  build/wafer-dev/test/Tools/wafer-package-metadata.test \
  build/wafer-dev/test/Tools/wafer-export-package-metadata.test
```

Expected: converter accepts only the fully provable fixture; all ambiguous current-v2 cases fail closed; runtime is not involved.

- [ ] **Step 6: Commit**

```bash
git add include/Wafer/Package/V2PackageConverter.h lib/Wafer/Package \
  tools/wafer-package-convert CMakeLists.txt unittests/Package \
  unittests/CMakeLists.txt test/Tools
git commit -m "Add explicit legacy package converter"
```

### Task 5: Pure RuntimeSession Preflight

**Files:**
- Create: `include/Wafer/Runtime/RuntimeSession.h`
- Create: `include/Wafer/Runtime/RuntimeIdentity.h`
- Create: `include/Wafer/Runtime/RuntimeEnvironment.h`
- Create: `lib/Wafer/Runtime/RuntimeEnvironment.cpp`
- Create: `lib/Wafer/Runtime/RuntimePreflight.cpp`
- Modify: `lib/Wafer/Runtime/CMakeLists.txt`
- Create: `unittests/Runtime/RuntimePreflightTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: one immutable owner-backed `package::LoadedPackageMetadata` handle borrowed for pure selection; one registry-produced, non-forgeable `VerifiedRuntimeEnvironmentInventory`; one invocation's actual dimensions/external bindings/state request plus by-value validated admission policy, closed reservation mode and `InvocationControl`; no authority, context, runtime artifact session, blob source or provider object. The resulting plan move-owns policy/mode/control, retains immutable metadata/inventory owners plus the exact canonical required-domain set; Task 6 performs authority/context/session acquisition and one-way bind only after this pure stage succeeds.
- Produces:

```cpp
namespace wafer::runtime {
namespace detail { class RuntimeSessionPlanStorage; }
struct RuntimeEnvironmentFacts;
class VerifiedRuntimeEnvironmentInventory;
class WaferRuntimeServiceRegistry;
class RuntimeAdmissionPolicy;
struct InvocationRequest;
class InvocationControl;
class RuntimeSessionPlan;

enum class RuntimeCapacityReservationMode : std::uint8_t {
  NonBlocking,
  WaitUntilInvocationDeadline
};

llvm::Expected<RuntimeSessionPlan>
preflightRuntimeSession(
    const package::LoadedPackageMetadata &package,
    const VerifiedRuntimeEnvironmentInventory &environment,
    InvocationRequest invocation,
    RuntimeAdmissionPolicy admissionPolicy,
    RuntimeCapacityReservationMode capacityMode,
    InvocationControl control);
} // namespace wafer::runtime
```

The plan retains the package metadata's immutable storage and non-forgeable metadata/inventory owner proofs so selected verified module/payload descriptors cannot dangle, but it cannot construct another public `LoadedPackageMetadata`, open a descriptor or obtain a verification/context capability. It owns selected target/projection/shape variant IDs, committed typed execution-instance/rank mappings, exact selected resource and state-group realization records/keys, resident versus streamed window/staging requirements, selected transport binding members/relocation values, `(DdrArenaId, ScopeInstanceId)` requirements, exact version-role-aware KAD slot arguments, completion-node instances and the sorted all-and-only `RuntimeCapacityDomainKey` set required by the selection. It contains no authority, provider handle or serialization API.

The retained keys and selections are explicit:

```cpp
struct ResourceRealizationLookupKey {
  TargetVariantId targetVariant;
  ExecutableVariantId executableVariant;
  RankClassId rankClass;
  ProjectionSetId projectionSet;
};

struct IterationDomainKey {
  ExecutableVariantId ownerVariant;
  IterationDomainId domain; // strong nonzero uint32_t
};

struct IterationDomainInstance {
  IterationDomainKey key;
  std::uint32_t actualCount;
};

struct GraphIterationKey {
  IterationDomainKey domain;
  std::uint32_t iteration; // 0 <= iteration < actualCount
};

struct CompletionNodeInstance {
  abi::CompletionNodeId node;
  std::optional<GraphIterationKey> iteration;
};

struct RuntimeGraphNodeInstanceKey {
  std::variant<abi::EntryId, abi::CompletionNodeId> templateNode;
  std::optional<GraphIterationKey> iteration;
};

struct RuntimeGraphTemplateNodeKey {
  std::variant<abi::EntryId, abi::CompletionNodeId> templateNode;
  std::optional<IterationDomainKey> iterationDomain;
};

struct VerifiedGraphInstanceSummary {
  std::uint64_t checkedTotalNodes;
  std::uint64_t checkedTotalEdges;
  std::uint64_t symbolicPeakLiveNodes;
  std::uint64_t symbolicPeakLiveEdges;
  std::uint32_t maxLiveIterationSpan;
};

namespace detail {
class VerifiedGraphInstancePlanStorage;
class VerifiedGraphInstanceCursorStorage;
} // namespace detail

class VerifiedGraphInstanceCursor;

enum class ActivationPredicateState : std::uint8_t {
  AwaitingCountCompletion,
  True,
  FalseSkippedSuccess
};

class VerifiedGraphInstancePlan {
public:
  const VerifiedGraphInstanceSummary &summary() const;
  llvm::ArrayRef<IterationDomainInstance> iterationDomains() const;
  llvm::Expected<VerifiedGraphInstanceCursor> createCursor() const;
private:
  friend llvm::Expected<RuntimeSessionPlan> preflightRuntimeSession(
      const package::LoadedPackageMetadata &,
      const VerifiedRuntimeEnvironmentInventory &, InvocationRequest,
      RuntimeAdmissionPolicy, RuntimeCapacityReservationMode,
      InvocationControl);
  explicit VerifiedGraphInstancePlan(
      std::shared_ptr<const detail::VerifiedGraphInstancePlanStorage> storage);
  std::shared_ptr<const detail::VerifiedGraphInstancePlanStorage> storage_;
};

class VerifiedGraphInstanceCursor {
public:
  VerifiedGraphInstanceCursor(VerifiedGraphInstanceCursor &&) noexcept;
  VerifiedGraphInstanceCursor &operator=(VerifiedGraphInstanceCursor &&) noexcept;
  ~VerifiedGraphInstanceCursor();
  llvm::ArrayRef<RuntimeGraphNodeInstanceKey> ready() const;
private:
  friend class detail::CompletionExecutorAccess;
  ActivationPredicateState predicateState(
      const abi::ActivationPredicateId &) const;
  explicit VerifiedGraphInstanceCursor(
      std::unique_ptr<detail::VerifiedGraphInstanceCursorStorage> storage);
  std::unique_ptr<detail::VerifiedGraphInstanceCursorStorage> storage_;
};

struct RuntimeCapacityDomainKey {
  ProviderIdentity provider;
  ProviderDeviceId device;
  std::uint64_t providerGeneration;
};

struct ResourceByteRange {
  std::uint64_t offset;
  std::uint64_t size;
};

struct ResourceInstanceKey {
  ResourceId resource;
  ResourceRealizationLookupKey realization;
  ScopeInstanceId scope;
  RuntimeCapacityDomainKey domain;
};

struct ResourceScopeInstantiationRule {
  package::proto::ResourceScopeKind scopeKind;
  std::optional<IterationDomainKey> iterationDomain;
};

struct SelectedResourceRealization {
  ResourceRealizationLookupKey realization;
  RuntimeCapacityDomainKey domain;
  ResourceScopeInstantiationRule scopeRule;
  package::proto::ResourceRealization record;
};

struct SelectedQuantStorageProfile {
  abi::QuantStorageAbiProfileId id;
  abi::QuantStorageAbiProfileV1 profile;
};

struct ModelSharedPersistentScope {};

struct ComponentPersistentScope {
  abi::DistributedProgramSemanticId distributedProgram;
  abi::ComponentId component;
};

using PersistentStateScopeStorage =
    std::variant<ModelSharedPersistentScope, ComponentPersistentScope,
                 abi::ExecutionInstanceId>;

class PersistentStateScopeKey {
public:
  enum class Kind : std::uint8_t {
    ModelShared,
    Component,
    ExecutionInstance
  };
  Kind kind() const;
  llvm::ArrayRef<std::uint8_t> encodeV1() const;
  static llvm::Expected<PersistentStateScopeKey>
  decodeV1(llvm::ArrayRef<std::uint8_t> bytes);
private:
  friend llvm::Expected<RuntimeSessionPlan> preflightRuntimeSession(
      const package::LoadedPackageMetadata &,
      const VerifiedRuntimeEnvironmentInventory &, InvocationRequest,
      RuntimeAdmissionPolicy, RuntimeCapacityReservationMode,
      InvocationControl);
  explicit PersistentStateScopeKey(PersistentStateScopeStorage storage);
  PersistentStateScopeStorage storage_;
};

struct StateGroupRealizationKey {
  abi::StateConsistencyGroupId group;
  TargetVariantId targetVariant;
  ExecutableVariantId executableVariant;
  RankClassId rankClass;
  ProjectionSetId projectionSet;
  ScopeInstanceId groupScope;
  PersistentStateScopeKey persistentScope;
};

struct SelectedStateGroupRealization {
  StateGroupRealizationKey key;
  package::proto::StateGroupRealization record;
  llvm::SmallVector<ResourceInstanceKey> members;
};

struct SelectedStreamWindow {
  abi::StreamWindowId id;
  abi::ResourceRealizationRecordKey sourceRealization;
  std::uint32_t stagingRealizationOrdinal;
  const package::VerifiedPayloadBlob *source;
  package::BlobRange sourceRange;
  llvm::SmallVector<package::BlobChunk> requiredChunks;
  ResourceByteRange stagingRange;
  llvm::SmallVector<RuntimeGraphTemplateNodeKey> consumerTemplates;
  RuntimeGraphTemplateNodeKey copyCompleteTemplate;
};

struct TransportActionKey {
  EntryId ownerEntry;
  std::uint64_t structuralOrdinal;
};

struct TransportBindingSelection {
  TransportActionKey action;
  TransportBindingMemberId member; // action-local uint32_t
  llvm::SmallVector<RelocationValue> relocationValues;
};

enum class ActivationBranch : std::uint8_t {
  True,
  False
};

struct SelectedActivationPredicate {
  abi::ActivationPredicateId id;
  ResourceRealizationLookupKey countRealization;
  RuntimeCapacityDomainKey domain;
  abi::ModelProgramMemberId expertMember;
  std::uint64_t countByteOffset;
  package::proto::CountScalarType countType;
  std::uint64_t countCapacityBytes;
  std::uint64_t maxCount;
  RuntimeGraphTemplateNodeKey countComplete;
  RuntimeGraphTemplateNodeKey evaluate;
  RuntimeGraphTemplateNodeKey conditionalJoin;
};

struct ExpertWaveCapacityEnvelope {
  RuntimeCapacityDomainKey domain;
  std::uint64_t maxActiveMembers;
  std::uint64_t moduleSlots;
  std::uint64_t residentCodeBytes;
  std::uint64_t stagingBytes;
  std::uint64_t copySlots;
  std::uint64_t commandSlots;
  std::uint64_t hostTransferBufferBytes;
};

struct SelectedExpertWave {
  std::uint32_t waveOrdinal;
  llvm::SmallVector<abi::ActivationPredicateId> members;
  llvm::SmallVector<RuntimeGraphTemplateNodeKey> controlledNodes;
  llvm::SmallVector<ExpertWaveCapacityEnvelope> capacityEnvelope;
};

namespace detail { class InvocationResourcePlanBuilder; }

class InvocationCapacityDemandId final {
public:
  bool operator==(const InvocationCapacityDemandId &) const;
private:
  friend class detail::InvocationResourcePlanBuilder;
  explicit InvocationCapacityDemandId(std::uint64_t value);
  std::uint64_t value_;
};

struct ModuleResidencyRequirement {
  const package::VerifiedModuleBlob *blob;
  RuntimeCapacityDomainKey domain;
  llvm::SmallVector<RuntimeGraphTemplateNodeKey> firstUseFrontier;
  llvm::SmallVector<InvocationCapacityDemandId> capacityDemands;
};

enum class ModuleVerificationMode : std::uint8_t {
  EagerActiveSet,
  AtFirstUse
};

enum class ModuleResidencyMode : std::uint8_t {
  EagerActiveSet,
  GraphLiveness
};

enum class RuntimeAdmissionLimitKind : std::uint8_t {
  TemplateGraphNodes,
  TemplateGraphEdges,
  TotalInstantiatedWork,
  LiveCursorFrontierNodes,
  LiveCursorStateBytes,
  LiveIterationSpan,
  SimultaneouslyLiveResourceInstances,
  SimultaneouslyLiveScopeInstances,
  HostMetadataBytes,
  AllocationCount,
  BytesPerArena,
  TotalRequestedBytes,
  SimultaneousStagingWindows,
  SimultaneousCopies,
  SimultaneousEvents,
  SimultaneousCommands,
  ActiveBlobLeases,
  OpenFileDescriptors,
  LoadedModules,
  ModuleBytes,
  SelectedModuleBytes,
  SimultaneouslyMappedModuleBytes,
  ConcurrentVerificationWorkers,
  SimultaneouslyVerifiedModuleBytes,
  ResidentProviderCodeBytes,
  HostVerificationBufferBytes,
  HostReadBufferBytes,
  HostStreamTransferBufferBytes,
  FailureRecords,
  DiagnosticBytes
};

struct RuntimeAdmissionLimitValues {
  std::uint64_t maxTemplateGraphNodes;
  std::uint64_t maxTemplateGraphEdges;
  std::uint64_t maxTotalInstantiatedWork;
  std::uint64_t maxLiveCursorFrontierNodes;
  std::uint64_t maxLiveCursorStateBytes;
  std::uint64_t maxLiveIterationSpan;
  std::uint64_t maxSimultaneouslyLiveResourceInstances;
  std::uint64_t maxSimultaneouslyLiveScopeInstances;
  std::uint64_t maxHostMetadataBytes;
  std::uint64_t maxAllocationCount;
  std::uint64_t maxBytesPerArena;
  std::uint64_t maxTotalRequestedBytes;
  std::uint64_t maxSimultaneousStagingWindows;
  std::uint64_t maxSimultaneousCopies;
  std::uint64_t maxSimultaneousEvents;
  std::uint64_t maxSimultaneousCommands;
  std::uint64_t maxActiveBlobLeases;
  std::uint64_t maxOpenFileDescriptors;
  std::uint64_t maxLoadedModules;
  std::uint64_t maxModuleBytes;
  std::uint64_t maxSelectedModuleBytes;
  std::uint64_t maxSimultaneouslyMappedModuleBytes;
  std::uint64_t maxConcurrentVerificationWorkers;
  std::uint64_t maxSimultaneouslyVerifiedModuleBytes;
  std::uint64_t maxResidentProviderCodeBytes;
  std::uint64_t maxHostVerificationBufferBytes;
  std::uint64_t maxHostReadBufferBytes;
  std::uint64_t maxHostStreamTransferBufferBytes;
  std::uint64_t maxFailureRecords;
  std::uint64_t maxDiagnosticBytes;
  ModuleVerificationMode moduleVerificationMode;
  ModuleResidencyMode moduleResidencyMode;
};

class RuntimeAdmissionPolicy {
public:
  static llvm::Expected<RuntimeAdmissionPolicy>
  create(RuntimeAdmissionLimitValues values);
  const RuntimeAdmissionLimitValues &limits() const;
  std::uint64_t limitFor(RuntimeAdmissionLimitKind kind) const;
private:
  explicit RuntimeAdmissionPolicy(RuntimeAdmissionLimitValues values);
  RuntimeAdmissionLimitValues values_;
};

struct ArenaByteRequirement {
  DdrArenaId arena;
  std::uint64_t bytes;
  std::uint64_t allocationCount;
};

struct RuntimeAdmissionSummary {
  std::uint64_t templateGraphNodes;
  std::uint64_t templateGraphEdges;
  std::uint64_t totalInstantiatedWork;
  std::uint64_t maxLiveCursorFrontierNodes;
  std::uint64_t maxLiveCursorStateBytes;
  std::uint64_t maxLiveIterationSpan;
  std::uint64_t simultaneouslyLiveResourceInstances;
  std::uint64_t simultaneouslyLiveScopeInstances;
  std::uint64_t hostMetadataBytes;
  std::uint64_t allocationCount;
  std::uint64_t totalRequestedBytes;
  llvm::SmallVector<ArenaByteRequirement> arenaBytes;
  std::uint64_t simultaneousStagingWindows;
  std::uint64_t simultaneousCopies;
  std::uint64_t simultaneousEvents;
  std::uint64_t simultaneousCommands;
  std::uint64_t activeBlobLeases;
  std::uint64_t openFileDescriptors;
  std::uint64_t simultaneouslyLiveModules;
  std::uint64_t largestModuleBytes;
  std::uint64_t selectedModuleBytes;
  std::uint64_t simultaneouslyMappedModuleBytes;
  std::uint64_t concurrentVerificationWorkers;
  std::uint64_t simultaneouslyVerifiedModuleBytes;
  std::uint64_t residentProviderCodeBytes;
  std::uint64_t hostVerificationBufferBytes;
  std::uint64_t hostReadBufferBytes;
  std::uint64_t hostStreamTransferBufferBytes;
  ModuleVerificationMode moduleVerificationMode;
  ModuleResidencyMode moduleResidencyMode;
};

struct RuntimeCapacityDomainDemand {
  RuntimeCapacityDomainKey domain;
  std::uint64_t allocationCount;
  std::uint64_t totalRequestedBytes;
  llvm::SmallVector<ArenaByteRequirement> arenaBytes;
  std::uint64_t simultaneousStagingWindows;
  std::uint64_t simultaneousCopies;
  std::uint64_t simultaneousEvents;
  std::uint64_t simultaneousCommands;
  std::uint64_t activeBlobLeases;
  std::uint64_t openFileDescriptors;
  std::uint64_t simultaneouslyLiveModules;
  std::uint64_t simultaneouslyMappedModuleBytes;
  std::uint64_t concurrentVerificationWorkers;
  std::uint64_t simultaneouslyVerifiedModuleBytes;
  std::uint64_t residentProviderCodeBytes;
  std::uint64_t hostMetadataBytes;
  std::uint64_t hostVerificationBufferBytes;
  std::uint64_t hostReadBufferBytes;
  std::uint64_t hostStreamTransferBufferBytes;
  llvm::SmallVector<ExpertWaveCapacityEnvelope> expertWaveEnvelopes;
};

enum class RuntimeCapacityLeaseKind : std::uint8_t {
  PerInstance,
  Aggregate
};

struct IterationDomainFixedDelta {
  abi::IterationDomainId domain;
  std::uint64_t delta;
};

struct SameIterationPlusFixedDeltaTerminal {
  RuntimeGraphTemplateNodeKey terminalTemplate;
  llvm::SmallVector<IterationDomainFixedDelta> deltas;
};

struct DomainFinalIterationMinusFixedTailTerminal {
  RuntimeGraphTemplateNodeKey terminalTemplate;
  abi::IterationDomainId domain;
  std::uint64_t fixedTail;
};

struct InvocationTerminalSet {
  llvm::SmallVector<RuntimeGraphTemplateNodeKey> terminalTemplates;
};

using SymbolicTerminalInstanceRule = std::variant<
    SameIterationPlusFixedDeltaTerminal,
    DomainFinalIterationMinusFixedTailTerminal,
    InvocationTerminalSet>;

struct RuntimeCapacityReleaseRule {
  InvocationCapacityDemandId demand;
  RuntimeCapacityDomainKey domain;
  RuntimeCapacityDomainKey terminalDomain;
  RuntimeCapacityLeaseKind leaseKind;
  SymbolicTerminalInstanceRule terminal;
  llvm::SmallVector<ArenaByteRequirement> arenaCapacity;
  std::uint64_t commands;
  std::uint64_t events;
  std::uint64_t copies;
  std::uint64_t stagingWindows;
  std::uint64_t blobLeases;
  std::uint64_t openFileDescriptors;
  std::uint64_t verificationWorkers;
  std::uint64_t verifiedModuleBytes;
  std::uint64_t hostBufferBytes;
  llvm::SmallVector<std::uint32_t> moduleRequirementOrdinals;
};

namespace detail { class InvocationResourcePlanStorage; }

class InvocationResourcePlan {
public:
  const RuntimeAdmissionSummary &summary() const;
  llvm::ArrayRef<RuntimeCapacityDomainDemand> domains() const;
  llvm::ArrayRef<RuntimeCapacityReleaseRule> releaseRules() const;
private:
  friend llvm::Expected<RuntimeSessionPlan> preflightRuntimeSession(
      const package::LoadedPackageMetadata &,
      const VerifiedRuntimeEnvironmentInventory &, InvocationRequest,
      RuntimeAdmissionPolicy, RuntimeCapacityReservationMode,
      InvocationControl);
  explicit InvocationResourcePlan(
      std::unique_ptr<detail::InvocationResourcePlanStorage> storage);
  std::unique_ptr<detail::InvocationResourcePlanStorage> storage_;
};

class RuntimeSessionPlan {
public:
  RuntimeSessionPlan(RuntimeSessionPlan &&) noexcept;
  RuntimeSessionPlan &operator=(RuntimeSessionPlan &&) noexcept;
  ~RuntimeSessionPlan();
  const package::VerifiedPackageManifest &manifest() const;
  const VerifiedRuntimeEnvironmentInventory &environmentInventory() const;
  llvm::ArrayRef<RuntimeCapacityDomainKey> requiredDomains() const;
  const abi::ModelEntrypointId &modelEntrypoint() const;
  const VerifiedGraphInstancePlan &graphInstances() const;
  llvm::ArrayRef<SelectedResourceRealization> resourceRealizations() const;
  llvm::ArrayRef<SelectedQuantStorageProfile> quantStorageProfiles() const;
  llvm::ArrayRef<SelectedStateGroupRealization> stateGroups() const;
  llvm::ArrayRef<SelectedStreamWindow> streamWindows() const;
  llvm::ArrayRef<SelectedActivationPredicate> activationPredicates() const;
  llvm::ArrayRef<SelectedExpertWave> expertWaves() const;
  llvm::ArrayRef<TransportBindingSelection> transportBindings() const;
  llvm::ArrayRef<ModuleResidencyRequirement> modules() const;
  const InvocationResourcePlan &resources() const;
private:
  friend llvm::Expected<RuntimeSessionPlan> preflightRuntimeSession(
      const package::LoadedPackageMetadata &,
      const VerifiedRuntimeEnvironmentInventory &, InvocationRequest,
      RuntimeAdmissionPolicy, RuntimeCapacityReservationMode,
      InvocationControl);
  explicit RuntimeSessionPlan(
      std::unique_ptr<detail::RuntimeSessionPlanStorage> storage);
  std::unique_ptr<detail::RuntimeSessionPlanStorage> storage_;
};
```

`SelectedResourceRealization::record` and `SelectedStateGroupRealization::record` are already verified manifest records copied into the plan; Task 6 consumes their storage/residency/state-policy facts directly. `abi::ResourceRealizationRecordKey` is the shared manifest identity owner `(ResourceId, target/executable/projection coverage sets, committed RankClassId coverage set)` used by `StreamWindowId`; runtime never narrows or redefines it. `ResourceRealizationLookupKey` is only the singular selected lookup tuple and cannot contain a candidate execution-instance branch. The plan retains one selected realization template plus domain/scope-instantiation rule, not one copy per iteration. Only the rolling cursor creates a live `ResourceInstanceKey` by adding tagged `ResourceId`, exact domain and runtime `ScopeInstanceId`; for iteration scope that ID includes the lazily materialized `GraphIterationKey`. A streamed immutable source never becomes a full allocation requirement: only its declared window templates and bounded live staging frontier do. `RuntimeAdmissionPolicy` values are positive deployment inputs, never manifest fields; the effective limit is their per-field minimum with the matching verified domain capability. `VerifiedGraphInstancePlan` is a non-forgeable template/count/total/peak/max-live-span proof; it does not store expanded nodes/edges. `InvocationResourcePlan` owns checked global/per-domain summaries plus symbolic DAG-terminal release rules and module requirement ordinals, but no mutable usage counter or provider handle. `RuntimeSessionPlan` and these nested proofs are move-only/non-forgeable and have no public constructor/mutator, so callers cannot skip preflight or admission by assembling selected IDs, counts, graph nodes, release rules or handles themselves.

Define the inputs without string-derived identity:

```cpp
struct RuntimeProviderHardLimits {
  std::uint64_t maxTotalAllocationCount;
  llvm::SmallVector<ArenaByteRequirement> arenaCapacity;
  std::uint64_t maxLoadedModules;
  std::uint64_t maxSimultaneouslyMappedModuleBytes;
  std::uint64_t maxConcurrentVerificationWorkers;
  std::uint64_t maxSimultaneouslyVerifiedModuleBytes;
  std::uint64_t maxResidentProviderCodeBytes;
  std::uint64_t maxStagingWindows;
  std::uint64_t maxSimultaneousCopies;
  std::uint64_t maxSimultaneousEvents;
  std::uint64_t maxSimultaneousCommands;
  std::uint64_t maxActiveBlobLeases;
  std::uint64_t maxOpenFileDescriptors;
  std::uint64_t maxHostMetadataBytes;
  std::uint64_t maxHostVerificationBufferBytes;
  std::uint64_t maxHostReadBufferBytes;
  std::uint64_t maxHostStreamTransferBufferBytes;
};

enum class ReservationAuthorityMode : std::uint8_t {
  ExclusiveLocal,
  ProviderOrCentralAtomic
};

struct RuntimeReservationAuthorityFacts {
  bool devicesMayBeSharedAcrossProcesses;
  std::optional<ReservationAuthorityMode> mode;
  ReservationAuthorityId authority;
  std::uint64_t fencingGeneration;
  std::optional<std::chrono::steady_clock::time_point> exclusiveLeaseExpiry;
  llvm::SmallVector<RuntimeCapacityDomainKey> coveredDomains;
};

struct RuntimeCapacityDomain {
  RuntimeCapacityDomainKey key;
  abi::TargetEnvironmentFingerprint targetFingerprint;
  TargetAbiFacts targetAbi;
  llvm::SmallVector<TargetCapability> capabilities;
  llvm::SmallVector<TargetErratum> errata;
  llvm::SmallVector<ProviderExecutionMode> providerModes;
  std::uint32_t addressWidthBits;
  llvm::SmallVector<ArenaCapability> arenas;
  RuntimeProviderHardLimits hardLimits; // count/byte/concurrency caps
};

struct ExecutionInstanceCapacityDomainBinding {
  abi::ExecutionInstanceId executionInstance;
  RuntimeCapacityDomainKey domain;
};

struct ProjectionCapacityDomainBinding {
  abi::ProjectionSetId projection;
  llvm::SmallVector<RuntimeCapacityDomainKey> domains;
};

struct RuntimeEnvironmentFacts {
  llvm::SmallVector<RuntimeCapacityDomain> domains;
  RuntimeReservationAuthorityFacts reservationAuthority;
  TopologyFacts topology;
  llvm::SmallVector<EndpointAvailability> endpoints;
  llvm::SmallVector<ExecutionInstanceCapacityDomainBinding>
      executionInstanceDomains;
  llvm::SmallVector<ProjectionCapacityDomainBinding> projectionDomains;
};

class VerifiedRuntimeEnvironmentInventory {
public:
  const abi::TopologySnapshotId &topologySnapshotId() const;
  llvm::ArrayRef<RuntimeCapacityDomain> domains() const;
  llvm::Expected<RuntimeCapacityDomainKey>
  domainFor(const abi::ExecutionInstanceId &) const;
  llvm::ArrayRef<RuntimeCapacityDomainKey>
  domainsFor(const abi::ProjectionSetId &) const;
  bool devicesMayBeSharedAcrossProcesses() const;
  std::optional<ReservationAuthorityMode> reservationAuthorityMode() const;
  const RuntimeEnvironmentFacts &facts() const;
private:
  friend class WaferRuntimeServiceRegistry;
  class Storage;
  explicit VerifiedRuntimeEnvironmentInventory(
      std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

struct InvocationStateContext {
  StateNamespaceId stateNamespace;
  llvm::SmallVector<StateGroupRequest> groupRequests; // create/attach/reset
};

struct InvocationRequest {
  abi::ModelEntrypointId modelEntrypoint;
  llvm::SmallVector<ActualDimension> dimensions;     // typed DimId + value
  llvm::SmallVector<ActualInvocationPolicyInteger> policyIntegers;
  llvm::SmallVector<ExternalResourceBinding> externalBindings;
  std::optional<InvocationStateContext> state;
};

namespace detail { class InvocationCancellationState; }

class InvocationControl;

class InvocationCancellationSource {
public:
  static InvocationCancellationSource create();
  InvocationControl control(
      std::optional<std::chrono::steady_clock::time_point> deadline) const;
  void requestCancellation() const; // thread-safe, idempotent
private:
  explicit InvocationCancellationSource(
      std::shared_ptr<detail::InvocationCancellationState> state);
  std::shared_ptr<detail::InvocationCancellationState> state_;
};

class InvocationControl {
public:
  InvocationControl(InvocationControl &&) noexcept;
  InvocationControl &operator=(InvocationControl &&) noexcept;
  InvocationControl(const InvocationControl &) = delete;
  std::optional<std::chrono::steady_clock::time_point> deadline() const;
  bool isCancellationRequested() const;
private:
  friend class InvocationCancellationSource;
  friend class RuntimeSession;
  InvocationControl(
      std::shared_ptr<const detail::InvocationCancellationState> state,
      std::optional<std::chrono::steady_clock::time_point> deadline);
  std::shared_ptr<const detail::InvocationCancellationState> state_;
  std::optional<std::chrono::steady_clock::time_point> deadline_;
};

enum class FailureKind : std::uint8_t {
  Provider,
  Device,
  Transport,
  NodeTimeout,
  StateConflict,
  Resource,
  Module,
  CallerCancellation,
  CallerDeadline,
  InternalContract
};

struct InvocationFailure {
  FailureKind kind;
  std::optional<EntryId> entry;
  std::optional<ExecutionInstanceId> executionInstance;
  std::optional<ResourceId> resource;
  std::optional<abi::StateConsistencyGroupId> stateGroup;
  std::optional<GraphIterationKey> graphIteration;
  std::optional<abi::StreamWindowId> streamWindow;
  std::optional<CompletionNodeId> completionNode;
  std::string diagnostic;
  bool diagnosticTruncated;
};
```

`diagnostic` is explanatory output; failure kind and stable typed IDs own programmatic behavior.

Define `TargetAbiFacts`, capability/erratum/provider-mode enums, `TopologyFacts`, endpoint/arena records and positive `RuntimeProviderHardLimits` in this header as typed integer/enum/ID fields mirroring the runtime-safe target/topology identity schemas; none contains a caller precomputed digest, zero-as-unbounded value or free-form capability string. Each `RuntimeCapacityDomain` binds one provider/device/monotonic nonzero generation, recomputed target fingerprint, ABI/mode/arena facts and every shared finite capacity used by reservation: per-arena bytes/allocation count, module slots and mapped/resident code bytes, concurrent command/event/copy/window queues, active blob leases/FDs and host metadata/verification/read/transfer buffers; a missing required category is unsupported, not infinity. Task 6's process/deployment registry alone may query its move-owned per-domain provider capabilities side-effect-free, join the complete result to its retained `TrustedDeploymentInventory` and seal `VerifiedRuntimeEnvironmentInventory`. That inventory is discovery proof only: it has no allocation/load/submit or reservation authority. Pure preflight selects an exact canonical domain subset and retains both inventory owner/generation and required-domain proof; it cannot ask a caller source for alternate facts. Authority acquisition and exact-domain snapshots occur only after the plan exists.

- [ ] **Step 1: Write no-side-effect selection failures**

Use registry-owned recording-provider call counters initialized to zero, but do not pass a capability to preflight. Cover:

```text
blob/manifest identity mismatch rejected by package loader first
unchecked construction/mutation of LoadedPackageMetadata or construction of VerifiedRuntimeEnvironmentInventory is impossible
caller-created facts/digest with no registry-owned provider query and trusted-inventory proof is impossible
malformed capability/erratum/provider-mode/topology/arena/endpoint relation is rejected
inventory/provider identity, device, endpoint, arena or mode mismatch is rejected
empty/duplicate/noncanonical capacity-domain set or zero/nonmonotonic per-domain generation is rejected
missing/duplicate ExecutionInstanceId-to-domain or projection-to-domain mapping is rejected
inventory reports shared deployment but the selected exact-domain set has no supported authority mode
rank mapped to a domain whose target fingerprint/capability cannot execute the selected target is rejected
projection endpoint belongs to a different/unlisted domain or topology edge is missing
changing any fingerprinted runtime fact changes the recomputed identity
target environment mismatch before shape evaluation
unknown ModelEntrypointId or entrypoint from another ModelInterfaceSemanticId
static EntryId/function symbol/raw diagnostic alias used as request entrypoint
entrypoint required IO/state contract mismatch or selected variant lacks its invocation mapping
stateful entrypoint missing namespace/group requests or stateless entrypoint carrying extra state context
pinned topology/projection mismatch
relocatable member not in ConcreteRecordSet/allowed_bindings
overlapping target priority or shape guard
bounded dimension underflow/overflow and no fallback
different rank-local shape choice request
missing/duplicate invocation-policy integer used by BoundedCountExpr
count-expression intermediate overflow, ceildiv by zero or result outside [1,max_count]
caller-supplied expanded schedule/count/order is not an accepted request field
symbolic total/peak/max-live-span overflow or proof inconsistent with template bounds
rolling cursor emits an out-of-range iteration edge instead of omitting it
cursor ready/issued/completed watermarks regress, duplicate or exceed proved live horizon
capacity demand lacks one typed terminal-instance rule, has duplicate release rules or joins another demand/domain owner
same-iteration delta or final-tail rule underflows/overflows, resolves before a consumer, or names an unreachable terminal
per-instance/aggregate lease kind disagrees with scope, including release while a later iteration can still use the demand
invocation-terminal set is incomplete, noncanonical or releases before every issued command is safely terminal
missing/duplicate committed RankClass or canonical ExecutionInstance mapping
resource realization overlap/hole or wrong target/executable/rank/projection coverage
selected realization record/key differs from the unique manifest record
logical quant descriptor/storage encoding/profile ref mismatch for selected tuple
selected target environment lacks a required quant/storage profile capability
artifact member used-profile set or fingerprint differs from selected resource/KAD closure
packed/scale/zero-point/staging/scratch resource SlotId geometry mismatch
state group request/member set/realization/scope or slot version-role mismatch
resident realization selected as streamed or streamed source becomes a full allocation
stream window source/staging/consumer/copy instance mismatch
same source range with missing/extra/wrong canonical chunk-ref set
missing/incompatible TransportBindingMemberId or relocation value outside its finite schema
activation predicate count resource/domain/range/type/max or count-complete instance mismatch
predicate/expert member belongs to another variant/model component or wave coverage is incomplete
provider domain lacks conditional-issue/skipped-success capability before first side effect
expert wave symbolic envelope/max-active count exceeds per-domain admission capacity
conditional state/user-terminal node survives package verification or join/reuse dominance is broken
SlotId/ResourceId/ResourceRealizationLookupKey/iteration ScopeInstanceId mismatch
declared arena/span exceeds selected target capability or address width
zero/unbounded admission field or manifest attempt to override deployment policy
template node/edge, total-instantiated-work, live-frontier/state/span or symbolic resource/scope/metadata/arena peak checked overflow
simultaneous window/copy/event/command/blob-lease/module count or byte liveness exceeds effective limit
same module count with oversized single/selected/mapped/provider-code byte peak
stream windows whose bounded host transfer buffers exceed simultaneous byte limit
eager_active_set residency unconditional active-set count or graph_liveness simultaneous-live peak exceeds provider limit
eager_active_set verification/residency for a dynamic wave touches only predicate-true members and rejects an over-limit active envelope before that wave's first dependent side effect
all four verification/residency combinations preserve false-member zero open/verify/load/copy/submit effects
million-iteration fixture grows retained plan/cursor memory with total iteration count
```

Assert every failure leaves object-blob-open and every registry-owned provider's allocate/import/copy/load/submit counters at zero and does not mutate authority/context/cache state. Add compile-time assertions that `LoadedPackageMetadata` is copyable but not publicly constructible/mutable, while `VerifiedRuntimeEnvironmentInventory`, `RuntimeAdmissionPolicy` and `RuntimeSessionPlan` are not publicly constructible. No test or production overload accepts a caller environment source, raw environment facts, authority, context or runtime artifact session. Add success cases for at least two typed model entrypoints sharing one interface but selecting different verified root/terminal subgraphs, including stateless invocation with absent state context and stateful invocation with exact namespace/group requests. Prove registry-query+trusted-inventory-derived canonical 2- and 4-device domain inventories drive target/projection filtering before one global shape-guard evaluation, all execution instances use their committed rank mapping and exact domain/generation, and TP/PP/EP transport endpoints close across the retained topology. The result retains the sorted all-and-only required-domain set; extra inventory domains remain discovery-only and cannot be silently added to later authority/context acquisition. Preflight move-owns the exact admission policy, reservation mode and control state, rejects already-cancelled/expired inputs, and preserves the same cancellation owner/deadline for bind/create/execution. One entry covering two rank classes expands to both instance sets without duplicate entry identity; each selected iteration domain is evaluated once; every selected resource/module/window demand is assigned to the owning rank/projection domain. The dry-run-visible plan retains the requested `ModelEntrypointId`, actual counts, rolling graph proof, exact selected `ResourceRealization`, capacity domain and `(TransportActionKey, TransportBindingMemberId, relocation values)` rather than a reconstructed schedule/flat resource/route view. Raising a deployment limit accepts the same legal manifest/selection without changing selected IDs, microbatch count, graph, windows or semantic identity; lowering it returns typed `admission_limit_exceeded`. The positive PP fixture uses count 2, at least two domains and distinct stage/iteration-scoped workspace keys.

Require the closed admission enum/CLI table to contain template nodes/edges, checked total instantiated work, live cursor frontier/state bytes/iteration span and simultaneous resource/scope/module/window peaks. Compile-time/text tests reject `ExpandedGraphNodes`, `ExpandedGraphEdges`, `maxExpandedGraph*` and any summary field that implies one retained object per instantiated node. The million-iteration case must pass with a sufficiently high total-work scalar limit while live cursor/state limits remain constant.

- [ ] **Step 2: Run tests and verify preflight is absent**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
```

Expected before implementation: compile fails on `RuntimeSessionPlan` and `preflightRuntimeSession`.

- [ ] **Step 3: Implement typed request and instance keys**

First implement `RuntimeIdentity.h` as non-Protobuf, non-interchangeable strong values for `InvocationId`, `EntryInstanceId`, transient `ScopeInstanceId`, durable `PersistentStateScopeKey`, `ResourceVersionId`, `StateGroupVersionId` and `StateNamespaceId`; only their declared session/caller/registry factories can construct them, and no generated-message conversion exists. Reuse shared `abi::ModelEntrypointId`; `InvocationRequest` accepts that typed ID directly and has no raw alias/symbol/static-entry alternative. The persistent scope codec is the closed V1 union of model-shared, component and complete execution-instance facts; it cannot contain invocation/entry/iteration identity. Define only the sealed `VerifiedRuntimeEnvironmentInventory` view and typed facts here; Task 6's registry is its sole production factory, and `preflightRuntimeSession` has no overload accepting raw facts, a caller source or caller-supplied digests. Implement `RuntimeAdmissionPolicy::create` as the sole validator for positive limits plus the orthogonal `ModuleVerificationMode` and `ModuleResidencyMode`; the manifest/request cannot carry either policy axis or a zero-as-unbounded field. Define actual dimensions as `DimId -> uint64_t`, invocation policy values as typed references to manifest-declared bounded integer fields and external bindings as typed `abi::ResourceId` tagged-union records. A stateful selected entrypoint requires exactly one `InvocationStateContext` with a nonzero validated `StateNamespaceId` and complete create/attach/reset group requests; a stateless entrypoint requires it absent. There is no all-zero/default namespace. The request has no per-resource state policy, expanded graph, microbatch count, admission override or ordering field. Preserve every committed execution instance's component/stage, prerequisite-class refs, partition/replica and dp/tp/pp/ep coordinates in the plan. Construct `InvocationId` once, derive `EntryInstanceId` from invocation/entry/execution-instance/optional `GraphIterationKey`, derive transient `ScopeInstanceId` strictly from selected resource scope, and independently derive `PersistentStateScopeKey` from the selected persistent group declaration/mapping.

Use checked unsigned arithmetic for declared offset/span/end and target address-width verification. Provider-returned actual arena base/capacity is checked during Task 6 realization. A single `ResourceId -> handle` map is not part of the API.

- [ ] **Step 4: Implement fixed selection order and exact ABI arguments**

First resolve `InvocationRequest.modelEntrypoint` against the verified model interface and restrict consideration to executable variants with one exact invocation mapping whose required IO/state contract matches the request. Unknown/cross-interface ID, missing mapping or wrong contract fails before target/shape evaluation. Then filter compatible `TargetVariantId` and `ProjectionSetId` against every domain participating in their committed execution-instance/projection mapping, requiring each rank's exact provider/device/generation and target capability plus all cross-domain endpoints. Evaluate the remaining executable variants' typed shape guards exactly once and materialize the chosen variant's committed execution-instance-to-rank-class/domain mapping without regrouping. Retain only the requested mapping's reachable initial-root to terminal template subgraph plus required cleanup/state terminals; another entrypoint's disconnected graph is not launched or charged. Resolve each entry's one canonical nonempty rank-class coverage set against that mapping; one `EntryId` may instantiate for multiple classes/instances/domains but is never copied into class-salted identities. Evaluate each selected variant's `BoundedCountExpr` exactly once with checked nonnegative constant/add/mul/ceildiv/min/max over actual dimensions and declared policy integers, require `[1,max_count]`, and retain the result.

Before reserving/constructing any runtime container or opening a selected blob, run an iterative symbolic size/liveness pass over the selected template, counts, activation-wave envelopes and rank/projection domain mapping. Checked-count total nodes/edges/work, resource/scope instances and global host metadata; for each domain separately compute every arena's bytes/allocation count, staging windows/copies/events/commands/blob leases/FDs, selected/live module bytes, provider-resident code and host verification/read/stream-transfer buffers. Verification and residency are separate axes: `ModuleVerificationMode::EagerActiveSet` charges the unconditional active set at create and each dynamic active envelope at its predicate boundary, while `AtFirstUse` charges the proved simultaneous first-use verification peak; `ModuleResidencyMode::EagerActiveSet` analogously loads only the unconditional or predicate-true active set, while `GraphLiveness` computes the exact simultaneous provider-resident peak. No mode may open, verify or load a predicate-false member or serialize/reorder nodes to reduce a peak.

Mint one nonzero plan-local `InvocationCapacityDemandId` for every resource/module/window/queue/buffer demand. Derive a complete `SymbolicTerminalInstanceRule` for each demand rather than retaining only a terminal template: `same_iteration_plus_fixed_delta`, `domain_final_iteration_minus_fixed_tail` or `invocation_terminal_set`, together with demand/terminal capacity domains and `PerInstance` versus `Aggregate` lease kind. Prove checked delta/tail bounds, exactly one reachable release instance or closed terminal set, and that release is no earlier than every consumer and issued-command terminal. Aggregate leases that cover future iterations cannot release at the current iteration's terminal. The rolling cursor instantiates a concrete terminal key only when the demand enters its bounded live horizon or a final boundary is known; it never builds O(total iterations) release vectors. Compute stream-buffer peak from declared window/reuse edges, never full payload size. Compare global and per-domain results with the per-field minimum of deployment policy and the matching verified domain capability, returning `size_overflow` or owner-qualified `admission_limit_exceeded` before any side effect. A domain that cannot report a required hard capacity is unsupported, not unbounded. Limits cannot truncate a graph, select a prefix, reduce count, merge windows, move a rank to another device or choose another variant.

After the symbolic pass succeeds, construct one `VerifiedGraphInstancePlan` retaining immutable template refs, actual counts, checked total work, symbolic per-domain/global peaks and `maxLiveIterationSpan`; do not reserve an O(total iterations) node/edge/resource vector. Its private `VerifiedGraphInstanceCursor` lazily assigns deterministic `GraphIterationKey`/`EntryInstanceId`, applies only verified delta 0/+1, omits out-of-range destinations and maintains bounded ready/issued/live frontiers plus monotonic completion/release watermarks. Cursor storage is proportional to template size plus proved live horizon, independent of total iteration count. Retain `RuntimeAdmissionSummary` and symbolic release rules inside `InvocationResourcePlan`. This is still pure per-invocation admission, not shared-capacity acquisition; Task 6 must atomically reserve its per-domain envelope before any blob/provider side effect. Do not re-run selection, accept an expanded caller schedule, or use provider throughput to change count/order.

For every used resource/node instance, form `(TargetVariantId, ExecutableVariantId, RankClassId, ProjectionSetId)`, require exactly one manifest realization, resolve its committed execution instance/projection to one exact `RuntimeCapacityDomainKey`, derive the declared `ScopeInstanceId` including lazy `GraphIterationKey` where required, and retain that domain in `ResourceInstanceKey`; never derive storage/arena/capacity from the logical resource or module or move an allocation to a less-loaded device. For low-precision resources, retain the logical `QuantizationDescriptor`, selected `StorageEncodingDescriptor`, exact duplicate-free `QuantStorageAbiProfileId` refs and their verified imported records. Require the owning domain's capability/profile registry, target-artifact member `usedQuantStorageProfiles`/fingerprint, KAD slots and packed/scale/zero-point/staging/scratch resources to match before object-blob open; preflight never evaluates scales, repacks bytes or chooses another native/composite entry. For resident immutable realizations retain a domain-scoped full-source acquisition requirement. For streamed realizations retain only verified source refs plus compiler-declared windows, domain-scoped staging `ResourceInstanceKey`s, exact source/destination ranges, canonical required chunk refs, consumer instances and copy completion. Preflight resolves every chunk ref against `VerifiedPayloadBlob::source().chunks()`, requires exact digest/offset/size equality and exact source-range coverage, then retains that set; executor cannot recompute overlap or migrate a window. Retain reuse relations without choosing another window size/order or full backing.

Resolve each selected state group on the same axes, require its exact member realization refs/scope/policy, and retain one group request/realization rather than independent member policies. Each instantiated KAD binding remains `SlotId -> (ResourceId, StateSlotVersionRole)`: `none` for nonstate, `current` for the published group snapshot, `candidate` for a snapshot-ready atomic candidate, or `in_place` for the held exclusive group epoch. For every pinned transport action, retain the projection's exact complete member; for every relocatable action, deterministically choose only from the verified `ConcreteRecordSet`/`FiniteTemplateSet.allowed_bindings`, materialize only values authorized by its unique `RelocationSchema`, and retain the action-local member ID plus exact values. No state-policy inference, endpoint/channel/FSM/resource search or member renumbering is allowed.

Resolve every selected activation predicate to the exact domain-scoped count/control realization and compiler-declared expert member, retain its checked scalar range/max/count-complete/evaluate/join templates, and copy finite expert waves without adding/removing members. Include each wave's maximum per-domain module/code/staging/copy/command/host-buffer envelope in `InvocationResourcePlan`; initial shared reservation holds the simultaneous wave peak, not all expert backing bytes. Require conditional issue/skipped-success capability in each participating domain before side effects. The plan/cursor retains predicate state but does not read counts during preflight, choose active experts, opportunistically shrink a wave or turn unsupported conditional execution into always-load-all.

Finally walk each entry template's KAD ordered slots and resolve executable `SlotId -> (ResourceId, StateSlotVersionRole) -> selected Resource/StateGroup realization template + domain + scope rule`; reject any missing, duplicate, mixed-group-role or extra relation. For a low-precision entry, this same walk requires the KAD profile ID and packed/scale/zero-point/scratch slot shape/capacity/alignment to match the retained descriptors/profile, without runtime formula interpretation. The stored ABI argument templates reference typed resource/group binding requirements; only the rolling cursor/runtime materializer supplies the current iteration-aware `ScopeInstanceId`, never a parallel `ResourceId -> handle` table or preexpanded argument list.

- [ ] **Step 5: Run focused preflight tests**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
build/wafer-dev/bin/WaferUnitTests --gtest_filter='RuntimePreflightTest.*'
```

Expected: all negative cases fail before blob/provider side effects; the success plan contains one coherent target/shape tuple, non-forgeable checked `VerifiedGraphInstancePlan`/`InvocationResourcePlan`, typed canonical execution coordinates, exact retained realization/member choices, exact per-entry argument templates and one verified symbolic terminal-instance rule per capacity demand. A million-iteration fixture reports exact total work and release order while plan plus cursor peak memory remains bounded by template size and `maxLiveIterationSpan`.

- [ ] **Step 6: Commit**

```bash
git add include/Wafer/Runtime/RuntimeSession.h \
  include/Wafer/Runtime/RuntimeEnvironment.h \
  include/Wafer/Runtime/RuntimeIdentity.h \
  lib/Wafer/Runtime/RuntimeEnvironment.cpp \
  lib/Wafer/Runtime/RuntimePreflight.cpp lib/Wafer/Runtime/CMakeLists.txt \
  unittests/Runtime/RuntimePreflightTest.cpp unittests/CMakeLists.txt
git commit -m "Add pure runtime session preflight"
```

### Task 6: Actual Module Proof, Residency, Resource, and Persistent State Instances

**Files:**
- Modify: `include/Wafer/Runtime/RuntimeSession.h`
- Modify: `include/Wafer/Runtime/RuntimeEnvironment.h`
- Create: `include/Wafer/Runtime/RuntimeProviderCapabilities.h`
- Create: `include/Wafer/Runtime/RuntimeServiceBootstrap.h`
- Create: `include/Wafer/Runtime/RuntimePackageBinding.h`
- Create: `include/Wafer/Runtime/RuntimeModules.h`
- Create: `include/Wafer/Runtime/RuntimeCapacityReservation.h`
- Create: `include/Wafer/Runtime/RuntimeServiceContext.h`
- Create: `include/Wafer/Runtime/ModuleResidencyManager.h`
- Create: `include/Wafer/Runtime/PersistentStateRegistry.h`
- Create: `include/Wafer/Runtime/WeightCache.h`
- Create: `lib/Wafer/Runtime/RuntimeModules.cpp`
- Create: `lib/Wafer/Runtime/RuntimeCapacityReservation.cpp`
- Create: `lib/Wafer/Runtime/RuntimeServiceContext.cpp`
- Create: `lib/Wafer/Runtime/RuntimeServiceBootstrap.cpp`
- Create: `lib/Wafer/Runtime/RuntimeServiceContextInternal.h`
- Create: `lib/Wafer/Runtime/RuntimePackageBinding.cpp`
- Create: `lib/Wafer/Runtime/RuntimePackageBindingInternal.h`
- Create: `lib/Wafer/Runtime/ModuleResidencyManager.cpp`
- Create: `lib/Wafer/Runtime/RuntimeResources.cpp`
- Create: `lib/Wafer/Runtime/PersistentStateRegistry.cpp`
- Create: `lib/Wafer/Runtime/RecordingProviderCapabilities.cpp`
- Modify: `lib/Wafer/Runtime/CMakeLists.txt`
- Create: `unittests/Runtime/RuntimeResourceTest.cpp`
- Create: `unittests/Runtime/RuntimeModuleTest.cpp`
- Create: `unittests/Runtime/RuntimeCapacityReservationTest.cpp`
- Create: `unittests/Runtime/RuntimeServiceContextTest.cpp`
- Create: `unittests/Runtime/RuntimeServiceBootstrapTest.cpp`
- Create: `unittests/Runtime/RuntimePackageBindingTest.cpp`
- Create: `unittests/Runtime/ModuleResidencyManagerTest.cpp`
- Create: `unittests/Runtime/PersistentStateRegistryTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: successful pure move-only `RuntimeSessionPlan`; a by-value immutable `package::LoadedPackageMetadata` handle sharing the exact preflight owner; one authenticated deployment bootstrap result that contains a move-only `TrustedDeploymentInventory` plus immutable/copyable `TrustedRuntimeDomainConstraintCatalog` sharing its owner/generation; one process/deployment registry that move-owns the inventory, a bounded canonical `RuntimeProviderCapabilitySet`, distinct verified deployment-authority/durable-state service capabilities plus their cross-service relation, and the lower `RuntimeArtifactHostBudgetCapability`. Production accepts no caller backend, authority, context, runtime artifact session, manager or proof callback.
- Produces:

```cpp
namespace wafer::runtime {
namespace detail {
class RegistryLeaseAccess;
class StateGroupAttachmentStorage;
class AtomicStateGroupUpdateStorage;
class StateGroupEpochLeaseStorage;
class DurableStateBackingRefStorage;
class StateBackingAttestationStorage;
class StateBackingStoreAccess;
class WeightLeaseAccess;
class WeightLeaseStorage;
class RuntimeModuleJoinAccess;
class ExecutableHandleAccess;
class ExecutableHandleStorage;
class ExecutableUseLeaseStorage;
class VerifiedLoadableEntryStorage;
class VerifiedLoadableModuleStorage;
class LoadedProviderCodeEvidenceStorage;
class LoadedModuleProofStorage;
class ModuleLeaseStorage;
class RuntimeCapacityReservationStorage;
class ReservationCommitAccess;
class RuntimeArtifactBudgetAccess;
class RuntimeServiceContextAccess;
class RuntimePackageBindingAccess;
class RuntimeBoundPackageStorage;
class BoundBlobSourceStorage;
class OpenedBlobLeaseStorage;
class ProviderCapabilitySetFactoryAccess;
class RuntimeExecutionCapabilityView;
class RuntimeResourceMaterializerAccess;
class CompletionExecutorAccess;
class RuntimeServiceBootstrapAccess;
class RuntimeServiceBootstrapStorage;
class TrustedDeploymentInventoryBootstrapAccess;
class DeploymentReservationAuthorityBootstrapAccess;
class RuntimeDurableStateServiceBootstrapAccess;
} // namespace detail

class WaferRuntimeServiceContext;
class VerifiedRuntimeEnvironmentSnapshot;
class WaferModuleResidencyManager;
class WaferResidentWeightCache;
class WaferPersistentStateRegistry;
class TrustedDeploymentInventory;
class RuntimeDomainConstraintId;
class TrustedRuntimeDomainConstraintRef;
class TrustedRuntimeDomainConstraintCatalog;
class TrustedDeploymentInventoryBootstrapResult;
class DeploymentReservationAuthorityCapability;
class RuntimeDurableStateServiceCapability;
class VerifiedRuntimeServiceBootstrap;

class RuntimeBoundPackage final {
public:
  RuntimeBoundPackage(RuntimeBoundPackage &&) noexcept;
  RuntimeBoundPackage &operator=(RuntimeBoundPackage &&) noexcept;
  RuntimeBoundPackage(const RuntimeBoundPackage &) = delete;
  ~RuntimeBoundPackage();
private:
  friend llvm::Expected<RuntimeBoundPackage> bindPackageForRuntime(
      package::LoadedPackageMetadata, RuntimeSessionPlan,
      std::shared_ptr<WaferRuntimeServiceContext>,
      artifact::RuntimeArtifactVerificationSession);
  friend class RuntimeSession;
  friend class detail::RuntimePackageBindingAccess;
  explicit RuntimeBoundPackage(
      std::unique_ptr<detail::RuntimeBoundPackageStorage> storage);
  std::unique_ptr<detail::RuntimeBoundPackageStorage> storage_;
};

llvm::Expected<RuntimeBoundPackage> bindPackageForRuntime(
    package::LoadedPackageMetadata metadata,
    RuntimeSessionPlan plan,
    std::shared_ptr<WaferRuntimeServiceContext> services,
    artifact::RuntimeArtifactVerificationSession verification);

struct StateGroupKey {
  StateNamespaceId callerNamespace;
  abi::ModelInterfaceSemanticId modelInterfaceSemanticId;
  abi::StateConsistencyGroupId group;
  PersistentStateScopeKey persistentScope;
};

struct ProviderAllocationReceipt {
  AllocationHandle handle;
  RuntimeCapacityDomainKey domain;
  std::uint64_t observedBase;
  std::uint64_t observedCapacity;
  std::uint64_t observedAlignment;
  DdrArenaId observedArena;
};

struct ProviderStateBackingAttestationFacts {
  ProviderIdentity provider;
  ProviderDeviceId device;
  std::uint64_t providerGeneration;
  std::uint64_t observedByteSize;
};

struct ProviderReimportReceipt {
  ProviderAllocationReceipt allocation;
  PageTableRoot observedPageTable;
  ProviderStateBackingAttestationFacts observedAttestation;
};

class RealizedAllocation final {
public:
  RealizedAllocation(RealizedAllocation &&) noexcept;
  RealizedAllocation &operator=(RealizedAllocation &&) noexcept;
  RealizedAllocation(const RealizedAllocation &) = delete;
  ~RealizedAllocation();
  const AllocationHandle &handle() const;
  const RuntimeCapacityDomainKey &domain() const;
  std::uint64_t actualBase() const;
  std::uint64_t actualCapacity() const;
  std::uint64_t actualAlignment() const;
  const DdrArenaId &arena() const;
  const ScopeInstanceId &scope() const;
private:
  friend class detail::RuntimeResourceMaterializerAccess;
  friend class WaferResidentWeightCache;
  friend class detail::StateBackingStoreAccess;
  class Storage;
  explicit RealizedAllocation(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

struct ImmutableWeightKey {
  abi::ContentDigest content;
  TensorStorageDescriptor storage;
  std::optional<abi::QuantizationDescriptor> quantization;
  std::optional<abi::StorageEncodingDescriptor> storageEncoding;
  llvm::SmallVector<abi::QuantStorageAbiProfileId> quantStorageProfiles;
  ShardDescriptor shard;
  PackingDescriptor packing;
  abi::TargetEnvironmentFingerprint targetEnvironmentFingerprint;
  DdrArenaId arena;
  ScopeInstanceId placementScope;
  ProviderIdentity provider;
  ProviderDeviceId providerDevice;
  std::uint64_t providerGeneration;
};

struct ProviderCodeCacheKey {
  abi::ContentDigest finalModuleContent;
  abi::TargetArtifactFingerprint targetArtifactFingerprint;
  abi::TargetEnvironmentFingerprint targetEnvironmentFingerprint;
  ProviderIdentity provider;
  ProviderDeviceId providerDevice;
  std::uint64_t providerGeneration;
};

class VerifiedLoadableEntry {
public:
  const abi::TargetArtifactMemberKey &key() const;
  llvm::StringRef entrySymbol() const;
  const abi::VerifiedKernelAbiDescriptor &kernelAbi() const;
  llvm::ArrayRef<abi::RankClassId> coveredRankClasses() const;
  llvm::ArrayRef<abi::CompletionExportId> completionExports() const;
private:
  friend class detail::RuntimeModuleJoinAccess;
  explicit VerifiedLoadableEntry(
      std::shared_ptr<const detail::VerifiedLoadableEntryStorage> storage);
  // Storage carries a private module-proof token; entries are not independently forged.
  std::shared_ptr<const detail::VerifiedLoadableEntryStorage> storage_;
};

class VerifiedLoadableModule {
public:
  VerifiedLoadableModule(VerifiedLoadableModule &&) noexcept;
  VerifiedLoadableModule &operator=(VerifiedLoadableModule &&) noexcept;
  ~VerifiedLoadableModule();
  const ProviderCodeCacheKey &cacheKey() const;
  const artifact::VerifiedElfMetadata &elfMetadata() const;
  llvm::ArrayRef<VerifiedLoadableEntry> entries() const;
  const abi::ImmutableByteBackingRef &backing() const;
  llvm::Error writeVerifiedBytesTo(abi::ByteSink &sink) const;
private:
  friend llvm::Expected<VerifiedLoadableModule> verifyLoadableModule(
      RuntimeBoundPackage &,
      const ModuleResidencyRequirement &);
  explicit VerifiedLoadableModule(
      std::unique_ptr<detail::VerifiedLoadableModuleStorage> storage);
  std::unique_ptr<detail::VerifiedLoadableModuleStorage> storage_;
};

llvm::Expected<VerifiedLoadableModule> verifyLoadableModule(
    RuntimeBoundPackage &package,
    const ModuleResidencyRequirement &module);

class LoadedProviderCodeEvidence {
public:
  const ProviderCodeCacheKey &cacheKey() const;
  const abi::ContentDigest &loadedContentDigest() const;
private:
  friend class WaferModuleResidencyManager;
  explicit LoadedProviderCodeEvidence(
      std::shared_ptr<const detail::LoadedProviderCodeEvidenceStorage> storage);
  std::shared_ptr<const detail::LoadedProviderCodeEvidenceStorage> storage_;
};

class LoadedModuleProof {
public:
  const LoadedProviderCodeEvidence &codeEvidence() const;
  llvm::ArrayRef<VerifiedLoadableEntry> entries() const;
private:
  friend class WaferModuleResidencyManager;
  explicit LoadedModuleProof(
      std::shared_ptr<const detail::LoadedModuleProofStorage> storage);
  std::shared_ptr<const detail::LoadedModuleProofStorage> storage_;
};

enum class RuntimeCapacityKind : std::uint8_t {
  ArenaBytes,
  AllocationCount,
  ModuleSlots,
  MappedModuleBytes,
  VerificationWorkers,
  SimultaneouslyVerifiedModuleBytes,
  ResidentProviderCodeBytes,
  Commands,
  Events,
  Copies,
  StagingWindows,
  BlobLeases,
  OpenFileDescriptors,
  HostMetadataBytes,
  HostVerificationBufferBytes,
  HostReadBufferBytes,
  HostTransferBufferBytes
};

class RuntimeCapacityReservation {
public:
  RuntimeCapacityReservation(RuntimeCapacityReservation &&) noexcept;
  RuntimeCapacityReservation &operator=(RuntimeCapacityReservation &&) noexcept;
  ~RuntimeCapacityReservation();
  llvm::ArrayRef<RuntimeCapacityDomainKey> domains() const;
private:
  friend class detail::ReservationCommitAccess;
  friend class WaferModuleResidencyManager;
  friend class WaferResidentWeightCache;
  friend class detail::CompletionExecutorAccess;
  explicit RuntimeCapacityReservation(
      std::unique_ptr<detail::RuntimeCapacityReservationStorage> storage);
  std::unique_ptr<detail::RuntimeCapacityReservationStorage> storage_;
};

class RuntimeCapacityReservationCoordinator final {
public:
  RuntimeCapacityReservationCoordinator(
      RuntimeCapacityReservationCoordinator &&) noexcept;
  RuntimeCapacityReservationCoordinator &operator=(
      RuntimeCapacityReservationCoordinator &&) noexcept;
  ~RuntimeCapacityReservationCoordinator();
  llvm::Expected<RuntimeCapacityReservation> reserve(
      const InvocationResourcePlan &resources,
      const VerifiedRuntimeEnvironmentSnapshot &environment,
      const InvocationControl &control,
      RuntimeCapacityReservationMode mode);
private:
  friend class detail::RuntimeServiceContextAccess;
  static llvm::Expected<RuntimeCapacityReservationCoordinator> create(
      const VerifiedRuntimeEnvironmentSnapshot &environment,
      ReservationAuthority authority);
  class Storage;
  explicit RuntimeCapacityReservationCoordinator(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class ModuleLease {
public:
  ModuleLease(ModuleLease &&) noexcept;
  ModuleLease &operator=(ModuleLease &&) noexcept;
  ~ModuleLease();
  const ProviderCodeCacheKey &key() const;
  const LoadedModuleProof &proof() const;
private:
  friend class WaferModuleResidencyManager;
  explicit ModuleLease(std::shared_ptr<detail::ModuleLeaseStorage> storage);
  std::shared_ptr<detail::ModuleLeaseStorage> storage_;
};

class ExecutableUseLease;

class ExecutableHandle {
public:
  ExecutableHandle(ExecutableHandle &&) noexcept;
  ExecutableHandle &operator=(ExecutableHandle &&) noexcept;
  ~ExecutableHandle();
  const abi::TargetArtifactMemberKey &entryKey() const;
private:
  friend class detail::CompletionExecutorAccess;
  friend class detail::ExecutableHandleAccess;
  ExecutableUseLease acquireUse() const;
  explicit ExecutableHandle(
      std::shared_ptr<detail::ExecutableHandleStorage> storage);
  std::shared_ptr<detail::ExecutableHandleStorage> storage_;
};

class ExecutableUseLease {
public:
  ExecutableUseLease(const ExecutableUseLease &) = default;
  ExecutableUseLease &operator=(const ExecutableUseLease &) = default;
  ExecutableUseLease(ExecutableUseLease &&) noexcept = default;
  ExecutableUseLease &operator=(ExecutableUseLease &&) noexcept = default;
  ~ExecutableUseLease() = default;
  const abi::TargetArtifactMemberKey &entryKey() const;
private:
  friend class detail::CompletionExecutorAccess;
  explicit ExecutableUseLease(
      std::shared_ptr<const detail::ExecutableUseLeaseStorage> storage);
  std::shared_ptr<const detail::ExecutableUseLeaseStorage> storage_;
};

class ProviderFunctionHandle;
class ProviderEntryLookupRequest;
class CommandHandle;
struct RuntimeProviderDomainQueryResult;
struct ProviderAllocationRequest;
struct ProviderImportRequest;
struct ProviderCopyRequest;
struct ProviderReadbackRequest;
struct ProviderTransferReceipt;
struct ProviderTransportRequest;
struct ProviderTransportReceipt;
struct ProviderCommandRequest;
struct ProviderCompletionObservation;

class ProviderEnvironmentCapability {
public:
  virtual ~ProviderEnvironmentCapability() = default;
  virtual llvm::Expected<RuntimeProviderDomainQueryResult>
  queryDomainEnvironment() const = 0; // side-effect-free
};

class ProviderModuleCapability {
public:
  virtual ~ProviderModuleCapability() = default;
  virtual llvm::Expected<ModuleHandle>
  loadVerifiedCode(const VerifiedLoadableModule &) = 0;
  virtual llvm::Expected<ProviderFunctionHandle>
  resolveFunction(const ModuleHandle &,
                  const ProviderEntryLookupRequest &) = 0;
  virtual llvm::Error unloadCode(ModuleHandle) = 0;
};

class ProviderAllocationCapability {
public:
  virtual ~ProviderAllocationCapability() = default;
  virtual llvm::Expected<ProviderAllocationReceipt>
  allocate(const ProviderAllocationRequest &) = 0;
  virtual llvm::Expected<ProviderAllocationReceipt>
  importExternal(const ProviderImportRequest &) = 0;
  virtual llvm::Error release(AllocationHandle) = 0;
};

class ProviderTransferCapability {
public:
  virtual ~ProviderTransferCapability() = default;
  virtual llvm::Expected<ProviderTransferReceipt>
  copy(const ProviderCopyRequest &) = 0;
  virtual llvm::Expected<ProviderTransferReceipt>
  readback(const ProviderReadbackRequest &) = 0;
};

class ProviderTransportCapability {
public:
  virtual ~ProviderTransportCapability() = default;
  virtual llvm::Expected<ProviderTransportReceipt>
  submitTransport(const ProviderTransportRequest &) = 0;
};

class ProviderCommandCapability {
public:
  virtual ~ProviderCommandCapability() = default;
  virtual llvm::Expected<CommandHandle>
  submit(const ProviderCommandRequest &) = 0;
};

class ProviderCompletionCapability {
public:
  virtual ~ProviderCompletionCapability() = default;
  virtual llvm::Expected<ProviderCompletionObservation>
  poll(llvm::ArrayRef<CommandHandle>) = 0;
  virtual llvm::Expected<ProviderCompletionObservation>
  waitAny(llvm::ArrayRef<CommandHandle>,
          std::chrono::steady_clock::time_point absoluteDeadline) = 0;
  virtual llvm::Error cancel(llvm::ArrayRef<CommandHandle>) = 0;
};

class RuntimeProviderDomainCapabilitySet final {
public:
  RuntimeProviderDomainCapabilitySet(
      RuntimeProviderDomainCapabilitySet &&) noexcept;
  RuntimeProviderDomainCapabilitySet &operator=(
      RuntimeProviderDomainCapabilitySet &&) noexcept;
  RuntimeProviderDomainCapabilitySet(
      const RuntimeProviderDomainCapabilitySet &) = delete;
  ~RuntimeProviderDomainCapabilitySet();
private:
  friend class WaferRuntimeServiceRegistry;
  friend class detail::ProviderCapabilitySetFactoryAccess;
  class Storage;
  explicit RuntimeProviderDomainCapabilitySet(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class RuntimeProviderCapabilitySet final {
public:
  static llvm::Expected<RuntimeProviderCapabilitySet> create(
      std::vector<RuntimeProviderDomainCapabilitySet> canonicalDomains,
      std::uint64_t maxDomains);
  RuntimeProviderCapabilitySet(RuntimeProviderCapabilitySet &&) noexcept;
  RuntimeProviderCapabilitySet &operator=(
      RuntimeProviderCapabilitySet &&) noexcept;
  RuntimeProviderCapabilitySet(
      const RuntimeProviderCapabilitySet &) = delete;
  ~RuntimeProviderCapabilitySet();
private:
  friend class WaferRuntimeServiceRegistry;
  friend class detail::RuntimeServiceBootstrapAccess;
  class Storage;
  explicit RuntimeProviderCapabilitySet(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

namespace detail {
class RuntimeExecutionCapabilityView final {
public:
  RuntimeExecutionCapabilityView(RuntimeExecutionCapabilityView &&) noexcept;
  RuntimeExecutionCapabilityView(
      const RuntimeExecutionCapabilityView &) = delete;
  ~RuntimeExecutionCapabilityView();
private:
  friend class RuntimeServiceContextAccess;
  friend class ::wafer::runtime::WaferModuleResidencyManager;
  friend class ::wafer::runtime::WaferResidentWeightCache;
  friend class ::wafer::runtime::WaferPersistentStateRegistry;
  friend class CompletionExecutorAccess;
  class Storage;
  explicit RuntimeExecutionCapabilityView(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};
} // namespace detail

class WaferModuleResidencyManager final {
public:
  WaferModuleResidencyManager(WaferModuleResidencyManager &&) noexcept;
  WaferModuleResidencyManager &operator=(
      WaferModuleResidencyManager &&) noexcept;
  ~WaferModuleResidencyManager();
  llvm::Expected<ModuleLease> acquire(
      RuntimeCapacityReservation &reservation,
      RuntimeBoundPackage &package,
      const ModuleResidencyRequirement &module);
private:
  friend class detail::RuntimeServiceContextAccess;
  static llvm::Expected<WaferModuleResidencyManager> create(
      detail::RuntimeExecutionCapabilityView &providers);
  class Storage;
  explicit WaferModuleResidencyManager(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class WeightLease {
public:
  WeightLease(WeightLease &&) noexcept;
  WeightLease &operator=(WeightLease &&) noexcept;
  ~WeightLease();
  const RealizedAllocation &realizedAllocation() const;
private:
  friend class detail::WeightLeaseAccess;
  explicit WeightLease(
      std::shared_ptr<const detail::WeightLeaseStorage> storage);
  std::shared_ptr<const detail::WeightLeaseStorage> storage_;
};

class VerifiedResidentAllocationRequest;
class VerifiedResidentWeightRequest;
struct DurableStateRecord;
class DurableJournalToken;
struct DurableJournalRequest;
struct DurableBatchCasRequest;

class DurableStateBackingRef {
public:
  std::uint16_t codecVersion() const;
  std::uint64_t byteSize() const;
  const PageGeometry &pageGeometry() const;
  const StateBackingIntegrity &integrity() const;
  llvm::ArrayRef<std::uint8_t> encode() const;
private:
  friend class detail::RegistryLeaseAccess;
  friend class detail::StateBackingStoreAccess;
  explicit DurableStateBackingRef(
      std::shared_ptr<const detail::DurableStateBackingRefStorage> storage);
  std::shared_ptr<const detail::DurableStateBackingRefStorage> storage_;
};

class StateBackingAttestation {
public:
  const ProviderIdentity &provider() const;
  const ProviderDeviceId &device() const;
  std::uint64_t providerGeneration() const;
private:
  friend class detail::StateBackingStoreAccess;
  explicit StateBackingAttestation(
      std::shared_ptr<const detail::StateBackingAttestationStorage> storage);
  std::shared_ptr<const detail::StateBackingAttestationStorage> storage_;
};

struct StateGroupMemberView {
  abi::ResourceId resource;
  ResourceVersionId version;
  DurableStateBackingRef durableBacking;
  const RealizedAllocation *allocation;
  PageTableRoot pageTable;
  StateBackingAttestation attestation;
};

class StateGroupAttachment {
public:
  StateGroupAttachment(StateGroupAttachment &&) noexcept;
  StateGroupAttachment &operator=(StateGroupAttachment &&) noexcept;
  ~StateGroupAttachment();
  const StateGroupKey &key() const;
  const StateGroupVersionId &version() const;
  llvm::ArrayRef<StateGroupMemberView> members() const;
  const StateGroupMemberView &member(const abi::ResourceId &) const;
private:
  friend class detail::RegistryLeaseAccess;
  explicit StateGroupAttachment(
      std::shared_ptr<const detail::StateGroupAttachmentStorage> storage);
  std::shared_ptr<const detail::StateGroupAttachmentStorage> storage_;
};

class AtomicStateGroupUpdate {
public:
  AtomicStateGroupUpdate(AtomicStateGroupUpdate &&) noexcept;
  AtomicStateGroupUpdate &operator=(AtomicStateGroupUpdate &&) noexcept;
  ~AtomicStateGroupUpdate();
  const StateGroupKey &key() const;
  const StateGroupVersionId &baseVersion() const;
  const StateGroupAttachment &base() const;
  const StateGroupAttachment &candidate() const;
private:
  friend class detail::RegistryLeaseAccess;
  explicit AtomicStateGroupUpdate(
      std::unique_ptr<detail::AtomicStateGroupUpdateStorage> storage);
  std::unique_ptr<detail::AtomicStateGroupUpdateStorage> storage_;
};

class StateGroupEpochLease {
public:
  StateGroupEpochLease(StateGroupEpochLease &&) noexcept;
  StateGroupEpochLease &operator=(StateGroupEpochLease &&) noexcept;
  ~StateGroupEpochLease();
  const StateGroupAttachment &attachment() const;
private:
  friend class detail::RegistryLeaseAccess;
  explicit StateGroupEpochLease(
      std::unique_ptr<detail::StateGroupEpochLeaseStorage> storage);
  std::unique_ptr<detail::StateGroupEpochLeaseStorage> storage_;
};

struct ReboundStateBacking {
  RealizedAllocation allocation;
  PageTableRoot pageTable;
  StateBackingAttestation attestation;
};

class ResidentAllocationStoreCapability {
public:
  virtual ~ResidentAllocationStoreCapability() = default;
  virtual llvm::Expected<ProviderAllocationReceipt>
  allocateAndInitialize(const VerifiedResidentAllocationRequest &) = 0;
  virtual llvm::Expected<ProviderReimportReceipt>
  reimportAndAttest(const DurableStateBackingRef &) = 0;
  virtual llvm::Error release(AllocationHandle) = 0;
};

class WaferResidentWeightCache final {
public:
  WaferResidentWeightCache(WaferResidentWeightCache &&) noexcept;
  WaferResidentWeightCache &operator=(WaferResidentWeightCache &&) noexcept;
  ~WaferResidentWeightCache();
  llvm::Expected<WeightLease> acquire(
      RuntimeCapacityReservation &reservation,
      const VerifiedResidentWeightRequest &request);
private:
  friend class detail::RuntimeServiceContextAccess;
  static llvm::Expected<WaferResidentWeightCache> create(
      detail::RuntimeExecutionCapabilityView &providers);
  class Storage;
  explicit WaferResidentWeightCache(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class TransactionalDurableStateStoreCapability {
public:
  virtual ~TransactionalDurableStateStoreCapability() = default;
  virtual llvm::Expected<DurableStateRecord>
  read(const StateGroupKey &) = 0;
  virtual llvm::Expected<DurableJournalToken>
  beginFencedJournal(const DurableJournalRequest &) = 0;
  virtual llvm::Error batchCompareAndSwap(
      DurableJournalToken &, const DurableBatchCasRequest &) = 0;
  virtual llvm::Error abandon(DurableJournalToken &) = 0;
};

class WaferPersistentStateRegistry final {
public:
  WaferPersistentStateRegistry(WaferPersistentStateRegistry &&) noexcept;
  WaferPersistentStateRegistry &operator=(
      WaferPersistentStateRegistry &&) noexcept;
  ~WaferPersistentStateRegistry();
  llvm::Expected<StateGroupAttachment>
  createGroup(const StateGroupKey &, const StateGroupDescriptor &,
              StateSnapshotMaterializer &);
  llvm::Expected<StateGroupAttachment>
  attachGroup(const StateGroupKey &,
              const StateGroupDescriptor &expected,
              const VerifiedRuntimeEnvironmentSnapshot &environment);
  llvm::Expected<AtomicStateGroupUpdate>
  beginAtomicUpdateGroup(const StateGroupKey &,
                         const StateGroupDescriptor &expected,
                         StateGroupVersionId expectedBase,
                         StateSnapshotMaterializer &);
  llvm::Expected<StateGroupEpochLease>
  beginInPlaceUpdateGroup(const StateGroupKey &,
                          const StateGroupDescriptor &expected);
  llvm::Expected<StateGroupAttachment>
  publishGroup(AtomicStateGroupUpdate &);
  llvm::Error markGroupDirtyBeforeFirstWrite(StateGroupEpochLease &);
  llvm::Error completeGroup(StateGroupEpochLease &);
  llvm::Error poisonGroup(StateGroupEpochLease &,
                          const InvocationFailure &);
  llvm::Expected<StateGroupAttachment>
  resetGroup(const StateGroupKey &, StateSnapshotMaterializer &);
private:
  friend class detail::RuntimeServiceContextAccess;
  static llvm::Expected<WaferPersistentStateRegistry> create(
      TransactionalDurableStateStoreCapability &durableStore,
      detail::RuntimeExecutionCapabilityView &providers);
  class Storage;
  explicit WaferPersistentStateRegistry(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class RuntimeDomainConstraintId final {
public:
  RuntimeDomainConstraintId(const RuntimeDomainConstraintId &) = default;
  bool operator==(const RuntimeDomainConstraintId &) const;
private:
  friend class TrustedRuntimeDomainConstraintCatalog;
  friend class detail::TrustedDeploymentInventoryBootstrapAccess;
  explicit RuntimeDomainConstraintId(std::uint64_t value);
  std::uint64_t value_;
};

class TrustedRuntimeDomainConstraintRef final {
public:
  TrustedRuntimeDomainConstraintRef(
      const TrustedRuntimeDomainConstraintRef &) = default;
private:
  friend class TrustedRuntimeDomainConstraintCatalog;
  class Storage;
  explicit TrustedRuntimeDomainConstraintRef(
      std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class TrustedRuntimeDomainConstraintCatalog final {
public:
  TrustedRuntimeDomainConstraintCatalog(
      const TrustedRuntimeDomainConstraintCatalog &) = default;
  llvm::ArrayRef<RuntimeDomainConstraintId> ids() const;
  llvm::Expected<TrustedRuntimeDomainConstraintRef>
  resolve(const RuntimeDomainConstraintId &id) const;
private:
  friend class detail::TrustedDeploymentInventoryBootstrapAccess;
  class Storage;
  explicit TrustedRuntimeDomainConstraintCatalog(
      std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class TrustedDeploymentInventory final {
public:
  TrustedDeploymentInventory(TrustedDeploymentInventory &&) noexcept;
  TrustedDeploymentInventory(const TrustedDeploymentInventory &) = delete;
  ~TrustedDeploymentInventory();
private:
  friend class detail::TrustedDeploymentInventoryBootstrapAccess;
  friend class detail::RuntimeServiceBootstrapAccess;
  class Storage;
  explicit TrustedDeploymentInventory(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class TrustedDeploymentInventoryBootstrapResult final {
public:
  TrustedDeploymentInventoryBootstrapResult(
      TrustedDeploymentInventoryBootstrapResult &&) noexcept;
  TrustedDeploymentInventoryBootstrapResult(
      const TrustedDeploymentInventoryBootstrapResult &) = delete;
  ~TrustedDeploymentInventoryBootstrapResult();
  TrustedRuntimeDomainConstraintCatalog domainConstraintCatalog() const;
  llvm::Expected<TrustedDeploymentInventory> takeInventory();
private:
  friend class detail::TrustedDeploymentInventoryBootstrapAccess;
  class Storage;
  explicit TrustedDeploymentInventoryBootstrapResult(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

namespace detail {
class TrustedDeploymentInventoryBootstrapAccess final {
public:
  static llvm::Expected<TrustedDeploymentInventoryBootstrapResult>
  loadAuthenticatedDeploymentInventory();
private:
  TrustedDeploymentInventoryBootstrapAccess() = delete;
};
} // namespace detail

class ReservationAuthorityCandidate {
public:
  virtual ~ReservationAuthorityCandidate() = default;
  virtual llvm::Error revalidateFence() = 0;
};

class DeploymentReservationAuthorityCapability final {
public:
  DeploymentReservationAuthorityCapability(
      DeploymentReservationAuthorityCapability &&) noexcept;
  DeploymentReservationAuthorityCapability(
      const DeploymentReservationAuthorityCapability &) = delete;
  ~DeploymentReservationAuthorityCapability();
private:
  friend class WaferRuntimeServiceRegistry;
  friend class detail::DeploymentReservationAuthorityBootstrapAccess;
  friend class detail::RuntimeServiceBootstrapAccess;
  llvm::Expected<std::unique_ptr<ReservationAuthorityCandidate>>
  acquireExactDomains(
      llvm::ArrayRef<RuntimeCapacityDomainKey> canonicalDomains);
  class Storage;
  explicit DeploymentReservationAuthorityCapability(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class RuntimeDurableStateServiceCapability final {
public:
  RuntimeDurableStateServiceCapability(
      RuntimeDurableStateServiceCapability &&) noexcept;
  RuntimeDurableStateServiceCapability(
      const RuntimeDurableStateServiceCapability &) = delete;
  ~RuntimeDurableStateServiceCapability();
private:
  friend class WaferRuntimeServiceRegistry;
  friend class detail::RuntimeDurableStateServiceBootstrapAccess;
  friend class detail::RuntimeServiceBootstrapAccess;
  class Storage;
  explicit RuntimeDurableStateServiceCapability(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class VerifiedRuntimeServiceBootstrap final {
public:
  VerifiedRuntimeServiceBootstrap(
      VerifiedRuntimeServiceBootstrap &&) noexcept;
  VerifiedRuntimeServiceBootstrap(
      const VerifiedRuntimeServiceBootstrap &) = delete;
  ~VerifiedRuntimeServiceBootstrap();
private:
  friend llvm::Expected<VerifiedRuntimeServiceBootstrap>
  createVerifiedRuntimeServiceBootstrap(
      RuntimeProviderCapabilitySet,
      TrustedDeploymentInventory,
      DeploymentReservationAuthorityCapability,
      std::optional<RuntimeDurableStateServiceCapability>,
      verification::RuntimeArtifactHostBudgetCapability);
  friend class WaferRuntimeServiceRegistry;
  explicit VerifiedRuntimeServiceBootstrap(
      std::unique_ptr<detail::RuntimeServiceBootstrapStorage> storage);
  std::unique_ptr<detail::RuntimeServiceBootstrapStorage> storage_;
};

llvm::Expected<VerifiedRuntimeServiceBootstrap>
createVerifiedRuntimeServiceBootstrap(
    RuntimeProviderCapabilitySet providers,
    TrustedDeploymentInventory inventory,
    DeploymentReservationAuthorityCapability authority,
    std::optional<RuntimeDurableStateServiceCapability> durableState,
    verification::RuntimeArtifactHostBudgetCapability hostBudget);

class ReservationAuthority final {
public:
  ReservationAuthority(ReservationAuthority &&) noexcept;
  ReservationAuthority &operator=(ReservationAuthority &&) noexcept;
  ReservationAuthority(const ReservationAuthority &) = delete;
  ~ReservationAuthority();
private:
  friend class WaferRuntimeServiceRegistry;
  class Storage;
  explicit ReservationAuthority(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class VerifiedRuntimeEnvironmentSnapshot final {
public:
  llvm::ArrayRef<RuntimeCapacityDomain> domains() const;
private:
  friend class WaferRuntimeServiceRegistry;
  class Storage;
  explicit VerifiedRuntimeEnvironmentSnapshot(
      std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

class WaferRuntimeServiceContext final {
public:
  const VerifiedRuntimeEnvironmentSnapshot &environment() const;
private:
  friend class WaferRuntimeServiceRegistry;
  friend class RuntimeSession;
  friend class detail::RuntimeArtifactBudgetAccess;
  friend class detail::RuntimePackageBindingAccess;
  friend class StateMigrationSessionPlan;
  class Storage;
  explicit WaferRuntimeServiceContext(std::shared_ptr<Storage> storage);
  std::shared_ptr<Storage> storage_;
};

class WaferRuntimeServiceRegistry final {
public:
  static llvm::Expected<WaferRuntimeServiceRegistry> create(
      VerifiedRuntimeServiceBootstrap bootstrap);
  WaferRuntimeServiceRegistry(WaferRuntimeServiceRegistry &&) noexcept;
  WaferRuntimeServiceRegistry &operator=(
      WaferRuntimeServiceRegistry &&) noexcept;
  ~WaferRuntimeServiceRegistry();
  llvm::Expected<VerifiedRuntimeEnvironmentInventory>
  discoverEnvironment() const;
  llvm::Expected<std::shared_ptr<WaferRuntimeServiceContext>>
  getOrCreateContext(const RuntimeSessionPlan &plan);
private:
  friend class detail::StateMigrationPlacementAccess;
  class Storage;
  explicit WaferRuntimeServiceRegistry(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

namespace detail {
class RuntimeArtifactBudgetAccess final {
public:
  static llvm::Expected<artifact::ArtifactVerificationBudgetCapability>
  issue(const std::shared_ptr<WaferRuntimeServiceContext> &context);
private:
  RuntimeArtifactBudgetAccess() = delete;
};
} // namespace detail

class RuntimeSession {
public:
  static llvm::Expected<std::unique_ptr<RuntimeSession>>
  create(RuntimeBoundPackage package);
  ~RuntimeSession();
};
} // namespace wafer::runtime
```

`RuntimeProviderCapabilitySet` is the bounded canonical sole owner of the nonempty `RuntimeProviderDomainCapabilitySet` vector. Each domain set covers exactly one capacity domain and internally contains environment discovery, module, allocation/import/release/reimport, copy/readback, transport submit, ordinary command submit and completion poll/wait/status/cancel capabilities from one opaque provider owner and one monotonic generation; missing/extra capability, duplicate domain, over-limit/noncanonical order or mixed owner/generation fails before the outer set is sealed. Transfer and transport requests are private-executor-built from the already-selected typed resource/window or binding-member/relocation facts, and their low-level capabilities return only owner/generation-bound raw receipts containing completion handles; they cannot mint verified submissions or complete nodes, and transport cannot choose endpoints. `createVerifiedRuntimeServiceBootstrap` later move-consumes that exact outer owner together with the independent trust/authority/state/host capabilities, so no caller can reorder/substitute a raw vector between verification and registry installation. Tx/dry-run/recording implementations are adapters installed through non-installed `detail::ProviderCapabilitySetFactoryAccess`; the recording fake factory exists only in unittest/no-card targets. No caller-supplied semantic backend, provider callback or individual capability injection appears on a session/executor API.

`detail::TrustedDeploymentInventoryBootstrapAccess::loadAuthenticatedDeploymentInventory()` is the sole production issuer for deployment inventory facts. One call returns a nonaggregate `TrustedDeploymentInventoryBootstrapResult`; the host copies its immutable `TrustedRuntimeDomainConstraintCatalog`, then consumes `takeInventory()` exactly once into runtime-service bootstrap. Both outputs retain the same authenticated owner/generation. The catalog enumerates only opaque typed `RuntimeDomainConstraintId` values and resolves one ID to an owner-backed `TrustedRuntimeDomainConstraintRef`; it exposes no provider/device/generation, capacity or compatibility fact. Unknown IDs, a second inventory take, refs mixed across catalog owners/generations and refs later inconsistent with the registry-produced environment inventory fail. Provider adapters and ordinary callers cannot construct inventory/catalog/ref storage; a fake issuer exists only in `RuntimeServiceBootstrapTest`.

Deployment reservation authority and durable-state service may have owners distinct from the provider owner. `TrustedDeploymentInventory`, the final move-only `DeploymentReservationAuthorityCapability` and optional `RuntimeDurableStateServiceCapability` come from three deployment-authenticated bootstrap accesses disjoint from `ProviderCapabilitySetFactoryAccess`; provider adapters cannot mint or mutate the trust facts that attest them. The sole `createVerifiedRuntimeServiceBootstrap` factory move-consumes the exact provider set, trusted inventory, authority, optional durable-state capability and runtime host budget, verifies provider/inventory device relations, authority coverage/fences, durable namespace/fencing relation and their owner/generations, then seals one non-repairable `VerifiedRuntimeServiceBootstrap`. The lower `RuntimeArtifactHostBudgetCapability` is independently trusted and typed; its type plus one-way move ownership prevents replacement, and the bundle retains it only for accounting. Bootstrap never claims provider/deployment semantic owner equality for that capability or uses it as execution authorization; stale lower-ledger state fails its first operational reservation. `WaferRuntimeServiceRegistry::create` accepts only that bundle, so no caller can substitute an equal-self-reporting authority/store/provider or re-pair a cross-service proof. The registry cannot create a runtime artifact session before an exact-domain context exists. `discoverEnvironment()` invokes only the move-owned side-effect-free discovery capabilities, compares the all-domain result to the retained trusted inventory and seals `VerifiedRuntimeEnvironmentInventory`; it cannot allocate, load, submit, acquire authority or open model bytes.

After pure preflight returns, `getOrCreateContext(plan)` revalidates the plan's metadata/inventory proof and exact domains, then internally asks the registry-retained authority service for exactly `plan.requiredDomains()`, validates complete canonical coverage, owner/generation/fence/expiry, derives an exact-domain `VerifiedRuntimeEnvironmentSnapshot`, and single-flights one shared context per provider-domain generations plus authority owner/fence. `ReservationAuthority` is entirely registry/context-private; callers cannot obtain, swap or re-pair it with another plan. An `ExclusiveLocal` lease is consumed by at most one live context; provider/central mode also reuses one process context while the external service provides cross-process atomicity. A stale context admits no new session/issue and can perform only fenced cleanup; a new environment/authority generation gets a distinct key.

The context signs a private `detail::RuntimeExecutionCapabilityView` that binds the exact provider capability subset to authority/fence, provider owners/generations and any verified durable-state relation. It is the sole input to the concrete coordinator and final semantic managers; their factories and proof constructors are private to non-installed `detail::RuntimeServiceContextAccess`. Manager/executor/migration code cannot enumerate a caller capability set, replace one operation or submit through another provider. `detail::RuntimeArtifactBudgetAccess` combines the registry-retained runtime host capability with this exact context owner/generation and its service verification child ledger to issue one `artifact::ArtifactVerificationBudgetCapability`; neither it nor WaferArtifact can access runtime semantic managers. Compile-time tests reject public construction/factories for context, execution view, coordinator and all three managers, and dependency tests reject any `WaferArtifact -> WaferRuntime` include/link edge.

The host creates `RuntimeArtifactVerificationSession` from that issued capability, validated artifact limits and a move-owned canonical context, then calls `bindPackageForRuntime(metadata, std::move(plan), context, std::move(session))`. Binding performs no blob open or provider side effect. It revalidates the immutable index/manifest backing provenance, plan metadata/inventory owner, exact context domains and runtime verification service owner/generation, then atomically converts every inert descriptor to a private `BoundBlobSource`; failure returns no partially bound source. `RuntimeBoundPackage` owns the move-only plan/context/session and retains one cheap immutable metadata handle. There is no unbind/downgrade, public source accessor or context/session replacement. `RuntimeSession::create(std::move(boundPackage))` is the only production factory and accepts no backend, context, session, limits, capacity mode or control because the last three invocation values were move-owned by preflight and transferred through the bound package.

Use the runtime-only strong values from `Wafer/Runtime/RuntimeIdentity.h`. `StateNamespaceIdV1` is exactly big-endian `uint16 version=1` plus 32 caller-owned opaque bytes; wrong length/version and all-zero payload fail. `ResourceVersionIdV1` and distinct `StateGroupVersionIdV1` are exactly big-endian `(uint64 registry_epoch, uint64 monotonic_counter)`, both fields nonzero. Registry initialization durably creates/loads the epoch; counters advance atomically per owner, survive restart, never reuse, hard-fail on exhaustion and never wrap. `PersistentStateScopeKeyV1` is a separate durable closed union: model-shared, `(DistributedProgramSemanticId, ComponentId)`, or complete `ExecutionInstanceId`. Preflight derives it only from the manifest group scope/selected mapping and rejects persistent state with invocation/entry/iteration scope. `InvocationId`, `EntryInstanceId` and transient `ScopeInstanceId` remain runtime-local composites with no persistence or Protobuf conversion and can never enter a registry key.

The registry's canonical durable key codec is versioned and uses fixed big-endian widths plus tagged `abi::ResourceId` and `PersistentStateScopeKey` encodings. Durable group records contain only versions, canonical `ResourceId -> (ResourceVersionId, DurableStateBackingRef)` map, `valid | poisoned | requires_reset`, fencing/dirty facts and atomic recovery journal; they never serialize `RealizedAllocation`, `AllocationHandle`, page-table pointer, physical address or provider-private transient ID. CAS, reset, poison and exclusive group epoch carry fencing versions so stale processes cannot publish or clear poison after restart, eliminating ABA. `StateGroupAttachment`, `AtomicStateGroupUpdate`, `StateGroupEpochLease` and `WeightLease` are move-only, registry/cache-issued non-forgeable leases with private implementation-token constructors. Each attachment is issued only after the backing store reimports every durable member for the current verified provider/device/generation and attests size/page geometry/integrity; it retains that complete transient allocation/page-table map for slot binding without a handle side table.

- [ ] **Step 1: Write module, allocation, reuse, and state-machine failures**

Cover:

```text
two rank-scoped workspaces with one ResourceId receive distinct handles
two iteration-scoped workspaces with one realization receive distinct handles for microbatches 0/1
singleton/session-scoped resource is not duplicated merely because the graph iterates
same ResourceId/scope with different selected realization tuple never aliases
resource allocation request exactly matches the retained realization record
provider allocation/reimport receipts cannot construct RealizedAllocation/ReboundStateBacking; repo manager validates every observed fact first
different provider capability with matching self-reported IDs cannot forge allocation/state proof across execution-view owner/fence
session-shared immutable weight with identical digest/layout/shard reuses one handle
same bytes with different packing/shard metadata do not share a cache entry
same bytes with different quantization/storage encoding/profile refs do not share a cache entry
same weight bytes on different target/arena/placement/provider/device/generation do not share a handle
resident source truncation, mutation, chunk hole or digest mismatch publishes no cache lease
streamed payload larger than device capacity uses only declared bounded staging windows
streamed source never enters WaferResidentWeightCache or triggers a full backing allocation
double-buffer staging reuse occurs only after the prior window's last consumer completion
stream window copy/digest/early-EOF failure suppresses consumers and reverses cleanup
two consecutive invocation sessions reuse resident weight/state-group snapshots but not workspace
arena returned base is misaligned, too small or over target address width
allocation/copy failure releases prior resources in reverse order
missing event/timeout/status/cancel capability rejects before first allocate/import/copy/load call
pre-realization cancellation/deadline leaves object-open/allocate/import/copy/load/submit counters zero
mid-realization cancellation releases only this invocation's private work and preserves shared leases/users
context execution view exposes structurally valid but different provider/device/facts than preflight inventory
provider generation changes between preflight and RuntimeSession::create
endpoint/arena/provider-mode inventory mismatch leaves all provider side-effect counts zero
LoadedPackageMetadata, RuntimeSessionPlan, exact context and RuntimeArtifactVerificationSession owner proofs differ, are stale or already moved
binding one descriptor fails after earlier descriptors were checked and returns no partial RuntimeBoundPackage/source
copying metadata then concurrently binding two distinct plans/sessions shares no runtime lease or ledger ownership
verification reader/worker/simultaneously-verified-byte reservation exceeds its retained limits
two verification sessions each fit artifact limits but shared FD/worker/verified-byte capacity admits only the globally available amount
caller-defined semantic residency/cache/registry manager or proof-returning callback has no production API
provider adapter cannot construct TrustedDeploymentInventory, authority or durable-state capability that attests itself
authenticated inventory bootstrap returns one inventory/catalog owner-generation pair; inventory can be taken only once
catalog lookup rejects unknown RuntimeDomainConstraintId and exposes no provider/device/generation facts
refs from different catalog owner/generations cannot form one StateMigrationDomainSelectionPolicy
cross-service proof then equal-self-reporting provider/authority/store substitution is rejected by owner/generation/fence
coordinator/final managers/service context have no public constructor or direct create factory
wrong ELF class/machine/ISA/MABI/attributes/export/undefined symbol fails before provider allocation/load
missing/duplicate/wrong ABI note, KAD, target/artifact fingerprint or member coverage fails before provider allocation/load
actual KAD/profile refs or artifact usedQuantStorageProfiles mismatch fails before provider allocation/load
eager_active_set verification: malformed second unconditional module still leaves allocation/provider-load counters zero
eager_active_set verification: a malformed predicate-true dynamic member fails before that wave's first dependent load/copy/submit and enters global cleanup
eager_active_set mutation after preverify is caught by load-time same-handle reacquire/reverification and publishes no ModuleLease
at_first_use verification: create validates every owner-backed module view/relation and symbolic peak without opening module bytes
at_first_use digest/ELF/KAD failure occurs before that module load/submit, runs global typed cleanup and publishes no output or atomic state update
at_first_use explicitly permits already-terminal independent work before a later verification failure; it never rewrites verification policy or graph order
every verification/residency combination leaves predicate-false module open/verify/load/copy/submit counters at zero
VerifiedLoadableModule/Entry/ModuleLease/ExecutableHandle cannot be publicly constructed or cross-module swapped
moving an ExecutableHandle or reallocating its owning container cannot invalidate an outstanding EntrySubmission
two concurrent submissions from one entry hold independent use leases through their terminal completions
module eviction while any ExecutableUseLease exists is rejected; failure cleanup releases the final use exactly once
same module bytes on another target fingerprint/environment/provider/device/generation do not share residency
same provider-code key with different package EntryId/static/rank/completion relations reuses code only
cache hit revalidates current package relation and issues a distinct noninterchangeable ModuleLease token
eager_active_set residency over max-loaded-modules rejects before unconditional load or before a predicate-true dynamic wave load
module count fits but single/selected/live provider-code byte reservation exceeds limit
graph_liveness residency loads at deterministic first use and unloads only at the verified symbolic terminal-instance rule after every using command completion
simultaneously-live module peak over limit or atomic reservation failure rejects before load
two and many sessions each pass pure preflight but aggregate arena/module/command/event/copy/window/blob/FD/host-buffer demand is reserved all-or-none
no observed concurrent usage exceeds any provider/device/generation hard limit and a rejected request owns zero partial dimensions
2- and 4-device TP/PP/EP sessions acquire one token covering the exact canonical nonempty domain set
one later domain insufficient/prepare failure after earlier prepares rolls every domain back to unchanged counters
deadline/cancellation during cross-domain prepare leaves zero token/lease and the next FIFO request progresses
one participating device restart invalidates the whole invocation before submit while unrelated domain caches stay isolated
resident/module cache hit charges only invocation-marginal capacity while private allocations remain fully charged
one cache miss plus concurrent single-flight waiters owns one prospective insertion reservation, never one per waiter
failed cache insertion/loser/cancellation returns prospective capacity without publishing a partial entry
capacity release occurs only at retained DAG last-terminal frontiers and never rewrites or serializes the graph
failure/cancellation releases only after issued users are terminal; another session's shared cache ownership remains charged
FIFO bounded-wait tickets cannot be bypassed forever, cancellation/deadline removes one waiter, and nonblocking returns typed overload immediately
provider generation change invalidates old manager tokens/cache capacity ownership and stale tokens cannot allocate/load/submit
shared deployment without a verified reservation authority is unsupported before blob/provider effects
exclusive-local lease expiry/stale fence prevents reserve and every later submit/release/publish
provider/central cross-process prepare failure rolls every domain/process token back atomically
two process-local coordinators cannot both spend one exclusive fence or exceed central hard capacity
two same-key sessions receive the identical service context and share FIFO, single-flight caches and marginal counters
one exclusive lease/fence cannot construct two live service contexts, even through two registry calls
concurrent exact-key module acquire single-flights; load/resolve failure publishes no lease/executable
many resident ModuleLeases retain no object FD beyond root plus admitted transient verification leases
StateNamespaceId rejects wrong version/length/all-zero and exact bytes isolate namespaces
attach poisoned group fails across a newly created RuntimeSession
same namespace/model/group with different PersistentStateScopeKey never attaches
persistent group declared at invocation/entry/iteration scope is rejected before registry access
group attach rejects missing/extra member or incompatible dtype/layout/page/storage/capacity
package artifact change with the same model-interface/group identity preserves compatible state
different model-interface or StateConsistencyGroup IDs never share snapshots
full_copy prepares every member before snapshot-ready and preserves the base on failure
page_cow shares untouched pages, eagerly copies declared footprint and isolates concurrent branches
page_cow rejects nonpaged member, undeclared write footprint and provider-fault dependence
two updates from one base group version race and exactly one whole-group CAS succeeds
stale/failing update publishes zero members and releases private pages/refcounts
page-table/backing/many-layer/rank-shard members become visible only as one group map
in-place update holds an exclusive group epoch across every member write through complete/poison
attach/update/reset/poison racing an in-place group epoch linearize without partial visibility
registry restart round-trips durable backing refs/full member map and continues group/member counters monotonically
restart reimport returns new transient handles for the same group/resource versions
provider identity/device or generation that cannot reimport/attest yields requires_reset/whole-group poison, never stale handles
atomic crash before group CAS preserves current and journal recovery reclaims every orphan candidate backing
in-place crash after epoch/dirty/first member write/before complete recovers as whole-group poison or requires_reset
successful in-place terminal clears durable dirty only after all member completions
stale fencing/ABA and group/member counter exhaustion hard-fail without wrap
in-place group reset atomically creates one new complete valid snapshot
```

Recording provider-capability events must include typed resource/scope/arena keys; names are diagnostic-only.

- [ ] **Step 2: Run tests and verify runtime materialization is absent**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
```

Expected before implementation: compile fails on actual-ELF/module proof, residency manager, `RuntimeSession::create`, registry and cache interfaces.

- [ ] **Step 3: Implement actual-module proof, residency, and checked realization**

Define opaque move-only allocation/module/command handles plus a non-forgeable move-only `ExecutableHandle` in `RuntimeProviderCapabilities.h`/`RuntimeModules.h`. The executable handle retains the module-local proof token and exact verified entry key; no constructor accepts a symbol string, arbitrary module handle or caller member tuple. Low-level interfaces return only provider facts/handles/observations. They cannot construct `RealizedAllocation`, `LoadedProviderCodeEvidence`, `ModuleLease`, resource/state proof or choose a semantic key; the context-owned final managers validate provider returns and seal those objects through private access.

`RuntimeSession::create` move-consumes only `RuntimeBoundPackage`. It rechecks the retained control cancellation/deadline, exact-domain context/authority/provider generations and the complete capability closure before first object open or provider effect, then uses the plan-retained reservation mode/control to acquire one capacity reservation. No overload accepts metadata, plan, backend, provider capability, context, artifact session, manager, limits or replacement control separately. The session borrows the context-owned `RuntimeCapacityReservationCoordinator`, `WaferModuleResidencyManager`, `WaferResidentWeightCache`, `WaferPersistentStateRegistry` and signed execution view; it never constructs or replaces them. Missing event, timeout, status, cancel, DTE/collective or copy capability returns a structured error with zero side effects.

Submit the non-forgeable per-domain `InvocationResourcePlan` to the service-context-owned `RuntimeCapacityReservationCoordinator` before opening any selected object. The coordinator canonical-orders all nonempty required `RuntimeCapacityDomainKey`s, obtains prepare tokens/locks in that order, checked-adds each domain's arena bytes/allocation counts, module slots/mapped/resident-code bytes, command/event/copy/window/blob/FD and host buffer dimensions, and commits every counter only after all prepares and generations still match. Any shortage, prepare error, generation change, cancellation or deadline rolls back prepared domains in reverse order and returns typed `runtime_overloaded(RuntimeCapacityKind, domain, owner)` with zero partial ownership. `NonBlocking` returns immediately. `WaitUntilInvocationDeadline` uses one service-context-local FIFO ticket spanning the complete domain set; the head commits only when all domains fit and removal on the retained control's cancellation/deadline lets successors progress without starvation. Neither capacity mode changes graph order, either module policy axis, variant, domain mapping, microbatch count or windowing.

`RuntimeCapacityReservationCoordinator` is a concrete final semantic owner created only inside `WaferRuntimeServiceContext` by move-consuming its verified `ReservationAuthority`; it has no public factory or caller-implemented interface, and only non-installed `detail::ReservationCommitAccess` can construct/release tokens. In `ExclusiveLocal` mode it validates that one deployment-issued lease covers the complete canonical domain set and that its expiry plus monotonic fencing generation remain current; process-local counters/locks are authoritative only while that lease is live, and the fence is rechecked before every reserve, provider issue and ownership-changing release/publish. In `ProviderOrCentralAtomic` mode every canonical-domain prepare/commit/rollback is sent through the retained cross-process fenced client; a local success cache cannot replace central commit. If inventory says devices may be shared and no verified capability can be acquired, context/session creation returns structured unsupported before any blob/provider effect. Expiry, authority generation change or stale fence stops new issue for the whole context, drives failure cleanup and prevents the old process from releasing or publishing ownership under the new generation.

Reservation accounting is marginal within each domain's manager-owned caches. An already resident compatible provider-code/weight entry owns intrinsic capacity until eviction, so a new invocation reserves only its use plus private/transient dimensions. A miss reserves one prospective insertion in that exact domain; concurrent exact-key single-flight waiters join it and never double charge. Successful insertion transfers intrinsic capacity ownership to that domain's cache entry; failed load/copy/cancellation/loser releases it without publishing a handle. Arena/session/invocation allocations remain charged by declared scope. A generation change in any participating domain invalidates the whole invocation token and prevents further submit, while only that domain's cache-capacity ownership is isolated/invalidated; cleanup may release old handles and a new canonical snapshot must re-preflight/re-reserve all domains.

Runtime artifact verification uses one fixed four-ledger joint reservation, not independent quotas. For every read demand, non-installed access first checks the session-local artifact ledger, then the common `HostVerificationRegistry` runtime subledger/physical parent, then the exact-domain service-context verification child ledger, then the current invocation `RuntimeCapacityReservation`; work demand uses the same stable owner order for worker/simultaneously-verified-bytes/workspace dimensions. Every step is nonblocking, later failure destroys/rolls back all earlier prepared leases in reverse order, and no code waits while holding a partial tuple. The usable amount is the checked componentwise minimum of artifact limits, retained admission policy, host hard cap, exact provider-domain hard limits, service child limits and invocation capacity. `ArtifactVerificationReadLease` is reservation-only; private source access must move-consume it before opening, and `OpenedBlobLease` owns both the handle and lease through close/backing destruction. `ArtifactVerificationWorkLease` covers hash/ELF/KAD worker, bytes and workspace for their full lifetime. Neither lease is publicly constructible/copyable/splittable/early-releasable/cross-session. Metadata sessions never join the service/invocation ledgers. Multiple runtime sessions therefore cannot each spend the full local limit outside host/service/invocation accounting, while worker-count changes affect throughput only.

Sparse expert-wave capacity is reserved as compiler-proved per-domain envelopes inside the same token. Predicate evaluation may acquire only a non-forgeable member sublease from that already-held envelope; it cannot reserve ad hoc after observing counts, exceed max-active members or consume another wave/domain's pool. False members consume no blob/module/copy/submit sublease. Cache hits reduce marginal physical code/weight ownership under the existing exact-key rules but never change which predicates/members execute.

Before every object acquire and after each verification boundary, recheck cancellation/deadline. Independent of both policy axes, first consume every selected blob's exact owner-backed `abi::VerifiedTargetArtifactModuleView`, require one complete aggregate profile/fingerprint/KAD/export/rank/completion relation, recompute its duplicate-free `usedQuantStorageProfiles` from the selected resource/KAD closure and require exact verified profile records/environment capability. This view/relation closure and the symbolic module/activation peaks are byte-free plan facts; they do not claim that an ELF has been read or verified.

With `ModuleVerificationMode::EagerActiveSet`, before any allocation or provider side effect, preverify every unconditional active module sequentially or with only verification-session-reserved concurrency. A dynamic expert member is not part of that create-time set: only after its trusted predicate is true may the executor verify the complete active wave, still before that wave's first dependent load/copy/submit. Each verification obtains reservation-only read/work leases from the same artifact session, privately acquires the package-root-bound `OpenedBlobLease`, keeps handles/FDs/workers/simultaneously verified bytes/workspace within both the held `RuntimeCapacityReservation` and session, and lets a sealed non-installed factory turn that exact same handle into an `abi::ImmutableByteBackingRef` only after bounded stat/read/full-digest/stat. File/mmap/CAS implementations retain the immutable handle/owner and expose `readAt`/`writeTo`, never a required contiguous buffer or path. WaferArtifact calls `artifact::buildLoadedTargetElfContract(moduleView)` and `artifact::verifyTargetElf(backing, contract, limits, encoding)` through runtime-private verification-session access. It does not require or fabricate a `TrustedTargetArtifactSetDeliveryRef`, caller member array or expected-field aggregate, and WaferArtifact has no dependency on WaferPackageFormat. `RuntimeModules.cpp` joins only the returned `artifact::VerifiedElfMetadata`/verified notes and that exact module view/backing into `VerifiedLoadableModule`/`VerifiedLoadableEntry`; package semantic validation alone is never presented as actual-ELF proof. Preverification may release each backing after validation. A create-time unconditional failure releases the complete reservation with object/allocation/provider-load/submit counters at zero; a dynamic-wave failure follows global typed cleanup but still has zero side effects for that wave. Predicate-false members never acquire a read/work lease.

With `ModuleVerificationMode::AtFirstUse`, `RuntimeSession::create` deliberately does not open module byte sources. It seals only owner-backed module views/relations, activation-aware deterministic first-use frontiers, symbolic terminal rules and already-reserved peaks. Task 7 must, at the exact unconditional or predicate-true first-use frontier and before any load or submit dependent on that module, acquire its package source, seal one same-handle immutable backing, and complete full content-digest plus ELF/KAD/fingerprint verification through the retained artifact session. A late verification failure enters the same global typed failure/cleanup graph, suppresses all dependent and not-yet-issued work, drains or cancels issued work to safe terminals, publishes no output and no atomic state-group candidate, and applies whole-group poison rules to any already-issued in-place state mutation. Earlier independent work may already have executed; that is the explicit at-first-use policy, not a reason to silently preverify all modules or reorder the manifest DAG.

At actual first provider load, repo-owned final `WaferModuleResidencyManager` looks up only the intrinsic `ProviderCodeCacheKey`. On a miss it reacquires the package source, seals a new owner-backed `abi::ImmutableByteBackingRef` and ensures full digest/ELF/KAD/fingerprint verification with the exact module view through the RuntimeSession-owned artifact verification session. Under `EagerActiveSet` verification this is a required load-time reverify after the earlier active-set gate, unless the implementation retains the same already-sealed immutable backing; under `AtFirstUse` it is the first byte verification. No combination may rely on a prior path check. The new `VerifiedLoadableModule` owns that backing and its `artifact::VerifiedElfMetadata`, and the injected low-level `ProviderModuleCapability` synchronously consumes bytes only through `backing().readAt`/`writeVerifiedBytesTo`; a locator and mandatory contiguous `MemoryBufferRef` are never passed or reopened. A successful return means the provider no longer reads source bytes. Only then may the final manager construct `LoadedProviderCodeEvidence`, containing copied intrinsic typed content/exports/KAD/fingerprint facts plus provider/device/generation/load receipt, never `VerifiedElfMetadata`, backing, locator or object FD. It destroys `VerifiedLoadableModule` and releases the backing after synchronous load. This evidence cannot authorize another byte source; it proves only that the current fenced provider handle loaded the exact digest. Cache state never stores the first package's `EntryId`, static digest, rank or completion relation, and the provider capability cannot construct evidence or a module/use lease.

On both cache hit and miss, separately call `verifyModuleUseRelation(current abi::VerifiedTargetArtifactModuleView, current VerifiedModuleBlob, LoadedProviderCodeEvidence)`. It joins the current package's canonical member keys/symbol/KAD/rank/completion coverage against the cached intrinsic facts/load receipt, then issues a fresh package-scoped `LoadedModuleProof` and private module-use token. `ModuleLease` combines that proof/token with a shared provider-code lease/handle. Thus byte-identical compatible code can be shared across packages, while different package/static/member relations never reuse proof entries or `ExecutableHandle`s. Cached/resident code consumes provider capacity but no object FD/backing; a miss/reload performs fresh acquire/reverification. A low-`RLIMIT_NOFILE` test keeps many resident modules across invocations and requires object FDs to return to the root plus currently admitted transient verification leases. The final manager resolves a provider function only through `ProviderModuleCapability`, then checks the lease's private module-use token and rechecks EntryId/symbol/KAD/rank/completion relations before constructing `ExecutableHandle`; missing/wrong/cross-package entry publishes none. `ExecutableHandleStorage` retains the package-scoped module lease/proof and exact entry token rather than a raw manager pointer. Only `CompletionExecutorAccess` can derive an `ExecutableUseLease`; its shared immutable storage keeps that exact module-use relation and provider-code lease alive across handle moves, vector reallocations and asynchronous terminal completion. Zero executable-use leases is a necessary eviction/unload condition.

With `ModuleResidencyMode::EagerActiveSet`, acquire module-use leases for the unconditional active set after its verification gates and before the first issue; for a dynamic wave, do so only after each predicate is true and before that wave's first dependent side effect. With `ModuleResidencyMode::GraphLiveness`, Task 7 acquires at the retained activation-aware deterministic first-use frontier. Both release only when the matching verified `SymbolicTerminalInstanceRule` materializes and every outstanding `ExecutableUseLease` is terminal; an iteration-local completion cannot release an aggregate lease that covers future iterations. Every acquire consumes the exact prospective/cache-use capacity already represented by the same `RuntimeCapacityReservation`; it cannot make a second independent capacity decision. The manager's provider-epoch-scoped `ProviderCodeCacheKey` is exactly `(final module ContentDigest, TargetArtifactFingerprint, TargetEnvironmentFingerprint, provider identity, device identity, verified monotonic generation)`; concurrent first code load is single-flight, zero-code-lease entries alone are eligible for bounded typed eviction, and provider limits never justify reordering or serializing the manifest DAG. Package-scoped use-proof creation is not skipped on a code-cache hit. Predicate-false members never create a module-use lease. A generation/identity change atomically invalidates old module/weight entries; stale handles can only be unloaded/released, never rebound to a new session.

Recheck control immediately before every allocate/import/initialization copy/module load. Create per-domain arena/session resources and all non-iteration-scoped instances during session construction. Keep iteration-scoped realization templates dormant; Task 7 asks the rolling cursor/materializer to create a `ResourceInstanceKey = (ResourceId, ResourceRealizationLookupKey, ScopeInstanceId, RuntimeCapacityDomainKey)` only when its iteration enters the proved live horizon and releases it only when that demand's verified `SymbolicTerminalInstanceRule` materializes at the cursor watermark. This uses already reserved peak capacity, never releases an aggregate lease while future iterations remain, and cannot trigger replanning or exceed `maxLiveIterationSpan`.

Build every `ArenaAllocationRequest` only from retained `SelectedResourceRealization::record` plus exact domain/scope rule, including low-precision encoding/profile-required capacity/alignment; do not read storage/capacity/alignment/arena/scope/shard/packing from flat logical resources, selected modules or new analysis. Validate `RealizedAllocation`'s complete provider/device/generation domain, actual base/capacity/alignment, arena and scope against that record, checked span arithmetic, owning domain address width and KAD packed/scale/zero-point/staging/scratch span; never recover facts from an opaque handle or query side table. The shared `ResourceRealizationRecordKey` remains attached to selected record/window identity and is never replaced by the runtime tuple. Runtime copies exact committed bytes and binds ordinary slots; it does not dequantize, interpret formulas, repack, migrate devices or switch profiles. Cancellation unwinds only this invocation's live cleanup stack; shared weight/module leases remain valid for other holders.

Import external IO, acquire resident immutable weights through the full content+storage+shard+packing+target+arena+placement+domain key, attach/create/reset state groups, and allocate static workspace/control/status by declared scope. Iteration workspace and streamed staging are cursor-live resources, not session-sized arrays. Repo-owned final `WaferResidentWeightCache` and `WeightLease` retain the full verified resident `RealizedAllocation`; they never erase actual domain/base/capacity/alignment/arena/scope before KAD address construction. A resident acquisition lazily obtains an `OpenedBlobLease`, streams/rechecks all bytes with positioned reads while the low-level `ResidentAllocationStoreCapability` performs exact-domain allocate/copy/readback, and publishes a cache lease only after every byte/provider check succeeds. The capability cannot choose the semantic key or construct a lease. A streamed realization never calls `WaferResidentWeightCache`, allocates full backing or falls back to bootstrap initialization; it retains the lazy source and only the bounded live staging envelope for Task 7. Early EOF, mutation, digest mismatch, copy/load failure or concurrent cache loser releases partial allocation/capacity and publishes nothing.

Every successful acquisition pushes one typed cleanup action. Constructor failure executes that stack in reverse order; destructor is idempotent and never changes manifest facts. The capacity token is retained by `RuntimeSession`; Task 7 reports only cursor-instantiated `(InvocationCapacityDemandId, concrete terminal instance)` pairs produced from the preverified closed rules to its private release API. Capacity for a command/event/copy/window/transient buffer is released after its exact last safe terminal, module-use capacity after both its symbolic terminal instance and every outstanding `ExecutableUseLease` are terminal, and session/cache-scoped capacity only when the corresponding aggregate lease closes/evicts. Failure/cancellation follows the same terminal/cleanup edges and cannot release bytes or queue slots still reachable by issued provider work.

- [ ] **Step 4: Implement registry-owned version and poison state**

Use exactly `StateGroupKey = (StateNamespaceId, ModelInterfaceSemanticId, StateConsistencyGroupId, PersistentStateScopeKey)`. Derive the durable scope from the selected manifest group scope through the closed V1 union; an invocation/entry/iteration or transient `ScopeInstanceId` input is unrepresentable. The canonical member map is owned by the selected manifest group realization, not the key. Create/attach first require the exact member set, then validate every member's logical descriptor, selected realization, capacity/page geometry/storage/alias-update relation and common group scope/policy; independently attached cross-version members are unrepresentable.

`beginAtomicUpdateGroup` atomically checks/captures the expected base `StateGroupVersionId` and full durable base member map, mints unpublished member `ResourceVersionId`s, and writes candidate backing refs/refcounts plus owner transaction to a durable recovery journal before invoking only the selected snapshot materialization. `full_copy` allocates/copies every member capacity before any entry issue. `page_cow` requires declared paged geometry, clones page tables, refcounts untouched base pages and eagerly allocates/copies only pages intersecting the compiler-declared bounded update footprint; provider faults and runtime-inferred write sets are forbidden. Only after every member succeeds does `state_snapshot_ready` release candidate-role slots. Current readers retain the base attachment through their terminals; candidate readers/writers see the complete candidate map.

`publishGroup` performs one compare-and-swap from the captured base group version to a newly minted group version and atomically replaces the entire durable member map, clears the journal and returns a newly rebound complete attachment. A stale/failing branch publishes no member and reclaims private backing refs/pages/refcounts; crash recovery preserves old current and uses the journal to reclaim every orphan candidate. Partial member publication has no API. For `in_place_poison_on_failure`, acquire one registry-owned exclusive group epoch and, immediately before the first possible member write, durably persist `dirty + fencing` through `markGroupDirtyBeforeFirstWrite`; hold the lease through all writers, terminal completion and group `complete` or durable `poison`. `completeGroup` clears dirty only after all terminal success, while post-issue failure, process crash, lease/fencing loss or recovery observing dirty poisons the whole group or marks `requires_reset`. Reset atomically establishes and returns one new complete snapshot. Attach/update/reset/poison serialize at registry linearization points, and stale fencing tokens cannot publish, clear dirty or clear poison.

Provide repo-owned final `WaferPersistentStateRegistry` with one durable codec and a no-card low-level store fake. `TransactionalDurableStateStoreCapability` exposes only fenced read/journal/batch-CAS I/O and `ResidentAllocationStoreCapability` exposes only reimport/attestation; neither can choose semantic keys, mint versions, construct group/update/epoch proofs or bypass whole-group state transitions. Restart reloads namespace/group/scope keys, current group version, canonical `ResourceId -> (ResourceVersionId, DurableStateBackingRef)` map, status/fencing/dirty state and recovery journal. Before returning an attachment, the final registry reimports every backing against the current verified provider identity/device/generation, validates its attestation and issues new session-local `RealizedAllocation`/page-table handles; the same resource version may legitimately receive different transient handles. Missing backing, incompatible provider generation, integrity/attestation/refcount failure returns no partial group and durably transitions to `requires_reset` or whole-group poison when an in-place partial write is possible. Session destruction cannot erase or validate unfinished state. Unrelated package/artifact rebuilds do not invalidate compatible model-interface/group identity.

- [ ] **Step 5: Run focused resource/state tests**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
build/wafer-dev/bin/WaferUnitTests \
  --gtest_filter='RuntimeResourceTest.*:RuntimeModuleTest.*:RuntimeCapacityReservationTest.*:RuntimeServiceBootstrapTest.*:RuntimeServiceContextTest.*:RuntimePackageBindingTest.*:ModuleResidencyManagerTest.*:PersistentStateRegistryTest.*'
```

Expected: actual-ELF proof and module residency/entry resolution, shared service-context single-flight/marginal accounting, exclusive/central authority fencing, typed realization/allocation checks, resident reuse and bounded streamed windows, reverse cleanup, durable exact state-group identity, full-copy/page-COW snapshot readiness, whole-group CAS and exclusive-epoch poison/reset semantics all pass.

- [ ] **Step 6: Commit**

```bash
git add include/Wafer/Runtime lib/Wafer/Runtime \
  unittests/Runtime/RuntimeResourceTest.cpp \
  unittests/Runtime/RuntimeModuleTest.cpp \
  unittests/Runtime/RuntimeCapacityReservationTest.cpp \
  unittests/Runtime/RuntimeServiceContextTest.cpp \
  unittests/Runtime/ModuleResidencyManagerTest.cpp \
  unittests/Runtime/PersistentStateRegistryTest.cpp unittests/CMakeLists.txt
git commit -m "Verify modules and materialize runtime resources"
```

### Task 7: Completion DAG Execution and Failure Semantics

**Files:**
- Modify: `include/Wafer/Runtime/RuntimeSession.h`
- Modify: `include/Wafer/Runtime/RuntimeProviderCapabilities.h`
- Create: `lib/Wafer/Runtime/CompletionExecutor.cpp`
- Create: `lib/Wafer/Runtime/DryRunProviderCapabilities.cpp`
- Create: `lib/Wafer/Runtime/TxProviderCapabilities.cpp`
- Modify: `lib/Wafer/Runtime/CMakeLists.txt`
- Create: `unittests/Runtime/RuntimeCompletionTest.cpp`
- Create: `unittests/Runtime/RuntimeProviderCapabilityTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: fully realized `RuntimeSession`; manifest-derived entry/completion graph; its context-owned signed execution capability view and typed provider command/event/status observations. No caller capability parameter.
- Produces:

```cpp
class EntrySubmission {
public:
  const ExecutableUseLease &executable() const;
  const EntryInstanceId &entryInstance() const;
  llvm::ArrayRef<KadArgument> orderedArguments() const;
private:
  friend class detail::CompletionExecutorAccess;
  EntrySubmission(ExecutableUseLease executable,
                  EntryInstanceId entryInstance,
                  llvm::SmallVector<KadArgument> orderedArguments);
  ExecutableUseLease executable_;
  EntryInstanceId entryInstance_;
  llvm::SmallVector<KadArgument> orderedArguments_;
};

struct OmittedFailureOwnerCount {
  FailureKind kind;
  std::optional<RuntimeCapacityDomainKey> domain;
  std::uint64_t omitted;
};

class BoundedInvocationFailureReport {
public:
  llvm::ArrayRef<InvocationFailure> retainedFirstLastSamples() const;
  llvm::ArrayRef<OmittedFailureOwnerCount> omittedByOwner() const;
  std::uint64_t totalObserved() const;
  std::uint64_t totalOmitted() const;
  std::uint64_t retainedDiagnosticBytes() const;
private:
  friend class detail::CompletionExecutorAccess;
  class Storage;
  explicit BoundedInvocationFailureReport(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

struct InvocationResult {
  enum class Outcome {
    Succeeded,
    Failed,
    Cancelled,
    DeadlineExceeded
  };
  Outcome outcome;
  std::optional<InvocationFailure> primaryFailure;
  BoundedInvocationFailureReport secondaryFailures;
};

llvm::Expected<InvocationResult>
executeInvocation(RuntimeSession &session);
```

- [ ] **Step 1: Write completion and failure-injection tests**

Required graphs and results:

```text
host command -> device local drain -> DTE wait -> copy visibility -> success
two rank branches -> stage barrier -> output copy -> success
two-rank reciprocal DTE issues both receiver-ready and both submits before first poll/wait
pipeline stages exhaust each ready issue wave before observing a completion
PP count 2 overlaps stage-0 iteration 1 with stage-1 iteration 0 through +1 edges
PP middle-iteration failure suppresses remaining dependent iterations and reaches safe peer waits
entry submissions retain exact IterationDomainId/iteration in EntryInstanceId and resource scope
entry submission contains only module-token-bound ExecutableHandle plus ordered KAD arguments, never symbol/module handle
cross-module VerifiedLoadableEntry or released ModuleLease cannot produce/retain an executable submission
moving/reallocating the original ExecutableHandle container cannot invalidate an outstanding submission
two concurrent submissions from one handle keep independent use leases until both terminal completions
eviction/unload while a submission use lease exists is rejected; failure cleanup releases it without UAF/double-release
local drain without trusted host/DTE completion is rejected before execution
missing event/timeout/status provider capability is rejected before first submit
already-cancelled/expired control before realization has zero object/provider side effects
one source cannot extend or shorten the retained absolute deadline by creating a second control object
another source with the same deadline cannot replace the RuntimeSession-retained control
mid-PP/mid-stream-copy cancellation suppresses descendants and safely terminates issued peers
caller deadline uses min(node deadline, caller steady-clock deadline) at poll/wait boundaries
mid-atomic cancellation discards whole candidate; mid-in-place cancellation durably poisons group
cancelling one InvocationId does not cancel another holder of shared weight/module leases
earlier provider failure beats later cancellation; cancellation/deadline beat cleanup failure
rank timeout stops descendants and joins one global failed result
transport error preserves peer/rank/stage/status source
transport submission uses retained EntryId/action-u64/member-u32 and exact relocation values
segmented count-phase completion dominates every data-phase submit
segmented partial-peer failure stops remaining data descendants
sparse expert predicate reads exact bounded count only after count-complete
zero-count false branch reaches skipped_success/join with blob-open/load/copy/submit counters all zero
every nonzero expert in the finite wave executes; runtime cannot drop one to fit capacity
active count above declared max/wave envelope fails without partial wave issue
conditional join dominates combine and staging reuse for both active and skipped paths
mid-wave failure suppresses dependent experts/combine, safely drains issued peers and releases the envelope
cleanup/peer errors enter the bounded first/last/sample report while primary launch error remains primary
full_copy/page_cow state_snapshot_ready dominates every candidate-role slot consumer
state_group_publish waits for every member/rank/stage/iteration writer and status success
concurrent stale group-version CAS publishes zero members and preserves the complete current map
atomic group failure reclaims every unpublished member allocation/page/refcount without switching current
in-place exclusive group epoch plus durable dirty/fencing marker precede every member write
in-place failure/crash recovers whole-group poison or requires_reset before cleanup/attach returns
stream window verifies only its declared source range/chunks from one selected lease
stream copy completion dominates every consumer and last-consumer completion dominates staging reuse
stream copy/digest/early-EOF failure submits no consumer and never allocates full immutable backing
```

For each case assert exact recording-provider command order, iteration keys, stage overlap and skipped descendants. The executor test receives a preflight-generated plan and must not re-evaluate `BoundedCountExpr` or accept a separately expanded graph fixture.

Add a differential fixture that runs the same small graph through a reference full expansion only inside the test and through production `VerifiedGraphInstanceCursor`, requiring identical command/terminal/release order. Then raise a domain count to at least one million and assert exact total-work accounting, first/middle/last boundary keys and failure suppression while plan+cursor+live resource memory stays bounded by template size, symbolic peak and `maxLiveIterationSpan`, not total iterations.

Inject failures/cleanup errors across a million-instance graph and large rank set. The primary typed failure is always retained; secondary storage and diagnostic bytes remain under `RuntimeAdmissionPolicy.maxFailureRecords/maxDiagnosticBytes`, preserve deterministic bounded first/last/samples, and report exact observed/omitted totals plus bounded kind/domain owner counts and truncation markers. Changing sufficient diagnostic limits may change only retained explanatory records, never issue order, failure precedence, cleanup, outcome or semantic identity.

- [ ] **Step 2: Run tests and verify executor is absent**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
```

Expected before implementation: compile fails on `executeInvocation` and typed provider completion requests/observations.

- [ ] **Step 3: Implement event-driven command and completion execution**

Complete the typed `ProviderCommandRequest`, transfer/transport requests and raw receipts, `ProviderCompletionObservation` and status/cancel records in `RuntimeProviderCapabilities.h`. They are low-level mechanism values bound to one domain/provider owner/generation and contain no semantic choice: `EntrySubmission`, `CopySubmission`, `TransportSubmission` and completion/status expectations remain private repo-owned inputs assembled by the executor. The private execution view selects the already-bound transfer, ordinary-command or transport capability for the request's exact domain and validates every returned receipt/handle/observation owner/fence before the executor can advance the cursor. Missing transfer/transport capability, a receipt from another domain/generation, duplicate completion handle or a provider attempt to replace the selected source/range/binding member fails before cursor advancement.

`RuntimeSession::create` has already validated the graph template/count proof's complete capability set before realization and retained one immutable `InvocationControl` containing cancellation-state identity and exact absolute deadline. `executeInvocation` accepts no second control. It revalidates the canonical domain set/generations, creates one cursor from `VerifiedGraphInstancePlan`, and never reevaluates counts or builds an expanded edge list. Observe retained cancellation/deadline before first issue, before/after every ready wave, before each lazy object/resource/module/copy action, at every poll boundary and before blocking wait.

The cursor mechanically materializes only the bounded live horizon: deterministic ready instances, delta-derived predecessor counts, issued/completed watermarks, iteration-scoped resource instances and symbolic release-rule instances. In each scheduler round exhaust every cursor-ready nonblocking module first-use/load, receiver-ready, `submitCopy`, `submitEntry` and `submitTransport` node across domains/ranks/stages/iterations, report issue/terminal observations back to the cursor, then poll outstanding completions; call `waitAny` only when the cursor has no issue-ready node. `waitAny` receives `min(node absolute timeout, retained caller deadline)`. Stable ordering may order one ready wave but cannot block a domain/rank/stage/iteration whose peer/successor issue frontier is not reached. As completion watermarks advance, release iteration-scoped resource/window/event/command capacity through the cursor-instantiated retained release rules; never retain historic instance objects just for diagnostics. Map host command, async copy, device drain, DTE/collective wait, stage barrier, status observation, module release, `state_snapshot_ready`, `state_group_publish`, `state_group_poison` and cleanup template nodes to the context-owned execution view/final managers/registry. Provider API names remain inside `TxProviderCapabilities.cpp`.

When a predicate's retained count-complete instance reaches trusted terminal success, read exactly the declared typed scalar range from its realized control resource, validate representation/capacity/max and evaluate `bounded_count_nonzero` once. On false, the cursor marks every controlled copy/module-first-use/entry/data instance `skipped_success`, performs no blob open/verify/load/copy/submit under any verification/residency combination, and completes the explicit skip edge into conditional join. On true, acquire exact member subleases from the wave envelope, run `EagerActiveSet` verification and/or residency for the complete true active set before that wave's first dependent side effect, and mechanically expose all declared active nodes; `AtFirstUse`/`GraphLiveness` work remains at each declared first-use frontier. If actual active members exceed max/envelope, fail rather than dropping an expert. Conditional join becomes success only after the selected active terminal or false skip terminal and then releases combine/reuse through its symbolic terminal rule. Expert cache hits/eviction obey exact member/source/storage/domain keys and cannot change activation or wave order. A provider lacking conditional issue is rejected before first side effect, never converted to load-all.

For `ModuleResidencyMode::GraphLiveness`, a ready activation-aware module first-use node asks the residency manager for the already-reserved exact key; a miss first runs `AtFirstUse` verification when selected, then provider load and entry resolution before any dependent submit. `EagerActiveSet` verification may already have run, but load still consumes the retained same backing or reacquires/reverifies it. `ModuleResidencyMode::EagerActiveSet` uses this same proof/lease/resolve path for the unconditional active set before first issue and for a dynamic true active set at its predicate boundary. Build each `EntrySubmission` only by acquiring an `ExecutableUseLease` from the resulting module-token-bound `ExecutableHandle`, plus the exact `EntryInstanceId` and KAD-ordered typed address/scalar arguments. It has no raw handle pointer, symbol string, arbitrary `ModuleHandle`, member key override or provider lookup freedom. The executor's outstanding-command record owns the entire submission/use lease until the typed terminal completion and required status evidence for that command; the selected provider command capability may borrow it only for the duration of submission. The lease transitively retains the current package's `ModuleLease`/proof token, so moving/destroying/reallocating the original handle cannot dangle an asynchronous command and eviction cannot unload code early. Only the concrete instance produced by the module demand's verified `SymbolicTerminalInstanceRule`, plus zero outstanding use leases, can release the module; invocation-aggregate rules cannot fire on an earlier iteration. Submit return, local drain or one rank's completion cannot end another outstanding use. Failure/cancel/deadline cleanup drains or cancels the command before releasing its use lease.

For each ready streamed-window copy node, use its retained `abi::StreamWindowId`, shared `ResourceRealizationRecordKey`, exact source range/chunk set, staging `ResourceInstanceKey`/range and consumer set. Through private session/source access, reserve and move one read lease into the selected source's bounded `OpenedBlobLease`, reserve a work lease for the exact range verification workspace, and stream/verify that same opened object into a bounded host transfer buffer; there is no public `streamRangeAndVerify` call. The work lease lives through hash/chunk verification and any backing it authorizes; the read lease/handle lives until the final same-object stat/read/hash boundary, or through async completion only when the provider still consumes that backing. The separately capacity-charged transfer buffer and destination staging allocation remain through async copy completion. `CopySubmission` cannot contain a path, undeclared range or resident initialization request. The manifest copy-completion edge releases consumers; the verified symbolic last-consumer instance releases the staging range for its next window. Multiple windows may pipeline/double-buffer, but provider stream count cannot change coverage, ordering or buffer size, and no path may allocate/read/cache the full immutable payload as a fallback.

Build transport submissions and completion/status requests mechanically from the plan-retained `(EntryId, TransportActionId uint64, TransportBindingMemberId uint32, relocation values)` and its complete binding record. The executor/provider adapter may turn those facts into provider handles/control slots, but neither may choose another member, synthesize an endpoint/channel/FSM/receiver assignment, or reopen the projection search.

On failure, cancellation or caller deadline, stop dependent and same-domain not-yet-issued nodes across all remaining iterations, cancel or drive already-issued peers to a safe terminal wait, allow only required status/poison/cleanup nodes and preserve precedence: the first observed provider/device failure beats later control signals; otherwise `Cancelled`/`DeadlineExceeded` beats cleanup failures. A bounded collector retains the primary typed failure unconditionally, accounts every later rank/stage/iteration/cleanup observation without retaining O(total) objects, and emits deterministic first/last/samples plus exact omitted counts grouped by bounded failure-kind/domain owner. It truncates explanatory strings/records at the deployment limits with explicit markers; diagnostic pressure cannot stop cleanup, change precedence or alter outcome. A provider success return does not complete a node whose typed completion evidence is still pending. Node timeout is a closed typed `FailureKind::NodeTimeout` under `Outcome::Failed`, distinct from caller `DeadlineExceeded`; diagnostics do not drive behavior. `state_snapshot_ready` is emitted only after the entire full-copy/page-COW candidate map is ready and dominates every candidate-role slot. Atomic `state_group_publish` calls `publishGroup` with the captured base `StateGroupVersionId` only after every relevant member/rank/stage/iteration writer completion and status success; failure/cancellation/deadline/stale CAS publishes zero members, preserves the old complete map and reclaims the complete candidate/journal. An in-place member write cannot issue until the exclusive `StateGroupEpochLease` is held and `markGroupDirtyBeforeFirstWrite` has durably committed the matching fencing marker. `completeGroup` clears dirty only after all terminal success; any post-issue failure/cancellation/deadline cannot finish before durable whole-group `state_group_poison`, and crash/recovery observing dirty yields poison or `requires_reset`. Cancellation is scoped to this `InvocationId`; it releases only this invocation's leases and never cancels shared cache/residency users. Partial member publish/poison has no executor or provider-capability operation.

- [ ] **Step 4: Implement dry-run and dynamic TX provider adapters**

Dry-run consumes the same plan/cursor but writes through a bounded `RuntimeTraceSink`; it never builds a full trace vector. Emit canonical capacity domains and rank/projection mappings, selected variant, checked total work/symbolic peaks/max-live-span, both module policy axes, module first-use templates and typed symbolic terminal rules, each iteration domain's actual count, then stream each live node/edge key, `ResourceInstanceKey`, quant/storage/profile binding, stream window, state role, arena, transport member and executable-use-bound argument as the cursor advances. A caller may request bounded first/last/sample records plus exact totals; asking for every record streams with backpressure and a trace-byte limit, not O(total) retention. Finish with `execution_status: not_executed`; never report numeric success, locator, raw handle or quant formula. Synthetic no-card capability sets are installed into the same `WaferRuntimeServiceRegistry`, go through `discoverEnvironment`, exact-domain internal authority/context creation and generation revalidation, and cannot inject a verified inventory directly. `TxProviderCapabilities` dynamically loads participating providers and builds one complete domain set per domain; registry creation returns structured unsupported when any set lacks trustworthy completion/status. Fake-TX verifies orchestration/failure order only; numeric correctness remains a separate board gate.

- [ ] **Step 5: Run focused completion tests**

Run:

```bash
cmake --build build/wafer-dev --target WaferUnitTests -- -j128
build/wafer-dev/bin/WaferUnitTests \
  --gtest_filter='RuntimeCompletionTest.*:RuntimeProviderCapabilityTest.*'
```

Expected: issue-before-observe multi-rank progress, two-microbatch PP stage overlap, iteration failure/cancellation/deadline suppression and precedence, exact streamed-window/transport binding use, whole-group snapshot/CAS/exclusive-epoch terminal semantics and provider shielding all pass; dry-run never reports executed or numeric success.

- [ ] **Step 6: Commit**

```bash
git add include/Wafer/Runtime lib/Wafer/Runtime \
  unittests/Runtime/RuntimeCompletionTest.cpp \
  unittests/Runtime/RuntimeProviderCapabilityTest.cpp unittests/CMakeLists.txt
git commit -m "Execute typed runtime completion graphs"
```

### Task 7A: Typed Cross-Model State Migration

**Files:**
- Create: `schema/wafer/state_migration.proto`
- Modify: `schema/CMakeLists.txt`
- Create: `include/Wafer/ABI/StateMigration.h`
- Create: `lib/Wafer/ABI/StateMigration.cpp`
- Modify: `lib/Wafer/ABI/CMakeLists.txt`
- Create: `include/Wafer/Package/StateMigrationDelivery.h`
- Create: `lib/Wafer/Package/StateMigrationDelivery.cpp`
- Create: `include/Wafer/Runtime/StateMigration.h`
- Create: `lib/Wafer/Runtime/StateMigration.cpp`
- Modify: `include/Wafer/Runtime/RuntimeServiceContext.h`
- Modify: `lib/Wafer/Runtime/RuntimeServiceContext.cpp`
- Modify: `tools/wafer-run/wafer-run.cpp`
- Modify: `tools/wafer-run/CMakeLists.txt`
- Modify: `include/Wafer/Runtime/PersistentStateRegistry.h`
- Modify: `lib/Wafer/Runtime/PersistentStateRegistry.cpp`
- Modify: `lib/Wafer/Runtime/RuntimeCapacityReservation.cpp`
- Modify: `lib/Wafer/Runtime/CMakeLists.txt`
- Create: `unittests/ABI/StateMigrationPlanTest.cpp`
- Create: `unittests/Package/StateMigrationDeliveryTest.cpp`
- Create: `unittests/Runtime/StateMigrationTest.cpp`
- Create: `test/Integration/state-migration-no-card.test`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: immutable old/new `LoadedPackageMetadata` handles from trusted deliveries; metadata-verified record-21 plan; one bounded all-required/no-extra migration target-artifact metadata bundle whose heterogeneous sets all derive from that same verified ProgramDelivery; registry-produced environment inventory; one typed namespace, move-owned control/reservation mode and deployment limits. No caller persistent-scope map, registry snapshot, authority, context, runtime artifact session or backend.
- Produces: WCRE record 21 `StateMigrationPlanV1` and non-interchangeable `StateMigrationPlanId`; metadata-only placement plan with exact old/new/transform domains and owner-bound N:M scope relation; exact-domain all-set runtime bind and context-owned snapshot; one atomic batch of complete new-model state-group snapshots while preserving every old-model group; structured `state_model_mismatch` when no plan exists.

```cpp
namespace wafer::abi {
struct StateMigrationAdmissionLimitValues {
  // Every value is positive and checked; zero never means unbounded.
  std::uint64_t maxPlanBytes;
  std::uint32_t maxRecursionDepth;
  std::uint64_t maxMigrationComponents;
  std::uint64_t maxArtifactSets;
  std::uint64_t maxArtifactModules;
  std::uint64_t maxArtifactMembers;
  std::uint64_t maxScopeSelectors;
  std::uint64_t maxOldResolvedScopes;
  std::uint64_t maxNewResolvedScopes;
  std::uint64_t maxScopeRelationEdges;
  std::uint64_t maxStateGroups;
  std::uint64_t maxStateMembers;
  std::uint64_t maxPages;
  std::uint64_t maxChunks;
  std::uint64_t maxSingleStateBytes;
  std::uint64_t maxTotalStateBytes;
  std::uint64_t maxSingleReadBytes;
  std::uint64_t maxTotalReadBytes;
  std::uint64_t maxSingleCopiedBytes;
  std::uint64_t maxTotalCopiedBytes;
  std::uint64_t maxSingleTransformedBytes;
  std::uint64_t maxTotalTransformedBytes;
  std::uint64_t maxSingleWrittenBytes;
  std::uint64_t maxTotalWrittenBytes;
  std::uint64_t maxJournalBytes;
  std::uint64_t maxCandidateBytes;
  std::uint64_t maxCowMetadataBytes;
  std::uint64_t maxCowRefcountOperations;
  std::uint64_t maxTransformScratchBytes;
  std::uint64_t maxTransformOutputBytes;
  std::uint64_t maxCopyChunkBytes;
  std::uint64_t maxHostBuffers;
  std::uint64_t maxHostBufferBytes;
  std::uint64_t maxSimultaneousSourceLeases;
  std::uint64_t maxSimultaneousDestinationLeases;
  std::uint64_t maxWorkers;
  std::uint64_t maxConcurrentCopies;
  std::uint64_t maxConcurrentModules;
  std::uint64_t maxConcurrentCommands;
  std::uint64_t maxConcurrentEvents;
  std::uint64_t maxOpenFileDescriptors;
  std::uint64_t maxMigrationDurationMilliseconds;
  std::uint64_t maxCpuTimeMilliseconds;
};

class StateMigrationAdmissionLimits {
public:
  static llvm::Expected<StateMigrationAdmissionLimits>
  create(StateMigrationAdmissionLimitValues values);
  const StateMigrationAdmissionLimitValues &values() const;
private:
  explicit StateMigrationAdmissionLimits(
      StateMigrationAdmissionLimitValues values);
  StateMigrationAdmissionLimitValues values_;
};

class VerifiedStateMigrationPlan {
public:
  const StateMigrationPlanId &id() const;
  const ModelInterfaceSemanticId &oldModel() const;
  const ModelInterfaceSemanticId &newModel() const;
  llvm::ArrayRef<VerifiedStateMigrationComponent> components() const;
private:
  friend llvm::Expected<VerifiedStateMigrationPlan>
  verifyStateMigrationPlan(
      std::unique_ptr<state_migration::proto::StateMigrationPlanV1>,
      const StateMigrationAdmissionLimits &, CanonicalEncodingContext &);
  class Storage;
  explicit VerifiedStateMigrationPlan(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};
} // namespace wafer::abi

namespace wafer::runtime::detail {
class StateMigrationPlacementAccess;
} // namespace wafer::runtime::detail

namespace wafer::package {
class TrustedStateMigrationPlanRef;
class StateMigrationPlanPublicationRequest;
class PublishedStateMigrationPlan;

class VerifiedStateMigrationPlanDelivery {
public:
  const abi::VerifiedStateMigrationPlan &plan() const;
private:
  friend llvm::Expected<VerifiedStateMigrationPlanDelivery>
  loadAndVerifyStateMigrationPlan(
      const TrustedStateMigrationPlanRef &,
      const abi::StateMigrationAdmissionLimits &,
      artifact::ArtifactMetadataVerificationSession &);
  class Storage;
  explicit VerifiedStateMigrationPlanDelivery(
      std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};

llvm::Expected<VerifiedStateMigrationPlanDelivery>
loadAndVerifyStateMigrationPlan(
    const TrustedStateMigrationPlanRef &reference,
    const abi::StateMigrationAdmissionLimits &limits,
    artifact::ArtifactMetadataVerificationSession &verification);

llvm::Expected<PublishedStateMigrationPlan>
publishStateMigrationPlan(
    StateMigrationPlanPublicationRequest request,
    const abi::StateMigrationAdmissionLimits &limits,
    abi::CanonicalEncodingContext &canonicalEncoding);

class VerifiedMigrationArtifactBundle final {
public:
  VerifiedMigrationArtifactBundle(
      VerifiedMigrationArtifactBundle &&) noexcept;
  VerifiedMigrationArtifactBundle(
      const VerifiedMigrationArtifactBundle &) = delete;
  ~VerifiedMigrationArtifactBundle();
private:
  friend llvm::Expected<VerifiedMigrationArtifactBundle>
  loadAndVerifyMigrationArtifactBundle(
      const delivery::VerifiedProgramDelivery &,
      const VerifiedStateMigrationPlanDelivery &,
      const abi::StateMigrationAdmissionLimits &,
      artifact::ArtifactMetadataVerificationSession &);
  friend class runtime::detail::StateMigrationPlacementAccess;
  class Storage;
  explicit VerifiedMigrationArtifactBundle(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

llvm::Expected<VerifiedMigrationArtifactBundle>
loadAndVerifyMigrationArtifactBundle(
    const delivery::VerifiedProgramDelivery &transformDelivery,
    const VerifiedStateMigrationPlanDelivery &plan,
    const abi::StateMigrationAdmissionLimits &limits,
    artifact::ArtifactMetadataVerificationSession &verification);
} // namespace wafer::package

namespace wafer::runtime {
namespace detail {
class StateMigrationPlacementAccess;
class StateMigrationExecutionAccess;
class StateMigrationPlacementPlanStorage;
class StateMigrationPlacementRequestStorage;
class VerifiedStatePlacementInventoryStorage;
class RuntimeBoundStateMigrationPlacementStorage;
class RuntimeBoundMigrationArtifactBundleStorage;
class VerifiedStateMigrationScopeRelationStorage;
class PublishedStateMigrationSummaryStorage;
class StateMigrationResultStorage;
class BoundedStateMigrationFailureReportStorage;
} // namespace detail

class StateMigrationPlacementPlan;
class RuntimeBoundStateMigrationPlacement;
class RuntimeBoundMigrationArtifactBundle;
class StateMigrationSessionPlan;
class VerifiedStateMigrationScopeRelation;
class StateMigrationResult;

class StateMigrationEndpointSelection final {
public:
  static llvm::Expected<StateMigrationEndpointSelection> create(
      abi::TargetVariantId target,
      abi::ExecutableVariantId executable,
      abi::ProjectionSetId projection);
  StateMigrationEndpointSelection(
      StateMigrationEndpointSelection &&) noexcept;
  StateMigrationEndpointSelection &operator=(
      StateMigrationEndpointSelection &&) noexcept;
  StateMigrationEndpointSelection(
      const StateMigrationEndpointSelection &) = delete;
  ~StateMigrationEndpointSelection();
private:
  class Storage;
  explicit StateMigrationEndpointSelection(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

enum class StateMigrationTopologyLocality : std::uint8_t {
  SameCapacityDomain,
  SameProvider,
  ConnectedDomains
};

class StateMigrationDomainSelectionPolicy final {
public:
  static llvm::Expected<StateMigrationDomainSelectionPolicy> create(
      llvm::ArrayRef<TrustedRuntimeDomainConstraintRef> prioritizedAllowed,
      StateMigrationTopologyLocality locality,
      bool allowVerifiedCrossProviderTransfer);
  StateMigrationDomainSelectionPolicy(
      StateMigrationDomainSelectionPolicy &&) noexcept;
  StateMigrationDomainSelectionPolicy &operator=(
      StateMigrationDomainSelectionPolicy &&) noexcept;
  StateMigrationDomainSelectionPolicy(
      const StateMigrationDomainSelectionPolicy &) = delete;
  ~StateMigrationDomainSelectionPolicy();
private:
  class Storage;
  explicit StateMigrationDomainSelectionPolicy(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

class StateMigrationPlacementRequest final {
public:
  static llvm::Expected<StateMigrationPlacementRequest> create(
      package::LoadedPackageMetadata oldMetadata,
      package::LoadedPackageMetadata newMetadata,
      package::VerifiedStateMigrationPlanDelivery plan,
      std::optional<package::VerifiedMigrationArtifactBundle>
          transformArtifacts,
      StateNamespaceId stateNamespace,
      StateMigrationEndpointSelection oldEndpoint,
      StateMigrationEndpointSelection newEndpoint,
      StateMigrationDomainSelectionPolicy domainPolicy,
      RuntimeAdmissionPolicy admissionPolicy,
      abi::StateMigrationAdmissionLimits migrationLimits,
      RuntimeCapacityReservationMode capacityMode,
      InvocationControl control);
  StateMigrationPlacementRequest(StateMigrationPlacementRequest &&) noexcept;
  StateMigrationPlacementRequest(
      const StateMigrationPlacementRequest &) = delete;
  ~StateMigrationPlacementRequest();
  const VerifiedStateMigrationScopeRelation &scopeRelation() const;
private:
  friend class WaferRuntimeServiceRegistry;
  friend class detail::StateMigrationPlacementAccess;
  explicit StateMigrationPlacementRequest(
      std::unique_ptr<detail::StateMigrationPlacementRequestStorage> storage);
  std::unique_ptr<detail::StateMigrationPlacementRequestStorage> storage_;
};

class VerifiedStatePlacementInventory final {
public:
  VerifiedStatePlacementInventory(VerifiedStatePlacementInventory &&) noexcept;
  VerifiedStatePlacementInventory(
      const VerifiedStatePlacementInventory &) = delete;
  ~VerifiedStatePlacementInventory();
private:
  friend class WaferRuntimeServiceRegistry;
  friend class detail::StateMigrationPlacementAccess;
  explicit VerifiedStatePlacementInventory(
      std::unique_ptr<detail::VerifiedStatePlacementInventoryStorage> storage);
  std::unique_ptr<detail::VerifiedStatePlacementInventoryStorage> storage_;
};

class VerifiedStateMigrationScopeRelation final {
public:
  VerifiedStateMigrationScopeRelation(
      VerifiedStateMigrationScopeRelation &&) noexcept;
  VerifiedStateMigrationScopeRelation(
      const VerifiedStateMigrationScopeRelation &) = delete;
  ~VerifiedStateMigrationScopeRelation();
private:
  friend class detail::StateMigrationPlacementAccess;
  explicit VerifiedStateMigrationScopeRelation(
      std::unique_ptr<detail::VerifiedStateMigrationScopeRelationStorage> storage);
  std::unique_ptr<detail::VerifiedStateMigrationScopeRelationStorage> storage_;
};

class StateMigrationPlacementPlan final {
public:
  StateMigrationPlacementPlan(StateMigrationPlacementPlan &&) noexcept;
  StateMigrationPlacementPlan(const StateMigrationPlacementPlan &) = delete;
  ~StateMigrationPlacementPlan();
  llvm::ArrayRef<RuntimeCapacityDomainKey> requiredDomains() const;
  const VerifiedStateMigrationScopeRelation &scopeRelation() const;
private:
  friend llvm::Expected<StateMigrationPlacementPlan>
  preflightStateMigrationPlacement(
      VerifiedStatePlacementInventory,
      const VerifiedRuntimeEnvironmentInventory &);
  friend class WaferRuntimeServiceRegistry;
  friend class detail::StateMigrationPlacementAccess;
  explicit StateMigrationPlacementPlan(
      std::unique_ptr<detail::StateMigrationPlacementPlanStorage> storage);
  std::unique_ptr<detail::StateMigrationPlacementPlanStorage> storage_;
};

llvm::Expected<StateMigrationPlacementPlan>
preflightStateMigrationPlacement(
    VerifiedStatePlacementInventory statePlacement,
    const VerifiedRuntimeEnvironmentInventory &inventory);

namespace detail {
class StateMigrationPlacementAccess final {
public:
  static llvm::Expected<VerifiedStatePlacementInventory>
  snapshotStatePlacementInventory(
      WaferRuntimeServiceRegistry &registry,
      StateMigrationPlacementRequest request);
  static llvm::Expected<std::shared_ptr<WaferRuntimeServiceContext>>
  getOrCreateContext(
      WaferRuntimeServiceRegistry &registry,
      const StateMigrationPlacementPlan &plan);
private:
  StateMigrationPlacementAccess() = delete;
};
} // namespace detail

class RuntimeBoundStateMigrationPlacement final {
public:
  RuntimeBoundStateMigrationPlacement(
      RuntimeBoundStateMigrationPlacement &&) noexcept;
  RuntimeBoundStateMigrationPlacement(
      const RuntimeBoundStateMigrationPlacement &) = delete;
  ~RuntimeBoundStateMigrationPlacement();
private:
  friend llvm::Expected<RuntimeBoundStateMigrationPlacement>
  bindStateMigrationPlacementForRuntime(
      StateMigrationPlacementPlan,
      std::shared_ptr<WaferRuntimeServiceContext>);
  friend class detail::StateMigrationExecutionAccess;
  friend llvm::Expected<RuntimeBoundMigrationArtifactBundle>
  bindMigrationArtifactBundleForRuntime(
      RuntimeBoundStateMigrationPlacement &,
      std::optional<artifact::RuntimeArtifactVerificationSession>);
  explicit RuntimeBoundStateMigrationPlacement(
      std::unique_ptr<detail::RuntimeBoundStateMigrationPlacementStorage> storage);
  std::unique_ptr<detail::RuntimeBoundStateMigrationPlacementStorage> storage_;
};

llvm::Expected<RuntimeBoundStateMigrationPlacement>
bindStateMigrationPlacementForRuntime(
    StateMigrationPlacementPlan placement,
    std::shared_ptr<WaferRuntimeServiceContext> services);

class RuntimeBoundMigrationArtifactBundle final {
public:
  RuntimeBoundMigrationArtifactBundle(
      RuntimeBoundMigrationArtifactBundle &&) noexcept;
  RuntimeBoundMigrationArtifactBundle(
      const RuntimeBoundMigrationArtifactBundle &) = delete;
  ~RuntimeBoundMigrationArtifactBundle();
private:
  friend llvm::Expected<RuntimeBoundMigrationArtifactBundle>
  bindMigrationArtifactBundleForRuntime(
      RuntimeBoundStateMigrationPlacement &,
      std::optional<artifact::RuntimeArtifactVerificationSession>);
  friend class detail::StateMigrationExecutionAccess;
  explicit RuntimeBoundMigrationArtifactBundle(
      std::unique_ptr<detail::RuntimeBoundMigrationArtifactBundleStorage> storage);
  std::unique_ptr<detail::RuntimeBoundMigrationArtifactBundleStorage> storage_;
};

llvm::Expected<RuntimeBoundMigrationArtifactBundle>
bindMigrationArtifactBundleForRuntime(
    RuntimeBoundStateMigrationPlacement &placement,
    std::optional<artifact::RuntimeArtifactVerificationSession> verification);

class VerifiedStateMigrationRegistrySnapshot final {
public:
  VerifiedStateMigrationRegistrySnapshot(
      VerifiedStateMigrationRegistrySnapshot &&) noexcept;
  VerifiedStateMigrationRegistrySnapshot(
      const VerifiedStateMigrationRegistrySnapshot &) = delete;
  ~VerifiedStateMigrationRegistrySnapshot();
private:
  friend class WaferPersistentStateRegistry;
  friend class detail::StateMigrationExecutionAccess;
  friend llvm::Expected<VerifiedStateMigrationRegistrySnapshot>
  issueStateMigrationRegistrySnapshot(
      RuntimeBoundStateMigrationPlacement);
  class Storage;
  explicit VerifiedStateMigrationRegistrySnapshot(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

llvm::Expected<VerifiedStateMigrationRegistrySnapshot>
issueStateMigrationRegistrySnapshot(
    RuntimeBoundStateMigrationPlacement placement);

class StateMigrationSessionPlan final {
public:
  StateMigrationSessionPlan(StateMigrationSessionPlan &&) noexcept;
  StateMigrationSessionPlan(const StateMigrationSessionPlan &) = delete;
  ~StateMigrationSessionPlan();
private:
  friend llvm::Expected<StateMigrationSessionPlan>
  preflightStateMigrationExecution(
      VerifiedStateMigrationRegistrySnapshot,
      RuntimeBoundMigrationArtifactBundle);
  friend llvm::Expected<StateMigrationResult>
  executeStateMigration(StateMigrationSessionPlan);
  class Storage;
  explicit StateMigrationSessionPlan(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

llvm::Expected<StateMigrationSessionPlan>
preflightStateMigrationExecution(
    VerifiedStateMigrationRegistrySnapshot snapshot,
    RuntimeBoundMigrationArtifactBundle artifacts);

struct PublishedStateGroupVersion {
  StateGroupKey key;
  StateGroupVersionId version;
};

enum class StateMigrationFailureKind : std::uint8_t {
  Placement,
  StateConflict,
  Backing,
  Artifact,
  Provider,
  Integrity,
  AtomicPublish,
  Cancellation,
  Deadline,
  InternalContract
};

enum class StateMigrationFailurePhase : std::uint8_t {
  PlacementInventory,
  PlacementPreflight,
  RuntimeBinding,
  RegistrySnapshot,
  SourceLease,
  CopyOrCow,
  Transform,
  Attestation,
  BatchCompareAndSwap,
  Cleanup
};

struct StateMigrationFailure {
  StateMigrationFailureKind kind;
  StateMigrationFailurePhase phase;
  abi::StateMigrationPlanId plan;
  std::optional<StateGroupKey> oldGroup;
  std::optional<StateGroupKey> newGroup;
  std::optional<PersistentStateScopeKey> oldScope;
  std::optional<PersistentStateScopeKey> newScope;
  std::optional<abi::TargetArtifactMemberKey> artifactMember;
  std::optional<RuntimeCapacityDomainKey> domain;
  std::string diagnostic;
  bool diagnosticTruncated;
};

class BoundedStateMigrationFailureReport final {
public:
  BoundedStateMigrationFailureReport(
      BoundedStateMigrationFailureReport &&) noexcept;
  BoundedStateMigrationFailureReport(
      const BoundedStateMigrationFailureReport &) = delete;
  ~BoundedStateMigrationFailureReport();
  llvm::ArrayRef<StateMigrationFailure> retainedFirstLastSamples() const;
  std::uint64_t totalObserved() const;
  std::uint64_t totalOmitted() const;
  std::uint64_t retainedDiagnosticBytes() const;
private:
  friend class detail::StateMigrationExecutionAccess;
  explicit BoundedStateMigrationFailureReport(
      std::unique_ptr<detail::BoundedStateMigrationFailureReportStorage> storage);
  std::unique_ptr<detail::BoundedStateMigrationFailureReportStorage> storage_;
};

class PublishedStateMigrationSummary final {
public:
  llvm::ArrayRef<PublishedStateGroupVersion> publishedGroups() const;
private:
  friend class detail::StateMigrationExecutionAccess;
  explicit PublishedStateMigrationSummary(
      std::shared_ptr<const detail::PublishedStateMigrationSummaryStorage> storage);
  std::shared_ptr<const detail::PublishedStateMigrationSummaryStorage> storage_;
};

class StateMigrationResult final {
public:
  enum class Outcome : std::uint8_t {
    Succeeded,
    Failed,
    Cancelled,
    DeadlineExceeded
  };
  Outcome outcome() const;
  const PublishedStateMigrationSummary *published() const;
  const StateMigrationFailure *primaryFailure() const;
  const BoundedStateMigrationFailureReport *secondaryFailures() const;
  StateMigrationResult(StateMigrationResult &&) noexcept;
  StateMigrationResult &operator=(StateMigrationResult &&) noexcept;
  StateMigrationResult(const StateMigrationResult &) = delete;
  ~StateMigrationResult();
private:
  friend llvm::Expected<StateMigrationResult>
  executeStateMigration(StateMigrationSessionPlan);
  explicit StateMigrationResult(
      std::unique_ptr<detail::StateMigrationResultStorage> storage);
  std::unique_ptr<detail::StateMigrationResultStorage> storage_;
};

llvm::Expected<StateMigrationResult>
executeStateMigration(StateMigrationSessionPlan plan);
} // namespace wafer::runtime
```

- [ ] **Step 1: Add schema, identity, trust, and atomicity failures**

`StateMigrationPlanTest` requires `state_migration.proto` to own exactly record 21/domain `wafer.state-migration-plan.v1` and claimed `StateMigrationPlanId`; `semantic_identity.proto` only owns the registered wrapper/domain enum and cannot redeclare the record. Assert `StateMigrationPlanId` is nonconstructible/noncomparable with every other digest ID. The plan contains typed old/new `ModelInterfaceSemanticId` and complete canonical old/new typed state descriptors: semantic type, bounded shape, access, alias/update, capacity, page geometry, storage and declared persistent scope. `StateMigrationScopeSelectorV1` is the ABI closed union `model_shared | component(DistributedProgramSemanticId, ComponentId) | execution_instance(ExecutionInstanceId)` and never serializes runtime `StateNamespaceId`, `PersistentStateScopeKey` or transient `ScopeInstanceId`. `full_copy` and `page_cow_snapshot` require canonical all-and-only explicit `StateMigrationScopePairV1` records plus bijective group/member maps; independently sorted old/new selector sets are never implicitly zipped. `trusted_transform` instead declares a finite bounded set of typed N:M components: every KAD input/output slot explicitly references its selector and component role, every old input group/member/value relation is read-only and covered exactly where declared, every new group/member/value has exactly one verified producer, outputs do not overlap, and partition/concat/split/index relations close over the component. There is no name/ordinal map, generic descriptor fingerprint, implicit/partial relation, unbounded split/merge, callback, raw function symbol or diagnostic alias.

Require the closed mode union:

```text
full_copy
page_cow_snapshot
trusted_transform(bounded components, TargetArtifactMemberKey/KAD per component,
                  exact N:M input/output slots and partition/concat/split/index,
                  scratch, completion and numeric/integrity policy)
```

Reject unknown fields/version/mode, missing/extra/duplicate group/member/selector/pair, nonbijective or implicitly zipped full-copy/COW relations, transform component/selector/slot input-output coverage hole/overlap/unbounded relation, cross-owner ID, descriptor field mismatch against either manifest, scope/policy mismatch, ambiguous alias/update, unsupported COW/refcount/fence, transform entry not in the trusted artifact bundle, KAD slot mismatch, transform write access to old state, undeclared scratch/completion or a claimed plan ID mismatch. Artifact references are canonical `(TargetVariantId, TargetArtifactSetId, TargetArtifactMemberKey)` triples; every required set/member appears once, no extra appears and all set roots derive from the same `VerifiedProgramDelivery`. Record construction and verification require the caller's `CanonicalEncodingContext`; limit-1 record/set/scope/scratch cases fail without partial ID/proof.

Delivery tests require a deployment-owned `TrustedStateMigrationPlanRef` binding capability acquisition, exact plan size/content digest and expected `StateMigrationPlanId`. The sole production loader is exactly `loadAndVerifyStateMigrationPlan(const TrustedStateMigrationPlanRef &, const abi::StateMigrationAdmissionLimits &, artifact::ArtifactMetadataVerificationSession &)`: it checks exact owner-backed bytes, limits, unknown fields and record-21 identity while private session access lends the retained canonical context and issues metadata read/work leases. There is no raw path/root, independent artifact limits/context, default session or unbounded overload. Concurrent plan/package/target metadata parses share the Artifact-metadata typed child ledger and common host parent but consume no device authority, service context or invocation capacity. `loadAndVerifyMigrationArtifactBundle` derives every required target ref from one verified ProgramDelivery, reuses that same metadata session and returns only canonical loaded target metadata plus inert descriptors; full-copy/COW require the artifact bundle absent, trusted-transform requires exact all-and-only coverage. Limit+1 and concurrent-load failures occur with zero generated-message allocation beyond the preflight ledger, zero actual-module open and no leaked reservation. Whole-root/path/old-new plan swaps and debug accept-any results cannot construct production proofs. The publisher/compiler side still accepts only old/new verified manifests plus explicit typed mapping/mode, transform relations and an explicit caller-owned canonical context; it cannot infer mappings from names, synthesize scratch or import raw JSON callbacks.

Runtime negatives cover no plan, wrong old/new metadata/plan/artifact-delivery owner, unknown domain-constraint ID, mixed catalog owners/generations, a catalog/ref from another authenticated inventory generation, caller expected-version/new-absent/scope-key claims, stale inventory, exact-domain mismatch, context from another placement, partial multi-set bind, stale registry snapshot epoch/fence, stale old group version, missing/poisoned old group, already-existing new group without the registry's reset precondition, provider/domain/service generation mismatch, copy/COW/transform failure including a second/later group, cancellation/deadline at every boundary, crash after journal/candidate/backing/transform but before batch CAS, stale fencing, concurrent migration to any same new group and counter exhaustion. Placement preflight has zero authority/registry/blob/provider effects. Only the exact context's final registry may internally issue `VerifiedStateMigrationRegistrySnapshot` from the placement plan's complete resolved relation; it is never a request/caller parameter. Placement move-owns one control/mode/policy/limits, bind retains them, execution preflight seals the one-use snapshot/context into `StateMigrationSessionPlan`, and execute accepts no replacement. Every failure preserves all old current groups, publishes zero new groups/members, reclaims the migration-wide candidate/journal and returns the typed cause. Reusing namespace/alias never triggers migration.

Exercise every positive `StateMigrationAdmissionLimits` field at limit-1/limit/limit+1, including component/artifact-set/module/member/selector/pair/resolved-old-scope/resolved-new-scope/relation-edge/group/member/page/chunk counts; single/total state, read, copied, transformed and written bytes; journal/candidate/COW metadata/refcount/transform scratch/output; source/destination leases; FD/buffer/worker/module/copy/command/event concurrency; wall and CPU time. Checked aggregate overflow fails before authority/registry lease/open/allocation/journal counters change. Add a 100GB-class sparse/fake-backing multi-group plan that cannot fit in host RAM: sufficient policies produce the same new state relation across worker/chunk choices, while no full group/backing mapping or copy is materialized. Compile-time tests require artifact metadata bundle, placement plan, bound placement, registry snapshot and session plan to be move-only/non-aggregate/private-construction; raw packages/targets, caller scope maps/snapshots/contexts/backends, callbacks and replacement sessions have no overload. Wrong delivery/set/member/KAD/selector owner, extra/missing artifact set, cross-delivery splice or stale runtime verification session fails before actual module open.

- [ ] **Step 2: Implement record 21 and trusted delivery**

Generate `WaferStateMigrationProto` by importing only shared `semantic_identity.proto`, `kernel_abi.proto` and necessary target-artifact typed keys. Define the complete canonical old/new state descriptor, selector, explicit scope-pair and transform slot-role relation directly as record-21 nested typed fields; do not import/embed `package_manifest.proto`, runtime keys or create an ABI-to-package dependency. Old/new `VerifiedPackageManifest` objects are external C++ verifier join inputs. Add descriptor-set record/domain/noninterchangeability gates. Before generated parsing, the production loader builds the fixed descriptor/category plan plus validated migration budget and delegates the exact owner-backed plan bytes solely to WaferABI `detail::preflightProtoWire`; only the returned schema/backing-bound proof may enter generated parsing, and `detail::verifyParsedProtoMatchesPreflight` must prove exact tag/wire/length/unknown/depth/component/group/member/selector/pair/string/descriptor/count/span agreement before adoption. Limit+1 tests instrument the generated arena/message allocator and require zero generated-message allocation before a successful preflight; the package wrapper must not implement a second wire parser. `verifyStateMigrationPlan` move-adopts one accepted message, validates mappings/descriptors/artifact refs, streams record 21 through the session-lent context and retains immutable canonical backing. `StateMigrationDelivery.cpp` implements bounded deterministic serialization, exact content digest, trusted reference loader and publisher. The production loader obtains immutable bytes, metadata read/work reservations and canonical context only through the caller's live `ArtifactMetadataVerificationSession`; the publisher/compiler path uses its explicit canonical context. It is runtime-safe and has no MLIR, raw path or host callback.

- [ ] **Step 3: Implement placement inventory and the two preflight stages**

`StateMigrationPlacementRequest::create` move-retains the two immutable metadata handles, verified plan, mode-appropriate optional `VerifiedMigrationArtifactBundle`, namespace, explicit old/new typed `(TargetVariantId, ExecutableVariantId, ProjectionSetId)` endpoints, validated `StateMigrationDomainSelectionPolicy`, `RuntimeAdmissionPolicy`, migration limits, reservation mode and one control. Before the inventory is moved into runtime-service bootstrap, the trusted host copies the co-issued `TrustedRuntimeDomainConstraintCatalog`; it resolves deployment-control-plane typed `RuntimeDomainConstraintId` values through that catalog and gives the policy one duplicate-free `prioritizedAllowed` vector of the resulting owner-backed refs. Every allowed constraint appears exactly once and vector order is the deterministic priority. The policy rejects mixed catalog owners/generations and carries locality and verified cross-provider-transfer permission, but no raw provider/device/generation or compatibility/capacity claim. The factory is the sole owner of package/migration semantic joins: it validates every descriptor, selector, explicit scope pair or transform selector-role edge against both manifests, resolves and retains the nonserialized `VerifiedStateMigrationScopeRelation` plus all canonical old/new state keys. It rejects implicit endpoint selection, invalid policy, caller runtime scope keys and already-cancelled/expired control. Runtime and migration overlapping operational limits take a checked per-field minimum. Full-copy/COW have zero transform artifact/module demand, so the retained module verification/residency enum values are observationally irrelevant and canonicalized for those modes; trusted-transform alone applies them to the bound transform module DAG. The service registry never parses selectors or interprets package semantics.

Before authority/context exists, `detail::StateMigrationPlacementAccess::snapshotStatePlacementInventory(registry, std::move(request))` consumes the request exactly once, checks its retained control before/after the one state-placement catalog query, and revalidates every domain-constraint ref's catalog owner/generation against the registry-retained authenticated inventory before considering placement facts. It uses only its sealed old keys plus the trusted durable-state metadata service/provider inventory owner facts, and returns `VerifiedStatePlacementInventory` that owns that same request/relation/policies/control together with current provider owner/generation/domain/placement and portable/reimport constraints. It performs no capacity reservation, execution lease, backing/blob open or provider command; a request cannot issue two inventories or be re-paired later. `preflightStateMigrationPlacement(std::move(statePlacement), environmentInventory)` combines the retained explicit policies/relation/state placement/artifact metadata with environment inventory, rechecks the constraint owner/generation against that verified discovery snapshot and produces one exact canonical old/new/transform domain set. It verifies all required/no-extra target sets/members and checked symbolic capacity, but obtains no authority/context/snapshot and opens no backing/blob. The plan binds state/environment inventory owners and generations; it cannot replace policy/control or silently choose another target/projection/domain.

Only then may `detail::StateMigrationPlacementAccess::getOrCreateContext(registry, placementPlan)` internally acquire/reuse exact-domain authority; `bindStateMigrationPlacementForRuntime(std::move(plan), context)` creates the owner-bound placement. For transform mode, context-private budget access creates one runtime artifact session and `bindMigrationArtifactBundleForRuntime(boundPlacement, std::move(session))` delegates the whole canonical heterogeneous set vector to `RuntimeTargetArtifactSetsBinder`, producing distinct `RuntimeBoundMigrationArtifactBundle` all-or-none; full-copy/COW produce a sealed empty bound bundle and require no runtime artifact session. Binding revalidates plan/context/session owners but opens no module. After artifact binding, `issueStateMigrationRegistrySnapshot(std::move(boundPlacement))` move-consumes the placement exactly once and asks only that context's final registry for all canonical old/new keys in the relation; the returned snapshot owns the placement/context/plan, so another snapshot cannot be issued or re-paired. It revalidates the earlier state-placement inventory owner/generation, current backing placement/portability, complete old versions/status/member maps, new absent/reset predicates and provider/service fences; stale inventory fails before backing/provider effects. Finally `preflightStateMigrationExecution(std::move(snapshot), std::move(boundArtifacts))` computes the checked `StateMigrationResourcePlan` and seals the one-use snapshot/context/artifact session/control into `StateMigrationSessionPlan`. Full-copy requires explicit logical/storage/page/scope bijection and bounded chunk flow; page-COW additionally requires fenced COW/refcount; transform resolves every bound set/member/KAD/profile/slot. Neither preflight acquires state execution leases, opens backing/module bytes, allocates, journals or calls providers.

- [ ] **Step 4: Implement journaled execution and one new-group CAS**

Use the plan-retained service context's authority-bound coordinator and acquire one multi-domain reservation before backing/provider effects. Execution returns the snapshot/resource-plan token to that same final `WaferPersistentStateRegistry`, which atomically acquires fenced read leases for every recorded old complete group, rechecks all old versions/new predicates/registry epoch/fence/provider generations, and releases all leases with zero side effects if any differs. It then opens one migration transaction and durable journal covering the complete relation, captured old versions, every new absent/reset precondition, plan ID and provider fences, and creates all new groups/member maps as unpublished candidates. `full_copy` uses bounded same-handle `readAt`/copy/hash/attest chunks capped by copy-chunk/host-buffer/source-destination-lease/FD/worker/command budgets and never maps an entire group into host RAM; COW and transform obey the same metadata/scratch/concurrency accounting. Trusted transforms use the retained runtime-bound artifact bundle/session, exact module view/backing, context-owned final module manager and ordinary executable-use/completion machinery; transforms cannot write old backing. After every group/member attests, call one registry `batchCompareAndSwapMigration`: atomically recheck all old versions, all new preconditions and fences, mint versions and publish every complete new group map together. There is no per-group CAS or compensation path. Success leaves all old groups/backings valid; retirement is separate. Any second/later group failure, limit/cancellation/deadline or pre-batch-CAS crash reclaims the entire candidate/journal and publishes zero new groups. Recovery distinguishes one pre-CAS orphan transaction from one fully committed batch without handle guessing. `StateMigrationResult` is private-construction and closed: success owns a canonical duplicate-free `PublishedStateGroupVersion {key, version}` vector from the batch receipt; failure/cancel/deadline owns `StateMigrationFailure` plus `BoundedStateMigrationFailureReport`, never invocation failure records. Migration failures use closed kind/phase, plan ID and optional typed old/new group/scope, artifact member and domain; they contain no component ordinal/name. Old-state preservation is an execution invariant for every arm, never a mutable result flag. Tests reject parallel key/version arrays, reordered/duplicate/missing published keys, invocation entry/iteration fields and impossible outcome/payload combinations.

- [ ] **Step 5: Add explicit host entry and integration gate**

Add a distinct `wafer-run migrate-state` flow taking trusted old/new/transform ProgramDelivery refs, trusted migration-plan ref, namespace, typed old/new endpoint selections, trusted domain-selection policy, migration limits, reservation mode and deadline. Authenticated deployment bootstrap first returns its paired inventory/catalog; the host copies the catalog, resolves only typed `RuntimeDomainConstraintId` policy inputs, and then moves the inventory into `VerifiedRuntimeServiceBootstrap`. It next uses Delivery metadata and Artifact metadata sessions issued from their distinct typed children of the same `HostVerificationRegistry`; `loadPackageBundleMetadata` loads old/new manifests, `loadAndVerifyStateMigrationPlan` loads record 21 and `loadAndVerifyMigrationArtifactBundle` metadata-loads all required/no-extra transform sets from one delivery. None can open actual modules or acquire device authority. Build the single `StateMigrationPlacementRequest`, then run registry `discoverEnvironment`, move-consume the request into `snapshotStatePlacementInventory`, and run pure placement. Both joins revalidate the refs' catalog owner/generation against the registry/environment inventories. Only its exact-domain plan may cause internal context/authority creation. Bind the placement and explicit `RuntimeBoundMigrationArtifactBundle` all-or-none, using a context-issued runtime artifact session only for transform mode; issue the exact-context final registry snapshot, run execution preflight and call `executeStateMigration(plan)` with no backend/context/snapshot replacement. The returned non-aggregate `StateMigrationResult` has a success arm carrying canonical `(StateGroupKey, StateGroupVersionId)` records or a failed/cancelled/deadline arm carrying bounded typed failures; preserving every old group is an invariant of every arm, not a caller boolean. Only after success may the host start a separate normal new-model invocation. Normal `InvocationRequest`/state attach has no migration plan/snapshot field and returns `state_model_mismatch` on owner mismatch; migration is never an implicit fallback.

The no-card integration builds two genuinely different model-interface identities with compatible multi-member state, publishes a typed multi-group full-copy plan and migrates prefill state to the new model. Separate fixtures exercise page-COW and a verified transform entry. A 100GB-class sparse/fake backing proves bounded host memory/chunking and every migration limit without real allocation. Assert one batch-CAS new-group visibility, old group continued attach/read, exact descriptor/member mappings, no partial publish under second/later-group failure, limit/cancel/crash, and no callback/name matching.

- [ ] **Step 6: Run focused gates**

```bash
cmake --build build/wafer-dev --target WaferUnitTests wafer-run -- -j128
build/wafer-dev/bin/WaferUnitTests \
  --gtest_filter='StateMigrationPlanTest.*:StateMigrationDeliveryTest.*:StateMigrationTest.*'
<configured-lit> -sv \
  build/wafer-dev/test/Integration/state-migration-no-card.test
python3 tools/check_deps.py
```

Expected: all three modes, exact delivery/identity and 100GB-class bounded-memory/limit gates pass; pure preflight has zero lease/open/allocate/journal side effects, execution acquires all old leases once, and every failure/limit/cancel/crash preserves old state and publishes no partial new group; missing plan remains `state_model_mismatch`.

- [ ] **Step 7: Commit**

```bash
git add schema/wafer/state_migration.proto schema/CMakeLists.txt \
  include/Wafer/ABI/StateMigration.h lib/Wafer/ABI/StateMigration.cpp \
  lib/Wafer/ABI/CMakeLists.txt include/Wafer/Package/StateMigrationDelivery.h \
  lib/Wafer/Package/StateMigrationDelivery.cpp \
  include/Wafer/Runtime/StateMigration.h lib/Wafer/Runtime/StateMigration.cpp \
  include/Wafer/Runtime/RuntimeServiceContext.h \
  lib/Wafer/Runtime/RuntimeServiceContext.cpp \
  tools/wafer-run/wafer-run.cpp tools/wafer-run/CMakeLists.txt \
  include/Wafer/Runtime/PersistentStateRegistry.h \
  lib/Wafer/Runtime/PersistentStateRegistry.cpp lib/Wafer/Runtime/CMakeLists.txt \
  lib/Wafer/Runtime/RuntimeCapacityReservation.cpp \
  unittests/ABI/StateMigrationPlanTest.cpp \
  unittests/Package/StateMigrationDeliveryTest.cpp \
  unittests/Runtime/StateMigrationTest.cpp \
  test/Integration/state-migration-no-card.test unittests/CMakeLists.txt
git commit -m "Add typed cross-model state migration"
```

### Task 8: No-Card Mainline, Legacy Removal, and Vertical Gate

**Files:**
- Modify: `tools/wafer-opt/wafer-opt.cpp`
- Modify: `tools/wafer-opt/CMakeLists.txt`
- Modify: `tools/wafer-run/wafer-run.cpp`
- Modify: `tools/wafer-run/CMakeLists.txt`
- Modify: `include/Wafer/Runtime/HostRuntime.h`
- Modify: `lib/Wafer/Runtime/HostRuntime.cpp`
- Delete: `tools/wafer_package_metadata.py`
- Delete: `tools/wafer_export_package_metadata.py`
- Delete: `tools/wafer_runtime_adapter.py`
- Delete: `test/Runtime/wafer_runtime_adapter_test.py`
- Modify: `test/CMakeLists.txt`
- Rewrite: `unittests/Runtime/HostRuntimeTest.cpp`
- Rewrite: `test/Runtime/wafer-run.test`
- Modify: `test/Tools/wafer-package-convert.test`
- Create: `test/Integration/package-runtime-no-card.test`
- Create: `test/Integration/package-runtime-stateful-streaming-no-card.test`
- Create: `test/Integration/package-runtime-pp-no-card.test`
- Create: `test/Integration/package-runtime-sparse-moe-no-card.test`
- Create: `test/Integration/package-runtime-capacity-authority-no-card.test`
- Delete: `test/Tools/wafer-package-metadata.test`
- Delete: `test/Tools/wafer-export-package-metadata.test`
- Delete: `test/Tools/Inputs/package-export-full-instr.txt`
- Delete: `test/Tools/Inputs/package-export-instr.txt`
- Delete: `test/Tools/Inputs/package-export-kernel.ll`
- Delete: `test/Tools/Inputs/package-export-model-interface.json`
- Delete: `test/Tools/Inputs/package-export-pipeline-model-interface.json`
- Move: `test/Tools/valid-model-package.json` to `test/Tools/Inputs/package-v2/valid-model-package.json`
- Move: `test/Tools/invalid-completion-package.json` to `test/Tools/Inputs/package-v2/invalid-completion-package.json`
- Move: `test/Tools/invalid-empty-modules-package.json` to `test/Tools/Inputs/package-v2/invalid-empty-modules-package.json`
- Move: `test/Tools/invalid-model-package.json` to `test/Tools/Inputs/package-v2/invalid-model-package.json`
- Move: `test/Tools/invalid-module-format-package.json` to `test/Tools/Inputs/package-v2/invalid-module-format-package.json`
- Move: `test/Tools/invalid-resident-constant-package.json` to `test/Tools/Inputs/package-v2/invalid-resident-constant-package.json`
- Move: `test/Tools/invalid-workspace-package.json` to `test/Tools/Inputs/package-v2/invalid-workspace-package.json`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`

**Interfaces:**
- Consumes: user-level `stablehlo-to-executable` driver output from prior plans; compiler-only package assembly; Delivery/Artifact metadata verification children from one host ledger; authenticated `VerifiedRuntimeServiceBootstrap`; registry-owned dry-run/recording/TX provider capability adapters.
- Produces: `wafer-opt --program-pipeline=stablehlo-to-executable` 在一个 `ProgramOutputTransaction` 内stage executable/target/package并输出单一trusted program-delivery commit/reference；`wafer-run --program-delivery=<trusted-ref> --entrypoint=<diagnostic-alias-or-typed-id> --deadline-ms=<positive-duration> --parse-limit=<kind=value>... --admission-limit=<kind=value>... --module-verification=eager_active_set|at_first_use --module-residency=eager_active_set|graph_liveness --backend=dry-run|fake-tx|tx`; static、stateful prefill/decode + bounded streaming和two-microbatch PP real program-chain no-card gates. Alias只在CLI层唯一解析为typed `ModelEntrypointId`，duration在调用前一次转换为nonserialized steady-clock absolute deadline，SIGINT只触发该invocation的cancellation source。

- [ ] **Step 1: Rewrite CLI tests to demand typed package behavior**

`test/Runtime/wafer-run.test` must check:

```text
wrong exact manifest blob digest fails before parse
delivery and artifact metadata sessions share one host parent while consuming zero device authority
package metadata loads index/manifest before exact-domain selection and opens zero module/payload blobs
preflight exact domains are the only domains internally authorized; caller authority/context/session injection has no API
one-way RuntimeBoundPackage binding rejects owner/generation mismatch and exposes no partial source
delivery recursion/count/AST/depth limit and invocation graph/resource/byte/liveness admission failures have zero object/provider side effects
manifest cannot override deployment limits; raising policy accepts identical semantic selection without truncation
wrong WCRE PackageManifestId fails after semantic validation
v2 JSON passed to normal runtime fails as unsupported package format
unknown/cross-interface ModelEntrypointId, ambiguous alias and static EntryId/symbol request fail before object/provider side effects
prefill/decode aliases resolve to distinct typed IDs and exact invocation root/terminal subgraphs
missing target/projection/shape compatibility fails before allocation
raw/caller-forged environment digest is not an accepted input
unknown/duplicate/missing/zero CLI limit kind is rejected before bundle open
low-precision profile/environment/artifact/KAD mismatch fails before blob open/allocation/load
missing completion capability fails before allocation/module load
dry-run prints typed ExecutionInstance coordinates and exact SlotId -> ResourceInstanceKey arguments
dry-run prints retained ResourceRealization and TransportAction/member/relocation selections
dry-run prints state-group key/member/current-candidate-in-place roles and whole-group terminals
dry-run prints every StreamWindowId, exact chunk/source/staging ranges, consumers and reuse edge
dry-run prints each IterationDomainId actual count, exact total/peak/live-span proof and bounded streamed node/edge iteration trace
dry-run streams each ActivationPredicateId state, finite expert-wave envelope, true/false skipped-success and conditional join without retaining a full trace
dry-run streams bounded completion-DAG records with exact totals and not_executed
fake-tx executes the same plan and records typed commands
fake-tx issues every ready rank/stage command before its first poll/wait
pre-issue cancellation has zero object/provider side effects and reports cancelled
caller deadline during PP/stream copy reports deadline_exceeded, uses safe waits and preserves failure precedence
wrong ELF machine/ISA/MABI/export/undefined/note/KAD/fingerprint, even after recomputed delivery digests, fails before provider allocation/load
second selected module with a bad note leaves allocation/load counters at zero
descriptor/module content mismatch publishes no module or executable handle
cross-module entry/member substitution fails before submit
opened blob truncation/mutation fails while locator replacement cannot redirect consumption
mutation after module preverification is caught by load-time reacquire/reverification, cleans realized resources and publishes no handle
all four module verification/residency combinations preserve the same entry DAG, enforce their verification/provider-code peaks and keep predicate-false effects at zero
same-iteration/final-tail/invocation-terminal release rules preserve capacity through the exact concrete last use without O(total iterations) vectors
provider generation change invalidates module/weight leases even when device ID and target fingerprint match
stateful two-invocation run reuses the complete published group snapshot and resident lease, not workspace
streamed payload larger than resident capacity uses bounded staging only and has no full backing/cache fallback
segmented count/displacement/peer-total/capacity mismatch fails before submit
PP count 2 shows stage overlap and injected middle failure suppresses later dependent iterations
sparse MoE count phase reads only typed bounded count scalars; zero-count experts perform zero blob/module/copy/submit effects
sparse MoE nonzero experts execute every declared active member in finite compiler-planned waves and combine only after conditional joins
sparse MoE active-count overflow, wrong count owner/range/type, unsupported conditional provider and mid-wave failure fail closed without load-all fallback
two-process exclusive-local/central-authority fixtures never exceed shared multi-domain capacity and stale fences cannot submit/release/publish
shared deployment with no reservation authority is unsupported before blob open/provider effects
compiler process and wafer-run process compute the same PackageManifestId
second target set or package-last staging failure exposes no program/package alias or index
crash before outer commit and commit-record CAS loss leave only unreachable roots rejected by production loader
whole-root, old/new and cross-tenant bundle substitutions fail against the deployment reference before other blob open
```

The static integration test must export a real reference program, run the named production driver through committed executable, every complete target artifact set and private package attachment, then make them visible through one unified-root rename or trusted `ProgramDeliveryCommitRecord`. It obtains a deployment-owned `VerifiedPackageBundleReference` from that commit, resolves one diagnostic alias to its typed `ModelEntrypointId`, then runs verified-environment no-card preflight and fake-provider orchestration. It also injects second-target, package-last, precommit-crash and commit-CAS failures and proves no accepted alias/index/reference exists. The stateful/streaming test compiles a real program with distinct typed prefill/decode entrypoints, one multi-member persistent consistency group, both full-copy and page-COW fixtures, a resident immutable shard, typed low-precision descriptor/profile fixtures and a streamed immutable payload larger than the configured resident arena. It invokes the exact prefill then decode mappings and proves their IO/state/root contracts, snapshot-ready/group-CAS reuse, stale loser zero-member publication, restart rebind with new transient handles, invocation workspace isolation, exact profile/KAD/resource binding, exact window/chunk coverage, double-buffer copy/consumer/reuse dominance and bounded host/device staging with no full-payload allocation or runtime repack. The PP integration test does the same for a real two-stage entrypoint whose selected `EntryGraphIterationDomain` evaluates to at least two microbatches; assert cross-iteration stage overlap, iteration-scoped workspace isolation and middle-iteration failure suppression/cleanup.

The sparse-MoE integration compiles a real count-phase plus expert/combine graph whose total expert module/weight footprint exceeds resident capacity but each finite declared wave fits its per-domain envelope. It drives mixed zero/nonzero typed counts, requires every nonzero expert and no zero expert to execute, checks false `skipped_success`/true terminal convergence at the exact conditional join and proves zero blob acquire/module load/copy/submit for false members. Count overflow, wrong owner/range/type, provider without conditional issue, envelope overrun and injected mid-wave failure are structured failures; runtime cannot reduce the active set, reorder waves or fall back to load-all. The capacity-authority integration starts two independent host processes against one fake shared 2- and 4-domain deployment: exclusive-local admits only the current complete-domain lease/fence, while fake-central atomic prepare/commit keeps aggregate multidimensional usage within hard limits. Lease expiry, process death, stale release/submit and central prepare failure prove fencing and rollback; declaring shared devices with neither authority is unsupported before object/provider effects. None may supply hand-written group/instruction/package JSON, a caller-expanded schedule or a self-authenticated bundle path. These gates prove package/runtime contract execution, resource/binding construction and completion ordering; they make no model numeric or hardware-completion claim.

- [ ] **Step 2: Run tests and confirm old CLI cannot satisfy them**

Run:

```bash
cmake --build build/wafer-dev --target wafer-opt wafer-run -- -j128
<configured-lit> -sv \
  build/wafer-dev/test/Runtime/wafer-run.test \
  build/wafer-dev/test/Integration/package-runtime-no-card.test \
  build/wafer-dev/test/Integration/package-runtime-stateful-streaming-no-card.test \
  build/wafer-dev/test/Integration/package-runtime-pp-no-card.test \
  build/wafer-dev/test/Integration/package-runtime-sparse-moe-no-card.test \
  build/wafer-dev/test/Integration/package-runtime-capacity-authority-no-card.test \
  build/wafer-dev/test/Integration/state-migration-no-card.test
```

Expected before implementation: tests fail because `wafer-run` expects `--package-metadata`, the production driver cannot emit a package bundle, and no typed RuntimeSession path is connected.

- [ ] **Step 3: Connect compiler-only package emission**

For `stablehlo-to-executable`, retain one outer `compiler::ProgramOutputTransaction` from candidate creation through the declared completion scope. After the sealed committed executable and every complete target-set attachment are present but still private, call `compiler::detail::PackageProgramOutputAccess::buildPackageAssemblyInputView` for the latest complete view and move it with validated `PackagePublicationLimits` through `beginPackageArtifactBuild`. Call `assemblePackage(session, parseLimits, artifactLimits)`, then move-consume the same live session/result in `stagePackageBundle`; the driver never obtains or supplies a canonical context, manifest backing factory, staging root or blob callback. Reject package output for candidate executable, incomplete artifact set, stale attachment generation, quota/backing mismatch or any failed member. Only after all attachments pass may the driver perform one unified-root rename or trusted `ProgramDeliveryCommitRecord` CAS. The package writer never independently renames a visible directory, and the driver does not expose an option that assembles package from raw instruction/LLVM files or hand-written manifest.

Link `wafer-opt` and `wafer-package-convert` to compiler-only `WaferPackageCompiler`. Link `WaferRuntime` and `wafer-run` only to runtime-safe `WaferPackageFormat`, `WaferArtifact` and their LLVM/system dependencies; neither may link `WaferIR`, `WaferTargetArtifacts`, `WaferPackageCompiler` or `MLIR*`, directly or transitively. `WaferArtifact` owns only runtime-neutral budget capability/session and must not include/link back to `WaferRuntime`. Extend `check_deps.py` and `check-deps.test` to inspect both source includes and generated CMake link interfaces for these exact edges.

- [ ] **Step 4: Replace host runtime entry with validated bundle loading**

Change `HostRuntime.h` into an umbrella for `RuntimeSession.h`, `RuntimeProviderCapabilities.h`, `RuntimePackageBinding.h`, `RuntimeServiceContext.h` and the final manager headers; remove old `RuntimePackage`, name-based `RuntimeBinding`, free lifecycle, `bindingOrder` and scalar `completionSource` structures. Before touching the bundle, `wafer-run` parses repeatable `--parse-limit=<kind=value>` and `--admission-limit=<kind=value>` options using closed `PackageParseScalarLimitKind` plus `record.<PackageRecordKind>` and `RuntimeAdmissionLimitKind` tables, together with separate closed `ModuleVerificationMode` and `ModuleResidencyMode` option tables. It then constructs validated `PackageParseLimits` and `RuntimeAdmissionPolicy`; each policy axis is required exactly once, and unknown/duplicate/missing/zero values fail while the manifest cannot override them. This CLI mapping is not a serialized policy ABI; embedded/daemon hosts directly construct the same validated objects from their trusted deployment control plane.

The host order is fixed. First the lower trusted bootstrap loads `TrustedHostVerificationInventory`, issues the unique physical ledger and creates one `HostVerificationRegistry` with requested limits no greater than sealed hard maxima. Its Delivery host child creates `ProgramDeliveryVerificationRegistry`, which issues `ProgramDeliveryVerificationCapability`; the free three-move factory builds a session retaining delivery parse limits/context, and `loadAndVerifyProgramDelivery(reference, session)` produces the only verified outer proof/package commitment. Destroy that Delivery session before continuing; the proof owns its immutable backing/root provenance and no lease is borrowed. Next the distinct Artifact-metadata host child creates `ArtifactMetadataVerificationRegistry`, issues `ArtifactMetadataVerificationBudgetCapability` and builds an `ArtifactMetadataVerificationSession`; `loadPackageBundleMetadata(packageRef, parseLimits, session)` reads only index/manifest and returns copyable inert `LoadedPackageMetadata`. Destroy the Artifact metadata session before pure preflight/bind; metadata owns immutable provenance but no session/lease/open authority. No model blob/module has opened and no device authority exists.

Independently authenticated deployment bootstrap builds one canonical provider capability set, reservation-authority capability and optional durable-state service, while `loadAuthenticatedDeploymentInventory()` returns one inventory/catalog pair. The host retains a cheap immutable catalog copy for later typed migration policy resolution and move-consumes the inventory, other service capabilities and Host registry runtime-artifact child through the sole `createVerifiedRuntimeServiceBootstrap` factory. `WaferRuntimeServiceRegistry::create(std::move(bootstrap))` is the only registry factory. It side-effect-free `discoverEnvironment()` and returns `VerifiedRuntimeEnvironmentInventory`. Resolve the CLI alias to typed `ModelEntrypointId`, create one absolute-deadline `InvocationControl`, then call `preflightRuntimeSession(metadata, inventory, std::move(invocation), std::move(admissionPolicy), capacityMode, std::move(control))`; this pure step move-owns policy/mode/control and fixes exact domains before authority. Only `registry.getOrCreateContext(plan)` internally obtains/reuses exact-domain authority. Context-private access combines its service owner with the retained runtime host child, issues `ArtifactVerificationBudgetCapability`, and the three-move runtime factory creates `RuntimeArtifactVerificationSession`. `bindPackageForRuntime(metadata, std::move(plan), context, std::move(session))` revalidates provenance and atomically binds every inert descriptor without opening it. Finally `RuntimeSession::create(std::move(boundPackage))` reserves capacity and realizes the invocation; `executeInvocation(session)` accepts no replacement control/capability. Consecutive prefill/decode invocations copy the same metadata handle but create distinct plans/bound packages/runtime sessions while sharing the registry context/cache/state. The tool never constructs a verified manifest/blob/environment/reference/proof/capability wrapper itself, passes alias text to runtime or reopens a verified locator.

For `tx.model` with descriptor-only BPM materialization, return a structured unsupported error; never fall back to `tx.module`. Dry-run prints `not_executed`; fake-TX is orchestration evidence only and prints no numeric-success claim.

- [ ] **Step 5: Remove Python and flat-JSON contract owners**

Delete the three Python tools and their CTest registration. Keep generated Python protobuf binding only for debug projection tests and keep schema-v2 fixtures only under converter inputs. The explicit C++ converter remains the sole JSON ingress.

Run the forbidden-channel search:

```bash
rg -n 'wafer_package_metadata|wafer_export_package_metadata|wafer_runtime_adapter' \
  include lib tools test unittests CMakeLists.txt
rg -n 'binding_order|completion_source|"instructions"' \
  include lib tools test unittests
```

Expected: the first command has no output; the second reports only `V2PackageConverter.cpp`, v2 converter diagnostics and `test/Tools/Inputs/package-v2` fixtures.

- [ ] **Step 6: Run focused and full no-card gates**

Run sequentially:

```bash
cmake --build build/wafer-dev --target check-wafer -- -j128
ctest --test-dir build/wafer-dev --output-on-failure
<configured-lit> -sv --show-unsupported build/wafer-dev/test
python3 tools/check_ir_organization.py --root .
python3 tools/check_deps.py
git diff --check
```

Expected:

- package/runtime C++ unit, converter, CLI and real program-chain no-card tests pass;
- `package-runtime-no-card.test` is actually executed, not unsupported;
- `package-runtime-stateful-streaming-no-card.test` executes two invocations with whole-group state and bounded stream windows, not a full-weight fallback;
- `package-runtime-pp-no-card.test` executes count>=2, stage overlap and injected failure rather than becoming unsupported;
- `package-runtime-sparse-moe-no-card.test` executes mixed zero/nonzero experts in bounded waves, with false members producing zero blob/module/copy/submit effects and no load-all fallback;
- `package-runtime-capacity-authority-no-card.test` runs two host processes against exclusive-local and fake-central authorities, proves stale-fence rejection/all-domain rollback, and executes rather than becoming unsupported;
- `state-migration-no-card.test` executes multi-group record-21 migration with bounded 100GB-class backing, one batch CAS and zero partial publish under second-group failure;
- default CTest no longer contains `wafer-runtime-adapter-python`;
- unsupported list contains no package/runtime mainline dependency;
- dependency output proves compiler tools link `WaferPackageCompiler`, runtime links runtime-safe `WaferArtifact`, and `WaferRuntime`/`wafer-run` have no compiler or MLIR edge;
- IR/dependency checkers exit 0 and `git diff --check` prints nothing.

- [ ] **Step 7: Update status and stable workflow notes**

In `tasks/progress.md`, advance Q4 only if compiler-generated manifest, complete artifact identity and single verifier gates all passed. Record Q6 no-card evidence but keep board launch/completion explicitly incomplete until the board plan passes. In `memory/general_dev.md`, replace old Python adapter commands with the exact C++ package/no-card commands above and retain the explicit boundary that board evidence is separate.

- [ ] **Step 8: Commit**

```bash
git add tools/wafer-opt tools/wafer-run include/Wafer/Runtime \
  lib/Wafer/Runtime test unittests/Runtime tasks/progress.md \
  memory/general_dev.md
git add -u tools test
git commit -m "Connect typed package runtime mainline"
```

## Final Self-Review Checklist

- Schema/codegen consumes one pinned Protobuf owner and generated C++/Python types; Python has no semantic validator or runtime plan.
- `PackageBlobDigest` and WCRE `PackageManifestId` have different types, algorithms, tests and failure ordering.
- Authenticated host inventory issues one hard-capped physical ledger; Delivery, Artifact-metadata and Runtime-artifact capabilities are noninterchangeable typed children that cannot reset or double-spend the parent.
- Delivery and Artifact metadata sessions are runtime-neutral and destroyed before pure preflight/bind; immutable metadata remains valid but carries no live lease/open/device authority.
- Deployment-owned parse/admission limits use checked count/byte/liveness arithmetic before reserve, blob open or provider side effects; manifest cannot override them or change semantics.
- Package assembler has no raw IR text, JSON, name matcher, parameter-count inference or partial artifact-set input.
- Single C++ verifier owns resource, variant, projection, KAD slot, artifact and completion legality for compiler and runtime.
- Shared quantization/storage/profile records remain one typed fact chain through resource realization, artifact fingerprint/KAD and slot binding; runtime never interprets/repackages them.
- V2 is an explicit converter-only ingress and cannot fabricate missing typed facts.
- Runtime accepts only non-forgeable bundle/environment inputs; preflight fixes target/projection/shape/rank, exact realization/member/relocation, slot and scope choices before allocation.
- Runtime bootstrap move-seals provider sets, independently authenticated inventory/authority/durable-state capabilities and runtime host budget once; inventory bootstrap co-issues one copyable typed domain-constraint catalog with the same owner/generation for later migration policy lookup and inventory revalidation. Registry owns all provider mechanisms and callers cannot inject/re-pair a backend, context, authority or session.
- `LoadedPackageMetadata` is a cheap immutable copyable handle with inert descriptors; pure preflight selects exact domains, internal context creation follows, and one-way all-or-none bind alone creates `RuntimeBoundPackage`/private sources.
- Runtime artifact read/work demands reserve session, common host, exact service-context and invocation-capacity ledgers in fixed nonblocking order with reverse rollback and lease-backed handle/workspace lifetimes.
- Finite entry-graph iteration domains stay as verified package templates; preflight evaluates each bounded count once and a rolling cursor lazily materializes only the bounded live horizon, including a two-microbatch PP gate and million-iteration constant-memory proof.
- Runtime owns lazy capability-bound blob sources; selected acquisitions rehash with positioned reads from one `OpenedBlobLease`, loaded module leases retain immutable evidence rather than object FDs, unselected blobs consume no object FD, and no consumer trusts a checked path.
- Actual ELF bytes, ABI notes, KAD/profile/fingerprint/member closure are reverified before provider load; only `VerifiedLoadableModule -> LoadedModuleProof/ModuleLease -> ExecutableHandle` reaches submit.
- Orthogonal `eager_active_set|at_first_use` verification and `eager_active_set|graph_liveness` residency modes respect activation-aware first-use/symbolic-last-terminal count and byte peaks, exact provider-epoch cache keys and false-member zero open/verify/load/copy/submit without reordering the entry DAG.
- Every capacity demand has one closed same-iteration/final-tail/invocation-terminal rule and per-instance/aggregate lease kind; release stays exact without O(total iterations) vectors.
- Arena/resource/weight/state instances retain typed actual allocation facts and obey declared scope; persistent registry stores durable backing refs/scope/version/journal/dirty state, reimports new transient handles per provider epoch, and preserves whole-group CAS/poison/recovery across process loss.
- Completion execution is event-driven across ranks/stages and covers host/device/DTE/copy/status, timeout/error aggregation and exact state publish/poison semantics.
- Migration record 21 serializes ABI selectors/explicit scope pairs or transform slot-role N:M relations, never runtime durable keys; one verified multi-set artifact bundle is all-required/no-extra and all sets come from one ProgramDelivery.
- Migration follows semantic request -> one-shot placement inventory -> exact-domain plan/context -> explicit runtime-bound artifact bundle -> one-use registry snapshot -> session plan; one migration-wide journal/batch CAS preserves all old state and publishes all new groups or none under bounded 100GB-class streaming.
- `WaferPackageFormat` remains runtime-safe and compiler-free; its only upper operational dependency is one-way to `WaferArtifact` for session/read-work leases, while `WaferArtifact` never links back. `WaferRuntime` remains free of compiler/MLIR links, and assembly/conversion stay in `WaferPackageCompiler`.
- No-card vertical gate starts from a real program and executes contract orchestration only; model numeric and board completion remain owned by the real-model/board plan.
