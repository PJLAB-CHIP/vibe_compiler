# Target Artifact Set Implementation Plan

> **执行约束：** 任务状态和直接前置以 `tasks/progress.md` 为准；本文 checkbox 只拆解实现步骤，不是独立状态源。

**Goal:** 从 staged committed `wafer.executable` 生成、验证并向同一 `ProgramOutputTransaction` 附着可由 package 直接消费的 complete `TargetArtifactSet`，闭合 WCRE semantic identity、static-function identity、Kernel ABI Descriptor、target LLVM、Direct DTE ABI、mandatory ELF ABI note、双 fingerprint、final module content digest 和多 member 交付。

**Architecture:** 先建立 pinned Protobuf codegen、唯一 WCRE/schema/symbol registry、shared quant/storage values以及pre-commit build/toolchain profile；verified executable entry 在单一 transaction 中从canonical closure proof生成 structure-preserving target LLVM 和 KAD；device link 在 final ELF 注入并回读验证 unique-KAD ABI notes 后计算 content digest；set builder 只有在一个 target variant 的全部 member 通过验证后才计算 root semantic digest并向outer transaction附着私有root。Package复用locator-free verified record；runtime-neutral metadata loader只验证最终program delivery绑定的物理root、record与unbound source descriptors且不打开module，runtime bundle owner随后把all-required metadata一次性绑定到独立runtime verification session，选择module后才same-handle复验ELF。任何层都不能解析 LLVM 文本、旁路文件名或partial cache恢复语义。

**Tech Stack:** C++17、MLIR/LLVM 20、LLVM Object/RISCV support、Protobuf 3.21.9 generated C++、RISC-V ELF、TX8 GCC/binutils、Python 3 device-link orchestration、lit/CTest/GTest。

## Global Constraints

- 设计 owner：`tasks/14-target-llvm-golden-packet.md` 与 `tasks/16-verification-plan.md`；accepted transport/projection 来自 `tasks/13-communication.md` 与 `tasks/04-topology-execution-mesh.md`，committed executable 来自 whole-variant plan。
- 本计划不得改变 WCRE V1 tag、big-endian width、record type、domain、digest preimage、ELF note 或 KAD 语义。ABI field number、value kind、collection semantics、identity inclusion、schema version 和 reserved number只在五份 normative Protobuf schema中定义，C++/ODS不得维护第二张表。
- `schema/wafer/semantic_identity.proto`、`schema/wafer/kernel_abi.proto`、`schema/wafer/target_artifact_set.proto`、`schema/wafer/package_manifest.proto`、`schema/wafer/state_migration.proto` 是唯一 field registry。前三份由本计划提交；package/runtime 计划提交后两份并复用/import共享types/options。每份 schema 的首次提交都是 ABI review gate；record 21只归state-migration owner。
- Protobuf deterministic serialization只是 pinned producer 的 delivery contract。Semantic identity必须先通过唯一 C++ verifier，再从 typed message构造 WCRE并做 domain-separated SHA-256。Final ELF、target-set/package Protobuf delivery blob和payload使用 raw-byte content digest；delivery digest不自嵌进自己的bytes。
- 每个domain-specific semantic ID/digest wrapper与`ContentDigest`都是不同且不可伪造的C++类型；generic semantic digest value只存在于`WaferABI`内部，禁止跨stage返回、隐式构造、比较或复用。文档、diagnostic 与代码统一使用 `descriptor semantic digest`。
- Compiler-only `WaferTargetArtifacts` adapter在transaction session内部消费 `CommitState::Committed` 的 `ExecutableOp`、`ExecutableVariantOp`、`ExecutableEntryOp`，从 typed op getter读取 `TargetVariantId`、`EntryId`、`RankClassId`、`ProjectionSetId`；它不公开接收任意raw op的assembly/relation API。Runtime-safe `WaferArtifact`不包含MLIR header或handle，只消费trusted delivery ref、owner-backed record和capability-bound bytes。KAD semantic digest是唯一ABI identity，ELF note按其32-byte digest排序，不创建另一份ABI ID。
- `wafer.static-function.v1` structural encoder由本计划的 `WaferCompilerIdentity` 唯一拥有。Whole-variant projection中的 entry adapter只解析 entry/function ref并调用该 API。`wafer.executable-semantic.v1` encoder由 whole-variant atomic commit唯一拥有，本计划只重验 committed digest。
- `QuantizationDescriptor`、`StorageEncodingDescriptor`、record-20 profile ID/registry和reviewed external symbol registry都由runtime-safe `WaferABI`唯一拥有；MLIR、KAD、artifact和package只使用verified adapter/import，不能重声明字段或从string恢复。
- 依赖分层固定为：runtime-safe `WaferABI`只链接LLVM ADT/Support、pinned Protobuf和generated schemas；runtime-safe `WaferArtifact`再链接LLVM Object并拥有ELF/set reader；基础MLIR-dependent `WaferCompilerIdentity`拥有static/pre-conversion IR-to-schema identity且不include/link conversion；`WaferInstrToTargetLLVM`唯一拥有verified conversion request/boundary/core并向下链接`WaferCompilerIdentity`；独立上层`WaferCompilerKernelAbiAdapter`链接conversion、base identity和ABI并拥有唯一post-conversion KAD adapter；compiler-only `WaferTargetArtifacts`再单向消费conversion与该adapter，拥有orchestration、device-link、private staging和outer-transaction attachment。只有`ProgramOutputTransaction` completion scope拥有最终publication；Package/runtime不得链接这些compiler targets。
- Tasks 1-5（包含Task 3A的完整runtime-safe KAD wire/value foundation）是跨计划共享基础，必须在 whole-variant projection/commit 前执行。Whole candidate从Task 5的verified build-profile registry选择typed low-precision profile ID，并直接复用Task 3A的Slot/completion/state enums；static-function identity可以在已通过全部whole-variant gate的commit-ready clone上计算，但不得在atomic commit前发布artifact、写入cache或成为package可见事实。commit后必须从committed function重算并逐字节匹配。
- Tasks 6-14 是post-commit target artifact path，只接受 `CommitState::Committed` 的 executable/variant/entry。Commit-ready clone不能进入KAD、device link、ELF、content digest或set attachment。
- Target conversion必须调用 correctness plan唯一的 `convertVerifiedWaferInstrModuleToTargetLLVM(VerifiedTargetConversionRequest)`，并在各自op generation内复用同一geometry/boundary contract；不得跨rewrite携带旧`VerifiedInstructionGeometry` proof。Conversion、KAD与fingerprint任一失败时，transaction不得产出 LLVM module、descriptor delivery bytes或artifact member。
- Target compile/link只能消费`VerifiedTargetToolchainInvocation`，全部binary/sysroot/CRT digest、canonical argv和allowed environment进入typed profile；caller path/flag/env/allowlist不能成为旁路codegen input。
- `TargetArtifactBuildLimits`和`ArtifactAdmissionLimits`是validated positive nonidentity service policy。前者不能改变canonical closure/partition/member set，后者不能截断delivery/KAD/note/profile；任一超限都返回结构化失败且不暴露partial proof/root/handle。
- Pinned projection把concrete endpoint/channel/FSM/receiver facts materialize进target code；relocatable projection只暴露已验证 finite member set对应的committed executable-entry `TargetEntrySlot` endpoint/control/status关系，conversion后KAD再记录并验证同一slot ABI。Runtime不获得route/resource搜索自由度。
- `.note.wafer.abi` 必须是 `SHT_NOTE`、`SHF_ALLOC`、非 writable/非 executable、alignment 4；每个module的每个unique KAD semantic digest恰好一条note并按32 raw bytes排序，多个entry可多对一引用相同verified KAD。Module content digest只能在note注入和final ELF verifier之后计算。
- `TargetArtifactSetId`只能在所有final module content digest存在后计算。任一member失败必须删除staging tree且不产生可见root；独立content-addressed blob存在不代表set accepted。
- `TargetArtifactSetVerifiedRecord`是locator-free package复用边界并由`WaferABI`唯一验证；`TargetArtifactSetDeliveryRoot`只增加physical locator/size且不能进入PackageManifest semantic record。Loader的expected delivery digest/size必须来自`ProgramOutputTransaction`或其它可信typed index。
- `TargetArtifactSet` coverage owner是`(ExecutableSemanticDigest, TargetVariantId)`；builder必须收集所有引用该target的committed shape variants，caller不能传单个`ExecutableVariantOp`、entry subset或module grouping。
- `ModulePartitionPolicyV1`完整进入artifact fingerprint。V1 grouping只按typed non-clonable connectivity、proved-pure clone dependency digests、module compatibility、exact core+deduplicated-clone bytes、exact unique-KAD note bytes和canonical sequential packing决定；final group关闭后才求profile union并计算fingerprint。path、enumeration/build completion order或启发式size estimate都不是输入。
- Production入口是 `wafer-opt --program-pipeline=stablehlo-to-executable`。Replay pass、hand-written LLVM、`--print-commands`、单module fixture与skipped target test都不能作为completion proof。
- 任务号、阶段号不得进入target、helper、schema message、CLI、artifact、diagnostic、pipeline或CMake target名称。

Pipeline position:
- Upstream artifact / IR: verified committed `wafer.executable`，包含完整variant/rank/entry/resource/transport/projection/completion事实、typed low-precision profile refs、accepted memory offsets和static instruction functions。
- Current stage responsibility: 生成static function semantic identity、target LLVM、KAD、target fingerprints，执行device link，注入并验证ELF ABI note，计算final content digest，并把一个target variant的完整artifact set作为私有root附着到outer transaction；不独立建立可见性。
- Output artifact / IR: complete verified `TargetArtifactSet`，以source executable semantic digest和typed `TargetVariantId`为root，member闭合entry/function/module/KAD/fingerprint/rank-class/completion关系。
- Downstream consumer: PackageManifest assembler通过`compiler::PackageAssemblyInputView`消费已封存的locator-free record/unbound source；standalone runtime先经trusted delivery ref加载all-required metadata，再原子batch-bind并按选择打开经program-delivery verifier绑定的module/KAD/member。
- User-level driver / named pipeline: `wafer-opt --program-pipeline=stablehlo-to-executable`。
- Explicit non-goals: 不定义execution/deployment `RuntimeSession`或package assembly session；这里只定义相互独立、runtime-neutral的artifact metadata verification session和module-byte runtime verification session。前者没有module/device authority，后者不能替代metadata parser。不恢复logical scheduling，不从名字/路径/LLVM文本猜ABI，不把partial cache升级为set，不新增provider route search或未验证raw DTE模式。
- Completion gate: actual `committed executable -> target LLVM -> LLVM IR -> target object -> CRT object -> final kcore .so -> verified TargetArtifactSet`链路执行；所有closure/profile/toolchain/ELF/KAD/fingerprint/member关系与build/admission limits通过；注入任一member失败或超限时无可见set root。

---

## Shared Pre-Commit Identity Foundation

### Task 1: Shared Protobuf 3.21.9 Dependency and Codegen Foundation

**Files:**
- Modify: `.gitmodules`
- Add: `third_party/protobuf` gitlink
- Modify: `cmake/third_party/WaferDependencyVersions.cmake`
- Modify: `cmake/third_party/WaferThirdParty.cmake`
- Create: `cmake/WaferProtoSupport.cmake`
- Modify: `tools/bootstrap_deps.py`
- Modify: `tools/check_deps.py`
- Modify: `test/Tools/check-deps.test`

**Consumes:** Protobuf version `3.21.9`, upstream tag `v21.9`, annotated tag object `82d8c457e4f57eb3faaa622bd5ef1e67ca6f7e01`, peeled commit `90b73ac3f0b10320315c2ca0d03a5a9b095d2f66`, repository `https://github.com/protocolbuffers/protobuf.git`.

**Produces:** exact source pin, `WaferProtoSupport` INTERFACE target linking `protobuf::libprotobuf`, exported exact-version check `wafer_require_protobuf_version("3.21.9")`, and one build-tree-only `wafer_add_proto_library(TARGET ... PROTOS ...)` helper shared by identity, KAD, target-set delivery, PackageManifest and state-migration schemas.

- [ ] **Step 1: Add failing dependency pin and helper checks**

  Extend `test/Tools/check-deps.test` to require:

  ```text
  Protobuf v21.9 90b73ac3f0b10320315c2ca0d03a5a9b095d2f66 tag-object 82d8c457e4f57eb3faaa622bd5ef1e67ca6f7e01
  ```

  Extend `check_deps.py` assertions so the test fails until all five `WAFER_PROTOBUF_*` values, `.gitmodules`, exact checkout commit, annotated tag object, `WaferProtoSupport`, `wafer_require_protobuf_version` and `wafer_add_proto_library` exist. A lightweight `--versions-only` run must not require a populated checkout.

- [ ] **Step 2: Run the failing dependency test**

  ```bash
  <configured-lit> -sv build/wafer-dev/test/Tools/check-deps.test
  ```

  Expected before implementation: missing `WAFER_PROTOBUF_*` pin and helper diagnostics.

- [ ] **Step 3: Pin source and bootstrap behavior**

  Add these exact variables:

  ```cmake
  set(WAFER_PROTOBUF_VERSION "3.21.9")
  set(WAFER_PROTOBUF_TAG "v21.9")
  set(WAFER_PROTOBUF_TAG_OBJECT "82d8c457e4f57eb3faaa622bd5ef1e67ca6f7e01")
  set(WAFER_PROTOBUF_COMMIT "90b73ac3f0b10320315c2ca0d03a5a9b095d2f66")
  set(WAFER_PROTOBUF_REPOSITORY "https://github.com/protocolbuffers/protobuf.git")
  ```

  Add `third_party/protobuf` as a submodule and add `--proto-sources` to `bootstrap_deps.py`; `--all` also syncs it. Fetch the exact annotated `v21.9` tag object in addition to the peeled commit. After sync, verify both `git rev-parse HEAD` and `git rev-parse v21.9` against the peeled commit and annotated tag object. A lightweight checkout mismatch must fail instead of silently checking out another 3.21 patch release.

- [ ] **Step 4: Add one stable CMake integration**

  In `WaferThirdParty.cmake`, configure the pinned source with tests/examples/install disabled and protoc/libprotobuf enabled, then include `cmake/WaferProtoSupport.cmake`. The helper contract is:

  ```cmake
  add_library(WaferProtoSupport INTERFACE)
  target_link_libraries(WaferProtoSupport INTERFACE protobuf::libprotobuf)
  set_property(TARGET WaferProtoSupport PROPERTY
    WAFER_PROTOBUF_VERSION "3.21.9")

  wafer_require_protobuf_version("3.21.9")

  wafer_add_proto_library(
    TARGET <target>
    PROTOS <repo-relative .proto files>
    [IMPORT_DIRS <source-tree import roots>...]
    [PYTHON_OUTPUT_DIR <build-tree directory>]
  )
  ```

  `wafer_require_protobuf_version(required)` reads the `WAFER_PROTOBUF_VERSION` target property and the version exported by the source-configured Protobuf targets, and issues `FATAL_ERROR` unless both exactly equal `required`; consumers do not run a second `find_package` or compare a locally declared version variable. The codegen helper invokes the pinned `protobuf::protoc`, preserves each path relative to the matched import root, writes `.pb.h/.pb.cc` only below `${CMAKE_CURRENT_BINARY_DIR}/generated`, exports that include directory, links `WaferProtoSupport`, and declares generated files byproducts. `IMPORT_DIRS` defaults to the repo `schema/` root and may use an absolute CMake path only when it resolves inside the source tree. When `PYTHON_OUTPUT_DIR` is present, the same protoc invocation family generates Python bindings below that build-tree directory and attaches them as target byproducts; it does not install or check in generated code. The helper rejects absolute proto source paths, out-of-repo import roots, source paths not covered by exactly one import root, duplicate targets and missing source files.

- [ ] **Step 5: Prove codegen uses the pinned tool**

  Add a configure-time checker in `check_deps.py` that inspects the CMake helper text and target references; do not run network access. Then run:

  ```bash
  python3 tools/check_deps.py
  cmake -S . -B build/wafer-dev -G Ninja
  cmake --build build/wafer-dev --target protoc -- -j128
  <configured-lit> -sv build/wafer-dev/test/Tools/check-deps.test
  ```

  Expected: exact version line, dependency checks exit 0, and pinned protoc target builds.

- [ ] **Step 6: Commit**

  ```bash
  git add .gitmodules third_party/protobuf \
    cmake/third_party/WaferDependencyVersions.cmake \
    cmake/third_party/WaferThirdParty.cmake cmake/WaferProtoSupport.cmake \
    tools/bootstrap_deps.py tools/check_deps.py test/Tools/check-deps.test
  git commit -m "Pin shared Protobuf code generation"
  ```

### Task 2: WCRE V1 and Typed Digest Foundation

**Files:**
- Create: `include/Wafer/ABI/Digest.h`
- Create: `include/Wafer/ABI/CanonicalEncoding.h`
- Create: `include/Wafer/ABI/WCRE.h`
- Create: `lib/Wafer/ABI/Digest.cpp`
- Create: `lib/Wafer/ABI/ImmutableByteBackingInternal.h`
- Create: `lib/Wafer/ABI/CanonicalEncoding.cpp`
- Create: `lib/Wafer/ABI/WCRE.cpp`
- Create: `lib/Wafer/ABI/CMakeLists.txt`
- Modify: `lib/Wafer/CMakeLists.txt`
- Create: `unittests/ABI/WCRETest.cpp`
- Create: `unittests/ABI/WCREStreamingTest.cpp`
- Create: `unittests/ABI/WCREProcessProbe.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/ABI/wcre-cross-process.test`

**Consumes:** WCRE V1 byte grammar, record types `0..21`, twenty-one fixed domains and SHA-256 preimage from `tasks/14`.

**Produces:** `WaferABI`, validated move-only `CanonicalEncodingContext` owning `CanonicalEncodingLimits` plus a bounded transaction-local `CanonicalScratchStore`, two-pass WCRE measurement/streaming, shared immutable `ImmutableByteBackingRef`/`CanonicalRecordBackingRef`, public raw-byte `ContentDigest`, and a fixed-fixture cross-process probe used only by tests. Generic semantic digest bytes are not a cross-stage value type; Task 3 exposes only record/domain-specific wrappers.

`WaferABI` links only `LLVMSupport`/LLVM ADT and `WaferProtoSupport`; its public errors are `llvm::Error`/`llvm::Expected`. Including or linking this target must not pull MLIR IR, dialect, parser, pass or conversion libraries.

```cpp
namespace wafer::abi {
inline constexpr uint16_t kWCREEncodingVersion = 1;

enum class DigestAlgorithm : uint16_t { Sha256 = 1 };

class ContentDigest final {
public:
  DigestAlgorithm algorithm() const;
  llvm::ArrayRef<uint8_t> bytes() const;
private:
  ContentDigest() = delete; // private verified/hash-result constructor omitted
  friend class ContentDigestBuilder;
  friend ContentDigest computeContentDigest(llvm::ArrayRef<uint8_t> bytes);
};

class ContentDigestBuilder final {
public:
  void update(llvm::ArrayRef<uint8_t> bytes);
  ContentDigest finalize();
};

ContentDigest computeContentDigest(llvm::ArrayRef<uint8_t> bytes);

struct CanonicalEncodingLimitConfig {
  uint32_t maxRecursiveDepth;
  uint64_t maxFields;
  uint64_t maxNestedRecords;
  uint64_t maxListElements;
  uint64_t maxSetElements;
  uint64_t maxSingleValueBytes;
  uint64_t maxTotalEncodedBytes;
  uint64_t maxPlanNodes;
  uint64_t maxPlanArenaBytes;
  uint64_t maxInlineRecordBytes;
  uint64_t maxSetSortScratchBytes;
  uint64_t maxSetSortRuns;
  uint64_t maxScratchOpenFiles;
  uint64_t maxConcurrentEncoders;
};

class CanonicalEncodingLimits final {
public:
  const CanonicalEncodingLimitConfig &config() const;
private:
  CanonicalEncodingLimits() = delete;
  friend llvm::Expected<CanonicalEncodingLimits>
  validateCanonicalEncodingLimits(const CanonicalEncodingLimitConfig &);
};

llvm::Expected<CanonicalEncodingLimits>
validateCanonicalEncodingLimits(const CanonicalEncodingLimitConfig &config);

class ByteSink {
public:
  virtual ~ByteSink() = default;
  virtual llvm::Error write(llvm::ArrayRef<uint8_t> bytes) = 0;
};

namespace detail {
struct ImmutableByteBackingStorage;
class SealedByteBackingFactory;
class CanonicalRecordEmitter;
} // namespace detail

class ImmutableByteBackingRef final {
public:
  ImmutableByteBackingRef(const ImmutableByteBackingRef &) = default;
  ImmutableByteBackingRef(ImmutableByteBackingRef &&) = default;
  uint64_t exactSize() const;
  const ContentDigest &contentDigest() const;
  llvm::Expected<size_t> readAt(
      uint64_t offset, llvm::MutableArrayRef<uint8_t> destination) const;
  llvm::Error writeTo(ByteSink &sink) const;
  llvm::Expected<ImmutableByteBackingRef>
  slice(uint64_t offset, uint64_t size) const;
private:
  ImmutableByteBackingRef() = delete;
  explicit ImmutableByteBackingRef(
      std::shared_ptr<const detail::ImmutableByteBackingStorage> storage);
  std::shared_ptr<const detail::ImmutableByteBackingStorage> storage_;
  friend class detail::SealedByteBackingFactory;
};

class CanonicalRecordSink : public ByteSink {
public:
  ~CanonicalRecordSink() override = default;
};

class CanonicalScratchStore {
public:
  virtual ~CanonicalScratchStore() = default;
  /* Implementations create bounded transaction-owned immutable scratch
     objects and runs; no locator is observable by the encoder result. */
};

class CanonicalEncodingContext final {
public:
  CanonicalEncodingContext(CanonicalEncodingContext &&) = default;
  CanonicalEncodingContext(const CanonicalEncodingContext &) = delete;
  const CanonicalEncodingLimits &limits() const;
private:
  CanonicalEncodingContext() = delete;
  friend llvm::Expected<CanonicalEncodingContext>
  createCanonicalEncodingContext(
      CanonicalEncodingLimits,
      std::unique_ptr<CanonicalScratchStore>);
  /* owns scratch lifetime, concurrency/FD reservations and collision registry */
};

llvm::Expected<CanonicalEncodingContext> createCanonicalEncodingContext(
    CanonicalEncodingLimits limits,
    std::unique_ptr<CanonicalScratchStore> scratch);

class CanonicalPlanSession final {
public:
  CanonicalPlanSession(CanonicalPlanSession &&) = default;
  CanonicalPlanSession(const CanonicalPlanSession &) = delete;
private:
  CanonicalPlanSession() = delete;
  /* bounded node/arena owner + exact CanonicalEncodingContext generation */
  friend llvm::Expected<CanonicalPlanSession>
  beginCanonicalPlan(CanonicalEncodingContext &);
};

llvm::Expected<CanonicalPlanSession>
beginCanonicalPlan(CanonicalEncodingContext &encoding);

class CanonicalRecordBackingRef final {
public:
  CanonicalRecordBackingRef(const CanonicalRecordBackingRef &) = default;
  CanonicalRecordBackingRef(CanonicalRecordBackingRef &&) = default;
  uint64_t exactSize() const;
  const ContentDigest &contentDigest() const;
  const ImmutableByteBackingRef &bytes() const;
  llvm::Error writeTo(ByteSink &sink) const;
private:
  CanonicalRecordBackingRef() = delete;
  explicit CanonicalRecordBackingRef(ImmutableByteBackingRef bytes);
  ImmutableByteBackingRef bytes_;
  friend class detail::CanonicalRecordEmitter;
};

namespace detail { // install-private ABI implementation, not a proof type API
enum class WCRERecordType : uint32_t {
  Nested = 0,
  TargetEnvironment = 1,
  TopologySnapshot = 2,
  ExecutionMesh = 3,
  ProjectionSet = 4,
  ExecutableSemantic = 5,
  StaticFunction = 6,
  KernelAbi = 7,
  TargetArtifact = 8,
  TargetArtifactSet = 9,
  PackageManifest = 10,
  ModelInterface = 11,
  DistributedProgram = 12,
  ModelBoundaryResourceId = 13,
  ModelInternalResourceId = 14,
  ModelDimensionId = 15,
  ExecutableResourceId = 16,
  TargetVariantId = 17,
  ExecutableVariantId = 18,
  EntryId = 19,
  QuantStorageAbiProfileId = 20,
  StateMigrationPlanId = 21,
};

enum class WCREDomain : uint8_t {
  TargetEnvironment,
  TopologySnapshot,
  ExecutionMesh,
  ProjectionSet,
  ExecutableSemantic,
  StaticFunction,
  KernelAbi,
  TargetArtifact,
  TargetArtifactSet,
  PackageManifest,
  ModelInterface,
  DistributedProgram,
  ModelBoundaryResource,
  ModelInternalResource,
  ModelDimension,
  ExecutableResource,
  TargetVariant,
  ExecutableVariant,
  ExecutableEntry,
  QuantStorageAbiProfile,
  StateMigrationPlan,
};

class SemanticDigestValue final { /* algorithm + exactly 32 bytes */ };

class WCRERecordPlan;
class WCREValuePlan final {
public:
  static llvm::Expected<WCREValuePlan>
  boolean(CanonicalPlanSession &plan, bool value);
  static llvm::Expected<WCREValuePlan>
  unsignedInteger(CanonicalPlanSession &plan, uint64_t value);
  static llvm::Expected<WCREValuePlan>
  signedInteger(CanonicalPlanSession &plan, int64_t value);
  static llvm::Expected<WCREValuePlan>
  bytes(CanonicalPlanSession &plan, const ImmutableByteBackingRef &value);
  static llvm::Expected<WCREValuePlan>
  ascii(CanonicalPlanSession &plan, llvm::StringRef value);
  static llvm::Expected<WCREValuePlan>
  record(CanonicalPlanSession &plan, WCRERecordPlan record);
  static llvm::Expected<WCREValuePlan>
  unionValue(CanonicalPlanSession &plan, uint32_t discriminant,
             WCREValuePlan payload);
private:
  WCREValuePlan() = delete;
};

class WCRERecordPlan final {
public:
  static llvm::Expected<WCRERecordPlan>
  create(CanonicalPlanSession &plan, WCRERecordType recordType,
         uint32_t schemaVersion);
  llvm::Error addField(uint32_t fieldNumber, WCREValuePlan value);
private:
  WCRERecordPlan() = delete;
};

enum class WCRECollectionKind : uint8_t { OrderedList, CanonicalSet };

class WCRECollectionPlanBuilder final {
public:
  WCRECollectionPlanBuilder(WCRECollectionPlanBuilder &&) = default;
  WCRECollectionPlanBuilder(const WCRECollectionPlanBuilder &) = delete;
  llvm::Error append(WCREValuePlan value);
  llvm::Expected<WCREValuePlan> finish();
private:
  WCRECollectionPlanBuilder() = delete;
  /* exact expected count + plan-session generation + bounded handle storage */
  friend llvm::Expected<WCRECollectionPlanBuilder>
  beginWCRECollection(CanonicalPlanSession &, WCRECollectionKind, uint64_t);
};

llvm::Expected<WCRECollectionPlanBuilder> beginWCRECollection(
    CanonicalPlanSession &plan, WCRECollectionKind kind,
    uint64_t expectedElementCount);

class CanonicalRecordMeasurement final {
public:
  uint64_t exactSize() const;
private:
  CanonicalRecordMeasurement() = delete;
};

class CanonicalRecordResult final {
public:
  const SemanticDigestValue &semanticDigest() const;
  const CanonicalRecordBackingRef &backing() const;
private:
  CanonicalRecordResult() = delete;
};

llvm::Expected<CanonicalRecordMeasurement> measureCanonicalRecord(
    const WCRERecordPlan &record, CanonicalPlanSession &plan);

llvm::Expected<CanonicalRecordResult> streamCanonicalRecord(
    WCREDomain domain, const WCRERecordPlan &record,
    const CanonicalRecordMeasurement &measurement,
    CanonicalPlanSession &plan);
} // namespace detail
} // namespace wafer::abi
```

- [ ] **Step 1: Add exhaustive failing WCRE tests**

  `WCRETest.cpp` must assert exact hex for false, true, u64, negative i64 two's-complement, empty/nonempty bytes, ASCII, nested record, ordered list, canonical set and union by streaming into a test sink whose capacity is explicitly below `maxInlineRecordBytes`. Cover zero/max widths, non-ASCII rejection, field zero rejection, decreasing/duplicate field rejection, duplicate set element rejection, unsupported record type, measurement/emission size disagreement and every checked length/count overflow. Exercise inline, sealed-file and content-addressed `ImmutableByteBackingRef`s with partial/interleaved `readAt`, full `writeTo`, nested slices and owner teardown; require the ref/slice to retain one stable handle/owner and exact digest. Reject slice/read arithmetic overflow, out-of-range access, caller-paired size/digest, mutable backing, post-verification path replacement and any factory that would reopen a locator.

  Add digest tests that verify the literal preimage prefix `WAFER\0`, `u16be(1)`, domain length/name and the measured `u64be(record_size)` before the first record byte; the same WCRE stream under all twenty-one domains must produce twenty-one distinct internal values, including `wafer.model-boundary-resource.v1`, `wafer.model-internal-resource.v1`, `wafer.model-dimension.v1`, `wafer.executable-resource.v1`, `wafer.target-variant.v1`, `wafer.executable-variant.v1`, `wafer.executable-entry.v1`, `wafer.quant-storage-abi-profile.v1` and `wafer.state-migration-plan.v1`. Verify debug spelling is exactly `sha256:<64 lowercase hex>`, `detail::SemanticDigestValue` is absent from installed/cross-stage API signatures, and semantic/content values cannot be constructed from or compared with one another. Feed the same multi-megabyte byte stream in one chunk and irregular chunks through `ContentDigestBuilder`; both must equal `computeContentDigest` without retaining a second copy.

  `WCREStreamingTest.cpp` builds 100,000-element and 1,000,000-element record/member-set fixtures whose source order is forward, reverse and independently shuffled. Feed them incrementally through `WCRECollectionPlanBuilder`; an over-limit/overflowing claimed count must fail before handle-storage reserve, append past expected count must fail immediately and early `finish` must return no value. Run each legal fixture once with a sufficiently large in-memory scratch implementation and once with a deliberately small memory window that forces multiple external-sort runs. Instrument plan arena, allocator, scratch, FD and concurrent-encoder peaks and require each to remain below its exact configured bound. All runs must produce the same exact size, content digest, semantic digest and byte stream. Test every limit at `limit-1/limit/limit+1`; zero, inconsistent cross-field limits, plan arena exhaustion, scratch exhaustion, short write, cancellation and run corruption fail with no backing/result and clean every plan/scratch object.

- [ ] **Step 2: Run and observe missing library failures**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  ```

- [ ] **Step 3: Implement bounded two-pass encoding without host-endian casts**

  Use fixed-size stack chunks and explicit sink helpers for `u16be/u32be/u64be`; never `reinterpret_cast` native integer storage. `addField` enforces strictly increasing nonzero field numbers. `orderedList` preserves caller order and streams each measured element directly. `canonicalSet` records semantic membership without pre-encoding the full set. `ascii` checks every byte is in `0x00..0x7f`; callers remain responsible for schema-level exclusion of human strings.

  `validateCanonicalEncodingLimits` rejects zero, overflow and infeasible cross-field configurations, including plan-node/arena versus collection/record maxima. `createCanonicalEncodingContext` move-binds those limits to one bounded scratch owner, concurrency/FD reservations and collision registry; it rejects a null/incompatible store, is move-only and has no default/global/unbounded factory. Each record-specific factory immediately calls `beginCanonicalPlan`, which reserves one encoder slot and returns a move-only session tied to the exact context generation. Every scalar/nested/union node charges the bounded plan arena before allocation. Byte nodes retain `ImmutableByteBackingRef`; ASCII nodes validate and copy only after charging `maxSingleValueBytes`. `beginWCRECollection` checks the untrusted expected count and complete handle-storage charge before reserve, `append` cannot exceed it, and `finish` requires exact count. Thus no caller first creates a proportional plan vector or leaves borrowed bytes dangling outside the context.

  `measureCanonicalRecord` recursively validates the complete session-owned plan, schema shape, counts and unknown fields, uses checked arithmetic for every header/value/element/record size, and never reserves from an untrusted count. It creates bounded scratch metadata only for SET ordering and returns an immutable measurement tied to that exact context/plan generation. `streamCanonicalRecord` rejects a changed context/plan/measurement, writes exactly the measured bytes, tees the stream into domain-separated SHA-256, raw `ContentDigestBuilder` and an immutable backing writer, and fails if any sink writes short or the final count differs. Plan arena and temporary sort state are released after the result's immutable backing is sealed; no result references plan nodes.

  `ImmutableByteBackingRef` is the common runtime-safe byte-lifetime boundary for canonical records, deterministic Protobuf payloads, ELF note slices and verified large artifacts. Its sole constructor friend `detail::SealedByteBackingFactory` and storage definition live in noninstalled `ImmutableByteBackingInternal.h`; only repository compiler/artifact/package sealed-source adapters and test access include it. The factory accepts an already owned inline buffer, stable opened-handle lease or content-addressed immutable owner, performs/consumes same-handle size+digest attestation and creates the const storage. It has no path/reopen, arbitrary provider callback or public `(owner,size,digest)` overload. Inline sources must be limit-checked then move-owned; file/CAS sources retain that immutable owner. `readAt` and `slice` use checked arithmetic, never expose a persistent contiguous view, and a slice retains the same owner while binding its own exact size/content digest. Owner replacement, digest/size mismatch, out-of-range/overlapping arithmetic and mutation fail before a ref is returned. Production parsers retain the backing they need; only noninstalled unit/debug adapters may copy an inline `ArrayRef`, after checking the later `ArtifactAdmissionLimits::maxInlineRecordBytes` gate.

  SET ordering remains lexicographic over each element's complete WCRE bytes, never a digest approximation. An inline set may share bounded immutable element backings. A larger set materializes measured elements into transaction-owned scratch objects, builds bounded sorted runs and performs deterministic external merge sort; prefix/digest accelerators are allowed only when an equal prefix/digest falls back to a streaming full-byte comparison. Duplicate complete elements fail. Scratch root, run size, worker count and memory/FD policy do not enter identity; every successful sufficiently provisioned configuration emits byte-identical output. RAII removes partial runs/backings after error or cancellation.

  `WCRERecordType::Nested` can be encoded only as a child passed to `WCREValuePlan::record`; `streamCanonicalRecord` rejects record type 0 as a top-level identity. Standalone resource/dimension/variant/entry/profile/migration records use legal top-level types 13-21. Add an explicit test that type 0 fails at the digest boundary and that each of types 13-21 is accepted as a canonical top-level record before its record-specific builder applies the stricter domain pairing. The fixed domain table lives in one `constexpr` switch; record-specific typed builders enforce the record-type/domain pairing before calling it, and no API accepts an arbitrary domain string. Strong ID wrappers store only their typed digest. Collision/delivery proof holders share `CanonicalRecordBackingRef`; no public encoder exposes an unbounded byte vector, and a test-only inline adapter must first reject `exactSize() > maxInlineRecordBytes`.

- [ ] **Step 4: Add a fixed-fixture cross-process golden probe**

  `WaferWCREProcessProbe` constructs one source-owned fixture containing every WCRE value form through `WCRERecordPlan`, measures and streams it through a fixed validated context, prints its exact record hex and all twenty-one typed semantic digest spellings, and accepts no external bytes/domain/options. It is a test target, not a user identity entry point.

  `unittests/CMakeLists.txt` builds the probe into the test runtime directory and adds it as a dependency of `check-wafer-lit`. `wcre-cross-process.test` invokes the probe twice in distinct processes, compares byte-identical output, and FileChecks committed record/digest literals. Malformed length, unknown tag and noncanonical record rejection remain direct `WCRETest` cases.

- [ ] **Step 5: Run focused gates**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests WaferWCREProcessProbe -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv build/wafer-dev/test/ABI/wcre-cross-process.test
  ```

  Expected: all WCRE value/golden/domain tests and the 100K/1M bounded-memory cases pass in supported processes.

- [ ] **Step 6: Commit**

  ```bash
  git add include/Wafer/ABI lib/Wafer/ABI lib/Wafer/CMakeLists.txt \
    unittests/ABI/WCRETest.cpp unittests/ABI/WCREStreamingTest.cpp \
    unittests/ABI/WCREProcessProbe.cpp \
    unittests/CMakeLists.txt test/ABI/wcre-cross-process.test
  git commit -m "Add canonical WCRE and typed digests"
  ```

### Task 3: Normative Semantic Identity Schema Registry

**Files:**
- Create: `schema/CMakeLists.txt`
- Create: `schema/wafer/semantic_identity.proto`
- Modify: `CMakeLists.txt`
- Create: `tools/wafer-artifact-inspect/CMakeLists.txt`
- Create: `tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp`
- Create: `include/Wafer/ABI/SemanticIdentity.h`
- Create: `lib/Wafer/ABI/SemanticIdentityInternal.h`
- Create: `lib/Wafer/ABI/SemanticIdentity.cpp`
- Create: `include/Wafer/ABI/ModelIdentity.h`
- Create: `lib/Wafer/ABI/ModelIdentity.cpp`
- Create: `include/Wafer/ABI/ResourceIdentity.h`
- Create: `lib/Wafer/ABI/ResourceIdentity.cpp`
- Create: `include/Wafer/ABI/DistributedIdentity.h`
- Create: `lib/Wafer/ABI/DistributedIdentity.cpp`
- Create: `include/Wafer/ABI/TargetIdentity.h`
- Create: `lib/Wafer/ABI/TargetIdentity.cpp`
- Create: `include/Wafer/ABI/ExecutableIdentity.h`
- Create: `lib/Wafer/ABI/ExecutableIdentity.cpp`
- Create: `include/Wafer/ABI/ExternalSymbolRegistry.h`
- Create: `lib/Wafer/ABI/ExternalSymbolRegistry.cpp`
- Create: `include/Wafer/ABI/Quantization.h`
- Create: `lib/Wafer/ABI/Quantization.cpp`
- Modify: `lib/Wafer/ABI/CMakeLists.txt`
- Create: `include/Wafer/Compiler/IdentityBuilderSupport.h`
- Create: `lib/Wafer/Compiler/IdentityBuilderSupport.cpp`
- Create: `lib/Wafer/Compiler/CMakeLists.txt`
- Modify: `lib/Wafer/CMakeLists.txt`
- Create: `unittests/ABI/SemanticIdentitySchemaTest.cpp`
- Create: `unittests/ABI/SemanticIdentityStreamingTest.cpp`
- Create: `unittests/ABI/ResourceIdentityTest.cpp`
- Create: `unittests/ABI/ScopedIdentityTest.cpp`
- Create: `unittests/ABI/ExecutableIdentityTest.cpp`
- Create: `unittests/ABI/ExternalSymbolRegistryTest.cpp`
- Create: `unittests/ABI/QuantizationTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/Tools/wafer-semantic-identity-schema.test`

**Consumes:** typed field ownership from `tasks/01-04`, `tasks/13-15` and the registry/evolution rules in `tasks/14`.

**Produces:** first ABI-reviewed `schema/wafer/semantic_identity.proto`, generated `WaferSemanticIdentityProto`, the only limit-aware reflection-driven Protobuf-to-WCRE projection API, one complete runtime-safe C++ value layer for cross-stage semantic/scoped IDs, eight typed digest builders for record types 13-20, the single versioned reviewed external-ABI symbol registry and the shared compiler-only `WaferCompilerIdentity` target/API slots. The schema commit itself fixes all field numbers; this plan does not duplicate them in prose.

```cpp
namespace detail {
llvm::Expected<CanonicalRecordResult> encodeIdentityMessage(
    const google::protobuf::Message &message,
    WCRERecordType expectedRecordType,
    WCREDomain expectedDomain,
    CanonicalEncodingContext &encoding);
} // namespace detail

class LowPrecisionComputeCapabilityV1;
class TargetEnvironmentFingerprint final {
public:
  DigestAlgorithm algorithm() const;
  llvm::ArrayRef<uint8_t> bytes() const;
private:
  TargetEnvironmentFingerprint() = delete;
  /* only buildTargetEnvironmentFingerprint constructs it */
};
class VerifiedTargetEnvironment final {
public:
  const TargetEnvironmentFingerprint &fingerprint() const;
  const identity::proto::TargetEnvironmentIdentity &message() const;
  const CanonicalRecordBackingRef &canonicalRecordBacking() const;
  llvm::ArrayRef<LowPrecisionComputeCapabilityV1>
  lowPrecisionCapabilities() const;
private:
  VerifiedTargetEnvironment() = delete;
};
class TopologySnapshotId;
class ExecutionMeshId;
class ProjectionSetId;
class ExecutableSemanticDigest;
class StaticFunctionDigest;
class KernelAbiSemanticDigest; // factory lands with kernel_abi.proto
class TargetArtifactFingerprint; // factory lands in Task 7
class TargetArtifactSetId;        // factory lands in Task 13
class PackageManifestId;          // factory lands with package_manifest.proto
class StateMigrationPlanId;       // record type 21; factory lands with state_migration.proto
class ModelInterfaceSemanticId;
class DistributedProgramSemanticId;
class QuantStorageAbiProfileId;   // record type 20

llvm::Expected<TargetEnvironmentFingerprint>
buildTargetEnvironmentFingerprint(
    const identity::proto::TargetEnvironmentIdentity &identity,
    CanonicalEncodingContext &encoding);
llvm::Expected<VerifiedTargetEnvironment> verifyTargetEnvironment(
    const identity::proto::TargetEnvironmentIdentity &identity,
    CanonicalEncodingContext &encoding);

class ExternalAbiSymbolIdentity final; // registry version + stable entry ordinal
class ExternalAbiSignature;
class TargetCompatibilityPredicate;
enum class ExternalAbiSymbolKind : uint8_t;
class VerifiedExternalSymbolEntry final {
public:
  const ExternalAbiSymbolIdentity &identity() const;
  llvm::StringRef asciiSpelling() const;
  const ExternalAbiSignature &signature() const;
  ExternalAbiSymbolKind kind() const;
  const TargetCompatibilityPredicate &targetCompatibility() const;
private:
  VerifiedExternalSymbolEntry() = delete;
};

class VerifiedExternalSymbolRegistry final {
public:
  uint32_t registryVersion() const;
  llvm::ArrayRef<VerifiedExternalSymbolEntry> canonicalEntries() const;
  llvm::Expected<const VerifiedExternalSymbolEntry *>
  resolve(llvm::StringRef lookupSpelling) const;
private:
  VerifiedExternalSymbolRegistry() = delete;
  /* reviewed repo-owned registry factory only */
};

class QuantizationDescriptor final {
public:
  uint32_t schemaVersion() const;
  const identity::proto::QuantizationDescriptorV1 &message() const;
private:
  QuantizationDescriptor() = delete;
  /* verifyQuantizationDescriptor only */
};

class StorageEncodingDescriptor final {
public:
  uint32_t schemaVersion() const;
  const identity::proto::StorageEncodingDescriptorV1 &message() const;
  uint64_t exactStorageBytes() const;
private:
  StorageEncodingDescriptor() = delete;
  /* verifyStorageEncodingDescriptor only */
};

llvm::Expected<QuantizationDescriptor> verifyQuantizationDescriptor(
    const identity::proto::QuantizationDescriptorV1 &descriptor);
llvm::Expected<StorageEncodingDescriptor> verifyStorageEncodingDescriptor(
    const identity::proto::StorageEncodingDescriptorV1 &descriptor,
    const QuantizationDescriptor *quantization);
```

- [ ] **Step 1: Add failing schema-registry reflection tests**

  Enumerate every message/field through generated descriptors and fail unless:

  - each top-level identity message has `wafer.wcre_record_type` and `wafer.wcre_schema_version`;
  - each participating field has explicit `wafer.wcre_value_kind`, `wafer.wcre_collection`, and `wafer.wcre_identity`;
  - union alternatives have unique nonzero `wafer.wcre_union_discriminant`;
  - no identity message uses Protobuf `map`, floating scalar, implicit enum number, unannotated repeated field or field number zero;
  - excluded fields are explicitly `EXCLUDED_DIAGNOSTIC` or `EXCLUDED_LOCATOR`;
  - all required identity fields are known to the semantic verifier and removed numbers are reserved.
  - semantic-identity-owned top-level records use each assigned type in `1-6`, `8-9`, `11-20` exactly once, never use type 0, and leave record types 7/10/21 to the KAD/package/state-migration owner schemas; nested-only messages use type 0 and cannot be passed to a semantic-digest API.

- [ ] **Step 2: Submit the schema as an ABI review unit**

  Define the custom options once in `semantic_identity.proto`:

  ```text
  message options: wafer.wcre_record_type, wafer.wcre_schema_version
  field options:   wafer.wcre_value_kind, wafer.wcre_collection,
                   wafer.wcre_identity, wafer.wcre_union_discriminant

  WcreValueKind: UNSPECIFIED, BOOL, U64, I64, BYTES, ASCII, RECORD, UNION
  WcreCollection: UNSPECIFIED, SCALAR, ORDERED, SET
  WcreIdentity:   UNSPECIFIED, INCLUDED,
                  EXCLUDED_DIAGNOSTIC, EXCLUDED_LOCATOR
  ```

  Use `package wafer.identity.proto`. The same file defines the IR-owned identity projection messages for model interface, distributed program, target environment, topology snapshot, execution mesh, projection set, committed executable, static function, target artifact and target artifact set, plus legal top-level `ModelBoundaryResourceIdentity` (record type 13), `ModelInternalResourceIdentity` (14), `ModelDimensionIdentity` (15), `ExecutableResourceIdentity` (16), `TargetVariantIdentity` (17), `ExecutableVariantIdentity` (18), `EntryIdentity` (19) and `QuantStorageAbiProfileIdentity` (20). Record 11 embeds tagged owner-local boundary/internal resource and dimension keys without public resource IDs, so it can compute the complete `ModelInterfaceSemanticId` without self-reference. Record 13/14 then encode `(ModelInterfaceSemanticId owner, tagged local resource key)`; record 15 encodes `(same owner, public model ResourceId, dimension ordinal)`. `EntryIdentity` and `QuantStorageAbiProfileIdentity` are the standalone owners of `EntryId` and `QuantStorageAbiProfileId`; references embed their registered identity record or verified typed ID rather than recreating either as a type-0 key. Type 0 is reserved for schema-local components whose parent field already fixes their meaning and that have no standalone semantic ID. `schema/wafer/state_migration.proto`, owned by the package/runtime plan, alone defines record 21 and the `StateMigrationPlanId` factory; it imports these shared typed owners and the same WCRE options rather than adding fields to `semantic_identity.proto`.

  The same schema also owns nested reusable wire messages for `ModelProgramMemberId`, `ModelEntrypointId`, `InvocationPolicyFieldId`, `StateConsistencyGroupId`, `ResourceIdentityRef`, `DistributedProgramSemanticId`, scoped `ComponentId`, `ExecutionInstanceId`, `ProjectionSetId`, `DdrArenaId`, `TargetCompatibilityContractV1`, `QuantizationDescriptorV1`, `StorageEncodingDescriptorV1`, `LowPrecisionComputeCapabilityV1`, versioned `QuantStorageAbiProfileRegistryV1`, `VerifiedTargetToolchainProfileV1`, `TargetBuildProfileV1`, `RankClassId`, `CompletionNodeId`, `IterationDomainId`, `ActivationPredicateId`, `ResourceRealizationRecordKey`, `StreamWindowId`, `TransportActionId`, `TransportBindingMemberId` and `TargetArtifactMemberKey`. The four model-scoped IDs retain `ModelInterfaceSemanticId` plus a nonzero canonical ordinal; `ModelEntrypointId` uses the model API ordinal and is not interchangeable with a program member or invocation field. `ActivationPredicateId` is exactly `(ExecutableVariantId, nonzero uint64 predicate ordinal)` and remains a nested typed composite referenced by executable/package activation records; it receives no top-level record/domain. `ResourceRealizationRecordKey` is exactly `(ResourceId, TargetVariantId, ExecutableVariantId, canonical nonempty RankClassId coverage set, canonical nonempty ProjectionSetId coverage set)`; `StreamWindowId` adds a nonzero window ordinal. This serialized coverage-owner type is intentionally distinct from runtime's singular `ResourceRealizationLookupKey`, and no conversion exists without verified axis selection. `TargetArtifactMemberKey` is exactly `(EntryId, StaticFunctionDigest, final module ContentDigest)` and becomes the delivery/package join key without acquiring a digest domain. The complete closed/versioned low-precision profile is the record-20 identity payload; registry and instruction/resource/KAD references carry its verified typed ID, never a free-form string or digest-only sidecar. Quantization/storage descriptors are nested typed facts, not new identities: `WaferABI` is their sole runtime-safe immutable value owner, and compiler IR adapters must project to/from these messages rather than define another field registry. Build/toolchain/compatibility messages contain only the reviewed registry/profile/bound facts later verified by Task 5 and never executable locators or runtime availability. These messages preserve the exact owner fields in `tasks/01`; apart from the assigned record-20 profile identity they do not allocate new top-level record types or digest domains. `kernel_abi.proto` remains the sole wire-number owner of `SlotId`, `CompletionExportId` and `StateSlotVersionRole`; target-set/package schemas import those definitions and semantic-identity scoped messages instead of redeclaring them. Each field maps one typed owner fact from the numbered documents. Locator/path, diagnostic, location and pass trace fields are either absent or explicitly excluded. There is no import path from these messages back into model/executable IR.

  Top-level CMake adds `schema/` before `lib/Wafer` and any schema consumer. `schema/CMakeLists.txt` first calls `wafer_require_protobuf_version("3.21.9")`, then creates `WaferSemanticIdentityProto` with `wafer_add_proto_library`; this is also the first real consumer test of the exported exact-version API.

  Review the entire `.proto` diff as one ABI change before merging: field number uniqueness, nested record ownership, ordered-versus-set choice, union discriminants, version 1 and reserved ranges are part of the review. Generated `.pb.h/.pb.cc` stay in the build tree.

- [ ] **Step 3: Implement one descriptor-driven projection**

  Implementation-private `detail::encodeIdentityMessage` reads only generated descriptors/options, begins one bounded `CanonicalPlanSession` and incrementally builds its `WCRERecordPlan`: numeric field order, absent optional omission, ordered collections, SET semantics, nested records and enum numeric values as u64. Repeated-field counts are checked before `beginWCRECollection`, and byte fields enter only as owner-backed slices or charged inline copies; it never first builds an unbounded plan/value vector. It requires the record-specific factory's fixed record/domain pair, validates that pairing against the one table, and passes the plan through Task 2's exact two-pass measure/stream path with the caller's encoding context. It returns `CanonicalRecordResult` so the factory can immediately wrap the internal digest and optionally retain the shared backing; no public/cross-stage API exposes the generic result. Large SETs use full-WCRE-byte external merge order, not digest order. Unknown fields, unregistered options/enums, map, non-ASCII identity string, missing identity-critical field, duplicate set element, wrong pair, plan/admission/scratch exhaustion or measurement/emission disagreement fail.

  Do not switch on C++ member names or declaration order. Record-specific builders populate generated messages from verified typed owners; they do not call a WCRE builder with locally copied field numbers. Every top-level record factory explicitly receives a validated non-default `CanonicalEncodingContext &`; there is no factory taking loose limits/store, global scratch or an unbounded convenience overload. Compilation requests/outer transactions own the compiler context for the full identity generation; deployment loaders construct a separate context from validated deployment policy and retain it through parse/verification. Define explicit non-interchangeable wrappers for every record 1-12 semantic result: target environment, topology snapshot, execution mesh, projection set, executable semantic digest, static-function digest, KAD digest, artifact fingerprint, artifact-set ID, package-manifest ID, model-interface semantic ID and distributed-program semantic ID. Strong wrappers retain only their fixed typed digest. A proof that must support collision comparison/delivery retains one shared immutable `CanonicalRecordBackingRef`; it exposes exact size/content digest/streaming, never a copied `ArrayRef` whose owner is ambiguous. This task implements verified factories for records 1-6 and 11-12; Task 3A implements record 7, Task 7 record 8, Task 13 record 9, and the package owner implements record 10 only when its schema/verifier lands. `verifyTargetEnvironment` owns complete capability validation plus record-1 computation and returns an immutable message/fingerprint/capability proof; `buildTargetEnvironmentFingerprint` is its narrow digest-only adapter. Compiler materialization and Task 5 context reuse that proof rather than pairing an arbitrary capability vector with a digest. No public API takes or returns `detail::SemanticDigestValue`, so a digest from one domain cannot satisfy another domain's parameter.

- [ ] **Step 4: Implement typed structural-ID value builders**

  `ModelIdentity.h` owns the model-interface-scoped values used by frontend, executable and package:

  ```cpp
  class ModelProgramMemberId;   // (ModelInterfaceSemanticId, nonzero u32)
  class ModelEntrypointId;      // (ModelInterfaceSemanticId, nonzero u32 API ordinal)
  class InvocationPolicyFieldId; // (ModelInterfaceSemanticId, nonzero u32)
  class StateConsistencyGroupId; // (ModelInterfaceSemanticId, nonzero u32)

  llvm::Expected<ModelProgramMemberId> buildModelProgramMemberId(
      ModelInterfaceSemanticId modelInterface, uint32_t memberOrdinal);
  llvm::Expected<ModelEntrypointId> buildModelEntrypointId(
      ModelInterfaceSemanticId modelInterface, uint32_t apiOrdinal);
  llvm::Expected<InvocationPolicyFieldId> buildInvocationPolicyFieldId(
      ModelInterfaceSemanticId modelInterface, uint32_t fieldOrdinal);
  llvm::Expected<StateConsistencyGroupId> buildStateConsistencyGroupId(
      ModelInterfaceSemanticId modelInterface, uint32_t groupOrdinal);
  ```

  Each value retains its model-interface owner, rejects zero and has a distinct generated-message type. The frontend owner remains responsible for exporter semantic-member order, unique API-entry ordinal assignment, invocation-field bounds/defaults and canonical state-group member/update ordering. Same ordinal under a different model, or the same model/ordinal under a different ID class, is unequal and cannot compile as an implicit conversion. Round-trip, ordering and wrong-owner tests cover `ModelEntrypointId` together with every other scoped type; it receives no new WCRE domain.

  `ResourceIdentity.h` defines non-interchangeable wrappers and exact typed resource inputs:

  ```cpp
  class EntryId; // record type 19 value from ExecutableIdentity.h, never a string
  class ModelBoundaryResourceId;
  class ModelInternalResourceId;
  class ModelResourceId; // typed union of the two model resource IDs
  class DimId;
  class ExecutableResourceId;
  class ResourceId; // direct closed union: boundary | internal | executable

  struct ModelBoundaryResourceLocalKey {
    uint64_t entryOrdinal;
    identity::proto::ModelBoundaryKind boundaryKind;
    uint64_t boundaryOrdinal;
    identity::proto::ResourceRole role;
  };
  struct ModelInternalResourceLocalKey {
    uint64_t entryOrdinal;
    uint64_t structuralOwnerOrdinal;
    identity::proto::ResourceRole role;
  };
  class ModelResourceLocalKey; // closed tagged boundary | internal local key
  struct ModelDimensionLocalKey {
    ModelResourceLocalKey resourceLocalKey;
    uint64_t tensorDimensionOrdinal;
  };
  struct ModelBoundaryResourceKey {
    ModelInterfaceSemanticId owner;
    ModelBoundaryResourceLocalKey localKey;
  };
  struct ModelInternalResourceKey {
    ModelInterfaceSemanticId owner;
    ModelInternalResourceLocalKey localKey;
  };
  struct ModelDimensionKey {
    ModelInterfaceSemanticId owner;
    ModelResourceId resourceId;
    uint64_t dimensionOrdinal;
  };
  struct ExecutableResourceKey {
    EntryId ownerEntry;
    identity::proto::ExecutableResourceScope scope;
    identity::proto::ResourceRole role;
    uint64_t structuralRootOrdinal;
  };

  llvm::Expected<ModelBoundaryResourceId>
  buildModelBoundaryResourceId(const ModelBoundaryResourceKey &key,
      CanonicalEncodingContext &encoding);
  llvm::Expected<ModelInternalResourceId>
  buildModelInternalResourceId(const ModelInternalResourceKey &key,
      CanonicalEncodingContext &encoding);
  llvm::Expected<DimId>
  buildDimId(const ModelDimensionKey &key,
      CanonicalEncodingContext &encoding);
  llvm::Expected<ExecutableResourceId>
  buildExecutableResourceId(const ExecutableResourceKey &key,
      CanonicalEncodingContext &encoding);
  ```

  Model-interface construction first verifies/canonicalizes the tagged local resource/dimension keys inside record 11 and computes the complete owner. Only then may the record-13/14 builders bind that owner to one local key; no public factory accepts a bare local key. Record 15 requires its explicit owner to equal the owner retained by the public model `ResourceId`, then binds the checked tensor dimension ordinal. Each digest builder validates enum/ordinal/owner relations, populates its generated top-level message, calls `encodeIdentityMessage` with record type 13/14/15/16, the explicit encoding context and the matching fixed domain, and returns a wrapper with a private raw-digest constructor. The context-local collision registry retains the shared canonical backing only as needed; each public ID value stores its typed digest and owner, not a record-sized byte vector. `ModelResourceId` is the restricted boundary/internal view used by model-only APIs and exposes its typed model owner. Unified `ResourceId` is a direct runtime-safe closed tagged value over `ModelBoundaryResourceId | ModelInternalResourceId | ExecutableResourceId`; it exposes kind/visitor/equality/total-order and verified conversion to/from the schema's one-level `ResourceIdentityRef` oneof. The union retains the concrete record type/domain, typed digest and owner, so equal local keys or even equal raw SHA-256 bytes under different alternatives/owners are neither equal nor interchangeable. It introduces no new top-level WCRE type/domain. `TargetEntrySlot`, package manifest resource/state relations and runtime/state keys reuse these values and imported messages rather than defining raw-byte/integer wrappers. Unit tests prove rename/path independence, type/domain separation, all three resource tagged round trips, deterministic ordering and one-input perturbation; same tagged local key under two model owners must produce unequal record-13/14 IDs, and cross-owner record-15, package resource, state/cache lookup or migration-free reuse must fail. Also reject bare-local factory attempts, unspecified/unknown union discriminants, nested/double tagging, digest/record mismatch, executable resource in a model-only dimension key, wrong resource-ID variant and an unverified type-19 `EntryId` reference. Program/payload semantic changes intentionally change the model owner and all public model IDs; cross-version state requires an explicit package/runtime migration relation.

  `DistributedIdentity.h` and `TargetIdentity.h` own the non-MLIR distributed/target values used by both compiler and runtime:

  ```cpp
  class DistributedProgramSemanticId; // wafer.distributed-program.v1 digest
  class ComponentId;                   // uint32 scoped to one program
  class ExecutionInstanceId;           // program + component + partition/replica coordinates
  class ProjectionSetId;               // wafer.projection-set.v1 digest
  class DdrArenaId;                    // environment fingerprint + nonzero uint32 ordinal
  class TargetCompatibilityContract;   // verified target/topology requirement bounds
  class LowPrecisionComputeCapabilityV1;
  class QuantStorageAbiProfileId;      // wafer.quant-storage-abi-profile.v1
  class QuantStorageAbiProfileV1;
  class QuantStorageAbiProfileRegistryV1;

  llvm::Expected<ExecutionInstanceId> buildExecutionInstanceId(
      DistributedProgramSemanticId program, ComponentId component,
      llvm::ArrayRef<uint64_t> partitionCoordinate,
      llvm::ArrayRef<uint64_t> replicaCoordinate);
  llvm::Expected<ProjectionSetId> buildProjectionSetId(
      const identity::proto::ProjectionSetIdentity &identity,
      CanonicalEncodingContext &encoding);
  llvm::Expected<DdrArenaId> buildDdrArenaId(
      TargetEnvironmentFingerprint environment, uint32_t arenaOrdinal);
  llvm::Expected<TargetCompatibilityContract>
  verifyTargetCompatibilityContract(
      const identity::proto::TargetCompatibilityContractV1 &contract,
      const VerifiedTargetEnvironment &environment);
  llvm::Expected<QuantStorageAbiProfileV1> verifyQuantStorageAbiProfile(
      const identity::proto::QuantStorageAbiProfileIdentity &profile,
      const TargetEnvironmentFingerprint &environment,
      CanonicalEncodingContext &encoding);
  llvm::Expected<QuantStorageAbiProfileRegistryV1>
  buildQuantStorageAbiProfileRegistry(
      uint32_t registrySchemaVersion,
      llvm::ArrayRef<QuantStorageAbiProfileV1> profiles,
      const TargetEnvironmentFingerprint &environment);
  ```

  `ComponentId` is meaningful only inside an `ExecutionInstanceId` with its distributed-program owner; `DdrArenaId` always retains its environment owner. Projection-set bytes are accepted only through the record-4/domain verifier. `QuantStorageAbiProfileV1` retains semantic/storage descriptor schema versions, one environment-owned matched capability record, packing ABI, KAD geometry rules, fixed implicit rounding/saturation pair and exact Wafer CRT command ABI versions. Its verifier encodes record 20, computes `wafer.quant-storage-abi-profile.v1`, stores the resulting non-interchangeable `QuantStorageAbiProfileId`, and returns the complete immutable profile rather than a raw digest. The versioned registry stores a canonical duplicate-free profile set indexed by that typed ID and rejects one ID paired with non-equivalent canonical profile bytes as a hard collision. Unknown version, unspecified/unknown closed enum, missing capability membership or any profile/storage mismatch fails; no string/digest-only constructor exists. Coordinate length/range checks, nonzero arena ordinal, digest algorithm/width and owner consistency fail before construction. No scoped composite receives another digest domain.

  `ExecutableIdentity.h` owns the remaining standalone ID wrappers and schema-verifying value builders:

  ```cpp
  class TargetVariantId;     // wafer.target-variant.v1, record type 17
  class ExecutableVariantId; // wafer.executable-variant.v1, record type 18
  class EntryId;             // wafer.executable-entry.v1, record type 19
  class RankClassId;         // (ExecutableVariantId, uint32 class ordinal)
  class CompletionNodeId;    // (ExecutableVariantId, uint64 node ordinal)
  class IterationDomainId;   // (ExecutableVariantId, nonzero uint32 ordinal)
  class ActivationPredicateId; // (ExecutableVariantId, nonzero uint64 ordinal)
  class ResourceRealizationRecordKey; // serialized coverage owner
  class StreamWindowId;      // (ResourceRealizationRecordKey, nonzero u32)
  class TransportActionId;   // (EntryId, entry-local uint64 ordinal)
  class TransportBindingMemberId; // (TransportActionId, action-local uint32 ordinal)

  llvm::Expected<TargetVariantId> buildTargetVariantId(
      const identity::proto::TargetVariantIdentity &identity,
      CanonicalEncodingContext &encoding);
  llvm::Expected<ExecutableVariantId> buildExecutableVariantId(
      const identity::proto::ExecutableVariantIdentity &identity,
      CanonicalEncodingContext &encoding);
  llvm::Expected<EntryId>
  buildEntryId(const identity::proto::EntryIdentity &identity,
      CanonicalEncodingContext &encoding);
  llvm::Expected<RankClassId>
  buildRankClassId(ExecutableVariantId variant, uint32_t classOrdinal);
  llvm::Expected<CompletionNodeId>
  buildCompletionNodeId(ExecutableVariantId variant, uint64_t nodeOrdinal);
  llvm::Expected<IterationDomainId>
  buildIterationDomainId(ExecutableVariantId variant, uint32_t domainOrdinal);
  llvm::Expected<ActivationPredicateId>
  buildActivationPredicateId(ExecutableVariantId variant,
                             uint64_t predicateOrdinal);
  llvm::Expected<ResourceRealizationRecordKey>
  buildResourceRealizationRecordKey(
      ResourceId resource, TargetVariantId target,
      ExecutableVariantId variant,
      llvm::ArrayRef<RankClassId> rankClassCoverage,
      llvm::ArrayRef<ProjectionSetId> projectionSetCoverage);
  llvm::Expected<StreamWindowId> buildStreamWindowId(
      ResourceRealizationRecordKey realization, uint32_t windowOrdinal);
  llvm::Expected<TransportActionId>
  buildTransportActionId(EntryId entry, uint64_t actionOrdinal);
  llvm::Expected<TransportBindingMemberId> buildTransportBindingMemberId(
      TransportActionId action, uint32_t memberOrdinal);
  ```

  These inputs are generated typed messages but are not trusted. Each top-level builder rejects unknown fields, missing/unspecified nested facts, noncanonical sets and an embedded ID whose canonical proof does not match its digest, then calls `encodeIdentityMessage` with its explicit encoding context and the exact record/domain pair 17/`wafer.target-variant.v1`, 18/`wafer.executable-variant.v1` or 19/`wafer.executable-entry.v1`. `TargetVariantIdentity` contains only source-environment compatibility, target ABI and required capabilities. `ExecutableVariantIdentity` contains exactly distributed-program semantic ID, normalized `ShapeGuardRef` and verified `TargetVariantId`; projection is a separate committed ref. `EntryIdentity` contains exactly distributed-program semantic ID, `ComponentId`, logical entry ordinal and optional per-rank-static `ExecutionInstanceId`; function/symbol, target, shape, rank class and module are excluded. `ComponentId` remains the `uint32` structural ordinal scoped to one distributed program's explicitly ordered component records. It is encoded together with that program semantic ID, never promoted to another global digest/domain and never derived from the component symbol, function name or map iteration. `RankClassId`, `CompletionNodeId`, `IterationDomainId`, `ActivationPredicateId`, `ResourceRealizationRecordKey`, `StreamWindowId`, `TransportActionId` and `TransportBindingMemberId` retain the exact scoped composites in tasks/01 rather than acquiring more digest domains. Predicate ordinal zero, duplicate key, wrong executable-variant owner and cross-variant activation reference fail; record-5/10 generated messages import the same nested value and descriptor-driven WCRE field rather than copying its tuple. Realization rank/projection coverage sets must be nonempty, canonically sorted, duplicate-free and owned by the declared executable variant/target; stream-window ordinal must be nonzero. Tests assert `ResourceRealizationRecordKey` and runtime `ResourceRealizationLookupKey` are not constructible/assignable/comparable. Owner verifiers remain responsible for canonical ordinal assignment, predicate canonical tuple/order/coverage, uniqueness, no realization overlap/hole and complete stream coverage.

  Every shared ID value has a private unchecked constructor, canonical comparison/order, an explicit generated-message encoder and an `llvm::Expected` decoder that rejects unknown fields, wrong owner, zero where forbidden, overflow and noncanonical nested values. All record-specific factories not expanded in this snippet, including records 2/3/5/6/11/12, use the same required trailing `CanonicalEncodingContext &` parameter; no cross-plan adapter may add an unbounded convenience overload. Add compile-time assertions that unrelated IDs are not constructible, assignable, comparable or implicitly convertible to one another or to raw bytes/integers. Add round-trip, wrong-owner, reordered-input and same-local-ordinal/different-owner tests for every scoped type. For record 20, perturb every profile field and require a different ID; reorder only canonical sets and require equality; reject a claimed ID/record mismatch, duplicate registry ID and an injected equal-ID/non-equivalent-record collision before lookup is exposed. Tests also rename component/function symbols without changing IDs, perturb every true owner field, reorder semantic sets, and prove path/name independence and mutual type/domain separation. `SemanticIdentityStreamingTest` constructs 100K and 1M-member generated messages with large SET projections and requires the Task 2 streaming and peak-memory gates through reflection; it verifies each strong ID remains fixed-size while collision records share one backing. Package/runtime headers include these ABI headers and generated messages; they do not redeclare opaque wrappers.

  `WaferCompilerIdentity` initially contains only compilable shared error/diagnostic adaptation in `IdentityBuilderSupport.h/.cpp`; it links `WaferABI` plus MLIR IR and does not reference ODS ops that have not landed. Cross-plan file ownership is fixed without placeholder headers: typed-program tasks create `include/Wafer/Compiler/{Model,Target,Distributed,Executable}IdentityBuilder.h` with matching `lib/Wafer/Compiler/*.cpp`; `ExecutableIdentityBuilder` is the only IR-to-message adapter for record types 17-19, and whole-variant tasks consume or extend that owner rather than recreating it. This plan adds static/KAD/target builders in later tasks. All are added to the one `WaferCompilerIdentity` target, extract facts from verified IR, call generated-message/value builders, and never own field numbers or a second WCRE codec.

  `ExternalSymbolRegistry.h/.cpp` is the only symbol truth shared by static closure resolution, CRT/symbol-set digest construction, target-call signature checks and final-link undefined-symbol policy. A reviewed versioned entry contains a stable structural identity, exact ASCII linker spelling, closed function/data kind, typed signature and target compatibility predicate. The lookup spelling is used only to resolve an entry; static-function identity encodes the stable entry identity/signature, and build-profile hashing encodes the same canonical entries plus their ABI spellings. The registry rejects duplicate spelling/identity, signature mismatch, incompatible target, unknown version and same identity with unequal entry facts. It has a private constructor and no factory accepting a caller string set or allowlist. Non-Wafer platform undefined symbols are a versioned closed subset relation in this same registry, not a third table.

  `Quantization.h/.cpp` verifies the shared mathematical affine/block-scaled descriptor and physical storage encoding independently. Closed enums and checked integers cover expressed/storage/accumulator/result types, axis/group/block semantics, scale/zero-point representation, rounding/saturation/tail/NaN/Inf/subnormal/overflow policy, packing/order/alignment/padding and exact checked byte count. Unknown fields/versions/enums, ambiguous scale ownership, overflow and semantic/storage mismatch fail. The values are non-aggregate, canonically comparable and contain no MLIR/path/name. Whole-variant's MLIR attrs/types are compiler adapters around these ABI values; KAD, profile registry, target lowering and package import the same generated messages/value verifiers. Add cross-layer fixtures that serialize once, verify in two processes and reject a one-field mismatch instead of silently converting between duplicate structs.

- [ ] **Step 5: Add descriptor-set and unknown-field gates**

  Create `wafer-artifact-inspect` as a typed diagnostic tool that links `WaferABI` and generated schemas. Its first command is `verify-schema`; it does not accept raw WCRE bytes or issue semantic identities. `wafer-semantic-identity-schema.test` asks pinned protoc for a descriptor set, runs this command, and FileChecks record type/version, every custom option, unique semantic-schema record assignments through 20 and deliberate 7/10/21 ownership gaps; once the state-migration schema lands, the global descriptor-set gate requires record 21 exactly once under that owner. Construct a wire message with an appended unknown field and verify `encodeIdentityMessage` rejects it before digest calculation.

- [ ] **Step 6: Run focused gates**

  ```bash
  cmake --build build/wafer-dev --target WaferSemanticIdentityProto WaferUnitTests wafer-artifact-inspect -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv build/wafer-dev/test/Tools/wafer-semantic-identity-schema.test
  ```

- [ ] **Step 7: Commit**

  ```bash
  git add schema CMakeLists.txt tools/wafer-artifact-inspect \
    include/Wafer/ABI/SemanticIdentity.h include/Wafer/ABI/ModelIdentity.h \
    include/Wafer/ABI/ResourceIdentity.h \
    include/Wafer/ABI/DistributedIdentity.h include/Wafer/ABI/TargetIdentity.h \
    include/Wafer/ABI/ExecutableIdentity.h include/Wafer/ABI/ExternalSymbolRegistry.h \
    include/Wafer/ABI/Quantization.h \
    lib/Wafer/ABI/SemanticIdentityInternal.h \
    lib/Wafer/ABI/SemanticIdentity.cpp lib/Wafer/ABI/ModelIdentity.cpp \
    lib/Wafer/ABI/ResourceIdentity.cpp \
    lib/Wafer/ABI/DistributedIdentity.cpp lib/Wafer/ABI/TargetIdentity.cpp \
    lib/Wafer/ABI/ExecutableIdentity.cpp lib/Wafer/ABI/ExternalSymbolRegistry.cpp \
    lib/Wafer/ABI/Quantization.cpp \
    lib/Wafer/ABI/CMakeLists.txt include/Wafer/Compiler \
    lib/Wafer/Compiler/IdentityBuilderSupport.cpp \
    lib/Wafer/Compiler/CMakeLists.txt lib/Wafer/CMakeLists.txt \
    unittests/ABI/SemanticIdentitySchemaTest.cpp \
    unittests/ABI/SemanticIdentityStreamingTest.cpp \
    unittests/ABI/ResourceIdentityTest.cpp \
    unittests/ABI/ScopedIdentityTest.cpp \
    unittests/ABI/ExecutableIdentityTest.cpp \
    unittests/ABI/ExternalSymbolRegistryTest.cpp \
    unittests/ABI/QuantizationTest.cpp unittests/CMakeLists.txt \
    test/Tools/wafer-semantic-identity-schema.test
  git commit -m "Define the semantic identity schema registry"
  ```

### Task 3A: Pre-Typed Kernel ABI Wire and Runtime-Safe Value Foundation

**Files:**
- Create: `schema/wafer/kernel_abi.proto`
- Modify: `schema/CMakeLists.txt`
- Create: `include/Wafer/ABI/ArtifactAdmissionLimits.h`
- Create: `lib/Wafer/ABI/ArtifactAdmissionLimits.cpp`
- Create: `lib/Wafer/ABI/ProtoAdmissionInternal.h`
- Create: `lib/Wafer/ABI/ProtoAdmissionInternal.cpp`
- Create: `include/Wafer/ABI/KernelAbiTypes.h`
- Create: `lib/Wafer/ABI/KernelAbiTypes.cpp`
- Create: `include/Wafer/ABI/KernelAbiDescriptor.h`
- Create: `lib/Wafer/ABI/KernelAbiDescriptor.cpp`
- Create: `lib/Wafer/ABI/KernelAbiProto.cpp`
- Create: `include/Wafer/ABI/Tx81CommandAbi.h`
- Create: `include/Wafer/ABI/Tx81Command.h`
- Create: `lib/Wafer/ABI/Tx81Command.cpp`
- Modify: `runtime/wafer_crt/include/wafer_tx81_crt.h`
- Modify: `lib/Wafer/ABI/CMakeLists.txt`
- Modify: `tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp`
- Create: `unittests/ABI/ArtifactAdmissionLimitsTest.cpp`
- Create: `unittests/ABI/ProtoAdmissionTest.cpp`
- Create: `unittests/ABI/KernelAbiTypesTest.cpp`
- Create: `unittests/ABI/KernelAbiDescriptorTest.cpp`
- Create: `unittests/ABI/Tx81CommandAbiTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/Tools/wafer-kernel-abi.test`

**Consumes:** Tasks 1-3's pinned Protobuf, WCRE core, semantic-identity messages/options, shared quant/storage values and target/projection identity wrappers. It consumes no MLIR or executable ODS type.

**Produces:** the complete ABI-reviewed `kernel_abi.proto`, generated `WaferKernelAbiProto`, sole numeric/wire owners for `SlotId`, `CompletionExportId`, `StateSlotVersionRole` and completion outcomes, the sole C-compatible `Tx81CommandAbi.h` with every low-precision/DTE V1 layout/prototype plus immutable runtime-safe command values/validators, validated `ArtifactAdmissionLimits` including reader/worker/module-byte/open-FD budgets, immutable runtime-safe projection/KAD values, record-7 semantic verifier/digest and limited deterministic delivery parser in `WaferABI`. Typed executable/whole/correctness/package plans depend on this task and must not duplicate these enums, command records/prototypes or admission fields.

```cpp
namespace wafer::abi {
struct ArtifactAdmissionLimitConfig {
  uint64_t maxDeliveryBytes;
  uint64_t maxKadDeliveryBytes;
  uint64_t maxInlineRecordBytes;
  uint32_t maxProtobufRecursionDepth;
  uint64_t maxAggregateStringBytes;
  uint64_t maxLocatorBytes;
  uint64_t maxProfileRecords;
  uint64_t maxKadDescriptors;
  uint64_t maxKadSlots;
  uint64_t maxCompletionExports;
  uint64_t maxAbiNotes;
  uint64_t maxElfProgramHeaders;
  uint64_t maxElfSectionHeaders;
  uint64_t maxArtifactMembers;
  uint64_t maxPhysicalBlobs;
  uint64_t maxSingleVerifiedModuleBytes;
  uint64_t maxActiveReaders;
  uint64_t maxConcurrentVerificationWorkers;
  uint64_t maxSimultaneouslyVerifiedModuleBytes;
  uint64_t maxOpenFileDescriptors;
  uint64_t maxNestedRecords;
  uint64_t maxAggregateRecords;
};

class ArtifactAdmissionLimits final {
public:
  const ArtifactAdmissionLimitConfig &config() const;
private:
  ArtifactAdmissionLimits() = delete;
  friend llvm::Expected<ArtifactAdmissionLimits>
  validateArtifactAdmissionLimits(const ArtifactAdmissionLimitConfig &);
};

llvm::Expected<ArtifactAdmissionLimits>
validateArtifactAdmissionLimits(const ArtifactAdmissionLimitConfig &config);

class SlotId final {
public:
  static llvm::Expected<SlotId> create(uint32_t ordinal);
  uint32_t ordinal() const;
private:
  SlotId() = delete;
};

class CompletionExportId final {
public:
  static llvm::Expected<CompletionExportId> create(uint32_t ordinal);
  uint32_t ordinal() const;
private:
  CompletionExportId() = delete;
};

class ProjectionRelocationSlot;
class RelocatableProjectionSlotSchema final {
public:
  uint32_t schemaVersion() const;
  llvm::ArrayRef<ProjectionRelocationSlot> orderedSlots() const;
private:
  RelocatableProjectionSlotSchema() = delete;
};

class ProjectionDependency final {
public:
  enum class Kind : uint8_t { Pinned, Relocatable };
  Kind kind() const;
  const ProjectionSetId &pinnedProjectionSetId() const;
  const RelocatableProjectionSlotSchema &relocatableSlotSchema() const;
private:
  ProjectionDependency() = delete;
};

class SerializedKernelAbiDescriptor final {
public:
  uint64_t exactSize() const;
  const ContentDigest &contentDigest() const;
  const ImmutableByteBackingRef &backing() const;
  llvm::Error writeTo(ByteSink &sink) const;
private:
  SerializedKernelAbiDescriptor() = delete;
  /* immutable bounded deterministic-Protobuf backing */
};

class VerifiedKernelAbiDescriptor final {
public:
  const KernelAbiDescriptor &descriptor() const;
  const CanonicalRecordBackingRef &canonicalRecordBacking() const;
  const KernelAbiSemanticDigest &digest() const;
  const SerializedKernelAbiDescriptor &delivery() const;
private:
  VerifiedKernelAbiDescriptor() = delete;
  friend llvm::Expected<VerifiedKernelAbiDescriptor>
  verifyKernelAbiDescriptor(const KernelAbiDescriptor &,
      CanonicalEncodingContext &);
  friend llvm::Expected<VerifiedKernelAbiDescriptor>
  parseAndVerifyKernelAbiDescriptor(
      const ImmutableByteBackingRef &, const ArtifactAdmissionLimits &,
      CanonicalEncodingContext &);
};

llvm::Expected<VerifiedKernelAbiDescriptor>
verifyKernelAbiDescriptor(
    const KernelAbiDescriptor &descriptor,
    CanonicalEncodingContext &encoding);
llvm::Expected<VerifiedKernelAbiDescriptor>
parseAndVerifyKernelAbiDescriptor(
    const ImmutableByteBackingRef &deliveryBytes,
    const ArtifactAdmissionLimits &limits,
    CanonicalEncodingContext &encoding);

enum class Tx81CommandKind : uint8_t {
  QuantizedGemm,
  MxfpDecode,
  DteSend,
  DteRecv,
  DteWait,
};

namespace detail {
struct VerifiedTx81CommandValueStorage;
class Tx81CommandValidator;
} // namespace detail

class VerifiedTx81CommandValue final {
public:
  VerifiedTx81CommandValue(const VerifiedTx81CommandValue &) = default;
  VerifiedTx81CommandValue(VerifiedTx81CommandValue &&) = default;
  Tx81CommandKind kind() const;
  /* typed per-kind const C-record accessors; wrong-kind access is rejected */
private:
  VerifiedTx81CommandValue() = delete;
  std::shared_ptr<const detail::VerifiedTx81CommandValueStorage> storage_;
  friend class detail::Tx81CommandValidator;
};

namespace detail { // noninstalled shared Protobuf admission engine
class ProtoWireAdmissionPlan;
class ProtoWireAdmissionBudget;
class VerifiedProtoWirePreflight;
class ProtoWireAdmissionPlanBuilder;

llvm::Expected<VerifiedProtoWirePreflight> preflightProtoWire(
    const ImmutableByteBackingRef &backing,
    const ProtoWireAdmissionPlan &plan,
    const ProtoWireAdmissionBudget &budget);

llvm::Error verifyParsedProtoMatchesPreflight(
    const google::protobuf::Message &message,
    const VerifiedProtoWirePreflight &preflight);
} // namespace detail
} // namespace wafer::abi
```

- [ ] **Step 1: Submit the complete `kernel_abi.proto` as one ABI review**

  Import `wafer/semantic_identity.proto` and reuse its WCRE options. Define the complete versioned descriptor: typed digest, mandatory target-environment fingerprint, LLVM symbol/return type, ordered slots, closed projection dependency and ordered completion exports. `ProjectionDependency` is exactly one of pinned `{ProjectionSetId}` or relocatable `{versioned ordered endpoint/control/status slots with finite allowed values}`. Empty/both arms, wrong owner, unknown schema version, unbounded value set and slot-schema mismatch fail.

  This schema is the sole numeric owner of distinct KAD-scoped zero-based contiguous `SlotId`/`CompletionExportId`, `StateSlotVersionRole { none=0, current, candidate, in_place }`, and completion outcomes `success=0`, `timeout=1`, `transport_error=2`, `peer_failure=3`, `pending=UINT32_MAX`; reserve every removed value. A slot includes role, LLVM representation, address space/access/alignment, shared semantic tensor and physical storage descriptors, capacity/range, alias and state-version role. Launch-visible packed/scale/zp/staging/cross-entry-scratch slots carry the exact record-20 profile ID; entry-local decode destination/scale copy/scratch are absent. No independent ABI ID, `ResourceId`, executable/rank salt, path/name, instruction list or schedule is legal. Review field numbers, WCRE kinds/order, unions, versions and reserves in one patch; generated C++ stays in the build tree.

  In the same shared-foundation review, submit C-compatible `Tx81CommandAbi.h` as the only layout/prototype owner for all `QuantizedGemm`, `MxfpDecode`, `DteSend`, `DteRecv` and `DteWait` V1 arms fixed by `tasks/14`. It uses fixed-width C types, explicit version/size/alignment, C `_Static_assert`/C++ `static_assert` for every offset and `extern "C"` guards. `wafer_tx81_crt.h` and compiler C++ include it; neither mirrors a record or prototype. Declaring all closed arms here does not make a target symbol available: the external-symbol registry/context keeps low-precision/DTE arms structured-unsupported until Tasks 9/10 land their audited CRT and lowering.

- [ ] **Step 2: Implement runtime-safe typed values and the sole record-7 verifier**

  Implement private constructors, canonical comparison and generated-message conversions in `KernelAbiTypes`/`KernelAbiDescriptor`; no header includes MLIR. `verifyKernelAbiDescriptor` checks all descriptor-local slot/type/storage/profile/projection/completion invariants, reconstructs record 7 through limit-aware `encodeIdentityMessage`, computes `wafer.kernel-abi.v1`, and streams pinned deterministic delivery bytes into one immutable bounded `SerializedKernelAbiDescriptor`. The proof shares its canonical/delivery backings and never stores parallel record-sized vectors. It can verify ABI semantics without knowing an `ExecutableOp`; the later compiler adapter proves cross-IR relations.

  `Tx81Command.cpp` owns the closed `Tx81CommandKind` union and per-arm layout-local validator for version/size/reserved bits, enums, counts/spans and status values. It returns only immutable private-construction `VerifiedTx81CommandValue`; it knows no MLIR, committed op, allocation or target artifact. Tests compile the same header as C and C++, compare every layout/prototype, reject wrong-kind access and every single-field mutation, and require no `Target/LowPrecisionAbi.h` or `Target/DTEAbi.h` mirror.

  Add positive IO/weight/state/workspace/launch-visible-low-precision/pinned/relocatable/completion fixtures and negative version/order/type/storage/profile/projection/unknown-field matrices. Assert non-aggregate proof types and compile-time noninterchangeability of SlotId, CompletionExportId and every imported enum. Typed/whole tests consume these exact generated values rather than creating ODS-local numeric copies.

- [ ] **Step 3: Apply admission limits before parsing or allocation**

  Define the validated positive `ArtifactAdmissionLimits` fixed later in this plan, including distinct delivery/KAD/inline-record byte caps plus active-reader, concurrent-verification-worker, simultaneous-module-byte and open-FD budgets. Validation rejects zero/overflow and infeasible cross-field combinations such as simultaneous bytes below one accepted module, worker count above reader capacity or FD capacity below the minimum handles required by one reader. A runtime verification session reserves these four dimensions atomically; live shared service capacity is supplied later by `ArtifactVerificationBudgetCapability` and the effective bound is the per-field minimum, never a second serialized/ad-hoc policy. Verification read/mapping buffers are charged to the simultaneous-module-byte reservation and canonical plan/scratch limits rather than an untracked fifth pool.

  Production `parseAndVerifyKernelAbiDescriptor` accepts only an owner-backed `ImmutableByteBackingRef` and runs two mandatory phases. First, WaferABI's single `ProtoAdmissionInternal` engine walks the wire stream with a schema-owned immutable admission plan derived from generated descriptors/options. Using fixed per-field/category counters plus a depth-bounded stack, it validates tags, legal wire types, lengths/varints, packed fields, recursive message boundaries and unknown fields; it checked-counts aggregate strings/bytes, nested records, slots, completions, profile refs and total records before constructing a generated message or any proportional typed graph. No caller can supply or weaken the admission plan, field map or expected counters. It returns a non-forgeable preflight proof bound to the exact backing digest/size/schema/version. Only then does the parser configure `CodedInputStream` total-byte/recursion limits and invoke generated parsing. A second descriptor walk must match every preflight count/span/category exactly before semantic-object construction, then the same KAD verifier runs with the explicit deployment-owned canonical encoding context. The proof's `SerializedKernelAbiDescriptor` retains that exact backing instead of copying or reserializing caller bytes. Counter overflow, wire/schema error and category limit breach have distinct errors and return no partial proof.

  Instrument generated-message/arena allocation in `ProtoAdmissionTest` and KAD tests: every recomputed-digest `limit+1`, packed/unpacked alternate encoding, deep nested length and oversized string case must fail during preflight with zero generated-message allocations; at-limit legal inputs proceed to parse, match the preflight proof and remain within the byte/count budget. A noninstalled inline test/debug adapter first rejects `bytes.size() > limits.maxInlineRecordBytes`, move-copies accepted bytes into a sealed backing, and then calls the production parser; it cannot create a trusted delivery ref. Raising a legal deployment/canonical limit changes acceptance only, never record bytes or digest.

- [ ] **Step 4: Prove schema and delivery determinism**

  Extend `wafer-artifact-inspect` with `verify-kad --input=<delivery.pb>` plus explicit validated limits. Run compiler-independent fixtures in two processes and require identical deterministic bytes/digest. Alternate legal Protobuf field order must reconstruct the same WCRE semantics; appended unknown fields, over-limit nesting/counts and claimed digest mismatch fail before returning a proof.

- [ ] **Step 5: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferKernelAbiProto WaferUnitTests wafer-artifact-inspect -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv build/wafer-dev/test/Tools/wafer-kernel-abi.test
  git add schema/wafer/kernel_abi.proto schema/CMakeLists.txt \
    include/Wafer/ABI/ArtifactAdmissionLimits.h \
    include/Wafer/ABI/KernelAbiTypes.h include/Wafer/ABI/KernelAbiDescriptor.h \
    include/Wafer/ABI/Tx81CommandAbi.h include/Wafer/ABI/Tx81Command.h \
    lib/Wafer/ABI/ArtifactAdmissionLimits.cpp \
    lib/Wafer/ABI/ProtoAdmissionInternal.h \
    lib/Wafer/ABI/ProtoAdmissionInternal.cpp lib/Wafer/ABI/KernelAbiTypes.cpp \
    lib/Wafer/ABI/KernelAbiDescriptor.cpp lib/Wafer/ABI/KernelAbiProto.cpp \
    lib/Wafer/ABI/Tx81Command.cpp \
    runtime/wafer_crt/include/wafer_tx81_crt.h \
    lib/Wafer/ABI/CMakeLists.txt tools/wafer-artifact-inspect \
    unittests/ABI/ArtifactAdmissionLimitsTest.cpp \
    unittests/ABI/ProtoAdmissionTest.cpp \
    unittests/ABI/KernelAbiTypesTest.cpp unittests/ABI/KernelAbiDescriptorTest.cpp \
    unittests/ABI/Tx81CommandAbiTest.cpp \
    unittests/CMakeLists.txt test/Tools/wafer-kernel-abi.test
  git commit -m "Define the runtime-safe kernel ABI contract"
  ```

### Task 4: MLIR Static Function Identity and Clone-Safe Closure Proof

**Files:**
- Create: `include/Wafer/Compiler/StaticFunctionIdentity.h`
- Create: `lib/Wafer/Compiler/StaticFunctionIdentity.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Create: `unittests/Compiler/StaticFunctionIdentityTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Consumes:** a fully gated commit-ready clone or committed executable module, typed `(EntryId, func::FuncOp)` roots, the reviewed external-symbol registry, normative static-function messages, registered type/attr schema and the closure-aware structural algorithm in `tasks/14`.

**Produces:** `WaferCompilerIdentity` and the only MLIR-dependent `wafer.static-function.v1` structural encoder. Whole-variant projection and artifact assembly call this API; runtime/package do not link it.

```cpp
namespace wafer::compiler_identity {
inline constexpr uint32_t kStaticFunctionEncoderVersion = 1;

class VerifiedStaticFunctionClosure;

class StaticFunctionIdentity final {
public:
  uint32_t encoderVersion() const;
  const abi::CanonicalRecordBackingRef &canonicalRecordBacking() const;
  const abi::StaticFunctionDigest &digest() const;
private:
  StaticFunctionIdentity() = delete;
  friend mlir::FailureOr<StaticFunctionIdentity>
  encodeStaticFunctionIdentity(mlir::func::FuncOp function,
                               const VerifiedStaticFunctionClosure &closure,
                               abi::CanonicalEncodingContext &encoding);
};

class ClosureStructuralDigest final; // compiler-local, no WCRE domain
class PrivateNodeStructuralDigest final; // compiler-local, no WCRE domain

enum class PrivateDependencyClass : uint8_t {
  NonClonable,
  CloneablePureFunction,
  CloneableImmutableGlobal,
};

class VerifiedCloneableDependency final {
public:
  PrivateDependencyClass dependencyClass() const;
  const PrivateNodeStructuralDigest &structuralDigest() const;
private:
  VerifiedCloneableDependency() = delete;
  /* complete purity/address/initializer proof retained by closure owner */
};

struct StaticFunctionRoot {
  abi::EntryId entryId;
  mlir::func::FuncOp function;
};

class VerifiedStaticClosureUnit final {
public:
  const ClosureStructuralDigest &closureStructuralDigest() const;
  llvm::ArrayRef<abi::EntryId> sortedEntryIds() const;
  llvm::ArrayRef<VerifiedCloneableDependency>
  sortedCloneableDependencies() const;
private:
  VerifiedStaticClosureUnit() = delete;
  /* constructed only inside the complete closure proof */
};

class VerifiedStaticFunctionClosure final {
public:
  llvm::ArrayRef<VerifiedStaticClosureUnit> canonicalUnits() const;
  const VerifiedStaticClosureUnit &unitForEntry(const abi::EntryId &entry) const;
private:
  VerifiedStaticFunctionClosure() = delete;
  /* resolveStaticFunctionClosure is the only factory */
};

mlir::FailureOr<VerifiedStaticFunctionClosure>
resolveStaticFunctionClosure(mlir::ModuleOp owner,
    llvm::ArrayRef<StaticFunctionRoot> roots,
    const abi::VerifiedExternalSymbolRegistry &externalSymbols,
    abi::CanonicalEncodingContext &encoding);

mlir::FailureOr<StaticFunctionIdentity>
encodeStaticFunctionIdentity(mlir::func::FuncOp function,
                             const VerifiedStaticFunctionClosure &closure,
                             abi::CanonicalEncodingContext &encoding);
} // namespace wafer::compiler_identity
```

- [ ] **Step 1: Add semantic-equivalence and differentiation tests**

  Positive equivalence pairs must differ only in SSA print names, locations, symbol name of the entry function, private helper/global names, declaration order, diagnostic attr or pass-trace attr and still yield identical closure proof plus streamed record/digest. Reparse in two independent MLIR contexts and clone once to prove pointer independence. Assert all proof/dependency types are non-aggregate, not publicly default/raw-digest constructible and expose only const accessors.

  Negative differentiation pairs change one constant bit pattern, signedness, tensor/memref shape, layout, memory space, operand order, result type, successor, block order, region body, function type, visibility, private helper body/global initializer, typed entry target or reviewed external ABI target and must change the digest. Unsupported unregistered dialect/type/attr, unresolved/ambiguous symbol, private recursion, cyclic global initializer, unclassified indirect call, digest/canonical-label collision, ordinary candidate entry that has not passed the commit-ready gate, opaque/resource attr and dynamic target geometry must fail before a backing is returned.

  Add a scale fixture with thousands of typed entries sharing one private helper and immutable lookup table. When `MemoryEffectOpInterface`, call graph, external registry and address-use analysis prove the helper/table pure and non-address-significant, entries remain independently partitionable and every unit references the same canonical clone dependency digests. Changing the helper to write memory, capture/compare its address, call an unknown indirect target or read a mutable global must classify the relation as non-clonable and connect all affected entries; an oversized resulting core fails closed. Equal helper/global structural records dedup, while an injected equal digest with unequal records fails.

- [ ] **Step 2: Resolve symbols structurally before encoding a function**

  `resolveStaticFunctionClosure` starts from all typed entry roots together under the transaction's bounded encoding context, resolves the complete function/global dependency graph, and proves every reference is exactly one of: another entry encoded by `EntryId`; a private function/global encoded by its bottom-up structural node digest and dependency classification; or a reviewed external ABI symbol encoded by the external registry's stable ABI identity. Source private symbol spelling and symbol-table path are lookup inputs only and never enter a digest. Canonical node/edge/set records use measured streaming/shared scratch rather than retaining a second printed/body byte copy. The resolver rejects unresolved/ambiguous refs, private recursion, cyclic global initializers, unclassified indirect calls and a digest collision whose streamed canonical node records differ.

  Classify entry-to-entry edges, mutable globals, address-taken/address-significant functions/globals, shared state/control, indirect/unknown effects and any unproved dependency as `NonClonable`. A private helper is `CloneablePureFunction` only when it has no address escape or indirect call, recursive `MemoryEffectOpInterface` plus reviewed external-registry analysis proves no side effects, all transitive globals are immutable/non-address-significant and no recursive SCC exists. A private global is `CloneableImmutableGlobal` only when immutable, initializer-acyclic and free of address identity/escape. Names never affect classification. Non-clonable connectivity alone forms canonical indivisible closure units; cloneable dependencies are retained as sorted structural-digest sets on each referencing unit and do not connect entries. The proof assigns digest-derived collision-checked labels and returns `VerifiedStaticClosureUnit[]` plus total `EntryId -> unit` mapping. Whole-variant commit and Task 8A reuse this proof/API; neither consumer reimplements graph discovery, purity analysis, partitioning or ref classification.

- [ ] **Step 3: Implement deterministic structural ordinals**

  Traverse regions by op region ordinal and blocks by region storage order. Assign block ordinals from zero; assign value ordinals to block arguments and op results in structural order. Encode operands/successors by ordinal, never printed names or pointers. Encode every op's registered name, explicit operand/result/successor/region counts, ordered operand refs, ordered result types, ASCII-sorted registered attrs, ordered successors and nested regions.

- [ ] **Step 4: Implement registered type/attr encoding**

  Populate the generated static-function identity messages for builtin integer/float/index/tensor/memref/function types and registered Wafer type/attrs. Encode float semantics plus exact fixed-width bits, integer bits and dense element payload. Encode every symbol reference only through the verified closure proof's typed entry/private-node-classification/external-ABI union; the source `SymbolRefAttr` spelling is never an identity field. Only `Location`, SSA spelling and schema-excluded diagnostic/pass-trace attrs are ignored; all unclassified fields fail.

  Finish through the record-6 factory, which calls limit-aware `encodeIdentityMessage(..., WCRERecordType::StaticFunction, encoding)` and the internal domain-separated digest core and returns `abi::StaticFunctionDigest` plus shared canonical backing. Only `encodeStaticFunctionIdentity` can construct the proof object. Do not print MLIR or serialize MLIR bytecode.

- [ ] **Step 5: Fix the cross-plan consumer handoff**

  Publish the following consumer contract in the header comments and unit test it through a small callback fixture: whole-variant resolves one `VerifiedStaticFunctionClosure` for all typed entries after every commit-ready gate, then `computeStaticFunctionSemanticDigest(ExecutableEntryOp)` looks up its root, calls `encodeStaticFunctionIdentity(function, closure, encoding)`, and stores only the returned typed digest plus transaction-shared collision backing inside that clone. It contains no structural walk, purity reclassification, second digest calculation, symbol-name encoding or field-number constants. Atomic commit reruns the same closure resolver/API from the committed module and requires identical non-clonable units, clone dependency records, canonical private labels and per-entry record streams/digests before exposing the executable. Task 8A reuses the committed closure proof algorithm and verifies equality; it does not create a second closure identity. The whole-variant plan owns its adapter source edit.

- [ ] **Step 6: Run focused gates**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  ```

- [ ] **Step 7: Commit**

  ```bash
  git add include/Wafer/Compiler/StaticFunctionIdentity.h \
    lib/Wafer/Compiler/StaticFunctionIdentity.cpp \
    lib/Wafer/Compiler/CMakeLists.txt \
    unittests/Compiler/StaticFunctionIdentityTest.cpp unittests/CMakeLists.txt
  git commit -m "Encode static function structural identity"
  ```

### Task 5: Pre-Commit Target Build Profile and Toolchain Foundation

**Files:**
- Create: `include/Wafer/ABI/TargetArtifacts.h`
- Create: `lib/Wafer/ABI/TargetArtifacts.cpp`
- Modify: `lib/Wafer/ABI/CMakeLists.txt`
- Create: `include/Wafer/Compiler/TargetBuildProfileBuilder.h`
- Create: `lib/Wafer/Compiler/TargetBuildProfileBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Create: `unittests/ABI/TargetBuildProfileTest.cpp`
- Create: `unittests/Compiler/TargetBuildProfileBuilderTest.cpp`
- Create: `unittests/Compiler/TargetCompilationContextTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Consumes:** Task 3's verified `QuantStorageAbiProfileRegistryV1` and `VerifiedExternalSymbolRegistry`, target environment fingerprint/capabilities, accepted storage-profile facts, the reviewed toolchain/device-link registry, exact CRT bytes and typed `ModulePartitionPolicyV1`.

**Produces:** immutable runtime-safe `TargetBuildProfile`; compiler-only non-forgeable `VerifiedTargetCompilationContext`; and move-only/private-construction `VerifiedTargetCompilationContextRegistry` that owns the canonical all-and-only context set keyed by typed `TargetVariantId` before whole-variant candidate formation. Task 3A's command ABI/value is available here, but the committed-op `VerifiedTx81CommandView`/resolver is created only by correctness Task 2 after Whole defines and commits `ExecutableEntryOp`. This task does not compute a target artifact fingerprint.

```cpp
namespace wafer::abi {
enum class PrelinkCloneLinkagePolicy : uint8_t {
  HiddenLinkOnceOdrComdatV1 = 1,
};

struct ModulePartitionPolicyV1 {
  static constexpr uint32_t kPolicyVersion = 1;
  static constexpr uint32_t kDependencyClassificationVersion = 1;
  uint32_t maxEntriesPerModule;
  uint64_t maxPrelinkObjectBytes;
  uint64_t maxCoreObjectBytes;
  uint64_t maxCloneObjectBytes;
  uint64_t maxDeduplicatedCloneBytesPerModule;
  uint64_t maxKadNoteBytes;
  PrelinkCloneLinkagePolicy cloneLinkagePolicy;
};

class VerifiedTargetToolchainProfileV1 final {
public:
  uint32_t profileVersion() const;
  const ContentDigest &compilerBinaryDigest() const;
  const ContentDigest &linkerBinaryDigest() const;
  const ContentDigest &objcopyBinaryDigest() const;
  const ContentDigest &sysrootDigest() const;
  const ContentDigest &crtObjectDigest() const;
  llvm::ArrayRef<std::string> canonicalCompileArguments() const;
  llvm::ArrayRef<std::string> canonicalLinkArguments() const;
  llvm::ArrayRef<std::string> canonicalObjcopyArguments() const;
  llvm::ArrayRef<std::pair<std::string, std::string>> allowedEnvironment() const;
private:
  VerifiedTargetToolchainProfileV1() = delete;
  /* reviewed registry verifier only */
};

class TargetBuildProfile final {
public:
  llvm::StringRef targetTriple() const;
  llvm::ArrayRef<std::string> isaExtensions() const;
  llvm::StringRef mabi() const;
  uint32_t kernelAbiSchemaVersion() const;
  uint32_t canonicalEncoderVersion() const;
  uint32_t staticFunctionEncoderVersion() const;
  const ContentDigest &crtAndSymbolSetDigest() const;
  const VerifiedExternalSymbolRegistry &externalSymbolRegistry() const;
  const QuantStorageAbiProfileRegistryV1 &quantStorageProfileRegistry() const;
  const VerifiedTargetToolchainProfileV1 &toolchainProfile() const;
  const ModulePartitionPolicyV1 &modulePartitionPolicy() const;
private:
  TargetBuildProfile() = delete;
  /* runtime-safe verifier or compiler builder only */
};

llvm::Expected<TargetBuildProfile> verifyTargetBuildProfile(
    const identity::proto::TargetBuildProfileV1 &profile,
    const TargetEnvironmentFingerprint &environment,
    const VerifiedExternalSymbolRegistry &externalSymbols,
    const ArtifactAdmissionLimits &limits);
} // namespace wafer::abi

namespace wafer::compiler {
namespace detail {
class TargetCompilationContextRegistryAccess;
class ExecutableProgramOutputAccess;
} // namespace detail

struct TargetToolchainLocators {
  std::string compiler;
  std::string linker;
  std::string objcopy;
  std::string sysroot;
};

class VerifiedTargetToolchainInvocation final {
public:
  const abi::VerifiedTargetToolchainProfileV1 &profile() const;
private:
  VerifiedTargetToolchainInvocation() = delete;
  /* verified locators and immutable derived argv/env remain private */
};

llvm::Expected<VerifiedTargetToolchainInvocation>
resolveTargetToolchainInvocation(
    const TargetToolchainLocators &locators,
    const abi::TargetBuildProfile &buildProfile);

struct TargetCompilationContextInput {
  abi::TargetVariantId targetVariantId;
  abi::VerifiedTargetEnvironment environment;
  abi::TargetCompatibilityContract topologyCompatibility;
  abi::TargetBuildProfile buildProfile;
  VerifiedTargetToolchainInvocation toolchain;
};

class VerifiedTargetCompilationContext final {
public:
  const abi::TargetVariantId &targetVariantId() const;
  const abi::VerifiedTargetEnvironment &environment() const;
  const abi::TargetEnvironmentFingerprint &environmentFingerprint() const;
  const abi::TargetCompatibilityContract &topologyCompatibility() const;
  const abi::TargetBuildProfile &buildProfile() const;
  const VerifiedTargetToolchainInvocation &toolchain() const;
  llvm::Error revalidateForTargetRequirement(
      const identity::proto::TargetVariantIdentity &requirement) const;
private:
  VerifiedTargetCompilationContext() = delete;
  /* bindTargetCompilationContext is the only factory */
};

llvm::Expected<VerifiedTargetCompilationContext>
bindTargetCompilationContext(TargetCompilationContextInput input);

class VerifiedTargetCompilationContextRegistry final {
public:
  VerifiedTargetCompilationContextRegistry(
      VerifiedTargetCompilationContextRegistry &&) noexcept;
  VerifiedTargetCompilationContextRegistry(
      const VerifiedTargetCompilationContextRegistry &) = delete;
  ~VerifiedTargetCompilationContextRegistry();
private:
  VerifiedTargetCompilationContextRegistry() = delete;
  class Storage;
  explicit VerifiedTargetCompilationContextRegistry(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
  friend llvm::Expected<VerifiedTargetCompilationContextRegistry>
  verifyTargetCompilationContextRegistry(
      llvm::SmallVector<identity::proto::TargetVariantIdentity>,
      llvm::SmallVector<VerifiedTargetCompilationContext>);
  friend class detail::ExecutableProgramOutputAccess;
  friend class detail::TargetCompilationContextRegistryAccess;
};

llvm::Expected<VerifiedTargetCompilationContextRegistry>
verifyTargetCompilationContextRegistry(
    llvm::SmallVector<identity::proto::TargetVariantIdentity>
        completeRequirements,
    llvm::SmallVector<VerifiedTargetCompilationContext> contexts);
} // namespace wafer::compiler

```

- [ ] **Step 1: Add failure-first profile and toolchain tests**

  Construct one environment with native INT8 and explicit MXFP capabilities, then build a registry containing both complete record-20 profiles. Verify canonical lookup by typed ID, source-order independence, and legal selection of either/both IDs by downstream verified instructions. Adding another verified profile is legal and does not mutate existing IDs. Reject unknown registry/profile version, missing/ambiguous environment capability, accepted-storage mismatch, duplicate ID, same-ID/non-equivalent canonical record, free-form profile name and raw digest substitution.

  Build a baseline toolchain profile and perturb compiler/linker/objcopy bytes, sysroot/CRT object bytes, target/ISA/MABI, every canonical argv list including COMDAT/localization controls, allowed environment, profile version, dependency-classification version, clone linkage policy and every core/clone/dedup limit one at a time. Each changes the verified profile; stale binary content, unregistered flag/env, mutable argv exposure and caller-supplied allowlist fail. Changing only a locator to byte-identical content preserves the profile. Verify `TargetBuildProfile.externalSymbolRegistry()` is the exact version/entry-identity set used to compute `crtAndSymbolSetDigest`; substituting an equal digest claim with a different registry version/entry fails. Bind two environments/profiles/invocations and reject every cross-pair, same fingerprint with a different capability/registry record, incompatible topology contract, target requirement mismatch and a context whose binary/environment generation changed after binding. Registry tests reject duplicate/missing/extra/cross-paired bindings, invalid requirement identity, forged private ordinal and lookup through another registry generation; source-order permutations produce the same canonical ID order. Move the registry through successive owners, destroy the source at each hop and require private lookup to remain valid. Assert build/toolchain/invocation/context/registry types are non-aggregate and have no public raw constructor or public size/vector/iteration/lookup accessor.

- [ ] **Step 2: Build the pre-commit profile from typed registries**

  Runtime-safe `verifyTargetBuildProfile` rejects unknown fields/versions, applies Task 3A admission limits before registry allocations, validates every toolchain/symbol/profile fact against the reviewed registries and environment, and returns the only immutable value consumed by fingerprint/set/runtime verification. Compiler-only `TargetBuildProfileBuilder` populates that generated message from attested locators/repo bytes and calls the same verifier; it does not construct the value directly. The global build profile contains the registry and registry schema, never a singular quant profile choice. Whole-variant candidate formation receives this object before producing low-precision tile/instruction/resource refs and may select only registry-owned `QuantStorageAbiProfileId`s that exactly match accepted storage; target environment IR itself does not own or synthesize the registry.

  Hash exact repo-owned CRT source/header, Task 3's canonical external symbol/signature entries, its versioned non-Wafer undefined subset and target ABI version. Paths/timestamps are excluded. The profile fixes dependency classification and `hidden_linkonce_odr_comdat_v1`, including digest-derived symbol/COMDAT construction and final localization arguments; unknown policy/version or a zero core/clone/dedup limit fails. The registry may expose a reviewed low-precision command ABI version to whole-variant formation before target implementation, but artifact publication requiring that profile stays unavailable until Task 9 lands the matching lowering/CRT/symbol proof; DTE publication similarly waits for Task 10. Those tasks update production symbol availability and therefore require rebuilding `TargetBuildProfile`, while the reviewed record-20 profile ID remains stable. Static closure resolution, target-call lowering, final-link undefined checks and `crtAndSymbolSetDigest` must resolve the same registry version and entry identities; no copied symbol table is permitted.

- [ ] **Step 3: Attest every tool input and close invocation arguments**

  Resolve the profile from the reviewed registry and caller-provided binary/sysroot locators, rehash every compiler/linker/objcopy/sysroot/CRT input, verify target/ISA/MABI and canonical arguments, capture only registry-approved codegen environment, and clear/reject all other environment variables. Paths remain locators. The invocation privately owns verified locators and immutable derived argv/env; no public compile/link API accepts a raw executable path, `fixedArguments`, extra flag, environment override or caller allowlist. Reattest immediately before every subprocess to detect replacement after resolution.

  `bindTargetCompilationContext` requires a typed target ID plus exact environment fingerprint/capability equality across the compatibility contract, build profile registries/toolchain facts and invocation profile, then stores one immutable generation-bound unit. `verifyTargetCompilationContextRegistry` separately move-consumes the complete canonical requirement messages and contexts, joins them by typed `TargetVariantId`, rejects missing/extra/duplicate/cross-paired entries, calls every context's `revalidateForTargetRequirement`, canonical-sorts by raw typed ID and seals the sole owner. It never accepts a caller-supplied digest-only entry or public size/iteration/lookup/mutable storage. Only `TargetCompilationContextRegistryAccess` and the lower executable-output adapter may inspect/resolve; Whole owners use the public move constructor to transfer the opaque registry and do not require reverse friends. `CompilationRequest` move-consumes this registry, not naked vectors, together with canonical policy/scratch ownership. The winning executable owner transfers it through committed attachment; `ProgramOutputTransaction` separately move-owns the `CanonicalEncodingSession`. No later API accepts an independent environment/profile/invocation tuple, replacement registry, public encoding/context accessor or default context.

- [ ] **Step 4: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add include/Wafer/ABI/TargetArtifacts.h \
    lib/Wafer/ABI/TargetArtifacts.cpp lib/Wafer/ABI/CMakeLists.txt \
    include/Wafer/Compiler/TargetBuildProfileBuilder.h \
    lib/Wafer/Compiler/TargetBuildProfileBuilder.cpp \
    lib/Wafer/Compiler/CMakeLists.txt \
    unittests/ABI/TargetBuildProfileTest.cpp \
    unittests/Compiler/TargetBuildProfileBuilderTest.cpp \
    unittests/Compiler/TargetCompilationContextTest.cpp unittests/CMakeLists.txt
  git commit -m "Verify target build and toolchain profiles"
  ```

## Post-Commit Target Artifact Path

### Task 6: Compiler Kernel ABI Construction and Cross-IR Verification

**Files:**
- Create: `include/Wafer/Compiler/KernelAbiBuilder.h`
- Create: `lib/Wafer/Compiler/KernelAbiBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Modify: `tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp`
- Create: `unittests/Compiler/KernelAbiBuilderTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/Tools/wafer-kernel-abi-from-executable.test`
- Create: `tools/check_target_dependency_layers.py`
- Create: `test/Tools/check-target-dependency-layers.test`

**Consumes:** Task 3A's complete runtime-safe KAD values/verifier, one live correctness-owned `conversion::ConvertedTargetLLVMModule` and one by-value owner-bound `conversion::ConvertedTargetEntryHandle` obtained from that module's bounded `entrySummary(ordinal)`. The public owner/summary expose no `ModuleOp`, `LLVMFuncOp`, `Operation *`, sealed boundary reference or handle-to-function map. The builder reuses the shared geometry contract but never consumes a pre-rewrite `VerifiedInstructionGeometry` object.

**Produces:** independent upper-layer CMake target `WaferCompilerKernelAbiAdapter`, containing the sole MLIR-to-KAD builder adapter in namespace `wafer::compiler_identity` and returning Task 3A's non-forgeable `VerifiedKernelAbiDescriptor`. It links `WaferInstrToTargetLLVM`, base `WaferCompilerIdentity` and `WaferABI`; base identity neither includes nor links conversion. It adds no schema, enum, ordinal or second semantic verifier.

```cpp
namespace wafer::compiler_identity {
mlir::FailureOr<abi::VerifiedKernelAbiDescriptor>
buildAndVerifyKernelAbiDescriptor(
    const conversion::ConvertedTargetLLVMModule &convertedModule,
    const conversion::ConvertedTargetEntryHandle &entry,
    const compiler::VerifiedTargetCompilationContext &targetContext,
    abi::CanonicalEncodingContext &encoding);
} // namespace wafer::compiler_identity
```

- [ ] **Step 1: Add compiler cross-IR failure tests**

  Positive cases cover committed IO, immutable weight, persistent state, workspace, launch-visible packed/scale/zp/stream-staging/cross-entry-scratch slots, pinned transport, relocatable endpoint/control/status slots and multiple completion exports. A pure tile-local MXFP decode destination/scale copy/scratch adds no KAD slot; a launch-visible source slot binds its exact profile ID and the adapter follows its sealed typed load/use-def relation into the internal SPM operand. Shape variants/ranks with identical function ABI produce the same Task 3A digest even when code/static-function digests differ.

  Negative cases cover candidate/noncommitted boundary, stale boundary/context/converted-module owner, a handle from another converted owner, forged/out-of-range ordinal, mismatched summary, missing or duplicate request-handle/entry-slot/LLVM parameter mapping, wrong scalar/pointer width, address space/access/alignment, semantic/storage/profile/capacity/alias mismatch, state-version policy mismatch, promoted tile-local scratch, missing launch-visible source/staging or transport slot, projection mismatch, completion export not bound to every covered variant DAG, `ResourceId` leakage and variant/rank salt. Passing a pre-conversion `VerifiedInstructionGeometry`, raw `ExecutableEntryOp`, independent `LLVMFuncOp`, summary without its owner or caller slot list is a compile-time/API failure. Compile-time checks require that the converted owner has only bounded `entryCount`/by-value `entrySummary` inspection and no public module, function, operation, boundary-view or mapping accessor, and that privately borrowed facts cannot be returned or retained. Assert the compiler target contains no copied numeric enum, schema field number, WCRE builder or alternate KAD verifier.

- [ ] **Step 2: Implement the one compiler adapter**

  First verify that the supplied handle was issued by the same live converted owner and that its ordinal/summary entry agrees. During the builder call only, `compiler_identity::detail::KernelAbiBuildAccess` borrows that owner's exact sealed boundary and handle-to-`LLVMFuncOp` relation, performs the cross-check and releases every borrow before returning; the result contains only the non-forgeable KAD. Re-run the shared geometry/unit/range contract over immutable boundary fields and borrowed converted LLVM parameter types; do not carry or trust the pre-rewrite proof object. Check every slot against its boundary-owned unique selected `ResourceRealizationRecordKey`/record for all covered axes and require semantic type, shared quant/storage descriptor, optional profile ID, capacity/range, state-version role and LLVM representation to agree. Only launch-visible roots become slots; follow their sealed typed load/use relation into entry-local SPM uses without serializing internal values. Recheck target context generation, mandatory environment, projection and completion relations, then call Task 3A's runtime-safe `verifyKernelAbiDescriptor(descriptor, encoding)` and adapt `llvm::Error` to an anchored MLIR diagnostic. Parameter names, `ResourceId`, LLVM text and vector position never recover ownership.

- [ ] **Step 3: Preserve conversion/KAD transaction ordering**

  Target conversion receives typed entry slots and returns a move-only converted-module owner whose bounded by-value summaries contain non-forgeable owner-bound handles but no authoritative boundary/MLIR view. Build each descriptor only after conversion succeeds and only from the live owner plus one of its handles; the target transaction exposes neither. A rewrite after conversion invalidates that owner and requires a new conversion, so no boundary/KAD proof can be reused across mutation. Return a KAD only after the scoped post-conversion cross-check and runtime-safe verifier pass. A second-entry KAD failure destroys the converted owner and every descriptor/object in that transaction. Package/runtime later compare selected `ResourceRealizationLookupKey` facts to the same descriptor fields; they never call this MLIR adapter.

- [ ] **Step 4: Lock the acyclic compiler dependency direction**

  `lib/Wafer/Compiler/CMakeLists.txt` creates `WaferCompilerKernelAbiAdapter` as a distinct target whose public implementation header may include the converted-owner types and whose source links `WaferInstrToTargetLLVM`, `WaferCompilerIdentity` and `WaferABI`. It does not add that header/source to base `WaferCompilerIdentity`. `check_target_dependency_layers.py` inspects the actual CMake target graph and installed-header include graph, standalone-compiles the base identity headers, and rejects any base-identity include/link to conversion, any conversion link back to the adapter/target artifacts, or any target-artifacts path that bypasses the adapter for KAD construction.

- [ ] **Step 5: Prove compiler output uses the shared wire verifier**

  `wafer-kernel-abi-from-executable.test` builds a descriptor from a committed entry through its converted owner/owner-bound handle, streams its Task 3A deterministic delivery backing, and invokes `wafer-artifact-inspect verify-kad` with explicit admission plus canonical contexts in two processes. Require identical digest/bytes and structured failure after mutating one executable slot, LLVM parameter, profile relation or delivery byte. The tool remains a runtime-safe consumer and does not learn MLIR.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferCompilerKernelAbiAdapter WaferUnitTests wafer-artifact-inspect -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv build/wafer-dev/test/Tools/wafer-kernel-abi-from-executable.test
  <configured-lit> -sv build/wafer-dev/test/Tools/check-target-dependency-layers.test
  python3 tools/check_target_dependency_layers.py
  git add include/Wafer/Compiler/KernelAbiBuilder.h \
    lib/Wafer/Compiler/KernelAbiBuilder.cpp lib/Wafer/Compiler/CMakeLists.txt \
    tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp \
    tools/check_target_dependency_layers.py \
    unittests/Compiler/KernelAbiBuilderTest.cpp unittests/CMakeLists.txt \
    test/Tools/wafer-kernel-abi-from-executable.test \
    test/Tools/check-target-dependency-layers.test
  git commit -m "Build kernel ABI descriptors from committed entries"
  ```

### Task 7: Final-Module Target Artifact Fingerprint Values

**Files:**
- Modify: `include/Wafer/ABI/TargetArtifacts.h`
- Modify: `lib/Wafer/ABI/TargetArtifacts.cpp`
- Modify: `lib/Wafer/ABI/CMakeLists.txt`
- Create: `unittests/ABI/TargetArtifactFingerprintTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Consumes:** Task 5's verified global `TargetBuildProfile`, Task 3A's closed projection dependency, one final packed module's canonical actual profile union and the normative target-artifact identity message.

**Produces:** typed `TargetArtifactFingerprint` distinct from the Task 3 `TargetEnvironmentFingerprint` and final content digest.

```cpp
namespace wafer::abi {
class UsedQuantStorageProfiles final {
public:
  llvm::ArrayRef<QuantStorageAbiProfileId> canonicalProfileIds() const;
private:
  UsedQuantStorageProfiles() = delete;
  friend llvm::Expected<UsedQuantStorageProfiles>
  verifyUsedQuantStorageProfiles(
      llvm::ArrayRef<QuantStorageAbiProfileId>,
      const QuantStorageAbiProfileRegistryV1 &);
};

llvm::Expected<UsedQuantStorageProfiles>
verifyUsedQuantStorageProfiles(
    llvm::ArrayRef<QuantStorageAbiProfileId> claimedProfileIds,
    const QuantStorageAbiProfileRegistryV1 &registry);

llvm::Expected<TargetBuildProfile> restrictTargetBuildProfileToUsedProfiles(
    const TargetBuildProfile &globalProfile,
    const UsedQuantStorageProfiles &usedProfiles,
    const ArtifactAdmissionLimits &limits);

class TargetArtifactFingerprintInput final {
public:
  const TargetEnvironmentFingerprint &targetEnvironmentFingerprint() const;
  const TargetBuildProfile &buildProfile() const;
  const UsedQuantStorageProfiles &usedQuantStorageProfiles() const;
  const ProjectionDependency &projectionDependency() const;
private:
  TargetArtifactFingerprintInput() = delete;
  friend llvm::Expected<TargetArtifactFingerprintInput>
  verifyTargetArtifactFingerprintInput(
      const TargetEnvironmentFingerprint &, const TargetBuildProfile &,
      const UsedQuantStorageProfiles &, const ProjectionDependency &);
};

llvm::Expected<TargetArtifactFingerprintInput>
verifyTargetArtifactFingerprintInput(
    const TargetEnvironmentFingerprint &environment,
    const TargetBuildProfile &buildProfile,
    const UsedQuantStorageProfiles &usedProfiles,
    const ProjectionDependency &projection);

class TargetArtifactFingerprint final {
public:
  DigestAlgorithm algorithm() const;
  llvm::ArrayRef<uint8_t> bytes() const;
private:
  TargetArtifactFingerprint() = delete;
  friend llvm::Expected<TargetArtifactFingerprint>
  buildTargetArtifactFingerprint(const TargetArtifactFingerprintInput &input,
      CanonicalEncodingContext &);
};

llvm::Expected<TargetArtifactFingerprint>
buildTargetArtifactFingerprint(
    const TargetArtifactFingerprintInput &input,
    CanonicalEncodingContext &encoding);
} // namespace wafer::abi
```

- [ ] **Step 1: Add one-field perturbation tests**

  Build a valid baseline and change, one at a time: environment digest, triple, extension set, MABI, KAD schema, WCRE/static encoder version, CRT/symbol-set digest, quant-profile registry schema version, every field of one actually used record-20 profile ID, each compiler/linker/objcopy/sysroot/CRT digest, canonical compile/link/objcopy arguments, allowed codegen environment, toolchain profile version, partition policy/dependency-classification version, entry/core/clone/dedup/prelink/note limit, clone linkage policy, pinned projection digest and relocatable slot schema. Every change must alter `wafer.target-artifact.v1`. Reordering canonical sets without changing them must not alter it. Unknown quant/profile/command/toolchain/linkage version, used profile not present in the environment-matched registry, profile/storage descriptor mismatch, caller-supplied flag/environment, zero limits and unknown partition policy version fail rather than selecting implementation defaults.

  Cover empty, single and multiple `usedQuantStorageProfiles` sets. INT8 and MXFP IDs in the same set are legal. Two different sets produce different fingerprints; source-order permutations do not. Adding, removing or reordering an unused profile in the global registry under the same schema version must not change an artifact fingerprint, while changing a used profile changes its record-20 ID and the fingerprint. Duplicate IDs, an unused ID injected by a caller, an omitted actual reference and a same-ID/non-equivalent-profile collision fail. Changing only a verified binary locator while all attested bytes remain equal does not change the fingerprint; changing one binary byte or one canonical argument does.

  Also assert a raw final ELF `ContentDigest` cannot satisfy an environment or artifact fingerprint parameter. `TargetArtifactFingerprint` must be non-aggregate, not publicly default/raw-digest/record constructible and expose no mutable record or digest accessor.

- [ ] **Step 2: Implement from the schema registry**

  Validate stable ASCII identifiers, typed enum/range facts, the registry schema, each complete record-20 profile against `TargetEnvironmentFingerprint`, the complete verified toolchain profile and all positive `ModulePartitionPolicyV1` core/clone/dedup limits plus fixed dependency/linkage policy. `UsedQuantStorageProfiles` is a canonical duplicate-free, possibly-empty set of typed IDs resolved against that registry. `restrictTargetBuildProfileToUsedProfiles` produces an artifact-scoped verified profile whose registry contains exactly the used records while preserving schema/toolchain/symbol/partition/linkage facts; it is the delivery/runtime form and cannot add an unused or omit a used record. Populate the generated target-artifact identity message with the registry schema version, actual used ID set and complete toolchain/partition/codegen/linkage facts, but not unused registry members; encode via `encodeIdentityMessage(..., encoding)`, and compute the fixed target-artifact domain. Pinned and relocatable forms are an explicit union; an empty union, both forms, unbounded allowed binding or projection digest/slot-schema mismatch fails. Partition/linkage policy, used quant/storage IDs and toolchain profile are code-generation inputs, not semantic module IDs; changing one changes the artifact fingerprint and may change final module content digests/set root.

  The runtime-safe factory validates a delivered final-module input semantically and uses an explicit canonical encoding context. The compiler path does not expose an API that accepts a raw used-ID vector. Task 8A scans each complete indivisible unit's verified instruction, launch-visible resource and pre-conversion profile refs, validates its canonical `UsedQuantStorageProfiles` evidence against the context registry, and retains it without constructing a fingerprint; Task 8B rederives the exact union from the converted instructions, resources and KADs before admitting the unit codegen/packing proof. Task 12 may pack units with different used sets when their environment/toolchain/projection/ABI compatibility key matches; when a group closes, it rescans/joins all unit/KAD/resource refs, computes the canonical duplicate-free union, calls `restrictTargetBuildProfileToUsedProfiles`, then constructs the group's only `TargetArtifactFingerprintInput` and fingerprint. Empty/plain, INT8 and MXFP units can therefore share one module with an exact mixed union. An incompatible module-wide profile relation fails its explicit verifier; profile-set difference alone is not a bucket or partition boundary.

- [ ] **Step 3: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add include/Wafer/ABI/TargetArtifacts.h \
    lib/Wafer/ABI/TargetArtifacts.cpp lib/Wafer/ABI/CMakeLists.txt \
    unittests/ABI/TargetArtifactFingerprintTest.cpp unittests/CMakeLists.txt
  git commit -m "Bind target artifact fingerprints to codegen inputs"
  ```

### Task 8A: Target Artifact Build Session and Whole-Coverage Preflight

**Files:**
- Create: `include/Wafer/Target/TargetArtifactBuildLimits.h`
- Create: `lib/Wafer/Target/TargetArtifactBuildLimits.cpp`
- Create: `include/Wafer/Target/TargetArtifactBuildSession.h`
- Create: `lib/Wafer/Target/TargetArtifactBuildSessionInternal.h`
- Create: `lib/Wafer/Target/TargetArtifactBuildSession.cpp`
- Create: `include/Wafer/Target/TargetCoverage.h`
- Create: `lib/Wafer/Target/TargetCoverage.cpp`
- Modify: `lib/Wafer/Target/CMakeLists.txt`
- Modify: `lib/Wafer/CMakeLists.txt`
- Create: `unittests/Target/TargetCoverageTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/Tools/wafer-target-coverage-preflight.test`

**Consumes:** the outer `ProgramOutputTransaction`, its staged executable token, one target variant and validated build limits; Task 4's canonical closure/clone proof; and the correctness plan's sealed committed entry/resource/pre-conversion boundary foundation. `ExecutableCompilationInput` must initially move-own one canonical complete `VerifiedTargetCompilationContextRegistry`; the selected winner transfers that same registry owner through `CommittedExecutableProgram` into the transaction's private executable lower commitment. Attachment never accepts a replacement vector/registry. The staged token binds that commitment and registry generation. `compiler::detail::TargetProgramOutputAccess::beginTargetArtifactBuild` is the only session factory and resolves exactly one context by `TargetVariantId` from that retained registry. The session binds stable canonical owner/context generation, exact context, staging capability, outer budgets and cancellation used by later preparation and final attachment; no token-only/digest-only context recovery or separate context/root/environment/profile/toolchain/entry/shape/module subset overload exists.

**Produces:** one move-only `TargetArtifactBuildSession` and one non-forgeable `VerifiedTargetCoverage` proving all-and-only committed shape variants, entries, closure units, clone dependencies, pre-conversion entry ABI contracts and service-resource feasibility for the target. It creates no converted module, KAD, object, packed module, delivery bytes or filesystem-visible root.

`lib/Wafer/Target/CMakeLists.txt` already exists from the correctness plan. Task 8A extends its compiler-only `WaferTargetArtifacts` sources and links only the ABI/compiler-identity/correctness boundary dependencies needed for session and pure coverage verification. It includes the correctness-owned request/boundary API but does not redeclare or invoke conversion. `WaferInstrToTargetLLVM` never includes or links `WaferTargetArtifacts`; package/runtime never link this target library or the Python orchestration helper.

```cpp
namespace wafer::target {
namespace detail {
class TargetArtifactBuildSessionAccess;
} // namespace detail

class TargetBuildReadLease final {
public:
  TargetBuildReadLease(TargetBuildReadLease &&) noexcept;
  TargetBuildReadLease(const TargetBuildReadLease &) = delete;
  ~TargetBuildReadLease();
private:
  TargetBuildReadLease() = delete;
  class Storage;
  explicit TargetBuildReadLease(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_; // reader + FD + buffer reservation
  friend class detail::TargetArtifactBuildSessionAccess;
};

enum class StagedObjectKind : uint8_t {
  CoreObject,
  CloneObject,
  AbiNoteSection,
  FinalElf,
  TargetSetRecord,
  TargetSetDelivery,
};

class OpenedStagedObjectLease final {
public:
  OpenedStagedObjectLease(OpenedStagedObjectLease &&) = default;
  llvm::Expected<size_t> readAt(
      uint64_t offset, llvm::MutableArrayRef<uint8_t> destination);
  llvm::Error finishAndVerify();
private:
  OpenedStagedObjectLease() = delete;
  class Storage;
  explicit OpenedStagedObjectLease(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_; // reservation + exact opened handle
  friend class detail::TargetArtifactBuildSessionAccess;
};

class StagedObjectRef final {
public:
  StagedObjectKind kind() const;
  uint64_t exactSize() const;
  const abi::ContentDigest &contentDigest() const;
private:
  StagedObjectRef() = delete;
  /* owner + staging generation + same-opened-object capability */
  friend class detail::TargetArtifactBuildSessionAccess;
};

class TargetArtifactBuildSession final {
public:
  TargetArtifactBuildSession(TargetArtifactBuildSession &&) = default;
  TargetArtifactBuildSession(const TargetArtifactBuildSession &) = delete;
private:
  TargetArtifactBuildSession() = delete;
  /* stable canonical owner token, context, target context, budgets,
     cancellation and VerifiedTargetStagingArea */
  friend class detail::TargetArtifactBuildSessionAccess;
};
} // namespace wafer::target

namespace wafer::compiler::detail {
class TargetProgramOutputAccess final {
public:
  static mlir::FailureOr<target::TargetArtifactBuildSession>
  beginTargetArtifactBuild(
      ProgramOutputTransaction &output,
      const StagedExecutableToken &executable,
      const abi::TargetVariantId &targetVariantId,
      const target::TargetArtifactBuildLimits &buildLimits);
private:
  TargetProgramOutputAccess() = delete;
  /* ProgramOutputTransaction friend only */
};
} // namespace wafer::compiler::detail

namespace wafer::target::detail {
struct EntryArtifactRelation {
  abi::EntryId entryId;
  llvm::SmallVector<abi::ExecutableVariantId> owningExecutableVariants;
  compiler_identity::StaticFunctionIdentity staticFunction;
  llvm::SmallVector<TargetEntrySlot> orderedSlots;
  llvm::SmallVector<abi::RankClassId> coveredRankClasses;
  llvm::SmallVector<abi::CompletionExportId> completionExports;
};

struct EntryFunctionKey {
  abi::EntryId entryId;
  abi::StaticFunctionDigest staticFunctionDigest;
};

class EntryAbiContractKey final; // pre-KAD typed slot/projection/completion contract

class ModuleCompatibilityKey final {
public:
  const abi::TargetEnvironmentFingerprint &environmentFingerprint() const;
  const abi::ProjectionDependency &projectionDependency() const;
private:
  ModuleCompatibilityKey() = delete;
  /* exact target ABI/toolchain/codegen facts excluding used-profile set and
     not-yet-formed final-module fingerprint */
};

struct EntryExportContract {
  EntryFunctionKey function;
  EntryAbiContractKey abiContract;
};

class ClosureUnitKey final {
public:
  llvm::ArrayRef<EntryExportContract> sortedEntryContracts() const;
  const compiler_identity::ClosureStructuralDigest &closureStructuralDigest() const;
  llvm::ArrayRef<compiler_identity::PrivateNodeStructuralDigest>
  sortedCloneDependencyDigests() const;
  const abi::UsedQuantStorageProfiles &actualUsedProfiles() const;
  const ModuleCompatibilityKey &moduleCompatibilityKey() const;
private:
  ClosureUnitKey() = delete;
  /* verifyTargetCoverage factory only */
};

class VerifiedClosureUnitRequest final {
public:
  const ClosureUnitKey &key() const;
  llvm::ArrayRef<EntryArtifactRelation> entryRelations() const;
  const abi::TargetEnvironmentFingerprint &environmentFingerprint() const;
  const abi::UsedQuantStorageProfiles &actualUsedProfiles() const;
  const ModuleCompatibilityKey &moduleCompatibilityKey() const;
private:
  VerifiedClosureUnitRequest() = delete;
  /* source op handles/generation token are private to friend preparation */
};

class VerifiedTargetCoverage final {
public:
  VerifiedTargetCoverage(const VerifiedTargetCoverage &) = delete;
  VerifiedTargetCoverage(VerifiedTargetCoverage &&) noexcept;
  const abi::ExecutableSemanticDigest &sourceExecutableDigest() const;
  const abi::TargetVariantId &targetVariantId() const;
  llvm::ArrayRef<abi::ExecutableVariantId> committedShapeVariantIds() const;
  llvm::ArrayRef<VerifiedClosureUnitRequest> closureUnits() const;
  const compiler::VerifiedTargetCompilationContext &targetContext() const;
  const abi::TargetBuildProfile &buildProfile() const;
  const TargetArtifactBuildLimits &buildLimits() const;
private:
  VerifiedTargetCoverage() = delete;
  /* source op handles/generation token are private to friend preparation */
};

mlir::FailureOr<VerifiedTargetCoverage> verifyTargetCoverage(
    TargetArtifactBuildSession &session);
} // namespace wafer::target::detail
```

`TargetArtifactBuildLimits` is a validated nonidentity service policy with checked positive bounds for private graph nodes/edges, cloneable dependencies/objects, closure units, entries, modules, KAD descriptors/notes, single/total core/clone/prelink bytes, single/total final-module bytes, staging disk, in-memory buffer count/bytes, active file descriptors and concurrent compiler/linker/objcopy workers. Its sandbox envelope also limits per-process/total wall-clock time, CPU time, RSS/address space, process count, stdout/stderr bytes and cancellation/kill/reap deadlines. Its constructor enforces repository hard safety caps and cross-field feasibility; zero never means unbounded. It is never serialized, hashed or used as a partition/caching key, and different sufficiently large limits cannot change keys, partition, arguments or bytes.

- [ ] **Step 1: Add session, coverage and bounded-preflight failure tests**

  Assert build-limits/session/staged-object/read-lease/coverage/closure/context proof types are not aggregates, publicly constructible or mutable and that no accessor exposes `Operation *`, `ExecutableOp`, `ExecutableVariantOp`, raw canonical context, staging path, argv/environment or mutable containers. `StagedObjectRef` exposes metadata only, has no public acquire/open/read operation and owns no persistent FD. No overload accepts an independent encoding context, staging root, environment/profile/toolchain, entry/variant subset, caller-supplied closure key, object path or extra flag. Move one registry owner along `ExecutableCompilationInput -> winner -> CommittedExecutableProgram -> lower commitment`, destroy each prior owner and prove target build still succeeds. Any attach overload accepting a new vector/registry is a compile-time failure. Starting with another transaction/token/target/context registry, swapping two staged tokens whose target IDs happen to match, reusing a moved session, pairing a staged ref/read lease from another session or generation, double-consuming a lease, or using a session after owner abort/commit/attachment-table mutation must fail stable owner/session/context-registry generation checks.

  Cover candidate entries, cross-target entries, missing owning shape variants, wrong target-context requirements, environment/profile/registry/toolchain cross-pairs, unresolved static/private calls or globals, invalid clone classification/proofs, duplicate or missing resource slots, completion-export mismatch, incompatible pinned projections and stale context. For every structural/count/service-limit category, test `limit-1/limit/limit+1`; graph/count breaches must fail before proportional reserve, staging creation or worker spawn, and invalid/overflowing configurations must fail validation. Exercise concurrent/repeated private staged-object acquisitions at exact reader/FD/buffer limits and prove all-or-none reservation, same-session enforcement, release on destruction and no FD retained by the ref. Instrument the process launcher and staging factory and require zero calls on every preflight failure. Snapshot the generic executable/module and empty staging area before each failure and require no IR mutation, converted module, KAD, object ref or delivery artifact afterward.

- [ ] **Step 2: Resolve complete target coverage and closure units from typed owners**

  `beginTargetArtifactBuild` derives the exact staged executable, `TargetVariantId`, retained `VerifiedTargetCompilationContext`, canonical encoding session, staging authority, outer cancellation and validated limits only from the outer transaction's executable lower commitment. It checks the token's commitment/registry generation, asks the retained registry for a total unique target-ID lookup and reruns the context's cross-pair verifier; missing, duplicate or stale contexts fail without a session. `verifyTargetCoverage` revalidates the target requirement, resolves all committed shape variants and recomputes executable identity. Before any staging or subprocess work, it builds a checked pure preflight over complete coverage: variants, entries, private graph nodes/edges, clone records, closure units, prospective KAD/note/object/module counts, sound byte/memory/FD/process/output bounds and the maximum known invocation envelope. It reserves the later service budget atomically; if any structural or operational bound cannot be proved, it fails without a partial reservation. Sufficiently larger policy limits cannot change semantic keys, partitioning or bytes.

  Call Task 4's sole resolver with the session-owned external registry/context and require classifications, non-clonable units, clone records/digests, labels, per-entry identities and total mapping to match the committed proof. Coverage does not walk a second graph, reclassify purity, recover registry content from a digest, invoke conversion or construct KADs. For each canonical static closure unit, derive one `EntryAbiContractKey` from every committed entry's ordered typed slots, semantic/storage/profile relations, projection dependency and completion exports. Partition target instantiations so one `(EntryId, StaticFunctionDigest)` export key has exactly one pre-conversion ABI contract inside an indivisible unit; the same entry/static pair with distinct shape contracts remains distinct, while distinct static digests for one entry may coexist.

  Generate the future export symbol mechanically from full lowercase-hex `EntryId` plus full lowercase-hex `StaticFunctionDigest` under one fixed ASCII prefix; never truncate, hash source text or reuse `sym_name`. Scan each unit's verified instructions, launch-visible resources and pre-conversion profile references, resolve record-20 IDs against the context registry, and retain the canonical `UsedQuantStorageProfiles` union as evidence. Build `ModuleCompatibilityKey` from exact environment, target ABI, toolchain/codegen, partition policy and projection facts, explicitly excluding the used-profile set and not-yet-formed final-module fingerprint. Form each `ClosureUnitKey` from sorted entry/export-contract keys, non-clonable closure digest, sorted clone dependency digests, actual used profiles and compatibility key. Reject source names/paths, caller IDs/profile lists and enumeration order as grouping facts. Return `VerifiedTargetCoverage` only when every committed target entry occurs exactly once and every closure, clone and owner relation is all-and-only.

- [ ] **Step 3: Run and commit the independent preflight boundary**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests wafer-opt -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv \
    build/wafer-dev/test/Tools/wafer-target-coverage-preflight.test
  git add include/Wafer/Target/TargetArtifactBuildLimits.h \
    include/Wafer/Target/TargetArtifactBuildSession.h \
    include/Wafer/Target/TargetCoverage.h \
    lib/Wafer/Target/TargetArtifactBuildLimits.cpp \
    lib/Wafer/Target/TargetArtifactBuildSessionInternal.h \
    lib/Wafer/Target/TargetArtifactBuildSession.cpp \
    lib/Wafer/Target/TargetCoverage.cpp \
    lib/Wafer/Target/CMakeLists.txt lib/Wafer/CMakeLists.txt \
    test/Tools/wafer-target-coverage-preflight.test \
    unittests/Target/TargetCoverageTest.cpp unittests/CMakeLists.txt
  git commit -m "Add target artifact coverage preflight"
  ```

### Task 9: Typed Low-Precision Command ABI and CRT Activation

**Files:**
- Modify: `include/Wafer/Compiler/Tx81CommandBuilder.h`
- Modify: `lib/Wafer/Compiler/Tx81CommandBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Modify: `include/Wafer/Target/TargetLegality.h`
- Modify: `lib/Wafer/Target/TargetLegality.cpp`
- Modify: `lib/Wafer/Target/CMakeLists.txt`
- Modify: `include/Wafer/Conversion/WaferInstrToTargetLLVM/WaferInstrToTargetLLVM.h`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/LowPrecisionLowering.cpp`
- Modify: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetCallBuilder.cpp`
- Modify: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/CMakeLists.txt`
- Modify: `runtime/wafer_crt/src/wafer_tx81_crt.c`
- Modify: `tools/check_target_crt_symbols.py`
- Modify: `tools/check_target_crt_conformance.py`
- Modify: `tools/check_target_dependency_layers.py`
- Create: `test/Transforms/lower-low-precision-to-target-llvm.mlir`
- Create: `test/Transforms/lower-low-precision-to-target-llvm-failure.mlir`
- Modify: `test/Tools/wafer-target-crt-symbols.test`
- Modify: `test/Tools/wafer-device-link.test`
- Modify: `test/Tools/check-target-dependency-layers.test`
- Modify: `unittests/Compiler/Tx81CommandBuilderTest.cpp`
- Modify: `unittests/Target/TargetLegalityTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Consumes:** whole-variant Task 4A's already existing, verified and committed `wafer.instr.quantized_gemm`/`wafer.instr.mxfp_decode`, their `QuantizationDescriptor`/`StorageEncodingDescriptor` and typed `QuantStorageAbiProfileId` refs, unique selected resource realizations, shared geometry and accepted completion/resource relations.

**Produces:** activation of the already-reviewed low-precision arms in the shared compiler resolver/target-legality/request pipeline, target LLVM calls and real CRT definitions for `wafer_tx81_quantized_gemm`/`wafer_tx81_mxfp_decode`. It changes no command layout, kind, prototype or runtime-safe value boundary, does not create/rewrite/choose instruction IR, and leaves plain GEMM quant-disabled.

Dependency direction is a testable contract: `WaferTargetLegality -> WaferIR/WaferCompilerIdentity/WaferABI`, `WaferInstrToTargetLLVM -> WaferTargetLegality/WaferCompilerIdentity/WaferABI`, `WaferCompilerKernelAbiAdapter -> WaferInstrToTargetLLVM/WaferCompilerIdentity/WaferABI`, and `WaferTargetArtifacts -> WaferInstrToTargetLLVM/WaferCompilerKernelAbiAdapter`. Base `WaferCompilerIdentity` has no conversion edge. Neither conversion nor CRT includes a `Wafer/Target/LowPrecisionAbi.h`, and no such artifact-layer header exists.

- [ ] **Step 1: Add failure-first typed legality tests**

  Positive INT8 starts from a committed instruction selected upstream against a matched native affine capability with exact data/scale/zero-point/storage descriptors. Positive FP8 starts from an upstream `explicit_composite` decode instruction into BF16/FP16 with explicit tile-local scratch, copy/decode completion and a dominated consumer. Negative cases mutate the committed proof to cover missing/ambiguous capability, unknown profile ID/descriptor/command version, profile not owned by the target build registry/environment, free-form dtype/packing string, implicit FP8 native execution, same capacity with different storage dtype/layout/packing, wrong scale/zp/scratch role, q/zp range, accumulator overflow, block/tail/count/span mismatch, unsupported NaN/Inf/subnormal/rounding/saturation policy, unaligned/out-of-range address and missing decode completion/scratch-reuse edge.

  Assert this task has no producer API for either instruction op and no dependency from target lowering back to tile/layout candidate selection. Missing upstream verified instruction/profile evidence is a structured preflight failure, never synthesized in the post-commit artifact stage. Extend Task 6's `check_target_dependency_layers.py` to inspect the actual CMake target graph plus installed-header include graph and reject `WaferInstrToTargetLLVM -> WaferCompilerKernelAbiAdapter/WaferTargetArtifacts`, base `WaferCompilerIdentity -> WaferInstrToTargetLLVM`, conversion/CRT inclusion of removed `Target/LowPrecisionAbi.h` or `Target/DTEAbi.h`, a second command struct/prototype definition, or absence of the required downward edges. The focused CMake build must also link all layers, so the script is a guard rather than the only proof.

  Assert neither op can lower through plain `wafer_tx81_gemm`, convert helpers, legacy `__FP8*`, varargs or an opaque descriptor pointer. A target without the exact capability returns structured unsupported before any target call is emitted.
  Update `check_target_crt_symbols.py` in the same change so it locates the production closure through a stable semantic heading/
  machine-readable boundary rather than the current `Q2-Q3` sentence. Remove task IDs from its regex, failure text and test golden;
  the checker may name the CRT closure and owning design file, but roadmap/queue numbering is not a tool protocol.

- [ ] **Step 2: Recheck the already-fixed shared C records**

  Consume the records already fixed in Task 3A's C-compatible `Wafer/ABI/Tx81CommandAbi.h`; `wafer_tx81_crt.h`, CRT C sources and compiler conversion all include that shared header instead of mirroring layouts. Task 9 reasserts but does not edit the following V1 facts. `WaferTx81QuantizedGemmCommandV1` is 160 bytes/alignment 8:

  ```text
  offset 0/2/4: u16 version=1 / u16 size=160 / u32 flags=0
  offset 8/16/24: u64 lhs_addr / rhs_addr / dst_addr
  offset 32/40: u64 scale_positive_addr / scale_negative_addr
  offset 48/52/56: u32 m / k / n
  offset 60/64: u32 left_batch / right_batch
  offset 68/72: u32 input_format / output_format
  offset 76: u32 transpose_bits (only bits 0/1)
  offset 80/84: u32 q0 / q1 (0..31)
  offset 88/92: u32 zero_point_left / zero_point_right (0..255)
  offset 96: u32 scale_mode (0 none, 1 positive, 2 negative, 3 both)
  offset 100/104/108: u32 rounding_mode / saturation_mode / reserved0=0
  offset 112/120/128: u64 lhs_span / rhs_span / dst_span
  offset 136/144: u64 scale_positive_span / scale_negative_span
  offset 152/156: u32 profile_version / reserved1=0
  ```

  `WaferTx81MxfpDecodeCommandV1` is 128 bytes/alignment 8:

  ```text
  offset 0/2/4: u16 version=1 / u16 size=128 / u32 flags=0
  offset 8/16/24/32: u64 packed_addr / scale_addr / scratch_addr / dst_addr
  offset 40/48/56/64: u64 packed_span / scale_span / scratch_span / dst_span
  offset 72/76/80: u32 element_count / block_count / block_size (V1: 32)
  offset 84/88/92: u32 source_encoding / scale_encoding / dst_format
  offset 96/100/104/108: u32 tail / nan / inf / subnormal_policy
  offset 112/116: u32 overflow_policy / rounding_mode
  offset 120/124: u32 profile_version / reserved0=0
  ```

  Add C `_Static_assert` and C++ `static_assert` for both sizes, alignments and every offset. Fixed prototypes are exactly:

  ```c
  int32_t wafer_tx81_quantized_gemm(uint64_t command_addr);
  int32_t wafer_tx81_mxfp_decode(uint64_t command_addr);
  ```

- [ ] **Step 3: Resolve commands only from typed IR/profile/realizations**

  `WaferABI::Tx81Command` validates layout-local closed enum/range/version/reserved relations and returns immutable private-construction command values; it never sees MLIR. `WaferCompilerIdentity::Tx81CommandBuilder` resolves a committed instruction's complete semantic quant descriptor, physical storage/packing descriptor, typed profile ID against the retained build registry, selected realizations and exact `compiler::VerifiedTargetCompilationContext`, then calls the runtime-safe validator. Native V1 permits only the reviewed INT8 affine subset. An existing FP8/block-scaled decode instruction always resolves as the explicit composite with packed/scale/scratch/destination operands already selected upstream; target lowering cannot invent the decode or pretend storage encoding is the compute type. `WaferTargetLegality` rechecks the resolver's op/root/operand ordinals, half-open allocation spans, alignment and shared geometry proof before the boundary factory seals the view. Shared geometry proves exact counts, spans, scale/zp coverage, accumulator bounds and target field widths without narrowing casts.

  Profile/descriptor values are copied as closed enums/integers, never recovered from op/resource names. The resolver returns a non-forgeable `compiler_identity::VerifiedTx81CommandView` or an anchored diagnostic and does not create a second quant/storage owner. `VerifiedTargetEntryBoundary` and then `VerifiedTargetConversionRequest` retain a complete typed op-occurrence-to-view join; no conversion API accepts a raw command aggregate.

- [ ] **Step 4: Lower through fixed prototypes and completion edges**

  Extend `TargetCallBuilder` with only the two shared-header `i32(i64 command_addr)` signatures and aligned stack/control records. `LowPrecisionLowering` joins each typed instruction occurrence to its request-sealed `VerifiedTx81CommandView`, emits exactly those fields and rejects missing/duplicate/wrong-kind/stale joins; it never calls an artifact resolver or reads a raw profile/command. Immediate nonzero status flows into the entry's typed error export and prevents consumers. For MXFP decode, return 0 means accepted execution, not local completion: explicit decode completion/local drain must dominate GEMM/other consumers and tile-local scratch reuse. KAD slots record only launch-visible packed source, scale/zero-point resource, stream staging or cross-entry scratch relations with their profile IDs. Entry-local decode destination, scale copy and scratch remain instruction/SPM facts and never become slots. LLVM parameter order remains the committed `TargetEntrySlot` order and KAD is built afterward.

- [ ] **Step 5: Implement audited CRT behavior and symbol closure**

  Quantized GEMM uses the reviewed `TsmGemm` wrapper sequence and enables `SetQuant`/scale wrappers only as authorized; V1 forbids bias, activation, sparse and implicit psum. MXFP decode uses repo-audited SPM load/store and public arithmetic wrappers, never a legacy helper ABI. Return codes are `0=accepted`, `-1=bad version/size/reserved`, `-2=bad field/range`, `-3=unsupported profile/encoding`; preserve them on the typed status/error surface.

  Add header/signature/source/object/device-link/conformance checks in the same patch. A symbol enters the required set only when its instruction occurs. Both command ABI versions, symbol registry and CRT bytes contribute to the typed profile/artifact fingerprint; no success stub or silent plain-GEMM fallback is allowed.

- [ ] **Step 6: Run focused gates**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests wafer-opt -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv build/wafer-dev/test \
    --filter='LowPrecision|lower-low-precision-to-target-llvm|wafer-target-crt-symbols|wafer-device-link'
  python3 tools/check_target_crt_symbols.py
  python3 tools/check_target_crt_conformance.py
  ```

  Expected: native INT8 and explicit FP8 decode paths execute; every descriptor/profile/storage mutation fails at its owner boundary; actual CRT objects/device link contain required symbols only for used operations.

- [ ] **Step 7: Commit**

  ```bash
  git add include/Wafer/Compiler/Tx81CommandBuilder.h \
    lib/Wafer/Compiler/Tx81CommandBuilder.cpp lib/Wafer/Compiler/CMakeLists.txt \
    include/Wafer/Target/TargetLegality.h lib/Wafer/Target/TargetLegality.cpp \
    lib/Wafer/Target/CMakeLists.txt \
    include/Wafer/Conversion/WaferInstrToTargetLLVM/WaferInstrToTargetLLVM.h \
    lib/Wafer/Conversion/WaferInstrToTargetLLVM/LowPrecisionLowering.cpp \
    lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetCallBuilder.cpp \
    lib/Wafer/Conversion/WaferInstrToTargetLLVM/CMakeLists.txt \
    runtime/wafer_crt tools/check_target_crt_symbols.py \
    tools/check_target_crt_conformance.py tools/check_target_dependency_layers.py \
    test/Transforms/lower-low-precision-to-target-llvm*.mlir \
    test/Tools/wafer-target-crt-symbols.test test/Tools/wafer-device-link.test \
    test/Tools/check-target-dependency-layers.test \
    unittests/Compiler/Tx81CommandBuilderTest.cpp \
    unittests/Target/TargetLegalityTest.cpp unittests/CMakeLists.txt
  git commit -m "Activate typed low-precision target commands"
  ```

### Task 10: Direct DTE ABI Activation from Accepted Transport

**Files:**
- Modify: `include/Wafer/Compiler/Tx81CommandBuilder.h`
- Modify: `lib/Wafer/Compiler/Tx81CommandBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Modify: `include/Wafer/Target/TargetLegality.h`
- Modify: `lib/Wafer/Target/TargetLegality.cpp`
- Modify: `lib/Wafer/Target/CMakeLists.txt`
- Modify: `include/Wafer/Conversion/WaferInstrToTargetLLVM/WaferInstrToTargetLLVM.h`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/DTELowering.cpp`
- Modify: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetCallBuilder.cpp`
- Modify: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/CMakeLists.txt`
- Modify: `runtime/wafer_crt/src/wafer_tx81_crt.c`
- Modify: `tools/check_target_crt_symbols.py`
- Modify: `tools/check_target_crt_conformance.py`
- Modify: `tools/check_target_dependency_layers.py`
- Create: `test/Transforms/lower-accepted-dte-to-target-llvm.mlir`
- Create: `test/Transforms/lower-accepted-dte-to-target-llvm-failure.mlir`
- Modify: `test/Tools/wafer-target-crt-symbols.test`
- Modify: `test/Tools/wafer-device-link.test`
- Modify: `test/Tools/check-target-dependency-layers.test`
- Modify: `unittests/Compiler/Tx81CommandBuilderTest.cpp`
- Modify: `unittests/Target/TargetLegalityTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Consumes:** whole-variant Tasks 9/10's already existing committed `InstrDTESendOp`/`InstrDTERecvOp`/`InstrDTEWaitOp`, entry-local `TransportActionId`, accepted `ExecutableTransportOp`, final pinned/relocatable projection, accepted addresses/ranges and the committed executable entry's typed `TargetEntrySlot` control/status/completion relations.

**Produces:** activation of the already-reviewed DTE arms in the shared compiler resolver/target-legality/request pipeline, request-sealed typed LLVM calls and real CRT implementations. It changes no command layout, kind, prototype or runtime-safe value boundary, creates no instruction/transport IR and never derives endpoint facts from logical peer.

- [ ] **Step 1: Add failure-first transport binding tests**

  Keep existing logical `buffer, peer, bytes` fixture structured-unsupported. Add failures for wrong version/record size, missing/stale action ID, no accepted transport, half-concrete member, projection mismatch, misaligned/missing context or status, zero timeout, missing src/dst address, length/packet/capacity mismatch, invalid direction/memory kind, missing local/remote FSM or stream, invalid block/channel/mode/tile, invalid stride/iteration, nonzero flags/reserved, receiver range/alignment error, absent status slot, missing wait, and status surface that cannot distinguish success/timeout/transport error/peer failure.

  Positive pinned and relocatable fixtures must share one logical p2p body but arrive as distinct already accepted committed projection forms. Assert there is no producer API in this task for DTE instruction or executable transport records; incomplete logical DTE is a structured preflight failure.

- [ ] **Step 2: Recheck the already-fixed V1 records and prototypes**

  Consume `WaferTx81DteCommandV1` from the sole shared C-compatible `Wafer/ABI/Tx81CommandAbi.h` exactly as fixed in Task 3A and `tasks/14`: size 160, alignment 8; `abi_version=1`, `record_bytes=160`; `action_id`; `src_addr/dst_addr`; `context_addr/status_addr`; positive `timeout_cycles`; `length/packet_bytes/packet_count`; direction `send=1, recv=2`; receiver memory `spm=1, ddr=2`; local/remote FSM and stream IDs; DTE block/channel; mode/tile endpoints; three explicit stride/iteration pairs; zero flags and reserved fields. Task 10 reruns C/C++ layout/golden checks but does not edit the shared header or value validator.

  ```text
  offset 0/4:    u32 abi_version / record_bytes
  offset 8:      u64 action_id
  offset 16/24:  u64 src_addr / dst_addr
  offset 32/40:  u64 context_addr / status_addr
  offset 48:     u64 timeout_cycles
  offset 56/60/64: u32 length / packet_bytes / packet_count
  offset 68/72:  u32 direction / receiver_memory_kind
  offset 76/80:  u32 local_fsm_id / remote_fsm_id
  offset 84/88:  u32 local_stream_id / remote_stream_id
  offset 92/96:  u32 dte_block_id / dte_channel
  offset 100/104/108: u32 mode / tile_this / dst_tile
  offset 112..135: u32 stride0, iteration0, stride1, iteration1,
                   stride2, iteration2
  offset 136/140: u32 flags / reserved0
  offset 144/152: u64 reserved1 / reserved2
  ```

  `wafer_tx81_crt.h` only includes the shared command header and adds no mirror layout or prototype. There is no `Wafer/Target/DTEAbi.h`.

  Add `WaferTx81DteContextV1 { uint64_t opaque[16]; }` with size 128/alignment 8 and a compile-time check that the private provider context fits. Add exact status layout:

  ```c
  typedef struct {
    uint32_t abi_version;
    uint32_t outcome;
    int32_t provider_code;
    uint32_t flags;
    uint64_t action_id;
    uint64_t observed_cycles;
  } WaferTx81DteStatusV1;
  ```

  Require size 32/alignment 8 and outcome values `success=0`, `timeout=1`, `transport_error=2`, `peer_failure=3`, `pending=UINT32_MAX`. Add exactly:

  ```c
  int32_t wafer_tx81_dte_recv(uint64_t command_addr);
  int32_t wafer_tx81_dte_send(uint64_t command_addr);
  int32_t wafer_tx81_dte_wait(uint64_t command_addr);
  ```

  Immediate returns are `0` accepted/status-updated, `-1` invalid command/version/range, `-2` provider issue/wait/release failure, `-3` unsupported target capability. Cross-check the status numeric values against generated KAD enums. These records group accepted facts; they are not a second transport owner or opaque payload.

- [ ] **Step 3: Resolve typed pinned and relocatable bindings**

  Extend the sole `WaferCompilerIdentity::Tx81CommandBuilder` resolver and `WaferTargetLegality` adapter; no resolver is added to the artifact layer:

  ```cpp
  mlir::FailureOr<compiler_identity::VerifiedTx81CommandView>
  resolveVerifiedTx81CommandView(
      mlir::Operation *committedInstruction,
      ExecutableEntryOp entry,
      const compiler::VerifiedTargetCompilationContext &context);
  ```

  It accepts only committed `InstrDTESendOp`, `InstrDTERecvOp` or `InstrDTEWaitOp`, revalidates owner/context generation and calls the runtime-safe per-field command validator. For pinned projection, materialize concrete endpoint/channel/FSM/receiver addresses and the complete reviewed call fields. For relocatable projection, map only the accepted field substitutions to the committed projection relocation schema and the executable entry's typed endpoint/control/status `TargetEntrySlot`s, preserving their finite allowed values. In both forms, `WaferTargetLegality` checks operation/root/operand ordinals, accepted allocation ranges, length and three stride/iteration pairs before `VerifiedTargetEntryBoundary` seals the op-to-command join. Resolver/conversion take no KAD input; after conversion, Task 8B invokes the Task 6 KAD builder to record and verify the exact same slot-to-LLVM-parameter relations.

  `dte_recv` initializes the exact local FSM/stream and receiver storage before peer send. `dte_send` uses only accepted block/channel, remote FSM/stream/address/endpoint and never equates remote destination with local source. Both set status to pending. `dte_wait` performs finite nonblocking polling using `timeout_cycles`, writes success/timeout/transport-error, releases provider state and is not a local NCC drain. Device CRT never invents peer failure; RuntimeSession joins another rank/stage failure into the same typed completion surface.

- [ ] **Step 4: Lower through fixed LLVM prototypes**

  Extend `TargetCallBuilder` with the three shared-header `i32(i64 command_addr)` signatures and construct each aligned command/context/status record without varargs. `DTELowering` must join every typed op occurrence to exactly one request-sealed DTE command view and rejects a missing/duplicate/wrong-kind/stale view; it does not resolve transport or include/link `WaferTargetArtifacts`. Pinned fields become checked constants/addresses. Relocatable fields load only from the verified request boundary's committed `TargetEntrySlot`s selected by the accepted projection relocation schema. Every send/recv token retains the same `TransportActionId`; a variadic wait emits one call per producer order. Immediate and terminal status flow to the entry's typed completion/error surface and cannot be discarded or replaced by `wafer_tx81_local_fence`. Lowering stores the exact entry-handle-to-function relation only inside the private converted-module owner; scoped KAD construction independently records and verifies the same slots, while conversion never reads a descriptor it has not yet produced.

- [ ] **Step 5: Implement CRT and only then promote symbols**

  Call the repo-vendored exact-block Direct DTE/FSM APIs evidenced by `direct_dte_and_fsm.h`, `kuiper_dte.h` and `kuiper_streamfsm.h`. Reject target revisions without exact block selection, nonblocking receive status or finite-cycle wait. Add header/source/signature/lowering/conformance checks in one patch. Remove the three symbols from the exclusion list only after pinned/relocatable positives, device-observable pending/success/timeout/transport-error, peer-failure value preservation in the typed status/KAD verifier, CRT object definition and actual device link all pass. RuntimeSession's actual cross-rank peer-failure join remains the package/runtime gate. Empty success stubs, unbounded waits and dynamic resource search are forbidden.

- [ ] **Step 6: Run focused gates**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='lower-accepted-dte-to-target-llvm|wafer-target-crt-symbols|wafer-device-link'
  python3 tools/check_target_crt_symbols.py
  python3 tools/check_target_crt_conformance.py
  ```

  Expected: incomplete logical DTE still fails; accepted pinned/relocatable cases emit fixed calls; CRT and device link define all three symbols; device outcomes and the runtime-owned peer-failure value remain distinguishable on one typed status surface.

- [ ] **Step 7: Commit**

  ```bash
  git add include/Wafer/Compiler/Tx81CommandBuilder.h \
    lib/Wafer/Compiler/Tx81CommandBuilder.cpp lib/Wafer/Compiler/CMakeLists.txt \
    include/Wafer/Target/TargetLegality.h lib/Wafer/Target/TargetLegality.cpp \
    lib/Wafer/Target/CMakeLists.txt \
    include/Wafer/Conversion/WaferInstrToTargetLLVM/WaferInstrToTargetLLVM.h \
    lib/Wafer/Conversion/WaferInstrToTargetLLVM/DTELowering.cpp \
    lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetCallBuilder.cpp \
    lib/Wafer/Conversion/WaferInstrToTargetLLVM/CMakeLists.txt \
    runtime/wafer_crt tools/check_target_crt_symbols.py \
    tools/check_target_crt_conformance.py tools/check_target_dependency_layers.py \
    test/Transforms/lower-accepted-dte-to-target-llvm*.mlir \
    test/Tools/wafer-target-crt-symbols.test test/Tools/wafer-device-link.test \
    test/Tools/check-target-dependency-layers.test \
    unittests/Compiler/Tx81CommandBuilderTest.cpp \
    unittests/Target/TargetLegalityTest.cpp unittests/CMakeLists.txt
  git commit -m "Bind accepted Direct DTE transport to target ABI"
  ```

### Task 8B: Atomic Core/Clone Conversion, KAD, and Prelink Preparation

**Files:**
- Create: `include/Wafer/Target/TargetModuleTransaction.h`
- Create: `lib/Wafer/Target/TargetModuleTransaction.cpp`
- Modify: `lib/Wafer/Target/CMakeLists.txt`
- Modify: `tools/wafer_device_link.py`
- Create: `unittests/Target/TargetModuleTransactionTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/Tools/wafer-prelink-closure-unit.test`

**Consumes:** one by-value move-only `VerifiedTargetCoverage` and its live `TargetArtifactBuildSession` from Task 8A; completed low-precision and DTE command-family activation from Tasks 9/10; the correctness-owned move-only conversion request/result API; and Task 6's sole post-conversion KAD adapter. It consumes all covered closure units and clone dependencies, never a caller-selected subset. The target plan includes the correctness headers and does not redeclare their request, boundary, converted-owner or factory types.

**Produces:** one move-only non-forgeable `PreparedTargetCoverage` containing every privately staged non-clonable core object and transaction-deduplicated clone dependency object for all committed shape variants referencing the target, while retaining the exact build-session/staging owner and generation needed by later reads. No packed module, published descriptor, raw MLIR handle, host path or filesystem root is visible.

Task 8B extends compiler-only `WaferTargetArtifacts` with downward `WaferInstrToTargetLLVM`, `WaferCompilerKernelAbiAdapter` and MLIR translation dependencies needed for verified conversion, KAD construction and object emission. `WaferInstrToTargetLLVM` never includes or links the KAD adapter or `WaferTargetArtifacts`, while base `WaferCompilerIdentity` never includes conversion, so no dependency cycle or artifact/session/staging concept enters conversion. The converted owner exposes only bounded by-value summaries with owner-bound handles. Only `compiler_identity::detail::KernelAbiBuildAccess` and `target::detail::ConvertedTargetModuleAccess` may borrow its private module, sealed boundaries and function mapping for the duration of their calls; neither adapter exposes or retains a `ModuleOp`, `LLVMFuncOp`, `Operation *`, boundary reference or mapping.

```cpp
namespace wafer::target::detail {
class PrelinkUnitKey final {
public:
  const ClosureUnitKey &closureKey() const;
  llvm::ArrayRef<abi::KernelAbiSemanticDigest> sortedKadDigests() const;
private:
  PrelinkUnitKey() = delete;
};

class UnitPackingKey final {
public:
  llvm::ArrayRef<EntryFunctionKey> sortedEntryFunctions() const;
  const abi::ContentDigest &coreObjectContentDigest() const;
  llvm::ArrayRef<abi::ContentDigest> sortedCloneObjectContentDigests() const;
  llvm::ArrayRef<abi::KernelAbiSemanticDigest> sortedKadDigests() const;
  const abi::UsedQuantStorageProfiles &actualUsedProfiles() const;
  const ModuleCompatibilityKey &moduleCompatibilityKey() const;
private:
  UnitPackingKey() = delete;
};

class PreparedTargetEntry final {
public:
  const abi::EntryId &entryId() const;
  llvm::ArrayRef<abi::ExecutableVariantId> owningExecutableVariants() const;
  const compiler_identity::StaticFunctionIdentity &staticFunction() const;
  const abi::VerifiedKernelAbiDescriptor &kernelAbi() const;
  llvm::StringRef entrySymbol() const;
  llvm::ArrayRef<abi::RankClassId> coveredRankClasses() const;
  llvm::ArrayRef<abi::CompletionExportId> completionExports() const;
private:
  PreparedTargetEntry() = delete;
  /* preparation internals are the only factory */
};

class PreparedCloneObject final {
public:
  const compiler_identity::PrivateNodeStructuralDigest &structuralDigest() const;
  const StagedObjectRef &stagedObject() const;
  const abi::ContentDigest &objectContentDigest() const;
  uint64_t exactCodeDataBytes() const;
private:
  PreparedCloneObject() = delete;
  /* one transaction-wide object per equal digest + equal structural record */
};

class PreparedClosureUnit final {
public:
  const PrelinkUnitKey &prelinkKey() const;
  const UnitPackingKey &packingKey() const;
  const StagedObjectRef &stagedCoreObject() const;
  const abi::ContentDigest &coreObjectContentDigest() const;
  uint64_t exactCoreCodeDataBytes() const;
  llvm::ArrayRef<compiler_identity::PrivateNodeStructuralDigest>
  sortedCloneDependencyDigests() const;
  llvm::ArrayRef<PreparedTargetEntry> entries() const;
  const abi::UsedQuantStorageProfiles &actualUsedProfiles() const;
  const ModuleCompatibilityKey &moduleCompatibilityKey() const;
private:
  PreparedClosureUnit() = delete;
  /* owns no published locator; staging lifetime belongs to set builder */
};

class PreparedTargetCoverage final {
public:
  PreparedTargetCoverage(const PreparedTargetCoverage &) = delete;
  PreparedTargetCoverage(PreparedTargetCoverage &&) noexcept;
  const abi::ExecutableSemanticDigest &sourceExecutableDigest() const;
  const abi::TargetVariantId &targetVariantId() const;
  llvm::ArrayRef<abi::ExecutableVariantId> committedShapeVariantIds() const;
  llvm::ArrayRef<PreparedClosureUnit> preparedClosureUnits() const;
  llvm::ArrayRef<PreparedCloneObject> preparedCloneObjects() const;
  const compiler::VerifiedTargetCompilationContext &targetContext() const;
  const abi::TargetBuildProfile &buildProfile() const;
  const TargetArtifactBuildLimits &buildLimits() const;
private:
  PreparedTargetCoverage() = delete;
  /* prepareTargetCoverage is the only factory; storage retains the exact
     session/staging owner and generation for every prepared join/ref */
};

mlir::FailureOr<PreparedTargetCoverage> prepareTargetCoverage(
    VerifiedTargetCoverage coverage,
    TargetArtifactBuildSession &session);
} // namespace wafer::target::detail
```

- [ ] **Step 1: Add atomic preparation and resource-envelope failures**

  Assert prepared-entry/clone/unit/coverage/key proof types are closed and immutable, and no target API exposes raw converted modules/functions/operations, an authoritative borrowed entry view, sealed boundary, handle-to-function map, mutable containers, staging paths, argv/environment, independent context components, caller keys, prelinked paths or extra flags. The converted owner may expose only bounded `entryCount` and by-value non-authoritative summaries. Compile-time checks require both conversion factories to accept only a by-value `mlir::OwningOpRef<mlir::ModuleOp>` and the entry-core factory to move-consume `llvm::SmallVector<VerifiedTargetEntryBoundary>`; no borrowed `ModuleOp`, `ArrayRef` boundary or raw-module overload may exist. Failure must destroy the owned isolated clone and every move-only boundary.
  The boundary/dependency private storage must also carry the committed executable's MLIR-context lifetime share; request creation
  rechecks it against each isolated module and the converted owner retains it. Destroying the caller/coverage frame before KAD/
  object emission remains safe, while a cross-context share fails before conversion mutation. No public context/share accessor or
  independent context argument is added.

  Cover conversion failure, unsupported structure, geometry overflow, missing or wrong command view, KAD/function mismatch, completion mismatch, stale coverage/session/context, wrong owner generation, closure or clone replay mismatch, mixed incompatible projection/module keys and mutation of the source executable after Task 8A verification. Substitute one context-bound compiler byte, canonical argument, registry entry or allowed environment value after session creation and require rejection before compilation. A two-entry unit with failure in its second KAD must return neither the first KAD nor an object.

  For every preparation byte/buffer/FD/worker/time/process/output limit, test `limit-1/limit/limit+1`. Add attested fake-tool modes that hang, spin CPU, fork, allocate beyond RSS/address-space, emit unbounded output, ignore graceful termination and stall during reap; add cancellation during compile. Each breach stops the sandboxed process group, closes bounded sinks, kills/reaps by deadline, revokes every partial core/clone ref and returns no proof. Reap failure marks the outer transaction permanently uncommittable and isolates the orphan.

- [ ] **Step 2: Move isolated IR through verified conversion, KAD, and object emission**

  Materialize each unit's non-clonable core as `mlir::OwningOpRef<mlir::ModuleOp> isolatedInput`. Obtain correctness-owned move-only `VerifiedTargetEntryBoundary` values from the sealed committed entry/resource/pre-KAD contract, assign only opaque request-local handles, then call exactly:

  ```cpp
  llvm::SmallVector<conversion::VerifiedTargetEntryBoundary> boundaries;
  /* Populated only by the correctness-owned private boundary access from the
     sealed all-and-only coverage-unit relations. */
  auto request = conversion::verifyEntryCoreConversionRequest(
      std::move(isolatedInput), std::move(boundaries), closureUnit,
      sessionTargetContext);
  ```

  The request exclusively owns the isolated clone and boundaries. For each independently materialized helper/global clone, call `verifyCloneDependencyConversionRequest(std::move(cloneInput), dependency, sessionTargetContext)`; it requires zero entry boundaries and replays Task 4's purity/address/structural proof plus fixed `hidden_linkonce_odr_comdat_v1`. Only `convertVerifiedWaferInstrModuleToTargetLLVM(std::move(*request))` consumes a closed request. Cloneable definitions are excluded from the core and referenced only through verified digest-derived prelink symbols. No source spelling, vector position or KAD recovers request ownership.

  Conversion returns a private move-only `ConvertedTargetLLVMModule` owner. Iterate only its bounded `entryCount()`, obtain each by-value `ConvertedTargetEntrySummary`, and pass the live owner plus that summary's owner-bound `ConvertedTargetEntryHandle` to Task 6's adapter. A summary is diagnostic/index data, never verifier authority. `KernelAbiBuildAccess` first rejects cross-module, stale or forged handles, then borrows the exact sealed boundary and mapped function only during its call, rechecks converted parameter types and returns only `VerifiedKernelAbiDescriptor`; no stale pre-rewrite `VerifiedInstructionGeometry` is carried. Require each KAD's canonical ABI fields to equal its pre-conversion `EntryAbiContractKey`, and require the rederived instruction/resource/KAD profile-ID union to equal the retained unit evidence. Do not retain handles in `PreparedTargetEntry`. If any KAD fails, destroy the complete converted owner before object emission.

  Create `PreparedTargetEntry` values in canonical `EntryId + static-function digest + KAD digest` order and form `PrelinkUnitKey = (ClosureUnitKey, sorted unique KAD digests)`. Through one scoped `ConvertedTargetModuleAccess` call, translate/compile the owned module directly into the session's sandboxed staging sink; the callback may neither return nor retain the module, functions or mapping. Tool output is accepted only by `VerifiedTargetStagingArea`, measured/hashed/stat-checked on the same opened handle and returned as the exact `CoreObject`/`CloneObject` `StagedObjectRef`; no path is exposed or reopened, and the ref retains no FD.

  Lower each unique cloneable dependency once per transaction. Apply `hidden_linkonce_odr_comdat_v1`: full digest-derived symbol/COMDAT signature, equal signature requires equal structural record and object bytes, and one final module selects one definition. For each readback, private `TargetArtifactBuildSessionAccess` reserves a same-session `TargetBuildReadLease`, move-consumes it with the exact `StagedObjectRef`, and returns one opened lease owning reader/FD/buffer capacity plus the handle. Parse, stat and digest on that handle, release it promptly, then form `UnitPackingKey = (sorted entry keys, core digest, sorted clone digests, sorted unique KAD digests, actual used profiles, ModuleCompatibilityKey)`. Cross-session/generation or repeated-lease pairing fails before open. Destroy the converted owner after scoped emission; prepared proofs retain only KADs, typed keys, measurements and FD-free staged capabilities.

- [ ] **Step 3: Prepare all coverage atomically**

  `prepareTargetCoverage` move-consumes coverage, revalidates session owner/generation, source generation/digest and every typed relation before cloning, then visits closure units and unique clone dependencies in canonical order. Track cumulative bytes, CPU/wall/output, buffers, FDs and workers against Task 8A's reservation before every action. Every subprocess joins the session process group; any failure revokes all earlier refs. Return proof only when every core/clone ref belongs to the same session/staging generation, every clone join is one-to-one and every committed target entry is prepared exactly once. This stage does not pack units, form a final profile union/fingerprint or create a module group.

- [ ] **Step 4: Add deterministic relation and object checks**

  Verify every prepared entry had exactly one owner-bound handle and private conversion mapping during scoped construction; each KAD slot bijects one converted LLVM parameter; owning variants match rank-class owners; and completion/projection/geometry relations close. Cross-owner handles, stale handles after owner destruction/move, forged ordinals, summary substitution and attempts to retain a handle/boundary/function past preparation must fail. Permute source and worker order, rename private symbols and vary sufficiently large service limits; require identical digests, labels, keys, sizes and bytes. A fresh equal transaction must produce byte-identical core/clone digests while its refs/tokens remain unusable across sessions. Test one `EntryId` with two static digests/full symbols and one entry/static pair with different ABI contracts in separate units. Mixed profile IDs prepare successfully; unused registry profiles alter neither keys nor bytes.

  Add thousands of entries sharing a proved-pure helper/immutable global: preparation creates bounded non-clonable cores plus exactly one transaction clone object per equal structural record. Adding an effect, address escape or mutable global joins only affected entries and an oversized indivisible core fails. Change one helper/global semantic bit and require static/clone/unit keys to change. Use LLVM Object/`llvm-readelf` checks to prove fixed hidden linkonce-ODR COMDAT references/definitions, equal-definition coalescing, unequal same-signature rejection and absence of source/private names. Structural/label/COMDAT collisions fail before final link.

- [ ] **Step 5: Run and commit the independent preparation boundary**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests wafer-opt -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv \
    build/wafer-dev/test/Tools/wafer-prelink-closure-unit.test
  git add include/Wafer/Target/TargetModuleTransaction.h \
    lib/Wafer/Target/TargetModuleTransaction.cpp \
    lib/Wafer/Target/CMakeLists.txt tools/wafer_device_link.py \
    test/Tools/wafer-prelink-closure-unit.test \
    unittests/Target/TargetModuleTransactionTest.cpp unittests/CMakeLists.txt
  git commit -m "Prepare target LLVM and kernel ABI atomically"
  ```

### Task 11: Mandatory ELF ABI Note Encode, Injection, and Readback

**Files:**
- Create: `include/Wafer/Artifact/ArtifactMetadataVerificationSession.h`
- Create: `include/Wafer/Artifact/RuntimeArtifactVerificationSession.h`
- Create: `include/Wafer/Artifact/ElfAbiNote.h`
- Create: `lib/Wafer/Artifact/ArtifactMetadataVerificationSessionInternal.h`
- Create: `lib/Wafer/Artifact/ArtifactMetadataVerificationSession.cpp`
- Create: `lib/Wafer/Artifact/RuntimeArtifactVerificationSessionInternal.h`
- Create: `lib/Wafer/Artifact/RuntimeArtifactVerificationSession.cpp`
- Create: `lib/Wafer/Artifact/ElfAbiNote.cpp`
- Create: `lib/Wafer/Artifact/CMakeLists.txt`
- Modify: `lib/Wafer/CMakeLists.txt`
- Modify: `lib/Wafer/Target/CMakeLists.txt`
- Modify: `tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp`
- Modify: `tools/wafer-artifact-inspect/CMakeLists.txt`
- Modify: `tools/wafer_device_link.py`
- Create: `unittests/Artifact/ElfAbiNoteTest.cpp`
- Create: `unittests/Artifact/ArtifactMetadataVerificationSessionTest.cpp`
- Create: `unittests/Artifact/RuntimeArtifactVerificationSessionTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/Tools/wafer-elf-abi-note.test`

**Consumes:** Task 3A's validated admission/canonical foundations; one lower `verification::ArtifactMetadataHostBudgetCapability` moved once into a process/deployment `ArtifactMetadataVerificationRegistry`; lower runtime host quotas issued from `WaferVerificationSupport::HostVerificationRegistry`; a runtime `ArtifactVerificationBudgetCapability` issued from a runtime host quota only after exact service-domain context binds the service owner; and verified KADs belonging to prepared/final target modules.

**Produces:** runtime-safe `WaferArtifact`; semantically independent move-only `ArtifactMetadataVerificationSession` and `RuntimeArtifactVerificationSession` domains whose typed quotas jointly charge the same lower host ledger; exact `.note.wafer.abi` section bytes, verified readback KADs and final-ELF eligibility for content digest. Sessions/capabilities are nonconvertible and have no upgrade API. This library links `WaferVerificationSupport`, `WaferABI` and LLVM Object/Support only and never includes/links `WaferRuntime`.

```cpp
namespace wafer::abi {
class VerifiedTargetArtifactModuleEvidence;
class VerifiedTargetArtifactSetRecordMember;
} // namespace wafer::abi

namespace wafer::verification {
class ArtifactMetadataHostBudgetCapability;
class RuntimeArtifactHostBudgetCapability;
} // namespace wafer::verification

namespace wafer::artifact {
namespace detail {
struct ArtifactMetadataVerificationRegistryStorage;
struct ArtifactMetadataVerificationBudgetStorage;
struct ArtifactMetadataVerificationSessionStorage;
struct ArtifactVerificationBudgetStorage;
struct RuntimeArtifactVerificationSessionStorage;
class ArtifactMetadataVerificationRegistryFactory;
class ArtifactMetadataVerificationRegistryAccess; // bounded session-capability issuer
class ArtifactMetadataVerificationSessionAccess; // metadata loaders only
class ArtifactVerificationBudgetFactory;       // noninstalled runtime/test issuer
class RuntimeArtifactVerificationSessionAccess; // loaders/runtime friends only
} // namespace detail

class ArtifactMetadataVerificationSession;
class ArtifactMetadataVerificationBudgetCapability;

class ArtifactMetadataVerificationRegistry final {
public:
  ArtifactMetadataVerificationRegistry(
      ArtifactMetadataVerificationRegistry &&) noexcept;
  ArtifactMetadataVerificationRegistry(
      const ArtifactMetadataVerificationRegistry &) = delete;
  ~ArtifactMetadataVerificationRegistry();
private:
  ArtifactMetadataVerificationRegistry() = delete;
  explicit ArtifactMetadataVerificationRegistry(std::shared_ptr<
      detail::ArtifactMetadataVerificationRegistryStorage> storage);
  std::shared_ptr<detail::ArtifactMetadataVerificationRegistryStorage> storage_;
  friend llvm::Expected<ArtifactMetadataVerificationRegistry>
  createArtifactMetadataVerificationRegistry(
      verification::ArtifactMetadataHostBudgetCapability);
  friend class detail::ArtifactMetadataVerificationRegistryFactory;
  friend class detail::ArtifactMetadataVerificationRegistryAccess;
};

llvm::Expected<ArtifactMetadataVerificationRegistry>
createArtifactMetadataVerificationRegistry(
    verification::ArtifactMetadataHostBudgetCapability hostBudget);

class ArtifactMetadataVerificationBudgetCapability final {
public:
  ArtifactMetadataVerificationBudgetCapability(
      ArtifactMetadataVerificationBudgetCapability &&) noexcept;
  ArtifactMetadataVerificationBudgetCapability(
      const ArtifactMetadataVerificationBudgetCapability &) = delete;
  ~ArtifactMetadataVerificationBudgetCapability();
private:
  ArtifactMetadataVerificationBudgetCapability() = delete;
  explicit ArtifactMetadataVerificationBudgetCapability(std::unique_ptr<
      detail::ArtifactMetadataVerificationBudgetStorage> storage);
  std::unique_ptr<detail::ArtifactMetadataVerificationBudgetStorage> storage_;
  friend class detail::ArtifactMetadataVerificationRegistryAccess;
};

namespace detail {
class ArtifactMetadataVerificationRegistryAccess final {
public:
  static llvm::Expected<ArtifactMetadataVerificationBudgetCapability>
  issueSessionBudget(ArtifactMetadataVerificationRegistry &registry);
private:
  ArtifactMetadataVerificationRegistryAccess() = delete;
};
} // namespace detail

class ArtifactMetadataVerificationSession final {
public:
  ArtifactMetadataVerificationSession(
      ArtifactMetadataVerificationSession &&) noexcept;
  ArtifactMetadataVerificationSession(
      const ArtifactMetadataVerificationSession &) = delete;
  ~ArtifactMetadataVerificationSession();
private:
  ArtifactMetadataVerificationSession() = delete;
  explicit ArtifactMetadataVerificationSession(
      std::unique_ptr<detail::ArtifactMetadataVerificationSessionStorage>);
  std::unique_ptr<detail::ArtifactMetadataVerificationSessionStorage> storage_;
  friend llvm::Expected<ArtifactMetadataVerificationSession>
  createArtifactMetadataVerificationSession(
      ArtifactMetadataVerificationBudgetCapability,
      abi::ArtifactAdmissionLimits,
      abi::CanonicalEncodingContext);
  friend class detail::ArtifactMetadataVerificationSessionAccess;
};

llvm::Expected<ArtifactMetadataVerificationSession>
createArtifactMetadataVerificationSession(
    ArtifactMetadataVerificationBudgetCapability budget,
    abi::ArtifactAdmissionLimits limits,
    abi::CanonicalEncodingContext encoding);

class ArtifactVerificationBudgetCapability final {
public:
  ArtifactVerificationBudgetCapability(
      ArtifactVerificationBudgetCapability &&) noexcept;
  ArtifactVerificationBudgetCapability(
      const ArtifactVerificationBudgetCapability &) = delete;
  ~ArtifactVerificationBudgetCapability();
private:
  ArtifactVerificationBudgetCapability() = delete;
  explicit ArtifactVerificationBudgetCapability(
      std::unique_ptr<detail::ArtifactVerificationBudgetStorage> storage);
  std::unique_ptr<detail::ArtifactVerificationBudgetStorage> storage_;
  friend class detail::ArtifactVerificationBudgetFactory;
};

class ArtifactVerificationReadLease final {
public:
  ArtifactVerificationReadLease(ArtifactVerificationReadLease &&) noexcept;
  ArtifactVerificationReadLease(const ArtifactVerificationReadLease &) = delete;
  ~ArtifactVerificationReadLease();
private:
  ArtifactVerificationReadLease() = delete;
  class Storage;
  explicit ArtifactVerificationReadLease(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_; // same-session reader + FD reservation
  friend class detail::RuntimeArtifactVerificationSessionAccess;
};

class ArtifactVerificationWorkLease final {
public:
  ArtifactVerificationWorkLease(ArtifactVerificationWorkLease &&) noexcept;
  ArtifactVerificationWorkLease(const ArtifactVerificationWorkLease &) = delete;
  ~ArtifactVerificationWorkLease();
private:
  ArtifactVerificationWorkLease() = delete;
  class Storage;
  explicit ArtifactVerificationWorkLease(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_; // worker + simultaneously verified bytes
  friend class detail::RuntimeArtifactVerificationSessionAccess;
};

class RuntimeArtifactVerificationSession final {
public:
  RuntimeArtifactVerificationSession(
      RuntimeArtifactVerificationSession &&) noexcept;
  RuntimeArtifactVerificationSession(
      const RuntimeArtifactVerificationSession &) = delete;
  ~RuntimeArtifactVerificationSession();
private:
  RuntimeArtifactVerificationSession() = delete;
  explicit RuntimeArtifactVerificationSession(
      std::unique_ptr<detail::RuntimeArtifactVerificationSessionStorage> storage);
  std::unique_ptr<detail::RuntimeArtifactVerificationSessionStorage> storage_;
  friend llvm::Expected<RuntimeArtifactVerificationSession>
  createRuntimeArtifactVerificationSession(
      ArtifactVerificationBudgetCapability,
      abi::ArtifactAdmissionLimits,
      abi::CanonicalEncodingContext);
  friend class detail::RuntimeArtifactVerificationSessionAccess;
};

llvm::Expected<RuntimeArtifactVerificationSession>
createRuntimeArtifactVerificationSession(
    ArtifactVerificationBudgetCapability budget,
    abi::ArtifactAdmissionLimits limits,
    abi::CanonicalEncodingContext encoding);

class SerializedWaferAbiNoteSection final {
public:
  uint64_t exactSize() const;
  const abi::ContentDigest &contentDigest() const;
  const abi::ImmutableByteBackingRef &backing() const;
  llvm::Error writeTo(abi::ByteSink &sink) const;
private:
  SerializedWaferAbiNoteSection() = delete;
  /* bounded immutable backing, not a semantic identity */
};

llvm::Expected<SerializedWaferAbiNoteSection> encodeWaferAbiNoteSection(
    llvm::ArrayRef<abi::VerifiedKernelAbiDescriptor> descriptors,
    llvm::endianness elfEndianness,
    const abi::ArtifactAdmissionLimits &limits);

llvm::Expected<llvm::SmallVector<abi::VerifiedKernelAbiDescriptor>>
readAndVerifyWaferAbiNotes(
    const abi::ImmutableByteBackingRef &elf,
    const abi::ArtifactAdmissionLimits &limits,
    abi::CanonicalEncodingContext &encoding);
} // namespace wafer::artifact
```

- [ ] **Step 1: Implement independent metadata and runtime verification sessions**

  Lower `WaferVerificationSupport::HostVerificationRegistry` is the sole process/deployment physical ledger for host readers, open FDs, workers, verified bytes and workspace. Its private domain issuer creates nonforgeable `ProgramDeliveryHostBudgetCapability`, `ArtifactMetadataHostBudgetCapability` and `RuntimeArtifactHostBudgetCapability`; these are nonconvertible child quotas over the same parent. `createArtifactMetadataVerificationRegistry` move-consumes one metadata host capability exactly once and creates the shared typed metadata-child ledger; a second registry cannot reuse that capability. Noninstalled `ArtifactMetadataVerificationRegistryAccess` asks the live registry to issue one move-only `ArtifactMetadataVerificationBudgetCapability`; issuance reserves no I/O and every capability retains the same child-ledger owner/generation. `createArtifactMetadataVerificationSession` move-consumes that capability, validated `ArtifactAdmissionLimits` and canonical context, reserves a child-session envelope without resetting the registry, and creates one metadata-session owner token. There is no direct registry-to-session overload. Only these sessions may open/preflight target/package/migration metadata; they have no device/provider/module authority and cannot create a runtime session.

  `RuntimeArtifactVerificationSessionInternal.h` defines the runtime `ArtifactVerificationBudgetFactory` and private access. A raw `RuntimeArtifactHostBudgetCapability` cannot construct a runtime session: the factory must first join it with one exact verified service-domain owner/generation and bounded service child quota to issue `ArtifactVerificationBudgetCapability`. `createRuntimeArtifactVerificationSession` move-consumes that capability, limits and canonical context; it neither accepts nor upgrades a metadata session. Metadata/runtime semantic owners, typed child quotas and local counters remain disjoint and nonconvertible, while every reservation charges the common host parent. Binder construction reserves nothing; only actual module leases issued after `RuntimeSession::create` additionally charge service child and live invocation capacity. Runtime binding must revalidate a metadata proof's exact backing/record owner; equality of loose limits/digests never substitutes for that proof.

  Within each semantic domain, only its private access can request an all-or-none multi-ledger reservation. Metadata read/work leases jointly retain Artifact-metadata child plus host-parent reader/FD/worker/verified-metadata-byte/workspace capacity through delivery-root/schema parsing, then close before a `Loaded*Metadata` value is returned. A runtime session by itself cannot mint an open-capable lease: only the later RuntimeSession/provider adapter may present its nonforgeable live invocation-capacity proof to `RuntimeArtifactVerificationSessionAccess`. That access atomically creates an `ArtifactVerificationReadLease` charging host, runtime child, service and invocation readers/FDs before open; a bound source move-consumes it and returns an opened lease owning reservation plus handle. `ArtifactVerificationWorkLease` analogously covers all joined worker/module-byte/workspace ledgers. The sealed backing retains both until ELF/provider use dies. Failure to reserve any ledger rolls all back. There is no public source acquire/read, context, capability, owner-token, scalar-reserve, default session, metadata-to-runtime conversion or host-ledger bypass. Compiler staging uses separate build leases that still charge outer program/build host policy.

  Unit tests move one metadata host capability into one small registry, issue a distinct move-only metadata budget capability for each session and prove their typed-child aggregate never resets; capability reuse, direct registry-to-session creation, foreign-registry capability/session pairing and registry finalization races fail. Create Runtime-artifact host quotas over the same deliberately small host parent and bind them to two service contexts. Interleave metadata/runtime sessions and prove host contention, semantic-child isolation, rollback, stale host/service generation rejection and RAII release. Prove binder creation reserves no invocation capacity and runtime session alone cannot mint a read/work lease. Test-only RuntimeSession access then supplies distinct invocation proofs: stale/cross-invocation use fails, and every actual lease atomically charges/rolls back host+child+service+invocation. A raw host runtime subcap, metadata registry/session or equal loose limits cannot create a runtime session. Compile/link checks prove `WaferArtifact` has no `WaferRuntime` include/link and external providers cannot construct storage or access the host registry directly.

- [ ] **Step 2: Add exact note byte tests**

  Assert exact little- and big-endian ELF note headers, `namesz=6`, name bytes `WAFER\0`, type `0x57414249`, 4-byte padding and target-independent descriptor:

  ```text
  ASCII("WABI")
  || u16be(1)
  || u16be(1)
  || u32be(1)
  || u32be(kernel_abi_schema_version)
  || u32be(descriptor_delivery_size)
  || 32-byte descriptor semantic digest
  || descriptor_delivery_bytes
  ```

  Emit exactly one note per unique descriptor semantic digest, sorted by its 32 raw bytes. Two entries may reference the same verified KAD; the encoder coalesces those references only when their canonical record and delivery bytes are byte-identical. The same digest paired with non-equivalent canonical/delivery bytes is a hard collision. Empty input, wrong embedded digest, duplicate note on readback, unsorted readback, truncated header/descriptor and nonzero malformed padding fail. No independent ABI ID is encoded or used for sorting.

- [ ] **Step 3: Implement encoder and same-verifier readback**

  The encoder takes only `VerifiedKernelAbiDescriptor`, so unverified delivery bytes cannot be embedded. It checked-counts references/unique notes/aggregate delivery bytes against explicit limits, indexes by typed digest, stream-compares canonical/delivery backings on repeated references and writes the canonical unique set into one bounded immutable `SerializedWaferAbiNoteSection`; no production API returns an unbounded vector. The reader accepts only the same owner-backed ELF used by the enclosing verifier, checks ELF and note-section sizes before traversal, and lets WaferArtifact's private backing adapter create an owner-retained read-only mapping from that stable handle, never reopen a path. Mapping/address-space reservations count against `maxSingleVerifiedModuleBytes` and `maxSimultaneouslyVerifiedModuleBytes`; a source that cannot provide a same-owner immutable mapping within those limits fails explicitly rather than using an implicit temp/default scratch path. LLVM Object APIs then perform the single structural traversal with checked offset arithmetic. The reader counts note/KAD/string/aggregate categories before allocation, creates admission-checked owner-backed slices for every descriptor payload, calls `parseAndVerifyKernelAbiDescriptor(payloadBacking, limits, encoding)`, and compares the embedded descriptor semantic digest. It rejects extra Wafer note types, repeated digest notes and multiple `.note.wafer.abi` sections. Add a positive two-entry fixture with one shared KAD and prove the section contains one note while both typed entry relations resolve to it; add over-limit note count/descriptor bytes, cross-backing slice substitution and integer-overflow mutations that fail before KAD allocation.

- [ ] **Step 4: Inject with all mandatory llvm-objcopy controls**

  Extend the internal `wafer_device_link.py` orchestration with a distinct LLVM objcopy locator supplied only from Task 5's `VerifiedTargetToolchainInvocation`; keep the separately attested TX8 objcopy only for the existing RISC-V attribute normalization. The script performs no PATH/default/environment discovery, accepts no extra flags, and emits no proof by itself. The C++ caller reattests the binary immediately before launch and derives the exact LLVM 20 form from the closed profile:

  ```text
  --add-section .note.wafer.abi=<file>
  --set-section-type .note.wafer.abi=7
  --set-section-flags .note.wafer.abi=alloc,readonly,contents
  --set-section-alignment .note.wafer.abi=4
  ```

  It then executes `wafer-artifact-inspect verify-elf-note --elf=<final.so> --expected-kad=<delivery.pb>...`. Objcopy exit 0 alone is not success.

- [ ] **Step 5: Verify section contract and tamper failures**

  Through LLVM `ELFObjectFile`, require section name, `SHT_NOTE`, `SHF_ALLOC`, no `SHF_WRITE`/`SHF_EXECINSTR`, alignment 4, exactly one note per expected unique KAD digest and no unexpected descriptor. Add lit mutations for PROGBITS, missing alloc, writable, alignment 1, wrong note type, duplicate digest note, changed delivery byte and changed semantic digest.

- [ ] **Step 6: Prove digest timing**

  Compute `ContentDigest` of the ELF before note injection and after injection and require different values. Expose only the post-readback value from the target artifact API; the pre-note value remains a test local.

- [ ] **Step 7: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests wafer-artifact-inspect -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv build/wafer-dev/test/Tools/wafer-elf-abi-note.test
  git add include/Wafer/Artifact/ArtifactMetadataVerificationSession.h \
    include/Wafer/Artifact/RuntimeArtifactVerificationSession.h \
    include/Wafer/Artifact/ElfAbiNote.h \
    lib/Wafer/Artifact/ArtifactMetadataVerificationSessionInternal.h \
    lib/Wafer/Artifact/ArtifactMetadataVerificationSession.cpp \
    lib/Wafer/Artifact/RuntimeArtifactVerificationSessionInternal.h \
    lib/Wafer/Artifact/RuntimeArtifactVerificationSession.cpp \
    lib/Wafer/Artifact/ElfAbiNote.cpp lib/Wafer/Artifact/CMakeLists.txt \
    lib/Wafer/CMakeLists.txt lib/Wafer/Target/CMakeLists.txt \
    tools/wafer-artifact-inspect tools/wafer_device_link.py \
    unittests/Artifact/ElfAbiNoteTest.cpp \
    unittests/Artifact/ArtifactMetadataVerificationSessionTest.cpp \
    unittests/Artifact/RuntimeArtifactVerificationSessionTest.cpp \
    unittests/CMakeLists.txt test/Tools/wafer-elf-abi-note.test
  git commit -m "Embed and verify mandatory Wafer ELF ABI notes"
  ```

### Task 12: Verified Device Link and Final Target Module

**Files:**
- Create: `include/Wafer/Artifact/TargetElfVerifier.h`
- Create: `lib/Wafer/Artifact/TargetElfVerifier.cpp`
- Modify: `lib/Wafer/Artifact/CMakeLists.txt`
- Create: `include/Wafer/Target/DeviceLink.h`
- Create: `lib/Wafer/Target/DeviceLink.cpp`
- Modify: `lib/Wafer/Target/CMakeLists.txt`
- Modify: `tools/wafer_device_link.py`
- Modify: `tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp`
- Modify: `test/Tools/wafer-device-link.test`
- Create: `test/Tools/wafer-device-link-verified-module.test`
- Create: `unittests/Target/DeviceLinkTest.cpp`
- Create: `unittests/Artifact/TargetElfVerifierTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Consumes:** one move-only whole-target `PreparedTargetCoverage` from Task 8B retaining the exact build session/staging owner and `VerifiedTargetCompilationContext`, plus Task 11's exact note encoder and explicit admission limits. Canonical packing move-consumes coverage before any group proof can escape; device link then move-consumes that packing owner. There is no independent context component, borrowed-coverage packing proof or single-group/subset link entry point.

**Produces:** non-forgeable `VerifiedTargetModules` covering the complete target, with each `VerifiedTargetModule` carrying a session-bound final-ELF `StagedObjectRef`, exact build profile, target environment/artifact fingerprints, post-note `ContentDigest` and typed per-entry relations; it neither exposes a locator nor retains every module byte in RAM.

```cpp
namespace wafer::abi {
class VerifiedTargetArtifactModuleView; // completed in Task 13
}

namespace wafer::artifact {
namespace detail {
struct TargetElfEntryContractStorage;
struct TargetElfContractStorage;
struct VerifiedElfMetadataStorage;
class TargetElfProofFactory;
class TargetElfVerifier;
} // namespace detail

class VerifiedTargetElfEntryContract final {
public:
  VerifiedTargetElfEntryContract(const VerifiedTargetElfEntryContract &) = default;
  VerifiedTargetElfEntryContract(VerifiedTargetElfEntryContract &&) = default;
  llvm::StringRef entrySymbol() const;
  const abi::VerifiedKernelAbiDescriptor &kernelAbi() const;
private:
  VerifiedTargetElfEntryContract() = delete;
  explicit VerifiedTargetElfEntryContract(
      std::shared_ptr<const detail::TargetElfEntryContractStorage> storage);
  std::shared_ptr<const detail::TargetElfEntryContractStorage> storage_;
  friend class detail::TargetElfProofFactory;
};

class VerifiedTargetElfContract final {
public:
  VerifiedTargetElfContract(const VerifiedTargetElfContract &) = default;
  VerifiedTargetElfContract(VerifiedTargetElfContract &&) = default;
  const abi::TargetBuildProfile &buildProfile() const;
  const abi::TargetEnvironmentFingerprint &environmentFingerprint() const;
  const abi::UsedQuantStorageProfiles &usedQuantStorageProfiles() const;
  const abi::TargetArtifactFingerprint &artifactFingerprint() const;
  llvm::ArrayRef<VerifiedTargetElfEntryContract> entries() const;
private:
  VerifiedTargetElfContract() = delete;
  explicit VerifiedTargetElfContract(
      std::shared_ptr<const detail::TargetElfContractStorage> storage);
  std::shared_ptr<const detail::TargetElfContractStorage> storage_;
  friend class detail::TargetElfProofFactory;
};

class VerifiedElfMetadata final {
public:
  VerifiedElfMetadata(VerifiedElfMetadata &&) = default;
  VerifiedElfMetadata(const VerifiedElfMetadata &) = delete;
  const abi::ImmutableByteBackingRef &verifiedBacking() const;
private:
  VerifiedElfMetadata() = delete;
  explicit VerifiedElfMetadata(
      std::shared_ptr<const detail::VerifiedElfMetadataStorage> storage);
  std::shared_ptr<const detail::VerifiedElfMetadataStorage> storage_;
  friend class detail::TargetElfVerifier;
};
class BoundTargetModuleSource;   // completed in Task 13

llvm::Expected<VerifiedTargetElfContract>
buildLoadedTargetElfContract(
    const abi::VerifiedTargetArtifactModuleView &module);

llvm::Expected<VerifiedElfMetadata> verifyTargetElf(
    const abi::ImmutableByteBackingRef &backing,
    const VerifiedTargetElfContract &contract,
    const abi::ArtifactAdmissionLimits &limits,
    abi::CanonicalEncodingContext &encoding);
} // namespace wafer::artifact

namespace wafer::target {
class ModuleGroupKey;
namespace detail {
class PreparedClosureUnit;
class PreparedCloneObject;
} // namespace detail

mlir::FailureOr<artifact::VerifiedTargetElfContract>
buildCompilerTargetElfContract(
    const ModuleGroupKey &group,
    TargetArtifactBuildSession &session);

mlir::FailureOr<artifact::VerifiedElfMetadata>
verifyAndMeasureStagedTargetElf(
    const StagedObjectRef &stagedElf,
    const artifact::VerifiedTargetElfContract &contract,
    const abi::ArtifactAdmissionLimits &limits,
    TargetArtifactBuildSession &session);

class ModuleGroupKey final {
public:
  llvm::ArrayRef<detail::UnitPackingKey> sortedUnitKeys() const;
  const abi::UsedQuantStorageProfiles &usedQuantStorageProfiles() const;
  const abi::TargetBuildProfile &restrictedBuildProfile() const;
  const abi::TargetArtifactFingerprint &targetArtifactFingerprint() const;
private:
  ModuleGroupKey() = delete;
  /* privately retains the all-and-only prepared unit/core/clone/entry joins
     selected by CanonicalTargetPacking */
}; // transaction-local proof, never serialized or hashed

namespace detail {
class CanonicalTargetPacking final {
public:
  CanonicalTargetPacking(const CanonicalTargetPacking &) = delete;
  CanonicalTargetPacking(CanonicalTargetPacking &&) noexcept;
  llvm::ArrayRef<ModuleGroupKey> moduleGroups() const;
private:
  CanonicalTargetPacking() = delete;
  /* owns the move-consumed PreparedTargetCoverage storage and exact
     session/staging generation; group keys are views into this owner */
};

mlir::FailureOr<CanonicalTargetPacking> verifyCanonicalTargetPacking(
    PreparedTargetCoverage preparedCoverage,
    TargetArtifactBuildSession &session);
} // namespace detail

class VerifiedTargetModuleEntry final {
public:
  const abi::EntryId &entryId() const;
  llvm::ArrayRef<abi::ExecutableVariantId> owningExecutableVariants() const;
  const compiler_identity::StaticFunctionIdentity &staticFunction() const;
  const abi::VerifiedKernelAbiDescriptor &kernelAbi() const;
  llvm::StringRef entrySymbol() const;
  llvm::ArrayRef<abi::RankClassId> coveredRankClasses() const;
  llvm::ArrayRef<abi::CompletionExportId> completionExports() const;
private:
  VerifiedTargetModuleEntry() = delete;
  /* DeviceLink.cpp constructs only after final ELF verification */
};

class VerifiedTargetModule final {
public:
  const StagedObjectRef &stagedElf() const;
  uint64_t finalElfSize() const;
  const abi::ContentDigest &moduleContentDigest() const;
  const abi::TargetBuildProfile &buildProfile() const;
  const abi::TargetEnvironmentFingerprint &targetEnvironmentFingerprint() const;
  const abi::UsedQuantStorageProfiles &usedQuantStorageProfiles() const;
  const abi::TargetArtifactFingerprint &targetArtifactFingerprint() const;
  const ModuleGroupKey &moduleGroupKey() const;
  llvm::ArrayRef<VerifiedTargetModuleEntry> entries() const;
private:
  VerifiedTargetModule() = delete;
};

class VerifiedTargetModules final {
public:
  const abi::ExecutableSemanticDigest &sourceExecutableDigest() const;
  const abi::TargetVariantId &targetVariantId() const;
  llvm::ArrayRef<abi::ExecutableVariantId> committedShapeVariantIds() const;
  const detail::CanonicalTargetPacking &packing() const;
  llvm::ArrayRef<VerifiedTargetModule> modules() const;
private:
  VerifiedTargetModules() = delete;
};

mlir::FailureOr<VerifiedTargetModules> packLinkAndVerifyTargetCoverage(
    detail::CanonicalTargetPacking packing,
    const abi::ArtifactAdmissionLimits &limits,
    TargetArtifactBuildSession &session);
} // namespace wafer::target
```

`detail::TargetElfProofFactory` and `detail::TargetElfVerifier` are implementation-only friends in `TargetElfVerifier.cpp`; their public reachability is exactly the two contract factories and `verifyTargetElf` above. Entry/contract copies share const storage containing owned strings, KAD/backing owners and canonical entry indexes. `VerifiedElfMetadata` is move-only because its storage retains the exact immutable backing, source-owner token, parsed ELF/note/export evidence and contract storage. It has no constructor/friend that can accept a different backing after verification.

- [ ] **Step 1: Add actual-link positive and negative tests**

  Positive starts with whole-target coverage containing two shape variants and at least three compiler-generated core objects plus shared clone objects from Task 8B. Include two entries with the same verified KAD digest and prove they share one ABI note; include the same `EntryId` with two static digests and prove both full typed export symbols coexist. Include plain, INT8 and MXFP units with different per-unit used sets but the same module compatibility key and require the first packed module's final union/fingerprint to contain exactly all three used facts. Exact entry/core+deduplicated-clone/note limits force a second module. The implementation compiles the repo CRT object, invokes the attested TX8 linker with each module's core objects plus each clone digest exactly once in canonical key order, injects each aggregate unique-KAD note set and verifies both final `.so` files. The test must not pass `--print-commands` and must inspect existing ELF files.

  Negative cases: caller-selected unit subset/group, premature split, limit violation, same KAD digest paired with non-equivalent canonical/delivery bytes, one module containing the same `(EntryId, StaticFunctionDigest)` with different KADs, missing/reordered/substituted core or clone object, missing/duplicate clone definition, same COMDAT signature with unequal structural/object bytes, wrong `ModuleGroupKey`, source/truncated/colliding export symbol, exported/non-hidden helper, cross-module non-clonable private undefined reference, unresolved clone reference, wrong `e_machine`, triple/ISA/MABI/attributes mismatch, missing/extra export, undefined `wafer_tx81_*`, non-Wafer undefined outside Task 3's reviewed subset, KAD/function or KAD/entry mismatch, note mismatch, omitted/injected profile in final union, restricted-profile/fingerprint mismatch and content digest computed from a different byte stream. Reject caller-assembled ELF contract fields, raw byte/path/digest/size verification and cross-session staged refs. Permute source enumeration, worker completion, ELF export, clone object and note iteration independently; canonical packing and typed joins must produce the same result, never a parallel-vector association. Assert `VerifiedTargetElfContract`, verified coverage/modules/module/entry/ELF metadata and source refs are non-aggregate, not publicly constructible and expose const accessors only.

- [ ] **Step 2: Pack the complete prepared coverage canonically**

  Implement packing once in `detail::verifyCanonicalTargetPacking`. Rehash every prepared core and transaction-unique clone object only through `TargetArtifactBuildSessionAccess`: reserve a `TargetBuildReadLease`, bind it to the same-session ref, stream the opened handle, then release it before advancing. Require each `UnitPackingKey` core digest/clone digest set/unique KADs/actual profiles/entry keys/compatibility key to match. Bucket only by equal `ModuleCompatibilityKey`; profile-set difference is not a boundary. Within each bucket, lexicographically sort `UnitPackingKey`s and apply the fixed sequential rule. Maintain candidate maps for `(EntryId, StaticFunctionDigest) -> KAD digest + full canonical export symbol` and `PrivateNodeStructuralDigest -> clone ContentDigest + structural record + COMDAT signature`. If appending introduces the same entry/function with another KAD/symbol, close before that unit; a conflict inside one unit fails. Same clone digest+record/object is counted/linked once; same digest/signature with unequal record/object is a collision. Different static digests for one EntryId remain legal because their full typed symbols differ.

  For each candidate append, checked-sum exact core bytes plus the candidate's deduplicated clone-object bytes, take the canonical union of all unit actual-profile sets, and stream `encodeWaferAbiNoteSection` over complete entry KAD references to obtain exact unique-note bytes including headers/padding. Rejoin all KAD/resource/instruction refs to prove the candidate profile union exact and run explicit module-wide profile compatibility; do not infer compatibility from string names. Append only when entry, core+clone, unique-note and clone/dedup policy limits remain positive/in range; otherwise close before that unit. An oversized non-clonable core or single clone object fails, while repeated equal KAD/clone records cost one note/object. When a group closes, call `restrictTargetBuildProfileToUsedProfiles(globalProfile, exactUnion, admissionLimits)`, build the only `TargetArtifactFingerprintInput` and compute its fingerprint through the explicit canonical encoding context. Construct `ModuleGroupKey` from exact sorted `UnitPackingKey`s plus that final fingerprint. Prove groups cover every prepared core/unit/entry exactly once, reference every required clone and produce unique future member keys. Count the same canonical groups against operational totals and fail rather than repartition to fit. `verifyCanonicalTargetPacking` move-consumes coverage and returns the sole owner of its storage/group views; `packLinkAndVerifyTargetCoverage` then move-consumes that packing. No second packer or borrowed proof exists.

  Tests cover `limit-1/limit/limit+1` for every semantic module-policy/core/clone/dedup limit and every operational module-count/final-byte/staging/process/worker/FD/reader/buffer/memory/CPU/wall/output budget, policy-version change, input/worker permutation and two identical runs. Concurrent/repeated readback of core/clone refs must remain within session/outer FD-reader-buffer limits, reject cross-session leases and leave no FD in long-lived proofs. After packing, destroy/reallocate the caller's former coverage container and prove group joins remain valid only through the packing owner; moving/destroying packing invalidates all internal views and a stale/different session fails before read. A concurrency or sufficiently high operational limit changes only scheduling/acceptance; every accepted run has the same ordered group list, profile unions/fingerprints, exact byte totals, keys and final bytes. Add hundreds/thousands of compatible plain/INT8/MXFP units with varied used sets and require the policy to pack by exact limits rather than profile buckets. A total-budget `limit+1` run may complete earlier private work but returns no module proof/root, and a graph/count preflight breach spawns no subprocess. No API exposes a group factory or accepts a caller group.

- [ ] **Step 3: Link exact packed objects in private staging**

  Invoke the Python helper only as session-owned command orchestration for `canonical core refs + unique clone refs -> CRT -> ELF -> note injection`; internal tool locators are temporary mappings issued from session-private opened staged leases, never caller inputs, and output is admitted by `VerifiedTargetStagingArea` on the same opened handle. It cannot compile caller LLVM, discover objects, accept flags or reorder inputs. Reattest binaries/sysroot/CRT, reserve and bind fresh `TargetBuildReadLease`s, and reverify each core/clone ref immediately before use; opened leases retain FD/reader/buffer capacity through the child invocation and close before the next bounded batch. For `hidden_linkonce_odr_comdat_v1`, link one definition per clone signature, reject unequal duplicates, then localize/retain hidden helpers so public exports contain only typed entries. Cross-module non-clonable undefined refs fail. Enforce session process/CPU/wall/RSS/output/FD/buffer/final-byte budgets. Cancellation/breach kills/reaps and revokes partial refs; exact-size overrun returns `build_limit_exceeded`. Fixed toolchain behavior yields identical bytes. Any group failure returns no `VerifiedTargetModules`.

- [ ] **Step 4: Verify every final ELF structurally**

  Keep ELF parsing/semantic checks in runtime-safe `WaferArtifact`, but expose no public raw buffer/path/expected-field verifier. Compiler calls `buildCompilerTargetElfContract` only with a `ModuleGroupKey` whose private packing proof retains the all-and-only prepared unit/core/clone/entry joins plus the same session; the factory rechecks every retained key/object/context relation before returning a contract, so a caller cannot pass a unit/clone/member subset. `verifyAndMeasureStagedTargetElf` asks private `TargetArtifactBuildSessionAccess` to reserve a build read lease and bind it to the exact same-session `FinalElf` ref, moves the resulting opened lease into one sealed `ImmutableByteBackingRef`, applies admission/session canonical context and calls the common ELF core. No staged-ref API can open the object independently.

  Task 13 completes `buildLoadedTargetElfContract` from one owner-backed `VerifiedTargetArtifactModuleView`, which already closes exactly one module evidence record and all-and-only parent members for its content digest; it accepts no delivery ref, member array or compiler/runtime context. Standalone target delivery obtains the backing from `BoundTargetModuleSource`; package runtime obtains it from its package-owned `BoundBlobSource`. Each source adapter performs capability-relative open plus same-handle `stat/read/full-digest/stat`, then a repository-private sealed-backing factory pins that handle/content-addressed immutable object. Both call the source-neutral `verifyTargetElf(backing, contract, limits, encoding)`. `VerifiedElfMetadata` retains that exact backing/owner and cannot authorize another byte source; provider load must consume the same backing/handle, or a source token joined to it must recheck expected size/digest before every new read. Both contract factories bind exact entry set, registry, KAD/profile union, environment/fingerprint and owner generation. Actual runtime inventory/provider compatibility is a later RuntimeSession join, not an ELF-loader input. `WaferArtifact` never links PackageFormat or accepts free expected content, and no verifier reopens a verified locator.

  Use LLVM object/RISC-V attribute parsers to check ELF class/data, `EM_RISCV`, target flags, ISA extensions, MABI, required entry exports, symbol binding/visibility/COMDAT and undefined symbols. Every `wafer_tx81_*` undefined symbol fails. Other undefined symbols must resolve to the versioned non-Wafer subset in Task 3's same `VerifiedExternalSymbolRegistry`; arbitrary caller lists and `--allow-shlib-undefined` are not evidence. Every clone definition is selected exactly once and ends hidden/local; helper symbols cannot appear in the dynamic/public export set, and no non-clonable private/clone symbol remains undefined. Compiler `DeviceLink.cpp`, target-set loader and runtime construct the same explicit aggregate contract, verify entry symbols/KADs/fingerprints plus registry/toolchain linkage-policy contribution to `crtAndSymbolSetDigest`, and call these shared byte-view APIs.

  Call `readAndVerifyWaferAbiNotes` on the same opened final-ELF lease, index unique notes by KAD digest and stream-compare shared canonical/delivery backings. Compare every prepared relation. Rejoin grouped instruction/resource/KAD profile refs, recompute exact union/restricted profile/fingerprint through the session context, and require equality with `ModuleGroupKey` and `VerifiedTargetElfContract`. Units may contribute different compatible subsets. Stream the same handle through `ContentDigestBuilder`, check pre/post stat and seal its exact `FinalElf` `StagedObjectRef`; no locator is recorded. Construct `VerifiedTargetModuleEntry`/module only after all checks. After every group passes, replay core/clone/unit/entry/shape coverage exactly once; any missing/duplicate fails the call.

- [ ] **Step 5: Keep both fingerprints and content digest distinct**

  Unit tests must fail compilation or explicit verifier calls when a content digest is supplied as either fingerprint. Change target environment, codegen/CRT profile and final ELF bytes independently and show the three typed facts respond at their boundaries. Exercise identical, mutated, truncated and size-mismatched bytes only through fake transaction `StagedObjectRef` and trusted `BoundTargetModuleSource` providers; replacement before private session-access acquisition or mutation during its full streaming verification must fail same-handle stat/digest checks before metadata is exposed. Assert neither source nor staged ref has a public acquire/open/read method; cross-session/build-generation leases and readback above FD/reader/buffer limits fail before open. The common verifier accepts only the resulting private-construction immutable backing, never a raw buffer/path or caller-paired expected aggregate.

- [ ] **Step 6: Run focused gates**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests wafer-artifact-inspect wafer-opt -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv build/wafer-dev/test \
    --filter='wafer-device-link(-verified-module)?\.test'
  ```

  Expected: the verified-module positive actually executes every tool; all target/ABI/undefined mutations fail after the relevant step.

- [ ] **Step 7: Commit**

  ```bash
  git add include/Wafer/Artifact/TargetElfVerifier.h \
    lib/Wafer/Artifact/TargetElfVerifier.cpp lib/Wafer/Artifact/CMakeLists.txt \
    include/Wafer/Target/DeviceLink.h lib/Wafer/Target/DeviceLink.cpp \
    lib/Wafer/Target/CMakeLists.txt tools/wafer_device_link.py \
    tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp \
    test/Tools/wafer-device-link*.test \
    unittests/Artifact/TargetElfVerifierTest.cpp \
    unittests/Target/DeviceLinkTest.cpp unittests/CMakeLists.txt
  git commit -m "Verify final target modules after device link"
  ```

### Task 13: Complete Multi-Member TargetArtifactSet Staging and Transaction Attachment

**Files:**
- Create: `schema/wafer/target_artifact_set.proto`
- Modify: `schema/CMakeLists.txt`
- Modify: `include/Wafer/ABI/TargetArtifacts.h`
- Modify: `lib/Wafer/ABI/TargetArtifacts.cpp`
- Create: `include/Wafer/Artifact/TargetArtifactSet.h`
- Create: `lib/Wafer/Artifact/TargetArtifactSetInternal.h`
- Create: `lib/Wafer/Artifact/TargetArtifactSet.cpp`
- Modify: `lib/Wafer/Artifact/CMakeLists.txt`
- Create: `include/Wafer/Target/TargetArtifactSetBuilder.h`
- Create: `lib/Wafer/Target/TargetArtifactSetBuilder.cpp`
- Modify: `lib/Wafer/Target/CMakeLists.txt`
- Modify: `lib/Wafer/Compiler/ProgramOutputTransactionInternal.h`
- Modify: `lib/Wafer/Compiler/ProgramOutputTransaction.cpp`
- Modify: `tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp`
- Modify: `tools/check_target_dependency_layers.py`
- Create: `unittests/Artifact/TargetArtifactSetReaderTest.cpp`
- Create: `unittests/Artifact/TargetArtifactSetSchemaTest.cpp`
- Create: `unittests/ABI/TargetArtifactSetRecordTest.cpp`
- Create: `unittests/Target/TargetArtifactSetBuilderTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/Tools/wafer-target-artifact-set.test`
- Modify: `test/Tools/check-target-dependency-layers.test`

**Consumes:** one outer private `compiler::ProgramOutputTransaction` with its `compiler::StagedExecutableToken`, one typed `TargetVariantId`, validated target build/admission limits and the transaction-owned target-context registry/encoding/staging/budget state. `compiler::detail::TargetProgramOutputAccess::beginTargetArtifactBuild` resolves and binds the exact `VerifiedTargetCompilationContext`; caller-supplied context, entry/shape subset, module group, staging root or standalone output root is forbidden.

**Produces:** generated `WaferTargetArtifactSetProto`, locator-free reusable `TargetArtifactSetVerifiedRecord`, deterministic versioned `TargetArtifactSetDeliveryRoot` plus external exact-byte `ContentDigest`, FD-free `LoadedTargetArtifactSetMetadata`, an atomic runtime-private one-way batch path to `RuntimeBoundTargetArtifactSets`, and the sole `compiler::StagedTargetArtifactSetToken` owned by the outer transaction and created through its private lower core. It performs no independent visibility transition, and metadata loading never opens a final module.

```cpp
namespace wafer::abi {
namespace detail {
struct ImmutableProtoMessageOwnerStorage;
struct TargetArtifactSetRecordStorage;
class ImmutableProtoMessageOwnerFactory;
class TargetArtifactSetRecordVerifier;
} // namespace detail

class ImmutableProtoMessageOwner final {
public:
  ImmutableProtoMessageOwner(const ImmutableProtoMessageOwner &) = default;
  ImmutableProtoMessageOwner(ImmutableProtoMessageOwner &&) = default;
private:
  ImmutableProtoMessageOwner() = delete;
  explicit ImmutableProtoMessageOwner(
      std::shared_ptr<const detail::ImmutableProtoMessageOwnerStorage> storage);
  std::shared_ptr<const detail::ImmutableProtoMessageOwnerStorage> storage_;
  friend class detail::ImmutableProtoMessageOwnerFactory;
  friend class detail::TargetArtifactSetRecordVerifier;
};

class ProtoSubmessagePathStep final {
public:
  static llvm::Expected<ProtoSubmessagePathStep> singular(uint32_t fieldNumber);
  static llvm::Expected<ProtoSubmessagePathStep>
  repeated(uint32_t fieldNumber, uint64_t index);
private:
  ProtoSubmessagePathStep() = delete;
};

class ImmutableTargetArtifactSetRecordRef final {
public:
  const wafer::target_artifact::proto::TargetArtifactSetVerifiedRecord &
  message() const;
private:
  ImmutableTargetArtifactSetRecordRef() = delete;
  /* retains shared owner + owner-resolved bounded reflection path */
};

llvm::Expected<ImmutableTargetArtifactSetRecordRef>
resolveTargetArtifactSetVerifiedRecord(
    const ImmutableProtoMessageOwner &owner,
    llvm::ArrayRef<ProtoSubmessagePathStep> path,
    const ArtifactAdmissionLimits &limits);

class TargetArtifactMemberKey final {
public:
  static llvm::Expected<TargetArtifactMemberKey> create(
      EntryId entry, StaticFunctionDigest function,
      ContentDigest finalModuleContent);
  const EntryId &entryId() const;
  const StaticFunctionDigest &staticFunctionDigest() const;
  const ContentDigest &moduleContentDigest() const;
  friend bool operator==(const TargetArtifactMemberKey &,
                         const TargetArtifactMemberKey &);
  friend bool operator<(const TargetArtifactMemberKey &,
                        const TargetArtifactMemberKey &);
private:
  TargetArtifactMemberKey() = delete;
};

class VerifiedTargetArtifactModuleEvidence final {
public:
  VerifiedTargetArtifactModuleEvidence(
      const VerifiedTargetArtifactModuleEvidence &) = default;
  VerifiedTargetArtifactModuleEvidence(
      VerifiedTargetArtifactModuleEvidence &&) = default;
  const ContentDigest &moduleContentDigest() const;
  uint64_t moduleSize() const;
  const TargetBuildProfile &buildProfile() const;
  const TargetEnvironmentFingerprint &targetEnvironmentFingerprint() const;
  const UsedQuantStorageProfiles &usedQuantStorageProfiles() const;
  const TargetArtifactFingerprint &targetArtifactFingerprint() const;
  llvm::ArrayRef<VerifiedKernelAbiDescriptor> uniqueKernelAbis() const;
private:
  VerifiedTargetArtifactModuleEvidence() = delete;
  VerifiedTargetArtifactModuleEvidence(
      std::shared_ptr<const detail::TargetArtifactSetRecordStorage> storage,
      uint64_t moduleOrdinal);
  std::shared_ptr<const detail::TargetArtifactSetRecordStorage> storage_;
  uint64_t moduleOrdinal_;
  friend class detail::TargetArtifactSetRecordVerifier;
};

class VerifiedTargetArtifactSetRecordMember final {
public:
  VerifiedTargetArtifactSetRecordMember(
      const VerifiedTargetArtifactSetRecordMember &) = default;
  VerifiedTargetArtifactSetRecordMember(
      VerifiedTargetArtifactSetRecordMember &&) = default;
  const TargetArtifactMemberKey &key() const;
  llvm::StringRef entrySymbol() const;
  const KernelAbiSemanticDigest &kernelAbiDigest() const;
  const VerifiedKernelAbiDescriptor &kernelAbi() const;
  llvm::ArrayRef<ExecutableVariantId> owningExecutableVariants() const;
  llvm::ArrayRef<RankClassId> coveredRankClasses() const;
  llvm::ArrayRef<CompletionExportId> completionExports() const;
  VerifiedTargetArtifactModuleEvidence moduleEvidence() const;
  const TargetBuildProfile &buildProfile() const;
  const TargetEnvironmentFingerprint &targetEnvironmentFingerprint() const;
  const UsedQuantStorageProfiles &usedQuantStorageProfiles() const;
  const TargetArtifactFingerprint &targetArtifactFingerprint() const;
  uint64_t moduleSize() const;
private:
  VerifiedTargetArtifactSetRecordMember() = delete;
  VerifiedTargetArtifactSetRecordMember(
      std::shared_ptr<const detail::TargetArtifactSetRecordStorage> storage,
      uint64_t memberOrdinal);
  std::shared_ptr<const detail::TargetArtifactSetRecordStorage> storage_;
  uint64_t memberOrdinal_;
  friend class detail::TargetArtifactSetRecordVerifier;
};

class VerifiedTargetArtifactModuleView final {
public:
  VerifiedTargetArtifactModuleView(
      const VerifiedTargetArtifactModuleView &) = default;
  VerifiedTargetArtifactModuleView(
      VerifiedTargetArtifactModuleView &&) = default;
  VerifiedTargetArtifactModuleEvidence moduleEvidence() const;
  const ContentDigest &moduleContentDigest() const;
  llvm::ArrayRef<VerifiedTargetArtifactSetRecordMember> members() const;
private:
  VerifiedTargetArtifactModuleView() = delete;
  VerifiedTargetArtifactModuleView(
      std::shared_ptr<const detail::TargetArtifactSetRecordStorage> storage,
      uint64_t moduleOrdinal,
      llvm::SmallVector<VerifiedTargetArtifactSetRecordMember> members);
  std::shared_ptr<const detail::TargetArtifactSetRecordStorage> storage_;
  uint64_t moduleOrdinal_;
  llvm::SmallVector<VerifiedTargetArtifactSetRecordMember> members_;
  friend class detail::TargetArtifactSetRecordVerifier;
};

class VerifiedTargetArtifactSetRecord final {
public:
  VerifiedTargetArtifactSetRecord(
      const VerifiedTargetArtifactSetRecord &) = default;
  VerifiedTargetArtifactSetRecord(
      VerifiedTargetArtifactSetRecord &&) = default;
  const TargetArtifactSetId &id() const;
  const ExecutableSemanticDigest &sourceExecutableDigest() const;
  const TargetVariantId &targetVariantId() const;
  llvm::ArrayRef<VerifiedTargetArtifactModuleView> modules() const;
  llvm::ArrayRef<VerifiedTargetArtifactSetRecordMember> members() const;
  const wafer::target_artifact::proto::TargetArtifactSetVerifiedRecord &
  message() const;
  const CanonicalRecordBackingRef &canonicalRecordBacking() const;
private:
  VerifiedTargetArtifactSetRecord() = delete;
  VerifiedTargetArtifactSetRecord(
      std::shared_ptr<const detail::TargetArtifactSetRecordStorage> storage,
      llvm::SmallVector<VerifiedTargetArtifactModuleView> modules,
      llvm::SmallVector<VerifiedTargetArtifactSetRecordMember> members);
  std::shared_ptr<const detail::TargetArtifactSetRecordStorage> storage_;
  llvm::SmallVector<VerifiedTargetArtifactModuleView> modules_;
  llvm::SmallVector<VerifiedTargetArtifactSetRecordMember> members_;
  friend class detail::TargetArtifactSetRecordVerifier;
};

llvm::Expected<VerifiedTargetArtifactSetRecord>
verifyTargetArtifactSetVerifiedRecordView(
    const ImmutableTargetArtifactSetRecordRef &record,
    const ArtifactAdmissionLimits &limits,
    CanonicalEncodingContext &encoding);

llvm::Expected<VerifiedTargetArtifactSetRecord>
parseAndVerifyTargetArtifactSetVerifiedRecord(
    const ImmutableByteBackingRef &bytes,
    const ArtifactAdmissionLimits &limits,
    CanonicalEncodingContext &encoding);
} // namespace wafer::abi

namespace wafer::artifact {
namespace detail {
struct TargetArtifactSetMetadataStorage;
struct UnboundTargetModuleSourceDescriptorStorage;
struct BoundTargetModuleSourceStorage;
struct TrustedTargetArtifactSetDeliveryStorage;
struct TargetArtifactSetRuntimeStorage;
struct RuntimeBoundTargetArtifactSetsStorage;
class ProgramDeliveryTargetRefFactory;
class TargetArtifactSetMetadataLoader;
class TargetArtifactSetsBindingAccess; // noninstalled runtime-private access
} // namespace detail

class UnboundTargetModuleSourceDescriptor final {
public:
  UnboundTargetModuleSourceDescriptor(
      const UnboundTargetModuleSourceDescriptor &) = default;
  UnboundTargetModuleSourceDescriptor(
      UnboundTargetModuleSourceDescriptor &&) = default;
  const abi::ContentDigest &expectedContentDigest() const;
  uint64_t expectedSize() const;
private:
  UnboundTargetModuleSourceDescriptor() = delete;
  explicit UnboundTargetModuleSourceDescriptor(std::shared_ptr<const
      detail::UnboundTargetModuleSourceDescriptorStorage> storage);
  std::shared_ptr<const detail::UnboundTargetModuleSourceDescriptorStorage>
      storage_;
  friend class detail::TargetArtifactSetMetadataLoader;
  friend class detail::TargetArtifactSetsBindingAccess;
};

class BoundTargetModuleSource final {
public:
  BoundTargetModuleSource(const BoundTargetModuleSource &) = default;
  BoundTargetModuleSource(BoundTargetModuleSource &&) = default;
  const abi::ContentDigest &expectedContentDigest() const;
  uint64_t expectedSize() const;
private:
  BoundTargetModuleSource() = delete;
  explicit BoundTargetModuleSource(
      std::shared_ptr<const detail::BoundTargetModuleSourceStorage> storage);
  std::shared_ptr<const detail::BoundTargetModuleSourceStorage> storage_;
  friend class detail::TargetArtifactSetsBindingAccess;
  friend class detail::RuntimeArtifactVerificationSessionAccess;
};

class TrustedTargetArtifactSetDeliveryRef final {
public:
  TrustedTargetArtifactSetDeliveryRef(
      TrustedTargetArtifactSetDeliveryRef &&) noexcept;
  TrustedTargetArtifactSetDeliveryRef(
      const TrustedTargetArtifactSetDeliveryRef &) = delete;
  ~TrustedTargetArtifactSetDeliveryRef();
  const abi::TargetArtifactSetId &setId() const;
private:
  TrustedTargetArtifactSetDeliveryRef() = delete;
  explicit TrustedTargetArtifactSetDeliveryRef(
      std::unique_ptr<detail::TrustedTargetArtifactSetDeliveryStorage> storage);
  std::unique_ptr<detail::TrustedTargetArtifactSetDeliveryStorage> storage_;
  friend class detail::ProgramDeliveryTargetRefFactory;
  friend class detail::TargetArtifactSetMetadataLoader;
};

class LoadedTargetArtifactSetMetadata final {
public:
  LoadedTargetArtifactSetMetadata(LoadedTargetArtifactSetMetadata &&) noexcept;
  LoadedTargetArtifactSetMetadata(
      const LoadedTargetArtifactSetMetadata &) = delete;
  ~LoadedTargetArtifactSetMetadata();
  const abi::VerifiedTargetArtifactSetRecord &verifiedRecord() const;
  const abi::TargetArtifactSetId &id() const;
  const abi::ExecutableSemanticDigest &sourceExecutableDigest() const;
  const abi::TargetVariantId &targetVariantId() const;
  llvm::ArrayRef<abi::VerifiedTargetArtifactModuleView> modules() const;
  const abi::ContentDigest &deliveryContentDigest() const;
  uint64_t deliverySize() const;
private:
  LoadedTargetArtifactSetMetadata() = delete;
  explicit LoadedTargetArtifactSetMetadata(
      std::shared_ptr<const detail::TargetArtifactSetMetadataStorage> storage);
  std::shared_ptr<const detail::TargetArtifactSetMetadataStorage> storage_;
  friend class detail::TargetArtifactSetMetadataLoader;
  friend class detail::TargetArtifactSetsBindingAccess;
};

class VerifiedTargetArtifactMember final {
public:
  VerifiedTargetArtifactMember(const VerifiedTargetArtifactMember &) = default;
  VerifiedTargetArtifactMember(VerifiedTargetArtifactMember &&) = default;
  const abi::VerifiedTargetArtifactSetRecordMember &recordMember() const;
  const abi::TargetArtifactMemberKey &key() const;
  const abi::EntryId &entryId() const;
  const abi::StaticFunctionDigest &staticFunctionDigest() const;
  llvm::StringRef entrySymbol() const;
  const abi::VerifiedKernelAbiDescriptor &kernelAbi() const;
  llvm::ArrayRef<abi::RankClassId> coveredRankClasses() const;
  llvm::ArrayRef<abi::CompletionExportId> completionExports() const;
  const abi::TargetBuildProfile &buildProfile() const;
  const abi::TargetEnvironmentFingerprint &targetEnvironmentFingerprint() const;
  const abi::UsedQuantStorageProfiles &usedQuantStorageProfiles() const;
  const abi::TargetArtifactFingerprint &targetArtifactFingerprint() const;
  const abi::ContentDigest &moduleContentDigest() const;
  uint64_t moduleSize() const;
  const BoundTargetModuleSource &moduleSource() const;
private:
  VerifiedTargetArtifactMember() = delete;
  VerifiedTargetArtifactMember(
      std::shared_ptr<const detail::TargetArtifactSetRuntimeStorage> storage,
      uint64_t memberOrdinal);
  std::shared_ptr<const detail::TargetArtifactSetRuntimeStorage> storage_;
  uint64_t memberOrdinal_;
  friend class detail::TargetArtifactSetsBindingAccess;
};

class TargetArtifactSet final {
public:
  TargetArtifactSet(TargetArtifactSet &&) noexcept;
  TargetArtifactSet(const TargetArtifactSet &) = delete;
  ~TargetArtifactSet();
  const abi::VerifiedTargetArtifactSetRecord &verifiedRecord() const;
  const abi::TargetArtifactSetId &id() const;
  const abi::ExecutableSemanticDigest &sourceExecutableDigest() const;
  const abi::TargetVariantId &targetVariantId() const;
  llvm::ArrayRef<VerifiedTargetArtifactMember> members() const;
  const abi::ContentDigest &deliveryContentDigest() const;
  uint64_t deliverySize() const;
  llvm::ArrayRef<BoundTargetModuleSource> uniqueModuleSources() const;
private:
  TargetArtifactSet() = delete;
  TargetArtifactSet(
      std::shared_ptr<const detail::TargetArtifactSetRuntimeStorage> storage,
      llvm::SmallVector<VerifiedTargetArtifactMember> members);
  std::shared_ptr<const detail::TargetArtifactSetRuntimeStorage> storage_;
  llvm::SmallVector<VerifiedTargetArtifactMember> members_;
  friend class detail::TargetArtifactSetsBindingAccess;
};

class RuntimeBoundTargetArtifactSets final {
public:
  RuntimeBoundTargetArtifactSets(RuntimeBoundTargetArtifactSets &&) noexcept;
  RuntimeBoundTargetArtifactSets(
      const RuntimeBoundTargetArtifactSets &) = delete;
  ~RuntimeBoundTargetArtifactSets();
  llvm::ArrayRef<TargetArtifactSet> sets() const;
private:
  RuntimeBoundTargetArtifactSets() = delete;
  RuntimeBoundTargetArtifactSets(
      std::shared_ptr<const detail::RuntimeBoundTargetArtifactSetsStorage>,
      llvm::SmallVector<TargetArtifactSet> sets);
  std::shared_ptr<const detail::RuntimeBoundTargetArtifactSetsStorage> storage_;
  llvm::SmallVector<TargetArtifactSet> sets_;
  friend class detail::TargetArtifactSetsBindingAccess;
};

namespace detail {
class TargetArtifactSetsBindingAccess final {
public:
  static llvm::Expected<RuntimeBoundTargetArtifactSets> bind(
      llvm::SmallVector<LoadedTargetArtifactSetMetadata> metadata,
      RuntimeArtifactVerificationSession runtimeSession);
private:
  TargetArtifactSetsBindingAccess() = delete;
};
} // namespace detail

llvm::Expected<LoadedTargetArtifactSetMetadata>
loadAndVerifyTargetArtifactSetMetadata(
    const TrustedTargetArtifactSetDeliveryRef &delivery,
    ArtifactMetadataVerificationSession &session);

} // namespace wafer::artifact

namespace wafer::target {
// Build/stage/attach APIs are declared in
// Wafer/Target/TargetArtifactSetBuilder.h.
mlir::FailureOr<compiler::StagedTargetArtifactSetToken>
buildAndAttachTargetArtifactSet(
    compiler::ProgramOutputTransaction &output,
    const compiler::StagedExecutableToken &executable,
    const abi::TargetVariantId &targetVariantId,
    const TargetArtifactBuildLimits &buildLimits,
    const abi::ArtifactAdmissionLimits &admissionLimits);
} // namespace wafer::target
```

`detail::ProgramDeliveryTargetRefFactory` is the sole issuer of the move-only trusted ref and binds one verified standalone delivery child commitment to its root capability, relative locator, external size/digest/set ID and delivery-owner generation. `detail::TargetArtifactSetMetadataLoader` alone constructs `LoadedTargetArtifactSetMetadata`. Its shared const metadata storage owns the verified record, exact delivery facts, root-owner capability/token, canonical `UnboundTargetModuleSourceDescriptor`s and exact metadata-session proof/backing owner, but no runtime-session token, module FD, ELF metadata or open-capable source. An unbound descriptor exposes only expected typed content metadata; it has no acquire/open/read, locator or friendship with runtime session access. Metadata values support inspection/selection only.

The noninstalled Artifact-owned `TargetArtifactSetInternal.h` defines the exact callable `detail::TargetArtifactSetsBindingAccess::bind(SmallVector<LoadedTargetArtifactSetMetadata>, RuntimeArtifactVerificationSession)`. The Runtime-owned bundle factory is its only caller and first proves plan/context plus canonical all-required/no-extra target coverage and creates a service-bound runtime artifact session; Artifact access does not repeat those Runtime semantics. It atomically move-consumes both arguments, validates only Artifact-local metadata backing/root owners, target uniqueness/order, session-local/hard feasibility, runtime-session liveness and every unbound descriptor, while reserving no invocation capacity and opening nothing. It then creates FD-free bound sources/sets and moves the artifact session into batch storage. Later `RuntimeSession::create` must consume this batch with the selected plan and all-or-none reserve invocation capacity before source access can mint a lease. There is no installed/public/single-set binder, intermediate result or recovery of either moved argument; any failure destroys the whole vector/session without a partial set.

- [ ] **Step 1: Add failing delivery-schema and complete-set tests**

  `TargetArtifactSetSchemaTest.cpp` requires two non-overlapping messages. `TargetArtifactSetVerifiedRecord` is locator-free, has nested-record WCRE field semantics, and embeds exactly one imported `TargetArtifactSetIdentity` plus its claimed `TargetArtifactSetId`. It stores one canonical `ModuleEvidence` per unique final module `ContentDigest` containing size, environment/artifact fingerprints, restricted build profile, exact used profiles and unique KAD delivery records. Lean members are keyed by imported `TargetArtifactMemberKey` and carry only symbol, KAD digest and owning variant/rank/completion joins; their build/profile/KAD accessors delegate to module evidence. The digest is a structural key, not a new `ModuleId`. `TargetArtifactSetDeliveryRoot` contains exactly one verified record plus physical blob size/safe relative locator records and has no semantic/WCRE message options. Neither redeclares source/target/fingerprint/member fields or carries unused global registry entries. Assert only `semantic_identity.proto::TargetArtifactSetIdentity` owns top-level WCRE record type 9 and its field numbers. Reject duplicate/missing/unreferenced module evidence, member-to-module/KAD join mismatch, maps, floating fields, locally redeclared digest/ID/KAD/member messages, missing reserved numbers, locator fields in the verified record, semantic options on delivery-only fields and a package-facing import of the delivery root. Serialize one verified fixture/root twice and require exact bytes/content digest; reorder source construction without changing canonical sets and require the same bytes. A 1,000-entry/one-module fixture must contain one build/profile/KAD evidence copy and serialized size must grow only by the lean member relation. Add 100K/1M-member fixtures that force WCRE external SET sorting and Protobuf file backing; measured peak memory/FDs remain within the canonical/admission contexts, strong member IDs remain fixed-size and record proofs share the owner/backing. For every member/module/KAD/string/blob/nested-record limit, instrument generated Protobuf allocation and require `limit+1`, length overflow, packed-field abuse and appended unknown fields to fail in wire preflight with zero generated-message allocation; at-limit fixtures must parse and match the preflight counters. Unsupported delivery version, changed expected delivery digest/size and a self-embedded delivery digest must fail before a set proof is returned.

  Positive uses two committed shape variants referencing one `TargetVariantId`, at least three members, two static functions, one artifact relation shared by ABI/code/projection-compatible rank classes across the shapes, a private helper/global closure, a policy limit that forces a second module, and multiple completion exports. A separate target requirement produces a separate set. Verify the compiler-created record, ABI verifier's owner-backed `message()` view/canonical stream, delivery-embedded record and loader proof are equal. Move-own a PackageManifest containing the same embedded record, resolve it through an owner-validated reflection path and require `verifyTargetArtifactSetVerifiedRecordView` to retain the parent owner without copying the full submessage or deterministic bytes; typed indexes are separately bounded small objects. Parsing the standalone record twice must produce equal streams without rebuilding/reordering members. Check each member carries:

  ```text
  EntryId + static-function semantic digest
  -> final module content digest + entry symbol + descriptor semantic digest
     + covered RankClassIds + completion exports
  ```

  The set root includes source executable semantic digest, typed `TargetVariantId`, every member's two fingerprints, every final module content digest and the canonical member list. `abi::TargetArtifactMemberKey` is the sole runtime-safe representation of `(EntryId, StaticFunctionDigest, final module ContentDigest)` and defines canonical comparison/hash; it is a structural lookup key, not a new semantic ID or `ModuleId`, and excludes symbol, KAD and locator. `VerifiedTargetArtifactMember` stores this key once; its entry/static/module accessors delegate to it. Package/runtime reuse the exact type. Members sort by its canonical fields; the final content digest itself is module identity. Replaying coverage per `ExecutableVariantId` must close every entry, rank class, projection and completion relation even when a compatible artifact member is shared.

  Negative cases: candidate executable, unknown/cross-executable target ID, no committed shape variant for the target, omitted/duplicated/cross-target shape variant, stale executable digest, duplicate/missing entry or rank class in either shape, mixed incompatible pinned projections, member reorder, same KAD with changed code/module digest, same code from another executable, entry symbol mismatch, descriptor semantic digest mismatch, fingerprint/policy mismatch, module blob/size mismatch, absolute/parent-traversing locator, one locator claimed with inconsistent digest/size, duplicate physical blob index entry, symlink escape, missing/extra completion export and incomplete per-variant coverage. Multiple entry members may reference one locator only when they carry the same final module content digest and size; shared-module identity is content equality, not path equality. Assert member/set proof types are non-aggregate, not publicly constructible/mutable and cannot be fabricated from delivered fields. Extend the dependency gate to standalone-compile installed `ProgramOutputTransaction.h` using only lower Compiler/Delivery dependencies and reject any Target/Package/Whole include, public upper-layer attach parameter or package-view return. Target-specific full objects may appear only in target sources; `ProgramOutputTransactionInternal.h` contains distinct lower commitments consumed by owner adapters.

- [ ] **Step 2: Submit `target_artifact_set.proto` as a focused ABI review**

  Use `package wafer.target_artifact.proto`; import `semantic_identity.proto` and `kernel_abi.proto`, never copying their messages/enums/options. Define only the versioned locator-free verified record and separate deterministic delivery root fixed by `tasks/14`. The verified record embeds imported `TargetArtifactSetIdentity`/claimed ID and all semantic verification evidence; its message is nested-only record type 0 with explicit field WCRE semantics so package schema may import/embed this record without reinterpreting fields. Define canonical unique module evidence keyed by final `ContentDigest`, then lean members joined through the exact `(EntryId, StaticFunctionDigest, final module ContentDigest)` key and KAD digest; no symbol/KAD/locator field is allowed to become a second module/member identity. The delivery root embeds the verified record and adds only one physical locator/size join per unique module digest; it is explicitly nonidentity and has no WCRE options. Reserve removed wire numbers, generate `WaferTargetArtifactSetProto` with the exact shared Protobuf helper, and keep generated files in the build tree. `WaferABI` links that generated target and owns `ImmutableProtoMessageOwner`, bounded owner/path resolution, `verifyTargetArtifactSetVerifiedRecordView` and the production `ImmutableByteBackingRef` parser.

  `detail::ImmutableProtoMessageOwnerFactory` is the only constructor friend. Its generated-message path is callable only by repository verified builders; its parsed-message path requires the exact non-forgeable `ProtoAdmissionInternal` preflight proof, generated parse result and immutable input backing, and rechecks descriptor/count equality before sealing storage. Therefore production code cannot first use an unbounded parser and later launder the message through a public `adopt`. A verifier never accepts a raw `(submessage pointer, shared owner)` pair, raw caller byte span or `deterministicBytes`; parsed owners/proofs retain the exact immutable input backing, while embedded views resolve their descriptor/path from the retained message owner. A noninstalled inline test adapter is capped by `ArtifactAdmissionLimits::maxInlineRecordBytes` and delegates to the production parser after move-owning bytes. `WaferPackageFormat` can link only `WaferABI`/generated schemas and reuse this verifier/preflight owner factory without `WaferArtifact`, MLIR or a copied semantic algorithm. Package/runtime may retain the verifier-returned exact message view/canonical/input backing or reference typed set/member keys, but must never rebuild fields from accessors or embed the delivery root into record 10.

  Register two schema-owned immutable admission plans with the shared engine: one for standalone `TargetArtifactSetVerifiedRecord`, one for `TargetArtifactSetDeliveryRoot`. They reference generated descriptors/options rather than a copied field-number table and account separately for module evidence, lean members, physical blobs, KAD deliveries/slots/completions, profile records, locators, strings and aggregate nested records. The preflight scanner must finish and bind exact counters/spans before either generated parser runs; the post-parse reflection check must equal the proof before owner construction. An embedded record resolved from a PackageManifest is accepted only when the parent `ImmutableProtoMessageOwner` storage carries the package parser's matching preflight provenance.

  `detail::TargetArtifactSetRecordVerifier` is the sole implementation-only friend of `VerifiedTargetArtifactModuleView`. It creates one const storage per module view containing the shared immutable parent message/backing owner, parent generation, one evidence index and the canonical all-and-only member index; caller code cannot manufacture or replace any component. Module views are cheap copyable immutable references whose storage keeps those relations alive after the parent parse frame exits. The verifier rejects any parent/module/member owner mismatch before publishing the record or a view.

  Delivery bytes use pinned deterministic C++ serialization and have an external `ContentDigest + size`; that pair is returned to package assembly and never encoded inside its own bytes. Review field numbers, canonical collection semantics, version and locator exclusion as one ABI patch before implementation.

- [ ] **Step 3: Orchestrate the whole-target proof chain once**

  `buildAndAttachTargetArtifactSet` performs only this production sequence inside the caller's private outer transaction:

  ```text
  TargetProgramOutputAccess::beginTargetArtifactBuild(
      output, compiler::StagedExecutableToken, TargetVariantId, build limits)
    -> TargetArtifactBuildSession(owner/context/staging/budgets/cancellation)
  verifyTargetCoverage(session)
    -> prepareTargetCoverage(move verified coverage, same session)
    -> verifyCanonicalTargetPacking(move prepared coverage, same session)
    -> packLinkAndVerifyTargetCoverage(move canonical packing,
                                       admission limits, same session)
    -> immutable complete VerifiedTargetModules
    -> build record/delivery and attach through same session
  ```

  It does not reimplement closure discovery, conversion, KAD construction, exact note sizing, packing or link verification and cannot access a per-unit/group factory. Tests inspect the proofs returned by Tasks 8A/8B/12 to reassert whole-target exactly-once coverage, canonical partition and `limit-1/limit/limit+1` behavior, including permutation/repeated-build/private-ref cases, but use the production APIs rather than a second algorithm.

- [ ] **Step 4: Build members from the complete modules proof before computing the root**

  Consume every module/entry from `VerifiedTargetModules`; there is no vector/subset overload. First create one `VerifiedTargetArtifactModuleEvidence` per unique final content digest, containing module size, restricted build profile, environment/final artifact fingerprints, exact final-module used-profile union and canonical unique KADs; equal digest with unequal evidence is a collision. Coalesce an entry sharing candidate only when `EntryId`, static digest, verified KAD digest, canonical entry symbol, projection dependency and completion exports are identical; union its canonically ordered owning variants/rank classes. Expand every resulting `VerifiedTargetModuleEntry` into one lean non-forgeable member that stores its typed key/symbol/KAD-digest/rank/completion relation and an exact join to module evidence; build/profile/KAD/size accessors delegate rather than copy. Reject missing/duplicate/unreferenced evidence or joins and never zip separately ordered arrays. Replay modules proof's source/target/shape IDs, clone/core coverage and all per-variant entry/rank/projection/completion relations. Only after all final content digests exist and coverage closes, populate the generated target-artifact-set identity message; the output-access friend calls record-9 factory with its private session context and computes `wafer.target-artifact-set.v1`.

  No partition key, path, build order or set ID is a member input. The acyclic order is static relation -> KAD -> prelink unit -> canonical packing -> final link/note -> final ELF content digest -> set root.

- [ ] **Step 5: Stage immutable bytes and attach without publishing**

  Construct the locator-free generated record from verified identity/ID/evidence, pass it through the repository-only generated-message path of `ImmutableProtoMessageOwnerFactory`, resolve the root view and call the record verifier through the session's private encoding context. Retain only owner-backed view, shared canonical backing and typed indexes. Populate `TargetArtifactSetDeliveryRoot` from that message plus one physical blob join per unique final-module `StagedObjectRef`; stream deterministic Protobuf directly into the session's bounded staging writer and raw digest builder, producing exact `TargetSetDelivery` `StagedObjectRef` without a full vector or self-digest. For record, delivery and each unique `FinalElf` readback, private `TargetArtifactBuildSessionAccess` signs a fresh `TargetBuildReadLease`, binds it to the exact same-session ref and retains reader/FD/buffer capacity in the opened handle through verification; close each bounded batch before fsync of the private capability-owned subtree. Multiple members share one FD-free ref. Cross-session/generation pairing, repeat use of a consumed lease and acquisition beyond outer/stage limits fail before open. Internal temporary locators never enter proof or identity and are not exposed.

  Build one noninstalled target-owned `StagedTargetArtifactSet` containing target/context identity, owner-backed verified record, delivery `StagedObjectRef`/digest/size, private-root capability and the complete immutable modules proof. Then call exactly `compiler::detail::TargetProgramOutputAccess::attachTargetArtifactSet(output, executableToken, std::move(stagedSet))` while the `TargetArtifactBuildSession` remains live. The adapter rechecks stable canonical owner token, session/staging/attachment-table generation, exact coverage and all staged object refs, constructs the distinct lower target commitment understood by the transaction private core, takes ownership and returns the sole `compiler::StagedTargetArtifactSetToken`; target code defines no parallel token and no rename/index write occurs. The installed `ProgramOutputTransaction` header declares no public attach accepting a target-layer object. The private core reserves logical/physical bytes, roots, metadata, FDs, workers and buffers all-or-none against creation-time `ProgramOutputLimits`; stage limits use the per-field minimum and never change partition. Any child failure permanently marks it uncommittable. Content dedup remains private until outer commit. Smaller completion scopes require separate transactions and cannot be upgraded.

  The compiler token is move-only, non-aggregate and bound to the transaction's stable owner/attachment identity; attaching another target advances the attachment table but does not change or stale any existing `compiler::StagedExecutableToken`/`compiler::StagedTargetArtifactSetToken`. Commit, abort or owner finalization invalidates them. Tokens expose only typed target/set IDs, not record, locator, root capability or module source. Package assembly never accepts a caller-supplied token array: after all required targets attach, the package owner's noninstalled `compiler::detail::PackageProgramOutputAccess::buildPackageAssemblyInputView(output, executableToken)` scans the lower attachment snapshot plus each owner-sealed typed access, proves every required target attachment exactly once with no extras, and returns the non-forgeable `compiler::PackageAssemblyInputView` defined by `tasks/15`. `ProgramOutputTransaction` itself neither constructs nor publicly returns that upper-layer view. Only the package-created view captures the current `AttachmentTableGeneration`; a later target/package attach stales an older view while stable tokens remain valid. The view exposes only locator-free verified records and capability-bound sources.

  On conversion, KAD, link, ELF, digest, set verifier, write, fsync or attach failure, abort the target attachment and leave the outer transaction uncommittable. Private immutable roots may be reclaimed immediately or later by GC, but without a trusted outer reference they are unreachable and never accepted.

- [ ] **Step 6: Verify metadata first, then bind one-way for runtime module access**

  `TrustedTargetArtifactSetDeliveryRef` is created only by a verified standalone `ProgramDeliveryCommitRecord` target-root commitment and owns an already-open root capability, safe relative locator, expected set ID, delivery digest and size. PackageManifest/bundle readers cannot create this ref because they embed only the locator-free record and bind their own module locators. There is no public raw `(path,digest,size)` constructor or loader overload. The runtime-neutral metadata entry is exactly `loadAndVerifyTargetArtifactSetMetadata(delivery, metadataSession)`. Private `ArtifactMetadataVerificationSessionAccess` validates host/artifact-child/session owner generations, reserves joined metadata reader/workspace/FD capacity and borrows retained admission/canonical contexts. It rejects oversized delivery structures, obtains one same-handle sealed backing for the exact delivery root after checked size plus streaming digest, and runs schema-owned wire preflight before generated-message allocation. Post-parse counters/spans must match before owner sealing. It resolves the locator-free record and calls `verifyTargetArtifactSetVerifiedRecordView`, the sole owner of record-9 recomputation, claimed-ID comparison, module-view/all-member joins, KAD/profile/build evidence validation, canonical order and fingerprint replay. It joins every module view to exactly one safe physical locator/size/digest and seals that relation as `UnboundTargetModuleSourceDescriptor`, rejecting missing/extra/duplicate blobs, but does not open or parse any module ELF. Returned metadata retains exact record/backing/root/descriptors plus immutable host/artifact-domain provenance, no active lease/FD; it may outlive the parsing session for inspection/batch bind, while stale host/root generation still fails binding.

  Standalone execution collects metadata for every target required by the verified program delivery, proves canonical all-required/no-extra coverage, then calls the noninstalled Artifact batch access with the complete metadata vector and one independently created service-bound `RuntimeArtifactVerificationSession` by value. Binding performs no open or invocation reservation: it prevalidates Artifact owners, descriptor generations, cross-set uniqueness and session-local/hard feasibility, then atomically creates `RuntimeBoundTargetArtifactSets` with FD-free sources and the moved artifact session. Cross-owner substitution, same IDs over another backing, stale metadata, missing/extra/reordered set, mismatched service domain or retry after move fails without a partial set. Package runtime performs its analogous batch bind from `LoadedPackageMetadata`/embedded module views and never constructs a standalone target delivery ref.

  Only after `RuntimeSession::create` has joined the selected plan, exact service owner and invocation capacity may its private adapter ask `RuntimeArtifactVerificationSessionAccess` to open a selected bound module. That access signs a reservation-only read lease jointly charging host/child/service/invocation ledgers; source acquire move-consumes it, opens exactly once and returns handle plus reservation. A work lease similarly covers worker/module bytes/workspace. Move both into the sealed backing, build the exact ELF contract and run `verifyTargetElf` plus provider load on that same handle before publishing a module handle. Close bounded batches; long-lived sets retain no FD. Reopening requires fresh joined leases and full verification. Replacement, mutation, cancellation, cross-session/invocation use and budget exhaustion fail before publication. No MLIR handle, module-byte copy or path-derived ID escapes.

  Compiler relation verification is not a public target API and never accepts an arbitrary `ExecutableOp`. During target attachment, `TargetProgramOutputAccess` compares the complete set proof with the same session-held sealed executable projection by invoking Task 8A's whole-coverage, Task 8B's preparation and Task 12's canonical-packing verifiers; it reuses their immutable proofs rather than implementing another graph or packing algorithm. Later, the package-owned `PackageProgramOutputAccess` checks the lower attachment snapshot against owner-sealed target facts and forms its view; the transaction does not depend on the package type. An internally valid set that omits one shape variant therefore cannot attach or enter package assembly. Package assembly consumes those sealed joins without replaying mutable IR, while package/runtime link runtime-safe `WaferArtifact` only for delivery and ELF verification. Neither path inspects the internal directory layout.

- [ ] **Step 7: Inject partition/member publication failures**

  Unit-test a filesystem/outer-transaction abstraction that fails closure/clone conversion, prelink emission/measurement, exact packing, link, verification, write, fsync or attach at each unit/module/member index. With failure on the last member, earlier private objects/ELFs may exist, but no trusted delivery ref can be created and the public loader has no callable orphan-root path. Inject every operational build limit at `limit-1/limit/limit+1`, including subprocess CPU/wall/RSS/output/cancel/reap: a preflight graph/count breach spawns no worker, a cumulative byte/FD/staging breach cleans or orphans only private staging, and no limit may cause member omission or noncanonical repartition. Inject every admission/canonical limit against a recomputed valid delivery digest and require failure before ELF open when the delivery structure itself exceeds policy. Prove a borrowed embedded-record proof keeps exactly one shared parent message/backing alive after PackageManifest's local parse frame exits, and that no API can pair an unrelated owner/path or mutate the message after verification.

  Metadata-load a 1,000-module set plus a heterogeneous second target under low `RLIMIT_NOFILE`, instrument opens and require only bounded delivery-metadata handles, zero module opens and FD-free metadata values. Finalize metadata sessions and prove immutable metadata remains inspectable but gains no open authority. The runtime-private test bundle owner proves all-required/no-extra and batch-binds the complete canonical vector once; assert this reserves zero invocation capacity. Missing/extra/reordered metadata, failure at the final descriptor, a second bind, copied record view, foreign backing owner or metadata/runtime session substitution must publish no individual set/source. Next create `RuntimeSession` from the selected plan, bound batch and one exact invocation capacity proof; failure must discard the batch without an open. Only then replace/mutate module locators and sign two concurrent runtime read/work pairs for one bound source, proving independent reads and exact joined host/service/invocation charges. Cancellation before backing seal destroys pending leases; an already returned backing remains valid until retained leases release. Cross-session/invocation pairing, module-view/backing substitution and `VerifiedElfMetadata` reuse fail. Under low FD limits, selected modules open only in bounded batches and the runtime batch retains no per-module FD; mutate after one verification and require fresh provider-time acquisition to catch it. Run the same matrix against package metadata/`BoundBlobSource`. Repeat staging twice and require identical partition/root/delivery bytes under different legal concurrency limits.

- [ ] **Step 8: Run focused gates**

  ```bash
  cmake --build build/wafer-dev --target WaferTargetArtifactSetProto WaferUnitTests wafer-artifact-inspect -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  <configured-lit> -sv build/wafer-dev/test/Tools/wafer-target-artifact-set.test
  <configured-lit> -sv build/wafer-dev/test/Tools/check-target-dependency-layers.test
  python3 tools/check_target_dependency_layers.py
  ```

- [ ] **Step 9: Commit**

  ```bash
  git add schema/wafer/target_artifact_set.proto schema/CMakeLists.txt \
    include/Wafer/ABI/TargetArtifacts.h lib/Wafer/ABI/TargetArtifacts.cpp \
    include/Wafer/Artifact/TargetArtifactSet.h \
    lib/Wafer/Artifact/TargetArtifactSetInternal.h \
    lib/Wafer/Artifact/TargetArtifactSet.cpp lib/Wafer/Artifact/CMakeLists.txt \
    include/Wafer/Target/TargetArtifactSetBuilder.h \
    lib/Wafer/Target/TargetArtifactSetBuilder.cpp \
    lib/Wafer/Target/CMakeLists.txt \
    lib/Wafer/Compiler/ProgramOutputTransactionInternal.h \
    lib/Wafer/Compiler/ProgramOutputTransaction.cpp \
    tools/wafer-artifact-inspect/wafer-artifact-inspect.cpp \
    tools/check_target_dependency_layers.py \
    unittests/Artifact/TargetArtifactSetReaderTest.cpp \
    unittests/Artifact/TargetArtifactSetSchemaTest.cpp \
    unittests/ABI/TargetArtifactSetRecordTest.cpp \
    unittests/Target/TargetArtifactSetBuilderTest.cpp unittests/CMakeLists.txt \
    test/Tools/wafer-target-artifact-set.test \
    test/Tools/check-target-dependency-layers.test
  git commit -m "Attach complete target artifact sets transactionally"
  ```

### Task 14: Named Driver Consumption and Actual End-to-End Gate

**Files:**
- Modify: `include/Wafer/Pipelines/Pipelines.h`
- Modify: `lib/Wafer/Pipelines/Pipelines.cpp`
- Modify: `lib/Wafer/Pipelines/CMakeLists.txt`
- Modify: `tools/wafer-opt/wafer-opt.cpp`
- Modify: `tools/wafer-opt/CMakeLists.txt`
- Modify: `test/lit.site.cfg.py.in`
- Modify: `test/lit.cfg.py`
- Create: `test/Tools/wafer-opt-target-artifact-set.test`
- Create: `test/Tools/Inputs/target-artifact-program/functions/forward.mlir`
- Create: `test/Tools/Inputs/target-artifact-program/functions/forward.meta`
- Create: `test/Tools/Inputs/target-artifact-program-invalid/functions/forward.mlir`
- Create: `test/Tools/Inputs/target-artifact-program-invalid/functions/forward.meta`

**Consumes:** the whole-variant `stablehlo-to-executable` named program pipeline result, its still-private `ProgramOutputTransaction`, and Task 13's staging/attachment builder.

**Produces:** production driver output containing one complete target artifact set for every accepted target variant; no manual pass assembly is required.

- [ ] **Step 1: Add a failing real driver test**

  The positive fixture enters through the real program-directory frontend, commits two shape variants that reference one target requirement, and contains a private helper closure plus enough entries/rank classes to force multiple modules under an explicit positive `ModulePartitionPolicyV1`. Run:

  ```bash
  wafer-opt --program-pipeline=stablehlo-to-executable \
    --input-program-dir %S/Inputs/target-artifact-program \
    --output-program-delivery %t.output \
    --execution-mesh-ranks=2 \
    --target-artifact-build-limits-profile=lit-v1 \
    --artifact-admission-limits-profile=lit-v1 \
    --canonical-encoding-limits-profile=lit-v1
  wafer-artifact-inspect verify-program-target-sets \
    --program-output-root=%t.output \
    --admission-limits-profile=lit-v1
  ```

  `verify-program-target-sets` first verifies the outer unified-root/`ProgramDeliveryCommitRecord` and obtains non-forgeable `TrustedTargetArtifactSetDeliveryRef`s from its typed index. Its repository-owned diagnostic service obtains one bounded `ArtifactMetadataHostBudgetCapability` from the shared host registry, move-creates one `ArtifactMetadataVerificationRegistry`, then creates bounded sessions from that registry for all refs and calls `loadAndVerifyTargetArtifactSetMetadata`. The resulting metadata report cannot bind or construct a runtime/migration input, and instrumentation requires zero final-module opens in this path. A separate noninstalled tool-only inspection source/session, disjoint from both production sessions and nonconvertible to runtime sources, performs the explicit final ELF magic/note/export byte checks under its own joined host limits. Together the test checks actual target LLVM translation, core/clone objects, target object, CRT object, linked final ELF, mandatory notes, both fingerprints, descriptor semantic digests, content digests and complete member count. It rejects logs containing `print-commands` as the only action and never derives trust from a target root itself.

  Require exactly one trusted committed set reference for the shared `TargetVariantId`, and have `verify-program-target-sets` report both owning `ExecutableVariantId`s with complete rank/entry coverage. Add a second target requirement to prove the driver stages a distinct set per target, then publishes both or neither, never one set per shape variant.

  The negative program uses only loader-consumed `functions/forward.mlir` and `functions/forward.meta`, reaches the shared target geometry gate with an out-of-range ABI dimension, and leaves `%t.failed` without a trusted program-delivery root/commit record. Private immutable orphans, if failure injection preserves them, are unreachable to the inspector/loader. It does not depend on ignored JSON or ad hoc metadata files.

- [ ] **Step 2: Extend the existing program driver, not a parallel CLI**

  `runWaferProgramPipeline` validates a canonical encoding limit profile, creates one bounded transaction-local scratch store, calls `createCanonicalEncodingContext`, then move-passes the result to `ProgramOutputTransaction::create(requestedScope, verifiedProgramOutputSink, programOutputLimits, canonicalEncodingContext)`; completion scope, verified sink, output limits and encoding context are immutable owners for that transaction. Whole-variant seals and attaches the committed executable without publishing. The driver resolves named configuration into a canonical target-requirement map of Task 5 `VerifiedTargetCompilationContext`s plus validated build/admission limits, then calls `buildAndAttachTargetArtifactSet` for every accepted target variant against the same outer transaction. Every record factory receives a non-const ref to the retained context through transaction-owned compiler APIs; no stage constructs scratch/default limits. Each attachment carries `TargetVariantId`, `TargetArtifactSetId`, private root capability, external delivery `ContentDigest`, exact delivery size and locator-free verified record. A locator alone is invalid. Limit profiles are service configuration, not program IR or identity, and cannot change partition/member selection. The driver does not pass entry/rank IDs as strings and does not invoke replay pass pipelines.

  Only after every target required by the creation-time completion scope attaches does the driver call parameterless `ProgramOutputTransaction::commit()`. The transaction rejects missing/extra scope attachments and any prior child failure, seals the typed delivery index and performs exactly one outer atomic visibility transition: rename one unified output root, or create/CAS one deterministic `schema/wafer/program_delivery.proto::ProgramDeliveryCommitRecord` that binds the executable and all immutable target roots. Success returns only `PublishedProgramDelivery`. A full package driver creates the transaction with `ProgramDeliveryCompletionScope::PackageBundle` and keeps these same roots private until package attachment succeeds; it cannot first commit target scope. Crash/CAS failure before this point yields no trusted reference; a retry may deduplicate immutable bytes but cannot discover an orphan as accepted.

  Keep `wafer-lower-groups-to-target-llvm` as replay-only coverage. Do not create a production `--emit-kad`, `--emit-note` or manual pass list interface.

- [ ] **Step 3: Configure an explicit target-toolchain test feature**

  CMake verifies clang, TX8 GCC/objcopy/readelf/nm, headers and required libraries before adding `tx8-device-toolchain` to lit features. The vertical test may use `REQUIRES: tx8-device-toolchain`, but completion requires the configured build used for acceptance to show the test as executed, not unsupported. Print the resolved tool versions into test diagnostics and bind them into `TargetBuildProfile`.

- [ ] **Step 4: Prove failure atomicity at driver level**

  Task 13's injected filesystem/device-link/limit matrix proves failure after earlier members staged. At driver level, use two targets and fail the second after the first is fully attached; also fail package attachment, crash immediately before outer commit, and inject commit-record CAS failure/multi-directory mode. In every case no new program/package alias/index or trusted target ref appears, although immutable orphan roots may await GC. Run the invalid geometry and build-budget `limit+1` cases under the same output parent and prove the previous successful committed root stays valid. A failed/over-budget target must never be omitted to let the rest publish.

- [ ] **Step 5: Run the actual vertical gate**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt wafer-artifact-inspect -- -j128
  <configured-lit> -sv --show-unsupported \
    build/wafer-dev/test/Tools/wafer-opt-target-artifact-set.test
  ```

  Expected: the test is supported and passes; every subprocess runs; the final trusted delivery reader reports every required target plus complete shape/member/partition coverage; all injected failures leave no new trusted root/ref.

- [ ] **Step 6: Commit**

  ```bash
  git add include/Wafer/Pipelines/Pipelines.h lib/Wafer/Pipelines/Pipelines.cpp \
    lib/Wafer/Pipelines/CMakeLists.txt \
    tools/wafer-opt \
    test/lit.site.cfg.py.in test/lit.cfg.py \
    test/Tools/wafer-opt-target-artifact-set.test \
    test/Tools/Inputs/target-artifact-program \
    test/Tools/Inputs/target-artifact-program-invalid
  git commit -m "Publish target artifact sets from the named driver"
  ```

### Task 15: Completion Audit, Queue Sync, and Full Verification

**Files:**
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`
- Modify: `memory/bugs.md` only when this implementation discovers a reusable failure pattern

**Consumes:** all artifacts and fresh verification results from Tasks 1-14.

**Produces:** synchronized task status, stable build/debug workflow and an auditable completion record. This task changes no IR/ABI contract.

- [ ] **Step 1: Run contract consistency searches**

  ```bash
  rg -n "deterministic Protobuf.*canonical|print.*LLVM.*ABI|ResourceId.*KernelAbi|wafer_tx81_dte_(send|recv|wait)" \
    include lib runtime tools test tasks
  rg -n "TargetArtifactSet|note.wafer.abi|computeSemanticDigest|computeContentDigest|encodeIdentityMessage" \
    include lib tools test
  rg -n "quantStorageAbi|fixedArguments|clangPath|DeviceLinkConfiguration|full stable ASCII symbol path|one note per expected entry" \
    include lib tools test tasks/plans
  rg -n "TargetArtifactSetDeliveryRoot" schema include lib tasks/plans
  rg -n "loadAndVerifyTargetArtifactSet\\(|diagnosticLocator|moduleLocator|rootLocator|acquireReadLease\\(\\) const|ConvertedTargetEntryView" \
    include lib tools test tasks/plans
  python3 tools/check_ir_organization.py --root .
  python3 tools/check_deps.py
  python3 tools/check_target_dependency_layers.py
  ```

  Inspect every hit. Every KAD reference must name its `descriptor semantic digest`; no raw Protobuf semantic ID, LLVM-text ABI recovery, KAD `ResourceId`, tile-local scratch slot, singular target-wide quant profile, source-private symbol spelling, caller tool flag/path, incomplete DTE symbol or duplicate schema option/symbol registry may remain. `TargetArtifactSetDeliveryRoot` may appear only in delivery loader/publisher code, never as a PackageManifest semantic field.

- [ ] **Step 2: Run all fresh verification**

  ```bash
  cmake --build build/wafer-dev --target check-wafer -- -j128
  ctest --test-dir build/wafer-dev --output-on-failure
  <configured-lit> -sv --show-unsupported build/wafer-dev/test
  git diff --check
  ```

  Explicitly inspect the unsupported list. WCRE/KAD/schema, actual device link, ELF note, atomic set and named-driver target artifact tests must have executed. `ctest passed` with any of these skipped is not completion.

- [ ] **Step 3: Replay the publication failure matrix**

  Run the multi-shape/multi-profile/multi-module success once and injected failures at closure classification/clone conversion, profile-union recomputation, toolchain reattestation, every build/admission/canonical/budget limit, conversion, KAD, core/clone object emission/measurement, exact-limit packing, link, note readback, final ELF verifier, content digest, private member write/fsync, target-set attachment and final outer commit. After every precommit failure, enumerate the trusted program-delivery namespace and prove no new commit/reference is reachable. Resolve every required target ref from the successful program delivery; create one Artifact-metadata registry, issue bounded child capabilities and move each into a metadata session to load the complete canonical metadata vector, compare all typed IDs/digests/coverage and prove zero module opens. Create one service-bound runtime artifact session; the runtime-private bundle owner proves all-required/no-extra and batch-move-consumes metadata plus that session without invocation reservation. Then `RuntimeSession::create` joins selected plan and invocation capacity; only its access opens/verifies modules from at least two heterogeneous sets. Cross-owner/missing/extra bind, failure on the last set, second bind, RuntimeSession creation failure, cross-session/invocation source use and any semantic child/shared-host budget exhaustion must fail before module publication.

- [ ] **Step 4: Synchronize progress and stable memory**

  Update `tasks/progress.md` only after all gates pass. Record exact configure/build/test commands and artifact inspector entry points in `memory/general_dev.md`. Add to `memory/bugs.md` only concrete issue/root-cause/prevention patterns observed during implementation; do not copy task status or ABI fields there.

- [ ] **Step 5: Commit the completion record**

  ```bash
  git add tasks/progress.md memory/general_dev.md
  git add memory/bugs.md
  git commit -m "Record target artifact set completion gates"
  ```

## Completion Evidence

The plan is complete only when all of the following are true in one fresh configured build:

- Protobuf `v21.9` tag object and peeled commit match exactly; semantic identity, KAD, target-set, PackageManifest and state-migration codegen share `WaferProtoSupport`.
- WCRE tests cover every value tag, endian width, field order, bounded external SET sort and all twenty-one fixed domains including record-20 quant profiles and record-21 state migration, plus unknown-field rejection, cross-process golden bytes and 100K/1M peak-memory gates.
- Static function identity is invariant to excluded presentation/private-symbol names, changes with private helper/global semantics, and uses one closure resolver that separates non-clonable connectivity from proved-pure helper/immutable-global clone dependencies; thousand-entry shared helpers stay partitionable and equal clone objects dedup deterministically.
- Global build profile exposes a verified multi-profile registry; every prelink unit retains its exact actual possibly-empty used-profile evidence, compatible units may pack across differing sets, and each closed final module alone computes the canonical union, restricted profile and fingerprint without unused-profile pollution.
- Compiler/linker/objcopy/sysroot/CRT bytes, canonical arguments, allowed environment, dependency classification and `hidden_linkonce_odr_comdat_v1` are attested by one non-forgeable toolchain profile; every invocation runs in a bounded cancellable process group and public target emission accepts no arbitrary flags or binary side channel.
- The complete typed target-context set exists only as one `VerifiedTargetCompilationContextRegistry` owner transferred from `ExecutableCompilationInput` through the committed executable lower attachment; target build resolves by typed ID through private access after prior owners die, with no replacement registry/vector or token-only recovery.
- KAD is generated with target LLVM in one transaction, records only launch-visible resource slots, is verified by one C++ semantic verifier, and is identical in compiler object, unique-digest ELF note and downstream typed reference. Multiple entries may share one equal KAD/note.
- Accepted Direct DTE facts, not logical peer or names, drive fixed target calls; success, timeout, transport error and peer failure remain distinct.
- Final ELF passes machine/ISA/MABI/attributes/export/undefined/note plus clone COMDAT/visibility/one-definition checks before raw-byte module content digest is computed; only full typed entry symbols are public.
- Environment fingerprint, artifact fingerprint and module content digest remain distinct typed facts and all match each set member.
- One target variant's set covers every committed shape variant that references it; deterministic core/clone partitioning and exact-limit packing produce one complete canonical member map and private attachment, while any unit/member/build-budget failure permanently prevents the single outer program-delivery commit.
- KAD/note/ELF/set readers apply explicit validated admission limits and a bounded canonical encoding context before proportional allocation; changing sufficient legal limits never changes identity, and an over-limit delivery exposes no proof or blob handle.
- Build-stage refs expose no open/read operation and every readback uses a live-session `TargetBuildReadLease`; metadata and runtime artifact sessions use nonconvertible typed child quotas while atomically charging the same lower host FD/reader/worker/bytes/workspace ledger, with runtime module work additionally charging service and invocation capacity.
- The locator-free `TargetArtifactSetVerifiedRecord` is the package-reusable boundary; standalone and PackageManifest-embedded proofs retain one immutable protobuf owner/path plus shared canonical backing without copying full record bytes. Physical locators exist only in `TargetArtifactSetDeliveryRoot`, whose external digest/size come from `ProgramOutputTransaction`.
- `wafer-opt --program-pipeline=stablehlo-to-executable` executes the actual positive chain. Standalone consumers call `loadAndVerifyTargetArtifactSetMetadata(const TrustedTargetArtifactSetDeliveryRef &, ArtifactMetadataVerificationSession &)` for every required target without opening modules; the runtime-private bundle owner then atomically one-way batch-binds the complete all-required/no-extra metadata vector to one independent `RuntimeArtifactVerificationSession` before any `BoundTargetModuleSource` exists or selected ELF opens. Package code analogously batch-binds embedded module views/`BoundBlobSource` and never calls the standalone metadata loader. No raw path/digest/size, replacement limits/context/capability, public/single-set binder or default/unbounded session overload exists.
