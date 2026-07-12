# Whole-Variant Executable Implementation Plan

> **执行约束：** 任务状态和直接前置以 `tasks/progress.md` 为准；本文 checkbox 只拆解实现步骤，不是独立状态源。

**Goal:** 将所有 distributed execution instances 的完整 traversal、layout、instruction、SPM/DDR、completion、transport 和 launch projection 在同一个候选事务中闭合，并只在全 variant 验证通过后原子生成 committed `wafer.executable`。

**Architecture:** production compiler在源module和candidate-private artifact store组成的whole-variant transaction上枚举
bounded candidates；每个候选对全部rank entries物化完整traversal，依次完成layout、low-precision path、instruction、
completion/liveness、SPM、multi-arena DDR/stream windows、immutable artifact bytes、transport和projection。candidate
resource/state-group与rank/entry同生命周期；ResourceView验证同一records，composer一次发布committed IR、ArtifactRef
index和blobs。任何失败销毁clone与private payload staging，源module/output store保持不变。

**Tech Stack:** C++17、MLIR/LLVM 20、ODS/TableGen、StableHLO/Shardy、Wafer model/distributed/executable dialect、SCF/Async/MemoryEffect interfaces、WCRE V1 + SHA-256、lit/GTest。

## Global Constraints

- 设计 owner 是 `tasks/01-architecture.md`、`tasks/04-topology-execution-mesh.md`、`tasks/06-group.md`、`tasks/07-tile-region.md`、`tasks/08-layout-materialization.md`、`tasks/09-spm-memory-planning.md`、`tasks/10-compute-movement.md`、`tasks/11-instruction-ir.md`、`tasks/12-ddr-memory-planning.md`、`tasks/13-communication.md`、`tasks/15-launch-runtime-package.md` 和 `tasks/16-verification-plan.md`。本计划只拆实施，不新增这些文档之外的 IR/ABI 语义。
- 前置计划必须已经提供typed model program/`ModelEntrypointId`/state groups/invocation fields/IR quant semantics、target low-precision
  capabilities、distributed IR、candidate executable rank/entry/resource/state-group、canonical ExecutionInstanceId及
  shared geometry proof。candidate group/resource已拥有stable IDs、snapshot/residency/arena/scope proposal。
- `CandidateExecutionEntry` 不是 op。candidate rank/entry/resource/transport/projection 只存在 `CommitState::Candidate` 的 whole-variant clone；target、artifact、package 和 runtime 必须拒绝 candidate state。
- `DirectFullShape` 是普通的第一个 traversal policy，和 tiled candidates 运行完全相同的 layout、instruction、completion、SPM、DDR、transport、projection、target preflight 和 commit gates。
- representative first/tail tile 只能 cheap-reject；passing 必须来自 full traversal coverage、全部 outputs/reductions 和全部 rank entries 的真实 materialization。
- `LayoutVariable`、`LayoutEdge`、candidate frontier、completion/liveness dataflow、`ExecutableResourceView` 和 rejected diagnostics 都是 transformation-local analysis，不进入 IR、artifact 或 package。
- SPM/DDR planner只materialize accepted offsets/windows。逻辑resource role/alias/update、group-level state policy和typed
  target/shape/execution-instance/projection-covered realizations的跨阶段owner是同一个candidate-to-committed
  `wafer.executable.resource/state_group` record；entry只持有有序
  `SlotId -> (ResourceId, StateSlotVersionRole)` binding。planner和commit都不得
  新造policy或`ResourceId`，且entry-internal SPM temp/psum/staging不因普通alloc获得runtime resource identity。
- `wafer.executable.transport` 是 accepted physical transport 的唯一 candidate IR owner；它不复制 p2p algorithm body，也不拥有 final projection mode、`ProjectionSetId` 或 projection digest。
- launch projection 是 pinned/relocatable union、`ProjectionSetId` 和 digest 的唯一 owner。relocatable 只允许 `ConcreteRecordSet` 或能在 compile time 完全展开并逐项验证的 `FiniteTemplateSet.allowed_bindings`；runtime 没有 route/resource 搜索自由度。
- final `RankClassId` 只能细分一个 distributed prerequisite class；不得跨 prerequisite class 合并，也不得改变 `ExecutionInstanceId`。class内local program、ABI、ordered slot resource/state role、shard/content/artifact/storage/capacity/arena/scope、memory、transport和completion必须逐项等价；不等价就拆分，static code仍可跨class复用。
- model user API只用typed `ModelEntrypointId`；candidate/committed `wafer.executable.invocation`把它映射到variant已有entry/completion roots和terminal contract。static `EntryId`、function/module symbol和diagnostic alias不能充当用户调用ID。
- IR层只用registered quant/float8 types和Wafer ODS block-scale/storage attrs；runtime-safe `abi::QuantizationDescriptor`/`abi::StorageEncodingDescriptor`由前置`WaferABI` schema/value owner及唯一compiler adapter提供。本计划不得创建同名WaferIR C++ struct。
- semantic IDs 使用 `tasks/14-target-llvm-golden-packet.md` 的 WCRE V1、domain-separated SHA-256 和共享 canonical encoding library。禁止用 MLIR 文本、symbol 名、pointer、遍历时地址或 raw Protobuf bytes 生成 identity。
- Before Task 1, execute the package/runtime plan's `Outer Program Delivery Foundation`: it uniquely creates
  `ProgramOutputTransaction`, bound scope/sink/limits/CanonicalEncodingSession, staging tokens/views and program-delivery
  schema/reader. This plan only adds executable-specific attachments and never creates a second transaction/delivery owner.
- Production whole-stage ownership starts with one move-only `ExecutableCompilationInput`, not a borrowed `ModuleOp`. It owns
  the frontend-created `MLIRContext`, module, `VerifiedProgramSource` coordinator, all limits/canonical owner and one canonical
  `VerifiedTargetCompilationContextRegistry`; the same registry moves through the winner, `CommittedExecutableProgram` and
  lower executable commitment. No commit/attach API accepts a replacement context vector.
- segmented communication 首个 production correctness path 固定为 `padded_fixed_capacity`。`bounded_variable_segments` 在 target ABI 证据落地前必须结构化拒绝，不能私自增加 runtime/CRT 字段。
- pass、pipeline、op、type、attr、C++ type、artifact 和 diagnostic 不得含 `Q0.3`、`Q0.4`、task/stage 编号或其它路线图编号。
- 每个任务先增加失败测试并确认失败，再写实现。每项完成一个独立 commit；不得把后续 task 的 scaffold 或 placeholder 混入当前提交。

```text
Pipeline position:
- Upstream artifact / IR:
  verified model/resources、target environment/topology/mesh、distributed components/instances/prerequisite
  classes，以及 `CommitState::Candidate` 的 executable rank/entry/resource identity and policy records。
- Current stage responsibility:
  在一个 clone 内为完整 variant 枚举并验证 traversal/layout/instruction/completion/SPM/DDR/transport/
  projection proposals，重算 executable resource view，完成 target ABI preflight，并原子 commit 全部 ranks。
- Output artifact / IR:
  `CommitState::Committed` 的 `wafer.executable`：完整 variant/rank/entry/resource/transport/projection/
  completion records 和 static rank functions；无 logical group、candidate state、pending completion、
  unbound transport 或未绑定 slot。
- Downstream consumer:
  structure-preserving target LLVM、Kernel ABI/TargetArtifactSet、PackageManifest 和 RuntimeSession。
- User-level driver / named pipeline:
  `wafer-opt --program-pipeline=stablehlo-to-executable` selects the direct
  `runStablehloToExecutableCompilation` driver. IR-only named pipelines are internal transform/debug components and cannot
  own the source, output transaction, payload materialization or commit.
- Explicit non-goals:
  不生成 Kernel ABI/ELF/PackageManifest，不分配 runtime handle/physical address，不让 runtime 重选
  layout/memory/transport，不保存 rejected candidate，不把单 group/pass 结果当作 committed executable。
- Completion gate:
  至少两个 canonical execution instances 经过 full traversal 和所有 whole-entry/whole-variant gates；
  注入一个 rank-specific failure 时源 module byte-for-byte 不变；成功结果通过 executable verifier，
  且 production driver 不存在 direct bypass。
```

The final coordinator order is fixed by artifact dependency, not by pass numbering:

```text
clone candidate executable set
  -> for each candidate variant, enumerate one global traversal policy and materialize every entry
  -> verify full iteration/output/reduction coverage
  -> materialize tensor/tile communication, including segmented count/data phases
  -> enumerate one whole-entry layout alternative and apply it on a fresh clone
  -> select verified low-precision native/composite paths and storage/profile refs
  -> lower all target-abstract tile ops to instruction IR
  -> verify shared physical geometry and target narrowing
  -> compute completion/liveness for every entry
  -> plan whole-entry SPM for every entry
  -> plan multi-arena DDR, streamed windows/staging and copy/reuse edges
  -> build exact no-I/O immutable materialization plans; keep streamed_planned
  -> materialize typed completion records using provisional window refs
  -> accept physical transport and bind completion/status/error refs
  -> materialize one finite launch projection set
  -> verify candidate resources/state/artifact plans through ExecutableResourceView
  -> rerun every whole-variant verifier and target preflight
  -> rank complete passing candidates, then stream-materialize bytes for the selected candidate only
  -> form final rank classes and final realization/window IDs
  -> finalize ArtifactRefs/streamed_windows, rebuild ResourceView, seal identity and atomically publish IR + blobs
```

No local stage result, callable or fragment may mark a candidate passing, publish IR to the source module, or cache an
identity before the fixed coordinator has rerun every later gate in this sequence.

## Implementation Dependency Order

The numbered sections are stable review labels, not a license to implement them in numeric order. Execute and commit them in
this dependency order:

```text
target shared identity/schema/context foundation
  -> package/runtime Outer Program Delivery Foundation
  -> typed model/program/distributed/target-environment foundation
  -> target correctness shared geometry core (formal conversion remains post-commit)
  -> Task 1 transaction ownership
  -> Task 2 traversal materialization
  -> Task 3A policy/limits/constraint-solver foundation
  -> Task 9 segmented logical/buffer lowering
  -> Task 4 layout
  -> Task 4A low precision
  -> Task 5 completion/liveness analysis
  -> Task 7 SPM
  -> Task 8 DDR/windows
  -> Task 8A no-I/O plan + selected-byte materializer APIs
  -> Task 6 completion/invocation/activation records
  -> Task 10 transport
  -> Task 11 projection
  -> Task 12 resource view
  -> Task 3B full coordinator integration and selected-candidate byte materialization
  -> Task 13 final verifier/seal/attach
  -> Task 14 production driver/outer executable-only commit
```

Task 8A's unit-tested byte API is deliberately not called by production selection until Task 3B. Task 3B is the first point
where one assignment can traverse every mandatory stage; Task 13 is the first point where it can become committed IR. This
ordering forbids temporary completion-without-windows, segmented-after-layout rewrites and payload reads for structurally
rejected candidates.
The temporary target fail-closed hotfix may land in parallel, but it does not satisfy the shared geometry prerequisite above.
Committed conversion/KAD integration intentionally follows Task 13 and is not required to create a candidate request here.

---

### Task 1: Whole-Variant Candidate Transaction

**Files:**
- Create: `include/Wafer/Compiler/ExecutableCompilationInput.h`
- Create: `lib/Wafer/Compiler/ExecutableCompilationInput.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Create: `include/Wafer/Transforms/Executable/WholeVariantPlanning.h`
- Create: `lib/Wafer/Transforms/Executable/WholeVariantPlanning.cpp`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `unittests/Transforms/WholeVariantPlanningTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: move-owned `ExecutableCompilationInput` containing the verified frontend context/module/source coordinator,
  target-materialization plan, target-context registry and typed model/policy owners, plus the same-owner outer transaction.
  Executable/candidate records do not need to exist in the immutable source; later formation creates them only in the clone.
- Produces: clone-local executable-set transaction that retains the complete input owner; failure destroys working clones/
  child staging and does not mutate the owned source or visible output.

```cpp
namespace wafer::compiler {
class ExecutableCompilationInput; // move-only, non-aggregate, no public ModuleOp/source accessor
}

namespace wafer {
class ExecutableCandidateTransaction; // owns module clone + candidate-private artifact staging

mlir::FailureOr<ExecutableCandidateTransaction>
beginExecutableCandidateTransaction(
    compiler::ExecutableCompilationInput input,
    compiler::ProgramOutputTransaction &output);
} // namespace wafer
```

- [ ] **Step 1: Add transaction failure tests**

  Add GTests for: invalid/stale frontend owner generation, invalid model handoff, mixed already-committed executable state in the
  source, verifier failure after mutating the clone and after writing a private payload, and successful isolation. Absence of an
  executable root in the verified frontend source is positive. A test-only factory may move-own an in-memory module but cannot
  materialize payload or attach/commit. Destroy the
  factory/request frame after forming the input and prove clone/source-coordinator lifetime remains valid. Save source generic
  IR plus visible output-store index before every call and compare both byte-for-byte on failure. Compile-time checks reject a
  raw-`ModuleOp` production overload.

- [ ] **Step 2: Run the focused test and confirm it fails**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  ```

  Expected before implementation: `WholeVariantPlanningTest` does not compile because the transaction API is absent.

- [ ] **Step 3: Implement clone ownership and symbol rebinding**

  Private construction moves the complete input into the transaction, rechecks frontend/model/source owner generations, clones
  the complete source module before any mutation, and opens a child content-addressed staging transaction under
  `ProgramOutputTransaction`. It rejects pre-existing committed executable output but does not require a candidate root yet. The
  owner retains `VerifiedProgramSource` and target-context registry across
  every retry; rejected assignments replace only the working clone/child staging. It has private scoped accessors and destroys
  working IR/staged blobs on failure. Never retain handles into the source module or publish a blob before final commit.

- [ ] **Step 4: Add a mutation guard used by later tasks**

  Implement a test-only `printGenericModule` helper and a `verifySourceUnchanged` assertion. The production API exposes only the owning clone; it does not expose a partial `apply()` method.

- [ ] **Step 5: Rebuild, run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add include/Wafer/Compiler/ExecutableCompilationInput.h \
    lib/Wafer/Compiler/ExecutableCompilationInput.cpp lib/Wafer/Compiler/CMakeLists.txt \
    include/Wafer/Transforms/Executable/WholeVariantPlanning.h \
    lib/Wafer/Transforms/Executable/WholeVariantPlanning.cpp \
    lib/Wafer/Transforms/CMakeLists.txt \
    unittests/Transforms/WholeVariantPlanningTest.cpp unittests/CMakeLists.txt
  git commit -m "Add whole-variant candidate transactions"
  ```

### Task 2: Complete Entry Traversal Materialization

**Files:**
- Create: `include/Wafer/Analysis/TraversalCoverageAnalysis.h`
- Create: `lib/Wafer/Analysis/TraversalCoverageAnalysis.cpp`
- Create: `lib/Wafer/Transforms/Executable/MaterializeEntryTraversals.cpp`
- Modify: `include/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h`
- Modify: `lib/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.cpp`
- Modify: `lib/Wafer/Analysis/CMakeLists.txt`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Modify: `include/Wafer/Transforms/Passes.td`
- Create: `test/Transforms/materialize-entry-traversals.mlir`
- Create: `test/Transforms/materialize-entry-traversals-failure.mlir`

**Interfaces:**
- Consumes: one candidate `ExecutableEntryOp`, its `CandidateExecutionContext`, all logical groups in its static/rank-parametric function, and a transformation-local traversal policy.
- Produces: the same candidate entry with every group replaced by structured full traversal containing tile-local regions; `TraversalCoverage` remains analysis-only.

```cpp
enum class TraversalPolicy { DirectFullShape, Tiled };

struct TraversalCandidate {
  TraversalPolicy policy;
  llvm::SmallVector<int64_t> tileSizes;
  llvm::SmallVector<int64_t> reductionSplitSizes;
};

struct TraversalCoverage {
  bool coversIterationDomain;
  bool coversAllResults;
  bool coversReductionDomain;
};

mlir::LogicalResult materializeEntryTraversals(
    ExecutableEntryOp entry, DistributedInstanceOp instance,
    const TraversalCandidate &candidate);

mlir::FailureOr<TraversalCoverage>
analyzeTraversalCoverage(ExecutableEntryOp entry);
```

- [ ] **Step 1: Write the failing traversal matrix**

  Positive split-input cases: direct full-shape; non-divisible tail; two outputs with different result domains; reduction split with loop-carried accumulator; nested `scf.if`; two logical groups in one entry. Negative cases: first/tail representative only, uncovered output slice, reduction gap/overlap, tile size zero, unsupported `scf.while`, and one group left in an otherwise lowered entry.

  The positive tail must contain an actual bounded loop shape:

  ```mlir
  scf.for %i = %c0 to %c10 step %c4 {
    %remaining = arith.subi %c10, %i
    %size = arith.minsi %remaining, %c4
    // The builder materializes one tile-region scope for [%i, %size).
    scf.yield
  }
  ```

- [ ] **Step 2: Demonstrate the representative-only bug**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/materialize-entry-traversals(-failure)?\.mlir'
  ```

  Expected before implementation: tests fail because the current selector validates representative tiles and commits one group/module result instead of materializing complete entry traversal.

- [ ] **Step 3: Extract reusable group-to-tile-region construction**

  Change the conversion builder to accept `CandidateExecutionContext` and explicit iteration offsets/sizes. It must derive rank identity from `ExecutableEntryOp -> DistributedInstanceOp`, never from a pass option or zero default. A rank-parametric function may stay shared only when traversal and lowered body use its typed partition/replica SSA/ABI. If a supported lowering requires a static peer/coordinate, specialize the function inside the candidate clone and update typed entry coverage before rewriting; never mutate one shared function differently for two instances.

- [ ] **Step 4: Materialize full domains**

  `DirectFullShape` emits one iteration only when the full domain is statically covered. `Tiled` emits SCF loops and tail sizes from structured iteration domains/indexing maps. Multi-output coverage is checked per result; reduction splits require exact, non-overlapping coverage and explicit accumulator carry.

- [ ] **Step 5: Verify coverage from resulting IR**

  `TraversalCoverageAnalysis` reads loop bounds, `tensor.extract_slice`/`insert_slice`, tile-region iteration values, yields and result writes. It must not trust `TraversalCandidate` or a `coverage=true` attr. Passing requires no residual `wafer.group` in the entry.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/materialize-entry-traversals(-failure)?\.mlir|Transforms/convert-group-to-tile-region\.mlir|Transforms/select-group-tile-(multi-output|reduction-split|tiled-control-flow-gap)\.mlir'
  git add include/Wafer/Analysis/TraversalCoverageAnalysis.h \
    lib/Wafer/Analysis/TraversalCoverageAnalysis.cpp \
    lib/Wafer/Transforms/Executable/MaterializeEntryTraversals.cpp \
    include/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h \
    lib/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.cpp \
    lib/Wafer/Analysis/CMakeLists.txt lib/Wafer/Transforms/CMakeLists.txt \
    include/Wafer/Transforms/Passes.td test/Transforms/materialize-entry-traversals*.mlir
  git commit -m "Materialize complete executable entry traversals"
  ```

### Task 3A: Candidate Policy and Constraint Solver Foundation

**Files:**
- Create: `include/Wafer/Planning/CandidateSearchLimits.h`
- Create: `lib/Wafer/Planning/CandidateSearchLimits.cpp`
- Create: `include/Wafer/Planning/CandidateConstraintSolver.h`
- Create: `lib/Wafer/Planning/CandidateConstraintSolver.cpp`
- Create: `include/Wafer/Planning/ExecutionSchedulePolicy.h`
- Create: `lib/Wafer/Planning/ExecutionSchedulePolicy.cpp`
- Modify: `lib/Wafer/Transforms/Group/SelectGroupTile.cpp`
- Create: `lib/Wafer/Planning/CMakeLists.txt`
- Modify: `lib/Wafer/CMakeLists.txt`
- Create: `unittests/Planning/CandidateConstraintSolverTest.cpp`
- Create: `unittests/Planning/ExecutionSchedulePolicyTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: verified model program projection plus typed local traversal/layout/split alternatives and compatibility summaries.
- Produces: validated bounded schedule policy, candidate fragments, canonical complete-assignment cursor and structured budget
  diagnostics. It does not clone executable IR, call stage transforms, read payload bytes or authorize commit.

```cpp
enum class TileSearchMode { FirstLegal, MinEstimatedTime };

struct CandidateSearchLimitValues {
  uint64_t maxGlobalCandidates;
  uint64_t maxCandidatesPerVariant;
  uint64_t maxTraversalAlternativesPerEntry;
  uint64_t maxLayoutAlternativesPerEntry;
  uint64_t maxInternalSplitAlternativesPerEntry;
  uint64_t maxSimultaneousClones;
  uint64_t maxEstimatedLiveCloneBytes;
  uint64_t maxWorkers;
  uint64_t maxRetainedDiagnostics;
  uint64_t maxIterationDomainsPerVariant;
  uint64_t maxCountExprNodes;
  uint64_t maxCountExprDepth;
  uint64_t maxExpandedTemplateNodes;
  uint64_t maxFragments, maxFragmentBytes, maxConstraintEdges;
  uint64_t maxSolverStates, maxCompatibleJoinOperations;
  uint64_t maxNogoods, maxNogoodBytes, maxFullAssignments;
  uint64_t maxArtifactMaterializationAttempts;
  uint64_t maxArtifactReadBytes, maxArtifactWriteBytes;
};

class CandidateSearchLimits final {
public:
  CandidateSearchLimits() = delete;
  static llvm::Expected<CandidateSearchLimits>
  create(const CandidateSearchLimitValues &values);
  // Read-only accessors for every positive checked field.
private:
  explicit CandidateSearchLimits(const CandidateSearchLimitValues &values);
};

enum class IterationDomainKind { Microbatch, ExpertWave, BoundedRepetition };
using IterationDomainOwner =
    std::variant<VariantWideDomainOwner, abi::ModelEntrypointId,
                 abi::ModelProgramMemberId>;

enum class BoundedCountExprKind {
  Constant, Dimension, InvocationField, Add, Multiply, CeilDiv, Minimum, Maximum
};

class BoundedCountExpr final {
public:
  static llvm::Expected<BoundedCountExpr> constant(uint64_t value);
  static llvm::Expected<BoundedCountExpr> dimension(abi::DimId id);
  static llvm::Expected<BoundedCountExpr> invocationField(
      abi::InvocationPolicyFieldId id);
  static llvm::Expected<BoundedCountExpr> binary(
      BoundedCountExprKind kind, BoundedCountExpr lhs,
      BoundedCountExpr rhs, const CandidateSearchLimits &limits);
private:
  BoundedCountExpr() = default;
};

struct IterationDomainPolicyInput {
  IterationDomainKind kind;
  IterationDomainOwner owner; // frontend-stable API/member owner; never a bare ComponentId
  BoundedCountExpr expression;
  uint32_t maxCount;
};

class ExecutionSchedulePolicy final {
public:
  static llvm::Expected<ExecutionSchedulePolicy>
  create(llvm::ArrayRef<IterationDomainPolicyInput> domains,
         const CandidateSearchLimits &limits,
         const VerifiedModelInterface &model);
  llvm::ArrayRef<IterationDomainPolicy> canonicalDomains() const;
  // Convenience builders create constant/DimId/InvocationPolicyFieldId and
  // checked add/mul/ceildiv/min/max nodes before create() validates the AST.
private:
  explicit ExecutionSchedulePolicy(
      llvm::SmallVector<IterationDomainPolicy> canonicalDomains);
};

struct ExecutablePlanningOptions {
  TileSearchMode tileSearchMode;
  TileSearchEffort tileSearchEffort;
  CandidateSearchLimits searchLimits;
  ExecutionSchedulePolicy schedulePolicy;
};

class CandidateFragment final {
public:
  CandidateFragment() = delete;
  CandidateFragment(const CandidateFragment &) = delete;
  CandidateFragment(CandidateFragment &&) = default;
private:
  class Impl; // closed typed decision literals and compatibility/demand summaries only
  explicit CandidateFragment(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class CandidateAssignment final {
public:
  CandidateAssignment() = delete;
  CandidateAssignment(CandidateAssignment &&) = default;
private:
  friend class CandidateAssignmentCursor;
  friend class detail::CandidateAssignmentReplayAccess;
  class Arena;
  CandidateAssignment(std::shared_ptr<const Arena> arena,
                      llvm::SmallVector<uint32_t> canonicalOrdinals);
  std::shared_ptr<const Arena> arena_;
  llvm::SmallVector<uint32_t> canonicalOrdinals_;
};

class CandidateAssignmentCursor final {
public:
  llvm::Expected<std::optional<CandidateAssignment>> next();
private:
  class Impl;
  explicit CandidateAssignmentCursor(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

mlir::FailureOr<llvm::SmallVector<CandidateFragment>>
buildCandidateFragments(ExecutableOp candidate,
                        const ExecutablePlanningOptions &options);

mlir::FailureOr<CandidateAssignmentCursor> solveCandidateConstraints(
    llvm::SmallVector<CandidateFragment> fragments,
    const CandidateSearchLimits &limits);
```

`solveCandidateConstraints` move-consumes the fragments into one bounded immutable arena. The cursor and every returned
assignment share that arena and use canonical ordinals; no raw `CandidateFragment *`, borrowed `ArrayRef`, MLIR handle or callable
escapes. Destroying the caller vector or cursor cannot dangle an assignment, and an ordinal from another arena cannot be
presented as authority.

- [ ] **Step 1: Add policy and limit failures**

  Cover zero/overflow limits, undeclared dimension/invocation refs, cross-owner expressions, u64 intermediate overflow,
  max/actual values above `UINT32_MAX`, duplicate domains and product/depth/node limits. Positive policies cover microbatch,
  expert wave and bounded repetition owned by typed model entrypoints/members.

- [ ] **Step 2: Implement bounded schedule policy values**

  Implement the private-construction factory and validate `CandidateSearchLimits`: positive max global/per-variant candidates,
  per-entry traversal/layout/internal-split alternatives, simultaneous clones, estimated live clone bytes, workers, retained
  diagnostics, iteration-domain count, count-expression nodes/depth and expanded template nodes. Validate a canonical
  `ExecutionSchedulePolicy` collection: each closed-kind domain has a typed owner, checked positive maxCount and AST using only
  constants/declared DimId/InvocationPolicyFieldId plus checked add/multiply/ceildiv/min/max. Reject duplicate/cross-owner
  domains, zero, overflow, undeclared refs, AST/count/product limit excess. Owners are frontend-stable ModelEntrypointId/
  ModelProgramMemberId (or variant-wide); after parallel/distributed formation the driver must revalidate their explicit
  source-member mapping to complete `(DistributedProgramSemanticId, ComponentId)` records before Task 6 binds graph nodes.
  Expression nodes/intermediates use checked u64, but factory proves `maxCount` and every final actual count are in
  `1..UINT32_MAX`; materialized max/actual/iteration wire values are u32 and never truncate.
  A single constant/field domain is only a convenience, not the protocol. This foundation has no MLIR mutation or serialized
  policy sidecar.

- [ ] **Step 3: Move reusable local frontier mechanics out of `SelectGroupTile.cpp`**

  Reuse its shape-driven traversal and reduction refinements to form typed local alternatives without copying planner state
  into IR. Keep `wafer-select-group-tile` as a replay adapter; it cannot commit a group or executable.

- [ ] **Step 4: Implement fragment joins, nogoods and complete-assignment cursor**

  Replace the naive global product with transaction-local typed `CandidateFragment`s per entry/component. A fragment contains
  only closed canonical decision literals plus boundary layout/resource/state/transport/artifact demand summaries; it has no
  `Operation *`, `Value`, `std::function`, function pointer, callback, stage object or mutation authority. It is private-construction
  analysis, never IR/sidecar/cache identity. Build fragments in typed owner order, memoize only
  within this call by exact source/context/local-decision key, propagate equality/capacity/conflict constraints, perform
  canonical compatible merge joins and record canonical nogoods/dominance. Checked fragment/edge/state/join/nogood/byte
  limits bound the solver; overflow/resource-token failure returns `search_budget_exhausted`. The cursor visits a canonical
  prefix of complete assignments only and never materializes the naive product. Tests cover 1000 entries with sparse
  alternatives/late conflicts, every fragment/join/solver/nogood/full-assignment limit, bounded backtracking and worker/memo
  1/4 byte-identical assignments/diagnostics. Repo-owned `detail::CandidateAssignmentReplayAccess` is the only interpreter: it
  reads literals by arena ordinal and invokes a fixed, compiled list of stage APIs against the current clone. No fragment or
  summary authorizes legality; Task 3B reruns every assignment through that access.
  Move the input vector into the solver, destroy the original vector and cursor, then replay one retained assignment through
  private test access; its typed decisions must remain valid. Compile-time tests reject copying fragments, raw-pointer
  assignment construction and any solver overload taking `ArrayRef<CandidateFragment>`.

- [ ] **Step 5: Emit stable diagnostics without serializing search state**

  Diagnostic payload may include candidate tile/reduction values, owner stable IDs and failure class. Do not add candidate labels, counts, costs or rejected reasons to `wafer.executable.*`.

- [ ] **Step 6: Run and commit the reusable foundation**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests wafer-opt -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add include/Wafer/Planning/CandidateSearchLimits.h lib/Wafer/Planning/CandidateSearchLimits.cpp \
    include/Wafer/Planning/CandidateConstraintSolver.h lib/Wafer/Planning/CandidateConstraintSolver.cpp \
    include/Wafer/Planning/ExecutionSchedulePolicy.h lib/Wafer/Planning/ExecutionSchedulePolicy.cpp \
    lib/Wafer/Transforms/Group/SelectGroupTile.cpp lib/Wafer/Planning/CMakeLists.txt \
    lib/Wafer/CMakeLists.txt \
    unittests/Planning/CandidateConstraintSolverTest.cpp \
    unittests/Planning/ExecutionSchedulePolicyTest.cpp unittests/CMakeLists.txt
  git commit -m "Add bounded executable candidate solving"
  ```

### Task 4: Whole-Entry Layout Finalization

**Files:**
- Create: `include/Wafer/Analysis/Layout/WholeEntryLayoutAnalysis.h`
- Create: `lib/Wafer/Analysis/Layout/WholeEntryLayoutAnalysis.cpp`
- Create: `lib/Wafer/Transforms/Executable/FinalizeEntryLayouts.cpp`
- Modify: `include/Wafer/Analysis/Group/LayoutPlanningAnalysis.h`
- Modify: `lib/Wafer/Analysis/Group/LayoutPlanningAnalysis.cpp`
- Modify: `include/Wafer/Transforms/Executable/WholeVariantPlanning.h`
- Modify: `lib/Wafer/Transforms/Executable/WholeVariantPlanning.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Analysis/CMakeLists.txt`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Transforms/finalize-entry-layouts.mlir`
- Create: `test/Transforms/finalize-entry-layouts-failure.mlir`

**Interfaces:**
- Consumes: one fully traversed candidate entry, early group layout proposals, target-abstract tile ops, actual memref use-def, target environment and current resource pressure.
- Produces: unique Wafer-tagged memref layouts and explicit `wafer.tile.materialize_layout`/constant storage transforms in the candidate clone; analysis graph is discarded.

```cpp
struct EntryLayoutAssignment {
  mlir::Value value;
  MemLayout layout;
};

struct LayoutMaterializationDemand {
  mlir::Value source;
  mlir::Operation *consumer;
  unsigned operandNumber;
  MemLayout targetLayout;
};

struct EntryLayoutAlternative {
  llvm::SmallVector<EntryLayoutAssignment> assignments;
  llvm::SmallVector<LayoutMaterializationDemand> materializations;
};

struct WholeEntryLayoutResult {
  int64_t materializationCount;
  int64_t addedSPMBytes;
  int64_t addedDDRBytes;
};

mlir::FailureOr<llvm::SmallVector<EntryLayoutAlternative>>
enumerateEntryLayoutAlternatives(ExecutableEntryOp entry,
                                 TargetEnvironmentOp environment);

mlir::FailureOr<WholeEntryLayoutResult> applyEntryLayoutAlternative(
    ExecutableEntryOp entry, const EntryLayoutAlternative &alternative);
```

- [ ] **Step 1: Add cross-group layout tests**

  Positive cases: producer and consumer groups retain compatible `Cx`; a real cut inserts one materialization; two outputs choose independent legal layouts; a compile-time constant uses a verified storage transform. Negative cases: conflicting loop-carried layouts, a no-op materialization, materialization whose full traversal exceeds SPM, and one group-local proposal that is incompatible with a later consumer.

- [ ] **Step 2: Run and confirm the early plan is not enough**

  ```bash
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/finalize-entry-layouts(-failure)?\.mlir|Transforms/dump-group-layout-plan\.mlir'
  ```

- [ ] **Step 3: Reuse local constraint extraction**

  Refactor `LayoutPlanningAnalysis` so op-interface constraint extraction is shared, while group-local proposals remain cheap analysis. `WholeEntryLayoutAnalysis` rebuilds `LayoutVariable`/`LayoutEdge` from all groups, SCF ties, entry boundaries and actual materialized movement.

- [ ] **Step 4: Enumerate bounded assignment alternatives**

  Hard domains come from `WaferLayoutOpInterface`, memref type/layout, boundary contract and `VerifiedInstructionGeometry`. Preferences only order candidates. Every alternative reports its full materialization buffers and DDR movement. Task 3B applies each alternative to its own clone and runs instruction, completion, SPM and DDR gates before it can pass; a memory-rejected alternative never becomes final layout.

- [ ] **Step 5: Rewrite and clean only the clone**

  Write accepted layouts into memref types, insert explicit movement/storage transforms, run layout canonicalization, then rerun the analysis. If cleanup changes storage or lifetime, mark SPM/DDR/completion analyses invalid so later callbacks recompute them.

  Register `wafer-finalize-entry-layouts` only as a replay pass for candidate-entry fixtures. It delegates to the same API and cannot mark a variant committed.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/finalize-entry-layouts(-failure)?\.mlir|Transforms/convert-tile-region-to-instr-layout-materialize\.mlir|Dialect/Wafer/Tile/Layout/'
  git add include/Wafer/Analysis/Layout/WholeEntryLayoutAnalysis.h \
    lib/Wafer/Analysis/Layout/WholeEntryLayoutAnalysis.cpp \
    lib/Wafer/Transforms/Executable/FinalizeEntryLayouts.cpp \
    include/Wafer/Analysis/Group/LayoutPlanningAnalysis.h \
    lib/Wafer/Analysis/Group/LayoutPlanningAnalysis.cpp \
    include/Wafer/Transforms/Executable/WholeVariantPlanning.h \
    lib/Wafer/Transforms/Executable/WholeVariantPlanning.cpp \
    include/Wafer/Transforms/Passes.td \
    lib/Wafer/Analysis/CMakeLists.txt lib/Wafer/Transforms/CMakeLists.txt \
    test/Transforms/finalize-entry-layouts*.mlir
  git commit -m "Finalize layouts across complete executable entries"
  ```

### Task 4A: Low-Precision Candidate Formation

**Files:**
- Create: `include/Wafer/Analysis/LowPrecisionLegality.h`
- Create: `lib/Wafer/Analysis/LowPrecisionLegality.cpp`
- Modify: `include/Wafer/IR/WaferAttrs.td`
- Modify: `include/Wafer/IR/WaferInterfaces.td`
- Modify: `lib/Wafer/IR/WaferInterfaces.cpp`
- Modify: `include/Wafer/IR/WaferOps.td`
- Modify: `include/Wafer/IR/Instr/InstructionOps.td`
- Modify: `lib/Wafer/IR/Instr/InstructionOps.cpp`
- Create: `lib/Wafer/Transforms/Executable/SelectLowPrecisionImplementations.cpp`
- Modify: `lib/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.cpp`
- Modify: `lib/Wafer/Analysis/CMakeLists.txt`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Transforms/select-low-precision-implementation.mlir`
- Create: `test/Transforms/select-low-precision-implementation-failure.mlir`

**Interfaces:**
- Consumes: verified registered MLIR Quant/float8 types plus Wafer block-scale attrs, accepted layout, target
  `LowPrecisionComputeCapabilityV1` and validated `TargetBuildProfile` profile registry from the transaction's exact
  `VerifiedTargetCompilationContext`, plus the shared compiler adapter for runtime-safe ABI descriptor projection.
- Produces: candidate IR `#wafer.storage_encoding` attrs, `QuantStorageAbiProfileId` refs and explicit native/composite tile +
  instruction IR; no command/CRT emission.

- [ ] **Step 1: Add semantic/storage/capability failure matrix**

Cover affine per-tensor/per-axis/per-group and block-scaled descriptors, signed-i8 mathematical/raw zp boundaries, fixed
rounding/saturation, malformed scale/block/tail/packed size, missing profile, and target mode mismatch. TX81 native positive is
limited to the capability-proven signed-i8/INT8-result/scale-disabled profile; f16 requires explicit convert. FP8 positive must
select packed decode to BF16/FP16 before ordinary GEMM.

- [ ] **Step 2: Implement the IR storage attr and matcher**

Keep mathematical quant fields in registered MLIR types/Wafer block-scale attrs and physical bit/byte order, block/scale
packing, alignment/padding and byte-count rules in a versioned closed `#wafer.storage_encoding` ODS attr. Match all fields
against one environment capability and one registry profile; return a non-forgeable result containing
`QuantStorageAbiProfileId`. Use the one `WaferCompilerIdentity` adapter to project verified IR facts to
`abi::QuantizationDescriptor`/`abi::StorageEncodingDescriptor`; do not define a WaferIR C++ descriptor struct. Cost never makes
an unmatched form legal.

- [ ] **Step 3: Add low-precision tile/instruction ops**

Implement and verify `wafer.tile.quantized_gemm`, `wafer.tile.block_scaled_decode`,
`wafer.instr.quantized_gemm` and `wafer.instr.mxfp_decode` from tasks/10-11. Native INT8 records exact q1-first/q0-second
shift and checked zp mapping; rounding/saturation equal profile constants. FP8 decode owns packed/scale/destination/scratch
effects and local completion. Do not reuse plain GEMM/convert flags or legacy helper names.

- [ ] **Step 4: Materialize resources and completion inputs at the right boundary**

Use `declareCandidateResource` only for launch-visible packed/scale/zp/staging or cross-entry scratch roots. Tile-local SPM
decode destination/scale copy/scratch remain instruction values and SPM demands. Emit decode-complete dependency consumed by
Task 5; it must dominate GEMM read and scratch reuse.

- [ ] **Step 5: Run and commit**

```bash
<configured-lit> -sv build/wafer-dev/test \
  --filter='Transforms/select-low-precision-implementation.*\.mlir|Dialect/Wafer/Instr/.*low-precision.*\.mlir'
git add include/Wafer/Analysis/LowPrecisionLegality.h lib/Wafer/Analysis/LowPrecisionLegality.cpp \
  include/Wafer/IR/WaferAttrs.td include/Wafer/IR/WaferInterfaces.td lib/Wafer/IR/WaferInterfaces.cpp \
  include/Wafer/IR/Instr lib/Wafer/IR/Instr lib/Wafer/Transforms/Executable \
  lib/Wafer/Conversion/WaferTileRegionToInstr lib/Wafer/Analysis/CMakeLists.txt \
  lib/Wafer/Transforms/CMakeLists.txt test/Transforms
git commit -m "Form typed low precision executable candidates"
```

### Task 5: Shared Completion and Liveness Analysis

**Files:**
- Create: `include/Wafer/Analysis/Completion/EntryCompletionAnalysis.h`
- Create: `lib/Wafer/Analysis/Completion/EntryCompletionAnalysis.cpp`
- Create: `include/Wafer/Analysis/Memory/EntryLivenessAnalysis.h`
- Create: `lib/Wafer/Analysis/Memory/EntryLivenessAnalysis.cpp`
- Modify: `lib/Wafer/Analysis/CMakeLists.txt`
- Create: `unittests/Analysis/CompletionAndLivenessTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: complete instruction-level candidate entry, including low-precision composite effects, SSA use-def,
  MemoryEffect interfaces, async tokens, SCF control flow, explicit local fences and DTE waits.
- Produces: recomputable completion graph and path-sensitive resource live segments; no IR attrs or side table survive the caller.

```cpp
enum class CompletionDomain {
  HostCommand,
  DeviceLocal,
  DMA,
  DTE,
  Collective,
  StageBarrier
};

struct EntryCompletionInfo;
struct EntryLivenessInfo;

mlir::FailureOr<EntryCompletionInfo>
analyzeEntryCompletion(ExecutableEntryOp entry);

mlir::FailureOr<EntryLivenessInfo>
analyzeEntryLiveness(ExecutableEntryOp entry,
                     const EntryCompletionInfo &completion);
```

- [ ] **Step 1: Add semantic unit tests**

  Parse entries containing straight-line DMA, `scf.if`, `scf.for` with loop-carried memref/token, local compute fence, DTE send/recv/wait, collective wait and function return. Assert:

  - mutually exclusive branches do not conflict unless a value joins afterward;
  - loop-carried values remain live across the backedge;
  - DMA source/destination remain live to their explicit completion boundary;
  - local fence closes only its declared local domain;
  - `wafer.instr.dte_wait` closes referenced DTE tokens but not unrelated tokens;
  - fixed-capacity segmented count buffers remain live through count completion, while payload staging remains live through the
    distinct data completion; a count wait cannot release payload staging;
  - MXFP decode destination/scratch remains live through its local completion and dominates GEMM consumption;
  - pending output/state-affecting completion at every exit is rejected;
  - busytable metadata never creates a happens-before edge.

- [ ] **Step 2: Run and confirm duplicated local event logic cannot satisfy the tests**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  ```

- [ ] **Step 3: Implement structured control-flow dataflow**

  Numbering operations is not a proof. Build block/region dataflow states with pending completion sets and resource last-use facts; join `scf.if` paths, iterate `scf.for` to a fixed point, and carry token/resource state through block arguments/yields. Reject unsupported multiblock/concurrency forms before producing a result.

- [ ] **Step 4: Separate completion domains**

  Query token producers and op interfaces/effects. A wait/fence removes only tokens in the domain and operand set it explicitly covers. Status/error observation remains attached to the completion node and is not implied by token consumption.

- [ ] **Step 5: Derive liveness from completion**

  For every root/view, extend read/write live segments through the matching issue completion. Return path conditions, structured positions and conflict queries; do not return a scalar first/last operation interval as the semantic result.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add include/Wafer/Analysis/Completion/EntryCompletionAnalysis.h \
    lib/Wafer/Analysis/Completion/EntryCompletionAnalysis.cpp \
    include/Wafer/Analysis/Memory/EntryLivenessAnalysis.h \
    lib/Wafer/Analysis/Memory/EntryLivenessAnalysis.cpp \
    lib/Wafer/Analysis/CMakeLists.txt \
    unittests/Analysis/CompletionAndLivenessTest.cpp unittests/CMakeLists.txt
  git commit -m "Analyze executable completion and resource liveness"
  ```

### Task 6: Typed Executable Completion Graph

**Files:**
- Modify: `include/Wafer/IR/Executable/ExecutableOps.td`
- Modify: `lib/Wafer/IR/Executable/ExecutableOps.cpp`
- Modify: `lib/Wafer/IR/CMakeLists.txt`
- Create: `lib/Wafer/Transforms/Executable/MaterializeCompletionGraph.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Dialect/Wafer/Executable/completion-graph.mlir`
- Create: `test/Dialect/Wafer/Executable/invalid-completion-graph.mlir`
- Create: `test/Transforms/materialize-completion-graph.mlir`
- Create: `test/Transforms/materialize-iterated-entry-graph.mlir`

**Interfaces:**
- Consumes: candidate executable variant, frontend typed `ModelEntrypointId` records, declared invocation fields, explicit
  bounded `ExecutionSchedulePolicy`, `EntryCompletionInfo` for every entry, executable state-group realizations, Task 8
  planned stream windows/copy demands and exact no-I/O immutable materialization plans.
- Produces: variant-owned finite iteration domains, typed completion nodes/iteration edges/terminal policy, entry
  `CompletionExportId -> node` refs, finite bounded-count activation predicates/expert waves and
  `wafer.executable.invocation` model-API root/terminal mappings; instruction bodies remain the schedule owner.

```cpp
mlir::LogicalResult materializeCompletionGraph(
    ExecutableVariantOp variant,
    const ExecutionSchedulePolicy &schedulePolicy,
    llvm::ArrayRef<std::pair<ExecutableEntryOp, EntryCompletionInfo>> entries);
```

- [ ] **Step 1: Add parser/verifier failures first**

  Cover duplicate node/domain IDs, dangling edges, zero/overflow/out-of-bound count, illegal delta, zero-delta cycle,
  non-progressing template cycle, expanded terminal hole, an entry export bound to two nodes, repeated same-kind nodes without
  distinct static occurrence key, a leaf bound to the wrong entry/rank/stage scope, output with no terminal-success path, local
  drain used as the only invocation terminal, missing status/error source for transport completion, window copy not dominating
  consumer or reuse before last consumer, and state snapshot/publish/poison inconsistent with group realization/slot roles.
  Positive PP uses a declared InvocationPolicyFieldId, two microbatches, two stages, overlap and a +1 edge. Add a multi-API
  positive mapping distinct typed prefill/decode `ModelEntrypointId`s to different initial roots while sharing lower-level
  entries; reject unknown IDs, missing API terminal, wrong IO/state-group contract and use of static `EntryId` as model API.
  Add a sparse MoE positive where segmented count completion controls two expert waves, zero-count expert copy/module/entry/data
  nodes skip through a conditional join, and active terminals dominate combine/reuse. Reject unbounded/wrong-owner count,
  evaluate-before-count, missing skip join, active path outside finite wave, conditional persistent-state write and user output.

- [ ] **Step 2: Define only the records required by numbered designs**

  Add typed ODS records for:

  ```text
  completion node: stable id, kind, entry/rank/stage/request scope,
                   timeout policy, status source, failure domain
  completion edge: predecessor node, successor node, success/error relation
                   optional IterationDomainId and iteration_delta = 0 | +1
  iteration domain: nonzero scoped id, typed BoundedCountExpr, max_count
  terminal policy: success terminals, failure aggregation, cleanup relation
  entry export: CompletionExportId, completion node
  model invocation: ModelEntrypointId, initial template roots,
                    required IO/state contract, terminal output/state/completion roots
  activation predicate: ActivationPredicateId, bounded count resource/member/offset/type/capacity,
                        producer completion, true/false activation
  expert wave: canonical predicate/member set, max active count,
               conditional capacity envelope, active/skip join and reuse roots
  ```

  Stable kinds cover host command, device local drain, DMA, resource-copy issue/complete, DTE/collective wait, stage barrier,
  status/error observe, cleanup/release, state-snapshot-ready, state-group-publish and state-group-poison. Use typed refs,
  candidate-only `PlannedStreamWindowRef`/state-group IDs and no provider API name or instruction list. Final
  `StreamWindowId` is impossible until Task 13 forms `RankClassId` and `ResourceRealizationRecordKey`.

- [ ] **Step 3: Run the absent-IR failure**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Dialect/Wafer/Executable/(invalid-)?completion-graph\.mlir|Transforms/materialize-(completion-graph|iterated-entry-graph)\.mlir'
  ```

- [ ] **Step 4: Materialize from analysis, then reverify from IR**

  The transform first normalizes the explicit policy into finite iteration domains, then creates nodes from explicit entry
  command/device/DMA/DTE/collective boundaries, edges and iteration deltas from the analyzed happens-before/stage graph, and
  exports for launch-visible completion. It assigns CompletionNodeId using canonical node fields plus static graph occurrence,
  not construction order. Before physical transport acceptance, a DTE status source is a typed `TransportActionId` reference;
  Task 10 must bind it to accepted status/error resources. Task 8 supplies planned window copy/reuse relations; the coordinator
  invokes this materializer only after Task 8 windows/materialization plans exist. Each model entrypoint is resolved
  only along the verified `ModelProgramMemberId -> distributed ComponentId/source-member -> candidate EntryId` relations and
  typed model/distributed edges to roots/terminals in this same graph; no symbol-name reachability search is allowed. Shared
  PP/MoE members and one EntryId reachable from multiple model APIs remain one graph node with multiple typed mappings.
  For typed segmented MoE, map each bounded count element through the same member/component/entry relation, create canonical
  `ActivationPredicateId`s and compiler-planned finite waves, and put weight-copy/module-first-use/expert-entry/data nodes on
  true activation. Add explicit predicate-evaluate and conditional-join nodes; false is skipped-success. Verify active and skip
  paths dominate combine/reuse, each wave envelope fits target capacity, and no conditional node mutates persistent state or
  owns a user terminal. Runtime does not choose wave membership.
  Completion nodes retain clone-local planned-window refs; Task 13 rewrites them to final typed IDs in the same commit that
  finalizes resource realizations. For `atomic_version`, `state_snapshot_ready` dominates every
  candidate slot consumer and one `state_group_publish` is dominated by all member writer-success nodes. For in-place policy,
  every path after any member write reaches group poison before failure terminal. Run the ordinary verifier after materialization; it
  must rederive bounded-expression legality, zero-delta acyclicity, positive progress and terminal reachability without access
  to `EntryCompletionInfo`.

- [ ] **Step 5: Ensure target/package boundaries reject incomplete graphs**

  Extend executable verification helpers used by target/package consumers so candidate state, missing terminal policy, unresolved export or local-drain-only completion fails before target mutation or package assembly.

  Register `wafer-materialize-completion-graph` as a candidate-only replay pass for the transform test. It cannot accept committed input or create committed state.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Dialect/Wafer/Executable/(invalid-)?completion-graph\.mlir|Transforms/materialize-(completion-graph|iterated-entry-graph)\.mlir'
  git add include/Wafer/IR/Executable/ExecutableOps.td \
    lib/Wafer/IR/Executable/ExecutableOps.cpp lib/Wafer/IR/CMakeLists.txt \
    lib/Wafer/Transforms/Executable/MaterializeCompletionGraph.cpp \
    include/Wafer/Transforms/Passes.td \
    lib/Wafer/Transforms/CMakeLists.txt \
    test/Dialect/Wafer/Executable/*completion-graph.mlir \
    test/Transforms/materialize-completion-graph.mlir \
    test/Transforms/materialize-iterated-entry-graph.mlir
  git commit -m "Materialize typed executable completion graphs"
  ```

### Task 7: Whole-Entry SPM Planning

**Files:**
- Create: `include/Wafer/Transforms/SPM/EntrySPMPlanning.h`
- Create: `lib/Wafer/Transforms/SPM/EntrySPMPlanning.cpp`
- Modify: `lib/Wafer/Transforms/SPM/PlanSPMMemory.cpp`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Transforms/plan-entry-spm-memory.mlir`
- Create: `test/Transforms/plan-entry-spm-memory-failure.mlir`
- Create: `test/Transforms/plan-entry-spm-memory-completion.mlir`

**Interfaces:**
- Consumes: one complete instruction-level executable entry, `EntryCompletionInfo`, `EntryLivenessInfo`, shared physical geometry, and SPM capability/reserved ranges from target environment/policy.
- Produces: accepted `wafer.spm.offset` facts on all compiler-managed SPM roots in the candidate clone; no allocation plan object survives.

```cpp
struct EntrySPMPlanSummary {
  int64_t highWaterBytes;
  int64_t allocationCount;
};

mlir::FailureOr<EntrySPMPlanSummary>
planEntrySPMMemory(ExecutableEntryOp entry,
                   const EntryCompletionInfo &completion,
                   const EntryLivenessInfo &liveness,
                   TargetEnvironmentOp environment);
```

- [ ] **Step 1: Add entry-scope failures**

  Positive cases: two groups reuse a range after explicit completion; mutually exclusive branches reuse; loop-carried accumulator remains live; DTE staging is held through wait; segmented count and data buffers have separate lifetimes; direct full-shape uses the same planner. Negative cases: group-local reuse overlaps a later group, segmented payload staging reused after count but before data completion, send source reused before DTE wait, recv destination read before wait, pending DMA/DTE token at exit, physical size overflow, alignment/range violation and capacity overflow.

- [ ] **Step 2: Confirm current tile-region planner scope is too small**

  ```bash
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/plan-entry-spm-memory.*\.mlir'
  ```

- [ ] **Step 3: Replace local event numbering with shared analyses**

  Move deterministic interval packing and offset materialization into `EntrySPMPlanning.cpp`. Demand size/alignment comes from `VerifiedInstructionGeometry` plus memref type/layout; conflicts come exclusively from `EntryLivenessInfo`. Delete the private `EventInfo`, branch bitset and lifetime reconstruction from `PlanSPMMemory.cpp`.

- [ ] **Step 4: Make the existing pass a replay adapter**

  `wafer-plan-spm-memory` locates candidate executable entries and invokes the entry API. A legacy fixture without entry identity may be wrapped by the test/replay adapter, but production coordinator always supplies typed entries and target environment. The pass must not commit a subset of entries. Task 3B invokes completion/liveness analysis, then SPM planning for every entry, inserts DDR/window planning, and only then materializes completion records.

- [ ] **Step 5: Require terminal closure before offsets are accepted**

  Before writing any offset, verify that every exit has no output/state/resource-affecting pending completion and every DTE buffer reuse is dominated by the matching wait. Local fence and busytable facts cannot close DTE completion.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/plan-(entry-)?spm-memory.*\.mlir'
  git add include/Wafer/Transforms/SPM/EntrySPMPlanning.h \
    lib/Wafer/Transforms/SPM/EntrySPMPlanning.cpp \
    lib/Wafer/Transforms/SPM/PlanSPMMemory.cpp \
    lib/Wafer/Transforms/CMakeLists.txt test/Transforms/plan-entry-spm-memory*.mlir
  git commit -m "Plan SPM across complete executable entries"
  ```

### Task 8: Multi-Arena Whole-Entry DDR Planning

**Files:**
- Create: `include/Wafer/Transforms/DDR/EntryDDRPlanning.h`
- Create: `lib/Wafer/Transforms/DDR/EntryDDRPlanning.cpp`
- Modify: `lib/Wafer/Transforms/DDR/PlanDDRMemory.cpp`
- Create: `lib/Wafer/Transforms/DDR/PlanStreamedImmutable.cpp`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Transforms/plan-entry-ddr-memory.mlir`
- Create: `test/Transforms/plan-entry-ddr-memory-failure.mlir`
- Create: `test/Transforms/plan-entry-ddr-memory-multi-arena.mlir`
- Create: `test/Transforms/plan-streamed-immutable.mlir`
- Create: `test/Transforms/plan-streamed-immutable-failure.mlir`

**Interfaces:**
- Consumes: complete memory-planned entries, candidate resource realizations with resident or `streamed_pending` policy,
  typed consumer slices, model-member/component/expert count relations, target arenas, liveness, accepted SPM and shared
  geometry.
- Produces: accepted offsets plus `streamed_planned` canonical source windows, compiler-created staging resources/ranges and
  copy/consumer/reuse demands. Sparse expert windows carry a clone-local typed activation owner/wave relation later resolved to
  `ActivationPredicateId`; final artifact/chunk digests remain Task 8A.

```cpp
struct AcceptedDDRAssignment {
  mlir::Value root;
  mlir::Attribute arenaId;
  mlir::Attribute placementDomain;
  int64_t offsetBytes;
  int64_t spanBytes;
};

struct EntryDDRPlanResult {
  llvm::SmallVector<AcceptedDDRAssignment> assignments;
  llvm::SmallVector<PlannedStreamWindow> streamWindows;
};

mlir::FailureOr<EntryDDRPlanResult>
planEntryDDRMemory(ExecutableEntryOp entry,
                   const EntryLivenessInfo &liveness,
                   TargetEnvironmentOp environment);

mlir::LogicalResult verifyVariantDDRResources(
    ExecutableVariantOp variant,
    llvm::ArrayRef<EntryDDRPlanResult> entryPlans);
```

- [ ] **Step 1: Write demand-class and arena tests**

  Positive cases include existing demand classes plus an immutable backing larger than resident capacity whose explicit
  consumer slices fit two staging lanes. Check exact logical/source coverage, copy dominance, double-buffer overlap and reuse
  after last consumer. Add many-expert weights whose total exceeds the arena but compiler-planned finite waves fit one bounded
  envelope; zero/nonzero activation is not evaluated here, but every expert window binds its typed count/member owner.

  Negative cases add hidden full-backing consumer, window hole/ambiguous overlap, staging capacity/alignment/bandwidth failure,
  missing copy edge and reuse-before-last-consumer.

- [ ] **Step 2: Run and expose the implicit single-arena assumption**

  ```bash
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/plan-entry-ddr-memory.*\.mlir'
  ```

- [ ] **Step 3: Recover typed demands**

  Resolve slots through candidate refs. For `streamed_pending`, derive windows only from typed logical consumer slices and
  accepted StorageEncoding byte mapping; call `declareCandidateResource` for launch-visible staging roots and create planned
  copy/consumer/reuse relations. For typed sparse MoE, form deterministic finite expert waves and a worst-case per-wave
  capacity envelope from target limits, and retain clone-local count/member activation refs for Task 6; do not evaluate actual
  counts, invent router choices/K splits or allocate all expert backing.

- [ ] **Step 4: Pack each arena instance independently**

  Use deterministic interval packing with `EntryLivenessInfo`. Reuse is allowed only when lifetime does not overlap inside the same arena instance. Call `verifyVariantDDRResources` after all entry plans to validate shared resource coverage; do not merge different per-rank/per-stage instances.

- [ ] **Step 5: Preserve the accepted-fact boundary**

  Materialize offsets and replace each `streamed_pending` with `streamed_planned` in the same clone. Planned records use
  transaction-private artifact handles and no final digest/StreamWindowId. Keep analysis local; ResourceView later rederives
  arena/window/edge coverage.

- [ ] **Step 6: Make the old pass delegate and remove duplicate lifetime logic**

  `PlanDDRMemory.cpp` remains a replay pass. Task 3B uses completion/liveness -> SPM -> DDR/window planning ->
  no-I/O immutable materialization plan -> completion/activation records -> variant resource/target checks -> candidate
  structural candidate completion; Task 3B then performs selected-candidate byte materialization through Task 8A.

- [ ] **Step 7: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/plan-(entry-)?ddr-memory.*\.mlir'
  git add include/Wafer/Transforms/DDR/EntryDDRPlanning.h \
    lib/Wafer/Transforms/DDR/EntryDDRPlanning.cpp \
    lib/Wafer/Transforms/DDR/PlanDDRMemory.cpp \
    lib/Wafer/Transforms/DDR/PlanStreamedImmutable.cpp \
    lib/Wafer/Transforms/CMakeLists.txt test/Transforms/plan-entry-ddr-memory*.mlir \
    test/Transforms/plan-streamed-immutable*.mlir
  git commit -m "Plan typed DDR arenas across executable entries"
  ```

### Task 8A: Immutable Artifact Materialization Transaction

**Files:**
- Create: `include/Wafer/Compiler/ImmutableArtifactMaterialization.h`
- Create: `lib/Wafer/Compiler/ImmutableArtifactMaterialization.cpp`
- Create: `lib/Wafer/Compiler/ImmutableArtifactMaterializationInternal.h`
- Create: `include/Wafer/Support/TensorPayloadReader.h`
- Create: `lib/Wafer/Support/TensorPayloadReader.cpp`
- Modify: `include/Wafer/Transforms/Executable/WholeVariantPlanning.h`
- Modify: `lib/Wafer/Transforms/Executable/WholeVariantPlanning.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Create: `unittests/Compiler/ImmutableArtifactMaterializationTest.cpp`
- Create: `test/Integration/immutable-artifact-transaction.test`

**Interfaces:**
- Consumes: frontend-owned `VerifiedProgramSource` root capability plus expected source size/digest/encoding, accepted
  quant/storage/layout, resident or `streamed_planned` realizations and validated
  `ImmutableArtifactMaterializationLimits`, through transaction-private materialization read/work leases jointly charged to the
  source coordinator and outer transaction. The plan builder runs during structural candidate evaluation; byte materialization
  runs only after the coordinator selects a complete passing candidate.
- Produces: first, non-forgeable no-I/O `ImmutableArtifactMaterializationPlan` records; then, only for the selected candidate,
  `MaterializedImmutableArtifact` proofs plus exact staged blobs/digests/fixed-policy chunk tables in the same transaction. IR
  remains resident-proposal/`streamed_planned` until Task 13.

```cpp
struct ImmutableArtifactLimitValues {
  uint64_t maxPlans, maxPlanNodes, maxRecipeBytes, maxSourceSlices;
  uint64_t maxArtifacts, maxChunkRecords, maxMaterializationAttempts;
  uint64_t maxSingleSourceReadBytes, maxTotalSourceReadBytes;
  uint64_t maxSingleOutputBytes, maxTotalOutputBytes;
  uint64_t maxStagingBytes, maxBufferBytes, maxWorkers;
  uint64_t maxConcurrentReaders, maxConcurrentWriters, maxOpenFiles;
  uint64_t maxDiagnosticBytes;
};

class ImmutableArtifactMaterializationLimits final {
public:
  static llvm::Expected<ImmutableArtifactMaterializationLimits>
  create(const ImmutableArtifactLimitValues &values);
  const ImmutableArtifactLimitValues &values() const;
  // All fields are positive; zero never means unbounded.
private:
  explicit ImmutableArtifactMaterializationLimits(
      ImmutableArtifactLimitValues values);
  ImmutableArtifactLimitValues values_;
};
```

- [ ] **Step 1: Add byte-level and atomicity failures**

Cover raw/reordered/packed resident payload, multiwindow low-precision payload, wrong NPY dtype/shape, overflow, window hole,
packing mismatch, source replacement/symlink/truncate/grow/concurrent mutation after initial verification, injected writer/
publication failure and candidate rejection after bytes were staged. Source IR and visible content store/index remain
byte-identical on every failure. Run the same payload with different legal IO-buffer limits and worker counts and require
identical blob bytes, content digest and chunk table. Add many structurally passing candidates over one large sparse source:
all get plans, only the deterministic winner reads/packs bytes; equal recipes reuse one staged proof. Candidate-specific
materialization failure advances under attempt/byte limits, while source-global digest/IO failure stops the search.

- [ ] **Step 2: Build exact no-I/O materialization plans**

From typed logical coverage, source refs, quant/storage attrs, residency/windows and fixed chunk policy, compute checked exact
output byte counts and a canonical transaction-local recipe key. The private plan binds every source slice and output relation
but opens no payload and writes no blob. Candidate cost/legality consumes this object; it is neither IR nor a serialized
sidecar. Enforce plan/count/estimated-byte limits and reject a key collision whose full recipe differs.

- [ ] **Step 3: Implement structured bounded payload reading for the selected candidate**

Have `VerifiedProgramSource` retain an open root capability plus validated relative locators and expected exact size/digests,
but expose no public acquire/open/read. `detail::ArtifactMaterializationSourceAccess` first move-consumes same-owner
`ArtifactMaterializationReadLease`/work lease issued by the candidate transaction under the source coordinator,
`ImmutableArtifactMaterializationLimits` and retained outer budgets; only then perform beneath/no-follow relative open and
return a move-only `SourceArtifactLease`. Parse NPY/tensor/resource headers, positioned reads, full
source digest and before/after stat from that same opened object. Never reopen by path after verification. Use checked
multidimensional indexing and descriptor rules for logical slice to physical bytes. Scale/zp/block tables use typed resource
refs. No parameter/file name or Python callback selects packing.

- [ ] **Step 4: Stream transform, hash and write**

Transform bounded input chunks directly to transaction-private content-addressed output while computing full SHA-256. Build
`ArtifactChunkingPolicyV1` exactly at 4 MiB boundaries from offset zero with one checked tail and SHA-256 per canonical chunk;
smaller IO reads may cross a chunk incrementally but cannot change its boundary. Validate all positive nonzero buffer/worker/
byte limits before opening output and prove a large packed payload never remains fully in RAM. Resident requires full logical
coverage; streamed windows reference every intersecting complete canonical chunk and require exact source/logical coverage.

- [ ] **Step 5: Finalize records without partial state**

Only after all bytes pass, construct private-ctor immutable proof objects in the owning
`ExecutableCandidateTransaction` proof map and bind them to staged handles. Do not mutate IR residency, create ArtifactRef or
assign final IDs here. Task 13 first forms RankClass coverage, then verifies the same proof/staged objects, assigns
`ResourceRealizationRecordKey`/`StreamWindowId`, rewrites completion refs and finalizes resident/`streamed_windows` in one
commit. Any failure rolls back proof entries and staged objects.

- [ ] **Step 6: Run and commit**

```bash
cmake --build build/wafer-dev --target WaferUnitTests wafer-opt -- -j128
ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
<configured-lit> -sv build/wafer-dev/test/Integration/immutable-artifact-transaction.test
git add include/Wafer/Compiler/ImmutableArtifactMaterialization.h \
  lib/Wafer/Compiler/ImmutableArtifactMaterialization.cpp \
  lib/Wafer/Compiler/ImmutableArtifactMaterializationInternal.h \
  include/Wafer/Support/TensorPayloadReader.h lib/Wafer/Support/TensorPayloadReader.cpp \
  include/Wafer/Transforms/Executable/WholeVariantPlanning.h \
  lib/Wafer/Transforms/Executable/WholeVariantPlanning.cpp \
  lib/Wafer/Compiler/CMakeLists.txt \
  unittests/Compiler/ImmutableArtifactMaterializationTest.cpp test/Integration/immutable-artifact-transaction.test
git commit -m "Materialize immutable executable artifacts atomically"
```

### Task 9: Segmented Collective Correctness Path

**Files:**
- Modify: `include/Wafer/IR/LinalgExt/CollectiveOps.td`
- Modify: `lib/Wafer/IR/LinalgExt/CollectiveOps.cpp`
- Modify: `include/Wafer/IR/Tile/CommOps.td`
- Modify: `lib/Wafer/IR/Tile/CommOps.cpp`
- Modify: `include/Wafer/IR/Instr/DTEOps.td`
- Modify: `lib/Wafer/IR/Instr/DTEOps.cpp`
- Create: `lib/Wafer/Transforms/Communication/MaterializeSegmentedAllToAll.cpp`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Modify: `include/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h`
- Modify: `lib/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.cpp`
- Create: `test/Dialect/Wafer/LinalgExt/Collective/segmented-all-to-all.mlir`
- Create: `test/Dialect/Wafer/LinalgExt/Collective/invalid-segmented-all-to-all.mlir`
- Create: `test/Dialect/Wafer/Tile/Comm/segmented-all-to-all.mlir`
- Create: `test/Transforms/convert-segmented-all-to-all.mlir`
- Create: `test/Transforms/convert-segmented-all-to-all-failure.mlir`

**Interfaces:**
- Consumes: explicit `wafer.linalg_ext.collective.segmented_all_to_all` with payloads, per-peer send/recv count and displacement buffers, static capacities, logical rank group, count-exchange token and data-phase token.
- Produces: `wafer.tile.segmented_all_to_all`, then a fixed-capacity two-phase unicast DTE body whose actual counts remain SSA/control facts visible to verifier and completion/liveness.

- [ ] **Step 1: Define verifier-first fixtures**

  Positive four-peer fixture uses uneven counts including a zero-count peer, monotonic displacements, equal total send/receive counts across peers, static capacity per peer and distinct count/data tokens. Negative cases: negative count, count above capacity, displacement plus count overflow, peer total mismatch, missing peer, data phase not dominated by count completion, reused token, result capacity overflow and policy `bounded_variable_segments` without target ABI support.

- [ ] **Step 2: Add the two typed ops**

  Implement exactly the two mnemonics fixed by `tasks/13-communication.md`. Both carry payload source/destination, send/recv counts and displacements, per-peer capacities, logical group and phase tokens. Tensor-level op owns distributed segmented semantics; tile op owns buffer-level staging/effects. Neither carries endpoint/channel/FSM, runtime handles or raw DTE fields.

- [ ] **Step 3: Run parser/verifier tests before lowering**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Dialect/Wafer/(LinalgExt/Collective|Tile/Comm)/(invalid-)?segmented-all-to-all\.mlir'
  ```

- [ ] **Step 4: Implement `padded_fixed_capacity` materialization**

  Build contiguous per-peer staging buffers sized to compile-time capacities. For every newly introduced launch-visible staging/control root, call the shared `declareCandidateResource` builder with owner EntryId, structural root ordinal, canonical role, type/access/scope and current axis coverage in the same rewrite transaction; an existing identical declaration is reused and any conflict fails. This transform does not hand-roll a `ResourceId` or defer declaration to ResourceView/commit. Count phase exchanges/validates counts and waits before data issue. Data phase sends each fixed-capacity segment using existing unicast DTE; actual count/displacement controls unpack/visibility and remains checked against capacity. Zero-count peers still participate in count/status matching but do not authorize out-of-range access. Task 3B inserts this materialization before whole-entry layout/instruction lowering, then invalidates and recomputes layout/completion/memory analyses.

- [ ] **Step 5: Fail closed for variable-byte transport**

  When policy is `bounded_variable_segments`, emit `unsupported_segment_policy: target ABI has no verified variable-byte Direct DTE contract` before mutating the candidate. Do not add dynamic byte-count slots or CRT calls in this task.

- [ ] **Step 6: Expose the exact facts required by later memory/event consumers**

  The transform creates explicit staging allocs, issue tokens, waits and registered memory effects. Its own verifier tests prove
  count completion dominates data issue and both buffer roots/effects remain visible. Tasks 5 and 7 consume these generic facts
  and add the later liveness/SPM integration checks; this task does not edit not-yet-created analysis/coordinator files.

- [ ] **Step 7: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='segmented-all-to-all'
  git add include/Wafer/IR/LinalgExt/CollectiveOps.td \
    lib/Wafer/IR/LinalgExt/CollectiveOps.cpp \
    include/Wafer/IR/Tile/CommOps.td lib/Wafer/IR/Tile/CommOps.cpp \
    include/Wafer/IR/Instr/DTEOps.td lib/Wafer/IR/Instr/DTEOps.cpp \
    lib/Wafer/Transforms/Communication/MaterializeSegmentedAllToAll.cpp \
    lib/Wafer/Transforms/CMakeLists.txt \
    include/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h \
    lib/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.cpp \
    test/Dialect/Wafer/LinalgExt/Collective/*segmented-all-to-all.mlir \
    test/Dialect/Wafer/Tile/Comm/segmented-all-to-all.mlir \
    test/Transforms/convert-segmented-all-to-all*.mlir
  git commit -m "Lower fixed-capacity segmented collectives"
  ```

### Task 10: Accepted Physical Transport Records

**Files:**
- Modify: `include/Wafer/IR/Instr/DTEOps.td`
- Modify: `lib/Wafer/IR/Instr/DTEOps.cpp`
- Modify: `include/Wafer/IR/Executable/ExecutableOps.td`
- Modify: `lib/Wafer/IR/Executable/ExecutableOps.cpp`
- Create: `include/Wafer/Transforms/Executable/PhysicalTransportPlanning.h`
- Create: `lib/Wafer/Transforms/Executable/PhysicalTransportPlanning.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Dialect/Wafer/Executable/transport.mlir`
- Create: `test/Dialect/Wafer/Executable/invalid-transport.mlir`
- Create: `test/Transforms/accept-physical-transport.mlir`
- Create: `test/Transforms/accept-physical-transport-failure.mlir`

**Interfaces:**
- Consumes: every entry-local DTE issue/wait action, accepted SPM/DDR offsets, resource effects/liveness, target environment limits, topology, execution mesh and completion graph.
- Produces: uncommitted `wafer.executable.transport` records with complete concrete binding members, typed relocation schema and status/error/completion refs; no final projection decision.

```cpp
mlir::LogicalResult assignTransportActionIds(ExecutableEntryOp entry);

mlir::LogicalResult acceptPhysicalTransports(
    ExecutableVariantOp variant, TargetEnvironmentOp environment,
    TargetTopologyOp topology, ExecutionMeshOp mesh);
```

- [ ] **Step 1: Add deterministic action-ID tests**

  Create equivalent entries with different symbol names and SSA print names. Check identical typed integer `TransportActionId` assignment from entry-local structured block/op order. Reordering independent communication actions changes the corresponding IDs deterministically. Duplicate/missing IDs fail verification.

- [ ] **Step 2: Add transport record positives and negatives**

  Positive fixtures include two ranks with matched send/recv/wait, one `requires_pinned` action and one relocation-eligible action with at least two complete binding members. Check action-local `TransportBindingMemberId` assignment after canonical complete-member sorting and preservation through relocation selection. Negative cases: duplicate member assignment/ID, peer/phase/bytes mismatch, segmented count/data mismatch, missing receiver range/offset/wait/status/error, stale action ref, endpoint unavailable, exact DTE block/channel/FSM conflict, capacity/alignment failure, relocation slot wrong kind/width/owner/allowed set, and any half-concrete member.

- [ ] **Step 3: Define the candidate record without duplicating schedule**

  `wafer.executable.transport` references entry + `uint64_t TransportActionId`. Its typed records contain logical rank/peer/buffer range/completion dependency; finite complete members with action-local `uint32_t TransportBindingMemberId`, local/remote endpoint, exact DTE block and channel, local/remote FSM and stream, packet resource class, receiver storage/range/capacity/alignment/ownership/lifetime, status/error and release; optional `RelocationSchema`; and `requires_pinned`/relocation eligibility. Receiver/status/control slots reference candidate executable resources created by their first materializing transform through the shared builder; a missing record rejects the candidate. Transport acceptance does not create resource identity or policy. It must not contain collective steps, DTE op lists, projection mode/id/digest or runtime handles.

- [ ] **Step 4: Allocate and match across the complete variant**

  Enumerate finite target-declared DTE/channel/FSM/resource limits in deterministic order. Match every send with receiver-ready, receive, wait, status and release across canonical instances. Validate exact SPM/DDR ranges and conflict lifetimes. If the typed target environment lacks a required capability/limit, reject with `missing_transport_capability`; never synthesize a default resource count.

- [ ] **Step 5: Build relocation schemas only from complete members**

  A slot describes mechanical substitution among an already verified allowed value set and identifies field kind, width, owner action/member and allowed values. If any required field is not relocatable, mark the action `requires_pinned`. Do not choose the final mode here.

  Register `wafer-accept-physical-transport` as a candidate-only replay pass. It reads target/topology/mesh and accepted offsets from the fixture and delegates to `acceptPhysicalTransports`.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Dialect/Wafer/Executable/(invalid-)?transport\.mlir|Transforms/accept-physical-transport(-failure)?\.mlir'
  git add include/Wafer/IR/Instr/DTEOps.td lib/Wafer/IR/Instr/DTEOps.cpp \
    include/Wafer/IR/Executable/ExecutableOps.td \
    lib/Wafer/IR/Executable/ExecutableOps.cpp \
    include/Wafer/Transforms/Executable/PhysicalTransportPlanning.h \
    lib/Wafer/Transforms/Executable/PhysicalTransportPlanning.cpp \
    include/Wafer/Transforms/Passes.td \
    lib/Wafer/Transforms/CMakeLists.txt \
    test/Dialect/Wafer/Executable/*transport.mlir \
    test/Transforms/accept-physical-transport*.mlir
  git commit -m "Accept typed physical transport bindings"
  ```

### Task 11: Accepted Launch Projection

**Files:**
- Modify: `include/Wafer/IR/Executable/ExecutableOps.td`
- Modify: `lib/Wafer/IR/Executable/ExecutableOps.cpp`
- Modify: `include/Wafer/Compiler/ExecutableIdentityBuilder.h`
- Modify: `lib/Wafer/Compiler/ExecutableIdentityBuilder.cpp`
- Modify: `include/Wafer/Compiler/StaticFunctionIdentity.h`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Create: `include/Wafer/Transforms/Executable/LaunchProjection.h`
- Create: `lib/Wafer/Transforms/Executable/LaunchProjection.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Dialect/Wafer/Executable/launch-projection.mlir`
- Create: `test/Dialect/Wafer/Executable/invalid-launch-projection.mlir`
- Create: `test/Transforms/materialize-launch-projection.mlir`
- Create: `test/Transforms/materialize-launch-projection-failure.mlir`

**Cross-plan prerequisite:** Before this task, execute the shared identity foundation from `tasks/plans/target-artifact-set.md` that creates runtime-safe `WaferABI`, generated identity schemas, schema-driven encoding, `WaferCompilerIdentity` support and the static-function encoder. The rest of the target-artifact plan remains downstream of committed executable. This task adds projection/executable adapters to that one compiler target; it must not call raw WCRE fields or create another encoder.

**Interfaces:**
- Consumes: complete candidate entries/resources, distributed prerequisite classes/instances, environment/topology/mesh
  fingerprints, accepted transport records, verified target build profile/external-symbol registry and the one all-entry
  `VerifiedStaticFunctionClosure` proof from the shared identity foundation.
- Produces: executable-root-owned accepted pinned or relocatable projection-set record with deterministic member priority, `ProjectionSetId` and `wafer.projection-set.v1` digest; the candidate variant stores only the typed ref.

```cpp
mlir::LogicalResult materializeLaunchProjection(
    ExecutableVariantOp variant, TargetEnvironmentOp environment,
    TargetTopologyOp topology, ExecutionMeshOp mesh,
    const compiler::VerifiedStaticFunctionClosure &closures);
```

- [ ] **Step 1: Add pinned and relocatable verifier tests**

  Pinned positive: every execution instance maps once to an available endpoint and one complete transport member vector; environment/topology/mesh digests match. Relocatable positives: ordered `ConcreteRecordSet`, and `FiniteTemplateSet` whose finite `allowed_bindings` expand to the same schema and all pass coverage/conflict checks.

  Negative cases: instance coverage hole/duplicate, unavailable endpoint, simultaneous instance collision, rank-parametric slot mismatch, per-rank-static specialization coordinate mismatch, stale environment/topology/mesh/entry digest, transport member not complete, `requires_pinned` used in relocatable mode, relocation slot mismatch, template with unbounded binding, invalid priority, digest mismatch and runtime-search-like wildcard.

- [ ] **Step 2: Define one typed union owner**

  Extend executable ODS so the candidate executable root owns exactly one accepted projection-set record for this variant with:

  ```text
  ProjectionSetId, mode, deterministic member priority, canonical digest,
  environment/topology/mesh refs and digests,
  distributed variant/component/ExecutionInstanceId/prerequisite-class refs,
  entry semantic digest, partition/replica coordinate, endpoint, entrypoint, block id,
  typed resource bindings, accepted transport member refs,
  pinned mapping | ConcreteRecordSet | FiniteTemplateSet(template, allowed_bindings)
  ```

  `ExecutableVariantOp` references the set ID and does not copy its fields. Multiple variants may intentionally reference the same identical set. The root verifier rejects duplicate owners for one ID. The record does not contain target module digest, runtime handle or search policy.

- [ ] **Step 3: Run the absent-projection failures**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Dialect/Wafer/Executable/(invalid-)?launch-projection\.mlir|Transforms/materialize-launch-projection(-failure)?\.mlir'
  ```

- [ ] **Step 4: Build finite candidate sets without reallocating transport**

  Derive logical-to-physical endpoint views from mesh/topology and consume only complete transport binding members. Pinned selects one complete vector. Relocatable constructs an ordered set from already verified member vectors; a finite template is expanded at compile time and each binding is checked for rank/resource/transport coverage and collision. No ring/tree/route/channel/FSM selection is repeated here.

- [ ] **Step 5: Resolve all static closures once and compute canonical identity through `WaferABI`**

  Call the shared `resolveStaticFunctionClosure` once over every typed entry root in the executable and the build profile's
  `VerifiedExternalSymbolRegistry`. The returned immutable proof may contain multiple disconnected closure units and a total
  `EntryId -> unit` mapping. For each entry, call the one shared `encodeStaticFunctionIdentity(function, closures)` and retain
  its typed `abi::StaticFunctionDigest`; do not return `DigestAttr`, resolve per entry, walk another symbol graph, hash private
  symbol paths or create an aggregate closure digest. Populate generated `ProjectionSetIdentity` from the verified projection
  op and call the shared record-specific encoder. Task 13 reruns the same all-entry resolver from the final clone and requires
  identical unit records/private labels/per-entry digests before commit; target artifact construction consumes that proof/API
  again rather than inventing another partitioner. Add tests proving private rename/declaration permutation/MLIR printing do
  not affect identities, while helper body/global initializer/endpoint/member/slot/priority changes do.

  Register `wafer-materialize-launch-projection` as a candidate-only replay pass. It must reject a candidate missing complete transport records and must never flip commit state.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Dialect/Wafer/Executable/(invalid-)?launch-projection\.mlir|Transforms/materialize-launch-projection(-failure)?\.mlir'
  git add include/Wafer/IR/Executable/ExecutableOps.td \
    lib/Wafer/IR/Executable/ExecutableOps.cpp \
    include/Wafer/Compiler/ExecutableIdentityBuilder.h \
    include/Wafer/Compiler/StaticFunctionIdentity.h \
    lib/Wafer/Compiler/ExecutableIdentityBuilder.cpp \
    lib/Wafer/Compiler/CMakeLists.txt \
    include/Wafer/Transforms/Executable/LaunchProjection.h \
    lib/Wafer/Transforms/Executable/LaunchProjection.cpp \
    include/Wafer/Transforms/Passes.td \
    lib/Wafer/Transforms/CMakeLists.txt \
    test/Dialect/Wafer/Executable/*launch-projection.mlir \
    test/Transforms/materialize-launch-projection*.mlir
  git commit -m "Materialize finite executable launch projections"
  ```

### Task 12: Executable Resource View

**Files:**
- Create: `include/Wafer/Analysis/Executable/ExecutableResourceView.h`
- Create: `lib/Wafer/Analysis/Executable/ExecutableResourceView.cpp`
- Modify: `lib/Wafer/Analysis/CMakeLists.txt`
- Create: `unittests/Analysis/ExecutableResourceViewTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: the owning `ExecutableCandidateTransaction` and its exact clone generation; every candidate resource/state group/
  model invocation mapping, typed model/distributed identity, IR quant/storage/profile refs, accepted SPM/DDR offsets,
  resident or planned-stream realizations, provisional windows/staging/copy/reuse refs, same-transaction
  `MaterializedImmutableArtifact` proofs/staged blobs, target arenas/placement, topology/mesh, transport/projection, memref
  roots/views/use-def and completion/liveness.
- Produces: private-construction transformation-local proof view over those exact transaction records. Task 13 calls the
  builder internally before class formation and again after final IDs/artifacts; no commit API accepts a caller-provided view.

```cpp
enum class ResourceViewPhase { CandidateBeforeFinalIds, FinalizedForCommit };

class ExecutableResourceView final {
public:
  ExecutableResourceView() = delete;
  uint64_t cloneGeneration() const;
  llvm::ArrayRef<ResourceViewRecord> resources() const;
  llvm::ArrayRef<StateGroupViewRecord> stateGroups() const;
  llvm::ArrayRef<InvocationViewRecord> invocations() const;
  llvm::ArrayRef<EntryBindingViewRecord> bindings() const;

private:
  friend mlir::FailureOr<ExecutableResourceView>
  buildExecutableResourceView(const ExecutableCandidateTransaction &,
                              ResourceViewPhase);
  // Private storage and constructor bind records to one transaction owner/generation.
};

mlir::FailureOr<ExecutableResourceView>
buildExecutableResourceView(const ExecutableCandidateTransaction &transaction,
                            ResourceViewPhase phase);
```

- [ ] **Step 1: Add resource reconstruction unit tests**

  Positive fixture covers candidate records for external input/output with symbolic bounded dimensions, imported immutable
  shard, resident immutable backing, persistent state group/full-copy/COW/in-place realization, launch-visible compiler
  workspace, inter-entry value, transport
  receiver/status/control resource and projection control slot across two target and two shape variants. Check exact existing
  `ResourceId`, `DimAttr` bounds, role/access/alias/update/consistency, explicit target/executable/ExecutionInstance/projection
  coverage, IR quant/storage/profile projection, storage/scope/arena/placement/range/alignment, safe cross-variant reuse,
  `StateSlotVersionRole`, model invocation roots/terminals and ordered entry slots. Add a streamed case with planned windows,
  staging/copy/reuse and same-transaction immutable proof, then a finalized case with exact ArtifactRef/
  `ResourceRealizationRecordKey`/`StreamWindowId`. Internal SPM temp remains absent from executable resources.

  Negative fixtures cover duplicate owner, realization overlap/hole across any axis, missing candidate record for launch-visible
  workspace/control/status, accidental resource record for internal SPM temp, unnamed/inferred role, missing slot, duplicate
  slot, wrong slot order, unbound function argument, resource range inconsistent with offset/descriptor, arena/placement
  mismatch, immutable write, state alias/update mismatch, content digest mismatch, transport/projection resource missing and
  wrong state-group member/policy/version role, missing/cross-transaction/stale materialized proof, wrong chunk/profile,
  provisional/final ID used in the wrong phase, invalid model invocation mapping and two facts that disagree about the same
  resource. Compile-time tests require the view to be non-aggregate, non-default/direct-constructible; runtime tests build a
  view, mutate the clone, and reject stale-generation/cross-transaction reuse.
  Add a sparse scale fixture with 100,000 resources across 10,000 ranks and multiple target/shape/projection records; assert
  bounded index/sort scratch and that no target x shape x instance x projection Cartesian product is allocated.

- [ ] **Step 2: Run and confirm no current analysis owns this boundary**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  ```

- [ ] **Step 3: Reconstruct from typed owners and current IR**

  Starting from each existing candidate `ExecutableResourceOp`, resolve its model/distributed identity, typed `DimAttr` bounds
  and realization coverage through variants, instances, shards and entry slots. Build bounded canonical sorted indexes keyed by
  the exact typed composite relations already present in resource uses, realizations, entries and projection members; use merge
  joins/set sweeps to prove every required relation has exactly one realization and no extras. Never materialize the Cartesian
  product of target/shape/execution-instance/projection axes. Follow memref views to roots; recompute physical size/
  range with shared geometry; validate accepted SPM/DDR offsets and arena instance; merge access/effect/liveness; cross-check
  transport/projection refs. Join each immutable realization to the owning transaction proof/staged blob; validate fixed chunk
  policy, logical/source coverage and planned copy/reuse. Reconstruct state-group member/axis/snapshot policy and every
  invocation API root/terminal contract. Every `mlir::Attribute` handle in the C++ view is required to be the concrete ODS attr declared
  by model/target/executable IR, never a free string. Missing launch-visible workspace/control/status records reject the
  candidate; this analysis never creates a `ResourceId`, realization policy or hash domain.

- [ ] **Step 4: Keep the view non-serializable**

  Do not add an op, attr, pass dump, file format or global cache for this view. The private object contains MLIR/proof handles
  into one clone generation. Any rewrite invalidates it; `commitExecutable` does not accept the object and instead invokes the
  friend builder at each required point. A caller cannot forge ranges or pass a stale view as commit authority.

- [ ] **Step 5: Validate entry ABI ordering without creating KAD**

  Derive numeric `SlotId` order and `StateSlotVersionRole` from the typed executable entry/function boundary established by the
  design. This task does not generate Kernel ABI descriptors; the target-artifact plan will cross-validate the committed mapping.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add include/Wafer/Analysis/Executable/ExecutableResourceView.h \
    lib/Wafer/Analysis/Executable/ExecutableResourceView.cpp \
    lib/Wafer/Analysis/CMakeLists.txt \
    unittests/Analysis/ExecutableResourceViewTest.cpp unittests/CMakeLists.txt
  git commit -m "Reconstruct executable resources from accepted IR"
  ```

### Task 3B: Whole-Variant Candidate Coordinator Integration

**Files:**
- Create: `lib/Wafer/Transforms/Executable/SelectExecutableVariant.cpp`
- Create: `lib/Wafer/Transforms/Executable/MaterializeTargetVariants.cpp`
- Create: `include/Wafer/Analysis/Executable/CandidateTargetPreflight.h`
- Create: `lib/Wafer/Analysis/Executable/CandidateTargetPreflight.cpp`
- Modify: `lib/Wafer/Analysis/CMakeLists.txt`
- Modify: `include/Wafer/IR/Executable/ExecutableOps.td`
- Modify: `lib/Wafer/IR/Executable/ExecutableOps.cpp`
- Modify: `lib/Wafer/IR/CMakeLists.txt`
- Modify: `include/Wafer/Transforms/Executable/WholeVariantPlanning.h`
- Modify: `lib/Wafer/Transforms/Executable/WholeVariantPlanning.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Transforms/select-executable-variant.mlir`
- Create: `test/Transforms/select-executable-variant-failure.mlir`

**Interfaces:**
- Consumes: one move-owned `ExecutableCompilationInput`, same-owner outer transaction, the Task 3A complete-assignment cursor,
  and every typed stage API implemented above. Planning options and exact target-context registry come only from the input owner.
- Produces: one move-owned candidate transaction that has passed the entire fixed coordinator sequence and contains selected
  immutable artifact proofs/staged blobs, or a structured aggregate rejection with source/output unchanged.

```cpp
mlir::LogicalResult materializeExecutableTargetVariants(
    ExecutableCandidateTransaction &transaction);

mlir::FailureOr<ExecutableCandidateTransaction>
selectExecutableCandidates(
    compiler::ExecutableCompilationInput input,
    compiler::ProgramOutputTransaction &output);
```

- [ ] **Step 1: Add all-rank, all-axis and late-failure tests**

  Cover two instances where only the last rank fails; two stages/components; two shape variants; two target variants; shared
  rank-parametric code; static-peer specialization; missing/duplicate/cross-paired target context; and `DirectFullShape`
  rejected only by a late SPM/transport gate while a tiled assignment passes. No rank/component/variant/blob may publish.
  Include 1000 entries with sparse fragments and late conflicts; prove the naive Cartesian product is never allocated.

- [ ] **Step 2: Materialize target requirements before frontier evaluation**

  Immediately after move-consuming the input and before allocating or mutating a working clone, Task 1 privately revalidates the
  sealed `VerifiedTargetProgramMaterializationPlan` against the all-and-only target requirements and exact canonical
  `VerifiedTargetCompilationContextRegistry`. Missing/extra/duplicate/cross-paired/stale contexts fail with no clone mutation or
  staging. Only after that gate does Task 1 create the full working clone. Through transaction-private access, materialize target
  environment/topology/mesh from the retained plan, recompute every
  identity/role/coverage fact from clone IR, and run the IR-local normalization/parallel/SPMD/distributed formation segment on
  that clone only. Formation must then produce exactly one candidate executable set with nonempty, canonical candidate variants;
  missing/two roots, zero/duplicate variants, mixed candidate/committed state and incomplete typed owner joins fail before solver
  evaluation. Any failure destroys the working clone and leaves the input-owned source byte-identical.

  Rejoin the materialized IR's rederived target requirements to that same retained registry and require exact equality with the
  pre-clone gate, then create root-owned candidate records containing only TargetVariantId, target ABI/capabilities and environment
  compatibility. This second join detects materializer/IR drift; it is not the first registry admission. Retain the full registry
  owner, not only context tokens, for low-precision, target preflight and later target attachment. No shape guard, rank
  class, module digest, tool path or runtime handle enters a target record.

- [ ] **Step 3: Replay each complete assignment through the static mandatory sequence**

  Use Task 3A's cursor in canonical order. For each assignment, reset a Task 1 working clone, apply each closed decision through
  repo-owned `CandidateAssignmentReplayAccess`, then visit entries by `ExecutionInstanceId` and call the concrete
  Task 2/9/4/4A/5/7/8/8A-plan/6/10/11/12 APIs in the global order. The access dispatch table is fixed code; there is no
  caller-supplied callable registry and no stage may return a generic passing bit. Recompute ordinary verifier,
  coverage, liveness/resource view and target preflight from the current clone after invalidating rewrites. A fragment summary
  only prunes incompatibility; it never authorizes legality.

  After Task 12, call `verifyCandidateTargetPreflight(transaction)`. Its compiler-private
  `CandidatePhysicalAllocationAccess` internally builds a fresh ResourceView and owner-bound
  `VerifiedPhysicalAllocationView`, then checks supported structure, registered external-symbol availability and every
  instruction geometry/range/narrowing relation. The API does not accept a caller ResourceView and cannot produce LLVM, KAD,
  object bytes or staged blobs. Any transaction rewrite invalidates the result.

- [ ] **Step 4: Select structurally complete candidates, then materialize bytes**

  `first-legal` uses the first structurally passing assignment; `min-estimated-time` ranks only structurally passing members of
  the same bounded canonical prefix with deterministic tie-break. For the proposed winner, call Task 8A's materializer through
  the same transaction/source capability and rerun artifact/resource/final target preflight. A candidate-specific packing/
  staging failure advances to the next structurally passing assignment only while attempt/read/write budgets remain; a
  source-global digest/mutation failure stops the search. Equal canonical recipes reuse one proof. No rejected candidate reads
  bytes, and no returned candidate has a provisional/missing artifact proof.

- [ ] **Step 5: Preserve deterministic failure semantics**

  If unvisited assignments remain when a search/materialization budget is exhausted, return `search_budget_exhausted`, not
  semantic illegality. Parallelism changes throughput only. Diagnostic payload may contain stable owner IDs, candidate values
  and failure class, but no frontier/nogood/cost trace enters IR, artifact identity or package.

- [ ] **Step 6: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt WaferUnitTests -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Transforms/select-(executable-variant|group-tile).*\.mlir|Integration/immutable-artifact-transaction\.test'
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add lib/Wafer/Transforms/Executable/SelectExecutableVariant.cpp \
    lib/Wafer/Transforms/Executable/MaterializeTargetVariants.cpp \
    include/Wafer/IR/Executable/ExecutableOps.td lib/Wafer/IR/Executable/ExecutableOps.cpp \
    lib/Wafer/IR/CMakeLists.txt include/Wafer/Analysis/Executable/CandidateTargetPreflight.h \
    lib/Wafer/Analysis/Executable/CandidateTargetPreflight.cpp lib/Wafer/Analysis/CMakeLists.txt \
    include/Wafer/Transforms/Executable/WholeVariantPlanning.h \
    lib/Wafer/Transforms/Executable/WholeVariantPlanning.cpp include/Wafer/Transforms/Passes.td \
    lib/Wafer/Transforms/CMakeLists.txt test/Transforms/select-executable-variant*.mlir
  git commit -m "Integrate whole-variant candidate selection"
  ```

### Task 13: Whole-Variant Verifier and Atomic Commit

**Files:**
- Create: `include/Wafer/IR/Executable/ExecutableVerification.h`
- Create: `lib/Wafer/IR/Executable/ExecutableVerification.cpp`
- Create: `include/Wafer/Transforms/Executable/CommitExecutable.h`
- Create: `lib/Wafer/Transforms/Executable/CommitExecutable.cpp`
- Create: `lib/Wafer/Transforms/Executable/ExecutableAttachmentStorage.h`
- Create: `lib/Wafer/Transforms/Executable/ExecutableProgramOutputAdapter.cpp`
- Modify: `lib/Wafer/Transforms/Executable/SelectExecutableVariant.cpp`
- Modify: `include/Wafer/Compiler/ExecutableIdentityBuilder.h`
- Modify: `lib/Wafer/Compiler/ExecutableIdentityBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Modify: `include/Wafer/IR/Executable/ExecutableOps.td`
- Modify: `lib/Wafer/IR/Executable/ExecutableOps.cpp`
- Modify: `lib/Wafer/IR/CMakeLists.txt`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Dialect/Wafer/Executable/committed-executable.mlir`
- Create: `test/Dialect/Wafer/Executable/invalid-committed-executable.mlir`
- Create: `unittests/Transforms/ExecutableCommitTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: move-owned fully evaluated candidate transaction with its clone/proof/staged-blob generation, complete
  completion/transport/projection records, verified target build profile and target correctness preflight.
- Produces: private-construction sealed `CommittedExecutableProgram`; only its owning outer `ProgramOutputTransaction` may
  attach it and return a transaction-scoped `compiler::StagedExecutableToken`. Nothing becomes visible in this task.

```cpp
class CommittedExecutableProgram final {
public:
  CommittedExecutableProgram() = delete;
  CommittedExecutableProgram(const CommittedExecutableProgram &) = delete;
  CommittedExecutableProgram(CommittedExecutableProgram &&) = default;
  const abi::ExecutableSemanticDigest &semanticDigest() const;

private:
  friend mlir::FailureOr<CommittedExecutableProgram>
  commitExecutable(ExecutableCandidateTransaction);
  // Owns the final clone, sealed child artifacts and the exact target-context registry.
};

mlir::FailureOr<CommittedExecutableProgram>
commitExecutable(ExecutableCandidateTransaction transaction);

namespace compiler::detail {
class ExecutableProgramOutputAccess final {
public:
  static mlir::FailureOr<compiler::StagedExecutableToken>
  attachCommittedExecutable(
      compiler::ProgramOutputTransaction &output,
      CommittedExecutableProgram committed);
private:
  ExecutableProgramOutputAccess() = delete;
};
} // namespace compiler::detail
```

`ProgramOutputTransaction` has no installed/public overload taking `CommittedExecutableProgram`. The non-installed adapter
owns the complete upper type, constructs the distinct lower executable attachment commitment/storage and invokes the outer
transaction's private attachment core. It moves the exact `VerifiedTargetCompilationContextRegistry` already carried by the
committed owner into that lower storage; it accepts no second context argument. Target/package adapters cannot construct or
reinterpret that storage.

- [ ] **Step 1: Add the full negative matrix**

  Verifier tests reject: missing/stale program semantic digest or model-interface ref; missing/invalid
  `ModelEntrypointId -> invocation roots/terminals`; residual `wafer.group`; incomplete traversal/output/reduction coverage;
  missing/forked final layout; absent/duplicate SPM or compiler DDR offset; pending completion; provisional window refs or
  `streamed_pending/streamed_planned` under committed state; missing/mismatched immutable proof/chunk/blob; invalid completion
  terminal/export; unmatched transport; missing projection; unbound/duplicate resource slot; candidate record under committed
  variant; committed record under candidate variant; mixed commit state across one executable set; overlapping/ambiguous
  shape-guard priority; rank coverage hole/duplicate; duplicate same-EntryId records; entry RankClassId coverage hole/overlap;
  per-rank shape guard; target/projection mismatch; physical geometry/narrowing failure; and final rank class spanning two
  prerequisite classes. Add a positive executable with two non-overlapping shape variants, each coherently covering all ranks,
  plus one static EntryId/module reused by two final classes and two model invocation APIs.

- [ ] **Step 2: Add atomicity GTests**

  For a two-rank, two-shape-variant source module, inject a failure independently at traversal, layout, SPM, DDR, artifact
  materialization, completion, transport, projection, pre-final/final resource view, static closure, target preflight and final
  publication, including failure only in the last variant/blob. Save source generic IR and visible output-store index and
  assert both byte-for-byte equal after each failure. Positive publication exposes all ranks, variants, ArtifactRefs and blobs
  together and leaves no candidate object reachable. Compile-time tests require `CommittedExecutableProgram` and
  `compiler::StagedExecutableToken` to be non-aggregate, non-default/direct-constructible and move-only. The sealed object exposes no
  `ModuleOp`/`Operation *` or diagnostic/IR snapshot sidecar; compile-time access tests ensure mutation between seal and attach
  is not expressible. Tests needing a print use a test-only friend that writes directly from the owned clone to a
  diagnostic-limit-bounded sink. Destroy the original `CompilationRequest` before commit and prove target contexts remain owned;
  compile-time/API tests reject any attach overload accepting a replacement registry/vector.

- [ ] **Step 3: Run and observe missing commit boundary**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt WaferUnitTests -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Dialect/Wafer/Executable/(invalid-)?committed-executable\.mlir'
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  ```

- [ ] **Step 4: Verify every gate before changing commit state**

  Inside `commitExecutable`, first build a fresh candidate-phase ResourceView from the exact transaction generation. For every
  variant, re-run traversal coverage, ordinary MLIR verification, model invocation mapping, completion reachability,
  memory/descriptor/range/state/artifact checks, cross-rank transport match, projection coverage/digest, resource/slot closure,
  the one all-entry static closure resolver and candidate-only target structure/geometry preflight from a newly built
  `VerifiedPhysicalAllocationView` on the current clone. Do not accept the Task 3B proof as an argument. This preflight emits no
  LLVM/KAD/object; committed target conversion will independently replay after attach.
  Then verify shape-guard priority/fallback determinism for the complete set. Do not accept a caller view or trust cached
  passing booleans.

- [ ] **Step 5: Form final rank classes conservatively**

  Within each variant, compare entries only inside one distributed prerequisite class. Equivalence requires same local program
  semantic digest; typed ordered slots and `StateSlotVersionRole`; per-instance shard/content/staged ArtifactRef proof;
  quant/storage/profile/capacity/arena/scope realization; state-group member/snapshot policy; resident/stream windows/chunks;
  final layout, SPM/DDR plan, transport template, projection requirements and completion exports. Any uncertainty keeps ranks
  separate. Tests require uneven TP shards or same code with different weight content to split, identical DP replicated facts
  to share, and the same EntryId/module to remain reusable across split classes. Write final `RankClassId` only after all classes
  for all variants are formed; preserve every `ExecutionInstanceId`. For each EntryId, materialize the sorted nonempty set of
  every class using that entry; do not duplicate the entry per class.

- [ ] **Step 6: Materialize durable executable owners**

  Normalize each existing realization/state-group record's `ExecutionInstanceId` coverage to `RankClassId` coverage only where
  Step 5 proved every member equivalent. Canonically create final `ResourceRealizationRecordKey` and `StreamWindowId`, consume
  the same-transaction `MaterializedImmutableArtifact` proofs/staged handles, write exact ArtifactRefs/fixed chunk tables,
  replace resident proposals or `streamed_planned` with final resident/`streamed_windows`, and rewrite all completion refs in
  one clone mutation. Do not create ResourceId, state policy, arena/placement, storage/residency choice or window schedule here.
  Preserve and verify every Task 6 `wafer.executable.invocation` mapping with its typed ModelEntrypointId and exact graph
  roots/terminals; commit must not first decide API reachability. Rewrite only refs whose final resource/window IDs are formed
  here, then reverify the same reachable subgraph and IO/state contract.
  Rebuild `ExecutableResourceView` in `FinalizedForCommit` phase after this mutation and revalidate all ranges/bindings,
  artifact/staged-store joins, state groups and invocation mappings; no provisional ref or transaction-only proof may remain
  as published IR.

- [ ] **Step 7: Flip state and compute executable identity last**

  Rerun the shared all-entry closure resolver and require identical closure units/private labels/per-entry typed static digests.
  Promote the same candidate rank/entry/resource/transport/projection/invocation records by setting every variant's
  `CommitState::Committed` only after all variants/resources/artifacts are complete. Call the one
  `ExecutableIdentityBuilder` to construct the generated message and return a private-construction verified
  `abi::ExecutableSemanticDigest`; write its IR attr representation only after verification. No `DigestAttr`/generic digest
  API, raw WCRE field registry, identity or resource policy is created during promotion.

- [ ] **Step 8: Seal and attach without publication**

  Seal the committed clone plus its child blob/index transaction and the exact input-owned target-context registry into
  `CommittedExecutableProgram`; no mutable op or raw
  staging handle is exposed. `detail::ExecutableProgramOutputAccess::attachCommittedExecutable` re-verifies the sealed generation, moves
  module/index/blobs/context registry into one lower executable commitment and returns only a scoped token used by target/package stages. It does
  not rename a root or update a trusted alias/index. Task 14's explicit executable-only completion scope may perform the outer
  commit after this token; full target/package drivers keep the same transaction private until all requested roots are
  attached and publish one trusted `ProgramDeliveryCommitRecord`. On any earlier failure source and visible store remain unchanged.

- [ ] **Step 9: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt WaferUnitTests -- -j128
  <configured-lit> -sv build/wafer-dev/test \
    --filter='Dialect/Wafer/Executable/(invalid-)?committed-executable\.mlir'
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add include/Wafer/IR/Executable/ExecutableVerification.h \
    lib/Wafer/IR/Executable/ExecutableVerification.cpp \
    include/Wafer/Transforms/Executable/CommitExecutable.h \
    lib/Wafer/Transforms/Executable/CommitExecutable.cpp \
    lib/Wafer/Transforms/Executable/ExecutableAttachmentStorage.h \
    lib/Wafer/Transforms/Executable/ExecutableProgramOutputAdapter.cpp \
    lib/Wafer/Transforms/Executable/SelectExecutableVariant.cpp \
    include/Wafer/Compiler/ExecutableIdentityBuilder.h \
    lib/Wafer/Compiler/ExecutableIdentityBuilder.cpp \
    lib/Wafer/Compiler/CMakeLists.txt \
    include/Wafer/IR/Executable/ExecutableOps.td \
    lib/Wafer/IR/Executable/ExecutableOps.cpp \
    lib/Wafer/IR/CMakeLists.txt lib/Wafer/Transforms/CMakeLists.txt \
    test/Dialect/Wafer/Executable/*committed-executable.mlir \
    unittests/Transforms/ExecutableCommitTest.cpp unittests/CMakeLists.txt
  git commit -m "Commit complete executable variants atomically"
  ```

### Task 14: `stablehlo-to-executable` Production Driver

Queue mapping: after Q0.3/Q0.4 and the independent Q5.C corpus are complete, this task's production-driver integration
closes Q7's mandatory HF commit path. Q7 may then proceed beside Q0.L; it does not wait for target artifact, package or
runtime work. Later real-model plans replay this evidence but do not create a second commit path.

**Files:**
- Modify: `include/Wafer/Pipelines/Pipelines.h`
- Modify: `lib/Wafer/Pipelines/Pipelines.cpp`
- Modify: `lib/Wafer/Pipelines/CMakeLists.txt`
- Modify: `tools/wafer-opt/wafer-opt.cpp`
- Create: `include/Wafer/Compiler/CompilationRequest.h`
- Create: `lib/Wafer/Compiler/CompilationRequest.cpp`
- Modify: `include/Wafer/Compiler/ExecutableCompilationInput.h`
- Modify: `lib/Wafer/Compiler/ExecutableCompilationInput.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Create: `test/Integration/stablehlo-to-executable.test`
- Create: `test/Integration/stablehlo-to-executable-hf-commit.test`
- Create: `test/Integration/stablehlo-to-executable-atomic-failure.test`
- Create: `test/Integration/stablehlo-to-executable-segmented.test`
- Modify: `test/CMakeLists.txt`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`

**Interfaces:**
- Consumes: verified program directory, target descriptor/topology input, configured real SPMD helper, one outer output
  transaction and, for the Q7 gate, Q5.C's source-backed `static_transformer` HF case; internally replays
  typed frontend/distributed identity and Tasks 1-13 while retaining every move-only owner.
- Produces: a staged executable token in one program-delivery transaction; the CLI's explicit executable-only scope may then
  commit one delivery root. There is no direct group-to-instr production bypass.

```cpp
void buildStablehloToExecutableTransformPipeline(mlir::OpPassManager &pm);

class CompilationRequest final {
public:
  CompilationRequest() = delete;
  CompilationRequest(const CompilationRequest &) = delete;
  CompilationRequest(CompilationRequest &&) noexcept;
  CompilationRequest &operator=(CompilationRequest &&) noexcept;
  ~CompilationRequest();
  static llvm::Expected<CompilationRequest> create(
      frontend::MaterializedFrontendProgram program,
      compiler::VerifiedTargetCompilationContextRegistry targets,
      compiler::VerifiedTargetProgramMaterializationPlan targetProgram,
      ParallelizationPolicy parallelization,
      SpmdPartitionerConfiguration spmd,
      ExecutablePlanningOptions executablePlanning,
      ImmutableArtifactMaterializationLimits artifactLimits,
      std::shared_ptr<const VerifiedCalibrationProfileSet> calibrationProfiles);
private:
  // Move-owns verified frontend IR + source root, its encoding-session token,
  // and the exact canonical target-context registry.
  class Storage;
  explicit CompilationRequest(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
};

mlir::FailureOr<compiler::StagedExecutableToken>
runStablehloToExecutableCompilation(CompilationRequest request,
                                    compiler::ProgramOutputTransaction &output);
```

`CompilationRequest::create` re-verifies the materialized frontend-model projection and the sealed target-plan/policy relations but preserves the frontend program's
private canonical-encoding owner/generation token. It privately moves the complete owner into Task 1's
`ExecutableCompilationInput`; no public API exposes the module, source coordinator or context registry.
`runStablehloToExecutableCompilation` uses only compiler-private
`detail::ExecutableProgramOutputAccess` to compare that token with the transaction owner before any clone/staging;
`ProgramOutputTransaction` exposes no public session/context/token accessor. Cross-transaction/context requests fail. Every
target/package child attachment inherits and rechecks the same token through its own owner-specific access adapter.
The driver never accepts a separate `MLIRContext`: it retains the frontend-owned context through the input and transaction-owned
working clone and cannot
re-pair a request with another context.
The factory rejects a null `calibrationProfiles`; callers always pass a verified shared set, with the canonical empty set as the
only representation of no calibration input.

- [ ] **Step 1: Add real two-instance and Q7 HF mandatory-commit vertical tests**

  Use the existing framework capture and real XLA SPMD helper, not a hand-written group fixture. Run:

  ```bash
  wafer-opt --program-pipeline=stablehlo-to-executable \
    --input-program-dir %t.input.program \
    --output-program-delivery %t.executable.delivery
  ```

  Check the output contains committed executable/variant/rank/entry/resource/completion/projection records, typed
  ModelEntrypointId invocation mappings, at least two canonical instances, accepted offsets and typed entry slots. Check no
  `wafer.group`, candidate state, default rank, unbound transport, pending/provisional window ref, planner trace or pass-only
  rank attr remains.

  In `stablehlo-to-executable-hf-commit.test`, generate Q5.C's `static_transformer` case through
  `test/Integration/Inputs/wafer_model_corpus.py`, then invoke the same production driver. Assert the HF/Megatron-style
  group reaches candidate selection and one whole-variant atomic commit; both the tiled candidate and
  `DirectFullShape` baseline are ordinary candidates under the same legality/commit sequence. Reject any debug/direct path
  that publishes group, instruction or executable output without the final committed token. Check a candidate-legality
  failure reports typed model/variant/entry/resource facts and leaves no staged executable. This test is Q7's completion
  proof and cannot be replaced by a hand-written group fixture.

- [ ] **Step 2: Add production atomic failure coverage**

  Inject a rank-specific illegal tail/transport mismatch through typed test input and check the command fails, does not write a
  committed executable module, and does not leave a partially replaced output directory. The diagnostic contains stable
  variant/entry/instance identifiers and the owning failure class. Add an ownership test that destroys the caller/request frame
  immediately after it is move-consumed; late immutable materialization and target-context transfer still succeed, while a
  stale source/context owner fails before clone/staging. Destroy the caller parse frame immediately after frontend materialization
  and prove late clone/target materialization plus failure cleanup still run before the retained MLIR context is destroyed.

- [ ] **Step 3: Add a segmented vertical fixture**

  Drive an explicit uneven four-peer segmented collective through tensor op, full traversal, fixed-capacity count/data phases, whole-entry memory, transport, projection and commit. Check `bounded_variable_segments` fails with the structured unsupported diagnostic and emits no executable.

- [ ] **Step 4: Run failing integration tests first**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  <configured-lit> -sv --show-unsupported build/wafer-dev/test \
    --filter='Integration/stablehlo-to-executable.*\.test'
  ```

  Expected before implementation: the program driver rejects the unknown pipeline.

- [ ] **Step 5: Implement the IR-only transform segment and direct orchestrator**

  `buildStablehloToExecutableTransformPipeline` consumes only a module where target environment/topology/mesh and
  model/distributed identity are already materialized. It performs the IR-local normalization/formation segment and cannot see
  `VerifiedProgramSource`, `ProgramOutputTransaction`, target-context registry, payload access or commit adapters. It is an
  internal/debug pipeline, not a production completion boundary.

  `runStablehloToExecutableCompilation` is the only production orchestrator. It move-consumes `CompilationRequest`, privately
  forms `ExecutableCompilationInput`, and immediately passes it to Task 1/3B to create a candidate transaction and full working
  clone. Target/mesh materialization, the IR-local segment, parallel/SPMD and every later mutation run only on that clone; the
  input-owned source module remains immutable for retry/failure comparison and payload ownership. The orchestrator then drives
  candidate selection and payload materialization and seals/attaches through the same outer transaction. It must not recover target, mesh, SPMD, limits or
  output policy from globals, environment variables or implicit defaults, and exposes no alternate sequence that commits after
  `wafer-select-group-tile`.

- [ ] **Step 6: Extend the program-directory driver**

  Accept exactly `stablehlo-to-executable` in `parseWaferProgramPipelineOptions`. First construct validated
  `CanonicalEncodingContext`, `VerifiedProgramOutputSink` and `ProgramOutputLimits`, then create the outer transaction with
  explicit `ExecutableOnly` completion scope; that transaction owns the encoding context through final commit. Open one
  `VerifiedProgramSource`; compiler-private `detail::FrontendProgramOutputAccess` borrows the frontend encoding session and
  invokes materialization without exposing it to the driver. Once the typed frontend owner exists, first canonicalize/revalidate
  the all-and-only `VerifiedTargetCompilationContextRegistry` from configured context inputs with zero IR mutation. Through the
  noninstalled compiler access, combine that registry, `MaterializedFrontendProgram::modelProgram()`,
  `TopologyAdmissionLimits` and the outer canonical session into `VerifiedTargetProgramMaterializationPlan`; then construct the
  mesh-dependent `ParallelizationPolicy` from the same immutable model view and plan. Resolve diagnostic aliases to typed IDs and
  construct `ExecutionSchedulePolicy`, then create `CompilationRequest` by move-owning the frontend program, exact registry,
  target plan and all policies/limits.

  `runStablehloToExecutableCompilation` seals the request, immediately creates the Task 1 working clone, materializes/reverifies
  target environment/topology/ExecutionMesh from the retained plan on that clone, and only then continues with structured
  parallel formation, real per-component SPMD, distributed assembly, Linalg/group construction and candidate entry creation.
  No pre-request or pre-clone module mutation occurs, and no move-only owner is handed to `OpPassManager`.
  `runStablehloToExecutableCompilation` attaches only a private staged token. This executable-only command then calls
  no-argument `output.commit()`; the factory-bound `ExecutableOnly` scope publishes one delivery root/commit record. Full target/
  package drivers call the staging API and do not publish here.

- [ ] **Step 7: Remove production bypasses**

  Search all tool and integration entrypoints. Existing `stablehlo-spmd-to-group` and `wafer-lower-groups-*` remain explicit replay/debug paths, but no completion claim, package/target entrypoint or production integration test may consume their output without mandatory executable commit.

  ```bash
  rg -n 'stablehlo-spmd-to-group|wafer-lower-groups-to-(selected-instr|target-llvm)' \
    tools lib test docs memory
  ```

- [ ] **Step 8: Run the focused and regression gates**

  ```bash
  cmake --build build/wafer-dev --target check-wafer WaferUnitTests -- -j128
  <configured-lit> -sv --show-unsupported build/wafer-dev/test \
    --filter='Integration/stablehlo-to-executable.*\.test|Tools/wafer-opt-(spmd|distributed-program).*\.test|Dialect/Wafer/Executable/|Transforms/(materialize-entry-traversals|finalize-entry-layouts|plan-entry-(spm|ddr)-memory|accept-physical-transport|materialize-launch-projection).*\.mlir'
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  python3 tools/check_ir_organization.py --root .
  git diff --check
  ```

  Completion requires all four integration tests, including `stablehlo-to-executable-hf-commit.test`, to execute;
  `unsupported`/skipped is not passing evidence.

- [ ] **Step 9: Update queue/memory and commit**

  Record the canonical program replay command in `memory/general_dev.md`. Mark Q0.3/Q0.4 done only when their supported
  gates execute and pass. Mark Q7 done separately only when Q5.C is complete and the HF mandatory-commit test above executes
  and passes with the required candidate/commit/failure assertions. Include exact fresh command evidence and leave
  target-artifact/package/runtime/board completion claims to downstream plans.

  ```bash
  git add include/Wafer/Pipelines/Pipelines.h lib/Wafer/Pipelines \
    include/Wafer/Compiler/CompilationRequest.h lib/Wafer/Compiler/CompilationRequest.cpp \
    include/Wafer/Compiler/ExecutableCompilationInput.h lib/Wafer/Compiler/ExecutableCompilationInput.cpp \
    tools/wafer-opt/wafer-opt.cpp include/Wafer/Transforms/Passes.td \
    test/Integration/stablehlo-to-executable*.test test/CMakeLists.txt \
    tasks/progress.md memory/general_dev.md
  git commit -m "Add the stablehlo to executable program driver"
  ```

## Global Verification

After Task 14, run the complete repository gate with the importer/SPMD features enabled:

```bash
python3 tools/check_ir_organization.py --root .
python3 tools/check_deps.py
cmake --build build/wafer-dev --target check-wafer WaferUnitTests -- -j128
ctest --test-dir build/wafer-dev --output-on-failure
<configured-lit> -sv --show-unsupported build/wafer-dev/test
git diff --check
```

Expected:

- IR/dependency checks exit 0 and C++ unit tests pass.
- all supported lit tests pass; all four `stablehlo-to-executable` tests, including the HF Q7 gate, are listed as executed,
  not unsupported.
- atomic failure tests leave source IR/output program unpublished.
- committed executable verification finds no logical group, candidate state, pending completion, unbound transport/projection/resource or cross-prerequisite rank class.
- target-artifact/package/runtime/board tests remain downstream evidence and are not claimed by this plan.

## Self-Review

- Spec coverage: full traversal, whole-entry layout, structured completion/liveness, whole-entry SPM, multi-arena DDR, segmented count/data communication, physical transport, finite launch projection, resource view, atomic commit and the production driver each have an explicit implementation owner and failing test.
- Driver coverage: every candidate policy, including `DirectFullShape`, crosses the same fixed all-entry coordinator; no pass,
  fragment callable or debug pipeline can create a committed executable independently.
- Ownership consistency: model/distributed IR provides logical identity; the same candidate-to-committed executable records own resource/rank policy and final entries/transport/projection/completion; instruction body owns schedule; WCRE has one shared `WaferABI` owner; `ExecutableResourceView` only verifies/completes existing resource records and is never serialized.
- Type consistency: `ExecutionInstanceId` is preserved from distributed instance through candidate and committed entry; final `RankClassId` only refines one prerequisite class; `DdrArenaId + placement domain` originates in typed target/resource declarations; `SlotId -> (ResourceId, StateSlotVersionRole)` is materialized exactly once at commit.
- Failure consistency: every mutation happens on an owning clone until the final verified body transfer; rank-specific failure, unsupported segmented policy or missing capability produces no partial committed output.
- Unresolved-marker scan: this plan contains no placeholder marker, unnamed file, open ABI field or task/stage ID in a proposed code object.
- Naming scan: before implementation completion run `rg -n 'Q0\.3|Q0\.4|task[_-]?[0-9]+|stage[_-]?[0-9]+' include lib tools test` and inspect every hit; roadmap IDs may occur only in task/plan documentation.
