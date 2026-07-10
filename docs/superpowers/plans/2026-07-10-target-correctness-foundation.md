# Target Correctness Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement this plan in the dependency order below. Every implementation task starts with a
> failing test and ends with a separate commit.

**Goal:** 消除 target LLVM recursive flatten silent miscompile，建立不可伪造的 entry-core/clone-dependency conversion
request，并让 instruction verifier、whole-executable preflight、target lowering 和 KAD 共用同一 physical geometry与
target narrowing证明。

**Architecture:** Task 1是可立即落地的临时fail-closed hotfix。pre-Whole阶段先实现共享structural geometry core；Whole
从candidate transaction内部fresh ResourceView构造`VerifiedPhysicalAllocationView`，只做structure/geometry preflight并原子
commit。commit后正式target路径才从sealed executable构造`wafer::conversion::VerifiedTargetConversionRequest`，重放closure、
entry boundary、external symbol和geometry，再用DialectConversion保持SCF/CF/call结构。任何public aggregate、raw symbol、
caller range或pass-local boolean都不能授权preflight/lowering。

**Tech Stack:** C++17、MLIR/LLVM 20 DialectConversion、SCF/CF/Func/Arith/LLVM dialect、WaferABI、
WaferCompilerIdentity、lit、GTest、host-clang semantic execution test。

## Global Constraints

- 设计owner是`tasks/11-instruction-ir.md`、`tasks/14-target-llvm-golden-packet.md`和
  `tasks/16-verification-plan.md`；本计划只拆实现。
- Task 1可以在shared foundation之前落地，但只是临时安全修复。Task 4等待target-artifact shared Proto/WCRE/context和
  typed candidate resource boundary；Tasks 2、3、5还必须等待Whole atomic executable commit以及target shared Kernel ABI
  value/schema、external symbol registry和static closure proof完成。
- 正式依赖方向固定为`WaferTargetLegality -> WaferIR/WaferCompilerIdentity/WaferABI`、
  `WaferInstrToTargetLLVM -> WaferTargetLegality/WaferCompilerIdentity/WaferABI`，
  `WaferTargetArtifacts -> WaferInstrToTargetLLVM`；conversion library不得反向链接`WaferTargetArtifacts`。
- production target conversion只接受sealed committed executable投影形成的verified request。candidate preflight只接受
  private `VerifiedPhysicalAllocationView`且不产LLVM/KAD/artifact。replay pass只能通过受限test/debug adapter重放同一
  verifier，不能生成artifact或成为production bypass。
- unsupported container、entry/clone kind、external symbol、geometry或allocation必须在clone mutation/tool launch前失败。
- target output不得残留Wafer、Func、SCF、CF、Arith或symbolic MemRef op。
- 所有`i64 -> i32/u32/u16`只消费verified geometry field，禁止cast截断或在call builder中重算。
- DTE在accepted transport/projection和typed control/status slot尚未接入request前保持structured unsupported；不能恢复裸
  logical-peer CRT call。

```text
Pipeline position:
- Upstream artifact / IR:
  pre-commit candidate current-generation ResourceView或sealed committed executable projection，memory-planned
  `wafer.instr.*`、accepted offsets/transport/projection、exact VerifiedTargetCompilationContext；post-commit另消费complete
  static closure/core/clone proofs。
- Current stage responsibility:
  先提供candidate/committed共用geometry core；commit后构造verified entry-core或clone-dependency request并保持
  control-flow/call结构完成full conversion；从同一op/root/allocation/prototype事实证明physical access与target ABI narrowing。
- Output artifact / IR:
  owned LLVM dialect module、converted entry handle relation和typed target CRT calls；无illegal residual dialect、raw
  external symbol或未验证geometry。
- Downstream consumer:
  target artifact builder的core/clone prelink、KAD cross-check、device link和ELF verification。
- User-level driver / named pipeline:
  production由`stablehlo-to-executable`的target-artifact stage调用；`--wafer-lower-instr-to-target-llvm`只作replay。
- Explicit non-goals:
  不形成module partition、KAD/ELF/set/package，不重新做layout/SPM/DDR/transport planning，不从symbol/name恢复ABI。
- Completion gate:
  false branch、loop trip count、nested branch和private call的实际执行语义正确；EntryCore与CloneDependency都走唯一
  verified API；stale/forged boundary、root/access/range/prototype失败；所有production narrowing字段复用同一proof。
```

## Dependency Order

```text
Task 1 temporary fail-closed hotfix (may land immediately)
target shared identity/schema/context + typed candidate executable prerequisites
  -> Task 4 shared structural/target geometry core
  -> Whole plan candidate allocation-view adapter, target preflight and atomic commit
  -> Task 2 sealed committed conversion request/boundary foundation
  -> Task 3 structure-preserving conversion + committed allocation-view adapter
  -> Task 5 complete geometry-family conversion coverage
  -> target plan Task 6 KAD builder and target-artifact conversion integration
  -> Task 6 verification/queue update
```

Task 1 does not satisfy the prerequisite for Tasks 2-5 and must be replaced, not retained as a broad production rejection gate.
Task 6 is deliberately delayed until the downstream target plan has consumed the conversion result and proved the post-conversion
KAD/artifact relation; an isolated conversion-library test cannot close the queue item.

---

### Task 1: Temporary Fail-Closed Target Hotfix

**Files:**
- Modify: `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp`
- Modify: `test/Transforms/lower-instr-to-target-llvm-failure.mlir`
- Create: `unittests/Transforms/TargetLLVMTransactionTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: current replay module before any LLVM skeleton creation.
- Produces: temporary structured rejection for any region/multiblock/call shape the old recursive walker would flatten; failure
  leaves source byte-for-byte unchanged.

- [ ] **Step 1: Add branch, loop, multiblock and call failures**

  Add `scf.if`, `scf.for`, two-block `cf.cond_br` and `func.call` split cases. Require
  `unsupported_target_structure: verified structure-preserving conversion is not available` and a GTest comparing generic IR
  before/after failure.

- [ ] **Step 2: Confirm the current path flattens or mutates**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt WaferUnitTests -- -j128
  /root/miniconda3/bin/lit -sv build/wafer-dev/test/Transforms/lower-instr-to-target-llvm-failure.mlir
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  ```

- [ ] **Step 3: Reject before any mutation**

  Add one temporary preflight as the first pass action. It rejects nested regions, multiblock functions and calls before
  constructing an LLVM function or erasing an op. It must not grow into the long-term verifier, inspect names or accept an
  override flag.

- [ ] **Step 4: Commit the safety fix**

  ```bash
  git add lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp \
    test/Transforms/lower-instr-to-target-llvm-failure.mlir \
    unittests/Transforms/TargetLLVMTransactionTest.cpp unittests/CMakeLists.txt
  git commit -m "Fail closed before unsupported target conversion"
  ```

### Task 2: Verified Conversion Request and Boundary Foundation

**Prerequisite:** target shared Tasks 1-5, shared runtime-safe `Tx81CommandAbi`/immutable command-value foundation and the Whole
plan's atomic committed executable owner are complete. The compiler command-view type does not exist before this task because
it depends on committed executable ops. Do not start from candidate IR or copy artifact-private types.

**Files:**
- Create: `include/Wafer/Conversion/WaferInstrToTargetLLVM/WaferInstrToTargetLLVM.h`
- Create: `include/Wafer/Compiler/Tx81CommandBuilder.h`
- Create: `lib/Wafer/Compiler/Tx81CommandBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/VerifiedTargetConversionRequest.cpp`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/VerifiedTargetEntryBoundary.cpp`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/CMakeLists.txt`
- Modify: `lib/Wafer/Conversion/CMakeLists.txt`
- Create: `unittests/Conversion/VerifiedTargetConversionRequestTest.cpp`
- Create: `unittests/Compiler/Tx81CommandBuilderTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: isolated module clone, `VerifiedStaticClosureUnit` or `VerifiedCloneableDependency`, exact
  `VerifiedTargetCompilationContext`, and for entry cores non-forgeable `VerifiedTargetEntryBoundary` values constructed by a
  compiler-private adapter from the sealed executable/resource slots, prototype, projection dependency, completion exports and
  command views.
- Produces: one move-only/non-aggregate request with closed kind `EntryCore | CloneDependency`.

```cpp
namespace wafer::conversion {

class VerifiedTargetEntryBoundary final {
public:
  VerifiedTargetEntryBoundary() = delete;
  VerifiedTargetEntryBoundary(const VerifiedTargetEntryBoundary &) = delete;
  VerifiedTargetEntryBoundary(VerifiedTargetEntryBoundary &&) noexcept;
  ~VerifiedTargetEntryBoundary();
  uint32_t requestLocalHandle() const;
  llvm::ArrayRef<VerifiedTargetEntrySlotView> orderedSlots() const;
private:
  friend class compiler::detail::TargetEntryBoundaryAccess;
  class Impl;
  explicit VerifiedTargetEntryBoundary(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

enum class TargetConversionRequestKind { EntryCore, CloneDependency };

class VerifiedTargetConversionRequest final {
public:
  VerifiedTargetConversionRequest() = delete;
  VerifiedTargetConversionRequest(const VerifiedTargetConversionRequest &) = delete;
  VerifiedTargetConversionRequest(VerifiedTargetConversionRequest &&) noexcept;
  ~VerifiedTargetConversionRequest();
  TargetConversionRequestKind kind() const;
private:
  friend mlir::FailureOr<VerifiedTargetConversionRequest>
  verifyEntryCoreConversionRequest(
      mlir::OwningOpRef<mlir::ModuleOp>,
      llvm::SmallVector<VerifiedTargetEntryBoundary>,
      const compiler_identity::VerifiedStaticClosureUnit &,
      const compiler::VerifiedTargetCompilationContext &);
  friend mlir::FailureOr<VerifiedTargetConversionRequest>
  verifyCloneDependencyConversionRequest(
      mlir::OwningOpRef<mlir::ModuleOp>,
      const compiler_identity::VerifiedCloneableDependency &,
      const compiler::VerifiedTargetCompilationContext &);
  class Impl;
  explicit VerifiedTargetConversionRequest(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

mlir::FailureOr<VerifiedTargetConversionRequest>
verifyEntryCoreConversionRequest(
    mlir::OwningOpRef<mlir::ModuleOp> isolatedInput,
    llvm::SmallVector<VerifiedTargetEntryBoundary> nonemptyBoundaries,
    const compiler_identity::VerifiedStaticClosureUnit &closure,
    const compiler::VerifiedTargetCompilationContext &context);

mlir::FailureOr<VerifiedTargetConversionRequest>
verifyCloneDependencyConversionRequest(
    mlir::OwningOpRef<mlir::ModuleOp> isolatedInput,
    const compiler_identity::VerifiedCloneableDependency &dependency,
    const compiler::VerifiedTargetCompilationContext &context);

} // namespace wafer::conversion
```

`VerifiedTargetEntrySlotView` is read-only and includes SlotId, source argument ordinal/type, target prototype type, typed
ResourceId/role/access/state-version role, alignment, capacity/alias relation and exact owner generation. No API accepts a caller
vector of these views as authority. The request factory move-consumes the nonempty boundary vector; it does not borrow or copy
move-only boundary owners. The compiler-private boundary factory reads the sealed executable projection and rejects
missing/duplicate/noncontiguous slots, wrong source argument, stale generation or typed boundary-relation mismatch. It does
not accept or construct a `KernelAbiDescriptor`; conversion consumes no KAD.
Both factories move-consume an `OwningOpRef<ModuleOp>` and retain it inside the request. There is no borrowed raw-module
overload, so caller mutation/destruction after sealing is not expressible; failure destroys the owned isolated clone. Because
`OwningOpRef` does not own its `MLIRContext`, each sealed boundary/clone-dependency proof privately carries the committed
executable's nonsemantic context-lifetime share. The factory checks every share against `isolatedInput->getContext()` and retains
it in the request; the converted owner inherits it. No callable overload accepts a caller context or lifetime token.

- [ ] **Step 1: Add construction/forgery failures**

  Require compile-time non-aggregate/non-default/non-copyable properties. Runtime negatives cover zero EntryCore boundaries,
  a CloneDependency with any boundary, swapped slot/resource, stale module generation, closure root mismatch, wrong target context,
  recursive/effectful/address-taken clone dependency, cross-request boundary reuse, mixed MLIR-context shares and a module whose
  caller context owner dies without transferring the sealed lifetime share.

- [ ] **Step 2: Implement two private verifier paths**

  EntryCore requires at least one exact boundary and complete closure root/entry/resource/completion relation. CloneDependency
  requires zero boundaries and replays pure-private helper/immutable non-address-significant global, no escape/effect/recursive
  SCC and fixed hidden-linkonce/COMDAT policy. Both scan the complete isolated module and bind its mutation generation plus target
  context owner token.

  Define the MLIR-aware `compiler_identity::VerifiedTx81CommandView` and its nonpublic resolver storage in
  `WaferCompilerIdentity` here, after committed `ExecutableEntryOp` exists. It wraps only a shared `abi::VerifiedTx81CommandValue`,
  committed instruction-occurrence key and verified slot bindings; it owns no C layout or second validator. The base resolver
  structurally rejects command-requiring families until their reviewed target-plan arms land, while direct-call entries form a
  valid request with an empty command-view relation. Target Tasks 9/10 modify this sole resolver for low-precision/DTE; they do
  not create another view type or change direct-call conversion.

- [ ] **Step 3: Fix library direction**

  `WaferInstrToTargetLLVM` links `WaferTargetLegality`, `WaferCompilerIdentity`, `WaferABI`, `WaferIR` and MLIR conversion
  libraries. It does not link `WaferTargetArtifacts`. Add `tools/check_deps.py` coverage for the forbidden reverse edge.

- [ ] **Step 4: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  python3 tools/check_deps.py
  git add include/Wafer/Conversion/WaferInstrToTargetLLVM \
    include/Wafer/Compiler/Tx81CommandBuilder.h lib/Wafer/Compiler/Tx81CommandBuilder.cpp \
    lib/Wafer/Compiler/CMakeLists.txt \
    lib/Wafer/Conversion/WaferInstrToTargetLLVM lib/Wafer/Conversion/CMakeLists.txt \
    unittests/Conversion/VerifiedTargetConversionRequestTest.cpp \
    unittests/Compiler/Tx81CommandBuilderTest.cpp unittests/CMakeLists.txt
  git commit -m "Add verified target conversion requests"
  ```

### Task 3: Structure-Preserving Conversion and Typed Calls

**Files:**
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/WaferInstrToTargetLLVM.cpp`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetAddressResolver.cpp`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetCallBuilder.cpp`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/CommittedPhysicalAllocationAccess.cpp`
- Create: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetMetadataCleanup.cpp`
- Modify: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/CMakeLists.txt`
- Modify: `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Modify: `include/Wafer/Transforms/Passes.td`
- Create: `test/Transforms/lower-instr-to-target-llvm-control-flow.mlir`
- Create: `test/Transforms/Inputs/target-control-flow-runtime.c`
- Modify: `test/lit.cfg.py`
- Modify: `test/lit.site.cfg.py.in`

**Interfaces:**

```cpp
namespace wafer::conversion {
class ConvertedTargetEntryHandle final {
public:
  ConvertedTargetEntryHandle(const ConvertedTargetEntryHandle &) = default;
  ConvertedTargetEntryHandle &operator=(
      const ConvertedTargetEntryHandle &) = default;
  uint32_t requestLocalHandle() const;
private:
  friend class ConvertedTargetLLVMModule;
  class OwnerToken;
  ConvertedTargetEntryHandle(std::shared_ptr<const OwnerToken>, uint32_t);
  std::shared_ptr<const OwnerToken> owner_;
  uint32_t ordinal_;
};

struct ConvertedTargetEntrySummary {
  ConvertedTargetEntryHandle handle;
  uint32_t requestLocalHandle;
  // Typed IDs and diagnostic-only converted signature summary; no MLIR handle,
  // boundary reference or verifier authority.
};

class ConvertedTargetLLVMModule final {
public:
  ConvertedTargetLLVMModule() = delete;
  ConvertedTargetLLVMModule(const ConvertedTargetLLVMModule &) = delete;
  ConvertedTargetLLVMModule(ConvertedTargetLLVMModule &&) noexcept;
  ~ConvertedTargetLLVMModule();
  uint32_t entryCount() const;
  llvm::Expected<ConvertedTargetEntrySummary>
  entrySummary(uint32_t ordinal) const;
private:
  friend class compiler_identity::detail::KernelAbiBuildAccess;
  friend class target::detail::ConvertedTargetModuleAccess;
  friend mlir::FailureOr<ConvertedTargetLLVMModule>
  convertVerifiedWaferInstrModuleToTargetLLVM(
      VerifiedTargetConversionRequest);
  class Impl;
  explicit ConvertedTargetLLVMModule(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

mlir::FailureOr<ConvertedTargetLLVMModule>
convertVerifiedWaferInstrModuleToTargetLLVM(
    VerifiedTargetConversionRequest request);
} // namespace wafer::conversion
```

There is no production overload accepting raw `ModuleOp`, target symbol strings, loose profile/options or public entry arrays.
The result exposes no `ModuleOp`, `LLVMFuncOp`, `Operation *`, authoritative borrowed entry view or mutator. `entrySummary`
returns a bounded value containing an owner-bound handle, not a reference. The KAD API requires both a live
`ConvertedTargetLLVMModule &` and that owner's handle; cross-module/stale/forged ordinal fails before private access. Only the
non-installed KAD/prelink access adapters may borrow the owned module, boundary and exact handle-to-function mapping during one
scoped call; neither adapter can return or retain MLIR/boundary handles past it.

- [ ] **Step 1: Add structure and semantic execution tests**

  Cover constant-false `scf.if` with distinct effects, a three-iteration `scf.for`, nested branch, two-block `cf.cond_br` and a
  private helper call. FileCheck requires corresponding LLVM blocks/branches/call and no Wafer/SCF/CF/Func residual. Translate,
  host-compile with a deterministic typed CRT stub and execute; the trace must prove branch choice, exact trip count and helper
  execution, not just emitted call count. Add a CloneDependency case with zero public entries and no KAD output.
  Compile-time access tests reject public module/function/boundary handles and any authoritative API taking an entry handle
  without the live converted-module owner. Cross-module handles fail; destroying/moving the owner cannot leave a borrowed proof
  view, while a module move preserves its pimpl owner token.

- [ ] **Step 2: Make target call construction consume registry entries**

  ```cpp
class TargetCallBuilder final {
public:
  TargetCallBuilder(
      mlir::LLVMTypeConverter &converter,
      const abi::VerifiedExternalSymbolRegistry &symbols,
      const compiler::VerifiedTargetCompilationContext &context);
  mlir::LogicalResult replaceWithDirectCall(
      mlir::Operation *op,
      const abi::VerifiedExternalSymbolEntry &callee,
      const target::VerifiedInstructionGeometry &geometry,
      mlir::ValueRange nonGeometryArgs,
      mlir::ConversionPatternRewriter &rewriter);
};
```

Resolve a closed prototype by typed instruction family/kind through the exact request context and join the same op occurrence
to its registered direct-call family. `replaceWithDirectCall` rejects a callee from another registry/context generation, op
occurrence mismatch or argument/prototype mismatch. It never accepts `StringRef`, invents a declaration or uses symbol spelling
as semantic evidence. Families whose shared ABI requires a `VerifiedTx81CommandView` remain structured unsupported here;
target-plan command-family tasks add a distinct typed command-call method and patterns without changing this direct-call API or
letting a raw command aggregate enter conversion.

- [ ] **Step 3: Apply full conversion on the request-owned clone**

  First use `conversion::detail::CommittedPhysicalAllocationAccess` to build a `VerifiedPhysicalAllocationView` from the sealed
  request and exact target context. Derive and target-verify every current instruction before conversion; any rewrite invalidates
  those proofs, so patterns either consume them before mutation or rederive for the new current op. No candidate ResourceView or
  pre-commit proof is available here.

  Register exhaustive production leaf patterns plus standard SCF-to-CF, Arith/Func/CF-to-LLVM patterns. DTE accepts only a
  complete verified command view with endpoint/channel/FSM/receiver/status/completion; otherwise its pattern rejects before
  mutation. Rewrite EntryCore
  boundaries from exact ordered slots; CloneDependency has no runtime entry boundary. Private helpers use standard conversion
  only when the request closure proof owns them.

  Use a full `ConversionTarget` that marks Wafer, Arith, CF, Func, MemRef and SCF illegal. Cleanup symbolic allocation/view ops
  only after all users are replaced; never lower SPM symbols to host heap. Residual/unknown metadata fails.

- [ ] **Step 4: Replace the temporary broad rejection gate**

  The replay pass first uses the test/debug boundary adapter to construct a verified request, calls the sole conversion API and
  only on success moves the returned body. Delete Task 1's broad SCF/CF/call rejection and the old recursive walk/single-block
  builder. Production artifact code never calls the replay adapter.

- [ ] **Step 5: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target wafer-opt -- -j128
  /root/miniconda3/bin/lit -sv \
    build/wafer-dev/test/Transforms/lower-instr-to-target-llvm.mlir \
    build/wafer-dev/test/Transforms/lower-instr-to-target-llvm-failure.mlir \
    build/wafer-dev/test/Transforms/lower-instr-to-target-llvm-control-flow.mlir
  git add lib/Wafer/Conversion/WaferInstrToTargetLLVM \
    lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp lib/Wafer/Transforms/CMakeLists.txt \
    include/Wafer/Transforms/Passes.td test/Transforms test/lit.cfg.py test/lit.site.cfg.py.in
  git commit -m "Preserve structure in verified target conversion"
  ```

### Task 4: Shared Physical Geometry and Allocation Proof

**Files:**
- Create: `include/Wafer/IR/InstructionGeometry.h`
- Create: `lib/Wafer/IR/Common/InstructionGeometry.cpp`
- Modify: `lib/Wafer/IR/CMakeLists.txt`
- Create: `include/Wafer/Target/TargetLegality.h`
- Create: `lib/Wafer/Target/TargetLegality.cpp`
- Create: `lib/Wafer/Target/CMakeLists.txt`
- Modify: `lib/Wafer/CMakeLists.txt`
- Create: `unittests/Dialect/Wafer/InstructionGeometryTest.cpp`
- Create: `unittests/Target/TargetLegalityTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: one current instruction op, exact target context and an owner-backed candidate or committed physical-allocation view.
- Produces: recomputable structural geometry, then target/allocation-verified geometry bound to the same op and mutation
  generation. Neither is IR, serializable state or a public aggregate.

```cpp
namespace wafer::geometry {

enum class PhysicalAccessKind : uint8_t { Read, Write, ReadWrite };

class InstructionGeometry final {
public:
  InstructionGeometry() = delete;
  InstructionGeometry(const InstructionGeometry &) = delete;
  InstructionGeometry(InstructionGeometry &&) noexcept;
  ~InstructionGeometry();
  llvm::ArrayRef<StructuralPhysicalAccessView> accesses() const;
  llvm::ArrayRef<StructuralGeometryFieldView> fields() const;
private:
  friend mlir::FailureOr<InstructionGeometry>
  deriveInstructionGeometry(mlir::Operation *);
  class Impl;
  explicit InstructionGeometry(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

mlir::FailureOr<InstructionGeometry>
deriveInstructionGeometry(mlir::Operation *op);

} // namespace wafer::geometry

namespace wafer::target {

enum class TargetIntegerField : uint8_t {
  AddressU64, DescriptorU32, ShapeU16, CountU32
};

class TargetGeometryLimits final {
public:
  TargetGeometryLimits() = delete;
  TargetGeometryLimits(const TargetGeometryLimits &) = delete;
  TargetGeometryLimits(TargetGeometryLimits &&) noexcept;
  ~TargetGeometryLimits();
private:
  friend llvm::Expected<TargetGeometryLimits>
  createTargetGeometryLimits(
      const compiler::VerifiedTargetCompilationContext &);
  class Impl;
  explicit TargetGeometryLimits(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

class VerifiedPhysicalAllocationView final {
public:
  VerifiedPhysicalAllocationView() = delete;
  VerifiedPhysicalAllocationView(const VerifiedPhysicalAllocationView &) = delete;
  VerifiedPhysicalAllocationView(VerifiedPhysicalAllocationView &&) noexcept;
  ~VerifiedPhysicalAllocationView();
private:
  friend class compiler::detail::CandidatePhysicalAllocationAccess;
  friend class conversion::detail::CommittedPhysicalAllocationAccess;
  friend class detail::TestPhysicalAllocationAccess;
  class Impl;
  explicit VerifiedPhysicalAllocationView(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

class VerifiedInstructionGeometry final {
public:
  VerifiedInstructionGeometry() = delete;
  VerifiedInstructionGeometry(const VerifiedInstructionGeometry &) = delete;
  VerifiedInstructionGeometry(VerifiedInstructionGeometry &&) noexcept;
  ~VerifiedInstructionGeometry();
  llvm::ArrayRef<PhysicalAccessView> accesses() const;
  llvm::ArrayRef<EncodedGeometryFieldView> encodedFields() const;
private:
  friend mlir::FailureOr<VerifiedInstructionGeometry>
  verifyInstructionGeometryForTarget(
      mlir::Operation *, const geometry::InstructionGeometry &,
      const TargetGeometryLimits &, const VerifiedPhysicalAllocationView &);
  class Impl;
  explicit VerifiedInstructionGeometry(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

mlir::FailureOr<VerifiedInstructionGeometry>
verifyInstructionGeometryForTarget(
    mlir::Operation *op, const geometry::InstructionGeometry &geometry,
    const TargetGeometryLimits &limits,
    const VerifiedPhysicalAllocationView &allocations);

} // namespace wafer::target
```

Each `PhysicalAccessView` carries operand/prototype ordinal, exact root identity, canonical root-to-view relation,
`Read|Write|ReadWrite`, and checked half-open `[begin,end)` relative interval. Each encoded field carries the closed CRT prototype
argument ordinal, exact u64 value and target integer kind. Views are diagnostic/read-only values; no production verifier accepts
one back as authority. `VerifiedInstructionGeometry` retains op identity plus mutation generation and becomes stale after any
rewrite. Candidate and committed allocation views use separate private adapters but the same class/core; the only fake access
class is compiled into the unit-test target.

- [ ] **Step 1: Add arithmetic, relation and anti-forgery tests**

  Cover add/multiply/ceildiv overflow, zero span, descriptor product mismatch, nondivisible bytes/elements, U32/U16 overflow,
  access beyond root, wrong root/view relation, swapped lhs/rhs operand, same shape with a different root, read recorded as write,
  stale proof after rewrite and cross-owner allocation view. Compile-time checks require all proof/limits/view types to be
  non-aggregate and non-direct-constructible.

- [ ] **Step 2: Implement checked family relations**

  Use structured memref/layout/index relations and shared physical layout helpers. Explicitly cover RDMA/WDMA stride windows,
  gather/scatter equal coverage, convert equal element count with different byte widths, GEMM M/K/N/batch, conv/pool/TDMA shape
  arrays, low-precision packed data/scale/scratch spans and DTE bound slice/capacity. No name or buffer ordinal heuristic.

- [ ] **Step 3: Bind limits and allocations to verified owners**

  `createTargetGeometryLimits` derives field widths/capacities only from the exact `VerifiedTargetCompilationContext` and cannot
  take loose caller maxima. Task 4 defines private adapter hooks only. The Whole plan's candidate adapter later replays its exact
  transaction generation/fresh ResourceView; the post-commit conversion adapter replays the sealed request. Neither accepts a
  map callback, raw range list or caller-provided ResourceView.

- [ ] **Step 4: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target WaferUnitTests -- -j128
  ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
  git add include/Wafer/IR/InstructionGeometry.h lib/Wafer/IR/Common/InstructionGeometry.cpp \
    lib/Wafer/IR/CMakeLists.txt include/Wafer/Target/TargetLegality.h \
    lib/Wafer/Target/TargetLegality.cpp lib/Wafer/Target/CMakeLists.txt lib/Wafer/CMakeLists.txt \
    unittests/Dialect/Wafer/InstructionGeometryTest.cpp unittests/Target/TargetLegalityTest.cpp \
    unittests/CMakeLists.txt
  git commit -m "Add owner-bound instruction geometry verification"
  ```

### Task 5: Complete Geometry Family Coverage

**Files:**
- Modify: `lib/Wafer/IR/Instr/InstructionOps.cpp`
- Modify: `lib/Wafer/IR/Instr/DTEOps.cpp`
- Modify: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/WaferInstrToTargetLLVM.cpp`
- Modify: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetAddressResolver.cpp`
- Modify: `lib/Wafer/Conversion/WaferInstrToTargetLLVM/TargetCallBuilder.cpp`
- Create: `test/Dialect/Wafer/Instr/Instruction/invalid-physical-geometry.mlir`
- Create: `test/Transforms/lower-instr-to-target-llvm-narrowing.mlir`

**Interfaces:**
- Consumes: structural `InstructionGeometry`, exact request context/allocations/prototype and current op generation.
- Produces: `VerifiedInstructionGeometry` used directly by address/call construction plus the exact converted entry mapping and
  request boundary facts consumed later by the target plan's KAD builder.

- [ ] **Step 1: Add the production family matrix**

  Cover every production instruction family from tasks/11 and tasks/14, including boundary values and one-above for every
  U32/U16 field, root OOB, descriptor mismatch, low-precision packed/scale span mismatch and stale proof. DTE remains an expected
  structured failure until its complete verified context lands.

- [ ] **Step 2: Reuse structural derivation in op verifiers**

  ODS/C++ verifiers call `deriveInstructionGeometry` for target-independent arithmetic/type/layout relations. They do not create
  target context, accepted allocation or persisted proof. Remove duplicate family arithmetic after parity tests pass.

- [ ] **Step 3: Verify target-specific facts once per current op**

  Committed conversion request preflight creates `TargetGeometryLimits` and a committed
  `VerifiedPhysicalAllocationView`, derives and target-verifies every instruction, then invalidates all proofs if any rewrite
  occurs. Conversion patterns rederive after their own prerequisite rewrite if needed; they never reuse stale objects. Whole's
  candidate adapter is a separate earlier implementation task and cannot pass its proof/result into this path.

- [ ] **Step 4: Emit only verified fields and registered prototypes**

  `TargetAddressResolver` uses the proof's exact root/view/access interval. `TargetCallBuilder` uses
  `EncodedGeometryFieldView` by prototype argument ordinal and the same `VerifiedExternalSymbolEntry`; it has no
  `constantI32(int64_t)`, string field lookup or independent shape arithmetic. Return the explicit request-local entry-handle to
  converted-function mapping and preserve the read-only boundary views needed downstream. The conversion library does not
  construct a KAD or reference the target-artifact-private `EntryAbiContractKey`; target Task 6 independently builds and verifies
  KADs from the converted mapping plus the same sealed executable boundary facts.

- [ ] **Step 5: Run and commit**

  ```bash
  cmake --build build/wafer-dev --target check-wafer -- -j128
  /root/miniconda3/bin/lit -sv \
    build/wafer-dev/test/Dialect/Wafer/Instr \
    build/wafer-dev/test/Transforms/lower-instr-to-target-llvm-narrowing.mlir \
    build/wafer-dev/test/Transforms/lower-instr-to-target-llvm-control-flow.mlir
  git add lib/Wafer/IR/Instr lib/Wafer/Conversion/WaferInstrToTargetLLVM \
    test/Dialect/Wafer/Instr test/Transforms
  git commit -m "Enforce verified geometry before target narrowing"
  ```

### Task 6: Completion Gate and Queue Update

**Files:**
- Modify: `tasks/progress.md`
- Modify: `memory/bugs.md` only if a new repeatable failure mode was found.
- Modify: `memory/general_dev.md` only for a stable command/workflow change.

- [ ] **Step 1: Run fresh full verification**

  ```bash
  git diff --check
  python3 tools/check_ir_organization.py --root .
  python3 tools/check_deps.py
  cmake --build build/wafer-dev --target check-wafer -- -j128
  ctest --test-dir build/wafer-dev --output-on-failure
  /root/miniconda3/bin/lit -sv --show-unsupported build/wafer-dev/test
  ```

  Confirm structure/geometry/semantic execution tests actually ran and were not unsupported.

- [ ] **Step 2: Record exact evidence**

  The target correctness queue item becomes done only when Task 1's broad gate is gone, both EntryCore and CloneDependency use
  the sole verified API, all production families either consume verified geometry or fail before mutation, and target artifact
  integration consumes the result without a raw request/symbol/range bypass.

- [ ] **Step 3: Commit queue/memory changes**

  ```bash
  git add tasks/progress.md memory
  git commit -m "Record target correctness completion evidence"
  ```

## Self-Review

- Pipeline coverage: temporary containment, shared prerequisites, entry/clone requests, structure semantics, physical access,
  narrowing, external symbols, KAD cross-check and downstream artifact consumption all have explicit owners.
- Dependency safety: conversion does not link target artifacts; target artifacts call one conversion API.
- Proof safety: request, boundary, limits, resolver and geometry are non-aggregate/private-construction and bind exact owner/
  generation. Public diagnostic views are never accepted back as authority.
- Large-load behavior: closure/geometry walks are bounded by target build/admission limits from shared plans; no whole-module text,
  raw range side table or unbounded diagnostic snapshot is retained.
- Completion semantics: branch/loop/call correctness is executed, not inferred from call counts or FileCheck alone.
