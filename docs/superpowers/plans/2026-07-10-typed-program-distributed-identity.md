# Typed Program and Distributed Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 program directory、target snapshot 和 SPMD 输出规范化成 typed model/distributed/candidate identity，并删除 default rank 0 与名字/sidecar 语义通道。

**Architecture:** Frontend import transaction materialize model interface、program graph、state groups、bounded invocation
fields and quant semantics；typed parallel formation materializes components/edges before per-component SPMD；target import
materializes environment/topology/mesh and low-precision capabilities；distributed assembly materializes canonical instances/
classes/shards。Candidate creation then constructs uncommitted executable rank/entry/resource/state-group records from
canonical coordinates; local lowering reads enclosing typed identity.

**Tech Stack:** Wafer ODS/TableGen、MLIR Symbol/SymbolTable、StableHLO/Shardy、LLVM JSON only at program-container import boundary、NPY payload verifier、lit/GTest。

## Global Constraints

- 设计 owner：`tasks/02-frontend-stablehlo-program.md`、`tasks/03-shardy-spmd.md`、`tasks/04-topology-execution-mesh.md`、`tasks/07-tile-region.md`。
- Program-directory JSON/NPY is loader serialization, not downstream IR semantics.
- Distributed identity first encodes local `(ComponentId, partition_coordinate, replica_coordinate)` keys; after the complete
  distributed-program digest is verified, public `ExecutionInstanceId` is `(DistributedProgramSemanticId, local key)`. The
  digest preimage never contains its own derived digest; flat rank is derived only.
- `axis_roles` is typed and never inferred from axis names.
- Candidate records use `CommitState::Candidate`; target/package/runtime reject them.
- Per-rank-static specialization must cover exactly one coordinate; rank-parametric body retains partition/replica SSA/ABI.
- Before Task 1, complete target-artifact shared Tasks 1-5: runtime-safe Proto/WCRE/schema, model/distributed/KAD value types,
  compiler IR-to-generated-message builders and `VerifiedTargetCompilationContext`. This plan consumes those APIs and never
  creates raw WCRE fields.
- `ModelEntrypointId`/`ModelProgramMemberId`/`InvocationPolicyFieldId`/`StateConsistencyGroupId` and `ComponentId` are typed scoped IDs; no
  function/group/field/expert name is an identity input.
- model identity uses a two-phase owner rule: record-11 encodes local resource/dimension keys; public ResourceId/DimId are
  created only after `ModelInterfaceSemanticId` and include that owner. Same local key under different models never aliases.
- `VerifiedProgramSource` owns the open program root capability and expected source refs from first load through artifact
  materialization. Downstream never receives a checked path or reopens outside that capability.
- `MaterializedFrontendProgram` is the only owner of the frontend-created `MLIRContext`, verified module and source coordinator. `CompilationRequest`
  move-consumes it, and the Whole plan's private `ExecutableCompilationInput` owner carries both through candidate cloning and
  payload materialization. No production API extracts a raw `ModuleOp` or source capability; the CLI's
  `stablehlo-to-executable` option selects a direct owner-aware driver, not a pure `OpPassManager` pipeline.
- Registered StableHLO/MLIR Quant types and FP8 types are the semantic source. Wafer attrs only add block-scale facts that
  those dialects cannot express; storage packing remains downstream. Runtime-safe descriptor values are owned by shared
  `WaferABI`, not duplicate WaferIR C++ structs.
- The package/runtime plan's Outer Program Delivery Foundation owns `ProgramOutputTransaction` and
  `CanonicalEncodingSession`; frontend code receives a session only through compiler-private
  `detail::FrontendProgramOutputAccess` and exposes no public context/token accessor.

## Dependency Order

```text
target-artifact shared Tasks 1-5
  (Proto/WCRE, model/distributed/KAD schema+value types, canonical context,
   target identity builders and VerifiedTargetCompilationContext)
  -> package/runtime Outer Program Delivery Foundation
  -> Tasks 1-2 typed model/frontend owner
  -> Task 3 target environment/mesh projection
  -> Task 4 parallel formation
  -> Tasks 5-6 distributed identity/assembly
  -> Task 7 candidate identity
  -> Task 8 vertical gate
```

The target correctness structure/geometry plan may proceed after this plan exposes the typed executable boundary; its temporary
fail-closed hotfix is not a substitute for the shared prerequisites above.

---

### Task 1: Typed Model Interface ODS

**Files:**
- Create: `include/Wafer/IR/Model/ModelOps.td`
- Create: `include/Wafer/Compiler/ModelIdentityBuilder.h`
- Create: `lib/Wafer/IR/Model/ModelOps.cpp`
- Create: `lib/Wafer/Compiler/ModelIdentityBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Modify: `include/Wafer/IR/WaferOps.td`
- Modify: `include/Wafer/IR/WaferAttrs.td`
- Modify: `include/Wafer/IR/WaferInterfaces.td`
- Modify: `lib/Wafer/IR/WaferInterfaces.cpp`
- Modify: `include/Wafer/IR/CMakeLists.txt`
- Modify: `lib/Wafer/IR/CMakeLists.txt`
- Create: `test/Dialect/Wafer/Model/model-interface.mlir`
- Create: `test/Dialect/Wafer/Model/invalid-model-interface.mlir`
- Create: `test/Dialect/Wafer/Model/program-graph.mlir`
- Create: `test/Dialect/Wafer/Model/quantization.mlir`

**Interfaces:**
- Consumes: verified function boundary and program metadata.
- Produces: `wafer.model.interface/entrypoint/resource/shape_constraints/alias/state_group/invocation_policy_integer/
  program_graph/program_member/program_edge` symbol graph plus verified registered quant types/Wafer block-scale attrs.

- [ ] **Step 1: Add parser/verifier tests**

Positive fixture:

```mlir
wafer.model.interface @model
  api_entrypoints [@model::@invoke]
  resources [@model::@input, @model::@output, @model::@weight, @model::@state]
  shape_constraints @model::@shapes {
  wafer.model.entrypoint @invoke {
    api_ordinal = 1,
    roots = [@model::@program::@member0],
    inputs = [@model::@input],
    outputs = [@model::@output],
    state_groups = [@model::@state_group]
  }
  wafer.model.shape_constraints @shapes {
    wafer.model.dim @batch bounds [1, 8]
    wafer.model.dim @sequence bounds [1, 4096]
    wafer.model.divisible @sequence by 16
  }
  wafer.model.resource @input {
    role = #wafer.model_role<external_input>,
    port = #wafer.model_port<input, 0>,
    type = tensor<?x?xf16>,
    dims = [@model::@shapes::@batch, @model::@shapes::@sequence],
    access = #wafer.access<read>,
    lifetime = #wafer.lifetime<invocation>
  }
  wafer.model.end
}
```

Implement this custom assembly only after generic parse/print and verifier tests pass. The invariant is one `model.interface`
symbol-table region containing all resource/constraint/alias symbols, while entry `func.func` refs resolve in the enclosing
module. Do not emit top-level sibling resource symbols or duplicate DimIds as strings.

Negative cases: duplicate ResourceId, non-contiguous API ordinals/ports, empty/unknown entrypoint root, entrypoint IO/state
contract mismatch, static EntryId/function-name-as-API, immutable write, missing state update/group, overlapping state-group
members, invalid invocation field bounds/default, missing/overlapping program ports/edges, unknown DimId, mismatched rank,
payload digest/size mismatch, alias to undeclared resource, malformed affine scale/zp axis and malformed block-scale descriptor.

- [ ] **Step 2: Define enums and attrs**

Add ODS enums with zero-valued `unspecified`:

```text
ModelRole: external_input, external_output, immutable_parameter, persistent_state
Access: read, write, read_write
Lifetime: invocation, executable, persistent
ModelPortKind: none, input, output
StateInitializerKind: none, zero, payload, external
AliasUpdateKind: alias, read_after_write, update, returned_alias
ProgramEdgeKind: tensor, state, control, segmented_dispatch, segmented_combine
StateGroupUpdateKind: read_only, may_update, publish_together
QuantizationKind: none, affine, block_scaled_float
```

Define `DimDeclOp(id, lower, upper)`, typed equality/divisibility clause attrs, `ModelPortAttr(kind, ordinal)`,
`StateInitializerAttr(kind, optional payload ref)`, and `DigestAttr(algorithm, bytes)`. Digest bytes must be exactly 32 for
SHA-256. Runtime invocation actions such as create/attach/reset are intentionally absent from frontend model IR.

Add `ModelEntrypointOp(apiOrdinal, roots, ordered inputs, ordered outputs, state-group access/update, diagnosticAlias)`,
`InvocationPolicyIntegerOp(localOrdinal, lower, upper, required, optionalDefault)`,
`StateGroupOp(StateConsistencyGroupId, canonical members, typed update relation)`, and scoped ID wrappers from shared
`WaferABI`. Public `ModelEntrypointId` and `InvocationPolicyFieldId` are built only after model-interface identity from
`(ModelInterfaceSemanticId, localOrdinal)`; no self-digest preimage. Entry roots resolve through typed program-member refs;
diagnostic alias and function symbols are nonidentity. Keep quant semantics in registered MLIR Quant/StableHLO affine forms
plus a typed ODS `BlockScaledFloatAttr` for otherwise-unrepresentable block axes/shape, scale encoding and numeric policies.
Do not create `Wafer/IR/Quantization.h`; the one compiler adapter projects verified IR to shared
`abi::QuantizationDescriptor` when delivery is required.

Implement one two-phase `ModelIdentityBuilder` fixed by `tasks/02`. Phase 1 encodes local entrypoint/policy ordinals,
`LocalBoundaryResourceKey`, `LocalInternalResourceKey`, `LocalDimKey` and their relations into record 11 and obtains
`ModelInterfaceSemanticId`. Phase 2 creates record-13/14 ResourceIds from `(owner, tagged local key)`, record-15 DimIds from
`(owner, ResourceId, dimension ordinal)`, and scoped entrypoint/policy IDs; there is no bare-key factory. Add tests proving
function/resource/dimension rename stability; intentional change on port/role/alias/dimension/program/payload semantics; and
different owner + identical local keys produce unequal IDs and reject cross-model cache/state relations.

Every builder method takes `abi::CanonicalEncodingContext &` from the outer transaction; no model ID API creates a default
scratch store or returns unbounded canonical bytes.

- [ ] **Step 3: Define symbol ops**

```tablegen
def Wafer_ModelInterfaceOp : Wafer_Op<"model.interface",
    [Symbol, SymbolTable, SingleBlockImplicitTerminator<"ModelEndOp">]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
    ArrayAttr:$api_entrypoints,
    ArrayAttr:$resources,
    SymbolRefAttr:$shape_constraints);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}
```

Define nested symbol ops for model entrypoint, resource, shape constraints, Dim declarations, alias, state group, invocation-policy integer,
program graph/member/edge. `ModelProgramGraphOp` owns a nonempty ordered member list; each member has a nonzero
`ModelProgramMemberId`, function refs and ordered typed ports. Edges carry source/destination member+port, kind, type/DimId,
required ResourceId/alias-update and segmented capacity. `ModelResourceOp` stores `TypeAttr`,
enum attrs, `ModelPortAttr`, symbol refs to Dim declarations, optional payload digest/bytes/encoding and typed initializer/import
policy. `ModelEntrypointOp` owns a nonzero API ordinal, typed function/member roots, ordered IO resources, state-group
access/update relation and nonidentity diagnostic alias. The interface verifier requires every api-entrypoint/resource and
entrypoint IO/state array element to be a nested typed ref; do not
store resource role, DimId, program edge, invocation field or state policy in free-form strings.

- [ ] **Step 4: Implement relation verifier**

The interface verifier resolves all refs, checks each API record's reachable typed member graph and port bijection against its
root function/member boundaries, then calls resource/constraint/alias/state-group/program-graph/invocation-field/quantization
verifiers. Every persistent resource belongs to exactly one group;
page-table/backing members that publish together cannot be split. Program graph defaults only to an explicit singleton, never a
missing record. Use numeric port/member/field ordinals; names are diagnostic only.

- [ ] **Step 5: Run and commit**

```bash
cmake --build build/wafer-dev --target wafer-opt -- -j128
<configured-lit> -sv build/wafer-dev/test/Dialect/Wafer/Model
python3 tools/check_ir_organization.py --root .
git add include/Wafer/IR lib/Wafer/IR \
  include/Wafer/Compiler/ModelIdentityBuilder.h include/Wafer/IR/WaferInterfaces.td \
  lib/Wafer/Compiler/ModelIdentityBuilder.cpp lib/Wafer/Compiler/CMakeLists.txt \
  lib/Wafer/IR/WaferInterfaces.cpp \
  test/Dialect/Wafer/Model
git commit -m "Add typed model interface IR"
```

### Task 2: Frontend Model Materialization

**Files:**
- Modify: `include/Wafer/Frontend/Program.h`
- Create: `include/Wafer/Frontend/ProgramSource.h`
- Create: `lib/Wafer/Frontend/ProgramSource.cpp`
- Create: `include/Wafer/Frontend/FrontendAdmissionLimits.h`
- Create: `lib/Wafer/Frontend/FrontendAdmissionLimits.cpp`
- Split: `lib/Wafer/Frontend/Program.cpp`
- Create: `lib/Wafer/Frontend/ProgramMetadata.cpp`
- Create: `lib/Wafer/Frontend/ProgramMaterialization.cpp`
- Modify: `lib/Wafer/Frontend/CMakeLists.txt`
- Modify: `tools/wafer-compile-stablehlo/wafer-compile-stablehlo.cpp`
- Create: `test/Tools/wafer-compile-stablehlo-stateful.test`
- Create: `test/Tools/Inputs/stateful-program/functions/forward.mlir`
- Create: `test/Tools/Inputs/stateful-program/functions/forward.meta`
- Create: `test/Tools/wafer-compile-stablehlo-program-graph.test`
- Create: `test/Tools/wafer-compile-stablehlo-low-precision.test`

**Interfaces:**
- Consumes: capability-opened program directory, exporter-native metadata/payload, registered StableHLO/Quant/FP8 IR and
  same-module typed/Shardy MPMD program graph.
- Produces:

```cpp
struct FrontendAdmissionLimitValues {
  uint64_t maxProgramIndexBytes, maxMetadataBytes;
  uint64_t maxSingleModuleBytes, maxTotalModuleBytes, maxSingleFunctionBytes;
  uint64_t maxTokens, maxNesting, maxOps, maxRegions, maxBlocks, maxValues;
  uint64_t maxTypes, maxAttrs, maxSymbols, maxModelEntrypoints;
  uint64_t maxResources, maxDims, maxConstraints, maxAliases, maxStateGroups;
  uint64_t maxInvocationFields, maxProgramMembers, maxProgramEdges;
  uint64_t maxPayloadRefs, maxLocatorBytes, maxTensorHeaderBytes;
  uint64_t maxSinglePayloadBytes, maxTotalDeclaredPayloadBytes;
  uint64_t maxSourceLeases, maxOpenFiles, maxParseWorkers;
  uint64_t maxPeakCloneBytes, maxDiagnosticBytes;
  uint64_t parseWallMillis, parseCpuMillis, parseRssBytes, killReapMillis;
};

class FrontendAdmissionLimits final {
public:
  static llvm::Expected<FrontendAdmissionLimits>
  create(const FrontendAdmissionLimitValues &values);
  const FrontendAdmissionLimitValues &values() const;
private:
  explicit FrontendAdmissionLimits(FrontendAdmissionLimitValues values);
  FrontendAdmissionLimitValues values_;
};

class VerifiedProgramSource final {
public:
  VerifiedProgramSource() = delete;
  VerifiedProgramSource(const VerifiedProgramSource &) = delete;
  VerifiedProgramSource(VerifiedProgramSource &&) = default;
  llvm::ArrayRef<SourceArtifactRef> artifacts() const;
private:
  friend class detail::FrontendSourceAccess;
  friend class detail::ArtifactMaterializationSourceAccess;
  // Owns the open root capability, expected refs and shared source-access
  // coordinator. There is no public acquire/open/read.
};

llvm::Expected<VerifiedProgramSource>
openVerifiedProgramSource(llvm::StringRef programDir,
                          const FrontendAdmissionLimits &limits);

class VerifiedModelProgramView final {
public:
  VerifiedModelProgramView() = delete;
  VerifiedModelProgramView(const VerifiedModelProgramView &) = delete;
  VerifiedModelProgramView(VerifiedModelProgramView &&) noexcept;
  ~VerifiedModelProgramView();
  const abi::ModelInterfaceSemanticId &semanticId() const;
  uint64_t moduleGeneration() const;
  llvm::ArrayRef<ModelEntrypointPolicyView> entrypoints() const;
  llvm::ArrayRef<ModelProgramMemberPolicyView> members() const;
  llvm::ArrayRef<ModelProgramEdgePolicyView> edges() const;
  llvm::ArrayRef<StateGroupPolicyView> stateGroups() const;
  llvm::ArrayRef<InvocationFieldPolicyView> invocationFields() const;
private:
  friend class MaterializedFrontendProgram;
  // Constructed only by frontend materialization from a freshly verified module.
  class Impl;
  explicit VerifiedModelProgramView(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class MaterializedFrontendProgram final {
public:
  MaterializedFrontendProgram() = delete;
  MaterializedFrontendProgram(const MaterializedFrontendProgram &) = delete;
  MaterializedFrontendProgram(MaterializedFrontendProgram &&) = default;
  const VerifiedModelProgramView &modelProgram() const;
  const FrontendProgramVerificationResult &summary() const;
private:
  friend class compiler::detail::FrontendProgramOutputAccess;
  friend class CompilationRequest;
  MaterializedFrontendProgram(std::unique_ptr<mlir::MLIRContext> context,
      mlir::OwningOpRef<mlir::ModuleOp> module,
      FrontendProgramVerificationResult summary,
      VerifiedModelProgramView modelProgram,
      VerifiedProgramSource source,
      compiler::CanonicalEncodingOwnerToken encodingOwner,
      uint64_t moduleGeneration);
  // Declared first so module_ and every dependent view die before context_.
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  FrontendProgramVerificationResult summary_;
  VerifiedModelProgramView modelProgram_;
  VerifiedProgramSource source_;
  compiler::CanonicalEncodingOwnerToken encodingOwner_;
  uint64_t moduleGeneration_;
};
```

There is no public session-borrowing free function. The noninstalled compiler-private
`detail::FrontendProgramOutputAccess::loadVerifyAndMaterializeStableHLOProgramDir(ProgramOutputTransaction &,
VerifiedProgramSource, raw_ostream &)` is the only callable entry; it internally creates and registers the sole frontend
`MLIRContext`, obtains the retained canonical session through the transaction pimpl and never exposes a session/context/token
accessor or accepts a caller context factory. The returned summary and model-program view
contain no mutable MLIR handles. `detail::FrontendSourceAccess` and
`detail::ArtifactMaterializationSourceAccess` are the only source-open friends; each requires its own non-forgeable stage lease
jointly charged against the retained source coordinator and stage/outer budgets. The resulting move-only
`SourceArtifactLease` owns the reader/FD reservation and exact handle; no caller can repeatedly acquire from the source directly.

- [ ] **Step 1: Add a stateful failing program gate**

The fixtures cover distinct typed prefill/decode model APIs with different IO/state roots, immutable weight, a persistent KV
backing + page-table state group, required/default bounded invocation integer fields, a singleton graph, PP=2 member graph,
router/heterogeneous-expert/combine segmented graph, affine
quant and block-scaled FP8. Verify the current tool does not materialize the required typed objects.

- [ ] **Step 2: Separate parse records from IR mutation**

`ProgramSource.cpp` opens the directory once as a root capability, validates every positive checked limit and safe relative
locator, and creates `SourceArtifactRef(expected size, SHA-256, encoding, locator)` without retaining one FD per payload.
Frontend-private access first acquires a same-owner source/stage lease, then uses beneath/no-follow relative open and returns a
move-only lease; consumers use stat-before, positioned reads/full
digest and stat-after on that same FD. `ProgramMetadata.cpp` parses JSON/NPY only through those leases and never exposes a
semantic field keyed by parameter name. It resolves metadata to verified API/root/boundary ordinals before IR mutation and
calls the two-phase `ModelIdentityBuilder`; a name/path appears only in diagnostics/locators. Rename-only directories keep
identity; same local structure under different program/payload content gets different owner-scoped IDs. Replacement, symlink,
truncate/grow and concurrent mutation between verification/acquisition must fail without a materialized proof.

Before parsing untrusted textual/bytecode MLIR, create the private context with the reviewed finite dialect/interface registry,
then enforce source byte limits and run the parser in a sandboxed
`FrontendParseWorker` whenever token/nesting/RSS cannot be enforced in-process. Bound worker wall/CPU/RSS/output, kill/reap its
process group on failure, then re-count returned ops/regions/blocks/values/types/attrs/symbols and model records before clone
mutation. Add 10k-resource/function and deep/high-fanout fixtures plus limit-1/limit+1, hang/RSS/output-bomb tests. Sufficient
limits and worker/buffer changes must leave IR/model IDs byte-identical; limit failure leaves outer output untouched.

Program members/edges do not come from a new JSON file. If registered Shardy MPMD is present, normalize it; otherwise require
registered `wafer.model.program_*` ops in the same module and verify them before clone mutation. Build ordered
`ModelEntrypointId`, `ModelProgramMemberId`, state groups and invocation-field local ordinals through typed builders. Preserve registered `!quant.uniform`,
`stablehlo.uniform_quantize/dequantize` and FP8 types; add only the typed block-scale relation required by tasks/05.

- [ ] **Step 3: Materialize on a module clone**

```cpp
auto context = createRegisteredFrontendMlirContext();
auto module = parseVerifiedFrontendModule(*context, sourceLease, limits);
auto clone = module.clone();
mlir::OpBuilder builder(clone->getBodyRegion());
auto interface = buildModelInterface(builder, *metadata, *clone);
if (mlir::failed(verifyModelInterface(interface)))
  return mlir::failure();
auto generation = currentModuleGeneration(*clone);
auto modelProgram = buildImmutableModelProgramView(
    interface, generation, encoding);
if (mlir::failed(modelProgram))
  return mlir::failure();
return MaterializedFrontendProgram(
    std::move(context), std::move(clone), summary,
    std::move(*modelProgram), std::move(source),
    encoding.ownerToken(), generation);
```

Normalize every old `wafer.frontend.dynamic_bounds` entry into Dim declarations and typed clauses; erase the old attr after
successful materialization. Materialize explicit singleton state/program groups when legal, but reject implicit split of related
state or name-derived PP/expert edges.
`VerifiedModelProgramView::Impl` copies only policy-validation projections with typed IDs/ordinals plus the exact model semantic
ID and module generation. It has no Operation/Block/Value/SymbolRef handle, mutator, serializer or independent semantic owner;
the private module remains the sole IR fact source and `CompilationRequest` must rederive and compare this projection.
Add compile-time tests proving only `CompilationRequest` can move the private module/source owner, and no public method returns
`ModuleOp`, `Operation *`, `MLIRContext *`, `VerifiedProgramSource`, root capability or source coordinator. Destroy every caller
parse frame/context handle immediately after factory return and require late clone/materialization plus failure cleanup to remain
safe; the member-order test destroys module and clone before the private context. The downstream owner-lifetime test
is implemented by the Whole plan: destroy this task's materialization and request frames, then complete late payload reads from
the move-owned `ExecutableCompilationInput`.

- [ ] **Step 4: Make verifier entry consume typed IR**

`verifyFrontendProgram` checks `wafer.model.interface`; program-dir-only names/paths are checked only through
`VerifiedProgramSource`. The downstream SPMD/whole-artifact chain receives only the complete move-owned frontend owner through
`CompilationRequest`/`ExecutableCompilationInput`; it does not receive a borrowed module, reparse metadata or recreate a checked path.

- [ ] **Step 5: Run frontend matrix**

```bash
<configured-lit> -sv build/wafer-dev/test \
  --filter='Tools/wafer-compile-stablehlo-(reference|stateful)\.test|Tools/wafer-pytorch-xla-capture-.*\.test'
```

Expected: typed static/dynamic/stateful/program-graph/low-precision programs pass; missing state group, invalid invocation
field, name-only graph, malformed segmented edge, quant scale/zp/block mismatch, illegal alias and payload mismatch fail.

- [ ] **Step 6: Commit**

```bash
git add include/Wafer/Frontend lib/Wafer/Frontend tools/wafer-compile-stablehlo test/Tools
git commit -m "Materialize typed model resources during frontend import"
```

### Task 3: Target Environment, Arena, and Mesh Roles

**Prerequisite:** complete target-artifact shared Tasks 1-5, including Proto/WCRE/schema, KAD value/schema foundation,
IR-to-generated-message builders and `VerifiedTargetCompilationContext`. This task consumes
`TargetEnvironmentIdentity`/topology/mesh builders and cannot create a local WCRE registry or target-context substitute.

**Files:**
- Create: `include/Wafer/IR/Target/EnvironmentOps.td`
- Create: `lib/Wafer/IR/Target/EnvironmentOps.cpp`
- Modify: `include/Wafer/IR/Target/TopologyOps.td`
- Modify: `lib/Wafer/IR/Target/TopologyOps.cpp`
- Create: `lib/Wafer/Transforms/Target/MaterializeTargetEnvironment.cpp`
- Create: `include/Wafer/Planning/TopologyAdmissionLimits.h`
- Create: `lib/Wafer/Planning/TopologyAdmissionLimits.cpp`
- Create: `include/Wafer/Compiler/TargetProgramMaterializationPlan.h`
- Create: `lib/Wafer/Compiler/TargetProgramMaterializationPlan.cpp`
- Create: `lib/Wafer/Transforms/Target/TargetProgramMaterializationInternal.h`
- Create: `include/Wafer/Compiler/TargetIdentityBuilder.h`
- Create: `lib/Wafer/Compiler/TargetIdentityBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Modify: `lib/Wafer/Transforms/Target/MaterializeTargetTopology.cpp`
- Modify: `lib/Wafer/Transforms/Target/MaterializeExecutionMesh.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `unittests/Compiler/TargetProgramMaterializationPlanTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Create: `test/Dialect/Wafer/Target/Environment/environment.mlir`
- Create: `test/Dialect/Wafer/Target/Environment/invalid-environment.mlir`
- Create: `test/Dialect/Wafer/Target/Environment/low-precision-capability.mlir`
- Modify: `test/Dialect/Wafer/Target/Topology/*.mlir`

**Interfaces:**
- Consumes: exact canonical `VerifiedTargetCompilationContextRegistry`, immutable frontend model/mesh request view, outer
  canonical owner and validated topology limits; production access never borrows the frontend module before request ownership.
- Produces: no-mutation `VerifiedTargetProgramMaterializationPlan` containing an immutable mesh-policy projection; after request creation the
  direct driver privately materializes/reverifies the environment/topology/mesh chain in the owned module.

```cpp
struct TopologyAdmissionLimitValues {
  uint64_t maxEnvironments, maxDevices, maxEndpoints, maxLinks, maxProcessGroups;
  uint64_t maxArenas, maxCapabilities, maxMeshAxes, maxMeshCoordinates;
  uint64_t maxProjectionMembers, maxFiniteBindings, maxDteSlots;
  uint64_t maxInputBytes, maxDerivedBytes, maxSortScratchBytes;
  uint64_t maxWorkers, maxDiagnosticBytes;
};

class TopologyAdmissionLimits final {
public:
  static llvm::Expected<TopologyAdmissionLimits>
  create(const TopologyAdmissionLimitValues &values);
  const TopologyAdmissionLimitValues &values() const;
private:
  explicit TopologyAdmissionLimits(TopologyAdmissionLimitValues values);
  TopologyAdmissionLimitValues values_;
};

class VerifiedTargetProgramMaterializationPlan final {
public:
  VerifiedTargetProgramMaterializationPlan(
      VerifiedTargetProgramMaterializationPlan &&) noexcept;
  VerifiedTargetProgramMaterializationPlan(
      const VerifiedTargetProgramMaterializationPlan &) = delete;
private:
  VerifiedTargetProgramMaterializationPlan() = delete;
  class Storage;
  explicit VerifiedTargetProgramMaterializationPlan(
      std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
  friend class detail::TargetProgramMaterializationPlanAccess;
  friend class detail::ParallelizationPolicyBuilder;
  friend class CompilationRequest;
};
```

- [ ] **Step 1: Write ODS negative matrix**

Cover duplicate `DdrArenaId`, invalid capacity/alignment/address width, topology referencing wrong environment, mesh role-count mismatch, unknown role, disconnected endpoint and stale environment fingerprint.

- [ ] **Step 2: Define environment records**

Define `TargetFamilyAttr`, `TargetRevisionAttr` and `AbiVersionAttr` as validated stable ASCII IDs, then use structured typed
attrs for the remainder:

```text
TargetEnvironmentOp
  family/revision/device_abi/target_crt_abi
  spm limits and reserved ranges
  engine/queue counts
  DdrArenaAttr[]
  DTypeCapabilityAttr[]
  LayoutCapabilityAttr[]
  InstructionLimitAttr[]
  PacketLimitAttr[]
  DteCapabilityAttr[]  # allowed/reserved block->channel map, FSM/stream/packet ranges,
                       # exact-block selection, nonblocking send/recv status, finite-cycle wait
  LowPrecisionComputeCapabilityV1[]
  RuntimeModeAttr[]
  ErratumAttr[]
  TargetEnvironmentFingerprint
```

Each capability attr has a typed kind enum plus versioned numeric/enum fields; capability and erratum kinds are not free-form
strings. `DdrArenaAttr` contains id, memory domain enum, capacity, largest contiguous, alignment, address width, bandwidth
limit, allowed placement-domain enums and runtime binding ABI. Unknown kinds fail parse/verification until the numbered design
and enum are extended.

`LowPrecisionComputeCapabilityV1` uses closed op/storage/expressed/accumulator/result/granularity/block/numeric-policy/
implementation-mode enums and exact raw-field encoding bounds. It is the environment-owned hardware legality input only.
`QuantStorageAbiProfileRegistryV1` belongs to the prevalidated `TargetBuildProfile` supplied by the artifact shared foundation
and CompilationRequest; it depends on storage/command/CRT/toolchain ABI and is not materialized inside TargetEnvironmentOp.
Add environment tests for signed-i8 mathematical/raw zp bounds, FP8 explicit-composite mode, unknown capability version and
one-field fingerprint perturbation.

- [ ] **Step 3: Add explicit references and roles**

`TargetTopologyOp` gains mandatory `environment`. `ExecutionMeshOp` gains mandatory `environment`, `topology`, and `axis_roles`; axes stay diagnostic names. Define role enum `dp/tp/pp/ep/partition/replica`.

- [ ] **Step 4: Preflight without mutation, then materialize only through the owned request**

Implement private-construction `TopologyAdmissionLimits` with positive counts/bytes for environments/devices/endpoints/links/
PGs/arenas/capabilities/mesh axes/coordinates/projection members/finite bindings/DTE slots/sort scratch/workers/diagnostics.
Before any module mutation, noninstalled `detail::TargetProgramMaterializationPlanAccess` uses private registry access to consume
the exact environment/topology configs, immutable model mesh request, limits and outer canonical session. It checked-counts
before coordinate/template expansion, calls `TargetIdentityBuilder`, verifies all-and-only target requirements, and seals IDs,
axis roles/shape/endpoint policy in an MLIR-handle-free private projection. The built-in default descriptor is one
context producer, not hard-coded op shape. Reject context cross-pair/stale generation before plan creation; sufficient-limits
differences must not change identities/member order.

After `CompilationRequest::create` move-owns frontend owner, exact registry, plan and policy, the direct driver calls the
noninstalled owner-aware materializer from `TargetProgramMaterializationInternal.h` only after sealing
`ExecutableCompilationInput` and creating the first candidate transaction. It mutates only that transaction's working clone,
never the input-owned source module,
creates environment/topology/mesh ops, then recomputes identity/role/coverage and compares every fact to the sealed plan before
parallel formation. `MaterializeTargetEnvironmentPass` and topology/mesh passes no longer receive a context registry: they only
validate/canonicalize already-materialized IR or replay explicit typed test inputs with no source/output authority. Remove
first-topology/`@default_mesh` discovery; no pass or global stores toolchain/context pointers.

- [ ] **Step 5: Run target tests**

```bash
<configured-lit> -sv build/wafer-dev/test \
  --filter='Dialect/Wafer/Target/(Environment|Topology)/|Transforms/materialize-(target-environment|target-topology|execution-mesh)\.mlir'
```

- [ ] **Step 6: Commit**

```bash
git add include/Wafer/IR/Target lib/Wafer/IR/Target \
  include/Wafer/Compiler/TargetProgramMaterializationPlan.h \
  lib/Wafer/Compiler/TargetProgramMaterializationPlan.cpp \
  include/Wafer/Compiler/TargetIdentityBuilder.h \
  lib/Wafer/Compiler/TargetIdentityBuilder.cpp lib/Wafer/Compiler/CMakeLists.txt \
  include/Wafer/Transforms lib/Wafer/Transforms/Target test \
  unittests/Compiler/TargetProgramMaterializationPlanTest.cpp unittests/CMakeLists.txt
git commit -m "Materialize typed target environment and mesh roles"
```

### Task 4: Structured Parallel Program Formation

**Files:**
- Create: `include/Wafer/IR/Parallel/ParallelOps.td`
- Create: `lib/Wafer/IR/Parallel/ParallelOps.cpp`
- Create: `include/Wafer/Planning/ParallelProgramFormation.h`
- Create: `lib/Wafer/Planning/ParallelProgramFormation.cpp`
- Create: `include/Wafer/Planning/DistributedFormationLimits.h`
- Create: `lib/Wafer/Planning/DistributedFormationLimits.cpp`
- Create: `lib/Wafer/Transforms/SPMD/MaterializeParallelProgram.cpp`
- Modify: `include/Wafer/IR/WaferOps.td`
- Modify: `lib/Wafer/IR/CMakeLists.txt`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Dialect/Wafer/Parallel/parallel-program.mlir`
- Create: `test/Dialect/Wafer/Parallel/invalid-parallel-program.mlir`
- Create: `test/Integration/source-backed-mpmd-formation.test`

**Interfaces:**
- Consumes: verified model program graph, execution mesh and validated `ParallelizationPolicy` from CompilationRequest.
- Produces: candidate-only `wafer.parallel.program/component/edge` graph consumed once by per-component SPMD.

```cpp
struct DistributedFormationLimitValues {
  uint64_t maxModelMembers, maxModelEdges, maxModelPorts;
  uint64_t maxParallelComponents, maxParallelEdges, maxParticipantPredicates;
  uint64_t maxMeshCoordinates, maxDistributedInstances, maxClasses;
  uint64_t maxShards, maxVariants, maxHighFanoutRefs;
  uint64_t maxSolverStates, maxSimultaneousClones, maxWorkers;
  uint64_t maxPeakIrBytes, maxAnalysisBytes, maxDiagnosticBytes;
};

class DistributedFormationLimits final {
public:
  static llvm::Expected<DistributedFormationLimits>
  create(const DistributedFormationLimitValues &values);
  const DistributedFormationLimitValues &values() const;
private:
  explicit DistributedFormationLimits(DistributedFormationLimitValues values);
  DistributedFormationLimitValues values_;
};
```

- [ ] **Step 1: Add source-backed failing gates**

Run the frontend fixtures from Task 2 through formation. Require PP=2 to form two stage components with typed tensor/state
edges, and the heterogeneous MoE source to form router/dispatch/expert/combine components with bounded segmented edges.
Rename functions/experts/buffers without changing the graph; reject missing/overlapping ports, unbounded segmented capacity and
name-only formation.

- [ ] **Step 2: Define parallel ODS and typed policy**

Implement `ParallelProgramOp`, `ParallelComponentOp` and `ParallelEdgeOp` exactly from tasks/03. Components store nonzero
program-scoped `ComponentId`, explicit source `ModelProgramMemberId` mapping, ordered local ABI and typed
`LogicalParticipantPredicate` over axis-role enums. `ParallelizationPolicy` is a versioned closed request value; it may select
preserve-members, pipeline partition, expert partition or composition using registered op interfaces/cut constraints. It may
not contain callbacks, symbol regexes or opaque payloads.

Give `ParallelizationPolicy` a private constructor and one noninstalled `detail::ParallelizationPolicyBuilder` that consumes the
verified ModelEntrypoint/ModelProgramMember graph plus the private immutable mesh projection in
`VerifiedTargetProgramMaterializationPlan` and `DistributedFormationLimits`; policy input references typed API/member/edge ordinals,
never aliases/names. The factory checked-counts members/edges/coordinates/solver states before expansion. Add 10k-member,
high-fanout and product-overflow gates; different sufficient limits/workers yield identical materialized graph and IDs.
The builder consumes only `MaterializedFrontendProgram::modelProgram()` and the target plan; neither exposes MLIR handles. It
cannot recover or mutate the private module. `CompilationRequest` then move-consumes the entire frontend owner, exact context
registry, target plan and policy, and replays module generation, handoff verification and model semantic identity before the
direct driver materializes target IR or starts clone/staging.

- [ ] **Step 3: Build in one transaction**

```cpp
mlir::FailureOr<ParallelProgramOp> materializeParallelProgram(
    ModelInterfaceOp model, ExecutionMeshOp mesh,
    const ParallelizationPolicy &policy,
    const DistributedFormationLimits &limits,
    mlir::RewriterBase &rewriter);
```

Build on a full module clone, assign ComponentId from explicit semantic component order, preserve model resource/state-group
relations and reject cross-component ordinary calls without typed edges. Policy/search data remains build provenance; only
materialized components/source mapping/participant predicates/edges can enter distributed identity.

- [ ] **Step 4: Add per-component SPMD bridge and verifier**

Verify exact model-member/output coverage, nonempty/nonoverlapping participants, edge port/type/DimId/ResourceId compatibility,
state alias/update closure and segmented count-before-data relation. Feed each component independently to Shardy propagation/
XLA SPMD, then pass all local outputs and the same typed graph to distributed assembly. No partial component output survives a
failure.

- [ ] **Step 5: Run and commit**

```bash
<configured-lit> -sv build/wafer-dev/test/Dialect/Wafer/Parallel \
  build/wafer-dev/test/Integration/source-backed-mpmd-formation.test
git add include/Wafer/IR/Parallel lib/Wafer/IR/Parallel \
  include/Wafer/Planning lib/Wafer/Planning lib/Wafer/Transforms/SPMD \
  include/Wafer/Transforms test
git commit -m "Add structured parallel program formation"
```

### Task 5: Distributed Program ODS

**Files:**
- Create: `include/Wafer/IR/Distributed/DistributedOps.td`
- Create: `lib/Wafer/IR/Distributed/DistributedOps.cpp`
- Create: `include/Wafer/Compiler/DistributedIdentityBuilder.h`
- Create: `lib/Wafer/Compiler/DistributedIdentityBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Modify: `include/Wafer/IR/WaferOps.td`
- Modify: `include/Wafer/IR/WaferAttrs.td`
- Modify: `lib/Wafer/IR/CMakeLists.txt`
- Create: `test/Dialect/Wafer/Distributed/distributed-program.mlir`
- Create: `test/Dialect/Wafer/Distributed/invalid-distributed-program.mlir`

**Interfaces:**
- Consumes: verified parallel program, model interface, mesh, component-local partitioned functions and shard records.
- Produces: verified distributed program symbol graph.

- [ ] **Step 1: Add representative typed IR**

The positive fixture must include:

- `dp x tp` mesh roles;
- PP=2 components and one state-group-preserving edge, plus a segmented expert edge fixture;
- two explicit instances with partition and replica coordinates;
- one rank-parametric class and one per-rank-static class;
- parameter/state resource shards;
- one coherent variant.

- [ ] **Step 2: Define ops from tasks/03**

Add symbol ops:

```text
wafer.distributed.program
wafer.distributed.component
wafer.distributed.instance
wafer.distributed.class
wafer.distributed.resource_shard
wafer.distributed.edge
wafer.distributed.variant
```

`DistributedProgramOp` owns an explicit ordered component list; each `DistributedComponentOp` stores its program-local
nonzero `uint32 ComponentId` copied unchanged from parallel formation plus explicit source-member mapping.
`DistributedInstanceOp` stores the local key `(ComponentId, partition coordinate,
replica coordinate)` and mesh ref; no flat-rank or precomputed full-program digest field. `DistributedClassOp` stores a typed
`rank_parametric|per_rank_static` enum and instance/function refs.

- [ ] **Step 3: Implement whole-program verifier**

Verify parallel-to-distributed component/source/participant/edge preservation, coordinate rank/bounds, exact instance coverage,
one component per class, mode rules, shard bounds/content identity, state-group/segmented edge identity and global variant
coverage. Reject physical endpoint, Wafer memory attrs, offsets and runtime attrs within the distributed symbol graph.

Implement `DistributedIdentityBuilder` to populate the generated distributed-program identity message from local keys only
after this verifier passes, compute the program digest, then expose scoped public
`ExecutionInstanceId=(DistributedProgramSemanticId, local key)` accessors and verify every ref. The digest preimage never
self-contains those derived full IDs. Task 6 calls it after complete assembly and stores/checks the returned digest; no pass
walks fields into raw WCRE.
Tests rename component/function/payload locators without changing identity and change coordinates, class membership, shard or
edge semantics to require a different digest.

- [ ] **Step 4: Run and commit**

```bash
cmake --build build/wafer-dev --target wafer-opt -- -j128
<configured-lit> -sv build/wafer-dev/test/Dialect/Wafer/Distributed
git add include/Wafer/IR/Distributed lib/Wafer/IR/Distributed \
  include/Wafer/Compiler/DistributedIdentityBuilder.h \
  lib/Wafer/Compiler/DistributedIdentityBuilder.cpp lib/Wafer/Compiler/CMakeLists.txt \
  include/Wafer/IR/WaferOps.td include/Wafer/IR/WaferAttrs.td \
  lib/Wafer/IR/CMakeLists.txt test/Dialect/Wafer/Distributed
git commit -m "Add typed distributed program IR"
```

### Task 6: Distributed Assembly from SPMD Output

**Files:**
- Create: `include/Wafer/Transforms/SPMD/SpmdExecutionLimits.h`
- Create: `lib/Wafer/Transforms/SPMD/SpmdExecutionLimits.cpp`
- Create: `include/Wafer/Transforms/SPMD/SpmdPartitionerConfiguration.h`
- Create: `lib/Wafer/Transforms/SPMD/SpmdPartitionerConfiguration.cpp`
- Create: `lib/Wafer/Transforms/SPMD/MaterializeDistributedProgram.cpp`
- Modify: `lib/Wafer/Transforms/SPMD/XlaSpmdPartitionerMain.cpp`
- Modify: `lib/Wafer/Frontend/Program.cpp`
- Modify: `tools/wafer-opt/wafer-opt.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Create: `test/Tools/wafer-opt-distributed-program.test`
- Modify: `test/Tools/wafer-opt-spmd-partition.test`

**Interfaces:**
- Consumes: verified parallel program, every component's partitioned StableHLO, typed model interface, parameter shard
  import records and execution mesh.
- Produces: `wafer.distributed.program`; JSON shard records are no longer read downstream.

```cpp
struct SpmdExecutionLimitValues {
  uint64_t maxInputBytes, maxOutputBytes, maxFunctions, maxOps;
  uint64_t maxRegions, maxBlocks, maxValues, maxTypes, maxAttrs;
  uint64_t maxShardRecords, maxInstances, maxWorkers;
  uint64_t wallMillis, cpuMillis, rssBytes, addressSpaceBytes;
  uint64_t maxProcesses, maxStdoutBytes, maxStderrBytes, killReapMillis;
};

class SpmdExecutionLimits final {
public:
  static llvm::Expected<SpmdExecutionLimits>
  create(const SpmdExecutionLimitValues &values);
  const SpmdExecutionLimitValues &values() const;
private:
  explicit SpmdExecutionLimits(SpmdExecutionLimitValues values);
  SpmdExecutionLimitValues values_;
};

enum class SpmdPartitionerProtocol { PinnedShardyXlaV1 };

class VerifiedSpmdPartitionerExecutable final {
public:
  VerifiedSpmdPartitionerExecutable() = delete;
  VerifiedSpmdPartitionerExecutable(const VerifiedSpmdPartitionerExecutable &) = delete;
  VerifiedSpmdPartitionerExecutable(VerifiedSpmdPartitionerExecutable &&) = default;
  static llvm::Expected<VerifiedSpmdPartitionerExecutable> open(
      llvm::StringRef locator, abi::ContentDigest expectedDigest,
      SpmdPartitionerProtocol protocol);
  const abi::ContentDigest &contentDigest() const;
private:
  class Impl; // Retains the opened executable capability and fixed argv/env attestation.
  explicit VerifiedSpmdPartitionerExecutable(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class SpmdPartitionerConfiguration final {
public:
  static llvm::Expected<SpmdPartitionerConfiguration> create(
      SpmdPartitionerProtocol protocol,
      VerifiedSpmdPartitionerExecutable executable,
      SpmdExecutionLimits limits);
  const SpmdExecutionLimits &limits() const;
private:
  SpmdPartitionerConfiguration(SpmdPartitionerProtocol protocol,
      VerifiedSpmdPartitionerExecutable executable,
      SpmdExecutionLimits limits);
  SpmdPartitionerProtocol protocol_;
  VerifiedSpmdPartitionerExecutable executable_;
  SpmdExecutionLimits limits_;
};
```

- [ ] **Step 1: Make real program test fail**

Extend singleton, source-backed PP=2 and heterogeneous MoE chains and check for
`wafer.parallel.* -> wafer.distributed.program/component/edge/instance/class/resource_shard`. Check absence of
`wafer.boundary_shards`, downstream JSON parsing and name-derived component diagnostics.

Add helper hang/fork/output-op/stdout/RSS bombs, oversized shard metadata and 10k-instance products. The private-construction
executable factory opens once, streams exact digest/size, retains the same executable capability plus fixed argv/env/protocol
attestation, and spawns from that handle (or immediately reattests the same object identity); it never verifies then reopens a
path. Configuration move-owns that proof and validated positive limits. No raw executable path, global helper lookup or extra
flags escape it.

- [ ] **Step 2: Preserve partition/replica SSA identity**

Delete the cleanup that folds `stablehlo.partition_id` / `replica_id` to constant zero unless a per-rank-static specialization record proves one coordinate. Rank-parametric output retains the SSA op or an explicit ABI value.

- [ ] **Step 3: Materialize canonical instances**

```cpp
for (ParallelComponentOp component : parallelProgram.getOrderedComponents()) {
  for (const MeshCoordinate &partition : enumerateParticipantCoordinates(component, mesh)) {
    for (const MeshCoordinate &replica : enumerateReplicaCoordinates(component, mesh)) {
      buildDistributedInstance(program, component, partition, replica);
    }
  }
}
```

The builder uses mesh axis roles and verified participant predicates, not axis or component names. Preserve typed PP/state and
segmented dispatch/combine edges through assembly. The initial implementation may conservatively create one prerequisite class
per instance, then merge only instances proven equivalent by typed function/resource/collective facts.

Before coordinate expansion, checked-count against the same `DistributedFormationLimits`. Run the external helper in a
sandboxed process group with `SpmdExecutionLimits` and bounded outputs; timeout/cancel/resource/output failure kills/reaps and
deletes partial files. Parent rechecks output bytes/structure/instances before clone mutation. Sufficient limits never change
partition options/result/DistributedProgramSemanticId.

- [ ] **Step 4: Import shard metadata once**

`ProgramMetadata.cpp` parses and validates `forward.parameter_shards.json`; `MaterializeDistributedProgram` consumes the C++ records and writes typed resource-shard ops. After successful materialization, later stages receive no path/name lookup API.

- [ ] **Step 5: Run real SPMD gates**

```bash
<configured-lit> -sv --show-unsupported build/wafer-dev/test \
  --filter='Tools/wafer-opt-(spmd-partition|distributed-program)\.test|Spmd/'
```

Expected: tests execute with `xla-spmd-helper`; if unsupported, this task remains incomplete.

- [ ] **Step 6: Commit**

```bash
git add include/Wafer/Transforms/SPMD lib/Wafer/Transforms/SPMD lib/Wafer/Frontend \
  tools/wafer-opt include/Wafer/Transforms/Passes.td test
git commit -m "Materialize canonical distributed execution identity"
```

### Task 7: Candidate Rank and Entry Identity

**Files:**
- Create: `include/Wafer/IR/Executable/ExecutableOps.td`
- Create: `lib/Wafer/IR/Executable/ExecutableOps.cpp`
- Modify: `include/Wafer/IR/WaferOps.td`
- Modify: `lib/Wafer/IR/CMakeLists.txt`
- Create: `lib/Wafer/Transforms/Executable/MaterializeCandidateEntries.cpp`
- Create: `include/Wafer/Compiler/ExecutableIdentityBuilder.h`
- Create: `lib/Wafer/Compiler/ExecutableIdentityBuilder.cpp`
- Modify: `lib/Wafer/Compiler/CMakeLists.txt`
- Create: `include/Wafer/Transforms/Executable/CandidateResourceBuilder.h`
- Create: `lib/Wafer/Transforms/Executable/CandidateResourceBuilder.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Modify: `lib/Wafer/Transforms/CMakeLists.txt`
- Modify: `include/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h`
- Modify: `lib/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.cpp`
- Modify: `include/Wafer/Transforms/Passes.td`
- Create: `test/Dialect/Wafer/Executable/candidate-identity.mlir`
- Create: `test/Transforms/materialize-candidate-entries.mlir`

**Interfaces:**
- Consumes: verified distributed program, explicit typed target requirements and normalized bounded shape-guard candidates
  from the production `CompilationRequest`.
- Produces: `CommitState::Candidate` executable variant with resource/rank/entry records and unchanged
  `ResourceId`/`ExecutionInstanceId`, plus complete typed source-member relations needed to form final model invocation mappings.

- [ ] **Step 1: Define identity subset**

```text
wafer.executable
wafer.executable.variant commit_state = candidate
wafer.executable.resource
wafer.executable.state_group
wafer.executable.rank
wafer.executable.entry
```

The executable root stores exact model-interface/distributed-program refs. Candidate entries retain typed
`ModelProgramMemberId -> ComponentId -> ExecutionInstanceId -> EntryId` source relations. Task 7 verifies that every declared
`ModelEntrypointId` root reaches at least one candidate entry through those relations, but it does not create a partial
`wafer.executable.invocation`: completion roots/terminals do not exist until the Whole plan's completion-graph task. The later
task must derive the mapping from these same IR relations, not from a side table or symbol reachability search.

`ExecutableResourceOp` initially projects the model/distributed `ResourceId`, role/type/access/lifetime/alias/update and
state-group ref. Its typed candidate realization records explicitly cover target/executable variants, canonical execution
instances and any required projection, and carry arena/placement, residency/streaming, storage/capacity and scope proposals;
no axis is represented by a missing field or wildcard. Immutable policy is explicit `resident` or `streamed_pending`.
`ExecutableStateGroupOp` projects each model group, complete member set and axis-covered realization; it selects
`atomic_version + (full_copy|page_cow)` or `in_place_poison_on_failure`, group scope and bounded update-footprint proposal.
Planning-created launch-visible workspace, cross-entry/resident staging and
control/status roots must create the same op form immediately using parent EntryId/scope, role and structural root ordinal,
never an SSA/symbol name. Entry-internal SPM temp/psum/staging with no KAD/runtime identity remains owned by instruction IR and
does not get an executable ResourceId. Accepted range/capacity are absent until planning/resource-view validation. Candidate
realizations use `ExecutionInstanceId` coverage; commit may normalize only proven-equivalent coverage to final `RankClassId`
refs. `ExecutableRankOp` references
exactly one distributed instance and prerequisite class. `ExecutableEntryOp` references a static/rank-parametric function and
its ExecutionInstance coverage and ordered candidate slot bindings with typed
`StateSlotVersionRole = none|current|candidate|in_place`; final `RankClassId` coverage is absent in candidate state and must be a nonempty canonical set
in committed state. One rank-parametric EntryId may cover multiple final classes; duplicate same-ID entry records are illegal.

- [ ] **Step 2: Write verifier negatives**

Reject candidate record consumed by target, final class in candidate state, duplicate/missing resource or state-group
projection, incomplete/overlapping group members/axes, incompatible slot version role, state without exact group/snapshot
  policy, runtime handle/address in a resource record, accepted offset fabricated before planning,
missing instance coverage, coordinate mismatch, cross-prerequisite merge, and entry function without required partition/
replica ABI. Also reject empty/overlapping normalized shape guards, duplicate target/shape axis key, implicit/default target
  requirement, `pending_projection` or `streamed_pending` on any committed or target-consumed record. Reject any shape/count
  expression reference to undeclared `InvocationPolicyFieldId`, unknown `ModelEntrypointId`, entry without a typed source-member
  relation, or an entrypoint root whose member/component has no candidate EntryId.

- [ ] **Step 3: Build records before rank-local lowering**

```cpp
struct CandidateExecutableRequest {
  const VerifiedModelProgramView &modelProgram;
  DistributedProgramOp distributedProgram;
  llvm::ArrayRef<TargetRequirement> targetRequirements;
  llvm::ArrayRef<NormalizedShapeGuard> shapeGuards;
  CandidateResourcePolicies resourcePolicies;
  llvm::ArrayRef<InvocationPolicyFieldDecl> invocationPolicyFields;
};

mlir::FailureOr<ExecutableOp>
materializeCandidateExecutable(const CandidateExecutableRequest &request,
                               mlir::RewriterBase &rewriter);

mlir::LogicalResult materializeCandidateEntries(
    ExecutableVariantOp candidate,
    DistributedProgramOp distributedProgram);
```

The production builder creates the single executable root, canonical target-requirement records/`TargetVariantId`s, the
cross-product of non-overlapping normalized shape guards and compatible target requirements, and each
`ExecutableVariantId`; then it calls the entry helper for every candidate variant. Initial realization records carry explicit
`pending_projection` coverage as allowed only by `tasks/01` candidate lifecycle; launch projection must replace it with nonempty
ProjectionSetId coverage before commit. Streamed immutable records carry `streamed_pending`; downstream DDR/artifact tasks must
replace it with final windows/artifact refs. Build all executable state-group projections in the same transaction and verify
every persistent resource appears exactly once. No pass discovers an existing first variant, creates one from a test fixture or reads
target/shape/resource policy from globals/defaults. It clones component functions into candidate scope only when
per-rank-static specialization is required. Rank-parametric entries share the function and bind typed partition/replica slots.
Before returning, it merge-joins the immutable model-program view, distributed source-member map and candidate entries by typed
IDs and proves every API root is covered. It writes only the existing model/distributed/entry refs; no pending invocation
sidecar, API-name map or provisional completion record survives.

Add the only API for compiler-created launch-visible resources:

```cpp
struct CandidateResourceDeclaration {
  ExecutableEntryOp owner;
  std::uint32_t structuralRootOrdinal;
  ExecutableResourceRole role;
  mlir::Type semanticType;
  Access access;
  ResourceScope scope;
  CandidateRealizationCoverage coverage;
};

mlir::FailureOr<ExecutableResourceOp>
declareCandidateResource(const CandidateResourceDeclaration &declaration,
                         mlir::RewriterBase &rewriter);
```

The builder derives compiler-created ResourceId from EntryId/scope/role/root ordinal through generated
`ExecutableResourceIdentity` and domain `wafer.executable-resource.v1`, creates or verifies the candidate op and axis-covered
realization, and accepts idempotent reuse only for an identical declaration. Model/distributed resources retain their existing
ResourceId. Later traversal/layout/segmented/transport transforms call this API in the rewrite that first creates a
launch-visible root; ResourceView and commit are not fallback creators. Tests cover renamed SSA/symbols,
duplicate/conflicting declarations and exclusion of entry-internal SPM temps.

- [ ] **Step 4: Remove pass-only rank semantics**

Delete `logicalRank` options from `DumpGroupToTileRegionPass` and `ConvertGroupToTileRegionPass`. Change lowering APIs to accept:

```cpp
struct CandidateExecutionContext {
  ExecutableEntryOp entry;
  DistributedInstanceOp instance;
};
```

Collective local-rank derivation uses the instance coordinate and rank group. Debug tests wrap groups in a candidate entry instead of passing `--logical-rank=0`.

- [ ] **Step 5: Search for forbidden defaults**

```bash
rg -n 'logicalRank|logical-rank|default_mesh|partition_id.*0|replica_id.*0' \
  include lib tools test
```

Expected: no production rank identity comes from a default or pass option; any remaining literal zero is a test value or mathematically proven specialization with explicit record.

- [ ] **Step 6: Run and commit**

```bash
<configured-lit> -sv \
  build/wafer-dev/test/Dialect/Wafer/Executable \
  build/wafer-dev/test/Transforms/materialize-candidate-entries.mlir \
  build/wafer-dev/test/Transforms/convert-group-to-tile-region.mlir \
  build/wafer-dev/test/Transforms/dump-group-to-tile-region-collective-logical-rank.mlir
git add include/Wafer/IR/Executable lib/Wafer/IR/Executable \
  include/Wafer/Compiler/ExecutableIdentityBuilder.h \
  lib/Wafer/Compiler/ExecutableIdentityBuilder.cpp lib/Wafer/Compiler/CMakeLists.txt \
  lib/Wafer/Transforms/Executable include/Wafer/Transforms \
  include/Wafer/Conversion lib/Wafer/Conversion test
git commit -m "Make candidate lowering consume typed execution identity"
```

### Task 8: Identity Vertical Gate

**Files:**
- Create: `test/Integration/typed-distributed-identity.test`
- Modify: `test/CMakeLists.txt`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md` if a stable program replay command changes.

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces: Q0.2 completion evidence.

- [ ] **Step 1: Add two-coordinate vertical check**

Run real capture -> frontend materialization -> SPMD -> distributed assembly -> candidate entries. Check:

- same rank-parametric function can cover two instances;
- the instances bind different parameter slices and logical peers;
- per-rank-static case produces distinct specialization coverage;
- PP=2 and heterogeneous expert inputs pass model graph -> parallel formation -> per-component SPMD -> distributed assembly;
- state resources project one complete executable state group, slot roles are typed, and streamed immutable policy remains
  explicit candidate-only pending state;
- invocation-policy IDs and affine/FP8 semantic descriptors survive frontend/distributed/candidate identity;
- distinct prefill/decode `ModelEntrypointId`s retain different typed root/member coverage into candidate EntryIds without
  creating an incomplete invocation graph;
- no package/target artifact is emitted from candidate state.

- [ ] **Step 2: Run complete identity suite**

```bash
cmake --build build/wafer-dev --target check-wafer -- -j128
<configured-lit> -sv --show-unsupported build/wafer-dev/test \
  --filter='Integration/typed-distributed-identity\.test|Tools/wafer-opt-(spmd-partition|distributed-program)\.test|Dialect/Wafer/(Model|Distributed|Executable)/'
```

- [ ] **Step 3: Update queue and commit**

Mark Q0.2 done only if the real SPMD-dependent tests executed. Commit the integration test and exact evidence.

## Self-Review

- Spec coverage: typed model ABI/program graph/state groups/invocation fields/quant semantics, target environment/arenas/
  low-precision profiles, structured MPMD formation, canonical coordinates, prerequisite classes, shards, global variants and
  candidate identity all have explicit implementation owners.
- Placeholder scan: each task has exact files, interfaces, test forms, commands and rejection criteria.
- Type consistency: `ResourceId` originates in model IR, is referenced by distributed shards, and is not replaced by payload names; `ExecutionInstanceId` originates in distributed IR and is preserved into candidate entries.
- Proof consistency: frontend program/source/model view and canonical owner token move together; candidate API coverage is
  rederived from typed IR relations, and only the later completion task creates `wafer.executable.invocation`.
