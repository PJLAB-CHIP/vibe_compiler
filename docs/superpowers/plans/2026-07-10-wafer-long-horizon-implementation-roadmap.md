# Wafer Long-Horizon Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or
> superpowers:executing-plans. Follow the dependency phases below, not the numeric order of sections in individual plans.

**Goal:** 把`tasks/01-16`已收敛合同拆成可独立评审、可验证、无pre/post-commit循环的完整实施序列，并覆盖未来复杂大模型、
多target、多进程共享runtime、长decode、稀疏MoE和100GB级逻辑payload/state负载。

**Architecture:** 本文只编排依赖和checkpoint，不声明新IR/ABI。仓库共有本路线图与六份子系统计划，共七份当前计划文件；
长期语义仍只由编号设计文档拥有。Target correctness被有意拆成pre-Whole geometry core与post-commit conversion两段，
candidate legality proof永远不能伪装成committed artifact authority。

**Tech Stack:** C++17、MLIR/LLVM 20、StableHLO/Shardy/OpenXLA、ODS/TableGen、Protobuf、WCRE/SHA-256、RISC-V ELF、
lit/CTest/GTest、C++ runtime/provider adapters。

## Global Constraints

- production compile最终只由`--program-pipeline=stablehlo-to-executable`选择的direct owner-aware driver组织；局部
  `OpPassManager` pipeline只作IR-local transform/replay/negative coverage，不能持有source、payload或commit authority。
- 新IR/ABI字段先进入对应编号设计文档；计划、audit和test fixture不能成为第二份语义源。
- `CanonicalEncodingContext`、output scope/sink/limits、executable/target/package attachments只由一个
  `ProgramOutputTransaction`拥有，最终只有一次program-delivery publication。
- target/shape/execution-instance/rank-class/projection关系按typed key join，不物化无界Cartesian product。
- candidate只做transformation-local structure/geometry preflight；LLVM/KAD/object/ELF/set严格发生在atomic executable之后。
- semantic identity统一走bounded WCRE和domain-separated SHA-256；raw Protobuf/ELF/payload只拥有content digest。
- runtime只从verified program delivery/package构造session；deployment/process-scoped service context共享capacity、cache、
  state registry和authority fence，单session不能建立第二套semantic managers。
- Delivery metadata、Artifact metadata和Runtime artifact使用不可互转typed sessions，但全部host reader/FD/worker/bytes先计入
  runtime-neutral `HostVerificationRegistry`唯一physical ledger；runtime actual验证再联合service child和invocation capacity。
- runtime先load owner-backed metadata并做纯preflight，得到exact capacity domains后才取得provider authority/context并one-way
  bind；provider机制只来自registry-owned canonical per-domain capability sets，session/executor不接caller backend。
- module policy保留两个正交closed axes：verification=`eager_active_set|at_first_use`，residency=
  `eager_active_set|graph_liveness`；long-running release只由typed symbolic terminal rules驱动，不展开百万iteration图。
- 所有限制值positive checked，0不表示unbounded；充分limits/worker变化不得改变semantic output。
- 每项先写失败测试，再实现并运行fresh gate，再独立commit。unsupported/skipped、partial publish或局部FileCheck不算主线完成。
- 各计划命令中的`<configured-lit>`表示当前build的`CMakeCache.txt`中`LLVM_EXTERNAL_LIT`值；不得写开发机绝对路径。
- task/stage编号只能出现在计划/队列，不能进入pass/op/type/artifact/CLI/build target/diagnostic长期名称。

## Baseline

仓库当前已有StableHLO frontend、framework capture、Shardy/XLA SPMD helper、Linalg/group/tile/instruction构件、局部memory
planning、straight-line target call emission、Wafer CRT和历史package/no-card工具。typed model/distributed/executable、完整
whole-variant commit、structure-preserving target conversion、bounded WCRE/KAD/atomic target set、outer delivery、Protobuf
package和真实RuntimeSession仍按`tasks/progress.md`推进。本文不把旧局部构件当作新checkpoint已经完成。

## Dependency Graph

```text
Target Correctness Task 1 temporary fail-closed hotfix (parallel containment only)

Source-backed corpus/reference preparation (Q5.C, parallel fixture-only work)

Target Artifact shared foundation
  Proto/WCRE/schema/value types, canonical context, static closure,
  external symbol registry, base Tx81 command ABI/value abstractions,
  VerifiedTargetCompilationContextRegistry
        |
        v
Package/Runtime Task 0 Outer Program Delivery Foundation
  + runtime-neutral HostVerificationRegistry/typed budget capabilities
        |
        v
Typed model/program/target environment/parallel/SPMD/distributed/candidate identity
        |
        v
Target Correctness Task 4
  WaferIR structural InstructionGeometry + WaferTargetLegality target core
        |
        v
Whole-Variant plan
  ExecutableCompilationInput owner -> candidate allocation-view adapter -> complete planning/preflight
  -> selected bytes -> fresh commit preflight -> atomic executable attach
        |\
        | +--> Q7 HF mandatory-commit integration (after Q5.C corpus preparation)
        |      replays the same production driver and may proceed beside committed conversion
        v
Target Correctness Tasks 2/3/5
  compiler command-view base + sealed conversion request
  -> structure-preserving LLVM -> complete geometry-family coverage
        |
        v
Target Artifact post-commit build
  KAD builder/post-conversion cross-check -> command-family activation -> core/clone prelink
  -> module packing -> ELF/KAD -> complete target-set attach
        |
        v
Package/Runtime remaining tasks
  package build attach -> one delivery commit -> delivery/artifact metadata loaders -> pure exact-domain preflight
  -> one-way runtime bind/shared service context -> rolling graph/capacity
  -> migration semantic request -> deployment placement inventory -> pure exact-domain migration preflight
  -> multi-set bind/execution snapshot -> registry-owned provider execution
        |
        v
Real-model, large-scale, board and calibration gates
```

Source-backed corpus/reference preparation is the explicit fixture-only Q5.C queue item and may run beside the early phases.
It advances no production gate by itself. Q7 becomes ready only after its Whole-Variant prerequisites and Q5.C are complete;
the remaining real-model/runtime/board gates wait for their explicit queue prerequisites.

## Plan Index

| Phase | Plan | Scope and completion output |
| --- | --- | --- |
| containment | [Target Correctness Foundation](2026-07-10-target-correctness-foundation.md) Task 1 | immediate no-mutation rejection for old flattening path; temporary only |
| corpus preparation | [Real Model and Board Gates](2026-07-10-real-model-board-gates.md) Task 1 | source-backed program/payload fixtures and independent references only; no production gate advancement |
| shared target | [Target Artifact Set](2026-07-10-target-artifact-set.md) shared tasks | bounded WCRE/schema/IDs/KAD values, runtime-safe command ABI/value, static closure/external registry, target compilation context registry; no artifact publication |
| outer owner | [Package and Runtime](2026-07-10-package-runtime.md) Task 0 | unique `ProgramOutputTransaction`, runtime-neutral host verification ledger, typed metadata sessions, owner/session generations, staging tokens/views and trusted delivery reader |
| typed handoff | [Typed Program and Distributed Identity](2026-07-10-typed-program-distributed-identity.md) | model APIs/resources/state/quant, target/mesh, MPMD/SPMD, canonical instances and candidate source relations |
| pre-commit legality | [Target Correctness Foundation](2026-07-10-target-correctness-foundation.md) Task 4 + [Whole-Variant Executable](2026-07-10-whole-variant-executable.md) | split structural/target geometry libraries, candidate adapter, complete planning, selected artifacts and atomic executable attach; Task 14 plus Q5.C closes Q7 mandatory HF commit integration |
| committed conversion | [Target Correctness Foundation](2026-07-10-target-correctness-foundation.md) Tasks 2/3/5 | MLIR-aware command-view base, sealed EntryCore/CloneDependency requests, structure semantics, registered calls and complete committed geometry coverage |
| target delivery | [Target Artifact Set](2026-07-10-target-artifact-set.md) post-commit tasks | post-conversion KAD cross-check, command-family activation, target build session/staging, deterministic module packing, verified ELF/KAD and complete target-set attachment |
| package/runtime | [Package and Runtime](2026-07-10-package-runtime.md) remaining tasks | package build session, single publication, two-phase metadata/runtime bootstrap, registry-owned provider capabilities, rolling execution/state/capacity |
| workload evidence | [Real Model and Board Gates](2026-07-10-real-model-board-gates.md) | Q5 static mainline, Q6.B board launch, split Q8.N no-card/scale and Q8.B board/numeric evidence, then Q9 calibration; Q7 evidence is replayed rather than rescheduled here |

## Stable Cross-Plan Interfaces

| Producer | Stable output | Consumer |
| --- | --- | --- |
| target shared foundation | `abi::CanonicalEncodingContext`, strong IDs/descriptors, external registry, runtime-safe command ABI/value, static closure proofs, `VerifiedTargetCompilationContextRegistry` | compilation input, typed identity, target legality/conversion/artifacts |
| outer delivery foundation | `compiler::ProgramOutputTransaction`, owner/attachment generations, stage-specific access adapters/tokens/views | frontend, Whole, target and package build stages |
| typed plan | move-owned frontend `MLIRContext` + `MaterializedFrontendProgram`, no-mutation `VerifiedTargetProgramMaterializationPlan`, immutable model/mesh policy projections, distributed instances/classes and candidate entry/source relations | `CompilationRequest` then owner-bound `ExecutableCompilationInput`; direct driver materializes only a working clone, no raw module/context handoff |
| WaferIR geometry | target-independent `geometry::InstructionGeometry` | op verifiers and WaferTargetLegality |
| WaferTargetLegality | `target::VerifiedPhysicalAllocationView` adapters and `VerifiedInstructionGeometry` | Whole candidate preflight and committed conversion; no reverse dependency |
| Whole plan | `compiler::StagedExecutableToken` for one sealed committed executable whose lower commitment owns the exact target-context registry | target build session and explicit executable-only delivery scope |
| committed correctness | `conversion::ConvertedTargetLLVMModule` from sealed request only | target core/clone prelink and KAD builder |
| target plan | `compiler::StagedTargetArtifactSetToken`, locator-free verified set/module views and unbound target-source descriptors | package assembly view and target metadata loader; runtime bind creates open-capable sources |
| package stage | `compiler::StagedPackageBundleToken` then trusted `PublishedProgramDelivery` | deployment loader/runtime |
| verification support | one trusted `HostVerificationRegistry` physical ledger with nonconvertible Delivery/Artifact/Runtime host capabilities | typed metadata registries and exact-domain runtime contexts; support layer knows no semantic upper type |
| metadata bootstrap | `VerifiedProgramDelivery`, `LoadedPackageMetadata`, `LoadedTargetArtifactSetMetadata` with immutable backing and unbound sources | pure target/shape/domain preflight after metadata sessions may be destroyed |
| runtime service registry | authenticated deployment inventory/constraint catalog, one sealed provider/authority/state/host bootstrap, exact-domain shared `WaferRuntimeServiceContext`, private runtime artifact session and `RuntimeBoundPackage` | RuntimeSessions without caller backend/context/managers; all actual I/O uses joint accounting |
| migration bootstrap | sealed `StateMigrationPlacementRequest`, registry-signed `VerifiedStatePlacementInventory`, pure exact-domain placement plan, independent `RuntimeBoundMigrationArtifactBundle` and final execution-registry snapshot | one-use migration session plan and whole-plan batch CAS; no backing/provider side effect before stale-placement revalidation |
| calibration publisher | owner-backed profiles and non-null `VerifiedCalibrationProfileSet` | cost ordering only; never legality |

If implementation needs a different semantic field or authority, update the numbered owner first. File/class placement refinements that
preserve these contracts remain implementation decisions within the owning plan.

## Review Checkpoints

- [ ] **Checkpoint 1: Shared bounded identity and outer ownership**

  WCRE 100K/1M record streaming stays within configured scratch/RSS; strong IDs do not retain copied canonical vectors. One outer
  transaction owns stable canonical generation and distinct attachment-view generation; multi-target attach and crash leave no
  partially trusted delivery. Delivery/Artifact/runtime typed verification owners share one lower physical host ledger without
  a semantic-layer dependency cycle or session-upgrade path.

- [ ] **Checkpoint 2: Typed APIs and distributed identity**

  Distinct prefill/decode `ModelEntrypointId`s, state groups, invocation fields, quant semantics, PP/EP graph and canonical
  execution instances survive source-backed frontend/SPMD. No default rank/name/path side channel remains.

- [ ] **Checkpoint 3: Whole candidate legality and atomic executable**

  Full traversal/layout/low-precision/completion/SPM/DDR/window/transport/projection/resource relations pass for every rank.
  Candidate target preflight uses fresh owner-bound allocation view and emits no LLVM/KAD. Any late rank/blob failure leaves source
  and visible delivery unchanged; success attaches one sealed committed executable. The direct driver move-owns
  `ExecutableCompilationInput`; request destruction cannot dangle source or context, and no pure pass pipeline can publish.

- [ ] **Checkpoint 4: Committed conversion and target artifacts**

  False branch/loop/call semantics execute correctly; EntryCore and CloneDependency use the sole sealed request API. Committed
  geometry is independently replayed. The target KAD builder performs the post-conversion boundary/function cross-check.
  Deterministic module profile union/clone dedup, ELF notes/KAD/fingerprints/digests and
  complete set coverage close before target attachment; no raw path/staging/expected-field contract exists.

- [ ] **Checkpoint 5: Package, runtime and shared capacity/state**

  Latest package assembly view creates one bounded build session and one final delivery commit. Delivery/artifact metadata proofs
  survive parse-session destruction but expose only unbound sources. Pure preflight chooses exact domains before context/authority;
  one-way bind creates `RuntimeBoundPackage`. RuntimeSessions share one service context/coordinator/cache/state registry per
  domain-generation/authority and use both orthogonal module modes, symbolic release rules, rolling graph cursor, sparse activation
  and cancellation/deadline. Migration first seals typed endpoints/domain policy and resolved scopes, obtains a no-authority
  registry-signed placement inventory, then purely forms an exact-domain plan, binds a heterogeneous artifact bundle and takes the
  final execution snapshot. That snapshot revalidates placement/backing generation before one batch CAS publishes every new group；full-copy/COW scope
  relation保持bijection，只有trusted transform允许显式N:M。

- [ ] **Checkpoint 6: Complex-model scalability**

  Configured scale suite covers 10k frontend records, 1000-entry sparse solving, 100k resource/10k-rank joins, 1M WCRE/graph
  instances, thousands of blobs/modules/windows, low FD limits, 70B/100GB logical payload/state accounting and cross-process
  oversubscription. Smaller real bytes still exercise complete digest/copy; fake capacity evidence is not reported as full-data
  throughput/numeric evidence.

- [ ] **Checkpoint 7: Board numeric and calibration**

  Static/dynamic/stateful/migration/TP/PP/segmented sparse MoE/INT8 explicit dequant/FP8 cases execute on configured hardware with
  independent numeric references and typed completion/failure evidence. Bounded owner-backed multi-target calibration set affects
  score ordering only.

## Verification Matrix

Run after every implementation checkpoint:

```bash
git diff --check
python3 tools/check_ir_organization.py --root .
python3 tools/check_deps.py
cmake --build build/wafer-dev --target check-wafer -- -j128
ctest --test-dir build/wafer-dev --output-on-failure
<configured-lit> -sv --show-unsupported build/wafer-dev/test
```

For scale and board checkpoints also run their explicitly configured builds/labels:

```bash
ctest --test-dir build/wafer-scale -L wafer-scale --output-on-failure
ctest --test-dir build/wafer-board -L board --output-on-failure
```

Review unsupported/skipped output. A gate is incomplete if its required feature was not configured or its tests did not execute.

## Execution Rule

Start the temporary correctness hotfix and target shared foundation independently. After shared target types land, implement the
outer delivery foundation, then typed identity. Land the split geometry libraries before Whole candidate planning; finish and
attach the executable before starting sealed conversion/artifact emission. If Q5.C is complete, Q7 may replay the same production
driver beside committed conversion; it does not wait for package/runtime. Package/runtime follows complete target attachments.
Within package/runtime, land the lower host ledger and typed metadata loaders before any RuntimeSession API; then implement pure
exact-domain preflight, registry context acquisition and one-way bind in that order. Migration lands its pure semantic request,
side-effect-free deployment placement inventory and pure exact-domain preflight before exact context, artifact bind, final registry
snapshot and execution-plan work. Provider adapters only implement low-level per-domain capability sets.
Large corpus/reference fixtures may be prepared early only through Q5.C; completing it alone does not advance Q5/Q7/Q8.N/Q8.B or any
scale/board/profile queue state. Those states advance only after their producer dependencies and fresh gates pass.
Each independently reviewable task is one commit; update`tasks/progress.md` only after the corresponding checkpoint is genuinely
met, not after scaffold, parser roundtrip or partial stage success.
