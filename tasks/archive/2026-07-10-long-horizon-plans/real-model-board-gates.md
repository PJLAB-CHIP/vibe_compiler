# Real Model and Board Gates Implementation Plan

> **执行约束：** 任务状态和直接前置以 `tasks/progress.md` 为准；本文 checkbox 只拆解实现步骤，不是独立状态源。

**Goal:** 用真实导出模型、独立数值参考和明确的 no-card/board 证据，验证从 typed program 到 RuntimeSession 的完整主线，并用通过 correctness gate 的 profile 数据校准复杂大模型规划。

**Architecture:** 测试 corpus 只负责生成 source program、payload 和 reference，不持有编译器语义。所有 compiler gates 调用
`stablehlo-to-executable` direct production driver、target artifact publisher 和 package assembler；所有 runtime gates从同一verified
program delivery经metadata preflight和one-way bind加载PackageManifest。No-card/board只实现registry-owned low-level provider
capabilities；repo-owned final executor分别验证failure semantics与真实numeric/completion/PMU。性能数据只进入cost model calibration，
不回写legality或IR合同。

**Tech Stack:** PyTorch/XLA pinned importer、Hugging Face configuration、StableHLO/Shardy、Wafer production drivers、Protobuf package、C++ RuntimeSession/fake-tx/Tx/KMD provider capability sets、NumPy reference、lit/CTest、board profiler/PMU。

## Global Constraints

- 设计 owner：`tasks/01-16`；本文编排Q5、Q6.B、Q8.N、Q8.B和Q9的实现与证据，并重放Whole-Variant
  计划已经闭合的Q7证据；不声明新的IR、KAD、package或provider ABI。
- 测试模型必须从框架或忠实 exporter 进入 verified program；手写 `wafer.group`、instruction IR、LLVM IR 或 manifest不能证明 vertical gate。
- 每个 corpus case明确 source revision/config、seed、dtype、shape/bounds、payload digest和reference算法；路径和参数名只作测试定位。
- compiler artifact必须经过 mandatory whole-variant commit、真实 device link、ELF/KAD/fingerprint验证和 manifest semantic verifier。
- shape variant、rank projection、resource placement、transport和completion由 committed executable/package决定；test harness不得重建计划。
- no-card、single-card、multi-card和profile是不同 CTest labels；无对应环境时 board/profile测试不注册，不能以 skipped/unsupported 计通过。
- 数值容差按输出 dtype/算法在测试代码中显式给出；只比较shape、摘要、module load或provider success不算 numeric correctness。
- 每项先提交失败 gate，再实现所需生产能力或测试夹具；只有路线图列出的全部前置子计划完成后才能将本计划
  gate作为主线完成证明。
- CLI中的`forward`/`prefill`/`decode`只允许作为verified unique diagnostic alias解析成`ModelEntrypointId`；
  `RuntimeSession`、InvocationRequest、cache和trace始终使用typed ID。production API同时接受canonical ID spelling，alias
  ambiguity/unknown在variant selection和provider side effect前失败。
- 大模型scale gate把真实byte-integrity路径与逻辑容量/并发压力分开：较小真实payload必须完整hash/copy，70B/100GB-class
  accounting可使用明确test-only sparse/CAS/provider capability，但不能据此声称真实100GB numeric或cryptographic吞吐已通过。

## Queue Mapping And Execution Order

Task编号是本计划内的稳定work-package标识，执行必须遵循下列依赖，而不是按章节数字猜顺序：

```text
Task 1 (Q5.C corpus, early and fixture-only)
  -> Whole-Variant Task 14 (Q7 mandatory HF commit gate)
  -> Task 2 (Q5 static production mainline)
       |\
       | +-> Task 7 static-board checkpoint (Q6.B)
       v
     Tasks 3-6 + Task 9 scale gate (Q8.N)
       -> Task 7 complex-board checkpoint (Q8.B; also requires Q6.B)
       -> Q8 aggregate
       -> Task 8 profile calibration (Q9)
       -> Task 10 closure
```

Task 7因此有两个独立状态边界：static board evidence可以在Q5后关闭Q6.B；dynamic/stateful/parallel/quant/
migration等complex board evidence必须等待Q8.N，才能关闭Q8.B。Task 8即使在本文中出现在Task 9之前，也不能在
Q8 aggregate之前执行。

---

### Task 1: Source-Backed Workload Corpus and Independent References

**Files:**
- Create: `test/Integration/Inputs/wafer_model_corpus.py`
- Create: `test/Integration/Inputs/wafer_reference.py`
- Create: `test/Integration/Inputs/hf/tiny-llama-static.json`
- Create: `test/Integration/Inputs/hf/tiny-llama-dynamic.json`
- Create: `test/Integration/Inputs/hf/tiny-moe.json`
- Create: `test/Integration/model-corpus-contract.test`
- Modify: `test/Tools/Inputs/wafer_pytorch_xla_capture.py`
- Modify: `test/Tools/wafer_pytorch_xla_capture_contract_test.py`

**Interfaces:**
- Consumes: pinned PyTorch/XLA/Hugging Face imports, checked-in model configs, deterministic seeds.
- Produces: exporter-native program directories, raw payloads, invocation tensors and independent CPU reference outputs.

- [ ] **Step 1: Add corpus contract tests**

The test invokes every case twice in separate directories and asserts byte-identical metadata/payload digests. Required cases:

```text
static_transformer       batch=1 sequence=16 hidden=64
bounded_transformer      batch=[1,4] sequence=[1,128]
stateful_prefill_decode  typed prefill/decode APIs, prefill=16 decode_steps=2 kv_pages=8
tp_dp_transformer        tp=2 dp=2
pp_dp_transformer        pp=2 dp=2
segmented_moe            ep=4 top_k=2 ragged_token_counts
int8_linear              i8 storage/native integer result, explicit convert/dequant to f16
fp8_linear               e4m3/e5m2 storage selected by target capability
```

Dimensions are compact CI examples, not compiler protocol constants. Model config, mesh and invocation remain separate inputs so later tests can vary one axis without changing the others.

- [ ] **Step 2: Implement deterministic exporters**

Extend the existing capture helper with importable functions rather than another shell-oriented production tool:

```python
def emit_case(case_name: str, output_dir: pathlib.Path, *, seed: int) -> CaseEvidence:
    ...

@dataclasses.dataclass(frozen=True)
class CaseEvidence:
    source_digest: str
    payload_digests: tuple[str, ...]
    invocation_paths: tuple[pathlib.Path, ...]
    reference_paths: tuple[pathlib.Path, ...]
```

`CaseEvidence` is test-only provenance. It is not copied into Wafer IR or PackageManifest.

- [ ] **Step 3: Keep reference independent**

`wafer_reference.py` executes eager CPU operators using exported weights/inputs and stores full output tensors. Stateful reference explicitly returns the next KV state for prefill and both decode steps. MoE reference computes router counts/displacements and expert outputs without calling compiler/runtime helpers.

- [ ] **Step 4: Run and commit**

```bash
<configured-lit> -sv build/wafer-dev/test/Integration/model-corpus-contract.test
python3 test/Tools/wafer_pytorch_xla_capture_contract_test.py
git add test/Integration test/Tools/Inputs/wafer_pytorch_xla_capture.py \
  test/Tools/wafer_pytorch_xla_capture_contract_test.py
git commit -m "Add source-backed model integration corpus"
```

### Task 2: Static Transformer Mainline Gate

Queue mapping: consumes Q5.C, Q7 and the completed compiler/target/package/runtime prerequisites; advances Q5 only.

**Files:**
- Create: `test/Integration/static-transformer-mainline.test`
- Create: `tools/wafer_verify_integration_artifacts.py`
- Modify: `test/lit.cfg.py`
- Modify: `test/lit.site.cfg.py.in`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `static_transformer` program directory and target environment snapshot.
- Produces: committed executable, complete TargetArtifactSet, validated PackageManifest and no-card RuntimeSession trace.

- [ ] **Step 1: Add one failing production-chain test**

The test must invoke only production entrypoints:

```bash
wafer-opt --program-pipeline=stablehlo-to-executable \
  --input-program-dir INPUT --output-program-delivery DELIVERY
wafer-run --program-delivery=DELIVERY --entrypoint forward \
  --backend=fake-tx --module-verification=eager_active_set \
  --module-residency=graph_liveness --inputs INVOCATION
```

`DELIVERY` resolves through deployment-owned trust configuration to one `TrustedProgramDeliveryRef`; it is not an independently
trusted output directory. The exact driver options are implemented by the executable/artifact/package plans. This gate must not
fall back to `stablehlo-spmd-to-group`, separate program/package publications, a hand-assembled pass pipeline, regex export or
schema-v2 JSON.
This test is enabled only after the Package/Runtime plan extends the same direct driver to `package_bundle` completion scope:
the one command stages executable, every required target set and the package before its sole commit. It must not consume the
Whole plan's focused `executable_only` delivery, which is intentionally not runtime-loadable.
The diagnostic `--backend` option selects a reviewed low-level capability implementation when constructing the service registry;
it does not inject a caller `RuntimeBackend`. The two module flags are independent closed axes, never one combined admission enum.
Here `--entrypoint forward` exercises only the verified CLI alias resolver; the trace must show the resolved
`ModelEntrypointId`. Add the equivalent `--entrypoint-id=<canonical-id>` invocation and require identical selected graph/output.

- [ ] **Step 2: Verify evidence, not textual shape**

`wafer_verify_integration_artifacts.py` first calls the C++ program-delivery verifier, derives target/package child refs only
from that verified delivery, then calls existing C++ inspection/verification entrypoints and checks:

- executable is committed and has no candidate/group/template state;
- TargetArtifactSet member coverage equals executable entry/function coverage;
- each final ELF has a valid `.note.wafer.abi`, KAD and content digest;
- manifest blob digest and semantic ID verify;
- fake-tx trace instantiated the selected resource realizations and recorded allocation, copies, entries and terminal completion.

The C++ gate destroys Delivery and Artifact metadata sessions after producing owner-backed metadata, then performs pure preflight,
exact-domain context acquisition and one-way runtime bind. Before bind, unbound sources expose no open/read path and provider/
module counters stay zero. Delivery/Artifact/runtime typed budgets share the configured host physical ledger.

The Python helper orchestrates and formats failures; it does not parse IR text, ELF layout or Protobuf semantics itself.

- [ ] **Step 3: Register honest features and labels**

Add `wafer-mainline-integration` only when importer, SPMD helper, target toolchain and package/runtime executables are all configured. Add a `wafer-integration` CTest label whose tests fail when a configured prerequisite disappears; do not add `REQUIRES` for dependencies guaranteed by the integration build profile.

- [ ] **Step 4: Run and commit**

```bash
cmake --build build/wafer-dev --target wafer-opt wafer-run -- -j128
<configured-lit> -sv build/wafer-dev/test/Integration/static-transformer-mainline.test
ctest --test-dir build/wafer-dev -L wafer-integration --output-on-failure
git add test/Integration/static-transformer-mainline.test tools/wafer_verify_integration_artifacts.py \
  test/lit.cfg.py test/lit.site.cfg.py.in test/CMakeLists.txt
git commit -m "Gate the static transformer production mainline"
```

### Task 3: Bounded Dynamic Variant Selection Gate

Queue mapping: contributes the dynamic-selection portion of Q8.N; it does not advance Q8.N until Tasks 4-6 and Task 9
also satisfy their no-card/scale gates.

**Files:**
- Create: `test/Integration/bounded-dynamic-variants.test`
- Create: `unittests/Runtime/DynamicVariantIntegrationTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Modify: `test/Integration/Inputs/wafer_model_corpus.py`

**Interfaces:**
- Consumes: one verified bounded program, at least two committed static variants, invocation dimensions.
- Produces: one coherent target/shape/rank selection before allocation, or a side-effect-free rejection.

- [ ] **Step 1: Add selection and side-effect failure matrix**

Cover lower/upper bounds, variant boundary, uncovered in-bounds tuple, out-of-bounds tuple, arithmetic overflow and per-rank disagreement injection. Provider call log must remain empty for every rejected invocation.

- [ ] **Step 2: Preflight two successful shapes without claiming numeric execution**

Compile once, preflight/record `(batch=1, sequence=16)` and `(batch=4, sequence=128)`, and assert different static executable variant IDs with one coherent variant across all ranks. For every used ResourceId require exactly one selected target/executable/RankClass/projection realization and check distinct capacity/layout where expected. The repo-owned executor with no-card low-level capabilities records arguments and completion behavior only; full-output comparison is deferred to Task 7 board execution.

- [ ] **Step 3: Verify static target artifacts**

Inspect each selected entry and reject dynamic dimensions in target instruction geometry, memory ranges and KAD slot capacity. Dynamic shape evaluation belongs to RuntimeSession preflight only.

- [ ] **Step 4: Run and commit**

```bash
cmake --build build/wafer-dev --target WaferUnitTests wafer-run -- -j128
ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
<configured-lit> -sv build/wafer-dev/test/Integration/bounded-dynamic-variants.test
git add test/Integration/bounded-dynamic-variants.test \
  test/Integration/Inputs/wafer_model_corpus.py \
  unittests/Runtime/DynamicVariantIntegrationTest.cpp unittests/CMakeLists.txt
git commit -m "Validate bounded dynamic executable selection"
```

### Task 4: Stateful Prefill and Decode Gate

Queue mapping: contributes stateful/KV/migration no-card evidence to Q8.N; numeric state evidence remains Q8.B.

**Files:**
- Create: `test/Integration/stateful-prefill-decode.test`
- Create: `unittests/Runtime/StatefulExecutionIntegrationTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Modify: `test/Integration/Inputs/wafer_reference.py`

**Interfaces:**
- Consumes: immutable parameters, persistent KV resources, invocation workspace/user IO and distinct typed prefill/decode
  model entrypoints mapped to their committed invocation roots/terminals.
- Produces: no-card version/lease/state-machine evidence and deterministic failure-state behavior; numeric/state-byte evidence is produced by Task 7.

- [ ] **Step 1: Add lifecycle assertions**

Run typed prefill followed by two typed decode invocations as three one-invocation `RuntimeSession`s sharing one exact-domain
service context. Assert their different `ModelEntrypointId`s resolve different API
roots/IO contracts while sharing the intended lower-level entries; static `EntryId` or alias text cannot substitute. Assert
immutable weight handle reuse, stable persistent state identity/version progression and distinct workspace instances per
concurrent or sequential invocation. Verify that the repo-owned registry, residency caches and capacity authority are reused and that the current version remains
attachable; the test cannot inject or retain a standalone semantic `PersistentStateRegistry` manager.
Check every state/weight/workspace/page-table resource selects exactly one compatible realization for its target, variant,
RankClass and projection before registry/provider calls.

- [ ] **Step 2: Add atomic-version failure**

Inject a decode entry failure after writing the candidate state. Completion DAG must not publish the candidate version; a new
session attaches the previous version and records the expected next entry/state-transition plan without claiming output
computation. Task 7 proves the next decode numeric output on hardware.

- [ ] **Step 3: Add in-place poison failure**

Compile a policy variant that permits in-place mutation. Inject failure after the first irreversible state write, then assert the registry persists poison and every later attach/invocation rejects before launch, including after session recreation.

- [ ] **Step 4: Gate explicit cross-model state migration**

Create old/new model-interface identities with two related state groups and canonical ABI scope selectors. Exercise full-copy
and fenced page-COW across at least two resolved scope keys, then require one batch CAS publishes all new groups/scopes while all
old groups remain attachable. Build a sealed `StateMigrationPlacementRequest` with exact typed old/new target/executable/projection
selections and domain policy; invalid selectors or endpoints fail before any registry query. The deployment registry move-consumes
that request and signs one all-and-only `VerifiedStatePlacementInventory` with no backing handle or authority; pure placement must
determine exact domains from it plus the verified environment before context. A second execution preflight follows all-or-none
artifact bind and the final registry snapshot, which must reject stale placement/catalog/provider generation before backing or
provider effects. Inject a failure in the second group and require zero new-group
publication. Without the plan, attach to the new model returns
`state_model_mismatch`; diagnostic alias/namespace reuse cannot trigger migration. Trusted-transform numeric execution remains a
Task 7 board case once its migration target artifact exists.

- [ ] **Step 5: Keep numeric state assertions at the execution-capable gate**

Persist reference digests and expected tensors for Task 7. No-card tests assert only typed slot/state transitions, CAS publish,
exclusive in-place lease, poison/reset and failure ordering; they must not report logits or KV bytes as numerically validated.

- [ ] **Step 6: Run and commit**

```bash
cmake --build build/wafer-dev --target WaferUnitTests wafer-run -- -j128
ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
<configured-lit> -sv build/wafer-dev/test/Integration/stateful-prefill-decode.test
git add test/Integration/stateful-prefill-decode.test \
  test/Integration/Inputs/wafer_reference.py \
  unittests/Runtime/StatefulExecutionIntegrationTest.cpp unittests/CMakeLists.txt
git commit -m "Gate stateful prefill and decode execution"
```

### Task 5: Composite Parallel and Segmented MoE Gates

Queue mapping: contributes parallel/MoE no-card evidence to Q8.N; multi-card numeric/completion evidence remains Q8.B.

**Files:**
- Create: `test/Integration/composite-parallel-transformer.test`
- Create: `test/Integration/segmented-moe.test`
- Create: `unittests/Runtime/MultiRankFailureIntegrationTest.cpp`
- Modify: `unittests/CMakeLists.txt`

**Interfaces:**
- Consumes: TP+DP, PP+DP and EP distributed programs with explicit coordinates, shards, transports and projections.
- Produces: coherent multi-rank realization/issue/completion evidence and aggregated no-card failure results; board numeric evidence is produced by Task 7.

- [ ] **Step 1: Gate coordinate identity**

Compile TP=2/DP=2 and PP=2/DP=2 cases. The PP case uses at least two microbatches and must expose the bounded iteration domain,
zero/+1 iteration edges and overlapping stage-ready frontier. Inspect typed records through the C++ inspector and assert
component/stage, partition, replica and axis-role coordinates separately. Verify at least one shared RankClass and one split
caused by uneven shard or transport plan; flat rank is printed only as an ordinal.

- [ ] **Step 2: Gate ragged segmented exchange**

The MoE case uses source-backed non-uniform per-peer token-count scenarios including zero-count peers. Validate typed count/displacement capacities, selected per-rank resource realizations, transport binding member IDs and that count-phase completion dominates every data transfer. Preserve router/expert/output references for Task 7; fake-tx does not claim it computed them.
Require at least one zero-count expert wave: its predicate evaluates false, controlled weight-copy/module-load/entry/data nodes
become `skipped_success`, and blob-open/load/copy/submit counters remain zero for that expert. Active and skip paths join before
combine/reuse; runtime may not choose different experts or fall back to always-load-all.

- [ ] **Step 3: Add segmented negative matrix**

Reject negative/overflowing count or displacement, send/receive total mismatch, data issue before count completion, destination capacity overflow, missing peer status and projection-member digest mismatch. Rejection occurs before any rank launch.

- [ ] **Step 4: Aggregate rank failure**

Fake provider injects timeout, transport error and device error at one chosen rank/stage. Assert successors do not launch, other ranks reach a safe recovery wait, resources unwind in reverse dependency order and state follows its declared publish/poison policy.
For PP, inject failure in the first of at least two microbatches and prove remaining iteration nodes are suppressed while already
issued peer/stage work reaches cancel or safe terminal wait without deadlock.

- [ ] **Step 5: Run and commit**

```bash
cmake --build build/wafer-dev --target WaferUnitTests wafer-run -- -j128
ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
<configured-lit> -sv \
  build/wafer-dev/test/Integration/composite-parallel-transformer.test \
  build/wafer-dev/test/Integration/segmented-moe.test
git add test/Integration/composite-parallel-transformer.test \
  test/Integration/segmented-moe.test \
  unittests/Runtime/MultiRankFailureIntegrationTest.cpp unittests/CMakeLists.txt
git commit -m "Gate composite parallel and segmented MoE execution"
```

### Task 6: Quantized and Mixed-Precision Gate

Queue mapping: contributes quantized/FP8 selection and descriptor evidence to Q8.N; hardware numeric evidence remains Q8.B.

**Files:**
- Create: `test/Integration/quantized-mixed-precision.test`
- Create: `unittests/Runtime/TargetCapabilitySelectionTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Modify: `test/Integration/Inputs/wafer_reference.py`

**Interfaces:**
- Consumes: typed storage/quant descriptors, target capability fingerprints and integer/FP8 variants.
- Produces: target-compatible selected artifacts and descriptor/capability evidence; full numeric comparison is produced by Task 7.

- [ ] **Step 1: Add native integer result and explicit dequantization path**

Compile signed INT8 weights/activations with explicit q0/q1/zero-point, verified accumulator bounds and a native INT8 GEMM
destination. When the model API requires FP16, require an explicit registered INT8-to-FP16 convert/dequant instruction and
separate destination resource; the target must reject a GEMM profile that directly claims native FP16 output without capability
evidence. Verify KAD/resource descriptors and the uniquely selected realization preserve logical/storage dtype, scale/zp
resources, packing, capacity, rounding/saturation and target distinctions. Preserve the dequantized reference for Task 7;
no-card does not claim output computation.

- [ ] **Step 2: Add FP8 capability matrix**

Compile supported E4M3 and E5M2 variants with explicit accumulator/result dtype. A compatible environment selects and records
the correct variant/realization; an environment missing the required capability rejects before module load rather than
selecting another wrapper with different semantics. Actual execution is Task 7.

- [ ] **Step 3: Cover descriptor failures**

Reject scale/zero-point shape mismatch, unsupported rounding/saturation, packed capacity mismatch, accumulator overflow risk,
native-GEMM FP16 result without a matching profile, missing explicit convert/dequant and fingerprint mismatch. The harness must
not infer quantization from dtype strings or function names.

- [ ] **Step 4: Run and commit**

```bash
cmake --build build/wafer-dev --target WaferUnitTests wafer-run -- -j128
ctest --test-dir build/wafer-dev -R '^WaferUnitTests$' --output-on-failure
<configured-lit> -sv build/wafer-dev/test/Integration/quantized-mixed-precision.test
git add test/Integration/quantized-mixed-precision.test \
  test/Integration/Inputs/wafer_reference.py \
  unittests/Runtime/TargetCapabilitySelectionTest.cpp unittests/CMakeLists.txt
git commit -m "Gate quantized and mixed precision variants"
```

### Task 7: Split Static And Complex Board Execution Suite

Queue mapping: the static-transformer checkpoint advances Q6.B after Q5. The complex checkpoint advances Q8.B only after
Q6.B and Q8.N (including Task 9) are complete; one checkpoint cannot mark the other done.

**Files:**
- Create: `test/Board/CMakeLists.txt`
- Create: `test/Board/wafer_board_test.cpp`
- Create: `test/Board/BoardEvidence.cpp`
- Create: `test/Board/BoardEvidence.h`
- Modify: `CMakeLists.txt`
- Modify: `lib/Wafer/Runtime/CMakeLists.txt`
- Modify: `tools/wafer-run/wafer-run.cpp`

**Interfaces:**
- Consumes: Task 2's validated static package for Q6.B; Tasks 3-6 plus Task 9's completed Q8.N evidence for the later Q8.B
  checkpoint; registry-produced verified environment inventory and canonical per-domain Tx/KMD capability sets; no caller
  backend/context/manager.
- Produces: separately recorded Q6.B static board evidence and Q8.B complex numeric/completion/failure evidence, each bound
  to package, environment, firmware/runtime and test input digests.

- [ ] **Step 1: Register board tests only in board builds**

Add `WAFER_ENABLE_BOARD_TESTS` and a required configure-time board inventory file. When enabled, configuration fails if runtime provider, target inventory or test devices are missing. Register tests with `LABELS board`; the default build does not register them and therefore cannot report them as passed or skipped.

- [ ] **Step 2: Implement evidence recorder**

`BoardEvidence` records package manifest ID/blob digest, TargetArtifactSet ID, environment/topology/projection fingerprints, provider/driver/firmware versions, board IDs, invocation/reference digests, per-entry timestamps, typed completion results and numeric metrics. It is a test result artifact, never a compiler/runtime input.

- [ ] **Step 3: Execute the Q6.B static checkpoint, then gated single-card complex cases**

Run the static transformer first; its complete allocation/load/copy/entry/completion/error path and full-output comparison
close Q6.B independently. Run both dynamic shapes, stateful prefill/two-decode, a small full-copy/page-COW state migration,
and quantized/FP8 cases only after Q8.N is complete; these contribute to Q8.B. For INT8, compare both native integer output and the explicit converted/dequantized FP16 output.
Copy full outputs/state back and compare with the independent references prepared by earlier tasks. Confirm allocation/copy/
module/entry/completion/cleanup calls execute rather than stopping after symbol discovery, and verify the runtime-selected
realization tuple matches the package record for every bound resource.

- [ ] **Step 4: Execute multi-card completion/failure cases**

Run TP+DP, PP+DP and segmented MoE across the declared topology. Compare router counts/displacements, expert receive order and final outputs to the independent reference. Report host command completion, device local drain, DTE/collective wait and rank/stage join separately. Inject one rank timeout and one transport error through the board test provider hook; verify bounded timeout, global stop, cleanup and persistent-state consistency.

- [ ] **Step 5: Execute trusted-transform migration**

Run at least one verified trusted-transform migration with an explicit non-bijective component across multiple heterogeneous
TargetArtifactSets and resolved scope keys, such as TP 2-shard to 4-shard plus KV reshard. Compare every new member's numeric bytes
and exact scope/slot/shape/storage coverage to an independent transform, prove every old input group remains current and unmodified,
and require one whole-plan batch CAS publishes all new groups/scopes. Exercise a placement inventory where old backing and transform
domains span different verified provider generations; accept only when the typed domain policy plus placement/environment inventories prove the
required reimport/transfer relation. Run all four transform module verification/residency combinations and require identical state.
Inject failure after a non-final output and require zero new-group publication. If the configured board/runtime lacks the
required transform command capability, the case must return the typed structured-unsupported result and Task 7 cannot claim
all migration modes complete; it may not silently fall back to host copy or report the full-copy/COW cases as transform
evidence.

- [ ] **Step 6: Run and commit**

```bash
cmake -S . -B build/wafer-board -G Ninja \
  -DWAFER_ENABLE_BOARD_TESTS=ON \
  -DWAFER_BOARD_INVENTORY=/etc/wafer/board-inventory.json
cmake --build build/wafer-board --target wafer-board-tests -- -j128
ctest --test-dir build/wafer-board -L board --output-on-failure
git add CMakeLists.txt test/Board lib/Wafer/Runtime/CMakeLists.txt tools/wafer-run/wafer-run.cpp
git commit -m "Add explicit board numeric and completion gates"
```

### Task 8: Profile Evidence and Cost Calibration

Queue mapping: advances Q9 only after Q8.N, Q8.B and the Q8 aggregate gate are complete. It must not run merely because
the static Q6.B checkpoint has board evidence.

**Files:**
- Create: `schema/wafer/calibration_profile.proto`
- Modify: `schema/CMakeLists.txt`
- Create: `include/Wafer/Analysis/Cost/ProfileDatabase.h`
- Create: `include/Wafer/Analysis/Cost/CalibrationProfileLimits.h`
- Create: `lib/Wafer/Analysis/Cost/ProfileDatabase.cpp`
- Create: `lib/Wafer/Analysis/Cost/CalibrationProfileLimits.cpp`
- Create: `tools/wafer-profile-inspect/wafer-profile-inspect.cpp`
- Create: `tools/wafer-profile-inspect/CMakeLists.txt`
- Modify: `CMakeLists.txt`
- Create: `tools/wafer_profile_report.py`
- Create: `unittests/Analysis/ProfileDatabaseTest.cpp`
- Modify: `lib/Wafer/Analysis/CMakeLists.txt`
- Modify: `unittests/CMakeLists.txt`
- Modify: `lib/Wafer/Analysis/Group/TilingDemandAnalysis.cpp`
- Modify: `lib/Wafer/Analysis/Group/LayoutPlanningAnalysis.cpp`
- Modify: `lib/Wafer/Transforms/Target/MaterializeExecutionMesh.cpp`
- Modify: `lib/Wafer/Transforms/Executable/PhysicalTransportPlanning.cpp`
- Modify: `lib/Wafer/Transforms/Executable/SelectExecutableVariant.cpp`
- Modify: `include/Wafer/Pipelines/Pipelines.h`
- Modify: `lib/Wafer/Pipelines/Pipelines.cpp`
- Modify: `tools/wafer-opt/wafer-opt.cpp`
- Modify: `test/Board/BoardEvidence.cpp`
- Create: `test/Integration/calibration-profile-loading.test`

**Interfaces:**
- Consumes: completed Q8 aggregate, Task 7's complex-board evidence and Task 9's scale evidence, with target/profile
  provenance and PMU/timing measurements.
- Produces: deterministic versioned profile delivery bytes, immutable verified cost observations and explicit planner scoring
  input; legality remains independent.

```cpp
struct CalibrationProfileLimitValues {
  uint64_t maxSingleDeliveryBytes, maxTotalDeliveryBytes;
  uint64_t maxProfiles;
  uint64_t maxProvenanceFields, maxStringBytes, maxEvidenceDigests;
  uint64_t maxObservations, maxSamplesPerObservation, maxTotalSamples;
  uint64_t maxNestedRecords, maxScratchBytes, maxSpillBytes;
  uint64_t maxBufferBytes, maxWorkers;
};

class CalibrationProfileLimits final {
public:
  static llvm::Expected<CalibrationProfileLimits>
  create(const CalibrationProfileLimitValues &values);
  const CalibrationProfileLimitValues &values() const;
private:
  explicit CalibrationProfileLimits(CalibrationProfileLimitValues values);
  CalibrationProfileLimitValues values_;
};

class VerifiedBoardCalibrationInput final {
public:
  VerifiedBoardCalibrationInput() = delete;
  VerifiedBoardCalibrationInput(const VerifiedBoardCalibrationInput &) = delete;
  VerifiedBoardCalibrationInput(VerifiedBoardCalibrationInput &&) = default;
private:
  friend class board::BoardEvidence;
  // BoardEvidence is the only production/test-suite friend. Impl streams typed
  // observations from bounded owner-backed evidence.
  class Impl;
  explicit VerifiedBoardCalibrationInput(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class SerializedCalibrationProfile {
public:
  SerializedCalibrationProfile() = delete;
  SerializedCalibrationProfile(const SerializedCalibrationProfile &) = delete;
  SerializedCalibrationProfile(SerializedCalibrationProfile &&) = default;
  const abi::ImmutableByteBackingRef &backing() const;
  abi::ContentDigest contentDigest() const;
private:
  friend llvm::Expected<SerializedCalibrationProfile>
  buildAndVerifyCalibrationProfile(
      VerifiedBoardCalibrationInput,
      const CalibrationProfileLimits &);
  SerializedCalibrationProfile(abi::ImmutableByteBackingRef backing,
                               abi::ContentDigest digest);
  abi::ImmutableByteBackingRef backing_;
  abi::ContentDigest digest_;
};

llvm::Expected<SerializedCalibrationProfile>
buildAndVerifyCalibrationProfile(
    VerifiedBoardCalibrationInput input,
    const CalibrationProfileLimits &limits);

llvm::Expected<std::shared_ptr<const VerifiedCalibrationProfile>>
parseAndVerifyCalibrationProfile(
    abi::ImmutableByteBackingRef backing,
    abi::ContentDigest expectedContentDigest,
    TargetEnvironmentFingerprint expected,
    const CalibrationProfileLimits &limits);

llvm::Expected<std::shared_ptr<const VerifiedCalibrationProfileSet>>
buildVerifiedCalibrationProfileSet(
    llvm::ArrayRef<std::shared_ptr<const VerifiedCalibrationProfile>> profiles,
    const CalibrationProfileLimits &limits);
```

- [ ] **Step 1: Define a non-semantic profile database**

Define the generated schema and runtime-safe loader fixed by `tasks/04`: integer-only observation samples/summaries, typed
family/geometry/layout/dtype/topology keys, protocol/provenance and correctness digests. Key observations by exact target
environment fingerprint and measurement protocol. Reject unknown fields/version, target mismatch, failed correctness, missing
provenance, mixed clock/power policy, duplicate key or insufficient repetitions. Do not key by op/function/file names.

Unit tests cover deterministic delivery, alternate wire order with equal verified semantics, appended unknown fields, wrong
environment, overflow, duplicate observations, corrupted content digest and loader result immutability. Loader checks exact
content digest before parse. Validate every positive limit before allocating; count/measure the complete message, then stream
pinned deterministic bytes into immutable backing while hashing. Parser consumes only owner-backed bytes and checked-counts
through a schema-aware bounded wire preflight before invoking generated Protobuf parse; parse-then-count cannot protect the first
allocation. It validates length/recursion/repeated counts and checked totals before reserve. There is no production raw
`ArrayRef`, unbounded `SmallVector` or JSON parser; an inline adapter is test-only and
bounded by the same delivery limit.

- [ ] **Step 2: Add reproducible board sweeps**

After all board correctness cases pass, collect warmup-excluded integer samples/median/dispersion for compute, DMA, DTE,
collective and SPM address/bank sweeps on at least two distinct target environment fingerprints. `BoardEvidence` seals a
`VerifiedBoardCalibrationInput` with exact successful numeric/completion evidence digests and owner-backed typed observation
stream, calls the C++ publisher and writes deterministic profile bytes atomically. Preserve bounded raw observations so
calibration can be audited; failed board evidence cannot construct the input or publish.

- [ ] **Step 3: Calibrate scoring only**

Load every profile before constructing the production `CompilationRequest`, canonicalize by exact environment fingerprint and
pass one non-null `shared_ptr<const VerifiedCalibrationProfileSet>` into tiling/layout/communication scoring and mesh/transport/
candidate cost ordering. Empty set is the only absent representation; null is rejected. The outer program-directory CLI may
accept repeated `--calibration-profile=<profile.pb> --calibration-profile-digest=sha256:<hex>` pairs only as locator/integrity
input; each becomes an owner-backed parse and exact target-context join before set construction. Duplicate/wrong/stale target
profiles fail. A missing target entry uses documented conservative analytic estimates and cannot borrow another target's data.
No analysis opens a path or reads report JSON/global state, and legality APIs receive no profile parameter. Profile data must
never make an illegal candidate legal or change verifier results.

`calibration-profile-loading.test` covers canonical empty fallback, two-target exact lookup, valid set injection, null set,
duplicate target, corrupt digest, unknown version, wrong/stale environment and proves the same invalid candidate is rejected
with/without calibration.

- [ ] **Step 4: Add before/after regression**

For static transformer and segmented MoE, compare selected candidate score and board latency before/after calibration while asserting identical committed legality invariants and numeric output. A regression outside the recorded confidence threshold rejects the profile publication. `wafer-profile-inspect` loads/verifies the typed profile and emits diagnostic JSON; `wafer_profile_report.py` only orchestrates that command and has no planner-input format.

- [ ] **Step 5: Run and commit**

```bash
cmake --build build/wafer-board --target WaferUnitTests wafer-board-tests wafer-profile-inspect -- -j128
ctest --test-dir build/wafer-board -R '^WaferUnitTests$' --output-on-failure
ctest --test-dir build/wafer-board -L board --output-on-failure
python3 tools/wafer_profile_report.py \
  --profile build/wafer-board/calibration/profile.pb \
  --output build/wafer-board/profile-report.json
git add schema include/Wafer/Analysis/Cost lib/Wafer/Analysis \
  CMakeLists.txt \
  lib/Wafer/Transforms/Target/MaterializeExecutionMesh.cpp \
  lib/Wafer/Transforms/Executable/PhysicalTransportPlanning.cpp \
  lib/Wafer/Transforms/Executable/SelectExecutableVariant.cpp \
  include/Wafer/Pipelines/Pipelines.h lib/Wafer/Pipelines/Pipelines.cpp \
  tools/wafer-opt tools/wafer-profile-inspect tools/wafer_profile_report.py \
  unittests test/Board test/Integration/calibration-profile-loading.test
git commit -m "Calibrate planner costs from verified board profiles"
```

### Task 9: Large-Model Scale, Stress, and Bounded-Memory Gate

Queue mapping: together with Tasks 3-6, advances Q8.N. It produces no Q6.B/Q8.B numeric or board evidence.

**Files:**
- Create: `test/Scale/CMakeLists.txt`
- Create: `test/Scale/LargeModelFixture.h`
- Create: `test/Scale/LargeModelFixture.cpp`
- Create: `test/Scale/large-model-compiler.test`
- Create: `unittests/Compiler/LargeModelPlanningTest.cpp`
- Create: `unittests/Runtime/LargeModelRuntimeTest.cpp`
- Create: `unittests/Runtime/CrossProcessCapacityTest.cpp`
- Modify: `unittests/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: production frontend/identity/planner/output/package/runtime APIs plus explicitly test-only sparse/CAS/provider/
  durable-store capabilities that cannot construct semantic proofs.
- Produces: Q8.N's quantified peak-memory, count/limit, FD, transaction atomicity, rolling-graph, reservation and migration evidence.
  It does not claim full 70B numeric correctness or physical 100GB transfer throughput.

- [ ] **Step 1: Register an honest scale profile**

  Add `WAFER_ENABLE_SCALE_TESTS` and CTest label `wafer-scale`. When enabled, configure fixed positive RSS/FD/disk/time budgets
  and fail if the platform cannot enforce them. The normal developer suite keeps small deterministic regressions; queue closure
  for large-load readiness requires a fresh configured scale run, never an unsupported/skipped test.

- [ ] **Step 2: Stress frontend identity and canonical encoding**

  Generate deterministic source-backed modules with at least 10,000 functions/resources, high-fanout model edges and distinct
  prefill/decode APIs. Verify frontend byte/token/IR/model limits fail before clone mutation and sufficient limits produce stable
  ModelInterface/Resource/Dim/Entrypoint IDs. Stream 100K and 1M WCRE records through bounded scratch/external merge sort; assert
  fixed-size strong IDs, shared backing, collision comparison and measured peak RSS/scratch below configured limits. No full
  canonical byte vector may be retained. Build/parse a million-sample calibration profile through
  `CalibrationProfileLimits` and owner-backed streaming delivery, then form a two-target `VerifiedCalibrationProfileSet`;
  delivery bytes/digest remain identical across sufficient buffer/worker limits.

- [ ] **Step 3: Stress candidate solving and relational coverage**

  Build 1,000 entries with sparse local alternatives and late compatibility conflicts. Require fragment propagation/merge joins/
  nogoods to avoid the naive product; clone/backtracking/diagnostic/artifact-attempt limits are enforced and sufficient worker/
  memo changes select byte-identical winners. Build 100,000 resources across 10,000 ranks and multiple target/shape/projection
  records; ResourceView must use bounded canonical sort/merge/set-sweep rather than a Cartesian relation.

- [ ] **Step 4: Stress payload, module and outer delivery accounting**

  Use thousands of modules/blobs/windows and a 70B-class logical weight layout with at least 100GB declared payload through a
  test-only immutable sparse/CAS capability. Exercise checked logical/physical/root/chunk/FD/buffer counts, fixed 4MiB chunk
  intersection and selected-candidate-only materialization without reading all fake bytes. Separately stream a smaller real
  payload end-to-end and verify every byte/chunk/content digest. Under low `RLIMIT_NOFILE`, unselected blobs stay unopened and
  active handles remain bounded. Inject second-target, last-blob, package, fsync and final-CAS failures; no partial delivery is
  visible and orphan GC cannot create a trusted ref. Run concurrent delivery metadata, package metadata and runtime verification
  so their nonconvertible child sessions contend on one HostVerificationRegistry parent; measured FD/reader/worker/byte usage
  never exceeds the physical cap or resets when sessions are recreated.

- [ ] **Step 5: Stress rolling graph, sparse MoE and long decode**

  Instantiate a million-iteration verified graph through `VerifiedGraphInstanceCursor` and bounded trace sink without building
  the expanded DAG. Verify symbolic/exact-or-conservative-proven peak, bounded frontier/release and early cancellation. Run a
  long decode schedule with persistent KV versions and a sparse MoE fixture containing many zero-count experts/waves; false
  branches open/load/copy/submit nothing, active waves stay within the compiler envelope and combine waits for conditional join.
  Run all four `eager_active_set|at_first_use` verification by `eager_active_set|graph_liveness` residency combinations. Compare
  small graphs to full expansion and prove same-iteration+delta/final-tail/invocation-terminal rules keep million-iteration release
  state bounded without unloading before the last command completion.

- [ ] **Step 6: Stress capacity authority and state migration**

  Oversubscribe multiple provider capacity domains from concurrent sessions and processes. Test FIFO bounded wait, all-or-none
  prepare/commit/rollback, cache marginal accounting, generation invalidation, exclusive-local fencing and central/provider
  atomic authority; shared deployment without either authority fails before side effects. Migrate a 100GB-class logical state
  with multiple groups/scopes through full-copy, page-COW and trusted-transform accounting. Placement preflight over metadata
  chooses exact old/new/transform domains with zero authority/registry/blob operations; only then acquire context, all-or-none bind
  the heterogeneous artifact bundle and snapshot every scope. Execution preflight still has zero execution leases/opens;
  execution acquires all old-group/scope fenced leases together and one batch CAS publishes all new groups. Second-group failure,
  cancellation and crash leave every old group current and no partial new group. Include a trusted-transform TP 2-shard to
  4-shard/KV-reshard component spanning multiple target sets with exact N:M scope/input/output slot relations; full-copy/COW must
  reject that non-bijective relation.

- [ ] **Step 7: Run and commit**

  ```bash
  cmake -S . -B build/wafer-scale -G Ninja -DWAFER_ENABLE_SCALE_TESTS=ON
  cmake --build build/wafer-scale --target WaferUnitTests check-wafer -- -j128
  ctest --test-dir build/wafer-scale -L wafer-scale --output-on-failure
  <configured-lit> -sv --show-unsupported build/wafer-scale/test/Scale
  git add CMakeLists.txt test/Scale test/CMakeLists.txt \
    unittests/Compiler/LargeModelPlanningTest.cpp \
    unittests/Runtime/LargeModelRuntimeTest.cpp \
    unittests/Runtime/CrossProcessCapacityTest.cpp unittests/CMakeLists.txt
  git commit -m "Add bounded large-model scale gates"
  ```

### Task 10: Full Long-Horizon Gate and Queue Closure

**Files:**
- Modify: `test/CMakeLists.txt`
- Modify: `tasks/progress.md`
- Modify: `memory/general_dev.md`

**Interfaces:**
- Consumes: completed Q5/Q7, Tasks 3-6 plus Task 9 Q8.N evidence, separately completed Task 7 Q6.B/Q8.B checkpoints,
  Q8 aggregate and the final Task 8 Q9 profile suite.
- Produces: reproducible verification report and evidence-backed Q5/Q6.B/Q8.N/Q8.B/Q8/Q9 status, plus a replay check
  that Q7 remains closed through the same production driver.

- [ ] **Step 1: Run compiler/no-card gates from a clean build**

```bash
python3 tools/check_ir_organization.py --root .
python3 tools/check_deps.py
cmake --build build/wafer-dev --target check-wafer -- -j128
ctest --test-dir build/wafer-dev -L wafer-integration --output-on-failure
<configured-lit> -sv --show-unsupported build/wafer-dev/test
ctest --test-dir build/wafer-scale -L wafer-scale --output-on-failure
```

Review the unsupported list and confirm every mandatory no-card gate actually ran.

Run the configured scale build and archive its peak-memory/FD/count evidence. Large-load readiness remains incomplete if the
`wafer-scale` suite was not configured and executed.

- [ ] **Step 2: Run board/profile gates separately**

```bash
ctest --test-dir build/wafer-board -L board --output-on-failure
```

Archive BoardEvidence and profile report with exact commit and dependency manifest. Do not update board-related queue status when this suite did not execute.

- [ ] **Step 3: Update queue and memory from evidence**

Advance only the queue rows whose completion gates have fresh evidence. Record stable commands/evidence interpretation in `memory/general_dev.md`; keep individual run status and machine inventory out of stable memory.

- [ ] **Step 4: Final self-review and commit**

```bash
git diff --check
rg -n 'wafer\.group|schema v2|binding_order|completion_source' test/Integration test/Board
git add test/CMakeLists.txt tasks/progress.md memory/general_dev.md
git commit -m "Close verified long-horizon workload gates"
```

Expected search result: occurrences are negative assertions or explicitly named migration rejection cases, never production inputs or reconstructed semantics.
