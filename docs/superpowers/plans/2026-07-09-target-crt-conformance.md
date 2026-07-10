# Target CRT Conformance Implementation Plan

> **Historical plan:** This completed conformance batch is superseded as an execution entry. Do not extend or treat its
> legacy supporting-document marker check as a current status/ABI owner; its removal is separately tracked by Q13.T.
> Current work is controlled by `tasks/progress.md`, the numbered target design, and the 2026-07-10 long-horizon roadmap.

> **Historical checklist note:** Checkbox state and command blocks below preserve the original planning snapshot. In
> particular, unchecked boxes do not indicate current work. Do not execute or update this plan; use the current queue and
> roadmap named above.

**Goal:** Turn the old DLCompiler TX81 CRT source audit into a current Wafer CRT conformance gate, then fix the highest-confidence mismatch in argmax/argmin writeback.

**Architecture:** Keep the production ABI Wafer-owned: target LLVM emits `wafer_tx81_*`, not old `__*` symbols. A documentation matrix records how every production CRT family uses or rejects old source evidence; a Python checker enforces the parts that are practical to statically validate. The only code behavior change in this batch is mapping argmax/argmin writeback destinations as SPM offsets before Kcore stores.

**Tech Stack:** C target CRT in `runtime/wafer_crt`, TX8 public C headers in `third_party/tx8_deps/include`, Python 3 lit helper scripts, MLIR/lit tests under `test/Tools`, numbered design docs under `tasks/`.

## Global Constraints

- Do not restore `libvr.a`, old `__*` ABI symbols, Direct DTE stubs, or composite helpers as production CRT ABI.
- Current target LLVM call ABI uses fixed `wafer_tx81_*` C prototypes with `uint64_t` address fields and `uint32_t` descriptor fields.
- `argmax/argmin` value/index destinations are SPM offsets; CRT maps them with `get_spm_memory_mapping` before writing writeback values.
- Ordinary compute/move symbols issue packets and do not implicitly wait; argmax/argmin are the exception because public wrappers return scalar result in `wb_data0/wb_data1`.
- Verification must run the focused lit tool test and the conformance checker directly; full lit is only needed as final regression.

---

### Task 1: Current CRT Conformance Matrix

**Files:**
- Create: `docs/tx8-deps-reverse-engineering/tx81-current-crt-conformance-matrix.md`
- Modify: `docs/tx8-deps-reverse-engineering/README.md`

**Interfaces:**
- Consumes: `docs/tx8-deps-reverse-engineering/tx81-dlcompiler-crt-source-audit.md`
- Produces: stable matrix sections named `direct-wrapper-derived`, `public-header-derived`, `intentionally-excluded`, and `mismatch-fixed-this-batch`

- [ ] **Step 1: Write the matrix document**

Use these sections:

```markdown
# TX81 Current CRT Conformance Matrix

## Scope

- Current production surface: `runtime/wafer_crt/include/wafer_tx81_crt.h` and `runtime/wafer_crt/src/wafer_tx81_crt.c`.
- Evidence source: old DLCompiler `third_party/wafer/crt/lib/Tx81` source audit.
- This matrix is an audit/checklist, not a new ABI source; the ABI remains `tasks/14-target-llvm-golden-packet.md`.

## Pipeline position

Pipeline position:
- Upstream artifact / IR: memory-planned `wafer.instr.*` with accepted SPM/DDR offset facts.
- Current stage responsibility: lower fixed instruction ABI calls to Wafer-owned target CRT functions and issue public TX8 wrappers.
- Output artifact / IR: target LLVM / device object linked with repo-local Wafer CRT object.
- Downstream consumer: `tools/wafer_device_link.py` and package metadata/runtime launch stages.
- User-level driver / named pipeline: `--wafer-lower-instr-to-target-llvm`, `wafer-lower-groups-to-target-llvm`, and device-link lit tools.
- Explicit non-goals: old `__*` ABI, `libvr.a`, Direct DTE prototype helpers, composite GELU/MXFP/reduce-mul helpers.
- Completion gate: conformance checker, CRT symbol checker, focused lit tests, and full lit regression.

## Matrix

| CRT family | Current status | Old-source evidence used | Required conformance rule |
| --- | --- | --- | --- |
| RDMA / WDMA | direct-wrapper-derived | `__Rdma4d`, `__Wdma4d` | use `AddSrcDst` then `ConfigStrideIteration`; convert `inner_bytes` to element count; do not import old vectorize fallback |
| GatherScatter | direct-wrapper-derived | `__GatherScatter`, `__Memcpy` | preserve byte `inner_bytes` and source/dest stride-iteration order |
| Memset / Bit2Fp / MaskMove | direct-wrapper-derived | `__Memset`, `__Bit2Fp`, `__MaskMove` | use public wrappers; mask is `uint32_t` field/offset, not a generic 64-bit pointer |
| Elementwise arithmetic / relation / logic | direct-wrapper-derived | `arith.c`, `relation.c`, `logic.c`, unary activation/transcendental files | use fixed per-kind symbols; relation/logic bool branch only for `Fmt_BOOL`; no VS/VuV production ABI |
| Convert | direct-wrapper-derived | dtype conversion files | INT8 source uses zero point, FP/INT narrowing uses rounding, plain wrappers ignore extra ABI fields |
| Reduce | direct-wrapper-derived | `__ReduceSum/Avg/Max/Min` | native production reduce is sum/avg/max/min only; reduce-mul remains excluded |
| GEMM | direct-wrapper-derived | `__Gemm` | issue wrapper sequence and explicitly disable psum/bias/scale/quant/fused activation |
| Conv / Depthwise / BackwardConv | mixed: conv old source plus public headers | `__Conv`, public wrapper headers | preserve NHWC/HWOI field order and explicitly disable optional/fused features; old default ReLU is a negative example |
| Pool / Unpool | public-header-derived | no old source file | keep current public wrapper mapping; do not infer old source evidence |
| TDMA pad/img2col | direct-wrapper-derived | `__Pad`, `__Img2col` | use only V0 production TDMA surface; transform-like TDMA wrappers remain excluded |
| Peripheral argmax/argmin | mismatch-fixed-this-batch | `__ArgMax`, `__ArgMin`, SPM mapping helper | wait for writeback registers, map SPM offset destinations, then store value/index |
| Peripheral factorize/elem_mask | public-header-derived | no old source file | keep as public-header-derived only |
| Peripheral bilinear/LUT/rand | direct-wrapper-derived | `__Bilinear`, `__Lut16`, `__Lut32`, `__RandGen` | use public wrappers with fixed ABI fields; bilinear scale is derived from shapes |
| Count / Direct DTE / composite helpers | intentionally-excluded | `__Count`, `__Send`, `__Recv`, GELU/MXFP/reduce_mul/channelnorm/etc. | no production CRT symbol until IR/ABI represents result, scratch, ordering, endpoint/channel, and completion semantics |
```

- [ ] **Step 2: Add README navigation**

Add one row to `docs/tx8-deps-reverse-engineering/README.md` pointing to `tx81-current-crt-conformance-matrix.md`.

- [ ] **Step 3: Verify document references**

Run:

```bash
rg -n 'tx81-current-crt-conformance-matrix|direct-wrapper-derived|mismatch-fixed-this-batch' \
  docs/tx8-deps-reverse-engineering/README.md \
  docs/tx8-deps-reverse-engineering/tx81-current-crt-conformance-matrix.md
```

Expected: README row and matrix status lines are printed.

### Task 2: Static CRT Conformance Checker

**Files:**
- Create: `tools/check_target_crt_conformance.py`
- Modify: `test/Tools/wafer-target-crt-symbols.test`

**Interfaces:**
- Consumes: `runtime/wafer_crt/src/wafer_tx81_crt.c`, `tasks/14-target-llvm-golden-packet.md`, `docs/tx8-deps-reverse-engineering/tx81-current-crt-conformance-matrix.md`
- Produces: command output `checked Wafer target CRT conformance rules`

- [ ] **Step 1: Write the failing lit test**

Add this RUN line to `test/Tools/wafer-target-crt-symbols.test` before the compile RUN:

```text
# RUN: %python %wafer_src_root/tools/check_target_crt_conformance.py --repo-root %wafer_src_root
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
<configured-lit> -sv build/wafer-dev/test/Tools/wafer-target-crt-symbols.test
```

Expected: FAIL because `tools/check_target_crt_conformance.py` does not exist yet.

- [ ] **Step 3: Implement the checker**

Create `tools/check_target_crt_conformance.py` with rule checks for:

```text
arg writeback waits and stores through get_spm_memory_mapping
mask move truncates the mask argument to the public uint32_t wrapper field
RDMA/WDMA use AddSrcDst + ConfigStrideIteration and element count from inner_bytes
GEMM/Conv explicitly disable psum/bias/scale/quant/fused activation
convert macros preserve zero-point / rounding / plain split
relation/logic bool branches key off Fmt_BOOL
conformance matrix has required status markers
old __* ABI symbols and composite helper names are absent from runtime CRT source
```

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
<configured-lit> -sv build/wafer-dev/test/Tools/wafer-target-crt-symbols.test
```

Expected: PASS.

### Task 3: ArgMax/ArgMin SPM Writeback Fix

**Files:**
- Modify: `runtime/wafer_crt/src/wafer_tx81_crt.c`
- Modify: `tasks/14-target-llvm-golden-packet.md`
- Modify: `docs/tx8-deps-reverse-engineering/tx81-dlcompiler-crt-source-audit.md`

**Interfaces:**
- Consumes: `get_spm_memory_mapping(uint64_t offset)` public TX8 helper
- Produces: `wafer_arg_writeback` stores value/index through mapped SPM addresses

- [ ] **Step 1: Make the checker fail on current CRT**

Run:

```bash
python3 tools/check_target_crt_conformance.py --repo-root .
```

Expected before the code fix: FAIL on arg writeback mapped-store rule.

- [ ] **Step 2: Implement SPM offset mapping**

Change `runtime/wafer_crt/src/wafer_tx81_crt.c` to add:

```c
extern int8_t *get_spm_memory_mapping(uint64_t offset);

static uint64_t wafer_spm_mapped_addr(uint64_t offset) {
  return (uint64_t)(uintptr_t)get_spm_memory_mapping(offset);
}
```

Then call:

```c
wafer_store_value(wafer_spm_mapped_addr(value_dst), format, instr->param.wb_data0);
wafer_store_u32(wafer_spm_mapped_addr(index_dst), (uint32_t)instr->param.wb_data1);
```

- [ ] **Step 3: Update design language**

Update `tasks/14-target-llvm-golden-packet.md` ABI rules so argmax/argmin value/index destinations are explicitly SPM offsets and CRT performs mapping before writes.

Update the old-source audit follow-up from conditional language to fixed status: this batch fixes mapped SPM writeback.

- [ ] **Step 4: Run checker**

Run:

```bash
python3 tools/check_target_crt_conformance.py --repo-root .
```

Expected: `checked Wafer target CRT conformance rules`.

### Task 4: Regression Verification and Commit

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: tasks 1-3 deliverables.
- Produces: one git commit.

- [ ] **Step 1: Run formatting/diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 2: Run focused tool tests**

Run:

```bash
<configured-lit> -sv build/wafer-dev/test/Tools/wafer-target-crt-symbols.test
```

Expected: PASS.

- [ ] **Step 3: Run full lit regression**

Run:

```bash
<configured-lit> -sv --show-unsupported build/wafer-dev/test
```

Expected: all supported tests pass; unsupported list is reported explicitly.

- [ ] **Step 4: Commit**

Run:

```bash
git status --short
git add docs/superpowers/plans/2026-07-09-target-crt-conformance.md \
  docs/tx8-deps-reverse-engineering/README.md \
  docs/tx8-deps-reverse-engineering/tx81-current-crt-conformance-matrix.md \
  docs/tx8-deps-reverse-engineering/tx81-dlcompiler-crt-source-audit.md \
  tasks/14-target-llvm-golden-packet.md \
  runtime/wafer_crt/src/wafer_tx81_crt.c \
  tools/check_target_crt_conformance.py \
  test/Tools/wafer-target-crt-symbols.test
git commit -m "Harden target CRT conformance against TX81 source audit"
```

Expected: commit succeeds and `git status --short` is clean afterward.

## Self-Review

- Spec coverage: the plan covers matrix documentation, static conformance enforcement, arg writeback behavior, design text, focused lit, full lit, and commit.
- Placeholder scan: no `TBD`, `TODO`, or unspecified test step remains.
- Type consistency: helper uses `uint64_t` input offset and returns a `uint64_t` mapped address for existing `wafer_store_*` helpers.
