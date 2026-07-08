# Target CRT Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Q2-Q3 by making Wafer-owned `wafer_tx81_*` symbols typed, repo-local, linkable, and covered by production symbol checks.

**Architecture:** The CRT header is the C ABI source of truth. Lowering emits fixed LLVM signatures that match the header, the repo-local CRT source defines every production symbol, and `tools/wafer_device_link.py` compiles that CRT object before linking and scanning the final kcore shared object for undefined Wafer-owned symbols.

**Tech Stack:** MLIR/LLVM dialect C++ lowering, TX8 public C wrapper headers, Python lit tool tests, Xuantie RISC-V GCC/readelf/nm, `FileCheck`.

## Global Constraints

- Work in the current checkout; do not create a worktree unless explicitly requested.
- Do not use `third_party/wafer_crt/lib/libvr.a` as the Wafer-owned CRT implementation.
- Do not add Direct DTE empty implementations; Direct DTE remains outside the production CRT closure.
- Production coverage must include every Q1 production `wafer_tx81_*` symbol, or the symbol must be structurally rejected and removed from the production closure.
- No production code without a failing test first.

---

### Task 1: Symbol Closure Checks

**Files:**
- Create: `tools/check_target_crt_symbols.py`
- Create: `test/Tools/wafer-target-crt-symbols.test`
- Read: `runtime/wafer_crt/include/wafer_tx81_crt.h`
- Read: `runtime/wafer_crt/src/wafer_tx81_crt.c`
- Read: `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp`

**Interfaces:**
- Produces: a command that fails when header/source/lowering disagree on production `wafer_tx81_*`.
- Consumes: concrete symbol names and C prototypes from `wafer_tx81_crt.h`.

- [x] **Step 1: Write failing lit test**

Add a lit test that runs:

```bash
%python %wafer_src_root/tools/check_target_crt_symbols.py --repo-root %wafer_src_root
```

Expected before implementation: FAIL because `runtime/wafer_crt/include/wafer_tx81_crt.h` does not exist.

- [x] **Step 2: Implement checker**

The checker reads `tasks/14-target-llvm-golden-packet.md` production symbol blocks, rejects Direct DTE symbols, parses `wafer_tx81_crt.h` prototypes, parses `wafer_tx81_crt.c` definitions, and verifies lowering references every production symbol except excluded DTE symbols.

- [x] **Step 3: Run red/green**

Run:

```bash
/root/miniconda3/bin/lit -sv build/wafer-dev/test/Tools/wafer-target-crt-symbols.test
```

Expected after Task 1 implementation but before CRT files: still FAIL for missing CRT files.

### Task 2: Typed Target LLVM Calls

**Files:**
- Modify: `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp`
- Modify: `test/Transforms/lower-instr-to-target-llvm.mlir`

**Interfaces:**
- Consumes: fixed `wafer_tx81_*` signature groups.
- Produces: LLVM dialect declarations/calls with `i64` addresses and `i32` scalar fields.

- [x] **Step 1: Write failing FileCheck expectations**

Update `lower-instr-to-target-llvm.mlir` so `LLVMIR` checks no longer accept `call void (...)`; they expect typed calls such as:

```llvm
call void @wafer_tx81_rdma(i64 %..., i64 %..., i32 12, i32 6, ...)
```

Expected before lowering change: FAIL because calls are vararg.

- [x] **Step 2: Implement typed call emission**

Replace the vararg call helper with signature-aware callee declaration. Use `i64` for addresses and `i32` for counts, formats, dimensions, shape, stride, iteration, kind, zero-point, rounding, scale, probability, and indices. Normalize rank-dependent reduce shape to fixed `n,h,w,c`.

- [x] **Step 3: Run red/green**

Run:

```bash
/root/miniconda3/bin/lit -sv build/wafer-dev/test/Transforms/lower-instr-to-target-llvm.mlir
```

Expected after implementation: PASS.

### Task 3: Repo-Local Wafer CRT

**Files:**
- Create: `runtime/wafer_crt/include/wafer_tx81_crt.h`
- Create: `runtime/wafer_crt/src/wafer_tx81_crt.c`
- Modify: `tools/check_target_crt_symbols.py`
- Create or modify: `test/Tools/wafer-target-crt-symbols.test`

**Interfaces:**
- Produces: definitions for every Q1 production `wafer_tx81_*` symbol.
- Consumes: TX8 public wrapper headers `instr_adapter.h` and `instr_adapter_plat.h`.

- [x] **Step 1: Add header/source skeleton and keep checker red**

Add prototypes and definitions for production symbols. Before the implementation is complete, run the checker and confirm it fails on missing definitions or mismatched prototypes.

- [x] **Step 2: Implement wrapper dispatch**

Each CRT function builds the public `Tsm*Instr` packet, calls the relevant public wrapper function, then calls `TsmExecute` or `TsmWaitfinish`. Use wrapper families for RDMA, WDMA, data move, CT arithmetic/relation/logic/transcendental/activation/reduce/convert/pool/unpool/peripheral, NE GEMM/conv/depthwise/backward conv, and sync.

- [x] **Step 3: Verify symbol closure**

Run:

```bash
/root/miniconda3/bin/lit -sv build/wafer-dev/test/Tools/wafer-target-crt-symbols.test
```

Expected after implementation: PASS.

### Task 4: Device Link Required-Symbol Gate

**Files:**
- Modify: `tools/wafer_device_link.py`
- Modify: `test/Tools/wafer-device-link.test`
- Read: `test/Tools/Inputs/minimal-target-kernel.ll`

**Interfaces:**
- Consumes: repo-local CRT source and target LLVM IR.
- Produces: target object, CRT object, linked kcore `.so`, and undefined `wafer_tx81_*` scan.

- [x] **Step 1: Write failing tool expectations**

Update `wafer-device-link.test` to expect a CRT compile command, no `third_party/wafer_crt/lib`, no `-lvr`, and a required-symbol scan command.

- [x] **Step 2: Implement linker changes**

Compile `runtime/wafer_crt/src/wafer_tx81_crt.c` with TX8 GCC and include both `runtime/wafer_crt/include` and `third_party/tx8_deps/include`. Link target object + CRT object + `-lcommon_util -linstr_tx81 -llibc_stub`. After linking, run `readelf` or `nm` and fail on undefined `wafer_tx81_*`.

- [x] **Step 3: Run red/green**

Run:

```bash
/root/miniconda3/bin/lit -sv build/wafer-dev/test/Tools/wafer-device-link.test
```

Expected after implementation: PASS.

### Task 5: Final Verification and Commit

**Files:**
- Modify: `tasks/progress.md`
- Modify if needed: `tasks/14-target-llvm-golden-packet.md`
- Modify if needed: `memory/general_dev.md` or `memory/bugs.md`

**Interfaces:**
- Consumes: completed tasks 1-4.
- Produces: verified Q2-Q3 closure status, or explicit remaining limitation if full closure cannot be honestly claimed.

- [x] **Step 1: Run focused tests**

Run:

```bash
/root/miniconda3/bin/lit -sv build/wafer-dev/test/Tools/wafer-target-crt-symbols.test
/root/miniconda3/bin/lit -sv build/wafer-dev/test/Transforms/lower-instr-to-target-llvm.mlir
/root/miniconda3/bin/lit -sv build/wafer-dev/test/Tools/wafer-device-link.test
```

- [x] **Step 2: Run broader lit**

Run:

```bash
/root/miniconda3/bin/lit -sv --show-unsupported build/wafer-dev/test
```

- [x] **Step 3: Update task status**

Only mark Q2-Q3 `done` if the production closure, typed calls, CRT source, device link, and required-symbol gate all pass. Otherwise leave it `doing` and record the exact remaining blocker.

- [x] **Step 4: Commit**

Run:

```bash
git status --short
git add <changed files>
git commit -m "Close target CRT symbol linkage"
```
