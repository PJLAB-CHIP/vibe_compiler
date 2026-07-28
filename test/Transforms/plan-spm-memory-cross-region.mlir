// RUN: split-file %s %t
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66048' %t/shared.mlir | FileCheck --check-prefix=SHARED %s
// RUN: not wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65792' %t/shared.mlir 2>&1 | FileCheck --check-prefix=OVERLAP %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/escape.mlir 2>&1 | FileCheck --check-prefix=ESCAPE %s
// RUN: wafer-opt --wafer-plan-spm-memory %t/cross-region-completion.mlir | FileCheck --check-prefix=CROSS-REGION-COMPLETION %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/live-across-direct-call.mlir 2>&1 | FileCheck --check-prefix=LIVE-DIRECT %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/live-across-external-call.mlir 2>&1 | FileCheck --check-prefix=LIVE-EXTERNAL %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/live-across-indirect-call.mlir 2>&1 | FileCheck --check-prefix=LIVE-INDIRECT %s

//--- shared.mlir
func.func private @safe_scalar_helper() {
  return
}

func.func @share_across_sibling_regions(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %resident = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %produced = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %produced, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.local_fence
    wafer.tile.yield %produced
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }

  // A closed callee without an SPM arena remains legal while the resident is
  // live; the fail-closed boundary targets only calls that may clobber SPM.
  func.call @safe_scalar_helper() : () -> ()

  %written = wafer.tile.region(%resident, %boundary
      : memref<128xf16, #wafer.memory<spm, tensor>>,
        memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%input: memref<128xf16, #wafer.memory<spm, tensor>>,
       %ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %overlapping = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %overlapping, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.wdma %input to %ddr
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }

  %final = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %reusable = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %reusable, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.local_fence
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SHARED-LABEL: func.func @share_across_sibling_regions
// SHARED: %[[RESIDENT:.+]] = wafer.tile.region
// SHARED: %[[PRODUCED:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
// SHARED: wafer.tile.yield %[[PRODUCED]] : memref<128xf16, #wafer.memory<spm, tensor>>
// SHARED: wafer.tile.region(%[[RESIDENT]],
// SHARED: ^bb0(%[[INPUT:.+]]: memref<128xf16, #wafer.memory<spm, tensor>>
// SHARED: %[[OVERLAPPING:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
// SHARED: wafer.instr.wdma %[[INPUT]]
// SHARED: %[[REUSABLE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}

// OVERLAP: capacity_overflow: SPM planning range [65536, 65792) has no valid static placement

//--- escape.mlir
func.func @escape_spm_ssa(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %resident = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %produced = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %produced
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  %c0 = arith.constant 0 : index
  %escaped = memref.load %resident[%c0]
      : memref<128xf16, #wafer.memory<spm, tensor>>
  return
}

// ESCAPE: unsupported_spm_planning_scope: SPM values outside wafer.tile.region may only flow through explicit tile-region operands/results

//--- cross-region-completion.mlir
func.func @cross_region_same_worker_completion(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %resident = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %produced = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %produced, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.tile.yield %produced
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  %forwarded = wafer.tile.region(%resident
      : memref<128xf16, #wafer.memory<spm, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%input: memref<128xf16, #wafer.memory<spm, tensor>>):
    // Pending NCC state is function-wide. This explicit later sibling
    // completion legitimately covers the same-worker producer stream.
    wafer.instr.local_fence
    wafer.tile.yield %input
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  return
}

// CROSS-REGION-COMPLETION-LABEL: func.func @cross_region_same_worker_completion
// CROSS-REGION-COMPLETION: wafer.instr.fill
// CROSS-REGION-COMPLETION: wafer.instr.local_fence

//--- live-across-direct-call.mlir
func.func @independently_planned_callee(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %callee_resident = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %spm
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  return
}

func.func @resident_across_direct_call(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %resident = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %spm
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  func.call @independently_planned_callee(%boundary)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> ()
  %forwarded = wafer.tile.region(%resident
      : memref<128xf16, #wafer.memory<spm, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%input: memref<128xf16, #wafer.memory<spm, tensor>>):
    wafer.tile.yield %input
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  return
}

// LIVE-DIRECT: unsupported_spm_planning_scope: live SPM storage crosses a call
// LIVE-DIRECT-SAME: independently planned physical SPM arena

//--- live-across-external-call.mlir
func.func private @unknown_spm_arena_effect()

func.func @resident_across_external_call(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %resident = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %spm
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  func.call @unknown_spm_arena_effect() : () -> ()
  %forwarded = wafer.tile.region(%resident
      : memref<128xf16, #wafer.memory<spm, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%input: memref<128xf16, #wafer.memory<spm, tensor>>):
    wafer.tile.yield %input
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  return
}

// LIVE-EXTERNAL: unsupported_spm_planning_scope: live SPM storage crosses a call
// LIVE-EXTERNAL-SAME: interprocedural arena/resource summaries are not available

//--- live-across-indirect-call.mlir
func.func @indirect_target() {
  return
}

func.func @resident_across_indirect_call(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %resident = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %spm
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  %callee = func.constant @indirect_target : () -> ()
  func.call_indirect %callee() : () -> ()
  %forwarded = wafer.tile.region(%resident
      : memref<128xf16, #wafer.memory<spm, tensor>>) ->
      (memref<128xf16, #wafer.memory<spm, tensor>>) {
  ^bb0(%input: memref<128xf16, #wafer.memory<spm, tensor>>):
    wafer.tile.yield %input
        : memref<128xf16, #wafer.memory<spm, tensor>>
  }
  return
}

// LIVE-INDIRECT: unsupported_spm_planning_scope: live SPM storage crosses a call
// LIVE-INDIRECT-SAME: independently planned physical SPM arena
