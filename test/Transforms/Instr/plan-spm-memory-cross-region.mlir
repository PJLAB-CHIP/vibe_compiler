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
  %spill = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %produced = wafer.tile.region(%boundary, %spill
      : memref<128xf16, #wafer.memory<ddr, tensor>>,
        memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%unused: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %materialized: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %resident = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %resident, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.wdma %resident to %materialized
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.ncc_join [0]
    wafer.tile.yield %materialized
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }

  // No SPM root crosses the region boundary, so a closed scalar callee is
  // transparent between the explicit store/completion and matching reload.
  func.call @safe_scalar_helper() : () -> ()

  %written = wafer.tile.region(%produced, %boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>,
        memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%input: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %reloaded = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %overlapping = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %reloaded
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<128xf16, #wafer.memory<ddr, tensor>>
       to memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %overlapping, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.wdma %reloaded to %ddr
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.ncc_join [0]
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
    wafer.instr.ncc_join [0]
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SHARED-LABEL: func.func @share_across_sibling_regions
// SHARED: %[[PRODUCED:.+]] = wafer.tile.region
// SHARED: %[[RESIDENT:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
// SHARED: wafer.instr.wdma %[[RESIDENT]]
// SHARED: wafer.tile.yield %{{.+}} : memref<128xf16, #wafer.memory<ddr, tensor>>
// SHARED: wafer.tile.region(%[[PRODUCED]],
// SHARED: ^bb0(%[[INPUT:[^ :]+]]: memref<128xf16, #wafer.memory<ddr, tensor>>
// SHARED: %[[RELOADED:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
// SHARED: %[[OVERLAPPING:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
// SHARED: wafer.instr.rdma %[[INPUT]] to %[[RELOADED]]
// SHARED: %[[REUSABLE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}

// OVERLAP: capacity_overflow: SPM planning range [65536, 65792) has no valid static placement

//--- escape.mlir
func.func @escape_spm_ssa(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %produced = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %escaped = memref.extract_aligned_pointer_as_index %produced
        : memref<128xf16, #wafer.memory<spm, tensor>> -> index
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// ESCAPE: unsupported_lifetime_alias: tracked SPM storage cannot escape through raw metadata

//--- cross-region-completion.mlir
func.func @cross_region_same_worker_completion(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %spill = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %materialized = wafer.tile.region(%boundary, %spill
      : memref<128xf16, #wafer.memory<ddr, tensor>>,
        memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%unused: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %produced = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %produced, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.wdma %produced to %ddr
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.ncc_join [0]
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  %forwarded = wafer.tile.region(%materialized
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%input: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %reloaded = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %reloaded
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<128xf16, #wafer.memory<ddr, tensor>>
       to memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    wafer.tile.yield %input
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CROSS-REGION-COMPLETION-LABEL: func.func @cross_region_same_worker_completion
// CROSS-REGION-COMPLETION: wafer.instr.fill
// CROSS-REGION-COMPLETION: wafer.instr.ncc_join [0]

//--- live-across-direct-call.mlir
func.func @independently_planned_callee(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %callee_result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %unused = memref.load %spm[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

func.func @resident_across_direct_call(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %before = memref.load %spm[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    func.call @independently_planned_callee(%ddr)
        : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> ()
    %after = memref.load %spm[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// LIVE-DIRECT: unsupported_spm_planning_scope: a call from an active or asynchronous SPM scope may dynamically execute another wafer.tile.region
// LIVE-DIRECT-SAME: independently planned physical SPM arena

//--- live-across-external-call.mlir
func.func private @unknown_spm_arena_effect()

func.func @resident_across_external_call(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %before = memref.load %spm[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    func.call @unknown_spm_arena_effect() : () -> ()
    %after = memref.load %spm[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// LIVE-EXTERNAL: unsupported_spm_planning_scope: a call from an active or asynchronous SPM scope may dynamically execute another wafer.tile.region
// LIVE-EXTERNAL-SAME: independently planned physical SPM arena

//--- live-across-indirect-call.mlir
func.func @indirect_target() {
  return
}

func.func @resident_across_indirect_call(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%ddr: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %before = memref.load %spm[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %callee = func.constant @indirect_target : () -> ()
    func.call_indirect %callee() : () -> ()
    %after = memref.load %spm[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %ddr
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// LIVE-INDIRECT: unsupported_spm_planning_scope: an indirect call from an active or asynchronous SPM scope may execute another wafer.tile.region
