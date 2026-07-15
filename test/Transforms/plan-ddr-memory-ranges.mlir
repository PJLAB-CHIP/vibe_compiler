// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=1024 ddr-largest-contiguous-bytes=1024' %s | FileCheck %s

func.func @plan_compiler_managed_ddr_range() {
  %ddr = memref.alloc()
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %region = wafer.tile.region(%ddr
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %input = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %input to %out
        {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
         dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
    wafer.tile.yield %out : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @plan_compiler_managed_ddr_range
// CHECK: memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<2x3xf16, #wafer.memory<ddr, tensor>>

func.func @keep_overlapping_ddr_ranges_distinct() {
  %ddr0 = memref.alloc()
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %ddr1 = memref.alloc()
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %spm0 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
  %spm1 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm0 to %ddr0
      {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
       dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
     to memref<2x3xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.wdma %spm1 to %ddr1
      {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
       dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
     to memref<2x3xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.local_fence
  return
}

// CHECK-LABEL: func.func @keep_overlapping_ddr_ranges_distinct
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>

func.func @keep_tile_region_results_live() {
  %ddr0 = memref.alloc()
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %result0 = wafer.tile.region(%ddr0
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    wafer.tile.yield %arg0
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }

  %ddr1 = memref.alloc()
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %result1 = wafer.tile.region(%ddr1
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    wafer.tile.yield %arg0
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }

  %sink = wafer.tile.region(
      %result0, %result1
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
       %arg1: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    wafer.tile.yield %arg0
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @keep_tile_region_results_live
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>

func.func @reuse_non_overlapping_ddr_ranges() {
  %ddr0 = memref.alloc()
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %spm0 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm0 to %ddr0
      {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
       dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
     to memref<2x3xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.local_fence

  %ddr1 = memref.alloc()
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %spm1 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm1 to %ddr1
      {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
       dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
     to memref<2x3xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.local_fence
  return
}

// CHECK-LABEL: func.func @reuse_non_overlapping_ddr_ranges
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>

func.func @reuse_exclusive_if_branch_ddr_ranges(%cond: i1) {
  scf.if %cond {
    %then_ddr = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    %then_spm = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %then_spm to %then_ddr
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
  } else {
    %else_ddr = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    %else_spm = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %else_spm to %else_ddr
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
  }
  return
}

// CHECK-LABEL: func.func @reuse_exclusive_if_branch_ddr_ranges
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>

func.func @for_carried_ddr_lifetime_blocks_reuse() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %init = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %next = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%carried = %init)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    %spm = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %spm to %carried
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
    scf.yield %next : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  %after = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %after_spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %after_spm to %after
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.local_fence
  return
}

// CHECK-LABEL: func.func @for_carried_ddr_lifetime_blocks_reuse
// CHECK: %[[INIT:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: %[[NEXT:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// CHECK: %[[AFTER:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
