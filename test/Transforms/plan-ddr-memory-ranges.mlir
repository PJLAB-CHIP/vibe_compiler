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
    wafer.instr.ncc_join [0]
    wafer.tile.yield %out : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @plan_compiler_managed_ddr_range
// CHECK: memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<2x3xf16, #wafer.memory<ddr, tensor>>

func.func @plan_stage_shifted_loop_ddr_view() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %ddr = memref.alloc()
      : memref<8xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %stage_displacement = arith.muli %c1, %c2 : index
    %shifted_iv = arith.addi %iv, %stage_displacement : index
    %tile = memref.subview %ddr[%shifted_iv] [1] [1]
        : memref<8xf16, #wafer.memory<ddr, tensor>>
       to memref<1xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %tile to %spm
        {byte_count = 2 : i64, inner_bytes = 2 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<1xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
       to memref<1xf16, #wafer.memory<spm, tensor>>
  }
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @plan_stage_shifted_loop_ddr_view
// CHECK: memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<8xf16, #wafer.memory<ddr, tensor>>
// CHECK: %[[SHIFTED:.+]] = arith.addi %{{.+}}, %{{.+}} : index
// CHECK: memref.subview %{{.+}}[%[[SHIFTED]]]

func.func @plan_identity_carried_ddr_view() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c7 = arith.constant 7 : index
  %ddr = memref.alloc()
      : memref<8xf16, #wafer.memory<ddr, tensor>>
  %init = memref.subview %ddr[%c7] [1] [1]
      : memref<8xf16, #wafer.memory<ddr, tensor>>
     to memref<1xf16, strided<[1], offset: ?>,
               #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
  %result = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%carried = %init)
      -> (memref<1xf16, strided<[1], offset: ?>,
                    #wafer.memory<ddr, tensor>>) {
    wafer.instr.rdma %carried to %spm
        {byte_count = 2 : i64, inner_bytes = 2 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<1xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
       to memref<1xf16, #wafer.memory<spm, tensor>>
    scf.yield %carried
        : memref<1xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
  }
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @plan_identity_carried_ddr_view
// CHECK: memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<8xf16, #wafer.memory<ddr, tensor>>
// CHECK: scf.for
// CHECK: wafer.instr.rdma

func.func @plan_nested_identity_carried_ddr_view() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %ddr = memref.alloc()
      : memref<8xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
  %outer_result = scf.for %outer_iv = %c0 to %c1 step %c1
      iter_args(%outer = %ddr)
      -> (memref<8xf16, #wafer.memory<ddr, tensor>>) {
    %middle_result = scf.for %middle_iv = %c0 to %c1 step %c1
        iter_args(%middle = %outer)
        -> (memref<8xf16, #wafer.memory<ddr, tensor>>) {
      %inner_result = scf.for %inner_iv = %c0 to %c1 step %c1
          iter_args(%inner = %middle)
          -> (memref<8xf16, #wafer.memory<ddr, tensor>>) {
        %tile = memref.subview %inner[%c7] [1] [1]
            : memref<8xf16, #wafer.memory<ddr, tensor>>
           to memref<1xf16, strided<[1], offset: ?>,
                     #wafer.memory<ddr, tensor>>
        wafer.instr.wdma %spm to %tile
            {byte_count = 2 : i64, dst_iterations = array<i64: 1, 1, 1>,
             dst_strides = array<i64: 0, 0, 0>, inner_bytes = 2 : i64}
            : memref<1xf16, #wafer.memory<spm, tensor>>
           to memref<1xf16, strided<[1], offset: ?>,
                     #wafer.memory<ddr, tensor>>
        scf.yield %inner
            : memref<8xf16, #wafer.memory<ddr, tensor>>
      }
      scf.yield %inner_result
          : memref<8xf16, #wafer.memory<ddr, tensor>>
    }
    scf.yield %middle_result
        : memref<8xf16, #wafer.memory<ddr, tensor>>
  }
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @plan_nested_identity_carried_ddr_view
// CHECK: memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>} : memref<8xf16, #wafer.memory<ddr, tensor>>
// CHECK-COUNT-3: scf.for
// CHECK: wafer.instr.wdma

func.func @plan_finite_rotating_carried_ddr_views() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %slot0 = memref.alloc()
      : memref<8xf16, #wafer.memory<ddr, tensor>>
  %slot1 = memref.alloc()
      : memref<8xf16, #wafer.memory<ddr, tensor>>
  %slot2 = memref.alloc()
      : memref<8xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
  scf.for %iv = %c0 to %c4 step %c1
      iter_args(%carried0 = %slot0,
                %carried1 = %slot1,
                %carried2 = %slot2)
      -> (memref<8xf16, #wafer.memory<ddr, tensor>>,
          memref<8xf16, #wafer.memory<ddr, tensor>>,
          memref<8xf16, #wafer.memory<ddr, tensor>>) {
    %tile = memref.subview %carried0[%iv] [1] [1]
        : memref<8xf16, #wafer.memory<ddr, tensor>>
       to memref<1xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %tile to %spm
        {byte_count = 2 : i64, inner_bytes = 2 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<1xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
       to memref<1xf16, #wafer.memory<spm, tensor>>
    %next0 = memref.cast %carried1
        : memref<8xf16, #wafer.memory<ddr, tensor>>
       to memref<8xf16, #wafer.memory<ddr, tensor>>
    %next1 = memref.cast %carried2
        : memref<8xf16, #wafer.memory<ddr, tensor>>
       to memref<8xf16, #wafer.memory<ddr, tensor>>
    %next2 = memref.cast %carried0
        : memref<8xf16, #wafer.memory<ddr, tensor>>
       to memref<8xf16, #wafer.memory<ddr, tensor>>
    scf.yield %next0, %next1, %next2
        : memref<8xf16, #wafer.memory<ddr, tensor>>,
          memref<8xf16, #wafer.memory<ddr, tensor>>,
          memref<8xf16, #wafer.memory<ddr, tensor>>
  }
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @plan_finite_rotating_carried_ddr_views
// CHECK-COUNT-3: memref.alloc() {{.*}}wafer.ddr.offset =
// CHECK: scf.for
// CHECK: wafer.instr.rdma

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
  wafer.instr.ncc_join [0]
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
  wafer.instr.ncc_join [0]

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
  wafer.instr.ncc_join [0]
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
    wafer.instr.ncc_join [0]
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
    wafer.instr.ncc_join [0]
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
    wafer.instr.ncc_join [0]
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
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @for_carried_ddr_lifetime_blocks_reuse
// CHECK: %[[INIT:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: %[[NEXT:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// CHECK: %[[AFTER:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
