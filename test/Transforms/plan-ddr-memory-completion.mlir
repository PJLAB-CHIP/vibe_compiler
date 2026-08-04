// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=2048 ddr-largest-contiguous-bytes=2048' %s | FileCheck %s

func.func @managed_wdma_roots_live_until_ncc_join() {
  %ddr0 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm0 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm0 to %ddr0
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>

  %ddr1 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm1 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm1 to %ddr1
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.ncc_join [0]

  %ddr2 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm2 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm2 to %ddr2
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @managed_wdma_roots_live_until_ncc_join
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// CHECK: wafer.instr.ncc_join [0]
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>

func.func @managed_rdma_roots_live_until_ncc_join() {
  %ddr0 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm0 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr0 to %spm0
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>

  %ddr1 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm1 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr1 to %spm1
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]

  %ddr2 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm2 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr2 to %spm2
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @managed_rdma_roots_live_until_ncc_join
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// CHECK: wafer.instr.ncc_join [0]
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>

func.func @both_branch_fences_complete_prior_issue(%cond: i1) {
  %ddr0 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm0 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr0 to %spm0
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  scf.if %cond {
    wafer.instr.ncc_join [0]
  } else {
    wafer.instr.ncc_join [0]
  }

  %ddr1 = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm1 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr1 to %spm1
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @both_branch_fences_complete_prior_issue
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: scf.if
// CHECK: wafer.instr.ncc_join [0]
// CHECK: else
// CHECK: wafer.instr.ncc_join [0]
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>

func.func @loop_body_issue_completed_in_body(
    %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    %ddr = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    %spm = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %ddr to %spm
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<128xf16, #wafer.memory<ddr, tensor>>
       to memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
  }
  return
}

// CHECK-LABEL: func.func @loop_body_issue_completed_in_body
// CHECK: scf.for
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: wafer.instr.rdma
// CHECK: wafer.instr.ncc_join [0]

func.func @managed_subview_extends_root_lifetime() {
  %root = memref.alloc()
      : memref<256xf16, #wafer.memory<ddr, tensor>>
  %view = memref.subview %root[64] [128] [1]
      : memref<256xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, strided<[1], offset: 64>, #wafer.memory<ddr, tensor>>
  %spm0 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %view to %spm0
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, strided<[1], offset: 64>, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>

  %other = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm1 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %other to %spm1
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]

  %after = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm2 = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %after to %spm2
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @managed_subview_extends_root_lifetime
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: memref.subview
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<512>
// CHECK: wafer.instr.ncc_join [0]
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
