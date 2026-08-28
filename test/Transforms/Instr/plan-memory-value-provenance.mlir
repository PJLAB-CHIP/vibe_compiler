// RUN: split-file %s %t
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/external.mlir | FileCheck --check-prefix=EXTERNAL %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=768 ddr-largest-contiguous-bytes=768' %t/ddr-select.mlir | FileCheck --check-prefix=DDR-SELECT %s
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66304' %t/spm-select.mlir | FileCheck --check-prefix=SPM-SELECT %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/loop-origin.mlir 2>&1 | FileCheck --check-prefix=LOOP-ORIGIN-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=512 ddr-largest-contiguous-bytes=256' %t/loop-origin.mlir | FileCheck --check-prefix=LOOP-ORIGIN %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/repeatable-origin.mlir 2>&1 | FileCheck --check-prefix=REPEATABLE-ORIGIN-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=512 ddr-largest-contiguous-bytes=256' %t/repeatable-origin.mlir | FileCheck --check-prefix=REPEATABLE-ORIGIN %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=512 ddr-largest-contiguous-bytes=256' %t/positive-loop-result.mlir | FileCheck --check-prefix=POSITIVE-LOOP-RESULT %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/memory-space-cast.mlir 2>&1 | FileCheck --check-prefix=SPACE-CAST-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=512 ddr-largest-contiguous-bytes=256' %t/memory-space-cast.mlir | FileCheck --check-prefix=SPACE-CAST %s

//--- external.mlir
func.func @external_if_results_keep_one_root(
    %input: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1) {
  %first = scf.if %cond
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    scf.yield %input : memref<128xf16, #wafer.memory<ddr, tensor>>
  } else {
    scf.yield %input : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  %spm0 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %first to %spm0
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]

  %second = scf.if %cond
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    scf.yield %input : memref<128xf16, #wafer.memory<ddr, tensor>>
  } else {
    scf.yield %input : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  %spm1 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %second to %spm1
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

func.func @external_for_result_keeps_boundary_root(
    %input: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lb: index, %ub: index, %step: index) {
  %looped = scf.for %i = %lb to %ub step %step
      iter_args(%iter = %input)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    scf.yield %iter : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %looped to %spm
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// EXTERNAL-LABEL: func.func @external_if_results_keep_one_root
// EXTERNAL-LABEL: func.func @external_for_result_keeps_boundary_root

//--- ddr-select.mlir
func.func @select_keeps_all_managed_ddr_roots_live(%cond: i1) {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %b = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %selected = arith.select %cond, %a, %b
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm0 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %selected to %spm0
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>

  %c = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm1 = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %c to %spm1
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// DDR-SELECT-LABEL: func.func @select_keeps_all_managed_ddr_roots_live
// DDR-SELECT: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// DDR-SELECT: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// DDR-SELECT: arith.select
// DDR-SELECT: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<512>

//--- spm-select.mlir
func.func @select_keeps_all_managed_spm_roots_live(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1) {
  %region = wafer.tile.region(%boundary, %cond
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%output: memref<128xf16, #wafer.memory<ddr, tensor>>, %condition: i1):
    %a = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    %selected = arith.select %condition, %a, %b
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %selected to %output
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>

    %c = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %c to %output
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.ncc_join [0]
    wafer.tile.yield %output
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-SELECT-LABEL: func.func @select_keeps_all_managed_spm_roots_live
// SPM-SELECT: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65536>
// SPM-SELECT: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65792>
// SPM-SELECT: arith.select
// SPM-SELECT: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<66048>

//--- loop-origin.mlir
func.func @loop_body_view_sees_all_carried_external_origins(
    %a: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %b: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lower: index, %upper: index, %step: index) {
  %looped = scf.for %i = %lower to %upper step %step
      iter_args(%iter = %a)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    %view = memref.subview %iter[0] [128] [1]
        : memref<128xf16, #wafer.memory<ddr, tensor>>
       to memref<128xf16, strided<[1]>, #wafer.memory<ddr, tensor>>
    %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %view to %spm
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<128xf16, strided<[1]>, #wafer.memory<ddr, tensor>>
       to memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    scf.yield %b : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// LOOP-ORIGIN-CAPACITY: memory_capacity_overflow
// LOOP-ORIGIN-LABEL: func.func @loop_body_view_sees_all_carried_external_origins
// LOOP-ORIGIN: memref.subview
// LOOP-ORIGIN: wafer.instr.rdma

//--- repeatable-origin.mlir
func.func @repeatable_branch_origin_crosses_backedge(
    %a: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %b: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %false = arith.constant false
  %true = arith.constant true
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  %flag_result, %buffer_result = scf.for %i = %c0 to %c2 step %c1
      iter_args(%flag = %false, %iter = %a)
      -> (i1, memref<128xf16, #wafer.memory<ddr, tensor>>) {
    scf.if %flag {
      wafer.instr.rdma %iter to %spm
          {byte_count = 256 : i64, inner_bytes = 256 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<128xf16, #wafer.memory<ddr, tensor>>
         to memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
    }
    %next = arith.xori %flag, %true : i1
    scf.yield %next, %b
        : i1, memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// REPEATABLE-ORIGIN-CAPACITY: memory_capacity_overflow
// REPEATABLE-ORIGIN-LABEL: func.func @repeatable_branch_origin_crosses_backedge
// REPEATABLE-ORIGIN: scf.for
// REPEATABLE-ORIGIN: wafer.instr.rdma

//--- positive-loop-result.mlir
func.func @positive_loop_result_uses_backedge_ddr_root() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %init = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %next = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %looped = scf.for %i = %c0 to %c2 step %c1
      iter_args(%iter = %init)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    scf.yield %next : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  %dynamic = memref.cast %looped
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, strided<[1], offset: ?>,
               #wafer.memory<ddr, tensor>>
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %dynamic to %spm
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, strided<[1], offset: ?>,
               #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// POSITIVE-LOOP-RESULT-LABEL: func.func @positive_loop_result_uses_backedge_ddr_root
// POSITIVE-LOOP-RESULT: %[[NEXT:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// POSITIVE-LOOP-RESULT: %[[LOOPED:.+]] = scf.for
// POSITIVE-LOOP-RESULT: memref.cast %[[LOOPED]]
// POSITIVE-LOOP-RESULT: wafer.instr.rdma

//--- memory-space-cast.mlir
func.func @memory_space_cast_keeps_managed_root_live() {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f16
  %a = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %erased = memref.memory_space_cast %a
      : memref<128xf16, #wafer.memory<ddr, tensor>> to memref<128xf16>
  %b = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  memref.store %one, %b[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %late = memref.load %erased[%c0] : memref<128xf16>
  return
}

// SPACE-CAST-CAPACITY: memory_capacity_overflow
// SPACE-CAST-LABEL: func.func @memory_space_cast_keeps_managed_root_live
// SPACE-CAST: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// SPACE-CAST: memref.memory_space_cast
// SPACE-CAST: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// SPACE-CAST: memref.load
