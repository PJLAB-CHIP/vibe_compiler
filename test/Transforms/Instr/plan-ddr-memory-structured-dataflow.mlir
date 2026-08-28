// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=512 ddr-largest-contiguous-bytes=512' %s | FileCheck %s

func.func private @forward_ddr_alias(
    %arg: memref<128xf16, #wafer.memory<ddr, tensor>>)
    -> memref<128xf16, #wafer.memory<ddr, tensor>> {
  return %arg : memref<128xf16, #wafer.memory<ddr, tensor>>
}

func.func @direct_alias_helper_preserves_caller_ownership() {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f16
  %a = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %alias = func.call @forward_ddr_alias(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>)
     -> memref<128xf16, #wafer.memory<ddr, tensor>>
  %b = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  memref.store %one, %b[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %late = memref.load %alias[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

// CHECK-LABEL: func.func @direct_alias_helper_preserves_caller_ownership
// CHECK: %[[A:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: %[[ALIAS:.+]] = call @forward_ddr_alias(%[[A]])
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// CHECK: memref.load %[[ALIAS]]

func.func private @forward_ddr_as_tensor(
    %arg: memref<128xf16, #wafer.memory<ddr, tensor>>) -> tensor<128xf16> {
  %tensor = bufferization.to_tensor %arg restrict writable
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return %tensor : tensor<128xf16>
}

func.func @type_erased_alias_helper_preserves_caller_ownership() {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f16
  %a = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %tensor = func.call @forward_ddr_as_tensor(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> tensor<128xf16>
  %alias = bufferization.to_memref %tensor : memref<128xf16>
  %b = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  memref.store %one, %b[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %late = memref.load %alias[%c0] : memref<128xf16>
  return
}

// CHECK-LABEL: func.func @type_erased_alias_helper_preserves_caller_ownership
// CHECK: %[[ERASED_A:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: %[[TENSOR:.+]] = call @forward_ddr_as_tensor(%[[ERASED_A]])
// CHECK: %[[ERASED_ALIAS:.+]] = bufferization.to_memref %[[TENSOR]]
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// CHECK: memref.load %[[ERASED_ALIAS]]

func.func @pre_loop_issue_completed_after_loop(
    %lb: index, %ub: index, %step: index) {
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
  scf.for %i = %lb to %ub step %step {
  }
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @pre_loop_issue_completed_after_loop
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: wafer.instr.rdma
// CHECK: scf.for
// CHECK: wafer.instr.ncc_join [0]

func.func @unrelated_loop_carried_async_token(
    %token: !async.token, %lb: index, %ub: index, %step: index) {
  %looped = scf.for %i = %lb to %ub step %step
      iter_args(%iter = %token) -> (!async.token) {
    scf.yield %iter : !async.token
  }
  async.await %looped : !async.token
  return
}

// CHECK-LABEL: func.func @unrelated_loop_carried_async_token
// CHECK: scf.for
// CHECK: async.await

func.func @if_result_preserves_managed_descriptor_ownership(%cond: i1) {
  %selected = scf.if %cond -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    %then_ddr = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    scf.yield %then_ddr : memref<128xf16, #wafer.memory<ddr, tensor>>
  } else {
    %else_ddr = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    scf.yield %else_ddr : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %selected to %spm
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// CHECK-LABEL: func.func @if_result_preserves_managed_descriptor_ownership
// CHECK: scf.if
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: wafer.instr.rdma

func.func @for_result_preserves_managed_descriptor_ownership(
    %lb: index, %ub: index, %step: index) {
  %init = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %next = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %looped = scf.for %i = %lb to %ub step %step
      iter_args(%iter = %init)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    scf.yield %next : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
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

// CHECK-LABEL: func.func @for_result_preserves_managed_descriptor_ownership
// CHECK: %[[INIT:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CHECK: %[[NEXT:.+]] = memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// CHECK: scf.for
// CHECK: wafer.instr.rdma %{{.*}}
