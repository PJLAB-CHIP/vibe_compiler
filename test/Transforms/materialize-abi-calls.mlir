// RUN: wafer-opt --wafer-materialize-abi-calls %s | FileCheck %s

func.func @abi_rdm_wdm_gemm(
    %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  %input_tile = memref.subview %input[1, 2] [2, 3] [1, 1]
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  %output_tile = memref.subview %output[1, 2] [2, 3] [1, 1]
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  %region = wafer.tile.region(%input_tile, %output_tile
      : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>):
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, cx>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<3x3xf16, #wafer.memory<spm, cx>>
    %acc = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<2x3xf16, #wafer.memory<spm, cx>>
    wafer.instr.rdma %in to %lhs
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 16, 0, 0>}
        : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, cx>>
    wafer.instr.gemm %lhs, %rhs into %acc
        {m = 2 : i64, k = 3 : i64, n = 3 : i64}
        : memref<2x3xf16, #wafer.memory<spm, cx>>,
          memref<3x3xf16, #wafer.memory<spm, cx>>
       into memref<2x3xf16, #wafer.memory<spm, cx>>
    wafer.instr.local_fence
    wafer.instr.wdma %acc to %out
        {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
         dst_strides = array<i64: 16, 0, 0>, inner_bytes = 6 : i64}
        : memref<2x3xf16, #wafer.memory<spm, cx>>
       to memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @abi_rdm_wdm_gemm_abi
// CHECK-NOT: wafer.instr
// CHECK-NOT: wafer.tile.region
// CHECK: call @wafer_rdma
// CHECK: call @wafer_gemm
// CHECK: call @wafer_local_fence
// CHECK: call @wafer_wdma
// CHECK: return

// -----

func.func @abi_ct_dte(%input: memref<8xf32, #wafer.memory<ddr, tensor>>)
    -> memref<8xf32, #wafer.memory<ddr, tensor>> {
  %output = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>}
      : memref<8xf32, #wafer.memory<ddr, tensor>>
  %region = wafer.tile.region(%input, %output
      : memref<8xf32, #wafer.memory<ddr, tensor>>,
        memref<8xf32, #wafer.memory<ddr, tensor>>)
      -> (memref<8xf32, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<8xf32, #wafer.memory<ddr, tensor>>,
       %out: memref<8xf32, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.0 : f32
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<8xf32, #wafer.memory<spm, tensor>>
    %fill = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65664>}
        : memref<8xf32, #wafer.memory<spm, tensor>>
    %acc = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<8xf32, #wafer.memory<spm, tensor>>
    %red_src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65920>}
        : memref<1x8xf32, #wafer.memory<spm, cx>>
    %reduced = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<1xf32, #wafer.memory<spm, cx>>
    %converted = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66176>}
        : memref<8xi32, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %in to %src
        {byte_count = 32 : i64, inner_bytes = 32 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<8xf32, #wafer.memory<ddr, tensor>>
       to memref<8xf32, #wafer.memory<spm, tensor>>
    wafer.instr.fill %fill, %zero
        : memref<8xf32, #wafer.memory<spm, tensor>>, f32
    wafer.instr.elementwise #wafer.elementwise_kind<add> %src, %fill into %acc
        : memref<8xf32, #wafer.memory<spm, tensor>>,
          memref<8xf32, #wafer.memory<spm, tensor>>
       into memref<8xf32, #wafer.memory<spm, tensor>>
    wafer.instr.reduce #wafer.reduce_kind<sum> %red_src into %reduced, %zero : f32
        {dimensions = array<i64: 1>}
        : memref<1x8xf32, #wafer.memory<spm, cx>>
       into memref<1xf32, #wafer.memory<spm, cx>>
    wafer.instr.convert %acc into %converted
        {src_dtype = f32, dst_dtype = i32}
        : memref<8xf32, #wafer.memory<spm, tensor>>
       to memref<8xi32, #wafer.memory<spm, tensor>>
    %send = wafer.instr.dte_send %acc {peer = 1 : i64, bytes = 32 : i64}
        : memref<8xf32, #wafer.memory<spm, tensor>> -> !async.token
    %recv = wafer.instr.dte_recv %fill {peer = 1 : i64, bytes = 32 : i64}
        : memref<8xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %send, %recv : !async.token, !async.token
    wafer.instr.wdma %acc to %out
        {byte_count = 32 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 32 : i64}
        : memref<8xf32, #wafer.memory<spm, tensor>>
       to memref<8xf32, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<8xf32, #wafer.memory<ddr, tensor>>
  }
  return %region : memref<8xf32, #wafer.memory<ddr, tensor>>
}

// CHECK-LABEL: func.func @abi_ct_dte_abi
// CHECK-SAME: (%arg0: i64, %arg1: i64)
// CHECK-NOT: wafer.instr
// CHECK-NOT: wafer.tile.region
// CHECK: call @wafer_rdma
// CHECK: call @wafer_fill
// CHECK: call @wafer_elementwise
// CHECK: call @wafer_reduce
// CHECK: call @wafer_convert
// CHECK: call @wafer_dte_send
// CHECK: call @wafer_dte_recv
// CHECK: call @wafer_dte_wait
// CHECK: call @wafer_wdma
// CHECK: return

// -----

func.func @abi_workspace(
    %input: memref<8xf32, #wafer.memory<ddr, tensor>>,
    %output: memref<8xf32, #wafer.memory<ddr, tensor>>) {
  %workspace = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<256>}
      : memref<8xf32, #wafer.memory<ddr, tensor>>
  %region = wafer.tile.region(%input, %workspace, %output
      : memref<8xf32, #wafer.memory<ddr, tensor>>,
        memref<8xf32, #wafer.memory<ddr, tensor>>,
        memref<8xf32, #wafer.memory<ddr, tensor>>)
      -> (memref<8xf32, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<8xf32, #wafer.memory<ddr, tensor>>,
       %tmp: memref<8xf32, #wafer.memory<ddr, tensor>>,
       %out: memref<8xf32, #wafer.memory<ddr, tensor>>):
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<8xf32, #wafer.memory<spm, tensor>>
    %copy = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65664>}
        : memref<8xf32, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %in to %src
        {byte_count = 32 : i64, inner_bytes = 32 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<8xf32, #wafer.memory<ddr, tensor>>
       to memref<8xf32, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %src to %tmp
        {byte_count = 32 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 32 : i64}
        : memref<8xf32, #wafer.memory<spm, tensor>>
       to memref<8xf32, #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %tmp to %copy
        {byte_count = 32 : i64, inner_bytes = 32 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<8xf32, #wafer.memory<ddr, tensor>>
       to memref<8xf32, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %copy to %out
        {byte_count = 32 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 32 : i64}
        : memref<8xf32, #wafer.memory<spm, tensor>>
       to memref<8xf32, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<8xf32, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @abi_workspace_abi
// CHECK-SAME: (%arg0: i64, %arg1: i64, %arg2: i64)
// CHECK: %[[C256:.+]] = arith.constant 256 : i64
// CHECK: %[[WORKSPACE:.+]] = arith.addi %arg2, %[[C256]]
// CHECK: call @wafer_rdma(%arg0
// CHECK: call @wafer_wdma({{.*}}, %[[WORKSPACE]]
// CHECK: call @wafer_rdma(%[[WORKSPACE]]
// CHECK: call @wafer_wdma({{.*}}, %arg1
// CHECK: return
// CHECK-DAG: func.func private @wafer_rdma(i64, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32
// CHECK-DAG: func.func private @wafer_gemm(i32, i32, i32, i64, i64, i64, i32) -> i32
// CHECK-DAG: func.func private @wafer_local_fence
// CHECK-DAG: func.func private @wafer_wdma(i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32
// CHECK-DAG: func.func private @wafer_fill(i32, i64, i64, i32) -> i32
// CHECK-DAG: func.func private @wafer_elementwise(i32, i32, i32, i32, i64, i32, i32) -> i32
// CHECK-DAG: func.func private @wafer_reduce(i32, i32, i32, i32, i64, i64, i64, i64, i32) -> i32
// CHECK-DAG: func.func private @wafer_convert(i32, i32, i32, i32, i64) -> i32
// CHECK-DAG: func.func private @wafer_dte_send(i32, i32, i64) -> i32
// CHECK-DAG: func.func private @wafer_dte_recv(i32, i32, i64) -> i32
// CHECK-DAG: func.func private @wafer_dte_wait(i32) -> i32
