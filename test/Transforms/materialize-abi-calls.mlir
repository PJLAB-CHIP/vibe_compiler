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
// CHECK-DAG: func.func private @wafer_rdma(i64, i32, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-DAG: func.func private @wafer_gemm
// CHECK-DAG: func.func private @wafer_local_fence
// CHECK-DAG: func.func private @wafer_wdma(i32, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
