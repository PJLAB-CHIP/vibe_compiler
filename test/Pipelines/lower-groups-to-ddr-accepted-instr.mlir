// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-ddr-accepted-instr)' %s | FileCheck %s

func.func @boundary_tiled_matmul_group(%lhs: tensor<4x8xf16>,
                                       %rhs: tensor<8x8xf16>,
                                       %out: tensor<4x8xf16>)
    -> tensor<4x8xf16> {
  %group = wafer.group ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x8xf16>)
      outs(%out : tensor<4x8xf16>) {
  ^bb0(%arg0: tensor<4x8xf16>, %arg1: tensor<8x8xf16>,
       %arg2: tensor<4x8xf16>):
    %lhs_tile = tensor.extract_slice %arg0[1, 0] [2, 4] [1, 1]
        : tensor<4x8xf16> to tensor<2x4xf16>
    %rhs_tile = tensor.extract_slice %arg1[0, 2] [4, 3] [1, 1]
        : tensor<8x8xf16> to tensor<4x3xf16>
    %empty = tensor.empty() : tensor<2x3xf16>
    %c0 = arith.constant 0.000000e+00 : f16
    %init = linalg.fill ins(%c0 : f16)
        outs(%empty : tensor<2x3xf16>) -> tensor<2x3xf16>
    %mm = linalg.matmul
        ins(%lhs_tile, %rhs_tile : tensor<2x4xf16>, tensor<4x3xf16>)
        outs(%init : tensor<2x3xf16>) -> tensor<2x3xf16>
    %updated = tensor.insert_slice %mm into %arg2[1, 2] [2, 3] [1, 1]
        : tensor<2x3xf16> into tensor<4x8xf16>
    wafer.group.yield %updated : tensor<4x8xf16>
  } : tensor<4x8xf16>
  return %group : tensor<4x8xf16>
}

// CHECK-LABEL: func.func @boundary_tiled_matmul_group
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.matmul
// CHECK-NOT: wafer.tile.load
// CHECK-NOT: wafer.tile.store
// CHECK: %[[LHS_TILE:.+]] = memref.subview
// CHECK-SAME: memref<2x4xf16, strided<[8, 1], offset: 8>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.spm.offset
// CHECK: wafer.instr.rdma %[[LHS_TILE]]
// CHECK-SAME: byte_count = 16 : i64
// CHECK-SAME: inner_bytes = 8 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: src_strides = array<i64: 16, 0, 0>
// CHECK: %[[RHS_TILE:.+]] = memref.subview
// CHECK-SAME: memref<4x3xf16, strided<[8, 1], offset: 2>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.instr.rdma %[[RHS_TILE]]
// CHECK-SAME: byte_count = 24 : i64
// CHECK-SAME: inner_bytes = 6 : i64
// CHECK-SAME: src_iterations = array<i64: 4, 1, 1>
// CHECK-SAME: src_strides = array<i64: 16, 0, 0>
// CHECK: wafer.instr.gemm
// CHECK: %[[OUT_TILE:.+]] = memref.subview
// CHECK-SAME: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.instr.wdma {{%.*}} to %[[OUT_TILE]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 16, 0, 0>
// CHECK-SAME: inner_bytes = 6 : i64
