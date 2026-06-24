// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-llvm)' %s | FileCheck %s

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
    %mm = linalg.matmul
        ins(%lhs_tile, %rhs_tile : tensor<2x4xf16>, tensor<4x3xf16>)
        outs(%empty : tensor<2x3xf16>) -> tensor<2x3xf16>
    %updated = tensor.insert_slice %mm into %arg2[1, 2] [2, 3] [1, 1]
        : tensor<2x3xf16> into tensor<4x8xf16>
    wafer.group.yield %updated : tensor<4x8xf16>
  } : tensor<4x8xf16>
  return %group : tensor<4x8xf16>
}

// CHECK-LABEL: llvm.func @boundary_tiled_matmul_group_abi
// CHECK-NOT: wafer.
// CHECK-NOT: func.func
// CHECK: llvm.call @wafer_rdma
// CHECK: llvm.call @wafer_gather_scatter
// CHECK: llvm.call @wafer_gemm
// CHECK: llvm.call @wafer_wdma
