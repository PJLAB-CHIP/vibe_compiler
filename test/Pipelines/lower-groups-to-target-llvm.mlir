// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-target-llvm)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-target-llvm)' %s | mlir-translate --mlir-to-llvmir | FileCheck --check-prefix=LLVMIR %s

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

// CHECK-LABEL: llvm.func @boundary_tiled_matmul_group
// CHECK-SAME: (%{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64)
// CHECK-NOT: wafer.
// CHECK-NOT: memref.
// CHECK-NOT: func.func
// CHECK: llvm.call @wafer_tx81_rdma
// CHECK: llvm.call @wafer_tx81_gemm
// CHECK: llvm.call @wafer_tx81_wdma
// CHECK: llvm.return

// LLVMIR-LABEL: define void @boundary_tiled_matmul_group(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// LLVMIR: call void (...) @wafer_tx81_rdma
// LLVMIR: call void (...) @wafer_tx81_gemm
// LLVMIR: call void (...) @wafer_tx81_wdma
