// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-tile-region-to-c-abi %s | FileCheck %s

module {
  func.func @row_sum_reduce(
      %input: tensor<4x8xf16>) -> tensor<4xf16> {
    %zero = arith.constant 0.000000e+00 : f16
    %empty = tensor.empty() : tensor<4xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<4xf16>) -> tensor<4xf16>
    %0 = linalg.reduce { arith.addf }
        ins(%input : tensor<4x8xf16>)
        outs(%init : tensor<4xf16>)
        dimensions = [1]
    return %0 : tensor<4xf16>
  }
}

// CHECK-LABEL: func.func @row_sum_reduce(
// CHECK-NOT: linalg.reduce
// CHECK-NOT: wafer.compute.reduce
// CHECK: wafer.abi.rdma <issue_only> %{{.*}} {bytes = 64 : i64}
// CHECK: wafer.abi.reduce <issue_only> <sum>
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK-SAME: init_value = 0.000000e+00 : f16
// CHECK: wafer.abi.wdma <issue_only> %{{.*}} {bytes = 8 : i64}
