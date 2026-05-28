// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-tile-region-to-c-abi %s | FileCheck %s

module {
  func.func @single_matmul(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%out : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// CHECK-LABEL: func.func @single_matmul(
// CHECK-NOT: wafer.load_tile
// CHECK: wafer.abi.rdma <issue_only> %{{.*}} {bytes = 64 : i64}
// CHECK: wafer.abi.rdma <issue_only> %{{.*}} {bytes = 256 : i64}
// CHECK-NOT: wafer.compute.gemm
// CHECK: wafer.abi.gemm <issue_only>
// CHECK-SAME: k = 8 : i64
// CHECK-SAME: m = 4 : i64
// CHECK-SAME: n = 16 : i64
// CHECK-NOT: wafer.store_tile
// CHECK: wafer.abi.wdma <issue_only> %{{.*}} {bytes = 128 : i64}
