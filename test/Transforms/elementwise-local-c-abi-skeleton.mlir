// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-to-c-abi-skeleton %s | FileCheck %s

module {
  func.func @same_shape_elementwise(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<4x8xf16>,
      %out: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %0 = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<4x8xf16>)
        outs(%out : tensor<4x8xf16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
}

// CHECK-LABEL: func.func @same_shape_elementwise(
// CHECK-NOT: linalg.elementwise
// CHECK-NOT: wafer.compute.elementwise
// CHECK: wafer.abi.rdma_1d <issue_only> %{{.*}} {bytes = 64 : i64}
// CHECK: wafer.abi.rdma_1d <issue_only> %{{.*}} {bytes = 64 : i64}
// CHECK: wafer.abi.elementwise <issue_only> <add>
// CHECK: wafer.abi.wdma_1d <issue_only> %{{.*}} {bytes = 64 : i64}
