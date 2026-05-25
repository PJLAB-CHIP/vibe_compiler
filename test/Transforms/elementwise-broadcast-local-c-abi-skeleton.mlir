// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-to-c-abi-skeleton %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @row_broadcast_elementwise(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<4xf16>,
      %out: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %0 = linalg.elementwise kind=#linalg.elementwise_kind<add>
        indexing_maps = [#map, #row, #map]
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<4xf16>)
        outs(%out : tensor<4x8xf16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
}

// CHECK-LABEL: func.func @row_broadcast_elementwise(
// CHECK-NOT: linalg.elementwise
// CHECK-NOT: wafer.compute.elementwise
// CHECK: wafer.abi.rdma_1d <issue_only> %{{.*}} {bytes = 64 : i64}
// CHECK: wafer.abi.rdma_1d <issue_only> %{{.*}} {bytes = 8 : i64}
// CHECK: wafer.abi.elementwise <issue_only> <add>
// CHECK-SAME: indexing_maps
// CHECK: wafer.abi.wdma_1d <issue_only> %{{.*}} {bytes = 64 : i64}
