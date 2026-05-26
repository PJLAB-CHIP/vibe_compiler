// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-to-c-abi-skeleton %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @same_shape_elementwise(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<4x8xf16>,
      %out: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %0 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<4x8xf16>)
        outs(%out : tensor<4x8xf16>) {
      ^bb0(%lhs_s: f16, %rhs_s: f16, %out_s: f16):
        %sum = arith.addf %lhs_s, %rhs_s : f16
        linalg.yield %sum : f16
    } -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
}

// CHECK-LABEL: func.func @same_shape_elementwise(
// CHECK-NOT: linalg.generic
// CHECK-NOT: wafer.compute.elementwise
// CHECK: wafer.abi.rdma <issue_only> %{{.*}} {bytes = 64 : i64}
// CHECK: wafer.abi.rdma <issue_only> %{{.*}} {bytes = 64 : i64}
// CHECK: wafer.abi.elementwise <issue_only> <add>
// CHECK: wafer.abi.wdma <issue_only> %{{.*}} {bytes = 64 : i64}
