// RUN: wafer-opt --wafer-form-groups --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings %s | FileCheck %s

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
// CHECK-SAME: %[[LHS:[^:]+]]: tensor<4x8xf16>
// CHECK-SAME: %[[RHS:[^:]+]]: tensor<8x16xf16>
// CHECK-SAME: %[[OUT:[^:]+]]: tensor<4x16xf16>
// CHECK-DAG: wafer.ddr.external_binding <input> %[[LHS]] {{.*}}bytes = 64 : i64{{.*}}read_only = true
// CHECK-DAG: wafer.ddr.external_binding <input> %[[RHS]] {{.*}}bytes = 256 : i64{{.*}}read_only = true
// CHECK-DAG: wafer.ddr.external_binding <output> %[[OUT]] {{.*}}bytes = 128 : i64{{.*}}read_only = false
// CHECK: wafer.tile_region
