// RUN: wafer-opt --wafer-form-groups --wafer-materialize-single-tile --wafer-check-spm-allocation %s | FileCheck %s

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
// CHECK-NOT: spm_offset
// CHECK-NOT: spm_allocation
// CHECK: wafer.tile_region
// CHECK: wafer.load_tile
// CHECK: wafer.compute.gemm
// CHECK: wafer.store_tile
// CHECK-NOT: spm_offset
// CHECK-NOT: spm_allocation
