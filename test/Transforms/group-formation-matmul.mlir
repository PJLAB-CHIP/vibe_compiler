// RUN: wafer-opt --wafer-form-groups %s | FileCheck %s

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

  func.func @multi_output_generic_stays_ungrouped(
      %x: tensor<4xf32>,
      %init0: tensor<4xf32>,
      %init1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0:2 = linalg.generic {
        indexing_maps = [
          affine_map<(i) -> (i)>,
          affine_map<(i) -> (i)>,
          affine_map<(i) -> (i)>
        ],
        iterator_types = ["parallel"]
      }
      ins(%x : tensor<4xf32>)
      outs(%init0, %init1 : tensor<4xf32>, tensor<4xf32>) {
    ^bb0(%value: f32, %out0: f32, %out1: f32):
      linalg.yield %value, %value : f32, f32
    } -> (tensor<4xf32>, tensor<4xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @single_matmul(
// CHECK-SAME: %[[LHS:[^:]+]]: tensor<4x8xf16>
// CHECK-SAME: %[[RHS:[^:]+]]: tensor<8x16xf16>
// CHECK-SAME: %[[OUT:[^:]+]]: tensor<4x16xf16>
// CHECK: %[[GROUP:.+]] = wafer.group ins(%[[LHS]], %[[RHS]] : tensor<4x8xf16>, tensor<8x16xf16>) outs(%[[OUT]] : tensor<4x16xf16>) {
// CHECK: ^bb0(%[[GLHS:[^:]+]]: tensor<4x8xf16>, %[[GRHS:[^:]+]]: tensor<8x16xf16>, %[[GOUT:[^:]+]]: tensor<4x16xf16>):
// CHECK:   %[[MM:.+]] = linalg.matmul ins(%[[GLHS]], %[[GRHS]] : tensor<4x8xf16>, tensor<8x16xf16>) outs(%[[GOUT]] : tensor<4x16xf16>) -> tensor<4x16xf16>
// CHECK:   wafer.group_yield %[[MM]] : tensor<4x16xf16>
// CHECK: } : tensor<4x16xf16>
// CHECK: return %[[GROUP]] : tensor<4x16xf16>

// CHECK-LABEL: func.func @multi_output_generic_stays_ungrouped(
// CHECK-NOT: wafer.group
// CHECK: linalg.generic
// CHECK: return
