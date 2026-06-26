// RUN: wafer-opt --wafer-form-logical-groups %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

module {
  func.func @nested_region_captures_become_group_inputs(
      %lhs: tensor<2xf32>,
      %rhs: tensor<2xf32>) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %neg_empty = tensor.empty() : tensor<2xf32>
    %neg = linalg.generic
        {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
        ins(%lhs : tensor<2xf32>)
        outs(%neg_empty : tensor<2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg_value = arith.negf %in : f32
      linalg.yield %neg_value : f32
    } -> tensor<2xf32>

    %cat_empty = tensor.empty() : tensor<4xf32>
    %cat = linalg.generic
        {indexing_maps = [#map], iterator_types = ["parallel"]}
        outs(%cat_empty : tensor<4xf32>) {
    ^bb0(%out: f32):
      %i = linalg.index 0 : index
      %dim = tensor.dim %neg, %c0 : tensor<2xf32>
      %take_lhs = arith.cmpi ult, %i, %dim : index
      %value = scf.if %take_lhs -> (f32) {
        %lhs_value = tensor.extract %neg[%i] : tensor<2xf32>
        scf.yield %lhs_value : f32
      } else {
        %rhs_i = arith.subi %i, %dim : index
        %rhs_value = tensor.extract %rhs[%rhs_i] : tensor<2xf32>
        scf.yield %rhs_value : f32
      }
      linalg.yield %value : f32
    } -> tensor<4xf32>

    return %cat : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @nested_region_captures_become_group_inputs
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[NEG:.+]] = wafer.group ins(%arg0 : tensor<2xf32>) outs(%{{.+}} : tensor<2xf32>)
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<4xf32>
// CHECK: %[[GROUP:.+]] = wafer.group
// CHECK-SAME: ins(%[[NEG]], %[[C0]], %arg1 : tensor<2xf32>, index, tensor<2xf32>)
// CHECK-SAME: outs(%[[OUT]] : tensor<4xf32>)
// CHECK: ^bb0(%[[GROUP_NEG:.+]]: tensor<2xf32>, %[[GROUP_C0:.+]]: index, %[[GROUP_RHS:.+]]: tensor<2xf32>, %{{.+}}: tensor<4xf32>):
// CHECK: tensor.dim %[[GROUP_NEG]], %[[GROUP_C0]]
// CHECK: tensor.extract %[[GROUP_NEG]]
// CHECK: tensor.extract %[[GROUP_RHS]]
// CHECK: wafer.group.yield
// CHECK: return %[[GROUP]] : tensor<4xf32>
