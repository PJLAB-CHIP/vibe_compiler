// RUN: wafer-opt --wafer-form-groups --wafer-materialize-single-tile %s | FileCheck %s

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
// CHECK-NOT: wafer.group
// CHECK: %[[REGION:.+]] = wafer.tile_region(%[[LHS]], %[[RHS]], %[[OUT]] : tensor<4x8xf16>, tensor<8x16xf16>, tensor<4x16xf16>) -> (tensor<4x16xf16>) {
// CHECK: ^bb0(%[[TLHS:[^:]+]]: tensor<4x8xf16>, %[[TRHS:[^:]+]]: tensor<8x16xf16>, %[[TOUT:[^:]+]]: tensor<4x16xf16>):
// CHECK: %[[LHS_T:.+]] = wafer.load_tile %[[TLHS]] : tensor<4x8xf16> -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
// CHECK: %[[RHS_T:.+]] = wafer.load_tile %[[TRHS]] : tensor<8x16xf16> -> !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
// CHECK: %[[LHS_CX:.+]] = wafer.layout.materialize %[[LHS_T]] : !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
// CHECK: %[[RHS_CX:.+]] = wafer.layout.materialize %[[RHS_T]] : !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
// CHECK: %[[MM:.+]] = wafer.compute.gemm %[[LHS_CX]], %[[RHS_CX]]
// CHECK: %[[MM_T:.+]] = wafer.layout.materialize %[[MM]]
// CHECK: wafer.store_tile %[[MM_T]], %[[TOUT]]
// CHECK: wafer.tile_yield %[[TOUT]] : tensor<4x16xf16>
// CHECK: return %[[REGION]] : tensor<4x16xf16>
