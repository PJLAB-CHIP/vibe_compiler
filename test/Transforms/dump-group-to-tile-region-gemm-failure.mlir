// RUN: wafer-opt --wafer-dump-group-to-tile-region='logical-rank=0' %s 2>&1 | FileCheck %s

func.func @matmul_unknown_init(%lhs: tensor<2x4xf16>,
                               %rhs: tensor<4x3xf16>,
                               %out: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %group = wafer.group
      ins(%lhs, %rhs : tensor<2x4xf16>, tensor<4x3xf16>)
      outs(%out : tensor<2x3xf16>) {
  ^bb0(%lhs_arg: tensor<2x4xf16>, %rhs_arg: tensor<4x3xf16>,
       %out_arg: tensor<2x3xf16>):
    %result = linalg.matmul
        ins(%lhs_arg, %rhs_arg : tensor<2x4xf16>, tensor<4x3xf16>)
        outs(%out_arg : tensor<2x3xf16>) -> tensor<2x3xf16>
    wafer.group.yield %result : tensor<2x3xf16>
  } : tensor<2x3xf16>
  return %group : tensor<2x3xf16>
}

func.func @matmul_nonzero_init(%lhs: tensor<2x4xf16>,
                               %rhs: tensor<4x3xf16>,
                               %out: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %group = wafer.group
      ins(%lhs, %rhs : tensor<2x4xf16>, tensor<4x3xf16>)
      outs(%out : tensor<2x3xf16>) {
  ^bb0(%lhs_arg: tensor<2x4xf16>, %rhs_arg: tensor<4x3xf16>,
       %out_arg: tensor<2x3xf16>):
    %one = arith.constant 1.0 : f16
    %init = linalg.fill ins(%one : f16)
        outs(%out_arg : tensor<2x3xf16>) -> tensor<2x3xf16>
    %result = linalg.matmul
        ins(%lhs_arg, %rhs_arg : tensor<2x4xf16>, tensor<4x3xf16>)
        outs(%init : tensor<2x3xf16>) -> tensor<2x3xf16>
    wafer.group.yield %result : tensor<2x3xf16>
  } : tensor<2x3xf16>
  return %group : tensor<2x3xf16>
}

func.func @batch_matmul_nonzero_init(%lhs: tensor<2x3x4xf32>,
                                     %rhs: tensor<2x4x5xf32>,
                                     %out: tensor<2x3x5xf32>)
    -> tensor<2x3x5xf32> {
  %group = wafer.group
      ins(%lhs, %rhs : tensor<2x3x4xf32>, tensor<2x4x5xf32>)
      outs(%out : tensor<2x3x5xf32>) {
  ^bb0(%lhs_arg: tensor<2x3x4xf32>, %rhs_arg: tensor<2x4x5xf32>,
       %out_arg: tensor<2x3x5xf32>):
    %one = arith.constant 1.0 : f32
    %init = linalg.fill ins(%one : f32)
        outs(%out_arg : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
    %result = linalg.batch_matmul
        ins(%lhs_arg, %rhs_arg : tensor<2x3x4xf32>, tensor<2x4x5xf32>)
        outs(%init : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
    wafer.group.yield %result : tensor<2x3x5xf32>
  } : tensor<2x3x5xf32>
  return %group : tensor<2x3x5xf32>
}

func.func @matmul_negative_zero_init(%lhs: tensor<2x4xf16>,
                                     %rhs: tensor<4x3xf16>,
                                     %out: tensor<2x3xf16>)
    -> tensor<2x3xf16> {
  %group = wafer.group
      ins(%lhs, %rhs : tensor<2x4xf16>, tensor<4x3xf16>)
      outs(%out : tensor<2x3xf16>) {
  ^bb0(%lhs_arg: tensor<2x4xf16>, %rhs_arg: tensor<4x3xf16>,
       %out_arg: tensor<2x3xf16>):
    %negative_zero = arith.constant -0.0 : f16
    %init = linalg.fill ins(%negative_zero : f16)
        outs(%out_arg : tensor<2x3xf16>) -> tensor<2x3xf16>
    %result = linalg.matmul
        ins(%lhs_arg, %rhs_arg : tensor<2x4xf16>, tensor<4x3xf16>)
        outs(%init : tensor<2x3xf16>) -> tensor<2x3xf16>
    wafer.group.yield %result : tensor<2x3xf16>
  } : tensor<2x3xf16>
  return %group : tensor<2x3xf16>
}

// CHECK: wafer.group_to_tile_region group @matmul_unknown_init#0
// CHECK-NEXT: failure matmul requires a provable zero-filled DPS init because wafer.tile.gemm has overwrite semantics
// CHECK: wafer.group_to_tile_region group @matmul_nonzero_init#0
// CHECK-NEXT: failure matmul requires a provable zero-filled DPS init because wafer.tile.gemm has overwrite semantics
// CHECK: wafer.group_to_tile_region group @batch_matmul_nonzero_init#0
// CHECK-NEXT: failure batch matmul requires a provable zero-filled DPS init because wafer.tile.gemm has overwrite semantics
// CHECK: wafer.group_to_tile_region group @matmul_negative_zero_init#0
// CHECK-NEXT: failure matmul requires a provable zero-filled DPS init because wafer.tile.gemm has overwrite semantics
