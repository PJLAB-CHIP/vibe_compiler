// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=first-legal tile-search-effort=quick max-search-candidates=1})' %s | FileCheck %s

func.func @normal_transpose(%lhs: tensor<2x3xf16>,
                            %rhs: tensor<4x3xf16>) -> tensor<2x4xf16> {
  %empty = tensor.empty() : tensor<2x4xf16>
  %zero = arith.constant 0.0 : f16
  %init = linalg.fill ins(%zero : f16)
      outs(%empty : tensor<2x4xf16>) -> tensor<2x4xf16>
  %result = linalg.matmul_transpose_b
      ins(%lhs, %rhs : tensor<2x3xf16>, tensor<4x3xf16>)
      outs(%init : tensor<2x4xf16>) -> tensor<2x4xf16>
  return %result : tensor<2x4xf16>
}

func.func @transpose_normal(%lhs: tensor<3x2xf16>,
                            %rhs: tensor<3x4xf16>) -> tensor<2x4xf16> {
  %empty = tensor.empty() : tensor<2x4xf16>
  %zero = arith.constant 0.0 : f16
  %init = linalg.fill ins(%zero : f16)
      outs(%empty : tensor<2x4xf16>) -> tensor<2x4xf16>
  %result = linalg.matmul_transpose_a
      ins(%lhs, %rhs : tensor<3x2xf16>, tensor<3x4xf16>)
      outs(%init : tensor<2x4xf16>) -> tensor<2x4xf16>
  return %result : tensor<2x4xf16>
}

func.func @transpose_transpose(%lhs: tensor<3x2xf16>,
                               %rhs: tensor<4x3xf16>) -> tensor<2x4xf16> {
  %empty = tensor.empty() : tensor<2x4xf16>
  %zero = arith.constant 0.0 : f16
  %init = linalg.fill ins(%zero : f16)
      outs(%empty : tensor<2x4xf16>) -> tensor<2x4xf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<3x2xf16>, tensor<4x3xf16>)
      outs(%init : tensor<2x4xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16, %acc: f16):
      %product = arith.mulf %lhs_value, %rhs_value : f16
      %sum = arith.addf %acc, %product : f16
      linalg.yield %sum : f16
  } -> tensor<2x4xf16>
  return %result : tensor<2x4xf16>
}

func.func @batch_transpose_normal(%lhs: tensor<2x3x2xf16>,
                                  %rhs: tensor<2x3x4xf16>)
    -> tensor<2x2x4xf16> {
  %empty = tensor.empty() : tensor<2x2x4xf16>
  %zero = arith.constant 0.0 : f16
  %init = linalg.fill ins(%zero : f16)
      outs(%empty : tensor<2x2x4xf16>) -> tensor<2x2x4xf16>
  %result = linalg.batch_matmul_transpose_a
      ins(%lhs, %rhs : tensor<2x3x2xf16>, tensor<2x3x4xf16>)
      outs(%init : tensor<2x2x4xf16>) -> tensor<2x2x4xf16>
  return %result : tensor<2x2x4xf16>
}

func.func @batch_normal_transpose(%lhs: tensor<2x2x3xf16>,
                                  %rhs: tensor<2x4x3xf16>)
    -> tensor<2x2x4xf16> {
  %empty = tensor.empty() : tensor<2x2x4xf16>
  %zero = arith.constant 0.0 : f16
  %init = linalg.fill ins(%zero : f16)
      outs(%empty : tensor<2x2x4xf16>) -> tensor<2x2x4xf16>
  %result = linalg.batch_matmul_transpose_b
      ins(%lhs, %rhs : tensor<2x2x3xf16>, tensor<2x4x3xf16>)
      outs(%init : tensor<2x2x4xf16>) -> tensor<2x2x4xf16>
  return %result : tensor<2x2x4xf16>
}

// CHECK-LABEL: func.func @normal_transpose
// CHECK: wafer.instr.gemm
// CHECK-SAME: lhs_orientation = #wafer.gemm_orientation<normal>
// CHECK-SAME: rhs_orientation = #wafer.gemm_orientation<transpose>
// CHECK-LABEL: func.func @transpose_normal
// CHECK: wafer.instr.gemm
// CHECK-SAME: lhs_orientation = #wafer.gemm_orientation<transpose>
// CHECK-SAME: rhs_orientation = #wafer.gemm_orientation<normal>
// CHECK-LABEL: func.func @transpose_transpose
// CHECK: wafer.instr.gemm
// CHECK-SAME: lhs_orientation = #wafer.gemm_orientation<transpose>
// CHECK-SAME: rhs_orientation = #wafer.gemm_orientation<transpose>
// CHECK-LABEL: func.func @batch_transpose_normal
// CHECK: wafer.instr.gemm
// CHECK-SAME: batch_count = 2
// CHECK-SAME: lhs_orientation = #wafer.gemm_orientation<transpose>
// CHECK-SAME: rhs_orientation = #wafer.gemm_orientation<normal>
// CHECK-LABEL: func.func @batch_normal_transpose
// CHECK: wafer.instr.gemm
// CHECK-SAME: batch_count = 2
// CHECK-SAME: lhs_orientation = #wafer.gemm_orientation<normal>
// CHECK-SAME: rhs_orientation = #wafer.gemm_orientation<transpose>
