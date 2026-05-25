// RUN: wafer-opt --wafer-form-groups %s | FileCheck %s

module {
  func.func @row_sum_reduce(
      %input: tensor<4x8xf16>) -> tensor<4xf16> {
    %zero = arith.constant 0.000000e+00 : f16
    %empty = tensor.empty() : tensor<4xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<4xf16>) -> tensor<4xf16>
    %0 = linalg.reduce { arith.addf }
        ins(%input : tensor<4x8xf16>)
        outs(%init : tensor<4xf16>)
        dimensions = [1]
    return %0 : tensor<4xf16>
  }
}

// CHECK-LABEL: func.func @row_sum_reduce(
// CHECK: %[[GROUP:.+]] = wafer.group
// CHECK: linalg.reduce
// CHECK-SAME: arith.addf
// CHECK: wafer.group_yield
// CHECK: return %[[GROUP]] : tensor<4xf16>
