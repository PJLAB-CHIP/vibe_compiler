// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s --check-prefix=IR

module {
  func.func @compare_pointwise(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>)
      -> tensor<4xi1> {
    %0 = "stablehlo.compare"(%lhs, %rhs) {
      comparison_direction = #stablehlo<comparison_direction GT>
    } : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    return %0 : tensor<4xi1>
  }
}

// IR-LABEL: func.func @compare_pointwise
// IR-NOT: stablehlo.
// IR: linalg.generic
// IR: arith.cmpf ogt
// IR-NOT: wafer.comm
