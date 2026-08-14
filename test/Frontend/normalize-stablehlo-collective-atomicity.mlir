// REQUIRES: stablehlo
// RUN: not wafer-opt --mlir-disable-threading --wafer-normalize-stablehlo-collectives \
// RUN:   --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=wafer.linalg_ext --implicit-check-not=tensor.empty

module {
  func.func @unsupported_second_collective(
      %input: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_gather"(%input) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0]]> : tensor<1x1xi64>
    } : (tensor<4xf32>) -> tensor<4xf32>
    %1 = "stablehlo.all_reduce"(%0) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %product = stablehlo.multiply %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%product) : (tensor<f32>) -> ()
    }) {
      replica_groups = dense<[[0]]> : tensor<1x1xi64>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// CHECK: failed to normalize residual StableHLO op before Wafer structured tensor-program scheduling
// CHECK: IR Dump After NormalizeStablehloCollectivesPass Failed
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.all_reduce
// CHECK: stablehlo.multiply
