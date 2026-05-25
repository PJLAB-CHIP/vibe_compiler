// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-normalize-constants %s | FileCheck %s

module {
  func.func @constant_tensor() -> tensor<2x2xf32> {
    %0 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @constant_tensor
// CHECK-NOT: stablehlo.constant
// CHECK: %[[CST:.+]] = arith.constant dense<{{.*}}> : tensor<2x2xf32>
// CHECK: return %[[CST]] : tensor<2x2xf32>
