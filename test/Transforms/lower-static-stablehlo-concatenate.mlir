// REQUIRES: stablehlo
// RUN: wafer-opt %s --wafer-lower-stablehlo-to-linalg | FileCheck %s

module {
  func.func @concatenate(%prefix: tensor<2x3xf16>, %suffix: tensor<1x3xf16>)
      -> tensor<3x3xf16> {
    %result = stablehlo.concatenate %prefix, %suffix, dim = 0
        : (tensor<2x3xf16>, tensor<1x3xf16>) -> tensor<3x3xf16>
    return %result : tensor<3x3xf16>
  }
}

// CHECK-LABEL: func.func @concatenate
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<3x3xf16>
// CHECK: %[[PREFIX:.*]] = tensor.insert_slice %arg0 into %[[EMPTY]][0, 0] [2, 3] [1, 1]
// CHECK: %[[RESULT:.*]] = tensor.insert_slice %arg1 into %[[PREFIX]][2, 0] [1, 3] [1, 1]
// CHECK: return %[[RESULT]] : tensor<3x3xf16>
// CHECK-NOT: stablehlo.concatenate
// CHECK-NOT: linalg.index
