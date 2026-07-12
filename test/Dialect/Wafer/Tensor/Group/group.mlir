// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %dest = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.group ins(%source : tensor<4xf32>)
                    outs(%dest : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>, %out: tensor<4xf32>):
    wafer.group.yield %in : tensor<4xf32>
  } : tensor<4xf32>
}

func.func @zero_inputs(%dest: tensor<4xf32>) -> tensor<4xf32> {
  %0 = wafer.group ins() outs(%dest : tensor<4xf32>) {
  ^bb0(%out: tensor<4xf32>):
    wafer.group.yield %out : tensor<4xf32>
  } : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: %[[SOURCE:.+]] = unrealized_conversion_cast to tensor<4xf32>
// CHECK: %[[DEST:.+]] = unrealized_conversion_cast to tensor<4xf32>
// CHECK: wafer.group ins(%[[SOURCE]] : tensor<4xf32>) outs(%[[DEST]] : tensor<4xf32>) {
// CHECK: ^bb0(%[[IN:.+]]: tensor<4xf32>, %{{.+}}: tensor<4xf32>):
// CHECK:   wafer.group.yield %[[IN]] : tensor<4xf32>
// CHECK: } : tensor<4xf32>
// CHECK-LABEL: func.func @zero_inputs
// CHECK: wafer.group ins() outs(%{{.+}} : tensor<4xf32>) {
