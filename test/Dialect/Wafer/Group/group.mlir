// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.group ins(%source : tensor<4xf32>)
                    outs(%source : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>, %out: tensor<4xf32>):
    wafer.group_yield %in : tensor<4xf32>
  } : tensor<4xf32>
}

// CHECK: %[[SOURCE:.+]] = unrealized_conversion_cast to tensor<4xf32>
// CHECK: wafer.group ins(%[[SOURCE]] : tensor<4xf32>) outs(%[[SOURCE]] : tensor<4xf32>) {
// CHECK: ^bb0(%[[IN:.+]]: tensor<4xf32>, %{{.+}}: tensor<4xf32>):
// CHECK:   wafer.group_yield %[[IN]] : tensor<4xf32>
// CHECK: } : tensor<4xf32>
