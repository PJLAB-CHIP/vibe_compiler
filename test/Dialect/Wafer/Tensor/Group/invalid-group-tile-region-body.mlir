// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %dest = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.group ins(%source : tensor<4xf32>)
                    outs(%dest : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>, %out: tensor<4xf32>):
    %tile = wafer.tile.region(%in : tensor<4xf32>) -> (tensor<4xf32>) {
    ^bb1(%arg0: tensor<4xf32>):
      wafer.tile.yield %arg0 : tensor<4xf32>
    }
    wafer.group.yield %tile : tensor<4xf32>
  } : tensor<4xf32>
}

// CHECK: 'wafer.group' op body cannot contain lower-level op 'wafer.tile.region'
