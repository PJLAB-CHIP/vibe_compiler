// RUN: wafer-opt -verify-diagnostics %s

func.func @duplicate_inputs(%value: tensor<4xf32>, %out: tensor<4xf32>)
    -> tensor<4xf32> {
  // expected-error@+1 {{'wafer.group' op requires unique boundary SSA values; input at index 1 is duplicated}}
  %result = wafer.group
      ins(%value, %value : tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%first: tensor<4xf32>, %second: tensor<4xf32>,
       %out_arg: tensor<4xf32>):
    wafer.group.yield %first : tensor<4xf32>
  } : tensor<4xf32>
  return %result : tensor<4xf32>
}
