// RUN: wafer-opt -verify-diagnostics %s

func.func @duplicate_outs(%input: tensor<4xf32>, %out: tensor<4xf32>)
    -> (tensor<4xf32>, tensor<4xf32>) {
  // expected-error@+1 {{'wafer.group' op requires unique boundary SSA values; out at index 1 aliases an earlier input or out}}
  %result:2 = wafer.group ins(%input : tensor<4xf32>)
      outs(%out, %out : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%input_arg: tensor<4xf32>, %first_out: tensor<4xf32>,
       %second_out: tensor<4xf32>):
    wafer.group.yield %first_out, %second_out
        : tensor<4xf32>, tensor<4xf32>
  } : tensor<4xf32>, tensor<4xf32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xf32>
}
