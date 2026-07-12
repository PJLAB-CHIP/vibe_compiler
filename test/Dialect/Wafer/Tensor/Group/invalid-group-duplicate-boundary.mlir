// RUN: wafer-opt -verify-diagnostics %s

func.func @input_aliases_out(%value: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{'wafer.group' op requires unique boundary SSA values; out at index 0 aliases an earlier input or out}}
  %result = wafer.group ins(%value : tensor<4xf32>)
      outs(%value : tensor<4xf32>) {
  ^bb0(%input: tensor<4xf32>, %out: tensor<4xf32>):
    wafer.group.yield %input : tensor<4xf32>
  } : tensor<4xf32>
  return %result : tensor<4xf32>
}
