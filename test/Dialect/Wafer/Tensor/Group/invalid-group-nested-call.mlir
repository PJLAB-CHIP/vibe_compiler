// RUN: wafer-opt -verify-diagnostics %s

func.func private @side_effect()

func.func @nested_call(%input: tensor<4xi32>, %out: tensor<4xi32>)
    -> tensor<4xi32> {
  // expected-error@+1 {{'wafer.group' op body cannot contain unsupported op 'func.call'}}
  %group = wafer.group ins(%input : tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
  ^bb0(%input_arg: tensor<4xi32>, %out_arg: tensor<4xi32>):
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%input_arg : tensor<4xi32>) outs(%out_arg : tensor<4xi32>) {
      ^bb0(%value: i32, %init: i32):
        func.call @side_effect() : () -> ()
        linalg.yield %value : i32
      } -> tensor<4xi32>
    wafer.group.yield %result : tensor<4xi32>
  } : tensor<4xi32>
  return %group : tensor<4xi32>
}
