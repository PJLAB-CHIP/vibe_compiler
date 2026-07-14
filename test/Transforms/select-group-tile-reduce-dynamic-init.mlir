// RUN: not wafer-opt --mlir-print-ir-after-failure --wafer-select-group-tile='logical-rank=0' %s 2>&1 | FileCheck %s

func.func @reject_dynamic_reduce_init(
    %input: tensor<2x4xf32>, %init: tensor<f32>, %out: tensor<2xf32>)
    -> tensor<2xf32> {
  %result = wafer.group
      ins(%input, %init : tensor<2x4xf32>, tensor<f32>)
      outs(%out : tensor<2xf32>) {
  ^bb0(%input_arg: tensor<2x4xf32>, %init_arg: tensor<f32>,
       %out_arg: tensor<2xf32>):
    %init_scalar = tensor.extract %init_arg[] : tensor<f32>
    %empty = tensor.empty() : tensor<2xf32>
    %filled = linalg.fill ins(%init_scalar : f32)
        outs(%empty : tensor<2xf32>) -> tensor<2xf32>
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input_arg : tensor<2x4xf32>)
        outs(%filled : tensor<2xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %next = arith.addf %value, %acc : f32
        linalg.yield %next : f32
      } -> tensor<2xf32>
    wafer.group.yield %sum : tensor<2xf32>
  } : tensor<2xf32>
  return %result : tensor<2xf32>
}

// CHECK: no_candidate: tile selection found no passing candidate
// CHECK-SAME: reduction init must be an arith.constant or typed init_value
// CHECK: IR Dump After SelectGroupTilePass Failed
// CHECK: func.func @reject_dynamic_reduce_init
// CHECK: wafer.group
// CHECK: tensor.extract
// CHECK: linalg.generic
// CHECK-NOT: wafer.instr.
