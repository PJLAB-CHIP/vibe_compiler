// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-memory-planned-instr)' %s 2>&1 | FileCheck %s

// A tensor recurrence whose body materializes a fresh result allocation needs
// explicit multi-instance or ping-pong placement. The memory-planned pipeline
// must reject it instead of assigning one static address to every dynamic
// allocation instance.
func.func @fresh_loop_result_allocation(
    %input: tensor<4xf32>, %out: tensor<4xf32>,
    %lb: index, %ub: index, %step: index) -> tensor<4xf32> {
  %group = wafer.group ins(%input, %lb, %ub, %step
      : tensor<4xf32>, index, index, index) outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: index, %arg2: index, %arg3: index,
       %arg4: tensor<4xf32>):
    %loop_result = scf.for %i = %arg1 to %arg2 step %arg3
        iter_args(%acc = %arg4) -> (tensor<4xf32>) {
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%acc, %arg0 : tensor<4xf32>, tensor<4xf32>)
          outs(%acc : tensor<4xf32>) {
        ^bb0(%acc_el: f32, %input_el: f32, %out_el: f32):
          %add = arith.addf %acc_el, %input_el : f32
          linalg.yield %add : f32
        } -> tensor<4xf32>
      scf.yield %sum : tensor<4xf32>
    }
    wafer.group.yield %loop_result : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// CHECK: unsupported_lifetime_alias: loop-body SPM allocation cannot be loop-carried without multi-instance placement
