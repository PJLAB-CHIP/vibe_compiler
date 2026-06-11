// RUN: wafer-opt --wafer-select-group-tile='print-candidate-summary spm-limit=73728' %s 2>&1 | FileCheck --check-prefixes=SUMMARY,IR %s

func.func @reduce_requires_internal_split(%input: tensor<1x4096xf32>,
                                          %out: tensor<1xf32>)
    -> tensor<1xf32> {
  %group = wafer.group ins(%input : tensor<1x4096xf32>)
      outs(%out : tensor<1xf32>) {
  ^bb0(%arg0: tensor<1x4096xf32>, %arg1: tensor<1xf32>):
    %init_scalar = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %filled = linalg.fill
        ins(%init_scalar : f32)
        outs(%empty : tensor<1xf32>) -> tensor<1xf32>
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%arg0 : tensor<1x4096xf32>)
        outs(%filled : tensor<1xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %add = arith.addf %value, %acc : f32
        linalg.yield %add : f32
      } -> tensor<1xf32>
    wafer.group.yield %sum : tensor<1xf32>
  } : tensor<1xf32>
  return %group : tensor<1xf32>
}

// SUMMARY: wafer.select_group_tile selected group @reduce_requires_internal_split#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[1]
// SUMMARY-SAME: split=[{{[1-9][0-9]*}}]
// SUMMARY-SAME: rejected=

// IR-LABEL: func.func @reduce_requires_internal_split_selected_group_0
// IR-NOT: wafer.group
// IR-NOT: linalg.generic
// IR: wafer.instr.reduce <sum>
// IR: wafer.instr.reduce <sum>
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma
