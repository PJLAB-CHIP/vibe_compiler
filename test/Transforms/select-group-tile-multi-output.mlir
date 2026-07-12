// RUN: wafer-opt --wafer-select-group-tile='logical-rank=0 print-candidate-summary' %s 2>&1 | FileCheck --implicit-check-not=selected_group --check-prefixes=SUMMARY,IR %s

func.func @multi_output_same_domain(%lhs: tensor<1x4096xf32>,
                                    %rhs: tensor<1x4096xf32>,
                                    %out0: tensor<1x4096xf32>,
                                    %out1: tensor<1x4096xf32>)
    -> (tensor<1x4096xf32>, tensor<1x4096xf32>) {
  %group:2 = wafer.group ins(%lhs, %rhs : tensor<1x4096xf32>, tensor<1x4096xf32>)
      outs(%out0, %out1 : tensor<1x4096xf32>, tensor<1x4096xf32>) {
  ^bb0(%arg0: tensor<1x4096xf32>, %arg1: tensor<1x4096xf32>,
       %arg2: tensor<1x4096xf32>, %arg3: tensor<1x4096xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%arg0, %arg1 : tensor<1x4096xf32>, tensor<1x4096xf32>)
        outs(%arg2 : tensor<1x4096xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<1x4096xf32>
    %product = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%arg0, %arg1 : tensor<1x4096xf32>, tensor<1x4096xf32>)
        outs(%arg3 : tensor<1x4096xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %mul = arith.mulf %lhs_el, %rhs_el : f32
        linalg.yield %mul : f32
      } -> tensor<1x4096xf32>
    wafer.group.yield %sum, %product : tensor<1x4096xf32>, tensor<1x4096xf32>
  } : tensor<1x4096xf32>, tensor<1x4096xf32>
  return %group#0, %group#1 : tensor<1x4096xf32>, tensor<1x4096xf32>
}

// SUMMARY: wafer.select_group_tile selected group @multi_output_same_domain#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[1,{{[0-9]+}}]
// SUMMARY-SAME: split=[]

// IR-LABEL: func.func @multi_output_same_domain
// IR-NOT: wafer.group
// IR-NOT: linalg.generic
// IR: memref.subview
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma
// IR: wafer.instr.elementwise <mul>
// IR: wafer.instr.wdma
