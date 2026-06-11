// RUN: wafer-opt --wafer-select-group-tile='tile-search=min-estimated-time print-candidate-summary' %s 2>&1 | FileCheck --check-prefixes=SUMMARY,IR %s

func.func @elementwise_last_dim_1000(%lhs: tensor<2x1000xf16>,
                                     %rhs: tensor<2x1000xf16>,
                                     %out: tensor<2x1000xf16>)
    -> tensor<2x1000xf16> {
  %group = wafer.group ins(%lhs, %rhs : tensor<2x1000xf16>, tensor<2x1000xf16>)
      outs(%out : tensor<2x1000xf16>) {
  ^bb0(%arg0: tensor<2x1000xf16>, %arg1: tensor<2x1000xf16>,
       %arg2: tensor<2x1000xf16>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%arg0, %arg1 : tensor<2x1000xf16>, tensor<2x1000xf16>)
        outs(%arg2 : tensor<2x1000xf16>) {
      ^bb0(%lhs_el: f16, %rhs_el: f16, %out_el: f16):
        %add = arith.addf %lhs_el, %rhs_el : f16
        linalg.yield %add : f16
      } -> tensor<2x1000xf16>
    wafer.group.yield %sum : tensor<2x1000xf16>
  } : tensor<2x1000xf16>
  return %group : tensor<2x1000xf16>
}

// SUMMARY: wafer.select_group_tile selected group @elementwise_last_dim_1000#0
// SUMMARY-SAME: mode=min-estimated-time
// SUMMARY-SAME: tile=[
// SUMMARY-SAME: estimated_cycles=
// SUMMARY-SAME: rejected=0

// IR-LABEL: func.func @elementwise_last_dim_1000_selected_group_0
// IR-NOT: wafer.group
// IR-NOT: linalg.generic
// IR: memref.subview
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma
