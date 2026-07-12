// RUN: wafer-opt --wafer-select-group-tile='logical-rank=0 print-candidate-summary tile-search-effort=quick' %s 2>&1 | FileCheck --implicit-check-not=selected_group --check-prefixes=SUMMARY,IR %s

func.func @nested_multi_output_atomic_commit(
    %add_lhs: tensor<4x4xf16>, %add_rhs: tensor<4x4xf16>,
    %mul_lhs: tensor<4x4xf16>, %mul_rhs: tensor<4x4xf16>,
    %out0: tensor<4x4xf16>, %out1: tensor<4x4xf16>,
    %cond: i1) -> (tensor<4x4xf16>, tensor<4x4xf16>) {
  %result:2 = scf.if %cond
      -> (tensor<4x4xf16>, tensor<4x4xf16>) {
    %group:2 = wafer.group
        ins(%add_lhs, %add_rhs, %mul_lhs, %mul_rhs
            : tensor<4x4xf16>, tensor<4x4xf16>,
              tensor<4x4xf16>, tensor<4x4xf16>)
        outs(%out0, %out1 : tensor<4x4xf16>, tensor<4x4xf16>) {
    ^bb0(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf16>,
         %arg2: tensor<4x4xf16>, %arg3: tensor<4x4xf16>,
         %arg4: tensor<4x4xf16>, %arg5: tensor<4x4xf16>):
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg0, %arg1 : tensor<4x4xf16>, tensor<4x4xf16>)
          outs(%arg4 : tensor<4x4xf16>) {
      ^bb0(%lhs_el: f16, %rhs_el: f16, %out_el: f16):
        %add = arith.addf %lhs_el, %rhs_el : f16
        linalg.yield %add : f16
      } -> tensor<4x4xf16>
      %product = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg2, %arg3 : tensor<4x4xf16>, tensor<4x4xf16>)
          outs(%arg5 : tensor<4x4xf16>) {
      ^bb0(%lhs_el: f16, %rhs_el: f16, %out_el: f16):
        %mul = arith.mulf %lhs_el, %rhs_el : f16
        linalg.yield %mul : f16
      } -> tensor<4x4xf16>
      wafer.group.yield %sum, %product
          : tensor<4x4xf16>, tensor<4x4xf16>
    } : tensor<4x4xf16>, tensor<4x4xf16>
    scf.yield %group#0, %group#1
        : tensor<4x4xf16>, tensor<4x4xf16>
  } else {
    scf.yield %out0, %out1 : tensor<4x4xf16>, tensor<4x4xf16>
  }
  return %result#0, %result#1 : tensor<4x4xf16>, tensor<4x4xf16>
}

// SUMMARY: wafer.select_group_tile selected group @nested_multi_output_atomic_commit#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[4,4]
// SUMMARY-SAME: split=[]
// SUMMARY-SAME: rejected=0

// IR-LABEL: func.func @nested_multi_output_atomic_commit
// IR-NOT: wafer.group
// IR-NOT: linalg.
// IR: scf.if
// IR: wafer.tile.region
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma
// IR: wafer.instr.elementwise <mul>
// IR: wafer.instr.wdma
// IR: scf.yield
