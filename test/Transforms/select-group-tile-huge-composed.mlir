// RUN: wafer-opt --wafer-select-group-tile='print-candidate-summary tile-search-effort=quick' %s 2>&1 | FileCheck --implicit-check-not=selected_group --check-prefixes=SUMMARY,IR %s

func.func @huge_if_branches_multi_output_tiled(
    %lhs: tensor<64x12288xf16>, %rhs: tensor<12288x64xf16>,
    %add_lhs: tensor<64x64xf16>, %add_rhs: tensor<64x64xf16>,
    %out0: tensor<64x64xf16>, %out1: tensor<64x64xf16>,
    %cond: i1) -> (tensor<64x64xf16>, tensor<64x64xf16>) {
  %result:2 = scf.if %cond
      -> (tensor<64x64xf16>, tensor<64x64xf16>) {
    %group:2 = wafer.group
        ins(%lhs, %rhs, %add_lhs, %add_rhs
            : tensor<64x12288xf16>, tensor<12288x64xf16>,
              tensor<64x64xf16>, tensor<64x64xf16>)
        outs(%out0, %out1 : tensor<64x64xf16>, tensor<64x64xf16>) {
    ^bb0(%arg0: tensor<64x12288xf16>, %arg1: tensor<12288x64xf16>,
         %arg2: tensor<64x64xf16>, %arg3: tensor<64x64xf16>,
         %arg4: tensor<64x64xf16>, %arg5: tensor<64x64xf16>):
      %mm = linalg.matmul
          ins(%arg0, %arg1 : tensor<64x12288xf16>, tensor<12288x64xf16>)
          outs(%arg4 : tensor<64x64xf16>) -> tensor<64x64xf16>
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg2, %arg3 : tensor<64x64xf16>, tensor<64x64xf16>)
          outs(%arg5 : tensor<64x64xf16>) {
      ^bb0(%lhs_el: f16, %rhs_el: f16, %out_el: f16):
        %add = arith.addf %lhs_el, %rhs_el : f16
        linalg.yield %add : f16
      } -> tensor<64x64xf16>
      wafer.group.yield %mm, %sum : tensor<64x64xf16>, tensor<64x64xf16>
    } : tensor<64x64xf16>, tensor<64x64xf16>
    scf.yield %group#0, %group#1
        : tensor<64x64xf16>, tensor<64x64xf16>
  } else {
    scf.yield %out0, %out1 : tensor<64x64xf16>, tensor<64x64xf16>
  }
  return %result#0, %result#1 : tensor<64x64xf16>, tensor<64x64xf16>
}

// SUMMARY: wafer.select_group_tile selected group @huge_if_branches_multi_output_tiled#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[
// SUMMARY-SAME: split=[
// SUMMARY-SAME: rejected=

// IR-LABEL: func.func @huge_if_branches_multi_output_tiled
// IR-NOT: wafer.group
// IR-NOT: linalg.
// IR: scf.if
// IR: wafer.tile.region
// IR: wafer.spm.offset
// IR: wafer.instr.gemm
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma
// IR: scf.yield
