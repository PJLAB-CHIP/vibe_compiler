// RUN: not wafer-opt --wafer-select-group-tile='tile-search-effort=quick' %s 2>&1 | FileCheck %s

func.func @large_if_matmul_bias_chain_tiling_gap(
    %lhs: tensor<768x512xf16>, %rhs: tensor<512x768xf16>,
    %bias: tensor<768x768xf16>, %out: tensor<768x768xf16>,
    %cond: i1) -> tensor<768x768xf16> {
  %group = wafer.group ins(%lhs, %rhs, %bias, %cond
      : tensor<768x512xf16>, tensor<512x768xf16>, tensor<768x768xf16>, i1)
      outs(%out : tensor<768x768xf16>) {
  ^bb0(%arg0: tensor<768x512xf16>, %arg1: tensor<512x768xf16>,
       %arg2: tensor<768x768xf16>, %arg3: i1,
       %arg4: tensor<768x768xf16>):
    %selected = scf.if %arg3 -> tensor<768x768xf16> {
      %mm = linalg.matmul
          ins(%arg0, %arg1 : tensor<768x512xf16>, tensor<512x768xf16>)
          outs(%arg4 : tensor<768x768xf16>) -> tensor<768x768xf16>
      %with_bias = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%mm, %arg2 : tensor<768x768xf16>, tensor<768x768xf16>)
          outs(%arg4 : tensor<768x768xf16>) {
        ^bb0(%mm_el: f16, %bias_el: f16, %out_el: f16):
          %sum = arith.addf %mm_el, %bias_el : f16
          linalg.yield %sum : f16
        } -> tensor<768x768xf16>
      scf.yield %with_bias : tensor<768x768xf16>
    } else {
      scf.yield %arg4 : tensor<768x768xf16>
    }
    wafer.group.yield %selected : tensor<768x768xf16>
  } : tensor<768x768xf16>
  return %group : tensor<768x768xf16>
}

// CHECK: no_candidate: tile selection found no passing candidate
// CHECK-SAME: candidate tile materialization requires linalg roots
