// RUN: not wafer-opt --wafer-select-group-tile='tile-search=bad-mode' %s 2>&1 | FileCheck --check-prefix=BAD-MODE %s
// RUN: not wafer-opt --wafer-select-group-tile='spm-limit=65537 max-candidates-per-dim=1' %s 2>&1 | FileCheck --check-prefix=SPM %s

func.func @if_group_spm_failure(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                                %out: tensor<4xf32>, %cond: i1)
    -> tensor<4xf32> {
  %group = wafer.group ins(%lhs, %rhs, %cond : tensor<4xf32>, tensor<4xf32>, i1)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: i1,
       %arg3: tensor<4xf32>):
    %selected = scf.if %arg2 -> tensor<4xf32> {
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
          outs(%arg3 : tensor<4xf32>) {
        ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
          %add = arith.addf %lhs_el, %rhs_el : f32
          linalg.yield %add : f32
        } -> tensor<4xf32>
      scf.yield %sum : tensor<4xf32>
    } else {
      scf.yield %arg1 : tensor<4xf32>
    }
    wafer.group.yield %selected : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// BAD-MODE: invalid_tile_search: expected first-legal or min-estimated-time

// SPM: no_candidate: R3.2h found no passing candidate
// SPM-SAME: last failure: R3.2f
// SPM-NOT: R3.2g
