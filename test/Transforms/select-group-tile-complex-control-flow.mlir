// RUN: wafer-opt --wafer-select-group-tile='logical-rank=0 print-candidate-summary' %s 2>&1 | FileCheck --implicit-check-not=selected_group --check-prefixes=SUMMARY,IR %s

func.func @if_matmul_bias_chain(%lhs: tensor<4x8xf16>,
                                %rhs: tensor<8x3xf16>,
                                %bias: tensor<4x3xf16>,
                                %out: tensor<4x3xf16>,
                                %cond: i1) -> tensor<4x3xf16> {
  %group = wafer.group ins(%lhs, %rhs, %bias, %cond
      : tensor<4x8xf16>, tensor<8x3xf16>, tensor<4x3xf16>, i1)
      outs(%out : tensor<4x3xf16>) {
  ^bb0(%arg0: tensor<4x8xf16>, %arg1: tensor<8x3xf16>,
       %arg2: tensor<4x3xf16>, %arg3: i1,
       %arg4: tensor<4x3xf16>):
    %selected = scf.if %arg3 -> tensor<4x3xf16> {
      %zero = arith.constant 0.0 : f16
      %init = linalg.fill ins(%zero : f16)
          outs(%arg4 : tensor<4x3xf16>) -> tensor<4x3xf16>
      %mm = linalg.matmul
          ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<8x3xf16>)
          outs(%init : tensor<4x3xf16>) -> tensor<4x3xf16>
      %with_bias = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%mm, %arg2 : tensor<4x3xf16>, tensor<4x3xf16>)
          outs(%arg4 : tensor<4x3xf16>) {
        ^bb0(%mm_el: f16, %bias_el: f16, %out_el: f16):
          %sum = arith.addf %mm_el, %bias_el : f16
          linalg.yield %sum : f16
        } -> tensor<4x3xf16>
      scf.yield %with_bias : tensor<4x3xf16>
    } else {
      %bias_only = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg4, %arg2 : tensor<4x3xf16>, tensor<4x3xf16>)
          outs(%arg4 : tensor<4x3xf16>) {
        ^bb0(%out_el: f16, %bias_el: f16, %init_el: f16):
          %sum = arith.addf %out_el, %bias_el : f16
          linalg.yield %sum : f16
        } -> tensor<4x3xf16>
      scf.yield %bias_only : tensor<4x3xf16>
    }
    wafer.group.yield %selected : tensor<4x3xf16>
  } : tensor<4x3xf16>
  return %group : tensor<4x3xf16>
}

func.func @loop_identity_recurrence(%input: tensor<4x4xf16>,
                                    %out: tensor<4x4xf16>,
                                    %lb: index, %ub: index, %step: index)
    -> tensor<4x4xf16> {
  %group = wafer.group ins(%input, %lb, %ub, %step
      : tensor<4x4xf16>, index, index, index)
      outs(%out : tensor<4x4xf16>) {
  ^bb0(%arg0: tensor<4x4xf16>, %arg1: index, %arg2: index,
       %arg3: index, %arg4: tensor<4x4xf16>):
    %loop_result = scf.for %i = %arg1 to %arg2 step %arg3
        iter_args(%acc = %arg4) -> (tensor<4x4xf16>) {
      scf.yield %acc : tensor<4x4xf16>
    }
    wafer.group.yield %loop_result : tensor<4x4xf16>
  } : tensor<4x4xf16>
  return %group : tensor<4x4xf16>
}

// SUMMARY: wafer.select_group_tile selected group @if_matmul_bias_chain#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[4,3]
// SUMMARY-SAME: representatives=1
// SUMMARY: wafer.select_group_tile selected group @loop_identity_recurrence#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[4,4]

// IR-LABEL: func.func @if_matmul_bias_chain
// IR-NOT: wafer.group
// IR-NOT: linalg.
// IR: wafer.tile.region
// IR: scf.if
// IR: wafer.spm.offset
// IR: wafer.instr.gemm
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma

// IR-LABEL: func.func @loop_identity_recurrence
// IR-NOT: wafer.group
// IR-NOT: linalg.
// IR: wafer.tile.region
// IR: wafer.spm.offset
// IR: scf.for
// IR: wafer.instr.local_fence
// IR: wafer.instr.wdma
