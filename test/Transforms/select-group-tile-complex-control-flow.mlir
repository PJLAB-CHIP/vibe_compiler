// RUN: wafer-opt --wafer-select-group-tile='print-candidate-summary' %s 2>&1 | FileCheck --implicit-check-not=selected_group --check-prefixes=SUMMARY,IR %s

func.func @large_if_matmul_bias_chain(%lhs: tensor<257x1000xf16>,
                                      %rhs: tensor<1000x129xf16>,
                                      %bias: tensor<257x129xf16>,
                                      %out: tensor<257x129xf16>,
                                      %cond: i1) -> tensor<257x129xf16> {
  %group = wafer.group ins(%lhs, %rhs, %bias, %cond
      : tensor<257x1000xf16>, tensor<1000x129xf16>, tensor<257x129xf16>, i1)
      outs(%out : tensor<257x129xf16>) {
  ^bb0(%arg0: tensor<257x1000xf16>, %arg1: tensor<1000x129xf16>,
       %arg2: tensor<257x129xf16>, %arg3: i1,
       %arg4: tensor<257x129xf16>):
    %selected = scf.if %arg3 -> tensor<257x129xf16> {
      %mm = linalg.matmul
          ins(%arg0, %arg1 : tensor<257x1000xf16>, tensor<1000x129xf16>)
          outs(%arg4 : tensor<257x129xf16>) -> tensor<257x129xf16>
      %with_bias = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%mm, %arg2 : tensor<257x129xf16>, tensor<257x129xf16>)
          outs(%arg4 : tensor<257x129xf16>) {
        ^bb0(%mm_el: f16, %bias_el: f16, %out_el: f16):
          %sum = arith.addf %mm_el, %bias_el : f16
          linalg.yield %sum : f16
        } -> tensor<257x129xf16>
      scf.yield %with_bias : tensor<257x129xf16>
    } else {
      %bias_only = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg4, %arg2 : tensor<257x129xf16>, tensor<257x129xf16>)
          outs(%arg4 : tensor<257x129xf16>) {
        ^bb0(%out_el: f16, %bias_el: f16, %init_el: f16):
          %sum = arith.addf %out_el, %bias_el : f16
          linalg.yield %sum : f16
        } -> tensor<257x129xf16>
      scf.yield %bias_only : tensor<257x129xf16>
    }
    wafer.group.yield %selected : tensor<257x129xf16>
  } : tensor<257x129xf16>
  return %group : tensor<257x129xf16>
}

func.func @large_loop_elementwise_accumulate(%input: tensor<2x1000xf16>,
                                             %out: tensor<2x1000xf16>,
                                             %lb: index, %ub: index,
                                             %step: index)
    -> tensor<2x1000xf16> {
  %group = wafer.group ins(%input, %lb, %ub, %step
      : tensor<2x1000xf16>, index, index, index)
      outs(%out : tensor<2x1000xf16>) {
  ^bb0(%arg0: tensor<2x1000xf16>, %arg1: index, %arg2: index,
       %arg3: index, %arg4: tensor<2x1000xf16>):
    %loop_result = scf.for %i = %arg1 to %arg2 step %arg3
        iter_args(%acc = %arg4) -> (tensor<2x1000xf16>) {
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%acc, %arg0 : tensor<2x1000xf16>, tensor<2x1000xf16>)
          outs(%acc : tensor<2x1000xf16>) {
        ^bb0(%acc_el: f16, %input_el: f16, %out_el: f16):
          %sum_el = arith.addf %acc_el, %input_el : f16
          linalg.yield %sum_el : f16
        } -> tensor<2x1000xf16>
      scf.yield %sum : tensor<2x1000xf16>
    }
    wafer.group.yield %loop_result : tensor<2x1000xf16>
  } : tensor<2x1000xf16>
  return %group : tensor<2x1000xf16>
}

// SUMMARY: wafer.select_group_tile selected group @large_if_matmul_bias_chain#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[257,129]
// SUMMARY-SAME: representatives=1
// SUMMARY: wafer.select_group_tile selected group @large_loop_elementwise_accumulate#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[2,1000]

// IR-LABEL: func.func @large_if_matmul_bias_chain
// IR-NOT: wafer.group
// IR-NOT: linalg.
// IR: wafer.tile.region
// IR: scf.if
// IR: wafer.spm.offset
// IR: wafer.instr.gemm
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma

// IR-LABEL: func.func @large_loop_elementwise_accumulate
// IR-NOT: wafer.group
// IR-NOT: linalg.
// IR: wafer.tile.region
// IR: scf.for
// IR: wafer.spm.offset
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma
