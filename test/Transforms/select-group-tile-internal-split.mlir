// RUN: wafer-opt --wafer-select-group-tile='print-candidate-summary spm-limit=73728' %s 2>&1 | FileCheck --check-prefixes=SUMMARY,IR %s

func.func @matmul_requires_k_split(%lhs: tensor<1x1024xf16>,
                                   %rhs: tensor<1024x1xf16>,
                                   %out: tensor<1x1xf16>)
    -> tensor<1x1xf16> {
  %group = wafer.group ins(%lhs, %rhs : tensor<1x1024xf16>, tensor<1024x1xf16>)
      outs(%out : tensor<1x1xf16>) {
  ^bb0(%arg0: tensor<1x1024xf16>, %arg1: tensor<1024x1xf16>,
       %arg2: tensor<1x1xf16>):
    %mm = linalg.matmul
        ins(%arg0, %arg1 : tensor<1x1024xf16>, tensor<1024x1xf16>)
        outs(%arg2 : tensor<1x1xf16>) -> tensor<1x1xf16>
    wafer.group.yield %mm : tensor<1x1xf16>
  } : tensor<1x1xf16>
  return %group : tensor<1x1xf16>
}

// SUMMARY: wafer.select_group_tile selected group @matmul_requires_k_split#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[1,1]
// SUMMARY-SAME: split=[{{[1-9][0-9]*}}]
// SUMMARY-SAME: rejected=

// IR-LABEL: func.func @matmul_requires_k_split_selected_group_0
// IR-NOT: wafer.group
// IR-NOT: linalg.matmul
// IR: memref.subview
// IR-SAME: [0, 0] [1, {{[0-9]+}}] [1, 1]
// IR: memref.subview
// IR-SAME: [0, 0] [{{[0-9]+}}, 1] [1, 1]
// IR: wafer.instr.gemm
// IR-SAME: k = {{[0-9]+}} : i64
// IR-NOT: k = 1024 : i64
// IR: wafer.instr.wdma
