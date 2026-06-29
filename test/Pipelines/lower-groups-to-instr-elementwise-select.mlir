// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-instr)' %s | FileCheck --check-prefix=INSTR %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-llvm)' %s | FileCheck --check-prefix=LLVM %s

func.func @elementwise_select_to_instr(%pred: tensor<4xi1>,
                                       %true_value: tensor<4xf32>,
                                       %false_value: tensor<4xf32>,
                                       %out: tensor<4xf32>)
    -> tensor<4xf32> {
  %group = wafer.group
      ins(%pred, %true_value, %false_value
          : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>,
       %arg3: tensor<4xf32>):
    %selected = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%arg0, %arg1, %arg2
          : tensor<4xi1>, tensor<4xf32>, tensor<4xf32>)
      outs(%arg3 : tensor<4xf32>) {
    ^bb0(%pred_scalar: i1, %true_scalar: f32, %false_scalar: f32,
         %out_scalar: f32):
      %0 = arith.select %pred_scalar, %true_scalar, %false_scalar : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>
    wafer.group.yield %selected : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// INSTR-LABEL: func.func @elementwise_select_to_instr
// INSTR-NOT: wafer.group
// INSTR-NOT: linalg.generic
// INSTR: wafer.instr.rdma
// INSTR-SAME: byte_count = 1 : i64
// INSTR: wafer.instr.elementwise <select>
// INSTR: wafer.instr.wdma

// LLVM-LABEL: llvm.func @elementwise_select_to_instr
// LLVM: llvm.call @wafer_select
// LLVM: llvm.func @wafer_select(i32, i32, i32, i32, i64, i32) -> i32
