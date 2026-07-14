// RUN: wafer-opt --wafer-select-group-tile='logical-rank=0 print-candidate-summary' %s 2>&1 | FileCheck --implicit-check-not=selected_group --check-prefixes=SUMMARY,IR %s

func.func @elementwise_small(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                             %out: tensor<4xf32>) -> tensor<4xf32> {
  %group = wafer.group ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
        outs(%arg2 : tensor<4xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<4xf32>
    wafer.group.yield %sum : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

func.func @static_slice_elementwise(%input: tensor<4x8xf16>,
                                    %out: tensor<4x8xf16>)
    -> tensor<4x8xf16> {
  %group = wafer.group ins(%input : tensor<4x8xf16>)
      outs(%out : tensor<4x8xf16>) {
  ^bb0(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>):
    %tile = tensor.extract_slice %arg0[1, 2] [2, 3] [1, 1]
        : tensor<4x8xf16> to tensor<2x3xf16>
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%tile, %tile : tensor<2x3xf16>, tensor<2x3xf16>)
        outs(%tile : tensor<2x3xf16>) {
      ^bb0(%lhs_el: f16, %rhs_el: f16, %out_el: f16):
        %add = arith.addf %lhs_el, %rhs_el : f16
        linalg.yield %add : f16
      } -> tensor<2x3xf16>
    %updated = tensor.insert_slice %sum into %arg1[1, 2] [2, 3] [1, 1]
        : tensor<2x3xf16> into tensor<4x8xf16>
    wafer.group.yield %updated : tensor<4x8xf16>
  } : tensor<4x8xf16>
  return %group : tensor<4x8xf16>
}

func.func @large_matmul_last_dim_1000(%lhs: tensor<257x1000xf16>,
                                      %rhs: tensor<1000x129xf16>,
                                      %out: tensor<257x129xf16>)
    -> tensor<257x129xf16> {
  %group = wafer.group ins(%lhs, %rhs : tensor<257x1000xf16>, tensor<1000x129xf16>)
      outs(%out : tensor<257x129xf16>) {
  ^bb0(%arg0: tensor<257x1000xf16>, %arg1: tensor<1000x129xf16>,
       %arg2: tensor<257x129xf16>):
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%arg2 : tensor<257x129xf16>) -> tensor<257x129xf16>
    %mm = linalg.matmul
        ins(%arg0, %arg1 : tensor<257x1000xf16>, tensor<1000x129xf16>)
        outs(%init : tensor<257x129xf16>) -> tensor<257x129xf16>
    wafer.group.yield %mm : tensor<257x129xf16>
  } : tensor<257x129xf16>
  return %group : tensor<257x129xf16>
}

func.func @reduce_sum_group(%input: tensor<2x4xf32>, %init: tensor<f32>,
                            %out: tensor<2xf32>) -> tensor<2xf32> {
  %group = wafer.group ins(%input, %init : tensor<2x4xf32>, tensor<f32>)
      outs(%out : tensor<2xf32>) {
  ^bb0(%arg0: tensor<2x4xf32>, %arg1: tensor<f32>, %arg2: tensor<2xf32>):
    %init_scalar = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2xf32>
    %filled = linalg.fill
        ins(%init_scalar : f32)
        outs(%empty : tensor<2xf32>) -> tensor<2xf32>
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%arg0 : tensor<2x4xf32>)
        outs(%filled : tensor<2xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %add = arith.addf %value, %acc : f32
        linalg.yield %add : f32
      } -> tensor<2xf32>
    wafer.group.yield %sum : tensor<2xf32>
  } : tensor<2xf32>
  return %group : tensor<2xf32>
}

func.func @if_group(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                    %out: tensor<4xf32>, %cond: i1) -> tensor<4xf32> {
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

func.func @loop_group(%input: tensor<4xf32>, %out: tensor<4xf32>,
                      %lb: index, %ub: index, %step: index)
    -> tensor<4xf32> {
  %group = wafer.group ins(%input, %lb, %ub, %step : tensor<4xf32>, index, index, index)
      outs(%out : tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: index, %arg2: index, %arg3: index,
       %arg4: tensor<4xf32>):
    %loop_result = scf.for %i = %arg1 to %arg2 step %arg3
        iter_args(%acc = %arg4) -> (tensor<4xf32>) {
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%acc, %arg0 : tensor<4xf32>, tensor<4xf32>)
          outs(%acc : tensor<4xf32>) {
        ^bb0(%acc_el: f32, %input_el: f32, %out_el: f32):
          %add = arith.addf %acc_el, %input_el : f32
          linalg.yield %add : f32
        } -> tensor<4xf32>
      scf.yield %sum : tensor<4xf32>
    }
    wafer.group.yield %loop_result : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// SUMMARY: wafer.select_group_tile selected group @elementwise_small#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[4]
// SUMMARY: wafer.select_group_tile selected group @static_slice_elementwise#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY: wafer.select_group_tile selected group @large_matmul_last_dim_1000#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[
// SUMMARY-SAME: representatives={{[1-9][0-9]*}}
// SUMMARY: wafer.select_group_tile selected group @reduce_sum_group#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY-SAME: tile=[2]
// SUMMARY: wafer.select_group_tile selected group @if_group#0
// SUMMARY-SAME: mode=first-legal
// SUMMARY: wafer.select_group_tile selected group @loop_group#0
// SUMMARY-SAME: mode=first-legal

// IR-LABEL: func.func @elementwise_small
// IR-NOT: wafer.group
// IR-NOT: linalg.generic
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma

// IR-LABEL: func.func @static_slice_elementwise
// IR-NOT: wafer.group
// IR: memref.subview
// IR-SAME: [1, 2] [2, 3] [1, 1]
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma

// IR-LABEL: func.func @large_matmul_last_dim_1000
// IR-NOT: wafer.group
// IR-NOT: linalg.matmul
// IR: memref.subview
// IR-SAME: [0, 0] [{{[0-9]+}}, {{[0-9]+}}] [1, 1]
// IR: memref.subview
// IR-SAME: [0, 0] [{{[0-9]+}}, {{[0-9]+}}] [1, 1]
// IR: wafer.instr.gemm
// IR-SAME: k = {{[0-9]+}} : i64
// IR: wafer.instr.wdma

// IR-LABEL: func.func @reduce_sum_group
// IR-NOT: wafer.group
// IR-NOT: linalg.generic
// IR-NOT: wafer.instr.reduce
// IR: wafer.instr.fill
// IR: wafer.instr.local_fence
// IR: wafer.instr.gather_scatter
// IR: wafer.instr.local_fence
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.local_fence
// IR: wafer.instr.gather_scatter
// IR: wafer.instr.local_fence
// IR: wafer.instr.wdma

// IR-LABEL: func.func @if_group
// IR-NOT: wafer.group
// IR: scf.if
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma

// IR-LABEL: func.func @loop_group
// IR-NOT: wafer.group
// IR: scf.for
// IR: wafer.instr.elementwise <add>
// IR: scf.yield
// IR: wafer.instr.wdma
