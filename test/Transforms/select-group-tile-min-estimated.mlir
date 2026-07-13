// RUN: wafer-opt --wafer-select-group-tile='logical-rank=0 tile-search=min-estimated-time print-candidate-summary' %s 2>&1 | FileCheck --implicit-check-not=selected_group --implicit-check-not=memref.global --implicit-check-not=memref.get_global --check-prefixes=SUMMARY,IR %s
// RUN: wafer-opt --wafer-select-group-tile='logical-rank=0 tile-search=min-estimated-time tile-search-effort=quick print-candidate-summary candidate-parallelism=2' %s 2>&1 | FileCheck --implicit-check-not=selected_group --implicit-check-not=memref.global --implicit-check-not=memref.get_global --check-prefixes=PARALLEL,IR %s
// RUN: wafer-opt --wafer-lower-groups-to-selected-instr %s | FileCheck --implicit-check-not=wafer.group --implicit-check-not=memref.global --implicit-check-not=memref.get_global --check-prefix=PIPE %s

func.func @elementwise_last_dim_65(%lhs: tensor<2x65xf16>,
                                   %rhs: tensor<2x65xf16>,
                                   %out: tensor<2x65xf16>)
    -> tensor<2x65xf16> {
  %group = wafer.group ins(%lhs, %rhs : tensor<2x65xf16>, tensor<2x65xf16>)
      outs(%out : tensor<2x65xf16>) {
  ^bb0(%arg0: tensor<2x65xf16>, %arg1: tensor<2x65xf16>,
       %arg2: tensor<2x65xf16>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%arg0, %arg1 : tensor<2x65xf16>, tensor<2x65xf16>)
        outs(%arg2 : tensor<2x65xf16>) {
      ^bb0(%lhs_el: f16, %rhs_el: f16, %out_el: f16):
        %add = arith.addf %lhs_el, %rhs_el : f16
        linalg.yield %add : f16
      } -> tensor<2x65xf16>
    wafer.group.yield %sum : tensor<2x65xf16>
  } : tensor<2x65xf16>
  return %group : tensor<2x65xf16>
}

func.func @constant_square(%input: tensor<2x4xf32>,
                           %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %two = arith.constant dense<2.000000e+00> : tensor<2x4xf32>
  %group = wafer.group ins(%input, %two
      : tensor<2x4xf32>, tensor<2x4xf32>)
      outs(%out : tensor<2x4xf32>) {
  ^bb0(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>,
       %arg2: tensor<2x4xf32>):
    %square = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<2x4xf32>)
        outs(%arg2 : tensor<2x4xf32>) {
      ^bb0(%value: f32, %exponent: f32, %out_value: f32):
        %pow = math.powf %value, %exponent : f32
        linalg.yield %pow : f32
      } -> tensor<2x4xf32>
    wafer.group.yield %square : tensor<2x4xf32>
  } : tensor<2x4xf32>
  return %group : tensor<2x4xf32>
}

func.func @constant_predicate_select(%lhs: tensor<2x4xf32>,
                                     %rhs: tensor<2x4xf32>,
                                     %out: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  %predicate = arith.constant dense<true> : tensor<2x4xi1>
  %group = wafer.group ins(%predicate, %lhs, %rhs
      : tensor<2x4xi1>, tensor<2x4xf32>, tensor<2x4xf32>)
      outs(%out : tensor<2x4xf32>) {
  ^bb0(%arg0: tensor<2x4xi1>, %arg1: tensor<2x4xf32>,
       %arg2: tensor<2x4xf32>, %arg3: tensor<2x4xf32>):
    %selected = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
      } ins(%arg0, %arg1, %arg2
          : tensor<2x4xi1>, tensor<2x4xf32>, tensor<2x4xf32>)
        outs(%arg3 : tensor<2x4xf32>) {
      ^bb0(%condition: i1, %true_value: f32, %false_value: f32,
           %out_value: f32):
        %value = arith.select %condition, %true_value, %false_value : f32
        linalg.yield %value : f32
      } -> tensor<2x4xf32>
    wafer.group.yield %selected : tensor<2x4xf32>
  } : tensor<2x4xf32>
  return %group : tensor<2x4xf32>
}

// SUMMARY: wafer.select_group_tile selected group @elementwise_last_dim_65#0
// SUMMARY-SAME: mode=min-estimated-time
// SUMMARY-SAME: tile=[2,65]
// SUMMARY-SAME: estimated_cycles=
// SUMMARY-SAME: rejected=0

// SUMMARY: wafer.select_group_tile selected group @constant_square#0
// SUMMARY-SAME: mode=min-estimated-time
// SUMMARY-SAME: candidates=

// SUMMARY: wafer.select_group_tile selected group @constant_predicate_select#0
// SUMMARY-SAME: mode=min-estimated-time
// SUMMARY-SAME: candidates=

// PARALLEL: wafer.select_group_tile selected group @elementwise_last_dim_65#0
// PARALLEL-SAME: mode=min-estimated-time
// PARALLEL-SAME: tile=[2,65]
// PARALLEL-SAME: candidates=
// PARALLEL-SAME: rejected=0

// PARALLEL: wafer.select_group_tile selected group @constant_square#0
// PARALLEL-SAME: mode=min-estimated-time
// PARALLEL-SAME: candidates=

// PARALLEL: wafer.select_group_tile selected group @constant_predicate_select#0
// PARALLEL-SAME: mode=min-estimated-time
// PARALLEL-SAME: candidates=

// IR-LABEL: func.func @elementwise_last_dim_65
// IR-NOT: wafer.group
// IR-NOT: linalg.generic
// IR: memref.subview
// IR: wafer.instr.elementwise <add>
// IR: wafer.instr.wdma

// IR-LABEL: func.func @constant_square
// IR-NOT: wafer.group
// IR-NOT: math.powf
// IR: wafer.instr.elementwise <mul>
// IR: wafer.instr.wdma

// IR-LABEL: func.func @constant_predicate_select
// IR-NOT: wafer.group
// IR-NOT: arith.select
// IR: wafer.instr.fill
// IR: wafer.instr.bit2fp
// IR: wafer.instr.mask_move
// IR: wafer.instr.wdma

// PIPE-LABEL: func.func @constant_predicate_select
// PIPE: wafer.instr.fill
// PIPE: wafer.instr.bit2fp
// PIPE: wafer.instr.mask_move
