// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' -mlir-pass-statistics -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=STATS

// STATS: ConvertTileRegionToInstrPass
// STATS-NEXT: {{ *}}(S) {{[1-9][0-9]*}} dataflow-ops-lowered
// STATS: PlaceRequiredNCCJoinsPass
// STATS-NEXT: {{ *}}(S) {{[1-9][0-9]*}} required-joins-inserted

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#row = affine_map<(d0, d1) -> (d1)>
#column = affine_map<(d0, d1) -> (d0)>

func.func @identity_maps_strip() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %lhs = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %rhs = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %lhs, %rhs
      {indexing_maps = [#id2, #id2, #id2]}
      : (memref<2x3xf32, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @identity_maps_strip
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.elementwise <add> %{{.*}}, %{{.*}} into %{{.*}}
// CHECK-NOT: indexing_maps

func.func @transpose_and_row_broadcast() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %matrix = memref.alloc() : memref<3x2xf32, #wafer.memory<spm, tensor>>
  %row_value = memref.alloc() : memref<3xf32, #wafer.memory<spm, tensor>>
  %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %matrix, %row_value
      {indexing_maps = [#transpose, #row, #id2]}
      : (memref<3x2xf32, #wafer.memory<spm, tensor>>,
         memref<3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @transpose_and_row_broadcast
// CHECK: %[[TRANSPOSED:.+]] = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %{{.*}} to %[[TRANSPOSED]]
// CHECK: %[[ROW:.+]] = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %{{.*}} to %[[ROW]]
// CHECK-SAME: byte_count = 24 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 12, 0, 0>
// CHECK-SAME: inner_bytes = 12 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: src_strides = array<i64: 0, 0, 0>
// CHECK: wafer.instr.elementwise <add> %[[TRANSPOSED]], %[[ROW]] into %{{.*}}
// CHECK: wafer.instr.ncc_join [0]
// CHECK-NOT: indexing_maps

func.func @column_broadcast() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %column_value = memref.alloc() : memref<2xf32, #wafer.memory<spm, tensor>>
  %matrix = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %sum = wafer.tile.elementwise #wafer.elementwise_kind<mul> %column_value, %matrix
      {indexing_maps = [#column, #id2, #id2]}
      : (memref<2xf32, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @column_broadcast
// CHECK: wafer.instr.gather_scatter %{{.*}} to %[[COLUMN:[^ ]+]] {
// CHECK: wafer.instr.elementwise <mul> %[[COLUMN]], %{{.*}} into %{{.*}}
// CHECK: wafer.instr.ncc_join [0]
// CHECK-NOT: indexing_maps

func.func @select_row_broadcast() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %predicate = memref.alloc() : memref<2x3xi1, #wafer.memory<spm, tensor>>
  %true_row = memref.alloc() : memref<3xf32, #wafer.memory<spm, tensor>>
  %false_value = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
      %predicate, %true_row, %false_value
      {indexing_maps = [#id2, #row, #id2, #id2]}
      : (memref<2x3xi1, #wafer.memory<spm, tensor>>,
         memref<3xf32, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @select_row_broadcast
// CHECK: wafer.instr.gather_scatter %{{.*}} to %[[TRUE:[^ ]+]] {
// CHECK: wafer.instr.gather_scatter %{{.*}} to %[[DEST:[^ ]+]] {
// CHECK: wafer.instr.bit2fp %{{.*}} into %[[MASK:[^ ]+]] :
// CHECK: wafer.instr.mask_move %[[TRUE]], %[[MASK]] into %[[DEST]]
// CHECK: wafer.instr.ncc_join [0]
// CHECK-NOT: indexing_maps
// CHECK-NOT: wafer.instr.ncc_join [0]

func.func @constant_true_select_to_fresh_copy() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %true_value = memref.alloc()
      : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %false_value = memref.alloc()
      : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %true = arith.constant true
  %predicate = memref.alloc() : memref<2x3xi1, #wafer.memory<spm, tensor>>
  wafer.tile.fill %predicate, %true
      : memref<2x3xi1, #wafer.memory<spm, tensor>>, i1
  %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
      %predicate, %true_value, %false_value
      {indexing_maps = [#id2, #id2, #id2, #id2]}
      : (memref<2x3xi1, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @constant_true_select_to_fresh_copy
// CHECK: %[[TRUE_VALUE:.+]] = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[FALSE_VALUE:.+]] = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
// CHECK-NOT: arith.constant
// CHECK-NOT: memref<2x3xi1
// CHECK: %[[TRUE_COPY:.+]] = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[TRUE_VALUE]] to %[[TRUE_COPY]]
// CHECK-NOT: wafer.instr.fill
// CHECK-NOT: wafer.instr.bit2fp
// CHECK-NOT: wafer.instr.mask_move

func.func @constant_false_select_to_fresh_copy() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %true_value = memref.alloc()
      : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %false_value = memref.alloc()
      : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %false = arith.constant false
  %predicate = memref.alloc() : memref<2x3xi1, #wafer.memory<spm, tensor>>
  wafer.tile.fill %predicate, %false
      : memref<2x3xi1, #wafer.memory<spm, tensor>>, i1
  %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
      %predicate, %true_value, %false_value
      {indexing_maps = [#id2, #id2, #id2, #id2]}
      : (memref<2x3xi1, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @constant_false_select_to_fresh_copy
// CHECK: %[[TRUE_VALUE:.+]] = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[FALSE_VALUE:.+]] = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
// CHECK-NOT: arith.constant
// CHECK-NOT: memref<2x3xi1
// CHECK: %[[FALSE_COPY:.+]] = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[FALSE_VALUE]] to %[[FALSE_COPY]]
// CHECK-NOT: wafer.instr.fill
// CHECK-NOT: wafer.instr.bit2fp
// CHECK-NOT: wafer.instr.mask_move

func.func @dead_private_bool_fill_is_removed() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %true = arith.constant true
  %predicate = memref.alloc() : memref<2x3xi1, #wafer.memory<spm, tensor>>
  wafer.tile.fill %predicate, %true
      : memref<2x3xi1, #wafer.memory<spm, tensor>>, i1
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @dead_private_bool_fill_is_removed
// CHECK-NOT: wafer.instr.fill

func.func @shared_constant_predicate_stays_dynamic() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %true = arith.constant true
  %predicate = memref.alloc() : memref<2x3xi1, #wafer.memory<spm, tensor>>
  wafer.tile.fill %predicate, %true
      : memref<2x3xi1, #wafer.memory<spm, tensor>>, i1
  %true_value = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %false_value = memref.alloc() : memref<2x3xf32, #wafer.memory<spm, tensor>>
  %selected0 = wafer.tile.elementwise #wafer.elementwise_kind<select>
      %predicate, %true_value, %false_value
      {indexing_maps = [#id2, #id2, #id2, #id2]}
      : (memref<2x3xi1, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
  %selected1 = wafer.tile.elementwise #wafer.elementwise_kind<select>
      %predicate, %false_value, %true_value
      {indexing_maps = [#id2, #id2, #id2, #id2]}
      : (memref<2x3xi1, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @shared_constant_predicate_stays_dynamic
// CHECK: wafer.instr.fill
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.mask_move
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.mask_move

func.func @nonidentity_chosen_map_stays_dynamic() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %true = arith.constant true
  %predicate = memref.alloc() : memref<2x2xi1, #wafer.memory<spm, tensor>>
  wafer.tile.fill %predicate, %true
      : memref<2x2xi1, #wafer.memory<spm, tensor>>, i1
  %true_value = memref.alloc() : memref<2x2xf32, #wafer.memory<spm, tensor>>
  %false_value = memref.alloc() : memref<2x2xf32, #wafer.memory<spm, tensor>>
  %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
      %predicate, %true_value, %false_value
      {indexing_maps = [#id2, #transpose, #id2, #id2]}
      : (memref<2x2xi1, #wafer.memory<spm, tensor>>,
         memref<2x2xf32, #wafer.memory<spm, tensor>>,
         memref<2x2xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x2xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @nonidentity_chosen_map_stays_dynamic
// CHECK: wafer.instr.fill
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.mask_move
