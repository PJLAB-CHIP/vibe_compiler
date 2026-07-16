// RUN: split-file %s %t
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %t/canonical-two-way-concat.mlir | FileCheck %s --check-prefix=CONCAT --implicit-check-not=linalg. --implicit-check-not=tensor.
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %t/rank-reduced-static-slices.mlir | FileCheck %s --check-prefix=RANK-REDUCED --implicit-check-not=linalg. --implicit-check-not=tensor.

//--- canonical-two-way-concat.mlir

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @canonical_two_way_concat(
      %first: tensor<2x2xf32>, %second: tensor<2x2xf32>,
      %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c2 = arith.constant 2 : index
    %result = linalg.generic {
        indexing_maps = [#id2],
        iterator_types = ["parallel", "parallel"]
      } outs(%out : tensor<2x4xf32>) {
    ^bb0(%old: f32):
      %row = linalg.index 0 : index
      %column = linalg.index 1 : index
      %in_first = arith.cmpi ult, %column, %c2 : index
      %value = scf.if %in_first -> f32 {
        %first_value = tensor.extract %first[%row, %column]
            : tensor<2x2xf32>
        scf.yield %first_value : f32
      } else {
        %second_column = arith.subi %column, %c2 : index
        %second_value = tensor.extract %second[%row, %second_column]
            : tensor<2x2xf32>
        scf.yield %second_value : f32
      }
      linalg.yield %value : f32
    } -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// CONCAT-LABEL: func.func @canonical_two_way_concat(
// CONCAT: %[[FIRST:.+]] = bufferization.to_memref %{{.+}} read_only
// CONCAT: %[[SECOND:.+]] = bufferization.to_memref %{{.+}} read_only
// CONCAT: wafer.tile.region(%[[FIRST]], %[[SECOND]],
// CONCAT-COUNT-2: wafer.instr.rdma
// CONCAT: wafer.instr.fill
// CONCAT: wafer.instr.gather_scatter {{.+}} {byte_count = 16 : i64,
// CONCAT-SAME: dst_iterations = array<i64: 2, 1, 1>
// CONCAT-SAME: dst_strides = array<i64: 16, 0, 0>
// CONCAT-SAME: inner_bytes = 8 : i64
// CONCAT: wafer.instr.gather_scatter {{.+}} {byte_count = 16 : i64,
// CONCAT-SAME: dst_iterations = array<i64: 2, 1, 1>
// CONCAT-SAME: dst_offset = 8 : i64
// CONCAT-SAME: dst_strides = array<i64: 16, 0, 0>
// CONCAT-SAME: inner_bytes = 8 : i64
// CONCAT: wafer.instr.wdma
// CONCAT-NOT: wafer.instr.elementwise
// CONCAT: return

//--- rank-reduced-static-slices.mlir

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @rank_reduced_static_slices(
      %input: tensor<2x4xf32>, %out: tensor<2x4xf32>)
      -> tensor<2x4xf32> {
    %row = tensor.extract_slice %input[0, 0] [1, 4] [1, 1]
        : tensor<2x4xf32> to tensor<4xf32>
    // The destination is the source program's writable output boundary.  The
    // following root reads the updated value, so the insert cannot be dropped
    // as an unread DPS init.
    %inserted = tensor.insert_slice %row into %out[1, 0] [1, 4] [1, 1]
        : tensor<4xf32> into tensor<2x4xf32>
    %result = linalg.generic {
        indexing_maps = [#id2, #id2],
        iterator_types = ["parallel", "parallel"]
      } ins(%inserted : tensor<2x4xf32>)
        outs(%out : tensor<2x4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %negated = arith.negf %value : f32
      linalg.yield %negated : f32
    } -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// RANK-REDUCED-LABEL: func.func @rank_reduced_static_slices(
// RANK-REDUCED: %[[INPUT:.+]] = bufferization.to_memref %{{.+}} read_only
// RANK-REDUCED: %[[OUTPUT:.+]] = bufferization.to_memref %{{.+}} : memref<2x4xf32, #wafer.memory<ddr, tensor>>
// RANK-REDUCED: wafer.tile.region(%[[INPUT]], %[[OUTPUT]]
// RANK-REDUCED: ^bb0(%[[REGION_INPUT:[^:]+]]: memref<2x4xf32, #wafer.memory<ddr, tensor>>, %[[REGION_OUTPUT:[^:]+]]: memref<2x4xf32, #wafer.memory<ddr, tensor>>):
// RANK-REDUCED: %[[ROW_VIEW:.+]] = memref.subview {{.+}}[0, 0] [1, 4] [1, 1]
// RANK-REDUCED-SAME: to memref<4xf32, strided<[1]>, #wafer.memory<ddr, tensor>>
// RANK-REDUCED: %[[ROW:.+]] = memref.alloc() {{.*}} : memref<4xf32, #wafer.memory<spm, tensor>>
// RANK-REDUCED: wafer.instr.rdma %[[ROW_VIEW]] to %[[ROW]] {byte_count = 16 : i64,
// RANK-REDUCED: %[[OLD_OUTPUT:.+]] = memref.alloc() {{.*}} : memref<2x4xf32, #wafer.memory<spm, tensor>>
// RANK-REDUCED: wafer.instr.rdma {{.+}} to %[[OLD_OUTPUT]] {byte_count = 32 : i64,
// RANK-REDUCED: %[[UPDATED:.+]] = memref.alloc() {{.*}} : memref<2x4xf32, #wafer.memory<spm, tensor>>
// RANK-REDUCED: wafer.instr.gather_scatter %[[OLD_OUTPUT]] to %[[UPDATED]] {byte_count = 32 : i64,
// RANK-REDUCED: wafer.instr.gather_scatter %[[ROW]] to %[[UPDATED]] {byte_count = 16 : i64,
// RANK-REDUCED-SAME: dst_offset = 16 : i64
// RANK-REDUCED-SAME: inner_bytes = 16 : i64
// RANK-REDUCED: wafer.instr.elementwise <neg> %[[UPDATED]]
// RANK-REDUCED: wafer.instr.wdma {{.+}} to %[[REGION_OUTPUT]] {byte_count = 32 : i64,
// RANK-REDUCED: return
