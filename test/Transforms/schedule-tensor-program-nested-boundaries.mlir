// RUN: split-file %s %t
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %t/nested-scalar-capture.mlir | FileCheck %s --check-prefix=CAPTURE --implicit-check-not=linalg.
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %t/static-external-slice.mlir | FileCheck %s --check-prefix=SLICE --implicit-check-not=linalg. --implicit-check-not=tensor.extract_slice
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %t/unsupported-linalg-index.mlir 2>&1 | FileCheck %s --check-prefix=INDEX

//--- nested-scalar-capture.mlir

#id = affine_map<(d0) -> (d0)>

module {
  func.func @nested_scalar_capture(
      %input: tensor<4xf32>, %bias: f32,
      %out: tensor<4xf32>) -> tensor<4xf32> {
    %result = linalg.generic {
        indexing_maps = [#id, #id], iterator_types = ["parallel"]
      } ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %sum = arith.addf %value, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CAPTURE-LABEL: func.func @nested_scalar_capture(
// CAPTURE-SAME: %[[INPUT:[^:]+]]: tensor<4xf32>, %[[BIAS:[^:]+]]: f32,
// CAPTURE: %[[INPUT_MEM:.+]] = bufferization.to_memref %[[INPUT]] read_only
// CAPTURE: wafer.tile.region(%[[BIAS]], %[[INPUT_MEM]],
// CAPTURE: ^bb0(%[[REGION_BIAS:[^:]+]]: f32,
// CAPTURE: wafer.instr.fill {{.+}}, %[[REGION_BIAS]]
// CAPTURE: wafer.instr.elementwise <add>

//--- static-external-slice.mlir

#id = affine_map<(d0) -> (d0)>

module {
  func.func @static_external_slice(
      %input: tensor<8xf32>, %out: tensor<4xf32>) -> tensor<4xf32> {
    %slice = tensor.extract_slice %input[2] [4] [1]
        : tensor<8xf32> to tensor<4xf32>
    %result = linalg.generic {
        indexing_maps = [#id, #id], iterator_types = ["parallel"]
      } ins(%slice : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      %negated = arith.negf %value : f32
      linalg.yield %negated : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// SLICE-LABEL: func.func @static_external_slice(
// SLICE: %[[INPUT_MEM:.+]] = bufferization.to_memref %{{.+}} read_only
// SLICE: wafer.tile.region(%[[INPUT_MEM]],
// SLICE: ^bb0(%[[REGION_INPUT:[^:]+]]: memref<8xf32, #wafer.memory<ddr, tensor>>,
// SLICE: %[[SUBVIEW:.+]] = memref.subview %[[REGION_INPUT]][2] [4] [1]
// SLICE-SAME: to memref<4xf32, strided<[1], offset: 2>, #wafer.memory<ddr, tensor>>
// SLICE: wafer.instr.rdma %[[SUBVIEW]]
// SLICE: wafer.instr.elementwise <neg>

//--- unsupported-linalg-index.mlir

#id = affine_map<(d0) -> (d0)>

module {
  func.func @unsupported_linalg_index(
      %input: tensor<4xi32>, %out: tensor<4xi32>) -> tensor<4xi32> {
    %result = linalg.generic {
        indexing_maps = [#id, #id], iterator_types = ["parallel"]
      } ins(%input : tensor<4xi32>) outs(%out : tensor<4xi32>) {
    ^bb0(%value: i32, %old: i32):
      %index = linalg.index 0 : index
      %index_i32 = arith.index_cast %index : index to i32
      %sum = arith.addi %value, %index_i32 : i32
      linalg.yield %sum : i32
    } -> tensor<4xi32>
    return %result : tensor<4xi32>
  }
}

// INDEX: error: no_complete_rank_variant: bounded scheduling search found no fully legal rank artifact
// INDEX: complete-tile-region: unsupported linalg.generic body op linalg.index
// INDEX-NOT: wafer.tile.region
