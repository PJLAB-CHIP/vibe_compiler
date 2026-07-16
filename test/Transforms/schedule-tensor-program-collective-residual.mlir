// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %s | FileCheck %s --check-prefixes=CHECK,RESHAPE

// This is the tensor-parallel output-projection boundary of a Llama-2 7B
// block: sequence 16, hidden size 4096 and TP16 all-reduce.  The collective
// result and residual add must remain in one scheduling scope.  The final DTE
// round completes before the local reduction completes, then the residual add
// consumes the exact same SPM accumulator.  Only the two external inputs are
// loaded and only the final result is stored; no collective intermediate is
// materialized in DDR.
func.func @tp16_all_reduce_feeds_residual_in_spm(
    %projection: tensor<16x4096xf16>,
    %residual: tensor<16x4096xf16>) -> tensor<16x4096xf16> {
  %collective_empty = tensor.empty() : tensor<16x4096xf16>
  %collective = wafer.linalg_ext.collective.all_reduce
      ins(%projection : tensor<16x4096xf16>)
      outs(%collective_empty : tensor<16x4096xf16>) {
  ^bb0(%lhs: f16, %rhs: f16):
    %sum = arith.addf %lhs, %rhs : f16
    wafer.linalg_ext.collective.yield %sum : f16
  } {channel_id = 29 : i64,
     rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                             8, 9, 10, 11, 12, 13, 14, 15>}
      -> tensor<16x4096xf16>

  %result_empty = tensor.empty() : tensor<16x4096xf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%collective, %residual
          : tensor<16x4096xf16>, tensor<16x4096xf16>)
      outs(%result_empty : tensor<16x4096xf16>) {
  ^bb0(%reduced: f16, %skip: f16, %old: f16):
    %sum = arith.addf %reduced, %skip : f16
    linalg.yield %sum : f16
  } -> tensor<16x4096xf16>
  return %result : tensor<16x4096xf16>
}

// A static shape-only view must not hide the local epilogue from scope
// discovery. This is the form produced when a rank-local 2-D projection is
// restored to the leading batch dimension before the residual add.
func.func @tp16_all_reduce_reshape_feeds_residual_in_spm(
    %projection: tensor<16x4096xf16>,
    %residual: tensor<1x16x4096xf16>) -> tensor<1x16x4096xf16> {
  %collective_empty = tensor.empty() : tensor<16x4096xf16>
  %collective = wafer.linalg_ext.collective.all_reduce
      ins(%projection : tensor<16x4096xf16>)
      outs(%collective_empty : tensor<16x4096xf16>) {
  ^bb0(%lhs: f16, %rhs: f16):
    %sum = arith.addf %lhs, %rhs : f16
    wafer.linalg_ext.collective.yield %sum : f16
  } {channel_id = 30 : i64,
     rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                             8, 9, 10, 11, 12, 13, 14, 15>}
      -> tensor<16x4096xf16>
  %reshaped = tensor.expand_shape %collective [[0, 1], [2]]
      output_shape [1, 16, 4096]
      : tensor<16x4096xf16> into tensor<1x16x4096xf16>
  %result_empty = tensor.empty() : tensor<1x16x4096xf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%reshaped, %residual
          : tensor<1x16x4096xf16>, tensor<1x16x4096xf16>)
      outs(%result_empty : tensor<1x16x4096xf16>) {
  ^bb0(%reduced: f16, %skip: f16, %old: f16):
    %sum = arith.addf %reduced, %skip : f16
    linalg.yield %sum : f16
  } -> tensor<1x16x4096xf16>
  return %result : tensor<1x16x4096xf16>
}

// CHECK-LABEL: func.func @tp16_all_reduce_feeds_residual_in_spm
// CHECK-COUNT-1: wafer.tile.region
// CHECK: wafer.instr.rdma {{.*}} to %[[PROJECTION:[^ ]+]]
// CHECK-COUNT-15: wafer.instr.dte_wait
// CHECK: wafer.instr.elementwise <add> %[[ACCUMULATOR:[^,]+]], {{[^ ]+}} into %[[ACCUMULATOR]]
// CHECK-NEXT: wafer.instr.local_fence
// CHECK: %[[RESIDUAL:[^ ]+]] = memref.alloc() {{.*}} : memref<16x4096xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.rdma {{.*}} to %[[RESIDUAL]]
// CHECK: wafer.instr.elementwise <add> %[[ACCUMULATOR]], %[[RESIDUAL]] into %[[RESULT:[^ ]+]]
// CHECK-NOT: wafer.instr.rdma
// CHECK: wafer.instr.wdma %[[RESULT]] to
// CHECK-NOT: wafer.instr.rdma
// CHECK-NOT: wafer.instr.wdma
// CHECK: wafer.tile.yield
// CHECK-NOT: wafer.tile.region

// RESHAPE-LABEL: func.func @tp16_all_reduce_reshape_feeds_residual_in_spm
// RESHAPE-COUNT-1: wafer.tile.region
// RESHAPE-COUNT-15: wafer.instr.dte_wait
// RESHAPE: wafer.instr.elementwise <add> %[[RESHAPED_ACC:[^,]+]], {{[^ ]+}} into %[[RESHAPED_ACC]]
// RESHAPE-NEXT: wafer.instr.local_fence
// RESHAPE: %[[RESHAPED_VIEW:[^ ]+]] = memref.reinterpret_cast %[[RESHAPED_ACC]]
// RESHAPE: %[[RESHAPED_RESIDUAL:[^ ]+]] = memref.alloc() {{.*}} : memref<1x16x4096xf16, #wafer.memory<spm, tensor>>
// RESHAPE-NEXT: wafer.instr.rdma {{.*}} to %[[RESHAPED_RESIDUAL]]
// RESHAPE: wafer.instr.elementwise <add> %[[RESHAPED_VIEW]], %[[RESHAPED_RESIDUAL]] into %[[RESHAPED_RESULT:[^ ]+]]
// RESHAPE-NOT: wafer.instr.wdma
// RESHAPE-COUNT-1: wafer.instr.wdma %[[RESHAPED_RESULT]] to
// RESHAPE-NOT: wafer.instr.wdma
// RESHAPE: wafer.tile.yield
// RESHAPE-NOT: wafer.tile.region
