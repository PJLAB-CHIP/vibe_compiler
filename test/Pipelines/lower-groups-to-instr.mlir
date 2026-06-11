// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-instr)' %s | FileCheck %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-memory-planned-instr)' %s | FileCheck --check-prefix=PLANNED %s

func.func @add_group(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
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

func.func @two_chained_groups(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                              %bias: tensor<4xf32>, %out: tensor<4xf32>)
    -> tensor<4xf32> {
  %tmp = tensor.empty() : tensor<4xf32>
  %first = wafer.group ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%tmp : tensor<4xf32>) {
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

  %second = wafer.group ins(%first, %bias : tensor<4xf32>, tensor<4xf32>)
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
  return %second : tensor<4xf32>
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

func.func @boundary_slice_group(%input: tensor<4x8xf16>,
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

func.func @boundary_tiled_matmul_group(%lhs: tensor<4x8xf16>,
                                       %rhs: tensor<8x8xf16>,
                                       %out: tensor<4x8xf16>)
    -> tensor<4x8xf16> {
  %group = wafer.group ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x8xf16>)
      outs(%out : tensor<4x8xf16>) {
  ^bb0(%arg0: tensor<4x8xf16>, %arg1: tensor<8x8xf16>,
       %arg2: tensor<4x8xf16>):
    %lhs_tile = tensor.extract_slice %arg0[1, 0] [2, 4] [1, 1]
        : tensor<4x8xf16> to tensor<2x4xf16>
    %rhs_tile = tensor.extract_slice %arg1[0, 2] [4, 3] [1, 1]
        : tensor<8x8xf16> to tensor<4x3xf16>
    %empty = tensor.empty() : tensor<2x3xf16>
    %c0 = arith.constant 0.000000e+00 : f16
    %init = linalg.fill ins(%c0 : f16)
        outs(%empty : tensor<2x3xf16>) -> tensor<2x3xf16>
    %mm = linalg.matmul
        ins(%lhs_tile, %rhs_tile : tensor<2x4xf16>, tensor<4x3xf16>)
        outs(%init : tensor<2x3xf16>) -> tensor<2x3xf16>
    %updated = tensor.insert_slice %mm into %arg2[1, 2] [2, 3] [1, 1]
        : tensor<2x3xf16> into tensor<4x8xf16>
    wafer.group.yield %updated : tensor<4x8xf16>
  } : tensor<4x8xf16>
  return %group : tensor<4x8xf16>
}

// CHECK-LABEL: func.func @add_group
// CHECK-SAME: (%{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>)
// CHECK-SAME: -> memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK-NOT: wafer.tile.load
// CHECK-NOT: wafer.tile.elementwise
// CHECK-NOT: wafer.tile.store
// CHECK: wafer.tile.region
// CHECK: wafer.instr.rdma
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.wdma

// CHECK-LABEL: func.func @if_group
// CHECK-SAME: (%{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4xf32, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: i1)
// CHECK-SAME: -> memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK-NOT: wafer.tile.load
// CHECK-NOT: wafer.tile.elementwise
// CHECK-NOT: wafer.tile.store
// CHECK: wafer.tile.region
// CHECK: scf.if
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.instr.wdma

// CHECK-LABEL: func.func @two_chained_groups
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK-NOT: wafer.tile.elementwise
// CHECK: wafer.tile.region
// CHECK: wafer.instr.elementwise <add>
// CHECK: wafer.tile.region
// CHECK: wafer.instr.elementwise <add>
// CHECK: return {{%.*}} : memref<4xf32, #wafer.memory<ddr, tensor>>

// CHECK-LABEL: func.func @loop_group
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK-NOT: wafer.tile.elementwise
// CHECK: wafer.tile.region
// CHECK: scf.for
// CHECK: wafer.instr.elementwise <add>
// CHECK: scf.yield {{%.*}} : memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.wdma

// CHECK-LABEL: func.func @boundary_slice_group
// CHECK-SAME: (%{{[^:]+}}: memref<4x8xf16, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4x8xf16, #wafer.memory<ddr, tensor>>)
// CHECK-SAME: -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.generic
// CHECK-NOT: wafer.tile.load
// CHECK-NOT: wafer.tile.store
// CHECK: %[[INPUT_TILE:.+]] = memref.subview
// CHECK-SAME: [1, 2] [2, 3] [1, 1]
// CHECK-SAME: memref<4x8xf16, #wafer.memory<ddr, tensor>>
// CHECK-SAME: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
// CHECK: %[[LOAD_DST:.+]] = memref.alloc() : memref<2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.rdma %[[INPUT_TILE]] to %[[LOAD_DST]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: inner_bytes = 6 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: src_strides = array<i64: 16, 0, 0>
// CHECK: wafer.instr.elementwise <add> %[[LOAD_DST]], %[[LOAD_DST]]
// CHECK: %[[OUTPUT_TILE:.+]] = memref.subview
// CHECK-SAME: [1, 2] [2, 3] [1, 1]
// CHECK-SAME: memref<4x8xf16, #wafer.memory<ddr, tensor>>
// CHECK-SAME: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.instr.wdma {{%.*}} to %[[OUTPUT_TILE]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 16, 0, 0>
// CHECK-SAME: inner_bytes = 6 : i64

// CHECK-LABEL: func.func @boundary_tiled_matmul_group
// CHECK-SAME: (%{{[^:]+}}: memref<4x8xf16, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<8x8xf16, #wafer.memory<ddr, tensor>>, %{{[^:]+}}: memref<4x8xf16, #wafer.memory<ddr, tensor>>)
// CHECK-SAME: -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.group
// CHECK-NOT: linalg.matmul
// CHECK-NOT: wafer.tile.load
// CHECK-NOT: wafer.tile.store
// CHECK: %[[LHS_TILE:.+]] = memref.subview
// CHECK-SAME: [1, 0] [2, 4] [1, 1]
// CHECK-SAME: memref<2x4xf16, strided<[8, 1], offset: 8>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.instr.rdma %[[LHS_TILE]]
// CHECK-SAME: byte_count = 16 : i64
// CHECK-SAME: inner_bytes = 8 : i64
// CHECK-SAME: src_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: src_strides = array<i64: 16, 0, 0>
// CHECK: %[[RHS_TILE:.+]] = memref.subview
// CHECK-SAME: [0, 2] [4, 3] [1, 1]
// CHECK-SAME: memref<4x3xf16, strided<[8, 1], offset: 2>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.instr.rdma %[[RHS_TILE]]
// CHECK-SAME: byte_count = 24 : i64
// CHECK-SAME: inner_bytes = 6 : i64
// CHECK-SAME: src_iterations = array<i64: 4, 1, 1>
// CHECK-SAME: src_strides = array<i64: 16, 0, 0>
// CHECK: wafer.instr.gemm
// CHECK-SAME: k = 4 : i64
// CHECK-SAME: m = 2 : i64
// CHECK-SAME: n = 3 : i64
// CHECK: %[[MATMUL_OUT_TILE:.+]] = memref.subview
// CHECK-SAME: [1, 2] [2, 3] [1, 1]
// CHECK-SAME: memref<2x3xf16, strided<[8, 1], offset: 10>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.instr.wdma {{%.*}} to %[[MATMUL_OUT_TILE]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-SAME: dst_iterations = array<i64: 2, 1, 1>
// CHECK-SAME: dst_strides = array<i64: 16, 0, 0>
// CHECK-SAME: inner_bytes = 6 : i64

// PLANNED-LABEL: func.func @boundary_slice_group
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x3xf16, #wafer.memory<spm, tensor>>
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<2x3xf16, #wafer.memory<spm, tensor>>
// PLANNED: wafer.instr.wdma

// PLANNED-LABEL: func.func @boundary_tiled_matmul_group
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<2x4xf16, #wafer.memory<spm, tensor>>
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>} : memref<4x3xf16, #wafer.memory<spm, tensor>>
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x3xf16, #wafer.memory<spm, tensor>>
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x4xf16, #wafer.memory<spm, cx>>
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<4x3xf16, #wafer.memory<spm, cx>>
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>} : memref<2x3xf16, #wafer.memory<spm, cx>>
// PLANNED: memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x3xf16, #wafer.memory<spm, tensor>>
// PLANNED: wafer.instr.wdma
