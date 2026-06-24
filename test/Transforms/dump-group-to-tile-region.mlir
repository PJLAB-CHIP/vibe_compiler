// RUN: wafer-opt --wafer-dump-group-to-tile-region %s 2>&1 | FileCheck %s

module {
  func.func @matmul_bias_relu(
      %lhs: tensor<4x8xf32>,
      %rhs: tensor<8x16xf32>,
      %bias: tensor<16xf32>,
      %out: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %0 = wafer.group ins(%lhs, %rhs, %bias : tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>)
        outs(%out : tensor<4x16xf32>) {
    ^bb0(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>,
         %arg2: tensor<16xf32>, %arg3: tensor<4x16xf32>):
      %c0 = arith.constant 0.000000e+00 : f32
      %init = linalg.fill
          ins(%c0 : f32)
          outs(%arg3 : tensor<4x16xf32>) -> tensor<4x16xf32>
      %mm = linalg.matmul
          ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>)
          outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
      %biased = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%mm, %arg2 : tensor<4x16xf32>, tensor<16xf32>)
          outs(%arg3 : tensor<4x16xf32>) {
        ^bb0(%lhs_el: f32, %bias_el: f32, %out_el: f32):
          %sum = arith.addf %lhs_el, %bias_el : f32
          linalg.yield %sum : f32
        } -> tensor<4x16xf32>
      %zero = arith.constant dense<0.000000e+00> : tensor<4x16xf32>
      %relu = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%biased, %zero : tensor<4x16xf32>, tensor<4x16xf32>)
          outs(%arg3 : tensor<4x16xf32>) {
        ^bb0(%value: f32, %zero_el: f32, %out_el: f32):
          %max = arith.maximumf %value, %zero_el : f32
          linalg.yield %max : f32
        } -> tensor<4x16xf32>
      wafer.group.yield %relu : tensor<4x16xf32>
    } : tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }

  func.func @two_independent_groups(
      %a0: tensor<4xf32>, %b0: tensor<4xf32>, %out0: tensor<4xf32>,
      %a1: tensor<8xf32>, %b1: tensor<8xf32>, %out1: tensor<8xf32>)
      -> (tensor<4xf32>, tensor<8xf32>) {
    %0 = wafer.group ins(%a0, %b0 : tensor<4xf32>, tensor<4xf32>)
        outs(%out0 : tensor<4xf32>) {
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
    %1 = wafer.group ins(%a1, %b1 : tensor<8xf32>, tensor<8xf32>)
        outs(%out1 : tensor<8xf32>) {
    ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>):
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
          outs(%arg2 : tensor<8xf32>) {
        ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
          %add = arith.addf %lhs_el, %rhs_el : f32
          linalg.yield %add : f32
        } -> tensor<8xf32>
      wafer.group.yield %sum : tensor<8xf32>
    } : tensor<8xf32>
    return %0, %1 : tensor<4xf32>, tensor<8xf32>
  }

  func.func @empty_fill_group(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
      %empty = tensor.empty() : tensor<4xf32>
      %c0 = arith.constant 0.000000e+00 : f32
      %fill = linalg.fill
          ins(%c0 : f32)
          outs(%empty : tensor<4xf32>) -> tensor<4xf32>
      wafer.group.yield %fill : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @reduce_sum_group(%input: tensor<2x4xf32>, %init: tensor<f32>,
                              %out: tensor<2xf32>) -> tensor<2xf32> {
    %0 = wafer.group ins(%input, %init : tensor<2x4xf32>, tensor<f32>)
        outs(%out : tensor<2xf32>) {
    ^bb0(%arg0: tensor<2x4xf32>, %arg1: tensor<f32>, %arg2: tensor<2xf32>):
      %init_scalar = tensor.extract %arg1[] : tensor<f32>
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
    return %0 : tensor<2xf32>
  }

  func.func @tensor_movement_group(%input: tensor<8xf32>,
                                   %patch: tensor<4xf32>,
                                   %out: tensor<8xf32>) -> tensor<8xf32> {
    %0 = wafer.group ins(%input, %patch : tensor<8xf32>, tensor<4xf32>)
        outs(%out : tensor<8xf32>) {
    ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>, %arg2: tensor<8xf32>):
      %slice = tensor.extract_slice %arg0[2] [4] [1]
          : tensor<8xf32> to tensor<4xf32>
      %inserted = tensor.insert_slice %slice into %arg2[2] [4] [1]
          : tensor<4xf32> into tensor<8xf32>
      wafer.group.yield %inserted : tensor<8xf32>
    } : tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @tensor_rank_reduced_slice_group(%input: tensor<2x4xf32>,
                                             %out: tensor<2x4xf32>)
      -> tensor<2x4xf32> {
    %0 = wafer.group ins(%input : tensor<2x4xf32>) outs(%out : tensor<2x4xf32>) {
    ^bb0(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>):
      %row = tensor.extract_slice %arg0[0, 0] [1, 4] [1, 1]
          : tensor<2x4xf32> to tensor<4xf32>
      %inserted = tensor.insert_slice %row into %arg1[1, 0] [1, 4] [1, 1]
          : tensor<4xf32> into tensor<2x4xf32>
      wafer.group.yield %inserted : tensor<2x4xf32>
    } : tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }

  func.func @tensor_reshape_group(%input: tensor<6xf32>,
                                  %out: tensor<6xf32>) -> tensor<6xf32> {
    %0 = wafer.group ins(%input : tensor<6xf32>) outs(%out : tensor<6xf32>) {
    ^bb0(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>):
      %expanded = tensor.expand_shape %arg0 [[0, 1]]
          output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>
      %collapsed = tensor.collapse_shape %expanded [[0, 1]]
          : tensor<2x3xf32> into tensor<6xf32>
      wafer.group.yield %collapsed : tensor<6xf32>
    } : tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @passthrough_movement_group(%input: tensor<2x3xf32>,
                                        %out: tensor<3x2xf32>)
      -> tensor<3x2xf32> {
    %0 = wafer.group ins(%input : tensor<2x3xf32>) outs(%out : tensor<3x2xf32>) {
    ^bb0(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>):
      %transposed = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d1, d0)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg0 : tensor<2x3xf32>)
          outs(%arg1 : tensor<3x2xf32>) {
        ^bb0(%value: f32, %out_el: f32):
          linalg.yield %value : f32
        } -> tensor<3x2xf32>
      wafer.group.yield %transposed : tensor<3x2xf32>
    } : tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }

  func.func @collective_group(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
      %ar = wafer.linalg_ext.collective.all_reduce
          ins(%arg0 : tensor<4xf32>)
          outs(%arg1 : tensor<4xf32>)
          {
          ^bb0(%lhs: f32, %rhs: f32):
            %sum = arith.addf %lhs, %rhs : f32
            wafer.linalg_ext.collective.yield %sum : f32
          } {rank_group = array<i64: 0, 1>} -> tensor<4xf32>
      wafer.group.yield %ar : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @collective_all_gather_group(%input: tensor<4xf32>,
                                         %out: tensor<8xf32>)
      -> tensor<8xf32> {
    %0 = wafer.group ins(%input : tensor<4xf32>) outs(%out : tensor<8xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<8xf32>):
      %ag = wafer.linalg_ext.collective.all_gather
          ins(%arg0 : tensor<4xf32>)
          outs(%arg1 : tensor<8xf32>)
          {axis = 0 : i64, rank_group = array<i64: 0, 1>}
          -> tensor<8xf32>
      wafer.group.yield %ag : tensor<8xf32>
    } : tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @collective_reduce_scatter_group(%input: tensor<8xf32>,
                                             %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<8xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>):
      %rs = wafer.linalg_ext.collective.reduce_scatter
          ins(%arg0 : tensor<8xf32>)
          outs(%arg1 : tensor<4xf32>)
          {
          ^bb0(%lhs: f32, %rhs: f32):
            %sum = arith.addf %lhs, %rhs : f32
            wafer.linalg_ext.collective.yield %sum : f32
          } {axis = 0 : i64, rank_group = array<i64: 0, 1>} -> tensor<4xf32>
      wafer.group.yield %rs : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @static_slice_support_op(%input: tensor<8xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<8xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>):
      %slice = tensor.extract_slice %arg0[0] [4] [1]
          : tensor<8xf32> to tensor<4xf32>
      wafer.group.yield %slice : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: wafer.group_to_tile_region group @matmul_bias_relu#0
// CHECK: bufferization.to_memref
// CHECK-SAME: #wafer.memory<ddr, tensor>
// CHECK: wafer.tile.region(
// CHECK-SAME: #wafer.memory<ddr, tensor>
// CHECK: wafer.tile.load
// CHECK-SAME: #wafer.memory<ddr, tensor>
// CHECK: wafer.tile.fill
// CHECK: wafer.tile.load
// CHECK: memref<4x8xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.materialize_layout
// CHECK: #wafer.memory<spm, cx>
// CHECK: wafer.tile.gemm
// CHECK: wafer.tile.materialize_layout
// CHECK: #wafer.memory<spm, tensor>
// CHECK: wafer.tile.elementwise <add>
// CHECK-SAME: indexing_maps
// CHECK: wafer.tile.elementwise <max>
// CHECK: wafer.tile.store
// CHECK-SAME: #wafer.memory<ddr, tensor>
// CHECK: wafer.tile.yield
// CHECK: bufferization.to_tensor
// CHECK-LABEL: wafer.group_to_tile_region group @two_independent_groups#0
// CHECK: wafer.tile.elementwise <add>
// CHECK-LABEL: wafer.group_to_tile_region group @two_independent_groups#1
// CHECK: wafer.tile.elementwise <add>
// CHECK-LABEL: wafer.group_to_tile_region group @empty_fill_group#0
// CHECK: memref.alloc
// CHECK: wafer.tile.fill
// CHECK: wafer.tile.store
// CHECK-LABEL: wafer.group_to_tile_region group @reduce_sum_group#0
// CHECK: memref.load
// CHECK-SAME: #wafer.memory<ddr, tensor>
// CHECK: wafer.tile.fill
// CHECK: wafer.tile.materialize_layout
// CHECK: #wafer.memory<spm, cx>
// CHECK: wafer.tile.reduce <sum>
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK: wafer.tile.materialize_layout
// CHECK: #wafer.memory<spm, tensor>
// CHECK: wafer.tile.store
// CHECK-LABEL: wafer.group_to_tile_region group @tensor_movement_group#0
// CHECK: memref.subview
// CHECK-SAME: [2] [4] [1]
// CHECK-SAME: memref<4xf32, strided<[1], offset: 2>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.load
// CHECK-SAME: memref<4xf32, strided<[1], offset: 2>, #wafer.memory<ddr, tensor>>
// CHECK: memref.subview
// CHECK-SAME: [2] [4] [1]
// CHECK-SAME: memref<4xf32, strided<[1], offset: 2>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.store
// CHECK-SAME: memref<4xf32, strided<[1], offset: 2>, #wafer.memory<ddr, tensor>>
// CHECK-LABEL: wafer.group_to_tile_region group @tensor_rank_reduced_slice_group#0
// CHECK: memref.subview
// CHECK-SAME: [0, 0] [1, 4] [1, 1]
// CHECK-SAME: memref<4xf32, strided<[1]>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.load
// CHECK-SAME: memref<4xf32, strided<[1]>, #wafer.memory<ddr, tensor>>
// CHECK: memref.subview
// CHECK-SAME: [1, 0] [1, 4] [1, 1]
// CHECK-SAME: memref<4xf32, strided<[1], offset: 4>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.store
// CHECK-SAME: memref<4xf32, strided<[1], offset: 4>, #wafer.memory<ddr, tensor>>
// CHECK-LABEL: wafer.group_to_tile_region group @tensor_reshape_group#0
// CHECK: wafer.tile.reshape
// CHECK: wafer.tile.reshape
// CHECK: wafer.tile.store
// CHECK-LABEL: wafer.group_to_tile_region group @passthrough_movement_group#0
// CHECK: wafer.tile.transpose
// CHECK-SAME: permutation = array<i64: 1, 0>
// CHECK: wafer.tile.store
// CHECK-LABEL: wafer.group_to_tile_region group @collective_group#0
// CHECK: wafer.tile.load
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK: memref.alloc
// CHECK-SAME: memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.all_reduce <sum>
// CHECK-SAME: bytes = 16 : i64
// CHECK-SAME: group_size = 2 : i64
// CHECK-SAME: local_rank = 0 : i64
// CHECK-SAME: rank_group = array<i64: 0, 1>
// CHECK: wafer.tile.store
// CHECK-LABEL: wafer.group_to_tile_region group @collective_all_gather_group#0
// CHECK: wafer.tile.load
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK: memref.alloc
// CHECK-SAME: memref<8xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.all_gather
// CHECK-SAME: bytes = 16 : i64
// CHECK-SAME: group_size = 2 : i64
// CHECK-SAME: local_rank = 0 : i64
// CHECK-SAME: rank_group = array<i64: 0, 1>
// CHECK: wafer.tile.store
// CHECK-LABEL: wafer.group_to_tile_region group @collective_reduce_scatter_group#0
// CHECK: wafer.tile.load
// CHECK-SAME: memref<8xf32, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.extract_slice
// CHECK-SAME: offsets = array<i64: 0>
// CHECK-SAME: sizes = array<i64: 4>
// CHECK: memref.alloc
// CHECK-SAME: memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.reduce_scatter <sum>
// CHECK-SAME: bytes = 16 : i64
// CHECK-SAME: group_size = 2 : i64
// CHECK-SAME: local_rank = 0 : i64
// CHECK-SAME: rank_group = array<i64: 0, 1>
// CHECK: wafer.tile.store
// CHECK-LABEL: wafer.group_to_tile_region group @static_slice_support_op#0
// CHECK: memref.subview
// CHECK-SAME: [0] [4] [1]
// CHECK-SAME: memref<4xf32, strided<[1]>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.load
// CHECK-SAME: memref<4xf32, strided<[1]>, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.store
