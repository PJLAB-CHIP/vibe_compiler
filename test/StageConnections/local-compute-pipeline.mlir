// RUN: wafer-opt --wafer-form-groups %s | FileCheck %s --check-prefix=GROUP --implicit-check-not=unrealized_conversion_cast
// RUN: wafer-opt --wafer-form-groups --wafer-materialize-single-tile %s | FileCheck %s --check-prefix=TILE --implicit-check-not=unrealized_conversion_cast
// RUN: wafer-opt --wafer-form-groups --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-tile-region-to-c-abi %s | FileCheck %s --check-prefix=ABI --implicit-check-not=unrealized_conversion_cast

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @single_matmul(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%out : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }

  func.func @same_shape_elementwise(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<4x8xf16>,
      %out: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %0 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<4x8xf16>)
        outs(%out : tensor<4x8xf16>) {
      ^bb0(%lhs_s: f16, %rhs_s: f16, %out_s: f16):
        %sum = arith.addf %lhs_s, %rhs_s : f16
        linalg.yield %sum : f16
    } -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }

  func.func @row_sum_reduce(
      %input: tensor<4x8xf16>) -> tensor<4xf16> {
    %zero = arith.constant 0.000000e+00 : f16
    %empty = tensor.empty() : tensor<4xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<4xf16>) -> tensor<4xf16>
    %0 = linalg.reduce { arith.addf }
        ins(%input : tensor<4x8xf16>)
        outs(%init : tensor<4xf16>)
        dimensions = [1]
    return %0 : tensor<4xf16>
  }
}

// GROUP-LABEL: func.func @single_matmul(
// GROUP: %[[MM_GROUP:.+]] = wafer.group
// GROUP: linalg.matmul
// GROUP: wafer.group_yield
// GROUP: return %[[MM_GROUP]] : tensor<4x16xf16>

// GROUP-LABEL: func.func @same_shape_elementwise(
// GROUP: %[[EW_GROUP:.+]] = wafer.group
// GROUP: linalg.generic
// GROUP: arith.addf
// GROUP: wafer.group_yield
// GROUP: return %[[EW_GROUP]] : tensor<4x8xf16>

// GROUP-LABEL: func.func @row_sum_reduce(
// GROUP: %[[REDUCE_GROUP:.+]] = wafer.group
// GROUP: linalg.reduce
// GROUP-SAME: arith.addf
// GROUP: wafer.group_yield
// GROUP: return %[[REDUCE_GROUP]] : tensor<4xf16>

// TILE-LABEL: func.func @single_matmul(
// TILE: wafer.tile_region
// TILE-NOT: wafer.group
// TILE: wafer.load_tile
// TILE: wafer.compute.gemm
// TILE: wafer.store_tile

// TILE-LABEL: func.func @same_shape_elementwise(
// TILE: wafer.tile_region
// TILE-NOT: wafer.group
// TILE: wafer.load_tile
// TILE: wafer.compute.elementwise
// TILE-SAME: <add>
// TILE: wafer.store_tile

// TILE-LABEL: func.func @row_sum_reduce(
// TILE: wafer.tile_region
// TILE-NOT: wafer.group
// TILE: wafer.load_tile
// TILE: wafer.compute.reduce
// TILE-SAME: <sum>
// TILE-SAME: dimensions = array<i64: 1>
// TILE: wafer.store_tile

// ABI-LABEL: func.func @single_matmul(
// ABI-NOT: linalg.matmul
// ABI-NOT: wafer.load_tile
// ABI: wafer.abi.rdma <issue_only>
// ABI: wafer.abi.rdma <issue_only>
// ABI-NOT: wafer.compute.gemm
// ABI: wafer.abi.gemm
// ABI-SAME: k = 8 : i64
// ABI-SAME: m = 4 : i64
// ABI-SAME: n = 16 : i64
// ABI-NOT: wafer.store_tile
// ABI: wafer.abi.wdma <issue_only>

// ABI-LABEL: func.func @same_shape_elementwise(
// ABI-NOT: linalg.generic
// ABI-NOT: wafer.load_tile
// ABI: wafer.abi.rdma <issue_only>
// ABI: wafer.abi.rdma <issue_only>
// ABI-NOT: wafer.compute.elementwise
// ABI: wafer.abi.elementwise
// ABI-SAME: <add>
// ABI-NOT: wafer.store_tile
// ABI: wafer.abi.wdma <issue_only>

// ABI-LABEL: func.func @row_sum_reduce(
// ABI-NOT: linalg.reduce
// ABI-NOT: wafer.load_tile
// ABI: wafer.abi.rdma <issue_only>
// ABI-NOT: wafer.compute.reduce
// ABI: wafer.abi.reduce
// ABI-SAME: <sum>
// ABI-SAME: dimensions = array<i64: 1>
// ABI-NOT: wafer.store_tile
// ABI: wafer.abi.wdma <issue_only>
