// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-fold-static-tensor-ops,canonicalize)' %s | FileCheck %s

#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0) -> (d0)>

module {
  func.func @rank_mask_constant_cleanup() -> (tensor<1x1x4x1xi1>, index) {
    %table = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %two = arith.constant dense<2> : tensor<i32>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index

    %slice = tensor.extract_slice %table[0] [1] [1]
        : tensor<4xi32> to tensor<1xi32>
    %rank = tensor.collapse_shape %slice []
        : tensor<1xi32> into tensor<i32>

    %sum_empty = tensor.empty() : tensor<i32>
    %sum = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = []
    } ins(%rank, %two : tensor<i32>, tensor<i32>)
      outs(%sum_empty : tensor<i32>) {
    ^bb0(%lhs: i32, %rhs: i32, %out: i32):
      %0 = arith.addi %lhs, %rhs : i32
      linalg.yield %0 : i32
    } -> tensor<i32>

    %extracted = tensor.extract %sum[] : tensor<i32>
    %idx0 = arith.index_cast %extracted : i32 to index
    %idx1 = arith.maxsi %idx0, %c0 : index
    %idx = arith.minsi %idx1, %c3 : index

    %broadcast_empty = tensor.empty() : tensor<1x1x4x1xi32>
    %broadcast = linalg.generic {
      indexing_maps = [#map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%rank : tensor<i32>)
      outs(%broadcast_empty : tensor<1x1x4x1xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x1x4x1xi32>

    %mask_empty = tensor.empty() : tensor<1x1x4x1xi1>
    %mask = linalg.generic {
      indexing_maps = [#map2, #map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%broadcast, %two : tensor<1x1x4x1xi32>, tensor<i32>)
      outs(%mask_empty : tensor<1x1x4x1xi1>) {
    ^bb0(%lhs: i32, %rhs: i32, %out: i1):
      %0 = arith.cmpi slt, %lhs, %rhs : i32
      linalg.yield %0 : i1
    } -> tensor<1x1x4x1xi1>

    return %mask, %idx : tensor<1x1x4x1xi1>, index
  }

  func.func @nested_rank_table_cleanup() -> index {
    %rank_table = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7,
                                        8, 9, 10, 11, 12, 13, 14, 15]>
        : tensor<16xi32>
    %group_table = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index

    %rank_slice = tensor.extract_slice %rank_table[0] [1] [1]
        : tensor<16xi32> to tensor<1xi32>
    %rank_tensor = tensor.collapse_shape %rank_slice []
        : tensor<1xi32> into tensor<i32>
    %rank_i32 = tensor.extract %rank_tensor[] : tensor<i32>
    %rank_index = arith.index_castui %rank_i32 : i32 to index
    %rank_nonnegative = arith.maxsi %rank_index, %c0 : index
    %rank = arith.minsi %rank_nonnegative, %c3 : index

    %group_slice = tensor.extract_slice %group_table[%rank] [1] [1]
        : tensor<4xi32> to tensor<1xi32>
    %group_tensor = tensor.collapse_shape %group_slice []
        : tensor<1xi32> into tensor<i32>
    %group_i32 = tensor.extract %group_tensor[] : tensor<i32>
    %group = arith.index_cast %group_i32 : i32 to index
    return %group : index
  }

  func.func @dynamic_rank_table_is_not_folded(%index: index) -> i32 {
    %table = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %value = tensor.extract %table[%index] : tensor<4xi32>
    return %value : i32
  }

  func.func @large_constant_generic_respects_budget()
      -> tensor<1048577xi32> {
    %input = arith.constant dense<1> : tensor<1048577xi32>
    %empty = tensor.empty() : tensor<1048577xi32>
    %result = linalg.generic {
      indexing_maps = [#map3, #map3],
      iterator_types = ["parallel"]
    } ins(%input : tensor<1048577xi32>)
      outs(%empty : tensor<1048577xi32>) {
    ^bb0(%in: i32, %out: i32):
      %sum = arith.addi %in, %in : i32
      linalg.yield %sum : i32
    } -> tensor<1048577xi32>
    return %result : tensor<1048577xi32>
  }
}

// CHECK-LABEL: func.func @rank_mask_constant_cleanup
// CHECK-DAG: %[[MASK:.+]] = arith.constant dense<true> : tensor<1x1x4x1xi1>
// CHECK-DAG: %[[IDX:.+]] = arith.constant 2 : index
// CHECK-NOT: linalg.generic
// CHECK: return %[[MASK]], %[[IDX]] : tensor<1x1x4x1xi1>, index

// CHECK-LABEL: func.func @nested_rank_table_cleanup
// CHECK-NOT: tensor.extract
// CHECK-NOT: tensor.extract_slice
// CHECK: %[[ZERO:.+]] = arith.constant 0 : index
// CHECK: return %[[ZERO]] : index

// CHECK-LABEL: func.func @dynamic_rank_table_is_not_folded
// CHECK: tensor.extract {{.+}}[%arg0] : tensor<4xi32>

// CHECK-LABEL: func.func @large_constant_generic_respects_budget
// CHECK: linalg.generic
