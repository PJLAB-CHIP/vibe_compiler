// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %s -o %t.once
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %t.once -o %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once
// RUN: wafer-opt --mlir-pass-statistics --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#swap12_3 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#swap12_4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#swap23_4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#swap01_3 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#broadcast3 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#broadcast4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#reduce_last4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#reduce_middle4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#batch_lhs_transposed = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#batch_rhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#batch_output_transposed = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>

module {
  func.func @aligned_reshape_elementwise_chain(
      %arg0: tensor<2x1024x64xf32>) -> tensor<2x1024x64xf32> {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
        output_shape [2, 1024, 1, 64] :
        tensor<2x1024x64xf32> into tensor<2x1024x1x64xf32>
    %reshaped = tensor.collapse_shape %expanded [[0], [1], [2, 3]] :
        tensor<2x1024x1x64xf32> into tensor<2x1024x64xf32>
    %input_empty = tensor.empty() : tensor<2x64x1024xf32>
    %input_transposed = linalg.transpose
        ins(%reshaped : tensor<2x1024x64xf32>)
        outs(%input_empty : tensor<2x64x1024xf32>)
        permutation = [0, 2, 1]
    %compute_empty = tensor.empty() : tensor<2x64x1024xf32>
    %computed = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input_transposed : tensor<2x64x1024xf32>)
        outs(%compute_empty : tensor<2x64x1024xf32>) {
      ^bb0(%input: f32, %output: f32):
        %negated = arith.negf %input : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x1024xf32>
    %result_empty = tensor.empty() : tensor<2x1024x64xf32>
    %result = linalg.transpose
        ins(%computed : tensor<2x64x1024xf32>)
        outs(%result_empty : tensor<2x1024x64xf32>)
        permutation = [0, 2, 1]
    return %result : tensor<2x1024x64xf32>
  }

  func.func @ragged_reshape_elementwise_chain(
      %arg0: tensor<2x1025x64xf32>) -> tensor<2x1025x64xf32> {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
        output_shape [2, 1025, 1, 64] :
        tensor<2x1025x64xf32> into tensor<2x1025x1x64xf32>
    %reshaped = tensor.collapse_shape %expanded [[0], [1], [2, 3]] :
        tensor<2x1025x1x64xf32> into tensor<2x1025x64xf32>
    %input_empty = tensor.empty() : tensor<2x64x1025xf32>
    %input_transposed = linalg.transpose
        ins(%reshaped : tensor<2x1025x64xf32>)
        outs(%input_empty : tensor<2x64x1025xf32>)
        permutation = [0, 2, 1]
    %compute_empty = tensor.empty() : tensor<2x64x1025xf32>
    %computed = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input_transposed : tensor<2x64x1025xf32>)
        outs(%compute_empty : tensor<2x64x1025xf32>) {
      ^bb0(%input: f32, %output: f32):
        %negated = arith.negf %input : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x1025xf32>
    %result_empty = tensor.empty() : tensor<2x1025x64xf32>
    %result = linalg.transpose
        ins(%computed : tensor<2x64x1025xf32>)
        outs(%result_empty : tensor<2x1025x64xf32>)
        permutation = [0, 2, 1]
    return %result : tensor<2x1025x64xf32>
  }

  func.func @ragged_broadcast_elementwise_chain(
      %arg0: tensor<2x1x64xf32>) -> tensor<2x64x1025xf32> {
    %broadcast_empty = tensor.empty() : tensor<2x1025x64xf32>
    %broadcast = linalg.generic {
        indexing_maps = [#broadcast3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<2x1x64xf32>)
        outs(%broadcast_empty : tensor<2x1025x64xf32>) {
      ^bb0(%input: f32, %output: f32):
        linalg.yield %input : f32
    } -> tensor<2x1025x64xf32>
    %compute_empty = tensor.empty() : tensor<2x1025x64xf32>
    %computed = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%broadcast : tensor<2x1025x64xf32>)
        outs(%compute_empty : tensor<2x1025x64xf32>) {
      ^bb0(%input: f32, %output: f32):
        %negated = arith.negf %input : f32
        linalg.yield %negated : f32
    } -> tensor<2x1025x64xf32>
    %result_empty = tensor.empty() : tensor<2x64x1025xf32>
    %result = linalg.transpose
        ins(%computed : tensor<2x1025x64xf32>)
        outs(%result_empty : tensor<2x64x1025xf32>)
        permutation = [0, 2, 1]
    return %result : tensor<2x64x1025xf32>
  }

  func.func @ragged_concat_elementwise_chain(
      %first: tensor<2x512x64xf32>, %second: tensor<2x512x64xf32>,
      %tail: tensor<2x1x64xf32>) -> tensor<2x1025x64xf32> {
    %first_empty = tensor.empty() : tensor<512x2x64xf32>
    %first_transposed = linalg.transpose
        ins(%first : tensor<2x512x64xf32>)
        outs(%first_empty : tensor<512x2x64xf32>) permutation = [1, 0, 2]
    %second_empty = tensor.empty() : tensor<512x2x64xf32>
    %second_transposed = linalg.transpose
        ins(%second : tensor<2x512x64xf32>)
        outs(%second_empty : tensor<512x2x64xf32>) permutation = [1, 0, 2]
    %tail_empty = tensor.empty() : tensor<1x2x64xf32>
    %tail_transposed = linalg.transpose
        ins(%tail : tensor<2x1x64xf32>)
        outs(%tail_empty : tensor<1x2x64xf32>) permutation = [1, 0, 2]
    %concat_empty = tensor.empty() : tensor<1025x2x64xf32>
    %concat_first = tensor.insert_slice %first_transposed into %concat_empty
        [0, 0, 0] [512, 2, 64] [1, 1, 1] :
        tensor<512x2x64xf32> into tensor<1025x2x64xf32>
    %concat_second = tensor.insert_slice %second_transposed into %concat_first
        [512, 0, 0] [512, 2, 64] [1, 1, 1] :
        tensor<512x2x64xf32> into tensor<1025x2x64xf32>
    %concat = tensor.insert_slice %tail_transposed into %concat_second
        [1024, 0, 0] [1, 2, 64] [1, 1, 1] :
        tensor<1x2x64xf32> into tensor<1025x2x64xf32>
    %compute_empty = tensor.empty() : tensor<1025x2x64xf32>
    %computed = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%concat : tensor<1025x2x64xf32>)
        outs(%compute_empty : tensor<1025x2x64xf32>) {
      ^bb0(%input: f32, %output: f32):
        %negated = arith.negf %input : f32
        linalg.yield %negated : f32
    } -> tensor<1025x2x64xf32>
    %result_empty = tensor.empty() : tensor<2x1025x64xf32>
    %result = linalg.transpose
        ins(%computed : tensor<1025x2x64xf32>)
        outs(%result_empty : tensor<2x1025x64xf32>) permutation = [1, 0, 2]
    return %result : tensor<2x1025x64xf32>
  }

  func.func @aligned_reshape_reduction_chain(
      %input: tensor<2x4x64x1024xf32>, %init: tensor<2x4x64xf32>)
      -> tensor<2x4x64xf32> {
    %expanded = tensor.expand_shape %input [[0], [1], [2], [3, 4]]
        output_shape [2, 4, 64, 1, 1024] :
        tensor<2x4x64x1024xf32> into tensor<2x4x64x1x1024xf32>
    %reshaped = tensor.collapse_shape %expanded [[0], [1], [2], [3, 4]] :
        tensor<2x4x64x1x1024xf32> into tensor<2x4x64x1024xf32>
    %input_empty = tensor.empty() : tensor<2x64x4x1024xf32>
    %input_transposed = linalg.transpose
        ins(%reshaped : tensor<2x4x64x1024xf32>)
        outs(%input_empty : tensor<2x64x4x1024xf32>)
        permutation = [0, 2, 1, 3]
    %init_empty = tensor.empty() : tensor<2x64x4xf32>
    %init_transposed = linalg.transpose
        ins(%init : tensor<2x4x64xf32>)
        outs(%init_empty : tensor<2x64x4xf32>) permutation = [0, 2, 1]
    %reduced = linalg.generic {
        indexing_maps = [#identity4, #reduce_last4],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%input_transposed : tensor<2x64x4x1024xf32>)
        outs(%init_transposed : tensor<2x64x4xf32>) {
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %accumulator, %element : f32
        linalg.yield %sum : f32
    } -> tensor<2x64x4xf32>
    %result_empty = tensor.empty() : tensor<2x4x64xf32>
    %result = linalg.transpose
        ins(%reduced : tensor<2x64x4xf32>)
        outs(%result_empty : tensor<2x4x64xf32>) permutation = [0, 2, 1]
    return %result : tensor<2x4x64xf32>
  }

  func.func @ragged_concat_reduction_chain(
      %first: tensor<2x4x512x64xf32>,
      %second: tensor<2x4x512x64xf32>,
      %tail: tensor<2x4x1x64xf32>, %init: tensor<2x4x64xf32>)
      -> tensor<2x4x64xf32> {
    %first_empty = tensor.empty() : tensor<2x4x64x512xf32>
    %first_transposed = linalg.transpose
        ins(%first : tensor<2x4x512x64xf32>)
        outs(%first_empty : tensor<2x4x64x512xf32>)
        permutation = [0, 1, 3, 2]
    %second_empty = tensor.empty() : tensor<2x4x64x512xf32>
    %second_transposed = linalg.transpose
        ins(%second : tensor<2x4x512x64xf32>)
        outs(%second_empty : tensor<2x4x64x512xf32>)
        permutation = [0, 1, 3, 2]
    %tail_empty = tensor.empty() : tensor<2x4x64x1xf32>
    %tail_transposed = linalg.transpose
        ins(%tail : tensor<2x4x1x64xf32>)
        outs(%tail_empty : tensor<2x4x64x1xf32>)
        permutation = [0, 1, 3, 2]
    %concat_empty = tensor.empty() : tensor<2x4x64x1025xf32>
    %concat_first = tensor.insert_slice %first_transposed into %concat_empty
        [0, 0, 0, 0] [2, 4, 64, 512] [1, 1, 1, 1] :
        tensor<2x4x64x512xf32> into tensor<2x4x64x1025xf32>
    %concat_second = tensor.insert_slice %second_transposed into %concat_first
        [0, 0, 0, 512] [2, 4, 64, 512] [1, 1, 1, 1] :
        tensor<2x4x64x512xf32> into tensor<2x4x64x1025xf32>
    %concat = tensor.insert_slice %tail_transposed into %concat_second
        [0, 0, 0, 1024] [2, 4, 64, 1] [1, 1, 1, 1] :
        tensor<2x4x64x1xf32> into tensor<2x4x64x1025xf32>
    %reduced = linalg.generic {
        indexing_maps = [#identity4, #reduce_last4],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%concat : tensor<2x4x64x1025xf32>)
        outs(%init : tensor<2x4x64xf32>) {
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %accumulator, %element : f32
        linalg.yield %sum : f32
    } -> tensor<2x4x64xf32>
    return %reduced : tensor<2x4x64xf32>
  }

  func.func @ragged_broadcast_reduction_partial_chain(
      %input: tensor<2x1x4x64xf32>, %init: tensor<2x64x4xf32>)
      -> tensor<2x64x4xf32> {
    %broadcast_empty = tensor.empty() : tensor<2x1031x4x64xf32>
    %broadcast = linalg.generic {
        indexing_maps = [#broadcast4, #identity4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1x4x64xf32>)
        outs(%broadcast_empty : tensor<2x1031x4x64xf32>) {
      ^bb0(%element: f32, %output: f32):
        linalg.yield %element : f32
    } -> tensor<2x1031x4x64xf32>
    %init_empty = tensor.empty() : tensor<2x4x64xf32>
    %init_transposed = linalg.transpose
        ins(%init : tensor<2x64x4xf32>)
        outs(%init_empty : tensor<2x4x64xf32>) permutation = [0, 2, 1]
    %reduced = linalg.generic {
        indexing_maps = [#identity4, #reduce_middle4],
        iterator_types = ["parallel", "reduction", "parallel", "parallel"]}
        ins(%broadcast : tensor<2x1031x4x64xf32>)
        outs(%init_transposed : tensor<2x4x64xf32>) {
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %accumulator, %element : f32
        linalg.yield %sum : f32
    } -> tensor<2x4x64xf32>
    %result_empty = tensor.empty() : tensor<2x64x4xf32>
    %result = linalg.transpose
        ins(%reduced : tensor<2x4x64xf32>)
        outs(%result_empty : tensor<2x64x4xf32>) permutation = [0, 2, 1]
    return %result : tensor<2x64x4xf32>
  }

  func.func @ragged_reshape_contraction_chain(
      %lhs: tensor<2x64x1031xf32>, %rhs: tensor<2x1031x128xf32>,
      %init: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
    %lhs_expanded = tensor.expand_shape %lhs [[0], [1], [2, 3]]
        output_shape [2, 64, 1, 1031] :
        tensor<2x64x1031xf32> into tensor<2x64x1x1031xf32>
    %lhs_reshaped = tensor.collapse_shape %lhs_expanded [[0], [1], [2, 3]] :
        tensor<2x64x1x1031xf32> into tensor<2x64x1031xf32>
    %lhs_empty = tensor.empty() : tensor<2x1031x64xf32>
    %lhs_transposed = linalg.transpose
        ins(%lhs_reshaped : tensor<2x64x1031xf32>)
        outs(%lhs_empty : tensor<2x1031x64xf32>) permutation = [0, 2, 1]
    %init_empty = tensor.empty() : tensor<2x128x64xf32>
    %init_transposed = linalg.transpose
        ins(%init : tensor<2x64x128xf32>)
        outs(%init_empty : tensor<2x128x64xf32>) permutation = [0, 2, 1]
    %contracted = linalg.generic {
        indexing_maps = [#batch_lhs_transposed, #batch_rhs,
                         #batch_output_transposed],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs_transposed, %rhs :
            tensor<2x1031x64xf32>, tensor<2x1031x128xf32>)
        outs(%init_transposed : tensor<2x128x64xf32>) {
      ^bb0(%left: f32, %right: f32, %accumulator: f32):
        %product = arith.mulf %left, %right : f32
        %sum = arith.addf %accumulator, %product : f32
        linalg.yield %sum : f32
    } -> tensor<2x128x64xf32>
    %result_empty = tensor.empty() : tensor<2x64x128xf32>
    %result = linalg.transpose
        ins(%contracted : tensor<2x128x64xf32>)
        outs(%result_empty : tensor<2x64x128xf32>) permutation = [0, 2, 1]
    return %result : tensor<2x64x128xf32>
  }
}

// CHECK-DAG: #[[SWAP12_3:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-DAG: #[[SWAP12_4:map[0-9]*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
// CHECK-DAG: #[[SWAP23_4:map[0-9]*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-DAG: #[[SWAP01_3:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-DAG: #[[BROADCAST3:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
// CHECK-DAG: #[[BROADCAST4:map[0-9]*]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
// CHECK-DAG: #[[REDUCE_OUTPUT:map[0-9]*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>

// CHECK-LABEL: func.func @aligned_reshape_elementwise_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: indexing_maps = [#[[SWAP12_3]], #[[SWAP12_3]]]
// CHECK-SAME: ins(%arg0 : tensor<2x1024x64xf32>)
// CHECK: arith.negf

// CHECK-LABEL: func.func @ragged_reshape_elementwise_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: indexing_maps = [#[[SWAP12_3]], #[[SWAP12_3]]]
// CHECK-SAME: ins(%arg0 : tensor<2x1025x64xf32>)
// CHECK: arith.negf

// CHECK-LABEL: func.func @ragged_broadcast_elementwise_chain
// CHECK-NOT: linalg.transpose
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: indexing_maps = [#[[BROADCAST3]], #[[SWAP12_3]]]
// CHECK-SAME: ins(%arg0 : tensor<2x1x64xf32>)
// CHECK: arith.negf

// CHECK-LABEL: func.func @ragged_concat_elementwise_chain
// CHECK-NOT: linalg.transpose
// CHECK-COUNT-3: tensor.insert_slice
// CHECK: [0, 1024, 0] [2, 1, 64]
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: indexing_maps = [#[[SWAP01_3]], #[[SWAP01_3]]]
// CHECK: arith.negf

// CHECK-LABEL: func.func @aligned_reshape_reduction_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: indexing_maps = [#[[SWAP12_4]], #[[REDUCE_OUTPUT]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%arg0 : tensor<2x4x64x1024xf32>)
// CHECK-SAME: outs(%arg1 : tensor<2x4x64xf32>)
// CHECK: arith.addf

// CHECK-LABEL: func.func @ragged_concat_reduction_chain
// CHECK-NOT: linalg.transpose
// CHECK-COUNT-3: tensor.insert_slice
// CHECK: [0, 0, 1024, 0] [2, 4, 1, 64]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[SWAP23_4]],
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: outs(%arg3 : tensor<2x4x64xf32>)
// CHECK: arith.addf

// CHECK-LABEL: func.func @ragged_broadcast_reduction_partial_chain
// CHECK-NOT: linalg.transpose
// CHECK: %[[BROADCAST:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[BROADCAST4]],
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel", "parallel"]
// CHECK-SAME: ins(%[[BROADCAST]] : tensor<2x1031x4x64xf32>)
// CHECK-SAME: outs(%arg1 : tensor<2x64x4xf32>)
// CHECK: arith.addf

// CHECK-LABEL: func.func @ragged_reshape_contraction_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.generic
// CHECK-COUNT-1: linalg.batch_matmul
// CHECK-SAME: ins(%arg0, %arg1 : tensor<2x64x1031xf32>, tensor<2x1031x128xf32>)
// CHECK-SAME: outs(%arg2 : tensor<2x64x128xf32>)

// STATS-DAG: (S) {{0+}} budget-exhausted-components
// STATS-DAG: (S) {{[1-9][0-9]*}} changed-components
// STATS-DAG: (S) {{[1-9][0-9]*}} multi-rule-changed-components
// STATS-DAG: (S) {{[1-9][0-9]*}} access-transforms-removed
