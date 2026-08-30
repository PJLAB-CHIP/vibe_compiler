// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %s -o %t.once
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %t.once -o %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once
// RUN: wafer-opt --mlir-pass-statistics --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#swap12_3 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#reduce_last4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#batch_lhs_transposed = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#batch_rhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#batch_output_transposed = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
#batch_broadcast_lhs = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#unsupported_lhs = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#unsupported_rhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
#unsupported_output = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>

module {
  func.func @ragged_long_alternating_chain(
      %arg0: tensor<2x1025x64xf32>) -> tensor<2x1025x64xf32> {
    %expanded0 = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
        output_shape [2, 1025, 1, 64] :
        tensor<2x1025x64xf32> into tensor<2x1025x1x64xf32>
    %reshaped0 = tensor.collapse_shape %expanded0 [[0], [1], [2, 3]] :
        tensor<2x1025x1x64xf32> into tensor<2x1025x64xf32>
    %transpose0_empty = tensor.empty() : tensor<2x64x1025xf32>
    %transpose0 = linalg.transpose
        ins(%reshaped0 : tensor<2x1025x64xf32>)
        outs(%transpose0_empty : tensor<2x64x1025xf32>)
        permutation = [0, 2, 1]
    %compute0_empty = tensor.empty() : tensor<2x64x1025xf32>
    %compute0 = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%transpose0 : tensor<2x64x1025xf32>)
        outs(%compute0_empty : tensor<2x64x1025xf32>) {
      ^bb0(%input: f32, %output: f32):
        %value = arith.negf %input : f32
        linalg.yield %value : f32
    } -> tensor<2x64x1025xf32>
    %transpose1_empty = tensor.empty() : tensor<2x1025x64xf32>
    %transpose1 = linalg.transpose
        ins(%compute0 : tensor<2x64x1025xf32>)
        outs(%transpose1_empty : tensor<2x1025x64xf32>)
        permutation = [0, 2, 1]
    %compute1_empty = tensor.empty() : tensor<2x1025x64xf32>
    %compute1 = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%transpose1 : tensor<2x1025x64xf32>)
        outs(%compute1_empty : tensor<2x1025x64xf32>) {
      ^bb0(%input: f32, %output: f32):
        %value = math.exp %input : f32
        linalg.yield %value : f32
    } -> tensor<2x1025x64xf32>
    %expanded1 = tensor.expand_shape %compute1 [[0], [1], [2, 3]]
        output_shape [2, 1025, 1, 64] :
        tensor<2x1025x64xf32> into tensor<2x1025x1x64xf32>
    %reshaped1 = tensor.collapse_shape %expanded1 [[0], [1], [2, 3]] :
        tensor<2x1025x1x64xf32> into tensor<2x1025x64xf32>
    %transpose2_empty = tensor.empty() : tensor<2x64x1025xf32>
    %transpose2 = linalg.transpose
        ins(%reshaped1 : tensor<2x1025x64xf32>)
        outs(%transpose2_empty : tensor<2x64x1025xf32>)
        permutation = [0, 2, 1]
    %compute2_empty = tensor.empty() : tensor<2x64x1025xf32>
    %compute2 = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%transpose2 : tensor<2x64x1025xf32>)
        outs(%compute2_empty : tensor<2x64x1025xf32>) {
      ^bb0(%input: f32, %output: f32):
        %value = arith.mulf %input, %input : f32
        linalg.yield %value : f32
    } -> tensor<2x64x1025xf32>
    %compute3_empty = tensor.empty() : tensor<2x64x1025xf32>
    %compute3 = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%compute2 : tensor<2x64x1025xf32>)
        outs(%compute3_empty : tensor<2x64x1025xf32>) {
      ^bb0(%input: f32, %output: f32):
        %value = math.rsqrt %input : f32
        linalg.yield %value : f32
    } -> tensor<2x64x1025xf32>
    %result_empty = tensor.empty() : tensor<2x1025x64xf32>
    %result = linalg.transpose
        ins(%compute3 : tensor<2x64x1025xf32>)
        outs(%result_empty : tensor<2x1025x64xf32>) permutation = [0, 2, 1]
    return %result : tensor<2x1025x64xf32>
  }

  func.func @ragged_two_elementwise_chain(
      %arg0: tensor<2x1025x64xf32>) -> tensor<2x1025x64xf32> {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
        output_shape [2, 1025, 1, 64] :
        tensor<2x1025x64xf32> into tensor<2x1025x1x64xf32>
    %reshaped = tensor.collapse_shape %expanded [[0], [1], [2, 3]] :
        tensor<2x1025x1x64xf32> into tensor<2x1025x64xf32>
    %input_empty = tensor.empty() : tensor<2x64x1025xf32>
    %input_transposed = linalg.transpose
        ins(%reshaped : tensor<2x1025x64xf32>)
        outs(%input_empty : tensor<2x64x1025xf32>) permutation = [0, 2, 1]
    %first_empty = tensor.empty() : tensor<2x64x1025xf32>
    %first = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input_transposed : tensor<2x64x1025xf32>)
        outs(%first_empty : tensor<2x64x1025xf32>) {
      ^bb0(%input: f32, %output: f32):
        %negated = arith.negf %input : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x1025xf32>
    %second_empty = tensor.empty() : tensor<2x64x1025xf32>
    %second = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%first : tensor<2x64x1025xf32>)
        outs(%second_empty : tensor<2x64x1025xf32>) {
      ^bb0(%input: f32, %output: f32):
        %exponential = math.exp %input : f32
        linalg.yield %exponential : f32
    } -> tensor<2x64x1025xf32>
    %result_empty = tensor.empty() : tensor<2x1025x64xf32>
    %result = linalg.transpose
        ins(%second : tensor<2x64x1025xf32>)
        outs(%result_empty : tensor<2x1025x64xf32>) permutation = [0, 2, 1]
    return %result : tensor<2x1025x64xf32>
  }

  func.func @ragged_elementwise_reduction_chain(
      %input: tensor<2x4x64x1031xf32>, %init: tensor<2x4x64xf32>)
      -> tensor<2x4x64xf32> {
    %expanded = tensor.expand_shape %input [[0], [1], [2], [3, 4]]
        output_shape [2, 4, 64, 1, 1031] :
        tensor<2x4x64x1031xf32> into tensor<2x4x64x1x1031xf32>
    %reshaped = tensor.collapse_shape %expanded [[0], [1], [2], [3, 4]] :
        tensor<2x4x64x1x1031xf32> into tensor<2x4x64x1031xf32>
    %input_empty = tensor.empty() : tensor<2x64x4x1031xf32>
    %input_transposed = linalg.transpose
        ins(%reshaped : tensor<2x4x64x1031xf32>)
        outs(%input_empty : tensor<2x64x4x1031xf32>)
        permutation = [0, 2, 1, 3]
    %elementwise_empty = tensor.empty() : tensor<2x64x4x1031xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity4, #identity4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%input_transposed : tensor<2x64x4x1031xf32>)
        outs(%elementwise_empty : tensor<2x64x4x1031xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x4x1031xf32>
    %init_empty = tensor.empty() : tensor<2x64x4xf32>
    %init_transposed = linalg.transpose
        ins(%init : tensor<2x4x64xf32>)
        outs(%init_empty : tensor<2x64x4xf32>) permutation = [0, 2, 1]
    %reduced = linalg.generic {
        indexing_maps = [#identity4, #reduce_last4],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%elementwise : tensor<2x64x4x1031xf32>)
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

  func.func @ragged_contraction_elementwise_chain(
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
    %elementwise_empty = tensor.empty() : tensor<2x128x64xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%contracted : tensor<2x128x64xf32>)
        outs(%elementwise_empty : tensor<2x128x64xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x128x64xf32>
    %result_empty = tensor.empty() : tensor<2x64x128xf32>
    %result = linalg.transpose
        ins(%elementwise : tensor<2x128x64xf32>)
        outs(%result_empty : tensor<2x64x128xf32>) permutation = [0, 2, 1]
    return %result : tensor<2x64x128xf32>
  }

  func.func @ragged_fanout_diamond_chain(
      %arg0: tensor<2x1025x64xf32>) -> tensor<2x1025x64xf32> {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
        output_shape [2, 1025, 1, 64] :
        tensor<2x1025x64xf32> into tensor<2x1025x1x64xf32>
    %reshaped = tensor.collapse_shape %expanded [[0], [1], [2, 3]] :
        tensor<2x1025x1x64xf32> into tensor<2x1025x64xf32>
    %shared_empty = tensor.empty() : tensor<2x64x1025xf32>
    %shared = linalg.transpose
        ins(%reshaped : tensor<2x1025x64xf32>)
        outs(%shared_empty : tensor<2x64x1025xf32>) permutation = [0, 2, 1]
    %left_empty = tensor.empty() : tensor<2x64x1025xf32>
    %left = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%shared : tensor<2x64x1025xf32>)
        outs(%left_empty : tensor<2x64x1025xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x1025xf32>
    %right_empty = tensor.empty() : tensor<2x64x1025xf32>
    %right = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%shared : tensor<2x64x1025xf32>)
        outs(%right_empty : tensor<2x64x1025xf32>) {
      ^bb0(%element: f32, %output: f32):
        %exponential = math.exp %element : f32
        linalg.yield %exponential : f32
    } -> tensor<2x64x1025xf32>
    %merge_empty = tensor.empty() : tensor<2x64x1025xf32>
    %merged = linalg.generic {
        indexing_maps = [#identity3, #identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%left, %right : tensor<2x64x1025xf32>, tensor<2x64x1025xf32>)
        outs(%merge_empty : tensor<2x64x1025xf32>) {
      ^bb0(%left_value: f32, %right_value: f32, %output: f32):
        %sum = arith.addf %left_value, %right_value : f32
        linalg.yield %sum : f32
    } -> tensor<2x64x1025xf32>
    %result_empty = tensor.empty() : tensor<2x1025x64xf32>
    %result = linalg.transpose
        ins(%merged : tensor<2x64x1025xf32>)
        outs(%result_empty : tensor<2x1025x64xf32>) permutation = [0, 2, 1]
    return %result : tensor<2x1025x64xf32>
  }

  func.func @aligned_mixed_elementwise_reduction_fanout(
      %input: tensor<2x4x1024x64xf32>, %init: tensor<2x4x64xf32>)
      -> (tensor<2x4x64x1024xf32>, tensor<2x4x64xf32>) {
    %shared_empty = tensor.empty() : tensor<2x4x64x1024xf32>
    %shared = linalg.transpose
        ins(%input : tensor<2x4x1024x64xf32>)
        outs(%shared_empty : tensor<2x4x64x1024xf32>)
        permutation = [0, 1, 3, 2]
    %elementwise_empty = tensor.empty() : tensor<2x4x64x1024xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity4, #identity4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%shared : tensor<2x4x64x1024xf32>)
        outs(%elementwise_empty : tensor<2x4x64x1024xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x4x64x1024xf32>
    %reduced = linalg.generic {
        indexing_maps = [#identity4, #reduce_last4],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%shared : tensor<2x4x64x1024xf32>)
        outs(%init : tensor<2x4x64xf32>) {
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %accumulator, %element : f32
        linalg.yield %sum : f32
    } -> tensor<2x4x64xf32>
    return %elementwise, %reduced :
        tensor<2x4x64x1024xf32>, tensor<2x4x64xf32>
  }

  func.func @ragged_mixed_elementwise_reduction_fanout(
      %input: tensor<2x4x1025x64xf32>, %init: tensor<2x4x64xf32>)
      -> (tensor<2x4x64x1025xf32>, tensor<2x4x64xf32>) {
    %shared_empty = tensor.empty() : tensor<2x4x64x1025xf32>
    %shared = linalg.transpose
        ins(%input : tensor<2x4x1025x64xf32>)
        outs(%shared_empty : tensor<2x4x64x1025xf32>)
        permutation = [0, 1, 3, 2]
    %elementwise_empty = tensor.empty() : tensor<2x4x64x1025xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity4, #identity4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%shared : tensor<2x4x64x1025xf32>)
        outs(%elementwise_empty : tensor<2x4x64x1025xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x4x64x1025xf32>
    %reduced = linalg.generic {
        indexing_maps = [#identity4, #reduce_last4],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%shared : tensor<2x4x64x1025xf32>)
        outs(%init : tensor<2x4x64xf32>) {
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %accumulator, %element : f32
        linalg.yield %sum : f32
    } -> tensor<2x4x64xf32>
    return %elementwise, %reduced :
        tensor<2x4x64x1025xf32>, tensor<2x4x64xf32>
  }

  func.func @ragged_mixed_elementwise_contraction_fanout(
      %lhs: tensor<2x64x1031xf32>, %rhs: tensor<2x64x128xf32>,
      %init: tensor<2x1031x128xf32>)
      -> (tensor<2x1031x64xf32>, tensor<2x1031x128xf32>) {
    %shared_empty = tensor.empty() : tensor<2x1031x64xf32>
    %shared = linalg.transpose
        ins(%lhs : tensor<2x64x1031xf32>)
        outs(%shared_empty : tensor<2x1031x64xf32>) permutation = [0, 2, 1]
    %elementwise_empty = tensor.empty() : tensor<2x1031x64xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%shared : tensor<2x1031x64xf32>)
        outs(%elementwise_empty : tensor<2x1031x64xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x1031x64xf32>
    %contracted = linalg.batch_matmul
        ins(%shared, %rhs : tensor<2x1031x64xf32>, tensor<2x64x128xf32>)
        outs(%init : tensor<2x1031x128xf32>) -> tensor<2x1031x128xf32>
    return %elementwise, %contracted :
        tensor<2x1031x64xf32>, tensor<2x1031x128xf32>
  }

  func.func @ragged_transient_init_use_chain(
      %input: tensor<2x4x1025x64xf32>,
      %data: tensor<2x4x64x1025xf32>)
      -> (tensor<2x4x64x1025xf32>, tensor<2x4x64x1025xf32>) {
    %shared_empty = tensor.empty() : tensor<2x4x64x1025xf32>
    %shared = linalg.transpose
        ins(%input : tensor<2x4x1025x64xf32>)
        outs(%shared_empty : tensor<2x4x64x1025xf32>)
        permutation = [0, 1, 3, 2]
    %elementwise_empty = tensor.empty() : tensor<2x4x64x1025xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity4, #identity4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%shared : tensor<2x4x64x1025xf32>)
        outs(%elementwise_empty : tensor<2x4x64x1025xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x4x64x1025xf32>
    %init_user = linalg.generic {
        indexing_maps = [#identity4, #identity4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%data : tensor<2x4x64x1025xf32>)
        outs(%shared : tensor<2x4x64x1025xf32>) {
      ^bb0(%element: f32, %output: f32):
        linalg.yield %element : f32
    } -> tensor<2x4x64x1025xf32>
    return %elementwise, %init_user :
        tensor<2x4x64x1025xf32>, tensor<2x4x64x1025xf32>
  }

  func.func @ragged_observable_fanout_barrier(
      %input: tensor<2x4x1025x64xf32>)
      -> (tensor<2x4x64x1025xf32>, tensor<2x4x64x1025xf32>) {
    %shared_empty = tensor.empty() : tensor<2x4x64x1025xf32>
    %shared = linalg.transpose
        ins(%input : tensor<2x4x1025x64xf32>)
        outs(%shared_empty : tensor<2x4x64x1025xf32>)
        permutation = [0, 1, 3, 2]
    %elementwise_empty = tensor.empty() : tensor<2x4x64x1025xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity4, #identity4],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%shared : tensor<2x4x64x1025xf32>)
        outs(%elementwise_empty : tensor<2x4x64x1025xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x4x64x1025xf32>
    return %shared, %elementwise :
        tensor<2x4x64x1025xf32>, tensor<2x4x64x1025xf32>
  }

  func.func @ragged_unsupported_contraction_fanout_barrier(
      %lhs: tensor<2x1031x64xf32>, %rhs: tensor<2x1031x128xf32>,
      %init: tensor<2x64x128xf32>)
      -> (tensor<2x64x1031xf32>, tensor<2x64x128xf32>) {
    %shared_empty = tensor.empty() : tensor<2x64x1031xf32>
    %shared = linalg.transpose
        ins(%lhs : tensor<2x1031x64xf32>)
        outs(%shared_empty : tensor<2x64x1031xf32>) permutation = [0, 2, 1]
    %elementwise_empty = tensor.empty() : tensor<2x64x1031xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%shared : tensor<2x64x1031xf32>)
        outs(%elementwise_empty : tensor<2x64x1031xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x1031xf32>
    %contracted = linalg.generic {
        indexing_maps = [#unsupported_lhs, #unsupported_rhs,
                         #unsupported_output],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%shared, %rhs :
            tensor<2x64x1031xf32>, tensor<2x1031x128xf32>)
        outs(%init : tensor<2x64x128xf32>) {
      ^bb0(%left: f32, %right: f32, %accumulator: f32):
        %product = arith.mulf %left, %right : f32
        %sum = arith.addf %accumulator, %product : f32
        linalg.yield %sum : f32
    } -> tensor<2x64x128xf32>
    return %elementwise, %contracted :
        tensor<2x64x1031xf32>, tensor<2x64x128xf32>
  }

  func.func @ragged_broadcast_contraction_chain(
      %lhs: tensor<1x64x1031xf32>, %rhs: tensor<2x1031x128xf32>,
      %init: tensor<2x64x128xf32>) -> tensor<2x128x64xf32> {
    %broadcast_empty = tensor.empty() : tensor<2x64x1031xf32>
    %broadcast = linalg.generic {
        indexing_maps = [#batch_broadcast_lhs, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%lhs : tensor<1x64x1031xf32>)
        outs(%broadcast_empty : tensor<2x64x1031xf32>) {
      ^bb0(%element: f32, %output: f32):
        linalg.yield %element : f32
    } -> tensor<2x64x1031xf32>
    %contracted = linalg.batch_matmul
        ins(%broadcast, %rhs :
            tensor<2x64x1031xf32>, tensor<2x1031x128xf32>)
        outs(%init : tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
    %elementwise_empty = tensor.empty() : tensor<2x64x128xf32>
    %elementwise = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%contracted : tensor<2x64x128xf32>)
        outs(%elementwise_empty : tensor<2x64x128xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x128xf32>
    %result_empty = tensor.empty() : tensor<2x128x64xf32>
    %result = linalg.transpose
        ins(%elementwise : tensor<2x64x128xf32>)
        outs(%result_empty : tensor<2x128x64xf32>) permutation = [0, 2, 1]
    return %result : tensor<2x128x64xf32>
  }

  func.func @ragged_general_reshape_elementwise(
      %input: tensor<2x4x1025x128xf32>) -> tensor<2x4x1025x128xf32> {
    %flat = tensor.collapse_shape %input [[0, 1], [2], [3]] :
        tensor<2x4x1025x128xf32> into tensor<8x1025x128xf32>
    %init = tensor.empty() : tensor<8x1025x128xf32>
    %computed = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%flat : tensor<8x1025x128xf32>)
        outs(%init : tensor<8x1025x128xf32>) {
      ^bb0(%element: f32, %output: f32):
        %negated = arith.negf %element : f32
        linalg.yield %negated : f32
    } -> tensor<8x1025x128xf32>
    %expanded = tensor.expand_shape %computed [[0, 1], [2], [3]]
        output_shape [2, 4, 1025, 128] :
        tensor<8x1025x128xf32> into tensor<2x4x1025x128xf32>
    return %expanded : tensor<2x4x1025x128xf32>
  }
}

// CHECK-LABEL: func.func @ragged_long_alternating_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK: arith.negf
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK: math.rsqrt
// CHECK: return %{{.*}} : tensor<2x1025x64xf32>

// CHECK-LABEL: func.func @ragged_two_elementwise_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK: arith.negf
// CHECK: linalg.generic
// CHECK: math.exp

// CHECK-LABEL: func.func @ragged_elementwise_reduction_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK: arith.negf
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.addf
// CHECK: return %{{.*}} : tensor<2x4x64xf32>

// CHECK-LABEL: func.func @ragged_contraction_elementwise_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK: linalg.batch_matmul
// CHECK: arith.negf
// CHECK: return %{{.*}} : tensor<2x64x128xf32>

// CHECK-LABEL: func.func @ragged_fanout_diamond_chain
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x1025x64xf32>)
// CHECK: arith.negf
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x1025x64xf32>)
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: return %{{.*}} : tensor<2x1025x64xf32>

// CHECK-LABEL: func.func @aligned_mixed_elementwise_reduction_fanout
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x4x1024x64xf32>)
// CHECK: arith.negf
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%arg0 : tensor<2x4x1024x64xf32>)
// CHECK: arith.addf
// CHECK: return %{{.*}}, %{{.*}} : tensor<2x4x64x1024xf32>, tensor<2x4x64xf32>

// CHECK-LABEL: func.func @ragged_mixed_elementwise_reduction_fanout
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x4x1025x64xf32>)
// CHECK: arith.negf
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%arg0 : tensor<2x4x1025x64xf32>)
// CHECK: arith.addf
// CHECK: return %{{.*}}, %{{.*}} : tensor<2x4x64x1025xf32>, tensor<2x4x64xf32>

// CHECK-LABEL: func.func @ragged_mixed_elementwise_contraction_fanout
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x64x1031xf32>)
// CHECK: arith.negf
// CHECK: linalg.batch_matmul_transpose_a
// CHECK-SAME: ins(%arg0, %arg1 : tensor<2x64x1031xf32>, tensor<2x64x128xf32>)
// CHECK: return %{{.*}}, %{{.*}} : tensor<2x1031x64xf32>, tensor<2x1031x128xf32>

// CHECK-LABEL: func.func @ragged_transient_init_use_chain
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x4x1025x64xf32>)
// CHECK: arith.negf
// CHECK: return %{{.*}}, %{{.*}} : tensor<2x4x64x1025xf32>, tensor<2x4x64x1025xf32>

// CHECK-LABEL: func.func @ragged_observable_fanout_barrier
// CHECK-COUNT-1: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}} : tensor<2x4x64x1025xf32>)
// CHECK: arith.negf
// CHECK: return %{{.*}}, %{{.*}} : tensor<2x4x64x1025xf32>, tensor<2x4x64x1025xf32>

// CHECK-LABEL: func.func @ragged_unsupported_contraction_fanout_barrier
// CHECK-COUNT-1: linalg.transpose
// CHECK: arith.negf
// CHECK: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: return %{{.*}}, %{{.*}} : tensor<2x64x1031xf32>, tensor<2x64x128xf32>

// CHECK-LABEL: func.func @ragged_broadcast_contraction_chain
// CHECK-NOT: linalg.transpose
// CHECK: linalg.batch_matmul
// CHECK: arith.negf
// CHECK: return %{{.*}} : tensor<2x128x64xf32>

// CHECK-LABEL: func.func @ragged_general_reshape_elementwise
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%arg0 : tensor<2x4x1025x128xf32>)
// CHECK: arith.negf
// CHECK: return %{{.*}} : tensor<2x4x1025x128xf32>

// STATS-DAG: (S) {{0+}} budget-exhausted-components
// STATS-DAG: (S) {{[1-9][0-9]*}} changed-components
// STATS-DAG: (S) {{[1-9][0-9]*}} multi-rule-changed-components
// STATS-DAG: (S) {{[1-9][0-9]*}} multi-root-components
// STATS-DAG: (S) {{[1-9][0-9]*}} access-transforms-removed
// STATS-DAG: (S) {{[1-9][0-9]*}} reshape-through-compute-applications
