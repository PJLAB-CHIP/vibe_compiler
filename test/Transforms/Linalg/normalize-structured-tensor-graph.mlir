// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %s -o %t.once
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %t.once -o %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#reduce_result = affine_map<(d0, d1, d2) -> (d0, d2)>
#reduce_last_result = affine_map<(d0, d1, d2) -> (d0, d1)>
#reduce_last_result4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#broadcast_input = affine_map<(d0, d1, d2) -> (d0, d2)>
#batch_lhs = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#batch_rhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#batch_output_transposed = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
#attention_q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#attention_k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#attention_v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#attention_s = affine_map<(b, m, k1, k2, n) -> ()>
#attention_o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>

module {
  func.func private @opaque_call(tensor<2x1025x64xf32>)
      -> tensor<2x1025x64xf32>

  // The aligned case covers exact inverse reshape elimination at rank 3.
  func.func @aligned_inverse_reshape(
      %arg0: tensor<2x1024x64xf32>) -> tensor<2x1024x64xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] :
        tensor<2x1024x64xf32> into tensor<2048x64xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1], [2]]
        output_shape [2, 1024, 64] :
        tensor<2048x64xf32> into tensor<2x1024x64xf32>
    return %expanded : tensor<2x1024x64xf32>
  }

  // The ragged case requires an exact access-map rewrite, not a shape policy.
  func.func @ragged_transpose_elementwise(
      %arg0: tensor<2x1025x64xf32>) -> tensor<2x64x1025xf32> {
    %transpose_init = tensor.empty() : tensor<2x64x1025xf32>
    %transposed = linalg.transpose
        ins(%arg0 : tensor<2x1025x64xf32>)
        outs(%transpose_init : tensor<2x64x1025xf32>)
        permutation = [0, 2, 1]
    %result_init = tensor.empty() : tensor<2x64x1025xf32>
    %result = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%transposed : tensor<2x64x1025xf32>)
        outs(%result_init : tensor<2x64x1025xf32>) {
      ^bb0(%input: f32, %output: f32):
        %negated = arith.negf %input : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x1025xf32>
    return %result : tensor<2x64x1025xf32>
  }

  // Absorbing the transpose must retain the exact batched GEMM payload and
  // one ordered reduction iterator for the downstream generic GEMM path.
  func.func @aligned_batch_matmul_transpose(
      %lhs: tensor<2x1024x64xf32>,
      %rhs: tensor<2x1024x128xf32>) -> tensor<2x64x128xf32> {
    %transpose_init = tensor.empty() : tensor<2x64x1024xf32>
    %transposed = linalg.transpose
        ins(%lhs : tensor<2x1024x64xf32>)
        outs(%transpose_init : tensor<2x64x1024xf32>)
        permutation = [0, 2, 1]
    %result_init = tensor.empty() : tensor<2x64x128xf32>
    %result = linalg.batch_matmul
        ins(%transposed, %rhs :
            tensor<2x64x1024xf32>, tensor<2x1024x128xf32>)
        outs(%result_init : tensor<2x64x128xf32>) -> tensor<2x64x128xf32>
    return %result : tensor<2x64x128xf32>
  }

  // A parallel-axis transpose may be absorbed without changing the ragged
  // reduction domain, combiner, init, or iterator order.
  func.func @ragged_reduction_transpose(
      %arg0: tensor<2x64x1031xf32>) -> tensor<2x64xf32> {
    %transpose_init = tensor.empty() : tensor<2x1031x64xf32>
    %transposed = linalg.transpose
        ins(%arg0 : tensor<2x64x1031xf32>)
        outs(%transpose_init : tensor<2x1031x64xf32>)
        permutation = [0, 2, 1]
    %result_init = tensor.empty() : tensor<2x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32)
        outs(%result_init : tensor<2x64xf32>) -> tensor<2x64xf32>
    %result = linalg.generic {
        indexing_maps = [#identity3, #reduce_result],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%transposed : tensor<2x1031x64xf32>)
        outs(%filled : tensor<2x64xf32>) {
      ^bb0(%input: f32, %accumulator: f32):
        %sum = arith.addf %accumulator, %input : f32
        linalg.yield %sum : f32
    } -> tensor<2x64xf32>
    return %result : tensor<2x64xf32>
  }

  // A broadcast feeding a reduction is intentionally retained because
  // absorbing it would hide reduction multiplicity behind a projected map.
  func.func @ragged_reduction_broadcast_barrier(
      %arg0: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %broadcast_init = tensor.empty() : tensor<2x1031x64xf32>
    %broadcast = linalg.generic {
        indexing_maps = [#broadcast_input, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<2x64xf32>)
        outs(%broadcast_init : tensor<2x1031x64xf32>) {
      ^bb0(%input: f32, %output: f32):
        linalg.yield %input : f32
    } -> tensor<2x1031x64xf32>
    %result_init = tensor.empty() : tensor<2x64xf32>
    %zero = arith.constant 0.0 : f32
    %filled = linalg.fill ins(%zero : f32)
        outs(%result_init : tensor<2x64xf32>) -> tensor<2x64xf32>
    %result = linalg.generic {
        indexing_maps = [#identity3, #reduce_result],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%broadcast : tensor<2x1031x64xf32>)
        outs(%filled : tensor<2x64xf32>) {
      ^bb0(%input: f32, %accumulator: f32):
        %sum = arith.addf %accumulator, %input : f32
        linalg.yield %sum : f32
    } -> tensor<2x64xf32>
    return %result : tensor<2x64xf32>
  }

  // Nested canonical concat assemblies are flattened to one exact ordered
  // chain. 1025 exercises the final one-element tail.
  func.func @ragged_nested_concat(
      %first: tensor<2x512x64xf32>,
      %second: tensor<2x512x64xf32>,
      %tail: tensor<2x1x64xf32>) -> tensor<2x1025x64xf32> {
    %inner_init = tensor.empty() : tensor<2x1024x64xf32>
    %inner_first = tensor.insert_slice %first into %inner_init[0, 0, 0]
        [2, 512, 64] [1, 1, 1] :
        tensor<2x512x64xf32> into tensor<2x1024x64xf32>
    %inner_second = tensor.insert_slice %second into %inner_first[0, 512, 0]
        [2, 512, 64] [1, 1, 1] :
        tensor<2x512x64xf32> into tensor<2x1024x64xf32>
    %outer_init = tensor.empty() : tensor<2x1025x64xf32>
    %outer_main = tensor.insert_slice %inner_second into %outer_init[0, 0, 0]
        [2, 1024, 64] [1, 1, 1] :
        tensor<2x1024x64xf32> into tensor<2x1025x64xf32>
    %outer_tail = tensor.insert_slice %tail into %outer_main[0, 1024, 0]
        [2, 1, 64] [1, 1, 1] :
        tensor<2x1x64xf32> into tensor<2x1025x64xf32>
    return %outer_tail : tensor<2x1025x64xf32>
  }

  // A gap means this is not a canonical concat assembly and must remain an
  // ordinary insert_slice chain.
  func.func @ragged_concat_gap(
      %first: tensor<2x512x64xf32>, %second: tensor<2x512x64xf32>)
      -> tensor<2x1025x64xf32> {
    %init = tensor.empty() : tensor<2x1025x64xf32>
    %inserted_first = tensor.insert_slice %first into %init[0, 0, 0]
        [2, 512, 64] [1, 1, 1] :
        tensor<2x512x64xf32> into tensor<2x1025x64xf32>
    %inserted_second = tensor.insert_slice %second into %inserted_first
        [0, 513, 0] [2, 512, 64] [1, 1, 1] :
        tensor<2x512x64xf32> into tensor<2x1025x64xf32>
    return %inserted_second : tensor<2x1025x64xf32>
  }

  // Identical segment transposes are extracted through concat. The shared
  // CSE'd init must also become dead after both old transpose users disappear.
  func.func @aligned_concat_transpose_extraction(
      %first: tensor<2x512x64xf32>,
      %second: tensor<2x512x64xf32>) -> tensor<1024x2x64xf32> {
    %first_init = tensor.empty() : tensor<512x2x64xf32>
    %first_transposed = linalg.transpose
        ins(%first : tensor<2x512x64xf32>)
        outs(%first_init : tensor<512x2x64xf32>)
        permutation = [1, 0, 2]
    %second_init = tensor.empty() : tensor<512x2x64xf32>
    %second_transposed = linalg.transpose
        ins(%second : tensor<2x512x64xf32>)
        outs(%second_init : tensor<512x2x64xf32>)
        permutation = [1, 0, 2]
    %concat_init = tensor.empty() : tensor<1024x2x64xf32>
    %concat_first = tensor.insert_slice %first_transposed into %concat_init
        [0, 0, 0] [512, 2, 64] [1, 1, 1] :
        tensor<512x2x64xf32> into tensor<1024x2x64xf32>
    %concat_second = tensor.insert_slice %second_transposed into %concat_first
        [512, 0, 0] [512, 2, 64] [1, 1, 1] :
        tensor<512x2x64xf32> into tensor<1024x2x64xf32>
    return %concat_second : tensor<1024x2x64xf32>
  }

  // Result reindex keeps the original iteration domain and scalar region,
  // while redirecting the output map to the transposed result coordinates.
  func.func @ragged_elementwise_result_reindex(
      %arg0: tensor<2x1025x64xf32>) -> tensor<2x64x1025xf32> {
    %compute_init = tensor.empty() : tensor<2x1025x64xf32>
    %computed = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<2x1025x64xf32>)
        outs(%compute_init : tensor<2x1025x64xf32>) {
      ^bb0(%input: f32, %output: f32):
        %negated = arith.negf %input : f32
        linalg.yield %negated : f32
    } -> tensor<2x1025x64xf32>
    %transpose_init = tensor.empty() : tensor<2x64x1025xf32>
    %transposed = linalg.transpose
        ins(%computed : tensor<2x1025x64xf32>)
        outs(%transpose_init : tensor<2x64x1025xf32>)
        permutation = [0, 2, 1]
    return %transposed : tensor<2x64x1025xf32>
  }

  // The read DPS init is reindexed explicitly. The inverse init/result
  // transposes then disappear while the last reduction axis stays unchanged.
  func.func @ragged_reduction_result_reindex(
      %input: tensor<2x4x64x1031xf32>, %init: tensor<2x64x4xf32>)
      -> tensor<2x64x4xf32> {
    %init_transpose_empty = tensor.empty() : tensor<2x4x64xf32>
    %init_transposed = linalg.transpose
        ins(%init : tensor<2x64x4xf32>)
        outs(%init_transpose_empty : tensor<2x4x64xf32>)
        permutation = [0, 2, 1]
    %reduced = linalg.generic {
        indexing_maps = [#identity4, #reduce_last_result4],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%input : tensor<2x4x64x1031xf32>)
        outs(%init_transposed : tensor<2x4x64xf32>) {
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %accumulator, %element : f32
        linalg.yield %sum : f32
    } -> tensor<2x4x64xf32>
    %result_empty = tensor.empty() : tensor<2x64x4xf32>
    %result = linalg.transpose
        ins(%reduced : tensor<2x4x64xf32>)
        outs(%result_empty : tensor<2x64x4xf32>)
        permutation = [0, 2, 1]
    return %result : tensor<2x64x4xf32>
  }

  // The original contraction stores a transposed result. Reindexing its
  // result and read init recovers the canonical existing batch_matmul form.
  func.func @ragged_batch_contraction_result_reindex(
      %lhs: tensor<2x64x1031xf32>, %rhs: tensor<2x1031x128xf32>,
      %init: tensor<2x64x128xf32>) -> tensor<2x64x128xf32> {
    %init_empty = tensor.empty() : tensor<2x128x64xf32>
    %init_transposed = linalg.transpose
        ins(%init : tensor<2x64x128xf32>)
        outs(%init_empty : tensor<2x128x64xf32>)
        permutation = [0, 2, 1]
    %contracted = linalg.generic {
        indexing_maps = [#batch_lhs, #batch_rhs,
                         #batch_output_transposed],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<2x64x1031xf32>, tensor<2x1031x128xf32>)
        outs(%init_transposed : tensor<2x128x64xf32>) {
      ^bb0(%left: f32, %right: f32, %accumulator: f32):
        %product = arith.mulf %left, %right : f32
        %sum = arith.addf %accumulator, %product : f32
        linalg.yield %sum : f32
    } -> tensor<2x128x64xf32>
    %result_empty = tensor.empty() : tensor<2x64x128xf32>
    %result = linalg.transpose
        ins(%contracted : tensor<2x128x64xf32>)
        outs(%result_empty : tensor<2x64x128xf32>)
        permutation = [0, 2, 1]
    return %result : tensor<2x64x128xf32>
  }

  // Calls are component barriers. The ordinary graph after the call may be
  // normalized, but no rule can match through the call boundary.
  func.func @call_barrier(
      %arg0: tensor<2x1025x64xf32>) -> tensor<2x64x1025xf32> {
    %called = func.call @opaque_call(%arg0)
        : (tensor<2x1025x64xf32>) -> tensor<2x1025x64xf32>
    %transpose_init = tensor.empty() : tensor<2x64x1025xf32>
    %transposed = linalg.transpose
        ins(%called : tensor<2x1025x64xf32>)
        outs(%transpose_init : tensor<2x64x1025xf32>)
        permutation = [0, 2, 1]
    %result_init = tensor.empty() : tensor<2x64x1025xf32>
    %result = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%transposed : tensor<2x64x1025xf32>)
        outs(%result_init : tensor<2x64x1025xf32>) {
      ^bb0(%input: f32, %output: f32):
        %negated = arith.negf %input : f32
        linalg.yield %negated : f32
    } -> tensor<2x64x1025xf32>
    return %result : tensor<2x64x1025xf32>
  }

  // Attention is an opaque input to the downstream ordinary component. The
  // transpose may be absorbed, but the semantic op itself is not imported.
  func.func @attention_opaque_downstream(
      %query: tensor<2x1025x64xf16>, %key: tensor<2x1031x64xf16>,
      %value: tensor<2x1031x128xf16>, %scale: f16)
      -> tensor<2x128x1025xf16> {
    %attention_init = tensor.empty() : tensor<2x1025x128xf16>
    %attention = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale :
            tensor<2x1025x64xf16>, tensor<2x1031x64xf16>,
            tensor<2x1031x128xf16>, f16)
        outs(%attention_init : tensor<2x1025x128xf16>)
        algorithm(#wafer.attention_algorithm<flash_attention>)
        indexing_maps = [#attention_q, #attention_k, #attention_v,
                         #attention_s, #attention_o] score {
    ^bb0(%attention_0_dot: f16, %attention_0_scale: f16):
      %attention_0_scaled = arith.mulf %attention_0_dot, %attention_0_scale : f16
      wafer.linalg_ext.attention.yield %attention_0_scaled : f16
    }
        -> tensor<2x1025x128xf16>
    %transpose_init = tensor.empty() : tensor<2x128x1025xf16>
    %transposed = linalg.transpose
        ins(%attention : tensor<2x1025x128xf16>)
        outs(%transpose_init : tensor<2x128x1025xf16>)
        permutation = [0, 2, 1]
    %result_init = tensor.empty() : tensor<2x128x1025xf16>
    %result = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%transposed : tensor<2x128x1025xf16>)
        outs(%result_init : tensor<2x128x1025xf16>) {
      ^bb0(%input: f16, %output: f16):
        %negated = arith.negf %input : f16
        linalg.yield %negated : f16
    } -> tensor<2x128x1025xf16>
    return %result : tensor<2x128x1025xf16>
  }
}

// CHECK-LABEL: func.func @aligned_inverse_reshape
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK: return %arg0 : tensor<2x1024x64xf32>

// CHECK-LABEL: func.func @ragged_transpose_elementwise
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x1025x64xf32>)
// CHECK: arith.negf

// CHECK-LABEL: func.func @aligned_batch_matmul_transpose
// CHECK-NOT: linalg.transpose
// CHECK: linalg.batch_matmul_transpose_a
// CHECK-SAME: ins(%arg0, %arg1 : tensor<2x1024x64xf32>, tensor<2x1024x128xf32>)

// CHECK-LABEL: func.func @ragged_reduction_transpose
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%arg0 : tensor<2x64x1031xf32>)
// CHECK: arith.addf

// CHECK-LABEL: func.func @ragged_reduction_broadcast_barrier
// CHECK: %[[BROADCAST:.*]] = linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x64xf32>)
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[BROADCAST]] : tensor<2x1031x64xf32>)

// CHECK-LABEL: func.func @ragged_nested_concat
// CHECK-COUNT-3: tensor.insert_slice
// CHECK-SAME: %arg2
// CHECK-SAME: [0, 1024, 0] [2, 1, 64]

// CHECK-LABEL: func.func @ragged_concat_gap
// CHECK-COUNT-2: tensor.insert_slice
// CHECK: [0, 513, 0] [2, 512, 64]

// CHECK-LABEL: func.func @aligned_concat_transpose_extraction
// CHECK-NOT: linalg.transpose
// CHECK-NOT: tensor<512x2x64xf32>
// CHECK-COUNT-2: tensor.insert_slice
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}} : tensor<2x1024x64xf32>)
// CHECK: return %{{.*}} : tensor<1024x2x64xf32>

// CHECK-LABEL: func.func @ragged_elementwise_result_reindex
// CHECK-NOT: linalg.transpose
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: ins(%arg0 : tensor<2x1025x64xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<2x64x1025xf32>)
// CHECK: arith.negf

// CHECK-LABEL: func.func @ragged_reduction_result_reindex
// CHECK-NOT: linalg.transpose
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%arg0 : tensor<2x4x64x1031xf32>)
// CHECK-SAME: outs(%arg1 : tensor<2x64x4xf32>)
// CHECK: arith.addf

// CHECK-LABEL: func.func @ragged_batch_contraction_result_reindex
// CHECK-NOT: linalg.transpose
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_matmul
// CHECK-SAME: ins(%arg0, %arg1 : tensor<2x64x1031xf32>, tensor<2x1031x128xf32>)
// CHECK-SAME: outs(%arg2 : tensor<2x64x128xf32>)

// CHECK-LABEL: func.func @call_barrier
// CHECK-COUNT-1: call @opaque_call
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}} : tensor<2x1025x64xf32>)

// CHECK-LABEL: func.func @attention_opaque_downstream
// CHECK-COUNT-1: wafer.linalg_ext.attention
// CHECK-SAME: algorithm(<flash_attention>)
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}} : tensor<2x1025x128xf16>)
// CHECK: arith.negf
