// RUN: wafer-opt %s --wafer-materialize-flash-decoding='output-tile-sizes=1,3 key-value-tile-size=3 split-count=2' | FileCheck %s
// RUN: sed 's/%%probability, %%updated_value/%%probability, %%key_out/' %s | not wafer-opt - --wafer-materialize-flash-decoding='output-tile-sizes=1,3 key-value-tile-size=3 split-count=2' 2>&1 | FileCheck %s --check-prefix=NOT-CONSUMED
// RUN: sed 's/return %%output, %%updated_key, %%updated_value/return %%output, %%updated_key, %%value_out/' %s | not wafer-opt - --wafer-materialize-flash-decoding='output-tile-sizes=1,3 key-value-tile-size=3 split-count=2' 2>&1 | FileCheck %s --check-prefix=NOT-RETURNED

module {
  func.func @functional_decode(
      %query: tensor<2x3xf16>,
      %past_key: tensor<5x3xf16>, %new_key: tensor<2x3xf16>,
      %past_value: tensor<5x3xf16>, %new_value: tensor<2x3xf16>,
      %out: tensor<2x3xf16>, %key_out: tensor<7x3xf16>,
      %value_out: tensor<7x3xf16>)
      -> (tensor<2x3xf16>, tensor<7x3xf16>, tensor<7x3xf16>) {
    %zero = arith.constant 0.0 : f16
    %lowest = arith.constant -6.550400e+04 : f16

    %key_prefix = tensor.insert_slice %past_key into %key_out[0, 0] [5, 3] [1, 1]
        : tensor<5x3xf16> into tensor<7x3xf16>
    %updated_key = tensor.insert_slice %new_key into %key_prefix[5, 0] [2, 3] [1, 1]
        : tensor<2x3xf16> into tensor<7x3xf16>
    %value_prefix = tensor.insert_slice %past_value into %value_out[0, 0] [5, 3] [1, 1]
        : tensor<5x3xf16> into tensor<7x3xf16>
    %updated_value = tensor.insert_slice %new_value into %value_prefix[5, 0] [2, 3] [1, 1]
        : tensor<2x3xf16> into tensor<7x3xf16>

    %key_transpose_empty = tensor.empty() : tensor<3x7xf16>
    %key_transposed = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%updated_key : tensor<7x3xf16>)
        outs(%key_transpose_empty : tensor<3x7xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<3x7xf16>
    %score_empty = tensor.empty() : tensor<2x7xf16>
    %score_init = linalg.fill ins(%zero : f16)
        outs(%score_empty : tensor<2x7xf16>) -> tensor<2x7xf16>
    %scores = linalg.matmul
        ins(%query, %key_transposed : tensor<2x3xf16>, tensor<3x7xf16>)
        outs(%score_init : tensor<2x7xf16>) -> tensor<2x7xf16>

    %max_empty = tensor.empty() : tensor<2xf16>
    %max_init = linalg.fill ins(%lowest : f16)
        outs(%max_empty : tensor<2xf16>) -> tensor<2xf16>
    %row_max = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%scores : tensor<2x7xf16>)
        outs(%max_init : tensor<2xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %next = arith.maximumf %acc, %value : f16
      linalg.yield %next : f16
    } -> tensor<2xf16>
    %max_broadcast_empty = tensor.empty() : tensor<2x7xf16>
    %max_broadcast = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%row_max : tensor<2xf16>)
        outs(%max_broadcast_empty : tensor<2x7xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<2x7xf16>
    %shift_empty = tensor.empty() : tensor<2x7xf16>
    %shifted = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%scores, %max_broadcast : tensor<2x7xf16>, tensor<2x7xf16>)
        outs(%shift_empty : tensor<2x7xf16>) {
    ^bb0(%value: f16, %maximum: f16, %unused: f16):
      %next = arith.subf %value, %maximum : f16
      linalg.yield %next : f16
    } -> tensor<2x7xf16>
    %exp_empty = tensor.empty() : tensor<2x7xf16>
    %exponential = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%shifted : tensor<2x7xf16>)
        outs(%exp_empty : tensor<2x7xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %next = math.exp %value : f16
      linalg.yield %next : f16
    } -> tensor<2x7xf16>
    %sum_empty = tensor.empty() : tensor<2xf16>
    %sum_init = linalg.fill ins(%zero : f16)
        outs(%sum_empty : tensor<2xf16>) -> tensor<2xf16>
    %row_sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%exponential : tensor<2x7xf16>)
        outs(%sum_init : tensor<2xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %next = arith.addf %acc, %value : f16
      linalg.yield %next : f16
    } -> tensor<2xf16>
    %sum_broadcast_empty = tensor.empty() : tensor<2x7xf16>
    %sum_broadcast = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%row_sum : tensor<2xf16>)
        outs(%sum_broadcast_empty : tensor<2x7xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<2x7xf16>
    %prob_empty = tensor.empty() : tensor<2x7xf16>
    %probability = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%exponential, %sum_broadcast : tensor<2x7xf16>, tensor<2x7xf16>)
        outs(%prob_empty : tensor<2x7xf16>) {
    ^bb0(%numerator: f16, %denominator: f16, %unused: f16):
      %next = arith.divf %numerator, %denominator : f16
      linalg.yield %next : f16
    } -> tensor<2x7xf16>
    %out_init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x3xf16>) -> tensor<2x3xf16>
    %output = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                         affine_map<(d0, d1, d2) -> (d2, d1)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%probability, %updated_value
          : tensor<2x7xf16>, tensor<7x3xf16>)
        outs(%out_init : tensor<2x3xf16>) {
    ^bb0(%prob: f16, %value: f16, %acc: f16):
      %product = arith.mulf %prob, %value : f16
      %next = arith.addf %acc, %product : f16
      linalg.yield %next : f16
    } -> tensor<2x3xf16>
    return %output, %updated_key, %updated_value
        : tensor<2x3xf16>, tensor<7x3xf16>, tensor<7x3xf16>
  }
}

// CHECK-LABEL: func.func @functional_decode
// No full score-shaped tensor may survive even as an otherwise data-less DPS
// destination: bufferization would still allocate it.
// CHECK-NOT: tensor<2x7xf16>
// Every fused score tile must retain the exact zero-fill DPS provenance. A
// slice of the original score result would both keep the full contraction
// alive and accumulate the score a second time.
// CHECK: %[[SCORE_INIT:.*]] = linalg.fill
// CHECK: linalg.matmul
// CHECK-SAME: outs(%[[SCORE_INIT]]
// CHECK-COUNT-2: math.log
// CHECK: arith.maximumf
// CHECK: return {{.*}} : tensor<2x3xf16>, tensor<7x3xf16>, tensor<7x3xf16>

// NOT-CONSUMED: flash_decoding_materialization_failed:
// NOT-CONSUMED-SAME: decode requires returned K/V appends consumed by attention
// NOT-RETURNED: flash_decoding_materialization_failed:
// NOT-RETURNED-SAME: decode requires returned K/V appends consumed by attention
