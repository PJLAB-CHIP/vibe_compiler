module {
  func.func @functional_decode(
      %query: tensor<2x1024x128xf16>,
      %past_key: tensor<2x1030x128xf16>,
      %new_key: tensor<2x1x128xf16>,
      %past_value: tensor<2x1030x64xf16>,
      %new_value: tensor<2x1x64xf16>,
      %key_out: tensor<2x1031x128xf16>,
      %value_out: tensor<2x1031x64xf16>)
      -> (tensor<2x1024x64xf16>, tensor<2x1031x128xf16>,
          tensor<2x1031x64xf16>) {
    %zero = arith.constant 0.0 : f16
    %lowest = arith.constant -6.550400e+04 : f16

    %key_prefix = tensor.insert_slice %past_key into %key_out[0, 0, 0]
        [2, 1030, 128] [1, 1, 1]
        : tensor<2x1030x128xf16> into tensor<2x1031x128xf16>
    %updated_key = tensor.insert_slice %new_key into %key_prefix[0, 1030, 0]
        [2, 1, 128] [1, 1, 1]
        : tensor<2x1x128xf16> into tensor<2x1031x128xf16>
    %value_prefix = tensor.insert_slice %past_value into %value_out[0, 0, 0]
        [2, 1030, 64] [1, 1, 1]
        : tensor<2x1030x64xf16> into tensor<2x1031x64xf16>
    %updated_value = tensor.insert_slice %new_value into %value_prefix[0, 1030, 0]
        [2, 1, 64] [1, 1, 1]
        : tensor<2x1x64xf16> into tensor<2x1031x64xf16>

    %key_transpose_empty = tensor.empty() : tensor<2x128x1031xf16>
    %key_transposed = linalg.generic {
        indexing_maps = [affine_map<(b, k, t) -> (b, t, k)>,
                         affine_map<(b, k, t) -> (b, k, t)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%updated_key : tensor<2x1031x128xf16>)
        outs(%key_transpose_empty : tensor<2x128x1031xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<2x128x1031xf16>

    %score_empty = tensor.empty() : tensor<2x1024x1031xf16>
    %score_init = linalg.fill ins(%zero : f16)
        outs(%score_empty : tensor<2x1024x1031xf16>)
        -> tensor<2x1024x1031xf16>
    %scores = linalg.generic {
        indexing_maps = [affine_map<(b, m, t, k) -> (b, m, k)>,
                         affine_map<(b, m, t, k) -> (b, k, t)>,
                         affine_map<(b, m, t, k) -> (b, m, t)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%query, %key_transposed
          : tensor<2x1024x128xf16>, tensor<2x128x1031xf16>)
        outs(%score_init : tensor<2x1024x1031xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
      %product = arith.mulf %lhs, %rhs : f16
      %next = arith.addf %acc, %product : f16
      linalg.yield %next : f16
    } -> tensor<2x1024x1031xf16>

    %max_empty = tensor.empty() : tensor<2x1024xf16>
    %max_init = linalg.fill ins(%lowest : f16)
        outs(%max_empty : tensor<2x1024xf16>) -> tensor<2x1024xf16>
    %row_max = linalg.generic {
        indexing_maps = [affine_map<(b, m, t) -> (b, m, t)>,
                         affine_map<(b, m, t) -> (b, m)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%scores : tensor<2x1024x1031xf16>)
        outs(%max_init : tensor<2x1024xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %next = arith.maximumf %acc, %value : f16
      linalg.yield %next : f16
    } -> tensor<2x1024xf16>
    %max_broadcast_empty = tensor.empty() : tensor<2x1024x1031xf16>
    %max_broadcast = linalg.generic {
        indexing_maps = [affine_map<(b, m, t) -> (b, m)>,
                         affine_map<(b, m, t) -> (b, m, t)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%row_max : tensor<2x1024xf16>)
        outs(%max_broadcast_empty : tensor<2x1024x1031xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<2x1024x1031xf16>
    %shift_empty = tensor.empty() : tensor<2x1024x1031xf16>
    %shifted = linalg.generic {
        indexing_maps = [affine_map<(b, m, t) -> (b, m, t)>,
                         affine_map<(b, m, t) -> (b, m, t)>,
                         affine_map<(b, m, t) -> (b, m, t)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%scores, %max_broadcast
          : tensor<2x1024x1031xf16>, tensor<2x1024x1031xf16>)
        outs(%shift_empty : tensor<2x1024x1031xf16>) {
    ^bb0(%value: f16, %maximum: f16, %unused: f16):
      %next = arith.subf %value, %maximum : f16
      linalg.yield %next : f16
    } -> tensor<2x1024x1031xf16>
    %exp_empty = tensor.empty() : tensor<2x1024x1031xf16>
    %exponential = linalg.generic {
        indexing_maps = [affine_map<(b, m, t) -> (b, m, t)>,
                         affine_map<(b, m, t) -> (b, m, t)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%shifted : tensor<2x1024x1031xf16>)
        outs(%exp_empty : tensor<2x1024x1031xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %next = math.exp %value : f16
      linalg.yield %next : f16
    } -> tensor<2x1024x1031xf16>

    %sum_empty = tensor.empty() : tensor<2x1024xf16>
    %sum_init = linalg.fill ins(%zero : f16)
        outs(%sum_empty : tensor<2x1024xf16>) -> tensor<2x1024xf16>
    %row_sum = linalg.generic {
        indexing_maps = [affine_map<(b, m, t) -> (b, m, t)>,
                         affine_map<(b, m, t) -> (b, m)>],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%exponential : tensor<2x1024x1031xf16>)
        outs(%sum_init : tensor<2x1024xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %next = arith.addf %acc, %value : f16
      linalg.yield %next : f16
    } -> tensor<2x1024xf16>
    %sum_broadcast_empty = tensor.empty() : tensor<2x1024x1031xf16>
    %sum_broadcast = linalg.generic {
        indexing_maps = [affine_map<(b, m, t) -> (b, m)>,
                         affine_map<(b, m, t) -> (b, m, t)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%row_sum : tensor<2x1024xf16>)
        outs(%sum_broadcast_empty : tensor<2x1024x1031xf16>) {
    ^bb0(%value: f16, %unused: f16):
      linalg.yield %value : f16
    } -> tensor<2x1024x1031xf16>
    %prob_empty = tensor.empty() : tensor<2x1024x1031xf16>
    %probability = linalg.generic {
        indexing_maps = [affine_map<(b, m, t) -> (b, m, t)>,
                         affine_map<(b, m, t) -> (b, m, t)>,
                         affine_map<(b, m, t) -> (b, m, t)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%exponential, %sum_broadcast
          : tensor<2x1024x1031xf16>, tensor<2x1024x1031xf16>)
        outs(%prob_empty : tensor<2x1024x1031xf16>) {
    ^bb0(%numerator: f16, %denominator: f16, %unused: f16):
      %next = arith.divf %numerator, %denominator : f16
      linalg.yield %next : f16
    } -> tensor<2x1024x1031xf16>

    %out_empty = tensor.empty() : tensor<2x1024x64xf16>
    %out_init = linalg.fill ins(%zero : f16)
        outs(%out_empty : tensor<2x1024x64xf16>) -> tensor<2x1024x64xf16>
    %output = linalg.generic {
        indexing_maps = [affine_map<(b, m, n, t) -> (b, m, t)>,
                         affine_map<(b, m, n, t) -> (b, t, n)>,
                         affine_map<(b, m, n, t) -> (b, m, n)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
      } ins(%probability, %updated_value
          : tensor<2x1024x1031xf16>, tensor<2x1031x64xf16>)
        outs(%out_init : tensor<2x1024x64xf16>) {
    ^bb0(%prob: f16, %value: f16, %acc: f16):
      %product = arith.mulf %prob, %value : f16
      %next = arith.addf %acc, %product : f16
      linalg.yield %next : f16
    } -> tensor<2x1024x64xf16>
    return %output, %updated_key, %updated_value
        : tensor<2x1024x64xf16>, tensor<2x1031x128xf16>,
          tensor<2x1031x64xf16>
  }
}
