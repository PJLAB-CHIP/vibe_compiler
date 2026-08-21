// RUN: not wafer-opt %s 2>&1 | FileCheck %s

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#bad_k = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>

func.func @invalid_attention_roles(
    %query: tensor<2x3x4xf16>, %key: tensor<2x3x4xf16>,
    %value: tensor<2x5x6xf16>, %scale: f16) -> tensor<2x3x6xf16> {
  %out = tensor.empty() : tensor<2x3x6xf16>
  // CHECK: indexing maps must form complete B/M/K1/K2/N attention roles
  %result = wafer.linalg_ext.attention
      ins(%query, %key, %value, %scale :
          tensor<2x3x4xf16>, tensor<2x3x4xf16>, tensor<2x5x6xf16>, f16)
      outs(%out : tensor<2x3x6xf16>)
      algorithm(#wafer.attention_algorithm<flash_attention>)
      indexing_maps = [#q, #bad_k, #v, #s, #o]
      -> tensor<2x3x6xf16>
  return %result : tensor<2x3x6xf16>
}
