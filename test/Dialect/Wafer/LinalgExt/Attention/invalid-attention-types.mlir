// RUN: not wafer-opt %s 2>&1 | FileCheck %s

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>

func.func @invalid_attention_types(
    %query: tensor<2x3x4xf16>, %key: tensor<2x5x4xf32>,
    %value: tensor<2x5x6xf16>, %scale: f32) -> tensor<2x3x6xf16> {
  %out = tensor.empty() : tensor<2x3x6xf16>
  // CHECK: query, key, value, and output must share one floating storage type
  %result = wafer.linalg_ext.attention
      ins(%query, %key, %value, %scale :
          tensor<2x3x4xf16>, tensor<2x5x4xf32>, tensor<2x5x6xf16>, f32)
      outs(%out : tensor<2x3x6xf16>)
      algorithm(<flash_attention>)
      indexing_maps = [#q, #k, #v, #s, #o]
      -> tensor<2x3x6xf16>
  return %result : tensor<2x3x6xf16>
}
