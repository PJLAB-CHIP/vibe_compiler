// RUN: wafer-opt %s | wafer-opt | FileCheck %s

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (b, m, k2)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>

func.func @tensor_attention(
    %query: tensor<2x3x4xf16>, %key: tensor<2x5x4xf16>,
    %value: tensor<2x5x6xf16>, %scale: f16,
    %mask_value: tensor<2x3x5xf16>) -> tensor<2x3x6xf16> {
  %out = tensor.empty() : tensor<2x3x6xf16>
  %result = wafer.linalg_ext.attention
      ins(%query, %key, %value, %scale, %mask_value :
          tensor<2x3x4xf16>, tensor<2x5x4xf16>, tensor<2x5x6xf16>, f16,
          tensor<2x3x5xf16>)
      outs(%out : tensor<2x3x6xf16>)
      algorithm(#wafer.attention_algorithm<flash_attention>)
      indexing_maps = [#q, #k, #v, #s, #mask, #o] score {
  ^bb0(%attention_0_dot: f16, %attention_0_scale: f16, %attention_0_mask: f16):
    %attention_0_scaled = arith.mulf %attention_0_dot, %attention_0_scale : f16
    %attention_0_masked = arith.addf %attention_0_scaled, %attention_0_mask : f16
    wafer.linalg_ext.attention.yield %attention_0_masked : f16
  }
      -> tensor<2x3x6xf16>
  return %result : tensor<2x3x6xf16>
}

// CHECK-LABEL: func.func @tensor_attention
// CHECK: wafer.linalg_ext.attention
// CHECK-SAME: algorithm(<flash_attention>)
// CHECK-SAME: indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5]

func.func @generic_flash_decoding(
    %query: tensor<2x3x4xf16>, %key: tensor<2x5x4xf16>,
    %value: tensor<2x5x6xf16>, %scale: f16) -> tensor<2x3x6xf16> {
  %out = tensor.empty() : tensor<2x3x6xf16>
  %result = "wafer.linalg_ext.attention"(
      %query, %key, %value, %scale, %out) <{
        algorithm = #wafer.attention_algorithm<flash_decoding>,
        indexing_maps = [#q, #k, #v, #s, #o]
      }> ({
      ^bb0(%dot: f16, %scale_arg: f16):
        %scaled = arith.mulf %dot, %scale_arg : f16
        wafer.linalg_ext.attention.yield %scaled : f16
      }) : (tensor<2x3x4xf16>, tensor<2x5x4xf16>, tensor<2x5x6xf16>,
            f16, tensor<2x3x6xf16>) -> tensor<2x3x6xf16>
  return %result : tensor<2x3x6xf16>
}

// CHECK-LABEL: func.func @generic_flash_decoding
// CHECK: wafer.linalg_ext.attention
// CHECK-SAME: algorithm(<flash_decoding>)

func.func @buffer_attention(
    %query: memref<2x3x4xf16>, %key: memref<2x5x4xf16>,
    %value: memref<2x5x6xf16>, %scale: f16,
    %out: memref<2x3x6xf16>) {
  wafer.linalg_ext.attention
      ins(%query, %key, %value, %scale :
          memref<2x3x4xf16>, memref<2x5x4xf16>, memref<2x5x6xf16>, f16)
      outs(%out : memref<2x3x6xf16>)
      algorithm(#wafer.attention_algorithm<flash_attention>)
      indexing_maps = [#q, #k, #v, #s, #o] score {
  ^bb0(%attention_1_dot: f16, %attention_1_scale: f16):
    %attention_1_scaled = arith.mulf %attention_1_dot, %attention_1_scale : f16
    wafer.linalg_ext.attention.yield %attention_1_scaled : f16
  }
  return
}

// CHECK-LABEL: func.func @buffer_attention
// CHECK: wafer.linalg_ext.attention
// CHECK-NOT: ->
