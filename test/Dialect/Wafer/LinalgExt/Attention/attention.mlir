// RUN: wafer-opt %s | wafer-opt | FileCheck %s

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#mask = affine_map<(b, m, k1, k2, n) -> (b, m, k2)>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>

func.func @tensor_attention(
    %query: tensor<2x1024x64xf16>, %key: tensor<2x1031x64xf16>,
    %value: tensor<2x1031x128xf16>, %scale: f16,
    %mask_value: tensor<2x1024x1031xf16>) -> tensor<2x1024x128xf16> {
  %out = tensor.empty() : tensor<2x1024x128xf16>
  %result = wafer.linalg_ext.attention
      ins(%query, %key, %value, %scale, %mask_value :
          tensor<2x1024x64xf16>, tensor<2x1031x64xf16>, tensor<2x1031x128xf16>, f16,
          tensor<2x1024x1031xf16>)
      outs(%out : tensor<2x1024x128xf16>)
      algorithm(#wafer.attention_algorithm<flash_attention>)
      indexing_maps = [#q, #k, #v, #s, #mask, #o] score {
  ^bb0(%attention_0_dot: f32, %attention_0_scale: f16, %attention_0_mask: f16):
    %attention_0_scale_wide = arith.extf %attention_0_scale : f16 to f32
    %attention_0_mask_wide = arith.extf %attention_0_mask : f16 to f32
    %attention_0_scaled = arith.mulf %attention_0_dot, %attention_0_scale_wide : f32
    %attention_0_masked = arith.addf %attention_0_scaled, %attention_0_mask_wide : f32
    wafer.linalg_ext.attention.yield %attention_0_masked : f32
  }
      -> tensor<2x1024x128xf16>
  return %result : tensor<2x1024x128xf16>
}

// CHECK-LABEL: func.func @tensor_attention
// CHECK: wafer.linalg_ext.attention
// CHECK-SAME: algorithm(<flash_attention>)
// CHECK-SAME: indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5]

func.func @generic_flash_decoding(
    %query: tensor<2x1024x64xf16>, %key: tensor<2x1031x64xf16>,
    %value: tensor<2x1031x128xf16>, %scale: f16) -> tensor<2x1024x128xf16> {
  %out = tensor.empty() : tensor<2x1024x128xf16>
  %result = "wafer.linalg_ext.attention"(
      %query, %key, %value, %scale, %out) <{
        algorithm = #wafer.attention_algorithm<flash_decoding>,
        indexing_maps = [#q, #k, #v, #s, #o],
        operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>
      }> ({
      ^bb0(%dot: f32, %scale_arg: f16):
        %scale_wide = arith.extf %scale_arg : f16 to f32
        %scaled = arith.mulf %dot, %scale_wide : f32
        wafer.linalg_ext.attention.yield %scaled : f32
      }) : (tensor<2x1024x64xf16>, tensor<2x1031x64xf16>, tensor<2x1031x128xf16>,
            f16, tensor<2x1024x128xf16>) -> tensor<2x1024x128xf16>
  return %result : tensor<2x1024x128xf16>
}

// CHECK-LABEL: func.func @generic_flash_decoding
// CHECK: wafer.linalg_ext.attention
// CHECK-SAME: algorithm(<flash_decoding>)

func.func @buffer_attention(
    %query: memref<2x1024x64xf16>, %key: memref<2x1031x64xf16>,
    %value: memref<2x1031x128xf16>, %scale: f16,
    %out: memref<2x1024x128xf16>) {
  wafer.linalg_ext.attention
      ins(%query, %key, %value, %scale :
          memref<2x1024x64xf16>, memref<2x1031x64xf16>, memref<2x1031x128xf16>, f16)
      outs(%out : memref<2x1024x128xf16>)
      algorithm(#wafer.attention_algorithm<flash_attention>)
      indexing_maps = [#q, #k, #v, #s, #o] score {
  ^bb0(%attention_1_dot: f32, %attention_1_scale: f16):
    %attention_1_scale_wide = arith.extf %attention_1_scale : f16 to f32
    %attention_1_scaled = arith.mulf %attention_1_dot, %attention_1_scale_wide : f32
    wafer.linalg_ext.attention.yield %attention_1_scaled : f32
  }
  return
}

// CHECK-LABEL: func.func @buffer_attention
// CHECK: wafer.linalg_ext.attention
// CHECK-NOT: ->
