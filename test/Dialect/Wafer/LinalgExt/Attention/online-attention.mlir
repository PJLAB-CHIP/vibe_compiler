// RUN: wafer-opt %s | wafer-opt | FileCheck %s

#q = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>
#k = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>
#v = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>
#s = affine_map<(b, h, m, k1, k2, n) -> ()>
#mask = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k2)>
#acc = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>
#row = affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>

func.func @online_attention(
    %query: tensor<2x16x1025x128xf16>,
    %key: tensor<2x16x1031x128xf16>,
    %value: tensor<2x16x1031x64xf16>,
    %scale: f32,
    %mask_value: tensor<2x16x1025x1031xf16>,
    %accumulator: tensor<2x16x1025x64xf16>,
    %maximum: tensor<2x16x1025xf32>,
    %sum: tensor<2x16x1025xf32>)
    -> (tensor<2x16x1025x64xf16>, tensor<2x16x1025xf32>,
        tensor<2x16x1025xf32>) {
  %next_accumulator, %next_maximum, %next_sum =
      wafer.linalg_ext.online_attention
      ins(%query, %key, %value, %scale, %mask_value :
          tensor<2x16x1025x128xf16>, tensor<2x16x1031x128xf16>,
          tensor<2x16x1031x64xf16>, f32,
          tensor<2x16x1025x1031xf16>)
      outs(%accumulator, %maximum, %sum :
          tensor<2x16x1025x64xf16>, tensor<2x16x1025xf32>,
          tensor<2x16x1025xf32>)
      indexing_maps = [#q, #k, #v, #s, #mask, #acc, #row, #row] score {
      ^bb0(%attention_0_dot: f16, %attention_0_scale: f32, %attention_0_mask: f16):
        %attention_0_converted = arith.extf %attention_0_dot : f16 to f32
        %attention_0_scaled = arith.mulf %attention_0_converted, %attention_0_scale : f32
        %attention_0_converted_mask = arith.extf %attention_0_mask : f16 to f32
        %attention_0_masked = arith.addf %attention_0_scaled, %attention_0_converted_mask : f32
        wafer.linalg_ext.attention.yield %attention_0_masked : f32
      }
      -> (tensor<2x16x1025x64xf16>, tensor<2x16x1025xf32>,
          tensor<2x16x1025xf32>)
  return %next_accumulator, %next_maximum, %next_sum :
      tensor<2x16x1025x64xf16>, tensor<2x16x1025xf32>,
      tensor<2x16x1025xf32>
}

// CHECK-LABEL: func.func @online_attention
// CHECK: wafer.linalg_ext.online_attention
// CHECK-SAME: indexing_maps = [#map, #map1, #map2, #map3, #map4, #map5, #map6, #map6]
// CHECK: return
