// RUN: not wafer-opt %s 2>&1 | FileCheck %s

#q = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>
#k = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>
#v = affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>
#s = affine_map<(b, h, m, k1, k2, n) -> ()>
#acc = affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>
#bad_row = affine_map<(b, h, m, k1, k2, n) -> (b, h, n)>
#row = affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>

func.func @bad_state_map(
    %query: tensor<2x16x1025x128xf16>,
    %key: tensor<2x16x1031x128xf16>,
    %value: tensor<2x16x1031x64xf16>, %scale: f32,
    %accumulator: tensor<2x16x1025x64xf16>,
    %maximum: tensor<2x16x64xf32>, %sum: tensor<2x16x1025xf32>) {
  // CHECK: maximum and sum maps must equal the accumulator row map
  %next_accumulator, %next_maximum, %next_sum =
      wafer.linalg_ext.online_attention
      ins(%query, %key, %value, %scale : tensor<2x16x1025x128xf16>,
          tensor<2x16x1031x128xf16>, tensor<2x16x1031x64xf16>, f32)
      outs(%accumulator, %maximum, %sum : tensor<2x16x1025x64xf16>,
          tensor<2x16x64xf32>, tensor<2x16x1025xf32>)
      indexing_maps = [#q, #k, #v, #s, #acc, #bad_row, #row]
      -> (tensor<2x16x1025x64xf16>, tensor<2x16x64xf32>,
          tensor<2x16x1025xf32>)
  return
}
