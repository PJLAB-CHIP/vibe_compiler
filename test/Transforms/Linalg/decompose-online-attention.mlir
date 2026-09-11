// RUN: wafer-opt %s --verify-each --pass-pipeline='builtin.module(wafer-decompose-online-attention)' | FileCheck %s
// RUN: wafer-opt %s --pass-pipeline='builtin.module(wafer-decompose-online-attention)' --mlir-print-op-generic | wafer-opt | FileCheck %s --check-prefix=ROUNDTRIP

#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#acc = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#row = affine_map<(b, m, k1, k2, n) -> (b, m)>

module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%query: tensor<2x1025x64xf16>,
                     %key: tensor<2x1031x64xf16>,
                     %value: tensor<2x1031x128xf16>)
        -> tensor<2x1025x128xf16> {
      %result = wafer.tile.region(
          %query, %key, %value : tensor<2x1025x64xf16>,
          tensor<2x1031x64xf16>, tensor<2x1031x128xf16>)
          -> (tensor<2x1025x128xf16>) {
      ^bb0(%query_arg: tensor<2x1025x64xf16>,
           %key_arg: tensor<2x1031x64xf16>,
           %value_arg: tensor<2x1031x128xf16>):
        %scale = arith.constant 1.0 : f32
        %accumulator = tensor.empty() : tensor<2x1025x128xf16>
        %maximum = tensor.empty() : tensor<2x1025xf32>
        %sum = tensor.empty() : tensor<2x1025xf32>
        %next_accumulator, %next_maximum, %next_sum =
            wafer.linalg_ext.online_attention
            ins(%query_arg, %key_arg, %value_arg, %scale :
                tensor<2x1025x64xf16>, tensor<2x1031x64xf16>,
                tensor<2x1031x128xf16>, f32)
            outs(%accumulator, %maximum, %sum : tensor<2x1025x128xf16>,
                tensor<2x1025xf32>, tensor<2x1025xf32>)
            indexing_maps = [#q, #k, #v, #s, #acc, #row, #row] score {
            ^bb0(%attention_0_dot: f16, %attention_0_scale: f32):
              %attention_0_converted = arith.extf %attention_0_dot : f16 to f32
              %attention_0_scaled = arith.mulf %attention_0_converted, %attention_0_scale : f32
              wafer.linalg_ext.attention.yield %attention_0_scaled : f32
            }
            -> (tensor<2x1025x128xf16>, tensor<2x1025xf32>,
                tensor<2x1025xf32>)
        wafer.tile.yield %next_accumulator : tensor<2x1025x128xf16>
      }
      return %result : tensor<2x1025x128xf16>
    }
  }
}

// CHECK-LABEL: func.func @entry
// CHECK-NOT: wafer.linalg_ext.attention
// CHECK-NOT: wafer.linalg_ext.online_attention
// CHECK-COUNT-2: math.exp
// CHECK: linalg.generic
// CHECK: wafer.tile.yield

// ROUNDTRIP-NOT: wafer.linalg_ext.online_attention
// ROUNDTRIP: linalg.generic
