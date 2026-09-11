// RUN: wafer-opt %s --split-input-file --verify-diagnostics
#q = affine_map<(b,m,k1,k2,n)->(b,m,k1)>
#k = affine_map<(b,m,k1,k2,n)->(b,k2,k1)>
#v = affine_map<(b,m,k1,k2,n)->(b,k2,n)>
#s = affine_map<(b,m,k1,k2,n)->()>
#o = affine_map<(b,m,k1,k2,n)->(b,m,n)>
module {
  func.func @test(%q: tensor<2x1025x128xf16>, %k: tensor<2x1031x128xf16>,
                  %v: tensor<2x1031x64xf16>, %scale: f32) -> tensor<2x1025x64xf16> {
    %init = tensor.empty() : tensor<2x1025x64xf16>
    // expected-error@+1 {{score must not capture values}}
    %result = wafer.linalg_ext.attention
      ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>, tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
      outs(%init : tensor<2x1025x64xf16>) algorithm(<flash_attention>)
      indexing_maps = [#q,#k,#v,#s,#o] score {
      ^bb0(%dot: f16, %scale_arg: f32):
        %wide = arith.extf %dot : f16 to f32
        %scaled = arith.mulf %wide, %scale : f32
        wafer.linalg_ext.attention.yield %scaled : f32
      } -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}

// -----
#q = affine_map<(b,m,k1,k2,n)->(b,m,k1)>
#k = affine_map<(b,m,k1,k2,n)->(b,k2,k1)>
#v = affine_map<(b,m,k1,k2,n)->(b,k2,n)>
#s = affine_map<(b,m,k1,k2,n)->()>
#o = affine_map<(b,m,k1,k2,n)->(b,m,n)>
module {
  func.func @test(%q: tensor<2x1025x128xf16>, %k: tensor<2x1031x128xf16>,
                  %v: tensor<2x1031x64xf16>, %scale: f32) -> tensor<2x1025x64xf16> {
    %init = tensor.empty() : tensor<2x1025x64xf16>
    // expected-error@+1 {{score arguments must match dot, scale, and optional mask scalar types}}
    %result = wafer.linalg_ext.attention
      ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>, tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
      outs(%init : tensor<2x1025x64xf16>) algorithm(<flash_attention>)
      indexing_maps = [#q,#k,#v,#s,#o] score {
      ^bb0(%dot: f32, %scale_arg: f32):
        %scaled = arith.mulf %dot, %scale_arg : f32
        wafer.linalg_ext.attention.yield %scaled : f32
      } -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}

// -----
#q = affine_map<(b,m,k1,k2,n)->(b,m,k1)>
#k = affine_map<(b,m,k1,k2,n)->(b,k2,k1)>
#v = affine_map<(b,m,k1,k2,n)->(b,k2,n)>
#s = affine_map<(b,m,k1,k2,n)->()>
#o = affine_map<(b,m,k1,k2,n)->(b,m,n)>
module {
  func.func @test(%q: tensor<2x1025x128xf16>, %k: tensor<2x1031x128xf16>,
                  %v: tensor<2x1031x64xf16>, %scale: f32) -> tensor<2x1025x64xf16> {
    %init = tensor.empty() : tensor<2x1025x64xf16>
    // expected-error@+1 {{score must contain only pure scalar operations}}
    %result = wafer.linalg_ext.attention
      ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>, tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
      outs(%init : tensor<2x1025x64xf16>) algorithm(<flash_attention>)
      indexing_maps = [#q,#k,#v,#s,#o] score {
      ^bb0(%dot: f16, %scale_arg: f32):
        %buffer = memref.alloc() : memref<1024xf32>
        %wide = arith.extf %dot : f16 to f32
        wafer.linalg_ext.attention.yield %wide : f32
      } -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
