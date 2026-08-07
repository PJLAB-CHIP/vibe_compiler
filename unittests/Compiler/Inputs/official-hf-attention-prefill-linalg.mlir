// Structured compiler input produced from the pinned official Hugging Face
// eager Llama attention prefill path. The mask remains the original function
// argument and the source scaling/conversion graph is preserved verbatim.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module @official_hf_attention_prefill {
  func.func @main(%arg0: tensor<1x1x1024x64xf16>,
                  %arg1: tensor<1x1x1024x1024xf16>,
                  %arg2: tensor<1x1x1024x64xf16>,
                  %arg3: tensor<1x1x1024x64xf16>)
      -> tensor<1x1x1024x64xf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f16
    %cst_2 = arith.constant dense<1.250000e-01>
        : tensor<1x1x1024x1024xf32>
    %collapsed = tensor.collapse_shape %arg3 [[0], [1, 2], [3]]
        : tensor<1x1x1024x64xf16> into tensor<1x1024x64xf16>
    %0 = tensor.empty() : tensor<1x1x64x1024xf16>
    %1 = linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%arg2 : tensor<1x1x1024x64xf16>)
        outs(%0 : tensor<1x1x64x1024xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x1x64x1024xf16>
    %collapsed_3 = tensor.collapse_shape %1 [[0], [1, 2], [3]]
        : tensor<1x1x64x1024xf16> into tensor<1x64x1024xf16>
    %2 = tensor.empty() : tensor<1x1024x1024xf16>
    %3 = linalg.fill ins(%cst_1 : f16)
        outs(%2 : tensor<1x1024x1024xf16>) -> tensor<1x1024x1024xf16>
    %4 = linalg.batch_matmul
        ins(%collapsed, %collapsed_3
            : tensor<1x1024x64xf16>, tensor<1x64x1024xf16>)
        outs(%3 : tensor<1x1024x1024xf16>) -> tensor<1x1024x1024xf16>
    %expanded = tensor.expand_shape %4 [[0], [1, 2], [3]]
        output_shape [1, 1, 1024, 1024]
        : tensor<1x1024x1024xf16> into tensor<1x1x1024x1024xf16>
    %5 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %6 = linalg.generic {
        indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%expanded : tensor<1x1x1024x1024xf16>)
        outs(%5 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f16, %out: f32):
      %36 = arith.extf %in : f16 to f32
      linalg.yield %36 : f32
    } -> tensor<1x1x1024x1024xf32>
    %7 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %8 = linalg.generic {
        indexing_maps = [#map1, #map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%6, %cst_2
            : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32>)
        outs(%7 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %36 = arith.mulf %in, %in_7 : f32
      linalg.yield %36 : f32
    } -> tensor<1x1x1024x1024xf32>
    %9 = tensor.empty() : tensor<1x1x1024x1024xf16>
    %10 = linalg.generic {
        indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%8 : tensor<1x1x1024x1024xf32>)
        outs(%9 : tensor<1x1x1024x1024xf16>) {
    ^bb0(%in: f32, %out: f16):
      %36 = arith.truncf %in : f32 to f16
      linalg.yield %36 : f16
    } -> tensor<1x1x1024x1024xf16>
    %11 = tensor.empty() : tensor<1x1x1024x1024xf16>
    %12 = linalg.generic {
        indexing_maps = [#map1, #map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%10, %arg1
            : tensor<1x1x1024x1024xf16>, tensor<1x1x1024x1024xf16>)
        outs(%11 : tensor<1x1x1024x1024xf16>) {
    ^bb0(%in: f16, %in_7: f16, %out: f16):
      %36 = arith.addf %in, %in_7 : f16
      linalg.yield %36 : f16
    } -> tensor<1x1x1024x1024xf16>
    %13 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %14 = linalg.generic {
        indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%12 : tensor<1x1x1024x1024xf16>)
        outs(%13 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f16, %out: f32):
      %36 = arith.extf %in : f16 to f32
      linalg.yield %36 : f32
    } -> tensor<1x1x1024x1024xf32>
    %15 = tensor.empty() : tensor<1x1x1024xf32>
    %16 = linalg.fill ins(%cst_0 : f32)
        outs(%15 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %17 = linalg.generic {
        indexing_maps = [#map1, #map2],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%14 : tensor<1x1x1024x1024xf32>)
        outs(%16 : tensor<1x1x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %36 = arith.maximumf %out, %in : f32
      linalg.yield %36 : f32
    } -> tensor<1x1x1024xf32>
    %18 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %19 = linalg.generic {
        indexing_maps = [#map2, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%17 : tensor<1x1x1024xf32>)
        outs(%18 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x1024x1024xf32>
    %20 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %21 = linalg.generic {
        indexing_maps = [#map1, #map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%14, %19
            : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32>)
        outs(%20 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %36 = arith.subf %in, %in_7 : f32
      linalg.yield %36 : f32
    } -> tensor<1x1x1024x1024xf32>
    %22 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %23 = linalg.generic {
        indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%21 : tensor<1x1x1024x1024xf32>)
        outs(%22 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %36 = math.exp %in : f32
      linalg.yield %36 : f32
    } -> tensor<1x1x1024x1024xf32>
    %24 = tensor.empty() : tensor<1x1x1024xf32>
    %25 = linalg.fill ins(%cst : f32)
        outs(%24 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %26 = linalg.generic {
        indexing_maps = [#map1, #map2],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%23 : tensor<1x1x1024x1024xf32>)
        outs(%25 : tensor<1x1x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %36 = arith.addf %out, %in : f32
      linalg.yield %36 : f32
    } -> tensor<1x1x1024xf32>
    %27 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %28 = linalg.generic {
        indexing_maps = [#map2, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%26 : tensor<1x1x1024xf32>)
        outs(%27 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x1024x1024xf32>
    %29 = tensor.empty() : tensor<1x1x1024x1024xf32>
    %30 = linalg.generic {
        indexing_maps = [#map1, #map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%23, %28
            : tensor<1x1x1024x1024xf32>, tensor<1x1x1024x1024xf32>)
        outs(%29 : tensor<1x1x1024x1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %36 = arith.divf %in, %in_7 : f32
      linalg.yield %36 : f32
    } -> tensor<1x1x1024x1024xf32>
    %31 = tensor.empty() : tensor<1x1x1024x1024xf16>
    %32 = linalg.generic {
        indexing_maps = [#map1, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%30 : tensor<1x1x1024x1024xf32>)
        outs(%31 : tensor<1x1x1024x1024xf16>) {
    ^bb0(%in: f32, %out: f16):
      %36 = arith.truncf %in : f32 to f16
      linalg.yield %36 : f16
    } -> tensor<1x1x1024x1024xf16>
    %collapsed_4 = tensor.collapse_shape %32 [[0], [1, 2], [3]]
        : tensor<1x1x1024x1024xf16> into tensor<1x1024x1024xf16>
    %collapsed_5 = tensor.collapse_shape %arg0 [[0], [1, 2], [3]]
        : tensor<1x1x1024x64xf16> into tensor<1x1024x64xf16>
    %33 = tensor.empty() : tensor<1x1024x64xf16>
    %34 = linalg.fill ins(%cst_1 : f16)
        outs(%33 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
    %35 = linalg.batch_matmul
        ins(%collapsed_4, %collapsed_5
            : tensor<1x1024x1024xf16>, tensor<1x1024x64xf16>)
        outs(%34 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
    %expanded_6 = tensor.expand_shape %35 [[0], [1, 2], [3]]
        output_shape [1, 1, 1024, 64]
        : tensor<1x1024x64xf16> into tensor<1x1x1024x64xf16>
    return %expanded_6 : tensor<1x1x1024x64xf16>
  }
}
