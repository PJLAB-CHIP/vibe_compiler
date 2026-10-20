module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x1024x4096xbf16>, %arg1: tensor<1x4096x1024xbf16>) -> tensor<1x4096x4096xbf16> {
    %0 = stablehlo.dot_general %arg1, %arg0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x4096x1024xbf16>, tensor<1x1024x4096xbf16>) -> tensor<1x4096x4096xbf16>
    return %0 : tensor<1x4096x4096xbf16>
  }
}
