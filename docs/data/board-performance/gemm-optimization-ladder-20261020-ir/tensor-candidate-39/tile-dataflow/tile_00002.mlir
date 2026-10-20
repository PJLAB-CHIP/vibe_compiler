#map = affine_map<(d0, d1) -> (d1)>
module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{replicated}", mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"], mhlo.use_auto_spmd_partitioning = false} {
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry(%arg0: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}, %arg1: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<1>}) -> memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> {
    %alloc = memref.alloc() : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
    %subview = memref.subview %alloc[0, 512, 0] [1, 256, 4096] [1, 1, 1] : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 2097152>, #wafer.memory<ddr, tensor>>
    wafer.tile.region(%arg1, %arg0, %subview : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 2097152>, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%arg2: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, %arg3: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, %arg4: memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 2097152>, #wafer.memory<ddr, tensor>>):
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_0 = memref.alloc() : memref<1x256x1024xbf16, #wafer.memory<spm, tensor>>
      %subview_1 = memref.subview %arg2[0, 512, 0] [1, 256, 1024] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x1024xbf16, strided<[4194304, 1024, 1], offset: 524288>, #wafer.memory<ddr, tensor>>
      wafer.tile.load %subview_1 into %alloc_0 : memref<1x256x1024xbf16, strided<[4194304, 1024, 1], offset: 524288>, #wafer.memory<ddr, tensor>> into memref<1x256x1024xbf16, #wafer.memory<spm, tensor>>
      scf.for %arg5 = %c0 to %c4096 step %c512 {
        %alloc_2 = memref.alloc() {alignment = 256 : i64} : memref<1x256x512xf32, #wafer.memory<spm, ncx>>
        wafer.tile.fill %alloc_2, %cst {fill_domain = #wafer.fill_domain<physical_footprint>} : memref<1x256x512xf32, #wafer.memory<spm, ncx>>, f32
        %c896 = arith.constant 896 : index
        %0 = scf.for %arg6 = %c0 to %c896 step %c128 iter_args(%arg7 = %alloc_2) -> (memref<1x256x512xf32, #wafer.memory<spm, ncx>>) {
          %subview_9 = memref.subview %arg2[0, 512, %arg6] [1, 256, 128] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x128xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %5 = affine.apply #map(%arg5, %arg6)
          %subview_10 = memref.subview %alloc_0[0, 0, %arg6] [1, 256, 128] [1, 1, 1] : memref<1x256x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x256x128xbf16, strided<[262144, 1024, 1], offset: ?>, #wafer.memory<spm, tensor>>
          %subview_11 = memref.subview %arg3[0, %arg6, %arg5] [1, 128, 512] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x128x512xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %alloc_12 = memref.alloc() : memref<1x128x512xbf16, #wafer.memory<spm, tensor>>
          wafer.tile.load %subview_11 into %alloc_12 : memref<1x128x512xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>> into memref<1x128x512xbf16, #wafer.memory<spm, tensor>>
          %6 = wafer.tile.materialize_layout %subview_10 : memref<1x256x128xbf16, strided<[262144, 1024, 1], offset: ?>, #wafer.memory<spm, tensor>> -> memref<1x256x128xbf16, #wafer.memory<spm, ncx>>
          %7 = wafer.tile.materialize_layout %alloc_12 : memref<1x128x512xbf16, #wafer.memory<spm, tensor>> -> memref<1x128x512xbf16, #wafer.memory<spm, ncx>>
          %8 = wafer.tile.gemm %6, %7 psum(%arg7 : memref<1x256x512xf32, #wafer.memory<spm, ncx>>) {batch_count = 1 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : (memref<1x256x128xbf16, #wafer.memory<spm, ncx>>, memref<1x128x512xbf16, #wafer.memory<spm, ncx>>) -> memref<1x256x512xf32, #wafer.memory<spm, ncx>>
          wafer.tile.copy_into %8 into %arg7 : memref<1x256x512xf32, #wafer.memory<spm, ncx>> into memref<1x256x512xf32, #wafer.memory<spm, ncx>>
          scf.yield %arg7 : memref<1x256x512xf32, #wafer.memory<spm, ncx>>
        } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[0]<>, 3>]>}
        %subview_3 = memref.subview %arg2[0, 512, %c896] [1, 256, 128] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x128xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        %subview_4 = memref.subview %alloc_0[0, 0, 896] [1, 256, 128] [1, 1, 1] : memref<1x256x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x256x128xbf16, strided<[262144, 1024, 1], offset: 896>, #wafer.memory<spm, tensor>>
        %subview_5 = memref.subview %arg3[0, %c896, %arg5] [1, 128, 512] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x128x512xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        %alloc_6 = memref.alloc() : memref<1x128x512xbf16, #wafer.memory<spm, tensor>>
        wafer.tile.load %subview_5 into %alloc_6 : memref<1x128x512xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>> into memref<1x128x512xbf16, #wafer.memory<spm, tensor>>
        %1 = wafer.tile.materialize_layout %subview_4 : memref<1x256x128xbf16, strided<[262144, 1024, 1], offset: 896>, #wafer.memory<spm, tensor>> -> memref<1x256x128xbf16, #wafer.memory<spm, ncx>>
        %2 = wafer.tile.materialize_layout %alloc_6 : memref<1x128x512xbf16, #wafer.memory<spm, tensor>> -> memref<1x128x512xbf16, #wafer.memory<spm, ncx>>
        %3 = wafer.tile.gemm %1, %2 psum(%0 : memref<1x256x512xf32, #wafer.memory<spm, ncx>>) {batch_count = 1 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : (memref<1x256x128xbf16, #wafer.memory<spm, ncx>>, memref<1x128x512xbf16, #wafer.memory<spm, ncx>>) -> memref<1x256x512xbf16, #wafer.memory<spm, ncx>>
        %alloc_7 = memref.alloc() {alignment = 256 : i64} : memref<1x256x512xbf16, #wafer.memory<spm, cx>>
        wafer.tile.copy_into %3 into %alloc_7 : memref<1x256x512xbf16, #wafer.memory<spm, ncx>> into memref<1x256x512xbf16, #wafer.memory<spm, cx>>
        %4 = wafer.tile.materialize_layout %alloc_7 : memref<1x256x512xbf16, #wafer.memory<spm, cx>> -> memref<1x256x512xbf16, #wafer.memory<spm, tensor>>
        %subview_8 = memref.subview %arg4[0, 0, %arg5] [1, 256, 512] [1, 1, 1] : memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 2097152>, #wafer.memory<ddr, tensor>> to memref<1x256x512xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        wafer.tile.store %4, %subview_8 : memref<1x256x512xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x512xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[1]<>, 2>]>}
    }
    return %alloc : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
  }
}

