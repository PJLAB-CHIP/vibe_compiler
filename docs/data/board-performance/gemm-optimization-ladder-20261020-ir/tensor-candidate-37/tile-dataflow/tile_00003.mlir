#map = affine_map<(d0) -> (d0 + 3072)>
module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{replicated}", mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"], mhlo.use_auto_spmd_partitioning = false} {
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry(%arg0: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}, %arg1: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<1>}) -> memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> {
    %alloc = memref.alloc() : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
    %subview = memref.subview %alloc[0, 0, 3072] [1, 1024, 1024] [1, 1, 1] : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 3072>, #wafer.memory<ddr, tensor>>
    wafer.tile.region(%arg1, %arg0, %subview : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 3072>, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%arg2: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, %arg3: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, %arg4: memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 3072>, #wafer.memory<ddr, tensor>>):
      %c256 = arith.constant 256 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      scf.for %arg5 = %c0 to %c1024 step %c256 {
        scf.for %arg6 = %c0 to %c1024 step %c256 {
          %alloc_0 = memref.alloc() {alignment = 256 : i64} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
          wafer.tile.fill %alloc_0, %cst {fill_domain = #wafer.fill_domain<physical_footprint>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>, f32
          %c768 = arith.constant 768 : index
          %0 = scf.for %arg7 = %c0 to %c768 step %c256 iter_args(%arg8 = %alloc_0) -> (memref<1x256x256xf32, #wafer.memory<spm, ncx>>) {
            %subview_7 = memref.subview %arg2[0, %arg5, %arg7] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
            %alloc_8 = memref.alloc() : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            wafer.tile.load %subview_7 into %alloc_8 : memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>> into memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            %6 = affine.apply #map(%arg6)
            %subview_9 = memref.subview %arg3[0, %arg7, %6] [1, 256, 256] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
            %alloc_10 = memref.alloc() : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            wafer.tile.load %subview_9 into %alloc_10 : memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>> into memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            %7 = wafer.tile.materialize_layout %alloc_8 : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %8 = wafer.tile.materialize_layout %alloc_10 : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %9 = wafer.tile.gemm %7, %8 psum(%arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>) {batch_count = 1 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : (memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x256xbf16, #wafer.memory<spm, ncx>>) -> memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            wafer.tile.copy_into %9 into %arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>> into memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            scf.yield %arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
          } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[0]<>, 3>]>}
          %subview_1 = memref.subview %arg2[0, %arg5, %c768] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %alloc_2 = memref.alloc() : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          wafer.tile.load %subview_1 into %alloc_2 : memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>> into memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %1 = affine.apply #map(%arg6)
          %subview_3 = memref.subview %arg3[0, %c768, %1] [1, 256, 256] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %alloc_4 = memref.alloc() : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          wafer.tile.load %subview_3 into %alloc_4 : memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>> into memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %2 = wafer.tile.materialize_layout %alloc_2 : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %3 = wafer.tile.materialize_layout %alloc_4 : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %4 = wafer.tile.gemm %2, %3 psum(%0 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>) {batch_count = 1 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : (memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x256xbf16, #wafer.memory<spm, ncx>>) -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_5 = memref.alloc() {alignment = 256 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, cx>>
          wafer.tile.copy_into %4 into %alloc_5 : memref<1x256x256xbf16, #wafer.memory<spm, ncx>> into memref<1x256x256xbf16, #wafer.memory<spm, cx>>
          %5 = wafer.tile.materialize_layout %alloc_5 : memref<1x256x256xbf16, #wafer.memory<spm, cx>> -> memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %subview_6 = memref.subview %arg4[0, %arg5, %arg6] [1, 256, 256] [1, 1, 1] : memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 3072>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          wafer.tile.store %5, %subview_6 : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[1]<>, 2>]>}
      } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[1]<>, 1>]>}
    }
    return %alloc : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
  }
}

