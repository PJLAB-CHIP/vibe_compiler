#map = affine_map<(d0) -> (d0 + 2048)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1 + 2048)>
#map3 = affine_map<(d0, d1) -> (d1 + 2048)>
module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{replicated}", mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"], mhlo.use_auto_spmd_partitioning = false} {
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry(%arg0: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}, %arg1: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<1>}) -> memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> {
    %alloc = memref.alloc() : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
    %subview = memref.subview %alloc[0, 0, 2048] [1, 1024, 1024] [1, 1, 1] : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>>
    wafer.tile.region(%arg1, %arg0, %subview : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%arg2: memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>>, %arg3: memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>>, %arg4: memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>>):
      %c256 = arith.constant 256 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %alloc_0 = memref.alloc() : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>>
      %subview_1 = memref.subview %arg3[0, 0, 2048] [1, 1024, 1024] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x1024x1024xbf16, strided<[4194304, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>>
      wafer.tile.load %subview_1 into %alloc_0 : memref<1x1024x1024xbf16, strided<[4194304, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>> into memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>>
      %0 = wafer.tile.peer_send %alloc_0 {bytes = 2097152 : i64, message = #wafer.dte_message<communication = 11, round = 0, slice = 0>, peer = 6 : i64} : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>> -> !async.token
      %1 = wafer.tile.peer_send %alloc_0 {bytes = 2097152 : i64, message = #wafer.dte_message<communication = 11, round = 0, slice = 1>, peer = 10 : i64} : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>> -> !async.token
      %2 = wafer.tile.peer_send %alloc_0 {bytes = 2097152 : i64, message = #wafer.dte_message<communication = 11, round = 0, slice = 2>, peer = 14 : i64} : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>> -> !async.token
      scf.for %arg5 = %c0 to %c1024 step %c256 {
        scf.for %arg6 = %c0 to %c1024 step %c256 {
          %alloc_2 = memref.alloc() {alignment = 256 : i64} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
          wafer.tile.fill %alloc_2, %cst {fill_domain = #wafer.fill_domain<physical_footprint>} : memref<1x256x256xf32, #wafer.memory<spm, ncx>>, f32
          %c768 = arith.constant 768 : index
          %3 = scf.for %arg7 = %c0 to %c768 step %c256 iter_args(%arg8 = %alloc_2) -> (memref<1x256x256xf32, #wafer.memory<spm, ncx>>) {
            %subview_9 = memref.subview %arg2[0, %arg5, %arg7] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
            %alloc_10 = memref.alloc() : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
            %12 = wafer.tile.peer_recv %alloc_10 {bytes = 131072 : i64, message = #wafer.dte_message<communication = 1, round = 0, slice = 1>, peer = 1 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
            %13 = wafer.tile.peer_send %alloc_10 {bytes = 131072 : i64, message = #wafer.dte_message<communication = 1, round = 0, slice = 2>, peer = 3 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
            %14 = affine.apply #map(%arg6)
            %subview_11 = memref.subview %arg3[0, %arg7, %14] [1, 256, 256] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
            %15 = affine.apply #map1(%arg5, %arg6, %arg7)
            %16 = affine.apply #map2(%arg5, %arg6, %arg7)
            %subview_12 = memref.subview %alloc_0[0, %arg7, %arg6] [1, 256, 256] [1, 1, 1] : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, strided<[1048576, 1024, 1], offset: ?>, #wafer.memory<spm, tensor>>
            %17 = wafer.tile.materialize_layout %alloc_10 : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %18 = wafer.tile.materialize_layout %subview_12 : memref<1x256x256xbf16, strided<[1048576, 1024, 1], offset: ?>, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
            %19 = wafer.tile.gemm %17, %18 psum(%arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>) {batch_count = 1 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : (memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x256xbf16, #wafer.memory<spm, ncx>>) -> memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            wafer.tile.copy_into %19 into %arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>> into memref<1x256x256xf32, #wafer.memory<spm, ncx>>
            scf.yield %arg8 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>
          } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[0]<>, 3>]>}
          %subview_3 = memref.subview %arg2[0, %arg5, %c768] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %alloc_4 = memref.alloc() : memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %4 = wafer.tile.peer_recv %alloc_4 {bytes = 131072 : i64, message = #wafer.dte_message<communication = 2, round = 0, slice = 1>, peer = 1 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
          %5 = wafer.tile.peer_send %alloc_4 {bytes = 131072 : i64, message = #wafer.dte_message<communication = 2, round = 0, slice = 2>, peer = 3 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> !async.token
          %6 = affine.apply #map(%arg6)
          %subview_5 = memref.subview %arg3[0, %c768, %6] [1, 256, 256] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          %7 = affine.apply #map3(%arg5, %arg6)
          %subview_6 = memref.subview %alloc_0[0, 768, %arg6] [1, 256, 256] [1, 1, 1] : memref<1x1024x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x256x256xbf16, strided<[1048576, 1024, 1], offset: ?>, #wafer.memory<spm, tensor>>
          %8 = wafer.tile.materialize_layout %alloc_4 : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %9 = wafer.tile.materialize_layout %subview_6 : memref<1x256x256xbf16, strided<[1048576, 1024, 1], offset: ?>, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %10 = wafer.tile.gemm %8, %9 psum(%3 : memref<1x256x256xf32, #wafer.memory<spm, ncx>>) {batch_count = 1 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : (memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x256xbf16, #wafer.memory<spm, ncx>>) -> memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %alloc_7 = memref.alloc() {alignment = 256 : i64} : memref<1x256x256xbf16, #wafer.memory<spm, cx>>
          wafer.tile.copy_into %10 into %alloc_7 : memref<1x256x256xbf16, #wafer.memory<spm, ncx>> into memref<1x256x256xbf16, #wafer.memory<spm, cx>>
          %11 = wafer.tile.materialize_layout %alloc_7 : memref<1x256x256xbf16, #wafer.memory<spm, cx>> -> memref<1x256x256xbf16, #wafer.memory<spm, tensor>>
          %subview_8 = memref.subview %arg4[0, %arg5, %arg6] [1, 256, 256] [1, 1, 1] : memref<1x1024x1024xbf16, strided<[16777216, 4096, 1], offset: 2048>, #wafer.memory<ddr, tensor>> to memref<1x256x256xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
          wafer.tile.store %11, %subview_8 : memref<1x256x256xbf16, #wafer.memory<spm, tensor>> -> memref<1x256x256xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[1]<>, 2>]>}
      } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[1]<>, 1>]>}
    }
    return %alloc : memref<1x4096x4096xbf16, #wafer.memory<ddr, tensor>>
  }
}

