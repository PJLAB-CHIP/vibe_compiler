module @IrToHlo.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.spmd_output_sharding = "{replicated}", mhlo.spmd_parameters_shardings = ["{replicated}", "{replicated}"], mhlo.use_auto_spmd_partitioning = false} {
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @entry(%arg0: memref<1x1024x4096xbf16, #wafer.memory<ddr, ncx>> {wafer.program_argument = #wafer.program_argument<0>}, %arg1: memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>> {wafer.program_argument = #wafer.program_argument<1>}) -> memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>> {
    %alloc = memref.alloc() : memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>
    wafer.tile.region(%arg1, %arg0, %alloc : memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>>, memref<1x1024x4096xbf16, #wafer.memory<ddr, ncx>>, memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>) -> () {
    ^bb0(%arg2: memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>>, %arg3: memref<1x1024x4096xbf16, #wafer.memory<ddr, ncx>>, %arg4: memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>):
      %subview = memref.subview %arg4[0, 2560, 0] [1, 256, 4096] [1, 1, 1] : memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>> to memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 10485760>, #wafer.memory<ddr, ncx>>
      %c256 = arith.constant 256 : index
      %c1024 = arith.constant 1024 : index
      %c4096 = arith.constant 4096 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      scf.for %arg5 = %c0 to %c4096 step %c1024 {
        %alloc_0 = memref.alloc() {alignment = 256 : i64} : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
        wafer.tile.fill %alloc_0, %cst {fill_domain = #wafer.fill_domain<physical_footprint>} : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>, f32
        %c768 = arith.constant 768 : index
        %0 = scf.for %arg6 = %c0 to %c768 step %c256 iter_args(%arg7 = %alloc_0) -> (memref<1x256x1024xf32, #wafer.memory<spm, ncx>>) {
          %subview_7 = memref.subview %arg2[0, 2560, %arg6] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, ncx>>
          %alloc_8 = memref.alloc() : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          wafer.tile.load %subview_7 into %alloc_8 : memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, ncx>> into memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
          %subview_9 = memref.subview %arg3[0, %arg6, %arg5] [1, 256, 1024] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, ncx>> to memref<1x256x1024xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, ncx>>
          %alloc_10 = memref.alloc() : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>
          %4 = wafer.tile.peer_recv %alloc_10 {bytes = 524288 : i64, message = #wafer.dte_message<communication = 1, round = 0, slice = 9>, peer = 9 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> -> !async.token
          %5 = wafer.tile.peer_send %alloc_10 {bytes = 524288 : i64, message = #wafer.dte_message<communication = 1, round = 0, slice = 10>, peer = 11 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> -> !async.token
          %6 = wafer.tile.gemm %alloc_8, %alloc_10 psum(%arg7 : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>) {batch_count = 1 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : (memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>) -> memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
          wafer.tile.copy_into %6 into %arg7 : memref<1x256x1024xf32, #wafer.memory<spm, ncx>> into memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
          scf.yield %arg7 : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>
        } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[0]<>, 3>]>}
        %subview_1 = memref.subview %arg2[0, 2560, %c768] [1, 256, 256] [1, 1, 1] : memref<1x4096x1024xbf16, #wafer.memory<ddr, ncx>> to memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, ncx>>
        %alloc_2 = memref.alloc() : memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
        wafer.tile.load %subview_1 into %alloc_2 : memref<1x256x256xbf16, strided<[4194304, 1024, 1], offset: ?>, #wafer.memory<ddr, ncx>> into memref<1x256x256xbf16, #wafer.memory<spm, ncx>>
        %subview_3 = memref.subview %arg3[0, %c768, %arg5] [1, 256, 1024] [1, 1, 1] : memref<1x1024x4096xbf16, #wafer.memory<ddr, ncx>> to memref<1x256x1024xbf16, strided<[4194304, 4096, 1], offset: ?>, #wafer.memory<ddr, ncx>>
        %alloc_4 = memref.alloc() : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>
        %1 = wafer.tile.peer_recv %alloc_4 {bytes = 524288 : i64, message = #wafer.dte_message<communication = 2, round = 0, slice = 9>, peer = 9 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> -> !async.token
        %2 = wafer.tile.peer_send %alloc_4 {bytes = 524288 : i64, message = #wafer.dte_message<communication = 2, round = 0, slice = 10>, peer = 11 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> -> !async.token
        %3 = wafer.tile.gemm %alloc_2, %alloc_4 psum(%0 : memref<1x256x1024xf32, #wafer.memory<spm, ncx>>) {batch_count = 1 : i64, lhs_batch_dims = array<i64: 0>, lhs_contracting_dim = 2 : i64, lhs_m_dim = 1 : i64, result_batch_dims = array<i64: 0>, result_m_dim = 1 : i64, result_n_dim = 2 : i64, rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64} : (memref<1x256x256xbf16, #wafer.memory<spm, ncx>>, memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>) -> memref<1x256x1024xbf16, #wafer.memory<spm, ncx>>
        %alloc_5 = memref.alloc() {alignment = 256 : i64} : memref<1x256x1024xbf16, #wafer.memory<spm, cx>>
        wafer.tile.copy_into %3 into %alloc_5 : memref<1x256x1024xbf16, #wafer.memory<spm, ncx>> into memref<1x256x1024xbf16, #wafer.memory<spm, cx>>
        %subview_6 = memref.subview %subview[0, 0, %arg5] [1, 256, 1024] [1, 1, 1] : memref<1x256x4096xbf16, strided<[16777216, 4096, 1], offset: 10485760>, #wafer.memory<ddr, ncx>> to memref<1x256x1024xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, ncx>>
        wafer.tile.store %alloc_5, %subview_6 : memref<1x256x1024xbf16, #wafer.memory<spm, cx>> -> memref<1x256x1024xbf16, strided<[16777216, 4096, 1], offset: ?>, #wafer.memory<ddr, ncx>>
      } {wafer.iteration_coordinates = #wafer.iteration_coordinates<[<distinct[1]<>, 2>]>}
    }
    return %alloc : memref<1x4096x4096xbf16, #wafer.memory<ddr, ncx>>
  }
}

