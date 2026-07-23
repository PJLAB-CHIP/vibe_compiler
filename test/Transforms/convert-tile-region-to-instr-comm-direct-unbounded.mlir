// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 17>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 17>,
       policy = "all_available",
       endpoints = array<i64>}

  func.func @direct_reduce_scatter_without_bounded_ring(
      %boundary: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(%boundary
        : memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %input = memref.alloc()
          : memref<68xf32, #wafer.memory<spm, tensor>>
      %recv = memref.alloc()
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %result = wafer.tile.reduce_scatter #wafer.reduce_kind<sum> %input using %recv
          {axis = 0 : i64, local_rank = 1 : i64, group_size = 17 : i64,
           rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15, 16>,
           bytes = 16 : i64, communication_id = 18 : i64}
          : (memref<68xf32, #wafer.memory<spm, tensor>>,
             memref<4xf32, #wafer.memory<spm, tensor>>)
         -> memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
}

// CHECK-LABEL: func.func @direct_reduce_scatter_without_bounded_ring
// CHECK: wafer.instr.dte_send
// CHECK-SAME: phase = reduce_scatter_direct
// CHECK: wafer.instr.dte_recv
// CHECK-SAME: phase = reduce_scatter_direct
// CHECK: wafer.instr.dte_wait
// CHECK-NOT: wafer.tile.reduce_scatter
