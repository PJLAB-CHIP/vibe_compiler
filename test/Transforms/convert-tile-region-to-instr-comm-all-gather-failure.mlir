// RUN: not wafer-opt --wafer-convert-tile-region-to-instr %s 2>&1 | FileCheck %s

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
wafer.execution.mesh @default_mesh
    {topology = @default, axes = ["rank"], shape = array<i64: 2>,
     policy = "all_available", endpoints = array<i64>}

func.func @reject_all_gather_aligned_layout(
    %boundary: memref<8xf32, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<8xf32, #wafer.memory<ddr, tensor>>)
      -> (memref<8xf32, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<8xf32, #wafer.memory<ddr, tensor>>):
    %local = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, cx>>
    %gather = "builtin.unrealized_conversion_cast"()
        : () -> memref<8xf32, #wafer.memory<spm, cx>>
    wafer.tile.all_gather %local into %gather
        {local_rank = 0 : i64, group_size = 2 : i64,
         rank_group = array<i64: 0, 1>, bytes = 16 : i64,
         communication_id = 13 : i64}
        : memref<4xf32, #wafer.memory<spm, cx>>
       -> memref<8xf32, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0
        : memref<8xf32, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: tile.all_gather lowering requires tensor or ntensor SPM layouts
