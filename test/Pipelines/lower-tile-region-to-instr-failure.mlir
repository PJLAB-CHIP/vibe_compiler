// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s 2>&1 | FileCheck %s

func.func @reject_tile_comm(%boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<4xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
    %local = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf16, #wafer.memory<spm, tensor>>
    %gather = "builtin.unrealized_conversion_cast"()
        : () -> memref<8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.all_gather %local into %gather
        {local_rank = 0 : i64, group_size = 2 : i64,
         rank_group = array<i64: 0, 1>, bytes = 8 : i64}
        : memref<4xf16, #wafer.memory<spm, tensor>>
       -> memref<8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0
        : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: tile communication lowering requires endpoint/local-rank facts
