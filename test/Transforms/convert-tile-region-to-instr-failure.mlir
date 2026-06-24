// RUN: not wafer-opt --wafer-convert-tile-region-to-instr %s 2>&1 | FileCheck %s

func.func @reject_reduce_scatter_comm(%boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<4xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
    %input = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf16, #wafer.memory<spm, tensor>>
    %recv = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf16, #wafer.memory<spm, tensor>>
    %result = wafer.tile.reduce_scatter #wafer.reduce_kind<sum> %input using %recv
        {local_rank = 0 : i64, group_size = 2 : i64,
         rank_group = array<i64: 0, 1>, bytes = 8 : i64}
        : (memref<4xf16, #wafer.memory<spm, tensor>>,
           memref<4xf16, #wafer.memory<spm, tensor>>)
       -> memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0
        : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: tile.reduce_scatter lowering requires explicit scatter-slot p2p schedule support
