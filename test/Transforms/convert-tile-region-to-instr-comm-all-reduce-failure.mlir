// RUN: not wafer-opt --wafer-convert-tile-region-to-instr %s 2>&1 | FileCheck %s

func.func @reject_all_reduce_aligned_layout(
    %boundary: memref<4xf32, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<4xf32, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
    %input = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, cx>>
    %recv = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, cx>>
    %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
        {local_rank = 0 : i64, group_size = 2 : i64,
         rank_group = array<i64: 0, 1>, bytes = 16 : i64}
        : (memref<4xf32, #wafer.memory<spm, cx>>,
           memref<4xf32, #wafer.memory<spm, cx>>)
       -> memref<4xf32, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0
        : memref<4xf32, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: tile.all_reduce lowering requires tensor SPM buffers
