// RUN: not wafer-opt --wafer-convert-tile-region-to-instr %s 2>&1 | FileCheck %s

func.func @reject_tile_comm(%boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<4xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
    %buffer = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf16, #wafer.memory<spm, tensor>>
    %token = wafer.tile.send %buffer {peer = 1 : i64, bytes = 8 : i64}
        : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.tile.wait %token : !async.token
    wafer.tile.yield %arg0
        : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: tile communication lowering requires endpoint/local-rank facts
