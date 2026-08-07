// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<ddr, tensor>>
  %0 = wafer.tile.region(
      %source : memref<4xf32, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>):
    %src = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    %dst = wafer.tile.materialize_layout %src
        : memref<4xf32, #wafer.memory<spm, tensor>>
       -> memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0
        : memref<4xf32, #wafer.memory<ddr, tensor>>
  }
}

// CHECK: layout materialize must change layout
