// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<ddr, tensor>>
  %0 = wafer.tile.region(
      %source : memref<4xf32, #wafer.memory<ddr, tensor>>) ->
      (memref<4xf32, #wafer.memory<ddr, tensor>>) {
  ^bb0:
    wafer.tile.yield
  }
}

// CHECK: expected 1 body block arguments matching wafer.tile.region inputs, got 0
