// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<ddr, tensor>>
  %wrong = "builtin.unrealized_conversion_cast"()
      : () -> memref<8xf32, #wafer.memory<ddr, tensor>>
  %0 = wafer.tile.region(%source, %wrong
      : memref<4xf32, #wafer.memory<ddr, tensor>>,
        memref<8xf32, #wafer.memory<ddr, tensor>>) ->
      (memref<4xf32, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf32, #wafer.memory<ddr, tensor>>,
       %bad: memref<8xf32, #wafer.memory<ddr, tensor>>):
    wafer.tile.yield %bad
        : memref<8xf32, #wafer.memory<ddr, tensor>>
  }
}

// CHECK: tile.yield type 'memref<8xf32, #wafer.memory<ddr, tensor>>' does not match wafer.tile.region result type 'memref<4xf32, #wafer.memory<ddr, tensor>>' at index 0
