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

// CHECK: region control flow edge from parent operands to Region #0: source has 1 operands, but target successor needs 0
