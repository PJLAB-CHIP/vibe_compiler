// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x16xf16, #wafer.memory<spm, tensor>>
  wafer.tile.load %source into %dest
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
    into memref<4x16xf16, #wafer.memory<spm, tensor>>
}

// CHECK: tile.load destination tensor type must match source tensor type
