// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %0 = wafer.tile.load %source
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     -> memref<4x16xf16, #wafer.memory<spm, tensor>>
}

// CHECK: tile.load result tensor type must match source tensor type
