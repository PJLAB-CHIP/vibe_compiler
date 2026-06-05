// RUN: wafer-opt %s | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<6xf32, #wafer.memory<spm, tensor>>
  %reshaped = wafer.tile.reshape %input
      : memref<6xf32, #wafer.memory<spm, tensor>>
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
}

// CHECK: wafer.tile.reshape
