// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x16xf16, #wafer.memory<spm, cx>>
  %dest = "builtin.unrealized_conversion_cast"() : () -> tensor<4x16xf16>
  wafer.tile.store %source, %dest
      : memref<4x16xf16, #wafer.memory<spm, cx>>
     -> tensor<4x16xf16>
}

// CHECK: tile.store source must use tensor layout for external writeback
