// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %cx = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3xf16, #wafer.memory<spm, cx>>
  %value = arith.constant 0.0 : f16
  wafer.tile.fill %cx, %value
      {fill_domain = #wafer.fill_domain<logical_valid>}
      : memref<2x3xf16, #wafer.memory<spm, cx>>, f16
}

// CHECK: logical_valid fill destination must use tensor layout

