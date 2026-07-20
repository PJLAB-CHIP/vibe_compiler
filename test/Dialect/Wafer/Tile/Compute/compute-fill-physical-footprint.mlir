// RUN: wafer-opt %s | FileCheck %s

module {
  %cx = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3xf16, #wafer.memory<spm, cx>>
  %value = arith.constant 3.33251953125E-1 : f16
  wafer.tile.fill %cx, %value
      {fill_domain = #wafer.fill_domain<physical_footprint>}
      : memref<2x3xf16, #wafer.memory<spm, cx>>, f16
}

// CHECK: wafer.tile.fill
// CHECK-SAME: fill_domain = #wafer.fill_domain<physical_footprint>
