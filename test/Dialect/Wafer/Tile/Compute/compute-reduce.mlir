// RUN: wafer-opt %s | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %sum = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (memref<4x8xf16, #wafer.memory<spm, cx>>)
     -> memref<4xf16, #wafer.memory<spm, cx>>
  %init = arith.constant 0.000000e+00 : f16
  %max = wafer.tile.reduce #wafer.reduce_kind<max> %input, %init
      {dimensions = array<i64: 0>}
      : (memref<4x8xf16, #wafer.memory<spm, cx>>, f16)
     -> memref<8xf16, #wafer.memory<spm, cx>>
}

// CHECK: wafer.tile.reduce <sum>
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK-SAME: init_value = 0.000000e+00 : f16
// CHECK: %[[INIT:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: wafer.tile.reduce <max> {{.*}}, %[[INIT]]
// CHECK-SAME: dimensions = array<i64: 0>
