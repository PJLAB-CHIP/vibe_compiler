// RUN: wafer-opt %s | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %sum = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
     -> !wafer.storage<tensor<4xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
}

// CHECK: wafer.tile.reduce <sum>
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK-SAME: init_value = 0.000000e+00 : f16
