// RUN: wafer-opt %s | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %sum = wafer.compute.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (!wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %max = wafer.abi.reduce <issue_only> <max> %input
      {dimensions = array<i64: 1>, init_value = -6.550400e+04 : f16}
      : (!wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
}

// CHECK: wafer.compute.reduce <sum>
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK-SAME: init_value = 0.000000e+00 : f16
// CHECK: wafer.abi.reduce <issue_only> <max>
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK-SAME: init_value = -6.550400e+04 : f16
