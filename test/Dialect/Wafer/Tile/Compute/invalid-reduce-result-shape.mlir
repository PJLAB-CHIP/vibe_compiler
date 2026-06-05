// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %sum = wafer.compute.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
     -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
}

// CHECK: reduce result rank must match input rank minus reduce dimensions
