// RUN: wafer-opt %s | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<6xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %reshaped = wafer.tile.reshape %input
      : !wafer.storage<tensor<6xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.storage<tensor<2x3xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: wafer.tile.reshape
