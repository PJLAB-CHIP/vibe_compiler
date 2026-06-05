// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x7xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %a, %b
      : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.storage<tensor<4x7xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: elementwise operand tensor types must match result tensor type
