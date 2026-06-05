// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %bad = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<*xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: storage logical type must be a ranked tensor
