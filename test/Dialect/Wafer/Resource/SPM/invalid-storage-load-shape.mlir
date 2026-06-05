// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4x8xf16>
  %0 = wafer.storage.load %source
      : tensor<4x8xf16>
     -> !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: storage.load result tensor type must match source tensor type
