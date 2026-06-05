// RUN: wafer-opt %s | FileCheck %s

module {
  %0 = wafer.storage.alloc
      : !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: wafer.storage.alloc
