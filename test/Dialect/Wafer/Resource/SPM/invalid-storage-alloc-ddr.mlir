// RUN: wafer-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error @below {{'wafer.storage.alloc' op result must use SPM memory space}}
  %0 = wafer.storage.alloc
      : !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
}
