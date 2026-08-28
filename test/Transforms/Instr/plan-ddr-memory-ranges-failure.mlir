// RUN: wafer-opt --wafer-plan-ddr-memory -split-input-file -verify-diagnostics %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-alignment-bytes=0' %s 2>&1 | FileCheck --check-prefix=BAD-ALIGN %s

func.func @dynamic_ddr_alloc_is_not_planned(%n: index) {
  // expected-error @below {{unsupported_compiler_managed_ddr}}
  %ddr = memref.alloc(%n) : memref<?xf16, #wafer.memory<ddr, tensor>>
  return
}

// BAD-ALIGN: invalid_ddr_resource_limit
