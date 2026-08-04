// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s

memref.global "private" constant @folded_splat
    : memref<4xf16, #wafer.memory<ddr, tensor>> = dense<1.0>

func.func @unused_folded_constant(
    %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %unused = memref.get_global @folded_splat
      : memref<4xf16, #wafer.memory<ddr, tensor>>
  return
}

// CHECK-NOT: memref.get_global
// CHECK-NOT: memref.global
// CHECK-NOT: llvm.mlir.global
// CHECK-LABEL: llvm.func @unused_folded_constant
