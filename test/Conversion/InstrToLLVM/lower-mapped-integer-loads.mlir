// RUN: wafer-opt %s --wafer-lower-instr-to-target-llvm --verify-each | FileCheck %s

// Input IDs are immutable program data; acquire once before the row loop.
// CHECK-LABEL: llvm.func @main
// CHECK: llvm.call @get_ddr_memory_mapping_with_size
// CHECK: llvm.br
// CHECK: llvm.load volatile
// CHECK-NOT: llvm.call @get_ddr_memory_mapping_with_size
// CHECK: llvm.return
module {
  func.func @main(%ids: memref<2x1025x1xi64, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %upper = arith.constant 1025 : index
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<2x1025x1xi64, #wafer.memory<spm, tensor>>
    scf.for %i = %zero to %upper step %one {
      %id = memref.load %ids[%zero, %i, %zero] : memref<2x1025x1xi64, #wafer.memory<ddr, tensor>>
      memref.store %id, %dest[%zero, %i, %zero] : memref<2x1025x1xi64, #wafer.memory<spm, tensor>>
    }
    return
  }
}
