// RUN: split-file %s %t
// RUN: wafer-opt --wafer-plan-ddr-memory --wafer-lower-instr-to-target-llvm %t/reset.mlir | FileCheck %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/outside.mlir 2>&1 | FileCheck %s --check-prefix=OUTSIDE

// The old source offset is one row. reinterpret_cast replaces it with zero;
// adding both offsets would incorrectly put the last dynamic read past root.
// CHECK-LABEL: llvm.func @reset
// CHECK: llvm.call @wafer_tx81_rdma
// OUTSIDE: ddr_range_overflow: source access end 5120 exceeds DDR root byte size 4096

//--- reset.mlir
func.func @reset(%root: memref<1x1024x32xi1, #wafer.memory<ddr, tensor>>) {
  %shifted = memref.subview %root[0, 1, 0] [1, 1023, 32] [1, 1, 1]
    : memref<1x1024x32xi1, #wafer.memory<ddr, tensor>>
    to memref<1x1023x32xi1, strided<[32768, 32, 1], offset: 32>, #wafer.memory<ddr, tensor>>
  %flat = memref.reinterpret_cast %shifted to offset: [0], sizes: [32768], strides: [1]
    : memref<1x1023x32xi1, strided<[32768, 32, 1], offset: 32>, #wafer.memory<ddr, tensor>>
    to memref<32768xi1, #wafer.memory<ddr, tensor>>
  %zero = arith.constant 0 : index
  %end = arith.constant 1024 : index
  %step = arith.constant 256 : index
  %width = arith.constant 32 : index
  %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<8192xi1, #wafer.memory<spm, tensor>>
  scf.for %row = %zero to %end step %step {
    %start = arith.muli %row, %width : index
    %window = memref.subview %flat[%start] [8192] [1]
      : memref<32768xi1, #wafer.memory<ddr, tensor>>
      to memref<8192xi1, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %window to %dest {byte_count = 1024 : i64, inner_bytes = 1024 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
      : memref<8192xi1, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
      to memref<8192xi1, #wafer.memory<spm, tensor>>
  }
  wafer.instr.ncc_join [0]
  return
}

//--- outside.mlir
func.func @outside(%root: memref<1x1024x32xi1, #wafer.memory<ddr, tensor>>) {
  %flat = memref.reinterpret_cast %root to offset: [0], sizes: [32768], strides: [1]
    : memref<1x1024x32xi1, #wafer.memory<ddr, tensor>>
    to memref<32768xi1, #wafer.memory<ddr, tensor>>
  %zero = arith.constant 0 : index
  %end = arith.constant 1025 : index
  %step = arith.constant 256 : index
  %width = arith.constant 32 : index
  %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<8192xi1, #wafer.memory<spm, tensor>>
  scf.for %row = %zero to %end step %step {
    %start = arith.muli %row, %width : index
    %window = memref.subview %flat[%start] [8192] [1]
      : memref<32768xi1, #wafer.memory<ddr, tensor>>
      to memref<8192xi1, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %window to %dest {byte_count = 1024 : i64, inner_bytes = 1024 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
      : memref<8192xi1, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
      to memref<8192xi1, #wafer.memory<spm, tensor>>
  }
  wafer.instr.ncc_join [0]
  return
}
