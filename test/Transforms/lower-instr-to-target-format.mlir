// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/supported.mlir | FileCheck %s --check-prefix=SUPPORTED
// RUN: not wafer-opt --mlir-disable-threading --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %t/i64.mlir 2>&1 | FileCheck %s --check-prefix=I64 --implicit-check-not=llvm.func --implicit-check-not=llvm.call
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/uint.mlir 2>&1 | FileCheck %s --check-prefix=UINT
// RUN: not wafer-opt --verify-each=false --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/f64.mlir 2>&1 | FileCheck %s --check-prefix=F64
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/ct-i16.mlir 2>&1 | FileCheck %s --check-prefix=CT-I16
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/ct-bool-add.mlir 2>&1 | FileCheck %s --check-prefix=CT-BOOL

//--- supported.mlir
module {
  func.func @supported(
      %source: memref<4xi16, #wafer.memory<ddr, tensor>>) {
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi16, #wafer.memory<spm, tensor>>
    %bool_lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<8xi1, #wafer.memory<spm, tensor>>
    %bool_rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<8xi1, #wafer.memory<spm, tensor>>
    %bool_out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<8xi1, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %source to %dest
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xi16, #wafer.memory<ddr, tensor>>
       to memref<4xi16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<logic_and>
        %bool_lhs, %bool_rhs into %bool_out
        : memref<8xi1, #wafer.memory<spm, tensor>>,
          memref<8xi1, #wafer.memory<spm, tensor>>
      into memref<8xi1, #wafer.memory<spm, tensor>>
    return
  }
}

// SUPPORTED-LABEL: llvm.func @supported
// SUPPORTED: llvm.call @wafer_tx81_rdma
// SUPPORTED: llvm.call @wafer_tx81_elementwise_logic_and

//--- i64.mlir
module {
  func.func @i64_is_not_proven(
      %source: memref<4xi64, #wafer.memory<ddr, tensor>>) {
    %dest = memref.alloc() : memref<4xi64, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %source to %dest
        {byte_count = 32 : i64, inner_bytes = 32 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xi64, #wafer.memory<ddr, tensor>>
       to memref<4xi64, #wafer.memory<spm, tensor>>
    return
  }
}

// I64: unsupported_target_format: profile 'wafer-tx81-single-card-kernel-v1', engine 'rdma', format 'i64' is unsupported: 64-bit-command-encoding-unproven
// I64: IR Dump After LowerInstrToTargetLLVMPass Failed
// I64: func.func @i64_is_not_proven
// I64: wafer.instr.rdma

//--- uint.mlir
module {
  func.func @unsigned_is_not_proven(
      %source: memref<4xui8, #wafer.memory<ddr, tensor>>) {
    %dest = memref.alloc() : memref<4xui8, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %source to %dest
        {byte_count = 4 : i64, inner_bytes = 4 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xui8, #wafer.memory<ddr, tensor>>
       to memref<4xui8, #wafer.memory<spm, tensor>>
    return
  }
}

// UINT: engine 'rdma', format 'u8' is unsupported: unsigned-command-encoding-unproven

//--- f64.mlir
module {
  func.func @f64_has_no_descriptor() {
    %source = memref.alloc() : memref<1x1x1x4xf64, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() : memref<1x1x1x4xf64, #wafer.memory<spm, tensor>>
    wafer.instr.tdma_data_move #wafer.instr_data_move_kind<pad> %source into %dest
        {source_shape = array<i64: 1, 1, 1, 4>,
         dest_shape = array<i64: 1, 1, 1, 4>,
         pads = array<i64: 0, 0, 0, 0>}
        : memref<1x1x1x4xf64, #wafer.memory<spm, tensor>>
       to memref<1x1x1x4xf64, #wafer.memory<spm, tensor>>
    return
  }
}

// F64: unsupported_target_dtype: tdma_data_move dest element type 'f64' has no logical target-format descriptor

//--- ct-i16.mlir
module {
  func.func @ct_i16_is_engine_specific() {
    %lhs = memref.alloc() : memref<4xi16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() : memref<4xi16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %lhs, %lhs into %dest
        : memref<4xi16, #wafer.memory<spm, tensor>>,
          memref<4xi16, #wafer.memory<spm, tensor>>
      into memref<4xi16, #wafer.memory<spm, tensor>>
    return
  }
}

// CT-I16: engine 'ct', format 'i16' is unsupported: generic-compute-integer-command-encoding-unproven

//--- ct-bool-add.mlir
module {
  func.func @ct_bool_add_is_not_a_registered_bool_kind() {
    %lhs = memref.alloc() : memref<8xi1, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() : memref<8xi1, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %lhs, %lhs into %dest
        : memref<8xi1, #wafer.memory<spm, tensor>>,
          memref<8xi1, #wafer.memory<spm, tensor>>
      into memref<8xi1, #wafer.memory<spm, tensor>>
    return
  }
}

// CT-BOOL: CT BOOL is restricted to the registered relation/logic elementwise kinds
