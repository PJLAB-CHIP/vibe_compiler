// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/supported.mlir | FileCheck %s --check-prefix=SUPPORTED
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/i64.mlir | FileCheck %s --check-prefix=I64
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/uint.mlir | FileCheck %s --check-prefix=UINT
// RUN: not wafer-opt --verify-each=false --wafer-lower-instr-to-target-llvm %t/f64.mlir 2>&1 | FileCheck %s --check-prefix=F64
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/ct-i16.mlir | FileCheck %s --check-prefix=CT-I16
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/ct-bool-add.mlir | FileCheck %s --check-prefix=CT-BOOL
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/ct-pool-i8.mlir | FileCheck %s --check-prefix=CT-POOL-I8
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/ct-pool-f32.mlir | FileCheck %s --check-prefix=CT-POOL-F32
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/ct-unpool-bf16.mlir | FileCheck %s --check-prefix=CT-UNPOOL-BF16
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/ct-reduce-f32.mlir | FileCheck %s --check-prefix=CT-REDUCE-F32
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/gemm-f32.mlir 2>&1 | FileCheck %s --check-prefix=GEMM-F32

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
  func.func @i64_rdma(
      %source: memref<4xi64, #wafer.memory<ddr, tensor>>) {
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi64, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %source to %dest
        {byte_count = 32 : i64, inner_bytes = 32 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xi64, #wafer.memory<ddr, tensor>>
       to memref<4xi64, #wafer.memory<spm, tensor>>
    return
  }
}

// I64-LABEL: llvm.func @i64_rdma
// I64: llvm.call @wafer_tx81_rdma

//--- uint.mlir
module {
  func.func @u8_rdma(
      %source: memref<4xui8, #wafer.memory<ddr, tensor>>) {
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xui8, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %source to %dest
        {byte_count = 4 : i64, inner_bytes = 4 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xui8, #wafer.memory<ddr, tensor>>
       to memref<4xui8, #wafer.memory<spm, tensor>>
    return
  }
}

// UINT-LABEL: llvm.func @u8_rdma
// UINT: llvm.call @wafer_tx81_rdma

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
  func.func @ct_i16_add() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xi16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %lhs, %lhs into %dest
        : memref<4xi16, #wafer.memory<spm, tensor>>,
          memref<4xi16, #wafer.memory<spm, tensor>>
      into memref<4xi16, #wafer.memory<spm, tensor>>
    return
  }
}

// CT-I16-LABEL: llvm.func @ct_i16_add
// CT-I16: llvm.call @wafer_tx81_elementwise_add

//--- ct-bool-add.mlir
module {
  func.func @ct_bool_add() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<8xi1, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<8xi1, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %lhs, %lhs into %dest
        : memref<8xi1, #wafer.memory<spm, tensor>>,
          memref<8xi1, #wafer.memory<spm, tensor>>
      into memref<8xi1, #wafer.memory<spm, tensor>>
    return
  }
}

// CT-BOOL-LABEL: llvm.func @ct_bool_add
// CT-BOOL: llvm.call @wafer_tx81_elementwise_add

//--- ct-pool-i8.mlir
module {
  func.func @ct_pool_i8() {
    %input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x2x4x64xi8, #wafer.memory<spm, ncx>>
    %output = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<1x1x2x64xi8, #wafer.memory<spm, ncx>>
    wafer.instr.pool #wafer.instr_pool_kind<max> %input into %output
        {source_shape = array<i64: 1, 2, 4, 64>,
         dest_shape = array<i64: 1, 1, 2, 64>,
         pads = array<i64: 0, 0, 0, 0>,
         kernel_strides = array<i64: 2, 2, 2, 2>}
        : memref<1x2x4x64xi8, #wafer.memory<spm, ncx>>
      into memref<1x1x2x64xi8, #wafer.memory<spm, ncx>>
    return
  }
}

// CT-POOL-I8-LABEL: llvm.func @ct_pool_i8
// CT-POOL-I8: llvm.call @wafer_tx81_pool_max

//--- ct-pool-f32.mlir
module {
  func.func @ct_pool_f32() {
    %input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x2x4x64xf32, #wafer.memory<spm, ncx>>
    %output = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<67584>}
        : memref<1x1x2x64xf32, #wafer.memory<spm, ncx>>
    wafer.instr.pool #wafer.instr_pool_kind<max> %input into %output
        {source_shape = array<i64: 1, 2, 4, 64>,
         dest_shape = array<i64: 1, 1, 2, 64>,
         pads = array<i64: 0, 0, 0, 0>,
         kernel_strides = array<i64: 2, 2, 2, 2>}
        : memref<1x2x4x64xf32, #wafer.memory<spm, ncx>>
      into memref<1x1x2x64xf32, #wafer.memory<spm, ncx>>
    return
  }
}

// CT-POOL-F32-LABEL: llvm.func @ct_pool_f32
// CT-POOL-F32: llvm.call @wafer_tx81_pool_max

//--- ct-unpool-bf16.mlir
module {
  func.func @ct_unpool_bf16() {
    %input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x1x2x64xbf16, #wafer.memory<spm, ncx>>
    %output = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<1x2x4x64xbf16, #wafer.memory<spm, ncx>>
    wafer.instr.unpool #wafer.instr_unpool_kind<avg> %input into %output
        {source_shape = array<i64: 1, 1, 2, 64>,
         dest_shape = array<i64: 1, 2, 4, 64>,
         kernel_strides = array<i64: 2, 2, 2, 2>}
        : memref<1x1x2x64xbf16, #wafer.memory<spm, ncx>>
      into memref<1x2x4x64xbf16, #wafer.memory<spm, ncx>>
    return
  }
}

// CT-UNPOOL-BF16-LABEL: llvm.func @ct_unpool_bf16
// CT-UNPOOL-BF16: llvm.call @wafer_tx81_unpool_avg

//--- ct-reduce-f32.mlir
module {
  func.func @ct_reduce_f32() {
    %input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4x64xf32, #wafer.memory<spm, cx>>
    %output = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66560>}
        : memref<64xf32, #wafer.memory<spm, cx>>
    wafer.instr.reduce #wafer.instr_reduce_kind<sum> %input into %output
        {dim = 1 : i64}
        : memref<4x64xf32, #wafer.memory<spm, cx>>
      into memref<64xf32, #wafer.memory<spm, cx>>
    return
  }
}

// CT-REDUCE-F32-LABEL: llvm.func @ct_reduce_f32
// CT-REDUCE-F32: llvm.call @wafer_tx81_reduce_sum

//--- gemm-f32.mlir
module {
  func.func @gemm_f32() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf32, #wafer.memory<spm, cx>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<3x4xf32, #wafer.memory<spm, cx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<2x4xf32, #wafer.memory<spm, cx>>
    wafer.instr.gemm %lhs, %rhs into %dst
        {m = 2 : i64, k = 3 : i64, n = 4 : i64}
        : memref<2x3xf32, #wafer.memory<spm, cx>>,
          memref<3x4xf32, #wafer.memory<spm, cx>>
      into memref<2x4xf32, #wafer.memory<spm, cx>>
    return
  }
}

// GEMM-F32: unsupported_target_instr: GEMM does not support f32
