// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR

// This is the terminal-instruction vertical for the typed CT capability
// path. It proves real wafer.instr operations, verifier geometry, target
// format validation and target-call argument order together; the board probe's
// direct CRT calls do not substitute for this path.
module {
  func.func @ct_reduce_pool_capability_vertical() {
    %reduce_input = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4x64xf16, #wafer.memory<spm, cx>>
    %reduce_output = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<64xf16, #wafer.memory<spm, cx>>
    %pool_input = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<66304>}
        : memref<1x3x5x64xf16, #wafer.memory<spm, ncx>>
    %pool_value = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<68352>}
        : memref<1x2x2x64xf16, #wafer.memory<spm, ncx>>
    %pool_index = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<68864>}
        : memref<1x2x2x64xi16, #wafer.memory<spm, ncx>>
    %unpool_output = memref.alloc()
        {wafer.spm.offset = #wafer.spm_offset<69376>}
        : memref<1x3x5x64xf16, #wafer.memory<spm, ncx>>

    wafer.instr.reduce #wafer.instr_reduce_kind<sum>
        %reduce_input into %reduce_output
        {dim = 1 : i64}
        : memref<4x64xf16, #wafer.memory<spm, cx>>
      into memref<64xf16, #wafer.memory<spm, cx>>
    wafer.instr.pool #wafer.instr_pool_kind<indexedmax> %pool_input
        into %pool_value, %pool_index
        {source_shape = array<i64: 1, 3, 5, 64>,
         dest_shape = array<i64: 1, 2, 2, 64>,
         pads = array<i64: 0, 0, 0, 0>,
         kernel_strides = array<i64: 3, 2, 2, 1>}
        : memref<1x3x5x64xf16, #wafer.memory<spm, ncx>>
      into memref<1x2x2x64xf16, #wafer.memory<spm, ncx>>,
           memref<1x2x2x64xi16, #wafer.memory<spm, ncx>>
    wafer.instr.unpool #wafer.instr_unpool_kind<mask>
        %pool_value, %pool_index into %unpool_output
        {source_shape = array<i64: 1, 2, 2, 64>,
         dest_shape = array<i64: 1, 3, 5, 64>,
         kernel_strides = array<i64: 3, 2, 2, 1>}
        : memref<1x2x2x64xf16, #wafer.memory<spm, ncx>>,
          memref<1x2x2x64xi16, #wafer.memory<spm, ncx>>
      into memref<1x3x5x64xf16, #wafer.memory<spm, ncx>>
    return
  }
}

// CHECK-LABEL: llvm.func @ct_reduce_pool_capability_vertical
// CHECK: llvm.call @wafer_tx81_reduce_sum
// CHECK: llvm.call @wafer_tx81_pool_indexedmax
// CHECK: llvm.call @wafer_tx81_unpool_mask

// LLVMIR-LABEL: define void @ct_reduce_pool_capability_vertical()
// LLVMIR: call void @wafer_tx81_reduce_sum(i64 65536, i64 66048, i32 1,
// LLVMIR-SAME: i32 1, i32 1, i32 4, i32 64, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_pool_indexedmax(i64 66304, i64 68352, i64 68864,
// LLVMIR-SAME: i32 118, i32 1, i32 3, i32 5, i32 64,
// LLVMIR-SAME: i32 1, i32 2, i32 2, i32 64,
// LLVMIR-SAME: i32 0, i32 0, i32 0, i32 0,
// LLVMIR-SAME: i32 3, i32 2, i32 2, i32 1, i32 2, i32 0)
// LLVMIR: call void @wafer_tx81_unpool_mask(i64 68352, i64 69376, i32 123,
// LLVMIR-SAME: i32 68864, i32 1, i32 2, i32 2, i32 64,
// LLVMIR-SAME: i32 1, i32 3, i32 5, i32 64,
// LLVMIR-SAME: i32 3, i32 2, i32 2, i32 1, i32 2, i32 0)
