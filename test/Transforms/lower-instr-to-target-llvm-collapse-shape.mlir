// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/compact.mlir | FileCheck %s --check-prefix=COMPACT
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/dynamic-offset.mlir | FileCheck %s --check-prefix=DYNAMIC
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/mixed-offset.mlir 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/aligned.mlir 2>&1 | FileCheck %s --check-prefix=ALIGNED

//--- compact.mlir

func.func @compact_collapse(
    %input: memref<1x4x16xf32, #wafer.memory<ddr, tensor>>,
    %output: memref<4x16xf32, #wafer.memory<ddr, tensor>>) {
  %collapsed = memref.collapse_shape %input [[0, 1], [2]]
      : memref<1x4x16xf32, #wafer.memory<ddr, tensor>>
     into memref<4x16xf32, #wafer.memory<ddr, tensor>>
  %tile = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4x16xf32, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %collapsed to %tile
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4x16xf32, #wafer.memory<ddr, tensor>>
     to memref<4x16xf32, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %tile to %output
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x16xf32, #wafer.memory<spm, tensor>>
     to memref<4x16xf32, #wafer.memory<ddr, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// COMPACT-LABEL: llvm.func @compact_collapse(
// COMPACT-SAME: %[[INPUT:.+]]: i64, %[[OUTPUT:.+]]: i64
// COMPACT-NOT: memref.collapse_shape
// COMPACT: llvm.call @wafer_tx81_rdma(%[[INPUT]],
// COMPACT: llvm.call @wafer_tx81_wdma({{.*}}%[[OUTPUT]]
// COMPACT: llvm.return

//--- dynamic-offset.mlir

func.func @dynamic_subview_collapse(
    %input: memref<1x1x128x64xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %row = %c0 to %c128 step %c64 {
    %view = memref.subview %input[0, 0, %row, 0] [1, 1, 64, 64]
        [1, 1, 1, 1]
        : memref<1x1x128x64xf16, #wafer.memory<ddr, tensor>>
       to memref<1x1x64x64xf16,
                    strided<[8192, 8192, 64, 1], offset: ?>,
                    #wafer.memory<ddr, tensor>>
    %collapsed = memref.collapse_shape %view [[0, 1], [2], [3]]
        : memref<1x1x64x64xf16,
                 strided<[8192, 8192, 64, 1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
       into memref<1x64x64xf16, strided<[8192, 64, 1], offset: ?>,
                   #wafer.memory<ddr, tensor>>
    %tile = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x64x64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %collapsed to %tile
        {byte_count = 8192 : i64, inner_bytes = 8192 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<1x64x64xf16, strided<[8192, 64, 1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
       to memref<1x64x64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
  }
  return
}

// DYNAMIC-LABEL: llvm.func @dynamic_subview_collapse(
// DYNAMIC: ^bb{{[0-9]+}}(%[[ROW:.+]]: i64):
// DYNAMIC: %[[STRIDE:.+]] = llvm.mlir.constant(128 : i64) : i64
// DYNAMIC: %[[DYNAMIC:.+]] = llvm.mul %[[ROW]], %[[STRIDE]] : i64
// DYNAMIC: %[[ADDRESS:.+]] = llvm.add %{{.+}}, %[[DYNAMIC]] : i64
// DYNAMIC: llvm.call @wafer_tx81_rdma(%[[ADDRESS]],
// DYNAMIC-NOT: memref.collapse_shape

func.func @dynamic_subview_expand(
    %input: memref<8192xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c8192 = arith.constant 8192 : index
  scf.for %start = %c0 to %c8192 step %c4096 {
    %view = memref.subview %input[%start] [4096] [1]
        : memref<8192xf16, #wafer.memory<ddr, tensor>>
       to memref<4096xf16, strided<[1], offset: ?>,
                    #wafer.memory<ddr, tensor>>
    %expanded = memref.expand_shape %view [[0, 1, 2]]
        output_shape [1, 64, 64]
        : memref<4096xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
       into memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>,
                   #wafer.memory<ddr, tensor>>
    %tile = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x64x64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %expanded to %tile
        {byte_count = 8192 : i64, inner_bytes = 8192 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
       to memref<1x64x64xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
  }
  return
}

// DYNAMIC-LABEL: llvm.func @dynamic_subview_expand(
// DYNAMIC: ^bb{{[0-9]+}}(%[[START:.+]]: i64):
// DYNAMIC: %[[ELEMENT_BYTES:.+]] = llvm.mlir.constant(2 : i64) : i64
// DYNAMIC: %[[DYNAMIC_EXPAND:.+]] = llvm.mul %[[START]], %[[ELEMENT_BYTES]] : i64
// DYNAMIC: %[[EXPANDED_ADDRESS:.+]] = llvm.add %{{.+}}, %[[DYNAMIC_EXPAND]] : i64
// DYNAMIC: llvm.call @wafer_tx81_rdma(%[[EXPANDED_ADDRESS]],
// DYNAMIC-NOT: memref.expand_shape

//--- mixed-offset.mlir

func.func @reject_mixed_static_dynamic_collapse(
    %input: memref<1x1x64x64xf16,
                   strided<[4096, 4096, 64, 1], offset: ?>,
                   #wafer.memory<ddr, tensor>>) {
  %collapsed = memref.collapse_shape %input [[0, 1], [2], [3]]
      : memref<1x1x64x64xf16,
               strided<[4096, 4096, 64, 1], offset: ?>,
               #wafer.memory<ddr, tensor>>
     into memref<1x64x64xf16, strided<[4096, 64, 1], offset: 0>,
                 #wafer.memory<ddr, tensor>>
  return
}

// The upstream memref verifier rejects a mixed static/dynamic offset before
// target conversion, so the target cannot silently reinterpret it as an
// address-preserving view.
// MIXED: 'memref.collapse_shape' op expected collapsed type to be {{.*}}offset: ?{{.*}} but found 'memref<1x64x64xf16, strided

//--- aligned.mlir

func.func @reject_aligned_collapse(
    %input: memref<1x4x16xf32, #wafer.memory<ddr, cx>>)
    -> memref<4x16xf32, #wafer.memory<ddr, cx>> {
  %collapsed = memref.collapse_shape %input [[0, 1], [2]]
      : memref<1x4x16xf32, #wafer.memory<ddr, cx>>
     into memref<4x16xf32, #wafer.memory<ddr, cx>>
  return %collapsed : memref<4x16xf32, #wafer.memory<ddr, cx>>
}

// ALIGNED: unsupported_target_address: collapse_shape requires matching Wafer tensor-layout memory and element types
