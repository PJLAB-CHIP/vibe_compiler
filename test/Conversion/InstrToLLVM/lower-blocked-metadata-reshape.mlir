// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | FileCheck %s
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/negative.mlir 2>&1 | FileCheck %s --check-prefix=BAD
// BAD: has no exact physical metadata view

//--- positive.mlir
// CHECK-LABEL: llvm.func @unit_1024
// CHECK: %[[ADDR1024:.*]] = llvm.mlir.constant(65536 : i64)
// CHECK: llvm.call @wafer_tx81_memset(%[[ADDR1024]],
// CHECK-NOT: memref.
func.func @unit_1024() {
  %one = arith.constant 1.0 : f16
  %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<2x1x1024x65xf16, #wafer.memory<spm, ncx>>
  %collapsed = memref.collapse_shape %buffer [[0], [1, 2], [3]]
      : memref<2x1x1024x65xf16, #wafer.memory<spm, ncx>> into memref<2x1024x65xf16, #wafer.memory<spm, ncx>>
  %expanded = memref.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [2, 1, 1024, 65]
      : memref<2x1024x65xf16, #wafer.memory<spm, ncx>> into memref<2x1x1024x65xf16, #wafer.memory<spm, ncx>>
  wafer.instr.fill %expanded, %one {fill_domain = #wafer.fill_domain<physical_footprint>}
      : memref<2x1x1024x65xf16, #wafer.memory<spm, ncx>>, f16
  wafer.instr.ncc_join [0]
  return
}
// CHECK-LABEL: llvm.func @unit_1025
// CHECK: %[[ADDR1025:.*]] = llvm.mlir.constant(65536 : i64)
// CHECK: llvm.call @wafer_tx81_memset(%[[ADDR1025]],
// CHECK-NOT: memref.
func.func @unit_1025() {
  %one = arith.constant 1.0 : f16
  %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<2x1x1025x65xf16, #wafer.memory<spm, ncx>>
  %collapsed = memref.collapse_shape %buffer [[0], [1, 2], [3]]
      : memref<2x1x1025x65xf16, #wafer.memory<spm, ncx>> into memref<2x1025x65xf16, #wafer.memory<spm, ncx>>
  %expanded = memref.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [2, 1, 1025, 65]
      : memref<2x1025x65xf16, #wafer.memory<spm, ncx>> into memref<2x1x1025x65xf16, #wafer.memory<spm, ncx>>
  wafer.instr.fill %expanded, %one {fill_domain = #wafer.fill_domain<physical_footprint>}
      : memref<2x1x1025x65xf16, #wafer.memory<spm, ncx>>, f16
  wafer.instr.ncc_join [0]
  return
}
// CHECK-LABEL: llvm.func @unit_1031
// CHECK: %[[ADDR1031:.*]] = llvm.mlir.constant(65536 : i64)
// CHECK: llvm.call @wafer_tx81_memset(%[[ADDR1031]],
// CHECK-NOT: memref.
func.func @unit_1031() {
  %one = arith.constant 1.0 : f16
  %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<2x1x1031x65xf16, #wafer.memory<spm, ncx>>
  %collapsed = memref.collapse_shape %buffer [[0], [1, 2], [3]]
      : memref<2x1x1031x65xf16, #wafer.memory<spm, ncx>> into memref<2x1031x65xf16, #wafer.memory<spm, ncx>>
  %expanded = memref.expand_shape %collapsed [[0], [1, 2], [3]] output_shape [2, 1, 1031, 65]
      : memref<2x1031x65xf16, #wafer.memory<spm, ncx>> into memref<2x1x1031x65xf16, #wafer.memory<spm, ncx>>
  wafer.instr.fill %expanded, %one {fill_domain = #wafer.fill_domain<physical_footprint>}
      : memref<2x1x1031x65xf16, #wafer.memory<spm, ncx>>, f16
  wafer.instr.ncc_join [0]
  return
}
//--- negative.mlir
func.func @non_equivalent() {
  %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1x1024x65xf16, #wafer.memory<spm, ncx>>
  %collapsed = memref.collapse_shape %buffer [[0], [1, 2]]
      : memref<1x1024x65xf16, #wafer.memory<spm, ncx>> into memref<1x66560xf16, #wafer.memory<spm, ncx>>
  return
}
