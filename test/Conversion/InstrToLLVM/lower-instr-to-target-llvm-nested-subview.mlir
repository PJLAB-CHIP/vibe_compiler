// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | FileCheck %s
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/invalid-child.mlir 2>&1 | FileCheck %s --check-prefix=BOUNDS

//--- positive.mlir

// A static child uses the already displaced address of its dynamic parent.
// Four batches, full 64-row blocks, and a separate 1025-row tail exercise
// inherited dynamic offsets without widening the requested column window.
func.func @nested_main(
    %input: memref<4x1024x1031xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1x64x1024xf16, #wafer.memory<spm, tensor>>
  scf.for %batch = %c0 to %c4 step %c1 {
    scf.for %row = %c0 to %c1024 step %c64 {
      %parent = memref.subview %input[%batch, %row, 0] [1, 64, 1031] [1, 1, 1]
          : memref<4x1024x1031xf16, #wafer.memory<ddr, tensor>>
         to memref<1x64x1031xf16, strided<[1055744, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      %child = memref.subview %parent[0, 0, 7] [1, 64, 1024] [1, 1, 1]
          : memref<1x64x1031xf16, strided<[1055744, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
         to memref<1x64x1024xf16, strided<[1055744, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      wafer.instr.rdma %child to %spm
          {byte_count = 131072 : i64, inner_bytes = 2048 : i64,
           src_strides = array<i64: 2062, 0, 0>, src_iterations = array<i64: 64, 1, 1>}
          : memref<1x64x1024xf16, strided<[1055744, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
         to memref<1x64x1024xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
    }
  }
  return
}

// CHECK-LABEL: llvm.func @nested_main(
// CHECK-SAME: %[[INPUT:.*]]: i64)
// CHECK: %[[BSTRIDE:.*]] = llvm.mlir.constant(2111488 : i64)
// CHECK: %[[BOFF:.*]] = llvm.mul {{.*}}, %[[BSTRIDE]] : i64
// CHECK: %[[BBASE:.*]] = llvm.add %[[INPUT]], %[[BOFF]] : i64
// CHECK: %[[RSTRIDE:.*]] = llvm.mlir.constant(2062 : i64)
// CHECK: %[[ROFF:.*]] = llvm.mul {{.*}}, %[[RSTRIDE]] : i64
// CHECK: %[[PARENT:.*]] = llvm.add %[[BBASE]], %[[ROFF]] : i64
// CHECK: %[[COL:.*]] = llvm.mlir.constant(14 : i64)
// CHECK: %[[CHILD:.*]] = llvm.add %[[PARENT]], %[[COL]] : i64
// CHECK: llvm.call @wafer_tx81_rdma(%[[CHILD]],
// CHECK-NOT: memref.subview
// LLVMIR-LABEL: define void @nested_main(
// LLVMIR: mul i64 {{.*}}, 2111488
// LLVMIR: mul i64 {{.*}}, 2062
// LLVMIR: %[[CHILD:.*]] = add i64 {{.*}}, 14
// LLVMIR: call void @wafer_tx81_rdma(i64 %[[CHILD]],

func.func @nested_tail(
    %input: memref<4x1025x1031xbf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1024xbf16, #wafer.memory<spm, tensor>>
  scf.for %batch = %c0 to %c4 step %c1 {
    %parent = memref.subview %input[%batch, 1024, 0] [1, 1, 1031] [1, 1, 1]
        : memref<4x1025x1031xbf16, #wafer.memory<ddr, tensor>>
       to memref<1x1x1031xbf16, strided<[1056775, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    %child = memref.subview %parent[0, 0, 7] [1, 1, 1024] [1, 1, 1]
        : memref<1x1x1031xbf16, strided<[1056775, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<1024xbf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %child to %spm
        {byte_count = 2048 : i64, inner_bytes = 2048 : i64,
         src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
        : memref<1024xbf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<1024xbf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
  }
  return
}

// CHECK-LABEL: llvm.func @nested_tail(
// CHECK-SAME: %[[INPUT:.*]]: i64)
// CHECK: %[[STATIC:.*]] = llvm.mlir.constant(2111488 : i64)
// CHECK: %[[BASE:.*]] = llvm.add %[[INPUT]], %[[STATIC]] : i64
// CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(2113550 : i64)
// CHECK: %[[BOFF:.*]] = llvm.mul {{.*}}, %[[STRIDE]] : i64
// CHECK: %[[PARENT:.*]] = llvm.add %[[BASE]], %[[BOFF]] : i64
// CHECK: %[[COL:.*]] = llvm.mlir.constant(14 : i64)
// CHECK: %[[CHILD:.*]] = llvm.add %[[PARENT]], %[[COL]] : i64
// CHECK: llvm.call @wafer_tx81_rdma(%[[CHILD]],
// CHECK-NOT: memref.subview
// LLVMIR-LABEL: define void @nested_tail(
// LLVMIR: add i64 {{.*}}, 2111488
// LLVMIR: mul i64 {{.*}}, 2113550
// LLVMIR: %[[CHILD:.*]] = add i64 {{.*}}, 14
// LLVMIR: call void @wafer_tx81_rdma(i64 %[[CHILD]],

//--- invalid-child.mlir

// A static child still needs bounds verification when its type inherits a
// dynamic offset. Its last column (8 + 1023) exceeds the parent extent.
func.func @invalid_static_child(
    %input: memref<4x1025x1031xbf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %batch = %c0 to %c4 step %c1 {
    %parent = memref.subview %input[%batch, 1024, 0] [1, 1, 1031] [1, 1, 1]
        : memref<4x1025x1031xbf16, #wafer.memory<ddr, tensor>>
       to memref<1x1x1031xbf16, strided<[1056775, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    %child = memref.subview %parent[0, 0, 8] [1, 1, 1024] [1, 1, 1]
        : memref<1x1x1031xbf16, strided<[1056775, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<1024xbf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
  }
  return
}
// BOUNDS: target_geometry_mismatch: dynamic tensor subview dimension #2 may access source coordinate 1031 outside static extent 1031
