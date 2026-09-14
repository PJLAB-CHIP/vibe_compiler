// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR

// The entry uses the existing DDR-only ABI. Loop-to-integer casts exercise the
// same finite-width clamp proof as a data-dependent integer index; scalar
// memory-load target lowering is deliberately not claimed by this witness.
// CHECK-LABEL: llvm.func @read(
// CHECK-SAME: %[[SOURCE:.*]]: i64)
// CHECK: %[[NARROW:.*]] = llvm.trunc {{.*}} : i64 to i32
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[LIMIT:.*]] = llvm.mlir.constant(1024 : i32)
// CHECK: %[[LOWER:.*]] = llvm.intr.smax(%[[NARROW]], %[[ZERO]])
// CHECK: %[[CLAMP:.*]] = llvm.intr.smin(%[[LOWER]], %[[LIMIT]])
// CHECK: %[[ROW:.*]] = llvm.sext %[[CLAMP]] : i32 to i64
// CHECK: %[[STATIC:.*]] = llvm.mlir.constant(131216 : i64)
// CHECK: %[[BASE:.*]] = llvm.add %[[SOURCE]], %[[STATIC]] : i64
// CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(128 : i64)
// CHECK: %[[BYTES:.*]] = llvm.mul %[[ROW]], %[[STRIDE]] : i64
// CHECK: %[[ADDRESS:.*]] = llvm.add %[[BASE]], %[[BYTES]] : i64
// CHECK: llvm.call @wafer_tx81_rdma(%[[ADDRESS]],
// CHECK-NOT: memref.
// LLVMIR-LABEL: define void @read(
// LLVMIR: call i32 @llvm.smax.i32
// LLVMIR: call i32 @llvm.smin.i32
// LLVMIR: sext i32 {{.*}} to i64
// LLVMIR: add i64 {{.*}}, 131216
// LLVMIR: mul i64 {{.*}}, 128
// LLVMIR: call void @wafer_tx81_rdma

module {
  func.func @read(%input: memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %end = arith.constant 1025 : index
    scf.for %position = %zero to %end step %one {
    %token = arith.index_cast %position : index to i64
    %small = arith.trunci %token : i64 to i32
    %lo = arith.constant 0 : i32
    %hi = arith.constant 1024 : i32
    %lower = arith.maxsi %small, %lo : i32
    %clamp = arith.minsi %lower, %hi : i32
    %row = arith.index_cast %clamp : i32 to index
    %view = memref.subview %input[1, %row, 8] [1, 1, 16] [1, 1, 1]
        : memref<2x1025x64xf16, #wafer.memory<ddr, tensor>>
       to memref<1x1x16xf16, strided<[65600, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    %local = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x1x16xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %view to %local
        {byte_count = 32 : i64, inner_bytes = 32 : i64,
         src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
        : memref<1x1x16xf16, strided<[65600, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<1x1x16xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    }
    return
  }
}
