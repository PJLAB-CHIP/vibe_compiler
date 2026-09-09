// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @strided_fill() {
  %storage = memref.alloc() : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
  %view = memref.subview %storage[0, 0, 3] [2, 1025, 64] [1, 1, 1]
      : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
     to memref<2x1025x64xf16, strided<[131200, 128, 1], offset: 3>, #wafer.memory<spm, tensor>>
  %one = arith.constant 1.0 : f16
  wafer.instr.fill %view, %one
      : memref<2x1025x64xf16, strided<[131200, 128, 1], offset: 3>, #wafer.memory<spm, tensor>>, f16
  return
}

// CHECK: logical_valid fill requires a static contiguous row-major view
