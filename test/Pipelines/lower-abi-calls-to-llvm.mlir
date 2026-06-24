// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-abi-calls-to-llvm)' %s | FileCheck %s

func.func @abi_entry(%input0_base: i64) -> i32 {
  %spm = arith.constant 65536 : i32
  %bytes = arith.constant 64 : i64
  %inner = arith.constant 64 : i64
  %stride0 = arith.constant 0 : i64
  %stride1 = arith.constant 0 : i64
  %stride2 = arith.constant 0 : i64
  %iter0 = arith.constant 1 : i64
  %iter1 = arith.constant 1 : i64
  %iter2 = arith.constant 1 : i64
  %fmt = arith.constant 2 : i32
  %s0 = call @wafer_rdma(%input0_base, %spm, %bytes, %inner,
                         %stride0, %stride1, %stride2,
                         %iter0, %iter1, %iter2, %fmt)
      : (i64, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32
  return %s0 : i32
}

func.func private @wafer_rdma(i64, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32

// CHECK: llvm.func @abi_entry
// CHECK: llvm.call @wafer_rdma
// CHECK-NOT: func.call
