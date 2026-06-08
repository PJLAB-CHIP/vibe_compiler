// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %f16 = arith.constant 0.000000e+00 : f16
  wafer.instr.fill %dst, %f16
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>, f16
}

// CHECK: error: 'wafer.instr.fill' op dest must be a SPM Wafer memref
