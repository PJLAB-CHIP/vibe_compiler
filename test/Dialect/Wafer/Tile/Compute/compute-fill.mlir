// RUN: wafer-opt %s | FileCheck %s

module {
  %out = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %cst = arith.constant 0.000000e+00 : f16
  wafer.compute.fill %out, %cst
      : !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>, f16
}

// CHECK: wafer.compute.fill
