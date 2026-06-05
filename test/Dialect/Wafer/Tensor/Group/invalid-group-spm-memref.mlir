// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %spm = "builtin.unrealized_conversion_cast"() : () -> memref<4xf32, #wafer.memory_space<spm>>
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.group ins(%spm : memref<4xf32, #wafer.memory_space<spm>>)
                    outs(%source : tensor<4xf32>) {
  ^bb0(%buffer: memref<4xf32, #wafer.memory_space<spm>>, %out: tensor<4xf32>):
    wafer.group.yield %out : tensor<4xf32>
  } : tensor<4xf32>
}

// CHECK: does not accept SPM memref inputs
