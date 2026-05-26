// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %output = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.launch @compiled_kernel
      inputs(%input : tensor<4xf32>)
      outputs(%output : tensor<4xf32>)
      {package_ref = "m0.pkg", spm_bytes = 4096 : i64, ddr_bytes = 8192 : i64}
      : tensor<8xf32>
}

// CHECK: launch result type must match output type at index 0
