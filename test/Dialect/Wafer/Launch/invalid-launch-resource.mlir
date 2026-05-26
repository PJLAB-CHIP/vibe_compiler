// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %output = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.launch @compiled_kernel
      inputs(%input : tensor<4xf32>)
      outputs(%output : tensor<4xf32>)
      {package_ref = "m0.pkg", spm_bytes = -1 : i64, ddr_bytes = 8192 : i64}
      : tensor<4xf32>
}

// CHECK: launch resource byte summaries must be non-negative
