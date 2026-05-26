// RUN: wafer-opt %s | FileCheck %s

module {
  func.func @external_bindings(
      %input: tensor<4xf32>,
      %output: tensor<4xf32>) {
    wafer.ddr.external_binding <input> %input
        {alignment = 256 : i64, bytes = 16 : i64, host_visible = true, read_only = true}
        : tensor<4xf32>
    wafer.ddr.external_binding <output> %output
        {alignment = 256 : i64, bytes = 16 : i64, host_visible = true, read_only = false}
        : tensor<4xf32>
    return
  }
}

// CHECK-LABEL: func.func @external_bindings(
// CHECK: wafer.ddr.external_binding <input>
// CHECK-SAME: bytes = 16 : i64
// CHECK-SAME: read_only = true
// CHECK: wafer.ddr.external_binding <output>
// CHECK-SAME: bytes = 16 : i64
// CHECK-SAME: read_only = false
