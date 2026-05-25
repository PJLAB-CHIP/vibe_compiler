// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_binding(%input: tensor<4xf32>) {
    wafer.ddr.external_binding <input> %input
        {alignment = 256 : i64, bytes = 8 : i64, host_visible = true, read_only = true}
        : tensor<4xf32>
    return
  }
}

// CHECK: DDR external binding bytes must match compact tensor storage size
