// REQUIRES: shardy
// RUN: not wafer-opt --wafer-apply-default-spmd-sharding %s 2>&1 | FileCheck %s

module {
  func.func @no_user_seed(%x: tensor<32x16xf32>) -> tensor<32x16xf32> {
    return %x : tensor<32x16xf32>
  }
}

// CHECK: requires wafer.execution.mesh for default SPMD input sharding seeds
