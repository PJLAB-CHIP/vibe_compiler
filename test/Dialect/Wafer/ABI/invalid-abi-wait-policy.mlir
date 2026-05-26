// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_wait_policy(%input: tensor<4xf32>) {
    %0 = wafer.abi.rdma <synchronous> %input
        {bytes = 16 : i64}
        : tensor<4xf32>
       -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    return
  }
}

// CHECK: C ABI skeleton ops must use issue_only wait policy
