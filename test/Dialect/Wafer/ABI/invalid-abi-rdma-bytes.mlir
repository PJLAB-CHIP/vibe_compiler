// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad_rdma(%input: tensor<4xf32>) {
    %0 = wafer.abi.rdma_1d <issue_only> %input
        {bytes = 8 : i64}
        : tensor<4xf32>
       -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    return
  }
}

// CHECK: ABI RDMA byte count must match compact tensor transfer size
