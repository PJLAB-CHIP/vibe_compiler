// RUN: wafer-opt %s | FileCheck %s

module {
  func.func @abi_issues(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<4x16xf16>) {
    %lhs_t = wafer.abi.rdma <issue_only> %lhs
        {bytes = 64 : i64}
        : tensor<4x8xf16>
       -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %rhs_t = wafer.abi.rdma <issue_only> %rhs
        {bytes = 256 : i64}
        : tensor<8x16xf16>
       -> !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %lhs_cx = wafer.layout.materialize %lhs_t
        : !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %rhs_cx = wafer.layout.materialize %rhs_t
        : !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %mm = wafer.abi.gemm <issue_only> %lhs_cx, %rhs_cx
        {m = 4 : i64, k = 8 : i64, n = 16 : i64}
        : (!wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
           !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
       -> !wafer.tile_buffer<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %mm_t = wafer.layout.materialize %mm
        : !wafer.tile_buffer<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
       -> !wafer.tile_buffer<tensor<4x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.abi.wdma <issue_only> %mm_t, %out
        {bytes = 128 : i64}
        : !wafer.tile_buffer<tensor<4x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> tensor<4x16xf16>
    return
  }
}

// CHECK-LABEL: func.func @abi_issues(
// CHECK: wafer.abi.rdma <issue_only>
// CHECK-SAME: bytes = 64 : i64
// CHECK: wafer.abi.gemm <issue_only>
// CHECK-SAME: k = 8 : i64
// CHECK-SAME: m = 4 : i64
// CHECK-SAME: n = 16 : i64
// CHECK: wafer.abi.wdma <issue_only>
// CHECK-SAME: bytes = 128 : i64
