// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-linalg-to-cabi{target=wafer tile-mapping=multi-tile-no-comm})' %s | FileCheck %s --check-prefix=IR

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 2 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}

  func.func @two_tile_matmul(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%out : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// IR-LABEL: func.func @two_tile_matmul(
// IR: wafer.ddr.external_binding <input>
// IR: wafer.ddr.external_binding <output>
// IR-NOT: wafer.group
// IR: wafer.tile_region
// IR-NOT: wafer.load_tile
// IR: wafer.abi.rdma <issue_only>
// IR: wafer.abi.rdma <issue_only>
// IR-NOT: wafer.compute.gemm
// IR: wafer.abi.gemm <issue_only>
// IR-NOT: wafer.store_tile
// IR: wafer.abi.wdma <issue_only>
// IR: wafer.tile_region
// IR: wafer.abi.rdma <issue_only>
// IR: wafer.abi.rdma <issue_only>
// IR: wafer.abi.gemm <issue_only>
// IR: wafer.abi.wdma <issue_only>
// IR-NOT: wafer.comm
