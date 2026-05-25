// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-multi-tile-no-comm %s | FileCheck %s

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

// CHECK-LABEL: func.func @two_tile_matmul(
// CHECK-NOT: wafer.group
// CHECK: wafer.tile_region
// CHECK: wafer.load_tile
// CHECK: wafer.load_tile
// CHECK: wafer.compute.gemm
// CHECK: wafer.store_tile
// CHECK: wafer.tile_region
// CHECK: wafer.load_tile
// CHECK: wafer.load_tile
// CHECK: wafer.compute.gemm
// CHECK: wafer.store_tile
// CHECK-NOT: wafer.comm
// CHECK: return
