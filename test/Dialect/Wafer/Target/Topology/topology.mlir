// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.target.topology @default
      {axes = ["card_y", "card_x", "tile_y", "tile_x"],
       id_encoding = "row_major_4d",
       tile_coords = array<i64: 0, 0, 0, 0,
                                0, 0, 0, 1>,
       tile_ids = array<i64: 0, 1>,
       available_tile_ids = array<i64: 0>,
       bad_tile_ids = array<i64: 1>,
       pg_disabled_tile_ids = array<i64>,
       links = array<i64: 0, 1>}
}

// CHECK: wafer.target.topology @default
// CHECK-SAME: available_tile_ids = array<i64: 0>
// CHECK-SAME: axes = ["card_y", "card_x", "tile_y", "tile_x"]
// CHECK-SAME: bad_tile_ids = array<i64: 1>
// CHECK-SAME: id_encoding = "row_major_4d"
// CHECK-SAME: links = array<i64: 0, 1>
// CHECK-SAME: pg_disabled_tile_ids = array<i64>
// CHECK-SAME: tile_coords = array<i64: 0, 0, 0, 0, 0, 0, 0, 1>
// CHECK-SAME: tile_ids = array<i64: 0, 1>
