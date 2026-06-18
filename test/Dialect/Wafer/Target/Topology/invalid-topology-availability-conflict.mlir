// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {axes = ["card_y", "card_x", "tile_y", "tile_x"],
       id_encoding = "row_major_4d",
       tile_coords = array<i64: 0, 0, 0, 0>,
       tile_ids = array<i64: 0>,
       available_tile_ids = array<i64: 0>,
       bad_tile_ids = array<i64: 0>,
       pg_disabled_tile_ids = array<i64>,
       links = array<i64>}
}

// CHECK: encoded tile id 0 appears in multiple availability sets
