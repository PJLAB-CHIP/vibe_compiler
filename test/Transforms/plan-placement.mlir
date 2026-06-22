// RUN: wafer-opt --wafer-plan-placement='logical-rank-count=6 card-y=1 card-x=2 tile-y=2 tile-x=2 bad-tile-ids=1,4' %s | FileCheck %s

module {
}

// CHECK: wafer.placement.map
// CHECK-SAME: bad_tile_ids = array<i64: 1, 4>
// CHECK-SAME: block_ids = array<i64: 0, 1, 2, 3, 4, 5>
// CHECK-SAME: card_x_count = 2 : i64
// CHECK-SAME: card_y_count = 1 : i64
// CHECK-SAME: logical_rank_count = 6 : i64
// CHECK-SAME: physical_tile_coords = array<i64: 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1>
// CHECK-SAME: tile_x_count = 2 : i64
// CHECK-SAME: tile_y_count = 2 : i64
