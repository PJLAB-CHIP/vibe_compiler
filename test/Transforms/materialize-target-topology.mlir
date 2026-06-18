// RUN: wafer-opt --wafer-materialize-target-topology %s | FileCheck %s
// RUN: wafer-opt --wafer-materialize-target-topology='tile-y-count=1 tile-x-count=2 bad-tile-ids=1 pg-disabled-tile-ids=0' %s | FileCheck --check-prefix=FILTER %s
// RUN: wafer-opt --wafer-materialize-target-topology='tile-y-count=1 tile-x-count=2 id-encoding=imported tile-id-remap=7,3 bad-tile-ids=3' %s | FileCheck --check-prefix=REMAP %s
// RUN: not wafer-opt --wafer-materialize-target-topology='id-encoding=imported' %s 2>&1 | FileCheck --check-prefix=BAD-CODEC %s

module {
}

// CHECK: wafer.target.topology @default
// CHECK-SAME: available_tile_ids = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>
// CHECK-SAME: axes = ["card_y", "card_x", "tile_y", "tile_x"]
// CHECK-SAME: bad_tile_ids = array<i64>
// CHECK-SAME: id_encoding = "row_major_4d"
// CHECK-SAME: links =
// CHECK-SAME: pg_disabled_tile_ids = array<i64>
// CHECK-SAME: tile_ids = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>

// FILTER: wafer.target.topology @default
// FILTER-SAME: available_tile_ids = array<i64>
// FILTER-SAME: bad_tile_ids = array<i64: 1>
// FILTER-SAME: pg_disabled_tile_ids = array<i64: 0>
// FILTER-SAME: tile_ids = array<i64: 0, 1>

// REMAP: wafer.target.topology @default
// REMAP-SAME: available_tile_ids = array<i64: 7>
// REMAP-SAME: bad_tile_ids = array<i64: 3>
// REMAP-SAME: id_encoding = "imported"
// REMAP-SAME: links = array<i64: 7, 3>
// REMAP-SAME: tile_ids = array<i64: 7, 3>

// BAD-CODEC: topology_failure: non-default id encoding requires tile-id-remap
