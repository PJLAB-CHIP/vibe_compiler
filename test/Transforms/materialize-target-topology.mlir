// RUN: wafer-opt --wafer-materialize-target-topology %s | FileCheck %s
// RUN: wafer-opt --wafer-materialize-target-topology='card-y-count=4 card-x-count=8 tile-y-count=4 tile-x-count=4 unavailable-tiles=0,0,0,1,0,0,1,0' %s | FileCheck --check-prefix=UNAVAILABLE %s
// RUN: wafer-opt --wafer-materialize-target-topology='card-y-count=4 card-x-count=8 card-interconnect=torus tile-y-count=4 tile-x-count=4' %s | FileCheck --check-prefix=TORUS %s
// RUN: not wafer-opt --wafer-materialize-target-topology='card-interconnect=ring' %s 2>&1 | FileCheck --check-prefix=BAD-INTERCONNECT %s
// RUN: not wafer-opt --wafer-materialize-target-topology='unavailable-tiles=0,0,0' %s 2>&1 | FileCheck --check-prefix=BAD-UNAVAILABLE %s

module {
}

// CHECK: wafer.target.topology @default
// CHECK-SAME: card_grid = array<i64: 1, 1>
// CHECK-SAME: card_interconnect = "mesh"
// CHECK-SAME: tile_grid = array<i64: 4, 4>
// CHECK-SAME: unavailable_tiles = array<i64>
// CHECK-NOT: tile_ids
// CHECK-NOT: links

// UNAVAILABLE: wafer.target.topology @default
// UNAVAILABLE-SAME: card_grid = array<i64: 4, 8>
// UNAVAILABLE-SAME: card_interconnect = "mesh"
// UNAVAILABLE-SAME: tile_grid = array<i64: 4, 4>
// UNAVAILABLE-SAME: unavailable_tiles = array<i64: 0, 0, 0, 1, 0, 0, 1, 0>

// TORUS: wafer.target.topology @default
// TORUS-SAME: card_grid = array<i64: 4, 8>
// TORUS-SAME: card_interconnect = "torus"
// TORUS-SAME: tile_grid = array<i64: 4, 4>

// BAD-INTERCONNECT: topology_failure: card-interconnect must be mesh or torus
// BAD-UNAVAILABLE: topology_failure: unavailable-tiles must contain card_y/card_x/tile_y/tile_x tuples
