// RUN: wafer-opt %s -o /dev/null
// One selected Card is a compiler-stage contract, not a leaf op invariant.

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
  }
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
  }
}
