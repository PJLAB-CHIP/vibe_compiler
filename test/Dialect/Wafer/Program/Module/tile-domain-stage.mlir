// RUN: wafer-opt %s -o /dev/null
// Complete Tile coverage is checked by the Card executable stage, not the
// local CardModule operation verifier.

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
  }
}
