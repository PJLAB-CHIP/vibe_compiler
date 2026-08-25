// RUN: wafer-opt %s -o /dev/null
// Tile availability is checked against the selected target at Card stage.

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64: 0, 0, 0, 1>}
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
    wafer.tile.module tile_id = 1 {}
  }
}
