// RUN: wafer-opt %s -o /dev/null
// Parent topology is a Card executable stage fact, not a local op invariant.

module {
  wafer.target.topology @first
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.target.topology @second
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
  }
}
