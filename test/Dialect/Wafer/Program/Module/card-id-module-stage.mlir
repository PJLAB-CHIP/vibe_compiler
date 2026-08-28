// RUN: wafer-opt %s -o /dev/null
// Card-grid membership is checked at the executable module stage.

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.tile.module card_id = 1 tile_id = 0 {}
}
