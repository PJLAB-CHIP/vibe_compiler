// RUN: wafer-opt %s -o /dev/null
// Parent topology is an executable stage fact, not a local op invariant.

module {
  wafer.tile.module card_id = 0 tile_id = 0 {}
}
