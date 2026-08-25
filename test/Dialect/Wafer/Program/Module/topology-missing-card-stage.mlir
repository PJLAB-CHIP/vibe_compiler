// RUN: wafer-opt %s -o /dev/null
// Parent topology is a Card executable stage fact, not a local op invariant.

module {
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
  }
}
