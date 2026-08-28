// RUN: wafer-opt -split-input-file -verify-diagnostics %s

module {
  // expected-error @+1 {{'wafer.tile.module' op card_id and tile_id must be non-negative}}
  wafer.tile.module card_id = -1 tile_id = 0 {}
}

// -----

module {
  // expected-error @+1 {{'wafer.tile.module' op card_id and tile_id must be non-negative}}
  wafer.tile.module card_id = 0 tile_id = -1 {}
}
