// RUN: not wafer-opt --wafer-strip-target-metadata %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

  func.func @unlowered_instr() {
    wafer.instr.ncc_join [0]
    return
  }
}

// CHECK: target_metadata_strip_failure: cannot strip target metadata while non-metadata Wafer ops remain
