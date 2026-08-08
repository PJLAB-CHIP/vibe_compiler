// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}

  wafer.card.program card_id = 0 {
    memref.global "private" @shared
        : memref<1xi32, #wafer.memory<ddr, tensor>>
    wafer.tile.program tile_id = 0 {
      %shared = memref.get_global @shared
          : memref<1xi32, #wafer.memory<ddr, tensor>>
      %value = arith.constant 7 : i32
    }
    wafer.tile.program tile_id = 1 {
      %shared = memref.get_global @shared
          : memref<1xi32, #wafer.memory<ddr, tensor>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %iv = %c0 to %c4 step %c1 {
      }
    }
  }
}

// CHECK: wafer.card.program card_id = 0
// CHECK: memref.global "private" @shared
// CHECK: wafer.tile.program tile_id = 0
// CHECK: memref.get_global @shared
// CHECK: arith.constant 7 : i32
// CHECK: wafer.tile.program tile_id = 1
// CHECK: memref.get_global @shared
// CHECK: scf.for
