// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}

  wafer.card.program card_id = 0 {
    wafer.tile.program tile_id = 0 {
      %buf = "builtin.unrealized_conversion_cast"()
          : () -> memref<4xf32, #wafer.memory<spm, tensor>>
      %send = wafer.instr.dte_send %buf
          {peer = 2 : i64, bytes = 16 : i64,
           message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
          : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    }
    wafer.tile.program tile_id = 1 {}
  }
}

// CHECK: 'wafer.instr.dte_send' op peer tile_id 2 is outside the available physical Tile domain for card_id 0
