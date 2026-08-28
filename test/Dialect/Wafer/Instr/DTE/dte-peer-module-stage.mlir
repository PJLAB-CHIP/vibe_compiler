// RUN: wafer-opt %s -o /dev/null
// Peer membership is checked once by executable lowering. The DTE op
// verifier remains local to its peer value, buffer, byte count, and token.

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}

  wafer.tile.module card_id = 0 tile_id = 0 {
    %buf = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    %send = wafer.instr.dte_send %buf
        {peer = 2 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
  }
  wafer.tile.module card_id = 0 tile_id = 1 {}
}
