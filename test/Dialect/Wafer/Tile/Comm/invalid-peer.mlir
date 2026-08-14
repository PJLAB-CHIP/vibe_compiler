// RUN: split-file %s %t
// RUN: wafer-opt -verify-diagnostics %t/out-of-topology.mlir
// RUN: wafer-opt -verify-diagnostics %t/ddr-buffer.mlir
// RUN: wafer-opt -verify-diagnostics %t/oversized-payload.mlir

//--- out-of-topology.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.card.program card_id = 0 {
    wafer.tile.program tile_id = 0 {
      %buffer = "builtin.unrealized_conversion_cast"()
          : () -> memref<4xf32, #wafer.memory<spm, tensor>>
      // expected-error @below {{peer tile_id 2 is outside the available physical Tile domain for card_id 0}}
      %token = wafer.tile.peer_recv %buffer
          {peer = 2 : i64, bytes = 16 : i64,
           message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
          : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    }
    wafer.tile.program tile_id = 1 {}
  }
}

//--- ddr-buffer.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<ddr, tensor>>
  // expected-error @below {{operand #0 must be Wafer SPM memref}}
  %token = wafer.tile.peer_send %buffer
      {peer = 1 : i64, bytes = 16 : i64,
       message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<ddr, tensor>> -> !async.token
}

//--- oversized-payload.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card"], shape = array<i64: 1>}
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  // expected-error @below {{DTE byte count exceeds buffer physical byte size}}
  %token = wafer.tile.peer_recv %buffer
      {peer = 1 : i64, bytes = 32 : i64,
       message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}
