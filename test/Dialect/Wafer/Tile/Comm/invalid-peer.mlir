// RUN: split-file %s %t
// RUN: wafer-opt -verify-diagnostics %t/wrong-phase.mlir
// RUN: wafer-opt -verify-diagnostics %t/out-of-mesh.mlir
// RUN: wafer-opt -verify-diagnostics %t/ddr-buffer.mlir
// RUN: wafer-opt -verify-diagnostics %t/oversized-payload.mlir

//--- wrong-phase.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  // expected-error @below {{peer tile message phase must be peer_dataflow, all_to_all, or collective_permute}}
  %token = wafer.tile.peer_send %buffer
      {peer = 1 : i64, bytes = 16 : i64,
       message = #wafer.dte_message<communication = 0, phase = all_gather_ring, round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}

//--- out-of-mesh.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  // expected-error @below {{peer tile logical rank must be within execution mesh rank count}}
  %token = wafer.tile.peer_recv %buffer
      {peer = 2 : i64, bytes = 16 : i64,
       message = #wafer.dte_message<communication = 0, phase = peer_dataflow, round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}

//--- ddr-buffer.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<ddr, tensor>>
  // expected-error @below {{operand #0 must be Wafer SPM memref}}
  %token = wafer.tile.peer_send %buffer
      {peer = 1 : i64, bytes = 16 : i64,
       message = #wafer.dte_message<communication = 0, phase = peer_dataflow, round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<ddr, tensor>> -> !async.token
}

//--- oversized-payload.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  // expected-error @below {{DTE byte count exceeds buffer physical byte size}}
  %token = wafer.tile.peer_recv %buffer
      {peer = 1 : i64, bytes = 32 : i64,
       message = #wafer.dte_message<communication = 0, phase = peer_dataflow, round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}
