// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 2>,
       policy = "all_available",
       endpoints = array<i64>}

  %buf = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %send = wafer.instr.dte_send %buf {peer = 2 : i64, bytes = 16 : i64,
      message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: DTE peer logical rank must be within execution mesh rank count
