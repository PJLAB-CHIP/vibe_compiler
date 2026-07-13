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

  %local = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %gather = "builtin.unrealized_conversion_cast"()
      : () -> memref<8xf32, #wafer.memory<spm, tensor>>

  wafer.tile.all_gather %local into %gather
      {bytes = 16 : i64, group_size = 2 : i64, local_rank = 0 : i64,
       rank_group = array<i64: 0, 2>, communication_id = 35 : i64}
      : memref<4xf32, #wafer.memory<spm, tensor>>
      -> memref<8xf32, #wafer.memory<spm, tensor>>
}

// CHECK: all_gather rank_group logical ranks must be within execution mesh rank count
