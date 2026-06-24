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

  func.func @invalid_rank_group_out_of_mesh(
      %input: tensor<2x4xf32>,
      %out: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK: linalg-ext collective rank_group logical ranks must be within execution mesh rank count
    %0 = wafer.linalg_ext.collective.all_gather
        ins(%input : tensor<2x4xf32>)
        outs(%out : tensor<4x4xf32>)
        {axis = 0 : i64, rank_group = array<i64: 0, 2>}
        -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
