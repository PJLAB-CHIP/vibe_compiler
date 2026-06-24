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

  func.func @invalid_source_target_pairs_out_of_mesh(
      %input: tensor<2x4xf32>,
      %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
    // CHECK: source_target_pairs logical ranks must be within execution mesh rank count
    %0 = wafer.linalg_ext.collective.collective_permute
        ins(%input : tensor<2x4xf32>)
        outs(%out : tensor<2x4xf32>)
        {source_target_pairs = array<i64: 0, 2>}
        -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}
