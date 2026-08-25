// RUN: wafer-opt %s -o /dev/null
// Partition bounds are module-stage facts; the standalone operation remains
// locally verifier-legal and is rejected by the compiler stage check.

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {axes = ["card"],
       shape = array<i64: 2>}

  func.func @invalid_partition_group_out_of_mesh(
      %input: tensor<2x4xf32>) -> tensor<4x4xf32> {
    %out = tensor.empty() : tensor<4x4xf32>
    %0 = wafer.linalg_ext.collective.all_gather
        ins(%input : tensor<2x4xf32>)
        outs(%out : tensor<4x4xf32>)
        {axis = 0 : i64, partition_group = array<i64: 0, 2>}
        -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
