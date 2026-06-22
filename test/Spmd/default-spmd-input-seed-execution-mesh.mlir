// REQUIRES: shardy
// RUN: wafer-opt --wafer-apply-default-spmd-sharding %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 2, 4>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 8>,
       policy = "all_available",
       endpoints = array<i64>}

  func.func @no_user_seed(
      %x: tensor<16x8xf32>,
      %small: tensor<7xf32>) -> tensor<16x8xf32> {
    return %x : tensor<16x8xf32>
  }
}

// CHECK: sdy.mesh @wafer_default_tile_mesh = <["rank"=8]>
// CHECK: func.func @no_user_seed(
// CHECK-SAME: %{{[^:]+}}: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{"rank"}, {}]>}
// CHECK-SAME: %{{[^:]+}}: tensor<7xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{}], replicated={"rank"}>}
