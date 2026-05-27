// REQUIRES: shardy
// RUN: wafer-opt --wafer-apply-default-spmd-sharding %s | FileCheck %s --check-prefix=DEFAULT
// RUN: wafer-opt --wafer-apply-default-spmd-sharding="tile-count=1" %s | FileCheck %s --check-prefix=TILE1
// RUN: not wafer-opt --wafer-apply-default-spmd-sharding="tile-count=0" %s 2>&1 | FileCheck %s --check-prefix=BADTILE

func.func @no_user_seed(
    %x: tensor<4096x4096xf32>,
    %w: tensor<4096x4096xf32>,
    %small: tensor<7xf32>,
    %scalar: tensor<f32>) -> tensor<4096x4096xf32> {
  return %x : tensor<4096x4096xf32>
}

sdy.mesh @user_mesh = <["tp"=2]>

func.func @existing_arg_seed(
    %x: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@user_mesh, [{"tp"}, {}]>},
    %w: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
  return %x : tensor<4096x4096xf32>
}

func.func @existing_intermediate_seed(
    %x: tensor<4096x4096xf32>,
    %w: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
  %0 = sdy.sharding_constraint %x <@user_mesh, [{"tp"}, {}]> : tensor<4096x4096xf32>
  return %0 : tensor<4096x4096xf32>
}

// DEFAULT: sdy.mesh @wafer_default_tile_mesh = <["tile"=16]>
// DEFAULT: func.func @no_user_seed(
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{"tile"}, {}]>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{"tile"}, {}]>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<7xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{}], replicated={"tile"}>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<f32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [], replicated={"tile"}>}
// DEFAULT-SAME: -> tensor<4096x4096xf32>
// DEFAULT-NOT: -> (tensor<4096x4096xf32> {sdy.sharding

// DEFAULT: func.func @existing_arg_seed(
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@user_mesh, [{"tp"}, {}]>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32>)
// DEFAULT: func.func @existing_intermediate_seed(
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32>,
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32>)
// DEFAULT: sdy.sharding_constraint

// TILE1: sdy.mesh @wafer_default_tile_mesh = <["tile"=1]>
// TILE1: func.func @no_user_seed(
// TILE1-SAME: %{{[^:]+}}: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{}, {}], replicated={"tile"}>}
// TILE1-SAME: %{{[^:]+}}: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{}, {}], replicated={"tile"}>}
// TILE1-SAME: %{{[^:]+}}: tensor<7xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{}], replicated={"tile"}>}
// TILE1-SAME: %{{[^:]+}}: tensor<f32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [], replicated={"tile"}>}

// BADTILE: tile-count must be in [1, 16]
