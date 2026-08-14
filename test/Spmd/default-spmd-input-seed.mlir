// REQUIRES: shardy
// RUN: wafer-opt --wafer-apply-default-spmd-sharding %s | FileCheck %s --check-prefix=DEFAULT

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}

  // The pass derives the unique execution mesh by typed module membership;
  // its symbol spelling is not part of the sharding contract.
  wafer.execution.mesh @logical_card_partitions
      {axes = ["card"],
       shape = array<i64: 1>}

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

  func.func @frontend_activation_seed(
      %x: tensor<1x4x64xf32>) -> tensor<1x4x64xf32> {
    %0 = stablehlo.custom_call @Sharding(%x)
        {backend_config = "",
         mhlo.sharding = "{devices=[1,1,1]0}"}
        : (tensor<1x4x64xf32>) -> tensor<1x4x64xf32>
    return %0 : tensor<1x4x64xf32>
  }
}

// DEFAULT: sdy.mesh @wafer_default_card_mesh = <["card"=1]>
// DEFAULT: func.func @no_user_seed(
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@wafer_default_card_mesh, [{}, {}], replicated={"card"}>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@wafer_default_card_mesh, [{}, {}], replicated={"card"}>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<7xf32> {sdy.sharding = #sdy.sharding<@wafer_default_card_mesh, [{}], replicated={"card"}>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<f32> {sdy.sharding = #sdy.sharding<@wafer_default_card_mesh, [], replicated={"card"}>}
// DEFAULT-SAME: -> tensor<4096x4096xf32>
// DEFAULT-NOT: -> (tensor<4096x4096xf32> {sdy.sharding

// DEFAULT: func.func @existing_arg_seed(
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32> {sdy.sharding = #sdy.sharding<@user_mesh, [{"tp"}, {}]>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32>)
// DEFAULT: func.func @existing_intermediate_seed(
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32>,
// DEFAULT-SAME: %{{[^:]+}}: tensor<4096x4096xf32>)
// DEFAULT: sdy.sharding_constraint

// DEFAULT: func.func @frontend_activation_seed(
// DEFAULT: sdy.sharding_constraint %{{[^ ]+}} <@wafer_default_card_mesh, [{}, {}, {}], replicated={"card"}>
// DEFAULT-NOT: stablehlo.custom_call @Sharding
