// REQUIRES: shardy
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-propagate-stablehlo-sharding)' %s | FileCheck %s --check-prefix=DEFAULT

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 16>,
       policy = "all_available",
       endpoints = array<i64>}

  func.func @main(%lhs: tensor<32x16xf32>, %rhs: tensor<16x8xf32>) -> tensor<32x8xf32> {
    %0 = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<32x16xf32>, tensor<16x8xf32>) -> tensor<32x8xf32>
    return %0 : tensor<32x8xf32>
  }
}

// DEFAULT: sdy.mesh @wafer_default_tile_mesh = <["rank"=16]>
// DEFAULT: %{{[^:]+}}: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{"rank"}, {}]>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{"rank"}, {}]>}
// DEFAULT: stablehlo.dot_general
