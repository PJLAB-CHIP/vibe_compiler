// REQUIRES: shardy
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-propagate-stablehlo-sharding{default-tile-count=16})' %s | FileCheck %s --check-prefix=DEFAULT
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-propagate-stablehlo-sharding{default-tile-count=1})' %s | FileCheck %s --check-prefix=TILE1

module {
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

// DEFAULT: sdy.mesh @wafer_default_tile_mesh = <["tile"=16]>
// DEFAULT: %{{[^:]+}}: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{"tile"}, {}]>}
// DEFAULT-SAME: %{{[^:]+}}: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{"tile"}, {}]>}
// DEFAULT: stablehlo.dot_general

// TILE1: sdy.mesh @wafer_default_tile_mesh = <["tile"=1]>
// TILE1: %{{[^:]+}}: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{}, {}], replicated={"tile"}>}
// TILE1-SAME: %{{[^:]+}}: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@wafer_default_tile_mesh, [{}, {}], replicated={"tile"}>}
