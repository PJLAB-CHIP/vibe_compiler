// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-local-stablehlo-to-cabi{target=wafer})' %s | FileCheck %s --check-prefix=IR

module {
  func.func @stablehlo_matmul(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>) -> tensor<4x16xf16> {
    %0 = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x8xf16>, tensor<8x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// IR: module attributes {wafer.target = #wafer.target<wafer>}
// IR-LABEL: func.func @stablehlo_matmul(
// IR-NOT: stablehlo.
// IR: wafer.ddr.external_binding <input>
// IR: wafer.tile_region
// IR: wafer.abi.gemm <issue_only>
// IR-NOT: linalg.matmul
