// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-local-linalg-to-cabi{target=wafer})' %s | FileCheck %s --check-prefix=IR
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-local-linalg-to-cabi{target=unknown})' %s 2>&1 | FileCheck %s --check-prefix=BAD-TARGET

module {
  func.func @single_matmul(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%out : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// IR: module attributes {wafer.target = #wafer.target<wafer>}
// IR-LABEL: func.func @single_matmul(
// IR: wafer.ddr.external_binding <input>
// IR: wafer.tile_region
// IR: wafer.abi.gemm <issue_only>
// IR: wafer.abi.wdma <issue_only>
// IR-NOT: linalg.matmul

// BAD-TARGET: unsupported Wafer compile target 'unknown'; expected 'wafer'
