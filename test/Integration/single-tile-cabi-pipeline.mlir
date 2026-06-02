// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-linalg-to-cabi{target=wafer})' %s | FileCheck %s --check-prefix=IR

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

// IR-LABEL: func.func @single_matmul(
// IR: wafer.ddr.external_binding <input>
// IR: wafer.ddr.external_binding <output>
// IR: wafer.tile_region
// IR-NOT: wafer.load_tile
// IR: wafer.abi.rdma <issue_only>
// IR-NOT: wafer.compute.gemm
// IR: wafer.abi.gemm <issue_only>
// IR-NOT: wafer.store_tile
// IR: wafer.abi.wdma <issue_only>
