// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-to-c-abi-skeleton %s | FileCheck %s --check-prefix=IR
// RUN: %python %wafer_src_root/tools/wafer_package_manifest.py --emit-m0-smoke > %t.manifest.json
// RUN: %python %wafer_src_root/tools/wafer_package_manifest.py --validate %t.manifest.json
// RUN: %python %wafer_src_root/tools/wafer_emit_c_abi_stub.py --manifest %t.manifest.json > %t.c
// RUN: cc -fsyntax-only %t.c

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
// IR: wafer.abi.rdma_1d <issue_only>
// IR-NOT: wafer.compute.gemm
// IR: wafer.abi.gemm <issue_only>
// IR-NOT: wafer.store_tile
// IR: wafer.abi.wdma_1d <issue_only>
