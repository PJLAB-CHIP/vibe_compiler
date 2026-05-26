// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-multi-tile-no-comm --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-to-c-abi-skeleton %s | FileCheck %s --check-prefix=IR
// RUN: %python %wafer_src_root/tools/wafer_package_manifest.py --emit-m1-no-comm-smoke > %t.manifest.json
// RUN: %python %wafer_src_root/tools/wafer_package_manifest.py --validate %t.manifest.json
// RUN: %python %wafer_src_root/tools/wafer_emit_c_abi_stub.py --manifest %t.manifest.json > %t.c
// RUN: FileCheck %s --check-prefix=C < %t.c
// RUN: cc -fsyntax-only %t.c

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 2 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}

  func.func @two_tile_matmul(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%out : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// IR-LABEL: func.func @two_tile_matmul(
// IR: wafer.ddr.external_binding <input>
// IR: wafer.ddr.external_binding <output>
// IR-NOT: wafer.group
// IR: wafer.tile_region
// IR-NOT: wafer.load_tile
// IR: wafer.abi.rdma <issue_only>
// IR: wafer.abi.rdma <issue_only>
// IR-NOT: wafer.compute.gemm
// IR: wafer.abi.gemm <issue_only>
// IR-NOT: wafer.store_tile
// IR: wafer.abi.wdma <issue_only>
// IR: wafer.tile_region
// IR: wafer.abi.rdma <issue_only>
// IR: wafer.abi.rdma <issue_only>
// IR: wafer.abi.gemm <issue_only>
// IR: wafer.abi.wdma <issue_only>
// IR-NOT: wafer.comm

// C: static const wafer_tile_launch_arg_t k_m1_two_tile_no_comm_tile_args[] = {
// C: {0u, 0u, 0u, 0u, 0u, 0u},
// C: {1u, 1u, 0u, 0u, 0u, 1u},
// C: int m1_two_tile_no_comm_tile_arg_count(void)
