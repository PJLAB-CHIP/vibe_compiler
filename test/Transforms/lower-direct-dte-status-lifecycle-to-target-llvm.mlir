// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{target-profile=wafer-tx81-single-card-kernel-v1 transport-prepared-before-entry=true logical-rank=15 transport-status-argument-index=0})' %s | FileCheck --check-prefix=CLUSTER %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{target-profile=wafer-tx81-single-card-kernel-v1 logical-rank=15 transport-status-argument-index=0})' %s | FileCheck --check-prefix=STANDALONE %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}

  // This rank has the transport status slot but no local Direct DTE op.
  func.func @main(%status: i64) {
    return
  }
}

// CLUSTER-LABEL: llvm.func @main
// CLUSTER-SAME: (%[[STATUS:.*]]: i64)
// CLUSTER: %[[RANK_COUNT:.*]] = llvm.mlir.constant(16 : i32) : i32
// CLUSTER-NEXT: llvm.call @wafer_tx81_direct_dte_begin_after_prepare(%[[STATUS]], %[[RANK_COUNT]])
// CLUSTER-NEXT: llvm.call @wafer_tx81_direct_dte_finish
// CLUSTER-NEXT: llvm.return
// CLUSTER-NOT: llvm.call @wafer_tx81_direct_dte_begin(
// CLUSTER-NOT: wafer_tx81_direct_dte_send_prepare
// CLUSTER-NOT: wafer_tx81_direct_dte_recv_prepare

// STANDALONE-LABEL: llvm.func @main
// STANDALONE-SAME: (%[[STATUS:.*]]: i64)
// STANDALONE: %[[RANK_COUNT:.*]] = llvm.mlir.constant(16 : i32) : i32
// STANDALONE-NEXT: llvm.call @wafer_tx81_direct_dte_begin(%[[STATUS]], %[[RANK_COUNT]])
// STANDALONE-NEXT: llvm.call @wafer_tx81_direct_dte_finish
// STANDALONE-NEXT: llvm.return
// STANDALONE-NOT: wafer_tx81_direct_dte_begin_after_prepare
