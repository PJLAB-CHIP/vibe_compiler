// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{target-profile=wafer-tx81-single-card-kernel-v1 logical-rank=0 transport-status-argument-index=0})' %s | FileCheck --check-prefix=V1 %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{target-profile=wafer-tx81-single-card-kernel-v2 logical-rank=0 transport-status-argument-index=0})' %s | FileCheck --check-prefix=V2 %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{target-profile=wafer-tx81-single-card-kernel-v3 logical-rank=0 transport-status-argument-index=0})' %s | FileCheck --check-prefix=V3 %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{target-profile=wafer-tx81-single-card-kernel-v3 logical-rank=0 transport-status-argument-index=0})' %s | mlir-translate --mlir-to-llvmir | FileCheck --check-prefix=LLVMIR %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}

  func.func @main(%status: i64) {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %recv = wafer.instr.dte_recv %dst
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 10, phase = collective_permute, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 65792, route_bindings = [], completion = sender_wait_receiver_fsm>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %selector = arith.constant 1 : i64
    %send = wafer.instr.dte_send %src, %selector
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, phase = collective_permute, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = selector_table, remote_receiver_address = 65792, route_bindings = [0, 65792, 0, 1, 66048, 1], completion = sender_wait_receiver_fsm>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %send, %recv : !async.token, !async.token
    return
  }
}

// V1-LABEL: llvm.func @main
// V1: %[[V1_SEND:.*]] = llvm.call @wafer_tx81_direct_dte_send_prepare
// V1-NOT: llvm.call @wafer_tx81_direct_dte_send_issue_v3
// V1: llvm.call @wafer_tx81_direct_dte_wait(%[[V1_SEND]])
// V1: llvm.return

// V2-LABEL: llvm.func @main
// V2: %[[V2_SEND:.*]] = llvm.call @wafer_tx81_direct_dte_send_prepare
// V2-NOT: llvm.call @wafer_tx81_direct_dte_send_issue_v3
// V2: llvm.call @wafer_tx81_direct_dte_wait(%[[V2_SEND]])
// V2: llvm.return

// V3-LABEL: llvm.func @main
// V3: llvm.call @wafer_tx81_direct_dte_begin
// V3: %[[V3_RECV:.*]] = llvm.call @wafer_tx81_direct_dte_recv_prepare
// V3-SAME: : (i64, i32, i32, i32, i32) -> i64
// V3: llvm.urem
// V3: llvm.icmp "eq"
// V3: llvm.select
// V3: llvm.select
// V3: %[[V3_SEND:.*]] = llvm.call @wafer_tx81_direct_dte_send_prepare
// V3-SAME: : (i64, i64, i32, i32, i32, i32, i32) -> i64
// V3-NEXT: llvm.call @wafer_tx81_direct_dte_send_issue_v3(%[[V3_SEND]])
// V3: llvm.call @wafer_tx81_direct_dte_wait(%[[V3_SEND]])
// V3: llvm.call @wafer_tx81_direct_dte_wait(%[[V3_RECV]])
// V3: llvm.call @wafer_tx81_direct_dte_finish
// V3: llvm.return

// LLVMIR: call void @wafer_tx81_direct_dte_begin
// LLVMIR: call i64 @wafer_tx81_direct_dte_recv_prepare
// LLVMIR: call i64 @wafer_tx81_direct_dte_send_prepare
// LLVMIR-NEXT: call void @wafer_tx81_direct_dte_send_issue_v3
// LLVMIR: call void @wafer_tx81_direct_dte_wait
// LLVMIR: call void @wafer_tx81_direct_dte_finish
