// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{physical-card-id=0 physical-tile-id=0 transport-status-argument-index=0})' %s | FileCheck --check-prefix=CURRENT %s
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{physical-card-id=0 physical-tile-id=0 transport-status-argument-index=0})' %s | mlir-translate --mlir-to-llvmir | FileCheck --check-prefix=LLVMIR %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["rank"], shape = array<i64: 16>}

  func.func @main(%status: i64) {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf32, #wafer.memory<spm, tensor>>
    %recv = wafer.instr.dte_recv %dst
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 10, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 65792, route_bindings = [], completion = sender_wait_receiver_fsm>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    %selector = arith.constant 1 : i64
    %send = wafer.instr.dte_send %src, %selector
        {peer = 1 : i64, bytes = 16 : i64,
         message = #wafer.dte_message<communication = 9, round = 0, slice = 0>,
         binding = #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = selector_table, remote_receiver_address = 65792, route_bindings = [0, 65792, 0, 1, 66048, 1], completion = sender_wait_receiver_fsm>}
        : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %send, %recv : !async.token, !async.token
    return
  }
}

// CURRENT-LABEL: llvm.func @main
// CURRENT: llvm.call @wafer_tx81_direct_dte_begin
// CURRENT: %[[RECV:.*]] = llvm.call @wafer_tx81_direct_dte_recv_prepare
// CURRENT-SAME: : (i64, i32, i32, i32, i32) -> i64
// CURRENT: llvm.urem
// CURRENT: llvm.icmp "eq"
// CURRENT: llvm.select
// CURRENT: llvm.select
// CURRENT: %[[SEND:.*]] = llvm.call @wafer_tx81_direct_dte_send_prepare
// CURRENT-SAME: : (i64, i64, i32, i32, i32, i32, i32) -> i64
// CURRENT-NEXT: llvm.call @wafer_tx81_direct_dte_send_issue_v3(%[[SEND]])
// CURRENT: llvm.call @wafer_tx81_direct_dte_wait(%[[SEND]])
// CURRENT: llvm.call @wafer_tx81_direct_dte_wait(%[[RECV]])
// CURRENT: llvm.call @wafer_tx81_direct_dte_finish
// CURRENT: llvm.return

// LLVMIR: call void @wafer_tx81_direct_dte_begin
// LLVMIR: call i64 @wafer_tx81_direct_dte_recv_prepare
// LLVMIR: call i64 @wafer_tx81_direct_dte_send_prepare
// LLVMIR-NEXT: call void @wafer_tx81_direct_dte_send_issue_v3
// LLVMIR: call void @wafer_tx81_direct_dte_wait
// LLVMIR: call void @wafer_tx81_direct_dte_finish
