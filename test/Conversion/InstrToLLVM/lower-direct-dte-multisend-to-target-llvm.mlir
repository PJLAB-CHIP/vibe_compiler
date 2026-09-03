// RUN: wafer-opt --split-input-file --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{card-id=0 tile-id=0 transport-status-argument-index=0})' %s | FileCheck --check-prefix=CURRENT %s
// RUN: wafer-opt --split-input-file --pass-pipeline='builtin.module(wafer-lower-instr-to-target-llvm{card-id=0 tile-id=0 transport-status-argument-index=0})' %s | mlir-translate --split-input-file --mlir-to-llvmir | FileCheck --check-prefix=LLVMIR %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["rank"], shape = array<i64: 16>}

  func.func @broadcast(%status: i64) {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x2x64xf16, #wafer.memory<spm, tensor>>
    %sent = wafer.instr.dte_broadcast %src
        {source_offset = 0 : i64, peers = array<i64: 1, 2>, bytes = 256 : i64,
         messages = [#wafer.dte_message<communication = 70, round = 0, slice = 0>,
                     #wafer.dte_message<communication = 70, round = 0, slice = 1>],
         bindings = [#wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 65792, route_bindings = [], completion = sender_wait_receiver_fsm>,
                     #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 1, remote_address_mode = absolute, remote_receiver_address = 66048, route_bindings = [], completion = sender_wait_receiver_fsm>]}
        : memref<1x2x64xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %sent : !async.token
    return
  }
}

// -----

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["rank"], shape = array<i64: 16>}

  func.func @scatter(%status: i64) {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x1025x128xf16, #wafer.memory<spm, tensor>>
    %sent = wafer.instr.dte_scatter %src
        {source_offset = 256 : i64, peers = array<i64: 1, 8, 15, 7>, bytes = 256 : i64,
         messages = [#wafer.dte_message<communication = 71, round = 0, slice = 0>,
                     #wafer.dte_message<communication = 71, round = 0, slice = 1>,
                     #wafer.dte_message<communication = 71, round = 0, slice = 2>,
                     #wafer.dte_message<communication = 71, round = 0, slice = 3>],
         bindings = [#wafer.direct_dte_binding<allocation = normal, receiver_fsm = 0, remote_address_mode = absolute, remote_receiver_address = 65792, route_bindings = [], completion = sender_wait_receiver_fsm>,
                     #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 1, remote_address_mode = absolute, remote_receiver_address = 66048, route_bindings = [], completion = sender_wait_receiver_fsm>,
                     #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 2, remote_address_mode = absolute, remote_receiver_address = 66304, route_bindings = [], completion = sender_wait_receiver_fsm>,
                     #wafer.direct_dte_binding<allocation = normal, receiver_fsm = 3, remote_address_mode = absolute, remote_receiver_address = 66560, route_bindings = [], completion = sender_wait_receiver_fsm>]}
        : memref<1x1025x128xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %sent : !async.token
    return
  }
}

// CURRENT-LABEL: llvm.func @broadcast
// CURRENT: llvm.call @wafer_tx81_direct_dte_begin
// CURRENT: %[[MULTI:.*]] = llvm.call @wafer_tx81_direct_dte_multisend_prepare
// CURRENT-SAME: : (i64, i32, i32, i32, i32, i32) -> i64
// CURRENT: llvm.call @wafer_tx81_direct_dte_multisend_add_destination(%[[MULTI]]
// CURRENT: llvm.call @wafer_tx81_direct_dte_multisend_add_destination(%[[MULTI]]
// CURRENT: llvm.call @wafer_tx81_direct_dte_send_issue(%[[MULTI]])
// CURRENT: llvm.call @wafer_tx81_direct_dte_wait(%[[MULTI]])
// CURRENT: llvm.call @wafer_tx81_direct_dte_finish
// CURRENT: llvm.return

// CURRENT-LABEL: llvm.func @scatter
// CURRENT: %[[SOURCE:.*]] = llvm.add
// CURRENT: %[[SCATTER:.*]] = llvm.call @wafer_tx81_direct_dte_multisend_prepare(%[[SOURCE]]
// CURRENT-COUNT-4: llvm.call @wafer_tx81_direct_dte_multisend_add_destination(%[[SCATTER]]
// CURRENT: llvm.call @wafer_tx81_direct_dte_send_issue(%[[SCATTER]])
// CURRENT: llvm.call @wafer_tx81_direct_dte_wait(%[[SCATTER]])
// CURRENT: llvm.call @wafer_tx81_direct_dte_finish
// CURRENT: llvm.return

// LLVMIR-LABEL: define void @broadcast
// LLVMIR: call i64 @wafer_tx81_direct_dte_multisend_prepare
// LLVMIR-COUNT-2: call void @wafer_tx81_direct_dte_multisend_add_destination
// LLVMIR: call void @wafer_tx81_direct_dte_send_issue
// LLVMIR: call void @wafer_tx81_direct_dte_wait

// LLVMIR-LABEL: define void @scatter
// LLVMIR: call i64 @wafer_tx81_direct_dte_multisend_prepare
// LLVMIR-COUNT-4: call void @wafer_tx81_direct_dte_multisend_add_destination
// LLVMIR: call void @wafer_tx81_direct_dte_send_issue
// LLVMIR: call void @wafer_tx81_direct_dte_wait
