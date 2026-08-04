// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66048' %s | FileCheck %s

func.func @ncc_join_extends_prior_local_write(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %source, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    %before_drain = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %before_drain, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    %after_drain = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %after_drain, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @ncc_join_extends_prior_local_write
// CHECK: %[[SOURCE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[SOURCE]]
// CHECK: %[[BEFORE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[BEFORE]]
// CHECK: wafer.instr.ncc_join [0]
// CHECK: %[[AFTER:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[AFTER]]

func.func @dte_recv_token_extends_destination_until_wait(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %dest = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %dest {peer = 1 : i64, bytes = 256 : i64,
        message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.ncc_join [0]
    %before_wait = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %before_wait, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.instr.dte_wait %token : !async.token
    %after_wait = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %after_wait, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @dte_recv_token_extends_destination_until_wait
// CHECK: %[[DEST:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[TOKEN:.+]] = wafer.instr.dte_recv %[[DEST]]
// CHECK: wafer.instr.ncc_join [0]
// CHECK: %[[BEFORE_WAIT:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[BEFORE_WAIT]]
// CHECK: wafer.instr.dte_wait %[[TOKEN]]
// CHECK: %[[AFTER_WAIT:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[AFTER_WAIT]]

func.func @ncc_join_extends_wdma_source_read(
    %output: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%output
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %source to %out
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    %before_fence = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %before_fence, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    %after_fence = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %after_fence, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.tile.yield %out : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @ncc_join_extends_wdma_source_read
// CHECK: %[[WDMA_SOURCE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.wdma %[[WDMA_SOURCE]]
// CHECK: %[[BEFORE_FENCE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[BEFORE_FENCE]]
// CHECK: wafer.instr.ncc_join [0]
// CHECK: %[[AFTER_FENCE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[AFTER_FENCE]]

func.func @branch_ncc_joins_clear_covered_write(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1) {
  %region = wafer.tile.region(%boundary, %cond
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %c: i1):
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %source, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    scf.if %c {
      wafer.instr.ncc_join [0]
    } else {
      wafer.instr.ncc_join [0]
    }
    %after_branch = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %after_branch, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @branch_ncc_joins_clear_covered_write
// CHECK: %[[BR_SOURCE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[BR_SOURCE]]
// CHECK: scf.if
// CHECK: %[[AFTER_BRANCH:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[AFTER_BRANCH]]

func.func @branch_dte_waits_clear_all_paths(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1) {
  %region = wafer.tile.region(%boundary, %cond
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %c: i1):
    %source = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %source
        {peer = 1 : i64, bytes = 256 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    scf.if %c {
      wafer.instr.dte_wait %token : !async.token
    } else {
      wafer.instr.dte_wait %token : !async.token
    }
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @branch_dte_waits_clear_all_paths
// CHECK: %[[DTE_SOURCE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[DTE_TOKEN:.+]] = wafer.instr.dte_send %[[DTE_SOURCE]]
// CHECK: scf.if
// CHECK: wafer.instr.dte_wait %[[DTE_TOKEN]]
// CHECK: else
// CHECK: wafer.instr.dte_wait %[[DTE_TOKEN]]

func.func @typed_dte_resource_does_not_observe_disjoint_pending_ncc_buffer(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %produced = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %received = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %produced, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    %token = wafer.instr.dte_recv %received
        {peer = 1 : i64, bytes = 256 : i64,
         message = #wafer.dte_message<communication = 1, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @typed_dte_resource_does_not_observe_disjoint_pending_ncc_buffer
// CHECK: %[[PRODUCED:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
// CHECK: %[[RECEIVED:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
// CHECK: wafer.instr.fill %[[PRODUCED]]
// CHECK: %[[RECV_TOKEN:.+]] = wafer.instr.dte_recv %[[RECEIVED]]
// CHECK: wafer.instr.dte_wait %[[RECV_TOKEN]]
