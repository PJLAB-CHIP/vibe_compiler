// RUN: wafer-opt --wafer-plan-spm-memory -split-input-file -verify-diagnostics %s

// -----

func.func @local_fence_does_not_complete_dte(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %source = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{missing_dte_completion: DTE token has a reachable path to wafer.tile.region exit without wafer.instr.dte_wait}}
    %token = wafer.instr.dte_send %source
        {peer = 1 : i64, bytes = 256 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.local_fence
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @dte_wait_does_not_complete_local_engine(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %source = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %source
        {peer = 1 : i64, bytes = 256 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without wafer.instr.local_fence}}
    wafer.instr.wdma %source to %arg0
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.dte_wait %token : !async.token
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @branch_only_one_local_fence(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1) {
  %region = wafer.tile.region(%boundary, %cond
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %c: i1):
    %zero = arith.constant 0.000000e+00 : f16
    %buffer = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without wafer.instr.local_fence}}
    wafer.instr.fill %buffer, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    scf.if %c {
      wafer.instr.local_fence
    } else {
    }
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @branch_only_one_dte_wait(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1) {
  %region = wafer.tile.region(%boundary, %cond
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %c: i1):
    %source = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{missing_dte_completion: DTE token has a reachable path to wafer.tile.region exit without wafer.instr.dte_wait}}
    %token = wafer.instr.dte_send %source
        {peer = 1 : i64, bytes = 256 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    scf.if %c {
      wafer.instr.dte_wait %token : !async.token
    } else {
    }
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @loop_may_skip_only_local_fence(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lb: index, %ub: index, %step: index) {
  %region = wafer.tile.region(%boundary, %lb, %ub, %step
      : memref<128xf16, #wafer.memory<ddr, tensor>>, index, index, index)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %l: index, %u: index, %s: index):
    %zero = arith.constant 0.000000e+00 : f16
    %buffer = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without wafer.instr.local_fence}}
    wafer.instr.fill %buffer, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    scf.for %i = %l to %u step %s {
      wafer.instr.local_fence
    }
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @loop_carried_dte_token_is_fail_closed(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lb: index, %ub: index, %step: index) {
  %region = wafer.tile.region(%boundary, %lb, %ub, %step
      : memref<128xf16, #wafer.memory<ddr, tensor>>, index, index, index)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %l: index, %u: index, %s: index):
    %source = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %source
        {peer = 1 : i64, bytes = 256 : i64,
         message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    // expected-error @below {{unsupported_completion_control_flow: SPM memory planning cannot prove loop-carried DTE token completion}}
    %looped = scf.for %i = %l to %u step %s
        iter_args(%iter = %token) -> (!async.token) {
      scf.yield %iter : !async.token
    }
    wafer.instr.dte_wait %looped : !async.token
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}
