// RUN: wafer-opt --allow-unregistered-dialect --wafer-plan-spm-memory -split-input-file -verify-diagnostics %s

// -----

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

func.func @ncc_join_does_not_complete_dte(
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
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

func.func @ncc_join_does_not_complete_dte(
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
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

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
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without a matching participant in wafer.instr.ncc_join}}
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

func.func @branch_only_one_ncc_join(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1) {
  %region = wafer.tile.region(%boundary, %cond
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %c: i1):
    %zero = arith.constant 0.000000e+00 : f16
    %buffer = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without a matching participant in wafer.instr.ncc_join}}
    wafer.instr.fill %buffer, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    scf.if %c {
      wafer.instr.ncc_join [0]
    } else {
    }
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

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
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
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

func.func @loop_may_skip_only_ncc_join(
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
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without a matching participant in wafer.instr.ncc_join}}
    wafer.instr.fill %buffer, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    scf.for %i = %l to %u step %s {
      wafer.instr.ncc_join [0]
    }
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

// A loop-local allocation has a fresh logical instance on each iteration, but
// its value-associated NCC accesses still name a resolved runtime address
// domain. The same worker therefore orders a prior WDMA read against the next
// iteration's fill when their physical ranges overlap; disjoint ranges require
// no completion edge. The join after the loop remains the real domain-exit cut.
wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

func.func @loop_local_origin_is_same_worker_ordered(
    %boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<4xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.000000e+00 : f16
    scf.for %i = %c0 to %c4 step %c2 {
      %buffer = memref.alloc()
          : memref<2xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %buffer, %zero
          : memref<2xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.ncc_join [0]
      %token = wafer.instr.dte_send %buffer
          {peer = 1 : i64, bytes = 4 : i64,
           message = #wafer.dte_message<communication = 7, round = 0, slice = 0>}
          : memref<2xf16, #wafer.memory<spm, tensor>> -> !async.token
      wafer.instr.dte_wait %token : !async.token
      wafer.instr.wdma %buffer to %arg0
          {byte_count = 4 : i64, dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>, inner_bytes = 4 : i64}
          : memref<2xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @loop_body_same_worker_stream_reaches_outer_completion(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lb: index, %ub: index, %step: index) {
  %region = wafer.tile.region(%boundary, %lb, %ub, %step
      : memref<128xf16, #wafer.memory<ddr, tensor>>, index, index, index)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %l: index, %u: index, %s: index):
    %zero = arith.constant 0.000000e+00 : f16
    scf.for %i = %l to %u step %s {
      %buffer = memref.alloc()
          : memref<128xf16, #wafer.memory<spm, tensor>>
      // Same-worker issue order covers consecutive loop iterations. The
      // unconditional outer completion covers the remaining pending stream.
      wafer.instr.fill %buffer, %zero
          : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    }
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

func.func @identity_loop_carried_dte_token_is_proven(
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
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    %looped = scf.for %i = %l to %u step %s
        iter_args(%iter = %token) -> (!async.token) {
      scf.yield %iter : !async.token
    }
    wafer.instr.dte_wait %looped : !async.token
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

func.func @dynamic_loop_local_dte_token_is_fail_closed(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lb: index, %ub: index, %step: index) {
  %region = wafer.tile.region(%boundary, %lb, %ub, %step
      : memref<128xf16, #wafer.memory<ddr, tensor>>, index, index, index)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %l: index, %u: index, %s: index):
    %source = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %initial = wafer.instr.dte_send %source
        {peer = 1 : i64, bytes = 256 : i64,
         message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    // expected-error @below {{unsupported_async_completion_flow: dynamically optional loop-carried DTE issue has no exact completion instance proof}}
    %looped = scf.for %i = %l to %u step %s
        iter_args(%iter = %initial) -> (!async.token) {
      %next = wafer.instr.dte_send %source
          {peer = 1 : i64, bytes = 256 : i64,
           message = #wafer.dte_message<communication = 0, round = 1, slice = 0>}
          : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
      scf.yield %next : !async.token
    }
    wafer.instr.dte_wait %looped : !async.token
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @dealloc_cannot_observe_pending_ncc_write(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %buffer = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without a matching participant in wafer.instr.ncc_join}}
    wafer.instr.fill %buffer, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    memref.dealloc %buffer
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @unknown_effect_cannot_cross_pending_ncc_write(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %buffer = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without a matching participant in wafer.instr.ncc_join}}
    wafer.instr.fill %buffer, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    "test.unknown_observer"() : () -> ()
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

wafer.target.topology @default
    {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
     tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}

func.func @dte_buffer_write_cannot_cross_pending_ncc_write(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %buffer = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{missing_local_completion: local Compute/Movement issue has a reachable path to wafer.tile.region exit without a matching participant in wafer.instr.ncc_join}}
    wafer.instr.fill %buffer, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    %token = wafer.instr.dte_recv %buffer
        {peer = 1 : i64, bytes = 256 : i64,
         message = #wafer.dte_message<communication = 1, round = 0, slice = 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
    wafer.instr.ncc_join [0]
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}
