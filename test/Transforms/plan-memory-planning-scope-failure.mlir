// RUN: split-file %s %t
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-call.mlir 2>&1 | FileCheck --check-prefix=DDR-CALL %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-effectful-alias-call.mlir 2>&1 | FileCheck --check-prefix=DDR-EFFECTFUL-ALIAS %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-type-erased-effect-call.mlir 2>&1 | FileCheck --check-prefix=DDR-ERASED-EFFECT %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-mixed-alias-call.mlir 2>&1 | FileCheck --check-prefix=DDR-MIXED-ALIAS %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-external-call.mlir 2>&1 | FileCheck --check-prefix=DDR-EXTERNAL-CALL %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-external-async-call.mlir 2>&1 | FileCheck --check-prefix=DDR-EXTERNAL-ASYNC %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-indirect.mlir 2>&1 | FileCheck --check-prefix=DDR-INDIRECT %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-async-descriptor.mlir 2>&1 | FileCheck --check-prefix=DDR-ASYNC-DESCRIPTOR %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/ddr-async-global.mlir 2>&1 | FileCheck --check-prefix=DDR-ASYNC-GLOBAL %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/spm-yield.mlir 2>&1 | FileCheck --check-prefix=SPM-YIELD %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/spm-async-region.mlir 2>&1 | FileCheck --check-prefix=SPM-ASYNC-REGION %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/spm-parallel-region.mlir 2>&1 | FileCheck --check-prefix=SPM-PARALLEL %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/spm-call-region.mlir 2>&1 | FileCheck --check-prefix=SPM-CALL %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/spm-external-call.mlir 2>&1 | FileCheck --check-prefix=SPM-EXTERNAL-CALL %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/spm-external-async-call.mlir 2>&1 | FileCheck --check-prefix=SPM-EXTERNAL-ASYNC %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/spm-indirect-region.mlir 2>&1 | FileCheck --check-prefix=SPM-INDIRECT %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/spm-outside-region.mlir 2>&1 | FileCheck --check-prefix=SPM-OUTSIDE %s

// Whole-function/rank DDR and SPM placement is sound only when the accepted
// execution scope cannot dynamically overlap another independently planned
// rank arena. Interprocedural/concurrent arena summaries are not yet an IR
// complete physical-Tile IR, so these cases fail closed.

//--- ddr-call.mlir
func.func @ddr_callee() {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f16
  %callee = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  memref.store %one, %callee[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @ddr_caller() {
  %c0 = arith.constant 0 : index
  %caller = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %before = memref.load %caller[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  func.call @ddr_callee() : () -> ()
  %after = memref.load %caller[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

// DDR-CALL: unsupported_ddr_planning_scope
// DDR-CALL-SAME: interprocedural arena and resource summary

//--- ddr-effectful-alias-call.mlir
func.func private @read_and_forward(
    %arg: memref<128xf16, #wafer.memory<ddr, tensor>>)
    -> memref<128xf16, #wafer.memory<ddr, tensor>> {
  %c0 = arith.constant 0 : index
  %unused = memref.load %arg[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return %arg : memref<128xf16, #wafer.memory<ddr, tensor>>
}

func.func @call_effectful_alias() {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %alias = func.call @read_and_forward(%ddr)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>)
     -> memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

// DDR-EFFECTFUL-ALIAS: unsupported_ddr_planning_scope
// DDR-EFFECTFUL-ALIAS-SAME: interprocedural arena and resource summary

//--- ddr-type-erased-effect-call.mlir
func.func private @write_type_erased_alias(
    %arg: memref<128xf16, #wafer.memory<ddr, tensor>>) -> tensor<128xf16> {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f16
  %tensor = bufferization.to_tensor %arg restrict writable
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %generic = bufferization.to_memref %tensor : memref<128xf16>
  memref.store %one, %generic[%c0] : memref<128xf16>
  return %tensor : tensor<128xf16>
}

func.func @call_type_erased_effect() {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %tensor = func.call @write_type_erased_alias(%ddr)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> tensor<128xf16>
  return
}

// DDR-ERASED-EFFECT: unsupported_ddr_planning_scope
// DDR-ERASED-EFFECT-SAME: interprocedural arena and resource summary

//--- ddr-mixed-alias-call.mlir
func.func private @select_alias_or_independent(
    %arg: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1)
    -> tensor<128xf16> {
  %alias = bufferization.to_tensor %arg restrict writable
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %independent = tensor.empty() : tensor<128xf16>
  %selected = arith.select %cond, %alias, %independent : tensor<128xf16>
  return %selected : tensor<128xf16>
}

func.func @call_mixed_alias(%cond: i1) {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %tensor = func.call @select_alias_or_independent(%ddr, %cond)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
     -> tensor<128xf16>
  return
}

// DDR-MIXED-ALIAS: unsupported_ddr_planning_scope
// DDR-MIXED-ALIAS-SAME: interprocedural arena and resource summary

//--- ddr-external-call.mlir
func.func private @unknown_ddr_effect()

func.func @external_call_with_ddr_arena() {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  func.call @unknown_ddr_effect() : () -> ()
  return
}

// DDR-EXTERNAL-CALL: unsupported_ddr_planning_scope
// DDR-EXTERNAL-CALL-SAME: interprocedural arena and resource summary

//--- ddr-external-async-call.mlir
async.func private @unknown_async_ddr_effect() -> !async.token

func.func @external_async_call_with_ddr_arena() {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %task = async.call @unknown_async_ddr_effect() : () -> !async.token
  async.await %task : !async.token
  return
}

// DDR-EXTERNAL-ASYNC: unsupported_ddr_planning_scope
// DDR-EXTERNAL-ASYNC-SAME: external async.call

//--- ddr-indirect.mlir
func.func @indirect_ddr_callee() {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f16
  %callee = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  memref.store %one, %callee[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

async.func @indirect_ddr_wrapper() -> !async.token {
  %callee = func.constant @indirect_ddr_callee : () -> ()
  func.call_indirect %callee() : () -> ()
  return
}

func.func @indirect_ddr_caller() {
  %c0 = arith.constant 0 : index
  %caller = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %task = async.call @indirect_ddr_wrapper() : () -> !async.token
  %late = memref.load %caller[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  async.await %task : !async.token
  return
}

// DDR-INDIRECT: unsupported_ddr_planning_scope
// DDR-INDIRECT-SAME: indirect calls in a module with DDR demands

//--- ddr-async-descriptor.mlir
async.func @async_rdma(
    %ddr: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %ddr to %spm
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

func.func @ddr_async_descriptor() {
  %ddr = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %task = async.call @async_rdma(%ddr)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  async.await %task : !async.token
  return
}

// DDR-ASYNC-DESCRIPTOR: unsupported_ddr_planning_scope
// DDR-ASYNC-DESCRIPTOR-SAME: call-aware descriptor and bandwidth summary

//--- ddr-async-global.mlir
memref.global "private" @hidden_ddr
    : memref<128xf16, #wafer.memory<ddr, tensor>> = dense<1.0>

async.func @read_hidden_ddr() -> !async.token {
  %c0 = arith.constant 0 : index
  %hidden = memref.get_global @hidden_ddr
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %unused = memref.load %hidden[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @ddr_async_global() {
  %task = async.call @read_hidden_ddr() : () -> !async.token
  async.await %task : !async.token
  return
}

// DDR-ASYNC-GLOBAL: unsupported_lifetime_alias

//--- spm-yield.mlir
func.func @spm_tensor_cannot_escape(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %escaped = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>) -> (tensor<128xf16>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %tensor = bufferization.to_tensor %spm restrict writable
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tensor : tensor<128xf16>
  }
  %c0 = arith.constant 0 : index
  %unused = tensor.extract %escaped[%c0] : tensor<128xf16>
  return
}

// SPM-YIELD: shaped data result at index 0 must be a Wafer DDR memref, got 'tensor<128xf16>'

//--- spm-async-region.mlir
async.func @async_owned_region(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

func.func @launch_async_region(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %task = async.call @async_owned_region(%arg0)
        : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
    async.await %task : !async.token
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-ASYNC-REGION: unsupported_spm_planning_scope

//--- spm-parallel-region.mlir
func.func @parallel_region(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
    %result = wafer.tile.region(%boundary
        : memref<128xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
      %spm = memref.alloc()
          : memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield %arg0
          : memref<128xf16, #wafer.memory<ddr, tensor>>
    }
    scf.reduce
  }
  return
}

// SPM-PARALLEL: unsupported_spm_planning_scope

//--- spm-call-region.mlir
func.func @callee_region(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

func.func @caller_region(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    func.call @callee_region(%arg0)
        : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> ()
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-CALL: unsupported_spm_planning_scope
// SPM-CALL-SAME: dynamically execute another wafer.tile.region

//--- spm-external-call.mlir
func.func private @unknown_spm_effect()

func.func @external_call_from_spm_scope(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    func.call @unknown_spm_effect() : () -> ()
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-EXTERNAL-CALL: unsupported_spm_planning_scope
// SPM-EXTERNAL-CALL-SAME: dynamically execute another wafer.tile.region

//--- spm-external-async-call.mlir
async.func private @unknown_async_spm_effect() -> !async.token

func.func @external_async_call_from_spm_scope(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %task = async.call @unknown_async_spm_effect() : () -> !async.token
    async.await %task : !async.token
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-EXTERNAL-ASYNC: unsupported_spm_planning_scope
// SPM-EXTERNAL-ASYNC-SAME: external async.call

//--- spm-indirect-region.mlir
func.func @indirect_spm_callee(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

func.func @indirect_spm_caller(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %callee = func.constant @indirect_spm_callee
        : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> ()
    func.call_indirect %callee(%arg0)
        : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> ()
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-INDIRECT: unsupported_spm_planning_scope
// SPM-INDIRECT-SAME: indirect call from an active or asynchronous SPM scope

//--- spm-outside-region.mlir
async.func @touch_spm_outside(
    %spm: memref<128xf16, #wafer.memory<spm, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %spm[%c0]
      : memref<128xf16, #wafer.memory<spm, tensor>>
  return
}

func.func @spm_outside_region(
    %spm: memref<128xf16, #wafer.memory<spm, tensor>>) {
  %task = async.call @touch_spm_outside(%spm)
      : (memref<128xf16, #wafer.memory<spm, tensor>>) -> !async.token
  return
}

// SPM-OUTSIDE: unsupported_spm_planning_scope
// SPM-OUTSIDE-SAME: func.func boundaries cannot own SPM values
