// RUN: split-file %s %t
// RUN: not wafer-opt --wafer-plan-spm-memory %t/unawaited.mlir 2>&1 | FileCheck --check-prefix=UNAWAITED %s
// RUN: not wafer-opt --wafer-plan-spm-memory %t/distinct-select.mlir 2>&1 | FileCheck --check-prefix=SELECT %s
// RUN: not wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65792' %t/awaited-overlap.mlir 2>&1 | FileCheck --check-prefix=CAPACITY %s
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65920' %t/awaited-overlap.mlir | FileCheck --check-prefix=AWAITED %s

// Generic async task identity is shared with DDR, while the SPM owner still
// owns the tile-local arena and exact DTE policy.

//--- unawaited.mlir
async.func @touch_unawaited_spm(
    %buffer: memref<128xf16, #wafer.memory<spm, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<spm, tensor>>
  return
}

func.func @unawaited_spm(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %spm = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %task = async.call @touch_unawaited_spm(%spm)
        : (memref<128xf16, #wafer.memory<spm, tensor>>) -> !async.token
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// UNAWAITED: missing_async_completion

//--- distinct-select.mlir
async.func @touch_selected_spm(
    %buffer: memref<64xf16, #wafer.memory<spm, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<64xf16, #wafer.memory<spm, tensor>>
  return
}

func.func @distinct_spm_select(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>, %condition: i1) {
  %result = wafer.tile.region(%boundary, %condition
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1):
    %a = memref.alloc()
        : memref<64xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc()
        : memref<64xf16, #wafer.memory<spm, tensor>>
    %a_task = async.call @touch_selected_spm(%a)
        : (memref<64xf16, #wafer.memory<spm, tensor>>) -> !async.token
    %b_task = async.call @touch_selected_spm(%b)
        : (memref<64xf16, #wafer.memory<spm, tensor>>) -> !async.token
    %selected = arith.select %cond, %a_task, %b_task : !async.token
    async.await %selected : !async.token
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SELECT: unsupported_async_completion_flow

//--- awaited-overlap.mlir
async.func @touch_awaited_spm(
    %buffer: memref<128xf16, #wafer.memory<spm, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<spm, tensor>>
  return
}

func.func @awaited_spm_keeps_root_live(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %result = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f16
    %a = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %task = async.call @touch_awaited_spm(%a)
        : (memref<128xf16, #wafer.memory<spm, tensor>>) -> !async.token
    %b = memref.alloc()
        : memref<64xf16, #wafer.memory<spm, tensor>>
    memref.store %one, %b[%c0]
        : memref<64xf16, #wafer.memory<spm, tensor>>
    async.await %task : !async.token
    wafer.tile.yield %arg0
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CAPACITY: capacity_overflow
// AWAITED-LABEL: func.func @awaited_spm_keeps_root_live
// AWAITED: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65536>
// AWAITED: async.call
// AWAITED: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65792>
// AWAITED: async.await
