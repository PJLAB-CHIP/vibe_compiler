// RUN: split-file %s %t
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/unawaited-token.mlir 2>&1 | FileCheck --check-prefix=UNAWAITED %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/unawaited-external.mlir 2>&1 | FileCheck --check-prefix=EXTERNAL-UNAWAITED %s
// RUN: wafer-opt --wafer-plan-ddr-memory %t/awaited-external.mlir | FileCheck --check-prefix=EXTERNAL-AWAITED %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/distinct-select.mlir 2>&1 | FileCheck --check-prefix=SELECT %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/preissued-if.mlir 2>&1 | FileCheck --check-prefix=IF %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/distinct-for.mlir 2>&1 | FileCheck --check-prefix=FOR %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/group-alias.mlir 2>&1 | FileCheck --check-prefix=GROUP-ALIAS %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/group-loop-task.mlir 2>&1 | FileCheck --check-prefix=GROUP-LOOP %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/nested-callee-task.mlir 2>&1 | FileCheck --check-prefix=CALLEE-TASK %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/callee-local-issue.mlir 2>&1 | FileCheck --check-prefix=CALLEE-LOCAL %s
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/erased-space-async.mlir 2>&1 | FileCheck --check-prefix=ERASED-ASYNC %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/async-value.mlir 2>&1 | FileCheck --check-prefix=VALUE-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=384 ddr-largest-contiguous-bytes=256' %t/async-value.mlir | FileCheck --check-prefix=VALUE %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/direct-group.mlir 2>&1 | FileCheck --check-prefix=GROUP-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=384 ddr-largest-contiguous-bytes=256' %t/direct-group.mlir | FileCheck --check-prefix=GROUP %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/branch-local-if.mlir | FileCheck --check-prefix=BRANCH-LOCAL %s

// Generic async handle roots and task completion identities are distinct
// facts. A use of a handle extends its associated roots, but only a supported
// terminal wait may discharge the exact task identities represented on that
// structured execution path.

//--- unawaited-token.mlir
async.func @touch_unawaited(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @unawaited_token() {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %token = async.call @touch_unawaited(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  return
}

// UNAWAITED: missing_async_completion

//--- unawaited-external.mlir
async.func @touch_external_unawaited(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @unawaited_external(
    %external: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %token = async.call @touch_external_unawaited(%external)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  return
}

// EXTERNAL-UNAWAITED: missing_async_completion

//--- awaited-external.mlir
async.func @touch_external_awaited(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @awaited_external(
    %external: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %token = async.call @touch_external_awaited(%external)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  async.await %token : !async.token
  return
}

// EXTERNAL-AWAITED-LABEL: func.func @awaited_external
// EXTERNAL-AWAITED: async.await

//--- distinct-select.mlir
async.func @touch_select_a(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @distinct_select(%condition: i1) {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %b = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %a_token = async.call @touch_select_a(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %b_token = async.call @touch_select_a(%b)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %selected = arith.select %condition, %a_token, %b_token : !async.token
  async.await %selected : !async.token
  return
}

// SELECT: unsupported_async_completion_flow

//--- preissued-if.mlir
async.func @touch_if_a(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @preissued_if(%condition: i1) {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %b = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %a_token = async.call @touch_if_a(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %b_token = async.call @touch_if_a(%b)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %selected = scf.if %condition -> (!async.token) {
    scf.yield %a_token : !async.token
  } else {
    scf.yield %b_token : !async.token
  }
  async.await %selected : !async.token
  return
}

// IF: missing_async_completion

//--- distinct-for.mlir
async.func @touch_for_a(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @distinct_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %b = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %a_token = async.call @touch_for_a(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %b_token = async.call @touch_for_a(%b)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %looped = scf.for %i = %c0 to %c1 step %c1
      iter_args(%iter = %a_token) -> (!async.token) {
    scf.yield %b_token : !async.token
  }
  async.await %looped : !async.token
  return
}

// FOR: unsupported_async_completion_flow

//--- group-alias.mlir
async.func @touch_group_alias(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @group_alias(%condition: i1) {
  %c1 = arith.constant 1 : index
  %first = async.create_group %c1 : !async.group
  %second = async.create_group %c1 : !async.group
  %selected = arith.select %condition, %first, %second : !async.group
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %token = async.call @touch_group_alias(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %rank = async.add_to_group %token, %selected : !async.token
  async.await_all %first
  async.await_all %second
  return
}

// GROUP-ALIAS: unsupported_async_completion_flow

//--- group-loop-task.mlir
async.func @touch_group_loop(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @group_loop_task() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %group = async.create_group %c2 : !async.group
  scf.for %i = %c0 to %c2 step %c1 {
    %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
    %token = async.call @touch_group_loop(%a)
        : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
    %rank = async.add_to_group %token, %group : !async.token
  }
  async.await_all %group
  return
}

// GROUP-LOOP: unsupported_async_completion_flow

//--- nested-callee-task.mlir
async.func @nested_inner(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

async.func @nested_outer(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %nested = async.call @nested_inner(%buffer)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  return
}

func.func @nested_callee_task() {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %outer = async.call @nested_outer(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  async.await %outer : !async.token
  return
}

// CALLEE-TASK: missing_async_completion

//--- callee-local-issue.mlir
async.func @issue_without_fence(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %buffer to %spm
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  return
}

func.func @callee_local_issue() {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %token = async.call @issue_without_fence(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  async.await %token : !async.token
  return
}

// CALLEE-LOCAL: missing_local_completion

//--- erased-space-async.mlir
async.func @touch_erased_space(%buffer: memref<128xf16>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0] : memref<128xf16>
  return
}

func.func @erased_space_async() {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %erased = memref.memory_space_cast %a
      : memref<128xf16, #wafer.memory<ddr, tensor>> to memref<128xf16>
  %token = async.call @touch_erased_space(%erased)
      : (memref<128xf16>) -> !async.token
  async.await %token : !async.token
  return
}

// ERASED-ASYNC: unsupported_async_completion_flow

//--- async-value.mlir
async.func @touch_value(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>)
    -> !async.value<i32> {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %value = arith.constant 7 : i32
  return %value : i32
}

func.func @async_value_keeps_root_live() {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %value = async.call @touch_value(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.value<i32>
  %b = memref.alloc() : memref<64xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<64xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %b to %spm
      {byte_count = 128 : i64, inner_bytes = 128 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<64xf16, #wafer.memory<ddr, tensor>>
     to memref<64xf16, #wafer.memory<spm, tensor>>
  wafer.instr.local_fence
  %ready = async.await %value : !async.value<i32>
  return
}

// VALUE-CAPACITY: memory_capacity_overflow
// VALUE-LABEL: func.func @async_value_keeps_root_live
// VALUE: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// VALUE: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// VALUE: async.await

//--- direct-group.mlir
async.func @touch_direct_group(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @direct_group_keeps_root_live() {
  %c1 = arith.constant 1 : index
  %group = async.create_group %c1 : !async.group
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %token = async.call @touch_direct_group(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %rank = async.add_to_group %token, %group : !async.token
  %b = memref.alloc() : memref<64xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<64xf16, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %b to %spm
      {byte_count = 128 : i64, inner_bytes = 128 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<64xf16, #wafer.memory<ddr, tensor>>
     to memref<64xf16, #wafer.memory<spm, tensor>>
  wafer.instr.local_fence
  async.await_all %group
  return
}

// GROUP-CAPACITY: memory_capacity_overflow
// GROUP-LABEL: func.func @direct_group_keeps_root_live
// GROUP: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// GROUP: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// GROUP: async.await_all

//--- branch-local-if.mlir
async.func @touch_branch_local(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @branch_local_if(%condition: i1) {
  %selected = scf.if %condition -> (!async.token) {
    %then_buffer = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    %then_token = async.call @touch_branch_local(%then_buffer)
        : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
    scf.yield %then_token : !async.token
  } else {
    %else_buffer = memref.alloc()
        : memref<128xf16, #wafer.memory<ddr, tensor>>
    %else_token = async.call @touch_branch_local(%else_buffer)
        : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
    scf.yield %else_token : !async.token
  }
  async.await %selected : !async.token
  %after = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

// BRANCH-LOCAL-LABEL: func.func @branch_local_if
// BRANCH-LOCAL-COUNT-2: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// BRANCH-LOCAL: async.await
// BRANCH-LOCAL: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
