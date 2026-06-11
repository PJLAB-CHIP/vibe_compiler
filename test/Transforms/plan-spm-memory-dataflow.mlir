// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66048' %s | FileCheck --check-prefix=IF %s
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66048' %s | FileCheck --check-prefix=LOOP-TEMP %s
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66048' %s | FileCheck --check-prefix=LOOP-CARRY %s
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66048' %s | FileCheck --check-prefix=ASYNC %s

func.func @if_branch_results_reuse(%output: memref<128xf16, #wafer.memory<ddr, tensor>>,
                                   %cond: i1) {
  %region = wafer.tile.region(%output, %cond
      : memref<128xf16, #wafer.memory<ddr, tensor>>, i1)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out: memref<128xf16, #wafer.memory<ddr, tensor>>, %c: i1):
    %zero = arith.constant 0.000000e+00 : f16
    %selected = scf.if %c -> (memref<128xf16, #wafer.memory<spm, tensor>>) {
      %then_buf = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %then_buf, %zero
          : memref<128xf16, #wafer.memory<spm, tensor>>, f16
      scf.yield %then_buf : memref<128xf16, #wafer.memory<spm, tensor>>
    } else {
      %else_buf = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %else_buf, %zero
          : memref<128xf16, #wafer.memory<spm, tensor>>, f16
      scf.yield %else_buf : memref<128xf16, #wafer.memory<spm, tensor>>
    }
    wafer.instr.wdma %selected to %out
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

func.func @loop_body_temp_reuses_after_loop(%boundary: memref<128xf16, #wafer.memory<ddr, tensor>>,
                                            %lb: index, %ub: index, %step: index) {
  %region = wafer.tile.region(%boundary, %lb, %ub, %step
      : memref<128xf16, #wafer.memory<ddr, tensor>>, index, index, index)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %l: index, %u: index, %s: index):
    %zero = arith.constant 0.000000e+00 : f16
    scf.for %i = %l to %u step %s {
      %loop_tmp = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %loop_tmp, %zero
          : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    }
    %after = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %after, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

func.func @loop_carried_result_conflicts_with_body_use(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lb: index, %ub: index, %step: index) {
  %region = wafer.tile.region(%boundary, %lb, %ub, %step
      : memref<128xf16, #wafer.memory<ddr, tensor>>, index, index, index)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>, %l: index, %u: index, %s: index):
    %zero = arith.constant 0.000000e+00 : f16
    %init = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %init, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    %looped = scf.for %i = %l to %u step %s iter_args(%iter = %init)
        -> (memref<128xf16, #wafer.memory<spm, tensor>>) {
      %next = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise #wafer.elementwise_kind<add> %iter, %iter into %next
          {indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ]}
          : memref<128xf16, #wafer.memory<spm, tensor>>,
            memref<128xf16, #wafer.memory<spm, tensor>>
        into memref<128xf16, #wafer.memory<spm, tensor>>
      scf.yield %next : memref<128xf16, #wafer.memory<spm, tensor>>
    }
    wafer.instr.wdma %looped to %arg0
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

func.func @async_token_extends_source_until_wait(%boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %source = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %source, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    %token = wafer.tile.send %source {peer = 1 : i64, bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>> -> !async.token
    %before_wait = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %before_wait, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.tile.wait %token : !async.token
    %after_wait = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %after_wait, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// IF-LABEL: func.func @if_branch_results_reuse
// IF: %[[THEN:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// IF: scf.yield %[[THEN]]
// IF-NOT: func.func @loop_body_temp_reuses_after_loop
// IF: %[[ELSE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// IF: scf.yield %[[ELSE]]
// IF-NOT: func.func @loop_body_temp_reuses_after_loop
// IF: wafer.instr.wdma

// LOOP-TEMP-LABEL: func.func @loop_body_temp_reuses_after_loop
// LOOP-TEMP: %[[TMP:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// LOOP-TEMP: wafer.instr.fill %[[TMP]]
// LOOP-TEMP-NOT: func.func @loop_carried_result_conflicts_with_body_use
// LOOP-TEMP: %[[AFTER:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// LOOP-TEMP: wafer.instr.fill %[[AFTER]]

// LOOP-CARRY-LABEL: func.func @loop_carried_result_conflicts_with_body_use
// LOOP-CARRY: %[[INIT:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// LOOP-CARRY: %[[NEXT:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<128xf16, #wafer.memory<spm, tensor>>
// LOOP-CARRY: scf.yield %[[NEXT]]

// ASYNC-LABEL: func.func @async_token_extends_source_until_wait
// ASYNC: %[[SOURCE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
// ASYNC: wafer.tile.send %[[SOURCE]]
// ASYNC: %[[BEFORE:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>} : memref<128xf16, #wafer.memory<spm, tensor>>
// ASYNC: wafer.tile.wait
// ASYNC: %[[AFTER:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
