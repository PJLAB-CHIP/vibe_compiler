// RUN: split-file %s %t
// RUN: not wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65792' %t/repeated-branch-spm.mlir 2>&1 | FileCheck --check-prefix=SPM-CAPACITY %s
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65920' %t/repeated-branch-spm.mlir | FileCheck --check-prefix=SPM %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/repeated-branch-ddr.mlir 2>&1 | FileCheck --check-prefix=DDR-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=384 ddr-largest-contiguous-bytes=256' %t/repeated-branch-ddr.mlir | FileCheck --check-prefix=DDR %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/token-select.mlir 2>&1 | FileCheck --check-prefix=TOKEN-SELECT-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=384 ddr-largest-contiguous-bytes=256' %t/token-select.mlir | FileCheck --check-prefix=TOKEN-SELECT %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/token-for.mlir 2>&1 | FileCheck --check-prefix=TOKEN-FOR-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=384 ddr-largest-contiguous-bytes=256' %t/token-for.mlir | FileCheck --check-prefix=TOKEN-FOR %s
// RUN: not wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65792' %t/captured-root-spm.mlir 2>&1 | FileCheck --check-prefix=CAPTURED-SPM-CAPACITY %s
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65920' %t/captured-root-spm.mlir | FileCheck --check-prefix=CAPTURED-SPM %s
// RUN: not wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=256 ddr-largest-contiguous-bytes=256' %t/captured-root-ddr.mlir 2>&1 | FileCheck --check-prefix=CAPTURED-DDR-CAPACITY %s
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-capacity-bytes=384 ddr-largest-contiguous-bytes=256' %t/captured-root-ddr.mlir | FileCheck --check-prefix=CAPTURED-DDR %s
// RUN: not wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=65792' %t/nested-tile-region.mlir 2>&1 | FileCheck --check-prefix=NESTED-TILE %s

// A syntactic scf.if inside scf.for is evaluated independently on every
// dynamic iteration. Opposite branches therefore cannot be treated as
// globally exclusive when an outer allocation remains live across iterations.

//--- repeated-branch-spm.mlir
func.func @repeated_branch_spm(
    %output_a: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %output_b: memref<64xf16, #wafer.memory<ddr, tensor>>) {
  %region_a, %region_b = wafer.tile.region(%output_a, %output_b
      : memref<128xf16, #wafer.memory<ddr, tensor>>,
        memref<64xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>,
          memref<64xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out_a: memref<128xf16, #wafer.memory<ddr, tensor>>,
       %out_b: memref<64xf16, #wafer.memory<ddr, tensor>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %true = arith.constant true
    %one = arith.constant 1.0 : f16
    %zero = arith.constant 0.0 : f16
    %a = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    %b = memref.alloc() : memref<64xf16, #wafer.memory<spm, tensor>>
    %unused = scf.for %i = %c0 to %c3 step %c1
        iter_args(%flag = %true) -> (i1) {
      scf.if %flag {
        %first = arith.cmpi eq, %i, %c0 : index
        scf.if %first {
          wafer.instr.fill %a, %one
              : memref<128xf16, #wafer.memory<spm, tensor>>, f16
          wafer.instr.local_fence
        }
        wafer.instr.wdma %a to %out_a
            {byte_count = 256 : i64, inner_bytes = 256 : i64,
             dst_iterations = array<i64: 1, 1, 1>,
             dst_strides = array<i64: 0, 0, 0>}
            : memref<128xf16, #wafer.memory<spm, tensor>>
           to memref<128xf16, #wafer.memory<ddr, tensor>>
        wafer.instr.local_fence
      } else {
        wafer.instr.fill %b, %zero
            : memref<64xf16, #wafer.memory<spm, tensor>>, f16
        wafer.instr.local_fence
        wafer.instr.wdma %b to %out_b
            {byte_count = 128 : i64, inner_bytes = 128 : i64,
             dst_iterations = array<i64: 1, 1, 1>,
             dst_strides = array<i64: 0, 0, 0>}
            : memref<64xf16, #wafer.memory<spm, tensor>>
           to memref<64xf16, #wafer.memory<ddr, tensor>>
        wafer.instr.local_fence
      }
      %next = arith.xori %flag, %true : i1
      scf.yield %next : i1
    }
    wafer.tile.yield %out_a, %out_b
        : memref<128xf16, #wafer.memory<ddr, tensor>>,
          memref<64xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-CAPACITY: capacity_overflow
// SPM-LABEL: func.func @repeated_branch_spm
// SPM: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65536>
// SPM: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65792>

//--- repeated-branch-ddr.mlir
func.func @repeated_branch_ddr() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %true = arith.constant true
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %b = memref.alloc() : memref<64xf16, #wafer.memory<ddr, tensor>>
  %spm_a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  %spm_b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<64xf16, #wafer.memory<spm, tensor>>
  %unused = scf.for %i = %c0 to %c3 step %c1
      iter_args(%flag = %true) -> (i1) {
    scf.if %flag {
      %first = arith.cmpi eq, %i, %c0 : index
      scf.if %first {
        wafer.instr.wdma %spm_a to %a
            {byte_count = 256 : i64, inner_bytes = 256 : i64,
             dst_iterations = array<i64: 1, 1, 1>,
             dst_strides = array<i64: 0, 0, 0>}
            : memref<128xf16, #wafer.memory<spm, tensor>>
           to memref<128xf16, #wafer.memory<ddr, tensor>>
        wafer.instr.local_fence
      }
      wafer.instr.rdma %a to %spm_a
          {byte_count = 256 : i64, inner_bytes = 256 : i64,
           src_iterations = array<i64: 1, 1, 1>,
           src_strides = array<i64: 0, 0, 0>}
          : memref<128xf16, #wafer.memory<ddr, tensor>>
         to memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.instr.local_fence
    } else {
      wafer.instr.wdma %spm_b to %b
          {byte_count = 128 : i64, inner_bytes = 128 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<64xf16, #wafer.memory<spm, tensor>>
         to memref<64xf16, #wafer.memory<ddr, tensor>>
      wafer.instr.local_fence
    }
    %next = arith.xori %flag, %true : i1
    scf.yield %next : i1
  }
  return
}

// DDR-CAPACITY: memory_capacity_overflow
// DDR-LABEL: func.func @repeated_branch_ddr
// DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>

//--- token-select.mlir
async.func @touch_select(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @token_select_keeps_root_live(%condition: i1) {
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %token = async.call @touch_select(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %selected = arith.select %condition, %token, %token : !async.token
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
  async.await %selected : !async.token
  return
}

// TOKEN-SELECT-CAPACITY: memory_capacity_overflow
// TOKEN-SELECT-LABEL: func.func @token_select_keeps_root_live
// TOKEN-SELECT: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// TOKEN-SELECT: arith.select
// TOKEN-SELECT: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// TOKEN-SELECT: async.await

//--- token-for.mlir
async.func @touch_for(
    %buffer: memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token {
  %c0 = arith.constant 0 : index
  %unused = memref.load %buffer[%c0]
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  return
}

func.func @token_for_keeps_root_live() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a = memref.alloc() : memref<128xf16, #wafer.memory<ddr, tensor>>
  %token = async.call @touch_for(%a)
      : (memref<128xf16, #wafer.memory<ddr, tensor>>) -> !async.token
  %looped = scf.for %i = %c0 to %c1 step %c1
      iter_args(%iter = %token) -> (!async.token) {
    scf.yield %iter : !async.token
  }
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
  async.await %looped : !async.token
  return
}

// TOKEN-FOR-CAPACITY: memory_capacity_overflow
// TOKEN-FOR-LABEL: func.func @token_for_keeps_root_live
// TOKEN-FOR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// TOKEN-FOR: scf.for
// TOKEN-FOR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>
// TOKEN-FOR: async.await

// A root captured from outside a loop may be read again in a later dynamic
// iteration. A body-local allocation after the static use cannot overwrite it.

//--- captured-root-spm.mlir
func.func @captured_spm_root_survives_backedge(
    %output: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%output
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %one = arith.constant 1.0 : f16
    %zero = arith.constant 0.0 : f16
    %captured = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %captured, %one
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.local_fence
    scf.for %i = %c0 to %c3 step %c1 {
      wafer.instr.wdma %captured to %out
          {byte_count = 256 : i64, inner_bytes = 256 : i64,
           dst_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>}
          : memref<128xf16, #wafer.memory<spm, tensor>>
         to memref<128xf16, #wafer.memory<ddr, tensor>>
      wafer.instr.local_fence
      %body = memref.alloc()
          : memref<64xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %body, %zero
          : memref<64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.local_fence
    }
    wafer.tile.yield %out
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CAPTURED-SPM-CAPACITY: capacity_overflow
// CAPTURED-SPM-LABEL: func.func @captured_spm_root_survives_backedge
// CAPTURED-SPM: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65536>
// CAPTURED-SPM: scf.for
// CAPTURED-SPM: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65792>

//--- captured-root-ddr.mlir
func.func @captured_ddr_root_survives_backedge() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %captured = memref.alloc()
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %spm_captured = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  %spm_body = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<64xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %spm_captured to %captured
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.local_fence
  scf.for %i = %c0 to %c3 step %c1 {
    wafer.instr.rdma %captured to %spm_captured
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<128xf16, #wafer.memory<ddr, tensor>>
       to memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
    %body = memref.alloc()
        : memref<64xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.wdma %spm_body to %body
        {byte_count = 128 : i64, inner_bytes = 128 : i64,
         dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>}
        : memref<64xf16, #wafer.memory<spm, tensor>>
       to memref<64xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
  }
  return
}

// CAPTURED-DDR-CAPACITY: memory_capacity_overflow
// CAPTURED-DDR-LABEL: func.func @captured_ddr_root_survives_backedge
// CAPTURED-DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// CAPTURED-DDR: scf.for
// CAPTURED-DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<256>

//--- nested-tile-region.mlir
func.func @nested_tile_region_clobbers_outer(
    %output: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %outer = wafer.tile.region(%output
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%outer_output: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.0 : f16
    %a = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %a, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.local_fence
    %inner = wafer.tile.region(%outer_output
        : memref<128xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%inner_output: memref<128xf16, #wafer.memory<ddr, tensor>>):
      %one = arith.constant 1.0 : f16
      %b = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %b, %one
          : memref<128xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.local_fence
      wafer.tile.yield %inner_output
          : memref<128xf16, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.wdma %a to %inner
        {byte_count = 256 : i64, inner_bytes = 256 : i64,
         dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
    wafer.instr.local_fence
    wafer.tile.yield %inner
        : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// Do not couple this regression to the exact owner diagnostic wording.
// NESTED-TILE: error:
