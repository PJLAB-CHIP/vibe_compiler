// RUN: wafer-opt --wafer-plan-ddr-memory -split-input-file -verify-diagnostics %s

// -----

func.func @external_root_missing_local_fence(
    %input: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{missing_local_completion: DDR-touching local Movement issue has a reachable path to entry exit without wafer.instr.local_fence}}
  wafer.instr.rdma %input to %spm
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  return
}

// -----

func.func @only_one_branch_fences_prior_issue(
    %output: memref<128xf16, #wafer.memory<ddr, tensor>>, %cond: i1) {
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{missing_local_completion: DDR-touching local Movement issue has a reachable path to entry exit without wafer.instr.local_fence}}
  wafer.instr.wdma %spm to %output
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  scf.if %cond {
    wafer.instr.local_fence
  } else {
  }
  return
}

// -----

func.func @pre_loop_issue_with_body_only_fence(
    %input: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lb: index, %ub: index, %step: index) {
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{missing_local_completion: DDR-touching local Movement issue has a reachable path to entry exit without wafer.instr.local_fence}}
  wafer.instr.rdma %input to %spm
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_iterations = array<i64: 1, 1, 1>,
       src_strides = array<i64: 0, 0, 0>}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
     to memref<128xf16, #wafer.memory<spm, tensor>>
  scf.for %i = %lb to %ub step %step {
    wafer.instr.local_fence
  }
  return
}

// -----

func.func @loop_body_issue_without_body_fence(
    %output: memref<128xf16, #wafer.memory<ddr, tensor>>,
    %lb: index, %ub: index, %step: index) {
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  scf.for %i = %lb to %ub step %step {
    // expected-error @below {{missing_local_completion: DDR-touching local Movement issue reaches an scf.for backedge without a body-local wafer.instr.local_fence}}
    wafer.instr.wdma %spm to %output
        {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}
