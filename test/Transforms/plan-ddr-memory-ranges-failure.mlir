// RUN: wafer-opt --wafer-plan-ddr-memory -split-input-file -verify-diagnostics %s

func.func @invalid_ddr_requirement_alignment() {
  // expected-error @below {{ddr_alignment_failure}}
  %ddr = memref.alloc()
      {wafer.ddr.requirement = #wafer.ddr_requirement<0, write>}
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  return
}

// -----

func.func @write_access_requires_write_requirement() {
  %ddr = memref.alloc()
      {wafer.ddr.requirement = #wafer.ddr_requirement<256, read>}
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536, 12, 256, 256, 257>}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{ddr_access_intent_mismatch}}
  wafer.instr.wdma %spm to %ddr
      {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
       dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
     to memref<2x3xf16, #wafer.memory<ddr, tensor>>
  return
}

// -----

func.func @read_access_requires_read_requirement() {
  %ddr = memref.alloc()
      {wafer.ddr.requirement = #wafer.ddr_requirement<256, write>}
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536, 12, 256, 256, 257>}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{ddr_access_intent_mismatch}}
  wafer.instr.rdma %ddr to %spm
      {byte_count = 12 : i64, inner_bytes = 6 : i64,
       src_iterations = array<i64: 2, 1, 1>,
       src_strides = array<i64: 6, 0, 0>}
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>
     to memref<2x3xf16, #wafer.memory<spm, tensor>>
  return
}

// -----

func.func @compiler_managed_ddr_without_requirement() {
  %ddr = memref.alloc() : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536, 12, 256, 256, 257>}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{missing_ddr_requirement}}
  wafer.instr.wdma %spm to %ddr
      {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
       dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
      : memref<2x3xf16, #wafer.memory<spm, tensor>>
     to memref<2x3xf16, #wafer.memory<ddr, tensor>>
  return
}
