// RUN: wafer-opt --wafer-plan-ddr-memory -split-input-file -verify-diagnostics %s

// -----

func.func @descriptor_payload_mismatch(
    %input: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536, 12, 256, 256, 257>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{descriptor_payload_mismatch}}
    wafer.instr.rdma %in to %loaded
        {byte_count = 16 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 6, 0, 0>}
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %out : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @descriptor_exceeds_root_range(
    %input: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536, 12, 256, 256, 257>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{ddr_range_overflow}}
    wafer.instr.rdma %in to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 64, 0, 0>}
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %out : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @dynamic_subview_is_not_accepted(
    %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %row: index) {
  %input_tile = memref.subview %input[%row, 2] [2, 3] [1, 1]
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
  %region = wafer.tile.region(%input_tile, %output
      : memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>,
        memref<4x8xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>,
       %out: memref<4x8xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536, 12, 256, 256, 257>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{unsupported_ddr_view}}
    wafer.instr.rdma %in to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 16, 0, 0>}
        : memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %out : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @compiler_managed_ddr_requires_range_requirement() {
  %ddr = memref.alloc() : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  %region = wafer.tile.region(%ddr
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%out: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %input = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536, 12, 256, 256, 257>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{unsupported_compiler_managed_ddr}}
    wafer.instr.wdma %input to %out
        {byte_count = 12 : i64, dst_iterations = array<i64: 2, 1, 1>,
         dst_strides = array<i64: 6, 0, 0>, inner_bytes = 6 : i64}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
       to memref<2x3xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}
