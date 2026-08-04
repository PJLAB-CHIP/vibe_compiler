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
    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{source descriptor payload must equal byte_count}}
    wafer.instr.rdma %in to %loaded
        {byte_count = 16 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 6, 0, 0>}
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
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
    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{source descriptor byte range exceeds physical byte size}}
    wafer.instr.rdma %in to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 64, 0, 0>}
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
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
    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{unsupported_ddr_view}}
    wafer.instr.rdma %in to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 16, 0, 0>}
        : memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    wafer.tile.yield %out : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @loop_carried_backedge_view_exceeds_root() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %ddr = memref.alloc()
      : memref<8xf16, #wafer.memory<ddr, tensor>>
  %init = memref.subview %ddr[%c0] [1] [1]
      : memref<8xf16, #wafer.memory<ddr, tensor>>
     to memref<1xf16, strided<[1], offset: ?>,
               #wafer.memory<ddr, tensor>>
  %next = memref.subview %ddr[%c8] [1] [1]
      : memref<8xf16, #wafer.memory<ddr, tensor>>
     to memref<1xf16, strided<[1], offset: ?>,
               #wafer.memory<ddr, tensor>>
  %spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
  %result = scf.for %iv = %c0 to %c2 step %c1
      iter_args(%carried = %init)
      -> (memref<1xf16, strided<[1], offset: ?>,
                    #wafer.memory<ddr, tensor>>) {
    // expected-error @below {{source access end 18 exceeds DDR root byte size 16}}
    wafer.instr.rdma %carried to %spm
        {byte_count = 2 : i64, inner_bytes = 2 : i64,
         src_iterations = array<i64: 1, 1, 1>,
         src_strides = array<i64: 0, 0, 0>}
        : memref<1xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
       to memref<1xf16, #wafer.memory<spm, tensor>>
    scf.yield %next
        : memref<1xf16, strided<[1], offset: ?>,
                 #wafer.memory<ddr, tensor>>
  }
  wafer.instr.ncc_join [0]
  return
}
