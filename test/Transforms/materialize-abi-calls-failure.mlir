// RUN: wafer-opt --wafer-materialize-abi-calls -split-input-file -verify-diagnostics %s

// -----

func.func @missing_spm_offset(
    %input: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc()
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{abi_materialization_failure: SPM memref has no accepted wafer.spm.offset}}
    wafer.instr.rdma %in to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 16, 0, 0>}
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %out : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @dynamic_ddr_offset(
    %input: memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>,
    %output: memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    // expected-error @below {{abi_materialization_failure: DDR view must have static non-negative byte offset}}
    wafer.instr.rdma %in to %loaded
        {byte_count = 12 : i64, inner_bytes = 6 : i64,
         src_iterations = array<i64: 2, 1, 1>,
         src_strides = array<i64: 16, 0, 0>}
        : memref<2x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %out : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// -----

func.func @non_constant_fill(
    %input: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0.000000e+00 : f16
    %value = arith.addf %c0, %c0 : f16
    // expected-error @below {{abi_materialization_failure: fill value must be an arith.constant scalar}}
    wafer.instr.fill %dest, %value
        : memref<2x3xf16, #wafer.memory<spm, tensor>>, f16
    wafer.tile.yield %out : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}
