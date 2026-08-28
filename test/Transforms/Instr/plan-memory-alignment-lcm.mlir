// RUN: split-file %s %t
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=67584 spm-alignment=384' %t/spm.mlir | FileCheck %s --check-prefix=SPM
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-alignment-bytes=384 ddr-capacity-bytes=2048 ddr-largest-contiguous-bytes=2048' %t/ddr.mlir | FileCheck %s --check-prefix=DDR
// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65664 spm-limit=66560 spm-alignment=16' %t/spm-encoding.mlir | FileCheck %s --check-prefix=SPM-ENCODING
// RUN: wafer-opt --wafer-plan-ddr-memory='ddr-alignment-bytes=16 ddr-capacity-bytes=2048 ddr-largest-contiguous-bytes=2048' %t/ddr-encoding.mlir | FileCheck %s --check-prefix=DDR-ENCODING

//--- spm.mlir
func.func @spm_non_dividing_alignment(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %aligned = memref.alloc() {alignment = 256 : i64}
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-LABEL: func.func @spm_non_dividing_alignment
// SPM: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<66048>

//--- ddr.mlir
func.func @ddr_non_dividing_alignment() {
  %large = memref.alloc()
      : memref<512xf16, #wafer.memory<ddr, tensor>>
  %aligned = memref.alloc() {alignment = 256 : i64}
      : memref<128xf16, #wafer.memory<ddr, tensor>>
  %large_spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<512xf16, #wafer.memory<spm, tensor>>
  %aligned_spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<66560>}
      : memref<128xf16, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %large_spm to %large
      {byte_count = 1024 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 1024 : i64}
      : memref<512xf16, #wafer.memory<spm, tensor>>
     to memref<512xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.wdma %aligned_spm to %aligned
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<128xf16, #wafer.memory<spm, tensor>>
     to memref<128xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// DDR-LABEL: func.func @ddr_non_dividing_alignment
// DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// DDR: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<1536>

//--- spm-encoding.mlir
func.func @spm_encoding_alignment(
    %boundary: memref<1x65xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<1x65xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<1x65xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<1x65xf16, #wafer.memory<ddr, tensor>>):
    %blocked = memref.alloc()
        : memref<1x65xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0 : memref<1x65xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// SPM-ENCODING-LABEL: func.func @spm_encoding_alignment
// SPM-ENCODING: memref.alloc() {{.*}}wafer.spm.offset = #wafer.spm_offset<65792>

//--- ddr-encoding.mlir
func.func @ddr_encoding_alignment() {
  %compact = memref.alloc() {alignment = 384 : i64}
      : memref<500xf16, #wafer.memory<ddr, tensor>>
  %blocked = memref.alloc()
      : memref<1x65xf16, #wafer.memory<ddr, cx>>
  %compact_spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<500xf16, #wafer.memory<spm, tensor>>
  %blocked_spm = memref.alloc()
      {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<1x65xf16, #wafer.memory<spm, cx>>
  wafer.instr.wdma %compact_spm to %compact
      {byte_count = 1000 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 1000 : i64}
      : memref<500xf16, #wafer.memory<spm, tensor>>
     to memref<500xf16, #wafer.memory<ddr, tensor>>
  wafer.instr.wdma %blocked_spm to %blocked
      {byte_count = 256 : i64, dst_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>, inner_bytes = 256 : i64}
      : memref<1x65xf16, #wafer.memory<spm, cx>>
     to memref<1x65xf16, #wafer.memory<ddr, cx>>
  wafer.instr.ncc_join [0]
  return
}

// DDR-ENCODING-LABEL: func.func @ddr_encoding_alignment
// DDR-ENCODING: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<0>
// DDR-ENCODING: memref.alloc() {{.*}}wafer.ddr.offset = #wafer.ddr_offset<1024>
