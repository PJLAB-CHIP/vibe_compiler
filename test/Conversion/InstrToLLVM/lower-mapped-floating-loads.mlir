// RUN: wafer-opt %s --wafer-lower-instr-to-target-llvm --verify-each | FileCheck %s

// CHECK-LABEL: llvm.func @copy_1024
// CHECK: llvm.call @get_ddr_memory_mapping_with_size
// CHECK: llvm.load volatile {{.*}} : !llvm.ptr -> i32
// CHECK: llvm.bitcast {{.*}} : i32 to f32
// CHECK: llvm.bitcast {{.*}} : f32 to i32
// CHECK: llvm.store volatile {{.*}} : i32, !llvm.ptr
// CHECK-NOT: llvm.call @get_ddr_memory_mapping_with_size
// CHECK: llvm.return
func.func @copy_1024(%input: memref<2x1024x1xf32, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %upper = arith.constant 1024 : index
  %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<2x1024x1xf32, #wafer.memory<spm, tensor>>
  scf.for %i = %zero to %upper step %one {
    %value = memref.load %input[%zero, %i, %zero] : memref<2x1024x1xf32, #wafer.memory<ddr, tensor>>
    memref.store %value, %out[%zero, %i, %zero] : memref<2x1024x1xf32, #wafer.memory<spm, tensor>>
  }
  return
}

// CHECK-LABEL: llvm.func @copy_1025
// CHECK: llvm.call @get_ddr_memory_mapping_with_size
// CHECK: llvm.load volatile {{.*}} : !llvm.ptr -> i32
// CHECK: llvm.bitcast {{.*}} : i32 to f32
// CHECK: llvm.bitcast {{.*}} : f32 to i32
// CHECK: llvm.store volatile {{.*}} : i32, !llvm.ptr
// CHECK-NOT: llvm.call @get_ddr_memory_mapping_with_size
// CHECK: llvm.return
func.func @copy_1025(%input: memref<2x1025x1xf32, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %upper = arith.constant 1025 : index
  %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<2x1025x1xf32, #wafer.memory<spm, tensor>>
  scf.for %i = %zero to %upper step %one {
    %value = memref.load %input[%zero, %i, %zero] : memref<2x1025x1xf32, #wafer.memory<ddr, tensor>>
    memref.store %value, %out[%zero, %i, %zero] : memref<2x1025x1xf32, #wafer.memory<spm, tensor>>
  }
  return
}

// CHECK-LABEL: llvm.func @copy_1031
// CHECK: llvm.call @get_ddr_memory_mapping_with_size
// CHECK: llvm.load volatile {{.*}} : !llvm.ptr -> i32
// CHECK: llvm.bitcast {{.*}} : i32 to f32
// CHECK: llvm.bitcast {{.*}} : f32 to i32
// CHECK: llvm.store volatile {{.*}} : i32, !llvm.ptr
// CHECK-NOT: llvm.call @get_ddr_memory_mapping_with_size
// CHECK: llvm.return
func.func @copy_1031(%input: memref<2x1031x1xf32, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>}) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %upper = arith.constant 1031 : index
  %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<2x1031x1xf32, #wafer.memory<spm, tensor>>
  scf.for %i = %zero to %upper step %one {
    %value = memref.load %input[%zero, %i, %zero] : memref<2x1031x1xf32, #wafer.memory<ddr, tensor>>
    memref.store %value, %out[%zero, %i, %zero] : memref<2x1031x1xf32, #wafer.memory<spm, tensor>>
  }
  return
}

// The rank-zero source is an actual scalar parameter; the destination
// exercises a full rank-three tensor rather than a scalar-only oracle.
// CHECK-LABEL: llvm.func @fill_1024
// CHECK: %[[RAW1024:.*]] = llvm.load volatile {{.*}} : !llvm.ptr -> i32
// CHECK: %[[VALUE1024:.*]] = llvm.bitcast %[[RAW1024]] : i32 to f32
// CHECK: llvm.call @wafer_tx81_memset({{.*}}, %[[RAW1024]],
// CHECK: llvm.call @wafer_tx81_wdma
func.func @fill_1024(%scalar: memref<f32, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>},
                    %output: memref<2x1024x1xf32, #wafer.memory<ddr, tensor>>) {
  %value = memref.load %scalar[] : memref<f32, #wafer.memory<ddr, tensor>>
  %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<2x1024x1xf32, #wafer.memory<spm, tensor>>
  wafer.instr.fill %out, %value : memref<2x1024x1xf32, #wafer.memory<spm, tensor>>, f32
  wafer.instr.ncc_join [0]
  wafer.instr.wdma %out to %output {byte_count = 8192 : i64, inner_bytes = 8192 : i64,
    dst_iterations = array<i64: 1, 1, 1>, dst_strides = array<i64: 0, 0, 0>}
    : memref<2x1024x1xf32, #wafer.memory<spm, tensor>> to memref<2x1024x1xf32, #wafer.memory<ddr, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// The rank-zero source is an actual scalar parameter; the destination
// exercises a full rank-three tensor rather than a scalar-only oracle.
// CHECK-LABEL: llvm.func @fill_1025
// CHECK: %[[RAW1025:.*]] = llvm.load volatile {{.*}} : !llvm.ptr -> i32
// CHECK: %[[VALUE1025:.*]] = llvm.bitcast %[[RAW1025]] : i32 to f32
// CHECK: llvm.call @wafer_tx81_memset({{.*}}, %[[RAW1025]],
// CHECK: llvm.call @wafer_tx81_wdma
func.func @fill_1025(%scalar: memref<f32, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>},
                    %output: memref<2x1025x1xf32, #wafer.memory<ddr, tensor>>) {
  %value = memref.load %scalar[] : memref<f32, #wafer.memory<ddr, tensor>>
  %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<2x1025x1xf32, #wafer.memory<spm, tensor>>
  wafer.instr.fill %out, %value : memref<2x1025x1xf32, #wafer.memory<spm, tensor>>, f32
  wafer.instr.ncc_join [0]
  wafer.instr.wdma %out to %output {byte_count = 8200 : i64, inner_bytes = 8200 : i64,
    dst_iterations = array<i64: 1, 1, 1>, dst_strides = array<i64: 0, 0, 0>}
    : memref<2x1025x1xf32, #wafer.memory<spm, tensor>> to memref<2x1025x1xf32, #wafer.memory<ddr, tensor>>
  wafer.instr.ncc_join [0]
  return
}

// The rank-zero source is an actual scalar parameter; the destination
// exercises a full rank-three tensor rather than a scalar-only oracle.
// CHECK-LABEL: llvm.func @fill_1031
// CHECK: %[[RAW1031:.*]] = llvm.load volatile {{.*}} : !llvm.ptr -> i32
// CHECK: %[[VALUE1031:.*]] = llvm.bitcast %[[RAW1031]] : i32 to f32
// CHECK: llvm.call @wafer_tx81_memset({{.*}}, %[[RAW1031]],
// CHECK: llvm.call @wafer_tx81_wdma
func.func @fill_1031(%scalar: memref<f32, #wafer.memory<ddr, tensor>> {wafer.program_argument = #wafer.program_argument<0>},
                    %output: memref<2x1031x1xf32, #wafer.memory<ddr, tensor>>) {
  %value = memref.load %scalar[] : memref<f32, #wafer.memory<ddr, tensor>>
  %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<2x1031x1xf32, #wafer.memory<spm, tensor>>
  wafer.instr.fill %out, %value : memref<2x1031x1xf32, #wafer.memory<spm, tensor>>, f32
  wafer.instr.ncc_join [0]
  wafer.instr.wdma %out to %output {byte_count = 8248 : i64, inner_bytes = 8248 : i64,
    dst_iterations = array<i64: 1, 1, 1>, dst_strides = array<i64: 0, 0, 0>}
    : memref<2x1031x1xf32, #wafer.memory<spm, tensor>> to memref<2x1031x1xf32, #wafer.memory<ddr, tensor>>
  wafer.instr.ncc_join [0]
  return
}
