// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | FileCheck %s --check-prefix=POS
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/derived-offset.mlir | FileCheck %s --check-prefix=DERIVED
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/unsupported-offset.mlir 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/dynamic-bound.mlir 2>&1 | FileCheck %s --check-prefix=BOUND
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/out-of-range.mlir 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/spm.mlir | FileCheck %s --check-prefix=SPM
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/cx.mlir 2>&1 | FileCheck %s --check-prefix=CX
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/ncx.mlir 2>&1 | FileCheck %s --check-prefix=NCX
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/dynamic-size.mlir 2>&1 | FileCheck %s --check-prefix=SIZE
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/dynamic-stride.mlir 2>&1 | FileCheck %s --check-prefix=STRIDE

//--- positive.mlir

func.func @bounded_dynamic_ddr_subview(
    %input: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %row = %c0 to %c4 step %c1 {
    %input_tile = memref.subview %input[%row, 2] [1, 3] [1, 1]
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
       to memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    %output_tile = memref.subview %output[%row, 2] [1, 3] [1, 1]
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
       to memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input_tile to %spm
        {byte_count = 6 : i64, inner_bytes = 6 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<1x3xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %spm to %output_tile
        {byte_count = 6 : i64, inner_bytes = 6 : i64,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<1x3xf16, #wafer.memory<spm, tensor>>
       to memref<1x3xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    wafer.instr.ncc_join [0]
  }
  return
}

// POS-LABEL: llvm.func @bounded_dynamic_ddr_subview(
// POS-SAME: %[[INPUT:.+]]: i64, %[[OUTPUT:.+]]: i64)
// POS: ^bb{{[0-9]+}}(%[[ROW:.+]]: i64):
// POS: %[[INPUT_STATIC:.+]] = llvm.mlir.constant(4 : i64) : i64
// POS: %[[INPUT_BASE:.+]] = llvm.add %[[INPUT]], %[[INPUT_STATIC]] : i64
// POS: %[[INPUT_STRIDE:.+]] = llvm.mlir.constant(16 : i64) : i64
// POS: %[[INPUT_DYNAMIC:.+]] = llvm.mul %[[ROW]], %[[INPUT_STRIDE]] : i64
// POS: %[[INPUT_ADDR:.+]] = llvm.add %[[INPUT_BASE]], %[[INPUT_DYNAMIC]] : i64
// POS: %[[OUTPUT_STATIC:.+]] = llvm.mlir.constant(4 : i64) : i64
// POS: %[[OUTPUT_BASE:.+]] = llvm.add %[[OUTPUT]], %[[OUTPUT_STATIC]] : i64
// POS: %[[OUTPUT_STRIDE:.+]] = llvm.mlir.constant(16 : i64) : i64
// POS: %[[OUTPUT_DYNAMIC:.+]] = llvm.mul %[[ROW]], %[[OUTPUT_STRIDE]] : i64
// POS: %[[OUTPUT_ADDR:.+]] = llvm.add %[[OUTPUT_BASE]], %[[OUTPUT_DYNAMIC]] : i64
// POS: llvm.call @wafer_tx81_rdma_v3(%[[INPUT_ADDR]],
// POS: llvm.call @wafer_tx81_wdma_v3({{.*}}%[[OUTPUT_ADDR]],
// POS-NOT: memref.subview
// POS-NOT: scf.for

// LLVMIR-LABEL: define void @bounded_dynamic_ddr_subview(
// LLVMIR: %[[INPUT_DYNAMIC:.+]] = mul i64 %{{.+}}, 16
// LLVMIR: %[[INPUT_ADDR:.+]] = add i64 %{{.+}}, %[[INPUT_DYNAMIC]]
// LLVMIR: %[[OUTPUT_DYNAMIC:.+]] = mul i64 %{{.+}}, 16
// LLVMIR: %[[OUTPUT_ADDR:.+]] = add i64 %{{.+}}, %[[OUTPUT_DYNAMIC]]
// LLVMIR: call void @wafer_tx81_rdma_v3(i64 %[[INPUT_ADDR]],
// LLVMIR: call void @wafer_tx81_wdma_v3({{.*}}i64 %[[OUTPUT_ADDR]],

//--- derived-offset.mlir

func.func @lower_stage_shifted_dynamic_offset(
    %input: memref<5xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %shifted = arith.addi %i, %c1 : index
    %view = memref.subview %input[%shifted] [1] [1]
        : memref<5xf16, #wafer.memory<ddr, tensor>>
       to memref<1xf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
    %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %view to %spm
        {byte_count = 2 : i64, inner_bytes = 2 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<1xf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
       to memref<1xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
  }
  return
}

// DERIVED-LABEL: llvm.func @lower_stage_shifted_dynamic_offset(
// DERIVED: llvm.add
// DERIVED: llvm.mul
// DERIVED: llvm.call @wafer_tx81_rdma_v3
// DERIVED-NOT: memref.subview

//--- unsupported-offset.mlir

func.func @reject_unsupported_dynamic_offset(
    %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
    %condition: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %selected = arith.select %condition, %i, %c0 : index
    %view = memref.subview %input[%selected] [1] [1]
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<1xf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
  }
  return
}

// UNKNOWN: unsupported_target_address: dynamic tensor subview offset {{.*}} must be a supported statically bounded index expression

//--- dynamic-bound.mlir

func.func @reject_dynamic_loop_bound(
    %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
    %bounds: memref<1xindex, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %upper = memref.load %bounds[%c0]
      : memref<1xindex, #wafer.memory<ddr, tensor>>
  scf.for %i = %c0 to %upper step %c1 {
    %view = memref.subview %input[%i] [1] [1]
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<1xf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
  }
  return
}

// BOUND: unsupported_target_address: dynamic tensor subview offset #0 requires constant non-negative scf.for bounds and a positive constant step

//--- out-of-range.mlir

func.func @reject_out_of_range_loop(
    %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %view = memref.subview %input[%i] [2] [1]
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<2xf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
  }
  return
}

// RANGE: target_geometry_mismatch: dynamic tensor subview dimension #0 may access source coordinate 4 outside static extent 4

//--- spm.mlir

func.func @bounded_dynamic_spm_subview() {
  %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4xf16, #wafer.memory<spm, tensor>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %view = memref.subview %spm[%i] [1] [1]
        : memref<4xf16, #wafer.memory<spm, tensor>>
       to memref<1xf16, strided<[1], offset: ?>, #wafer.memory<spm, tensor>>
  }
  return
}

// SPM-LABEL: llvm.func @bounded_dynamic_spm_subview
// SPM: llvm.add
// SPM-NOT: memref.subview

//--- cx.mlir

func.func @reject_dynamic_cx_subview(
    %input: memref<4x8xf16, #wafer.memory<ddr, cx>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %view = memref.subview %input[%i, 0] [1, 8] [1, 1]
        : memref<4x8xf16, #wafer.memory<ddr, cx>>
       to memref<1x8xf16, strided<[8, 1], offset: ?>, #wafer.memory<ddr, cx>>
  }
  return
}

// CX: unsupported_target_address: dynamic subview requires matching tensor-layout source and result memory spaces

//--- ncx.mlir

func.func @reject_dynamic_ncx_subview(
    %input: memref<1x4x8xf16, #wafer.memory<ddr, ncx>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %view = memref.subview %input[0, %i, 0] [1, 1, 8] [1, 1, 1]
        : memref<1x4x8xf16, #wafer.memory<ddr, ncx>>
       to memref<1x1x8xf16, strided<[32, 8, 1], offset: ?>, #wafer.memory<ddr, ncx>>
  }
  return
}

// NCX: unsupported_target_address: dynamic subview requires matching tensor-layout source and result memory spaces

//--- dynamic-size.mlir

func.func @reject_dynamic_subview_size(
    %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %view = memref.subview %input[%i] [%c1] [1]
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<?xf16, strided<[1], offset: ?>, #wafer.memory<ddr, tensor>>
  }
  return
}

// SIZE: unsupported_target_address: dynamic tensor subview requires static source and result shapes

//--- dynamic-stride.mlir

func.func @reject_dynamic_subview_stride(
    %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %view = memref.subview %input[%i] [1] [%c1]
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<1xf16, strided<[?], offset: ?>, #wafer.memory<ddr, tensor>>
  }
  return
}

// STRIDE: unsupported_target_address: dynamic tensor subview requires static non-negative sizes and positive strides
