// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | FileCheck %s --check-prefix=POS
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %t/positive.mlir | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVMIR
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/out-of-range.mlir 2>&1 | FileCheck %s --check-prefix=RANGE
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/unbounded.mlir 2>&1 | FileCheck %s --check-prefix=UNBOUNDED
// RUN: not wafer-opt %t/conflicting-forms.mlir 2>&1 | FileCheck %s --check-prefix=CONFLICT

//--- positive.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}

  func.func @bounded_dynamic_gather_scatter() {
    %source = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x1025x2xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<81920>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1025 = arith.constant 1025 : index
    scf.for %tuple = %c0 to %c1025 step %c1 {
      %byte_offset = arith.muli %tuple, %c2 : index
      wafer.instr.gather_scatter %source to %dest
          src_offset_value(%byte_offset)
          {byte_count = 2 : i64, inner_bytes = 2 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x1025x2xf16, #wafer.memory<spm, tensor>>
         to memref<1xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
    }
    return
  }
}

// POS-LABEL: llvm.func @bounded_dynamic_gather_scatter
// POS: %[[SOURCE:.+]] = llvm.mlir.constant(65536 : i64) : i64
// POS: ^bb{{[0-9]+}}(%[[TUPLE:.+]]: i64):
// POS: %[[DYNAMIC:.+]] = llvm.mul %[[TUPLE]], %{{.+}} : i64
// POS: %[[ADDRESS:.+]] = llvm.add %[[SOURCE]], %[[DYNAMIC]] : i64
// POS: llvm.call @wafer_tx81_gather_scatter(%[[ADDRESS]]
// POS-NOT: wafer.
// POS-NOT: scf.for

// LLVMIR-LABEL: define void @bounded_dynamic_gather_scatter()
// LLVMIR: %[[DYNAMIC:.+]] = mul i64 %{{.+}}, 2
// LLVMIR: %[[ADDRESS:.+]] = add i64 65536, %[[DYNAMIC]]
// LLVMIR: call void @wafer_tx81_gather_scatter(i64 %[[ADDRESS]]

//--- out-of-range.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}

  func.func @out_of_range_dynamic_gather_scatter() {
    %source = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x1025x2xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<81920>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c5000 = arith.constant 5000 : index
    scf.for %tuple = %c0 to %c5000 step %c1 {
      %byte_offset = arith.muli %tuple, %c2 : index
      wafer.instr.gather_scatter %source to %dest
          src_offset_value(%byte_offset)
          {byte_count = 2 : i64, inner_bytes = 2 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<2x1025x2xf16, #wafer.memory<spm, tensor>>
         to memref<1xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
    }
    return
  }
}

// RANGE: target_geometry_mismatch: source dynamic descriptor byte range [0, 10000) exceeds the physical buffer 'memref<2x1025x2xf16, #wafer.memory<spm, tensor>>' (bytes=8200)

//--- unbounded.mlir

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}

  func.func @unbounded_dynamic_gather_scatter(%byte_offset: index) {
    %source = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x1025x2xf16, #wafer.memory<spm, tensor>>
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<81920>}
        : memref<1xf16, #wafer.memory<spm, tensor>>
    wafer.instr.gather_scatter %source to %dest
        src_offset_value(%byte_offset)
        {byte_count = 2 : i64, inner_bytes = 2 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<2x1025x2xf16, #wafer.memory<spm, tensor>>
       to memref<1xf16, #wafer.memory<spm, tensor>>
    return
  }
}

// UNBOUNDED: unsupported_target_dynamic_offset: source offset requires a constant-bounded non-negative index expression

//--- conflicting-forms.mlir

func.func @conflicting_dynamic_gather_scatter(%byte_offset: index) {
  %source = memref.alloc()
      : memref<2x1025x2xf16, #wafer.memory<spm, tensor>>
  %dest = memref.alloc() : memref<1xf16, #wafer.memory<spm, tensor>>
  wafer.instr.gather_scatter %source to %dest
      src_offset_value(%byte_offset)
      {byte_count = 2 : i64, inner_bytes = 2 : i64,
       src_offset = 2 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<2x1025x2xf16, #wafer.memory<spm, tensor>>
     to memref<1xf16, #wafer.memory<spm, tensor>>
  return
}

// CONFLICT: each endpoint byte offset must use either its static attribute or its SSA value, not both
