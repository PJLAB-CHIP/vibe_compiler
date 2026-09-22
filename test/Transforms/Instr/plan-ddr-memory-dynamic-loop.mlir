// RUN: wafer-opt --wafer-plan-ddr-memory %s | FileCheck %s
// RUN: wafer-opt --wafer-plan-ddr-memory --wafer-lower-instr-to-target-llvm %s | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM
// Runtime loop starts remain on a proven 128-row grid. 1031/1025 also
// execute a separate static remainder load; the last dynamic loop is empty.

module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}

  func.func @dynamic_lower_1024(%source: memref<1x1024x64xbf16, #wafer.memory<ddr, tensor>>) {
    %zero = arith.constant 0 : index
    %step = arith.constant 128 : index
    %outer_step = arith.constant 256 : index
    %bias = arith.constant 127 : index
    %end = arith.constant 1024 : index
    %full_end = arith.constant 1024 : index
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<1x128x64xbf16, #wafer.memory<spm, tensor>>
    scf.for %q = %zero to %end step %outer_step {
      %difference = arith.subi %q, %bias : index
      %positive = arith.maxsi %difference, %zero : index
      %blocks = arith.ceildivsi %positive, %step : index
      %lower = arith.muli %blocks, %step : index
      scf.for %k = %lower to %full_end step %step {
        %view = memref.subview %source[0, %k, 0] [1, 128, 64] [1, 1, 1]
            : memref<1x1024x64xbf16, #wafer.memory<ddr, tensor>>
           to memref<1x128x64xbf16, strided<[65536, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        wafer.instr.rdma %view to %dest {byte_count = 16384 : i64, inner_bytes = 16384 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
            : memref<1x128x64xbf16, strided<[65536, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
           to memref<1x128x64xbf16, #wafer.memory<spm, tensor>>
      }
    }
    wafer.instr.ncc_join [0]
    return
  }
  // CHECK-LABEL: func.func @dynamic_lower_1024
  // CHECK: %[[LOWER1024:.+]] = arith.muli
  // CHECK: scf.for %{{.+}} = %[[LOWER1024]] to
  // CHECK: wafer.instr.rdma
  // LLVM-LABEL: define void @dynamic_lower_1024
  // LLVM: sdiv i64
  // LLVM: mul i64
  // LLVM: call void @wafer_tx81_rdma

  func.func @dynamic_lower_1025(%source: memref<1x1025x64xbf16, #wafer.memory<ddr, tensor>>) {
    %zero = arith.constant 0 : index
    %step = arith.constant 128 : index
    %outer_step = arith.constant 256 : index
    %bias = arith.constant 127 : index
    %end = arith.constant 1025 : index
    %full_end = arith.constant 1024 : index
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<1x128x64xbf16, #wafer.memory<spm, tensor>>
    scf.for %q = %zero to %end step %outer_step {
      %difference = arith.subi %q, %bias : index
      %positive = arith.maxsi %difference, %zero : index
      %blocks = arith.ceildivsi %positive, %step : index
      %lower = arith.muli %blocks, %step : index
      scf.for %k = %lower to %full_end step %step {
        %view = memref.subview %source[0, %k, 0] [1, 128, 64] [1, 1, 1]
            : memref<1x1025x64xbf16, #wafer.memory<ddr, tensor>>
           to memref<1x128x64xbf16, strided<[65600, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        wafer.instr.rdma %view to %dest {byte_count = 16384 : i64, inner_bytes = 16384 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
            : memref<1x128x64xbf16, strided<[65600, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
           to memref<1x128x64xbf16, #wafer.memory<spm, tensor>>
      }
    }
    %tail_dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<81920>} : memref<1x1x64xbf16, #wafer.memory<spm, tensor>>
    %tail_view = memref.subview %source[0, 1024, 0] [1, 1, 64] [1, 1, 1]
        : memref<1x1025x64xbf16, #wafer.memory<ddr, tensor>>
       to memref<1x1x64xbf16, strided<[65600, 64, 1], offset: 65536>, #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %tail_view to %tail_dest {byte_count = 128 : i64, inner_bytes = 128 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
        : memref<1x1x64xbf16, strided<[65600, 64, 1], offset: 65536>, #wafer.memory<ddr, tensor>>
       to memref<1x1x64xbf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    return
  }
  // CHECK-LABEL: func.func @dynamic_lower_1025
  // CHECK: %[[LOWER1025:.+]] = arith.muli
  // CHECK: scf.for %{{.+}} = %[[LOWER1025]] to
  // CHECK: wafer.instr.rdma
  // LLVM-LABEL: define void @dynamic_lower_1025
  // LLVM: sdiv i64
  // LLVM: mul i64
  // LLVM: call void @wafer_tx81_rdma

  func.func @dynamic_lower_1031(%source: memref<1x1031x64xbf16, #wafer.memory<ddr, tensor>>) {
    %zero = arith.constant 0 : index
    %step = arith.constant 128 : index
    %outer_step = arith.constant 256 : index
    %bias = arith.constant 127 : index
    %end = arith.constant 1031 : index
    %full_end = arith.constant 1024 : index
    %dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<1x128x64xbf16, #wafer.memory<spm, tensor>>
    scf.for %q = %zero to %end step %outer_step {
      %difference = arith.subi %q, %bias : index
      %positive = arith.maxsi %difference, %zero : index
      %blocks = arith.ceildivsi %positive, %step : index
      %lower = arith.muli %blocks, %step : index
      scf.for %k = %lower to %full_end step %step {
        %view = memref.subview %source[0, %k, 0] [1, 128, 64] [1, 1, 1]
            : memref<1x1031x64xbf16, #wafer.memory<ddr, tensor>>
           to memref<1x128x64xbf16, strided<[65984, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        wafer.instr.rdma %view to %dest {byte_count = 16384 : i64, inner_bytes = 16384 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
            : memref<1x128x64xbf16, strided<[65984, 64, 1], offset: ?>, #wafer.memory<ddr, tensor>>
           to memref<1x128x64xbf16, #wafer.memory<spm, tensor>>
      }
    }
    %tail_dest = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<81920>} : memref<1x7x64xbf16, #wafer.memory<spm, tensor>>
    %tail_view = memref.subview %source[0, 1024, 0] [1, 7, 64] [1, 1, 1]
        : memref<1x1031x64xbf16, #wafer.memory<ddr, tensor>>
       to memref<1x7x64xbf16, strided<[65984, 64, 1], offset: 65536>, #wafer.memory<ddr, tensor>>
    wafer.instr.rdma %tail_view to %tail_dest {byte_count = 896 : i64, inner_bytes = 896 : i64, src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
        : memref<1x7x64xbf16, strided<[65984, 64, 1], offset: 65536>, #wafer.memory<ddr, tensor>>
       to memref<1x7x64xbf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    return
  }
  // CHECK-LABEL: func.func @dynamic_lower_1031
  // CHECK: %[[LOWER1031:.+]] = arith.muli
  // CHECK: scf.for %{{.+}} = %[[LOWER1031]] to
  // CHECK: wafer.instr.rdma
  // LLVM-LABEL: define void @dynamic_lower_1031
  // LLVM: sdiv i64
  // LLVM: mul i64
  // LLVM: call void @wafer_tx81_rdma
}
