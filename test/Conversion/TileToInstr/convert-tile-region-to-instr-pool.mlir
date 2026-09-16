// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @pool_max_1024() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%local: i1):
    %input = memref.alloc() : memref<1x5x1026x8xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.pool #wafer.reduce_kind<max> %input
        {kernel = array<i64: 2, 3>, strides = array<i64: 2, 1>,
         dilations = array<i64: 1, 1>}
        : memref<1x5x1026x8xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1024x8xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %local : i1
  }
  return
}

// CHECK-LABEL: func.func @pool_max_1024
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x2x1024x8xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.pool <max> %{{.+}} into %[[DEST]]
// CHECK-SAME: dest_shape = array<i64: 1, 2, 1024, 8>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 1, 2>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK-SAME: source_shape = array<i64: 1, 5, 1026, 8>
// CHECK-NOT: wafer.tile.pool
// CHECK: return

func.func @pool_max_1025() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%local: i1):
    %input = memref.alloc() : memref<1x5x1027x8xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.pool #wafer.reduce_kind<max> %input
        {kernel = array<i64: 2, 3>, strides = array<i64: 2, 1>,
         dilations = array<i64: 1, 1>}
        : memref<1x5x1027x8xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1025x8xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %local : i1
  }
  return
}

// CHECK-LABEL: func.func @pool_max_1025
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x2x1025x8xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.pool <max> %{{.+}} into %[[DEST]]
// CHECK-SAME: dest_shape = array<i64: 1, 2, 1025, 8>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 1, 2>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK-SAME: source_shape = array<i64: 1, 5, 1027, 8>
// CHECK-NOT: wafer.tile.pool
// CHECK: return

func.func @pool_avg_1024() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%local: i1):
    %input = memref.alloc() : memref<1x5x1026x8xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.pool #wafer.reduce_kind<avg> %input
        {kernel = array<i64: 2, 3>, strides = array<i64: 2, 1>,
         dilations = array<i64: 1, 1>}
        : memref<1x5x1026x8xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1024x8xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %local : i1
  }
  return
}

// CHECK-LABEL: func.func @pool_avg_1024
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x2x1024x8xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.pool <avg> %{{.+}} into %[[DEST]]
// CHECK-SAME: dest_shape = array<i64: 1, 2, 1024, 8>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 1, 2>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK-SAME: source_shape = array<i64: 1, 5, 1026, 8>
// CHECK-NOT: wafer.tile.pool
// CHECK: return

func.func @pool_avg_1025() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%local: i1):
    %input = memref.alloc() : memref<1x5x1027x8xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.pool #wafer.reduce_kind<avg> %input
        {kernel = array<i64: 2, 3>, strides = array<i64: 2, 1>,
         dilations = array<i64: 1, 1>}
        : memref<1x5x1027x8xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1025x8xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %local : i1
  }
  return
}

// CHECK-LABEL: func.func @pool_avg_1025
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x2x1025x8xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.pool <avg> %{{.+}} into %[[DEST]]
// CHECK-SAME: dest_shape = array<i64: 1, 2, 1025, 8>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 1, 2>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK-SAME: source_shape = array<i64: 1, 5, 1027, 8>
// CHECK-NOT: wafer.tile.pool
// CHECK: return

func.func @pool_min_1024() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%local: i1):
    %input = memref.alloc() : memref<1x5x1026x8xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.pool #wafer.reduce_kind<min> %input
        {kernel = array<i64: 2, 3>, strides = array<i64: 2, 1>,
         dilations = array<i64: 1, 1>}
        : memref<1x5x1026x8xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1024x8xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %local : i1
  }
  return
}

// CHECK-LABEL: func.func @pool_min_1024
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x2x1024x8xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.pool <min> %{{.+}} into %[[DEST]]
// CHECK-SAME: dest_shape = array<i64: 1, 2, 1024, 8>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 1, 2>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK-SAME: source_shape = array<i64: 1, 5, 1026, 8>
// CHECK-NOT: wafer.tile.pool
// CHECK: return

func.func @pool_min_1025() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%local: i1):
    %input = memref.alloc() : memref<1x5x1027x8xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.pool #wafer.reduce_kind<min> %input
        {kernel = array<i64: 2, 3>, strides = array<i64: 2, 1>,
         dilations = array<i64: 1, 1>}
        : memref<1x5x1027x8xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1025x8xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %local : i1
  }
  return
}

// CHECK-LABEL: func.func @pool_min_1025
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x2x1025x8xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.pool <min> %{{.+}} into %[[DEST]]
// CHECK-SAME: dest_shape = array<i64: 1, 2, 1025, 8>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 1, 2>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK-SAME: source_shape = array<i64: 1, 5, 1027, 8>
// CHECK-NOT: wafer.tile.pool
// CHECK: return

func.func @pool_sum_1024() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%local: i1):
    %input = memref.alloc() : memref<1x5x1026x8xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.pool #wafer.reduce_kind<sum> %input
        {kernel = array<i64: 2, 3>, strides = array<i64: 2, 1>,
         dilations = array<i64: 1, 1>}
        : memref<1x5x1026x8xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1024x8xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %local : i1
  }
  return
}

// CHECK-LABEL: func.func @pool_sum_1024
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x2x1024x8xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.pool <sum> %{{.+}} into %[[DEST]]
// CHECK-SAME: dest_shape = array<i64: 1, 2, 1024, 8>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 1, 2>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK-SAME: source_shape = array<i64: 1, 5, 1026, 8>
// CHECK-NOT: wafer.tile.pool
// CHECK: return

func.func @pool_sum_1025() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%local: i1):
    %input = memref.alloc() : memref<1x5x1027x8xf16, #wafer.memory<spm, ncx>>
    %result = wafer.tile.pool #wafer.reduce_kind<sum> %input
        {kernel = array<i64: 2, 3>, strides = array<i64: 2, 1>,
         dilations = array<i64: 1, 1>}
        : memref<1x5x1027x8xf16, #wafer.memory<spm, ncx>>
       to memref<1x2x1025x8xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %local : i1
  }
  return
}

// CHECK-LABEL: func.func @pool_sum_1025
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x2x1025x8xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.pool <sum> %{{.+}} into %[[DEST]]
// CHECK-SAME: dest_shape = array<i64: 1, 2, 1025, 8>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 1, 2>
// CHECK-SAME: pads = array<i64: 0, 0, 0, 0>
// CHECK-SAME: source_shape = array<i64: 1, 5, 1027, 8>
// CHECK-NOT: wafer.tile.pool
// CHECK: return
