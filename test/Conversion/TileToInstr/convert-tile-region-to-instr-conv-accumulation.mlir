// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s > %t
// RUN: FileCheck %s --check-prefix=COUNT < %t
// RUN: FileCheck %s --check-prefix=ORDER < %t

// COUNT-LABEL: func.func @seven
// COUNT-COUNT-7: wafer.instr.elementwise <mul>
// COUNT-NOT: wafer.instr.elementwise <mul>
// COUNT-NOT: wafer.instr.conv
// COUNT: return
func.func @seven() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %input = memref.alloc() : memref<1x1x1024x7xf16, #wafer.memory<spm, ncx>>
    %weight = memref.alloc() : memref<1x1x2x7xf16, #wafer.memory<spm, cx>>
    %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 0, 0, 0, 0>, unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>}
      : (memref<1x1x1024x7xf16, #wafer.memory<spm, ncx>>, memref<1x1x2x7xf16, #wafer.memory<spm, cx>>) -> memref<1x1x1024x2xf32, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// COUNT-LABEL: func.func @one
// COUNT-COUNT-1: wafer.instr.elementwise <mul>
// COUNT-NOT: wafer.instr.elementwise <mul>
// COUNT-NOT: wafer.instr.conv
// COUNT: return
func.func @one() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %input = memref.alloc() : memref<1x1x1025x1xf16, #wafer.memory<spm, ncx>>
    %weight = memref.alloc() : memref<1x1x2x1xf16, #wafer.memory<spm, cx>>
    %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 0, 0, 0, 0>, unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>}
      : (memref<1x1x1025x1xf16, #wafer.memory<spm, ncx>>, memref<1x1x2x1xf16, #wafer.memory<spm, cx>>) -> memref<1x1x1025x2xf32, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// COUNT-LABEL: func.func @six
// COUNT-COUNT-6: wafer.instr.elementwise <mul>
// COUNT-NOT: wafer.instr.elementwise <mul>
// COUNT-NOT: wafer.instr.conv
// COUNT: return
func.func @six() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %input = memref.alloc() : memref<1x2x1027x1xf16, #wafer.memory<spm, ncx>>
    %weight = memref.alloc() : memref<2x3x2x1xf16, #wafer.memory<spm, cx>>
    %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 0, 0, 0, 0>, unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>}
      : (memref<1x2x1027x1xf16, #wafer.memory<spm, ncx>>, memref<2x3x2x1xf16, #wafer.memory<spm, cx>>) -> memref<1x1x1025x2xf32, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// COUNT-LABEL: func.func @four
// COUNT-COUNT-4: wafer.instr.elementwise <mul>
// COUNT-NOT: wafer.instr.elementwise <mul>
// COUNT-NOT: wafer.instr.conv
// COUNT: return
func.func @four() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %input = memref.alloc() : memref<1x3x2063x1xf16, #wafer.memory<spm, ncx>>
    %weight = memref.alloc() : memref<2x2x2x1xf16, #wafer.memory<spm, cx>>
    %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 0, 0, 0, 0>, unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 1, 2>, dilations = array<i64: 2, 1>}
      : (memref<1x3x2063x1xf16, #wafer.memory<spm, ncx>>, memref<2x2x2x1xf16, #wafer.memory<spm, cx>>) -> memref<1x1x1031x2xf32, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// COUNT-LABEL: func.func @bfloat
// COUNT: wafer.instr.conv <conv>
// COUNT-NOT: wafer.instr.elementwise
// COUNT: return
func.func @bfloat() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %input = memref.alloc() : memref<1x1x1031x7xbf16, #wafer.memory<spm, ncx>>
    %weight = memref.alloc() : memref<1x1x2x7xbf16, #wafer.memory<spm, cx>>
    %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 0, 0, 0, 0>, unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>}
      : (memref<1x1x1031x7xbf16, #wafer.memory<spm, ncx>>, memref<1x1x2x7xbf16, #wafer.memory<spm, cx>>) -> memref<1x1x1031x2xf32, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// COUNT-LABEL: func.func @half
// COUNT: wafer.instr.conv <conv>
// COUNT-NOT: wafer.instr.elementwise
// COUNT: return
func.func @half() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
    %input = memref.alloc() : memref<1x1x1024x7xf16, #wafer.memory<spm, ncx>>
    %weight = memref.alloc() : memref<1x1x2x7xf16, #wafer.memory<spm, cx>>
    %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 0, 0, 0, 0>, unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>}
      : (memref<1x1x1024x7xf16, #wafer.memory<spm, ncx>>, memref<1x1x2x7xf16, #wafer.memory<spm, cx>>) -> memref<1x1x1024x2xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// K=7 proves the three remainder terms all update lane 0, while each
// complete group updates four independent lanes. The merge consumes the
// actual final accumulators, including the reusable spare buffer.
// ORDER-LABEL: func.func @seven
// ORDER: wafer.instr.fill %[[P0:[^, ]+]],
// ORDER: wafer.instr.fill %[[P1:[^, ]+]],
// ORDER: wafer.instr.fill %[[P2:[^, ]+]],
// ORDER: wafer.instr.fill %[[P3:[^, ]+]],
// ORDER: wafer.instr.elementwise <mul> {{.*}} into %[[PRODUCT:[^ ]+]]
// ORDER: wafer.instr.elementwise <add> %[[P0]], %[[PRODUCT]] into %[[SPARE:[^ ]+]]
// ORDER: wafer.instr.elementwise <add> %[[P1]], %[[PRODUCT]] into %[[P0]]
// ORDER: wafer.instr.elementwise <add> %[[P2]], %[[PRODUCT]] into %[[P1]]
// ORDER: wafer.instr.elementwise <add> %[[P3]], %[[PRODUCT]] into %[[P2]]
// ORDER: wafer.instr.elementwise <add> %[[SPARE]], %[[PRODUCT]] into %[[P3]]
// ORDER: wafer.instr.elementwise <add> %[[P3]], %[[PRODUCT]] into %[[SPARE]]
// ORDER: wafer.instr.elementwise <add> %[[SPARE]], %[[PRODUCT]] into %[[P3]]
// ORDER: wafer.instr.elementwise <add> %[[P3]], %[[P0]] into %[[SPARE]]
// ORDER: wafer.instr.elementwise <add> %[[SPARE]], %[[P1]] into %[[P3]]
// ORDER: wafer.instr.elementwise <add> %[[P3]], %[[P2]] into %[[SPARE]]
// ORDER: wafer.instr.gather_scatter %[[SPARE]]
// ORDER: return
