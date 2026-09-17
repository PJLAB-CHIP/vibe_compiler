// RUN: wafer-opt %s -split-input-file -verify-diagnostics
// Tiny shapes isolate verifier failures; real-scale positive lowering and
// exact byte coverage live in BroadcastLoweringPreservesEveryMappedByte.

func.func @duplicate(%source: memref<2x2x4xf16, #wafer.memory<spm, tensor>>) {
  // expected-error@+1 {{broadcast dimensions must be unique}}
  %result = wafer.tile.broadcast %source {dimensions = array<i64: 1, 1, 3>}
      : memref<2x2x4xf16, #wafer.memory<spm, tensor>>
     -> memref<1x2x2x4xf16, #wafer.memory<spm, tensor>>
  return
}

// -----

func.func @out_of_range(%source: memref<2x3x4xf16, #wafer.memory<spm, tensor>>) {
  // expected-error@+1 {{broadcast dimensions must be within result tensor rank}}
  %result = wafer.tile.broadcast %source {dimensions = array<i64: 1, 2, 4>}
      : memref<2x3x4xf16, #wafer.memory<spm, tensor>>
     -> memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
  return
}

// -----

func.func @mismatched_shape(%source: memref<2x3x4xf16, #wafer.memory<spm, tensor>>) {
  // expected-error@+1 {{broadcast source shape must match mapped result dims}}
  %result = wafer.tile.broadcast %source {dimensions = array<i64: 2, 1, 3>}
      : memref<2x3x4xf16, #wafer.memory<spm, tensor>>
     -> memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
  return
}
