// RUN: not wafer-opt --allow-unregistered-dialect \
// RUN:   --wafer-convert-tile-region-to-instr %s 2>&1 \
// RUN:   | FileCheck %s

func.func @unknown_effect(
    %slot: memref<4xf32, #wafer.memory<spm, tensor>>, %zero: f32) {
  wafer.instr.fill %slot, %zero
      : memref<4xf32, #wafer.memory<spm, tensor>>, f32
  "test.untracked_observer"() : () -> ()
  return
}

// CHECK: instruction_completion_failure: operation without a typed memory-effect contract may observe pending NCC work
