// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.instr.local_fence
}

// CHECK: wafer.instr.local_fence
