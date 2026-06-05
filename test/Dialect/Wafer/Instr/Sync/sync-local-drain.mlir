// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.instr.local_drain
}

// CHECK: wafer.instr.local_drain
