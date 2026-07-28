// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.instr.ncc_join [0]
  wafer.instr.ncc_join [0, 2]
}

// CHECK: wafer.instr.ncc_join [0]
// CHECK: wafer.instr.ncc_join [0, 2]
