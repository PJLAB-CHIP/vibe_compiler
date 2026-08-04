// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.instr.ncc_join [0]
}

// CHECK: wafer.instr.ncc_join [0]
