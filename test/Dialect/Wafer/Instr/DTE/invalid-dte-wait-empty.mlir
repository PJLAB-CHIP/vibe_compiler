// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  "wafer.instr.dte_wait"() : () -> ()
}

// CHECK: DTE wait must have at least one token
