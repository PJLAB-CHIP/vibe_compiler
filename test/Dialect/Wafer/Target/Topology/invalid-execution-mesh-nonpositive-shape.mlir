// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.execution.mesh @mesh
      {axes = ["partition"], shape = array<i64: 0>}
}

// CHECK: shape entries must be positive
