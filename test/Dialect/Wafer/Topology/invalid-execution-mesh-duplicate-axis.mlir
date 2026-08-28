// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.execution.mesh @mesh
      {axes = ["partition", "partition"], shape = array<i64: 1, 1>}
}

// CHECK: axis name must be unique
