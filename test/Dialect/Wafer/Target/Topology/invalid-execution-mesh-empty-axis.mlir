// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.execution.mesh @mesh
      {axes = [""], shape = array<i64: 1>}
}

// CHECK: axis name must not be empty
