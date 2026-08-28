// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.execution.mesh @mesh
      {axes = ["partition_x", "partition_y"],
       shape = array<i64: 9223372036854775807, 2>}
}

// CHECK: partition count is too large to verify
