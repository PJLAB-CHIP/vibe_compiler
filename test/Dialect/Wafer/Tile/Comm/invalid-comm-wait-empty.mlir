// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  "wafer.tile.wait"() : () -> ()
}

// CHECK: comm wait must have at least one token
