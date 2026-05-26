// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.sync.local_drain
}

// CHECK: wafer.sync.local_drain
