// RUN: wafer-opt --wafer-materialize-execution-mesh %s | FileCheck %s
// RUN: wafer-opt --wafer-materialize-execution-mesh='mesh-name=debug topology-name=default policy=explicit axes=y,x shape=2,2 endpoints=0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1' %s | FileCheck --check-prefix=EXPLICIT %s
// RUN: not wafer-opt --wafer-materialize-execution-mesh='topology-name=missing' %s 2>&1 | FileCheck --check-prefix=MISSING %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}
}

// CHECK: wafer.execution.mesh @default_mesh
// CHECK-SAME: axes = ["rank"]
// CHECK-SAME: endpoints = array<i64>
// CHECK-SAME: policy = "all_available"
// CHECK-SAME: shape = array<i64: 16>
// CHECK-SAME: topology = @default

// EXPLICIT: wafer.execution.mesh @debug
// EXPLICIT-SAME: axes = ["y", "x"]
// EXPLICIT-SAME: endpoints = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1>
// EXPLICIT-SAME: policy = "explicit"
// EXPLICIT-SAME: shape = array<i64: 2, 2>
// EXPLICIT-SAME: topology = @default

// MISSING: execution_mesh_failure: referenced target topology not found: missing
