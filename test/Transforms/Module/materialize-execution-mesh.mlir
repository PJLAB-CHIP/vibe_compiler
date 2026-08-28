// RUN: wafer-opt --wafer-materialize-execution-mesh='shape=1' %s | FileCheck %s
// RUN: wafer-opt --wafer-materialize-execution-mesh='mesh-name=debug axes=y,x shape=2,2' %s | FileCheck --check-prefix=LOGICAL %s

module {}

// CHECK: wafer.execution.mesh @default_mesh
// CHECK-SAME: axes = ["card_partition"]
// CHECK-SAME: shape = array<i64: 1>

// LOGICAL: wafer.execution.mesh @debug
// LOGICAL-SAME: axes = ["y", "x"]
// LOGICAL-SAME: shape = array<i64: 2, 2>
