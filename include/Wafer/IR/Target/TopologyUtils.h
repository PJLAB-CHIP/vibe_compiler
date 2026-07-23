//===- TopologyUtils.h - Target topology IR utilities --------*- C++ -*-===//

#ifndef WAFER_IR_TARGET_TOPOLOGYUTILS_H
#define WAFER_IR_TARGET_TOPOLOGYUTILS_H

#include "mlir/IR/BuiltinOps.h"

namespace wafer {

/// Clones the direct typed target-topology and execution-mesh facts from one
/// module into another. This keeps private standalone compiler artifacts
/// self-contained without converting topology-derived choices into persistent
/// side data.
void cloneTargetExecutionFacts(mlir::ModuleOp sourceModule,
                               mlir::ModuleOp destinationModule);

} // namespace wafer

#endif // WAFER_IR_TARGET_TOPOLOGYUTILS_H
