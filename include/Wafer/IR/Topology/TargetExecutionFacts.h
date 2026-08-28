//===- TargetExecutionFacts.h - Clone target execution IR -*- C++ -*-===//

#ifndef WAFER_IR_TOPOLOGY_TARGETEXECUTIONFACTS_H
#define WAFER_IR_TOPOLOGY_TARGETEXECUTIONFACTS_H

#include "mlir/IR/BuiltinOps.h"

namespace wafer {

/// Clones the direct typed target-topology and execution-mesh facts from one
/// module into another. This keeps standalone compiler-generated modules
/// self-contained without converting topology-derived choices into persistent
/// side data.
void cloneTargetExecutionFacts(mlir::ModuleOp sourceModule,
                               mlir::ModuleOp destinationModule);

} // namespace wafer

#endif // WAFER_IR_TOPOLOGY_TARGETEXECUTIONFACTS_H
