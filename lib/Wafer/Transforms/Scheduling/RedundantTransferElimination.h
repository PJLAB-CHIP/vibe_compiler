//===- RedundantTransferElimination.h - Exact storage coalescing -*- C++ -*-===//

#ifndef WAFER_LIB_TRANSFORMS_SCHEDULING_REDUNDANTTRANSFERELIMINATION_H
#define WAFER_LIB_TRANSFORMS_SCHEDULING_REDUNDANTTRANSFERELIMINATION_H

#include "mlir/IR/BuiltinOps.h"

namespace wafer::tensor_program_scheduling {

/// Coalesces a movement-defined SPM allocation with its exact source storage
/// only when current IR proves that no observable state is lost. This is a
/// policy-free canonicalization over one disposable actual clone.
unsigned elideRedundantFullBufferTransfers(mlir::ModuleOp module);

} // namespace wafer::tensor_program_scheduling

#endif // WAFER_LIB_TRANSFORMS_SCHEDULING_REDUNDANTTRANSFERELIMINATION_H
