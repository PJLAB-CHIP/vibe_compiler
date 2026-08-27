//===- RedundantTransferElimination.h - Exact storage coalescing -*- C++ -*-===//

#ifndef WAFER_LIB_TRANSFORMS_SCHEDULING_REDUNDANTTRANSFERELIMINATION_H
#define WAFER_LIB_TRANSFORMS_SCHEDULING_REDUNDANTTRANSFERELIMINATION_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace wafer::tensor_program_scheduling {

/// Coalesces a movement-defined SPM allocation with its exact source storage
/// only when current IR proves that no observable state is lost. This is a
/// policy-free canonicalization over one disposable actual clone.
unsigned elideRedundantFullBufferTransfers(mlir::ModuleOp module);

/// Same transformation with an invocation-local notification for every exact
/// destination value replaced by a current source/view. The callback lets a
/// caller-owned relation recorder retarget its current SSA facts in the same
/// transaction; it is not a side table or ownership inference hook.
unsigned elideRedundantFullBufferTransfers(
    mlir::ModuleOp module,
    llvm::function_ref<void(mlir::Value oldValue, mlir::Value newValue)>
        notifyReplacement);

} // namespace wafer::tensor_program_scheduling

#endif // WAFER_LIB_TRANSFORMS_SCHEDULING_REDUNDANTTRANSFERELIMINATION_H
