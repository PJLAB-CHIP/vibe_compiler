//===- PassPipeline.h - Verified pass pipeline execution ------*- C++ -*-===//

#ifndef WAFER_SUPPORT_PASSPIPELINE_H
#define WAFER_SUPPORT_PASSPIPELINE_H

#include "Wafer/Support/CompileTiming.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace wafer::support {

/// Runs one explicitly constructed pass pipeline with verification enabled.
/// The caller owns the IR and pipeline boundary; this helper only applies the
/// repository-wide verifier and timing policy.
inline mlir::LogicalResult
runPassPipeline(mlir::ModuleOp module, llvm::StringRef pipelineLabel,
                llvm::function_ref<void(mlir::OpPassManager &)> builder) {
  mlir::PassManager manager(module.getContext());
  manager.enableVerifier(true);
  attachCompileTiming(manager, pipelineLabel);
  builder(manager);
  return manager.run(module);
}

} // namespace wafer::support

#endif // WAFER_SUPPORT_PASSPIPELINE_H
