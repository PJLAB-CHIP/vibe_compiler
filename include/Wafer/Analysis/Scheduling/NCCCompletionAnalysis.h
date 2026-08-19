//===- NCCCompletionAnalysis.h - Current-IR pending NCC facts -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_NCCCOMPLETIONANALYSIS_H
#define WAFER_ANALYSIS_NCCCOMPLETIONANALYSIS_H

#include "Wafer/IR/NCCCompletion.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace wafer::analysis {

namespace detail {
class NCCCompletionAnalysisBuilder;
}

struct NCCPendingWorkerSummary {
  uint32_t issuedWorkerMask = 0;
  bool hasCrossWorkerWindow = false;
};

struct NCCOperationPendingState {
  mlir::Operation *operation = nullptr;
  uint32_t pendingBefore = 0;
  uint32_t pendingAfter = 0;
};

/// Query-local control-flow analysis of typed NCC issue and completion
/// operations. Results refer only to the unchanged input IR epoch and are
/// recomputed after mutation. Direct calls are expanded through current symbol
/// definitions; recursion, indirect calls and unsupported CFG fail closed.
class NCCCompletionAnalysis {
public:
  static mlir::FailureOr<NCCCompletionAnalysis>
  create(mlir::ModuleOp module, std::string *failureReason = nullptr);

  const NCCPendingWorkerSummary &getSummary() const { return summary; }
  llvm::ArrayRef<NCCOperationPendingState> getOperationStates() const {
    return states;
  }
  std::optional<NCCOperationPendingState>
  getOperationState(mlir::Operation *operation) const;

private:
  friend class detail::NCCCompletionAnalysisBuilder;

  NCCCompletionAnalysis(NCCPendingWorkerSummary summary,
                        llvm::SmallVector<NCCOperationPendingState, 32> states,
                        llvm::DenseMap<mlir::Operation *, size_t> stateIndices)
      : summary(summary), states(std::move(states)),
        stateIndices(std::move(stateIndices)) {}

  NCCPendingWorkerSummary summary;
  llvm::SmallVector<NCCOperationPendingState, 32> states;
  llvm::DenseMap<mlir::Operation *, size_t> stateIndices;
};

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_NCCCOMPLETIONANALYSIS_H
