//===- NCCCompletionAnalysis.cpp - Current-IR pending NCC facts -------===//

#include "Wafer/Analysis/Instr/NCCCompletionAnalysis.h"

#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

namespace wafer::analysis {
namespace detail {

class NCCCompletionAnalysisBuilder {
public:
  NCCCompletionAnalysisBuilder(mlir::ModuleOp module,
                               std::string *failureReason)
      : module(module), failureReason(failureReason) {}

  mlir::FailureOr<NCCCompletionAnalysis> build() {
    if (!module)
      return fail("NCC completion analysis requires a module");
    for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
      if (function.isExternal())
        continue;
      if (!function.getBody().hasOneBlock())
        return fail("NCC completion analysis requires structured single-block "
                    "functions");
      uint32_t pending = 0;
      llvm::DenseSet<mlir::Operation *> activeCalls;
      if (mlir::failed(analyzeFunction(function, pending, activeCalls)))
        return mlir::failure();
    }
    return NCCCompletionAnalysis(summary, std::move(states),
                                 std::move(stateIndices));
  }

private:
  mlir::FailureOr<NCCCompletionAnalysis> fail(llvm::StringRef message) {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  }

  mlir::LogicalResult failResult(llvm::StringRef message) {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  }

  void record(mlir::Operation *operation, uint32_t before, uint32_t after) {
    auto [entry, inserted] = stateIndices.try_emplace(operation, states.size());
    if (inserted) {
      states.push_back({operation, before, after});
      return;
    }
    NCCOperationPendingState &state = states[entry->second];
    state.pendingBefore |= before;
    state.pendingAfter |= after;
  }

  mlir::LogicalResult
  analyzeFunction(mlir::func::FuncOp function, uint32_t &pending,
                  llvm::DenseSet<mlir::Operation *> &activeCalls) {
    if (!activeCalls.insert(function.getOperation()).second)
      return failResult("NCC completion analysis rejects recursive calls");
    mlir::LogicalResult result =
        analyzeBlock(function.getBody().front(), pending, activeCalls);
    activeCalls.erase(function.getOperation());
    return result;
  }

  mlir::LogicalResult
  analyzeBlock(mlir::Block &block, uint32_t &pending,
               llvm::DenseSet<mlir::Operation *> &activeCalls) {
    for (mlir::Operation &operation : block)
      if (mlir::failed(analyzeOperation(&operation, pending, activeCalls)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult
  analyzeOperation(mlir::Operation *operation, uint32_t &pending,
                   llvm::DenseSet<mlir::Operation *> &activeCalls) {
    const uint32_t before = pending;
    if (std::optional<SingleExecutionRegionFlow> flow =
            getSingleExecutionRegionFlow(operation)) {
      if (mlir::failed(
              analyzeBlock(flow->region->front(), pending, activeCalls)))
        return mlir::failure();
      record(operation, before, pending);
      return mlir::success();
    }
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
      uint32_t thenPending = pending;
      if (mlir::failed(analyzeBlock(ifOp.getThenRegion().front(), thenPending,
                                    activeCalls)))
        return mlir::failure();
      uint32_t elsePending = pending;
      if (!ifOp.getElseRegion().empty() &&
          mlir::failed(analyzeBlock(ifOp.getElseRegion().front(), elsePending,
                                    activeCalls)))
        return mlir::failure();
      pending = thenPending | elsePending;
      record(operation, before, pending);
      return mlir::success();
    }
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      uint32_t bodyPending = pending;
      if (mlir::failed(
              analyzeBlock(*forOp.getBody(), bodyPending, activeCalls)))
        return mlir::failure();
      pending |= bodyPending;
      record(operation, before, pending);
      return mlir::success();
    }
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
      auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (!callee || callee.isExternal() || !callee.getBody().hasOneBlock())
        return failResult("NCC completion analysis requires a defined direct "
                          "single-block callee");
      if (mlir::failed(analyzeFunction(callee, pending, activeCalls)))
        return mlir::failure();
      record(operation, before, pending);
      return mlir::success();
    }
    if (mlir::isa<mlir::CallOpInterface>(operation))
      return failResult("NCC completion analysis rejects indirect or unknown "
                        "call operations");

    NCCOperationCompletion completion = getNCCOperationCompletion(operation);
    if (completion.issueWorker) {
      const uint32_t worker = static_cast<uint32_t>(*completion.issueWorker);
      if (worker >= kNCCWorkerCount)
        return failResult("NCC completion operation has an invalid worker");
      const uint32_t workerMask = uint32_t{1} << worker;
      summary.issuedWorkerMask |= workerMask;
      pending |= workerMask;
      summary.hasCrossWorkerWindow |= llvm::popcount(pending) >= 2;
    }
    if (completion.kind == NCCCompletionKind::ParticipantJoin ||
        completion.kind == NCCCompletionKind::SynchronousWriteback) {
      if (completion.participantMask == 0 ||
          (completion.participantMask & ~kAllNCCWorkersMask) != 0)
        return failResult(
            "NCC completion operation has an invalid participant mask");
      pending &= ~completion.participantMask;
    }
    record(operation, before, pending);
    return mlir::success();
  }

  mlir::ModuleOp module;
  std::string *failureReason;
  NCCPendingWorkerSummary summary;
  llvm::SmallVector<NCCOperationPendingState, 32> states;
  llvm::DenseMap<mlir::Operation *, size_t> stateIndices;
};

} // namespace detail

mlir::FailureOr<NCCCompletionAnalysis>
NCCCompletionAnalysis::create(mlir::ModuleOp module,
                              std::string *failureReason) {
  return detail::NCCCompletionAnalysisBuilder(module, failureReason).build();
}

std::optional<NCCOperationPendingState>
NCCCompletionAnalysis::getOperationState(mlir::Operation *operation) const {
  auto found = stateIndices.find(operation);
  return found == stateIndices.end()
             ? std::nullopt
             : std::optional<NCCOperationPendingState>(states[found->second]);
}

} // namespace wafer::analysis
