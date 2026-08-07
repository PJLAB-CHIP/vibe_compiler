//===- BoundedRankExecutor.h - Independent rank pipeline work -*- C++ -*-===//

#ifndef WAFER_COMPILER_BOUNDEDRANKEXECUTOR_H
#define WAFER_COMPILER_BOUNDEDRANKEXECUTOR_H

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Threading.h"

#include "llvm/ADT/ArrayRef.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace wafer::compiler::detail {

/// The effective worker count is always bounded by the current independent
/// workload and the MLIR context pool.  Keep the default request unconstrained
/// so host capability, rather than a compiler-internal four-worker constant,
/// determines how many genuinely independent rank/candidate pipelines run.
inline constexpr unsigned kMaximumBoundedRankPipelineWorkers =
    std::numeric_limits<unsigned>::max();

inline unsigned getBoundedRankPipelineWorkerCount(
    mlir::MLIRContext *context, size_t rankCount,
    unsigned requestedMaximum = kMaximumBoundedRankPipelineWorkers) {
  if (!context || rankCount <= 1 || !context->isMultithreadingEnabled())
    return 1;
  return std::max<unsigned>(
      1, std::min<unsigned>({std::max<unsigned>(1, requestedMaximum),
                             static_cast<unsigned>(rankCount),
                             context->getThreadPool().getMaxConcurrency()}));
}

/// Runs only mutually independent rank-module transformations. Callers retain
/// ownership of all result slots and perform any cross-rank transport,
/// resource, ABI-domain, or selection decision after this function returns.
/// Contiguous shards preserve deterministic diagnostic ordering.
template <typename FunctionT>
unsigned runBoundedRankPipelines(
    mlir::MLIRContext *context, size_t rankCount, FunctionT &&function,
    unsigned requestedMaximum = kMaximumBoundedRankPipelineWorkers) {
  const unsigned workerCount =
      getBoundedRankPipelineWorkerCount(context, rankCount, requestedMaximum);
  if (workerCount == 1) {
    for (size_t rank = 0; rank < rankCount; ++rank)
      function(rank);
    return workerCount;
  }

  // Pass pipelines may lazily request registered dialects. Resolve that
  // context-global initialization before independent rank workers enter the
  // parallel region; concurrent first-use loading is not an MLIR-supported
  // rank-local operation.
  context->loadAllAvailableDialects();

  std::shared_ptr<wafer::support::CompileTimingSession> timingSession =
      wafer::support::getActiveCompileTimingSession();
  std::shared_ptr<wafer::support::CompileWorkStatisticsSession>
      workStatisticsSession =
          wafer::support::getActiveCompileWorkStatisticsSession();
  mlir::parallelFor(context, 0, workerCount, [&](size_t worker) {
    wafer::support::ScopedCompileTimingActivation timingActivation(
        timingSession);
    wafer::support::ScopedCompileWorkStatisticsActivation
        workStatisticsActivation(workStatisticsSession);
    const size_t begin = rankCount * worker / workerCount;
    const size_t end = rankCount * (worker + 1) / workerCount;
    for (size_t rank = begin; rank < end; ++rank)
      function(rank);
  });
  return workerCount;
}

/// Uses bounded shared-context parallelism when every rank module belongs to
/// one owner context. Qualification inputs may still arrive from isolated
/// contexts; those retain their deterministic serial path.
template <typename FunctionT>
unsigned runBoundedRankModulePipelines(
    llvm::ArrayRef<mlir::ModuleOp> modules, FunctionT &&function,
    unsigned requestedMaximum = kMaximumBoundedRankPipelineWorkers) {
  if (modules.empty())
    return 1;
  mlir::ModuleOp firstModule = modules.front();
  mlir::MLIRContext *context = firstModule.getContext();
  if (llvm::any_of(modules, [&](mlir::ModuleOp module) {
        return module.getOperation()->getContext() != context;
      })) {
    for (size_t rank = 0; rank < modules.size(); ++rank)
      function(rank);
    return 1;
  }
  return runBoundedRankPipelines(context, modules.size(),
                                 std::forward<FunctionT>(function),
                                 requestedMaximum);
}

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_BOUNDEDRANKEXECUTOR_H
