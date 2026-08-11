//===- BoundedTileExecutor.h - Independent physical Tile work -*- C++ -*-===//

#ifndef WAFER_COMPILER_BOUNDEDTILEEXECUTOR_H
#define WAFER_COMPILER_BOUNDEDTILEEXECUTOR_H

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
/// workload and the MLIR context pool. Keep the default request unconstrained
/// so host capability determines how many independent Tile pipelines run.
inline constexpr unsigned kMaximumBoundedTilePipelineWorkers =
    std::numeric_limits<unsigned>::max();

inline unsigned getBoundedTilePipelineWorkerCount(
    mlir::MLIRContext *context, size_t tileCount,
    unsigned requestedMaximum = kMaximumBoundedTilePipelineWorkers) {
  if (!context || tileCount <= 1 || !context->isMultithreadingEnabled())
    return 1;
  return std::max<unsigned>(
      1, std::min<unsigned>({std::max<unsigned>(1, requestedMaximum),
                             static_cast<unsigned>(tileCount),
                             context->getThreadPool().getMaxConcurrency()}));
}

/// Runs only mutually independent physical Tile transformations. Callers own
/// all result slots and perform whole-card transport, resource, ABI-domain, or
/// admission decisions only after this function returns. Contiguous shards
/// preserve deterministic diagnostic ordering.
template <typename FunctionT>
unsigned runBoundedTilePipelines(
    mlir::MLIRContext *context, size_t tileCount, FunctionT &&function,
    unsigned requestedMaximum = kMaximumBoundedTilePipelineWorkers) {
  const unsigned workerCount =
      getBoundedTilePipelineWorkerCount(context, tileCount, requestedMaximum);
  if (workerCount == 1) {
    for (size_t tile = 0; tile < tileCount; ++tile)
      function(tile);
    return workerCount;
  }

  // Pass pipelines may lazily request registered dialects. Resolve that
  // context-global initialization before independent Tile workers enter the
  // parallel region.
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
    const size_t begin = tileCount * worker / workerCount;
    const size_t end = tileCount * (worker + 1) / workerCount;
    for (size_t tile = begin; tile < end; ++tile)
      function(tile);
  });
  return workerCount;
}

/// Uses bounded shared-context parallelism when every physical Tile module
/// belongs to one owner context. Inputs from isolated contexts retain a
/// deterministic serial path.
template <typename FunctionT>
unsigned runBoundedTileModulePipelines(
    llvm::ArrayRef<mlir::ModuleOp> modules, FunctionT &&function,
    unsigned requestedMaximum = kMaximumBoundedTilePipelineWorkers) {
  if (modules.empty())
    return 1;
  mlir::ModuleOp firstModule = modules.front();
  mlir::MLIRContext *context = firstModule.getContext();
  if (llvm::any_of(modules, [&](mlir::ModuleOp module) {
        return module.getOperation()->getContext() != context;
      })) {
    for (size_t tile = 0; tile < modules.size(); ++tile)
      function(tile);
    return 1;
  }
  return runBoundedTilePipelines(context, modules.size(),
                                 std::forward<FunctionT>(function),
                                 requestedMaximum);
}

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_BOUNDEDTILEEXECUTOR_H
