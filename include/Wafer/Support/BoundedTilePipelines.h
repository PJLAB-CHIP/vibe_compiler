//===- BoundedTilePipelines.h - Independent Tile work ---------*- C++ -*-===//

#ifndef WAFER_SUPPORT_BOUNDEDTILEPIPELINES_H
#define WAFER_SUPPORT_BOUNDEDTILEPIPELINES_H

#include "Wafer/Support/BoundedParallel.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <utility>

namespace wafer::support {

/// The effective worker count is always bounded by the current independent
/// workload and the MLIR context pool. Keep the default request unconstrained
/// so host capability determines how many independent Tile pipelines run.
inline constexpr unsigned kMaximumBoundedTilePipelineWorkers =
    kMaximumBoundedParallelWorkers;

inline unsigned getBoundedTilePipelineWorkerCount(
    mlir::MLIRContext *context, size_t tileCount,
    unsigned requestedMaximum = kMaximumBoundedTilePipelineWorkers) {
  return getBoundedParallelWorkerCount(context, tileCount, requestedMaximum);
}

/// Runs only mutually independent Tile transformations. Callers own
/// all result slots and perform card transport, resource, or ABI checks
/// only after this function returns. Contiguous shards
/// preserve deterministic diagnostic ordering.
template <typename FunctionT>
unsigned runBoundedTilePipelines(
    mlir::MLIRContext *context, size_t tileCount, FunctionT &&function,
    unsigned requestedMaximum = kMaximumBoundedTilePipelineWorkers) {
  return runBoundedParallelWork(
      context, tileCount, std::forward<FunctionT>(function), requestedMaximum);
}

/// Uses bounded shared-context parallelism when every Tile module
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

} // namespace wafer::support

#endif // WAFER_SUPPORT_BOUNDEDTILEPIPELINES_H
