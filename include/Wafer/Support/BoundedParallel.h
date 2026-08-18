//===- BoundedParallel.h - Shared-context parallel work -------*- C++ -*-===//

#ifndef WAFER_SUPPORT_BOUNDEDPARALLEL_H
#define WAFER_SUPPORT_BOUNDEDPARALLEL_H

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/CompileWorkStatistics.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Threading.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace wafer::support {

inline constexpr unsigned kMaximumBoundedParallelWorkers =
    std::numeric_limits<unsigned>::max();

inline unsigned getBoundedParallelWorkerCount(
    mlir::MLIRContext *context, size_t workCount,
    unsigned requestedMaximum = kMaximumBoundedParallelWorkers) {
  if (!context || workCount <= 1 || !context->isMultithreadingEnabled())
    return 1;
  return std::max<unsigned>(
      1, std::min<unsigned>({std::max<unsigned>(1, requestedMaximum),
                             static_cast<unsigned>(workCount),
                             context->getThreadPool().getMaxConcurrency()}));
}

/// Runs independent shared-context work in stable contiguous shards.  The
/// caller owns indexed result slots and performs any cross-item merge only
/// after this function returns.  Invocation timing and work ledgers are
/// propagated explicitly to the bounded worker threads.
template <typename FunctionT>
unsigned runBoundedParallelWork(
    mlir::MLIRContext *context, size_t workCount, FunctionT &&function,
    unsigned requestedMaximum = kMaximumBoundedParallelWorkers) {
  const unsigned workerCount = getBoundedParallelWorkerCount(
      context, workCount, requestedMaximum);
  if (workerCount == 1) {
    for (size_t index = 0; index < workCount; ++index)
      function(index);
    return workerCount;
  }

  context->loadAllAvailableDialects();
  std::shared_ptr<CompileTimingSession> timingSession =
      getActiveCompileTimingSession();
  std::shared_ptr<CompileWorkStatisticsSession> workStatisticsSession =
      getActiveCompileWorkStatisticsSession();
  mlir::parallelFor(context, 0, workerCount, [&](size_t worker) {
    ScopedCompileTimingActivation timingActivation(timingSession);
    ScopedCompileWorkStatisticsActivation workStatisticsActivation(
        workStatisticsSession);
    const size_t begin = workCount * worker / workerCount;
    const size_t end = workCount * (worker + 1) / workerCount;
    for (size_t index = begin; index < end; ++index)
      function(index);
  });
  return workerCount;
}

} // namespace wafer::support

#endif // WAFER_SUPPORT_BOUNDEDPARALLEL_H
