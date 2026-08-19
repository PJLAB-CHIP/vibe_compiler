//===- CompilationStatistics.h - Compile measurements --------*- C++ -*-===//

#ifndef WAFER_COMPILER_COMPILATIONSTATISTICS_H
#define WAFER_COMPILER_COMPILATIONSTATISTICS_H

#include <chrono>
#include <cstdint>
#include <sys/resource.h>

namespace wafer::compiler::detail {

using CompileClock = std::chrono::steady_clock;

inline int64_t elapsedCompileMilliseconds(CompileClock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             CompileClock::now() - start)
      .count();
}

inline uint64_t getCompilePeakRSSKiB() {
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0)
    return 0;
  return static_cast<uint64_t>(usage.ru_maxrss);
}

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COMPILATIONSTATISTICS_H
