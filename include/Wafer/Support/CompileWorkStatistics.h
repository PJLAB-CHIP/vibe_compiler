//===- CompileWorkStatistics.h - Invocation compile work ------*- C++ -*-===//

#ifndef WAFER_SUPPORT_COMPILEWORKSTATISTICS_H
#define WAFER_SUPPORT_COMPILEWORKSTATISTICS_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace wafer::support {

/// Stable units of compiler work. They count actual compiler actions, not
/// elapsed time, estimated target cycles, or persisted planning state.
enum class CompileWorkKind : size_t {
  TileMemoryPlanning,
  TileToInstructionLowering,
  SPMPlanning,
  DDRPlanning,
  TargetABIModuleClone,
  TargetLowering,
  TargetTranslation,
  Count,
};

struct CompileWorkStatistics {
  uint64_t tileMemoryPlanningInvocations = 0;
  uint64_t tileToInstructionLowerings = 0;
  uint64_t spmPlanningInvocations = 0;
  uint64_t ddrPlanningInvocations = 0;
  uint64_t targetABIModuleClones = 0;
  uint64_t targetLoweringInvocations = 0;
  uint64_t targetTranslationInvocations = 0;
};

/// Thread-safe owner for one compile transaction. Bounded worker owners
/// explicitly propagate the shared session; the snapshot is emitted only as
/// an invocation-local diagnostic and is never serialized.
class CompileWorkStatisticsSession {
public:
  void record(CompileWorkKind kind, uint64_t amount = 1) {
    counters[static_cast<size_t>(kind)].fetch_add(amount,
                                                  std::memory_order_relaxed);
  }

  CompileWorkStatistics snapshot() const {
    auto read = [&](CompileWorkKind kind) {
      return counters[static_cast<size_t>(kind)].load(
          std::memory_order_relaxed);
    };
    return {
        read(CompileWorkKind::TileMemoryPlanning),
        read(CompileWorkKind::TileToInstructionLowering),
        read(CompileWorkKind::SPMPlanning),
        read(CompileWorkKind::DDRPlanning),
        read(CompileWorkKind::TargetABIModuleClone),
        read(CompileWorkKind::TargetLowering),
        read(CompileWorkKind::TargetTranslation),
    };
  }

private:
  static constexpr size_t kCounterCount =
      static_cast<size_t>(CompileWorkKind::Count);
  std::array<std::atomic<uint64_t>, kCounterCount> counters{};
};

inline thread_local std::shared_ptr<CompileWorkStatisticsSession>
    activeCompileWorkStatisticsSession;

inline std::shared_ptr<CompileWorkStatisticsSession>
getActiveCompileWorkStatisticsSession() {
  return activeCompileWorkStatisticsSession;
}

class ScopedCompileWorkStatisticsActivation {
public:
  explicit ScopedCompileWorkStatisticsActivation(
      std::shared_ptr<CompileWorkStatisticsSession> session)
      : previous(std::move(activeCompileWorkStatisticsSession)) {
    activeCompileWorkStatisticsSession = std::move(session);
  }

  ScopedCompileWorkStatisticsActivation(
      const ScopedCompileWorkStatisticsActivation &) = delete;
  ScopedCompileWorkStatisticsActivation &
  operator=(const ScopedCompileWorkStatisticsActivation &) = delete;

  ~ScopedCompileWorkStatisticsActivation() {
    activeCompileWorkStatisticsSession = std::move(previous);
  }

private:
  std::shared_ptr<CompileWorkStatisticsSession> previous;
};

inline void recordCompileWork(CompileWorkKind kind, uint64_t amount = 1) {
  if (std::shared_ptr<CompileWorkStatisticsSession> session =
          getActiveCompileWorkStatisticsSession())
    session->record(kind, amount);
}

} // namespace wafer::support

#endif // WAFER_SUPPORT_COMPILEWORKSTATISTICS_H
