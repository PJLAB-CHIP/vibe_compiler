//===- PlanningProfile.h - Optional physical-search profile ---*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGPROFILE_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGPROFILE_H

#include "Wafer/Planning/PhysicalDataflow/PlanningMemo.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace wafer::compiler::detail {

struct PlanningMemoProfile {
  uint64_t lookups = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t entries = 0;
  uint64_t peakEntries = 0;
};

struct PlanningProfileStatistics {
  std::array<PlanningMemoProfile, static_cast<size_t>(PlanningMemoKind::Count)>
      memos{};
  uint64_t peakFrontierDepth = 0;
  uint64_t candidateActualizations = 0;
  uint64_t acceptedCandidates = 0;
  uint64_t winnerHandoffs = 0;
  std::optional<uint64_t> timeToFirstAcceptedMilliseconds;
};

/// Invocation-local observation sink. It is constructed only for an explicit
/// profiling compile and is never consulted by planning or candidate control.
class PlanningProfileSink {
public:
  void beginSearch();
  void recordMemoLookup(PlanningMemoKind kind, bool hit);
  void recordMemoEntries(PlanningMemoKind kind, size_t entries);
  void observeFrontierDepth(size_t depth);
  void recordCandidateActualization(uint64_t count, bool accepted);
  void recordWinnerHandoff();

  const PlanningProfileStatistics &getStatistics() const { return statistics; }

private:
  PlanningProfileStatistics statistics;
  std::optional<std::chrono::steady_clock::time_point> searchStart;
};

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_PLANNINGPROFILE_H
