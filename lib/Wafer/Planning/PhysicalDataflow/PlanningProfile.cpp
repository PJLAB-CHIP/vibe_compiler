//===- PlanningProfile.cpp - Optional physical-search profile ---------===//

#include "Wafer/Planning/PhysicalDataflow/PlanningProfile.h"

#include <algorithm>

namespace wafer::compiler::detail {

void PlanningProfileSink::beginSearch() {
  searchStart = std::chrono::steady_clock::now();
}

void PlanningProfileSink::recordMemoLookup(PlanningMemoKind kind, bool hit) {
  if (kind == PlanningMemoKind::Count)
    return;
  PlanningMemoProfile &memo = statistics.memos[static_cast<size_t>(kind)];
  ++memo.lookups;
  if (hit)
    ++memo.hits;
  else
    ++memo.misses;
}

void PlanningProfileSink::recordMemoEntries(PlanningMemoKind kind,
                                            size_t entries) {
  if (kind == PlanningMemoKind::Count)
    return;
  PlanningMemoProfile &memo = statistics.memos[static_cast<size_t>(kind)];
  memo.entries = entries;
  memo.peakEntries = std::max<uint64_t>(memo.peakEntries, entries);
}

void PlanningProfileSink::observeFrontierDepth(size_t depth) {
  statistics.peakFrontierDepth =
      std::max<uint64_t>(statistics.peakFrontierDepth, depth);
}

void PlanningProfileSink::recordCandidateActualization(uint64_t count,
                                                       bool accepted) {
  statistics.candidateActualizations += count;
  if (!accepted)
    return;
  ++statistics.acceptedCandidates;
  if (!statistics.timeToFirstAcceptedMilliseconds && searchStart) {
    statistics.timeToFirstAcceptedMilliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *searchStart)
            .count();
  }
}

void PlanningProfileSink::recordWinnerHandoff() { ++statistics.winnerHandoffs; }

} // namespace wafer::compiler::detail
