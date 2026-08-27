//===- PlanningProfile.cpp - Optional physical-search profile ---------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProfile.h"

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

void PlanningProfileSink::recordWinnerHandoff() {
  ++statistics.winnerHandoffs;
}

void printPlanningProfile(llvm::raw_ostream &output,
                          const PlanningProfileSink &profile) {
  const PlanningProfileStatistics &statistics = profile.getStatistics();
  output << "wafer-compile: planning-profile"
         << " peak_frontier_depth=" << statistics.peakFrontierDepth
         << " candidate_actualizations="
         << statistics.candidateActualizations
         << " accepted_candidates=" << statistics.acceptedCandidates
         << " winner_handoffs=" << statistics.winnerHandoffs
         << " time_to_first_accepted_ms=";
  if (statistics.timeToFirstAcceptedMilliseconds)
    output << *statistics.timeToFirstAcceptedMilliseconds;
  else
    output << "unknown";
  output << '\n';
  for (unsigned index = 0;
       index < static_cast<unsigned>(PlanningMemoKind::Count); ++index) {
    PlanningMemoKind kind = static_cast<PlanningMemoKind>(index);
    const PlanningMemoProfile &memo = statistics.memos[index];
    output << "wafer-compile: planning-profile-memo"
           << " kind=" << stringifyPlanningMemoKind(kind)
           << " lookups=" << memo.lookups << " hits=" << memo.hits
           << " misses=" << memo.misses << " entries=" << memo.entries
           << " peak_entries=" << memo.peakEntries << '\n';
  }
}

} // namespace wafer::compiler::detail
