/*
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modified by the Wafer project: C++17/std-library port, typed solve outcomes,
fixed-capacity-only search, checked arithmetic, and deterministic global fuel.
*/

#include "wafer_third_party/minimalloc/minimalloc.h"

#include "minimalloc_internal.h"
#include "sweeper.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace wafer_third_party {
namespace minimalloc {
namespace internal {
namespace {

using PreorderIndex = std::int64_t;
constexpr Offset kNoOffset = -1;

enum class SearchStatus {
  kFeasible,
  kNotFound,
  kRoundLimit,
  kBudgetExhausted,
  kInternalError,
};

struct SearchBudget {
  explicit SearchBudget(std::uint64_t limit) : limit(limit) {}

  bool consume() {
    if (consumed == limit)
      return false;
    ++consumed;
    return true;
  }

  std::uint64_t limit = 0;
  std::uint64_t consumed = 0;
};

struct SectionData {
  Offset floor = 0;
  std::int64_t total = 0;
};

struct OrderData {
  Offset offset = 0;
  PreorderIndex preorderIndex = 0;
};

struct OffsetChange {
  BufferIndex bufferIndex = -1;
  Offset minOffset = 0;
};

struct SectionChange {
  SectionIndex sectionIndex = -1;
  Offset floor = 0;
};

struct PreorderData {
  Area area = 0;
  TimeValue lower = 0;
  std::uint64_t overlaps = 0;
  std::int64_t sections = 0;
  std::int64_t size = 0;
  std::int64_t total = 0;
  TimeValue upper = 0;
  std::int64_t width = 0;
  BufferIndex bufferIndex = -1;
};

class PreorderingComparator {
public:
  explicit PreorderingComparator(std::string heuristic)
      : heuristic(std::move(heuristic)) {}

  bool operator()(const PreorderData &lhs, const PreorderData &rhs) const {
    for (char character : heuristic) {
      if (character == 'A' && lhs.area != rhs.area)
        return lhs.area > rhs.area;
      if (character == 'C' && lhs.sections != rhs.sections)
        return lhs.sections > rhs.sections;
      if (character == 'L' && lhs.lower != rhs.lower)
        return lhs.lower > rhs.lower;
      if (character == 'O' && lhs.overlaps != rhs.overlaps)
        return lhs.overlaps > rhs.overlaps;
      if (character == 'T' && lhs.total != rhs.total)
        return lhs.total > rhs.total;
      if (character == 'U' && lhs.upper != rhs.upper)
        return lhs.upper > rhs.upper;
      if (character == 'W' && lhs.width != rhs.width)
        return lhs.width > rhs.width;
      if (character == 'Z' && lhs.size != rhs.size)
        return lhs.size > rhs.size;
    }
    return lhs.bufferIndex < rhs.bufferIndex;
  }

private:
  std::string heuristic;
};

struct MinOffsetUpdate {
  std::optional<std::vector<OffsetChange>> changes;
  bool impossible = false;
};

class SolverImpl {
public:
  SolverImpl(const SolveOptions &options, const Problem &problem,
             const SweepResult &sweep, SearchBudget &budget,
             std::uint64_t &backtracks)
      : options(options), problem(problem), sweep(sweep), budget(budget),
        backtracks(backtracks) {}

  SearchStatus solve(Solution &result) {
    if (problem.buffers.empty()) {
      result = Solution();
      return SearchStatus::kFeasible;
    }
    if (sweep.bufferData.size() != problem.buffers.size() ||
        sweep.sections.empty() || sweep.partitions.empty())
      return SearchStatus::kInternalError;

    assignment.offsets.assign(problem.buffers.size(), kNoOffset);
    solution.offsets.assign(problem.buffers.size(), kNoOffset);
    minOffsets.assign(problem.buffers.size(), 0);
    sectionData.resize(sweep.sections.size());

    for (std::size_t rawIndex = 0; rawIndex < problem.buffers.size();
         ++rawIndex) {
      const BufferIndex bufferIndex = static_cast<BufferIndex>(rawIndex);
      const BufferData &data = sweep.bufferData[rawIndex];
      if (data.sectionSpans.empty())
        return SearchStatus::kInternalError;
      for (const SectionSpan &span : data.sectionSpans) {
        if (span.sectionRange.lower < 0 ||
            span.sectionRange.upper < span.sectionRange.lower ||
            span.sectionRange.upper >
                static_cast<SectionIndex>(sectionData.size()))
          return SearchStatus::kInternalError;
        const std::int64_t windowSize = span.window.upper - span.window.lower;
        for (SectionIndex section = span.sectionRange.lower;
             section < span.sectionRange.upper; ++section) {
          SectionData &entry = sectionData[static_cast<std::size_t>(section)];
          if (entry.total > problem.capacity - windowSize)
            return SearchStatus::kNotFound;
          entry.total += windowSize;
        }
      }
      if (problem.buffers[rawIndex].offset)
        minOffsets[rawIndex] = *problem.buffers[rawIndex].offset;
      (void)bufferIndex;
    }
    cuts = sweep.calculateCuts();

    SearchStatus status = SearchStatus::kInternalError;
    if (options.preorderingHeuristics.size() == 1) {
      roundNodesRemaining = std::numeric_limits<std::uint64_t>::max();
      const PreorderingComparator comparator(
          options.preorderingHeuristics.front());
      status = solvePartitions(comparator);
    } else {
      status = roundRobin();
    }
    if (status != SearchStatus::kFeasible)
      return status;
    if (!updateSolutionHeight())
      return SearchStatus::kInternalError;
    result = solution;
    return SearchStatus::kFeasible;
  }

private:
  SearchStatus roundRobin() {
    std::uint64_t nodeLimit =
        static_cast<std::uint64_t>(problem.buffers.size());
    while (true) {
      if (nodeLimit <= std::numeric_limits<std::uint64_t>::max() / 2)
        nodeLimit *= 2;
      else
        nodeLimit = std::numeric_limits<std::uint64_t>::max();

      bool everyHeuristicHitRoundLimit = true;
      for (const std::string &heuristic : options.preorderingHeuristics) {
        roundNodesRemaining = nodeLimit;
        const SearchStatus status =
            solvePartitions(PreorderingComparator(heuristic));
        if (status == SearchStatus::kFeasible ||
            status == SearchStatus::kNotFound ||
            status == SearchStatus::kBudgetExhausted ||
            status == SearchStatus::kInternalError)
          return status;
        if (status != SearchStatus::kRoundLimit)
          everyHeuristicHitRoundLimit = false;
      }
      if (!everyHeuristicHitRoundLimit)
        return SearchStatus::kInternalError;
    }
  }

  SearchStatus solvePartitions(const PreorderingComparator &comparator) {
    for (const Partition &partition : sweep.partitions) {
      const SearchStatus status = subSolve(partition, comparator);
      if (status != SearchStatus::kFeasible)
        return status;
    }
    return SearchStatus::kFeasible;
  }

  SearchStatus subSolve(const Partition &partition,
                        const PreorderingComparator &comparator) {
    if (partition.bufferIndices.empty() || partition.sectionRange.lower < 0 ||
        partition.sectionRange.upper <= partition.sectionRange.lower ||
        partition.sectionRange.upper >
            static_cast<SectionIndex>(sectionData.size()))
      return SearchStatus::kInternalError;

    std::vector<PreorderData> preordering;
    preordering.reserve(partition.bufferIndices.size());
    for (BufferIndex bufferIndex : partition.bufferIndices) {
      if (bufferIndex < 0 ||
          bufferIndex >= static_cast<BufferIndex>(problem.buffers.size()))
        return SearchStatus::kInternalError;
      const Buffer &buffer =
          problem.buffers[static_cast<std::size_t>(bufferIndex)];
      const BufferData &data =
          sweep.bufferData[static_cast<std::size_t>(bufferIndex)];
      if (data.sectionSpans.empty())
        return SearchStatus::kInternalError;

      std::int64_t total = 0;
      for (const SectionSpan &span : data.sectionSpans) {
        for (SectionIndex section = span.sectionRange.lower;
             section < span.sectionRange.upper; ++section) {
          total = std::max(
              total, sectionData[static_cast<std::size_t>(section)].total);
        }
      }
      const std::int64_t sections =
          data.sectionSpans.back().sectionRange.upper -
          data.sectionSpans.front().sectionRange.lower;
      const std::int64_t width = buffer.lifespan.upper - buffer.lifespan.lower;
      preordering.push_back({BufferArea(buffer), buffer.lifespan.lower,
                             static_cast<std::uint64_t>(data.overlaps.size()),
                             sections, buffer.size, total,
                             buffer.lifespan.upper, width, bufferIndex});
    }
    if (options.staticPreordering)
      std::sort(preordering.begin(), preordering.end(), comparator);

    std::vector<OrderData> ordering(preordering.size());
    for (std::size_t index = 0; index < ordering.size(); ++index)
      ordering[index].preorderIndex = static_cast<PreorderIndex>(index);
    return search(partition, comparator, preordering, ordering, 0, 0);
  }

  std::vector<SectionChange>
  updateSectionData(const std::set<SectionIndex> &affectedSections,
                    BufferIndex bufferIndex, bool &impossible) {
    std::vector<SectionChange> changes;
    const Offset offset =
        assignment.offsets[static_cast<std::size_t>(bufferIndex)];
    const BufferData &data =
        sweep.bufferData[static_cast<std::size_t>(bufferIndex)];
    for (const SectionSpan &span : data.sectionSpans) {
      Offset height = 0;
      if (!CheckedAdd(offset, span.window.upper, &height) ||
          height > problem.capacity) {
        impossible = true;
        height = problem.capacity;
      }
      const std::int64_t windowSize = span.window.upper - span.window.lower;
      for (SectionIndex section = span.sectionRange.lower;
           section < span.sectionRange.upper; ++section) {
        SectionData &entry = sectionData[static_cast<std::size_t>(section)];
        changes.push_back({section, entry.floor});
        entry.floor = height;
        entry.total -= windowSize;
      }
    }
    for (SectionIndex section : affectedSections) {
      Offset minimum = std::numeric_limits<Offset>::max();
      for (BufferIndex other :
           sweep.sections[static_cast<std::size_t>(section)]) {
        if (assignment.offsets[static_cast<std::size_t>(other)] == kNoOffset)
          minimum =
              std::min(minimum, minOffsets[static_cast<std::size_t>(other)]);
      }
      SectionData &entry = sectionData[static_cast<std::size_t>(section)];
      if (minimum != std::numeric_limits<Offset>::max() &&
          entry.floor < minimum) {
        changes.push_back({section, entry.floor});
        entry.floor = minimum;
      }
    }
    return changes;
  }

  void restoreSectionData(const std::vector<SectionChange> &changes,
                          BufferIndex bufferIndex) {
    for (auto change = changes.rbegin(); change != changes.rend(); ++change)
      sectionData[static_cast<std::size_t>(change->sectionIndex)].floor =
          change->floor;
    const BufferData &data =
        sweep.bufferData[static_cast<std::size_t>(bufferIndex)];
    for (const SectionSpan &span : data.sectionSpans) {
      const std::int64_t windowSize = span.window.upper - span.window.lower;
      for (SectionIndex section = span.sectionRange.lower;
           section < span.sectionRange.upper; ++section) {
        sectionData[static_cast<std::size_t>(section)].total += windowSize;
      }
    }
  }

  MinOffsetUpdate updateMinOffsets(BufferIndex bufferIndex,
                                   std::set<SectionIndex> &affectedSections) {
    bool hatless = true;
    std::vector<OffsetChange> changes;
    const Offset offset =
        assignment.offsets[static_cast<std::size_t>(bufferIndex)];
    const BufferData &data =
        sweep.bufferData[static_cast<std::size_t>(bufferIndex)];
    bool impossible = false;
    for (const Overlap &overlap : data.overlaps) {
      const BufferIndex other = overlap.bufferIndex;
      if (assignment.offsets[static_cast<std::size_t>(other)] != kNoOffset)
        continue;
      hatless = false;
      Offset height = 0;
      if (!CheckedAdd(offset, overlap.effectiveSize, &height)) {
        impossible = true;
        continue;
      }
      Offset &minimum = minOffsets[static_cast<std::size_t>(other)];
      if (minimum >= height)
        continue;
      changes.push_back({other, minimum});
      minimum = height;

      const Buffer &otherBuffer =
          problem.buffers[static_cast<std::size_t>(other)];
      const Offset remainder = minimum % otherBuffer.alignment;
      if (remainder > 0) {
        const Offset increment = otherBuffer.alignment - remainder;
        if (!CheckedAdd(minimum, increment, &minimum)) {
          impossible = true;
          continue;
        }
      }
      if (minimum > problem.capacity - otherBuffer.size)
        impossible = true;
      if (otherBuffer.offset && minimum > *otherBuffer.offset)
        impossible = true;

      if (options.unallocatedFloor) {
        const BufferData &otherData =
            sweep.bufferData[static_cast<std::size_t>(other)];
        for (const SectionSpan &span : otherData.sectionSpans) {
          for (SectionIndex section = span.sectionRange.lower;
               section < span.sectionRange.upper; ++section)
            affectedSections.insert(section);
        }
      }
    }
    MinOffsetUpdate result;
    if (!hatless)
      result.changes = std::move(changes);
    result.impossible = impossible;
    return result;
  }

  void restoreMinOffsets(const std::vector<OffsetChange> &changes) {
    for (auto change = changes.rbegin(); change != changes.rend(); ++change)
      minOffsets[static_cast<std::size_t>(change->bufferIndex)] =
          change->minOffset;
  }

  bool check(const Partition &partition, Offset offset) const {
    for (SectionIndex section = partition.sectionRange.lower;
         section < partition.sectionRange.upper; ++section) {
      Offset floor = sectionData[static_cast<std::size_t>(section)].floor;
      const std::int64_t total =
          sectionData[static_cast<std::size_t>(section)].total;
      if (options.monotonicFloor)
        floor = std::max(offset, floor);
      if (options.sectionInference && floor > problem.capacity - total)
        return false;
      if (!options.sectionInference && floor > problem.capacity)
        return false;
    }
    return true;
  }

  std::vector<OrderData>
  computeOrdering(const std::vector<PreorderData> &preordering,
                  const std::vector<OrderData> &original) const {
    std::vector<OrderData> ordering;
    ordering.reserve(original.size());
    for (const OrderData &entry : original) {
      const BufferIndex bufferIndex =
          preordering[static_cast<std::size_t>(entry.preorderIndex)]
              .bufferIndex;
      if (assignment.offsets[static_cast<std::size_t>(bufferIndex)] !=
          kNoOffset)
        continue;
      ordering.push_back({minOffsets[static_cast<std::size_t>(bufferIndex)],
                          entry.preorderIndex});
    }
    if (options.dynamicOrdering) {
      std::sort(ordering.begin(), ordering.end(),
                [](const OrderData &lhs, const OrderData &rhs) {
                  if (lhs.offset != rhs.offset)
                    return lhs.offset < rhs.offset;
                  return lhs.preorderIndex < rhs.preorderIndex;
                });
    }
    return ordering;
  }

  Offset calculateMinHeight(const std::vector<PreorderData> &preordering,
                            const std::vector<OrderData> &ordering) const {
    Offset minimum = std::numeric_limits<Offset>::max();
    for (const OrderData &entry : ordering) {
      const BufferIndex bufferIndex =
          preordering[static_cast<std::size_t>(entry.preorderIndex)]
              .bufferIndex;
      const Buffer &buffer =
          problem.buffers[static_cast<std::size_t>(bufferIndex)];
      minimum = std::min(minimum, entry.offset + buffer.size);
    }
    return minimum;
  }

  SearchStatus search(const Partition &partition,
                      const PreorderingComparator &comparator,
                      const std::vector<PreorderData> &preordering,
                      const std::vector<OrderData> &originalOrdering,
                      Offset minOffset, PreorderIndex minPreorderIndex) {
    if (roundNodesRemaining == 0)
      return SearchStatus::kRoundLimit;
    if (!budget.consume())
      return SearchStatus::kBudgetExhausted;
    if (roundNodesRemaining != std::numeric_limits<std::uint64_t>::max())
      --roundNodesRemaining;

    const std::vector<OrderData> ordering =
        computeOrdering(preordering, originalOrdering);
    if (ordering.empty()) {
      for (BufferIndex bufferIndex : partition.bufferIndices) {
        solution.offsets[static_cast<std::size_t>(bufferIndex)] =
            assignment.offsets[static_cast<std::size_t>(bufferIndex)];
      }
      return SearchStatus::kFeasible;
    }

    const Offset minHeight = calculateMinHeight(preordering, ordering);
    for (const OrderData &entry : ordering) {
      const Offset offset = entry.offset;
      const PreorderIndex preorderIndex = entry.preorderIndex;
      const BufferIndex bufferIndex =
          preordering[static_cast<std::size_t>(preorderIndex)].bufferIndex;
      const Buffer &buffer =
          problem.buffers[static_cast<std::size_t>(bufferIndex)];
      if (options.canonicalOnly &&
          (offset < minOffset ||
           (offset == minOffset && preorderIndex < minPreorderIndex)))
        continue;
      if (options.checkDominance && offset >= minHeight)
        continue;
      if (buffer.offset && offset > *buffer.offset)
        continue;
      if (offset < 0 || offset > problem.capacity - buffer.size)
        continue;

      assignment.offsets[static_cast<std::size_t>(bufferIndex)] = offset;
      std::set<SectionIndex> affectedSections;
      MinOffsetUpdate minUpdate =
          updateMinOffsets(bufferIndex, affectedSections);
      bool impossible = minUpdate.impossible;
      const std::vector<SectionChange> sectionChanges =
          updateSectionData(affectedSections, bufferIndex, impossible);

      SearchStatus status = SearchStatus::kNotFound;
      if (!impossible && check(partition, offset)) {
        status = options.dynamicDecomposition
                     ? dynamicallyDecompose(partition, comparator, preordering,
                                            ordering, offset, preorderIndex,
                                            bufferIndex)
                     : search(partition, comparator, preordering, ordering,
                              offset, preorderIndex);
      }

      restoreSectionData(sectionChanges, bufferIndex);
      if (minUpdate.changes)
        restoreMinOffsets(*minUpdate.changes);
      assignment.offsets[static_cast<std::size_t>(bufferIndex)] = kNoOffset;

      if (status != SearchStatus::kNotFound)
        return status;
      if (!minUpdate.changes && options.hatlessPruning)
        break;
    }
    if (backtracks != std::numeric_limits<std::uint64_t>::max())
      ++backtracks;
    return SearchStatus::kNotFound;
  }

  SearchStatus dynamicallyDecompose(
      const Partition &partition, const PreorderingComparator &comparator,
      const std::vector<PreorderData> &preordering,
      const std::vector<OrderData> &originalOrdering, Offset minOffset,
      PreorderIndex minPreorderIndex, BufferIndex bufferIndex) {
    solution.offsets[static_cast<std::size_t>(bufferIndex)] =
        assignment.offsets[static_cast<std::size_t>(bufferIndex)];
    const BufferData &data =
        sweep.bufferData[static_cast<std::size_t>(bufferIndex)];
    if (data.sectionSpans.empty())
      return SearchStatus::kInternalError;

    std::vector<SectionIndex> cutpoints = {partition.sectionRange.lower};
    const SectionIndex lower = data.sectionSpans.front().sectionRange.lower;
    const SectionIndex upper = data.sectionSpans.back().sectionRange.upper;
    for (SectionIndex section = lower; section + 1 < upper; ++section) {
      CutCount &cut = cuts[static_cast<std::size_t>(section)];
      if (cut <= 0)
        return SearchStatus::kInternalError;
      if (--cut == 0)
        cutpoints.push_back(section + 1);
    }

    SearchStatus status = SearchStatus::kFeasible;
    if (cutpoints.size() == 1) {
      status = search(partition, comparator, preordering, originalOrdering,
                      minOffset, minPreorderIndex);
    } else {
      cutpoints.push_back(partition.sectionRange.upper);
      for (std::size_t cutIndex = 1; cutIndex < cutpoints.size(); ++cutIndex) {
        const SectionRange range = {cutpoints[cutIndex - 1],
                                    cutpoints[cutIndex]};
        std::vector<BufferIndex> bufferIndices;
        for (BufferIndex other : partition.bufferIndices) {
          if (assignment.offsets[static_cast<std::size_t>(other)] != kNoOffset)
            continue;
          const BufferData &otherData =
              sweep.bufferData[static_cast<std::size_t>(other)];
          const SectionRange otherRange = {
              otherData.sectionSpans.front().sectionRange.lower,
              otherData.sectionSpans.back().sectionRange.upper};
          if (!(otherRange.upper <= range.lower ||
                range.upper <= otherRange.lower))
            bufferIndices.push_back(other);
        }
        if (bufferIndices.empty())
          continue;
        status = subSolve({std::move(bufferIndices), range}, comparator);
        if (status != SearchStatus::kFeasible)
          break;
      }
    }

    for (SectionIndex section = lower; section + 1 < upper; ++section)
      ++cuts[static_cast<std::size_t>(section)];
    return status;
  }

  bool updateSolutionHeight() {
    solution.height = 0;
    if (solution.offsets.size() != problem.buffers.size())
      return false;
    for (std::size_t index = 0; index < problem.buffers.size(); ++index) {
      const Offset offset = solution.offsets[index];
      const Buffer &buffer = problem.buffers[index];
      Offset height = 0;
      if (offset < 0 || !CheckedAdd(offset, buffer.size, &height) ||
          height > problem.capacity || offset % buffer.alignment != 0)
        return false;
      solution.height = std::max(solution.height, height);
    }
    return true;
  }

  const SolveOptions &options;
  const Problem &problem;
  const SweepResult &sweep;
  SearchBudget &budget;
  std::uint64_t &backtracks;

  Solution assignment;
  Solution solution;
  std::vector<Offset> minOffsets;
  std::vector<SectionData> sectionData;
  std::vector<CutCount> cuts;
  std::uint64_t roundNodesRemaining = std::numeric_limits<std::uint64_t>::max();
};

SolveResult MakeResult(SolveStatus status, const SearchBudget &budget,
                       std::uint64_t backtracks, std::string message) {
  SolveResult result;
  result.status = status;
  result.searchNodes = budget.consumed;
  result.backtracks = backtracks;
  result.message = std::move(message);
  return result;
}

} // namespace
} // namespace internal

SolveResult Solve(const Problem &problem, const SolveOptions &options) {
  internal::SearchBudget budget(options.nodeBudget);
  std::uint64_t backtracks = 0;
  const internal::ValidationResult validation =
      internal::Validate(problem, options);
  if (validation.status == internal::ValidationStatus::kInvalid) {
    return internal::MakeResult(SolveStatus::kInvalidProblem, budget,
                                backtracks, validation.message);
  }
  if (validation.status == internal::ValidationStatus::kProvenInfeasible) {
    return internal::MakeResult(SolveStatus::kProvenInfeasible, budget,
                                backtracks, validation.message);
  }
  if (problem.buffers.empty()) {
    SolveResult result = internal::MakeResult(SolveStatus::kFeasible, budget,
                                              backtracks, "empty problem");
    result.solution = Solution();
    return result;
  }

  const internal::SweepResult sweep = internal::Sweep(problem);
  Solution solution;
  internal::SolverImpl solver(options, problem, sweep, budget, backtracks);
  const internal::SearchStatus status = solver.solve(solution);
  switch (status) {
  case internal::SearchStatus::kFeasible: {
    SolveResult result = internal::MakeResult(
        SolveStatus::kFeasible, budget, backtracks, "feasible placement found");
    result.solution = std::move(solution);
    return result;
  }
  case internal::SearchStatus::kNotFound:
    return internal::MakeResult(
        SolveStatus::kProvenInfeasible, budget, backtracks,
        "fixed-capacity search completed without a placement");
  case internal::SearchStatus::kBudgetExhausted:
    return internal::MakeResult(SolveStatus::kResourceExhausted, budget,
                                backtracks,
                                "deterministic search-node budget exhausted");
  case internal::SearchStatus::kRoundLimit:
    return internal::MakeResult(SolveStatus::kInternalError, budget, backtracks,
                                "heuristic round limit escaped round-robin");
  case internal::SearchStatus::kInternalError:
    return internal::MakeResult(SolveStatus::kInternalError, budget, backtracks,
                                "invalid internal sweep or placement state");
  }
  return internal::MakeResult(SolveStatus::kInternalError, budget, backtracks,
                              "unknown solver state");
}

} // namespace minimalloc
} // namespace wafer_third_party
