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

Modified by the Wafer project: dependency-free tests for the curated API.
*/

#include "wafer_third_party/minimalloc/minimalloc.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mm = wafer_third_party::minimalloc;

namespace {

int failures = 0;

void Expect(bool condition, const char *message) {
  if (condition)
    return;
  std::cerr << "FAIL: " << message << '\n';
  ++failures;
}

mm::Buffer MakeBuffer(std::string id, std::int64_t lower, std::int64_t upper,
                      std::int64_t size, std::int64_t alignment = 1) {
  mm::Buffer buffer;
  buffer.id = std::move(id);
  buffer.lifespan = {lower, upper};
  buffer.size = size;
  buffer.alignment = alignment;
  return buffer;
}

bool SpatiallyDisjoint(mm::Offset lhsOffset, std::int64_t lhsSize,
                       mm::Offset rhsOffset, std::int64_t rhsSize) {
  return lhsOffset + lhsSize <= rhsOffset || rhsOffset + rhsSize <= lhsOffset;
}

bool BruteForcePlace(const mm::Problem &problem,
                     const std::vector<std::set<std::size_t>> &conflicts,
                     std::size_t index, std::vector<mm::Offset> &offsets) {
  if (index == problem.buffers.size())
    return true;
  const mm::Buffer &buffer = problem.buffers[index];
  for (mm::Offset offset = 0; offset <= problem.capacity - buffer.size;
       offset += buffer.alignment) {
    bool legal = true;
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (conflicts[index].find(prior) == conflicts[index].end())
        continue;
      if (!SpatiallyDisjoint(offset, buffer.size, offsets[prior],
                             problem.buffers[prior].size)) {
        legal = false;
        break;
      }
    }
    if (!legal)
      continue;
    offsets[index] = offset;
    if (BruteForcePlace(problem, conflicts, index + 1, offsets))
      return true;
  }
  return false;
}

bool BruteForceFeasible(const mm::Problem &problem,
                        const std::vector<std::set<std::size_t>> &conflicts) {
  std::vector<mm::Offset> offsets(problem.buffers.size(), 0);
  return BruteForcePlace(problem, conflicts, 0, offsets);
}

std::vector<std::set<std::size_t>>
IntervalConflicts(const mm::Problem &problem) {
  std::vector<std::set<std::size_t>> conflicts(problem.buffers.size());
  for (std::size_t lhs = 0; lhs < problem.buffers.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < problem.buffers.size(); ++rhs) {
      const mm::Lifespan &a = problem.buffers[lhs].lifespan;
      const mm::Lifespan &b = problem.buffers[rhs].lifespan;
      if (a.lower < b.upper && b.lower < a.upper) {
        conflicts[lhs].insert(rhs);
        conflicts[rhs].insert(lhs);
      }
    }
  }
  return conflicts;
}

void TestEmptyProblem() {
  mm::SolveOptions options;
  options.nodeBudget = 0;
  const mm::SolveResult result = mm::Solve(mm::Problem(), options);
  Expect(result.status == mm::SolveStatus::kFeasible,
         "empty problem is feasible without consuming fuel");
  Expect(result.searchNodes == 0, "empty problem consumes no search nodes");
}

void TestReuseAndAlignment() {
  mm::Problem problem;
  problem.capacity = 16;
  problem.buffers = {MakeBuffer("a", 0, 2, 7, 8), MakeBuffer("b", 2, 4, 9, 8)};
  const mm::SolveResult result = mm::Solve(problem);
  Expect(result.status == mm::SolveStatus::kFeasible,
         "non-overlapping buffers reuse capacity");
  Expect(result.solution.offsets.size() == 2,
         "solution contains every buffer offset");
  if (result.solution.offsets.size() == 2) {
    Expect(result.solution.offsets[0] % 8 == 0,
           "first offset satisfies alignment");
    Expect(result.solution.offsets[1] % 8 == 0,
           "second offset satisfies alignment");
  }
}

void TestRejectsZeroSizeBuffer() {
  mm::Problem problem;
  problem.capacity = 8;
  problem.buffers = {MakeBuffer("empty", 0, 3, 0, 4),
                     MakeBuffer("payload", 0, 3, 8, 4)};
  const mm::SolveResult result = mm::Solve(problem);
  Expect(result.status == mm::SolveStatus::kInvalidProblem,
         "zero-size buffers stay outside the positive-rectangle core");
}

void TestProvenInfeasible() {
  mm::Problem problem;
  problem.capacity = 9;
  problem.buffers = {MakeBuffer("a", 0, 3, 5), MakeBuffer("b", 1, 2, 5)};
  const mm::SolveResult result = mm::Solve(problem);
  Expect(result.status == mm::SolveStatus::kProvenInfeasible,
         "complete fixed-capacity failure is distinguished from exhaustion");
}

void TestResourceExhausted() {
  mm::Problem problem;
  problem.capacity = 12;
  problem.buffers = {MakeBuffer("a", 0, 3, 4), MakeBuffer("b", 0, 3, 4)};
  mm::SolveOptions options;
  options.nodeBudget = 0;
  const mm::SolveResult result = mm::Solve(problem, options);
  Expect(result.status == mm::SolveStatus::kResourceExhausted,
         "zero deterministic fuel produces resource exhaustion");
  Expect(result.searchNodes == 0, "exhausted result reports exact node count");
}

void TestGlobalSearchBudgetAcrossPartitionsAndHeuristicRounds() {
  mm::Problem partitions;
  partitions.capacity = 1;
  partitions.buffers = {MakeBuffer("first", 0, 1, 1),
                        MakeBuffer("second", 2, 3, 1)};
  mm::SolveOptions partitionOptions;
  partitionOptions.nodeBudget = 3;
  const mm::SolveResult partitionResult =
      mm::Solve(partitions, partitionOptions);
  Expect(partitionResult.status == mm::SolveStatus::kResourceExhausted,
         "one global budget is shared across disconnected partitions");
  Expect(partitionResult.searchNodes == 3,
         "partition exhaustion reports the shared global work count");

  // This pinned fixture requires more than the first 2*N-node round across
  // WAT/TAW/TWA. A budget of 2*N+1 therefore enters the second heuristic and
  // must exhaust globally instead of resetting its work count.
  mm::Problem rounds;
  rounds.capacity = 28;
  rounds.buffers = {MakeBuffer("b0", 8, 17, 9),  MakeBuffer("b1", 10, 11, 11),
                    MakeBuffer("b2", 3, 12, 2),  MakeBuffer("b3", 4, 5, 10),
                    MakeBuffer("b4", 4, 7, 7),   MakeBuffer("b5", 10, 13, 3),
                    MakeBuffer("b6", 1, 9, 8),   MakeBuffer("b7", 12, 21, 8),
                    MakeBuffer("b8", 7, 8, 2),   MakeBuffer("b9", 12, 21, 7),
                    MakeBuffer("b10", 10, 20, 1)};
  const mm::SolveResult unlimited = mm::Solve(rounds);
  Expect(unlimited.status == mm::SolveStatus::kFeasible &&
             unlimited.searchNodes > 6 * rounds.buffers.size(),
         "round fixture actually rotates through the pinned heuristics");

  mm::SolveOptions roundOptions;
  roundOptions.nodeBudget = 2 * rounds.buffers.size() + 1;
  const mm::SolveResult roundResult = mm::Solve(rounds, roundOptions);
  Expect(roundResult.status == mm::SolveStatus::kResourceExhausted,
         "one global budget is shared across heuristic rounds");
  Expect(roundResult.searchNodes == roundOptions.nodeBudget,
         "heuristic exhaustion reports the shared global work count");
}

void TestGreedyCounterexample() {
  // This five-buffer interval graph has a valid height-12 placement but a
  // common size-first, lowest-gap first-fit order rejects it.
  mm::Problem problem;
  problem.capacity = 12;
  problem.buffers = {MakeBuffer("a", 5, 10, 1), MakeBuffer("b", 4, 5, 7),
                     MakeBuffer("c", 6, 8, 5), MakeBuffer("d", 6, 7, 3),
                     MakeBuffer("e", 4, 8, 3)};
  const mm::SolveResult result = mm::Solve(problem);
  Expect(result.status == mm::SolveStatus::kFeasible,
         "canonical search solves the first-fit counterexample");
  if (result.solution.offsets.size() == problem.buffers.size()) {
    for (std::size_t lhs = 0; lhs < problem.buffers.size(); ++lhs) {
      for (std::size_t rhs = lhs + 1; rhs < problem.buffers.size(); ++rhs) {
        const auto &a = problem.buffers[lhs];
        const auto &b = problem.buffers[rhs];
        const bool timeOverlap = a.lifespan.lower < b.lifespan.upper &&
                                 b.lifespan.lower < a.lifespan.upper;
        if (timeOverlap) {
          Expect(SpatiallyDisjoint(result.solution.offsets[lhs], a.size,
                                   result.solution.offsets[rhs], b.size),
                 "overlapping lifespans receive disjoint windows");
        }
      }
    }
  }
}

void TestDeterminism() {
  mm::Problem problem;
  problem.capacity = 32;
  problem.buffers = {MakeBuffer("a", 0, 5, 8, 4), MakeBuffer("b", 1, 4, 8, 4),
                     MakeBuffer("c", 2, 6, 8, 4)};
  const mm::SolveResult first = mm::Solve(problem);
  const mm::SolveResult second = mm::Solve(problem);
  Expect(first.status == second.status && first.solution == second.solution &&
             first.searchNodes == second.searchNodes &&
             first.backtracks == second.backtracks,
         "same problem and fuel produce identical placement and work counts");
}

void TestInvalidInputs() {
  mm::Problem zeroLife;
  zeroLife.capacity = 8;
  zeroLife.buffers = {MakeBuffer("bad", 1, 1, 4)};
  Expect(mm::Solve(zeroLife).status == mm::SolveStatus::kInvalidProblem,
         "zero lifespan is rejected without entering the sweeper");

  mm::Problem badGap;
  badGap.capacity = 8;
  mm::Buffer buffer = MakeBuffer("bad-gap", 0, 4, 4);
  buffer.gaps.push_back({{0, 2}, std::nullopt});
  badGap.buffers.push_back(std::move(buffer));
  Expect(mm::Solve(badGap).status == mm::SolveStatus::kInvalidProblem,
         "endpoint gap is rejected as ambiguous");

  mm::Problem heuristicAreaOverflow;
  heuristicAreaOverflow.capacity = std::numeric_limits<std::int64_t>::max();
  heuristicAreaOverflow.buffers = {
      MakeBuffer("large-area", 0, std::numeric_limits<std::int64_t>::max(), 2)};
  Expect(mm::Solve(heuristicAreaOverflow).status == mm::SolveStatus::kFeasible,
         "heuristic-only area overflow saturates without changing legality");

  mm::Problem lifespanWidthOverflow;
  lifespanWidthOverflow.capacity = 8;
  lifespanWidthOverflow.buffers = {
      MakeBuffer("invalid-width", std::numeric_limits<std::int64_t>::min(),
                 std::numeric_limits<std::int64_t>::max(), 1)};
  Expect(mm::Solve(lifespanWidthOverflow).status ==
             mm::SolveStatus::kInvalidProblem,
         "unrepresentable lifespan width remains a true input error");
}

void TestSmallIntervalOracle() {
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  auto next = [&state](std::uint64_t bound) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (state >> 32) % bound;
  };
  for (int caseIndex = 0; caseIndex < 250; ++caseIndex) {
    mm::Problem problem;
    problem.capacity = 4 + static_cast<std::int64_t>(next(9));
    const std::size_t count = 1 + static_cast<std::size_t>(next(6));
    for (std::size_t index = 0; index < count; ++index) {
      const std::int64_t lower = static_cast<std::int64_t>(next(7));
      const std::int64_t upper = lower + 1 + static_cast<std::int64_t>(next(4));
      const std::int64_t size = 1 + static_cast<std::int64_t>(next(6));
      const std::int64_t alignment = next(2) == 0 ? 1 : 2;
      problem.buffers.push_back(
          MakeBuffer(std::to_string(index), lower, upper, size, alignment));
    }
    const auto conflicts = IntervalConflicts(problem);
    const bool expected = BruteForceFeasible(problem, conflicts);
    const mm::SolveResult result = mm::Solve(problem);
    const bool actual = result.status == mm::SolveStatus::kFeasible;
    Expect(actual == expected,
           "curated solver agrees with exhaustive small interval oracle");
  }
}

mm::Problem
EncodeConflictGraph(std::int64_t capacity,
                    const std::vector<std::int64_t> &sizes,
                    const std::vector<std::set<std::size_t>> &conflicts) {
  const std::size_t count = sizes.size();
  std::vector<std::vector<std::size_t>> activeSlots(count);
  std::size_t nextSlot = 0;
  for (std::size_t lhs = 0; lhs < count; ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < count; ++rhs) {
      if (conflicts[lhs].find(rhs) == conflicts[lhs].end())
        continue;
      activeSlots[lhs].push_back(nextSlot);
      activeSlots[rhs].push_back(nextSlot);
      ++nextSlot;
    }
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (activeSlots[index].empty())
      activeSlots[index].push_back(nextSlot++);
  }

  mm::Problem problem;
  problem.capacity = capacity;
  for (std::size_t index = 0; index < count; ++index) {
    const std::vector<std::size_t> &slots = activeSlots[index];
    mm::Buffer buffer = MakeBuffer(
        std::to_string(index), static_cast<std::int64_t>(slots.front() * 2),
        static_cast<std::int64_t>(slots.back() * 2 + 1), sizes[index]);
    for (std::size_t slot = 1; slot < slots.size(); ++slot) {
      buffer.gaps.push_back(
          {{static_cast<std::int64_t>(slots[slot - 1] * 2 + 1),
            static_cast<std::int64_t>(slots[slot] * 2)},
           std::nullopt});
    }
    problem.buffers.push_back(std::move(buffer));
  }
  return problem;
}

void TestSmallConflictGraphOracle() {
  std::uint64_t state = 0xd1b54a32d192ed03ULL;
  auto next = [&state](std::uint64_t bound) {
    state = state * 2862933555777941757ULL + 3037000493ULL;
    return (state >> 29) % bound;
  };
  for (int caseIndex = 0; caseIndex < 200; ++caseIndex) {
    const std::size_t count = 1 + static_cast<std::size_t>(next(6));
    std::vector<std::int64_t> sizes;
    std::vector<std::set<std::size_t>> conflicts(count);
    for (std::size_t index = 0; index < count; ++index)
      sizes.push_back(1 + static_cast<std::int64_t>(next(6)));
    for (std::size_t lhs = 0; lhs < count; ++lhs) {
      for (std::size_t rhs = lhs + 1; rhs < count; ++rhs) {
        if (next(3) == 0)
          continue;
        conflicts[lhs].insert(rhs);
        conflicts[rhs].insert(lhs);
      }
    }
    const std::int64_t capacity = 4 + static_cast<std::int64_t>(next(9));
    const mm::Problem encoded = EncodeConflictGraph(capacity, sizes, conflicts);
    const bool expected = BruteForceFeasible(encoded, conflicts);
    const mm::SolveResult result = mm::Solve(encoded);
    const bool actual = result.status == mm::SolveStatus::kFeasible;
    Expect(actual == expected,
           "gap encoding agrees with exhaustive arbitrary-conflict oracle");
  }
}

} // namespace

int main() {
  TestEmptyProblem();
  TestReuseAndAlignment();
  TestRejectsZeroSizeBuffer();
  TestProvenInfeasible();
  TestResourceExhausted();
  TestGlobalSearchBudgetAcrossPartitionsAndHeuristicRounds();
  TestGreedyCounterexample();
  TestDeterminism();
  TestInvalidInputs();
  TestSmallIntervalOracle();
  TestSmallConflictGraphOracle();
  if (failures == 0)
    std::cout << "Wafer curated MiniMalloc tests passed\n";
  return failures == 0 ? 0 : 1;
}
