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

Modified by the Wafer project: C++17/std-library port, typed result API,
fixed-capacity-only boundary, strict validation, and deterministic global fuel.
*/

#ifndef WAFER_THIRD_PARTY_MINIMALLOC_MINIMALLOC_H
#define WAFER_THIRD_PARTY_MINIMALLOC_MINIMALLOC_H

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace wafer_third_party {
namespace minimalloc {

template <typename T> struct Interval {
  T lower = 0;
  T upper = 0;

  bool operator==(const Interval<T> &other) const {
    return lower == other.lower && upper == other.upper;
  }
  bool operator<(const Interval<T> &other) const {
    if (lower != other.lower)
      return lower < other.lower;
    return upper < other.upper;
  }
};

using BufferIndex = std::int64_t;
using Capacity = std::int64_t;
using Offset = std::int64_t;
using TimeValue = std::int64_t;
using Area = std::int64_t;
using Lifespan = Interval<TimeValue>;
using Window = Interval<Offset>;

struct Gap {
  // This curated API accepts sorted, non-adjacent, non-empty gaps strictly
  // inside the owning lifespan. Endpoint-touching and adjacent upstream forms
  // are intentionally outside Wafer's retained adapter domain.
  Lifespan lifespan;
  std::optional<Window> window;

  bool operator==(const Gap &other) const {
    return lifespan == other.lifespan && window == other.window;
  }
};

struct Buffer {
  std::string id;
  Lifespan lifespan;
  // Strictly positive. Wafer removes zero-byte demands before this boundary.
  std::int64_t size = 0;
  std::int64_t alignment = 1;
  std::vector<Gap> gaps;
  std::optional<Offset> offset;

  bool operator==(const Buffer &other) const;
};

struct Problem {
  std::vector<Buffer> buffers;
  Capacity capacity = 0;

  bool operator==(const Problem &other) const;
};

struct Solution {
  std::vector<Offset> offsets;
  Offset height = 0;

  bool operator==(const Solution &other) const;
};

enum class SolveStatus {
  kFeasible,
  kProvenInfeasible,
  kResourceExhausted,
  kInvalidProblem,
  kInternalError,
};

struct SolveOptions {
  // One unit is one visited recursive search node. The budget is global across
  // partitions, heuristics, and round-robin attempts. UINT64_MAX is unlimited.
  std::uint64_t nodeBudget = std::numeric_limits<std::uint64_t>::max();

  bool canonicalOnly = true;
  bool sectionInference = true;
  bool dynamicOrdering = true;
  bool checkDominance = true;
  bool unallocatedFloor = true;
  bool staticPreordering = true;
  bool dynamicDecomposition = true;
  bool monotonicFloor = true;
  bool hatlessPruning = true;
  std::vector<std::string> preorderingHeuristics = {"WAT", "TAW", "TWA"};
};

struct SolveResult {
  SolveStatus status = SolveStatus::kInternalError;
  Solution solution;
  std::uint64_t searchNodes = 0;
  std::uint64_t backtracks = 0;
  std::string message;

  bool feasible() const { return status == SolveStatus::kFeasible; }
};

// Solves one fixed-capacity static packing problem. This function performs no
// wall-clock checks and never intentionally throws; callers control only a
// deterministic search-node budget.
SolveResult Solve(const Problem &problem,
                  const SolveOptions &options = SolveOptions());

const char *ToString(SolveStatus status);

} // namespace minimalloc
} // namespace wafer_third_party

#endif // WAFER_THIRD_PARTY_MINIMALLOC_MINIMALLOC_H
