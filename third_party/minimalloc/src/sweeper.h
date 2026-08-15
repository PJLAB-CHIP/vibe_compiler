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

Modified for Wafer: C++17/std-library port and internal namespace.
*/

#ifndef WAFER_THIRD_PARTY_MINIMALLOC_SWEEPER_H
#define WAFER_THIRD_PARTY_MINIMALLOC_SWEEPER_H

#include "wafer_third_party/minimalloc/minimalloc.h"

#include <cstdint>
#include <set>
#include <vector>

namespace wafer_third_party {
namespace minimalloc {
namespace internal {

using SectionIndex = std::int64_t;
using CutCount = std::int64_t;
using SectionRange = Interval<SectionIndex>;
using Section = std::set<BufferIndex>;

struct SectionSpan {
  SectionRange sectionRange;
  Window window;
};

struct Partition {
  std::vector<BufferIndex> bufferIndices;
  SectionRange sectionRange;
};

struct Overlap {
  BufferIndex bufferIndex = -1;
  std::int64_t effectiveSize = 0;

  bool operator<(const Overlap &other) const {
    if (bufferIndex != other.bufferIndex)
      return bufferIndex < other.bufferIndex;
    return effectiveSize < other.effectiveSize;
  }
};

struct BufferData {
  std::vector<SectionSpan> sectionSpans;
  std::set<Overlap> overlaps;
};

struct SweepResult {
  std::vector<Section> sections;
  std::vector<Partition> partitions;
  std::vector<BufferData> bufferData;

  std::vector<CutCount> calculateCuts() const;
};

enum class SweepPointType { kRight, kLeft };

struct SweepPoint {
  BufferIndex bufferIndex = -1;
  TimeValue timeValue = 0;
  SweepPointType pointType = SweepPointType::kRight;
  Window window;
  bool endpoint = false;

  bool operator<(const SweepPoint &other) const;
};

SweepResult Sweep(const Problem &problem);

} // namespace internal
} // namespace minimalloc
} // namespace wafer_third_party

#endif // WAFER_THIRD_PARTY_MINIMALLOC_SWEEPER_H
