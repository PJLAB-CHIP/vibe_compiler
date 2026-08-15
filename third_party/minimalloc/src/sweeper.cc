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

Modified for Wafer: C++17/std-library port, deterministic ordered
sets, and internal namespace.
*/

#include "sweeper.h"

#include "minimalloc_internal.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <set>
#include <vector>

namespace wafer_third_party {
namespace minimalloc {
namespace internal {
namespace {

bool Contains(const std::set<TimeValue> &values, TimeValue value) {
  return values.find(value) != values.end();
}

std::vector<SweepPoint> CreatePoints(const Problem &problem) {
  std::vector<SweepPoint> allPoints;
  allPoints.reserve(problem.buffers.size() * 2);
  for (std::size_t rawIndex = 0; rawIndex < problem.buffers.size();
       ++rawIndex) {
    const BufferIndex bufferIndex = static_cast<BufferIndex>(rawIndex);
    const Buffer &buffer = problem.buffers[rawIndex];
    const Lifespan &lifespan = buffer.lifespan;
    const Window fullWindow = {0, buffer.size};
    std::deque<SweepPoint> points;
    std::set<TimeValue> leftTimes;
    std::set<TimeValue> rightTimes;

    // A windowed gap means the buffer retains only that spatial sub-window.
    for (const Gap &gap : buffer.gaps) {
      if (!gap.window)
        continue;
      points.push_back({bufferIndex, gap.lifespan.lower, SweepPointType::kLeft,
                        *gap.window, false});
      points.push_back({bufferIndex, gap.lifespan.upper, SweepPointType::kRight,
                        *gap.window, false});
      leftTimes.insert(gap.lifespan.lower);
      rightTimes.insert(gap.lifespan.upper);
    }

    if (points.empty() || points.front().timeValue != lifespan.lower) {
      points.push_front({bufferIndex, lifespan.lower, SweepPointType::kLeft,
                         fullWindow, false});
    }
    if (points.empty() || points.back().timeValue != lifespan.upper) {
      points.push_back({bufferIndex, lifespan.upper, SweepPointType::kRight,
                        fullWindow, false});
    }
    points.front().endpoint = true;
    points.back().endpoint = true;
    rightTimes.insert(lifespan.lower);
    leftTimes.insert(lifespan.upper);

    for (const Gap &gap : buffer.gaps) {
      if (gap.window)
        continue;
      if (!Contains(rightTimes, gap.lifespan.lower)) {
        points.push_back({bufferIndex, gap.lifespan.lower,
                          SweepPointType::kRight, fullWindow, false});
        rightTimes.insert(gap.lifespan.lower);
      }
      if (!Contains(leftTimes, gap.lifespan.upper)) {
        points.push_back({bufferIndex, gap.lifespan.upper,
                          SweepPointType::kLeft, fullWindow, false});
        leftTimes.insert(gap.lifespan.upper);
      }
      leftTimes.insert(gap.lifespan.lower);
      rightTimes.insert(gap.lifespan.upper);
    }

    // Windowed gaps can leave implicit full-window intervals around their
    // endpoints. Materialize those transitions when not already represented.
    for (const Gap &gap : buffer.gaps) {
      if (!Contains(rightTimes, gap.lifespan.lower)) {
        points.push_back({bufferIndex, gap.lifespan.lower,
                          SweepPointType::kRight, fullWindow, false});
      }
      if (!Contains(leftTimes, gap.lifespan.upper)) {
        points.push_back({bufferIndex, gap.lifespan.upper,
                          SweepPointType::kLeft, fullWindow, false});
      }
    }

    allPoints.insert(allPoints.end(), points.begin(), points.end());
  }
  std::sort(allPoints.begin(), allPoints.end());
  return allPoints;
}

} // namespace

bool SweepPoint::operator<(const SweepPoint &other) const {
  if (timeValue != other.timeValue)
    return timeValue < other.timeValue;
  if (pointType != other.pointType)
    return pointType < other.pointType;
  return bufferIndex < other.bufferIndex;
}

SweepResult Sweep(const Problem &problem) {
  SweepResult result;
  const std::vector<SweepPoint> points = CreatePoints(problem);
  Section active;
  Section alive;
  bool haveSectionTime = false;
  TimeValue lastSectionTime = 0;
  SectionIndex lastSectionIndex = 0;

  result.bufferData.resize(problem.buffers.size());
  std::vector<SectionIndex> bufferSectionStart(problem.buffers.size(), -1);
  for (const SweepPoint &point : points) {
    const BufferIndex bufferIndex = point.bufferIndex;
    const Buffer &buffer =
        problem.buffers[static_cast<std::size_t>(bufferIndex)];
    if (!haveSectionTime) {
      haveSectionTime = true;
      lastSectionTime = point.timeValue;
    }

    if (point.pointType == SweepPointType::kRight) {
      if (lastSectionTime < point.timeValue) {
        lastSectionTime = point.timeValue;
        result.sections.push_back(active);
      }
      active.erase(bufferIndex);
      if (point.endpoint)
        alive.erase(bufferIndex);
      const SectionRange sectionRange = {
          bufferSectionStart[static_cast<std::size_t>(bufferIndex)],
          static_cast<SectionIndex>(result.sections.size())};
      result.bufferData[static_cast<std::size_t>(bufferIndex)]
          .sectionSpans.push_back({sectionRange, point.window});
      if (alive.empty()) {
        result.partitions.back().sectionRange = {
            lastSectionIndex,
            static_cast<SectionIndex>(result.sections.size())};
        lastSectionIndex = static_cast<SectionIndex>(result.sections.size());
      }
    }

    if (point.pointType == SweepPointType::kLeft) {
      if (alive.empty())
        result.partitions.push_back(Partition());
      if (point.endpoint) {
        result.partitions.back().bufferIndices.push_back(bufferIndex);
        for (BufferIndex aliveIndex : alive) {
          const Buffer &aliveBuffer =
              problem.buffers[static_cast<std::size_t>(aliveIndex)];
          if (const auto size = EffectiveSize(aliveBuffer, buffer)) {
            result.bufferData[static_cast<std::size_t>(aliveIndex)]
                .overlaps.insert({bufferIndex, *size});
          }
          if (const auto size = EffectiveSize(buffer, aliveBuffer)) {
            result.bufferData[static_cast<std::size_t>(bufferIndex)]
                .overlaps.insert({aliveIndex, *size});
          }
        }
      }
      active.insert(bufferIndex);
      if (point.endpoint)
        alive.insert(bufferIndex);
      bufferSectionStart[static_cast<std::size_t>(bufferIndex)] =
          static_cast<SectionIndex>(result.sections.size());
    }
  }
  return result;
}

std::vector<CutCount> SweepResult::calculateCuts() const {
  if (sections.empty())
    return {};
  std::vector<CutCount> cuts(sections.size() - 1, 0);
  for (const BufferData &data : bufferData) {
    if (data.sectionSpans.empty())
      continue;
    const SectionIndex lower = data.sectionSpans.front().sectionRange.lower;
    const SectionIndex upper = data.sectionSpans.back().sectionRange.upper;
    for (SectionIndex section = lower; section + 1 < upper; ++section)
      ++cuts[static_cast<std::size_t>(section)];
  }
  return cuts;
}

} // namespace internal
} // namespace minimalloc
} // namespace wafer_third_party
