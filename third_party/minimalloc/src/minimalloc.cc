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

Modified for Wafer: C++17/std-library port, checked model
validation, internal namespace, and typed status support.
*/

#include "wafer_third_party/minimalloc/minimalloc.h"

#include "minimalloc_internal.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace wafer_third_party {
namespace minimalloc {

bool Buffer::operator==(const Buffer &other) const {
  return id == other.id && lifespan == other.lifespan && size == other.size &&
         alignment == other.alignment && gaps == other.gaps &&
         offset == other.offset;
}

bool Problem::operator==(const Problem &other) const {
  return buffers == other.buffers && capacity == other.capacity;
}

bool Solution::operator==(const Solution &other) const {
  return offsets == other.offsets && height == other.height;
}

const char *ToString(SolveStatus status) {
  switch (status) {
  case SolveStatus::kFeasible:
    return "feasible";
  case SolveStatus::kProvenInfeasible:
    return "proven_infeasible";
  case SolveStatus::kResourceExhausted:
    return "resource_exhausted";
  case SolveStatus::kInvalidProblem:
    return "invalid_problem";
  case SolveStatus::kInternalError:
    return "internal_error";
  }
  return "unknown";
}

namespace internal {
namespace {

enum class PointType { kLeft, kLeftGap, kRightGap, kRight };

struct Point {
  std::size_t bufferIndex = 0;
  TimeValue timeValue = 0;
  PointType pointType = PointType::kLeft;
  std::optional<Window> window;

  bool operator<(const Point &other) const {
    if (timeValue != other.timeValue)
      return timeValue < other.timeValue;
    if (pointType != other.pointType)
      return pointType < other.pointType;
    return bufferIndex < other.bufferIndex;
  }
};

ValidationResult Invalid(std::string message) {
  return {ValidationStatus::kInvalid, std::move(message)};
}

ValidationResult Infeasible(std::string message) {
  return {ValidationStatus::kProvenInfeasible, std::move(message)};
}

bool IsValidHeuristicCharacter(char character) {
  switch (character) {
  case 'A':
  case 'C':
  case 'L':
  case 'O':
  case 'T':
  case 'U':
  case 'W':
  case 'Z':
    return true;
  default:
    return false;
  }
}

} // namespace

bool CheckedAdd(std::int64_t lhs, std::int64_t rhs, std::int64_t *result) {
  if (rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs)
    return false;
  if (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs)
    return false;
  *result = lhs + rhs;
  return true;
}

bool CheckedSubtract(std::int64_t lhs, std::int64_t rhs, std::int64_t *result) {
  if (rhs > 0 && lhs < std::numeric_limits<std::int64_t>::min() + rhs)
    return false;
  if (rhs < 0 && lhs > std::numeric_limits<std::int64_t>::max() + rhs)
    return false;
  *result = lhs - rhs;
  return true;
}

bool CheckedMultiply(std::int64_t lhs, std::int64_t rhs, std::int64_t *result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<std::int64_t>::max() / lhs)
    return false;
  *result = lhs * rhs;
  return true;
}

ValidationResult Validate(const Problem &problem, const SolveOptions &options) {
  if (problem.capacity < 0)
    return Invalid("capacity must be non-negative");
  if (problem.buffers.size() >
      static_cast<std::size_t>(std::numeric_limits<BufferIndex>::max()))
    return Invalid("buffer count exceeds the supported index range");
  if (options.preorderingHeuristics.empty())
    return Invalid("at least one preordering heuristic is required");
  for (const std::string &heuristic : options.preorderingHeuristics) {
    if (heuristic.empty())
      return Invalid("preordering heuristic must not be empty");
    for (char character : heuristic) {
      if (!IsValidHeuristicCharacter(character))
        return Invalid("preordering heuristic contains an unknown key");
    }
  }

  for (std::size_t index = 0; index < problem.buffers.size(); ++index) {
    const Buffer &buffer = problem.buffers[index];
    const std::string prefix = "buffer " + std::to_string(index) + ": ";
    if (buffer.size <= 0)
      return Invalid(prefix + "size must be positive");
    if (buffer.alignment <= 0)
      return Invalid(prefix + "alignment must be positive");
    if (buffer.lifespan.lower >= buffer.lifespan.upper)
      return Invalid(prefix + "lifespan must be non-empty and half-open");

    std::int64_t duration = 0;
    if (!CheckedSubtract(buffer.lifespan.upper, buffer.lifespan.lower,
                         &duration) ||
        duration <= 0)
      return Invalid(prefix + "lifespan width overflows");
    // Area is only a preordering weight. Its exact product is not part of the
    // fixed-capacity legality contract, so overflow is handled by saturation
    // in BufferArea() rather than rejecting an otherwise feasible problem.

    TimeValue previousUpper = buffer.lifespan.lower;
    for (const Gap &gap : buffer.gaps) {
      if (gap.lifespan.lower >= gap.lifespan.upper)
        return Invalid(prefix + "gap must be non-empty and half-open");
      // The retained sweeper represents the outer endpoints as active. Keeping
      // gaps strictly inside makes that representation unambiguous.
      if (gap.lifespan.lower <= buffer.lifespan.lower ||
          gap.lifespan.upper >= buffer.lifespan.upper)
        return Invalid(prefix + "gap must be strictly inside the lifespan");
      if (gap.lifespan.lower <= previousUpper)
        return Invalid(prefix + "gaps must be sorted and non-adjacent");
      previousUpper = gap.lifespan.upper;
      if (gap.window) {
        if (gap.window->lower < 0 || gap.window->lower >= gap.window->upper ||
            gap.window->upper > buffer.size)
          return Invalid(prefix + "gap window must be a non-empty sub-window");
      }
    }

    if (buffer.offset) {
      if (*buffer.offset < 0)
        return Invalid(prefix + "fixed offset must be non-negative");
      if (*buffer.offset % buffer.alignment != 0)
        return Invalid(prefix + "fixed offset violates alignment");
    }

    if (buffer.size > problem.capacity)
      return Infeasible(prefix + "size exceeds capacity");
    if (buffer.offset && *buffer.offset > problem.capacity - buffer.size)
      return Infeasible(prefix + "fixed placement exceeds capacity");
  }
  return {ValidationStatus::kValid, {}};
}

Area BufferArea(const Buffer &buffer) {
  std::int64_t duration = 0;
  std::int64_t area = 0;
  // Validate() establishes the duration subtraction before the solver is
  // entered. Area is heuristic-only, so a product overflow ranks as the
  // largest possible area instead of changing feasibility.
  const bool durationOk =
      CheckedSubtract(buffer.lifespan.upper, buffer.lifespan.lower, &duration);
  const bool areaOk = CheckedMultiply(buffer.size, duration, &area);
  return durationOk && areaOk ? area : std::numeric_limits<Area>::max();
}

std::optional<std::int64_t> EffectiveSize(const Buffer &lower,
                                          const Buffer &upper) {
  if (lower.lifespan.upper <= upper.lifespan.lower ||
      upper.lifespan.upper <= lower.lifespan.lower)
    return std::nullopt;

  const Window lowerWindow = {0, lower.size};
  const Window upperWindow = {0, upper.size};
  std::vector<Point> points = {
      {0, lower.lifespan.lower, PointType::kLeft, lowerWindow},
      {0, lower.lifespan.upper, PointType::kRight, std::nullopt},
      {1, upper.lifespan.lower, PointType::kLeft, upperWindow},
      {1, upper.lifespan.upper, PointType::kRight, std::nullopt}};
  for (const Gap &gap : lower.gaps) {
    points.push_back({0, gap.lifespan.lower, PointType::kRightGap, gap.window});
    points.push_back({0, gap.lifespan.upper, PointType::kLeftGap, lowerWindow});
  }
  for (const Gap &gap : upper.gaps) {
    points.push_back({1, gap.lifespan.lower, PointType::kRightGap, gap.window});
    points.push_back({1, gap.lifespan.upper, PointType::kLeftGap, upperWindow});
  }
  std::sort(points.begin(), points.end());

  std::array<std::optional<Window>, 2> windows;
  std::optional<std::int64_t> effectiveSize;
  std::optional<TimeValue> lastTime;
  for (const Point &point : points) {
    if (lastTime && point.timeValue > *lastTime && windows[0] && windows[1]) {
      const std::int64_t difference = windows[0]->upper - windows[1]->lower;
      if (!effectiveSize || *effectiveSize < difference)
        effectiveSize = difference;
    }
    lastTime = point.timeValue;
    windows[point.bufferIndex] = point.window;
  }
  return effectiveSize;
}

} // namespace internal
} // namespace minimalloc
} // namespace wafer_third_party
