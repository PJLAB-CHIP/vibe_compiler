//===- OptimizationConfig.h - Compiler optimization policy -----*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
#define WAFER_SUPPORT_OPTIMIZATIONCONFIG_H

#include <cstdint>
#include <optional>

namespace wafer {

/// Public deterministic work limits for the search optimization policy.
/// Width bounds simultaneously retained expandable branches, not total visited
/// structures. Trials bounds actual candidate attempts, including failures.
/// Neither value participates in IR legality or cost.
struct SearchLimits {
  uint64_t width = 8;
  uint64_t trials = 42;

  friend constexpr bool operator==(SearchLimits lhs, SearchLimits rhs) {
    return lhs.width == rhs.width && lhs.trials == rhs.trials;
  }
  friend constexpr bool operator!=(SearchLimits lhs, SearchLimits rhs) {
    return !(lhs == rhs);
  }
};

/// Invocation-local public optimization policy. Search selects the
/// compiler-owned planning controller and its deterministic work limits; none
/// selects the mandatory fully-gated baseline. Internal Region and Temporal
/// scheduling allowances are not public axes.
class OptimizationConfig {
public:
  static constexpr OptimizationConfig search(SearchLimits limits = {}) {
    return OptimizationConfig(Policy::Search, limits);
  }
  static constexpr OptimizationConfig none() {
    return OptimizationConfig(Policy::None, {});
  }

  constexpr bool isSearch() const { return policy == Policy::Search; }
  constexpr bool isNone() const { return policy == Policy::None; }
  constexpr std::optional<SearchLimits> getSearchLimits() const {
    return isSearch() ? std::optional<SearchLimits>(limits) : std::nullopt;
  }

  friend constexpr bool operator==(OptimizationConfig lhs,
                                   OptimizationConfig rhs) {
    return lhs.policy == rhs.policy &&
           (lhs.isNone() || lhs.limits == rhs.limits);
  }
  friend constexpr bool operator!=(OptimizationConfig lhs,
                                   OptimizationConfig rhs) {
    return !(lhs == rhs);
  }

private:
  enum class Policy : uint8_t { Search, None };

  explicit constexpr OptimizationConfig(Policy policy, SearchLimits limits)
      : policy(policy), limits(limits) {}

  Policy policy;
  SearchLimits limits;
};

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
