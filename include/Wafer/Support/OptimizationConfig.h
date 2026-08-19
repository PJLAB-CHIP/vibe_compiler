//===- OptimizationConfig.h - Compiler optimization policy -----*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
#define WAFER_SUPPORT_OPTIMIZATIONCONFIG_H

#include <cstdint>

namespace wafer {

/// Invocation-local public optimization policy. Search exposes the full
/// compiler-owned candidate domain; none selects the mandatory fully-gated
/// baseline. Individual internal mechanisms are not user-configurable axes.
class OptimizationConfig {
public:
  /// Fresh representative profiles show that one actual candidate already
  /// adds substantial compile work. Keep the production default explicit and
  /// let callers request a different invocation-local budget.
  static constexpr uint64_t kDefaultSearchCandidateEvaluations = 1;

  static constexpr OptimizationConfig
  search(uint64_t maximumCandidateEvaluations =
             kDefaultSearchCandidateEvaluations) {
    return OptimizationConfig(Policy::Search, maximumCandidateEvaluations);
  }
  static constexpr OptimizationConfig none() {
    return OptimizationConfig(Policy::None, 0);
  }

  constexpr bool isSearch() const { return policy == Policy::Search; }
  constexpr bool isNone() const { return policy == Policy::None; }
  constexpr uint64_t getMaximumSearchCandidateEvaluations() const {
    return maximumSearchCandidateEvaluations;
  }

  friend constexpr bool operator==(OptimizationConfig lhs,
                                   OptimizationConfig rhs) {
    return lhs.policy == rhs.policy &&
           lhs.maximumSearchCandidateEvaluations ==
               rhs.maximumSearchCandidateEvaluations;
  }
  friend constexpr bool operator!=(OptimizationConfig lhs,
                                   OptimizationConfig rhs) {
    return !(lhs == rhs);
  }

private:
  enum class Policy : uint8_t { Search, None };

  explicit constexpr OptimizationConfig(Policy policy,
                                        uint64_t maximumSearchEvaluations)
      : policy(policy),
        maximumSearchCandidateEvaluations(maximumSearchEvaluations) {}

  Policy policy;
  uint64_t maximumSearchCandidateEvaluations;
};

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
