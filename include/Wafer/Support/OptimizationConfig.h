//===- OptimizationConfig.h - Compiler optimization policy -----*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
#define WAFER_SUPPORT_OPTIMIZATIONCONFIG_H

#include <cstdint>

namespace wafer {

/// Invocation-local public optimization policy. Search selects the
/// compiler-owned planning controller; none selects the mandatory fully-gated
/// baseline. Individual mechanisms and work allowances are not public axes.
class OptimizationConfig {
public:
  static constexpr OptimizationConfig search() {
    return OptimizationConfig(Policy::Search);
  }
  static constexpr OptimizationConfig none() {
    return OptimizationConfig(Policy::None);
  }

  constexpr bool isSearch() const { return policy == Policy::Search; }
  constexpr bool isNone() const { return policy == Policy::None; }

  friend constexpr bool operator==(OptimizationConfig lhs,
                                   OptimizationConfig rhs) {
    return lhs.policy == rhs.policy;
  }
  friend constexpr bool operator!=(OptimizationConfig lhs,
                                   OptimizationConfig rhs) {
    return !(lhs == rhs);
  }

private:
  enum class Policy : uint8_t { Search, None };

  explicit constexpr OptimizationConfig(Policy policy) : policy(policy) {}

  Policy policy;
};

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
