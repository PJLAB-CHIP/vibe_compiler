//===- OptimizationConfig.h - Compiler optimization policy -----*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
#define WAFER_SUPPORT_OPTIMIZATIONCONFIG_H

#include <cstdint>

namespace wafer {

/// Invocation-local public optimization policy. Production exposes the full
/// compiler-owned candidate domain; none selects the mandatory fully-gated
/// baseline. Individual internal mechanisms are not user-configurable axes.
class OptimizationConfig {
public:
  static constexpr OptimizationConfig production() {
    return OptimizationConfig(Policy::Production);
  }
  static constexpr OptimizationConfig none() {
    return OptimizationConfig(Policy::None);
  }

  constexpr bool isProduction() const { return policy == Policy::Production; }
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
  enum class Policy : uint8_t { Production, None };

  explicit constexpr OptimizationConfig(Policy policy) : policy(policy) {}

  Policy policy;
};

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
