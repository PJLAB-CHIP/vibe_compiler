//===- ExactPBQPSolver.h - Exact finite PBQP ----------------*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_EXACTPBQPSOLVER_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_EXACTPBQPSOLVER_H

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace wafer::compiler::detail {

using ExactPBQPCost = uint64_t;
inline constexpr ExactPBQPCost kExactPBQPInfinity =
    std::numeric_limits<ExactPBQPCost>::max();

struct ExactPBQPVariable {
  std::vector<ExactPBQPCost> unaryCosts;
};

struct ExactPBQPBinaryFactor {
  uint32_t lhs = 0;
  uint32_t rhs = 0;
  uint32_t lhsStates = 0;
  uint32_t rhsStates = 0;
  std::vector<ExactPBQPCost> costs;
};

struct ExactPBQPProblem {
  std::vector<ExactPBQPVariable> variables;
  std::vector<ExactPBQPBinaryFactor> factors;
};

enum class ExactPBQPStatus : uint8_t {
  Optimal,
  NoSolution,
  Indeterminate,
  BrokenContract,
};

struct ExactPBQPResult {
  ExactPBQPStatus status = ExactPBQPStatus::BrokenContract;
  std::vector<uint32_t> assignment;
  std::optional<ExactPBQPCost> cost;
  ExactPBQPCost lowerBound = 0;
  uint64_t work = 0;
};

/// Applies exact degree-0/1/2 PBQP reductions, then exhaustively solves the
/// residual core with stable semantic tie-breaking. `workLimit` is a compiler
/// resource bound: exhaustion returns Indeterminate and never NoSolution.
ExactPBQPResult solveExactPBQP(const ExactPBQPProblem &problem,
                               uint64_t workLimit);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_EXACTPBQPSOLVER_H
