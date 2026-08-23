//===- RepresentationPBQPSolver.h - Exact finite PBQP ------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONPBQPSOLVER_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONPBQPSOLVER_H

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace wafer::compiler::detail {

using RepresentationPBQPCost = uint64_t;
inline constexpr RepresentationPBQPCost kRepresentationPBQPInfinity =
    std::numeric_limits<RepresentationPBQPCost>::max();

struct RepresentationPBQPVariable {
  std::vector<RepresentationPBQPCost> unaryCosts;
};

struct RepresentationPBQPBinaryFactor {
  uint32_t lhs = 0;
  uint32_t rhs = 0;
  uint32_t lhsStates = 0;
  uint32_t rhsStates = 0;
  std::vector<RepresentationPBQPCost> costs;
};

struct RepresentationPBQPProblem {
  std::vector<RepresentationPBQPVariable> variables;
  std::vector<RepresentationPBQPBinaryFactor> factors;
};

enum class RepresentationPBQPStatus : uint8_t {
  Optimal,
  NoSolution,
  Indeterminate,
  BrokenContract,
};

struct RepresentationPBQPResult {
  RepresentationPBQPStatus status = RepresentationPBQPStatus::BrokenContract;
  std::vector<uint32_t> assignment;
  std::optional<RepresentationPBQPCost> cost;
  RepresentationPBQPCost lowerBound = 0;
  uint64_t work = 0;
};

/// Applies exact degree-0/1/2 PBQP reductions, then exhaustively solves the
/// residual core with stable semantic tie-breaking. `workLimit` is a compiler
/// resource bound: exhaustion returns Indeterminate and never NoSolution.
RepresentationPBQPResult
solveRepresentationPBQP(const RepresentationPBQPProblem &problem,
                        uint64_t workLimit);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_REPRESENTATIONPBQPSOLVER_H
