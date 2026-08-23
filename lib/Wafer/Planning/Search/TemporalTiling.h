//===- TemporalTiling.h - Complete iterator wave domain -*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <string>
#include <tuple>

namespace wafer::compiler::detail {

struct TemporalNodeAssignment {
  StructuredDAGNodeID node = 0;
  llvm::SmallVector<int64_t, 4> iteratorTileSizes;
  /// Permutation of exactly the iterator dimensions that produce more than
  /// one wave. One-wave dimensions are absent and create no duplicate state.
  llvm::SmallVector<uint32_t, 4> waveLoopOrder;

  friend bool operator==(const TemporalNodeAssignment &lhs,
                         const TemporalNodeAssignment &rhs) {
    return lhs.node == rhs.node &&
           lhs.iteratorTileSizes == rhs.iteratorTileSizes &&
           lhs.waveLoopOrder == rhs.waveLoopOrder;
  }
  friend bool operator<(const TemporalNodeAssignment &lhs,
                        const TemporalNodeAssignment &rhs) {
    return std::tie(lhs.node, lhs.iteratorTileSizes, lhs.waveLoopOrder) <
           std::tie(rhs.node, rhs.iteratorTileSizes, rhs.waveLoopOrder);
  }
};

class TemporalNodeDomain {
public:
  static mlir::FailureOr<TemporalNodeDomain>
  create(const StructuredDAGNode &node,
         const StructuredDAGNodePlacement &placement,
         std::string *failureReason = nullptr);

  TemporalNodeAssignment getFirstAssignment() const;
  mlir::FailureOr<std::optional<TemporalNodeAssignment>>
  getNextAssignment(const TemporalNodeAssignment &assignment) const;
  bool contains(const TemporalNodeAssignment &assignment) const;

private:
  TemporalNodeDomain(StructuredDAGNodeID node,
                     llvm::SmallVector<int64_t, 4> localExtents)
      : node(node), localExtents(std::move(localExtents)) {}

  StructuredDAGNodeID node;
  llvm::SmallVector<int64_t, 4> localExtents;
};

struct CardTemporalAssignment {
  llvm::SmallVector<TemporalNodeAssignment, 16> nodes;

  friend bool operator==(const CardTemporalAssignment &lhs,
                         const CardTemporalAssignment &rhs) {
    return lhs.nodes == rhs.nodes;
  }
};

class CardTemporalDomain {
public:
  static mlir::FailureOr<CardTemporalDomain>
  create(const StructuredDAGAnalysis &dag,
         llvm::ArrayRef<StructuredDAGNodePlacement> placements,
         std::string *failureReason = nullptr);

  CardTemporalAssignment getFirstAssignment() const;
  mlir::FailureOr<std::optional<CardTemporalAssignment>>
  getNextAssignment(const CardTemporalAssignment &assignment) const;
  bool contains(const CardTemporalAssignment &assignment) const;

private:
  explicit CardTemporalDomain(llvm::SmallVector<TemporalNodeDomain, 16> domains)
      : domains(std::move(domains)) {}
  llvm::SmallVector<TemporalNodeDomain, 16> domains;
};

} // namespace wafer::compiler::detail
