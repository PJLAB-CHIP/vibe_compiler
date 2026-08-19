//===- ComputeImplementation.h - Structured compute choices -*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"

#include "mlir/Support/LogicalResult.h"

#include <optional>

namespace wafer::compiler::detail {

struct ComputeImplementationChoice {
  StructuredDAGNodeID node = 0;
  StructuredComputeImplementation implementation =
      StructuredComputeImplementation::Natural;

  friend bool operator==(const ComputeImplementationChoice &lhs,
                         const ComputeImplementationChoice &rhs) {
    return lhs.node == rhs.node && lhs.implementation == rhs.implementation;
  }
};

struct CardComputeImplementationAssignment {
  llvm::SmallVector<ComputeImplementationChoice, 16> nodes;

  friend bool operator==(const CardComputeImplementationAssignment &lhs,
                         const CardComputeImplementationAssignment &rhs) {
    return lhs.nodes == rhs.nodes;
  }
};

/// Complete lazy per-node implementation domain. Natural lowering is always
/// first. Reciprocal exists only for a statically exact pointwise `1/x`
/// generic and remains a sibling rather than replacing natural division.
class CardComputeImplementationDomain {
public:
  static mlir::FailureOr<CardComputeImplementationDomain>
  create(const CardProgramAnalysis &program,
         std::string *failureReason = nullptr);

  CardComputeImplementationAssignment getFirstAssignment() const;
  mlir::FailureOr<std::optional<CardComputeImplementationAssignment>>
  getNextAssignment(
      const CardComputeImplementationAssignment &assignment) const;
  bool contains(const CardComputeImplementationAssignment &assignment) const;

private:
  struct NodeDomain {
    StructuredDAGNodeID node = 0;
    bool supportsReciprocal = false;
  };

  explicit CardComputeImplementationDomain(
      llvm::SmallVector<NodeDomain, 16> nodes)
      : nodes(std::move(nodes)) {}

  llvm::SmallVector<NodeDomain, 16> nodes;
};

} // namespace wafer::compiler::detail
