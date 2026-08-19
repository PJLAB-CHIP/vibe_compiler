//===- SpatialPlacement.h - Structured iterator placement -----*- C++ -*-===//
#pragma once

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"
#include "Wafer/IR/Target/TargetTopology.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace wafer::compiler::detail {

/// One block partition and physical embedding for a structured DAG node.
/// `iteratorFactors` covers every iterator. `tiles[linearCoordinate]` maps the
/// canonical row-major logical partition coordinate to one distinct physical
/// Tile. The object owns no temporal, layout, movement or scheduling choice.
struct SpatialPlacementAssignment {
  StructuredDAGNodeID node = 0;
  llvm::SmallVector<uint32_t, 4> iteratorFactors;
  llvm::SmallVector<TileId, 16> tiles;
  std::optional<TileId> reductionMergeTile;

  friend bool operator==(const SpatialPlacementAssignment &lhs,
                         const SpatialPlacementAssignment &rhs) {
    return lhs.node == rhs.node && lhs.iteratorFactors == rhs.iteratorFactors &&
           lhs.tiles == rhs.tiles &&
           lhs.reductionMergeTile == rhs.reductionMergeTile;
  }
  friend bool operator<(const SpatialPlacementAssignment &lhs,
                        const SpatialPlacementAssignment &rhs);

  StructuredDAGNodePlacement getNodePlacement() const;
};

/// Compact complete block-partition domain for one current structured node.
/// Factor vectors and ordered physical embeddings are advanced lazily; no
/// point vector, selected winner, score or default statistic is constructed.
class SpatialPlacementDomain {
public:
  static mlir::FailureOr<SpatialPlacementDomain>
  create(const StructuredDAGNode &node, llvm::ArrayRef<TileId> availableTiles);

  SpatialPlacementAssignment getFirstAssignment() const;
  /// Deterministic constructive proposal with the largest participant count
  /// representable by this exact domain. Ties avoid reduction partitioning
  /// first and then prefer earlier iterator dimensions. This changes only
  /// visitation order; `getNextAssignment` remains the complete enumeration.
  SpatialPlacementAssignment getMaximumParticipantAssignment() const;
  SpatialPlacementAssignment
  getMaximumParticipantAssignment(uint64_t maximumParticipants) const;
  mlir::FailureOr<std::optional<SpatialPlacementAssignment>>
  getNextAssignment(const SpatialPlacementAssignment &assignment) const;
  bool contains(const SpatialPlacementAssignment &assignment) const;

  StructuredDAGNodeID getNode() const { return node; }
  llvm::ArrayRef<int64_t> getIteratorExtents() const { return iteratorExtents; }
  llvm::ArrayRef<uint8_t> getReductionIterators() const {
    return reductionIterators;
  }
  llvm::ArrayRef<int64_t> getMaximumFactors() const { return maximumFactors; }
  llvm::ArrayRef<TileId> getAvailableTiles() const { return availableTiles; }

private:
  SpatialPlacementDomain(StructuredDAGNodeID node,
                         llvm::SmallVector<int64_t, 4> iteratorExtents,
                         llvm::SmallVector<int64_t, 4> maximumFactors,
                         llvm::SmallVector<uint8_t, 4> reductionIterators,
                         llvm::SmallVector<TileId, 16> availableTiles)
      : node(node), iteratorExtents(std::move(iteratorExtents)),
        maximumFactors(std::move(maximumFactors)),
        reductionIterators(std::move(reductionIterators)),
        availableTiles(std::move(availableTiles)) {}

  StructuredDAGNodeID node;
  llvm::SmallVector<int64_t, 4> iteratorExtents;
  llvm::SmallVector<int64_t, 4> maximumFactors;
  llvm::SmallVector<uint8_t, 4> reductionIterators;
  llvm::SmallVector<TileId, 16> availableTiles;
};

/// One complete spatial assignment across every structured DAG node. Node
/// assignments are stored in stable DAG-id order; independent branches may
/// therefore choose unrelated physical Tile embeddings.
struct CardSpatialPlacementAssignment {
  llvm::SmallVector<SpatialPlacementAssignment, 16> nodes;

  friend bool operator==(const CardSpatialPlacementAssignment &lhs,
                         const CardSpatialPlacementAssignment &rhs) {
    return lhs.nodes == rhs.nodes;
  }
  friend bool operator<(const CardSpatialPlacementAssignment &lhs,
                        const CardSpatialPlacementAssignment &rhs) {
    return std::lexicographical_compare(lhs.nodes.begin(), lhs.nodes.end(),
                                        rhs.nodes.begin(), rhs.nodes.end());
  }
};

struct CardSpatialPlacementEvaluation {
  analysis::ExactDemandStatus status =
      analysis::ExactDemandStatus::IndeterminateFailure;
  std::optional<analysis::LogicalShardTrial> trial;
  std::string detail;
};

/// Lazy Cartesian product of the per-node domains. It holds only the compact
/// node domains and one assignment supplied by the caller; no global point
/// vector or local winner is materialized.
class CardSpatialPlacementDomain {
public:
  static mlir::FailureOr<CardSpatialPlacementDomain>
  create(const StructuredDAGAnalysis &dag,
         llvm::ArrayRef<TileId> availableTiles);

  CardSpatialPlacementAssignment getFirstAssignment() const;
  CardSpatialPlacementAssignment getMaximumParticipantAssignment() const;
  mlir::FailureOr<CardSpatialPlacementAssignment>
  getConstructiveAssignment(const StructuredDAGAnalysis &dag,
                            analysis::IREpoch epoch,
                            std::string *failureReason = nullptr) const;
  mlir::FailureOr<std::optional<CardSpatialPlacementAssignment>>
  getNextAssignment(const CardSpatialPlacementAssignment &assignment) const;
  bool contains(const CardSpatialPlacementAssignment &assignment) const;
  llvm::SmallVector<StructuredDAGNodePlacement, 16>
  getNodePlacements(const CardSpatialPlacementAssignment &assignment) const;
  CardSpatialPlacementEvaluation
  evaluate(const StructuredDAGAnalysis &dag, analysis::IREpoch epoch,
           const CardSpatialPlacementAssignment &assignment) const;

private:
  explicit CardSpatialPlacementDomain(
      llvm::SmallVector<SpatialPlacementDomain, 16> nodeDomains)
      : nodeDomains(std::move(nodeDomains)) {}

  llvm::SmallVector<SpatialPlacementDomain, 16> nodeDomains;
};

} // namespace wafer::compiler::detail
