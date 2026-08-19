//===- CoupledRegion.h - Connected node-shard partitions -*- C++ -*-===//

#pragma once

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"
#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"
#include "Wafer/IR/Target/TargetTopology.h"
#include "Wafer/Planning/Search/TemporalTiling.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace wafer::compiler::detail {

class CardPhysicalRepresentationDomain;
struct CardPhysicalRepresentationAssignment;
class CardComputeImplementationDomain;
struct CardComputeImplementationAssignment;
class CardDataMovementDomain;
struct CardDataMovementAssignment;

/// One selected connected group of structured node shards on one Tile. Node
/// IDs are strictly increasing. The object carries no edge action, traversal
/// cache, score, or actual IR handle.
struct CoupledRegionGroup {
  TileId tile{0};
  llvm::SmallVector<StructuredDAGNodeID, 4> nodes;

  friend bool operator==(const CoupledRegionGroup &lhs,
                         const CoupledRegionGroup &rhs) {
    return lhs.tile == rhs.tile && lhs.nodes == rhs.nodes;
  }
  friend bool operator<(const CoupledRegionGroup &lhs,
                        const CoupledRegionGroup &rhs) {
    if (lhs.tile != rhs.tile)
      return lhs.tile.getValue() < rhs.tile.getValue();
    return lhs.nodes < rhs.nodes;
  }
};

/// Complete partition of every selected node/Tile shard. Groups are ordered
/// by Tile and then smallest node ID.
struct CoupledRegionAssignment {
  llvm::SmallVector<CoupledRegionGroup, 32> groups;

  friend bool operator==(const CoupledRegionAssignment &lhs,
                         const CoupledRegionAssignment &rhs) {
    return lhs.groups == rhs.groups;
  }
  friend bool operator<(const CoupledRegionAssignment &lhs,
                        const CoupledRegionAssignment &rhs) {
    return lhs.groups < rhs.groups;
  }
};

/// Lazy complete domain of legal connected partitions. Enumeration advances
/// canonical restricted-growth labels supplied by the current assignment; it
/// stores no vector of partition points and selects no preferred group.
class CoupledRegionDomain {
public:
  struct TileDomain {
    TileId tile{0};
    llvm::SmallVector<StructuredDAGNodeID, 16> nodes;
    llvm::SmallVector<uint8_t, 256> fusableEdges;
    llvm::SmallVector<uint8_t, 256> forbiddenInternalEdges;
  };

  static mlir::FailureOr<CoupledRegionDomain>
  create(const StructuredDAGAnalysis &dag,
         const analysis::LogicalShardTrial &trial,
         std::string *failureReason = nullptr);

  CoupledRegionAssignment getFirstAssignment() const;
  /// Deterministic repair for a whole connected-component neighborhood. It
  /// greedily merges legal adjacent groups until no further merge is legal;
  /// the exact domain and its enumeration remain unchanged.
  CoupledRegionAssignment getFusionOrientedAssignment() const;
  mlir::FailureOr<std::optional<CoupledRegionAssignment>>
  getNextAssignment(const CoupledRegionAssignment &assignment) const;
  bool contains(const CoupledRegionAssignment &assignment) const;

private:
  explicit CoupledRegionDomain(llvm::SmallVector<TileDomain, 16> tiles)
      : tiles(std::move(tiles)) {}

  llvm::SmallVector<TileDomain, 16> tiles;
};

struct CardCoupledRegionMaterialization {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations relations;
};

/// Applies one explicit complete group partition. The domain is supplied by
/// the caller so apply validates membership without rebuilding or selecting a
/// second grouping domain.
mlir::FailureOr<CardCoupledRegionMaterialization> materializeCardCoupledRegions(
    mlir::ModuleOp tensorProgram, const CardProgramAnalysis &program,
    CardId cardId, const analysis::LogicalShardTrial &trial,
    const CoupledRegionDomain &domain,
    const CoupledRegionAssignment &assignment,
    const CardTemporalDomain &temporalDomain,
    const CardTemporalAssignment &temporalAssignment,
    const CardPhysicalRepresentationDomain &representationDomain,
    const CardPhysicalRepresentationAssignment &representationAssignment,
    const CardDataMovementDomain &movementDomain,
    const CardDataMovementAssignment &movementAssignment,
    std::string *failureReason = nullptr);

mlir::FailureOr<CardCoupledRegionMaterialization>
materializeCardCoupledRegionsWithImplementations(
    mlir::ModuleOp tensorProgram, const CardProgramAnalysis &program,
    CardId cardId, const analysis::LogicalShardTrial &trial,
    const CoupledRegionDomain &domain,
    const CoupledRegionAssignment &assignment,
    const CardTemporalDomain &temporalDomain,
    const CardTemporalAssignment &temporalAssignment,
    const CardPhysicalRepresentationDomain &representationDomain,
    const CardPhysicalRepresentationAssignment &representationAssignment,
    const CardComputeImplementationDomain &implementationDomain,
    const CardComputeImplementationAssignment &implementationAssignment,
    const CardDataMovementDomain &movementDomain,
    const CardDataMovementAssignment &movementAssignment,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail
