//===- DataMovement.h - Explicit physical data movement ------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Search/PhysicalRepresentation.h"

namespace wafer::compiler::detail {

enum class DataMovementKind : uint8_t {
  Retained,
  Refetch,
  DDR,
  Recompute,
  Peer,
};

struct DataMovementFragment {
  TileId sourceTile{0};
  MemLayout sourceLayout = MemLayout::Tensor;
  MemLayout transportLayout = MemLayout::Tensor;
  mlir::Type elementType;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  llvm::SmallVector<int64_t, 4> sourceBaseOffsets;
  llvm::SmallVector<int64_t, 4> sourceShape;
  llvm::SmallVector<int64_t, 4> destinationOffsets;
  uint64_t logicalBytes = 0;
  uint64_t physicalBytes = 0;
  llvm::SmallVector<TileLink, 8> route;

  friend bool operator==(const DataMovementFragment &lhs,
                         const DataMovementFragment &rhs) {
    return lhs.sourceTile == rhs.sourceTile &&
           lhs.sourceLayout == rhs.sourceLayout &&
           lhs.transportLayout == rhs.transportLayout &&
           lhs.elementType == rhs.elementType && lhs.offsets == rhs.offsets &&
           lhs.sizes == rhs.sizes &&
           lhs.sourceBaseOffsets == rhs.sourceBaseOffsets &&
           lhs.sourceShape == rhs.sourceShape &&
           lhs.destinationOffsets == rhs.destinationOffsets &&
           lhs.logicalBytes == rhs.logicalBytes &&
           lhs.physicalBytes == rhs.physicalBytes && lhs.route == rhs.route;
  }
};

struct DataMovementChoice {
  StructuredDAGEdgeID edge = 0;
  TileId destinationTile{0};
  DataMovementKind kind = DataMovementKind::DDR;
  MemLayout consumerLayout = MemLayout::Tensor;
  llvm::SmallVector<DataMovementFragment, 4> fragments;
  /// -1 is unicast. Non-negative values identify one shared multicast tree;
  /// the canonical identity is the minimum destination Tile in that group.
  int64_t multicastGroup = -1;

  friend bool operator==(const DataMovementChoice &lhs,
                         const DataMovementChoice &rhs) {
    return lhs.edge == rhs.edge && lhs.destinationTile == rhs.destinationTile &&
           lhs.kind == rhs.kind && lhs.consumerLayout == rhs.consumerLayout &&
           lhs.fragments == rhs.fragments &&
           lhs.multicastGroup == rhs.multicastGroup;
  }
};

enum class ReductionGatherKind : uint8_t { DDR, Peer };

struct ReductionGatherChoice {
  StructuredDAGNodeID node = 0;
  unsigned resultIndex = 0;
  TileId mergeTile{0};
  ReductionGatherKind kind = ReductionGatherKind::DDR;
  MemLayout mergeLayout = MemLayout::Tensor;
  llvm::SmallVector<DataMovementFragment, 8> fragments;

  friend bool operator==(const ReductionGatherChoice &lhs,
                         const ReductionGatherChoice &rhs) {
    return lhs.node == rhs.node && lhs.resultIndex == rhs.resultIndex &&
           lhs.mergeTile == rhs.mergeTile && lhs.kind == rhs.kind &&
           lhs.mergeLayout == rhs.mergeLayout && lhs.fragments == rhs.fragments;
  }
};

struct CardDataMovementAssignment {
  llvm::SmallVector<DataMovementChoice, 32> edges;
  llvm::SmallVector<ReductionGatherChoice, 4> reductions;

  friend bool operator==(const CardDataMovementAssignment &lhs,
                         const CardDataMovementAssignment &rhs) {
    return lhs.edges == rhs.edges && lhs.reductions == rhs.reductions;
  }
};

struct DataReuseFact {
  StructuredDAGEdgeID edge = 0;
  TileId destinationTile{0};
  llvm::SmallVector<TileId, 4> equivalentDestinationTiles;
  llvm::SmallVector<uint32_t, 4> temporalInvariantIterators;
};

/// Lazy movement domain for every exact data-input demand. Retained is
/// available only inside one selected group; DDR is the canonical explicit
/// boundary; pure producers may be recomputed; remote ownership admits peer
/// fragments whose route state lazily covers every verifier-legal simple path.
class CardDataMovementDomain {
public:
  static mlir::FailureOr<CardDataMovementDomain>
  create(const CardProgramAnalysis &program, CardId cardId,
         const analysis::LogicalShardTrial &trial,
         const CoupledRegionDomain &coupledDomain,
         const CoupledRegionAssignment &coupledAssignment,
         const CardTemporalDomain &temporalDomain,
         const CardTemporalAssignment &temporalAssignment,
         const CardPhysicalRepresentationDomain &representationDomain,
         const CardPhysicalRepresentationAssignment &representationAssignment,
         std::string *failureReason = nullptr);

  CardDataMovementAssignment getFirstAssignment() const;
  mlir::FailureOr<std::optional<CardDataMovementAssignment>>
  getNextAssignment(const CardDataMovementAssignment &assignment) const;
  bool contains(const CardDataMovementAssignment &assignment) const;
  llvm::SmallVector<DataReuseFact, 32> getReuseFacts() const;

private:
  struct DemandDomain {
    StructuredDAGEdgeID edge = 0;
    TileId destinationTile{0};
    MemLayout consumerLayout = MemLayout::Tensor;
    bool retained = false;
    bool recompute = false;
    bool peer = false;
    llvm::SmallVector<DataMovementFragment, 4> fragments;
    llvm::SmallVector<TileId, 4> equivalentDestinationTiles;
    llvm::SmallVector<uint32_t, 4> temporalInvariantIterators;
  };

  struct ReductionDomain {
    StructuredDAGNodeID node = 0;
    unsigned resultIndex = 0;
    TileId mergeTile{0};
    MemLayout mergeLayout = MemLayout::Tensor;
    llvm::SmallVector<DataMovementFragment, 8> fragments;
  };

  CardDataMovementDomain(TargetTopology topology, CardId cardId,
                         llvm::SmallVector<DemandDomain, 32> demands,
                         llvm::SmallVector<ReductionDomain, 4> reductions)
      : topology(std::move(topology)), cardId(cardId),
        demands(std::move(demands)), reductions(std::move(reductions)) {}

  DataMovementChoice getFirstChoice(const DemandDomain &domain) const;
  mlir::FailureOr<std::optional<DataMovementChoice>>
  getNextChoice(const DemandDomain &domain,
                const DataMovementChoice &choice) const;
  bool contains(const DemandDomain &domain,
                const DataMovementChoice &choice) const;

  TargetTopology topology;
  CardId cardId{0};
  llvm::SmallVector<DemandDomain, 32> demands;
  llvm::SmallVector<ReductionDomain, 4> reductions;
};

} // namespace wafer::compiler::detail
