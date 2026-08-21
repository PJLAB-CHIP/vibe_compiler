//===- Buffering.h - Rotating-buffer candidate domain -------*- C++ -*-===//

#pragma once

#include "Wafer/Planning/Search/DataMovement.h"

#include "Wafer/Target/Core/TargetMemory.h"

namespace wafer::compiler::detail {

/// One independently materializable rotating-buffer scope. An empty edge set
/// with one slot is the serialized identity. A multi-slot choice names every
/// exact logical edge whose producer-to-consumer dependency must be witnessed
/// by the same actual static loop.
struct BufferingChoice {
  TileId tile{0};
  llvm::SmallVector<StructuredDAGNodeID, 4> groupNodes;
  llvm::SmallVector<StructuredDAGEdgeID, 4> pipelinedEdges;
  uint32_t slotCount = 1;

  friend bool operator==(const BufferingChoice &lhs,
                         const BufferingChoice &rhs) {
    return lhs.tile == rhs.tile && lhs.groupNodes == rhs.groupNodes &&
           lhs.pipelinedEdges == rhs.pipelinedEdges &&
           lhs.slotCount == rhs.slotCount;
  }
};

struct CardBufferingAssignment {
  llvm::SmallVector<BufferingChoice, 16> groups;

  friend bool operator==(const CardBufferingAssignment &lhs,
                         const CardBufferingAssignment &rhs) {
    return lhs.groups == rhs.groups;
  }
};

/// Lazy Cartesian domain over serialized and rotating-buffer choices. The
/// finite multiplicity bound is derived from selected temporal steady waves
/// and an exact single-fragment physical-footprint lower bound. Final alias,
/// coexistence, lifetime and common-loop legality remain actual-IR gates.
class CardBufferingDomain {
public:
  static mlir::FailureOr<CardBufferingDomain>
  create(const CardProgramAnalysis &program,
         const SpatialAssignment &spatial,
         const analysis::ExactDemandProof &demand,
         const CoupledRegionDomain &coupledDomain,
         const CoupledRegionAssignment &coupledAssignment,
         const CardTemporalDomain &temporalDomain,
         const CardTemporalAssignment &temporalAssignment,
         const CardPhysicalRepresentationDomain &representationDomain,
         const CardPhysicalRepresentationAssignment &representationAssignment,
         const CardDataMovementDomain &movementDomain,
         const CardDataMovementAssignment &movementAssignment,
         const TargetMemoryPolicy &memory,
         std::string *failureReason = nullptr);

  CardBufferingAssignment getFirstAssignment() const;
  mlir::FailureOr<std::optional<CardBufferingAssignment>>
  getNextAssignment(const CardBufferingAssignment &assignment) const;
  bool contains(const CardBufferingAssignment &assignment) const;

private:
  struct EdgeDomain {
    StructuredDAGEdgeID edge = 0;
    uint32_t maximumSlots = 1;
  };

  struct GroupDomain {
    TileId tile{0};
    llvm::SmallVector<StructuredDAGNodeID, 4> nodes;
    llvm::SmallVector<EdgeDomain, 4> edges;
  };

  explicit CardBufferingDomain(llvm::SmallVector<GroupDomain, 16> groups)
      : groups(std::move(groups)) {}

  BufferingChoice getFirstChoice(const GroupDomain &group) const;
  mlir::FailureOr<std::optional<BufferingChoice>>
  getNextChoice(const GroupDomain &group, const BufferingChoice &choice) const;
  bool contains(const GroupDomain &group, const BufferingChoice &choice) const;

  llvm::SmallVector<GroupDomain, 16> groups;
};

} // namespace wafer::compiler::detail
