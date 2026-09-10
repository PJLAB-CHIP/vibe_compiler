//===- RootRegionWork.h - Derived single-root Tile work -------*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_ROOTREGIONWORK_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_ROOTREGIONWORK_H

#include "Wafer/Planning/PhysicalDataflow/ExactDemand.h"

#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::analysis {

struct RootRegionWorkId {
  compiler::detail::SemanticRootKey root;
  TileId tile{0};

  friend bool operator==(const RootRegionWorkId &lhs,
                         const RootRegionWorkId &rhs) {
    return lhs.root == rhs.root && lhs.tile == rhs.tile;
  }
  friend bool operator!=(const RootRegionWorkId &lhs,
                         const RootRegionWorkId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const RootRegionWorkId &lhs,
                        const RootRegionWorkId &rhs) {
    if (lhs.root != rhs.root)
      return lhs.root < rhs.root;
    return lhs.tile.getValue() < rhs.tile.getValue();
  }
};

struct RootUseId {
  uint32_t operand = 0;
  DemandDestination destination;

  friend bool operator==(const RootUseId &lhs, const RootUseId &rhs) {
    return lhs.operand == rhs.operand && lhs.destination == rhs.destination;
  }
  friend bool operator<(const RootUseId &lhs, const RootUseId &rhs) {
    if (lhs.operand != rhs.operand)
      return lhs.operand < rhs.operand;
    return lhs.destination < rhs.destination;
  }
};

struct SupportValueId {
  compiler::detail::SemanticRootKey valuePath;
  uint32_t result = 0;

  friend bool operator==(const SupportValueId &lhs, const SupportValueId &rhs) {
    return lhs.valuePath == rhs.valuePath && lhs.result == rhs.result;
  }
  friend bool operator!=(const SupportValueId &lhs, const SupportValueId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const SupportValueId &lhs, const SupportValueId &rhs) {
    if (lhs.valuePath != rhs.valuePath)
      return lhs.valuePath < rhs.valuePath;
    return lhs.result < rhs.result;
  }
};

enum class RootBoundaryKind : uint8_t {
  StructuredResult,
  ProgramInput,
  Constant,
  CapturedValue,
};

struct RootBoundaryId {
  RootBoundaryKind kind = RootBoundaryKind::ProgramInput;
  compiler::detail::SemanticRootKey semantic;
  uint32_t index = 0;

  friend bool operator==(const RootBoundaryId &lhs, const RootBoundaryId &rhs) {
    return lhs.kind == rhs.kind && lhs.semantic == rhs.semantic &&
           lhs.index == rhs.index;
  }
  friend bool operator!=(const RootBoundaryId &lhs, const RootBoundaryId &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const RootBoundaryId &lhs, const RootBoundaryId &rhs) {
    if (lhs.kind != rhs.kind)
      return lhs.kind < rhs.kind;
    if (lhs.semantic != rhs.semantic)
      return lhs.semantic < rhs.semantic;
    return lhs.index < rhs.index;
  }
};

struct RootExecutionWork {
  compiler::detail::LogicalShardId shard;
  llvm::SmallVector<compiler::detail::IteratorInterval, 4> iterationDomain;
};

struct RootOperandUseWork {
  RootUseId id;
  ExactIndexSet consumerExecutionDomain;
  ExactIndexSet operandDemand;
};

struct RootOperandWork {
  uint32_t operand = 0;
  DemandOperandKind kind = DemandOperandKind::DataInput;
  std::vector<RootOperandUseWork> uses;
};

struct RootBoundaryUseWork {
  RootUseId id;
  std::optional<ExactIndexSet> requiredDomain;
  llvm::SmallVector<OwnerIntersection, 4> eligibleFinalOwners;
};

struct RootBoundaryWork {
  RootBoundaryId id;
  /// Current-IR-epoch lookup handle. It is not part of identity or ordering.
  mlir::Value sourceValue;
  std::optional<ExactIndexSet> requiredDomain;
  std::vector<RootBoundaryUseWork> consumerUses;
};

enum class RootSupportInputKind : uint8_t {
  SupportValue,
  Boundary,
  ExactEmpty,
};

struct RootSupportInputWork {
  uint32_t operand = 0;
  RootSupportInputKind kind = RootSupportInputKind::ExactEmpty;
  std::optional<SupportValueId> supportValue;
  std::optional<RootBoundaryId> boundary;
  ExactIndexSet requiredDomain;
};

struct RootSupportValueWork {
  SupportValueId id;
  /// Current-IR-epoch lookup handle. It is not part of identity or ordering.
  mlir::Operation *operation = nullptr;
  uint32_t result = 0;
  ExactIndexSet requiredDomain;
  llvm::SmallVector<RootSupportInputWork, 2> inputs;
  llvm::SmallVector<RootBoundaryId, 2> captures;
  std::vector<RootUseId> consumerUses;
};

enum class RootInvariantUseKind : uint8_t {
  Operand,
  RegionCapture,
};

struct RootInvariantInputWork {
  RootInvariantUseKind kind = RootInvariantUseKind::Operand;
  std::optional<uint32_t> operand;
  RootBoundaryId boundary;
};

struct RootContributionWork {
  compiler::detail::ReductionGroupId group;
  TileId mergeTile{0};
  ReductionInitialization initialization =
      ReductionInitialization::IdentityPerContributionInitOnceAtMerge;
  ReductionAlgebraKind algebra = ReductionAlgebraKind::StandardPartialReduction;
  std::optional<CoupledReductionRule> coupledRule;
  ReductionContribution contribution;
};

struct RootResultWork {
  uint32_t result = 0;
  std::optional<compiler::detail::LogicalShardId> ownerShard;
  std::optional<compiler::detail::ReductionGroupId> reductionGroup;
  ExactIndexSet domain;
};

/// Complete query-local work owned by one semantic root on one Tile. It has
/// no temporal, region-group, representation, movement, storage, event,
/// schedule, target, or materialized-IR fields.
struct RootRegionWork {
  RootRegionWorkId id;
  /// Current-IR-epoch lookup handle for the semantic root.
  mlir::Operation *rootOperation = nullptr;
  std::vector<RootExecutionWork> execution;
  std::vector<RootContributionWork> contributions;
  std::vector<ReductionMergeRequirement> merges;
  std::vector<RootOperandWork> operands;
  std::vector<RootSupportValueWork> supportValues;
  std::vector<RootBoundaryWork> boundaries;
  std::vector<RootInvariantInputWork> invariantInputs;
  std::vector<RootResultWork> results;
};

struct NoRootRegionWork {};

enum class UnsupportedRootRegionWorkReason : uint8_t {
  MissingSemanticIdentity,
  UnsupportedCapture,
  UnsupportedSupportSemantics,
};

struct UnsupportedRootRegionWork {
  UnsupportedRootRegionWorkReason reason =
      UnsupportedRootRegionWorkReason::MissingSemanticIdentity;
  RootRegionWorkId site;
  std::string detail;
};

enum class BrokenRootRegionWorkReason : uint8_t {
  AssignmentProofMismatch,
  MissingBoundary,
  SupportGraphCycle,
  InterfaceContradiction,
};

struct BrokenRootRegionWork {
  BrokenRootRegionWorkReason reason =
      BrokenRootRegionWorkReason::AssignmentProofMismatch;
  RootRegionWorkId site;
  std::string detail;
};

struct RootRegionWorkLimitReached {
  RootRegionWorkId site;
  uint64_t predictedPieces = 0;
  uint64_t pieceLimit = 0;
  std::string detail;
};

using RootRegionWorkOutcome =
    std::variant<RootRegionWork, NoRootRegionWork, UnsupportedRootRegionWork,
                 BrokenRootRegionWork, RootRegionWorkLimitReached>;

} // namespace wafer::analysis

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_ROOTREGIONWORK_H
