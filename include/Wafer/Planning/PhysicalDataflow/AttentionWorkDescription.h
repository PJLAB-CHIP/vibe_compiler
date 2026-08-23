//===- AttentionWorkDescription.h - Typed attention work DAG -*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_ATTENTIONWORKDESCRIPTION_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_ATTENTIONWORKDESCRIPTION_H

#include "Wafer/Planning/PhysicalDataflow/SchedulePlan.h"

#include "mlir/IR/AffineMap.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

struct AttentionWorkScopeId {
  ExecutionInstanceId execution;
  std::optional<ReductionGroupId> group;

  friend bool operator==(const AttentionWorkScopeId &lhs,
                         const AttentionWorkScopeId &rhs) {
    return lhs.execution == rhs.execution && lhs.group == rhs.group;
  }
  friend bool operator<(const AttentionWorkScopeId &lhs,
                        const AttentionWorkScopeId &rhs) {
    if (!(lhs.execution == rhs.execution))
      return lhs.execution < rhs.execution;
    return lhs.group < rhs.group;
  }
};

enum class AttentionActionKind : uint8_t {
  QueryKeyContraction,
  ScaleMask,
  RowMaximum,
  Exponential,
  RowSum,
  ValueContraction,
  StateUpdate,
  StateMerge,
  Finalize,
};

struct AttentionActionId {
  AttentionWorkScopeId scope;
  AttentionActionKind kind = AttentionActionKind::QueryKeyContraction;

  friend bool operator==(const AttentionActionId &lhs,
                         const AttentionActionId &rhs) {
    return lhs.scope == rhs.scope && lhs.kind == rhs.kind;
  }
  friend bool operator<(const AttentionActionId &lhs,
                        const AttentionActionId &rhs) {
    if (!(lhs.scope == rhs.scope))
      return lhs.scope < rhs.scope;
    return lhs.kind < rhs.kind;
  }
};

enum class AttentionValueKind : uint8_t {
  ScoreBlock,
  ScaledMaskedScoreBlock,
  ProbabilityBlock,
  BlockMaximum,
  BlockSum,
  BlockAccumulator,
  RunningMaximum,
  RunningSum,
  RunningAccumulator,
  FinalOutput,
};

/// Target-abstract storage required by the selected Linalg decomposition but
/// not observable as an attention algorithm value. These roles are part of
/// the immutable work description so feasibility accounts for every buffer
/// that deterministic lowering must materialize; TileRegion lowering does not
/// discover or choose them from SPM pressure.
enum class AttentionScratchKind : uint8_t {
  ConvertedScoreBlock,
  ScaleBlock,
  ScaledScoreBlock,
  BroadcastMaskBlock,
  ConvertedMaskBlock,
  BroadcastMaximumBlock,
  ShiftedScoreBlock,
  WideProbabilityBlock,
};

struct AttentionValueId {
  AttentionWorkScopeId scope;
  AttentionValueKind kind = AttentionValueKind::ScoreBlock;

  friend bool operator==(const AttentionValueId &lhs,
                         const AttentionValueId &rhs) {
    return lhs.scope == rhs.scope && lhs.kind == rhs.kind;
  }
  friend bool operator<(const AttentionValueId &lhs,
                        const AttentionValueId &rhs) {
    if (!(lhs.scope == rhs.scope))
      return lhs.scope < rhs.scope;
    return lhs.kind < rhs.kind;
  }
};

struct AttentionScratchId {
  AttentionWorkScopeId scope;
  AttentionScratchKind kind = AttentionScratchKind::ConvertedScoreBlock;

  friend bool operator==(const AttentionScratchId &lhs,
                         const AttentionScratchId &rhs) {
    return lhs.scope == rhs.scope && lhs.kind == rhs.kind;
  }
  friend bool operator<(const AttentionScratchId &lhs,
                        const AttentionScratchId &rhs) {
    if (!(lhs.scope == rhs.scope))
      return lhs.scope < rhs.scope;
    return lhs.kind < rhs.kind;
  }
};

enum class AttentionOperandRole : uint8_t { Query, Key, Value, Mask };

struct AttentionOperandFragmentProjection {
  PhysicalVersionId version;
  StorageObjectId storage;
  analysis::ExactIndexSet exactDomain;
  mlir::Type elementType;
};

struct AttentionOperandDescription {
  AttentionWorkScopeId scope;
  AttentionOperandRole role = AttentionOperandRole::Query;
  /// Exact demand in the normalized attention operand's own coordinates.
  analysis::ExactIndexSet exactDomain;
  /// Maximum operand window resident for one selected temporal wave. This is
  /// projected from the same current attention iteration domain as
  /// `exactDomain`; it is not an SPM estimate or allocation decision.
  analysis::ExactIndexSet residentDomain;
  mlir::Type elementType;
  mlir::AffineMap indexingMap;
  /// Physical leaf projections that reconstruct this logical operand demand.
  /// Their ranks may differ from the final operand because support transforms
  /// remain explicit in the source SSA graph.
  std::vector<AttentionOperandFragmentProjection> fragments;
};

struct AttentionValueDescription {
  AttentionValueId id;
  analysis::ExactIndexSet exactDomain;
  analysis::ExactIndexSet residentDomain;
  mlir::Type elementType;
  mlir::AffineMap indexingMap;
  std::optional<PhysicalVersionId> physicalVersion;
  std::optional<StorageObjectId> storage;
};

struct AttentionScratchDescription {
  AttentionScratchId id;
  analysis::ExactIndexSet exactDomain;
  analysis::ExactIndexSet residentDomain;
  mlir::Type elementType;
  mlir::AffineMap indexingMap;
  AttentionActionId definition;
  std::vector<AttentionActionId> uses;
};

struct AttentionActionDescription {
  AttentionActionId id;
  analysis::ExactIndexSet logicalWorkDomain;
  std::vector<AttentionValueId> inputs;
  std::vector<AttentionValueId> outputs;
};

struct AttentionGatherProjection {
  AttentionValueId value;
  ReductionGatherId gather;
  StorageObjectId staging;
};

struct AttentionSimultaneousValueGroup {
  AttentionWorkScopeId scope;
  std::vector<AttentionValueId> values;
};

struct AttentionWorkDescription {
  SemanticRootKey root;
  AttentionAlgorithm algorithm = AttentionAlgorithm::FlashAttention;
  std::vector<AttentionActionDescription> actions;
  std::vector<AttentionValueDescription> values;
  std::vector<AttentionScratchDescription> scratch;
  std::vector<AttentionOperandDescription> operands;
  std::vector<AttentionGatherProjection> gathers;
  std::vector<AttentionSimultaneousValueGroup> simultaneousValues;
};

struct CanonicalAttentionWorkCoordinate {
  std::vector<AttentionWorkDescription> roots;
};

enum class BrokenAttentionWorkProjectionReason : uint8_t {
  InvalidAttentionSemantics,
  PlanWorkMismatch,
  DuplicateIdentity,
  MissingOperandProjection,
  MissingPhysicalVersion,
  MissingStorageBinding,
  MissingScheduleNode,
  ComponentMismatch,
  MovementMismatch,
  ResourceMismatch,
};

struct BrokenAttentionWorkProjection {
  BrokenAttentionWorkProjectionReason reason =
      BrokenAttentionWorkProjectionReason::InvalidAttentionSemantics;
  std::optional<SemanticRootKey> root;
  std::string detail;
};

using CanonicalAttentionWorkProjectionOutcome =
    std::variant<CanonicalAttentionWorkCoordinate,
                 BrokenAttentionWorkProjection>;

const CanonicalAttentionWorkCoordinate *getCanonicalAttentionWorkCoordinate(
    const CanonicalAttentionWorkProjectionOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_ATTENTIONWORKDESCRIPTION_H
