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

enum class AttentionOperandRole : uint8_t { Query, Key, Value, Mask };

struct AttentionOperandProjection {
  AttentionWorkScopeId scope;
  AttentionOperandRole role = AttentionOperandRole::Query;
  PhysicalVersionId version;
  StorageObjectId storage;
  analysis::ExactIndexSet exactDomain;
  mlir::Type elementType;
  mlir::AffineMap indexingMap;
};

struct AttentionValueDescription {
  AttentionValueId id;
  analysis::ExactIndexSet exactDomain;
  mlir::Type elementType;
  mlir::AffineMap indexingMap;
  std::optional<PhysicalVersionId> physicalVersion;
  std::optional<StorageObjectId> storage;
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
  std::vector<AttentionOperandProjection> operands;
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
