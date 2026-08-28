//===- ExactDemand.h - Exact logical dependency proof --------*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_EXACTDEMAND_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_EXACTDEMAND_H

#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialAssignment.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Analysis/Presburger/PresburgerRelation.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlir {
class Operation;
class Value;
} // namespace mlir

namespace wafer::analysis {

using compiler::detail::LogicalShardId;
using compiler::detail::ReductionGroupId;
using compiler::detail::SemanticRootKey;

enum class ExactIndexSetForm : uint8_t {
  BoxUnion,
  StridedBoxUnion,
  BoundedAffineImageUnion,
  RowMajorIntervalUnion,
  GeneralPresburger,
};

/// Exact query-local integer set together with the construction normal form
/// that bounds production operations. The Presburger value and the normal
/// form describe one semantic set; neither is a cache or side channel.
class ExactIndexSet {
public:
  ExactIndexSet();
  ExactIndexSet(mlir::presburger::PresburgerSet set, ExactIndexSetForm form,
                llvm::ArrayRef<StaticRectangularIndexSet> boxes = {});

  unsigned getRank() const { return set.getSpace().getNumSetDimVars(); }
  bool isEmpty() const { return set.isIntegerEmpty(); }
  ExactIndexSetForm getForm() const { return form; }
  const mlir::presburger::PresburgerSet &getPresburgerSet() const {
    return set;
  }
  llvm::ArrayRef<StaticRectangularIndexSet> getBoxes() const { return boxes; }

private:
  mlir::presburger::PresburgerSet set;
  ExactIndexSetForm form = ExactIndexSetForm::BoxUnion;
  llvm::SmallVector<StaticRectangularIndexSet, 8> boxes;
};

/// Recovers a finite disjoint rectangular normal form without changing the
/// represented integer set. Fails when the current exact Presburger set is not
/// representable as a finite box union.
mlir::FailureOr<ExactIndexSet>
normalizeFiniteExactIndexSet(const ExactIndexSet &set);

enum class DemandOperandKind : uint8_t {
  DataInput,
  InitInput,
};

struct StructuredResultSource {
  SemanticRootKey root;
  mlir::Operation *operation = nullptr;
  uint32_t result = 0;
};

struct ProgramInputSource {
  uint32_t argument = 0;
};

struct ConstantSource {
  SemanticRootKey valuePath;
  mlir::Operation *operation = nullptr;
  uint32_t result = 0;
};

using DemandSource =
    std::variant<StructuredResultSource, ProgramInputSource, ConstantSource>;

struct FinalResultOwner {
  SemanticRootKey root;
  uint32_t result = 0;
  std::optional<LogicalShardId> shard;
  std::optional<ReductionGroupId> reductionGroup;
  TileId tile{0};
  ExactIndexSet domain;
};

struct OwnerIntersection {
  std::optional<LogicalShardId> ownerShard;
  std::optional<ReductionGroupId> reductionGroup;
  TileId tile{0};
  ExactIndexSet domain;
};

enum class TensorTransformKind : uint8_t {
  ExpandShape,
  CollapseShape,
  ExtractSlice,
  InsertSlice,
  Pad,
  Cast,
};

struct TensorTransformInputDemand {
  uint32_t operand = 0;
  ExactIndexSet demand;
};

struct TensorTransform {
  mlir::Operation *operation = nullptr;
  uint32_t result = 0;
  TensorTransformKind kind = TensorTransformKind::Cast;
  ExactIndexSet outputDemand;
  llvm::SmallVector<TensorTransformInputDemand, 2> operandDemands;
};

struct OperandReconstruction {
  std::vector<TensorTransform> steps;
};

struct SourceDemand {
  DemandSource source;
  ExactIndexSet requiredDomain;
  llvm::SmallVector<OwnerIntersection, 8> eligibleFinalOwners;
};

struct DestinationDemand {
  LogicalShardId destinationShard;
  TileId destinationTile{0};
  ExactIndexSet consumerExecutionDomain;
  ExactIndexSet operandDemand;
  std::vector<SourceDemand> sources;
  OperandReconstruction reconstruction;
};

struct DependencyDemand {
  SemanticRootKey consumer;
  mlir::Operation *consumerOperation = nullptr;
  uint32_t consumerOperand = 0;
  DemandOperandKind kind = DemandOperandKind::DataInput;
  std::vector<DestinationDemand> perDestination;
};

enum class ReductionInitialization : uint8_t {
  IdentityPerContributionInitOnceAtMerge,
  CoupledIdentityPerContribution,
};

enum class ReductionAlgebraKind : uint8_t {
  StandardPartialReduction,
  CoupledReduction,
};

struct ReductionResultSlice {
  uint32_t result = 0;
  ExactIndexSet domain;
};

struct CoupledReductionComponentRequirement {
  wafer::CoupledReductionComponentKind kind =
      wafer::CoupledReductionComponentKind::Maximum;
  mlir::AffineMap indexingMap;
  mlir::Type elementType;
  ExactIndexSet domain;
};

struct CoupledReductionComponentSlice {
  wafer::CoupledReductionComponentKind kind =
      wafer::CoupledReductionComponentKind::Maximum;
  ExactIndexSet domain;
};

struct CoupledReductionRule {
  wafer::CoupledReductionMergeKind mergeKind =
      wafer::CoupledReductionMergeKind::OnlineAttention;
  wafer::CoupledReductionFinalizationKind finalizationKind =
      wafer::CoupledReductionFinalizationKind::NormalizeAccumulator;
};

struct ReductionContribution {
  LogicalShardId shard;
  TileId tile{0};
  ExactIndexSet iterationDomain;
  llvm::SmallVector<ReductionResultSlice, 3> results;
  llvm::SmallVector<CoupledReductionComponentSlice, 3> components;
};

struct ReductionMergeRequirement {
  ReductionGroupId group;
  TileId mergeTile{0};
  ReductionInitialization initialization =
      ReductionInitialization::IdentityPerContributionInitOnceAtMerge;
  ReductionAlgebraKind algebra =
      ReductionAlgebraKind::StandardPartialReduction;
  llvm::SmallVector<ReductionResultSlice, 3> results;
  std::optional<CoupledReductionRule> coupledRule;
  llvm::SmallVector<CoupledReductionComponentRequirement, 3> components;
  std::vector<ReductionContribution> contributions;
};

struct ExactDemandProof {
  std::vector<FinalResultOwner> finalOwners;
  std::vector<DependencyDemand> dependencyDemands;
  std::vector<ReductionMergeRequirement> reductionMerges;
};

enum class RelationOperationKind : uint8_t {
  BuildRelationGraph,
  ValidateAssignment,
  ConstructRelation,
  Image,
  Union,
  Intersection,
  ResultAvailability,
  ReductionCompletion,
};

struct DemandFailureSite {
  std::optional<SemanticRootKey> root;
  std::optional<uint32_t> result;
  std::optional<uint32_t> operand;
  std::optional<LogicalShardId> shard;
  RelationOperationKind operation = RelationOperationKind::BuildRelationGraph;
};

enum class UnsupportedDemandReason : uint8_t {
  MissingStructuredIndexing,
  MissingTensorTransfer,
  DynamicShape,
  UnsupportedControlFlow,
  MissingReductionAlgebra,
};

struct UnsupportedDemandSemantics {
  UnsupportedDemandReason reason =
      UnsupportedDemandReason::MissingStructuredIndexing;
  DemandFailureSite site;
  std::string detail;
};

struct DemandWorkLimitReached {
  DemandFailureSite site;
  uint64_t predictedWork = 0;
  uint64_t workLimit = 0;
  std::string detail;
};

enum class InvalidSpatialAssignmentReason : uint8_t {
  RootCoverage,
  ExecutionPartition,
  ResultCoverage,
  ReductionGroup,
};

struct InvalidSpatialAssignment {
  InvalidSpatialAssignmentReason reason =
      InvalidSpatialAssignmentReason::RootCoverage;
  DemandFailureSite site;
  std::string detail;
};

enum class BrokenDemandContractReason : uint8_t {
  InterfaceContradiction,
  RelationGraphCycle,
  InternalExactnessFailure,
};

struct BrokenDemandContract {
  BrokenDemandContractReason reason =
      BrokenDemandContractReason::InterfaceContradiction;
  DemandFailureSite site;
  std::string detail;
};

using ExactDemandOutcome =
    std::variant<ExactDemandProof, UnsupportedDemandSemantics,
                 DemandWorkLimitReached, InvalidSpatialAssignment,
                 BrokenDemandContract>;

enum class ExactDemandOutcomeCategory : uint8_t {
  Satisfied,
  UnsupportedSemantics,
  IndeterminateResourceExhaustion,
  CompilerContractError,
};

ExactDemandOutcomeCategory classifyExactDemandOutcome(
    const ExactDemandOutcome &outcome);

const ExactDemandProof *getExactDemandProof(const ExactDemandOutcome &outcome);

} // namespace wafer::analysis

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_EXACTDEMAND_H
