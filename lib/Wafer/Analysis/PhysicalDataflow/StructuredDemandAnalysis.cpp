//===- StructuredDemandAnalysis.cpp - Exact structured demand ----------===//

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace wafer::compiler::detail {
namespace {

using analysis::BrokenDemandContract;
using analysis::BrokenDemandContractReason;
using analysis::ConstantSource;
using analysis::CoupledReductionComponentRequirement;
using analysis::CoupledReductionComponentSlice;
using analysis::CoupledReductionRule;
using analysis::DemandFailureSite;
using analysis::DemandOperandKind;
using analysis::DemandSource;
using analysis::DemandWorkLimitReached;
using analysis::DependencyDemand;
using analysis::DestinationDemand;
using analysis::ExactDemandOutcome;
using analysis::ExactDemandProof;
using analysis::ExactIndexSet;
using analysis::ExactIndexSetForm;
using analysis::FinalResultOwner;
using analysis::IndexRelation;
using analysis::IndexRelationLimits;
using analysis::IndexRelationResult;
using analysis::IndexRelationStatus;
using analysis::IndexSetResult;
using analysis::InvalidSpatialAssignment;
using analysis::InvalidSpatialAssignmentReason;
using analysis::OperandReconstruction;
using analysis::OwnerIntersection;
using analysis::ProgramInputSource;
using analysis::ReductionAlgebraKind;
using analysis::ReductionContribution;
using analysis::ReductionInitialization;
using analysis::ReductionMergeRequirement;
using analysis::ReductionResultSlice;
using analysis::RelationOperationKind;
using analysis::SourceDemand;
using analysis::StaticRectangularIndexSet;
using analysis::StructuredResultSource;
using analysis::TensorTransform;
using analysis::TensorTransformInputDemand;
using analysis::TensorTransformKind;
using analysis::UnsupportedDemandReason;
using analysis::UnsupportedDemandSemantics;

using DemandFailure =
    std::variant<UnsupportedDemandSemantics, DemandWorkLimitReached,
                 InvalidSpatialAssignment, BrokenDemandContract>;

template <typename T>
using DemandResult =
    std::variant<T, UnsupportedDemandSemantics, DemandWorkLimitReached,
                 InvalidSpatialAssignment, BrokenDemandContract>;

template <typename T> const T *getValue(const DemandResult<T> &result) {
  return std::get_if<T>(&result);
}

template <typename T> T *getValue(DemandResult<T> &result) {
  return std::get_if<T>(&result);
}

template <typename T> ExactDemandOutcome takeFailure(DemandResult<T> &&result) {
  return std::visit(
      [](auto &&value) -> ExactDemandOutcome {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, T>) {
          BrokenDemandContract failure;
          failure.reason = BrokenDemandContractReason::InternalExactnessFailure;
          failure.detail = "demand result unexpectedly contained a value";
          return failure;
        } else {
          return std::forward<decltype(value)>(value);
        }
      },
      std::move(result));
}

DemandFailure broken(BrokenDemandContractReason reason,
                     RelationOperationKind operation, llvm::StringRef detail,
                     std::optional<SemanticRootKey> root = std::nullopt) {
  BrokenDemandContract failure;
  failure.reason = reason;
  failure.site.root = std::move(root);
  failure.site.operation = operation;
  failure.detail = detail.str();
  return failure;
}

DemandFailure unsupported(UnsupportedDemandReason reason,
                          RelationOperationKind operation,
                          llvm::StringRef detail,
                          std::optional<SemanticRootKey> root = std::nullopt) {
  UnsupportedDemandSemantics failure;
  failure.reason = reason;
  failure.site.root = std::move(root);
  failure.site.operation = operation;
  failure.detail = detail.str();
  return failure;
}

DemandFailure workLimit(RelationOperationKind operation, uint64_t predictedWork,
                        uint64_t limit, llvm::StringRef detail,
                        std::optional<SemanticRootKey> root = std::nullopt) {
  DemandWorkLimitReached failure;
  failure.site.root = std::move(root);
  failure.site.operation = operation;
  failure.predictedWork = predictedWork;
  failure.workLimit = limit;
  failure.detail = detail.str();
  return failure;
}

DemandFailure
invalidAssignment(InvalidSpatialAssignmentReason reason, llvm::StringRef detail,
                  std::optional<SemanticRootKey> root = std::nullopt) {
  InvalidSpatialAssignment failure;
  failure.reason = reason;
  failure.site.root = std::move(root);
  failure.site.operation = RelationOperationKind::ValidateAssignment;
  failure.detail = detail.str();
  return failure;
}

template <typename T> DemandResult<T> asResult(DemandFailure failure) {
  return std::visit(
      [](auto &&value) -> DemandResult<T> {
        return std::forward<decltype(value)>(value);
      },
      std::move(failure));
}

template <typename T> DemandFailure getFailure(DemandResult<T> &&result) {
  return std::visit(
      [](auto &&value) -> DemandFailure {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, T>) {
          return BrokenDemandContract{
              BrokenDemandContractReason::InternalExactnessFailure,
              {},
              "demand result unexpectedly contained a value"};
        } else {
          return std::forward<decltype(value)>(value);
        }
      },
      std::move(result));
}

ExactDemandOutcome getFailureOutcome(const DemandFailure &failure) {
  return std::visit(
      [](const auto &value) -> ExactDemandOutcome { return value; }, failure);
}

DemandResult<ExactIndexSet>
makeBoxUnion(unsigned rank, llvm::ArrayRef<StaticRectangularIndexSet> boxes,
             const IndexRelationLimits &limits, RelationOperationKind operation,
             std::optional<SemanticRootKey> root = std::nullopt) {
  if (boxes.size() > limits.maxRectangularPieces)
    return asResult<ExactIndexSet>(workLimit(
        operation, boxes.size(), limits.maxRectangularPieces,
        "rectangle piece count exceeds exact-demand work limit", root));
  std::optional<mlir::presburger::PresburgerSet> set;
  for (const StaticRectangularIndexSet &box : boxes) {
    if (box.offsets.size() != rank || box.sizes.size() != rank)
      return asResult<ExactIndexSet>(
          broken(BrokenDemandContractReason::InterfaceContradiction, operation,
                 "rectangle rank does not match exact set", root));
    IndexSetResult piece =
        IndexRelation::staticRectangularDomain(box.offsets, box.sizes, limits);
    if (!piece.isExact()) {
      if (piece.status == IndexRelationStatus::ResourceExhausted)
        return asResult<ExactIndexSet>(workLimit(operation, boxes.size(),
                                                 limits.maxRectangularPieces,
                                                 piece.reason, root));
      return asResult<ExactIndexSet>(
          broken(BrokenDemandContractReason::InternalExactnessFailure,
                 operation, piece.reason, root));
    }
    set = set ? set->unionSet(*piece.set) : std::move(*piece.set);
  }
  if (!set)
    set = mlir::presburger::PresburgerSet::getEmpty(
        mlir::presburger::PresburgerSpace::getSetSpace(rank));
  return ExactIndexSet(std::move(*set), ExactIndexSetForm::BoxUnion, boxes);
}

DemandResult<ExactIndexSet> makeShardSet(const ExecutionShard &shard,
                                         const IndexRelationLimits &limits) {
  StaticRectangularIndexSet box;
  for (const IteratorInterval &interval : shard.iterationDomain) {
    box.offsets.push_back(interval.offset);
    box.sizes.push_back(interval.size);
  }
  return makeBoxUnion(shard.iterationDomain.size(), {box}, limits,
                      RelationOperationKind::ValidateAssignment,
                      shard.shard.root);
}

DemandResult<ExactIndexSet>
unionExactSets(const ExactIndexSet &lhs, const ExactIndexSet &rhs,
               const IndexRelationLimits &limits,
               RelationOperationKind operation,
               std::optional<SemanticRootKey> root = std::nullopt) {
  if (lhs.getRank() != rhs.getRank())
    return asResult<ExactIndexSet>(
        broken(BrokenDemandContractReason::InterfaceContradiction, operation,
               "cannot union exact sets with different ranks", root));
  uint64_t predicted = lhs.getPresburgerSet().getNumDisjuncts() +
                       rhs.getPresburgerSet().getNumDisjuncts();
  if (predicted > limits.maxRectangularPieces)
    return asResult<ExactIndexSet>(
        workLimit(operation, predicted, limits.maxRectangularPieces,
                  "exact set union exceeds work limit", root));
  llvm::SmallVector<StaticRectangularIndexSet, 8> boxes;
  ExactIndexSetForm form = ExactIndexSetForm::GeneralPresburger;
  if (lhs.getForm() == ExactIndexSetForm::BoxUnion &&
      rhs.getForm() == ExactIndexSetForm::BoxUnion) {
    boxes.append(lhs.getBoxes().begin(), lhs.getBoxes().end());
    boxes.append(rhs.getBoxes().begin(), rhs.getBoxes().end());
    form = ExactIndexSetForm::BoxUnion;
  }
  return ExactIndexSet(lhs.getPresburgerSet().unionSet(rhs.getPresburgerSet()),
                       form, boxes);
}

DemandResult<ExactIndexSet>
intersectExactSets(const ExactIndexSet &lhs, const ExactIndexSet &rhs,
                   const IndexRelationLimits &limits,
                   RelationOperationKind operation,
                   std::optional<SemanticRootKey> root = std::nullopt) {
  if (lhs.getRank() != rhs.getRank())
    return asResult<ExactIndexSet>(
        broken(BrokenDemandContractReason::InterfaceContradiction, operation,
               "cannot intersect exact sets with different ranks", root));
  const uint64_t predicted =
      static_cast<uint64_t>(lhs.getPresburgerSet().getNumDisjuncts()) *
      rhs.getPresburgerSet().getNumDisjuncts();
  if (predicted > limits.maxRectangularPieces)
    return asResult<ExactIndexSet>(
        workLimit(operation, predicted, limits.maxRectangularPieces,
                  "exact set intersection exceeds work limit", root));
  if (lhs.getForm() == ExactIndexSetForm::BoxUnion &&
      rhs.getForm() == ExactIndexSetForm::BoxUnion) {
    llvm::SmallVector<StaticRectangularIndexSet, 8> boxes;
    for (const StaticRectangularIndexSet &lhsBox : lhs.getBoxes()) {
      for (const StaticRectangularIndexSet &rhsBox : rhs.getBoxes()) {
        StaticRectangularIndexSet overlap;
        bool nonempty = true;
        for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] : llvm::zip_equal(
                 lhsBox.offsets, lhsBox.sizes, rhsBox.offsets, rhsBox.sizes)) {
          const int64_t begin = std::max(lhsOffset, rhsOffset);
          const int64_t end =
              std::min(lhsOffset + lhsSize, rhsOffset + rhsSize);
          if (begin >= end)
            nonempty = false;
          overlap.offsets.push_back(begin);
          overlap.sizes.push_back(std::max<int64_t>(0, end - begin));
        }
        if (nonempty)
          boxes.push_back(std::move(overlap));
      }
    }
    return makeBoxUnion(lhs.getRank(), boxes, limits, operation, root);
  }
  return ExactIndexSet(lhs.getPresburgerSet().intersect(rhs.getPresburgerSet()),
                       ExactIndexSetForm::GeneralPresburger);
}

DemandResult<ExactIndexSet>
imageExactSet(const IndexRelation &relation, const ExactIndexSet &domain,
              const IndexRelationLimits &limits,
              RelationOperationKind operation,
              std::optional<SemanticRootKey> root = std::nullopt) {
  if (domain.getRank() != relation.getDestinationRank())
    return asResult<ExactIndexSet>(
        broken(BrokenDemandContractReason::InterfaceContradiction, operation,
               "relation destination rank does not match exact demand", root));
  if (domain.isEmpty())
    return makeBoxUnion(relation.getSourceRank(), {}, limits, operation, root);

  if (domain.getForm() == ExactIndexSetForm::BoxUnion) {
    llvm::SmallVector<StaticRectangularIndexSet, 8> boxes;
    for (const StaticRectangularIndexSet &box : domain.getBoxes()) {
      analysis::StaticRectangularIndexSetPiecesResult image =
          relation.getExactStaticRectangularImagePieces(box.offsets, box.sizes,
                                                        limits);
      if (image.isExact()) {
        if (boxes.size() + image.domains.size() > limits.maxRectangularPieces)
          return asResult<ExactIndexSet>(
              workLimit(operation, boxes.size() + image.domains.size(),
                        limits.maxRectangularPieces,
                        "relation image exceeds rectangle work limit", root));
        boxes.append(std::move(image.domains));
        continue;
      }
      if (image.status == IndexRelationStatus::ResourceExhausted)
        return asResult<ExactIndexSet>(
            workLimit(operation, limits.maxRectangularPieces + 1,
                      limits.maxRectangularPieces, image.reason, root));
      boxes.clear();
      break;
    }
    if (!boxes.empty())
      return makeBoxUnion(relation.getSourceRank(), boxes, limits, operation,
                          root);
  }

  const uint64_t predicted = domain.getPresburgerSet().getNumDisjuncts();
  if (predicted > limits.maxDisjuncts)
    return asResult<ExactIndexSet>(
        workLimit(operation, predicted, limits.maxDisjuncts,
                  "affine image exceeds Presburger disjunct work limit", root));
  IndexSetResult image = relation.image(domain.getPresburgerSet(), limits);
  if (!image.isExact()) {
    if (image.status == IndexRelationStatus::Unsupported)
      return asResult<ExactIndexSet>(
          unsupported(UnsupportedDemandReason::MissingTensorTransfer, operation,
                      image.reason, root));
    if (image.status == IndexRelationStatus::ResourceExhausted)
      return asResult<ExactIndexSet>(workLimit(
          operation, predicted, limits.maxDisjuncts, image.reason, root));
    return asResult<ExactIndexSet>(
        broken(BrokenDemandContractReason::InternalExactnessFailure, operation,
               image.reason, root));
  }
  return ExactIndexSet(std::move(*image.set),
                       ExactIndexSetForm::BoundedAffineImageUnion);
}

struct StructuredOperandFact {
  uint32_t operand = 0;
  DemandOperandKind kind = DemandOperandKind::DataInput;
  IndexRelation iterationToOperand;
};

struct StructuredResultFact {
  uint32_t result = 0;
  mlir::AffineMap map;
  IndexRelation iterationToResult;
  llvm::SmallVector<int64_t, 4> shape;
};

struct CoupledReductionComponentFact {
  wafer::CoupledReductionComponentKind kind =
      wafer::CoupledReductionComponentKind::Maximum;
  mlir::AffineMap map;
  mlir::Type elementType;
  IndexRelation iterationToComponent;
};

struct CoupledReductionFact {
  llvm::SmallVector<unsigned, 2> reductionIterators;
  llvm::SmallVector<CoupledReductionComponentFact, 3> components;
  CoupledReductionRule rule;
};

struct StructuredOperationFact {
  SemanticRootKey root;
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<int64_t, 8> iterationShape;
  llvm::SmallVector<mlir::utils::IteratorType, 8> iteratorTypes;
  llvm::SmallVector<StructuredOperandFact, 8> operands;
  llvm::SmallVector<StructuredResultFact, 3> results;
  std::optional<CoupledReductionFact> coupledReduction;
};

DemandResult<StructuredOperationFact>
deriveStructuredOperationFact(const SemanticRootBinding &binding,
                              const IndexRelationLimits &limits) {
  StructuredOperationFact fact;
  fact.root = binding.key;
  fact.operation = binding.operation;
  auto dps =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(binding.operation);
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(binding.operation);
  if (!dps || !tiling)
    return asResult<StructuredOperationFact>(unsupported(
        UnsupportedDemandReason::MissingStructuredIndexing,
        RelationOperationKind::BuildRelationGraph,
        "structured root lacks destination or tiling semantics", fact.root));
  fact.iteratorTypes = tiling.getLoopIteratorTypes();

  llvm::SmallVector<mlir::AffineMap, 8> maps;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(binding.operation)) {
    fact.iterationShape = linalg.getStaticLoopRanges();
    maps = linalg.getIndexingMapsArray();
  } else if (auto attention = mlir::dyn_cast<wafer::LinalgExtAttentionOp>(
                 binding.operation)) {
    fact.iterationShape = attention.getStaticLoopRanges();
    maps = attention.getIndexingMapsArray();
  } else {
    return asResult<StructuredOperationFact>(unsupported(
        UnsupportedDemandReason::MissingStructuredIndexing,
        RelationOperationKind::BuildRelationGraph,
        "structured root has no current typed indexing contract", fact.root));
  }
  if (fact.iteratorTypes.size() != fact.iterationShape.size() ||
      maps.size() != binding.operation->getNumOperands() ||
      llvm::any_of(fact.iterationShape,
                   [](int64_t extent) { return extent <= 0; }))
    return asResult<StructuredOperationFact>(unsupported(
        UnsupportedDemandReason::DynamicShape,
        RelationOperationKind::BuildRelationGraph,
        "structured root requires positive static iterator/indexing facts",
        fact.root));

  auto attention =
      mlir::dyn_cast<wafer::LinalgExtAttentionOp>(binding.operation);
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(binding.operation);
  for (mlir::OpOperand &operand : binding.operation->getOpOperands()) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(operand.get().getType());
    if (!type)
      continue;
    if (!type.hasStaticShape())
      return asResult<StructuredOperationFact>(unsupported(
          UnsupportedDemandReason::DynamicShape,
          RelationOperationKind::BuildRelationGraph,
          "structured operand requires a static tensor type", fact.root));
    bool readsPayload = false;
    if (linalg)
      readsPayload = linalg.payloadUsesValueFromOperand(&operand);
    else if (attention)
      readsPayload = dps.isDpsInput(&operand);
    if (!readsPayload)
      continue;
    IndexRelationResult relation = IndexRelation::fromAffineMap(
        maps[operand.getOperandNumber()], fact.iterationShape, type.getShape(),
        limits);
    if (!relation.isExact()) {
      if (relation.status == IndexRelationStatus::ResourceExhausted)
        return asResult<StructuredOperationFact>(workLimit(
            RelationOperationKind::ConstructRelation, limits.maxDisjuncts + 1,
            limits.maxDisjuncts, relation.reason, fact.root));
      return asResult<StructuredOperationFact>(
          unsupported(UnsupportedDemandReason::MissingStructuredIndexing,
                      RelationOperationKind::ConstructRelation, relation.reason,
                      fact.root));
    }
    fact.operands.push_back({static_cast<uint32_t>(operand.getOperandNumber()),
                             dps.isDpsInit(&operand)
                                 ? DemandOperandKind::InitInput
                                 : DemandOperandKind::DataInput,
                             std::move(*relation.relation)});
  }

  for (mlir::OpResult result : binding.operation->getResults()) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!type || !type.hasStaticShape())
      return asResult<StructuredOperationFact>(unsupported(
          UnsupportedDemandReason::DynamicShape,
          RelationOperationKind::BuildRelationGraph,
          "structured result requires a static tensor type", fact.root));
    mlir::AffineMap map;
    if (linalg)
      map = linalg.getIndexingMapMatchingResult(result);
    else if (attention && result.getResultNumber() == 0)
      map = attention.getOutputMap();
    if (!map)
      return asResult<StructuredOperationFact>(
          broken(BrokenDemandContractReason::InterfaceContradiction,
                 RelationOperationKind::BuildRelationGraph,
                 "structured result has no indexing map", fact.root));
    IndexRelationResult relation = IndexRelation::fromAffineMap(
        map, fact.iterationShape, type.getShape(), limits);
    if (!relation.isExact())
      return asResult<StructuredOperationFact>(
          unsupported(UnsupportedDemandReason::MissingStructuredIndexing,
                      RelationOperationKind::ConstructRelation, relation.reason,
                      fact.root));
    fact.results.push_back({static_cast<uint32_t>(result.getResultNumber()),
                            map, std::move(*relation.relation),
                            llvm::to_vector(type.getShape())});
  }

  if (auto coupled = mlir::dyn_cast<wafer::WaferCoupledReductionOpInterface>(
          binding.operation)) {
    wafer::CoupledReductionDescription description =
        coupled.getCoupledReductionDescription();
    if (description.reductionIterators.empty() ||
        description.components.empty())
      return asResult<StructuredOperationFact>(
          broken(BrokenDemandContractReason::InterfaceContradiction,
                 RelationOperationKind::BuildRelationGraph,
                 "coupled reduction interface returned an empty description",
                 fact.root));

    CoupledReductionFact coupledFact;
    llvm::SmallBitVector seenIterators(fact.iterationShape.size(), false);
    for (unsigned iterator : description.reductionIterators) {
      if (iterator >= fact.iterationShape.size() ||
          seenIterators.test(iterator) ||
          fact.iteratorTypes[iterator] != mlir::utils::IteratorType::reduction)
        return asResult<StructuredOperationFact>(
            broken(BrokenDemandContractReason::InterfaceContradiction,
                   RelationOperationKind::BuildRelationGraph,
                   "coupled reduction interface returned invalid reduction "
                   "iterators",
                   fact.root));
      seenIterators.set(iterator);
      coupledFact.reductionIterators.push_back(iterator);
    }

    std::set<uint8_t> seenComponents;
    for (const wafer::CoupledReductionComponent &component :
         description.components) {
      const uint8_t componentKey = static_cast<uint8_t>(component.kind);
      if (!seenComponents.insert(componentKey).second ||
          !component.indexingMap || !component.elementType ||
          component.indexingMap.getNumDims() != fact.iterationShape.size() ||
          component.indexingMap.getNumSymbols() != 0 ||
          !component.indexingMap.isProjectedPermutation())
        return asResult<StructuredOperationFact>(
            broken(BrokenDemandContractReason::InterfaceContradiction,
                   RelationOperationKind::BuildRelationGraph,
                   "coupled reduction interface returned an invalid component",
                   fact.root));

      llvm::SmallVector<int64_t, 4> componentShape;
      for (mlir::AffineExpr expression : component.indexingMap.getResults()) {
        auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!dimension || dimension.getPosition() >= fact.iterationShape.size())
          return asResult<StructuredOperationFact>(broken(
              BrokenDemandContractReason::InterfaceContradiction,
              RelationOperationKind::BuildRelationGraph,
              "coupled reduction component map is not a static projection",
              fact.root));
        componentShape.push_back(fact.iterationShape[dimension.getPosition()]);
      }
      IndexRelationResult relation = IndexRelation::fromAffineMap(
          component.indexingMap, fact.iterationShape, componentShape, limits);
      if (!relation.isExact()) {
        if (relation.status == IndexRelationStatus::ResourceExhausted)
          return asResult<StructuredOperationFact>(workLimit(
              RelationOperationKind::ConstructRelation, limits.maxDisjuncts + 1,
              limits.maxDisjuncts, relation.reason, fact.root));
        return asResult<StructuredOperationFact>(broken(
            BrokenDemandContractReason::InterfaceContradiction,
            RelationOperationKind::ConstructRelation,
            "coupled reduction component relation is not exact", fact.root));
      }
      coupledFact.components.push_back({component.kind, component.indexingMap,
                                        component.elementType,
                                        std::move(*relation.relation)});
    }
    coupledFact.rule = {description.mergeKind, description.finalizationKind};
    fact.coupledReduction = std::move(coupledFact);
  }
  return fact;
}

struct SupportOperandFact {
  uint32_t operand = 0;
  wafer::TensorIndexingOperandRole role =
      wafer::TensorIndexingOperandRole::Source;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> strides;
  IndexRelation resultToOperand;
};

struct SupportTransferFact {
  mlir::Operation *operation = nullptr;
  uint32_t result = 0;
  TensorTransformKind kind = TensorTransformKind::Cast;
  llvm::SmallVector<SupportOperandFact, 2> operands;
};

struct ResultRelationFact {
  explicit ResultRelationFact(mlir::OpResult result) : result(result) {}

  mlir::OpResult result;
  std::optional<SupportTransferFact> transfer;
  std::optional<DemandFailure> failure;
};

std::optional<TensorTransformKind>
getTensorTransformKind(wafer::TensorIndexingTransformKind kind) {
  switch (kind) {
  case wafer::TensorIndexingTransformKind::ExpandShape:
    return TensorTransformKind::ExpandShape;
  case wafer::TensorIndexingTransformKind::CollapseShape:
    return TensorTransformKind::CollapseShape;
  case wafer::TensorIndexingTransformKind::ExtractSlice:
    return TensorTransformKind::ExtractSlice;
  case wafer::TensorIndexingTransformKind::InsertSlice:
    return TensorTransformKind::InsertSlice;
  case wafer::TensorIndexingTransformKind::Pad:
    return TensorTransformKind::Pad;
  case wafer::TensorIndexingTransformKind::Cast:
    return TensorTransformKind::Cast;
  }
  return std::nullopt;
}

DemandResult<SupportTransferFact>
deriveSupportTransfer(mlir::OpResult result,
                      const IndexRelationLimits &limits) {
  mlir::Operation *operation = result.getOwner();
  if (!operation || !mlir::isMemoryEffectFree(operation))
    return asResult<SupportTransferFact>(unsupported(
        UnsupportedDemandReason::MissingTensorTransfer,
        RelationOperationKind::BuildRelationGraph,
        "tensor support result lacks a pure exact transfer contract"));
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  if (!resultType || !resultType.hasStaticShape())
    return asResult<SupportTransferFact>(
        unsupported(UnsupportedDemandReason::DynamicShape,
                    RelationOperationKind::BuildRelationGraph,
                    "tensor support result requires a static ranked tensor"));

  SupportTransferFact transfer;
  transfer.operation = operation;
  auto indexing =
      mlir::dyn_cast<wafer::WaferTensorIndexingOpInterface>(operation);
  if (!indexing)
    return asResult<SupportTransferFact>(
        unsupported(UnsupportedDemandReason::MissingTensorTransfer,
                    RelationOperationKind::BuildRelationGraph,
                    (llvm::Twine("tensor support operation ") +
                     operation->getName().getStringRef() +
                     " has no exact indexing interface")
                        .str()));
  mlir::FailureOr<wafer::TensorIndexingDescription> description =
      indexing.getTensorIndexingDescription(result.getResultNumber());
  if (mlir::failed(description))
    return asResult<SupportTransferFact>(unsupported(
        UnsupportedDemandReason::MissingTensorTransfer,
        RelationOperationKind::BuildRelationGraph,
        "tensor indexing interface has no static exact description"));
  std::optional<TensorTransformKind> transformKind =
      getTensorTransformKind(description->kind);
  if (!transformKind || description->result != result.getResultNumber() ||
      description->operands.empty() ||
      !llvm::is_sorted(description->operands,
                       [](const auto &lhs, const auto &rhs) {
                         return lhs.operand < rhs.operand;
                       }))
    return asResult<SupportTransferFact>(
        broken(BrokenDemandContractReason::InterfaceContradiction,
               RelationOperationKind::BuildRelationGraph,
               "tensor indexing interface returned a malformed description"));
  transfer.result = description->result;
  transfer.kind = *transformKind;

  const size_t sourceCount =
      llvm::count_if(description->operands, [](const auto &operand) {
        return operand.role == wafer::TensorIndexingOperandRole::Source;
      });
  const size_t destinationCount =
      llvm::count_if(description->operands, [](const auto &operand) {
        return operand.role == wafer::TensorIndexingOperandRole::Destination;
      });
  const bool isInsert =
      description->kind == wafer::TensorIndexingTransformKind::InsertSlice;
  if (sourceCount != 1 || destinationCount != (isInsert ? 1u : 0u) ||
      description->operands.size() != (isInsert ? 2u : 1u))
    return asResult<SupportTransferFact>(
        broken(BrokenDemandContractReason::InterfaceContradiction,
               RelationOperationKind::BuildRelationGraph,
               "tensor indexing interface returned invalid operand roles"));

  llvm::SmallBitVector seenOperands(operation->getNumOperands());
  auto addRelation =
      [&](const wafer::TensorIndexingOperandDescription &operand,
          IndexRelationResult relation) -> std::optional<DemandFailure> {
    if (operand.operand >= operation->getNumOperands() ||
        seenOperands.test(operand.operand))
      return broken(BrokenDemandContractReason::InterfaceContradiction,
                    RelationOperationKind::ConstructRelation,
                    "tensor indexing interface returned invalid operands");
    seenOperands.set(operand.operand);
    if (!relation.isExact()) {
      if (relation.status == IndexRelationStatus::ResourceExhausted)
        return workLimit(RelationOperationKind::ConstructRelation,
                         limits.maxRectangularPieces + 1,
                         limits.maxRectangularPieces, relation.reason);
      return unsupported(UnsupportedDemandReason::MissingTensorTransfer,
                         RelationOperationKind::ConstructRelation,
                         relation.reason);
    }
    transfer.operands.push_back({operand.operand, operand.role, operand.offsets,
                                 operand.strides,
                                 std::move(*relation.relation)});
    return std::nullopt;
  };

  for (const wafer::TensorIndexingOperandDescription &operand :
       description->operands) {
    if (operand.operand >= operation->getNumOperands())
      return asResult<SupportTransferFact>(
          broken(BrokenDemandContractReason::InterfaceContradiction,
                 RelationOperationKind::ConstructRelation,
                 "tensor indexing interface returned an out-of-range operand"));
    auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
        operation->getOperand(operand.operand).getType());
    if (!operandType || !operandType.hasStaticShape())
      return asResult<SupportTransferFact>(unsupported(
          UnsupportedDemandReason::DynamicShape,
          RelationOperationKind::ConstructRelation,
          "tensor indexing operand requires a static ranked tensor"));

    IndexRelationResult relation;
    switch (description->kind) {
    case wafer::TensorIndexingTransformKind::ExpandShape:
    case wafer::TensorIndexingTransformKind::CollapseShape:
    case wafer::TensorIndexingTransformKind::Cast:
      relation = IndexRelation::staticReshape(resultType.getShape(),
                                              operandType.getShape(), limits);
      break;
    case wafer::TensorIndexingTransformKind::ExtractSlice:
      relation = IndexRelation::staticSlice(
          resultType.getShape(), operandType.getShape(), operand.offsets,
          operand.strides, limits);
      break;
    case wafer::TensorIndexingTransformKind::InsertSlice:
      if (operand.role == wafer::TensorIndexingOperandRole::Destination) {
        relation = IndexRelation::identity(operandType.getShape(), limits);
      } else {
        if (!llvm::all_of(operand.strides,
                          [](int64_t stride) { return stride == 1; }))
          return asResult<SupportTransferFact>(
              unsupported(UnsupportedDemandReason::MissingTensorTransfer,
                          RelationOperationKind::ConstructRelation,
                          "insert_slice requires unit-stride exact semantics"));
        relation = IndexRelation::staticInsertSlice(resultType.getShape(),
                                                    operandType.getShape(),
                                                    operand.offsets, limits);
      }
      break;
    case wafer::TensorIndexingTransformKind::Pad:
      relation = IndexRelation::staticInsertSlice(resultType.getShape(),
                                                  operandType.getShape(),
                                                  operand.offsets, limits);
      break;
    }
    if (auto failure = addRelation(operand, std::move(relation)))
      return asResult<SupportTransferFact>(std::move(*failure));
  }
  return transfer;
}

DemandResult<ExactIndexSet>
imageInsertDestination(const SupportTransferFact &transfer,
                       const ExactIndexSet &demand,
                       const IndexRelationLimits &limits) {
  if (demand.getForm() != ExactIndexSetForm::BoxUnion)
    return asResult<ExactIndexSet>(unsupported(
        UnsupportedDemandReason::MissingTensorTransfer,
        RelationOperationKind::Image,
        "insert_slice destination demand requires finite box form"));
  auto source = llvm::find_if(transfer.operands, [](const auto &operand) {
    return operand.role == wafer::TensorIndexingOperandRole::Source;
  });
  if (source == transfer.operands.end() ||
      source->operand >= transfer.operation->getNumOperands())
    return asResult<ExactIndexSet>(
        broken(BrokenDemandContractReason::InterfaceContradiction,
               RelationOperationKind::Image,
               "insert_slice transfer has no source operand"));
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
      transfer.operation->getOperand(source->operand).getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      source->offsets.size() != static_cast<size_t>(sourceType.getRank()))
    return asResult<ExactIndexSet>(
        broken(BrokenDemandContractReason::InterfaceContradiction,
               RelationOperationKind::Image,
               "insert_slice source description is not static"));

  llvm::SmallVector<StaticRectangularIndexSet, 8> pieces;
  for (const StaticRectangularIndexSet &rectangle : demand.getBoxes()) {
    StaticRectangularIndexSet overlap;
    bool hasOverlap = true;
    for (auto [demandOffset, demandSize, pieceOffset, pieceSize] :
         llvm::zip_equal(rectangle.offsets, rectangle.sizes, source->offsets,
                         sourceType.getShape())) {
      const int64_t begin = std::max(demandOffset, pieceOffset);
      const int64_t end =
          std::min(demandOffset + demandSize, pieceOffset + pieceSize);
      if (begin >= end)
        hasOverlap = false;
      overlap.offsets.push_back(begin);
      overlap.sizes.push_back(std::max<int64_t>(0, end - begin));
    }
    if (!hasOverlap) {
      pieces.push_back(rectangle);
      continue;
    }
    StaticRectangularIndexSet core = rectangle;
    for (size_t dimension = 0; dimension < core.offsets.size(); ++dimension) {
      const int64_t coreBegin = core.offsets[dimension];
      const int64_t coreEnd = coreBegin + core.sizes[dimension];
      const int64_t overlapBegin = overlap.offsets[dimension];
      const int64_t overlapEnd = overlapBegin + overlap.sizes[dimension];
      if (coreBegin < overlapBegin) {
        StaticRectangularIndexSet lower = core;
        lower.sizes[dimension] = overlapBegin - coreBegin;
        pieces.push_back(std::move(lower));
        core.offsets[dimension] = overlapBegin;
        core.sizes[dimension] = coreEnd - overlapBegin;
      }
      if (overlapEnd < coreEnd) {
        StaticRectangularIndexSet upper = core;
        upper.offsets[dimension] = overlapEnd;
        upper.sizes[dimension] = coreEnd - overlapEnd;
        pieces.push_back(std::move(upper));
        core.sizes[dimension] = overlapEnd - core.offsets[dimension];
      }
    }
  }
  return makeBoxUnion(demand.getRank(), pieces, limits,
                      RelationOperationKind::Image);
}

DemandResult<ExactIndexSet> imageSupportOperand(
    const SupportTransferFact &transfer, const SupportOperandFact &operand,
    const ExactIndexSet &demand, const IndexRelationLimits &limits) {
  if (transfer.kind == TensorTransformKind::InsertSlice &&
      operand.role == wafer::TensorIndexingOperandRole::Destination)
    return imageInsertDestination(transfer, demand, limits);
  return imageExactSet(operand.resultToOperand, demand, limits,
                       RelationOperationKind::Image);
}

struct DemandKey {
  SemanticRootKey consumer;
  uint32_t operand = 0;
  LogicalShardId destination;

  friend bool operator<(const DemandKey &lhs, const DemandKey &rhs) {
    if (lhs.consumer != rhs.consumer)
      return lhs.consumer < rhs.consumer;
    if (lhs.operand != rhs.operand)
      return lhs.operand < rhs.operand;
    return lhs.destination < rhs.destination;
  }
};

struct BoundaryState {
  DemandSource source;
  ExactIndexSet demand;
};

struct QueryState {
  std::map<DemandKey, ExactIndexSet> demands;
};

void appendRootKey(std::vector<int64_t> &key, const SemanticRootKey &root) {
  key.push_back(static_cast<int64_t>(root.anchorKind));
  key.push_back(root.anchorIndex);
  key.push_back(root.path.size());
  for (const SemanticRootPathStep &step : root.path) {
    key.push_back(static_cast<int64_t>(step.relation));
    key.push_back(step.producerResult);
    key.push_back(step.consumerOperand);
  }
}

std::vector<int64_t>
getSpatialAssignmentKey(const SpatialAssignment &assignment) {
  std::vector<int64_t> key;
  key.push_back(assignment.nodes.size());
  for (const NodeExecutionPartition &node : assignment.nodes) {
    appendRootKey(key, node.root);
    key.push_back(node.shards.size());
    for (const ExecutionShard &shard : node.shards) {
      appendRootKey(key, shard.shard.root);
      key.push_back(shard.shard.coordinate.size());
      key.insert(key.end(), shard.shard.coordinate.begin(),
                 shard.shard.coordinate.end());
      key.push_back(shard.tile.getValue());
      key.push_back(shard.iterationDomain.size());
      for (const IteratorInterval &interval : shard.iterationDomain) {
        key.push_back(interval.offset);
        key.push_back(interval.size);
      }
    }
    key.push_back(node.reductionGroups.size());
    for (const ReductionGroupPlacement &placement : node.reductionGroups) {
      appendRootKey(key, placement.group.root);
      key.push_back(placement.group.resultGroup);
      key.push_back(placement.group.parallelCoordinate.size());
      key.insert(key.end(), placement.group.parallelCoordinate.begin(),
                 placement.group.parallelCoordinate.end());
      key.push_back(placement.mergeTile.getValue());
    }
  }
  return key;
}

const NodeExecutionPartition *
findNodeAssignment(const SpatialAssignment &assignment,
                   const SemanticRootKey &root) {
  auto found = llvm::lower_bound(
      assignment.nodes, root,
      [](const NodeExecutionPartition &node, const SemanticRootKey &candidate) {
        return node.root < candidate;
      });
  return found == assignment.nodes.end() || found->root != root ? nullptr
                                                                : &*found;
}

llvm::SmallVector<int64_t, 8>
getIntervalCounts(const NodeExecutionPartition &node, size_t rank) {
  llvm::SmallVector<int64_t, 8> counts(rank, 0);
  for (const ExecutionShard &shard : node.shards)
    for (auto [iterator, coordinate] : llvm::enumerate(shard.shard.coordinate))
      counts[iterator] = std::max<int64_t>(counts[iterator], coordinate + 1);
  return counts;
}

bool boxesOverlap(const StaticRectangularIndexSet &lhs,
                  const StaticRectangularIndexSet &rhs) {
  if (lhs.offsets.size() != rhs.offsets.size())
    return false;
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhs.offsets, lhs.sizes, rhs.offsets, rhs.sizes))
    if (lhsOffset >= rhsOffset + rhsSize || rhsOffset >= lhsOffset + lhsSize)
      return false;
  return true;
}

std::vector<int64_t> boxKey(const ExactIndexSet &set) {
  std::vector<int64_t> key;
  key.push_back(static_cast<int64_t>(set.getBoxes().size()));
  for (const StaticRectangularIndexSet &box : set.getBoxes()) {
    key.insert(key.end(), box.offsets.begin(), box.offsets.end());
    key.insert(key.end(), box.sizes.begin(), box.sizes.end());
  }
  return key;
}

bool haveSameBoxUnion(const ExactIndexSet &lhs, const ExactIndexSet &rhs) {
  return lhs.getRank() == rhs.getRank() &&
         lhs.getForm() == ExactIndexSetForm::BoxUnion &&
         rhs.getForm() == ExactIndexSetForm::BoxUnion &&
         boxKey(lhs) == boxKey(rhs);
}

bool finalOwnerLess(const FinalResultOwner &lhs, const FinalResultOwner &rhs) {
  if (lhs.root != rhs.root)
    return lhs.root < rhs.root;
  if (lhs.result != rhs.result)
    return lhs.result < rhs.result;
  if (lhs.reductionGroup != rhs.reductionGroup)
    return lhs.reductionGroup < rhs.reductionGroup;
  if (lhs.shard != rhs.shard)
    return lhs.shard < rhs.shard;
  return lhs.tile.getValue() < rhs.tile.getValue();
}

bool demandSourceLess(const DemandSource &lhs, const DemandSource &rhs) {
  if (lhs.index() != rhs.index())
    return lhs.index() < rhs.index();
  if (const auto *lhsStructured = std::get_if<StructuredResultSource>(&lhs)) {
    const auto &rhsStructured = std::get<StructuredResultSource>(rhs);
    if (lhsStructured->root != rhsStructured.root)
      return lhsStructured->root < rhsStructured.root;
    return lhsStructured->result < rhsStructured.result;
  }
  if (const auto *lhsInput = std::get_if<ProgramInputSource>(&lhs))
    return lhsInput->argument < std::get<ProgramInputSource>(rhs).argument;
  const auto &lhsConstant = std::get<ConstantSource>(lhs);
  const auto &rhsConstant = std::get<ConstantSource>(rhs);
  if (lhsConstant.valuePath != rhsConstant.valuePath)
    return lhsConstant.valuePath < rhsConstant.valuePath;
  return lhsConstant.result < rhsConstant.result;
}

struct AvailabilityResult {
  std::vector<FinalResultOwner> finalOwners;
  std::vector<ReductionMergeRequirement> reductionMerges;
};

DemandResult<AvailabilityResult>
deriveResultAvailability(llvm::ArrayRef<StructuredOperationFact> facts,
                         const SpatialAssignment &assignment,
                         const IndexRelationLimits &limits) {
  AvailabilityResult result;
  for (const StructuredOperationFact &fact : facts) {
    const NodeExecutionPartition *node =
        findNodeAssignment(assignment, fact.root);
    if (!node)
      return asResult<AvailabilityResult>(invalidAssignment(
          InvalidSpatialAssignmentReason::RootCoverage,
          "spatial assignment omits one semantic root", fact.root));
    llvm::SmallVector<int64_t, 8> intervalCounts =
        getIntervalCounts(*node, fact.iterationShape.size());
    const bool hasSpatialReduction = llvm::any_of(
        llvm::zip_equal(intervalCounts, fact.iteratorTypes), [](auto values) {
          return std::get<0>(values) > 1 &&
                 std::get<1>(values) == mlir::utils::IteratorType::reduction;
        });

    if (auto attention =
            mlir::dyn_cast<wafer::LinalgExtAttentionOp>(fact.operation)) {
      if (!fact.coupledReduction)
        return asResult<AvailabilityResult>(
            broken(BrokenDemandContractReason::InterfaceContradiction,
                   RelationOperationKind::ReductionCompletion,
                   "attention root has no coupled reduction facts", fact.root));
      const bool partitionsKeyValue = llvm::any_of(
          fact.coupledReduction->reductionIterators,
          [&](unsigned iterator) { return intervalCounts[iterator] > 1; });
      if (attention.getAlgorithm() == wafer::AttentionAlgorithm::FlashAttention
              ? partitionsKeyValue
              : !partitionsKeyValue)
        return asResult<AvailabilityResult>(invalidAssignment(
            InvalidSpatialAssignmentReason::ReductionGroup,
            attention.getAlgorithm() ==
                    wafer::AttentionAlgorithm::FlashAttention
                ? "flash attention assignment partitions K2"
                : "flash decoding assignment leaves K2 unpartitioned",
            fact.root));
    }

    if (hasSpatialReduction && fact.coupledReduction) {
      const bool hasUnmodeledReduction =
          llvm::any_of(llvm::enumerate(fact.iteratorTypes), [&](auto entry) {
            const unsigned iterator = entry.index();
            return entry.value() == mlir::utils::IteratorType::reduction &&
                   intervalCounts[iterator] > 1 &&
                   !llvm::is_contained(
                       fact.coupledReduction->reductionIterators, iterator);
          });
      if (hasUnmodeledReduction)
        return asResult<AvailabilityResult>(unsupported(
            UnsupportedDemandReason::MissingReductionAlgebra,
            RelationOperationKind::ReductionCompletion,
            "spatial reduction is outside the coupled component algebra",
            fact.root));
    }

    for (const StructuredResultFact &resultFact : fact.results) {
      struct ShardImage {
        const ExecutionShard *shard = nullptr;
        ExactIndexSet iteration;
        ExactIndexSet result;
      };
      llvm::SmallVector<ShardImage, 16> images;
      for (const ExecutionShard &shard : node->shards) {
        DemandResult<ExactIndexSet> iteration = makeShardSet(shard, limits);
        if (!getValue(iteration))
          return asResult<AvailabilityResult>(getFailure(std::move(iteration)));
        DemandResult<ExactIndexSet> image = imageExactSet(
            resultFact.iterationToResult, *getValue(iteration), limits,
            RelationOperationKind::ResultAvailability, fact.root);
        if (!getValue(image))
          return asResult<AvailabilityResult>(getFailure(std::move(image)));
        images.push_back({&shard, std::move(*getValue(iteration)),
                          std::move(*getValue(image))});
      }

      if (!hasSpatialReduction) {
        for (size_t lhs = 0; lhs < images.size(); ++lhs)
          for (size_t rhs = lhs + 1; rhs < images.size(); ++rhs)
            for (const StaticRectangularIndexSet &lhsBox :
                 images[lhs].result.getBoxes())
              for (const StaticRectangularIndexSet &rhsBox :
                   images[rhs].result.getBoxes())
                if (boxesOverlap(lhsBox, rhsBox))
                  return asResult<AvailabilityResult>(invalidAssignment(
                      InvalidSpatialAssignmentReason::ResultCoverage,
                      "non-reduction result owners overlap", fact.root));
        for (ShardImage &image : images)
          if (!image.result.isEmpty())
            result.finalOwners.push_back(
                {fact.root, resultFact.result, image.shard->shard, std::nullopt,
                 image.shard->tile, std::move(image.result)});
        continue;
      }

      auto partial =
          mlir::dyn_cast<mlir::PartialReductionOpInterface>(fact.operation);
      auto coupled = mlir::dyn_cast<wafer::WaferCoupledReductionOpInterface>(
          fact.operation);
      if (!partial && !coupled)
        return asResult<AvailabilityResult>(unsupported(
            UnsupportedDemandReason::MissingReductionAlgebra,
            RelationOperationKind::ReductionCompletion,
            "spatial reduction lacks typed partial mechanics", fact.root));
      if (coupled && (!fact.coupledReduction || fact.results.size() != 1))
        return asResult<AvailabilityResult>(broken(
            BrokenDemandContractReason::InterfaceContradiction,
            RelationOperationKind::ReductionCompletion,
            "coupled reduction requires one final result and complete source "
            "facts",
            fact.root));

      std::map<std::vector<int64_t>, llvm::SmallVector<ShardImage *, 8>> groups;
      for (ShardImage &image : images) {
        if (image.result.getForm() != ExactIndexSetForm::BoxUnion)
          return asResult<AvailabilityResult>(unsupported(
              UnsupportedDemandReason::MissingReductionAlgebra,
              RelationOperationKind::ReductionCompletion,
              "spatial reduction result requires finite output boxes",
              fact.root));
        groups[boxKey(image.result)].push_back(&image);
      }
      for (auto &[key, contributions] : groups) {
        (void)key;
        ShardImage &representative = *contributions.front();
        ReductionGroupId group;
        group.root = fact.root;
        group.resultGroup = resultFact.result;
        for (auto [iterator, type] : llvm::enumerate(fact.iteratorTypes))
          if (type == mlir::utils::IteratorType::parallel)
            group.parallelCoordinate.push_back(
                representative.shard->shard.coordinate[iterator]);
        auto placement =
            llvm::find_if(node->reductionGroups,
                          [&](const ReductionGroupPlacement &candidate) {
                            return candidate.group == group;
                          });
        if (placement == node->reductionGroups.end())
          return asResult<AvailabilityResult>(invalidAssignment(
              InvalidSpatialAssignmentReason::ReductionGroup,
              "spatial reduction omits an output-group merge placement",
              fact.root));

        ReductionMergeRequirement requirement;
        requirement.group = group;
        requirement.mergeTile = placement->mergeTile;
        requirement.results.push_back(
            {resultFact.result, representative.result});
        if (coupled) {
          if (contributions.size() < 2)
            return asResult<AvailabilityResult>(invalidAssignment(
                InvalidSpatialAssignmentReason::ReductionGroup,
                "coupled reduction group has fewer than two contributions",
                fact.root));
          requirement.initialization =
              ReductionInitialization::CoupledIdentityPerContribution;
          requirement.algebra = ReductionAlgebraKind::CoupledReduction;
          requirement.coupledRule = fact.coupledReduction->rule;
        } else {
          requirement.algebra = ReductionAlgebraKind::StandardPartialReduction;
        }
        for (auto [contributionIndex, contribution] :
             llvm::enumerate(contributions)) {
          ReductionContribution item;
          item.shard = contribution->shard->shard;
          item.tile = contribution->shard->tile;
          item.iterationDomain = contribution->iteration;
          if (!coupled) {
            item.results.push_back({resultFact.result, contribution->result});
          } else {
            for (auto [componentIndex, component] :
                 llvm::enumerate(fact.coupledReduction->components)) {
              DemandResult<ExactIndexSet> componentDomain = imageExactSet(
                  component.iterationToComponent, contribution->iteration,
                  limits, RelationOperationKind::ReductionCompletion,
                  fact.root);
              if (!getValue(componentDomain))
                return asResult<AvailabilityResult>(
                    getFailure(std::move(componentDomain)));
              if (getValue(componentDomain)->getForm() !=
                  ExactIndexSetForm::BoxUnion)
                return asResult<AvailabilityResult>(unsupported(
                    UnsupportedDemandReason::MissingReductionAlgebra,
                    RelationOperationKind::ReductionCompletion,
                    "coupled reduction component requires finite output "
                    "boxes",
                    fact.root));
              if (contributionIndex == 0) {
                requirement.components.push_back({component.kind, component.map,
                                                  component.elementType,
                                                  *getValue(componentDomain)});
              } else if (!haveSameBoxUnion(
                             requirement.components[componentIndex].domain,
                             *getValue(componentDomain))) {
                return asResult<AvailabilityResult>(invalidAssignment(
                    InvalidSpatialAssignmentReason::ReductionGroup,
                    "coupled contributions disagree on component domain",
                    fact.root));
              }
              item.components.push_back(
                  {component.kind, std::move(*getValue(componentDomain))});
            }
          }
          requirement.contributions.push_back(std::move(item));
        }
        result.finalOwners.push_back({fact.root, resultFact.result,
                                      std::nullopt, group, placement->mergeTile,
                                      representative.result});
        result.reductionMerges.push_back(std::move(requirement));
      }
    }
    const size_t derivedMergeCount =
        llvm::count_if(result.reductionMerges,
                       [&](const ReductionMergeRequirement &requirement) {
                         return requirement.group.root == fact.root;
                       });
    if (derivedMergeCount != node->reductionGroups.size())
      return asResult<AvailabilityResult>(invalidAssignment(
          InvalidSpatialAssignmentReason::ReductionGroup,
          "spatial assignment has missing or unused reduction groups",
          fact.root));
  }
  for (ReductionMergeRequirement &requirement : result.reductionMerges) {
    llvm::sort(requirement.contributions, [](const ReductionContribution &lhs,
                                             const ReductionContribution &rhs) {
      if (lhs.shard != rhs.shard)
        return lhs.shard < rhs.shard;
      return lhs.tile.getValue() < rhs.tile.getValue();
    });
    llvm::sort(requirement.results, [](const ReductionResultSlice &lhs,
                                       const ReductionResultSlice &rhs) {
      return lhs.result < rhs.result;
    });
  }
  llvm::sort(result.reductionMerges, [](const ReductionMergeRequirement &lhs,
                                        const ReductionMergeRequirement &rhs) {
    if (lhs.group != rhs.group)
      return lhs.group < rhs.group;
    return lhs.mergeTile.getValue() < rhs.mergeTile.getValue();
  });
  llvm::sort(result.finalOwners, finalOwnerLess);
  return result;
}

DemandResult<SourceDemand>
buildSourceDemand(const DemandSource &source, const ExactIndexSet &required,
                  llvm::ArrayRef<FinalResultOwner> owners,
                  const IndexRelationLimits &limits) {
  SourceDemand result;
  result.source = source;
  result.requiredDomain = required;
  if (const auto *structured =
          std::get_if<StructuredResultSource>(&result.source)) {
    for (const FinalResultOwner &owner : owners) {
      if (owner.root != structured->root || owner.result != structured->result)
        continue;
      DemandResult<ExactIndexSet> intersection = intersectExactSets(
          required, owner.domain, limits, RelationOperationKind::Intersection,
          structured->root);
      if (!getValue(intersection))
        return asResult<SourceDemand>(getFailure(std::move(intersection)));
      if (!getValue(intersection)->isEmpty())
        result.eligibleFinalOwners.push_back(
            {owner.shard, owner.reductionGroup, owner.tile,
             std::move(*getValue(intersection))});
    }
    if (!required.isEmpty() && result.eligibleFinalOwners.empty())
      return asResult<SourceDemand>(invalidAssignment(
          InvalidSpatialAssignmentReason::ResultCoverage,
          "structured boundary demand has no final owner", structured->root));
  }
  llvm::sort(result.eligibleFinalOwners,
             [](const OwnerIntersection &lhs, const OwnerIntersection &rhs) {
               if (lhs.reductionGroup != rhs.reductionGroup)
                 return lhs.reductionGroup < rhs.reductionGroup;
               if (lhs.ownerShard != rhs.ownerShard)
                 return lhs.ownerShard < rhs.ownerShard;
               return lhs.tile.getValue() < rhs.tile.getValue();
             });
  return result;
}

} // namespace

class StructuredRelationFacts::Impl {
public:
  StructuredDAGAnalysis dag;
  SemanticRootAnalysis semanticRoots;
  llvm::SmallVector<StructuredOperationFact, 16> operations;
  std::vector<ResultRelationFact> reverseResults;

  Impl(StructuredDAGAnalysis dag, SemanticRootAnalysis semanticRoots,
       llvm::SmallVector<StructuredOperationFact, 16> operations,
       std::vector<ResultRelationFact> reverseResults)
      : dag(std::move(dag)), semanticRoots(std::move(semanticRoots)),
        operations(std::move(operations)),
        reverseResults(std::move(reverseResults)) {}
};

StructuredRelationFacts::StructuredRelationFacts(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

StructuredRelationFacts::~StructuredRelationFacts() = default;
StructuredRelationFacts::StructuredRelationFacts(
    StructuredRelationFacts &&) noexcept = default;
StructuredRelationFacts &StructuredRelationFacts::operator=(
    StructuredRelationFacts &&) noexcept = default;

mlir::FailureOr<StructuredRelationFacts>
StructuredRelationFacts::create(const StructuredDAGAnalysis &dag,
                                const IndexRelationLimits &limits,
                                std::string *failureReason) {
  mlir::FailureOr<SemanticRootAnalysis> roots =
      SemanticRootAnalysis::create(dag, failureReason);
  if (mlir::failed(roots))
    return mlir::failure();
  llvm::SmallVector<StructuredOperationFact, 16> operations;
  for (const SemanticRootBinding &binding : roots->getRoots()) {
    DemandResult<StructuredOperationFact> fact =
        deriveStructuredOperationFact(binding, limits);
    if (!getValue(fact)) {
      if (failureReason)
        *failureReason =
            std::visit([](const auto &failure) { return failure.detail; },
                       getFailure(std::move(fact)));
      return mlir::failure();
    }
    operations.push_back(std::move(*getValue(fact)));
  }

  std::vector<ResultRelationFact> reverseResults;
  mlir::Block &body = dag.getFunction().getBody().front();
  for (mlir::Operation &operation : llvm::reverse(body.without_terminator())) {
    for (mlir::OpResult result : operation.getResults()) {
      ResultRelationFact relation(result);
      if (!roots->find(&operation) &&
          !operation.hasTrait<mlir::OpTrait::ConstantLike>() &&
          !mlir::isa<mlir::tensor::EmptyOp>(&operation)) {
        DemandResult<SupportTransferFact> transfer =
            deriveSupportTransfer(result, limits);
        if (getValue(transfer))
          relation.transfer.emplace(std::move(*getValue(transfer)));
        else
          relation.failure.emplace(getFailure(std::move(transfer)));
      }
      reverseResults.push_back(std::move(relation));
    }
  }
  return StructuredRelationFacts(
      std::make_unique<Impl>(dag, std::move(*roots), std::move(operations),
                             std::move(reverseResults)));
}

const StructuredDAGAnalysis &StructuredRelationFacts::getDAG() const {
  return impl->dag;
}

const SemanticRootAnalysis &StructuredRelationFacts::getSemanticRoots() const {
  return impl->semanticRoots;
}

StructuredRelationAnalysis::StructuredRelationAnalysis(
    mlir::Operation *operation) {
  auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(operation);
  if (!function) {
    failureReason = "structured relation analysis requires func.func";
    return;
  }
  mlir::FailureOr<StructuredDAGAnalysis> dag =
      StructuredDAGAnalysis::create(function, &failureReason);
  if (mlir::failed(dag))
    return;
  mlir::FailureOr<StructuredRelationFacts> built =
      StructuredRelationFacts::create(*dag, IndexRelationLimits(),
                                      &failureReason);
  if (mlir::succeeded(built))
    facts.emplace(std::move(*built));
}

analysis::ExactDemandOutcome
deriveExactDemand(const StructuredRelationFacts &facts,
                  const SpatialAssignment &assignment,
                  const IndexRelationLimits &limits) {
  llvm::SmallVector<NodeIterationSpace, 16> iterationSpaces;
  llvm::SmallVector<TileId, 16> tiles;
  for (const StructuredOperationFact &fact : facts.impl->operations) {
    NodeIterationSpace space;
    space.root = fact.root;
    space.iteratorExtents.assign(fact.iterationShape.begin(),
                                 fact.iterationShape.end());
    iterationSpaces.push_back(std::move(space));
  }
  for (const NodeExecutionPartition &node : assignment.nodes) {
    for (const ExecutionShard &shard : node.shards)
      if (!llvm::is_contained(tiles, shard.tile))
        tiles.push_back(shard.tile);
    for (const ReductionGroupPlacement &group : node.reductionGroups)
      if (!llvm::is_contained(tiles, group.mergeTile))
        tiles.push_back(group.mergeTile);
  }
  std::string assignmentFailure;
  mlir::FailureOr<SpatialPlanningProblem> problem =
      SpatialPlanningProblem::create(iterationSpaces, tiles,
                                     &assignmentFailure);
  if (mlir::failed(problem) || mlir::failed(validateSpatialAssignmentStructure(
                                   *problem, assignment, &assignmentFailure)))
    return InvalidSpatialAssignment{
        InvalidSpatialAssignmentReason::ExecutionPartition,
        DemandFailureSite{std::nullopt, std::nullopt, std::nullopt,
                          std::nullopt,
                          RelationOperationKind::ValidateAssignment},
        assignmentFailure};

  DemandResult<AvailabilityResult> availability =
      deriveResultAvailability(facts.impl->operations, assignment, limits);
  if (!getValue(availability))
    return takeFailure(std::move(availability));

  struct SeedInfo {
    const StructuredOperationFact *consumer = nullptr;
    const StructuredOperandFact *operand = nullptr;
    const ExecutionShard *destination = nullptr;
    ExactIndexSet execution;
    ExactIndexSet operandDemand;
  };
  std::map<DemandKey, SeedInfo> seeds;
  llvm::DenseMap<mlir::Value, QueryState> state;
  for (const StructuredOperationFact &consumer : facts.impl->operations) {
    const NodeExecutionPartition *node =
        findNodeAssignment(assignment, consumer.root);
    for (const StructuredOperandFact &operand : consumer.operands) {
      for (const ExecutionShard &destination : node->shards) {
        DemandResult<ExactIndexSet> execution =
            makeShardSet(destination, limits);
        if (!getValue(execution))
          return takeFailure(std::move(execution));
        DemandResult<ExactIndexSet> demand =
            imageExactSet(operand.iterationToOperand, *getValue(execution),
                          limits, RelationOperationKind::Image, consumer.root);
        if (!getValue(demand))
          return takeFailure(std::move(demand));
        DemandKey key{consumer.root, operand.operand, destination.shard};
        seeds.emplace(key, SeedInfo{&consumer, &operand, &destination,
                                    *getValue(execution), *getValue(demand)});
        state[consumer.operation->getOperand(operand.operand)].demands.emplace(
            key, std::move(*getValue(demand)));
      }
    }
  }

  std::map<DemandKey, llvm::SmallVector<BoundaryState, 4>> boundaries;
  std::map<DemandKey, std::vector<TensorTransform>> reconstructions;
  mlir::Block &body = facts.impl->dag.getFunction().getBody().front();
  for (const ResultRelationFact &relation : facts.impl->reverseResults) {
    mlir::OpResult result = relation.result;
    mlir::Operation &operation = *result.getOwner();
    auto current = state.find(result);
    if (current == state.end())
      continue;
    // Propagation inserts earlier SSA values into the DenseMap and may rehash
    // it. Keep the current value state outside the map before any insertion so
    // all DemandKeys are propagated exactly once.
    QueryState currentState = current->second;
    if (const SemanticRootBinding *root =
            facts.impl->semanticRoots.find(&operation)) {
      for (auto &[key, demand] : currentState.demands)
        boundaries[key].push_back(
            {StructuredResultSource{
                 root->key, &operation,
                 static_cast<uint32_t>(result.getResultNumber())},
             demand});
      continue;
    }
    if (operation.hasTrait<mlir::OpTrait::ConstantLike>()) {
      const SemanticValueBinding *value =
          facts.impl->semanticRoots.find(result);
      if (!value)
        return BrokenDemandContract{
            BrokenDemandContractReason::InterfaceContradiction,
            DemandFailureSite{std::nullopt,
                              static_cast<uint32_t>(result.getResultNumber()),
                              std::nullopt, std::nullopt,
                              RelationOperationKind::BuildRelationGraph},
            "constant boundary has no semantic value path"};
      for (auto &[key, demand] : currentState.demands)
        boundaries[key].push_back(
            {ConstantSource{value->key, &operation,
                            static_cast<uint32_t>(result.getResultNumber())},
             demand});
      continue;
    }
    if (mlir::isa<mlir::tensor::EmptyOp>(&operation)) {
      bool allEmpty = llvm::all_of(currentState.demands, [](const auto &v) {
        return v.second.isEmpty();
      });
      if (!allEmpty)
        return UnsupportedDemandSemantics{
            UnsupportedDemandReason::MissingTensorTransfer,
            DemandFailureSite{std::nullopt,
                              static_cast<uint32_t>(result.getResultNumber()),
                              std::nullopt, std::nullopt,
                              RelationOperationKind::BuildRelationGraph},
            "payload reads an uninitialized tensor.empty boundary"};
      continue;
    }

    if (relation.failure)
      return getFailureOutcome(*relation.failure);
    if (!relation.transfer)
      return BrokenDemandContract{
          BrokenDemandContractReason::InterfaceContradiction,
          DemandFailureSite{std::nullopt,
                            static_cast<uint32_t>(result.getResultNumber()),
                            std::nullopt, std::nullopt,
                            RelationOperationKind::BuildRelationGraph},
          "relation fact has neither a boundary nor a transfer"};
    const SupportTransferFact &transfer = *relation.transfer;
    for (auto &[key, demand] : currentState.demands) {
      TensorTransform step;
      step.operation = &operation;
      step.result = static_cast<uint32_t>(result.getResultNumber());
      step.kind = transfer.kind;
      step.outputDemand = demand;
      for (const SupportOperandFact &operand : transfer.operands) {
        DemandResult<ExactIndexSet> mapped =
            imageSupportOperand(transfer, operand, demand, limits);
        if (!getValue(mapped))
          return takeFailure(std::move(mapped));
        step.operandDemands.push_back({operand.operand, *getValue(mapped)});
        mlir::Value operandValue = operation.getOperand(operand.operand);
        auto existing = state[operandValue].demands.find(key);
        if (existing == state[operandValue].demands.end()) {
          state[operandValue].demands.emplace(key,
                                              std::move(*getValue(mapped)));
        } else {
          DemandResult<ExactIndexSet> joined =
              unionExactSets(existing->second, *getValue(mapped), limits,
                             RelationOperationKind::Union, key.consumer);
          if (!getValue(joined))
            return takeFailure(std::move(joined));
          existing->second = std::move(*getValue(joined));
        }
      }
      reconstructions[key].push_back(std::move(step));
    }
  }

  for (mlir::BlockArgument argument : body.getArguments()) {
    auto current = state.find(argument);
    if (current == state.end())
      continue;
    for (auto &[key, demand] : current->second.demands)
      boundaries[key].push_back(
          {ProgramInputSource{static_cast<uint32_t>(argument.getArgNumber())},
           demand});
  }

  ExactDemandProof proof;
  proof.finalOwners = std::move(getValue(availability)->finalOwners);
  proof.reductionMerges = std::move(getValue(availability)->reductionMerges);
  std::optional<std::pair<SemanticRootKey, uint32_t>> currentDependency;
  for (auto &[key, seed] : seeds) {
    if (!currentDependency || currentDependency->first != key.consumer ||
        currentDependency->second != key.operand) {
      DependencyDemand dependency;
      dependency.consumer = key.consumer;
      dependency.consumerOperation = seed.consumer->operation;
      dependency.consumerOperand = key.operand;
      dependency.kind = seed.operand->kind;
      proof.dependencyDemands.push_back(std::move(dependency));
      currentDependency = std::make_pair(key.consumer, key.operand);
    }
    DestinationDemand destination;
    destination.destinationShard = key.destination;
    destination.destinationTile = seed.destination->tile;
    destination.consumerExecutionDomain = seed.execution;
    destination.operandDemand = seed.operandDemand;
    auto boundary = boundaries.find(key);
    if (boundary == boundaries.end())
      return BrokenDemandContract{
          BrokenDemandContractReason::InternalExactnessFailure,
          DemandFailureSite{key.consumer, std::nullopt, key.operand,
                            key.destination,
                            RelationOperationKind::BuildRelationGraph},
          "consumer operand demand reaches no typed source boundary"};
    for (const BoundaryState &source : boundary->second) {
      DemandResult<SourceDemand> demand = buildSourceDemand(
          source.source, source.demand, proof.finalOwners, limits);
      if (!getValue(demand))
        return takeFailure(std::move(demand));
      destination.sources.push_back(std::move(*getValue(demand)));
    }
    llvm::sort(destination.sources,
               [](const SourceDemand &lhs, const SourceDemand &rhs) {
                 return demandSourceLess(lhs.source, rhs.source);
               });
    auto reconstruction = reconstructions.find(key);
    if (reconstruction != reconstructions.end()) {
      std::reverse(reconstruction->second.begin(),
                   reconstruction->second.end());
      destination.reconstruction.steps = std::move(reconstruction->second);
    }
    proof.dependencyDemands.back().perDestination.push_back(
        std::move(destination));
  }
  return proof;
}

class DemandPlanningSession::Cache {
public:
  std::map<std::vector<int64_t>, ExactDemandOutcome> outcomes;
};

mlir::FailureOr<DemandPlanningSession>
DemandPlanningSession::create(const StructuredDAGAnalysis &dag,
                              const IndexRelationLimits &limits,
                              std::string *failureReason) {
  mlir::FailureOr<StructuredRelationFacts> facts =
      StructuredRelationFacts::create(dag, limits, failureReason);
  if (mlir::failed(facts))
    return mlir::failure();
  return DemandPlanningSession(std::move(*facts), limits);
}

DemandPlanningSession::DemandPlanningSession(StructuredRelationFacts facts,
                                             IndexRelationLimits limits)
    : facts(std::move(facts)), limits(limits),
      cache(std::make_unique<Cache>()) {}

DemandPlanningSession::DemandPlanningSession(
    DemandPlanningSession &&) noexcept = default;
DemandPlanningSession &
DemandPlanningSession::operator=(DemandPlanningSession &&) noexcept = default;
DemandPlanningSession::~DemandPlanningSession() = default;

ExactDemandOutcome
DemandPlanningSession::query(const SpatialAssignment &assignment) {
  if (closed)
    return BrokenDemandContract{
        BrokenDemandContractReason::InterfaceContradiction,
        DemandFailureSite{std::nullopt, std::nullopt, std::nullopt,
                          std::nullopt,
                          RelationOperationKind::BuildRelationGraph},
        "exact-demand planning session is closed"};
  std::vector<int64_t> key = getSpatialAssignmentKey(assignment);
  auto cached = cache->outcomes.find(key);
  if (cached != cache->outcomes.end())
    return cached->second;
  ExactDemandOutcome outcome = deriveExactDemand(facts, assignment, limits);
  if (std::holds_alternative<ExactDemandProof>(outcome) ||
      std::holds_alternative<UnsupportedDemandSemantics>(outcome))
    cache->outcomes.emplace(std::move(key), outcome);
  return outcome;
}

void DemandPlanningSession::close() {
  cache->outcomes.clear();
  closed = true;
}

} // namespace wafer::compiler::detail
