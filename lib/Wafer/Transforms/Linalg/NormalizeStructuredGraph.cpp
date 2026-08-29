//===- NormalizeStructuredGraph.cpp - Relation-driven normalization -----===//

#include "Wafer/Transforms/Linalg/StructuredGraphNormalization.h"

#include "StructuredGraphEGraph.h"
#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>

namespace wafer {

#define GEN_PASS_DEF_NORMALIZESTRUCTUREDTENSORGRAPHPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

using analysis::IndexRelation;
using analysis::IndexRelationQueryResult;
using analysis::IndexRelationResult;
using analysis::IndexRelationStatus;
using structured_graph_normalization::EGraphNode;
using structured_graph_normalization::EGraphNodeKind;
using structured_graph_normalization::EGraphOutcome;
using structured_graph_normalization::EGraphOutcomeKind;
using structured_graph_normalization::EGraphWorkBudget;

enum class RelationMaterialization {
  None,
  Identity,
  ProjectedMap,
  Reshape,
};

class TypeStore {
public:
  uint32_t intern(mlir::Type type) {
    auto [entry, inserted] = ids.try_emplace(type, values.size() + 1);
    if (inserted)
      values.push_back(type);
    return entry->second;
  }

  mlir::Type lookup(uint32_t id) const {
    return id == 0 || id > values.size() ? mlir::Type{} : values[id - 1];
  }

  mlir::RankedTensorType lookupRankedTensor(uint32_t id) const {
    return mlir::dyn_cast_or_null<mlir::RankedTensorType>(lookup(id));
  }

private:
  llvm::DenseMap<mlir::Type, uint32_t> ids;
  llvm::SmallVector<mlir::Type, 32> values;
};

bool canMaterializeStaticReshape(mlir::RankedTensorType source,
                                 mlir::RankedTensorType destination) {
  if (!source || !destination || !source.hasStaticShape() ||
      !destination.hasStaticShape() ||
      source.getElementType() != destination.getElementType() ||
      source.getEncoding() != destination.getEncoding() ||
      source.getNumElements() != destination.getNumElements())
    return false;
  return mlir::getReassociationIndicesForReshape(source, destination)
      .has_value();
}

struct RelationRecord {
  IndexRelation relation;
  uint32_t destinationTypeId = 0;
  uint32_t sourceTypeId = 0;
  std::optional<mlir::AffineMap> projectedMap;
  RelationMaterialization materialization = RelationMaterialization::None;
  bool total = false;
  bool singleValued = false;
  bool injective = false;
  bool bijective = false;
  bool identity = false;
};

class RelationStore {
public:
  RelationStore(TypeStore &types, mlir::MLIRContext *context,
                uint64_t maximumQueries)
      : types(types), context(context), maximumQueries(maximumQueries) {}

  std::optional<uint32_t>
  intern(IndexRelation relation, uint32_t destinationTypeId,
         uint32_t sourceTypeId, std::optional<mlir::AffineMap> projectedMap,
         RelationMaterialization materialization, bool total) {
    if (!destinationTypeId || !sourceTypeId ||
        relation.getStatus() != IndexRelationStatus::Exact || !total ||
        (materialization == RelationMaterialization::ProjectedMap &&
         !projectedMap) ||
        (materialization != RelationMaterialization::None &&
         materialization != RelationMaterialization::ProjectedMap &&
         materialization != RelationMaterialization::Reshape))
      return std::nullopt;
    if (materialization == RelationMaterialization::Reshape &&
        destinationTypeId != sourceTypeId &&
        !canMaterializeStaticReshape(
            types.lookupRankedTensor(sourceTypeId),
            types.lookupRankedTensor(destinationTypeId)))
      return std::nullopt;
    const bool identity =
        destinationTypeId == sourceTypeId &&
        (materialization == RelationMaterialization::Reshape ||
         (projectedMap && projectedMap->isIdentity()));
    const RelationMaterialization canonicalMaterialization =
        identity ? RelationMaterialization::Identity : materialization;
    for (auto [index, existing] : llvm::enumerate(records)) {
      if (canonicalMaterialization == RelationMaterialization::None)
        break;
      if (existing.destinationTypeId != destinationTypeId ||
          existing.sourceTypeId != sourceTypeId ||
          existing.materialization != canonicalMaterialization)
        continue;
      if (canonicalMaterialization == RelationMaterialization::Identity ||
          materialization == RelationMaterialization::Reshape ||
          existing.projectedMap == projectedMap)
        return index + 1;
    }

    RelationRecord record{std::move(relation), destinationTypeId, sourceTypeId,
                          projectedMap};
    record.total = true;
    record.singleValued = true;
    record.materialization = canonicalMaterialization;
    record.identity = identity;
    if (record.identity) {
      record.materialization = RelationMaterialization::Identity;
      record.injective = true;
      record.bijective = true;
    } else if (materialization == RelationMaterialization::Reshape) {
      // staticReshape is the canonical row-major bijection between two
      // equal-element-count static tensor types.  This is a construction
      // fact, not a generic Presburger query.
      record.injective = true;
      record.bijective = true;
    } else if (materialization == RelationMaterialization::ProjectedMap) {
      IndexRelationQueryResult injective = record.relation.isInjective();
      if (injective.status == IndexRelationStatus::ResourceExhausted) {
        workLimitReached = true;
        return std::nullopt;
      }
      record.injective = injective.isProvenTrue();
      auto destination = types.lookupRankedTensor(destinationTypeId);
      auto source = types.lookupRankedTensor(sourceTypeId);
      record.bijective =
          record.injective && destination && source &&
          destination.getNumElements() == source.getNumElements();
    }
    records.push_back(std::move(record));
    return records.size();
  }

  const RelationRecord *lookup(uint32_t id) const {
    return id == 0 || id > records.size() ? nullptr : &records[id - 1];
  }

  RelationRecord *lookup(uint32_t id) {
    return id == 0 || id > records.size() ? nullptr : &records[id - 1];
  }

  bool consumeQuery() {
    if (queryCount >= maximumQueries) {
      workLimitReached = true;
      return false;
    }
    ++queryCount;
    return true;
  }

  uint64_t getQueryCount() const { return queryCount; }
  bool hasWorkLimitReached() const { return workLimitReached; }

  uint32_t getFacts(uint32_t relationId, WaferEGraphRelationFacts *facts) {
    if (!consumeQuery())
      return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
    const RelationRecord *record = lookup(relationId);
    if (!record || !facts)
      return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
    uint32_t flags = 0;
    if (record->identity)
      flags |= WAFER_EGRAPH_RELATION_IDENTITY;
    if (record->total)
      flags |= WAFER_EGRAPH_RELATION_TOTAL;
    if (record->singleValued)
      flags |= WAFER_EGRAPH_RELATION_SINGLE_VALUED;
    if (record->injective)
      flags |= WAFER_EGRAPH_RELATION_INJECTIVE;
    if (record->bijective)
      flags |= WAFER_EGRAPH_RELATION_BIJECTIVE;
    if (record->materialization != RelationMaterialization::None)
      flags |= WAFER_EGRAPH_RELATION_MATERIALIZABLE;
    *facts = WaferEGraphRelationFacts{record->destinationTypeId,
                                      record->sourceTypeId, flags, 0};
    return WAFER_EGRAPH_CALLBACK_EXACT;
  }

  uint32_t compose(uint32_t outerId, uint32_t innerId, uint32_t *resultId) {
    if (!consumeQuery())
      return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
    const RelationRecord *outer = lookup(outerId);
    const RelationRecord *inner = lookup(innerId);
    if (!outer || !inner || !resultId)
      return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
    if (outer->sourceTypeId != inner->destinationTypeId)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    const uint64_t cacheKey = (uint64_t{outerId} << 32) | innerId;
    if (auto found = compositionCache.find(cacheKey);
        found != compositionCache.end()) {
      if (found->second == 0)
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      *resultId = found->second;
      return WAFER_EGRAPH_CALLBACK_EXACT;
    }
    if (outer->identity) {
      compositionCache.try_emplace(cacheKey, innerId);
      *resultId = innerId;
      return WAFER_EGRAPH_CALLBACK_EXACT;
    }
    if (inner->identity) {
      compositionCache.try_emplace(cacheKey, outerId);
      *resultId = outerId;
      return WAFER_EGRAPH_CALLBACK_EXACT;
    }

    std::optional<uint32_t> id;
    if (outer->materialization == RelationMaterialization::Reshape &&
        inner->materialization == RelationMaterialization::Reshape) {
      auto destination = types.lookupRankedTensor(outer->destinationTypeId);
      auto source = types.lookupRankedTensor(inner->sourceTypeId);
      if (destination && source) {
        IndexRelationResult reshape = IndexRelation::staticReshape(
            destination.getShape(), source.getShape());
        if (reshape.isExact()) {
          const RelationMaterialization materialization =
              outer->destinationTypeId == inner->sourceTypeId ||
                      canMaterializeStaticReshape(source, destination)
                  ? RelationMaterialization::Reshape
                  : RelationMaterialization::None;
          id = intern(std::move(*reshape.get()), outer->destinationTypeId,
                      inner->sourceTypeId, std::nullopt, materialization,
                      /*total=*/true);
        }
      }
    } else if (outer->materialization == RelationMaterialization::Reshape ||
               inner->materialization == RelationMaterialization::Reshape) {
      // A projected map composed with a general row-major reshape normally
      // has no single Tensor/Linalg materialization form.  Do not invoke the
      // generic Presburger composition for this optional alternative: an
      // adjacent reshape pair can still close to identity in its own e-class,
      // after which congruence exposes the useful projected-map rewrite.
      compositionCache.try_emplace(cacheKey, 0);
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    } else {
      IndexRelationResult composed = outer->relation.compose(inner->relation);
      if (!composed.isExact()) {
        compositionCache.try_emplace(cacheKey, 0);
        return statusFor(composed.status);
      }
      std::optional<mlir::AffineMap> projected;
      if (outer->projectedMap && inner->projectedMap)
        projected = inner->projectedMap->compose(*outer->projectedMap);
      else
        projected = composed.get()->getProjectedAffineMap(context);
      if (projected) {
        auto destination = types.lookupRankedTensor(outer->destinationTypeId);
        auto source = types.lookupRankedTensor(inner->sourceTypeId);
        if (destination && source) {
          IndexRelationResult canonical = IndexRelation::fromAffineMap(
              *projected, destination.getShape(), source.getShape());
          if (canonical.isExact() &&
              canonical.get()->hasTotalBoundedAffineMapConstruction())
            id = intern(std::move(*canonical.get()), outer->destinationTypeId,
                        inner->sourceTypeId, projected,
                        RelationMaterialization::ProjectedMap,
                        /*total=*/true);
        }
      }
      if (!id && !workLimitReached)
        id = intern(std::move(*composed.get()), outer->destinationTypeId,
                    inner->sourceTypeId, std::nullopt,
                    RelationMaterialization::None, /*total=*/true);
    }
    if (!id)
      compositionCache.try_emplace(cacheKey, 0);
    if (!id)
      return workLimitReached ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                              : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    compositionCache.try_emplace(cacheKey, *id);
    *resultId = *id;
    return WAFER_EGRAPH_CALLBACK_EXACT;
  }

private:
  static uint32_t statusFor(IndexRelationStatus status) {
    switch (status) {
    case IndexRelationStatus::ResourceExhausted:
      return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
    case IndexRelationStatus::Unsupported:
    case IndexRelationStatus::SoundBound:
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    case IndexRelationStatus::Invalid:
      return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
    case IndexRelationStatus::Exact:
      return WAFER_EGRAPH_CALLBACK_EXACT;
    }
    return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
  }

  TypeStore &types;
  mlir::MLIRContext *context = nullptr;
  uint64_t maximumQueries = 0;
  uint64_t queryCount = 0;
  bool workLimitReached = false;
  llvm::SmallVector<RelationRecord, 32> records;
  llvm::DenseMap<uint64_t, uint32_t> compositionCache;
};

struct AccessDescription {
  mlir::Value source;
  IndexRelation relation;
  std::optional<mlir::AffineMap> projectedMap;
  RelationMaterialization materialization = RelationMaterialization::None;
};

struct ConcatDescription {
  mlir::RankedTensorType resultType;
  int64_t axis = 0;
  llvm::SmallVector<mlir::Value, 4> inputs;
};

struct ComputeRecipe {
  mlir::Operation *operation = nullptr;
  EGraphNodeKind kind = EGraphNodeKind::Elementwise;
  mlir::RankedTensorType resultType;
  uint32_t resultTypeId = 0;
  uint32_t iterationTypeId = 0;
  uint32_t dataInputCount = 0;
  llvm::SmallVector<mlir::AffineMap, 4> originalMaps;
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes;
  bool initIsRead = true;
  bool hasCanonicalContractionPayload = false;
};

class InsertedOperationRecorder final : public mlir::RewriterBase::Listener {
public:
  InsertedOperationRecorder(mlir::Block *componentBlock,
                            mlir::OpBuilder::Listener *previous,
                            llvm::SmallVectorImpl<mlir::Operation *> &inserted)
      : componentBlock(componentBlock), previous(previous), inserted(inserted) {
  }

  void notifyOperationInserted(mlir::Operation *operation,
                               mlir::OpBuilder::InsertPoint oldPoint) final {
    if (operation->getBlock() == componentBlock && !oldPoint.isSet())
      inserted.push_back(operation);
    if (previous)
      previous->notifyOperationInserted(operation, oldPoint);
  }

private:
  mlir::Block *componentBlock = nullptr;
  mlir::OpBuilder::Listener *previous = nullptr;
  llvm::SmallVectorImpl<mlir::Operation *> &inserted;
};

bool isStaticRankedTensor(mlir::Type type) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return tensor && tensor.hasStaticShape();
}

bool isPureStructuredOperation(mlir::Operation *operation) {
  return operation && mlir::isPure(operation);
}

bool isAccessPayload(mlir::linalg::LinalgOp operation) {
  if (!operation || !operation.hasPureTensorSemantics() ||
      operation->getNumResults() != 1 || operation->getNumRegions() != 1 ||
      !operation->getRegion(0).hasOneBlock() ||
      operation.getDpsInputOperands().size() != 1 ||
      operation.getDpsInitsMutable().size() != 1 ||
      !llvm::all_of(operation.getIteratorTypesArray(),
                    mlir::linalg::isParallelIterator))
    return false;
  mlir::Block &body = operation->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  return yield && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == body.getArgument(0) &&
         body.without_terminator().empty();
}

std::optional<AccessDescription>
getAccessDescription(mlir::Operation *operation) {
  if (!operation || operation->getNumResults() != 1 ||
      !isPureStructuredOperation(operation))
    return std::nullopt;
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(operation->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return std::nullopt;

  if (auto collapse =
          mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(operation)) {
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(collapse.getSrc().getType());
    if (!sourceType || !sourceType.hasStaticShape())
      return std::nullopt;
    IndexRelationResult relation = IndexRelation::staticReshape(
        resultType.getShape(), sourceType.getShape());
    if (!relation.isExact())
      return std::nullopt;
    return AccessDescription{collapse.getSrc(), std::move(*relation.get()),
                             std::nullopt, RelationMaterialization::Reshape};
  }
  if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation)) {
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(expand.getSrc().getType());
    if (!sourceType || !sourceType.hasStaticShape())
      return std::nullopt;
    IndexRelationResult relation = IndexRelation::staticReshape(
        resultType.getShape(), sourceType.getShape());
    if (!relation.isExact())
      return std::nullopt;
    return AccessDescription{expand.getSrc(), std::move(*relation.get()),
                             std::nullopt, RelationMaterialization::Reshape};
  }

  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!isAccessPayload(linalg))
    return std::nullopt;
  mlir::OpOperand *input = linalg.getDpsInputOperand(0);
  mlir::OpOperand *output = linalg.getDpsInitOperand(0);
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(input->get().getType());
  if (!sourceType || !sourceType.hasStaticShape())
    return std::nullopt;
  llvm::SmallVector<int64_t, 4> ranges = linalg.getStaticLoopRanges();
  if (llvm::is_contained(ranges, mlir::ShapedType::kDynamic))
    return std::nullopt;
  mlir::AffineMap sourceMap = linalg.getMatchingIndexingMap(input);
  mlir::AffineMap resultMap = linalg.getMatchingIndexingMap(output);
  IndexRelationResult relation = IndexRelation::fromCommonIterationDomain(
      resultMap, resultType.getShape(), sourceMap, sourceType.getShape(),
      ranges);
  if (!relation.isExact())
    return std::nullopt;

  std::optional<mlir::AffineMap> projected;
  mlir::AffineMap resultToIteration = mlir::inversePermutation(resultMap);
  if (resultToIteration) {
    mlir::AffineMap candidate = sourceMap.compose(resultToIteration);
    IndexRelationResult candidateRelation = IndexRelation::fromAffineMap(
        candidate, resultType.getShape(), sourceType.getShape());
    if (candidate.isProjectedPermutation(/*allowZeroInResults=*/true) &&
        candidateRelation.isExact() &&
        candidateRelation.get()->hasTotalBoundedAffineMapConstruction() &&
        candidateRelation.get()->isEquivalentTo(*relation.get()).isProvenTrue())
      projected = candidate;
  }
  if (!projected)
    return std::nullopt;
  return AccessDescription{input->get(), std::move(*relation.get()), projected,
                           RelationMaterialization::ProjectedMap};
}

std::optional<llvm::SmallVector<int64_t, 6>>
getStaticValues(mlir::ArrayRef<mlir::OpFoldResult> values) {
  llvm::SmallVector<int64_t, 6> result;
  result.reserve(values.size());
  for (mlir::OpFoldResult value : values) {
    std::optional<int64_t> constant = mlir::getConstantIntValue(value);
    if (!constant)
      return std::nullopt;
    result.push_back(*constant);
  }
  return result;
}

std::optional<ConcatDescription>
getConcatDescription(mlir::Operation *operation) {
  auto root = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(operation);
  if (!root)
    return std::nullopt;
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(root.getType());
  if (!resultType || !resultType.hasStaticShape() || resultType.getRank() == 0)
    return std::nullopt;

  struct Piece {
    mlir::Value source;
    llvm::SmallVector<int64_t, 6> offsets;
    llvm::SmallVector<int64_t, 6> sizes;
  };
  llvm::SmallVector<Piece, 4> reversePieces;
  mlir::tensor::InsertSliceOp current = root;
  mlir::Operation *expectedUser = nullptr;
  while (current) {
    if (expectedUser &&
        (!current.getResult().hasOneUse() ||
         *current.getResult().getUsers().begin() != expectedUser))
      return std::nullopt;
    if (current.getType() != resultType ||
        current.getDest().getType() != resultType)
      return std::nullopt;
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(current.getSource().getType());
    auto offsets = getStaticValues(current.getMixedOffsets());
    auto sizes = getStaticValues(current.getMixedSizes());
    auto strides = getStaticValues(current.getMixedStrides());
    if (!sourceType || !sourceType.hasStaticShape() ||
        sourceType.getRank() != resultType.getRank() ||
        sourceType.getElementType() != resultType.getElementType() ||
        !offsets || !sizes || !strides ||
        offsets->size() != static_cast<size_t>(resultType.getRank()) ||
        sizes->size() != offsets->size() ||
        strides->size() != offsets->size() ||
        !llvm::all_of(*strides, [](int64_t stride) { return stride == 1; }) ||
        !llvm::equal(*sizes, sourceType.getShape()))
      return std::nullopt;
    reversePieces.push_back(
        Piece{current.getSource(), std::move(*offsets), std::move(*sizes)});

    mlir::Value destination = current.getDest();
    if (auto previous =
            destination.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
      expectedUser = current.getOperation();
      current = previous;
      continue;
    }
    auto empty = destination.getDefiningOp<mlir::tensor::EmptyOp>();
    if (!empty || empty.getType() != resultType ||
        !empty.getResult().hasOneUse() ||
        *empty.getResult().getUsers().begin() != current.getOperation())
      return std::nullopt;
    break;
  }

  std::reverse(reversePieces.begin(), reversePieces.end());
  std::optional<int64_t> axis;
  for (const Piece &piece : reversePieces)
    for (int64_t dimension = 0; dimension < resultType.getRank(); ++dimension) {
      if (piece.offsets[dimension] == 0 &&
          piece.sizes[dimension] == resultType.getDimSize(dimension))
        continue;
      if (axis && *axis != dimension)
        return std::nullopt;
      axis = dimension;
    }
  if (!axis)
    axis = 0;

  ConcatDescription result{resultType, *axis, {}};
  int64_t nextOffset = 0;
  for (const Piece &piece : reversePieces) {
    for (int64_t dimension = 0; dimension < resultType.getRank(); ++dimension) {
      if (dimension == *axis) {
        if (piece.offsets[dimension] != nextOffset ||
            piece.sizes[dimension] <= 0 ||
            piece.sizes[dimension] >
                std::numeric_limits<int64_t>::max() - nextOffset)
          return std::nullopt;
        nextOffset += piece.sizes[dimension];
      } else if (piece.offsets[dimension] != 0 ||
                 piece.sizes[dimension] != resultType.getDimSize(dimension)) {
        return std::nullopt;
      }
    }
    result.inputs.push_back(piece.source);
  }
  if (nextOffset != resultType.getDimSize(*axis))
    return std::nullopt;
  return result;
}

EGraphNodeKind classifyLinalg(mlir::linalg::LinalgOp operation) {
  if (mlir::linalg::isaContractionOpInterface(operation))
    return EGraphNodeKind::Contraction;
  if (llvm::any_of(operation.getIteratorTypesArray(),
                   mlir::linalg::isReductionIterator))
    return EGraphNodeKind::Reduction;
  return EGraphNodeKind::Elementwise;
}

bool hasCanonicalContractionPayload(mlir::linalg::LinalgOp operation) {
  if (!operation || operation.getDpsInputOperands().size() != 2 ||
      operation.getDpsInitsMutable().size() != 1 ||
      operation->getNumRegions() != 1 || !operation->getRegion(0).hasOneBlock())
    return false;
  mlir::Block &body = operation->getRegion(0).front();
  if (body.getNumArguments() != 3 ||
      body.getArgument(0).getType() != body.getArgument(1).getType() ||
      body.getArgument(0).getType() != body.getArgument(2).getType())
    return false;
  auto payload = body.without_terminator();
  if (llvm::range_size(payload) != 2)
    return false;
  mlir::Operation &multiply = *payload.begin();
  mlir::Operation &add = *std::next(payload.begin());
  mlir::Value product;
  if (auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(multiply)) {
    auto sum = mlir::dyn_cast<mlir::arith::AddFOp>(add);
    if (!sum || mul.getFastmath() != mlir::arith::FastMathFlags::none ||
        sum.getFastmath() != mlir::arith::FastMathFlags::none ||
        mul.getLhs() != body.getArgument(0) ||
        mul.getRhs() != body.getArgument(1) ||
        sum.getLhs() != body.getArgument(2) || sum.getRhs() != mul.getResult())
      return false;
    product = sum.getResult();
  } else if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(multiply)) {
    auto sum = mlir::dyn_cast<mlir::arith::AddIOp>(add);
    if (!sum ||
        mul.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none ||
        sum.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none ||
        mul.getLhs() != body.getArgument(0) ||
        mul.getRhs() != body.getArgument(1) ||
        sum.getLhs() != body.getArgument(2) || sum.getRhs() != mul.getResult())
      return false;
    product = sum.getResult();
  } else {
    return false;
  }
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  return yield && yield.getNumOperands() == 1 && yield.getOperand(0) == product;
}

bool isSupportedCompute(mlir::Operation *operation) {
  if (!operation || mlir::isa<LinalgExtAttentionOp>(operation) ||
      operation->getNumResults() != 1 ||
      !isStaticRankedTensor(operation->getResult(0).getType()) ||
      !isPureStructuredOperation(operation))
    return false;
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (!linalg || !linalg.hasPureTensorSemantics() ||
      operation->getNumRegions() != 1 || !operation->getRegion(0).hasOneBlock())
    return false;
  if (mlir::linalg::isaContractionOpInterface(linalg) &&
      !hasCanonicalContractionPayload(linalg))
    return false;
  if (llvm::is_contained(linalg.getStaticLoopRanges(),
                         mlir::ShapedType::kDynamic))
    return false;
  return llvm::all_of(operation->getOperandTypes(), isStaticRankedTensor);
}

bool isSupportedProducer(mlir::Operation *operation) {
  return operation && operation->getNumResults() == 1 &&
         isStaticRankedTensor(operation->getResult(0).getType()) &&
         (getAccessDescription(operation) || getConcatDescription(operation) ||
          isSupportedCompute(operation));
}

void eraseDeadProducerClosure(mlir::RewriterBase &rewriter,
                              llvm::ArrayRef<mlir::Operation *> seeds) {
  llvm::SmallVector<mlir::Operation *, 16> worklist(seeds.begin(), seeds.end());
  while (!worklist.empty()) {
    mlir::Operation *operation = worklist.pop_back_val();
    bool unusedTensorEmpty =
        mlir::isa<mlir::tensor::EmptyOp>(operation) && operation->use_empty();
    if (!unusedTensorEmpty && !mlir::isOpTriviallyDead(operation))
      continue;
    llvm::SmallVector<mlir::Operation *, 4> producers;
    for (mlir::Value operand : operation->getOperands())
      if (mlir::Operation *producer = operand.getDefiningOp())
        producers.push_back(producer);
    rewriter.eraseOp(operation);
    llvm::append_range(worklist, producers);
  }
}

enum class MatmulVariant {
  None,
  Matmul,
  MatmulTransposeA,
  MatmulTransposeB,
  BatchMatmul,
  BatchMatmulTransposeA,
  BatchMatmulTransposeB,
};

MatmulVariant
classifyMatmulMaps(llvm::ArrayRef<mlir::AffineMap> maps,
                   llvm::ArrayRef<mlir::utils::IteratorType> iters,
                   int64_t resultRank) {
  if (maps.size() != 3)
    return MatmulVariant::None;
  mlir::MLIRContext *context = maps.front().getContext();
  if (resultRank == 2 && iters == llvm::ArrayRef<mlir::utils::IteratorType>{
                                      mlir::utils::IteratorType::parallel,
                                      mlir::utils::IteratorType::parallel,
                                      mlir::utils::IteratorType::reduction}) {
    auto d0 = mlir::getAffineDimExpr(0, context);
    auto d1 = mlir::getAffineDimExpr(1, context);
    auto d2 = mlir::getAffineDimExpr(2, context);
    auto map = [&](mlir::AffineExpr a, mlir::AffineExpr b) {
      return mlir::AffineMap::get(3, 0, {a, b}, context);
    };
    if (maps[2] != map(d0, d1))
      return MatmulVariant::None;
    bool lhsT = maps[0] == map(d2, d0);
    bool lhsN = maps[0] == map(d0, d2);
    bool rhsT = maps[1] == map(d1, d2);
    bool rhsN = maps[1] == map(d2, d1);
    if (lhsN && rhsN)
      return MatmulVariant::Matmul;
    if (lhsT && rhsN)
      return MatmulVariant::MatmulTransposeA;
    if (lhsN && rhsT)
      return MatmulVariant::MatmulTransposeB;
    return MatmulVariant::None;
  }
  if (resultRank == 3 && iters == llvm::ArrayRef<mlir::utils::IteratorType>{
                                      mlir::utils::IteratorType::parallel,
                                      mlir::utils::IteratorType::parallel,
                                      mlir::utils::IteratorType::parallel,
                                      mlir::utils::IteratorType::reduction}) {
    auto d0 = mlir::getAffineDimExpr(0, context);
    auto d1 = mlir::getAffineDimExpr(1, context);
    auto d2 = mlir::getAffineDimExpr(2, context);
    auto d3 = mlir::getAffineDimExpr(3, context);
    auto map = [&](std::initializer_list<mlir::AffineExpr> exprs) {
      return mlir::AffineMap::get(4, 0, exprs, context);
    };
    if (maps[2] != map({d0, d1, d2}))
      return MatmulVariant::None;
    bool lhsT = maps[0] == map({d0, d3, d1});
    bool lhsN = maps[0] == map({d0, d1, d3});
    bool rhsT = maps[1] == map({d0, d2, d3});
    bool rhsN = maps[1] == map({d0, d3, d2});
    if (lhsN && rhsN)
      return MatmulVariant::BatchMatmul;
    if (lhsT && rhsN)
      return MatmulVariant::BatchMatmulTransposeA;
    if (lhsN && rhsT)
      return MatmulVariant::BatchMatmulTransposeB;
  }
  return MatmulVariant::None;
}

mlir::Operation *createMatmulVariant(mlir::RewriterBase &rewriter,
                                     mlir::Location location,
                                     MatmulVariant variant,
                                     mlir::ValueRange inputs,
                                     mlir::ValueRange outputs) {
  if (inputs.size() != 2 || outputs.size() != 1)
    return nullptr;
  switch (variant) {
  case MatmulVariant::Matmul:
    return rewriter.create<mlir::linalg::MatmulOp>(location, inputs, outputs)
        .getOperation();
  case MatmulVariant::MatmulTransposeA:
    return rewriter
        .create<mlir::linalg::MatmulTransposeAOp>(location, inputs, outputs)
        .getOperation();
  case MatmulVariant::MatmulTransposeB:
    return rewriter
        .create<mlir::linalg::MatmulTransposeBOp>(location, inputs, outputs)
        .getOperation();
  case MatmulVariant::BatchMatmul:
    return rewriter
        .create<mlir::linalg::BatchMatmulOp>(location, inputs, outputs)
        .getOperation();
  case MatmulVariant::BatchMatmulTransposeA:
    return rewriter
        .create<mlir::linalg::BatchMatmulTransposeAOp>(location, inputs,
                                                       outputs)
        .getOperation();
  case MatmulVariant::BatchMatmulTransposeB:
    return rewriter
        .create<mlir::linalg::BatchMatmulTransposeBOp>(location, inputs,
                                                       outputs)
        .getOperation();
  case MatmulVariant::None:
    return nullptr;
  }
  return nullptr;
}

struct MultiUseConsumerUpdate {
  mlir::linalg::LinalgOp operation;
  llvm::SmallVector<mlir::AffineMap, 4> maps;
  llvm::SmallVector<unsigned, 2> operands;
  MatmulVariant contractionVariant = MatmulVariant::None;
};

mlir::LogicalResult tryPropagateMultiUseProjectedAccess(
    mlir::Operation *operation, mlir::IRRewriter &rewriter,
    StructuredGraphNormalizationStatistics &statistics, bool &changed) {
  changed = false;
  std::optional<AccessDescription> access = getAccessDescription(operation);
  if (!access || !access->projectedMap || operation->getNumResults() != 1 ||
      llvm::range_size(operation->getResult(0).getUses()) < 2)
    return mlir::success();
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(access->source.getType());
  if (!sourceType || !sourceType.hasStaticShape())
    return mlir::success();

  llvm::SmallVector<MultiUseConsumerUpdate, 4> updates;
  llvm::DenseMap<mlir::Operation *, unsigned> updateIndices;
  for (mlir::OpOperand &use : operation->getResult(0).getUses()) {
    auto consumer = mlir::dyn_cast<mlir::linalg::LinalgOp>(use.getOwner());
    if (!consumer || !isSupportedCompute(consumer) ||
        use.getOperandNumber() >= consumer.getNumDpsInputs())
      return mlir::success();
    EGraphNodeKind kind = classifyLinalg(consumer);
    if (kind != EGraphNodeKind::Contraction &&
        !mlir::isa<mlir::linalg::GenericOp>(consumer))
      return mlir::success();
    unsigned updateIndex = 0;
    auto found = updateIndices.find(consumer.getOperation());
    if (found == updateIndices.end()) {
      updateIndex = updates.size();
      updateIndices.try_emplace(consumer.getOperation(), updateIndex);
      updates.push_back(MultiUseConsumerUpdate{
          consumer, consumer.getIndexingMapsArray(), {}, MatmulVariant::None});
    } else {
      updateIndex = found->second;
    }
    MultiUseConsumerUpdate &update = updates[updateIndex];
    unsigned operandNumber = use.getOperandNumber();
    mlir::AffineMap newMap =
        access->projectedMap->compose(update.maps[operandNumber]);
    llvm::SmallVector<int64_t, 4> loopShape = consumer.getStaticLoopRanges();
    IndexRelationResult relation =
        IndexRelation::fromAffineMap(newMap, loopShape, sourceType.getShape());
    if (!relation.isExact() ||
        !relation.get()->hasTotalBoundedAffineMapConstruction())
      return mlir::success();
    update.maps[operandNumber] = newMap;
    update.operands.push_back(operandNumber);
  }

  for (MultiUseConsumerUpdate &update : updates) {
    if (!mlir::inversePermutation(mlir::concatAffineMaps(update.maps)))
      return mlir::success();
    if (classifyLinalg(update.operation) != EGraphNodeKind::Contraction)
      continue;
    if (!hasCanonicalContractionPayload(update.operation))
      return mlir::success();
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        update.operation->getResult(0).getType());
    if (!resultType)
      return mlir::success();
    update.contractionVariant = classifyMatmulMaps(
        update.maps, update.operation.getIteratorTypesArray(),
        resultType.getRank());
    if (update.contractionVariant == MatmulVariant::None)
      return mlir::success();
  }

  llvm::SmallVector<mlir::Operation *, 4> oldProducers;
  for (mlir::Value operand : operation->getOperands())
    if (mlir::Operation *producer = operand.getDefiningOp())
      oldProducers.push_back(producer);

  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>, 2>
      contractionReplacements;
  for (MultiUseConsumerUpdate &update : updates) {
    if (update.contractionVariant == MatmulVariant::None)
      continue;
    llvm::SmallVector<mlir::Value, 4> operands(
        update.operation->getOperands().begin(),
        update.operation->getOperands().end());
    for (unsigned operandNumber : update.operands)
      operands[operandNumber] = access->source;
    rewriter.setInsertionPoint(update.operation);
    mlir::Operation *replacement = createMatmulVariant(
        rewriter, update.operation.getLoc(), update.contractionVariant,
        llvm::ArrayRef(operands).take_front(update.operation.getNumDpsInputs()),
        llvm::ArrayRef(operands).drop_front(
            update.operation.getNumDpsInputs()));
    if (!replacement) {
      for (auto &inserted : llvm::reverse(contractionReplacements))
        rewriter.eraseOp(inserted.second);
      return mlir::failure();
    }
    llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
    for (mlir::NamedAttribute attribute :
         update.operation->getDiscardableAttrs())
      if (attribute.getName() !=
          mlir::linalg::LinalgDialect::kMemoizedIndexingMapsAttrName)
        attributes.push_back(attribute);
    replacement->setDiscardableAttrs(attributes);
    contractionReplacements.emplace_back(update.operation, replacement);
  }

  for (MultiUseConsumerUpdate &update : updates) {
    if (update.contractionVariant != MatmulVariant::None)
      continue;
    auto generic = mlir::cast<mlir::linalg::GenericOp>(update.operation);
    rewriter.modifyOpInPlace(generic, [&] {
      for (unsigned operandNumber : update.operands)
        generic->setOperand(operandNumber, access->source);
      generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(update.maps));
      generic->removeDiscardableAttr(
          mlir::linalg::LinalgDialect::kMemoizedIndexingMapsAttrName);
    });
  }
  for (auto [oldOperation, replacement] : contractionReplacements)
    rewriter.replaceOp(oldOperation, replacement->getResults());

  rewriter.eraseOp(operation);
  eraseDeadProducerClosure(rewriter, oldProducers);
  ++statistics.accessTransformsRemoved;
  ++statistics.multiUseAccessPropagations;
  changed = true;
  return mlir::success();
}

mlir::LogicalResult propagateMultiUseProjectedAccesses(
    mlir::func::FuncOp function, mlir::IRRewriter &rewriter,
    StructuredGraphNormalizationStatistics &statistics) {
  while (true) {
    llvm::SmallVector<mlir::Operation *, 8> candidates;
    function.walk([&](mlir::Operation *operation) {
      if (operation->getNumResults() == 1 &&
          llvm::range_size(operation->getResult(0).getUses()) >= 2 &&
          getAccessDescription(operation))
        candidates.push_back(operation);
    });
    bool changed = false;
    for (mlir::Operation *candidate : candidates) {
      if (mlir::failed(tryPropagateMultiUseProjectedAccess(
              candidate, rewriter, statistics, changed)))
        return mlir::failure();
      if (changed)
        break;
    }
    if (!changed)
      return mlir::success();
  }
}

class ComponentExpression {
public:
  ComponentExpression(mlir::Value root,
                      const StructuredGraphNormalizationOptions &options)
      : root(root), options(options),
        relations(types, root.getContext(), options.maximumRelationQueries) {
    mlir::Operation *parent = root.getParentBlock()->getParentOp();
    uint32_t order = 1;
    parent->walk([&](mlir::Operation *operation) {
      sourceOrder.try_emplace(operation, order++);
    });
  }

  mlir::FailureOr<StructuredGraphNormalizationOutcome>
  apply(mlir::RewriterBase &rewriter,
        StructuredGraphNormalizationStatistics &statistics) {
    uint32_t rootNode = import(root, /*allowProducer=*/true);
    if (relations.hasWorkLimitReached()) {
      ++statistics.components;
      ++statistics.budgetExhaustedComponents;
      return StructuredGraphNormalizationOutcome::BudgetExhausted;
    }
    if (nodes.size() == 1 || !hasPotentialRewrite)
      return StructuredGraphNormalizationOutcome::Unchanged;
    ++statistics.components;
    EGraphOutcome result = structured_graph_normalization::runEGraph(
        nodes, rootNode,
        EGraphWorkBudget{options.maximumENodes, options.maximumMatches,
                         options.maximumIterations},
        getRelationService());
    addStatistics(statistics, result.statistics);
    if (result.kind == EGraphOutcomeKind::BudgetExhausted) {
      ++statistics.budgetExhaustedComponents;
      return StructuredGraphNormalizationOutcome::BudgetExhausted;
    }
    if (result.kind == EGraphOutcomeKind::Unchanged) {
      ++statistics.unchangedComponents;
      return StructuredGraphNormalizationOutcome::Unchanged;
    }
    if (result.kind != EGraphOutcomeKind::Changed)
      return mlir::failure();
    if (mlir::failed(validateExtraction(result.expression, result.rootNode)))
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *, 16> insertedOperations;
    mlir::FailureOr<mlir::Value> replacement = mlir::failure();
    {
      mlir::OpBuilder::Listener *previousListener = rewriter.getListener();
      InsertedOperationRecorder recorder(root.getParentBlock(),
                                         previousListener, insertedOperations);
      rewriter.setListener(&recorder);
      auto restoreListener = llvm::make_scope_exit(
          [&] { rewriter.setListener(previousListener); });
      replacement = materialize(rewriter, result.expression, result.rootNode);
    }
    if (mlir::failed(replacement) || replacement->getType() != root.getType() ||
        *replacement == root) {
      for (mlir::Operation *operation : llvm::reverse(insertedOperations))
        rewriter.eraseOp(operation);
      return mlir::failure();
    }
    mlir::Operation *oldRoot = root.getDefiningOp();
    llvm::SmallVector<mlir::Operation *, 4> oldProducers;
    for (mlir::Value operand : oldRoot->getOperands())
      if (mlir::Operation *producer = operand.getDefiningOp())
        oldProducers.push_back(producer);
    rewriter.replaceOp(oldRoot, *replacement);
    eraseDeadProducerClosure(rewriter, oldProducers);
    if (result.statistics.inputAccessOccurrences >=
        result.statistics.outputAccessOccurrences)
      statistics.accessTransformsRemoved +=
          result.statistics.inputAccessOccurrences -
          result.statistics.outputAccessOccurrences;
    if (result.statistics.inputConcatOccurrences >=
        result.statistics.outputConcatOccurrences)
      statistics.concatTransformsRemoved +=
          result.statistics.inputConcatOccurrences -
          result.statistics.outputConcatOccurrences;
    ++statistics.changedComponents;
    unsigned appliedRuleKinds = 0;
    appliedRuleKinds += result.statistics.identityApplications != 0;
    appliedRuleKinds += result.statistics.compositionApplications != 0;
    appliedRuleKinds += result.statistics.computeAbsorptionApplications != 0;
    appliedRuleKinds += result.statistics.concatApplications != 0;
    appliedRuleKinds += result.statistics.resultReindexApplications != 0;
    if (appliedRuleKinds >= 2)
      ++statistics.multiRuleChangedComponents;
    return StructuredGraphNormalizationOutcome::Changed;
  }

private:
  uint32_t addInput(mlir::Value value) {
    inputRecipes.push_back(value);
    EGraphNode node;
    node.kind = EGraphNodeKind::Input;
    node.semanticId = inputRecipes.size();
    node.typeId = types.intern(value.getType());
    node.sourceOrder = value.getDefiningOp()
                           ? sourceOrder.lookup(value.getDefiningOp())
                           : inputRecipes.size();
    nodes.push_back(std::move(node));
    return nodes.size() - 1;
  }

  uint32_t import(mlir::Value value, bool allowProducer) {
    auto found = imported.find(value);
    if (found != imported.end())
      return found->second;
    mlir::Operation *operation = value.getDefiningOp();
    if (!allowProducer || !operation || !isSupportedProducer(operation) ||
        (value != root && !value.hasOneUse())) {
      uint32_t node = addInput(value);
      imported.try_emplace(value, node);
      return node;
    }

    if (std::optional<ConcatDescription> concat =
            getConcatDescription(operation)) {
      EGraphNode node;
      node.kind = EGraphNodeKind::Concat;
      node.typeId = types.intern(concat->resultType);
      node.axis = concat->axis;
      node.sourceOrder = sourceOrder.lookup(operation);
      for (mlir::Value input : concat->inputs)
        node.children.push_back(import(input, /*allowProducer=*/true));
      uint32_t index = nodes.size();
      nodes.push_back(std::move(node));
      imported.try_emplace(value, index);
      hasPotentialRewrite = true;
      return index;
    }

    if (std::optional<AccessDescription> access =
            getAccessDescription(operation)) {
      uint32_t child = import(access->source, /*allowProducer=*/true);
      uint32_t destinationTypeId = types.intern(value.getType());
      uint32_t sourceTypeId = types.intern(access->source.getType());
      std::optional<uint32_t> relation = relations.intern(
          std::move(access->relation), destinationTypeId, sourceTypeId,
          access->projectedMap, access->materialization, /*total=*/true);
      if (!relation) {
        uint32_t node = addInput(value);
        imported.try_emplace(value, node);
        return node;
      }
      EGraphNode node;
      node.kind = EGraphNodeKind::Access;
      node.typeId = destinationTypeId;
      node.children.push_back(child);
      node.relations.push_back(*relation);
      node.sourceOrder = sourceOrder.lookup(operation);
      uint32_t index = nodes.size();
      nodes.push_back(std::move(node));
      imported.try_emplace(value, index);
      hasPotentialRewrite = true;
      return index;
    }

    auto linalg = mlir::cast<mlir::linalg::LinalgOp>(operation);
    EGraphNode node;
    node.kind = classifyLinalg(linalg);
    node.typeId = types.intern(value.getType());
    node.dataInputCount = linalg.getNumDpsInputs();
    node.sourceOrder = sourceOrder.lookup(operation);
    for (mlir::Value operand : operation->getOperands())
      node.children.push_back(import(operand, /*allowProducer=*/true));

    llvm::SmallVector<int64_t, 4> loopShape = linalg.getStaticLoopRanges();
    mlir::Type iterationType = mlir::RankedTensorType::get(
        loopShape, mlir::IntegerType::get(root.getContext(), 1));
    uint32_t iterationTypeId = types.intern(iterationType);
    llvm::SmallVector<mlir::AffineMap, 4> maps = linalg.getIndexingMapsArray();
    for (auto [operand, map] : llvm::zip(operation->getOperands(), maps)) {
      auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
      IndexRelationResult relation =
          IndexRelation::fromAffineMap(map, loopShape, operandType.getShape());
      if (!relation.isExact()) {
        uint32_t input = addInput(value);
        imported.try_emplace(value, input);
        return input;
      }
      std::optional<uint32_t> relationId = relations.intern(
          std::move(*relation.get()), iterationTypeId,
          types.intern(operandType), map, RelationMaterialization::ProjectedMap,
          /*total=*/true);
      if (!relationId) {
        uint32_t input = addInput(value);
        imported.try_emplace(value, input);
        return input;
      }
      node.relations.push_back(*relationId);
    }

    mlir::Block &body = operation->getRegion(0).front();
    bool initIsRead = false;
    for (mlir::BlockArgument argument :
         body.getArguments().drop_front(linalg.getNumDpsInputs()))
      initIsRead |= !argument.use_empty();
    computeRecipes.push_back(
        ComputeRecipe{operation, node.kind,
                      mlir::cast<mlir::RankedTensorType>(value.getType()),
                      node.typeId, iterationTypeId, node.dataInputCount, maps,
                      linalg.getIteratorTypesArray(), initIsRead,
                      node.kind != EGraphNodeKind::Contraction ||
                          hasCanonicalContractionPayload(linalg)});
    node.semanticId = computeRecipes.size();
    uint32_t index = nodes.size();
    nodes.push_back(std::move(node));
    imported.try_emplace(value, index);
    return index;
  }

  const ComputeRecipe *getRecipe(uint32_t id) const {
    return id == 0 || id > computeRecipes.size() ? nullptr
                                                 : &computeRecipes[id - 1];
  }

  bool getMaps(llvm::ArrayRef<uint32_t> relationIds,
               llvm::SmallVectorImpl<mlir::AffineMap> &maps) const {
    maps.clear();
    maps.reserve(relationIds.size());
    for (uint32_t relationId : relationIds) {
      const RelationRecord *relation = relations.lookup(relationId);
      if (!relation || !relation->projectedMap)
        return false;
      maps.push_back(*relation->projectedMap);
    }
    return true;
  }

  bool validateCompute(uint32_t semanticId, uint32_t kind,
                       uint32_t resultTypeId,
                       llvm::ArrayRef<uint32_t> relationIds,
                       llvm::ArrayRef<uint32_t> operandTypeIds,
                       uint32_t dataInputCount) const {
    const ComputeRecipe *recipe = getRecipe(semanticId);
    auto resultType = types.lookupRankedTensor(resultTypeId);
    if (!recipe || kind != static_cast<uint32_t>(recipe->kind) || !resultType ||
        relationIds.size() != operandTypeIds.size() ||
        relationIds.size() != recipe->operation->getNumOperands() ||
        dataInputCount != recipe->dataInputCount)
      return false;

    uint32_t iterationTypeId = 0;
    llvm::SmallVector<mlir::AffineMap, 4> maps;
    for (auto [index, relationId] : llvm::enumerate(relationIds)) {
      const RelationRecord *relation = relations.lookup(relationId);
      if (!relation || !relation->projectedMap)
        return false;
      if (iterationTypeId == 0)
        iterationTypeId = relation->destinationTypeId;
      if (relation->destinationTypeId != iterationTypeId)
        return false;
      bool sourceMatchesOperand =
          relation->sourceTypeId == operandTypeIds[index];
      bool ignoredReindexedInit = index >= dataInputCount &&
                                  !recipe->initIsRead &&
                                  relation->sourceTypeId == resultTypeId;
      if (!sourceMatchesOperand && !ignoredReindexedInit)
        return false;
      maps.push_back(*relation->projectedMap);
    }
    auto iterationType = types.lookupRankedTensor(iterationTypeId);
    if (!iterationType ||
        iterationType.getRank() !=
            static_cast<int64_t>(recipe->iteratorTypes.size()) ||
        maps.empty() || maps.back().getNumResults() != resultType.getRank())
      return false;
    for (mlir::AffineMap map : maps)
      if (map.getNumDims() != recipe->iteratorTypes.size() ||
          !map.isProjectedPermutation(/*allowZeroInResults=*/true))
        return false;
    // Generic Linalg verification requires the concatenated operand maps to
    // determine every loop bound.  In particular, absorbing a broadcast into
    // a reduction can otherwise erase the only map that names a reduction
    // dimension even though the scalar payload is still meaningful.
    if (!mlir::inversePermutation(mlir::concatAffineMaps(maps)))
      return false;

    bool mapsUnchanged = resultTypeId == recipe->resultTypeId &&
                         llvm::equal(maps, recipe->originalMaps);
    if (mapsUnchanged)
      return true;
    if (recipe->kind == EGraphNodeKind::Contraction) {
      if (!recipe->hasCanonicalContractionPayload)
        return false;
      MatmulVariant variant =
          classifyMatmulMaps(maps, recipe->iteratorTypes, resultType.getRank());
      return variant != MatmulVariant::None;
    }
    return mlir::isa<mlir::linalg::GenericOp>(recipe->operation) ||
           recipe->kind != EGraphNodeKind::Contraction;
  }

  bool validateConcat(uint32_t resultTypeId, int32_t axis,
                      llvm::ArrayRef<uint32_t> inputTypeIds) const {
    auto resultType = types.lookupRankedTensor(resultTypeId);
    if (!resultType || !resultType.hasStaticShape() || axis < 0 ||
        axis >= resultType.getRank() || inputTypeIds.empty())
      return false;
    int64_t extent = 0;
    for (uint32_t inputTypeId : inputTypeIds) {
      auto inputType = types.lookupRankedTensor(inputTypeId);
      if (!inputType || !inputType.hasStaticShape() ||
          inputType.getRank() != resultType.getRank() ||
          inputType.getElementType() != resultType.getElementType() ||
          inputType.getEncoding() != resultType.getEncoding())
        return false;
      for (int64_t dimension = 0; dimension < resultType.getRank(); ++dimension)
        if (dimension != axis &&
            inputType.getDimSize(dimension) != resultType.getDimSize(dimension))
          return false;
      if (inputType.getDimSize(axis) <= 0 ||
          llvm::AddOverflow(extent, inputType.getDimSize(axis), extent))
        return false;
    }
    return extent == resultType.getDimSize(axis);
  }

  uint32_t factorConcat(uint32_t resultTypeId, int32_t destinationAxis,
                        llvm::ArrayRef<uint32_t> inputRelationIds,
                        llvm::ArrayRef<uint32_t> sourceTypeIds,
                        uint32_t *resultRelationId,
                        uint32_t *sourceConcatTypeId, int32_t *sourceAxis) {
    if (inputRelationIds.size() < 2 ||
        inputRelationIds.size() != sourceTypeIds.size() || !resultRelationId ||
        !sourceConcatTypeId || !sourceAxis)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    auto resultType = types.lookupRankedTensor(resultTypeId);
    const RelationRecord *first = relations.lookup(inputRelationIds.front());
    if (!resultType || !first || !first->projectedMap || destinationAxis < 0 ||
        destinationAxis >= resultType.getRank())
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    mlir::AffineMap map = *first->projectedMap;
    std::optional<unsigned> mappedAxis;
    for (auto [sourceDimension, expression] :
         llvm::enumerate(map.getResults())) {
      auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!dim || dim.getPosition() != static_cast<unsigned>(destinationAxis))
        continue;
      if (mappedAxis)
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      mappedAxis = sourceDimension;
    }
    if (!mappedAxis)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    auto firstSource = types.lookupRankedTensor(sourceTypeIds.front());
    if (!firstSource || !firstSource.hasStaticShape())
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    llvm::SmallVector<int64_t, 6> combinedShape(firstSource.getShape());
    combinedShape[*mappedAxis] = 0;
    for (auto [relationId, sourceTypeId] :
         llvm::zip_equal(inputRelationIds, sourceTypeIds)) {
      const RelationRecord *relation = relations.lookup(relationId);
      auto sourceType = types.lookupRankedTensor(sourceTypeId);
      auto destinationType =
          relation ? types.lookupRankedTensor(relation->destinationTypeId)
                   : mlir::RankedTensorType{};
      if (!relation || !relation->projectedMap ||
          *relation->projectedMap != map ||
          relation->sourceTypeId != sourceTypeId || !sourceType ||
          !destinationType || sourceType.getRank() != firstSource.getRank() ||
          sourceType.getElementType() != firstSource.getElementType() ||
          sourceType.getEncoding() != firstSource.getEncoding() ||
          sourceType.getDimSize(*mappedAxis) !=
              destinationType.getDimSize(destinationAxis))
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      for (int64_t dimension = 0; dimension < sourceType.getRank(); ++dimension)
        if (dimension != static_cast<int64_t>(*mappedAxis) &&
            sourceType.getDimSize(dimension) !=
                firstSource.getDimSize(dimension))
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      if (llvm::AddOverflow(combinedShape[*mappedAxis],
                            sourceType.getDimSize(*mappedAxis),
                            combinedShape[*mappedAxis]))
        return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
    }
    auto combinedType = mlir::RankedTensorType::get(
        combinedShape, firstSource.getElementType(), firstSource.getEncoding());
    uint32_t combinedTypeId = types.intern(combinedType);
    IndexRelationResult combined = IndexRelation::fromAffineMap(
        map, resultType.getShape(), combinedType.getShape());
    if (!combined.isExact())
      return combined.status == IndexRelationStatus::ResourceExhausted
                 ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                 : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    std::optional<uint32_t> relationId = relations.intern(
        std::move(*combined.get()), resultTypeId, combinedTypeId, map,
        RelationMaterialization::ProjectedMap, /*total=*/true);
    if (!relationId)
      return relations.hasWorkLimitReached()
                 ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                 : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    *resultRelationId = *relationId;
    *sourceConcatTypeId = combinedTypeId;
    *sourceAxis = *mappedAxis;
    return WAFER_EGRAPH_CALLBACK_EXACT;
  }

  uint32_t reindexCompute(uint32_t semanticId, uint32_t kind,
                          uint32_t sourceResultTypeId,
                          uint32_t destinationResultTypeId,
                          uint32_t resultRelationId,
                          llvm::ArrayRef<uint32_t> inputRelationIds,
                          uint32_t dataInputCount,
                          llvm::MutableArrayRef<uint32_t> outputRelationIds,
                          uint32_t *reindexReadInit) {
    const ComputeRecipe *recipe = getRecipe(semanticId);
    const RelationRecord *resultRelation = relations.lookup(resultRelationId);
    auto sourceResultType = types.lookupRankedTensor(sourceResultTypeId);
    auto destinationResultType =
        types.lookupRankedTensor(destinationResultTypeId);
    if (!recipe || kind != static_cast<uint32_t>(recipe->kind) ||
        !resultRelation || !resultRelation->bijective || !reindexReadInit ||
        !resultRelation->projectedMap || !sourceResultType ||
        !destinationResultType ||
        resultRelation->sourceTypeId != sourceResultTypeId ||
        resultRelation->destinationTypeId != destinationResultTypeId ||
        sourceResultType.getRank() != destinationResultType.getRank() ||
        inputRelationIds.size() != outputRelationIds.size() ||
        inputRelationIds.size() != recipe->originalMaps.size() ||
        dataInputCount != recipe->dataInputCount)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    llvm::SmallVector<mlir::AffineMap, 4> maps;
    if (!getMaps(inputRelationIds, maps))
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    mlir::AffineMap resultMap = *resultRelation->projectedMap;
    mlir::AffineMap sourceToDestination = mlir::inversePermutation(resultMap);
    if (!resultMap.isPermutation() || !sourceToDestination)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    auto iterationType = types.lookupRankedTensor(recipe->iterationTypeId);
    if (!iterationType)
      return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;

    // Keep the iteration domain and every data-input map unchanged.  Only
    // redirect each DPS init/result map from the old result coordinates to
    // the bijectively reindexed result coordinates.  The identity iteration
    // bijection preserves reduction axes, their order, and multiplicity.
    for (unsigned index = 0; index < dataInputCount; ++index)
      outputRelationIds[index] = inputRelationIds[index];
    for (unsigned index = dataInputCount; index < maps.size(); ++index) {
      mlir::AffineMap oldMap = maps[index];
      if (!oldMap.isProjectedPermutation() ||
          oldMap.getNumResults() != sourceResultType.getRank())
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      for (mlir::AffineExpr expression : oldMap.getResults()) {
        auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!dimension ||
            dimension.getPosition() >= recipe->iteratorTypes.size() ||
            recipe->iteratorTypes[dimension.getPosition()] !=
                mlir::utils::IteratorType::parallel)
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      }
      mlir::AffineMap newMap = sourceToDestination.compose(oldMap);
      IndexRelationResult relation = IndexRelation::fromAffineMap(
          newMap, iterationType.getShape(), destinationResultType.getShape());
      if (!relation.isExact())
        return relation.status == IndexRelationStatus::ResourceExhausted
                   ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                   : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      std::optional<uint32_t> relationId = relations.intern(
          std::move(*relation.get()), recipe->iterationTypeId,
          destinationResultTypeId, newMap,
          RelationMaterialization::ProjectedMap, /*total=*/true);
      if (!relationId)
        return relations.hasWorkLimitReached()
                   ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                   : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      outputRelationIds[index] = *relationId;
    }
    *reindexReadInit = recipe->initIsRead ? 1 : 0;
    return WAFER_EGRAPH_CALLBACK_EXACT;
  }

  uint32_t reparameterizeElementwise(
      uint32_t semanticId, uint32_t sourceResultTypeId,
      uint32_t destinationResultTypeId, uint32_t resultRelationId,
      llvm::ArrayRef<uint32_t> inputRelationIds, uint32_t dataInputCount,
      llvm::MutableArrayRef<uint32_t> outputComputeRelationIds,
      llvm::MutableArrayRef<uint32_t> outputOperandAccessRelationIds,
      llvm::MutableArrayRef<uint32_t> outputOperandAccessTypeIds) {
    const ComputeRecipe *recipe = getRecipe(semanticId);
    const RelationRecord *resultRelation = relations.lookup(resultRelationId);
    auto sourceResultType = types.lookupRankedTensor(sourceResultTypeId);
    auto destinationResultType =
        types.lookupRankedTensor(destinationResultTypeId);
    auto linalg =
        recipe ? mlir::dyn_cast<mlir::linalg::LinalgOp>(recipe->operation)
               : mlir::linalg::LinalgOp{};
    if (!recipe || recipe->kind != EGraphNodeKind::Elementwise || !linalg ||
        linalg.hasIndexSemantics() || !resultRelation ||
        !resultRelation->bijective || !resultRelation->projectedMap ||
        resultRelation->sourceTypeId != sourceResultTypeId ||
        resultRelation->destinationTypeId != destinationResultTypeId ||
        !sourceResultType || !destinationResultType ||
        sourceResultType.getRank() != destinationResultType.getRank() ||
        inputRelationIds.size() != recipe->originalMaps.size() ||
        inputRelationIds.size() != outputComputeRelationIds.size() ||
        inputRelationIds.size() != outputOperandAccessRelationIds.size() ||
        inputRelationIds.size() != outputOperandAccessTypeIds.size() ||
        dataInputCount != recipe->dataInputCount ||
        !llvm::all_of(recipe->iteratorTypes, mlir::linalg::isParallelIterator))
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    llvm::SmallVector<mlir::AffineMap, 4> maps;
    if (!getMaps(inputRelationIds, maps) || maps.empty())
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    mlir::AffineMap oldOutputMap = maps.back();
    mlir::AffineMap resultMap = *resultRelation->projectedMap;
    mlir::AffineMap outputToIteration = mlir::inversePermutation(oldOutputMap);
    if (!oldOutputMap.isPermutation() || !resultMap.isPermutation() ||
        !outputToIteration ||
        oldOutputMap.getNumResults() != sourceResultType.getRank() ||
        destinationResultType.getRank() !=
            static_cast<int64_t>(recipe->iteratorTypes.size()))
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    mlir::AffineMap newToOldIteration = outputToIteration.compose(resultMap);
    mlir::AffineMap identity = mlir::AffineMap::getMultiDimIdentityMap(
        destinationResultType.getRank(), root.getContext());
    auto iterationType = mlir::RankedTensorType::get(
        destinationResultType.getShape(),
        mlir::IntegerType::get(root.getContext(), 1));
    uint32_t iterationTypeId = types.intern(iterationType);
    auto internProjected =
        [&](mlir::AffineMap map, uint32_t destinationTypeId,
            uint32_t sourceTypeId, llvm::ArrayRef<int64_t> destinationShape,
            llvm::ArrayRef<int64_t> sourceShape) -> std::optional<uint32_t> {
      IndexRelationResult relation =
          IndexRelation::fromAffineMap(map, destinationShape, sourceShape);
      if (!relation.isExact() ||
          !relation.get()->hasTotalBoundedAffineMapConstruction())
        return std::nullopt;
      return relations.intern(std::move(*relation.get()), destinationTypeId,
                              sourceTypeId, map,
                              RelationMaterialization::ProjectedMap,
                              /*total=*/true);
    };

    for (unsigned index = 0; index < dataInputCount; ++index) {
      const RelationRecord *oldRelation =
          relations.lookup(inputRelationIds[index]);
      auto childType = oldRelation
                           ? types.lookupRankedTensor(oldRelation->sourceTypeId)
                           : mlir::RankedTensorType{};
      if (!oldRelation || !childType || childType.getEncoding())
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      auto accessType = mlir::RankedTensorType::get(
          destinationResultType.getShape(), childType.getElementType());
      uint32_t accessTypeId = types.intern(accessType);
      mlir::AffineMap accessMap = maps[index].compose(newToOldIteration);
      std::optional<uint32_t> accessRelationId = internProjected(
          accessMap, accessTypeId, oldRelation->sourceTypeId,
          destinationResultType.getShape(), childType.getShape());
      std::optional<uint32_t> computeRelationId = internProjected(
          identity, iterationTypeId, accessTypeId,
          destinationResultType.getShape(), accessType.getShape());
      if (!accessRelationId || !computeRelationId)
        return relations.hasWorkLimitReached()
                   ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                   : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      outputComputeRelationIds[index] = *computeRelationId;
      outputOperandAccessRelationIds[index] = *accessRelationId;
      outputOperandAccessTypeIds[index] = accessTypeId;
    }

    for (unsigned index = dataInputCount; index < inputRelationIds.size();
         ++index) {
      std::optional<uint32_t> computeRelationId = internProjected(
          identity, iterationTypeId, destinationResultTypeId,
          destinationResultType.getShape(), destinationResultType.getShape());
      if (!computeRelationId)
        return relations.hasWorkLimitReached()
                   ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                   : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      outputComputeRelationIds[index] = *computeRelationId;
      if (recipe->initIsRead) {
        outputOperandAccessRelationIds[index] = resultRelationId;
        outputOperandAccessTypeIds[index] = destinationResultTypeId;
      } else {
        outputOperandAccessRelationIds[index] = 0;
        outputOperandAccessTypeIds[index] = 0;
      }
    }
    return WAFER_EGRAPH_CALLBACK_EXACT;
  }

  WaferEGraphRelationService getRelationService() {
    return WaferEGraphRelationService{
        this,
        [](void *context, uint32_t relationId,
           WaferEGraphRelationFacts *facts) -> uint32_t {
          return static_cast<ComponentExpression *>(context)
              ->relations.getFacts(relationId, facts);
        },
        [](void *context, uint32_t outer, uint32_t inner,
           uint32_t *result) -> uint32_t {
          return static_cast<ComponentExpression *>(context)->relations.compose(
              outer, inner, result);
        },
        [](void *context, uint32_t semanticId, uint32_t kind,
           uint32_t resultTypeId, const uint32_t *relationIds,
           const uint32_t *operandTypeIds, uint64_t relationCount,
           uint32_t dataInputCount) -> uint32_t {
          auto *component = static_cast<ComponentExpression *>(context);
          if ((relationCount && (!relationIds || !operandTypeIds)) ||
              relationCount > UINT32_MAX)
            return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
          if (!component->relations.consumeQuery())
            return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
          return component->validateCompute(
                     semanticId, kind, resultTypeId,
                     llvm::ArrayRef(relationIds, relationCount),
                     llvm::ArrayRef(operandTypeIds, relationCount),
                     dataInputCount)
                     ? WAFER_EGRAPH_CALLBACK_EXACT
                     : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        },
        [](void *context, uint32_t semanticId, uint32_t kind,
           uint32_t sourceResultTypeId, uint32_t destinationResultTypeId,
           uint32_t relationId, const uint32_t *inputRelations,
           uint64_t relationCount, uint32_t dataInputCount,
           uint32_t *outputRelations, uint32_t *reindexReadInit) -> uint32_t {
          if ((relationCount && (!inputRelations || !outputRelations)) ||
              relationCount > UINT32_MAX || !reindexReadInit)
            return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
          auto *component = static_cast<ComponentExpression *>(context);
          if (!component->relations.consumeQuery())
            return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
          return component->reindexCompute(
              semanticId, kind, sourceResultTypeId, destinationResultTypeId,
              relationId, llvm::ArrayRef(inputRelations, relationCount),
              dataInputCount,
              llvm::MutableArrayRef(outputRelations, relationCount),
              reindexReadInit);
        },
        [](void *context, uint32_t semanticId, uint32_t sourceResultTypeId,
           uint32_t destinationResultTypeId, uint32_t resultRelationId,
           const uint32_t *inputRelations, uint64_t relationCount,
           uint32_t dataInputCount, uint32_t *outputComputeRelations,
           uint32_t *outputOperandAccessRelations,
           uint32_t *outputOperandAccessTypeIds) -> uint32_t {
          if ((relationCount && (!inputRelations || !outputComputeRelations ||
                                 !outputOperandAccessRelations ||
                                 !outputOperandAccessTypeIds)) ||
              relationCount > UINT32_MAX)
            return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
          auto *component = static_cast<ComponentExpression *>(context);
          if (!component->relations.consumeQuery())
            return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
          return component->reparameterizeElementwise(
              semanticId, sourceResultTypeId, destinationResultTypeId,
              resultRelationId, llvm::ArrayRef(inputRelations, relationCount),
              dataInputCount,
              llvm::MutableArrayRef(outputComputeRelations, relationCount),
              llvm::MutableArrayRef(outputOperandAccessRelations,
                                    relationCount),
              llvm::MutableArrayRef(outputOperandAccessTypeIds, relationCount));
        },
        [](void *context, uint32_t resultTypeId, int32_t axis,
           const uint32_t *inputTypeIds, uint64_t inputCount) -> uint32_t {
          if ((inputCount && !inputTypeIds) || inputCount > UINT32_MAX)
            return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
          auto *component = static_cast<ComponentExpression *>(context);
          if (!component->relations.consumeQuery())
            return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
          return component->validateConcat(
                     resultTypeId, axis,
                     llvm::ArrayRef(inputTypeIds, inputCount))
                     ? WAFER_EGRAPH_CALLBACK_EXACT
                     : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        },
        [](void *context, uint32_t resultTypeId, int32_t destinationAxis,
           const uint32_t *relations, const uint32_t *sourceTypes,
           uint64_t inputCount, uint32_t *resultRelation,
           uint32_t *sourceConcatType, int32_t *sourceAxis) -> uint32_t {
          if ((inputCount && (!relations || !sourceTypes)) ||
              inputCount > UINT32_MAX)
            return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
          auto *component = static_cast<ComponentExpression *>(context);
          if (!component->relations.consumeQuery())
            return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
          return component->factorConcat(
              resultTypeId, destinationAxis,
              llvm::ArrayRef(relations, inputCount),
              llvm::ArrayRef(sourceTypes, inputCount), resultRelation,
              sourceConcatType, sourceAxis);
        }};
  }

  mlir::LogicalResult validateExtraction(llvm::ArrayRef<EGraphNode> expression,
                                         uint32_t rootNode) const {
    if (expression.empty() || rootNode >= expression.size() ||
        types.lookup(expression[rootNode].typeId) != root.getType())
      return mlir::failure();
    for (auto [index, node] : llvm::enumerate(expression)) {
      if (!types.lookup(node.typeId))
        return mlir::failure();
      llvm::SmallVector<uint32_t, 4> operandTypeIds;
      for (uint32_t child : node.children) {
        if (child >= index)
          return mlir::failure();
        operandTypeIds.push_back(expression[child].typeId);
      }
      switch (node.kind) {
      case EGraphNodeKind::Input:
        if (node.semanticId == 0 || node.semanticId > inputRecipes.size() ||
            inputRecipes[node.semanticId - 1].getType() !=
                types.lookup(node.typeId))
          return mlir::failure();
        break;
      case EGraphNodeKind::Access: {
        if (operandTypeIds.size() != 1 || node.relations.size() != 1)
          return mlir::failure();
        const RelationRecord *relation =
            relations.lookup(node.relations.front());
        if (!relation || relation->destinationTypeId != node.typeId ||
            relation->sourceTypeId != operandTypeIds.front() ||
            relation->materialization == RelationMaterialization::None)
          return mlir::failure();
        break;
      }
      case EGraphNodeKind::Concat:
        if (!validateConcat(node.typeId, node.axis, operandTypeIds))
          return mlir::failure();
        break;
      case EGraphNodeKind::Elementwise:
      case EGraphNodeKind::Contraction:
      case EGraphNodeKind::Reduction:
        if (!validateCompute(node.semanticId, static_cast<uint32_t>(node.kind),
                             node.typeId, node.relations, operandTypeIds,
                             node.dataInputCount))
          return mlir::failure();
        break;
      }
    }
    return mlir::success();
  }

  mlir::FailureOr<mlir::Value>
  materialize(mlir::RewriterBase &rewriter,
              llvm::ArrayRef<EGraphNode> expression, uint32_t rootNode) {
    if (expression.empty() || rootNode >= expression.size())
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 16> values;
    for (auto [index, node] : llvm::enumerate(expression)) {
      llvm::SmallVector<mlir::Value, 4> operands;
      for (uint32_t child : node.children) {
        if (child >= index)
          return mlir::failure();
        operands.push_back(values[child]);
      }
      mlir::FailureOr<mlir::Value> value =
          materializeNode(rewriter, node, operands);
      if (mlir::failed(value))
        return mlir::failure();
      values.push_back(*value);
    }
    return values[rootNode];
  }

  mlir::FailureOr<mlir::Value>
  materializeNode(mlir::RewriterBase &rewriter, const EGraphNode &node,
                  llvm::ArrayRef<mlir::Value> operands) {
    switch (node.kind) {
    case EGraphNodeKind::Input:
      if (node.semanticId == 0 || node.semanticId > inputRecipes.size())
        return mlir::failure();
      return inputRecipes[node.semanticId - 1];
    case EGraphNodeKind::Access:
      return materializeAccess(rewriter, node, operands);
    case EGraphNodeKind::Concat:
      return materializeConcat(rewriter, node, operands);
    case EGraphNodeKind::Elementwise:
    case EGraphNodeKind::Contraction:
    case EGraphNodeKind::Reduction:
      return materializeCompute(rewriter, node, operands);
    }
    return mlir::failure();
  }

  mlir::FailureOr<mlir::Value>
  materializeAccess(mlir::RewriterBase &rewriter, const EGraphNode &node,
                    llvm::ArrayRef<mlir::Value> operands) {
    if (operands.size() != 1 || node.relations.size() != 1)
      return mlir::failure();
    const RelationRecord *relation = relations.lookup(node.relations.front());
    auto resultType = types.lookupRankedTensor(node.typeId);
    if (!relation || !resultType ||
        relation->sourceTypeId != types.intern(operands.front().getType()))
      return mlir::failure();
    if (relation->materialization == RelationMaterialization::Identity)
      return operands.front();
    rewriter.setInsertionPoint(root.getDefiningOp());
    mlir::Location location = root.getLoc();
    if (relation->materialization == RelationMaterialization::Reshape) {
      auto sourceType =
          mlir::cast<mlir::RankedTensorType>(operands.front().getType());
      auto reassociation =
          mlir::getReassociationIndicesForReshape(sourceType, resultType);
      if (!reassociation)
        return mlir::failure();
      if (sourceType.getRank() > resultType.getRank())
        return rewriter
            .create<mlir::tensor::CollapseShapeOp>(
                location, resultType, operands.front(), *reassociation)
            .getResult();
      return rewriter
          .create<mlir::tensor::ExpandShapeOp>(location, resultType,
                                               operands.front(), *reassociation)
          .getResult();
    }
    if (relation->materialization != RelationMaterialization::ProjectedMap ||
        !relation->projectedMap)
      return mlir::failure();
    mlir::Value output =
        rewriter
            .create<mlir::tensor::EmptyOp>(location, resultType.getShape(),
                                           resultType.getElementType())
            .getResult();
    mlir::AffineMap identity = mlir::AffineMap::getMultiDimIdentityMap(
        resultType.getRank(), rewriter.getContext());
    llvm::SmallVector<mlir::utils::IteratorType, 6> iterators(
        resultType.getRank(), mlir::utils::IteratorType::parallel);
    auto generic = rewriter.create<mlir::linalg::GenericOp>(
        location, mlir::TypeRange{resultType}, operands,
        mlir::ValueRange{output},
        llvm::ArrayRef<mlir::AffineMap>{*relation->projectedMap, identity},
        iterators,
        [&](mlir::OpBuilder &builder, mlir::Location nestedLocation,
            mlir::ValueRange arguments) {
          builder.create<mlir::linalg::YieldOp>(nestedLocation,
                                                arguments.front());
        });
    return generic.getResult(0);
  }

  mlir::FailureOr<mlir::Value>
  materializeConcat(mlir::RewriterBase &rewriter, const EGraphNode &node,
                    llvm::ArrayRef<mlir::Value> operands) {
    llvm::SmallVector<uint32_t, 4> operandTypes;
    for (mlir::Value operand : operands)
      operandTypes.push_back(types.intern(operand.getType()));
    if (!validateConcat(node.typeId, node.axis, operandTypes))
      return mlir::failure();
    auto resultType = types.lookupRankedTensor(node.typeId);
    rewriter.setInsertionPoint(root.getDefiningOp());
    mlir::Location location = root.getLoc();
    mlir::Value result =
        rewriter
            .create<mlir::tensor::EmptyOp>(location, resultType.getShape(),
                                           resultType.getElementType())
            .getResult();
    llvm::SmallVector<mlir::OpFoldResult, 6> offsets(resultType.getRank(),
                                                     rewriter.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult, 6> strides(resultType.getRank(),
                                                     rewriter.getIndexAttr(1));
    int64_t nextOffset = 0;
    for (mlir::Value operand : operands) {
      auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
      llvm::SmallVector<mlir::OpFoldResult, 6> sizes;
      for (int64_t extent : operandType.getShape())
        sizes.push_back(rewriter.getIndexAttr(extent));
      offsets[node.axis] = rewriter.getIndexAttr(nextOffset);
      nextOffset += operandType.getDimSize(node.axis);
      result = rewriter
                   .create<mlir::tensor::InsertSliceOp>(
                       location, operand, result, offsets, sizes, strides)
                   .getResult();
    }
    return result;
  }

  mlir::FailureOr<mlir::Value>
  materializeCompute(mlir::RewriterBase &rewriter, const EGraphNode &node,
                     llvm::ArrayRef<mlir::Value> operands) {
    const ComputeRecipe *recipe = getRecipe(node.semanticId);
    auto resultType = types.lookupRankedTensor(node.typeId);
    llvm::SmallVector<uint32_t, 4> operandTypeIds;
    for (mlir::Value operand : operands)
      operandTypeIds.push_back(types.intern(operand.getType()));
    if (!recipe ||
        !validateCompute(node.semanticId, static_cast<uint32_t>(node.kind),
                         node.typeId, node.relations, operandTypeIds,
                         node.dataInputCount))
      return mlir::failure();
    llvm::SmallVector<mlir::AffineMap, 4> maps;
    if (!getMaps(node.relations, maps))
      return mlir::failure();
    if (node.typeId == recipe->resultTypeId &&
        llvm::equal(operands, recipe->operation->getOperands()) &&
        llvm::equal(maps, recipe->originalMaps))
      return recipe->operation->getResult(0);

    llvm::SmallVector<mlir::Value, 4> inputs(
        operands.take_front(node.dataInputCount).begin(),
        operands.take_front(node.dataInputCount).end());
    llvm::SmallVector<mlir::Value, 2> outputs(
        operands.drop_front(node.dataInputCount).begin(),
        operands.drop_front(node.dataInputCount).end());
    rewriter.setInsertionPoint(root.getDefiningOp());
    if (node.typeId != recipe->resultTypeId) {
      if (outputs.size() != 1)
        return mlir::failure();
      if (!recipe->initIsRead)
        outputs.front() = rewriter
                              .create<mlir::tensor::EmptyOp>(
                                  root.getLoc(), resultType.getShape(),
                                  resultType.getElementType())
                              .getResult();
    }

    llvm::SmallVector<mlir::NamedAttribute, 4> attributes;
    for (mlir::NamedAttribute attribute :
         recipe->operation->getDiscardableAttrs())
      if (attribute.getName() !=
          mlir::linalg::LinalgDialect::kMemoizedIndexingMapsAttrName)
        attributes.push_back(attribute);

    MatmulVariant variant =
        classifyMatmulMaps(maps, recipe->iteratorTypes, resultType.getRank());
    if (recipe->kind == EGraphNodeKind::Contraction &&
        recipe->hasCanonicalContractionPayload &&
        variant != MatmulVariant::None) {
      mlir::Operation *created = createMatmulVariant(rewriter, root.getLoc(),
                                                     variant, inputs, outputs);
      if (created) {
        created->setDiscardableAttrs(attributes);
        return created->getResult(0);
      }
    }

    auto generic = rewriter.create<mlir::linalg::GenericOp>(
        recipe->operation->getLoc(), mlir::TypeRange{resultType}, inputs,
        outputs, maps, recipe->iteratorTypes, /*bodyBuild=*/nullptr,
        attributes);
    rewriter.cloneRegionBefore(recipe->operation->getRegion(0),
                               generic.getRegion(),
                               generic.getRegion().begin());
    return generic.getResult(0);
  }

  static void addStatistics(
      StructuredGraphNormalizationStatistics &target,
      const structured_graph_normalization::EGraphStatistics &source) {
    target.relationQueries += source.relationQueries;
    target.eNodes += source.eNodes;
    target.eClasses += source.eClasses;
    target.rewriteMatches += source.rewriteMatches;
    target.eClassMerges += source.eClassMerges;
    target.rebuildWork += source.rebuildWork;
    target.iterations += source.iterations;
    target.extractionWork += source.extractionWork;
    target.identityApplications += source.identityApplications;
    target.compositionApplications += source.compositionApplications;
    target.computeAbsorptionApplications +=
        source.computeAbsorptionApplications;
    target.concatApplications += source.concatApplications;
    target.resultReindexApplications += source.resultReindexApplications;
    target.abiInputRecords += source.inputRecords;
    target.abiOutputRecords += source.outputRecords;
    target.abiInputBytes += source.inputBytes;
    target.abiOutputBytes += source.outputBytes;
  }

  mlir::Value root;
  const StructuredGraphNormalizationOptions &options;
  TypeStore types;
  RelationStore relations;
  llvm::DenseMap<mlir::Operation *, uint32_t> sourceOrder;
  llvm::DenseMap<mlir::Value, uint32_t> imported;
  llvm::SmallVector<mlir::Value, 16> inputRecipes;
  llvm::SmallVector<ComputeRecipe, 16> computeRecipes;
  llvm::SmallVector<EGraphNode, 16> nodes;
  bool hasPotentialRewrite = false;
};

llvm::SmallVector<mlir::Value, 16>
collectComponentRoots(mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::Value, 16> roots;
  function.walk([&](mlir::Operation *operation) {
    if (!isSupportedProducer(operation))
      return;
    mlir::Value result = operation->getResult(0);
    if (!result.hasOneUse() || !isSupportedProducer(*result.getUsers().begin()))
      roots.push_back(result);
  });
  return roots;
}

uint64_t countOperations(mlir::Operation *root) {
  uint64_t count = 0;
  root->walk([&](mlir::Operation *) { ++count; });
  return count;
}

struct LogicalTransformCount {
  uint64_t accesses = 0;
  uint64_t concats = 0;
};

LogicalTransformCount countLogicalTransforms(mlir::func::FuncOp function) {
  LogicalTransformCount count;
  function.walk([&](mlir::Operation *operation) {
    if (getAccessDescription(operation))
      ++count.accesses;
    if (getConcatDescription(operation))
      ++count.concats;
  });
  return count;
}

bool isStrictlyBetter(LogicalTransformCount after,
                      LogicalTransformCount before) {
  return after.accesses < before.accesses ||
         (after.accesses == before.accesses && after.concats < before.concats);
}

mlir::FailureOr<StructuredGraphNormalizationOutcome>
normalizeFunction(mlir::func::FuncOp function,
                  const StructuredGraphNormalizationOptions &options,
                  StructuredGraphNormalizationStatistics &statistics) {
  mlir::IRRewriter rewriter(function.getContext());
  bool changed = false;
  bool exhausted = false;
  while (true) {
    LogicalTransformCount before = countLogicalTransforms(function);
    bool roundChanged = false;
    const uint64_t initialMultiUsePropagations =
        statistics.multiUseAccessPropagations;
    if (mlir::failed(
            propagateMultiUseProjectedAccesses(function, rewriter, statistics)))
      return mlir::failure();
    roundChanged |=
        statistics.multiUseAccessPropagations != initialMultiUsePropagations;
    for (mlir::Value root : collectComponentRoots(function)) {
      ComponentExpression component(root, options);
      mlir::FailureOr<StructuredGraphNormalizationOutcome> outcome =
          component.apply(rewriter, statistics);
      if (mlir::failed(outcome))
        return mlir::failure();
      roundChanged |= *outcome == StructuredGraphNormalizationOutcome::Changed;
      exhausted |=
          *outcome == StructuredGraphNormalizationOutcome::BudgetExhausted;
    }
    if (!roundChanged)
      break;
    LogicalTransformCount after = countLogicalTransforms(function);
    if (!isStrictlyBetter(after, before))
      return mlir::failure();
    changed = true;
  }
  if (changed)
    return StructuredGraphNormalizationOutcome::Changed;
  if (exhausted)
    return StructuredGraphNormalizationOutcome::BudgetExhausted;
  return StructuredGraphNormalizationOutcome::Unchanged;
}

struct NormalizeStructuredTensorGraphPass final
    : impl::NormalizeStructuredTensorGraphPassBase<
          NormalizeStructuredTensorGraphPass> {
  using impl::NormalizeStructuredTensorGraphPassBase<
      NormalizeStructuredTensorGraphPass>::
      NormalizeStructuredTensorGraphPassBase;

  void runOnOperation() final {
    StructuredGraphNormalizationStatistics statistics;
    mlir::FailureOr<StructuredGraphNormalizationOutcome> outcome =
        normalizeStructuredTensorGraph(
            getOperation(),
            StructuredGraphNormalizationOptions{maximumRelationQueries,
                                                maximumENodes, maximumMatches,
                                                maximumIterations},
            &statistics);
    if (mlir::failed(outcome)) {
      getOperation().emitError(
          "structured tensor graph normalization failed its typed rewrite "
          "contract");
      signalPassFailure();
      return;
    }
    numComponents += statistics.components;
    numInputOperations += statistics.inputOperations;
    numOutputOperations += statistics.outputOperations;
    numChangedComponents += statistics.changedComponents;
    numMultiRuleChangedComponents += statistics.multiRuleChangedComponents;
    numBudgetExhaustedComponents += statistics.budgetExhaustedComponents;
    numENodes += statistics.eNodes;
    numEClasses += statistics.eClasses;
    numRewriteMatches += statistics.rewriteMatches;
    numEClassMerges += statistics.eClassMerges;
    numRebuildWork += statistics.rebuildWork;
    numIterations += statistics.iterations;
    numExtractionWork += statistics.extractionWork;
    numAccessTransformsRemoved += statistics.accessTransformsRemoved;
    numMultiUseAccessPropagations += statistics.multiUseAccessPropagations;
    numConcatTransformsRemoved += statistics.concatTransformsRemoved;
    numRelationQueries += statistics.relationQueries;
    numIdentityApplications += statistics.identityApplications;
    numCompositionApplications += statistics.compositionApplications;
    numComputeAbsorptionApplications +=
        statistics.computeAbsorptionApplications;
    numConcatApplications += statistics.concatApplications;
    numResultReindexApplications += statistics.resultReindexApplications;
    numABIInputRecords += statistics.abiInputRecords;
    numABIOutputRecords += statistics.abiOutputRecords;
    numABIInputBytes += statistics.abiInputBytes;
    numABIOutputBytes += statistics.abiOutputBytes;
  }
};

} // namespace

mlir::FailureOr<StructuredGraphNormalizationOutcome>
normalizeStructuredTensorGraph(
    mlir::func::FuncOp function,
    const StructuredGraphNormalizationOptions &options,
    StructuredGraphNormalizationStatistics *outputStatistics) {
  if (!function || options.maximumRelationQueries == 0 ||
      options.maximumENodes == 0 || options.maximumMatches == 0 ||
      options.maximumIterations == 0)
    return mlir::failure();
  StructuredGraphNormalizationStatistics local;
  local.inputOperations = countOperations(function);
  mlir::FailureOr<StructuredGraphNormalizationOutcome> outcome =
      normalizeFunction(function, options, local);
  if (mlir::failed(outcome) || mlir::failed(mlir::verify(function)))
    return mlir::failure();
  local.outputOperations = countOperations(function);
  if (outputStatistics)
    *outputStatistics = local;
  return outcome;
}

} // namespace wafer
