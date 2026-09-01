//===- NormalizeStructuredGraph.cpp - Relation-driven normalization -----===//

#include "Wafer/Transforms/Linalg/StructuredGraphNormalization.h"

#include "StructuredGraphEGraph.h"
#include "Wafer/Analysis/Linalg/IndexRelation.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
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
  llvm::SmallVector<uint32_t, 4> compositionFactors;
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
    records.back().compositionFactors.push_back(records.size());
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
    if (record->materialization == RelationMaterialization::Reshape &&
        record->relation.hasCanonicalRowMajorReshapeConstruction())
      flags |= WAFER_EGRAPH_RELATION_CANONICAL_RESHAPE;
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

    std::vector<uint32_t> factors(outer->compositionFactors.begin(),
                                  outer->compositionFactors.end());
    factors.insert(factors.end(), inner->compositionFactors.begin(),
                   inner->compositionFactors.end());
    bool reducedFactors = false;
    while (factors.size() >= 2) {
      bool removed = false;
      for (size_t length = factors.size(); length >= 2 && !removed; --length)
        for (size_t begin = 0; begin + length <= factors.size(); ++begin) {
          std::vector<uint32_t> candidate(factors.begin() + begin,
                                          factors.begin() + begin + length);
          auto found = compositionByFactors.find(candidate);
          const RelationRecord *record = found == compositionByFactors.end()
                                             ? nullptr
                                             : lookup(found->second);
          if (!record || !record->identity)
            continue;
          factors.erase(factors.begin() + begin,
                        factors.begin() + begin + length);
          reducedFactors = removed = true;
          break;
        }
      if (!removed)
        break;
    }
    if (auto found = compositionByFactors.find(factors);
        found != compositionByFactors.end()) {
      compositionCache.try_emplace(cacheKey, found->second);
      *resultId = found->second;
      return WAFER_EGRAPH_CALLBACK_EXACT;
    }

    auto destination = types.lookupRankedTensor(outer->destinationTypeId);
    auto source = types.lookupRankedTensor(inner->sourceTypeId);
    if (!destination || !source) {
      compositionCache.try_emplace(cacheKey, 0);
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    }
    IndexRelationResult composed;
    if (factors.empty() && outer->destinationTypeId == inner->sourceTypeId) {
      composed = IndexRelation::identity(destination.getShape());
    } else if (factors.size() == 1) {
      const RelationRecord *record = lookup(factors.front());
      if (!record || record->destinationTypeId != outer->destinationTypeId ||
          record->sourceTypeId != inner->sourceTypeId) {
        compositionCache.try_emplace(cacheKey, 0);
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      }
      compositionCache.try_emplace(cacheKey, factors.front());
      compositionByFactors.try_emplace(factors, factors.front());
      *resultId = factors.front();
      return WAFER_EGRAPH_CALLBACK_EXACT;
    } else if (reducedFactors) {
      const RelationRecord *first = lookup(factors[0]);
      const RelationRecord *second = lookup(factors[1]);
      if (!first || !second) {
        compositionCache.try_emplace(cacheKey, 0);
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      }
      composed = first->relation.compose(second->relation);
      for (size_t index = 2; composed.isExact() && index < factors.size();
           ++index) {
        const RelationRecord *next = lookup(factors[index]);
        if (!next) {
          compositionCache.try_emplace(cacheKey, 0);
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        }
        composed = composed.get()->compose(next->relation);
      }
    } else if (outer->materialization == RelationMaterialization::Reshape &&
               inner->materialization == RelationMaterialization::Reshape) {
      composed = IndexRelation::staticReshape(destination.getShape(),
                                              source.getShape());
    } else {
      composed = outer->relation.compose(inner->relation);
    }
    if (!composed.isExact()) {
      compositionCache.try_emplace(cacheKey, 0);
      return statusFor(composed.status);
    }

    std::optional<uint32_t> id;
    if (composed.get()->hasCanonicalRowMajorReshapeConstruction() &&
        (outer->destinationTypeId == inner->sourceTypeId ||
         canMaterializeStaticReshape(source, destination)))
      id = intern(std::move(*composed.get()), outer->destinationTypeId,
                  inner->sourceTypeId, std::nullopt,
                  RelationMaterialization::Reshape, /*total=*/true);

    std::optional<mlir::AffineMap> projected;
    if (!id) {
      if (outer->projectedMap && inner->projectedMap)
        projected = inner->projectedMap->compose(*outer->projectedMap);
      else
        projected = composed.get()->getProjectedAffineMap(context);
    }
    if (!id && projected) {
      IndexRelationResult canonical = IndexRelation::fromAffineMap(
          *projected, destination.getShape(), source.getShape());
      if (canonical.isExact() &&
          canonical.get()->hasTotalBoundedAffineMapConstruction())
        id = intern(std::move(*canonical.get()), outer->destinationTypeId,
                    inner->sourceTypeId, projected,
                    RelationMaterialization::ProjectedMap, /*total=*/true);
    }
    if (!id && !workLimitReached)
      id = intern(std::move(*composed.get()), outer->destinationTypeId,
                  inner->sourceTypeId, std::nullopt,
                  RelationMaterialization::None, /*total=*/true);
    if (!id)
      compositionCache.try_emplace(cacheKey, 0);
    if (!id)
      return workLimitReached ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                              : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    compositionCache.try_emplace(cacheKey, *id);
    compositionByFactors.try_emplace(factors, *id);
    if (RelationRecord *record = lookup(*id);
        record && record->compositionFactors.size() == 1 &&
        record->compositionFactors.front() == *id &&
        record->materialization == RelationMaterialization::None)
      record->compositionFactors.assign(factors.begin(), factors.end());
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
  std::map<std::vector<uint32_t>, uint32_t> compositionByFactors;
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
  llvm::SmallVector<mlir::Operation *, 4> implementation;
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
    mlir::Operation *operation = nullptr;
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
    reversePieces.push_back(Piece{current.getOperation(), current.getSource(),
                                  std::move(*offsets), std::move(*sizes)});

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

  ConcatDescription result{resultType, *axis, {}, {}};
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
    result.implementation.push_back(piece.operation);
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

class ComponentExpression {
public:
  ComponentExpression(llvm::ArrayRef<mlir::Value> componentRoots,
                      llvm::ArrayRef<mlir::Operation *> componentOperations,
                      mlir::Operation *anchor,
                      const StructuredGraphNormalizationOptions &options)
      : roots(componentRoots.begin(), componentRoots.end()), anchor(anchor),
        options(options), relations(types, roots.front().getContext(),
                                    options.maximumRelationQueries) {
    for (mlir::Operation *operation : componentOperations)
      this->componentOperations.insert(operation);
    mlir::Operation *parent = roots.front().getParentBlock()->getParentOp();
    uint32_t order = 1;
    parent->walk([&](mlir::Operation *operation) {
      sourceOrder.try_emplace(operation, order++);
    });
  }

  mlir::FailureOr<StructuredGraphNormalizationOutcome>
  apply(mlir::RewriterBase &rewriter,
        StructuredGraphNormalizationStatistics &statistics) {
    llvm::SmallVector<uint32_t, 4> rootNodes;
    for (mlir::Value root : roots)
      rootNodes.push_back(import(root, /*allowProducer=*/true));
    if (relations.hasWorkLimitReached()) {
      ++statistics.components;
      ++statistics.budgetExhaustedComponents;
      return StructuredGraphNormalizationOutcome::BudgetExhausted;
    }
    if (nodes.size() == 1 || !hasPotentialRewrite)
      return StructuredGraphNormalizationOutcome::Unchanged;
    ++statistics.components;
    statistics.multiRootComponents += roots.size() > 1;
    EGraphOutcome result = structured_graph_normalization::runEGraph(
        nodes, rootNodes,
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
    if (mlir::failed(validateExtraction(result.expression, result.rootNodes)))
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *, 16> insertedOperations;
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> replacements =
        mlir::failure();
    {
      mlir::OpBuilder::Listener *previousListener = rewriter.getListener();
      InsertedOperationRecorder recorder(roots.front().getParentBlock(),
                                         previousListener, insertedOperations);
      rewriter.setListener(&recorder);
      auto restoreListener = llvm::make_scope_exit(
          [&] { rewriter.setListener(previousListener); });
      replacements = materialize(rewriter, result.expression, result.rootNodes);
    }
    if (mlir::failed(replacements) || replacements->size() != roots.size() ||
        llvm::any_of(llvm::zip_equal(roots, *replacements),
                     [](auto values) {
                       auto [root, replacement] = values;
                       return root.getType() != replacement.getType();
                     }) ||
        llvm::all_of(llvm::zip_equal(roots, *replacements), [](auto values) {
          return std::get<0>(values) == std::get<1>(values);
        })) {
      for (mlir::Operation *operation : llvm::reverse(insertedOperations))
        rewriter.eraseOp(operation);
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Operation *, 16> oldClosure;
    for (mlir::Value root : roots) {
      mlir::Operation *owner = root.getDefiningOp();
      oldClosure.push_back(owner);
      for (mlir::Value operand : owner->getOperands())
        if (mlir::Operation *producer = operand.getDefiningOp())
          oldClosure.push_back(producer);
    }
    for (auto [root, replacement] : llvm::zip_equal(roots, *replacements))
      if (root != replacement)
        rewriter.replaceAllUsesWith(root, replacement);
    eraseDeadProducerClosure(rewriter, oldClosure);
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
    appliedRuleKinds +=
        result.statistics.reshapeThroughComputeApplications != 0;
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
        !componentOperations.contains(operation)) {
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
        loopShape, mlir::IntegerType::get(roots.front().getContext(), 1));
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

  bool getEffectiveIteratorTypes(
      const ComputeRecipe &recipe, llvm::ArrayRef<mlir::AffineMap> maps,
      mlir::RankedTensorType iterationType,
      llvm::SmallVectorImpl<mlir::utils::IteratorType> &iteratorTypes) const {
    iteratorTypes.clear();
    if (!iterationType || maps.empty() ||
        maps.size() != recipe.originalMaps.size())
      return false;
    if (iterationType.getRank() ==
        static_cast<int64_t>(recipe.iteratorTypes.size())) {
      llvm::append_range(iteratorTypes, recipe.iteratorTypes);
      return true;
    }

    auto inferFromDpsResults =
        [&](llvm::ArrayRef<mlir::AffineMap> candidate, unsigned rank,
            llvm::SmallVectorImpl<mlir::utils::IteratorType> &result) {
          llvm::SmallVector<bool, 6> resultAxes(rank, false);
          for (mlir::AffineMap map :
               candidate.drop_front(recipe.dataInputCount)) {
            if (map.getNumDims() != rank)
              return false;
            for (mlir::AffineExpr expression : map.getResults()) {
              auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
              if (!dimension || dimension.getPosition() >= rank)
                return false;
              resultAxes[dimension.getPosition()] = true;
            }
          }
          result.clear();
          for (bool appearsInResult : resultAxes)
            result.push_back(appearsInResult
                                 ? mlir::utils::IteratorType::parallel
                                 : mlir::utils::IteratorType::reduction);
          return true;
        };

    auto originalIterationType =
        types.lookupRankedTensor(recipe.iterationTypeId);
    llvm::SmallVector<mlir::utils::IteratorType, 6> inferredOriginal;
    if (!originalIterationType ||
        !inferFromDpsResults(recipe.originalMaps,
                             originalIterationType.getRank(),
                             inferredOriginal) ||
        !llvm::equal(inferredOriginal, recipe.iteratorTypes) ||
        !inferFromDpsResults(maps, iterationType.getRank(), iteratorTypes))
      return false;

    int64_t originalReductionElements = 1;
    int64_t newReductionElements = 1;
    for (auto [extent, iterator] : llvm::zip_equal(
             originalIterationType.getShape(), recipe.iteratorTypes))
      if (iterator == mlir::utils::IteratorType::reduction &&
          (extent <= 0 || llvm::MulOverflow(originalReductionElements, extent,
                                            originalReductionElements)))
        return false;
    for (auto [extent, iterator] :
         llvm::zip_equal(iterationType.getShape(), iteratorTypes))
      if (iterator == mlir::utils::IteratorType::reduction &&
          (extent <= 0 || llvm::MulOverflow(newReductionElements, extent,
                                            newReductionElements)))
        return false;
    return originalReductionElements == newReductionElements;
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
    if (!iterationType || maps.empty() ||
        maps.back().getNumResults() != resultType.getRank())
      return false;
    llvm::SmallVector<mlir::utils::IteratorType, 6> iteratorTypes;
    if (!getEffectiveIteratorTypes(*recipe, maps, iterationType, iteratorTypes))
      return false;
    bool mapsUnchanged = resultTypeId == recipe->resultTypeId &&
                         llvm::equal(maps, recipe->originalMaps);
    if (mapsUnchanged)
      return true;
    for (mlir::AffineMap map : maps)
      if (map.getNumDims() != static_cast<unsigned>(iterationType.getRank()) ||
          !map.isProjectedPermutation(/*allowZeroInResults=*/true))
        return false;
    // Generic Linalg verification requires the concatenated operand maps to
    // determine every loop bound.  In particular, absorbing a broadcast into
    // a reduction can otherwise erase the only map that names a reduction
    // dimension even though the scalar payload is still meaningful.
    if (!mlir::inversePermutation(mlir::concatAffineMaps(maps)))
      return false;

    if (recipe->kind == EGraphNodeKind::Contraction) {
      if (!recipe->hasCanonicalContractionPayload)
        return false;
      MatmulVariant variant =
          classifyMatmulMaps(maps, iteratorTypes, resultType.getRank());
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
        destinationResultType.getRank(), roots.front().getContext());
    auto iterationType = mlir::RankedTensorType::get(
        destinationResultType.getShape(),
        mlir::IntegerType::get(roots.front().getContext(), 1));
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

  uint32_t reparameterizeReshapeCompute(
      uint32_t semanticId, uint32_t kind, uint32_t resultTypeId,
      uint32_t accessOperand, uint32_t accessRelationId,
      llvm::ArrayRef<uint32_t> inputRelationIds, uint32_t dataInputCount,
      uint32_t *innerResultTypeId, uint32_t *outerResultRelationId,
      llvm::MutableArrayRef<uint32_t> outputComputeRelationIds,
      llvm::MutableArrayRef<uint32_t> outputOperandAccessRelationIds,
      llvm::MutableArrayRef<uint32_t> outputOperandAccessTypeIds) {
    const ComputeRecipe *recipe = getRecipe(semanticId);
    const RelationRecord *accessRelation = relations.lookup(accessRelationId);
    auto oldResultType = types.lookupRankedTensor(resultTypeId);
    if (!recipe || kind != static_cast<uint32_t>(recipe->kind) ||
        recipe->operation->getNumResults() != 1 || !oldResultType ||
        !accessRelation ||
        !accessRelation->relation.hasCanonicalRowMajorReshapeConstruction() ||
        accessRelation->materialization != RelationMaterialization::Reshape ||
        accessOperand >= dataInputCount ||
        inputRelationIds.size() != recipe->originalMaps.size() ||
        inputRelationIds.size() != outputComputeRelationIds.size() ||
        inputRelationIds.size() != outputOperandAccessRelationIds.size() ||
        inputRelationIds.size() != outputOperandAccessTypeIds.size() ||
        dataInputCount != recipe->dataInputCount || !innerResultTypeId ||
        !outerResultRelationId || inputRelationIds.size() != dataInputCount + 1)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(recipe->operation);
    if (!linalg || linalg.hasIndexSemantics())
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    llvm::SmallVector<mlir::AffineMap, 4> maps;
    if (!getMaps(inputRelationIds, maps) || maps.empty())
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    const RelationRecord *selectedRelation =
        relations.lookup(inputRelationIds[accessOperand]);
    auto accessType =
        types.lookupRankedTensor(accessRelation->destinationTypeId);
    auto sourceType = types.lookupRankedTensor(accessRelation->sourceTypeId);
    auto oldIterationType =
        selectedRelation
            ? types.lookupRankedTensor(selectedRelation->destinationTypeId)
            : mlir::RankedTensorType{};
    if (!selectedRelation || !accessType || !sourceType || !oldIterationType ||
        selectedRelation->sourceTypeId != accessRelation->destinationTypeId ||
        sourceType.getRank() == accessType.getRank() ||
        sourceType.getElementType() != accessType.getElementType() ||
        sourceType.getEncoding() != accessType.getEncoding() ||
        oldIterationType.getRank() !=
            static_cast<int64_t>(recipe->iteratorTypes.size()) ||
        maps[accessOperand].getNumResults() != accessType.getRank() ||
        !canMaterializeStaticReshape(sourceType, accessType))
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    llvm::SmallVector<unsigned, 6> operandToOldIterator;
    llvm::SmallVector<bool, 6> seenOldIterator(oldIterationType.getRank(),
                                               false);
    for (mlir::AffineExpr expression : maps[accessOperand].getResults()) {
      auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!dimension || dimension.getPosition() >= seenOldIterator.size() ||
          seenOldIterator[dimension.getPosition()])
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      seenOldIterator[dimension.getPosition()] = true;
      operandToOldIterator.push_back(dimension.getPosition());
    }
    std::optional<llvm::SmallVector<mlir::ReassociationIndices>> reassociation =
        mlir::getReassociationIndicesForReshape(sourceType, accessType);
    if (!reassociation)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    llvm::SmallVector<llvm::SmallVector<unsigned, 2>, 6> oldToNew(
        oldIterationType.getRank());
    llvm::SmallVector<llvm::SmallVector<unsigned, 2>, 6> newToOld;
    llvm::SmallVector<int64_t, 6> newIterationShape;
    llvm::SmallVector<mlir::utils::IteratorType, 6> newIteratorTypes;
    auto addIterator = [&](int64_t extent, mlir::utils::IteratorType iterator,
                           llvm::ArrayRef<unsigned> oldAxes) {
      unsigned newAxis = newIterationShape.size();
      newIterationShape.push_back(extent);
      newIteratorTypes.push_back(iterator);
      newToOld.emplace_back(oldAxes.begin(), oldAxes.end());
      for (unsigned oldAxis : oldAxes)
        oldToNew[oldAxis].push_back(newAxis);
    };

    if (sourceType.getRank() > accessType.getRank()) {
      if (reassociation->size() != static_cast<size_t>(accessType.getRank()))
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      llvm::SmallVector<int, 6> oldIteratorToOperand(oldIterationType.getRank(),
                                                     -1);
      for (auto [operandAxis, oldIterator] :
           llvm::enumerate(operandToOldIterator))
        oldIteratorToOperand[oldIterator] = operandAxis;
      for (unsigned oldIterator = 0;
           oldIterator < static_cast<unsigned>(oldIterationType.getRank());
           ++oldIterator) {
        int operandAxis = oldIteratorToOperand[oldIterator];
        if (operandAxis < 0) {
          addIterator(oldIterationType.getDimSize(oldIterator),
                      recipe->iteratorTypes[oldIterator], {oldIterator});
          continue;
        }
        llvm::ArrayRef<int64_t> sourceAxes = (*reassociation)[operandAxis];
        if (sourceAxes.empty())
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        for (int64_t sourceAxis : sourceAxes) {
          if (sourceAxis < 0 || sourceAxis >= sourceType.getRank())
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          addIterator(sourceType.getDimSize(sourceAxis),
                      recipe->iteratorTypes[oldIterator], {oldIterator});
        }
      }
    } else {
      if (reassociation->size() != static_cast<size_t>(sourceType.getRank()))
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      llvm::SmallVector<int, 6> groupAtOldIterator(oldIterationType.getRank(),
                                                   -1);
      llvm::SmallVector<bool, 6> consumed(oldIterationType.getRank(), false);
      llvm::SmallVector<llvm::SmallVector<unsigned, 2>, 6> groupOldIterators;
      for (auto [sourceAxis, operandAxes] : llvm::enumerate(*reassociation)) {
        llvm::SmallVector<unsigned, 2> oldIterators;
        for (int64_t operandAxis : operandAxes) {
          if (operandAxis < 0 ||
              operandAxis >= static_cast<int64_t>(operandToOldIterator.size()))
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          oldIterators.push_back(operandToOldIterator[operandAxis]);
        }
        bool consecutive = true;
        for (auto pair :
             llvm::zip(oldIterators, llvm::drop_begin(oldIterators)))
          consecutive &= std::get<1>(pair) == std::get<0>(pair) + 1;
        if (oldIterators.empty() || !llvm::is_sorted(oldIterators) ||
            !consecutive ||
            llvm::any_of(llvm::drop_begin(oldIterators),
                         [&](unsigned oldIterator) {
                           return recipe->iteratorTypes[oldIterator] !=
                                  recipe->iteratorTypes[oldIterators.front()];
                         }))
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        unsigned group = groupOldIterators.size();
        for (unsigned oldIterator : oldIterators) {
          if (consumed[oldIterator])
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          consumed[oldIterator] = true;
        }
        groupAtOldIterator[oldIterators.front()] = group;
        groupOldIterators.push_back(std::move(oldIterators));
        (void)sourceAxis;
      }
      for (unsigned oldIterator = 0;
           oldIterator < static_cast<unsigned>(oldIterationType.getRank());
           ++oldIterator) {
        int group = groupAtOldIterator[oldIterator];
        if (group >= 0) {
          llvm::ArrayRef<unsigned> oldIterators = groupOldIterators[group];
          addIterator(sourceType.getDimSize(group),
                      recipe->iteratorTypes[oldIterator], oldIterators);
          oldIterator = oldIterators.back();
          continue;
        }
        if (consumed[oldIterator])
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        addIterator(oldIterationType.getDimSize(oldIterator),
                    recipe->iteratorTypes[oldIterator], {oldIterator});
      }
    }

    int64_t oldReductionElements = 1;
    int64_t newReductionElements = 1;
    for (auto [extent, iterator] :
         llvm::zip_equal(oldIterationType.getShape(), recipe->iteratorTypes))
      if (iterator == mlir::utils::IteratorType::reduction &&
          (extent <= 0 || llvm::MulOverflow(oldReductionElements, extent,
                                            oldReductionElements)))
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    for (auto [extent, iterator] :
         llvm::zip_equal(newIterationShape, newIteratorTypes))
      if (iterator == mlir::utils::IteratorType::reduction &&
          (extent <= 0 || llvm::MulOverflow(newReductionElements, extent,
                                            newReductionElements)))
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    if (oldReductionElements != newReductionElements)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    struct ReparameterizedOperand {
      mlir::RankedTensorType oldType;
      mlir::RankedTensorType newType;
      mlir::AffineMap map;
    };
    llvm::SmallVector<ReparameterizedOperand, 4> reparameterized;
    for (auto [relationId, oldMap] : llvm::zip_equal(inputRelationIds, maps)) {
      const RelationRecord *oldRelation = relations.lookup(relationId);
      auto oldOperandType =
          oldRelation ? types.lookupRankedTensor(oldRelation->sourceTypeId)
                      : mlir::RankedTensorType{};
      if (!oldOperandType || oldMap.getNumResults() != oldOperandType.getRank())
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      llvm::SmallVector<int64_t, 6> newShape;
      llvm::SmallVector<mlir::AffineExpr, 6> newResults;
      for (unsigned operandAxis = 0;
           operandAxis < static_cast<unsigned>(oldOperandType.getRank());) {
        mlir::AffineExpr expression = oldMap.getResult(operandAxis);
        if (auto constant =
                mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
          if (constant.getValue() != 0)
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          newShape.push_back(oldOperandType.getDimSize(operandAxis));
          newResults.push_back(
              mlir::getAffineConstantExpr(0, roots.front().getContext()));
          ++operandAxis;
          continue;
        }
        auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!dimension || dimension.getPosition() >= oldToNew.size())
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        unsigned oldIterator = dimension.getPosition();
        if (oldToNew[oldIterator].empty())
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        unsigned newIterator = oldToNew[oldIterator].front();
        llvm::ArrayRef<unsigned> collapsedOld = newToOld[newIterator];
        if (collapsedOld.size() > 1) {
          if (oldIterator != collapsedOld.front() ||
              operandAxis + collapsedOld.size() >
                  static_cast<unsigned>(oldOperandType.getRank()))
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          for (auto [offset, requiredOldIterator] :
               llvm::enumerate(collapsedOld)) {
            auto groupedDimension = mlir::dyn_cast<mlir::AffineDimExpr>(
                oldMap.getResult(operandAxis + offset));
            if (!groupedDimension ||
                groupedDimension.getPosition() != requiredOldIterator)
              return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          }
          newShape.push_back(newIterationShape[newIterator]);
          newResults.push_back(
              mlir::getAffineDimExpr(newIterator, roots.front().getContext()));
          operandAxis += collapsedOld.size();
          continue;
        }
        if (oldToNew[oldIterator].size() == 1) {
          newShape.push_back(oldOperandType.getDimSize(operandAxis));
          newResults.push_back(
              mlir::getAffineDimExpr(newIterator, roots.front().getContext()));
          ++operandAxis;
          continue;
        }
        for (unsigned splitIterator : oldToNew[oldIterator]) {
          newShape.push_back(newIterationShape[splitIterator]);
          newResults.push_back(mlir::getAffineDimExpr(
              splitIterator, roots.front().getContext()));
        }
        ++operandAxis;
      }
      auto newOperandType =
          mlir::RankedTensorType::get(newShape, oldOperandType.getElementType(),
                                      oldOperandType.getEncoding());
      if (newOperandType != oldOperandType &&
          !canMaterializeStaticReshape(oldOperandType, newOperandType))
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      reparameterized.push_back(ReparameterizedOperand{
          oldOperandType, newOperandType,
          mlir::AffineMap::get(newIterationShape.size(), 0, newResults,
                               roots.front().getContext())});
    }
    if (reparameterized[accessOperand].newType != sourceType ||
        reparameterized.back().oldType != oldResultType)
      return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;

    auto newResultType = reparameterized.back().newType;
    auto newIterationType = mlir::RankedTensorType::get(
        newIterationShape,
        mlir::IntegerType::get(roots.front().getContext(), 1));
    uint32_t newResultTypeId = types.intern(newResultType);
    uint32_t newIterationTypeId = types.intern(newIterationType);
    auto internProjected =
        [&](mlir::AffineMap map, uint32_t sourceTypeId,
            llvm::ArrayRef<int64_t> sourceShape) -> std::optional<uint32_t> {
      IndexRelationResult relation =
          IndexRelation::fromAffineMap(map, newIterationShape, sourceShape);
      if (!relation.isExact() ||
          !relation.get()->hasTotalBoundedAffineMapConstruction())
        return std::nullopt;
      return relations.intern(std::move(*relation.get()), newIterationTypeId,
                              sourceTypeId, map,
                              RelationMaterialization::ProjectedMap,
                              /*total=*/true);
    };
    auto internReshape =
        [&](uint32_t destinationTypeId, mlir::RankedTensorType destinationType,
            uint32_t sourceTypeId, mlir::RankedTensorType sourceTensorType)
        -> std::optional<uint32_t> {
      if (destinationTypeId == sourceTypeId &&
          destinationType == sourceTensorType) {
        IndexRelationResult relation =
            IndexRelation::identity(destinationType.getShape());
        if (!relation.isExact())
          return std::nullopt;
        mlir::AffineMap identity = mlir::AffineMap::getMultiDimIdentityMap(
            destinationType.getRank(), roots.front().getContext());
        return relations.intern(std::move(*relation.get()), destinationTypeId,
                                sourceTypeId, identity,
                                RelationMaterialization::ProjectedMap,
                                /*total=*/true);
      }
      IndexRelationResult relation = IndexRelation::staticReshape(
          destinationType.getShape(), sourceTensorType.getShape());
      if (!relation.isExact() ||
          !canMaterializeStaticReshape(sourceTensorType, destinationType))
        return std::nullopt;
      return relations.intern(std::move(*relation.get()), destinationTypeId,
                              sourceTypeId, std::nullopt,
                              RelationMaterialization::Reshape,
                              /*total=*/true);
    };

    for (unsigned index = 0; index < inputRelationIds.size(); ++index) {
      const RelationRecord *oldRelation =
          relations.lookup(inputRelationIds[index]);
      if (!oldRelation)
        return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      mlir::RankedTensorType oldOperandType = reparameterized[index].oldType;
      mlir::RankedTensorType newOperandType = reparameterized[index].newType;
      uint32_t newOperandTypeId = types.intern(newOperandType);
      std::optional<uint32_t> computeRelation =
          internProjected(reparameterized[index].map, newOperandTypeId,
                          newOperandType.getShape());
      if (!computeRelation)
        return relations.hasWorkLimitReached()
                   ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                   : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      outputComputeRelationIds[index] = *computeRelation;
      const bool selectedInput = index == accessOperand;
      const bool unreadInit = index >= dataInputCount && !recipe->initIsRead;
      if (selectedInput || unreadInit || newOperandType == oldOperandType) {
        outputOperandAccessRelationIds[index] = 0;
        outputOperandAccessTypeIds[index] = 0;
        continue;
      }
      std::optional<uint32_t> operandAccess =
          internReshape(newOperandTypeId, newOperandType,
                        oldRelation->sourceTypeId, oldOperandType);
      if (!operandAccess)
        return relations.hasWorkLimitReached()
                   ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                   : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
      outputOperandAccessRelationIds[index] = *operandAccess;
      outputOperandAccessTypeIds[index] = newOperandTypeId;
    }

    std::optional<uint32_t> outerRelation = internReshape(
        resultTypeId, oldResultType, newResultTypeId, newResultType);
    if (!outerRelation)
      return relations.hasWorkLimitReached()
                 ? WAFER_EGRAPH_CALLBACK_WORK_LIMIT
                 : WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
    *innerResultTypeId = newResultTypeId;
    *outerResultRelationId = *outerRelation;
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
        [](void *context, uint32_t semanticId, uint32_t kind,
           uint32_t resultTypeId, uint32_t accessOperand,
           uint32_t accessRelationId, const uint32_t *inputRelations,
           uint64_t relationCount, uint32_t dataInputCount,
           uint32_t *innerResultTypeId, uint32_t *outerResultRelationId,
           uint32_t *outputComputeRelations,
           uint32_t *outputOperandAccessRelations,
           uint32_t *outputOperandAccessTypeIds) -> uint32_t {
          if ((relationCount && (!inputRelations || !outputComputeRelations ||
                                 !outputOperandAccessRelations ||
                                 !outputOperandAccessTypeIds)) ||
              relationCount > UINT32_MAX || !innerResultTypeId ||
              !outerResultRelationId)
            return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
          auto *component = static_cast<ComponentExpression *>(context);
          if (!component->relations.consumeQuery())
            return WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
          return component->reparameterizeReshapeCompute(
              semanticId, kind, resultTypeId, accessOperand, accessRelationId,
              llvm::ArrayRef(inputRelations, relationCount), dataInputCount,
              innerResultTypeId, outerResultRelationId,
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

  mlir::LogicalResult
  validateExtraction(llvm::ArrayRef<EGraphNode> expression,
                     llvm::ArrayRef<uint32_t> rootNodes) const {
    if (expression.empty() || rootNodes.size() != roots.size() ||
        llvm::any_of(llvm::zip_equal(roots, rootNodes), [&](auto values) {
          auto [root, rootNode] = values;
          return rootNode >= expression.size() ||
                 types.lookup(expression[rootNode].typeId) != root.getType();
        }))
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

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
  materialize(mlir::RewriterBase &rewriter,
              llvm::ArrayRef<EGraphNode> expression,
              llvm::ArrayRef<uint32_t> rootNodes) {
    if (expression.empty() || rootNodes.size() != roots.size() ||
        llvm::any_of(rootNodes,
                     [&](uint32_t root) { return root >= expression.size(); }))
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
    llvm::SmallVector<mlir::Value, 4> results;
    for (uint32_t rootNode : rootNodes)
      results.push_back(values[rootNode]);
    return results;
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
    rewriter.setInsertionPoint(anchor);
    mlir::Location location = anchor->getLoc();
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
    rewriter.setInsertionPoint(anchor);
    mlir::Location location = anchor->getLoc();
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
    const RelationRecord *firstRelation =
        node.relations.empty() ? nullptr
                               : relations.lookup(node.relations.front());
    auto iterationType =
        firstRelation
            ? types.lookupRankedTensor(firstRelation->destinationTypeId)
            : mlir::RankedTensorType{};
    llvm::SmallVector<mlir::utils::IteratorType, 6> iteratorTypes;
    if (!getEffectiveIteratorTypes(*recipe, maps, iterationType, iteratorTypes))
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
    rewriter.setInsertionPoint(anchor);
    if (node.typeId != recipe->resultTypeId) {
      if (outputs.size() != 1)
        return mlir::failure();
      if (!recipe->initIsRead)
        outputs.front() = rewriter
                              .create<mlir::tensor::EmptyOp>(
                                  anchor->getLoc(), resultType.getShape(),
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
        classifyMatmulMaps(maps, iteratorTypes, resultType.getRank());
    if (recipe->kind == EGraphNodeKind::Contraction &&
        recipe->hasCanonicalContractionPayload &&
        variant != MatmulVariant::None) {
      mlir::Operation *created = createMatmulVariant(rewriter, anchor->getLoc(),
                                                     variant, inputs, outputs);
      if (created) {
        created->setDiscardableAttrs(attributes);
        return created->getResult(0);
      }
    }

    auto generic = rewriter.create<mlir::linalg::GenericOp>(
        recipe->operation->getLoc(), mlir::TypeRange{resultType}, inputs,
        outputs, maps, iteratorTypes, /*bodyBuild=*/nullptr, attributes);
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
    target.reshapeThroughComputeApplications +=
        source.reshapeThroughComputeApplications;
    target.abiInputRecords += source.inputRecords;
    target.abiOutputRecords += source.outputRecords;
    target.abiInputBytes += source.inputBytes;
    target.abiOutputBytes += source.outputBytes;
  }

  llvm::SmallVector<mlir::Value, 4> roots;
  llvm::DenseSet<mlir::Operation *> componentOperations;
  mlir::Operation *anchor = nullptr;
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

struct StructuredComponent {
  llvm::SmallVector<mlir::Operation *, 16> operations;
  llvm::SmallVector<mlir::Value, 4> roots;
  mlir::Operation *anchor = nullptr;
};

llvm::SmallVector<StructuredComponent, 8>
collectStructuredComponents(mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::Operation *, 32> operations;
  function.walk([&](mlir::Operation *operation) {
    if (isSupportedProducer(operation))
      operations.push_back(operation);
  });
  llvm::DenseMap<mlir::Operation *, unsigned> indices;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *>
      concatImplementationOwner;
  llvm::SmallVector<unsigned, 32> parents;
  for (auto [index, operation] : llvm::enumerate(operations)) {
    indices.try_emplace(operation, index);
    parents.push_back(index);
    if (std::optional<ConcatDescription> concat =
            getConcatDescription(operation))
      for (mlir::Operation *implementation : concat->implementation)
        concatImplementationOwner.try_emplace(implementation, operation);
  }
  auto find = [&](unsigned value) {
    unsigned root = value;
    while (parents[root] != root)
      root = parents[root];
    while (parents[value] != value) {
      unsigned next = parents[value];
      parents[value] = root;
      value = next;
    }
    return root;
  };
  auto unite = [&](unsigned lhs, unsigned rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (lhs > rhs)
      std::swap(lhs, rhs);
    parents[rhs] = lhs;
  };
  for (auto [index, operation] : llvm::enumerate(operations)) {
    llvm::SmallVector<mlir::Value, 8> semanticOperands;
    if (std::optional<ConcatDescription> concat =
            getConcatDescription(operation))
      semanticOperands.append(concat->inputs.begin(), concat->inputs.end());
    else
      semanticOperands.append(operation->operand_begin(),
                              operation->operand_end());
    for (mlir::Value operand : semanticOperands) {
      mlir::Operation *definition = operand.getDefiningOp();
      auto found = indices.find(definition);
      if (found != indices.end() &&
          definition->getBlock() == operation->getBlock())
        unite(index, found->second);
    }
  }

  llvm::DenseMap<unsigned, unsigned> componentByRoot;
  llvm::SmallVector<StructuredComponent, 8> components;
  for (auto [index, operation] : llvm::enumerate(operations)) {
    unsigned root = find(index);
    auto [entry, inserted] =
        componentByRoot.try_emplace(root, components.size());
    if (inserted)
      components.emplace_back();
    components[entry->second].operations.push_back(operation);
  }

  llvm::SmallVector<StructuredComponent, 8> result;
  for (StructuredComponent &component : components) {
    if (component.operations.empty())
      continue;
    mlir::Block *block = component.operations.front()->getBlock();
    llvm::DenseSet<mlir::Operation *> members;
    bool oneBlock = true;
    for (mlir::Operation *operation : component.operations) {
      oneBlock &= operation->getBlock() == block;
      members.insert(operation);
      if (!component.anchor || component.anchor->isBeforeInBlock(operation))
        component.anchor = operation;
    }
    if (!oneBlock || !component.anchor)
      continue;
    auto isComponentUser =
        [&](mlir::Operation *user,
            const llvm::DenseSet<mlir::Operation *> &componentMembers) {
          if (componentMembers.contains(user))
            return true;
          auto owner = concatImplementationOwner.find(user);
          return owner != concatImplementationOwner.end() &&
                 componentMembers.contains(owner->second);
        };
    for (mlir::Operation *operation : component.operations) {
      mlir::Value value = operation->getResult(0);
      if (value.use_empty() || llvm::any_of(value.getUsers(), [&](auto *user) {
            return !isComponentUser(user, members);
          }))
        component.roots.push_back(value);
    }
    if (component.roots.empty())
      continue;
    bool hasLegalCommitPoint = true;
    for (mlir::Value root : component.roots)
      for (mlir::Operation *user : root.getUsers())
        if (!isComponentUser(user, members) &&
            (user->getBlock() != block ||
             !component.anchor->isBeforeInBlock(user))) {
          hasLegalCommitPoint = false;
          break;
        }
    if (hasLegalCommitPoint) {
      result.push_back(std::move(component));
      continue;
    }

    llvm::DenseSet<mlir::Value> originalRoots(component.roots.begin(),
                                              component.roots.end());
    llvm::SmallVector<mlir::Value, 16> fallbackRoots;
    for (mlir::Operation *operation : component.operations) {
      mlir::Value value = operation->getResult(0);
      llvm::SmallPtrSet<mlir::Operation *, 4> semanticUsers;
      for (mlir::Operation *user : value.getUsers()) {
        auto owner = concatImplementationOwner.find(user);
        semanticUsers.insert(
            owner == concatImplementationOwner.end() ? user : owner->second);
      }
      if (originalRoots.contains(value) || semanticUsers.size() > 1)
        fallbackRoots.push_back(value);
    }

    llvm::DenseSet<mlir::Value> coveredRoots;
    llvm::DenseSet<mlir::Operation *> assignedMembers;
    for (mlir::Value selectedRoot : fallbackRoots) {
      if (coveredRoots.contains(selectedRoot))
        continue;
      mlir::Operation *selectedOwner = selectedRoot.getDefiningOp();
      if (!selectedOwner || !members.contains(selectedOwner) ||
          assignedMembers.contains(selectedOwner))
        continue;

      llvm::DenseSet<mlir::Operation *> ancestors;
      llvm::SmallVector<mlir::Operation *, 16> worklist{selectedOwner};
      while (!worklist.empty()) {
        mlir::Operation *operation = worklist.pop_back_val();
        if (!members.contains(operation) ||
            assignedMembers.contains(operation) ||
            !ancestors.insert(operation).second)
          continue;
        llvm::SmallVector<mlir::Value, 8> semanticOperands;
        if (std::optional<ConcatDescription> concat =
                getConcatDescription(operation))
          semanticOperands.append(concat->inputs.begin(), concat->inputs.end());
        else
          semanticOperands.append(operation->operand_begin(),
                                  operation->operand_end());
        for (mlir::Value operand : semanticOperands)
          if (mlir::Operation *producer = operand.getDefiningOp())
            worklist.push_back(producer);
      }

      llvm::DenseSet<mlir::Operation *> sliceMembers;
      for (mlir::Operation *operation : llvm::reverse(component.operations)) {
        if (!ancestors.contains(operation) ||
            assignedMembers.contains(operation))
          continue;
        const bool sharedWithAnotherSlice =
            operation != selectedOwner &&
            llvm::any_of(operation->getUsers(), [&](mlir::Operation *user) {
              auto owner = concatImplementationOwner.find(user);
              mlir::Operation *semanticUser =
                  owner == concatImplementationOwner.end() ? user
                                                           : owner->second;
              return !sliceMembers.contains(semanticUser);
            });
        if (sharedWithAnotherSlice)
          continue;
        sliceMembers.insert(operation);
      }

      StructuredComponent slice;
      for (mlir::Operation *operation : component.operations)
        if (sliceMembers.contains(operation)) {
          slice.operations.push_back(operation);
          if (!slice.anchor || slice.anchor->isBeforeInBlock(operation))
            slice.anchor = operation;
        }
      if (slice.operations.empty() || !slice.anchor)
        continue;
      for (mlir::Operation *operation : slice.operations) {
        mlir::Value value = operation->getResult(0);
        if (value.use_empty() ||
            llvm::any_of(value.getUsers(), [&](auto *user) {
              return !isComponentUser(user, sliceMembers);
            }))
          slice.roots.push_back(value);
      }
      if (!llvm::is_contained(slice.roots, selectedRoot))
        continue;
      const bool sliceHasLegalCommitPoint =
          llvm::all_of(slice.roots, [&](mlir::Value root) {
            return llvm::all_of(root.getUsers(), [&](mlir::Operation *user) {
              return isComponentUser(user, sliceMembers) ||
                     (user->getBlock() == block &&
                      slice.anchor->isBeforeInBlock(user));
            });
          });
      if (!sliceHasLegalCommitPoint)
        continue;
      for (mlir::Value root : slice.roots)
        coveredRoots.insert(root);
      assignedMembers.insert(sliceMembers.begin(), sliceMembers.end());
      result.push_back(std::move(slice));
    }
  }
  return result;
}

uint64_t countOperations(mlir::Operation *root) {
  uint64_t count = 0;
  root->walk([&](mlir::Operation *) { ++count; });
  return count;
}

inline constexpr llvm::StringLiteral kOutputClosureAttr =
    "wafer.normalization_output_closure";

uint64_t closeShapedFunctionOutputs(mlir::func::FuncOp function,
                                    llvm::StringRef marker) {
  if (!function.getBody().hasOneBlock())
    return 0;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp)
    return 0;
  mlir::OpBuilder builder(returnOp);
  llvm::SmallVector<mlir::Value, 4> outputs(returnOp.getOperands().begin(),
                                            returnOp.getOperands().end());
  uint64_t closureCount = 0;
  for (auto [index, output] : llvm::enumerate(outputs)) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(output.getType());
    mlir::Operation *definition = output.getDefiningOp();
    if (!type || !type.hasStaticShape() ||
        (definition &&
         mlir::isa<mlir::DestinationStyleOpInterface>(definition) &&
         mlir::isa<mlir::TilingInterface>(definition)))
      continue;
    mlir::Value empty = builder.create<mlir::tensor::EmptyOp>(
        output.getLoc(), type.getShape(), type.getElementType(),
        type.getEncoding());
    mlir::AffineMap identity = mlir::AffineMap::getMultiDimIdentityMap(
        type.getRank(), function.getContext());
    llvm::SmallVector<mlir::utils::IteratorType, 6> iterators(
        type.getRank(), mlir::utils::IteratorType::parallel);
    auto closure = builder.create<mlir::linalg::GenericOp>(
        output.getLoc(), mlir::TypeRange{type}, mlir::ValueRange{output},
        mlir::ValueRange{empty},
        llvm::ArrayRef<mlir::AffineMap>{identity, identity}, iterators,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
            mlir::ValueRange arguments) {
          nestedBuilder.create<mlir::linalg::YieldOp>(nestedLocation,
                                                      arguments.front());
        });
    if (!marker.empty())
      closure->setAttr(marker, builder.getUnitAttr());
    outputs[index] = closure.getResult(0);
    ++closureCount;
  }
  if (closureCount != 0)
    returnOp.getOperandsMutable().assign(outputs);
  return closureCount;
}

uint64_t removeShapedFunctionOutputClosures(mlir::func::FuncOp function,
                                            bool &retainedTransformation) {
  retainedTransformation = false;
  if (!function.getBody().hasOneBlock())
    return 0;
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  if (!returnOp)
    return 0;
  llvm::SmallVector<mlir::Value, 4> outputs(returnOp.getOperands().begin(),
                                            returnOp.getOperands().end());
  llvm::SmallVector<mlir::Operation *, 4> dead;
  uint64_t removedClosures = 0;
  for (auto [index, output] : llvm::enumerate(outputs)) {
    auto generic = output.getDefiningOp<mlir::linalg::GenericOp>();
    if (!generic || !generic->hasAttr(kOutputClosureAttr))
      continue;
    generic->removeAttr(kOutputClosureAttr);
    llvm::SmallVector<mlir::AffineMap, 2> maps(generic.getIndexingMapsArray());
    mlir::Block &body = generic.getRegion().front();
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    bool exactIdentity =
        generic.getNumResults() == 1 && generic.getNumDpsInputs() == 1 &&
        generic.getNumDpsInits() == 1 && maps.size() == 2 &&
        maps[0].isIdentity() && maps[1].isIdentity() &&
        llvm::all_of(generic.getIteratorTypesArray(),
                     [](auto iterator) {
                       return iterator == mlir::utils::IteratorType::parallel;
                     }) &&
        body.without_terminator().empty() && yield &&
        yield.getValues().size() == 1 &&
        yield.getValues().front() == body.getArgument(0) &&
        generic.getDpsInits().front().getDefiningOp<mlir::tensor::EmptyOp>();
    if (!exactIdentity) {
      retainedTransformation = true;
      continue;
    }
    outputs[index] = generic.getDpsInputs().front();
    dead.push_back(generic.getOperation());
    ++removedClosures;
  }
  if (removedClosures != 0)
    returnOp.getOperandsMutable().assign(outputs);
  for (mlir::Operation *operation : llvm::reverse(dead)) {
    auto generic = mlir::cast<mlir::linalg::GenericOp>(operation);
    mlir::Value init = generic.getDpsInits().front();
    generic.erase();
    if (mlir::Operation *definition = init.getDefiningOp();
        definition && definition->use_empty())
      definition->erase();
  }
  function.walk([&](mlir::Operation *operation) {
    if (!operation->hasAttr(kOutputClosureAttr))
      return;
    operation->removeAttr(kOutputClosureAttr);
    retainedTransformation = true;
  });
  return removedClosures;
}

mlir::FailureOr<StructuredGraphNormalizationOutcome>
normalizeFunction(mlir::func::FuncOp function,
                  const StructuredGraphNormalizationOptions &options,
                  StructuredGraphNormalizationStatistics &statistics) {
  mlir::IRRewriter rewriter(function.getContext());
  bool changed = false;
  bool exhausted = false;
  llvm::SmallVector<StructuredComponent, 8> components =
      collectStructuredComponents(function);
  for (const StructuredComponent &description : components) {
    ComponentExpression component(description.roots, description.operations,
                                  description.anchor, options);
    mlir::FailureOr<StructuredGraphNormalizationOutcome> outcome =
        component.apply(rewriter, statistics);
    if (mlir::failed(outcome))
      return mlir::failure();
    changed |= *outcome == StructuredGraphNormalizationOutcome::Changed;
    exhausted |=
        *outcome == StructuredGraphNormalizationOutcome::BudgetExhausted;
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
    numMultiRootComponents += statistics.multiRootComponents;
    numConcatTransformsRemoved += statistics.concatTransformsRemoved;
    numRelationQueries += statistics.relationQueries;
    numIdentityApplications += statistics.identityApplications;
    numCompositionApplications += statistics.compositionApplications;
    numComputeAbsorptionApplications +=
        statistics.computeAbsorptionApplications;
    numConcatApplications += statistics.concatApplications;
    numResultReindexApplications += statistics.resultReindexApplications;
    numReshapeThroughComputeApplications +=
        statistics.reshapeThroughComputeApplications;
    numABIInputRecords += statistics.abiInputRecords;
    numABIOutputRecords += statistics.abiOutputRecords;
    numABIInputBytes += statistics.abiInputBytes;
    numABIOutputBytes += statistics.abiOutputBytes;
    auto counter = [&](llvm::StringRef name, uint64_t value) {
      wafer::support::addCompileCounter("structured-egraph", name, value);
    };
    counter("components", statistics.components);
    counter("changed-components", statistics.changedComponents);
    counter("multi-rule-changed-components",
            statistics.multiRuleChangedComponents);
    counter("unchanged-components", statistics.unchangedComponents);
    counter("budget-exhausted-components",
            statistics.budgetExhaustedComponents);
    counter("input-operations", statistics.inputOperations);
    counter("output-operations", statistics.outputOperations);
    counter("access-transforms-removed", statistics.accessTransformsRemoved);
    counter("concat-transforms-removed", statistics.concatTransformsRemoved);
    counter("multi-root-components", statistics.multiRootComponents);
    counter("relation-queries", statistics.relationQueries);
    counter("e-nodes", statistics.eNodes);
    counter("e-classes", statistics.eClasses);
    counter("rewrite-matches", statistics.rewriteMatches);
    counter("e-class-merges", statistics.eClassMerges);
    counter("rebuild-work", statistics.rebuildWork);
    counter("iterations", statistics.iterations);
    counter("extraction-work", statistics.extractionWork);
    counter("identity-applications", statistics.identityApplications);
    counter("composition-applications", statistics.compositionApplications);
    counter("compute-absorption-applications",
            statistics.computeAbsorptionApplications);
    counter("concat-applications", statistics.concatApplications);
    counter("result-reindex-applications",
            statistics.resultReindexApplications);
    counter("reshape-through-compute-applications",
            statistics.reshapeThroughComputeApplications);
    counter("abi-input-records", statistics.abiInputRecords);
    counter("abi-output-records", statistics.abiOutputRecords);
    counter("abi-input-bytes", statistics.abiInputBytes);
    counter("abi-output-bytes", statistics.abiOutputBytes);
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
  const uint64_t addedOutputClosures =
      closeShapedFunctionOutputs(function, kOutputClosureAttr);
  mlir::FailureOr<StructuredGraphNormalizationOutcome> outcome =
      normalizeFunction(function, options, local);
  if (mlir::failed(outcome)) {
    bool retainedTransformation = false;
    removeShapedFunctionOutputClosures(function, retainedTransformation);
    return mlir::failure();
  }
  bool retainedOutputTransformation = false;
  uint64_t strippedOutputClosures = removeShapedFunctionOutputClosures(
      function, retainedOutputTransformation);
  uint64_t consumedOutputClosures =
      addedOutputClosures - strippedOutputClosures;
  if (local.accessTransformsRemoved < consumedOutputClosures)
    return mlir::failure();
  local.accessTransformsRemoved -= consumedOutputClosures;
  if (addedOutputClosures != 0 && retainedOutputTransformation &&
      *outcome == StructuredGraphNormalizationOutcome::Unchanged)
    outcome = StructuredGraphNormalizationOutcome::Changed;
  if (mlir::failed(mlir::verify(function)))
    return mlir::failure();
  local.outputOperations = countOperations(function);
  if (*outcome == StructuredGraphNormalizationOutcome::Changed &&
      !retainedOutputTransformation &&
      local.inputOperations == local.outputOperations &&
      local.accessTransformsRemoved == 0 && local.concatTransformsRemoved == 0)
    outcome = StructuredGraphNormalizationOutcome::Unchanged;
  if (outputStatistics)
    *outputStatistics = local;
  return outcome;
}

mlir::LogicalResult closeStructuredProgramOutputs(mlir::func::FuncOp function) {
  if (!function || mlir::failed(mlir::verify(function)))
    return mlir::failure();
  closeShapedFunctionOutputs(function, /*marker=*/{});
  return mlir::verify(function);
}

} // namespace wafer
