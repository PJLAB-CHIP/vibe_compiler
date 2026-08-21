//===- SelectedAttentionDecomposition.cpp - Winner attention IR -------===//

#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"

#include "Wafer/Planning/PhysicalDataflow/AttentionLinalgOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <type_traits>

namespace wafer::compiler::detail {
namespace {

using namespace attention_linalg;

void setFailure(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
}

mlir::FailureOr<analysis::StaticRectangularIndexSet>
singleBox(const analysis::ExactIndexSet &domain) {
  if (domain.getForm() != analysis::ExactIndexSetForm::BoxUnion ||
      domain.getBoxes().size() != 1)
    return mlir::failure();
  return domain.getBoxes().front();
}

mlir::FailureOr<mlir::Value>
assembleOperand(mlir::RewriterBase &rewriter, mlir::Location loc,
                mlir::Value source,
                llvm::ArrayRef<const AttentionOperandProjection *> pieces) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  if (!sourceType || !sourceType.hasStaticShape() || pieces.empty())
    return mlir::failure();
  llvm::SmallVector<analysis::StaticRectangularIndexSet, 4> boxes;
  for (const AttentionOperandProjection *piece : pieces) {
    mlir::FailureOr<analysis::StaticRectangularIndexSet> box =
        singleBox(piece->exactDomain);
    if (mlir::failed(box) || box->offsets.size() != sourceType.getRank())
      return mlir::failure();
    boxes.push_back(std::move(*box));
  }
  if (boxes.size() == 1) {
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    for (int64_t offset : boxes.front().offsets)
      offsets.push_back(rewriter.getIndexAttr(offset));
    return createSlice(rewriter, loc, source, offsets, boxes.front().sizes);
  }

  llvm::SmallVector<int64_t, 4> lower(sourceType.getRank(),
                                      std::numeric_limits<int64_t>::max());
  llvm::SmallVector<int64_t, 4> upper(sourceType.getRank(), 0);
  for (const auto &box : boxes)
    for (unsigned dim = 0; dim < box.offsets.size(); ++dim) {
      lower[dim] = std::min(lower[dim], box.offsets[dim]);
      upper[dim] = std::max(upper[dim], box.offsets[dim] + box.sizes[dim]);
    }
  llvm::SmallVector<int64_t, 4> shape;
  for (unsigned dim = 0; dim < lower.size(); ++dim) {
    if (lower[dim] < 0 || upper[dim] <= lower[dim])
      return mlir::failure();
    shape.push_back(upper[dim] - lower[dim]);
  }
  mlir::Value assembled =
      createEmpty(rewriter, loc, shape, sourceType.getElementType());
  for (const auto &box : boxes) {
    llvm::SmallVector<mlir::OpFoldResult, 4> sourceOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> destinationOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (unsigned dim = 0; dim < box.offsets.size(); ++dim) {
      sourceOffsets.push_back(rewriter.getIndexAttr(box.offsets[dim]));
      destinationOffsets.push_back(
          rewriter.getIndexAttr(box.offsets[dim] - lower[dim]));
      sizes.push_back(rewriter.getIndexAttr(box.sizes[dim]));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    mlir::Value slice =
        createSlice(rewriter, loc, source, sourceOffsets, box.sizes);
    assembled = rewriter
                    .create<mlir::tensor::InsertSliceOp>(loc, slice, assembled,
                                                         destinationOffsets,
                                                         sizes, strides)
                    .getResult();
  }
  return assembled;
}

mlir::FailureOr<mlir::AffineMap> compressMap(mlir::AffineMap map,
                                             llvm::ArrayRef<unsigned> axes) {
  if (!map)
    return mlir::failure();
  llvm::SmallVector<int64_t, 8> newPosition(map.getNumDims(), -1);
  for (auto [position, axis] : llvm::enumerate(axes)) {
    if (axis >= newPosition.size() || newPosition[axis] >= 0)
      return mlir::failure();
    newPosition[axis] = position;
  }
  llvm::SmallVector<mlir::AffineExpr, 6> results;
  for (mlir::AffineExpr result : map.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(result);
    if (!dimension || dimension.getPosition() >= newPosition.size() ||
        newPosition[dimension.getPosition()] < 0)
      return mlir::failure();
    results.push_back(mlir::getAffineDimExpr(
        newPosition[dimension.getPosition()], map.getContext()));
  }
  return mlir::AffineMap::get(axes.size(), 0, results, map.getContext());
}

mlir::Value convertScalar(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, mlir::FloatType target) {
  auto source = mlir::dyn_cast<mlir::FloatType>(value.getType());
  if (!source || source == target)
    return value;
  return source.getWidth() < target.getWidth()
             ? builder.create<mlir::arith::ExtFOp>(loc, target, value)
                   .getResult()
             : builder.create<mlir::arith::TruncFOp>(loc, target, value)
                   .getResult();
}

mlir::FailureOr<mlir::Value> createContraction(
    mlir::RewriterBase &rewriter, mlir::Location loc, mlir::Value lhs,
    mlir::Value rhs, llvm::ArrayRef<unsigned> originalAxes,
    llvm::ArrayRef<unsigned> originalReductionAxes, mlir::AffineMap lhsMap,
    mlir::AffineMap rhsMap, mlir::AffineMap outputMap,
    llvm::ArrayRef<int64_t> outputShape, mlir::FloatType outputElementType) {
  mlir::FailureOr<mlir::AffineMap> compactLhs =
      compressMap(lhsMap, originalAxes);
  mlir::FailureOr<mlir::AffineMap> compactRhs =
      compressMap(rhsMap, originalAxes);
  mlir::FailureOr<mlir::AffineMap> compactOutput =
      compressMap(outputMap, originalAxes);
  if (mlir::failed(compactLhs) || mlir::failed(compactRhs) ||
      mlir::failed(compactOutput))
    return mlir::failure();
  mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(outputElementType, 0.0));
  mlir::Value init =
      createFill(rewriter, loc, outputShape, outputElementType, zero);
  llvm::SmallVector<mlir::utils::IteratorType, 8> iterators(
      originalAxes.size(), mlir::utils::IteratorType::parallel);
  for (unsigned reduction : originalReductionAxes) {
    auto position = llvm::find(originalAxes, reduction);
    if (position == originalAxes.end())
      return mlir::failure();
    iterators[position - originalAxes.begin()] =
        mlir::utils::IteratorType::reduction;
  }
  auto generic = rewriter.create<mlir::linalg::GenericOp>(
      loc, mlir::TypeRange{init.getType()}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{init},
      llvm::ArrayRef<mlir::AffineMap>{*compactLhs, *compactRhs, *compactOutput},
      iterators,
      [&](mlir::OpBuilder &nested, mlir::Location nestedLoc,
          mlir::ValueRange arguments) {
        mlir::Value left =
            convertScalar(nested, nestedLoc, arguments[0], outputElementType);
        mlir::Value right =
            convertScalar(nested, nestedLoc, arguments[1], outputElementType);
        mlir::Value product =
            nested.create<mlir::arith::MulFOp>(nestedLoc, left, right);
        mlir::Value sum = nested.create<mlir::arith::AddFOp>(nestedLoc, product,
                                                             arguments[2]);
        nested.create<mlir::linalg::YieldOp>(nestedLoc, sum);
      });
  return generic.getResult(0);
}

enum class ReductionKind { Maximum, Sum };

mlir::FailureOr<mlir::Value>
createRowReduction(mlir::RewriterBase &rewriter, mlir::Location loc,
                   mlir::Value input, unsigned reductionRank,
                   llvm::ArrayRef<int64_t> outputShape,
                   mlir::FloatType elementType, ReductionKind kind) {
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() < reductionRank ||
      inputType.getElementType() != elementType)
    return mlir::failure();
  mlir::Value identity = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(elementType,
                                 kind == ReductionKind::Maximum
                                     ? -std::numeric_limits<double>::infinity()
                                     : 0.0));
  mlir::Value init =
      createFill(rewriter, loc, outputShape, elementType, identity);
  unsigned inputRank = inputType.getRank();
  mlir::AffineMap inputMap =
      mlir::AffineMap::getMultiDimIdentityMap(inputRank, rewriter.getContext());
  mlir::AffineMap outputMap = mapForDims(rewriter.getContext(), inputRank,
                                         sequence(inputRank - reductionRank));
  llvm::SmallVector<mlir::utils::IteratorType, 8> iterators(
      inputRank, mlir::utils::IteratorType::parallel);
  for (unsigned index = inputRank - reductionRank; index < inputRank; ++index)
    iterators[index] = mlir::utils::IteratorType::reduction;
  auto generic = rewriter.create<mlir::linalg::GenericOp>(
      loc, mlir::TypeRange{init.getType()}, mlir::ValueRange{input},
      mlir::ValueRange{init},
      llvm::ArrayRef<mlir::AffineMap>{inputMap, outputMap}, iterators,
      [&](mlir::OpBuilder &nested, mlir::Location nestedLoc,
          mlir::ValueRange arguments) {
        mlir::Value combined =
            kind == ReductionKind::Maximum
                ? nested
                      .create<mlir::arith::MaximumFOp>(nestedLoc, arguments[0],
                                                       arguments[1])
                      .getResult()
                : nested
                      .create<mlir::arith::AddFOp>(nestedLoc, arguments[0],
                                                   arguments[1])
                      .getResult();
        nested.create<mlir::linalg::YieldOp>(nestedLoc, combined);
      });
  return generic.getResult(0);
}

const AttentionValueDescription *
findValue(const AttentionWorkDescription &description,
          const AttentionValueId &id) {
  auto found = llvm::find_if(
      description.values,
      [&](const AttentionValueDescription &value) { return value.id == id; });
  return found == description.values.end() ? nullptr : &*found;
}

llvm::SmallVector<int64_t, 6> shapeOf(const AttentionValueDescription &value) {
  if (value.exactDomain.getBoxes().size() != 1)
    return {};
  return llvm::SmallVector<int64_t, 6>(
      value.exactDomain.getBoxes().front().sizes.begin(),
      value.exactDomain.getBoxes().front().sizes.end());
}

} // namespace

PreparedAttentionDecompositionOutcome prepareSelectedAttentionDecomposition(
    const CanonicalAttentionWorkCoordinate &work,
    const FullFeasibilityProof &proof) {
  if (proof.coverage != FullFeasibilityCoverage::EveryPlannedResourceClosed)
    return BrokenPreparedAttention{
        BrokenPreparedAttentionReason::MissingFullProof,
        "attention decomposition requires full resource coverage"};
  std::set<AttentionActionId> expected;
  for (const AttentionWorkDescription &root : work.roots)
    for (const AttentionActionDescription &action : root.actions)
      if (!expected.insert(action.id).second)
        return BrokenPreparedAttention{
            BrokenPreparedAttentionReason::DuplicateIdentity,
            "attention work has duplicate action ID"};
  std::set<AttentionActionId> observed(
      proof.dependencyKey.attentionActions.begin(),
      proof.dependencyKey.attentionActions.end());
  if (expected != observed)
    return BrokenPreparedAttention{
        BrokenPreparedAttentionReason::ActionCoverageMismatch,
        "attention work and feasibility proof action sets differ"};
  for (const AttentionWorkDescription &root : work.roots) {
    std::set<AttentionValueId> values;
    for (const AttentionValueDescription &value : root.values)
      if (!values.insert(value.id).second ||
          value.physicalVersion.has_value() != value.storage.has_value())
        return BrokenPreparedAttention{
            BrokenPreparedAttentionReason::InvalidWorkDescription,
            "attention work has duplicate or partial value binding"};
    for (const AttentionActionDescription &action : root.actions)
      for (const AttentionValueId &value :
           llvm::concat<const AttentionValueId>(action.inputs, action.outputs))
        if (!values.count(value))
          return BrokenPreparedAttention{
              BrokenPreparedAttentionReason::InvalidWorkDescription,
              "attention action references an unknown value"};
  }
  return PreparedAttentionDecomposition{work};
}

mlir::FailureOr<SelectedAttentionRootMaterialization>
emitSelectedAttentionDecomposition(mlir::RewriterBase &rewriter,
                                   LinalgExtAttentionOp attention,
                                   const AttentionWorkDescription &description,
                                   std::string *failureReason) {
  if (!attention || attention.getAlgorithm() != description.algorithm ||
      attention->getNumResults() != 1) {
    setFailure(failureReason,
               "selected attention op does not match prepared description");
    return mlir::failure();
  }
  mlir::FailureOr<AttentionIterationRoles> roles =
      attention.getIterationRoles();
  if (mlir::failed(roles)) {
    setFailure(failureReason, "selected attention iterator roles are invalid");
    return mlir::failure();
  }
  auto queryType =
      mlir::dyn_cast<mlir::RankedTensorType>(attention.getQuery().getType());
  auto keyType =
      mlir::dyn_cast<mlir::RankedTensorType>(attention.getKey().getType());
  auto valueType =
      mlir::dyn_cast<mlir::RankedTensorType>(attention.getValue().getType());
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(attention.getOutput().getType());
  auto scaleType =
      mlir::dyn_cast<mlir::FloatType>(attention.getScale().getType());
  if (!queryType || !keyType || !valueType || !outputType || !scaleType ||
      !queryType.hasStaticShape() || !keyType.hasStaticShape() ||
      !valueType.hasStaticShape() || !outputType.hasStaticShape()) {
    setFailure(failureReason,
               "selected attention requires static ranked float tensors");
    return mlir::failure();
  }

  std::map<AttentionWorkScopeId,
           std::map<AttentionOperandRole,
                    llvm::SmallVector<const AttentionOperandProjection *, 2>>>
      operands;
  for (const AttentionOperandProjection &operand : description.operands)
    operands[operand.scope][operand.role].push_back(&operand);

  SelectedAttentionRootMaterialization result;
  result.root = description.root;
  std::map<AttentionActionId, size_t> actionMapping;
  std::map<AttentionValueId, mlir::Value> values;
  auto recordAction = [&](AttentionActionId id,
                          llvm::ArrayRef<mlir::Operation *> operations) {
    auto [position, inserted] =
        actionMapping.try_emplace(id, result.actions.size());
    if (!inserted)
      return false;
    result.actions.push_back({id, {}});
    llvm::append_range(result.actions.back().operations, operations);
    return true;
  };
  auto recordValue = [&](AttentionValueId id, mlir::Value value) {
    if (!value || !values.try_emplace(id, value).second)
      return false;
    result.values.push_back({id, value});
    return true;
  };

  struct State {
    mlir::Value maximum;
    mlir::Value sum;
    mlir::Value accumulator;
  };
  std::map<ReductionGroupId,
           std::vector<std::pair<AttentionWorkScopeId, State>>>
      contributions;
  mlir::Value assembledOutput = attention.getOutput();
  mlir::Location loc = attention.getLoc();
  rewriter.setInsertionPoint(attention);

  llvm::SmallVector<unsigned, 8> scoreAxes;
  llvm::append_range(scoreAxes, roles->batch);
  llvm::append_range(scoreAxes, roles->query);
  llvm::append_range(scoreAxes, roles->keyValueReduction);
  llvm::SmallVector<unsigned, 8> qkAxes;
  llvm::append_range(qkAxes, roles->batch);
  llvm::append_range(qkAxes, roles->query);
  llvm::append_range(qkAxes, roles->keyValueReduction);
  llvm::append_range(qkAxes, roles->queryKeyReduction);
  llvm::SmallVector<unsigned, 8> pvAxes;
  llvm::append_range(pvAxes, roles->batch);
  llvm::append_range(pvAxes, roles->query);
  llvm::append_range(pvAxes, roles->keyValueReduction);
  llvm::append_range(pvAxes, roles->valueOutput);
  mlir::FailureOr<mlir::AffineMap> scoreMap =
      compressMap(mapForDims(rewriter.getContext(),
                             attention.getIterationDomainRank(), scoreAxes),
                  scoreAxes);
  if (mlir::failed(scoreMap))
    return mlir::failure();

  std::vector<AttentionWorkScopeId> rootScopes;
  std::vector<AttentionWorkScopeId> mergeScopes;
  for (const AttentionActionDescription &action : description.actions) {
    auto &scopes = std::holds_alternative<RequiredRootExecution>(
                       action.id.scope.execution.source)
                       ? rootScopes
                       : mergeScopes;
    if (!llvm::is_contained(scopes, action.id.scope))
      scopes.push_back(action.id.scope);
  }
  llvm::sort(rootScopes);
  llvm::sort(mergeScopes);

  auto action =
      [&](AttentionWorkScopeId scope,
          AttentionActionKind kind) -> const AttentionActionDescription * {
    auto found = llvm::find_if(
        description.actions, [&](const AttentionActionDescription &entry) {
          return entry.id.scope == scope && entry.id.kind == kind;
        });
    return found == description.actions.end() ? nullptr : &*found;
  };

  for (const AttentionWorkScopeId &scope : rootScopes) {
    auto scopeOperands = operands.find(scope);
    if (scopeOperands == operands.end())
      return mlir::failure();
    auto getOperand = [&](AttentionOperandRole role,
                          mlir::Value source) -> mlir::FailureOr<mlir::Value> {
      auto pieces = scopeOperands->second.find(role);
      if (pieces == scopeOperands->second.end())
        return mlir::failure();
      return assembleOperand(rewriter, loc, source, pieces->second);
    };
    mlir::FailureOr<mlir::Value> query =
        getOperand(AttentionOperandRole::Query, attention.getQuery());
    mlir::FailureOr<mlir::Value> key =
        getOperand(AttentionOperandRole::Key, attention.getKey());
    mlir::FailureOr<mlir::Value> value =
        getOperand(AttentionOperandRole::Value, attention.getValue());
    mlir::FailureOr<mlir::Value> mask = mlir::failure();
    if (attention.getMask())
      mask = getOperand(AttentionOperandRole::Mask, attention.getMask());
    if (mlir::failed(query) || mlir::failed(key) || mlir::failed(value) ||
        (attention.getMask() && mlir::failed(mask))) {
      setFailure(failureReason,
                 "selected attention operand pieces cannot be assembled");
      return mlir::failure();
    }

    const AttentionValueDescription *scoreDescription =
        findValue(description, {scope, AttentionValueKind::ScoreBlock});
    const AttentionValueDescription *scaledDescription = findValue(
        description, {scope, AttentionValueKind::ScaledMaskedScoreBlock});
    const AttentionValueDescription *probabilityDescription =
        findValue(description, {scope, AttentionValueKind::ProbabilityBlock});
    const AttentionValueDescription *maximumDescription =
        findValue(description, {scope, AttentionValueKind::BlockMaximum});
    const AttentionValueDescription *sumDescription =
        findValue(description, {scope, AttentionValueKind::BlockSum});
    const AttentionValueDescription *accumulatorDescription =
        findValue(description, {scope, AttentionValueKind::BlockAccumulator});
    if (!scoreDescription || !scaledDescription || !probabilityDescription ||
        !maximumDescription || !sumDescription || !accumulatorDescription)
      return mlir::failure();
    auto scoreType =
        mlir::dyn_cast<mlir::FloatType>(scoreDescription->elementType);
    auto probabilityType =
        mlir::dyn_cast<mlir::FloatType>(probabilityDescription->elementType);
    auto accumulatorType =
        mlir::dyn_cast<mlir::FloatType>(accumulatorDescription->elementType);
    if (!scoreType || !probabilityType || !accumulatorType)
      return mlir::failure();

    mlir::FailureOr<mlir::Value> score = createContraction(
        rewriter, loc, *query, *key, qkAxes, roles->queryKeyReduction,
        attention.getQueryMap(), attention.getKeyMap(),
        mapForDims(rewriter.getContext(), attention.getIterationDomainRank(),
                   scoreAxes),
        shapeOf(*scoreDescription), scoreType);
    if (mlir::failed(score))
      return mlir::failure();
    if (!recordValue({scope, AttentionValueKind::ScoreBlock}, *score) ||
        !recordAction({scope, AttentionActionKind::QueryKeyContraction},
                      {score->getDefiningOp()}))
      return mlir::failure();

    mlir::Value scaledInit =
        createEmpty(rewriter, loc, shapeOf(*scaledDescription), scoreType);
    llvm::SmallVector<mlir::Value, 2> scaleInputs{*score};
    llvm::SmallVector<mlir::AffineMap, 3> scaleMaps{
        mlir::AffineMap::getMultiDimIdentityMap(
            shapeOf(*scaledDescription).size(), rewriter.getContext())};
    if (attention.getMask()) {
      scaleInputs.push_back(*mask);
      mlir::FailureOr<mlir::AffineMap> compactMask =
          compressMap(attention.getMaskMap().value(), scoreAxes);
      if (mlir::failed(compactMask))
        return mlir::failure();
      scaleMaps.push_back(*compactMask);
    }
    scaleMaps.push_back(mlir::AffineMap::getMultiDimIdentityMap(
        shapeOf(*scaledDescription).size(), rewriter.getContext()));
    llvm::SmallVector<mlir::utils::IteratorType, 6> scaleIterators(
        shapeOf(*scaledDescription).size(),
        mlir::utils::IteratorType::parallel);
    auto scaledOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{scaledInit.getType()}, scaleInputs,
        mlir::ValueRange{scaledInit}, scaleMaps, scaleIterators,
        [&](mlir::OpBuilder &nested, mlir::Location nestedLoc,
            mlir::ValueRange arguments) {
          mlir::Value scale =
              convertScalar(nested, nestedLoc, attention.getScale(), scoreType);
          mlir::Value result = nested.create<mlir::arith::MulFOp>(
              nestedLoc, arguments[0], scale);
          if (attention.getMask()) {
            mlir::Value maskValue =
                convertScalar(nested, nestedLoc, arguments[1], scoreType);
            result = nested.create<mlir::arith::AddFOp>(nestedLoc, result,
                                                        maskValue);
          }
          nested.create<mlir::linalg::YieldOp>(nestedLoc, result);
        });
    mlir::Value scaled = scaledOp.getResult(0);
    if (!recordValue({scope, AttentionValueKind::ScaledMaskedScoreBlock},
                     scaled) ||
        !recordAction({scope, AttentionActionKind::ScaleMask},
                      {scaledOp.getOperation()}))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> maximum = createRowReduction(
        rewriter, loc, scaled, roles->keyValueReduction.size(),
        shapeOf(*maximumDescription), scoreType, ReductionKind::Maximum);
    if (mlir::failed(maximum) ||
        !recordValue({scope, AttentionValueKind::BlockMaximum}, *maximum) ||
        !recordAction({scope, AttentionActionKind::RowMaximum},
                      {maximum->getDefiningOp()}))
      return mlir::failure();

    mlir::AffineMap rowMap =
        mapForDims(rewriter.getContext(), shapeOf(*scaledDescription).size(),
                   sequence(shapeOf(*maximumDescription).size()));
    mlir::AffineMap scoreIdentity = mlir::AffineMap::getMultiDimIdentityMap(
        shapeOf(*scaledDescription).size(), rewriter.getContext());
    mlir::Value broadcastMaximum = createPointwise(
        rewriter, loc, {*maximum}, {rowMap}, shapeOf(*scaledDescription),
        scoreType, PointwiseKind::Identity);
    mlir::Value shifted = createPointwise(
        rewriter, loc, {scaled, broadcastMaximum},
        {scoreIdentity, scoreIdentity}, shapeOf(*scaledDescription), scoreType,
        PointwiseKind::Subtract);
    mlir::Value exponent = createPointwise(
        rewriter, loc, {shifted}, {scoreIdentity},
        shapeOf(*probabilityDescription), scoreType, PointwiseKind::Exp);
    mlir::FailureOr<mlir::Value> probability =
        createFloatConvert(rewriter, loc, exponent, probabilityType);
    if (mlir::failed(probability) ||
        !recordValue({scope, AttentionValueKind::ProbabilityBlock},
                     *probability) ||
        !recordAction({scope, AttentionActionKind::Exponential},
                      {exponent.getDefiningOp(), probability->getDefiningOp()}))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> sum = createRowReduction(
        rewriter, loc, exponent, roles->keyValueReduction.size(),
        shapeOf(*sumDescription), scoreType, ReductionKind::Sum);
    if (mlir::failed(sum) ||
        !recordValue({scope, AttentionValueKind::BlockSum}, *sum) ||
        !recordAction({scope, AttentionActionKind::RowSum},
                      {sum->getDefiningOp()}))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> accumulator = createContraction(
        rewriter, loc, *probability, *value, pvAxes, roles->keyValueReduction,
        mapForDims(rewriter.getContext(), attention.getIterationDomainRank(),
                   scoreAxes),
        attention.getValueMap(), attention.getOutputMap(),
        shapeOf(*accumulatorDescription), accumulatorType);
    if (mlir::failed(accumulator) ||
        !recordValue({scope, AttentionValueKind::BlockAccumulator},
                     *accumulator) ||
        !recordAction({scope, AttentionActionKind::ValueContraction},
                      {accumulator->getDefiningOp()}))
      return mlir::failure();

    State state{*maximum, *sum, *accumulator};
    for (auto [kind, value] :
         {std::pair{AttentionValueKind::RunningMaximum, state.maximum},
          std::pair{AttentionValueKind::RunningSum, state.sum},
          std::pair{AttentionValueKind::RunningAccumulator, state.accumulator}})
      if (!recordValue({scope, kind}, value))
        return mlir::failure();
    if (!recordAction({scope, AttentionActionKind::StateUpdate},
                      {state.maximum.getDefiningOp(), state.sum.getDefiningOp(),
                       state.accumulator.getDefiningOp()}))
      return mlir::failure();

    if (scope.group) {
      contributions[*scope.group].push_back({scope, state});
      continue;
    }

    mlir::FailureOr<mlir::Value> accumulatorForFinalize =
        createFloatConvert(rewriter, loc, state.accumulator,
                           mlir::cast<mlir::FloatType>(
                               mlir::cast<mlir::ShapedType>(state.sum.getType())
                                   .getElementType()));
    if (mlir::failed(accumulatorForFinalize))
      return mlir::failure();
    mlir::AffineMap outputIdentity = mlir::AffineMap::getMultiDimIdentityMap(
        outputType.getRank(), rewriter.getContext());
    mlir::AffineMap outputRowMap =
        mapForDims(rewriter.getContext(), outputType.getRank(),
                   sequence(outputType.getRank() - 1));
    mlir::Value normalized = createPointwise(
        rewriter, loc, {*accumulatorForFinalize, state.sum},
        {outputIdentity, outputRowMap},
        mlir::cast<mlir::RankedTensorType>(accumulatorForFinalize->getType())
            .getShape(),
        mlir::cast<mlir::ShapedType>(accumulatorForFinalize->getType())
            .getElementType(),
        PointwiseKind::Divide);
    mlir::FailureOr<mlir::Value> stored = createFloatConvert(
        rewriter, loc, normalized,
        mlir::cast<mlir::FloatType>(outputType.getElementType()));
    if (mlir::failed(stored) ||
        !recordValue({scope, AttentionValueKind::FinalOutput}, *stored) ||
        !recordAction({scope, AttentionActionKind::Finalize},
                      {normalized.getDefiningOp(), stored->getDefiningOp()}))
      return mlir::failure();
    mlir::FailureOr<analysis::StaticRectangularIndexSet> outputBox = singleBox(
        findValue(description, {scope, AttentionValueKind::FinalOutput})
            ->exactDomain);
    if (mlir::failed(outputBox))
      return mlir::failure();
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (auto [offset, size] :
         llvm::zip_equal(outputBox->offsets, outputBox->sizes)) {
      offsets.push_back(rewriter.getIndexAttr(offset));
      sizes.push_back(rewriter.getIndexAttr(size));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    assembledOutput =
        rewriter
            .create<mlir::tensor::InsertSliceOp>(loc, *stored, assembledOutput,
                                                 offsets, sizes, strides)
            .getResult();
  }

  for (const AttentionWorkScopeId &scope : mergeScopes) {
    if (!scope.group || !contributions.count(*scope.group) ||
        contributions[*scope.group].empty())
      return mlir::failure();
    State merged = contributions[*scope.group].front().second;
    llvm::SmallVector<mlir::Operation *, 3> mergeOps;
    for (const auto &[inputScope, current] :
         llvm::drop_begin(contributions[*scope.group])) {
      auto maximumType = mlir::cast<mlir::FloatType>(
          mlir::cast<mlir::ShapedType>(merged.maximum.getType())
              .getElementType());
      llvm::SmallVector<int64_t, 6> rowShape(
          mlir::cast<mlir::ShapedType>(merged.maximum.getType()).getShape());
      llvm::SmallVector<int64_t, 6> outputShape(
          mlir::cast<mlir::ShapedType>(merged.accumulator.getType())
              .getShape());
      mlir::AffineMap rowIdentity = mlir::AffineMap::getMultiDimIdentityMap(
          rowShape.size(), rewriter.getContext());
      mlir::AffineMap outputIdentity = mlir::AffineMap::getMultiDimIdentityMap(
          outputShape.size(), rewriter.getContext());
      mlir::AffineMap outputRowMap = mapForDims(
          rewriter.getContext(), outputShape.size(), sequence(rowShape.size()));
      mlir::Value maximum =
          createPointwise(rewriter, loc, {merged.maximum, current.maximum},
                          {rowIdentity, rowIdentity}, rowShape, maximumType,
                          PointwiseKind::Maximum);
      mlir::Value leftDelta = createPointwise(
          rewriter, loc, {merged.maximum, maximum}, {rowIdentity, rowIdentity},
          rowShape, maximumType, PointwiseKind::Subtract);
      mlir::Value rightDelta = createPointwise(
          rewriter, loc, {current.maximum, maximum}, {rowIdentity, rowIdentity},
          rowShape, maximumType, PointwiseKind::Subtract);
      mlir::Value leftScale =
          createPointwise(rewriter, loc, {leftDelta}, {rowIdentity}, rowShape,
                          maximumType, PointwiseKind::Exp);
      mlir::Value rightScale =
          createPointwise(rewriter, loc, {rightDelta}, {rowIdentity}, rowShape,
                          maximumType, PointwiseKind::Exp);
      mlir::Value sum = createPointwise(
          rewriter, loc, {leftScale, merged.sum, rightScale, current.sum},
          {rowIdentity, rowIdentity, rowIdentity, rowIdentity}, rowShape,
          maximumType, PointwiseKind::WeightedAdd);
      mlir::FailureOr<mlir::Value> leftAccumulator =
          createFloatConvert(rewriter, loc, merged.accumulator, maximumType);
      mlir::FailureOr<mlir::Value> rightAccumulator =
          createFloatConvert(rewriter, loc, current.accumulator, maximumType);
      if (mlir::failed(leftAccumulator) || mlir::failed(rightAccumulator))
        return mlir::failure();
      mlir::Value accumulator = createPointwise(
          rewriter, loc,
          {leftScale, *leftAccumulator, rightScale, *rightAccumulator},
          {outputRowMap, outputIdentity, outputRowMap, outputIdentity},
          outputShape, maximumType, PointwiseKind::WeightedAdd);
      auto storedType = mlir::cast<mlir::FloatType>(
          mlir::cast<mlir::ShapedType>(merged.accumulator.getType())
              .getElementType());
      mlir::FailureOr<mlir::Value> storedAccumulator =
          createFloatConvert(rewriter, loc, accumulator, storedType);
      if (mlir::failed(storedAccumulator))
        return mlir::failure();
      merged = {maximum, sum, *storedAccumulator};
      mergeOps = {maximum.getDefiningOp(), sum.getDefiningOp(),
                  storedAccumulator->getDefiningOp()};
    }
    for (auto [kind, value] :
         {std::pair{AttentionValueKind::RunningMaximum, merged.maximum},
          std::pair{AttentionValueKind::RunningSum, merged.sum},
          std::pair{AttentionValueKind::RunningAccumulator,
                    merged.accumulator}})
      if (!recordValue({scope, kind}, value))
        return mlir::failure();
    if (mergeOps.empty())
      mergeOps = {merged.maximum.getDefiningOp(), merged.sum.getDefiningOp(),
                  merged.accumulator.getDefiningOp()};
    if (!recordAction({scope, AttentionActionKind::StateMerge}, mergeOps))
      return mlir::failure();

    auto sumType = mlir::cast<mlir::FloatType>(
        mlir::cast<mlir::ShapedType>(merged.sum.getType()).getElementType());
    mlir::FailureOr<mlir::Value> accumulator =
        createFloatConvert(rewriter, loc, merged.accumulator, sumType);
    if (mlir::failed(accumulator))
      return mlir::failure();
    llvm::SmallVector<int64_t, 6> outputShape(
        mlir::cast<mlir::ShapedType>(accumulator->getType()).getShape());
    llvm::SmallVector<int64_t, 6> rowShape(
        mlir::cast<mlir::ShapedType>(merged.sum.getType()).getShape());
    mlir::AffineMap outputIdentity = mlir::AffineMap::getMultiDimIdentityMap(
        outputShape.size(), rewriter.getContext());
    mlir::AffineMap outputRowMap = mapForDims(
        rewriter.getContext(), outputShape.size(), sequence(rowShape.size()));
    mlir::Value normalized =
        createPointwise(rewriter, loc, {*accumulator, merged.sum},
                        {outputIdentity, outputRowMap}, outputShape, sumType,
                        PointwiseKind::Divide);
    mlir::FailureOr<mlir::Value> stored = createFloatConvert(
        rewriter, loc, normalized,
        mlir::cast<mlir::FloatType>(outputType.getElementType()));
    if (mlir::failed(stored) ||
        !recordValue({scope, AttentionValueKind::FinalOutput}, *stored) ||
        !recordAction({scope, AttentionActionKind::Finalize},
                      {normalized.getDefiningOp(), stored->getDefiningOp()}))
      return mlir::failure();
    const AttentionValueDescription *outputDescription =
        findValue(description, {scope, AttentionValueKind::FinalOutput});
    if (!outputDescription)
      return mlir::failure();
    mlir::FailureOr<analysis::StaticRectangularIndexSet> outputBox =
        singleBox(outputDescription->exactDomain);
    if (mlir::failed(outputBox))
      return mlir::failure();
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (auto [offset, size] :
         llvm::zip_equal(outputBox->offsets, outputBox->sizes)) {
      offsets.push_back(rewriter.getIndexAttr(offset));
      sizes.push_back(rewriter.getIndexAttr(size));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    assembledOutput =
        rewriter
            .create<mlir::tensor::InsertSliceOp>(loc, *stored, assembledOutput,
                                                 offsets, sizes, strides)
            .getResult();
  }

  rewriter.replaceOp(attention, assembledOutput);
  result.result = assembledOutput;
  llvm::sort(result.actions, [](const AttentionActionMaterialization &lhs,
                                const AttentionActionMaterialization &rhs) {
    return lhs.id < rhs.id;
  });
  llvm::sort(result.values, [](const AttentionValueMaterialization &lhs,
                               const AttentionValueMaterialization &rhs) {
    return lhs.id < rhs.id;
  });
  return result;
}

} // namespace wafer::compiler::detail
