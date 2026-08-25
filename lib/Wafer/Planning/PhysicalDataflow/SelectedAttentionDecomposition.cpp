//===- SelectedAttentionDecomposition.cpp - Winner attention IR -------===//

#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"

#include "Wafer/Planning/PhysicalDataflow/AttentionLinalgOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <tuple>
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

struct ContiguousOperandDomain {
  llvm::SmallVector<analysis::StaticRectangularIndexSet, 4> pieces;
  analysis::StaticRectangularIndexSet bounds;
};

mlir::FailureOr<ContiguousOperandDomain>
normalizeOperandDomain(mlir::RankedTensorType sourceType,
                       const analysis::ExactIndexSet &domain) {
  if (!sourceType || !sourceType.hasStaticShape() ||
      domain.getForm() != analysis::ExactIndexSetForm::BoxUnion ||
      domain.getBoxes().empty())
    return mlir::failure();

  ContiguousOperandDomain result;
  for (const analysis::StaticRectangularIndexSet &box : domain.getBoxes()) {
    if (box.offsets.size() != sourceType.getRank() ||
        box.sizes.size() != sourceType.getRank())
      return mlir::failure();
    for (unsigned dim = 0; dim < box.offsets.size(); ++dim) {
      int64_t end = 0;
      if (box.offsets[dim] < 0 || box.sizes[dim] <= 0 ||
          llvm::AddOverflow(box.offsets[dim], box.sizes[dim], end) ||
          end > sourceType.getDimSize(dim))
        return mlir::failure();
    }
    result.pieces.push_back(box);
  }
  llvm::sort(result.pieces, [](const auto &lhs, const auto &rhs) {
    if (lhs.offsets != rhs.offsets)
      return std::lexicographical_compare(
          lhs.offsets.begin(), lhs.offsets.end(), rhs.offsets.begin(),
          rhs.offsets.end());
    return std::lexicographical_compare(lhs.sizes.begin(), lhs.sizes.end(),
                                        rhs.sizes.begin(), rhs.sizes.end());
  });
  result.pieces.erase(std::unique(result.pieces.begin(), result.pieces.end(),
                                  [](const auto &lhs, const auto &rhs) {
                                    return lhs.offsets == rhs.offsets &&
                                           lhs.sizes == rhs.sizes;
                                  }),
                      result.pieces.end());

  llvm::SmallVector<int64_t, 4> lower(sourceType.getRank(),
                                      std::numeric_limits<int64_t>::max());
  llvm::SmallVector<int64_t, 4> upper(sourceType.getRank(), 0);
  int64_t coveredElements = 0;
  for (const auto &box : result.pieces) {
    int64_t boxElements = 1;
    for (unsigned dim = 0; dim < box.offsets.size(); ++dim) {
      lower[dim] = std::min(lower[dim], box.offsets[dim]);
      upper[dim] = std::max(upper[dim], box.offsets[dim] + box.sizes[dim]);
      if (llvm::MulOverflow(boxElements, box.sizes[dim], boxElements))
        return mlir::failure();
    }
    if (llvm::AddOverflow(coveredElements, boxElements, coveredElements))
      return mlir::failure();
  }
  for (size_t lhs = 0; lhs < result.pieces.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < result.pieces.size(); ++rhs) {
      bool overlaps = true;
      for (unsigned dim = 0; dim < result.pieces[lhs].offsets.size(); ++dim) {
        const int64_t lhsEnd =
            result.pieces[lhs].offsets[dim] + result.pieces[lhs].sizes[dim];
        const int64_t rhsEnd =
            result.pieces[rhs].offsets[dim] + result.pieces[rhs].sizes[dim];
        if (lhsEnd <= result.pieces[rhs].offsets[dim] ||
            rhsEnd <= result.pieces[lhs].offsets[dim]) {
          overlaps = false;
          break;
        }
      }
      if (overlaps)
        return mlir::failure();
    }

  int64_t assembledElements = 1;
  result.bounds.offsets = lower;
  for (unsigned dim = 0; dim < lower.size(); ++dim) {
    if (lower[dim] < 0 || upper[dim] <= lower[dim])
      return mlir::failure();
    const int64_t extent = upper[dim] - lower[dim];
    if (llvm::MulOverflow(assembledElements, extent, assembledElements))
      return mlir::failure();
    result.bounds.sizes.push_back(extent);
  }
  if (coveredElements != assembledElements)
    return mlir::failure();
  return result;
}

mlir::FailureOr<mlir::Value>
assembleOperand(mlir::RewriterBase &rewriter, mlir::Location loc,
                mlir::Value source,
                const analysis::ExactIndexSet &exactDomain) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  mlir::FailureOr<ContiguousOperandDomain> normalized =
      normalizeOperandDomain(sourceType, exactDomain);
  if (mlir::failed(normalized))
    return mlir::failure();
  if (normalized->pieces.size() == 1) {
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    for (int64_t offset : normalized->pieces.front().offsets)
      offsets.push_back(rewriter.getIndexAttr(offset));
    return createSlice(rewriter, loc, source, offsets,
                       normalized->pieces.front().sizes);
  }

  mlir::Value assembled = createEmpty(rewriter, loc, normalized->bounds.sizes,
                                      sourceType.getElementType());
  for (const auto &box : normalized->pieces) {
    llvm::SmallVector<mlir::OpFoldResult, 4> sourceOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> destinationOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    for (unsigned dim = 0; dim < box.offsets.size(); ++dim) {
      sourceOffsets.push_back(rewriter.getIndexAttr(box.offsets[dim]));
      destinationOffsets.push_back(rewriter.getIndexAttr(
          box.offsets[dim] - normalized->bounds.offsets[dim]));
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

mlir::FailureOr<mlir::OpFoldResult>
addOperandBlockOffset(mlir::OpBuilder &builder, mlir::Location loc,
                      int64_t base, mlir::OpFoldResult relative) {
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(relative)) {
    int64_t combined = 0;
    if (llvm::AddOverflow(base, *constant, combined))
      return mlir::failure();
    return mlir::OpFoldResult(builder.getIndexAttr(combined));
  }
  if (base == 0)
    return relative;
  mlir::Value relativeValue =
      mlir::getValueOrCreateConstantIndexOp(builder, loc, relative);
  mlir::Value baseValue =
      builder.create<mlir::arith::ConstantIndexOp>(loc, base);
  return mlir::OpFoldResult(
      builder.create<mlir::arith::AddIOp>(loc, baseValue, relativeValue)
          .getResult());
}

mlir::FailureOr<mlir::Value>
assembleTemporalOperandBlock(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source,
                             const AttentionOperandDescription &description,
                             llvm::ArrayRef<unsigned> keyValueAxes,
                             llvm::ArrayRef<mlir::OpFoldResult> keyValueOffsets,
                             llvm::ArrayRef<int64_t> keyValueSizes) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  mlir::FailureOr<ContiguousOperandDomain> exact =
      normalizeOperandDomain(sourceType, description.exactDomain);
  mlir::FailureOr<ContiguousOperandDomain> resident =
      normalizeOperandDomain(sourceType, description.residentDomain);
  if (mlir::failed(exact) || mlir::failed(resident) ||
      keyValueOffsets.size() != keyValueAxes.size() ||
      keyValueSizes.size() != keyValueAxes.size() ||
      description.indexingMap.getNumResults() !=
          static_cast<unsigned>(sourceType.getRank()) ||
      exact->bounds.offsets.size() != resident->bounds.offsets.size())
    return mlir::failure();

  llvm::SmallVector<mlir::OpFoldResult, 6> offsets;
  llvm::SmallVector<int64_t, 6> sizes;
  for (auto [dimension, expression] :
       llvm::enumerate(description.indexingMap.getResults())) {
    auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!iterator ||
        resident->bounds.offsets[dimension] != exact->bounds.offsets[dimension])
      return mlir::failure();
    auto keyValueAxis = llvm::find(keyValueAxes, iterator.getPosition());
    if (keyValueAxis == keyValueAxes.end()) {
      if (resident->bounds.sizes[dimension] != exact->bounds.sizes[dimension])
        return mlir::failure();
      offsets.push_back(builder.getIndexAttr(exact->bounds.offsets[dimension]));
      sizes.push_back(exact->bounds.sizes[dimension]);
      continue;
    }
    const unsigned index = keyValueAxis - keyValueAxes.begin();
    if (keyValueSizes[index] <= 0 ||
        keyValueSizes[index] > resident->bounds.sizes[dimension])
      return mlir::failure();
    mlir::FailureOr<mlir::OpFoldResult> offset = addOperandBlockOffset(
        builder, loc, exact->bounds.offsets[dimension], keyValueOffsets[index]);
    if (mlir::failed(offset))
      return mlir::failure();
    offsets.push_back(*offset);
    sizes.push_back(keyValueSizes[index]);
  }
  return createSlice(builder, loc, source, offsets, sizes);
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
    mlir::OpBuilder &rewriter, mlir::Location loc, mlir::Value lhs,
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
  enum class ContractionDimension : uint8_t { Batch, M, N, K };
  llvm::SmallVector<bool, 8> reductions(originalAxes.size(), false);
  for (unsigned reduction : originalReductionAxes) {
    auto position = llvm::find(originalAxes, reduction);
    if (position == originalAxes.end())
      return mlir::failure();
    reductions[position - originalAxes.begin()] = true;
  }
  auto containsDimension = [](mlir::AffineMap map, unsigned dimension) {
    return llvm::any_of(map.getResults(), [&](mlir::AffineExpr expression) {
      auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      return dim && dim.getPosition() == dimension;
    });
  };
  llvm::SmallVector<ContractionDimension, 8> dimensions;
  for (unsigned dimension = 0; dimension < originalAxes.size(); ++dimension) {
    const bool inLhs = containsDimension(*compactLhs, dimension);
    const bool inRhs = containsDimension(*compactRhs, dimension);
    const bool inOutput = containsDimension(*compactOutput, dimension);
    if (inLhs && inRhs && inOutput && !reductions[dimension])
      dimensions.push_back(ContractionDimension::Batch);
    else if (inLhs && !inRhs && inOutput && !reductions[dimension])
      dimensions.push_back(ContractionDimension::M);
    else if (!inLhs && inRhs && inOutput && !reductions[dimension])
      dimensions.push_back(ContractionDimension::N);
    else if (inLhs && inRhs && !inOutput && reductions[dimension])
      dimensions.push_back(ContractionDimension::K);
    else
      return mlir::failure();
  }
  auto mapDimensions = [&](mlir::AffineMap map)
      -> mlir::FailureOr<llvm::SmallVector<ContractionDimension, 8>> {
    llvm::SmallVector<ContractionDimension, 8> result;
    for (mlir::AffineExpr expression : map.getResults()) {
      auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!dim || dim.getPosition() >= dimensions.size())
        return mlir::failure();
      result.push_back(dimensions[dim.getPosition()]);
    }
    return result;
  };
  mlir::FailureOr<llvm::SmallVector<ContractionDimension, 8>> lhsDimensions =
      mapDimensions(*compactLhs);
  mlir::FailureOr<llvm::SmallVector<ContractionDimension, 8>> rhsDimensions =
      mapDimensions(*compactRhs);
  mlir::FailureOr<llvm::SmallVector<ContractionDimension, 8>> outputDimensions =
      mapDimensions(*compactOutput);
  if (mlir::failed(lhsDimensions) || mlir::failed(rhsDimensions) ||
      mlir::failed(outputDimensions))
    return mlir::failure();
  auto groupedShape = [](mlir::RankedTensorType type,
                         llvm::ArrayRef<ContractionDimension> actual,
                         llvm::ArrayRef<ContractionDimension> order)
      -> std::optional<llvm::SmallVector<int64_t, 4>> {
    if (!type || !type.hasStaticShape() ||
        type.getRank() != static_cast<int64_t>(actual.size()))
      return std::nullopt;
    llvm::SmallVector<int64_t, 4> grouped;
    size_t position = 0;
    for (ContractionDimension expected : order) {
      int64_t extent = 1;
      bool observed = false;
      while (position < actual.size() && actual[position] == expected) {
        int64_t next = 0;
        if (type.getDimSize(position) <= 0 ||
            llvm::MulOverflow(extent, type.getDimSize(position), next))
          return std::nullopt;
        extent = next;
        observed = true;
        ++position;
      }
      if (!observed)
        return std::nullopt;
      grouped.push_back(extent);
    }
    if (position != actual.size())
      return std::nullopt;
    return grouped;
  };
  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  auto resultType = mlir::RankedTensorType::get(outputShape, outputElementType);
  const bool hasBatch =
      llvm::is_contained(dimensions, ContractionDimension::Batch);
  llvm::SmallVector<ContractionDimension, 3> lhsOrder;
  llvm::SmallVector<ContractionDimension, 3> rhsOrder;
  llvm::SmallVector<ContractionDimension, 3> outputOrder;
  if (hasBatch) {
    lhsOrder.push_back(ContractionDimension::Batch);
    rhsOrder.push_back(ContractionDimension::Batch);
    outputOrder.push_back(ContractionDimension::Batch);
  }
  lhsOrder.append({ContractionDimension::M, ContractionDimension::K});
  outputOrder.append({ContractionDimension::M, ContractionDimension::N});
  bool transposeRhs = false;
  llvm::SmallVector<ContractionDimension, 3> normalRhsOrder = rhsOrder;
  normalRhsOrder.append({ContractionDimension::K, ContractionDimension::N});
  llvm::SmallVector<ContractionDimension, 3> transposedRhsOrder = rhsOrder;
  transposedRhsOrder.append({ContractionDimension::N, ContractionDimension::K});
  std::optional<llvm::SmallVector<int64_t, 4>> lhsShape =
      groupedShape(lhsType, *lhsDimensions, lhsOrder);
  std::optional<llvm::SmallVector<int64_t, 4>> rhsShape =
      groupedShape(rhsType, *rhsDimensions, normalRhsOrder);
  if (!rhsShape) {
    rhsShape = groupedShape(rhsType, *rhsDimensions, transposedRhsOrder);
    transposeRhs = true;
  }
  std::optional<llvm::SmallVector<int64_t, 4>> resultShape =
      groupedShape(resultType, *outputDimensions, outputOrder);
  if (!lhsShape || !rhsShape || !resultShape ||
      lhsShape->size() != resultShape->size() ||
      rhsShape->size() != resultShape->size() ||
      (*lhsShape)[lhsShape->size() - 2] !=
          (*resultShape)[resultShape->size() - 2] ||
      (*rhsShape)[rhsShape->size() - (transposeRhs ? 2 : 1)] !=
          (*resultShape)[resultShape->size() - 1] ||
      lhsShape->back() !=
          (*rhsShape)[rhsShape->size() - (transposeRhs ? 1 : 2)])
    return mlir::failure();

  auto reshape =
      [&](mlir::Value value,
          llvm::ArrayRef<int64_t> shape) -> mlir::FailureOr<mlir::Value> {
    auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
    return createExactStaticReshape(
        rewriter, loc, value,
        mlir::RankedTensorType::get(shape, type.getElementType()));
  };
  mlir::FailureOr<mlir::Value> collapsedLhs = reshape(lhs, *lhsShape);
  mlir::FailureOr<mlir::Value> collapsedRhs = reshape(rhs, *rhsShape);
  if (mlir::failed(collapsedLhs) || mlir::failed(collapsedRhs))
    return mlir::failure();
  mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
      loc, rewriter.getFloatAttr(outputElementType, 0.0));
  mlir::Value init =
      createFill(rewriter, loc, *resultShape, outputElementType, zero);
  mlir::Value contracted;
  if (hasBatch) {
    contracted =
        transposeRhs
            ? rewriter
                  .create<mlir::linalg::BatchMatmulTransposeBOp>(
                      loc, mlir::ValueRange{*collapsedLhs, *collapsedRhs},
                      mlir::ValueRange{init})
                  .getResult(0)
            : rewriter
                  .create<mlir::linalg::BatchMatmulOp>(
                      loc, mlir::ValueRange{*collapsedLhs, *collapsedRhs},
                      mlir::ValueRange{init})
                  .getResult(0);
  } else {
    contracted =
        transposeRhs
            ? rewriter
                  .create<mlir::linalg::MatmulTransposeBOp>(
                      loc, mlir::ValueRange{*collapsedLhs, *collapsedRhs},
                      mlir::ValueRange{init})
                  .getResult(0)
            : rewriter
                  .create<mlir::linalg::MatmulOp>(
                      loc, mlir::ValueRange{*collapsedLhs, *collapsedRhs},
                      mlir::ValueRange{init})
                  .getResult(0);
  }
  return createExactStaticReshape(rewriter, loc, contracted, resultType);
}

enum class ReductionKind { Maximum, Sum };

mlir::FailureOr<mlir::Value>
createRowReduction(mlir::OpBuilder &rewriter, mlir::Location loc,
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

class AttentionStructuredInsertionListener final
    : public mlir::RewriterBase::ForwardingListener {
public:
  AttentionStructuredInsertionListener(
      mlir::OpBuilder::Listener *forwardTo,
      llvm::SmallVectorImpl<mlir::Operation *> &inserted)
      : mlir::RewriterBase::ForwardingListener(forwardTo), inserted(inserted) {}

  void notifyOperationInserted(mlir::Operation *operation,
                               mlir::OpBuilder::InsertPoint previous) final {
    mlir::RewriterBase::ForwardingListener::notifyOperationInserted(operation,
                                                                    previous);
    if (mlir::isa<mlir::DestinationStyleOpInterface>(operation) &&
        mlir::isa<mlir::TilingInterface>(operation))
      inserted.push_back(operation);
  }

  void notifyOperationErased(mlir::Operation *operation) final {
    mlir::RewriterBase::ForwardingListener::notifyOperationErased(operation);
    llvm::erase(inserted, operation);
  }

private:
  llvm::SmallVectorImpl<mlir::Operation *> &inserted;
};

std::optional<int64_t> domainElements(const analysis::ExactIndexSet &domain) {
  if (domain.getForm() != analysis::ExactIndexSetForm::BoxUnion ||
      domain.getBoxes().empty())
    return std::nullopt;
  int64_t total = 0;
  for (const analysis::StaticRectangularIndexSet &box : domain.getBoxes()) {
    int64_t elements = 1;
    for (int64_t size : box.sizes)
      if (size <= 0 || llvm::MulOverflow(elements, size, elements))
        return std::nullopt;
    if (llvm::AddOverflow(total, elements, total))
      return std::nullopt;
  }
  return total;
}

} // namespace

PreparedAttentionDecompositionOutcome prepareSelectedAttentionDecomposition(
    const CanonicalAttentionWorkCoordinate &work) {
  std::set<AttentionActionId> expected;
  for (const AttentionWorkDescription &root : work.roots)
    for (const AttentionActionDescription &action : root.actions)
      if (!expected.insert(action.id).second)
        return BrokenPreparedAttention{
            BrokenPreparedAttentionReason::DuplicateIdentity,
            "attention work has duplicate action ID"};
  for (const AttentionWorkDescription &root : work.roots) {
    std::set<std::pair<AttentionWorkScopeId, AttentionOperandRole>> operands;
    for (const AttentionOperandDescription &operand : root.operands)
      if (!operands.emplace(operand.scope, operand.role).second ||
          operand.fragments.empty() ||
          operand.exactDomain.getForm() !=
              analysis::ExactIndexSetForm::BoxUnion ||
          operand.residentDomain.getForm() !=
              analysis::ExactIndexSetForm::BoxUnion ||
          operand.exactDomain.getBoxes().empty() ||
          operand.residentDomain.getBoxes().empty() ||
          operand.exactDomain.getRank() != operand.residentDomain.getRank() ||
          !operand.residentDomain.getPresburgerSet()
               .subtract(operand.exactDomain.getPresburgerSet())
               .isIntegerEmpty())
        return BrokenPreparedAttention{
            BrokenPreparedAttentionReason::InvalidWorkDescription,
            "attention operand is duplicate, incomplete, or outside its "
            "exact demand"};
    std::set<AttentionValueId> values;
    for (const AttentionValueDescription &value : root.values)
      if (!values.insert(value.id).second ||
          value.physicalVersion.has_value() != value.storage.has_value() ||
          value.exactDomain.getForm() !=
              analysis::ExactIndexSetForm::BoxUnion ||
          value.residentDomain.getForm() !=
              analysis::ExactIndexSetForm::BoxUnion ||
          value.exactDomain.getBoxes().empty() ||
          value.residentDomain.getBoxes().empty() ||
          value.exactDomain.getRank() != value.residentDomain.getRank() ||
          !value.residentDomain.getPresburgerSet()
               .subtract(value.exactDomain.getPresburgerSet())
               .isIntegerEmpty())
        return BrokenPreparedAttention{
            BrokenPreparedAttentionReason::InvalidWorkDescription,
            "attention work has duplicate, partial, or non-resident value "
            "binding"};
    for (const AttentionActionDescription &action : root.actions)
      for (const AttentionValueId &value :
           llvm::concat<const AttentionValueId>(action.inputs, action.outputs))
        if (!values.count(value))
          return BrokenPreparedAttention{
              BrokenPreparedAttentionReason::InvalidWorkDescription,
              "attention action references an unknown value"};
    std::set<AttentionScratchId> scratchIds;
    for (const AttentionScratchDescription &scratch : root.scratch) {
      if (!scratchIds.insert(scratch.id).second || scratch.uses.empty() ||
          !(scratch.id.scope == scratch.definition.scope) ||
          scratch.exactDomain.getForm() !=
              analysis::ExactIndexSetForm::BoxUnion ||
          scratch.residentDomain.getForm() !=
              analysis::ExactIndexSetForm::BoxUnion ||
          scratch.exactDomain.getBoxes().empty() ||
          scratch.residentDomain.getBoxes().empty() ||
          scratch.exactDomain.getRank() != scratch.residentDomain.getRank() ||
          !scratch.residentDomain.getPresburgerSet()
               .subtract(scratch.exactDomain.getPresburgerSet())
               .isIntegerEmpty() ||
          !expected.count(scratch.definition))
        return BrokenPreparedAttention{
            BrokenPreparedAttentionReason::InvalidWorkDescription,
            "attention lowering scratch is duplicate, incomplete, or outside "
            "the action plan"};
      for (const AttentionActionId &use : scratch.uses)
        if (!(use.scope == scratch.id.scope) || !expected.count(use))
          return BrokenPreparedAttention{
              BrokenPreparedAttentionReason::InvalidWorkDescription,
              "attention lowering scratch use is outside its action scope"};
    }
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

  llvm::SmallVector<mlir::Operation *, 16> insertedStructuredOperations;
  mlir::OpBuilder::Listener fallbackListener;
  mlir::OpBuilder::Listener *previousListener = rewriter.getListener();
  AttentionStructuredInsertionListener insertionListener(
      previousListener ? previousListener : &fallbackListener,
      insertedStructuredOperations);
  rewriter.setListener(&insertionListener);
  auto restoreListener =
      llvm::make_scope_exit([&] { rewriter.setListener(previousListener); });

  std::map<AttentionWorkScopeId,
           std::map<AttentionOperandRole, const AttentionOperandDescription *>>
      operands;
  for (const AttentionOperandDescription &operand : description.operands)
    if (!operands[operand.scope].try_emplace(operand.role, &operand).second) {
      setFailure(failureReason,
                 "selected attention operand description is duplicated");
      return mlir::failure();
    }

  SelectedAttentionRootMaterialization result;
  result.root = description.root;
  std::map<AttentionActionId, size_t> actionMapping;
  std::map<std::pair<AttentionWorkScopeId, AttentionOperandRole>, size_t>
      operandMapping;
  std::map<AttentionValueId, size_t> valueMapping;
  std::map<AttentionScratchId, size_t> scratchMapping;
  auto recordAction = [&](AttentionActionId id,
                          llvm::ArrayRef<mlir::Operation *> operations) {
    auto [position, inserted] =
        actionMapping.try_emplace(id, result.actions.size());
    if (inserted)
      result.actions.push_back({id, {}, {}});
    AttentionActionMaterialization &materialization =
        result.actions[position->second];
    for (mlir::Operation *operation : operations) {
      if (operation &&
          !llvm::is_contained(materialization.operations, operation))
        materialization.operations.push_back(operation);
    }
    for (mlir::Operation *operation : insertedStructuredOperations)
      if (!llvm::is_contained(materialization.structuredOperations, operation))
        materialization.structuredOperations.push_back(operation);
    insertedStructuredOperations.clear();
    return true;
  };
  auto recordOperand = [&](AttentionWorkScopeId scope,
                           AttentionOperandRole role, mlir::Value value) {
    if (!value)
      return false;
    auto [position, inserted] = operandMapping.try_emplace(
        std::make_pair(scope, role), result.operands.size());
    if (inserted)
      result.operands.push_back({scope, role, {}});
    auto &occurrences = result.operands[position->second].occurrences;
    if (!llvm::is_contained(occurrences, value))
      occurrences.push_back(value);
    return true;
  };
  auto recordValue = [&](AttentionValueId id, mlir::Value value) {
    if (!value)
      return false;
    auto [position, inserted] =
        valueMapping.try_emplace(id, result.values.size());
    if (inserted)
      result.values.push_back({id, {}});
    auto &occurrences = result.values[position->second].occurrences;
    if (!llvm::is_contained(occurrences, value))
      occurrences.push_back(value);
    return true;
  };
  auto recordScratch = [&](AttentionScratchId id, mlir::Value value) {
    if (!value)
      return false;
    auto [position, inserted] =
        scratchMapping.try_emplace(id, result.scratch.size());
    if (inserted)
      result.scratch.push_back({id, {}});
    auto &occurrences = result.scratch[position->second].occurrences;
    if (!llvm::is_contained(occurrences, value))
      occurrences.push_back(value);
    return true;
  };
  std::set<mlir::Operation *> scopedOperations;
  auto recordScopeOperations = [&](AttentionWorkScopeId scope) {
    AttentionScopeOperationMaterialization materialization{scope, {}};
    for (const AttentionActionMaterialization &action : result.actions)
      if (action.id.scope == scope)
        for (mlir::Operation *operation : action.structuredOperations) {
          if (!scopedOperations.insert(operation).second)
            return false;
          materialization.operations.push_back(operation);
        }
    if (materialization.operations.empty())
      return false;
    result.scopes.push_back(std::move(materialization));
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
    auto scaledType =
        mlir::dyn_cast<mlir::FloatType>(scaledDescription->elementType);
    auto probabilityType =
        mlir::dyn_cast<mlir::FloatType>(probabilityDescription->elementType);
    auto accumulatorType =
        mlir::dyn_cast<mlir::FloatType>(accumulatorDescription->elementType);
    if (!scoreType || !scaledType || !probabilityType || !accumulatorType)
      return mlir::failure();
    mlir::FailureOr<analysis::StaticRectangularIndexSet> exactScore =
        singleBox(scoreDescription->exactDomain);
    mlir::FailureOr<analysis::StaticRectangularIndexSet> residentScore =
        singleBox(scoreDescription->residentDomain);
    if (mlir::failed(exactScore) || mlir::failed(residentScore) ||
        exactScore->sizes.size() != scoreAxes.size() ||
        residentScore->sizes.size() != scoreAxes.size())
      return mlir::failure();
    const bool temporalSplit = exactScore->sizes != residentScore->sizes;

    auto findOperand =
        [&](AttentionOperandRole role) -> const AttentionOperandDescription * {
      auto found = scopeOperands->second.find(role);
      return found == scopeOperands->second.end() ? nullptr : found->second;
    };
    const AttentionOperandDescription *queryDescription =
        findOperand(AttentionOperandRole::Query);
    const AttentionOperandDescription *keyDescription =
        findOperand(AttentionOperandRole::Key);
    const AttentionOperandDescription *valueDescription =
        findOperand(AttentionOperandRole::Value);
    const AttentionOperandDescription *maskDescription =
        attention.getMask() ? findOperand(AttentionOperandRole::Mask) : nullptr;
    if (!queryDescription || !keyDescription || !valueDescription ||
        (attention.getMask() && !maskDescription))
      return mlir::failure();
    auto getOperand = [&](AttentionOperandRole role, mlir::Value source,
                          const AttentionOperandDescription &operand,
                          bool assemble) -> mlir::FailureOr<mlir::Value> {
      if (!assemble)
        return source;
      mlir::FailureOr<mlir::Value> assembled =
          assembleOperand(rewriter, loc, source, operand.exactDomain);
      if (mlir::failed(assembled) && failureReason) {
        std::string detail;
        llvm::raw_string_ostream stream(detail);
        stream << "selected attention operand cannot be assembled: role="
               << static_cast<unsigned>(role) << " source=" << source.getType()
               << " rank=" << operand.exactDomain.getRank() << " form="
               << static_cast<unsigned>(operand.exactDomain.getForm())
               << " boxes=" << operand.exactDomain.getBoxes().size()
               << " physical_fragments=" << operand.fragments.size();
        *failureReason = stream.str();
      }
      return assembled;
    };
    mlir::FailureOr<mlir::Value> query = getOperand(
        AttentionOperandRole::Query, attention.getQuery(), *queryDescription,
        /*assemble=*/true);
    mlir::FailureOr<mlir::Value> key = getOperand(
        AttentionOperandRole::Key, attention.getKey(), *keyDescription,
        /*assemble=*/!temporalSplit);
    mlir::FailureOr<mlir::Value> value = getOperand(
        AttentionOperandRole::Value, attention.getValue(), *valueDescription,
        /*assemble=*/!temporalSplit);
    mlir::FailureOr<mlir::Value> mask = mlir::failure();
    if (attention.getMask())
      mask = getOperand(AttentionOperandRole::Mask, attention.getMask(),
                        *maskDescription, /*assemble=*/!temporalSplit);
    if (mlir::failed(query) || mlir::failed(key) || mlir::failed(value) ||
        (attention.getMask() && mlir::failed(mask))) {
      if (!failureReason || failureReason->empty())
        setFailure(failureReason,
                   "selected attention operand pieces cannot be assembled");
      return mlir::failure();
    }
    if (!recordOperand(scope, AttentionOperandRole::Query, *query) ||
        (!temporalSplit &&
         (!recordOperand(scope, AttentionOperandRole::Key, *key) ||
          !recordOperand(scope, AttentionOperandRole::Value, *value) ||
          (attention.getMask() &&
           !recordOperand(scope, AttentionOperandRole::Mask, *mask)))))
      return mlir::failure();

    auto emitScaleMask = [&](mlir::OpBuilder &builder, mlir::Value score,
                             mlir::Value selectedMask,
                             llvm::ArrayRef<int64_t> scoreShape)
        -> mlir::FailureOr<mlir::Value> {
      mlir::AffineMap scoreIdentity = mlir::AffineMap::getMultiDimIdentityMap(
          scoreShape.size(), builder.getContext());
      mlir::Value convertedScore = score;
      if (scoreType != scaledType) {
        mlir::FailureOr<mlir::Value> converted =
            createFloatConvert(builder, loc, score, scaledType);
        if (mlir::failed(converted) ||
            !recordScratch({scope, AttentionScratchKind::ConvertedScoreBlock},
                           *converted))
          return mlir::failure();
        convertedScore = *converted;
      }

      mlir::Value convertedScale =
          convertScalar(builder, loc, attention.getScale(), scaledType);
      mlir::Value scaleBlock =
          createFill(builder, loc, scoreShape, scaledType, convertedScale);
      if (!recordScratch({scope, AttentionScratchKind::ScaleBlock}, scaleBlock))
        return mlir::failure();
      mlir::Value scaledScore =
          createPointwise(builder, loc, {convertedScore, scaleBlock},
                          {scoreIdentity, scoreIdentity}, scoreShape,
                          scaledType, PointwiseKind::Multiply);
      if (!attention.getMask())
        return scaledScore;
      if (!selectedMask)
        return mlir::failure();
      if (!recordScratch({scope, AttentionScratchKind::ScaledScoreBlock},
                         scaledScore))
        return mlir::failure();

      mlir::FailureOr<mlir::AffineMap> compactMask =
          compressMap(*attention.getMaskMap(), scoreAxes);
      auto selectedMaskType =
          mlir::dyn_cast<mlir::RankedTensorType>(selectedMask.getType());
      if (mlir::failed(compactMask) || !selectedMaskType)
        return mlir::failure();
      mlir::Value materializedMask = selectedMask;
      if (!compactMask->isIdentity() ||
          selectedMaskType.getRank() !=
              static_cast<int64_t>(scoreShape.size()) ||
          !llvm::equal(selectedMaskType.getShape(), scoreShape)) {
        materializedMask = createPointwise(
            builder, loc, {selectedMask}, {*compactMask}, scoreShape,
            selectedMaskType.getElementType(), PointwiseKind::Identity);
        if (!recordScratch({scope, AttentionScratchKind::BroadcastMaskBlock},
                           materializedMask))
          return mlir::failure();
      }
      auto materializedMaskType =
          mlir::cast<mlir::RankedTensorType>(materializedMask.getType());
      if (materializedMaskType.getElementType() != scaledType) {
        mlir::FailureOr<mlir::Value> converted =
            createFloatConvert(builder, loc, materializedMask, scaledType);
        if (mlir::failed(converted) ||
            !recordScratch({scope, AttentionScratchKind::ConvertedMaskBlock},
                           *converted))
          return mlir::failure();
        materializedMask = *converted;
      }
      return createPointwise(builder, loc, {scaledScore, materializedMask},
                             {scoreIdentity, scoreIdentity}, scoreShape,
                             scaledType, PointwiseKind::Add);
    };

    auto emitProbability = [&](mlir::OpBuilder &builder, mlir::Value scaled,
                               mlir::Value maximum,
                               llvm::ArrayRef<int64_t> scoreShape,
                               llvm::ArrayRef<int64_t> probabilityShape,
                               llvm::ArrayRef<int64_t> rowShape)
        -> mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> {
      mlir::AffineMap rowMap = mapForDims(
          builder.getContext(), scoreShape.size(), sequence(rowShape.size()));
      mlir::AffineMap scoreIdentity = mlir::AffineMap::getMultiDimIdentityMap(
          scoreShape.size(), builder.getContext());
      mlir::Value broadcastMaximum =
          createPointwise(builder, loc, {maximum}, {rowMap}, scoreShape,
                          scaledType, PointwiseKind::Identity);
      mlir::Value shifted =
          createPointwise(builder, loc, {scaled, broadcastMaximum},
                          {scoreIdentity, scoreIdentity}, scoreShape,
                          scaledType, PointwiseKind::Subtract);
      mlir::Value exponent =
          createPointwise(builder, loc, {shifted}, {scoreIdentity},
                          probabilityShape, scaledType, PointwiseKind::Exp);
      if (!recordScratch({scope, AttentionScratchKind::BroadcastMaximumBlock},
                         broadcastMaximum) ||
          !recordScratch({scope, AttentionScratchKind::ShiftedScoreBlock},
                         shifted))
        return mlir::failure();
      mlir::Value probability = exponent;
      if (scaledType != probabilityType) {
        if (!recordScratch({scope, AttentionScratchKind::WideProbabilityBlock},
                           exponent))
          return mlir::failure();
        mlir::FailureOr<mlir::Value> converted =
            createFloatConvert(builder, loc, exponent, probabilityType);
        if (mlir::failed(converted))
          return mlir::failure();
        probability = *converted;
      }
      return std::pair<mlir::Value, mlir::Value>{probability, exponent};
    };

    State state;
    if (temporalSplit) {
      auto maximumType =
          mlir::dyn_cast<mlir::FloatType>(maximumDescription->elementType);
      auto sumType =
          mlir::dyn_cast<mlir::FloatType>(sumDescription->elementType);
      if (!maximumType || !sumType || maximumType != scaledType ||
          sumType != scaledType || roles->keyValueReduction.empty()) {
        setFailure(failureReason,
                   "selected attention block state types are inconsistent");
        return mlir::failure();
      }

      llvm::SmallVector<int64_t, 4> keyValueExtents;
      llvm::SmallVector<int64_t, 4> keyValueTiles;
      for (unsigned axis : roles->keyValueReduction) {
        auto position = llvm::find(scoreAxes, axis);
        if (position == scoreAxes.end())
          return mlir::failure();
        unsigned index = position - scoreAxes.begin();
        int64_t extent = exactScore->sizes[index];
        int64_t tile = residentScore->sizes[index];
        if (extent <= 0 || tile <= 0 || tile > extent)
          return mlir::failure();
        keyValueExtents.push_back(extent);
        keyValueTiles.push_back(tile);
      }
      for (auto [position, axis] : llvm::enumerate(scoreAxes))
        if (!llvm::is_contained(roles->keyValueReduction, axis) &&
            residentScore->sizes[position] != exactScore->sizes[position]) {
          setFailure(failureReason,
                     "selected attention recurrence received a non-K2 "
                     "temporal tile");
          return mlir::failure();
        }

      auto blockShape = [&](const AttentionValueDescription &description,
                            llvm::ArrayRef<int64_t> keyValueSizes)
          -> mlir::FailureOr<llvm::SmallVector<int64_t, 6>> {
        mlir::FailureOr<analysis::StaticRectangularIndexSet> exact =
            singleBox(description.exactDomain);
        if (mlir::failed(exact) ||
            description.indexingMap.getNumResults() != exact->sizes.size())
          return mlir::failure();
        llvm::SmallVector<int64_t, 6> shape(exact->sizes.begin(),
                                            exact->sizes.end());
        for (auto [dimension, expression] :
             llvm::enumerate(description.indexingMap.getResults())) {
          auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
          if (!iterator)
            return mlir::failure();
          auto keyValueAxis =
              llvm::find(roles->keyValueReduction, iterator.getPosition());
          if (keyValueAxis != roles->keyValueReduction.end())
            shape[dimension] =
                keyValueSizes[keyValueAxis - roles->keyValueReduction.begin()];
        }
        return shape;
      };
      llvm::SmallVector<int64_t, 6> rowShape = shapeOf(*maximumDescription);
      llvm::SmallVector<int64_t, 6> outputShape =
          shapeOf(*accumulatorDescription);
      if (rowShape.empty() || outputShape.empty())
        return mlir::failure();
      mlir::Value negativeInfinity = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getFloatAttr(maximumType,
                                     -std::numeric_limits<double>::infinity()));
      mlir::Value zeroSum = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getFloatAttr(sumType, 0.0));
      mlir::Value zeroAccumulator = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getFloatAttr(accumulatorType, 0.0));
      state = {
          createFill(rewriter, loc, rowShape, maximumType, negativeInfinity),
          createFill(rewriter, loc, rowShape, sumType, zeroSum),
          createFill(rewriter, loc, outputShape, accumulatorType,
                     zeroAccumulator)};
      if (!recordAction({scope, AttentionActionKind::StateUpdate},
                        {state.maximum.getDefiningOp(),
                         state.sum.getDefiningOp(),
                         state.accumulator.getDefiningOp()}))
        return mlir::failure();
      for (auto [kind, value] :
           {std::pair{AttentionValueKind::RunningMaximum, state.maximum},
            std::pair{AttentionValueKind::RunningSum, state.sum},
            std::pair{AttentionValueKind::RunningAccumulator,
                      state.accumulator}})
        if (!recordValue({scope, kind}, value))
          return mlir::failure();

      auto emitBlock = [&](mlir::OpBuilder &builder,
                           llvm::ArrayRef<mlir::OpFoldResult> keyValueOffsets,
                           llvm::ArrayRef<int64_t> keyValueSizes,
                           State previous) -> mlir::FailureOr<State> {
        mlir::FailureOr<mlir::Value> blockKey = assembleTemporalOperandBlock(
            builder, loc, attention.getKey(), *keyDescription,
            roles->keyValueReduction, keyValueOffsets, keyValueSizes);
        mlir::FailureOr<mlir::Value> blockValue = assembleTemporalOperandBlock(
            builder, loc, attention.getValue(), *valueDescription,
            roles->keyValueReduction, keyValueOffsets, keyValueSizes);
        mlir::FailureOr<mlir::Value> blockMask = mlir::failure();
        if (attention.getMask())
          blockMask = assembleTemporalOperandBlock(
              builder, loc, attention.getMask(), *maskDescription,
              roles->keyValueReduction, keyValueOffsets, keyValueSizes);
        if (mlir::failed(blockKey) || mlir::failed(blockValue) ||
            (attention.getMask() && mlir::failed(blockMask)))
          return mlir::failure();
        if (!recordOperand(scope, AttentionOperandRole::Key, *blockKey) ||
            !recordOperand(scope, AttentionOperandRole::Value, *blockValue) ||
            (attention.getMask() &&
             !recordOperand(scope, AttentionOperandRole::Mask, *blockMask)))
          return mlir::failure();

        mlir::FailureOr<llvm::SmallVector<int64_t, 6>> scoreShape =
            blockShape(*scoreDescription, keyValueSizes);
        mlir::FailureOr<llvm::SmallVector<int64_t, 6>> scaledShape =
            blockShape(*scaledDescription, keyValueSizes);
        mlir::FailureOr<llvm::SmallVector<int64_t, 6>> probabilityShape =
            blockShape(*probabilityDescription, keyValueSizes);
        if (mlir::failed(scoreShape) || mlir::failed(scaledShape) ||
            mlir::failed(probabilityShape))
          return mlir::failure();

        mlir::FailureOr<mlir::Value> score = createContraction(
            builder, loc, *query, *blockKey, qkAxes, roles->queryKeyReduction,
            attention.getQueryMap(), attention.getKeyMap(),
            mapForDims(builder.getContext(), attention.getIterationDomainRank(),
                       scoreAxes),
            *scoreShape, scoreType);
        if (mlir::failed(score) ||
            !recordValue({scope, AttentionValueKind::ScoreBlock}, *score) ||
            !recordAction({scope, AttentionActionKind::QueryKeyContraction},
                          {score->getDefiningOp()}))
          return mlir::failure();

        mlir::FailureOr<mlir::Value> scaledResult = emitScaleMask(
            builder, *score, attention.getMask() ? *blockMask : mlir::Value{},
            *scaledShape);
        if (mlir::failed(scaledResult))
          return mlir::failure();
        mlir::Value scaled = *scaledResult;
        if (!recordValue({scope, AttentionValueKind::ScaledMaskedScoreBlock},
                         scaled) ||
            !recordAction({scope, AttentionActionKind::ScaleMask},
                          {scaled.getDefiningOp()}))
          return mlir::failure();

        mlir::FailureOr<mlir::Value> blockMaximum = createRowReduction(
            builder, loc, scaled, roles->keyValueReduction.size(), rowShape,
            scaledType, ReductionKind::Maximum);
        if (mlir::failed(blockMaximum) ||
            !recordValue({scope, AttentionValueKind::BlockMaximum},
                         *blockMaximum) ||
            !recordAction({scope, AttentionActionKind::RowMaximum},
                          {blockMaximum->getDefiningOp()}))
          return mlir::failure();

        mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> probabilityResult =
            emitProbability(builder, scaled, *blockMaximum, *scaledShape,
                            *probabilityShape, rowShape);
        if (mlir::failed(probabilityResult) ||
            !recordValue({scope, AttentionValueKind::ProbabilityBlock},
                         probabilityResult->first) ||
            !recordAction({scope, AttentionActionKind::Exponential},
                          {probabilityResult->first.getDefiningOp()}))
          return mlir::failure();
        mlir::Value probability = probabilityResult->first;
        mlir::Value exponent = probabilityResult->second;

        mlir::FailureOr<mlir::Value> blockSum = createRowReduction(
            builder, loc, exponent, roles->keyValueReduction.size(), rowShape,
            scaledType, ReductionKind::Sum);
        if (mlir::failed(blockSum) ||
            !recordValue({scope, AttentionValueKind::BlockSum}, *blockSum) ||
            !recordAction({scope, AttentionActionKind::RowSum},
                          {blockSum->getDefiningOp()}))
          return mlir::failure();

        mlir::FailureOr<mlir::Value> blockAccumulator = createContraction(
            builder, loc, probability, *blockValue, pvAxes,
            roles->keyValueReduction,
            mapForDims(builder.getContext(), attention.getIterationDomainRank(),
                       scoreAxes),
            attention.getValueMap(), attention.getOutputMap(), outputShape,
            accumulatorType);
        if (mlir::failed(blockAccumulator) ||
            !recordValue({scope, AttentionValueKind::BlockAccumulator},
                         *blockAccumulator) ||
            !recordAction({scope, AttentionActionKind::ValueContraction},
                          {blockAccumulator->getDefiningOp()}))
          return mlir::failure();

        mlir::AffineMap rowIdentity = mlir::AffineMap::getMultiDimIdentityMap(
            rowShape.size(), builder.getContext());
        mlir::AffineMap outputIdentity =
            mlir::AffineMap::getMultiDimIdentityMap(outputShape.size(),
                                                    builder.getContext());
        mlir::AffineMap outputRowMap =
            mapForDims(builder.getContext(), outputShape.size(),
                       sequence(rowShape.size()));
        mlir::Value updatedMaximum =
            createPointwise(builder, loc, {previous.maximum, *blockMaximum},
                            {rowIdentity, rowIdentity}, rowShape, maximumType,
                            PointwiseKind::Maximum);
        mlir::Value previousDelta =
            createPointwise(builder, loc, {previous.maximum, updatedMaximum},
                            {rowIdentity, rowIdentity}, rowShape, maximumType,
                            PointwiseKind::Subtract);
        mlir::Value blockDelta =
            createPointwise(builder, loc, {*blockMaximum, updatedMaximum},
                            {rowIdentity, rowIdentity}, rowShape, maximumType,
                            PointwiseKind::Subtract);
        mlir::Value previousScale =
            createPointwise(builder, loc, {previousDelta}, {rowIdentity},
                            rowShape, maximumType, PointwiseKind::Exp);
        mlir::Value blockScale =
            createPointwise(builder, loc, {blockDelta}, {rowIdentity}, rowShape,
                            maximumType, PointwiseKind::Exp);
        mlir::Value updatedSum = createPointwise(
            builder, loc, {previousScale, previous.sum, blockScale, *blockSum},
            {rowIdentity, rowIdentity, rowIdentity, rowIdentity}, rowShape,
            sumType, PointwiseKind::WeightedAdd);
        mlir::FailureOr<mlir::Value> previousAccumulator =
            createFloatConvert(builder, loc, previous.accumulator, maximumType);
        mlir::FailureOr<mlir::Value> currentAccumulator =
            createFloatConvert(builder, loc, *blockAccumulator, maximumType);
        if (mlir::failed(previousAccumulator) ||
            mlir::failed(currentAccumulator))
          return mlir::failure();
        mlir::Value updatedAccumulator = createPointwise(
            builder, loc,
            {previousScale, *previousAccumulator, blockScale,
             *currentAccumulator},
            {outputRowMap, outputIdentity, outputRowMap, outputIdentity},
            outputShape, maximumType, PointwiseKind::WeightedAdd);
        mlir::FailureOr<mlir::Value> storedAccumulator = createFloatConvert(
            builder, loc, updatedAccumulator, accumulatorType);
        if (mlir::failed(storedAccumulator))
          return mlir::failure();
        State updated{updatedMaximum, updatedSum, *storedAccumulator};
        for (auto [kind, stateValue] :
             {std::pair{AttentionValueKind::RunningMaximum, updated.maximum},
              std::pair{AttentionValueKind::RunningSum, updated.sum},
              std::pair{AttentionValueKind::RunningAccumulator,
                        updated.accumulator}})
          if (!recordValue({scope, kind}, stateValue))
            return mlir::failure();
        if (!recordAction({scope, AttentionActionKind::StateUpdate},
                          {updated.maximum.getDefiningOp(),
                           updated.sum.getDefiningOp(),
                           updated.accumulator.getDefiningOp()}))
          return mlir::failure();
        return updated;
      };

      std::function<mlir::FailureOr<State>(
          mlir::OpBuilder &, unsigned,
          llvm::SmallVectorImpl<mlir::OpFoldResult> &,
          llvm::SmallVectorImpl<int64_t> &, State)>
          emitKeyValueBlocks;
      emitKeyValueBlocks =
          [&](mlir::OpBuilder &builder, unsigned dimension,
              llvm::SmallVectorImpl<mlir::OpFoldResult> &offsets,
              llvm::SmallVectorImpl<int64_t> &sizes,
              State initial) -> mlir::FailureOr<State> {
        if (dimension == keyValueExtents.size())
          return emitBlock(builder, offsets, sizes, initial);
        const int64_t extent = keyValueExtents[dimension];
        const int64_t tile = keyValueTiles[dimension];
        if (tile == extent) {
          offsets.push_back(builder.getIndexAttr(0));
          sizes.push_back(extent);
          mlir::FailureOr<State> result = emitKeyValueBlocks(
              builder, dimension + 1, offsets, sizes, initial);
          sizes.pop_back();
          offsets.pop_back();
          return result;
        }
        State current = initial;
        for (int64_t offset = 0; offset < extent; offset += tile) {
          const int64_t blockSize = std::min(tile, extent - offset);
          offsets.push_back(builder.getIndexAttr(offset));
          sizes.push_back(blockSize);
          mlir::FailureOr<State> body = emitKeyValueBlocks(
              builder, dimension + 1, offsets, sizes, current);
          sizes.pop_back();
          offsets.pop_back();
          if (mlir::failed(body))
            return mlir::failure();
          current = *body;
        }
        return current;
      };
      llvm::SmallVector<mlir::OpFoldResult, 4> blockOffsets;
      llvm::SmallVector<int64_t, 4> blockSizes;
      mlir::FailureOr<State> emitted = emitKeyValueBlocks(
          rewriter, /*dimension=*/0, blockOffsets, blockSizes, state);
      if (mlir::failed(emitted))
        return mlir::failure();
      state = *emitted;
    } else {
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

      llvm::SmallVector<int64_t, 6> selectedScoreShape =
          shapeOf(*scaledDescription);
      mlir::FailureOr<mlir::Value> scaledResult = emitScaleMask(
          rewriter, *score, attention.getMask() ? *mask : mlir::Value{},
          selectedScoreShape);
      if (mlir::failed(scaledResult))
        return mlir::failure();
      mlir::Value scaled = *scaledResult;
      if (!recordValue({scope, AttentionValueKind::ScaledMaskedScoreBlock},
                       scaled) ||
          !recordAction({scope, AttentionActionKind::ScaleMask},
                        {scaled.getDefiningOp()}))
        return mlir::failure();

      mlir::FailureOr<mlir::Value> maximum = createRowReduction(
          rewriter, loc, scaled, roles->keyValueReduction.size(),
          shapeOf(*maximumDescription), scaledType, ReductionKind::Maximum);
      if (mlir::failed(maximum) ||
          !recordValue({scope, AttentionValueKind::BlockMaximum}, *maximum) ||
          !recordAction({scope, AttentionActionKind::RowMaximum},
                        {maximum->getDefiningOp()}))
        return mlir::failure();

      llvm::SmallVector<int64_t, 6> selectedProbabilityShape =
          shapeOf(*probabilityDescription);
      llvm::SmallVector<int64_t, 6> selectedRowShape =
          shapeOf(*maximumDescription);
      mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> probabilityResult =
          emitProbability(rewriter, scaled, *maximum, selectedScoreShape,
                          selectedProbabilityShape, selectedRowShape);
      if (mlir::failed(probabilityResult) ||
          !recordValue({scope, AttentionValueKind::ProbabilityBlock},
                       probabilityResult->first) ||
          !recordAction({scope, AttentionActionKind::Exponential},
                        {probabilityResult->first.getDefiningOp()}))
        return mlir::failure();
      mlir::Value probability = probabilityResult->first;
      mlir::Value exponent = probabilityResult->second;

      mlir::FailureOr<mlir::Value> sum = createRowReduction(
          rewriter, loc, exponent, roles->keyValueReduction.size(),
          shapeOf(*sumDescription), scaledType, ReductionKind::Sum);
      if (mlir::failed(sum) ||
          !recordValue({scope, AttentionValueKind::BlockSum}, *sum) ||
          !recordAction({scope, AttentionActionKind::RowSum},
                        {sum->getDefiningOp()}))
        return mlir::failure();

      mlir::FailureOr<mlir::Value> accumulator = createContraction(
          rewriter, loc, probability, *value, pvAxes, roles->keyValueReduction,
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

      state = {*maximum, *sum, *accumulator};
      for (auto [kind, value] :
           {std::pair{AttentionValueKind::RunningMaximum, state.maximum},
            std::pair{AttentionValueKind::RunningSum, state.sum},
            std::pair{AttentionValueKind::RunningAccumulator,
                      state.accumulator}})
        if (!recordValue({scope, kind}, value))
          return mlir::failure();
      if (!recordAction({scope, AttentionActionKind::StateUpdate},
                        {state.maximum.getDefiningOp(),
                         state.sum.getDefiningOp(),
                         state.accumulator.getDefiningOp()}))
        return mlir::failure();
    }

    if (scope.group) {
      contributions[*scope.group].push_back({scope, state});
      if (!recordScopeOperations(scope))
        return mlir::failure();
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
    if (!recordScopeOperations(scope))
      return mlir::failure();
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
    if (!recordScopeOperations(scope))
      return mlir::failure();
  }

  std::set<AttentionActionId> describedActions;
  for (const AttentionActionDescription &action : description.actions)
    describedActions.insert(action.id);
  std::set<AttentionActionId> emittedActions;
  for (const AttentionActionMaterialization &action : result.actions)
    if (action.operations.empty() || !emittedActions.insert(action.id).second) {
      setFailure(failureReason,
                 "selected attention action materialization is incomplete");
      return mlir::failure();
    }
  if (emittedActions != describedActions) {
    setFailure(failureReason,
               "selected attention action set differs from prepared work");
    return mlir::failure();
  }

  std::map<AttentionValueId, const AttentionValueMaterialization *>
      materializedValues;
  for (const AttentionValueMaterialization &value : result.values)
    if (value.occurrences.empty() ||
        !materializedValues.try_emplace(value.id, &value).second) {
      setFailure(failureReason,
                 "selected attention value occurrence mapping is incomplete");
      return mlir::failure();
    }
  std::map<AttentionWorkScopeId, size_t> scoreOccurrences;
  for (const AttentionValueMaterialization &value : result.values)
    if (value.id.kind == AttentionValueKind::ScoreBlock)
      scoreOccurrences[value.id.scope] = value.occurrences.size();
  for (const AttentionValueDescription &value : description.values) {
    auto materialized = materializedValues.find(value.id);
    if (materialized == materializedValues.end() ||
        value.residentDomain.getBoxes().size() != 1) {
      setFailure(failureReason,
                 "selected attention omitted one described value");
      return mlir::failure();
    }
    const analysis::StaticRectangularIndexSet &resident =
        value.residentDomain.getBoxes().front();
    int64_t materializedElements = 0;
    const size_t occurrenceCount = materialized->second->occurrences.size();
    for (mlir::Value occurrence : materialized->second->occurrences) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(occurrence.getType());
      if (!type || !type.hasStaticShape() ||
          type.getElementType() != value.elementType ||
          type.getRank() != static_cast<int64_t>(resident.sizes.size()) ||
          value.indexingMap.getNumResults() != resident.sizes.size()) {
        setFailure(failureReason,
                   "selected attention occurrence type differs from plan");
        return mlir::failure();
      }
      int64_t elements = 1;
      for (auto [dimension, expression] :
           llvm::enumerate(value.indexingMap.getResults())) {
        auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!iterator)
          return mlir::failure();
        const int64_t extent = type.getDimSize(dimension);
        const bool keyValueAxis = llvm::is_contained(roles->keyValueReduction,
                                                     iterator.getPosition());
        if (extent <= 0 ||
            (keyValueAxis ? extent > resident.sizes[dimension]
                          : extent != resident.sizes[dimension]) ||
            llvm::MulOverflow(elements, extent, elements)) {
          setFailure(
              failureReason,
              "selected attention occurrence exceeds its resident domain");
          return mlir::failure();
        }
      }
      if (llvm::AddOverflow(materializedElements, elements,
                            materializedElements))
        return mlir::failure();
    }
    auto scoreCount = scoreOccurrences.find(value.id.scope);
    switch (value.id.kind) {
    case AttentionValueKind::ScoreBlock:
    case AttentionValueKind::ScaledMaskedScoreBlock:
    case AttentionValueKind::ProbabilityBlock: {
      std::optional<int64_t> exactElements = domainElements(value.exactDomain);
      if (!exactElements || materializedElements != *exactElements ||
          (value.id.kind != AttentionValueKind::ScoreBlock &&
           (scoreCount == scoreOccurrences.end() ||
            occurrenceCount != scoreCount->second))) {
        setFailure(failureReason,
                   "selected attention block occurrences do not cover exact "
                   "work");
        return mlir::failure();
      }
      break;
    }
    case AttentionValueKind::BlockMaximum:
    case AttentionValueKind::BlockSum:
    case AttentionValueKind::BlockAccumulator:
      if (scoreCount == scoreOccurrences.end() ||
          occurrenceCount != scoreCount->second) {
        setFailure(failureReason,
                   "selected attention block state occurrence count differs");
        return mlir::failure();
      }
      break;
    case AttentionValueKind::RunningMaximum:
    case AttentionValueKind::RunningSum:
    case AttentionValueKind::RunningAccumulator:
      if (scoreCount != scoreOccurrences.end()) {
        const AttentionValueDescription *score = findValue(
            description, {value.id.scope, AttentionValueKind::ScoreBlock});
        const bool split =
            score && !score->exactDomain.getPresburgerSet().isEqual(
                         score->residentDomain.getPresburgerSet());
        const size_t expected = split ? scoreCount->second + 1 : size_t{1};
        if (occurrenceCount != expected) {
          setFailure(failureReason,
                     "selected attention running state occurrence count "
                     "differs");
          return mlir::failure();
        }
      } else if (occurrenceCount != 1) {
        return mlir::failure();
      }
      break;
    case AttentionValueKind::FinalOutput:
      if (occurrenceCount != 1)
        return mlir::failure();
      break;
    }
  }
  if (materializedValues.size() != description.values.size()) {
    setFailure(failureReason,
               "selected attention materialized an undescribed value");
    return mlir::failure();
  }

  std::map<AttentionScratchId, const AttentionScratchMaterialization *>
      materializedScratch;
  for (const AttentionScratchMaterialization &scratch : result.scratch)
    if (scratch.occurrences.empty() ||
        !materializedScratch.try_emplace(scratch.id, &scratch).second) {
      setFailure(failureReason,
                 "selected attention scratch occurrence mapping is incomplete");
      return mlir::failure();
    }
  for (const AttentionScratchDescription &scratch : description.scratch) {
    auto materialized = materializedScratch.find(scratch.id);
    auto scoreCount = scoreOccurrences.find(scratch.id.scope);
    if (materialized == materializedScratch.end() ||
        scoreCount == scoreOccurrences.end() ||
        scratch.residentDomain.getBoxes().size() != 1) {
      setFailure(failureReason,
                 "selected attention omitted one planned lowering scratch");
      return mlir::failure();
    }
    const analysis::StaticRectangularIndexSet &resident =
        scratch.residentDomain.getBoxes().front();
    int64_t materializedElements = 0;
    const size_t occurrenceCount = materialized->second->occurrences.size();
    for (mlir::Value occurrence : materialized->second->occurrences) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(occurrence.getType());
      if (!type || !type.hasStaticShape() ||
          type.getElementType() != scratch.elementType ||
          type.getRank() != static_cast<int64_t>(resident.sizes.size()) ||
          scratch.indexingMap.getNumResults() != resident.sizes.size()) {
        setFailure(failureReason,
                   "selected attention scratch type differs from plan");
        return mlir::failure();
      }
      int64_t elements = 1;
      for (auto [dimension, expression] :
           llvm::enumerate(scratch.indexingMap.getResults())) {
        auto iterator = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
        if (!iterator)
          return mlir::failure();
        const int64_t extent = type.getDimSize(dimension);
        const bool keyValueAxis = llvm::is_contained(roles->keyValueReduction,
                                                     iterator.getPosition());
        if (extent <= 0 ||
            (keyValueAxis ? extent > resident.sizes[dimension]
                          : extent != resident.sizes[dimension]) ||
            llvm::MulOverflow(elements, extent, elements)) {
          setFailure(failureReason,
                     "selected attention scratch exceeds its resident domain");
          return mlir::failure();
        }
      }
      if (llvm::AddOverflow(materializedElements, elements,
                            materializedElements))
        return mlir::failure();
    }
    std::optional<int64_t> exactElements = domainElements(scratch.exactDomain);
    if (!exactElements || materializedElements != *exactElements ||
        occurrenceCount != scoreCount->second) {
      setFailure(failureReason,
                 "selected attention scratch occurrences do not cover exact "
                 "work");
      return mlir::failure();
    }
  }
  if (materializedScratch.size() != description.scratch.size()) {
    setFailure(failureReason,
               "selected attention materialized undescribed lowering scratch");
    return mlir::failure();
  }
  if (result.operands.size() != description.operands.size() ||
      llvm::any_of(result.operands, [](const auto &operand) {
        return operand.occurrences.empty();
      })) {
    setFailure(failureReason,
               "selected attention omitted one planned operand occurrence");
    return mlir::failure();
  }
  if (!insertedStructuredOperations.empty()) {
    setFailure(failureReason,
               "selected attention created structured operations outside an "
               "action boundary");
    return mlir::failure();
  }

  rewriter.replaceOp(attention, assembledOutput);
  result.result = assembledOutput;
  llvm::sort(result.actions, [](const AttentionActionMaterialization &lhs,
                                const AttentionActionMaterialization &rhs) {
    return lhs.id < rhs.id;
  });
  llvm::sort(result.operands, [](const AttentionOperandMaterialization &lhs,
                                 const AttentionOperandMaterialization &rhs) {
    return std::tie(lhs.scope, lhs.role) < std::tie(rhs.scope, rhs.role);
  });
  llvm::sort(result.values, [](const AttentionValueMaterialization &lhs,
                               const AttentionValueMaterialization &rhs) {
    return lhs.id < rhs.id;
  });
  llvm::sort(result.scratch, [](const AttentionScratchMaterialization &lhs,
                                const AttentionScratchMaterialization &rhs) {
    return lhs.id < rhs.id;
  });
  llvm::sort(result.scopes,
             [](const AttentionScopeOperationMaterialization &lhs,
                const AttentionScopeOperationMaterialization &rhs) {
               return lhs.scope < rhs.scope;
             });
  return result;
}

} // namespace wafer::compiler::detail
