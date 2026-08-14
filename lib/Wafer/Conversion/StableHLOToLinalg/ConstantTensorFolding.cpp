//===- ConstantTensorFolding.cpp - Static tensor residual cleanup -------===//

#include "ConstantTensorFoldingInternal.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace wafer::stablehlo_normalization {

static constexpr int64_t kMaximumFoldedTensorElements = 1 << 20;
static constexpr uint64_t kMaximumFoldedTensorBytes = 16 * 1024 * 1024;
static constexpr uint64_t kMaximumScalarEvaluations = 4 * 1024 * 1024;

static bool isWithinConstantFoldBudget(mlir::RankedTensorType type) {
  if (!type || !type.hasStaticShape() || type.getNumElements() < 0 ||
      type.getNumElements() > kMaximumFoldedTensorElements)
    return false;
  mlir::Type elementType = type.getElementType();
  if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(elementType))
    return false;
  const uint64_t elementBytes = std::max<uint64_t>(
      1, llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8u));
  const uint64_t elements = static_cast<uint64_t>(type.getNumElements());
  return elements <= kMaximumFoldedTensorBytes / elementBytes;
}

static bool allStatic(llvm::ArrayRef<int64_t> values) {
  return llvm::all_of(values, [](int64_t value) {
    return !mlir::ShapedType::isDynamic(value);
  });
}

static int64_t getLinearIndex(llvm::ArrayRef<int64_t> shape,
                              llvm::ArrayRef<int64_t> indices) {
  int64_t linear = 0;
  for (auto [dim, index] : llvm::enumerate(indices))
    linear = linear * shape[dim] + index;
  return linear;
}

static void delinearizeIndex(int64_t linear, llvm::ArrayRef<int64_t> shape,
                             llvm::SmallVectorImpl<int64_t> &indices) {
  indices.assign(shape.size(), 0);
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    int64_t size = shape[dim];
    indices[dim] = linear % size;
    linear /= size;
  }
}

static mlir::DenseElementsAttr getDenseConstantAttr(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return {};
  return mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
}

static std::optional<mlir::Attribute>
getConstantTensorElement(mlir::Value value, llvm::ArrayRef<int64_t> indices);

static std::optional<int64_t> getConstantIntegerValue(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (!attr || !attr.getValue().isSignedIntN(64))
      return std::nullopt;
    return attr.getInt();
  }

  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return getConstantIntegerValue(cast.getIn());
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    std::optional<int64_t> input = getConstantIntegerValue(cast.getIn());
    if (!input || *input < 0)
      return std::nullopt;
    return input;
  }

  auto evaluateSignedBinary = [&](mlir::Value lhs, mlir::Value rhs,
                                  bool takeMaximum) -> std::optional<int64_t> {
    std::optional<int64_t> lhsValue = getConstantIntegerValue(lhs);
    std::optional<int64_t> rhsValue = getConstantIntegerValue(rhs);
    if (!lhsValue || !rhsValue)
      return std::nullopt;
    return takeMaximum ? std::max(*lhsValue, *rhsValue)
                       : std::min(*lhsValue, *rhsValue);
  };
  if (auto maximum = value.getDefiningOp<mlir::arith::MaxSIOp>())
    return evaluateSignedBinary(maximum.getLhs(), maximum.getRhs(), true);
  if (auto minimum = value.getDefiningOp<mlir::arith::MinSIOp>())
    return evaluateSignedBinary(minimum.getLhs(), minimum.getRhs(), false);

  if (auto extract = value.getDefiningOp<mlir::tensor::ExtractOp>()) {
    llvm::SmallVector<int64_t> indices;
    indices.reserve(extract.getIndices().size());
    for (mlir::Value index : extract.getIndices()) {
      std::optional<int64_t> constantIndex = getConstantIntegerValue(index);
      if (!constantIndex)
        return std::nullopt;
      indices.push_back(*constantIndex);
    }
    std::optional<mlir::Attribute> element =
        getConstantTensorElement(extract.getTensor(), indices);
    if (!element)
      return std::nullopt;
    auto integer = mlir::dyn_cast<mlir::IntegerAttr>(*element);
    if (!integer || !integer.getValue().isSignedIntN(64))
      return std::nullopt;
    return integer.getInt();
  }

  return std::nullopt;
}

static std::optional<int64_t> getConstantFoldResult(mlir::OpFoldResult value) {
  if (mlir::Attribute attribute = value.dyn_cast<mlir::Attribute>()) {
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
      return integer.getInt();
  }
  if (mlir::Value dynamic = value.dyn_cast<mlir::Value>())
    return getConstantIntegerValue(dynamic);
  return std::nullopt;
}

static bool areValidIndices(llvm::ArrayRef<int64_t> shape,
                            llvm::ArrayRef<int64_t> indices) {
  if (shape.size() != indices.size())
    return false;
  return llvm::all_of(llvm::zip(shape, indices), [](auto dimAndIndex) {
    auto [dim, index] = dimAndIndex;
    return dim >= 0 && index >= 0 && index < dim;
  });
}

static std::optional<mlir::Attribute>
getConstantTensorElement(mlir::Value value, llvm::ArrayRef<int64_t> indices) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() ||
      !areValidIndices(type.getShape(), indices))
    return std::nullopt;

  if (mlir::DenseElementsAttr attr = getDenseConstantAttr(value)) {
    int64_t linear = getLinearIndex(type.getShape(), indices);
    if (linear < 0 || linear >= type.getNumElements())
      return std::nullopt;
    auto values = attr.getValues<mlir::Attribute>();
    return *(values.begin() + linear);
  }

  if (auto slice = value.getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(slice.getSourceType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        sourceType.getRank() != type.getRank())
      return std::nullopt;

    llvm::SmallVector<int64_t> sourceIndices;
    sourceIndices.reserve(indices.size());
    for (auto [index, offset, stride] :
         llvm::zip(indices, slice.getMixedOffsets(), slice.getMixedStrides())) {
      std::optional<int64_t> constantOffset = getConstantFoldResult(offset);
      std::optional<int64_t> constantStride = getConstantFoldResult(stride);
      if (!constantOffset || !constantStride || *constantStride <= 0)
        return std::nullopt;
      sourceIndices.push_back(*constantOffset + index * *constantStride);
    }
    return getConstantTensorElement(slice.getSource(), sourceIndices);
  }

  mlir::Value reshapeSource;
  if (auto collapse = value.getDefiningOp<mlir::tensor::CollapseShapeOp>())
    reshapeSource = collapse.getSrc();
  else if (auto expand = value.getDefiningOp<mlir::tensor::ExpandShapeOp>())
    reshapeSource = expand.getSrc();
  if (reshapeSource) {
    auto sourceType =
        mlir::dyn_cast<mlir::RankedTensorType>(reshapeSource.getType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        sourceType.getElementType() != type.getElementType() ||
        sourceType.getNumElements() != type.getNumElements())
      return std::nullopt;
    int64_t linear = getLinearIndex(type.getShape(), indices);
    llvm::SmallVector<int64_t> sourceIndices;
    delinearizeIndex(linear, sourceType.getShape(), sourceIndices);
    return getConstantTensorElement(reshapeSource, sourceIndices);
  }

  return std::nullopt;
}

static mlir::LogicalResult
replaceWithDenseConstant(mlir::Operation *op, mlir::Value result,
                         mlir::DenseElementsAttr attr,
                         mlir::PatternRewriter &rewriter) {
  auto constant = rewriter.create<mlir::arith::ConstantOp>(
      op->getLoc(), result.getType(), attr);
  rewriter.replaceOp(op, constant.getResult());
  return mlir::success();
}

static mlir::LogicalResult
foldConstantTensorExtractSlice(mlir::tensor::ExtractSliceOp slice,
                               mlir::PatternRewriter &rewriter) {
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(slice.getSourceType());
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      sourceType.getRank() != resultType.getRank())
    return mlir::failure();
  if (!allStatic(slice.getStaticOffsets()) ||
      !allStatic(slice.getStaticSizes()) ||
      !allStatic(slice.getStaticStrides()))
    return mlir::failure();

  mlir::DenseElementsAttr sourceAttr = getDenseConstantAttr(slice.getSource());
  if (!sourceAttr)
    return mlir::failure();
  if (!isWithinConstantFoldBudget(resultType))
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> sourceValues;
  for (mlir::Attribute value : sourceAttr.getValues<mlir::Attribute>())
    sourceValues.push_back(value);

  llvm::SmallVector<mlir::Attribute> resultValues;
  resultValues.reserve(resultType.getNumElements());
  llvm::SmallVector<int64_t> resultIndices;
  llvm::SmallVector<int64_t> sourceIndices(sourceType.getRank(), 0);
  llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  llvm::ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  llvm::ArrayRef<int64_t> strides = slice.getStaticStrides();
  for (int64_t linear = 0; linear < resultType.getNumElements(); ++linear) {
    delinearizeIndex(linear, resultShape, resultIndices);
    for (auto [dim, index] : llvm::enumerate(resultIndices))
      sourceIndices[dim] = offsets[dim] + index * strides[dim];
    int64_t sourceLinear = getLinearIndex(sourceShape, sourceIndices);
    if (sourceLinear < 0 ||
        sourceLinear >= static_cast<int64_t>(sourceValues.size()))
      return mlir::failure();
    resultValues.push_back(sourceValues[sourceLinear]);
  }

  auto resultAttr = mlir::DenseElementsAttr::get(resultType, resultValues);
  return replaceWithDenseConstant(slice.getOperation(), slice.getResult(),
                                  resultAttr, rewriter);
}

static mlir::LogicalResult
foldConstantTensorReshape(mlir::Operation *op, mlir::Value source,
                          mlir::Value result, mlir::PatternRewriter &rewriter) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      sourceType.getElementType() != resultType.getElementType() ||
      sourceType.getNumElements() != resultType.getNumElements())
    return mlir::failure();
  if (!isWithinConstantFoldBudget(resultType))
    return mlir::failure();

  mlir::DenseElementsAttr sourceAttr = getDenseConstantAttr(source);
  if (!sourceAttr)
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> values;
  for (mlir::Attribute value : sourceAttr.getValues<mlir::Attribute>())
    values.push_back(value);
  auto resultAttr = mlir::DenseElementsAttr::get(resultType, values);
  return replaceWithDenseConstant(op, result, resultAttr, rewriter);
}

static mlir::LogicalResult
foldConstantTensorExtract(mlir::tensor::ExtractOp extract,
                          mlir::PatternRewriter &rewriter) {
  auto sourceType =
      mlir::dyn_cast<mlir::RankedTensorType>(extract.getTensor().getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      extract.getIndices().size() != static_cast<size_t>(sourceType.getRank()))
    return mlir::failure();

  llvm::SmallVector<int64_t> indices;
  indices.reserve(extract.getIndices().size());
  for (mlir::Value index : extract.getIndices()) {
    std::optional<int64_t> constantIndex = getConstantIntegerValue(index);
    if (!constantIndex)
      return mlir::failure();
    indices.push_back(*constantIndex);
  }

  std::optional<mlir::Attribute> element =
      getConstantTensorElement(extract.getTensor(), indices);
  if (!element)
    return mlir::failure();
  auto value = mlir::dyn_cast<mlir::TypedAttr>(*element);
  if (!value)
    return mlir::failure();

  auto constant =
      rewriter.create<mlir::arith::ConstantOp>(extract.getLoc(), value);
  rewriter.replaceOp(extract, constant.getResult());
  return mlir::success();
}

static std::optional<int64_t>
getConstantLinearIndex(mlir::AffineMap map, llvm::ArrayRef<int64_t> indices,
                       llvm::ArrayRef<int64_t> shape) {
  if (map.getNumSymbols() != 0 || map.getNumResults() != shape.size())
    return std::nullopt;
  if (shape.empty())
    return int64_t{0};

  llvm::SmallVector<int64_t> operandIndices;
  operandIndices.reserve(shape.size());
  for (mlir::AffineExpr expr : map.getResults()) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() >= indices.size())
      return std::nullopt;
    operandIndices.push_back(indices[dimExpr.getPosition()]);
  }
  return getLinearIndex(shape, operandIndices);
}

static std::optional<mlir::Attribute>
getDenseElementAt(mlir::DenseElementsAttr attr, mlir::AffineMap map,
                  llvm::ArrayRef<int64_t> resultIndices) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;

  std::optional<int64_t> linear =
      getConstantLinearIndex(map, resultIndices, type.getShape());
  if (!linear)
    return std::nullopt;

  llvm::SmallVector<mlir::Attribute> values;
  for (mlir::Attribute value : attr.getValues<mlir::Attribute>())
    values.push_back(value);
  if (*linear < 0 || *linear >= static_cast<int64_t>(values.size()))
    return std::nullopt;
  return values[*linear];
}

static std::optional<mlir::Attribute>
evaluateConstantScalarOp(mlir::Operation *op,
                         llvm::DenseMap<mlir::Value, mlir::Attribute> &values) {
  auto lookup = [&](mlir::Value value) -> std::optional<mlir::Attribute> {
    auto it = values.find(value);
    if (it == values.end())
      return std::nullopt;
    return it->second;
  };

  if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(op)) {
    std::optional<mlir::Attribute> lhs = lookup(add.getLhs());
    std::optional<mlir::Attribute> rhs = lookup(add.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    auto lhsAttr = mlir::dyn_cast<mlir::IntegerAttr>(*lhs);
    auto rhsAttr = mlir::dyn_cast<mlir::IntegerAttr>(*rhs);
    if (!lhsAttr || !rhsAttr || lhsAttr.getType() != rhsAttr.getType())
      return std::nullopt;
    return mlir::IntegerAttr::get(lhsAttr.getType(),
                                  lhsAttr.getValue() + rhsAttr.getValue());
  }

  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
    std::optional<mlir::Attribute> lhs = lookup(cmp.getLhs());
    std::optional<mlir::Attribute> rhs = lookup(cmp.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    auto lhsAttr = mlir::dyn_cast<mlir::IntegerAttr>(*lhs);
    auto rhsAttr = mlir::dyn_cast<mlir::IntegerAttr>(*rhs);
    if (!lhsAttr || !rhsAttr)
      return std::nullopt;

    llvm::APInt lhsValue = lhsAttr.getValue();
    llvm::APInt rhsValue = rhsAttr.getValue();
    bool result = false;
    switch (cmp.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      result = lhsValue == rhsValue;
      break;
    case mlir::arith::CmpIPredicate::ne:
      result = lhsValue != rhsValue;
      break;
    case mlir::arith::CmpIPredicate::slt:
      result = lhsValue.slt(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::sle:
      result = lhsValue.sle(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::sgt:
      result = lhsValue.sgt(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::sge:
      result = lhsValue.sge(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::ult:
      result = lhsValue.ult(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::ule:
      result = lhsValue.ule(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::ugt:
      result = lhsValue.ugt(rhsValue);
      break;
    case mlir::arith::CmpIPredicate::uge:
      result = lhsValue.uge(rhsValue);
      break;
    }
    return mlir::IntegerAttr::get(op->getResult(0).getType(), result ? 1 : 0);
  }

  return std::nullopt;
}

static mlir::LogicalResult
foldConstantLinalgGeneric(mlir::linalg::GenericOp generic,
                          mlir::PatternRewriter &rewriter) {
  if (generic->getNumResults() != 1 || generic.getNumDpsInits() != 1)
    return mlir::failure();
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](auto iteratorType) {
        return iteratorType == mlir::utils::IteratorType::parallel;
      }))
    return mlir::failure();

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return mlir::failure();
  if (!isWithinConstantFoldBudget(resultType))
    return mlir::failure();

  const uint64_t bodyOperations = static_cast<uint64_t>(
      std::distance(generic.getBody()->without_terminator().begin(),
                    generic.getBody()->without_terminator().end()));
  const uint64_t elements = static_cast<uint64_t>(resultType.getNumElements());
  if (bodyOperations != 0 &&
      elements > kMaximumScalarEvaluations / bodyOperations)
    return mlir::failure();

  llvm::SmallVector<mlir::AffineMap> indexingMaps =
      generic.getIndexingMapsArray();
  if (indexingMaps.size() !=
      generic.getNumDpsInputs() + generic.getNumDpsInits())
    return mlir::failure();

  llvm::SmallVector<std::optional<mlir::DenseElementsAttr>> operandAttrs;
  operandAttrs.reserve(generic.getNumDpsInputs() + generic.getNumDpsInits());
  for (mlir::Value input : generic.getDpsInputs()) {
    mlir::DenseElementsAttr attr = getDenseConstantAttr(input);
    if (!attr)
      return mlir::failure();
    operandAttrs.push_back(attr);
  }
  for (auto [index, init] : llvm::enumerate(generic.getDpsInits())) {
    mlir::DenseElementsAttr attr = getDenseConstantAttr(init);
    if (!attr && !generic.getBody()
                      ->getArgument(generic.getNumDpsInputs() + index)
                      .use_empty())
      return mlir::failure();
    if (attr)
      operandAttrs.push_back(attr);
    else
      operandAttrs.push_back(std::nullopt);
  }

  auto yield =
      mlir::dyn_cast<mlir::linalg::YieldOp>(generic.getBody()->getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> resultValues;
  resultValues.reserve(resultType.getNumElements());
  llvm::SmallVector<int64_t> resultIndices;
  for (int64_t linear = 0; linear < resultType.getNumElements(); ++linear) {
    delinearizeIndex(linear, resultType.getShape(), resultIndices);

    llvm::DenseMap<mlir::Value, mlir::Attribute> scalarValues;
    for (auto [index, attr] : llvm::enumerate(operandAttrs)) {
      if (!attr)
        continue;
      std::optional<mlir::Attribute> value =
          getDenseElementAt(*attr, indexingMaps[index], resultIndices);
      if (!value)
        return mlir::failure();
      scalarValues[generic.getBody()->getArgument(index)] = *value;
    }

    for (mlir::Operation &op : generic.getBody()->without_terminator()) {
      if (op.getNumResults() != 1)
        return mlir::failure();
      std::optional<mlir::Attribute> value =
          evaluateConstantScalarOp(&op, scalarValues);
      if (!value)
        return mlir::failure();
      scalarValues[op.getResult(0)] = *value;
    }

    auto it = scalarValues.find(yield.getValues().front());
    if (it == scalarValues.end())
      return mlir::failure();
    resultValues.push_back(it->second);
  }

  auto resultAttr = mlir::DenseElementsAttr::get(resultType, resultValues);
  return replaceWithDenseConstant(generic.getOperation(), generic.getResult(0),
                                  resultAttr, rewriter);
}

struct ConstantExtractSlicePattern final
    : mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    return foldConstantTensorExtractSlice(op, rewriter);
  }
};

struct ConstantCollapseShapePattern final
    : mlir::OpRewritePattern<mlir::tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::CollapseShapeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    return foldConstantTensorReshape(op, op.getSrc(), op.getResult(), rewriter);
  }
};

struct ConstantExpandShapePattern final
    : mlir::OpRewritePattern<mlir::tensor::ExpandShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExpandShapeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    return foldConstantTensorReshape(op, op.getSrc(), op.getResult(), rewriter);
  }
};

struct ConstantExtractPattern final
    : mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractOp op,
                  mlir::PatternRewriter &rewriter) const final {
    return foldConstantTensorExtract(op, rewriter);
  }
};

struct ConstantLinalgGenericPattern final
    : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const final {
    return foldConstantLinalgGeneric(op, rewriter);
  }
};

mlir::LogicalResult foldConstantTensorOps(mlir::Operation *root) {
  if (!root)
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *, 32> candidates;
  root->walk([&](mlir::Operation *op) {
    if (mlir::isa<mlir::tensor::ExtractSliceOp, mlir::tensor::CollapseShapeOp,
                  mlir::tensor::ExpandShapeOp, mlir::tensor::ExtractOp,
                  mlir::linalg::GenericOp>(op))
      candidates.push_back(op);
  });
  if (candidates.empty())
    return mlir::success();

  mlir::RewritePatternSet patterns(root->getContext());
  patterns.add<ConstantExtractSlicePattern, ConstantCollapseShapePattern,
               ConstantExpandShapePattern, ConstantExtractPattern,
               ConstantLinalgGenericPattern>(root->getContext());
  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  mlir::GreedyRewriteConfig config;
  config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;
  return mlir::applyOpPatternsAndFold(candidates, frozenPatterns, config);
}

} // namespace wafer::stablehlo_normalization
#endif

namespace wafer {
#define GEN_PASS_DEF_FOLDSTATICTENSOROPSPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

struct FoldStaticTensorOpsPass
    : public impl::FoldStaticTensorOpsPassBase<FoldStaticTensorOpsPass> {
  using impl::FoldStaticTensorOpsPassBase<
      FoldStaticTensorOpsPass>::FoldStaticTensorOpsPassBase;

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    mlir::ModuleOp module = getOperation();
    mlir::OwningOpRef<mlir::ModuleOp> transaction =
        mlir::cast<mlir::ModuleOp>(module->clone());
    if (mlir::failed(
            stablehlo_normalization::foldConstantTensorOps(*transaction)) ||
        mlir::failed(mlir::verify(*transaction))) {
      signalPassFailure();
      return;
    }
    module.getBodyRegion().takeBody(transaction->getBodyRegion());
#endif
  }
};

} // namespace
} // namespace wafer
