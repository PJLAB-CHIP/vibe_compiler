//===- GenericLowering.cpp - Generic linalg lowering ----------------===//

#include "Internal.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

std::optional<unsigned> TileRegionBodyEmitter::getPassthroughInputIndex(
    mlir::linalg::GenericOp generic) {
  if (!generic.getBody()->without_terminator().empty())
    return std::nullopt;
  auto yield =
      mlir::dyn_cast<mlir::linalg::YieldOp>(generic.getBody()->getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return std::nullopt;

  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(yield.getValues()[0]);
  if (!blockArg)
    return std::nullopt;
  unsigned argNumber = blockArg.getArgNumber();
  if (argNumber >= generic.getNumDpsInputs())
    return std::nullopt;
  return argNumber;
}

bool TileRegionBodyEmitter::isIdentityMap(mlir::AffineMap map,
                                          int64_t rank) const {
  return map.getNumDims() == static_cast<unsigned>(rank) &&
         map.getNumSymbols() == 0 && map.isIdentity();
}

mlir::AffineMap
TileRegionBodyEmitter::getIdentityMap(mlir::RankedTensorType tensorType) const {
  return mlir::AffineMap::getMultiDimIdentityMap(tensorType.getRank(),
                                                 tensorType.getContext());
}

bool TileRegionBodyEmitter::isScalarSplatValue(mlir::Attribute attr,
                                               double expected) const {
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr))
    return floatAttr.getValueAsDouble() == expected;
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getValue().getSExtValue() == static_cast<int64_t>(expected);
  if (auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!elements.isSplat())
      return false;
    if (auto floatType =
            mlir::dyn_cast<mlir::FloatType>(elements.getElementType()))
      return elements.getSplatValue<mlir::APFloat>().convertToDouble() ==
             expected;
    if (auto intType =
            mlir::dyn_cast<mlir::IntegerType>(elements.getElementType()))
      return elements.getSplatValue<mlir::APInt>().getSExtValue() ==
             static_cast<int64_t>(expected);
  }
  return false;
}

mlir::Attribute TileRegionBodyEmitter::getBlockArgumentConstantAttr(
    mlir::linalg::GenericOp generic, mlir::BlockArgument arg) const {
  if (arg.getOwner() != generic.getBody() ||
      arg.getArgNumber() >= generic.getNumDpsInputs())
    return {};
  mlir::Value input = generic.getDpsInputs()[arg.getArgNumber()];
  if (auto constant = input.getDefiningOp<mlir::arith::ConstantOp>())
    return constant.getValue();
  if (auto it = tensorAttrs.find(input); it != tensorAttrs.end())
    return it->second;
  return {};
}

bool TileRegionBodyEmitter::isScalarLikeConstant(
    mlir::linalg::GenericOp generic, mlir::Value value, double expected) const {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    return isScalarSplatValue(constant.getValue(), expected);
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (mlir::Attribute attr = getBlockArgumentConstantAttr(generic, arg))
      return isScalarSplatValue(attr, expected);
  }
  return false;
}

mlir::FailureOr<ElementwiseExprValue>
TileRegionBodyEmitter::getElementwiseExprValue(
    llvm::DenseMap<mlir::Value, ElementwiseExprValue> &values,
    mlir::Value value) {
  auto it = values.find(value);
  if (it == values.end())
    return failElementwiseExprValue(
        "unsupported linalg.generic scalar expression");
  return it->second;
}

mlir::FailureOr<mlir::Value>
TileRegionBodyEmitter::materializeElementwiseExprOperand(
    ElementwiseExprValue exprValue, mlir::RankedTensorType resultTensorType,
    mlir::Location loc, mlir::OpBuilder &builder) {
  auto sourceType =
      mlir::dyn_cast<mlir::MemRefType>(exprValue.buffer.getType());
  if (!sourceType)
    return failValue("elementwise expression operand is not an SPM memref");
  auto operandTensorType = mlir::RankedTensorType::get(
      resultTensorType.getShape(), sourceType.getElementType());
  auto operandType = makeSPMMemRefType(operandTensorType, MemLayout::Tensor);
  if (sourceType == operandType &&
      isIdentityMap(exprValue.indexingMap, resultTensorType.getRank()))
    return exprValue.buffer;

  auto sourceTensorType = mlir::RankedTensorType::get(
      sourceType.getShape(), sourceType.getElementType());
  if (!exprValue.indexingMap.isProjectedPermutation())
    return failValue("elementwise expression indexing map must be a "
                     "projected permutation");

  if (sourceTensorType.getRank() == resultTensorType.getRank()) {
    llvm::SmallVector<int64_t, 4> permutation(resultTensorType.getRank(), -1);
    for (auto [sourceDim, expr] :
         llvm::enumerate(exprValue.indexingMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr)
        return failValue("unsupported elementwise transpose map");
      unsigned resultDim = dimExpr.getPosition();
      if (resultDim >= permutation.size())
        return failValue("elementwise transpose map dim out of range");
      permutation[resultDim] = static_cast<int64_t>(sourceDim);
    }
    if (llvm::any_of(permutation, [](int64_t dim) { return dim < 0; }))
      return failValue("elementwise transpose map is incomplete");
    if (sourceType == operandType &&
        llvm::all_of(llvm::enumerate(permutation), [](auto indexed) {
          return static_cast<int64_t>(indexed.index()) == indexed.value();
        }))
      return exprValue.buffer;
    return builder
        .create<MoveTransposeOp>(
            loc, operandType, exprValue.buffer,
            mlir::DenseI64ArrayAttr::get(builder.getContext(), permutation))
        .getResult();
  }

  if (sourceTensorType.getRank() < resultTensorType.getRank()) {
    llvm::SmallVector<int64_t, 4> dimensions;
    for (mlir::AffineExpr expr : exprValue.indexingMap.getResults()) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr)
        return failValue("unsupported elementwise broadcast map");
      dimensions.push_back(dimExpr.getPosition());
    }
    return builder
        .create<MoveBroadcastOp>(
            loc, operandType, exprValue.buffer,
            mlir::DenseI64ArrayAttr::get(builder.getContext(), dimensions))
        .getResult();
  }

  return failValue("unsupported elementwise expression rank relation");
}

mlir::FailureOr<ElementwiseExprValue>
TileRegionBodyEmitter::createElementwiseFillExprValue(
    mlir::Location loc, mlir::Value scalar,
    mlir::RankedTensorType resultTensorType, mlir::OpBuilder &builder) {
  if (!isScalarType(scalar.getType()))
    return failElementwiseExprValue(
        "elementwise scalar splat source must be scalar typed");
  auto splatTensorType = mlir::RankedTensorType::get(
      resultTensorType.getShape(), scalar.getType());
  auto alloc = builder.create<mlir::memref::AllocOp>(
      loc, makeSPMMemRefType(splatTensorType, MemLayout::Tensor));
  recordScratchAllocation(alloc);
  auto fill = builder.create<ComputeFillOp>(loc, alloc.getResult(), scalar,
                                            /*fill_domain=*/FillDomainAttr{});
  recordStructuredComputeOperation(fill);
  return ElementwiseExprValue{alloc.getResult(),
                              getIdentityMap(resultTensorType)};
}

mlir::FailureOr<ElementwiseExprValue>
TileRegionBodyEmitter::createElementwiseOpExprValue(
    mlir::Location loc, ComputeElementwiseKind kind,
    llvm::ArrayRef<ElementwiseExprValue> operands,
    mlir::RankedTensorType resultTensorType, mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 2> materializedInputs;
  for (ElementwiseExprValue operand : operands) {
    mlir::FailureOr<mlir::Value> materialized =
        materializeElementwiseExprOperand(operand, resultTensorType, loc,
                                          builder);
    if (mlir::failed(materialized))
      return mlir::failure();
    materializedInputs.push_back(*materialized);
  }

  auto kindAttr = ComputeElementwiseKindAttr::get(builder.getContext(), kind);
  auto elementwise = builder.create<ComputeElementwiseOp>(
      loc, makeSPMMemRefType(resultTensorType, MemLayout::Tensor), kindAttr,
      materializedInputs, mlir::ArrayAttr{});
  recordStructuredComputeOperation(elementwise);
  return ElementwiseExprValue{elementwise.getResult(),
                              getIdentityMap(resultTensorType)};
}

mlir::LogicalResult TileRegionBodyEmitter::convertElementwiseScalarOp(
    mlir::linalg::GenericOp generic, mlir::Operation *op,
    llvm::DenseMap<mlir::Value, ElementwiseExprValue> &values,
    mlir::RankedTensorType resultTensorType, mlir::OpBuilder &builder) {
  // Scalar operations are nested implementation details of one structured
  // generic.  Preserve both locations on every Tile operation emitted for
  // that scalar expression: the scalar location remains useful diagnostics,
  // while the generic location keeps diagnostic context for the containing
  // structured operation. Executable producer/consumer relations are carried
  // separately by typed current-IR mappings.
  mlir::Location loc = mlir::FusedLoc::get(generic.getContext(),
                                           {generic.getLoc(), op->getLoc()});
  auto lookup = [&](mlir::Value value) {
    auto it = values.find(value);
    if (it != values.end())
      return mlir::FailureOr<ElementwiseExprValue>(it->second);

    if (isScalarType(value.getType())) {
      mlir::FailureOr<mlir::Value> scalar = getScalarValue(value, builder);
      if (mlir::failed(scalar))
        return mlir::FailureOr<ElementwiseExprValue>(mlir::failure());
      mlir::FailureOr<ElementwiseExprValue> splat =
          createElementwiseFillExprValue(loc, *scalar, resultTensorType,
                                         builder);
      if (mlir::failed(splat))
        return splat;
      values[value] = *splat;
      return splat;
    }

    return failElementwiseExprValue(
        "unsupported linalg.generic scalar expression");
  };
  auto createUnary = [&](mlir::Value input,
                         ComputeElementwiseKind kind) -> mlir::LogicalResult {
    mlir::FailureOr<ElementwiseExprValue> operand = lookup(input);
    if (mlir::failed(operand))
      return mlir::failure();
    mlir::FailureOr<ElementwiseExprValue> result = createElementwiseOpExprValue(
        loc, kind, {*operand}, resultTensorType, builder);
    if (mlir::failed(result))
      return mlir::failure();
    values[op->getResult(0)] = *result;
    return mlir::success();
  };
  auto createBinary = [&](mlir::Value lhs, mlir::Value rhs,
                          ComputeElementwiseKind kind) -> mlir::LogicalResult {
    mlir::FailureOr<ElementwiseExprValue> lhsValue = lookup(lhs);
    mlir::FailureOr<ElementwiseExprValue> rhsValue = lookup(rhs);
    if (mlir::failed(lhsValue) || mlir::failed(rhsValue))
      return mlir::failure();
    mlir::FailureOr<ElementwiseExprValue> result = createElementwiseOpExprValue(
        loc, kind, {*lhsValue, *rhsValue}, resultTensorType, builder);
    if (mlir::failed(result))
      return mlir::failure();
    values[op->getResult(0)] = *result;
    return mlir::success();
  };
  auto createTernary = [&](mlir::Value first, mlir::Value second,
                           mlir::Value third,
                           ComputeElementwiseKind kind) -> mlir::LogicalResult {
    mlir::FailureOr<ElementwiseExprValue> firstValue = lookup(first);
    mlir::FailureOr<ElementwiseExprValue> secondValue = lookup(second);
    mlir::FailureOr<ElementwiseExprValue> thirdValue = lookup(third);
    if (mlir::failed(firstValue) || mlir::failed(secondValue) ||
        mlir::failed(thirdValue))
      return mlir::failure();
    mlir::FailureOr<ElementwiseExprValue> result = createElementwiseOpExprValue(
        loc, kind, {*firstValue, *secondValue, *thirdValue}, resultTensorType,
        builder);
    if (mlir::failed(result))
      return mlir::failure();
    values[op->getResult(0)] = *result;
    return mlir::success();
  };
  auto createConvert = [&](mlir::Value input) -> mlir::LogicalResult {
    if (op->getNumResults() != 1 ||
        !mlir::isa<mlir::FloatType>(op->getResult(0).getType()))
      return fail("elementwise floating conversion must have one floating "
                  "result");
    mlir::FailureOr<ElementwiseExprValue> operand = lookup(input);
    if (mlir::failed(operand))
      return mlir::failure();
    auto convertedTensorType = mlir::RankedTensorType::get(
        resultTensorType.getShape(), op->getResult(0).getType());
    mlir::FailureOr<mlir::Value> materialized =
        materializeElementwiseExprOperand(*operand, convertedTensorType, loc,
                                          builder);
    if (mlir::failed(materialized))
      return mlir::failure();
    auto convert = builder.create<ComputeConvertOp>(
        loc, makeSPMMemRefType(convertedTensorType, MemLayout::Tensor),
        *materialized);
    recordStructuredComputeOperation(convert);
    values[op->getResult(0)] = ElementwiseExprValue{
        convert.getResult(), getIdentityMap(convertedTensorType)};
    return mlir::success();
  };

  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    if (constant->getNumResults() != 1 || !isScalarType(constant.getType()))
      return fail("unsupported linalg.generic constant expression");
    mlir::Operation *cloned = builder.clone(*constant.getOperation());
    cloned->setLoc(loc);
    mlir::FailureOr<ElementwiseExprValue> value =
        createElementwiseFillExprValue(loc, cloned->getResult(0),
                                       resultTensorType, builder);
    if (mlir::failed(value))
      return mlir::failure();
    values[constant.getResult()] = *value;
    return mlir::success();
  }

  if (auto fromElements = mlir::dyn_cast<mlir::tensor::FromElementsOp>(op)) {
    if (fromElements.getElements().size() != 1 ||
        fromElements->getNumResults() != 1)
      return fail("unsupported tensor.from_elements in elementwise body");
    mlir::FailureOr<ElementwiseExprValue> value =
        lookup(fromElements.getElements().front());
    if (mlir::failed(value))
      return mlir::failure();
    values[fromElements.getResult()] = *value;
    return mlir::success();
  }

  if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(op)) {
    if (!extract.getIndices().empty() || extract->getNumResults() != 1)
      return fail("unsupported tensor.extract in elementwise body");
    mlir::FailureOr<ElementwiseExprValue> value = lookup(extract.getTensor());
    if (mlir::failed(value))
      return mlir::failure();
    values[extract.getResult()] = *value;
    return mlir::success();
  }

  if (auto addf = mlir::dyn_cast<mlir::arith::AddFOp>(op))
    return createBinary(addf.getLhs(), addf.getRhs(),
                        ComputeElementwiseKind::Add);
  if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(op))
    return createBinary(addi.getLhs(), addi.getRhs(),
                        ComputeElementwiseKind::Add);
  if (auto subf = mlir::dyn_cast<mlir::arith::SubFOp>(op))
    return createBinary(subf.getLhs(), subf.getRhs(),
                        ComputeElementwiseKind::Sub);
  if (auto subi = mlir::dyn_cast<mlir::arith::SubIOp>(op))
    return createBinary(subi.getLhs(), subi.getRhs(),
                        ComputeElementwiseKind::Sub);
  if (auto mulf = mlir::dyn_cast<mlir::arith::MulFOp>(op))
    return createBinary(mulf.getLhs(), mulf.getRhs(),
                        ComputeElementwiseKind::Mul);
  if (auto muli = mlir::dyn_cast<mlir::arith::MulIOp>(op))
    return createBinary(muli.getLhs(), muli.getRhs(),
                        ComputeElementwiseKind::Mul);
  if (auto divf = mlir::dyn_cast<mlir::arith::DivFOp>(op)) {
    if (activeImplementation == StructuredComputeImplementation::Reciprocal) {
      if (!isScalarLikeConstant(generic, divf.getLhs(), 1.0))
        return fail("selected reciprocal implementation requires exact 1/x");
      return createUnary(divf.getRhs(), ComputeElementwiseKind::Recip);
    }
    return createBinary(divf.getLhs(), divf.getRhs(),
                        ComputeElementwiseKind::Div);
  }
  if (auto divsi = mlir::dyn_cast<mlir::arith::DivSIOp>(op))
    return createBinary(divsi.getLhs(), divsi.getRhs(),
                        ComputeElementwiseKind::Div);
  if (mlir::isa<mlir::arith::DivUIOp>(op))
    return fail("unsigned integer division cannot be represented by the "
                "current target elementwise kind");
  if (auto maxf = mlir::dyn_cast<mlir::arith::MaximumFOp>(op))
    return createBinary(maxf.getLhs(), maxf.getRhs(),
                        ComputeElementwiseKind::Max);
  if (mlir::isa<mlir::arith::MaxNumFOp>(op))
    return fail("arith.maxnumf NaN semantics cannot be represented by the "
                "current target elementwise kind");
  if (auto maxsi = mlir::dyn_cast<mlir::arith::MaxSIOp>(op))
    return createBinary(maxsi.getLhs(), maxsi.getRhs(),
                        ComputeElementwiseKind::Max);
  if (mlir::isa<mlir::arith::MaxUIOp>(op))
    return fail("unsigned integer maximum cannot be represented by the "
                "current target elementwise kind");
  if (auto minf = mlir::dyn_cast<mlir::arith::MinimumFOp>(op))
    return createBinary(minf.getLhs(), minf.getRhs(),
                        ComputeElementwiseKind::Min);
  if (mlir::isa<mlir::arith::MinNumFOp>(op))
    return fail("arith.minnumf NaN semantics cannot be represented by the "
                "current target elementwise kind");
  if (auto minsi = mlir::dyn_cast<mlir::arith::MinSIOp>(op))
    return createBinary(minsi.getLhs(), minsi.getRhs(),
                        ComputeElementwiseKind::Min);
  if (mlir::isa<mlir::arith::MinUIOp>(op))
    return fail("unsigned integer minimum cannot be represented by the "
                "current target elementwise kind");
  if (auto negf = mlir::dyn_cast<mlir::arith::NegFOp>(op))
    return createUnary(negf.getOperand(), ComputeElementwiseKind::Neg);
  if (auto extf = mlir::dyn_cast<mlir::arith::ExtFOp>(op))
    return createConvert(extf.getIn());
  if (auto truncf = mlir::dyn_cast<mlir::arith::TruncFOp>(op))
    return createConvert(truncf.getIn());
  if (auto exp = mlir::dyn_cast<mlir::math::ExpOp>(op))
    return createUnary(exp.getOperand(), ComputeElementwiseKind::Exp);
  if (auto log = mlir::dyn_cast<mlir::math::LogOp>(op))
    return createUnary(log.getOperand(), ComputeElementwiseKind::Ln);
  if (auto sqrt = mlir::dyn_cast<mlir::math::SqrtOp>(op))
    return createUnary(sqrt.getOperand(), ComputeElementwiseKind::Sqrt);
  if (auto rsqrt = mlir::dyn_cast<mlir::math::RsqrtOp>(op))
    return createUnary(rsqrt.getOperand(), ComputeElementwiseKind::Rsqrt);
  if (auto tanh = mlir::dyn_cast<mlir::math::TanhOp>(op))
    return createUnary(tanh.getOperand(), ComputeElementwiseKind::Tanh);
  if (auto cmpf = mlir::dyn_cast<mlir::arith::CmpFOp>(op)) {
    std::optional<ComputeElementwiseKind> kind =
        inferCompareKind(cmpf.getPredicate());
    if (!kind)
      return mlir::failure();
    return createBinary(cmpf.getLhs(), cmpf.getRhs(), *kind);
  }
  if (auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
    std::optional<ComputeElementwiseKind> kind =
        inferCompareKind(cmpi.getPredicate());
    if (!kind)
      return mlir::failure();
    return createBinary(cmpi.getLhs(), cmpi.getRhs(), *kind);
  }
  if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op))
    return createTernary(select.getCondition(), select.getTrueValue(),
                         select.getFalseValue(),
                         ComputeElementwiseKind::Select);
  if (auto powf = mlir::dyn_cast<mlir::math::PowFOp>(op)) {
    if (!isScalarLikeConstant(generic, powf.getRhs(), 2.0))
      return fail("only powf with exponent 2 is supported in elementwise "
                  "body");
    return createBinary(powf.getLhs(), powf.getLhs(),
                        ComputeElementwiseKind::Mul);
  }

  return fail("unsupported linalg.generic body op " +
              op->getName().getStringRef().str());
}

mlir::LogicalResult TileRegionBodyEmitter::convertElementwiseGenericExpression(
    mlir::linalg::GenericOp generic, mlir::OpBuilder &builder) {
  if (generic->getNumResults() == 0 ||
      generic.getNumDpsInits() != generic->getNumResults())
    return fail("elementwise generic requires one destination per result");
  llvm::SmallVector<mlir::RankedTensorType, 2> resultTensorTypes;
  resultTensorTypes.reserve(generic->getNumResults());
  for (mlir::Type type : generic->getResultTypes()) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensor)
      return fail("generic result is not a ranked tensor");
    resultTensorTypes.push_back(tensor);
  }
  mlir::RankedTensorType expressionTensorType = resultTensorTypes.front();
  if (llvm::any_of(llvm::drop_begin(resultTensorTypes),
                   [&](mlir::RankedTensorType type) {
                     return type.getShape() != expressionTensorType.getShape();
                   }))
    return fail("multi-result elementwise generic requires one result shape");

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      generic.getIndexingMapsArray();
  if (indexingMaps.size() !=
      generic.getNumDpsInputs() + generic.getNumDpsInits())
    return fail("elementwise generic indexing map count mismatch");
  for (auto [resultNumber, resultType] : llvm::enumerate(resultTensorTypes)) {
    mlir::AffineMap resultMap =
        indexingMaps[generic.getNumDpsInputs() + resultNumber];
    if (!isIdentityMap(resultMap, resultType.getRank()))
      return fail("elementwise generic result map must be identity");
  }

  llvm::DenseMap<mlir::Value, ElementwiseExprValue> values;
  mlir::Block &body = *generic.getBody();
  unsigned bodyArgIndex = 0;
  for (mlir::Value input : generic.getDpsInputs()) {
    if (isScalarType(input.getType())) {
      if (indexingMaps[bodyArgIndex].getNumResults() != 0)
        return fail("elementwise scalar input requires a scalar indexing map");
      mlir::FailureOr<mlir::Value> scalar = getScalarValue(input, builder);
      if (mlir::failed(scalar))
        return mlir::failure();
      mlir::FailureOr<ElementwiseExprValue> splat =
          createElementwiseFillExprValue(generic.getLoc(), *scalar,
                                         expressionTensorType, builder);
      if (mlir::failed(splat))
        return mlir::failure();
      values[body.getArgument(bodyArgIndex)] = *splat;
    } else {
      mlir::FailureOr<mlir::Value> buffer =
          getOrMaterializeStructuredInput(input, MemLayout::Tensor, builder);
      if (mlir::failed(buffer))
        return mlir::failure();
      values[body.getArgument(bodyArgIndex)] =
          ElementwiseExprValue{*buffer, indexingMaps[bodyArgIndex]};
    }
    ++bodyArgIndex;
  }
  for (mlir::Value init : generic.getDpsInits()) {
    mlir::BlockArgument bodyArg = body.getArgument(bodyArgIndex);
    if (bodyArg.use_empty()) {
      ++bodyArgIndex;
      continue;
    }
    mlir::FailureOr<mlir::Value> buffer =
        getOrMaterialize(init, MemLayout::Tensor, builder);
    if (mlir::failed(buffer))
      return mlir::failure();
    values[bodyArg] = ElementwiseExprValue{*buffer, indexingMaps[bodyArgIndex]};
    ++bodyArgIndex;
  }

  for (mlir::Operation &op : body.without_terminator()) {
    if (mlir::failed(convertElementwiseScalarOp(generic, &op, values,
                                                expressionTensorType, builder)))
      return mlir::failure();
  }

  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != generic->getNumResults())
    return fail("unsupported linalg.generic yield");
  for (auto [resultNumber, yieldedValue, resultType] :
       llvm::enumerate(yield.getValues(), resultTensorTypes)) {
    mlir::FailureOr<ElementwiseExprValue> yielded =
        getElementwiseExprValue(values, yieldedValue);
    if (mlir::failed(yielded))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> result = materializeElementwiseExprOperand(
        *yielded, resultType, generic.getLoc(), builder);
    if (mlir::failed(result))
      return mlir::failure();
    record(generic->getResult(resultNumber), MemLayout::Tensor, *result);
  }
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertPassthroughGeneric(
    mlir::linalg::GenericOp generic, mlir::OpBuilder &builder) {
  if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
    return fail("unsupported passthrough generic arity");
  std::optional<unsigned> inputIndex = getPassthroughInputIndex(generic);
  if (!inputIndex)
    return fail("unsupported linalg.generic passthrough body");

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      generic.getIndexingMapsArray();
  if (indexingMaps.size() != generic.getNumDpsInputs() + 1)
    return fail("passthrough generic indexing map count mismatch");

  mlir::Value inputValue = generic.getDpsInputs()[*inputIndex];
  auto inputTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(inputValue.getType());
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!inputTensorType || !resultTensorType)
    return fail("passthrough generic operands/results must be ranked tensors");

  mlir::AffineMap inputMap = indexingMaps[*inputIndex];
  mlir::AffineMap resultMap = indexingMaps.back();
  if (!isIdentityMap(resultMap, resultTensorType.getRank()))
    return fail("passthrough generic result map must be identity");

  // Preserve an exact external identity as an SSA buffer relation. This is
  // especially important for the temporary output anchor wrapped around a
  // tensor.insert_slice assembly: the assembly may already have written the
  // complete result to DDR, so loading it into SPM merely to copy it back
  // would contradict the selected residency proof.
  if (inputTensorType == resultTensorType &&
      isIdentityMap(inputMap, resultTensorType.getRank())) {
    if (auto external = externalBuffers.find(inputValue);
        external != externalBuffers.end()) {
      mlir::Value result = generic->getResult(0);
      mlir::Value externalBuffer = external->second;
      externalBuffers[result] = externalBuffer;
      if (writableExternalBuffers.contains(inputValue))
        writableExternalBuffers.insert(result);
      if (auto output = externalOutputIndices.find(inputValue);
          output != externalOutputIndices.end())
        externalOutputIndices[result] = output->second;
      if (mlir::Value base = directYieldBuffers.lookup(inputValue))
        directYieldBuffers[result] = base;
      return mlir::success();
    }
  }

  mlir::FailureOr<mlir::Value> source =
      getOrMaterializeStructuredInput(inputValue, MemLayout::Tensor, builder);
  if (mlir::failed(source))
    return mlir::failure();

  mlir::MemRefType resultType =
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  mlir::Value result;
  if (inputTensorType == resultTensorType &&
      isIdentityMap(inputMap, resultTensorType.getRank())) {
    result = builder
                 .create<MoveCopyOp>(generic.getLoc(), resultType, *source,
                                     CardDDRResourceAttr{})
                 .getResult();
  } else if (inputTensorType.getRank() == resultTensorType.getRank()) {
    llvm::SmallVector<int64_t, 4> permutation(resultTensorType.getRank(), -1);
    for (auto [sourceDim, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr)
        return fail("unsupported passthrough transpose map");
      unsigned resultDim = dimExpr.getPosition();
      if (resultDim >= permutation.size())
        return fail("passthrough transpose map dim out of range");
      permutation[resultDim] = static_cast<int64_t>(sourceDim);
    }
    if (llvm::any_of(permutation, [](int64_t dim) { return dim < 0; }))
      return fail("passthrough transpose map is incomplete");
    result =
        builder
            .create<MoveTransposeOp>(
                generic.getLoc(), resultType, *source,
                mlir::DenseI64ArrayAttr::get(generic.getContext(), permutation))
            .getResult();
  } else if (inputTensorType.getRank() < resultTensorType.getRank()) {
    llvm::SmallVector<int64_t, 4> dimensions;
    for (mlir::AffineExpr expr : inputMap.getResults()) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr)
        return fail("unsupported passthrough broadcast map");
      dimensions.push_back(dimExpr.getPosition());
    }
    result = builder
                 .create<MoveBroadcastOp>(generic.getLoc(), resultType, *source,
                                          mlir::DenseI64ArrayAttr::get(
                                              generic.getContext(), dimensions))
                 .getResult();
  } else {
    return fail("unsupported passthrough movement rank relation");
  }

  record(generic->getResult(0), MemLayout::Tensor, result);
  return mlir::success();
}

mlir::Value TileRegionBodyEmitter::createZeroScalar(mlir::Location loc,
                                                    mlir::Type type,
                                                    mlir::OpBuilder &builder) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    auto zero = mlir::FloatAttr::get(
        floatType, llvm::APFloat::getZero(floatType.getFloatSemantics()));
    return builder.create<mlir::arith::ConstantOp>(loc, zero).getResult();
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    auto zero = mlir::IntegerAttr::get(intType, 0);
    return builder.create<mlir::arith::ConstantOp>(loc, zero).getResult();
  }
  if (mlir::isa<mlir::IndexType>(type))
    return builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
  return {};
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>>
TileRegionBodyEmitter::matchTwoWayConcatGeneric(mlir::linalg::GenericOp generic,
                                                int64_t &concatAxis) {
  concatAxis = -1;
  if (generic.getNumDpsInputs() != 0 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1)
    return mlir::failure();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!resultTensorType || !resultTensorType.hasStaticShape())
    return mlir::failure();
  int64_t rank = resultTensorType.getRank();

  llvm::SmallVector<mlir::AffineMap, 1> maps = generic.getIndexingMapsArray();
  if (maps.size() != 1 || !isIdentityMap(maps.front(), rank))
    return mlir::failure();
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      generic.getIteratorTypesArray();
  if (iteratorTypes.size() != static_cast<size_t>(rank) ||
      !llvm::all_of(iteratorTypes, [](mlir::utils::IteratorType type) {
        return type == mlir::utils::IteratorType::parallel;
      }))
    return mlir::failure();

  auto yield =
      mlir::dyn_cast<mlir::linalg::YieldOp>(generic.getBody()->getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return mlir::failure();
  auto ifOp = yield.getValues().front().getDefiningOp<mlir::scf::IfOp>();
  if (!ifOp || ifOp->getNumResults() != 1 || !ifOp.thenBlock() ||
      !ifOp.elseBlock() || ifOp->getBlock() != generic.getBody())
    return mlir::failure();

  auto thenYield =
      mlir::dyn_cast<mlir::scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYield =
      mlir::dyn_cast<mlir::scf::YieldOp>(ifOp.elseBlock()->getTerminator());
  if (!thenYield || !elseYield || thenYield.getResults().size() != 1 ||
      elseYield.getResults().size() != 1)
    return mlir::failure();
  auto firstExtract =
      thenYield.getResults().front().getDefiningOp<mlir::tensor::ExtractOp>();
  auto secondExtract =
      elseYield.getResults().front().getDefiningOp<mlir::tensor::ExtractOp>();
  if (!firstExtract || !secondExtract ||
      firstExtract->getBlock() != ifOp.thenBlock() ||
      secondExtract->getBlock() != ifOp.elseBlock())
    return mlir::failure();

  auto firstType = mlir::dyn_cast<mlir::RankedTensorType>(
      firstExtract.getTensor().getType());
  auto secondType = mlir::dyn_cast<mlir::RankedTensorType>(
      secondExtract.getTensor().getType());
  if (!firstType || !secondType || !firstType.hasStaticShape() ||
      !secondType.hasStaticShape() || firstType.getRank() != rank ||
      secondType.getRank() != rank ||
      firstType.getElementType() != resultTensorType.getElementType() ||
      secondType.getElementType() != resultTensorType.getElementType())
    return mlir::failure();

  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t first = firstType.getDimSize(dim);
    int64_t second = secondType.getDimSize(dim);
    int64_t result = resultTensorType.getDimSize(dim);
    if (first == result && second == result)
      continue;
    if (first + second == result && concatAxis < 0) {
      concatAxis = dim;
      continue;
    }
    return mlir::failure();
  }
  if (concatAxis < 0 || firstType.getDimSize(concatAxis) <= 0 ||
      secondType.getDimSize(concatAxis) <= 0)
    return mlir::failure();

  auto cmp = ifOp.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::ult ||
      cmp->getBlock() != generic.getBody())
    return mlir::failure();
  auto boundaryConstant = cmp.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
  auto boundaryAttr =
      boundaryConstant
          ? mlir::dyn_cast<mlir::IntegerAttr>(boundaryConstant.getValue())
          : mlir::IntegerAttr{};
  if (!boundaryAttr) {
    auto boundaryIt = scalarAttrs.find(cmp.getRhs());
    if (boundaryIt != scalarAttrs.end())
      boundaryAttr = mlir::dyn_cast<mlir::IntegerAttr>(boundaryIt->second);
  }
  if (!boundaryAttr || !mlir::isa<mlir::IndexType>(cmp.getRhs().getType()) ||
      boundaryAttr.getInt() != firstType.getDimSize(concatAxis))
    return mlir::failure();

  llvm::DenseSet<mlir::Operation *> bodySkeleton;
  bodySkeleton.insert(cmp.getOperation());
  bodySkeleton.insert(ifOp.getOperation());
  auto matchesLinalgIndex = [&](mlir::Value value, int64_t dim) {
    auto index = value.getDefiningOp<mlir::linalg::IndexOp>();
    if (!index || index->getParentOp() != generic.getOperation() ||
        index.getDim() != static_cast<uint64_t>(dim))
      return false;
    bodySkeleton.insert(index.getOperation());
    return true;
  };
  if (!matchesLinalgIndex(cmp.getLhs(), concatAxis) ||
      firstExtract.getIndices().size() != static_cast<size_t>(rank) ||
      secondExtract.getIndices().size() != static_cast<size_t>(rank))
    return mlir::failure();

  mlir::arith::SubIOp axisSubtract;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (!matchesLinalgIndex(firstExtract.getIndices()[dim], dim))
      return mlir::failure();
    if (dim != concatAxis) {
      if (!matchesLinalgIndex(secondExtract.getIndices()[dim], dim))
        return mlir::failure();
      continue;
    }

    auto subtract =
        secondExtract.getIndices()[dim].getDefiningOp<mlir::arith::SubIOp>();
    if (!subtract || subtract->getBlock() != ifOp.elseBlock() ||
        !matchesLinalgIndex(subtract.getLhs(), dim) ||
        subtract.getRhs() != cmp.getRhs())
      return mlir::failure();
    axisSubtract = subtract;
  }

  for (mlir::Operation &op : generic.getBody()->without_terminator()) {
    if (!bodySkeleton.contains(&op))
      return mlir::failure();
  }
  for (mlir::Operation &op : ifOp.thenBlock()->without_terminator()) {
    if (&op != firstExtract.getOperation())
      return mlir::failure();
  }
  for (mlir::Operation &op : ifOp.elseBlock()->without_terminator()) {
    if (&op != axisSubtract.getOperation() &&
        &op != secondExtract.getOperation())
      return mlir::failure();
  }

  return llvm::SmallVector<mlir::Value, 2>{firstExtract.getTensor(),
                                           secondExtract.getTensor()};
}

mlir::LogicalResult TileRegionBodyEmitter::convertTwoWayConcatGeneric(
    mlir::linalg::GenericOp generic, llvm::ArrayRef<mlir::Value> concatInputs,
    int64_t concatAxis, mlir::OpBuilder &builder) {
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!resultTensorType)
    return fail("concat generic result is not a ranked tensor");
  if (concatInputs.size() != 2)
    return fail("concat generic requires two inputs");

  mlir::FailureOr<mlir::Value> first = getOrMaterializeStructuredInput(
      concatInputs[0], MemLayout::Tensor, builder);
  mlir::FailureOr<mlir::Value> second = getOrMaterializeStructuredInput(
      concatInputs[1], MemLayout::Tensor, builder);
  if (mlir::failed(first) || mlir::failed(second))
    return mlir::failure();

  mlir::MemRefType resultType =
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  auto seed =
      builder.create<mlir::memref::AllocOp>(generic.getLoc(), resultType);
  recordScratchAllocation(seed);
  mlir::Value zero = createZeroScalar(
      generic.getLoc(), resultTensorType.getElementType(), builder);
  if (!zero)
    return fail("concat generic requires numeric element type");
  auto fill =
      builder.create<ComputeFillOp>(generic.getLoc(), seed.getResult(), zero,
                                    /*fill_domain=*/FillDomainAttr{});
  recordStructuredComputeOperation(fill);

  auto firstType = mlir::cast<mlir::MemRefType>((*first).getType());
  auto secondType = mlir::cast<mlir::MemRefType>((*second).getType());
  llvm::SmallVector<int64_t, 4> offsets(resultTensorType.getRank(), 0);
  llvm::SmallVector<int64_t, 4> strides(resultTensorType.getRank(), 1);
  llvm::SmallVector<int64_t, 4> firstSizes(firstType.getShape().begin(),
                                           firstType.getShape().end());
  llvm::SmallVector<int64_t, 4> secondSizes(secondType.getShape().begin(),
                                            secondType.getShape().end());

  mlir::MLIRContext *context = generic.getContext();
  builder.create<MoveInsertSliceOp>(
      generic.getLoc(), *first, seed.getResult(),
      mlir::DenseI64ArrayAttr::get(context, offsets),
      mlir::DenseI64ArrayAttr::get(context, firstSizes),
      mlir::DenseI64ArrayAttr::get(context, strides), CardDDRResourceAttr{});

  offsets[concatAxis] = firstType.getDimSize(concatAxis);
  builder.create<MoveInsertSliceOp>(
      generic.getLoc(), *second, seed.getResult(),
      mlir::DenseI64ArrayAttr::get(context, offsets),
      mlir::DenseI64ArrayAttr::get(context, secondSizes),
      mlir::DenseI64ArrayAttr::get(context, strides), CardDDRResourceAttr{});

  record(generic->getResult(0), MemLayout::Tensor, seed.getResult());
  return mlir::success();
}

mlir::FailureOr<bool> TileRegionBodyEmitter::tryConvertTiledTwoWayConcatGeneric(
    mlir::linalg::GenericOp generic, mlir::OpBuilder &builder) {
  auto noMatch = []() { return mlir::FailureOr<bool>(false); };
  if (generic.getNumDpsInputs() != 0 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1)
    return noMatch();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!resultTensorType || !resultTensorType.hasStaticShape() ||
      llvm::any_of(resultTensorType.getShape(),
                   [](int64_t extent) { return extent != 1; }))
    return noMatch();
  int64_t rank = resultTensorType.getRank();

  llvm::SmallVector<mlir::AffineMap, 1> maps = generic.getIndexingMapsArray();
  if (maps.size() != 1 || !isIdentityMap(maps.front(), rank))
    return noMatch();
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      generic.getIteratorTypesArray();
  if (iteratorTypes.size() != static_cast<size_t>(rank) ||
      !llvm::all_of(iteratorTypes, [](mlir::utils::IteratorType type) {
        return type == mlir::utils::IteratorType::parallel;
      }))
    return noMatch();

  auto yield =
      mlir::dyn_cast<mlir::linalg::YieldOp>(generic.getBody()->getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return noMatch();
  auto ifOp = yield.getValues().front().getDefiningOp<mlir::scf::IfOp>();
  if (!ifOp || ifOp->getNumResults() != 1 || !ifOp.thenBlock() ||
      !ifOp.elseBlock() || ifOp->getBlock() != generic.getBody())
    return noMatch();
  auto thenYield =
      mlir::dyn_cast<mlir::scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYield =
      mlir::dyn_cast<mlir::scf::YieldOp>(ifOp.elseBlock()->getTerminator());
  if (!thenYield || !elseYield || thenYield.getResults().size() != 1 ||
      elseYield.getResults().size() != 1)
    return noMatch();
  auto firstExtract =
      thenYield.getResults().front().getDefiningOp<mlir::tensor::ExtractOp>();
  auto secondExtract =
      elseYield.getResults().front().getDefiningOp<mlir::tensor::ExtractOp>();
  if (!firstExtract || !secondExtract ||
      firstExtract->getBlock() != ifOp.thenBlock() ||
      secondExtract->getBlock() != ifOp.elseBlock())
    return noMatch();

  auto firstType = mlir::dyn_cast<mlir::RankedTensorType>(
      firstExtract.getTensor().getType());
  auto secondType = mlir::dyn_cast<mlir::RankedTensorType>(
      secondExtract.getTensor().getType());
  if (!firstType || !secondType || !firstType.hasStaticShape() ||
      !secondType.hasStaticShape() || firstType.getRank() != rank ||
      secondType.getRank() != rank ||
      firstType.getElementType() != resultTensorType.getElementType() ||
      secondType.getElementType() != resultTensorType.getElementType() ||
      firstExtract.getIndices().size() != static_cast<size_t>(rank) ||
      secondExtract.getIndices().size() != static_cast<size_t>(rank))
    return noMatch();

  llvm::DenseSet<mlir::Operation *> bodySkeleton;
  bodySkeleton.insert(ifOp.getOperation());
  auto matchGlobalIndex = [&](mlir::Value value,
                              int64_t dim) -> std::optional<mlir::Value> {
    auto apply = value.getDefiningOp<mlir::affine::AffineApplyOp>();
    if (!apply || apply->getParentOp() != generic.getOperation() ||
        apply.getMapOperands().size() != 2)
      return std::nullopt;
    mlir::AffineMap map = apply.getAffineMap();
    if (map.getNumDims() != 2 || map.getNumSymbols() != 0 ||
        map.getNumResults() != 1)
      return std::nullopt;
    auto add = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(map.getResult(0));
    if (!add || add.getKind() != mlir::AffineExprKind::Add)
      return std::nullopt;
    auto lhs = mlir::dyn_cast<mlir::AffineDimExpr>(add.getLHS());
    auto rhs = mlir::dyn_cast<mlir::AffineDimExpr>(add.getRHS());
    if (!lhs || !rhs || lhs.getPosition() == rhs.getPosition() ||
        lhs.getPosition() >= 2 || rhs.getPosition() >= 2)
      return std::nullopt;

    for (unsigned localOperand = 0; localOperand < 2; ++localOperand) {
      auto index = apply.getMapOperands()[localOperand]
                       .getDefiningOp<mlir::linalg::IndexOp>();
      if (!index || index->getParentOp() != generic.getOperation() ||
          index.getDim() != static_cast<uint64_t>(dim))
        continue;
      unsigned offsetOperand = 1 - localOperand;
      if ((lhs.getPosition() != localOperand &&
           rhs.getPosition() != localOperand) ||
          (lhs.getPosition() != offsetOperand &&
           rhs.getPosition() != offsetOperand))
        return std::nullopt;
      mlir::Value offset = apply.getMapOperands()[offsetOperand];
      if (mlir::Operation *definition = offset.getDefiningOp();
          definition && generic->isProperAncestor(definition))
        return std::nullopt;
      bodySkeleton.insert(index.getOperation());
      bodySkeleton.insert(apply.getOperation());
      return offset;
    }
    return std::nullopt;
  };

  llvm::SmallVector<mlir::Value, 4> globalIndices;
  globalIndices.reserve(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    std::optional<mlir::Value> offset =
        matchGlobalIndex(firstExtract.getIndices()[dim], dim);
    if (!offset)
      return noMatch();
    globalIndices.push_back(*offset);
  }

  auto cmp = ifOp.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::ult ||
      cmp->getBlock() != generic.getBody())
    return noMatch();
  bodySkeleton.insert(cmp.getOperation());
  int64_t concatAxis = -1;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (cmp.getLhs() == firstExtract.getIndices()[dim]) {
      concatAxis = dim;
      break;
    }
  }
  if (concatAxis < 0)
    return noMatch();

  auto boundaryConstant = cmp.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
  auto boundaryAttr =
      boundaryConstant
          ? mlir::dyn_cast<mlir::IntegerAttr>(boundaryConstant.getValue())
          : mlir::IntegerAttr{};
  if (!boundaryAttr) {
    auto boundary = scalarAttrs.find(cmp.getRhs());
    if (boundary != scalarAttrs.end())
      boundaryAttr = mlir::dyn_cast<mlir::IntegerAttr>(boundary->second);
  }
  if (!boundaryAttr || !mlir::isa<mlir::IndexType>(cmp.getRhs().getType()) ||
      boundaryAttr.getInt() != firstType.getDimSize(concatAxis) ||
      boundaryAttr.getInt() <= 0)
    return noMatch();

  for (int64_t dim = 0; dim < rank; ++dim) {
    if (dim != concatAxis &&
        firstType.getDimSize(dim) != secondType.getDimSize(dim))
      return noMatch();
    if (dim != concatAxis) {
      if (secondExtract.getIndices()[dim] != firstExtract.getIndices()[dim])
        return noMatch();
      continue;
    }
    auto subtract =
        secondExtract.getIndices()[dim].getDefiningOp<mlir::arith::SubIOp>();
    if (!subtract || subtract->getBlock() != ifOp.elseBlock() ||
        subtract.getLhs() != firstExtract.getIndices()[dim] ||
        subtract.getRhs() != cmp.getRhs())
      return noMatch();
  }

  for (mlir::Operation &op : generic.getBody()->without_terminator())
    if (!bodySkeleton.contains(&op))
      return noMatch();
  for (mlir::Operation &op : ifOp.thenBlock()->without_terminator())
    if (&op != firstExtract.getOperation())
      return noMatch();
  for (mlir::Operation &op : ifOp.elseBlock()->without_terminator())
    if (!mlir::isa<mlir::arith::SubIOp>(op) &&
        &op != secondExtract.getOperation())
      return noMatch();

  llvm::SmallVector<mlir::Value, 4> convertedIndices;
  convertedIndices.reserve(globalIndices.size());
  for (auto [dim, index] : llvm::enumerate(globalIndices)) {
    mlir::FailureOr<mlir::Value> converted = getScalarValue(index, builder);
    if (mlir::failed(converted)) {
      std::string owner = "entry-block-argument";
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(index)) {
        if (mlir::Operation *parent = argument.getOwner()->getParentOp())
          owner = parent->getName().getStringRef().str();
      } else if (mlir::Operation *definition = index.getDefiningOp()) {
        owner = definition->getName().getStringRef().str();
      } else {
        owner = "value-without-defining-operation";
      }
      setFailureReason(failureReason,
                       "tiled concat global offset dim " + std::to_string(dim) +
                           " has no converted scalar value (owner=" + owner +
                           ")");
      return mlir::failure();
    }
    convertedIndices.push_back(*converted);
  }
  mlir::Value boundary = builder
                             .create<mlir::arith::ConstantIndexOp>(
                                 generic.getLoc(), boundaryAttr.getInt())
                             .getResult();
  auto condition = builder.create<mlir::arith::CmpIOp>(
      generic.getLoc(), mlir::arith::CmpIPredicate::ult,
      convertedIndices[concatAxis], boundary);
  mlir::MemRefType tileType =
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  auto convertedIf = builder.create<mlir::scf::IfOp>(
      generic.getLoc(), mlir::TypeRange{tileType}, condition.getResult(),
      /*withElseRegion=*/true);

  auto buildBranch = [&](mlir::Block *block, mlir::Value source,
                         bool second) -> mlir::LogicalResult {
    eraseImplicitYield(block);
    mlir::OpBuilder branchBuilder(block, block->end());
    auto external = externalBuffers.find(source);
    if (external == externalBuffers.end())
      return fail("tiled concat source is not an explicit DDR value");
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    offsets.reserve(convertedIndices.size());
    for (mlir::Value index : convertedIndices)
      offsets.push_back(index);
    if (second) {
      mlir::Value adjusted = branchBuilder.create<mlir::arith::SubIOp>(
          generic.getLoc(), convertedIndices[concatAxis], boundary);
      offsets[concatAxis] = adjusted;
    }
    llvm::SmallVector<int64_t, 4> sizes(rank, 1);
    llvm::SmallVector<int64_t, 4> strides(rank, 1);
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(external->second.getType());
    if (!sourceType || sourceType.getRank() != rank ||
        sourceType.getElementType() != resultTensorType.getElementType())
      return fail("tiled concat DDR source type does not match its tile");
    llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> mixedStrides;
    for (int64_t size : sizes)
      mixedSizes.push_back(branchBuilder.getIndexAttr(size));
    for (int64_t stride : strides)
      mixedStrides.push_back(branchBuilder.getIndexAttr(stride));
    auto subviewType = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            resultTensorType.getShape(), sourceType, offsets, mixedSizes,
            mixedStrides));
    auto view = branchBuilder.create<mlir::memref::SubViewOp>(
        generic.getLoc(), subviewType, external->second, offsets, mixedSizes,
        mixedStrides);
    auto tile =
        branchBuilder.create<mlir::memref::AllocOp>(generic.getLoc(), tileType);
    recordScratchAllocation(tile);
    branchBuilder.create<StorageLoadOp>(generic.getLoc(), view.getResult(),
                                        tile.getResult());
    branchBuilder.create<mlir::scf::YieldOp>(generic.getLoc(),
                                             tile.getResult());
    return mlir::success();
  };
  if (mlir::failed(buildBranch(convertedIf.thenBlock(),
                               firstExtract.getTensor(), /*second=*/false)) ||
      mlir::failed(buildBranch(convertedIf.elseBlock(),
                               secondExtract.getTensor(), /*second=*/true)))
    return mlir::failure();

  record(generic->getResult(0), MemLayout::Tensor, convertedIf.getResult(0));
  return true;
}

mlir::LogicalResult
TileRegionBodyEmitter::convertGeneric(mlir::linalg::GenericOp generic,
                                      mlir::OpBuilder &builder) {
  if (generic->getNumResults() == 0 ||
      generic.getNumDpsInits() != generic->getNumResults())
    return fail("unsupported linalg.generic arity");

  if (hasReductionIterator(generic))
    return convertReduceGeneric(generic, builder);
  if (generic->getNumResults() == 1 && getPassthroughInputIndex(generic))
    return convertPassthroughGeneric(generic, builder);
  int64_t concatAxis = -1;
  if (generic->getNumResults() == 1) {
    if (mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> concatInputs =
            matchTwoWayConcatGeneric(generic, concatAxis);
        mlir::succeeded(concatInputs))
      return convertTwoWayConcatGeneric(generic, *concatInputs, concatAxis,
                                        builder);
    mlir::FailureOr<bool> tiledConcat =
        tryConvertTiledTwoWayConcatGeneric(generic, builder);
    if (mlir::failed(tiledConcat))
      return mlir::failure();
    if (*tiledConcat)
      return mlir::success();
  }
  return convertElementwiseGenericExpression(generic, builder);
}

} // namespace wafer::tensor_program_to_tile_region
