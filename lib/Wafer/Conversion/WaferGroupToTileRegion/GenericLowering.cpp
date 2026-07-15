//===- GenericLowering.cpp - Generic linalg lowering ----------------===//

#include "Internal.h"

using namespace wafer;

namespace wafer::group_to_tile_region {

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
  builder.create<ComputeFillOp>(loc, alloc.getResult(), scalar);
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
      materializedInputs);
  return ElementwiseExprValue{elementwise.getResult(),
                              getIdentityMap(resultTensorType)};
}

mlir::LogicalResult TileRegionBodyEmitter::convertElementwiseScalarOp(
    mlir::linalg::GenericOp generic, mlir::Operation *op,
    llvm::DenseMap<mlir::Value, ElementwiseExprValue> &values,
    mlir::RankedTensorType resultTensorType, mlir::OpBuilder &builder) {
  auto lookup = [&](mlir::Value value) {
    auto it = values.find(value);
    if (it != values.end())
      return mlir::FailureOr<ElementwiseExprValue>(it->second);

    if (isScalarType(value.getType())) {
      mlir::FailureOr<mlir::Value> scalar = getScalarValue(value);
      if (mlir::failed(scalar))
        return mlir::FailureOr<ElementwiseExprValue>(mlir::failure());
      mlir::FailureOr<ElementwiseExprValue> splat =
          createElementwiseFillExprValue(value.getLoc(), *scalar,
                                         resultTensorType, builder);
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
        op->getLoc(), kind, {*operand}, resultTensorType, builder);
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
        op->getLoc(), kind, {*lhsValue, *rhsValue}, resultTensorType, builder);
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
        op->getLoc(), kind, {*firstValue, *secondValue, *thirdValue},
        resultTensorType, builder);
    if (mlir::failed(result))
      return mlir::failure();
    values[op->getResult(0)] = *result;
    return mlir::success();
  };

  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    if (constant->getNumResults() != 1 || !isScalarType(constant.getType()))
      return fail("unsupported linalg.generic constant expression");
    mlir::Operation *cloned = builder.clone(*constant.getOperation());
    mlir::FailureOr<ElementwiseExprValue> value =
        createElementwiseFillExprValue(constant.getLoc(), cloned->getResult(0),
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
    if (isScalarLikeConstant(generic, divf.getLhs(), 1.0))
      return createUnary(divf.getRhs(), ComputeElementwiseKind::Recip);
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
  if (auto exp = mlir::dyn_cast<mlir::math::ExpOp>(op))
    return createUnary(exp.getOperand(), ComputeElementwiseKind::Exp);
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
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!resultTensorType)
    return fail("generic result is not a ranked tensor");

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      generic.getIndexingMapsArray();
  if (indexingMaps.size() !=
      generic.getNumDpsInputs() + generic.getNumDpsInits())
    return fail("elementwise generic indexing map count mismatch");
  mlir::AffineMap resultMap = indexingMaps.back();
  if (!isIdentityMap(resultMap, resultTensorType.getRank()))
    return fail("elementwise generic result map must be identity");

  llvm::DenseMap<mlir::Value, ElementwiseExprValue> values;
  mlir::Block &body = *generic.getBody();
  unsigned bodyArgIndex = 0;
  for (mlir::Value input : generic.getDpsInputs()) {
    mlir::FailureOr<mlir::Value> buffer =
        getOrMaterialize(input, MemLayout::Tensor, builder);
    if (mlir::failed(buffer))
      return mlir::failure();
    values[body.getArgument(bodyArgIndex)] =
        ElementwiseExprValue{*buffer, indexingMaps[bodyArgIndex]};
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
                                                resultTensorType, builder)))
      return mlir::failure();
  }

  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return fail("unsupported linalg.generic yield");
  mlir::FailureOr<ElementwiseExprValue> yielded =
      getElementwiseExprValue(values, yield.getValues().front());
  if (mlir::failed(yielded))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> result = materializeElementwiseExprOperand(
      *yielded, resultTensorType, generic.getLoc(), builder);
  if (mlir::failed(result))
    return mlir::failure();
  record(generic->getResult(0), MemLayout::Tensor, *result);
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

  mlir::FailureOr<mlir::Value> source =
      getOrMaterialize(inputValue, MemLayout::Tensor, builder);
  if (mlir::failed(source))
    return mlir::failure();

  mlir::AffineMap inputMap = indexingMaps[*inputIndex];
  mlir::AffineMap resultMap = indexingMaps.back();
  if (!isIdentityMap(resultMap, resultTensorType.getRank()))
    return fail("passthrough generic result map must be identity");

  mlir::MemRefType resultType =
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  mlir::Value result;
  if (inputTensorType == resultTensorType &&
      isIdentityMap(inputMap, resultTensorType.getRank())) {
    result = builder.create<MoveCopyOp>(generic.getLoc(), resultType, *source)
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

  mlir::FailureOr<mlir::Value> first =
      getOrMaterialize(concatInputs[0], MemLayout::Tensor, builder);
  mlir::FailureOr<mlir::Value> second =
      getOrMaterialize(concatInputs[1], MemLayout::Tensor, builder);
  if (mlir::failed(first) || mlir::failed(second))
    return mlir::failure();

  mlir::MemRefType resultType =
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  auto seed =
      builder.create<mlir::memref::AllocOp>(generic.getLoc(), resultType);
  mlir::Value zero = createZeroScalar(
      generic.getLoc(), resultTensorType.getElementType(), builder);
  if (!zero)
    return fail("concat generic requires numeric element type");
  builder.create<ComputeFillOp>(generic.getLoc(), seed.getResult(), zero);

  auto firstType = mlir::cast<mlir::MemRefType>((*first).getType());
  auto secondType = mlir::cast<mlir::MemRefType>((*second).getType());
  llvm::SmallVector<int64_t, 4> offsets(resultTensorType.getRank(), 0);
  llvm::SmallVector<int64_t, 4> strides(resultTensorType.getRank(), 1);
  llvm::SmallVector<int64_t, 4> firstSizes(firstType.getShape().begin(),
                                           firstType.getShape().end());
  llvm::SmallVector<int64_t, 4> secondSizes(secondType.getShape().begin(),
                                            secondType.getShape().end());

  mlir::MLIRContext *context = generic.getContext();
  auto firstInsert = builder.create<MoveInsertSliceOp>(
      generic.getLoc(), resultType, *first, seed.getResult(),
      mlir::DenseI64ArrayAttr::get(context, offsets),
      mlir::DenseI64ArrayAttr::get(context, firstSizes),
      mlir::DenseI64ArrayAttr::get(context, strides));

  offsets[concatAxis] = firstType.getDimSize(concatAxis);
  auto secondInsert = builder.create<MoveInsertSliceOp>(
      generic.getLoc(), resultType, *second, firstInsert.getResult(),
      mlir::DenseI64ArrayAttr::get(context, offsets),
      mlir::DenseI64ArrayAttr::get(context, secondSizes),
      mlir::DenseI64ArrayAttr::get(context, strides));

  record(generic->getResult(0), MemLayout::Tensor, secondInsert.getResult());
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertGeneric(mlir::linalg::GenericOp generic,
                                      mlir::OpBuilder &builder) {
  if (generic.getNumDpsInits() != 1 || generic->getNumResults() != 1)
    return fail("unsupported linalg.generic arity");

  if (hasReductionIterator(generic))
    return convertReduceGeneric(generic, builder);
  if (getPassthroughInputIndex(generic))
    return convertPassthroughGeneric(generic, builder);
  int64_t concatAxis = -1;
  if (mlir::FailureOr<llvm::SmallVector<mlir::Value, 2>> concatInputs =
          matchTwoWayConcatGeneric(generic, concatAxis);
      mlir::succeeded(concatInputs))
    return convertTwoWayConcatGeneric(generic, *concatInputs, concatAxis,
                                      builder);
  return convertElementwiseGenericExpression(generic, builder);
}

} // namespace wafer::group_to_tile_region
