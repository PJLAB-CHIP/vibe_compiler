//===- NamedComputeLowering.cpp - Named compute lowering ------------===//

#include "Internal.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

namespace {

static std::optional<unsigned> findMapResult(mlir::AffineMap map,
                                             mlir::AffineExpr expected) {
  expected =
      mlir::simplifyAffineExpr(expected, map.getNumDims(), map.getNumSymbols());
  std::optional<unsigned> found;
  for (auto [index, rawExpression] : llvm::enumerate(map.getResults())) {
    mlir::AffineExpr expression = mlir::simplifyAffineExpr(
        rawExpression, map.getNumDims(), map.getNumSymbols());
    if (expression != expected)
      continue;
    if (found)
      return std::nullopt;
    found = static_cast<unsigned>(index);
  }
  return found;
}

static bool isPermutation(llvm::ArrayRef<int64_t> permutation, unsigned rank) {
  if (permutation.size() != rank)
    return false;
  llvm::SmallVector<bool, 4> seen(rank, false);
  for (int64_t dim : permutation) {
    if (dim < 0 || static_cast<unsigned>(dim) >= rank || seen[dim])
      return false;
    seen[dim] = true;
  }
  return true;
}

static bool isIdentityPermutation(llvm::ArrayRef<int64_t> permutation) {
  return llvm::all_of(llvm::enumerate(permutation), [](auto indexed) {
    return static_cast<int64_t>(indexed.index()) == indexed.value();
  });
}

static mlir::RankedTensorType
permuteTensorType(mlir::RankedTensorType source,
                  llvm::ArrayRef<int64_t> resultToSource) {
  llvm::SmallVector<int64_t, 4> shape;
  shape.reserve(resultToSource.size());
  for (int64_t sourceDim : resultToSource)
    shape.push_back(source.getDimSize(sourceDim));
  return mlir::RankedTensorType::get(shape, source.getElementType());
}

} // namespace

mlir::FailureOr<OrdinaryConv2DGeometry>
inferOrdinaryConv2DGeometry(mlir::linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return mlir::failure();
  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getDpsInputs()[0].getType());
  auto weightType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getDpsInputs()[1].getType());
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !weightType || !outputType || inputType.getRank() != 4 ||
      weightType.getRank() != 4 || outputType.getRank() != 4 ||
      !inputType.hasStaticShape() || !weightType.hasStaticShape() ||
      !outputType.hasStaticShape() ||
      llvm::any_of(inputType.getShape(),
                   [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(weightType.getShape(),
                   [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(outputType.getShape(),
                   [](int64_t value) { return value <= 0; }))
    return mlir::failure();

  mlir::FailureOr<mlir::linalg::ConvolutionDimensions> inferred =
      mlir::linalg::inferConvolutionDims(op);
  if (mlir::failed(inferred) || inferred->batch.size() != 1 ||
      inferred->outputImage.size() != 2 ||
      inferred->outputChannel.size() != 1 || inferred->filterLoop.size() != 2 ||
      inferred->inputChannel.size() != 1 || !inferred->depth.empty() ||
      inferred->strides.size() != 2 || inferred->dilations.size() != 2 ||
      llvm::any_of(inferred->strides,
                   [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(inferred->dilations,
                   [](int64_t value) { return value <= 0; }))
    return mlir::failure();

  llvm::SmallVector<mlir::utils::IteratorType, 8> iteratorTypes =
      op.getIteratorTypesArray();
  llvm::SmallVector<bool, 8> classified(iteratorTypes.size(), false);
  auto classify = [&](llvm::ArrayRef<unsigned> loops,
                      mlir::utils::IteratorType expected) {
    for (unsigned loop : loops) {
      if (loop >= iteratorTypes.size() || classified[loop] ||
          iteratorTypes[loop] != expected)
        return false;
      classified[loop] = true;
    }
    return true;
  };
  if (!classify(inferred->batch, mlir::utils::IteratorType::parallel) ||
      !classify(inferred->outputImage, mlir::utils::IteratorType::parallel) ||
      !classify(inferred->outputChannel, mlir::utils::IteratorType::parallel) ||
      !classify(inferred->filterLoop, mlir::utils::IteratorType::reduction) ||
      !classify(inferred->inputChannel, mlir::utils::IteratorType::reduction) ||
      llvm::any_of(classified, [](bool value) { return !value; }))
    return mlir::failure();

  llvm::SmallVector<mlir::AffineMap, 3> maps = op.getIndexingMapsArray();
  if (maps.size() != 3 || llvm::any_of(maps, [](mlir::AffineMap map) {
        return map.getNumSymbols() != 0;
      }))
    return mlir::failure();
  mlir::AffineMap inputMap = maps[0];
  mlir::AffineMap weightMap = maps[1];
  mlir::AffineMap outputMap = maps[2];
  if (inputMap.getNumResults() != 4 || weightMap.getNumResults() != 4 ||
      outputMap.getNumResults() != 4)
    return mlir::failure();

  mlir::MLIRContext *context = op.getContext();
  auto loopExpr = [&](unsigned loop) {
    return mlir::getAffineDimExpr(loop, context);
  };
  auto findLoop = [&](mlir::AffineMap map,
                      unsigned loop) -> std::optional<unsigned> {
    return findMapResult(map, loopExpr(loop));
  };
  std::optional<unsigned> inputBatch =
      findLoop(inputMap, inferred->batch.front());
  std::optional<unsigned> inputChannel =
      findLoop(inputMap, inferred->inputChannel.front());
  std::optional<unsigned> weightOutputChannel =
      findLoop(weightMap, inferred->outputChannel.front());
  std::optional<unsigned> weightInputChannel =
      findLoop(weightMap, inferred->inputChannel.front());
  std::optional<unsigned> outputBatch =
      findLoop(outputMap, inferred->batch.front());
  std::optional<unsigned> outputChannel =
      findLoop(outputMap, inferred->outputChannel.front());
  if (!inputBatch || !inputChannel || !weightOutputChannel ||
      !weightInputChannel || !outputBatch || !outputChannel)
    return mlir::failure();

  llvm::SmallVector<unsigned, 2> inputSpatial;
  llvm::SmallVector<unsigned, 2> weightSpatial;
  llvm::SmallVector<unsigned, 2> outputSpatial;
  for (unsigned index = 0; index < 2; ++index) {
    mlir::AffineExpr inputWindow =
        loopExpr(inferred->outputImage[index]) * inferred->strides[index] +
        loopExpr(inferred->filterLoop[index]) * inferred->dilations[index];
    std::optional<unsigned> inputDim = findMapResult(inputMap, inputWindow);
    std::optional<unsigned> weightDim =
        findLoop(weightMap, inferred->filterLoop[index]);
    std::optional<unsigned> outputDim =
        findLoop(outputMap, inferred->outputImage[index]);
    if (!inputDim || !weightDim || !outputDim)
      return mlir::failure();
    inputSpatial.push_back(*inputDim);
    weightSpatial.push_back(*weightDim);
    outputSpatial.push_back(*outputDim);
  }

  OrdinaryConv2DGeometry geometry;
  geometry.inputToNHWC = {static_cast<int64_t>(*inputBatch),
                          static_cast<int64_t>(inputSpatial[0]),
                          static_cast<int64_t>(inputSpatial[1]),
                          static_cast<int64_t>(*inputChannel)};
  // The target-abstract canonical weight order follows the hardware-neutral
  // logical X/Y convention used by instruction packing: X (width), Y
  // (height), output channel, input channel.
  geometry.weightToXYOI = {static_cast<int64_t>(weightSpatial[1]),
                           static_cast<int64_t>(weightSpatial[0]),
                           static_cast<int64_t>(*weightOutputChannel),
                           static_cast<int64_t>(*weightInputChannel)};
  geometry.outputToNHWC = {static_cast<int64_t>(*outputBatch),
                           static_cast<int64_t>(outputSpatial[0]),
                           static_cast<int64_t>(outputSpatial[1]),
                           static_cast<int64_t>(*outputChannel)};
  if (!isPermutation(geometry.inputToNHWC, 4) ||
      !isPermutation(geometry.weightToXYOI, 4) ||
      !isPermutation(geometry.outputToNHWC, 4))
    return mlir::failure();
  geometry.outputFromNHWC.assign(4, -1);
  for (auto [canonicalDim, sourceDim] : llvm::enumerate(geometry.outputToNHWC))
    geometry.outputFromNHWC[sourceDim] = canonicalDim;
  geometry.stridesHW = inferred->strides;
  geometry.dilationsHW = inferred->dilations;
  // Affine window maps carry stride and dilation, but cannot encode padding
  // boundary values or how total padding is split before/after. Accept the
  // map-only form only when the current operand/result geometry proves that
  // no implicit padding or unpadding is required. An explicit upstream pad is
  // already part of the input tensor and therefore also satisfies this exact
  // zero-padding relation. We never infer ambiguous padding from shapes.
  auto inferValidOutput = [](int64_t input, int64_t kernel, int64_t stride,
                             int64_t dilation) -> std::optional<int64_t> {
    if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 ||
        kernel - 1 > (std::numeric_limits<int64_t>::max() - 1) / dilation)
      return std::nullopt;
    int64_t effectiveKernel = (kernel - 1) * dilation + 1;
    if (input < effectiveKernel)
      return std::nullopt;
    return (input - effectiveKernel) / stride + 1;
  };
  mlir::RankedTensorType canonicalInput =
      permuteTensorType(inputType, geometry.inputToNHWC);
  mlir::RankedTensorType canonicalWeight =
      permuteTensorType(weightType, geometry.weightToXYOI);
  mlir::RankedTensorType canonicalOutput =
      permuteTensorType(outputType, geometry.outputToNHWC);
  std::optional<int64_t> expectedH = inferValidOutput(
      canonicalInput.getDimSize(1), canonicalWeight.getDimSize(1),
      geometry.stridesHW[0], geometry.dilationsHW[0]);
  std::optional<int64_t> expectedW = inferValidOutput(
      canonicalInput.getDimSize(2), canonicalWeight.getDimSize(0),
      geometry.stridesHW[1], geometry.dilationsHW[1]);
  if (!expectedH || !expectedW ||
      canonicalOutput.getDimSize(0) != canonicalInput.getDimSize(0) ||
      canonicalInput.getDimSize(3) != canonicalWeight.getDimSize(3) ||
      canonicalOutput.getDimSize(3) != canonicalWeight.getDimSize(2) ||
      canonicalOutput.getDimSize(1) != *expectedH ||
      canonicalOutput.getDimSize(2) != *expectedW)
    return mlir::failure();
  geometry.pads.assign(4, 0);
  geometry.unpads.assign(4, 0);
  return geometry;
}

mlir::LogicalResult
TileRegionBodyEmitter::verifyNamedLinalgPayloads(TensorProgramScope scope) {
  mlir::WalkResult result = scope.getFunction().walk([&](mlir::Operation *op) {
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op)) {
      if (mlir::failed(verifyExactFillPayload(fill)))
        return mlir::WalkResult::interrupt();
    } else if (mlir::isa<mlir::linalg::MatmulOp,
                         mlir::linalg::MatmulTransposeAOp,
                         mlir::linalg::MatmulTransposeBOp>(op)) {
      if (mlir::failed(verifyExactGemmPayload(
              mlir::cast<mlir::linalg::LinalgOp>(op), "matmul")))
        return mlir::WalkResult::interrupt();
    } else if (mlir::isa<mlir::linalg::BatchMatmulOp,
                         mlir::linalg::BatchMatmulTransposeAOp,
                         mlir::linalg::BatchMatmulTransposeBOp>(op)) {
      if (mlir::failed(verifyExactGemmPayload(
              mlir::cast<mlir::linalg::LinalgOp>(op), "batch matmul")))
        return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::verifyExactFillPayload(mlir::linalg::FillOp fill) {
  mlir::linalg::LinalgOp op = fill;
  if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
      op.getRegionInputArgs().size() != 1 ||
      op.getRegionOutputArgs().size() != 1)
    return fail("linalg.fill requires the canonical scalar payload");

  mlir::Block &body = op->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != op.getRegionInputArgs().front() ||
      !body.without_terminator().empty())
    return fail("linalg.fill requires the canonical scalar payload");
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::verifyExactGemmPayload(mlir::linalg::LinalgOp op,
                                              llvm::StringRef subject) {
  if (!hasExactGemmPayload(op))
    return fail(
        (subject + " requires an exact multiply-accumulate payload").str());
  return mlir::success();
}

bool TileRegionBodyEmitter::hasExactGemmPayload(
    mlir::linalg::LinalgOp op) const {
  if (op->getNumRegions() != 1 || op->getRegion(0).empty() ||
      op.getRegionInputArgs().size() != 2 ||
      op.getRegionOutputArgs().size() != 1)
    return false;

  mlir::Block &body = op->getRegion(0).front();
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  llvm::SmallVector<mlir::Operation *, 2> payloadOps;
  for (mlir::Operation &payloadOp : body.without_terminator())
    payloadOps.push_back(&payloadOp);
  if (!yield || yield.getValues().size() != 1 || payloadOps.size() != 2)
    return false;

  mlir::Value lhs = op.getRegionInputArgs()[0];
  mlir::Value rhs = op.getRegionInputArgs()[1];
  mlir::Value accumulator = op.getRegionOutputArgs()[0];
  auto matchesPair = [](mlir::Value first, mlir::Value second,
                        mlir::Value expectedFirst, mlir::Value expectedSecond) {
    return (first == expectedFirst && second == expectedSecond) ||
           (first == expectedSecond && second == expectedFirst);
  };

  mlir::Value sum;
  if (auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(payloadOps[0])) {
    auto add = mlir::dyn_cast<mlir::arith::AddFOp>(payloadOps[1]);
    if (!add || !matchesPair(mul.getLhs(), mul.getRhs(), lhs, rhs) ||
        !matchesPair(add.getLhs(), add.getRhs(), mul.getResult(), accumulator))
      return false;
    sum = add.getResult();
  } else if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(payloadOps[0])) {
    auto add = mlir::dyn_cast<mlir::arith::AddIOp>(payloadOps[1]);
    if (!add ||
        mul.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none ||
        add.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none ||
        !matchesPair(mul.getLhs(), mul.getRhs(), lhs, rhs) ||
        !matchesPair(add.getLhs(), add.getRhs(), mul.getResult(), accumulator))
      return false;
    sum = add.getResult();
  } else {
    return false;
  }
  return yield.getValues().front() == sum;
}

mlir::LogicalResult
TileRegionBodyEmitter::convertFill(mlir::linalg::FillOp fill,
                                   mlir::OpBuilder &builder) {
  mlir::linalg::LinalgOp op = fill;
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      fill->getNumResults() != 1)
    return fail("unsupported linalg.fill arity");
  if (mlir::failed(verifyExactFillPayload(fill)))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> value =
      getScalarValue(op.getDpsInputs()[0], builder);
  if (mlir::failed(value))
    return mlir::failure();

  fillInitScalars[fill.getResult(0)] = *value;
  if (auto attrIt = scalarAttrs.find(op.getDpsInputs()[0]);
      attrIt != scalarAttrs.end())
    fillInitAttrs[fill.getResult(0)] = attrIt->second;

  // Target GEMM/reduce consume a typed scalar initialization while their
  // complete traversal writes every result tile. Preserve that source-level
  // fact without materializing a full-shape fill in SPM.
  if (onlyFeedsScalarInitializedComputeInit(fill.getResult(0))) {
    // The fill result still names the same destination object.  Preserve an
    // explicit DDR destination across the proof-only fill so a tiled compute
    // can carry that object through its traversal loops and store each
    // produced tile directly. This is buffer identity propagation; the fill
    // scalar remains attached to each target compute tile below.
    mlir::Value init = op.getDpsInits().front();
    mlir::Value externalBuffer = externalBuffers.lookup(init);
    if (externalBuffer) {
      externalBuffers[fill.getResult(0)] = externalBuffer;
      if (writableExternalBuffers.contains(init))
        writableExternalBuffers.insert(fill.getResult(0));
      std::optional<unsigned> outputIndex;
      if (auto outputIndexIt = externalOutputIndices.find(init);
          outputIndexIt != externalOutputIndices.end())
        outputIndex = outputIndexIt->second;
      if (outputIndex)
        externalOutputIndices[fill.getResult(0)] = *outputIndex;
      mlir::Value baseBuffer = directYieldBuffers.lookup(init);
      if (baseBuffer)
        directYieldBuffers[fill.getResult(0)] = baseBuffer;
    }
    return mlir::success();
  }

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(fill.getResult(0).getType());
  if (!resultTensorType)
    return fail("linalg.fill result is not a ranked tensor");
  auto result = builder.create<mlir::memref::AllocOp>(
      fill.getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Tensor));
  recordScratchAllocation(result);
  auto compute =
      builder.create<ComputeFillOp>(fill.getLoc(), result.getResult(), *value,
                                    /*fill_domain=*/FillDomainAttr{});
  recordStructuredComputeOperation(compute);
  record(fill.getResult(0), MemLayout::Tensor, result.getResult());
  return mlir::success();
}

bool TileRegionBodyEmitter::hasPositiveZeroFilledComputeInit(
    mlir::linalg::LinalgOp op) const {
  if (op.getNumDpsInits() != 1)
    return false;

  auto isPositiveZero = [](mlir::Attribute attr) {
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
      const llvm::APFloat &value = floatAttr.getValue();
      return value.isZero() && !value.isNegative();
    }
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getValue().isZero();
    if (auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
      if (!elements.isSplat())
        return false;
      if (mlir::isa<mlir::FloatType>(elements.getElementType())) {
        llvm::APFloat value = elements.getSplatValue<mlir::APFloat>();
        return value.isZero() && !value.isNegative();
      }
      if (mlir::isa<mlir::IntegerType>(elements.getElementType()))
        return elements.getSplatValue<mlir::APInt>().isZero();
    }
    return false;
  };

  auto attrIt = fillInitAttrs.find(op.getDpsInits().front());
  return attrIt != fillInitAttrs.end() && isPositiveZero(attrIt->second);
}

mlir::FailureOr<std::pair<GemmOrientation, GemmOrientation>>
TileRegionBodyEmitter::inferRank2GemmOrientations(mlir::linalg::LinalgOp op) {
  llvm::SmallVector<mlir::utils::IteratorType, 3> iterators =
      op.getIteratorTypesArray();
  if (iterators != llvm::ArrayRef<mlir::utils::IteratorType>{
                       mlir::utils::IteratorType::parallel,
                       mlir::utils::IteratorType::parallel,
                       mlir::utils::IteratorType::reduction})
    return mlir::failure();
  llvm::SmallVector<mlir::AffineMap, 3> maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return mlir::failure();
  mlir::MLIRContext *context = op.getContext();
  auto d0 = mlir::getAffineDimExpr(0, context);
  auto d1 = mlir::getAffineDimExpr(1, context);
  auto d2 = mlir::getAffineDimExpr(2, context);
  auto map = [&](mlir::AffineExpr first, mlir::AffineExpr second) {
    return mlir::AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0,
                                {first, second}, context);
  };
  std::optional<GemmOrientation> lhs;
  std::optional<GemmOrientation> rhs;
  if (maps[0] == map(d0, d2))
    lhs = GemmOrientation::Normal;
  else if (maps[0] == map(d2, d0))
    lhs = GemmOrientation::Transpose;
  if (maps[1] == map(d2, d1))
    rhs = GemmOrientation::Normal;
  else if (maps[1] == map(d1, d2))
    rhs = GemmOrientation::Transpose;
  if (!lhs || !rhs || maps[2] != map(d0, d1))
    return mlir::failure();
  return std::pair{*lhs, *rhs};
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::createAccumulatorCombine(
    mlir::Location loc, ComputeReduceKind kind, mlir::Value accumulator,
    mlir::Value partial, mlir::RankedTensorType resultTensorType,
    mlir::OpBuilder &builder) {
  ComputeElementwiseKind elementwiseKind;
  switch (kind) {
  case ComputeReduceKind::Sum:
    elementwiseKind = ComputeElementwiseKind::Add;
    break;
  case ComputeReduceKind::Max:
    elementwiseKind = ComputeElementwiseKind::Max;
    break;
  case ComputeReduceKind::Min:
    elementwiseKind = ComputeElementwiseKind::Min;
    break;
  case ComputeReduceKind::Avg:
    return failValue("explicit accumulator combine does not support average");
  }

  mlir::FailureOr<mlir::Value> previous =
      getOrMaterialize(accumulator, MemLayout::Tensor, builder);
  if (mlir::failed(previous))
    return mlir::failure();

  mlir::Type tensorBufferType =
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor);
  mlir::Value partialTensor = partial;
  if (partialTensor.getType() != tensorBufferType) {
    partialTensor =
        builder
            .create<LayoutMaterializeOp>(loc, tensorBufferType, partialTensor)
            .getResult();
  }
  if ((*previous).getType() != tensorBufferType)
    return failValue("explicit accumulator type mismatch");

  auto kindAttr =
      ComputeElementwiseKindAttr::get(builder.getContext(), elementwiseKind);
  auto combined = builder.create<ComputeElementwiseIntoOp>(
      loc, kindAttr, mlir::ValueRange{*previous, partialTensor}, *previous);
  recordStructuredComputeOperation(combined);
  return *previous;
}

mlir::LogicalResult
TileRegionBodyEmitter::convertMatmul(mlir::linalg::LinalgOp op,
                                     mlir::OpBuilder &builder) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return fail("unsupported matmul arity");
  if (mlir::failed(verifyExactGemmPayload(op, "matmul")))
    return mlir::failure();
  mlir::FailureOr<std::pair<GemmOrientation, GemmOrientation>> orientations =
      inferRank2GemmOrientations(op);
  if (mlir::failed(orientations))
    return fail("unsupported rank-2 matmul indexing maps");
  bool overwriteInit = hasPositiveZeroFilledComputeInit(op);

  mlir::FailureOr<mlir::Value> lhs = getOrMaterializeStructuredInput(
      op.getDpsInputs()[0], MemLayout::Cx, builder);
  mlir::FailureOr<mlir::Value> rhs = getOrMaterializeStructuredInput(
      op.getDpsInputs()[1], MemLayout::Cx, builder);
  if (mlir::failed(lhs) || mlir::failed(rhs))
    return mlir::failure();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultTensorType)
    return fail("matmul result is not a ranked tensor");

  GemmOrientationAttr lhsOrientation;
  GemmOrientationAttr rhsOrientation;
  if (orientations->first != GemmOrientation::Normal ||
      orientations->second != GemmOrientation::Normal) {
    lhsOrientation =
        GemmOrientationAttr::get(builder.getContext(), orientations->first);
    rhsOrientation =
        GemmOrientationAttr::get(builder.getContext(), orientations->second);
  }
  auto gemm = builder.create<ComputeGemmOp>(
      op->getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::Cx), *lhs,
      *rhs, lhsOrientation, rhsOrientation, mlir::IntegerAttr{},
      mlir::DenseI64ArrayAttr{}, mlir::IntegerAttr{}, mlir::IntegerAttr{},
      mlir::DenseI64ArrayAttr{}, mlir::IntegerAttr{}, mlir::IntegerAttr{},
      mlir::DenseI64ArrayAttr{}, mlir::IntegerAttr{}, mlir::IntegerAttr{});
  recordStructuredComputeOperation(gemm);
  if (!overwriteInit) {
    mlir::FailureOr<mlir::Value> combined = createAccumulatorCombine(
        op->getLoc(), ComputeReduceKind::Sum, op.getDpsInits().front(),
        gemm.getResult(), resultTensorType, builder);
    if (mlir::failed(combined))
      return mlir::failure();
    record(op->getResult(0), MemLayout::Tensor, *combined);
    return mlir::success();
  }
  record(op->getResult(0), MemLayout::Cx, gemm.getResult());
  return mlir::success();
}

mlir::FailureOr<unsigned> TileRegionBodyEmitter::findOperandDimForLoop(
    mlir::AffineMap map, unsigned loopDim, llvm::StringRef role) {
  for (auto [operandDim, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (!dimExpr)
      return failUnsigned("batch matmul requires projected permutation " +
                          role.str() + " indexing map");
    if (dimExpr.getPosition() == loopDim)
      return static_cast<unsigned>(operandDim);
  }
  return failUnsigned("batch matmul indexing map is missing " + role.str() +
                      " loop dimension");
}

bool TileRegionBodyEmitter::mapContainsLoopDim(mlir::AffineMap map,
                                               unsigned loopDim) const {
  for (mlir::AffineExpr expr : map.getResults()) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
    if (dimExpr && dimExpr.getPosition() == loopDim)
      return true;
  }
  return false;
}

mlir::FailureOr<BatchedGemmAttrs>
TileRegionBodyEmitter::inferBatchMatmulAttrs(mlir::linalg::LinalgOp op) {
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      op.getIteratorTypesArray();
  llvm::SmallVector<unsigned, 1> reductionLoops;
  for (auto [index, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      reductionLoops.push_back(static_cast<unsigned>(index));
  }
  if (reductionLoops.size() != 1)
    return mlir::failure();

  llvm::SmallVector<mlir::AffineMap, 4> maps = op.getIndexingMapsArray();
  if (maps.size() != 3)
    return mlir::failure();
  mlir::AffineMap lhsMap = maps[0];
  mlir::AffineMap rhsMap = maps[1];
  mlir::AffineMap resultMap = maps[2];

  std::optional<unsigned> mLoop;
  std::optional<unsigned> nLoop;
  llvm::SmallVector<unsigned, 4> batchLoops;
  unsigned reductionLoop = reductionLoops.front();
  for (unsigned loopDim = 0; loopDim < iteratorTypes.size(); ++loopDim) {
    if (loopDim == reductionLoop)
      continue;
    bool inLhs = mapContainsLoopDim(lhsMap, loopDim);
    bool inRhs = mapContainsLoopDim(rhsMap, loopDim);
    bool inResult = mapContainsLoopDim(resultMap, loopDim);
    if (inLhs && !inRhs && inResult) {
      if (mLoop)
        return mlir::failure();
      mLoop = loopDim;
      continue;
    }
    if (!inLhs && inRhs && inResult) {
      if (nLoop)
        return mlir::failure();
      nLoop = loopDim;
      continue;
    }
    if (inLhs && inRhs && inResult) {
      batchLoops.push_back(loopDim);
      continue;
    }
    return mlir::failure();
  }
  if (!mLoop || !nLoop || batchLoops.empty())
    return mlir::failure();

  BatchedGemmAttrs attrs;
  auto lhsMDim = findOperandDimForLoop(lhsMap, *mLoop, "lhs M");
  auto lhsKDim =
      findOperandDimForLoop(lhsMap, reductionLoop, "lhs contracting");
  auto rhsKDim =
      findOperandDimForLoop(rhsMap, reductionLoop, "rhs contracting");
  auto rhsNDim = findOperandDimForLoop(rhsMap, *nLoop, "rhs N");
  auto resultMDim = findOperandDimForLoop(resultMap, *mLoop, "result M");
  auto resultNDim = findOperandDimForLoop(resultMap, *nLoop, "result N");
  if (mlir::failed(lhsMDim) || mlir::failed(lhsKDim) || mlir::failed(rhsKDim) ||
      mlir::failed(rhsNDim) || mlir::failed(resultMDim) ||
      mlir::failed(resultNDim))
    return mlir::failure();
  attrs.lhsMDim = *lhsMDim;
  attrs.lhsContractingDim = *lhsKDim;
  attrs.rhsContractingDim = *rhsKDim;
  attrs.rhsNDim = *rhsNDim;
  attrs.resultMDim = *resultMDim;
  attrs.resultNDim = *resultNDim;

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultTensorType || !resultTensorType.hasStaticShape())
    return mlir::failure();

  attrs.batchCount = 1;
  for (unsigned batchLoop : batchLoops) {
    auto lhsBatchDim = findOperandDimForLoop(lhsMap, batchLoop, "lhs batch");
    auto rhsBatchDim = findOperandDimForLoop(rhsMap, batchLoop, "rhs batch");
    auto resultBatchDim =
        findOperandDimForLoop(resultMap, batchLoop, "result batch");
    if (mlir::failed(lhsBatchDim) || mlir::failed(rhsBatchDim) ||
        mlir::failed(resultBatchDim))
      return mlir::failure();
    attrs.lhsBatchDims.push_back(*lhsBatchDim);
    attrs.rhsBatchDims.push_back(*rhsBatchDim);
    attrs.resultBatchDims.push_back(*resultBatchDim);
    int64_t dimSize = resultTensorType.getDimSize(*resultBatchDim);
    if (mlir::ShapedType::isDynamic(dimSize) || dimSize <= 0)
      return mlir::failure();
    if (attrs.batchCount > std::numeric_limits<int64_t>::max() / dimSize)
      return mlir::failure();
    attrs.batchCount *= dimSize;
  }
  return attrs;
}

mlir::LogicalResult
TileRegionBodyEmitter::convertBatchMatmul(mlir::linalg::LinalgOp op,
                                          mlir::OpBuilder &builder) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op->getNumResults() != 1)
    return fail("unsupported batch matmul arity");
  if (mlir::failed(verifyExactGemmPayload(op, "batch matmul")))
    return mlir::failure();
  bool overwriteInit = hasPositiveZeroFilledComputeInit(op);

  mlir::FailureOr<mlir::Value> lhs = getOrMaterializeStructuredInput(
      op.getDpsInputs()[0], MemLayout::NCx, builder);
  mlir::FailureOr<mlir::Value> rhs = getOrMaterializeStructuredInput(
      op.getDpsInputs()[1], MemLayout::NCx, builder);
  if (mlir::failed(lhs) || mlir::failed(rhs))
    return mlir::failure();

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultTensorType)
    return fail("batch matmul result is not a ranked tensor");
  if (resultTensorType.getRank() != 3)
    return fail(
        "target batch matmul requires one explicit leading batch dimension; "
        "multiple batch dimensions must be flattened before NCx "
        "materialization");

  mlir::FailureOr<BatchedGemmAttrs> attrs = inferBatchMatmulAttrs(op);
  if (mlir::failed(attrs))
    return fail("unsupported batch matmul indexing");

  GemmOrientation lhsOrientation = attrs->lhsMDim < attrs->lhsContractingDim
                                       ? GemmOrientation::Normal
                                       : GemmOrientation::Transpose;
  GemmOrientation rhsOrientation = attrs->rhsContractingDim < attrs->rhsNDim
                                       ? GemmOrientation::Normal
                                       : GemmOrientation::Transpose;
  GemmOrientationAttr lhsOrientationAttr;
  GemmOrientationAttr rhsOrientationAttr;
  if (lhsOrientation != GemmOrientation::Normal ||
      rhsOrientation != GemmOrientation::Normal) {
    lhsOrientationAttr =
        GemmOrientationAttr::get(builder.getContext(), lhsOrientation);
    rhsOrientationAttr =
        GemmOrientationAttr::get(builder.getContext(), rhsOrientation);
  }
  auto gemm = builder.create<ComputeGemmOp>(
      op->getLoc(), makeSPMMemRefType(resultTensorType, MemLayout::NCx), *lhs,
      *rhs, lhsOrientationAttr, rhsOrientationAttr,
      builder.getI64IntegerAttr(attrs->batchCount),
      builder.getDenseI64ArrayAttr(attrs->lhsBatchDims),
      builder.getI64IntegerAttr(attrs->lhsMDim),
      builder.getI64IntegerAttr(attrs->lhsContractingDim),
      builder.getDenseI64ArrayAttr(attrs->rhsBatchDims),
      builder.getI64IntegerAttr(attrs->rhsContractingDim),
      builder.getI64IntegerAttr(attrs->rhsNDim),
      builder.getDenseI64ArrayAttr(attrs->resultBatchDims),
      builder.getI64IntegerAttr(attrs->resultMDim),
      builder.getI64IntegerAttr(attrs->resultNDim));
  recordStructuredComputeOperation(gemm);
  if (!overwriteInit) {
    mlir::FailureOr<mlir::Value> combined = createAccumulatorCombine(
        op->getLoc(), ComputeReduceKind::Sum, op.getDpsInits().front(),
        gemm.getResult(), resultTensorType, builder);
    if (mlir::failed(combined))
      return mlir::failure();
    record(op->getResult(0), MemLayout::Tensor, *combined);
    return mlir::success();
  }
  record(op->getResult(0), MemLayout::NCx, gemm.getResult());
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertConvolution(mlir::linalg::LinalgOp op,
                                          mlir::OpBuilder &builder) {
  if (mlir::failed(verifyExactGemmPayload(op, "ordinary 2-D convolution")))
    return mlir::failure();
  mlir::FailureOr<OrdinaryConv2DGeometry> geometry =
      inferOrdinaryConv2DGeometry(op);
  if (mlir::failed(geometry))
    return fail("convolution indexing maps do not describe one ordinary "
                "static 2-D convolution");

  bool overwriteInit = hasPositiveZeroFilledComputeInit(op);

  auto inputTensor =
      mlir::cast<mlir::RankedTensorType>(op.getDpsInputs()[0].getType());
  auto weightTensor =
      mlir::cast<mlir::RankedTensorType>(op.getDpsInputs()[1].getType());
  auto resultTensor =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::FailureOr<mlir::Value> input = getOrMaterializeStructuredInput(
      op.getDpsInputs()[0], MemLayout::NCx, builder);
  mlir::FailureOr<mlir::Value> weight = getOrMaterializeStructuredInput(
      op.getDpsInputs()[1], MemLayout::NCx, builder);
  if (mlir::failed(input) || mlir::failed(weight))
    return mlir::failure();

  auto transpose =
      [&](mlir::Value source, mlir::RankedTensorType sourceType,
          llvm::ArrayRef<int64_t> permutation) -> mlir::FailureOr<mlir::Value> {
    if (!isPermutation(permutation, sourceType.getRank()))
      return failValue("convolution canonicalization produced an invalid "
                       "permutation");
    mlir::RankedTensorType targetTensor =
        permuteTensorType(sourceType, permutation);
    mlir::Type targetType = makeSPMMemRefType(targetTensor, MemLayout::NCx);
    if (isIdentityPermutation(permutation) && source.getType() == targetType)
      return source;
    return builder
        .create<MoveTransposeOp>(op->getLoc(), targetType, source,
                                 builder.getDenseI64ArrayAttr(permutation))
        .getResult();
  };

  mlir::FailureOr<mlir::Value> canonicalInput =
      transpose(*input, inputTensor, geometry->inputToNHWC);
  mlir::FailureOr<mlir::Value> canonicalWeight =
      transpose(*weight, weightTensor, geometry->weightToXYOI);
  if (mlir::failed(canonicalInput) || mlir::failed(canonicalWeight))
    return mlir::failure();

  mlir::RankedTensorType canonicalResultTensor =
      permuteTensorType(resultTensor, geometry->outputToNHWC);
  auto convolution = builder.create<ComputeConvOp>(
      op->getLoc(), makeSPMMemRefType(canonicalResultTensor, MemLayout::NCx),
      *canonicalInput, *canonicalWeight,
      builder.getDenseI64ArrayAttr(geometry->pads),
      builder.getDenseI64ArrayAttr(geometry->unpads),
      builder.getDenseI64ArrayAttr(geometry->stridesHW),
      builder.getDenseI64ArrayAttr(geometry->dilationsHW));
  recordStructuredComputeOperation(convolution);

  mlir::FailureOr<mlir::Value> sourceOrderedResult = transpose(
      convolution.getResult(), canonicalResultTensor, geometry->outputFromNHWC);
  if (mlir::failed(sourceOrderedResult))
    return mlir::failure();
  if (!overwriteInit) {
    mlir::FailureOr<mlir::Value> combined = createAccumulatorCombine(
        op->getLoc(), ComputeReduceKind::Sum, op.getDpsInits().front(),
        *sourceOrderedResult, resultTensor, builder);
    if (mlir::failed(combined))
      return mlir::failure();
    record(op->getResult(0), MemLayout::Tensor, *combined);
    return mlir::success();
  }
  record(op->getResult(0), MemLayout::NCx, *sourceOrderedResult);
  return mlir::success();
}

std::optional<ComputeElementwiseKind>
TileRegionBodyEmitter::inferCompareKind(mlir::arith::CmpFPredicate predicate) {
  switch (predicate) {
  case mlir::arith::CmpFPredicate::OEQ:
    return ComputeElementwiseKind::Eq;
  case mlir::arith::CmpFPredicate::UNE:
    return ComputeElementwiseKind::Ne;
  case mlir::arith::CmpFPredicate::OLT:
    return ComputeElementwiseKind::Lt;
  case mlir::arith::CmpFPredicate::OLE:
    return ComputeElementwiseKind::Le;
  case mlir::arith::CmpFPredicate::OGT:
    return ComputeElementwiseKind::Gt;
  case mlir::arith::CmpFPredicate::OGE:
    return ComputeElementwiseKind::Ge;
  case mlir::arith::CmpFPredicate::UEQ:
  case mlir::arith::CmpFPredicate::ONE:
  case mlir::arith::CmpFPredicate::ULT:
  case mlir::arith::CmpFPredicate::ULE:
  case mlir::arith::CmpFPredicate::UGT:
  case mlir::arith::CmpFPredicate::UGE:
  case mlir::arith::CmpFPredicate::AlwaysFalse:
  case mlir::arith::CmpFPredicate::ORD:
  case mlir::arith::CmpFPredicate::UNO:
  case mlir::arith::CmpFPredicate::AlwaysTrue:
    (void)fail("arith.cmpf predicate does not match the current target "
               "comparison NaN semantics");
    return std::nullopt;
  }
  llvm_unreachable("unknown cmpf predicate");
}

std::optional<ComputeElementwiseKind>
TileRegionBodyEmitter::inferCompareKind(mlir::arith::CmpIPredicate predicate) {
  switch (predicate) {
  case mlir::arith::CmpIPredicate::eq:
    return ComputeElementwiseKind::Eq;
  case mlir::arith::CmpIPredicate::ne:
    return ComputeElementwiseKind::Ne;
  case mlir::arith::CmpIPredicate::slt:
    return ComputeElementwiseKind::Lt;
  case mlir::arith::CmpIPredicate::sle:
    return ComputeElementwiseKind::Le;
  case mlir::arith::CmpIPredicate::sgt:
    return ComputeElementwiseKind::Gt;
  case mlir::arith::CmpIPredicate::sge:
    return ComputeElementwiseKind::Ge;
  case mlir::arith::CmpIPredicate::ult:
  case mlir::arith::CmpIPredicate::ule:
  case mlir::arith::CmpIPredicate::ugt:
  case mlir::arith::CmpIPredicate::uge:
    (void)fail("unsigned arith.cmpi predicate cannot be represented by the "
               "current signed target comparison kind");
    return std::nullopt;
  }
  llvm_unreachable("unknown cmpi predicate");
}

std::optional<ComputeReduceKind>
TileRegionBodyEmitter::inferReduceKind(mlir::linalg::GenericOp generic) {
  if (generic.getRegionInputArgs().size() != 1 ||
      generic.getRegionOutputArgs().size() != 1) {
    (void)fail("linalg.generic reduction requires one input and one "
               "accumulator");
    return std::nullopt;
  }
  return matchExactReductionKind(generic.getRegionOutputArgs(), /*redPos=*/0,
                                 generic.getRegionInputArgs().front(),
                                 "linalg.generic reduction", failureReason);
}

mlir::FailureOr<mlir::TypedAttr>
TileRegionBodyEmitter::getNeutralReduceInit(mlir::Type elementType,
                                            ComputeReduceKind kind) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    switch (kind) {
    case ComputeReduceKind::Sum:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(floatType, 0.0));
    case ComputeReduceKind::Max:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(
          floatType, llvm::APFloat::getInf(floatType.getFloatSemantics(),
                                           /*Negative=*/true)));
    case ComputeReduceKind::Min:
      return mlir::cast<mlir::TypedAttr>(mlir::FloatAttr::get(
          floatType, llvm::APFloat::getInf(floatType.getFloatSemantics(),
                                           /*Negative=*/false)));
    case ComputeReduceKind::Avg:
      break;
    }
  }

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    unsigned width = intType.getWidth();
    switch (kind) {
    case ComputeReduceKind::Sum:
      return mlir::cast<mlir::TypedAttr>(
          mlir::IntegerAttr::get(intType, llvm::APInt(width, 0)));
    case ComputeReduceKind::Max:
      return mlir::cast<mlir::TypedAttr>(mlir::IntegerAttr::get(
          intType, llvm::APInt::getSignedMinValue(width)));
    case ComputeReduceKind::Min:
      return mlir::cast<mlir::TypedAttr>(mlir::IntegerAttr::get(
          intType, llvm::APInt::getSignedMaxValue(width)));
    case ComputeReduceKind::Avg:
      break;
    }
  }

  (void)fail("reduction accumulator requires a float or integer element type");
  return mlir::failure();
}

bool TileRegionBodyEmitter::hasReductionIterator(
    mlir::linalg::GenericOp generic) const {
  for (mlir::utils::IteratorType iteratorType :
       generic.getIteratorTypesArray()) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      return true;
  }
  return false;
}

mlir::LogicalResult TileRegionBodyEmitter::getReductionInputDims(
    mlir::linalg::GenericOp generic,
    llvm::SmallVectorImpl<int64_t> &inputDims) {
  inputDims.clear();
  if (generic.getNumDpsInputs() != 1)
    return fail("unsupported reduction input arity");

  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps =
      generic.getIndexingMapsArray();
  if (indexingMaps.empty())
    return fail("reduction generic has no indexing map");
  mlir::AffineMap inputMap = indexingMaps[0];

  llvm::SmallVector<unsigned, 4> reductionLoopDims;
  for (auto [index, iteratorType] :
       llvm::enumerate(generic.getIteratorTypesArray())) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      reductionLoopDims.push_back(static_cast<unsigned>(index));
  }
  if (reductionLoopDims.empty())
    return fail("reduction generic has no reduction dimensions");

  for (unsigned loopDim : reductionLoopDims) {
    std::optional<int64_t> inputDim;
    for (auto [dimIndex, expr] : llvm::enumerate(inputMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dimExpr)
        return fail("unsupported reduction input indexing map");
      if (dimExpr.getPosition() == loopDim) {
        inputDim = static_cast<int64_t>(dimIndex);
        break;
      }
    }
    if (!inputDim)
      return fail("reduction dimension is not present in input map");
    inputDims.push_back(*inputDim);
  }
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::createReduceOp(
    mlir::Location loc, mlir::Type resultType, ComputeReduceKindAttr kindAttr,
    mlir::Value input, llvm::ArrayRef<int64_t> dims, mlir::Value init,
    mlir::Attribute initAttr, mlir::OpBuilder &builder, mlir::Value &result) {
  mlir::TypedAttr typedInit = mlir::dyn_cast_or_null<mlir::TypedAttr>(initAttr);
  if (initAttr && !typedInit)
    return fail("reduction init_value must be a typed scalar attribute");
  auto reduce = builder.create<ComputeReduceOp>(
      loc, resultType, kindAttr, input, init,
      builder.getDenseI64ArrayAttr(dims), typedInit);
  recordStructuredComputeOperation(reduce);
  result = reduce.getResult();
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertReduceGeneric(mlir::linalg::GenericOp generic,
                                            mlir::OpBuilder &builder) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1)
    return fail((llvm::Twine("unsupported reduction generic arity: inputs=") +
                 llvm::Twine(generic.getNumDpsInputs()) +
                 ", inits=" + llvm::Twine(generic.getNumDpsInits()) +
                 ", results=" + llvm::Twine(generic->getNumResults()) +
                 ", loops=" + llvm::Twine(generic.getNumLoops()))
                    .str());

  std::optional<ComputeReduceKind> kind = inferReduceKind(generic);
  if (!kind)
    return mlir::failure();

  llvm::SmallVector<int64_t, 4> reduceDims;
  if (mlir::failed(getReductionInputDims(generic, reduceDims)))
    return mlir::failure();

  mlir::Value initTensor = generic.getDpsInits()[0];
  mlir::Value initScalar;
  mlir::Attribute initAttr;
  bool explicitAccumulatorCombine = false;

  auto inputTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
      generic.getDpsInputs()[0].getType());
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(generic->getResult(0).getType());
  if (!inputTensorType || !resultTensorType)
    return fail("reduction generic operands/results must be ranked tensors");

  if (auto attrIt = fillInitAttrs.find(initTensor);
      attrIt != fillInitAttrs.end()) {
    initAttr = attrIt->second;
  } else if (auto scalarIt = fillInitScalars.find(initTensor);
             scalarIt != fillInitScalars.end()) {
    initScalar = scalarIt->second;
    auto constant = initScalar.getDefiningOp<mlir::arith::ConstantOp>();
    auto typedValue = constant
                          ? mlir::dyn_cast<mlir::TypedAttr>(constant.getValue())
                          : mlir::TypedAttr{};
    if (!constant || !typedValue)
      return fail(
          "reduction fill init must be an arith.constant or typed init_value");
  } else {
    // An arbitrary DPS init (including an scf iter_arg) is ordinary Linalg
    // semantics.  Compute the reduction from its neutral element, then
    // combine that partial with the materialized init using the exact body
    // combiner.  This avoids recovering accumulator roles from op names or
    // defining-operation shapes.
    mlir::FailureOr<mlir::TypedAttr> neutral =
        getNeutralReduceInit(resultTensorType.getElementType(), *kind);
    if (mlir::failed(neutral))
      return mlir::failure();
    initAttr = *neutral;
    explicitAccumulatorCombine = true;
  }

  mlir::FailureOr<mlir::Value> input = getOrMaterializeStructuredInput(
      generic.getDpsInputs()[0], alignedLayoutForTensor(inputTensorType),
      builder);
  if (mlir::failed(input))
    return mlir::failure();

  mlir::Type reduceResultType = makeSPMMemRefType(
      resultTensorType, alignedLayoutForTensor(resultTensorType));
  mlir::Value reduceResult;
  auto kindAttr = ComputeReduceKindAttr::get(generic.getContext(), *kind);
  if (mlir::failed(createReduceOp(generic.getLoc(), reduceResultType, kindAttr,
                                  *input, reduceDims, initScalar, initAttr,
                                  builder, reduceResult)))
    return mlir::failure();

  if (explicitAccumulatorCombine) {
    mlir::FailureOr<mlir::Value> combined =
        createAccumulatorCombine(generic.getLoc(), *kind, initTensor,
                                 reduceResult, resultTensorType, builder);
    if (mlir::failed(combined))
      return mlir::failure();
    record(generic->getResult(0), MemLayout::Tensor, *combined);
    return mlir::success();
  }

  record(generic->getResult(0), alignedLayoutForTensor(resultTensorType),
         reduceResult);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
