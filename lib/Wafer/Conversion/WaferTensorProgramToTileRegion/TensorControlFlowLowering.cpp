//===- TensorControlFlowLowering.cpp - Tensor/control-flow lowering -===//

#include "Internal.h"

#include <functional>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<mlir::Type>
TileRegionBodyEmitter::convertControlFlowType(mlir::Type type) {
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return makeSPMMemRefType(tensorType, MemLayout::Tensor);
  if (isScalarType(type))
    return type;
  return failType("control-flow value is not a ranked tensor or scalar");
}

mlir::LogicalResult
TileRegionBodyEmitter::recordControlFlowValue(mlir::Value original,
                                              mlir::Value converted) {
  if (mlir::isa<mlir::RankedTensorType>(original.getType())) {
    record(original, MemLayout::Tensor, converted);
    return mlir::success();
  }
  if (isScalarType(original.getType())) {
    scalarValues[original] = converted;
    return mlir::success();
  }
  return fail("control-flow value is not a ranked tensor or scalar");
}

mlir::FailureOr<mlir::Value>
TileRegionBodyEmitter::materializeControlFlowValue(mlir::Value original,
                                                   mlir::OpBuilder &builder) {
  if (mlir::isa<mlir::RankedTensorType>(original.getType()))
    return getOrMaterialize(original, MemLayout::Tensor, builder);
  if (isScalarType(original.getType()))
    return getScalarValue(original);
  return failValue("control-flow yield is not a ranked tensor or scalar");
}

mlir::LogicalResult
TileRegionBodyEmitter::convertSupportOp(mlir::Operation *op,
                                        mlir::OpBuilder &builder) {
  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    if (constant->getNumResults() == 0)
      return mlir::success();

    mlir::Value originalResult = constant.getResult();
    if (onlyFeedsUnreadDpsInit(originalResult))
      return mlir::success();
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(originalResult.getType());
    if (!tensorType) {
      mlir::Operation *cloned = builder.clone(*constant.getOperation());
      scalarValues[originalResult] = cloned->getResult(0);
      scalarAttrs[originalResult] = constant.getValue();
      return mlir::success();
    }

    if (getScalarSplatAttr(tensorType, constant.getValue())) {
      mlir::FailureOr<mlir::Value> materialized = materializeTensorConstant(
          originalResult, constant.getValue(), MemLayout::Tensor, builder);
      if (mlir::failed(materialized))
        return mlir::failure();
      tensorAttrs[originalResult] = constant.getValue();
      return mlir::success();
    }

    mlir::Operation *cloned = builder.clone(*constant.getOperation());
    auto ddr = builder.create<mlir::bufferization::ToMemrefOp>(
        constant.getLoc(), makeDDRMemRefType(tensorType), cloned->getResult(0),
        /*read_only=*/true);
    auto load = builder.create<StorageLoadOp>(
        constant.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor),
        ddr.getMemref());
    record(originalResult, MemLayout::Tensor, load.getResult());
    externalBuffers[originalResult] = ddr.getMemref();
    tensorAttrs[originalResult] = constant.getValue();
    return mlir::success();
  }

  if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(op)) {
    if (onlyFeedsUnreadDpsInit(empty.getResult()))
      return mlir::success();
    bool onlyFeedsSlices = !empty.getResult().use_empty();
    for (mlir::OpOperand &use : empty.getResult().getUses()) {
      auto extractSlice =
          mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
      if (!extractSlice || extractSlice.getSource() != empty.getResult()) {
        onlyFeedsSlices = false;
        break;
      }
    }
    // tensor.empty carries no observable contents.  When traversal tiling has
    // replaced every use with a tile slice, materialize each live tile below
    // instead of allocating the original full tensor in SPM.
    if (onlyFeedsSlices)
      return mlir::success();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(empty.getType());
    if (!tensorType)
      return fail("tensor.empty result is not a ranked tensor");
    auto alloc = builder.create<mlir::memref::AllocOp>(
        empty.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor));
    record(empty.getResult(), MemLayout::Tensor, alloc.getResult());
    return mlir::success();
  }

  if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractOp>(op))
    return convertTensorExtract(extract, builder);
  if (auto extractSlice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op))
    return convertTensorExtractSlice(extractSlice, builder);
  if (auto insertSlice = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op))
    return convertTensorInsertSlice(insertSlice, builder);
  if (auto expandShape = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op))
    return convertTensorReshape(expandShape.getOperation(),
                                expandShape.getSrc(), expandShape.getResult(),
                                builder);
  if (auto collapseShape = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op))
    return convertTensorReshape(collapseShape.getOperation(),
                                collapseShape.getSrc(),
                                collapseShape.getResult(), builder);
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return convertScfIf(ifOp, builder);
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
    return convertScfFor(forOp, builder);

  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertNestedOp(mlir::Operation *op,
                                       mlir::OpBuilder &builder) {
  if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(op))
    return fail("nested collective materialization is not implemented");
  if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op))
    return convertFill(fill, builder);
  if (mlir::isa<mlir::linalg::MatmulOp>(op))
    return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder);
  if (mlir::isa<mlir::linalg::BatchMatmulOp>(op))
    return convertBatchMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder);
  if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
    return convertGeneric(generic, builder);
  if (mlir::isa<mlir::arith::ConstantOp, mlir::tensor::EmptyOp,
                mlir::tensor::ExtractOp, mlir::tensor::ExtractSliceOp,
                mlir::tensor::InsertSliceOp, mlir::tensor::ExpandShapeOp,
                mlir::tensor::CollapseShapeOp, mlir::scf::IfOp,
                mlir::scf::ForOp>(op))
    return convertSupportOp(op, builder);
  return fail("unsupported op inside structured control-flow " +
              op->getName().getStringRef().str());
}

mlir::LogicalResult
TileRegionBodyEmitter::convertScfYield(mlir::scf::YieldOp yield,
                                       mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 4> yielded;
  auto convertedFor = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
      builder.getInsertionBlock() ? builder.getInsertionBlock()->getParentOp()
                                  : nullptr);
  for (auto [index, value] : llvm::enumerate(yield.getResults())) {
    if (convertedFor && index < convertedFor.getRegionIterArgs().size() &&
        isWaferDDRMemRefType(
            convertedFor.getRegionIterArgs()[index].getType())) {
      auto external = externalBuffers.find(value);
      if (external == externalBuffers.end())
        return fail("external loop yield has no DDR buffer version");
      yielded.push_back(external->second);
      continue;
    }
    mlir::FailureOr<mlir::Value> converted =
        materializeControlFlowValue(value, builder);
    if (mlir::failed(converted))
      return mlir::failure();
    yielded.push_back(*converted);
  }
  builder.create<mlir::scf::YieldOp>(yield.getLoc(), yielded);
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertScfBlock(mlir::Block &source,
                                       mlir::OpBuilder &builder) {
  for (mlir::Operation &op : source.without_terminator()) {
    if (mlir::failed(convertNestedOp(&op, builder)))
      return mlir::failure();
  }
  auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(source.getTerminator());
  if (!yield)
    return fail("structured control-flow body must terminate with scf.yield");
  return convertScfYield(yield, builder);
}

void TileRegionBodyEmitter::eraseImplicitYield(mlir::Block *block) {
  if (!block || block->empty())
    return;
  if (mlir::isa<mlir::scf::YieldOp>(block->back()))
    block->back().erase();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertScfIf(mlir::scf::IfOp ifOp,
                                    mlir::OpBuilder &builder) {
  mlir::FailureOr<mlir::Value> condition = getScalarValue(ifOp.getCondition());
  if (mlir::failed(condition))
    return mlir::failure();

  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (mlir::Type type : ifOp->getResultTypes()) {
    mlir::FailureOr<mlir::Type> converted = convertControlFlowType(type);
    if (mlir::failed(converted))
      return mlir::failure();
    resultTypes.push_back(*converted);
  }

  bool hasElse = !ifOp.getElseRegion().empty();
  auto convertedIf = builder.create<mlir::scf::IfOp>(ifOp.getLoc(), resultTypes,
                                                     *condition, hasElse);

  {
    StateSnapshot outer = snapshotState();
    mlir::Block *thenBlock = convertedIf.thenBlock();
    eraseImplicitYield(thenBlock);
    mlir::OpBuilder thenBuilder(thenBlock, thenBlock->end());
    if (mlir::failed(convertScfBlock(*ifOp.thenBlock(), thenBuilder)))
      return mlir::failure();
    restoreState(outer);
  }

  if (hasElse) {
    StateSnapshot outer = snapshotState();
    mlir::Block *elseBlock = convertedIf.elseBlock();
    eraseImplicitYield(elseBlock);
    mlir::OpBuilder elseBuilder(elseBlock, elseBlock->end());
    if (mlir::failed(convertScfBlock(*ifOp.elseBlock(), elseBuilder)))
      return mlir::failure();
    restoreState(outer);
  }

  for (auto [original, converted] :
       llvm::zip(ifOp->getResults(), convertedIf->getResults())) {
    if (mlir::failed(recordControlFlowValue(original, converted)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertScfFor(mlir::scf::ForOp forOp,
                                     mlir::OpBuilder &builder) {
  mlir::FailureOr<mlir::Value> lowerBound =
      getScalarValue(forOp.getLowerBound());
  mlir::FailureOr<mlir::Value> upperBound =
      getScalarValue(forOp.getUpperBound());
  mlir::FailureOr<mlir::Value> step = getScalarValue(forOp.getStep());
  if (mlir::failed(lowerBound) || mlir::failed(upperBound) ||
      mlir::failed(step))
    return mlir::failure();

  struct ExternalCarry {
    bool external = false;
    bool writable = false;
    std::optional<unsigned> outputIndex;
    mlir::Value baseBuffer;
  };
  llvm::SmallVector<mlir::Value, 4> initArgs;
  llvm::SmallVector<ExternalCarry, 4> externalCarries;
  auto sourceYield =
      mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
  auto tracesExternalDestination = [&](mlir::Value value,
                                       mlir::Value expectedDestination) {
    llvm::DenseSet<mlir::Value> visited;
    std::function<bool(mlir::Value, mlir::Value)> trace =
        [&](mlir::Value current, mlir::Value expected) -> bool {
      if (current == expected)
        return true;
      if (!visited.insert(current).second)
        return false;
      if (auto insert = current.getDefiningOp<mlir::tensor::InsertSliceOp>())
        return trace(insert.getDest(), expected);
      auto nestedFor = current.getDefiningOp<mlir::scf::ForOp>();
      auto result = mlir::dyn_cast<mlir::OpResult>(current);
      if (!nestedFor || !result ||
          result.getResultNumber() >= nestedFor.getNumRegionIterArgs())
        return false;
      unsigned index = result.getResultNumber();
      auto nestedYield =
          mlir::cast<mlir::scf::YieldOp>(nestedFor.getBody()->getTerminator());
      return trace(nestedFor.getInitArgs()[index], expected) &&
             trace(nestedYield.getResults()[index],
                   nestedFor.getRegionIterArgs()[index]);
    };
    return trace(value, expectedDestination);
  };

  for (auto [index, init] : llvm::enumerate(forOp.getInitArgs())) {
    ExternalCarry carry;
    if (auto external = externalBuffers.find(init);
        external != externalBuffers.end() &&
        writableExternalBuffers.contains(init) &&
        externalOutputIndices.contains(init) &&
        tracesExternalDestination(sourceYield.getResults()[index],
                                  forOp.getRegionIterArgs()[index])) {
      initArgs.push_back(external->second);
      carry.external = true;
      carry.writable = writableExternalBuffers.contains(init);
      carry.baseBuffer = external->second;
      if (auto base = directYieldBuffers.find(init);
          base != directYieldBuffers.end())
        carry.baseBuffer = base->second;
      if (auto index = externalOutputIndices.find(init);
          index != externalOutputIndices.end())
        carry.outputIndex = index->second;
      externalCarries.push_back(carry);
      continue;
    }
    mlir::FailureOr<mlir::Value> converted =
        materializeControlFlowValue(init, builder);
    if (mlir::failed(converted))
      return mlir::failure();
    initArgs.push_back(*converted);
    externalCarries.push_back(carry);
  }

  auto convertedFor = builder.create<mlir::scf::ForOp>(
      forOp.getLoc(), *lowerBound, *upperBound, *step, initArgs);
  eraseImplicitYield(convertedFor.getBody());

  StateSnapshot outer = snapshotState();
  scalarValues[forOp.getInductionVar()] = convertedFor.getInductionVar();
  for (auto [original, converted, carry] :
       llvm::zip(forOp.getRegionIterArgs(), convertedFor.getRegionIterArgs(),
                 externalCarries)) {
    if (carry.external) {
      externalBuffers[original] = converted;
      if (carry.writable)
        writableExternalBuffers.insert(original);
      if (carry.outputIndex)
        externalOutputIndices[original] = *carry.outputIndex;
      directYieldBuffers[original] = carry.baseBuffer;
    } else if (mlir::failed(recordControlFlowValue(original, converted))) {
      return mlir::failure();
    }
  }

  mlir::OpBuilder bodyBuilder(convertedFor.getBody(),
                              convertedFor.getBody()->end());
  if (mlir::failed(convertScfBlock(*forOp.getBody(), bodyBuilder)))
    return mlir::failure();
  restoreState(outer);

  for (auto [original, converted, carry] : llvm::zip(
           forOp->getResults(), convertedFor->getResults(), externalCarries)) {
    if (carry.external) {
      externalBuffers[original] = converted;
      if (carry.writable)
        writableExternalBuffers.insert(original);
      if (carry.outputIndex)
        externalOutputIndices[original] = *carry.outputIndex;
      directYieldBuffers[original] = carry.baseBuffer;
    } else if (mlir::failed(recordControlFlowValue(original, converted))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertTensorExtract(mlir::tensor::ExtractOp extract,
                                            mlir::OpBuilder &builder) {
  auto bufferIt = externalBuffers.find(extract.getTensor());
  if (bufferIt == externalBuffers.end())
    return fail("tensor.extract from tile-local tensor is not representable");

  llvm::SmallVector<mlir::Value, 4> indices;
  for (mlir::Value index : extract.getIndices()) {
    auto scalarIt = scalarValues.find(index);
    if (scalarIt == scalarValues.end())
      return fail("missing index value for tensor.extract");
    indices.push_back(scalarIt->second);
  }

  auto load = builder.create<mlir::memref::LoadOp>(extract.getLoc(),
                                                   bufferIt->second, indices);
  scalarValues[extract.getResult()] = load.getResult();
  return mlir::success();
}

bool TileRegionBodyEmitter::allStatic(llvm::ArrayRef<int64_t> values) const {
  return llvm::all_of(values, [](int64_t value) {
    return value != mlir::ShapedType::kDynamic;
  });
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::materializeDdrSubview(
    mlir::Location loc, mlir::Value sourceDdr,
    mlir::RankedTensorType tileTensorType,
    llvm::ArrayRef<mlir::OpFoldResult> offsets, llvm::ArrayRef<int64_t> sizes,
    llvm::ArrayRef<int64_t> strides, mlir::OpBuilder &builder) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(sourceDdr.getType());
  if (!sourceType)
    return failValue("external tile view source is not a memref");
  if (sourceType.getElementType() != tileTensorType.getElementType())
    return failValue("external tile view element type mismatch");
  if (sourceType.getRank() != static_cast<int64_t>(offsets.size()) ||
      sourceType.getRank() != static_cast<int64_t>(sizes.size()) ||
      sourceType.getRank() != static_cast<int64_t>(strides.size()))
    return failValue("external tile view rank mismatch");

  llvm::SmallVector<mlir::OpFoldResult, 4> convertedOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedSizes;
  llvm::SmallVector<mlir::OpFoldResult, 4> mixedStrides;
  convertedOffsets.reserve(offsets.size());
  for (mlir::OpFoldResult offset : offsets) {
    if (auto value = mlir::dyn_cast<mlir::Value>(offset)) {
      mlir::FailureOr<mlir::Value> converted = getScalarValue(value);
      if (mlir::failed(converted))
        return mlir::failure();
      convertedOffsets.push_back(*converted);
    } else {
      convertedOffsets.push_back(offset);
    }
  }
  for (int64_t size : sizes)
    mixedSizes.push_back(builder.getIndexAttr(size));
  for (int64_t stride : strides)
    mixedStrides.push_back(builder.getIndexAttr(stride));

  auto subviewType = mlir::cast<mlir::MemRefType>(
      mlir::memref::SubViewOp::inferRankReducedResultType(
          tileTensorType.getShape(), sourceType, convertedOffsets, mixedSizes,
          mixedStrides));
  auto subview = builder.create<mlir::memref::SubViewOp>(
      loc, subviewType, sourceDdr, convertedOffsets, mixedSizes, mixedStrides);
  return subview.getResult();
}

std::optional<unsigned>
TileRegionBodyEmitter::getSingleTensorProgramReturnOperandIndex(
    mlir::Value value) const {
  if (!value.hasOneUse())
    return std::nullopt;
  mlir::OpOperand &use = *value.getUses().begin();
  if (!mlir::isa<mlir::func::ReturnOp>(use.getOwner()))
    return std::nullopt;
  return use.getOperandNumber();
}

bool TileRegionBodyEmitter::isLinearInsertChainToTensorProgramReturn(
    mlir::tensor::InsertSliceOp insertSlice,
    unsigned expectedOutputIndex) const {
  mlir::Value current = insertSlice.getResult();
  while (current.hasOneUse()) {
    mlir::OpOperand &use = *current.getUses().begin();
    if (mlir::isa<mlir::func::ReturnOp>(use.getOwner()))
      return use.getOperandNumber() == expectedOutputIndex;

    auto nextInsert =
        mlir::dyn_cast<mlir::tensor::InsertSliceOp>(use.getOwner());
    if (!nextInsert || nextInsert.getDest() != current)
      return false;
    current = nextInsert.getResult();
  }
  return false;
}

bool TileRegionBodyEmitter::hasNoObservableDestUseExceptInsert(
    mlir::tensor::InsertSliceOp insertSlice) const {
  mlir::Value dest = insertSlice.getDest();
  mlir::OpOperand *destOperand = &insertSlice->getOpOperand(1);
  for (mlir::OpOperand &use : dest.getUses()) {
    if (&use == destOperand)
      continue;
    if (isUnreadLinalgDpsInitUse(use))
      continue;
    auto extractSlice =
        mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
    if (extractSlice && extractSlice.getSource() == dest &&
        onlyFeedsUnreadDpsInit(extractSlice.getResult()))
      continue;
    return false;
  }
  return true;
}

mlir::LogicalResult TileRegionBodyEmitter::convertTensorExtractSlice(
    mlir::tensor::ExtractSliceOp extractSlice, mlir::OpBuilder &builder) {
  if (!allStatic(extractSlice.getStaticSizes()) ||
      !allStatic(extractSlice.getStaticStrides()))
    return fail(
        "dynamic tensor.extract_slice size/stride is not representable");

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(extractSlice.getType());
  if (!resultTensorType)
    return fail("tensor.extract_slice result is not a ranked tensor");

  if (onlyFeedsUnreadDpsInit(extractSlice.getResult()))
    return mlir::success();

  // A slice of tensor.empty is itself an undefined tile.  Allocate only that
  // tile in SPM; compact traversal offsets may be dynamic even though the tile
  // shape is static.
  if (extractSlice.getSource().getDefiningOp<mlir::tensor::EmptyOp>()) {
    auto alloc = builder.create<mlir::memref::AllocOp>(
        extractSlice.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor));
    record(extractSlice.getResult(), MemLayout::Tensor, alloc.getResult());
    return mlir::success();
  }

  bool hasFillInitMarker = fillInitAttrs.contains(extractSlice.getSource()) ||
                           fillInitScalars.contains(extractSlice.getSource());
  if (hasFillInitMarker &&
      onlyFeedsGemmOverwriteInit(extractSlice.getResult())) {
    if (auto attrIt = fillInitAttrs.find(extractSlice.getSource());
        attrIt != fillInitAttrs.end())
      fillInitAttrs[extractSlice.getResult()] = attrIt->second;
    if (auto scalarIt = fillInitScalars.find(extractSlice.getSource());
        scalarIt != fillInitScalars.end())
      fillInitScalars[extractSlice.getResult()] = scalarIt->second;
    return mlir::success();
  }

  if (auto attrIt = tensorAttrs.find(extractSlice.getSource());
      attrIt != tensorAttrs.end() &&
      getScalarSplatAttr(resultTensorType, attrIt->second)) {
    tensorAttrs[extractSlice.getResult()] = attrIt->second;
    return mlir::success();
  }

  if (auto externalIt = externalBuffers.find(extractSlice.getSource());
      externalIt != externalBuffers.end()) {
    mlir::FailureOr<mlir::Value> tileView = materializeDdrSubview(
        extractSlice.getLoc(), externalIt->second, resultTensorType,
        extractSlice.getMixedOffsets(), extractSlice.getStaticSizes(),
        extractSlice.getStaticStrides(), builder);
    if (mlir::failed(tileView))
      return mlir::failure();

    auto load = builder.create<StorageLoadOp>(
        extractSlice.getLoc(),
        makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *tileView);
    record(extractSlice.getResult(), MemLayout::Tensor, load.getResult());
    return mlir::success();
  }

  if (!allStatic(extractSlice.getStaticOffsets()))
    return fail("dynamic tile-local tensor.extract_slice is not representable");

  mlir::FailureOr<mlir::Value> source =
      getOrMaterialize(extractSlice.getSource(), MemLayout::Tensor, builder);
  if (mlir::failed(source))
    return mlir::failure();

  mlir::MLIRContext *context = extractSlice.getContext();
  auto offsets =
      mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticOffsets());
  auto sizes =
      mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticSizes());
  auto strides =
      mlir::DenseI64ArrayAttr::get(context, extractSlice.getStaticStrides());
  auto move = builder.create<MoveExtractSliceOp>(
      extractSlice.getLoc(),
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *source, offsets,
      sizes, strides);
  record(extractSlice.getResult(), MemLayout::Tensor, move.getResult());
  if (auto attrIt = fillInitAttrs.find(extractSlice.getSource());
      attrIt != fillInitAttrs.end())
    fillInitAttrs[extractSlice.getResult()] = attrIt->second;
  if (auto scalarIt = fillInitScalars.find(extractSlice.getSource());
      scalarIt != fillInitScalars.end())
    fillInitScalars[extractSlice.getResult()] = scalarIt->second;
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertTensorInsertSlice(
    mlir::tensor::InsertSliceOp insertSlice, mlir::OpBuilder &builder) {
  if (!allStatic(insertSlice.getStaticSizes()) ||
      !allStatic(insertSlice.getStaticStrides()))
    return fail("dynamic tensor.insert_slice size/stride is not representable");

  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getType());
  if (!resultTensorType)
    return fail("tensor.insert_slice result is not a ranked tensor");
  auto sourceTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(insertSlice.getSourceType());
  if (!sourceTensorType)
    return fail("tensor.insert_slice source is not a ranked tensor");

  auto externalIt = externalBuffers.find(insertSlice.getDest());
  auto outputIndexIt = externalOutputIndices.find(insertSlice.getDest());
  std::optional<unsigned> yieldIndex =
      getSingleTensorProgramReturnOperandIndex(insertSlice.getResult());
  bool isDirectYield = outputIndexIt != externalOutputIndices.end() &&
                       yieldIndex && outputIndexIt->second == *yieldIndex;
  bool isLinearInsertChain = outputIndexIt != externalOutputIndices.end() &&
                             isLinearInsertChainToTensorProgramReturn(
                                 insertSlice, outputIndexIt->second);
  bool isLoopYield = insertSlice.getResult().hasOneUse() &&
                     mlir::isa<mlir::scf::YieldOp>(
                         insertSlice.getResult().use_begin()->getOwner());
  if (externalIt != externalBuffers.end() &&
      writableExternalBuffers.contains(insertSlice.getDest()) &&
      outputIndexIt != externalOutputIndices.end() &&
      hasNoObservableDestUseExceptInsert(insertSlice) &&
      (isDirectYield || isLinearInsertChain || isLoopYield)) {
    mlir::Value externalBuffer = externalIt->second;
    unsigned outputIndex = outputIndexIt->second;
    mlir::FailureOr<mlir::Value> source =
        getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
    if (mlir::failed(source))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> tileView = materializeDdrSubview(
        insertSlice.getLoc(), externalBuffer, sourceTensorType,
        insertSlice.getMixedOffsets(), insertSlice.getStaticSizes(),
        insertSlice.getStaticStrides(), builder);
    if (mlir::failed(tileView))
      return mlir::failure();

    builder.create<StorageStoreOp>(insertSlice.getLoc(), *source, *tileView);
    mlir::Value result = insertSlice.getResult();
    externalBuffers[result] = externalBuffer;
    writableExternalBuffers.insert(result);
    externalOutputIndices[result] = outputIndex;
    auto base = directYieldBuffers.find(insertSlice.getDest());
    directYieldBuffers[result] =
        base == directYieldBuffers.end() ? externalBuffer : base->second;
    return mlir::success();
  }

  if (!allStatic(insertSlice.getStaticOffsets()))
    return fail("dynamic tile-local tensor.insert_slice is not representable");

  mlir::FailureOr<mlir::Value> source =
      getOrMaterialize(insertSlice.getSource(), MemLayout::Tensor, builder);
  mlir::FailureOr<mlir::Value> dest =
      getOrMaterialize(insertSlice.getDest(), MemLayout::Tensor, builder);
  if (mlir::failed(source) || mlir::failed(dest))
    return mlir::failure();

  mlir::MLIRContext *context = insertSlice.getContext();
  auto offsets =
      mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticOffsets());
  auto sizes =
      mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticSizes());
  auto strides =
      mlir::DenseI64ArrayAttr::get(context, insertSlice.getStaticStrides());
  auto move = builder.create<MoveInsertSliceOp>(
      insertSlice.getLoc(),
      makeSPMMemRefType(resultTensorType, MemLayout::Tensor), *source, *dest,
      offsets, sizes, strides);
  record(insertSlice.getResult(), MemLayout::Tensor, move.getResult());
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertTensorReshape(
    mlir::Operation *op, mlir::Value sourceValue, mlir::Value resultValue,
    mlir::OpBuilder &builder) {
  auto resultTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(resultValue.getType());
  if (!resultTensorType)
    return fail("tensor reshape result is not a ranked tensor");

  if (auto external = externalBuffers.find(sourceValue);
      external != externalBuffers.end()) {
    mlir::Value view;
    mlir::MemRefType resultType = makeDDRMemRefType(resultTensorType);
    if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(op)) {
      view = builder
                 .create<mlir::memref::CollapseShapeOp>(
                     op->getLoc(), resultType, external->second,
                     collapse.getReassociationIndices())
                 .getResult();
    } else if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(op)) {
      view = builder
                 .create<mlir::memref::ExpandShapeOp>(
                     op->getLoc(), resultType, external->second,
                     expand.getReassociationIndices())
                 .getResult();
    } else {
      return fail("unsupported external tensor reshape operation");
    }
    externalBuffers[resultValue] = view;
    if (writableExternalBuffers.contains(sourceValue))
      writableExternalBuffers.insert(resultValue);
    if (auto index = externalOutputIndices.find(sourceValue);
        index != externalOutputIndices.end())
      externalOutputIndices[resultValue] = index->second;
    if (auto base = directYieldBuffers.find(sourceValue);
        base != directYieldBuffers.end())
      directYieldBuffers[resultValue] = base->second;
    return mlir::success();
  }

  MemLayout sourceLayout = MemLayout::Tensor;
  mlir::Value source = lookupAny(sourceValue, sourceLayout);
  if (!source) {
    mlir::FailureOr<mlir::Value> loaded =
        getOrMaterialize(sourceValue, MemLayout::Tensor, builder);
    if (mlir::failed(loaded))
      return mlir::failure();
    source = *loaded;
    sourceLayout = MemLayout::Tensor;
  }

  auto reshape = builder.create<ViewReshapeOp>(
      op->getLoc(), makeSPMMemRefType(resultTensorType, sourceLayout), source);
  record(resultValue, sourceLayout, reshape.getResult());
  return mlir::success();
}

mlir::FailureOr<mlir::Value>
TileRegionBodyEmitter::getScalarValue(mlir::Value original) {
  auto it = scalarValues.find(original);
  if (it == scalarValues.end())
    return failValue("missing scalar value for tile compute");
  return it->second;
}

bool TileRegionBodyEmitter::isUnreadLinalgDpsInitUse(
    mlir::OpOperand &use) const {
  auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(use.getOwner());
  if (!linalgOp)
    return false;
  llvm::ArrayRef<mlir::BlockArgument> outputArgs =
      linalgOp.getRegionOutputArgs();
  for (int64_t index = 0; index < linalgOp.getNumDpsInits(); ++index) {
    if (linalgOp.getDpsInitOperand(index) != &use)
      continue;
    return static_cast<size_t>(index) < outputArgs.size() &&
           outputArgs[index].use_empty();
  }
  return false;
}

bool TileRegionBodyEmitter::onlyFeedsUnreadDpsInit(
    mlir::Value value, llvm::DenseSet<mlir::Value> &visited) const {
  if (value.use_empty() || !visited.insert(value).second)
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    if (isUnreadLinalgDpsInitUse(use))
      continue;
    auto extractSlice =
        mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
    if (!extractSlice || extractSlice.getSource() != value ||
        !onlyFeedsUnreadDpsInit(extractSlice.getResult(), visited))
      return false;
  }
  return true;
}

bool TileRegionBodyEmitter::onlyFeedsUnreadDpsInit(mlir::Value value) const {
  llvm::DenseSet<mlir::Value> visited;
  return onlyFeedsUnreadDpsInit(value, visited);
}

bool TileRegionBodyEmitter::onlyFeedsGemmOverwriteInit(
    mlir::Value value, llvm::DenseSet<mlir::Value> &visited) const {
  if (value.use_empty() || !visited.insert(value).second)
    return false;

  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp>(owner)) {
      auto dpsOp = mlir::cast<mlir::linalg::LinalgOp>(owner);
      if (dpsOp.isDpsInit(&use))
        continue;
      return false;
    }

    auto extractSlice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(owner);
    if (!extractSlice || extractSlice.getSource() != value ||
        !onlyFeedsGemmOverwriteInit(extractSlice.getResult(), visited))
      return false;
  }
  return true;
}

bool TileRegionBodyEmitter::onlyFeedsGemmOverwriteInit(
    mlir::Value value) const {
  llvm::DenseSet<mlir::Value> visited;
  return onlyFeedsGemmOverwriteInit(value, visited);
}

} // namespace wafer::tensor_program_to_tile_region
