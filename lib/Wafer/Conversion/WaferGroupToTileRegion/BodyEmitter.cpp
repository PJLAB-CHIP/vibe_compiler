//===- BodyEmitter.cpp - Group body lowering orchestration -----------===//

#include "Internal.h"

using namespace wafer;

namespace wafer::group_to_tile_region {

void setFailureReason(std::string *failureReason, llvm::StringRef reason) {
  if (failureReason)
    *failureReason = reason.str();
}

std::optional<ComputeReduceKind>
matchExactReductionKind(llvm::ArrayRef<mlir::BlockArgument> iterCarriedArgs,
                        unsigned redPos, mlir::Value expectedReducedValue,
                        llvm::StringRef subject, std::string *failureReason) {
  llvm::SmallVector<mlir::Operation *, 1> combinerOps;
  mlir::Value reducedValue =
      mlir::matchReduction(iterCarriedArgs, redPos, combinerOps);
  if (!reducedValue || reducedValue != expectedReducedValue ||
      combinerOps.size() != 1) {
    setFailureReason(failureReason,
                     (subject +
                      " requires one exact combiner wired to the reduced value "
                      "and accumulator")
                         .str());
    return std::nullopt;
  }

  mlir::Operation *combiner = combinerOps.front();
  mlir::Block *combinerBlock = combiner->getBlock();
  if (!combinerBlock ||
      !llvm::all_of(combinerBlock->without_terminator(),
                    [&](mlir::Operation &op) { return &op == combiner; })) {
    setFailureReason(
        failureReason,
        (subject + " cannot erase additional reduction payload operations")
            .str());
    return std::nullopt;
  }
  // Linalg reductions and Wafer collectives both permit an
  // implementation-selected reduction tree. Choosing that tree is distinct
  // from reassociating an ordinary scalar expression.
  if (mlir::isa<mlir::arith::AddFOp>(combiner))
    return ComputeReduceKind::Sum;
  if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(combiner)) {
    if (addi.getOverflowFlags() != mlir::arith::IntegerOverflowFlags::none) {
      setFailureReason(
          failureReason,
          (subject + " cannot preserve integer overflow flags").str());
      return std::nullopt;
    }
    return ComputeReduceKind::Sum;
  }
  if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MaxSIOp>(combiner))
    return ComputeReduceKind::Max;
  if (mlir::isa<mlir::arith::MinimumFOp, mlir::arith::MinSIOp>(combiner))
    return ComputeReduceKind::Min;

  if (mlir::isa<mlir::arith::MaxNumFOp, mlir::arith::MinNumFOp>(combiner)) {
    setFailureReason(failureReason,
                     (subject +
                      " cannot preserve maxnum/minnum NaN semantics with the "
                      "current reduce kind")
                         .str());
    return std::nullopt;
  }
  if (mlir::isa<mlir::arith::MaxUIOp, mlir::arith::MinUIOp>(combiner)) {
    setFailureReason(failureReason,
                     (subject +
                      " cannot preserve unsigned min/max semantics with the "
                      "current reduce kind")
                         .str());
    return std::nullopt;
  }

  setFailureReason(failureReason,
                   (subject + " requires an exact sum, signed min/max, or IEEE "
                              "minimum/maximum combiner")
                       .str());
  return std::nullopt;
}

TileRegionBodyEmitter::TileRegionBodyEmitter(std::string *failureReason,
                                             int64_t currentLogicalRank)
    : failureReason(failureReason), currentLogicalRank(currentLogicalRank) {}

mlir::FailureOr<TileRegionOp>
TileRegionBodyEmitter::emit(GroupOp group, mlir::ValueRange convertedInputs,
                            mlir::ValueRange convertedOuts,
                            mlir::ConversionPatternRewriter &rewriter) {
  if (currentLogicalRank < 0)
    return failAndReturn("logical-rank must be non-negative");

  llvm::DenseSet<mlir::Value> boundaryValues;
  for (mlir::Value input : group.getInputs()) {
    if (!boundaryValues.insert(input).second)
      return failAndReturn("group boundary SSA values must be unique");
  }
  for (mlir::Value out : group.getOuts()) {
    if (!boundaryValues.insert(out).second)
      return failAndReturn("group boundary SSA values must be unique");
  }
  if (mlir::failed(verifyNamedLinalgPayloads(group)))
    return mlir::failure();

  GroupLayoutPlan layoutPlan;
  if (mlir::failed(collectGroupLayoutPlan(group, layoutPlan)))
    return failAndReturn("group-to-tile-region layout planning failed");
  if (!layoutPlan.succeeded)
    return failAndReturn(layoutPlan.failureReason);

  llvm::SmallVector<mlir::Value, 4> tileRegionInputs;
  for (auto [original, converted] :
       llvm::zip(group.getInputs(), convertedInputs)) {
    if (isElidableConstantBoundary(original))
      continue;
    mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
        original, converted, /*readOnly=*/true, rewriter);
    if (mlir::failed(boundary))
      return mlir::failure();
    tileRegionInputs.push_back(*boundary);
  }
  for (auto [original, converted] : llvm::zip(group.getOuts(), convertedOuts)) {
    mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
        original, converted, /*readOnly=*/false, rewriter);
    if (mlir::failed(boundary))
      return mlir::failure();
    tileRegionInputs.push_back(*boundary);
  }

  llvm::SmallVector<mlir::Type, 2> tileRegionResultTypes;
  for (mlir::Type resultType : group.getResultTypes()) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
      return failAndReturn("group result is not a ranked tensor");
    tileRegionResultTypes.push_back(makeDDRMemRefType(tensorType));
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto tileRegion = rewriter.create<TileRegionOp>(
      group.getLoc(), tileRegionResultTypes, tileRegionInputs);
  mlir::Block *tileBlock = new mlir::Block();
  tileRegion.getBody().push_back(tileBlock);
  for (mlir::Value input : tileRegion.getInputs())
    tileBlock->addArgument(input.getType(), input.getLoc());

  rewriter.setInsertionPointToStart(tileBlock);
  if (mlir::failed(initializeBoundary(group, tileRegion, rewriter)))
    return mlir::failure();

  for (OpLayoutPlan &opPlan : layoutPlan.ops) {
    if (mlir::failed(convertOp(opPlan, rewriter)))
      return mlir::failure();
  }

  if (mlir::failed(finishRegion(group, tileRegion, rewriter)))
    return mlir::failure();

  return tileRegion;
}

mlir::LogicalResult TileRegionBodyEmitter::fail(llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return mlir::failure();
}

mlir::FailureOr<TileRegionOp>
TileRegionBodyEmitter::failAndReturn(llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return mlir::failure();
}

mlir::FailureOr<mlir::Value>
TileRegionBodyEmitter::failValue(llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return mlir::failure();
}

mlir::FailureOr<ElementwiseExprValue>
TileRegionBodyEmitter::failElementwiseExprValue(llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return mlir::failure();
}

mlir::FailureOr<int64_t>
TileRegionBodyEmitter::failI64(llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return mlir::failure();
}

mlir::FailureOr<SelectedCollectiveRankGroup>
TileRegionBodyEmitter::failSelectedCollectiveRankGroup(llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return mlir::failure();
}

mlir::FailureOr<unsigned>
TileRegionBodyEmitter::failUnsigned(llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return mlir::failure();
}

mlir::FailureOr<mlir::Type>
TileRegionBodyEmitter::failType(llvm::StringRef reason) {
  setFailureReason(failureReason, reason);
  return mlir::failure();
}

mlir::MemRefType TileRegionBodyEmitter::makeWaferMemRefType(
    mlir::RankedTensorType tensorType, MemorySpace space, MemLayout layout) {
  auto *context = tensorType.getContext();
  return mlir::MemRefType::get(tensorType.getShape(),
                               tensorType.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               MemoryAttr::get(context, space, layout));
}

mlir::MemRefType
TileRegionBodyEmitter::makeSPMMemRefType(mlir::RankedTensorType tensorType,
                                         MemLayout layout) {
  return makeWaferMemRefType(tensorType, MemorySpace::SPM, layout);
}

mlir::MemRefType
TileRegionBodyEmitter::makeDDRMemRefType(mlir::RankedTensorType tensorType) {
  return makeWaferMemRefType(tensorType, MemorySpace::DDR, MemLayout::Tensor);
}

bool TileRegionBodyEmitter::isScalarType(mlir::Type type) const {
  return mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(type);
}

StateSnapshot TileRegionBodyEmitter::snapshotState() const {
  return {buffers,
          scalarValues,
          scalarAttrs,
          tensorAttrs,
          externalBuffers,
          writableExternalBuffers,
          externalOutputIndices,
          directYieldBuffers,
          fillInitScalars,
          fillInitAttrs};
}

void TileRegionBodyEmitter::restoreState(const StateSnapshot &snapshot) {
  buffers = snapshot.buffers;
  scalarValues = snapshot.scalarValues;
  scalarAttrs = snapshot.scalarAttrs;
  tensorAttrs = snapshot.tensorAttrs;
  externalBuffers = snapshot.externalBuffers;
  writableExternalBuffers = snapshot.writableExternalBuffers;
  externalOutputIndices = snapshot.externalOutputIndices;
  directYieldBuffers = snapshot.directYieldBuffers;
  fillInitScalars = snapshot.fillInitScalars;
  fillInitAttrs = snapshot.fillInitAttrs;
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::materializeDdrBoundary(
    mlir::Value original, mlir::Value converted, bool readOnly,
    mlir::ConversionPatternRewriter &rewriter) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
  if (!tensorType) {
    if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
            original.getType()))
      return converted;
    return failValue("group boundary is not a ranked tensor or scalar");
  }
  if (!readOnly && original.getDefiningOp<mlir::tensor::EmptyOp>()) {
    auto alloc = rewriter.create<mlir::memref::AllocOp>(
        original.getLoc(), makeDDRMemRefType(tensorType));
    return alloc.getResult();
  }
  auto toMemref = rewriter.create<mlir::bufferization::ToMemrefOp>(
      original.getLoc(), makeDDRMemRefType(tensorType), converted, readOnly);
  return toMemref.getMemref();
}

MemLayout TileRegionBodyEmitter::alignedLayoutForTensor(
    mlir::RankedTensorType tensorType) const {
  return tensorType.getRank() > 2 ? MemLayout::NCx : MemLayout::Cx;
}

void TileRegionBodyEmitter::record(mlir::Value original, MemLayout layout,
                                   mlir::Value buffer) {
  BufferVersions &versions = buffers[original];
  switch (layout) {
  case MemLayout::Tensor:
    versions.tensor = buffer;
    break;
  case MemLayout::NTensor:
    versions.nTensor = buffer;
    break;
  case MemLayout::Cx:
    versions.cx = buffer;
    break;
  case MemLayout::NCx:
    versions.nCx = buffer;
    break;
  }
}

mlir::Value TileRegionBodyEmitter::lookup(mlir::Value original,
                                          MemLayout layout) const {
  auto it = buffers.find(original);
  if (it == buffers.end())
    return {};
  const BufferVersions &versions = it->second;
  switch (layout) {
  case MemLayout::Tensor:
    return versions.tensor;
  case MemLayout::NTensor:
    return versions.nTensor;
  case MemLayout::Cx:
    return versions.cx;
  case MemLayout::NCx:
    return versions.nCx;
  }
  llvm_unreachable("unknown memory layout");
}

mlir::Value TileRegionBodyEmitter::lookupAny(mlir::Value original,
                                             MemLayout &layout) const {
  auto it = buffers.find(original);
  if (it == buffers.end())
    return {};
  const BufferVersions &versions = it->second;
  if (versions.tensor) {
    layout = MemLayout::Tensor;
    return versions.tensor;
  }
  if (versions.cx) {
    layout = MemLayout::Cx;
    return versions.cx;
  }
  if (versions.nTensor) {
    layout = MemLayout::NTensor;
    return versions.nTensor;
  }
  if (versions.nCx) {
    layout = MemLayout::NCx;
    return versions.nCx;
  }
  return {};
}

mlir::TypedAttr
TileRegionBodyEmitter::getScalarSplatAttr(mlir::RankedTensorType tensorType,
                                          mlir::Attribute attr) const {
  if (auto typed = mlir::dyn_cast<mlir::TypedAttr>(attr)) {
    if (typed.getType() == tensorType.getElementType())
      return typed;
  }

  auto elements = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!elements || !elements.isSplat() ||
      elements.getElementType() != tensorType.getElementType())
    return {};

  mlir::Type elementType = tensorType.getElementType();
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
    return mlir::FloatAttr::get(floatType,
                                elements.getSplatValue<mlir::APFloat>());
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType))
    return mlir::IntegerAttr::get(intType,
                                  elements.getSplatValue<mlir::APInt>());
  return {};
}

bool TileRegionBodyEmitter::isElidableConstantBoundary(
    mlir::Value value) const {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  if (isScalarType(value.getType()))
    return true;
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  return tensorType &&
         static_cast<bool>(getScalarSplatAttr(tensorType, constant.getValue()));
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::materializeTensorConstant(
    mlir::Value original, mlir::Attribute attr, MemLayout targetLayout,
    mlir::OpBuilder &builder) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
  if (!tensorType)
    return failValue("constant tensor materialization requires ranked tensor");

  mlir::TypedAttr scalarAttr = getScalarSplatAttr(tensorType, attr);
  if (!scalarAttr)
    return failValue("constant tensor materialization requires splat attr");

  auto scalar =
      builder.create<mlir::arith::ConstantOp>(original.getLoc(), scalarAttr);
  auto tensorBuffer = builder.create<mlir::memref::AllocOp>(
      original.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor));
  builder.create<ComputeFillOp>(original.getLoc(), tensorBuffer.getResult(),
                                scalar.getResult());
  record(original, MemLayout::Tensor, tensorBuffer.getResult());
  if (targetLayout == MemLayout::Tensor)
    return tensorBuffer.getResult();

  auto materialized = builder.create<LayoutMaterializeOp>(
      original.getLoc(), makeSPMMemRefType(tensorType, targetLayout),
      tensorBuffer.getResult());
  record(original, targetLayout, materialized.getResult());
  return materialized.getResult();
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::getOrMaterialize(
    mlir::Value original, MemLayout targetLayout, mlir::OpBuilder &builder) {
  if (mlir::Value existing = lookup(original, targetLayout))
    return existing;

  MemLayout sourceLayout = MemLayout::Tensor;
  mlir::Value source = lookupAny(original, sourceLayout);
  if (!source) {
    if (auto attrIt = tensorAttrs.find(original); attrIt != tensorAttrs.end()) {
      mlir::FailureOr<mlir::Value> constant = materializeTensorConstant(
          original, attrIt->second, targetLayout, builder);
      if (mlir::succeeded(constant))
        return *constant;
    }

    auto externalIt = externalBuffers.find(original);
    if (externalIt == externalBuffers.end())
      return failValue("missing buffer for value");

    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType)
      return failValue("cannot materialize non-ranked-tensor value");

    auto load = builder.create<StorageLoadOp>(
        original.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor),
        externalIt->second);
    record(original, MemLayout::Tensor, load.getResult());
    source = load.getResult();
    sourceLayout = MemLayout::Tensor;
  }
  if (sourceLayout == targetLayout)
    return source;

  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
  if (!tensorType)
    return failValue("cannot materialize non-ranked-tensor value");

  mlir::Type resultType = makeSPMMemRefType(tensorType, targetLayout);
  auto materialize = builder.create<LayoutMaterializeOp>(original.getLoc(),
                                                         resultType, source);
  record(original, targetLayout, materialize.getResult());
  return materialize.getResult();
}

mlir::LogicalResult TileRegionBodyEmitter::initializeBoundary(
    GroupOp group, TileRegionOp tileRegion, mlir::OpBuilder &builder) {
  mlir::Block &groupBlock = group.getBody().front();
  mlir::Block &tileBlock = tileRegion.getBody().front();
  if (groupBlock.getNumArguments() !=
      group.getInputs().size() + group.getOuts().size())
    return fail("group boundary argument count mismatch");

  unsigned tileArgIndex = 0;
  unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
  for (mlir::BlockArgument groupArg : groupBlock.getArguments()) {
    unsigned argIndex = groupArg.getArgNumber();
    if (argIndex < inputCount &&
        isElidableConstantBoundary(group.getInputs()[argIndex])) {
      auto constant =
          group.getInputs()[argIndex].getDefiningOp<mlir::arith::ConstantOp>();
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(groupArg.getType());
      if (tensorType) {
        tensorAttrs[groupArg] = constant.getValue();
      } else {
        mlir::Operation *cloned = builder.clone(*constant.getOperation());
        scalarValues[groupArg] = cloned->getResult(0);
        scalarAttrs[groupArg] = constant.getValue();
      }
      continue;
    }

    if (tileArgIndex >= tileBlock.getNumArguments())
      return fail("tile-region boundary argument count mismatch");
    mlir::BlockArgument tileArg = tileBlock.getArgument(tileArgIndex++);
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(groupArg.getType());
    if (!tensorType) {
      if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
              groupArg.getType())) {
        scalarValues[groupArg] = tileArg;
        continue;
      }
      return fail("group boundary is not a ranked tensor or scalar");
    }
    externalBuffers[groupArg] = tileArg;
    if (argIndex >= inputCount) {
      writableExternalBuffers.insert(groupArg);
      externalOutputIndices[groupArg] = argIndex - inputCount;
    }
  }
  if (tileArgIndex != tileBlock.getNumArguments())
    return fail("tile-region boundary argument count mismatch");
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertOp(const OpLayoutPlan &opPlan,
                                                     mlir::OpBuilder &builder) {
  if (opPlan.kind == OpTilingDemandKind::Failure)
    return fail(opPlan.failureReason);

  mlir::Operation *op = opPlan.op;
  if (opPlan.kind == OpTilingDemandKind::Support)
    return convertSupportOp(op, builder);
  if (opPlan.kind == OpTilingDemandKind::LinalgExtCollective)
    return convertLinalgExtCollective(op, opPlan.collectiveInfo, builder);

  if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(op))
    return convertFill(fill, builder);
  if (mlir::isa<mlir::linalg::MatmulOp>(op))
    return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder);
  if (mlir::isa<mlir::linalg::BatchMatmulOp>(op))
    return convertBatchMatmul(mlir::cast<mlir::linalg::LinalgOp>(op), builder);
  if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op))
    return convertGeneric(generic, builder);

  return fail("unsupported linalg op " + op->getName().getStringRef().str());
}

mlir::LogicalResult
TileRegionBodyEmitter::finishRegion(GroupOp group, TileRegionOp tileRegion,
                                    mlir::OpBuilder &builder) {
  auto yield =
      mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
  if (!yield)
    return fail("group terminator is not wafer.group.yield");

  unsigned inputCount = static_cast<unsigned>(group.getInputs().size());
  llvm::SmallVector<mlir::Value, 2> yieldedValues;
  for (auto [index, value] : llvm::enumerate(yield.getValues())) {
    mlir::BlockArgument groupOutput =
        group.getBody().front().getArgument(inputCount + index);
    auto outputIt = externalBuffers.find(groupOutput);
    if (outputIt == externalBuffers.end())
      return fail("group result has no output boundary");
    mlir::Value output = outputIt->second;

    if (auto directIt = directYieldBuffers.find(value);
        directIt != directYieldBuffers.end()) {
      if (directIt->second != output)
        return fail("direct boundary storeback target mismatch");
      yieldedValues.push_back(output);
      continue;
    }

    mlir::FailureOr<mlir::Value> tensorBuffer =
        getOrMaterialize(value, MemLayout::Tensor, builder);
    if (mlir::failed(tensorBuffer))
      return mlir::failure();

    builder.create<StorageStoreOp>(value.getLoc(), *tensorBuffer, output);
    yieldedValues.push_back(output);
  }

  builder.create<TileYieldOp>(group.getLoc(), yieldedValues);
  return mlir::success();
}

} // namespace wafer::group_to_tile_region
