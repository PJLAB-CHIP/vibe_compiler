//===- BodyEmitter.cpp - Tensor program body lowering orchestration --===//

#include "Internal.h"
#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

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
  // This helper only classifies an exact combiner. It does not prove that a
  // candidate may regroup that combiner: reduction-split materialization has
  // a separate numeric-legality gate over the current structured IR.
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

TileRegionBodyEmitter::TileRegionBodyEmitter(
    std::string *failureReason, int64_t currentLogicalRank,
    std::optional<TargetImplementationKind> selectedAlternative,
    bool useDirectMappedBoundaryTransfer)
    : failureReason(failureReason), currentLogicalRank(currentLogicalRank),
      selectedAlternative(selectedAlternative),
      useDirectMappedBoundaryTransfer(useDirectMappedBoundaryTransfer) {}

mlir::FailureOr<TileRegionOp>
TileRegionBodyEmitter::emit(TensorProgramScope scope,
                            mlir::RewriterBase &rewriter) {
  if (currentLogicalRank < 0)
    return failAndReturn("logical-rank must be non-negative");

  llvm::DenseSet<mlir::Value> boundaryValues;
  for (mlir::Value input : scope.getInputs()) {
    if (!boundaryValues.insert(input).second)
      return failAndReturn("tensor program boundary SSA values must be unique");
  }
  for (mlir::Value out : scope.getOutputs()) {
    if (!boundaryValues.insert(out).second)
      return failAndReturn("tensor program boundary SSA values must be unique");
  }
  if (mlir::failed(verifyNamedLinalgPayloads(scope)))
    return mlir::failure();

  // Snapshot only the current source operations before creating the target
  // region in the same function. This is an iteration worklist, not a second
  // semantic plan; every conversion decision is made from the live operation.
  llvm::SmallVector<mlir::Operation *, 16> sourceOps;
  for (mlir::Operation &op : scope.getBody().without_terminator())
    sourceOps.push_back(&op);

  llvm::SmallVector<mlir::Value, 4> tileRegionInputs;
  for (mlir::Value original : scope.getInputs()) {
    if (isElidableConstantBoundary(original))
      continue;
    mlir::FailureOr<mlir::Value> boundary =
        materializeDdrBoundary(original, original, /*readOnly=*/true, rewriter);
    if (mlir::failed(boundary))
      return mlir::failure();
    tileRegionInputs.push_back(*boundary);
  }
  for (mlir::Value original : scope.getOutputs()) {
    mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
        original, original, /*readOnly=*/false, rewriter);
    if (mlir::failed(boundary))
      return mlir::failure();
    tileRegionInputs.push_back(*boundary);
  }

  llvm::SmallVector<mlir::Type, 2> tileRegionResultTypes;
  for (mlir::Type resultType : scope.getResultTypes()) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
      return failAndReturn("tensor program result is not a ranked tensor");
    tileRegionResultTypes.push_back(makeDDRMemRefType(tensorType));
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto tileRegion = rewriter.create<TileRegionOp>(
      scope.getLoc(), tileRegionResultTypes, tileRegionInputs);
  mlir::Block *tileBlock = new mlir::Block();
  tileRegion.getBody().push_back(tileBlock);
  for (mlir::Value input : tileRegion.getInputs())
    tileBlock->addArgument(input.getType(), input.getLoc());

  rewriter.setInsertionPointToStart(tileBlock);
  if (mlir::failed(initializeBoundary(scope, tileRegion, rewriter)))
    return mlir::failure();

  for (mlir::Operation *op : sourceOps) {
    if (mlir::failed(convertOp(op, rewriter)))
      return mlir::failure();
  }

  if (selectedAlternative && !selectedAlternativeMaterialized)
    return failAndReturn(
        "selected target implementation alternative did not match any "
        "source operation in the materialized clone");

  if (mlir::failed(finishRegion(scope, tileRegion, rewriter)))
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
          compilerOwnedDDRBuffers,
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
  compilerOwnedDDRBuffers = snapshot.compilerOwnedDDRBuffers;
  externalBuffers = snapshot.externalBuffers;
  writableExternalBuffers = snapshot.writableExternalBuffers;
  externalOutputIndices = snapshot.externalOutputIndices;
  directYieldBuffers = snapshot.directYieldBuffers;
  fillInitScalars = snapshot.fillInitScalars;
  fillInitAttrs = snapshot.fillInitAttrs;
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::materializeDdrBoundary(
    mlir::Value original, mlir::Value converted, bool readOnly,
    mlir::RewriterBase &rewriter) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
  if (!tensorType) {
    if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
            original.getType()))
      return converted;
    return failValue(
        "tensor program boundary is not a ranked tensor or scalar");
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
                                scalar.getResult(),
                                /*fill_domain=*/FillDomainAttr{});
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
  mlir::MemRefType stagedBoundarySourceType;
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

    auto externalType =
        mlir::dyn_cast<mlir::MemRefType>(externalIt->second.getType());
    mlir::MemRefType directDestinationType =
        makeSPMMemRefType(tensorType, targetLayout);
    analysis::IndexRelationResult identity =
        analysis::IndexRelation::identity(tensorType.getShape());
    if (useDirectMappedBoundaryTransfer && targetLayout != MemLayout::Tensor &&
        externalType && identity.isExact() &&
        mlir::succeeded(analysis::TransferRealizability::proveMappedDma(
            externalType, directDestinationType, *identity.get()))) {
      auto destination = builder.create<mlir::memref::AllocOp>(
          original.getLoc(), directDestinationType);
      builder.create<StorageLoadOp>(original.getLoc(), externalIt->second,
                                    destination.getResult());
      record(original, targetLayout, destination.getResult());
      return destination.getResult();
    }

    auto destination = builder.create<mlir::memref::AllocOp>(
        original.getLoc(), makeSPMMemRefType(tensorType, MemLayout::Tensor));
    builder.create<StorageLoadOp>(original.getLoc(), externalIt->second,
                                  destination.getResult());
    stagedBoundarySourceType = externalType;
    record(original, MemLayout::Tensor, destination.getResult());
    source = destination.getResult();
    sourceLayout = MemLayout::Tensor;
  }
  if (sourceLayout == targetLayout)
    return source;

  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
  if (!tensorType)
    return failValue("cannot materialize non-ranked-tensor value");

  mlir::Type resultType = makeSPMMemRefType(tensorType, targetLayout);
  if (stagedBoundarySourceType) {
    auto temporaryType = mlir::cast<mlir::MemRefType>(source.getType());
    auto stagedDestType = mlir::cast<mlir::MemRefType>(resultType);
    analysis::IndexRelationResult identity =
        analysis::IndexRelation::identity(tensorType.getShape());
    if (!identity.isExact() ||
        mlir::failed(analysis::TransferRealizability::proveStagedMovement(
            stagedBoundarySourceType, temporaryType, stagedDestType,
            *identity.get(), *identity.get())))
      return failValue("boundary staged movement is not exactly realizable");
  }
  auto materialize = builder.create<LayoutMaterializeOp>(original.getLoc(),
                                                         resultType, source);
  record(original, targetLayout, materialize.getResult());
  return materialize.getResult();
}

mlir::LogicalResult
TileRegionBodyEmitter::initializeBoundary(TensorProgramScope scope,
                                          TileRegionOp tileRegion,
                                          mlir::OpBuilder &builder) {
  mlir::Block &sourceBlock = scope.getBody();
  mlir::Block &tileBlock = tileRegion.getBody().front();
  if (sourceBlock.getNumArguments() !=
      scope.getInputCount() + scope.getOutputCount())
    return fail("tensor program boundary argument count mismatch");

  unsigned tileArgIndex = 0;
  unsigned inputCount = scope.getInputCount();
  for (mlir::BlockArgument sourceArg : sourceBlock.getArguments()) {
    unsigned argIndex = sourceArg.getArgNumber();
    if (argIndex < inputCount &&
        isElidableConstantBoundary(scope.getInputs()[argIndex])) {
      auto constant =
          scope.getInputs()[argIndex].getDefiningOp<mlir::arith::ConstantOp>();
      auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(sourceArg.getType());
      if (tensorType) {
        tensorAttrs[sourceArg] = constant.getValue();
      } else {
        mlir::Operation *cloned = builder.clone(*constant.getOperation());
        scalarValues[sourceArg] = cloned->getResult(0);
        scalarAttrs[sourceArg] = constant.getValue();
      }
      continue;
    }

    if (tileArgIndex >= tileBlock.getNumArguments())
      return fail("tile-region boundary argument count mismatch");
    mlir::BlockArgument tileArg = tileBlock.getArgument(tileArgIndex++);
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(sourceArg.getType());
    if (!tensorType) {
      if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(
              sourceArg.getType())) {
        scalarValues[sourceArg] = tileArg;
        continue;
      }
      return fail("tensor program boundary is not a ranked tensor or scalar");
    }
    externalBuffers[sourceArg] = tileArg;
    if (argIndex >= inputCount) {
      writableExternalBuffers.insert(sourceArg);
      externalOutputIndices[sourceArg] = argIndex - inputCount;
    }
  }
  if (tileArgIndex != tileBlock.getNumArguments())
    return fail("tile-region boundary argument count mismatch");
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::convertOp(mlir::Operation *op,
                                                     mlir::OpBuilder &builder) {
  if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(op))
    return convertLinalgExtCollective(op, builder);

  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(op)) {
    if (!linalg.hasOnlyProjectedPermutations())
      return fail("unsupported linalg indexing maps");
    return materializeSourceImplementation(op, builder);
  }

  if (mlir::isa<mlir::arith::ConstantOp, mlir::bufferization::ToMemrefOp,
                mlir::bufferization::ToTensorOp, mlir::tensor::EmptyOp,
                mlir::memref::AllocOp, mlir::tensor::ExtractOp,
                mlir::tensor::ExtractSliceOp, mlir::tensor::InsertSliceOp,
                mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp,
                mlir::scf::IfOp, mlir::scf::ForOp>(op))
    return convertSupportOp(op, builder);

  return fail("unsupported tensor-program op " +
              op->getName().getStringRef().str());
}

mlir::LogicalResult TileRegionBodyEmitter::materializeSourceImplementation(
    mlir::Operation *operation, mlir::OpBuilder &builder) {
  auto interface =
      mlir::dyn_cast<WaferTargetImplementationOpInterface>(operation);
  if (!interface)
    return fail("source operation has no target implementation interface: " +
                operation->getName().getStringRef().str());

  llvm::SmallVector<TargetImplementationCandidate, 2> candidates;
  interface.collectTargetImplementationCandidates(WaferTargetCapabilities{},
                                                  candidates);
  if (candidates.empty())
    return fail("source target implementation is explicitly unsupported: " +
                operation->getName().getStringRef().str());

  const TargetImplementationCandidate *selected = &candidates.front();
  if (selectedAlternative) {
    auto alternative = llvm::find_if(candidates, [&](const auto &candidate) {
      return candidate.kind == *selectedAlternative;
    });
    if (alternative != candidates.end()) {
      selected = &*alternative;
      selectedAlternativeMaterialized = true;
    }
  }
  if (mlir::failed(interface.materializeSelectedTargetImplementation(
          *selected, *this, builder))) {
    if (failureReason && !failureReason->empty())
      return mlir::failure();
    return fail("selected target implementation failed to materialize");
  }
  return mlir::success();
}

mlir::LogicalResult TileRegionBodyEmitter::materializeTargetImplementation(
    mlir::Operation *source, const TargetImplementationCandidate &candidate,
    mlir::OpBuilder &builder) {
  switch (candidate.kind) {
  case TargetImplementationKind::Fill:
    if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(source))
      return convertFill(fill, builder);
    break;
  case TargetImplementationKind::Gemm:
    if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
                  mlir::linalg::MatmulTransposeBOp, mlir::linalg::GenericOp>(
            source))
      return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(source), builder);
    break;
  case TargetImplementationKind::BatchGemm:
    if (mlir::isa<mlir::linalg::BatchMatmulOp,
                  mlir::linalg::BatchMatmulTransposeAOp,
                  mlir::linalg::BatchMatmulTransposeBOp>(source))
      return convertBatchMatmul(mlir::cast<mlir::linalg::LinalgOp>(source),
                                builder);
    break;
  case TargetImplementationKind::Generic:
  case TargetImplementationKind::GenericReciprocal:
    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(source))
      return convertGeneric(generic,
                            candidate.kind ==
                                TargetImplementationKind::GenericReciprocal,
                            builder);
    break;
  }
  return fail("selected target implementation does not match source op");
}

mlir::LogicalResult
TileRegionBodyEmitter::finishRegion(TensorProgramScope scope,
                                    TileRegionOp tileRegion,
                                    mlir::OpBuilder &builder) {
  auto returnOp =
      mlir::dyn_cast<mlir::func::ReturnOp>(scope.getBody().getTerminator());
  if (!returnOp)
    return fail("tensor program terminator is not func.return");

  unsigned inputCount = scope.getInputCount();
  llvm::SmallVector<mlir::Value, 2> yieldedValues;
  for (auto [index, value] : llvm::enumerate(returnOp.getOperands())) {
    mlir::BlockArgument outputArgument =
        scope.getBody().getArgument(inputCount + index);
    auto outputIt = externalBuffers.find(outputArgument);
    if (outputIt == externalBuffers.end())
      return fail("tensor program result has no output boundary");
    mlir::Value output = outputIt->second;

    if (auto directIt = directYieldBuffers.find(value);
        directIt != directYieldBuffers.end()) {
      if (directIt->second != output)
        return fail("direct boundary storeback target mismatch");
      yieldedValues.push_back(output);
      continue;
    }

    MemLayout currentLayout = MemLayout::Tensor;
    mlir::Value current = lookupAny(value, currentLayout);
    auto outputType = mlir::dyn_cast<mlir::MemRefType>(output.getType());
    auto currentType = current
                           ? mlir::dyn_cast<mlir::MemRefType>(current.getType())
                           : mlir::MemRefType{};
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    analysis::IndexRelationResult identity =
        tensorType ? analysis::IndexRelation::identity(tensorType.getShape())
                   : analysis::IndexRelationResult{};
    if (useDirectMappedBoundaryTransfer && current &&
        currentLayout != MemLayout::Tensor && currentType && outputType &&
        identity.isExact() &&
        mlir::succeeded(analysis::TransferRealizability::proveMappedDma(
            currentType, outputType, *identity.get()))) {
      builder.create<StorageStoreOp>(value.getLoc(), current, output);
    } else {
      mlir::FailureOr<mlir::Value> tensorBuffer =
          getOrMaterialize(value, MemLayout::Tensor, builder);
      if (mlir::failed(tensorBuffer))
        return mlir::failure();
      builder.create<StorageStoreOp>(value.getLoc(), *tensorBuffer, output);
    }
    yieldedValues.push_back(output);
  }

  builder.create<TileYieldOp>(scope.getLoc(), yieldedValues);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
