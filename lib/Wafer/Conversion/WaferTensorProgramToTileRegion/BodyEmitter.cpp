//===- BodyEmitter.cpp - Tensor program body lowering orchestration --===//

#include "Internal.h"
#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"

#include <limits>

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
    std::string *failureReason, int64_t currentLogicalPartition,
    llvm::ArrayRef<CandidatePeerEndpoint> peerEndpoints,
    llvm::ArrayRef<CandidateSelectedDDRStage> selectedDDRStages,
    TileRegionEmissionRelations *emissionRelations,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes)
    : failureReason(failureReason),
      currentLogicalPartition(currentLogicalPartition),
      peerEndpoints(peerEndpoints.begin(), peerEndpoints.end()),
      selectedDDRStages(selectedDDRStages),
      emissionRelations(emissionRelations) {
  for (const StructuredOperationNodeMapping &mapping : operationNodes) {
    if (!mapping.operation)
      continue;
    auto &nodes = structuredNodeIds[mapping.operation];
    if (!llvm::is_contained(nodes, mapping.structuredNodeId))
      nodes.push_back(mapping.structuredNodeId);
  }
}

mlir::FailureOr<TileRegionOp>
TileRegionBodyEmitter::emit(TensorProgramScope scope,
                            mlir::RewriterBase &rewriter) {
  if (currentLogicalPartition < 0)
    return failAndReturn("logical partition must be non-negative");

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
  // Narrow per-root construction: consumer-side boundary supplies enter the
  // region as read-only DDR boundaries exactly like source inputs; the
  // sibling structured producer's compute is never pulled into this scope.
  for (mlir::Value original : scope.getBoundaryArguments()) {
    mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
        original, original, /*readOnly=*/true, rewriter);
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

  // Every Tile observes the same consumer-first message order after
  // omitting messages on which it is not an endpoint.  Message identity is
  // intentionally not an execution order: its stable numbering is independent
  // of operation ordinals and may place a later consumer first.  Emitting one
  // exact endpoint and its completion at a time gives the conservative
  // baseline one live receiver at most, while this query-local common total
  // order prevents mutual-send cycles across Tiles.  Only the resulting peer
  // op/wait order survives in Tile IR.
  llvm::SmallVector<const CandidatePeerEndpoint *, 8> endpointSchedule;
  endpointSchedule.reserve(peerEndpoints.size());
  for (const CandidatePeerEndpoint &endpoint : peerEndpoints)
    endpointSchedule.push_back(&endpoint);
  llvm::sort(endpointSchedule, [](const CandidatePeerEndpoint *lhs,
                                  const CandidatePeerEndpoint *rhs) {
    return std::tuple(lhs->consumerScheduleOrdinal, lhs->consumerOperand,
                      lhs->communicationId, lhs->payloadSlice,
                      static_cast<uint8_t>(lhs->kind), lhs->peer.getValue()) <
           std::tuple(rhs->consumerScheduleOrdinal, rhs->consumerOperand,
                      rhs->communicationId, rhs->payloadSlice,
                      static_cast<uint8_t>(rhs->kind), rhs->peer.getValue());
  });
  llvm::DenseSet<std::pair<int64_t, int64_t>> localMessages;
  for (const CandidatePeerEndpoint *endpoint : endpointSchedule)
    if (!localMessages
             .insert({endpoint->communicationId, endpoint->payloadSlice})
             .second)
      return failAndReturn(
          "selected peer schedule contains a duplicate local message");
  size_t nextEndpoint = 0;
  llvm::DenseSet<const CandidatePeerEndpoint *> emittedEndpoints;
  auto emitReadyEndpoints = [&]() -> mlir::LogicalResult {
    while (nextEndpoint < endpointSchedule.size()) {
      const CandidatePeerEndpoint *endpoint = endpointSchedule[nextEndpoint];
      MemLayout availableLayout = MemLayout::Tensor;
      if (!lookupAny(endpoint->value, availableLayout) &&
          !externalBuffers.contains(endpoint->value) &&
          !tensorAttrs.contains(endpoint->value))
        break;
      if (mlir::failed(emitPeerEndpoint(*endpoint, rewriter)))
        return mlir::failure();
      emittedEndpoints.insert(endpoint);
      ++nextEndpoint;
    }
    return mlir::success();
  };

  for (mlir::Operation *op : sourceOps) {
    for (mlir::Value operand : op->getOperands()) {
      auto pendingReceive = llvm::find_if(
          endpointSchedule, [&](const CandidatePeerEndpoint *endpoint) {
            return endpoint->kind == CandidatePeerEndpointKind::Receive &&
                   endpoint->value == operand &&
                   !emittedEndpoints.contains(endpoint);
          });
      if (pendingReceive != endpointSchedule.end()) {
        const CandidatePeerEndpoint *blocked = endpointSchedule[nextEndpoint];
        std::string detail;
        llvm::raw_string_ostream diagnostic(detail);
        diagnostic << "globally ordered peer receive (communication="
                   << (*pendingReceive)->communicationId
                   << ", slice=" << (*pendingReceive)->payloadSlice
                   << ") was not ready before its first SSA use by "
                   << op->getName() << "; preceding endpoint (communication="
                   << blocked->communicationId
                   << ", slice=" << blocked->payloadSlice << ", kind="
                   << (blocked->kind == CandidatePeerEndpointKind::Send
                           ? "send"
                           : "receive")
                   << ") has no materialized buffer";
        if (mlir::Operation *definition = blocked->value.getDefiningOp()) {
          diagnostic << "; definition=" << definition->getName();
          if (definition->getBlock() == op->getBlock())
            diagnostic << ", definition_before_use="
                       << definition->isBeforeInBlock(op);
          if (auto slice =
                  mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(definition))
            if (mlir::Operation *source = slice.getSource().getDefiningOp())
              diagnostic << ", source_definition=" << source->getName();
        }
        return failAndReturn(detail);
      }
    }
    if (mlir::failed(convertOp(op, rewriter)))
      return mlir::failure();
    if (mlir::failed(emitReadyEndpoints()))
      return mlir::failure();
  }

  if (nextEndpoint != endpointSchedule.size())
    return failAndReturn(
        "selected peer endpoint was not materialized from current SSA");

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

mlir::Location
TileRegionBodyEmitter::getMaterializationLocation(mlir::Value original) const {
  return original.getLoc();
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

mlir::FailureOr<SelectedCollectivePartitionGroup>
TileRegionBodyEmitter::failSelectedCollectivePartitionGroup(
    llvm::StringRef reason) {
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
          compilerOwnedBuffers,
          externalBuffers,
          writableExternalBuffers,
          selectedDDRStageExternalBuffers,
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
  compilerOwnedBuffers = snapshot.compilerOwnedBuffers;
  externalBuffers = snapshot.externalBuffers;
  writableExternalBuffers = snapshot.writableExternalBuffers;
  selectedDDRStageExternalBuffers = snapshot.selectedDDRStageExternalBuffers;
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

  mlir::Location loc = getMaterializationLocation(original);
  auto scalar = builder.create<mlir::arith::ConstantOp>(loc, scalarAttr);
  auto tensorBuffer = builder.create<mlir::memref::AllocOp>(
      loc, makeSPMMemRefType(tensorType, MemLayout::Tensor));
  builder.create<ComputeFillOp>(loc, tensorBuffer.getResult(),
                                scalar.getResult(),
                                /*fill_domain=*/FillDomainAttr{});
  record(original, MemLayout::Tensor, tensorBuffer.getResult());
  if (targetLayout == MemLayout::Tensor)
    return tensorBuffer.getResult();

  auto materialized = builder.create<LayoutMaterializeOp>(
      loc, makeSPMMemRefType(tensorType, targetLayout),
      tensorBuffer.getResult());
  record(original, targetLayout, materialized.getResult());
  return materialized.getResult();
}

mlir::FailureOr<mlir::Value> TileRegionBodyEmitter::getOrMaterialize(
    mlir::Value original, MemLayout targetLayout, mlir::OpBuilder &builder) {
  if (mlir::Value existing = lookup(original, targetLayout))
    return existing;

  mlir::Location materializationLoc = getMaterializationLocation(original);

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
    if (externalIt == externalBuffers.end()) {
      // Query-local materialization can introduce a read-only DDR reopen while
      // converting an enclosing structured traversal. Recover exactly the
      // same typed buffer relation as convertSupportOp when its allocation has
      // already been cloned, instead of relying on a stale operation snapshot
      // order.
      if (auto toTensor =
              original.getDefiningOp<mlir::bufferization::ToTensorOp>()) {
        auto converted = compilerOwnedBuffers.find(toTensor.getMemref());
        if (converted == compilerOwnedBuffers.end()) {
          auto allocation =
              toTensor.getMemref().getDefiningOp<mlir::memref::AllocOp>();
          if (allocation && isWaferDDRMemRefType(allocation.getType()) &&
              allocation.getDynamicSizes().empty() &&
              allocation.getSymbolOperands().empty()) {
            mlir::Operation *cloned = builder.clone(*allocation.getOperation());
            compilerOwnedBuffers[allocation.getResult()] = cloned->getResult(0);
            converted = compilerOwnedBuffers.find(toTensor.getMemref());
          }
        }
        if (converted != compilerOwnedBuffers.end() &&
            isWaferDDRMemRefType(converted->second.getType())) {
          externalBuffers[original] = converted->second;
          if (toTensor.getWritable())
            writableExternalBuffers.insert(original);
          externalIt = externalBuffers.find(original);
        }
      }
    }
    if (externalIt == externalBuffers.end()) {
      std::string detail;
      llvm::raw_string_ostream diagnostic(detail);
      diagnostic << "missing buffer for value; type=" << original.getType();
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(original)) {
        diagnostic << "; definition=block-argument#" << argument.getArgNumber();
      } else if (mlir::Operation *definition = original.getDefiningOp()) {
        diagnostic << "; definition=" << definition->getName();
        if (auto result = mlir::dyn_cast<mlir::OpResult>(original))
          diagnostic << '#' << result.getResultNumber();
        diagnostic << "; operands=[";
        for (auto [index, operand] :
             llvm::enumerate(definition->getOperands())) {
          if (index != 0)
            diagnostic << ',';
          if (mlir::Operation *operandDefinition = operand.getDefiningOp())
            diagnostic << operandDefinition->getName();
          else if (auto operandArgument =
                       mlir::dyn_cast<mlir::BlockArgument>(operand))
            diagnostic << "block-argument#" << operandArgument.getArgNumber();
          else
            diagnostic << "unknown";
        }
        diagnostic << "]; users=[";
        for (auto [index, use] : llvm::enumerate(original.getUses())) {
          if (index != 0)
            diagnostic << ',';
          diagnostic << use.getOwner()->getName() << ":operand#"
                     << use.getOperandNumber();
        }
        diagnostic << ']';
        diagnostic << "; definition_ir=";
        definition->print(diagnostic, mlir::OpPrintingFlags().skipRegions());
      } else {
        diagnostic << "; definition=unknown";
      }
      return failValue(detail);
    }

    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(original.getType());
    if (!tensorType)
      return failValue("cannot materialize non-ranked-tensor value");

    auto externalType =
        mlir::dyn_cast<mlir::MemRefType>(externalIt->second.getType());
    auto destination = builder.create<mlir::memref::AllocOp>(
        materializationLoc, makeSPMMemRefType(tensorType, MemLayout::Tensor));
    builder.create<StorageLoadOp>(materializationLoc, externalIt->second,
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
  auto materialize = builder.create<LayoutMaterializeOp>(materializationLoc,
                                                         resultType, source);
  record(original, targetLayout, materialize.getResult());
  return materialize.getResult();
}

mlir::FailureOr<mlir::Value>
TileRegionBodyEmitter::getOrMaterializeStructuredInput(
    mlir::Value original, MemLayout targetLayout, mlir::OpBuilder &builder) {
  mlir::FailureOr<mlir::Value> result =
      getOrMaterialize(original, targetLayout, builder);
  if (mlir::succeeded(result))
    for (uint32_t node : activeStructuredNodes)
      recordOperandBuffer(node, *result);
  return result;
}

void TileRegionBodyEmitter::recordOperationResultBuffer(
    uint32_t structuredNodeId, mlir::Value buffer) {
  if (!emissionRelations || !buffer)
    return;
  auto &relations =
      emissionRelations->materializedBuffers.operationResultBuffers;
  if (!llvm::any_of(relations, [&](const auto &relation) {
        return relation.structuredNodeId == structuredNodeId &&
               relation.buffer == buffer;
      }))
    relations.push_back({structuredNodeId, buffer});
}

void TileRegionBodyEmitter::recordOperandBuffer(uint32_t structuredNodeId,
                                                mlir::Value buffer) {
  if (!emissionRelations || !buffer)
    return;
  auto &relations = emissionRelations->materializedBuffers.operandBuffers;
  if (!llvm::any_of(relations, [&](const auto &relation) {
        return relation.structuredNodeId == structuredNodeId &&
               relation.buffer == buffer;
      }))
    relations.push_back({structuredNodeId, buffer});
}

void TileRegionBodyEmitter::recordOutputBuffer(unsigned outputIndex,
                                               mlir::Value buffer) {
  if (!emissionRelations || !buffer)
    return;
  auto &relations = emissionRelations->materializedBuffers.outputBuffers;
  if (!llvm::any_of(relations, [&](const auto &relation) {
        return relation.outputIndex == outputIndex &&
               relation.buffer == buffer;
      }))
    relations.push_back({outputIndex, buffer});
}

mlir::LogicalResult
TileRegionBodyEmitter::initializeBoundary(TensorProgramScope scope,
                                          TileRegionOp tileRegion,
                                          mlir::OpBuilder &builder) {
  mlir::Block &sourceBlock = scope.getBody();
  mlir::Block &tileBlock = tileRegion.getBody().front();
  if (sourceBlock.getNumArguments() != scope.getInputCount() +
                                          scope.getOutputCount() +
                                          scope.getBoundaryArgumentCount())
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
    if (argIndex >= inputCount &&
        argIndex < inputCount + scope.getOutputCount()) {
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
  auto convertStructured =
      [&](llvm::function_ref<mlir::LogicalResult()> convert) {
        llvm::SmallVector<uint32_t, 2> previous = activeStructuredNodes;
        auto node = structuredNodeIds.find(op);
        activeStructuredNodes.clear();
        if (node != structuredNodeIds.end())
          activeStructuredNodes.append(node->second.begin(),
                                       node->second.end());
        mlir::LogicalResult result = convert();
        if (mlir::succeeded(result)) {
          for (mlir::Value value : op->getResults()) {
            MemLayout layout = MemLayout::Tensor;
            if (mlir::Value buffer = lookupAny(value, layout))
              for (uint32_t current : activeStructuredNodes)
                recordOperationResultBuffer(current, buffer);
          }
        }
        activeStructuredNodes = std::move(previous);
        return result;
      };

  if (mlir::isa<WaferLinalgExtCollectiveOpInterface>(op))
    return convertStructured(
        [&] { return convertLinalgExtCollective(op, builder); });

  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(op)) {
    // Affine window expressions are the canonical structured representation
    // of convolution. Admit them only when Linalg itself proves convolution
    // dimensions; all other non-projected maps remain unsupported rather than
    // falling through a generic or name-based route. A scalar/reduction
    // boundary whose result dims are constant positions (extent one, the
    // complete result slice of an unpartitioned coordinate) is admitted on
    // top of projected permutations.
    const bool mapsAreProjectedOrConstant = [&] {
      for (mlir::Attribute attribute : linalg.getIndexingMaps()) {
        auto mapAttribute = mlir::dyn_cast<mlir::AffineMapAttr>(attribute);
        if (!mapAttribute)
          return false;
        for (mlir::AffineExpr expression : mapAttribute.getValue().getResults())
          if (!mlir::isa<mlir::AffineDimExpr, mlir::AffineConstantExpr>(
                  expression))
            return false;
      }
      return true;
    }();
    if (!linalg.hasOnlyProjectedPermutations() &&
        !mapsAreProjectedOrConstant &&
        mlir::failed(mlir::linalg::inferConvolutionDims(linalg)))
      return fail("unsupported linalg indexing maps");
    return convertStructured(
        [&] { return convertStructuredOp(op, builder); });
  }

  if (mlir::isa<
          mlir::affine::AffineApplyOp, mlir::arith::ConstantOp,
          mlir::bufferization::MaterializeInDestinationOp,
          mlir::bufferization::ToMemrefOp, mlir::bufferization::ToTensorOp,
          mlir::tensor::EmptyOp, mlir::memref::AllocOp, mlir::tensor::ExtractOp,
          mlir::tensor::PadOp, mlir::tensor::ExtractSliceOp,
          mlir::tensor::InsertSliceOp, mlir::tensor::ExpandShapeOp,
          mlir::tensor::CollapseShapeOp, mlir::scf::IfOp, mlir::scf::ForOp>(op))
    return convertSupportOp(op, builder);

  return fail("unsupported tensor-program op " +
              op->getName().getStringRef().str());
}

mlir::LogicalResult
TileRegionBodyEmitter::emitPeerEndpoint(const CandidatePeerEndpoint &endpoint,
                                        mlir::OpBuilder &builder) {
  mlir::FailureOr<mlir::Value> buffer =
      getOrMaterialize(endpoint.value, MemLayout::Tensor, builder);
  if (mlir::failed(buffer))
    return mlir::failure();
  if (endpoint.bytes == 0 ||
      endpoint.bytes >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      endpoint.communicationId < 0 || endpoint.payloadSlice < 0)
    return fail("selected peer endpoint fields are not representable");
  auto message =
      DTEMessageAttr::get(builder.getContext(), endpoint.communicationId,
                          /*round=*/0, endpoint.payloadSlice);
  mlir::Location endpointLoc = endpoint.value.getLoc();
  mlir::Value token;
  if (endpoint.kind == CandidatePeerEndpointKind::Send) {
    token =
        builder
            .create<CommPeerSendOp>(
                endpointLoc, mlir::async::TokenType::get(builder.getContext()),
                *buffer, builder.getI64IntegerAttr(endpoint.peer.getValue()),
                builder.getI64IntegerAttr(endpoint.bytes), message)
            .getToken();
  } else {
    auto toTensor =
        endpoint.value.getDefiningOp<mlir::bufferization::ToTensorOp>();
    auto allocation =
        toTensor ? toTensor.getMemref().getDefiningOp<mlir::memref::AllocOp>()
                 : mlir::memref::AllocOp();
    if (!toTensor || !allocation || !isWaferSPMMemRefType(allocation.getType()))
      return fail("selected peer receive requires one direct static SPM "
                  "allocation-backed tensor");
    token =
        builder
            .create<CommPeerRecvOp>(
                endpointLoc, mlir::async::TokenType::get(builder.getContext()),
                *buffer, builder.getI64IntegerAttr(endpoint.peer.getValue()),
                builder.getI64IntegerAttr(endpoint.bytes), message)
            .getToken();
  }
  builder.create<mlir::async::AwaitOp>(endpointLoc, token);
  return mlir::success();
}

mlir::LogicalResult
TileRegionBodyEmitter::convertStructuredOp(mlir::Operation *operation,
                                           mlir::OpBuilder &builder) {
  if (auto fill = mlir::dyn_cast<mlir::linalg::FillOp>(operation))
    return convertFill(fill, builder);
  auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation);
  if (linalg && mlir::succeeded(mlir::linalg::inferConvolutionDims(linalg)))
    return convertConvolution(linalg, builder);
  if (mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::MatmulTransposeAOp,
                mlir::linalg::MatmulTransposeBOp>(operation))
    return convertMatmul(mlir::cast<mlir::linalg::LinalgOp>(operation),
                         builder);
  if (mlir::isa<mlir::linalg::BatchMatmulOp,
                mlir::linalg::BatchMatmulTransposeAOp,
                mlir::linalg::BatchMatmulTransposeBOp>(operation))
    return convertBatchMatmul(mlir::cast<mlir::linalg::LinalgOp>(operation),
                              builder);
  if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(operation)) {
    mlir::linalg::LinalgOp linalg = generic;
    if (hasExactGemmPayload(linalg) &&
        mlir::succeeded(inferRank2GemmOrientations(linalg)))
      return convertMatmul(linalg, builder);
    return convertGeneric(generic, builder);
  }
  return fail("unsupported structured source operation: " +
              operation->getName().getStringRef().str());
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

    // A per-output spatial mapping leaves results not owned by this physical
    // Tile equal to its private scheduling destination. Preserve the common
    // full-card result ABI while emitting no load, compute, or store for that
    // result.
    if (value == outputArgument) {
      yieldedValues.push_back(output);
      continue;
    }

    auto getReshapeBase = [](mlir::Value current) {
      llvm::DenseSet<mlir::Value> visited;
      while (visited.insert(current).second) {
        if (auto collapse =
                current.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
          current = collapse.getSrc();
          continue;
        }
        if (auto expand =
                current.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
          current = expand.getSrc();
          continue;
        }
        break;
      }
      return current;
    };
    if (auto directIt = directYieldBuffers.find(value);
        directIt != directYieldBuffers.end()) {
      // A functional result may add/remove unit dimensions after the target
      // compute. Its restored scheduling destination is then an exact memref
      // reshape view of the public output. Compare typed SSA bases rather than
      // raw
      // value identity; arbitrary subviews or offset-changing aliases remain
      // rejected.
      if (getReshapeBase(directIt->second) != getReshapeBase(output))
        return fail("direct boundary storeback target mismatch");
      recordOutputBuffer(index, directIt->second);
      yieldedValues.push_back(output);
      continue;
    }

    mlir::FailureOr<mlir::Value> tensorBuffer =
        getOrMaterialize(value, MemLayout::Tensor, builder);
    if (mlir::failed(tensorBuffer))
      return mlir::failure();
    recordOutputBuffer(index, *tensorBuffer);
    builder.create<StorageStoreOp>(value.getLoc(), *tensorBuffer, output);
    yieldedValues.push_back(output);
  }

  builder.create<TileYieldOp>(scope.getLoc(), yieldedValues);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
