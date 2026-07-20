//===- FullBufferHandoff.cpp - Explicit cross-task SPM residency --------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace wafer::tensor_program_scheduling {
namespace {

struct ExternalConsumerPath {
  TileRegionOp consumer;
  unsigned operandIndex = 0;
  mlir::Value terminalValue;
  llvm::SmallVector<mlir::Operation *, 2> views;
};

struct ConsumerRDMARewrite {
  InstrRDMAOp rdma;
  llvm::SmallVector<mlir::Operation *, 2> views;
};

struct ConsumerRewrite {
  ExternalConsumerPath path;
  mlir::BlockArgument blockArgument;
  llvm::SmallVector<ConsumerRDMARewrite, 2> rdmas;
  llvm::SmallVector<mlir::Operation *, 2> localViews;
};

static std::optional<int64_t> getPhysicalBytes(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape())
    return std::nullopt;
  auto encoding = mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
      memrefType.getMemorySpace());
  if (!encoding)
    return std::nullopt;
  mlir::FailureOr<int64_t> bytes =
      encoding.getPhysicalFootprintBytes(memrefType);
  if (mlir::failed(bytes) || *bytes <= 0)
    return std::nullopt;
  return *bytes;
}

static std::optional<int64_t> getValidElements(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape())
    return std::nullopt;
  auto encoding = mlir::dyn_cast_or_null<WaferPhysicalEncodingAttrInterface>(
      memrefType.getMemorySpace());
  if (!encoding)
    return std::nullopt;
  mlir::FailureOr<int64_t> elements = encoding.getValidElementCount(memrefType);
  return mlir::succeeded(elements) ? std::optional<int64_t>(*elements)
                                   : std::nullopt;
}

static bool isUnitDescriptor(llvm::ArrayRef<int64_t> strides,
                             llvm::ArrayRef<int64_t> iterations) {
  return strides.size() == 3 && iterations.size() == 3 &&
         llvm::all_of(strides, [](int64_t value) { return value == 0; }) &&
         llvm::all_of(iterations, [](int64_t value) { return value == 1; });
}

static bool isCompleteWDMA(InstrWDMAOp wdma, int64_t expectedBytes) {
  return wdma.getByteCountAttr().getInt() == expectedBytes &&
         wdma.getInnerBytesAttr().getInt() == expectedBytes &&
         isUnitDescriptor(wdma.getDstStrides(), wdma.getDstIterations());
}

static bool isCompleteRDMA(InstrRDMAOp rdma, int64_t expectedBytes) {
  std::optional<int64_t> destBytes = getPhysicalBytes(rdma.getDest().getType());
  return destBytes && *destBytes == expectedBytes &&
         rdma.getByteCountAttr().getInt() == expectedBytes &&
         rdma.getInnerBytesAttr().getInt() == expectedBytes &&
         isUnitDescriptor(rdma.getSrcStrides(), rdma.getSrcIterations());
}

static bool isRegionOwnedAllocation(mlir::Value value, TileRegionOp owner) {
  while (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
             value.getDefiningOp()))
    value = view.getViewSource();
  auto allocation = value.getDefiningOp<mlir::memref::AllocOp>();
  if (!allocation)
    return false;
  for (mlir::Operation *parent = allocation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent == owner.getOperation())
      return true;
    if (mlir::isa<TileRegionOp>(parent))
      return false;
  }
  return false;
}

static bool hasEquivalentStaticStorage(mlir::Type source, mlir::Type result) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source);
  auto resultType = mlir::dyn_cast<mlir::MemRefType>(result);
  MemoryAttr sourceMemory =
      sourceType ? getWaferMemoryAttr(sourceType) : MemoryAttr{};
  MemoryAttr resultMemory =
      resultType ? getWaferMemoryAttr(resultType) : MemoryAttr{};
  std::optional<int64_t> sourceBytes = getPhysicalBytes(source);
  std::optional<int64_t> resultBytes = getPhysicalBytes(result);
  std::optional<int64_t> sourceElements = getValidElements(source);
  std::optional<int64_t> resultElements = getValidElements(result);
  return sourceType && resultType && sourceType.hasStaticShape() &&
         resultType.hasStaticShape() && sourceMemory && resultMemory &&
         sourceMemory.getLayout() == MemLayout::Tensor &&
         resultMemory.getLayout() == MemLayout::Tensor &&
         sourceType.getElementType() == resultType.getElementType() &&
         sourceBytes && resultBytes && *sourceBytes == *resultBytes &&
         sourceElements && resultElements && *sourceElements == *resultElements;
}

static analysis::IndexRelationResult
getStaticViewRelation(mlir::Operation *operation, mlir::Value source) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto resultType =
      operation && operation->getNumResults() == 1
          ? mlir::dyn_cast<mlir::MemRefType>(operation->getResult(0).getType())
          : mlir::MemRefType{};
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return {analysis::IndexRelationStatus::Unsupported, std::nullopt,
            "static view relation requires static memrefs"};

  if (auto collapse =
          mlir::dyn_cast<mlir::memref::CollapseShapeOp>(operation)) {
    if (collapse.getSrc() != source)
      return {analysis::IndexRelationStatus::Invalid, std::nullopt,
              "collapse source mismatch"};
    return analysis::IndexRelation::staticReshape(resultType.getShape(),
                                                  sourceType.getShape());
  }
  if (auto expand = mlir::dyn_cast<mlir::memref::ExpandShapeOp>(operation)) {
    if (expand.getSrc() != source || !expand.getOutputShape().empty())
      return {analysis::IndexRelationStatus::Unsupported, std::nullopt,
              "dynamic expand shape is unsupported"};
    return analysis::IndexRelation::staticReshape(resultType.getShape(),
                                                  sourceType.getShape());
  }
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(operation)) {
    if (cast.getSource() != source)
      return {analysis::IndexRelationStatus::Invalid, std::nullopt,
              "cast source mismatch"};
    return analysis::IndexRelation::staticReshape(resultType.getShape(),
                                                  sourceType.getShape());
  }
  if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(operation)) {
    if (subview.getSource() != source ||
        sourceType.getRank() != resultType.getRank())
      return {analysis::IndexRelationStatus::Unsupported, std::nullopt,
              "rank-reducing subview is unsupported"};
    return analysis::IndexRelation::staticSlice(
        resultType.getShape(), sourceType.getShape(),
        subview.getStaticOffsets(), subview.getStaticStrides());
  }
  return {analysis::IndexRelationStatus::Unsupported, std::nullopt,
          "operation has no supported static view relation"};
}

static bool provesCompleteStaticView(mlir::Operation *operation,
                                     mlir::Value source) {
  if (!operation || operation->getNumResults() != 1 ||
      !hasEquivalentStaticStorage(source.getType(),
                                  operation->getResult(0).getType()))
    return false;
  auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());
  auto resultType =
      mlir::cast<mlir::MemRefType>(operation->getResult(0).getType());
  analysis::IndexRelationResult relation =
      getStaticViewRelation(operation, source);
  analysis::IndexSetResult destinationDomain =
      analysis::IndexRelation::staticDomain(resultType.getShape());
  analysis::IndexSetResult sourceDomain =
      analysis::IndexRelation::staticDomain(sourceType.getShape());
  if (!relation.isExact() || !destinationDomain.isExact() ||
      !sourceDomain.isExact() || !relation.get()->isBijective().isProvenTrue())
    return false;
  analysis::IndexSetResult image =
      relation.get()->image(*destinationDomain.set);
  return image.isExact() && image.set->isEqual(*sourceDomain.set);
}

static bool isStaticExternalView(mlir::Operation *operation,
                                 mlir::Value source) {
  if (!provesCompleteStaticView(operation, source))
    return false;
  if (auto collapse = mlir::dyn_cast<mlir::memref::CollapseShapeOp>(operation))
    return collapse.getSrc() == source;
  if (auto expand = mlir::dyn_cast<mlir::memref::ExpandShapeOp>(operation))
    return expand.getSrc() == source && expand.getOutputShape().empty();
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(operation))
    return cast.getSource() == source;
  return false;
}

static bool isStaticFullSubview(mlir::memref::SubViewOp subview,
                                mlir::Value source);

static bool isStaticFullStorageView(mlir::Operation *operation,
                                    mlir::Value source) {
  if (isStaticExternalView(operation, source))
    return true;
  auto subview = mlir::dyn_cast_or_null<mlir::memref::SubViewOp>(operation);
  return subview && isStaticFullSubview(subview, source);
}

static bool collectExternalConsumers(
    mlir::Value value, llvm::ArrayRef<mlir::Operation *> prefix,
    llvm::SmallVectorImpl<ExternalConsumerPath> &paths,
    llvm::SmallPtrSetImpl<mlir::Operation *> &externalViews,
    llvm::DenseSet<mlir::Value> &active) {
  if (!active.insert(value).second)
    return false;
  auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };
  if (value.use_empty())
    return finish(false);

  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto consumer = mlir::dyn_cast<TileRegionOp>(owner)) {
      if (use.getOperandNumber() >= consumer.getInputs().size())
        return finish(false);
      paths.push_back(ExternalConsumerPath{
          consumer, use.getOperandNumber(), value,
          llvm::SmallVector<mlir::Operation *, 2>(prefix)});
      continue;
    }

    if (!isStaticExternalView(owner, value))
      return finish(false);
    externalViews.insert(owner);
    llvm::SmallVector<mlir::Operation *, 2> next(prefix);
    next.push_back(owner);
    if (!collectExternalConsumers(owner->getResult(0), next, paths,
                                  externalViews, active))
      return finish(false);
  }
  return finish(true);
}

static bool isStaticFullSubview(mlir::memref::SubViewOp subview,
                                mlir::Value source) {
  return provesCompleteStaticView(subview, source);
}

static bool
collectConsumerRDMAs(mlir::Value value, int64_t expectedBytes,
                     llvm::ArrayRef<mlir::Operation *> prefix,
                     llvm::SmallVectorImpl<ConsumerRDMARewrite> &rdmas,
                     llvm::SmallVectorImpl<mlir::Operation *> &views,
                     llvm::DenseSet<mlir::Value> &active) {
  if (!active.insert(value).second)
    return false;
  auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };
  if (value.use_empty())
    return finish(false);

  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(owner)) {
      if (use.getOperandNumber() != 0 || !isCompleteRDMA(rdma, expectedBytes))
        return finish(false);
      rdmas.push_back(ConsumerRDMARewrite{
          rdma, llvm::SmallVector<mlir::Operation *, 2>(prefix)});
      continue;
    }
    if (use.getOperandNumber() != 0 || !isStaticFullStorageView(owner, value))
      return finish(false);
    views.push_back(owner);
    llvm::SmallVector<mlir::Operation *, 2> next(prefix);
    next.push_back(owner);
    if (!collectConsumerRDMAs(owner->getResult(0), expectedBytes, next, rdmas,
                              views, active))
      return finish(false);
  }
  return finish(true);
}

static mlir::MemRefType getSPMType(mlir::MemRefType type) {
  MemoryAttr memory = getWaferMemoryAttr(type);
  if (!memory)
    return {};
  return mlir::MemRefType::get(
      type.getShape(), type.getElementType(), type.getLayout(),
      MemoryAttr::get(type.getContext(), MemorySpace::SPM, memory.getLayout()));
}

static mlir::Value cloneStaticViewPath(mlir::OpBuilder &builder,
                                       llvm::ArrayRef<mlir::Operation *> views,
                                       mlir::Value source) {
  mlir::Value current = source;
  for (mlir::Operation *operation : views) {
    auto oldType =
        mlir::cast<mlir::MemRefType>(operation->getResult(0).getType());
    mlir::MemRefType newType = getSPMType(oldType);
    if (auto collapse =
            mlir::dyn_cast<mlir::memref::CollapseShapeOp>(operation)) {
      current = builder
                    .create<mlir::memref::CollapseShapeOp>(
                        collapse.getLoc(), newType, current,
                        collapse.getReassociationIndices())
                    .getResult();
      continue;
    }
    if (auto expand = mlir::dyn_cast<mlir::memref::ExpandShapeOp>(operation)) {
      current = builder
                    .create<mlir::memref::ExpandShapeOp>(
                        expand.getLoc(), newType, current,
                        expand.getReassociationIndices())
                    .getResult();
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(operation)) {
      current =
          builder.create<mlir::memref::CastOp>(cast.getLoc(), newType, current);
      continue;
    }
    auto subview = mlir::cast<mlir::memref::SubViewOp>(operation);
    current =
        builder
            .create<mlir::memref::SubViewOp>(
                subview.getLoc(), newType, current, subview.getMixedOffsets(),
                subview.getMixedSizes(), subview.getMixedStrides())
            .getResult();
  }
  return current;
}

static mlir::Value cloneViewPathIntoRegion(const ExternalConsumerPath &path,
                                           mlir::BlockArgument source) {
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(source.getOwner());
  return cloneStaticViewPath(builder, path.views, source);
}

static bool
collectProducerDestination(mlir::Value value, InstrWDMAOp expectedWDMA,
                           TileYieldOp yield, unsigned resultIndex,
                           llvm::SmallVectorImpl<mlir::Operation *> &views,
                           llvm::DenseSet<mlir::Value> &active, bool &sawWDMA) {
  if (!active.insert(value).second)
    return false;
  auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (owner == yield.getOperation() && use.getOperandNumber() == resultIndex)
      continue;
    if (owner == expectedWDMA.getOperation() && use.getOperandNumber() == 1) {
      if (sawWDMA)
        return finish(false);
      sawWDMA = true;
      continue;
    }
    if (use.getOperandNumber() != 0 || !isStaticFullStorageView(owner, value))
      return finish(false);
    views.push_back(owner);
    if (!collectProducerDestination(owner->getResult(0), expectedWDMA, yield,
                                    resultIndex, views, active, sawWDMA))
      return finish(false);
  }
  return finish(true);
}

static void eraseDeadViews(llvm::ArrayRef<mlir::Operation *> views) {
  llvm::SmallPtrSet<mlir::Operation *, 8> unique(views.begin(), views.end());
  llvm::SmallVector<mlir::Operation *, 8> pending(unique.begin(), unique.end());
  bool changed = true;
  while (changed && !pending.empty()) {
    changed = false;
    for (size_t index = 0; index < pending.size();) {
      mlir::Operation *operation = pending[index];
      if (!operation->use_empty()) {
        ++index;
        continue;
      }
      pending.erase(pending.begin() + index);
      operation->erase();
      changed = true;
    }
  }
}

static bool tryPromoteResult(TileRegionOp producer, unsigned resultIndex) {
  if (producer.getBody().empty() || resultIndex >= producer.getNumResults())
    return false;
  auto oldResultType = mlir::dyn_cast<mlir::MemRefType>(
      producer.getResult(resultIndex).getType());
  MemoryAttr oldResultMemory =
      oldResultType ? getWaferMemoryAttr(oldResultType) : MemoryAttr{};
  if (!oldResultType || !isWaferDDRMemRefType(oldResultType) ||
      !oldResultMemory || oldResultMemory.getLayout() != MemLayout::Tensor)
    return false;
  std::optional<int64_t> resultBytes = getPhysicalBytes(oldResultType);
  if (!resultBytes)
    return false;

  auto yield =
      mlir::dyn_cast<TileYieldOp>(producer.getBody().front().getTerminator());
  if (!yield || resultIndex >= yield.getValues().size())
    return false;
  auto outputArgument =
      mlir::dyn_cast<mlir::BlockArgument>(yield.getValues()[resultIndex]);
  if (!outputArgument ||
      outputArgument.getOwner() != &producer.getBody().front())
    return false;

  unsigned outputArgumentIndex = outputArgument.getArgNumber();
  if (outputArgumentIndex >= producer.getInputs().size())
    return false;
  mlir::Value oldOutput = producer.getInputs()[outputArgumentIndex];
  auto oldOutputAllocation = oldOutput.getDefiningOp<mlir::memref::AllocOp>();
  if (!oldOutputAllocation ||
      !isWaferDDRMemRefType(oldOutputAllocation.getType()) ||
      !oldOutput.hasOneUse())
    return false;
  mlir::OpOperand &oldOutputUse = *oldOutput.getUses().begin();
  if (oldOutputUse.getOwner() != producer.getOperation() ||
      oldOutputUse.getOperandNumber() != outputArgumentIndex)
    return false;

  InstrWDMAOp producerWDMA;
  for (InstrWDMAOp wdma : producer.getBody().getOps<InstrWDMAOp>()) {
    if (!hasEquivalentStaticStorage(wdma.getDest().getType(),
                                    outputArgument.getType()))
      continue;
    if (producerWDMA)
      return false;
    producerWDMA = wdma;
  }
  if (!producerWDMA || !isCompleteWDMA(producerWDMA, *resultBytes))
    return false;
  auto residentType =
      mlir::dyn_cast<mlir::MemRefType>(producerWDMA.getSource().getType());
  auto residentAllocation =
      producerWDMA.getSource().getDefiningOp<mlir::memref::AllocOp>();
  mlir::MemRefType expectedResidentType = getSPMType(oldResultType);
  if (!residentType || !isWaferSPMMemRefType(residentType) ||
      residentType != expectedResidentType || !residentAllocation ||
      !isRegionOwnedAllocation(producerWDMA.getSource(), producer))
    return false;

  for (mlir::Operation *operation = producerWDMA->getNextNode(); operation;
       operation = operation->getNextNode())
    if (operation != yield.getOperation() &&
        !mlir::isa<SyncLocalFenceOp>(operation))
      return false;

  llvm::SmallVector<mlir::Operation *, 2> producerDestinationViews;
  llvm::DenseSet<mlir::Value> activeProducerDestination;
  bool sawProducerWDMA = false;
  if (!collectProducerDestination(outputArgument, producerWDMA, yield,
                                  resultIndex, producerDestinationViews,
                                  activeProducerDestination, sawProducerWDMA) ||
      !sawProducerWDMA)
    return false;

  llvm::SmallVector<ExternalConsumerPath, 4> paths;
  llvm::SmallPtrSet<mlir::Operation *, 4> externalViewSet;
  llvm::DenseSet<mlir::Value> activeExternal;
  if (!collectExternalConsumers(producer.getResult(resultIndex), {}, paths,
                                externalViewSet, activeExternal) ||
      paths.empty())
    return false;

  llvm::SmallVector<ConsumerRewrite, 4> rewrites;
  llvm::DenseSet<std::pair<mlir::Operation *, unsigned>> seenOperands;
  for (ExternalConsumerPath &path : paths) {
    if (!seenOperands.insert({path.consumer.getOperation(), path.operandIndex})
             .second ||
        path.consumer.getBody().empty() ||
        path.operandIndex >= path.consumer.getInputs().size() ||
        path.consumer.getInputs()[path.operandIndex] != path.terminalValue)
      return false;
    mlir::BlockArgument argument =
        path.consumer.getBody().front().getArgument(path.operandIndex);
    std::optional<int64_t> terminalBytes = getPhysicalBytes(argument.getType());
    if (!terminalBytes || *terminalBytes != *resultBytes)
      return false;
    ConsumerRewrite rewrite{path, argument};
    llvm::DenseSet<mlir::Value> activeConsumer;
    if (!collectConsumerRDMAs(argument, *resultBytes, {}, rewrite.rdmas,
                              rewrite.localViews, activeConsumer) ||
        rewrite.rdmas.empty())
      return false;
    rewrites.push_back(std::move(rewrite));
  }

  mlir::OpResult producerResult =
      mlir::cast<mlir::OpResult>(producer.getResult(resultIndex));
  producerResult.setType(residentType);
  yield->setOperand(resultIndex, producerWDMA.getSource());
  producerWDMA.erase();
  eraseDeadViews(producerDestinationViews);

  llvm::SmallVector<mlir::Operation *, 8> localViews;
  for (ConsumerRewrite &rewrite : rewrites) {
    rewrite.path.consumer->setOperand(rewrite.path.operandIndex,
                                      producerResult);
    rewrite.blockArgument.setType(residentType);
    mlir::Value terminalSPM =
        cloneViewPathIntoRegion(rewrite.path, rewrite.blockArgument);
    for (ConsumerRDMARewrite &rdmaRewrite : rewrite.rdmas) {
      InstrRDMAOp rdma = rdmaRewrite.rdma;
      mlir::OpBuilder viewBuilder(rewrite.blockArgument.getContext());
      if (mlir::Operation *definition = terminalSPM.getDefiningOp())
        viewBuilder.setInsertionPointAfter(definition);
      else
        viewBuilder.setInsertionPointToStart(rewrite.blockArgument.getOwner());
      mlir::Value localSource =
          cloneStaticViewPath(viewBuilder, rdmaRewrite.views, terminalSPM);
      mlir::OpBuilder builder(rdma);
      builder.create<InstrGatherScatterOp>(
          rdma.getLoc(), localSource, rdma.getDest(), rdma.getByteCountAttr(),
          rdma.getInnerBytesAttr(), mlir::IntegerAttr{}, mlir::IntegerAttr{},
          rdma.getSrcStridesAttr(), rdma.getSrcIterationsAttr(),
          rdma.getSrcStridesAttr(), rdma.getSrcIterationsAttr());
      rdma.erase();
    }
    localViews.append(rewrite.localViews.begin(), rewrite.localViews.end());
  }
  eraseDeadViews(localViews);

  llvm::SmallVector<mlir::Operation *, 8> externalViews(externalViewSet.begin(),
                                                        externalViewSet.end());
  eraseDeadViews(externalViews);

  if (outputArgument.use_empty()) {
    producer.getInputsMutable().erase(outputArgumentIndex);
    producer.getBody().front().eraseArgument(outputArgumentIndex);
  }

  if (oldOutputAllocation->use_empty())
    oldOutputAllocation.erase();
  return true;
}

} // namespace

unsigned promoteFullBufferHandoffs(mlir::ModuleOp module) {
  llvm::SmallVector<std::pair<TileRegionOp, unsigned>, 8> candidates;
  module.walk([&](TileRegionOp region) {
    for (unsigned index = 0; index < region.getNumResults(); ++index)
      candidates.push_back({region, index});
  });

  unsigned promoted = 0;
  for (auto [region, index] : candidates)
    if (region->getBlock() && index < region.getNumResults() &&
        tryPromoteResult(region, index))
      ++promoted;
  return promoted;
}

} // namespace wafer::tensor_program_scheduling
