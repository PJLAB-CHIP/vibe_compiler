//===- WaferTensorProgramToCardProgram.cpp - Card spatial mapping ------===//

#include "Wafer/Conversion/WaferTensorProgramToCardProgram/WaferTensorProgramToCardProgram.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/Target/PhysicalTopology.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace wafer {
namespace {

static mlir::LogicalResult failCardProgram(std::string *failureReason,
                                           llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

template <typename T>
static mlir::FailureOr<T> failCardProgramValue(std::string *failureReason,
                                               llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

static bool isCardSharedDeclaration(mlir::Operation &operation) {
  if (!mlir::isa<mlir::SymbolOpInterface>(&operation))
    return false;
  return llvm::all_of(operation.getRegions(),
                      [](mlir::Region &region) { return region.empty(); });
}

static bool relationsBelongTo(
    mlir::Operation *root,
    const StructuredMaterializationRelations &relations) {
  llvm::DenseSet<const void *> liveValues;
  root->walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults())
      liveValues.insert(result.getAsOpaquePointer());
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          liveValues.insert(argument.getAsOpaquePointer());
  });
  auto allLive = [&](const auto &entries) {
    return llvm::all_of(entries, [&](const auto &entry) {
      return entry.buffer &&
             liveValues.contains(entry.buffer.getAsOpaquePointer());
    });
  };
  return allLive(relations.operationResultBuffers) &&
         allLive(relations.operandBuffers) && allLive(relations.outputBuffers);
}

static mlir::FailureOr<mlir::func::FuncOp>
getSourceTensorProgram(mlir::ModuleOp module, std::string *failureReason) {
  mlir::func::FuncOp program;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (program)
      return failCardProgramValue<mlir::func::FuncOp>(
          failureReason,
          "card spatial mapping requires exactly one defined direct "
          "tensor-program function");
    program = function;
  }
  if (!program)
    return failCardProgramValue<mlir::func::FuncOp>(
        failureReason,
        "card spatial mapping requires exactly one defined direct "
        "tensor-program function");
  if (!program.getBody().hasOneBlock())
    return failCardProgramValue<mlir::func::FuncOp>(
        failureReason, "card spatial mapping requires a single-block tensor "
                       "program");
  if (program.getNumResults() == 0)
    return failCardProgramValue<mlir::func::FuncOp>(
        failureReason, "card spatial mapping requires tensor-program results");

  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      program.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != program.getNumResults())
    return failCardProgramValue<mlir::func::FuncOp>(
        failureReason,
        "card spatial mapping requires a complete functional return");
  for (unsigned index = 0; index < program.getNumResults(); ++index) {
    mlir::Type resultType = program.getResultTypes()[index];
    if (!mlir::isa<mlir::RankedTensorType>(resultType) ||
        returnOp.getOperand(index).getType() != resultType)
      return failCardProgramValue<mlir::func::FuncOp>(
          failureReason, "card spatial mapping requires ranked tensor results");
  }
  return program;
}

/// Builds the private scheduling form consumed by TileRegion materialization.
/// Source arguments remain untouched and keep their ordinary frontend ABI.
/// One compiler-owned destination is appended per result only inside this
/// private clone; it is removed again before the CardProgram is exposed.
static mlir::LogicalResult
appendSchedulingOutputDestinations(mlir::func::FuncOp program,
                                   std::string *failureReason) {
  for (mlir::Type resultType : program.getResultTypes()) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType || !tensorType.hasStaticShape())
      return failCardProgram(
          failureReason,
          "card scheduling destinations require static ranked results");
    program.insertArgument(program.getNumArguments(), resultType,
                           mlir::DictionaryAttr{}, program.getLoc());
  }
  if (mlir::failed(mlir::verify(program)))
    return failCardProgram(
        failureReason,
        "private card scheduling boundary is not verifier-legal");
  return mlir::success();
}

static mlir::LogicalResult verifyLogicalMesh(mlir::ModuleOp module,
                                             std::string *failureReason) {
  llvm::SmallVector<ExecutionMeshOp, 2> meshes(
      module.getOps<ExecutionMeshOp>());
  if (meshes.size() != 1)
    return failCardProgram(
        failureReason,
        "card spatial mapping requires exactly one direct logical "
        "execution mesh");

  int64_t partitionCount = 1;
  for (int64_t dimension : meshes.front().getShapeAttr().asArrayRef()) {
    if (dimension <= 0 ||
        partitionCount > std::numeric_limits<int64_t>::max() / dimension)
      return failCardProgram(failureReason,
                             "logical execution mesh is not representable");
    partitionCount *= dimension;
  }
  if (partitionCount != 1)
    return failCardProgram(
        failureReason,
        "card spatial mapping currently requires one logical card "
        "partition");
  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>
getStaticOutputDomains(mlir::func::FuncOp program, std::string *failureReason) {
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> domains;
  domains.reserve(program.getNumResults());
  for (mlir::Type type : program.getResultTypes()) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType || !tensorType.hasStaticShape())
      return failCardProgramValue<
          llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>(
          failureReason, "card spatial mapping requires static tensor-program "
                         "output shapes");
    if (tensorType.getRank() == 0 ||
        llvm::any_of(tensorType.getShape(),
                     [](int64_t extent) { return extent <= 0; }))
      return failCardProgramValue<
          llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>(
          failureReason, "card spatial mapping requires positive-rank "
                         "nonempty output domains");
    domains.emplace_back(tensorType.getShape());
  }
  return domains;
}

static mlir::FailureOr<mlir::func::FuncOp>
takeLoweredTensorProgram(mlir::ModuleOp shardModule,
                         std::string *failureReason) {
  mlir::func::FuncOp program;
  for (mlir::func::FuncOp function : shardModule.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (program)
      return failCardProgramValue<mlir::func::FuncOp>(
          failureReason,
          "spatial output lowering produced more than one executable "
          "function");
    program = function;
  }
  if (!program)
    return failCardProgramValue<mlir::func::FuncOp>(
        failureReason,
        "spatial output lowering produced no executable function");
  program->remove();
  return program;
}

static mlir::FailureOr<mlir::func::FuncOp>
createNoWorkEntry(mlir::func::FuncOp sourceProgram,
                  std::string *failureReason) {
  // A no-work Tile needs the verified symbol/signature contract, not a copy
  // of the executable body that is immediately discarded. Preserve the op
  // shell and construct the only region state this artifact can contain.
  auto entry = mlir::cast<mlir::func::FuncOp>(
      sourceProgram->cloneWithoutRegions());
  mlir::Block *block = entry.addEntryBlock();
  mlir::OpBuilder builder(block, block->end());
  mlir::ValueRange outputs(block->getArguments());
  outputs = outputs.take_back(entry.getNumResults());
  builder.create<mlir::func::ReturnOp>(entry.getLoc(), outputs);
  if (mlir::failed(mlir::verify(entry))) {
    entry->destroy();
    return failCardProgramValue<mlir::func::FuncOp>(
        failureReason, "failed to create verifier-legal no-work entry");
  }
  return entry;
}

/// Removes the private scheduling destinations after TileRegion
/// materialization. Outputs remain full-card tensors: active Tiles write only
/// their selected subviews, while no-work Tiles perform no writes. Replacing
/// each internal destination by tensor.empty makes the eventual
/// compiler-owned allocation the single output root; target ABI preparation
/// later replaces that root with the user output pointer through its verified
/// result path. The source argument boundary is never inferred or changed.
static mlir::LogicalResult
removeSchedulingOutputDestinations(mlir::func::FuncOp entry,
                                   unsigned sourceArgumentCount,
                                   StructuredMaterializationRelations &relations,
                                   std::string *failureReason) {
  const unsigned resultCount = entry.getNumResults();
  if (resultCount == 0 ||
      entry.getNumArguments() != sourceArgumentCount + resultCount)
    return failCardProgram(
        failureReason,
        "physical Tile entry does not have the exact private scheduling "
        "boundary");
  const unsigned outputBase = sourceArgumentCount;
  mlir::OpBuilder builder(&entry.getBody().front(),
                          entry.getBody().front().begin());
  llvm::BitVector eraseArguments(entry.getNumArguments());
  for (unsigned index = 0; index < resultCount; ++index) {
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(entry.getResultTypes()[index]);
    mlir::BlockArgument destination = entry.getArgument(outputBase + index);
    if (!resultType || !resultType.hasStaticShape() ||
        destination.getType() != resultType)
      return failCardProgram(
          failureReason,
          "physical Tile scheduling destination is not a static tensor");
    auto empty = builder.create<mlir::tensor::EmptyOp>(
        entry.getLoc(), resultType.getShape(), resultType.getElementType());
    auto retarget = [&](auto &entries) {
      for (auto &relation : entries)
        if (relation.buffer == destination)
          relation.buffer = empty.getResult();
    };
    retarget(relations.operationResultBuffers);
    retarget(relations.operandBuffers);
    retarget(relations.outputBuffers);
    destination.replaceAllUsesWith(empty.getResult());
    eraseArguments.set(outputBase + index);
  }
  entry.eraseArguments(eraseArguments);
  if (mlir::failed(mlir::verify(entry)))
    return failCardProgram(
        failureReason,
        "physical Tile functional result boundary is not verifier-legal");
  return mlir::success();
}

static mlir::LogicalResult
verifyDirectSourceMembers(mlir::ModuleOp sourceModule,
                          mlir::func::FuncOp sourceProgram,
                          std::string *failureReason) {
  for (mlir::Operation &operation :
       sourceModule.getBody()->without_terminator()) {
    if (&operation == sourceProgram.getOperation() ||
        mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation) ||
        isCardSharedDeclaration(operation))
      continue;
    return failCardProgram(
        failureReason,
        (llvm::Twine("card spatial mapping does not support direct "
                     "module operation '") +
         operation.getName().getStringRef() + "'")
            .str());
  }
  return mlir::success();
}

static void cloneModuleFacts(mlir::ModuleOp sourceModule,
                             mlir::ModuleOp destinationModule) {
  mlir::OpBuilder builder(destinationModule.getBodyRegion());
  mlir::IRMapping mapping;
  for (TargetTopologyOp topology : sourceModule.getOps<TargetTopologyOp>())
    builder.clone(*topology.getOperation(), mapping);
  for (ExecutionMeshOp mesh : sourceModule.getOps<ExecutionMeshOp>())
    builder.clone(*mesh.getOperation(), mapping);
}

} // namespace

static mlir::LogicalResult lowerTensorProgramToCardProgramImpl(
    mlir::ModuleOp sourceModule, PhysicalCardId cardId,
    const CardSpatialMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule, std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations,
    std::optional<PhysicalTileId> materializeOnlyTileId) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule)
    return failCardProgram(failureReason, "source module is null");

  {
    mlir::ScopedDiagnosticHandler suppress(
        sourceModule.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    if (mlir::failed(mlir::verify(sourceModule)))
      return failCardProgram(failureReason,
                             "source module is not verifier-legal");
  }

  std::string topologyFailure;
  mlir::FailureOr<PhysicalTopology> topology =
      PhysicalTopology::create(sourceModule, &topologyFailure);
  if (mlir::failed(topology))
    return failCardProgram(failureReason, topologyFailure);
  std::optional<llvm::ArrayRef<PhysicalTileId>> availableTiles =
      topology->getAvailableTileIds(cardId);
  if (!availableTiles)
    return failCardProgram(failureReason,
                           "requested physical card_id is outside topology");
  if (availableTiles->empty())
    return failCardProgram(failureReason,
                           "requested physical card has no available Tiles");
  if (materializeOnlyTileId &&
      !llvm::is_contained(*availableTiles, *materializeOnlyTileId))
    return failCardProgram(failureReason,
                           "failure probe names an unavailable physical Tile");
  if (mlir::failed(verifyLogicalMesh(sourceModule, failureReason)))
    return mlir::failure();

  mlir::FailureOr<mlir::func::FuncOp> sourceProgram =
      getSourceTensorProgram(sourceModule, failureReason);
  if (mlir::failed(sourceProgram) ||
      mlir::failed(verifyDirectSourceMembers(sourceModule, *sourceProgram,
                                             failureReason)))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>
      outputDomains = getStaticOutputDomains(*sourceProgram, failureReason);
  if (mlir::failed(outputDomains))
    return mlir::failure();

  const unsigned sourceArgumentCount = sourceProgram->getNumArguments();
  mlir::IRMapping sourceToScheduling;
  mlir::OwningOpRef<mlir::ModuleOp> schedulingModule =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone(sourceToScheduling));
  auto schedulingProgram = sourceToScheduling.lookupOrNull(*sourceProgram);
  if (!schedulingProgram ||
      mlir::failed(appendSchedulingOutputDestinations(
          mlir::cast<mlir::func::FuncOp>(schedulingProgram), failureReason)))
    return mlir::failure();

  llvm::DenseSet<mlir::Operation *> nodeOperations;
  llvm::DenseSet<uint32_t> nodeIds;
  llvm::SmallVector<StructuredOperationNodeMapping, 16>
      schedulingOperationNodes;
  schedulingOperationNodes.reserve(operationNodes.size());
  for (const StructuredOperationNodeMapping &node : operationNodes) {
    if (!node.operation || !nodeOperations.insert(node.operation).second ||
        !nodeIds.insert(node.structuredNodeId).second)
      return failCardProgram(
          failureReason,
          "card structured operation-node mapping is null or duplicated");
    mlir::Operation *cloned = sourceToScheduling.lookupOrNull(node.operation);
    if (!cloned)
      return failCardProgram(
          failureReason,
          "card structured operation-node mapping is outside the tensor "
          "program");
    schedulingOperationNodes.push_back({cloned, node.structuredNodeId});
  }

  llvm::DenseSet<mlir::Operation *> temporalSources;
  llvm::SmallVector<StructuredOpTemporalTile, 16>
      schedulingOperationTemporalTiles;
  schedulingOperationTemporalTiles.reserve(
      mapping.operationTemporalTiles.size());
  for (const StructuredOpTemporalTile &tile : mapping.operationTemporalTiles) {
    if (!tile.operation || !temporalSources.insert(tile.operation).second)
      return failCardProgram(
          failureReason,
          "card structured temporal mapping is null or duplicated");
    mlir::Operation *cloned = sourceToScheduling.lookupOrNull(tile.operation);
    if (!cloned)
      return failCardProgram(
          failureReason,
          "card structured temporal mapping is outside the tensor program");
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(cloned);
    if (!tiling ||
        tile.iteratorTileSizes.size() != tiling.getLoopIteratorTypes().size() ||
        llvm::any_of(tile.iteratorTileSizes,
                     [](int64_t size) { return size <= 0; }))
      return failCardProgram(
          failureReason,
          "card structured temporal tile does not match its iteration "
          "domain");
    llvm::SmallVector<int64_t, 4> ranges;
    if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(cloned)) {
      ranges = linalg.getStaticLoopRanges();
    } else if (cloned->getNumResults() == 1) {
      auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
          cloned->getResult(0).getType());
      if (resultType && resultType.hasStaticShape() &&
          resultType.getRank() ==
              static_cast<int64_t>(tile.iteratorTileSizes.size()))
        ranges.assign(resultType.getShape().begin(),
                      resultType.getShape().end());
    }
    if (ranges.size() != tile.iteratorTileSizes.size() ||
        llvm::any_of(llvm::zip_equal(ranges, tile.iteratorTileSizes),
                     [](auto values) {
                       auto [range, tileSize] = values;
                       return mlir::ShapedType::isDynamic(range) ||
                              range <= 0 || tileSize > range;
                     }))
      return failCardProgram(
          failureReason,
          "card structured temporal tile is outside its static iteration "
          "domain");
    schedulingOperationTemporalTiles.push_back(
        StructuredOpTemporalTile{cloned, tile.iteratorTileSizes});
  }
  for (mlir::Operation &operation :
       (*sourceProgram).getBody().front().without_terminator()) {
    if (!mlir::isa<mlir::DestinationStyleOpInterface>(&operation) ||
        !mlir::isa<mlir::TilingInterface>(&operation))
      continue;
    if (!temporalSources.contains(&operation))
      return failCardProgram(
          failureReason,
          "card structured temporal mapping must cover every scheduled "
          "operation exactly once");
  }

  llvm::SmallVector<SpatialEdgeStrategy, 16> schedulingEdgeStrategies;
  schedulingEdgeStrategies.reserve(mapping.edgeStrategies.size());
  for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies) {
    mlir::Operation *producer =
        sourceToScheduling.lookupOrNull(strategy.producer);
    mlir::Operation *consumer =
        sourceToScheduling.lookupOrNull(strategy.consumer);
    if (!producer || !consumer)
      return failCardProgram(
          failureReason,
          "card edge strategy is outside the source tensor program");
    SpatialEdgeStrategy mapped = strategy;
    mapped.producer = producer;
    mapped.consumer = consumer;
    schedulingEdgeStrategies.push_back(std::move(mapped));
  }

  // Every selected structured producer/data-input dependency is an exact
  // direct SSA edge or a statically provable unary pure support chain. DPS
  // init operands are initialization state rather than data edges and remain
  // governed by the consumer's typed lowering.
  llvm::DenseSet<std::pair<mlir::Operation *, unsigned>> coveredDataInputs;
  for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies) {
    if (!strategy.producer || !strategy.consumer ||
        strategy.producerResult >= strategy.producer->getNumResults() ||
        strategy.consumerOperand >= strategy.consumer->getNumOperands() ||
        mlir::failed(deriveUnaryPureSupportChain(
            strategy.producer, strategy.producerResult, strategy.consumer,
            strategy.consumerOperand, failureReason)))
      return failCardProgram(
          failureReason,
          "card edge strategy must name an exact structured SSA dependency");
    auto dps =
        mlir::dyn_cast<mlir::DestinationStyleOpInterface>(strategy.consumer);
    bool isDataInput =
        dps &&
        llvm::any_of(dps.getDpsInputOperands(), [&](mlir::OpOperand *operand) {
          return operand->getOperandNumber() == strategy.consumerOperand;
        });
    if (!isDataInput)
      return failCardProgram(
          failureReason,
          "card edge strategy must target a structured data input");
    coveredDataInputs.insert({strategy.consumer, strategy.consumerOperand});
  }
  for (mlir::Operation &operation :
       (*sourceProgram).getBody().front().without_terminator()) {
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(&operation);
    if (!dps || !mlir::isa<mlir::TilingInterface>(&operation))
      continue;
    for (mlir::OpOperand *operand : dps.getDpsInputOperands()) {
      auto producerResult = mlir::dyn_cast<mlir::OpResult>(operand->get());
      mlir::Operation *producer =
          producerResult ? producerResult.getOwner() : nullptr;
      if (!producer || producer->getBlock() != operation.getBlock() ||
          !mlir::isa<mlir::TilingInterface>(producer) ||
          !mlir::isa<mlir::DestinationStyleOpInterface>(producer))
        continue;
      if (!coveredDataInputs.contains(
              {&operation, operand->getOperandNumber()}))
        return failCardProgram(
            failureReason,
            "card spatial mapping is missing a direct structured edge action");
    }
  }

  llvm::DenseSet<int64_t> availableTileValues;
  for (PhysicalTileId tileId : *availableTiles)
    availableTileValues.insert(tileId.getValue());
  for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies) {
    if (!availableTileValues.contains(strategy.destinationTile.getValue()) ||
        (strategy.action != SpatialEdgeAction::PeerFragments &&
         !availableTileValues.contains(strategy.sourceTile.getValue())))
      return failCardProgram(
          failureReason,
          "card edge strategy names an unavailable physical Tile");
    for (const SpatialEdgeFragment &fragment : strategy.fragments)
      if (!availableTileValues.contains(fragment.sourceTile.getValue()))
        return failCardProgram(
            failureReason,
            "card edge fragment names an unavailable physical Tile");
  }

  if (mapping.outputs.size() != outputDomains->size())
    return failCardProgram(
        failureReason,
        "card spatial mapping must cover every function result exactly once");
  llvm::SmallVector<const CardOutputSpatialMapping *, 4> outputMappings(
      outputDomains->size(), nullptr);
  for (const CardOutputSpatialMapping &output : mapping.outputs) {
    if (output.outputIndex >= outputDomains->size())
      return failCardProgram(
          failureReason,
          "card spatial mapping output index is outside function results");
    if (outputMappings[output.outputIndex])
      return failCardProgram(failureReason,
                             "card spatial mapping output index is duplicated");
    if (output.shardDimension >= (*outputDomains)[output.outputIndex].size())
      return failCardProgram(
          failureReason,
          "card spatial mapping shard dimension is outside output domain");
    if (output.activeTileIds.empty())
      return failCardProgram(failureReason,
                             "card spatial mapping output has no active Tiles");
    if (output.temporalTileSizes.size() !=
        (*outputDomains)[output.outputIndex].size())
      return failCardProgram(
          failureReason,
          "card spatial mapping temporal tile rank differs from output");
    for (auto [tileSize, extent] : llvm::zip_equal(
             output.temporalTileSizes, (*outputDomains)[output.outputIndex]))
      if (tileSize <= 0 || tileSize > extent)
        return failCardProgram(
            failureReason,
            "card spatial mapping temporal tile is outside output domain");

    llvm::DenseSet<int64_t> outputTiles;
    for (PhysicalTileId tileId : output.activeTileIds) {
      if (!availableTileValues.contains(tileId.getValue()))
        return failCardProgram(
            failureReason,
            "card spatial mapping names an unavailable physical Tile");
      if (!outputTiles.insert(tileId.getValue()).second)
        return failCardProgram(
            failureReason,
            "card spatial mapping output contains a duplicate physical Tile");
    }
    const int64_t extent =
        (*outputDomains)[output.outputIndex][output.shardDimension];
    if (output.activeTileIds.size() > static_cast<size_t>(extent))
      return failCardProgram(
          failureReason,
          "card spatial mapping output has more active Tiles than nonempty "
          "shards");
    outputMappings[output.outputIndex] = &output;
  }
  if (llvm::is_contained(outputMappings, nullptr))
    return failCardProgram(
        failureReason,
        "card spatial mapping must cover every function result exactly once");

  mlir::OwningOpRef<mlir::ModuleOp> result =
      mlir::ModuleOp::create(sourceModule.getLoc());
  result->getOperation()->setAttrs(sourceModule->getAttrDictionary());
  cloneModuleFacts(sourceModule, *result);

  mlir::OpBuilder moduleBuilder(result->getBodyRegion());
  auto card = moduleBuilder.create<CardProgramOp>(
      sourceModule.getLoc(),
      moduleBuilder.getI64IntegerAttr(cardId.getValue()));
  card.getBody().push_back(new mlir::Block());
  mlir::Block &cardBody = card.getBody().front();
  mlir::OpBuilder cardBuilder(&cardBody, cardBody.end());
  mlir::IRMapping declarationMapping;
  for (mlir::Operation &operation :
       sourceModule.getBody()->without_terminator()) {
    if (isCardSharedDeclaration(operation) &&
        !mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation))
      cardBuilder.clone(operation, declarationMapping);
  }

  llvm::SmallVector<int64_t, 4> coveredShardExtents(outputDomains->size(), 0);
  StructuredMaterializationRelations resultRelations;
  for (PhysicalTileId tileId : *availableTiles) {
    auto tile = cardBuilder.create<TileProgramOp>(
        sourceModule.getLoc(),
        cardBuilder.getI64IntegerAttr(tileId.getValue()));
    tile.getBody().push_back(new mlir::Block());
    mlir::Block &tileBody = tile.getBody().front();

    mlir::func::FuncOp entry;
    StructuredMaterializationRelations tileRelations;
    llvm::SmallVector<SpatialOutputShard, 4> tileShards;
    for (auto [outputIndex, output] : llvm::enumerate(outputMappings)) {
      auto active = llvm::find(output->activeTileIds, tileId);
      if (active == output->activeTileIds.end())
        continue;
      const size_t activeOrdinal = static_cast<size_t>(
          std::distance(output->activeTileIds.begin(), active));
      const size_t activeShardCount = output->activeTileIds.size();
      const llvm::SmallVector<int64_t, 4> &domain =
          (*outputDomains)[outputIndex];
      const int64_t shardExtent = domain[output->shardDimension];
      const int64_t baseShardSize =
          shardExtent / static_cast<int64_t>(activeShardCount);
      const int64_t largerShardCount =
          shardExtent % static_cast<int64_t>(activeShardCount);
      const int64_t size =
          baseShardSize +
          (static_cast<int64_t>(activeOrdinal) < largerShardCount);
      const int64_t offset =
          static_cast<int64_t>(activeOrdinal) * baseShardSize +
          std::min<int64_t>(static_cast<int64_t>(activeOrdinal),
                            largerShardCount);
      SpatialOutputShard shard;
      shard.outputIndex = static_cast<unsigned>(outputIndex);
      shard.offsets.assign(domain.size(), 0);
      shard.sizes = domain;
      shard.offsets[output->shardDimension] = offset;
      shard.sizes[output->shardDimension] = size;
      shard.temporalTileSizes.reserve(domain.size());
      for (auto [temporalSize, shardSize] :
           llvm::zip_equal(output->temporalTileSizes, shard.sizes))
        shard.temporalTileSizes.push_back(std::min(temporalSize, shardSize));
      coveredShardExtents[outputIndex] += size;
      tileShards.push_back(std::move(shard));
    }

    const bool materializeTile =
        !materializeOnlyTileId || tileId == *materializeOnlyTileId;
    const bool hasEdgeAction =
        materializeTile &&
        llvm::any_of(
            mapping.edgeStrategies, [&](const SpatialEdgeStrategy &strategy) {
              return isSpatialEdgeStrategyIncidentOnTile(strategy, tileId);
            });
    if (hasEdgeAction) {
      mlir::OwningOpRef<mlir::ModuleOp> loweredShard;
      if (mlir::failed(lowerSpatialEdgeStrategiesToTileRegionModule(
              *schedulingModule, sourceArgumentCount, tileShards, tileId,
              mapping.materializationMode, schedulingEdgeStrategies,
              loweredShard, failureReason,
              /*currentLogicalPartition=*/0, schedulingOperationTemporalTiles,
              schedulingOperationNodes, &tileRelations)))
        return mlir::failure();
      mlir::FailureOr<mlir::func::FuncOp> loweredEntry =
          takeLoweredTensorProgram(*loweredShard, failureReason);
      if (mlir::failed(loweredEntry))
        return mlir::failure();
      entry = *loweredEntry;
    } else if (materializeTile && !tileShards.empty()) {
      mlir::OwningOpRef<mlir::ModuleOp> loweredShard;
      if (mlir::failed(lowerSpatialOutputShardsToTileRegionModule(
              *schedulingModule, sourceArgumentCount, tileShards, loweredShard,
              failureReason,
              /*currentLogicalPartition=*/0, schedulingOperationTemporalTiles,
              schedulingOperationNodes, &tileRelations)))
        return mlir::failure();
      mlir::FailureOr<mlir::func::FuncOp> loweredEntry =
          takeLoweredTensorProgram(*loweredShard, failureReason);
      if (mlir::failed(loweredEntry))
        return mlir::failure();
      entry = *loweredEntry;
    } else {
      mlir::FailureOr<mlir::func::FuncOp> noWork = createNoWorkEntry(
          mlir::cast<mlir::func::FuncOp>(schedulingProgram), failureReason);
      if (mlir::failed(noWork))
        return mlir::failure();
      entry = *noWork;
    }
    tileBody.push_back(entry.getOperation());
    // Peer endpoint verification resolves physical identities through the
    // enclosing module topology, so transform and verify the entry only after
    // it has been attached to its final Card/Tile IR scope.
    if (mlir::failed(removeSchedulingOutputDestinations(
            entry, sourceArgumentCount, tileRelations, failureReason)))
      return mlir::failure();
    resultRelations.operationResultBuffers.append(
        tileRelations.operationResultBuffers.begin(),
        tileRelations.operationResultBuffers.end());
    resultRelations.operandBuffers.append(tileRelations.operandBuffers.begin(),
                                          tileRelations.operandBuffers.end());
    resultRelations.outputBuffers.append(tileRelations.outputBuffers.begin(),
                                         tileRelations.outputBuffers.end());
  }

  if (!relationsBelongTo(result->getOperation(), resultRelations))
    return failCardProgram(
        failureReason,
        "CardProgram materialization produced a buffer relation outside the "
        "current IR");

  for (auto [outputIndex, covered] : llvm::enumerate(coveredShardExtents)) {
    const CardOutputSpatialMapping *output = outputMappings[outputIndex];
    if (covered != (*outputDomains)[outputIndex][output->shardDimension])
      return failCardProgram(failureReason,
                             "card spatial shards do not cover output "
                             "domain exactly");
  }
  {
    mlir::ScopedDiagnosticHandler suppress(
        sourceModule.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    if (mlir::failed(mlir::verify(*result)))
      return failCardProgram(failureReason,
                             "materialized card program is not verifier-legal");
  }

  if (materializationRelations)
    *materializationRelations = std::move(resultRelations);
  cardModule = std::move(result);
  return mlir::success();
}

mlir::LogicalResult lowerTensorProgramToCardProgram(
    mlir::ModuleOp sourceModule, PhysicalCardId cardId,
    const CardSpatialMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule, std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations) {
  return lowerTensorProgramToCardProgramImpl(
      sourceModule, cardId, mapping, cardModule, failureReason,
      operationNodes, materializationRelations,
      /*materializeOnlyTileId=*/std::nullopt);
}

mlir::LogicalResult lowerTensorProgramToCardProgramFailureProbe(
    mlir::ModuleOp sourceModule, PhysicalCardId cardId,
    PhysicalTileId probeTileId, const CardSpatialMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &probeCardModule,
    std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations) {
  return lowerTensorProgramToCardProgramImpl(
      sourceModule, cardId, mapping, probeCardModule, failureReason,
      operationNodes, materializationRelations, probeTileId);
}

} // namespace wafer
