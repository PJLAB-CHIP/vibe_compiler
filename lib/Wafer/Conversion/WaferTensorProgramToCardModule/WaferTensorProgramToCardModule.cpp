//===- WaferTensorProgramToCardModule.cpp - Card spatial mapping ------===//

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/Target/TargetTopology.h"
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

static mlir::LogicalResult failCardModule(std::string *failureReason,
                                          llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

template <typename T>
static mlir::FailureOr<T> failCardModuleValue(std::string *failureReason,
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

static bool
relationsBelongTo(mlir::Operation *root,
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
      return failCardModuleValue<mlir::func::FuncOp>(
          failureReason,
          "card spatial mapping requires exactly one defined direct "
          "tensor-program function");
    program = function;
  }
  if (!program)
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason,
        "card spatial mapping requires exactly one defined direct "
        "tensor-program function");
  if (!program.getBody().hasOneBlock())
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason, "card spatial mapping requires a single-block tensor "
                       "program");
  if (program.getNumResults() == 0)
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason, "card spatial mapping requires tensor-program results");

  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      program.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != program.getNumResults())
    return failCardModuleValue<mlir::func::FuncOp>(
        failureReason,
        "card spatial mapping requires a complete functional return");
  for (unsigned index = 0; index < program.getNumResults(); ++index) {
    mlir::Type resultType = program.getResultTypes()[index];
    if (!mlir::isa<mlir::RankedTensorType>(resultType) ||
        returnOp.getOperand(index).getType() != resultType)
      return failCardModuleValue<mlir::func::FuncOp>(
          failureReason, "card spatial mapping requires ranked tensor results");
  }
  return program;
}

/// Builds the private scheduling form consumed by TileRegion materialization.
/// Source arguments remain untouched and keep their ordinary frontend ABI.
/// One compiler-owned destination is appended per result only inside this
/// private clone; it is removed again before the CardModule is exposed.
static mlir::LogicalResult
appendSchedulingOutputDestinations(mlir::func::FuncOp program,
                                   std::string *failureReason) {
  for (mlir::Type resultType : program.getResultTypes()) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType || !tensorType.hasStaticShape())
      return failCardModule(
          failureReason,
          "card scheduling destinations require static ranked results");
    program.insertArgument(program.getNumArguments(), resultType,
                           mlir::DictionaryAttr{}, program.getLoc());
  }
  if (mlir::failed(mlir::verify(program)))
    return failCardModule(
        failureReason,
        "private card scheduling boundary is not verifier-legal");
  return mlir::success();
}

static mlir::LogicalResult verifyLogicalMesh(mlir::ModuleOp module,
                                             std::string *failureReason) {
  llvm::SmallVector<ExecutionMeshOp, 2> meshes(
      module.getOps<ExecutionMeshOp>());
  if (meshes.size() != 1)
    return failCardModule(
        failureReason,
        "card spatial mapping requires exactly one direct logical "
        "execution mesh");

  int64_t partitionCount = 1;
  for (int64_t dimension : meshes.front().getShapeAttr().asArrayRef()) {
    if (dimension <= 0 ||
        partitionCount > std::numeric_limits<int64_t>::max() / dimension)
      return failCardModule(failureReason,
                            "logical execution mesh is not representable");
    partitionCount *= dimension;
  }
  if (partitionCount != 1)
    return failCardModule(
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
      return failCardModuleValue<
          llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>(
          failureReason, "card spatial mapping requires static tensor-program "
                         "output shapes");
    if (tensorType.getRank() == 0 ||
        llvm::any_of(tensorType.getShape(),
                     [](int64_t extent) { return extent <= 0; }))
      return failCardModuleValue<
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
      return failCardModuleValue<mlir::func::FuncOp>(
          failureReason,
          "spatial output lowering produced more than one executable "
          "function");
    program = function;
  }
  if (!program)
    return failCardModuleValue<mlir::func::FuncOp>(
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
  // shell and construct the only region state this module can contain.
  auto entry =
      mlir::cast<mlir::func::FuncOp>(sourceProgram->cloneWithoutRegions());
  mlir::Block *block = entry.addEntryBlock();
  mlir::OpBuilder builder(block, block->end());
  mlir::ValueRange outputs(block->getArguments());
  outputs = outputs.take_back(entry.getNumResults());
  builder.create<mlir::func::ReturnOp>(entry.getLoc(), outputs);
  if (mlir::failed(mlir::verify(entry))) {
    entry->destroy();
    return failCardModuleValue<mlir::func::FuncOp>(
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
static mlir::LogicalResult removeSchedulingOutputDestinations(
    mlir::func::FuncOp entry, unsigned sourceArgumentCount,
    StructuredMaterializationRelations &relations, std::string *failureReason) {
  const unsigned resultCount = entry.getNumResults();
  if (resultCount == 0 ||
      entry.getNumArguments() != sourceArgumentCount + resultCount)
    return failCardModule(
        failureReason, "Tile entry does not have the exact private scheduling "
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
      return failCardModule(
          failureReason, "Tile scheduling destination is not a static tensor");
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
  if (mlir::failed(mlir::verify(entry))) {
    llvm::errs() << "--- entry IR on boundary verify failure ---\n";
    entry->getParentOfType<mlir::ModuleOp>().print(llvm::errs());
    return failCardModule(
        failureReason, "Tile functional result boundary is not verifier-legal");
  }
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
    return failCardModule(
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

/// Validated, schedule-ready input shared by whole-card and single-Tile
/// materialization. Owns the private scheduling clone; the per-Tile lowering
/// below only consumes its validated pieces.
struct TileMaterializationPreparation {
  mlir::OwningOpRef<mlir::ModuleOp> schedulingModule;
  mlir::func::FuncOp schedulingProgram;
  unsigned sourceArgumentCount = 0;
  llvm::SmallVector<TileId, 16> availableTiles;
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> outputDomains;
  llvm::SmallVector<const OutputTileMapping *, 4> outputMappings;
  llvm::SmallVector<StructuredOpTemporalTile, 16>
      schedulingOperationTemporalTiles;
  llvm::SmallVector<StructuredOperationNodeMapping, 16>
      schedulingOperationNodes;
  llvm::SmallVector<SpatialEdgeStrategy, 16> schedulingEdgeStrategies;
};

static mlir::FailureOr<TileMaterializationPreparation>
prepareTileMaterialization(mlir::ModuleOp sourceModule, CardId cardId,
                           const TileMapping &mapping,
                           llvm::ArrayRef<StructuredOperationNodeMapping>
                               operationNodes,
                           std::string *failureReason) {
  TileMaterializationPreparation preparation;
  if (failureReason)
    failureReason->clear();
  if (!sourceModule)
    return failCardModuleValue<TileMaterializationPreparation>(
        failureReason, "source module is null");

  {
    mlir::ScopedDiagnosticHandler suppress(
        sourceModule.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    if (mlir::failed(mlir::verify(sourceModule)))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason, "source module is not verifier-legal");
  }

  std::string topologyFailure;
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(sourceModule, &topologyFailure);
  if (mlir::failed(topology))
    return failCardModuleValue<TileMaterializationPreparation>(
        failureReason, topologyFailure);
  std::optional<llvm::ArrayRef<TileId>> availableTiles =
      topology->getAvailableTileIds(cardId);
  if (!availableTiles)
    return failCardModuleValue<TileMaterializationPreparation>(
        failureReason, "requested card_id is outside target topology");
  if (availableTiles->empty())
    return failCardModuleValue<TileMaterializationPreparation>(
        failureReason, "requested card has no available Tiles");
  preparation.availableTiles.assign(availableTiles->begin(),
                                    availableTiles->end());
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
  preparation.outputDomains = std::move(*outputDomains);

  preparation.sourceArgumentCount = sourceProgram->getNumArguments();
  mlir::IRMapping sourceToScheduling;
  preparation.schedulingModule = mlir::cast<mlir::ModuleOp>(
      sourceModule->clone(sourceToScheduling));
  auto schedulingProgram =
      sourceToScheduling.lookupOrNull(*sourceProgram);
  if (!schedulingProgram ||
      mlir::failed(appendSchedulingOutputDestinations(
          mlir::cast<mlir::func::FuncOp>(schedulingProgram), failureReason)))
    return mlir::failure();
  preparation.schedulingProgram =
      mlir::cast<mlir::func::FuncOp>(schedulingProgram);

  llvm::DenseSet<mlir::Operation *> nodeOperations;
  llvm::DenseSet<uint32_t> nodeIds;
  preparation.schedulingOperationNodes.reserve(operationNodes.size());
  for (const StructuredOperationNodeMapping &node : operationNodes) {
    if (!node.operation || !nodeOperations.insert(node.operation).second ||
        !nodeIds.insert(node.structuredNodeId).second)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured operation-node mapping is null or duplicated");
    mlir::Operation *cloned = sourceToScheduling.lookupOrNull(node.operation);
    if (!cloned)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured operation-node mapping is outside the tensor "
          "program");
    preparation.schedulingOperationNodes.push_back({cloned, node.structuredNodeId});
  }

  llvm::DenseSet<mlir::Operation *> temporalSources;
  preparation.schedulingOperationTemporalTiles.reserve(
      mapping.operationTemporalTiles.size());
  for (const StructuredOpTemporalTile &tile : mapping.operationTemporalTiles) {
    if (!tile.operation || !temporalSources.insert(tile.operation).second)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal mapping is null or duplicated");
    mlir::Operation *cloned = sourceToScheduling.lookupOrNull(tile.operation);
    if (!cloned)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal mapping is outside the tensor program");
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(cloned);
    if (!tiling ||
        tile.iteratorTileSizes.size() != tiling.getLoopIteratorTypes().size() ||
        llvm::any_of(tile.iteratorTileSizes,
                     [](int64_t size) { return size <= 0; }))
      return failCardModuleValue<TileMaterializationPreparation>(
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
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal tile is outside its static iteration "
          "domain");
    preparation.schedulingOperationTemporalTiles.push_back(
        StructuredOpTemporalTile{cloned, tile.iteratorTileSizes});
  }
  for (mlir::Operation &operation :
       (*sourceProgram).getBody().front().without_terminator()) {
    if (!mlir::isa<mlir::DestinationStyleOpInterface>(&operation) ||
        !mlir::isa<mlir::TilingInterface>(&operation))
      continue;
    if (!temporalSources.contains(&operation))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card structured temporal mapping must cover every scheduled "
          "operation exactly once");
  }

  preparation.schedulingEdgeStrategies.reserve(mapping.edgeStrategies.size());
  for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies) {
    mlir::Operation *producer =
        sourceToScheduling.lookupOrNull(strategy.producer);
    mlir::Operation *consumer =
        sourceToScheduling.lookupOrNull(strategy.consumer);
    if (!producer || !consumer)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card edge strategy is outside the source tensor program");
    SpatialEdgeStrategy mapped = strategy;
    mapped.producer = producer;
    mapped.consumer = consumer;
    preparation.schedulingEdgeStrategies.push_back(std::move(mapped));
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
      return failCardModuleValue<TileMaterializationPreparation>(
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
      return failCardModuleValue<TileMaterializationPreparation>(
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
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card spatial mapping is missing a direct structured edge action");
    }
  }

  llvm::DenseSet<int64_t> availableTileValues;
  for (TileId tileId : preparation.availableTiles)
    availableTileValues.insert(tileId.getValue());
  for (const SpatialEdgeStrategy &strategy : mapping.edgeStrategies) {
    if (!availableTileValues.contains(strategy.destinationTile.getValue()) ||
        (strategy.action != SpatialEdgeAction::PeerFragments &&
         !availableTileValues.contains(strategy.sourceTile.getValue())))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason, "card edge strategy names an unavailable Tile");
    for (const SpatialEdgeFragment &fragment : strategy.fragments)
      if (!availableTileValues.contains(fragment.sourceTile.getValue()))
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card edge fragment names an unavailable Tile");
  }

  if (mapping.outputs.size() != preparation.outputDomains.size())
    return failCardModuleValue<TileMaterializationPreparation>(
        failureReason,
        "card spatial mapping must cover every function result exactly once");
  preparation.outputMappings.assign(preparation.outputDomains.size(), nullptr);
  for (const OutputTileMapping &output : mapping.outputs) {
    if (output.outputIndex >= preparation.outputDomains.size())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping output index is outside function results");
    if (preparation.outputMappings[output.outputIndex])
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping output index is duplicated");
    if (output.shardDimension >=
        preparation.outputDomains[output.outputIndex].size())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping shard dimension is outside output domain");
    if (output.activeTileIds.empty())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping output has no active Tiles");
    if (output.temporalTileSizes.size() !=
        preparation.outputDomains[output.outputIndex].size())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping temporal tile rank differs from output");
    for (auto [tileSize, extent] : llvm::zip_equal(
             output.temporalTileSizes,
             preparation.outputDomains[output.outputIndex]))
      if (tileSize <= 0 || tileSize > extent)
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card spatial mapping temporal tile is outside output domain");

    llvm::DenseSet<int64_t> outputTiles;
    for (TileId tileId : output.activeTileIds) {
      if (!availableTileValues.contains(tileId.getValue()))
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card spatial mapping names an unavailable Tile");
      if (!outputTiles.insert(tileId.getValue()).second)
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card spatial mapping output contains a duplicate Tile");
    }
    const int64_t extent =
        preparation.outputDomains[output.outputIndex][output.shardDimension];
    if (output.activeTileIds.size() > static_cast<size_t>(extent))
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card spatial mapping output has more active Tiles than nonempty "
          "shards");
    preparation.outputMappings[output.outputIndex] = &output;
  }
  if (llvm::is_contained(preparation.outputMappings, nullptr))
    return failCardModuleValue<TileMaterializationPreparation>(
        failureReason,
        "card spatial mapping must cover every function result exactly once");
  return preparation;
}

/// Computes one Tile's balanced nonempty output shards from the validated
/// mapping and adds their shard extents to the coverage ledger.
static void collectTileOutputShards(
    const TileMaterializationPreparation &preparation, TileId tileId,
    llvm::SmallVectorImpl<SpatialOutputShard> &tileShards,
    llvm::SmallVectorImpl<int64_t> &coveredShardExtents) {
  for (auto [outputIndex, output] :
       llvm::enumerate(preparation.outputMappings)) {
    auto active = llvm::find(output->activeTileIds, tileId);
    if (active == output->activeTileIds.end())
      continue;
    const size_t activeOrdinal = static_cast<size_t>(
        std::distance(output->activeTileIds.begin(), active));
    const size_t activeShardCount = output->activeTileIds.size();
    const llvm::SmallVector<int64_t, 4> &domain =
        preparation.outputDomains[outputIndex];
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
}

/// Lowers one Tile entry from the prepared scheduling clone. The returned
/// function is detached from any module; the caller attaches it to its final
/// IR scope before `removeSchedulingOutputDestinations` runs, because peer
/// endpoint verification resolves physical identities through the enclosing
/// module topology.
static mlir::FailureOr<mlir::func::FuncOp> lowerTileEntry(
    const TileMaterializationPreparation &preparation, CardId cardId,
    TileId tileId, const TileMapping &mapping,
    llvm::ArrayRef<SpatialOutputShard> tileShards, bool materializeTile,
    std::string *failureReason,
    StructuredMaterializationRelations *tileRelations) {
  mlir::func::FuncOp entry;
  const bool hasEdgeAction =
      materializeTile &&
      llvm::any_of(
          mapping.edgeStrategies, [&](const SpatialEdgeStrategy &strategy) {
            return isSpatialEdgeStrategyIncidentOnTile(strategy, tileId);
          });
  if (hasEdgeAction) {
    mlir::OwningOpRef<mlir::ModuleOp> loweredShard;
    if (mlir::failed(lowerSpatialEdgeStrategiesToTileRegionModule(
            *preparation.schedulingModule, preparation.sourceArgumentCount,
            tileShards, tileId, mapping.materializationMode,
            preparation.schedulingEdgeStrategies, loweredShard, failureReason,
            /*currentLogicalPartition=*/0,
            preparation.schedulingOperationTemporalTiles,
            preparation.schedulingOperationNodes, tileRelations)))
      return mlir::failure();
    mlir::FailureOr<mlir::func::FuncOp> loweredEntry =
        takeLoweredTensorProgram(*loweredShard, failureReason);
    if (mlir::failed(loweredEntry))
      return mlir::failure();
    entry = *loweredEntry;
  } else if (materializeTile && !tileShards.empty()) {
    mlir::OwningOpRef<mlir::ModuleOp> loweredShard;
    if (mlir::failed(lowerSpatialOutputShardsToTileRegionModule(
            *preparation.schedulingModule, preparation.sourceArgumentCount,
            tileShards, loweredShard, failureReason,
            /*currentLogicalPartition=*/0,
            preparation.schedulingOperationTemporalTiles,
            preparation.schedulingOperationNodes, tileRelations)))
      return mlir::failure();
    mlir::FailureOr<mlir::func::FuncOp> loweredEntry =
        takeLoweredTensorProgram(*loweredShard, failureReason);
    if (mlir::failed(loweredEntry))
      return mlir::failure();
    entry = *loweredEntry;
  } else {
    mlir::FailureOr<mlir::func::FuncOp> noWork = createNoWorkEntry(
        preparation.schedulingProgram, failureReason);
    if (mlir::failed(noWork))
      return mlir::failure();
    entry = *noWork;
  }
  // Deterministic baseline contract: every TileRegion carries exactly one
  // structured compute root; independent roots on one Tile form multiple
  // sequential regions. Search-owned region grouping is never rewritten.
  if (mapping.materializationMode ==
          SpatialDataflowMaterializationMode::IndependentDDRStages &&
      mlir::failed(splitStructuredRootBoundaries(entry, *tileRelations,
                                                failureReason)))
    return mlir::failure();
  return entry;
}

mlir::LogicalResult lowerTensorProgramToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule, std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations) {
  mlir::FailureOr<TileMaterializationPreparation> preparation =
      prepareTileMaterialization(sourceModule, cardId, mapping, operationNodes,
                                 failureReason);
  if (mlir::failed(preparation))
    return mlir::failure();

  mlir::OwningOpRef<mlir::ModuleOp> result =
      mlir::ModuleOp::create(sourceModule.getLoc());
  result->getOperation()->setAttrs(sourceModule->getAttrDictionary());
  cloneModuleFacts(sourceModule, *result);

  mlir::OpBuilder moduleBuilder(result->getBodyRegion());
  auto card = moduleBuilder.create<CardModuleOp>(
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

  llvm::SmallVector<int64_t, 4> coveredShardExtents(
      preparation->outputDomains.size(), 0);
  StructuredMaterializationRelations resultRelations;
  for (TileId tileId : preparation->availableTiles) {
    auto tile = cardBuilder.create<TileModuleOp>(
        sourceModule.getLoc(),
        cardBuilder.getI64IntegerAttr(tileId.getValue()));
    tile.getBody().push_back(new mlir::Block());
    mlir::Block &tileBody = tile.getBody().front();

    StructuredMaterializationRelations tileRelations;
    llvm::SmallVector<SpatialOutputShard, 4> tileShards;
    collectTileOutputShards(*preparation, tileId, tileShards,
                            coveredShardExtents);
    mlir::FailureOr<mlir::func::FuncOp> entry =
        lowerTileEntry(*preparation, cardId, tileId, mapping, tileShards,
                       /*materializeTile=*/true, failureReason,
                       &tileRelations);
    if (mlir::failed(entry))
      return mlir::failure();
    tileBody.push_back(entry->getOperation());
    // Peer endpoint verification resolves physical identities through the
    // enclosing module topology, so transform and verify the entry only after
    // it has been attached to its final Card/Tile IR scope.
    if (mlir::failed(removeSchedulingOutputDestinations(
            *entry, preparation->sourceArgumentCount, tileRelations,
            failureReason)))
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
    return failCardModule(
        failureReason,
        "CardModule materialization produced a buffer relation outside the "
        "current IR");

  for (auto [outputIndex, covered] : llvm::enumerate(coveredShardExtents)) {
    const OutputTileMapping *output = preparation->outputMappings[outputIndex];
    if (covered !=
        preparation->outputDomains[outputIndex][output->shardDimension])
      return failCardModule(failureReason,
                            "card spatial shards do not cover output "
                            "domain exactly");
  }
  {
    mlir::ScopedDiagnosticHandler suppress(
        sourceModule.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    if (mlir::failed(mlir::verify(*result)))
      return failCardModule(failureReason,
                            "materialized CardModule is not verifier-legal");
  }

  if (materializationRelations)
    *materializationRelations = std::move(resultRelations);
  cardModule = std::move(result);
  return mlir::success();
}

mlir::LogicalResult lowerTensorProgramToTileModule(
    mlir::ModuleOp sourceModule, CardId cardId, TileId tileId,
    const TileMapping &mapping, mlir::OwningOpRef<mlir::ModuleOp> &tileModule,
    std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations) {
  mlir::FailureOr<TileMaterializationPreparation> preparation =
      prepareTileMaterialization(sourceModule, cardId, mapping, operationNodes,
                                 failureReason);
  if (mlir::failed(preparation))
    return mlir::failure();
  if (!llvm::is_contained(preparation->availableTiles, tileId))
    return failCardModule(failureReason, "requested Tile is unavailable");

  StructuredMaterializationRelations tileRelations;
  llvm::SmallVector<SpatialOutputShard, 4> tileShards;
  llvm::SmallVector<int64_t, 4> coveredShardExtents(
      preparation->outputDomains.size(), 0);
  collectTileOutputShards(*preparation, tileId, tileShards,
                          coveredShardExtents);
  mlir::FailureOr<mlir::func::FuncOp> entry =
      lowerTileEntry(*preparation, cardId, tileId, mapping, tileShards,
                     /*materializeTile=*/true, failureReason, &tileRelations);
  if (mlir::failed(entry))
    return mlir::failure();

  // The probe scope is the Tile entry function itself: no CardModule shell,
  // no sibling Tile modules and no no-work wrappers are materialized.
  mlir::OwningOpRef<mlir::ModuleOp> result =
      mlir::ModuleOp::create(sourceModule.getLoc());
  result->getOperation()->setAttrs(sourceModule->getAttrDictionary());
  cloneModuleFacts(sourceModule, *result);
  result->getBody()->push_back(entry->getOperation());
  if (mlir::failed(removeSchedulingOutputDestinations(
          *entry, preparation->sourceArgumentCount, tileRelations,
          failureReason)))
    return mlir::failure();
  if (!relationsBelongTo(result->getOperation(), tileRelations))
    return failCardModule(
        failureReason,
        "single-Tile materialization produced a buffer relation outside the "
        "current IR");
  {
    mlir::ScopedDiagnosticHandler suppress(
        sourceModule.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    if (mlir::failed(mlir::verify(*result)))
      return failCardModule(failureReason,
                            "materialized single-Tile module is not "
                            "verifier-legal");
  }

  if (materializationRelations)
    *materializationRelations = std::move(tileRelations);
  tileModule = std::move(result);
  return mlir::success();
}

} // namespace wafer
