//===- WaferTensorProgramToCardModule.cpp - Card spatial mapping ------===//

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/Target/TargetTopology.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/BoundedParallel.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
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
    if (llvm::any_of(tensorType.getShape(),
                     [](int64_t extent) { return extent <= 0; }))
      return failCardModuleValue<
          llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4>>(
          failureReason,
          "card spatial mapping requires nonempty static output domains");
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
    StructuredMaterializationRelations &relations, std::string *failureReason,
    uint64_t boundaryArgumentCount = 0) {
  const unsigned resultCount = entry.getNumResults();
  if (resultCount == 0 ||
      entry.getNumArguments() !=
          sourceArgumentCount + resultCount + boundaryArgumentCount)
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
  llvm::SmallVector<SpatialEdgeMaterializationFacts, 16>
      schedulingEdgeFacts;
};

/// Minimal immutable source for one final baseline region.  It preserves the
/// full function ABI and module facts, but its body contains only the exact
/// SSA closure needed by this region's output shards and selected edge
/// endpoints.  The lowerer still owns one atomic clone of this narrow scope;
/// it never clones the complete TensorProgram once per structured root.
struct BaselineRegionSource {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp program;
  llvm::SmallVector<StructuredOpTemporalTile, 8> operationTemporalTiles;
  llvm::SmallVector<StructuredOperationNodeMapping, 8> operationNodes;
  llvm::SmallVector<SpatialEdgeStrategy, 8> edgeStrategies;
  llvm::SmallVector<SpatialEdgeMaterializationFacts, 8> edgeFacts;
  uint64_t operationCount = 0;
  bool analyzedClosure = false;
};

struct BaselineRegionClosureKey {
  llvm::SmallVector<unsigned, 4> outputs;
  llvm::SmallVector<std::tuple<mlir::Operation *, unsigned, mlir::Operation *,
                               unsigned>,
                    4>
      edges;

  bool operator==(const BaselineRegionClosureKey &other) const {
    return outputs == other.outputs && edges == other.edges;
  }
};

struct BaselineRegionClosureCache {
  struct Entry {
    BaselineRegionClosureKey key;
    llvm::SmallVector<mlir::Operation *, 8> operations;
  };

  std::mutex mutex;
  llvm::SmallVector<Entry, 8> entries;
};

struct TileEntryMaterializationStatistics {
  uint64_t baselineRegionClosureAnalyses = 0;
  uint64_t baselineRegionClosureAnalysisOperations = 0;
};

struct TileMaterializationSourcePreparation {
  mlir::ModuleOp sourceModule;
  mlir::func::FuncOp sourceProgram;
  CardId cardId{0};
  unsigned sourceArgumentCount = 0;
  llvm::SmallVector<TileId, 16> availableTiles;
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> outputDomains;
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
};

static mlir::FailureOr<TileMaterializationSourcePreparation>
prepareTileMaterializationSource(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    std::string *failureReason) {
  TileMaterializationSourcePreparation preparation;
  if (failureReason)
    failureReason->clear();
  if (!sourceModule)
    return failCardModuleValue<TileMaterializationSourcePreparation>(
        failureReason, "source module is null");

  {
    mlir::ScopedDiagnosticHandler suppress(
        sourceModule.getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    if (mlir::failed(mlir::verify(sourceModule)))
      return failCardModuleValue<TileMaterializationSourcePreparation>(
          failureReason, "source module is not verifier-legal");
  }

  std::string topologyFailure;
  mlir::FailureOr<TargetTopology> topology =
      TargetTopology::create(sourceModule, &topologyFailure);
  if (mlir::failed(topology))
    return failCardModuleValue<TileMaterializationSourcePreparation>(
        failureReason, topologyFailure);
  std::optional<llvm::ArrayRef<TileId>> availableTiles =
      topology->getAvailableTileIds(cardId);
  if (!availableTiles)
    return failCardModuleValue<TileMaterializationSourcePreparation>(
        failureReason, "requested card_id is outside target topology");
  if (availableTiles->empty())
    return failCardModuleValue<TileMaterializationSourcePreparation>(
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
  preparation.sourceModule = sourceModule;
  preparation.sourceProgram = *sourceProgram;
  preparation.cardId = cardId;
  preparation.sourceArgumentCount = sourceProgram->getNumArguments();

  llvm::DenseSet<mlir::Operation *> nodeOperations;
  llvm::DenseSet<uint32_t> nodeIds;
  preparation.operationNodes.reserve(operationNodes.size());
  for (const StructuredOperationNodeMapping &node : operationNodes) {
    if (!node.operation || !nodeOperations.insert(node.operation).second ||
        !nodeIds.insert(node.structuredNodeId).second ||
        node.operation->getParentOfType<mlir::ModuleOp>() != sourceModule)
      return failCardModuleValue<TileMaterializationSourcePreparation>(
          failureReason,
          "card structured operation-node mapping is null, duplicated or "
          "outside the tensor program");
    preparation.operationNodes.push_back(node);
  }
  return preparation;
}

static mlir::FailureOr<TileMaterializationPreparation>
prepareTileMaterialization(
    const TileMaterializationSourcePreparation &source,
    const TileMapping &mapping, std::string *failureReason) {
  TileMaterializationPreparation preparation;
  mlir::func::FuncOp sourceProgram = source.sourceProgram;
  preparation.sourceArgumentCount = source.sourceArgumentCount;
  preparation.availableTiles = source.availableTiles;
  preparation.outputDomains = source.outputDomains;

  mlir::IRMapping sourceToScheduling;
  preparation.schedulingModule = mlir::cast<mlir::ModuleOp>(
      source.sourceModule->clone(sourceToScheduling));
  auto schedulingProgram =
      sourceToScheduling.lookupOrNull(sourceProgram);
  if (!schedulingProgram ||
      mlir::failed(appendSchedulingOutputDestinations(
          mlir::cast<mlir::func::FuncOp>(schedulingProgram), failureReason)))
    return mlir::failure();
  preparation.schedulingProgram =
      mlir::cast<mlir::func::FuncOp>(schedulingProgram);

  preparation.schedulingOperationNodes.reserve(source.operationNodes.size());
  for (const StructuredOperationNodeMapping &node : source.operationNodes) {
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
       sourceProgram.getBody().front().without_terminator()) {
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
  llvm::DenseMap<mlir::Operation *, uint64_t> schedulingOrdinals;
  uint64_t schedulingOrdinal = 0;
  for (mlir::Operation &operation :
       preparation.schedulingProgram.getBody().front().without_terminator())
    schedulingOrdinals.try_emplace(&operation, schedulingOrdinal++);
  preparation.schedulingEdgeFacts.reserve(mapping.edgeStrategies.size());
  for (auto [edgeIndex, strategy] :
       llvm::enumerate(mapping.edgeStrategies)) {
    SpatialEdgeStrategy &mapped =
        preparation.schedulingEdgeStrategies[edgeIndex];
    if (!strategy.producer || !strategy.consumer ||
        strategy.producerResult >= strategy.producer->getNumResults() ||
        strategy.consumerOperand >= strategy.consumer->getNumOperands())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card edge strategy must name an exact structured SSA dependency");
    mlir::FailureOr<llvm::SmallVector<mlir::Operation *, 4>> supportChain =
        deriveUnaryPureSupportChain(mapped.producer, mapped.producerResult,
                                    mapped.consumer, mapped.consumerOperand,
                                    failureReason);
    auto ordinal = schedulingOrdinals.find(mapped.consumer);
    if (mlir::failed(supportChain) || ordinal == schedulingOrdinals.end())
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "card edge strategy must name an exact structured SSA dependency");
    if (mapped.consumerOffsets.empty() || mapped.consumerSizes.empty()) {
      if (!mapped.consumerOffsets.empty() || !mapped.consumerSizes.empty() ||
          mlir::failed(deriveSpatialEdgeConsumerResultDomain(
              mapped, mapped.consumerOffsets, mapped.consumerSizes,
              failureReason)))
        return failCardModuleValue<TileMaterializationPreparation>(
            failureReason,
            "card edge strategy has no exact consumer result domain");
    }
    preparation.schedulingEdgeFacts.push_back(
        {/*hasSupportPath=*/!supportChain->empty(),
         /*consumerScheduleOrdinal=*/ordinal->second});
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
       sourceProgram.getBody().front().without_terminator()) {
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
    if (output.shardDimension &&
        *output.shardDimension >=
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
    if (!output.shardDimension && output.activeTileIds.size() != 1)
      return failCardModuleValue<TileMaterializationPreparation>(
          failureReason,
          "unpartitioned card output requires exactly one active Tile");
    const int64_t extent = output.shardDimension
                               ? preparation.outputDomains[output.outputIndex]
                                     [*output.shardDimension]
                               : 1;
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
    if (!output->shardDimension) {
      SpatialOutputShard shard;
      shard.outputIndex = static_cast<unsigned>(outputIndex);
      shard.offsets.assign(domain.size(), 0);
      shard.sizes = domain;
      shard.temporalTileSizes = output->temporalTileSizes;
      ++coveredShardExtents[outputIndex];
      tileShards.push_back(std::move(shard));
      continue;
    }
    const unsigned shardDimension = *output->shardDimension;
    const int64_t shardExtent = domain[shardDimension];
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
    shard.offsets[shardDimension] = offset;
    shard.sizes[shardDimension] = size;
    shard.temporalTileSizes.reserve(domain.size());
    for (auto [temporalSize, shardSize] :
         llvm::zip_equal(output->temporalTileSizes, shard.sizes))
      shard.temporalTileSizes.push_back(std::min(temporalSize, shardSize));
    coveredShardExtents[outputIndex] += size;
    tileShards.push_back(std::move(shard));
  }
}

static mlir::FailureOr<BaselineRegionSource> buildBaselineRegionSource(
    const TileMaterializationPreparation &preparation,
    llvm::ArrayRef<SpatialOutputShard> outputShards,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    BaselineRegionClosureCache &closureCache, std::string *failureReason) {
  BaselineRegionSource result;
  mlir::func::FuncOp sourceProgram = preparation.schedulingProgram;
  mlir::Block &sourceBody = sourceProgram.getBody().front();
  auto sourceReturn =
      mlir::dyn_cast<mlir::func::ReturnOp>(sourceBody.getTerminator());
  if (!sourceReturn)
    return failCardModuleValue<BaselineRegionSource>(
        failureReason, "baseline region source has no functional return");

  BaselineRegionClosureKey closureKey;
  for (const SpatialOutputShard &shard : outputShards) {
    if (shard.outputIndex >= sourceReturn.getNumOperands())
      return failCardModuleValue<BaselineRegionSource>(
          failureReason,
          "baseline region output is outside the functional result domain");
    closureKey.outputs.push_back(shard.outputIndex);
  }
  llvm::sort(closureKey.outputs);
  closureKey.outputs.erase(
      std::unique(closureKey.outputs.begin(), closureKey.outputs.end()),
      closureKey.outputs.end());
  for (const SpatialEdgeStrategy &strategy : edgeStrategies)
    closureKey.edges.push_back(
        {strategy.producer, strategy.producerResult, strategy.consumer,
         strategy.consumerOperand});

  llvm::SmallVector<mlir::Operation *, 8> closureOperations;
  {
    std::lock_guard<std::mutex> lock(closureCache.mutex);
    auto known = llvm::find_if(closureCache.entries, [&](const auto &entry) {
      return entry.key == closureKey;
    });
    if (known != closureCache.entries.end()) {
      closureOperations = known->operations;
    } else {
      llvm::DenseSet<mlir::Operation *> needed;
      std::function<mlir::LogicalResult(mlir::Operation *)> collectClosure =
          [&](mlir::Operation *operation) -> mlir::LogicalResult {
        if (!operation || operation->getBlock() != &sourceBody)
          return failCardModule(
              failureReason,
              "baseline region endpoint is outside the scheduling body");
        if (!needed.insert(operation).second)
          return mlir::success();
        for (mlir::Value operand : operation->getOperands()) {
          mlir::Operation *definition = operand.getDefiningOp();
          if (definition && definition->getBlock() == &sourceBody &&
              mlir::failed(collectClosure(definition)))
            return mlir::failure();
        }
        return mlir::success();
      };
      for (unsigned output : closureKey.outputs)
        if (mlir::Operation *definition =
                sourceReturn.getOperand(output).getDefiningOp())
          if (mlir::failed(collectClosure(definition)))
            return mlir::failure();
      for (const SpatialEdgeStrategy &strategy : edgeStrategies)
        if (mlir::failed(collectClosure(strategy.producer)) ||
            mlir::failed(collectClosure(strategy.consumer)))
          return mlir::failure();
      if (needed.empty())
        return failCardModuleValue<BaselineRegionSource>(
            failureReason,
            "baseline region has no materializable SSA closure");

      // Dropping an independent pure branch is valid. An effectful operation
      // cannot be assigned to a root from SSA alone.
      for (mlir::Operation &operation : sourceBody.without_terminator()) {
        if (!needed.contains(&operation) &&
            !mlir::isMemoryEffectFree(&operation))
          return failCardModuleValue<BaselineRegionSource>(
              failureReason,
              "baseline region cannot isolate an effectful sibling operation");
        if (needed.contains(&operation))
          closureOperations.push_back(&operation);
      }
      closureCache.entries.push_back({closureKey, closureOperations});
      result.analyzedClosure = true;
    }
  }
  result.operationCount = closureOperations.size();
  llvm::DenseSet<mlir::Operation *> needed(closureOperations.begin(),
                                           closureOperations.end());
  llvm::BitVector selectedOutputs(sourceProgram.getNumResults());
  for (unsigned output : closureKey.outputs)
    selectedOutputs.set(output);

  mlir::ModuleOp schedulingModule = preparation.schedulingModule.get();
  result.module = mlir::ModuleOp::create(sourceProgram.getLoc());
  result.module->getOperation()->setAttrs(
      schedulingModule->getAttrDictionary());
  cloneModuleFacts(schedulingModule, *result.module);
  mlir::OpBuilder moduleBuilder(result.module->getBodyRegion());
  mlir::IRMapping declarationMapping;
  for (mlir::Operation &operation :
       schedulingModule.getBody()->without_terminator())
    if (isCardSharedDeclaration(operation) &&
        !mlir::isa<TargetTopologyOp, ExecutionMeshOp>(operation))
      moduleBuilder.clone(operation, declarationMapping);

  result.program = mlir::cast<mlir::func::FuncOp>(
      sourceProgram->cloneWithoutRegions());
  mlir::Block *targetBody = result.program.addEntryBlock();
  mlir::IRMapping mapping;
  for (auto [sourceArgument, targetArgument] :
       llvm::zip_equal(sourceProgram.getArguments(),
                       result.program.getArguments()))
    mapping.map(sourceArgument, targetArgument);

  mlir::OpBuilder bodyBuilder(targetBody, targetBody->end());
  for (mlir::Operation &operation : sourceBody.without_terminator())
    if (needed.contains(&operation))
      bodyBuilder.clone(operation, mapping);

  llvm::SmallVector<mlir::Value, 4> returned;
  returned.reserve(sourceProgram.getNumResults());
  for (unsigned output = 0; output < sourceProgram.getNumResults(); ++output) {
    if (!selectedOutputs.test(output)) {
      returned.push_back(result.program.getArgument(
          preparation.sourceArgumentCount + output));
      continue;
    }
    mlir::Value value =
        mapping.lookupOrNull(sourceReturn.getOperand(output));
    if (!value)
      return failCardModuleValue<BaselineRegionSource>(
          failureReason,
          "baseline region did not clone its selected output value");
    returned.push_back(value);
  }
  bodyBuilder.create<mlir::func::ReturnOp>(sourceReturn.getLoc(), returned);
  result.module->getBody()->push_back(result.program.getOperation());

  for (const StructuredOpTemporalTile &tile :
       preparation.schedulingOperationTemporalTiles) {
    mlir::Operation *operation = mapping.lookupOrNull(tile.operation);
    if (operation)
      result.operationTemporalTiles.push_back(
          StructuredOpTemporalTile{operation, tile.iteratorTileSizes});
  }
  for (const StructuredOperationNodeMapping &node :
       preparation.schedulingOperationNodes) {
    mlir::Operation *operation = mapping.lookupOrNull(node.operation);
    if (operation)
      result.operationNodes.push_back(
          {operation, node.structuredNodeId});
  }
  for (const SpatialEdgeStrategy &strategy : edgeStrategies) {
    auto original = llvm::find_if(
        preparation.schedulingEdgeStrategies,
        [&](const SpatialEdgeStrategy &candidate) {
          return candidate.producer == strategy.producer &&
                 candidate.producerResult == strategy.producerResult &&
                 candidate.consumer == strategy.consumer &&
                 candidate.consumerOperand == strategy.consumerOperand &&
                 candidate.sourceTile == strategy.sourceTile &&
                 candidate.destinationTile == strategy.destinationTile &&
                 candidate.action == strategy.action;
        });
    if (original == preparation.schedulingEdgeStrategies.end())
      return failCardModuleValue<BaselineRegionSource>(
          failureReason,
          "baseline region edge has no shared materialization facts");
    const size_t edgeIndex = static_cast<size_t>(std::distance(
        preparation.schedulingEdgeStrategies.begin(), original));
    if (edgeIndex >= preparation.schedulingEdgeFacts.size())
      return failCardModuleValue<BaselineRegionSource>(
          failureReason,
          "baseline region edge facts are outside the selected edge domain");
    SpatialEdgeStrategy mapped = strategy;
    mapped.producer = mapping.lookupOrNull(strategy.producer);
    mapped.consumer = mapping.lookupOrNull(strategy.consumer);
    if (!mapped.producer || !mapped.consumer)
      return failCardModuleValue<BaselineRegionSource>(
          failureReason,
          "baseline region did not clone a selected edge endpoint");
    result.edgeStrategies.push_back(std::move(mapped));
    result.edgeFacts.push_back(preparation.schedulingEdgeFacts[edgeIndex]);
  }

  if (mlir::failed(mlir::verify(*result.module)))
    return failCardModuleValue<BaselineRegionSource>(
        failureReason, "baseline region source is not verifier-legal");
  return result;
}

static mlir::FailureOr<mlir::func::FuncOp> lowerTileEntryFromSource(
    mlir::ModuleOp sourceModule, mlir::func::FuncOp sourceProgram,
    unsigned sourceArgumentCount,
    llvm::ArrayRef<StructuredOpTemporalTile> operationTemporalTiles,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    TileId tileId, const TileMapping &mapping,
    llvm::ArrayRef<SpatialOutputShard> tileShards, bool materializeTile,
    llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies,
    llvm::ArrayRef<SpatialEdgeMaterializationFacts> edgeFacts,
    std::string *failureReason,
    StructuredMaterializationRelations *tileRelations) {
  mlir::func::FuncOp entry;
  const bool hasEdgeAction =
      materializeTile &&
      llvm::any_of(edgeStrategies, [&](const SpatialEdgeStrategy &strategy) {
        return isSpatialEdgeStrategyIncidentOnTile(strategy, tileId);
      });
  if (hasEdgeAction) {
    mlir::OwningOpRef<mlir::ModuleOp> loweredShard;
    if (mlir::failed(lowerSpatialEdgeStrategiesToTileRegionModule(
            sourceModule, sourceArgumentCount, tileShards, tileId,
            mapping.materializationMode, edgeStrategies, loweredShard,
            failureReason,
            /*currentLogicalPartition=*/0, operationTemporalTiles,
            operationNodes, tileRelations, edgeFacts)))
      return mlir::failure();
    mlir::FailureOr<mlir::func::FuncOp> loweredEntry =
        takeLoweredTensorProgram(*loweredShard, failureReason);
    if (mlir::failed(loweredEntry))
      return mlir::failure();
    entry = *loweredEntry;
  } else if (materializeTile && !tileShards.empty()) {
    mlir::OwningOpRef<mlir::ModuleOp> loweredShard;
    if (mlir::failed(lowerSpatialOutputShardsToTileRegionModule(
            sourceModule, sourceArgumentCount, tileShards, loweredShard,
            failureReason,
            /*currentLogicalPartition=*/0, operationTemporalTiles,
            operationNodes, tileRelations)))
      return mlir::failure();
    mlir::FailureOr<mlir::func::FuncOp> loweredEntry =
        takeLoweredTensorProgram(*loweredShard, failureReason);
    if (mlir::failed(loweredEntry))
      return mlir::failure();
    entry = *loweredEntry;
  } else {
    mlir::FailureOr<mlir::func::FuncOp> noWork =
        createNoWorkEntry(sourceProgram, failureReason);
    if (mlir::failed(noWork))
      return mlir::failure();
    entry = *noWork;
  }
  return entry;
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
    StructuredMaterializationRelations *tileRelations,
    std::optional<llvm::ArrayRef<SpatialEdgeStrategy>>
        narrowedEdgeStrategies = std::nullopt) {
  const llvm::ArrayRef<SpatialEdgeStrategy> edgeStrategies =
      narrowedEdgeStrategies
          ? *narrowedEdgeStrategies
          : llvm::ArrayRef<SpatialEdgeStrategy>(
                preparation.schedulingEdgeStrategies);
  return lowerTileEntryFromSource(
      *preparation.schedulingModule, preparation.schedulingProgram,
      preparation.sourceArgumentCount,
      preparation.schedulingOperationTemporalTiles,
      preparation.schedulingOperationNodes, tileId, mapping, tileShards,
      materializeTile, edgeStrategies,
      narrowedEdgeStrategies
          ? llvm::ArrayRef<SpatialEdgeMaterializationFacts>{}
          : llvm::ArrayRef<SpatialEdgeMaterializationFacts>(
                preparation.schedulingEdgeFacts),
      failureReason, tileRelations);
}
/// Concatenates already-materialized per-component Tile entries of one Tile
/// into one entry function. Every component entry shares the full-card
/// input/result ABI and carries no boundary arguments; the merged entry keeps
/// the first entry's destination allocations as the single shared
/// destination set, rebases every later entry onto them and returns them
/// exactly once. Independent structured roots on one Tile therefore form
/// multiple sequential regions without any post-hoc region repair.
static mlir::FailureOr<mlir::func::FuncOp> concatenateTileEntries(
    llvm::SmallVectorImpl<mlir::func::FuncOp> &entries,
    unsigned sourceArgumentCount, StructuredMaterializationRelations &relations,
    std::string *failureReason) {
  if (entries.empty())
    return failCardModule(
        failureReason,
        "Tile entry concatenation needs at least one component entry");
  mlir::func::FuncOp first = entries.front();
  const unsigned resultCount = first.getNumResults();
  if (first.getNumArguments() != sourceArgumentCount + resultCount)
    return failCardModule(
        failureReason,
        "component entry does not have the exact private scheduling boundary");
  // Every SSA rebase performed by the concatenation also retargets the
  // current-IR buffer relations, so the merged relations describe exactly the
  // values that survive in the merged entry.
  auto retargetRelations = [&](mlir::Value oldValue, mlir::Value newValue) {
    if (oldValue == newValue)
      return;
    auto retarget = [&](auto &entries) {
      for (auto &relation : entries)
        if (relation.buffer == oldValue)
          relation.buffer = newValue;
    };
    retarget(relations.operationResultBuffers);
    retarget(relations.operandBuffers);
    retarget(relations.outputBuffers);
  };
  mlir::Block &mergedBody = first.getBody().front();
  // The first entry's trailing tensor boundary becomes the merged return;
  // later entries retarget any relation referencing their own trailing
  // to_tensor results onto the corresponding merged boundary.
  llvm::SmallVector<mlir::Value, 4> mergedBoundaryTensors;
  {
    mlir::Operation *lastRegion = nullptr;
    for (mlir::Operation &operation : mergedBody.without_terminator())
      if (mlir::isa<TileRegionOp>(operation))
        lastRegion = &operation;
    for (mlir::Operation *node = lastRegion ? lastRegion->getNextNode()
                                            : nullptr;
         node; node = node->getNextNode()) {
      if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(node))
        mergedBoundaryTensors.push_back(toTensor.getResult());
    }
  }
  if (mergedBoundaryTensors.size() != resultCount)
    return failCardModule(
        failureReason,
        "component entry does not expose one tensor boundary per result");
  auto entryDestinations = [&](mlir::func::FuncOp entry)
      -> llvm::SmallVector<mlir::Value, 4> {
    llvm::SmallVector<mlir::Value, 4> destinations;
    for (mlir::Operation &operation : entry.getBody().front()) {
      auto region = mlir::dyn_cast<TileRegionOp>(operation);
      if (!region)
        continue;
      for (unsigned index = 0; index < resultCount; ++index) {
        mlir::Value operand =
            region.getOperand(sourceArgumentCount + index);
        if (llvm::is_contained(destinations, operand))
          continue;
        destinations.push_back(operand);
      }
      if (!destinations.empty())
        break;
    }
    return destinations;
  };
  const llvm::SmallVector<mlir::Value, 4> sharedDestinations =
      entryDestinations(first);
  if (sharedDestinations.size() != resultCount)
    return failCardModule(
        failureReason,
        "component entry does not expose one destination per result");
  for (mlir::func::FuncOp entry : llvm::drop_begin(entries)) {
    if (entry.getNumArguments() != first.getNumArguments() ||
        entry.getResultTypes() != first.getResultTypes())
      return failCardModule(
          failureReason,
          "component entries do not share one full-card ABI");
    const llvm::SmallVector<mlir::Value, 4> destinations =
        entryDestinations(entry);
    if (destinations.size() != resultCount)
      return failCardModule(
          failureReason,
          "component entry does not expose one destination per result");
    // Rebase the entry onto the merged arguments and the shared
    // destinations, then splice its body (minus its own trailing tensor
    // boundary) in front of the merged return.
    for (unsigned index = 0; index < entry.getNumArguments(); ++index) {
      retargetRelations(entry.getArgument(index),
                        mergedBody.getArgument(index));
      entry.getArgument(index).replaceAllUsesWith(
          mergedBody.getArgument(index));
    }
    for (auto [destination, shared] :
         llvm::zip_equal(destinations, sharedDestinations)) {
      retargetRelations(destination, shared);
      mlir::Value mutableDestination = destination;
      mutableDestination.replaceAllUsesWith(shared);
    }
    // Each region keeps its own destination block arguments; the region
    // operands were already rebased onto the shared destination allocations,
    // so the block arguments receive the shared values through the rewired
    // operand list and every relation keeps naming its in-region buffer.
    mlir::Block &entryBody = entry.getBody().front();
    mlir::Operation *lastRegion = nullptr;
    for (mlir::Operation &operation : entryBody.without_terminator())
      if (mlir::isa<TileRegionOp>(operation))
        lastRegion = &operation;
    // Everything after the last region is the entry's own tensor boundary
    // (to_tensor/return); the merged entry returns the shared destinations.
    // Relations and uses referencing the entry's trailing to_tensor results
    // retarget onto the merged tensor boundary before the ops are erased.
    {
      llvm::SmallVector<mlir::Value, 4> entryBoundaryTensors;
      for (mlir::Operation *node = lastRegion ? lastRegion->getNextNode()
                                              : nullptr;
           node; node = node->getNextNode()) {
        if (auto toTensor =
                mlir::dyn_cast<mlir::bufferization::ToTensorOp>(node))
          entryBoundaryTensors.push_back(toTensor.getResult());
      }
      if (entryBoundaryTensors.size() != resultCount)
        return failCardModule(
            failureReason,
            "component entry does not expose one tensor boundary per result");
      for (auto [boundary, merged] :
           llvm::zip_equal(entryBoundaryTensors, mergedBoundaryTensors)) {
        retargetRelations(boundary, merged);
        mlir::Value mutableBoundary = boundary;
        mutableBoundary.replaceAllUsesWith(merged);
      }
    }
    for (mlir::Operation *node = lastRegion ? lastRegion->getNextNode()
                                            : nullptr;
         node && node != entryBody.getTerminator();) {
      mlir::Operation *next = node->getNextNode();
      node->erase();
      node = next;
    }
    // Rebased entry-level views whose results no longer have users (for
    // example a duplicated destination to_memref) are dead after the SSA
    // rebase and must not reach the merged entry.
    for (mlir::Operation &operation :
         llvm::make_early_inc_range(entryBody.without_terminator())) {
      if (mlir::isa<TileRegionOp>(operation))
        continue;
      if (llvm::all_of(operation.getResults(),
                       [](mlir::OpResult result) { return result.use_empty(); }))
        operation.erase();
    }
    for (mlir::Operation &operation :
         llvm::make_early_inc_range(entryBody.without_terminator()))
      operation.moveBefore(mergedBody.getTerminator());
    entry->erase();
  }
  return first;
}

/// Deterministic baseline structural postcondition: the canonical carrier
/// and the per-component construction give every compute TileRegion exactly
/// one structured compute root. A zero-root compute region violates the
/// contract as much as a multi-root region; a movement-only boundary region
/// legitimately carries no compute root.
static mlir::LogicalResult verifyOneStructuredRootPerRegion(
    mlir::func::FuncOp entry, TileId tileId,
    const StructuredMaterializationRelations &tileRelations,
    std::string *failureReason) {
  bool contractViolated = false;
  unsigned violatingRegion = 0;
  unsigned regionOrdinal = 0;
  llvm::SmallVector<uint32_t, 4> violatingRoots;
  llvm::SmallVector<uint32_t, 4> roots;
  entry.walk([&](TileRegionOp region) {
    if (contractViolated || region->getParentOfType<TileRegionOp>())
      return;
    const unsigned currentRegion = regionOrdinal++;
    bool hasCompute = false;
    region.walk([&](mlir::Operation *operation) {
      if (mlir::isa<ComputeFillOp, ComputeConvertOp, ComputeGemmOp,
                    ComputeConvOp, ComputeElementwiseOp, ComputeReduceOp>(
              operation))
        hasCompute = true;
    });
    if (!hasCompute)
      return;
    roots.clear();
    // Body emission records the physical buffer chosen for a structured
    // result. That value may be a TileRegion block argument, an allocation
    // result or a nested-loop result; requiring it to equal an arbitrary op
    // result found by a walk drops the block-argument case and reports a
    // source-only peer-send region as rootless. Region ancestry is the exact
    // ownership predicate for all of those current-SSA forms.
    for (const auto &relation : tileRelations.operationResultBuffers) {
      mlir::Value buffer = relation.buffer;
      if (buffer && region.getBody().isAncestor(buffer.getParentRegion()) &&
          !llvm::is_contained(roots, relation.structuredNodeId))
        roots.push_back(relation.structuredNodeId);
    }
    if (roots.size() != 1) {
      contractViolated = true;
      violatingRegion = currentRegion;
      violatingRoots = roots;
    }
  });
  if (contractViolated) {
    std::string detail;
    llvm::raw_string_ostream diagnostic(detail);
    diagnostic << "carrier-boundary materialization on Tile "
               << tileId.getValue() << " produced compute region "
               << violatingRegion << " with " << violatingRoots.size()
               << " structured roots";
    if (!violatingRoots.empty()) {
      diagnostic << " [";
      llvm::interleaveComma(violatingRoots, diagnostic);
      diagnostic << ']';
    }
    diagnostic << "; exactly one is required; entry carries "
               << tileRelations.operationResultBuffers.size()
               << " structured result-buffer relations";
    return failCardModule(failureReason, diagnostic.str());
  }
  return mlir::success();
}

/// Lowers one Tile entry for the deterministic baseline
/// (IndependentDDRStages). Structured roots on the Tile that are not
/// connected by any same-Tile dependency edge form separate dependency
/// components; each component is materialized through its own narrow closure
/// and the resulting regions are concatenated in canonical node order, so
/// every TileRegion carries exactly one structured compute root by
/// construction. Roots connected by carrier edges stay in one component: the
/// canonical carrier emits the exact DDR store/reload and splits the region
/// at construction time.
static mlir::FailureOr<mlir::func::FuncOp> lowerBaselineTileEntry(
    const TileMaterializationPreparation &preparation, CardId cardId,
    TileId tileId, const TileMapping &mapping,
    llvm::ArrayRef<SpatialOutputShard> tileShards,
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>> observableOutputRootNodes,
    StructuredMaterializationRelations &tileRelations,
    BaselineRegionClosureCache &closureCache, std::string *failureReason,
    TileEntryMaterializationStatistics *statistics = nullptr) {
  auto mappedNode = [&](mlir::Operation *operation)
      -> std::optional<uint32_t> {
    for (const StructuredOperationNodeMapping &node :
         preparation.schedulingOperationNodes)
      if (node.operation == operation)
        return node.structuredNodeId;
    return std::nullopt;
  };
  llvm::SmallVector<uint32_t, 4> roots;
  for (const SpatialOutputShard &shard : tileShards) {
    if (observableOutputRootNodes.empty())
      return failCardModule(
          failureReason,
          "baseline materialization requires observable root identity");
    if (shard.outputIndex >= observableOutputRootNodes.size())
      return failCardModule(failureReason,
                            "output shard index is outside the observable "
                            "root mapping");
    const auto &shardRoots = observableOutputRootNodes[shard.outputIndex];
    if (shardRoots.size() != 1)
      return failCardModule(
          failureReason,
          "baseline output must have exactly one nearest structured root");
    if (!llvm::is_contained(roots, shardRoots.front()))
      roots.push_back(shardRoots.front());
  }
  // Program-result shards do not cover intermediate producer-only Tiles.
  // Add exactly the structured endpoints that own a selected physical action
  // on this Tile. Placement participation alone is not evidence of actual
  // work.
  for (const SpatialEdgeStrategy &strategy :
       preparation.schedulingEdgeStrategies) {
    std::optional<uint32_t> producerNode = mappedNode(strategy.producer);
    std::optional<uint32_t> consumerNode = mappedNode(strategy.consumer);
    const bool materializesProducer =
        producerNode &&
        (strategy.action == SpatialEdgeAction::PeerFragments
             ? llvm::any_of(strategy.fragments,
                            [&](const SpatialEdgeFragment &fragment) {
                              return fragment.sourceTile == tileId;
                            })
             : strategy.sourceTile == tileId);
    const bool materializesConsumer =
        consumerNode && strategy.destinationTile == tileId;
    if (materializesProducer && !llvm::is_contained(roots, *producerNode))
      roots.push_back(*producerNode);
    if (materializesConsumer && !llvm::is_contained(roots, *consumerNode))
      roots.push_back(*consumerNode);
  }
  if (roots.empty())
    return lowerTileEntry(
        preparation, cardId, tileId, mapping, /*tileShards=*/{},
        /*materializeTile=*/true, failureReason, &tileRelations,
        llvm::ArrayRef<SpatialEdgeStrategy>{});
  // Same-Tile dependency edges between two structured roots connect their
  // components. A strategy without two structured endpoints keeps the whole
  // Tile in one component: its support closure cannot be attributed to one
  // root here.
  llvm::DenseMap<uint32_t, uint32_t> parent;
  auto findRoot = [&](uint32_t node) {
    while (parent.lookup(node) != node)
      node = parent[node];
    return node;
  };
  for (uint32_t root : roots)
    parent[root] = root;
  bool singleComponent = false;
  for (const SpatialEdgeStrategy &strategy :
       preparation.schedulingEdgeStrategies) {
    if (strategy.destinationTile != tileId)
      continue;
    std::optional<uint32_t> producerNode = mappedNode(strategy.producer);
    std::optional<uint32_t> consumerNode = mappedNode(strategy.consumer);
    if (!producerNode || !consumerNode) {
      singleComponent = true;
      break;
    }
    if (!llvm::is_contained(roots, *producerNode) ||
        !llvm::is_contained(roots, *consumerNode))
      continue;
    parent[findRoot(*producerNode)] = findRoot(*consumerNode);
  }
  llvm::SmallVector<std::pair<uint32_t, uint32_t>, 4> rootComponents;
  for (uint32_t root : roots)
    rootComponents.push_back({root, findRoot(root)});
  if (singleComponent || rootComponents.size() <= 1 ||
      llvm::all_of(rootComponents, [&](const auto &pair) {
        return pair.second == rootComponents.front().second;
      })) {
    mlir::FailureOr<mlir::func::FuncOp> entry = lowerTileEntry(
        preparation, cardId, tileId, mapping, tileShards,
        /*materializeTile=*/true, failureReason, &tileRelations);
    if (mlir::failed(entry))
      return entry;
    // Every same-Tile dependency is a selected RegionCut in the deterministic
    // baseline. Its canonical store/reload materializer forms the region
    // boundary before this complete entry is returned.
    if (mlir::failed(verifyOneStructuredRootPerRegion(*entry, tileId,
                                                      tileRelations,
                                                      failureReason)))
      return mlir::failure();
    return entry;
  }

  // Materialize one entry per dependency component and concatenate the
  // regions in canonical root order.
  llvm::sort(rootComponents, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  llvm::SmallVector<mlir::func::FuncOp, 4> componentEntries;
  for (const auto &[root, component] : rootComponents) {
    // One entry per distinct component id; roots are already unique.
    if (llvm::any_of(rootComponents, [&](const auto &other) {
          return other.second == component && other.first < root;
        }))
      continue;
    llvm::SmallVector<SpatialOutputShard, 4> componentShards;
    for (const SpatialOutputShard &shard : tileShards) {
      const auto &shardRoots = observableOutputRootNodes[shard.outputIndex];
      if (shardRoots.size() != 1 ||
          findRoot(shardRoots.front()) != component)
        continue;
      componentShards.push_back(shard);
    }
    llvm::SmallVector<SpatialEdgeStrategy, 8> componentStrategies;
    for (const SpatialEdgeStrategy &strategy :
         preparation.schedulingEdgeStrategies) {
      std::optional<uint32_t> producerNode = mappedNode(strategy.producer);
      std::optional<uint32_t> consumerNode = mappedNode(strategy.consumer);
      const bool ownsProducerEndpoint =
          producerNode && llvm::is_contained(roots, *producerNode) &&
          findRoot(*producerNode) == component &&
          (strategy.action == SpatialEdgeAction::PeerFragments
               ? llvm::any_of(strategy.fragments,
                              [&](const SpatialEdgeFragment &fragment) {
                                return fragment.sourceTile == tileId;
                              })
               : strategy.sourceTile == tileId);
      const bool ownsConsumerEndpoint =
          consumerNode && llvm::is_contained(roots, *consumerNode) &&
          findRoot(*consumerNode) == component &&
          strategy.destinationTile == tileId;
      // A remote sibling is deliberately absent from this Tile's root set,
      // but its selected edge action is still the consumer component's typed
      // boundary. Requiring both endpoints to be local drops PeerFragments
      // and lets ordinary output traversal re-fuse that remote producer.
      if (ownsProducerEndpoint || ownsConsumerEndpoint)
        componentStrategies.push_back(strategy);
    }
    StructuredMaterializationRelations componentRelations;
    mlir::FailureOr<BaselineRegionSource> regionSource =
        buildBaselineRegionSource(preparation, componentShards,
                                  componentStrategies, closureCache,
                                  failureReason);
    if (mlir::failed(regionSource))
      return mlir::failure();
    if (statistics && regionSource->analyzedClosure) {
      ++statistics->baselineRegionClosureAnalyses;
      statistics->baselineRegionClosureAnalysisOperations +=
          regionSource->operationCount;
    }
    mlir::FailureOr<mlir::func::FuncOp> entry = lowerTileEntryFromSource(
        *regionSource->module, regionSource->program,
        preparation.sourceArgumentCount, regionSource->operationTemporalTiles,
        regionSource->operationNodes, tileId, mapping, componentShards,
        /*materializeTile=*/true, regionSource->edgeStrategies,
        regionSource->edgeFacts, failureReason, &componentRelations);
    if (mlir::failed(entry))
      return mlir::failure();
    tileRelations.operationResultBuffers.append(
        componentRelations.operationResultBuffers.begin(),
        componentRelations.operationResultBuffers.end());
    tileRelations.operandBuffers.append(
        componentRelations.operandBuffers.begin(),
        componentRelations.operandBuffers.end());
    tileRelations.outputBuffers.append(componentRelations.outputBuffers.begin(),
                                       componentRelations.outputBuffers.end());
    componentEntries.push_back(*entry);
  }
  mlir::FailureOr<mlir::func::FuncOp> merged = concatenateTileEntries(
      componentEntries, preparation.sourceArgumentCount, tileRelations,
      failureReason);
  if (mlir::failed(merged))
    return merged;
  if (mlir::failed(verifyOneStructuredRootPerRegion(*merged, tileId,
                                                    tileRelations,
                                                    failureReason)))
    return mlir::failure();
  return merged;
}

static mlir::LogicalResult lowerPreparedTensorProgramToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    const TileMaterializationPreparation &preparation,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule, std::string *failureReason,
    StructuredMaterializationRelations *materializationRelations,
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>>
        observableOutputRootNodes,
    CardModuleMaterializationStatistics *statistics) {
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

  struct MaterializedTileEntry {
    mlir::OwningOpRef<mlir::ModuleOp> owner;
    mlir::func::FuncOp entry;
    StructuredMaterializationRelations relations;
    llvm::SmallVector<int64_t, 4> coveredShardExtents;
    TileEntryMaterializationStatistics statistics;
    std::string failureReason;
    bool succeeded = false;
  };
  std::vector<MaterializedTileEntry> tileEntries(
      preparation.availableTiles.size());
  BaselineRegionClosureCache baselineClosureCache;
  unsigned materializationWorkers = 1;
  {
    mlir::ParallelDiagnosticHandler parallelDiagnostics(
        sourceModule.getContext());
    materializationWorkers = wafer::support::runBoundedParallelWork(
        sourceModule.getContext(), preparation.availableTiles.size(),
        [&](size_t tileIndex) {
          parallelDiagnostics.setOrderIDForThread(tileIndex);
          auto eraseDiagnosticOrder = llvm::make_scope_exit(
              [&] { parallelDiagnostics.eraseOrderIDForThread(); });
          const TileId tileId = preparation.availableTiles[tileIndex];
          MaterializedTileEntry &result = tileEntries[tileIndex];
          result.coveredShardExtents.assign(preparation.outputDomains.size(),
                                            0);
          llvm::SmallVector<SpatialOutputShard, 4> tileShards;
          collectTileOutputShards(preparation, tileId, tileShards,
                                  result.coveredShardExtents);
          mlir::FailureOr<mlir::func::FuncOp> entry;
          if (mapping.materializationMode ==
              SpatialDataflowMaterializationMode::IndependentDDRStages)
            entry = lowerBaselineTileEntry(
                preparation, cardId, tileId, mapping, tileShards,
                observableOutputRootNodes, result.relations,
                baselineClosureCache,
                &result.failureReason, &result.statistics);
          else
            entry = lowerTileEntry(
                preparation, cardId, tileId, mapping, tileShards,
                /*materializeTile=*/true, &result.failureReason,
                &result.relations);
          if (mlir::failed(entry))
            return;
          result.owner = mlir::ModuleOp::create(sourceModule.getLoc());
          result.owner->getBody()->push_back(entry->getOperation());
          result.entry = *entry;
          result.succeeded = true;
        });
  }

  llvm::SmallVector<int64_t, 4> coveredShardExtents(
      preparation.outputDomains.size(), 0);
  StructuredMaterializationRelations resultRelations;
  CardModuleMaterializationStatistics collectedStatistics;
  collectedStatistics.tileEntryMaterializations = tileEntries.size();
  collectedStatistics.maximumTileMaterializationWorkers =
      materializationWorkers;
  for (auto [tileIndex, materialized] : llvm::enumerate(tileEntries)) {
    const TileId tileId = preparation.availableTiles[tileIndex];
    if (!materialized.succeeded) {
      if (failureReason)
        *failureReason = materialized.failureReason;
      return mlir::failure();
    }
    for (auto [outputIndex, covered] :
         llvm::enumerate(materialized.coveredShardExtents))
      coveredShardExtents[outputIndex] += covered;
    collectedStatistics.regionClosureAnalyses +=
        materialized.statistics.baselineRegionClosureAnalyses;
    collectedStatistics.regionClosureAnalysisOperations +=
        materialized.statistics.baselineRegionClosureAnalysisOperations;

    auto tile = cardBuilder.create<TileModuleOp>(
        sourceModule.getLoc(),
        cardBuilder.getI64IntegerAttr(tileId.getValue()));
    tile.getBody().push_back(new mlir::Block());
    mlir::Block &tileBody = tile.getBody().front();

    materialized.entry->remove();
    tileBody.push_back(materialized.entry.getOperation());
    // Peer endpoint verification resolves physical identities through the
    // enclosing module topology, so transform and verify the entry only after
    // it has been attached to its final Card/Tile IR scope.
    if (mlir::failed(removeSchedulingOutputDestinations(
            materialized.entry, preparation.sourceArgumentCount,
            materialized.relations, failureReason)))
      return mlir::failure();
    resultRelations.operationResultBuffers.append(
        materialized.relations.operationResultBuffers.begin(),
        materialized.relations.operationResultBuffers.end());
    resultRelations.operandBuffers.append(
        materialized.relations.operandBuffers.begin(),
        materialized.relations.operandBuffers.end());
    resultRelations.outputBuffers.append(
        materialized.relations.outputBuffers.begin(),
        materialized.relations.outputBuffers.end());
  }

  if (!relationsBelongTo(result->getOperation(), resultRelations))
    return failCardModule(
        failureReason,
        "CardModule materialization produced a buffer relation outside the "
        "current IR");

  for (auto [outputIndex, covered] : llvm::enumerate(coveredShardExtents)) {
    const OutputTileMapping *output = preparation.outputMappings[outputIndex];
    const int64_t expected = output->shardDimension
                                 ? preparation.outputDomains[outputIndex]
                                       [*output->shardDimension]
                                 : 1;
    if (covered != expected)
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
  if (statistics)
    *statistics = collectedStatistics;
  cardModule = std::move(result);
  return mlir::success();
}


struct TileMaterializationSession::Impl {
  mlir::ModuleOp sourceModule;
  CardId cardId{0};
  TileMapping mapping;
  TileMaterializationPreparation preparation;
  llvm::SmallVector<llvm::SmallVector<uint32_t, 2>, 4>
      observableOutputRootNodes;
};

struct TileMaterializationSourceSession::Impl {
  TileMaterializationSourcePreparation preparation;
};

TileMaterializationSourceSession::TileMaterializationSourceSession(
    std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

TileMaterializationSourceSession::~TileMaterializationSourceSession() =
    default;
TileMaterializationSourceSession::TileMaterializationSourceSession(
    TileMaterializationSourceSession &&) noexcept = default;
TileMaterializationSourceSession &
TileMaterializationSourceSession::operator=(
    TileMaterializationSourceSession &&) noexcept = default;

mlir::FailureOr<std::unique_ptr<TileMaterializationSourceSession>>
TileMaterializationSourceSession::create(
    mlir::ModuleOp sourceModule, CardId cardId,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    std::string *failureReason) {
  mlir::FailureOr<TileMaterializationSourcePreparation> preparation =
      prepareTileMaterializationSource(sourceModule, cardId, operationNodes,
                                       failureReason);
  if (mlir::failed(preparation))
    return mlir::failure();
  auto state = std::make_unique<Impl>();
  state->preparation = std::move(*preparation);
  return std::unique_ptr<TileMaterializationSourceSession>(
      new TileMaterializationSourceSession(std::move(state)));
}

TileMaterializationSession::TileMaterializationSession(
    std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

TileMaterializationSession::~TileMaterializationSession() = default;
TileMaterializationSession::TileMaterializationSession(
    TileMaterializationSession &&) noexcept = default;
TileMaterializationSession &TileMaterializationSession::operator=(
    TileMaterializationSession &&) noexcept = default;

mlir::FailureOr<std::unique_ptr<TileMaterializationSession>>
TileMaterializationSession::create(
    const TileMaterializationSourceSession &sourceSession,
    const TileMapping &mapping, std::string *failureReason,
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>>
        observableOutputRootNodes) {
  const TileMaterializationSourcePreparation &source =
      sourceSession.impl->preparation;
  auto state = std::make_unique<Impl>();
  state->sourceModule = source.sourceModule;
  state->cardId = source.cardId;
  state->mapping = mapping;
  state->observableOutputRootNodes.assign(observableOutputRootNodes.begin(),
                                          observableOutputRootNodes.end());
  mlir::FailureOr<TileMaterializationPreparation> preparation =
      prepareTileMaterialization(source, state->mapping, failureReason);
  if (mlir::failed(preparation))
    return mlir::failure();
  state->preparation = std::move(*preparation);
  return std::unique_ptr<TileMaterializationSession>(
      new TileMaterializationSession(std::move(state)));
}

mlir::FailureOr<std::unique_ptr<TileMaterializationSession>>
TileMaterializationSession::create(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>>
        observableOutputRootNodes) {
  mlir::FailureOr<std::unique_ptr<TileMaterializationSourceSession>> source =
      TileMaterializationSourceSession::create(
          sourceModule, cardId, operationNodes, failureReason);
  if (mlir::failed(source))
    return mlir::failure();
  return create(**source, mapping, failureReason, observableOutputRootNodes);
}

mlir::LogicalResult TileMaterializationSession::lowerCardModule(
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule,
    StructuredMaterializationRelations *materializationRelations,
    std::string *failureReason,
    CardModuleMaterializationStatistics *statistics) const {
  return lowerPreparedTensorProgramToCardModule(
      impl->sourceModule, impl->cardId, impl->mapping, impl->preparation,
      cardModule, failureReason, materializationRelations,
      impl->observableOutputRootNodes, statistics);
}


mlir::LogicalResult lowerTensorProgramToCardModule(
    mlir::ModuleOp sourceModule, CardId cardId, const TileMapping &mapping,
    mlir::OwningOpRef<mlir::ModuleOp> &cardModule, std::string *failureReason,
    llvm::ArrayRef<StructuredOperationNodeMapping> operationNodes,
    StructuredMaterializationRelations *materializationRelations,
    llvm::ArrayRef<llvm::SmallVector<uint32_t, 2>>
        observableOutputRootNodes) {
  mlir::FailureOr<std::unique_ptr<TileMaterializationSession>> session =
      TileMaterializationSession::create(
          sourceModule, cardId, mapping, failureReason, operationNodes,
          observableOutputRootNodes);
  if (mlir::failed(session))
    return mlir::failure();
  return (*session)->lowerCardModule(cardModule, materializationRelations,
                                     failureReason);
}

} // namespace wafer
