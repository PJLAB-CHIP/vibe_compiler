//===- SelectedEdgeFinalization.cpp - Selected TileRegion finalization ----===//

#include "SelectedEdgeLoweringInternal.h"

#include "EdgeFragmentPlanning.h"

#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/Twine.h"

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {
namespace {

mlir::LogicalResult rebindSelectedReceiveEndpoints(
    llvm::MutableArrayRef<CandidatePeerEndpoint> endpoints,
    llvm::ArrayRef<MappedStrategy> mappedStrategies,
    std::string *failureReason) {
  for (CandidatePeerEndpoint &endpoint : endpoints) {
    if (endpoint.kind != CandidatePeerEndpointKind::Receive ||
        !endpoint.selectedFragment)
      continue;
    llvm::SmallVector<mlir::bufferization::ToTensorOp, 2> matches;
    if (endpoint.carrierBuffer)
      for (mlir::Operation *user : endpoint.carrierBuffer.getUsers())
        if (auto toTensor =
                mlir::dyn_cast<mlir::bufferization::ToTensorOp>(user);
            toTensor && toTensor.getMemref() == endpoint.carrierBuffer &&
            !toTensor.getResult().use_empty())
          matches.push_back(toTensor);
    if (matches.size() != 1 || matches.front().getResult().use_empty()) {
      std::string detail;
      llvm::raw_string_ostream diagnostic(detail);
      diagnostic << "selected receive fragment did not survive as one exact "
                    "consumer value (matches="
                 << matches.size();
      if (matches.size() == 1 && matches.front().getResult().use_empty())
        diagnostic << ", unused";
      diagnostic << ", peer=" << endpoint.peer.getValue()
                 << ", communication_id=" << endpoint.communicationId
                 << ", payload_slice=" << endpoint.payloadSlice;
      if (endpoint.selectedFragment) {
        diagnostic << ", offsets=[";
        llvm::interleaveComma(endpoint.selectedFragment->offsets, diagnostic);
        diagnostic << "], sizes=[";
        llvm::interleaveComma(endpoint.selectedFragment->sizes, diagnostic);
        diagnostic << ']';
      }
      for (const MappedStrategy &mapped : mappedStrategies) {
        auto fragment =
            llvm::find_if(mapped.strategy.fragments,
                          [&](const SpatialEdgeFragment &candidate) {
                            return &candidate == endpoint.selectedFragment;
                          });
        if (fragment == mapped.strategy.fragments.end())
          continue;
        mlir::Operation *producer = mapped.sourceProducer;
        mlir::Operation *consumer = mapped.sourceConsumer;
        if (!producer || !consumer)
          break;
        diagnostic
            << ", edge=" << producer->getName() << ':'
            << producer->getResult(mapped.strategy.producerResult).getType();
        diagnostic
            << " -> " << consumer->getName() << " operand "
            << mapped.strategy.consumerOperand << ':'
            << consumer->getOperand(mapped.strategy.consumerOperand).getType();
        diagnostic << ", support=" << mapped.hasProducerToConsumerChain
                   << ", producer_demand_offsets=[";
        llvm::interleaveComma(mapped.strategy.producerOffsets, diagnostic);
        diagnostic << "], producer_demand_sizes=[";
        llvm::interleaveComma(mapped.strategy.producerSizes, diagnostic);
        diagnostic << "], consumer_offsets=[";
        llvm::interleaveComma(mapped.strategy.consumerOffsets, diagnostic);
        diagnostic << "], consumer_sizes=[";
        llvm::interleaveComma(mapped.strategy.consumerSizes, diagnostic);
        diagnostic << ']';
        break;
      }
      diagnostic << ')';
      return reportSelectedEdgeFailure(failureReason, diagnostic.str());
    }
    endpoint.value = matches.front().getResult();
  }
  return mlir::success();
}

void eraseUnreadDirectPrivateLoads(mlir::Operation *operation) {
  llvm::SmallVector<StorageLoadOp, 8> unreadLoads;
  operation->walk([&](StorageLoadOp load) {
    auto allocation = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
    if (allocation && allocation.getResult().hasOneUse())
      unreadLoads.push_back(load);
  });
  for (StorageLoadOp load : unreadLoads) {
    auto allocation = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
    load.erase();
    if (allocation.getResult().use_empty())
      allocation.erase();
  }
}

} // namespace

mlir::LogicalResult
materializeSelectedOutputs(SelectedEdgeLoweringState &state) {
  TensorProgramScope scope = state.scope;
  TileId currentTile = state.currentTile;
  auto outputShards = state.outputShards;
  auto &mappedTemporalTiles = state.mapping.operationTemporalTiles;
  auto &mappedOperationNodes = state.mapping.operationNodes;
  auto &mappedStrategies = state.mapping.strategies;
  auto &preserved = state.preservedOperations;
  std::string *failureReason = state.failureReason;
  llvm::SmallVector<mlir::Operation *, 8> preservedOperations;
  for (mlir::Operation *operation : preserved)
    preservedOperations.push_back(operation);
  if (!outputShards.empty()) {
    mlir::func::ReturnOp returnOp = scope.getReturn();
    const bool directFullOutputs =
        outputShards.size() == scope.getOutputCount() &&
        llvm::all_of(outputShards, [&](const SpatialOutputShard &shard) {
          if (shard.outputIndex >= returnOp.getNumOperands() ||
              shard.temporalTileSizes != shard.sizes)
            return false;
          mlir::Operation *root =
              returnOp.getOperand(shard.outputIndex).getDefiningOp();
          return root &&
                 isFullStaticResultDomain(root, /*resultNumber=*/0,
                                          shard.offsets, shard.sizes) &&
                 isOneFullTemporalWave(root, mappedTemporalTiles) &&
                 llvm::any_of(
                     mappedStrategies, [&](const MappedStrategy &mapped) {
                       return mapped.strategy.destinationTile == currentTile &&
                              mapped.consumer == root;
                     });
        });
    if (directFullOutputs) {
      // Keep a one-wave observable consumer on its current SSA path so an
      // incoming selected peer fragment remains the exact value consumed by
      // that operation.  The ordinary full-result insert still writes the
      // functional output boundary; only the unnecessary clone-and-retile is
      // omitted.
      for (const SpatialOutputShard &shard : outputShards) {
        mlir::Value value = returnOp.getOperand(shard.outputIndex);
        mlir::Operation *root = value.getDefiningOp();
        mlir::FailureOr<mlir::Value> boundary =
            getCandidateOutputBoundary(scope, shard.outputIndex, failureReason);
        if (mlir::failed(boundary))
          return mlir::failure();
        mlir::OpBuilder builder(returnOp);
        returnOp->setOperand(shard.outputIndex,
                             insertExactSlice(builder, root->getLoc(), value,
                                              *boundary, shard.offsets,
                                              shard.sizes));
        preserved.insert(root);
      }
    } else if (mlir::failed(materializeCandidateOutputTileSlices(
                   scope, outputShards, mappedTemporalTiles, failureReason,
                   &mappedOperationNodes, preservedOperations))) {
      return mlir::failure();
    }
  } else {
    mlir::func::ReturnOp returnOp = scope.getReturn();
    for (unsigned index = 0; index < scope.getOutputCount(); ++index) {
      mlir::FailureOr<mlir::Value> output =
          getCandidateOutputBoundary(scope, index, failureReason);
      if (mlir::failed(output))
        return mlir::failure();
      returnOp->setOperand(index, *output);
    }
  }
  return mlir::success();
}

mlir::LogicalResult
verifySelectedReceiveExecutionSinks(SelectedEdgeLoweringState &state) {
  TensorProgramScope scope = state.scope;
  auto &endpoints = state.endpoints;
  std::string *failureReason = state.failureReason;
  mlir::func::ReturnOp finalReturn = scope.getReturn();
  for (const CandidatePeerEndpoint &receive : endpoints) {
    if (receive.kind != CandidatePeerEndpointKind::Receive)
      continue;
    mlir::Operation *receiveDefinition = receive.value.getDefiningOp();
    bool reachesExecutionSink = false;
    for (mlir::Value output : finalReturn.getOperands()) {
      llvm::DenseSet<mlir::Value> visited;
      if (isInBackwardClosure(output, receiveDefinition, visited)) {
        reachesExecutionSink = true;
        break;
      }
    }
    for (const CandidatePeerEndpoint &send : endpoints) {
      if (reachesExecutionSink || send.kind != CandidatePeerEndpointKind::Send)
        continue;
      llvm::DenseSet<mlir::Value> visited;
      if (isInBackwardClosure(send.value, receiveDefinition, visited))
        reachesExecutionSink = true;
    }
    // An explicit DDR stage intentionally breaks the tensor SSA path after
    // its functional insert: the store stays live, while a fresh read-only
    // to_tensor view prevents downstream temporal tiling from fusing the
    // producer into the consumer wave. Count that selected store as an
    // execution sink only when the receive reaches the inserted source and
    // the destination is rooted in a marked compiler-owned stage. The later
    // TileRegion split still proves the ordered store/reload interval.
    if (!reachesExecutionSink)
      reachesExecutionSink =
          selectedValueReachesDDRStage(state, receiveDefinition);
    if (!reachesExecutionSink)
      return reportSelectedEdgeFailure(
          failureReason,
          (llvm::Twine("selected receive does not reach an execution sink "
                       "after output traversal (communication_id=") +
           llvm::Twine(receive.communicationId) +
           ", payload_slice=" + llvm::Twine(receive.payloadSlice) + ")")
              .str());
  }
  return mlir::success();
}

mlir::LogicalResult finishSelectedEdgeLowering(
    SelectedEdgeLoweringState &state,
    mlir::OwningOpRef<mlir::ModuleOp> &resultModule,
    StructuredMaterializationRelations *materializationRelations) {
  mlir::ModuleOp sourceModule = state.sourceModule;
  unsigned functionalArgumentCount = state.functionalArgumentCount;
  int64_t currentLogicalPartition = state.currentLogicalPartition;
  bool requireOneStructuredRootPerRegion =
      state.requireOneStructuredRootPerRegion;
  auto &candidate = state.candidate;
  TensorProgramScope scope = state.scope;
  auto &mappedOperationNodes = state.mapping.operationNodes;
  auto &mappedStrategies = state.mapping.strategies;
  auto &endpoints = state.endpoints;
  auto &materialized = state.materializedSources;
  auto &selectedDDRStages = state.selectedDDRStages;
  auto &preserved = state.preservedOperations;
  std::string *failureReason = state.failureReason;
  auto &module = resultModule;
  // Send/receive-only Tiles need the unused remainder removed before body
  // conversion. Ordinary local actions retain the same source closure as the
  // direct output-shard path; its materializer and atomic conversion own that
  // cleanup.
  if (!endpoints.empty() || !materialized.empty())
    eraseDeadExcept(scope, preserved);

  if (mlir::failed(rebindSelectedReceiveEndpoints(endpoints, mappedStrategies,
                                                  failureReason)))
    return mlir::failure();

  for (const CandidatePeerEndpoint &endpoint : endpoints) {
    if (endpoint.kind != CandidatePeerEndpointKind::Receive)
      continue;
    mlir::Operation *defining = endpoint.value.getDefiningOp();
    if (!defining || defining->getBlock() != &scope.getBody() ||
        endpoint.value.use_empty())
      return reportSelectedEdgeFailure(
          failureReason,
          "selected consumer traversal does not use an exact receive fragment");
  }

  llvm::DenseSet<mlir::Operation *> liveOperations;
  candidate->walk(
      [&](mlir::Operation *operation) { liveOperations.insert(operation); });
  llvm::erase_if(mappedOperationNodes, [&](const auto &mapping) {
    return !liveOperations.contains(mapping.operation);
  });
  TileRegionEmissionRelations emissionRelations;
  if (mlir::failed(convertTensorProgramToTileRegionModuleInPlace(
          *candidate, sourceModule.getContext(), functionalArgumentCount,
          currentLogicalPartition, failureReason,
          /*suppressDiagnostics=*/true,
          // The unread-load cleanup below is part of this same atomic
          // construction. Verify the final TileRegion module once after that
          // mutation instead of verifying the large intermediate module and
          // then immediately verifying it again.
          /*verifyResult=*/false, /*populateFallbackFailureReason=*/true,
          endpoints, selectedDDRStages, &emissionRelations,
          mappedOperationNodes, requireOneStructuredRootPerRegion))) {
    return mlir::failure();
  }
  // Endpoint identity is not an SSA use. Once carrier construction is final, an
  // eager load into an otherwise unread private allocation is unobservable.
  eraseUnreadDirectPrivateLoads(candidate->getOperation());
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "analysis-phase", "selected-edge-materialization", "verify");
    if (mlir::failed(mlir::verify(*candidate)))
      return reportSelectedEdgeFailure(
          failureReason, "edge-action TileRegion result is not verifier-legal");
  }
  if (materializationRelations)
    *materializationRelations =
        std::move(emissionRelations.materializedBuffers);
  module = std::move(candidate);
  return mlir::success();
}

} // namespace wafer::tensor_program_to_tile_region
