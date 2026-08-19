//===- RootRegionEmitter.cpp - Direct structured-stage regions -------===//

#include "Internal.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <tuple>

using namespace wafer;

namespace wafer::tensor_program_to_tile_region {

mlir::FailureOr<TileRegionOp>
TileRegionBodyEmitter::emitStructuredStages(TensorProgramScope scope,
                                            mlir::RewriterBase &rewriter) {
  if (currentLogicalPartition < 0)
    return failAndReturn("logical partition must be non-negative");
  if (mlir::failed(verifyNamedLinalgPayloads(scope)))
    return mlir::failure();

  llvm::SmallVector<mlir::Operation *, 16> sourceOperations;
  for (mlir::Operation &operation : scope.getBody().without_terminator())
    sourceOperations.push_back(&operation);

  auto getTopLevelOperation = [&](mlir::Operation *operation) {
    while (operation && operation->getBlock() != &scope.getBody())
      operation = operation->getParentOp();
    return operation && operation->getBlock() == &scope.getBody() ? operation
                                                                  : nullptr;
  };

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<uint32_t, 2>>
      operationNodes;
  llvm::SmallVector<uint32_t, 8> nodeOrder;

  // A scalar linalg.fill that is consumed only as the initialization of one
  // supported structured compute is part of that consumer's root.  The
  // corresponding DPS-init dependency intentionally has no spatial carrier:
  // convertFill records the scalar value and the consumer initializes each
  // result tile directly.  Keep every other fill as an independent root so a
  // non-unique or materially observed initialization still requires an
  // explicit boundary.
  for (mlir::Operation *topLevel : sourceOperations) {
    topLevel->walk([&](mlir::linalg::FillOp fill) {
      if (!structuredNodeIds.contains(fill.getOperation()) ||
          !onlyFeedsScalarInitializedComputeInit(fill.getResult(0)))
        return;

      llvm::DenseSet<mlir::Value> visited;
      llvm::SmallVector<uint32_t, 2> consumerNodes;
      std::function<void(mlir::Value)> collectConsumerNodes =
          [&](mlir::Value value) {
            if (!value || !visited.insert(value).second)
              return;
            for (mlir::OpOperand &use : value.getUses()) {
              mlir::Operation *owner = use.getOwner();
              auto mapped = structuredNodeIds.find(owner);
              if (mapped != structuredNodeIds.end()) {
                auto dps =
                    mlir::dyn_cast<mlir::DestinationStyleOpInterface>(owner);
                if (!dps || !dps.isDpsInit(&use))
                  continue;
                for (uint32_t node : mapped->second)
                  if (!llvm::is_contained(consumerNodes, node))
                    consumerNodes.push_back(node);
                continue;
              }
              for (mlir::Value result : owner->getResults())
                if (mlir::isa<mlir::RankedTensorType>(result.getType()))
                  collectConsumerNodes(result);
            }
          };
      collectConsumerNodes(fill.getResult(0));
      if (consumerNodes.size() == 1)
        structuredNodeIds[fill.getOperation()] = consumerNodes;
    });
  }

  for (mlir::Operation *topLevel : sourceOperations) {
    llvm::SmallVector<uint32_t, 2> &nodes = operationNodes[topLevel];
    topLevel->walk([&](mlir::Operation *nested) {
      auto found = structuredNodeIds.find(nested);
      if (found == structuredNodeIds.end())
        return;
      for (uint32_t node : found->second) {
        if (!llvm::is_contained(nodes, node))
          nodes.push_back(node);
        if (!llvm::is_contained(nodeOrder, node))
          nodeOrder.push_back(node);
      }
    });
  }
  if (nodeOrder.empty())
    return failAndReturn(
        "structured-stage construction has 0 structured roots");

  llvm::DenseSet<mlir::Operation *> selectedStageAllocations;
  for (const CandidateSelectedDDRStage &stage : selectedDDRStages) {
    mlir::Operation *allocation = stage.buffer.getDefiningOp();
    if (!allocation || !mlir::isa<mlir::memref::AllocOp>(allocation))
      return failAndReturn(
          "selected DDR stage is not backed by one explicit allocation");
    selectedStageAllocations.insert(allocation);
  }

  llvm::DenseMap<uint32_t, uint32_t> componentParents;
  for (uint32_t node : nodeOrder)
    componentParents.try_emplace(node, node);
  std::function<uint32_t(uint32_t)> findComponent = [&](uint32_t node) {
    uint32_t parent = componentParents.lookup(node);
    if (parent == node)
      return node;
    uint32_t root = findComponent(parent);
    componentParents[node] = root;
    return root;
  };
  auto uniteComponents = [&](uint32_t lhs, uint32_t rhs) {
    lhs = findComponent(lhs);
    rhs = findComponent(rhs);
    if (lhs == rhs)
      return;
    if (rhs < lhs)
      std::swap(lhs, rhs);
    componentParents[rhs] = lhs;
  };

  // The selected DDR allocation is the graph cut. Every remaining current-SSA
  // producer dependency joins its structured owners into the same stage.
  for (mlir::Operation *topLevel : sourceOperations) {
    llvm::ArrayRef<uint32_t> consumers = operationNodes.lookup(topLevel);
    if (consumers.empty())
      continue;
    for (uint32_t consumer : consumers.drop_front())
      uniteComponents(consumers.front(), consumer);
    for (uint32_t consumer : consumers) {
      llvm::DenseSet<mlir::Operation *> visited;
      std::function<void(mlir::Operation *)> joinDependencies =
          [&](mlir::Operation *operation) {
            mlir::Operation *dependency = getTopLevelOperation(operation);
            if (!dependency || selectedStageAllocations.contains(dependency) ||
                !visited.insert(dependency).second)
              return;
            llvm::ArrayRef<uint32_t> producers =
                operationNodes.lookup(dependency);
            if (!producers.empty()) {
              for (uint32_t producer : producers)
                uniteComponents(consumer, producer);
              return;
            }
            for (mlir::Value operand : dependency->getOperands())
              if (mlir::Operation *definition = operand.getDefiningOp())
                joinDependencies(definition);
          };
      for (mlir::Value operand : topLevel->getOperands())
        if (mlir::Operation *definition = operand.getDefiningOp())
          joinDependencies(definition);
    }
  }

  struct StructuredStageClosure {
    llvm::SmallVector<uint32_t, 2> nodes;
    llvm::SmallVector<mlir::Operation *, 16> operations;
  };
  llvm::SmallVector<StructuredStageClosure, 8> rootClosures;
  llvm::DenseMap<uint32_t, unsigned> componentStages;
  for (uint32_t node : nodeOrder) {
    uint32_t component = findComponent(node);
    auto [position, inserted] =
        componentStages.try_emplace(component, rootClosures.size());
    if (inserted)
      rootClosures.emplace_back();
    rootClosures[position->second].nodes.push_back(node);
  }
  for (StructuredStageClosure &stage : rootClosures) {
    auto isStageNode = [&](uint32_t node) {
      return llvm::is_contained(stage.nodes, node);
    };
    llvm::DenseSet<mlir::Operation *> closure;
    std::function<mlir::LogicalResult(mlir::Operation *)> addDependencies;
    addDependencies = [&](mlir::Operation *operation) -> mlir::LogicalResult {
      mlir::Operation *topLevel = getTopLevelOperation(operation);
      if (!topLevel || selectedStageAllocations.contains(topLevel) ||
          closure.contains(topLevel))
        return mlir::success();
      for (uint32_t owner : operationNodes.lookup(topLevel))
        if (!isStageNode(owner)) {
          std::string detail;
          llvm::raw_string_ostream diagnostic(detail);
          diagnostic << "structured stage [";
          llvm::interleaveComma(stage.nodes, diagnostic);
          diagnostic << "] closure reaches root " << owner << " operation "
                     << topLevel->getName()
                     << " without crossing its selected DDR boundary";
          return fail(diagnostic.str());
        }
      closure.insert(topLevel);
      for (mlir::Value operand : topLevel->getOperands()) {
        mlir::Operation *definition = operand.getDefiningOp();
        if (definition && mlir::failed(addDependencies(definition)))
          return mlir::failure();
      }
      return mlir::success();
    };

    for (mlir::Operation *topLevel : sourceOperations)
      if (llvm::any_of(operationNodes.lookup(topLevel), isStageNode) &&
          mlir::failed(addDependencies(topLevel)))
        return mlir::failure();

    llvm::SmallVector<mlir::Operation *, 16> forwardWorklist;
    for (mlir::Operation *operation : closure)
      forwardWorklist.push_back(operation);
    while (!forwardWorklist.empty()) {
      mlir::Operation *operation = forwardWorklist.pop_back_val();
      for (mlir::Value result : operation->getResults()) {
        for (mlir::OpOperand &use : result.getUses()) {
          mlir::Operation *user = getTopLevelOperation(use.getOwner());
          if (!user || mlir::isa<mlir::func::ReturnOp>(user) ||
              selectedStageAllocations.contains(user) || closure.contains(user))
            continue;
          bool belongsToAnotherRoot =
              llvm::any_of(operationNodes.lookup(user),
                           [&](uint32_t owner) { return !isStageNode(owner); });
          if (belongsToAnotherRoot)
            continue;
          if (mlir::failed(addDependencies(user)))
            return mlir::failure();
          forwardWorklist.push_back(user);
        }
      }
    }

    for (mlir::Operation *operation : sourceOperations)
      if (closure.contains(operation))
        stage.operations.push_back(operation);
    if (stage.operations.empty())
      return failAndReturn("structured stage has no live construction closure");
  }

  struct MovementClosure {
    unsigned endpoint = 0;
    llvm::SmallVector<mlir::Operation *, 8> operations;
  };
  llvm::SmallVector<MovementClosure, 8> movementClosures;
  movementClosures.reserve(peerEndpoints.size());
  for (auto [endpointIndex, endpoint] : llvm::enumerate(peerEndpoints)) {
    llvm::DenseSet<mlir::Operation *> closure;
    std::function<mlir::LogicalResult(mlir::Operation *)> addDependencies;
    addDependencies = [&](mlir::Operation *operation) -> mlir::LogicalResult {
      mlir::Operation *topLevel = getTopLevelOperation(operation);
      if (!topLevel || selectedStageAllocations.contains(topLevel) ||
          closure.contains(topLevel))
        return mlir::success();
      if (!operationNodes.lookup(topLevel).empty())
        return mlir::success();
      closure.insert(topLevel);
      for (mlir::Value operand : topLevel->getOperands())
        if (mlir::Operation *definition = operand.getDefiningOp())
          if (mlir::failed(addDependencies(definition)))
            return mlir::failure();
      return mlir::success();
    };
    mlir::Operation *definition = endpoint.value.getDefiningOp();
    if (!definition || mlir::failed(addDependencies(definition)))
      return failAndReturn(
          "selected peer endpoint has no root-independent value closure");

    // A receive movement region also owns the exact rootless store chain that
    // persists the fragment into its selected DDR stage. The consumer root
    // itself is an explicit stop boundary.
    if (endpoint.kind == CandidatePeerEndpointKind::Receive) {
      llvm::SmallVector<mlir::Operation *, 8> worklist;
      for (mlir::Operation *operation : closure)
        worklist.push_back(operation);
      while (!worklist.empty()) {
        mlir::Operation *operation = worklist.pop_back_val();
        for (mlir::Value result : operation->getResults()) {
          for (mlir::OpOperand &use : result.getUses()) {
            mlir::Operation *user = getTopLevelOperation(use.getOwner());
            if (!user || mlir::isa<mlir::func::ReturnOp>(user) ||
                selectedStageAllocations.contains(user) ||
                closure.contains(user) || !operationNodes.lookup(user).empty())
              continue;
            if (mlir::failed(addDependencies(user)))
              return mlir::failure();
            worklist.push_back(user);
          }
        }
      }
    }

    MovementClosure movement;
    movement.endpoint = static_cast<unsigned>(endpointIndex);
    for (mlir::Operation *operation : sourceOperations)
      if (closure.contains(operation))
        movement.operations.push_back(operation);
    if (movement.operations.empty())
      return failAndReturn(
          "selected peer endpoint has an empty movement closure");
    movementClosures.push_back(std::move(movement));
  }

  const unsigned rootCount = rootClosures.size();
  const unsigned eventCount = rootCount + movementClosures.size();
  llvm::DenseMap<uint32_t, unsigned> rootEvents;
  for (auto [rootIndex, root] : llvm::enumerate(rootClosures))
    for (uint32_t node : root.nodes)
      rootEvents.try_emplace(node, static_cast<unsigned>(rootIndex));
  std::vector<llvm::SmallVector<unsigned, 4>> successors(eventCount);
  llvm::SmallVector<unsigned, 16> predecessorCounts(eventCount, 0);
  llvm::DenseSet<uint64_t> eventEdges;
  auto addEventEdge = [&](unsigned source, unsigned destination) {
    if (source == destination || source >= eventCount ||
        destination >= eventCount)
      return;
    uint64_t key = (static_cast<uint64_t>(source) << 32) | destination;
    if (!eventEdges.insert(key).second)
      return;
    successors[source].push_back(destination);
    ++predecessorCounts[destination];
  };
  for (unsigned rootIndex = 1; rootIndex < rootCount; ++rootIndex)
    addEventEdge(rootIndex - 1, rootIndex);

  llvm::SmallVector<unsigned, 8> endpointOrder;
  for (unsigned endpointIndex = 0; endpointIndex < peerEndpoints.size();
       ++endpointIndex)
    endpointOrder.push_back(endpointIndex);
  llvm::sort(endpointOrder, [&](unsigned lhsIndex, unsigned rhsIndex) {
    const CandidatePeerEndpoint &lhs = peerEndpoints[lhsIndex];
    const CandidatePeerEndpoint &rhs = peerEndpoints[rhsIndex];
    return std::tuple(lhs.consumerScheduleOrdinal, lhs.consumerOperand,
                      lhs.communicationId, lhs.payloadSlice,
                      static_cast<uint8_t>(lhs.kind), lhs.peer.getValue()) <
           std::tuple(rhs.consumerScheduleOrdinal, rhs.consumerOperand,
                      rhs.communicationId, rhs.payloadSlice,
                      static_cast<uint8_t>(rhs.kind), rhs.peer.getValue());
  });
  for (unsigned endpointOrdinal = 1; endpointOrdinal < endpointOrder.size();
       ++endpointOrdinal)
    addEventEdge(rootCount + endpointOrder[endpointOrdinal - 1],
                 rootCount + endpointOrder[endpointOrdinal]);
  for (unsigned endpointIndex = 0; endpointIndex < peerEndpoints.size();
       ++endpointIndex) {
    const CandidatePeerEndpoint &endpoint = peerEndpoints[endpointIndex];
    auto root = rootEvents.find(endpoint.structuredNodeId);
    if (root == rootEvents.end())
      return failAndReturn("selected peer endpoint names an unknown root");
    const unsigned endpointEvent = rootCount + endpointIndex;
    if (endpoint.kind == CandidatePeerEndpointKind::Send)
      addEventEdge(root->second, endpointEvent);
    else
      addEventEdge(endpointEvent, root->second);
  }

  llvm::SmallVector<unsigned, 16> eventOrder;
  llvm::BitVector emittedEvents(eventCount);
  while (eventOrder.size() != eventCount) {
    std::optional<unsigned> ready;
    for (unsigned event = 0; event < eventCount; ++event)
      if (!emittedEvents.test(event) && predecessorCounts[event] == 0) {
        ready = event;
        break;
      }
    if (!ready)
      return failAndReturn("structured roots and ordered peer movements form a "
                           "dependency cycle");
    emittedEvents.set(*ready);
    eventOrder.push_back(*ready);
    for (unsigned successor : successors[*ready])
      --predecessorCounts[successor];
  }

  llvm::SmallVector<mlir::Value, 8> sourceInputs;
  for (mlir::Value original : scope.getInputs()) {
    if (isElidableConstantBoundary(original))
      continue;
    mlir::FailureOr<mlir::Value> boundary =
        materializeDdrBoundary(original, original, /*readOnly=*/true, rewriter);
    if (mlir::failed(boundary))
      return mlir::failure();
    sourceInputs.push_back(*boundary);
  }

  llvm::SmallVector<mlir::Value, 4> currentOutputs;
  for (mlir::Value original : scope.getOutputs()) {
    mlir::FailureOr<mlir::Value> boundary = materializeDdrBoundary(
        original, original, /*readOnly=*/false, rewriter);
    if (mlir::failed(boundary))
      return mlir::failure();
    currentOutputs.push_back(*boundary);
  }

  llvm::SmallVector<mlir::Value, 8> stageBuffers;
  stageBuffers.reserve(selectedDDRStages.size());
  llvm::DenseSet<mlir::Value> seenStageBuffers;
  for (const CandidateSelectedDDRStage &stage : selectedDDRStages) {
    if (!seenStageBuffers.insert(stage.buffer).second)
      return failAndReturn("selected DDR stage buffer is duplicated");
    auto type = mlir::dyn_cast<mlir::MemRefType>(stage.buffer.getType());
    if (!type || !type.hasStaticShape() || !isWaferDDRMemRefType(type))
      return failAndReturn(
          "selected DDR stage requires one static Wafer DDR buffer");
    auto allocation =
        rewriter.create<mlir::memref::AllocOp>(stage.buffer.getLoc(), type);
    stageBuffers.push_back(allocation.getResult());
    if (emissionRelations)
      emissionRelations->selectedDDRStages.push_back(
          MaterializedSelectedDDRStage{allocation, stage.producerNode});
  }

  llvm::BitVector claimedEndpoints(peerEndpoints.size());
  TileRegionOp lastRegion;
  for (unsigned event : eventOrder) {
    const bool isRoot = event < rootCount;
    llvm::ArrayRef<mlir::Operation *> operations =
        isRoot
            ? llvm::ArrayRef<mlir::Operation *>(rootClosures[event].operations)
            : llvm::ArrayRef<mlir::Operation *>(
                  movementClosures[event - rootCount].operations);
    buffers.clear();
    scalarValues.clear();
    scalarAttrs.clear();
    tensorAttrs.clear();
    compilerOwnedBuffers.clear();
    externalBuffers.clear();
    writableExternalBuffers.clear();
    selectedDDRStageExternalBuffers.clear();
    externalOutputIndices.clear();
    directYieldBuffers.clear();
    fillInitScalars.clear();
    fillInitAttrs.clear();
    activeStructuredNodes.clear();

    llvm::SmallVector<mlir::Value, 16> regionInputs(sourceInputs.begin(),
                                                    sourceInputs.end());
    regionInputs.append(currentOutputs.begin(), currentOutputs.end());
    regionInputs.append(stageBuffers.begin(), stageBuffers.end());
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    for (mlir::Value output : currentOutputs)
      resultTypes.push_back(output.getType());

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto region = rewriter.create<TileRegionOp>(scope.getLoc(), resultTypes,
                                                regionInputs);
    mlir::Block *body = new mlir::Block();
    region.getBody().push_back(body);
    for (mlir::Value input : region.getInputs())
      body->addArgument(input.getType(), input.getLoc());
    rewriter.setInsertionPointToStart(body);

    unsigned argumentIndex = 0;
    const unsigned inputCount = scope.getInputCount();
    for (mlir::BlockArgument sourceArgument : scope.getBody().getArguments()) {
      const unsigned sourceIndex = sourceArgument.getArgNumber();
      if (sourceIndex < inputCount &&
          isElidableConstantBoundary(scope.getInputs()[sourceIndex])) {
        auto constant = scope.getInputs()[sourceIndex]
                            .getDefiningOp<mlir::arith::ConstantOp>();
        auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(sourceArgument.getType());
        if (tensorType)
          tensorAttrs[sourceArgument] = constant.getValue();
        else {
          mlir::Operation *cloned = rewriter.clone(*constant.getOperation());
          scalarValues[sourceArgument] = cloned->getResult(0);
          scalarAttrs[sourceArgument] = constant.getValue();
        }
        continue;
      }
      if (argumentIndex >= body->getNumArguments())
        return failAndReturn("root region boundary argument count mismatch");
      mlir::BlockArgument regionArgument = body->getArgument(argumentIndex++);
      if (mlir::isa<mlir::RankedTensorType>(sourceArgument.getType())) {
        externalBuffers[sourceArgument] = regionArgument;
        if (sourceIndex >= inputCount) {
          writableExternalBuffers.insert(sourceArgument);
          externalOutputIndices[sourceArgument] = sourceIndex - inputCount;
        }
      } else if (isScalarType(sourceArgument.getType())) {
        scalarValues[sourceArgument] = regionArgument;
      } else {
        return failAndReturn(
            "root region boundary is not a ranked tensor or scalar");
      }
    }
    for (auto [stageIndex, stage] : llvm::enumerate(selectedDDRStages)) {
      if (argumentIndex >= body->getNumArguments())
        return failAndReturn("root region omitted a selected DDR stage");
      mlir::BlockArgument regionArgument = body->getArgument(argumentIndex++);
      compilerOwnedBuffers[stage.buffer] = regionArgument;
      selectedDDRStageExternalBuffers.insert(regionArgument);
    }
    if (argumentIndex != body->getNumArguments())
      return failAndReturn("root region has an extra boundary argument");

    llvm::SmallVector<const CandidatePeerEndpoint *, 1> localEndpoints;
    if (!isRoot) {
      const unsigned endpointIndex =
          movementClosures[event - rootCount].endpoint;
      if (claimedEndpoints.test(endpointIndex))
        return failAndReturn(
            "one selected peer endpoint belongs to multiple movement regions");
      claimedEndpoints.set(endpointIndex);
      localEndpoints.push_back(&peerEndpoints[endpointIndex]);
    }
    size_t nextEndpoint = 0;
    auto emitReadyEndpoints = [&]() -> mlir::LogicalResult {
      while (nextEndpoint < localEndpoints.size()) {
        const CandidatePeerEndpoint *endpoint = localEndpoints[nextEndpoint];
        MemLayout layout = MemLayout::Tensor;
        if (!lookupAny(endpoint->value, layout) &&
            !externalBuffers.contains(endpoint->value) &&
            !tensorAttrs.contains(endpoint->value))
          break;
        if (mlir::failed(emitPeerEndpoint(*endpoint, rewriter)))
          return mlir::failure();
        ++nextEndpoint;
      }
      return mlir::success();
    };
    if (mlir::failed(emitReadyEndpoints()))
      return mlir::failure();
    for (mlir::Operation *operation : operations) {
      if (selectedStageAllocations.contains(operation))
        continue;
      if (mlir::failed(convertOp(operation, rewriter)) ||
          mlir::failed(emitReadyEndpoints()))
        return mlir::failure();
    }
    if (nextEndpoint != localEndpoints.size())
      return failAndReturn(
          "selected peer endpoint was not materialized in its movement region");

    llvm::SmallVector<mlir::Value, 4> yieldedOutputs;
    for (mlir::Value output : scope.getOutputs()) {
      auto found = externalBuffers.find(output);
      if (found == externalBuffers.end())
        return failAndReturn("root region lost an output DDR boundary");
      yieldedOutputs.push_back(found->second);
    }
    rewriter.create<TileYieldOp>(scope.getLoc(), yieldedOutputs);
    currentOutputs.assign(region.getResults().begin(),
                          region.getResults().end());
    lastRegion = region;
  }

  if (claimedEndpoints.count() != peerEndpoints.size()) {
    const int endpointIndex = claimedEndpoints.find_first_unset();
    const CandidatePeerEndpoint &endpoint = peerEndpoints[endpointIndex];
    std::string detail;
    llvm::raw_string_ostream diagnostic(detail);
    diagnostic << "selected peer endpoint has no structured root region owner"
               << " (communication=" << endpoint.communicationId
               << ", slice=" << endpoint.payloadSlice << ", kind="
               << (endpoint.kind == CandidatePeerEndpointKind::Send
                       ? "send"
                       : "receive");
    if (mlir::Operation *definition = endpoint.value.getDefiningOp())
      diagnostic << ", definition=" << definition->getName();
    if (!endpoint.value.use_empty()) {
      diagnostic << ", users=[";
      llvm::interleaveComma(
          endpoint.value.getUsers(), diagnostic,
          [&](mlir::Operation *user) { diagnostic << user->getName(); });
      diagnostic << ']';
    }
    diagnostic << ')';
    return failAndReturn(diagnostic.str());
  }
  return lastRegion;
}

} // namespace wafer::tensor_program_to_tile_region
