//===- BoundaryMovement.cpp - Close physical Tile boundaries ----------===//

#include "BoundaryMovement.h"

#include "StructuredToTile.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace wafer::compiler::detail {
namespace {

struct InputPlan {
  unsigned index = 0;
  mlir::Value originalOperand;
  mlir::BlockArgument originalArgument;
  llvm::SmallVector<mlir::bufferization::ToMemrefOp, 2> bridges;
  std::optional<unsigned> peerRelation;
  bool unused = false;
};

struct ResultPlan {
  unsigned index = 0;
  mlir::OpResult originalResult;
  mlir::bufferization::ToTensorOp bridge;
  mlir::Value spmValue;
  mlir::Value ddrDestination;
  mlir::bufferization::ToMemrefOp outputBridge;
  mlir::memref::CopyOp outputCopy;
  llvm::SmallVector<unsigned, 2> peerRelations;
  bool hasSameTileConsumer = false;
  bool functionReturn = false;
};

struct RegionPlan {
  TileRegionOp operation;
  TileModuleOp tileModule;
  llvm::SmallVector<InputPlan, 8> inputs;
  llvm::SmallVector<ResultPlan, 8> results;
};

struct PeerPlan {
  unsigned relationIndex = 0;
  TileRegionOp sourceRegion;
  unsigned sourceResult = 0;
  mlir::Value sourceSPM;
  TileRegionOp destinationRegion;
  unsigned destinationInput = 0;
  mlir::MemRefType destinationSPMType;
  uint64_t sourceTile = 0;
  uint64_t destinationTile = 0;
  uint64_t bytes = 0;
};

static BoundaryMovementResult fail(BoundaryMovementFailureKind kind,
                                   llvm::StringRef detail) {
  BoundaryMovementResult result;
  result.failure = kind;
  result.detail = detail.str();
  return result;
}

static bool collectSPMBridges(
    mlir::BlockArgument argument,
    llvm::SmallVectorImpl<mlir::bufferization::ToMemrefOp> &bridges) {
  for (mlir::Operation *user : argument.getUsers()) {
    auto current = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(user);
    if (!current || !isWaferSPMMemRefType(current.getMemref().getType()))
      return false;
    bridges.push_back(current);
  }
  return true;
}

static mlir::bufferization::ToMemrefOp
getOnlySPMBridge(mlir::BlockArgument argument) {
  llvm::SmallVector<mlir::bufferization::ToMemrefOp, 2> bridges;
  if (!collectSPMBridges(argument, bridges) || bridges.size() != 1)
    return {};
  return bridges.front();
}

static mlir::bufferization::ToTensorOp getYieldBridge(TileRegionOp region,
                                                      unsigned resultIndex) {
  if (!region || region.getBody().empty())
    return {};
  auto yield =
      mlir::dyn_cast<TileYieldOp>(region.getBody().front().getTerminator());
  if (!yield || resultIndex >= yield.getValues().size())
    return {};
  auto bridge = yield.getValues()[resultIndex]
                    .getDefiningOp<mlir::bufferization::ToTensorOp>();
  if (!bridge || !isWaferSPMMemRefType(bridge.getMemref().getType()))
    return {};
  return bridge;
}

static mlir::MemRefType getDDRType(mlir::MemRefType spmType) {
  return mlir::MemRefType::get(spmType.getShape(), spmType.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               MemoryAttr::get(spmType.getContext(),
                                               MemorySpace::DDR,
                                               MemLayout::Tensor));
}

static TileRegionOp getRegionOwner(mlir::Value value) {
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
    return mlir::dyn_cast<TileRegionOp>(result.getOwner());
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  return argument && argument.getOwner()
             ? mlir::dyn_cast_or_null<TileRegionOp>(
                   argument.getOwner()->getParentOp())
             : TileRegionOp{};
}

static bool logicalTypesMatch(mlir::MemRefType lhs, mlir::MemRefType rhs) {
  return lhs && rhs && lhs.getShape() == rhs.getShape() &&
         lhs.getElementType() == rhs.getElementType();
}

static std::optional<MemLayout> getLayout(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  MemoryAttr memory = memref ? getWaferMemoryAttr(memref) : MemoryAttr{};
  return memory ? std::optional<MemLayout>(memory.getLayout()) : std::nullopt;
}

static mlir::LogicalResult
preflight(mlir::ModuleOp module, StructuredMaterializationRelations &relations,
          llvm::SmallVectorImpl<RegionPlan> &regions,
          llvm::SmallVectorImpl<PeerPlan> &peers, std::string &detail) {
  llvm::DenseMap<mlir::Value, llvm::SmallVector<unsigned, 2>> peerSource;
  llvm::DenseMap<mlir::Value, unsigned> peerDestination;
  for (auto [index, relation] : llvm::enumerate(relations.boundaryRelations)) {
    peerSource[relation.sourceEndpoint].push_back(index);
    if (!peerDestination.try_emplace(relation.destinationEndpoint, index)
             .second) {
      detail = "one Tile boundary destination has multiple producers";
      return mlir::failure();
    }
  }

  llvm::DenseSet<unsigned> matchedOutputs;
  llvm::DenseMap<mlir::Operation *, unsigned> regionIndex;
  module.walk([&](TileRegionOp region) {
    regionIndex.try_emplace(region, regions.size());
    RegionPlan plan;
    plan.operation = region;
    plan.tileModule = region->getParentOfType<TileModuleOp>();
    regions.push_back(std::move(plan));
  });
  if (regions.empty()) {
    detail = "physical boundary closure found no TileRegion";
    return mlir::failure();
  }

  for (RegionPlan &plan : regions) {
    TileRegionOp region = plan.operation;
    if (!plan.tileModule || !region.getBody().hasOneBlock()) {
      detail = "TileRegion has no physical Tile owner or single-block body";
      return mlir::failure();
    }
    mlir::Block &block = region.getBody().front();
    for (auto [index, inputAndArgument] :
         llvm::enumerate(llvm::zip(region.getInputs(), block.getArguments()))) {
      mlir::Value input = std::get<0>(inputAndArgument);
      mlir::BlockArgument argument = std::get<1>(inputAndArgument);
      if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
        continue;
      llvm::SmallVector<mlir::bufferization::ToMemrefOp, 2> bridges;
      if (!collectSPMBridges(argument, bridges)) {
        detail = "logical TileRegion input has a non-buffer bridge user";
        return mlir::failure();
      }
      auto peer = peerDestination.find(argument);
      if (bridges.empty() && peer != peerDestination.end()) {
        detail = "cross-Tile destination has no current SPM consumer";
        return mlir::failure();
      }
      if (bridges.size() > 1 && peer != peerDestination.end()) {
        detail = "cross-Tile destination requires one explicit receive buffer";
        return mlir::failure();
      }
      InputPlan inputPlan;
      inputPlan.index = index;
      inputPlan.originalOperand = input;
      inputPlan.originalArgument = argument;
      inputPlan.bridges = std::move(bridges);
      inputPlan.unused = inputPlan.bridges.empty();
      if (peer != peerDestination.end())
        inputPlan.peerRelation = peer->second;
      plan.inputs.push_back(std::move(inputPlan));
    }

    for (auto [index, result] : llvm::enumerate(region.getResults())) {
      if (!mlir::isa<mlir::RankedTensorType>(result.getType()))
        continue;
      mlir::bufferization::ToTensorOp bridge = getYieldBridge(region, index);
      if (!bridge) {
        detail = "logical TileRegion result has no current SPM yield bridge";
        return mlir::failure();
      }
      ResultPlan resultPlan;
      resultPlan.index = index;
      resultPlan.originalResult = result;
      resultPlan.bridge = bridge;
      resultPlan.spmValue = bridge.getMemref();
      if (auto found = peerSource.find(result); found != peerSource.end())
        resultPlan.peerRelations.append(found->second.begin(),
                                        found->second.end());

      for (mlir::Operation *user : result.getUsers()) {
        if (mlir::isa<TileRegionOp>(user)) {
          resultPlan.hasSameTileConsumer = true;
          continue;
        }
        auto toMemref = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(user);
        if (!toMemref) {
          detail = "logical TileRegion result has an unknown boundary user";
          return mlir::failure();
        }
        if (isWaferDDRMemRefType(toMemref.getMemref().getType())) {
          if (resultPlan.outputBridge) {
            detail = "TileRegion result has multiple boundary bridges";
            return mlir::failure();
          }
          if (!llvm::all_of(toMemref.getMemref().getUsers(),
                            [](mlir::Operation *bridgeUser) {
                              return mlir::isa<mlir::func::ReturnOp>(
                                  bridgeUser);
                            })) {
            detail = "DDR TileRegion bridge is not an entry return";
            return mlir::failure();
          }
          resultPlan.outputBridge = toMemref;
          resultPlan.functionReturn = true;
          continue;
        }
        for (mlir::Operation *bridgeUser : toMemref.getMemref().getUsers()) {
          auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(bridgeUser);
          if (!copy || copy.getSource() != toMemref.getMemref()) {
            detail = "TileRegion output bridge has a non-publication user";
            return mlir::failure();
          }
          bool matched = false;
          for (auto [outputIndex, output] :
               llvm::enumerate(relations.structuralOutputs)) {
            if (copy.getTarget() != output.endpoint)
              continue;
            if (resultPlan.ddrDestination ||
                !matchedOutputs.insert(outputIndex).second) {
              detail = "observable output relation is duplicated";
              return mlir::failure();
            }
            resultPlan.ddrDestination = copy.getTarget();
            resultPlan.outputBridge = toMemref;
            resultPlan.outputCopy = copy;
            matched = true;
          }
          if (!matched) {
            detail = "TileRegion publication copy has no output relation";
            return mlir::failure();
          }
        }
      }
      plan.results.push_back(std::move(resultPlan));
    }
    for (const ResultPlan &result : plan.results) {
      if (!result.functionReturn)
        continue;
      bool covered = llvm::any_of(plan.results, [&](const ResultPlan &other) {
        return other.spmValue == result.spmValue && other.ddrDestination;
      });
      if (!covered) {
        detail = "entry return has no matching observable output destination";
        return mlir::failure();
      }
    }
  }
  if (matchedOutputs.size() != relations.structuralOutputs.size()) {
    detail = "not every observable output has one current publication copy";
    return mlir::failure();
  }

  for (auto [relationIndex, relation] :
       llvm::enumerate(relations.boundaryRelations)) {
    auto source = mlir::dyn_cast<mlir::OpResult>(relation.sourceEndpoint);
    auto destination =
        mlir::dyn_cast<mlir::BlockArgument>(relation.destinationEndpoint);
    TileRegionOp sourceRegion = getRegionOwner(relation.sourceEndpoint);
    TileRegionOp destinationRegion =
        getRegionOwner(relation.destinationEndpoint);
    if (!source || !destination || !sourceRegion || !destinationRegion ||
        sourceRegion == destinationRegion ||
        source.getResultNumber() >= sourceRegion.getNumResults() ||
        destination.getArgNumber() >= destinationRegion.getInputs().size()) {
      detail = "cross-Tile relation does not name current Region endpoints";
      return mlir::failure();
    }
    mlir::bufferization::ToTensorOp sourceBridge =
        getYieldBridge(sourceRegion, source.getResultNumber());
    mlir::bufferization::ToMemrefOp destinationBridge =
        getOnlySPMBridge(destination);
    auto sourceTile = sourceRegion->getParentOfType<TileModuleOp>();
    auto destinationTile = destinationRegion->getParentOfType<TileModuleOp>();
    mlir::MemRefType sourceType =
        sourceBridge
            ? mlir::cast<mlir::MemRefType>(sourceBridge.getMemref().getType())
            : mlir::MemRefType{};
    mlir::MemRefType destinationType =
        destinationBridge ? mlir::cast<mlir::MemRefType>(
                                destinationBridge.getMemref().getType())
                          : mlir::MemRefType{};
    if (!sourceBridge || !destinationBridge || !sourceTile ||
        !destinationTile || sourceTile == destinationTile ||
        !logicalTypesMatch(sourceType, destinationType)) {
      detail = "cross-Tile relation has no exact typed peer payload";
      return mlir::failure();
    }
    std::optional<WaferPhysicalTensorInfo> destinationPhysical =
        computeWaferPhysicalTensorInfo(destinationType);
    if (!destinationPhysical || destinationPhysical->physicalBytes <= 0) {
      detail = "cross-Tile relation has no exact physical payload bytes";
      return mlir::failure();
    }
    peers.push_back(PeerPlan{
        static_cast<unsigned>(relationIndex), sourceRegion,
        source.getResultNumber(), sourceBridge.getMemref(), destinationRegion,
        destination.getArgNumber(), destinationType,
        static_cast<uint64_t>(sourceTile.getTileIdAttr().getInt()),
        static_cast<uint64_t>(destinationTile.getTileIdAttr().getInt()),
        static_cast<uint64_t>(destinationPhysical->physicalBytes)});
  }
  return mlir::success();
}

static const PeerPlan *findPeer(llvm::ArrayRef<PeerPlan> peers,
                                unsigned relationIndex) {
  auto found = llvm::find_if(peers, [&](const PeerPlan &peer) {
    return peer.relationIndex == relationIndex;
  });
  return found == peers.end() ? nullptr : &*found;
}

static mlir::FailureOr<mlir::Value>
resolveDDRInput(mlir::Value original,
                const llvm::DenseMap<mlir::Value, mlir::Value> &stagedResults) {
  if (auto mapped = stagedResults.find(original); mapped != stagedResults.end())
    return mapped->second;
  if (auto bridge = original.getDefiningOp<mlir::bufferization::ToTensorOp>()) {
    if (isWaferDDRMemRefType(bridge.getMemref().getType()))
      return bridge.getMemref();
  }
  if (isWaferDDRMemRefType(original.getType()))
    return original;
  return mlir::failure();
}

static mlir::LogicalResult moveOutputDestinationBefore(mlir::Value destination,
                                                       TileRegionOp region) {
  mlir::Operation *definition = destination.getDefiningOp();
  if (!definition)
    return mlir::success();
  if (definition->getBlock() != region->getBlock())
    return mlir::failure();
  if (definition->isBeforeInBlock(region))
    return mlir::success();
  auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(definition);
  auto allStatic = [](llvm::ArrayRef<mlir::OpFoldResult> values) {
    return llvm::all_of(values, [](mlir::OpFoldResult value) {
      return mlir::getConstantIntValue(value).has_value();
    });
  };
  if (!subview || !allStatic(subview.getMixedOffsets()) ||
      !allStatic(subview.getMixedSizes()) ||
      !allStatic(subview.getMixedStrides()))
    return mlir::failure();
  mlir::DominanceInfo dominance(region->getParentOfType<mlir::ModuleOp>());
  for (mlir::Value operand : subview->getOperands())
    if (!dominance.dominates(operand, region))
      return mlir::failure();
  definition->moveBefore(region);
  return mlir::success();
}

static void eraseDeadBridges(mlir::ModuleOp module) {
  bool changed = true;
  mlir::IRRewriter rewriter(module.getContext());
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Operation *, 16> dead;
    module.walk([&](mlir::Operation *operation) {
      if (mlir::isa<mlir::bufferization::ToTensorOp,
                    mlir::bufferization::ToMemrefOp>(operation) &&
          operation->use_empty())
        dead.push_back(operation);
    });
    for (mlir::Operation *operation : llvm::reverse(dead)) {
      if (!operation->use_empty())
        continue;
      rewriter.eraseOp(operation);
      changed = true;
    }
  }
}

static mlir::LogicalResult apply(mlir::ModuleOp module,
                                 llvm::MutableArrayRef<RegionPlan> regions,
                                 llvm::ArrayRef<PeerPlan> peers,
                                 BoundaryMovementStatistics &statistics) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::DenseMap<mlir::Value, mlir::Value> stagedResults;
  llvm::SmallVector<TileRegionOp, 16> oldRegions;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
      returnOperands;
  llvm::DenseMap<mlir::Operation *, llvm::DenseSet<unsigned>> functionResults;

  for (RegionPlan &plan : regions) {
    TileRegionOp oldRegion = plan.operation;
    oldRegions.push_back(oldRegion);
    llvm::DenseMap<unsigned, mlir::Value> resultDestinations;
    for (ResultPlan &result : plan.results) {
      if (result.ddrDestination) {
        if (mlir::failed(
                moveOutputDestinationBefore(result.ddrDestination, oldRegion)))
          return mlir::failure();
        resultDestinations.try_emplace(result.index, result.ddrDestination);
        stagedResults.try_emplace(result.originalResult, result.ddrDestination);
        continue;
      }
      if (!result.hasSameTileConsumer)
        continue;
      rewriter.setInsertionPoint(oldRegion);
      auto staging = rewriter.create<mlir::memref::AllocOp>(
          oldRegion.getLoc(),
          getDDRType(mlir::cast<mlir::MemRefType>(result.spmValue.getType())));
      resultDestinations.try_emplace(result.index, staging.getResult());
      stagedResults.try_emplace(result.originalResult, staging.getResult());
      ++statistics.interRegionDDRStages;
    }

    llvm::SmallVector<mlir::Value, 12> newInputs;
    llvm::DenseMap<unsigned, const InputPlan *> inputPlans;
    for (InputPlan &input : plan.inputs) {
      inputPlans.try_emplace(input.index, &input);
    }
    for (auto [index, input] : llvm::enumerate(oldRegion.getInputs())) {
      if (!mlir::isa<mlir::RankedTensorType>(input.getType())) {
        newInputs.push_back(input);
        continue;
      }
      const InputPlan *inputPlan = inputPlans.lookup(index);
      if (!inputPlan)
        return mlir::failure();
      if (inputPlan->peerRelation || inputPlan->unused)
        continue;
      mlir::FailureOr<mlir::Value> ddr =
          resolveDDRInput(inputPlan->originalOperand, stagedResults);
      if (mlir::failed(ddr))
        return mlir::failure();
      newInputs.push_back(*ddr);
    }

    llvm::SmallVector<unsigned, 8> destinationResultIndices;
    for (const ResultPlan &result : plan.results) {
      auto destination = resultDestinations.find(result.index);
      if (destination == resultDestinations.end())
        continue;
      newInputs.push_back(destination->second);
      destinationResultIndices.push_back(result.index);
    }
    llvm::SmallVector<mlir::Type, 4> scalarResultTypes;
    for (mlir::Value result : oldRegion.getResults())
      if (!mlir::isa<mlir::RankedTensorType>(result.getType()))
        scalarResultTypes.push_back(result.getType());

    rewriter.setInsertionPoint(oldRegion);
    auto newRegion = rewriter.create<TileRegionOp>(
        oldRegion.getLoc(), scalarResultTypes, newInputs);
    newRegion.getBody().takeBody(oldRegion.getBody());
    mlir::Block &block = newRegion.getBody().front();
    llvm::DenseMap<unsigned, mlir::BlockArgument> destinationArguments;
    for (unsigned resultIndex : destinationResultIndices) {
      mlir::Value destination = resultDestinations.lookup(resultIndex);
      mlir::BlockArgument argument =
          block.addArgument(destination.getType(), destination.getLoc());
      destinationArguments.try_emplace(resultIndex, argument);
    }

    llvm::SmallVector<unsigned, 8> erasedArguments;
    for (InputPlan &input : plan.inputs) {
      mlir::BlockArgument argument = block.getArgument(input.index);
      if (input.unused) {
        erasedArguments.push_back(input.index);
        continue;
      }
      if (input.peerRelation) {
        const PeerPlan *peer = findPeer(peers, *input.peerRelation);
        if (!peer || input.bridges.size() != 1)
          return mlir::failure();
        rewriter.setInsertionPointToStart(&block);
        auto allocation = rewriter.create<mlir::memref::AllocOp>(
            argument.getLoc(), peer->destinationSPMType);
        auto receive = rewriter.create<CommPeerRecvOp>(
            argument.getLoc(), allocation.getResult(), peer->sourceTile,
            peer->bytes,
            DTEMessageAttr::get(rewriter.getContext(), peer->relationIndex, 0,
                                0));
        rewriter.create<mlir::async::AwaitOp>(argument.getLoc(),
                                              receive.getToken());
        erasedArguments.push_back(input.index);
        ++statistics.peerReceives;
        ++statistics.peerWaits;
        rewriter.replaceAllUsesWith(input.bridges.front().getMemref(),
                                    allocation.getResult());
        rewriter.eraseOp(input.bridges.front());
        ++statistics.tensorBridgesRemoved;
      } else {
        mlir::FailureOr<mlir::Value> ddr =
            resolveDDRInput(input.originalOperand, stagedResults);
        if (mlir::failed(ddr))
          return mlir::failure();
        argument.setType((*ddr).getType());
        for (mlir::bufferization::ToMemrefOp bridge : input.bridges) {
          rewriter.setInsertionPointToStart(&block);
          auto allocation = rewriter.create<mlir::memref::AllocOp>(
              argument.getLoc(),
              mlir::cast<mlir::MemRefType>(bridge.getMemref().getType()));
          rewriter.create<StorageLoadOp>(argument.getLoc(), argument,
                                         allocation.getResult());
          rewriter.replaceAllUsesWith(bridge.getMemref(),
                                      allocation.getResult());
          rewriter.eraseOp(bridge);
          ++statistics.ddrLoads;
          ++statistics.tensorBridgesRemoved;
        }
      }
    }
    llvm::sort(erasedArguments, std::greater<unsigned>());
    for (unsigned index : erasedArguments)
      block.eraseArgument(index);

    auto yield = mlir::cast<TileYieldOp>(block.getTerminator());
    rewriter.setInsertionPoint(yield);
    llvm::SmallVector<mlir::Value, 8> sendTokens;
    for (const ResultPlan &result : plan.results) {
      for (unsigned relationIndex : result.peerRelations) {
        const PeerPlan *peer = findPeer(peers, relationIndex);
        if (!peer)
          return mlir::failure();
        mlir::Value source = result.spmValue;
        if (source.getType() != peer->destinationSPMType) {
          if (!logicalTypesMatch(mlir::cast<mlir::MemRefType>(source.getType()),
                                 peer->destinationSPMType) ||
              getLayout(source.getType()) ==
                  getLayout(peer->destinationSPMType))
            return mlir::failure();
          source =
              rewriter
                  .create<LayoutMaterializeOp>(result.originalResult.getLoc(),
                                               peer->destinationSPMType, source)
                  .getResult();
        }
        auto send = rewriter.create<CommPeerSendOp>(
            result.originalResult.getLoc(), source, peer->destinationTile,
            peer->bytes,
            DTEMessageAttr::get(rewriter.getContext(), relationIndex, 0, 0));
        sendTokens.push_back(send.getToken());
        ++statistics.peerSends;
      }
    }
    for (const ResultPlan &result : plan.results) {
      auto destination = destinationArguments.find(result.index);
      if (destination == destinationArguments.end())
        continue;
      rewriter.create<StorageStoreOp>(result.originalResult.getLoc(),
                                      result.spmValue, destination->second);
      ++statistics.ddrStores;
    }
    for (mlir::Value token : sendTokens) {
      rewriter.create<mlir::async::AwaitOp>(yield.getLoc(), token);
      ++statistics.peerWaits;
    }
    llvm::SmallVector<mlir::Value, 4> scalarYields;
    for (auto [index, value] : llvm::enumerate(yield.getValues()))
      if (!mlir::isa<mlir::RankedTensorType>(
              oldRegion.getResult(index).getType()))
        scalarYields.push_back(value);
    rewriter.modifyOpInPlace(
        yield, [&] { yield.getValuesMutable().assign(scalarYields); });

    unsigned scalarResult = 0;
    for (mlir::Value oldResult : oldRegion.getResults()) {
      if (mlir::isa<mlir::RankedTensorType>(oldResult.getType()))
        continue;
      rewriter.replaceAllUsesWith(oldResult,
                                  newRegion.getResult(scalarResult++));
    }
    for (ResultPlan &result : plan.results) {
      if (result.outputCopy) {
        rewriter.eraseOp(result.outputCopy);
        ++statistics.outputCopiesRemoved;
      }
      if (result.functionReturn) {
        for (mlir::OpOperand &use : result.outputBridge.getMemref().getUses()) {
          auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(use.getOwner());
          auto function = returnOp
                              ? returnOp->getParentOfType<mlir::func::FuncOp>()
                              : mlir::func::FuncOp{};
          if (!returnOp || !function)
            return mlir::failure();
          returnOperands[returnOp.getOperation()].push_back(
              use.getOperandNumber());
          functionResults[function.getOperation()].insert(
              use.getOperandNumber());
        }
      }
      if (result.outputBridge && result.outputBridge.getMemref().use_empty()) {
        rewriter.eraseOp(result.outputBridge);
        ++statistics.tensorBridgesRemoved;
      }
      if (result.bridge && result.bridge->getResult(0).use_empty()) {
        rewriter.eraseOp(result.bridge);
        ++statistics.tensorBridgesRemoved;
      }
    }
  }

  for (auto &[operation, indices] : returnOperands) {
    llvm::sort(indices, std::greater<unsigned>());
    for (unsigned index : indices)
      operation->eraseOperands(index, 1);
  }
  for (auto &[operation, indices] : functionResults) {
    auto function = mlir::cast<mlir::func::FuncOp>(operation);
    llvm::SmallVector<mlir::Type, 4> results;
    for (auto [index, type] :
         llvm::enumerate(function.getFunctionType().getResults()))
      if (!indices.contains(index))
        results.push_back(type);
    function.setType(mlir::FunctionType::get(
        function.getContext(), function.getFunctionType().getInputs(),
        results));
  }
  eraseDeadBridges(module);
  for (TileRegionOp region : llvm::reverse(oldRegions)) {
    if (!region->use_empty())
      return mlir::failure();
    rewriter.eraseOp(region);
  }
  eraseDeadBridges(module);
  return mlir::success();
}

} // namespace

mlir::LogicalResult verifyPhysicalTileDataflow(mlir::ModuleOp module) {
  if (!module || mlir::failed(mlir::verify(module)) ||
      mlir::failed(verifyTileRegionStorageBoundaries(module)))
    return mlir::failure();
  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::linalg::LinalgOp, mlir::bufferization::ToTensorOp,
                  mlir::bufferization::ToMemrefOp>(operation) ||
        (mlir::isa<mlir::memref::CopyOp>(operation) &&
         !operation->getParentOfType<TileRegionOp>())) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    auto region = mlir::dyn_cast<TileRegionOp>(operation);
    if (!region)
      return mlir::WalkResult::advance();
    for (mlir::Value input : region.getInputs())
      if (mlir::isa<mlir::ShapedType>(input.getType()) &&
          !isWaferDDRMemRefType(input.getType())) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    for (mlir::Type type : region.getResultTypes())
      if (mlir::isa<mlir::ShapedType>(type) && !isWaferDDRMemRefType(type)) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    return mlir::WalkResult::advance();
  });
  return mlir::success(!illegal);
}

BoundaryMovementResult
materializeTileBoundaryMovement(mlir::ModuleOp module,
                                StructuredMaterializationRelations &relations) {
  if (!module || mlir::failed(verifyStructuredComputeLowered(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations)))
    return fail(BoundaryMovementFailureKind::BrokenContract,
                "boundary movement requires structured-compute-lowered "
                "current IR and live endpoint relations");
  llvm::SmallVector<RegionPlan, 16> regions;
  llvm::SmallVector<PeerPlan, 16> peers;
  std::string detail;
  if (mlir::failed(preflight(module, relations, regions, peers, detail)))
    return fail(BoundaryMovementFailureKind::Unsupported, detail);

  BoundaryMovementResult result;
  if (mlir::failed(apply(module, regions, peers, result.statistics)))
    return fail(BoundaryMovementFailureKind::CompilerFailure,
                "preflighted boundary movement failed while rewriting "
                "current IR");
  relations.boundaryRelations.clear();
  relations.structuralOutputs.clear();
  rebuildCurrentBufferOwnerRelations(module, relations);
  if (mlir::failed(verifyPhysicalTileDataflow(module)) ||
      mlir::failed(checkStructuredBufferRelationsCurrent(module, relations)))
    return fail(BoundaryMovementFailureKind::CompilerFailure,
                "boundary movement produced invalid physical Tile IR or "
                "buffer relations");
  return result;
}

} // namespace wafer::compiler::detail
