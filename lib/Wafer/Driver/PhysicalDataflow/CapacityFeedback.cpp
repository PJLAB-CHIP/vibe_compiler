//===- CapacityFeedback.cpp - Actual input access attribution ------------===//

#include "CapacityFeedback.h"

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <memory>
#include <optional>
#include <utility>

namespace wafer::compiler::detail {
namespace {

std::optional<int64_t> getProgramArgument(mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return std::nullopt;
  auto function =
      mlir::dyn_cast<mlir::func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.front())
    return std::nullopt;
  auto identity = function.getArgAttrOfType<ProgramArgumentAttr>(
      argument.getArgNumber(), kWaferProgramArgumentAttrName);
  return identity ? std::optional<int64_t>(identity.getIndex()) : std::nullopt;
}

std::optional<int64_t>
getTensorInput(mlir::Value value, InputCapacityFeedback &feedback,
               const llvm::DenseSet<mlir::Operation *> &fusedProducers) {
  llvm::DenseSet<mlir::Value> visited;
  while (visited.insert(value).second) {
    if (auto identity = getProgramArgument(value))
      return identity;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto region =
          mlir::dyn_cast<TileRegionOp>(argument.getOwner()->getParentOp());
      if (!region || argument.getArgNumber() >= region.getInputs().size())
        return std::nullopt;
      value = region.getInputs()[argument.getArgNumber()];
      continue;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return std::nullopt;
    if (auto producer =
            mlir::dyn_cast<mlir::linalg::LinalgOp>(result.getOwner());
        producer && fusedProducers.contains(producer) &&
        producer.hasPureTensorSemantics() && !producer.getNumReductionLoops()) {
      // Follow an already selected fused unary computation's input demand,
      // not its numeric value or storage identity. Equal permutation maps
      // prove that this step preserves the scope's indexing coordinates.
      auto outputMap = analysis::getStructuredResultMap(result);
      mlir::OpOperand *input = nullptr;
      for (mlir::OpOperand &operand : producer->getOpOperands()) {
        if (!producer.payloadUsesValueFromOperand(&operand))
          continue;
        if (input)
          return std::nullopt;
        input = &operand;
      }
      if (input && mlir::succeeded(outputMap) && outputMap->isPermutation() &&
          producer.getMatchingIndexingMap(input) == *outputMap) {
        value = input->get();
        continue;
      }
    }
    auto indexing = analysis::deriveTensorResultIndexing(result);
    if (indexing.status ==
        analysis::TensorResultIndexingStatus::BrokenContract) {
      feedback.status = CapacityFeedbackStatus::BrokenContract;
      feedback.detail = indexing.detail;
    }
    if (!indexing.isExact() || indexing.indexing->operands.size() != 1 ||
        indexing.indexing->operands.front().role !=
            TensorIndexingOperandRole::Source)
      return std::nullopt;
    value = result.getOwner()->getOperand(
        indexing.indexing->operands.front().operand);
  }
  return std::nullopt;
}

// A derived non-view producer can also read an input. Its consumer scope
// must participate in ambiguity detection even when no parameter map is
// available. Stop at another explicit scope, which owns that producer's
// accesses independently; this is only an SSA dependency walk, not an alias
// or numeric provenance proof through computation.
std::set<int64_t> getUnattributedInputs(
    mlir::Value value,
    const llvm::DenseSet<mlir::Operation *> &scopeBoundaries) {
  std::set<int64_t> inputs;
  llvm::SmallVector<mlir::Value> pending{value};
  llvm::DenseSet<mlir::Value> visited;
  while (!pending.empty()) {
    mlir::Value current = pending.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    if (auto input = getProgramArgument(current)) {
      inputs.insert(*input);
      continue;
    }
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      if (auto region =
              mlir::dyn_cast<TileRegionOp>(argument.getOwner()->getParentOp()))
        pending.push_back(region.getInputs()[argument.getArgNumber()]);
      continue;
    }
    auto *producer = current.getDefiningOp();
    if (!producer || mlir::isa<TileRegionOp>(producer) ||
        scopeBoundaries.contains(producer))
      continue;
    producer->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *nested) {
      // A separately materialized Region owns its input accesses. Its result
      // consumer must not be registered as another reader of all those inputs.
      if (mlir::isa<TileRegionOp>(nested) || scopeBoundaries.contains(nested))
        return mlir::WalkResult::skip();
      for (mlir::Value operand : nested->getOperands())
        if (mlir::isa<mlir::TensorType>(operand.getType()))
          pending.push_back(operand);
      return mlir::WalkResult::advance();
    });
  }
  return inputs;
}

/// One immutable current function. Only RDMA/GS writers can preserve the
/// external input origin; compute, opaque writes and receives remain unknown.
class InputOrigins {
public:
  explicit InputOrigins(mlir::func::FuncOp function) {
    function.walk([&](mlir::Operation *operation) {
      if (operation == function.getOperation() ||
          operation->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>())
        return;
      auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
      if (!effects) {
        if (!mlir::isMemoryEffectFree(operation)) {
          opaqueMemoryWrite = true;
          for (mlir::Value operand : operation->getOperands())
            markUnknown(operand);
        }
        return;
      }
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (const auto &effect : instances) {
        if (!mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect()))
          continue;
        if (mlir::Value value = effect.getValue()) {
          if (!mlir::isa<mlir::BaseMemRefType>(value.getType()))
            continue;
          for (mlir::Value root : roots.getStorageRoots(value)) {
            auto &accesses = writers[root];
            if (!llvm::is_contained(accesses, operation))
              accesses.push_back(operation);
          }
        } else if (!mlir::isa<WaferInstructionOpInterface>(operation)) {
          opaqueMemoryWrite |=
              effect.getResource() == mlir::SideEffects::DefaultResource::get();
          for (mlir::Value operand : operation->getOperands())
            markUnknown(operand);
        }
      }
    });
  }

  std::optional<int64_t> get(mlir::Value value) {
    if (opaqueMemoryWrite)
      return std::nullopt;
    std::optional<int64_t> identity;
    const auto &storage = roots.getStorageRoots(value);
    if (storage.empty())
      return std::nullopt;
    for (mlir::Value root : storage) {
      auto origin = getRoot(root);
      if (!origin || (identity && identity != origin))
        return std::nullopt;
      identity = origin;
    }
    return identity;
  }

private:
  void markUnknown(mlir::Value value) {
    if (mlir::isa<mlir::BaseMemRefType>(value.getType()))
      for (mlir::Value root : roots.getStorageRoots(value))
        unknown.insert(root);
  }

  std::optional<int64_t> getRoot(mlir::Value root) {
    if (unknown.contains(root))
      return std::nullopt;
    if (auto found = origins.find(root); found != origins.end())
      return found->second;
    if (!active.insert(root).second)
      return std::nullopt;
    auto finish = [&](std::optional<int64_t> value) {
      active.erase(root);
      origins.try_emplace(root, value);
      return value;
    };
    auto found = writers.find(root);
    if (auto argument = getProgramArgument(root))
      return finish(found == writers.end() ? argument : std::nullopt);
    if (found == writers.end() || found->second.empty())
      return finish(std::nullopt);
    std::optional<int64_t> identity;
    for (mlir::Operation *writer : found->second) {
      mlir::Value source;
      if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(writer))
        source = rdma.getSource();
      else if (auto copy = mlir::dyn_cast<InstrGatherScatterOp>(writer))
        source = copy.getSource();
      if (!source)
        return finish(std::nullopt);
      auto origin = get(source);
      if (!origin || (identity && identity != origin))
        return finish(std::nullopt);
      identity = origin;
    }
    return finish(identity);
  }

  StorageRootMemo roots;
  bool opaqueMemoryWrite = false;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation *, 2>> writers;
  llvm::DenseSet<mlir::Value> unknown, active;
  llvm::DenseMap<mlir::Value, std::optional<int64_t>> origins;
};

} // namespace

InputCapacityFeedback
deriveInputCapacityFeedback(CardId card, TileId tile,
                            const SPMMemoryPlanningFailure &failure,
                            llvm::ArrayRef<const TemporalDomain *> domains,
                            llvm::ArrayRef<TemporalChoice> choices) {
  InputCapacityFeedback result;
  if (domains.size() != choices.size()) {
    result.status = CapacityFeedbackStatus::BrokenContract;
    result.detail = "capacity feedback has mismatched temporal domains";
    return result;
  }
  using Scope = std::pair<size_t, size_t>;
  struct InputAccess {
    std::set<TemporalCoordinate> coordinates;
    bool complete = true;
  };
  std::map<int64_t, std::map<Scope, InputAccess>> inputs;
  llvm::DenseSet<mlir::Operation *> scopeBoundaries;
  for (const auto &choice : choices)
    for (const auto &scope : choice.scopes)
      scopeBoundaries.insert(scope.operation);
  for (auto [domainIndex, domain] : llvm::enumerate(domains)) {
    auto owner = domain->getRegion()->getParentOfType<TileModuleOp>();
    if (!owner || owner.getCardIdAttr().getInt() != card.getValue() ||
        owner.getTileIdAttr().getInt() != tile.getValue())
      continue;
    const auto &choice = choices[domainIndex];
    llvm::DenseSet<mlir::Operation *> fusedProducers;
    if (choice.kind == TemporalTraversalKind::Joint)
      for (const auto &fusion : domain->getFusions())
        fusedProducers.insert(fusion.producer.getOwner());
    auto descriptors = domain->getScopeDescriptors(choice.kind);
    if (descriptors.size() != choice.scopes.size()) {
      result.status = CapacityFeedbackStatus::BrokenContract;
      result.detail = "capacity feedback has mismatched temporal scopes";
      return result;
    }
    for (auto [scopeIndex, scope] : llvm::enumerate(choice.scopes))
      for (mlir::OpOperand &operand : scope.operation->getOpOperands()) {
        auto input = getTensorInput(operand.get(), result, fusedProducers);
        if (result.status == CapacityFeedbackStatus::BrokenContract)
          return result;
        if (!input) {
          for (int64_t dependency :
               getUnattributedInputs(operand.get(), scopeBoundaries))
            inputs[dependency][{domainIndex, scopeIndex}].complete = false;
          continue;
        }
        // Retain the whole reader set. An unsupported reader must not make
        // the analyzable subset appear to be a complete access relation.
        auto &access = inputs[*input][{domainIndex, scopeIndex}];
        auto map = analysis::getStructuredOperandMap(operand);
        if (mlir::failed(map)) {
          access.complete = false;
          continue;
        }
        auto simplified = mlir::simplifyAffineMap(*map);
        for (auto [iterator, capability] :
             llvm::enumerate(descriptors[scopeIndex].iteratorCapabilities))
          if (capability == IteratorTilingCapability::Tileable &&
              descriptors[scopeIndex].iterationExtents[iterator] > 1 &&
              llvm::any_of(simplified.getResults(), [&](mlir::AffineExpr expr) {
                return expr.isFunctionOfDim(iterator);
              }))
            access.coordinates.insert({domainIndex, scopeIndex, iterator});
      }
  }
  llvm::DenseMap<mlir::Operation *, std::unique_ptr<InputOrigins>> origins;
  auto observe = [&](const SPMMemoryPlanningFailure::DemandEvidence &demand) {
    mlir::Value allocation = demand.allocation;
    auto function =
        allocation.getParentRegion()->getParentOfType<mlir::func::FuncOp>();
    if (!function) {
      ++result.unavailableDemands;
      return;
    }
    auto &index = origins[function.getOperation()];
    if (!index)
      index = std::make_unique<InputOrigins>(function);
    auto input = index->get(demand.allocation);
    auto found = input ? inputs.find(*input) : inputs.end();
    if (found == inputs.end()) {
      ++result.unavailableDemands;
      return;
    }
    if (llvm::any_of(found->second, [](const auto &reader) {
          return !reader.second.complete;
        })) {
      ++result.ambiguousInputs;
      return;
    }
    // A unique immutable input can have several fully known readers. These
    // coordinates are a proposal association, never a unique buffer owner or
    // a claim that changing any one reader makes the allocation fit.
    result.sharedInputDemands += found->second.size() > 1;
    for (const auto &[scope, access] : found->second)
      result.coordinates.insert(access.coordinates.begin(),
                                access.coordinates.end());
  };
  for (const auto &demand : failure.individuallyOversizedDemands)
    observe(demand);
  for (const auto &demand : failure.capacityConflictDemands)
    observe(demand);
  return result;
}

} // namespace wafer::compiler::detail
