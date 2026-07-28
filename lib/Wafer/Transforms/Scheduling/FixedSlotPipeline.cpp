//===- FixedSlotPipeline.cpp - Static fixed-slot pipelining -------------===//

#include "Wafer/Transforms/SoftwarePipelining.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
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
#include <vector>

namespace wafer {
namespace {

struct DependencyEdge {
  unsigned predecessor = 0;
  bool advancesStage = false;
};

struct BufferAccess {
  mlir::Value root;
  bool write = false;
  unsigned operation = 0;
};

struct AllocationPlan {
  mlir::memref::AllocOp allocation;
  unsigned firstStage = 0;
  unsigned lastStage = 0;
  unsigned slotCount = 0;
};

struct FixedSlotPipelinePlan {
  llvm::SmallVector<mlir::Operation *, 16> operations;
  llvm::DenseMap<mlir::Operation *, unsigned> operationIndices;
  llvm::DenseMap<mlir::Operation *, InstrFamily> instructionFamilies;
  llvm::DenseMap<mlir::Operation *, NCCWorker> instructionWorkers;
  llvm::SmallVector<llvm::SmallVector<DependencyEdge, 4>, 16> predecessors;
  llvm::DenseMap<mlir::Operation *, unsigned> stages;
  llvm::SmallVector<AllocationPlan, 4> allocations;
  uint64_t tripCount = 0;
  unsigned maxStage = 0;
  unsigned slotAllocationCount = 0;
};

static constexpr llvm::StringLiteral kPointerPermutationMarker =
    "wafer.fixed_slot_pointer_permutation";

static mlir::FailureOr<FixedSlotPipelinePlan>
failPlan(std::string *failureReason, llvm::Twine message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

static mlir::FailureOr<StaticFixedSlotPipelineCandidate>
failCandidate(std::string *failureReason, llvm::Twine message) {
  if (failureReason)
    *failureReason = message.str();
  return mlir::failure();
}

static std::optional<uint64_t>
getPositiveStaticTripCount(mlir::scf::ForOp loop) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *lower >= *upper)
    return std::nullopt;
  __int128 span = static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
  // The pinned SCF pipeliner computes `upper - lower` in int64_t before
  // divideCeilSigned. Keep its precondition explicit so an extreme but valid
  // index interval cannot trigger signed overflow inside the utility.
  if (span > static_cast<__int128>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  __int128 count =
      (span + static_cast<__int128>(*step) - 1) / static_cast<__int128>(*step);
  if (count <= 0 ||
      count > static_cast<__int128>(std::numeric_limits<uint64_t>::max()))
    return std::nullopt;
  return static_cast<uint64_t>(count);
}

static bool isNestedInFor(mlir::scf::ForOp loop) {
  for (mlir::Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp())
    if (mlir::isa<mlir::scf::ForOp>(parent))
      return true;
  return false;
}

static bool hasPhysicalPlacementFacts(mlir::ModuleOp module) {
  bool placed = false;
  module.walk([&](mlir::memref::AllocOp allocation) {
    placed |= allocation->hasAttr(kWaferSPMOffsetAttrName) ||
              allocation->hasAttr(kWaferDDROffsetAttrName);
  });
  return placed;
}

static mlir::FailureOr<mlir::Value> getStaticAliasRoot(mlir::Value value,
                                                       mlir::scf::ForOp loop) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto type = mlir::dyn_cast<mlir::BaseMemRefType>(value.getType());
    if (!type)
      return mlir::failure();
    if (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
            value.getDefiningOp())) {
      value = view.getViewSource();
      continue;
    }
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArg.getOwner();
      mlir::Operation *parent = owner ? owner->getParentOp() : nullptr;
      if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(parent)) {
        unsigned index = blockArg.getArgNumber();
        if (owner == &tileRegion.getBody().front() &&
            index < tileRegion.getInputs().size()) {
          value = tileRegion.getInputs()[index];
          continue;
        }
      }
      if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
        if (owner == forOp.getBody() && blockArg.getArgNumber() > 0) {
          unsigned index = blockArg.getArgNumber() - 1;
          auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
              forOp.getBody()->getTerminator());
          if (index < forOp.getInitArgs().size() && yield &&
              index < yield.getResults().size() &&
              yield.getResults()[index] == blockArg) {
            value = forOp.getInitArgs()[index];
            continue;
          }
        }
      }
      return value;
    }
    if (value.getDefiningOp<mlir::memref::AllocOp>())
      return value;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !loop->isAncestor(definition))
      return value;
    return mlir::failure();
  }
  return mlir::failure();
}

static bool isFunctionEntryArgument(mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || !argument.getOwner())
    return false;
  auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      argument.getOwner()->getParentOp());
  return function && argument.getOwner() == &function.getBody().front();
}

static bool isFunctionEntryBackedMemref(mlir::Value value) {
  if (isFunctionEntryArgument(value))
    return true;
  auto toMemref = value.getDefiningOp<mlir::bufferization::ToMemrefOp>();
  return toMemref && isFunctionEntryArgument(toMemref.getTensor());
}

static bool areProvenDistinct(mlir::Value lhs, mlir::Value rhs,
                              mlir::scf::ForOp loop) {
  if (lhs == rhs)
    return false;
  auto lhsAllocation = lhs.getDefiningOp<mlir::memref::AllocOp>();
  auto rhsAllocation = rhs.getDefiningOp<mlir::memref::AllocOp>();
  // Two distinct allocation roots are disjoint. Do not generalize one fresh
  // allocation against an arbitrary SSA producer: an unrecognized producer
  // may itself be an alias of that allocation.
  if (lhsAllocation && rhsAllocation)
    return true;
  // An allocation dynamically created in the selected loop cannot alias a
  // value available before that allocation in the same iteration. Likewise a
  // function argument predates every memref.alloc in the function. Unknown
  // SSA producers remain unproven because they may return an alias.
  if ((lhsAllocation && loop->isAncestor(lhsAllocation.getOperation())) ||
      (rhsAllocation && loop->isAncestor(rhsAllocation.getOperation())) ||
      (lhsAllocation && isFunctionEntryBackedMemref(rhs)) ||
      (rhsAllocation && isFunctionEntryBackedMemref(lhs)))
    return true;

  auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
  if (!lhsType || !rhsType)
    return false;
  MemoryAttr lhsMemory = getWaferMemoryAttr(lhsType);
  MemoryAttr rhsMemory = getWaferMemoryAttr(rhsType);
  return lhsMemory && rhsMemory && lhsMemory.getSpace() != rhsMemory.getSpace();
}

static bool isSupportedPureBodyOperation(mlir::Operation *operation) {
  return operation->getNumRegions() == 0 && mlir::isMemoryEffectFree(operation);
}

static bool isStaticLoopAllocation(mlir::Operation *operation) {
  auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation);
  return allocation && allocation.getDynamicSizes().empty() &&
         allocation.getSymbolOperands().empty() &&
         allocation.getType().hasStaticShape();
}

static void addDependency(FixedSlotPipelinePlan &plan, unsigned predecessor,
                          unsigned successor, bool advancesStage) {
  if (predecessor == successor)
    return;
  auto &edges = plan.predecessors[successor];
  auto found = llvm::find_if(edges, [&](const DependencyEdge &edge) {
    return edge.predecessor == predecessor;
  });
  if (found == edges.end()) {
    edges.push_back({predecessor, advancesStage});
    return;
  }
  found->advancesStage |= advancesStage;
}

static bool isCrossEngineDependency(const FixedSlotPipelinePlan &plan,
                                    unsigned predecessor, unsigned successor) {
  auto predecessorFamily =
      plan.instructionFamilies.find(plan.operations[predecessor]);
  auto successorFamily =
      plan.instructionFamilies.find(plan.operations[successor]);
  return predecessorFamily != plan.instructionFamilies.end() &&
         successorFamily != plan.instructionFamilies.end() &&
         predecessorFamily->second != successorFamily->second;
}

static mlir::LogicalResult collectInstructionAccesses(
    mlir::Operation *operation, unsigned operationIndex, mlir::scf::ForOp loop,
    llvm::SmallVectorImpl<BufferAccess> &accesses, std::string *failureReason) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects) {
    if (failureReason)
      *failureReason =
          "fixed-slot instruction has no typed memory-effect interface";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);

  // Engine occupancy is owned by the typed instruction family/worker. Coarse
  // SPM/DDR resource effects are accepted only when a value-associated effect
  // of the same kind and address space witnesses the actual storage root.
  // Unknown/rootless storage must never disappear from the dependency DAG.
  for (const auto &instance : instances) {
    if (instance.getValue())
      continue;
    mlir::SideEffects::Resource *resource = instance.getResource();
    if (resource == WaferComputeResource::get() ||
        resource == WaferMovementResource::get())
      continue;

    std::optional<MemorySpace> memorySpace;
    if (resource == WaferSPMResource::get())
      memorySpace = MemorySpace::SPM;
    else if (resource == WaferDDRResource::get())
      memorySpace = MemorySpace::DDR;
    if (!memorySpace) {
      if (failureReason)
        *failureReason =
            "fixed-slot instruction has an unsupported rootless resource "
            "effect";
      return mlir::failure();
    }

    bool rootlessRead =
        mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    bool rootlessWrite =
        mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
    if (!rootlessRead && !rootlessWrite) {
      if (failureReason)
        *failureReason =
            "fixed-slot instruction has an unsupported rootless storage "
            "effect";
      return mlir::failure();
    }
    bool witnessed = llvm::any_of(instances, [&](const auto &candidate) {
      mlir::Value value = candidate.getValue();
      auto type =
          value ? mlir::dyn_cast<mlir::MemRefType>(value.getType()) : nullptr;
      MemoryAttr memory = type ? getWaferMemoryAttr(type) : MemoryAttr{};
      if (!memory || memory.getSpace() != *memorySpace)
        return false;
      return (rootlessRead &&
              mlir::isa<mlir::MemoryEffects::Read>(candidate.getEffect())) ||
             (rootlessWrite &&
              mlir::isa<mlir::MemoryEffects::Write>(candidate.getEffect()));
    });
    if (!witnessed) {
      if (failureReason)
        *failureReason =
            "fixed-slot instruction has an unwitnessed rootless storage "
            "effect";
      return mlir::failure();
    }
  }

  for (const auto &instance : instances) {
    if (mlir::isa<mlir::MemoryEffects::Allocate>(instance.getEffect()))
      continue;
    mlir::Value value = instance.getValue();
    if (!value)
      continue;
    mlir::FailureOr<mlir::Value> root = getStaticAliasRoot(value, loop);
    if (mlir::failed(root)) {
      if (failureReason)
        *failureReason =
            "fixed-slot instruction has an unsupported alias producer";
      return mlir::failure();
    }
    bool read = mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    bool write = mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect()) ||
                 mlir::isa<mlir::MemoryEffects::Free>(instance.getEffect());
    if (!read && !write) {
      if (failureReason)
        *failureReason =
            "fixed-slot instruction has an unsupported memory effect";
      return mlir::failure();
    }
    auto found = llvm::find_if(accesses, [&](const BufferAccess &access) {
      return access.operation == operationIndex && access.root == *root;
    });
    if (found == accesses.end())
      accesses.push_back({*root, write, operationIndex});
    else
      found->write |= write;
  }
  return mlir::success();
}

static mlir::FailureOr<FixedSlotPipelinePlan>
buildFixedSlotPipelinePlan(mlir::scf::ForOp loop, std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (!loop || !loop.getBody() || isNestedInFor(loop))
    return failPlan(failureReason,
                    "fixed-slot candidate requires a non-nested scf.for");
  if (!loop.getRegion().hasOneBlock())
    return failPlan(failureReason,
                    "fixed-slot candidate requires a single-block scf.for");

  std::optional<uint64_t> tripCount = getPositiveStaticTripCount(loop);
  if (!tripCount)
    return failPlan(
        failureReason,
        "fixed-slot candidate requires a static positive trip count and step");

  FixedSlotPipelinePlan plan;
  plan.tripCount = *tripCount;
  llvm::SmallVector<mlir::memref::AllocOp, 4> allocations;
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions() != 0)
      return failPlan(
          failureReason,
          "fixed-slot candidate rejects nested regions in the loop body");
    if (mlir::isa<mlir::memref::AllocOp>(operation)) {
      if (!isStaticLoopAllocation(&operation))
        return failPlan(
            failureReason,
            "fixed-slot candidate requires static loop-local allocations");
      allocations.push_back(mlir::cast<mlir::memref::AllocOp>(operation));
      continue;
    }

    unsigned index = static_cast<unsigned>(plan.operations.size());
    plan.operationIndices.try_emplace(&operation, index);
    plan.operations.push_back(&operation);

    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(&operation);
    if (!instruction) {
      if (!isSupportedPureBodyOperation(&operation))
        return failPlan(
            failureReason,
            "fixed-slot candidate encountered an unknown effectful operation");
      continue;
    }
    if (instruction.getInstructionFamily() == InstrFamily::DTE)
      return failPlan(failureReason,
                      "fixed-slot candidate does not yet support Direct DTE");
    NCCCompletionContract contract = getNCCCompletionContract(&operation);
    if (contract.behavior != LocalInstructionCompletion::OrderedPending ||
        !contract.issueWorker)
      return failPlan(
          failureReason,
          "fixed-slot candidate rejects completion and synchronous islands");
    plan.instructionFamilies.try_emplace(&operation,
                                         instruction.getInstructionFamily());
    plan.instructionWorkers.try_emplace(&operation, *contract.issueWorker);
  }
  if (plan.operations.empty() || plan.instructionFamilies.size() < 2)
    return failPlan(
        failureReason,
        "fixed-slot candidate requires at least two typed instructions");
  plan.predecessors.resize(plan.operations.size());

  // Preserve all direct SSA definitions. Memory dependencies below add the
  // value-associated RAW/WAR/WAW edges that instruction ops express through
  // effects rather than SSA results.
  for (auto [successor, operation] : llvm::enumerate(plan.operations)) {
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      auto found = plan.operationIndices.find(definition);
      if (found == plan.operationIndices.end())
        continue;
      addDependency(plan, found->second, static_cast<unsigned>(successor),
                    isCrossEngineDependency(plan, found->second,
                                            static_cast<unsigned>(successor)));
    }
  }

  llvm::SmallVector<BufferAccess, 16> accesses;
  for (auto [index, operation] : llvm::enumerate(plan.operations)) {
    if (!plan.instructionFamilies.contains(operation))
      continue;
    llvm::SmallVector<BufferAccess, 8> current;
    if (mlir::failed(collectInstructionAccesses(operation,
                                                static_cast<unsigned>(index),
                                                loop, current, failureReason)))
      return mlir::failure();
    for (const BufferAccess &access : current) {
      for (const BufferAccess &prior : accesses) {
        if (!access.write && !prior.write)
          continue;
        if (access.root != prior.root &&
            !areProvenDistinct(access.root, prior.root, loop))
          return failPlan(
              failureReason,
              "fixed-slot candidate cannot prove two accessed roots distinct");
        if (areProvenDistinct(access.root, prior.root, loop))
          continue;

        mlir::Operation *priorOperation = plan.operations[prior.operation];
        auto priorWorker = plan.instructionWorkers.find(priorOperation);
        auto currentWorker = plan.instructionWorkers.find(operation);
        if (priorWorker == plan.instructionWorkers.end() ||
            currentWorker == plan.instructionWorkers.end() ||
            priorWorker->second != currentWorker->second)
          return failPlan(
              failureReason,
              "fixed-slot candidate rejects a cross-worker memory hazard");
        addDependency(plan, prior.operation, static_cast<unsigned>(index),
                      isCrossEngineDependency(plan, prior.operation,
                                              static_cast<unsigned>(index)));
      }
    }
    // Operand effects on one typed instruction describe one atomic issue.
    // Compare its normalized roots only with earlier issues; the operation's
    // own verifier/interface owns source/destination alias legality.
    accesses.append(current);
  }

  // Original block order is already a topological order for direct SSA and
  // effect dependencies. A cross-engine dependency advances one iteration
  // stage; same-engine ordering remains in the same stage.
  for (auto [index, operation] : llvm::enumerate(plan.operations)) {
    unsigned stage = 0;
    for (const DependencyEdge &edge :
         plan.predecessors[static_cast<unsigned>(index)]) {
      mlir::Operation *predecessor = plan.operations[edge.predecessor];
      auto found = plan.stages.find(predecessor);
      if (found == plan.stages.end())
        return failPlan(failureReason,
                        "fixed-slot dependency DAG is not topological");
      stage = std::max(stage, found->second + (edge.advancesStage ? 1U : 0U));
    }
    plan.stages.try_emplace(operation, stage);
    plan.maxStage = std::max(plan.maxStage, stage);
  }
  if (plan.maxStage == 0)
    return failPlan(
        failureReason,
        "fixed-slot candidate has no proven cross-engine pipeline stage");
  if (plan.tripCount < static_cast<uint64_t>(plan.maxStage) + 1)
    return failPlan(
        failureReason,
        "fixed-slot trip count is smaller than the derived stage count");

  // Loop-external roots are not renamed by this transform. If a written root
  // is touched in more than one stage, the steady kernel can reverse an
  // original loop-carried RAW/WAR/WAW order (for example, stage 1 of
  // iteration i+1 can issue before stage 2 of iteration i). Same-stage
  // accesses retain source and iteration order, and read-only roots have no
  // such carried hazard. Until an address/range analysis proves iterations
  // disjoint, reject the cross-stage case rather than relying on the
  // busytable to serialize an already incorrect issue order.
  struct ExternalRootStageSummary {
    bool hasWrite = false;
    std::optional<unsigned> firstStage;
    bool crossesStage = false;
  };
  llvm::DenseMap<mlir::Value, ExternalRootStageSummary> externalRoots;
  for (const BufferAccess &access : accesses) {
    auto allocation = access.root.getDefiningOp<mlir::memref::AllocOp>();
    if (allocation && allocation->getParentOp() == loop.getOperation())
      continue;
    unsigned stage = plan.stages.lookup(plan.operations[access.operation]);
    ExternalRootStageSummary &summary = externalRoots[access.root];
    summary.hasWrite |= access.write;
    if (!summary.firstStage)
      summary.firstStage = stage;
    else
      summary.crossesStage |= *summary.firstStage != stage;
  }
  for (const auto &[root, summary] : externalRoots) {
    (void)root;
    if (summary.hasWrite && summary.crossesStage)
      return failPlan(
          failureReason,
          "fixed-slot candidate rejects a loop-external cross-stage write "
          "hazard without iteration-disjoint ranges");
  }

  llvm::DenseMap<mlir::Operation *, std::pair<unsigned, unsigned>>
      allocationStageSpans;
  for (const BufferAccess &access : accesses) {
    auto allocation = access.root.getDefiningOp<mlir::memref::AllocOp>();
    if (!allocation || allocation->getParentOp() != loop.getOperation())
      continue;
    unsigned stage = plan.stages.lookup(plan.operations[access.operation]);
    auto [found, inserted] = allocationStageSpans.try_emplace(
        allocation.getOperation(), std::make_pair(stage, stage));
    if (!inserted) {
      found->second.first = std::min(found->second.first, stage);
      found->second.second = std::max(found->second.second, stage);
    }
  }

  for (mlir::memref::AllocOp allocation : allocations) {
    auto span = allocationStageSpans.find(allocation.getOperation());
    if (span == allocationStageSpans.end())
      return failPlan(
          failureReason,
          "fixed-slot loop allocation has no proven instruction lifetime");
    unsigned slotCount = span->second.second - span->second.first + 1;
    if (slotCount == 0 || plan.slotAllocationCount >
                              std::numeric_limits<unsigned>::max() - slotCount)
      return failPlan(failureReason,
                      "fixed-slot allocation count is not representable");
    plan.allocations.push_back(
        {allocation, span->second.first, span->second.second, slotCount});
    plan.slotAllocationCount += slotCount;
  }
  if (plan.allocations.empty() || plan.slotAllocationCount < 2)
    return failPlan(
        failureReason,
        "fixed-slot candidate did not derive a real multi-buffer allocation");

  // The SCF utility supports only distance-zero/one recurrences whose yielded
  // value is defined in the loop or outside it. Reject unsupported original
  // recurrence before cloning; slot rotation is added below with explicit
  // stage-zero pointer identities, while slot count separately represents the
  // allocation's cross-stage memory lifetime.
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getNumRegionIterArgs())
    return failPlan(failureReason,
                    "fixed-slot candidate has an invalid scf.for recurrence");
  for (mlir::Value value : yield.getOperands()) {
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition) {
      auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
      bool supportedPassThrough =
          blockArg && blockArg.getOwner() == loop.getBody() &&
          blockArg.getArgNumber() > 0 &&
          mlir::isa<mlir::BaseMemRefType>(blockArg.getType());
      if (!supportedPassThrough)
        return failPlan(
            failureReason,
            "fixed-slot candidate rejects unsupported pass-through loop "
            "recurrence");
      continue;
    }
    if (definition && loop->isAncestor(definition) &&
        !plan.operationIndices.contains(definition))
      return failPlan(
          failureReason,
          "fixed-slot candidate has an unsupported recurrence definition");
  }
  return plan;
}

static mlir::FailureOr<mlir::scf::ForOp>
materializeSlotsAndRotation(mlir::scf::ForOp loop,
                            const FixedSlotPipelinePlan &plan,
                            std::string *failureReason) {
  mlir::IRRewriter rewriter(loop.getContext());
  rewriter.setInsertionPoint(loop);

  llvm::SmallVector<mlir::Value, 8> initArgs(loop.getInitArgs().begin(),
                                             loop.getInitArgs().end());
  for (const AllocationPlan &allocationPlan : plan.allocations) {
    mlir::memref::AllocOp sourceAllocation = allocationPlan.allocation;
    for (unsigned index = 0; index < allocationPlan.slotCount; ++index) {
      mlir::Operation *clone = rewriter.clone(*sourceAllocation.getOperation());
      auto clonedAllocation = mlir::cast<mlir::memref::AllocOp>(clone);
      initArgs.push_back(clonedAllocation.getResult());
    }
  }

  auto preparedLoop = rewriter.create<mlir::scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      initArgs);
  mlir::Block *preparedBody = preparedLoop.getBody();
  if (!preparedBody->empty() &&
      mlir::isa<mlir::scf::YieldOp>(preparedBody->back()))
    rewriter.eraseOp(&preparedBody->back());
  rewriter.setInsertionPointToStart(preparedBody);

  mlir::IRMapping mapping;
  mapping.map(loop.getInductionVar(), preparedLoop.getInductionVar());
  mapping.map(
      loop.getRegionIterArgs(),
      preparedLoop.getRegionIterArgs().take_front(loop.getNumRegionIterArgs()));

  unsigned slotArgument = loop.getNumRegionIterArgs();
  for (const AllocationPlan &allocationPlan : plan.allocations) {
    mlir::memref::AllocOp sourceAllocation = allocationPlan.allocation;
    mapping.map(sourceAllocation.getResult(),
                preparedLoop.getRegionIterArgs()[slotArgument]);
    slotArgument += allocationPlan.slotCount;
  }

  std::vector<std::pair<mlir::Operation *, unsigned>> schedule;
  schedule.reserve(plan.operations.size() + plan.slotAllocationCount);
  for (mlir::Operation *operation : plan.operations) {
    mlir::Operation *clone = rewriter.clone(*operation, mapping);
    schedule.emplace_back(clone, plan.stages.lookup(operation));
  }

  std::vector<std::pair<mlir::Operation *, unsigned>> pointerPermutations;
  pointerPermutations.reserve(loop.getNumRegionIterArgs() +
                              plan.slotAllocationCount);
  auto originalYield =
      mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  llvm::SmallVector<mlir::Value, 8> yieldValues;
  yieldValues.reserve(originalYield.getNumOperands() +
                      plan.slotAllocationCount);
  for (mlir::Value value : originalYield.getOperands()) {
    mlir::Value mapped = mapping.lookupOrDefault(value);
    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (blockArg && blockArg.getOwner() == loop.getBody() &&
        blockArg.getArgNumber() > 0) {
      auto identity = rewriter.create<mlir::memref::CastOp>(
          loop.getLoc(), mapped.getType(), mapped);
      identity->setAttr(kPointerPermutationMarker, rewriter.getUnitAttr());
      yieldValues.push_back(identity.getResult());
      pointerPermutations.emplace_back(identity.getOperation(), 0);
      continue;
    }
    yieldValues.push_back(mapped);
  }

  slotArgument = loop.getNumRegionIterArgs();
  for (const AllocationPlan &allocationPlan : plan.allocations) {
    llvm::SmallVector<mlir::Value, 4> rotated;
    rotated.reserve(allocationPlan.slotCount);
    for (unsigned index = 0; index < allocationPlan.slotCount; ++index) {
      unsigned sourceIndex = (index + 1) % allocationPlan.slotCount;
      mlir::Value source =
          preparedLoop.getRegionIterArgs()[slotArgument + sourceIndex];
      auto identity = rewriter.create<mlir::memref::CastOp>(
          loop.getLoc(), source.getType(), source);
      identity->setAttr(kPointerPermutationMarker, rewriter.getUnitAttr());
      rotated.push_back(identity.getResult());
      // This is a pointer permutation, not the buffer's final memory use.
      // Slot count encodes reuse distance; stage zero keeps the distance-one
      // SCF recurrence valid for slot families spanning three or more stages.
      pointerPermutations.emplace_back(identity.getOperation(), 0);
    }
    yieldValues.append(rotated);
    slotArgument += allocationPlan.slotCount;
  }
  rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), yieldValues);

  for (auto [oldResult, newResult] :
       llvm::zip(loop.getResults(),
                 preparedLoop.getResults().take_front(loop.getNumResults())))
    oldResult.replaceAllUsesWith(newResult);
  rewriter.eraseOp(loop);

  // Rotation definitions produce the values observed through next
  // iteration's region arguments. SCF's cyclic schedule verifier therefore
  // needs them before every current-iteration consumer in operation order.
  // They are pure pointer permutations at stage zero; buffer memory lifetime
  // is represented separately by the number of rotated slots. The remaining
  // operations retain the already topological source order.
  schedule.insert(schedule.begin(), pointerPermutations.begin(),
                  pointerPermutations.end());
  mlir::scf::PipeliningOption options;
  options.getScheduleFn =
      [schedule = std::move(schedule)](
          mlir::scf::ForOp,
          std::vector<std::pair<mlir::Operation *, unsigned>> &result) {
        result = schedule;
      };
  options.supportDynamicLoops = false;
  options.peelEpilogue = true;

  rewriter.setInsertionPoint(preparedLoop);
  bool modifiedIR = false;
  mlir::FailureOr<mlir::scf::ForOp> pipelined =
      mlir::scf::pipelineForLoop(rewriter, preparedLoop, options, &modifiedIR);
  if (mlir::failed(pipelined)) {
    if (failureReason)
      *failureReason = modifiedIR
                           ? "SCF pipelining failed after mutating the private "
                             "candidate clone"
                           : "SCF pipelining rejected the verified schedule";
    return mlir::failure();
  }
  return *pipelined;
}

static void eraseSynthesizedPointerPermutations(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::memref::CastOp, 8> identities;
  module.walk([&](mlir::memref::CastOp cast) {
    if (!cast->hasAttr(kPointerPermutationMarker))
      return;
    cast->removeAttr(kPointerPermutationMarker);
    if (cast.getSource().getType() == cast.getResult().getType())
      identities.push_back(cast);
  });
  for (mlir::memref::CastOp identity : identities) {
    identity.getResult().replaceAllUsesWith(identity.getSource());
    identity.erase();
  }
}

/// The pinned SCF utility rebuilds a static kernel upper bound as
/// `upper - maxStage * step`. Canonicalize exactly that mechanical expression;
/// a whole-module canonicalizer here would be too broad because this transform
/// must preserve unrelated source operations and pre-existing casts.
static mlir::LogicalResult canonicalizePipelinedKernelUpperBound(
    mlir::scf::ForOp loop, int64_t originalUpper, int64_t originalStep,
    unsigned maxStage) {
  __int128 upper = static_cast<__int128>(originalUpper) -
                   static_cast<__int128>(originalStep) * maxStage;
  if (upper < std::numeric_limits<int64_t>::min() ||
      upper > std::numeric_limits<int64_t>::max())
    return mlir::failure();

  mlir::Value mechanicalUpper = loop.getUpperBound();
  mlir::OpBuilder builder(loop);
  auto canonicalUpper = builder.create<mlir::arith::ConstantIndexOp>(
      loop.getLoc(), static_cast<int64_t>(upper));
  loop.setUpperBound(canonicalUpper);

  // Only erase the two operation kinds emitted for this bound by the pinned
  // utility. Original loop bounds are direct constants at this transform's
  // admission boundary and therefore cannot be consumed by this cleanup.
  llvm::SmallVector<mlir::Value, 2> worklist{mechanicalUpper};
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    mlir::Operation *operation = value.getDefiningOp();
    if (!operation || !operation->use_empty() ||
        !mlir::isa<mlir::arith::SubIOp, mlir::arith::MulIOp>(operation))
      continue;
    worklist.append(operation->operand_begin(), operation->operand_end());
    operation->erase();
  }
  return mlir::success();
}

} // namespace

mlir::FailureOr<StaticFixedSlotPipelineCandidate>
deriveStaticFixedSlotPipelineCandidate(mlir::ModuleOp sourceModule,
                                       mlir::scf::ForOp sourceLoop,
                                       std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (!sourceModule || !sourceLoop ||
      !sourceModule->isAncestor(sourceLoop.getOperation()))
    return failCandidate(
        failureReason,
        "fixed-slot candidate loop is not owned by the source module");
  if (hasPhysicalPlacementFacts(sourceModule))
    return failCandidate(
        failureReason,
        "fixed-slot candidate requires unplaced input without physical "
        "SPM/DDR offset facts");

  std::optional<uint64_t> sourceTripCount =
      getPositiveStaticTripCount(sourceLoop);
  if (sourceTripCount && *sourceTripCount == 1) {
    if (isNestedInFor(sourceLoop) || !sourceLoop.getRegion().hasOneBlock())
      return failCandidate(
          failureReason,
          "fixed-slot identity requires a non-nested single-block scf.for");
    mlir::IRMapping identityMapping;
    mlir::OwningOpRef<mlir::ModuleOp> identity(
        mlir::cast<mlir::ModuleOp>(sourceModule->clone(identityMapping)));
    StaticFixedSlotPipelineCandidate result;
    result.module = std::move(identity);
    result.stageCount = 1;
    result.slotAllocationCount = 0;
    return result;
  }

  if (mlir::failed(buildFixedSlotPipelinePlan(sourceLoop, failureReason)))
    return mlir::failure();

  mlir::IRMapping cloneMapping;
  mlir::Operation *clonedOperation = sourceModule->clone(cloneMapping);
  mlir::OwningOpRef<mlir::ModuleOp> candidate(
      mlir::cast<mlir::ModuleOp>(clonedOperation));
  auto clonedLoop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
      cloneMapping.lookupOrNull(sourceLoop.getOperation()));
  if (!clonedLoop)
    return failCandidate(
        failureReason,
        "fixed-slot candidate clone did not preserve the selected loop");

  mlir::FailureOr<FixedSlotPipelinePlan> clonedPlan =
      buildFixedSlotPipelinePlan(clonedLoop, failureReason);
  if (mlir::failed(clonedPlan))
    return mlir::failure();
  std::optional<int64_t> clonedUpper =
      mlir::getConstantIntValue(clonedLoop.getUpperBound());
  std::optional<int64_t> clonedStep =
      mlir::getConstantIntValue(clonedLoop.getStep());
  if (!clonedUpper || !clonedStep)
    return failCandidate(
        failureReason,
        "fixed-slot clone lost its admitted static loop bounds");
  mlir::FailureOr<mlir::scf::ForOp> pipelined =
      materializeSlotsAndRotation(clonedLoop, *clonedPlan, failureReason);
  if (mlir::failed(pipelined))
    return mlir::failure();
  if (mlir::failed(canonicalizePipelinedKernelUpperBound(
          *pipelined, *clonedUpper, *clonedStep, clonedPlan->maxStage)))
    return failCandidate(
        failureReason,
        "fixed-slot candidate failed kernel-bound canonicalization");
  eraseSynthesizedPointerPermutations(*candidate);
  if (mlir::failed(normalizeMinimumNCCJoins(*candidate)))
    return failCandidate(
        failureReason,
        "fixed-slot candidate failed NCC completion normalization");
  if (mlir::failed(mlir::verify(*candidate)))
    return failCandidate(
        failureReason,
        "fixed-slot candidate failed verification after materialization");

  StaticFixedSlotPipelineCandidate result;
  result.module = std::move(candidate);
  result.stageCount = clonedPlan->maxStage + 1;
  result.slotAllocationCount = clonedPlan->slotAllocationCount;
  return result;
}

} // namespace wafer
