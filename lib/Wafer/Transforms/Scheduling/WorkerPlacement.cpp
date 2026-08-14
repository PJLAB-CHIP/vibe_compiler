//===- WorkerPlacement.cpp - Typed NCC worker alternatives --------------===//

#include "Wafer/Transforms/WorkerPlacement.h"

#include "Wafer/Analysis/StaticBufferRange.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/WaferInterfaces.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer {
namespace {

struct Access {
  mlir::Value value;
  llvm::SmallVector<mlir::Value, 2> roots;
  const mlir::SideEffects::Resource *resource = nullptr;
  std::optional<analysis::StaticByteRange> range;
  bool writes = false;
};

static void fail(std::string *failureReason, llvm::StringRef message) {
  if (failureReason)
    *failureReason = message.str();
}

static void collectAccessRoots(mlir::Value value,
                               llvm::SmallVectorImpl<mlir::Value> &roots,
                               llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;

  if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = blockArgument.getOwner();
    mlir::Operation *parent = owner ? owner->getParentOp() : nullptr;
    if (auto region = mlir::dyn_cast_or_null<TileRegionOp>(parent)) {
      if (owner == &region.getBody().front() &&
          blockArgument.getArgNumber() < region.getInputs().size()) {
        collectAccessRoots(region.getInputs()[blockArgument.getArgNumber()],
                           roots, visited);
        return;
      }
    }
    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
      unsigned argument = blockArgument.getArgNumber();
      if (owner == loop.getBody() && argument != 0) {
        unsigned index = argument - 1;
        if (index < loop.getInitArgs().size())
          collectAccessRoots(loop.getInitArgs()[index], roots, visited);
        auto yield =
            mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        if (yield && index < yield.getNumOperands())
          collectAccessRoots(yield.getOperand(index), roots, visited);
        if (!roots.empty())
          return;
      }
    }
    roots.push_back(value);
    return;
  }

  mlir::Operation *definition = value.getDefiningOp();
  if (!definition) {
    roots.push_back(value);
    return;
  }
  if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definition)) {
    collectAccessRoots(view.getViewSource(), roots, visited);
    return;
  }
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
    if (auto region = mlir::dyn_cast<TileRegionOp>(definition)) {
      if (!region.getBody().empty()) {
        auto yield = mlir::dyn_cast<TileYieldOp>(
            region.getBody().front().getTerminator());
        if (yield && result.getResultNumber() < yield.getValues().size())
          collectAccessRoots(yield.getValues()[result.getResultNumber()], roots,
                             visited);
      }
      if (!roots.empty())
        return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(definition)) {
      unsigned index = result.getResultNumber();
      if (index < loop.getInitArgs().size())
        collectAccessRoots(loop.getInitArgs()[index], roots, visited);
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
      if (yield && index < yield.getNumOperands())
        collectAccessRoots(yield.getOperand(index), roots, visited);
      if (!roots.empty())
        return;
    }
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(definition)) {
      unsigned index = result.getResultNumber();
      for (mlir::Region &branch : ifOp->getRegions()) {
        if (branch.empty())
          continue;
        auto yield =
            mlir::dyn_cast<mlir::scf::YieldOp>(branch.front().getTerminator());
        if (yield && index < yield.getNumOperands())
          collectAccessRoots(yield.getOperand(index), roots, visited);
      }
      if (!roots.empty())
        return;
    }
  }
  roots.push_back(value);
}

static llvm::SmallVector<mlir::Value, 2> getAccessRoots(mlir::Value value) {
  llvm::SmallVector<mlir::Value, 2> roots;
  llvm::DenseSet<mlir::Value> visited;
  collectAccessRoots(value, roots, visited);
  return roots;
}

static bool areKnownDistinctRoots(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs)
    return false;
  bool lhsFresh = lhs.getDefiningOp<mlir::memref::AllocOp>() ||
                  lhs.getDefiningOp<mlir::memref::AllocaOp>();
  bool rhsFresh = rhs.getDefiningOp<mlir::memref::AllocOp>() ||
                  rhs.getDefiningOp<mlir::memref::AllocaOp>();
  if (lhsFresh && rhsFresh)
    return true;
  auto lhsGlobal = lhs.getDefiningOp<mlir::memref::GetGlobalOp>();
  auto rhsGlobal = rhs.getDefiningOp<mlir::memref::GetGlobalOp>();
  return lhsGlobal && rhsGlobal && lhsGlobal.getName() != rhsGlobal.getName();
}

static bool accessesAreProvenDisjoint(const Access &lhs, const Access &rhs) {
  if (lhs.resource != rhs.resource)
    return true;
  auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.value.getType());
  auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.value.getType());
  MemoryAttr lhsMemory = lhsType ? getWaferMemoryAttr(lhsType) : MemoryAttr{};
  MemoryAttr rhsMemory = rhsType ? getWaferMemoryAttr(rhsType) : MemoryAttr{};
  if (lhsMemory && rhsMemory && lhsMemory.getSpace() != rhsMemory.getSpace())
    return true;
  if (lhs.roots.empty() || rhs.roots.empty())
    return false;
  for (mlir::Value lhsRoot : lhs.roots) {
    for (mlir::Value rhsRoot : rhs.roots) {
      if (areKnownDistinctRoots(lhsRoot, rhsRoot))
        continue;
      if (lhsRoot == rhsRoot && lhs.range && rhs.range &&
          analysis::staticByteRangesAreDisjoint(*lhs.range, *rhs.range))
        continue;
      return false;
    }
  }
  return true;
}

static mlir::LogicalResult
collectAccesses(mlir::Operation *operation,
                llvm::SmallVectorImpl<Access> &accesses,
                std::string *failureReason) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects) {
    fail(failureReason,
         "worker placement requires typed value-associated effects");
    return mlir::failure();
  }
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  for (const auto &instance : instances) {
    if (mlir::isa<mlir::MemoryEffects::Allocate>(instance.getEffect()))
      continue;
    mlir::Value value = instance.getValue();
    if (!value) {
      // Rootless non-default effects are typed engine or address-space
      // occupancy. They do not create a data-dependency component; actual
      // value effects below own the range hazard.
      if (instance.getResource() == mlir::SideEffects::DefaultResource::get()) {
        fail(failureReason,
             "worker placement found an unresolved rootless effect");
        return mlir::failure();
      }
      continue;
    }
    bool reads = mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    accesses.push_back({value, getAccessRoots(value), instance.getResource(),
                        analysis::getStaticByteRange(value),
                        /*writes=*/!reads});
  }
  if (accesses.empty()) {
    fail(failureReason,
         "worker placement found no value-associated instruction access");
    return mlir::failure();
  }
  return mlir::success();
}

static bool hasSupportedStructuredAncestors(mlir::Operation *operation) {
  for (mlir::Operation *parent = operation->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (mlir::isa<mlir::ModuleOp, mlir::func::FuncOp, mlir::scf::ForOp,
                  mlir::scf::IfOp, TileRegionOp>(parent))
      continue;
    return false;
  }
  return true;
}

static mlir::LogicalResult
verifyExactDirectDTECompletion(mlir::ModuleOp module,
                               std::string *failureReason) {
  bool valid = true;
  module.walk([&](mlir::Operation *operation) {
    if (!valid)
      return;
    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation);
    if (!instruction || instruction.getInstructionFamily() != InstrFamily::DTE)
      return;

    if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation)) {
      for (mlir::Value token : wait.getTokens()) {
        mlir::Operation *issue = token.getDefiningOp();
        if (!mlir::isa_and_nonnull<InstrDTESendOp, InstrDTERecvOp>(issue) ||
            issue->getBlock() != wait->getBlock() ||
            !issue->isBeforeInBlock(wait)) {
          valid = false;
          return;
        }
      }
      return;
    }

    mlir::Value token;
    if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
      token = send.getToken();
    else if (auto receive = mlir::dyn_cast<InstrDTERecvOp>(operation))
      token = receive.getToken();
    else {
      valid = false;
      return;
    }
    if (!token.hasOneUse()) {
      valid = false;
      return;
    }
    auto wait = mlir::dyn_cast<InstrDTEWaitOp>(*token.getUsers().begin());
    if (!wait || wait->getBlock() != operation->getBlock() ||
        !operation->isBeforeInBlock(wait))
      valid = false;
  });
  if (valid)
    return mlir::success();
  fail(failureReason,
       "worker placement requires every Direct-DTE issue to have one exact "
       "same-block wait");
  return mlir::failure();
}

} // namespace

mlir::FailureOr<NCCWorkerPlacementCandidate>
deriveDisjointNCCWorkerPlacementCandidate(mlir::ModuleOp sourceModule,
                                          std::string *failureReason) {
  wafer::support::ScopedCompileTimingSpan timing(
      "optimization", "disjoint-worker-placement",
      "deriveDisjointNCCWorkerPlacementCandidate");
  auto phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "deriveDisjointNCCWorkerPlacementCandidate",
      "preflight");
  if (failureReason)
    failureReason->clear();
  if (!sourceModule) {
    fail(failureReason, "worker placement requires a source module");
    return mlir::failure();
  }

  bool hasPhysicalFact = false;
  bool hasDTE = false;
  bool hasUnsupportedObserver = false;
  bool hasNonzeroWorker = false;
  sourceModule.walk([&](mlir::Operation *operation) {
    if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation))
      hasPhysicalFact |= allocation->hasAttr(kWaferSPMOffsetAttrName) ||
                         allocation->hasAttr(kWaferDDROffsetAttrName);
    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation);
    hasDTE |=
        instruction && instruction.getInstructionFamily() == InstrFamily::DTE;
    NCCSynchronizationContract completion = getNCCSynchronizationContract(operation);
    hasUnsupportedObserver |=
        completion.behavior == NCCSynchronizationBehavior::SynchronousWriteback;
    hasNonzeroWorker |=
        completion.issueWorker && *completion.issueWorker != NCCWorker::Worker0;
  });
  if (hasPhysicalFact) {
    fail(failureReason,
         "worker placement requires an unplaced complete-rank module");
    return mlir::failure();
  }
  if (hasNonzeroWorker) {
    fail(failureReason,
         "worker placement source already has a nonzero worker assignment");
    return mlir::failure();
  }
  if (hasDTE && mlir::failed(verifyExactDirectDTECompletion(sourceModule,
                                                            failureReason))) {
    return mlir::failure();
  }
  if (hasUnsupportedObserver) {
    fail(failureReason, "worker placement rejects synchronous completion");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Operation *, 16> issues;
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "deriveDisjointNCCWorkerPlacementCandidate",
      "collect-issues-and-accesses");
  sourceModule.walk([&](mlir::Operation *operation) {
    if (mlir::isa<WaferNCCIssueOpInterface>(operation))
      issues.push_back(operation);
  });
  if (issues.size() < 2) {
    fail(failureReason,
         "worker placement requires at least two typed NCC issues");
    return mlir::failure();
  }
  if (llvm::any_of(issues, [](mlir::Operation *operation) {
        return !hasSupportedStructuredAncestors(operation);
      })) {
    fail(failureReason,
         "worker placement requires supported structured control flow");
    return mlir::failure();
  }

  llvm::SmallVector<llvm::SmallVector<Access, 4>, 16> accesses(issues.size());
  for (auto [index, operation] : llvm::enumerate(issues))
    if (mlir::failed(
            collectAccesses(operation, accesses[index], failureReason)))
      return mlir::failure();

  llvm::DenseMap<mlir::Operation *, size_t> issueIndices;
  for (auto [index, operation] : llvm::enumerate(issues))
    issueIndices.try_emplace(operation, index);

  // Build directed work lanes in stable issue order. An issue with no prior
  // SSA or RAW/WAR/WAW predecessor starts an independent lane. A dependent
  // issue continues the most recent predecessor lane. At a merge of several
  // lanes, the other predecessor workers remain distinct and the exact NCC
  // completion normalizer below inserts only the participant join required
  // before the merge issue. Treating this graph as one undirected component
  // would erase the useful input-prefetch window at every fan-in.
  llvm::SmallVector<unsigned, 16> issueComponents;
  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "deriveDisjointNCCWorkerPlacementCandidate",
      "build-dependency-lanes");
  issueComponents.reserve(issues.size());
  std::vector<std::optional<size_t>> latestPredecessors(issues.size());
  auto deriveLatestPredecessor = [&](size_t index) {
    mlir::Operation *operation = issues[index];
    std::optional<size_t> latestPredecessor;
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      auto found = issueIndices.find(definition);
      if (found != issueIndices.end() && found->second < index)
        latestPredecessor =
            std::max(latestPredecessor.value_or(0), found->second);
    }
    for (size_t predecessor = 0; predecessor < index; ++predecessor) {
      bool dependency = false;
      for (const Access &left : accesses[predecessor]) {
        for (const Access &right : accesses[index]) {
          if (!left.writes && !right.writes)
            continue;
          if (!accessesAreProvenDisjoint(left, right)) {
            dependency = true;
            break;
          }
        }
        if (dependency)
          break;
      }
      if (dependency)
        latestPredecessor =
            std::max(latestPredecessor.value_or(0), predecessor);
    }
    latestPredecessors[index] = latestPredecessor;
  };
  if (sourceModule.getContext()->isMultithreadingEnabled() && issues.size() > 1)
    mlir::parallelFor(sourceModule.getContext(), 0, issues.size(),
                      deriveLatestPredecessor);
  else
    for (size_t index = 0; index < issues.size(); ++index)
      deriveLatestPredecessor(index);

  unsigned componentCount = 0;
  for (std::optional<size_t> latestPredecessor : latestPredecessors) {
    if (latestPredecessor)
      issueComponents.push_back(issueComponents[*latestPredecessor]);
    else
      issueComponents.push_back(componentCount++);
  }
  if (componentCount < 2) {
    fail(failureReason, "worker placement found only one dependency lane");
    return mlir::failure();
  }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "transformation-phase", "deriveDisjointNCCWorkerPlacementCandidate",
      "clone-and-assign-workers");
  mlir::OwningOpRef<mlir::ModuleOp> candidate =
      mlir::cast<mlir::ModuleOp>(sourceModule->clone());
  llvm::SmallVector<mlir::Operation *, 16> clonedIssues;
  llvm::SmallVector<SyncNCCJoinOp, 8> oldJoins;
  candidate->walk([&](mlir::Operation *operation) {
    if (mlir::isa<WaferNCCIssueOpInterface>(operation))
      clonedIssues.push_back(operation);
    if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation))
      oldJoins.push_back(join);
  });
  if (clonedIssues.size() != issues.size()) {
    fail(failureReason,
         "worker placement clone changed the typed issue domain");
    return mlir::failure();
  }
  for (SyncNCCJoinOp join : oldJoins)
    join.erase();

  uint32_t participantMask = 0;
  for (auto [index, operation] : llvm::enumerate(clonedIssues)) {
    uint32_t worker = issueComponents[index] % kNCCWorkerCount;
    if (mlir::failed(
            setNCCIssueWorker(operation, static_cast<NCCWorker>(worker)))) {
      fail(failureReason,
           "worker placement could not update a typed NCC issue");
      return mlir::failure();
    }
    participantMask |= uint32_t{1} << worker;
  }
  if (llvm::popcount(participantMask) < 2) {
    fail(failureReason, "worker placement did not activate a nonzero worker");
    return mlir::failure();
  }

  phaseTiming = std::make_unique<wafer::support::ScopedCompileTimingSpan>(
      "analysis-phase", "deriveDisjointNCCWorkerPlacementCandidate",
      "normalize-verify-and-analyze-windows");
  if (mlir::failed(placeRequiredNCCJoins(*candidate)) ||
      mlir::failed(mlir::verify(*candidate))) {
    fail(failureReason,
         "worker placement failed exact completion or IR verification");
    return mlir::failure();
  }
  NCCWorkerWindowSummary workerWindows = analyzeNCCWorkerWindows(*candidate);
  if (!workerWindows.hasCrossWorkerWindow ||
      workerWindows.issuedWorkerMask != participantMask) {
    fail(failureReason,
         "worker placement found no exact cross-worker issue window");
    return mlir::failure();
  }

  return NCCWorkerPlacementCandidate{std::move(candidate), componentCount,
                                     participantMask};
}

} // namespace wafer
