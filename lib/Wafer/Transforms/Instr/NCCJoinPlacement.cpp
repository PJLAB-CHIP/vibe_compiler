//===- NCCJoinPlacement.cpp - Current-IR required NCC joins ------------===//

#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"

#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Target/TargetMemory.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <map>
#include <vector>

#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>

using namespace wafer;
namespace wafer {
#define GEN_PASS_DEF_PLACEREQUIREDNCCJOINSPASS
#define GEN_PASS_DEF_REBUILDREQUIREDNCCJOINSPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"
} // namespace wafer

namespace {

struct NCCOutstandingAccessSummary {
  uint32_t workers = 0;
  llvm::DenseSet<mlir::Value> kcoreWrites;
  llvm::DenseMap<mlir::Value, uint32_t> readers;
  llvm::DenseMap<mlir::Value, uint32_t> writers;
};

static void addMask(llvm::DenseMap<mlir::Value, uint32_t> &masks,
                    mlir::Value root, uint32_t mask) {
  if (root && mask)
    masks[root] |= mask;
}

static void clearMask(llvm::DenseMap<mlir::Value, uint32_t> &masks,
                      uint32_t completed) {
  for (auto iterator = masks.begin(); iterator != masks.end();) {
    iterator->second &= ~completed;
    if (iterator->second != 0) {
      ++iterator;
      continue;
    }
    auto dead = iterator++;
    masks.erase(dead);
  }
}

static void clearSynchronizedWorkers(NCCOutstandingAccessSummary &state,
                                     uint32_t completed) {
  state.workers &= ~completed;
  clearMask(state.readers, completed);
  clearMask(state.writers, completed);
}

static void
mergeOutstandingAccessSummaries(NCCOutstandingAccessSummary &destination,
                                const NCCOutstandingAccessSummary &source) {
  destination.workers |= source.workers;
  destination.kcoreWrites.insert(source.kcoreWrites.begin(), source.kcoreWrites.end());
  for (const auto &entry : source.readers)
    destination.readers[entry.first] |= entry.second;
  for (const auto &entry : source.writers)
    destination.writers[entry.first] |= entry.second;
}

static bool haveEqualMasks(const llvm::DenseMap<mlir::Value, uint32_t> &lhs,
                           const llvm::DenseMap<mlir::Value, uint32_t> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(lhs, [&](const auto &entry) {
    auto found = rhs.find(entry.first);
    return found != rhs.end() && found->second == entry.second;
  });
}

static bool
haveEqualOutstandingAccessSummaries(const NCCOutstandingAccessSummary &lhs,
                                    const NCCOutstandingAccessSummary &rhs) {
  return lhs.workers == rhs.workers &&
         lhs.kcoreWrites.size() == rhs.kcoreWrites.size() &&
         llvm::all_of(lhs.kcoreWrites, [&](mlir::Value v) { return rhs.kcoreWrites.contains(v); }) &&
         haveEqualMasks(lhs.readers, rhs.readers) &&
         haveEqualMasks(lhs.writers, rhs.writers);
}

static void collectAccessRoots(mlir::Value value,
                               llvm::SmallVectorImpl<mlir::Value> &roots,
                               llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;

  if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = blockArgument.getOwner();
    mlir::Operation *parent = owner ? owner->getParentOp() : nullptr;
    if (auto tileRegion = mlir::dyn_cast_or_null<TileRegionOp>(parent);
        tileRegion && tileRegion.getBody().hasOneBlock() &&
        owner == &tileRegion.getBody().front() &&
        blockArgument.getArgNumber() < tileRegion.getInputs().size()) {
      collectAccessRoots(tileRegion.getInputs()[blockArgument.getArgNumber()],
                         roots, visited);
      return;
    }
    if (auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent);
        loop && owner == loop.getBody() && blockArgument.getArgNumber() > 0) {
      unsigned index = blockArgument.getArgNumber() - 1;
      auto yield = mlir::cast<mlir::scf::YieldOp>(owner->getTerminator());
      if (yield.getOperand(index) == blockArgument) {
        collectAccessRoots(loop.getInitArgs()[index], roots, visited);
        return;
      }
    }
    // Keep same-iteration SCF state variables distinct here. Expanding each
    // one independently through init/yield loses the relational fact that a
    // rotating-buffer recurrence is a permutation of disjoint allocations.
    // `areKnownDistinctRoots` proves such pairs by induction below.
    roots.push_back(value);
    return;
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  mlir::Operation *definition = result ? result.getOwner() : nullptr;
  if (!definition) {
    roots.push_back(value);
    return;
  }
  if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definition)) {
    collectAccessRoots(view.getViewSource(), roots, visited);
    return;
  }
  if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(definition);
      tileRegion && tileRegion.getBody().hasOneBlock()) {
    auto yield = mlir::dyn_cast<TileYieldOp>(
        tileRegion.getBody().front().getTerminator());
    if (yield && result.getResultNumber() < yield.getValues().size()) {
      collectAccessRoots(yield.getValues()[result.getResultNumber()], roots,
                         visited);
      return;
    }
  }
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(definition)) {
    for (mlir::Region &region : ifOp->getRegions()) {
      if (region.empty())
        continue;
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
      if (yield && result.getResultNumber() < yield.getOperands().size())
        collectAccessRoots(yield.getOperands()[result.getResultNumber()], roots,
                           visited);
    }
    if (!roots.empty())
      return;
  }
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(definition)) {
    unsigned resultNumber = result.getResultNumber();
    if (resultNumber < forOp.getInitArgs().size())
      collectAccessRoots(forOp.getInitArgs()[resultNumber], roots, visited);
    auto yield =
        mlir::dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    if (yield && resultNumber < yield.getOperands().size())
      collectAccessRoots(yield.getOperands()[resultNumber], roots, visited);
    if (!roots.empty())
      return;
  }
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(definition);
      cast && cast.getInputs().size() == 1) {
    collectAccessRoots(cast.getInputs().front(), roots, visited);
    return;
  }
  roots.push_back(value);
}

static llvm::SmallVector<mlir::Value, 4> getAccessRoots(mlir::Value value) {
  llvm::SmallVector<mlir::Value, 4> roots;
  llvm::DenseSet<mlir::Value> visited;
  collectAccessRoots(value, roots, visited);
  return roots;
}

static bool areKnownDistinctRootsImpl(
    mlir::Value lhs, mlir::Value rhs,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::Value>> &activePairs) {
  if (lhs == rhs)
    return false;
  if (lhs.getDefiningOp<mlir::memref::AllocOp>() &&
      rhs.getDefiningOp<mlir::memref::AllocOp>())
    return true;
  auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
  MemoryAttr lhsMemory = lhsType ? getWaferMemoryAttr(lhsType) : MemoryAttr{};
  MemoryAttr rhsMemory = rhsType ? getWaferMemoryAttr(rhsType) : MemoryAttr{};
  if (lhsMemory && rhsMemory && lhsMemory.getSpace() != rhsMemory.getSpace())
    return true;

  auto lhsArgument = mlir::dyn_cast<mlir::BlockArgument>(lhs);
  auto rhsArgument = mlir::dyn_cast<mlir::BlockArgument>(rhs);
  if (!lhsArgument || !rhsArgument ||
      lhsArgument.getOwner() != rhsArgument.getOwner() ||
      lhsArgument.getArgNumber() == 0 || rhsArgument.getArgNumber() == 0)
    return false;
  auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
      lhsArgument.getOwner()->getParentOp());
  if (!loop || lhsArgument.getOwner() != loop.getBody())
    return false;

  auto active = llvm::find_if(activePairs, [&](const auto &pair) {
    return (pair.first == lhs && pair.second == rhs) ||
           (pair.first == rhs && pair.second == lhs);
  });
  // Coinductive backedge: every pair on the active chain has already proved
  // its distinct init state. Re-entering that pair therefore closes the
  // induction over the loop recurrence.
  if (active != activePairs.end())
    return true;

  unsigned lhsIndex = lhsArgument.getArgNumber() - 1;
  unsigned rhsIndex = rhsArgument.getArgNumber() - 1;
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (lhsIndex >= loop.getInitArgs().size() ||
      rhsIndex >= loop.getInitArgs().size() || !yield ||
      lhsIndex >= yield.getOperands().size() ||
      rhsIndex >= yield.getOperands().size())
    return false;

  activePairs.push_back({lhs, rhs});
  bool distinct =
      areKnownDistinctRootsImpl(loop.getInitArgs()[lhsIndex],
                                loop.getInitArgs()[rhsIndex], activePairs) &&
      areKnownDistinctRootsImpl(yield.getOperands()[lhsIndex],
                                yield.getOperands()[rhsIndex], activePairs);
  activePairs.pop_back();
  return distinct;
}

static bool areKnownDistinctRoots(mlir::Value lhs, mlir::Value rhs) {
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 4> activePairs;
  return areKnownDistinctRootsImpl(lhs, rhs, activePairs);
}

static uint32_t
getAliasingMask(mlir::Value value,
                const llvm::DenseMap<mlir::Value, uint32_t> &masks) {
  uint32_t result = 0;
  for (mlir::Value root : getAccessRoots(value)) {
    for (const auto &entry : masks) {
      if (!areKnownDistinctRoots(root, entry.first))
        result |= entry.second;
    }
  }
  return result;
}

static llvm::SmallVector<int64_t, kNCCWorkerCount>
getParticipants(uint32_t mask) {
  llvm::SmallVector<int64_t, kNCCWorkerCount> participants;
  for (uint32_t worker = 0; worker < kNCCWorkerCount; ++worker) {
    if (mask & (uint32_t{1} << worker))
      participants.push_back(static_cast<int64_t>(worker));
  }
  return participants;
}

static void recordNCCIssue(mlir::Operation *operation, uint32_t workerMask,
                           NCCOutstandingAccessSummary &state) {
  state.workers |= workerMask;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  for (const auto &instance : instances) {
    mlir::Value value = instance.getValue();
    if (!value)
      continue;
    bool read = mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    bool allocate =
        mlir::isa<mlir::MemoryEffects::Allocate>(instance.getEffect());
    if (allocate)
      continue;
    for (mlir::Value root : getAccessRoots(value)) {
      if (read)
        addMask(state.readers, root, workerMask);
      else
        addMask(state.writers, root, workerMask);
    }
  }
}

static uint32_t
getExternalConflictMask(mlir::Operation *operation,
                        const NCCOutstandingAccessSummary &state,
                        bool ignoreTypedIssueResources = false) {
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return 0;
  // Once an NCC address lifetime has advanced through a same-worker issue
  // chain, static packing may reuse that address for a later NCC allocation.
  // Direct DTE is a different completion domain and does not participate in
  // the NCC busytable, so every still-pending NCC worker must be completed
  // before a DTE issue can become the first access to such a reused address.
  // The exact DTE wait remains independent and never completes NCC work.
  uint32_t conflicts =
      mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation) ? state.workers : 0;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  bool ignoreTypedResources =
      ignoreTypedIssueResources ||
      mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(operation);
  for (const auto &instance : instances) {
    mlir::Value value = instance.getValue();
    if (!value) {
      // An NCC issue's custom Wafer resource effects describe typed engine
      // and memory-space occupancy. Its value-associated effects below are
      // the address hazard contract, so the custom resource must not turn two
      // proven-distinct roots into a blocking cross-worker join. A Direct DTE
      // issue has the same property: its value-associated buffer
      // effect is the NCC visibility boundary, while its rootless
      // communication resource describes transport occupancy. The exact DTE
      // wait carries no NCC buffer observation and must not become an implicit
      // NCC join. Unknown and other synchronous observers retain the
      // conservative all-pending conflict.
      if (ignoreTypedResources &&
          instance.getResource() != mlir::SideEffects::DefaultResource::get())
        continue;
      if (!mlir::isa<mlir::MemoryEffects::Allocate, mlir::MemoryEffects::Free>(
              instance.getEffect()))
        conflicts |= state.workers;
      continue;
    }
    // alloc/free delimit compiler-owned lifetime but do not execute a Kcore or
    // host memory access. Actual reuse is ordered by the next typed issue:
    // same-worker ranges use busytable order and cross-worker conflicts join
    // immediately before that issue.
    if (mlir::isa<mlir::MemoryEffects::Allocate, mlir::MemoryEffects::Free>(
            instance.getEffect()))
      continue;
    bool read = mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
    conflicts |= getAliasingMask(value, state.writers);
    if (!read)
      conflicts |= getAliasingMask(value, state.readers);
  }
  return conflicts;
}

/// Resolve a scalar carried into a tile residency region back to the enclosing
/// SSA value. Region partitioning may turn a previously local loop bound into
/// a region input; that transport does not make a static bound dynamic and
/// must not change required-join legality.
static mlir::Value resolveTileRegionScalarForwarding(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (value && seen.insert(value).second) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = argument.getOwner();
      auto region =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp{};
      if (!region || region.getBody().empty() ||
          owner != &region.getBody().front() ||
          argument.getArgNumber() >= region.getInputs().size())
        break;
      value = region.getInputs()[argument.getArgNumber()];
      continue;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    auto region = result ? mlir::dyn_cast<TileRegionOp>(result.getOwner())
                         : TileRegionOp{};
    if (!region || region.getBody().empty())
      break;
    auto yield =
        mlir::dyn_cast<TileYieldOp>(region.getBody().front().getTerminator());
    if (!yield || result.getResultNumber() >= yield.getValues().size())
      break;
    value = yield.getValues()[result.getResultNumber()];
  }
  return value;
}

class RequiredNCCJoinPlacement {
public:
  mlir::LogicalResult run(mlir::func::FuncOp function) {
    if (function.isExternal())
      return mlir::success();
    if (!function.getBody().hasOneBlock()) {
      bool hasIssue = false;
      function.walk([&](WaferNCCIssueOpInterface) { hasIssue = true; });
      if (!hasIssue)
        return mlir::success();
      return function.emitError()
             << "required_ncc_join_failure: exact NCC synchronization "
                "placement requires structured single-block function IR";
    }
    NCCOutstandingAccessSummary state;
    if (mlir::failed(processBlock(function.getBody().front(), state)))
      return mlir::failure();
    return apply(function.getOperation());
  }

  mlir::LogicalResult run(TileRegionOp tileRegion) {
    NCCOutstandingAccessSummary state;
    if (mlir::failed(processOperation(tileRegion.getOperation(), state)))
      return mlir::failure();
    return apply(tileRegion.getOperation());
  }

private:
  struct LoopPhase {
    mlir::scf::ForOp loop;
    bool first;
    friend bool operator==(const LoopPhase &lhs, const LoopPhase &rhs) {
      return lhs.loop == rhs.loop && lhs.first == rhs.first;
    }
  };
  struct JoinRequest {
    llvm::SmallVector<LoopPhase, 4> phases;
    uint32_t mask;
    bool kcoreRelease = false;
  };
  llvm::SmallVector<LoopPhase, 4> phases;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<JoinRequest, 4>> requests;
  llvm::DenseSet<mlir::Operation *> nonemptyLoops;

  static void mergeRequest(llvm::SmallVectorImpl<JoinRequest> &into,
                           JoinRequest request) {
    for (auto &existing : into)
      if (existing.phases == request.phases) {
        existing.mask |= request.mask;
        existing.kcoreRelease |= request.kcoreRelease;
        return;
      }
    into.push_back(std::move(request));
  }

  void insertNCCJoinBefore(mlir::Operation *operation, uint32_t mask,
                           NCCOutstandingAccessSummary &state) {
    mask &= state.workers;
    if (!mask)
      return;
    mergeRequest(requests[operation], {phases, mask});
    clearSynchronizedWorkers(state, mask);
    state.kcoreWrites.clear();
  }

  void releaseKcoreWritesBefore(mlir::Operation *op,
                                NCCOutstandingAccessSummary &state) {
    if (state.kcoreWrites.empty())
      return;
    auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
    if (!effects)
      return;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    bool conflict = false;
    for (const auto &effect : instances) {
      if (!effect.getValue() || !mlir::isa<mlir::MemoryEffects::Read,
                                          mlir::MemoryEffects::Write>(effect.getEffect()))
        continue;
      for (auto root : getAccessRoots(effect.getValue()))
        for (auto written : state.kcoreWrites)
          conflict |= !areKnownDistinctRoots(root, written);
    }
    if (conflict) {
      mergeRequest(requests[op], {phases, 0, true});
      state.kcoreWrites.clear();
    }
  }

  static void simplifyRequests(llvm::SmallVectorImpl<JoinRequest> &items) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (size_t i = 0; i < items.size() && !changed; ++i)
        for (size_t j = i + 1; j < items.size(); ++j) {
          uint32_t common = items[i].mask & items[j].mask;
          bool commonRelease = items[i].kcoreRelease && items[j].kcoreRelease;
          if ((!common && !commonRelease) || items[i].phases.size() != items[j].phases.size())
            continue;
          unsigned differences = 0, differentPhase = 0;
          bool sameLoops = true;
          for (unsigned k = 0; k < items[i].phases.size(); ++k) {
            sameLoops &= items[i].phases[k].loop == items[j].phases[k].loop;
            if (items[i].phases[k].first != items[j].phases[k].first) {
              ++differences;
              differentPhase = k;
            }
          }
          if (!sameLoops || differences != 1)
            continue;
          JoinRequest combined = items[i];
          combined.phases.erase(combined.phases.begin() + differentPhase);
          combined.mask = common;
          combined.kcoreRelease = commonRelease;
          items[i].mask &= ~common;
          items[j].mask &= ~common;
          if (commonRelease) {
            items[i].kcoreRelease = false;
            items[j].kcoreRelease = false;
          }
          mergeRequest(items, std::move(combined));
          changed = true;
          break;
        }
    }
  }

  bool canHoistEntryJoin(mlir::Operation *operation,
                         mlir::scf::ForOp loop, uint32_t mask, bool kcoreRelease) const {
    if (!nonemptyLoops.contains(loop.getOperation()))
      return false;
    for (mlir::Operation *current = operation; current != loop.getOperation();) {
      if (!current->getBlock())
        return false;
      for (auto &previous : *current->getBlock()) {
        if (&previous == current)
          break;
        bool issuesParticipant = false;
        previous.walk([&](mlir::Operation *nested) {
          if (kcoreRelease && mlir::isa<mlir::memref::StoreOp>(nested))
            issuesParticipant = true;
          auto completion = getNCCOperationCompletion(nested);
          if (completion.issueWorker)
            issuesParticipant |= (mask & (uint32_t{1} <<
                static_cast<uint32_t>(*completion.issueWorker))) != 0;
        });
        if (issuesParticipant)
          return false;
      }
      current = current->getParentOp();
      if (!current)
        return false;
      if (current != loop.getOperation() && !mlir::isa<TileRegionOp>(current) &&
          !nonemptyLoops.contains(current))
        return false;
    }
    return true;
  }

  // Build the exact disjunction of phase requests as a reduced decision DAG.
  // Destructive pairwise cube merging alone can leave complementary outer
  // phases split across joins, hiding guaranteed reentry completion.
  mlir::Value buildGuard(mlir::OpBuilder &builder, mlir::Location location,
                         llvm::ArrayRef<const JoinRequest *> group) {
    llvm::SmallVector<mlir::scf::ForOp, 4> loops;
    for (auto request : group)
      for (auto phase : request->phases)
        if (!llvm::is_contained(loops, phase.loop))
          loops.push_back(phase.loop);
    // All variables are ancestors of the same observer; order outer first.
    llvm::sort(loops, [](auto a, auto b) { return a->isAncestor(b); });
    using Cube = std::vector<int8_t>;
    using Cover = std::vector<Cube>;
    Cover initial;
    for (auto request : group) {
      Cube cube(loops.size(), -1);
      for (auto phase : request->phases)
        cube[llvm::find(loops, phase.loop) - loops.begin()] = phase.first;
      initial.push_back(std::move(cube));
    }
    auto yes =
        builder.create<mlir::arith::ConstantIntOp>(location, 1, 1).getResult();
    auto no =
        builder.create<mlir::arith::ConstantIntOp>(location, 0, 1).getResult();
    std::map<Cover, mlir::Value> memo;
    std::function<mlir::Value(Cover)> build = [&](Cover cover) -> mlir::Value {
      if (cover.empty())
        return no;
      for (const auto &cube : cover)
        if (llvm::all_of(cube, [](int8_t value) { return value < 0; }))
          return yes;
      llvm::sort(cover);
      cover.erase(std::unique(cover.begin(), cover.end()), cover.end());
      if (auto found = memo.find(cover); found != memo.end())
        return found->second;
      unsigned index = 0;
      for (; index < loops.size(); ++index)
        if (llvm::any_of(cover,
                         [&](const auto &cube) { return cube[index] >= 0; }))
          break;
      Cover first, later;
      for (auto cube : cover) {
        auto value = cube[index];
        cube[index] = -1;
        if (value != 0)
          first.push_back(cube);
        if (value != 1)
          later.push_back(std::move(cube));
      }
      auto a = build(std::move(first)), b = build(std::move(later));
      mlir::Value result = a;
      if (a != b) {
        auto compare = builder.create<mlir::arith::CmpIOp>(
            location, mlir::arith::CmpIPredicate::eq,
            loops[index].getInductionVar(), loops[index].getLowerBound());
        result = builder.createOrFold<mlir::arith::SelectOp>(location, compare,
                                                             a, b);
      }
      memo.emplace(std::move(cover), result);
      return result;
    };
    return build(std::move(initial));
  }

  mlir::LogicalResult apply(mlir::Operation *root) {
    llvm::SmallVector<mlir::Operation *> points;
    root->walk([&](mlir::Operation *op) {
      if (requests.count(op) || mlir::isa<SyncNCCJoinOp>(op))
        points.push_back(op);
    });
    for (mlir::Operation *point : points) {
      auto found = requests.find(point);
      if (found != requests.end()) {
        simplifyRequests(found->second);
        llvm::SmallVector<std::pair<uint32_t, bool>, 8> emitted;
        for (const JoinRequest &request : found->second) {
          if (!request.mask && !request.kcoreRelease)
            continue;
          auto key = std::make_pair(request.mask, request.kcoreRelease);
          if (llvm::is_contained(emitted, key))
            continue;
          emitted.push_back(key);
          llvm::SmallVector<const JoinRequest *, 4> group;
          for (const auto &other : found->second)
            if (other.mask == request.mask &&
                other.kcoreRelease == request.kcoreRelease)
              group.push_back(&other);
          bool hoisted = group.size() == 1 && !request.phases.empty() &&
                         llvm::all_of(request.phases,
                                      [](const LoopPhase &phase) {
                                        return phase.first;
                                      }) &&
                         canHoistEntryJoin(point, request.phases.front().loop,
                                           request.mask, request.kcoreRelease);
          mlir::Operation *anchor = point;
          if (hoisted) {
            auto loop = request.phases.front().loop;
            anchor = loop.getOperation();
          }
          mlir::OpBuilder builder(anchor);
          mlir::UnitAttr boundary;
          if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(point))
            boundary = join.getScheduleBoundaryAttr();
          if (!hoisted && !llvm::any_of(group, [](auto item) {
                return item->phases.empty();
              })) {
            auto condition = buildGuard(builder, point->getLoc(), group);
            auto branch = builder.create<mlir::scf::IfOp>(point->getLoc(),
                                                          condition, false);
            builder.setInsertionPointToStart(&branch.getThenRegion().front());
          }
          if (request.mask)
            builder.create<SyncNCCJoinOp>(point->getLoc(), getParticipants(request.mask), boundary);
          else if (request.kcoreRelease) {
            builder.getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
            builder.create<mlir::LLVM::FenceOp>(point->getLoc(), mlir::LLVM::AtomicOrdering::release,
                                               kTargetKcoreReleaseScope);
          }
        }
      }
      if (mlir::isa<SyncNCCJoinOp>(point))
        point->erase();
    }
    return mlir::success();
  }

  mlir::LogicalResult processBlock(mlir::Block &block,
                                   NCCOutstandingAccessSummary &state) {
    for (auto iterator = block.begin(); iterator != block.end();) {
      mlir::Operation *operation = &*iterator++;
      if (mlir::failed(processOperation(operation, state)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult processOperation(mlir::Operation *operation,
                                       NCCOutstandingAccessSummary &state) {
    if (auto fence = mlir::dyn_cast<mlir::LLVM::FenceOp>(operation);
        fence && fence.getSyncscope() == kTargetKcoreReleaseScope &&
        fence.getOrdering() == mlir::LLVM::AtomicOrdering::release) {
      state.kcoreWrites.clear();
      return mlir::success();
    }
    if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(operation)) {
      if (!tileRegion.getBody().hasOneBlock())
        return tileRegion.emitError()
               << "required_ncc_join_failure: outstanding NCC access "
                  "placement requires a single-block wafer.tile.region";
      if (mlir::failed(processBlock(tileRegion.getBody().front(), state)))
        return mlir::failure();
      return mlir::success();
    }

    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
      NCCOutstandingAccessSummary thenState = state;
      if (mlir::failed(processBlock(ifOp.getThenRegion().front(), thenState)))
        return mlir::failure();
      NCCOutstandingAccessSummary elseState = state;
      if (!ifOp.getElseRegion().empty() &&
          mlir::failed(processBlock(ifOp.getElseRegion().front(), elseState)))
        return mlir::failure();
      state = std::move(thenState);
      mergeOutstandingAccessSummaries(state, elseState);
      return mlir::success();
    }

    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
      auto getConstantIndex = [](mlir::Value value) -> std::optional<int64_t> {
        value = resolveTileRegionScalarForwarding(value);
        if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
          return constant;
        if (!value || !value.getType().isIndex())
          return std::nullopt;
        mlir::FailureOr<int64_t> constant =
            mlir::ValueBoundsConstraintSet::computeConstantBound(
                mlir::presburger::BoundType::EQ,
                mlir::ValueBoundsConstraintSet::Variable(value));
        return mlir::succeeded(constant) ? std::optional<int64_t>(*constant)
                                         : std::nullopt;
      };
      std::optional<int64_t> lower = getConstantIndex(forOp.getLowerBound());
      std::optional<int64_t> upper = getConstantIndex(forOp.getUpperBound());
      std::optional<int64_t> step = getConstantIndex(forOp.getStep());
      if (lower && upper && step && *step > 0 && *lower >= *upper)
        return mlir::success();

      bool guaranteedToExecute =
          lower && upper && step && *step > 0 && *lower < *upper;
      std::optional<__int128> tripCount;
      if (guaranteedToExecute) {
        __int128 span =
            static_cast<__int128>(*upper) - static_cast<__int128>(*lower);
        tripCount = (span + static_cast<__int128>(*step) - 1) /
                    static_cast<__int128>(*step);
      }
      if (guaranteedToExecute)
        nonemptyLoops.insert(forOp.getOperation());
      NCCOutstandingAccessSummary bodyState = state;
      bool repeated = !tripCount || *tripCount > 1;
      if (repeated)
        phases.push_back({forOp, true});
      auto firstResult = processBlock(*forOp.getBody(), bodyState);
      if (repeated)
        phases.pop_back();
      if (mlir::failed(firstResult))
        return mlir::failure();
      if (tripCount && *tripCount == 1) {
        state = std::move(bodyState);
        return mlir::success();
      }

      // Analyze entry and backedges without editing the shared body. The
      // backedge header is a monotone union of states reachable after at least
      // one iteration; entry-only requirements remain separate requests.
      const unsigned convergenceLimit = static_cast<unsigned>(kNCCWorkerCount) *
          (static_cast<unsigned>(std::distance(forOp.getBody()->begin(),
                                              forOp.getBody()->end())) +
           state.readers.size() + state.writers.size() + 1) + 1;
      NCCOutstandingAccessSummary header = bodyState;
      bool converged = false;
      for (unsigned iteration = 0; iteration < convergenceLimit; ++iteration) {
        NCCOutstandingAccessSummary nextState = header;
        phases.push_back({forOp, false});
        auto nextResult = processBlock(*forOp.getBody(), nextState);
        phases.pop_back();
        if (mlir::failed(nextResult))
          return mlir::failure();
        auto merged = header;
        mergeOutstandingAccessSummaries(merged, nextState);
        bodyState = std::move(nextState);
        if (haveEqualOutstandingAccessSummaries(merged, header)) {
          converged = true;
          break;
        }
        header = std::move(merged);
      }
      if (!converged)
        return forOp.emitError()
               << "required_ncc_join_failure: NCC loop backedge "
                  "access state did not reach a finite fixed point";
      if (guaranteedToExecute)
        state = std::move(bodyState);
      else
        // Preserve the zero-trip path while carrying every pending access
        // from one-or-more iterations to the first actual observer/terminal.
        mergeOutstandingAccessSummaries(state, bodyState);
      return mlir::success();
    }

    if (mlir::isa<mlir::func::ReturnOp, mlir::async::YieldOp>(operation)) {
      insertNCCJoinBefore(operation, state.workers, state);
      return mlir::success();
    }
    if (mlir::isa<TileYieldOp, mlir::scf::YieldOp>(operation))
      return mlir::success();

    // These operations only change the tensor/memref SSA view of one storage
    // object. They do not execute a Kcore/host read and therefore cannot turn
    // a pending NCC issue into a completion boundary.
    if (mlir::isa<mlir::bufferization::ToMemrefOp,
                  mlir::bufferization::ToTensorOp>(operation))
      return mlir::success();

    NCCOperationCompletion contract = getNCCOperationCompletion(operation);
    if (contract.issueWorker) {
      uint32_t worker = static_cast<uint32_t>(*contract.issueWorker);
      if (worker >= kNCCWorkerCount)
        return operation->emitError()
               << "required_ncc_join_failure: NCC issue worker is "
                  "outside the typed worker domain";
      uint32_t workerMask = uint32_t{1} << worker;
      // The target busytable orders an NCC worker's own RAW/WAR/WAW chain,
      // including compiler-managed store/reload and offset reuse. Complete
      // only prior conflicting workers before issuing the new access;
      // operation kind and residency structure are not completion events.
      uint32_t crossWorkerConflicts =
          getExternalConflictMask(operation, state,
                                  /*ignoreTypedIssueResources=*/true) &
          ~workerMask;
      insertNCCJoinBefore(operation, crossWorkerConflicts, state);
      releaseKcoreWritesBefore(operation, state);
      recordNCCIssue(operation, workerMask, state);
    }

    if (contract.kind == NCCCompletionKind::ParticipantJoin) {
      uint32_t requested = contract.participantMask & kAllNCCWorkersMask;
      if (auto join = mlir::dyn_cast<SyncNCCJoinOp>(operation)) {
        insertNCCJoinBefore(operation, requested, state);
        return mlir::success();
      }
      clearSynchronizedWorkers(state, requested);
      return mlir::success();
    }
    if (contract.kind == NCCCompletionKind::SynchronousWriteback) {
      clearSynchronizedWorkers(state, contract.participantMask);
      return mlir::success();
    }
    if (contract.kind == NCCCompletionKind::OrderedAsynchronousIssue)
      return mlir::success();

    // A Direct DTE wait observes only its own typed event. It neither consumes
    // nor joins pending NCC worker issues, so independent NCC work
    // may remain in flight while the transport is awaited.
    if (mlir::isa<InstrDTEWaitOp>(operation))
      return mlir::success();

    // A Direct DTE issue observes only its explicit SPM buffer. Ignore the
    // rootless communication-resource effect here and complete exactly the
    // NCC workers whose value-associated accesses conflict with that buffer.
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp>(operation)) {
      uint32_t conflicts = getExternalConflictMask(
          operation, state, /*ignoreTypedIssueResources=*/true);
      insertNCCJoinBefore(operation, conflicts, state);
      releaseKcoreWritesBefore(operation, state);
      return mlir::success();
    }

    if (mlir::isa<mlir::CallOpInterface>(operation)) {
      insertNCCJoinBefore(operation, state.workers, state);
      return mlir::success();
    }

    if (!mlir::isa<mlir::MemoryEffectOpInterface>(operation) &&
        !mlir::isMemoryEffectFree(operation)) {
      if (state.workers == 0)
        return mlir::success();
      return operation->emitError()
             << "required_ncc_join_failure: operation without a typed "
                "memory-effect contract may observe pending NCC work";
    }

    uint32_t conflicts = getExternalConflictMask(operation, state);
    insertNCCJoinBefore(operation, conflicts, state);
    if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(operation))
      for (auto root : getAccessRoots(store.getMemref()))
        state.kcoreWrites.insert(root);
    return mlir::success();
  }
};

static void eraseDerivedNCCJoins(mlir::Operation *root) {
  llvm::SmallVector<SyncNCCJoinOp, 16> joins;
  root->walk([&](SyncNCCJoinOp join) {
    if (!join.getScheduleBoundary())
      joins.push_back(join);
  });
  for (SyncNCCJoinOp join : llvm::reverse(joins))
    join.erase();
  llvm::SmallVector<mlir::LLVM::FenceOp> releases;
  root->walk([&](mlir::LLVM::FenceOp fence) {
    if (fence.getOrdering() == mlir::LLVM::AtomicOrdering::release &&
        fence.getSyncscope() == kTargetKcoreReleaseScope)
      releases.push_back(fence);
  });
  for (auto release : releases)
    release.erase();
  bool changed = true;
  while (changed) {
    changed = false;
    root->walk<mlir::WalkOrder::PostOrder>([&](mlir::Operation *op) {
      if (op != root && mlir::isa<mlir::scf::IfOp, mlir::arith::CmpIOp,
                                  mlir::arith::AndIOp>(op) &&
          mlir::isOpTriviallyDead(op)) {
        op->erase();
        changed = true;
      }
    });
  }
}

static mlir::LogicalResult
normalizeRequiredNCCJoinsInPlace(mlir::func::FuncOp function,
                                 bool rebuildDerivedJoins) {
  if (rebuildDerivedJoins)
    eraseDerivedNCCJoins(function.getOperation());
  return RequiredNCCJoinPlacement().run(function);
}

static mlir::LogicalResult
normalizeRequiredNCCJoinsInPlace(mlir::ModuleOp module,
                                 bool rebuildDerivedJoins) {
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    if (mlir::failed(
            normalizeRequiredNCCJoinsInPlace(function, rebuildDerivedJoins)))
      return mlir::failure();
  return mlir::success();
}

struct PlaceRequiredNCCJoinsPass
    : public wafer::impl::PlaceRequiredNCCJoinsPassBase<
          PlaceRequiredNCCJoinsPass> {
  using wafer::impl::PlaceRequiredNCCJoinsPassBase<
      PlaceRequiredNCCJoinsPass>::PlaceRequiredNCCJoinsPassBase;

  void runOnOperation() final {
    unsigned before = 0;
    getOperation().walk([&](SyncNCCJoinOp) { ++before; });
    if (mlir::succeeded(normalizeRequiredNCCJoinsInPlace(
            getOperation(), /*rebuildDerivedJoins=*/false))) {
      unsigned after = 0;
      getOperation().walk([&](SyncNCCJoinOp) { ++after; });
      if (after > before)
        numRequiredJoinsInserted += after - before;
      return;
    }
    signalPassFailure();
  }
};

struct RebuildRequiredNCCJoinsPass
    : public wafer::impl::RebuildRequiredNCCJoinsPassBase<
          RebuildRequiredNCCJoinsPass> {
  using wafer::impl::RebuildRequiredNCCJoinsPassBase<
      RebuildRequiredNCCJoinsPass>::RebuildRequiredNCCJoinsPassBase;

  void runOnOperation() final {
    unsigned before = 0;
    getOperation().walk([&](SyncNCCJoinOp) { ++before; });
    if (mlir::succeeded(normalizeRequiredNCCJoinsInPlace(
            getOperation(), /*rebuildDerivedJoins=*/true))) {
      unsigned after = 0;
      getOperation().walk([&](SyncNCCJoinOp) { ++after; });
      numDerivedJoinsRemoved += before;
      numRequiredJoinsInserted += after;
      return;
    }
    signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult wafer::placeRequiredNCCJoins(mlir::func::FuncOp function) {
  if (!function)
    return mlir::failure();
  wafer::support::ScopedCompileTimingSpan timing(
      "lowering-phase", "tile-region-to-instr", "required-ncc-join-placement");
  if (mlir::failed(
          normalizeRequiredNCCJoinsInPlace(function,
                                           /*rebuildDerivedJoins=*/false))) {
    timing.markFailed();
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult wafer::placeRequiredNCCJoins(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  return normalizeRequiredNCCJoinsInPlace(module,
                                          /*rebuildDerivedJoins=*/false);
}

mlir::LogicalResult
wafer::rebuildRequiredNCCJoins(mlir::func::FuncOp function) {
  if (!function)
    return mlir::failure();
  wafer::support::ScopedCompileTimingSpan timing(
      "lowering-phase", "tile-region-to-instr", "fresh-ncc-join-rebuild");
  if (mlir::failed(
          normalizeRequiredNCCJoinsInPlace(function,
                                           /*rebuildDerivedJoins=*/true))) {
    timing.markFailed();
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult wafer::rebuildRequiredNCCJoins(mlir::ModuleOp module) {
  if (!module)
    return mlir::failure();
  return normalizeRequiredNCCJoinsInPlace(module, /*rebuildDerivedJoins=*/true);
}

mlir::LogicalResult
wafer::rebuildRequiredNCCJoinsForIsolatedTileRegion(TileRegionOp tileRegion) {
  if (!tileRegion)
    return mlir::failure();
  eraseDerivedNCCJoins(tileRegion.getOperation());
  return RequiredNCCJoinPlacement().run(tileRegion);
}

mlir::LogicalResult wafer::detail::placeRequiredNCCJoinsInPrivateFunction(
    mlir::func::FuncOp function) {
  if (!function)
    return mlir::failure();
  return normalizeRequiredNCCJoinsInPlace(function,
                                          /*rebuildDerivedJoins=*/false);
}
