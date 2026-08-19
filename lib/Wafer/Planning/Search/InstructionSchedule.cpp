//===- InstructionSchedule.cpp - Exact Instr schedule domain ----------===//

#include "Wafer/Planning/Search/InstructionSchedule.h"

#include "Wafer/Analysis/Scheduling/NCCCompletionAnalysis.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <functional>
#include <map>

namespace wafer::compiler::detail {
namespace {

struct BufferAccess {
  mlir::Value root;
  bool write = false;
};

mlir::Value getStorageRoot(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    if (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(
            value.getDefiningOp())) {
      value = view.getViewSource();
      continue;
    }
    break;
  }
  return value;
}

void collectBufferAccesses(mlir::Operation *operation,
                           llvm::SmallVectorImpl<BufferAccess> &accesses) {
  auto record = [&](mlir::Value value, bool write) {
    if (!value || !mlir::isa<mlir::BaseMemRefType>(value.getType()))
      return;
    value = getStorageRoot(value);
    auto found = llvm::find_if(accesses, [&](const BufferAccess &access) {
      return access.root == value;
    });
    if (found == accesses.end())
      accesses.push_back({value, write});
    else
      found->write |= write;
  };
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation)) {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
    effects.getEffects(instances);
    for (const auto &effect : instances)
      if (mlir::Value value = effect.getValue())
        record(value,
               !llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect()));
  }
  if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation)) {
    for (mlir::Value token : wait.getTokens()) {
      if (auto send = token.getDefiningOp<InstrDTESendOp>())
        record(send.getBuffer(), false);
      else if (auto receive = token.getDefiningOp<InstrDTERecvOp>())
        record(receive.getBuffer(), true);
    }
  }
}

bool eventsConflict(mlir::Operation *lhs, mlir::Operation *rhs) {
  llvm::SmallVector<BufferAccess, 4> lhsAccesses;
  llvm::SmallVector<BufferAccess, 4> rhsAccesses;
  collectBufferAccesses(lhs, lhsAccesses);
  collectBufferAccesses(rhs, rhsAccesses);
  for (const BufferAccess &left : lhsAccesses)
    for (const BufferAccess &right : rhsAccesses)
      if (left.root == right.root && (left.write || right.write))
        return true;
  return false;
}

bool dependsOnEvent(mlir::Operation *operation, mlir::Operation *event) {
  llvm::SmallVector<mlir::Operation *, 8> worklist;
  llvm::DenseSet<mlir::Operation *> visited;
  for (mlir::Value operand : operation->getOperands())
    if (mlir::Operation *definition = operand.getDefiningOp())
      worklist.push_back(definition);
  while (!worklist.empty()) {
    mlir::Operation *definition = worklist.pop_back_val();
    if (definition == event)
      return true;
    if (!definition || definition->getBlock() != operation->getBlock() ||
        !visited.insert(definition).second ||
        mlir::isa<WaferInstructionOpInterface>(definition) ||
        definition->getNumRegions() != 0 ||
        !mlir::isMemoryEffectFree(definition))
      continue;
    for (mlir::Value operand : definition->getOperands())
      if (mlir::Operation *predecessor = operand.getDefiningOp())
        worklist.push_back(predecessor);
  }
  return false;
}

bool isSynchronousBoundary(mlir::Operation *operation) {
  return getNCCOperationCompletion(operation).kind ==
         NCCCompletionKind::SynchronousWriteback;
}

bool isInstructionEvent(mlir::Operation *operation) {
  return mlir::isa<WaferInstructionOpInterface>(operation) &&
         !mlir::isa<SyncNCCJoinOp>(operation);
}

bool isWindowBarrier(mlir::Operation *operation) {
  if (operation->hasTrait<mlir::OpTrait::IsTerminator>() ||
      operation->getNumRegions() != 0 ||
      mlir::isa<mlir::CallOpInterface>(operation))
    return true;
  if (isInstructionEvent(operation) || mlir::isa<SyncNCCJoinOp>(operation))
    return false;
  return !mlir::isMemoryEffectFree(operation);
}

void addPredecessor(
    llvm::SmallVectorImpl<llvm::SmallVector<unsigned, 4>> &predecessors,
    unsigned predecessor, unsigned successor) {
  if (predecessor != successor &&
      !llvm::is_contained(predecessors[successor], predecessor))
    predecessors[successor].push_back(predecessor);
}

void buildPredecessors(
    llvm::ArrayRef<mlir::Operation *> operations,
    llvm::SmallVectorImpl<llvm::SmallVector<unsigned, 4>> &predecessors) {
  predecessors.resize(operations.size());
  for (unsigned later = 0; later < operations.size(); ++later) {
    for (unsigned earlier = 0; earlier < later; ++earlier) {
      if (dependsOnEvent(operations[later], operations[earlier]) ||
          eventsConflict(operations[earlier], operations[later]) ||
          isSynchronousBoundary(operations[earlier]) ||
          isSynchronousBoundary(operations[later]))
        addPredecessor(predecessors, earlier, later);
    }
  }
}

llvm::SmallVector<unsigned, 16> getFirstTopologicalOrder(
    const CardInstructionScheduleDomain::WindowDomain &window) {
  const unsigned count = window.operations.size();
  llvm::SmallVector<unsigned, 16> indegrees(count, 0);
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> successors(count);
  for (unsigned successor = 0; successor < count; ++successor)
    for (unsigned predecessor : window.predecessors[successor]) {
      ++indegrees[successor];
      successors[predecessor].push_back(successor);
    }
  llvm::SmallVector<unsigned, 16> order;
  llvm::SmallVector<uint8_t, 16> emitted(count, 0);
  while (order.size() < count) {
    unsigned selected = count;
    for (unsigned index = 0; index < count; ++index)
      if (!emitted[index] && indegrees[index] == 0) {
        selected = index;
        break;
      }
    if (selected == count)
      return {};
    emitted[selected] = 1;
    order.push_back(selected);
    for (unsigned successor : successors[selected])
      --indegrees[successor];
  }
  return order;
}

std::optional<llvm::SmallVector<unsigned, 16>> getNextTopologicalOrder(
    const CardInstructionScheduleDomain::WindowDomain &window,
    llvm::ArrayRef<unsigned> current) {
  const unsigned count = window.operations.size();
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> successors(count);
  for (unsigned successor = 0; successor < count; ++successor)
    for (unsigned predecessor : window.predecessors[successor])
      successors[predecessor].push_back(successor);

  for (size_t reverse = 0; reverse < current.size(); ++reverse) {
    const size_t position = current.size() - reverse - 1;
    llvm::SmallVector<unsigned, 16> indegrees(count, 0);
    llvm::SmallVector<uint8_t, 16> emitted(count, 0);
    for (unsigned successor = 0; successor < count; ++successor)
      indegrees[successor] = window.predecessors[successor].size();
    auto emit = [&](unsigned selected) {
      emitted[selected] = 1;
      for (unsigned successor : successors[selected])
        --indegrees[successor];
    };
    bool prefixValid = true;
    for (size_t prefix = 0; prefix < position; ++prefix) {
      unsigned selected = current[prefix];
      if (selected >= count || emitted[selected] || indegrees[selected] != 0) {
        prefixValid = false;
        break;
      }
      emit(selected);
    }
    if (!prefixValid)
      return std::nullopt;
    unsigned replacement = count;
    for (unsigned candidate = current[position] + 1; candidate < count;
         ++candidate)
      if (!emitted[candidate] && indegrees[candidate] == 0) {
        replacement = candidate;
        break;
      }
    if (replacement == count)
      continue;
    llvm::SmallVector<unsigned, 16> next(current.begin(),
                                         current.begin() + position);
    next.push_back(replacement);
    emit(replacement);
    while (next.size() < count) {
      unsigned selected = count;
      for (unsigned candidate = 0; candidate < count; ++candidate)
        if (!emitted[candidate] && indegrees[candidate] == 0) {
          selected = candidate;
          break;
        }
      if (selected == count)
        return std::nullopt;
      next.push_back(selected);
      emit(selected);
    }
    return next;
  }
  return std::nullopt;
}

std::optional<InstructionResourceKind> getEngineResource(InstrFamily family) {
  switch (family) {
  case InstrFamily::CT:
    return InstructionResourceKind::CT;
  case InstrFamily::NE:
    return InstructionResourceKind::NE;
  case InstrFamily::RDMA:
    return InstructionResourceKind::RDMA;
  case InstrFamily::WDMA:
    return InstructionResourceKind::WDMA;
  case InstrFamily::TDMA:
    return InstructionResourceKind::TDMA;
  case InstrFamily::DTE:
    return InstructionResourceKind::DirectDTE;
  }
  return std::nullopt;
}

void appendResource(llvm::SmallVectorImpl<InstructionResourceKey> &resources,
                    InstructionResourceKey resource) {
  if (!llvm::is_contained(resources, resource))
    resources.push_back(resource);
}

InstructionResourceEvent getResourceEvent(TileId tile,
                                          mlir::Operation *operation) {
  InstructionResourceEvent event{tile, operation, {}};
  if (auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation))
    if (std::optional<InstructionResourceKind> engine =
            getEngineResource(instruction.getInstructionFamily()))
      appendResource(event.resources, {*engine, tile.getValue(), 0});
  NCCOperationCompletion completion = getNCCOperationCompletion(operation);
  if (completion.issueWorker)
    appendResource(event.resources,
                   {InstructionResourceKind::NCCWorker, tile.getValue(),
                    static_cast<int64_t>(*completion.issueWorker)});
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation)) {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
    effects.getEffects(instances);
    for (const auto &effect : instances) {
      if (effect.getResource() == WaferSPMResource::get())
        appendResource(event.resources,
                       {InstructionResourceKind::TileSPM, tile.getValue(), 0});
      if (effect.getResource() == WaferDDRResource::get())
        appendResource(event.resources,
                       {InstructionResourceKind::CardDDR, 0, 0});
    }
  }
  if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
    appendResource(event.resources,
                   {InstructionResourceKind::DirectedPeerLink, tile.getValue(),
                    static_cast<int64_t>(send.getPeer())});
  if (auto receive = mlir::dyn_cast<InstrDTERecvOp>(operation))
    appendResource(event.resources,
                   {InstructionResourceKind::DirectedPeerLink,
                    static_cast<int64_t>(receive.getPeer()), tile.getValue()});
  llvm::sort(event.resources);
  return event;
}

bool resourcesDisjoint(const InstructionResourceEvent &lhs,
                       const InstructionResourceEvent &rhs) {
  return llvm::none_of(lhs.resources, [&](const auto &left) {
    if (left.kind == InstructionResourceKind::TileSPM)
      return false;
    return llvm::is_contained(rhs.resources, left);
  });
}

} // namespace

mlir::FailureOr<CardInstructionScheduleDomain>
CardInstructionScheduleDomain::create(
    llvm::ArrayRef<TileInstructionModule> modules, std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<CardInstructionScheduleDomain> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  if (modules.empty())
    return fail("instruction schedule requires a nonempty Tile domain");
  llvm::DenseSet<int64_t> seenTiles;
  int64_t previousTile = -1;
  llvm::SmallVector<WindowDomain, 32> windows;
  llvm::SmallVector<WorkerDomain, 64> workers;
  llvm::SmallVector<OperationSnapshot, 128> snapshot;
  for (const TileInstructionModule &tile : modules) {
    if (!tile.module || tile.tile.getValue() <= previousTile ||
        !seenTiles.insert(tile.tile.getValue()).second ||
        mlir::failed(mlir::verify(tile.module)))
      return fail("instruction schedule received a malformed Tile module");
    previousTile = tile.tile.getValue();
    mlir::ModuleOp module = tile.module;
    bool hasPreexistingPlacement = false;
    bool hasNoncanonicalWorker = false;
    module.walk([&](mlir::Operation *operation) {
      snapshot.push_back(OperationSnapshot{
          operation, operation->getAttrDictionary(),
          llvm::SmallVector<mlir::Value, 4>(operation->operand_begin(),
                                            operation->operand_end()),
          llvm::SmallVector<mlir::Type, 2>(operation->result_type_begin(),
                                           operation->result_type_end())});
      if (mlir::isa<WaferNCCIssueOpInterface>(operation))
        workers.push_back({tile.tile, operation});
      if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation))
        hasPreexistingPlacement |=
            allocation->hasAttr(kWaferSPMOffsetAttrName) ||
            allocation->hasAttr(kWaferDDROffsetAttrName);
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
        hasPreexistingPlacement |= send.getBinding().has_value();
      if (auto receive = mlir::dyn_cast<InstrDTERecvOp>(operation))
        hasPreexistingPlacement |= receive.getBinding().has_value();
      if (std::optional<NCCWorker> worker = getNCCIssueWorker(operation))
        hasNoncanonicalWorker |= *worker != NCCWorker::Worker0;
    });
    if (hasPreexistingPlacement || hasNoncanonicalWorker)
      return fail("instruction schedule requires canonical unplaced worker0 "
                  "Instr input");
    std::function<void(mlir::Region &)> collectRegion;
    collectRegion = [&](mlir::Region &region) {
      for (mlir::Block &block : region) {
        llvm::SmallVector<mlir::Operation *, 16> current;
        auto flush = [&] {
          if (current.empty())
            return;
          WindowDomain window;
          window.tile = tile.tile;
          window.block = &block;
          window.operations = std::move(current);
          buildPredecessors(window.operations, window.predecessors);
          windows.push_back(std::move(window));
          current.clear();
        };
        for (mlir::Operation &operation : block) {
          if (mlir::isa<SyncNCCJoinOp>(operation))
            continue;
          if (isSynchronousBoundary(&operation)) {
            flush();
            current.push_back(&operation);
            flush();
            continue;
          }
          if (isWindowBarrier(&operation))
            flush();
          if (isInstructionEvent(&operation))
            current.push_back(&operation);
          for (mlir::Region &nested : operation.getRegions())
            collectRegion(nested);
        }
        flush();
      }
    };
    for (mlir::Region &region : module->getRegions())
      collectRegion(region);
  }
  for (const WindowDomain &window : windows)
    if (window.operations.empty() || getFirstTopologicalOrder(window).empty())
      return fail("instruction schedule dependency graph is cyclic");
  return CardInstructionScheduleDomain(
      std::move(windows), std::move(workers),
      llvm::SmallVector<TileInstructionModule, 16>(modules.begin(),
                                                   modules.end()),
      std::move(snapshot));
}

InstructionWindowOrder
CardInstructionScheduleDomain::getFirstOrder(const WindowDomain &window) const {
  InstructionWindowOrder result{window.tile, window.block, {}};
  for (unsigned index : getFirstTopologicalOrder(window))
    result.operations.push_back(window.operations[index]);
  return result;
}

bool CardInstructionScheduleDomain::contains(
    const WindowDomain &window, const InstructionWindowOrder &order) const {
  if (order.tile != window.tile || order.block != window.block ||
      order.operations.size() != window.operations.size())
    return false;
  llvm::DenseMap<mlir::Operation *, unsigned> positions;
  for (auto [position, operation] : llvm::enumerate(order.operations)) {
    if (!operation || !llvm::is_contained(window.operations, operation) ||
        !positions.try_emplace(operation, position).second)
      return false;
  }
  for (unsigned successor = 0; successor < window.operations.size();
       ++successor)
    for (unsigned predecessor : window.predecessors[successor])
      if (positions.lookup(window.operations[predecessor]) >=
          positions.lookup(window.operations[successor]))
        return false;
  return true;
}

bool CardInstructionScheduleDomain::isCurrent(
    llvm::ArrayRef<TileInstructionModule> current) const {
  if (!isEpochCurrent() || current.size() != modules.size())
    return false;
  for (auto [expected, actual] : llvm::zip_equal(modules, current))
    if (expected.tile != actual.tile || expected.module != actual.module)
      return false;
  return true;
}

bool CardInstructionScheduleDomain::isEpochCurrent() const {
  llvm::SmallVector<mlir::Operation *, 128> current;
  for (const TileInstructionModule &tile : modules) {
    if (!tile.module)
      return false;
    mlir::ModuleOp module = tile.module;
    module.walk(
        [&](mlir::Operation *operation) { current.push_back(operation); });
  }
  if (current.size() != snapshot.size())
    return false;
  for (auto [operation, saved] : llvm::zip_equal(current, snapshot))
    if (operation != saved.operation ||
        operation->getAttrDictionary() != saved.attributes ||
        !llvm::equal(operation->getOperands(), saved.operands) ||
        !llvm::equal(operation->getResultTypes(), saved.resultTypes))
      return false;
  return true;
}

mlir::FailureOr<std::optional<InstructionWindowOrder>>
CardInstructionScheduleDomain::getNextOrder(
    const WindowDomain &window, const InstructionWindowOrder &order) const {
  if (!contains(window, order))
    return mlir::failure();
  llvm::SmallVector<unsigned, 16> current;
  for (mlir::Operation *operation : order.operations) {
    auto found = llvm::find(window.operations, operation);
    current.push_back(static_cast<unsigned>(found - window.operations.begin()));
  }
  std::optional<llvm::SmallVector<unsigned, 16>> next =
      getNextTopologicalOrder(window, current);
  if (!next)
    return std::optional<InstructionWindowOrder>{};
  InstructionWindowOrder result{window.tile, window.block, {}};
  for (unsigned index : *next)
    result.operations.push_back(window.operations[index]);
  return std::optional<InstructionWindowOrder>(std::move(result));
}

CardInstructionScheduleAssignment
CardInstructionScheduleDomain::getFirstAssignment() const {
  CardInstructionScheduleAssignment result;
  for (const WindowDomain &window : windows)
    result.windows.push_back(getFirstOrder(window));
  for (const WorkerDomain &worker : workers)
    result.workers.push_back(
        {worker.tile, worker.operation, NCCWorker::Worker0});
  return result;
}

bool CardInstructionScheduleDomain::contains(
    const CardInstructionScheduleAssignment &assignment) const {
  if (!isEpochCurrent() || assignment.windows.size() != windows.size() ||
      assignment.workers.size() != workers.size())
    return false;
  for (auto [domain, order] : llvm::zip_equal(windows, assignment.windows))
    if (!contains(domain, order))
      return false;
  for (auto [domain, choice] : llvm::zip_equal(workers, assignment.workers))
    if (choice.tile != domain.tile || choice.operation != domain.operation ||
        static_cast<uint32_t>(choice.worker) >= kNCCWorkerCount)
      return false;
  return true;
}

mlir::FailureOr<std::optional<CardInstructionScheduleAssignment>>
CardInstructionScheduleDomain::getNextAssignment(
    const CardInstructionScheduleAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  CardInstructionScheduleAssignment next = assignment;
  for (size_t reverse = 0; reverse < workers.size(); ++reverse) {
    const size_t index = workers.size() - reverse - 1;
    uint32_t worker = static_cast<uint32_t>(next.workers[index].worker);
    if (worker + 1 >= kNCCWorkerCount)
      continue;
    next.workers[index].worker = static_cast<NCCWorker>(worker + 1);
    for (size_t reset = index + 1; reset < workers.size(); ++reset)
      next.workers[reset].worker = NCCWorker::Worker0;
    return std::optional<CardInstructionScheduleAssignment>(std::move(next));
  }
  for (InstructionWorkerChoice &worker : next.workers)
    worker.worker = NCCWorker::Worker0;
  for (size_t reverse = 0; reverse < windows.size(); ++reverse) {
    const size_t index = windows.size() - reverse - 1;
    auto advanced = getNextOrder(windows[index], next.windows[index]);
    if (mlir::failed(advanced))
      return mlir::failure();
    if (!*advanced)
      continue;
    next.windows[index] = std::move(**advanced);
    for (size_t reset = index + 1; reset < windows.size(); ++reset)
      next.windows[reset] = getFirstOrder(windows[reset]);
    return std::optional<CardInstructionScheduleAssignment>(std::move(next));
  }
  return std::optional<CardInstructionScheduleAssignment>{};
}

mlir::FailureOr<ScheduledInstructionModules>
applyInstructionSchedule(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules,
                         llvm::ArrayRef<TileId> tileIds,
                         const CardInstructionScheduleDomain &domain,
                         const CardInstructionScheduleAssignment &assignment,
                         std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<ScheduledInstructionModules> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  if (modules.size() != tileIds.size() || !domain.contains(assignment))
    return fail("instruction schedule apply received a stale assignment");
  llvm::SmallVector<TileInstructionModule, 16> current;
  for (auto [module, tile] : llvm::zip_equal(modules, tileIds))
    current.push_back({tile, module ? module.get() : mlir::ModuleOp{}});
  if (!domain.isCurrent(current))
    return fail("instruction schedule apply received a different IR epoch");
  for (const InstructionWorkerChoice &choice : assignment.workers)
    if (mlir::failed(setNCCIssueWorker(choice.operation, choice.worker)))
      return fail("instruction schedule could not apply a typed worker");
  for (const InstructionWindowOrder &window : assignment.windows) {
    mlir::Operation *last = window.operations.front();
    for (mlir::Operation *operation : window.operations)
      if (last->isBeforeInBlock(operation))
        last = operation;
    auto insertionPoint = last->getIterator();
    ++insertionPoint;
    for (mlir::Operation *operation : window.operations)
      operation->moveBefore(window.block, insertionPoint);
  }
  for (mlir::OwningOpRef<mlir::ModuleOp> &module : modules) {
    if (!module)
      return fail("instruction schedule lost an owned Tile module");
    llvm::SmallVector<SyncNCCJoinOp, 8> joins;
    (*module)->walk([&](SyncNCCJoinOp join) { joins.push_back(join); });
    for (SyncNCCJoinOp join : joins)
      join.erase();
    if (mlir::failed(rebuildRequiredNCCJoins(*module)) ||
        mlir::failed(mlir::verify(*module)))
      return fail("instruction schedule failed completion reconstruction");
  }
  return ScheduledInstructionModules{std::move(modules)};
}

mlir::FailureOr<CardInstructionResourceAnalysis>
analyzeInstructionResources(llvm::ArrayRef<TileInstructionModule> modules,
                            std::string *failureReason) {
  auto fail = [&](llvm::StringRef message)
      -> mlir::FailureOr<CardInstructionResourceAnalysis> {
    if (failureReason)
      *failureReason = message.str();
    return mlir::failure();
  };
  CardInstructionResourceAnalysis result;
  std::map<InstructionResourceKey, llvm::SmallVector<mlir::Operation *, 4>>
      shared;
  for (const TileInstructionModule &tile : modules) {
    mlir::ModuleOp module = tile.module;
    auto completion =
        analysis::NCCCompletionAnalysis::create(tile.module, failureReason);
    if (mlir::failed(completion))
      return mlir::failure();
    bool unresolved = false;
    module.walk([&](mlir::func::ReturnOp operation) {
      auto state = completion->getOperationState(operation);
      unresolved |= state && state->pendingBefore != 0;
    });
    if (unresolved)
      return fail("instruction resource analysis found unresolved NCC work");

    module.walk([&](mlir::Block *block) {
      std::array<llvm::SmallVector<mlir::Operation *, 4>, kNCCWorkerCount>
          pendingNCC;
      llvm::DenseMap<mlir::Value, mlir::Operation *> pendingDTE;
      for (mlir::Operation &operation : *block) {
        if (!isInstructionEvent(&operation) &&
            !mlir::isa<SyncNCCJoinOp>(operation))
          continue;
        InstructionResourceEvent event =
            getResourceEvent(tile.tile, &operation);
        for (const auto &pending : pendingNCC)
          for (mlir::Operation *issue : pending) {
            InstructionResourceEvent issueEvent =
                getResourceEvent(tile.tile, issue);
            if (resourcesDisjoint(issueEvent, event))
              result.overlapWitnesses.push_back({tile.tile, issue, &operation});
          }
        for (const auto &[token, issue] : pendingDTE) {
          (void)token;
          InstructionResourceEvent issueEvent =
              getResourceEvent(tile.tile, issue);
          if (resourcesDisjoint(issueEvent, event))
            result.overlapWitnesses.push_back({tile.tile, issue, &operation});
        }
        for (const InstructionResourceKey &resource : event.resources)
          if (resource.kind == InstructionResourceKind::CardDDR ||
              resource.kind == InstructionResourceKind::DirectedPeerLink)
            shared[resource].push_back(&operation);
        result.events.push_back(event);

        NCCOperationCompletion contract = getNCCOperationCompletion(&operation);
        if (contract.issueWorker &&
            contract.kind == NCCCompletionKind::OrderedAsynchronousIssue)
          pendingNCC[static_cast<uint32_t>(*contract.issueWorker)].push_back(
              &operation);
        if (contract.kind == NCCCompletionKind::ParticipantJoin ||
            contract.kind == NCCCompletionKind::SynchronousWriteback)
          for (unsigned worker = 0; worker < kNCCWorkerCount; ++worker)
            if ((contract.participantMask & (uint32_t{1} << worker)) != 0)
              pendingNCC[worker].clear();
        if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
          pendingDTE.try_emplace(send.getToken(), &operation);
        if (auto receive = mlir::dyn_cast<InstrDTERecvOp>(operation))
          pendingDTE.try_emplace(receive.getToken(), &operation);
        if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(operation))
          for (mlir::Value token : wait.getTokens())
            pendingDTE.erase(token);
      }
    });
  }
  for (auto &[resource, users] : shared) {
    llvm::SmallVector<mlir::Operation *, 4> unique;
    for (mlir::Operation *user : users)
      if (!llvm::is_contained(unique, user))
        unique.push_back(user);
    if (unique.size() > 1)
      result.sharedResources.push_back({resource, std::move(unique)});
  }
  return result;
}

} // namespace wafer::compiler::detail
