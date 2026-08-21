//===- CanonicalFeasibilityProof.cpp - Close canonical resources ------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalFeasibilityProof.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"
#include "Wafer/Transforms/MemoryPlanning/StaticMemoryPacking.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

using memory_planning::detail::ArenaRange;
using memory_planning::detail::PackingBackend;
using memory_planning::detail::PackingConflict;
using memory_planning::detail::PackingResult;
using memory_planning::detail::PackingStatus;
using memory_planning::detail::StaticPackingDemand;
using memory_planning::detail::StaticPackingProblem;

CanonicalFeasibilityOutcome broken(BrokenCanonicalFeasibilityReason reason,
                                   llvm::StringRef detail) {
  return BrokenCanonicalFeasibility{reason, detail.str()};
}

struct Footprint {
  int64_t bytes = 0;
  int64_t alignment = 1;
};

enum class FootprintStatus { Success, Unsupported, Overflow };

std::optional<int64_t> alignUp(int64_t value, int64_t alignment) {
  if (value < 0 || alignment <= 0)
    return std::nullopt;
  int64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  int64_t result = 0;
  if (llvm::AddOverflow(value, alignment - remainder, result))
    return std::nullopt;
  return result;
}

std::optional<int64_t> checkedLCM(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0)
    return std::nullopt;
  int64_t gcd = std::gcd(lhs, rhs);
  int64_t result = 0;
  if (llvm::MulOverflow(lhs / gcd, rhs, result))
    return std::nullopt;
  return result;
}

FootprintStatus computeFootprint(const analysis::ExactIndexSet &domain,
                                 mlir::Type elementType, MemLayout encoding,
                                 int64_t requiredAlignment, Footprint &result) {
  if (domain.getForm() != analysis::ExactIndexSetForm::BoxUnion ||
      domain.getBoxes().empty() || !elementType || !elementType.isIntOrFloat())
    return FootprintStatus::Unsupported;
  result = {};
  for (const analysis::StaticRectangularIndexSet &box : domain.getBoxes()) {
    auto type = mlir::MemRefType::get(
        box.sizes, elementType, mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(elementType.getContext(), MemorySpace::SPM, encoding));
    mlir::FailureOr<analysis::PhysicalLayoutRelation> layout =
        analysis::PhysicalLayoutRelation::create(type);
    if (mlir::failed(layout))
      return FootprintStatus::Unsupported;
    std::optional<int64_t> alignment =
        checkedLCM(layout->getMinimumAlignmentBytes(), requiredAlignment);
    if (!alignment)
      return FootprintStatus::Overflow;
    result.alignment = std::max(result.alignment, *alignment);
    std::optional<int64_t> begin = alignUp(result.bytes, *alignment);
    if (!begin || llvm::AddOverflow(*begin, layout->getPhysicalFootprintBytes(),
                                    result.bytes))
      return FootprintStatus::Overflow;
  }
  return FootprintStatus::Success;
}

analysis::RootRegionWorkId workOf(const ExecutionInstanceId &execution) {
  return std::visit([](const auto &source) { return source.work; },
                    execution.source);
}

bool intervalsOverlap(const FeasibilityResourceDemand &lhs,
                      const FeasibilityResourceDemand &rhs) {
  return lhs.begin <= rhs.end && rhs.begin <= lhs.end;
}

struct TileLess {
  bool operator()(TileId lhs, TileId rhs) const {
    return lhs.getValue() < rhs.getValue();
  }
};

struct DemandBuilder {
  std::map<FeasibilityResourceId, FeasibilityResourceDemand> demands;
  std::set<FeasibilityResourceConflict> forcedConflicts;
  std::optional<CanonicalFeasibilityOutcome> failure;

  void add(FeasibilityResourceDemand demand) {
    if (failure)
      return;
    if (!demands.try_emplace(demand.id, demand).second)
      failure = broken(BrokenCanonicalFeasibilityReason::DuplicateIdentity,
                       "canonical resource demand ID is duplicated");
  }

  void conflict(FeasibilityResourceId lhs, FeasibilityResourceId rhs) {
    if (lhs == rhs)
      return;
    if (rhs < lhs)
      std::swap(lhs, rhs);
    forcedConflicts.insert({std::move(lhs), std::move(rhs)});
  }
};

} // namespace

CanonicalFeasibilityOutcome buildCanonicalFeasibilityProof(
    const CanonicalStorageCoordinate &storage,
    const CanonicalMovementCoordinate &movements,
    const CanonicalScheduleCoordinate &schedule,
    const CanonicalAttentionWorkCoordinate &attention,
    const TargetMemoryPolicy &memory,
    const CanonicalFeasibilityOptions &options) {
  if (memory.spmBase < 0 || memory.spmLimit <= memory.spmBase ||
      memory.spmAlignment <= 0 || memory.ddrCapacityBytes < 0 ||
      memory.ddrLargestContiguousBytes < 0 || memory.ddrAlignmentBytes <= 0)
    return broken(BrokenCanonicalFeasibilityReason::InvalidResourceProblem,
                  "target memory policy is invalid");
  if (storage.plan.storageObjects.empty() || storage.resources.empty() ||
      storage.lifetimes.empty() || schedule.plan.order.empty())
    return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                  "canonical resource inputs are incomplete");

  std::map<ScheduleNodeId, int64_t> schedulePositions;
  for (auto [position, node] : llvm::enumerate(schedule.plan.order)) {
    if (position > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        !schedulePositions.try_emplace(node, static_cast<int64_t>(position))
             .second)
      return broken(BrokenCanonicalFeasibilityReason::DuplicateIdentity,
                    "canonical schedule node is duplicated");
  }
  std::set<ScheduleNodeId> workerNodes;
  for (const ScheduleWorkerBinding &binding : schedule.plan.workerBindings)
    if (binding.worker != NCCWorker::Worker0 ||
        !workerNodes.insert(binding.node).second)
      return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                    "canonical worker coverage is invalid");
  if (workerNodes.size() != schedulePositions.size())
    return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                  "canonical schedule order/worker coverage differs");
  for (const auto &[node, position] : schedulePositions)
    if (!workerNodes.count(node))
      return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                    "canonical schedule node has no worker binding");

  std::map<AttentionWorkScopeId, std::vector<AttentionActionId>> scopeActions;
  std::set<AttentionActionId> attentionActions;
  for (const AttentionWorkDescription &root : attention.roots)
    for (const AttentionActionDescription &action : root.actions) {
      if (!attentionActions.insert(action.id).second)
        return broken(BrokenCanonicalFeasibilityReason::DuplicateIdentity,
                      "attention action ID is duplicated");
      scopeActions[action.id.scope].push_back(action.id);
    }
  size_t maximumActions = 0;
  for (auto &[scope, actions] : scopeActions) {
    llvm::sort(actions);
    maximumActions = std::max(maximumActions, actions.size());
  }
  if (maximumActions >
      static_cast<size_t>(std::numeric_limits<int64_t>::max() - 2))
    return broken(BrokenCanonicalFeasibilityReason::ArithmeticOverflow,
                  "attention action count overflows event scaling");
  int64_t stride = static_cast<int64_t>(maximumActions) + 2;
  if (stride <= 1)
    stride = 2;
  auto scaledBegin = [&](const ScheduleNodeId &node) -> std::optional<int64_t> {
    auto position = schedulePositions.find(node);
    if (position == schedulePositions.end())
      return std::nullopt;
    int64_t result = 0;
    if (llvm::MulOverflow(position->second, stride, result))
      return std::nullopt;
    return result;
  };
  auto scaledEnd = [&](const ScheduleNodeId &node) -> std::optional<int64_t> {
    std::optional<int64_t> begin = scaledBegin(node);
    int64_t result = 0;
    if (!begin || llvm::AddOverflow(*begin, stride - 1, result))
      return std::nullopt;
    return result;
  };

  std::map<StorageObjectId, const StorageObjectPlan *> storageObjects;
  for (const StorageObjectPlan &object : storage.plan.storageObjects)
    if (!storageObjects.try_emplace(object.id, &object).second)
      return broken(BrokenCanonicalFeasibilityReason::DuplicateIdentity,
                    "storage object ID is duplicated");
  std::map<StorageObjectId, const StorageResourceDescription *>
      storageResources;
  for (const StorageResourceDescription &resource : storage.resources)
    if (!storageResources.try_emplace(resource.object, &resource).second)
      return broken(BrokenCanonicalFeasibilityReason::DuplicateIdentity,
                    "storage resource ID is duplicated");
  std::map<StorageObjectId, const StorageLifetimeDescription *>
      storageLifetimes;
  for (const StorageLifetimeDescription &lifetime : storage.lifetimes)
    if (!storageLifetimes.try_emplace(lifetime.object, &lifetime).second)
      return broken(BrokenCanonicalFeasibilityReason::DuplicateIdentity,
                    "storage lifetime ID is duplicated");
  if (storageObjects.size() != storageResources.size() ||
      storageObjects.size() != storageLifetimes.size())
    return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                  "storage object/resource/lifetime coverage differs");

  DemandBuilder demandBuilder;
  for (const auto &[id, object] : storageObjects) {
    auto resource = storageResources.find(id);
    auto lifetime = storageLifetimes.find(id);
    if (resource == storageResources.end() ||
        lifetime == storageLifetimes.end() || lifetime->second->uses.empty())
      return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                    "storage object is missing resource or lifetime");
    std::optional<int64_t> begin = scaledBegin(lifetime->second->definition);
    if (!begin)
      return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                    "storage definition is absent from schedule");
    int64_t end = *begin;
    for (const StorageAccessSite &use : lifetime->second->uses) {
      std::optional<int64_t> useEnd = scaledEnd(use);
      if (!useEnd || *useEnd < *begin)
        return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                      "storage use is missing or precedes definition");
      end = std::max(end, *useEnd);
    }
    Footprint footprint;
    FootprintStatus status = computeFootprint(
        resource->second->exactDomain, resource->second->elementType,
        resource->second->encoding, memory.spmAlignment, footprint);
    FeasibilityResourceId resourceId{FeasibilityResourceOrigin{id}};
    if (status == FootprintStatus::Unsupported)
      return UnsupportedCanonicalFeasibility{
          UnsupportedCanonicalFeasibilityReason::UnsupportedElementOrEncoding,
          resourceId, "storage footprint is not representable"};
    if (status == FootprintStatus::Overflow)
      return broken(BrokenCanonicalFeasibilityReason::ArithmeticOverflow,
                    "storage footprint arithmetic overflows");
    demandBuilder.add({resourceId, object->tile, footprint.bytes,
                       footprint.alignment, *begin, end});
  }
  if (demandBuilder.failure)
    return std::move(*demandBuilder.failure);

  std::map<AttentionValueId, FeasibilityResourceId> attentionResources;
  for (const AttentionWorkDescription &root : attention.roots) {
    std::map<AttentionWorkScopeId, std::map<AttentionActionId, int64_t>>
        localPositions;
    for (const auto &[scope, actions] : scopeActions) {
      if (workOf(scope.execution).root != root.root)
        continue;
      std::optional<int64_t> base = scaledBegin(scope.execution);
      if (!base)
        continue;
      for (auto [index, action] : llvm::enumerate(actions)) {
        int64_t position = 0;
        if (index + 1 >
                static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
            llvm::AddOverflow(*base, static_cast<int64_t>(index + 1), position))
          return broken(BrokenCanonicalFeasibilityReason::ArithmeticOverflow,
                        "attention action position overflows");
        localPositions[scope].emplace(action, position);
      }
    }

    for (const AttentionValueDescription &value : root.values) {
      if (value.physicalVersion.has_value() != value.storage.has_value())
        return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                      "attention physical/storage binding is partial");
      if (value.physicalVersion) {
        FeasibilityResourceId id{FeasibilityResourceOrigin{*value.storage}};
        if (!demandBuilder.demands.count(id))
          return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                        "attention physical value has no storage demand");
        attentionResources.emplace(value.id, id);
        continue;
      }
      auto positions = localPositions.find(value.id.scope);
      if (positions == localPositions.end())
        return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                      "attention scratch scope is absent from schedule");
      std::optional<int64_t> begin;
      std::optional<int64_t> end;
      for (const AttentionActionDescription &action : root.actions) {
        if (!(action.id.scope == value.id.scope))
          continue;
        auto position = positions->second.find(action.id);
        if (position == positions->second.end())
          return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                        "attention action has no local position");
        if (llvm::is_contained(action.outputs, value.id))
          begin = begin ? std::min(*begin, position->second) : position->second;
        if (llvm::is_contained(action.inputs, value.id))
          end = end ? std::max(*end, position->second) : position->second;
      }
      if (!begin || !end || *end < *begin)
        return broken(BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                      "attention scratch definition/use is incomplete");
      Footprint footprint;
      FootprintStatus status =
          computeFootprint(value.exactDomain, value.elementType,
                           MemLayout::Tensor, memory.spmAlignment, footprint);
      FeasibilityResourceId id{FeasibilityResourceOrigin{value.id}};
      if (status == FootprintStatus::Unsupported)
        return UnsupportedCanonicalFeasibility{
            UnsupportedCanonicalFeasibilityReason::UnsupportedElementOrEncoding,
            id, "attention scratch footprint is not representable"};
      if (status == FootprintStatus::Overflow)
        return broken(BrokenCanonicalFeasibilityReason::ArithmeticOverflow,
                      "attention scratch footprint arithmetic overflows");
      demandBuilder.add({id, workOf(value.id.scope.execution).tile,
                         footprint.bytes, footprint.alignment, *begin, *end});
      attentionResources.emplace(value.id, id);
    }
    for (const AttentionSimultaneousValueGroup &group : root.simultaneousValues)
      for (size_t lhs = 0; lhs < group.values.size(); ++lhs)
        for (size_t rhs = lhs + 1; rhs < group.values.size(); ++rhs) {
          auto lhsResource = attentionResources.find(group.values[lhs]);
          auto rhsResource = attentionResources.find(group.values[rhs]);
          if (lhsResource == attentionResources.end() ||
              rhsResource == attentionResources.end())
            return broken(
                BrokenCanonicalFeasibilityReason::PlanCoverageMismatch,
                "attention simultaneous group has no resource demand");
          demandBuilder.conflict(lhsResource->second, rhsResource->second);
        }
  }
  if (demandBuilder.failure)
    return std::move(*demandBuilder.failure);

  std::vector<FeasibilityResourceDemand> allDemands;
  allDemands.reserve(demandBuilder.demands.size());
  for (const auto &[id, demand] : demandBuilder.demands)
    allDemands.push_back(demand);
  for (size_t lhs = 0; lhs < allDemands.size(); ++lhs)
    for (size_t rhs = lhs + 1; rhs < allDemands.size(); ++rhs)
      if (allDemands[lhs].tile == allDemands[rhs].tile &&
          intervalsOverlap(allDemands[lhs], allDemands[rhs]))
        demandBuilder.conflict(allDemands[lhs].id, allDemands[rhs].id);

  CanonicalFeasibilityCoordinate coordinate;
  std::map<TileId, std::vector<FeasibilityResourceDemand>, TileLess>
      demandsByTile;
  for (const FeasibilityResourceDemand &demand : allDemands)
    demandsByTile[demand.tile].push_back(demand);
  for (auto &[tile, demands] : demandsByTile) {
    llvm::sort(demands, [](const auto &lhs, const auto &rhs) {
      return lhs.id < rhs.id;
    });
    TileSPMResourceProblem problem;
    problem.tile = tile;
    problem.arenaBegin = memory.spmBase;
    problem.arenaEnd = memory.spmLimit;
    problem.demands = demands;
    std::set<FeasibilityResourceId> ids;
    for (const FeasibilityResourceDemand &demand : demands)
      ids.insert(demand.id);
    for (const FeasibilityResourceConflict &conflict :
         demandBuilder.forcedConflicts)
      if (ids.count(conflict.lhs) && ids.count(conflict.rhs))
        problem.conflicts.push_back(conflict);
    coordinate.problem.tileSPM.push_back(problem);

    int64_t capacity = memory.spmLimit - memory.spmBase;
    for (const FeasibilityResourceDemand &demand : demands)
      if (demand.sizeBytes > capacity)
        return CanonicalResourceRejection{
            CanonicalResourceRejectionReason::SPMCapacity,
            tile,
            {demand.id},
            std::nullopt,
            demand.sizeBytes,
            capacity};

    StaticPackingProblem packing;
    packing.arena = ArenaRange{memory.spmBase, memory.spmLimit};
    std::map<FeasibilityResourceId, unsigned> indices;
    for (auto [index, demand] : llvm::enumerate(demands)) {
      indices.emplace(demand.id, static_cast<unsigned>(index));
      int64_t span = 0;
      if (llvm::SubOverflow(demand.end, demand.begin, span))
        return broken(BrokenCanonicalFeasibilityReason::ArithmeticOverflow,
                      "resource lifetime span overflows");
      packing.demands.push_back(
          StaticPackingDemand{demand.sizeBytes, demand.alignmentBytes, span,
                              demand.begin, static_cast<unsigned>(index)});
    }
    for (const FeasibilityResourceConflict &conflict : problem.conflicts) {
      unsigned lhs = indices.at(conflict.lhs);
      unsigned rhs = indices.at(conflict.rhs);
      if (rhs < lhs)
        std::swap(lhs, rhs);
      packing.conflicts.push_back(PackingConflict{lhs, rhs});
    }
    if (memory_planning::detail::validatePackingProblem(packing))
      return broken(BrokenCanonicalFeasibilityReason::InvalidResourceProblem,
                    "normalized SPM packing problem is invalid");
    PackingResult packed = memory_planning::detail::packStaticMemory(
        packing, options.packingSearchNodeBudget);
    if (packed.status == PackingStatus::Feasible) {
      if (memory_planning::detail::validatePlacements(packing,
                                                      packed.placements))
        return broken(BrokenCanonicalFeasibilityReason::InvalidSolverResult,
                      "SPM solver returned an invalid placement");
      ResourceProblemProof proof;
      proof.kind = ResourceProofKind::TileSPM;
      proof.method = packed.backend == PackingBackend::MiniMalloc
                         ? ResourceProofMethod::ExactSolverProof
                         : ResourceProofMethod::ValidatedPlacement;
      proof.tile = tile;
      for (const FeasibilityResourceDemand &demand : demands)
        proof.resources.push_back(demand.id);
      coordinate.proof.resourceProblems.push_back(std::move(proof));
      continue;
    }
    if (packed.status == PackingStatus::ProvenInfeasible) {
      std::vector<FeasibilityResourceId> causal;
      for (unsigned index : packed.capacityConflictDemandIndices)
        if (index < demands.size())
          causal.push_back(demands[index].id);
      for (unsigned index : packed.individuallyOversizedDemandIndices)
        if (index < demands.size())
          causal.push_back(demands[index].id);
      if (causal.empty())
        for (const FeasibilityResourceDemand &demand : demands)
          causal.push_back(demand.id);
      llvm::sort(causal);
      causal.erase(std::unique(causal.begin(), causal.end()), causal.end());
      std::optional<int64_t> required;
      if (!packed.capacityConflictDemandIndices.empty() ||
          !packed.individuallyOversizedDemandIndices.empty()) {
        required = 0;
        for (const FeasibilityResourceId &id : causal) {
          auto found = llvm::find_if(
              demands, [&](const auto &demand) { return demand.id == id; });
          if (found != demands.end() &&
              llvm::AddOverflow(*required, found->sizeBytes, *required))
            return broken(BrokenCanonicalFeasibilityReason::ArithmeticOverflow,
                          "capacity certificate byte sum overflows");
        }
      }
      return CanonicalResourceRejection{
          CanonicalResourceRejectionReason::SPMCapacity,
          tile,
          std::move(causal),
          std::nullopt,
          required,
          capacity};
    }
    if (packed.status == PackingStatus::ResourceExhausted ||
        packed.status == PackingStatus::HeuristicNoFit)
      return CanonicalFeasibilityIndeterminate{
          packed.status == PackingStatus::HeuristicNoFit
              ? CanonicalFeasibilityIndeterminateReason::HeuristicNoFit
              : CanonicalFeasibilityIndeterminateReason::ResourceWorkExhausted,
          tile,
          ids.empty()
              ? std::vector<FeasibilityResourceId>{}
              : std::vector<FeasibilityResourceId>(ids.begin(), ids.end())};
    return broken(
        packed.status == PackingStatus::ArithmeticOverflow
            ? BrokenCanonicalFeasibilityReason::ArithmeticOverflow
        : packed.status == PackingStatus::InvalidSolverResult
            ? BrokenCanonicalFeasibilityReason::InvalidSolverResult
            : BrokenCanonicalFeasibilityReason::InvalidResourceProblem,
        "SPM packing failed with an invalid status");
  }

  std::set<MovementActionId> movementActions;
  for (const MovementResourceDescription &movement : movements.resources) {
    if (!movementActions.insert(movement.action).second)
      return broken(BrokenCanonicalFeasibilityReason::DuplicateIdentity,
                    "movement action resource is duplicated");
    Footprint payload;
    FootprintStatus status =
        computeFootprint(movement.exactDomain, movement.elementType,
                         MemLayout::Tensor, memory.ddrAlignmentBytes, payload);
    if (status == FootprintStatus::Unsupported)
      return UnsupportedCanonicalFeasibility{
          UnsupportedCanonicalFeasibilityReason::UnsupportedElementOrEncoding,
          std::nullopt, "movement payload footprint is not representable"};
    if (status == FootprintStatus::Overflow)
      return broken(BrokenCanonicalFeasibilityReason::ArithmeticOverflow,
                    "movement payload footprint arithmetic overflows");
    int64_t limit =
        std::min(memory.ddrCapacityBytes, memory.ddrLargestContiguousBytes);
    coordinate.problem.ddrPayloads.push_back(
        {movement.action, payload.bytes, limit});
    if (payload.bytes > limit)
      return CanonicalResourceRejection{
          CanonicalResourceRejectionReason::DDRPayloadLimit,
          std::nullopt,
          {},
          movement.action,
          payload.bytes,
          limit};
  }
  llvm::sort(coordinate.problem.ddrPayloads,
             [](const DDRPayloadProblem &lhs, const DDRPayloadProblem &rhs) {
               return lhs.action < rhs.action;
             });
  ResourceProblemProof ddrProof;
  ddrProof.kind = ResourceProofKind::DDRPayload;
  ddrProof.method = ResourceProofMethod::DirectCapacityProof;
  ddrProof.movements.assign(movementActions.begin(), movementActions.end());
  coordinate.proof.resourceProblems.push_back(std::move(ddrProof));

  coordinate.problem.scheduleNodes.assign(workerNodes.begin(),
                                          workerNodes.end());
  ResourceProblemProof scheduleProof;
  scheduleProof.kind = ResourceProofKind::ScheduleCoverage;
  scheduleProof.method = ResourceProofMethod::DirectCapacityProof;
  scheduleProof.scheduleNodes = coordinate.problem.scheduleNodes;
  coordinate.proof.resourceProblems.push_back(std::move(scheduleProof));

  coordinate.problem.attentionActions.assign(attentionActions.begin(),
                                             attentionActions.end());
  ResourceProblemProof attentionProof;
  attentionProof.kind = ResourceProofKind::AttentionCoverage;
  attentionProof.method = ResourceProofMethod::DirectCapacityProof;
  attentionProof.attentionActions = coordinate.problem.attentionActions;
  coordinate.proof.resourceProblems.push_back(std::move(attentionProof));

  coordinate.proof.dependencyKey.spmBegin = memory.spmBase;
  coordinate.proof.dependencyKey.spmEnd = memory.spmLimit;
  coordinate.proof.dependencyKey.spmAlignment = memory.spmAlignment;
  coordinate.proof.dependencyKey.ddrCapacityBytes = memory.ddrCapacityBytes;
  coordinate.proof.dependencyKey.ddrLargestContiguousBytes =
      memory.ddrLargestContiguousBytes;
  coordinate.proof.dependencyKey.ddrAlignmentBytes = memory.ddrAlignmentBytes;
  for (const auto &[id, demand] : demandBuilder.demands)
    coordinate.proof.dependencyKey.resources.push_back(id);
  coordinate.proof.dependencyKey.movements.assign(movementActions.begin(),
                                                  movementActions.end());
  coordinate.proof.dependencyKey.scheduleNodes =
      coordinate.problem.scheduleNodes;
  coordinate.proof.dependencyKey.attentionActions =
      coordinate.problem.attentionActions;

  llvm::sort(coordinate.problem.tileSPM, [](const TileSPMResourceProblem &lhs,
                                            const TileSPMResourceProblem &rhs) {
    return lhs.tile.getValue() < rhs.tile.getValue();
  });
  llvm::sort(
      coordinate.proof.resourceProblems,
      [](const ResourceProblemProof &lhs, const ResourceProblemProof &rhs) {
        if (lhs.kind != rhs.kind)
          return lhs.kind < rhs.kind;
        if (lhs.tile.has_value() != rhs.tile.has_value())
          return lhs.tile.has_value() < rhs.tile.has_value();
        return lhs.tile && lhs.tile->getValue() < rhs.tile->getValue();
      });
  return coordinate;
}

} // namespace wafer::compiler::detail
