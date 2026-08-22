//===- CanonicalMovementPlan.cpp - Deterministic DDR carrier ----------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

CanonicalMovementPlanOutcome
broken(BrokenMovementPlanReason reason, llvm::StringRef detail,
       std::optional<analysis::RootRegionWorkId> work = std::nullopt) {
  return BrokenMovementPlan{reason, std::move(work), detail.str()};
}

CanonicalMovementPlanOutcome
unsupported(UnsupportedMovementFeature feature, llvm::StringRef detail,
            std::optional<analysis::RootRegionWorkId> work = std::nullopt) {
  return UnsupportedMovementPlan{feature, std::move(work), detail.str()};
}

struct MovementActionLess {
  bool operator()(const MovementActionId &lhs,
                  const MovementActionId &rhs) const {
    if (lhs.index() != rhs.index())
      return lhs.index() < rhs.index();
    if (const auto *value = std::get_if<ExternalLoadId>(&lhs))
      return *value < std::get<ExternalLoadId>(rhs);
    if (const auto *value = std::get_if<DDRBoundaryTransferId>(&lhs))
      return *value < std::get<DDRBoundaryTransferId>(rhs);
    if (const auto *value = std::get_if<ReductionGatherId>(&lhs))
      return *value < std::get<ReductionGatherId>(rhs);
    return std::get<ResultPublicationId>(lhs) <
           std::get<ResultPublicationId>(rhs);
  }
};

bool boxContains(const analysis::StaticRectangularIndexSet &outer,
                 const analysis::StaticRectangularIndexSet &inner) {
  if (outer.offsets.size() != inner.offsets.size())
    return false;
  for (auto [outerOffset, outerSize, innerOffset, innerSize] :
       llvm::zip_equal(outer.offsets, outer.sizes, inner.offsets, inner.sizes))
    if (innerOffset < outerOffset ||
        innerOffset + innerSize > outerOffset + outerSize)
      return false;
  return true;
}

bool finiteDomainContains(const analysis::ExactIndexSet &outer,
                          const analysis::ExactIndexSet &inner) {
  if (outer.getForm() != analysis::ExactIndexSetForm::BoxUnion ||
      inner.getForm() != analysis::ExactIndexSetForm::BoxUnion)
    return true;
  return llvm::all_of(inner.getBoxes(), [&](const auto &innerBox) {
    return llvm::any_of(outer.getBoxes(), [&](const auto &outerBox) {
      return boxContains(outerBox, innerBox);
    });
  });
}

enum class OutputReachability : uint8_t {
  NoReturn,
  PureReturn,
  EffectfulPath,
};

OutputReachability reachesFunctionReturn(
    mlir::Value value,
    const llvm::DenseSet<mlir::Operation *> &structuredOperations) {
  llvm::DenseSet<mlir::Value> visited;
  llvm::SmallVector<mlir::Value, 8> worklist{value};
  bool effectful = false;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (mlir::OpOperand &use : current.getUses()) {
      mlir::Operation *owner = use.getOwner();
      if (mlir::isa<mlir::func::ReturnOp>(owner))
        return OutputReachability::PureReturn;
      if (structuredOperations.contains(owner))
        continue;
      if (!mlir::isMemoryEffectFree(owner)) {
        effectful = true;
        continue;
      }
      for (mlir::Value result : owner->getResults())
        worklist.push_back(result);
    }
  }
  return effectful ? OutputReachability::EffectfulPath
                   : OutputReachability::NoReturn;
}

struct CoordinateBuilder {
  CanonicalMovementCoordinate coordinate;
  std::set<MovementActionId, MovementActionLess> actions;
  std::optional<CanonicalMovementPlanOutcome> failure;

  void addResource(MovementActionId action,
                   const RepresentationResourceDescription &resource,
                   std::optional<TileId> sourceTile,
                   std::optional<TileId> destinationTile,
                   std::optional<analysis::RootRegionWorkId> work) {
    if (failure)
      return;
    if (!actions.insert(action).second) {
      failure =
          broken(BrokenMovementPlanReason::DuplicateAction,
                 "canonical movement has a duplicate action", std::move(work));
      return;
    }
    coordinate.resources.push_back({std::move(action), resource.exactDomain,
                                    resource.elementType, sourceTile,
                                    destinationTile});
  }
};

} // namespace

CanonicalMovementPlanOutcome buildCanonicalMovementPlan(
    const RegionPlan &regions,
    const CanonicalRepresentationCoordinate &representations,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  if (regions.groups.empty() || rootWorks.empty())
    return broken(BrokenMovementPlanReason::PlanWorkMismatch,
                  "canonical movement requires regions and root work");

  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  llvm::DenseSet<mlir::Operation *> structuredOperations;
  for (const analysis::RootRegionWork &work : rootWorks) {
    if (!works.try_emplace(work.id, &work).second)
      return broken(BrokenMovementPlanReason::DuplicateRootWork,
                    "canonical movement has duplicate root work", work.id);
    if (work.rootOperation)
      structuredOperations.insert(work.rootOperation);
  }

  std::map<PhysicalVersionId, const RepresentationResourceDescription *>
      resources;
  for (const RepresentationResourceDescription &resource :
       representations.resources)
    if (!resources.try_emplace(resource.version, &resource).second)
      return broken(BrokenMovementPlanReason::MissingPhysicalVersion,
                    "canonical representation has duplicate resources");
  if (resources.size() != representations.plan.primaryVersions.size())
    return broken(BrokenMovementPlanReason::MissingPhysicalVersion,
                  "canonical representation plan/resource mismatch");

  CoordinateBuilder builder;
  std::set<PhysicalVersionId> carriedExecutionResults;
  for (const RegionGroupPlan &group : regions.groups) {
    if (group.mandatoryRoots.size() != 1)
      return broken(BrokenMovementPlanReason::PlanWorkMismatch,
                    "canonical movement input is not singleton regions");
    const analysis::RootRegionWorkId workId = group.mandatoryRoots.front();
    auto work = works.find(workId);
    if (work == works.end())
      return broken(BrokenMovementPlanReason::PlanWorkMismatch,
                    "region has no root work", workId);

    for (const ExternalUseBinding &binding : group.externalBindings) {
      BoundaryRegionValueId destinationLogical{workId, binding.fragment};
      PhysicalVersionId destination{destinationLogical};
      auto destinationResource = resources.find(destination);
      if (destinationResource == resources.end())
        continue;
      switch (binding.fragment.source.kind) {
      case analysis::RootBoundaryKind::ProgramInput:
      case analysis::RootBoundaryKind::Constant:
      case analysis::RootBoundaryKind::CapturedValue: {
        ExternalLoadPlan load{{destinationLogical}, destination};
        builder.coordinate.plan.externalLoads.push_back(load);
        builder.addResource(load.id, *destinationResource->second, std::nullopt,
                            workId.tile, workId);
        break;
      }
      case analysis::RootBoundaryKind::StructuredResult: {
        if (!binding.fragment.ownerTile ||
            (binding.fragment.ownerShard.has_value() ==
             binding.fragment.reductionGroup.has_value())) {
          builder.failure =
              broken(BrokenMovementPlanReason::PlanWorkMismatch,
                     "structured fragment has no unique final owner", workId);
          break;
        }
        analysis::RootRegionWorkId sourceWork{binding.fragment.source.semantic,
                                              *binding.fragment.ownerTile};
        if (!works.count(sourceWork)) {
          builder.failure =
              broken(BrokenMovementPlanReason::PlanWorkMismatch,
                     "structured fragment owner has no root work", sourceWork);
          break;
        }
        ExecutionInstanceId sourceExecution;
        if (binding.fragment.ownerShard)
          sourceExecution.source =
              RequiredRootExecution{sourceWork, *binding.fragment.ownerShard};
        else
          sourceExecution.source = RequiredMergeExecution{
              sourceWork, *binding.fragment.reductionGroup};
        PhysicalVersionId source{ExecutionResultValueId{
            sourceExecution, binding.fragment.source.index}};
        auto sourceResource = resources.find(source);
        if (sourceResource == resources.end()) {
          builder.failure = broken(
              BrokenMovementPlanReason::MissingPhysicalVersion,
              "structured fragment has no source physical version", sourceWork);
          break;
        }
        if (!finiteDomainContains(sourceResource->second->exactDomain,
                                  destinationResource->second->exactDomain)) {
          builder.failure = broken(
              BrokenMovementPlanReason::CoverageMismatch,
              "structured DDR fragment is outside its source version", workId);
          break;
        }
        DDRBoundaryTransferPlan transfer{
            {destinationLogical}, source, destination};
        builder.coordinate.plan.ddrTransfers.push_back(transfer);
        carriedExecutionResults.insert(source);
        builder.addResource(transfer.id, *destinationResource->second,
                            *binding.fragment.ownerTile, workId.tile, workId);
        break;
      }
      }
      if (builder.failure)
        break;
    }
    if (builder.failure)
      break;
  }

  for (const analysis::RootRegionWork &mergeWork : rootWorks) {
    for (const analysis::ReductionMergeRequirement &merge : mergeWork.merges) {
      ExecutionInstanceId mergeExecution{
          RequiredMergeExecution{mergeWork.id, merge.group}};
      for (const analysis::ReductionContribution &contribution :
           merge.contributions) {
        if (contribution.tile == merge.mergeTile)
          continue;
        analysis::RootRegionWorkId sourceWork{merge.group.root,
                                              contribution.tile};
        if (!works.count(sourceWork)) {
          builder.failure = broken(
              BrokenMovementPlanReason::PlanWorkMismatch,
              "reduction contribution has no source root work", sourceWork);
          break;
        }
        ExecutionInstanceId sourceExecution{
            RequiredRootExecution{sourceWork, contribution.shard}};
        for (const analysis::ReductionResultSlice &result :
             contribution.results) {
          ReductionPartialValueId logical{sourceExecution, merge.group,
                                          result.result};
          PhysicalVersionId source{logical};
          auto resource = resources.find(source);
          if (resource == resources.end()) {
            builder.failure = broken(
                BrokenMovementPlanReason::MissingPhysicalVersion,
                "reduction gather has no partial physical version", sourceWork);
            break;
          }
          ReductionGatherPlan gather{{merge.group, contribution.shard, logical},
                                     source,
                                     mergeExecution};
          builder.coordinate.plan.reductionGathers.push_back(gather);
          builder.addResource(gather.id, *resource->second, contribution.tile,
                              merge.mergeTile, mergeWork.id);
        }
        for (const analysis::CoupledReductionComponentSlice &component :
             contribution.components) {
          CoupledComponentValueId logical{sourceExecution, merge.group,
                                          component.kind};
          PhysicalVersionId source{logical};
          auto resource = resources.find(source);
          if (resource == resources.end()) {
            builder.failure = broken(
                BrokenMovementPlanReason::MissingPhysicalVersion,
                "coupled gather has no component physical version", sourceWork);
            break;
          }
          ReductionGatherPlan gather{{merge.group, contribution.shard, logical},
                                     source,
                                     mergeExecution};
          builder.coordinate.plan.reductionGathers.push_back(gather);
          builder.addResource(gather.id, *resource->second, contribution.tile,
                              merge.mergeTile, mergeWork.id);
        }
        if (builder.failure)
          break;
      }
      if (builder.failure)
        break;
    }
    if (builder.failure)
      break;
  }

  if (!builder.failure)
    for (const analysis::RootRegionWork &work : rootWorks) {
      for (const analysis::RootResultWork &result : work.results) {
        if (!work.rootOperation ||
            result.result >= work.rootOperation->getNumResults()) {
          builder.failure =
              broken(BrokenMovementPlanReason::PlanWorkMismatch,
                     "publication result has no current root value", work.id);
          break;
        }
        OutputReachability reach = reachesFunctionReturn(
            work.rootOperation->getResult(result.result), structuredOperations);
        if (reach == OutputReachability::NoReturn)
          continue;
        if (reach == OutputReachability::EffectfulPath) {
          builder.failure = unsupported(
              UnsupportedMovementFeature::EffectfulOutputPath,
              "program output path crosses an effectful non-structured op",
              work.id);
          break;
        }
        ExecutionInstanceId execution;
        if (result.ownerShard)
          execution.source = RequiredRootExecution{work.id, *result.ownerShard};
        else if (result.reductionGroup)
          execution.source =
              RequiredMergeExecution{work.id, *result.reductionGroup};
        else {
          builder.failure =
              broken(BrokenMovementPlanReason::PlanWorkMismatch,
                     "publication result has no execution owner", work.id);
          break;
        }
        ExecutionResultValueId logical{execution, result.result};
        PhysicalVersionId source{logical};
        auto resource = resources.find(source);
        if (resource == resources.end()) {
          builder.failure =
              broken(BrokenMovementPlanReason::MissingPhysicalVersion,
                     "program output has no source physical version", work.id);
          break;
        }
        ResultPublicationPlan publication{{logical}, source};
        builder.coordinate.plan.publications.push_back(publication);
        carriedExecutionResults.insert(source);
        builder.addResource(publication.id, *resource->second, work.id.tile,
                            std::nullopt, work.id);
      }
      if (builder.failure)
        break;
    }
  if (builder.failure)
    return std::move(*builder.failure);

  for (const PhysicalVersionPlan &version :
       representations.plan.primaryVersions) {
    const auto *result =
        std::get_if<ExecutionResultValueId>(&version.id.logicalValue);
    if (!result || carriedExecutionResults.count(version.id))
      continue;
    builder.coordinate.plan.discards.push_back(
        {ResultDiscardId{*result}, version.id});
  }

  llvm::sort(builder.coordinate.plan.externalLoads,
             [](const ExternalLoadPlan &lhs, const ExternalLoadPlan &rhs) {
               return lhs.id < rhs.id;
             });
  llvm::sort(
      builder.coordinate.plan.ddrTransfers,
      [](const DDRBoundaryTransferPlan &lhs,
         const DDRBoundaryTransferPlan &rhs) { return lhs.id < rhs.id; });
  llvm::sort(builder.coordinate.plan.reductionGathers,
             [](const ReductionGatherPlan &lhs,
                const ReductionGatherPlan &rhs) { return lhs.id < rhs.id; });
  llvm::sort(builder.coordinate.plan.publications,
             [](const ResultPublicationPlan &lhs,
                const ResultPublicationPlan &rhs) { return lhs.id < rhs.id; });
  llvm::sort(builder.coordinate.plan.discards,
             [](const ResultDiscardPlan &lhs, const ResultDiscardPlan &rhs) {
               return lhs.id < rhs.id;
             });
  llvm::sort(builder.coordinate.resources,
             [&](const MovementResourceDescription &lhs,
                 const MovementResourceDescription &rhs) {
               return MovementActionLess{}(lhs.action, rhs.action);
             });
  return std::move(builder.coordinate);
}

} // namespace wafer::compiler::detail
