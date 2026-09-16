//===- CanonicalRegionPlan.cpp - Singleton region plan ----------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>

namespace wafer::compiler::detail {
namespace {

CanonicalRegionPlanOutcome
broken(BrokenRegionPlanReason reason, llvm::StringRef detail,
       std::optional<analysis::RootRegionWorkId> work = std::nullopt) {
  return BrokenRegionPlan{reason, std::move(work), detail.str()};
}

bool isStructuredBoundary(const analysis::RootBoundaryWork &boundary) {
  return boundary.id.kind == analysis::RootBoundaryKind::StructuredResult;
}

CanonicalRegionPlanOutcome buildGroup(const analysis::RootRegionWork &work) {
  RegionGroupPlan group;
  group.tile = work.id.tile;
  group.mandatoryRoots.push_back(work.id);

  std::set<ExecutionInstanceId> executionIds;
  std::set<LogicalShardId> executionShards;
  for (const analysis::RootExecutionWork &execution : work.execution) {
    if (execution.shard.root != work.id.root ||
        !executionShards.insert(execution.shard).second)
      return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                    "root work has a duplicate or foreign execution shard",
                    work.id);
    ExecutionInstancePlan instance;
    instance.id.source = RequiredRootExecution{work.id, execution.shard};
    if (!executionIds.insert(instance.id).second)
      return broken(BrokenRegionPlanReason::DuplicateExecution,
                    "canonical region has a duplicate root execution", work.id);
    group.executions.push_back(std::move(instance));
  }

  std::set<ReductionGroupId> mergeGroups;
  for (const analysis::ReductionMergeRequirement &merge : work.merges) {
    if (merge.group.root != work.id.root || merge.mergeTile != work.id.tile ||
        !mergeGroups.insert(merge.group).second)
      return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                    "root work has a duplicate or foreign merge requirement",
                    work.id);
    ExecutionInstancePlan instance;
    instance.id.source = RequiredMergeExecution{work.id, merge.group};
    if (!executionIds.insert(instance.id).second)
      return broken(BrokenRegionPlanReason::DuplicateExecution,
                    "canonical region has a duplicate merge execution",
                    work.id);
    group.executions.push_back(std::move(instance));
  }

  for (const analysis::RootContributionWork &contribution : work.contributions)
    if (contribution.group.root != work.id.root ||
        contribution.contribution.tile != work.id.tile ||
        !executionShards.count(contribution.contribution.shard))
      return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                    "root contribution has no matching required execution",
                    work.id);

  for (const analysis::RootResultWork &result : work.results) {
    const bool ownedByExecution =
        result.ownerShard && executionShards.count(*result.ownerShard);
    const bool ownedByMerge =
        result.reductionGroup && mergeGroups.count(*result.reductionGroup);
    if (ownedByExecution == ownedByMerge)
      return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                    "root result has no unique execution or merge owner",
                    work.id);
  }

  std::set<DemandFragmentId> fragmentIds;
  std::set<analysis::RootUseId> boundaryUses;
  for (const analysis::RootBoundaryWork &boundary : work.boundaries) {
    for (const analysis::RootBoundaryUseWork &use : boundary.consumerUses) {
      bool local = std::visit(
          [&](const auto &consumer) {
            using T = std::decay_t<decltype(consumer)>;
            if constexpr (std::is_same_v<T, LogicalShardId>)
              return executionShards.count(consumer) != 0;
            else
              return mergeGroups.count(consumer) != 0;
          },
          use.id.destination);
      if (analysis::getDemandRoot(use.id.destination) != work.id.root || !local)
        return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                      "root boundary use has no local execution or merge",
                      work.id);
      boundaryUses.insert(use.id);
      if (use.access == analysis::RootBoundaryAccessKind::IndexedTensor) {
        auto gather =
            mlir::dyn_cast_or_null<mlir::tensor::GatherOp>(work.rootOperation);
        if (!gather || use.id.operand != 0 || use.requiredDomain ||
            boundary.sourceValue != gather.getSource() ||
            (boundary.id.kind != analysis::RootBoundaryKind::ProgramInput &&
             boundary.id.kind != analysis::RootBoundaryKind::Constant))
          return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                        "indexed table boundary lost its explicit gather source",
                        work.id);
      } else if (!use.requiredDomain && boundary.sourceValue &&
                 mlir::isa<mlir::ShapedType>(boundary.sourceValue.getType())) {
        return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                      "tensor availability requires an indexed access contract",
                      work.id);
      }
      if ((!use.requiredDomain && !use.eligibleFinalOwners.empty()) ||
          (!isStructuredBoundary(boundary) && !use.eligibleFinalOwners.empty()))
        return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                      "boundary use has owners outside structured tensor "
                      "demand",
                      work.id);
      if (use.requiredDomain && !use.requiredDomain->isEmpty() &&
          isStructuredBoundary(boundary) && use.eligibleFinalOwners.empty())
        return broken(BrokenRegionPlanReason::MissingBoundaryFragment,
                      "nonempty structured boundary use has no final owner",
                      work.id);

      auto appendFragment = [&](std::optional<LogicalShardId> ownerShard,
                                std::optional<ReductionGroupId> reductionGroup,
                                std::optional<TileId> ownerTile) {
        DemandFragmentId fragment{boundary.id, use.id, ownerShard,
                                  reductionGroup, ownerTile};
        if (!fragmentIds.insert(fragment).second)
          return false;
        group.externalBindings.push_back({std::move(fragment)});
        return true;
      };
      if (use.eligibleFinalOwners.empty()) {
        if (!appendFragment(std::nullopt, std::nullopt, std::nullopt))
          return broken(BrokenRegionPlanReason::DuplicateFragment,
                        "canonical region has a duplicate boundary fragment",
                        work.id);
        continue;
      }
      for (const analysis::OwnerIntersection &owner : use.eligibleFinalOwners) {
        if (owner.ownerShard.has_value() == owner.reductionGroup.has_value())
          return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                        "boundary owner must name exactly one shard or merge "
                        "group",
                        work.id);
        if (!appendFragment(owner.ownerShard, owner.reductionGroup, owner.tile))
          return broken(BrokenRegionPlanReason::DuplicateFragment,
                        "canonical region has a duplicate owner fragment",
                        work.id);
      }
    }
  }

  for (const analysis::RootOperandWork &operand : work.operands)
    for (const analysis::RootOperandUseWork &use : operand.uses)
      if (!boundaryUses.count(use.id))
        return broken(BrokenRegionPlanReason::MissingBoundaryFragment,
                      "root operand use has no canonical boundary fragment",
                      work.id);

  llvm::sort(group.executions,
             [](const ExecutionInstancePlan &lhs,
                const ExecutionInstancePlan &rhs) { return lhs.id < rhs.id; });
  llvm::sort(group.externalBindings,
             [](const ExternalUseBinding &lhs, const ExternalUseBinding &rhs) {
               return lhs.fragment < rhs.fragment;
             });
  RegionPlan plan;
  plan.groups.push_back(std::move(group));
  return plan;
}

} // namespace

CanonicalRegionPlanOutcome
buildCanonicalRegionPlan(llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  if (rootWorks.empty())
    return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                  "canonical region plan has no root work");

  std::set<analysis::RootRegionWorkId> workIds;
  RegionPlan result;
  for (const analysis::RootRegionWork &work : rootWorks) {
    if (!work.rootOperation || (work.execution.empty() && work.merges.empty()))
      return broken(BrokenRegionPlanReason::WorkStructureMismatch,
                    "canonical region input has empty root work", work.id);
    if (!workIds.insert(work.id).second)
      return broken(BrokenRegionPlanReason::DuplicateRootWork,
                    "canonical region input has duplicate root work", work.id);
    CanonicalRegionPlanOutcome group = buildGroup(work);
    const RegionPlan *plan = getRegionPlan(group);
    if (!plan)
      return group;
    result.groups.push_back(plan->groups.front());
  }
  llvm::sort(result.groups,
             [](const RegionGroupPlan &lhs, const RegionGroupPlan &rhs) {
               if (lhs.tile != rhs.tile)
                 return lhs.tile.getValue() < rhs.tile.getValue();
               return std::lexicographical_compare(
                   lhs.mandatoryRoots.begin(), lhs.mandatoryRoots.end(),
                   rhs.mandatoryRoots.begin(), rhs.mandatoryRoots.end());
             });
  return result;
}

} // namespace wafer::compiler::detail
