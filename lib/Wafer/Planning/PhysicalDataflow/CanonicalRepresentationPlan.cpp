//===- CanonicalRepresentationPlan.cpp - Tensor primary versions -----===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

CanonicalRepresentationPlanOutcome
broken(BrokenRepresentationPlanReason reason, llvm::StringRef detail,
       std::optional<analysis::RootRegionWorkId> work = std::nullopt) {
  return BrokenRepresentationPlan{reason, std::move(work), detail.str()};
}

CanonicalRepresentationPlanOutcome
unsupported(UnsupportedRepresentationFeature feature, llvm::StringRef detail,
            std::optional<analysis::RootRegionWorkId> work = std::nullopt) {
  return UnsupportedRepresentationPlan{feature, std::move(work), detail.str()};
}

std::optional<mlir::Type> getElementType(mlir::Type type) {
  auto shaped = mlir::dyn_cast_or_null<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasRank())
    return std::nullopt;
  return shaped.getElementType();
}

struct CoordinateBuilder {
  CanonicalRepresentationCoordinate coordinate;
  std::set<PhysicalVersionId> versions;
  std::set<RepresentationUseId> uses;
  std::map<RegionValueVersionId, PhysicalVersionId> primaryByLogical;
  std::optional<CanonicalRepresentationPlanOutcome> failure;

  void add(RegionValueVersionId logicalValue,
           const analysis::ExactIndexSet &domain, mlir::Type elementType,
           std::optional<analysis::RootRegionWorkId> work) {
    if (failure || domain.isEmpty())
      return;
    if (!elementType || !elementType.isIntOrFloat()) {
      failure = unsupported(
          UnsupportedRepresentationFeature::ElementType,
          "canonical representation requires an integer or floating element "
          "type",
          std::move(work));
      return;
    }
    mlir::FailureOr<analysis::ExactIndexSet> normalized =
        analysis::normalizeFiniteExactIndexSet(domain);
    if (mlir::failed(normalized)) {
      failure = unsupported(
          UnsupportedRepresentationFeature::TensorEncoding,
          (llvm::Twine(
               "Tensor encoding requires a finite disjoint rectangle union: "
               "logical-kind=") +
           llvm::Twine(logicalValue.index()) +
           ", rank=" + llvm::Twine(domain.getRank()) +
           ", form=" + llvm::Twine(static_cast<unsigned>(domain.getForm())) +
           ", disjuncts=" +
           llvm::Twine(domain.getPresburgerSet().getNumDisjuncts()))
              .str(),
          std::move(work));
      return;
    }
    for (const analysis::StaticRectangularIndexSet &box :
         normalized->getBoxes()) {
      auto type = mlir::MemRefType::get(
          box.sizes, elementType, mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(elementType.getContext(), MemorySpace::SPM,
                          MemLayout::Tensor));
      if (mlir::failed(analysis::PhysicalLayoutRelation::create(type))) {
        failure = unsupported(
            UnsupportedRepresentationFeature::TensorEncoding,
            "Tensor encoding cannot represent one exact logical box",
            std::move(work));
        return;
      }
    }
    PhysicalVersionId version{logicalValue};
    if (!versions.insert(version).second ||
        !primaryByLogical.try_emplace(logicalValue, version).second) {
      failure = broken(BrokenRepresentationPlanReason::DuplicateLogicalValue,
                       "canonical representation has a duplicate logical "
                       "value",
                       std::move(work));
      return;
    }
    coordinate.plan.logicalValues.push_back({logicalValue, version});
    coordinate.plan.physicalVersions.push_back({version, MemLayout::Tensor});
    if (const auto *boundary =
            std::get_if<BoundaryRegionValueId>(&logicalValue)) {
      RepresentationUseId use = BoundaryRepresentationUseId{*boundary};
      if (!uses.insert(use).second) {
        failure = broken(BrokenRepresentationPlanReason::DuplicateLogicalValue,
                         "canonical representation has a duplicate use",
                         std::move(work));
        return;
      }
      coordinate.plan.uses.push_back({std::move(use), version});
    }
    coordinate.resources.push_back({std::move(version), std::move(*normalized),
                                    elementType, MemLayout::Tensor});
  }

  void addUse(RepresentationUseId use, const RegionValueVersionId &logical,
              std::optional<analysis::RootRegionWorkId> work) {
    if (failure)
      return;
    auto primary = primaryByLogical.find(logical);
    if (primary == primaryByLogical.end() || !uses.insert(use).second) {
      failure = broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                       "representation use has no unique logical value",
                       std::move(work));
      return;
    }
    coordinate.plan.uses.push_back({std::move(use), primary->second});
  }
};

const analysis::RootBoundaryWork *
findBoundary(const analysis::RootRegionWork &work,
             const analysis::RootBoundaryId &id) {
  auto boundary = llvm::find_if(
      work.boundaries, [&](const analysis::RootBoundaryWork &candidate) {
        return candidate.id == id;
      });
  return boundary == work.boundaries.end() ? nullptr : &*boundary;
}

const analysis::RootBoundaryUseWork *
findBoundaryUse(const analysis::RootBoundaryWork &boundary,
                const analysis::RootUseId &id) {
  auto use = llvm::find_if(boundary.consumerUses,
                           [&](const analysis::RootBoundaryUseWork &candidate) {
                             return candidate.id == id;
                           });
  return use == boundary.consumerUses.end() ? nullptr : &*use;
}

const analysis::OwnerIntersection *
findOwner(const analysis::RootBoundaryUseWork &use,
          const DemandFragmentId &fragment) {
  auto owner = llvm::find_if(
      use.eligibleFinalOwners,
      [&](const analysis::OwnerIntersection &candidate) {
        return candidate.ownerShard == fragment.ownerShard &&
               candidate.reductionGroup == fragment.reductionGroup &&
               fragment.ownerTile && candidate.tile == *fragment.ownerTile;
      });
  return owner == use.eligibleFinalOwners.end() ? nullptr : &*owner;
}

} // namespace

CanonicalRepresentationPlanOutcome buildCanonicalRepresentationPlan(
    const RegionPlan &regions, const TemporalPlan &temporal,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  if (regions.groups.empty() || rootWorks.empty())
    return broken(BrokenRepresentationPlanReason::PlanWorkMismatch,
                  "canonical representation requires regions and root work");

  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  std::map<ReductionGroupId, const analysis::ReductionMergeRequirement *>
      merges;
  for (const analysis::RootRegionWork &work : rootWorks) {
    if (!works.try_emplace(work.id, &work).second)
      return broken(BrokenRepresentationPlanReason::DuplicateRootWork,
                    "canonical representation has duplicate root work",
                    work.id);
    for (const analysis::ReductionMergeRequirement &merge : work.merges)
      if (!merges.try_emplace(merge.group, &merge).second)
        return broken(BrokenRepresentationPlanReason::DuplicateLogicalValue,
                      "canonical representation has duplicate merge work",
                      work.id);
  }

  std::set<RegionExecutionId> temporalExecutions;
  for (const TemporalScopePlan &scope : temporal.scopes) {
    if (!temporalExecutions.insert(scope.id.execution).second &&
        isTopLevelScope(scope.id))
      return broken(BrokenRepresentationPlanReason::PlanWorkMismatch,
                    "canonical temporal plan has duplicate execution scope");
  }

  std::map<analysis::RootRegionWorkId, const RegionGroupPlan *> groups;
  std::set<RegionExecutionId> rootExecutions;
  std::set<ExecutionInstanceId> mergeExecutions;
  for (const RegionGroupPlan &group : regions.groups) {
    if (group.mandatoryRoots.empty())
      return broken(BrokenRepresentationPlanReason::PlanWorkMismatch,
                    "representation input has an empty region group");
    for (const analysis::RootRegionWorkId &work : group.mandatoryRoots)
      if (group.tile != work.tile || !groups.try_emplace(work, &group).second)
        return broken(BrokenRepresentationPlanReason::PlanWorkMismatch,
                      "representation region/root ownership is inconsistent");
    for (const ExecutionInstancePlan &execution : group.executions)
      if (std::holds_alternative<RequiredRootExecution>(execution.id.source))
        rootExecutions.insert(RegionExecutionId(execution.id));
      else
        mergeExecutions.insert(execution.id);
    for (const ReplicaExecutionPlan &replica : group.replicas)
      rootExecutions.insert(RegionExecutionId(replica.id));
  }
  if (rootExecutions != temporalExecutions || groups.size() != works.size())
    return broken(BrokenRepresentationPlanReason::PlanWorkMismatch,
                  "canonical region, temporal and root work do not agree");

  CoordinateBuilder builder;
  std::vector<std::pair<analysis::RootRegionWorkId, LocalUseBinding>> localUses;
  for (const auto &[workId, work] : works) {
    const RegionGroupPlan &group = *groups.find(workId)->second;

    for (const ExternalUseBinding &binding : group.externalBindings) {
      if (binding.fragment.use.destinationShard.root != workId.root)
        continue;
      const analysis::RootBoundaryWork *boundary =
          findBoundary(*work, binding.fragment.source);
      const analysis::RootBoundaryUseWork *use =
          boundary ? findBoundaryUse(*boundary, binding.fragment.use) : nullptr;
      if (!boundary || !use) {
        builder.failure =
            broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                   "external fragment has no root boundary descriptor", workId);
        break;
      }
      if (!use->requiredDomain)
        continue;
      const analysis::ExactIndexSet *domain = &*use->requiredDomain;
      if (binding.fragment.ownerTile) {
        const analysis::OwnerIntersection *owner =
            findOwner(*use, binding.fragment);
        if (!owner) {
          builder.failure = broken(
              BrokenRepresentationPlanReason::MissingValueDescriptor,
              "external fragment has no exact owner intersection", workId);
          break;
        }
        domain = &owner->domain;
      }
      if (domain->isEmpty())
        continue;
      std::optional<mlir::Type> elementType =
          getElementType(boundary->sourceValue.getType());
      if (!elementType) {
        builder.failure = broken(
            BrokenRepresentationPlanReason::MissingValueDescriptor,
            "shaped boundary fragment has no ranked element type", workId);
        break;
      }
      builder.add(BoundaryRegionValueId{workId, binding.fragment}, *domain,
                  *elementType, workId);
    }
    if (builder.failure)
      break;

    for (const LocalUseBinding &binding : group.localBindings)
      if (binding.fragment.use.destinationShard.root == workId.root)
        localUses.push_back({workId, binding});

    for (const analysis::RootSupportValueWork &support : work->supportValues) {
      if (!support.operation ||
          support.result >= support.operation->getNumResults()) {
        builder.failure =
            broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                   "support value has no current shaped result", workId);
        break;
      }
      std::optional<mlir::Type> elementType = getElementType(
          support.operation->getResult(support.result).getType());
      if (!elementType) {
        builder.failure =
            broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                   "support value is not a ranked shaped value", workId);
        break;
      }
      builder.add(SupportRegionValueId{workId, support.id},
                  support.requiredDomain, *elementType, workId);
    }
    if (builder.failure)
      break;

    for (const analysis::RootResultWork &result : work->results) {
      if (!work->rootOperation ||
          result.result >= work->rootOperation->getNumResults()) {
        builder.failure =
            broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                   "root result has no current shaped value", workId);
        break;
      }
      ExecutionInstanceId execution;
      if (result.ownerShard)
        execution.source = RequiredRootExecution{workId, *result.ownerShard};
      else if (result.reductionGroup)
        execution.source =
            RequiredMergeExecution{workId, *result.reductionGroup};
      else {
        builder.failure =
            broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                   "root result has no execution owner", workId);
        break;
      }
      if ((!result.reductionGroup &&
           !rootExecutions.count(RegionExecutionId(execution))) ||
          (result.reductionGroup && !mergeExecutions.count(execution))) {
        builder.failure = broken(
            BrokenRepresentationPlanReason::PlanWorkMismatch,
            "root result execution is absent from the region plan", workId);
        break;
      }
      std::optional<mlir::Type> elementType = getElementType(
          work->rootOperation->getResult(result.result).getType());
      if (!elementType) {
        builder.failure =
            broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                   "root result is not a ranked shaped value", workId);
        break;
      }
      builder.add(
          ExecutionResultValueId{RegionExecutionId(execution), result.result},
          result.domain, *elementType, workId);
    }
    if (builder.failure)
      break;

    for (const analysis::RootContributionWork &contribution :
         work->contributions) {
      ExecutionInstanceId execution{
          RequiredRootExecution{workId, contribution.contribution.shard}};
      if (!rootExecutions.count(RegionExecutionId(execution))) {
        builder.failure = broken(
            BrokenRepresentationPlanReason::PlanWorkMismatch,
            "partial contribution execution is absent from the region plan",
            workId);
        break;
      }
      for (const analysis::ReductionResultSlice &result :
           contribution.contribution.results) {
        if (!work->rootOperation ||
            result.result >= work->rootOperation->getNumResults()) {
          builder.failure =
              broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                     "partial result has no current shaped value", workId);
          break;
        }
        std::optional<mlir::Type> elementType = getElementType(
            work->rootOperation->getResult(result.result).getType());
        if (!elementType) {
          builder.failure =
              broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                     "partial result is not a ranked shaped value", workId);
          break;
        }
        builder.add(ReductionPartialValueId{execution, contribution.group,
                                            result.result},
                    result.domain, *elementType, workId);
      }
      auto requirement = merges.find(contribution.group);
      for (const analysis::CoupledReductionComponentSlice &component :
           contribution.contribution.components) {
        if (requirement == merges.end()) {
          builder.failure = broken(
              BrokenRepresentationPlanReason::MissingValueDescriptor,
              "coupled contribution has no merge component descriptor", workId);
          break;
        }
        auto descriptor = llvm::find_if(
            requirement->second->components,
            [&](const analysis::CoupledReductionComponentRequirement &entry) {
              return entry.kind == component.kind;
            });
        if (descriptor == requirement->second->components.end()) {
          builder.failure =
              broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                     "coupled contribution has an unknown component", workId);
          break;
        }
        builder.add(CoupledComponentValueId{execution, contribution.group,
                                            component.kind},
                    component.domain, descriptor->elementType, workId);
      }
      if (builder.failure)
        break;
    }
    if (builder.failure)
      break;

    for (const analysis::ReductionMergeRequirement &merge : work->merges) {
      ExecutionInstanceId execution{
          RequiredMergeExecution{workId, merge.group}};
      if (!mergeExecutions.count(execution)) {
        builder.failure =
            broken(BrokenRepresentationPlanReason::PlanWorkMismatch,
                   "merge execution is absent from the region plan", workId);
        break;
      }
      for (const analysis::CoupledReductionComponentRequirement &component :
           merge.components)
        builder.add(
            CoupledComponentValueId{execution, merge.group, component.kind},
            component.domain, component.elementType, workId);
      if (builder.failure)
        break;
    }
    if (builder.failure)
      break;
  }
  if (builder.failure)
    return std::move(*builder.failure);

  for (const RegionGroupPlan &group : regions.groups) {
    for (const ReplicaExecutionPlan &replica : group.replicas) {
      analysis::RootRegionWorkId consumerWorkId{
          replica.id.fragment.use.destinationShard.root, group.tile};
      auto consumerWork = works.find(consumerWorkId);
      auto producerWork = works.find(replica.id.producer.work);
      const analysis::RootBoundaryWork *boundary =
          consumerWork == works.end()
              ? nullptr
              : findBoundary(*consumerWork->second, replica.id.fragment.source);
      const analysis::RootBoundaryUseWork *use =
          boundary ? findBoundaryUse(*boundary, replica.id.fragment.use)
                   : nullptr;
      const analysis::ExactIndexSet *domain =
          use && use->requiredDomain ? &*use->requiredDomain : nullptr;
      if (use && replica.id.fragment.ownerTile) {
        const analysis::OwnerIntersection *owner =
            findOwner(*use, replica.id.fragment);
        domain = owner ? &owner->domain : nullptr;
      }
      mlir::Operation *producer = producerWork == works.end()
                                      ? nullptr
                                      : producerWork->second->rootOperation;
      if (!boundary || !use || !domain || domain->isEmpty() || !producer ||
          replica.id.fragment.source.index >= producer->getNumResults())
        return broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                      "replica result has no exact value descriptor",
                      consumerWorkId);
      std::optional<mlir::Type> elementType = getElementType(
          producer->getResult(replica.id.fragment.source.index).getType());
      if (!elementType)
        return broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                      "replica result is not a ranked shaped value",
                      consumerWorkId);
      builder.add(ExecutionResultValueId{RegionExecutionId(replica.id),
                                         replica.id.fragment.source.index},
                  *domain, *elementType, consumerWorkId);
      if (builder.failure)
        return std::move(*builder.failure);
    }
  }

  for (const auto &[consumerWork, binding] : localUses) {
    if (binding.fragment.source.kind !=
        analysis::RootBoundaryKind::StructuredResult)
      return broken(BrokenRepresentationPlanReason::MissingValueDescriptor,
                    "local representation use is not a structured result",
                    consumerWork);
    RegionValueVersionId logical = std::visit(
        [&](const auto &execution) -> RegionValueVersionId {
          return ExecutionResultValueId{RegionExecutionId(execution),
                                        binding.fragment.source.index};
        },
        binding.producer);
    builder.addUse(LocalRepresentationUseId{consumerWork, binding.fragment},
                   logical, consumerWork);
    if (builder.failure)
      return std::move(*builder.failure);
  }

  llvm::sort(builder.coordinate.plan.logicalValues);
  llvm::sort(builder.coordinate.plan.physicalVersions,
             [](const PhysicalVersionPlan &lhs,
                const PhysicalVersionPlan &rhs) { return lhs.id < rhs.id; });
  llvm::sort(builder.coordinate.plan.uses);
  llvm::sort(builder.coordinate.resources,
             [](const RepresentationResourceDescription &lhs,
                const RepresentationResourceDescription &rhs) {
               return lhs.version < rhs.version;
             });
  if (builder.coordinate.plan.logicalValues.size() !=
          builder.coordinate.resources.size() ||
      builder.coordinate.plan.physicalVersions.size() !=
          builder.coordinate.resources.size())
    return broken(BrokenRepresentationPlanReason::PlanWorkMismatch,
                  "representation plan and resources have different versions");
  return std::move(builder.coordinate);
}

} // namespace wafer::compiler::detail
