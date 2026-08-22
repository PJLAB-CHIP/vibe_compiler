//===- CanonicalStoragePlan.cpp - Fresh single-slot storage -----------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

CanonicalStoragePlanOutcome
broken(BrokenStoragePlanReason reason, llvm::StringRef detail,
       std::optional<StorageObjectId> object = std::nullopt) {
  return BrokenStoragePlan{reason, std::move(object), detail.str()};
}

StorageObjectId objectForVersion(const PhysicalVersionId &version) {
  return StorageObjectId{StorageObjectOrigin{version}};
}

StorageObjectId objectForGather(const ReductionGatherId &gather) {
  return StorageObjectId{StorageObjectOrigin{ReductionGatherStagingId{gather}}};
}

llvm::StringRef objectKind(const StorageObjectId &object) {
  if (std::holds_alternative<ReductionGatherStagingId>(object.origin))
    return "reduction-gather-staging";
  const PhysicalVersionId &version = std::get<PhysicalVersionId>(object.origin);
  switch (version.logicalValue.index()) {
  case 0:
    return "boundary-version";
  case 1:
    return "support-version";
  case 2:
    return "execution-result-version";
  case 3:
    return "reduction-partial-version";
  case 4:
    return "coupled-component-version";
  }
  return "unknown-version";
}

analysis::RootRegionWorkId workOf(const ExecutionInstanceId &execution);

std::string objectDescription(const StorageObjectId &object) {
  std::string description = objectKind(object).str();
  const auto *version = std::get_if<PhysicalVersionId>(&object.origin);
  if (!version)
    return description;
  std::visit(
      [&](const auto &logical) {
        using T = std::decay_t<decltype(logical)>;
        if constexpr (std::is_same_v<T, ExecutionResultValueId>) {
          const analysis::RootRegionWorkId work = workOf(logical.execution);
          description = (llvm::Twine(description) +
                         "(anchor=" + llvm::Twine(work.root.anchorIndex) +
                         ",tile=" + llvm::Twine(work.tile.getValue()) +
                         ",result=" + llvm::Twine(logical.result) + ")")
                            .str();
        }
      },
      version->logicalValue);
  return description;
}

analysis::RootRegionWorkId workOf(const ExecutionInstanceId &execution) {
  return std::visit([](const auto &source) { return source.work; },
                    execution.source);
}

TileId tileOf(const PhysicalVersionId &version) {
  return std::visit(
      [](const auto &logical) {
        using T = std::decay_t<decltype(logical)>;
        if constexpr (std::is_same_v<T, BoundaryRegionValueId> ||
                      std::is_same_v<T, SupportRegionValueId>)
          return logical.work.tile;
        else
          return workOf(logical.execution).tile;
      },
      version.logicalValue);
}

bool sameFiniteResource(const analysis::ExactIndexSet &lhs,
                        const analysis::ExactIndexSet &rhs) {
  if (lhs.getRank() != rhs.getRank() || lhs.getForm() != rhs.getForm() ||
      lhs.getBoxes().size() != rhs.getBoxes().size())
    return false;
  if (lhs.getBoxes().empty())
    return false;
  auto boxLess = [](const analysis::StaticRectangularIndexSet &left,
                    const analysis::StaticRectangularIndexSet &right) {
    if (left.offsets != right.offsets)
      return std::lexicographical_compare(
          left.offsets.begin(), left.offsets.end(), right.offsets.begin(),
          right.offsets.end());
    return std::lexicographical_compare(left.sizes.begin(), left.sizes.end(),
                                        right.sizes.begin(), right.sizes.end());
  };
  llvm::SmallVector<analysis::StaticRectangularIndexSet, 4> lhsBoxes(
      lhs.getBoxes().begin(), lhs.getBoxes().end());
  llvm::SmallVector<analysis::StaticRectangularIndexSet, 4> rhsBoxes(
      rhs.getBoxes().begin(), rhs.getBoxes().end());
  llvm::sort(lhsBoxes, boxLess);
  llvm::sort(rhsBoxes, boxLess);
  for (auto [lhsBox, rhsBox] : llvm::zip_equal(lhsBoxes, rhsBoxes))
    if (lhsBox.offsets != rhsBox.offsets || lhsBox.sizes != rhsBox.sizes)
      return false;
  return true;
}

struct PendingObject {
  StorageObjectPlan plan;
  StorageResourceDescription resource;
  std::optional<StorageAccessSite> definition;
  std::set<StorageAccessSite> uses;
};

struct CoordinateBuilder {
  CanonicalStorageCoordinate coordinate;
  std::map<StorageObjectId, PendingObject> objects;
  std::optional<CanonicalStoragePlanOutcome> failure;

  PendingObject *addObject(StorageObjectId id, TileId tile,
                           const analysis::ExactIndexSet &domain,
                           mlir::Type elementType, MemLayout encoding) {
    auto [position, inserted] = objects.try_emplace(
        id, PendingObject{
                StorageObjectPlan{id, tile},
                StorageResourceDescription{id, domain, elementType, encoding},
                std::nullopt,
                {}});
    if (!inserted) {
      failure = broken(BrokenStoragePlanReason::DuplicateStorageObject,
                       "canonical storage has a duplicate object", id);
      return nullptr;
    }
    coordinate.plan.storageObjects.push_back(position->second.plan);
    coordinate.resources.push_back(position->second.resource);
    return &position->second;
  }

  PendingObject *find(const StorageObjectId &id) {
    auto object = objects.find(id);
    return object == objects.end() ? nullptr : &object->second;
  }

  PendingObject *findVersion(const PhysicalVersionId &version) {
    StorageObjectId id = objectForVersion(version);
    PendingObject *object = find(id);
    if (!object)
      failure =
          broken(BrokenStoragePlanReason::MissingPhysicalVersion,
                 "storage action references a missing physical version", id);
    return object;
  }

  void define(PendingObject &object, StorageAccessSite site) {
    if (failure)
      return;
    if (object.definition) {
      failure =
          broken(BrokenStoragePlanReason::DuplicateDefinition,
                 "storage object has more than one definition", object.plan.id);
      return;
    }
    object.definition = std::move(site);
  }

  void use(PendingObject &object, StorageAccessSite site) {
    if (!failure)
      object.uses.insert(std::move(site));
  }
};

} // namespace

CanonicalStoragePlanOutcome buildCanonicalStoragePlan(
    const CanonicalRepresentationCoordinate &representations,
    const CanonicalMovementCoordinate &movements,
    const SerializedExecutionPlan &serialized) {
  if (representations.plan.primaryVersions.empty() ||
      serialized.executions.empty())
    return broken(BrokenStoragePlanReason::EmptyStorageInput,
                  "canonical storage requires versions and executions");

  std::set<ExecutionInstanceId> executions;
  std::map<analysis::RootRegionWorkId, std::vector<ExecutionInstanceId>>
      rootExecutionsByWork;
  std::map<ReductionGroupId, ExecutionInstanceId> mergeExecutions;
  for (const ExecutionInstanceId &execution : serialized.executions) {
    if (!executions.insert(execution).second)
      return broken(BrokenStoragePlanReason::MissingSerializedExecution,
                    "serialized plan has a duplicate execution");
    if (const auto *root =
            std::get_if<RequiredRootExecution>(&execution.source)) {
      rootExecutionsByWork[root->work].push_back(execution);
      continue;
    }
    const auto &merge = std::get<RequiredMergeExecution>(execution.source);
    if (!mergeExecutions.try_emplace(merge.group, execution).second)
      return broken(BrokenStoragePlanReason::MissingSerializedExecution,
                    "serialized plan has duplicate merge group execution");
  }

  std::map<PhysicalVersionId, const RepresentationResourceDescription *>
      versionResources;
  for (const RepresentationResourceDescription &resource :
       representations.resources)
    if (!versionResources.try_emplace(resource.version, &resource).second)
      return broken(BrokenStoragePlanReason::DuplicatePhysicalVersion,
                    "representation has duplicate version resources",
                    objectForVersion(resource.version));
  if (versionResources.size() != representations.plan.primaryVersions.size())
    return broken(BrokenStoragePlanReason::MissingPhysicalVersion,
                  "representation plan/resource count differs");
  for (const PhysicalVersionPlan &version :
       representations.plan.primaryVersions) {
    auto resource = versionResources.find(version.id);
    if (resource == versionResources.end())
      return broken(BrokenStoragePlanReason::MissingPhysicalVersion,
                    "representation plan has no version resource",
                    objectForVersion(version.id));
    if (resource->second->encoding != version.encoding)
      return broken(BrokenStoragePlanReason::ResourceMismatch,
                    "version encoding differs from its resource",
                    objectForVersion(version.id));
  }

  std::set<MovementActionId> plannedActions;
  auto addAction = [&](MovementActionId action) -> bool {
    return plannedActions.insert(std::move(action)).second;
  };
  for (const ExternalLoadPlan &load : movements.plan.externalLoads)
    if (!addAction(load.id))
      return broken(BrokenStoragePlanReason::DuplicateMovementAction,
                    "movement plan has a duplicate external load");
  for (const DDRBoundaryTransferPlan &transfer : movements.plan.ddrTransfers)
    if (!addAction(transfer.id))
      return broken(BrokenStoragePlanReason::DuplicateMovementAction,
                    "movement plan has a duplicate DDR transfer");
  for (const ReductionGatherPlan &gather : movements.plan.reductionGathers)
    if (!addAction(gather.id))
      return broken(BrokenStoragePlanReason::DuplicateMovementAction,
                    "movement plan has a duplicate reduction gather");
  for (const ResultPublicationPlan &publication : movements.plan.publications)
    if (!addAction(publication.id))
      return broken(BrokenStoragePlanReason::DuplicateMovementAction,
                    "movement plan has a duplicate publication");

  std::map<MovementActionId, const MovementResourceDescription *>
      movementResources;
  for (const MovementResourceDescription &resource : movements.resources)
    if (!movementResources.try_emplace(resource.action, &resource).second)
      return broken(BrokenStoragePlanReason::DuplicateMovementAction,
                    "movement has duplicate action resources");
  if (movementResources.size() != plannedActions.size())
    return broken(BrokenStoragePlanReason::MissingMovementResource,
                  "movement plan/resource count differs");
  for (const MovementActionId &action : plannedActions)
    if (!movementResources.count(action))
      return broken(BrokenStoragePlanReason::MissingMovementResource,
                    "movement action has no resource description");
  for (const auto &[action, resource] : movementResources)
    if (!plannedActions.count(action))
      return broken(BrokenStoragePlanReason::MissingMovementResource,
                    "movement resource has no planned action");

  std::set<PhysicalVersionId> discardedExecutionResults;
  for (const ResultDiscardPlan &discard : movements.plan.discards) {
    PhysicalVersionId expected{discard.id.source};
    if (discard.source != expected || !versionResources.count(discard.source))
      return broken(BrokenStoragePlanReason::PlanBindingMismatch,
                    "result discard identity and source version differ");
    if (!discardedExecutionResults.insert(discard.source).second)
      return broken(BrokenStoragePlanReason::DuplicateResultDiscard,
                    "movement plan has a duplicate result discard");
  }

  CoordinateBuilder builder;
  std::set<PhysicalVersionId> carriedExecutionResults;
  for (const PhysicalVersionPlan &version :
       representations.plan.primaryVersions) {
    const RepresentationResourceDescription &resource =
        *versionResources.at(version.id);
    TileId tile = tileOf(version.id);
    StorageObjectId objectId = objectForVersion(version.id);
    PendingObject *object =
        builder.addObject(objectId, tile, resource.exactDomain,
                          resource.elementType, resource.encoding);
    if (!object)
      break;
    builder.coordinate.plan.versionBindings.push_back({version.id, objectId});

    std::visit(
        [&](const auto &logical) {
          using T = std::decay_t<decltype(logical)>;
          if constexpr (std::is_same_v<T, BoundaryRegionValueId>) {
            ExecutionInstanceId consumer{RequiredRootExecution{
                logical.work, logical.fragment.use.destinationShard}};
            if (!executions.count(consumer)) {
              builder.failure = broken(
                  BrokenStoragePlanReason::MissingSerializedExecution,
                  "boundary version consumer is not serialized", objectId);
              return;
            }
            builder.use(*object, StorageAccessSite{consumer});
          } else if constexpr (std::is_same_v<T, SupportRegionValueId>) {
            auto roots = rootExecutionsByWork.find(logical.work);
            if (roots == rootExecutionsByWork.end() ||
                roots->second.size() != 1) {
              builder.failure = broken(
                  BrokenStoragePlanReason::MissingSerializedExecution,
                  "support version has no unique root execution", objectId);
              return;
            }
            builder.define(*object, StorageAccessSite{roots->second.front()});
            builder.use(*object, StorageAccessSite{roots->second.front()});
          } else {
            if (!executions.count(logical.execution)) {
              builder.failure = broken(
                  BrokenStoragePlanReason::MissingSerializedExecution,
                  "physical version producer is not serialized", objectId);
              return;
            }
            builder.define(*object, StorageAccessSite{logical.execution});
            if constexpr (std::is_same_v<T, CoupledComponentValueId>)
              if (std::holds_alternative<RequiredMergeExecution>(
                      logical.execution.source))
                builder.use(*object, StorageAccessSite{logical.execution});
          }
        },
        version.id.logicalValue);
    if (builder.failure)
      break;
  }
  if (builder.failure)
    return std::move(*builder.failure);

  auto actionResource = [&](const MovementActionId &action)
      -> const MovementResourceDescription & {
    return *movementResources.at(action);
  };
  auto checkResource = [&](PendingObject &object,
                           const MovementActionId &action) -> bool {
    const MovementResourceDescription &movement = actionResource(action);
    if (movement.elementType != object.resource.elementType ||
        !sameFiniteResource(movement.exactDomain,
                            object.resource.exactDomain)) {
      builder.failure = broken(
          BrokenStoragePlanReason::ResourceMismatch,
          (llvm::Twine("movement and storage resources differ: ") +
           objectDescription(object.plan.id) + ", element-type-equal=" +
           llvm::Twine(movement.elementType == object.resource.elementType) +
           ", movement-rank=" + llvm::Twine(movement.exactDomain.getRank()) +
           ", storage-rank=" +
           llvm::Twine(object.resource.exactDomain.getRank()) +
           ", movement-form=" +
           llvm::Twine(static_cast<unsigned>(movement.exactDomain.getForm())) +
           ", storage-form=" +
           llvm::Twine(
               static_cast<unsigned>(object.resource.exactDomain.getForm())) +
           ", movement-boxes=" +
           llvm::Twine(movement.exactDomain.getBoxes().size()) +
           ", storage-boxes=" +
           llvm::Twine(object.resource.exactDomain.getBoxes().size()))
              .str(),
          object.plan.id);
      return false;
    }
    return true;
  };

  for (const ExternalLoadPlan &load : movements.plan.externalLoads) {
    if (load.destination != PhysicalVersionId{load.id.destination})
      return broken(BrokenStoragePlanReason::PlanBindingMismatch,
                    "external load ID and destination version differ");
    PendingObject *destination = builder.findVersion(load.destination);
    if (!destination)
      break;
    MovementActionId action = load.id;
    if (!checkResource(*destination, action))
      break;
    builder.define(*destination, StorageAccessSite{std::move(action)});
  }
  if (builder.failure)
    return std::move(*builder.failure);

  for (const DDRBoundaryTransferPlan &transfer : movements.plan.ddrTransfers) {
    if (transfer.destination != PhysicalVersionId{transfer.id.destination})
      return broken(BrokenStoragePlanReason::PlanBindingMismatch,
                    "DDR transfer ID and destination version differ");
    PendingObject *source = builder.findVersion(transfer.source);
    PendingObject *destination = builder.findVersion(transfer.destination);
    if (!source || !destination)
      break;
    MovementActionId action = transfer.id;
    if (!checkResource(*destination, action))
      break;
    builder.use(*source, StorageAccessSite{action});
    if (std::holds_alternative<ExecutionResultValueId>(
            transfer.source.logicalValue))
      carriedExecutionResults.insert(transfer.source);
    builder.define(*destination, StorageAccessSite{std::move(action)});
  }
  if (builder.failure)
    return std::move(*builder.failure);

  std::set<PhysicalVersionId> gatheredSources;
  for (const ReductionGatherPlan &gather : movements.plan.reductionGathers) {
    PhysicalVersionId expectedSource = std::visit(
        [](const auto &logical) { return PhysicalVersionId{logical}; },
        gather.id.value);
    const auto *requiredMerge =
        std::get_if<RequiredMergeExecution>(&gather.mergeExecution.source);
    if (gather.source != expectedSource || !requiredMerge ||
        requiredMerge->group != gather.id.group)
      return broken(BrokenStoragePlanReason::PlanBindingMismatch,
                    "reduction gather identity and endpoints differ");
    PendingObject *source = builder.findVersion(gather.source);
    if (!source)
      break;
    if (!executions.count(gather.mergeExecution)) {
      builder.failure =
          broken(BrokenStoragePlanReason::MissingSerializedExecution,
                 "reduction gather merge is not serialized", source->plan.id);
      break;
    }
    MovementActionId action = gather.id;
    if (!checkResource(*source, action))
      break;
    builder.use(*source, StorageAccessSite{action});
    gatheredSources.insert(gather.source);

    const MovementResourceDescription &movement = actionResource(action);
    StorageObjectId stagingId = objectForGather(gather.id);
    PendingObject *staging = builder.addObject(
        stagingId, workOf(gather.mergeExecution).tile, movement.exactDomain,
        movement.elementType, source->resource.encoding);
    if (!staging)
      break;
    builder.define(*staging, StorageAccessSite{action});
    builder.use(*staging, StorageAccessSite{gather.mergeExecution});
    builder.coordinate.plan.gatherStagingBindings.push_back(
        {gather.id, stagingId});
  }
  if (builder.failure)
    return std::move(*builder.failure);

  for (const ResultPublicationPlan &publication : movements.plan.publications) {
    if (publication.source != PhysicalVersionId{publication.id.source})
      return broken(BrokenStoragePlanReason::PlanBindingMismatch,
                    "publication ID and source version differ");
    PendingObject *source = builder.findVersion(publication.source);
    if (!source)
      break;
    MovementActionId action = publication.id;
    if (!checkResource(*source, action))
      break;
    builder.use(*source, StorageAccessSite{std::move(action)});
    carriedExecutionResults.insert(publication.source);
  }
  if (builder.failure)
    return std::move(*builder.failure);

  for (const PhysicalVersionPlan &version :
       representations.plan.primaryVersions) {
    const auto *result =
        std::get_if<ExecutionResultValueId>(&version.id.logicalValue);
    if (!result)
      continue;
    const bool carried = carriedExecutionResults.count(version.id);
    const bool discarded = discardedExecutionResults.count(version.id);
    if (carried && discarded)
      return broken(BrokenStoragePlanReason::PlanBindingMismatch,
                    "execution result is both carried and discarded",
                    objectForVersion(version.id));
    if (!carried && !discarded)
      return broken(BrokenStoragePlanReason::MissingUse,
                    "execution result has no carrier or explicit discard",
                    objectForVersion(version.id));
    if (discarded) {
      PendingObject *object = builder.findVersion(version.id);
      if (!object)
        break;
      builder.use(*object, StorageAccessSite{result->execution});
    }
  }
  if (builder.failure)
    return std::move(*builder.failure);

  for (const PhysicalVersionPlan &version :
       representations.plan.primaryVersions) {
    PendingObject *object = builder.findVersion(version.id);
    if (!object)
      break;
    std::visit(
        [&](const auto &logical) {
          using T = std::decay_t<decltype(logical)>;
          if constexpr (std::is_same_v<T, ReductionPartialValueId> ||
                        std::is_same_v<T, CoupledComponentValueId>) {
            if (!std::holds_alternative<RequiredRootExecution>(
                    logical.execution.source))
              return;
            auto merge = mergeExecutions.find(logical.group);
            if (merge == mergeExecutions.end()) {
              builder.failure = broken(
                  BrokenStoragePlanReason::MissingSerializedExecution,
                  "reduction value has no serialized merge", object->plan.id);
              return;
            }
            if (gatheredSources.count(version.id))
              return;
            if (workOf(logical.execution).tile != workOf(merge->second).tile) {
              builder.failure = broken(BrokenStoragePlanReason::MissingUse,
                                       "remote reduction value has no gather",
                                       object->plan.id);
              return;
            }
            builder.use(*object, StorageAccessSite{merge->second});
          }
        },
        version.id.logicalValue);
    if (builder.failure)
      break;
  }
  if (builder.failure)
    return std::move(*builder.failure);

  for (auto &[id, object] : builder.objects) {
    if (!object.definition)
      return broken(BrokenStoragePlanReason::MissingDefinition,
                    "storage object has no definition", id);
    if (object.uses.empty())
      return broken(
          BrokenStoragePlanReason::MissingUse,
          (llvm::Twine("storage object has no use: ") + objectDescription(id))
              .str(),
          id);
    builder.coordinate.lifetimes.push_back(
        {id, *object.definition,
         std::vector<StorageAccessSite>(object.uses.begin(),
                                        object.uses.end())});
  }

  llvm::sort(builder.coordinate.plan.storageObjects,
             [](const StorageObjectPlan &lhs, const StorageObjectPlan &rhs) {
               return lhs.id < rhs.id;
             });
  llvm::sort(builder.coordinate.plan.versionBindings,
             [](const PhysicalVersionStorageBinding &lhs,
                const PhysicalVersionStorageBinding &rhs) {
               return lhs.version < rhs.version;
             });
  llvm::sort(builder.coordinate.plan.gatherStagingBindings,
             [](const ReductionGatherStorageBinding &lhs,
                const ReductionGatherStorageBinding &rhs) {
               return lhs.gather < rhs.gather;
             });
  llvm::sort(builder.coordinate.resources,
             [](const StorageResourceDescription &lhs,
                const StorageResourceDescription &rhs) {
               return lhs.object < rhs.object;
             });
  llvm::sort(builder.coordinate.lifetimes,
             [](const StorageLifetimeDescription &lhs,
                const StorageLifetimeDescription &rhs) {
               return lhs.object < rhs.object;
             });
  return std::move(builder.coordinate);
}

} // namespace wafer::compiler::detail
