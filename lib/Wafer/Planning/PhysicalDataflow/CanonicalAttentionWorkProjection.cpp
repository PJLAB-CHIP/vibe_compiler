//===- CanonicalAttentionWorkProjection.cpp - Project attention ------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalAttentionWorkProjection.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringRef.h"

#include <map>
#include <set>
#include <type_traits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

CanonicalAttentionWorkProjectionOutcome
broken(BrokenAttentionWorkProjectionReason reason, llvm::StringRef detail,
       std::optional<SemanticRootKey> root = std::nullopt) {
  return BrokenAttentionWorkProjection{reason, std::move(root), detail.str()};
}

analysis::RootRegionWorkId workOf(const ExecutionInstanceId &execution) {
  return std::visit([](const auto &source) { return source.work; },
                    execution.source);
}

bool sameDomain(const analysis::ExactIndexSet &lhs,
                const analysis::ExactIndexSet &rhs) {
  if (lhs.getRank() != rhs.getRank() || lhs.getForm() != rhs.getForm() ||
      lhs.getBoxes().size() != rhs.getBoxes().size() || lhs.getBoxes().empty())
    return false;
  for (auto [lhsBox, rhsBox] : llvm::zip_equal(lhs.getBoxes(), rhs.getBoxes()))
    if (lhsBox.offsets != rhsBox.offsets || lhsBox.sizes != rhsBox.sizes)
      return false;
  return true;
}

llvm::SmallVector<unsigned, 8> joinAxes(llvm::ArrayRef<unsigned> first,
                                        llvm::ArrayRef<unsigned> second,
                                        llvm::ArrayRef<unsigned> third = {},
                                        llvm::ArrayRef<unsigned> fourth = {}) {
  llvm::SmallVector<unsigned, 8> result;
  llvm::append_range(result, first);
  llvm::append_range(result, second);
  llvm::append_range(result, third);
  llvm::append_range(result, fourth);
  return result;
}

mlir::FailureOr<mlir::AffineMap> mapForAxes(mlir::MLIRContext *context,
                                            unsigned rank,
                                            llvm::ArrayRef<unsigned> axes) {
  llvm::SmallVector<mlir::AffineExpr, 8> results;
  llvm::SmallBitVector observed(rank, false);
  for (unsigned axis : axes) {
    if (axis >= rank || observed.test(axis))
      return mlir::failure();
    observed.set(axis);
    results.push_back(mlir::getAffineDimExpr(axis, context));
  }
  return mlir::AffineMap::get(rank, 0, results, context);
}

mlir::FailureOr<analysis::ExactIndexSet>
projectDomain(llvm::ArrayRef<IteratorInterval> iterationDomain,
              mlir::AffineMap map) {
  if (!map || map.getNumDims() != iterationDomain.size())
    return mlir::failure();
  llvm::SmallVector<int64_t, 8> offsets;
  llvm::SmallVector<int64_t, 8> sizes;
  llvm::SmallBitVector observed(iterationDomain.size(), false);
  for (mlir::AffineExpr result : map.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(result);
    if (!dimension || dimension.getPosition() >= iterationDomain.size() ||
        observed.test(dimension.getPosition()))
      return mlir::failure();
    observed.set(dimension.getPosition());
    const IteratorInterval &interval = iterationDomain[dimension.getPosition()];
    if (interval.size <= 0)
      return mlir::failure();
    offsets.push_back(interval.offset);
    sizes.push_back(interval.size);
  }
  analysis::IndexSetResult set =
      analysis::IndexRelation::staticRectangularDomain(offsets, sizes);
  if (!set.isExact())
    return mlir::failure();
  analysis::StaticRectangularIndexSet rectangle{
      llvm::SmallVector<int64_t, 4>(offsets.begin(), offsets.end()),
      llvm::SmallVector<int64_t, 4>(sizes.begin(), sizes.end())};
  return analysis::ExactIndexSet(
      std::move(*set.set), analysis::ExactIndexSetForm::BoxUnion, {rectangle});
}

std::optional<AttentionOperandRole> operandRole(unsigned operand,
                                                bool hasMask) {
  switch (operand) {
  case 0:
    return AttentionOperandRole::Query;
  case 1:
    return AttentionOperandRole::Key;
  case 2:
    return AttentionOperandRole::Value;
  case 4:
    if (hasMask)
      return AttentionOperandRole::Mask;
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

mlir::AffineMap operandMap(LinalgExtAttentionOp attention,
                           AttentionOperandRole role) {
  switch (role) {
  case AttentionOperandRole::Query:
    return attention.getQueryMap();
  case AttentionOperandRole::Key:
    return attention.getKeyMap();
  case AttentionOperandRole::Value:
    return attention.getValueMap();
  case AttentionOperandRole::Mask:
    return *attention.getMaskMap();
  }
  return {};
}

struct ScopeContext {
  AttentionWorkScopeId id;
  const analysis::RootRegionWork *work = nullptr;
  const analysis::RootExecutionWork *execution = nullptr;
  const analysis::ReductionMergeRequirement *merge = nullptr;
};

struct ProjectionFacts {
  std::map<PhysicalVersionId, const RepresentationResourceDescription *>
      versionResources;
  std::map<PhysicalVersionId, StorageObjectId> storageBindings;
  std::map<ReductionGatherId, StorageObjectId> gatherStaging;
  std::map<PhysicalVersionId, std::vector<const ReductionGatherPlan *>>
      gathersBySource;
  std::set<ScheduleNodeId> scheduleNodes;
};

} // namespace

CanonicalAttentionWorkProjectionOutcome buildCanonicalAttentionWorkProjection(
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    const CanonicalRepresentationCoordinate &representations,
    const CanonicalMovementCoordinate &movements,
    const CanonicalStorageCoordinate &storage,
    const CanonicalScheduleCoordinate &schedule) {
  ProjectionFacts facts;
  for (const RepresentationResourceDescription &resource :
       representations.resources)
    if (!facts.versionResources.try_emplace(resource.version, &resource).second)
      return broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                    "attention projection received duplicate versions");
  if (facts.versionResources.size() !=
      representations.plan.primaryVersions.size())
    return broken(BrokenAttentionWorkProjectionReason::MissingPhysicalVersion,
                  "representation plan/resource count differs");
  for (const PhysicalVersionStorageBinding &binding :
       storage.plan.versionBindings)
    if (!facts.storageBindings.try_emplace(binding.version, binding.object)
             .second)
      return broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                    "attention projection received duplicate storage binding");
  for (const ReductionGatherStorageBinding &binding :
       storage.plan.gatherStagingBindings)
    if (!facts.gatherStaging.try_emplace(binding.gather, binding.stagingObject)
             .second)
      return broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                    "attention projection received duplicate gather staging");
  for (const ReductionGatherPlan &gather : movements.plan.reductionGathers)
    facts.gathersBySource[gather.source].push_back(&gather);
  for (const ScheduleNodeId &node : schedule.plan.order)
    if (!facts.scheduleNodes.insert(node).second)
      return broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                    "attention projection received duplicate schedule node");
  std::set<ScheduleNodeId> workerNodes;
  for (const ScheduleWorkerBinding &binding : schedule.plan.workerBindings)
    if (binding.worker != NCCWorker::Worker0 ||
        !workerNodes.insert(binding.node).second)
      return broken(BrokenAttentionWorkProjectionReason::MissingScheduleNode,
                    "canonical schedule worker coverage is invalid");
  if (facts.scheduleNodes != workerNodes)
    return broken(BrokenAttentionWorkProjectionReason::MissingScheduleNode,
                  "schedule order/worker coverage differs");

  std::map<SemanticRootKey, std::vector<const analysis::RootRegionWork *>>
      worksByRoot;
  std::map<SemanticRootKey, LinalgExtAttentionOp> attentionByRoot;
  for (const analysis::RootRegionWork &work : rootWorks) {
    auto attention =
        mlir::dyn_cast_or_null<LinalgExtAttentionOp>(work.rootOperation);
    if (!attention)
      continue;
    worksByRoot[work.id.root].push_back(&work);
    auto [position, inserted] =
        attentionByRoot.try_emplace(work.id.root, attention);
    if (!inserted && position->second != attention)
      return broken(BrokenAttentionWorkProjectionReason::PlanWorkMismatch,
                    "one semantic attention root names several operations",
                    work.id.root);
  }

  CanonicalAttentionWorkCoordinate result;
  for (auto &[root, works] : worksByRoot) {
    LinalgExtAttentionOp attention = attentionByRoot.at(root);
    mlir::FailureOr<AttentionIterationRoles> roles =
        attention.getIterationRoles();
    if (mlir::failed(roles))
      return broken(
          BrokenAttentionWorkProjectionReason::InvalidAttentionSemantics,
          "attention iterator roles are invalid", root);
    CoupledReductionDescription components =
        attention.getCoupledReductionDescription();
    if (components.components.size() != 3)
      return broken(
          BrokenAttentionWorkProjectionReason::InvalidAttentionSemantics,
          "attention does not expose three coupled components", root);
    std::map<CoupledReductionComponentKind, CoupledReductionComponent>
        componentByKind;
    for (const CoupledReductionComponent &component : components.components)
      if (!componentByKind.try_emplace(component.kind, component).second)
        return broken(BrokenAttentionWorkProjectionReason::ComponentMismatch,
                      "attention exposes duplicate component kind", root);
    for (CoupledReductionComponentKind kind :
         {CoupledReductionComponentKind::Maximum,
          CoupledReductionComponentKind::Sum,
          CoupledReductionComponentKind::Accumulator})
      if (!componentByKind.count(kind))
        return broken(BrokenAttentionWorkProjectionReason::ComponentMismatch,
                      "attention is missing a component kind", root);

    llvm::SmallVector<unsigned, 8> scoreAxes =
        joinAxes(roles->batch, roles->query, roles->keyValueReduction);
    llvm::SmallVector<unsigned, 8> qkWorkAxes =
        joinAxes(roles->batch, roles->query, roles->queryKeyReduction,
                 roles->keyValueReduction);
    llvm::SmallVector<unsigned, 8> pvWorkAxes =
        joinAxes(roles->batch, roles->query, roles->keyValueReduction,
                 roles->valueOutput);
    mlir::FailureOr<mlir::AffineMap> scoreMap = mapForAxes(
        attention.getContext(), attention.getIterationDomainRank(), scoreAxes);
    mlir::FailureOr<mlir::AffineMap> qkWorkMap = mapForAxes(
        attention.getContext(), attention.getIterationDomainRank(), qkWorkAxes);
    mlir::FailureOr<mlir::AffineMap> pvWorkMap = mapForAxes(
        attention.getContext(), attention.getIterationDomainRank(), pvWorkAxes);
    if (mlir::failed(scoreMap) || mlir::failed(qkWorkMap) ||
        mlir::failed(pvWorkMap))
      return broken(
          BrokenAttentionWorkProjectionReason::InvalidAttentionSemantics,
          "attention action maps are not projected permutations", root);

    auto valueType =
        mlir::dyn_cast<mlir::ShapedType>(attention.getValue().getType());
    auto outputType =
        mlir::dyn_cast<mlir::ShapedType>(attention.getOutput().getType());
    if (!valueType || !outputType)
      return broken(
          BrokenAttentionWorkProjectionReason::InvalidAttentionSemantics,
          "attention value/output types are not shaped", root);

    AttentionWorkDescription description;
    description.root = root;
    description.algorithm = attention.getAlgorithm();
    std::set<AttentionActionId> actionIds;
    std::set<AttentionValueId> valueIds;
    std::optional<CanonicalAttentionWorkProjectionOutcome> projectionFailure;
    std::set<PhysicalVersionId> consumedGatherSources;
    std::map<ReductionGroupId, std::vector<AttentionWorkScopeId>>
        contributionScopes;
    std::map<ReductionGroupId, AttentionWorkScopeId> mergeScopes;
    std::vector<ScopeContext> rootScopes;
    std::vector<ScopeContext> mergedScopes;

    for (const analysis::RootRegionWork *work : works) {
      for (const analysis::RootExecutionWork &execution : work->execution) {
        ExecutionInstanceId executionId{
            RequiredRootExecution{work->id, execution.shard}};
        if (!facts.scheduleNodes.count(ScheduleNodeId{executionId}))
          return broken(
              BrokenAttentionWorkProjectionReason::MissingScheduleNode,
              "attention root execution is absent from schedule", root);
        std::optional<ReductionGroupId> group;
        if (attention.getAlgorithm() == AttentionAlgorithm::FlashDecoding) {
          const analysis::RootContributionWork *found = nullptr;
          for (const analysis::RootContributionWork &contribution :
               work->contributions)
            if (contribution.contribution.shard == execution.shard) {
              if (found || contribution.algebra !=
                               analysis::ReductionAlgebraKind::CoupledReduction)
                return broken(
                    BrokenAttentionWorkProjectionReason::PlanWorkMismatch,
                    "FD execution has ambiguous coupled contribution", root);
              found = &contribution;
            }
          if (!found)
            return broken(BrokenAttentionWorkProjectionReason::PlanWorkMismatch,
                          "FD execution has no coupled contribution", root);
          group = found->group;
        } else if (!work->contributions.empty()) {
          return broken(BrokenAttentionWorkProjectionReason::PlanWorkMismatch,
                        "FA execution unexpectedly has contributions", root);
        }
        AttentionWorkScopeId scope{executionId, group};
        rootScopes.push_back({scope, work, &execution, nullptr});
        if (group)
          contributionScopes[*group].push_back(scope);
      }
      for (const analysis::ReductionMergeRequirement &merge : work->merges) {
        if (attention.getAlgorithm() != AttentionAlgorithm::FlashDecoding ||
            merge.algebra != analysis::ReductionAlgebraKind::CoupledReduction)
          return broken(BrokenAttentionWorkProjectionReason::PlanWorkMismatch,
                        "attention merge is not an FD coupled merge", root);
        ExecutionInstanceId executionId{
            RequiredMergeExecution{work->id, merge.group}};
        if (!facts.scheduleNodes.count(ScheduleNodeId{executionId}))
          return broken(
              BrokenAttentionWorkProjectionReason::MissingScheduleNode,
              "attention merge is absent from schedule", root);
        AttentionWorkScopeId scope{executionId, merge.group};
        if (!mergeScopes.try_emplace(merge.group, scope).second)
          return broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                        "attention has duplicate merge scope", root);
        mergedScopes.push_back({scope, work, nullptr, &merge});
      }
    }
    if (attention.getAlgorithm() == AttentionAlgorithm::FlashDecoding &&
        (mergeScopes.empty() ||
         mergeScopes.size() != contributionScopes.size()))
      return broken(BrokenAttentionWorkProjectionReason::PlanWorkMismatch,
                    "FD contribution and merge groups differ", root);

    auto addValue = [&](AttentionWorkScopeId scope, AttentionValueKind kind,
                        const analysis::ExactIndexSet &domain,
                        mlir::Type elementType, mlir::AffineMap indexingMap,
                        std::optional<PhysicalVersionId> physical =
                            std::nullopt) -> std::optional<AttentionValueId> {
      AttentionValueId id{scope, kind};
      if (!valueIds.insert(id).second) {
        projectionFailure =
            broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                   "attention value ID is duplicated", root);
        return std::nullopt;
      }
      AttentionValueDescription value{id,          domain,   elementType,
                                      indexingMap, physical, std::nullopt};
      if (physical) {
        auto resource = facts.versionResources.find(*physical);
        if (resource == facts.versionResources.end()) {
          projectionFailure = broken(
              BrokenAttentionWorkProjectionReason::MissingPhysicalVersion,
              "attention value has no physical version resource", root);
          return std::nullopt;
        }
        if (resource->second->elementType != elementType ||
            !sameDomain(resource->second->exactDomain, domain)) {
          projectionFailure = broken(
              BrokenAttentionWorkProjectionReason::ComponentMismatch,
              "attention value resource differs from source semantics", root);
          return std::nullopt;
        }
        auto binding = facts.storageBindings.find(*physical);
        if (binding == facts.storageBindings.end()) {
          projectionFailure =
              broken(BrokenAttentionWorkProjectionReason::MissingStorageBinding,
                     "attention value has no storage binding", root);
          return std::nullopt;
        }
        value.storage = binding->second;
      }
      description.values.push_back(std::move(value));
      return id;
    };
    auto addAction = [&](AttentionWorkScopeId scope, AttentionActionKind kind,
                         const analysis::ExactIndexSet &domain,
                         llvm::ArrayRef<AttentionValueId> inputs,
                         llvm::ArrayRef<AttentionValueId> outputs) -> bool {
      AttentionActionId id{scope, kind};
      if (!actionIds.insert(id).second)
        return false;
      description.actions.push_back(
          {id, domain,
           std::vector<AttentionValueId>(inputs.begin(), inputs.end()),
           std::vector<AttentionValueId>(outputs.begin(), outputs.end())});
      return true;
    };

    for (const ScopeContext &scope : rootScopes) {
      mlir::FailureOr<analysis::ExactIndexSet> scoreDomain =
          projectDomain(scope.execution->iterationDomain, *scoreMap);
      mlir::FailureOr<analysis::ExactIndexSet> rowDomain = projectDomain(
          scope.execution->iterationDomain,
          componentByKind.at(CoupledReductionComponentKind::Maximum)
              .indexingMap);
      mlir::FailureOr<analysis::ExactIndexSet> outputDomain = projectDomain(
          scope.execution->iterationDomain, attention.getOutputMap());
      mlir::FailureOr<analysis::ExactIndexSet> qkDomain =
          projectDomain(scope.execution->iterationDomain, *qkWorkMap);
      mlir::FailureOr<analysis::ExactIndexSet> pvDomain =
          projectDomain(scope.execution->iterationDomain, *pvWorkMap);
      if (mlir::failed(scoreDomain) || mlir::failed(rowDomain) ||
          mlir::failed(outputDomain) || mlir::failed(qkDomain) ||
          mlir::failed(pvDomain))
        return broken(BrokenAttentionWorkProjectionReason::ResourceMismatch,
                      "attention execution domains cannot be projected", root);

      mlir::Type maximumType =
          componentByKind.at(CoupledReductionComponentKind::Maximum)
              .elementType;
      mlir::Type sumType =
          componentByKind.at(CoupledReductionComponentKind::Sum).elementType;
      mlir::Type accumulatorType =
          componentByKind.at(CoupledReductionComponentKind::Accumulator)
              .elementType;
      auto addScratch = [&](AttentionValueKind kind,
                            const analysis::ExactIndexSet &domain,
                            mlir::Type type, mlir::AffineMap map) {
        return addValue(scope.id, kind, domain, type, map);
      };
      auto score = addScratch(AttentionValueKind::ScoreBlock, *scoreDomain,
                              maximumType, *scoreMap);
      auto scaled = addScratch(AttentionValueKind::ScaledMaskedScoreBlock,
                               *scoreDomain, maximumType, *scoreMap);
      auto probability =
          addScratch(AttentionValueKind::ProbabilityBlock, *scoreDomain,
                     valueType.getElementType(), *scoreMap);
      auto blockMaximum =
          addScratch(AttentionValueKind::BlockMaximum, *rowDomain, maximumType,
                     componentByKind.at(CoupledReductionComponentKind::Maximum)
                         .indexingMap);
      auto blockSum = addScratch(
          AttentionValueKind::BlockSum, *rowDomain, sumType,
          componentByKind.at(CoupledReductionComponentKind::Sum).indexingMap);
      auto blockAccumulator = addScratch(
          AttentionValueKind::BlockAccumulator, *outputDomain, accumulatorType,
          componentByKind.at(CoupledReductionComponentKind::Accumulator)
              .indexingMap);

      std::optional<PhysicalVersionId> runningMaximumPhysical;
      std::optional<PhysicalVersionId> runningSumPhysical;
      std::optional<PhysicalVersionId> runningAccumulatorPhysical;
      if (scope.id.group) {
        runningMaximumPhysical = PhysicalVersionId{
            CoupledComponentValueId{scope.id.execution, *scope.id.group,
                                    CoupledReductionComponentKind::Maximum}};
        runningSumPhysical = PhysicalVersionId{
            CoupledComponentValueId{scope.id.execution, *scope.id.group,
                                    CoupledReductionComponentKind::Sum}};
        runningAccumulatorPhysical = PhysicalVersionId{CoupledComponentValueId{
            scope.id.execution, *scope.id.group,
            CoupledReductionComponentKind::Accumulator}};
      }
      auto runningMaximum = addValue(
          scope.id, AttentionValueKind::RunningMaximum, *rowDomain, maximumType,
          componentByKind.at(CoupledReductionComponentKind::Maximum)
              .indexingMap,
          runningMaximumPhysical);
      auto runningSum = addValue(
          scope.id, AttentionValueKind::RunningSum, *rowDomain, sumType,
          componentByKind.at(CoupledReductionComponentKind::Sum).indexingMap,
          runningSumPhysical);
      auto runningAccumulator = addValue(
          scope.id, AttentionValueKind::RunningAccumulator, *outputDomain,
          accumulatorType,
          componentByKind.at(CoupledReductionComponentKind::Accumulator)
              .indexingMap,
          runningAccumulatorPhysical);
      if (!score || !scaled || !probability || !blockMaximum || !blockSum ||
          !blockAccumulator || !runningMaximum || !runningSum ||
          !runningAccumulator) {
        if (projectionFailure)
          return std::move(*projectionFailure);
        return broken(BrokenAttentionWorkProjectionReason::ResourceMismatch,
                      "attention value projection is incomplete", root);
      }

      AttentionValueId finalOutput;
      bool hasFinalOutput = false;
      if (!scope.id.group) {
        PhysicalVersionId physical{
            ExecutionResultValueId{scope.id.execution, 0}};
        auto output = addValue(scope.id, AttentionValueKind::FinalOutput,
                               *outputDomain, outputType.getElementType(),
                               attention.getOutputMap(), physical);
        if (!output) {
          if (projectionFailure)
            return std::move(*projectionFailure);
          return broken(
              BrokenAttentionWorkProjectionReason::MissingPhysicalVersion,
              "FA final output has no version/storage projection", root);
        }
        finalOutput = *output;
        hasFinalOutput = true;
      }

      for (const PhysicalVersionPlan &version :
           representations.plan.primaryVersions) {
        const auto *boundary =
            std::get_if<BoundaryRegionValueId>(&version.id.logicalValue);
        if (!boundary || boundary->work != scope.work->id ||
            boundary->fragment.use.destinationShard !=
                std::get<RequiredRootExecution>(scope.id.execution.source)
                    .shard)
          continue;
        std::optional<AttentionOperandRole> role =
            operandRole(boundary->fragment.use.operand,
                        static_cast<bool>(attention.getMask()));
        if (!role)
          continue;
        auto resource = facts.versionResources.find(version.id);
        auto binding = facts.storageBindings.find(version.id);
        if (resource == facts.versionResources.end() ||
            binding == facts.storageBindings.end())
          return broken(
              BrokenAttentionWorkProjectionReason::MissingStorageBinding,
              "attention operand has no version/storage resource", root);
        description.operands.push_back(
            {scope.id, *role, version.id, binding->second,
             resource->second->exactDomain, resource->second->elementType,
             operandMap(attention, *role)});
      }
      for (AttentionOperandRole required :
           {AttentionOperandRole::Query, AttentionOperandRole::Key,
            AttentionOperandRole::Value})
        if (llvm::none_of(description.operands,
                          [&](const AttentionOperandProjection &operand) {
                            return operand.scope == scope.id &&
                                   operand.role == required;
                          }))
          return broken(
              BrokenAttentionWorkProjectionReason::MissingOperandProjection,
              "attention execution is missing Q/K/V operand projection", root);
      if (attention.getMask() &&
          llvm::none_of(description.operands,
                        [&](const AttentionOperandProjection &operand) {
                          return operand.scope == scope.id &&
                                 operand.role == AttentionOperandRole::Mask;
                        }))
        return broken(
            BrokenAttentionWorkProjectionReason::MissingOperandProjection,
            "masked attention execution has no mask projection", root);

      if (!addAction(scope.id, AttentionActionKind::QueryKeyContraction,
                     *qkDomain, {}, {*score}) ||
          !addAction(scope.id, AttentionActionKind::ScaleMask, *scoreDomain,
                     {*score}, {*scaled}) ||
          !addAction(scope.id, AttentionActionKind::RowMaximum, *scoreDomain,
                     {*scaled}, {*blockMaximum}) ||
          !addAction(scope.id, AttentionActionKind::Exponential, *scoreDomain,
                     {*scaled, *blockMaximum}, {*probability}) ||
          !addAction(scope.id, AttentionActionKind::RowSum, *scoreDomain,
                     {*probability}, {*blockSum}) ||
          !addAction(scope.id, AttentionActionKind::ValueContraction, *pvDomain,
                     {*probability}, {*blockAccumulator}) ||
          !addAction(scope.id, AttentionActionKind::StateUpdate, *outputDomain,
                     {*blockMaximum, *blockSum, *blockAccumulator,
                      *runningMaximum, *runningSum, *runningAccumulator},
                     {*runningMaximum, *runningSum, *runningAccumulator}))
        return broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                      "attention execution has duplicate action IDs", root);
      if (hasFinalOutput &&
          !addAction(scope.id, AttentionActionKind::Finalize, *outputDomain,
                     {*runningSum, *runningAccumulator}, {finalOutput}))
        return broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                      "attention execution has duplicate finalize", root);
      description.simultaneousValues.push_back(
          {scope.id, {*blockMaximum, *blockSum, *blockAccumulator}});
      description.simultaneousValues.push_back(
          {scope.id, {*runningMaximum, *runningSum, *runningAccumulator}});
    }

    for (const ScopeContext &scope : mergedScopes) {
      auto contribution = contributionScopes.find(*scope.id.group);
      if (contribution == contributionScopes.end() ||
          contribution->second.empty())
        return broken(BrokenAttentionWorkProjectionReason::PlanWorkMismatch,
                      "attention merge has no contribution scopes", root);
      mlir::Type maximumType =
          componentByKind.at(CoupledReductionComponentKind::Maximum)
              .elementType;
      mlir::Type sumType =
          componentByKind.at(CoupledReductionComponentKind::Sum).elementType;
      mlir::Type accumulatorType =
          componentByKind.at(CoupledReductionComponentKind::Accumulator)
              .elementType;
      auto physicalFor = [&](CoupledReductionComponentKind kind) {
        return PhysicalVersionId{
            CoupledComponentValueId{scope.id.execution, *scope.id.group, kind}};
      };
      PhysicalVersionId maximumPhysical =
          physicalFor(CoupledReductionComponentKind::Maximum);
      PhysicalVersionId sumPhysical =
          physicalFor(CoupledReductionComponentKind::Sum);
      PhysicalVersionId accumulatorPhysical =
          physicalFor(CoupledReductionComponentKind::Accumulator);
      auto maximumResource = facts.versionResources.find(maximumPhysical);
      auto sumResource = facts.versionResources.find(sumPhysical);
      auto accumulatorResource =
          facts.versionResources.find(accumulatorPhysical);
      PhysicalVersionId outputPhysical{
          ExecutionResultValueId{scope.id.execution, 0}};
      auto outputResource = facts.versionResources.find(outputPhysical);
      if (maximumResource == facts.versionResources.end() ||
          sumResource == facts.versionResources.end() ||
          accumulatorResource == facts.versionResources.end() ||
          outputResource == facts.versionResources.end())
        return broken(
            BrokenAttentionWorkProjectionReason::MissingPhysicalVersion,
            "attention merge is missing component/output version", root);
      auto runningMaximum =
          addValue(scope.id, AttentionValueKind::RunningMaximum,
                   maximumResource->second->exactDomain, maximumType,
                   componentByKind.at(CoupledReductionComponentKind::Maximum)
                       .indexingMap,
                   maximumPhysical);
      auto runningSum = addValue(
          scope.id, AttentionValueKind::RunningSum,
          sumResource->second->exactDomain, sumType,
          componentByKind.at(CoupledReductionComponentKind::Sum).indexingMap,
          sumPhysical);
      auto runningAccumulator = addValue(
          scope.id, AttentionValueKind::RunningAccumulator,
          accumulatorResource->second->exactDomain, accumulatorType,
          componentByKind.at(CoupledReductionComponentKind::Accumulator)
              .indexingMap,
          accumulatorPhysical);
      auto finalOutput = addValue(scope.id, AttentionValueKind::FinalOutput,
                                  outputResource->second->exactDomain,
                                  outputType.getElementType(),
                                  attention.getOutputMap(), outputPhysical);
      if (!runningMaximum || !runningSum || !runningAccumulator ||
          !finalOutput) {
        if (projectionFailure)
          return std::move(*projectionFailure);
        return broken(BrokenAttentionWorkProjectionReason::ResourceMismatch,
                      "attention merge value projection is incomplete", root);
      }
      std::vector<AttentionValueId> mergeInputs;
      for (const AttentionWorkScopeId &inputScope : contribution->second) {
        mergeInputs.push_back({inputScope, AttentionValueKind::RunningMaximum});
        mergeInputs.push_back({inputScope, AttentionValueKind::RunningSum});
        mergeInputs.push_back(
            {inputScope, AttentionValueKind::RunningAccumulator});
      }
      if (!addAction(scope.id, AttentionActionKind::StateMerge,
                     outputResource->second->exactDomain, mergeInputs,
                     {*runningMaximum, *runningSum, *runningAccumulator}) ||
          !addAction(scope.id, AttentionActionKind::Finalize,
                     outputResource->second->exactDomain,
                     {*runningSum, *runningAccumulator}, {*finalOutput}))
        return broken(BrokenAttentionWorkProjectionReason::DuplicateIdentity,
                      "attention merge has duplicate action IDs", root);
      description.simultaneousValues.push_back(
          {scope.id, {*runningMaximum, *runningSum, *runningAccumulator}});
    }

    if (attention.getAlgorithm() == AttentionAlgorithm::FlashDecoding) {
      for (const ScopeContext &scope : rootScopes) {
        const AttentionWorkScopeId &scopeId = scope.id;
        AttentionWorkScopeId mergeScope = mergeScopes.at(*scopeId.group);
        for (CoupledReductionComponentKind component :
             {CoupledReductionComponentKind::Maximum,
              CoupledReductionComponentKind::Sum,
              CoupledReductionComponentKind::Accumulator}) {
          AttentionValueKind valueKind =
              component == CoupledReductionComponentKind::Maximum
                  ? AttentionValueKind::RunningMaximum
              : component == CoupledReductionComponentKind::Sum
                  ? AttentionValueKind::RunningSum
                  : AttentionValueKind::RunningAccumulator;
          AttentionValueId value{scopeId, valueKind};
          PhysicalVersionId physical{CoupledComponentValueId{
              scopeId.execution, *scopeId.group, component}};
          bool remote = workOf(scopeId.execution).tile !=
                        workOf(mergeScope.execution).tile;
          auto gathers = facts.gathersBySource.find(physical);
          size_t gatherCount = gathers == facts.gathersBySource.end()
                                   ? 0
                                   : gathers->second.size();
          if ((!remote && gatherCount != 0) || (remote && gatherCount != 1))
            return broken(BrokenAttentionWorkProjectionReason::MovementMismatch,
                          "attention component gather locality is inconsistent",
                          root);
          if (!remote)
            continue;
          const ReductionGatherPlan &gather = *gathers->second.front();
          auto staging = facts.gatherStaging.find(gather.id);
          if (staging == facts.gatherStaging.end() ||
              !facts.scheduleNodes.count(
                  ScheduleNodeId{MovementActionId{gather.id}}))
            return broken(
                BrokenAttentionWorkProjectionReason::MovementMismatch,
                "remote attention component lacks staging/schedule projection",
                root);
          description.gathers.push_back({value, gather.id, staging->second});
          consumedGatherSources.insert(physical);
        }
      }
      for (const auto &[source, gathers] : facts.gathersBySource) {
        const auto *component =
            std::get_if<CoupledComponentValueId>(&source.logicalValue);
        if (component && component->group.root == root &&
            !consumedGatherSources.count(source))
          return broken(BrokenAttentionWorkProjectionReason::MovementMismatch,
                        "attention has an unprojected component gather", root);
      }
    } else if (!description.gathers.empty()) {
      return broken(BrokenAttentionWorkProjectionReason::MovementMismatch,
                    "FA unexpectedly projects component gathers", root);
    }

    llvm::sort(description.actions, [](const AttentionActionDescription &lhs,
                                       const AttentionActionDescription &rhs) {
      return lhs.id < rhs.id;
    });
    llvm::sort(description.values, [](const AttentionValueDescription &lhs,
                                      const AttentionValueDescription &rhs) {
      return lhs.id < rhs.id;
    });
    llvm::sort(description.operands, [](const AttentionOperandProjection &lhs,
                                        const AttentionOperandProjection &rhs) {
      if (!(lhs.scope == rhs.scope))
        return lhs.scope < rhs.scope;
      if (lhs.role != rhs.role)
        return lhs.role < rhs.role;
      return lhs.version < rhs.version;
    });
    llvm::sort(description.gathers, [](const AttentionGatherProjection &lhs,
                                       const AttentionGatherProjection &rhs) {
      return lhs.value < rhs.value;
    });
    for (AttentionSimultaneousValueGroup &group :
         description.simultaneousValues)
      llvm::sort(group.values);
    llvm::sort(description.simultaneousValues,
               [](const AttentionSimultaneousValueGroup &lhs,
                  const AttentionSimultaneousValueGroup &rhs) {
                 if (!(lhs.scope == rhs.scope))
                   return lhs.scope < rhs.scope;
                 return std::lexicographical_compare(
                     lhs.values.begin(), lhs.values.end(), rhs.values.begin(),
                     rhs.values.end());
               });
    result.roots.push_back(std::move(description));
  }

  llvm::sort(result.roots, [](const AttentionWorkDescription &lhs,
                              const AttentionWorkDescription &rhs) {
    return lhs.root < rhs.root;
  });
  return result;
}

} // namespace wafer::compiler::detail
