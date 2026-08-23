//===- RepresentationDomain.cpp - Physical version layout domain -----===//

#include "Wafer/Planning/PhysicalDataflow/RepresentationDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <set>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

RepresentationDomainResult
failed(RepresentationDomainFailureKind kind, llvm::StringRef detail,
       std::optional<RegionValueVersionId> value = {}) {
  return {{},
          RepresentationDomainFailure{kind, std::move(value), detail.str()}};
}

std::vector<MemLayout>
getLegalLayouts(const RepresentationResourceDescription &resource) {
  std::vector<MemLayout> layouts;
  if (!resource.elementType || !resource.elementType.isIntOrFloat() ||
      resource.exactDomain.isEmpty())
    return layouts;
  for (MemLayout layout :
       {MemLayout::Tensor, MemLayout::NTensor, MemLayout::Cx, MemLayout::NCx}) {
    bool legal = true;
    for (const analysis::StaticRectangularIndexSet &box :
         resource.exactDomain.getBoxes()) {
      auto type = mlir::MemRefType::get(
          box.sizes, resource.elementType, mlir::MemRefLayoutAttrInterface{},
          MemoryAttr::get(resource.elementType.getContext(), MemorySpace::SPM,
                          layout));
      if (mlir::failed(analysis::PhysicalLayoutRelation::create(type))) {
        legal = false;
        break;
      }
    }
    if (legal)
      layouts.push_back(layout);
  }
  return layouts;
}

bool versionOrder(const PhysicalVersionPlan &lhs,
                  const PhysicalVersionPlan &rhs) {
  if (lhs.id.derivation.size() != rhs.id.derivation.size())
    return lhs.id.derivation.size() < rhs.id.derivation.size();
  return lhs.id < rhs.id;
}

bool haveSameExactResource(const RepresentationResourceDescription &lhs,
                           const RepresentationResourceDescription &rhs) {
  if (lhs.elementType != rhs.elementType ||
      lhs.exactDomain.getRank() != rhs.exactDomain.getRank() ||
      lhs.exactDomain.getBoxes().size() != rhs.exactDomain.getBoxes().size())
    return false;
  for (auto [lhsBox, rhsBox] :
       llvm::zip_equal(lhs.exactDomain.getBoxes(), rhs.exactDomain.getBoxes()))
    if (lhsBox.offsets != rhsBox.offsets || lhsBox.sizes != rhsBox.sizes)
      return false;
  return true;
}

} // namespace

std::vector<RepresentationDomain::UseOption>
RepresentationDomain::getUseOptions(
    size_t index, llvm::ArrayRef<uint32_t> primaryLayoutIndices) const {
  std::vector<UseOption> options;
  const UseDomain &use = uses[index];
  const PhysicalVersionId &primary = values[use.valueIndex].primary;
  const MemLayout primaryEncoding =
      values[use.valueIndex].layouts[primaryLayoutIndices[use.valueIndex]];
  options.push_back({primary, primaryEncoding});
  for (MemLayout layout : use.layouts) {
    if (layout == primaryEncoding)
      continue;
    PhysicalVersionId shared = primary;
    shared.derivation.push_back(
        {PhysicalVersionDerivationKind::LayoutConversion, primaryEncoding,
         layout, SharedRepresentationAnchor{}});
    options.push_back({std::move(shared), layout});
    PhysicalVersionId local = primary;
    local.derivation.push_back({PhysicalVersionDerivationKind::LayoutConversion,
                                primaryEncoding, layout,
                                UseRepresentationAnchor{use.use}});
    options.push_back({std::move(local), layout});
  }
  if (use.aliasSourceIndex) {
    const ValueDomain &source = values[*use.aliasSourceIndex];
    const MemLayout sourceEncoding =
        source.layouts[primaryLayoutIndices[*use.aliasSourceIndex]];
    if (llvm::is_contained(use.layouts, sourceEncoding)) {
      PhysicalVersionId alias = primary;
      alias.derivation.push_back(
          {PhysicalVersionDerivationKind::AliasView, sourceEncoding,
           sourceEncoding, UseRepresentationAnchor{use.use}, source.value});
      options.push_back({std::move(alias), sourceEncoding});
    }
  }
  return options;
}

std::optional<RepresentationPlan>
RepresentationDomain::buildPlan(const RepresentationCursor &cursor) const {
  if (cursor.primaryLayoutIndices.size() != values.size() ||
      cursor.useOptionIndices.size() != uses.size() ||
      !isPrimaryAssignmentLegal(cursor))
    return std::nullopt;
  RepresentationPlan plan;
  std::map<PhysicalVersionId, MemLayout> versions;
  for (auto [index, value] : llvm::enumerate(values)) {
    if (cursor.primaryLayoutIndices[index] >= value.layouts.size())
      return std::nullopt;
    const MemLayout encoding =
        value.layouts[cursor.primaryLayoutIndices[index]];
    plan.logicalValues.push_back({value.value, value.primary});
    if (!versions.try_emplace(value.primary, encoding).second)
      return std::nullopt;
  }
  for (auto [index, use] : llvm::enumerate(uses)) {
    std::vector<UseOption> options =
        getUseOptions(index, cursor.primaryLayoutIndices);
    if (cursor.useOptionIndices[index] >= options.size())
      return std::nullopt;
    const UseOption &selected = options[cursor.useOptionIndices[index]];
    plan.uses.push_back({use.use, selected.version});
    auto [entry, inserted] =
        versions.try_emplace(selected.version, selected.encoding);
    if (!inserted && entry->second != selected.encoding)
      return std::nullopt;
  }
  for (const auto &[id, encoding] : versions)
    plan.physicalVersions.push_back({id, encoding});
  llvm::sort(plan.logicalValues);
  llvm::sort(plan.physicalVersions, versionOrder);
  llvm::sort(plan.uses);
  return plan;
}

bool RepresentationDomain::isPrimaryAssignmentLegal(
    const RepresentationCursor &cursor) const {
  if (cursor.primaryLayoutIndices.size() != values.size())
    return false;
  for (const TupleConstraint &constraint : constraints) {
    std::vector<MemLayout> tuple;
    for (size_t valueIndex : constraint.valueIndices) {
      if (valueIndex >= values.size() ||
          cursor.primaryLayoutIndices[valueIndex] >=
              values[valueIndex].layouts.size())
        return false;
      tuple.push_back(
          values[valueIndex].layouts[cursor.primaryLayoutIndices[valueIndex]]);
    }
    if (!llvm::is_contained(constraint.legalTuples, tuple))
      return false;
  }
  return true;
}

bool RepresentationDomain::advanceCursor(RepresentationCursor &next) const {
  if (isPrimaryAssignmentLegal(next)) {
    for (size_t reverse = 0; reverse < uses.size(); ++reverse) {
      const size_t index = uses.size() - reverse - 1;
      const size_t optionCount =
          getUseOptions(index, next.primaryLayoutIndices).size();
      if (++next.useOptionIndices[index] < optionCount) {
        for (size_t reset = index + 1; reset < uses.size(); ++reset)
          next.useOptionIndices[reset] = 0;
        return true;
      }
      next.useOptionIndices[index] = 0;
    }
  }
  for (size_t reverse = 0; reverse < values.size(); ++reverse) {
    const size_t index = values.size() - reverse - 1;
    if (++next.primaryLayoutIndices[index] < values[index].layouts.size()) {
      for (size_t reset = index + 1; reset < values.size(); ++reset)
        next.primaryLayoutIndices[reset] = 0;
      std::fill(next.useOptionIndices.begin(), next.useOptionIndices.end(), 0);
      return true;
    }
    next.primaryLayoutIndices[index] = 0;
  }
  return false;
}

RepresentationProposalResult
RepresentationDomain::getPBQPProposal(uint64_t workLimit) const {
  RepresentationPBQPProblem problem;
  problem.variables.reserve(values.size() + uses.size() + constraints.size());
  for (const ValueDomain &value : values)
    problem.variables.push_back(RepresentationPBQPVariable{
        std::vector<RepresentationPBQPCost>(value.layouts.size(), 0)});

  struct UseState {
    uint32_t valueState = 0;
    std::optional<uint32_t> aliasSourceState;
    uint32_t option = 0;
  };
  std::vector<std::vector<UseState>> useStates(uses.size());
  for (auto [useIndex, use] : llvm::enumerate(uses)) {
    std::vector<uint32_t> primary(values.size(), 0);
    const uint32_t aliasStates =
        use.aliasSourceIndex ? static_cast<uint32_t>(
                                   values[*use.aliasSourceIndex].layouts.size())
                             : 1;
    for (uint32_t valueState = 0;
         valueState < values[use.valueIndex].layouts.size(); ++valueState) {
      primary[use.valueIndex] = valueState;
      for (uint32_t aliasState = 0; aliasState < aliasStates; ++aliasState) {
        if (use.aliasSourceIndex)
          primary[*use.aliasSourceIndex] = aliasState;
        std::vector<UseOption> options = getUseOptions(useIndex, primary);
        for (uint32_t option = 0; option < options.size(); ++option)
          useStates[useIndex].push_back(
              {valueState,
               use.aliasSourceIndex ? std::optional<uint32_t>(aliasState)
                                    : std::nullopt,
               option});
      }
    }
    if (useStates[useIndex].empty())
      return {};
    const uint32_t useVariable =
        static_cast<uint32_t>(problem.variables.size());
    problem.variables.push_back(RepresentationPBQPVariable{
        std::vector<RepresentationPBQPCost>(useStates[useIndex].size(), 0)});
    auto appendEqualityFactor = [&](size_t valueIndex, bool aliasSource) {
      RepresentationPBQPBinaryFactor factor;
      factor.lhs = static_cast<uint32_t>(valueIndex);
      factor.rhs = useVariable;
      factor.lhsStates =
          static_cast<uint32_t>(values[valueIndex].layouts.size());
      factor.rhsStates = static_cast<uint32_t>(useStates[useIndex].size());
      for (uint32_t state = 0; state < factor.lhsStates; ++state)
        for (const UseState &useState : useStates[useIndex]) {
          const uint32_t selected =
              aliasSource ? *useState.aliasSourceState : useState.valueState;
          factor.costs.push_back(
              state == selected ? 0 : kRepresentationPBQPInfinity);
        }
      problem.factors.push_back(std::move(factor));
    };
    appendEqualityFactor(use.valueIndex, /*aliasSource=*/false);
    if (use.aliasSourceIndex)
      appendEqualityFactor(*use.aliasSourceIndex, /*aliasSource=*/true);
  }

  for (const TupleConstraint &constraint : constraints) {
    if (constraint.valueIndices.empty() || constraint.legalTuples.empty())
      return {};
    const uint32_t tupleVariable =
        static_cast<uint32_t>(problem.variables.size());
    problem.variables.push_back(RepresentationPBQPVariable{
        std::vector<RepresentationPBQPCost>(constraint.legalTuples.size(), 0)});
    for (auto [position, valueIndex] :
         llvm::enumerate(constraint.valueIndices)) {
      if (valueIndex >= values.size())
        return {};
      RepresentationPBQPBinaryFactor factor;
      factor.lhs = static_cast<uint32_t>(valueIndex);
      factor.rhs = tupleVariable;
      factor.lhsStates =
          static_cast<uint32_t>(values[valueIndex].layouts.size());
      factor.rhsStates = static_cast<uint32_t>(constraint.legalTuples.size());
      factor.costs.reserve(static_cast<size_t>(factor.lhsStates) *
                           factor.rhsStates);
      for (MemLayout layout : values[valueIndex].layouts)
        for (const std::vector<MemLayout> &tuple : constraint.legalTuples)
          factor.costs.push_back(
              tuple[position] == layout ? 0 : kRepresentationPBQPInfinity);
      problem.factors.push_back(std::move(factor));
    }
  }

  RepresentationPBQPResult solved = solveRepresentationPBQP(problem, workLimit);
  RepresentationProposalResult result;
  result.status = solved.status;
  result.cost = solved.cost;
  result.work = solved.work;
  if (solved.status != RepresentationPBQPStatus::Optimal ||
      solved.assignment.size() < values.size())
    return result;
  RepresentationCursor cursor;
  cursor.primaryLayoutIndices.assign(solved.assignment.begin(),
                                     solved.assignment.begin() + values.size());
  cursor.useOptionIndices.resize(uses.size());
  for (size_t use = 0; use < uses.size(); ++use) {
    const uint32_t state = solved.assignment[values.size() + use];
    if (state >= useStates[use].size()) {
      result.status = RepresentationPBQPStatus::BrokenContract;
      result.plan.reset();
      return result;
    }
    cursor.useOptionIndices[use] = useStates[use][state].option;
  }
  result.plan = buildPlan(cursor);
  if (!result.plan)
    result.status = RepresentationPBQPStatus::BrokenContract;
  return result;
}

RepresentationSuccessor RepresentationDomain::getFirstPlan() const {
  RepresentationCursor cursor;
  cursor.primaryLayoutIndices.assign(values.size(), 0);
  cursor.useOptionIndices.assign(uses.size(), 0);
  while (!isPrimaryAssignmentLegal(cursor))
    if (!advanceCursor(cursor))
      return {RepresentationSuccessorKind::End};
  std::optional<RepresentationPlan> plan = buildPlan(cursor);
  if (!plan)
    return {RepresentationSuccessorKind::CompilerBug,
            {},
            {},
            "representation domain has no valid first plan"};
  return {RepresentationSuccessorKind::Plan, std::move(plan),
          std::move(cursor)};
}

std::optional<RepresentationCursor>
RepresentationDomain::getCursor(const RepresentationPlan &plan) const {
  if (plan.logicalValues.size() != values.size())
    return std::nullopt;
  RepresentationCursor cursor;
  cursor.primaryLayoutIndices.resize(values.size());
  cursor.useOptionIndices.resize(uses.size());
  for (auto [index, value] : llvm::enumerate(values)) {
    auto logical = llvm::find_if(
        plan.logicalValues, [&](const LogicalRepresentationPlan &candidate) {
          return candidate.value == value.value &&
                 candidate.primary == value.primary;
        });
    auto physical = llvm::find_if(plan.physicalVersions,
                                  [&](const PhysicalVersionPlan &candidate) {
                                    return candidate.id == value.primary;
                                  });
    if (logical == plan.logicalValues.end() ||
        physical == plan.physicalVersions.end())
      return std::nullopt;
    auto layout = llvm::find(value.layouts, physical->encoding);
    if (layout == value.layouts.end())
      return std::nullopt;
    cursor.primaryLayoutIndices[index] =
        static_cast<uint32_t>(std::distance(value.layouts.begin(), layout));
  }
  for (auto [index, use] : llvm::enumerate(uses)) {
    auto binding =
        llvm::find_if(plan.uses, [&](const PhysicalUseBinding &candidate) {
          return candidate.use == use.use;
        });
    if (binding == plan.uses.end())
      return std::nullopt;
    std::vector<UseOption> options =
        getUseOptions(index, cursor.primaryLayoutIndices);
    auto selected = llvm::find_if(options, [&](const UseOption &option) {
      return option.version == binding->version;
    });
    if (selected == options.end())
      return std::nullopt;
    cursor.useOptionIndices[index] =
        static_cast<uint32_t>(std::distance(options.begin(), selected));
  }
  std::optional<RepresentationPlan> rebuilt = buildPlan(cursor);
  if (!rebuilt || !(*rebuilt == plan))
    return std::nullopt;
  return cursor;
}

bool RepresentationDomain::contains(const RepresentationPlan &plan) const {
  return getCursor(plan).has_value();
}

RepresentationSuccessor
RepresentationDomain::getNextPlan(const RepresentationCursor &cursor) const {
  std::optional<RepresentationPlan> current = buildPlan(cursor);
  if (!current || !contains(*current))
    return {RepresentationSuccessorKind::CompilerBug,
            {},
            {},
            "representation cursor is outside the current domain"};
  RepresentationCursor next = cursor;
  while (advanceCursor(next)) {
    if (!isPrimaryAssignmentLegal(next))
      continue;
    std::optional<RepresentationPlan> plan = buildPlan(next);
    if (!plan)
      return {RepresentationSuccessorKind::CompilerBug,
              {},
              {},
              "representation successor is malformed"};
    return {RepresentationSuccessorKind::Plan, std::move(plan),
            std::move(next)};
  }
  return {RepresentationSuccessorKind::End};
}

const RepresentationResourceDescription *
RepresentationDomain::findResource(const RegionValueVersionId &value) const {
  auto found = llvm::find_if(values, [&](const ValueDomain &candidate) {
    return candidate.value == value;
  });
  return found == values.end() ? nullptr : &found->resource;
}

RepresentationDomainResult buildRepresentationDomain(
    const CanonicalRepresentationCoordinate &canonical,
    llvm::ArrayRef<IdentityAliasRequirement> aliases,
    llvm::ArrayRef<RepresentationLayoutTupleConstraint> constraints) {
  if (canonical.plan.logicalValues.size() != canonical.resources.size())
    return failed(RepresentationDomainFailureKind::BrokenContract,
                  "representation logical/resource inventories disagree");
  std::map<RegionValueVersionId, RepresentationResourceDescription> resources;
  for (const RepresentationResourceDescription &resource : canonical.resources)
    if (!resources.try_emplace(resource.version.logicalValue, resource).second)
      return failed(RepresentationDomainFailureKind::BrokenContract,
                    "representation resources contain a duplicate value",
                    resource.version.logicalValue);

  std::vector<RepresentationDomain::ValueDomain> values;
  for (const LogicalRepresentationPlan &logical :
       canonical.plan.logicalValues) {
    if (!(logical.primary.logicalValue == logical.value) ||
        !logical.primary.derivation.empty())
      return failed(RepresentationDomainFailureKind::BrokenContract,
                    "logical representation has a malformed primary",
                    logical.value);
    auto resource = resources.find(logical.value);
    if (resource == resources.end())
      return failed(RepresentationDomainFailureKind::BrokenContract,
                    "logical representation has no exact resource",
                    logical.value);
    std::vector<MemLayout> layouts = getLegalLayouts(resource->second);
    if (layouts.empty())
      return failed(RepresentationDomainFailureKind::UnsupportedSemantics,
                    "logical value has no legal physical encoding",
                    logical.value);
    values.push_back(
        {logical.value, logical.primary, resource->second, std::move(layouts)});
  }
  llvm::sort(values, [](const RepresentationDomain::ValueDomain &lhs,
                        const RepresentationDomain::ValueDomain &rhs) {
    return lhs.primary < rhs.primary;
  });

  std::vector<RepresentationDomain::UseDomain> uses;
  std::set<RepresentationUseId> observedUses;
  for (const PhysicalUseBinding &binding : canonical.plan.uses) {
    if (!binding.version.derivation.empty() ||
        !observedUses.insert(binding.use).second)
      return failed(RepresentationDomainFailureKind::BrokenContract,
                    "canonical representation has a malformed use binding",
                    binding.version.logicalValue);
    auto value = llvm::find_if(values, [&](const auto &candidate) {
      return candidate.value == binding.version.logicalValue;
    });
    if (value == values.end())
      return failed(RepresentationDomainFailureKind::BrokenContract,
                    "representation use references a missing logical value",
                    binding.version.logicalValue);
    uses.push_back({binding.use,
                    static_cast<size_t>(std::distance(values.begin(), value)),
                    value->layouts, std::nullopt});
  }
  llvm::sort(uses, [](const RepresentationDomain::UseDomain &lhs,
                      const RepresentationDomain::UseDomain &rhs) {
    return lhs.use < rhs.use;
  });
  std::set<RepresentationUseId> aliasUses;
  for (const IdentityAliasRequirement &alias : aliases) {
    if (!aliasUses.insert(alias.use).second)
      return failed(RepresentationDomainFailureKind::BrokenContract,
                    "representation alias use is duplicated");
    auto use = llvm::find_if(uses, [&](const auto &candidate) {
      return candidate.use == alias.use;
    });
    auto source = llvm::find_if(values, [&](const auto &candidate) {
      return candidate.value == alias.source;
    });
    if (use == uses.end() || source == values.end())
      return failed(RepresentationDomainFailureKind::BrokenContract,
                    "representation alias references a missing use or source");
    const size_t sourceIndex =
        static_cast<size_t>(std::distance(values.begin(), source));
    if (sourceIndex == use->valueIndex ||
        !haveSameExactResource(values[use->valueIndex].resource,
                               source->resource))
      return failed(RepresentationDomainFailureKind::UnsupportedSemantics,
                    "representation alias is not an exact identity view",
                    values[use->valueIndex].value);
    use->aliasSourceIndex = sourceIndex;
  }
  std::vector<RepresentationDomain::TupleConstraint> normalizedConstraints;
  for (const RepresentationLayoutTupleConstraint &constraint : constraints) {
    if (constraint.values.empty() || constraint.legalTuples.empty())
      return failed(RepresentationDomainFailureKind::BrokenContract,
                    "representation tuple constraint is empty");
    RepresentationDomain::TupleConstraint normalized;
    std::set<size_t> seen;
    for (const RegionValueVersionId &valueId : constraint.values) {
      auto value = llvm::find_if(values, [&](const auto &candidate) {
        return candidate.value == valueId;
      });
      if (value == values.end())
        return failed(RepresentationDomainFailureKind::BrokenContract,
                      "representation tuple references a missing value",
                      valueId);
      const size_t index =
          static_cast<size_t>(std::distance(values.begin(), value));
      if (!seen.insert(index).second)
        return failed(RepresentationDomainFailureKind::BrokenContract,
                      "representation tuple repeats one value", valueId);
      normalized.valueIndices.push_back(index);
    }
    for (const std::vector<MemLayout> &tuple : constraint.legalTuples) {
      if (tuple.size() != normalized.valueIndices.size())
        return failed(RepresentationDomainFailureKind::BrokenContract,
                      "representation tuple rank is inconsistent");
      bool legal = true;
      for (auto [layout, valueIndex] :
           llvm::zip_equal(tuple, normalized.valueIndices))
        legal &= llvm::is_contained(values[valueIndex].layouts, layout);
      if (legal)
        normalized.legalTuples.push_back(tuple);
    }
    llvm::sort(normalized.legalTuples);
    normalized.legalTuples.erase(std::unique(normalized.legalTuples.begin(),
                                             normalized.legalTuples.end()),
                                 normalized.legalTuples.end());
    if (normalized.legalTuples.empty())
      return failed(RepresentationDomainFailureKind::UnsupportedSemantics,
                    "representation tuple constraint has no legal state");
    normalizedConstraints.push_back(std::move(normalized));
  }
  return {RepresentationDomain(std::move(values), std::move(uses),
                               std::move(normalizedConstraints)),
          {}};
}

} // namespace wafer::compiler::detail
