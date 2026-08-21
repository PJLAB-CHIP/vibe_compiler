//===- RootRegionWorkAnalysis.cpp - Single-root work query -------------===//

#include "Wafer/Planning/PhysicalDataflow/RootRegionWorkAnalysis.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace wafer::compiler::detail {
namespace {

using analysis::BrokenRootRegionWork;
using analysis::BrokenRootRegionWorkReason;
using analysis::ConstantSource;
using analysis::ExactIndexSet;
using analysis::ExactIndexSetForm;
using analysis::NoRootRegionWork;
using analysis::ProgramInputSource;
using analysis::RootBoundaryId;
using analysis::RootBoundaryKind;
using analysis::RootBoundaryWork;
using analysis::RootContributionWork;
using analysis::RootExecutionWork;
using analysis::RootInvariantInputWork;
using analysis::RootInvariantUseKind;
using analysis::RootOperandUseWork;
using analysis::RootOperandWork;
using analysis::RootRegionWork;
using analysis::RootRegionWorkId;
using analysis::RootRegionWorkLimitReached;
using analysis::RootRegionWorkOutcome;
using analysis::RootResultWork;
using analysis::RootSupportInputKind;
using analysis::RootSupportInputWork;
using analysis::RootSupportValueWork;
using analysis::RootUseId;
using analysis::StaticRectangularIndexSet;
using analysis::StructuredResultSource;
using analysis::SupportValueId;
using analysis::UnsupportedRootRegionWork;
using analysis::UnsupportedRootRegionWorkReason;

RootRegionWorkOutcome broken(const RootRegionWorkId &site,
                             BrokenRootRegionWorkReason reason,
                             llvm::StringRef detail) {
  return BrokenRootRegionWork{reason, site, detail.str()};
}

RootRegionWorkOutcome unsupported(const RootRegionWorkId &site,
                                  UnsupportedRootRegionWorkReason reason,
                                  llvm::StringRef detail) {
  return UnsupportedRootRegionWork{reason, site, detail.str()};
}

bool boxLess(const StaticRectangularIndexSet &lhs,
             const StaticRectangularIndexSet &rhs) {
  if (lhs.offsets != rhs.offsets)
    return std::lexicographical_compare(lhs.offsets.begin(), lhs.offsets.end(),
                                        rhs.offsets.begin(), rhs.offsets.end());
  return std::lexicographical_compare(lhs.sizes.begin(), lhs.sizes.end(),
                                      rhs.sizes.begin(), rhs.sizes.end());
}

bool sameBox(const StaticRectangularIndexSet &lhs,
             const StaticRectangularIndexSet &rhs) {
  return lhs.offsets == rhs.offsets && lhs.sizes == rhs.sizes;
}

using ExactSetMerge = std::variant<ExactIndexSet, BrokenRootRegionWork,
                                   RootRegionWorkLimitReached>;

ExactSetMerge mergeExactSets(const ExactIndexSet &lhs, const ExactIndexSet &rhs,
                             const analysis::IndexRelationLimits &limits,
                             const RootRegionWorkId &site) {
  if (lhs.getRank() != rhs.getRank())
    return BrokenRootRegionWork{
        BrokenRootRegionWorkReason::InterfaceContradiction, site,
        "cannot merge root-work exact sets with different ranks"};
  if (lhs.getForm() == ExactIndexSetForm::BoxUnion &&
      rhs.getForm() == ExactIndexSetForm::BoxUnion &&
      lhs.getBoxes().size() == rhs.getBoxes().size() &&
      llvm::equal(lhs.getBoxes(), rhs.getBoxes(), sameBox))
    return lhs;
  const uint64_t predictedDisjuncts =
      static_cast<uint64_t>(lhs.getPresburgerSet().getNumDisjuncts()) +
      rhs.getPresburgerSet().getNumDisjuncts();
  if (predictedDisjuncts > limits.maxDisjuncts)
    return RootRegionWorkLimitReached{
        site, predictedDisjuncts, limits.maxDisjuncts,
        "root-work exact union exceeds the disjunct limit"};

  mlir::presburger::PresburgerSet set =
      lhs.getPresburgerSet().unionSet(rhs.getPresburgerSet());
  if (lhs.getForm() != ExactIndexSetForm::BoxUnion ||
      rhs.getForm() != ExactIndexSetForm::BoxUnion)
    return ExactIndexSet(
        std::move(set),
        lhs.getForm() == rhs.getForm() ? lhs.getForm()
                                       : ExactIndexSetForm::GeneralPresburger);

  llvm::SmallVector<StaticRectangularIndexSet, 8> boxes;
  boxes.append(lhs.getBoxes().begin(), lhs.getBoxes().end());
  boxes.append(rhs.getBoxes().begin(), rhs.getBoxes().end());
  llvm::sort(boxes, boxLess);
  boxes.erase(std::unique(boxes.begin(), boxes.end(), sameBox), boxes.end());
  if (boxes.size() > limits.maxRectangularPieces)
    return RootRegionWorkLimitReached{
        site, boxes.size(), limits.maxRectangularPieces,
        "root-work exact union exceeds the rectangle-piece limit"};
  return ExactIndexSet(std::move(set), ExactIndexSetForm::BoxUnion, boxes);
}

struct BoundaryDescription {
  RootBoundaryId id;
  mlir::Value value;
};

std::optional<BoundaryDescription>
describeBoundary(const analysis::DemandSource &source,
                 const StructuredDAGAnalysis &dag) {
  if (const auto *structured = std::get_if<StructuredResultSource>(&source)) {
    if (!structured->operation ||
        structured->result >= structured->operation->getNumResults())
      return std::nullopt;
    return BoundaryDescription{
        {RootBoundaryKind::StructuredResult, structured->root,
         structured->result},
        structured->operation->getResult(structured->result)};
  }
  if (const auto *input = std::get_if<ProgramInputSource>(&source)) {
    mlir::func::FuncOp function = dag.getFunction();
    if (!function || function.isExternal() ||
        input->argument >= function.getNumArguments())
      return std::nullopt;
    return BoundaryDescription{
        {RootBoundaryKind::ProgramInput, {}, input->argument},
        function.getArgument(input->argument)};
  }
  const auto &constant = std::get<ConstantSource>(source);
  if (!constant.operation ||
      constant.result >= constant.operation->getNumResults())
    return std::nullopt;
  return BoundaryDescription{
      {RootBoundaryKind::Constant, constant.valuePath, constant.result},
      constant.operation->getResult(constant.result)};
}

std::optional<BoundaryDescription> describeInvariantBoundary(
    mlir::Value value, const StructuredDemandView &view,
    std::optional<SemanticRootKey> fallbackKey = std::nullopt) {
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    return BoundaryDescription{{RootBoundaryKind::ProgramInput,
                                {},
                                static_cast<uint32_t>(argument.getArgNumber())},
                               value};
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  const SemanticValueBinding *binding = view.getValueBinding(value);
  if (!result || (!binding && !fallbackKey))
    return std::nullopt;
  const RootBoundaryKind kind =
      result.getOwner()->hasTrait<mlir::OpTrait::ConstantLike>()
          ? RootBoundaryKind::Constant
          : RootBoundaryKind::CapturedValue;
  return BoundaryDescription{{kind, binding ? binding->key : *fallbackKey,
                              static_cast<uint32_t>(result.getResultNumber())},
                             value};
}

std::optional<SemanticRootKey>
deriveRegionCaptureKey(mlir::Operation *root, mlir::Value capture,
                       const SemanticRootKey &rootKey) {
  if (!root || root->getNumRegions() != 1 || root->getRegion(0).empty() ||
      !root->getRegion(0).hasOneBlock())
    return std::nullopt;
  mlir::Block &body = root->getRegion(0).front();
  mlir::Operation *terminator = body.getTerminator();
  if (!terminator)
    return std::nullopt;

  llvm::DenseMap<mlir::Value, SemanticRootKey> paths;
  llvm::SmallVector<mlir::Value, 16> worklist;
  auto update = [&](mlir::Value value, const SemanticRootKey &candidate) {
    auto existing = paths.find(value);
    if (existing != paths.end() && !(candidate < existing->second))
      return;
    paths[value] = candidate;
    worklist.push_back(value);
  };
  for (auto [result, yielded] : llvm::enumerate(terminator->getOperands())) {
    SemanticRootKey candidate = rootKey;
    candidate.path.push_back({SemanticRootPathRelation::RegionBranch,
                              static_cast<uint32_t>(result),
                              static_cast<uint32_t>(result)});
    update(yielded, candidate);
  }

  std::optional<SemanticRootKey> best;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    SemanticRootKey path = paths.lookup(value);
    if (value == capture) {
      if (!best || path < *best)
        best = std::move(path);
      continue;
    }
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    mlir::Operation *owner = result ? result.getOwner() : nullptr;
    if (!owner || !root->isProperAncestor(owner))
      continue;
    for (mlir::OpOperand &operand : owner->getOpOperands()) {
      SemanticRootKey candidate = path;
      candidate.path.push_back(
          {SemanticRootPathRelation::SSAUseDef,
           static_cast<uint32_t>(result.getResultNumber()),
           static_cast<uint32_t>(operand.getOperandNumber())});
      update(operand.get(), candidate);
    }
  }
  return best;
}

struct SupportInputKey {
  uint32_t operand = 0;
  RootSupportInputKind kind = RootSupportInputKind::ExactEmpty;
  std::optional<SupportValueId> supportValue;
  std::optional<RootBoundaryId> boundary;

  friend bool operator<(const SupportInputKey &lhs,
                        const SupportInputKey &rhs) {
    return std::tie(lhs.operand, lhs.kind, lhs.supportValue, lhs.boundary) <
           std::tie(rhs.operand, rhs.kind, rhs.supportValue, rhs.boundary);
  }
};

struct PendingSupportValue {
  RootSupportValueWork work;
  bool hasRequiredDomain = false;
  std::map<SupportInputKey, RootSupportInputWork> inputs;
};

struct SelectedRecipe {
  const analysis::DestinationDemand *destination = nullptr;
  RootUseId use;
};

class WorkBuilder {
public:
  WorkBuilder(const StructuredDAGAnalysis &dag,
              const SpatialAssignment &assignment,
              const analysis::ExactDemandProof &proof,
              const StructuredDemandView &view, RootRegionWorkId site,
              const analysis::IndexRelationLimits &limits)
      : dag(dag), assignment(assignment), proof(proof), view(view), site(site),
        limits(limits) {
    work.id = site;
  }

  RootRegionWorkOutcome build() {
    const SemanticRootBinding *binding = view.getRootBinding(site.root);
    if (!binding || !binding->operation)
      return broken(site, BrokenRootRegionWorkReason::AssignmentProofMismatch,
                    "root-work query names an unknown semantic root");
    work.rootOperation = binding->operation;

    auto node = llvm::lower_bound(
        assignment.nodes, site.root,
        [](const NodeExecutionPartition &candidate,
           const SemanticRootKey &root) { return candidate.root < root; });
    if (node == assignment.nodes.end() || node->root != site.root)
      return broken(site, BrokenRootRegionWorkReason::AssignmentProofMismatch,
                    "root-work query has no closed node assignment");
    for (const ExecutionShard &shard : node->shards)
      if (shard.tile == site.tile)
        work.execution.push_back({shard.shard, shard.iterationDomain});

    for (const analysis::ReductionMergeRequirement &merge :
         proof.reductionMerges) {
      if (merge.group.root != site.root)
        continue;
      if (merge.mergeTile == site.tile)
        work.merges.push_back(merge);
      for (const analysis::ReductionContribution &contribution :
           merge.contributions)
        if (contribution.tile == site.tile)
          work.contributions.push_back({merge.group, merge.mergeTile,
                                        merge.initialization, merge.algebra,
                                        merge.coupledRule, contribution});
    }
    for (const analysis::FinalResultOwner &owner : proof.finalOwners)
      if (owner.root == site.root && owner.tile == site.tile)
        work.results.push_back(
            {owner.result, owner.shard, owner.reductionGroup, owner.domain});

    if (work.execution.empty() && work.contributions.empty() &&
        work.merges.empty())
      return NoRootRegionWork{};

    collectSelectedRecipes();
    if (failure)
      return std::move(*failure);
    collectSupportValues();
    if (failure)
      return std::move(*failure);
    collectInvariantInputs();
    if (failure)
      return std::move(*failure);
    finalizeSupportOrder();
    if (failure)
      return std::move(*failure);
    finalizeStableOrder();
    return std::move(work);
  }

private:
  void setFailure(RootRegionWorkOutcome outcome) {
    if (!failure)
      failure = std::move(outcome);
  }

  bool mergeInto(ExactIndexSet &target, const ExactIndexSet &source) {
    ExactSetMerge merged = mergeExactSets(target, source, limits, site);
    if (auto *value = std::get_if<ExactIndexSet>(&merged)) {
      target = std::move(*value);
      return true;
    }
    if (auto *broken = std::get_if<BrokenRootRegionWork>(&merged))
      setFailure(*broken);
    else
      setFailure(std::get<RootRegionWorkLimitReached>(merged));
    return false;
  }

  bool addBoundary(const analysis::SourceDemand &source, const RootUseId &use) {
    std::optional<BoundaryDescription> described =
        describeBoundary(source.source, dag);
    if (!described) {
      setFailure(broken(site, BrokenRootRegionWorkReason::MissingBoundary,
                        "root-work source boundary is malformed"));
      return false;
    }
    auto [position, inserted] = boundaries.try_emplace(described->id);
    RootBoundaryWork &boundary = position->second;
    if (inserted) {
      boundary.id = described->id;
      boundary.sourceValue = described->value;
      boundary.requiredDomain = source.requiredDomain;
    } else {
      if (boundary.sourceValue != described->value ||
          !boundary.requiredDomain ||
          !mergeInto(*boundary.requiredDomain, source.requiredDomain))
        return false;
    }
    if (!llvm::is_contained(boundary.consumerUses, use))
      boundary.consumerUses.push_back(use);
    auto existing = boundaryIdsByValue.find(described->value);
    if (existing != boundaryIdsByValue.end() &&
        existing->second != described->id) {
      setFailure(
          broken(site, BrokenRootRegionWorkReason::MissingBoundary,
                 "one source value has conflicting boundary identities"));
      return false;
    }
    boundaryIdsByValue[described->value] = described->id;
    return true;
  }

  bool addInvariantBoundary(const BoundaryDescription &described,
                            const RootUseId &use) {
    auto [position, inserted] = boundaries.try_emplace(described.id);
    RootBoundaryWork &boundary = position->second;
    if (inserted) {
      boundary.id = described.id;
      boundary.sourceValue = described.value;
    } else if (boundary.sourceValue != described.value) {
      setFailure(broken(site, BrokenRootRegionWorkReason::MissingBoundary,
                        "invariant boundary identity changed source value"));
      return false;
    }
    if (!llvm::is_contained(boundary.consumerUses, use))
      boundary.consumerUses.push_back(use);
    boundaryIdsByValue[described.value] = described.id;
    return true;
  }

  std::optional<SupportValueId> getSupportId(mlir::Operation *operation,
                                             uint32_t result) {
    if (!operation || result >= operation->getNumResults())
      return std::nullopt;
    mlir::Value value = operation->getResult(result);
    const SemanticValueBinding *binding = view.getValueBinding(value);
    if (!binding)
      return std::nullopt;
    return SupportValueId{binding->key, result};
  }

  void collectSelectedRecipes() {
    std::map<uint32_t, RootOperandWork> operands;
    std::set<LogicalShardId> executionShards;
    for (const RootExecutionWork &execution : work.execution)
      executionShards.insert(execution.shard);
    for (const analysis::DependencyDemand &dependency :
         proof.dependencyDemands) {
      if (dependency.consumer != site.root ||
          dependency.consumerOperation != work.rootOperation)
        continue;
      for (const analysis::DestinationDemand &destination :
           dependency.perDestination) {
        if (destination.destinationTile != site.tile)
          continue;
        if (executionShards.find(destination.destinationShard) ==
            executionShards.end()) {
          setFailure(
              broken(site, BrokenRootRegionWorkReason::AssignmentProofMismatch,
                     "root operand demand names no local execution shard"));
          return;
        }
        RootUseId use{dependency.consumerOperand, destination.destinationShard};
        auto [position, inserted] =
            operands.try_emplace(dependency.consumerOperand);
        RootOperandWork &operand = position->second;
        if (inserted) {
          operand.operand = dependency.consumerOperand;
          operand.kind = dependency.kind;
        } else if (operand.kind != dependency.kind) {
          setFailure(broken(site,
                            BrokenRootRegionWorkReason::InterfaceContradiction,
                            "root operand changes its demand role"));
          return;
        }
        operand.uses.push_back({use, destination.consumerExecutionDomain,
                                destination.operandDemand});
        recipes.push_back({&destination, use});
        if (!destination.operandDemand.isEmpty() &&
            destination.sources.empty()) {
          setFailure(broken(site, BrokenRootRegionWorkReason::MissingBoundary,
                            "nonempty root operand demand has no source "
                            "boundary"));
          return;
        }
        for (const analysis::SourceDemand &source : destination.sources)
          if (!addBoundary(source, use))
            return;
      }
    }
    for (auto &[operand, value] : operands) {
      (void)operand;
      work.operands.push_back(std::move(value));
    }
  }

  void collectSupportValues() {
    for (const SelectedRecipe &recipe : recipes) {
      for (const analysis::TensorTransform &step :
           recipe.destination->reconstruction.steps) {
        std::optional<SupportValueId> id =
            getSupportId(step.operation, step.result);
        if (!id) {
          setFailure(unsupported(
              site, UnsupportedRootRegionWorkReason::MissingSemanticIdentity,
              "support result has no stable semantic value identity"));
          return;
        }
        mlir::Value result = step.operation->getResult(step.result);
        auto existing = supportIdsByValue.find(result);
        if (existing != supportIdsByValue.end() && existing->second != *id) {
          setFailure(broken(site,
                            BrokenRootRegionWorkReason::InterfaceContradiction,
                            "support result has conflicting identities"));
          return;
        }
        supportIdsByValue[result] = *id;
        auto [position, inserted] = support.try_emplace(*id);
        PendingSupportValue &pending = position->second;
        if (inserted) {
          pending.work.id = *id;
          pending.work.operation = step.operation;
          pending.work.result = step.result;
          pending.work.requiredDomain = step.outputDemand;
          pending.hasRequiredDomain = true;
        } else {
          if (pending.work.operation != step.operation ||
              pending.work.result != step.result ||
              !mergeInto(pending.work.requiredDomain, step.outputDemand))
            return;
        }
        if (!llvm::is_contained(pending.work.consumerUses, recipe.use))
          pending.work.consumerUses.push_back(recipe.use);
      }
    }

    for (const SelectedRecipe &recipe : recipes) {
      for (const analysis::TensorTransform &step :
           recipe.destination->reconstruction.steps) {
        std::optional<SupportValueId> id =
            getSupportId(step.operation, step.result);
        auto pending = id ? support.find(*id) : support.end();
        if (pending == support.end()) {
          setFailure(broken(site,
                            BrokenRootRegionWorkReason::InterfaceContradiction,
                            "support result disappeared during work assembly"));
          return;
        }
        if (!mlir::isMemoryEffectFree(step.operation)) {
          setFailure(unsupported(
              site,
              UnsupportedRootRegionWorkReason::UnsupportedSupportSemantics,
              "root support operation is not memory-effect-free"));
          return;
        }
        llvm::SetVector<mlir::Value> captures;
        mlir::getUsedValuesDefinedAbove(step.operation->getRegions(), captures);
        for (mlir::Value capture : captures) {
          if (mlir::isa<mlir::ShapedType>(capture.getType())) {
            setFailure(unsupported(
                site, UnsupportedRootRegionWorkReason::UnsupportedCapture,
                "support operation captures a shaped value outside its "
                "typed inputs"));
            return;
          }
          std::optional<SemanticRootKey> captureKey = deriveRegionCaptureKey(
              step.operation, capture, pending->second.work.id.valuePath);
          std::optional<BoundaryDescription> boundary =
              describeInvariantBoundary(capture, view, std::move(captureKey));
          if (!boundary || !addInvariantBoundary(*boundary, recipe.use)) {
            if (!failure)
              setFailure(unsupported(
                  site,
                  UnsupportedRootRegionWorkReason::MissingSemanticIdentity,
                  "support capture has no stable boundary identity"));
            return;
          }
          if (!llvm::is_contained(pending->second.work.captures, boundary->id))
            pending->second.work.captures.push_back(boundary->id);
        }
        for (const analysis::TensorTransformInputDemand &input :
             step.operandDemands) {
          if (input.operand >= step.operation->getNumOperands()) {
            setFailure(
                broken(site, BrokenRootRegionWorkReason::InterfaceContradiction,
                       "support input names an out-of-range operand"));
            return;
          }
          mlir::Value source = step.operation->getOperand(input.operand);
          RootSupportInputWork inputWork;
          inputWork.operand = input.operand;
          inputWork.requiredDomain = input.demand;
          SupportInputKey key;
          key.operand = input.operand;
          auto supportSource = supportIdsByValue.find(source);
          auto boundarySource = boundaryIdsByValue.find(source);
          if (supportSource != supportIdsByValue.end()) {
            inputWork.kind = RootSupportInputKind::SupportValue;
            inputWork.supportValue = supportSource->second;
            key.kind = inputWork.kind;
            key.supportValue = inputWork.supportValue;
          } else if (boundarySource != boundaryIdsByValue.end()) {
            inputWork.kind = RootSupportInputKind::Boundary;
            inputWork.boundary = boundarySource->second;
            key.kind = inputWork.kind;
            key.boundary = inputWork.boundary;
          } else if (input.demand.isEmpty() &&
                     source.getDefiningOp<mlir::tensor::EmptyOp>()) {
            inputWork.kind = RootSupportInputKind::ExactEmpty;
            key.kind = inputWork.kind;
          } else {
            setFailure(broken(site, BrokenRootRegionWorkReason::MissingBoundary,
                              "support input has no exact boundary or prior "
                              "support value"));
            return;
          }
          auto [inputPosition, inputInserted] =
              pending->second.inputs.try_emplace(key, inputWork);
          if (!inputInserted &&
              !mergeInto(inputPosition->second.requiredDomain, input.demand))
            return;
        }
      }
    }
    for (auto &[id, pending] : support) {
      (void)id;
      if (!pending.hasRequiredDomain) {
        setFailure(broken(site,
                          BrokenRootRegionWorkReason::InterfaceContradiction,
                          "support value has no required domain"));
        return;
      }
      std::optional<uint32_t> previousOperand;
      for (auto &[key, input] : pending.inputs) {
        (void)key;
        if (previousOperand && *previousOperand == input.operand) {
          setFailure(broken(
              site, BrokenRootRegionWorkReason::InterfaceContradiction,
              "one support operand resolves to multiple source identities"));
          return;
        }
        previousOperand = input.operand;
        pending.work.inputs.push_back(std::move(input));
      }
    }
  }

  void collectInvariantInputs() {
    if (work.execution.empty())
      return;
    llvm::DenseSet<uint32_t> demandedOperands;
    for (const RootOperandWork &operand : work.operands)
      demandedOperands.insert(operand.operand);
    for (auto [operandIndex, operand] :
         llvm::enumerate(work.rootOperation->getOperands())) {
      if (demandedOperands.contains(operandIndex))
        continue;
      if (mlir::isa<mlir::ShapedType>(operand.getType()))
        continue;
      std::optional<BoundaryDescription> boundary =
          describeInvariantBoundary(operand, view);
      if (!boundary) {
        setFailure(unsupported(
            site, UnsupportedRootRegionWorkReason::MissingSemanticIdentity,
            "invariant root operand has no stable boundary identity"));
        return;
      }
      for (const RootExecutionWork &execution : work.execution)
        if (!addInvariantBoundary(
                *boundary,
                {static_cast<uint32_t>(operandIndex), execution.shard}))
          return;
      work.invariantInputs.push_back({RootInvariantUseKind::Operand,
                                      static_cast<uint32_t>(operandIndex),
                                      boundary->id});
    }

    llvm::SetVector<mlir::Value> captures;
    mlir::getUsedValuesDefinedAbove(work.rootOperation->getRegions(), captures);
    llvm::DenseSet<mlir::Value> directOperands(
        work.rootOperation->getOperands().begin(),
        work.rootOperation->getOperands().end());
    for (mlir::Value capture : captures) {
      if (directOperands.contains(capture))
        continue;
      if (mlir::isa<mlir::ShapedType>(capture.getType())) {
        setFailure(unsupported(
            site, UnsupportedRootRegionWorkReason::UnsupportedCapture,
            "root region captures a shaped value outside exact operand "
            "demand"));
        return;
      }
      std::optional<SemanticRootKey> captureKey =
          deriveRegionCaptureKey(work.rootOperation, capture, site.root);
      std::optional<BoundaryDescription> boundary =
          describeInvariantBoundary(capture, view, std::move(captureKey));
      if (!boundary) {
        setFailure(unsupported(
            site, UnsupportedRootRegionWorkReason::MissingSemanticIdentity,
            "root region capture has no stable boundary identity"));
        return;
      }
      for (const RootExecutionWork &execution : work.execution)
        if (!addInvariantBoundary(*boundary, {0, execution.shard}))
          return;
      work.invariantInputs.push_back(
          {RootInvariantUseKind::RegionCapture, std::nullopt, boundary->id});
    }
  }

  void finalizeSupportOrder() {
    std::map<SupportValueId, unsigned> indegree;
    std::map<SupportValueId, llvm::SmallVector<SupportValueId, 2>> successors;
    for (const auto &[id, pending] : support) {
      (void)pending;
      indegree.try_emplace(id, 0);
    }
    for (const auto &[destination, pending] : support)
      for (const RootSupportInputWork &input : pending.work.inputs) {
        if (input.kind != RootSupportInputKind::SupportValue ||
            !input.supportValue)
          continue;
        if (*input.supportValue == destination ||
            !support.count(*input.supportValue)) {
          setFailure(broken(site, BrokenRootRegionWorkReason::SupportGraphCycle,
                            "support graph has a self-edge or missing source"));
          return;
        }
        llvm::SmallVector<SupportValueId, 2> &edges =
            successors[*input.supportValue];
        if (!llvm::is_contained(edges, destination)) {
          edges.push_back(destination);
          ++indegree[destination];
        }
      }
    std::set<SupportValueId> ready;
    for (const auto &[id, degree] : indegree)
      if (degree == 0)
        ready.insert(id);
    while (!ready.empty()) {
      SupportValueId id = *ready.begin();
      ready.erase(ready.begin());
      work.supportValues.push_back(std::move(support[id].work));
      for (const SupportValueId &successor : successors[id])
        if (--indegree[successor] == 0)
          ready.insert(successor);
    }
    if (work.supportValues.size() != support.size())
      setFailure(broken(site, BrokenRootRegionWorkReason::SupportGraphCycle,
                        "root support graph contains a cycle"));
  }

  void finalizeStableOrder() {
    for (auto &[id, boundary] : boundaries) {
      (void)id;
      llvm::sort(boundary.consumerUses);
      work.boundaries.push_back(std::move(boundary));
    }
    for (RootOperandWork &operand : work.operands)
      llvm::sort(operand.uses,
                 [](const RootOperandUseWork &lhs,
                    const RootOperandUseWork &rhs) { return lhs.id < rhs.id; });
    for (RootSupportValueWork &supportValue : work.supportValues)
      llvm::sort(supportValue.captures);
    for (RootSupportValueWork &supportValue : work.supportValues)
      llvm::sort(supportValue.consumerUses);
    llvm::sort(work.invariantInputs, [](const RootInvariantInputWork &lhs,
                                        const RootInvariantInputWork &rhs) {
      return std::tie(lhs.kind, lhs.boundary, lhs.operand) <
             std::tie(rhs.kind, rhs.boundary, rhs.operand);
    });
    llvm::sort(work.contributions, [](const RootContributionWork &lhs,
                                      const RootContributionWork &rhs) {
      if (lhs.group != rhs.group)
        return lhs.group < rhs.group;
      return lhs.contribution.shard < rhs.contribution.shard;
    });
    llvm::sort(
        work.results, [](const RootResultWork &lhs, const RootResultWork &rhs) {
          return std::tie(lhs.result, lhs.reductionGroup, lhs.ownerShard) <
                 std::tie(rhs.result, rhs.reductionGroup, rhs.ownerShard);
        });
  }

  const StructuredDAGAnalysis &dag;
  const SpatialAssignment &assignment;
  const analysis::ExactDemandProof &proof;
  const StructuredDemandView &view;
  RootRegionWorkId site;
  analysis::IndexRelationLimits limits;
  RootRegionWork work;
  std::optional<RootRegionWorkOutcome> failure;
  std::vector<SelectedRecipe> recipes;
  std::map<RootBoundaryId, RootBoundaryWork> boundaries;
  llvm::DenseMap<mlir::Value, RootBoundaryId> boundaryIdsByValue;
  std::map<SupportValueId, PendingSupportValue> support;
  llvm::DenseMap<mlir::Value, SupportValueId> supportIdsByValue;
};

} // namespace

mlir::FailureOr<RootRegionWorkAnalysis> RootRegionWorkAnalysis::create(
    const StructuredDAGAnalysis &dag, const SpatialAssignment &assignment,
    const analysis::ExactDemandProof &proof, std::string *failureReason) {
  mlir::FailureOr<StructuredDemandView> view =
      StructuredDemandView::create(dag, assignment, proof, failureReason);
  if (mlir::failed(view))
    return mlir::failure();
  return RootRegionWorkAnalysis(dag, assignment, proof, std::move(*view));
}

analysis::RootRegionWorkOutcome RootRegionWorkAnalysis::query(
    const SemanticRootKey &root, TileId tile,
    const analysis::IndexRelationLimits &limits) const {
  return WorkBuilder(dag, assignment, proof, view, {root, tile}, limits)
      .build();
}

} // namespace wafer::compiler::detail
