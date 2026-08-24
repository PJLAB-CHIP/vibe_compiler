//===- BaselineAttentionMaterialization.cpp --------------------------===//

#include "Wafer/Planning/Baseline/BaselineAttentionMaterialization.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/CardDataflowConstruction.h"
#include "Wafer/Planning/PhysicalDataflow/SemanticRootAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPlan.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandView.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"
#include "Wafer/Support/CompileTiming.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <type_traits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
  return mlir::failure();
}

analysis::RootRegionWorkId workOf(const ExecutionInstanceId &execution) {
  return std::visit([](const auto &source) { return source.work; },
                    execution.source);
}

mlir::FailureOr<mlir::func::FuncOp> getProgram(mlir::ModuleOp module,
                                               std::string *failureReason) {
  mlir::func::FuncOp program;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (program)
      return fail<mlir::func::FuncOp>(
          failureReason,
          "selected attention materialization requires one program function");
    program = function;
  }
  if (!program || !program.getBody().hasOneBlock())
    return fail<mlir::func::FuncOp>(
        failureReason,
        "selected attention materialization requires one function block");
  return program;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticIterationExtents(mlir::Operation *operation,
                          std::string *failureReason) {
  auto tiling = mlir::dyn_cast_or_null<mlir::TilingInterface>(operation);
  if (!tiling)
    return fail<llvm::SmallVector<int64_t, 4>>(
        failureReason, "selected attention operation has no TilingInterface");
  mlir::OpBuilder builder(operation);
  llvm::SmallVector<mlir::Range> domain = tiling.getIterationDomain(builder);
  llvm::SmallVector<int64_t, 4> extents;
  extents.reserve(domain.size());
  for (const mlir::Range &range : domain) {
    std::optional<int64_t> offset = mlir::getConstantIntValue(range.offset);
    std::optional<int64_t> size = mlir::getConstantIntValue(range.size);
    std::optional<int64_t> stride = mlir::getConstantIntValue(range.stride);
    if (!offset || !size || !stride || *offset != 0 || *size <= 0 ||
        *stride != 1)
      return fail<llvm::SmallVector<int64_t, 4>>(
          failureReason,
          "selected attention operation has a dynamic iteration domain");
    extents.push_back(*size);
  }
  if (extents.size() != tiling.getLoopIteratorTypes().size())
    return fail<llvm::SmallVector<int64_t, 4>>(
        failureReason,
        "selected attention operation iteration rank is inconsistent");
  return extents;
}

std::string demandDetail(const analysis::ExactDemandOutcome &outcome) {
  return std::visit(
      [](const auto &value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
          return {};
        else
          return value.detail;
      },
      outcome);
}

struct PreservedOperationPlan {
  const NodeExecutionPartition *spatial = nullptr;
  llvm::SmallVector<int64_t, 4> temporalTileSizes;
  SemanticRootKey root;
};

mlir::FailureOr<NodeSpatialPlan> rebindNodeSpatialPlan(
    const NodeExecutionPartition &partition, const SemanticRootKey &root,
    llvm::ArrayRef<int64_t> iterationExtents, std::string *failureReason) {
  if (partition.shards.empty())
    return fail<NodeSpatialPlan>(failureReason,
                                 "preserved spatial partition has no shards");
  llvm::SmallVector<uint32_t, 4> factors(iterationExtents.size(), 0);
  llvm::SmallVector<const ExecutionShard *, 16> shards;
  shards.reserve(partition.shards.size());
  for (const ExecutionShard &shard : partition.shards) {
    if (shard.shard.coordinate.size() != iterationExtents.size())
      return fail<NodeSpatialPlan>(failureReason,
                                   "preserved spatial partition rank changed");
    for (auto [axis, coordinate] : llvm::enumerate(shard.shard.coordinate))
      factors[axis] = std::max<uint32_t>(factors[axis], coordinate + 1);
    shards.push_back(&shard);
  }
  if (llvm::is_contained(factors, 0u))
    return fail<NodeSpatialPlan>(
        failureReason, "preserved spatial partition has an empty axis");
  llvm::sort(shards, [](const ExecutionShard *lhs, const ExecutionShard *rhs) {
    return std::lexicographical_compare(
        lhs->shard.coordinate.begin(), lhs->shard.coordinate.end(),
        rhs->shard.coordinate.begin(), rhs->shard.coordinate.end());
  });

  NodeSpatialPlan result;
  result.root = root;
  for (auto [axis, factor] : llvm::enumerate(factors))
    result.axes.push_back({static_cast<uint32_t>(axis),
                           IteratorPartitionScheme::BalancedParts, factor});
  for (const ExecutionShard *shard : shards)
    result.embedding.push_back(shard->tile);
  for (const ReductionGroupPlacement &placement : partition.reductionGroups) {
    ReductionGroupId group = placement.group;
    group.root = root;
    result.reductionMerges.push_back({std::move(group), placement.mergeTile});
  }
  llvm::sort(result.reductionMerges);
  return result;
}

mlir::FailureOr<llvm::SmallVector<StructuredDAGNodePlacement, 64>>
buildSelectedNodePlacements(const StructuredDAGAnalysis &dag,
                            const SemanticRootAnalysis &roots,
                            const SpatialAssignment &spatial,
                            std::string *failureReason) {
  llvm::SmallVector<StructuredDAGNodePlacement, 64> placements;
  for (const StructuredDAGNode &node : dag.getNodes()) {
    const SemanticRootBinding *root = roots.find(node.operation);
    if (!root)
      return fail<llvm::SmallVector<StructuredDAGNodePlacement, 64>>(
          failureReason, "selected structured op has no semantic root");
    auto partition = llvm::find_if(
        spatial.nodes, [&](const NodeExecutionPartition &candidate) {
          return candidate.root == root->key;
        });
    if (partition == spatial.nodes.end() || partition->shards.empty())
      return fail<llvm::SmallVector<StructuredDAGNodePlacement, 64>>(
          failureReason, "selected structured op has no spatial partition");
    StructuredDAGNodePlacement placement;
    placement.node = node.id;
    placement.iteratorPartitionFactors.assign(
        partition->shards.front().shard.coordinate.size(), 0);
    for (const ExecutionShard &shard : partition->shards) {
      placement.tiles.push_back(shard.tile);
      for (auto [axis, coordinate] : llvm::enumerate(shard.shard.coordinate))
        placement.iteratorPartitionFactors[axis] = std::max<uint32_t>(
            placement.iteratorPartitionFactors[axis], coordinate + 1);
    }
    placements.push_back(std::move(placement));
  }
  return placements;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>> getSelectedAttentionTemporalTile(
    mlir::Operation *operation, const AttentionActionId &action,
    const AttentionIterationRoles &roles,
    llvm::ArrayRef<int64_t> sourceTemporalTile,
    llvm::ArrayRef<int64_t> operationExtents, std::string *failureReason) {
  llvm::SmallVector<int64_t, 4> result(operationExtents.begin(),
                                       operationExtents.end());
  if (std::holds_alternative<RequiredMergeExecution>(
          action.scope.execution.source))
    return result;
  auto axisProduct =
      [&](llvm::ArrayRef<unsigned> axes) -> std::optional<int64_t> {
    int64_t product = 1;
    for (unsigned axis : axes) {
      int64_t next = 0;
      if (axis >= sourceTemporalTile.size() ||
          llvm::MulOverflow(product, sourceTemporalTile[axis], next))
        return std::nullopt;
      product = next;
    }
    return product;
  };
  auto setDirect = [&](llvm::ArrayRef<unsigned> axes) {
    if (axes.size() != result.size())
      return false;
    for (auto [index, axis] : llvm::enumerate(axes)) {
      if (axis >= sourceTemporalTile.size() || sourceTemporalTile[axis] <= 0)
        return false;
      result[index] = std::min(result[index], sourceTemporalTile[axis]);
    }
    return true;
  };
  auto setGrouped = [&](llvm::ArrayRef<llvm::ArrayRef<unsigned>> groups) {
    if (groups.size() != result.size())
      return false;
    for (auto [index, axes] : llvm::enumerate(groups)) {
      std::optional<int64_t> product = axisProduct(axes);
      if (!product || *product <= 0)
        return false;
      result[index] = std::min(result[index], *product);
    }
    return true;
  };

  llvm::SmallVector<unsigned, 8> rowAxes;
  llvm::append_range(rowAxes, roles.batch);
  llvm::append_range(rowAxes, roles.query);
  llvm::SmallVector<unsigned, 8> scoreAxes(rowAxes.begin(), rowAxes.end());
  llvm::append_range(scoreAxes, roles.keyValueReduction);

  bool valid = false;
  switch (action.kind) {
  case AttentionActionKind::ScaleMask:
  case AttentionActionKind::Exponential:
    valid = setDirect(scoreAxes);
    break;
  case AttentionActionKind::RowMaximum:
  case AttentionActionKind::RowSum:
    valid = mlir::isa<mlir::linalg::FillOp>(operation) ? setDirect(rowAxes)
                                                       : setDirect(scoreAxes);
    break;
  case AttentionActionKind::QueryKeyContraction: {
    llvm::SmallVector<llvm::ArrayRef<unsigned>, 4> groups;
    if (!roles.batch.empty())
      groups.push_back(roles.batch);
    groups.push_back(roles.query);
    groups.push_back(roles.keyValueReduction);
    if (!mlir::isa<mlir::linalg::FillOp>(operation))
      groups.push_back(roles.queryKeyReduction);
    valid = setGrouped(groups);
    break;
  }
  case AttentionActionKind::ValueContraction: {
    llvm::SmallVector<llvm::ArrayRef<unsigned>, 4> groups;
    if (!roles.batch.empty())
      groups.push_back(roles.batch);
    groups.push_back(roles.query);
    groups.push_back(roles.valueOutput);
    if (!mlir::isa<mlir::linalg::FillOp>(operation))
      groups.push_back(roles.keyValueReduction);
    valid = setGrouped(groups);
    break;
  }
  case AttentionActionKind::StateUpdate:
  case AttentionActionKind::StateMerge:
  case AttentionActionKind::Finalize:
    valid = true;
    break;
  }
  if (!valid)
    return fail<llvm::SmallVector<int64_t, 4>>(
        failureReason,
        "selected attention action cannot project its temporal tile");
  return result;
}

mlir::FailureOr<analysis::ExactIndexSet>
makeBoxUnion(unsigned rank,
             llvm::ArrayRef<analysis::StaticRectangularIndexSet> boxes) {
  std::optional<mlir::presburger::PresburgerSet> set;
  for (const analysis::StaticRectangularIndexSet &box : boxes) {
    analysis::IndexSetResult piece =
        analysis::IndexRelation::staticRectangularDomain(box.offsets,
                                                         box.sizes);
    if (!piece.isExact())
      return mlir::failure();
    if (!set)
      set = std::move(*piece.set);
    else
      set->unionInPlace(*piece.set);
  }
  if (!set)
    set = mlir::presburger::PresburgerSet::getEmpty(
        mlir::presburger::PresburgerSpace::getSetSpace(rank));
  return analysis::ExactIndexSet(std::move(*set),
                                 analysis::ExactIndexSetForm::BoxUnion, boxes);
}

bool dependsOnValue(mlir::Value value, mlir::Value target,
                    llvm::DenseSet<mlir::Value> &visited) {
  if (value == target)
    return true;
  if (!value || !visited.insert(value).second)
    return false;
  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return false;
  mlir::Operation *operation = result.getOwner();
  if (!operation || mlir::isa<mlir::DestinationStyleOpInterface>(operation))
    return false;
  auto indexing =
      mlir::dyn_cast<wafer::WaferTensorIndexingOpInterface>(operation);
  if (!indexing)
    return false;
  mlir::FailureOr<wafer::TensorIndexingDescription> description =
      indexing.getTensorIndexingDescription(result.getResultNumber());
  if (mlir::failed(description))
    return false;
  for (const wafer::TensorIndexingOperandDescription &operand :
       description->operands)
    if (operand.operand < operation->getNumOperands() &&
        dependsOnValue(operation->getOperand(operand.operand), target, visited))
      return true;
  return false;
}

mlir::FailureOr<analysis::IndexRelation> buildTensorIndexRelation(
    mlir::Operation *operation, unsigned resultNumber,
    const wafer::TensorIndexingDescription &description,
    const wafer::TensorIndexingOperandDescription &operand) {
  if (!operation || resultNumber >= operation->getNumResults() ||
      operand.operand >= operation->getNumOperands())
    return mlir::failure();
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getResult(resultNumber).getType());
  auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(
      operation->getOperand(operand.operand).getType());
  if (!resultType || !operandType || !resultType.hasStaticShape() ||
      !operandType.hasStaticShape())
    return mlir::failure();

  analysis::IndexRelationResult relation;
  switch (description.kind) {
  case wafer::TensorIndexingTransformKind::ExpandShape:
  case wafer::TensorIndexingTransformKind::CollapseShape:
  case wafer::TensorIndexingTransformKind::Cast:
    relation = analysis::IndexRelation::staticReshape(resultType.getShape(),
                                                      operandType.getShape());
    break;
  case wafer::TensorIndexingTransformKind::ExtractSlice:
    relation = analysis::IndexRelation::staticSlice(
        resultType.getShape(), operandType.getShape(), operand.offsets,
        operand.strides);
    break;
  case wafer::TensorIndexingTransformKind::InsertSlice:
    relation = operand.role == wafer::TensorIndexingOperandRole::Source
                   ? analysis::IndexRelation::staticInsertSlice(
                         resultType.getShape(), operandType.getShape(),
                         operand.offsets)
                   : analysis::IndexRelation::identity(operandType.getShape());
    break;
  case wafer::TensorIndexingTransformKind::Pad:
    relation = analysis::IndexRelation::staticInsertSlice(
        resultType.getShape(), operandType.getShape(), operand.offsets);
    break;
  }
  if (!relation.isExact())
    return mlir::failure();
  return std::move(*relation.relation);
}

mlir::FailureOr<analysis::ExactIndexSet>
mapRootDomainThroughOutputPath(const analysis::ExactIndexSet &rootDomain,
                               mlir::Value rootValue, mlir::Value outputValue,
                               std::string *detail = nullptr) {
  auto reject =
      [&](llvm::StringRef message) -> mlir::FailureOr<analysis::ExactIndexSet> {
    if (detail)
      *detail = message.str();
    return mlir::failure();
  };
  struct Step {
    analysis::IndexRelation relation;
  };
  llvm::SmallVector<Step, 4> outputToRoot;
  mlir::Value current = outputValue;
  while (current != rootValue) {
    auto result = mlir::dyn_cast<mlir::OpResult>(current);
    if (!result)
      return reject("output path reaches a block argument");
    mlir::Operation *operation = result.getOwner();
    auto indexing =
        mlir::dyn_cast_or_null<wafer::WaferTensorIndexingOpInterface>(
            operation);
    if (!indexing)
      return reject((llvm::Twine("output support lacks tensor indexing: ") +
                     operation->getName().getStringRef())
                        .str());
    mlir::FailureOr<wafer::TensorIndexingDescription> description =
        indexing.getTensorIndexingDescription(result.getResultNumber());
    if (mlir::failed(description))
      return reject("output support has no indexing description");
    const wafer::TensorIndexingOperandDescription *selected = nullptr;
    for (const wafer::TensorIndexingOperandDescription &operand :
         description->operands) {
      if (operand.operand >= operation->getNumOperands())
        return reject("output support indexing operand is outside arity");
      llvm::DenseSet<mlir::Value> visited;
      if (!dependsOnValue(operation->getOperand(operand.operand), rootValue,
                          visited))
        continue;
      if (selected)
        return reject("output support has multiple paths to one root");
      selected = &operand;
    }
    if (!selected)
      return reject((llvm::Twine("output support does not depend on root: ") +
                     operation->getName().getStringRef())
                        .str());
    mlir::FailureOr<analysis::IndexRelation> relation =
        buildTensorIndexRelation(operation, result.getResultNumber(),
                                 *description, *selected);
    if (mlir::failed(relation))
      return reject("output support relation is not exact");
    outputToRoot.push_back({std::move(*relation)});
    current = operation->getOperand(selected->operand);
  }

  analysis::ExactIndexSet mapped = rootDomain;
  for (const Step &step : llvm::reverse(outputToRoot)) {
    analysis::IndexSetResult preimage =
        step.relation.preimage(mapped.getPresburgerSet());
    if (!preimage.isExact())
      return reject("output support preimage is not exact");
    mapped =
        analysis::ExactIndexSet(std::move(*preimage.set),
                                analysis::ExactIndexSetForm::GeneralPresburger);
  }
  return analysis::normalizeFiniteExactIndexSet(mapped);
}

mlir::FailureOr<llvm::SmallVector<OutputTileMapping, 4>> buildOutputMappings(
    const StructuredDAGAnalysis &dag, const SpatialAssignment &spatial,
    const analysis::ExactDemandProof &demand,
    const std::map<mlir::Operation *, llvm::SmallVector<int64_t, 4>>
        &temporalByOperation,
    const std::map<mlir::Operation *, AttentionActionId> &attentionActions,
    const CanonicalAttentionWorkCoordinate &attentionWork,
    std::string *failureReason) {
  mlir::FailureOr<StructuredDemandView> view =
      StructuredDemandView::create(dag, spatial, demand, failureReason);
  if (mlir::failed(view))
    return mlir::failure();
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(
      dag.getFunction().getBody().front().getTerminator());
  if (!returnOp)
    return fail<llvm::SmallVector<OutputTileMapping, 4>>(
        failureReason, "selected function has no return operation");
  auto findFinalValue = [&](const AttentionActionId &action)
      -> const AttentionValueDescription * {
    for (const AttentionWorkDescription &root : attentionWork.roots) {
      auto value = llvm::find_if(
          root.values, [&](const AttentionValueDescription &candidate) {
            return candidate.id.scope == action.scope &&
                   candidate.id.kind == AttentionValueKind::FinalOutput;
          });
      if (value != root.values.end())
        return &*value;
    }
    return nullptr;
  };
  llvm::SmallVector<OutputTileMapping, 4> outputs;
  for (auto [outputIndex, roots] :
       llvm::enumerate(dag.getObservableOutputRootNodes())) {
    if (roots.empty())
      return fail<llvm::SmallVector<OutputTileMapping, 4>>(
          failureReason,
          (llvm::Twine("selected output ") + llvm::Twine(outputIndex) +
           " has no structured semantic root")
              .str());
    llvm::SmallVector<const StructuredDAGNode *, 16> rootNodes;
    std::optional<SemanticRootKey> attentionRoot;
    std::set<AttentionWorkScopeId> attentionFinalScopes;
    for (StructuredDAGNodeID rootId : roots) {
      const StructuredDAGNode *root = dag.getNode(rootId);
      if (!root)
        return fail<llvm::SmallVector<OutputTileMapping, 4>>(
            failureReason, "selected output root is absent from the DAG");
      if (roots.size() > 1) {
        auto action = attentionActions.find(root->operation);
        if (action == attentionActions.end() ||
            action->second.kind != AttentionActionKind::Finalize)
          return fail<llvm::SmallVector<OutputTileMapping, 4>>(
              failureReason,
              "multi-root selected output is not an attention finalization");
        SemanticRootKey current = workOf(action->second.scope.execution).root;
        if (attentionRoot && !(*attentionRoot == current))
          return fail<llvm::SmallVector<OutputTileMapping, 4>>(
              failureReason,
              "selected output mixes distinct attention semantic roots");
        attentionRoot = current;
        if (!attentionFinalScopes.insert(action->second.scope).second ||
            !findFinalValue(action->second))
          return fail<llvm::SmallVector<OutputTileMapping, 4>>(
              failureReason,
              "selected output duplicates or lacks one attention final scope");
      }
      rootNodes.push_back(root);
    }
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
        dag.getFunction().getResultTypes()[outputIndex]);
    if (!outputType || !outputType.hasStaticShape())
      return fail<llvm::SmallVector<OutputTileMapping, 4>>(
          failureReason, "selected output shape is not static");

    OutputTileMapping output;
    output.outputIndex = outputIndex;
    std::map<int64_t, llvm::SmallVector<analysis::StaticRectangularIndexSet, 4>>
        boxesByTile;
    llvm::SmallVector<std::pair<TileId, analysis::StaticRectangularIndexSet>,
                      16>
        mappedOwnerBoxes;
    for (const StructuredDAGNode *root : rootNodes) {
      auto rootType = mlir::dyn_cast<mlir::RankedTensorType>(
          root->operation->getResult(0).getType());
      if (!rootType || !rootType.hasStaticShape())
        return fail<llvm::SmallVector<OutputTileMapping, 4>>(
            failureReason, "selected output root shape is not static");
      llvm::SmallVector<const analysis::FinalResultOwner *, 4> owners =
          view->getFinalOwners(root->id, 0);
      if (owners.empty())
        return fail<llvm::SmallVector<OutputTileMapping, 4>>(
            failureReason, "selected output root has no exact final owner");
      auto finalAction = attentionActions.find(root->operation);
      const AttentionValueDescription *finalValue =
          roots.size() > 1 && finalAction != attentionActions.end()
              ? findFinalValue(finalAction->second)
              : nullptr;
      if (finalValue) {
        if (owners.size() != 1 ||
            finalValue->exactDomain.getForm() !=
                analysis::ExactIndexSetForm::BoxUnion ||
            finalValue->exactDomain.getBoxes().empty())
          return fail<llvm::SmallVector<OutputTileMapping, 4>>(
              failureReason,
              "attention final scope has no single owner/finite domain");
        for (const analysis::StaticRectangularIndexSet &box :
             finalValue->exactDomain.getBoxes())
          mappedOwnerBoxes.push_back({owners.front()->tile, box});
        continue;
      }
      for (const analysis::FinalResultOwner *owner : owners) {
        std::string pathFailure;
        mlir::FailureOr<analysis::ExactIndexSet> mapped =
            mapRootDomainThroughOutputPath(
                owner->domain, root->operation->getResult(0),
                returnOp.getOperand(outputIndex), &pathFailure);
        if (mlir::failed(mapped))
          return fail<llvm::SmallVector<OutputTileMapping, 4>>(
              failureReason,
              (llvm::Twine(
                   "selected output owner cannot cross output support; ") +
               "root=" + llvm::Twine(root->id) +
               ", operation=" + root->operation->getName().getStringRef() +
               ", detail=" + pathFailure)
                  .str());
        for (const analysis::StaticRectangularIndexSet &box :
             mapped->getBoxes())
          mappedOwnerBoxes.push_back({owner->tile, box});
      }
    }
    if (mappedOwnerBoxes.empty())
      return fail<llvm::SmallVector<OutputTileMapping, 4>>(
          failureReason, "selected output support maps no owner domain");
    for (size_t dimension = 0; dimension < outputType.getRank(); ++dimension) {
      llvm::SmallVector<std::pair<int64_t, int64_t>, 16> intervals;
      for (const auto &[tile, box] : mappedOwnerBoxes) {
        (void)tile;
        if (box.offsets.size() != static_cast<size_t>(outputType.getRank()) ||
            box.sizes.size() != static_cast<size_t>(outputType.getRank()))
          return mlir::failure();
        intervals.push_back({box.offsets[dimension],
                             box.offsets[dimension] + box.sizes[dimension]});
      }
      llvm::sort(intervals);
      int64_t coveredEnd = 0;
      for (const auto &[begin, end] : intervals) {
        if (begin > coveredEnd)
          break;
        coveredEnd = std::max(coveredEnd, end);
      }
      if (coveredEnd >= outputType.getDimSize(dimension))
        continue;
      const int64_t commonOffset = intervals.front().first;
      const int64_t commonEnd = intervals.front().second;
      if (llvm::any_of(intervals, [&](const auto &interval) {
            return interval.first != commonOffset ||
                   interval.second != commonEnd;
          }))
        return fail<llvm::SmallVector<OutputTileMapping, 4>>(
            failureReason,
            "selected output support leaves an ambiguously partitioned axis");
      for (auto &[tile, box] : mappedOwnerBoxes) {
        (void)tile;
        box.offsets[dimension] = 0;
        box.sizes[dimension] = outputType.getDimSize(dimension);
      }
    }
    for (const auto &[tile, box] : mappedOwnerBoxes)
      boxesByTile[tile.getValue()].push_back(box);
    for (auto &[tile, boxes] : boxesByTile) {
      mlir::FailureOr<analysis::ExactIndexSet> tileDomain =
          makeBoxUnion(outputType.getRank(), boxes);
      if (mlir::failed(tileDomain))
        return mlir::failure();
      mlir::FailureOr<analysis::ExactIndexSet> normalized =
          analysis::normalizeFiniteExactIndexSet(*tileDomain);
      if (mlir::failed(normalized) || normalized->getBoxes().size() != 1)
        return fail<llvm::SmallVector<OutputTileMapping, 4>>(
            failureReason,
            "selected output Tile domain is not one exact rectangle");
      const analysis::StaticRectangularIndexSet &box =
          normalized->getBoxes().front();
      output.shards.push_back({TileId(tile), box.offsets, box.sizes});
    }
    analysis::IndexSetResult fullOutput =
        analysis::IndexRelation::staticDomain(outputType.getShape());
    std::optional<mlir::presburger::PresburgerSet> covered;
    if (!fullOutput.isExact())
      return mlir::failure();
    for (const OutputTileShard &shard : output.shards) {
      analysis::IndexSetResult box =
          analysis::IndexRelation::staticRectangularDomain(shard.offsets,
                                                           shard.sizes);
      if (!box.isExact() ||
          (covered && !covered->intersect(*box.set).isIntegerEmpty()))
        return fail<llvm::SmallVector<OutputTileMapping, 4>>(
            failureReason, "selected output Tile domains overlap");
      covered = covered ? covered->unionSet(*box.set) : std::move(*box.set);
    }
    if (!covered || !covered->isEqual(*fullOutput.set))
      return fail<llvm::SmallVector<OutputTileMapping, 4>>(
          failureReason, "selected output Tile domains do not cover output");

    output.temporalTileSizes.assign(outputType.getShape().begin(),
                                    outputType.getShape().end());
    llvm::SmallVector<int64_t, 4> combinedTemporal(outputType.getRank(), 0);
    bool hasTemporal = false;
    for (const StructuredDAGNode *root : rootNodes) {
      auto temporal = temporalByOperation.find(root->operation);
      if (temporal == temporalByOperation.end())
        continue;
      std::optional<llvm::SmallVector<int64_t, 4>> resultTile =
          getStructuredResultTileShape(root->operation, 0, temporal->second);
      if (!resultTile)
        return fail<llvm::SmallVector<OutputTileMapping, 4>>(
            failureReason, "selected output root has no temporal result tile");
      auto rootType = mlir::cast<mlir::RankedTensorType>(
          root->operation->getResult(0).getType());
      llvm::SmallVector<int64_t, 4> mappedSizes;
      if (rootType.getRank() == outputType.getRank()) {
        mappedSizes.assign(resultTile->begin(), resultTile->end());
        for (auto [size, extent] :
             llvm::zip_equal(mappedSizes, outputType.getShape()))
          size = std::min(size, extent);
      } else {
        llvm::SmallVector<int64_t, 4> zeroOffsets(resultTile->size(), 0);
        analysis::IndexSetResult rootTileSet =
            analysis::IndexRelation::staticRectangularDomain(zeroOffsets,
                                                             *resultTile);
        if (!rootTileSet.isExact())
          return mlir::failure();
        analysis::ExactIndexSet rootTile(
            std::move(*rootTileSet.set), analysis::ExactIndexSetForm::BoxUnion,
            {analysis::StaticRectangularIndexSet{zeroOffsets, *resultTile}});
        mlir::FailureOr<analysis::ExactIndexSet> mappedTile =
            mapRootDomainThroughOutputPath(rootTile,
                                           root->operation->getResult(0),
                                           returnOp.getOperand(outputIndex));
        if (mlir::failed(mappedTile) || mappedTile->getBoxes().size() != 1)
          return fail<llvm::SmallVector<OutputTileMapping, 4>>(
              failureReason,
              "selected output temporal tile cannot cross output support");
        mappedSizes = mappedTile->getBoxes().front().sizes;
      }
      if (mappedSizes.size() != combinedTemporal.size())
        return mlir::failure();
      for (auto [combined, size] :
           llvm::zip_equal(combinedTemporal, mappedSizes))
        combined = std::max(combined, size);
      hasTemporal = true;
    }
    if (hasTemporal)
      output.temporalTileSizes = std::move(combinedTemporal);
    llvm::sort(output.shards, [](const OutputTileShard &lhs,
                                 const OutputTileShard &rhs) {
      if (lhs.tile != rhs.tile)
        return lhs.tile.getValue() < rhs.tile.getValue();
      if (lhs.offsets != rhs.offsets)
        return std::lexicographical_compare(
            lhs.offsets.begin(), lhs.offsets.end(), rhs.offsets.begin(),
            rhs.offsets.end());
      return std::lexicographical_compare(lhs.sizes.begin(), lhs.sizes.end(),
                                          rhs.sizes.begin(), rhs.sizes.end());
    });
    outputs.push_back(std::move(output));
  }
  return outputs;
}

} // namespace

mlir::FailureOr<AttentionMaterializationSource>
prepareAttentionMaterializationSource(
    mlir::ModuleOp source, CardId cardId,
    const CardProgramAnalysis &sourceProgram, const CompleteCandidatePlan &plan,
    llvm::ArrayRef<TileId> availableTiles,
    CandidateMaterializationStatistics *statistics,
    std::string *failureReason) {
  wafer::support::ScopedCompileTimingSpan totalTiming(
      "conversion", "complete-candidate",
      "selected-attention-to-materialization-source");
  (void)cardId;
  if (!source || availableTiles.empty() ||
      plan.preparedAttention.work.roots.empty())
    return fail<AttentionMaterializationSource>(
        failureReason,
        "selected attention materialization input is incomplete");

  mlir::IRMapping cloneMapping;
  mlir::OwningOpRef<mlir::ModuleOp> selected =
      mlir::cast<mlir::ModuleOp>(source->clone(cloneMapping));
  mlir::FailureOr<mlir::func::FuncOp> program =
      getProgram(*selected, failureReason);
  if (mlir::failed(program))
    return mlir::failure();

  mlir::FailureOr<SemanticRootAnalysis> sourceRoots =
      SemanticRootAnalysis::create(sourceProgram.dag, failureReason);
  if (mlir::failed(sourceRoots))
    return mlir::failure();
  std::map<SemanticRootKey, const NodeExecutionPartition *> sourcePartitions;
  for (const NodeExecutionPartition &partition : plan.spatial.nodes)
    if (!sourcePartitions.try_emplace(partition.root, &partition).second)
      return fail<AttentionMaterializationSource>(
          failureReason, "source spatial partition identity is duplicated");
  std::map<SemanticRootKey, llvm::SmallVector<int64_t, 4>> sourceTemporalTiles;
  std::map<ExecutionInstanceId, llvm::SmallVector<int64_t, 4>>
      sourceTemporalByExecution;
  for (const TemporalScopePlan &scope : plan.temporal.scopes) {
    const ExecutionInstanceId *execution = getRequiredExecution(scope.id);
    if (!execution || !isTopLevelScope(scope.id))
      return fail<AttentionMaterializationSource>(
          failureReason,
          "selected attention requires top-level temporal scopes");
    if (!sourceTemporalByExecution
             .try_emplace(*execution, scope.iteratorTileSizes)
             .second)
      return fail<AttentionMaterializationSource>(
          failureReason, "source temporal execution is duplicated");
    SemanticRootKey root = workOf(*execution).root;
    auto [position, inserted] =
        sourceTemporalTiles.try_emplace(root, scope.iteratorTileSizes);
    if (!inserted) {
      if (position->second.size() != scope.iteratorTileSizes.size())
        return fail<AttentionMaterializationSource>(
            failureReason, "source temporal tile rank is inconsistent");
      for (auto [current, candidate] :
           llvm::zip_equal(position->second, scope.iteratorTileSizes))
        current = std::max(current, candidate);
    }
  }
  std::map<mlir::Operation *, PreservedOperationPlan> preservedOperations;
  for (const StructuredDAGNode &node : sourceProgram.dag.getNodes()) {
    const SemanticRootBinding *root = sourceRoots->find(node.operation);
    mlir::Operation *cloned = cloneMapping.lookupOrNull(node.operation);
    if (!root || !cloned)
      return fail<AttentionMaterializationSource>(
          failureReason, "source structured operation was not cloned");
    auto spatial = sourcePartitions.find(root->key);
    auto temporal = sourceTemporalTiles.find(root->key);
    if (spatial == sourcePartitions.end() ||
        temporal == sourceTemporalTiles.end())
      return fail<AttentionMaterializationSource>(
          failureReason,
          "source structured operation has no resolved physical plan");
    if (!preservedOperations
             .try_emplace(cloned,
                          PreservedOperationPlan{spatial->second,
                                                 temporal->second, root->key})
             .second)
      return fail<AttentionMaterializationSource>(
          failureReason, "source structured clone identity is duplicated");
  }

  std::map<SemanticRootKey, mlir::Operation *> sourceAttention;
  for (const analysis::RootRegionWork &work : plan.rootWorks) {
    if (!work.rootOperation ||
        !mlir::isa<LinalgExtAttentionOp>(work.rootOperation))
      continue;
    auto [position, inserted] =
        sourceAttention.try_emplace(work.id.root, work.rootOperation);
    if (!inserted && position->second != work.rootOperation)
      return fail<AttentionMaterializationSource>(
          failureReason,
          "one selected attention root names several source operations");
  }

  std::map<mlir::Operation *, TileId> selectedOperationTiles;
  std::map<mlir::Operation *, AttentionActionId> selectedOperationActions;
  std::map<SemanticRootKey, AttentionIterationRoles> attentionRoles;
  mlir::IRRewriter rewriter(source.getContext());
  for (const AttentionWorkDescription &description :
       plan.preparedAttention.work.roots) {
    auto original = sourceAttention.find(description.root);
    mlir::Operation *mapped = original == sourceAttention.end()
                                  ? nullptr
                                  : cloneMapping.lookupOrNull(original->second);
    auto attention = mlir::dyn_cast_or_null<LinalgExtAttentionOp>(mapped);
    if (!attention)
      return fail<AttentionMaterializationSource>(
          failureReason, "selected attention root has no cloned operation");
    mlir::FailureOr<AttentionIterationRoles> roles =
        attention.getIterationRoles();
    if (mlir::failed(roles) ||
        !attentionRoles.try_emplace(description.root, *roles).second)
      return fail<AttentionMaterializationSource>(
          failureReason, "selected attention iterator roles are invalid");
    mlir::FailureOr<SelectedAttentionRootMaterialization> materialized =
        emitSelectedAttentionDecomposition(rewriter, attention, description,
                                           failureReason);
    if (mlir::failed(materialized))
      return mlir::failure();
    for (const AttentionScopeOperationMaterialization &scope :
         materialized->scopes) {
      TileId tile = workOf(scope.scope.execution).tile;
      for (mlir::Operation *operation : scope.operations) {
        auto [position, inserted] =
            selectedOperationTiles.try_emplace(operation, tile);
        if (!inserted && position->second != tile)
          return fail<AttentionMaterializationSource>(
              failureReason,
              "selected attention operation belongs to several Tiles");
      }
    }
    for (const AttentionActionMaterialization &action : materialized->actions) {
      TileId tile = workOf(action.id.scope.execution).tile;
      for (mlir::Operation *operation : action.structuredOperations) {
        auto assigned = selectedOperationTiles.find(operation);
        if (assigned == selectedOperationTiles.end() ||
            assigned->second != tile)
          return fail<AttentionMaterializationSource>(
              failureReason,
              "selected attention action operation has no scope Tile");
        if (!selectedOperationActions.try_emplace(operation, action.id).second)
          return fail<AttentionMaterializationSource>(
              failureReason,
              "selected structured operation belongs to several actions");
      }
    }
  }
  if (mlir::failed(mlir::verify(*selected)))
    return fail<AttentionMaterializationSource>(
        failureReason,
        "selected attention TensorProgram is not verifier-legal");

  mlir::FailureOr<StructuredDAGAnalysis> dag =
      StructuredDAGAnalysis::create(*program, failureReason);
  if (mlir::failed(dag))
    return mlir::failure();
  mlir::FailureOr<SemanticRootAnalysis> semanticRoots =
      SemanticRootAnalysis::create(*dag, failureReason);
  if (mlir::failed(semanticRoots))
    return mlir::failure();

  llvm::SmallVector<NodeIterationSpace, 64> iterationSpaces;
  SpatialPlan spatialPlan;
  llvm::SmallVector<StructuredOperationNodeMapping, 64> operationNodes;
  llvm::SmallVector<StructuredOpTemporalTile, 64> temporalTiles;
  std::map<mlir::Operation *, llvm::SmallVector<int64_t, 4>>
      temporalByOperation;
  std::map<mlir::Operation *, SemanticRootKey> plannedRoots;
  for (const StructuredDAGNode &node : dag->getNodes()) {
    auto tile = selectedOperationTiles.find(node.operation);
    auto preserved = preservedOperations.find(node.operation);
    if (tile == selectedOperationTiles.end() &&
        preserved == preservedOperations.end())
      return fail<AttentionMaterializationSource>(
          failureReason, "selected graph has an unowned structured operation");
    if (tile != selectedOperationTiles.end() &&
        preserved != preservedOperations.end())
      return fail<AttentionMaterializationSource>(
          failureReason,
          "selected structured operation has two physical owners");
    const SemanticRootBinding *root = semanticRoots->find(node.operation);
    if (!root)
      return fail<AttentionMaterializationSource>(
          failureReason,
          "selected attention actual op has no semantic identity");
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> extents =
        getStaticIterationExtents(node.operation, failureReason);
    if (mlir::failed(extents))
      return mlir::failure();
    iterationSpaces.push_back({root->key, *extents});
    NodeSpatialPlan nodePlan;
    llvm::SmallVector<int64_t, 4> selectedTemporal;
    if (tile != selectedOperationTiles.end()) {
      nodePlan.root = root->key;
      for (auto [iterator, extent] : llvm::enumerate(*extents)) {
        (void)extent;
        nodePlan.axes.push_back({static_cast<uint32_t>(iterator),
                                 IteratorPartitionScheme::BalancedParts, 1});
      }
      nodePlan.embedding.push_back(tile->second);
      auto action = selectedOperationActions.find(node.operation);
      if (action == selectedOperationActions.end())
        return fail<AttentionMaterializationSource>(
            failureReason,
            "selected attention structured op has no action owner");
      SemanticRootKey attentionRoot =
          workOf(action->second.scope.execution).root;
      plannedRoots.emplace(node.operation, attentionRoot);
      auto roles = attentionRoles.find(attentionRoot);
      if (roles == attentionRoles.end())
        return fail<AttentionMaterializationSource>(
            failureReason, "selected attention action has no iterator roles");
      llvm::ArrayRef<int64_t> sourceTemporal;
      auto temporal =
          sourceTemporalByExecution.find(action->second.scope.execution);
      if (temporal != sourceTemporalByExecution.end())
        sourceTemporal = temporal->second;
      else if (!std::holds_alternative<RequiredMergeExecution>(
                   action->second.scope.execution.source))
        return fail<AttentionMaterializationSource>(
            failureReason,
            "selected attention action has no temporal coordinate");
      mlir::FailureOr<llvm::SmallVector<int64_t, 4>> projected =
          getSelectedAttentionTemporalTile(node.operation, action->second,
                                           roles->second, sourceTemporal,
                                           *extents, failureReason);
      if (mlir::failed(projected))
        return mlir::failure();
      selectedTemporal = std::move(*projected);
    } else {
      plannedRoots.emplace(node.operation, preserved->second.root);
      mlir::FailureOr<NodeSpatialPlan> rebound = rebindNodeSpatialPlan(
          *preserved->second.spatial, root->key, *extents, failureReason);
      if (mlir::failed(rebound))
        return mlir::failure();
      nodePlan = std::move(*rebound);
      if (preserved->second.temporalTileSizes.size() != extents->size())
        return fail<AttentionMaterializationSource>(
            failureReason, "preserved operation temporal tile rank changed");
      selectedTemporal = preserved->second.temporalTileSizes;
    }
    spatialPlan.nodes.push_back(std::move(nodePlan));
    operationNodes.push_back({node.operation, node.id});
    temporalByOperation.emplace(node.operation, selectedTemporal);
    temporalTiles.push_back({node.operation, std::move(selectedTemporal), {}});
  }

  mlir::FailureOr<SpatialPlanningProblem> problem =
      SpatialPlanningProblem::create(iterationSpaces, availableTiles,
                                     failureReason);
  if (mlir::failed(problem))
    return mlir::failure();
  llvm::sort(spatialPlan.nodes);
  mlir::FailureOr<SpatialAssignment> spatial =
      closeSpatialPlanStructure(*problem, spatialPlan, failureReason);
  if (mlir::failed(spatial))
    return mlir::failure();
  analysis::ExactDemandProof selectedDemand;
  {
    wafer::support::ScopedCompileTimingSpan demandTiming(
        "query", "complete-candidate", "selected-attention-exact-demand");
    mlir::FailureOr<DemandPlanningSession> demandSession =
        DemandPlanningSession::create(*dag, analysis::IndexRelationLimits(),
                                      failureReason);
    if (mlir::failed(demandSession))
      return mlir::failure();
    analysis::ExactDemandOutcome demandOutcome = demandSession->query(*spatial);
    const analysis::ExactDemandProof *demand =
        analysis::getExactDemandProof(demandOutcome);
    if (!demand)
      return fail<AttentionMaterializationSource>(failureReason,
                                                  demandDetail(demandOutcome));
    selectedDemand = *demand;
    demandSession->close();
  }

  mlir::FailureOr<llvm::SmallVector<OutputTileMapping, 4>> outputs =
      buildOutputMappings(*dag, *spatial, selectedDemand, temporalByOperation,
                          selectedOperationActions, plan.preparedAttention.work,
                          failureReason);
  if (mlir::failed(outputs))
    return mlir::failure();
  CardMaterializationPlan assignment;
  assignment.spatial = std::move(*spatial);
  assignment.demand = std::move(selectedDemand);
  mlir::FailureOr<llvm::SmallVector<StructuredDAGNodePlacement, 64>>
      placements = buildSelectedNodePlacements(
          *dag, *semanticRoots, assignment.spatial, failureReason);
  if (mlir::failed(placements))
    return mlir::failure();
  assignment.nodePlacements.assign(std::make_move_iterator(placements->begin()),
                                   std::make_move_iterator(placements->end()));
  assignment.mapping.materializationMode =
      SpatialDataflowMaterializationMode::IndependentDDRStages;
  assignment.mapping.outputs = std::move(*outputs);
  assignment.mapping.operationTemporalTiles = std::move(temporalTiles);
  if (mlir::failed(addCardDataflowConstruction(assignment, *dag, plan.movement,
                                               failureReason)))
    return mlir::failure();

  // Selected attention actions are one semantic root execution, not a family
  // of independently scheduled candidate roots. Keep every operation in one
  // action scope under a shared materialization identity so its block states
  // remain in one TileRegion. Cross-scope and ordinary-root dependencies keep
  // explicit DDR/peer cuts below.
  std::map<AttentionWorkScopeId, uint32_t> scopeRepresentatives;
  for (const StructuredOperationNodeMapping &mapping : operationNodes) {
    auto action = selectedOperationActions.find(mapping.operation);
    if (action == selectedOperationActions.end())
      continue;
    auto [representative, inserted] = scopeRepresentatives.try_emplace(
        action->second.scope, mapping.structuredNodeId);
    if (!inserted)
      representative->second =
          std::min(representative->second, mapping.structuredNodeId);
  }
  for (StructuredOperationNodeMapping &mapping : operationNodes) {
    auto action = selectedOperationActions.find(mapping.operation);
    if (action == selectedOperationActions.end())
      continue;
    auto representative = scopeRepresentatives.find(action->second.scope);
    if (representative == scopeRepresentatives.end())
      return mlir::failure();
    mapping.structuredNodeId = representative->second;
  }
  std::map<uint32_t, SemanticRootKey> rootsByNode;
  for (const StructuredOperationNodeMapping &mapping : operationNodes) {
    auto root = plannedRoots.find(mapping.operation);
    if (root == plannedRoots.end())
      return mlir::failure();
    auto [position, inserted] =
        rootsByNode.try_emplace(mapping.structuredNodeId, root->second);
    if (!inserted && position->second != root->second)
      return fail<AttentionMaterializationSource>(
          failureReason,
          "one selected structured node maps to several semantic roots");
  }
  llvm::SmallVector<CandidateNodeRootRelation, 64> nodeRoots;
  for (const auto &[node, root] : rootsByNode)
    nodeRoots.push_back({node, root});
  if (statistics) {
    ++statistics->spatialCoordinateQueries;
    statistics->exactDemandSatisfiedEdges = dag->getEdges().size();
  }
  return AttentionMaterializationSource{
      std::move(selected), std::move(assignment), std::move(operationNodes),
      std::move(nodeRoots)};
}

} // namespace wafer::compiler::detail
