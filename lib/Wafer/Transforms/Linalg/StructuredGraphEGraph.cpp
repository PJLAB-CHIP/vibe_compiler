//===- StructuredGraphEGraph.cpp - Request-local egg adapter ------------===//

#include "StructuredGraphEGraph.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <limits>

namespace wafer::structured_graph_normalization {
namespace {

EGraphStatistics convertStatistics(const WaferEGraphStatistics &statistics) {
  return EGraphStatistics{statistics.inputNodes,
                          statistics.inputRelations,
                          statistics.relationQueries,
                          statistics.eNodes,
                          statistics.eClasses,
                          statistics.rewriteMatches,
                          statistics.eClassMerges,
                          statistics.rebuildWork,
                          statistics.iterations,
                          statistics.extractionWork,
                          statistics.outputNodes,
                          statistics.inputComputeOccurrences,
                          statistics.outputComputeOccurrences,
                          statistics.inputAccessOccurrences,
                          statistics.outputAccessOccurrences,
                          statistics.inputConcatOccurrences,
                          statistics.outputConcatOccurrences,
                          statistics.identityApplications,
                          statistics.compositionApplications,
                          statistics.computeAbsorptionApplications,
                          statistics.concatApplications,
                          statistics.resultReindexApplications,
                          statistics.reshapeThroughComputeApplications,
                          statistics.inputRecords,
                          statistics.outputRecords,
                          statistics.inputBytes,
                          statistics.outputBytes,
                          statistics.searchLimitReached,
                          statistics.applicationChecks,
                          statistics.unchangedApplications};
}

EGraphOutcomeKind convertStatus(uint32_t status) {
  switch (status) {
  case WAFER_EGRAPH_CHANGED:
    return EGraphOutcomeKind::Changed;
  case WAFER_EGRAPH_UNCHANGED:
    return EGraphOutcomeKind::Unchanged;
  case WAFER_EGRAPH_BUDGET_EXHAUSTED:
    return EGraphOutcomeKind::BudgetExhausted;
  case WAFER_EGRAPH_INVALID_INPUT:
    return EGraphOutcomeKind::InvalidInput;
  default:
    return EGraphOutcomeKind::InternalError;
  }
}

bool isSupportedNodeKind(uint32_t kind) {
  return kind >= WAFER_EGRAPH_INPUT && kind <= WAFER_EGRAPH_CONCAT;
}

bool validateNodeShape(const EGraphNode &node) {
  switch (node.kind) {
  case EGraphNodeKind::Input:
    return node.semanticId != 0 && node.children.empty() &&
           node.relations.empty() && node.dataInputCount == 0 && node.axis < 0;
  case EGraphNodeKind::Access:
    return node.semanticId == 0 && node.children.size() == 1 &&
           node.relations.size() == 1 && node.dataInputCount == 0 &&
           node.axis < 0;
  case EGraphNodeKind::Concat:
    return node.semanticId == 0 && !node.children.empty() &&
           node.relations.empty() && node.dataInputCount == 0 && node.axis >= 0;
  case EGraphNodeKind::Elementwise:
  case EGraphNodeKind::Contraction:
  case EGraphNodeKind::Reduction:
    return node.semanticId != 0 && !node.children.empty() &&
           node.relations.size() == node.children.size() &&
           node.dataInputCount <= node.children.size() && node.axis < 0;
  }
  return false;
}

} // namespace

EGraphOutcome runEGraph(llvm::ArrayRef<EGraphNode> nodes,
                        llvm::ArrayRef<uint32_t> rootNodes,
                        EGraphWorkBudget budget,
                        WaferEGraphRelationService relationService,
                        bool forcePanicForTesting) {
  EGraphOutcome invalid;
  invalid.kind = EGraphOutcomeKind::InvalidInput;
  if (nodes.size() > std::numeric_limits<uint32_t>::max() ||
      rootNodes.empty() ||
      llvm::any_of(rootNodes,
                   [&](uint32_t root) { return root >= nodes.size(); }) ||
      budget.maximumENodes == 0 || budget.maximumMatches == 0 ||
      budget.maximumIterations == 0 || !relationService.context ||
      !relationService.getRelationFacts || !relationService.composeRelations ||
      !relationService.validateCompute || !relationService.reindexCompute ||
      !relationService.reparameterizeElementwise ||
      !relationService.reparameterizeReshapeCompute ||
      !relationService.validateConcat || !relationService.factorConcat)
    return invalid;

  llvm::SmallVector<WaferEGraphNode, 16> abiNodes;
  llvm::SmallVector<uint32_t, 32> children;
  llvm::SmallVector<uint32_t, 32> relations;
  abiNodes.reserve(nodes.size());
  for (auto [index, node] : llvm::enumerate(nodes)) {
    if (node.typeId == 0 || !validateNodeShape(node) ||
        node.children.size() > UINT32_MAX || node.relations.size() > UINT32_MAX)
      return invalid;
    uint64_t childEnd = children.size() + node.children.size();
    uint64_t relationEnd = relations.size() + node.relations.size();
    if (childEnd > UINT32_MAX || relationEnd > UINT32_MAX)
      return invalid;
    for (uint32_t child : node.children) {
      if (child >= index)
        return invalid;
      children.push_back(child);
    }
    relations.append(node.relations.begin(), node.relations.end());
    abiNodes.push_back(WaferEGraphNode{
        static_cast<uint32_t>(node.kind), node.semanticId, node.typeId,
        static_cast<uint32_t>(childEnd - node.children.size()),
        static_cast<uint32_t>(node.children.size()),
        static_cast<uint32_t>(relationEnd - node.relations.size()),
        static_cast<uint32_t>(node.relations.size()), node.dataInputCount,
        node.axis, node.sourceOrder});
  }

  WaferEGraphRequest request{
      WAFER_EGRAPH_SCHEMA_VERSION,
      forcePanicForTesting
          ? static_cast<uint32_t>(WAFER_EGRAPH_FLAG_FORCE_PANIC)
          : 0,
      abiNodes.data(),
      abiNodes.size(),
      children.data(),
      children.size(),
      relations.data(),
      relations.size(),
      rootNodes.data(),
      rootNodes.size(),
      budget.maximumIterations,
      budget.maximumENodes,
      budget.maximumMatches,
      relationService};
  WaferEGraphResult *result = waferRunStructuredEGraph(&request);
  if (!result)
    return EGraphOutcome{EGraphOutcomeKind::InternalError};
  auto release =
      llvm::make_scope_exit([&] { waferFreeStructuredEGraphResult(result); });

  EGraphOutcome outcome;
  outcome.kind = convertStatus(result->status);
  outcome.statistics = convertStatistics(result->statistics);
  if (outcome.kind != EGraphOutcomeKind::Changed)
    return outcome;
  if (!result->nodes || result->nodeCount == 0 ||
      result->nodeCount > UINT32_MAX || !result->rootNodes ||
      result->rootCount == 0 || result->rootCount > UINT32_MAX ||
      (result->childCount != 0 && !result->children) ||
      (result->relationCount != 0 && !result->relations))
    return EGraphOutcome{EGraphOutcomeKind::InternalError};

  llvm::ArrayRef<WaferEGraphNode> resultNodes(result->nodes, result->nodeCount);
  llvm::ArrayRef<uint32_t> resultChildren(result->children, result->childCount);
  llvm::ArrayRef<uint32_t> resultRelations(result->relations,
                                           result->relationCount);
  llvm::ArrayRef<uint32_t> resultRoots(result->rootNodes, result->rootCount);
  if (llvm::any_of(resultRoots,
                   [&](uint32_t root) { return root >= result->nodeCount; }))
    return EGraphOutcome{EGraphOutcomeKind::InternalError};
  outcome.expression.reserve(resultNodes.size());
  for (auto [index, node] : llvm::enumerate(resultNodes)) {
    uint64_t childEnd =
        static_cast<uint64_t>(node.childOffset) + node.childCount;
    uint64_t relationEnd =
        static_cast<uint64_t>(node.relationOffset) + node.relationCount;
    if (!isSupportedNodeKind(node.kind) || node.typeId == 0 ||
        childEnd > resultChildren.size() ||
        relationEnd > resultRelations.size())
      return EGraphOutcome{EGraphOutcomeKind::InternalError};
    EGraphNode converted;
    converted.kind = static_cast<EGraphNodeKind>(node.kind);
    converted.semanticId = node.semanticId;
    converted.typeId = node.typeId;
    converted.dataInputCount = node.dataInputCount;
    converted.axis = node.axis;
    converted.sourceOrder = node.sourceOrder;
    converted.relations.append(resultRelations.begin() + node.relationOffset,
                               resultRelations.begin() + relationEnd);
    for (uint32_t child :
         resultChildren.slice(node.childOffset, node.childCount)) {
      if (child >= index)
        return EGraphOutcome{EGraphOutcomeKind::InternalError};
      converted.children.push_back(child);
    }
    if (!validateNodeShape(converted))
      return EGraphOutcome{EGraphOutcomeKind::InternalError};
    outcome.expression.push_back(std::move(converted));
  }
  outcome.rootNodes.append(resultRoots.begin(), resultRoots.end());
  return outcome;
}

} // namespace wafer::structured_graph_normalization
