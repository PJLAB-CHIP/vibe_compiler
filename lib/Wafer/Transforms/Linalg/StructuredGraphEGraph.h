//===- StructuredGraphEGraph.h - Request-local egg adapter ----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHEGRAPH_H
#define WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHEGRAPH_H

#include "StructuredGraphEGraphABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace wafer::structured_graph_normalization {

enum class EGraphNodeKind : uint32_t {
  Input = WAFER_EGRAPH_INPUT,
  Access = WAFER_EGRAPH_ACCESS,
  Elementwise = WAFER_EGRAPH_ELEMENTWISE,
  Contraction = WAFER_EGRAPH_CONTRACTION,
  Reduction = WAFER_EGRAPH_REDUCTION,
  Concat = WAFER_EGRAPH_CONCAT,
};

struct EGraphNode {
  EGraphNodeKind kind = EGraphNodeKind::Input;
  uint32_t semanticId = 0;
  uint32_t typeId = 0;
  llvm::SmallVector<uint32_t, 4> children;
  llvm::SmallVector<uint32_t, 4> relations;
  uint32_t dataInputCount = 0;
  int32_t axis = -1;
  uint32_t sourceOrder = 0;
};

struct EGraphWorkBudget {
  uint64_t maximumENodes = 0;
  uint64_t maximumMatches = 0;
  uint32_t maximumIterations = 0;
};

struct EGraphStatistics {
  uint64_t inputNodes = 0;
  uint64_t inputRelations = 0;
  uint64_t relationQueries = 0;
  uint64_t eNodes = 0;
  uint64_t eClasses = 0;
  uint64_t rewriteMatches = 0;
  uint64_t eClassMerges = 0;
  uint64_t rebuildWork = 0;
  uint64_t iterations = 0;
  uint64_t extractionWork = 0;
  uint64_t outputNodes = 0;
  uint64_t inputComputeOccurrences = 0;
  uint64_t outputComputeOccurrences = 0;
  uint64_t inputAccessOccurrences = 0;
  uint64_t outputAccessOccurrences = 0;
  uint64_t inputConcatOccurrences = 0;
  uint64_t outputConcatOccurrences = 0;
  uint64_t identityApplications = 0;
  uint64_t compositionApplications = 0;
  uint64_t computeAbsorptionApplications = 0;
  uint64_t concatApplications = 0;
  uint64_t resultReindexApplications = 0;
  uint64_t reshapeThroughComputeApplications = 0;
  uint64_t inputRecords = 0;
  uint64_t outputRecords = 0;
  uint64_t inputBytes = 0;
  uint64_t outputBytes = 0;
  uint64_t searchLimitReached = 0;
};

enum class EGraphOutcomeKind {
  Changed,
  Unchanged,
  BudgetExhausted,
  InvalidInput,
  InternalError,
};

struct EGraphOutcome {
  EGraphOutcomeKind kind = EGraphOutcomeKind::InternalError;
  llvm::SmallVector<EGraphNode, 16> expression;
  llvm::SmallVector<uint32_t, 4> rootNodes;
  EGraphStatistics statistics;
};

EGraphOutcome runEGraph(llvm::ArrayRef<EGraphNode> nodes,
                        llvm::ArrayRef<uint32_t> rootNodes,
                        EGraphWorkBudget budget,
                        WaferEGraphRelationService relationService,
                        bool forcePanicForTesting = false);

} // namespace wafer::structured_graph_normalization

#endif // WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHEGRAPH_H
