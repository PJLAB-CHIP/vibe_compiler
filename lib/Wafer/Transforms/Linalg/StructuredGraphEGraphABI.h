//===- StructuredGraphEGraphABI.h - Rust e-graph C ABI ------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHEGRAPHABI_H
#define WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHEGRAPHABI_H

#include <cstdint>

extern "C" {

enum : uint32_t {
  WAFER_EGRAPH_SCHEMA_VERSION = 2,
  WAFER_EGRAPH_FLAG_FORCE_PANIC = 1u << 0,
};

enum WaferEGraphStatus : uint32_t {
  WAFER_EGRAPH_CHANGED = 0,
  WAFER_EGRAPH_UNCHANGED = 1,
  WAFER_EGRAPH_BUDGET_EXHAUSTED = 2,
  WAFER_EGRAPH_INVALID_INPUT = 3,
  WAFER_EGRAPH_INTERNAL_ERROR = 4,
};

enum WaferEGraphCallbackStatus : uint32_t {
  WAFER_EGRAPH_CALLBACK_EXACT = 0,
  WAFER_EGRAPH_CALLBACK_UNSUPPORTED = 1,
  WAFER_EGRAPH_CALLBACK_WORK_LIMIT = 2,
  WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR = 3,
};

enum WaferEGraphNodeKind : uint32_t {
  WAFER_EGRAPH_INPUT = 1,
  WAFER_EGRAPH_ACCESS = 2,
  WAFER_EGRAPH_ELEMENTWISE = 3,
  WAFER_EGRAPH_CONTRACTION = 4,
  WAFER_EGRAPH_REDUCTION = 5,
  WAFER_EGRAPH_CONCAT = 6,
};

enum WaferEGraphRelationFlag : uint32_t {
  WAFER_EGRAPH_RELATION_IDENTITY = 1u << 0,
  WAFER_EGRAPH_RELATION_TOTAL = 1u << 1,
  WAFER_EGRAPH_RELATION_SINGLE_VALUED = 1u << 2,
  WAFER_EGRAPH_RELATION_INJECTIVE = 1u << 3,
  WAFER_EGRAPH_RELATION_BIJECTIVE = 1u << 4,
  WAFER_EGRAPH_RELATION_MATERIALIZABLE = 1u << 5,
  WAFER_EGRAPH_RELATION_CANONICAL_RESHAPE = 1u << 6,
};

struct WaferEGraphNode {
  uint32_t kind;
  uint32_t semanticId;
  uint32_t typeId;
  uint32_t childOffset;
  uint32_t childCount;
  uint32_t relationOffset;
  uint32_t relationCount;
  uint32_t dataInputCount;
  int32_t axis;
  uint32_t sourceOrder;
};

struct WaferEGraphRelationFacts {
  uint32_t destinationTypeId;
  uint32_t sourceTypeId;
  uint32_t flags;
  uint32_t reserved;
};

using WaferEGraphGetRelationFactsFn = uint32_t (*)(
    void *context, uint32_t relationId, WaferEGraphRelationFacts *facts);

using WaferEGraphComposeRelationsFn = uint32_t (*)(void *context,
                                                   uint32_t outerRelationId,
                                                   uint32_t innerRelationId,
                                                   uint32_t *resultRelationId);

using WaferEGraphValidateComputeFn = uint32_t (*)(
    void *context, uint32_t semanticId, uint32_t kind, uint32_t resultTypeId,
    const uint32_t *relations, const uint32_t *operandTypeIds,
    uint64_t relationCount, uint32_t dataInputCount);

using WaferEGraphReindexComputeFn =
    uint32_t (*)(void *context, uint32_t semanticId, uint32_t kind,
                 uint32_t sourceResultTypeId, uint32_t destinationResultTypeId,
                 uint32_t resultRelationId, const uint32_t *inputRelations,
                 uint64_t relationCount, uint32_t dataInputCount,
                 uint32_t *outputRelations, uint32_t *reindexReadInit);

using WaferEGraphReparameterizeElementwiseFn = uint32_t (*)(
    void *context, uint32_t semanticId, uint32_t sourceResultTypeId,
    uint32_t destinationResultTypeId, uint32_t resultRelationId,
    const uint32_t *inputRelations, uint64_t relationCount,
    uint32_t dataInputCount, uint32_t *outputComputeRelations,
    uint32_t *outputOperandAccessRelations,
    uint32_t *outputOperandAccessTypeIds);

using WaferEGraphReparameterizeReshapeComputeFn = uint32_t (*)(
    void *context, uint32_t semanticId, uint32_t kind, uint32_t resultTypeId,
    uint32_t accessOperand, uint32_t accessRelationId,
    const uint32_t *inputRelations, uint64_t relationCount,
    uint32_t dataInputCount, uint32_t *innerResultTypeId,
    uint32_t *outerResultRelationId, uint32_t *outputComputeRelations,
    uint32_t *outputOperandAccessRelations,
    uint32_t *outputOperandAccessTypeIds);

using WaferEGraphValidateConcatFn = uint32_t (*)(void *context,
                                                 uint32_t resultTypeId,
                                                 int32_t axis,
                                                 const uint32_t *inputTypeIds,
                                                 uint64_t inputCount);

using WaferEGraphFactorConcatFn = uint32_t (*)(
    void *context, uint32_t resultTypeId, int32_t destinationAxis,
    const uint32_t *inputRelationIds, const uint32_t *sourceTypeIds,
    uint64_t inputCount, uint32_t *resultRelationId,
    uint32_t *sourceConcatTypeId, int32_t *sourceAxis);

struct WaferEGraphRelationService {
  void *context;
  WaferEGraphGetRelationFactsFn getRelationFacts;
  WaferEGraphComposeRelationsFn composeRelations;
  WaferEGraphValidateComputeFn validateCompute;
  WaferEGraphReindexComputeFn reindexCompute;
  WaferEGraphReparameterizeElementwiseFn reparameterizeElementwise;
  WaferEGraphReparameterizeReshapeComputeFn reparameterizeReshapeCompute;
  WaferEGraphValidateConcatFn validateConcat;
  WaferEGraphFactorConcatFn factorConcat;
};

struct WaferEGraphRequest {
  uint32_t schemaVersion;
  uint32_t flags;
  const WaferEGraphNode *nodes;
  uint64_t nodeCount;
  const uint32_t *children;
  uint64_t childCount;
  const uint32_t *relations;
  uint64_t relationCount;
  const uint32_t *rootNodes;
  uint64_t rootCount;
  uint32_t maximumIterations;
  uint64_t maximumENodes;
  uint64_t maximumMatches;
  WaferEGraphRelationService relationService;
};

struct WaferEGraphStatistics {
  uint64_t inputNodes;
  uint64_t inputRelations;
  uint64_t relationQueries;
  uint64_t eNodes;
  uint64_t eClasses;
  uint64_t rewriteMatches;
  uint64_t eClassMerges;
  uint64_t rebuildWork;
  uint64_t iterations;
  uint64_t extractionWork;
  uint64_t outputNodes;
  uint64_t inputComputeOccurrences;
  uint64_t outputComputeOccurrences;
  uint64_t inputAccessOccurrences;
  uint64_t outputAccessOccurrences;
  uint64_t inputConcatOccurrences;
  uint64_t outputConcatOccurrences;
  uint64_t identityApplications;
  uint64_t compositionApplications;
  uint64_t computeAbsorptionApplications;
  uint64_t concatApplications;
  uint64_t resultReindexApplications;
  uint64_t reshapeThroughComputeApplications;
  uint64_t inputRecords;
  uint64_t outputRecords;
  uint64_t inputBytes;
  uint64_t outputBytes;
};

struct WaferEGraphResult {
  uint32_t status;
  uint32_t reserved;
  WaferEGraphNode *nodes;
  uint64_t nodeCount;
  uint32_t *children;
  uint64_t childCount;
  uint32_t *relations;
  uint64_t relationCount;
  uint32_t *rootNodes;
  uint64_t rootCount;
  WaferEGraphStatistics statistics;
};

WaferEGraphResult *waferRunStructuredEGraph(const WaferEGraphRequest *request);

void waferFreeStructuredEGraphResult(WaferEGraphResult *result);

} // extern "C"

static_assert(sizeof(WaferEGraphNode) == 40, "Wafer e-graph node ABI changed");
static_assert(sizeof(WaferEGraphRelationFacts) == 16,
              "Wafer e-graph relation facts ABI changed");

#endif // WAFER_TRANSFORMS_LINALG_STRUCTUREDGRAPHEGRAPHABI_H
