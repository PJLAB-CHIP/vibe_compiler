//===- StructuredGraphEGraphTest.cpp - Rust dynamic-rule C ABI tests ----===//

#include "Wafer/Transforms/Linalg/StructuredGraphEGraph.h"
#include "Wafer/Transforms/Linalg/StructuredGraphEGraphABI.h"

#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

#include <future>
#include <map>
#include <utility>

namespace {

using namespace wafer::structured_graph_normalization;

constexpr uint32_t kMaterializable = WAFER_EGRAPH_RELATION_TOTAL |
                                     WAFER_EGRAPH_RELATION_SINGLE_VALUED |
                                     WAFER_EGRAPH_RELATION_MATERIALIZABLE;

struct FakeRelation {
  uint32_t destinationType = 0;
  uint32_t sourceType = 0;
  uint32_t flags = 0;
};

struct FakeRelationService {
  std::map<uint32_t, FakeRelation> relations;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> compositions;
  uint32_t composeStatus = WAFER_EGRAPH_CALLBACK_EXACT;
  uint32_t commonConcatInputRelation = 0;
  uint32_t commonConcatResultRelation = 0;
  uint32_t commonConcatSourceType = 0;
  int32_t commonConcatSourceAxis = -1;
  uint32_t reindexInputRelation = 0;
  llvm::SmallVector<uint32_t, 4> reindexedRelations;
  bool reindexReadInit = false;

  WaferEGraphRelationService getABI() {
    return WaferEGraphRelationService{
        this,
        [](void *context, uint32_t relationId,
           WaferEGraphRelationFacts *facts) -> uint32_t {
          auto *service = static_cast<FakeRelationService *>(context);
          auto found = service->relations.find(relationId);
          if (!facts || found == service->relations.end())
            return WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
          *facts = WaferEGraphRelationFacts{found->second.destinationType,
                                            found->second.sourceType,
                                            found->second.flags, 0};
          return WAFER_EGRAPH_CALLBACK_EXACT;
        },
        [](void *context, uint32_t outer, uint32_t inner,
           uint32_t *result) -> uint32_t {
          auto *service = static_cast<FakeRelationService *>(context);
          if (service->composeStatus != WAFER_EGRAPH_CALLBACK_EXACT)
            return service->composeStatus;
          auto found = service->compositions.find({outer, inner});
          if (!result || found == service->compositions.end())
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          *result = found->second;
          return WAFER_EGRAPH_CALLBACK_EXACT;
        },
        [](void *context, uint32_t semanticId, uint32_t kind,
           uint32_t resultTypeId, const uint32_t *relationIds,
           const uint32_t *operandTypeIds, uint64_t relationCount,
           uint32_t dataInputCount) -> uint32_t {
          auto *service = static_cast<FakeRelationService *>(context);
          if (!semanticId || kind < WAFER_EGRAPH_ELEMENTWISE ||
              kind > WAFER_EGRAPH_REDUCTION || !relationCount || !relationIds ||
              !operandTypeIds || dataInputCount > relationCount)
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          uint32_t iterationType = 0;
          for (uint64_t index = 0; index < relationCount; ++index) {
            auto found = service->relations.find(relationIds[index]);
            if (found == service->relations.end())
              return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
            if (!iterationType)
              iterationType = found->second.destinationType;
            if (found->second.destinationType != iterationType)
              return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
            const bool sourceMatches =
                found->second.sourceType == operandTypeIds[index];
            const bool replacedUnreadInit =
                index >= dataInputCount &&
                found->second.sourceType == resultTypeId;
            if (!sourceMatches && !replacedUnreadInit)
              return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          }
          return WAFER_EGRAPH_CALLBACK_EXACT;
        },
        [](void *context, uint32_t, uint32_t, uint32_t, uint32_t,
           uint32_t resultRelation, const uint32_t *, uint64_t relationCount,
           uint32_t, uint32_t *outputRelations,
           uint32_t *reindexReadInit) -> uint32_t {
          auto *service = static_cast<FakeRelationService *>(context);
          if (resultRelation != service->reindexInputRelation ||
              relationCount != service->reindexedRelations.size() ||
              (relationCount && !outputRelations) || !reindexReadInit)
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          llvm::copy(service->reindexedRelations, outputRelations);
          *reindexReadInit = service->reindexReadInit ? 1 : 0;
          return WAFER_EGRAPH_CALLBACK_EXACT;
        },
        [](void *, uint32_t, uint32_t, uint32_t, uint32_t, const uint32_t *,
           uint64_t, uint32_t, uint32_t *, uint32_t *, uint32_t *) -> uint32_t {
          return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
        },
        [](void *, uint32_t resultTypeId, int32_t axis,
           const uint32_t *inputTypeIds, uint64_t inputCount) -> uint32_t {
          if (!resultTypeId || axis < 0 || !inputCount || !inputTypeIds)
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          return WAFER_EGRAPH_CALLBACK_EXACT;
        },
        [](void *context, uint32_t, int32_t, const uint32_t *relations,
           const uint32_t *, uint64_t inputCount, uint32_t *resultRelation,
           uint32_t *sourceConcatType, int32_t *sourceAxis) -> uint32_t {
          auto *service = static_cast<FakeRelationService *>(context);
          if (inputCount < 2 || !relations || !resultRelation ||
              !sourceConcatType || !sourceAxis ||
              !service->commonConcatInputRelation ||
              !llvm::all_of(
                  llvm::ArrayRef<uint32_t>(relations, inputCount),
                  [&](uint32_t relation) {
                    return relation == service->commonConcatInputRelation;
                  }))
            return WAFER_EGRAPH_CALLBACK_UNSUPPORTED;
          *resultRelation = service->commonConcatResultRelation;
          *sourceConcatType = service->commonConcatSourceType;
          *sourceAxis = service->commonConcatSourceAxis;
          return WAFER_EGRAPH_CALLBACK_EXACT;
        }};
  }
};

EGraphNode input(uint32_t semanticId, uint32_t typeId,
                 uint32_t sourceOrder = 0) {
  EGraphNode node;
  node.kind = EGraphNodeKind::Input;
  node.semanticId = semanticId;
  node.typeId = typeId;
  node.sourceOrder = sourceOrder;
  return node;
}

EGraphNode access(uint32_t typeId, uint32_t child, uint32_t relation,
                  uint32_t sourceOrder = 0) {
  EGraphNode node;
  node.kind = EGraphNodeKind::Access;
  node.typeId = typeId;
  node.children.push_back(child);
  node.relations.push_back(relation);
  node.sourceOrder = sourceOrder;
  return node;
}

EGraphNode concat(uint32_t typeId, llvm::ArrayRef<uint32_t> children,
                  int32_t axis, uint32_t sourceOrder = 0) {
  EGraphNode node;
  node.kind = EGraphNodeKind::Concat;
  node.typeId = typeId;
  llvm::append_range(node.children, children);
  node.axis = axis;
  node.sourceOrder = sourceOrder;
  return node;
}

EGraphNode compute(EGraphNodeKind kind, uint32_t semanticId, uint32_t typeId,
                   llvm::ArrayRef<uint32_t> children,
                   llvm::ArrayRef<uint32_t> relations, uint32_t dataInputCount,
                   uint32_t sourceOrder = 0) {
  EGraphNode node;
  node.kind = kind;
  node.semanticId = semanticId;
  node.typeId = typeId;
  llvm::append_range(node.children, children);
  llvm::append_range(node.relations, relations);
  node.dataInputCount = dataInputCount;
  node.sourceOrder = sourceOrder;
  return node;
}

EGraphWorkBudget budget() {
  return {/*maximumENodes=*/256, /*maximumMatches=*/512,
          /*maximumIterations=*/8};
}

FakeRelationService makeInverseAccessService() {
  FakeRelationService service;
  service.relations.emplace(1, FakeRelation{2, 1, kMaterializable});
  service.relations.emplace(2, FakeRelation{1, 2, kMaterializable});
  service.relations.emplace(
      3, FakeRelation{1, 1,
                      kMaterializable | WAFER_EGRAPH_RELATION_IDENTITY |
                          WAFER_EGRAPH_RELATION_INJECTIVE |
                          WAFER_EGRAPH_RELATION_BIJECTIVE});
  service.compositions.emplace(std::pair{2u, 1u}, 3);
  return service;
}

TEST(StructuredGraphEGraphTest,
     DynamicCompositionThenIdentityCreatesAndExtractsNewNodes) {
  FakeRelationService service = makeInverseAccessService();
  llvm::SmallVector<EGraphNode, 3> nodes;
  nodes.push_back(input(1, 1, 1));
  nodes.push_back(access(2, 0, 1, 2));
  nodes.push_back(access(1, 1, 2, 3));

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/2, budget(), service.getABI());
  ASSERT_EQ(outcome.kind, EGraphOutcomeKind::Changed);
  ASSERT_EQ(outcome.expression.size(), 1u);
  EXPECT_EQ(outcome.expression.front().kind, EGraphNodeKind::Input);
  EXPECT_EQ(outcome.statistics.inputAccessOccurrences, 2u);
  EXPECT_EQ(outcome.statistics.outputAccessOccurrences, 0u);
  EXPECT_EQ(outcome.statistics.inputRecords, 7u);
  EXPECT_EQ(outcome.statistics.outputRecords, 1u);
  EXPECT_GT(outcome.statistics.inputBytes, outcome.statistics.outputBytes);
  EXPECT_GE(outcome.statistics.compositionApplications, 1u);
  EXPECT_GE(outcome.statistics.identityApplications, 1u);
}

TEST(StructuredGraphEGraphTest,
     CommonConcatAccessFeedsCompositionAndIdentityInLaterIterations) {
  FakeRelationService service;
  service.relations.emplace(1, FakeRelation{2, 1, kMaterializable});
  service.relations.emplace(6, FakeRelation{3, 4, kMaterializable});
  service.relations.emplace(7, FakeRelation{4, 3, kMaterializable});
  service.relations.emplace(
      8, FakeRelation{4, 4,
                      kMaterializable | WAFER_EGRAPH_RELATION_IDENTITY |
                          WAFER_EGRAPH_RELATION_INJECTIVE |
                          WAFER_EGRAPH_RELATION_BIJECTIVE});
  service.commonConcatInputRelation = 1;
  service.commonConcatResultRelation = 6;
  service.commonConcatSourceType = 4;
  service.commonConcatSourceAxis = 0;
  service.compositions.emplace(std::pair{7u, 6u}, 8);

  llvm::SmallVector<EGraphNode, 6> nodes;
  nodes.push_back(input(1, 1, 1));
  nodes.push_back(input(2, 1, 2));
  nodes.push_back(access(2, 0, 1, 3));
  nodes.push_back(access(2, 1, 1, 4));
  nodes.push_back(concat(3, {2, 3}, /*axis=*/0, 5));
  nodes.push_back(access(4, 4, 7, 6));

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/5, budget(), service.getABI());
  ASSERT_EQ(outcome.kind, EGraphOutcomeKind::Changed);
  ASSERT_EQ(outcome.expression.size(), 3u);
  EXPECT_EQ(outcome.expression.back().kind, EGraphNodeKind::Concat);
  EXPECT_EQ(outcome.expression.back().typeId, 4u);
  EXPECT_EQ(outcome.statistics.outputAccessOccurrences, 0u);
  EXPECT_GE(outcome.statistics.concatApplications, 1u);
  EXPECT_GE(outcome.statistics.compositionApplications, 1u);
  EXPECT_GE(outcome.statistics.identityApplications, 1u);
}

TEST(StructuredGraphEGraphTest, SingleInputConcatExtractsItsInput) {
  FakeRelationService service;
  llvm::SmallVector<EGraphNode, 2> nodes;
  nodes.push_back(input(1, 1, 1));
  nodes.push_back(concat(1, {0}, /*axis=*/0, 2));

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/1, budget(), service.getABI());
  ASSERT_EQ(outcome.kind, EGraphOutcomeKind::Changed);
  ASSERT_EQ(outcome.expression.size(), 1u);
  EXPECT_EQ(outcome.expression.front().kind, EGraphNodeKind::Input);
  EXPECT_EQ(outcome.statistics.outputConcatOccurrences, 0u);
  EXPECT_GE(outcome.statistics.concatApplications, 1u);
}

TEST(StructuredGraphEGraphTest,
     ComputeAbsorptionPreservesComputeOccurrenceAndSemanticId) {
  FakeRelationService service;
  service.relations.emplace(1, FakeRelation{2, 1, kMaterializable});
  service.relations.emplace(4, FakeRelation{9, 2, kMaterializable});
  service.relations.emplace(5, FakeRelation{9, 1, kMaterializable});
  service.compositions.emplace(std::pair{4u, 1u}, 5);

  llvm::SmallVector<EGraphNode, 4> nodes;
  nodes.push_back(input(1, 1, 1));
  nodes.push_back(access(2, 0, 1, 2));
  nodes.push_back(input(2, 2, 3));
  nodes.push_back(compute(EGraphNodeKind::Elementwise, /*semanticId=*/17,
                          /*typeId=*/2, {1, 2}, {4, 4},
                          /*dataInputCount=*/1, 4));

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/3, budget(), service.getABI());
  ASSERT_EQ(outcome.kind, EGraphOutcomeKind::Changed);
  ASSERT_EQ(outcome.expression.back().kind, EGraphNodeKind::Elementwise);
  EXPECT_EQ(outcome.expression.back().semanticId, 17u);
  EXPECT_EQ(outcome.statistics.inputComputeOccurrences, 1u);
  EXPECT_EQ(outcome.statistics.outputComputeOccurrences, 1u);
  EXPECT_EQ(outcome.statistics.outputAccessOccurrences, 0u);
  EXPECT_GE(outcome.statistics.computeAbsorptionApplications, 1u);
}

TEST(StructuredGraphEGraphTest, ResultReindexRemovesOnlyTheOuterAccess) {
  FakeRelationService service;
  service.relations.emplace(10, FakeRelation{9, 1, kMaterializable});
  service.relations.emplace(11, FakeRelation{9, 2, kMaterializable});
  service.relations.emplace(
      12,
      FakeRelation{3, 2, kMaterializable | WAFER_EGRAPH_RELATION_BIJECTIVE});
  service.relations.emplace(13, FakeRelation{10, 1, kMaterializable});
  service.relations.emplace(14, FakeRelation{10, 3, kMaterializable});
  service.reindexInputRelation = 12;
  service.reindexedRelations = {13, 14};

  llvm::SmallVector<EGraphNode, 4> nodes;
  nodes.push_back(input(1, 1, 1));
  nodes.push_back(input(2, 2, 2));
  nodes.push_back(compute(EGraphNodeKind::Elementwise, /*semanticId=*/23,
                          /*typeId=*/2, {0, 1}, {10, 11},
                          /*dataInputCount=*/1, 3));
  nodes.push_back(access(3, 2, 12, 4));

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/3, budget(), service.getABI());
  ASSERT_EQ(outcome.kind, EGraphOutcomeKind::Changed);
  EXPECT_EQ(outcome.expression.back().kind, EGraphNodeKind::Elementwise);
  EXPECT_EQ(outcome.expression.back().semanticId, 23u);
  EXPECT_EQ(outcome.expression.back().typeId, 3u);
  EXPECT_EQ(outcome.statistics.outputComputeOccurrences, 1u);
  EXPECT_EQ(outcome.statistics.outputAccessOccurrences, 0u);
  EXPECT_GE(outcome.statistics.resultReindexApplications, 1u);
}

TEST(StructuredGraphEGraphTest,
     MovingOneAccessFromResultToReadInitIsNotStrictlyBetter) {
  FakeRelationService service;
  service.relations.emplace(10, FakeRelation{9, 1, kMaterializable});
  service.relations.emplace(11, FakeRelation{9, 2, kMaterializable});
  service.relations.emplace(
      12,
      FakeRelation{3, 2, kMaterializable | WAFER_EGRAPH_RELATION_BIJECTIVE});
  service.relations.emplace(13, FakeRelation{9, 1, kMaterializable});
  service.relations.emplace(14, FakeRelation{9, 3, kMaterializable});
  service.reindexInputRelation = 12;
  service.reindexedRelations = {13, 14};
  service.reindexReadInit = true;

  llvm::SmallVector<EGraphNode, 4> nodes;
  nodes.push_back(input(1, 1, 1));
  nodes.push_back(input(2, 2, 2));
  nodes.push_back(compute(EGraphNodeKind::Reduction, /*semanticId=*/29,
                          /*typeId=*/2, {0, 1}, {10, 11},
                          /*dataInputCount=*/1, 3));
  nodes.push_back(access(3, 2, 12, 4));

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/3, budget(), service.getABI());
  EXPECT_EQ(outcome.kind, EGraphOutcomeKind::Unchanged);
  EXPECT_TRUE(outcome.expression.empty());
  EXPECT_GE(outcome.statistics.resultReindexApplications, 1u);
}

TEST(StructuredGraphEGraphTest, MatchBudgetExhaustionHasNoPartialExpression) {
  FakeRelationService service = makeInverseAccessService();
  llvm::SmallVector<EGraphNode, 3> nodes;
  nodes.push_back(input(1, 1));
  nodes.push_back(access(2, 0, 1));
  nodes.push_back(access(1, 1, 2));
  EGraphWorkBudget exhausted = budget();
  exhausted.maximumMatches = 1;

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/2, exhausted, service.getABI());
  EXPECT_EQ(outcome.kind, EGraphOutcomeKind::BudgetExhausted);
  EXPECT_TRUE(outcome.expression.empty());
}

TEST(StructuredGraphEGraphTest, TypedRelationWorkLimitStopsTheRequest) {
  FakeRelationService service = makeInverseAccessService();
  service.composeStatus = WAFER_EGRAPH_CALLBACK_WORK_LIMIT;
  llvm::SmallVector<EGraphNode, 3> nodes;
  nodes.push_back(input(1, 1));
  nodes.push_back(access(2, 0, 1));
  nodes.push_back(access(1, 1, 2));

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/2, budget(), service.getABI());
  EXPECT_EQ(outcome.kind, EGraphOutcomeKind::BudgetExhausted);
  EXPECT_TRUE(outcome.expression.empty());
}

TEST(StructuredGraphEGraphTest, TypedRelationInternalErrorStopsTheRequest) {
  FakeRelationService service = makeInverseAccessService();
  service.composeStatus = WAFER_EGRAPH_CALLBACK_INTERNAL_ERROR;
  llvm::SmallVector<EGraphNode, 3> nodes;
  nodes.push_back(input(1, 1));
  nodes.push_back(access(2, 0, 1));
  nodes.push_back(access(1, 1, 2));

  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/2, budget(), service.getABI());
  EXPECT_EQ(outcome.kind, EGraphOutcomeKind::InternalError);
  EXPECT_TRUE(outcome.expression.empty());
}

TEST(StructuredGraphEGraphTest, RepeatedAndConcurrentRequestsAreDeterministic) {
  auto run = [] {
    FakeRelationService service = makeInverseAccessService();
    llvm::SmallVector<EGraphNode, 3> nodes;
    nodes.push_back(input(1, 1, 1));
    nodes.push_back(access(2, 0, 1, 2));
    nodes.push_back(access(1, 1, 2, 3));
    return runEGraph(nodes, /*rootNode=*/2, budget(), service.getABI());
  };
  EGraphOutcome reference = run();
  ASSERT_EQ(reference.kind, EGraphOutcomeKind::Changed);
  llvm::SmallVector<std::future<EGraphOutcome>, 4> requests;
  for (unsigned index = 0; index < 4; ++index)
    requests.push_back(std::async(std::launch::async, run));
  for (std::future<EGraphOutcome> &request : requests) {
    EGraphOutcome outcome = request.get();
    ASSERT_EQ(outcome.kind, reference.kind);
    ASSERT_EQ(outcome.expression.size(), reference.expression.size());
    EXPECT_EQ(outcome.expression.front().kind,
              reference.expression.front().kind);
    EXPECT_EQ(outcome.statistics.rewriteMatches,
              reference.statistics.rewriteMatches);
    EXPECT_EQ(outcome.statistics.eNodes, reference.statistics.eNodes);
  }
}

TEST(StructuredGraphEGraphTest, RustPanicIsContainedAtTheCABI) {
  FakeRelationService service = makeInverseAccessService();
  llvm::SmallVector<EGraphNode, 1> nodes;
  nodes.push_back(input(1, 1));
  EGraphOutcome outcome =
      runEGraph(nodes, /*rootNode=*/0, budget(), service.getABI(),
                /*forcePanicForTesting=*/true);
  EXPECT_EQ(outcome.kind, EGraphOutcomeKind::InternalError);
  EXPECT_TRUE(outcome.expression.empty());
}

TEST(StructuredGraphEGraphTest, RejectsMalformedRawABIRecords) {
  FakeRelationService service;
  WaferEGraphNode node{/*kind=*/99,
                       /*semanticId=*/1,
                       /*typeId=*/1,
                       /*childOffset=*/0,
                       /*childCount=*/0,
                       /*relationOffset=*/0,
                       /*relationCount=*/0,
                       /*dataInputCount=*/0,
                       /*axis=*/-1,
                       /*sourceOrder=*/0};
  WaferEGraphRequest request{/*schemaVersion=*/WAFER_EGRAPH_SCHEMA_VERSION,
                             /*flags=*/0,
                             /*nodes=*/&node,
                             /*nodeCount=*/1,
                             /*children=*/nullptr,
                             /*childCount=*/0,
                             /*relations=*/nullptr,
                             /*relationCount=*/0,
                             /*rootNode=*/0,
                             /*maximumIterations=*/2,
                             /*maximumENodes=*/16,
                             /*maximumMatches=*/16,
                             /*relationService=*/service.getABI()};
  WaferEGraphResult *result = waferRunStructuredEGraph(&request);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->status, WAFER_EGRAPH_INVALID_INPUT);
  waferFreeStructuredEGraphResult(result);

  node.kind = WAFER_EGRAPH_INPUT;
  node.semanticId = 0;
  result = waferRunStructuredEGraph(&request);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->status, WAFER_EGRAPH_INVALID_INPUT);
  waferFreeStructuredEGraphResult(result);
}

} // namespace
