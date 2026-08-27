//===- RegionDomainTest.cpp -------------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"

#include "TestSupport/Planning/SpatialDemandTestSupport.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

class RegionDomainTest : public ::testing::Test {
protected:
  RegionDomainTest() {
    registerWaferCoreDialects(registry);
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  static mlir::func::FuncOp function(mlir::ModuleOp module) {
    return *module.getOps<mlir::func::FuncOp>().begin();
  }

  static mlir::FailureOr<std::vector<analysis::RootRegionWork>>
  buildWorks(const StructuredDAGAnalysis &dag, std::string *failureReason) {
    llvm::SmallVector<StructuredDAGNodePlacement, 8> placements;
    for (const StructuredDAGNode &node : dag.getNodes()) {
      auto tiling = mlir::cast<mlir::TilingInterface>(node.operation);
      placements.push_back(StructuredDAGNodePlacement{
          node.id,
          llvm::SmallVector<uint32_t, 4>(tiling.getLoopIteratorTypes().size(),
                                         1),
          {TileId(0)}});
    }
    auto closed =
        wafer::test::buildTestSpatialDemand(dag, placements, failureReason);
    if (mlir::failed(closed))
      return mlir::failure();
    auto domain = RootWorkDomain::create(dag, closed->spatial, closed->demand,
                                         {TileId(0)}, failureReason);
    if (mlir::failed(domain))
      return mlir::failure();
    RootWorkCollectionOutcome outcome = collectRootWorks(*domain);
    RootWorkCollection *collection = getRootWorkCollection(outcome);
    if (!collection)
      return mlir::failure();
    return std::move(collection->works);
  }

  static std::optional<std::vector<RegionPlan>>
  enumerate(const RegionDomain &domain, size_t limit = 100000) {
    std::vector<RegionPlan> plans;
    RegionSuccessor current = domain.getFirstPlan();
    while (current.getKind() == RegionSuccessorKind::Plan) {
      if (!current.getPlan() || !current.getCursor() ||
          !domain.contains(*current.getPlan()) || plans.size() >= limit)
        return std::nullopt;
      plans.push_back(*current.getPlan());
      RegionCursor cursor = *current.getCursor();
      current = domain.getNextPlan(cursor);
    }
    if (current.getKind() != RegionSuccessorKind::End)
      return std::nullopt;
    return plans;
  }

  static bool selectedGroupsConnected(
      const StructuredDAGAnalysis &dag, llvm::ArrayRef<uint32_t> labels,
      llvm::ArrayRef<const StructuredDAGEdge *> internalEdges,
      llvm::ArrayRef<uint8_t> choices) {
    const uint32_t groupCount = *llvm::max_element(labels) + 1;
    for (uint32_t group = 0; group < groupCount; ++group) {
      llvm::SmallVector<size_t, 8> members;
      for (auto [node, label] : llvm::enumerate(labels))
        if (label == group)
          members.push_back(node);
      if (members.size() < 2)
        continue;
      std::set<size_t> reached{members.front()};
      llvm::SmallVector<size_t, 8> worklist{members.front()};
      while (!worklist.empty()) {
        size_t current = worklist.pop_back_val();
        for (auto [edge, choice] : llvm::zip_equal(internalEdges, choices)) {
          if (choice == 0)
            continue;
          size_t next = dag.getNodes().size();
          if (edge->producer == current)
            next = edge->consumer;
          else if (edge->consumer == current)
            next = edge->producer;
          if (!llvm::is_contained(members, next) ||
              !reached.insert(next).second)
            continue;
          worklist.push_back(next);
        }
      }
      if (reached.size() != members.size())
        return false;
    }
    return true;
  }

  static uint64_t
  countChoiceAssignments(const StructuredDAGAnalysis &dag,
                         llvm::ArrayRef<uint32_t> labels,
                         llvm::ArrayRef<const StructuredDAGEdge *> choiceEdges,
                         size_t edge, llvm::SmallVectorImpl<uint8_t> &choices) {
    if (edge == choiceEdges.size()) {
      std::map<StructuredDAGNodeID, StructuredDAGNodeID> directConsumer;
      std::set<StructuredDAGNodeID> storedProducer;
      for (auto [internal, choice] : llvm::zip_equal(choiceEdges, choices)) {
        if (choice == 1)
          storedProducer.insert(internal->producer);
        if (choice != 2)
          continue;
        auto [position, inserted] =
            directConsumer.try_emplace(internal->producer, internal->consumer);
        if ((!inserted && position->second != internal->consumer) ||
            storedProducer.count(internal->producer))
          return 0;
      }
      for (const auto &[producer, consumer] : directConsumer) {
        (void)consumer;
        if (storedProducer.count(producer))
          return 0;
      }
      return selectedGroupsConnected(dag, labels, choiceEdges, choices) ? 1 : 0;
    }
    uint64_t count = 0;
    const StructuredDAGEdge &current = *choiceEdges[edge];
    llvm::SmallVector<uint8_t, 5> allowed{0, 3, 4};
    if (labels[current.producer] == labels[current.consumer])
      allowed = {0, 1, 2, 3, 4};
    for (uint8_t choice : allowed) {
      choices.push_back(choice);
      count +=
          countChoiceAssignments(dag, labels, choiceEdges, edge + 1, choices);
      choices.pop_back();
    }
    return count;
  }

  static uint64_t countReferencePlans(const StructuredDAGAnalysis &dag,
                                      size_t node,
                                      llvm::SmallVectorImpl<uint32_t> &labels) {
    if (node != dag.getNodes().size()) {
      uint32_t maximum = 0;
      for (uint32_t label : labels)
        maximum = std::max(maximum, label);
      uint64_t count = 0;
      for (uint32_t label = 0; label <= maximum + (labels.empty() ? 0 : 1);
           ++label) {
        labels.push_back(label);
        count += countReferencePlans(dag, node + 1, labels);
        labels.pop_back();
      }
      return count;
    }
    llvm::SmallVector<const StructuredDAGEdge *, 8> choiceEdges;
    for (const StructuredDAGEdge &edge : dag.getEdges())
      choiceEdges.push_back(&edge);
    llvm::SmallVector<uint8_t, 8> choices;
    return countChoiceAssignments(dag, labels, choiceEdges, 0, choices);
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

constexpr llvm::StringLiteral kTwoNodeChain = R"mlir(
module {
  func.func @main(%input: tensor<1x2x1xf16>) -> tensor<1x2x1xf16> {
    %e0 = tensor.empty() : tensor<1x2x1xf16>
    %a = linalg.map ins(%input : tensor<1x2x1xf16>)
        outs(%e0 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<1x2x1xf16>
    %b = linalg.map ins(%a : tensor<1x2x1xf16>)
        outs(%e1 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    return %b : tensor<1x2x1xf16>
  }
}
)mlir";

TEST_F(RegionDomainTest, TwoNodeChainEnumeratesEveryCurrentUseForm) {
  auto module = parse(kTwoNodeChain);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto works = buildWorks(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
  auto domain = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  auto plans = enumerate(*domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 7u);
  std::vector<RegionPlan> proposals = domain->getProposals();
  EXPECT_GE(proposals.size(), 4u);
  EXPECT_EQ(std::set<RegionPlan>(proposals.begin(), proposals.end()).size(),
            proposals.size());
  for (const RegionPlan &proposal : proposals) {
    EXPECT_TRUE(domain->contains(proposal));
    EXPECT_TRUE(llvm::is_contained(*plans, proposal));
  }

  unsigned singletonExternal = 0;
  unsigned singletonReplica = 0;
  unsigned storedRequired = 0;
  unsigned directRequired = 0;
  unsigned storedReplica = 0;
  unsigned directReplica = 0;
  for (const RegionPlan &plan : *plans) {
    if (plan.groups.size() == 2) {
      auto localGroup =
          llvm::find_if(plan.groups, [](const RegionGroupPlan &group) {
            return !group.localBindings.empty();
          });
      if (localGroup == plan.groups.end()) {
        ++singletonExternal;
      } else {
        ++singletonReplica;
        EXPECT_EQ(localGroup->replicas.size(), 1u);
        EXPECT_TRUE(std::holds_alternative<ReplicaExecutionId>(
            localGroup->localBindings.front().producer));
      }
      continue;
    }
    ASSERT_EQ(plan.groups.size(), 1u);
    const RegionGroupPlan &group = plan.groups.front();
    ASSERT_EQ(group.localBindings.size(), 1u);
    EXPECT_TRUE(llvm::none_of(
        group.externalBindings, [](const ExternalUseBinding &binding) {
          return binding.fragment.source.kind ==
                 analysis::RootBoundaryKind::StructuredResult;
        }));
    const LocalUseBinding &binding = group.localBindings.front();
    const bool direct = binding.delivery == LocalUseDelivery::DirectNestedValue;
    if (std::holds_alternative<ExecutionInstanceId>(binding.producer)) {
      direct ? ++directRequired : ++storedRequired;
    } else {
      ASSERT_EQ(group.replicas.size(), 1u);
      direct ? ++directReplica : ++storedReplica;
    }
  }
  EXPECT_EQ(singletonExternal, 1u);
  EXPECT_EQ(singletonReplica, 2u);
  EXPECT_EQ(storedRequired, 1u);
  EXPECT_EQ(directRequired, 1u);
  EXPECT_EQ(storedReplica, 1u);
  EXPECT_EQ(directReplica, 1u);

  RegionPlan invalid = plans->front();
  invalid.groups.front().mandatoryRoots.push_back(
      invalid.groups.front().mandatoryRoots.front());
  EXPECT_FALSE(domain->contains(invalid));

  std::reverse(works->begin(), works->end());
  auto reordered = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(reordered)) << failureReason;
  auto reorderedPlans = enumerate(*reordered);
  ASSERT_TRUE(reorderedPlans);
  EXPECT_EQ(*reorderedPlans, *plans);
}

TEST_F(RegionDomainTest,
       CrossTileFragmentKeepsBoundaryAndBothExplicitReplicaSiblings) {
  auto module = parse(kTwoNodeChain);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  ASSERT_EQ(dag->getNodes().size(), 2u);
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements;
  for (const StructuredDAGNode &node : dag->getNodes()) {
    auto tiling = mlir::cast<mlir::TilingInterface>(node.operation);
    placements.push_back({node.id,
                          llvm::SmallVector<uint32_t, 4>(
                              tiling.getLoopIteratorTypes().size(), 1),
                          {TileId(node.id)}});
  }
  auto closed =
      wafer::test::buildTestSpatialDemand(*dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(closed)) << failureReason;
  auto rootDomain =
      RootWorkDomain::create(*dag, closed->spatial, closed->demand,
                             {TileId(0), TileId(1)}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(rootDomain)) << failureReason;
  RootWorkCollectionOutcome rootOutcome = collectRootWorks(*rootDomain);
  RootWorkCollection *works = getRootWorkCollection(rootOutcome);
  ASSERT_NE(works, nullptr);
  auto domain = RegionDomain::create(works->works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  auto plans = enumerate(*domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 3u);
  unsigned external = 0;
  unsigned storedReplica = 0;
  unsigned directReplica = 0;
  for (const RegionPlan &plan : *plans) {
    ASSERT_EQ(plan.groups.size(), 2u);
    auto consumer =
        llvm::find_if(plan.groups, [](const RegionGroupPlan &group) {
          return group.tile == TileId(1);
        });
    ASSERT_NE(consumer, plan.groups.end());
    if (consumer->replicas.empty()) {
      ++external;
      EXPECT_TRUE(consumer->localBindings.empty());
      continue;
    }
    ASSERT_EQ(consumer->replicas.size(), 1u);
    ASSERT_EQ(consumer->localBindings.size(), 1u);
    if (consumer->localBindings.front().delivery ==
        LocalUseDelivery::DirectNestedValue)
      ++directReplica;
    else
      ++storedReplica;
  }
  EXPECT_EQ(external, 1u);
  EXPECT_EQ(storedReplica, 1u);
  EXPECT_EQ(directReplica, 1u);
}

TEST_F(RegionDomainTest, FanoutIncludesSharedRequiredAndSplitReplicaPlans) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<1x2x1xf16>)
      -> (tensor<1x2x1xf16>, tensor<1x2x1xf16>) {
    %e0 = tensor.empty() : tensor<1x2x1xf16>
    %a = linalg.map ins(%input : tensor<1x2x1xf16>)
        outs(%e0 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<1x2x1xf16>
    %b = linalg.map ins(%a : tensor<1x2x1xf16>)
        outs(%e1 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<1x2x1xf16>
    %c = linalg.map ins(%a : tensor<1x2x1xf16>)
        outs(%e2 : tensor<1x2x1xf16>) (%v: f16) { linalg.yield %v : f16 }
    return %b, %c : tensor<1x2x1xf16>, tensor<1x2x1xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto works = buildWorks(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
  auto domain = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  auto plans = enumerate(*domain);
  ASSERT_TRUE(plans);
  bool foundSharedRequired = false;
  bool foundSplitReplicas = false;
  for (const RegionPlan &plan : *plans) {
    for (const RegionGroupPlan &group : plan.groups) {
      if (group.localBindings.size() != 2)
        continue;
      if (group.replicas.empty() &&
          llvm::all_of(group.localBindings, [](const LocalUseBinding &binding) {
            return binding.delivery == LocalUseDelivery::StoredRegionValue &&
                   std::holds_alternative<ExecutionInstanceId>(
                       binding.producer);
          }))
        foundSharedRequired = true;
      if (group.replicas.size() == 2 &&
          llvm::all_of(group.localBindings, [](const LocalUseBinding &binding) {
            return std::holds_alternative<ReplicaExecutionId>(binding.producer);
          }))
        foundSplitReplicas = true;
    }
  }
  EXPECT_TRUE(foundSharedRequired);
  EXPECT_TRUE(foundSplitReplicas);
}

TEST_F(RegionDomainTest,
       BoundedGraphPlansAreUniqueMembersOfTheIndependentRawSuperset) {
  constexpr llvm::StringLiteral chain = R"mlir(
module { func.func @main(%x: tensor<2xf16>) -> tensor<2xf16> {
  %e0 = tensor.empty() : tensor<2xf16>
  %a = linalg.map ins(%x : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e1 = tensor.empty() : tensor<2xf16>
  %b = linalg.map ins(%a : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e2 = tensor.empty() : tensor<2xf16>
  %c = linalg.map ins(%b : tensor<2xf16>) outs(%e2 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  return %c : tensor<2xf16> } }
)mlir";
  constexpr llvm::StringLiteral fanout = R"mlir(
module { func.func @main(%x: tensor<2xf16>)
    -> (tensor<2xf16>, tensor<2xf16>) {
  %e0 = tensor.empty() : tensor<2xf16>
  %a = linalg.map ins(%x : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e1 = tensor.empty() : tensor<2xf16>
  %b = linalg.map ins(%a : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e2 = tensor.empty() : tensor<2xf16>
  %c = linalg.map ins(%a : tensor<2xf16>) outs(%e2 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  return %b, %c : tensor<2xf16>, tensor<2xf16> } }
)mlir";
  constexpr llvm::StringLiteral fanin = R"mlir(
module { func.func @main(%x: tensor<2xf16>, %y: tensor<2xf16>)
    -> tensor<2xf16> {
  %e0 = tensor.empty() : tensor<2xf16>
  %a = linalg.map ins(%x : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e1 = tensor.empty() : tensor<2xf16>
  %b = linalg.map ins(%y : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e2 = tensor.empty() : tensor<2xf16>
  %c = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<2xf16>, tensor<2xf16>)
      outs(%e2 : tensor<2xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %old: f16):
      %sum = arith.addf %lhs, %rhs : f16
      linalg.yield %sum : f16
  } -> tensor<2xf16>
  return %c : tensor<2xf16> } }
)mlir";
  constexpr llvm::StringLiteral diamond = R"mlir(
module { func.func @main(%x: tensor<2xf16>) -> tensor<2xf16> {
  %e0 = tensor.empty() : tensor<2xf16>
  %a = linalg.map ins(%x : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e1 = tensor.empty() : tensor<2xf16>
  %b = linalg.map ins(%a : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e2 = tensor.empty() : tensor<2xf16>
  %c = linalg.map ins(%a : tensor<2xf16>) outs(%e2 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e3 = tensor.empty() : tensor<2xf16>
  %d = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%b, %c : tensor<2xf16>, tensor<2xf16>)
      outs(%e3 : tensor<2xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %old: f16):
      %sum = arith.addf %lhs, %rhs : f16
      linalg.yield %sum : f16
  } -> tensor<2xf16>
  return %d : tensor<2xf16> } }
)mlir";
  constexpr llvm::StringLiteral disconnected = R"mlir(
module { func.func @main(%x: tensor<2xf16>, %y: tensor<2xf16>)
    -> (tensor<2xf16>, tensor<2xf16>) {
  %e0 = tensor.empty() : tensor<2xf16>
  %a = linalg.map ins(%x : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  %e1 = tensor.empty() : tensor<2xf16>
  %b = linalg.map ins(%y : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
      (%v: f16) { linalg.yield %v : f16 }
  return %a, %b : tensor<2xf16>, tensor<2xf16> } }
)mlir";
  for (llvm::StringRef source : {chain, fanout, fanin, diamond, disconnected}) {
    auto module = parse(source);
    ASSERT_TRUE(module);
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto works = buildWorks(*dag, &failureReason);
    ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
    auto domain = RegionDomain::create(*works, &failureReason);
    ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
    auto plans = enumerate(*domain);
    ASSERT_TRUE(plans);
    llvm::SmallVector<uint32_t, 8> labels;
    const uint64_t independentSuperset =
        countReferencePlans(*dag, 0, labels);
    EXPECT_GT(plans->size(), 0u);
    EXPECT_LE(plans->size(), independentSuperset);
    EXPECT_EQ(std::set<RegionPlan>(plans->begin(), plans->end()).size(),
              plans->size());
    for (const RegionPlan &plan : *plans)
      EXPECT_TRUE(domain->contains(plan));
  }
}

TEST_F(RegionDomainTest, RealRaggedChainProducesStableSelectedPrefixes) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<2x2x1031x128xbf16>)
      -> tensor<2x2x1031x128xbf16> {
    %e0 = tensor.empty() : tensor<2x2x1031x128xbf16>
    %a = linalg.map ins(%input : tensor<2x2x1031x128xbf16>)
        outs(%e0 : tensor<2x2x1031x128xbf16>) (%v: bf16) {
      linalg.yield %v : bf16 }
    %e1 = tensor.empty() : tensor<2x2x1031x128xbf16>
    %b = linalg.map ins(%a : tensor<2x2x1031x128xbf16>)
        outs(%e1 : tensor<2x2x1031x128xbf16>) (%v: bf16) {
      linalg.yield %v : bf16 }
    return %b : tensor<2x2x1031x128xbf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  const std::string before = print(module->getOperation());
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto works = buildWorks(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
  auto domain = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  auto plans = enumerate(*domain);
  ASSERT_TRUE(plans);
  EXPECT_EQ(plans->size(), 7u);
  EXPECT_EQ(print(module->getOperation()), before);
}

} // namespace
