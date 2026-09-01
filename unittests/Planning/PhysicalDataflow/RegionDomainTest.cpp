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
#include <array>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

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

  static mlir::FailureOr<std::vector<analysis::RootRegionWork>>
  buildWorksOnTiles(const StructuredDAGAnalysis &dag,
                    llvm::ArrayRef<TileId> tiles, std::string *failureReason) {
    if (tiles.empty())
      return mlir::failure();
    llvm::SmallVector<StructuredDAGNodePlacement, 8> placements;
    for (const StructuredDAGNode &node : dag.getNodes()) {
      auto tiling = mlir::cast<mlir::TilingInterface>(node.operation);
      llvm::SmallVector<uint32_t, 4> factors(
          tiling.getLoopIteratorTypes().size(), 1);
      if (tiles.size() > 1) {
        if (factors.empty())
          return mlir::failure();
        factors.front() = tiles.size();
      }
      placements.push_back(StructuredDAGNodePlacement{
          node.id, std::move(factors),
          llvm::SmallVector<TileId, 4>(tiles.begin(), tiles.end())});
    }
    auto closed =
        wafer::test::buildTestSpatialDemand(dag, placements, failureReason);
    if (mlir::failed(closed))
      return mlir::failure();
    auto domain = RootWorkDomain::create(dag, closed->spatial, closed->demand,
                                         tiles, failureReason);
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
  ASSERT_EQ(plans->size(), 4u);
  std::vector<RegionPlan> proposals = domain->getProposals(4);
  EXPECT_GE(proposals.size(), 3u);
  EXPECT_EQ(std::set<RegionPlan>(proposals.begin(), proposals.end()).size(),
            proposals.size());
  for (const RegionPlan &proposal : proposals) {
    EXPECT_TRUE(domain->contains(proposal));
    EXPECT_TRUE(llvm::is_contained(*plans, proposal));
  }

  unsigned singletonExternal = 0;
  unsigned singletonReplica = 0;
  unsigned localRequired = 0;
  unsigned localReplica = 0;
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
    if (std::holds_alternative<ExecutionInstanceId>(binding.producer)) {
      ++localRequired;
    } else {
      ASSERT_EQ(group.replicas.size(), 1u);
      ++localReplica;
    }
  }
  EXPECT_EQ(singletonExternal, 1u);
  EXPECT_EQ(singletonReplica, 1u);
  EXPECT_EQ(localRequired, 1u);
  EXPECT_EQ(localReplica, 1u);

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
  ASSERT_EQ(plans->size(), 2u);
  unsigned external = 0;
  unsigned explicitReplica = 0;
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
    ++explicitReplica;
  }
  EXPECT_EQ(external, 1u);
  EXPECT_EQ(explicitReplica, 1u);
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
            return std::holds_alternative<ExecutionInstanceId>(
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
    const uint64_t independentSuperset = countReferencePlans(*dag, 0, labels);
    EXPECT_GT(plans->size(), 0u);
    EXPECT_LE(plans->size(), independentSuperset);
    EXPECT_EQ(std::set<RegionPlan>(plans->begin(), plans->end()).size(),
              plans->size());
    for (const RegionPlan &plan : *plans)
      EXPECT_TRUE(domain->contains(plan));

    const std::set<RegionPlan> rawPlans(plans->begin(), plans->end());
    std::vector<RegionPlan> proposals = domain->getProposals(4);
    EXPECT_GT(proposals.size(), 0u);
    EXPECT_LE(proposals.size(), 4u);
    EXPECT_EQ(std::set<RegionPlan>(proposals.begin(), proposals.end()).size(),
              proposals.size());
    for (const RegionPlan &proposal : proposals) {
      EXPECT_TRUE(domain->contains(proposal));
      EXPECT_EQ(rawPlans.count(proposal), 1u);
    }
    auto plansAfterProposalQuery = enumerate(*domain);
    ASSERT_TRUE(plansAfterProposalQuery);
    EXPECT_EQ(*plansAfterProposalQuery, *plans);
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
  EXPECT_EQ(plans->size(), 4u);
  EXPECT_EQ(print(module->getOperation()), before);
}

TEST_F(RegionDomainTest, RaggedChainProposalsSampleOneCoherentMergeSequence) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %e0 = tensor.empty() : tensor<2x1025x128xf16>
    %a = linalg.map ins(%input : tensor<2x1025x128xf16>)
        outs(%e0 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2x1025x128xf16>
    %b = linalg.map ins(%a : tensor<2x1025x128xf16>)
        outs(%e1 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2x1025x128xf16>
    %c = linalg.map ins(%b : tensor<2x1025x128xf16>)
        outs(%e2 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e3 = tensor.empty() : tensor<2x1025x128xf16>
    %d = linalg.map ins(%c : tensor<2x1025x128xf16>)
        outs(%e3 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    return %d : tensor<2x1025x128xf16>
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
  std::vector<RegionPlan> proposals = domain->getProposals(4);
  std::vector<uint64_t> fusionLevels;
  for (const RegionPlan &proposal : proposals) {
    ASSERT_TRUE(domain->contains(proposal));
    if (llvm::any_of(proposal.groups, [](const RegionGroupPlan &group) {
          return !group.replicas.empty();
        }))
      continue;
    uint64_t level = 0;
    for (const RegionGroupPlan &group : proposal.groups)
      level += group.mandatoryRoots.size() - 1;
    fusionLevels.push_back(level);
  }
  ASSERT_EQ(fusionLevels.size(), 4u);
  EXPECT_EQ(
      std::vector<uint64_t>(fusionLevels.begin(), fusionLevels.begin() + 4),
      (std::vector<uint64_t>{0, 1, 2, 3}));

  std::reverse(works->begin(), works->end());
  auto reordered = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(reordered)) << failureReason;
  EXPECT_EQ(reordered->getProposals(4), proposals);
}

TEST_F(RegionDomainTest,
       EqualSemanticMergesAdvanceAcrossTilesBeforeGrowingOneTileAgain) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %e0 = tensor.empty() : tensor<2x1025x128xf16>
    %a = linalg.map ins(%input : tensor<2x1025x128xf16>)
        outs(%e0 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2x1025x128xf16>
    %b = linalg.map ins(%a : tensor<2x1025x128xf16>)
        outs(%e1 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2x1025x128xf16>
    %c = linalg.map ins(%b : tensor<2x1025x128xf16>)
        outs(%e2 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e3 = tensor.empty() : tensor<2x1025x128xf16>
    %d = linalg.map ins(%c : tensor<2x1025x128xf16>)
        outs(%e3 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    return %d : tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  llvm::SmallVector<TileId, 2> tiles{TileId(0), TileId(1)};
  auto works = buildWorksOnTiles(*dag, tiles, &failureReason);
  ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
  auto domain = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  std::vector<RegionPlan> proposals = domain->getProposals(4);
  ASSERT_EQ(proposals.size(), 4u);

  std::set<int64_t> tilesWithFusedGroup;
  for (const RegionGroupPlan &group : proposals[1].groups)
    if (group.mandatoryRoots.size() > 1)
      tilesWithFusedGroup.insert(group.tile.getValue());
  EXPECT_EQ(tilesWithFusedGroup, (std::set<int64_t>{0, 1}));
  EXPECT_EQ(domain->getProposalMetrics(proposals[1]).fusionMerges, 2u);
  EXPECT_EQ(domain->getProposalMetrics(proposals[1]).maximumRootsPerRegion, 2u);
  EXPECT_EQ(domain->getProposalMetrics(proposals[2]).maximumRootsPerRegion, 2u);
  EXPECT_EQ(domain->getProposalMetrics(proposals[3]).maximumRootsPerRegion, 4u);
}

TEST_F(RegionDomainTest,
       FixedRegionCountRefinementEscapesTheGreedyCoarseningPath) {
  // This intentionally tiny graph is an independently bounded partition
  // oracle. Real-scale aligned/ragged coverage is provided by the neighboring
  // RegionDomain tests over the same proposal mechanism.
  std::string source;
  llvm::raw_string_ostream os(source);
  constexpr llvm::StringLiteral tensorType = "tensor<1x1x2xf16>";
  auto emitMap = [&](llvm::StringRef result, llvm::StringRef empty,
                     llvm::ArrayRef<llvm::StringRef> operands) {
    os << "    " << empty << " = tensor.empty() : " << tensorType << "\n";
    os << "    " << result << " = linalg.map ins(";
    llvm::interleaveComma(operands, os);
    os << " : ";
    llvm::interleaveComma(operands, os,
                          [&](llvm::StringRef) { os << tensorType; });
    os << ") outs(" << empty << " : " << tensorType << ") (";
    for (size_t index = 0; index < operands.size(); ++index) {
      if (index != 0)
        os << ", ";
      os << "%v" << index << ": f16";
    }
    os << ") {\n";
    std::string accumulated = "%v0";
    for (size_t index = 1; index < operands.size(); ++index) {
      os << "      %sum" << index << " = arith.addf " << accumulated << ", %v"
         << index << " : f16\n";
      accumulated = "%sum" + std::to_string(index);
    }
    if (operands.size() == 1) {
      os << "      linalg.yield %v0 : f16\n";
    } else {
      os << "      linalg.yield %sum" << operands.size() - 1 << " : f16\n";
    }
    os << "    }\n";
  };
  os << "module {\n"
        "  func.func @main(%x0: "
     << tensorType << ", %x1: " << tensorType << ") -> " << tensorType
     << " {\n";
  emitMap("%a", "%e0", {"%x0"});
  emitMap("%b", "%e1", {"%x1"});
  llvm::SmallVector<llvm::StringRef, 24> inputs;
  inputs.append(3, "%a");
  inputs.append(3, "%b");
  emitMap("%c", "%e2", inputs);
  inputs.clear();
  inputs.append(3, "%a");
  inputs.append(13, "%b");
  inputs.append(5, "%c");
  emitMap("%d", "%e3", inputs);
  inputs.clear();
  inputs.append(8, "%b");
  inputs.append(2, "%c");
  inputs.append(2, "%d");
  emitMap("%result", "%e4", inputs);
  os << "    return %result : " << tensorType << "\n  }\n}\n";
  os.flush();

  auto module = parse(source);
  ASSERT_TRUE(module) << source;
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto works = buildWorks(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
  auto domain = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  std::vector<RegionPlan> proposals = domain->getProposals(4);
  ASSERT_EQ(proposals.size(), 4u);
  RegionProposalMetrics refined = domain->getProposalMetrics(proposals[1]);
  EXPECT_EQ(refined.regions, 3u);
  EXPECT_EQ(refined.fusionMerges, 2u);
  // The unrefined maximum-gain matching seed internalizes 16 uses. The
  // fixed-count refinement crosses that local optimum without changing the
  // number of Regions.
  EXPECT_EQ(refined.localBindings, 21u);
  EXPECT_TRUE(refined.exactLogicalBytesKnown);
  EXPECT_EQ(refined.exactLogicalBytes, 84u);

  struct WeightedEdge {
    size_t producer;
    size_t consumer;
    uint64_t uses;
  };
  constexpr std::array<WeightedEdge, 8> edges{{
      {0, 2, 3},
      {1, 2, 3},
      {0, 3, 3},
      {1, 3, 13},
      {2, 3, 5},
      {1, 4, 8},
      {2, 4, 2},
      {3, 4, 2},
  }};
  auto isConnected = [&](llvm::ArrayRef<uint32_t> labels) {
    for (uint32_t group = 0; group < 3; ++group) {
      auto first = llvm::find(labels, group);
      if (first == labels.end())
        return false;
      std::array<uint8_t, 5> reached{};
      llvm::SmallVector<size_t, 5> worklist{
          static_cast<size_t>(std::distance(labels.begin(), first))};
      reached[worklist.front()] = 1;
      while (!worklist.empty()) {
        const size_t current = worklist.pop_back_val();
        for (const WeightedEdge &edge : edges) {
          size_t next = 5;
          if (edge.producer == current)
            next = edge.consumer;
          else if (edge.consumer == current)
            next = edge.producer;
          if (next == 5 || labels[next] != group || reached[next])
            continue;
          reached[next] = 1;
          worklist.push_back(next);
        }
      }
      for (size_t vertex = 0; vertex < labels.size(); ++vertex)
        if (labels[vertex] == group && !reached[vertex])
          return false;
    }
    return true;
  };
  auto isAcyclic = [&](llvm::ArrayRef<uint32_t> labels) {
    bool quotient[3][3]{};
    std::array<uint32_t, 3> indegree{};
    for (const WeightedEdge &edge : edges) {
      const uint32_t source = labels[edge.producer];
      const uint32_t destination = labels[edge.consumer];
      if (source == destination || quotient[source][destination])
        continue;
      quotient[source][destination] = true;
      ++indegree[destination];
    }
    llvm::SmallVector<uint32_t, 3> ready;
    for (uint32_t group = 0; group < 3; ++group)
      if (indegree[group] == 0)
        ready.push_back(group);
    uint32_t visited = 0;
    while (!ready.empty()) {
      const uint32_t source = ready.pop_back_val();
      ++visited;
      for (uint32_t destination = 0; destination < 3; ++destination)
        if (quotient[source][destination] && --indegree[destination] == 0)
          ready.push_back(destination);
    }
    return visited == 3;
  };
  uint64_t optimalLocalBindings = 0;
  llvm::SmallVector<uint32_t, 5> labels(5, 0);
  auto enumeratePartitions = [&](auto &&self, size_t vertex,
                                 uint32_t groupCount) -> void {
    if (vertex == labels.size()) {
      if (groupCount != 3 || !isConnected(labels) || !isAcyclic(labels))
        return;
      uint64_t score = 0;
      for (const WeightedEdge &edge : edges)
        if (labels[edge.producer] == labels[edge.consumer])
          score += edge.uses;
      optimalLocalBindings = std::max(optimalLocalBindings, score);
      return;
    }
    for (uint32_t group = 0; group <= groupCount && group < 3; ++group) {
      labels[vertex] = group;
      self(self, vertex + 1, std::max(groupCount, group + 1));
    }
  };
  enumeratePartitions(enumeratePartitions, 1, 1);

  std::array<size_t, edges.size()> edgeOrder{};
  std::iota(edgeOrder.begin(), edgeOrder.end(), 0);
  llvm::sort(edgeOrder, [&](size_t lhs, size_t rhs) {
    return std::tie(edges[lhs].uses, edges[lhs].producer, edges[lhs].consumer) >
           std::tie(edges[rhs].uses, edges[rhs].producer, edges[rhs].consumer);
  });
  std::array<uint8_t, 5> matched{};
  uint64_t greedyMatchingBindings = 0;
  for (size_t index : edgeOrder) {
    const WeightedEdge &edge = edges[index];
    if (matched[edge.producer] || matched[edge.consumer])
      continue;
    matched[edge.producer] = 1;
    matched[edge.consumer] = 1;
    greedyMatchingBindings += edge.uses;
  }
  EXPECT_EQ(greedyMatchingBindings, 16u);
  EXPECT_EQ(optimalLocalBindings, 21u);
  EXPECT_GT(refined.localBindings, greedyMatchingBindings);
  EXPECT_EQ(refined.localBindings, optimalLocalBindings);
  EXPECT_EQ(refined.exactLogicalBytes, optimalLocalBindings * 4);
}

TEST_F(RegionDomainTest,
       IncumbentRefinementReturnsOnlyDifferentRawDomainMembers) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %e0 = tensor.empty() : tensor<2x1025x128xf16>
    %a = linalg.map ins(%input : tensor<2x1025x128xf16>)
        outs(%e0 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2x1025x128xf16>
    %b = linalg.map ins(%a : tensor<2x1025x128xf16>)
        outs(%e1 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2x1025x128xf16>
    %c = linalg.map ins(%b : tensor<2x1025x128xf16>)
        outs(%e2 : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    return %c : tensor<2x1025x128xf16>
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
  std::vector<RegionPlan> initial = domain->getProposals(3);
  ASSERT_EQ(initial.size(), 3u);
  std::vector<RegionPlan> refinements =
      domain->getRefinementProposals(initial[1], 2);
  EXPECT_LE(refinements.size(), 2u);
  for (const RegionPlan &plan : refinements) {
    EXPECT_TRUE(domain->contains(plan));
    EXPECT_FALSE(plan == initial[1]);
  }
  EXPECT_TRUE(domain->getRefinementProposals(RegionPlan{}, 2).empty());

  std::reverse(works->begin(), works->end());
  auto reordered = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(reordered)) << failureReason;
  std::vector<RegionPlan> reorderedInitial = reordered->getProposals(3);
  ASSERT_EQ(reorderedInitial, initial);
  EXPECT_EQ(reordered->getRefinementProposals(reorderedInitial[1], 2),
            refinements);
}

TEST_F(RegionDomainTest, ExactLogicalPayloadOrdersTheCoherentMergeSequence) {
  auto module = parse(R"mlir(
module {
  func.func @main(%small: tensor<2x1024x16xf16>,
                  %large: tensor<2x1024x128xf16>)
      -> (tensor<2x1024x16xf16>, tensor<2x1024x128xf16>) {
    %small_e0 = tensor.empty() : tensor<2x1024x16xf16>
    %small_p = linalg.map ins(%small : tensor<2x1024x16xf16>)
        outs(%small_e0 : tensor<2x1024x16xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %small_e1 = tensor.empty() : tensor<2x1024x16xf16>
    %small_c = linalg.map ins(%small_p : tensor<2x1024x16xf16>)
        outs(%small_e1 : tensor<2x1024x16xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %large_e0 = tensor.empty() : tensor<2x1024x128xf16>
    %large_p = linalg.map ins(%large : tensor<2x1024x128xf16>)
        outs(%large_e0 : tensor<2x1024x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %large_e1 = tensor.empty() : tensor<2x1024x128xf16>
    %large_c = linalg.map ins(%large_p : tensor<2x1024x128xf16>)
        outs(%large_e1 : tensor<2x1024x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    return %small_c, %large_c
        : tensor<2x1024x16xf16>, tensor<2x1024x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto works = buildWorks(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
  std::map<analysis::RootRegionWorkId, int64_t> trailingExtent;
  for (const analysis::RootRegionWork &work : *works) {
    auto type = work.rootOperation
                    ? mlir::dyn_cast<mlir::RankedTensorType>(
                          work.rootOperation->getResult(0).getType())
                    : mlir::RankedTensorType{};
    ASSERT_TRUE(type);
    trailingExtent.emplace(work.id, type.getShape().back());
  }
  auto domain = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  std::vector<RegionPlan> proposals = domain->getProposals(3);
  ASSERT_EQ(proposals.size(), 3u);

  auto fusedGroups = [](const RegionPlan &plan) {
    std::vector<RegionGroupPlan> result;
    for (const RegionGroupPlan &group : plan.groups)
      if (group.mandatoryRoots.size() > 1)
        result.push_back(group);
    return result;
  };
  std::vector<RegionGroupPlan> firstFused = fusedGroups(proposals[1]);
  ASSERT_EQ(firstFused.size(), 1u);
  ASSERT_EQ(firstFused.front().mandatoryRoots.size(), 2u);
  EXPECT_TRUE(llvm::all_of(firstFused.front().mandatoryRoots,
                           [&](const analysis::RootRegionWorkId &work) {
                             return trailingExtent.at(work) == 128;
                           }));
  EXPECT_EQ(fusedGroups(proposals[2]).size(), 2u);

  auto allPlans = enumerate(*domain);
  ASSERT_TRUE(allPlans);
  for (const RegionPlan &proposal : proposals) {
    RegionProposalMetrics selected = domain->getProposalMetrics(proposal);
    ASSERT_TRUE(selected.exactLogicalBytesKnown);
    uint64_t bestExactBytes = 0;
    for (const RegionPlan &plan : *allPlans) {
      RegionProposalMetrics candidate = domain->getProposalMetrics(plan);
      if (candidate.fusionMerges == selected.fusionMerges &&
          candidate.exactLogicalBytesKnown)
        bestExactBytes = std::max(bestExactBytes, candidate.exactLogicalBytes);
    }
    EXPECT_EQ(selected.exactLogicalBytes, bestExactBytes);
  }
}

TEST_F(RegionDomainTest, FanoutGainCountsOnlyTheUseThatBecomesLocal) {
  auto module = parse(R"mlir(
module {
  func.func @main(%fanout: tensor<2x1025x128xf16>,
                  %independent: tensor<2x1025x192xf16>)
      -> (tensor<2x1025x128xf16>, tensor<2x1025x128xf16>,
          tensor<2x1025x192xf16>) {
    %fanout_e = tensor.empty() : tensor<2x1025x128xf16>
    %producer = linalg.map ins(%fanout : tensor<2x1025x128xf16>)
        outs(%fanout_e : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %left_e = tensor.empty() : tensor<2x1025x128xf16>
    %left = linalg.map ins(%producer : tensor<2x1025x128xf16>)
        outs(%left_e : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %right_e = tensor.empty() : tensor<2x1025x128xf16>
    %right = linalg.map ins(%producer : tensor<2x1025x128xf16>)
        outs(%right_e : tensor<2x1025x128xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %independent_e0 = tensor.empty() : tensor<2x1025x192xf16>
    %independent_p = linalg.map
        ins(%independent : tensor<2x1025x192xf16>)
        outs(%independent_e0 : tensor<2x1025x192xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    %independent_e1 = tensor.empty() : tensor<2x1025x192xf16>
    %independent_c = linalg.map
        ins(%independent_p : tensor<2x1025x192xf16>)
        outs(%independent_e1 : tensor<2x1025x192xf16>) (%v: f16) {
      linalg.yield %v : f16 }
    return %left, %right, %independent_c
        : tensor<2x1025x128xf16>, tensor<2x1025x128xf16>,
          tensor<2x1025x192xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto works = buildWorks(*dag, &failureReason);
  ASSERT_TRUE(mlir::succeeded(works)) << failureReason;
  std::map<analysis::RootRegionWorkId, int64_t> trailingExtent;
  for (const analysis::RootRegionWork &work : *works) {
    auto type = mlir::cast<mlir::RankedTensorType>(
        work.rootOperation->getResult(0).getType());
    trailingExtent.emplace(work.id, type.getShape().back());
  }
  auto domain = RegionDomain::create(*works, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  std::vector<RegionPlan> proposals = domain->getProposals(4);
  ASSERT_EQ(proposals.size(), 4u);
  auto firstFused =
      llvm::find_if(proposals[1].groups, [](const RegionGroupPlan &group) {
        return group.mandatoryRoots.size() > 1;
      });
  ASSERT_NE(firstFused, proposals[1].groups.end());
  EXPECT_TRUE(llvm::all_of(firstFused->mandatoryRoots,
                           [&](const analysis::RootRegionWorkId &work) {
                             return trailingExtent.at(work) == 192;
                           }));
  auto allPlans = enumerate(*domain);
  ASSERT_TRUE(allPlans);
  RegionProposalMetrics selected = domain->getProposalMetrics(proposals[1]);
  ASSERT_TRUE(selected.exactLogicalBytesKnown);
  uint64_t bestExactBytes = 0;
  for (const RegionPlan &plan : *allPlans) {
    RegionProposalMetrics candidate = domain->getProposalMetrics(plan);
    if (candidate.fusionMerges == selected.fusionMerges &&
        candidate.exactLogicalBytesKnown)
      bestExactBytes = std::max(bestExactBytes, candidate.exactLogicalBytes);
  }
  EXPECT_EQ(selected.exactLogicalBytes, bestExactBytes);
}

} // namespace
