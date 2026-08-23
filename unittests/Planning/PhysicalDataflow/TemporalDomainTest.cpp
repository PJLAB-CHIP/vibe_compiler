//===- TemporalDomainTest.cpp ----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

TraversalScopeId makeScope(uint32_t rootIndex, int64_t tile,
                           uint32_t piece = 0) {
  SemanticRootKey root;
  root.anchorIndex = rootIndex;
  RootRegionWorkId work{root, TileId(tile)};
  LogicalShardId shard{root, {static_cast<uint32_t>(tile)}};
  TraversalScopeId result;
  result.execution = ExecutionInstanceId{RequiredRootExecution{work, shard}};
  result.invocation = TopLevelWorkPieceId{piece};
  return result;
}

TemporalScopeDescriptor makeDescriptor(TraversalScopeId id,
                                       llvm::ArrayRef<int64_t> offsets,
                                       llvm::ArrayRef<int64_t> extents) {
  TemporalScopeDescriptor result;
  result.id = std::move(id);
  result.iterationOffsets.assign(offsets.begin(), offsets.end());
  result.iterationExtents.assign(extents.begin(), extents.end());
  result.iteratorCapabilities.assign(extents.size(),
                                     IteratorTilingCapability::Tileable);
  return result;
}

ExactIndexSet makeBoxSet(llvm::ArrayRef<int64_t> offsets,
                         llvm::ArrayRef<int64_t> sizes) {
  IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(set.isExact()) << set.reason;
  StaticRectangularIndexSet box;
  box.offsets.assign(offsets.begin(), offsets.end());
  box.sizes.assign(sizes.begin(), sizes.end());
  return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion, {box});
}

std::optional<std::vector<TemporalPlan>> enumerate(const TemporalDomain &domain,
                                                   size_t limit = 200000) {
  std::vector<TemporalPlan> plans;
  TemporalSuccessor next = domain.getFirstPlan();
  while (next.getKind() == TemporalSuccessorKind::Plan) {
    if (!next.getPlan() || !next.getCursor() || plans.size() >= limit ||
        !domain.contains(*next.getPlan()))
      return std::nullopt;
    plans.push_back(*next.getPlan());
    TemporalCursor cursor = *next.getCursor();
    next = domain.getNextPlan(cursor);
  }
  if (next.getKind() != TemporalSuccessorKind::End)
    return std::nullopt;
  return plans;
}

class TemporalDomainTest : public ::testing::Test {
protected:
  TemporalDomainTest() {
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

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(TemporalDomainTest,
       EveryPositiveSizeAndActiveOrderMatchesIndependentOracle) {
  TemporalScopeDescriptor descriptor =
      makeDescriptor(makeScope(0, 0), {17, 29}, {2, 3});
  TemporalDomainResult result = buildTemporalDomain({descriptor});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);

  std::set<TemporalScopePlan> actual;
  for (const TemporalPlan &plan : *plans) {
    ASSERT_EQ(plan.scopes.size(), 1u);
    actual.insert(plan.scopes.front());
  }
  std::set<TemporalScopePlan> expected;
  for (int64_t first = 1; first <= 2; ++first)
    for (int64_t second = 1; second <= 3; ++second) {
      llvm::SmallVector<uint32_t, 4> active;
      if (first < 2)
        active.push_back(0);
      if (second < 3)
        active.push_back(1);
      do {
        expected.insert({descriptor.id, {first, second}, active});
      } while (std::next_permutation(active.begin(), active.end()));
    }
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(actual.size(), 8u);
  EXPECT_TRUE(actual.count({descriptor.id, {2, 3}, {}}));
  EXPECT_TRUE(actual.count({descriptor.id, {1, 2}, {0, 1}}));
  EXPECT_TRUE(actual.count({descriptor.id, {1, 2}, {1, 0}}));
}

TEST_F(TemporalDomainTest,
       IntervalProposalAndMidpointChildrenPreserveEveryLargeExtentPoint) {
  std::string failureReason;
  auto proposed = splitTemporalSizeInterval({1, 1025}, 128, &failureReason);
  ASSERT_TRUE(mlir::succeeded(proposed)) << failureReason;
  EXPECT_EQ(proposed->singleton, (TemporalSizeInterval{128, 128}));
  EXPECT_EQ(proposed->above, (TemporalSizeInterval{129, 1025}));
  EXPECT_EQ(proposed->below, (TemporalSizeInterval{1, 127}));

  std::vector<TemporalSizeInterval> pending;
  pending.push_back(proposed->singleton);
  pending.push_back(*proposed->above);
  pending.push_back(*proposed->below);
  std::set<int64_t> leaves;
  while (!pending.empty()) {
    TemporalSizeInterval interval = pending.back();
    pending.pop_back();
    if (interval.lower == interval.upper) {
      EXPECT_TRUE(leaves.insert(interval.lower).second);
      continue;
    }
    auto split =
        splitTemporalSizeInterval(interval, std::nullopt, &failureReason);
    ASSERT_TRUE(mlir::succeeded(split)) << failureReason;
    pending.push_back(split->singleton);
    if (split->above)
      pending.push_back(*split->above);
    if (split->below)
      pending.push_back(*split->below);
  }
  ASSERT_EQ(leaves.size(), 1025u);
  EXPECT_EQ(*leaves.begin(), 1);
  EXPECT_EQ(*leaves.rbegin(), 1025);
  EXPECT_TRUE(
      mlir::failed(splitTemporalSizeInterval({1, 1025}, 0, &failureReason)));
}

TEST_F(TemporalDomainTest,
       KahnOrdersMatchPermutationFilterForPrecedenceDiamond) {
  TemporalScopeDescriptor descriptor =
      makeDescriptor(makeScope(0, 0), {0, 0, 0, 0}, {2, 2, 2, 2});
  descriptor.precedence = {{0, 2}, {1, 2}, {1, 3}};
  TemporalDomainResult result = buildTemporalDomain({descriptor});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);

  std::set<llvm::SmallVector<uint32_t, 4>> actual;
  for (const TemporalPlan &plan : *plans)
    if (plan.scopes.front().iteratorTileSizes ==
        llvm::SmallVector<int64_t, 4>({1, 1, 1, 1}))
      actual.insert(plan.scopes.front().waveLoopOrder);

  std::set<llvm::SmallVector<uint32_t, 4>> expected;
  llvm::SmallVector<uint32_t, 4> permutation{0, 1, 2, 3};
  do {
    auto position = [&](uint32_t value) {
      return std::distance(permutation.begin(), llvm::find(permutation, value));
    };
    if (position(0) < position(2) && position(1) < position(2) &&
        position(1) < position(3))
      expected.insert(permutation);
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  EXPECT_EQ(actual, expected);
}

TEST_F(TemporalDomainTest,
       PerTileRemaindersRemainIndependentCartesianVariables) {
  TemporalScopeDescriptor first = makeDescriptor(makeScope(0, 0), {0}, {3});
  TemporalScopeDescriptor second = makeDescriptor(makeScope(0, 1), {3}, {2});
  TemporalDomainResult result = buildTemporalDomain({second, first});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), 6u);
  std::set<std::pair<int64_t, int64_t>> sizes;
  for (const TemporalPlan &plan : *plans) {
    ASSERT_EQ(plan.scopes.size(), 2u);
    sizes.emplace(plan.scopes[0].iteratorTileSizes.front(),
                  plan.scopes[1].iteratorTileSizes.front());
  }
  EXPECT_EQ(sizes.size(), 6u);
  EXPECT_TRUE(sizes.count({3, 2}));
  EXPECT_TRUE(sizes.count({1, 1}));
}

TEST_F(TemporalDomainTest,
       FullExtentRankZeroAndNestedClassUseOneCurrentSchema) {
  TemporalScopeDescriptor parent =
      makeDescriptor(makeScope(0, 0), {0, 0, 0}, {2, 1025, 128});
  parent.iteratorCapabilities[0] = IteratorTilingCapability::FullExtentOnly;

  TemporalScopeDescriptor scalar = makeDescriptor(makeScope(1, 1), {}, {});

  DemandFragmentId relation;
  relation.source.semantic.anchorIndex = 0;
  relation.source.kind = RootBoundaryKind::StructuredResult;
  relation.use.destinationShard =
      std::get<RequiredRootExecution>(
          std::get<ExecutionInstanceId>(parent.id.execution).source)
          .shard;
  NestedInvocationClassId invocation;
  invocation.parent = parent.id.execution;
  invocation.uses.push_back({relation, {0, 1024, 0}, {2, 1, 128}});
  invocation.producerOffsets = {0, 1024, 0};
  invocation.producerExtents = {2, 1, 128};
  TemporalScopeDescriptor child =
      makeDescriptor(makeScope(2, 0), {0, 1024, 0}, {2, 1, 128});
  child.id.invocation = invocation;
  child.parentScope = parent.id;

  TemporalDomainResult result = buildTemporalDomain({child, scalar, parent});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  TemporalSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(first.getPlan(), nullptr);
  ASSERT_EQ(first.getPlan()->scopes.size(), 3u);
  EXPECT_EQ(first.getPlan()->scopes[0].id, parent.id);
  EXPECT_EQ(first.getPlan()->scopes[1].id, scalar.id);
  EXPECT_EQ(first.getPlan()->scopes[2].id, child.id);

  TemporalPlan member = *first.getPlan();
  member.scopes[0].iteratorTileSizes = {2, 1031, 64};
  EXPECT_FALSE(result.domain->contains(member));
  member.scopes[0].iteratorTileSizes = {2, 1025, 64};
  member.scopes[0].waveLoopOrder = {2};
  EXPECT_TRUE(result.domain->contains(member));
  EXPECT_TRUE(member.scopes[1].iteratorTileSizes.empty());
}

TEST_F(TemporalDomainTest,
       RankSixCoupledStateKeepsEveryIteratorInOneExplicitVector) {
  TemporalScopeDescriptor descriptor = makeDescriptor(
      makeScope(0, 0), {0, 0, 0, 0, 0, 0}, {2, 4, 1024, 64, 1031, 128});
  TemporalDomainResult result = buildTemporalDomain({descriptor});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  TemporalSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(first.getPlan(), nullptr);
  ASSERT_EQ(first.getPlan()->scopes.size(), 1u);
  EXPECT_EQ(first.getPlan()->scopes.front().iteratorTileSizes,
            descriptor.iterationExtents);
  TemporalPlan member = *first.getPlan();
  member.scopes.front().iteratorTileSizes = {2, 4, 1024, 64, 128, 128};
  member.scopes.front().waveLoopOrder = {4};
  EXPECT_TRUE(result.domain->contains(member));
}

TEST_F(TemporalDomainTest, ExactAlignedAndRaggedWavesCoverNonzeroIntervals) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::string failureReason;
    auto waves = buildTemporalAxisWaves({37, extent}, 128, &failureReason);
    ASSERT_TRUE(mlir::succeeded(waves)) << failureReason;
    ASSERT_FALSE(waves->empty());
    int64_t cursor = 37;
    int64_t total = 0;
    for (const IteratorInterval &wave : *waves) {
      EXPECT_EQ(wave.offset, cursor);
      EXPECT_GT(wave.size, 0);
      EXPECT_LE(wave.size, 128);
      cursor += wave.size;
      total += wave.size;
    }
    EXPECT_EQ(total, extent);
    EXPECT_EQ(cursor, 37 + extent);
    EXPECT_EQ(waves->back().size, extent % 128 == 0 ? 128 : extent % 128);
  }
}

TEST_F(TemporalDomainTest,
       ProductionDerivationKeepsRequiredAndReplicaIdentitiesDistinct) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<2x1025x128xbf16>)
      -> tensor<2x1025x128xbf16> {
    %empty = tensor.empty() : tensor<2x1025x128xbf16>
    %result = linalg.map ins(%input : tensor<2x1025x128xbf16>)
        outs(%empty : tensor<2x1025x128xbf16>) (%value: bf16) {
      linalg.yield %value : bf16
    }
    return %result : tensor<2x1025x128xbf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto operation =
      llvm::find_if(function.getBody().front(), [](mlir::Operation &candidate) {
        return mlir::isa<mlir::linalg::MapOp>(candidate);
      });
  ASSERT_NE(operation, function.getBody().front().end());

  SemanticRootKey root;
  RootRegionWork work;
  work.id = {root, TileId(0)};
  work.rootOperation = &*operation;
  LogicalShardId shard{root, {0, 0, 0}};
  work.execution.push_back({shard, {{0, 2}, {0, 1025}, {0, 128}}});
  RequiredRootExecution required{work.id, shard};

  DemandFragmentId fragment;
  fragment.source.kind = RootBoundaryKind::StructuredResult;
  fragment.source.semantic = root;
  fragment.ownerShard = shard;
  fragment.ownerTile = TileId(0);
  fragment.use.destinationShard = shard;
  ReplicaExecutionId replica{required, fragment};

  RegionGroupPlan first;
  first.tile = TileId(0);
  first.mandatoryRoots.push_back(work.id);
  first.executions.push_back({ExecutionInstanceId{required}});
  RegionGroupPlan second;
  second.tile = TileId(1);
  second.replicas.push_back({replica});
  RegionPlan regions{{first, second}};

  TemporalDomainResult result = buildTemporalDomain(regions, {work});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  TemporalSuccessor firstPlan = result.domain->getFirstPlan();
  ASSERT_EQ(firstPlan.getKind(), TemporalSuccessorKind::Plan);
  ASSERT_NE(firstPlan.getPlan(), nullptr);
  ASSERT_EQ(firstPlan.getPlan()->scopes.size(), 2u);
  EXPECT_TRUE(std::holds_alternative<ExecutionInstanceId>(
      firstPlan.getPlan()->scopes[0].id.execution));
  EXPECT_TRUE(std::holds_alternative<ReplicaExecutionId>(
      firstPlan.getPlan()->scopes[1].id.execution));
  EXPECT_EQ(firstPlan.getPlan()->scopes[0].iteratorTileSizes,
            (llvm::SmallVector<int64_t, 4>{2, 1025, 128}));
  EXPECT_EQ(firstPlan.getPlan()->scopes[1].iteratorTileSizes,
            (llvm::SmallVector<int64_t, 4>{2, 1025, 128}));
}

TEST_F(TemporalDomainTest,
       ParentChoiceDerivesExactNestedMainTailAndHaloClasses) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<1x1027x130x1xbf16>,
                  %filter: tensor<3x3x1x1xbf16>)
      -> tensor<1x1025x128x1xbf16> {
    %e0 = tensor.empty() : tensor<1x1027x130x1xbf16>
    %producer = linalg.map ins(%input : tensor<1x1027x130x1xbf16>)
        outs(%e0 : tensor<1x1027x130x1xbf16>) (%value: bf16) {
      linalg.yield %value : bf16
    }
    %e1 = tensor.empty() : tensor<1x1025x128x1xbf16>
    %zero = arith.constant 0.0 : bf16
    %init = linalg.fill ins(%zero : bf16)
        outs(%e1 : tensor<1x1025x128x1xbf16>)
        -> tensor<1x1025x128x1xbf16>
    %consumer = linalg.conv_2d_nhwc_hwcf
        {dilations = dense<1> : tensor<2xi64>,
         strides = dense<1> : tensor<2xi64>}
        ins(%producer, %filter
            : tensor<1x1027x130x1xbf16>, tensor<3x3x1x1xbf16>)
        outs(%init : tensor<1x1025x128x1xbf16>)
        -> tensor<1x1025x128x1xbf16>
    return %consumer : tensor<1x1025x128x1xbf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  mlir::Operation *producerOperation = nullptr;
  mlir::Operation *consumerOperation = nullptr;
  function.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::linalg::MapOp>(operation))
      producerOperation = operation;
    if (operation->getName().getStringRef() == "linalg.conv_2d_nhwc_hwcf")
      consumerOperation = operation;
  });
  ASSERT_NE(producerOperation, nullptr);
  ASSERT_NE(consumerOperation, nullptr);
  auto consumerLinalg = mlir::cast<mlir::linalg::LinalgOp>(consumerOperation);
  llvm::SmallVector<int64_t, 8> consumerRanges =
      consumerLinalg.getStaticLoopRanges();
  ASSERT_FALSE(consumerRanges.empty());

  SemanticRootKey producerRoot;
  producerRoot.anchorIndex = 0;
  SemanticRootKey consumerRoot;
  consumerRoot.anchorIndex = 1;
  RootRegionWork producer;
  producer.id = {producerRoot, TileId(0)};
  producer.rootOperation = producerOperation;
  LogicalShardId producerShard{producerRoot, {0, 0, 0, 0}};
  producer.execution.push_back(
      {producerShard, {{0, 1}, {0, 1027}, {0, 130}, {0, 1}}});
  RootRegionWork consumer;
  consumer.id = {consumerRoot, TileId(0)};
  consumer.rootOperation = consumerOperation;
  LogicalShardId consumerShard{
      consumerRoot, llvm::SmallVector<uint32_t, 4>(consumerRanges.size(), 0)};
  RootExecutionWork consumerWork;
  consumerWork.shard = consumerShard;
  for (int64_t extent : consumerRanges)
    consumerWork.iterationDomain.push_back({0, extent});
  consumer.execution.push_back(std::move(consumerWork));

  RootBoundaryId boundaryId;
  boundaryId.kind = RootBoundaryKind::StructuredResult;
  boundaryId.semantic = producerRoot;
  boundaryId.index = 0;
  RootUseId useId{0, consumerShard};
  RootBoundaryUseWork use;
  use.id = useId;
  use.requiredDomain = makeBoxSet({0, 0, 0, 0}, {1, 1027, 130, 1});
  OwnerIntersection owner;
  owner.ownerShard = producerShard;
  owner.tile = TileId(0);
  owner.domain = makeBoxSet({0, 0, 0, 0}, {1, 1027, 130, 1});
  use.eligibleFinalOwners.push_back(owner);
  RootBoundaryWork boundary;
  boundary.id = boundaryId;
  boundary.sourceValue = producerOperation->getResult(0);
  boundary.requiredDomain = makeBoxSet({0, 0, 0, 0}, {1, 1027, 130, 1});
  boundary.consumerUses.push_back(use);
  consumer.boundaries.push_back(boundary);

  ExecutionInstanceId producerExecution{
      RequiredRootExecution{producer.id, producerShard}};
  ExecutionInstanceId consumerExecution{
      RequiredRootExecution{consumer.id, consumerShard}};
  ExecutionInstancePlan producerPlan;
  producerPlan.id = producerExecution;
  producerPlan.placement =
      ExecutionInstancePlan::NestedUnder{consumerExecution.source};
  ExecutionInstancePlan consumerPlan;
  consumerPlan.id = consumerExecution;
  RegionGroupPlan group;
  group.tile = TileId(0);
  group.mandatoryRoots = {producer.id, consumer.id};
  group.executions = {producerPlan, consumerPlan};
  DemandFragmentId fragment;
  fragment.source = boundaryId;
  fragment.use = useId;
  fragment.ownerShard = producerShard;
  fragment.ownerTile = TileId(0);
  group.localBindings.push_back({fragment, RegionExecutionId(producerExecution),
                                 LocalUseDelivery::DirectNestedValue});

  TemporalDomainResult result =
      buildTemporalDomain(RegionPlan{{group}}, {producer, consumer});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  TemporalSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan)
      << first.getDetail().str();
  ASSERT_NE(first.getPlan(), nullptr);
  ASSERT_EQ(first.getPlan()->scopes.size(), 2u);
  EXPECT_TRUE(isTopLevelScope(first.getPlan()->scopes.front().id));
  EXPECT_FALSE(isTopLevelScope(first.getPlan()->scopes.back().id));
  ASSERT_NE(first.getCursor(), nullptr);
  TemporalSuccessor next = result.domain->getNextPlan(*first.getCursor());
  ASSERT_EQ(next.getKind(), TemporalSuccessorKind::Plan)
      << next.getDetail().str();
  ASSERT_NE(next.getPlan(), nullptr);
  EXPECT_TRUE(result.domain->contains(*next.getPlan()));

  auto height = llvm::find(consumerRanges, int64_t{1025});
  ASSERT_NE(height, consumerRanges.end());
  const uint32_t heightAxis =
      static_cast<uint32_t>(std::distance(consumerRanges.begin(), height));
  TemporalPlan prefix;
  prefix.scopes.push_back(first.getPlan()->scopes.front());
  prefix.scopes.front().iteratorTileSizes[heightAxis] = 128;
  prefix.scopes.front().waveLoopOrder = {heightAxis};
  TemporalSuccessor completed = result.domain->completePrefix(prefix);
  ASSERT_EQ(completed.getKind(), TemporalSuccessorKind::Plan)
      << completed.getDetail().str();
  ASSERT_NE(completed.getPlan(), nullptr);
  ASSERT_EQ(completed.getPlan()->scopes.size(), 3u);
  unsigned main = 0;
  unsigned tail = 0;
  for (const TemporalScopePlan &scope :
       llvm::drop_begin(completed.getPlan()->scopes)) {
    const auto *invocation =
        std::get_if<NestedInvocationClassId>(&scope.id.invocation);
    ASSERT_NE(invocation, nullptr);
    ASSERT_EQ(invocation->producerExtents.size(), 4u);
    if (invocation->producerExtents[1] == 130) {
      ASSERT_EQ(invocation->uses.size(), 1u);
      EXPECT_EQ(invocation->uses.front().requestedExtents[1], 130);
      EXPECT_EQ(invocation->producerExtents[2], 130);
      ++main;
    } else {
      EXPECT_EQ(invocation->producerExtents[1], 3);
      EXPECT_EQ(invocation->producerOffsets[1], 0);
      ++tail;
    }
  }
  EXPECT_EQ(main, 1u);
  EXPECT_EQ(tail, 1u);

  prefix.scopes.front().iteratorTileSizes[heightAxis] = 256;
  TemporalSuccessor changed = result.domain->completePrefix(prefix);
  ASSERT_EQ(changed.getKind(), TemporalSuccessorKind::Plan)
      << changed.getDetail().str();
  ASSERT_NE(changed.getPlan(), nullptr);
  EXPECT_EQ(changed.getPlan()->scopes.size(), 3u);
  EXPECT_FALSE(*changed.getPlan() == *completed.getPlan());

  prefix.scopes.front().iteratorTileSizes.assign(consumerRanges.size(), 1);
  prefix.scopes.front().waveLoopOrder.clear();
  for (auto [axis, extent] : llvm::enumerate(consumerRanges))
    if (extent > 1)
      prefix.scopes.front().waveLoopOrder.push_back(
          static_cast<uint32_t>(axis));
  TemporalSuccessor limited = result.domain->completePrefix(prefix);
  EXPECT_EQ(limited.getKind(), TemporalSuccessorKind::Indeterminate);
  EXPECT_NE(limited.getDetail().find("work limit"), llvm::StringRef::npos);
  TemporalSuccessor repeated = result.domain->completePrefix(prefix);
  EXPECT_EQ(repeated.getKind(), TemporalSuccessorKind::Indeterminate);
  EXPECT_EQ(repeated.getDetail(), limited.getDetail());
}

TEST_F(TemporalDomainTest,
       SharedNestedExecutionMergesCompatibleRegionUsesIntoOneScope) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @main(%input: tensor<2x1025x128xbf16>)
      -> tensor<2x1025x128xbf16> {
    %e0 = tensor.empty() : tensor<2x1025x128xbf16>
    %producer = linalg.map ins(%input : tensor<2x1025x128xbf16>)
        outs(%e0 : tensor<2x1025x128xbf16>) (%value: bf16) {
      linalg.yield %value : bf16
    }
    %e1 = tensor.empty() : tensor<2x1025x128xbf16>
    %consumer = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer, %producer
            : tensor<2x1025x128xbf16>, tensor<2x1025x128xbf16>)
        outs(%e1 : tensor<2x1025x128xbf16>) {
      ^bb0(%lhs: bf16, %rhs: bf16, %old: bf16):
        linalg.yield %lhs : bf16
    } -> tensor<2x1025x128xbf16>
    return %consumer : tensor<2x1025x128xbf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  mlir::Operation *producerOperation = nullptr;
  mlir::Operation *consumerOperation = nullptr;
  function.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::linalg::MapOp>(operation))
      producerOperation = operation;
    if (mlir::isa<mlir::linalg::GenericOp>(operation))
      consumerOperation = operation;
  });
  ASSERT_NE(producerOperation, nullptr);
  ASSERT_NE(consumerOperation, nullptr);

  SemanticRootKey producerRoot;
  SemanticRootKey consumerRoot;
  consumerRoot.anchorIndex = 1;
  RootRegionWork producer;
  producer.id = {producerRoot, TileId(0)};
  producer.rootOperation = producerOperation;
  LogicalShardId producerShard{producerRoot, {0, 0, 0}};
  producer.execution.push_back({producerShard, {{0, 2}, {0, 1025}, {0, 128}}});
  RootRegionWork consumer;
  consumer.id = {consumerRoot, TileId(0)};
  consumer.rootOperation = consumerOperation;
  LogicalShardId consumerShard{consumerRoot, {0, 0, 0}};
  consumer.execution.push_back({consumerShard, {{0, 2}, {0, 1025}, {0, 128}}});

  RootBoundaryId boundaryId;
  boundaryId.kind = RootBoundaryKind::StructuredResult;
  boundaryId.semantic = producerRoot;
  RootBoundaryWork boundary;
  boundary.id = boundaryId;
  boundary.sourceValue = producerOperation->getResult(0);
  boundary.requiredDomain = makeBoxSet({0, 0, 0}, {2, 1025, 128});
  for (uint32_t operand : {0u, 1u}) {
    RootBoundaryUseWork use;
    use.id = {operand, consumerShard};
    use.requiredDomain = makeBoxSet({0, 0, 0}, {2, 1025, 128});
    OwnerIntersection owner;
    owner.ownerShard = producerShard;
    owner.tile = TileId(0);
    owner.domain = makeBoxSet({0, 0, 0}, {2, 1025, 128});
    use.eligibleFinalOwners.push_back(std::move(owner));
    boundary.consumerUses.push_back(std::move(use));
  }
  consumer.boundaries.push_back(std::move(boundary));

  ExecutionInstanceId producerExecution{
      RequiredRootExecution{producer.id, producerShard}};
  ExecutionInstanceId consumerExecution{
      RequiredRootExecution{consumer.id, consumerShard}};
  ExecutionInstancePlan producerPlan;
  producerPlan.id = producerExecution;
  producerPlan.placement =
      ExecutionInstancePlan::NestedUnder{consumerExecution.source};
  RegionGroupPlan group;
  group.tile = TileId(0);
  group.mandatoryRoots = {producer.id, consumer.id};
  group.executions = {producerPlan, ExecutionInstancePlan{consumerExecution}};
  for (uint32_t operand : {0u, 1u}) {
    DemandFragmentId fragment;
    fragment.source = boundaryId;
    fragment.use = {operand, consumerShard};
    fragment.ownerShard = producerShard;
    fragment.ownerTile = TileId(0);
    group.localBindings.push_back({fragment,
                                   RegionExecutionId(producerExecution),
                                   LocalUseDelivery::DirectNestedValue});
  }

  TemporalDomainResult result =
      buildTemporalDomain(RegionPlan{{group}}, {producer, consumer});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  TemporalSuccessor first = result.domain->getFirstPlan();
  ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Plan)
      << first.getDetail().str();
  ASSERT_NE(first.getPlan(), nullptr);
  ASSERT_EQ(first.getPlan()->scopes.size(), 2u);
  const auto *invocation = std::get_if<NestedInvocationClassId>(
      &first.getPlan()->scopes.back().id.invocation);
  ASSERT_NE(invocation, nullptr);
  ASSERT_EQ(invocation->uses.size(), 2u);
  EXPECT_EQ(invocation->uses[0].relation.use.operand, 0u);
  EXPECT_EQ(invocation->uses[1].relation.use.operand, 1u);
}

TEST_F(TemporalDomainTest,
       CyclesMalformedRanksAndMissingNestedRelationsFailTyped) {
  TemporalScopeDescriptor cycle =
      makeDescriptor(makeScope(0, 0), {0, 0}, {1024, 128});
  cycle.precedence = {{0, 1}, {1, 0}};
  TemporalDomainResult cycleResult = buildTemporalDomain({cycle});
  ASSERT_FALSE(cycleResult.succeeded());
  ASSERT_TRUE(cycleResult.failure);
  EXPECT_EQ(cycleResult.failure->kind,
            TemporalDomainFailureKind::BrokenContract);

  TemporalScopeDescriptor malformed = cycle;
  malformed.precedence.clear();
  malformed.iteratorCapabilities.pop_back();
  TemporalDomainResult malformedResult = buildTemporalDomain({malformed});
  ASSERT_FALSE(malformedResult.succeeded());
  ASSERT_TRUE(malformedResult.failure);
  EXPECT_EQ(malformedResult.failure->kind,
            TemporalDomainFailureKind::BrokenContract);

  auto module = parse(R"mlir(
module {
  func.func @main(%input: tensor<2x1025x128xbf16>)
      -> tensor<2x1025x128xbf16> {
    %empty = tensor.empty() : tensor<2x1025x128xbf16>
    %result = linalg.map ins(%input : tensor<2x1025x128xbf16>)
        outs(%empty : tensor<2x1025x128xbf16>) (%value: bf16) {
      linalg.yield %value : bf16
    }
    return %result : tensor<2x1025x128xbf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  auto operation =
      llvm::find_if(function.getBody().front(), [](mlir::Operation &candidate) {
        return mlir::isa<mlir::linalg::MapOp>(candidate);
      });
  ASSERT_NE(operation, function.getBody().front().end());
  SemanticRootKey root;
  RootRegionWork work;
  work.id = {root, TileId(0)};
  work.rootOperation = &*operation;
  LogicalShardId shard{root, {0, 0, 0}};
  work.execution.push_back({shard, {{0, 2}, {0, 1025}, {0, 128}}});
  ExecutionInstancePlan nested;
  nested.id = ExecutionInstanceId{RequiredRootExecution{work.id, shard}};
  nested.placement = ExecutionInstancePlan::NestedUnder{nested.id.source};
  RegionGroupPlan group;
  group.tile = TileId(0);
  group.mandatoryRoots.push_back(work.id);
  group.executions.push_back(nested);
  TemporalDomainResult nestedResult =
      buildTemporalDomain(RegionPlan{{group}}, {work});
  ASSERT_FALSE(nestedResult.succeeded());
  ASSERT_TRUE(nestedResult.failure);
  EXPECT_EQ(nestedResult.failure->kind,
            TemporalDomainFailureKind::BrokenContract);
  EXPECT_NE(nestedResult.failure->detail.find("direct region-use"),
            std::string::npos);
}

} // namespace
