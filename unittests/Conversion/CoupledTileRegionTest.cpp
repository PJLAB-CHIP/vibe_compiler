//===- CoupledTileRegionTest.cpp --------------------------------------===//

#include "Wafer/Planning/Search/CoupledRegion.h"
#include "Wafer/Planning/Search/PhysicalRepresentation.h"

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <array>
#include <memory>
#include <optional>
#include <set>

namespace {

using wafer::TileId;
using wafer::compiler::detail::CardPhysicalRepresentationAssignment;
using wafer::compiler::detail::CardPhysicalRepresentationDomain;
using wafer::compiler::detail::CardProgramAnalysis;
using wafer::compiler::detail::CardTemporalAssignment;
using wafer::compiler::detail::CardTemporalDomain;
using wafer::compiler::detail::CoupledRegionAssignment;
using wafer::compiler::detail::CoupledRegionDomain;
using wafer::compiler::detail::CoupledRegionGroup;
using wafer::compiler::detail::PhysicalRepresentationChoice;
using wafer::compiler::detail::PhysicalValueRole;
using wafer::compiler::detail::StaticOutputDomains;
using wafer::compiler::detail::StructuredDAGAnalysis;
using wafer::compiler::detail::StructuredDAGNodePlacement;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context,
                                        llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source,
                                                 mlir::ParserConfig(&context));
}

template <typename OpT> unsigned countOps(mlir::Operation *operation) {
  unsigned count = 0;
  operation->walk([&](OpT) { ++count; });
  return count;
}

struct PreparedCase {
  std::unique_ptr<CardProgramAnalysis> program;
  wafer::analysis::LogicalShardTrial trial;
  CoupledRegionDomain domain;
  CardTemporalDomain temporalDomain;
  CardTemporalAssignment temporalAssignment;
  CardPhysicalRepresentationDomain representationDomain;
  CardPhysicalRepresentationAssignment representationAssignment;
};

mlir::FailureOr<PreparedCase> prepare(mlir::ModuleOp module,
                                      std::string *failureReason) {
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  mlir::FailureOr<StructuredDAGAnalysis> dag =
      StructuredDAGAnalysis::create(function, failureReason);
  if (mlir::failed(dag))
    return mlir::failure();
  llvm::SmallVector<StructuredDAGNodePlacement, 8> placements;
  llvm::SmallVector<wafer::StructuredOperationNodeMapping, 16> operationNodes;
  for (const auto &node : dag->getNodes()) {
    auto tiling = mlir::cast<mlir::TilingInterface>(node.operation);
    placements.push_back(StructuredDAGNodePlacement{
        node.id,
        llvm::SmallVector<uint32_t, 4>(tiling.getLoopIteratorTypes().size(), 1),
        {TileId(0)},
        std::nullopt});
    operationNodes.push_back({node.operation, node.id});
  }
  wafer::analysis::IREpoch epoch = wafer::analysis::IREpoch::mint();
  mlir::FailureOr<wafer::analysis::LogicalShardTrial> trial =
      wafer::compiler::detail::buildLogicalShardTrial(*dag, placements, epoch,
                                                      failureReason);
  if (mlir::failed(trial))
    return mlir::failure();
  mlir::FailureOr<CoupledRegionDomain> domain =
      CoupledRegionDomain::create(*dag, *trial, failureReason);
  if (mlir::failed(domain))
    return mlir::failure();
  mlir::FailureOr<CardTemporalDomain> temporalDomain =
      CardTemporalDomain::create(*dag, placements, failureReason);
  if (mlir::failed(temporalDomain))
    return mlir::failure();
  CardTemporalAssignment temporalAssignment =
      temporalDomain->getFirstAssignment();
  mlir::FailureOr<wafer::TargetTopology> topology =
      wafer::TargetTopology::create(module, failureReason);
  if (mlir::failed(topology))
    return mlir::failure();
  llvm::SmallVector<TileId, 16> available{TileId(0), TileId(1), TileId(2),
                                          TileId(3)};
  StaticOutputDomains outputDomains;
  for (mlir::Type result : function.getResultTypes()) {
    auto tensor = mlir::cast<mlir::RankedTensorType>(result);
    outputDomains.emplace_back(tensor.getShape());
  }
  auto program = std::make_unique<CardProgramAnalysis>(
      std::move(*topology), available, std::move(*dag),
      std::move(outputDomains), std::move(operationNodes), epoch);
  CoupledRegionAssignment coupledAssignment = domain->getFirstAssignment();
  auto representationDomain = CardPhysicalRepresentationDomain::create(
      *program, *trial, *domain, coupledAssignment, *temporalDomain,
      temporalAssignment, failureReason);
  if (mlir::failed(representationDomain))
    return mlir::failure();
  CardPhysicalRepresentationAssignment representationAssignment =
      representationDomain->getFirstAssignment();
  return PreparedCase{std::move(program),
                      std::move(*trial),
                      std::move(*domain),
                      std::move(*temporalDomain),
                      std::move(temporalAssignment),
                      std::move(*representationDomain),
                      std::move(representationAssignment)};
}

std::set<uint32_t>
getEmittedNodes(const wafer::StructuredMaterializationRelations &relations) {
  std::set<uint32_t> result;
  for (const wafer::StructuredOperationEmissionRelation &relation :
       relations.operationEmissions)
    if (relation.operation)
      result.insert(relation.structuredNodeId);
  return result;
}

CoupledRegionAssignment
getMaximalAssignment(const CoupledRegionDomain &domain) {
  CoupledRegionAssignment current = domain.getFirstAssignment();
  while (true) {
    auto next = domain.getNextAssignment(current);
    EXPECT_TRUE(mlir::succeeded(next));
    if (mlir::failed(next) || !*next)
      return current;
    current = std::move(**next);
  }
}

constexpr llvm::StringLiteral chain = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<4xf16>) -> tensor<4xf16> {
    %e0 = tensor.empty() : tensor<4xf16>
    %a = linalg.map ins(%input : tensor<4xf16>) outs(%e0 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    %e1 = tensor.empty() : tensor<4xf16>
    %b = linalg.map ins(%a : tensor<4xf16>) outs(%e1 : tensor<4xf16>)
        (%v: f16) { %x = arith.mulf %v, %v : f16
                    linalg.yield %x : f16 }
    %e2 = tensor.empty() : tensor<4xf16>
    %c = linalg.map ins(%b : tensor<4xf16>) outs(%e2 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    return %c : tensor<4xf16>
  }
})mlir";

TEST(CoupledTileRegionTest, MaterializesMaximalAndIntermediateChainCuts) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context, chain);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  CoupledRegionAssignment maximal = getMaximalAssignment(prepared->domain);
  ASSERT_EQ(maximal.groups.size(), 1u);
  ASSERT_EQ(maximal.groups.front().nodes,
            (llvm::SmallVector<uint32_t, 4>{0, 1, 2}));
  auto maximalIR = wafer::compiler::detail::materializeCardCoupledRegions(
      *module, *prepared->program, wafer::CardId(0), prepared->trial,
      prepared->domain, maximal, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      prepared->representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(maximalIR)) << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(maximalIR->module->getOperation()),
            1u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(maximalIR->module->getOperation()),
            1u);
  EXPECT_EQ(getEmittedNodes(maximalIR->relations),
            (std::set<uint32_t>{0, 1, 2}));

  CoupledRegionAssignment cut{{
      CoupledRegionGroup{TileId(0), {0, 1}},
      CoupledRegionGroup{TileId(0), {2}},
  }};
  ASSERT_TRUE(prepared->domain.contains(cut));
  auto cutIR = wafer::compiler::detail::materializeCardCoupledRegions(
      *module, *prepared->program, wafer::CardId(0), prepared->trial,
      prepared->domain, cut, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      prepared->representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(cutIR)) << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(cutIR->module->getOperation()), 2u);
  EXPECT_EQ(countOps<wafer::StorageStoreOp>(cutIR->module->getOperation()), 2u);

  std::set<std::set<uint32_t>> regionNodes;
  cutIR->module->walk([&](wafer::TileRegionOp region) {
    std::set<uint32_t> nodes;
    for (const wafer::StructuredOperationEmissionRelation &relation :
         cutIR->relations.operationEmissions)
      if (relation.operation &&
          relation.operation->getParentOfType<wafer::TileRegionOp>() == region)
        nodes.insert(relation.structuredNodeId);
    if (!nodes.empty())
      regionNodes.insert(std::move(nodes));
  });
  EXPECT_EQ(regionNodes, (std::set<std::set<uint32_t>>{{0, 1}, {2}}));
}

TEST(CoupledTileRegionTest,
     MaterializesSelectedTemporalLoopsInsideOneCoupledRegion) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context, chain);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  CoupledRegionAssignment maximal = getMaximalAssignment(prepared->domain);
  ASSERT_EQ(maximal.groups.size(), 1u);
  for (auto &node : prepared->temporalAssignment.nodes) {
    ASSERT_EQ(node.iteratorTileSizes.size(), 1u);
    node.iteratorTileSizes.front() = 2;
    node.waveLoopOrder = {0};
  }
  ASSERT_TRUE(prepared->temporalDomain.contains(prepared->temporalAssignment));
  auto representationDomain = CardPhysicalRepresentationDomain::create(
      *prepared->program, prepared->trial, prepared->domain, maximal,
      prepared->temporalDomain, prepared->temporalAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(representationDomain)) << failureReason;
  CardPhysicalRepresentationAssignment representationAssignment =
      representationDomain->getFirstAssignment();
  auto materialized = wafer::compiler::detail::materializeCardCoupledRegions(
      *module, *prepared->program, wafer::CardId(0), prepared->trial,
      prepared->domain, maximal, prepared->temporalDomain,
      prepared->temporalAssignment, *representationDomain,
      representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
            1u);
  EXPECT_GT(countOps<mlir::scf::ForOp>(materialized->module->getOperation()),
            0u);
  // Two output waves each store once. All three selected nodes stay between
  // the wave's input load and output store; there is no intermediate DDR
  // store/reload between the coupled elementwise operations.
  EXPECT_EQ(
      countOps<wafer::StorageLoadOp>(materialized->module->getOperation()), 2u);
  EXPECT_EQ(
      countOps<wafer::StorageStoreOp>(materialized->module->getOperation()),
      2u);
  EXPECT_EQ(countOps<wafer::ComputeElementwiseOp>(
                materialized->module->getOperation()),
            6u);
  EXPECT_EQ(getEmittedNodes(materialized->relations),
            (std::set<uint32_t>{0, 1, 2}));
}

TEST(CoupledTileRegionTest, PreservesInternalProducerParallelWaveLoopOrder) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @chain2d(%input: tensor<5x7xf16>) -> tensor<5x7xf16> {
    %e0 = tensor.empty() : tensor<5x7xf16>
    %producer = linalg.map ins(%input : tensor<5x7xf16>)
        outs(%e0 : tensor<5x7xf16>) (%v: f16) {
      %x = arith.addf %v, %v : f16
      linalg.yield %x : f16
    }
    %e1 = tensor.empty() : tensor<5x7xf16>
    %consumer = linalg.map ins(%producer : tensor<5x7xf16>)
        outs(%e1 : tensor<5x7xf16>) (%v: f16) {
      %x = arith.mulf %v, %v : f16
      linalg.yield %x : f16
    }
    return %consumer : tensor<5x7xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::linalg::MapOp, 2> operations;
  module->walk(
      [&](mlir::linalg::MapOp operation) { operations.push_back(operation); });
  ASSERT_EQ(operations.size(), 2u);
  std::array<wafer::StructuredOperationNodeMapping, 2> nodes = {
      wafer::StructuredOperationNodeMapping{operations[0], 0},
      wafer::StructuredOperationNodeMapping{operations[1], 1}};
  std::array<TileId, 4> tiles = {TileId(0), TileId(1), TileId(2), TileId(3)};
  auto materialize = [&](llvm::SmallVector<uint32_t, 2> order) {
    wafer::StructuredNodeShardGroup group;
    group.shards.push_back(
        wafer::StructuredNodeIterationShard{0, TileId(0), {0, 0}, {5, 7}});
    group.shards.push_back(
        wafer::StructuredNodeIterationShard{1, TileId(0), {0, 0}, {5, 7}});
    group.temporalTiles.push_back(
        wafer::StructuredNodeTemporalTile{0, {2, 3}, std::move(order)});
    group.temporalTiles.push_back(
        wafer::StructuredNodeTemporalTile{1, {5, 7}, {}});
    mlir::OwningOpRef<mlir::ModuleOp> result;
    std::string failureReason;
    EXPECT_TRUE(mlir::succeeded(wafer::lowerStructuredNodeGroupsToCardModule(
        *module, wafer::CardId(0), tiles, nodes, llvm::ArrayRef(&group, 1),
        result, nullptr, &failureReason)))
        << failureReason;
    return result;
  };
  auto getNestingStep = [](mlir::ModuleOp result) -> std::optional<int64_t> {
    std::optional<int64_t> step;
    result.walk([&](mlir::scf::ForOp outer) {
      bool hasNestedLoop = false;
      outer.getRegion().walk([&](mlir::scf::ForOp) { hasNestedLoop = true; });
      if (hasNestedLoop && !step)
        step = mlir::getConstantIntValue(outer.getStep());
    });
    return step;
  };
  mlir::OwningOpRef<mlir::ModuleOp> firstMajor = materialize({0, 1});
  mlir::OwningOpRef<mlir::ModuleOp> secondMajor = materialize({1, 0});
  ASSERT_TRUE(firstMajor && secondMajor);
  EXPECT_EQ(getNestingStep(*firstMajor), 2);
  EXPECT_EQ(getNestingStep(*secondMajor), 3);
}

TEST(CoupledTileRegionTest,
     SharesOneSelectedProducerPhysicalVersionAcrossFanout) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = parse(*context, R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @fanout(%input: tensor<4xf16>)
      -> (tensor<4xf16>, tensor<4xf16>) {
    %e0 = tensor.empty() : tensor<4xf16>
    %a = linalg.map ins(%input : tensor<4xf16>) outs(%e0 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    %e1 = tensor.empty() : tensor<4xf16>
    %b = linalg.map ins(%a : tensor<4xf16>) outs(%e1 : tensor<4xf16>)
        (%v: f16) { %x = arith.mulf %v, %v : f16
                    linalg.yield %x : f16 }
    %e2 = tensor.empty() : tensor<4xf16>
    %c = linalg.map ins(%a : tensor<4xf16>) outs(%e2 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    return %b, %c : tensor<4xf16>, tensor<4xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  CoupledRegionAssignment maximal = getMaximalAssignment(prepared->domain);
  CardPhysicalRepresentationAssignment representation =
      prepared->representationAssignment;
  for (PhysicalRepresentationChoice &choice : representation.values) {
    if (choice.node == 0 && choice.role == PhysicalValueRole::Result)
      choice.layout = wafer::MemLayout::Cx;
    if ((choice.node == 1 || choice.node == 2) &&
        choice.role == PhysicalValueRole::Operand && choice.index == 0)
      choice.layout = wafer::MemLayout::Cx;
  }
  ASSERT_TRUE(prepared->representationDomain.contains(representation));
  auto materialized = materializeCardCoupledRegions(
      *module, *prepared->program, wafer::CardId(0), prepared->trial,
      prepared->domain, maximal, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      representation, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;

  unsigned producerVersions = 0;
  for (const wafer::StructuredOperationBufferRelation &relation :
       materialized->relations.operationResultBuffers) {
    if (relation.structuredNodeId != 0)
      continue;
    auto type = mlir::dyn_cast<mlir::MemRefType>(relation.buffer.getType());
    if (type &&
        wafer::getWaferMemoryAttr(type).getLayout() == wafer::MemLayout::Cx)
      ++producerVersions;
  }
  EXPECT_EQ(producerVersions, 1u);
  EXPECT_EQ(countOps<wafer::LayoutMaterializeOp>(
                materialized->module->getOperation()),
            3u);
}

TEST(CoupledTileRegionTest,
     MaterializesFaninFanoutAndDiamondWithSharedProducerVersion) {
  constexpr llvm::StringLiteral fanin = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @fanin(%lhs: tensor<4xf16>, %rhs: tensor<4xf16>) -> tensor<4xf16> {
    %e0 = tensor.empty() : tensor<4xf16>
    %a = linalg.map ins(%lhs : tensor<4xf16>) outs(%e0 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    %e1 = tensor.empty() : tensor<4xf16>
    %b = linalg.map ins(%rhs : tensor<4xf16>) outs(%e1 : tensor<4xf16>)
        (%v: f16) { %x = arith.mulf %v, %v : f16
                    linalg.yield %x : f16 }
    %e2 = tensor.empty() : tensor<4xf16>
    %c = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]
      } ins(%a, %b : tensor<4xf16>, tensor<4xf16>)
        outs(%e2 : tensor<4xf16>) {
      ^bb0(%x: f16, %y: f16, %old: f16):
        %sum = arith.addf %x, %y : f16
        linalg.yield %sum : f16
    } -> tensor<4xf16>
    return %c : tensor<4xf16>
  }
})mlir";
  constexpr llvm::StringLiteral fanout = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @fanout(%input: tensor<4xf16>)
      -> (tensor<4xf16>, tensor<4xf16>) {
    %e0 = tensor.empty() : tensor<4xf16>
    %a = linalg.map ins(%input : tensor<4xf16>) outs(%e0 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    %e1 = tensor.empty() : tensor<4xf16>
    %b = linalg.map ins(%a : tensor<4xf16>) outs(%e1 : tensor<4xf16>)
        (%v: f16) { %x = arith.mulf %v, %v : f16
                    linalg.yield %x : f16 }
    %e2 = tensor.empty() : tensor<4xf16>
    %c = linalg.map ins(%a : tensor<4xf16>) outs(%e2 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    return %b, %c : tensor<4xf16>, tensor<4xf16>
  }
})mlir";
  constexpr llvm::StringLiteral supportFanin = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @support_fanin(%lhs: tensor<2xf16>, %rhs: tensor<2xf16>)
      -> tensor<4xf16> {
    %e0 = tensor.empty() : tensor<2xf16>
    %a = linalg.map ins(%lhs : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    %e1 = tensor.empty() : tensor<2xf16>
    %b = linalg.map ins(%rhs : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
        (%v: f16) { %x = arith.mulf %v, %v : f16
                    linalg.yield %x : f16 }
    %assembled_empty = tensor.empty() : tensor<4xf16>
    %left = tensor.insert_slice %a into %assembled_empty[0] [2] [1]
        : tensor<2xf16> into tensor<4xf16>
    %assembled = tensor.insert_slice %b into %left[2] [2] [1]
        : tensor<2xf16> into tensor<4xf16>
    %e2 = tensor.empty() : tensor<4xf16>
    %c = linalg.map ins(%assembled : tensor<4xf16>)
        outs(%e2 : tensor<4xf16>) (%v: f16) {
      %x = arith.addf %v, %v : f16
      linalg.yield %x : f16
    }
    return %c : tensor<4xf16>
  }
})mlir";
  constexpr llvm::StringLiteral observableChain = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @observable_chain(%input: tensor<4xf16>)
      -> (tensor<4xf16>, tensor<4xf16>) {
    %e0 = tensor.empty() : tensor<4xf16>
    %a = linalg.map ins(%input : tensor<4xf16>) outs(%e0 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    %e1 = tensor.empty() : tensor<4xf16>
    %b = linalg.map ins(%a : tensor<4xf16>) outs(%e1 : tensor<4xf16>)
        (%v: f16) { %x = arith.mulf %v, %v : f16
                    linalg.yield %x : f16 }
    return %a, %b : tensor<4xf16>, tensor<4xf16>
  }
})mlir";
  constexpr llvm::StringLiteral diamond = R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @diamond(%input: tensor<4xf16>) -> tensor<4xf16> {
    %e0 = tensor.empty() : tensor<4xf16>
    %a = linalg.map ins(%input : tensor<4xf16>) outs(%e0 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    %e1 = tensor.empty() : tensor<4xf16>
    %b = linalg.map ins(%a : tensor<4xf16>) outs(%e1 : tensor<4xf16>)
        (%v: f16) { %x = arith.mulf %v, %v : f16
                    linalg.yield %x : f16 }
    %e2 = tensor.empty() : tensor<4xf16>
    %c = linalg.map ins(%a : tensor<4xf16>) outs(%e2 : tensor<4xf16>)
        (%v: f16) { %x = arith.addf %v, %v : f16
                    linalg.yield %x : f16 }
    %e3 = tensor.empty() : tensor<4xf16>
    %d = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]
      } ins(%b, %c : tensor<4xf16>, tensor<4xf16>)
        outs(%e3 : tensor<4xf16>) {
      ^bb0(%x: f16, %y: f16, %old: f16):
        %sum = arith.addf %x, %y : f16
        linalg.yield %sum : f16
    } -> tensor<4xf16>
    return %d : tensor<4xf16>
  }
})mlir";

  for (auto [source, outputCount] :
       {std::pair<llvm::StringRef, unsigned>{fanin, 1},
        std::pair<llvm::StringRef, unsigned>{supportFanin, 1},
        std::pair<llvm::StringRef, unsigned>{observableChain, 2},
        std::pair<llvm::StringRef, unsigned>{fanout, 2},
        std::pair<llvm::StringRef, unsigned>{diamond, 1}}) {
    SCOPED_TRACE(source.str());
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    auto module = parse(*context, source);
    ASSERT_TRUE(module);
    std::string failureReason;
    auto prepared = prepare(*module, &failureReason);
    ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
    CoupledRegionAssignment maximal = getMaximalAssignment(prepared->domain);
    ASSERT_EQ(maximal.groups.size(), 1u);
    auto materialized = wafer::compiler::detail::materializeCardCoupledRegions(
        *module, *prepared->program, wafer::CardId(0), prepared->trial,
        prepared->domain, maximal, prepared->temporalDomain,
        prepared->temporalAssignment, prepared->representationDomain,
        prepared->representationAssignment, &failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(
        countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
        1u);
    EXPECT_EQ(
        countOps<wafer::StorageStoreOp>(materialized->module->getOperation()),
        outputCount);
    EXPECT_EQ(getEmittedNodes(materialized->relations).size(),
              maximal.groups.front().nodes.size());

    if (source == observableChain || source == fanout || source == diamond) {
      std::set<mlir::Operation *> sourceVersions;
      for (const wafer::StructuredOperationEmissionRelation &relation :
           materialized->relations.operationEmissions)
        if (relation.structuredNodeId == 0 && relation.operation)
          sourceVersions.insert(relation.operation);
      EXPECT_EQ(sourceVersions.size(), 1u);
    }
  }
}

} // namespace
