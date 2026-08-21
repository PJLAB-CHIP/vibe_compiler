//===- DataMovementTest.cpp ------------------------------------------===//

#include "Wafer/Planning/Search/DataMovement.h"

#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"
#include "TestSupport/Planning/SpatialDemandTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Async/IR/Async.h"
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
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <map>
#include <memory>
#include <optional>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  registerWaferCoreDialects(registry);
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

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context) {
  return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<4xf16>) -> tensor<4xf16> {
    %e0 = tensor.empty() : tensor<4xf16>
    %producer = linalg.map ins(%input : tensor<4xf16>)
        outs(%e0 : tensor<4xf16>) (%v: f16) {
      %x = arith.addf %v, %v : f16
      linalg.yield %x : f16
    }
    %e1 = tensor.empty() : tensor<4xf16>
    %consumer = linalg.map ins(%producer : tensor<4xf16>)
        outs(%e1 : tensor<4xf16>) (%v: f16) {
      %x = arith.mulf %v, %v : f16
      linalg.yield %x : f16
    }
    return %consumer : tensor<4xf16>
  }
}
)mlir",
                                                 mlir::ParserConfig(&context));
}

mlir::OwningOpRef<mlir::ModuleOp> parseBroadcast(mlir::MLIRContext &context) {
  return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @broadcast(%input: tensor<2xf16>) -> tensor<6x2xf16> {
    %e0 = tensor.empty() : tensor<2xf16>
    %producer = linalg.map ins(%input : tensor<2xf16>)
        outs(%e0 : tensor<2xf16>) (%v: f16) {
      %x = arith.addf %v, %v : f16
      linalg.yield %x : f16
    }
    %e1 = tensor.empty() : tensor<6x2xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%producer : tensor<2xf16>) outs(%e1 : tensor<6x2xf16>) {
      ^bb0(%v: f16, %old: f16):
        %sum = arith.addf %v, %v : f16
        linalg.yield %sum : f16
    } -> tensor<6x2xf16>
    return %consumer : tensor<6x2xf16>
  }
}
)mlir",
                                                 mlir::ParserConfig(&context));
}

mlir::OwningOpRef<mlir::ModuleOp> parseReduction(mlir::MLIRContext &context) {
  return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @reduce(%input: tensor<4x8xf16>, %init: tensor<4xf16>)
      -> tensor<4xf16> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<4x8xf16>) outs(%init : tensor<4xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<4xf16>
    return %sum : tensor<4xf16>
  }
}
)mlir",
                                                 mlir::ParserConfig(&context));
}

struct Prepared {
  std::unique_ptr<CardProgramAnalysis> program;
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  CoupledRegionDomain coupledDomain;
  CoupledRegionAssignment coupledAssignment;
  CardTemporalDomain temporalDomain;
  CardTemporalAssignment temporalAssignment;
  CardPhysicalRepresentationDomain representationDomain;
  CardPhysicalRepresentationAssignment representationAssignment;
};

template <typename OpT> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](OpT) { ++count; });
  return count;
}

mlir::FailureOr<Prepared>
prepare(mlir::ModuleOp module,
        llvm::ArrayRef<StructuredDAGNodePlacement> placements,
        bool maximalGroups, std::string *failureReason) {
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  auto dag = StructuredDAGAnalysis::create(function, failureReason);
  if (mlir::failed(dag) || dag->getNodes().size() != placements.size())
    return mlir::failure();
  auto spatialDemand =
      wafer::test::buildTestSpatialDemand(*dag, placements, failureReason);
  if (mlir::failed(spatialDemand))
    return mlir::failure();
  auto coupled = CoupledRegionDomain::create(
      *dag, spatialDemand->spatial, spatialDemand->demand, failureReason);
  auto temporal = CardTemporalDomain::create(*dag, placements, failureReason);
  auto topology = TargetTopology::create(module, failureReason);
  if (mlir::failed(coupled) || mlir::failed(temporal) || mlir::failed(topology))
    return mlir::failure();
  CoupledRegionAssignment coupledAssignment = coupled->getFirstAssignment();
  if (maximalGroups)
    while (true) {
      auto next = coupled->getNextAssignment(coupledAssignment);
      if (mlir::failed(next))
        return mlir::failure();
      if (!*next)
        break;
      coupledAssignment = std::move(**next);
    }
  CardTemporalAssignment temporalAssignment = temporal->getFirstAssignment();
  llvm::SmallVector<StructuredOperationNodeMapping, 16> operationNodes;
  for (const StructuredDAGNode &node : dag->getNodes())
    operationNodes.push_back({node.operation, node.id});
  StaticOutputDomains outputDomains;
  for (mlir::Type type : function.getResultTypes())
    outputDomains.emplace_back(
        mlir::cast<mlir::RankedTensorType>(type).getShape());
  auto program = std::make_unique<CardProgramAnalysis>(
      std::move(*topology),
      llvm::SmallVector<TileId, 16>{TileId(0), TileId(1), TileId(2), TileId(3)},
      std::move(*dag), std::move(outputDomains), std::move(operationNodes));
  auto representation = CardPhysicalRepresentationDomain::create(
      *program, spatialDemand->spatial, spatialDemand->demand, *coupled,
      coupledAssignment, *temporal, temporalAssignment, failureReason);
  if (mlir::failed(representation))
    return mlir::failure();
  CardPhysicalRepresentationAssignment representationAssignment =
      representation->getFirstAssignment();
  return Prepared{std::move(program),
                  std::move(spatialDemand->spatial),
                  std::move(spatialDemand->demand),
                  std::move(*coupled),
                  std::move(coupledAssignment),
                  std::move(*temporal),
                  std::move(temporalAssignment),
                  std::move(*representation),
                  std::move(representationAssignment)};
}

TEST(DataMovementTest, EnumeratesDDRRecomputeAndEverySimplePeerRoute) {
  auto context = createContext();
  auto module = parse(*context);
  ASSERT_TRUE(module);
  std::string failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements{
      StructuredDAGNodePlacement{0, {2}, {TileId(0), TileId(1)}},
      StructuredDAGNodePlacement{1, {2}, {TileId(2), TileId(3)}}};
  auto prepared =
      prepare(*module, placements, /*maximalGroups=*/true, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto domain = CardDataMovementDomain::create(
      *prepared->program, CardId(0), prepared->spatial, prepared->demand,
      prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      prepared->representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardDataMovementAssignment current = domain->getFirstAssignment();
  CardDataMovementAssignment ddr = current;
  ASSERT_EQ(current.edges.size(), 2u);
  EXPECT_TRUE(llvm::all_of(current.edges, [](const auto &choice) {
    return choice.kind == DataMovementKind::DDR;
  }));
  unsigned assignments = 0;
  unsigned peerChoices = 0;
  std::optional<CardDataMovementAssignment> peerAssignment;
  while (true) {
    ASSERT_TRUE(domain->contains(current));
    ++assignments;
    for (const DataMovementChoice &choice : current.edges)
      if (choice.kind == DataMovementKind::Peer) {
        ++peerChoices;
        ASSERT_FALSE(choice.fragments.empty());
        for (const DataMovementFragment &fragment : choice.fragments) {
          EXPECT_GT(fragment.logicalBytes, 0u);
          EXPECT_GT(fragment.physicalBytes, 0u);
          if (fragment.sourceTile != choice.destinationTile)
            EXPECT_FALSE(fragment.route.empty());
        }
      }
    if (llvm::all_of(current.edges,
                     [](const DataMovementChoice &choice) {
                       return choice.kind == DataMovementKind::Peer;
                     }) &&
        !peerAssignment)
      peerAssignment = current;
    auto next = domain->getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
  EXPECT_EQ(assignments, 16u);
  EXPECT_GT(peerChoices, 0u);
  ASSERT_TRUE(peerAssignment);
  CardDataMovementAssignment cyclic = *peerAssignment;
  for (DataMovementChoice &choice : cyclic.edges) {
    if (choice.kind != DataMovementKind::Peer)
      continue;
    for (DataMovementFragment &fragment : choice.fragments) {
      if (fragment.route.empty())
        continue;
      fragment.route.push_back(
          TileLink{fragment.route.back().destination, fragment.sourceTile});
      break;
    }
    break;
  }
  EXPECT_FALSE(domain->contains(cyclic));

  auto materialized = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, ddr, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<TileRegionOp>(materialized->module->getOperation()), 4u);
  EXPECT_GT(countOps<StorageLoadOp>(materialized->module->getOperation()), 0u);
  EXPECT_GT(countOps<StorageStoreOp>(materialized->module->getOperation()), 0u);

  CardDataMovementAssignment recompute = ddr;
  for (DataMovementChoice &choice : recompute.edges)
    choice.kind = DataMovementKind::Recompute;
  ASSERT_TRUE(domain->contains(recompute));
  auto recomputed = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, recompute, &failureReason);
  ASSERT_TRUE(mlir::succeeded(recomputed)) << failureReason;
  EXPECT_EQ(countOps<ComputeElementwiseOp>(recomputed->module->getOperation()),
            6u);

  auto peer = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, *peerAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(peer)) << failureReason;
  EXPECT_GT(countOps<CommPeerSendOp>(peer->module->getOperation()), 0u);
  EXPECT_EQ(countOps<CommPeerSendOp>(peer->module->getOperation()),
            countOps<CommPeerRecvOp>(peer->module->getOperation()));
  EXPECT_EQ(countOps<mlir::async::AwaitOp>(peer->module->getOperation()),
            countOps<CommPeerSendOp>(peer->module->getOperation()) +
                countOps<CommPeerRecvOp>(peer->module->getOperation()));
  EXPECT_EQ(countOps<StorageLoadOp>(peer->module->getOperation()), 2u);
  EXPECT_EQ(countOps<StorageStoreOp>(peer->module->getOperation()), 2u);

  CardPhysicalRepresentationAssignment convertedRepresentations =
      prepared->representationAssignment;
  for (PhysicalRepresentationChoice &choice : convertedRepresentations.values) {
    if (choice.node == 0 && choice.role == PhysicalValueRole::Result)
      choice.layout = MemLayout::Cx;
    if (choice.node == 1 && choice.role == PhysicalValueRole::Operand &&
        choice.index == 0)
      choice.layout = MemLayout::NCx;
  }
  ASSERT_TRUE(
      prepared->representationDomain.contains(convertedRepresentations));
  auto convertedDomain = CardDataMovementDomain::create(
      *prepared->program, CardId(0), prepared->spatial, prepared->demand,
      prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      convertedRepresentations, &failureReason);
  ASSERT_TRUE(mlir::succeeded(convertedDomain)) << failureReason;
  std::optional<CardDataMovementAssignment> convertedPeer;
  CardDataMovementAssignment converted = convertedDomain->getFirstAssignment();
  while (true) {
    if (llvm::all_of(converted.edges, [](const DataMovementChoice &choice) {
          return choice.kind == DataMovementKind::Peer;
        })) {
      convertedPeer = converted;
      break;
    }
    auto next = convertedDomain->getNextAssignment(converted);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    converted = std::move(**next);
  }
  ASSERT_TRUE(convertedPeer);
  auto convertedIR = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, convertedRepresentations,
      *convertedDomain, *convertedPeer, &failureReason);
  ASSERT_TRUE(mlir::succeeded(convertedIR)) << failureReason;
  EXPECT_GT(countOps<LayoutMaterializeOp>(convertedIR->module->getOperation()),
            0u);
}

TEST(DataMovementTest, RetainedBelongsOnlyToTheSelectedCoupledGroup) {
  auto context = createContext();
  auto module = parse(*context);
  ASSERT_TRUE(module);
  std::string failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements{
      StructuredDAGNodePlacement{0, {1}, {TileId(0)}},
      StructuredDAGNodePlacement{1, {1}, {TileId(0)}}};
  auto prepared =
      prepare(*module, placements, /*maximalGroups=*/true, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto domain = CardDataMovementDomain::create(
      *prepared->program, CardId(0), prepared->spatial, prepared->demand,
      prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      prepared->representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardDataMovementAssignment retained = domain->getFirstAssignment();
  ASSERT_EQ(retained.edges.size(), 1u);
  EXPECT_EQ(retained.edges.front().kind, DataMovementKind::Retained);
  auto next = domain->getNextAssignment(retained);
  ASSERT_TRUE(mlir::succeeded(next));
  ASSERT_TRUE(*next);
  EXPECT_EQ((**next).edges.front().kind, DataMovementKind::Refetch);
  auto end = domain->getNextAssignment(**next);
  ASSERT_TRUE(mlir::succeeded(end));
  EXPECT_FALSE(*end);
  auto materialized = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, retained, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<TileRegionOp>(materialized->module->getOperation()), 1u);
  auto refetched = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, **next, &failureReason);
  ASSERT_TRUE(mlir::succeeded(refetched)) << failureReason;
  EXPECT_GT(countOps<StorageLoadOp>(refetched->module->getOperation()),
            countOps<StorageLoadOp>(materialized->module->getOperation()));
  EXPECT_GT(countOps<StorageStoreOp>(refetched->module->getOperation()),
            countOps<StorageStoreOp>(materialized->module->getOperation()));
}

TEST(DataMovementTest, AssemblesSeveralRemoteOwnershipFragmentsWithoutDDR) {
  auto context = createContext();
  auto module = parse(*context);
  ASSERT_TRUE(module);
  std::string failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements{
      StructuredDAGNodePlacement{0, {2}, {TileId(0), TileId(1)}},
      StructuredDAGNodePlacement{1, {1}, {TileId(2)}}};
  auto prepared =
      prepare(*module, placements, /*maximalGroups=*/true, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto domain = CardDataMovementDomain::create(
      *prepared->program, CardId(0), prepared->spatial, prepared->demand,
      prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      prepared->representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardDataMovementAssignment current = domain->getFirstAssignment();
  std::optional<CardDataMovementAssignment> peer;
  while (true) {
    if (current.edges.size() == 1 &&
        current.edges.front().kind == DataMovementKind::Peer) {
      peer = current;
      break;
    }
    auto next = domain->getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
  ASSERT_TRUE(peer);
  ASSERT_EQ(peer->edges.front().fragments.size(), 2u);
  EXPECT_EQ(peer->edges.front().fragments[0].destinationOffsets,
            (llvm::SmallVector<int64_t, 4>{0}));
  EXPECT_EQ(peer->edges.front().fragments[1].destinationOffsets,
            (llvm::SmallVector<int64_t, 4>{2}));
  auto materialized = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, *peer, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<StorageLoadOp>(materialized->module->getOperation()), 2u);
  EXPECT_EQ(countOps<StorageStoreOp>(materialized->module->getOperation()), 1u);
  EXPECT_EQ(countOps<CommPeerSendOp>(materialized->module->getOperation()),
            countOps<CommPeerRecvOp>(materialized->module->getOperation()));
}

TEST(DataMovementTest, CombinesLocalAndRemoteFragmentsExactly) {
  auto context = createContext();
  auto module = parse(*context);
  ASSERT_TRUE(module);
  std::string failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements{
      StructuredDAGNodePlacement{0, {2}, {TileId(0), TileId(1)}},
      StructuredDAGNodePlacement{1, {1}, {TileId(0)}}};
  auto prepared =
      prepare(*module, placements, /*maximalGroups=*/false, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto domain = CardDataMovementDomain::create(
      *prepared->program, CardId(0), prepared->spatial, prepared->demand,
      prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      prepared->representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardDataMovementAssignment current = domain->getFirstAssignment();
  std::optional<CardDataMovementAssignment> peer;
  while (true) {
    if (current.edges.size() == 1 &&
        current.edges.front().kind == DataMovementKind::Peer) {
      peer = current;
      break;
    }
    auto next = domain->getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
  ASSERT_TRUE(peer);
  ASSERT_EQ(peer->edges.front().fragments.size(), 2u);
  EXPECT_TRUE(llvm::any_of(peer->edges.front().fragments,
                           [](const DataMovementFragment &fragment) {
                             return fragment.route.empty();
                           }));
  auto materialized = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, *peer, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<StorageLoadOp>(materialized->module->getOperation()), 3u);
  EXPECT_EQ(countOps<StorageStoreOp>(materialized->module->getOperation()), 2u);
  EXPECT_GT(countOps<CommPeerSendOp>(materialized->module->getOperation()), 0u);
}

TEST(DataMovementTest, EnumeratesPartialAndMaximalMulticastPartitions) {
  auto context = createContext();
  auto module = parseBroadcast(*context);
  ASSERT_TRUE(module);
  std::string failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements{
      StructuredDAGNodePlacement{0, {1}, {TileId(0)}},
      StructuredDAGNodePlacement{
          1, {3, 1}, {TileId(1), TileId(2), TileId(3)}}};
  auto prepared =
      prepare(*module, placements, /*maximalGroups=*/true, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto domain = CardDataMovementDomain::create(
      *prepared->program, CardId(0), prepared->spatial, prepared->demand,
      prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      prepared->representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardDataMovementAssignment current = domain->getFirstAssignment();
  bool sawPair = false;
  bool sawAll = false;
  auto reuseFacts = domain->getReuseFacts();
  ASSERT_EQ(reuseFacts.size(), 3u);
  for (const DataReuseFact &fact : reuseFacts) {
    EXPECT_EQ(fact.equivalentDestinationTiles.size(), 3u);
    EXPECT_EQ(fact.temporalInvariantIterators,
              (llvm::SmallVector<uint32_t, 4>{0}));
  }
  std::optional<CardDataMovementAssignment> partial;
  std::optional<CardDataMovementAssignment> maximal;
  while (true) {
    ASSERT_TRUE(domain->contains(current));
    std::map<int64_t, unsigned> groupSizes;
    for (const DataMovementChoice &choice : current.edges)
      if (choice.multicastGroup >= 0)
        ++groupSizes[choice.multicastGroup];
    for (const auto &[group, size] : groupSizes) {
      (void)group;
      if (size == 2) {
        sawPair = true;
        if (!partial)
          partial = current;
      }
      if (size == 3) {
        sawAll = true;
        if (!maximal)
          maximal = current;
      }
    }
    auto next = domain->getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
  EXPECT_TRUE(sawPair);
  EXPECT_TRUE(sawAll);
  ASSERT_TRUE(partial);
  ASSERT_TRUE(maximal);
  CardDataMovementAssignment unicast = *maximal;
  for (DataMovementChoice &choice : unicast.edges)
    choice.multicastGroup = -1;
  ASSERT_TRUE(domain->contains(unicast));
  auto unicastIR = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, unicast, &failureReason);
  ASSERT_TRUE(mlir::succeeded(unicastIR)) << failureReason;
  auto multicastIR = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, *maximal, &failureReason);
  ASSERT_TRUE(mlir::succeeded(multicastIR)) << failureReason;
  auto partialIR = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, *partial, &failureReason);
  ASSERT_TRUE(mlir::succeeded(partialIR)) << failureReason;
  EXPECT_LT(countOps<CommPeerSendOp>(multicastIR->module->getOperation()),
            countOps<CommPeerSendOp>(unicastIR->module->getOperation()));
  EXPECT_LE(countOps<CommPeerSendOp>(partialIR->module->getOperation()),
            countOps<CommPeerSendOp>(unicastIR->module->getOperation()));
  EXPECT_EQ(countOps<StorageLoadOp>(multicastIR->module->getOperation()), 1u);
  EXPECT_EQ(countOps<StorageStoreOp>(multicastIR->module->getOperation()), 3u);
}

TEST(DataMovementTest, MaterializesPartialReductionPeerGather) {
  auto context = createContext();
  auto module = parseReduction(*context);
  ASSERT_TRUE(module);
  std::string failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 1> placements{
      StructuredDAGNodePlacement{
          0, {2, 2}, {TileId(0), TileId(1), TileId(2), TileId(3)}}};
  auto prepared =
      prepare(*module, placements, /*maximalGroups=*/true, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto domain = CardDataMovementDomain::create(
      *prepared->program, CardId(0), prepared->spatial, prepared->demand,
      prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, prepared->representationDomain,
      prepared->representationAssignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CardDataMovementAssignment ddr = domain->getFirstAssignment();
  ASSERT_EQ(ddr.reductions.size(), 2u);
  EXPECT_TRUE(llvm::all_of(ddr.reductions, [](const auto &reduction) {
    return reduction.kind == ReductionGatherKind::DDR;
  }));
  auto ddrIR = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, ddr, &failureReason);
  ASSERT_TRUE(mlir::succeeded(ddrIR)) << failureReason;
  EXPECT_EQ(countOps<StorageStoreOp>(ddrIR->module->getOperation()), 6u);
  auto next = domain->getNextAssignment(ddr);
  ASSERT_TRUE(mlir::succeeded(next));
  ASSERT_TRUE(*next);
  ASSERT_TRUE(llvm::any_of((**next).reductions, [](const auto &reduction) {
    return reduction.kind == ReductionGatherKind::Peer;
  }));
  auto peer = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->spatial,
      prepared->demand,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, prepared->representationAssignment,
      *domain, **next, &failureReason);
  ASSERT_TRUE(mlir::succeeded(peer)) << failureReason;
  EXPECT_GT(countOps<CommPeerSendOp>(peer->module->getOperation()), 0u);
  EXPECT_EQ(countOps<CommPeerSendOp>(peer->module->getOperation()),
            countOps<CommPeerRecvOp>(peer->module->getOperation()));
  EXPECT_GT(countOps<TileRegionOp>(peer->module->getOperation()), 6u);
  EXPECT_LT(countOps<StorageStoreOp>(peer->module->getOperation()), 6u);
}

} // namespace
