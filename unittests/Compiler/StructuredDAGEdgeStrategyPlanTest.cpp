//===- StructuredDAGEdgeStrategyPlanTest.cpp
//----------------------------------------===//

#include "../../lib/Wafer/Compiler/StructuredDAGEdgeStrategyPlan.h"
#include "../../lib/Wafer/Compiler/StructuredDAGCandidateSchedule.h"
#include "../../lib/Wafer/Compiler/StructuredDAGExactDemandQuery.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <limits>
#include <memory>
#include <string>

using namespace wafer;
using namespace wafer::compiler::detail;

namespace {

class StructuredDAGEdgeStrategyPlanTest : public ::testing::Test {
protected:
  StructuredDAGEdgeStrategyPlanTest() {
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

  StructuredDAGAnalysis buildDAG(mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function, &failureReason);
    EXPECT_TRUE(mlir::succeeded(dag)) << failureReason;
    if (mlir::failed(dag))
      std::abort();
    return std::move(*dag);
  }

  static StructuredDAGNodePlacement
  placement(StructuredDAGNodeID node, unsigned dimension,
            std::initializer_list<int64_t> tileValues) {
    StructuredDAGNodePlacement result;
    result.node = node;
    result.spatialPartition =
        StructuredDAGSpatialPartition{dimension, dimension};
    for (int64_t value : tileValues)
      result.tiles.push_back(TileId(value));
    return result;
  }

  static llvm::SmallVector<TileId, 16> available(unsigned count) {
    llvm::SmallVector<TileId, 16> result;
    for (unsigned value = 0; value < count; ++value)
      result.push_back(TileId(value));
    return result;
  }

  static const SpatialEdgeStrategy *
  findDestination(const StructuredDAGEdgeStrategyPlan &plan,
                  TileId destination) {
    auto found = llvm::find_if(plan.strategies,
                               [&](const SpatialEdgeStrategy &strategy) {
                                 return strategy.destinationTile == destination;
                               });
    return found == plan.strategies.end() ? nullptr : &*found;
  }

  static llvm::SmallVector<const SpatialEdgeFragment *, 8>
  collectFragments(const StructuredDAGEdgeStrategyPlan &plan,
                   SpatialEdgeFragmentKind kind) {
    llvm::SmallVector<const SpatialEdgeFragment *, 8> result;
    for (const SpatialEdgeStrategy &strategy : plan.strategies)
      for (const SpatialEdgeFragment &fragment : strategy.fragments)
        if (fragment.kind == kind)
          result.push_back(&fragment);
    return result;
  }

  static llvm::SmallVector<const StructuredDAGResourceReservation *, 16>
  collectReservations(const StructuredDAGCandidateSchedule &schedule,
                      StructuredDAGReservationResource resource) {
    llvm::SmallVector<const StructuredDAGResourceReservation *, 16> result;
    for (const StructuredDAGResourceReservation &reservation :
         schedule.resourceReservations)
      if (reservation.resource == resource)
        result.push_back(&reservation);
    llvm::sort(result, [](const StructuredDAGResourceReservation *lhs,
                          const StructuredDAGResourceReservation *rhs) {
      return std::tuple(lhs->startTime, lhs->finishTime, lhs->edge,
                        lhs->sourceTile.getValue(),
                        lhs->destinationTile.getValue()) <
             std::tuple(rhs->startTime, rhs->finishTime, rhs->edge,
                        rhs->sourceTile.getValue(),
                        rhs->destinationTile.getValue());
    });
    return result;
  }

  static constexpr llvm::StringLiteral kChain = R"mlir(
module {
  func.func @chain(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out0 = tensor.empty() : tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %consumer = linalg.map ins(%producer : tensor<8xf16>)
        outs(%out1 : tensor<8xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<8xf16>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kFanout = R"mlir(
module {
  func.func @fanout(%input: tensor<8xf16>)
      -> (tensor<8xf16>, tensor<8xf16>) {
    %out0 = tensor.empty() : tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %out2 = tensor.empty() : tensor<8xf16>
    %source = linalg.map ins(%input : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %left = linalg.map ins(%source : tensor<8xf16>)
        outs(%out1 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %right = linalg.map ins(%source : tensor<8xf16>)
        outs(%out2 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %left, %right : tensor<8xf16>, tensor<8xf16>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       PartialPlacementOverlapTransfersOnlyMissingExactDomains) {
  auto module = parse(kChain);
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements = {
      placement(0, 0, {0, 1, 2, 3}), placement(1, 0, {0, 1, 4, 5})};
  std::string failureReason;
  auto plan =
      deriveStructuredDAGEdgeStrategyPlan(dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(plan->strategies.size(), 4u);
  auto resident = collectFragments(*plan, SpatialEdgeFragmentKind::Resident);
  auto peer = collectFragments(*plan, SpatialEdgeFragmentKind::Peer);
  EXPECT_TRUE(
      llvm::none_of(plan->strategies, [](const SpatialEdgeStrategy &strategy) {
        return strategy.action == SpatialEdgeAction::PeerFragments &&
               llvm::any_of(
                   strategy.fragments, [](const SpatialEdgeFragment &fragment) {
                     return fragment.kind == SpatialEdgeFragmentKind::Resident;
                   });
      }));
  EXPECT_EQ(resident.size(), 0u);
  ASSERT_EQ(peer.size(), 2u);
  EXPECT_EQ(plan->totalPeerBytes, 8u);

  const SpatialEdgeStrategy *destinationZero =
      findDestination(*plan, TileId(0));
  const SpatialEdgeStrategy *destinationOne = findDestination(*plan, TileId(1));
  ASSERT_NE(destinationZero, nullptr);
  ASSERT_NE(destinationOne, nullptr);
  EXPECT_EQ(destinationZero->action, SpatialEdgeAction::LocalShardResidency);
  EXPECT_EQ(destinationOne->action, SpatialEdgeAction::LocalShardResidency);
  EXPECT_TRUE(destinationZero->fragments.empty());
  EXPECT_TRUE(destinationOne->fragments.empty());

  EXPECT_EQ(peer[0]->sourceTile, TileId(2));
  ASSERT_NE(findDestination(*plan, TileId(4)), nullptr);
  EXPECT_EQ(peer[0]->offsets, (llvm::SmallVector<int64_t, 4>{4}));
  EXPECT_EQ(peer[0]->sizes, (llvm::SmallVector<int64_t, 4>{2}));
  EXPECT_EQ(peer[0]->bytes, 4u);
  EXPECT_EQ(peer[0]->payloadSlice, 0u);
  EXPECT_EQ(peer[1]->sourceTile, TileId(3));
  ASSERT_NE(findDestination(*plan, TileId(5)), nullptr);
  EXPECT_EQ(peer[1]->payloadSlice, 1u);
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       DifferingShardDimensionsChargeOnlyTheResidentIntersection) {
  auto module = parse(R"mlir(
module {
  func.func @chain(%input: tensor<4x4xf16>) -> tensor<4x4xf16> {
    %out0 = tensor.empty() : tensor<4x4xf16>
    %out1 = tensor.empty() : tensor<4x4xf16>
    %producer = linalg.map ins(%input : tensor<4x4xf16>)
        outs(%out0 : tensor<4x4xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %consumer = linalg.map ins(%producer : tensor<4x4xf16>)
        outs(%out1 : tensor<4x4xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<4x4xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements = {
      placement(0, 0, {0, 1}), placement(1, 1, {1, 2})};
  std::string failureReason;
  auto plan =
      deriveStructuredDAGEdgeStrategyPlan(dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(plan->strategies.size(), 2u);
  auto resident = collectFragments(*plan, SpatialEdgeFragmentKind::Resident);
  auto peer = collectFragments(*plan, SpatialEdgeFragmentKind::Peer);
  ASSERT_EQ(resident.size(), 1u);
  ASSERT_EQ(peer.size(), 3u);

  EXPECT_EQ(resident[0]->sourceTile, TileId(1));
  EXPECT_EQ(resident[0]->offsets, (llvm::SmallVector<int64_t, 4>{2, 0}));
  EXPECT_EQ(resident[0]->sizes, (llvm::SmallVector<int64_t, 4>{2, 2}));
  EXPECT_EQ(resident[0]->bytes, 0u);
  const SpatialEdgeStrategy *destinationOne = findDestination(*plan, TileId(1));
  ASSERT_NE(destinationOne, nullptr);
  EXPECT_EQ(destinationOne->producerOffsets,
            (llvm::SmallVector<int64_t, 4>{0, 0}));
  EXPECT_EQ(destinationOne->producerSizes,
            (llvm::SmallVector<int64_t, 4>{4, 2}));
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       SameNumberedTransposeShardsStillRequireExactPeerFragments) {
  auto module = parse(R"mlir(
module {
  func.func @transpose_edge(%input: tensor<4x4xf16>) -> tensor<4x4xf16> {
    %out0 = tensor.empty() : tensor<4x4xf16>
    %out1 = tensor.empty() : tensor<4x4xf16>
    %producer = linalg.map ins(%input : tensor<4x4xf16>)
        outs(%out0 : tensor<4x4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%producer : tensor<4x4xf16>)
        outs(%out1 : tensor<4x4xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<4x4xf16>
    return %consumer : tensor<4x4xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 2u);
  ASSERT_EQ(dag.getEdges().size(), 1u);

  // Both placements use Tiles {0, 1} and result dimension 0.  The
  // consumer indexing map turns that result dimension into producer dimension
  // 1, so each destination needs one local and one remote half-fragment.
  std::string failureReason;
  auto plan = deriveStructuredDAGEdgeStrategyPlan(
      dag, {placement(0, 0, {0, 1}), placement(1, 0, {0, 1})}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_EQ(plan->strategies.size(), 2u);
  auto resident = collectFragments(*plan, SpatialEdgeFragmentKind::Resident);
  auto peer = collectFragments(*plan, SpatialEdgeFragmentKind::Peer);
  EXPECT_EQ(resident.size(), 2u);
  EXPECT_EQ(peer.size(), 2u);
  EXPECT_EQ(plan->totalPeerBytes, 16u);
  EXPECT_TRUE(
      llvm::all_of(plan->strategies, [](const SpatialEdgeStrategy &strategy) {
        return strategy.action == SpatialEdgeAction::PeerFragments;
      }));
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       ExactEdgePlanCarriesTypedProducerAndConsumerLayouts) {
  auto module = parse(R"mlir(
module {
  func.func @layout_edge(%lhs: tensor<8x8xf16>, %rhs: tensor<8x8xf16>)
      -> tensor<8x8xf16> {
    %matmulOut = tensor.empty() : tensor<8x8xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%matmulOut : tensor<8x8xf16>) -> tensor<8x8xf16>
    %product = linalg.matmul ins(%lhs, %rhs : tensor<8x8xf16>,
                                tensor<8x8xf16>)
        outs(%init : tensor<8x8xf16>) -> tensor<8x8xf16>
    %mapOut = tensor.empty() : tensor<8x8xf16>
    %result = linalg.map ins(%product : tensor<8x8xf16>)
        outs(%mapOut : tensor<8x8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %result : tensor<8x8xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 3u);
  std::string failureReason;
  auto plan = deriveStructuredDAGEdgeStrategyPlan(dag,
                                                  {placement(0, 0, {0, 1}),
                                                   placement(1, 0, {0, 1}),
                                                   placement(2, 0, {0, 1})},
                                                  &failureReason);
  ASSERT_TRUE(mlir::succeeded(plan)) << failureReason;
  ASSERT_FALSE(plan->strategies.empty());
  for (const SpatialEdgeStrategy &strategy : plan->strategies) {
    EXPECT_TRUE(strategy.hasLayoutAssignment);
    EXPECT_EQ(strategy.producerLayout, MemLayout::Cx);
    EXPECT_EQ(strategy.consumerLayout, MemLayout::Tensor);
  }
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       SearchEventScheduleUsesReadyRunningAndCompletedTransitions) {
  auto module = parse(kChain);
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements = {
      placement(0, 0, {0, 1, 2, 3}), placement(1, 0, {4, 5, 6, 7})};
  std::string failureReason;
  auto scheduled = scheduleStructuredDAGCandidate(
      dag, available(8), placements, /*localResidencies=*/{}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(scheduled)) << failureReason;
  EXPECT_EQ(scheduled->dispatchedWaves.size(), 6u);
  EXPECT_GT(scheduled->eventCount, 0u);
  EXPECT_GT(scheduled->makespan, 0u);
  EXPECT_EQ(scheduled->dispatchedWaves.front().wave,
            (SymbolicWaveClass{0, SymbolicWaveKind::Prologue}));
  // The class-matched dependency allows the consumer prologue to begin while
  // later producer classes remain outstanding.
  auto consumerPrologue =
      llvm::find_if(scheduled->dispatchedWaves, [](const RunningOpWave &wave) {
        return wave.wave == SymbolicWaveClass{1, SymbolicWaveKind::Prologue};
      });
  auto producerTail =
      llvm::find_if(scheduled->dispatchedWaves, [](const RunningOpWave &wave) {
        return wave.wave == SymbolicWaveClass{0, SymbolicWaveKind::Tail};
      });
  ASSERT_NE(consumerPrologue, scheduled->dispatchedWaves.end());
  ASSERT_NE(producerTail, scheduled->dispatchedWaves.end());
  EXPECT_LT(consumerPrologue->startTime, producerTail->finishTime);
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       PeerDataReadyEventGatesConsumerAndOverlapsProducerCompute) {
  auto module = parse(kChain);
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements = {
      placement(0, 0, {0}), placement(1, 0, {1})};
  llvm::SmallVector<StructuredDAGPeerMovement, 1> movements = {
      StructuredDAGPeerMovement{/*edge=*/0,
                                TileId(0),
                                TileId(1),
                                /*duration=*/2,
                                /*bufferCount=*/1,
                                {TileLink{TileId(0), TileId(1)}}}};
  std::string failureReason;
  auto scheduled = scheduleStructuredDAGCandidate(dag, available(2), placements,
                                                  /*localResidencies=*/{},
                                                  &failureReason, movements);
  ASSERT_TRUE(mlir::succeeded(scheduled)) << failureReason;
  EXPECT_GT(scheduled->movementEventCount, 0u);
  EXPECT_EQ(scheduled->peerMovementWork, 6u)
      << "one finite movement is scheduled for each symbolic wave class";

  auto producerPrologue =
      llvm::find_if(scheduled->dispatchedWaves, [](const RunningOpWave &wave) {
        return wave.wave == SymbolicWaveClass{0, SymbolicWaveKind::Prologue};
      });
  auto producerSteady =
      llvm::find_if(scheduled->dispatchedWaves, [](const RunningOpWave &wave) {
        return wave.wave == SymbolicWaveClass{0, SymbolicWaveKind::Steady};
      });
  auto consumerPrologue =
      llvm::find_if(scheduled->dispatchedWaves, [](const RunningOpWave &wave) {
        return wave.wave == SymbolicWaveClass{1, SymbolicWaveKind::Prologue};
      });
  ASSERT_NE(producerPrologue, scheduled->dispatchedWaves.end());
  ASSERT_NE(producerSteady, scheduled->dispatchedWaves.end());
  ASSERT_NE(consumerPrologue, scheduled->dispatchedWaves.end());
  EXPECT_GE(consumerPrologue->startTime, producerPrologue->finishTime + 2);
  EXPECT_LT(consumerPrologue->startTime, producerSteady->finishTime)
      << "peer work must not occupy the producer compute Tile";
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       BufferMultiplicityControlsProducerReuseWithoutBlockingCompute) {
  auto module = parse(kChain);
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements = {
      placement(0, 0, {0}), placement(1, 0, {1})};
  auto schedule = [&](uint8_t bufferCount) {
    llvm::SmallVector<StructuredDAGPeerMovement, 1> movements = {
        StructuredDAGPeerMovement{/*edge=*/0,
                                  TileId(0),
                                  TileId(1),
                                  /*duration=*/2,
                                  bufferCount,
                                  {TileLink{TileId(0), TileId(1)}}}};
    std::string failureReason;
    auto result = scheduleStructuredDAGCandidate(dag, available(2), placements,
                                                 /*localResidencies=*/{},
                                                 &failureReason, movements);
    EXPECT_TRUE(mlir::succeeded(result)) << failureReason;
    return result;
  };
  auto single = schedule(1);
  auto doubled = schedule(2);
  ASSERT_TRUE(mlir::succeeded(single));
  ASSERT_TRUE(mlir::succeeded(doubled));

  auto producerSteadyStart =
      [](const StructuredDAGCandidateSchedule &candidate) {
        auto found = llvm::find_if(
            candidate.dispatchedWaves, [](const RunningOpWave &wave) {
              return wave.wave ==
                     SymbolicWaveClass{0, SymbolicWaveKind::Steady};
            });
        EXPECT_NE(found, candidate.dispatchedWaves.end());
        return found == candidate.dispatchedWaves.end() ? ScheduleTime{0}
                                                        : found->startTime;
      };
  EXPECT_GT(producerSteadyStart(*single), producerSteadyStart(*doubled));
  EXPECT_GT(single->makespan, doubled->makespan);

  auto noc =
      collectReservations(*doubled, StructuredDAGReservationResource::NoCLink);
  auto compute =
      collectReservations(*doubled, StructuredDAGReservationResource::Compute);
  ASSERT_FALSE(noc.empty());
  EXPECT_TRUE(llvm::any_of(compute, [&](const auto *reservation) {
    return reservation->startTime < noc.front()->finishTime &&
           noc.front()->startTime < reservation->finishTime;
  })) << "movement and compute must occupy independent resource calendars";
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       TileSPMCalendarSerializesPerTileAndOverlapsAcrossTiles) {
  auto module = parse(kFanout);
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  std::string failureReason;

  llvm::SmallVector<StructuredDAGNodePlacement, 3> disjointPlacements = {
      placement(0, 0, {0}), placement(1, 0, {1}), placement(2, 0, {2})};
  llvm::SmallVector<StructuredDAGLocalMovement, 2> disjointMovements = {
      StructuredDAGLocalMovement{0, StructuredDAGLocalMovementResource::TileSPM,
                                 TileId(1), 2, 2},
      StructuredDAGLocalMovement{1, StructuredDAGLocalMovementResource::TileSPM,
                                 TileId(2), 3, 2}};
  auto disjoint = scheduleStructuredDAGCandidate(
      dag, available(3), disjointPlacements, /*localResidencies=*/{},
      &failureReason, /*peerMovements=*/{}, disjointMovements);
  ASSERT_TRUE(mlir::succeeded(disjoint)) << failureReason;
  auto independent =
      collectReservations(*disjoint, StructuredDAGReservationResource::TileSPM);
  ASSERT_GE(independent.size(), 2u);
  const ScheduleTime firstStart = independent.front()->startTime;
  EXPECT_TRUE(llvm::any_of(independent, [&](const auto *reservation) {
    return reservation->sourceTile == TileId(1) &&
           reservation->startTime == firstStart;
  }));
  EXPECT_TRUE(llvm::any_of(independent, [&](const auto *reservation) {
    return reservation->sourceTile == TileId(2) &&
           reservation->startTime == firstStart;
  }));

  llvm::SmallVector<StructuredDAGNodePlacement, 3> sharedPlacements = {
      placement(0, 0, {0}), placement(1, 0, {1}), placement(2, 0, {1})};
  llvm::SmallVector<StructuredDAGLocalMovement, 2> sharedMovements = {
      StructuredDAGLocalMovement{0, StructuredDAGLocalMovementResource::TileSPM,
                                 TileId(1), 2, 2},
      StructuredDAGLocalMovement{1, StructuredDAGLocalMovementResource::TileSPM,
                                 TileId(1), 3, 2}};
  auto shared = scheduleStructuredDAGCandidate(
      dag, available(2), sharedPlacements, /*localResidencies=*/{},
      &failureReason, /*peerMovements=*/{}, sharedMovements);
  ASSERT_TRUE(mlir::succeeded(shared)) << failureReason;
  auto serialized =
      collectReservations(*shared, StructuredDAGReservationResource::TileSPM);
  ASSERT_GE(serialized.size(), 2u);
  for (auto [previous, current] :
       llvm::zip(serialized, llvm::drop_begin(serialized)))
    EXPECT_LE(previous->finishTime, current->startTime);
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       NoCLinksSerializePerLinkAndCardDDRSerializesAcrossTiles) {
  auto module = parse(kFanout);
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  std::string failureReason;

  llvm::SmallVector<StructuredDAGNodePlacement, 3> separateDestinations = {
      placement(0, 0, {0}), placement(1, 0, {1}), placement(2, 0, {2})};
  llvm::SmallVector<StructuredDAGPeerMovement, 2> separateLinks = {
      StructuredDAGPeerMovement{
          0, TileId(0), TileId(1), 2, 2, {TileLink{TileId(0), TileId(1)}}},
      StructuredDAGPeerMovement{
          1, TileId(0), TileId(2), 3, 2, {TileLink{TileId(0), TileId(2)}}}};
  auto parallelLinks = scheduleStructuredDAGCandidate(
      dag, available(3), separateDestinations, /*localResidencies=*/{},
      &failureReason, separateLinks);
  ASSERT_TRUE(mlir::succeeded(parallelLinks)) << failureReason;
  auto parallelNoC = collectReservations(
      *parallelLinks, StructuredDAGReservationResource::NoCLink);
  ASSERT_GE(parallelNoC.size(), 2u);
  EXPECT_EQ(parallelNoC[0]->startTime, parallelNoC[1]->startTime);

  llvm::SmallVector<StructuredDAGNodePlacement, 3> sharedDestination = {
      placement(0, 0, {0}), placement(1, 0, {1}), placement(2, 0, {1})};
  llvm::SmallVector<StructuredDAGPeerMovement, 2> sharedLink = {
      StructuredDAGPeerMovement{
          0, TileId(0), TileId(1), 2, 2, {TileLink{TileId(0), TileId(1)}}},
      StructuredDAGPeerMovement{
          1, TileId(0), TileId(1), 3, 2, {TileLink{TileId(0), TileId(1)}}}};
  auto serializedLink = scheduleStructuredDAGCandidate(
      dag, available(2), sharedDestination, /*localResidencies=*/{},
      &failureReason, sharedLink);
  ASSERT_TRUE(mlir::succeeded(serializedLink)) << failureReason;
  auto serializedNoC = collectReservations(
      *serializedLink, StructuredDAGReservationResource::NoCLink);
  ASSERT_GE(serializedNoC.size(), 2u);
  for (auto [previous, current] :
       llvm::zip(serializedNoC, llvm::drop_begin(serializedNoC)))
    EXPECT_LE(previous->finishTime, current->startTime);

  llvm::SmallVector<StructuredDAGLocalMovement, 2> ddrMovements = {
      StructuredDAGLocalMovement{0, StructuredDAGLocalMovementResource::CardDDR,
                                 TileId(1), 2, 2},
      StructuredDAGLocalMovement{1, StructuredDAGLocalMovementResource::CardDDR,
                                 TileId(2), 3, 2}};
  auto serializedDDR = scheduleStructuredDAGCandidate(
      dag, available(3), separateDestinations, /*localResidencies=*/{},
      &failureReason, /*peerMovements=*/{}, ddrMovements);
  ASSERT_TRUE(mlir::succeeded(serializedDDR)) << failureReason;
  auto ddr = collectReservations(*serializedDDR,
                                 StructuredDAGReservationResource::CardDDR);
  ASSERT_GE(ddr.size(), 2u);
  for (auto [previous, current] : llvm::zip(ddr, llvm::drop_begin(ddr)))
    EXPECT_LE(previous->finishTime, current->startTime);
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       SearchScheduleRetainsAndReleasesLocalEdgeWaveResidency) {
  auto module = parse(kChain);
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements = {
      placement(0, 0, {0, 1, 2, 3}), placement(1, 0, {0, 1, 4, 5})};
  std::string failureReason;
  llvm::SmallVector<StructuredDAGLocalResidency, 4> localResidencies = {
      StructuredDAGLocalResidency{/*edge=*/0, TileId(0), /*bytes=*/4},
      StructuredDAGLocalResidency{/*edge=*/0, TileId(1), /*bytes=*/4}};

  auto scheduled = scheduleStructuredDAGCandidate(
      dag, available(8), placements, localResidencies, &failureReason);
  ASSERT_TRUE(mlir::succeeded(scheduled)) << failureReason;
  // The overlapping placement serializes producer and consumer waves.  All
  // three producer classes are therefore live before the first matching
  // consumer class completes: 3 classes * 2 elements * sizeof(f16).
  EXPECT_EQ(scheduled->peakLiveSPMBytes, 12u);

  localResidencies.front().footprintBytes =
      std::numeric_limits<uint64_t>::max();
  auto overCapacity = scheduleStructuredDAGCandidate(
      dag, available(8), placements, localResidencies, &failureReason);
  EXPECT_TRUE(mlir::failed(overCapacity));
  EXPECT_NE(failureReason.find("SPM high-water exceeds capacity"),
            std::string::npos)
      << failureReason;
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       DerivesExactPeerFragmentsForOneToManyReductionDemand) {
  auto module = parse(R"mlir(
module {
  func.func @reduce(%input: tensor<8x4xf16>) -> tensor<8xf16> {
    %out0 = tensor.empty() : tensor<8x4xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8x4xf16>)
        outs(%out0 : tensor<8x4xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out1 : tensor<8xf16>) -> tensor<8xf16>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%producer : tensor<8x4xf16>) outs(%init : tensor<8xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<8xf16>
    return %sum : tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  std::string failureReason;

  // The producer shard follows the reduction result's parallel iterator.
  // The unrecoverable reduction iterator remains wholly local, so this is a
  // valid residency even though a result point does not identify one producer
  // point.
  auto aligned = deriveStructuredDAGEdgeStrategyPlan(dag,
                                                     {placement(0, 0, {0, 1}),
                                                      placement(1, 0, {0, 1}),
                                                      placement(2, 0, {0, 1})},
                                                     &failureReason);
  ASSERT_TRUE(mlir::succeeded(aligned)) << failureReason;
  ASSERT_EQ(aligned->strategies.size(), 2u);
  EXPECT_TRUE(llvm::all_of(
      aligned->strategies, [](const SpatialEdgeStrategy &strategy) {
        return strategy.action == SpatialEdgeAction::LocalShardResidency;
      }));

  llvm::SmallVector<StructuredDAGNodePlacement, 3> placements = {
      placement(0, 0, {0, 1}), placement(1, 0, {2, 3}),
      placement(2, 0, {2, 3})};
  auto demandPlan =
      deriveStructuredDAGEdgeDemandPlan(dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(demandPlan)) << failureReason;
  ASSERT_EQ(demandPlan->demands.size(), 2u);
  for (const StructuredDAGEdgeDemand &demand : demandPlan->demands) {
    analysis::IndexSetResult exactDemand{
        analysis::IndexRelationStatus::Exact, demand.producerDemand, {}};
    auto rectangle = exactDemand.getExactStaticRectangularDomain();
    ASSERT_TRUE(rectangle.isExact()) << rectangle.reason;
    EXPECT_EQ(rectangle.domain->sizes, (llvm::SmallVector<int64_t, 4>{4, 4}));
    const llvm::SmallVector<int64_t, 4> firstOffsets{0, 0};
    const llvm::SmallVector<int64_t, 4> secondOffsets{4, 0};
    EXPECT_TRUE(rectangle.domain->offsets == firstOffsets ||
                rectangle.domain->offsets == secondOffsets);
    EXPECT_EQ(demand.producerShardOwnership.size(), 2u);
  }
  auto disjoint =
      deriveStructuredDAGEdgeStrategyPlan(dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(disjoint)) << failureReason;
  ASSERT_EQ(disjoint->strategies.size(), 2u);
  EXPECT_TRUE(llvm::all_of(
      disjoint->strategies, [](const SpatialEdgeStrategy &strategy) {
        return strategy.action == SpatialEdgeAction::PeerFragments &&
               strategy.producerSizes ==
                   llvm::SmallVector<int64_t, 4>({4, 4}) &&
               strategy.fragments.size() == 1 &&
               strategy.fragments.front().kind == SpatialEdgeFragmentKind::Peer;
      }));
  EXPECT_EQ(disjoint->totalPeerBytes, 64u);
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       StridedLogicalDemandSurvivesUntilCanonicalPhysicalLowering) {
  // The typed exact-demand query accepts the exact strided demand; only the
  // canonical dense-rectangle carrier rejects the physical assignment here.
  // Placement legality is never rewritten by that carrier failure.
  auto module = parse(R"mlir(
module {
  func.func @strided_edge(%input: tensor<8xf16>) -> tensor<4xf16> {
    %producerOut = tensor.empty() : tensor<8xf16>
    %producer = linalg.map ins(%input : tensor<8xf16>)
        outs(%producerOut : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %consumerOut = tensor.empty() : tensor<4xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0 * 2)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%producer : tensor<8xf16>)
        outs(%consumerOut : tensor<4xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<4xf16>
    return %consumer : tensor<4xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 2u);
  ASSERT_EQ(dag.getEdges().size(), 1u);

  std::string failureReason;
  auto demandPlan = deriveStructuredDAGEdgeDemandPlan(
      dag, {placement(0, 0, {0}), placement(1, 0, {1})}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(demandPlan)) << failureReason;
  ASSERT_EQ(demandPlan->demands.size(), 1u);
  const auto &logicalDemand = demandPlan->demands.front().producerDemand;
  auto contains = [&](int64_t index) {
    return logicalDemand.containsPoint(llvm::ArrayRef<int64_t>(index));
  };
  EXPECT_TRUE(contains(0));
  EXPECT_TRUE(contains(2));
  EXPECT_TRUE(contains(4));
  EXPECT_TRUE(contains(6));
  EXPECT_FALSE(contains(1));
  EXPECT_FALSE(contains(7));

  failureReason.clear();
  auto lowered = lowerStructuredDAGEdgeDemandPlanToCanonicalStrategies(
      dag, *demandPlan, &failureReason);
  EXPECT_TRUE(mlir::failed(lowered));
  EXPECT_NE(failureReason.find("dense logical rectangle"), std::string::npos)
      << failureReason;
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       DuplicateTileOwnershipIsReportedAsMalformedTrial) {
  auto module = parse(kChain);
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);

  // Repeating one producer Tile cannot form a partition: the typed query
  // reports a malformed trial instead of turning the missing upper half into
  // a placement legality bool. Proven coverage holes of well-formed trials
  // are asserted by the exact-demand query suite directly.
  std::string failureReason;
  auto demandPlan = deriveStructuredDAGEdgeDemandPlan(
      dag, {placement(0, 0, {0, 0}), placement(1, 0, {1, 2})}, &failureReason);
  EXPECT_TRUE(mlir::failed(demandPlan));
  EXPECT_NE(failureReason.find("several owner domains"), std::string::npos)
      << failureReason;
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       BroadcastDemandIsExactForEveryConsumerShard) {
  auto module = parse(R"mlir(
module {
  func.func @broadcast_edge(%input: tensor<4xf16>) -> tensor<8x4xf16> {
    %producerOut = tensor.empty() : tensor<4xf16>
    %producer = linalg.map ins(%input : tensor<4xf16>)
        outs(%producerOut : tensor<4xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %consumerOut = tensor.empty() : tensor<8x4xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%producer : tensor<4xf16>)
        outs(%consumerOut : tensor<8x4xf16>) {
      ^bb0(%value: f16, %old: f16):
        linalg.yield %value : f16
    } -> tensor<8x4xf16>
    return %consumer : tensor<8x4xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);

  std::string failureReason;
  auto demandPlan = deriveStructuredDAGEdgeDemandPlan(
      dag, {placement(0, 0, {0}), placement(1, 0, {1, 2})}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(demandPlan)) << failureReason;
  ASSERT_EQ(demandPlan->demands.size(), 2u);
  for (const StructuredDAGEdgeDemand &demand : demandPlan->demands) {
    analysis::IndexSetResult exactDemand{
        analysis::IndexRelationStatus::Exact, demand.producerDemand, {}};
    auto rectangle = exactDemand.getExactStaticRectangularDomain();
    ASSERT_TRUE(rectangle.isExact()) << rectangle.reason;
    EXPECT_EQ(rectangle.domain->offsets, (llvm::SmallVector<int64_t, 4>{0}));
    EXPECT_EQ(rectangle.domain->sizes, (llvm::SmallVector<int64_t, 4>{4}));
  }
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       DerivationIsDeterministicForFanoutEdges) {
  auto module = parse(R"mlir(
module {
  func.func @fanout(%input: tensor<8xf16>)
      -> (tensor<8xf16>, tensor<8xf16>) {
    %out0 = tensor.empty() : tensor<8xf16>
    %out1 = tensor.empty() : tensor<8xf16>
    %out2 = tensor.empty() : tensor<8xf16>
    %source = linalg.map ins(%input : tensor<8xf16>)
        outs(%out0 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %left = linalg.map ins(%source : tensor<8xf16>)
        outs(%out1 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %right = linalg.map ins(%source : tensor<8xf16>)
        outs(%out2 : tensor<8xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %left, %right : tensor<8xf16>, tensor<8xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  llvm::SmallVector<StructuredDAGNodePlacement, 3> placements = {
      placement(0, 0, {0, 1}), placement(1, 0, {2, 3}),
      placement(2, 0, {4, 5})};
  std::string failureReason;
  auto first =
      deriveStructuredDAGEdgeStrategyPlan(dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(first)) << failureReason;
  auto second =
      deriveStructuredDAGEdgeStrategyPlan(dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(second)) << failureReason;
  auto firstPeer = collectFragments(*first, SpatialEdgeFragmentKind::Peer);
  auto secondPeer = collectFragments(*second, SpatialEdgeFragmentKind::Peer);
  ASSERT_EQ(firstPeer.size(), 4u);
  ASSERT_EQ(secondPeer.size(), firstPeer.size());
  for (auto [lhs, rhs] : llvm::zip_equal(firstPeer, secondPeer)) {
    EXPECT_EQ(lhs->sourceTile, rhs->sourceTile);
    EXPECT_EQ(lhs->offsets, rhs->offsets);
    EXPECT_EQ(lhs->sizes, rhs->sizes);
    EXPECT_EQ(lhs->bytes, rhs->bytes);
    EXPECT_EQ(lhs->communicationId, rhs->communicationId);
    EXPECT_EQ(lhs->payloadSlice, rhs->payloadSlice);
  }
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       InitOperandEdgesCarryNoSpatialActionInAnyPlacement) {
  // The typed exact-demand query proves the init dependency (an explicit
  // init producer is an independent root with an exact demand); the
  // canonical carrier still emits no strategy for it: the
  // card-materialization contract keeps the init governed by the consumer's
  // typed lowering, so the plan must not derive any strategy for it,
  // whether the nodes share Tiles or are disjoint.
  auto module = parse(R"mlir(
module {
  func.func @gemm(%lhs: tensor<8x8xf16>, %rhs: tensor<8x8xf16>)
      -> tensor<8x8xf16> {
    %out = tensor.empty() : tensor<8x8xf16>
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<8x8xf16>) -> tensor<8x8xf16>
    %result = linalg.matmul ins(%lhs, %rhs : tensor<8x8xf16>,
                                tensor<8x8xf16>)
        outs(%init : tensor<8x8xf16>) -> tensor<8x8xf16>
    return %result : tensor<8x8xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 2u);
  ASSERT_EQ(dag.getEdges().size(), 1u);

  std::string failureReason;
  auto coupled = deriveStructuredDAGEdgeStrategyPlan(
      dag, {placement(0, 0, {0, 1}), placement(1, 0, {0, 1})}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(coupled)) << failureReason;
  EXPECT_TRUE(coupled->strategies.empty());
  EXPECT_EQ(coupled->totalPeerBytes, 0u);

  auto disjoint = deriveStructuredDAGEdgeStrategyPlan(
      dag, {placement(0, 0, {0, 1}), placement(1, 0, {2, 3})}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(disjoint)) << failureReason;
  EXPECT_TRUE(disjoint->strategies.empty());
  EXPECT_EQ(disjoint->totalPeerBytes, 0u);

  // The typed exact-demand query still proves the init dependency with its
  // exact demand, in every placement.
  analysis::IREpoch epoch = analysis::IREpoch::mint();
  StructuredDAGExactDemandQuery demandQuery(dag, epoch);
  for (auto placements :
       {llvm::SmallVector<StructuredDAGNodePlacement, 2>{
            placement(0, 0, {0, 1}), placement(1, 0, {0, 1})},
        llvm::SmallVector<StructuredDAGNodePlacement, 2>{
            placement(0, 0, {0, 1}), placement(1, 0, {2, 3})}}) {
    std::string trialFailure;
    auto trial = buildLogicalShardTrial(dag, placements, epoch, &trialFailure);
    ASSERT_TRUE(mlir::succeeded(trial)) << trialFailure;
    analysis::ExactDemandResult demand = demandQuery.query(0, *trial);
    EXPECT_EQ(demand.status, analysis::ExactDemandStatus::Satisfied)
        << demand.detail;
    EXPECT_EQ(demand.dependencyKind, analysis::DemandEdgeKind::InitInput);
  }
}

TEST_F(StructuredDAGEdgeStrategyPlanTest,
       SupportChainEdgesCarryNoSpatialActionInAnyPlacement) {
  // A DAG edge that reaches the nearest structured producer through a pure
  // support chain (here tensor.expand_shape) is not a direct current-SSA
  // edge. The typed exact-demand query proves the composed support relation
  // with its exact demand; the canonical carrier still emits no strategy:
  // the card-materialization contract resolves the chain inside the
  // consumer's typed lowering, so the plan must not derive a strategy for
  // it, whether the nodes share Tiles or are disjoint.
  auto module = parse(R"mlir(
module {
  func.func @chain(%input: tensor<8x8xf16>) -> tensor<2x4x8xf16> {
    %out0 = tensor.empty() : tensor<8x8xf16>
    %out1 = tensor.empty() : tensor<2x4x8xf16>
    %producer = linalg.map ins(%input : tensor<8x8xf16>)
        outs(%out0 : tensor<8x8xf16>) (%value: f16) {
      %result = arith.addf %value, %value : f16
      linalg.yield %result : f16
    }
    %expanded = tensor.expand_shape %producer [[0, 1], [2]]
        output_shape [2, 4, 8]
        : tensor<8x8xf16> into tensor<2x4x8xf16>
    %consumer = linalg.map ins(%expanded : tensor<2x4x8xf16>)
        outs(%out1 : tensor<2x4x8xf16>) (%value: f16) {
      %result = arith.mulf %value, %value : f16
      linalg.yield %result : f16
    }
    return %consumer : tensor<2x4x8xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  StructuredDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 2u);
  ASSERT_EQ(dag.getEdges().size(), 1u);

  std::string failureReason;
  auto coupled = deriveStructuredDAGEdgeStrategyPlan(
      dag, {placement(0, 0, {0, 1}), placement(1, 0, {0, 1})}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(coupled)) << failureReason;
  EXPECT_TRUE(coupled->strategies.empty());
  EXPECT_EQ(coupled->totalPeerBytes, 0u);

  auto disjoint = deriveStructuredDAGEdgeStrategyPlan(
      dag, {placement(0, 0, {0, 1}), placement(1, 0, {2, 3})}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(disjoint)) << failureReason;
  EXPECT_TRUE(disjoint->strategies.empty());
  EXPECT_EQ(disjoint->totalPeerBytes, 0u);

  // The typed exact-demand query proves the composed support relation with
  // its exact demand, in every placement.
  analysis::IREpoch epoch = analysis::IREpoch::mint();
  StructuredDAGExactDemandQuery demandQuery(dag, epoch);
  for (auto placements :
       {llvm::SmallVector<StructuredDAGNodePlacement, 2>{
            placement(0, 0, {0, 1}), placement(1, 0, {0, 1})},
        llvm::SmallVector<StructuredDAGNodePlacement, 2>{
            placement(0, 0, {0, 1}), placement(1, 0, {2, 3})}}) {
    std::string trialFailure;
    auto trial = buildLogicalShardTrial(dag, placements, epoch, &trialFailure);
    ASSERT_TRUE(mlir::succeeded(trial)) << trialFailure;
    analysis::ExactDemandResult demand = demandQuery.query(0, *trial);
    EXPECT_EQ(demand.status, analysis::ExactDemandStatus::Satisfied)
        << demand.detail;
    EXPECT_EQ(demand.dependencyKind, analysis::DemandEdgeKind::DataInput);
  }
}

} // namespace
