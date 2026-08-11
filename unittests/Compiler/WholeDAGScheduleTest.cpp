//===- WholeDAGScheduleTest.cpp -----------------------------------------===//

#include "../../lib/Wafer/Compiler/WholeDAGSchedule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <tuple>

using namespace wafer;
using namespace wafer::compiler::detail;

namespace {

SymbolicWaveClass wave(CardDAGNodeID node, SymbolicWaveKind kind) {
  return {node, kind};
}

llvm::SmallVector<PhysicalTileId, 4>
tiles(std::initializer_list<int64_t> values) {
  llvm::SmallVector<PhysicalTileId, 4> result;
  for (int64_t value : values)
    result.push_back(PhysicalTileId(value));
  return result;
}

class WholeDAGScheduleTest : public ::testing::Test {
protected:
  WholeDAGScheduleTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                    mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::func::FuncOp getOnlyFunction(mlir::ModuleOp module) {
    mlir::func::FuncOp function;
    for (mlir::func::FuncOp candidate : module.getOps<mlir::func::FuncOp>()) {
      if (function)
        return {};
      function = candidate;
    }
    return function;
  }

  CardDAGAnalysis buildDAG(mlir::ModuleOp module) {
    std::string failureReason;
    auto dag = CardDAGAnalysis::create(getOnlyFunction(module), &failureReason);
    EXPECT_TRUE(mlir::succeeded(dag)) << failureReason;
    if (mlir::failed(dag))
      std::abort();
    return std::move(*dag);
  }

  WholeDAGScheduleState buildState(const CardDAGAnalysis &dag,
                                   llvm::ArrayRef<PhysicalTileId> tileIDs) {
    std::string failureReason;
    auto state = WholeDAGScheduleState::create(dag, tileIDs, &failureReason);
    EXPECT_TRUE(mlir::succeeded(state)) << failureReason;
    if (mlir::failed(state))
      std::abort();
    return std::move(*state);
  }

  WholeDAGScheduleEvent runWave(WholeDAGScheduleState &state,
                                SymbolicWaveClass symbolicWave,
                                llvm::ArrayRef<PhysicalTileId> tileIDs,
                                WholeDAGTime duration) {
    std::string failureReason;
    EXPECT_TRUE(mlir::succeeded(state.dispatchReadyWave(
        symbolicWave, tileIDs, duration, &failureReason)))
        << failureReason;
    auto event = state.advanceToNextEvent(&failureReason);
    EXPECT_TRUE(mlir::succeeded(event)) << failureReason;
    return mlir::succeeded(event) ? std::move(*event) : WholeDAGScheduleEvent{};
  }

  static constexpr llvm::StringLiteral kIndependentProgram = R"mlir(
module {
  func.func @independent(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>)
      -> (tensor<8xf32>, tensor<8xf32>) {
    %out0 = tensor.empty() : tensor<8xf32>
    %out1 = tensor.empty() : tensor<8xf32>
    %first = linalg.map ins(%lhs : tensor<8xf32>)
        outs(%out0 : tensor<8xf32>) (%value: f32) {
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    }
    %second = linalg.map ins(%rhs : tensor<8xf32>)
        outs(%out1 : tensor<8xf32>) (%value: f32) {
      %result = arith.mulf %value, %value : f32
      linalg.yield %result : f32
    }
    return %first, %second : tensor<8xf32>, tensor<8xf32>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kChainProgram = R"mlir(
module {
  func.func @chain(%input: tensor<8xf32>) -> tensor<8xf32> {
    %out0 = tensor.empty() : tensor<8xf32>
    %out1 = tensor.empty() : tensor<8xf32>
    %producer = linalg.map ins(%input : tensor<8xf32>)
        outs(%out0 : tensor<8xf32>) (%value: f32) {
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    }
    %consumer = linalg.map ins(%producer : tensor<8xf32>)
        outs(%out1 : tensor<8xf32>) (%value: f32) {
      %result = arith.mulf %value, %value : f32
      linalg.yield %result : f32
    }
    return %consumer : tensor<8xf32>
  }
}
)mlir";

  static constexpr llvm::StringLiteral kDiamondProgram = R"mlir(
module {
  func.func @diamond(%input: tensor<8xf32>) -> tensor<8xf32> {
    %out0 = tensor.empty() : tensor<8xf32>
    %out1 = tensor.empty() : tensor<8xf32>
    %out2 = tensor.empty() : tensor<8xf32>
    %out3 = tensor.empty() : tensor<8xf32>
    %source = linalg.map ins(%input : tensor<8xf32>)
        outs(%out0 : tensor<8xf32>) (%value: f32) {
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    }
    %left = linalg.map ins(%source : tensor<8xf32>)
        outs(%out1 : tensor<8xf32>) (%value: f32) {
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    }
    %right = linalg.map ins(%source : tensor<8xf32>)
        outs(%out2 : tensor<8xf32>) (%value: f32) {
      %result = arith.mulf %value, %value : f32
      linalg.yield %result : f32
    }
    %join = linalg.map ins(%left, %right : tensor<8xf32>, tensor<8xf32>)
        outs(%out3 : tensor<8xf32>) (%lhs: f32, %rhs: f32) {
      %result = arith.addf %lhs, %rhs : f32
      linalg.yield %result : f32
    }
    return %join : tensor<8xf32>
  }
}
)mlir";

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(WholeDAGScheduleTest, BuildsStableDiamondNodeEdgeAndWaveIDs) {
  auto module = parse(kDiamondProgram);
  ASSERT_TRUE(module);
  CardDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 4u);
  ASSERT_EQ(dag.getEdges().size(), 4u);
  ASSERT_EQ(dag.getObservableDependencyComponents().size(), 1u);
  EXPECT_FALSE(dag.supportsIndependentComponentPlacement());
  EXPECT_EQ(dag.getObservableDependencyComponents().front().nodes,
            (llvm::SmallVector<CardDAGNodeID, 4>{0, 1, 2, 3}));
  EXPECT_EQ(dag.getObservableDependencyComponents().front().observableOutputs,
            (llvm::SmallVector<uint32_t, 2>{0}));

  for (auto [index, node] : llvm::enumerate(dag.getNodes())) {
    EXPECT_EQ(node.id, index);
    ASSERT_TRUE(node.operation);
    EXPECT_EQ(node.operation->getName().getStringRef(), "linalg.map");
    EXPECT_EQ(node.waveClasses[0], wave(index, SymbolicWaveKind::Prologue));
    EXPECT_EQ(node.waveClasses[1], wave(index, SymbolicWaveKind::Steady));
    EXPECT_EQ(node.waveClasses[2], wave(index, SymbolicWaveKind::Tail));
    EXPECT_EQ(node.waveClasses[2].getStableOrdinal(), index * 3 + 2);
  }

  llvm::SmallVector<std::tuple<CardDAGNodeID, CardDAGNodeID>, 4> edges;
  for (const CardDAGEdge &edge : dag.getEdges())
    edges.emplace_back(edge.producer, edge.consumer);
  EXPECT_EQ(edges,
            (llvm::SmallVector<std::tuple<CardDAGNodeID, CardDAGNodeID>, 4>{
                {0, 1}, {0, 2}, {1, 3}, {2, 3}}));

  auto secondModule = parse(kDiamondProgram);
  ASSERT_TRUE(secondModule);
  CardDAGAnalysis second = buildDAG(*secondModule);
  ASSERT_EQ(second.getEdges().size(), dag.getEdges().size());
  for (auto [lhs, rhs] : llvm::zip_equal(dag.getEdges(), second.getEdges())) {
    EXPECT_EQ(lhs.id, rhs.id);
    EXPECT_EQ(lhs.producer, rhs.producer);
    EXPECT_EQ(lhs.producerResult, rhs.producerResult);
    EXPECT_EQ(lhs.consumer, rhs.consumer);
    EXPECT_EQ(lhs.consumerOperand, rhs.consumerOperand);
  }

  WholeDAGScheduleState state = buildState(dag, tiles({8, 2, 5}));
  ASSERT_EQ(state.getTileStates().size(), 3u);
  EXPECT_EQ(state.getTileStates()[0].tile, PhysicalTileId(2));
  EXPECT_EQ(state.getTileStates()[1].tile, PhysicalTileId(5));
  EXPECT_EQ(state.getTileStates()[2].tile, PhysicalTileId(8));
  EXPECT_EQ(state.getReadyWaves(), (llvm::SmallVector<SymbolicWaveClass, 8>{
                                       wave(0, SymbolicWaveKind::Prologue)}));
}

TEST_F(WholeDAGScheduleTest,
       ProvesIndependentObservableComponentsWithDifferentOutputShapes) {
  auto module = parse(R"mlir(
module {
  func.func @branches(%lhs: tensor<8xf32>, %rhs: tensor<5x3xf32>)
      -> (tensor<8xf32>, tensor<5x3xf32>) {
    %out0 = tensor.empty() : tensor<8xf32>
    %out1 = tensor.empty() : tensor<5x3xf32>
    %first = linalg.map ins(%lhs : tensor<8xf32>)
        outs(%out0 : tensor<8xf32>) (%value: f32) {
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    }
    %second = linalg.map ins(%rhs : tensor<5x3xf32>)
        outs(%out1 : tensor<5x3xf32>) (%value: f32) {
      %result = arith.mulf %value, %value : f32
      linalg.yield %result : f32
    }
    return %first, %second : tensor<8xf32>, tensor<5x3xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  CardDAGAnalysis dag = buildDAG(*module);
  ASSERT_TRUE(dag.supportsIndependentComponentPlacement());
  ASSERT_EQ(dag.getObservableDependencyComponents().size(), 2u);
  EXPECT_EQ(dag.getObservableDependencyComponents()[0].nodes,
            (llvm::SmallVector<CardDAGNodeID, 4>{0}));
  EXPECT_EQ(dag.getObservableDependencyComponents()[0].observableOutputs,
            (llvm::SmallVector<uint32_t, 2>{0}));
  EXPECT_EQ(dag.getObservableDependencyComponents()[1].nodes,
            (llvm::SmallVector<CardDAGNodeID, 4>{1}));
  EXPECT_EQ(dag.getObservableDependencyComponents()[1].observableOutputs,
            (llvm::SmallVector<uint32_t, 2>{1}));
}

TEST_F(WholeDAGScheduleTest,
       SharedStructuredProducerKeepsObservableBranchesJoint) {
  auto module = parse(R"mlir(
module {
  func.func @fanout(%input: tensor<8xf32>)
      -> (tensor<8xf32>, tensor<8xf32>) {
    %out0 = tensor.empty() : tensor<8xf32>
    %out1 = tensor.empty() : tensor<8xf32>
    %out2 = tensor.empty() : tensor<8xf32>
    %source = linalg.map ins(%input : tensor<8xf32>)
        outs(%out0 : tensor<8xf32>) (%value: f32) {
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    }
    %left = linalg.map ins(%source : tensor<8xf32>)
        outs(%out1 : tensor<8xf32>) (%value: f32) {
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    }
    %right = linalg.map ins(%source : tensor<8xf32>)
        outs(%out2 : tensor<8xf32>) (%value: f32) {
      %result = arith.mulf %value, %value : f32
      linalg.yield %result : f32
    }
    return %left, %right : tensor<8xf32>, tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  CardDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getObservableDependencyComponents().size(), 1u);
  EXPECT_FALSE(dag.supportsIndependentComponentPlacement());
  EXPECT_EQ(dag.getObservableDependencyComponents().front().nodes,
            (llvm::SmallVector<CardDAGNodeID, 4>{0, 1, 2}));
  EXPECT_EQ(dag.getObservableDependencyComponents().front().observableOutputs,
            (llvm::SmallVector<uint32_t, 2>{0, 1}));
}

TEST_F(WholeDAGScheduleTest,
       EffectfulOrphanPreventsIndependentComponentPlacement) {
  auto module = parse(R"mlir(
module {
  func.func @effect(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                    %memory: memref<8xf32>)
      -> (tensor<8xf32>, tensor<8xf32>) {
    %out0 = tensor.empty() : tensor<8xf32>
    %out1 = tensor.empty() : tensor<8xf32>
    %first = linalg.map ins(%lhs : tensor<8xf32>)
        outs(%out0 : tensor<8xf32>) (%value: f32) {
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    }
    linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } outs(%memory : memref<8xf32>) {
      ^bb0(%old: f32):
        linalg.yield %old : f32
    }
    %second = linalg.map ins(%rhs : tensor<8xf32>)
        outs(%out1 : tensor<8xf32>) (%value: f32) {
      %result = arith.mulf %value, %value : f32
      linalg.yield %result : f32
    }
    return %first, %second : tensor<8xf32>, tensor<8xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);
  CardDAGAnalysis dag = buildDAG(*module);
  EXPECT_FALSE(dag.supportsIndependentComponentPlacement());
  EXPECT_EQ(dag.getNodes().size(), 3u);
  EXPECT_TRUE(llvm::any_of(dag.getObservableDependencyComponents(),
                           [](const CardDAGDependencyComponent &component) {
                             return component.observableOutputs.empty();
                           }));
}

TEST_F(WholeDAGScheduleTest,
       DispatchesIndependentOpsAtOneEventDeterministically) {
  auto module = parse(kIndependentProgram);
  ASSERT_TRUE(module);
  CardDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 2u);
  ASSERT_TRUE(dag.getEdges().empty());

  WholeDAGScheduleState first = buildState(dag, tiles({3, 0}));
  WholeDAGScheduleState second = buildState(dag, tiles({0, 3}));
  SymbolicWaveClass lhs = wave(0, SymbolicWaveKind::Prologue);
  SymbolicWaveClass rhs = wave(1, SymbolicWaveKind::Prologue);
  EXPECT_EQ(first.getReadyWaves(),
            (llvm::SmallVector<SymbolicWaveClass, 8>{lhs, rhs}));

  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      first.dispatchReadyWave(lhs, tiles({0}), 4, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      first.dispatchReadyWave(rhs, tiles({3}), 4, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      second.dispatchReadyWave(rhs, tiles({3}), 4, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      second.dispatchReadyWave(lhs, tiles({0}), 4, &failureReason)))
      << failureReason;

  EXPECT_EQ(first.getCurrentTime(), 0u);
  ASSERT_EQ(first.getRunningWaves().size(), 2u);
  EXPECT_EQ(first.getRunningWaves()[0].wave, lhs);
  EXPECT_EQ(first.getRunningWaves()[1].wave, rhs);
  auto firstEvent = first.advanceToNextEvent(&failureReason);
  auto secondEvent = second.advanceToNextEvent(&failureReason);
  ASSERT_TRUE(mlir::succeeded(firstEvent)) << failureReason;
  ASSERT_TRUE(mlir::succeeded(secondEvent)) << failureReason;
  EXPECT_EQ(firstEvent->time, 4u);
  EXPECT_EQ(firstEvent->completedWaves,
            (llvm::SmallVector<SymbolicWaveClass, 4>{lhs, rhs}));
  EXPECT_EQ(firstEvent->completedWaves, secondEvent->completedWaves);
  EXPECT_EQ(firstEvent->readyWaves, secondEvent->readyWaves);
  EXPECT_EQ(firstEvent->readyWaves, (llvm::SmallVector<SymbolicWaveClass, 4>{
                                        wave(0, SymbolicWaveKind::Steady),
                                        wave(1, SymbolicWaveKind::Steady)}));
  EXPECT_EQ(first.getTileState(PhysicalTileId(0))->availableTime, 4u);
  EXPECT_EQ(first.getTileState(PhysicalTileId(3))->availableTime, 4u);
}

TEST_F(WholeDAGScheduleTest,
       SameClassDependenciesPipelineConsumerBeforeProducerTail) {
  auto module = parse(kChainProgram);
  ASSERT_TRUE(module);
  CardDAGAnalysis dag = buildDAG(*module);
  ASSERT_EQ(dag.getNodes().size(), 2u);
  ASSERT_EQ(dag.getEdges().size(), 1u);
  WholeDAGScheduleState state = buildState(dag, tiles({0, 1}));

  SymbolicWaveClass producerP = wave(0, SymbolicWaveKind::Prologue);
  SymbolicWaveClass producerS = wave(0, SymbolicWaveKind::Steady);
  SymbolicWaveClass producerT = wave(0, SymbolicWaveKind::Tail);
  SymbolicWaveClass consumerP = wave(1, SymbolicWaveKind::Prologue);
  SymbolicWaveClass consumerS = wave(1, SymbolicWaveKind::Steady);
  SymbolicWaveClass consumerT = wave(1, SymbolicWaveKind::Tail);

  WholeDAGScheduleEvent producerPDone =
      runWave(state, producerP, tiles({0}), 2);
  EXPECT_EQ(producerPDone.time, 2u);
  EXPECT_EQ(state.getWaveStatus(consumerP), DAGScheduleStatus::Ready);
  EXPECT_NE(state.getWaveStatus(producerT), DAGScheduleStatus::Completed);

  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(
      state.dispatchReadyWave(producerS, tiles({0}), 4, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      state.dispatchReadyWave(consumerP, tiles({1}), 1, &failureReason)))
      << failureReason;
  auto event = state.advanceToNextEvent(&failureReason);
  ASSERT_TRUE(mlir::succeeded(event)) << failureReason;
  EXPECT_EQ(event->time, 3u);
  EXPECT_EQ(state.getWaveStatus(consumerS), DAGScheduleStatus::Waiting);

  event = state.advanceToNextEvent(&failureReason);
  ASSERT_TRUE(mlir::succeeded(event)) << failureReason;
  EXPECT_EQ(event->time, 6u);
  EXPECT_EQ(state.getWaveStatus(producerT), DAGScheduleStatus::Ready);
  EXPECT_EQ(state.getWaveStatus(consumerS), DAGScheduleStatus::Ready);

  ASSERT_TRUE(mlir::succeeded(
      state.dispatchReadyWave(producerT, tiles({0}), 5, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      state.dispatchReadyWave(consumerS, tiles({1}), 1, &failureReason)))
      << failureReason;
  event = state.advanceToNextEvent(&failureReason);
  ASSERT_TRUE(mlir::succeeded(event)) << failureReason;
  EXPECT_EQ(event->time, 7u);
  EXPECT_EQ(state.getWaveStatus(consumerT), DAGScheduleStatus::Waiting);
  EXPECT_EQ(state.getWaveStatus(producerT), DAGScheduleStatus::Running);

  event = state.advanceToNextEvent(&failureReason);
  ASSERT_TRUE(mlir::succeeded(event)) << failureReason;
  EXPECT_EQ(event->time, 11u);
  EXPECT_EQ(state.getWaveStatus(consumerT), DAGScheduleStatus::Ready);
  WholeDAGScheduleEvent consumerTDone =
      runWave(state, consumerT, tiles({1}), 1);
  EXPECT_EQ(consumerTDone.time, 12u);
  EXPECT_TRUE(state.isComplete());
}

TEST_F(WholeDAGScheduleTest, DiamondFanoutAndFaninUseEventReadiness) {
  auto module = parse(kDiamondProgram);
  ASSERT_TRUE(module);
  CardDAGAnalysis dag = buildDAG(*module);
  WholeDAGScheduleState state = buildState(dag, tiles({0, 1, 2}));

  runWave(state, wave(0, SymbolicWaveKind::Prologue), tiles({2}), 1);
  EXPECT_EQ(state.getWaveStatus(wave(1, SymbolicWaveKind::Prologue)),
            DAGScheduleStatus::Ready);
  EXPECT_EQ(state.getWaveStatus(wave(2, SymbolicWaveKind::Prologue)),
            DAGScheduleStatus::Ready);
  EXPECT_EQ(state.getWaveStatus(wave(3, SymbolicWaveKind::Prologue)),
            DAGScheduleStatus::Waiting);

  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(state.dispatchReadyWave(
      wave(1, SymbolicWaveKind::Prologue), tiles({0}), 3, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(state.dispatchReadyWave(
      wave(2, SymbolicWaveKind::Prologue), tiles({1}), 1, &failureReason)))
      << failureReason;
  auto event = state.advanceToNextEvent(&failureReason);
  ASSERT_TRUE(mlir::succeeded(event)) << failureReason;
  EXPECT_EQ(event->time, 2u);
  EXPECT_EQ(state.getWaveStatus(wave(3, SymbolicWaveKind::Prologue)),
            DAGScheduleStatus::Waiting);

  event = state.advanceToNextEvent(&failureReason);
  ASSERT_TRUE(mlir::succeeded(event)) << failureReason;
  EXPECT_EQ(event->time, 4u);
  EXPECT_EQ(state.getWaveStatus(wave(3, SymbolicWaveKind::Prologue)),
            DAGScheduleStatus::Ready);
  EXPECT_NE(state.getWaveStatus(wave(1, SymbolicWaveKind::Tail)),
            DAGScheduleStatus::Completed);
  EXPECT_NE(state.getWaveStatus(wave(2, SymbolicWaveKind::Tail)),
            DAGScheduleStatus::Completed);
}

TEST_F(WholeDAGScheduleTest, RejectsOverlappingTileSetsAtOneEvent) {
  auto module = parse(kIndependentProgram);
  ASSERT_TRUE(module);
  CardDAGAnalysis dag = buildDAG(*module);
  WholeDAGScheduleState state = buildState(dag, tiles({0, 1, 2}));
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(state.dispatchReadyWave(
      wave(0, SymbolicWaveKind::Prologue), tiles({2, 0}), 3, &failureReason)))
      << failureReason;
  EXPECT_TRUE(mlir::failed(state.dispatchReadyWave(
      wave(1, SymbolicWaveKind::Prologue), tiles({2}), 1, &failureReason)));
  EXPECT_TRUE(mlir::succeeded(state.dispatchReadyWave(
      wave(1, SymbolicWaveKind::Prologue), tiles({1}), 1, &failureReason)))
      << failureReason;
  ASSERT_EQ(state.getRunningWaves().size(), 2u);
  EXPECT_EQ(state.getRunningWaves()[1].tiles,
            (llvm::SmallVector<PhysicalTileId, 4>{PhysicalTileId(0),
                                                  PhysicalTileId(2)}));
}

} // namespace
