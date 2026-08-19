//===- CardBaselinePlacementTest.cpp -------------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselinePlacement.h"
#include "TestSupport/CodeGen/CardExecutableTestSupport.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "gtest/gtest.h"

using namespace wafer;
using namespace wafer::compiler::detail;
using namespace wafer::compiler::testing;

namespace {

mlir::FailureOr<StructuredDAGAnalysis> analyze(ParsedProgram &program) {
  if (!program.module)
    return mlir::failure();
  auto functions = program.module->getOps<mlir::func::FuncOp>();
  if (functions.empty())
    return mlir::failure();
  return StructuredDAGAnalysis::create(*functions.begin());
}

llvm::SmallVector<StructuredDAGNodePlacement, 16>
makeUnpartitionedPlacements(const StructuredDAGAnalysis &dag) {
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements;
  for (const StructuredDAGNode &node : dag.getNodes()) {
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(node.operation);
    EXPECT_TRUE(static_cast<bool>(tiling));
    StructuredDAGNodePlacement placement;
    placement.node = node.id;
    placement.tiles = {TileId(0)};
    if (tiling)
      placement.iteratorPartitionFactors.assign(
          tiling.getLoopIteratorTypes().size(), 1);
    placements.push_back(std::move(placement));
  }
  return placements;
}

TEST(CardBaselinePlacementTest, ClosesDependentCoordinateWithExactDemand) {
  ParsedProgram program = parseDependentProgram();
  auto dag = analyze(program);
  ASSERT_TRUE(mlir::succeeded(dag));
  auto placements = makeUnpartitionedPlacements(*dag);
  CardBaselinePlacementVerdict verdict;
  std::string failureReason;
  auto closure = closeCardBaselinePlacement(
      *dag, placements, analysis::IREpoch::mint(), &failureReason, &verdict);
  ASSERT_TRUE(mlir::succeeded(closure)) << failureReason;
  EXPECT_EQ(verdict.status, analysis::ExactDemandStatus::Satisfied);
  ASSERT_EQ(closure->outputPlacements.size(), 1u);
  EXPECT_FALSE(closure->outputPlacements.front().shardDimension);
  EXPECT_EQ(closure->outputPlacements.front().tiles,
            (llvm::SmallVector<TileId, 16>{TileId(0)}));
}

TEST(CardBaselinePlacementTest, ClosesMultiProducerConsumerInput) {
  ParsedProgram program = parseMultiProducerJoinProgram();
  auto dag = analyze(program);
  ASSERT_TRUE(mlir::succeeded(dag));
  ASSERT_GE(dag->getNodes().size(), 2u);
  ASSERT_TRUE(llvm::any_of(dag->getEdges(), [&](const StructuredDAGEdge &edge) {
    return llvm::count_if(dag->getEdges(), [&](const StructuredDAGEdge &other) {
             return other.consumer == edge.consumer &&
                    other.consumerOperand == edge.consumerOperand;
           }) > 1;
  }));
  auto placements = makeUnpartitionedPlacements(*dag);
  CardBaselinePlacementVerdict verdict;
  std::string failureReason;
  auto closure = closeCardBaselinePlacement(
      *dag, placements, analysis::IREpoch::mint(), &failureReason, &verdict);
  ASSERT_TRUE(mlir::succeeded(closure)) << failureReason;
  EXPECT_EQ(verdict.status, analysis::ExactDemandStatus::Satisfied);
}

TEST(CardBaselinePlacementTest, MalformedCoordinateIsIndeterminate) {
  ParsedProgram program = parseDependentProgram();
  auto dag = analyze(program);
  ASSERT_TRUE(mlir::succeeded(dag));
  auto placements = makeUnpartitionedPlacements(*dag);
  ASSERT_GT(placements.size(), 1u);
  placements.pop_back();
  CardBaselinePlacementVerdict verdict;
  std::string failureReason;
  auto closure = closeCardBaselinePlacement(
      *dag, placements, analysis::IREpoch::mint(), &failureReason, &verdict);
  EXPECT_TRUE(mlir::failed(closure));
  EXPECT_EQ(verdict.status,
            analysis::ExactDemandStatus::IndeterminateFailure);
  EXPECT_NE(failureReason.find("requires every DAG node"), std::string::npos);
}

TEST(CardBaselinePlacementTest, CanonicalAxesContainOnlyParallelIterators) {
  ParsedProgram program = parseTwoReductionAxisProgram();
  auto dag = analyze(program);
  ASSERT_TRUE(mlir::succeeded(dag));
  bool sawReductionIterator = false;
  bool sawParallelAxis = false;
  for (const StructuredDAGNode &node : dag->getNodes()) {
    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(node.operation);
    ASSERT_TRUE(static_cast<bool>(tiling));
    auto iteratorTypes = tiling.getLoopIteratorTypes();
    sawReductionIterator |= llvm::is_contained(
        iteratorTypes, mlir::utils::IteratorType::reduction);
    auto axes = getCardBaselineSpatialAxes(node);
    if (!axes)
      continue;
    for (const CardBaselineSpatialAxis &axis : *axes) {
      ASSERT_LT(axis.iteratorDimension, iteratorTypes.size());
      EXPECT_EQ(iteratorTypes[axis.iteratorDimension],
                mlir::utils::IteratorType::parallel);
      sawParallelAxis = true;
    }
  }
  EXPECT_TRUE(sawReductionIterator);
  EXPECT_TRUE(sawParallelAxis);
}

} // namespace
