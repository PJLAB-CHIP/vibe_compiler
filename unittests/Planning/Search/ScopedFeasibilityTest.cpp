//===- ScopedFeasibilityTest.cpp -------------------------------------===//

#include "Wafer/Planning/Search/ScopedFeasibility.h"

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <array>
#include <memory>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  registerWaferCoreDialects(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context,
                                        llvm::StringRef body) {
  std::string source = (llvm::Twine(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
)mlir") + body + "\n}")
                           .str();
  return mlir::parseSourceString<mlir::ModuleOp>(source,
                                                 mlir::ParserConfig(&context));
}

std::string print(mlir::Operation *operation) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  operation->print(stream);
  return result;
}

struct Prepared {
  std::unique_ptr<CardProgramAnalysis> program;
  analysis::LogicalShardTrial trial;
  CoupledRegionDomain coupledDomain;
  CoupledRegionAssignment coupledAssignment;
  CardTemporalDomain temporalDomain;
  CardTemporalAssignment temporalAssignment;
};

mlir::FailureOr<Prepared> prepare(mlir::ModuleOp module,
                                  std::string *failureReason) {
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  auto dag = StructuredDAGAnalysis::create(function, failureReason);
  if (mlir::failed(dag) || dag->getNodes().size() != 1)
    return mlir::failure();
  const StructuredDAGNode &node = dag->getNodes().front();
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(node.operation);
  if (!tiling)
    return mlir::failure();
  StructuredDAGNodePlacement placement{
      node.id,
      llvm::SmallVector<uint32_t, 4>(tiling.getLoopIteratorTypes().size(), 1),
      {TileId(0)},
      std::nullopt};
  analysis::IREpoch epoch = analysis::IREpoch::mint();
  auto trial = buildLogicalShardTrial(*dag, llvm::ArrayRef(&placement, 1),
                                      epoch, failureReason);
  if (mlir::failed(trial))
    return mlir::failure();
  auto coupled = CoupledRegionDomain::create(*dag, *trial, failureReason);
  auto temporal = CardTemporalDomain::create(
      *dag, llvm::ArrayRef(&placement, 1), failureReason);
  if (mlir::failed(coupled) || mlir::failed(temporal))
    return mlir::failure();
  auto topology = TargetTopology::create(module, failureReason);
  if (mlir::failed(topology))
    return mlir::failure();
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(function.getResultTypes().front());
  StaticOutputDomains outputDomains;
  outputDomains.emplace_back(resultType.getShape().begin(),
                             resultType.getShape().end());
  mlir::Operation *nodeOperation = node.operation;
  StructuredDAGNodeID nodeId = node.id;
  auto program = std::make_unique<CardProgramAnalysis>(
      std::move(*topology), llvm::SmallVector<TileId, 16>{TileId(0)},
      std::move(*dag), std::move(outputDomains),
      llvm::SmallVector<StructuredOperationNodeMapping, 16>{
          {nodeOperation, nodeId}},
      epoch);
  CoupledRegionAssignment coupledAssignment = coupled->getFirstAssignment();
  CardTemporalAssignment temporalAssignment = temporal->getFirstAssignment();
  return Prepared{std::move(program),   std::move(*trial),
                  std::move(*coupled),  std::move(coupledAssignment),
                  std::move(*temporal), std::move(temporalAssignment)};
}

constexpr llvm::StringLiteral map64 = R"mlir(
  func.func @map(%input: tensor<64xf16>) -> tensor<64xf16> {
    %empty = tensor.empty() : tensor<64xf16>
    %result = linalg.map ins(%input : tensor<64xf16>)
        outs(%empty : tensor<64xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    return %result : tensor<64xf16>
  }
)mlir";

TEST(ScopedFeasibilityTest,
     ReturnsProvenBoundsAndExplicitMissingCoordinatesWithoutMutation) {
  auto context = createContext();
  auto module = parse(*context, map64);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  const std::string before = print(module->getOperation());
  TargetMemoryPolicy memory;
  memory.spmBase = 0;
  memory.spmLimit = 1024;
  memory.spmAlignment = 1;
  std::array<FeasibilityCoordinate, 4> unresolved = {
      FeasibilityCoordinate::EventSchedule,
      FeasibilityCoordinate::PhysicalRepresentation,
      FeasibilityCoordinate::Buffering, FeasibilityCoordinate::DataMovement};
  auto result = analyzeScopedFeasibility(
      *prepared->program, prepared->trial, prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, memory, unresolved);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->kind, ScopedFeasibilityKind::Deferred);
  EXPECT_EQ(result->reason, ScopedFeasibilityReason::MissingCoordinates);
  EXPECT_EQ(
      result->requiredCoordinates,
      (llvm::SmallVector<FeasibilityCoordinate, 4>{
          FeasibilityCoordinate::PhysicalRepresentation,
          FeasibilityCoordinate::DataMovement, FeasibilityCoordinate::Buffering,
          FeasibilityCoordinate::EventSchedule}));
  ASSERT_EQ(result->lowerBounds.size(), 1u);
  EXPECT_EQ(result->lowerBounds.front().minimumRequiredBytes, 128u);
  EXPECT_EQ(print(module->getOperation()), before);
}

TEST(ScopedFeasibilityTest, NonBindingOverfullEstimateCannotReject) {
  auto context = createContext();
  auto module = parse(*context, map64);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  TargetMemoryPolicy memory;
  memory.spmBase = 0;
  memory.spmLimit = 200;
  memory.spmAlignment = 1;
  std::array<FeasibilityCoordinate, 1> unresolved = {
      FeasibilityCoordinate::PhysicalRepresentation};
  auto result = analyzeScopedFeasibility(
      *prepared->program, prepared->trial, prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, memory, unresolved);
  ASSERT_TRUE(mlir::succeeded(result));
  ASSERT_EQ(result->lowerBounds.size(), 1u);
  ASSERT_TRUE(
      result->lowerBounds.front().nonBindingResidencyEstimateBytes.has_value());
  EXPECT_GT(*result->lowerBounds.front().nonBindingResidencyEstimateBytes,
            200u);
  EXPECT_EQ(result->lowerBounds.front().minimumRequiredBytes, 128u);
  EXPECT_EQ(result->kind, ScopedFeasibilityKind::Deferred);
  EXPECT_FALSE(result->rejection);
}

TEST(ScopedFeasibilityTest, RejectsOnlyAProvenSingleBufferOverflow) {
  auto context = createContext();
  auto module = parse(*context, map64);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  TargetMemoryPolicy memory;
  memory.spmBase = 0;
  memory.spmLimit = 64;
  memory.spmAlignment = 1;
  std::array<FeasibilityCoordinate, 4> unresolved = {
      FeasibilityCoordinate::PhysicalRepresentation,
      FeasibilityCoordinate::DataMovement, FeasibilityCoordinate::Buffering,
      FeasibilityCoordinate::EventSchedule};
  auto result = analyzeScopedFeasibility(
      *prepared->program, prepared->trial, prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, memory, unresolved);
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->kind, ScopedFeasibilityKind::ExactRejection);
  EXPECT_EQ(result->reason,
            ScopedFeasibilityReason::MinimumFootprintExceedsSPM);
  ASSERT_TRUE(result->rejection);
  EXPECT_EQ(result->rejection->minimumRequiredBytes, 128u);
  EXPECT_EQ(result->rejection->capacityBytes, 64u);
  EXPECT_EQ(result->rejection->key.node, 0u);
  EXPECT_EQ(result->rejection->key.tile, TileId(0));
  EXPECT_TRUE(result->requiredCoordinates.empty());

  CardTemporalAssignment smaller = prepared->temporalAssignment;
  ASSERT_EQ(smaller.nodes.size(), 1u);
  smaller.nodes.front().iteratorTileSizes = {16};
  smaller.nodes.front().waveLoopOrder = {0};
  ASSERT_TRUE(prepared->temporalDomain.contains(smaller));
  auto sibling = analyzeScopedFeasibility(
      *prepared->program, prepared->trial, prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain, smaller, memory,
      unresolved);
  ASSERT_TRUE(mlir::succeeded(sibling));
  EXPECT_EQ(sibling->kind, ScopedFeasibilityKind::Deferred);
  EXPECT_FALSE(sibling->rejection);
  ASSERT_EQ(sibling->lowerBounds.size(), 1u);
  EXPECT_EQ(sibling->lowerBounds.front().minimumRequiredBytes, 32u);
}

TEST(ScopedFeasibilityTest,
     UnsupportedFootprintIsIndeterminateWithoutDefaults) {
  auto context = createContext();
  auto module = parse(*context, R"mlir(
  func.func @map(%input: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
    %empty = tensor.empty() : tensor<4xcomplex<f32>>
    %result = linalg.map ins(%input : tensor<4xcomplex<f32>>)
        outs(%empty : tensor<4xcomplex<f32>>) (%value: complex<f32>) {
      linalg.yield %value : complex<f32>
    }
    return %result : tensor<4xcomplex<f32>>
  }
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  TargetMemoryPolicy memory;
  auto result = analyzeScopedFeasibility(
      *prepared->program, prepared->trial, prepared->coupledDomain,
      prepared->coupledAssignment, prepared->temporalDomain,
      prepared->temporalAssignment, memory,
      llvm::ArrayRef<FeasibilityCoordinate>{});
  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->kind, ScopedFeasibilityKind::Indeterminate);
  EXPECT_EQ(result->reason, ScopedFeasibilityReason::UnsupportedFootprint);
  EXPECT_FALSE(result->rejection);
}

} // namespace
