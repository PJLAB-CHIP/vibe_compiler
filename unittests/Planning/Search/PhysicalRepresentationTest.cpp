//===- PhysicalRepresentationTest.cpp --------------------------------===//

#include "Wafer/Planning/Search/PhysicalRepresentation.h"

#include "Wafer/Analysis/PhysicalDataflow/PhysicalLayoutRelation.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>

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

struct Prepared {
  std::unique_ptr<CardProgramAnalysis> program;
  analysis::LogicalShardTrial trial;
  CoupledRegionDomain coupledDomain;
  CoupledRegionAssignment coupledAssignment;
  CardTemporalDomain temporalDomain;
  CardTemporalAssignment temporalAssignment;
  CardPhysicalRepresentationDomain representationDomain;
};

mlir::FailureOr<Prepared> prepare(mlir::ModuleOp module,
                                  std::string *failureReason) {
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  auto dag = StructuredDAGAnalysis::create(function, failureReason);
  if (mlir::failed(dag) || dag->getNodes().size() != 1)
    return mlir::failure();
  const StructuredDAGNode &node = dag->getNodes().front();
  auto tiling = mlir::cast<mlir::TilingInterface>(node.operation);
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
  auto topology = TargetTopology::create(module, failureReason);
  if (mlir::failed(coupled) || mlir::failed(temporal) || mlir::failed(topology))
    return mlir::failure();
  CoupledRegionAssignment coupledAssignment = coupled->getFirstAssignment();
  CardTemporalAssignment temporalAssignment = temporal->getFirstAssignment();
  StaticOutputDomains outputDomains;
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(function.getResultTypes().front());
  outputDomains.emplace_back(resultType.getShape());
  mlir::Operation *operation = node.operation;
  StructuredDAGNodeID nodeId = node.id;
  auto program = std::make_unique<CardProgramAnalysis>(
      std::move(*topology), llvm::SmallVector<TileId, 16>{TileId(0)},
      std::move(*dag), std::move(outputDomains),
      llvm::SmallVector<StructuredOperationNodeMapping, 16>{
          {operation, nodeId}},
      epoch);
  auto representation = CardPhysicalRepresentationDomain::create(
      *program, *trial, *coupled, coupledAssignment, *temporal,
      temporalAssignment, failureReason);
  if (mlir::failed(representation))
    return mlir::failure();
  return Prepared{std::move(program),        std::move(*trial),
                  std::move(*coupled),       std::move(coupledAssignment),
                  std::move(*temporal),      std::move(temporalAssignment),
                  std::move(*representation)};
}

constexpr llvm::StringLiteral map = R"mlir(
  func.func @map(%input: tensor<4xf16>) -> tensor<4xf16> {
    %empty = tensor.empty() : tensor<4xf16>
    %result = linalg.map ins(%input : tensor<4xf16>)
        outs(%empty : tensor<4xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    return %result : tensor<4xf16>
  }
)mlir";

TEST(PhysicalRepresentationTest,
     DomainMatchesIndependentValueLayoutCartesianProduct) {
  auto context = createContext();
  auto module = parse(*context, map);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;

  std::set<CardPhysicalRepresentationAssignment> actual;
  CardPhysicalRepresentationAssignment current =
      prepared->representationDomain.getFirstAssignment();
  while (true) {
    ASSERT_TRUE(prepared->representationDomain.contains(current));
    actual.insert(current);
    auto next = prepared->representationDomain.getNextAssignment(current);
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    current = std::move(**next);
  }
  ASSERT_EQ(current.values.size(), 2u);
  llvm::SmallVector<MemLayout, 4> layouts{MemLayout::Tensor, MemLayout::NTensor,
                                          MemLayout::Cx, MemLayout::NCx};
  std::set<CardPhysicalRepresentationAssignment> reference;
  for (MemLayout input : layouts)
    for (MemLayout result : layouts) {
      CardPhysicalRepresentationAssignment assignment =
          prepared->representationDomain.getFirstAssignment();
      assignment.values[0].layout = input;
      assignment.values[1].layout = result;
      if (prepared->representationDomain.contains(assignment))
        reference.insert(std::move(assignment));
    }
  EXPECT_EQ(actual, reference);
  EXPECT_EQ(actual.size(), 16u);
}

TEST(PhysicalRepresentationTest,
     SelectedResultVersionMaterializesAndIsRelationVisible) {
  auto context = createContext();
  auto module = parse(*context, map);
  ASSERT_TRUE(module);
  std::string failureReason;
  auto prepared = prepare(*module, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  CardPhysicalRepresentationAssignment assignment =
      prepared->representationDomain.getFirstAssignment();
  auto resultChoice = llvm::find_if(assignment.values, [](const auto &choice) {
    return choice.role == PhysicalValueRole::Result;
  });
  ASSERT_NE(resultChoice, assignment.values.end());
  resultChoice->layout = MemLayout::Cx;
  ASSERT_TRUE(prepared->representationDomain.contains(assignment));

  auto materialized = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->trial,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, assignment, &failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized->module)));
  EXPECT_EQ(
      [&] {
        unsigned count = 0;
        materialized->module->walk([&](LayoutMaterializeOp) { ++count; });
        return count;
      }(),
      2u);
  bool sawSelectedResult = false;
  for (const StructuredOperationBufferRelation &relation :
       materialized->relations.operationResultBuffers) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(relation.buffer.getType());
    sawSelectedResult |=
        type && getWaferMemoryAttr(type).getLayout() == MemLayout::Cx;
  }
  EXPECT_TRUE(sawSelectedResult);

  CardPhysicalRepresentationAssignment malformed = assignment;
  malformed.values.pop_back();
  EXPECT_FALSE(prepared->representationDomain.contains(malformed));
  const std::string sourceBefore = [&] {
    std::string text;
    llvm::raw_string_ostream stream(text);
    module->print(stream);
    stream.flush();
    return text;
  }();
  auto rejected = materializeCardCoupledRegions(
      *module, *prepared->program, CardId(0), prepared->trial,
      prepared->coupledDomain, prepared->coupledAssignment,
      prepared->temporalDomain, prepared->temporalAssignment,
      prepared->representationDomain, malformed, &failureReason);
  EXPECT_TRUE(mlir::failed(rejected));
  std::string sourceAfter;
  llvm::raw_string_ostream stream(sourceAfter);
  module->print(stream);
  stream.flush();
  EXPECT_EQ(sourceAfter, sourceBefore);
}

TEST(PhysicalRepresentationTest, RejectsUnsupportedPhysicalEncodingDomain) {
  auto context = createContext();
  auto module = parse(*context, R"mlir(
  func.func @map(%input: tensor<4xvector<2xf16>>)
      -> tensor<4xvector<2xf16>> {
    %empty = tensor.empty() : tensor<4xvector<2xf16>>
    %result = linalg.map ins(%input : tensor<4xvector<2xf16>>)
        outs(%empty : tensor<4xvector<2xf16>>) (%value: vector<2xf16>) {
      linalg.yield %value : vector<2xf16>
    }
    return %result : tensor<4xvector<2xf16>>
  }
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(prepare(*module, &failureReason)));
  EXPECT_EQ(failureReason, "physical operand layout domain is empty");
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
}

} // namespace
