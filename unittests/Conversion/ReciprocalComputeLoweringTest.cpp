//===- ReciprocalComputeLoweringTest.cpp -----------------------------===//

#include "Wafer/Analysis/Structured/StructuredDAGAnalysis.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/CoupledTileRegion.h"
#include "Wafer/Driver/CompilationInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <array>
#include <memory>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

struct Parsed {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

Parsed parse(int64_t extent, double numerator) {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x)mlir"
         << extent << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
         << "    %empty = tensor.empty() : tensor<2x" << extent << "x128xf16>\n"
         << R"mlir(    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%input : )mlir"
         << "tensor<2x" << extent << "x128xf16>) outs(%empty : tensor<2x"
         << extent << "x128xf16>) {\n"
         << "      ^bb0(%value: f16, %old: f16):\n"
         << "        %numerator = arith.constant " << numerator << " : f16\n"
         << "        %division = arith.divf %numerator, %value : f16\n"
         << "        linalg.yield %division : f16\n"
         << "    } -> tensor<2x" << extent << "x128xf16>\n"
         << "    return %result : tensor<2x" << extent << "x128xf16>\n"
         << "  }\n}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      stream.str(), mlir::ParserConfig(context.get()));
  return {std::move(context), std::move(module)};
}

unsigned countKind(mlir::Operation *root, ComputeElementwiseKind kind) {
  unsigned count = 0;
  root->walk([&](ComputeElementwiseOp operation) {
    count += operation.getKind() == kind;
  });
  return count;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lower(mlir::ModuleOp source, int64_t extent,
      StructuredComputeImplementation implementation,
      std::string &failureReason) {
  auto function = *source.getOps<mlir::func::FuncOp>().begin();
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  if (mlir::failed(dag) || dag->getNodes().size() != 1)
    return mlir::failure();
  const StructuredDAGNode &node = dag->getNodes().front();
  StructuredNodeShardGroup group;
  group.shards.push_back({node.id, TileId(0), {0, 0, 0}, {2, extent, 128}, {}});
  group.implementations.push_back({node.id, implementation});
  mlir::OwningOpRef<mlir::ModuleOp> card;
  std::array<StructuredOperationNodeMapping, 1> mappings = {
      StructuredOperationNodeMapping{node.operation, node.id}};
  if (mlir::failed(lowerStructuredNodeGroupsToCardModule(
          source, CardId(0), llvm::ArrayRef<TileId>{TileId(0)}, mappings,
          llvm::ArrayRef{group}, card, nullptr, &failureReason)))
    return mlir::failure();
  return card;
}

TEST(ReciprocalComputeLoweringTest,
     ExplicitImplementationLowersAlignedAndRaggedWithoutASearchAxis) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    Parsed source = parse(extent, 1.0);
    ASSERT_TRUE(source.module);
    std::string failureReason;
    auto natural =
        lower(*source.module, extent, StructuredComputeImplementation::Natural,
              failureReason);
    ASSERT_TRUE(mlir::succeeded(natural)) << failureReason;
    auto reciprocal =
        lower(*source.module, extent,
              StructuredComputeImplementation::Reciprocal, failureReason);
    ASSERT_TRUE(mlir::succeeded(reciprocal)) << failureReason;
    EXPECT_GT(
        countKind(natural->get().getOperation(), ComputeElementwiseKind::Div),
        0u);
    EXPECT_EQ(
        countKind(natural->get().getOperation(), ComputeElementwiseKind::Recip),
        0u);
    EXPECT_EQ(countKind(reciprocal->get().getOperation(),
                        ComputeElementwiseKind::Div),
              0u);
    EXPECT_GT(countKind(reciprocal->get().getOperation(),
                        ComputeElementwiseKind::Recip),
              0u);
  }
}

TEST(ReciprocalComputeLoweringTest,
     ExplicitReciprocalRejectsANonUnitNumeratorBeforeCardPublication) {
  Parsed source = parse(1025, 2.0);
  ASSERT_TRUE(source.module);
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(lower(*source.module, 1025,
                                 StructuredComputeImplementation::Reciprocal,
                                 failureReason)));
  EXPECT_NE(failureReason.find("requires exact 1/x"), std::string::npos);
}

} // namespace
