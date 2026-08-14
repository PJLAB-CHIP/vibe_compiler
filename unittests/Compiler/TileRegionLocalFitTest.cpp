//===- TileRegionLocalFitTest.cpp --------------------------------------===//

#include "../../lib/Wafer/Compiler/TileRegionLocalFit.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

class TileRegionLocalFitTest : public ::testing::Test {
protected:
  TileRegionLocalFitTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(unsigned elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  func.func @main(%boundary: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    %unused = wafer.tile.region(
        %boundary : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr: memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>):
      %zero = arith.constant 0.000000e+00 : f16
      %spm = memref.alloc()
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.load %ddr into %spm
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
            into memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.tile.fill %spm, %zero
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>, f16
      wafer.tile.yield %ddr
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
})mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(TileRegionLocalFitTest, DistinguishesFitAndExactCapacityRejection) {
  auto fitting = parse(/*elements=*/16);
  auto oversized = parse(/*elements=*/2000000);
  ASSERT_TRUE(fitting);
  ASSERT_TRUE(oversized);
  wafer::TileRegionOp fittingRegion;
  wafer::TileRegionOp oversizedRegion;
  fitting->walk([&](wafer::TileRegionOp region) { fittingRegion = region; });
  oversized->walk(
      [&](wafer::TileRegionOp region) { oversizedRegion = region; });
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::TileRegionToInstrLoweringSession loweringSession(*context);

  auto fit = wafer::compiler::detail::evaluateTileRegionLocalFit(
      fittingRegion, loweringSession, diagnostics);
  auto rejected = wafer::compiler::detail::evaluateTileRegionLocalFit(
      oversizedRegion, loweringSession, diagnostics);
  diagnostics.flush();

  EXPECT_TRUE(fit.fits());
  EXPECT_TRUE(rejected.isProvenExactRejection());
  EXPECT_EQ(rejected.gate, "spm-allocation");
  EXPECT_EQ(rejected.finalization.spmPlanningFailureKind,
            wafer::SPMMemoryPlanningFailureKind::CapacityOverflow);
  EXPECT_NE(diagnosticText.find("outcome=exact-rejection"), std::string::npos);
}

TEST_F(TileRegionLocalFitTest, MissingRegionRemainsIndeterminate) {
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  wafer::TileRegionToInstrLoweringSession loweringSession(*context);
  auto result = wafer::compiler::detail::evaluateTileRegionLocalFit(
      wafer::TileRegionOp{}, loweringSession, diagnostics);
  EXPECT_EQ(
      result.status,
      wafer::compiler::detail::TileRegionLocalFitStatus::IndeterminateFailure);
  EXPECT_EQ(result.gate, "local-fit-contract");
}

TEST_F(TileRegionLocalFitTest, UnsupportedLifetimeRemainsIndeterminate) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %boundary: memref<16xf16, #wafer.memory<ddr, tensor>>,
      %condition: i1) {
    %unused = wafer.tile.region(
        %boundary, %condition
        : memref<16xf16, #wafer.memory<ddr, tensor>>, i1) ->
        (memref<16xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr: memref<16xf16, #wafer.memory<ddr, tensor>>, %initial: i1):
      %loop = scf.while (%arg = %initial) : (i1) -> i1 {
        scf.condition(%arg) %arg : i1
      } do {
      ^bb0(%arg: i1):
        scf.yield %arg : i1
      }
      wafer.tile.yield %ddr
          : memref<16xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  wafer::TileRegionOp region;
  module->walk([&](wafer::TileRegionOp candidate) { region = candidate; });
  ASSERT_TRUE(region);

  wafer::TileRegionToInstrLoweringSession loweringSession(*context);
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto result = wafer::compiler::detail::evaluateTileRegionLocalFit(
      region, loweringSession, diagnostics);
  diagnostics.flush();

  EXPECT_EQ(
      result.status,
      wafer::compiler::detail::TileRegionLocalFitStatus::IndeterminateFailure);
  EXPECT_EQ(result.finalization.spmPlanningFailureKind,
            wafer::SPMMemoryPlanningFailureKind::UnsupportedLifetime);
  EXPECT_NE(diagnosticText.find("outcome=indeterminate"), std::string::npos);
}

TEST_F(TileRegionLocalFitTest,
       FailedRegionConversionRollsBackIRAndReportsStableStageFailure) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%token: i1) {
    %unused_result = wafer.tile.region(%token : i1) -> (i1) {
    ^bb0(%unused: i1):
      %source = memref.alloc()
          : memref<2x2xf32, #wafer.memory<spm, cx>>
      %reduced = wafer.tile.reduce <avg> %source
          {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f32}
          : (memref<2x2xf32, #wafer.memory<spm, cx>>)
         -> memref<2xf32, #wafer.memory<spm, cx>>
      wafer.tile.yield %unused : i1
    }
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  wafer::TileRegionOp region;
  module->walk([&](wafer::TileRegionOp candidate) { region = candidate; });
  ASSERT_TRUE(region);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  region.print(beforeStream);
  beforeStream.flush();

  std::string failureReason;
  EXPECT_TRUE(
      mlir::failed(wafer::convertTileRegionToInstr(region, &failureReason)));
  EXPECT_EQ(failureReason, "tile-region to instruction conversion failed");

  std::string after;
  llvm::raw_string_ostream afterStream(after);
  region.print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
  EXPECT_TRUE(wafer::containsTileDataflowOperations(region));
}

} // namespace
