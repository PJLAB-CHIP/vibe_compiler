//===- ScheduledRankFinalizationTest.cpp ---------------------------------===//

#include "../../lib/Wafer/Compiler/ScheduledRankFinalization.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

class ScheduledRankFinalizationTest : public ::testing::Test {
protected:
  ScheduledRankFinalizationTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> candidateWithSPMElements(int64_t elements) {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  func.func @main(%boundary: memref<1xf16, #wafer.memory<ddr, tensor>>) {
    %unused = wafer.tile.region(%boundary
        : memref<1xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<1xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%ddr: memref<1xf16, #wafer.memory<ddr, tensor>>):
      %zero = arith.constant 0.000000e+00 : f16
      %spm = memref.alloc() : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>
      wafer.instr.fill %spm, %zero
          : memref<)mlir"
       << elements << R"mlir(xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.local_fence
      wafer.tile.yield %ddr
          : memref<1xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
})mlir";
    return mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> candidateWithDDRPlacement() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.ddr.offset = #wafer.ddr_offset<0>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(ScheduledRankFinalizationTest,
       FiltersFailedAlternativeAndRecomputesSurvivorCost) {
  mlir::OwningOpRef<mlir::ModuleOp> overflow =
      candidateWithSPMElements(/*elements=*/2'000'000);
  mlir::OwningOpRef<mlir::ModuleOp> valid =
      candidateWithSPMElements(/*elements=*/128);
  ASSERT_TRUE(overflow);
  ASSERT_TRUE(valid);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });

  std::vector<wafer::ScheduledRankCandidate> frontier;
  frontier.emplace_back(std::move(overflow), /*estimatedTimePs=*/111,
                        /*discoveryOrder=*/3,
                        /*reservedBaseline=*/false);
  frontier.emplace_back(std::move(valid), /*estimatedTimePs=*/999,
                        /*discoveryOrder=*/4,
                        /*reservedBaseline=*/true);
  mlir::FailureOr<std::vector<wafer::compiler::detail::FinalizedRankCandidate>>
      finalized =
          wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
              std::move(frontier));

  ASSERT_TRUE(mlir::succeeded(finalized)) << diagnostics;
  ASSERT_EQ(finalized->size(), 1u);
  EXPECT_EQ(finalized->front().discoveryOrder, 4);
  EXPECT_TRUE(finalized->front().reservedBaseline);
  EXPECT_NE(finalized->front().estimatedTimePs, 999);
  EXPECT_GT(finalized->front().estimatedTimePs, 0);
  EXPECT_NE(diagnostics.find("capacity_overflow"), std::string::npos)
      << diagnostics;
}

TEST_F(ScheduledRankFinalizationTest, FailsOnlyWhenNoAlternativeSurvives) {
  mlir::OwningOpRef<mlir::ModuleOp> overflow =
      candidateWithSPMElements(/*elements=*/2'000'000);
  ASSERT_TRUE(overflow);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  std::vector<wafer::ScheduledRankCandidate> frontier;
  frontier.emplace_back(std::move(overflow), /*estimatedTimePs=*/111,
                        /*discoveryOrder=*/3,
                        /*reservedBaseline=*/true);

  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
          std::move(frontier))));
  EXPECT_NE(diagnostics.find("capacity_overflow"), std::string::npos)
      << diagnostics;
}

TEST_F(ScheduledRankFinalizationTest,
       RejectsWholeVariantPlacementBeforeRankFinalization) {
  mlir::OwningOpRef<mlir::ModuleOp> placed = candidateWithDDRPlacement();
  ASSERT_TRUE(placed);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  std::vector<wafer::ScheduledRankCandidate> frontier;
  frontier.emplace_back(std::move(placed), /*estimatedTimePs=*/0,
                        /*discoveryOrder=*/5,
                        /*reservedBaseline=*/true);

  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
          std::move(frontier))));
  EXPECT_NE(diagnostics.find("rank_frontier_contains_whole_variant_facts"),
            std::string::npos)
      << diagnostics;
}

} // namespace
