//===- ScheduledRankFinalizationTest.cpp ---------------------------------===//

#include "../../lib/Wafer/Compiler/ScheduledRankFinalization.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
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
      wafer.instr.ncc_join [0]
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
       FiltersFailedAlternativeAndClosesSurvivorExactCost) {
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
  frontier.emplace_back(
      std::move(overflow), /*stableOrdinal=*/3, wafer::RankArtifactKind::Spill,
      /*reservedBaseline=*/false, wafer::RankBufferingKind::StaticFixedSlot,
      /*bufferingPlanOrdinal=*/2);
  frontier.emplace_back(std::move(valid), /*stableOrdinal=*/4,
                        wafer::RankArtifactKind::Spill,
                        /*reservedBaseline=*/true);
  mlir::FailureOr<std::vector<wafer::compiler::detail::FinalizedRankCandidate>>
      finalized =
      wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
              std::move(frontier));

  ASSERT_TRUE(mlir::succeeded(finalized)) << diagnostics;
  ASSERT_EQ(finalized->size(), 1u);
  EXPECT_EQ(finalized->front().stableOrdinal, 4);
  EXPECT_EQ(finalized->front().artifactKind, wafer::RankArtifactKind::Spill);
  EXPECT_TRUE(finalized->front().reservedBaseline);
  EXPECT_NE(diagnostics.find("capacity_overflow"), std::string::npos)
      << diagnostics;
}

TEST_F(ScheduledRankFinalizationTest,
       PreservesFixedSlotBufferingIdentityForSurvivingAlternative) {
  mlir::OwningOpRef<mlir::ModuleOp> baseline =
      candidateWithSPMElements(/*elements=*/128);
  mlir::OwningOpRef<mlir::ModuleOp> fixed =
      candidateWithSPMElements(/*elements=*/256);
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(fixed);

  std::vector<wafer::ScheduledRankCandidate> frontier;
  frontier.emplace_back(std::move(baseline), /*stableOrdinal=*/4,
                        wafer::RankArtifactKind::Spill,
                        /*reservedBaseline=*/true);
  frontier.emplace_back(
      std::move(fixed), /*stableOrdinal=*/4, wafer::RankArtifactKind::Spill,
      /*reservedBaseline=*/false, wafer::RankBufferingKind::StaticFixedSlot,
      /*bufferingPlanOrdinal=*/6);

  auto finalized =
      wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
          std::move(frontier));
  ASSERT_TRUE(mlir::succeeded(finalized));
  ASSERT_EQ(finalized->size(), 2u);
  auto fixedCandidate = llvm::find_if(*finalized, [](const auto &candidate) {
    return candidate.bufferingKind == wafer::RankBufferingKind::StaticFixedSlot;
  });
  ASSERT_NE(fixedCandidate, finalized->end());
  EXPECT_EQ(fixedCandidate->stableOrdinal, 4);
  EXPECT_EQ(fixedCandidate->bufferingPlanOrdinal, 6u);
  EXPECT_FALSE(fixedCandidate->reservedBaseline);
}

TEST_F(ScheduledRankFinalizationTest,
       FinalizesBaselineFreeRequestShardAndPreservesCanonicalOrder) {
  mlir::OwningOpRef<mlir::ModuleOp> candidate =
      candidateWithSPMElements(/*elements=*/128);
  ASSERT_TRUE(candidate);

  std::vector<wafer::ScheduledRankCandidate> frontier;
  frontier.emplace_back(
      std::move(candidate), /*stableOrdinal=*/7, wafer::RankArtifactKind::Spill,
      /*reservedBaseline=*/false, wafer::RankBufferingKind::Single,
      /*bufferingPlanOrdinal=*/0, wafer::RankWorkerPlacementKind::Unplaced,
      /*workerPlacementPlanOrdinal=*/0, /*frontierOrderOrdinal=*/91);
  auto finalized =
      wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
          std::move(frontier),
          /*requireReservedBaseline=*/false);
  ASSERT_TRUE(mlir::succeeded(finalized));
  ASSERT_EQ(finalized->size(), 1u);
  EXPECT_FALSE(finalized->front().reservedBaseline);
  EXPECT_EQ(finalized->front().frontierOrderOrdinal, 91u);

  auto empty = wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
      {}, /*requireReservedBaseline=*/false);
  ASSERT_TRUE(mlir::succeeded(empty));
  EXPECT_TRUE(empty->empty());
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
  frontier.emplace_back(std::move(overflow), /*stableOrdinal=*/3,
                        wafer::RankArtifactKind::Spill,
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
  frontier.emplace_back(std::move(placed), /*stableOrdinal=*/5,
                        wafer::RankArtifactKind::Spill,
                        /*reservedBaseline=*/true);

  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
          std::move(frontier))));
  EXPECT_NE(diagnostics.find("rank_frontier_contains_whole_variant_facts"),
            std::string::npos)
      << diagnostics;
}

} // namespace
