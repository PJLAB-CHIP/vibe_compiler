//===- WholeVariantCoordinatorTest.cpp ----------------------------------===//

#include "../../lib/Wafer/Compiler/WholeVariantCoordinator.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

class WholeVariantCoordinatorTest : public ::testing::Test {
protected:
  WholeVariantCoordinatorTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  std::string moduleWithBody(llvm::StringRef body, int64_t rankCount) const {
    std::string source;
    llvm::raw_string_ostream os(source);
    os << R"mlir(module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    if (rankCount == 1)
      os << R"mlir(  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>, policy = "explicit", shape = array<i64: 1>, topology = @default}
)mlir";
    else
      os << R"mlir(  wafer.execution.mesh @default_mesh {axes = ["rank"], endpoints = array<i64>, policy = "all_available", shape = array<i64: 16>, topology = @default}
)mlir";
    os << "  func.func @main() {\n" << body << "\n    return\n  }\n}\n";
    return source;
  }

  wafer::compiler::detail::RankVariantCandidate
  candidate(llvm::StringRef body, int64_t rankCount, int64_t cost,
            int64_t discoveryOrder, bool reservedBaseline = false) {
    std::string source = moduleWithBody(body, rankCount);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
    EXPECT_TRUE(module);
    return {std::move(module), cost, discoveryOrder, reservedBaseline};
  }

  static constexpr llvm::StringLiteral kSendMismatched = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer {peer = 1 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 10, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
)mlir";

  static constexpr llvm::StringLiteral kRecvMismatched = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 11, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
)mlir";

  static constexpr llvm::StringLiteral kSendMatched = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer {peer = 1 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 20, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
)mlir";

  static constexpr llvm::StringLiteral kRecvMatched = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<4xf32, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer {peer = 0 : i64, bytes = 16 : i64, message = #wafer.dte_message<communication = 20, phase = collective_permute, round = 0, slice = 0>} : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
    wafer.instr.dte_wait %token : !async.token
)mlir";

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(WholeVariantCoordinatorTest,
       UsesCoordinatedFallbackBeyondTheBestFirstVisitBound) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  frontiers[0].push_back(candidate(kSendMismatched, 16, 0, 0));
  frontiers[0].push_back(candidate(kSendMatched, 16, 100, 5, true));
  frontiers[1].push_back(candidate(kRecvMismatched, 16, 0, 0));
  frontiers[1].push_back(candidate(kRecvMatched, 16, 100, 5, true));
  for (int rank = 2; rank < 15; ++rank) {
    frontiers[rank].push_back(candidate("", 16, 0, 0));
    frontiers[rank].push_back(candidate("", 16, 100, 5, true));
  }
  // This rank models signature deduplication: its conservative partition is
  // already represented by an earlier discovery order.
  frontiers[15].push_back(candidate("", 16, 0, 2, true));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->ranks.size(), 16u);
  ASSERT_EQ(accepted->selectedDiscoveryOrders.size(), 16u);
  for (int rank = 0; rank < 15; ++rank)
    EXPECT_EQ(accepted->selectedDiscoveryOrders[rank], 5);
  EXPECT_EQ(accepted->selectedDiscoveryOrders[15], 2);

  wafer::InstrDTESendOp send;
  accepted->ranks[0].getModule().walk(
      [&](wafer::InstrDTESendOp operation) { send = operation; });
  wafer::InstrDTERecvOp recv;
  accepted->ranks[1].getModule().walk(
      [&](wafer::InstrDTERecvOp operation) { recv = operation; });
  ASSERT_TRUE(send);
  ASSERT_TRUE(recv);
  ASSERT_TRUE(send.getBinding());
  EXPECT_EQ(send.getBinding(), recv.getBinding());
  EXPECT_EQ(send.getMessage().getCommunicationId(), 20);
}

TEST_F(WholeVariantCoordinatorTest,
       RejectsCompleteFrontierWithoutMutatingCandidateBindings) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  frontiers[0].push_back(candidate(kSendMismatched, 16, 0, 0, true));
  frontiers[1].push_back(candidate(kRecvMismatched, 16, 0, 0, true));
  for (int rank = 2; rank < 16; ++rank)
    frontiers[rank].push_back(candidate("", 16, 0, 0, true));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  EXPECT_TRUE(mlir::failed(accepted));
  EXPECT_NE(diagnosticText.find("reserved all-baseline variant failed"),
            std::string::npos)
      << diagnosticText;
  frontiers[0][0].module->walk([](wafer::InstrDTESendOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
  frontiers[1][0].module->walk([](wafer::InstrDTERecvOp operation) {
    EXPECT_FALSE(operation.getBinding());
  });
}

TEST_F(WholeVariantCoordinatorTest, RetainsBaselineWhenExactCostsAreEqual) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  frontiers[0].push_back(candidate("", 1, 0, 3, true));
  frontiers[0].push_back(candidate("", 1, 0, 1));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->selectedDiscoveryOrders.size(), 1u);
  EXPECT_EQ(accepted->selectedDiscoveryOrders.front(), 3);
  ASSERT_EQ(accepted->selectedReservedBaselines.size(), 1u);
  EXPECT_TRUE(accepted->selectedReservedBaselines.front());
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsStrictlyDominatingAlternativeAfterBaselineAcceptance) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  frontiers[0].push_back(
      candidate("    wafer.instr.local_fence", 1, 100, 5, true));
  frontiers[0].push_back(candidate("", 1, 0, 0));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->selectedDiscoveryOrders.size(), 1u);
  EXPECT_EQ(accepted->selectedDiscoveryOrders.front(), 0);
  ASSERT_EQ(accepted->selectedReservedBaselines.size(), 1u);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
}

TEST_F(WholeVariantCoordinatorTest,
       DoesNotUseAlternativeToMaskReservedBaselineFailure) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  constexpr llvm::StringLiteral oversizedSPM = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<1000000xf32, #wafer.memory<spm, tensor>>
)mlir";
  frontiers[0].push_back(candidate(oversizedSPM, 1, 100, 5, true));
  frontiers[0].push_back(candidate("", 1, 0, 0));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  EXPECT_TRUE(mlir::failed(accepted));
  EXPECT_NE(diagnosticText.find("reserved all-baseline variant failed"),
            std::string::npos)
      << diagnosticText;
}

} // namespace
