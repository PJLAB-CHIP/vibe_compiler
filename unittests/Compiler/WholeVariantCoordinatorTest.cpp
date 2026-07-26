//===- WholeVariantCoordinatorTest.cpp ----------------------------------===//

#include "../../lib/Wafer/Compiler/WholeVariantCoordinator.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/ExecutableBundleInternal.h"
#include "../../lib/Wafer/Compiler/ScheduledRankFinalization.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

class WholeVariantCoordinatorTest : public ::testing::Test {
protected:
  WholeVariantCoordinatorTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_shared<mlir::MLIRContext>(registry);
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

  wafer::compiler::detail::RankVariantCandidate candidate(
      llvm::StringRef body, int64_t rankCount, int64_t /*legacyCost*/,
      int64_t stableOrdinal, bool reservedBaseline = false,
      wafer::RankArtifactKind artifactKind = wafer::RankArtifactKind::Spill) {
    std::string source = moduleWithBody(body, rankCount);
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
    EXPECT_TRUE(module);
    return {std::move(module), stableOrdinal, artifactKind, reservedBaseline};
  }

  wafer::frontend::FrontendProgramVerificationResult
  replicated1DProgram(unsigned inputCount, int64_t elementCount,
                      llvm::StringRef dtype, unsigned outputCount = 1,
                      int64_t rankCount = 1) const {
    auto makeBoundary = [&](int64_t index) {
      wafer::frontend::ProgramBoundaryBinding binding;
      binding.index = index;
      binding.programIndex = index;
      binding.distribution =
          wafer::frontend::ProgramDistributionKind::Replicated;
      binding.globalShape = {elementCount};
      binding.localShape = {elementCount};
      binding.dtype = dtype.str();
      for (int64_t rank = 0; rank < rankCount; ++rank) {
        wafer::frontend::ProgramRankSlice slice;
        slice.logicalRank = rank;
        slice.replicaId = rank;
        slice.offsets = {0};
        slice.sizes = {elementCount};
        slice.strides = {1};
        binding.rankSlices.push_back(std::move(slice));
      }
      return binding;
    };

    wafer::frontend::FrontendProgramVerificationResult program;
    program.logicalRankCount = rankCount;
    program.programUserInputCount = inputCount;
    for (unsigned index = 0; index < inputCount; ++index)
      program.distributedInputs.push_back(makeBoundary(index));
    for (unsigned index = 0; index < outputCount; ++index)
      program.distributedOutputs.push_back(makeBoundary(index));
    return program;
  }

  llvm::Expected<wafer::compiler::ExecutableBundle> buildDefaultBundle(
      mlir::ModuleOp source,
      const wafer::frontend::FrontendProgramVerificationResult &program,
      wafer::compiler::ExecutionConfig executionConfig,
      llvm::raw_ostream &diagnostics) const {
    std::string sourceText;
    llvm::raw_string_ostream sourceStream(sourceText);
    source.print(sourceStream);
    sourceStream.flush();

    mlir::DialectRegistry productionRegistry;
    wafer::compiler::detail::registerCompilationDialects(productionRegistry);
    auto productionContext =
        std::make_shared<mlir::MLIRContext>(productionRegistry);
    productionContext->loadAllAvailableDialects();
    mlir::OwningOpRef<mlir::ModuleOp> productionSource =
        mlir::parseSourceString<mlir::ModuleOp>(sourceText,
                                                productionContext.get());
    if (!productionSource)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "cannot reparse production source");
    llvm::Expected<wafer::compiler::ExecutableBundle> bundle =
        wafer::compiler::detail::buildExecutableBundle(
            productionContext, *productionSource, program, executionConfig,
            diagnostics, std::nullopt);
    // buildExecutableBundle moves the context into the successful bundle.
    // Destroy its source while that owner is still alive.
    if (bundle)
      productionSource = nullptr;
    return bundle;
  }

  mlir::FailureOr<wafer::compiler::detail::AcceptedWholeVariant>
  selectProductionVariant(
      mlir::ModuleOp source,
      const wafer::frontend::FrontendProgramVerificationResult &program,
      llvm::raw_ostream &diagnostics,
      std::optional<wafer::analysis::InstructionProgramCost> *baselineCost =
          nullptr,
      int64_t rankCount = 1) {
    std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(
        static_cast<size_t>(rankCount));
    for (int64_t rank = 0; rank < rankCount; ++rank) {
      wafer::TensorProgramSchedulingConfig schedulingConfig;
      schedulingConfig.logicalRank = rank;
      schedulingConfig.candidateParallelism = 1;
      auto scheduled =
          wafer::buildScheduledRankCandidateFrontier(source, schedulingConfig);
      if (mlir::failed(scheduled))
        return mlir::failure();
      auto finalized =
          wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
              std::move(*scheduled));
      if (mlir::failed(finalized))
        return mlir::failure();

      for (wafer::compiler::detail::FinalizedRankCandidate &candidate :
           *finalized) {
        if (baselineCost && rank == 0 && candidate.reservedBaseline)
          *baselineCost = wafer::analysis::analyzeInstructionProgramCost(
              candidate.module.get(),
              wafer::analysis::getTargetScheduleCostPolicy(
                  wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
        frontiers[rank].push_back(
            {std::move(candidate.module), candidate.stableOrdinal,
             candidate.artifactKind, candidate.reservedBaseline});
      }
    }
    auto executionConfig =
        wafer::compiler::ExecutionConfig::createForSingleCard(
            rankCount, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
            wafer::TargetLaunchABIId::perRankPointerBlockV1());
    if (!executionConfig)
      return mlir::failure();
    auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
        frontiers, program, *executionConfig, diagnostics);
    if (mlir::failed(accepted))
      return mlir::failure();

    // Replay the same source through the default executable-bundle owner and
    // require byte-for-byte committed rank IR correspondence. The detailed
    // checks below therefore describe the artifact that wafer-compile commits,
    // not merely a coordinator seam result.
    llvm::Expected<wafer::compiler::ExecutableBundle> production =
        buildDefaultBundle(source, program, *executionConfig, diagnostics);
    if (!production) {
      diagnostics << llvm::toString(production.takeError()) << "\n";
      return mlir::failure();
    }
    if (production->getRankExecutables().size() != accepted->ranks.size())
      return mlir::failure();
    for (auto [expected, committed] :
         llvm::zip_equal(accepted->ranks, production->getRankExecutables())) {
      std::string expectedText;
      llvm::raw_string_ostream expectedStream(expectedText);
      expected.getModule().print(expectedStream);
      std::string committedText;
      llvm::raw_string_ostream committedStream(committedText);
      committed.getModule().print(committedStream);
      if (expectedStream.str() != committedStream.str())
        return mlir::failure();
    }
    return accepted;
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
  std::shared_ptr<mlir::MLIRContext> context;
};

TEST_F(WholeVariantCoordinatorTest,
       UsesCoordinatedFallbackBeyondTheBestFirstVisitBound) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  frontiers[0].push_back(candidate(kSendMismatched, 16, 0, 0));
  frontiers[0].push_back(candidate(kSendMatched, 16, 0, 5));
  frontiers[0].push_back(
      candidate((kSendMatched + "    wafer.instr.local_fence\n").str(), 16, 100,
                9, true));
  frontiers[1].push_back(candidate(kRecvMismatched, 16, 0, 0));
  frontiers[1].push_back(candidate(kRecvMatched, 16, 0, 5));
  frontiers[1].push_back(
      candidate((kRecvMatched + "    wafer.instr.local_fence\n").str(), 16, 100,
                9, true));
  for (int rank = 2; rank < 16; ++rank) {
    frontiers[rank].push_back(candidate("", 16, 0, 0));
    frontiers[rank].push_back(candidate("", 16, 0, 5));
    frontiers[rank].push_back(
        candidate("    wafer.instr.local_fence", 16, 100, 9, true));
  }

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->ranks.size(), 16u);
  ASSERT_EQ(accepted->selectedStableOrdinals.size(), 16u);
  for (int rank = 0; rank < 16; ++rank)
    EXPECT_EQ(accepted->selectedStableOrdinals[rank], 5);

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
       DoesNotMixDistinctGenerationOrdinalsAcrossRanks) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (int rank = 0; rank < 16; ++rank) {
    frontiers[rank].push_back(
        candidate("    wafer.instr.local_fence", 16, 100, 9, true));
    frontiers[rank].push_back(candidate("", 16, 0, rank == 0 ? 0 : 2));
  }

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->selectedStableOrdinals.size(), 16u);
  EXPECT_TRUE(llvm::all_of(accepted->selectedStableOrdinals,
                           [](int64_t ordinal) { return ordinal == 9; }));
  EXPECT_TRUE(llvm::all_of(accepted->selectedReservedBaselines,
                           [](bool reserved) { return reserved; }));
}

TEST_F(WholeVariantCoordinatorTest,
       DoesNotMixPhysicalArtifactKindsAcrossRanks) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));

  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (int rank = 0; rank < 16; ++rank) {
    frontiers[rank].push_back(
        candidate("    wafer.instr.local_fence", 16, 100, 9, true));
    frontiers[rank].push_back(
        candidate("", 16, 0, 0, false,
                  rank == 0 ? wafer::RankArtifactKind::Resident
                            : wafer::RankArtifactKind::ResidentReady));
  }

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->selectedArtifactKinds.size(), 16u);
  EXPECT_TRUE(llvm::all_of(accepted->selectedArtifactKinds,
                           [](wafer::RankArtifactKind kind) {
                             return kind == wafer::RankArtifactKind::Spill;
                           }));
  EXPECT_TRUE(llvm::all_of(accepted->selectedReservedBaselines,
                           [](bool reserved) { return reserved; }));
}

TEST_F(WholeVariantCoordinatorTest,
       RejectsCompleteFrontierWithoutMutatingCandidateBindings) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
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
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
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
  ASSERT_EQ(accepted->selectedStableOrdinals.size(), 1u);
  EXPECT_EQ(accepted->selectedStableOrdinals.front(), 3);
  ASSERT_EQ(accepted->selectedReservedBaselines.size(), 1u);
  EXPECT_TRUE(accepted->selectedReservedBaselines.front());
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsStrictlyDominatingAlternativeAfterBaselineAcceptance) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
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
  ASSERT_EQ(accepted->selectedStableOrdinals.size(), 1u);
  EXPECT_EQ(accepted->selectedStableOrdinals.front(), 0);
  ASSERT_EQ(accepted->selectedReservedBaselines.size(), 1u);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
}

TEST_F(WholeVariantCoordinatorTest,
       ReservedBaselineModeBypassesThePreferredProductionWinner) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  frontiers[0].push_back(
      candidate("    wafer.instr.local_fence", 1, 0, 9, true));
  frontiers[0].push_back(candidate("", 1, 0, 1));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  std::string productionDiagnosticText;
  llvm::raw_string_ostream productionDiagnostics(productionDiagnosticText);
  auto production = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, productionDiagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::Production);
  ASSERT_TRUE(mlir::succeeded(production)) << productionDiagnosticText;
  ASSERT_EQ(production->selectedStableOrdinals.size(), 1u);
  EXPECT_EQ(production->selectedStableOrdinals.front(), 1);
  EXPECT_FALSE(production->selectedReservedBaselines.front());

  std::string baselineDiagnosticText;
  llvm::raw_string_ostream baselineDiagnostics(baselineDiagnosticText);
  auto baseline = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, baselineDiagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::ReservedBaseline);
  ASSERT_TRUE(mlir::succeeded(baseline)) << baselineDiagnosticText;
  ASSERT_EQ(baseline->selectedStableOrdinals.size(), 1u);
  EXPECT_EQ(baseline->selectedStableOrdinals.front(), 9);
  EXPECT_TRUE(baseline->selectedReservedBaselines.front());

  unsigned fences = 0;
  baseline->ranks.front().getModule().walk(
      [&](wafer::SyncLocalFenceOp) { ++fences; });
  EXPECT_EQ(fences, 1u);
}

TEST_F(WholeVariantCoordinatorTest,
       CharacterizationModesRequireExactPerRankCollectiveDTEPhases) {
  using Mode = wafer::compiler::detail::WholeVariantSelectionMode;
  struct Alternative {
    Mode mode;
    llvm::StringRef phase;
    llvm::StringRef secondPhase;
  };
  const Alternative alternatives[] = {
      {Mode::CharacterizeAllGatherDirect, "all_gather_direct", {}},
      {Mode::CharacterizeAllGatherRing, "all_gather_ring", {}},
      {Mode::CharacterizeReduceScatterDirect, "reduce_scatter_direct", {}},
      {Mode::CharacterizeReduceScatterRing, "reduce_scatter_ring", {}},
      {Mode::CharacterizeAllReduceRing, "all_reduce_ring", {}},
      {Mode::CharacterizeAllReduceTree, "all_reduce_tree_reduce",
       "all_reduce_tree_broadcast"},
  };
  auto bodyFor = [](int64_t rank, const Alternative &alternative,
                    int64_t communication) {
    std::string body;
    llvm::raw_string_ostream os(body);
    os << "    %buffer = memref.alloc() {wafer.spm.offset = "
          "#wafer.spm_offset<65536>} : memref<4xf32, "
          "#wafer.memory<spm, tensor>>\n";
    auto issue = [&](bool send, int64_t peer, llvm::StringRef phase,
                     int64_t round) {
      os << "    %token" << round << " = wafer.instr.dte_"
         << (send ? "send" : "recv") << " %buffer {peer = " << peer
         << " : i64, bytes = 16 : i64, message = "
            "#wafer.dte_message<communication = "
         << communication << ", phase = " << phase << ", round = " << round
         << ", slice = 0>} : memref<4xf32, "
            "#wafer.memory<spm, tensor>> -> !async.token\n"
         << "    wafer.instr.dte_wait %token" << round << " : !async.token\n";
    };
    int64_t peer = rank % 2 == 0 ? rank + 1 : rank - 1;
    issue(/*send=*/rank % 2 == 0, peer, alternative.phase, /*round=*/0);
    if (!alternative.secondPhase.empty())
      issue(/*send=*/rank % 2 != 0, peer, alternative.secondPhase,
            /*round=*/1);
    return body;
  };

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (int64_t rank = 0; rank < 16; ++rank)
    for (auto [ordinal, alternative] : llvm::enumerate(alternatives))
      frontiers[rank].push_back(candidate(
          bodyFor(rank, alternative, 100 + static_cast<int64_t>(ordinal)), 16,
          0, static_cast<int64_t>(ordinal),
          /*reservedBaseline=*/ordinal == 0));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  for (auto [ordinal, alternative] : llvm::enumerate(alternatives)) {
    std::string diagnosticText;
    llvm::raw_string_ostream diagnostics(diagnosticText);
    auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
        frontiers, program, *config, diagnostics, alternative.mode);
    ASSERT_TRUE(mlir::succeeded(accepted)) << ordinal << ": " << diagnosticText;
    ASSERT_EQ(accepted->selectedStableOrdinals.size(), 16u);
    EXPECT_TRUE(
        llvm::all_of(accepted->selectedStableOrdinals, [&](int64_t selected) {
          return selected == static_cast<int64_t>(ordinal);
        }));
  }

  auto expectRejected = [&](auto &invalidFrontiers, llvm::StringRef reason) {
    std::string diagnosticText;
    llvm::raw_string_ostream diagnostics(diagnosticText);
    auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
        invalidFrontiers, program, *config, diagnostics,
        Mode::CharacterizeAllGatherDirect);
    EXPECT_TRUE(mlir::failed(accepted))
        << reason.str() << ": " << diagnosticText;
    EXPECT_NE(diagnosticText.find("no fully accepted whole variant matches"),
              std::string::npos)
        << reason.str() << ": " << diagnosticText;
  };

  std::vector<wafer::compiler::detail::RankVariantFrontier> missing(16);
  for (int64_t rank = 0; rank < 16; ++rank)
    missing[rank].push_back(candidate(
        rank < 14 ? bodyFor(rank, alternatives[0], 200) : "", 16, 0, 0,
        /*reservedBaseline=*/true));
  expectRejected(missing, "two accepted ranks omit the requested phase");

  Alternative mixed = {Mode::CharacterizeAllGatherDirect,
                       "all_gather_direct", "all_gather_ring"};
  std::vector<wafer::compiler::detail::RankVariantFrontier> mixedFrontiers(16);
  for (int64_t rank = 0; rank < 16; ++rank)
    mixedFrontiers[rank].push_back(candidate(
        bodyFor(rank, mixed, 201), 16, 0, 0,
        /*reservedBaseline=*/true));
  expectRejected(mixedFrontiers,
                 "every accepted rank mixes direct and ring phases");

  const llvm::StringRef unrelatedPhases[] = {
      "reduce_scatter_direct", "collective_permute", "all_to_all"};
  for (auto [index, phase] : llvm::enumerate(unrelatedPhases)) {
    Alternative unrelated = {Mode::CharacterizeAllGatherDirect,
                             "all_gather_direct", phase};
    std::vector<wafer::compiler::detail::RankVariantFrontier>
        unrelatedFrontiers(16);
    for (int64_t rank = 0; rank < 16; ++rank)
      unrelatedFrontiers[rank].push_back(candidate(
          bodyFor(rank, unrelated, 202 + static_cast<int64_t>(index)), 16, 0,
          0, /*reservedBaseline=*/true));
    expectRejected(unrelatedFrontiers,
                   "every accepted rank mixes unrelated collective phases");
  }
}

TEST_F(WholeVariantCoordinatorTest,
       CharacterizationModeFailsWhenAcceptedPhaseIsMissing) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  frontiers[0].push_back(candidate("", 1, 0, 0, true));
  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics,
      wafer::compiler::detail::WholeVariantSelectionMode::
          CharacterizeAllReduceRing);
  EXPECT_TRUE(mlir::failed(accepted));
  EXPECT_NE(diagnosticText.find("no fully accepted whole variant matches"),
            std::string::npos)
      << diagnosticText;
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsFinalParetoWinnerInsteadOfFirstBaselineImprovement) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  frontiers[0].push_back(
      candidate("    wafer.instr.local_fence\n    wafer.instr.local_fence", 1,
                200, 5, true));
  // The earlier stable ordinal is visited first, but the later candidate has a
  // strictly lower final instruction count and must replace it on the exact
  // Pareto frontier.
  frontiers[0].push_back(candidate("    wafer.instr.local_fence", 1, 0, 0));
  frontiers[0].push_back(candidate("", 1, 100, 1));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->selectedStableOrdinals.size(), 1u);
  EXPECT_EQ(accepted->selectedStableOrdinals.front(), 1);
}

TEST_F(WholeVariantCoordinatorTest,
       UsesValidatedSPMHighWaterInExactParetoSelection) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));
  constexpr llvm::StringLiteral highWater = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<131072>} : memref<4xf32, #wafer.memory<spm, tensor>>
)mlir";
  constexpr llvm::StringLiteral lowWater = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<4xf32, #wafer.memory<spm, tensor>>
)mlir";
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  frontiers[0].push_back(candidate(highWater, 1, 0, 5, true));
  frontiers[0].push_back(candidate(lowWater, 1, 0, 0));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->selectedStableOrdinals.size(), 1u);
  EXPECT_EQ(accepted->selectedStableOrdinals.front(), 0);
}

TEST_F(WholeVariantCoordinatorTest,
       StaticPolicyAcceptsKnownExecutionGainWithinSPMCapacity) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));
  constexpr llvm::StringLiteral lowWaterWithFence = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.instr.local_fence
)mlir";
  constexpr llvm::StringLiteral highWaterWithoutFence = R"mlir(
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<131072>} : memref<4xf32, #wafer.memory<spm, tensor>>
)mlir";
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  frontiers[0].push_back(candidate(lowWaterWithFence, 1, 0, 5, true));
  frontiers[0].push_back(candidate(highWaterWithoutFence, 1, 0, 0));

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->selectedStableOrdinals.size(), 1u);
  EXPECT_EQ(accepted->selectedStableOrdinals.front(), 0);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsTargetReciprocalFromProductionRankFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%input: tensor<16xf32>) -> tensor<16xf32> {
    %out = tensor.empty() : tensor<16xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<16xf32>) outs(%out : tensor<16xf32>) {
    ^bb0(%value: f32, %init: f32):
      %one = arith.constant 1.0 : f32
      %reciprocal = arith.divf %one, %value : f32
      linalg.yield %reciprocal : f32
    } -> tensor<16xf32>
    return %result : tensor<16xf32>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig schedulingConfig;
  schedulingConfig.logicalRank = 0;
  schedulingConfig.candidateParallelism = 1;
  auto scheduled =
      wafer::buildScheduledRankCandidateFrontier(*source, schedulingConfig);
  ASSERT_TRUE(mlir::succeeded(scheduled));
  auto finalized =
      wafer::compiler::detail::finalizeScheduledRankCandidateFrontier(
          std::move(*scheduled));
  ASSERT_TRUE(mlir::succeeded(finalized));

  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(1);
  std::optional<uint64_t> baselineInstructions;
  std::optional<uint64_t> reciprocalInstructions;
  std::optional<uint64_t> baselineHighWater;
  std::optional<uint64_t> reciprocalHighWater;
  for (wafer::compiler::detail::FinalizedRankCandidate &candidate :
       *finalized) {
    bool hasReciprocal = false;
    candidate.module->walk([&](wafer::InstrElementwiseOp elementwise) {
      hasReciprocal |=
          elementwise.getKind() == wafer::InstrElementwiseKind::Recip;
    });
    wafer::analysis::InstructionProgramCost exact =
        wafer::analysis::analyzeInstructionProgramCost(
            candidate.module.get(),
            wafer::analysis::getTargetScheduleCostPolicy(
                wafer::TargetProfileId::waferTx81SingleCardKernelV1()));
    if (candidate.reservedBaseline) {
      baselineInstructions = exact.instructionCount.value;
      baselineHighWater = exact.spmHighWaterBytes.value;
    }
    if (hasReciprocal &&
        (!reciprocalInstructions ||
         exact.instructionCount.value < *reciprocalInstructions)) {
      reciprocalInstructions = exact.instructionCount.value;
      reciprocalHighWater = exact.spmHighWaterBytes.value;
    }
    frontiers[0].push_back({std::move(candidate.module),
                            candidate.stableOrdinal, candidate.artifactKind,
                            candidate.reservedBaseline});
  }
  ASSERT_TRUE(baselineInstructions);
  ASSERT_TRUE(reciprocalInstructions);
  ASSERT_TRUE(baselineHighWater);
  ASSERT_TRUE(reciprocalHighWater);
  EXPECT_LT(*reciprocalInstructions, *baselineInstructions);
  EXPECT_LE(*reciprocalHighWater, *baselineHighWater);

  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(config));
  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  auto makeBoundary = [](int64_t index) {
    wafer::frontend::ProgramBoundaryBinding binding;
    binding.index = index;
    binding.programIndex = index;
    binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    binding.globalShape = {16};
    binding.localShape = {16};
    binding.dtype = "f32";
    wafer::frontend::ProgramRankSlice slice;
    slice.logicalRank = 0;
    slice.replicaId = 0;
    slice.offsets = {0};
    slice.sizes = {16};
    slice.strides = {1};
    binding.rankSlices.push_back(std::move(slice));
    return binding;
  };
  program.programUserInputCount = 1;
  program.distributedInputs = {makeBoundary(0)};
  wafer::frontend::ProgramBoundaryBinding output = makeBoundary(0);
  program.distributedOutputs.push_back(std::move(output));
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = wafer::compiler::detail::selectAcceptedWholeVariant(
      frontiers, program, *config, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->ranks.size(), 1u);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());

  bool sawReciprocal = false;
  bool sawDivision = false;
  accepted->ranks.front().getModule().walk(
      [&](wafer::InstrElementwiseOp elementwise) {
        sawReciprocal |=
            elementwise.getKind() == wafer::InstrElementwiseKind::Recip;
        sawDivision |=
            elementwise.getKind() == wafer::InstrElementwiseKind::Div;
      });
  EXPECT_TRUE(sawReciprocal);
  EXPECT_FALSE(sawDivision);

  llvm::Expected<wafer::compiler::ExecutableBundle> production =
      buildDefaultBundle(*source, program, *config, diagnostics);
  ASSERT_TRUE(static_cast<bool>(production))
      << diagnosticText
      << (production ? "" : llvm::toString(production.takeError()));
  ASSERT_EQ(production->getRankExecutables().size(), 1u);
  bool committedReciprocal = false;
  bool committedDivision = false;
  production->getRankExecutables().front().getModule().walk(
      [&](wafer::InstrElementwiseOp elementwise) {
        committedReciprocal |=
            elementwise.getKind() == wafer::InstrElementwiseKind::Recip;
        committedDivision |=
            elementwise.getKind() == wafer::InstrElementwiseKind::Div;
      });
  EXPECT_TRUE(committedReciprocal);
  EXPECT_FALSE(committedDivision);
}

TEST_F(WholeVariantCoordinatorTest,
       DoesNotUseAlternativeToMaskReservedBaselineFailure) {
  auto config = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
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

TEST_F(WholeVariantCoordinatorTest,
       SelectsModularReassociationFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%a: tensor<16xi8>, %b: tensor<16xi8>,
                  %c: tensor<16xi8>) -> tensor<16xi8> {
    %out = tensor.empty() : tensor<16xi8>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
      outs(%out : tensor<16xi8>) {
    ^bb0(%av: i8, %bv: i8, %cv: i8, %unused: i8):
      %aa = arith.muli %av, %av : i8
      %aab = arith.addi %aa, %bv : i8
      %result = arith.addi %aab, %cv : i8
      linalg.yield %result : i8
    } -> tensor<16xi8>
    return %r : tensor<16xi8>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(
      *source, replicated1DProgram(3, 16, "i8"), diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  ASSERT_TRUE(baselineCost->dataDependencyDepth.isKnown());
  ASSERT_TRUE(accepted->resourceCost.maximumRankDataDependencyDepth.isKnown());
  EXPECT_LT(accepted->resourceCost.maximumRankDataDependencyDepth.value,
            baselineCost->dataDependencyDepth.value);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());

  llvm::DenseMap<mlir::Value, wafer::InstrElementwiseKind> writers;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    writers[op.getDest()] = op.getKind();
  });
  bool sawReassociatedJoin = false;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    if (op.getKind() != wafer::InstrElementwiseKind::Add)
      return;
    bool readsMul = false;
    bool readsAdd = false;
    for (mlir::Value input : op.getInputs()) {
      auto found = writers.find(input);
      if (found == writers.end())
        continue;
      readsMul |= found->second == wafer::InstrElementwiseKind::Mul;
      readsAdd |= found->second == wafer::InstrElementwiseKind::Add;
    }
    sawReassociatedJoin |= readsMul && readsAdd;
  });
  EXPECT_TRUE(sawReassociatedJoin);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsBalancedModularAdditionTreeFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%a: tensor<16xi8>, %b: tensor<16xi8>,
                  %c: tensor<16xi8>, %d: tensor<16xi8>)
      -> tensor<16xi8> {
    %out = tensor.empty() : tensor<16xi8>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c, %d : tensor<16xi8>, tensor<16xi8>, tensor<16xi8>,
                            tensor<16xi8>) outs(%out : tensor<16xi8>) {
    ^bb0(%av: i8, %bv: i8, %cv: i8, %dv: i8, %unused: i8):
      %ab = arith.addi %av, %bv : i8
      %abc = arith.addi %ab, %cv : i8
      %result = arith.addi %abc, %dv : i8
      linalg.yield %result : i8
    } -> tensor<16xi8>
    return %r : tensor<16xi8>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(
      *source, replicated1DProgram(4, 16, "i8"), diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  ASSERT_TRUE(baselineCost->dataDependencyDepth.isKnown());
  ASSERT_TRUE(accepted->resourceCost.maximumRankDataDependencyDepth.isKnown());
  EXPECT_LT(accepted->resourceCost.maximumRankDataDependencyDepth.value,
            baselineCost->dataDependencyDepth.value);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());

  llvm::DenseMap<mlir::Value, wafer::InstrElementwiseKind> writers;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    writers[op.getDest()] = op.getKind();
  });
  bool sawBalancedJoin = false;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    if (op.getKind() != wafer::InstrElementwiseKind::Add)
      return;
    unsigned addInputs = 0;
    for (mlir::Value input : op.getInputs()) {
      auto found = writers.find(input);
      addInputs += found != writers.end() &&
                   found->second == wafer::InstrElementwiseKind::Add;
    }
    sawBalancedJoin |= addInputs == 2;
  });
  EXPECT_TRUE(sawBalancedJoin);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsModularDistributiveContractionFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%a: tensor<16xi8>, %b: tensor<16xi8>,
                  %c: tensor<16xi8>) -> tensor<16xi8> {
    %out = tensor.empty() : tensor<16xi8>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
      outs(%out : tensor<16xi8>) {
    ^bb0(%av: i8, %bv: i8, %cv: i8, %unused: i8):
      %ab = arith.muli %av, %bv : i8
      %ac = arith.muli %av, %cv : i8
      %result = arith.subi %ab, %ac : i8
      linalg.yield %result : i8
    } -> tensor<16xi8>
    return %r : tensor<16xi8>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(
      *source, replicated1DProgram(3, 16, "i8"), diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  const auto &winnerCost = accepted->resourceCost.rankCosts.front();
  EXPECT_LT(winnerCost.instructionCount.value,
            baselineCost->instructionCount.value);
  EXPECT_LT(winnerCost.compute.vectorOtherLogicalOps.value,
            baselineCost->compute.vectorOtherLogicalOps.value);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
  unsigned multiplies = 0;
  unsigned subtracts = 0;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    multiplies += op.getKind() == wafer::InstrElementwiseKind::Mul;
    subtracts += op.getKind() == wafer::InstrElementwiseKind::Sub;
  });
  EXPECT_EQ(multiplies, 1u);
  EXPECT_EQ(subtracts, 1u);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsModularCommonFactorFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%a: tensor<16xi8>, %b: tensor<16xi8>,
                  %c: tensor<16xi8>) -> tensor<16xi8> {
    %out = tensor.empty() : tensor<16xi8>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
      outs(%out : tensor<16xi8>) {
    ^bb0(%av: i8, %bv: i8, %cv: i8, %unused: i8):
      %ab = arith.muli %av, %bv : i8
      %ac = arith.muli %av, %cv : i8
      %result = arith.addi %ab, %ac : i8
      linalg.yield %result : i8
    } -> tensor<16xi8>
    return %r : tensor<16xi8>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(
      *source, replicated1DProgram(3, 16, "i8"), diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  const auto &winnerCost = accepted->resourceCost.rankCosts.front();
  EXPECT_LT(winnerCost.instructionCount.value,
            baselineCost->instructionCount.value);
  EXPECT_LT(winnerCost.compute.vectorOtherLogicalOps.value,
            baselineCost->compute.vectorOtherLogicalOps.value);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
  unsigned adds = 0;
  unsigned multiplies = 0;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    adds += op.getKind() == wafer::InstrElementwiseKind::Add;
    multiplies += op.getKind() == wafer::InstrElementwiseKind::Mul;
  });
  EXPECT_EQ(adds, 1u);
  EXPECT_EQ(multiplies, 1u);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsRelaxedBF16CommonFactorFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%a: tensor<16xbf16>, %b: tensor<16xbf16>,
                  %c: tensor<16xbf16>) -> tensor<16xbf16> {
    %out = tensor.empty() : tensor<16xbf16>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a, %b, %c : tensor<16xbf16>, tensor<16xbf16>, tensor<16xbf16>)
      outs(%out : tensor<16xbf16>) {
    ^bb0(%av: bf16, %bv: bf16, %cv: bf16, %unused: bf16):
      %ab = arith.mulf %av, %bv : bf16
      %ac = arith.mulf %av, %cv : bf16
      %result = arith.addf %ab, %ac : bf16
      linalg.yield %result : bf16
    } -> tensor<16xbf16>
    return %r : tensor<16xbf16>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(
      *source, replicated1DProgram(3, 16, "bf16"), diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  const auto &winnerCost = accepted->resourceCost.rankCosts.front();
  EXPECT_LT(winnerCost.instructionCount.value,
            baselineCost->instructionCount.value);
  EXPECT_LT(winnerCost.compute.vectorF16Bf16LogicalOps.value,
            baselineCost->compute.vectorF16Bf16LogicalOps.value);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());

  unsigned adds = 0;
  unsigned multiplies = 0;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    adds += op.getKind() == wafer::InstrElementwiseKind::Add;
    multiplies += op.getKind() == wafer::InstrElementwiseKind::Mul;
  });
  EXPECT_EQ(adds, 1u);
  EXPECT_EQ(multiplies, 1u);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsConsumerLocalRecomputationFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%input: tensor<262144xf32>, %other: tensor<262144xf32>)
      -> (tensor<262144xf32>, tensor<262144xf32>, tensor<262144xf32>) {
    %producer_out = tensor.empty() : tensor<262144xf32>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<262144xf32>)
      outs(%producer_out : tensor<262144xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.negf %value : f32
      linalg.yield %result : f32
    } -> tensor<262144xf32>
    %left_out = tensor.empty() : tensor<262144xf32>
    %left = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<262144xf32>)
      outs(%left_out : tensor<262144xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    } -> tensor<262144xf32>
    %middle_out = tensor.empty() : tensor<262144xf32>
    %middle = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%other : tensor<262144xf32>)
      outs(%middle_out : tensor<262144xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.negf %value : f32
      linalg.yield %result : f32
    } -> tensor<262144xf32>
    %right_out = tensor.empty() : tensor<262144xf32>
    %right = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<262144xf32>)
      outs(%right_out : tensor<262144xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.mulf %value, %value : f32
      linalg.yield %result : f32
    } -> tensor<262144xf32>
    return %left, %middle, %right : tensor<262144xf32>, tensor<262144xf32>,
                                    tensor<262144xf32>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted =
      selectProductionVariant(*source, replicated1DProgram(2, 262144, "f32", 3),
                              diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
  unsigned negations = 0;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    negations += op.getKind() == wafer::InstrElementwiseKind::Neg;
  });
  EXPECT_GE(negations, 3u);
  EXPECT_LT(accepted->resourceCost.aggregateDDRReadBytes.value,
            baselineCost->ddrReadBytes.value);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsSharedProducerFusionFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%input: tensor<16xf32>)
      -> (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) {
    %producer_out = tensor.empty() : tensor<16xf32>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<16xf32>) outs(%producer_out : tensor<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.negf %value : f32
      linalg.yield %result : f32
    } -> tensor<16xf32>
    %left_out = tensor.empty() : tensor<16xf32>
    %left = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<16xf32>) outs(%left_out : tensor<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.addf %value, %value : f32
      linalg.yield %result : f32
    } -> tensor<16xf32>
    %right_out = tensor.empty() : tensor<16xf32>
    %right = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<16xf32>) outs(%right_out : tensor<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.mulf %value, %value : f32
      linalg.yield %result : f32
    } -> tensor<16xf32>
    return %producer, %left, %right : tensor<16xf32>, tensor<16xf32>,
                                      tensor<16xf32>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted =
      selectProductionVariant(*source, replicated1DProgram(1, 16, "f32", 3),
                              diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
  EXPECT_LT(accepted->resourceCost.aggregateDDRReadBytes.value,
            baselineCost->ddrReadBytes.value);

  mlir::ModuleOp winner = accepted->ranks.front().getModule();
  unsigned negations = 0;
  unsigned fusedResultCount = 0;
  unsigned sharedInputLoads = 0;
  winner.walk([&](wafer::InstrElementwiseOp op) {
    negations += op.getKind() == wafer::InstrElementwiseKind::Neg;
  });
  winner.walk([&](wafer::TileRegionOp region) {
    fusedResultCount = std::max(fusedResultCount, region.getNumResults());
    if (region.getNumResults() == 2)
      region.walk([&](wafer::InstrRDMAOp) { ++sharedInputLoads; });
  });
  EXPECT_EQ(negations, 1u);
  EXPECT_EQ(fusedResultCount, 2u);
  EXPECT_EQ(sharedInputLoads, 1u);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsStaticLoopInvariantHoistFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%input: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %empty = tensor.empty() : tensor<16xf32>
    %loop = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%iter = %input) -> tensor<16xf32> {
      %invariant = linalg.generic {
          indexing_maps = [affine_map<(d0)->(d0)>,
                           affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}
        ins(%input : tensor<16xf32>) outs(%empty : tensor<16xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %negated = arith.negf %value : f32
        linalg.yield %negated : f32
      } -> tensor<16xf32>
      scf.yield %invariant : tensor<16xf32>
    }
    %final_out = tensor.empty() : tensor<16xf32>
    %final = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%loop : tensor<16xf32>) outs(%final_out : tensor<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %negated = arith.negf %value : f32
      linalg.yield %negated : f32
    } -> tensor<16xf32>
    return %final : tensor<16xf32>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(
      *source, replicated1DProgram(1, 16, "f32"), diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
  EXPECT_LT(accepted->resourceCost.aggregateCompute.vectorF32LogicalOps.value,
            baselineCost->compute.vectorF32LogicalOps.value);
  bool instructionInsideLoop = false;
  accepted->ranks.front().getModule().walk([&](wafer::InstrElementwiseOp op) {
    for (mlir::scf::ForOp loop = op->getParentOfType<mlir::scf::ForOp>(); loop;
         loop = loop->getParentOfType<mlir::scf::ForOp>()) {
      auto upper =
          loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      instructionInsideLoop |= upper && upper.value() == 4;
    }
  });
  EXPECT_FALSE(instructionInsideLoop);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsMovementFirstReadyOrderFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%a: tensor<16xf32>, %b: tensor<16xf32>)
      -> tensor<16xf32> {
    %producer_out = tensor.empty() : tensor<16xf32>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%a : tensor<16xf32>) outs(%producer_out : tensor<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %negated = arith.negf %value : f32
      linalg.yield %negated : f32
    } -> tensor<16xf32>
    %consumer_out = tensor.empty() : tensor<16xf32>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>,
                         affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}
      ins(%producer, %a, %b : tensor<16xf32>, tensor<16xf32>,
                                tensor<16xf32>)
      outs(%consumer_out : tensor<16xf32>) {
    ^bb0(%pv: f32, %av: f32, %bv: f32, %unused: f32):
      %left = arith.addf %pv, %av : f32
      %result = arith.addf %left, %bv : f32
      linalg.yield %result : f32
    } -> tensor<16xf32>
    return %consumer : tensor<16xf32>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);
  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(
      *source, replicated1DProgram(2, 16, "f32"), diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
  const auto &winnerCost = accepted->resourceCost.rankCosts.front();
  ASSERT_TRUE(winnerCost.readyOrderPriorityInversions.isKnown());
  ASSERT_TRUE(baselineCost->readyOrderPriorityInversions.isKnown());
  EXPECT_LT(winnerCost.readyOrderPriorityInversions.value,
            baselineCost->readyOrderPriorityInversions.value);
  EXPECT_LE(winnerCost.dataDependencyDepth.value,
            baselineCost->dataDependencyDepth.value);
  EXPECT_LE(winnerCost.instructionCount.value,
            baselineCost->instructionCount.value);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsResidentReuseAcrossMixedShapeConsumersFromProductionFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%input: tensor<2x4xf16>)
      -> (tensor<2x4xf32>, tensor<2xf32>) {
    %converted_empty = tensor.empty() : tensor<2x4xf32>
    %converted = linalg.generic {
        indexing_maps = [affine_map<(d0, d1)->(d0, d1)>,
                         affine_map<(d0, d1)->(d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<2x4xf16>)
      outs(%converted_empty : tensor<2x4xf32>) {
    ^bb0(%value: f16, %unused: f32):
      %extended = arith.extf %value : f16 to f32
      linalg.yield %extended : f32
    } -> tensor<2x4xf32>
    %zero = arith.constant 0.0 : f32
    %reduced_empty = tensor.empty() : tensor<2xf32>
    %reduced_init = linalg.fill ins(%zero : f32)
        outs(%reduced_empty : tensor<2xf32>) -> tensor<2xf32>
    %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0, d1)->(d0, d1)>,
                         affine_map<(d0, d1)->(d0)>],
        iterator_types = ["parallel", "reduction"]}
      ins(%converted : tensor<2x4xf32>)
      outs(%reduced_init : tensor<2xf32>) {
    ^bb0(%value: f32, %accumulator: f32):
      %sum = arith.addf %accumulator, %value : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    %negated_empty = tensor.empty() : tensor<2x4xf32>
    %negated = linalg.generic {
        indexing_maps = [affine_map<(d0, d1)->(d0, d1)>,
                         affine_map<(d0, d1)->(d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%converted : tensor<2x4xf32>)
      outs(%negated_empty : tensor<2x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %negative = arith.negf %value : f32
      linalg.yield %negative : f32
    } -> tensor<2x4xf32>
    return %negated, %reduced : tensor<2x4xf32>, tensor<2xf32>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);

  auto makeBoundary = [](int64_t index, llvm::ArrayRef<int64_t> shape,
                         llvm::StringRef dtype) {
    wafer::frontend::ProgramBoundaryBinding binding;
    binding.index = index;
    binding.programIndex = index;
    binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    binding.globalShape.assign(shape.begin(), shape.end());
    binding.localShape.assign(shape.begin(), shape.end());
    binding.dtype = dtype.str();
    wafer::frontend::ProgramRankSlice slice;
    slice.logicalRank = 0;
    slice.replicaId = 0;
    slice.offsets.assign(shape.size(), 0);
    slice.sizes.assign(shape.begin(), shape.end());
    slice.strides.assign(shape.size(), 1);
    binding.rankSlices.push_back(std::move(slice));
    return binding;
  };
  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 1;
  program.distributedInputs.push_back(makeBoundary(0, {2, 4}, "f16"));
  program.distributedOutputs.push_back(makeBoundary(0, {2, 4}, "f32"));
  program.distributedOutputs.push_back(makeBoundary(1, {2}, "f32"));

  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted =
      selectProductionVariant(*source, program, diagnostics, &baselineCost);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
  const auto &winnerCost = accepted->resourceCost.rankCosts.front();
  EXPECT_LT(winnerCost.ddrReadBytes.value, baselineCost->ddrReadBytes.value);
  EXPECT_LT(winnerCost.ddrWriteBytes.value, baselineCost->ddrWriteBytes.value);

  mlir::ModuleOp winner = accepted->ranks.front().getModule();
  bool hasSPMProducerResult = false;
  bool hasSPMConsumerOperand = false;
  winner.walk([&](wafer::TileRegionOp region) {
    hasSPMProducerResult |=
        llvm::any_of(region.getResultTypes(), wafer::isWaferSPMMemRefType);
    hasSPMConsumerOperand |=
        llvm::any_of(region.getOperandTypes(), wafer::isWaferSPMMemRefType);
  });
  EXPECT_TRUE(hasSPMProducerResult);
  EXPECT_TRUE(hasSPMConsumerOperand);
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsFixedCxGemmWithDirectMappedMovementInWholeWinner) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64: 0, 0, 0, 0>,
    policy = "explicit", shape = array<i64: 1>, topology = @default
  }
  func.func @main(%lhs: tensor<1x1xf16>,
                  %rhs: tensor<1x1xf16>) -> tensor<1x1xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<1x1xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<1x1xf16>) -> tensor<1x1xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<1x1xf16>, tensor<1x1xf16>)
        outs(%init : tensor<1x1xf16>) -> tensor<1x1xf16>
    return %result : tensor<1x1xf16>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);

  auto makeBoundary = [](int64_t index, llvm::ArrayRef<int64_t> shape) {
    wafer::frontend::ProgramBoundaryBinding binding;
    binding.index = index;
    binding.programIndex = index;
    binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    binding.globalShape.assign(shape.begin(), shape.end());
    binding.localShape.assign(shape.begin(), shape.end());
    binding.dtype = "f16";
    wafer::frontend::ProgramRankSlice slice;
    slice.logicalRank = 0;
    slice.replicaId = 0;
    slice.offsets.assign(shape.size(), 0);
    slice.sizes.assign(shape.begin(), shape.end());
    slice.strides.assign(shape.size(), 1);
    binding.rankSlices.push_back(std::move(slice));
    return binding;
  };
  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 1;
  program.programUserInputCount = 2;
  program.distributedInputs.push_back(makeBoundary(0, {1, 1}));
  program.distributedInputs.push_back(makeBoundary(1, {1, 1}));
  program.distributedOutputs.push_back(makeBoundary(0, {1, 1}));

  wafer::TensorProgramSchedulingConfig schedulingConfig;
  schedulingConfig.logicalRank = 0;
  schedulingConfig.candidateParallelism = 1;
  auto rankFrontier =
      wafer::buildScheduledRankCandidateFrontier(*source, schedulingConfig);
  ASSERT_TRUE(mlir::succeeded(rankFrontier));
  bool sawDirectMappedRoute = false;
  for (wafer::ScheduledRankCandidate &candidate : *rankFrontier) {
    candidate.module->walk([&](mlir::Operation *operation) {
      if (mlir::isa<wafer::InstrRDMAOp, wafer::InstrWDMAOp>(operation))
        sawDirectMappedRoute |= operation->hasAttr("src_offset") ||
                                operation->hasAttr("dst_offset");
    });
  }
  EXPECT_TRUE(sawDirectMappedRoute);

  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(*source, program, diagnostics);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  EXPECT_FALSE(accepted->selectedReservedBaselines.front());
  mlir::ModuleOp winner = accepted->ranks.front().getModule();

  unsigned gemmCount = 0;
  winner.walk([&](wafer::InstrGemmOp gemm) {
    ++gemmCount;
    for (mlir::Type type : {gemm.getLhs().getType(), gemm.getRhs().getType(),
                            gemm.getDest().getType()}) {
      auto memref = mlir::cast<mlir::MemRefType>(type);
      EXPECT_EQ(wafer::getWaferMemoryAttr(memref).getLayout(),
                wafer::MemLayout::Cx);
    }
  });
  EXPECT_GT(gemmCount, 0u);
  unsigned gatherScatterCount = 0;
  winner.walk([&](wafer::InstrGatherScatterOp) { ++gatherScatterCount; });
  std::string winnerText;
  llvm::raw_string_ostream winnerStream(winnerText);
  winner.print(winnerStream);
  EXPECT_EQ(gatherScatterCount, 0u) << winnerStream.str();
  unsigned mappedMovementCount = 0;
  winner.walk([&](mlir::Operation *operation) {
    if (!mlir::isa<wafer::InstrRDMAOp, wafer::InstrWDMAOp>(operation))
      return;
    mappedMovementCount +=
        operation->hasAttr("src_offset") || operation->hasAttr("dst_offset");
  });
  EXPECT_GT(mappedMovementCount, 0u) << winnerStream.str();
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsTopologyRingAllGatherFromProductionAllRankFrontier) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64>,
    policy = "all_available", shape = array<i64: 16>, topology = @default
  }
  func.func @main(%input: tensor<4xf32>) -> tensor<64xf32> {
    %out = tensor.empty() : tensor<64xf32>
    %gathered = wafer.linalg_ext.collective.all_gather
        ins(%input : tensor<4xf32>) outs(%out : tensor<64xf32>)
        {axis = 0 : i64, channel_id = 47 : i64,
         rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                                8, 9, 10, 11, 12, 13, 14, 15>}
        -> tensor<64xf32>
    return %gathered : tensor<64xf32>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);

  auto makeBoundary = [](int64_t elements) {
    wafer::frontend::ProgramBoundaryBinding binding;
    binding.index = 0;
    binding.programIndex = 0;
    binding.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    binding.globalShape = {elements};
    binding.localShape = {elements};
    binding.dtype = "f32";
    for (int64_t rank = 0; rank < 16; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId = rank;
      slice.offsets = {0};
      slice.sizes = {elements};
      slice.strides = {1};
      binding.rankSlices.push_back(std::move(slice));
    }
    return binding;
  };
  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 1;
  program.distributedInputs.push_back(makeBoundary(4));
  program.distributedOutputs.push_back(makeBoundary(64));

  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(*source, program, diagnostics,
                                          &baselineCost, /*rankCount=*/16);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  ASSERT_EQ(accepted->ranks.size(), 16u);
  ASSERT_TRUE(accepted->resourceCost.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(accepted->resourceCost.minimumHopLinkByteDemand.value, 3840u);
  // Both schedules inject 16 * 15 * 16 = 3840 bytes.  On this 4x4 mesh the
  // minimum-hop Hamiltonian ring uses one hop per send, while the direct
  // all-pairs schedule has 10240 link-bytes of shortest-path demand.
  EXPECT_EQ(accepted->resourceCost.aggregateNoC.aggregateTransmitBytes.value,
            3840u);
  EXPECT_LT(accepted->resourceCost.minimumHopLinkByteDemand.value, 10240u);
  for (const auto &rank : accepted->ranks) {
    bool sawDirect = false;
    bool sawRing = false;
    rank.getModule().walk([&](wafer::InstrDTESendOp send) {
      sawDirect |= send.getMessage().getPhase() ==
                   wafer::DTEProtocolPhase::AllGatherDirect;
      sawRing |= send.getMessage().getPhase() ==
                 wafer::DTEProtocolPhase::AllGatherRing;
    });
    rank.getModule().walk([&](wafer::InstrDTERecvOp recv) {
      sawDirect |= recv.getMessage().getPhase() ==
                   wafer::DTEProtocolPhase::AllGatherDirect;
      sawRing |= recv.getMessage().getPhase() ==
                 wafer::DTEProtocolPhase::AllGatherRing;
    });
    EXPECT_FALSE(sawDirect);
    EXPECT_TRUE(sawRing);
  }
}

TEST_F(WholeVariantCoordinatorTest,
       SelectsTreeAllReduceFromProductionAllRankFrontier) {
  // Keep the reserved Auto candidate and the explicit Tree candidate
  // genuinely distinct: i8 addition is exactly reassociable and sixteen
  // elements form one nonzero Ring chunk per rank. Floating-point Auto must
  // already use the ordered tree and is covered by the lowering tests.
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64>,
    policy = "all_available", shape = array<i64: 16>, topology = @default
  }
  func.func @main(%input: tensor<16xi8>) -> tensor<16xi8> {
    %out = tensor.empty() : tensor<16xi8>
    %reduced = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<16xi8>) outs(%out : tensor<16xi8>) {
      ^bb0(%lhs: i8, %rhs: i8):
        %sum = arith.addi %lhs, %rhs : i8
        wafer.linalg_ext.collective.yield %sum : i8
    } {channel_id = 53 : i64,
       rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                              8, 9, 10, 11, 12, 13, 14, 15>}
        -> tensor<16xi8>
    return %reduced : tensor<16xi8>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);

  std::optional<wafer::analysis::InstructionProgramCost> baselineCost;
  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(
      *source,
      replicated1DProgram(1, 16, "i8", /*outputCount=*/1,
                          /*rankCount=*/16),
      diagnostics, &baselineCost, /*rankCount=*/16);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_TRUE(baselineCost);
  ASSERT_EQ(accepted->ranks.size(), 16u);
  EXPECT_TRUE(llvm::none_of(accepted->selectedReservedBaselines,
                            [](bool reserved) { return reserved; }));
  ASSERT_TRUE(baselineCost->instructionCount.isKnown());
  ASSERT_TRUE(accepted->resourceCost.aggregateInstructionCount.isKnown());
  EXPECT_LT(accepted->resourceCost.aggregateInstructionCount.value,
            baselineCost->instructionCount.value * 16);
  ASSERT_TRUE(accepted->resourceCost.minimumHopLinkByteDemand.isKnown());
  EXPECT_EQ(accepted->resourceCost.minimumHopLinkByteDemand.value, 480u);
  ASSERT_TRUE(
      accepted->resourceCost.aggregateNoC.aggregateTransmitBytes.isKnown());
  EXPECT_EQ(accepted->resourceCost.aggregateNoC.aggregateTransmitBytes.value,
            480u);
  for (const auto &rank : accepted->ranks) {
    bool sawTree = false;
    bool sawRing = false;
    rank.getModule().walk([&](wafer::InstrDTESendOp send) {
      sawTree |= send.getMessage().getPhase() ==
                     wafer::DTEProtocolPhase::AllReduceTreeReduce ||
                 send.getMessage().getPhase() ==
                     wafer::DTEProtocolPhase::AllReduceTreeBroadcast;
      sawRing |= send.getMessage().getPhase() ==
                 wafer::DTEProtocolPhase::AllReduceRing;
    });
    rank.getModule().walk([&](wafer::InstrDTERecvOp recv) {
      sawTree |= recv.getMessage().getPhase() ==
                     wafer::DTEProtocolPhase::AllReduceTreeReduce ||
                 recv.getMessage().getPhase() ==
                     wafer::DTEProtocolPhase::AllReduceTreeBroadcast;
      sawRing |= recv.getMessage().getPhase() ==
                 wafer::DTEProtocolPhase::AllReduceRing;
    });
    EXPECT_TRUE(sawTree);
    EXPECT_FALSE(sawRing);
  }
}

TEST_F(WholeVariantCoordinatorTest,
       TilesLargeKShardedF16GemmAllReduceInProductionWholeVariant) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default {
    card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
    tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>
  }
  wafer.execution.mesh @default_mesh {
    axes = ["rank"], endpoints = array<i64>,
    policy = "all_available", shape = array<i64: 16>, topology = @default
  }
  func.func @main(%lhs: tensor<4096x256xf16>,
                  %rhs: tensor<256x4096xf16>) -> tensor<4096x4096xf16> {
    %zero = arith.constant 0.0 : f16
    %partial_out = tensor.empty() : tensor<4096x4096xf16>
    %partial_init = linalg.fill ins(%zero : f16)
        outs(%partial_out : tensor<4096x4096xf16>)
        -> tensor<4096x4096xf16>
    %partial = linalg.matmul
        ins(%lhs, %rhs : tensor<4096x256xf16>, tensor<256x4096xf16>)
        outs(%partial_init : tensor<4096x4096xf16>)
        -> tensor<4096x4096xf16>
    %reduced_out = tensor.empty() : tensor<4096x4096xf16>
    %reduced = wafer.linalg_ext.collective.all_reduce
        ins(%partial : tensor<4096x4096xf16>)
        outs(%reduced_out : tensor<4096x4096xf16>) {
      ^bb0(%lhs_value: f16, %rhs_value: f16):
        %sum = arith.addf %lhs_value, %rhs_value : f16
        wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 61 : i64,
       rank_group = array<i64: 0, 1, 2, 3, 4, 5, 6, 7,
                              8, 9, 10, 11, 12, 13, 14, 15>}
        -> tensor<4096x4096xf16>
    return %reduced : tensor<4096x4096xf16>
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(source);

  auto makeBoundary = [](int64_t index, llvm::ArrayRef<int64_t> globalShape,
                         llvm::ArrayRef<int64_t> localShape,
                         wafer::frontend::ProgramDistributionKind distribution,
                         bool shardFirstDimension) {
    wafer::frontend::ProgramBoundaryBinding binding;
    binding.index = index;
    binding.programIndex = index;
    binding.distribution = distribution;
    binding.globalShape.assign(globalShape.begin(), globalShape.end());
    binding.localShape.assign(localShape.begin(), localShape.end());
    binding.dtype = "f16";
    for (int64_t rank = 0; rank < 16; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId =
          distribution == wafer::frontend::ProgramDistributionKind::Replicated
              ? rank
              : 0;
      slice.offsets.assign(globalShape.size(), 0);
      if (distribution ==
          wafer::frontend::ProgramDistributionKind::Partitioned) {
        const size_t shardDimension = shardFirstDimension ? 0 : 1;
        slice.offsets[shardDimension] = rank * localShape[shardDimension];
      }
      slice.sizes.assign(localShape.begin(), localShape.end());
      slice.strides.assign(globalShape.size(), 1);
      binding.rankSlices.push_back(std::move(slice));
    }
    return binding;
  };

  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  program.programUserInputCount = 2;
  program.distributedInputs.push_back(
      makeBoundary(0, {4096, 4096}, {4096, 256},
                   wafer::frontend::ProgramDistributionKind::Partitioned,
                   /*shardFirstDimension=*/false));
  program.distributedInputs.push_back(
      makeBoundary(1, {4096, 4096}, {256, 4096},
                   wafer::frontend::ProgramDistributionKind::Partitioned,
                   /*shardFirstDimension=*/true));
  program.distributedOutputs.push_back(
      makeBoundary(0, {4096, 4096}, {4096, 4096},
                   wafer::frontend::ProgramDistributionKind::Replicated,
                   /*shardFirstDimension=*/false));

  std::string diagnosticText;
  llvm::raw_string_ostream diagnostics(diagnosticText);
  auto accepted = selectProductionVariant(*source, program, diagnostics,
                                          /*baselineCost=*/nullptr,
                                          /*rankCount=*/16);
  ASSERT_TRUE(mlir::succeeded(accepted)) << diagnosticText;
  ASSERT_EQ(accepted->ranks.size(), 16u);

  constexpr int64_t fullOutputExtent = 4096;
  constexpr int64_t localContractingExtent = 256;
  constexpr int64_t fullOutputBytes = fullOutputExtent * fullOutputExtent * 2;
  const wafer::WaferTargetPolicy targetPolicy =
      wafer::getDefaultWaferTargetPolicy();

  unsigned totalGemmCount = 0;
  unsigned totalDTEIssueCount = 0;
  unsigned totalSPMAllocationCount = 0;
  for (const wafer::compiler::RankExecutable &rank : accepted->ranks) {
    EXPECT_EQ(rank.getTransportContract(),
              wafer::compiler::TransportContract::DirectDTE);
    mlir::ModuleOp winner = rank.getModule();

    int64_t largestGemmOutputTileBytes = 0;
    winner.walk([&](wafer::InstrGemmOp gemm) {
      ++totalGemmCount;
      EXPECT_GT(gemm.getM(), 0);
      EXPECT_GT(gemm.getN(), 0);
      EXPECT_LT(gemm.getM(), fullOutputExtent);
      EXPECT_LT(gemm.getN(), fullOutputExtent);
      EXPECT_EQ(gemm.getK(), localContractingExtent);
      EXPECT_TRUE(gemm->getParentOfType<mlir::scf::ForOp>());
      largestGemmOutputTileBytes =
          std::max(largestGemmOutputTileBytes,
                   static_cast<int64_t>(gemm.getM() * gemm.getN() * 2));
    });

    auto checkDTEIssue = [&](auto issue) {
      ++totalDTEIssueCount;
      EXPECT_GT(issue.getBytes(), 0);
      EXPECT_LT(issue.getBytes(), fullOutputBytes);
      EXPECT_LE(issue.getBytes(), largestGemmOutputTileBytes);
      EXPECT_TRUE(issue.getBinding());
      EXPECT_TRUE(
          issue.getOperation()->template getParentOfType<mlir::scf::ForOp>());
    };
    winner.walk([&](wafer::InstrDTESendOp send) { checkDTEIssue(send); });
    winner.walk([&](wafer::InstrDTERecvOp recv) { checkDTEIssue(recv); });

    winner.walk([&](mlir::memref::AllocOp allocation) {
      if (!wafer::isWaferSPMMemRefType(allocation.getType()))
        return;
      ++totalSPMAllocationCount;
      llvm::ArrayRef<int64_t> shape = allocation.getType().getShape();
      EXPECT_FALSE(shape.size() == 2 && shape[0] == fullOutputExtent &&
                   shape[1] == fullOutputExtent);
      auto offset = allocation->getAttrOfType<wafer::SPMOffsetAttr>(
          wafer::kWaferSPMOffsetAttrName);
      ASSERT_TRUE(offset);
      std::optional<wafer::WaferPhysicalTensorInfo> physical =
          wafer::computeWaferPhysicalTensorInfo(allocation.getType());
      ASSERT_TRUE(physical);
      ASSERT_GE(physical->physicalBytes, 0);
      EXPECT_GE(offset.getOffset(), targetPolicy.memory.spmBase);
      EXPECT_LE(offset.getOffset() + physical->physicalBytes,
                targetPolicy.memory.spmLimit);
    });
  }
  EXPECT_GT(totalGemmCount, 0u);
  EXPECT_GT(totalDTEIssueCount, 0u);
  EXPECT_GT(totalSPMAllocationCount, 0u);
}

} // namespace
