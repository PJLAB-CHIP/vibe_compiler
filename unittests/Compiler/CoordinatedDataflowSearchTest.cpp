//===- CoordinatedDataflowSearchTest.cpp ---------------------------------===//

#include "../../lib/Wafer/Compiler/CoordinatedDataflowSearch.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>

namespace {

class CoordinatedDataflowSearchTest : public ::testing::Test {
protected:
  CoordinatedDataflowSearchTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out = tensor.empty() : tensor<8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>) outs(%out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %result : tensor<8xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeChainProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<8xf16>) -> tensor<8xf16> {
    %producer_out = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>)
        outs(%producer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<8xf16>
    %consumer_out = tensor.empty() : tensor<8xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%producer : tensor<8xf16>)
        outs(%consumer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %squared = arith.mulf %value, %value : f16
      linalg.yield %squared : f16
    } -> tensor<8xf16>
    return %consumer : tensor<8xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeMixedShapeFanoutProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<8xf16>)
      -> (tensor<8xf16>, tensor<4xf16>, tensor<4xf16>) {
    %producer_out = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<8xf16>) outs(%producer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<8xf16>
    %whole_out = tensor.empty() : tensor<8xf16>
    %whole = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<8xf16>) outs(%whole_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    } -> tensor<8xf16>
    %low_slice = tensor.extract_slice %producer[0] [4] [1]
        : tensor<8xf16> to tensor<4xf16>
    %low_out = tensor.empty() : tensor<4xf16>
    %low = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%low_slice : tensor<4xf16>) outs(%low_out : tensor<4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %square = arith.mulf %value, %value : f16
      linalg.yield %square : f16
    } -> tensor<4xf16>
    %high_slice = tensor.extract_slice %producer[4] [4] [1]
        : tensor<8xf16> to tensor<4xf16>
    %high_out = tensor.empty() : tensor<4xf16>
    %high = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%high_slice : tensor<4xf16>) outs(%high_out : tensor<4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %difference = arith.subf %value, %value : f16
      linalg.yield %difference : f16
    } -> tensor<4xf16>
    return %whole, %low, %high
        : tensor<8xf16>, tensor<4xf16>, tensor<4xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeMatmulProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<2x3xf16>, %rhs: tensor<3x2xf16>)
      -> tensor<2x2xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<2x2xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x2xf16>) -> tensor<2x2xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<2x3xf16>, tensor<3x2xf16>)
        outs(%init : tensor<2x2xf16>) -> tensor<2x2xf16>
    return %result : tensor<2x2xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeDiamondFaninTailProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%left: tensor<10xf16>, %right: tensor<10xf16>)
      -> tensor<10xf16> {
    %producer_out = tensor.empty() : tensor<10xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%left : tensor<10xf16>) outs(%producer_out : tensor<10xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<10xf16>
    %upper_out = tensor.empty() : tensor<10xf16>
    %upper = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<10xf16>) outs(%upper_out : tensor<10xf16>) {
    ^bb0(%value: f16, %old: f16):
      %square = arith.mulf %value, %value : f16
      linalg.yield %square : f16
    } -> tensor<10xf16>
    %lower_out = tensor.empty() : tensor<10xf16>
    %lower = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer, %right : tensor<10xf16>, tensor<10xf16>)
      outs(%lower_out : tensor<10xf16>) {
    ^bb0(%first: f16, %second: f16, %old: f16):
      %sum = arith.addf %first, %second : f16
      linalg.yield %sum : f16
    } -> tensor<10xf16>
    %result_out = tensor.empty() : tensor<10xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%upper, %lower : tensor<10xf16>, tensor<10xf16>)
      outs(%result_out : tensor<10xf16>) {
    ^bb0(%first: f16, %second: f16, %old: f16):
      %difference = arith.subf %first, %second : f16
      linalg.yield %difference : f16
    } -> tensor<10xf16>
    return %result : tensor<10xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeSharedInputContractionProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<4x8xf16>, %first_rhs: tensor<8x6xf16>,
                  %second_rhs: tensor<8x6xf16>)
      -> (tensor<4x6xf16>, tensor<4x6xf16>) {
    %zero = arith.constant 0.0 : f16
    %first_out = tensor.empty() : tensor<4x6xf16>
    %first_init = linalg.fill ins(%zero : f16)
        outs(%first_out : tensor<4x6xf16>) -> tensor<4x6xf16>
    %first = linalg.matmul
        ins(%lhs, %first_rhs : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%first_init : tensor<4x6xf16>) -> tensor<4x6xf16>
    %second_out = tensor.empty() : tensor<4x6xf16>
    %second_init = linalg.fill ins(%zero : f16)
        outs(%second_out : tensor<4x6xf16>) -> tensor<4x6xf16>
    %second = linalg.matmul
        ins(%lhs, %second_rhs : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%second_init : tensor<4x6xf16>) -> tensor<4x6xf16>
    return %first, %second : tensor<4x6xf16>, tensor<4x6xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeTransposedWeightContractionProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%activation: tensor<4x8xf16>,
                  %weight: tensor<6x8xf16>) -> tensor<4x6xf16> {
    %transpose_out = tensor.empty() : tensor<8x6xf16>
    %transpose = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%weight : tensor<6x8xf16>)
      outs(%transpose_out : tensor<8x6xf16>) {
    ^bb0(%value: f16, %old: f16):
      linalg.yield %value : f16
    } -> tensor<8x6xf16>
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<4x6xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<4x6xf16>) -> tensor<4x6xf16>
    %result = linalg.matmul
        ins(%activation, %transpose : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%init : tensor<4x6xf16>) -> tensor<4x6xf16>
    return %result : tensor<4x6xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeInvariantMatmulProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<4x8xf16>, %rhs: tensor<8x6xf16>)
      -> tensor<4x6xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<4x6xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<4x6xf16>) -> tensor<4x6xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%init : tensor<4x6xf16>) -> tensor<4x6xf16>
    return %result : tensor<4x6xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeStructuredConcatProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%first: tensor<1x2x2x4xf16>,
                  %second: tensor<1x2x2x4xf16>)
      -> tensor<1x2x2x8xf16> {
    %c4 = arith.constant 4 : index
    %out = tensor.empty() : tensor<1x2x2x8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3)
                                     -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } outs(%out : tensor<1x2x2x8xf16>) {
    ^bb0(%old: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %i3 = linalg.index 3 : index
      %in_first = arith.cmpi ult, %i3, %c4 : index
      %value = scf.if %in_first -> f16 {
        %first_value = tensor.extract %first[%i0, %i1, %i2, %i3]
            : tensor<1x2x2x4xf16>
        scf.yield %first_value : f16
      } else {
        %second_i3 = arith.subi %i3, %c4 : index
        %second_value = tensor.extract %second[%i0, %i1, %i2, %second_i3]
            : tensor<1x2x2x4xf16>
        scf.yield %second_value : f16
      }
      linalg.yield %value : f16
    } -> tensor<1x2x2x8xf16>
    return %result : tensor<1x2x2x8xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CoordinatedDataflowSearchTest,
       ReservesMandatoryGenerationTerminalAndRepairBeforeWork) {
  auto terminal = wafer::compiler::detail::CoordinatedWorkLedger::
      getTerminalActionUpperBound(/*rankCount=*/4);
  ASSERT_TRUE(mlir::succeeded(terminal));
  ASSERT_TRUE(terminal->getTotal());
  constexpr uint64_t repair = 5;
  constexpr uint64_t mandatoryGeneration = 8;
  uint64_t capacity = mandatoryGeneration + *terminal->getTotal() + repair + 2;
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      /*rankCount=*/4, capacity, repair);
  ASSERT_TRUE(mlir::succeeded(ledger));
  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.mandatoryGenerationReserved, mandatoryGeneration);
  EXPECT_EQ(snapshot.terminalReserved, *terminal->getTotal());
  EXPECT_EQ(snapshot.repairReserved, repair);
  EXPECT_EQ(snapshot.unreserved, 2u);
  EXPECT_TRUE(ledger->tryConsumeGeneration(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 2));
  EXPECT_FALSE(ledger->tryConsumeGeneration(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1));
  snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.terminalReserved, *terminal->getTotal());
  EXPECT_EQ(snapshot.repairReserved, repair);
}

TEST_F(CoordinatedDataflowSearchTest,
       OwnsAllRanksAsActualTileClonesUnderOneReservation) {
  auto source = makeProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      /*rankCount=*/4);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 4;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);
  const auto &baseline = frontier->front();
  EXPECT_TRUE(baseline.reservedBaseline);
  EXPECT_EQ(baseline.terminalReservation.id,
            ledger->getMandatoryBaselineReservation().id);
  llvm::SmallSet<uint64_t, 8> reservations;
  unsigned baselineCount = 0;
  for (const auto &variant : *frontier) {
    baselineCount += variant.reservedBaseline;
    EXPECT_TRUE(reservations.insert(variant.terminalReservation.id).second);
    ASSERT_EQ(variant.ranks.size(), 4u);
    for (auto [logicalRank, rank] : llvm::enumerate(variant.ranks)) {
      EXPECT_EQ(rank.logicalRank, static_cast<int64_t>(logicalRank));
      EXPECT_TRUE(wafer::containsTileDataflowOperations(
          rank.module.get().getOperation()));
      EXPECT_TRUE(rank.selectedTileIR);
      bool hasInstruction = false;
      rank.module.get().walk([&](mlir::Operation *operation) {
        hasInstruction |=
            mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp>(
                operation);
      });
      EXPECT_FALSE(hasInstruction);
    }
    EXPECT_NE(variant.ranks[0].module.get(), variant.ranks[1].module.get());
    EXPECT_EQ(variant.frontierDigest,
              wafer::compiler::detail::computeCoordinatedTileFrontierDigest(
                  *frontier));
  }
  EXPECT_EQ(baselineCount, 1u);
  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.mandatoryGenerationReserved, 0u);
  EXPECT_GT(snapshot.terminalReserved, 0u);
}

TEST_F(CoordinatedDataflowSearchTest,
       CoupledTraversalRemovesTheIntermediateDDRRoundTrip) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);

  auto countTransfers = [](mlir::ModuleOp module) {
    unsigned count = 0;
    module.walk([&](mlir::Operation *operation) {
      count +=
          mlir::isa<wafer::StorageLoadOp, wafer::StorageStoreOp>(operation);
    });
    return count;
  };
  const unsigned baselineTransfers =
      countTransfers(*frontier->front().ranks.front().module);
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        return countTransfers(*variant.ranks.front().module) <
               baselineTransfers;
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       ActualResidencyActionsCoverFusedSeparatedSplitAndSelectiveSpill) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  auto materialize = [&](wafer::CompleteRankTraversalComposition composition,
                         wafer::CandidateTileResidencyAction residency) {
    std::string failureReason;
    auto candidate = wafer::materializeCompleteRankCandidateTileProgram(
        *source, /*logicalRank=*/0, /*candidateTileSizes=*/{8},
        /*candidateReductionTileSizes=*/{},
        wafer::CandidateTileTraversalKind::ResultDriven, composition, residency,
        wafer::CandidateBoundaryMovementAction::Staged,
        wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidate)) << failureReason;
    return candidate;
  };

  auto fused =
      materialize(wafer::CompleteRankTraversalComposition::Coupled,
                  wafer::CandidateTileResidencyAction::KeepSingleRegion);
  auto separated =
      materialize(wafer::CompleteRankTraversalComposition::Separated,
                  wafer::CandidateTileResidencyAction::KeepSingleRegion);
  auto split = materialize(
      wafer::CompleteRankTraversalComposition::Separated,
      wafer::CandidateTileResidencyAction::SplitAtExplicitDDRBoundary);
  auto selective =
      materialize(wafer::CompleteRankTraversalComposition::Coupled,
                  wafer::CandidateTileResidencyAction::SelectiveSpill);
  ASSERT_TRUE(mlir::succeeded(fused));
  ASSERT_TRUE(mlir::succeeded(separated));
  ASSERT_TRUE(mlir::succeeded(split));
  ASSERT_TRUE(mlir::succeeded(selective));

  auto count = [](mlir::ModuleOp module, auto tag) {
    unsigned result = 0;
    module.walk([&](decltype(tag)) { ++result; });
    return result;
  };
  EXPECT_EQ(count(**fused, wafer::TileRegionOp{}), 1u);
  EXPECT_EQ(count(**separated, wafer::TileRegionOp{}), 1u);
  EXPECT_EQ(count(**split, wafer::TileRegionOp{}), 2u);
  EXPECT_EQ(count(**selective, wafer::TileRegionOp{}), 1u);
  const auto transferCount = [&](mlir::ModuleOp module) {
    return count(module, wafer::StorageLoadOp{}) +
           count(module, wafer::StorageStoreOp{});
  };
  EXPECT_LT(transferCount(**fused), transferCount(**separated));
  EXPECT_GT(transferCount(**selective), transferCount(**fused));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**fused)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**separated)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**split)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**selective)));

  (*split)->walk([&](wafer::TileRegionOp region) {
    for (mlir::Type type : region.getOperandTypes())
      if (mlir::isa<mlir::ShapedType>(type))
        EXPECT_TRUE(wafer::isWaferDDRMemRefType(type));
    for (mlir::Type type : region.getResultTypes())
      if (mlir::isa<mlir::ShapedType>(type))
        EXPECT_TRUE(wafer::isWaferDDRMemRefType(type));
  });

  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(llvm::any_of(*frontier, [&](const auto &variant) {
    return count(*variant.ranks.front().module, wafer::TileRegionOp{}) == 2;
  }));
  EXPECT_TRUE(llvm::any_of(*frontier, [&](const auto &variant) {
    mlir::ModuleOp module = *variant.ranks.front().module;
    return count(module, wafer::TileRegionOp{}) == 1 &&
           transferCount(module) > transferCount(**fused);
  }));
}

TEST_F(CoordinatedDataflowSearchTest,
       RepairClonesTheLiveAllRankTileParentWithoutSourceReplay) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  std::string materializationFailure;
  auto rankZero = wafer::materializeCompleteRankCandidateTileProgram(
      *source, /*logicalRank=*/0, /*candidateTileSizes=*/{8},
      /*candidateReductionTileSizes=*/{},
      wafer::CandidateTileTraversalKind::ResultDriven,
      wafer::CompleteRankTraversalComposition::Coupled,
      wafer::CandidateTileResidencyAction::KeepSingleRegion,
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed,
      &materializationFailure);
  ASSERT_TRUE(mlir::succeeded(rankZero)) << materializationFailure;

  auto terminal = wafer::compiler::detail::CoordinatedWorkLedger::
      getTerminalActionUpperBound(/*rankCount=*/2);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(2);
  ASSERT_TRUE(mlir::succeeded(terminal));
  ASSERT_TRUE(mlir::succeeded(ledger));
  auto parentReservation = ledger->tryReserveTerminalAction(*terminal);
  ASSERT_TRUE(parentReservation);

  wafer::compiler::detail::CoordinatedTileVariant parent;
  parent.stableSemanticOrdinal = 7;
  parent.terminalReservation = *parentReservation;
  std::string selectedText;
  llvm::raw_string_ostream selectedStream(selectedText);
  (*rankZero)->print(selectedStream);
  selectedStream.flush();
  auto selected =
      std::make_shared<const std::string>(std::move(selectedText));
  auto rankOne = mlir::cast<mlir::ModuleOp>((*rankZero)->clone());
  parent.ranks.emplace_back(/*logicalRank=*/0, std::move(*rankZero), selected);
  parent.ranks.emplace_back(/*logicalRank=*/1, std::move(rankOne), selected);
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::verifyCoordinatedTileVariant(parent, 2)));
  const std::string parentDigest =
      wafer::compiler::detail::computeCoordinatedTileVariantContentDigest(
          parent);

  wafer::compiler::detail::CoordinatedWorkEstimate failedAttempt;
  failedAttempt.set(
      wafer::compiler::detail::CoordinatedWorkKind::TileToInstrLowering, 2);
  ASSERT_TRUE(mlir::succeeded(ledger->completeTerminalAction(
      parent.terminalReservation, failedAttempt)));
  const auto beforeRepair = ledger->getSnapshot();

  std::string repairFailure;
  auto repaired = wafer::compiler::detail::materializeCoordinatedTileRepair(
      parent,
      wafer::compiler::detail::CoordinatedTileRepairAction::SelectiveSpill,
      /*stableSemanticOrdinal=*/8, *ledger, &repairFailure);
  ASSERT_TRUE(mlir::succeeded(repaired)) << repairFailure;
  EXPECT_EQ(repaired->repairDepth, 1u);
  EXPECT_FALSE(repaired->reservedBaseline);
  EXPECT_NE(repaired->terminalReservation.id, parent.terminalReservation.id);
  ASSERT_EQ(repaired->ranks.size(), 2u);
  EXPECT_EQ(
      wafer::compiler::detail::computeCoordinatedTileVariantContentDigest(
          parent),
      parentDigest);
  EXPECT_NE(
      wafer::compiler::detail::computeCoordinatedTileVariantContentDigest(
          *repaired),
      parentDigest);
  for (const auto &rank : repaired->ranks) {
    unsigned stores = 0;
    unsigned loads = 0;
    rank.module.get().walk([&](wafer::StorageStoreOp) { ++stores; });
    rank.module.get().walk([&](wafer::StorageLoadOp) { ++loads; });
    EXPECT_GT(stores, 0u);
    EXPECT_GT(loads, 0u);
  }
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::verifyCoordinatedTileVariant(*repaired, 2)));

  const auto afterRepair = ledger->getSnapshot();
  EXPECT_EQ(afterRepair.repairReserved + 3, beforeRepair.repairReserved);
  EXPECT_EQ(afterRepair.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    RepairExpansion)],
            beforeRepair.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    RepairExpansion)] +
                1);
  EXPECT_EQ(afterRepair.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    ActualTileClone)],
            beforeRepair.consumedByKind[static_cast<size_t>(
                wafer::compiler::detail::CoordinatedWorkKind::
                    ActualTileClone)] +
                2);

  auto nestedRepair =
      wafer::compiler::detail::materializeCoordinatedTileRepair(
          *repaired,
          wafer::compiler::detail::CoordinatedTileRepairAction::SelectiveSpill,
          /*stableSemanticOrdinal=*/9, *ledger, &repairFailure);
  EXPECT_TRUE(mlir::failed(nestedRepair));
  EXPECT_EQ(ledger->getSnapshot().repairReserved, afterRepair.repairReserved);
}

TEST_F(CoordinatedDataflowSearchTest,
       MixedShapeThreeConsumerFanoutMaterializesActualBoundedVersions) {
  auto source = makeMixedShapeFanoutProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);

  auto countTransfers = [](mlir::ModuleOp module) {
    unsigned count = 0;
    module.walk([&](mlir::Operation *operation) {
      count +=
          mlir::isa<wafer::StorageLoadOp, wafer::StorageStoreOp>(operation);
    });
    return count;
  };
  const unsigned baselineTransfers =
      countTransfers(*frontier->front().ranks.front().module);
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        mlir::ModuleOp module = *variant.ranks.front().module;
        auto function = *module.getOps<mlir::func::FuncOp>().begin();
        return function.getNumResults() == 3 &&
               countTransfers(module) < baselineTransfers &&
               mlir::succeeded(mlir::verify(module));
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       ExactDirectMappedBoundaryMovementIsAnActualFrontierSibling) {
  auto source = makeMatmulProgram();
  ASSERT_TRUE(source);
  auto materialize = [&](wafer::CandidateBoundaryMovementAction movement) {
    std::string failureReason;
    auto candidate = wafer::materializeCompleteRankCandidateTileProgram(
        *source, /*logicalRank=*/0, /*candidateTileSizes=*/{2, 2},
        /*candidateReductionTileSizes=*/{},
        wafer::CandidateTileTraversalKind::ResultDriven,
        wafer::CompleteRankTraversalComposition::Coupled,
        wafer::CandidateTileResidencyAction::KeepSingleRegion, movement,
        wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidate)) << failureReason;
    return candidate;
  };
  auto staged = materialize(wafer::CandidateBoundaryMovementAction::Staged);
  auto direct =
      materialize(wafer::CandidateBoundaryMovementAction::ExactDirectMapped);
  ASSERT_TRUE(mlir::succeeded(staged));
  ASSERT_TRUE(mlir::succeeded(direct));

  auto countLayoutMaterializations = [](mlir::ModuleOp module) {
    unsigned count = 0;
    module.walk([&](wafer::LayoutMaterializeOp) { ++count; });
    return count;
  };
  auto hasDirectMappedLoad = [](mlir::ModuleOp module) {
    bool found = false;
    module.walk([&](wafer::StorageLoadOp load) {
      auto destination =
          mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
      found |=
          destination && wafer::getWaferMemoryAttr(destination).getLayout() !=
                             wafer::MemLayout::Tensor;
    });
    return found;
  };
  EXPECT_LT(countLayoutMaterializations(**direct),
            countLayoutMaterializations(**staged));
  EXPECT_FALSE(hasDirectMappedLoad(**staged));
  EXPECT_TRUE(hasDirectMappedLoad(**direct));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**direct)));

  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        return hasDirectMappedLoad(*variant.ranks.front().module);
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       DiamondFaninUsesOneCoupledTraversalWithAnExactTail) {
  auto source = makeDiamondFaninTailProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);

  auto countTransfers = [](mlir::ModuleOp module) {
    unsigned count = 0;
    module.walk([&](mlir::Operation *operation) {
      count +=
          mlir::isa<wafer::StorageLoadOp, wafer::StorageStoreOp>(operation);
    });
    return count;
  };
  const unsigned baselineTransfers =
      countTransfers(*frontier->front().ranks.front().module);
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        mlir::ModuleOp module = *variant.ranks.front().module;
        llvm::SmallSet<int64_t, 4> oneDimensionalSPMExtents;
        module.walk([&](mlir::memref::AllocOp alloc) {
          mlir::MemRefType type = alloc.getType();
          if (wafer::isWaferSPMMemRefType(type) && type.getRank() == 1)
            oneDimensionalSPMExtents.insert(type.getDimSize(0));
        });
        return countTransfers(module) < baselineTransfers &&
               oneDimensionalSPMExtents.contains(8) &&
               oneDimensionalSPMExtents.contains(2) &&
               mlir::succeeded(mlir::verify(module));
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       SharedInputContractionsReuseTheActualBoundaryTile) {
  auto source = makeSharedInputContractionProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        mlir::ModuleOp module = *variant.ranks.front().module;
        unsigned gemms = 0;
        unsigned loads = 0;
        module.walk([&](wafer::ComputeGemmOp) { ++gemms; });
        module.walk([&](wafer::StorageLoadOp) { ++loads; });
        return gemms == 2 && loads == 3 &&
               mlir::succeeded(mlir::verify(module));
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       TransposedWeightIsFusedWithoutAFullRuntimeTransposeRoundTrip) {
  auto source = makeTransposedWeightContractionProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        mlir::ModuleOp module = *variant.ranks.front().module;
        unsigned gemms = 0;
        bool hasFullTransposeBuffer = false;
        bool storesFullTranspose = false;
        bool hasDirectMappedLoad = false;
        module.walk([&](wafer::ComputeGemmOp) { ++gemms; });
        module.walk([&](mlir::memref::AllocOp alloc) {
          mlir::MemRefType type = alloc.getType();
          hasFullTransposeBuffer |= type.getRank() == 2 &&
                                    type.getDimSize(0) == 8 &&
                                    type.getDimSize(1) == 6;
        });
        module.walk([&](wafer::StorageStoreOp store) {
          auto type =
              mlir::dyn_cast<mlir::MemRefType>(store.getSource().getType());
          storesFullTranspose |= type && type.getRank() == 2 &&
                                 type.getDimSize(0) == 8 &&
                                 type.getDimSize(1) == 6;
        });
        module.walk([&](wafer::StorageLoadOp load) {
          auto type =
              mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
          hasDirectMappedLoad |=
              type && wafer::getWaferMemoryAttr(type).getLayout() !=
                          wafer::MemLayout::Tensor;
        });
        return gemms > 0 && !hasFullTransposeBuffer && !storesFullTranspose &&
               hasDirectMappedLoad && mlir::succeeded(mlir::verify(module));
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       ReadOnlyInvariantLoadMovesOutsideTheConsumerNLoop) {
  auto source = makeInvariantMatmulProgram();
  ASSERT_TRUE(source);
  auto materialize = [&](wafer::CandidateLoopMovementAction loopMovement) {
    std::string failureReason;
    auto candidate = wafer::materializeCompleteRankCandidateTileProgram(
        *source, /*logicalRank=*/0, /*candidateTileSizes=*/{4, 4},
        /*candidateReductionTileSizes=*/{},
        wafer::CandidateTileTraversalKind::ResultDriven,
        wafer::CompleteRankTraversalComposition::Coupled,
        wafer::CandidateTileResidencyAction::KeepSingleRegion,
        wafer::CandidateBoundaryMovementAction::ExactDirectMapped, loopMovement,
        &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidate)) << failureReason;
    return candidate;
  };
  auto constructed =
      materialize(wafer::CandidateLoopMovementAction::AsConstructed);
  auto hoisted = materialize(
      wafer::CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary);
  ASSERT_TRUE(mlir::succeeded(constructed));
  ASSERT_TRUE(mlir::succeeded(hoisted));

  auto maximumLoadLoopDepth = [](mlir::ModuleOp module, int64_t firstExtent,
                                 int64_t secondExtent) {
    unsigned maximum = 0;
    module.walk([&](wafer::StorageLoadOp load) {
      auto type = mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
      if (!type || type.getRank() != 2 || type.getDimSize(0) != firstExtent ||
          type.getDimSize(1) != secondExtent)
        return;
      unsigned depth = 0;
      for (mlir::scf::ForOp loop = load->getParentOfType<mlir::scf::ForOp>();
           loop; loop = loop->getParentOfType<mlir::scf::ForOp>())
        ++depth;
      maximum = std::max(maximum, depth);
    });
    return maximum;
  };
  EXPECT_EQ(maximumLoadLoopDepth(**constructed, 4, 8), 2u);
  EXPECT_EQ(maximumLoadLoopDepth(**hoisted, 4, 8), 1u);
  EXPECT_EQ(maximumLoadLoopDepth(**hoisted, 8, 4), 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**hoisted)));

  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        return maximumLoadLoopDepth(*variant.ranks.front().module, 4, 8) == 1 &&
               maximumLoadLoopDepth(*variant.ranks.front().module, 8, 4) == 2;
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       StructuredControlFlowKeepsBaselineWhenWholeTileIsUnsupported) {
  auto source = makeStructuredConcatProgram();
  ASSERT_TRUE(source);
  std::string materializationFailure;
  auto direct = wafer::materializeCompleteRankCandidateTileProgram(
      *source, /*logicalRank=*/0, /*candidateTileSizes=*/{1, 1, 1, 1},
      /*candidateReductionTileSizes=*/{},
      wafer::CandidateTileTraversalKind::ResultDriven,
      wafer::CompleteRankTraversalComposition::Coupled,
      wafer::CandidateTileResidencyAction::KeepSingleRegion,
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed,
      &materializationFailure);
  ASSERT_TRUE(mlir::succeeded(direct)) << materializationFailure;
  auto full = wafer::materializeCompleteRankCandidateTileProgram(
      *source, /*logicalRank=*/0, /*candidateTileSizes=*/{1, 2, 2, 8},
      /*candidateReductionTileSizes=*/{},
      wafer::CandidateTileTraversalKind::ResultDriven,
      wafer::CompleteRankTraversalComposition::Coupled,
      wafer::CandidateTileResidencyAction::KeepSingleRegion,
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed,
      &materializationFailure);
  EXPECT_TRUE(mlir::failed(full));
  EXPECT_NE(materializationFailure.find(
                "unsupported linalg.generic body op linalg.index"),
            std::string::npos)
      << materializationFailure;
  unsigned directBranches = 0;
  (*direct)->walk([&](mlir::scf::IfOp) { ++directBranches; });
  EXPECT_GT(directBranches, 0u);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(2);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 2;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_EQ(frontier->size(), 1u);
  EXPECT_TRUE(frontier->front().reservedBaseline);
  ASSERT_EQ(frontier->front().ranks.size(), 2u);
  for (const auto &rank : frontier->front().ranks) {
    unsigned branches = 0;
    unsigned loads = 0;
    rank.module.get().walk([&](mlir::scf::IfOp) { ++branches; });
    rank.module.get().walk([&](wafer::StorageLoadOp) { ++loads; });
    EXPECT_GT(branches, 0u);
    EXPECT_GE(loads, 2u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*rank.module)));
  }
}

TEST_F(CoordinatedDataflowSearchTest,
       NonePolicyKeepsOnlyTheMandatoryConservativeVariant) {
  auto source = makeProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(2);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 2;
  config.optimizations = wafer::OptimizationConfig::none();
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_EQ(frontier->size(), 1u);
  EXPECT_TRUE(frontier->front().reservedBaseline);
}

TEST_F(CoordinatedDataflowSearchTest,
       FrontierDigestIsIndependentOfGenerationParallelism) {
  auto firstSource = makeProgram();
  auto secondSource = makeProgram();
  ASSERT_TRUE(firstSource);
  ASSERT_TRUE(secondSource);
  auto firstLedger = wafer::compiler::detail::CoordinatedWorkLedger::create(4);
  auto secondLedger = wafer::compiler::detail::CoordinatedWorkLedger::create(4);
  ASSERT_TRUE(mlir::succeeded(firstLedger));
  ASSERT_TRUE(mlir::succeeded(secondLedger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig serial;
  serial.rankCount = 4;
  serial.candidateParallelism = 1;
  auto parallel = serial;
  parallel.candidateParallelism = 8;
  auto first = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *firstSource, serial, *firstLedger);
  auto second = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *secondSource, parallel, *secondLedger);
  ASSERT_TRUE(mlir::succeeded(first));
  ASSERT_TRUE(mlir::succeeded(second));
  EXPECT_EQ(
      wafer::compiler::detail::computeCoordinatedTileFrontierDigest(*first),
      wafer::compiler::detail::computeCoordinatedTileFrontierDigest(*second));
}

TEST_F(CoordinatedDataflowSearchTest,
       BudgetExhaustionPreservesBaselineReservationAndDigest) {
  auto firstSource = makeProgram();
  auto secondSource = makeProgram();
  ASSERT_TRUE(firstSource);
  ASSERT_TRUE(secondSource);
  constexpr int64_t rankCount = 1;
  auto terminal = wafer::compiler::detail::CoordinatedWorkLedger::
      getTerminalActionUpperBound(rankCount);
  ASSERT_TRUE(mlir::succeeded(terminal));
  ASSERT_TRUE(terminal->getTotal());
  const uint64_t capacity = 2 + *terminal->getTotal();
  auto firstLedger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      rankCount, capacity, /*repairReserve=*/0);
  auto secondLedger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      rankCount, capacity, /*repairReserve=*/0);
  ASSERT_TRUE(mlir::succeeded(firstLedger));
  ASSERT_TRUE(mlir::succeeded(secondLedger));

  wafer::compiler::detail::CoordinatedDataflowSearchConfig serial;
  serial.rankCount = rankCount;
  serial.candidateParallelism = 1;
  auto parallel = serial;
  parallel.candidateParallelism = 8;
  auto first = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *firstSource, serial, *firstLedger);
  auto second = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *secondSource, parallel, *secondLedger);
  ASSERT_TRUE(mlir::succeeded(first));
  ASSERT_TRUE(mlir::succeeded(second));
  ASSERT_EQ(first->size(), 1u);
  ASSERT_EQ(second->size(), 1u);
  EXPECT_TRUE(first->front().reservedBaseline);
  EXPECT_TRUE(second->front().reservedBaseline);
  EXPECT_EQ(
      wafer::compiler::detail::computeCoordinatedTileFrontierDigest(*first),
      wafer::compiler::detail::computeCoordinatedTileFrontierDigest(*second));
  for (const auto *ledger : {&*firstLedger, &*secondLedger}) {
    auto snapshot = ledger->getSnapshot();
    EXPECT_EQ(snapshot.terminalReserved, *terminal->getTotal());
    EXPECT_EQ(snapshot.repairReserved, 0u);
    EXPECT_EQ(snapshot.unreserved, 0u);
  }
}

} // namespace
