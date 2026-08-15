#include "../../lib/Wafer/Compiler/NoCCommunicationAction.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/WholeCardCandidateEvaluation.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

static std::string moduleSnapshot(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

struct TransportCounts {
  unsigned loads = 0;
  unsigned sends = 0;
  unsigned recvs = 0;
  unsigned waits = 0;
};

static TransportCounts countTransport(mlir::ModuleOp module) {
  TransportCounts counts;
  module.walk([&](mlir::Operation *operation) {
    counts.loads += mlir::isa<wafer::InstrRDMAOp>(operation) ? 1u : 0u;
    counts.sends += mlir::isa<wafer::InstrDTESendOp>(operation) ? 1u : 0u;
    counts.recvs += mlir::isa<wafer::InstrDTERecvOp>(operation) ? 1u : 0u;
    counts.waits += mlir::isa<wafer::InstrDTEWaitOp>(operation) ? 1u : 0u;
  });
  return counts;
}

class NoCCommunicationActionTest : public ::testing::Test {
protected:
  NoCCommunicationActionTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseRank() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 4>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %buffer
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    %lane0 = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %lane1 = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %lane2 = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %result0 = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %result1 = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %result2 = memref.alloc()
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.instr.fill %lane0, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %lane1, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %lane2, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.elementwise <neg> %lane0 into %result0
        : memref<4xf16, #wafer.memory<spm, tensor>>
          into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <neg> %lane1 into %result1
        : memref<4xf16, #wafer.memory<spm, tensor>>
          into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.elementwise <neg> %lane2 into %result2
        : memref<4xf16, #wafer.memory<spm, tensor>>
          into memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.ncc_join [0]
    return
  }
}
)mlir",
        mlir::ParserConfig(context.get()));
  }

  wafer::frontend::FrontendProgramVerificationResult
  makeProgram(bool withBoundary = true) {
    wafer::frontend::FrontendProgramVerificationResult program;
    program.logicalRankCount = 4;
    if (!withBoundary)
      return program;

    wafer::frontend::ProgramBoundaryBinding input;
    input.index = 0;
    input.programIndex = 0;
    input.distribution = wafer::frontend::ProgramDistributionKind::Replicated;
    input.globalShape = {4};
    input.localShape = {4};
    input.dtype = "f16";
    for (int64_t rank = 0; rank < 4; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId = rank;
      slice.offsets = {0};
      slice.sizes = {4};
      slice.strides = {1};
      input.rankSlices.push_back(std::move(slice));
    }
    program.distributedInputs.push_back(std::move(input));
    return program;
  }

  void makeParents(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> &owners,
                   llvm::SmallVectorImpl<mlir::ModuleOp> &views) {
    for (unsigned rank = 0; rank < 4; ++rank) {
      mlir::OwningOpRef<mlir::ModuleOp> module = parseRank();
      ASSERT_TRUE(module);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      views.push_back(*module);
      owners.push_back(std::move(module));
    }
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(NoCCommunicationActionTest,
       QueryIsReadOnlyAndMaterializesBothFanoutRecipes) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owners;
  llvm::SmallVector<mlir::ModuleOp, 4> parents;
  makeParents(owners, parents);
  std::vector<std::string> before;
  for (mlir::ModuleOp parent : parents)
    before.push_back(moduleSnapshot(parent));

  wafer::compiler::detail::NoCCommunicationActionProvider provider;
  wafer::compiler::detail::CoordinatedCommunicationActionPoints points;
  std::string failure;
  ASSERT_TRUE(
      mlir::succeeded(provider.query(parents, makeProgram(), points, &failure)))
      << failure;
  ASSERT_EQ(points.size(), 2u);
  EXPECT_EQ(points[0]->getIdentity().providerKey, provider.getStableKey());
  EXPECT_EQ(points[0]->getIdentity().stableOrdinal, 1u);
  EXPECT_EQ(points[1]->getIdentity().providerKey, provider.getStableKey());
  EXPECT_EQ(points[1]->getIdentity().stableOrdinal, 2u);
  for (size_t rank = 0; rank < parents.size(); ++rank)
    EXPECT_EQ(moduleSnapshot(parents[rank]), before[rank]);

  auto materializeAndCheck = [&](size_t pointIndex,
                                 unsigned expectedMaximumSends) {
    mlir::FailureOr<wafer::compiler::detail::CoordinatedCommunicationAction>
        action =
            wafer::compiler::detail::materializeCoordinatedCommunicationAction(
                parents, makeProgram(), *points[pointIndex], &failure);
    ASSERT_TRUE(mlir::succeeded(action)) << failure;
    EXPECT_EQ(action->identity, points[pointIndex]->getIdentity());
    ASSERT_EQ(action->rankModules.size(), parents.size());

    TransportCounts aggregate;
    unsigned maximumSends = 0;
    for (const mlir::OwningOpRef<mlir::ModuleOp> &owner : action->rankModules) {
      mlir::ModuleOp module = *owner;
      EXPECT_FALSE(
          wafer::containsTileDataflowOperations(module.getOperation()));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
      TransportCounts counts = countTransport(module);
      aggregate.loads += counts.loads;
      aggregate.sends += counts.sends;
      aggregate.recvs += counts.recvs;
      aggregate.waits += counts.waits;
      maximumSends = std::max(maximumSends, counts.sends);
    }
    EXPECT_EQ(aggregate.loads, 1u);
    EXPECT_EQ(aggregate.sends, 3u);
    EXPECT_EQ(aggregate.recvs, 3u);
    EXPECT_EQ(aggregate.waits, 6u);
    EXPECT_EQ(maximumSends, expectedMaximumSends);
  };

  materializeAndCheck(/*pointIndex=*/0, /*expectedMaximumSends=*/3);
  materializeAndCheck(/*pointIndex=*/1, /*expectedMaximumSends=*/1);
  for (size_t rank = 0; rank < parents.size(); ++rank)
    EXPECT_EQ(moduleSnapshot(parents[rank]), before[rank]);
}

TEST_F(NoCCommunicationActionTest,
       NonMatchAppendsNoPointAndDoesNotMutateParents) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owners;
  llvm::SmallVector<mlir::ModuleOp, 4> parents;
  makeParents(owners, parents);
  std::vector<std::string> before;
  for (mlir::ModuleOp parent : parents)
    before.push_back(moduleSnapshot(parent));

  wafer::compiler::detail::NoCCommunicationActionProvider provider;
  wafer::compiler::detail::CoordinatedCommunicationActionPoints points;
  std::string failure;
  EXPECT_TRUE(mlir::succeeded(provider.query(
      parents, makeProgram(/*withBoundary=*/false), points, &failure)))
      << failure;
  EXPECT_TRUE(points.empty());
  for (size_t rank = 0; rank < parents.size(); ++rank)
    EXPECT_EQ(moduleSnapshot(parents[rank]), before[rank]);
}

TEST_F(NoCCommunicationActionTest,
       PointReprovesCapabilityWithoutRetainingQueriedOperations) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owners;
  llvm::SmallVector<mlir::ModuleOp, 4> parents;
  makeParents(owners, parents);

  wafer::compiler::detail::NoCCommunicationActionProvider provider;
  wafer::compiler::detail::CoordinatedCommunicationActionPoints points;
  std::string failure;
  ASSERT_TRUE(
      mlir::succeeded(provider.query(parents, makeProgram(), points, &failure)))
      << failure;
  ASSERT_EQ(points.size(), 2u);

  for (mlir::ModuleOp parent : parents) {
    llvm::SmallVector<wafer::InstrRDMAOp, 2> loads;
    parent.walk([&](wafer::InstrRDMAOp load) { loads.push_back(load); });
    for (wafer::InstrRDMAOp load : llvm::reverse(loads))
      load.erase();
    ASSERT_TRUE(mlir::succeeded(mlir::verify(parent)));
  }
  std::vector<std::string> current;
  for (mlir::ModuleOp parent : parents)
    current.push_back(moduleSnapshot(parent));

  auto action =
      wafer::compiler::detail::materializeCoordinatedCommunicationAction(
          parents, makeProgram(), *points.front(), &failure);
  EXPECT_TRUE(mlir::failed(action));
  EXPECT_FALSE(failure.empty());
  for (size_t rank = 0; rank < parents.size(); ++rank)
    EXPECT_EQ(moduleSnapshot(parents[rank]), current[rank]);
}

TEST_F(NoCCommunicationActionTest,
       ActualNoCProviderComposesWithCommonWorkerPlacement) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> owners;
  llvm::SmallVector<mlir::ModuleOp, 4> parents;
  makeParents(owners, parents);
  std::vector<std::string> before;
  for (mlir::ModuleOp parent : parents)
    before.push_back(moduleSnapshot(parent));

  wafer::compiler::detail::NoCCommunicationActionProvider provider;
  llvm::SmallVector<
      const wafer::compiler::detail::CoordinatedCommunicationActionProvider *,
      1>
      providers{&provider};
  wafer::frontend::FrontendProgramVerificationResult program = makeProgram();
  wafer::compiler::detail::CoordinatedScheduleRecipeStatistics statistics;
  uint64_t work = 0;
  auto actions = wafer::compiler::detail::deriveCoordinatedScheduleActions(
      parents, wafer::OptimizationConfig::production(), &work,
      wafer::compiler::detail::CoordinatedScheduleActionFamily::Production,
      &statistics, &program, providers);
  ASSERT_TRUE(mlir::succeeded(actions));

  bool sawNoCWorkerComposition = false;
  for (const auto &action : *actions) {
    if (!action.communicationActionKey ||
        action.workerPlacementKind !=
            wafer::compiler::detail::CoordinatedWorkerPlacementKind::
                DisjointComponents)
      continue;
    sawNoCWorkerComposition = true;
    EXPECT_EQ(action.communicationActionKey->providerKey,
              provider.getStableKey());
    EXPECT_TRUE(action.communicationActionKey->stableOrdinal == 1u ||
                action.communicationActionKey->stableOrdinal == 2u);
    ASSERT_EQ(action.rankModules.size(), parents.size());

    TransportCounts aggregate;
    for (const auto &owner : action.rankModules) {
      mlir::ModuleOp module = *owner;
      EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
      bool hasNonzeroWorker = false;
      module.walk([&](mlir::Operation *operation) {
        std::optional<wafer::NCCWorker> worker =
            wafer::getNCCIssueWorker(operation);
        hasNonzeroWorker |= worker && *worker != wafer::NCCWorker::Worker0;
      });
      EXPECT_TRUE(hasNonzeroWorker);
      TransportCounts counts = countTransport(module);
      aggregate.loads += counts.loads;
      aggregate.sends += counts.sends;
      aggregate.recvs += counts.recvs;
      aggregate.waits += counts.waits;
    }
    EXPECT_EQ(aggregate.loads, 1u);
    EXPECT_EQ(aggregate.sends, 3u);
    EXPECT_EQ(aggregate.recvs, 3u);
    EXPECT_EQ(aggregate.waits, 6u);
  }
  EXPECT_TRUE(sawNoCWorkerComposition);
  EXPECT_EQ(work, statistics.actualRankClones);
  EXPECT_LE(statistics.successfulActionClones,
            wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions);
  for (size_t rank = 0; rank < parents.size(); ++rank)
    EXPECT_EQ(moduleSnapshot(parents[rank]), before[rank]);
}

} // namespace
