#include "../../lib/Wafer/Compiler/NoCIntermediateDataflow.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/NoCResidentDataflow.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/Common/OpVerifierUtils.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

static unsigned countRDMA(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](wafer::InstrRDMAOp) { ++count; });
  return count;
}

static unsigned countWDMA(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](wafer::InstrWDMAOp) { ++count; });
  return count;
}

static unsigned countElementwise(mlir::ModuleOp module,
                                 wafer::InstrElementwiseKind kind) {
  unsigned count = 0;
  module.walk([&](wafer::InstrElementwiseOp operation) {
    if (operation.getKind() == kind)
      ++count;
  });
  return count;
}

static unsigned countDDRAllocations(mlir::ModuleOp module) {
  unsigned count = 0;
  module.walk([&](mlir::memref::AllocOp allocation) {
    if (wafer::detail::hasWaferMemorySpace(allocation.getType(),
                                           wafer::MemorySpace::DDR))
      ++count;
  });
  return count;
}

static uint64_t countTerminalOperations(mlir::ModuleOp module) {
  uint64_t count = 0;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp,
                  wafer::SyncNCCJoinOp>(operation))
      ++count;
  });
  return count;
}

static std::string moduleSnapshot(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

class NoCIntermediateDataflowTest : public ::testing::Test {
protected:
  NoCIntermediateDataflowTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseTwoCutRank() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 16>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(
      %input: memref<8xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %input : memref<8xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<8xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%source: memref<8xf32, #wafer.memory<ddr, tensor>>):
      %tile0 = memref.subview %source[0] [4] [1]
          : memref<8xf32, #wafer.memory<ddr, tensor>>
         to memref<4xf32, strided<[1], offset: 0>,
                   #wafer.memory<ddr, tensor>>
      %tile1 = memref.subview %source[4] [4] [1]
          : memref<8xf32, #wafer.memory<ddr, tensor>>
         to memref<4xf32, strided<[1], offset: 4>,
                   #wafer.memory<ddr, tensor>>
      %loaded0 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %produced0 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65792>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %spill0 = memref.alloc()
          : memref<4xf32, #wafer.memory<ddr, tensor>>
      %consumed0 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<66048>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %result0 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<66304>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %tile0 to %loaded0
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, strided<[1], offset: 0>,
                   #wafer.memory<ddr, tensor>>
         to memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded0, %loaded0 into %produced0
          : memref<4xf32, #wafer.memory<spm, tensor>>,
            memref<4xf32, #wafer.memory<spm, tensor>>
        into memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %produced0 to %spill0
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
         to memref<4xf32, #wafer.memory<ddr, tensor>>
      wafer.instr.rdma %spill0 to %consumed0
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, #wafer.memory<ddr, tensor>>
         to memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <mul> %consumed0, %consumed0 into %result0
          : memref<4xf32, #wafer.memory<spm, tensor>>,
            memref<4xf32, #wafer.memory<spm, tensor>>
        into memref<4xf32, #wafer.memory<spm, tensor>>
      memref.dealloc %spill0
          : memref<4xf32, #wafer.memory<ddr, tensor>>

      %loaded1 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<66560>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %produced1 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<66816>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %spill1 = memref.alloc()
          : memref<4xf32, #wafer.memory<ddr, tensor>>
      %consumed1 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<67072>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %result1 = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<67328>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %tile1 to %loaded1
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, strided<[1], offset: 4>,
                   #wafer.memory<ddr, tensor>>
         to memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded1, %loaded1 into %produced1
          : memref<4xf32, #wafer.memory<spm, tensor>>,
            memref<4xf32, #wafer.memory<spm, tensor>>
        into memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %produced1 to %spill1
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
         to memref<4xf32, #wafer.memory<ddr, tensor>>
      wafer.instr.rdma %spill1 to %consumed1
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, #wafer.memory<ddr, tensor>>
         to memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <mul> %consumed1, %consumed1 into %result1
          : memref<4xf32, #wafer.memory<spm, tensor>>,
            memref<4xf32, #wafer.memory<spm, tensor>>
        into memref<4xf32, #wafer.memory<spm, tensor>>
      memref.dealloc %spill1
          : memref<4xf32, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %source
          : memref<8xf32, #wafer.memory<ddr, tensor>>
    }
    wafer.instr.ncc_join [0]
    return
  }
})mlir",
        mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseOutputPublisherRank() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  func.func @main(
      %input: memref<4xf32, #wafer.memory<ddr, tensor>>,
      %output: memref<4xf32, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %input, %output :
          memref<4xf32, #wafer.memory<ddr, tensor>>,
          memref<4xf32, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf32, #wafer.memory<ddr, tensor>>) {
    ^bb0(%source: memref<4xf32, #wafer.memory<ddr, tensor>>,
         %required: memref<4xf32, #wafer.memory<ddr, tensor>>):
      %loaded = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      %produced = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65792>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %source to %loaded
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, #wafer.memory<ddr, tensor>>
         to memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.elementwise <add> %loaded, %loaded into %produced
          : memref<4xf32, #wafer.memory<spm, tensor>>,
            memref<4xf32, #wafer.memory<spm, tensor>>
        into memref<4xf32, #wafer.memory<spm, tensor>>
      wafer.instr.wdma %produced to %required
          {byte_count = 16 : i64, inner_bytes = 16 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<4xf32, #wafer.memory<spm, tensor>>
         to memref<4xf32, #wafer.memory<ddr, tensor>>
      wafer.tile.yield %required
          : memref<4xf32, #wafer.memory<ddr, tensor>>
    }
    return
  }
})mlir",
        mlir::ParserConfig(context.get()));
  }

  wafer::frontend::FrontendProgramVerificationResult makeOutputProgram() {
    wafer::frontend::FrontendProgramVerificationResult result;
    result.logicalRankCount = 2;
    result.programUserInputCount = 1;
    auto makeBinding = [](int64_t index) {
      wafer::frontend::ProgramBoundaryBinding binding;
      binding.index = index;
      binding.programIndex = index;
      binding.distribution =
          wafer::frontend::ProgramDistributionKind::Replicated;
      binding.globalShape = {4};
      binding.localShape = {4};
      binding.dtype = "f32";
      for (int64_t rank = 0; rank < 2; ++rank) {
        wafer::frontend::ProgramRankSlice slice;
        slice.logicalRank = rank;
        slice.replicaId = rank;
        slice.offsets = {0};
        slice.sizes = {4};
        slice.strides = {1};
        binding.rankSlices.push_back(std::move(slice));
      }
      return binding;
    };
    result.distributedInputs.push_back(makeBinding(0));
    result.distributedOutputs.push_back(makeBinding(0));
    return result;
  }

  wafer::frontend::FrontendProgramVerificationResult
  makeProgram(bool partitioned = false) {
    wafer::frontend::FrontendProgramVerificationResult result;
    result.logicalRankCount = 16;
    result.programUserInputCount = 1;
    wafer::frontend::ProgramBoundaryBinding input;
    input.index = 0;
    input.programIndex = 0;
    input.distribution =
        partitioned ? wafer::frontend::ProgramDistributionKind::Partitioned
                    : wafer::frontend::ProgramDistributionKind::Replicated;
    input.globalShape = {partitioned ? 128 : 8};
    input.localShape = {8};
    input.dtype = "f32";
    for (int64_t rank = 0; rank < 16; ++rank) {
      wafer::frontend::ProgramRankSlice slice;
      slice.logicalRank = rank;
      slice.replicaId = partitioned ? 0 : rank;
      slice.offsets = {partitioned ? rank * 8 : 0};
      slice.sizes = {8};
      slice.strides = {1};
      input.rankSlices.push_back(std::move(slice));
    }
    result.distributedInputs.push_back(std::move(input));
    return result;
  }

  llvm::Expected<wafer::compiler::ExecutionConfig> makeConfig() {
    return wafer::compiler::ExecutionConfig::createForSingleCard(
        16, wafer::RuntimeLaunchKind::Kernel);
  }

  void makeFirstProducerDifferent(mlir::ModuleOp module) {
    wafer::InstrWDMAOp firstStore;
    module.walk([&](wafer::InstrWDMAOp store) {
      if (!firstStore)
        firstStore = store;
    });
    ASSERT_TRUE(firstStore);
    wafer::InstrElementwiseOp producer;
    module.walk([&](wafer::InstrElementwiseOp candidate) {
      if (!producer && candidate.getDest() == firstStore.getSource())
        producer = candidate;
    });
    ASSERT_TRUE(producer);
    producer.setKind(wafer::InstrElementwiseKind::Mul);
  }

  void makeFirstProducerReadItsPriorState(mlir::ModuleOp module,
                                          float initialValue) {
    wafer::InstrWDMAOp firstStore;
    module.walk([&](wafer::InstrWDMAOp store) {
      if (!firstStore)
        firstStore = store;
    });
    ASSERT_TRUE(firstStore);
    wafer::InstrElementwiseOp producer;
    module.walk([&](wafer::InstrElementwiseOp candidate) {
      if (!producer && candidate.getDest() == firstStore.getSource())
        producer = candidate;
    });
    ASSERT_TRUE(producer);
    mlir::OpBuilder builder(producer);
    auto initial = builder.create<mlir::arith::ConstantOp>(
        producer.getLoc(), builder.getF32FloatAttr(initialValue));
    builder.create<wafer::InstrFillOp>(
        producer.getLoc(), producer.getDest(), initial.getResult(),
        wafer::FillDomainAttr{}, wafer::NCCWorker::Worker0);
    producer->setOperand(0, producer.getDest());
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
  }

  void replaceProducersWithConstantFills(mlir::ModuleOp module, float value) {
    llvm::SmallVector<wafer::InstrWDMAOp, 2> stores;
    module.walk([&](wafer::InstrWDMAOp store) { stores.push_back(store); });
    ASSERT_EQ(stores.size(), 2u);
    for (wafer::InstrWDMAOp store : stores) {
      wafer::InstrElementwiseOp producer;
      module.walk([&](wafer::InstrElementwiseOp candidate) {
        if (!producer && candidate.getDest() == store.getSource())
          producer = candidate;
      });
      ASSERT_TRUE(producer);
      mlir::OpBuilder builder(producer);
      auto constant = builder.create<mlir::arith::ConstantOp>(
          producer.getLoc(), builder.getF32FloatAttr(value));
      builder.create<wafer::InstrFillOp>(
          producer.getLoc(), producer.getDest(), constant.getResult(),
          wafer::FillDomainAttr{}, wafer::NCCWorker::Worker0);
      producer.erase();
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
  }

  void overwriteFirstProducerThroughAliasAfterSpill(mlir::ModuleOp module) {
    wafer::InstrWDMAOp firstStore;
    module.walk([&](wafer::InstrWDMAOp store) {
      if (!firstStore)
        firstStore = store;
    });
    ASSERT_TRUE(firstStore);
    mlir::OpBuilder builder(firstStore);
    builder.setInsertionPointAfter(firstStore);
    auto alias = builder.create<mlir::memref::CastOp>(
        firstStore.getLoc(), firstStore.getSource().getType(),
        firstStore.getSource());
    auto zero = builder.create<mlir::arith::ConstantOp>(
        firstStore.getLoc(), builder.getF32FloatAttr(0.0));
    builder.create<wafer::InstrFillOp>(firstStore.getLoc(), alias, zero,
                                       wafer::FillDomainAttr{},
                                       wafer::NCCWorker::Worker0);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));
  }

  size_t findDirectIntermediateOwner() {
    std::vector<mlir::OwningOpRef<mlir::ModuleOp>> ownedModules;
    llvm::SmallVector<mlir::ModuleOp, 16> modules;
    for (size_t rank = 0; rank < 16; ++rank) {
      auto module = parseTwoCutRank();
      EXPECT_TRUE(module);
      if (!module)
        return 0;
      modules.push_back(*module);
      ownedModules.push_back(std::move(module));
    }
    int64_t communicationId = 0;
    EXPECT_EQ(wafer::compiler::detail::materializeNoCIntermediateHandoffs(
                  modules, makeProgram(), communicationId,
                  wafer::compiler::detail::NoCFanoutKind::Direct),
              2u);
    for (auto [rank, module] : llvm::enumerate(modules)) {
      unsigned sends = 0;
      module.walk([&](wafer::CommPeerSendOp send) {
        if (send.getMessage().getRound() == 1)
          ++sends;
      });
      if (sends > 1)
        return rank;
    }
    ADD_FAILURE() << "direct intermediate materialization had no fanout owner";
    return 0;
  }

  void padToTerminalBudget(mlir::ModuleOp module) {
    wafer::InstrElementwiseOp paddingTemplate;
    module.walk([&](wafer::InstrElementwiseOp candidate) {
      paddingTemplate = candidate;
    });
    ASSERT_TRUE(paddingTemplate);
    const uint64_t initial = countTerminalOperations(module);
    ASSERT_LE(initial, wafer::detail::kStaticTerminalOperationBudget);
    mlir::OpBuilder builder(paddingTemplate);
    builder.setInsertionPointAfter(paddingTemplate);
    for (uint64_t count = initial;
         count < wafer::detail::kStaticTerminalOperationBudget; ++count)
      builder.clone(*paddingTemplate.getOperation());
    ASSERT_EQ(countTerminalOperations(module),
              wafer::detail::kStaticTerminalOperationBudget);
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(NoCIntermediateDataflowTest,
       ReplacesReplicatedIntermediateSpillsAndRotatesOwners) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers) {
    auto module = parseTwoCutRank();
    ASSERT_TRUE(module);
    frontier.push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }
  auto config = makeConfig();
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, makeProgram(), *config, &failure)))
      << failure;
  for (const auto &frontier : frontiers) {
    ASSERT_EQ(frontier.size(), 5u);
    for (size_t unplacedIndex : {1u, 3u}) {
      const auto &unplaced = frontier[unplacedIndex];
      const auto &workerPlaced = frontier[unplacedIndex + 1];
      EXPECT_EQ(unplaced.stableOrdinal, workerPlaced.stableOrdinal);
      EXPECT_EQ(unplaced.artifactKind, wafer::RankArtifactKind::Resident);
      EXPECT_EQ(workerPlaced.artifactKind, wafer::RankArtifactKind::Resident);
      EXPECT_EQ(unplaced.workerPlacementKind,
                wafer::RankWorkerPlacementKind::Unplaced);
      EXPECT_EQ(workerPlaced.workerPlacementKind,
                wafer::RankWorkerPlacementKind::DisjointComponents);
      EXPECT_EQ(workerPlaced.workerPlacementPlanOrdinal, 1u);
    }
  }

  constexpr size_t kIntermediateDirect = 1;
  unsigned baselineRDMA = 0;
  unsigned baselineWDMA = 0;
  unsigned baselineDDRAllocations = 0;
  unsigned residentRDMA = 0;
  unsigned residentWDMA = 0;
  unsigned residentDDRAllocations = 0;
  unsigned residentProducers = 0;
  unsigned residentConsumers = 0;
  unsigned sends = 0;
  unsigned recvs = 0;
  std::set<size_t> ownerRanks;
  std::set<std::pair<int64_t, size_t>> ownerByCommunicationId;
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    mlir::ModuleOp baseline = *frontiers[rank][0].module;
    mlir::ModuleOp resident = *frontiers[rank][kIntermediateDirect].module;
    baselineRDMA += countRDMA(baseline);
    baselineWDMA += countWDMA(baseline);
    baselineDDRAllocations += countDDRAllocations(baseline);
    residentRDMA += countRDMA(resident);
    residentWDMA += countWDMA(resident);
    residentDDRAllocations += countDDRAllocations(resident);
    residentProducers +=
        countElementwise(resident, wafer::InstrElementwiseKind::Add);
    residentConsumers +=
        countElementwise(resident, wafer::InstrElementwiseKind::Mul);
    resident.walk([&](wafer::InstrDTESendOp send) {
      if (send.getMessage().getRound() == 1) {
        ++sends;
        ownerRanks.insert(rank);
        ownerByCommunicationId.insert(
            {send.getMessage().getCommunicationId(), rank});
      }
    });
    resident.walk([&](wafer::InstrDTERecvOp recv) {
      if (recv.getMessage().getRound() == 1)
        ++recvs;
    });
  }
  EXPECT_EQ(baselineRDMA, 64u);
  EXPECT_EQ(baselineWDMA, 32u);
  EXPECT_EQ(baselineDDRAllocations, 32u);
  EXPECT_EQ(residentRDMA, 2u);
  EXPECT_EQ(residentWDMA, 0u);
  EXPECT_EQ(residentDDRAllocations, 0u);
  EXPECT_EQ(sends, 30u);
  EXPECT_EQ(recvs, 30u);
  EXPECT_EQ(ownerRanks, (std::set<size_t>{0u, 2u}));
  EXPECT_EQ(ownerByCommunicationId,
            (std::set<std::pair<int64_t, size_t>>{{0, 0u}, {1, 2u}}));
  // Each exact intermediate tile now has one compute producer and sixteen
  // consumers. Peer producer closures, including their now-dead input RDMA,
  // are removed from the same accepted IR.
  EXPECT_EQ(residentProducers, 2u);
  EXPECT_EQ(residentConsumers, 32u);
}

TEST_F(NoCIntermediateDataflowTest,
       RoutesEquivalentProducedValueToEveryRequiredOutputPublisher) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> ownedModules;
  llvm::SmallVector<mlir::ModuleOp, 2> modules;
  for (size_t rank = 0; rank < 2; ++rank) {
    auto module = parseOutputPublisherRank();
    ASSERT_TRUE(module);
    modules.push_back(*module);
    ownedModules.push_back(std::move(module));
  }

  int64_t communicationId = 12;
  EXPECT_EQ(wafer::compiler::detail::materializeNoCOutputPublications(
                modules, makeOutputProgram(), communicationId,
                wafer::compiler::detail::NoCFanoutKind::Direct),
            1u);
  EXPECT_EQ(communicationId, 13);

  unsigned loads = 0;
  unsigned producers = 0;
  unsigned stores = 0;
  unsigned sends = 0;
  unsigned recvs = 0;
  for (auto [rank, module] : llvm::enumerate(modules)) {
    loads += countRDMA(module);
    producers += countElementwise(module, wafer::InstrElementwiseKind::Add);
    stores += countWDMA(module);
    module.walk([&](wafer::CommPeerSendOp send) {
      ++sends;
      EXPECT_EQ(rank, 0u);
      EXPECT_EQ(send.getPeerAttr().getInt(), 1);
      EXPECT_EQ(send.getMessage().getPhase(),
                wafer::DTEProtocolPhase::PeerDataflow);
      EXPECT_EQ(send.getMessage().getRound(), 2);
    });
    module.walk([&](wafer::CommPeerRecvOp recv) {
      ++recvs;
      EXPECT_EQ(rank, 1u);
      EXPECT_EQ(recv.getPeerAttr().getInt(), 0);
      EXPECT_EQ(recv.getMessage().getPhase(),
                wafer::DTEProtocolPhase::PeerDataflow);
      EXPECT_EQ(recv.getMessage().getRound(), 2);
    });
    EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
  }
  EXPECT_EQ(loads, 1u);
  EXPECT_EQ(producers, 1u);
  EXPECT_EQ(stores, 2u);
  EXPECT_EQ(sends, 1u);
  EXPECT_EQ(recvs, 1u);
}

TEST_F(NoCIntermediateDataflowTest,
       RejectsIncompleteRequiredOutputDescriptorAtomically) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> ownedModules;
  llvm::SmallVector<mlir::ModuleOp, 2> modules;
  for (size_t rank = 0; rank < 2; ++rank) {
    auto module = parseOutputPublisherRank();
    ASSERT_TRUE(module);
    modules.push_back(*module);
    ownedModules.push_back(std::move(module));
  }
  wafer::InstrWDMAOp incomplete;
  modules.back().walk([&](wafer::InstrWDMAOp store) { incomplete = store; });
  ASSERT_TRUE(incomplete);
  incomplete->setAttr(
      "byte_count",
      mlir::IntegerAttr::get(mlir::IntegerType::get(context.get(), 64), 8));
  std::vector<std::string> snapshots;
  for (mlir::ModuleOp module : modules)
    snapshots.push_back(moduleSnapshot(module));

  int64_t communicationId = 20;
  EXPECT_EQ(wafer::compiler::detail::materializeNoCOutputPublications(
                modules, makeOutputProgram(), communicationId,
                wafer::compiler::detail::NoCFanoutKind::Direct),
            0u);
  EXPECT_EQ(communicationId, 20);
  for (size_t rank = 0; rank < modules.size(); ++rank)
    EXPECT_EQ(moduleSnapshot(modules[rank]), snapshots[rank]);
}

TEST_F(NoCIntermediateDataflowTest,
       KeepsNonEquivalentProducerSpillOutsideCompatibleGroup) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    auto module = parseTwoCutRank();
    ASSERT_TRUE(module);
    if (rank == 15)
      makeFirstProducerDifferent(*module);
    frontiers[rank].push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }
  auto config = makeConfig();
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, makeProgram(), *config, &failure)))
      << failure;
  for (const auto &frontier : frontiers)
    ASSERT_EQ(frontier.size(), 3u);

  constexpr size_t kIntermediateDirect = 1;
  unsigned aggregateWDMA = 0;
  unsigned aggregateDDRAllocations = 0;
  for (const auto &frontier : frontiers) {
    aggregateWDMA += countWDMA(*frontier[kIntermediateDirect].module);
    aggregateDDRAllocations +=
        countDDRAllocations(*frontier[kIntermediateDirect].module);
  }
  EXPECT_EQ(aggregateWDMA, 1u);
  EXPECT_EQ(aggregateDDRAllocations, 1u);
  EXPECT_EQ(countWDMA(*frontiers[15][kIntermediateDirect].module), 1u);
  EXPECT_EQ(countDDRAllocations(*frontiers[15][kIntermediateDirect].module),
            1u);
}

TEST_F(NoCIntermediateDataflowTest,
       DoesNotInventIntermediateReuseAcrossPartitionedGlobalTiles) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers) {
    auto module = parseTwoCutRank();
    ASSERT_TRUE(module);
    frontier.push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }
  auto config = makeConfig();
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, makeProgram(/*partitioned=*/true), *config, &failure)))
      << failure;
  for (const auto &frontier : frontiers) {
    ASSERT_EQ(frontier.size(), 1u);
    EXPECT_EQ(countWDMA(*frontier.front().module), 2u);
    EXPECT_EQ(countDDRAllocations(*frontier.front().module), 2u);
  }
}

TEST_F(NoCIntermediateDataflowTest,
       DifferentReadModifyWriteInitialStateKeepsThatSpillCut) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> ownedModules;
  llvm::SmallVector<mlir::ModuleOp, 2> modules;
  for (size_t rank = 0; rank < 2; ++rank) {
    auto module = parseTwoCutRank();
    ASSERT_TRUE(module);
    makeFirstProducerReadItsPriorState(*module, rank == 0 ? 0.0f : 1.0f);
    modules.push_back(*module);
    ownedModules.push_back(std::move(module));
  }
  auto program = makeProgram();
  program.logicalRankCount = 2;
  program.distributedInputs.front().rankSlices.resize(2);
  int64_t communicationId = 0;
  EXPECT_EQ(wafer::compiler::detail::materializeNoCIntermediateHandoffs(
                modules, program, communicationId,
                wafer::compiler::detail::NoCFanoutKind::Direct),
            1u);
  unsigned aggregateWDMA = 0;
  unsigned aggregateDDRAllocations = 0;
  for (mlir::ModuleOp module : modules) {
    aggregateWDMA += countWDMA(module);
    aggregateDDRAllocations += countDDRAllocations(module);
  }
  EXPECT_EQ(aggregateWDMA, 2u);
  EXPECT_EQ(aggregateDDRAllocations, 2u);
}

TEST_F(NoCIntermediateDataflowTest,
       BoundaryFreePureDefinitionsCanShareResidentIntermediate) {
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (auto &frontier : frontiers) {
    auto module = parseTwoCutRank();
    ASSERT_TRUE(module);
    replaceProducersWithConstantFills(*module, 1.0f);
    frontier.push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }
  wafer::frontend::FrontendProgramVerificationResult program;
  program.logicalRankCount = 16;
  auto config = makeConfig();
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, program, *config, &failure)))
      << failure;
  for (const auto &frontier : frontiers) {
    ASSERT_GT(frontier.size(), 1u);
    bool foundZeroSpill = false;
    for (const auto &candidate : llvm::drop_begin(frontier))
      foundZeroSpill |= countWDMA(*candidate.module) == 0 &&
                        countDDRAllocations(*candidate.module) == 0;
    EXPECT_TRUE(foundZeroSpill);
  }
}

TEST_F(NoCIntermediateDataflowTest,
       AliasOverwriteAfterSpillPreventsOwnerBufferReuse) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> ownedModules;
  llvm::SmallVector<mlir::ModuleOp, 2> modules;
  for (size_t rank = 0; rank < 2; ++rank) {
    auto module = parseTwoCutRank();
    ASSERT_TRUE(module);
    overwriteFirstProducerThroughAliasAfterSpill(*module);
    modules.push_back(*module);
    ownedModules.push_back(std::move(module));
  }
  auto program = makeProgram();
  program.logicalRankCount = 2;
  program.distributedInputs.front().rankSlices.resize(2);
  int64_t communicationId = 0;
  EXPECT_EQ(wafer::compiler::detail::materializeNoCIntermediateHandoffs(
                modules, program, communicationId,
                wafer::compiler::detail::NoCFanoutKind::Direct),
            1u);
  unsigned aggregateWDMA = 0;
  unsigned aggregateDDRAllocations = 0;
  for (mlir::ModuleOp module : modules) {
    aggregateWDMA += countWDMA(module);
    aggregateDDRAllocations += countDDRAllocations(module);
  }
  EXPECT_EQ(aggregateWDMA, 2u);
  EXPECT_EQ(aggregateDDRAllocations, 2u);
}

TEST_F(NoCIntermediateDataflowTest,
       ExhaustingTheLastCommunicationIdUsesTheExplicitSentinel) {
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> ownedModules;
  llvm::SmallVector<mlir::ModuleOp, 2> modules;
  for (size_t rank = 0; rank < 2; ++rank) {
    auto module = parseTwoCutRank();
    ASSERT_TRUE(module);
    if (rank == 1)
      makeFirstProducerDifferent(*module);
    modules.push_back(*module);
    ownedModules.push_back(std::move(module));
  }
  auto program = makeProgram();
  program.logicalRankCount = 2;
  program.distributedInputs.front().rankSlices.resize(2);
  int64_t communicationId = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(wafer::compiler::detail::materializeNoCIntermediateHandoffs(
                modules, program, communicationId,
                wafer::compiler::detail::NoCFanoutKind::Direct),
            1u);
  EXPECT_EQ(communicationId, -1);
}

TEST_F(NoCIntermediateDataflowTest,
       LateTerminalGateRejectsDirectTupleWithoutMutatingItsSeeds) {
  const size_t paddedOwner = findDirectIntermediateOwner();
  std::vector<wafer::compiler::detail::RankVariantFrontier> frontiers(16);
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    auto module = parseTwoCutRank();
    ASSERT_TRUE(module);
    if (rank == paddedOwner)
      padToTerminalBudget(*module);
    frontiers[rank].push_back(
        {std::move(module), /*stableOrdinal=*/0, wafer::RankArtifactKind::Spill,
         /*reservedBaseline=*/true, wafer::RankBufferingKind::Single,
         /*bufferingPlanOrdinal=*/0});
  }
  std::vector<std::string> snapshots;
  for (const auto &frontier : frontiers)
    snapshots.push_back(moduleSnapshot(*frontier.front().module));

  auto config = makeConfig();
  ASSERT_TRUE(static_cast<bool>(config)) << llvm::toString(config.takeError());
  std::string failure;
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::appendNoCResidentDataflowCandidates(
          frontiers, makeProgram(), *config, &failure)))
      << failure;
  for (size_t rank = 0; rank < frontiers.size(); ++rank) {
    ASSERT_GE(frontiers[rank].size(), 1u);
    EXPECT_EQ(moduleSnapshot(*frontiers[rank].front().module), snapshots[rank]);
    for (const auto &candidate : llvm::drop_begin(frontiers[rank])) {
      unsigned roundOneSends = 0;
      (*candidate.module).walk([&](wafer::InstrDTESendOp send) {
        if (send.getMessage().getRound() == 1)
          ++roundOneSends;
      });
      EXPECT_LE(roundOneSends, 1u);
    }
  }
}

} // namespace
