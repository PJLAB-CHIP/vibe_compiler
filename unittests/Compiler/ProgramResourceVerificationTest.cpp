//===- ProgramResourceVerificationTest.cpp ------------------------------===//

#include "Wafer/Compiler/Testing.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

#include <memory>

namespace {

class ProgramResourceVerificationTest : public ::testing::Test {
protected:
  ProgramResourceVerificationTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    wafer::registerWaferCoreDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
    auto created = wafer::compiler::ExecutionConfig::createForSingleCard(
        1, wafer::RuntimeLaunchKind::Kernel);
    EXPECT_TRUE(static_cast<bool>(created));
    if (created)
      config = std::make_unique<wafer::compiler::ExecutionConfig>(*created);
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  parseWithoutVerification(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get(), /*verifyAfterParse=*/false));
  }

  llvm::SmallVector<mlir::ModuleOp, 16>
  completeTileDomain(mlir::OwningOpRef<mlir::ModuleOp> activeModule) {
    constexpr llvm::StringLiteral idleSource = R"mlir(
module {
  func.func @main() {
    return
  }
}
)mlir";

    ownedModules.clear();
    ownedModules.push_back(std::move(activeModule));
    for (int64_t tile = 1; tile < config->getTileCount(); ++tile) {
      auto idleModule = parse(idleSource);
      if (!idleModule) {
        ADD_FAILURE() << "failed to parse idle Tile module";
        return {};
      }
      ownedModules.push_back(std::move(idleModule));
    }

    llvm::SmallVector<mlir::ModuleOp, 16> modules;
    modules.reserve(ownedModules.size());
    for (const auto &module : ownedModules)
      modules.push_back(*module);
    return modules;
  }

  mlir::FailureOr<wafer::analysis::CardInstructionProgramCost>
  verifyResources(llvm::ArrayRef<mlir::ModuleOp> modules,
                  const wafer::compiler::ExecutionConfig &executionConfig) {
    llvm::SmallVector<wafer::TileId, 16> tileIds;
    tileIds.reserve(modules.size());
    for (size_t tile = 0; tile < modules.size(); ++tile)
      tileIds.emplace_back(static_cast<int64_t>(tile));
    return wafer::compiler::testing::verifyProgramResources(
        modules, tileIds, executionConfig);
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<wafer::compiler::ExecutionConfig> config;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> ownedModules;
};

TEST_F(ProgramResourceVerificationTest,
       VerifiesExactResourceSummaryForCompleteTileDomain) {
  auto module = parse(R"mlir(
module {
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %buffer
        {byte_count = 8 : i64, inner_bytes = 8 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<4xf16, #wafer.memory<ddr, tensor>>
       to memref<4xf16, #wafer.memory<spm, tensor>>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(config);
  llvm::SmallVector<mlir::ModuleOp, 16> modules =
      completeTileDomain(std::move(module));
  ASSERT_EQ(modules.size(), 16u);

  auto resources = verifyResources(modules, *config);
  ASSERT_TRUE(mlir::succeeded(resources));
  EXPECT_EQ(resources->tileCosts.size(), 16u);
  EXPECT_EQ(resources->aggregateDDRReadBytes.value, 8u);
  EXPECT_EQ(resources->minimumHopLinkByteDemand.value, 0u);
  EXPECT_EQ(resources->maximumTileSPMHighWaterBytes.value, 8u);
}

TEST_F(ProgramResourceVerificationTest,
       RejectsSPMUseAcrossTileResidencyDomains) {
  auto module = parseWithoutVerification(R"mlir(
module {
  func.func @main(
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>,
      %output: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %resident = wafer.tile.region(
        %input : memref<4xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<4xf16, #wafer.memory<spm, tensor>>) {
    ^bb0(%source: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %buffer = memref.alloc()
          {wafer.spm.offset = #wafer.spm_offset<65536>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.rdma %source to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %buffer
          : memref<4xf16, #wafer.memory<spm, tensor>>
    }
    %done = wafer.tile.region(%resident, %output
        : memref<4xf16, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%source: memref<4xf16, #wafer.memory<spm, tensor>>,
         %destination: memref<4xf16, #wafer.memory<ddr, tensor>>):
      wafer.instr.wdma %source to %destination
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           dst_strides = array<i64: 0, 0, 0>,
           dst_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<spm, tensor>>
         to memref<4xf16, #wafer.memory<ddr, tensor>>
      wafer.instr.ncc_join [0]
      wafer.tile.yield %destination
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::failed(mlir::verify(*module)));
}

TEST_F(ProgramResourceVerificationTest,
       RejectsSPMHighWaterBeyondEstablishedPerTileCapacity) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<3080184>}
        : memref<8xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %buffer, %zero
        : memref<8xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(config);
  llvm::SmallVector<mlir::ModuleOp, 16> modules =
      completeTileDomain(std::move(module));
  ASSERT_EQ(modules.size(), 16u);
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  EXPECT_TRUE(mlir::failed(verifyResources(modules, *config)));
}

TEST_F(ProgramResourceVerificationTest,
       RejectsMissingSPMHighWaterInsteadOfBypassingCapacity) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %buffer, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(config);
  llvm::SmallVector<mlir::ModuleOp, 16> modules =
      completeTileDomain(std::move(module));
  ASSERT_EQ(modules.size(), 16u);
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  EXPECT_TRUE(mlir::failed(verifyResources(modules, *config)));
}

TEST_F(ProgramResourceVerificationTest,
       OmitsUnavailableDynamicTrafficFromNumericCostWithoutGuessing) {
  auto module = parse(R"mlir(
module {
  func.func @main(
      %bounds: memref<1xi64, #wafer.memory<ddr, tensor>>,
      %input: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %raw_n = memref.load %bounds[%c0]
        : memref<1xi64, #wafer.memory<ddr, tensor>>
    %n = arith.index_cast %raw_n : i64 to index
    scf.for %i = %c0 to %n step %c1 {
      wafer.instr.rdma %input to %buffer
          {byte_count = 8 : i64, inner_bytes = 8 : i64,
           src_strides = array<i64: 0, 0, 0>,
           src_iterations = array<i64: 1, 1, 1>}
          : memref<4xf16, #wafer.memory<ddr, tensor>>
         to memref<4xf16, #wafer.memory<spm, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(config);
  llvm::SmallVector<mlir::ModuleOp, 16> modules =
      completeTileDomain(std::move(module));
  ASSERT_EQ(modules.size(), 16u);
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  auto resources = verifyResources(modules, *config);
  ASSERT_TRUE(mlir::succeeded(resources));
  EXPECT_FALSE(resources->aggregateDDRReadBytes.isKnown());
}

TEST_F(ProgramResourceVerificationTest, AccountsDirectCallClosureFromEntry) {
  auto module = parse(R"mlir(
module {
  func.func @main() {
    func.call @helper() : () -> ()
    return
  }
  func.func private @helper() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %buffer, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(config);
  llvm::SmallVector<mlir::ModuleOp, 16> modules =
      completeTileDomain(std::move(module));
  ASSERT_EQ(modules.size(), 16u);

  auto resources = verifyResources(modules, *config);
  ASSERT_TRUE(mlir::succeeded(resources));
  EXPECT_EQ(resources->aggregateInstructionCount.value, 1u);
  EXPECT_EQ(resources->maximumTileSPMHighWaterBytes.value, 264u);
}

TEST_F(ProgramResourceVerificationTest,
       AggregatesTheCompleteSixteenTileDomain) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.0 : f16
    wafer.instr.fill %buffer, %zero
        : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    return
  }
}
)mlir";
  auto cardConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(cardConfig));

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::ModuleOp, 16> modules;
  for (int tile = 0; tile < 16; ++tile) {
    owners.push_back(parse(source));
    ASSERT_TRUE(owners.back());
    modules.push_back(*owners.back());
  }

  auto resources = verifyResources(modules, *cardConfig);
  ASSERT_TRUE(mlir::succeeded(resources));
  EXPECT_EQ(resources->tileCosts.size(), 16u);
  EXPECT_EQ(resources->aggregateInstructionCount.value, 16u);
  EXPECT_EQ(resources->maximumTileSPMHighWaterBytes.value, 8u);
  EXPECT_EQ(resources->summedTileSPMHighWaterBytes.value, 128u);
}

TEST_F(ProgramResourceVerificationTest, ModelsRouteWorkFromTargetTopology) {
  constexpr llvm::StringLiteral sendSource = R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_send %buffer
        {peer = 1 : i64, bytes = 4 : i64,
         message = #wafer.dte_message<communication = 7, round = 0, slice = 0>}
        : memref<4xi8, #wafer.memory<spm, tensor>> -> !async.token
    return
  }
}
)mlir";
  constexpr llvm::StringLiteral receiveSource = R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main() {
    %buffer = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<4xi8, #wafer.memory<spm, tensor>>
    %token = wafer.instr.dte_recv %buffer
        {peer = 0 : i64, bytes = 4 : i64,
         message = #wafer.dte_message<communication = 7, round = 0, slice = 0>}
        : memref<4xi8, #wafer.memory<spm, tensor>> -> !async.token
    return
  }
}
)mlir";
  constexpr llvm::StringLiteral idleSource = R"mlir(
module {
  wafer.target.topology @topology
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  func.func @main() {
    return
  }
}
)mlir";
  auto cardConfig = wafer::compiler::ExecutionConfig::createForSingleCard(
      1, wafer::RuntimeLaunchKind::Kernel);
  ASSERT_TRUE(static_cast<bool>(cardConfig));

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::ModuleOp, 16> modules;
  for (int tile = 0; tile < 16; ++tile) {
    llvm::StringRef source = tile == 0   ? llvm::StringRef(sendSource)
                             : tile == 1 ? llvm::StringRef(receiveSource)
                                         : llvm::StringRef(idleSource);
    owners.push_back(parse(source));
    ASSERT_TRUE(owners.back());
    modules.push_back(*owners.back());
  }
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  auto resources = verifyResources(modules, *cardConfig);
  ASSERT_TRUE(mlir::succeeded(resources));
  ASSERT_TRUE(resources->aggregateNoC.aggregateTransmitBytes.isKnown());
  ASSERT_TRUE(resources->aggregateNoC.aggregateReceiveBytes.isKnown());
  EXPECT_EQ(resources->aggregateNoC.aggregateTransmitBytes.value, 4u);
  EXPECT_EQ(resources->aggregateNoC.aggregateReceiveBytes.value, 4u);
  EXPECT_EQ(resources->minimumHopLinkByteDemand.value, 4u);
  EXPECT_EQ(resources->minimumHopMessageDemand.value, 1u);
  EXPECT_EQ(resources->directedNoCLinkCount.value, 48u);
  EXPECT_EQ(resources->idealizedMinimumPeakLinkByteDemand.value, 1u);
  EXPECT_EQ(resources->modeledNoCRoute.peakDirectedLinkByteDemand.value, 4u);
  EXPECT_EQ(resources->maximumNoCHopCount.value, 1u);
}

TEST_F(ProgramResourceVerificationTest,
       RejectsDuplicateExplicitTileIdentity) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() {
    return
  }
}
)mlir";
  ASSERT_TRUE(config);

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::ModuleOp, 16> modules;
  llvm::SmallVector<wafer::TileId, 16> tileIds;
  for (int64_t tile = 0; tile < config->getTileCount(); ++tile) {
    owners.push_back(parse(source));
    ASSERT_TRUE(owners.back());
    modules.push_back(*owners.back());
    tileIds.emplace_back(tile == 15 ? 14 : tile);
  }
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  EXPECT_TRUE(mlir::failed(wafer::compiler::testing::verifyProgramResources(
      modules, tileIds, *config)));
}

} // namespace
