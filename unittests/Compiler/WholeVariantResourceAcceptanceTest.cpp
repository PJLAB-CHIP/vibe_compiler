//===- WholeVariantResourceAcceptanceTest.cpp ---------------------------===//

#include "Wafer/Compiler/Testing.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitAll.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

#include <memory>

namespace {

class WholeVariantResourceAcceptanceTest : public ::testing::Test {
protected:
  WholeVariantResourceAcceptanceTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    wafer::registerAllDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
    auto created = wafer::compiler::ExecutionConfig::createForSingleCard(
        1, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
        wafer::TargetLaunchABIId::perRankPointerBlockV1());
    EXPECT_TRUE(static_cast<bool>(created));
    if (created)
      config = std::make_unique<wafer::compiler::ExecutionConfig>(*created);
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<wafer::compiler::ExecutionConfig> config;
};

TEST_F(WholeVariantResourceAcceptanceTest,
       AcceptsExactResourceSummaryForCompleteRankDomain) {
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
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};

  auto accepted =
      wafer::compiler::testing::acceptWholeVariantResources(modules, *config);
  ASSERT_TRUE(mlir::succeeded(accepted));
  EXPECT_EQ(accepted->rankCosts.size(), 1u);
  EXPECT_EQ(accepted->aggregateDDRReadBytes.value, 8u);
  EXPECT_EQ(accepted->maximumRankSPMHighWaterBytes.value, 8u);
}

TEST_F(WholeVariantResourceAcceptanceTest,
       ResolvesExplicitSPMHandoffToTheProducerAllocation) {
  auto module = parse(R"mlir(
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
      wafer.instr.local_fence
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
      wafer.instr.local_fence
      wafer.tile.yield %destination
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(config);
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};

  auto accepted =
      wafer::compiler::testing::acceptWholeVariantResources(modules, *config);
  ASSERT_TRUE(mlir::succeeded(accepted));
  EXPECT_EQ(accepted->aggregateDDRReadBytes.value, 8u);
  EXPECT_EQ(accepted->aggregateDDRWriteBytes.value, 8u);
  EXPECT_EQ(accepted->maximumRankSPMHighWaterBytes.value, 8u);
}

TEST_F(WholeVariantResourceAcceptanceTest,
       RejectsSPMHighWaterBeyondEstablishedPerRankCapacity) {
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
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  EXPECT_TRUE(mlir::failed(
      wafer::compiler::testing::acceptWholeVariantResources(modules, *config)));
}

TEST_F(WholeVariantResourceAcceptanceTest,
       RejectsUnknownWholeVariantTrafficInsteadOfGuessing) {
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
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });

  EXPECT_TRUE(mlir::failed(
      wafer::compiler::testing::acceptWholeVariantResources(modules, *config)));
}

TEST_F(WholeVariantResourceAcceptanceTest,
       AccountsTheAcceptedDirectCallClosureFromItsEntry) {
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
  llvm::SmallVector<mlir::ModuleOp, 1> modules{*module};

  auto accepted =
      wafer::compiler::testing::acceptWholeVariantResources(modules, *config);
  ASSERT_TRUE(mlir::succeeded(accepted));
  EXPECT_EQ(accepted->aggregateInstructionCount.value, 1u);
  EXPECT_EQ(accepted->maximumRankSPMHighWaterBytes.value, 264u);
}

TEST_F(WholeVariantResourceAcceptanceTest,
       AggregatesTheCompleteSixteenRankCardDomain) {
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
      16, wafer::TargetProfileId::waferTx81SingleCardKernelV1(),
      wafer::TargetLaunchABIId::perRankPointerBlockV1());
  ASSERT_TRUE(static_cast<bool>(cardConfig));

  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 16> owners;
  llvm::SmallVector<mlir::ModuleOp, 16> modules;
  for (int rank = 0; rank < 16; ++rank) {
    owners.push_back(parse(source));
    ASSERT_TRUE(owners.back());
    modules.push_back(*owners.back());
  }

  auto accepted = wafer::compiler::testing::acceptWholeVariantResources(
      modules, *cardConfig);
  ASSERT_TRUE(mlir::succeeded(accepted));
  EXPECT_EQ(accepted->rankCosts.size(), 16u);
  EXPECT_EQ(accepted->aggregateInstructionCount.value, 16u);
  EXPECT_EQ(accepted->maximumRankSPMHighWaterBytes.value, 8u);
  EXPECT_EQ(accepted->summedRankSPMHighWaterBytes.value, 128u);
}

} // namespace
