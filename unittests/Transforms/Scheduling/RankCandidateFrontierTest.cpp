//===- RankCandidateFrontierTest.cpp - Actual-clone frontier tests ------===//

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/WaferTensorProgramToTileRegion.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/PhysicalDataflow.h"
#include "Wafer/Transforms/TensorProgramScheduling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {

TEST(RankCandidateFrontierTest,
     PreservesReservedSpillAndResidentAsIndependentPlacedClones) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  wafer::WaferDialect>();
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  wafer::registerTargetImplementationExternalModels(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
module {
  func.func @main(%lhs: tensor<256x128xf16>,
                  %rhs: tensor<128x1024xf16>) -> tensor<256x1024xf16> {
    %zero = arith.constant 0.0 : f16
    %matmul_empty = tensor.empty() : tensor<256x1024xf16>
    %matmul_init = linalg.fill ins(%zero : f16)
        outs(%matmul_empty : tensor<256x1024xf16>) -> tensor<256x1024xf16>
    %matmul = linalg.matmul
        ins(%lhs, %rhs : tensor<256x128xf16>, tensor<128x1024xf16>)
        outs(%matmul_init : tensor<256x1024xf16>) -> tensor<256x1024xf16>
    %collective_out = tensor.empty() : tensor<256x1024xf16>
    %collective = wafer.linalg_ext.collective.all_reduce
        ins(%matmul : tensor<256x1024xf16>)
        outs(%collective_out : tensor<256x1024xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16):
      %sum = arith.addf %lhs_value, %rhs_value : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 43 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<256x1024xf16>
    return %collective : tensor<256x1024xf16>
  }
}
)mlir",
      &context);
  ASSERT_TRUE(source);

  wafer::TensorProgramSchedulingConfig config;
  config.logicalRank = 0;
  config.candidateParallelism = 1;
  auto frontier = wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GE(frontier->size(), 2u);

  unsigned baselineCount = 0;
  bool sawResidentAlternative = false;
  for (wafer::ScheduledRankCandidate &candidate : *frontier) {
    baselineCount += candidate.reservedBaseline;
    bool candidateResident = false;
    candidate.module->walk([&](wafer::TileRegionOp region) {
      candidateResident |=
          llvm::any_of(region.getResultTypes(), [](mlir::Type type) {
            return wafer::isWaferSPMMemRefType(type);
          });
    });
    sawResidentAlternative |= !candidate.reservedBaseline && candidateResident;

    candidate.module->walk([&](mlir::memref::AllocOp allocation) {
      if (wafer::isWaferSPMMemRefType(allocation.getType()))
        EXPECT_TRUE(allocation->hasAttr(wafer::kWaferSPMOffsetAttrName));
      if (wafer::isWaferDDRMemRefType(allocation.getType()))
        EXPECT_FALSE(allocation->hasAttr(wafer::kWaferDDROffsetAttrName));
    });
    candidate.module->walk([](wafer::InstrDTESendOp send) {
      EXPECT_FALSE(send.getBinding().has_value());
    });
    candidate.module->walk([](wafer::InstrDTERecvOp recv) {
      EXPECT_FALSE(recv.getBinding().has_value());
    });
  }
  EXPECT_EQ(baselineCount, 1u);
  EXPECT_TRUE(sawResidentAlternative);

  config.candidateParallelism = 4;
  auto parallelFrontier =
      wafer::buildScheduledRankCandidateFrontier(*source, config);
  ASSERT_TRUE(mlir::succeeded(parallelFrontier));
  ASSERT_EQ(parallelFrontier->size(), frontier->size());
  for (auto &&[serial, parallel] :
       llvm::zip_equal(*frontier, *parallelFrontier)) {
    EXPECT_EQ(serial.estimatedTimePs, parallel.estimatedTimePs);
    EXPECT_EQ(serial.discoveryOrder, parallel.discoveryOrder);
    EXPECT_EQ(serial.reservedBaseline, parallel.reservedBaseline);
    std::string serialText;
    llvm::raw_string_ostream serialStream(serialText);
    serial.module->print(serialStream);
    std::string parallelText;
    llvm::raw_string_ostream parallelStream(parallelText);
    parallel.module->print(parallelStream);
    EXPECT_EQ(serialStream.str(), parallelStream.str());
  }

  // Candidate generation and evaluation never mutate the shared source.
  bool sourceContainsTileRegion = false;
  source->walk([&](wafer::TileRegionOp) { sourceContainsTileRegion = true; });
  EXPECT_FALSE(sourceContainsTileRegion);
}

} // namespace
