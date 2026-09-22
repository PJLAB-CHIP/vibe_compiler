//===- RotatingBuffersTest.cpp ----------------------------------------===//

#include "Wafer/Transforms/Tile/RotatingBuffers.h"
#include "Wafer/Transforms/Tile/StorageInitialization.h"

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Instr/MemoryPlanning.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <functional>
#include <memory>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::shared_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  auto context = std::make_shared<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

TEST(RotatingBuffersTest, CurrentRotatingAllocationFeedsActualMiniMalloc) {
  auto context = createContext();
  mlir::Location loc = mlir::UnknownLoc::get(context.get());
  auto module = mlir::ModuleOp::create(loc);
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto function = moduleBuilder.create<mlir::func::FuncOp>(
      loc, "main",
      moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
  mlir::Block *entry = function.addEntryBlock();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
  auto region =
      builder.create<TileRegionOp>(loc, mlir::TypeRange{}, mlir::ValueRange{});
  region.getBody().push_back(new mlir::Block());
  mlir::OpBuilder regionBuilder =
      mlir::OpBuilder::atBlockBegin(&region.getBody().front());
  auto type = mlir::MemRefType::get(
      {2, 1031, 128}, regionBuilder.getF16Type(),
      mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor));
  auto allocation = regionBuilder.create<mlir::memref::AllocOp>(loc, type);
  auto lower = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto upper = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 1024);
  auto step = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 128);
  auto zero = regionBuilder.create<mlir::arith::ConstantOp>(
      loc, regionBuilder.getFloatAttr(regionBuilder.getF16Type(), 0.0));
  auto loop = regionBuilder.create<mlir::scf::ForOp>(loc, lower, upper, step);
  mlir::OpBuilder loopBuilder = mlir::OpBuilder::atBlockBegin(loop.getBody());
  auto fill = loopBuilder.create<InstrFillOp>(
      loc, allocation.getResult(), zero, FillDomainAttr(), NCCWorker::Worker0);
  regionBuilder.setInsertionPointAfter(loop);
  regionBuilder.create<mlir::memref::DeallocOp>(loc, allocation);
  regionBuilder.create<TileYieldOp>(loc);
  builder.setInsertionPointAfter(region);
  builder.create<mlir::func::ReturnOp>(loc);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

  StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {fill, allocation, MaterializedBufferRole::Scratch});
  mlir::OwningOpRef<mlir::ModuleOp> owned(module);
  auto rotated = materializeRotatingAllocations(
      std::move(owned), {{allocation, loop, /*multiplicity=*/2}}, relations);
  ASSERT_TRUE(rotated.succeeded())
      << (rotated.failure ? rotated.failure->detail : "");
  ASSERT_EQ(rotated.materialized->slots.size(), 2u);
  EXPECT_EQ(relations.buffers.size(), 2u);
  ASSERT_TRUE(mlir::succeeded(
      wafer::rebuildRequiredNCCJoins(*rotated.materialized->module)));
  ASSERT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
      *rotated.materialized->module, /*spmBase=*/0,
      /*spmLimit=*/3 * 1024 * 1024, /*spmAlignment=*/16)));
  unsigned placed = 0;
  rotated.materialized->module->walk([&](mlir::memref::AllocOp current) {
    if (isWaferSPMMemRefType(current.getType())) {
      ++placed;
      EXPECT_TRUE(current->hasAttr(kWaferSPMOffsetAttrName));
    }
  });
  EXPECT_EQ(placed, 2u);
}

} // namespace
