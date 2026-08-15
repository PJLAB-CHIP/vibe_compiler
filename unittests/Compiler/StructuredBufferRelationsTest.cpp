//===- StructuredBufferRelationsTest.cpp -------------------------------===//

#include "../../lib/Wafer/Compiler/StructuredBufferRelations.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Pipelines/Pipelines.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

class StructuredBufferRelationsTest : public ::testing::Test {
protected:
  StructuredBufferRelationsTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> createModule() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main() {
    %token = arith.constant false
    %unused = wafer.tile.region(%token : i1) -> (i1) {
    ^bb0(%tile_token: i1):
      %source = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      %copy = wafer.tile.copy %source
          : memref<4xf16, #wafer.memory<spm, tensor>>
         -> memref<4xf16, #wafer.memory<spm, tensor>>
      wafer.tile.yield %tile_token : i1
    }
    return
  }
}
)mlir",
                                                   context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredBufferRelationsTest,
       ConversionListenerRetargetsCurrentIRBuffer) {
  mlir::OwningOpRef<mlir::ModuleOp> module = createModule();
  ASSERT_TRUE(module);
  wafer::MoveCopyOp copy;
  module->walk([&](wafer::MoveCopyOp operation) { copy = operation; });
  ASSERT_TRUE(copy);

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, copy.getResult()});
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));

  wafer::compiler::detail::StructuredBufferReplacementListener listener(
      relations);
  wafer::TileRegionToInstrLoweringSession loweringSession(*context);
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  wafer::TileRegionOp region = *function.getOps<wafer::TileRegionOp>().begin();
  ASSERT_TRUE(mlir::succeeded(
      wafer::convertTileRegionToInstr(region, loweringSession, &listener)));
  EXPECT_TRUE(listener.preservedAllRelations());
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
  EXPECT_FALSE(mlir::isa_and_nonnull<wafer::MoveCopyOp>(
      relations.operationResultBuffers.front().buffer.getDefiningOp()));

  ASSERT_TRUE(mlir::succeeded(wafer::compiler::detail::runPassPipeline(
      *module, "test-required-ncc-join-placement",
      [](mlir::OpPassManager &manager) {
        wafer::addRequiredNCCJoinPlacementPass(
            manager.nest<mlir::func::FuncOp>());
      })));
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
}

TEST_F(StructuredBufferRelationsTest,
       CurrentIRCheckRejectsUntrackedReplacementWithoutDereference) {
  mlir::OwningOpRef<mlir::ModuleOp> module = createModule();
  ASSERT_TRUE(module);
  wafer::MoveCopyOp copy;
  module->walk([&](wafer::MoveCopyOp operation) { copy = operation; });
  ASSERT_TRUE(copy);

  wafer::StructuredMaterializationRelations relations;
  relations.operationResultBuffers.push_back(
      {/*structuredNodeId=*/7, copy.getResult()});
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  wafer::TileRegionOp region = *function.getOps<wafer::TileRegionOp>().begin();
  ASSERT_TRUE(mlir::succeeded(wafer::convertTileRegionToInstr(region)));

  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));

  wafer::compiler::detail::retainCurrentStructuredBufferRelations(
      module->getOperation(), relations);
  EXPECT_TRUE(relations.operationResultBuffers.empty());
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
}

} // namespace
