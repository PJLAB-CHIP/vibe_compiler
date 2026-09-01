//===- StructuredBufferRelationsTest.cpp -------------------------------===//

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

class StructuredBufferRelationsTest : public ::testing::Test {
protected:
  StructuredBufferRelationsTest() {
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        text, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredBufferRelationsTest,
       ListenerRetargetsActualValueWithoutSourceNodeParity) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %source = memref.alloc()
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      %first = wafer.tile.materialize_layout %source
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
         -> memref<2x1024x64xf16, #wafer.memory<spm, ncx>>
      %second = wafer.tile.materialize_layout %first
          : memref<2x1024x64xf16, #wafer.memory<spm, ncx>>
         -> memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      %c0 = arith.constant 0 : index
      %value = memref.load %second[%c0, %c0, %c0]
          : memref<2x1024x64xf16, #wafer.memory<spm, tensor>>
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  llvm::SmallVector<LayoutMaterializeOp, 2> layouts;
  module->walk([&](LayoutMaterializeOp op) { layouts.push_back(op); });
  ASSERT_EQ(layouts.size(), 2u);
  mlir::memref::LoadOp load;
  module->walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(load);

  StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {load, layouts.front().getResult(), MaterializedBufferRole::Operand});
  StructuredBufferReplacementListener listener(relations);
  listener.notifyOperationReplaced(layouts.front(), layouts.back().getResult());
  EXPECT_TRUE(listener.finalizeAfterRewrite());
  ASSERT_EQ(relations.buffers.size(), 1u);
  EXPECT_EQ(relations.buffers.front().buffer, layouts.back().getResult());
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
}

TEST_F(StructuredBufferRelationsTest,
       LoweringRecorderTransfersScratchToLiveActualOperation) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %zero = arith.constant 0.000000e+00 : f16
      %buffer = memref.alloc()
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      wafer.tile.fill %buffer, %zero
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>, f16
      wafer.instr.fill %buffer, %zero
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>, f16
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  ComputeFillOp source;
  InstrFillOp lowered;
  mlir::memref::AllocOp allocation;
  module->walk([&](ComputeFillOp op) { source = op; });
  module->walk([&](InstrFillOp op) { lowered = op; });
  module->walk([&](mlir::memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(source && lowered && allocation);

  StructuredMaterializationRelations relations;
  StructuredBufferReplacementListener listener(relations);
  listener.recordScratchAllocation(source, allocation);
  listener.recordLoweredOperation(source, lowered);
  EXPECT_TRUE(listener.finalizeAfterRewrite())
      << listener.getFailureReason().str();
  EXPECT_TRUE(llvm::any_of(relations.buffers, [&](const auto &relation) {
    return relation.owner == lowered && relation.buffer == allocation &&
           relation.role == MaterializedBufferRole::Scratch;
  }));
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
}

TEST_F(StructuredBufferRelationsTest,
       CurrentViewRelationRemainsLiveAndFailsOnStaleOwner) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %buffer = memref.alloc()
          : memref<2x1031x64xf16, #wafer.memory<spm, tensor>>
      %view = memref.subview %buffer[0, 0, 0] [1, 1031, 64] [1, 1, 1]
          : memref<2x1031x64xf16, #wafer.memory<spm, tensor>>
            to memref<1x1031x64xf16, strided<[65984, 64, 1]>,
                      #wafer.memory<spm, tensor>>
      %c0 = arith.constant 0 : index
      %value = memref.load %view[%c0, %c0, %c0]
          : memref<1x1031x64xf16, strided<[65984, 64, 1]>,
                   #wafer.memory<spm, tensor>>
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::memref::SubViewOp view;
  mlir::memref::LoadOp load;
  module->walk([&](mlir::memref::SubViewOp op) { view = op; });
  module->walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(view && load);

  StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {load, view.getResult(), MaterializedBufferRole::Operand});
  EXPECT_EQ(relations.buffers.front().buffer, view.getResult());
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));

  load->erase();
  EXPECT_TRUE(
      mlir::failed(checkStructuredBufferRelationsCurrent(*module, relations)));
  retainCurrentStructuredBufferRelations(*module, relations);
  EXPECT_TRUE(relations.buffers.empty());
}

TEST_F(StructuredBufferRelationsTest,
       ConditionalBufferRelationRetainsAllCurrentStorageRoots) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%condition: i1) {
      %selected = scf.if %condition
          -> (memref<2x1025x64xf16, #wafer.memory<spm, tensor>>) {
        %then = memref.alloc()
            : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
        scf.yield %then
            : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      } else {
        %else = memref.alloc()
            : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
        scf.yield %else
            : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      }
      %c0 = arith.constant 0 : index
      %value = memref.load %selected[%c0, %c0, %c0]
          : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  mlir::scf::IfOp branch;
  mlir::memref::LoadOp load;
  module->walk([&](mlir::scf::IfOp op) { branch = op; });
  module->walk([&](mlir::memref::LoadOp op) { load = op; });
  ASSERT_TRUE(branch && load);

  StructuredMaterializationRelations relations;
  relations.buffers.push_back(
      {load, branch.getResult(0), MaterializedBufferRole::Operand});
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));

  StorageRootMemo roots;
  EXPECT_EQ(roots.getStorageRoots(branch.getResult(0)).size(), 2u);
  EXPECT_EQ(relations.buffers.front().buffer, branch.getResult(0));
}

TEST_F(StructuredBufferRelationsTest,
       CrossTileAndOutputEndpointsMustRemainLiveAndUnique) {
  auto module = parse(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1024x64xf16>) {
      %source = wafer.tile.region(
          %input : tensor<2x1024x64xf16>)
          -> (tensor<2x1024x64xf16>) {
      ^bb0(%arg0: tensor<2x1024x64xf16>):
        wafer.tile.yield %arg0 : tensor<2x1024x64xf16>
      }
      return
    }
  }
  wafer.tile.module card_id = 0 tile_id = 1 {
    func.func @entry(%input: tensor<2x1024x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x1024x64xf16>)
          -> (tensor<2x1024x64xf16>) {
      ^bb0(%arg0: tensor<2x1024x64xf16>):
        wafer.tile.yield %arg0 : tensor<2x1024x64xf16>
      }
      return
    }
  }
}
)mlir");
  ASSERT_TRUE(module);
  llvm::SmallVector<TileRegionOp, 2> regions;
  module->walk([&](TileRegionOp op) { regions.push_back(op); });
  ASSERT_EQ(regions.size(), 2u);
  StructuredMaterializationRelations relations;
  relations.boundaryRelations.push_back(
      {regions[0].getResult(0), regions[1].getBody().getArgument(0)});
  relations.structuralOutputs.push_back({0, regions[1].getResult(0)});
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
  relations.boundaryRelations.push_back(relations.boundaryRelations.front());
  EXPECT_TRUE(
      mlir::failed(checkStructuredBufferRelationsCurrent(*module, relations)));
}

} // namespace
