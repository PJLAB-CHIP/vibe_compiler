//===- MovementFusionTest.cpp ----------------------------------------===//

#include "Wafer/Transforms/Tile/MovementFusion.h"
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

TEST(MovementFusionTest, TransposeFusesPrivateLayoutConsumer) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (bool observer : {false, true}) {
        auto context = createContext();
        auto source =
            llvm::formatv("memref<2x128x{0}x{1}, #wafer.memory<spm, cx>>",
                          extent, dtype)
                .str();
        auto intermediate =
            llvm::formatv("memref<2x{0}x128x{1}, #wafer.memory<spm, cx>>",
                          extent, dtype)
                .str();
        auto output =
            llvm::formatv("memref<2x{0}x128x{1}, #wafer.memory<spm, tensor>>",
                          extent, dtype)
                .str();
        auto ddr =
            llvm::formatv("memref<2x{0}x128x{1}, #wafer.memory<ddr, tensor>>",
                          extent, dtype)
                .str();
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { func.func @main(%out: " << ddr << ") {\n"
            << "wafer.tile.region(%out : " << ddr
            << ") -> () {\n^bb0(%ddr: " << ddr << "):\n"
            << "%source = memref.alloc() : " << source << "\n"
            << "%transpose = wafer.tile.transpose %source {permutation = "
               "array<i64: 0, 2, 1>} : "
            << source << " -> " << intermediate << "\n"
            << "%layout = wafer.tile.materialize_layout %transpose : "
            << intermediate << " -> " << output << "\n";
        if (observer)
          out << "wafer.tile.store %transpose, %ddr : " << intermediate
              << " -> " << ddr << "\n";
        out << "wafer.tile.store %layout, %ddr : " << output << " -> " << ddr
            << "\nwafer.tile.yield\n}\nreturn\n}}\n";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(module) << text;
        mlir::IRRewriter rewriter(context.get());
        fuseTransposeLayoutMovements(*module, rewriter);
        unsigned layouts = 0, transposes = 0;
        module->walk([&](LayoutMaterializeOp) { ++layouts; });
        module->walk([&](MoveTransposeOp op) {
          ++transposes;
          EXPECT_EQ(getWaferMemoryAttr(
                        mlir::cast<mlir::MemRefType>(op.getResult().getType()))
                        .getLayout(),
                    observer ? MemLayout::Cx : MemLayout::Tensor);
          EXPECT_EQ(op.getPermutation(), llvm::ArrayRef<int64_t>({0, 2, 1}));
        });
        EXPECT_EQ(transposes, 1u);
        EXPECT_EQ(layouts, observer ? 1u : 0u);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      }
}

TEST(MovementFusionTest, DMALayoutFusionPreservesObserversAndSnapshots) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned mode : {0u, 1u, 2u, 3u}) {
      auto context = createContext();
      auto type = [&](llvm::StringRef space, llvm::StringRef layout) {
        return llvm::formatv("memref<1x{0}x192xbf16, #wafer.memory<{1}, {2}>>",
                             extent, space, layout)
            .str();
      };
      auto ddr = type("ddr", "ncx");
      auto tensor = type("spm", "tensor");
      auto ncx = type("spm", "ncx");
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "module { func.func @entry(%a: " << ddr << ", %b: " << ddr
          << ") { wafer.tile.region(%a, %b : " << ddr << ", " << ddr
          << ") -> () { ^bb0(%src: " << ddr << ", %dst: " << ddr << "): "
          << "%buffer = memref.alloc() : " << tensor
          << " wafer.tile.load %src into %buffer : " << ddr << " into "
          << tensor;
      if (mode == 1)
        out << " wafer.tile.store %buffer, %dst : " << tensor << " -> " << ddr;
      out << " %compute = wafer.tile.materialize_layout %buffer : " << tensor
          << " -> " << ncx
          << " %publication = wafer.tile.materialize_layout %compute : " << ncx
          << " -> " << tensor;
      if (mode == 2)
        out << " %zero = arith.constant 0.0 : bf16"
            << " wafer.tile.fill %compute, %zero {fill_domain = "
               "#wafer.fill_domain<physical_footprint>} : "
            << ncx << ", bf16";
      out << " wafer.tile.store %publication, %dst : " << tensor << " -> "
          << ddr;
      if (mode == 3)
        out << " wafer.tile.store %publication, %dst : " << tensor << " -> "
            << ddr;
      out << " wafer.tile.yield } return } }";
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
      ASSERT_TRUE(module) << text;
      mlir::IRRewriter rewriter(context.get());
      fuseDMALayoutMovements(*module, rewriter);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      unsigned layouts = 0;
      module->walk([&](LayoutMaterializeOp) { ++layouts; });
      EXPECT_EQ(layouts, mode == 0 ? 0u : 1u);
      module->walk([&](StorageLoadOp load) {
        EXPECT_EQ(getWaferMemoryAttr(
                      mlir::cast<mlir::MemRefType>(load.getDest().getType()))
                      .getLayout(),
                  mode == 1 ? MemLayout::Tensor : MemLayout::NCx);
      });
    }
}

} // namespace
