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

} // namespace
