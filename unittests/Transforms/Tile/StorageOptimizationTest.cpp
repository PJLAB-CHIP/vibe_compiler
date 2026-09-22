//===- StorageOptimizationTest.cpp ----------------------------------------===//

#include "Wafer/Transforms/Tile/StorageOptimization.h"
#include "Wafer/Transforms/Tile/LoopPipelining.h"
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

unsigned countLoops(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](mlir::scf::ForOp) { ++count; });
  return count;
}

mlir::LogicalResult optimizeStorageForTest(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  preservePrivateScalarBroadcasts(module, rewriter);
  return optimizeStorage(module, rewriter);
}

TEST(StorageOptimizationTest, PrivateScalarBroadcastAvoidsCPUReadback) {
  for (auto dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (unsigned variant = 0; variant < 4; ++variant) {
        SCOPED_TRACE(::testing::Message()
                     << dtype << ':' << extent << ':' << variant);
        auto context = createContext();
        const std::string scalar =
            "memref<" + std::string(dtype) + ", #wafer.memory<spm, tensor>>";
        const std::string source =
            "memref<" + std::string(dtype) + ", #wafer.memory<ddr, tensor>>";
        const std::string tensor = "memref<2x" + std::to_string(extent) +
                                   "x32x" + dtype +
                                   ", #wafer.memory<spm, tensor>>";
        std::string text;
        llvm::raw_string_ostream out(text);
        // Rank zero is the actual scalar source; its consumers exercise a
        // real tensor and multiple dynamic iterations.
        out << "module { func.func @main(%source: " << source
            << ") { wafer.tile.region(%source : " << source
            << ") -> () { ^bb0(%in: " << source << "):"
            << "%s = memref.alloc() : " << scalar
            << "\nwafer.tile.load %in into %s : " << source << " into "
            << scalar << "\n%value = memref.load %s[] : " << scalar
            << "\n%v = memref.alloc() : " << tensor
            << "\n%c0 = arith.constant 0 : index\n"
               "%c1 = arith.constant 1 : index\n"
               "%c4 = arith.constant 4 : index\n";
        if (variant == 1)
          out << "wafer.tile.load %in into %s : " << source << " into "
              << scalar << '\n';
        if (variant == 2)
          out << "%alias = memref.cast %s : " << scalar << " to " << scalar
              << "\nwafer.tile.fill %alias, %value : " << scalar << ", "
              << dtype << '\n';
        if (variant == 3)
          out << "wafer.tile.fill %v, %value : " << tensor << ", " << dtype
              << '\n';
        out << "scf.for %iv = %c0 to %c4 step %c1 {\n"
               "wafer.tile.elementwise_into <mul> %v, %value into %v : "
            << tensor << ", " << dtype << " into " << tensor
            << "\n}\nwafer.tile.yield } return } }";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(module) << text;
        ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
        auto &result = module;
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*result)));
        unsigned reads = 0;
        result->walk([&](mlir::memref::LoadOp) { ++reads; });
        EXPECT_EQ(reads, variant ? 1u : 0u);
        if (variant)
          continue;
        TileRegionOp region;
        result->walk([&](TileRegionOp op) { region = op; });
        TileRegionToInstrLoweringSession session(*context);
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
        unsigned units = 0;
        region.walk([&](InstrElementwiseOp op) {
          ++units;
          EXPECT_EQ(op.getRhsUnitElements(), 1);
          EXPECT_EQ(mlir::cast<mlir::MemRefType>(op.getInputs()[1].getType())
                        .getRank(),
                    0);
        });
        EXPECT_EQ(units, 1u);
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*result)));
        unsigned joins = 0;
        result->walk([&](SyncNCCJoinOp op) {
          ++joins;
          EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
        });
        EXPECT_EQ(joins, 1u);
        EXPECT_TRUE(mlir::succeeded(
            planSPMMemoryModule(*result, 0, 3 * 1024 * 1024, 16)));
      }
}

TEST(StorageOptimizationTest,
     GemmWritebackKeepsExplicitDestinationAndSeparatePsum) {
  for (auto dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031})
      for (unsigned variant = 0; variant < 6; ++variant) {
        SCOPED_TRACE(::testing::Message()
                     << dtype << "/" << extent << "/" << variant);
        auto context = createContext();
        std::string lhs = "memref<2x" + std::to_string(extent) + "x16x" +
                          dtype + ", #wafer.memory<spm, ncx>>";
        std::string rhs = "memref<2x16x16x" + std::string(dtype) +
                          ", #wafer.memory<spm, ncx>>";
        std::string dest = "memref<2x" + std::to_string(extent) +
                           "x16xf32, #wafer.memory<spm, ncx>>";
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { func.func @main() { wafer.tile.region() -> () {\n"
               "%a = memref.alloc() : "
            << lhs << "\n%b = memref.alloc() : " << rhs
            << "\n%d = memref.alloc() : " << dest
            << "\n%p = memref.alloc() : " << dest
            << "\n%alias = memref.cast %d : " << dest << " to " << dest
            << "\n%r = wafer.tile.gemm %a, %b";
        if (variant == 1 || variant == 2 || variant == 5)
          out << " psum(%"
              << (variant == 1   ? "p"
                  : variant == 2 ? "d"
                                 : "alias")
              << " : " << dest << ")";
        out << " {batch_count = 2 : i64, lhs_batch_dims = array<i64: 0>, "
               "lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64, "
               "rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, "
               "rhs_n_dim = 2 : i64, result_batch_dims = array<i64: 0>, "
               "result_m_dim = 1 : i64, result_n_dim = 2 : i64} : ("
            << lhs << ", " << rhs << ") -> " << dest << "\n";
        if (variant == 4)
          out << "%observer = wafer.tile.copy %d : " << dest << " -> " << dest
              << "\n";
        out << "wafer.tile.copy_into %r into %d : " << dest << " into " << dest
            << "\n";
        if (variant == 3)
          out << "%extra = wafer.tile.copy %r : " << dest << " -> " << dest
              << "\n";
        out << "wafer.tile.yield } return } }";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(module) << text;
        ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
        unsigned into = 0, copies = 0;
        TileRegionOp region;
        mlir::Value target;
        module->walk([&](ComputeGemmIntoOp op) {
          ++into;
          target = op.getDest();
        });
        module->walk([&](MoveCopyIntoOp) { ++copies; });
        module->walk([&](TileRegionOp op) { region = op; });
        EXPECT_EQ(into, variant < 2 ? 1u : 0u);
        EXPECT_EQ(copies, variant < 2 ? 0u : 1u);
        TileRegionToInstrLoweringSession session(*context);
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
        unsigned gemms = 0;
        region.walk([&](InstrGemmOp op) {
          ++gemms;
          if (target) {
            EXPECT_EQ(op.getDest(), target);
          }
          EXPECT_NE(op.getDest(), op.getPsum());
        });
        EXPECT_EQ(gemms, 1u);
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
        unsigned joins = 0;
        module->walk([&](SyncNCCJoinOp) { ++joins; });
        EXPECT_EQ(joins, 1u);
        EXPECT_TRUE(mlir::succeeded(
            planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
      }
}

TEST(StorageOptimizationTest,
     GemmWritebackThroughMetadataReshapesPreservesStorage) {
  enum class Case {
    Single,
    Chain,
    SeparatePsum,
    AliasedPsum,
    ResultShared,
    ViewShared,
    Observer,
    MaterializingReshape,
    DifferentMapping,
    FoldBatchPadding
  };
  for (auto dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031})
      for (auto test : {Case::Single, Case::Chain, Case::SeparatePsum,
                        Case::AliasedPsum, Case::ResultShared, Case::ViewShared,
                        Case::Observer, Case::MaterializingReshape,
                        Case::DifferentMapping, Case::FoldBatchPadding}) {
        SCOPED_TRACE(::testing::Message() << dtype << '/' << extent << '/'
                                          << static_cast<int>(test));
        auto context = createContext();
        std::string lhs = "memref<2x" + std::to_string(extent) + "x16x" +
                          dtype + ", #wafer.memory<spm, ncx>>";
        std::string rhs = "memref<2x16x16x" + std::string(dtype) +
                          ", #wafer.memory<spm, ncx>>";
        std::string result = "memref<2x" + std::to_string(extent) +
                             "x16xf32, #wafer.memory<spm, ncx>>";
        std::string view = "memref<2x1x" + std::to_string(extent) +
                           "x16xf32, #wafer.memory<spm, ncx>>";
        bool chain = test == Case::Chain || test == Case::ViewShared;
        std::string dest = chain ? "memref<2x1x1x" + std::to_string(extent) +
                                       "x16xf32, #wafer.memory<spm, ncx>>"
                                 : view;
        if (test == Case::DifferentMapping)
          dest = view = "memref<2x16x" + std::to_string(extent) +
                        "xf32, #wafer.memory<spm, ncx>>";
        if (test == Case::FoldBatchPadding)
          dest = view = "memref<1x2x" + std::to_string(extent) +
                        "x16xf32, #wafer.memory<spm, ncx>>";
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { func.func @main() { wafer.tile.region() -> () {\n"
               "%a = memref.alloc() : "
            << lhs << "\n%b = memref.alloc() : " << rhs
            << "\n%d = memref.alloc() : " << dest
            << "\n%p = memref.alloc() : " << result << '\n';
        if (test == Case::AliasedPsum)
          out << "%alias = wafer.tile.reshape %d : " << dest << " -> " << result
              << '\n';
        out << "%r = wafer.tile.gemm %a, %b";
        if (test == Case::SeparatePsum || test == Case::AliasedPsum)
          out << " psum(%" << (test == Case::SeparatePsum ? "p" : "alias")
              << " : " << result << ')';
        out << " {batch_count = 2 : i64, lhs_batch_dims = array<i64: 0>, "
               "lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64, "
               "rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, "
               "rhs_n_dim = 2 : i64, result_batch_dims = array<i64: 0>, "
               "result_m_dim = 1 : i64, result_n_dim = 2 : i64} : ("
            << lhs << ", " << rhs << ") -> " << result << '\n';
        out << "%v = wafer.tile."
            << (test == Case::MaterializingReshape ? "reshape_copy" : "reshape")
            << " %r : " << result << " -> " << view << '\n';
        if (chain)
          out << "%w = wafer.tile.reshape %v : " << view << " -> " << dest
              << '\n';
        if (test == Case::Observer)
          out << "%old = wafer.tile.copy %d : " << dest << " -> " << dest
              << '\n';
        out << "wafer.tile.copy_into %" << (chain ? "w" : "v")
            << " into %d : " << dest << " into " << dest << '\n';
        if (test == Case::ResultShared || test == Case::ViewShared)
          out << "%extra = wafer.tile.copy %"
              << (test == Case::ResultShared ? "r" : "v") << " : "
              << (test == Case::ResultShared ? result : view) << " -> "
              << (test == Case::ResultShared ? result : view) << '\n';
        // Observe the original destination, independently of its new view.
        out << "%read = wafer.tile.copy %d : " << dest << " -> " << dest
            << "\nwafer.tile.yield } return } }";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(module) << text;
        mlir::Value originalDest;
        module->walk([&](MoveCopyIntoOp op) { originalDest = op.getDest(); });
        ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
        bool folded = test == Case::Single || test == Case::Chain ||
                      test == Case::SeparatePsum ||
                      (test == Case::FoldBatchPadding && extent == 1024);
        unsigned into = 0, copies = 0, functional = 0;
        TileRegionOp region;
        module->walk([&](ComputeGemmIntoOp op) {
          ++into;
          auto inverse = op.getDest().getDefiningOp<ViewReshapeOp>();
          ASSERT_TRUE(inverse);
          EXPECT_EQ(inverse.getSource(), originalDest);
          EXPECT_EQ(
              mlir::cast<mlir::MemRefType>(op.getDest().getType()).getShape(),
              llvm::ArrayRef<int64_t>({2, extent, 16}));
          EXPECT_EQ(static_cast<bool>(op.getPsum()),
                    test == Case::SeparatePsum);
        });
        module->walk([&](ComputeGemmOp) { ++functional; });
        module->walk([&](MoveCopyIntoOp op) {
          ++copies;
          EXPECT_EQ(op.getDest(), originalDest);
        });
        module->walk([&](TileRegionOp op) { region = op; });
        EXPECT_EQ(into, folded ? 1u : 0u);
        EXPECT_EQ(functional, folded ? 0u : 1u);
        EXPECT_EQ(copies, folded ? 0u : 1u);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        TileRegionToInstrLoweringSession session(*context);
        if (test == Case::DifferentMapping ||
            (test == Case::FoldBatchPadding && extent != 1024)) {
          // Equal element counts do not imply an NCx metadata alias. The
          // existing lowering must reject this verifier-valid unsupported view.
          EXPECT_TRUE(mlir::failed(convertTileRegionToInstr(region, session)));
          continue;
        }
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
        unsigned gemms = 0;
        region.walk([&](InstrGemmOp op) {
          ++gemms;
          if (folded) {
            auto inverse =
                op.getDest().getDefiningOp<mlir::memref::ReinterpretCastOp>();
            ASSERT_TRUE(inverse);
            EXPECT_EQ(inverse.getSource(), originalDest);
          }
          EXPECT_NE(op.getDest(), op.getPsum());
        });
        EXPECT_EQ(gemms, 1u);
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
        unsigned joins = 0;
        module->walk([&](SyncNCCJoinOp op) {
          ++joins;
          EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
        });
        EXPECT_EQ(joins, 1u);
        EXPECT_TRUE(mlir::succeeded(
            planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
      }
}

TEST(StorageOptimizationTest,
     GemmCopyChainForwardsFinalDestinationWithoutOverwritingPsum) {
  enum class Case {
    Pair,
    Chain,
    LoopBranch,
    LateRead,
    AliasRead,
    Escape,
    AliasedDestination,
    Observer,
    LateDestination
  };
  for (auto dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031})
      for (auto test :
           {Case::Pair, Case::Chain, Case::LoopBranch, Case::LateRead,
            Case::AliasRead, Case::Escape, Case::AliasedDestination,
            Case::Observer, Case::LateDestination}) {
        SCOPED_TRACE(::testing::Message() << dtype << '/' << extent << '/'
                                          << static_cast<int>(test));
        auto context = createContext();
        std::string lhs = "memref<2x" + std::to_string(extent) + "x16x" +
                          dtype + ", #wafer.memory<spm, ncx>>";
        std::string rhs = "memref<2x16x16x" + std::string(dtype) +
                          ", #wafer.memory<spm, ncx>>";
        std::string result = "memref<2x" + std::to_string(extent) +
                             "x16xf32, #wafer.memory<spm, ncx>>";
        std::string dest = "memref<2x1x" + std::to_string(extent) +
                           "x16xf32, #wafer.memory<spm, ncx>>";
        bool loop = test == Case::LoopBranch;
        bool folded = test == Case::Pair || test == Case::Chain || loop;
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { ";
        if (test == Case::Escape)
          out << "func.func private @escape(" << dest << ")\n";
        out << "func.func @main(%n: index, %cond: i1) { "
               "wafer.tile.region(%n, %cond : index, i1) -> () {\n"
               "^bb0(%bound: index, %condition: i1):\n"
               "%a = memref.alloc() : "
            << lhs << "\n%b = memref.alloc() : " << rhs << '\n';
        if (test != Case::LateDestination)
          out << "%d = memref.alloc() : " << dest << '\n';
        if (loop)
          out << "%c0 = arith.constant 0 : index\n"
                 "%c256 = arith.constant 256 : index\n"
                 "scf.for %iv = %c0 to %bound step %c256 {\n"
                 "scf.if %condition {\n";
        out << "%p = memref.alloc() : " << dest
            << "\n%one = arith.constant 1.0 : f32\n"
               "wafer.tile.fill %p, %one {fill_domain = "
               "#wafer.fill_domain<physical_footprint>} : "
            << dest << ", f32"
            << "\n%psum = wafer.tile.reshape %p : " << dest << " -> " << result
            << '\n';
        if (test == Case::AliasRead)
          out << "%alias = memref.cast %p : " << dest << " to " << dest << '\n';
        if (test == Case::Chain)
          out << "%middle = memref.alloc() : " << dest << '\n';
        out << "%r = wafer.tile.gemm %a, %b psum(%psum : " << result
            << ") {batch_count = 2 : i64, lhs_batch_dims = array<i64: 0>, "
               "lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64, "
               "rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, "
               "rhs_n_dim = 2 : i64, result_batch_dims = array<i64: 0>, "
               "result_m_dim = 1 : i64, result_n_dim = 2 : i64} : ("
            << lhs << ", " << rhs << ") -> " << result
            << "\n%v = wafer.tile.reshape %r : " << result << " -> " << dest
            << "\nwafer.tile.copy_into %v into %p : " << dest << " into "
            << dest << '\n';
        if (test == Case::Observer)
          out << "%observe = wafer.tile.copy %d : " << dest << " -> " << dest
              << '\n';
        if (test == Case::LateDestination)
          out << "%d = memref.alloc() : " << dest << '\n';
        if (test == Case::Chain)
          out << "wafer.tile.copy_into %p into %middle : " << dest << " into "
              << dest << '\n';
        out << "memref.copy %" << (test == Case::Chain ? "middle" : "p")
            << ", %" << (test == Case::AliasedDestination ? "p" : "d") << " : "
            << dest << " to " << dest << '\n';
        if (test == Case::LateRead || test == Case::AliasRead)
          out << "%extra = wafer.tile.copy %"
              << (test == Case::AliasRead ? "alias" : "p") << " : " << dest
              << " -> " << dest << '\n';
        if (test == Case::Escape)
          out << "func.call @escape(%p) : (" << dest << ") -> ()\n";
        out << "%read = wafer.tile.copy %d : " << dest << " -> " << dest
            << '\n';
        if (loop)
          out << "} }\n";
        out << "wafer.tile.yield } return } }";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(module) << text;
        mlir::Value psum, destination;
        module->walk([&](ComputeGemmOp op) { psum = op.getPsum(); });
        module->walk(
            [&](mlir::memref::CopyOp op) { destination = op.getTarget(); });
        ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
        auto &output = module;
        unsigned into = 0, functional = 0, copies = 0;
        output->walk([&](ComputeGemmIntoOp op) {
          ++into;
          EXPECT_EQ(op.getPsum(), psum);
          auto view = op.getDest().getDefiningOp<ViewReshapeOp>();
          ASSERT_TRUE(view);
          EXPECT_EQ(view.getSource(), destination);
        });
        output->walk([&](ComputeGemmOp op) {
          ++functional;
          EXPECT_EQ(op.getPsum(), psum);
        });
        output->walk([&](mlir::Operation *op) {
          copies += mlir::isa<MoveCopyIntoOp, mlir::memref::CopyOp>(op);
        });
        EXPECT_EQ(into, folded ? 1u : 0u);
        EXPECT_EQ(functional, folded ? 0u : 1u);
        EXPECT_EQ(copies, folded ? 0u : 2u);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*output)));
        // Unknown external effects are a preservation test, not a target call.
        if (test == Case::Escape)
          continue;
        TileRegionOp region;
        output->walk([&](TileRegionOp op) { region = op; });
        TileRegionToInstrLoweringSession session(*context);
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*output, session)));
        unsigned gemms = 0;
        output->walk([&](InstrGemmOp op) {
          ++gemms;
          if (folded) {
            auto view =
                op.getDest().getDefiningOp<mlir::memref::ReinterpretCastOp>();
            ASSERT_TRUE(view);
            EXPECT_EQ(view.getSource(), destination);
          }
          EXPECT_NE(op.getDest(), op.getPsum());
        });
        EXPECT_EQ(gemms, 1u);
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*output)));
        output->walk([&](SyncNCCJoinOp op) {
          EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
        });
        EXPECT_TRUE(mlir::succeeded(
            planSPMMemoryModule(*output, 0, 3 * 1024 * 1024, 16)));
      }
}

TEST(StorageOptimizationTest, EliminateOverwrittenLayoutInitialization) {
  enum class Case {
    Fill,
    Copy,
    EarlyRead,
    Escape,
    Partial,
    Conditional,
    Loop,
    SharedSource,
    DeadOnly
  };
  for (auto layout : {"ncx", "cx", "ntensor"})
    for (auto dtype : {"f16", "bf16", "f32"})
      for (int64_t extent : {1024, 1025, 1031})
        for (auto test : {Case::Fill, Case::Copy, Case::EarlyRead, Case::Escape,
                          Case::Partial, Case::Conditional, Case::Loop,
                          Case::SharedSource, Case::DeadOnly}) {
          SCOPED_TRACE(::testing::Message()
                       << layout << '/' << dtype << '/' << extent << '/'
                       << static_cast<int>(test));
          auto context = createContext();
          std::string shape = "2x" + std::to_string(extent) + "x16x" + dtype;
          std::string source =
              "memref<" + shape + ", #wafer.memory<spm, " + layout + ">>";
          std::string dest =
              "memref<" + shape + ", #wafer.memory<spm, tensor>>";
          std::string text;
          llvm::raw_string_ostream out(text);
          out << "module { ";
          if (test == Case::Escape)
            out << "func.func private @observe(" << dest << ")\n";
          out << "func.func @main() { wafer.tile.region() -> () {\n"
                 "%a = memref.alloc() : "
              << source << "\n%one = arith.constant 1.0 : " << dtype
              << "\n%two = arith.constant 2.0 : " << dtype
              << "\n%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : "
                 "index\n"
                 "%cond = arith.constant false\n"
                 "wafer.tile.fill %a, %one {fill_domain = "
                 "#wafer.fill_domain<physical_footprint>} : "
              << source << ", " << dtype
              << "\n%d = wafer.tile.materialize_layout %a : " << source
              << " -> " << dest << '\n';
          if (test == Case::EarlyRead)
            out << "%old = wafer.tile.copy %d : " << dest << " -> " << dest
                << '\n';
          if (test == Case::Escape)
            out << "func.call @observe(%d) : (" << dest << ") -> ()\n";
          if (test == Case::Copy) {
            out << "%new = memref.alloc() : " << dest
                << "\nwafer.tile.fill %new, %two : " << dest << ", " << dtype
                << "\nwafer.tile.copy_into %new into %d : " << dest << " into "
                << dest << '\n';
          } else if (test == Case::Partial) {
            std::string sub = "memref<2x" + std::to_string(extent - 1) +
                              "x16x" + dtype + ", strided<[" +
                              std::to_string(extent * 16) +
                              ", 16, 1]>, #wafer.memory<spm, tensor>>";
            out << "%sub = memref.subview %d[0, 0, 0] [2, " << extent - 1
                << ", 16] [1, 1, 1] : " << dest << " to " << sub
                << "\nwafer.tile.fill %sub, %two : " << sub << ", " << dtype
                << '\n';
          } else {
            if (test == Case::Conditional)
              out << "scf.if %cond {\n";
            if (test == Case::Loop)
              out << "scf.for %i = %c0 to %c0 step %c1 {\n";
            out << "wafer.tile.fill %d, %two : " << dest << ", " << dtype
                << '\n';
            if (test == Case::Conditional || test == Case::Loop)
              out << "}\n";
          }
          if (test != Case::DeadOnly)
            out << "%read = wafer.tile.copy %d : " << dest << " -> " << dest
                << '\n';
          if (test == Case::SharedSource)
            out << "%shared = wafer.tile.copy %a : " << source << " -> "
                << source << '\n';
          out << "wafer.tile.yield } return } }";
          auto module =
              mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
          ASSERT_TRUE(module) << text;
          ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
          bool removed = test == Case::Fill || test == Case::Copy ||
                         test == Case::SharedSource || test == Case::DeadOnly;
          unsigned layouts = 0, fills = 0, writes = 0;
          TileRegionOp region;
          module->walk([&](LayoutMaterializeOp) { ++layouts; });
          module->walk([&](ComputeFillOp op) {
            ++fills;
            auto constant =
                op.getValue().getDefiningOp<mlir::arith::ConstantOp>();
            ASSERT_TRUE(constant);
            EXPECT_EQ(op.getValue().getType(),
                      mlir::cast<mlir::MemRefType>(op.getDest().getType())
                          .getElementType());
            auto value = mlir::cast<mlir::FloatAttr>(constant.getValue())
                             .getValueAsDouble();
            if (removed && test != Case::SharedSource)
              EXPECT_EQ(value, 2.0);
            else
              EXPECT_TRUE(value == 1.0 || value == 2.0);
          });
          module->walk([&](MoveCopyIntoOp) { ++writes; });
          module->walk([&](TileRegionOp op) { region = op; });
          EXPECT_EQ(layouts, removed ? 0u : 1u);
          EXPECT_EQ(fills,
                    test == Case::DeadOnly
                        ? 0u
                        : (removed && test != Case::SharedSource ? 1u : 2u));
          EXPECT_EQ(writes, test == Case::Copy ? 1u : 0u);
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          // An opaque external callee is a deliberate effect boundary, not a
          // device program. All other variants reach actual Instr/SPM planning.
          if (test == Case::Escape)
            continue;
          TileRegionToInstrLoweringSession session(*context);
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
          module->walk([&](SyncNCCJoinOp op) {
            EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
            EXPECT_FALSE(op->getParentOfType<mlir::scf::IfOp>());
          });
          EXPECT_TRUE(mlir::succeeded(
              planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
        }
}

TEST(StorageOptimizationTest, ForwardInitialReadWithoutMergingLiveStorage) {
  enum class Case {
    Direct,
    IndependentWrite,
    OldSourceWriteAfter,
    SourceWrite,
    SourceAliasWrite,
    EarlyRead,
    Escape,
    Nested,
    ConditionalConsumer,
    PermutedLayout,
    Pipeline
  };
  for (auto dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (auto kind : {"add", "max"})
        for (auto test :
             {Case::Direct, Case::IndependentWrite, Case::OldSourceWriteAfter,
              Case::SourceWrite, Case::SourceAliasWrite, Case::EarlyRead,
              Case::Escape, Case::Nested, Case::ConditionalConsumer,
              Case::PermutedLayout, Case::Pipeline}) {
          SCOPED_TRACE(::testing::Message()
                       << dtype << '/' << extent << '/' << kind << '/'
                       << static_cast<int>(test));
          auto context = createContext();
          // The leading unit axis keeps NCx bank padding equal for odd lengths.
          // The last-axis-65 variant instead requires an actual permutation.
          std::string shape = "1x2x" + std::to_string(extent) +
                              (test == Case::PermutedLayout ? "x65x" : "x64x") +
                              dtype;
          std::string source =
              "memref<" + shape + ", #wafer.memory<spm, tensor>>";
          std::string dest = "memref<" + shape + ", #wafer.memory<spm, ncx>>";
          std::string text;
          llvm::raw_string_ostream out(text);
          out << "#id = affine_map<(a,b,c,d)->(a,b,c,d)>\nmodule { ";
          if (test == Case::Escape)
            out << "func.func private @observe(" << source << ")\n";
          out << "func.func @main(%end: index, %cond: i1) { "
                 "wafer.tile.region(%end, %cond : index, i1) -> () "
                 "{\n^bb0(%bound: index, %condition: i1):\n"
                 "%a = memref.alloc() : "
              << source << "\n%b = memref.alloc() : " << dest
              << "\n%one = arith.constant 1.0 : " << dtype
              << "\n%c0 = arith.constant 0 : index\n"
                 "%c1 = arith.constant 1 : index\n"
                 "wafer.tile.fill %a, %one : "
              << source << ", " << dtype
              << "\nwafer.tile.fill %b, %one {fill_domain = "
                 "#wafer.fill_domain<physical_footprint>} : "
              << dest << ", " << dtype << '\n';
          if (test == Case::Nested)
            out << "scf.for %i = %c0 to %bound step %c1 { scf.if %condition "
                   "{\n";
          out << "%d = wafer.tile.materialize_layout %a : " << source << " -> "
              << dest << '\n';
          if (test == Case::SourceAliasWrite)
            out << "%alias = wafer.tile.reshape %a : " << source << " -> "
                << source << '\n';
          if (test == Case::SourceWrite || test == Case::SourceAliasWrite)
            out << "wafer.tile.fill %"
                << (test == Case::SourceWrite ? "a" : "alias")
                << ", %one : " << source << ", " << dtype << '\n';
          if (test == Case::IndependentWrite)
            out << "wafer.tile.fill %b, %one {fill_domain = "
                   "#wafer.fill_domain<physical_footprint>} : "
                << dest << ", " << dtype << '\n';
          if (test == Case::EarlyRead)
            out << "%early = wafer.tile.copy %d : " << dest << " -> " << dest
                << '\n';
          if (test == Case::Escape)
            out << "func.call @observe(%a) : (" << source << ") -> ()\n";
          if (test == Case::ConditionalConsumer)
            out << "scf.if %condition {\n";
          out << "wafer.tile.elementwise_into <" << kind
              << "> %d, %b into %d {indexing_maps = [#id, #id, #id]} : " << dest
              << ", " << dest << " into " << dest << '\n';
          if (test == Case::ConditionalConsumer)
            out << "}\n";
          // Both values are observed after the writer. Storage coalescing here
          // would destroy the old source; forwarding only its read is legal.
          out << "%old = wafer.tile.copy %a : " << source << " -> " << source
              << "\n%new = wafer.tile.copy %d : " << dest << " -> " << dest
              << '\n';
          if (test == Case::OldSourceWriteAfter)
            out << "wafer.tile.fill %a, %one : " << source << ", " << dtype
                << '\n';
          if (test == Case::Nested)
            out << "} }\n";
          out << "wafer.tile.yield } return } }";
          auto module =
              mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
          ASSERT_TRUE(module) << text;
          LayoutMaterializeOp layout;
          ComputeElementwiseIntoOp compute;
          module->walk([&](LayoutMaterializeOp op) { layout = op; });
          module->walk([&](ComputeElementwiseIntoOp op) { compute = op; });
          mlir::Value original = layout.getSource();
          bool forwarded =
              test == Case::Direct || test == Case::IndependentWrite ||
              test == Case::OldSourceWriteAfter || test == Case::Nested;
          if (test == Case::Pipeline) {
            mlir::IRRewriter rewriter(context.get());
            llvm::DenseSet<mlir::Operation *> excluded{compute};
            eliminateUnusedStorageInitialization(module->getOperation(),
                                                 rewriter, excluded);
          } else {
            ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
          }
          unsigned layouts = 0;
          module->walk([&](LayoutMaterializeOp) { ++layouts; });
          EXPECT_EQ(layouts, forwarded ? 0u : 1u);
          EXPECT_NE(compute.getDest(), original);
          EXPECT_EQ(compute.getInputs()[0],
                    forwarded ? original : compute.getDest());
          if (forwarded) {
            EXPECT_TRUE(
                compute.getDest().getDefiningOp<mlir::memref::AllocOp>());
          }
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          // Unknown external effects are a preservation boundary, not a target
          // call.
          if (test == Case::Escape)
            continue;
          TileRegionOp region;
          module->walk([&](TileRegionOp op) { region = op; });
          TileRegionToInstrLoweringSession session(*context);
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
          unsigned computes = 0;
          module->walk([&](InstrElementwiseOp op) {
            if (op.getKind() != (llvm::StringRef(kind) == "add"
                                     ? InstrElementwiseKind::Add
                                     : InstrElementwiseKind::Max))
              return;
            ++computes;
            EXPECT_NE(op.getDest(), original);
            if (forwarded) {
              EXPECT_EQ(op.getInputs()[0], original);
            }
          });
          EXPECT_EQ(computes, 1u);
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
          module->walk([&](SyncNCCJoinOp op) {
            EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
            EXPECT_FALSE(op->getParentOfType<mlir::scf::IfOp>());
          });
          EXPECT_TRUE(mlir::succeeded(
              planSPMMemoryModule(*module, 0, 8 * 1024 * 1024, 16)));
        }
}

TEST(StorageOptimizationTest, EliminateGemmInitializationThroughCompleteViews) {
  for (auto dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031})
      for (unsigned variant = 0; variant < 4; ++variant) {
        SCOPED_TRACE(::testing::Message()
                     << dtype << '/' << extent << '/' << variant);
        auto context = createContext();
        std::string lhs = "memref<2x" + std::to_string(extent) + "x16x" +
                          dtype + ", #wafer.memory<spm, ncx>>";
        std::string rhs = "memref<2x16x16x" + std::string(dtype) +
                          ", #wafer.memory<spm, ncx>>";
        std::string narrow = "memref<2x" + std::to_string(extent) +
                             "x16xf32, #wafer.memory<spm, ncx>>";
        std::string wide = "memref<2x1x" + std::to_string(extent) +
                           "x16xf32, #wafer.memory<spm, ncx>>";
        std::string wider = "memref<2x1x1x" + std::to_string(extent) +
                            "x16xf32, #wafer.memory<spm, ncx>>";
        std::string tensor = "memref<2x1x" + std::to_string(extent) +
                             "x16xf32, #wafer.memory<spm, tensor>>";
        std::string attributes =
            " {batch_count = 2 : i64, lhs_batch_dims = array<i64: 0>, "
            "lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64, "
            "rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, "
            "rhs_n_dim = 2 : i64, result_batch_dims = array<i64: 0>, "
            "result_m_dim = 1 : i64, result_n_dim = 2 : i64} : (" +
            lhs + ", " + rhs + ")";
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { func.func @main() { wafer.tile.region() -> () {\n"
               "%a = memref.alloc() : "
            << lhs << "\n%b = memref.alloc() : " << rhs
            << "\n%init = memref.alloc() : " << tensor
            << "\n%one = arith.constant 1.0 : f32\nwafer.tile.fill %init, %one "
               ": "
            << tensor
            << ", f32\n%d = wafer.tile.materialize_layout %init : " << tensor
            << " -> " << wide << "\n%w = wafer.tile.reshape %d : " << wide
            << " -> " << wider << "\n%n = wafer.tile.reshape %w : " << wider
            << " -> " << narrow << '\n';
        if (variant == 1)
          out << "%p = memref.alloc() : " << narrow
              << "\nwafer.tile.fill %p, %one {fill_domain = "
                 "#wafer.fill_domain<physical_footprint>} : "
              << narrow << ", f32\n";
        if (variant == 3)
          out << "%old = wafer.tile.copy %w : " << wider << " -> " << wider
              << '\n';
        if (variant == 2) {
          out << "%r = wafer.tile.gemm %a, %b psum(%n : " << narrow << ')'
              << attributes << " -> " << narrow
              << "\n%rwide = wafer.tile.reshape %r : " << narrow << " -> "
              << wide << "\nwafer.tile.copy_into %rwide into %d : " << wide
              << " into " << wide << '\n';
        } else {
          out << "wafer.tile.gemm_into %a, %b into %n";
          if (variant == 1)
            out << " psum(%p : " << narrow << ')';
          out << attributes << " into " << narrow << '\n';
        }
        out << "%read = wafer.tile.copy %d : " << wide << " -> " << wide
            << "\nwafer.tile.yield } return } }";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(module) << text;
        ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
        unsigned layouts = 0, fills = 0;
        TileRegionOp region;
        module->walk([&](LayoutMaterializeOp) { ++layouts; });
        module->walk([&](ComputeFillOp) { ++fills; });
        module->walk([&](TileRegionOp op) { region = op; });
        EXPECT_EQ(layouts, variant < 2 ? 0u : 1u);
        EXPECT_EQ(fills, variant == 0 ? 0u : 1u);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        TileRegionToInstrLoweringSession session(*context);
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
        unsigned gemms = 0;
        region.walk([&](InstrGemmOp op) {
          ++gemms;
          EXPECT_EQ(static_cast<bool>(op.getPsum()),
                    variant == 1 || variant == 2);
        });
        EXPECT_EQ(gemms, 1u);
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
        unsigned joins = 0;
        module->walk([&](SyncNCCJoinOp) { ++joins; });
        EXPECT_EQ(joins, 1u);
        EXPECT_TRUE(mlir::succeeded(
            planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
      }
}

TEST(StorageOptimizationTest,
     StorageInitializationPreservesSelectedPipelineObjects) {
  for (unsigned variant = 0; variant < 3; ++variant) {
    auto context = createContext();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module { func.func @main() { wafer.tile.region() -> () {
  %a = memref.alloc() : memref<2x1025x16xbf16, #wafer.memory<spm, ncx>>
  %d = wafer.tile.materialize_layout %a : memref<2x1025x16xbf16, #wafer.memory<spm, ncx>> -> memref<2x1025x16xbf16, #wafer.memory<spm, tensor>>
  %one = arith.constant 1.0 : bf16
  wafer.tile.fill %d, %one : memref<2x1025x16xbf16, #wafer.memory<spm, tensor>>, bf16
  wafer.tile.yield
} return } }
)mlir",
                                                          context.get());
    ASSERT_TRUE(module);
    llvm::DenseSet<mlir::Operation *> excluded;
    module->walk([&](mlir::Operation *op) {
      if ((variant == 0 && mlir::isa<LayoutMaterializeOp>(op)) ||
          (variant == 1 && mlir::isa<ComputeFillOp>(op)) ||
          (variant == 2 && mlir::isa<TileRegionOp>(op)))
        excluded.insert(op);
    });
    mlir::IRRewriter rewriter(context.get());
    eliminateUnusedStorageInitialization(module->getOperation(), rewriter,
                                         excluded);
    unsigned layouts = 0, fills = 0;
    module->walk([&](LayoutMaterializeOp) { ++layouts; });
    module->walk([&](ComputeFillOp) { ++fills; });
    EXPECT_EQ(layouts, 1u);
    EXPECT_EQ(fills, 1u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST(StorageOptimizationTest,
     ReusePrivateScalarArithmeticButPreserveRepeatedInput) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned variant = 0; variant < 3; ++variant) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << variant);
      auto context = createContext();
      std::string type = "memref<2x" + std::to_string(extent) +
                         "x16xf32, #wafer.memory<spm, tensor>>";
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "module { func.func @main() { wafer.tile.region() -> () {\n"
             "%a = memref.alloc() : "
          << type
          << "\n"
             "%scalar = arith.constant 0.125 : f32\n"
             "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : index\n"
             "%c8 = arith.constant 8 : index\n"
             "%p = wafer.tile.elementwise <exp> %a : ("
          << type << ") -> " << type << "\n";
      if (variant == 2)
        out << "scf.for %iv = %c0 to %c8 step %c1 {\n";
      out << "%r = wafer.tile.elementwise <mul> %p, %scalar : (" << type
          << ", f32) -> " << type << "\n";
      if (variant == 1)
        out << "%old = wafer.tile.copy %p : " << type << " -> " << type << "\n";
      if (variant == 2)
        out << "}\n";
      out << "wafer.tile.yield } return } }";
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
      ASSERT_TRUE(module) << text;
      ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
      unsigned multiplyInto = 0;
      module->walk([&](ComputeElementwiseIntoOp op) {
        if (op.getKind() == ComputeElementwiseKind::Mul) {
          ++multiplyInto;
          EXPECT_EQ(op.getInputs()[0], op.getDest());
        }
      });
      EXPECT_EQ(multiplyInto, variant == 0 ? 1u : 0u);
      TileRegionToInstrLoweringSession session(*context);
      module->walk([&](TileRegionOp op) {
        EXPECT_TRUE(mlir::succeeded(convertTileRegionToInstr(op, session)));
      });
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
      module->walk([&](SyncNCCJoinOp op) {
        EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
      });
      EXPECT_TRUE(mlir::succeeded(
          planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
    }
}

TEST(StorageOptimizationTest, PrivatePointwisePublicationExposesSelectLastUse) {
  enum class Case {
    Private,
    SharedRead,
    ExtraWrite,
    View,
    Escape,
    Loop,
    OldRead,
    NoReader,
    SourceShared
  };
  for (auto dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031})
      for (Case test : {Case::Private, Case::SharedRead, Case::ExtraWrite,
                        Case::View, Case::Escape, Case::Loop, Case::OldRead,
                        Case::NoReader, Case::SourceShared}) {
        SCOPED_TRACE(::testing::Message() << dtype << ':' << extent << ':'
                                          << static_cast<int>(test));
        auto context = createContext();
        std::string type = "memref<2x" + std::to_string(extent) + "x16x" +
                           dtype + ", #wafer.memory<spm, tensor>>";
        std::string mask = "memref<2x" + std::to_string(extent) +
                           "x16xi1, #wafer.memory<spm, tensor>>";
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { ";
        if (test == Case::Escape)
          out << "func.func private @escape(" << type << ")\n";
        out << "func.func @main() { wafer.tile.region() -> () {\n"
               "%input = memref.alloc() : "
            << type << "\n%true = memref.alloc() : " << type
            << "\n%mask = memref.alloc() : " << mask
            << "\n%published = memref.alloc() : " << type
            << "\n%c0 = arith.constant 0 : index\n"
               "%c1 = arith.constant 1 : index\n"
               "%c7 = arith.constant 7 : index\n"
               "%zero = arith.constant 0.0 : "
            << dtype
            << "\n%value = wafer.tile.elementwise <sub> %input, %input : ("
            << type << ", " << type << ") -> " << type << '\n';
        if (test == Case::OldRead)
          out << "%old = wafer.tile.copy %published : " << type << " -> "
              << type << '\n';
        out << "wafer.tile.copy_into %value into %published : " << type
            << " into " << type << '\n';
        if (test == Case::ExtraWrite)
          out << "wafer.tile.fill %published, %zero : " << type << ", " << dtype
              << '\n';
        if (test == Case::View)
          out << "%view = memref.cast %published : " << type << " to " << type
              << '\n';
        if (test == Case::Escape)
          out << "func.call @escape(%published) : (" << type << ") -> ()\n";
        if (test == Case::Loop)
          out << "scf.for %iv = %c0 to %c7 step %c1 {\n";
        if (test != Case::NoReader)
          out << "%selected = wafer.tile.elementwise <select> %mask, %true, "
              << (test == Case::View ? "%view" : "%published") << " : (" << mask
              << ", " << type << ", " << type << ") -> " << type << '\n';
        if (test == Case::Loop)
          out << "}\n";
        if (test == Case::SharedRead || test == Case::SourceShared)
          out << "%retained = wafer.tile.copy "
              << (test == Case::SharedRead ? "%published" : "%value") << " : "
              << type << " -> " << type << '\n';
        out << "wafer.tile.yield } return } }";
        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(module) << text;
        ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
        auto &result = module;
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*result)));
        unsigned selectInto = 0, selects = 0, copies = 0;
        result->walk([&](ComputeElementwiseIntoOp op) {
          if (op.getKind() != ComputeElementwiseKind::Select)
            return;
          ++selectInto;
          EXPECT_EQ(op.getInputs()[2], op.getDest());
          auto producer = op.getDest().getDefiningOp<ComputeElementwiseOp>();
          ASSERT_TRUE(producer);
          EXPECT_EQ(producer.getKind(), ComputeElementwiseKind::Sub);
        });
        result->walk([&](ComputeElementwiseOp op) {
          selects += op.getKind() == ComputeElementwiseKind::Select;
        });
        result->walk([&](MoveCopyIntoOp) { ++copies; });
        EXPECT_EQ(selectInto, test == Case::Private ? 1u : 0u);
        EXPECT_EQ(selects,
                  test == Case::Private || test == Case::NoReader ? 0u : 1u);
        if (test == Case::Private || test == Case::SharedRead) {
          EXPECT_EQ(copies, 0u);
        }
        if (test != Case::Private && test != Case::SharedRead)
          continue;
        TileRegionToInstrLoweringSession session(*context);
        result->walk([&](TileRegionOp region) {
          EXPECT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
        });
        unsigned transfers = 0, masks = 0;
        result->walk([&](InstrGatherScatterOp) { ++transfers; });
        result->walk([&](InstrMaskMoveOp) { ++masks; });
        EXPECT_EQ(transfers, test == Case::Private ? 0u : 2u);
        EXPECT_EQ(masks, 1u);
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*result)));
        EXPECT_TRUE(mlir::succeeded(
            planSPMMemoryModule(*result, 0, 3 * 1024 * 1024, 16)));
        result->walk([&](mlir::memref::AllocOp allocation) {
          EXPECT_TRUE(allocation->hasAttr(kWaferSPMOffsetAttrName));
        });
      }
}

TEST(StorageOptimizationTest,
     PrivatePublicationPreservesPipelineBindingsAndDynamicCopies) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto context = createContext();
    auto type = llvm::formatv(
                    "memref<2x{0}x16xf16, #wafer.memory<spm, tensor>>", extent)
                    .str();
    auto text = llvm::formatv(R"mlir(module {{ func.func @main() {{
      wafer.tile.region() -> () {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c7 = arith.constant 7 : index
        scf.for %iv = %c0 to %c7 step %c1 {{
          %a = memref.alloc() : {0}
          %d = memref.alloc() : {0}
          %p = wafer.tile.elementwise <sub> %a, %a : ({0}, {0}) -> {0}
          wafer.tile.copy_into %p into %d : {0} into {0}
          %r = wafer.tile.elementwise <exp> %d : ({0}) -> {0}
        }
        wafer.tile.yield
      }
      return
    } })mlir",
                              type)
                    .str();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module) << text;
    TilePipelineChoice choice;
    module->walk([&](mlir::scf::ForOp loop) { choice.loop = loop; });
    for (auto &op : choice.loop.getBody()->without_terminator()) {
      auto compute = mlir::dyn_cast<ComputeElementwiseOp>(op);
      choice.operations.push_back(
          {&op, compute && compute.getKind() == ComputeElementwiseKind::Exp
                    ? 1u
                    : 0u});
    }
    auto prepared = prepareLoopPipelines(*module, {choice});
    ASSERT_TRUE(prepared.succeeded())
        << (prepared.failure ? prepared.failure->detail : "");
    auto materialized =
        pipelineLoops(std::move(module), std::move(*prepared.prepared));
    ASSERT_TRUE(materialized.succeeded())
        << (materialized.failure ? materialized.failure->detail : "");
    llvm::DenseSet<mlir::Operation *> exclusions;
    for (const auto &pipeline : materialized.materialized->pipelines)
      for (const auto &binding : pipeline.operations)
        exclusions.insert(binding.operation);
    mlir::IRRewriter rewriter(context.get());
    preservePrivateScalarBroadcasts(*materialized.materialized->module,
                                    rewriter, exclusions);
    ASSERT_TRUE(mlir::succeeded(optimizeStorage(
        *materialized.materialized->module, rewriter, exclusions)));
    EXPECT_TRUE(
        mlir::succeeded(verifyPipelinedModule(*materialized.materialized)));
    ASSERT_EQ(materialized.materialized->pipelines.size(), 1u);
    auto &pipeline = materialized.materialized->pipelines.front();
    uint64_t copies = 0, readers = 0;
    for (const auto &binding : pipeline.operations) {
      uint64_t executions =
          binding.phase == PipelinePhase::Kernel ? pipeline.kernelTripCount : 1;
      if (auto copy = mlir::dyn_cast<MoveCopyIntoOp>(binding.operation)) {
        copies += executions;
        EXPECT_NE(copy.getSource(), copy.getDest());
      }
      if (auto compute =
              mlir::dyn_cast<ComputeElementwiseOp>(binding.operation))
        if (compute.getKind() == ComputeElementwiseKind::Exp)
          readers += executions;
    }
    EXPECT_EQ(copies, 7u);
    EXPECT_EQ(readers, 7u);
    EXPECT_TRUE(
        mlir::succeeded(mlir::verify(*materialized.materialized->module)));
  }
}

TEST(StorageOptimizationTest,
     AdjacentWritebackPreservesDestinationAndRejectsUnprovenReuse) {
  enum class Case {
    Disjoint,
    InPlace,
    Observer,
    Loop,
    Materialized,
    AliasView,
    PartialOverlap,
    UnknownAlias,
    Mapped,
    MappedAlias,
    MappedOtherInput,
    DifferentLayout,
    ExtraUse,
    InterveningRead
  };
  for (int64_t extent : {1024, 1025, 1031}) {
    for (Case test : {Case::Disjoint, Case::InPlace, Case::Observer, Case::Loop,
                      Case::Materialized, Case::AliasView, Case::PartialOverlap,
                      Case::UnknownAlias, Case::Mapped, Case::MappedAlias,
                      Case::MappedOtherInput, Case::DifferentLayout,
                      Case::ExtraUse, Case::InterveningRead}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<int>(test));
      auto context = createContext();
      mlir::Location loc = mlir::UnknownLoc::get(context.get());
      mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
      mlir::OpBuilder builder(context.get());
      auto memory =
          MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor);
      auto type =
          mlir::MemRefType::get({2, extent, 2}, builder.getF16Type(),
                                mlir::MemRefLayoutAttrInterface{}, memory);
      builder.setInsertionPointToStart(module->getBody());
      if (test == Case::UnknownAlias) {
        auto external = builder.create<mlir::func::FuncOp>(
            loc, "buffers", builder.getFunctionType({}, {type, type}));
        external.setPrivate();
      }
      auto function = builder.create<mlir::func::FuncOp>(
          loc, "main", builder.getFunctionType({}, {}));
      auto *entry = function.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      auto region = builder.create<TileRegionOp>(loc, mlir::TypeRange{},
                                                 entry->getArguments());
      auto *body = new mlir::Block();
      region.getBody().push_back(body);
      builder.setInsertionPointToStart(body);
      mlir::Value dest, input;
      if (test == Case::UnknownAlias) {
        auto call = builder.create<mlir::func::CallOp>(
            loc, "buffers", mlir::TypeRange{type, type}, mlir::ValueRange{});
        dest = call.getResult(0);
        input = call.getResult(1);
      } else if (test == Case::PartialOverlap) {
        auto bytes =
            mlir::MemRefType::get({8 * extent + 2}, builder.getI8Type(),
                                  mlir::MemRefLayoutAttrInterface{}, memory);
        auto storage = builder.create<mlir::memref::AllocOp>(loc, bytes);
        auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto two = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
        dest = builder.create<mlir::memref::ViewOp>(loc, type, storage, zero,
                                                    mlir::ValueRange{});
        input = builder.create<mlir::memref::ViewOp>(loc, type, storage, two,
                                                     mlir::ValueRange{});
      } else {
        auto destType =
            test == Case::DifferentLayout
                ? mlir::MemRefType::get(type.getShape(), type.getElementType(),
                                        mlir::MemRefLayoutAttrInterface{},
                                        MemoryAttr::get(context.get(),
                                                        MemorySpace::SPM,
                                                        MemLayout::Cx))
                : type;
        dest = builder.create<mlir::memref::AllocOp>(loc, destType);
        input = builder.create<mlir::memref::AllocOp>(loc, type);
      }
      mlir::Value observer;
      if (test == Case::Observer || test == Case::AliasView)
        observer = builder.create<mlir::memref::CastOp>(loc, type, dest);
      if (test == Case::AliasView)
        input = observer;
      if (test == Case::InPlace || test == Case::MappedAlias)
        input = dest;
      if (test == Case::Materialized) {
        auto cxType = mlir::MemRefType::get(
            type.getShape(), type.getElementType(),
            mlir::MemRefLayoutAttrInterface{},
            MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Cx));
        dest = builder.create<LayoutMaterializeOp>(loc, cxType, dest);
        input = builder.create<ComputeElementwiseOp>(
            loc, cxType, ComputeElementwiseKind::Add,
            mlir::ValueRange{dest, dest}, mlir::ArrayAttr{});
        type = cxType;
      }
      mlir::scf::ForOp loop;
      if (test == Case::Loop) {
        auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        auto one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
        auto end = builder.create<mlir::arith::ConstantIndexOp>(loc, 7);
        loop = builder.create<mlir::scf::ForOp>(loc, zero, end, one,
                                                mlir::ValueRange{dest});
        builder.setInsertionPointToStart(loop.getBody());
        dest = loop.getRegionIterArgs().front();
        input = dest;
      }
      mlir::AffineMap identity =
          mlir::AffineMap::getMultiDimIdentityMap(3, context.get());
      mlir::AffineMap inputMap =
          test == Case::Mapped || test == Case::MappedAlias
              ? mlir::AffineMap::getPermutationMap(
                    llvm::ArrayRef<unsigned>{2, 1, 0}, context.get())
              : identity;
      mlir::Value other = input;
      mlir::AffineMap otherMap = inputMap;
      if (test == Case::MappedOtherInput) {
        other = input;
        input = dest;
        otherMap = mlir::AffineMap::getPermutationMap(
            llvm::ArrayRef<unsigned>{2, 1, 0}, context.get());
      }
      auto maps = builder.getAffineMapArrayAttr({inputMap, otherMap, identity});
      auto value = builder.create<ComputeElementwiseOp>(
          loc, type, ComputeElementwiseKind::Add,
          mlir::ValueRange{input, other}, maps);
      if (test == Case::InterveningRead)
        builder.create<MoveCopyOp>(loc, type, dest, DDRResourceAttr());
      builder.create<MoveCopyIntoOp>(loc, value, dest);
      if (test == Case::ExtraUse)
        builder.create<MoveCopyOp>(loc, type, value, DDRResourceAttr());
      MoveCopyOp reader;
      if (test == Case::Observer)
        reader =
            builder.create<MoveCopyOp>(loc, type, observer, DDRResourceAttr());
      if (loop) {
        builder.create<mlir::scf::YieldOp>(loc, dest);
        builder.setInsertionPointAfter(loop);
      }
      builder.create<TileYieldOp>(loc);
      builder.setInsertionPointAfter(region);
      builder.create<mlir::func::ReturnOp>(loc);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*module)));
      unsigned functional = 0, copies = 0, updates = 0;
      region.walk([&](ComputeElementwiseOp) { ++functional; });
      region.walk([&](MoveCopyIntoOp) { ++copies; });
      region.walk([&](ComputeElementwiseIntoOp update) {
        ++updates;
        EXPECT_EQ(update.getDest(), dest);
      });
      bool eliminate = test == Case::Disjoint || test == Case::InPlace ||
                       test == Case::Observer || test == Case::Loop ||
                       test == Case::Materialized || test == Case::Mapped ||
                       test == Case::MappedOtherInput;
      EXPECT_EQ(updates, eliminate ? 1u : 0u);
      EXPECT_EQ(functional, eliminate && test != Case::Materialized ? 0u : 1u);
      EXPECT_EQ(copies, eliminate ? 0u : 1u);
      if (reader) {
        EXPECT_EQ(reader.getSource(), observer);
      }
      if (loop) {
        EXPECT_EQ(
            mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator())
                .getOperand(0),
            dest);
      }
      if (!eliminate)
        continue;
      unsigned beforeAllocations = 0;
      region.walk([&](mlir::memref::AllocOp) { ++beforeAllocations; });
      TileRegionToInstrLoweringSession session(*context);
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      unsigned instructions = 0, afterAllocations = 0, transfers = 0;
      InstrElementwiseOp firstInstruction;
      region.walk([&](InstrElementwiseOp instruction) {
        ++instructions;
        if (test == Case::Materialized) {
          if (firstInstruction) {
            EXPECT_EQ(instruction.getDest(),
                      firstInstruction.getInputs().front());
            EXPECT_EQ(instruction.getInputs().front(),
                      firstInstruction.getDest());
          } else {
            firstInstruction = instruction;
          }
        } else {
          EXPECT_EQ(instruction.getDest(), dest);
        }
      });
      region.walk([&](InstrGatherScatterOp) { ++transfers; });
      region.walk([&](mlir::memref::AllocOp) { ++afterAllocations; });
      EXPECT_EQ(instructions, test == Case::Materialized ? 2u : 1u);
      EXPECT_EQ(transfers, test == Case::Mapped ? 2u
                           : test == Case::Observer ||
                                   test == Case::Materialized ||
                                   test == Case::MappedOtherInput
                               ? 1u
                               : 0u);
      EXPECT_EQ(afterAllocations,
                beforeAllocations +
                    (test == Case::Materialized || test == Case::Mapped ? 2u
                     : test == Case::Observer || test == Case::MappedOtherInput
                         ? 1u
                         : 0u));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
      EXPECT_TRUE(mlir::succeeded(
          planSPMMemoryModule(*module, 0, 3 * 1024 * 1024, 16)));
    }
  }
}

TEST(StorageOptimizationTest,
     LoopCarriedElementwiseUsesExplicitExistingDestination) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    mlir::Location loc = mlir::UnknownLoc::get(context.get());
    auto module = mlir::ModuleOp::create(loc);
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());
    auto function = moduleBuilder.create<mlir::func::FuncOp>(
        loc, "main",
        moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
    mlir::Block *entry = function.addEntryBlock();
    mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
    auto region = builder.create<TileRegionOp>(loc, mlir::TypeRange{},
                                               mlir::ValueRange{});
    region.getBody().push_back(new mlir::Block());
    mlir::OpBuilder regionBuilder =
        mlir::OpBuilder::atBlockBegin(&region.getBody().front());
    auto type = mlir::MemRefType::get(
        {2, extent, 128}, regionBuilder.getF32Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::NCx));
    auto state = regionBuilder.create<mlir::memref::AllocOp>(loc, type);
    auto input = regionBuilder.create<mlir::memref::AllocOp>(loc, type);
    auto lower = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 7);
    auto step = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto loop = regionBuilder.create<mlir::scf::ForOp>(loc, lower, upper, step,
                                                       mlir::ValueRange{state});
    if (!loop.getBody()->empty() &&
        mlir::isa<mlir::scf::YieldOp>(loop.getBody()->back()))
      loop.getBody()->back().erase();
    mlir::OpBuilder loopBuilder = mlir::OpBuilder::atBlockEnd(loop.getBody());
    mlir::AffineMap identity =
        mlir::AffineMap::getMultiDimIdentityMap(type.getRank(), context.get());
    auto maps =
        loopBuilder.getAffineMapArrayAttr({identity, identity, identity});
    auto next = loopBuilder.create<ComputeElementwiseOp>(
        loc, type, ComputeElementwiseKind::Add,
        mlir::ValueRange{loop.getRegionIterArgs().front(), input}, maps);
    loopBuilder.create<mlir::scf::YieldOp>(loc, next.getResult());
    regionBuilder.setInsertionPointAfter(loop);
    regionBuilder.create<TileYieldOp>(loc);
    builder.setInsertionPointAfter(region);
    builder.create<mlir::func::ReturnOp>(loc);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

    mlir::OwningOpRef<mlir::ModuleOp> owned(module);
    ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*owned)));
    EXPECT_EQ(countLoops(owned->getOperation()), 1u);
    unsigned functional = 0;
    ComputeElementwiseIntoOp update;
    owned->walk([&](ComputeElementwiseOp) { ++functional; });
    owned->walk(
        [&](ComputeElementwiseIntoOp operation) { update = operation; });
    EXPECT_EQ(functional, 0u);
    ASSERT_TRUE(update);
    auto currentLoop = update->getParentOfType<mlir::scf::ForOp>();
    ASSERT_TRUE(currentLoop);
    EXPECT_EQ(update.getDest(), currentLoop.getRegionIterArgs().front());
    EXPECT_EQ(update.getInputs().front(), update.getDest());
    EXPECT_EQ(
        mlir::cast<mlir::scf::YieldOp>(currentLoop.getBody()->getTerminator())
            .getOperand(0),
        update.getDest());
    unsigned loopAllocations = 0;
    currentLoop.walk([&](mlir::memref::AllocOp) { ++loopAllocations; });
    EXPECT_EQ(loopAllocations, 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*owned)));
  }
}

TEST(StorageOptimizationTest,
     LoopCarriedLayoutChangeUsesExplicitExistingDestination) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    mlir::Location loc = mlir::UnknownLoc::get(context.get());
    auto module = mlir::ModuleOp::create(loc);
    mlir::OpBuilder moduleBuilder(module.getBodyRegion());
    auto function = moduleBuilder.create<mlir::func::FuncOp>(
        loc, "main",
        moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
    mlir::Block *entry = function.addEntryBlock();
    mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
    auto region = builder.create<TileRegionOp>(loc, mlir::TypeRange{},
                                               mlir::ValueRange{});
    region.getBody().push_back(new mlir::Block());
    mlir::OpBuilder regionBuilder =
        mlir::OpBuilder::atBlockBegin(&region.getBody().front());
    auto tensorType = mlir::MemRefType::get(
        {2, extent, 128}, regionBuilder.getF16Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Tensor));
    auto cxType = mlir::MemRefType::get(
        {2, extent, 128}, regionBuilder.getF16Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, MemLayout::Cx));
    auto source = regionBuilder.create<mlir::memref::AllocOp>(loc, tensorType);
    auto state = regionBuilder.create<mlir::memref::AllocOp>(loc, cxType);
    auto lower = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto upper = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 7);
    auto step = regionBuilder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto loop = regionBuilder.create<mlir::scf::ForOp>(loc, lower, upper, step,
                                                       mlir::ValueRange{state});
    if (!loop.getBody()->empty() &&
        mlir::isa<mlir::scf::YieldOp>(loop.getBody()->back()))
      loop.getBody()->back().erase();
    mlir::OpBuilder loopBuilder = mlir::OpBuilder::atBlockEnd(loop.getBody());
    auto next = loopBuilder.create<LayoutMaterializeOp>(loc, cxType, source);
    loopBuilder.create<mlir::scf::YieldOp>(loc, next.getResult());
    regionBuilder.setInsertionPointAfter(loop);
    regionBuilder.create<TileYieldOp>(loc);
    builder.setInsertionPointAfter(region);
    builder.create<mlir::func::ReturnOp>(loc);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(module)));

    mlir::OwningOpRef<mlir::ModuleOp> owned(module);
    ASSERT_TRUE(mlir::succeeded(optimizeStorageForTest(*owned)));
    EXPECT_EQ(countLoops(owned->getOperation()), 1u);
    unsigned functional = 0;
    MoveCopyIntoOp update;
    owned->walk([&](LayoutMaterializeOp) { ++functional; });
    owned->walk([&](MoveCopyIntoOp operation) { update = operation; });
    EXPECT_EQ(functional, 0u);
    ASSERT_TRUE(update);
    auto currentLoop = update->getParentOfType<mlir::scf::ForOp>();
    ASSERT_TRUE(currentLoop);
    EXPECT_EQ(update.getDest(), currentLoop.getRegionIterArgs().front());
    EXPECT_EQ(getWaferMemoryAttr(
                  mlir::cast<mlir::MemRefType>(update.getSource().getType()))
                  .getLayout(),
              MemLayout::Tensor);
    EXPECT_EQ(getWaferMemoryAttr(
                  mlir::cast<mlir::MemRefType>(update.getDest().getType()))
                  .getLayout(),
              MemLayout::Cx);
    EXPECT_EQ(
        mlir::cast<mlir::scf::YieldOp>(currentLoop.getBody()->getTerminator())
            .getOperand(0),
        update.getDest());
    unsigned loopAllocations = 0;
    currentLoop.walk([&](mlir::memref::AllocOp) { ++loopAllocations; });
    EXPECT_EQ(loopAllocations, 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*owned)));
  }
}

} // namespace
