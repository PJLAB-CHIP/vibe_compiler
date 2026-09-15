//===- CapacityFeedbackTest.cpp ----------------------------------------===//

#include "Wafer/Driver/PhysicalDataflow/CapacityFeedback.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/Target/TargetMemory.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Linalg/CommunicationRegionClosure.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/ExecutionStructure.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

namespace {
using namespace wafer;
using namespace wafer::compiler::detail;

class CapacityFeedbackTest : public ::testing::Test {
protected:
  CapacityFeedbackTest() {
    mlir::DialectRegistry registry;
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  source(int64_t extent, unsigned tiles, bool sharedInput = false,
         bool derivedInput = false, int64_t columns = 2048,
         bool opaqueDerived = false, llvm::StringRef assembly = {}) {
    std::string suffix =
        std::to_string(extent) + "x" + std::to_string(columns) + "xf16>";
    std::string input = "tensor<1x" + suffix;
    std::string output = "tensor<1x2x" + suffix;
    std::string text;
    llvm::raw_string_ostream ir(text);
    ir << "module {\n";
    for (unsigned tile = 0; tile < tiles; ++tile) {
      ir << "wafer.tile.module card_id = 0 tile_id = " << tile
         << " { func.func @entry(%a: " << input
         << " {wafer.program_argument = #wafer.program_argument<0>}, %b: "
         << input << " {wafer.program_argument = #wafer.program_argument<1>}) "
         << "-> (" << output << ", " << output << ") {\n"
         << "%r:2 = wafer.tile.region(%a, %b : " << input << ", " << input
         << ") -> (" << output << ", " << output << ") {\n"
         << "^bb0(%x: " << input << ", %y: " << input << "):\n";
      if (derivedInput && opaqueDerived)
        ir << "%derived = tensor.generate { ^bb0(%ib: index, %im: index, %ik: "
              "index): "
           << "%value = tensor.extract %x[%ib, %im, %ik] : " << input << "\n"
           << "tensor.yield %value : f16 } : " << input << "\n";
      else if (derivedInput)
        ir << "%derived_empty = tensor.empty() : " << input << "\n"
           << "%derived = linalg.generic {indexing_maps = ["
           << "affine_map<(b, m, k) -> (b, m, k)>, "
           << "affine_map<(b, m, k) -> (b, m, k)>], "
           << "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]} "
           << "ins(%x : " << input << ") outs(%derived_empty : " << input
           << ") { ^bb0(%value: f16, %old: f16): "
           << "%negated = arith.negf %value : f16\n"
           << "linalg.yield %negated : f16 } -> " << input << "\n";
      if (!assembly.empty()) {
        const std::string half = "tensor<1x" + std::to_string(extent / 2) +
                                 "x" + std::to_string(columns) + "xf16>";
        if (assembly == "full") {
          ir << "%assembled = tensor.insert_slice %y into %x[0, 0, 0] [1, "
             << extent << ", " << columns << "] [1, 1, 1] : " << input
             << " into " << input << "\n";
        } else if (assembly == "partial" || assembly == "unknown_remainder") {
          ir << "%piece = tensor.extract_slice %y[0, 0, 0] [1, " << extent / 2
             << ", " << columns << "] [1, 1, 1] : " << input << " to " << half
             << "\n%assembled = tensor.insert_slice %piece into "
             << (assembly == "partial" ? "%x" : "%derived") << "[0, 0, 0] [1, "
             << extent / 2 << ", " << columns << "] [1, 1, 1] : " << half
             << " into " << input << "\n";
        } else if (assembly == "overwritten_unknown") {
          const int64_t rest = extent - extent / 2;
          const std::string tail = "tensor<1x" + std::to_string(rest) + "x" +
                                   std::to_string(columns) + "xf16>";
          ir << "%first = tensor.extract_slice %y[0, 0, 0] [1, " << extent / 2
             << ", " << columns << "] [1, 1, 1] : " << input << " to " << half
             << "\n%last = tensor.extract_slice %y[0, " << extent / 2
             << ", 0] [1, " << rest << ", " << columns
             << "] [1, 1, 1] : " << input << " to " << tail
             << "\n%updated = tensor.insert_slice %first into %derived[0, 0, "
                "0] [1, "
             << extent / 2 << ", " << columns << "] [1, 1, 1] : " << half
             << " into " << input
             << "\n%assembled = tensor.insert_slice %last into %updated[0, "
             << extent / 2 << ", 0] [1, " << rest << ", " << columns
             << "] [1, 1, 1] : " << tail << " into " << input << "\n";
        } else {
          const std::string wide = "tensor<1x" + std::to_string(2 * extent) +
                                   "x" + std::to_string(columns) + "xf16>";
          ir << "%base = tensor.empty() : " << wide
             << "\n%prefix = tensor.insert_slice %x into %base[0, 0, 0] [1, "
             << extent << ", " << columns << "] [1, 1, 1] : " << input
             << " into " << wide
             << "\n%both = tensor.insert_slice %y into %prefix[0, "
             << (assembly == "overlap" ? extent / 2 : extent) << ", 0] [1, "
             << extent << ", " << columns << "] [1, 1, 1] : " << input
             << " into " << wide
             << "\n%assembled = tensor.extract_slice %both[0, "
             << (assembly == "read_suffix" ? extent
                 : assembly == "overlap"   ? 0
                                           : extent / 2)
             << ", 0] [1, " << extent << ", " << columns
             << "] [1, 1, 1] : " << wide << " to " << input << "\n";
        }
      }
      for (unsigned scope = 0; scope < 2; ++scope)
        ir << "%e" << scope << " = tensor.empty() : " << output << "\n"
           << "%v" << scope << " = linalg.generic {indexing_maps = ["
           << "affine_map<(b, r, m, k) -> (b, m, k)>, "
           << "affine_map<(b, r, m, k) -> (b, r, m, k)>], "
           << "iterator_types = [\"parallel\", \"parallel\", \"parallel\", "
              "\"parallel\"]} ins("
           << (scope == 1 && !assembly.empty() ? "%assembled"
               : scope == 0 || sharedInput     ? "%x"
               : derivedInput                  ? "%derived"
                                               : "%y")
           << " : " << input << ") outs(%e" << scope << " : " << output
           << ") { ^bb0(%value: f16, %old: f16): linalg.yield %value : f16 } "
           << "-> " << output << "\n";
      ir << "wafer.tile.yield %v0, %v1 : " << output << ", " << output
         << "\n}\nreturn %r#0, %r#1 : " << output << ", " << output << "\n}}\n";
    }
    ir << "}\n";
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  actual(int64_t extent, bool writeInterference = false, bool hasABI = true,
         int64_t columns = 2048, int64_t inputID = 0) {
    std::string shape =
        "1x" + std::to_string(extent) + "x" + std::to_string(columns) + "xf16";
    std::string ddr = "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
    std::string spm = "memref<" + shape + ", #wafer.memory<spm, tensor>>";
    std::string text;
    llvm::raw_string_ostream ir(text);
    ir << "module { func.func @entry(%input: " << ddr;
    if (hasABI)
      ir << " {wafer.program_argument = #wafer.program_argument<" << inputID
         << ">}";
    ir << ") {\n%unused = wafer.tile.region(%input : " << ddr << ") -> (" << ddr
       << ") {\n^bb0(%ddr: " << ddr << "):\n"
       << "%c0 = arith.constant 0 : index\n"
       << "%c1 = arith.constant 1 : index\n%c2 = arith.constant 2 : index\n"
       << "%zero = arith.constant 0.0 : f16\n"
       << "%stage = memref.alloc() : " << spm << "\n"
       << "%copy = memref.alloc() : " << spm << "\n"
       << "scf.for %i = %c0 to %c2 step %c1 {\n"
       << "wafer.instr.rdma %ddr to %stage {byte_count = "
       << extent * columns * 2
       << " : i64, inner_bytes = " << extent * columns * 2
       << " : i64, src_strides = array<i64: 0, 0, 0>, "
          "src_iterations = array<i64: 1, 1, 1>} : "
       << ddr << " to " << spm << "\n"
       << "wafer.instr.gather_scatter %stage to %copy {byte_count = "
       << extent * columns * 2
       << " : i64, inner_bytes = " << extent * columns * 2
       << " : i64, src_strides = array<i64: 0, 0, 0>, "
          "src_iterations = array<i64: 1, 1, 1>, "
          "dst_strides = array<i64: 0, 0, 0>, "
          "dst_iterations = array<i64: 1, 1, 1>} : "
       << spm << " to " << spm << "\n}\n";
    if (writeInterference)
      ir << "memref.store %zero, %stage[%c0, %c0, %c0] : " << spm << "\n";
    ir << "wafer.tile.yield %ddr : " << ddr << "\n}\nreturn\n}}\n";
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  }

  InputCapacityFeedback observe(mlir::ModuleOp source,
                                mlir::OwningOpRef<mlir::ModuleOp> actual,
                                unsigned tile, bool joint = false,
                                bool expectOversized = true) {
    std::vector<TemporalDomain> domains;
    std::vector<TemporalChoice> choices;
    source.walk([&](TileRegionOp region) {
      auto built = buildTemporalDomain(region);
      EXPECT_TRUE(built.succeeded());
      if (!built.succeeded())
        return;
      auto first = joint ? built.domain->getFirstChoice()
                         : built.domain->getFirstIndependentChoice();
      choices.push_back(*first.getChoice());
      domains.push_back(std::move(*built.domain));
    });
    std::vector<const TemporalDomain *> references;
    for (const auto &domain : domains)
      references.push_back(&domain);
    EXPECT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*actual)));
    StructuredMaterializationRelations relations;
    rebuildCurrentBufferOwnerRelations(*actual, relations);
    TileMemoryPlanningFailure failure;
    InputCapacityFeedback feedback;
    bool observed = false;
    auto observer = [&](const SPMMemoryPlanningFailure &certificate,
                        const StructuredMaterializationRelations &) {
      observed = true;
      EXPECT_EQ(certificate.kind,
                SPMMemoryPlanningFailureKind::CapacityOverflow);
      if (expectOversized)
        EXPECT_FALSE(certificate.individuallyOversizedDemands.empty());
      else {
        EXPECT_TRUE(certificate.individuallyOversizedDemands.empty());
        EXPECT_FALSE(certificate.capacityConflictDemands.empty());
      }
      feedback = deriveInputCapacityFeedback(CardId(0), TileId(tile),
                                             certificate, references, choices);
    };
    EXPECT_TRUE(mlir::failed(planTileMemory(std::move(actual), &failure,
                                            &relations, false, observer)));
    EXPECT_TRUE(observed);
    EXPECT_TRUE(failure.spmCapacityOverflow);
    EXPECT_EQ(feedback.status, CapacityFeedbackStatus::Valid)
        << feedback.detail;
    return feedback;
  }

  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CapacityFeedbackTest, ImmutableInitializersSurvivePublicationFences) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned variant : {0u, 1u, 2u}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(variant);
      auto program = source(extent, 4);
      ASSERT_TRUE(program);
      mlir::Builder attrs(context.get());
      auto tensor =
          mlir::RankedTensorType::get({1, extent, 2048}, attrs.getF16Type());
      auto contents =
          mlir::DenseElementsAttr::get(tensor, attrs.getF16FloatAttr(0.5));
      program->walk([&](mlir::func::FuncOp function) {
        mlir::OpBuilder builder(&function.front(), function.front().begin());
        auto literal = builder.create<mlir::arith::ConstantOp>(
            function.getLoc(), contents);
        function.getArgument(0).replaceAllUsesWith(literal);
      });
      auto instructions = actual(extent);
      ASSERT_TRUE(instructions);
      auto function = *instructions->getOps<mlir::func::FuncOp>().begin();
      auto memory =
          mlir::cast<mlir::MemRefType>(function.getArgument(0).getType());
      mlir::OpBuilder builder(instructions->getBody(),
                              instructions->getBody()->begin());
      auto initial = variant == 1 ? mlir::DenseElementsAttr::get(
                                        tensor, attrs.getF16FloatAttr(0.75))
                                  : contents;
      builder.create<mlir::memref::GlobalOp>(
          instructions->getLoc(), "coefficients",
          builder.getStringAttr("private"), memory, initial, variant != 2,
          builder.getI64IntegerAttr(256));
      builder.setInsertionPointToStart(&function.front());
      auto global = builder.create<mlir::memref::GetGlobalOp>(
          function.getLoc(), memory, "coefficients");
      function.getArgument(0).replaceAllUsesWith(global);
      builder.create<mlir::LLVM::FenceOp>(function.getLoc(),
                                          mlir::LLVM::AtomicOrdering::release);
      auto feedback = observe(*program, std::move(instructions), 3);
      EXPECT_EQ(feedback.status, CapacityFeedbackStatus::Valid);
      if (variant == 0) {
        EXPECT_EQ(feedback.coordinates.size(), 2u);
        EXPECT_EQ(feedback.coordinates.count({3, 0, 2}), 1u);
        EXPECT_EQ(feedback.coordinates.count({3, 0, 3}), 1u);
      } else {
        EXPECT_TRUE(feedback.coordinates.empty());
        EXPECT_GT(feedback.unavailableDemands, 0u);
      }
    }
}

TEST_F(CapacityFeedbackTest, FusedOutputAxesReachTheirConsumerParameters) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned tiles : {4u, 16u})
      for (bool transpose : {false, true}) {
        SCOPED_TRACE(extent);
        SCOPED_TRACE(tiles);
        SCOPED_TRACE(transpose);
        std::string lhs = "tensor<1x" + std::to_string(extent) + "x2048xf16>";
        std::string rhs = "tensor<1x2048x1024xf16>";
        std::string wide = "tensor<1x" + std::to_string(extent) + "x1024xf32>";
        std::string narrow =
            transpose ? "tensor<1x1024x" + std::to_string(extent) + "xf16>"
                      : "tensor<1x" + std::to_string(extent) + "x1024xf16>";
        std::string text;
        llvm::raw_string_ostream ir(text);
        ir << "module {\n";
        for (unsigned tile = 0; tile < tiles; ++tile) {
          ir << "wafer.tile.module card_id = 0 tile_id = " << tile
             << " { func.func @entry(%a: " << lhs
             << " {wafer.program_argument = #wafer.program_argument<0>}, %b: "
             << rhs
             << " {wafer.program_argument = #wafer.program_argument<1>}) -> "
             << narrow << " { %r = wafer.tile.region(%a, %b : " << lhs << ", "
             << rhs << ") -> (" << narrow << ") { ^bb0(%x: " << lhs
             << ", %y: " << rhs << "): %zero = arith.constant 0.0 : f32 "
             << "%empty = tensor.empty() : " << wide
             << " %init = linalg.fill ins(%zero : f32) outs(%empty : " << wide
             << ") -> " << wide
             << " %g = linalg.batch_matmul ins(%x, %y : " << lhs << ", " << rhs
             << ") outs(%init : " << wide << ") -> " << wide
             << " %out = tensor.empty() : " << narrow
             << " %v = linalg.generic {indexing_maps = [affine_map<(b,m,n)->"
             << (transpose ? "(b,n,m)>, " : "(b,m,n)>, ")
             << "affine_map<(b,m,n)->(b,m,n)>], iterator_types = "
             << "[\"parallel\",\"parallel\",\"parallel\"]} ins(%g : " << wide
             << ") outs(%out : " << narrow
             << ") { ^bb0(%value: f32, %old: f16): "
             << "%c = arith.truncf %value : f32 to f16 "
             << "linalg.yield %c : f16 } -> " << narrow
             << " wafer.tile.yield %v : " << narrow
             << " } return %r : " << narrow << " } }\n";
        }
        ir << "}\n";
        auto program =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(program);
        auto feedback = observe(*program, actual(extent), tiles - 1, true);
        EXPECT_EQ(feedback.status, CapacityFeedbackStatus::Valid);
        EXPECT_EQ(feedback.ambiguousInputs, 0u);
        EXPECT_EQ(feedback.unavailableDemands, 0u);
        ASSERT_EQ(feedback.coordinates.size(), 2u);
        EXPECT_EQ(feedback.coordinates.count({tiles - 1, 0, 3}), 1u);
        EXPECT_EQ(
            feedback.coordinates.count({tiles - 1, 1, transpose ? 2u : 1u}),
            1u);
        // N is absent from this actual A-panel demand. The output map moves
        // the M repair into the consumer without claiming its unrelated axis.
        EXPECT_EQ(feedback.coordinates.count({tiles - 1, 0, 1}), 0u);
      }
}

TEST_F(CapacityFeedbackTest,
       ActualInputSelectsOnlyItsScopeAndMappedDimensions) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned tiles : {4u, 16u}) {
      auto program = source(extent, tiles);
      auto instructions = actual(extent);
      ASSERT_TRUE(program && instructions);
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*program)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*instructions)));
      auto feedback = observe(*program, std::move(instructions), tiles - 1);
      EXPECT_EQ(feedback.unavailableDemands, 0u);
      EXPECT_EQ(feedback.ambiguousInputs, 0u);
      ASSERT_EQ(feedback.coordinates.size(), 2u);
      EXPECT_EQ(feedback.coordinates.count({tiles - 1, 0, 2}), 1u);
      EXPECT_EQ(feedback.coordinates.count({tiles - 1, 0, 3}), 1u);
    }
}

TEST_F(CapacityFeedbackTest, SharedInputAssociatesEveryProvenReader) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned tiles : {4u, 16u}) {
      auto program = source(extent, tiles, true);
      auto feedback = observe(*program, actual(extent), tiles - 1);
      EXPECT_EQ(feedback.unavailableDemands, 0u);
      EXPECT_EQ(feedback.ambiguousInputs, 0u);
      EXPECT_GT(feedback.sharedInputDemands, 0u);
      ASSERT_EQ(feedback.coordinates.size(), 4u);
      for (size_t scope : {0u, 1u}) {
        EXPECT_EQ(feedback.coordinates.count({tiles - 1, scope, 2}), 1u);
        EXPECT_EQ(feedback.coordinates.count({tiles - 1, scope, 3}), 1u);
        // Broadcast output-only dimension r is not an input access axis.
        EXPECT_EQ(feedback.coordinates.count({tiles - 1, scope, 1}), 0u);
      }
    }
}

TEST_F(CapacityFeedbackTest, InsertedPiecesRetainOnlyDemandedInputReaders) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned tiles : {4u, 16u})
      for (llvm::StringRef assembly :
           {"partial", "full", "read_suffix", "cross_pieces", "overlap",
            "unknown_remainder", "overwritten_unknown"}) {
        SCOPED_TRACE(extent);
        SCOPED_TRACE(tiles);
        SCOPED_TRACE(assembly.str());
        const bool opaque = assembly.contains("unknown");
        auto program =
            source(extent, tiles, false, opaque, 2048, opaque, assembly);
        ASSERT_TRUE(program);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*program)));
        for (int64_t input : {0, 1}) {
          auto feedback = observe(
              *program, actual(extent, false, true, 2048, input), tiles - 1);
          if (assembly == "unknown_remainder" && input == 0) {
            EXPECT_GT(feedback.ambiguousInputs, 0u);
            EXPECT_TRUE(feedback.coordinates.empty());
            continue;
          }
          EXPECT_EQ(feedback.ambiguousInputs, 0u);
          EXPECT_EQ(feedback.unavailableDemands, 0u);
          const bool secondReadsFirst = assembly == "partial" ||
                                        assembly == "cross_pieces" ||
                                        assembly == "overlap";
          const size_t expected = input == 0 && secondReadsFirst ? 4 : 2;
          EXPECT_EQ(feedback.coordinates.size(), expected);
          for (size_t scope : {0u, 1u})
            for (size_t axis : {2u, 3u}) {
              const bool reads =
                  input == 0 ? scope == 0 || secondReadsFirst : scope == 1;
              EXPECT_EQ(feedback.coordinates.count({tiles - 1, scope, axis}),
                        reads);
            }
        }
      }
}

TEST_F(CapacityFeedbackTest, ActualLiveConflictUsesTheSameCoordinateProof) {
  for (int64_t extent : {1024, 1025, 1031}) {
    // Each buffer fits; their simultaneous source-read/destination-write
    // lifetime during the GS exceeds the actual default SPM range.
    auto program = source(extent, 4, false, false, 1024);
    auto instructions = actual(extent, false, true, 1024);
    ASSERT_TRUE(program && instructions);
    auto feedback = observe(*program, std::move(instructions), 2,
                            /*joint=*/false, /*expectOversized=*/false);
    ASSERT_EQ(feedback.coordinates.size(), 2u);
    EXPECT_EQ(feedback.coordinates.count({2, 0, 2}), 1u);
    EXPECT_EQ(feedback.coordinates.count({2, 0, 3}), 1u);
  }
}

TEST_F(CapacityFeedbackTest, FusedUnaryInputUsesItsProvenIndexCoordinates) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto program = source(extent, 4, false, true);
    ASSERT_TRUE(program);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*program)));
    auto feedback = observe(*program, actual(extent), 2, true);
    EXPECT_EQ(feedback.ambiguousInputs, 0u);
    EXPECT_GT(feedback.sharedInputDemands, 0u);
    ASSERT_EQ(feedback.coordinates.size(), 4u);
    for (size_t scope : {0u, 1u}) {
      EXPECT_EQ(feedback.coordinates.count({2, scope, 2}), 1u);
      EXPECT_EQ(feedback.coordinates.count({2, scope, 3}), 1u);
    }
  }
}

TEST_F(CapacityFeedbackTest, SeparateRegionResultDoesNotInventAnInputReader) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto program = source(extent, 4, false, true);
    ASSERT_TRUE(program);
    llvm::SmallVector<TileRegionOp> regions;
    program->walk([&](TileRegionOp region) { regions.push_back(region); });
    for (TileRegionOp region : regions) {
      auto producer =
          *region.getBody().front().getOps<mlir::linalg::GenericOp>().begin();
      mlir::Value result = producer.getResult(0);
      auto *empty = producer.getDpsInitOperand(0)->get().getDefiningOp();
      mlir::OpBuilder builder(region);
      auto prior = builder.create<TileRegionOp>(
          region.getLoc(), mlir::TypeRange{result.getType()},
          mlir::ValueRange{region.getInputs().front()});
      prior.getBody().push_back(new mlir::Block());
      auto &body = prior.getBody().front();
      auto input = body.addArgument(result.getType(), region.getLoc());
      result.replaceAllUsesWith(region.getBody().getArgument(1));
      producer.getDpsInputOperand(0)->set(input);
      empty->moveBefore(&body, body.end());
      producer->moveBefore(&body, body.end());
      builder.setInsertionPointToEnd(&body);
      builder.create<TileYieldOp>(region.getLoc(), mlir::ValueRange{result});
      region->setOperand(1, prior.getResult(0));
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*program)));
    auto feedback = observe(*program, actual(extent), 2, true);
    EXPECT_EQ(feedback.ambiguousInputs, 0u);
    ASSERT_EQ(feedback.coordinates.size(), 4u);
    // The input is read by the prior region and the downstream direct use.
    // Reading the prior region's computed result is a separate access.
    EXPECT_EQ(feedback.coordinates.count({4, 0, 1}), 1u);
    EXPECT_EQ(feedback.coordinates.count({4, 0, 2}), 1u);
    EXPECT_EQ(feedback.coordinates.count({5, 0, 2}), 1u);
    EXPECT_EQ(feedback.coordinates.count({5, 0, 3}), 1u);
  }
}

TEST_F(CapacityFeedbackTest, UnmappedProducerCannotHideAnAmbiguousConsumer) {
  for (int64_t extent : {1024, 1025, 1031}) {
    // The tensor.generate has no admitted Temporal fusion/indexing contract.
    // Its known SSA dependency must prevent attribution to only the other use.
    auto program = source(extent, 4, false, true, 2048, true);
    ASSERT_TRUE(program);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*program)));
    auto feedback = observe(*program, actual(extent), 2, true);
    EXPECT_TRUE(feedback.coordinates.empty());
    EXPECT_GT(feedback.ambiguousInputs, 0u);
  }
}

TEST_F(CapacityFeedbackTest,
       ActualPipelineRebuildsOriginsAfterLoopReplacement) {
  for (int64_t extent : {1024, 1025, 1031}) {
    auto program = source(extent, 4);
    ASSERT_TRUE(program);
    const std::string shape = "1x" + std::to_string(extent) + "x2048xf16";
    const std::string ddr = "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
    const std::string spm = "memref<" + shape + ", #wafer.memory<spm, tensor>>";
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module { func.func @entry(%input: " << ddr
        << " {wafer.program_argument = #wafer.program_argument<0>}, %output: "
        << ddr << ") {\n"
        << "\"wafer.tile.region\"(%input, %output) ({^bb0(%src: " << ddr
        << ", %dst: " << ddr << "):\n"
        << "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : index\n"
        << "%c4 = arith.constant 4 : index\n"
        << "scf.for %i = %c0 to %c4 step %c1 {\n"
        << "%buffer = memref.alloc() : " << spm << "\n"
        << "wafer.tile.load %src into %buffer : " << ddr << " into " << spm
        << "\nwafer.tile.store %buffer, %dst : " << spm << " -> " << ddr
        << "\n}\n\"wafer.tile.yield\"() : () -> ()\n}) : (" << ddr << ", "
        << ddr << ") -> ()\nreturn\n}}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations;
    rebuildCurrentBufferOwnerRelations(*module, relations);
    ASSERT_TRUE(hasDistanceOneLoadPipeline(*module));
    auto pipelined =
        materializeDistanceOneLoadPipelines(std::move(module), relations);
    ASSERT_TRUE(pipelined.succeeded()) << pipelined.failure->detail;
    ASSERT_EQ(pipelined.materialized->pipelines.size(), 1u);
    module = std::move(pipelined.materialized->module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    TileRegionToInstrLoweringSession lowering(*context);
    module->walk([&](TileRegionOp region) {
      EXPECT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, lowering)));
    });
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    auto feedback = observe(*program, std::move(module), 3);
    ASSERT_EQ(feedback.coordinates.size(), 2u);
    EXPECT_EQ(feedback.coordinates.count({3, 0, 2}), 1u);
    EXPECT_EQ(feedback.coordinates.count({3, 0, 3}), 1u);
    EXPECT_EQ(feedback.ambiguousInputs, 0u);
    EXPECT_EQ(feedback.unavailableDemands, 0u);
  }
}

TEST_F(CapacityFeedbackTest, ActualRegionMergeDoesNotRequireOldBodyHandles) {
  for (int64_t extent : {1024, 1025, 1031}) {
    const std::string type =
        "tensor<1x" + std::to_string(extent) + "x2048xf16>";
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module { wafer.target.topology @target {card_grid = array<i64: 1, "
           "1>, "
        << "card_interconnect = \"mesh\", tile_grid = array<i64: 4, 4>, "
        << "unavailable_tiles = array<i64>}\n";
    for (unsigned tile = 0; tile < 4; ++tile) {
      out << "wafer.tile.module card_id = 0 tile_id = " << tile
          << " { func.func @entry(%input: " << type
          << " {wafer.program_argument = #wafer.program_argument<0>}, %remote: "
          << type << " {wafer.cross_tile_boundary_input}) -> " << type
          << " {\n";
      for (unsigned region = 0; region < 2; ++region)
        out << "%r" << region << " = wafer.tile.region("
            << (region ? "%remote" : "%input") << " : " << type << ") -> ("
            << type << ") { ^bb0(%arg: " << type << "):\n"
            << "%empty = tensor.empty() : " << type << "\n"
            << "%v = linalg.generic {indexing_maps = ["
            << "affine_map<(b, m, k) -> (b, m, k)>, "
            << "affine_map<(b, m, k) -> (b, m, k)>], "
            << "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]} "
            << "ins(%arg : " << type << ") outs(%empty : " << type
            << ") { ^bb1(%value: f16, %old: f16): "
            << "%negated = arith.negf %value : f16\n"
            << "linalg.yield %negated : f16 } -> " << type << "\n"
            << "wafer.tile.yield %v : " << type << "\n}\n";
      out << "return %r1 : " << type << "\n}}\n";
    }
    out << "}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    mlir::OwningOpRef<mlir::ModuleOp> original(module->clone());
    llvm::SmallVector<llvm::SmallVector<TileRegionOp, 2>, 4> regions(4);
    module->walk([&](TileModuleOp tile) {
      tile.walk([&](TileRegionOp region) {
        regions[tile.getTileId()].push_back(region);
      });
    });
    StructuredMaterializationRelations relations;
    for (unsigned tile = 0; tile < 4; ++tile) {
      ASSERT_EQ(regions[tile].size(), 2u);
      relations.boundaryRelations.push_back(
          {regions[tile][0].getResult(0),
           regions[tile ^ 1][1].getBody().getArgument(0)});
      relations.structuralOutputs.push_back({0, regions[tile][1].getResult(0)});
    }
    CommunicationRegionClosureStatistics closure;
    SpatialRegionMaterializationFailure failure;
    ASSERT_TRUE(mlir::succeeded(closeCrossTileCommunicationRegions(
        *module, relations, &closure, &failure)))
        << failure.detail;
    ASSERT_EQ(closure.mergedRegions, 8u);
    ASSERT_EQ(closure.mergedTileScopes, 4u);
    auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    auto compute = lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(compute.succeeded()) << compute.detail;
    BoundaryMovementOptions options;
    options.transport = BoundaryMovementTransport::SharedDDR;
    auto movement =
        materializeTileBoundaryMovement(*module, relations, options);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    std::string detail;
    auto standalone =
        createStandaloneTileModules(std::move(module), &detail, &relations);
    ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
    ASSERT_EQ(standalone->size(), 4u);
    auto &tile = standalone->back();
    ASSERT_EQ(tile.tileId.getValue(), 3u);
    TileRegionToInstrLoweringSession lowering(*context);
    tile.module->walk([&](TileRegionOp region) {
      EXPECT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, lowering)));
    });
    ASSERT_TRUE(mlir::succeeded(
        convertBufferizationCopiesToInstr(*tile.module, lowering)));
    auto feedback = observe(*original, std::move(tile.module), 3);
    ASSERT_EQ(feedback.coordinates.size(), 2u);
    EXPECT_EQ(feedback.coordinates.count({6, 0, 1}), 1u);
    EXPECT_EQ(feedback.coordinates.count({6, 0, 2}), 1u);
    EXPECT_EQ(feedback.ambiguousInputs, 0u);
  }
}

TEST_F(CapacityFeedbackTest, NonMovementWriterAndMissingABIStayUnavailable) {
  for (bool interference : {false, true}) {
    auto program = source(1025, 4);
    auto feedback =
        observe(*program, actual(1025, interference, interference), 1);
    EXPECT_TRUE(feedback.coordinates.empty());
    EXPECT_GT(feedback.unavailableDemands, 0u);
  }
}

} // namespace
