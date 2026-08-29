//===- LayoutOptimizationTest.cpp ----------------------------===//

#include "Wafer/Transforms/Tile/LayoutOptimization.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

class LayoutOptimizationTest : public ::testing::Test {
protected:
  LayoutOptimizationTest() {
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(uint64_t extent,
                                          bool writableResults) {
    std::string users = writableResults ? R"mlir(
    %zero = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    memref.store %zero, %first[%c0, %c0, %c0] : memref<2xEXTENTx64xf16, #wafer.memory<spm, ncx>>
    memref.store %zero, %second[%c0, %c0, %c0] : memref<2xEXTENTx64xf16, #wafer.memory<spm, ncx>>
)mlir"
                                        : R"mlir(
    %back0 = wafer.tile.materialize_layout %first : memref<2xEXTENTx64xf16, #wafer.memory<spm, ncx>> -> memref<2xEXTENTx64xf16, #wafer.memory<spm, tensor>>
    %back1 = wafer.tile.materialize_layout %second : memref<2xEXTENTx64xf16, #wafer.memory<spm, ncx>> -> memref<2xEXTENTx64xf16, #wafer.memory<spm, tensor>>
)mlir";
    std::string text = R"mlir(
module {
  func.func @main() {
    %source = memref.alloc() : memref<2xEXTENTx64xf16, #wafer.memory<spm, tensor>>
    %first = wafer.tile.materialize_layout %source : memref<2xEXTENTx64xf16, #wafer.memory<spm, tensor>> -> memref<2xEXTENTx64xf16, #wafer.memory<spm, ncx>>
    %second = wafer.tile.materialize_layout %source : memref<2xEXTENTx64xf16, #wafer.memory<spm, tensor>> -> memref<2xEXTENTx64xf16, #wafer.memory<spm, ncx>>
USERS
    return
  }
}
)mlir";
    auto replaceAll = [&](llvm::StringRef marker, llvm::StringRef value) {
      size_t position = 0;
      while ((position = text.find(marker.str(), position)) !=
             std::string::npos) {
        text.replace(position, marker.size(), value.str());
        position += value.size();
      }
    };
    replaceAll("USERS", users);
    replaceAll("EXTENT", std::to_string(extent));
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  parseReadOnlyUses(uint64_t extent, unsigned useCount,
                    bool writeSourceBeforeSecondUse) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "module {\n"
           << "  func.func @main() -> f16 {\n"
           << "    %source = memref.alloc() : memref<2x" << extent
           << "x64xf16, #wafer.memory<spm, tensor>>\n"
           << "    %c0 = arith.constant 0 : index\n"
           << "    %zero = arith.constant 0.000000e+00 : f16\n";
    for (unsigned index = 0; index < useCount; ++index) {
      if (index == 1 && writeSourceBeforeSecondUse)
        stream << "    memref.store %zero, %source[%c0, %c0, %c0] "
               << ": memref<2x" << extent
               << "x64xf16, #wafer.memory<spm, tensor>>\n";
      stream << "    %layout" << index
             << " = wafer.tile.materialize_layout %source : memref<2x" << extent
             << "x64xf16, #wafer.memory<spm, tensor>> -> memref<2x" << extent
             << "x64xf16, #wafer.memory<spm, ncx>>\n"
             << "    %value" << index << " = memref.load %layout" << index
             << "[%c0, %c0, %c0] : memref<2x" << extent
             << "x64xf16, #wafer.memory<spm, ncx>>\n";
    }
    std::string accumulator = "%value0";
    for (unsigned index = 1; index < useCount; ++index) {
      std::string next = "%sum" + std::to_string(index);
      stream << "    " << next << " = arith.addf " << accumulator << ", %value"
             << index << " : f16\n";
      accumulator = std::move(next);
    }
    stream << "    return " << accumulator << " : f16\n"
           << "  }\n"
           << "}\n";
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(LayoutOptimizationTest,
       ReusesReadOnlyConversionsAndClosesDeadChainAtRealisticScale) {
  for (uint64_t extent : {UINT64_C(1024), UINT64_C(1025)}) {
    SCOPED_TRACE(extent);
    auto module = parse(extent, /*writableResults=*/false);
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations;
    llvm::SmallVector<LayoutOptimizationInput, 1> candidates{
        LayoutOptimizationInput{*module, &relations}};
    LayoutOptimizationResult result = optimizeTileLayouts(candidates);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.invocations, 1u);
    EXPECT_EQ(result.statistics.hardOnlyInvocations, 1u);
    EXPECT_EQ(result.statistics.layoutMaterializationsBefore, 4u);
    EXPECT_EQ(result.statistics.sharedMaterializationsReused, 1u);
    EXPECT_EQ(result.statistics.unusedMaterializationsErased, 3u);
    EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest,
       DoesNotAliasIndependentlyWrittenMaterializations) {
  for (uint64_t extent : {UINT64_C(1024), UINT64_C(1025)}) {
    SCOPED_TRACE(extent);
    auto module = parse(extent, /*writableResults=*/true);
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations;
    llvm::SmallVector<LayoutOptimizationInput, 1> candidates{
        LayoutOptimizationInput{*module, &relations}};
    LayoutOptimizationResult result = optimizeTileLayouts(candidates);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.sharedMaterializationsReused, 0u);
    EXPECT_EQ(result.statistics.unusedMaterializationsErased, 0u);
    EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 2u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest,
       SharesOneExactReadOnlyConversionAcrossFifteenUses) {
  for (uint64_t extent : {UINT64_C(1024), UINT64_C(1025)}) {
    SCOPED_TRACE(extent);
    auto module = parseReadOnlyUses(extent, /*useCount=*/15,
                                    /*writeSourceBeforeSecondUse=*/false);
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations;
    llvm::SmallVector<LayoutOptimizationInput, 1> candidates{
        LayoutOptimizationInput{*module, &relations}};
    LayoutOptimizationResult result = optimizeTileLayouts(candidates);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.layoutMaterializationsBefore, 15u);
    EXPECT_EQ(result.statistics.sharedMaterializationsReused, 14u);
    EXPECT_EQ(result.statistics.unusedMaterializationsErased, 0u);
    EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 1u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest,
       DoesNotReuseAConversionAcrossAnAliasingSourceWrite) {
  for (uint64_t extent : {UINT64_C(1024), UINT64_C(1025)}) {
    SCOPED_TRACE(extent);
    auto module = parseReadOnlyUses(extent, /*useCount=*/2,
                                    /*writeSourceBeforeSecondUse=*/true);
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations;
    llvm::SmallVector<LayoutOptimizationInput, 1> candidates{
        LayoutOptimizationInput{*module, &relations}};
    LayoutOptimizationResult result = optimizeTileLayouts(candidates);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.sharedMaterializationsReused, 0u);
    EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 2u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

} // namespace
