//===- LayoutOptimizationTest.cpp --------------------------------------===//

#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

template <typename OpTy> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](OpTy) { ++count; });
  return count;
}

class LayoutOptimizationTest : public ::testing::Test {
protected:
  LayoutOptimizationTest() {
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        text, mlir::ParserConfig(context.get()));
  }

  static StructuredMaterializationRelations
  outputRelation(mlir::ModuleOp module) {
    TileRegionOp region;
    module.walk([&](TileRegionOp current) { region = current; });
    EXPECT_TRUE(region);
    StructuredMaterializationRelations relations;
    if (region)
      relations.structuralOutputs.push_back({0, region.getResult(0)});
    return relations;
  }

  static std::string makeSharedContractionSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%lhs: tensor<2x)mlir"
           << extent << R"mlir(x128xf16>,
                     %rhs0: tensor<2x128x64xf16>,
                     %rhs1: tensor<2x128x64xf16>) {
      %result = wafer.tile.region(
          %lhs, %rhs0, %rhs1 : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>, tensor<2x128x64xf16>,
                               tensor<2x128x64xf16>)
          -> (tensor<2x)mlir"
           << extent << R"mlir(x64xf16>) {
      ^bb0(%local_lhs: tensor<2x)mlir"
           << extent << R"mlir(x128xf16>, %local_rhs0: tensor<2x128x64xf16>,
           %local_rhs1: tensor<2x128x64xf16>):
        %view = tensor.extract_slice %local_lhs[0, 0, 0]
            [2, )mlir"
           << extent << R"mlir(, 128] [1, 1, 1]
            : tensor<2x)mlir"
           << extent << R"mlir(x128xf16> to tensor<2x)mlir" << extent
           << R"mlir(x128xf16>
        %empty0 = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>
        %mm0 = linalg.batch_matmul
            ins(%view, %local_rhs0 : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>, tensor<2x128x64xf16>)
            outs(%empty0 : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>) -> tensor<2x)mlir" << extent
           << R"mlir(x64xf16>
        %empty1 = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>
        %mm1 = linalg.batch_matmul
            ins(%view, %local_rhs1 : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>, tensor<2x128x64xf16>)
            outs(%empty1 : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>) -> tensor<2x)mlir" << extent
           << R"mlir(x64xf16>
        %sum_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>
        %sum = linalg.generic {
            indexing_maps = [#id, #id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%mm0, %mm1 : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>, tensor<2x)mlir" << extent
           << R"mlir(x64xf16>)
            outs(%sum_empty : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>) {
          ^bb1(%a: f16, %b: f16, %old: f16):
            %next = arith.addf %a, %b : f16
            linalg.yield %next : f16
        } -> tensor<2x)mlir"
           << extent << R"mlir(x64xf16>
        wafer.tile.yield %sum : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>
      }
      return
    }
  }
}
)mlir";
    return text;
  }

  static std::string makeFanoutContractionSource(int64_t extent,
                                                 unsigned useCount) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "module {\n"
           << "  wafer.tile.module card_id = 0 tile_id = 0 {\n"
           << "    func.func @entry(%lhs: tensor<2x" << extent << "x128xf16>";
    for (unsigned index = 0; index < useCount; ++index)
      stream << ", %rhs" << index << ": tensor<2x128x64xf16>";
    stream << ") {\n"
           << "      %result = wafer.tile.region(%lhs";
    for (unsigned index = 0; index < useCount; ++index)
      stream << ", %rhs" << index;
    stream << " : tensor<2x" << extent << "x128xf16>";
    for (unsigned index = 0; index < useCount; ++index)
      stream << ", tensor<2x128x64xf16>";
    stream << ") -> (tensor<2x" << extent << "x64xf16>) {\n"
           << "      ^bb0(%local_lhs: tensor<2x" << extent << "x128xf16>";
    for (unsigned index = 0; index < useCount; ++index)
      stream << ", %local_rhs" << index << ": tensor<2x128x64xf16>";
    stream << "):\n"
           << "        %view = tensor.extract_slice %local_lhs[0, 0, 0] "
              "[2, "
           << extent << ", 128] [1, 1, 1] : tensor<2x" << extent
           << "x128xf16> to tensor<2x" << extent << "x128xf16>\n";
    for (unsigned index = 0; index < useCount; ++index)
      stream << "        %empty" << index << " = tensor.empty() : tensor<2x"
             << extent << "x64xf16>\n"
             << "        %mm" << index << " = linalg.batch_matmul ins(%view, "
             << "%local_rhs" << index << " : tensor<2x" << extent
             << "x128xf16>, tensor<2x128x64xf16>) outs(%empty" << index
             << " : tensor<2x" << extent << "x64xf16>) -> tensor<2x" << extent
             << "x64xf16>\n";
    stream << "        wafer.tile.yield %mm0 : tensor<2x" << extent
           << "x64xf16>\n"
           << "      }\n"
           << "      return\n"
           << "    }\n"
           << "  }\n"
           << "}\n";
    return text;
  }

  static std::string makeElementwiseSource(llvm::StringRef map,
                                           llvm::StringRef type,
                                           unsigned rank) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "#id = " << map << "\n"
           << "module {\n"
           << "  wafer.tile.module card_id = 0 tile_id = 0 {\n"
           << "    func.func @entry(%input: " << type << ") {\n"
           << "      %result = wafer.tile.region(%input : " << type << ") -> ("
           << type << ") {\n"
           << "      ^bb0(%local: " << type << "):\n"
           << "        %empty = tensor.empty() : " << type << "\n"
           << "        %mapped = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [";
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      if (dimension != 0)
        stream << ", ";
      stream << "\"parallel\"";
    }
    stream << "]} ins(%local : " << type << ") outs(%empty : " << type
           << ") {\n"
           << "        ^bb1(%value: f16, %old: f16):\n"
           << "          %next = arith.addf %value, %value : f16\n"
           << "          linalg.yield %next : f16\n"
           << "        } -> " << type << "\n"
           << "        wafer.tile.yield %mapped : " << type << "\n"
           << "      }\n"
           << "      return\n"
           << "    }\n"
           << "  }\n"
           << "}\n";
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(LayoutOptimizationTest,
       AssignsCurrentValueUsesAndSharesOneConversionAtRealisticScale) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeSharedContractionSource(extent));
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations = outputRelation(*module);
    LayoutOptimizationResult result =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.bufferizationInvocations, 1u);
    EXPECT_EQ(result.statistics.hardOnlyInvocations, 1u);
    EXPECT_GE(result.statistics.valueGroups, 1u);
    EXPECT_GE(result.statistics.useBindings, 1u);
    EXPECT_EQ(result.statistics.selectedMaterializations, 1u);
    EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 1u);
    EXPECT_EQ(result.statistics.redundantPublicationCopies, 0u);
    EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*module)));
    EXPECT_TRUE(mlir::succeeded(
        checkStructuredBufferRelationsCurrent(*module, relations)));

    LayoutMaterializeOp conversion;
    module->walk([&](LayoutMaterializeOp current) { conversion = current; });
    ASSERT_TRUE(conversion);
    EXPECT_GE(std::distance(conversion.getResult().use_begin(),
                            conversion.getResult().use_end()),
              2);
    auto sourceMemory = getWaferMemoryAttr(
        mlir::cast<mlir::MemRefType>(conversion.getSource().getType()));
    auto resultMemory = getWaferMemoryAttr(
        mlir::cast<mlir::MemRefType>(conversion.getResult().getType()));
    ASSERT_TRUE(sourceMemory && resultMemory);
    EXPECT_NE(sourceMemory.getLayout(), resultMemory.getLayout());

    ASSERT_EQ(relations.structuralOutputs.size(), 1u);
    auto outputType = mlir::dyn_cast<mlir::MemRefType>(
        relations.structuralOutputs.front().endpoint.getType());
    ASSERT_TRUE(outputType);
    EXPECT_TRUE(isWaferDDRMemRefType(outputType));
    LayoutOptimizationResult repeated =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    EXPECT_EQ(repeated.status, ExactPBQPStatus::BrokenContract);
  }
}

TEST_F(LayoutOptimizationTest,
       RankFiveAndSixCurrentValuesBufferizeWithoutShapeSpecialCases) {
  struct Case {
    llvm::StringRef map;
    llvm::StringRef type;
    unsigned rank;
  };
  for (const Case &testCase :
       {Case{"affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>",
             "tensor<2x4x1024x16x64xf16>", 5},
        Case{"affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, "
             "d4, d5)>",
             "tensor<2x4x8x1025x16x64xf16>", 6}}) {
    SCOPED_TRACE(testCase.rank);
    auto module = parse(
        makeElementwiseSource(testCase.map, testCase.type, testCase.rank));
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations = outputRelation(*module);
    LayoutOptimizationResult result =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.bufferizationInvocations, 1u);
    EXPECT_EQ(result.statistics.redundantPublicationCopies, 0u);
  }
}

TEST_F(LayoutOptimizationTest,
       BindsOneRaggedOutputPieceWithoutFullSPMOrDDRPublicationCopy) {
  constexpr llvm::StringLiteral text = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 3 {
    func.func @entry(%input: tensor<1x1025x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<1x1025x64xf16>)
          -> (tensor<2x1025x64xf16>) {
      ^bb0(%local: tensor<1x1025x64xf16>):
        %empty = tensor.empty() : tensor<1x1025x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x1025x64xf16>)
            outs(%empty : tensor<1x1025x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x1025x64xf16>
        %full = tensor.empty() : tensor<2x1025x64xf16>
        %inserted = tensor.insert_slice %mapped into %full[1, 0, 0]
            [1, 1025, 64] [1, 1, 1]
            : tensor<1x1025x64xf16> into tensor<2x1025x64xf16>
        wafer.tile.yield %inserted : tensor<2x1025x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.outputDestinations, 1u);
  EXPECT_EQ(result.statistics.outputSubviews, 1u);
  EXPECT_EQ(result.statistics.redundantPublicationCopies, 0u);
  EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(module->getOperation()), 0u);
  ASSERT_EQ(relations.structuralOutputs.size(), 1u);
  auto subview = relations.structuralOutputs.front()
                     .endpoint.getDefiningOp<mlir::memref::SubViewOp>();
  ASSERT_TRUE(subview);
  EXPECT_EQ(subview.getStaticOffsets(), llvm::ArrayRef<int64_t>({1, 0, 0}));
  EXPECT_EQ(subview.getStaticSizes(), llvm::ArrayRef<int64_t>({1, 1025, 64}));
  unsigned fullSPMAllocations = 0;
  module->walk([&](mlir::memref::AllocOp allocation) {
    auto type = allocation.getType();
    if (isWaferSPMMemRefType(type) &&
        type.getShape() == llvm::ArrayRef<int64_t>({2, 1025, 64}))
      ++fullSPMAllocations;
  });
  EXPECT_EQ(fullSPMAllocations, 0u);
}

TEST_F(LayoutOptimizationTest,
       OneTwoAndFifteenReadUsesShareExactlyOneActualConversion) {
  for (auto [extent, uses] :
       {std::pair<int64_t, unsigned>{1024, 1}, {1025, 2}, {1031, 15}}) {
    SCOPED_TRACE(::testing::Message() << extent << "/" << uses);
    auto module = parse(makeFanoutContractionSource(extent, uses));
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations = outputRelation(*module);
    LayoutOptimizationResult result =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.selectedMaterializations, 1u);
    EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 1u);
    LayoutMaterializeOp conversion;
    module->walk([&](LayoutMaterializeOp current) { conversion = current; });
    ASSERT_TRUE(conversion);
    EXPECT_EQ(std::distance(conversion.getResult().use_begin(),
                            conversion.getResult().use_end()),
              uses);
  }
}

TEST_F(LayoutOptimizationTest, InterveningCurrentWritePreventsConversionReuse) {
  std::string text = makeSharedContractionSource(/*extent=*/1025);
  constexpr llvm::StringLiteral marker = "        %empty1";
  size_t position = text.find(marker.str());
  ASSERT_NE(position, std::string::npos);
  text.insert(position,
              R"mlir(        %view_buffer = bufferization.to_memref %view
            : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
        %c0 = arith.constant 0 : index
        %zero = arith.constant 0.000000e+00 : f16
        memref.store %zero, %view_buffer[%c0, %c0, %c0]
            : memref<2x1025x128xf16, #wafer.memory<spm, tensor>>
)mlir");
  auto module = parse(text);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.selectedMaterializations, 2u);
  EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 2u);
}

TEST_F(LayoutOptimizationTest,
       IndependentRequestsProduceByteEquivalentLayoutResolvedIR) {
  const std::string source =
      makeFanoutContractionSource(/*extent=*/1031, /*useCount=*/15);
  auto first = parse(source);
  auto second = parse(source);
  ASSERT_TRUE(first && second);
  StructuredMaterializationRelations firstRelations = outputRelation(*first);
  StructuredMaterializationRelations secondRelations = outputRelation(*second);
  LayoutOptimizationResult firstResult =
      resolveCurrentLayoutsAndBufferize(*first, firstRelations);
  LayoutOptimizationResult secondResult =
      resolveCurrentLayoutsAndBufferize(*second, secondRelations);
  ASSERT_TRUE(firstResult.succeeded()) << firstResult.detail;
  ASSERT_TRUE(secondResult.succeeded()) << secondResult.detail;
  std::string firstText;
  llvm::raw_string_ostream firstStream(firstText);
  first->print(firstStream);
  firstStream.flush();
  std::string secondText;
  llvm::raw_string_ostream secondStream(secondText);
  second->print(secondStream);
  secondStream.flush();
  EXPECT_EQ(firstText, secondText);
  EXPECT_EQ(firstResult.statistics.solverWork,
            secondResult.statistics.solverWork);
}

TEST_F(LayoutOptimizationTest,
       ExplicitCurrentNTensorAllocationSurvivesOneShotBufferization) {
  constexpr llvm::StringLiteral text = R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%unused: tensor<2x1024x64xf16>) {
      %result = wafer.tile.region(
          %unused : tensor<2x1024x64xf16>)
          -> (tensor<2x1024x64xf16>) {
      ^bb0(%arg0: tensor<2x1024x64xf16>):
        %zero = arith.constant 0.000000e+00 : f16
        %allocation = bufferization.alloc_tensor()
            {memory_space = #wafer.memory<spm, ntensor>}
            : tensor<2x1024x64xf16>
        %filled = linalg.fill ins(%zero : f16)
            outs(%allocation : tensor<2x1024x64xf16>)
            -> tensor<2x1024x64xf16>
        wafer.tile.yield %filled : tensor<2x1024x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  unsigned ntensorAllocations = 0;
  module->walk([&](mlir::memref::AllocOp allocation) {
    MemoryAttr memory = getWaferMemoryAttr(allocation.getType());
    ntensorAllocations += memory && memory.getSpace() == MemorySpace::SPM &&
                          memory.getLayout() == MemLayout::NTensor;
  });
  EXPECT_EQ(ntensorAllocations, 1u);
}

TEST_F(LayoutOptimizationTest, RankTwoContractionSelectsCxUseLayout) {
  constexpr llvm::StringLiteral text = R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%lhs: tensor<1025x128xf16>,
                     %rhs: tensor<128x64xf16>) {
      %result = wafer.tile.region(
          %lhs, %rhs : tensor<1025x128xf16>, tensor<128x64xf16>)
          -> (tensor<1025x64xf16>) {
      ^bb0(%local_lhs: tensor<1025x128xf16>,
           %local_rhs: tensor<128x64xf16>):
        %view = tensor.extract_slice %local_lhs[0, 0] [1025, 128] [1, 1]
            : tensor<1025x128xf16> to tensor<1025x128xf16>
        %empty = tensor.empty() : tensor<1025x64xf16>
        %matmul = linalg.matmul
            ins(%view, %local_rhs : tensor<1025x128xf16>,
                                     tensor<128x64xf16>)
            outs(%empty : tensor<1025x64xf16>) -> tensor<1025x64xf16>
        wafer.tile.yield %matmul : tensor<1025x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  LayoutMaterializeOp conversion;
  module->walk([&](LayoutMaterializeOp current) { conversion = current; });
  ASSERT_TRUE(conversion);
  MemoryAttr memory = getWaferMemoryAttr(
      mlir::cast<mlir::MemRefType>(conversion.getResult().getType()));
  ASSERT_TRUE(memory);
  EXPECT_EQ(memory.getLayout(), MemLayout::Cx);
}

TEST_F(LayoutOptimizationTest,
       CompatibleReshapeChainRemainsMetadataOnlyAtRealisticScale) {
  constexpr llvm::StringLiteral text = R"mlir(
#id = affine_map<(m, n) -> (m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1024x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x1024x64xf16>)
          -> (tensor<2x1024x64xf16>) {
      ^bb0(%local: tensor<2x1024x64xf16>):
        %collapsed = tensor.collapse_shape %local [[0, 1], [2]]
            : tensor<2x1024x64xf16> into tensor<2048x64xf16>
        %empty = tensor.empty() : tensor<2048x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel"]}
            ins(%collapsed : tensor<2048x64xf16>)
            outs(%empty : tensor<2048x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<2048x64xf16>
        %expanded = tensor.expand_shape %mapped [[0, 1], [2]]
            output_shape [2, 1024, 64]
            : tensor<2048x64xf16> into tensor<2x1024x64xf16>
        wafer.tile.yield %expanded : tensor<2x1024x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 0u);
  EXPECT_EQ(countOps<mlir::memref::CollapseShapeOp>(module->getOperation()),
            1u);
  EXPECT_EQ(countOps<mlir::memref::ExpandShapeOp>(module->getOperation()), 1u);
  EXPECT_EQ(result.statistics.redundantPublicationCopies, 0u);
}

TEST_F(LayoutOptimizationTest,
       CompatibleOuterReshapeCarriesCxDirectlyIntoContraction) {
  constexpr llvm::StringLiteral text = R"mlir(
#id3 = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1025x128xf16>,
                     %rhs: tensor<128x64xf16>) {
      %result = wafer.tile.region(
          %input, %rhs : tensor<2x1025x128xf16>, tensor<128x64xf16>)
          -> (tensor<2050x64xf16>) {
      ^bb0(%local_input: tensor<2x1025x128xf16>,
           %local_rhs: tensor<128x64xf16>):
        %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %producer = linalg.generic {
            indexing_maps = [#id3, #id3],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local_input : tensor<2x1025x128xf16>)
            outs(%producer_empty : tensor<2x1025x128xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>
        %collapsed = tensor.collapse_shape %producer [[0, 1], [2]] :
            tensor<2x1025x128xf16> into tensor<2050x128xf16>
        %output = tensor.empty() : tensor<2050x64xf16>
        %matmul = linalg.matmul
            ins(%collapsed, %local_rhs :
                tensor<2050x128xf16>, tensor<128x64xf16>)
            outs(%output : tensor<2050x64xf16>) -> tensor<2050x64xf16>
        wafer.tile.yield %matmul : tensor<2050x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 0u);
  mlir::memref::CollapseShapeOp collapse;
  module->walk(
      [&](mlir::memref::CollapseShapeOp operation) { collapse = operation; });
  ASSERT_TRUE(collapse);
  MemoryAttr sourceMemory = getWaferMemoryAttr(
      mlir::cast<mlir::MemRefType>(collapse.getSrc().getType()));
  MemoryAttr resultMemory = getWaferMemoryAttr(collapse.getResultType());
  ASSERT_TRUE(sourceMemory);
  ASSERT_TRUE(resultMemory);
  EXPECT_EQ(sourceMemory.getLayout(), MemLayout::Cx);
  EXPECT_EQ(resultMemory.getLayout(), MemLayout::Cx);
}

TEST_F(LayoutOptimizationTest,
       ChannelChangingReshapeRequiresOneActualCxMaterialization) {
  constexpr llvm::StringLiteral text = R"mlir(
#id3 = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1025x128xf16>,
                     %rhs: tensor<131200x64xf16>) {
      %result = wafer.tile.region(
          %input, %rhs : tensor<2x1025x128xf16>, tensor<131200x64xf16>)
          -> (tensor<2x64xf16>) {
      ^bb0(%local_input: tensor<2x1025x128xf16>,
           %local_rhs: tensor<131200x64xf16>):
        %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
        %producer = linalg.generic {
            indexing_maps = [#id3, #id3],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local_input : tensor<2x1025x128xf16>)
            outs(%producer_empty : tensor<2x1025x128xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x128xf16>
        %collapsed = tensor.collapse_shape %producer [[0], [1, 2]] :
            tensor<2x1025x128xf16> into tensor<2x131200xf16>
        %output = tensor.empty() : tensor<2x64xf16>
        %matmul = linalg.matmul
            ins(%collapsed, %local_rhs :
                tensor<2x131200xf16>, tensor<131200x64xf16>)
            outs(%output : tensor<2x64xf16>) -> tensor<2x64xf16>
        wafer.tile.yield %matmul : tensor<2x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.layoutMaterializationsAfter, 1u);
  EXPECT_EQ(countOps<LayoutMaterializeOp>(module->getOperation()), 1u);
  mlir::memref::CollapseShapeOp collapse;
  module->walk(
      [&](mlir::memref::CollapseShapeOp operation) { collapse = operation; });
  ASSERT_TRUE(collapse);
  MemoryAttr sourceMemory = getWaferMemoryAttr(
      mlir::cast<mlir::MemRefType>(collapse.getSrc().getType()));
  MemoryAttr resultMemory = getWaferMemoryAttr(collapse.getResultType());
  ASSERT_TRUE(sourceMemory);
  ASSERT_TRUE(resultMemory);
  EXPECT_NE(sourceMemory.getLayout(), MemLayout::Cx);
  EXPECT_NE(resultMemory.getLayout(), MemLayout::Cx);
}

TEST_F(LayoutOptimizationTest,
       CalledTensorHelperUsesSPMWhileUncalledEntryUsesDDR) {
  constexpr llvm::StringLiteral text = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func private @helper(%input: tensor<2x1024x64xf16>)
        -> tensor<2x1024x64xf16> {
      %empty = tensor.empty() : tensor<2x1024x64xf16>
      %mapped = linalg.generic {
          indexing_maps = [#id, #id],
          iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%input : tensor<2x1024x64xf16>)
          outs(%empty : tensor<2x1024x64xf16>) {
        ^bb0(%value: f16, %old: f16):
          %next = arith.addf %value, %value : f16
          linalg.yield %next : f16
      } -> tensor<2x1024x64xf16>
      return %mapped : tensor<2x1024x64xf16>
    }
    func.func @entry(%input: tensor<2x1024x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x1024x64xf16>)
          -> (tensor<2x1024x64xf16>) {
      ^bb0(%local: tensor<2x1024x64xf16>):
        %called = func.call @helper(%local)
            : (tensor<2x1024x64xf16>) -> tensor<2x1024x64xf16>
        wafer.tile.yield %called : tensor<2x1024x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  mlir::func::FuncOp helper;
  mlir::func::FuncOp entry;
  module->walk([&](mlir::func::FuncOp function) {
    if (function.getSymName() == "helper")
      helper = function;
    if (function.getSymName() == "entry")
      entry = function;
  });
  ASSERT_TRUE(helper && entry);
  auto helperArg =
      mlir::dyn_cast<mlir::MemRefType>(helper.getArgument(0).getType());
  auto entryArg =
      mlir::dyn_cast<mlir::MemRefType>(entry.getArgument(0).getType());
  ASSERT_TRUE(helperArg && entryArg);
  EXPECT_TRUE(isWaferSPMMemRefType(helperArg));
  EXPECT_TRUE(isWaferDDRMemRefType(entryArg));
}

TEST_F(LayoutOptimizationTest,
       MultipleObservableResultsReceiveDistinctActualDestinations) {
  constexpr llvm::StringLiteral text = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1025x64xf16>) {
      %first, %second = wafer.tile.region(
          %input : tensor<2x1025x64xf16>)
          -> (tensor<2x1025x64xf16>, tensor<2x1025x64xf16>) {
      ^bb0(%local: tensor<2x1025x64xf16>):
        %empty0 = tensor.empty() : tensor<2x1025x64xf16>
        %mapped0 = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<2x1025x64xf16>)
            outs(%empty0 : tensor<2x1025x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x64xf16>
        %empty1 = tensor.empty() : tensor<2x1025x64xf16>
        %mapped1 = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<2x1025x64xf16>)
            outs(%empty1 : tensor<2x1025x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.mulf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<2x1025x64xf16>
        wafer.tile.yield %mapped0, %mapped1
            : tensor<2x1025x64xf16>, tensor<2x1025x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  TileRegionOp region;
  module->walk([&](TileRegionOp current) { region = current; });
  ASSERT_TRUE(region);
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  relations.structuralOutputs.push_back({1, region.getResult(1)});
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.outputDestinations, 2u);
  EXPECT_EQ(result.statistics.outputSubviews, 2u);
  EXPECT_EQ(result.statistics.redundantPublicationCopies, 0u);
  ASSERT_EQ(relations.structuralOutputs.size(), 2u);
  EXPECT_NE(relations.structuralOutputs[0].endpoint,
            relations.structuralOutputs[1].endpoint);
  for (const auto &output : relations.structuralOutputs)
    EXPECT_TRUE(isWaferDDRMemRefType(output.endpoint.getType()));
}

TEST_F(LayoutOptimizationTest,
       CrossTileSourcePublishesTheActualPieceInsteadOfAFullTensorShell) {
  constexpr llvm::StringLiteral text = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @source(%input: tensor<2x1025x64xf16>) {
      %published = wafer.tile.region(
          %input : tensor<2x1025x64xf16>)
          -> (tensor<2x1025x64xf16>) {
      ^bb0(%local: tensor<2x1025x64xf16>):
        %piece = tensor.extract_slice %local[1, 0, 0] [1, 1025, 64]
            [1, 1, 1] : tensor<2x1025x64xf16> to tensor<1x1025x64xf16>
        %empty = tensor.empty() : tensor<2x1025x64xf16>
        %full = tensor.insert_slice %piece into %empty[1, 0, 0]
            [1, 1025, 64] [1, 1, 1]
            : tensor<1x1025x64xf16> into tensor<2x1025x64xf16>
        wafer.tile.yield %full : tensor<2x1025x64xf16>
      }
      return
    }
  }
  wafer.tile.module card_id = 0 tile_id = 1 {
    func.func @destination(%input: tensor<2x1025x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x1025x64xf16>)
          -> (tensor<2x1025x64xf16>) {
      ^bb0(%external: tensor<2x1025x64xf16>):
        %piece = tensor.extract_slice %external[1, 0, 0] [1, 1025, 64]
            [1, 1, 1] : tensor<2x1025x64xf16> to tensor<1x1025x64xf16>
        %mapped_empty = tensor.empty() : tensor<1x1025x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%piece : tensor<1x1025x64xf16>)
            outs(%mapped_empty : tensor<1x1025x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x1025x64xf16>
        %empty = tensor.empty() : tensor<2x1025x64xf16>
        %full = tensor.insert_slice %mapped into %empty[1, 0, 0]
            [1, 1025, 64] [1, 1, 1]
            : tensor<1x1025x64xf16> into tensor<2x1025x64xf16>
        wafer.tile.yield %full : tensor<2x1025x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(text);
  ASSERT_TRUE(module);
  llvm::SmallVector<TileRegionOp, 2> regions;
  module->walk([&](TileRegionOp region) { regions.push_back(region); });
  ASSERT_EQ(regions.size(), 2u);
  StructuredMaterializationRelations relations;
  relations.boundaryRelations.push_back(
      {regions[0].getResult(0), regions[1].getBody().getArgument(0)});
  relations.structuralOutputs.push_back({0, regions[1].getResult(0)});
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_EQ(result.statistics.boundarySourceViewsElided, 1u);
  ASSERT_EQ(relations.boundaryRelations.size(), 1u);
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
      relations.boundaryRelations.front().sourceEndpoint.getType());
  ASSERT_TRUE(sourceType);
  EXPECT_EQ(sourceType.getShape(), llvm::ArrayRef<int64_t>({1, 1025, 64}));
  EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(module->getOperation()), 0u);
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
}

TEST_F(LayoutOptimizationTest, ZeroBudgetAndNonPieceOutputFailBeforeMutation) {
  constexpr llvm::StringLiteral malformed = R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1025x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x1025x64xf16>)
          -> (tensor<2x1025x64xf16>) {
      ^bb0(%local: tensor<2x1025x64xf16>):
        %piece = tensor.extract_slice %local[0, 0, 0] [1, 1025, 64]
            [1, 1, 1] : tensor<2x1025x64xf16> to tensor<1x1025x64xf16>
        %updated = tensor.insert_slice %piece into %local[1, 0, 0]
            [1, 1025, 64] [1, 1, 1]
            : tensor<1x1025x64xf16> into tensor<2x1025x64xf16>
        wafer.tile.yield %updated : tensor<2x1025x64xf16>
      }
      return
    }
  }
}
)mlir";
  auto module = parse(malformed);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  LayoutOptimizationResult noBudget =
      resolveCurrentLayoutsAndBufferize(*module, relations, /*workLimit=*/0);
  EXPECT_EQ(noBudget.status, ExactPBQPStatus::Indeterminate);
  LayoutOptimizationResult unsupported =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  EXPECT_EQ(unsupported.status, ExactPBQPStatus::NoSolution);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

} // namespace
