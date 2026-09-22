//===- LayoutOptimizationTest.cpp --------------------------------------===//

#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Analysis/Tile/TransferRealizability.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Tile/BooleanReduction.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LoopSubsetState.h"
#include "Wafer/Transforms/Tile/ScalarExecution.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"
#include "Wafer/Transforms/Tile/TensorInitialization.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <array>
#include <chrono>
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

  static std::string
  makeLargeConnectedLayoutSource(int64_t extent, unsigned diamondCount,
                                 bool mixedOperators = false) {
    const std::string type =
        "tensor<" + std::to_string(extent) + "x128x128xf16>";
    const std::string expandedType =
        "tensor<" + std::to_string(extent) + "x1x128x128xf16>";
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "module {\n"
           << "  wafer.tile.module card_id = 0 tile_id = 0 {\n"
           << "    func.func @entry(%input: " << type << ", %pre_rhs0: " << type
           << ", %pre_rhs1: " << type << ", %post_rhs: " << type;
    for (unsigned index = 0; index < diamondCount; ++index)
      stream << ", %left_rhs" << index << ": " << type << ", %right_rhs"
             << index << ": " << type;
    if (mixedOperators)
      stream << ", %matrix_rhs: tensor<128x128xf16>, "
                "%channel_rhs: tensor<16384x128xf16>";
    stream << ") {\n"
           << "      %result = wafer.tile.region(%input, %pre_rhs0, "
              "%pre_rhs1, %post_rhs";
    for (unsigned index = 0; index < diamondCount; ++index)
      stream << ", %left_rhs" << index << ", %right_rhs" << index;
    if (mixedOperators)
      stream << ", %matrix_rhs, %channel_rhs";
    stream << " : " << type << ", " << type << ", " << type << ", " << type;
    for (unsigned index = 0; index < diamondCount; ++index)
      stream << ", " << type << ", " << type;
    if (mixedOperators)
      stream << ", tensor<128x128xf16>, tensor<16384x128xf16>";
    stream << ") -> (" << type << ") {\n"
           << "      ^bb0(%local_input: " << type
           << ", %local_pre_rhs0: " << type << ", %local_pre_rhs1: " << type
           << ", %local_post_rhs: " << type;
    for (unsigned index = 0; index < diamondCount; ++index)
      stream << ", %local_left_rhs" << index << ": " << type
             << ", %local_right_rhs" << index << ": " << type;
    if (mixedOperators)
      stream << ", %local_matrix_rhs: tensor<128x128xf16>, "
                "%local_channel_rhs: tensor<16384x128xf16>";
    stream << "):\n"
           << "        %view = tensor.extract_slice %local_input[0, 0, 0] ["
           << extent << ", 128, 128] [1, 1, 1] : " << type << " to " << type
           << "\n";

    auto emitMatmul = [&](llvm::StringRef name, llvm::StringRef lhs,
                          llvm::StringRef rhs) {
      stream << "        %empty_" << name << " = tensor.empty() : " << type
             << "\n"
             << "        %" << name << " = linalg.batch_matmul ins(" << lhs
             << ", " << rhs << " : " << type << ", " << type << ") outs(%empty_"
             << name << " : " << type << ") -> " << type << "\n";
    };
    emitMatmul("pre0", "%view", "%local_pre_rhs0");
    emitMatmul("pre1", "%view", "%local_pre_rhs1");
    stream << "        %view_buffer = bufferization.to_memref %view : memref<"
           << extent
           << "x128x128xf16, #wafer.memory<spm, tensor>>\n"
              "        %c0 = arith.constant 0 : index\n"
              "        %zero = arith.constant 0.000000e+00 : f16\n"
              "        memref.store %zero, %view_buffer[%c0, %c0, %c0] : "
              "memref<"
           << extent << "x128x128xf16, #wafer.memory<spm, tensor>>\n";
    emitMatmul("post", "%view", "%local_post_rhs");
    if (mixedOperators) {
      stream << "        %mixed_fill_empty = tensor.empty() : " << type
             << "\n"
                "        %mixed_fill = linalg.fill ins(%zero : f16) "
                "outs(%mixed_fill_empty : "
             << type << ") -> " << type << "\n";
      emitMatmul("mixed_fill_left", "%mixed_fill", "%local_pre_rhs0");
      emitMatmul("mixed_fill_right", "%mixed_fill", "%local_pre_rhs1");
    }

    std::string current = "%post";
    for (unsigned index = 0; index < diamondCount; ++index) {
      const std::string left = "left" + std::to_string(index);
      const std::string right = "right" + std::to_string(index);
      const std::string join = "join" + std::to_string(index);
      emitMatmul(left, current, "%local_left_rhs" + std::to_string(index));
      emitMatmul(right, current, "%local_right_rhs" + std::to_string(index));
      emitMatmul(join, "%" + left, "%" + right);
      current = "%" + join;
      if (mixedOperators && (index + 1) % 4 == 0) {
        const std::string empty = "elementwise_empty" + std::to_string(index);
        const std::string elementwise = "elementwise" + std::to_string(index);
        stream << "        %" << empty << " = tensor.empty() : " << type << "\n"
               << "        %" << elementwise << " = linalg.generic {\n"
               << "            indexing_maps = ["
                  "affine_map<(b, m, n) -> (b, m, n)>, "
                  "affine_map<(b, m, n) -> (b, m, n)>],\n"
               << "            iterator_types = [\"parallel\", "
                  "\"parallel\", \"parallel\"]}\n"
               << "            ins(" << current << " : " << type << ") outs(%"
               << empty << " : " << type << ") {\n"
               << "          ^bb1(%value: f16, %old: f16):\n"
               << "            %next = arith.addf %value, %value : f16\n"
               << "            linalg.yield %next : f16\n"
               << "        } -> " << type << "\n";
        current = "%" + elementwise;
      }
      if (mixedOperators && (index + 1) % 8 == 0) {
        const std::string empty = "transpose_empty" + std::to_string(index);
        const std::string transpose = "transpose" + std::to_string(index);
        stream << "        %" << empty << " = tensor.empty() : " << type << "\n"
               << "        %" << transpose << " = linalg.transpose ins("
               << current << " : " << type << ") outs(%" << empty << " : "
               << type << ") permutation = [0, 2, 1]\n";
        current = "%" + transpose;
      }
      if ((index + 1) % 8 != 0)
        continue;
      const std::string expanded = "expanded" + std::to_string(index);
      const std::string collapsed = "collapsed" + std::to_string(index);
      stream << "        %" << expanded << " = tensor.expand_shape " << current
             << " [[0], [1, 2], [3]] output_shape [" << extent
             << ", 1, 128, 128] : " << type << " into " << expandedType << "\n"
             << "        %" << collapsed << " = tensor.collapse_shape %"
             << expanded << " [[0], [1, 2], [3]] : " << expandedType << " into "
             << type << "\n";
      current = "%" + collapsed;
      if (!mixedOperators || index != 7)
        continue;

      const std::string scoreType =
          "tensor<" + std::to_string(extent) + "x128x128xf32>";
      const std::string rowType =
          "tensor<" + std::to_string(extent) + "x128xf32>";
      stream
          << "        %attn_zero = arith.constant 0.000000e+00 : f32\n"
             "        %attn_one = arith.constant 1.000000e+00 : f32\n"
             "        %attn_neg_inf = arith.constant 0xFF800000 : f32\n"
          << "        %attn_score_empty = tensor.empty() : " << scoreType
          << "\n"
          << "        %attn_score_zero = linalg.fill ins(%attn_zero : f32) "
             "outs(%attn_score_empty : "
          << scoreType << ") -> " << scoreType << "\n"
          << "        %attn_score = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m, k1, k2) -> (b, m, k1)>, "
             "affine_map<(b, m, k1, k2) -> (b, k2, k1)>, "
             "affine_map<(b, m, k1, k2) -> (b, m, k2)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\", "
             "\"reduction\", \"parallel\"]}\n"
          << "            ins(" << current << ", %local_left_rhs7 : " << type
          << ", " << type << ") outs(%attn_score_zero : " << scoreType
          << ") {\n"
          << "          ^bb1(%query: f16, %key: f16, %acc: f32):\n"
          << "            %query_f32 = arith.extf %query : f16 to f32\n"
          << "            %key_f32 = arith.extf %key : f16 to f32\n"
          << "            %qk = arith.mulf %query_f32, %key_f32 : f32\n"
          << "            %qk_acc = arith.addf %qk, %acc : f32\n"
          << "            linalg.yield %qk_acc : f32\n"
          << "        } -> " << scoreType << "\n"
          << "        %attn_scaled = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m, k2) -> (b, m, k2)>, "
             "affine_map<(b, m, k2) -> ()>, "
             "affine_map<(b, m, k2) -> (b, m, k2)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\", "
             "\"parallel\"]}\n"
          << "            ins(%attn_score, %attn_one : " << scoreType
          << ", f32) outs(%attn_score : " << scoreType << ") {\n"
          << "          ^bb1(%score: f32, %scale: f32, %old: f32):\n"
          << "            %scaled = arith.mulf %score, %scale : f32\n"
          << "            linalg.yield %scaled : f32\n"
          << "        } -> " << scoreType << "\n"
          << "        %attn_max_empty = tensor.empty() : " << rowType << "\n"
          << "        %attn_old_max = linalg.fill ins(%attn_neg_inf : f32) "
             "outs(%attn_max_empty : "
          << rowType << ") -> " << rowType << "\n"
          << "        %attn_new_max = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m, k2) -> (b, m, k2)>, "
             "affine_map<(b, m, k2) -> (b, m)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\", "
             "\"reduction\"]}\n"
          << "            ins(%attn_scaled : " << scoreType
          << ") outs(%attn_old_max : " << rowType << ") {\n"
          << "          ^bb1(%score: f32, %old: f32):\n"
          << "            %maximum = arith.maximumf %score, %old : f32\n"
          << "            linalg.yield %maximum : f32\n"
          << "        } -> " << rowType << "\n"
          << "        %attn_norm = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m) -> (b, m)>, "
             "affine_map<(b, m) -> (b, m)>, "
             "affine_map<(b, m) -> (b, m)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\"]}\n"
          << "            ins(%attn_old_max, %attn_new_max : " << rowType
          << ", " << rowType << ") outs(%attn_old_max : " << rowType << ") {\n"
          << "          ^bb1(%old_max: f32, %new_max: f32, %old: f32):\n"
          << "            %delta = arith.subf %old_max, %new_max : f32\n"
          << "            %factor = math.exp %delta : f32\n"
          << "            linalg.yield %factor : f32\n"
          << "        } -> " << rowType << "\n"
          << "        %attn_sum_empty = tensor.empty() : " << rowType << "\n"
          << "        %attn_old_sum = linalg.fill ins(%attn_zero : f32) "
             "outs(%attn_sum_empty : "
          << rowType << ") -> " << rowType << "\n"
          << "        %attn_scaled_sum = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m) -> (b, m)>, "
             "affine_map<(b, m) -> (b, m)>, "
             "affine_map<(b, m) -> (b, m)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\"]}\n"
          << "            ins(%attn_old_sum, %attn_norm : " << rowType << ", "
          << rowType << ") outs(%attn_old_sum : " << rowType << ") {\n"
          << "          ^bb1(%sum: f32, %factor: f32, %old: f32):\n"
          << "            %scaled_sum = arith.mulf %sum, %factor : f32\n"
          << "            linalg.yield %scaled_sum : f32\n"
          << "        } -> " << rowType << "\n"
          << "        %attn_probability = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m, k2) -> (b, m, k2)>, "
             "affine_map<(b, m, k2) -> (b, m)>, "
             "affine_map<(b, m, k2) -> (b, m, k2)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\", "
             "\"parallel\"]}\n"
          << "            ins(%attn_scaled, %attn_new_max : " << scoreType
          << ", " << rowType << ") outs(%attn_scaled : " << scoreType << ") {\n"
          << "          ^bb1(%score: f32, %maximum: f32, %old: f32):\n"
          << "            %delta = arith.subf %score, %maximum : f32\n"
          << "            %probability = math.exp %delta : f32\n"
          << "            linalg.yield %probability : f32\n"
          << "        } -> " << scoreType << "\n"
          << "        %attn_new_sum = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m, k2) -> (b, m, k2)>, "
             "affine_map<(b, m, k2) -> (b, m)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\", "
             "\"reduction\"]}\n"
          << "            ins(%attn_probability : " << scoreType
          << ") outs(%attn_scaled_sum : " << rowType << ") {\n"
          << "          ^bb1(%probability: f32, %sum: f32):\n"
          << "            %next_sum = arith.addf %probability, %sum : f32\n"
          << "            linalg.yield %next_sum : f32\n"
          << "        } -> " << rowType << "\n"
          << "        %attn_acc_empty = tensor.empty() : " << type << "\n"
          << "        %attn_old_acc = linalg.fill ins(%zero : f16) "
             "outs(%attn_acc_empty : "
          << type << ") -> " << type << "\n"
          << "        %attn_scaled_acc = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m, n) -> (b, m, n)>, "
             "affine_map<(b, m, n) -> (b, m)>, "
             "affine_map<(b, m, n) -> (b, m, n)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\", "
             "\"parallel\"]}\n"
          << "            ins(%attn_old_acc, %attn_norm : " << type << ", "
          << rowType << ") outs(%attn_old_acc : " << type << ") {\n"
          << "          ^bb1(%acc: f16, %factor: f32, %old: f16):\n"
          << "            %factor_f16 = arith.truncf %factor : f32 to f16\n"
          << "            %scaled_acc = arith.mulf %acc, %factor_f16 : f16\n"
          << "            linalg.yield %scaled_acc : f16\n"
          << "        } -> " << type << "\n"
          << "        %attn_pv = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m, k2, n) -> (b, m, k2)>, "
             "affine_map<(b, m, k2, n) -> (b, k2, n)>, "
             "affine_map<(b, m, k2, n) -> (b, m, n)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\", "
             "\"reduction\", \"parallel\"]}\n"
          << "            ins(%attn_probability, %local_right_rhs7 : "
          << scoreType << ", " << type << ") outs(%attn_scaled_acc : " << type
          << ") {\n"
          << "          ^bb1(%probability: f32, %value: f16, %acc: f16):\n"
          << "            %probability_f16 = arith.truncf %probability : f32 "
             "to f16\n"
          << "            %weighted = arith.mulf %probability_f16, %value : "
             "f16\n"
          << "            %next_acc = arith.addf %weighted, %acc : f16\n"
          << "            linalg.yield %next_acc : f16\n"
          << "        } -> " << type << "\n"
          << "        %attn_finalized = linalg.generic {\n"
          << "            indexing_maps = ["
             "affine_map<(b, m, n) -> (b, m, n)>, "
             "affine_map<(b, m, n) -> (b, m)>, "
             "affine_map<(b, m, n) -> (b, m, n)>],\n"
          << "            iterator_types = [\"parallel\", \"parallel\", "
             "\"parallel\"]}\n"
          << "            ins(%attn_pv, %attn_new_sum : " << type << ", "
          << rowType << ") outs(%attn_pv : " << type << ") {\n"
          << "          ^bb1(%acc: f16, %sum: f32, %old: f16):\n"
          << "            %acc_f32 = arith.extf %acc : f16 to f32\n"
          << "            %normalized = arith.divf %acc_f32, %sum : f32\n"
          << "            %result = arith.truncf %normalized : f32 to f16\n"
          << "            linalg.yield %result : f16\n"
          << "        } -> " << type << "\n";
      current = "%attn_finalized";
    }
    if (mixedOperators) {
      const std::string matrixType =
          "tensor<" + std::to_string(extent) + "x128xf16>";
      const std::string channelType =
          "tensor<" + std::to_string(extent) + "x16384xf16>";
      stream << "        %mixed_reduce_empty = tensor.empty() : " << matrixType
             << "\n"
                "        %mixed_reduce_init = linalg.fill ins(%zero : f16) "
                "outs(%mixed_reduce_empty : "
             << matrixType << ") -> " << matrixType << "\n"
             << "        %mixed_reduced = linalg.generic {\n"
             << "            indexing_maps = ["
                "affine_map<(b, m, k) -> (b, m, k)>, "
                "affine_map<(b, m, k) -> (b, m)>],\n"
             << "            iterator_types = [\"parallel\", \"parallel\", "
                "\"reduction\"]}\n"
             << "            ins(" << current << " : " << type
             << ") outs(%mixed_reduce_init : " << matrixType << ") {\n"
             << "          ^bb1(%value: f16, %sum: f16):\n"
             << "            %next = arith.addf %value, %sum : f16\n"
             << "            linalg.yield %next : f16\n"
             << "        } -> " << matrixType << "\n"
             << "        %mixed_matrix_empty = tensor.empty() : " << matrixType
             << "\n"
             << "        %mixed_matrix = linalg.matmul "
                "ins(%mixed_reduced, %local_matrix_rhs : "
             << matrixType << ", tensor<128x128xf16>) "
             << "outs(%mixed_matrix_empty : " << matrixType << ") -> "
             << matrixType << "\n"
             << "        %mixed_channel = tensor.collapse_shape %mixed_fill"
             << " [[0], [1, 2]] : " << type << " into " << channelType << "\n"
             << "        %mixed_channel_empty = tensor.empty() : " << matrixType
             << "\n"
             << "        %mixed_channel_matmul = linalg.matmul "
                "ins(%mixed_channel, %local_channel_rhs : "
             << channelType << ", tensor<16384x128xf16>) "
             << "outs(%mixed_channel_empty : " << matrixType << ") -> "
             << matrixType << "\n";
    }
    stream << "        wafer.tile.yield " << current << " : " << type
           << "\n"
              "      }\n"
              "      return\n"
              "    }\n"
              "  }\n"
              "}\n";
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

TEST_F(LayoutOptimizationTest, BooleanReductionsPreserveInitAndReachInstr) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool all : {false, true})
      for (bool initial : {false, true}) {
        SCOPED_TRACE(extent);
        SCOPED_TRACE(all);
        SCOPED_TRACE(initial);
        const std::string shape = "2x" + std::to_string(extent);
        const std::string input = "tensor<" + shape + "x64xf16>";
        const std::string bits = "tensor<" + shape + "x64xi1>";
        const std::string reduced = "tensor<" + shape + "x1xi1>";
        const std::string output = "tensor<" + shape + "x1xf16>";
        std::string text;
        llvm::raw_string_ostream ir(text);
        ir << "#id = affine_map<(b,m,k)->(b,m,k)>\n"
           << "#row = affine_map<(b,m,k)->(b,m,0)>\n"
           << "module { wafer.tile.module card_id = 0 tile_id = 0 { "
           << "func.func @entry(%x: " << input << ") { "
           << "%r = wafer.tile.region(%x : " << input << ") -> (" << output
           << ") { ^bb0(%a: " << input << "): "
           << "%z = arith.constant 0.0 : f16 "
           << "%one = arith.constant 1.0 : f16 "
           << "%init = arith.constant " << (initial ? "true" : "false")
           << " %e = tensor.empty() : " << bits
           << " %p = linalg.generic {indexing_maps = [#id,#id], "
              "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
           << "ins(%a : " << input << ") outs(%e : " << bits << ") { "
           << "^bb1(%v: f16, %old: i1): %c = arith.cmpf ogt, %v, %z : f16 "
           << "linalg.yield %c : i1 } -> " << bits
           << " %re = tensor.empty() : " << reduced
           << " %ri = linalg.fill ins(%init : i1) outs(%re : " << reduced
           << ") -> " << reduced
           << " %rr = linalg.generic {indexing_maps = [#id,#row], "
              "iterator_types = [\"parallel\",\"parallel\",\"reduction\"]} "
           << "ins(%p : " << bits << ") outs(%ri : " << reduced << ") { "
           << "^bb2(%v: i1, %old: i1): %c = arith." << (all ? "andi" : "ori")
           << " %v, %old : i1 "
           << "linalg.yield %c : i1 } -> " << reduced
           << " %oe = tensor.empty() : " << output
           << " %o = linalg.generic {indexing_maps = [#id,#id], "
              "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
           << "ins(%rr : " << reduced << ") outs(%oe : " << output << ") { "
           << "^bb3(%v: i1, %old: f16): %s = arith.select %v, %one, %z : f16 "
           << "linalg.yield %s : f16 } -> " << output
           << " wafer.tile.yield %o : " << output << " } return } } }";
        auto module = parse(text);
        ASSERT_TRUE(module);
        auto relations = outputRelation(*module);
        ASSERT_TRUE(
            mlir::succeeded(lowerBooleanReductions(*module, relations)));
        EXPECT_EQ(countOps<mlir::arith::AndIOp>(*module), 0u);
        EXPECT_EQ(countOps<mlir::arith::OrIOp>(*module), 0u);
        EXPECT_EQ(countOps<mlir::arith::MinimumFOp>(*module), all ? 1u : 0u);
        EXPECT_EQ(countOps<mlir::arith::MaximumFOp>(*module), all ? 0u : 1u);
        unsigned reductions = 0;
        module->walk([&](mlir::linalg::GenericOp op) {
          if (!op.getNumReductionLoops())
            return;
          ++reductions;
          EXPECT_EQ(op.getIndexingMapsArray().back().getResult(2),
                    mlir::getAffineConstantExpr(0, context.get()));
          auto fill =
              op.getDpsInits().front().getDefiningOp<mlir::linalg::FillOp>();
          ASSERT_TRUE(fill);
          auto constant = fill.getDpsInputs()
                              .front()
                              .getDefiningOp<mlir::arith::ConstantOp>();
          ASSERT_TRUE(constant);
          EXPECT_EQ(mlir::cast<mlir::FloatAttr>(constant.getValue())
                        .getValueAsDouble(),
                    initial ? 1.0 : 0.0);
        });
        EXPECT_EQ(reductions, 1u);
        auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        auto lowered = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
        auto movement = materializeTileBoundaryMovement(*module, relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        std::string detail;
        auto standalone =
            createStandaloneTileModules(std::move(module), &detail, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
        ASSERT_EQ(standalone->size(), 1u);
        auto &tile = standalone->front();
        TileRegionToInstrLoweringSession session(*context);
        llvm::SmallVector<TileRegionOp> regions;
        tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
        for (auto region : regions)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*tile.module, session)));
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
        EXPECT_GT(countOps<InstrBit2FpOp>(*tile.module), 0u);
        TileMemoryPlanningFailure failure;
        auto planned = planTileMemory(std::move(tile.module), &failure);
        ASSERT_TRUE(mlir::succeeded(planned));
        EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
      }
}

TEST_F(LayoutOptimizationTest, BooleanReductionLeavesOtherCombinersUnchanged) {
  for (unsigned form = 0; form < 4; ++form) {
    SCOPED_TRACE(form);
    const std::string element = form == 1 ? "i32" : "i1";
    const std::string extent = form == 2 ? "?" : "1025";
    const std::string input = "tensor<2x" + extent + "x64x" + element + ">";
    const std::string output = "tensor<2x" + extent + "x1x" + element + ">";
    const std::string combine = form == 0 ? "xori" : "andi";
    std::string text;
    llvm::raw_string_ostream ir(text);
    ir << "#id = affine_map<(b,m,k)->(b,m,k)>\n"
       << "#row = affine_map<(b,m,k)->(b,m,0)>\n"
       << "module { func.func @entry(%x: " << input << ", %init: " << output
       << ") { %r = wafer.tile.region(%x, %init : " << input << ", " << output
       << ") -> (" << output << ") { ^bb0(%a: " << input << ", %i: " << output
       << "): "
       << "%rr = linalg.generic {indexing_maps = [#id,#row], "
          "iterator_types = [\"parallel\",\"parallel\",\"reduction\"]} "
       << "ins(%a : " << input << ") outs(%i : " << output << ") { "
       << "^bb1(%v: " << element << ", %old: " << element << "): "
       << "%c = arith." << combine << " %v, %old : " << element;
    if (form == 3)
      ir << " %d = arith.ori %c, %v : " << element;
    ir << " linalg.yield %" << (form == 3 ? "d" : "c") << " : " << element
       << " } -> " << output << " wafer.tile.yield %rr : " << output
       << " } return } }";
    auto module = parse(text);
    ASSERT_TRUE(module);
    auto before = countOps<mlir::linalg::GenericOp>(*module);
    auto relations = outputRelation(*module);
    ASSERT_TRUE(mlir::succeeded(lowerBooleanReductions(*module, relations)));
    EXPECT_EQ(countOps<mlir::linalg::GenericOp>(*module), before);
    EXPECT_EQ(countOps<mlir::arith::SelectOp>(*module), 0u);
    EXPECT_EQ(countOps<mlir::arith::MinimumFOp>(*module), 0u);
    EXPECT_EQ(countOps<mlir::arith::MaximumFOp>(*module), 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest, ConstantViewsBecomeExplicitSelectedSPMReads) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool predicate : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(predicate);
      std::string constantElement = predicate ? "i1" : "f32";
      std::string full = "tensor<2x" + std::to_string(extent + 32) + "x16x" +
                         constantElement + ">";
      std::string window =
          "tensor<2x" + std::to_string(extent) + "x16x" + constantElement + ">";
      std::string output = "tensor<2x" + std::to_string(extent) + "x16xf32>";
      std::string text =
          "module { wafer.tile.module card_id = 0 tile_id = 0 { "
          "func.func @entry(%input: " +
          output + ") -> " + output +
          " { %result = wafer.tile.region(%input : " + output + ") -> (" +
          output + ") { ^bb0(%local: " + output +
          "): %constant = arith.constant dense<" +
          (predicate ? "false" : "0.0") + "> : " + full +
          " %view = tensor.extract_slice %constant[0, 7, 0] [2, " +
          std::to_string(extent) + ", 16] [1, 1, 1] : " + full + " to " +
          window + " %empty = tensor.empty() : " + output +
          " %mapped = linalg.generic {indexing_maps = "
          "[affine_map<(b,m,n)->(b,m,n)>, "
          "affine_map<(b,m,n)->(b,m,n)>, affine_map<(b,m,n)->(b,m,n)>], "
          "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
          "ins(%view, %local : " +
          window + ", " + output + ") outs(%empty : " + output +
          ") { ^bb0(%c: " + constantElement + ", %v: f32, %old: f32): " +
          (predicate ? "%zero = arith.constant 0.0 : f32 %x = arith.select %c, "
                       "%v, %zero : f32 "
                     : "%x = arith.addf %c, %v : f32 ") +
          "linalg.yield %x : f32 } -> " + output +
          " wafer.tile.yield %mapped : " + output +
          " } return %result : " + output + " } } }";
      auto module = parse(text);
      ASSERT_TRUE(module);
      module->walk([&](mlir::arith::ConstantOp constant) {
        auto type = mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
        if (!type)
          return;
        llvm::SmallVector<mlir::Attribute> elements;
        for (int64_t index = 0; index < type.getNumElements(); ++index) {
          bool bit = (index / 16) % 31 < index % 16;
          elements.push_back(
              predicate ? mlir::Attribute(mlir::IntegerAttr::get(
                              type.getElementType(), bit))
                        : mlir::Attribute(mlir::FloatAttr::get(
                              type.getElementType(), bit ? 0.25 : -0.5)));
        }
        constant.setValueAttr(mlir::DenseElementsAttr::get(type, elements));
      });
      auto relations = outputRelation(*module);
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      unsigned copiedWindows = 0;
      module->walk([&](mlir::memref::CopyOp copy) {
        mlir::Value source = copy.getSource();
        while (auto view = source.getDefiningOp<mlir::memref::SubViewOp>())
          source = view.getSource();
        if (!source.getDefiningOp<mlir::memref::GetGlobalOp>())
          return;
        auto type = mlir::cast<mlir::MemRefType>(copy.getSource().getType());
        EXPECT_EQ(llvm::to_vector(type.getShape()),
                  (llvm::SmallVector<int64_t>{2, extent, 16}));
        EXPECT_TRUE(isWaferDDRMemRefType(type));
        EXPECT_TRUE(isWaferSPMMemRefType(copy.getTarget().getType()));
        ++copiedWindows;
      });
      EXPECT_EQ(copiedWindows, 1u);
      module->walk([&](mlir::linalg::LinalgOp op) {
        for (auto input : op.getDpsInputs())
          EXPECT_TRUE(isWaferSPMMemRefType(input.getType()));
      });
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}

TEST_F(LayoutOptimizationTest, FunctionBoundarySpaceIsQueriedOncePerFunction) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    const std::string type = "tensor<1x" + std::to_string(extent) + "x64xf16>";
    std::string text;
    llvm::raw_string_ostream ir(text);
    ir << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
       << "func.func private @helper(%x: " << type << ") -> " << type
       << " { return %x : " << type << " }\n"
       << "func.func @entry(";
    for (unsigned i = 0; i < 1024; ++i)
      ir << (i ? ", " : "") << "%arg" << i << ": " << type;
    ir << ") -> " << type << " {\n"
       << "%result = wafer.tile.region(%arg0 : " << type << ") -> (" << type
       << ") { ^bb0(%x: " << type << "):\n"
       << "%forwarded = func.call @helper(%x) : (" << type << ") -> " << type
       << "\n"
       << "%empty = tensor.empty() : " << type << "\n"
       << "%value = linalg.add ins(%x, %forwarded : " << type << ", " << type
       << ") outs(%empty : " << type << ") -> " << type << "\n"
       << "wafer.tile.yield %value : " << type
       << " }\nreturn %result : " << type << "\n} } }";
    auto module = parse(text);
    ASSERT_TRUE(module);
    TileRegionOp region;
    module->walk([&](TileRegionOp op) { region = op; });
    StructuredMaterializationRelations relations;
    relations.structuralOutputs = {{0, region.getResult(0)}};
    auto result = resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.functionBoundaryQueries, 2u);
    EXPECT_EQ(result.statistics.outputDestinations, 1u);
    unsigned functions = 0;
    module->walk([&](mlir::func::FuncOp function) {
      ++functions;
      auto expected =
          function.isPrivate() ? MemorySpace::SPM : MemorySpace::DDR;
      EXPECT_EQ(function.getNumArguments(), function.isPrivate() ? 1u : 1025u);
      for (auto type : function.getArgumentTypes()) {
        auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
        ASSERT_TRUE(memref);
        auto memory = getWaferMemoryAttr(memref);
        ASSERT_TRUE(memory);
        EXPECT_EQ(memory.getSpace(), expected);
      }
      for (auto type : function.getResultTypes()) {
        auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
        ASSERT_TRUE(memref);
        auto memory = getWaferMemoryAttr(memref);
        ASSERT_TRUE(memory);
        EXPECT_EQ(memory.getSpace(), expected);
      }
    });
    EXPECT_EQ(functions, 2u);
    EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*module)));
    EXPECT_TRUE(mlir::succeeded(
        checkStructuredBufferRelationsCurrent(*module, relations)));
  }
}

TEST_F(LayoutOptimizationTest,
       SegmentedLoopsCarryOnlyTheirUpdatedOutputSubset) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool nested : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(nested);
      const std::string type =
          "tensor<1x" + std::to_string(extent) + "x64xf16>";
      constexpr unsigned segments = 32;
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
          << "func.func @entry(%input: " << type << ") {\n"
          << "%result = wafer.tile.region(%input : " << type << ") -> (" << type
          << ") { ^bb0(%local: " << type << "):\n"
          << "%c0 = arith.constant 0 : index\n"
          << "%c1 = arith.constant 1 : index\n"
          << "%c3 = arith.constant 3 : index\n"
          << "%c32 = arith.constant 32 : index\n"
          << "%end = arith.constant " << extent / 32 * 32 << " : index\n"
          << "%outer = scf.for %i = %c0 to %end step %c32 "
          << "iter_args(%state = %local) -> " << type << " {\n";
      for (unsigned i = 0; i < segments; ++i) {
        auto initial = i ? "%loop" + std::to_string(i - 1) : "%state";
        out << "%loop" << i
            << " = scf.for %j = %c0 to %c3 step %c1 iter_args(%tile = "
            << initial << ") -> " << type << " {\n";
        if (nested)
          out << "%nested = scf.for %k = %c0 to %c3 step %c1 "
              << "iter_args(%inner = %tile) -> " << type << " {\n";
        std::string state = nested ? "%inner" : "%tile";
        out << "%slice = tensor.extract_slice " << state
            << "[0, %i, 0] [1, 32, 64] "
            << "[1, 1, 1] : " << type << " to tensor<1x32x64xf16>\n"
            << "%sum = linalg.add ins(%slice, %slice : tensor<1x32x64xf16>, "
            << "tensor<1x32x64xf16>) outs(%slice : tensor<1x32x64xf16>) "
            << "-> tensor<1x32x64xf16>\n"
            << "%updated = tensor.insert_slice %sum into " << state
            << "[0, %i, 0] "
            << "[1, 32, 64] [1, 1, 1] : tensor<1x32x64xf16> into " << type
            << "\nscf.yield %updated : " << type << "\n}\n";
        if (nested)
          out << "scf.yield %nested : " << type << "\n}\n";
      }
      out << "scf.yield %loop" << segments - 1 << " : " << type << "\n}\n"
          << "wafer.tile.yield %outer : " << type << "\n}\nreturn\n}}}\n";
      auto module = parse(text);
      ASSERT_TRUE(module);
      auto relations = outputRelation(*module);
      ASSERT_TRUE(
          mlir::succeeded(normalizeLoopSubsetState(*module, relations)));
      unsigned localStates = 0, fullStates = 0;
      module->walk([&](mlir::scf::ForOp loop) {
        ASSERT_EQ(loop.getNumRegionIterArgs(), 1u);
        auto state = mlir::cast<mlir::RankedTensorType>(
            loop.getRegionIterArgs().front().getType());
        if (loop->getParentOfType<mlir::scf::ForOp>()) {
          ++localStates;
          EXPECT_EQ(state.getShape(), (llvm::ArrayRef<int64_t>{1, 32, 64}));
          EXPECT_EQ(countOps<mlir::tensor::ExtractSliceOp>(loop), 0u);
          EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(loop), 0u);
          EXPECT_EQ(countOps<mlir::linalg::AddOp>(loop), 1u);
        } else {
          ++fullStates;
          EXPECT_EQ(state.getDimSize(1), extent);
        }
      });
      EXPECT_EQ(localStates, segments * (nested ? 2 : 1));
      EXPECT_EQ(fullStates, 1u);
      // Adjacent segment handoffs fold to one extraction/writeback per output
      // tile. The final 1/7 rows remain the original state, not an empty
      // tensor.
      EXPECT_EQ(countOps<mlir::tensor::ExtractSliceOp>(*module), 1u);
      EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(*module), 1u);
      mlir::tensor::InsertSliceOp writeback;
      module->walk([&](mlir::tensor::InsertSliceOp op) { writeback = op; });
      ASSERT_TRUE(writeback);
      auto outer = writeback->getParentOfType<mlir::scf::ForOp>();
      ASSERT_TRUE(outer);
      EXPECT_EQ(writeback.getDest(), outer.getRegionIterArgs().front());
      EXPECT_EQ(writeback.getStaticSizes(),
                (llvm::ArrayRef<int64_t>{1, 32, 64}));
      auto result = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(result.succeeded()) << result.detail;
      EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*module)));
      EXPECT_EQ(result.statistics.bufferizationInvocations, 1u);
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      ASSERT_EQ(standalone->size(), 1u);
      auto &tile = standalone->front();
      TileRegionToInstrLoweringSession session(*context);
      llvm::SmallVector<TileRegionOp, 2> regions;
      tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
      for (TileRegionOp region : regions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      TileMemoryPlanningFailure memoryFailure;
      auto planned = planTileMemory(std::move(tile.module), &memoryFailure);
      ASSERT_TRUE(mlir::succeeded(planned));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
    }
}

TEST_F(LayoutOptimizationTest, IndependentAndSwappedSubsetStatesRemainDistinct) {
  for (bool swap : {false, true}) {
    SCOPED_TRACE(swap);
    std::string text = R"mlir(
module { wafer.tile.module card_id = 0 tile_id = 0 {
  func.func @entry(%input: tensor<1x1031x64xf16>) {
    %r:2 = wafer.tile.region(%input : tensor<1x1031x64xf16>)
        -> (tensor<1x1031x64xf16>, tensor<1x1031x64xf16>) {
    ^bb0(%local: tensor<1x1031x64xf16>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %loop:3 = scf.for %i = %c0 to %c3 step %c1
          iter_args(%a = %local, %b = %local, %n = %c0)
          -> (tensor<1x1031x64xf16>, tensor<1x1031x64xf16>, index) {
        %sa = tensor.extract_slice %a[0, 0, 0] [1, 4, 64] [1, 1, 1]
            : tensor<1x1031x64xf16> to tensor<4x64xf16>
        %sb = tensor.extract_slice %b[0, 7, 0] [1, 4, 64] [1, 1, 1]
            : tensor<1x1031x64xf16> to tensor<1x4x64xf16>
        %va = linalg.add ins(%sa, %sa : tensor<4x64xf16>, tensor<4x64xf16>)
            outs(%sa : tensor<4x64xf16>) -> tensor<4x64xf16>
        %vb = linalg.add ins(%sb, %sb : tensor<1x4x64xf16>, tensor<1x4x64xf16>)
            outs(%sb : tensor<1x4x64xf16>) -> tensor<1x4x64xf16>
        %za = tensor.insert_slice %va into %a[0, 0, 0] [1, 4, 64] [1, 1, 1]
            : tensor<4x64xf16> into tensor<1x1031x64xf16>
        %zb = tensor.insert_slice %vb into %b[0, 7, 0] [1, 4, 64] [1, 1, 1]
            : tensor<1x4x64xf16> into tensor<1x1031x64xf16>
        %next = arith.addi %n, %c1 : index
)mlir";
    text += swap ? "scf.yield %zb, %za, %next" : "scf.yield %za, %zb, %next";
    text += R"mlir( : tensor<1x1031x64xf16>, tensor<1x1031x64xf16>, index
      }
      wafer.tile.yield %loop#0, %loop#1
          : tensor<1x1031x64xf16>, tensor<1x1031x64xf16>
    }
    return
  }
}})mlir";
    auto module = parse(text);
    ASSERT_TRUE(module);
    auto relations = outputRelation(*module);
    ASSERT_TRUE(mlir::succeeded(normalizeLoopSubsetState(*module, relations)));
    mlir::scf::ForOp loop;
    module->walk([&](mlir::scf::ForOp op) { loop = op; });
    ASSERT_TRUE(loop);
    unsigned tensorStates = 0, scalarStates = 0;
    for (auto argument : loop.getRegionIterArgs()) {
      if (auto type = mlir::dyn_cast<mlir::RankedTensorType>(argument.getType())) {
        ++tensorStates;
        EXPECT_EQ(type.getNumElements(), swap ? 1031 * 64 : 4 * 64);
      } else {
        ++scalarStates;
        EXPECT_TRUE(argument.getType().isIndex());
      }
    }
    EXPECT_EQ(tensorStates, 2u);
    EXPECT_EQ(scalarStates, 1u);
    EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(loop), swap ? 2u : 0u);
    EXPECT_EQ(countOps<mlir::linalg::AddOp>(loop), 2u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest,
       ConditionalSubsetPublicationPreservesObserversAndDominance) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned variant : {0u, 1u, 2u}) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << variant);
      std::string full = "tensor<1x" + std::to_string(extent) + "x64xf16>";
      std::string slice = "tensor<1x128x64xf16>";
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
          << "func.func @entry(%input: " << full << ", %condition: i1) {\n"
          << "%result = wafer.tile.region(%input, %condition : " << full
          << ", i1) -> (" << full << ") { ^bb0(%local: " << full
          << ", %take: i1):\n%c0 = arith.constant 0 : index\n"
          << "%old = tensor.extract_slice %local[0, 0, 0] [1, 128, 64] "
          << "[1, 1, 1] : " << full << " to " << slice << "\n"
          << "%conditional = scf.if %take -> " << slice << " {\n"
          << "%empty = tensor.empty() : " << slice << "\n"
          << "%sum = linalg.add ins(%old, %old : " << slice << ", " << slice
          << ") outs(%empty : " << slice << ") -> " << slice << "\n"
          << "scf.yield %sum : " << slice << "\n} else {\n"
          << "scf.yield %old : " << slice << "\n}\n";
      if (variant == 1)
        out << "%observed = tensor.extract %conditional[%c0, %c0, %c0] : "
            << slice << "\n";
      if (variant == 2)
        out << "%later = linalg.add ins(%local, %local : " << full << ", "
            << full << ") outs(%local : " << full << ") -> " << full << "\n";
      out << "%inserted = tensor.insert_slice %conditional into "
          << (variant == 2 ? "%later" : "%local")
          << "[0, 0, 0] [1, 128, 64] [1, 1, 1] : " << slice << " into " << full
          << "\nwafer.tile.yield %inserted : " << full << "\n}\nreturn\n}}}\n";
      auto module = parse(text);
      ASSERT_TRUE(module);
      auto relations = outputRelation(*module);
      ASSERT_TRUE(
          mlir::succeeded(normalizeLoopSubsetState(*module, relations)));
      mlir::scf::IfOp branch;
      module->walk([&](mlir::scf::IfOp op) { branch = op; });
      ASSERT_TRUE(branch);
      auto type =
          mlir::cast<mlir::RankedTensorType>(branch.getResult(0).getType());
      EXPECT_EQ(type.getDimSize(1), variant ? 128 : extent);
      EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(branch),
                variant ? 0u : 2u);
      EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(*module),
                variant ? 1u : 2u);
      for (auto yield : {branch.thenYield(), branch.elseYield()}) {
        if (variant)
          continue;
        auto edge =
            yield.getOperand(0).getDefiningOp<mlir::tensor::InsertSliceOp>();
        ASSERT_TRUE(edge);
        EXPECT_EQ(edge.getStaticOffsets(), (llvm::ArrayRef<int64_t>{0, 0, 0}));
        EXPECT_EQ(edge.getStaticSizes(), (llvm::ArrayRef<int64_t>{1, 128, 64}));
        EXPECT_EQ(edge.getStaticStrides(), (llvm::ArrayRef<int64_t>{1, 1, 1}));
        EXPECT_EQ(edge.getDest(),
                  branch.elseYield()
                      .getOperand(0)
                      .getDefiningOp<mlir::tensor::InsertSliceOp>()
                      .getDest());
      }
      EXPECT_EQ(countOps<mlir::tensor::ExtractOp>(*module),
                variant == 1 ? 1u : 0u);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}

TEST_F(LayoutOptimizationTest, SubsetPromotionPreservesUnprovedLoopState) {
  for (llvm::StringRef boundary :
       {"empty", "dynamic", "variant", "observed", "fork", "nested"}) {
    SCOPED_TRACE(boundary.str());
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
        << "func.func @entry(%input: tensor<1x1031x64xf16>, %limit: index) {\n"
        << "%result = wafer.tile.region(%input, %limit : "
           "tensor<1x1031x64xf16>, index) "
        << "-> (tensor<1x1031x64xf16>) { ^bb0(%local: tensor<1x1031x64xf16>, "
           "%bound: index):\n"
        << "%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : index\n"
        << "%end = arith.constant " << (boundary == "empty" ? 0 : 3)
        << " : index\n"
        << "%loop = scf.for %i = %c0 to "
        << (boundary == "dynamic" ? "%bound" : "%end") << " step %c1 "
        << "iter_args(%state = %local) -> tensor<1x1031x64xf16> {\n";
    std::string offset = boundary == "variant" ? "%i" : "0";
    out << "%slice = tensor.extract_slice %state[0, " << offset
        << ", 0] [1, 4, 64] [1, 1, 1] : tensor<1x1031x64xf16> "
        << "to tensor<1x4x64xf16>\n"
        << "%sum = linalg.add ins(%slice, %slice : tensor<1x4x64xf16>, "
        << "tensor<1x4x64xf16>) outs(%slice : tensor<1x4x64xf16>) "
        << "-> tensor<1x4x64xf16>\n";
    if (boundary == "observed")
      out << "%read = tensor.extract %state[%c0, %c0, %c0] "
          << ": tensor<1x1031x64xf16>\n";
    if (boundary == "fork")
      out << "%other = tensor.insert_slice %sum into %state[0, 4, 0] "
          << "[1, 4, 64] [1, 1, 1] : tensor<1x4x64xf16> "
          << "into tensor<1x1031x64xf16>\n";
    if (boundary == "nested")
      out << "%inner = scf.for %j = %c0 to %bound step %c1 "
          << "iter_args(%forwarded = %state) -> tensor<1x1031x64xf16> {\n"
          << "scf.yield %forwarded : tensor<1x1031x64xf16>\n}\n";
    out << "%updated = tensor.insert_slice %sum into %state[0, " << offset
        << ", 0] [1, 4, 64] [1, 1, 1] : tensor<1x4x64xf16> "
        << "into tensor<1x1031x64xf16>\n"
        << "scf.yield %updated : tensor<1x1031x64xf16>\n}\n"
        << "wafer.tile.yield %loop : tensor<1x1031x64xf16>\n}\nreturn\n}}}\n";
    auto module = parse(text);
    ASSERT_TRUE(module);
    std::string before;
    llvm::raw_string_ostream beforeStream(before);
    module->print(beforeStream);
    auto relations = outputRelation(*module);
    ASSERT_TRUE(mlir::succeeded(normalizeLoopSubsetState(*module, relations)));
    std::string after;
    llvm::raw_string_ostream afterStream(after);
    module->print(afterStream);
    EXPECT_EQ(before, after);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest, EmptySlicesUseLocalStorageBeforeBufferization) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool reduced : {false, true})
      for (bool initialized : {false, true}) {
        SCOPED_TRACE(extent);
        SCOPED_TRACE(reduced);
        SCOPED_TRACE(initialized);
        std::string full = "tensor<64x2x" + std::to_string(extent) + "x64xf16>";
        std::string type = "tensor<" + std::string(reduced ? "2x" : "1x2x") +
                           std::to_string(extent) + "x64xf16>";
        std::string map = reduced ? "affine_map<(a,b,c)->(a,b,c)>"
                                  : "affine_map<(a,b,c,d)->(a,b,c,d)>";
        std::string iterators =
            reduced ? "[\"parallel\",\"parallel\",\"parallel\"]"
                    : "[\"parallel\",\"parallel\",\"parallel\",\"parallel\"]";
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { wafer.tile.module card_id = 0 tile_id = 0 { func.func "
               "@entry(%input: "
            << type << ") -> (" << type << ", " << type << ") {\n"
            << "%a, %b = wafer.tile.region(%input : " << type << ") -> ("
            << type << ", " << type << ") { ^bb0(%x: " << type << "):\n"
            << "%base = tensor.empty() : " << full << "\n";
        if (initialized)
          out << "%one = arith.constant 1.25 : f16\n%filled = linalg.fill "
                 "ins(%one : f16) outs(%base : "
              << full << ") -> " << full << "\n";
        for (unsigned i : {0u, 1u}) {
          out << "%slice" << i << " = tensor.extract_slice %"
              << (initialized ? "filled" : "base") << "[" << (i ? 9 : 5)
              << ",0,0,0] [1,2," << extent << ",64] [1,1,1,1] : " << full
              << " to " << type << "\n"
              << "%value" << i << " = linalg.generic {indexing_maps = [" << map
              << "," << map << "], iterator_types = " << iterators
              << "} ins(%x : " << type << ") outs(%slice" << i << " : " << type
              << ") { ^bb1(%v: f16, %init: f16):\n"
              << "%next = arith.addf %v, " << (initialized ? "%init" : "%v")
              << " : f16\nlinalg.yield %next : f16 } -> " << type << "\n";
        }
        out << "wafer.tile.yield %value0, %value1 : " << type << ", " << type
            << " }\nreturn %a, %b : " << type << ", " << type << " } } }";
        auto module = parse(text);
        ASSERT_TRUE(module);
        TileRegionOp region;
        module->walk([&](TileRegionOp op) { region = op; });
        StructuredMaterializationRelations relations;
        relations.structuralOutputs = {{0, region.getResult(0)},
                                       {1, region.getResult(1)}};
        auto result = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(result.succeeded()) << result.detail;
        unsigned large = 0;
        module->walk([&](mlir::memref::AllocOp op) {
          large += op.getType().getShape() ==
                   llvm::ArrayRef<int64_t>{64, 2, extent, 64};
        });
        EXPECT_EQ(large != 0, initialized);
        EXPECT_EQ(countOps<mlir::linalg::FillOp>(*module),
                  initialized ? 1u : 0u);
        EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*module)));
        EXPECT_TRUE(mlir::succeeded(
            checkStructuredBufferRelationsCurrent(*module, relations)));
      }
}

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
    EXPECT_EQ(result.statistics.invocations, 1u);
    EXPECT_EQ(result.statistics.bufferizationInvocations, 1u);
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
       CanonicalAssignmentClosesAlignedAndRaggedInsufficientBudgetCases) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (uint64_t workLimit : {UINT64_C(0), UINT64_C(1)}) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << workLimit);
      auto module = parse(makeFanoutContractionSource(extent, /*useCount=*/15));
      ASSERT_TRUE(module);
      StructuredMaterializationRelations relations = outputRelation(*module);
      LayoutOptimizationResult result =
          resolveCurrentLayoutsAndBufferize(*module, relations, workLimit);
      ASSERT_EQ(result.status, ExactPBQPStatus::Feasible) << result.detail;
      EXPECT_EQ(result.statistics.canonicalAssignmentsBuilt, 1u);
      EXPECT_EQ(result.statistics.canonicalAssignmentFallbacks, 1u);
      EXPECT_EQ(result.statistics.bufferizationInvocations, 1u);
      EXPECT_GT(result.statistics.pbqpVariables, 0u);
      EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*module)));
      EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
      ASSERT_EQ(relations.structuralOutputs.size(), 1u);
      EXPECT_TRUE(isWaferDDRMemRefType(
          relations.structuralOutputs.front().endpoint.getType()));
    }
  }
}

TEST_F(LayoutOptimizationTest,
       NestedReductionStateReanalyzesAfterInnerDestinationBinding) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::string inputType = "tensor<2x8x" + std::to_string(extent) + "xf16>";
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
        << "func.func @entry(%input: " << inputType << ") {\n"
        << "%result = wafer.tile.region(%input : " << inputType
        << ") -> (tensor<2xf16>) {\n^bb0(%local: " << inputType << "):\n"
        << "%c0 = arith.constant 0 : index\n%c4 = arith.constant 4 : index\n"
        << "%c8 = arith.constant 8 : index\n%c32 = arith.constant 32 : index\n"
        << "%end = arith.constant " << extent / 32 * 32 << " : index\n"
        << "%zero = arith.constant 0.0 : f16\n"
        << "%empty = tensor.empty() : tensor<2xf16>\n"
        << "%init = linalg.fill ins(%zero : f16) outs(%empty : tensor<2xf16>) "
           "-> tensor<2xf16>\n"
        << "%outer = scf.for %x = %c0 to %c8 step %c4 iter_args(%a = %init) -> "
           "(tensor<2xf16>) {\n"
        << "%inner = scf.for %y = %c0 to %end step %c32 iter_args(%b = %a) -> "
           "(tensor<2xf16>) {\n"
        << "%slice = tensor.extract_slice %local[0, %x, %y] [2, 4, 32] [1, 1, "
           "1] : "
        << inputType << " to tensor<2x4x32xf16>\n"
        << "%sum = linalg.reduce ins(%slice : tensor<2x4x32xf16>) outs(%b : "
           "tensor<2xf16>) "
        << "dimensions = [1, 2] (%v: f16, %old: f16) {\n"
        << "%next = arith.addf %old, %v : f16\nlinalg.yield %next : f16\n}\n"
        << "scf.yield %sum : tensor<2xf16>\n}\n";
    if (extent % 32) {
      std::string tailType =
          "tensor<2x4x" + std::to_string(extent % 32) + "xf16>";
      out << "%tail = tensor.extract_slice %local[0, %x, " << extent / 32 * 32
          << "] [2, 4, " << extent % 32 << "] [1, 1, 1] : " << inputType
          << " to " << tailType << "\n"
          << "%final = linalg.reduce ins(%tail : " << tailType
          << ") outs(%inner : tensor<2xf16>) "
          << "dimensions = [1, 2] (%v: f16, %old: f16) {\n"
          << "%next = arith.addf %old, %v : f16\nlinalg.yield %next : f16\n}\n"
          << "scf.yield %final : tensor<2xf16>\n";
    } else {
      out << "scf.yield %inner : tensor<2xf16>\n";
    }
    out << "}\nwafer.tile.yield %outer : tensor<2xf16>\n}\nreturn\n}}}\n";
    auto module = parse(text);
    ASSERT_TRUE(module);
    auto relations = outputRelation(*module);
    std::string diagnosticsText;
    llvm::raw_string_ostream diagnostics(diagnosticsText);
    auto timing =
        std::make_shared<wafer::support::CompileTimingSession>(diagnostics);
    auto result = [&] {
      wafer::support::ScopedCompileTimingActivation activation(timing);
      return resolveCurrentLayoutsAndBufferize(*module, relations, 0);
    }();
    timing->finishAndPrintSummary();
    EXPECT_NE(diagnosticsText.find("category=loop-state-binding "
                                   "name=reused-final-analysis value=1 "
                                   "overflow=false"),
              std::string::npos)
        << diagnosticsText;
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.statistics.bufferizationInvocations, 1u);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(*module), 2u);
    EXPECT_EQ(
        countOps<mlir::bufferization::MaterializeInDestinationOp>(*module), 0u);
    module->walk([&](mlir::scf::ForOp loop) {
      for (mlir::Value initial : loop.getInitArgs()) {
        auto allocation = initial.getDefiningOp<mlir::memref::AllocOp>();
        if (allocation) {
          EXPECT_FALSE(loop->isAncestor(allocation));
        }
      }
    });
    EXPECT_TRUE(mlir::succeeded(
        checkStructuredBufferRelationsCurrent(*module, relations)));
    auto lowered = lowerStructuredComputeToTile(*module, relations);
    EXPECT_TRUE(lowered.succeeded()) << lowered.detail;
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest, DynamicTileCostDoesNotBecomeAnImpossibleLayout) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::string text;
    llvm::raw_string_ostream out(text);
    out << R"mlir(module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x)mlir"
        << extent << R"mlir(x64xf16>) {
      %result = wafer.tile.region(%input : tensor<2x)mlir"
        << extent << R"mlir(x64xf16>) -> (tensor<2x)mlir" << extent
        << R"mlir(xf16>) {
      ^bb0(%local: tensor<2x)mlir"
        << extent << R"mlir(x64xf16>):
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %end = arith.constant )mlir"
        << extent << R"mlir( : index
        %zero = arith.constant 0.0 : f16
        %empty = tensor.empty() : tensor<2x)mlir"
        << extent << R"mlir(xf16>
        %full = scf.for %i = %c0 to %end step %c64
            iter_args(%state = %empty) -> (tensor<2x)mlir"
        << extent << R"mlir(xf16>) {
          %left = arith.subi %end, %i : index
          %size = arith.minsi %left, %c64 : index
          %piece = tensor.extract_slice %local[0, %i, 0] [2, %size, 64] [1, 1, 1]
              : tensor<2x)mlir"
        << extent << R"mlir(x64xf16> to tensor<2x?x64xf16>
          %init = tensor.empty(%size) : tensor<2x?xf16>
          %zeros = linalg.fill ins(%zero : f16) outs(%init : tensor<2x?xf16>)
              -> tensor<2x?xf16>
          %sum = linalg.reduce ins(%piece : tensor<2x?x64xf16>)
              outs(%zeros : tensor<2x?xf16>) dimensions = [2]
              (%x: f16, %acc: f16) {
                %next = arith.addf %x, %acc : f16
                linalg.yield %next : f16
              }
          %next = tensor.insert_slice %sum into %state[0, %i] [2, %size] [1, 1]
              : tensor<2x?xf16> into tensor<2x)mlir"
        << extent << R"mlir(xf16>
          scf.yield %next : tensor<2x)mlir"
        << extent << R"mlir(xf16>
        }
        wafer.tile.yield %full : tensor<2x)mlir"
        << extent << R"mlir(xf16>
      }
      return
    }
  }
})mlir";
    auto module = parse(text);
    ASSERT_TRUE(module);
    auto relations = outputRelation(*module);
    auto prepared = prepareCurrentLayoutInput(*module, relations);
    ASSERT_TRUE(prepared.succeeded()) << prepared.detail;
    auto query = queryCurrentLayoutAssignment(*module);
    ASSERT_TRUE(query.query) << query.outcome.detail;
    auto assignment = query.query->solve(0);
    ASSERT_EQ(assignment.status, ExactPBQPStatus::Feasible);
    ASSERT_TRUE(assignment.cost);
    EXPECT_GT(*assignment.cost, 1u);
    auto applied = query.query->apply(*module, relations, assignment);
    // The query must remain feasible even when the actual downstream cannot
    // yet describe this dynamic blocked-layout copy. Only that actual copy
    // proof may reject the choice, with a typed unsupported result.
    EXPECT_EQ(applied.status, ExactPBQPStatus::NoSolution) << applied.detail;
    EXPECT_NE(applied.detail.find("no exact GatherScatter proof"),
              std::string::npos);
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

TEST_F(LayoutOptimizationTest,
       MandatoryConversionReusesReadOnlyNestedPreparation) {
  for (int64_t extent : {1024, 1025, 1031})
    for (llvm::StringRef mutation :
         {"none", "source", "alias", "free", "unknown", "result"}) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << mutation.str());
      std::string shape = "2x" + std::to_string(extent) + "x64xbf16";
      std::string source = "memref<" + shape + ", #wafer.memory<spm, tensor>>";
      std::string target = "memref<" + shape + ", #wafer.memory<spm, ncx>>";
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "module { ";
      if (mutation == "unknown")
        out << "func.func private @unknown(" << source << ") ";
      out << "func.func @entry(%take: i1) { "
          << "wafer.tile.region(%take : i1) -> () { ^bb0(%condition: i1): "
          << "%c0 = arith.constant 0 : index %c1 = arith.constant 1 : index "
          << "%step = arith.constant 512 : index "
          << "%end = arith.constant " << extent << " : index "
          << "%zero = arith.constant 0.0 : bf16 "
          << "%source = memref.alloc() : " << source
          << " %dest = memref.alloc() : " << target
          << " wafer.tile.fill %source, %zero : " << source << ", bf16 "
          << "scf.for %q = %c0 to %end step %step { "
          << "%trips = arith.divui %q, %step : index "
          << "scf.for %i = %c0 to %trips step %c1 { scf.if %condition { "
          << "%early = wafer.tile.materialize_layout %source : " << source
          << " -> " << target
          << " wafer.tile.elementwise_into <add> %early, %early into "
          << (mutation == "result" ? "%early" : "%dest") << " : " << target
          << ", " << target << " into " << target << " } } ";
      if (mutation == "source")
        out << "memref.store %zero, %source[%c0,%c0,%c0] : " << source;
      if (mutation == "alias") {
        std::string alias = "memref<?x?x64xbf16, #wafer.memory<spm, tensor>>";
        out << "%alias = memref.cast %source : " << source << " to " << alias
            << " memref.store %zero, %alias[%c0,%c0,%c0] : " << alias;
      }
      if (mutation == "free")
        out << "memref.dealloc %source : " << source;
      if (mutation == "unknown")
        out << "func.call @unknown(%source) : (" << source << ") -> ()";
      out << " } %late = wafer.tile.materialize_layout %source : " << source
          << " -> " << target
          << " wafer.tile.elementwise_into <add> %late, %late into %dest : "
          << target << ", " << target << " into " << target
          << " wafer.tile.yield } return } }";
      auto module = parse(text);
      ASSERT_TRUE(module) << text;
      std::string before;
      llvm::raw_string_ostream beforeStream(before);
      module->print(beforeStream);
      StructuredMaterializationRelations relations;
      auto placement = optimizePhysicalMovementPlacement(
          *module, relations, LayoutMaterializationPlacement::LoopInvariant);
      ASSERT_TRUE(mlir::succeeded(placement));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      if (mutation != "none") {
        std::string after;
        llvm::raw_string_ostream afterStream(after);
        module->print(afterStream);
        EXPECT_EQ(before, after);
        continue;
      }
      ASSERT_EQ(countOps<LayoutMaterializeOp>(*module), 1u);
      LayoutMaterializeOp conversion;
      module->walk([&](LayoutMaterializeOp op) { conversion = op; });
      EXPECT_FALSE(conversion->getParentOfType<mlir::scf::ForOp>());
      EXPECT_FALSE(conversion->getParentOfType<mlir::scf::IfOp>());
      EXPECT_EQ(std::distance(conversion.getResult().use_begin(),
                              conversion.getResult().use_end()),
                4);
      // The scalar domain contains zero, one and (for the ragged extents) two
      // inner iterations. Both branch outcomes share the mandatory snapshot.
      unsigned iterations = 0;
      for (int64_t q = 0; q < extent; q += 512)
        iterations += q / 512;
      EXPECT_EQ(iterations, extent == 1024 ? 1u : 3u);
      TileRegionToInstrLoweringSession session(*context);
      TileRegionOp region;
      module->walk([&](TileRegionOp op) { region = op; });
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
      EXPECT_EQ(countOps<LayoutMaterializeOp>(*module), 0u);
      EXPECT_GT(countOps<InstrGatherScatterOp>(*module), 0u);
      EXPECT_EQ(countOps<InstrElementwiseOp>(*module), 2u);
      module->walk([&](SyncNCCJoinOp join) {
        EXPECT_FALSE(join->getParentOfType<mlir::scf::ForOp>());
      });
      TileMemoryPlanningFailure failure;
      auto planned = planTileMemory(std::move(module), &failure);
      ASSERT_TRUE(mlir::succeeded(planned));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
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
       LargeConnectedDiamondGraphMinimizesActualMaterializations) {
  constexpr int64_t extent = 1025;
  constexpr unsigned diamondCount = 32;
  constexpr unsigned expectedContractions = 3 + diamondCount * 3;
  constexpr unsigned expectedViews = diamondCount / 8;
  const std::string source =
      makeLargeConnectedLayoutSource(extent, diamondCount);

  auto exhausted = parse(source);
  ASSERT_TRUE(exhausted);
  StructuredMaterializationRelations exhaustedRelations =
      outputRelation(*exhausted);
  LayoutOptimizationResult exhaustedResult = resolveCurrentLayoutsAndBufferize(
      *exhausted, exhaustedRelations, /*workLimit=*/0);
  ASSERT_EQ(exhaustedResult.status, ExactPBQPStatus::Feasible)
      << exhaustedResult.detail;
  EXPECT_EQ(exhaustedResult.statistics.canonicalAssignmentsBuilt, 1u);
  EXPECT_EQ(exhaustedResult.statistics.canonicalAssignmentFallbacks, 1u);
  EXPECT_EQ(exhaustedResult.statistics.bufferizationInvocations, 1u);
  EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*exhausted)));
  EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
      exhausted->getOperation(), exhaustedRelations)));

  auto first = parse(source);
  auto second = parse(source);
  ASSERT_TRUE(first && second);
  EXPECT_EQ(countOps<mlir::linalg::BatchMatmulOp>(first->getOperation()),
            expectedContractions);
  StructuredMaterializationRelations firstRelations = outputRelation(*first);
  StructuredMaterializationRelations secondRelations = outputRelation(*second);
  auto start = std::chrono::steady_clock::now();
  LayoutOptimizationResult firstResult =
      resolveCurrentLayoutsAndBufferize(*first, firstRelations);
  const auto wallMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  LayoutOptimizationResult secondResult =
      resolveCurrentLayoutsAndBufferize(*second, secondRelations);
  ASSERT_TRUE(firstResult.succeeded()) << firstResult.detail;
  ASSERT_TRUE(secondResult.succeeded()) << secondResult.detail;

  EXPECT_EQ(firstResult.statistics.tupleVariables, expectedContractions);
  EXPECT_GE(firstResult.statistics.valueGroups, expectedContractions);
  EXPECT_GE(firstResult.statistics.useBindings, expectedContractions * 2);
  EXPECT_GT(firstResult.statistics.pbqpVariables,
            firstResult.statistics.valueGroups);
  EXPECT_GT(firstResult.statistics.pbqpFactors,
            firstResult.statistics.tupleVariables);
  EXPECT_GT(firstResult.statistics.solverWork, 0u);
  EXPECT_EQ(firstResult.statistics.selectedMaterializations, 2u);
  EXPECT_EQ(firstResult.statistics.layoutMaterializationsAfter, 2u);
  EXPECT_EQ(countOps<LayoutMaterializeOp>(first->getOperation()), 2u);
  EXPECT_EQ(countOps<mlir::memref::ExpandShapeOp>(first->getOperation()),
            expectedViews);
  EXPECT_EQ(countOps<mlir::memref::CollapseShapeOp>(first->getOperation()),
            expectedViews);
  EXPECT_EQ(firstResult.statistics.bufferizationInvocations, 1u);
  EXPECT_EQ(firstResult.statistics.redundantPublicationCopies, 0u);
  EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*first)));
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*first, firstRelations)));

  std::string firstText;
  llvm::raw_string_ostream firstStream(firstText);
  first->print(firstStream);
  firstStream.flush();
  std::string secondText;
  llvm::raw_string_ostream secondStream(secondText);
  second->print(secondStream);
  secondStream.flush();
  EXPECT_EQ(secondText, firstText);
  EXPECT_EQ(secondResult.statistics.pbqpVariables,
            firstResult.statistics.pbqpVariables);
  EXPECT_EQ(secondResult.statistics.pbqpFactors,
            firstResult.statistics.pbqpFactors);
  EXPECT_EQ(secondResult.statistics.solverWork,
            firstResult.statistics.solverWork);

  RecordProperty("layout_pbqp_variables",
                 static_cast<int>(firstResult.statistics.pbqpVariables));
  RecordProperty("layout_pbqp_factors",
                 static_cast<int>(firstResult.statistics.pbqpFactors));
  RecordProperty("layout_pbqp_solver_work",
                 static_cast<int>(firstResult.statistics.solverWork));
  RecordProperty("layout_wall_ms", static_cast<int>(wallMilliseconds));
}

TEST_F(LayoutOptimizationTest,
       MixedOperatorGraphConsumesDecomposedAttentionWithoutSemanticOps) {
  constexpr int64_t extent = 1025;
  constexpr unsigned diamondCount = 16;
  const std::string source = makeLargeConnectedLayoutSource(
      extent, diamondCount, /*mixedOperators=*/true);
  auto exhausted = parse(source);
  ASSERT_TRUE(exhausted);
  StructuredMaterializationRelations exhaustedRelations =
      outputRelation(*exhausted);
  LayoutOptimizationResult exhaustedResult = resolveCurrentLayoutsAndBufferize(
      *exhausted, exhaustedRelations, /*workLimit=*/0);
  ASSERT_EQ(exhaustedResult.status, ExactPBQPStatus::Feasible)
      << exhaustedResult.detail;
  EXPECT_EQ(exhaustedResult.statistics.canonicalAssignmentsBuilt, 1u);
  EXPECT_EQ(exhaustedResult.statistics.canonicalAssignmentFallbacks, 1u);
  EXPECT_EQ(exhaustedResult.statistics.bufferizationInvocations, 1u);
  EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*exhausted)));
  EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
      exhausted->getOperation(), exhaustedRelations)));

  auto first = parse(source);
  auto second = parse(source);
  ASSERT_TRUE(first && second) << source;

  EXPECT_EQ(countOps<mlir::linalg::BatchMatmulOp>(first->getOperation()), 53u);
  EXPECT_EQ(countOps<mlir::linalg::MatmulOp>(first->getOperation()), 2u);
  EXPECT_EQ(countOps<mlir::linalg::GenericOp>(first->getOperation()), 15u);
  EXPECT_EQ(countOps<mlir::linalg::FillOp>(first->getOperation()), 6u);
  EXPECT_EQ(countOps<mlir::linalg::TransposeOp>(first->getOperation()), 2u);
  EXPECT_EQ(countOps<mlir::math::ExpOp>(first->getOperation()), 2u);
  EXPECT_EQ(countOps<LinalgExtAttentionOp>(first->getOperation()), 0u);
  EXPECT_EQ(countOps<LinalgExtOnlineAttentionOp>(first->getOperation()), 0u);

  StructuredMaterializationRelations firstRelations = outputRelation(*first);
  StructuredMaterializationRelations secondRelations = outputRelation(*second);
  auto start = std::chrono::steady_clock::now();
  LayoutOptimizationResult firstResult =
      resolveCurrentLayoutsAndBufferize(*first, firstRelations);
  const auto wallMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  LayoutOptimizationResult secondResult =
      resolveCurrentLayoutsAndBufferize(*second, secondRelations);
  ASSERT_TRUE(firstResult.succeeded()) << firstResult.detail;
  ASSERT_TRUE(secondResult.succeeded()) << secondResult.detail;

  EXPECT_EQ(firstResult.statistics.selectedMaterializations, 14u);
  EXPECT_EQ(firstResult.statistics.layoutMaterializationsAfter, 14u);
  EXPECT_EQ(countOps<LayoutMaterializeOp>(first->getOperation()), 14u);
  unsigned tensorToNCx = 0;
  unsigned nCxToTensor = 0;
  unsigned tensorToCx = 0;
  unsigned cxToTensor = 0;
  first->walk([&](LayoutMaterializeOp operation) {
    auto source = getWaferMemoryAttr(
        mlir::cast<mlir::MemRefType>(operation.getSource().getType()));
    auto result = getWaferMemoryAttr(
        mlir::cast<mlir::MemRefType>(operation.getResult().getType()));
    ASSERT_TRUE(source && result);
    tensorToNCx += source.getLayout() == MemLayout::Tensor &&
                   result.getLayout() == MemLayout::NCx;
    nCxToTensor += source.getLayout() == MemLayout::NCx &&
                   result.getLayout() == MemLayout::Tensor;
    tensorToCx += source.getLayout() == MemLayout::Tensor &&
                  result.getLayout() == MemLayout::Cx;
    cxToTensor += source.getLayout() == MemLayout::Cx &&
                  result.getLayout() == MemLayout::Tensor;
  });
  EXPECT_EQ(tensorToNCx, 5u);
  EXPECT_EQ(nCxToTensor, 2u);
  EXPECT_EQ(tensorToCx, 4u);
  EXPECT_EQ(cxToTensor, 3u);
  EXPECT_EQ(countOps<mlir::memref::ExpandShapeOp>(first->getOperation()), 2u);
  EXPECT_EQ(countOps<mlir::memref::CollapseShapeOp>(first->getOperation()), 3u);
  EXPECT_EQ(firstResult.statistics.bufferizationInvocations, 1u);
  EXPECT_EQ(firstResult.statistics.redundantPublicationCopies, 0u);
  EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*first)));
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*first, firstRelations)));

  std::string firstText;
  llvm::raw_string_ostream firstStream(firstText);
  first->print(firstStream);
  firstStream.flush();
  std::string secondText;
  llvm::raw_string_ostream secondStream(secondText);
  second->print(secondStream);
  secondStream.flush();
  EXPECT_EQ(secondText, firstText);
  EXPECT_EQ(secondResult.statistics.solverWork,
            firstResult.statistics.solverWork);
  RecordProperty("mixed_layout_pbqp_variables",
                 static_cast<int>(firstResult.statistics.pbqpVariables));
  RecordProperty("mixed_layout_pbqp_factors",
                 static_cast<int>(firstResult.statistics.pbqpFactors));
  RecordProperty("mixed_layout_pbqp_solver_work",
                 static_cast<int>(firstResult.statistics.solverWork));
  RecordProperty("mixed_layout_wall_ms", static_cast<int>(wallMilliseconds));
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

TEST_F(LayoutOptimizationTest,
       LocalAndRemotePartialWindowsUseCoordinatesWithinTheProducerPiece) {
  for (int64_t rows : {1024, 1025, 1031}) {
    SCOPED_TRACE(rows);
    std::string text = R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @source(%input: tensor<2xFULLx64xf16>) {
      %published = wafer.tile.region(%input : tensor<2xFULLx64xf16>)
          -> (tensor<2xFULLx64xf16>) {
      ^bb0(%local: tensor<2xFULLx64xf16>):
        %piece = tensor.extract_slice %local[1, 5, 0] [1, ROWS, 64]
            [1, 1, 1] : tensor<2xFULLx64xf16> to tensor<1xROWSx64xf16>
        %empty = tensor.empty() : tensor<2xFULLx64xf16>
        %full = tensor.insert_slice %piece into %empty[1, 5, 0]
            [1, ROWS, 64] [1, 1, 1]
            : tensor<1xROWSx64xf16> into tensor<2xFULLx64xf16>
        wafer.tile.yield %full : tensor<2xFULLx64xf16>
      }
      %used = wafer.tile.region(%published, %published : tensor<2xFULLx64xf16>, tensor<2xFULLx64xf16>)
          -> (tensor<1xLOCALx64xf16>) {
      ^bb0(%unused: tensor<2xFULLx64xf16>, %local: tensor<2xFULLx64xf16>):
        %piece = tensor.extract_slice %local[1, 5, 0] [1, LOCAL, 64]
            [1, 1, 1] : tensor<2xFULLx64xf16> to tensor<1xLOCALx64xf16>
        wafer.tile.yield %piece : tensor<1xLOCALx64xf16>
      }
      return
    }
  }
  wafer.tile.module card_id = 0 tile_id = 1 {
    func.func @destination(%input: tensor<2xFULLx64xf16>) {
      %used = wafer.tile.region(%input : tensor<2xFULLx64xf16>)
          -> (tensor<1x1x64xf16>) {
      ^bb0(%external: tensor<2xFULLx64xf16>):
        %piece = tensor.extract_slice %external[1, END, 0] [1, 1, 64]
            [1, 1, 1] : tensor<2xFULLx64xf16> to tensor<1x1x64xf16>
        wafer.tile.yield %piece : tensor<1x1x64xf16>
      }
      return
    }
  }
}
)mlir";
    for (auto [token, value] :
         {std::pair<llvm::StringRef, int64_t>{"FULL", 2 * rows},
          {"ROWS", rows},
          {"LOCAL", rows - 1},
          {"END", rows + 4}}) {
      size_t position = 0;
      while ((position = text.find(token.str(), position)) !=
             std::string::npos) {
        std::string replacement = std::to_string(value);
        text.replace(position, token.size(), replacement);
        position += replacement.size();
      }
    }
    auto module = parse(text);
    ASSERT_TRUE(module);
    llvm::SmallVector<TileRegionOp, 3> regions;
    module->walk([&](TileRegionOp region) { regions.push_back(region); });
    ASSERT_EQ(regions.size(), 3u);
    StructuredMaterializationRelations relations;
    relations.boundaryRelations.push_back(
        {regions[0].getResult(0), regions[2].getBody().getArgument(0)});
    auto result = prepareCurrentLayoutInput(*module, relations);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    const auto &relation = relations.boundaryRelations.front();
    auto expected = mlir::RankedTensorType::get(
        {1, rows, 64}, mlir::Float16Type::get(context.get()));
    EXPECT_EQ(relation.sourceEndpoint.getType(), expected);
    EXPECT_EQ(relation.destinationEndpoint.getType(), expected);
    EXPECT_EQ(regions[1].getInputs()[0].getType(), expected);
    EXPECT_EQ(regions[1].getBody().getArgument(0).getType(), expected);
    EXPECT_TRUE(regions[1].getBody().getArgument(0).use_empty());
    auto argument =
        mlir::cast<mlir::BlockArgument>(relation.destinationEndpoint);
    bool sawRemoteWindow = false;
    for (auto *user : argument.getUsers())
      if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
        EXPECT_EQ(slice.getStaticOffsets(),
                  llvm::ArrayRef<int64_t>({0, rows - 1, 0}));
        EXPECT_EQ(slice.getStaticSizes(), llvm::ArrayRef<int64_t>({1, 1, 64}));
        sawRemoteWindow = true;
      }
    EXPECT_TRUE(sawRemoteWindow);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest,
       SameTileExactInsertExtractBoundaryUsesOnlyCompactStorage) {
  constexpr llvm::StringLiteral text = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1025x64xf16>) {
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
      %result = wafer.tile.region(
          %published : tensor<2x1025x64xf16>)
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
  relations.structuralOutputs.push_back({0, regions[1].getResult(0)});
  LayoutOptimizationResult result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  EXPECT_GE(result.statistics.boundarySourceViewsElided, 1u);
  EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(module->getOperation()), 0u);
  unsigned fullSPMAllocations = 0;
  module->walk([&](mlir::memref::AllocOp allocation) {
    if (isWaferSPMMemRefType(allocation.getType()) &&
        allocation.getType().getShape() ==
            llvm::ArrayRef<int64_t>({2, 1025, 64}))
      ++fullSPMAllocations;
  });
  EXPECT_EQ(fullSPMAllocations, 0u);
  EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*module)));
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
}

TEST_F(LayoutOptimizationTest,
       PartialTensorUpdatePublishesItsCompleteCurrentValue) {
  constexpr llvm::StringLiteral input = R"mlir(
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
  auto module = parse(input);
  ASSERT_TRUE(module);
  StructuredMaterializationRelations relations = outputRelation(*module);
  auto result =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(result.succeeded()) << result.detail;
  ASSERT_EQ(relations.structuralOutputs.size(), 1u);
  auto output = relations.structuralOutputs.front().endpoint
                    .getDefiningOp<mlir::memref::SubViewOp>();
  ASSERT_TRUE(output);
  EXPECT_EQ(output.getStaticOffsets(), (llvm::ArrayRef<int64_t>{0, 0, 0}));
  EXPECT_EQ(output.getStaticSizes(), (llvm::ArrayRef<int64_t>{2, 1025, 64}));
  EXPECT_EQ(result.statistics.outputDestinations, 1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_TRUE(mlir::succeeded(
      checkStructuredBufferRelationsCurrent(*module, relations)));
}

TEST_F(LayoutOptimizationTest, DynamicBoundaryReadsRequireWindowContainment) {
  for (bool contained : {true, false}) {
    SCOPED_TRACE(contained);
    std::string text = R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x2048x64xf16>) -> tensor<1x128x64xf16> {
      %published = wafer.tile.region(%input : tensor<2x2048x64xf16>)
          -> (tensor<2x2048x64xf16>) {
      ^bb0(%local: tensor<2x2048x64xf16>):
        %origin = arith.constant 512 : index
        %piece = tensor.extract_slice %local[1, 512, 0] [1, 1024, 64] [1, 1, 1]
          : tensor<2x2048x64xf16> to tensor<1x1024x64xf16>
        %empty = tensor.empty() : tensor<2x2048x64xf16>
        %full = tensor.insert_slice %piece into %empty[1, %origin, 0] [1, 1024, 64] [1, 1, 1]
          : tensor<1x1024x64xf16> into tensor<2x2048x64xf16>
        wafer.tile.yield %full : tensor<2x2048x64xf16>
      }
      %result = wafer.tile.region(%published : tensor<2x2048x64xf16>)
          -> (tensor<1x128x64xf16>) {
      ^bb0(%local: tensor<2x2048x64xf16>):
        %lo = arith.constant 512 : index
        %hi = arith.constant LIMIT : index
        %step = arith.constant 128 : index
        %empty = tensor.empty() : tensor<1x128x64xf16>
        %last = scf.for %iv = %lo to %hi step %step iter_args(%state = %empty)
            -> tensor<1x128x64xf16> {
          %piece = tensor.extract_slice %local[1, %iv, 0] [1, 128, 64] [1, 1, 1]
            : tensor<2x2048x64xf16> to tensor<1x128x64xf16>
          scf.yield %piece : tensor<1x128x64xf16>
        }
        wafer.tile.yield %last : tensor<1x128x64xf16>
      }
      return %result : tensor<1x128x64xf16>
    }
  }
}
)mlir";
    text.replace(text.find("LIMIT"), 5, contained ? "1536" : "1664");
    auto module = parse(text);
    ASSERT_TRUE(module);
    StructuredMaterializationRelations relations;
    auto result = prepareCurrentLayoutInput(*module, relations);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    llvm::SmallVector<TileRegionOp> regions;
    module->walk([&](TileRegionOp region) { regions.push_back(region); });
    ASSERT_EQ(regions.size(), 2u);
    auto shape = mlir::cast<mlir::RankedTensorType>(regions[0].getResult(0).getType());
    EXPECT_EQ(shape.getShape(), contained ? (llvm::ArrayRef<int64_t>{1, 1024, 64})
                                         : (llvm::ArrayRef<int64_t>{2, 2048, 64}));
    EXPECT_EQ(regions[1].getBody().getArgument(0).getType(), shape);
    EXPECT_EQ(result.statistics.boundarySourceViewsElided, contained ? 1u : 0u);
    unsigned dynamicReads = 0;
    regions[1].walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() != regions[1].getBody().getArgument(0))
        return;
      ++dynamicReads;
      EXPECT_EQ(slice.getStaticOffsets()[0], contained ? 0 : 1);
      EXPECT_EQ(slice.getStaticSizes(), (llvm::ArrayRef<int64_t>{1, 128, 64}));
      auto offset = mlir::cast<mlir::Value>(slice.getMixedOffsets()[1]);
      auto loop = slice->getParentOfType<mlir::scf::ForOp>();
      auto delta = mlir::ValueBoundsConstraintSet::computeConstantDelta(
          offset, loop.getInductionVar());
      ASSERT_TRUE(mlir::succeeded(delta));
      EXPECT_EQ(*delta, contained ? -512 : 0);
    });
    EXPECT_EQ(dynamicReads, 1u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(LayoutOptimizationTest,
       ConstrainedAssignmentsUseIndependentActualClones) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto base = parse(makeSharedContractionSource(extent));
    ASSERT_TRUE(base);
    auto relations = outputRelation(*base);
    auto prepared = prepareCurrentLayoutInput(*base, relations);
    ASSERT_TRUE(prepared.succeeded()) << prepared.detail;
    auto query = queryCurrentLayoutAssignment(*base);
    ASSERT_TRUE(query.query) << query.outcome.detail;
    auto center = query.query->solve(1048576);
    ASSERT_EQ(center.status, ExactPBQPStatus::Optimal);
    std::vector<ExactPBQPResult> assignments;
    assignments.push_back(center);
    for (const auto &constraint : query.query->alternatives(center)) {
      auto alternative = query.query->solve(1048576, constraint);
      if ((alternative.status == ExactPBQPStatus::Optimal ||
           alternative.status == ExactPBQPStatus::Feasible) &&
          alternative.assignment != center.assignment) {
        assignments.push_back(std::move(alternative));
        break;
      }
    }
    ASSERT_EQ(assignments.size(), 2u)
        << "the real-scale query must expose a second legal layout";
    auto print = [](mlir::ModuleOp module) {
      std::string text;
      llvm::raw_string_ostream stream(text);
      module.print(stream);
      return text;
    };
    const std::string before = print(*base);
    std::vector<std::string> actual;
    for (const auto &assignment : assignments) {
      mlir::IRMapping mapping;
      mlir::OwningOpRef<mlir::ModuleOp> clone(
          mlir::cast<mlir::ModuleOp>(base->getOperation()->clone(mapping)));
      StructuredMaterializationRelations remapped;
      for (const auto &relation : relations.buffers)
        remapped.buffers.push_back({mapping.lookup(relation.owner),
                                    mapping.lookup(relation.buffer),
                                    relation.role});
      for (const auto &relation : relations.structuralOutputs)
        remapped.structuralOutputs.push_back(
            {relation.outputIndex, mapping.lookup(relation.endpoint)});
      for (const auto &relation : relations.boundaryRelations)
        remapped.boundaryRelations.push_back(
            {mapping.lookup(relation.sourceEndpoint),
             mapping.lookup(relation.destinationEndpoint)});
      auto applied = query.query->apply(*clone, remapped, assignment, &mapping);
      ASSERT_TRUE(applied.succeeded()) << applied.detail;
      EXPECT_EQ(applied.statistics.bufferizationInvocations, 1u);
      EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*clone)));
      EXPECT_TRUE(mlir::succeeded(
          checkStructuredBufferRelationsCurrent(*clone, remapped)));
      actual.push_back(print(*clone));
      auto lowered = lowerStructuredComputeToTile(*clone, remapped);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(countOps<mlir::linalg::BatchMatmulOp>(clone->getOperation()),
                0u);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*clone)));
    }
    EXPECT_NE(actual[0], actual[1]);
    EXPECT_EQ(print(*base), before);
    EXPECT_EQ(query.query->solve(1048576).assignment, center.assignment);
  }
}

TEST_F(LayoutOptimizationTest,
       ConvolutionChannelSubviewHonorsAssignedUseLayout) {
  for (int64_t extent : {1024, 1025})
    for (int64_t channels : {24, 32}) {
      for (bool padded : {false, true}) {
        for (uint64_t workLimit : {0u, 100000u}) {
          SCOPED_TRACE(extent);
          SCOPED_TRACE(channels);
          SCOPED_TRACE(padded);
          SCOPED_TRACE(workLimit);
          const int64_t sourceHeight = padded ? 3 : 5;
          const int64_t sourceWidth = padded ? extent : extent + 2;
          const std::string sourceType =
              "tensor<1x" + std::to_string(sourceHeight) + "x" +
              std::to_string(sourceWidth) + "x64xf16>";
          const std::string sliceType = "tensor<1x" +
                                        std::to_string(sourceHeight) + "x" +
                                        std::to_string(sourceWidth) + "x" +
                                        std::to_string(channels) + "xf16>";
          const std::string inputType = "tensor<1x5x" +
                                        std::to_string(extent + 2) + "x" +
                                        std::to_string(channels) + "xf16>";
          const std::string weightType =
              "tensor<3x3x" + std::to_string(channels) + "x16xf16>";
          const std::string outputType =
              "tensor<1x3x" + std::to_string(extent) + "x16xf16>";
          std::string text;
          llvm::raw_string_ostream out(text);
          out << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
              << "func.func @entry(%image: " << sourceType
              << ", %weight: " << weightType << ") -> " << outputType << " {\n"
              << "%r = wafer.tile.region(%image, %weight : " << sourceType
              << ", " << weightType << ") -> (" << outputType << ") {\n"
              << "^bb0(%i: " << sourceType << ", %w: " << weightType << "):\n"
              << "%slice = tensor.extract_slice %i[0, 0, 0, 16] [1, "
              << sourceHeight << ", " << sourceWidth << ", " << channels
              << "] [1, 1, 1, 1] : " << sourceType << " to " << sliceType
              << "\n"
              << "%zero = arith.constant 0.0 : f16\n";
          if (padded)
            out << "%padded = tensor.pad %slice low[0, 1, 1, 0] high[0, 1, 1, "
                   "0] {\n"
                << "^bb0(%n: index, %h: index, %v: index, %c: index):\n"
                << "tensor.yield %zero : f16\n} : " << sliceType << " to "
                << inputType << "\n";
          out << "%empty = tensor.empty() : " << outputType << "\n"
              << "%init = linalg.fill ins(%zero : f16) outs(%empty : "
              << outputType << ") -> " << outputType << "\n"
              << "%conv = linalg.conv_2d_nhwc_hwcf ins(%"
              << (padded ? "padded" : "slice") << ", %w : " << inputType << ", "
              << weightType << ") outs(%init : " << outputType << ") -> "
              << outputType << "\n"
              << "wafer.tile.yield %conv : " << outputType
              << "\n}\nreturn %r : " << outputType << "\n}}}\n";
          auto module = parse(text);
          ASSERT_TRUE(module);
          auto relations = outputRelation(*module);
          if (padded) {
            auto unprepared = queryCurrentLayoutAssignment(*module);
            EXPECT_FALSE(unprepared.query);
            EXPECT_EQ(unprepared.outcome.status,
                      ExactPBQPStatus::BrokenContract);
            EXPECT_EQ(countOps<mlir::tensor::PadOp>(*module), 1u);
          }
          ASSERT_TRUE(
              prepareCurrentLayoutInput(*module, relations).succeeded());
          EXPECT_EQ(countOps<mlir::tensor::PadOp>(*module), 0u);
          auto query = queryCurrentLayoutAssignment(*module);
          ASSERT_TRUE(query.query) << query.outcome.detail;
          auto assignment = query.query->solve(workLimit);
          ASSERT_TRUE(assignment.status == ExactPBQPStatus::Optimal ||
                      assignment.status == ExactPBQPStatus::Feasible);
          auto applied = query.query->apply(*module, relations, assignment);
          ASSERT_TRUE(applied.succeeded()) << applied.detail;
          unsigned convolutions = 0;
          module->walk([&](mlir::linalg::Conv2DNhwcHwcfOp conv) {
            ++convolutions;
            auto type =
                mlir::cast<mlir::MemRefType>(conv.getInputs()[0].getType());
            EXPECT_EQ(getWaferMemoryAttr(type).getLayout(), MemLayout::NCx);
          });
          EXPECT_EQ(convolutions, 1u);
          auto lowered = lowerStructuredComputeToTile(*module, relations);
          ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
          EXPECT_EQ(countOps<ComputeConvOp>(*module), 1u);
        }
      }
    }
}

TEST_F(LayoutOptimizationTest, UniformGeneratePreservesBodyEffects) {
  auto module = parse(R"mlir(
module {
  func.func @entry(%output: memref<2x1025x32xf16>) -> tensor<2x1025x32xf16> {
    %value = tensor.generate {
    ^bb0(%b: index, %m: index, %n: index):
      %zero = arith.constant 0.0 : f16
      memref.store %zero, %output[%b, %m, %n] : memref<2x1025x32xf16>
      tensor.yield %zero : f16
    } : tensor<2x1025x32xf16>
    return %value : tensor<2x1025x32xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  mlir::IRRewriter rewriter(context.get());
  auto initialized = lowerUniformTensorInitializers(rewriter, *module);
  ASSERT_TRUE(mlir::succeeded(initialized));
  EXPECT_EQ(initialized->generates, 0u);
  EXPECT_EQ(countOps<mlir::tensor::GenerateOp>(*module), 1u);
  EXPECT_EQ(countOps<mlir::memref::StoreOp>(*module), 1u);
  EXPECT_EQ(countOps<mlir::linalg::FillOp>(*module), 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  StructuredMaterializationRelations relations;
  auto prepared = prepareCurrentLayoutInput(*module, relations);
  EXPECT_EQ(prepared.status, ExactPBQPStatus::NoSolution) << prepared.detail;
  EXPECT_EQ(countOps<mlir::memref::StoreOp>(*module), 1u);
}

TEST_F(LayoutOptimizationTest, UniformGenerateIsDPSBeforeLayoutAssignment) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (bool insideConstant : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(insideConstant);
      const std::string type =
          "tensor<2x" + std::to_string(extent) + "x32xf16>";
      std::string text = "module { wafer.tile.module card_id = 0 tile_id = 0 {"
                         "func.func @entry(%input: " +
                         type + ") -> " + type +
                         " {"
                         "%r = wafer.tile.region(%input : " +
                         type + ") -> (" + type +
                         ") {"
                         "^bb0(%arg: " +
                         type + "): ";
      if (!insideConstant)
        text += "%zero = arith.constant 0.0 : f16 ";
      text += "%g = tensor.generate { ^bb0(%b: index, %m: index, %n: index): ";
      if (insideConstant)
        text += "%zero = arith.constant 0.0 : f16 ";
      text += "tensor.yield %zero : f16 } : " + type +
              " %empty = tensor.empty() : " + type +
              " %sum = linalg.add ins(%arg, %g : " + type + ", " + type +
              ") outs(%empty : " + type + ") -> " + type +
              " wafer.tile.yield %sum : " + type + " } return %r : " + type +
              " } } }";
      auto module = parse(text);
      ASSERT_TRUE(module);
      auto relations = outputRelation(*module);
      // This is also a spatial-only region: no temporal pass is required to
      // expose the all-padding initializer before the layout query.
      auto unprepared = queryCurrentLayoutAssignment(*module);
      EXPECT_EQ(unprepared.outcome.status, ExactPBQPStatus::BrokenContract);
      ASSERT_FALSE(unprepared.query);
      auto prepared = prepareCurrentLayoutInput(*module, relations);
      ASSERT_TRUE(prepared.succeeded()) << prepared.detail;
      EXPECT_EQ(countOps<mlir::tensor::GenerateOp>(*module), 0u);
      EXPECT_EQ(countOps<mlir::linalg::FillOp>(*module), 1u);
      auto query = queryCurrentLayoutAssignment(*module);
      ASSERT_TRUE(query.query) << query.outcome.detail;
      auto assignment = query.query->solve(100000);
      auto applied = query.query->apply(*module, relations, assignment);
      ASSERT_TRUE(applied.succeeded()) << applied.detail;
      module->walk([&](mlir::linalg::FillOp fill) {
        auto buffer =
            mlir::cast<mlir::MemRefType>(fill.getOutputs()[0].getType());
        EXPECT_TRUE(isWaferSPMMemRefType(buffer));
        EXPECT_TRUE(buffer.getElementType().isF16());
        EXPECT_EQ(buffer.getNumElements(), 2 * extent * 32);
      });
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(countOps<ComputeFillOp>(*module), 1u);
      EXPECT_EQ(countOps<ComputeElementwiseOp>(*module), 1u);
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      ASSERT_EQ(standalone->size(), 1u);
      auto &tile = standalone->front();
      TileRegionToInstrLoweringSession session(*context);
      llvm::SmallVector<TileRegionOp, 2> regions;
      tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
      for (TileRegionOp region : regions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      TileMemoryPlanningFailure memoryFailure;
      auto planned = planTileMemory(std::move(tile.module), &memoryFailure);
      ASSERT_TRUE(mlir::succeeded(planned));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
    }
  }
}

TEST_F(LayoutOptimizationTest,
       LoopInvariantPlacementIsAnActualBufferizedChoice) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (bool localSource : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(localSource);
      const std::string output =
          "tensor<2x" + std::to_string(extent) + "x64xf16>";
      std::string source = makeSharedContractionSource(extent);
      size_t start =
          source.find(localSource ? "        %view =" : "        %empty0 =");
      ASSERT_NE(start, std::string::npos);
      source.insert(start,
                    "        %c0 = arith.constant 0 : index\n"
                    "        %c1 = arith.constant 1 : index\n"
                    "        %c3 = arith.constant 3 : index\n"
                    "        %initial = tensor.empty() : " +
                        output +
                        "\n"
                        "        %loop = scf.for %i = %c0 to %c3 step %c1 "
                        "iter_args(%state = %initial) -> (" +
                        output + ") {\n");
      size_t end = source.find("        wafer.tile.yield %sum");
      ASSERT_NE(end, std::string::npos);
      source.replace(end, std::string("        wafer.tile.yield %sum").size(),
                     "        scf.yield %sum : " + output +
                         "\n        }\n        wafer.tile.yield %loop");
      auto module = parse(source);
      ASSERT_TRUE(module);
      StructuredMaterializationRelations relations;
      ASSERT_TRUE(prepareCurrentLayoutInput(*module, relations).succeeded());
      auto query = queryCurrentLayoutAssignment(*module);
      ASSERT_TRUE(query.query) << query.outcome.detail;
      auto assignment = query.query->solve(100000);
      ASSERT_EQ(assignment.status, ExactPBQPStatus::Optimal);
      EXPECT_EQ(query.query->hasLoopInvariantPlacement(assignment),
                !localSource);
      for (auto placement : {LayoutMaterializationPlacement::FirstUse,
                             LayoutMaterializationPlacement::LoopInvariant}) {
        mlir::IRMapping mapping;
        mlir::OwningOpRef<mlir::ModuleOp> clone(
            mlir::cast<mlir::ModuleOp>(module->getOperation()->clone(mapping)));
        StructuredMaterializationRelations cloneRelations;
        auto result = query.query->apply(*clone, cloneRelations, assignment,
                                         &mapping, placement);
        ASSERT_TRUE(result.succeeded()) << result.detail;
        const bool hoist =
            !localSource &&
            placement == LayoutMaterializationPlacement::LoopInvariant;
        EXPECT_EQ(result.statistics.loopInvariantMaterializations > 0, hoist);
        unsigned inside = 0, outside = 0;
        clone->walk([&](LayoutMaterializeOp copy) {
          auto type = mlir::cast<mlir::MemRefType>(copy.getSource().getType());
          if (type.getShape() != llvm::ArrayRef<int64_t>({2, extent, 128}))
            return;
          if (copy->getParentOfType<mlir::scf::ForOp>())
            ++inside;
          else
            ++outside;
        });
        EXPECT_EQ(inside > 0, !hoist);
        EXPECT_EQ(outside > 0, hoist);
        auto lowered = lowerStructuredComputeToTile(*clone, cloneRelations);
        ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*clone)));
      }
    }
  }
}

TEST_F(LayoutOptimizationTest, AssignmentRejectsAnUnmappedOwnerBeforeMutation) {
  auto base = parse(makeSharedContractionSource(1025));
  ASSERT_TRUE(base);
  auto relations = outputRelation(*base);
  ASSERT_TRUE(prepareCurrentLayoutInput(*base, relations).succeeded());
  auto query = queryCurrentLayoutAssignment(*base);
  ASSERT_TRUE(query.query);
  auto assignment = query.query->solve(1048576);
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      mlir::cast<mlir::ModuleOp>(base->getOperation()->clone(mapping)));
  StructuredMaterializationRelations remapped;
  auto result = query.query->apply(*clone, remapped, assignment);
  EXPECT_EQ(result.status, ExactPBQPStatus::BrokenContract);
  EXPECT_EQ(result.statistics.bufferizationInvocations, 0u);
  EXPECT_EQ(countOps<mlir::linalg::BatchMatmulOp>(clone->getOperation()), 2u);
}

TEST_F(LayoutOptimizationTest,
       RecurrenceLayoutIsIndependentOfInitializerLayout) {
  for (llvm::StringRef dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025}) {
      // A bounded recurrence over realistic tensors isolates layout and entry
      // ownership. Product GEMM tests separately cover distinct K windows.
      std::string lhs =
          "tensor<2x" + std::to_string(extent) + "x64x" + dtype.str() + ">";
      std::string rhs = "tensor<2x64x64x" + dtype.str() + ">";
      std::string state = "tensor<2x" + std::to_string(extent) + "x64xf32>";
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "module { wafer.tile.module card_id = 0 tile_id = 0 { func.func "
             "@entry(%lhs: "
          << lhs << ", %rhs: " << rhs << ", %initial: " << state << ") -> "
          << state << " { "
          << "%result = wafer.tile.region(%lhs, %rhs, %initial : " << lhs
          << ", " << rhs << ", " << state << ") -> (" << state
          << ") { ^bb0(%a: " << lhs << ", %b: " << rhs << ", %init: " << state
          << "): "
          << "%c0 = arith.constant 0 : index %c64 = arith.constant 64 : index "
             "%end = arith.constant "
          << extent << " : index "
          << "%loop = scf.for %k = %c0 to %end step %c64 iter_args(%acc = "
             "%init) -> ("
          << state << ") { "
          << "%next = linalg.batch_matmul ins(%a, %b : " << lhs << ", " << rhs
          << ") outs(%acc : " << state << ") -> " << state
          << " scf.yield %next : " << state
          << " } wafer.tile.yield %loop : " << state
          << " } return %result : " << state << " } } }";
      auto module = parse(text);
      ASSERT_TRUE(module) << text;
      auto relations = outputRelation(*module);
      auto result = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(result.succeeded()) << result.detail;
      unsigned states = 0, stateRoundtrips = 0;
      module->walk([&](mlir::scf::ForOp loop) {
        for (auto argument : loop.getRegionIterArgs()) {
          auto type = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
          if (!type || !type.getElementType().isF32())
            continue;
          ++states;
          EXPECT_EQ(getWaferMemoryAttr(type).getLayout(), MemLayout::NCx);
        }
      });
      module->walk([&](LayoutMaterializeOp copy) {
        auto type = mlir::cast<mlir::MemRefType>(copy.getSource().getType());
        if (type.getElementType().isF32() &&
            copy->getParentOfType<mlir::scf::ForOp>())
          ++stateRoundtrips;
      });
      EXPECT_EQ(states, 1u);
      EXPECT_EQ(stateRoundtrips, 0u);
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}

TEST_F(LayoutOptimizationTest, PointwiseChainsRetainEveryLegalLayoutChoice) {
  for (int64_t extent : {1024, 1025, 1031})
    for (auto layout : {MemLayout::Tensor, MemLayout::NCx}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(static_cast<unsigned>(layout));
      std::string type = "tensor<2x" + std::to_string(extent) + "x64xf16>";
      std::string text;
      llvm::raw_string_ostream ir(text);
      ir << "#id = affine_map<(b,m,k)->(b,m,k)>\n"
         << "module { wafer.tile.module card_id = 0 tile_id = 0 { "
         << "func.func @entry(%unused: " << type << ") { "
         << "%r = wafer.tile.region(%unused : " << type << ") -> (" << type
         << ") { ^bb0(%a: " << type << "): "
         << "%one = arith.constant 1.0 : f16 "
         << "%v0 = bufferization.alloc_tensor() copy(%a) {memory_space = "
         << "#wafer.memory<spm, " << stringifyMemLayout(layout)
         << ">} : " << type;
      for (unsigned i = 1; i <= 3; ++i) {
        ir << " %e" << i << " = ";
        if (i == 3)
          ir << "bufferization.alloc_tensor() {memory_space = "
             << "#wafer.memory<spm, " << stringifyMemLayout(layout) << ">}";
        else
          ir << "tensor.empty()";
        ir << " : " << type << " %v" << i
           << " = linalg.generic {indexing_maps = [#id,#id], "
              "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
           << "ins(%v" << i - 1 << " : " << type << ") outs(%e" << i << " : "
           << type << ") { ^bb" << i << "(%x: f16, %old: f16): "
           << "%sum = arith.addf %x, %one : f16 linalg.yield %sum : f16 } -> "
           << type;
      }
      ir << " wafer.tile.yield %v3 : " << type << " } return } } }";
      auto module = parse(text);
      ASSERT_TRUE(module);
      auto relations = outputRelation(*module);
      auto result = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_EQ(result.status, ExactPBQPStatus::Optimal) << result.detail;
      EXPECT_EQ(result.statistics.layoutMaterializationsAfter,
                layout == MemLayout::Tensor ? 0u : 1u);
      unsigned pointwise = 0;
      module->walk([&](mlir::linalg::GenericOp op) {
        ++pointwise;
        auto type =
            mlir::cast<mlir::MemRefType>(op.getDpsInits().front().getType());
        auto source =
            mlir::cast<mlir::MemRefType>(op.getDpsInputs().front().getType());
        auto identity = analysis::IndexRelation::identity(type.getShape());
        ASSERT_TRUE(identity.isExact());
        EXPECT_TRUE(mlir::succeeded(
            analysis::TransferRealizability::provePhysicalTraversal(
                source, type, type.getShape(), *identity.get(),
                *identity.get())));
        // Aligned encodings can have identical element order. Ragged shapes
        // distinguish the blocked choice from the compact one.
        if (extent != 1024 || layout == MemLayout::Tensor) {
          EXPECT_EQ(getWaferMemoryAttr(type).getLayout(), layout);
        }
      });
      EXPECT_EQ(pointwise, 3u);
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(countOps<ComputeElementwiseOp>(*module), 3u);
      EXPECT_EQ(countOps<LayoutMaterializeOp>(*module),
                layout == MemLayout::Tensor ? 0u : 1u);
      module->walk([&](ComputeElementwiseOp op) {
        auto type = mlir::cast<mlir::MemRefType>(op.getResult().getType());
        if (extent != 1024 || layout == MemLayout::Tensor) {
          EXPECT_EQ(getWaferMemoryAttr(type).getLayout(), layout);
        }
        EXPECT_TRUE(mlir::isa<mlir::FloatType>(op.getInputs()[1].getType()));
      });
    }
}

TEST_F(LayoutOptimizationTest, PayloadDestinationsPreserveOldValueObservers) {
  for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031})
      for (bool permuted : {false, true})
        for (bool observeOld : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << dtype.str() << "/" << extent << "/" << observeOld
                       << "/" << permuted);
          auto type =
              "tensor<2x" + std::to_string(extent) + "x32x" + dtype.str() + ">";
          std::string text;
          llvm::raw_string_ostream ir(text);
          ir << "#id = affine_map<(b,m,n)->"
             << (permuted ? "(b,n,m)>\n" : "(b,m,n)>\n")
             << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
             << "func.func @entry(%input: " << type << ") {\n"
             << "%r = wafer.tile.region(%input : " << type << ") -> (" << type
             << ") { ^bb0(%a: " << type << "):\n"
             << "%factor = arith.constant 2.0 : " << dtype << "\n"
             << "%empty = tensor.empty() : " << type << "\n"
             << "%score = linalg.generic {indexing_maps = [#id, #id], "
                "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
                "ins(%a : "
             << type << ") outs(%empty : " << type << ") { ^bb1(%x: " << dtype
             << ", %old: " << dtype << "): %neg = arith.negf %x : " << dtype
             << " linalg.yield %neg : " << dtype << " } -> " << type << "\n"
             << "%result = linalg.generic {indexing_maps = [#id, #id], "
                "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
                "ins(%score : "
             << type << ") outs(%score : " << type << ") { ^bb1(%x: " << dtype
             << ", %old: " << dtype
             << "): %scaled = arith.mulf %x, %factor : " << dtype
             << " %sum = arith.addf %scaled, "
             << (observeOld ? "%x" : "%factor") << " : " << dtype
             << " linalg.yield %sum : " << dtype << " } -> " << type
             << "\nwafer.tile.yield %result : " << type << "\n}\nreturn\n}}}\n";
          auto module = parse(text);
          ASSERT_TRUE(module) << text;
          auto relations = outputRelation(*module);
          auto prepared = prepareCurrentLayoutInput(*module, relations);
          ASSERT_TRUE(prepared.succeeded()) << prepared.detail;
          unsigned multiplications = 0;
          module->walk([&](mlir::linalg::GenericOp op) {
            if (!mlir::isa<mlir::arith::MulFOp>(op.getBody()->front()))
              return;
            ++multiplications;
            EXPECT_EQ(op.getDpsInits().front(), op.getDpsInputs().front());
            EXPECT_TRUE(op.getIndexingMapsArray().front().isIdentity());
            EXPECT_TRUE(op.getIndexingMapsArray().back().isIdentity());
            EXPECT_EQ(mlir::cast<mlir::ShapedType>(op.getResult(0).getType())
                          .getShape(),
                      llvm::ArrayRef<int64_t>({2, extent, 32}));
          });
          EXPECT_EQ(multiplications, 1u);
          auto query = queryCurrentLayoutAssignment(*module);
          ASSERT_TRUE(query.query) << query.outcome.detail;
          auto assignment = query.query->solve(1048576);
          auto layout = query.query->apply(*module, relations, assignment);
          ASSERT_TRUE(layout.succeeded()) << layout.detail;
          mlir::linalg::GenericOp multiply, add;
          module->walk([&](mlir::linalg::GenericOp op) {
            if (mlir::isa<mlir::arith::MulFOp>(op.getBody()->front()))
              multiply = op;
            if (mlir::isa<mlir::arith::AddFOp>(op.getBody()->front()))
              add = op;
          });
          ASSERT_TRUE(multiply && add);
          mlir::AliasAnalysis aliases(module->getOperation());
          auto result = multiply.getDpsInits().front();
          if (observeOld) {
            EXPECT_TRUE(aliases.alias(result, add.getDpsInputs()[1]).isNo());
          } else {
            EXPECT_TRUE(
                aliases.alias(result, add.getDpsInits().front()).isMust());
          }
          auto lowered = lowerStructuredComputeToTile(*module, relations);
          ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
          auto movement = materializeTileBoundaryMovement(*module, relations);
          ASSERT_TRUE(movement.succeeded()) << movement.detail;
          std::string detail;
          auto standalone = createStandaloneTileModules(std::move(module),
                                                        &detail, &relations);
          ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
          ASSERT_EQ(standalone->size(), 1u);
          auto &tile = standalone->front();
          TileRegionToInstrLoweringSession session(*context);
          llvm::SmallVector<TileRegionOp> regions;
          tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
          for (auto region : regions)
            ASSERT_TRUE(
                mlir::succeeded(convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(
              convertBufferizationCopiesToInstr(*tile.module, session)));
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
          TileMemoryPlanningFailure failure;
          auto planned = planTileMemory(std::move(tile.module), &failure);
          ASSERT_TRUE(mlir::succeeded(planned));
        }
}

TEST_F(LayoutOptimizationTest, PayloadSSAExposesCompactPredicatesAndCasts) {
  for (auto element : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(extent);
      std::string shape = "2x" + std::to_string(extent);
      std::string input = "tensor<" + shape + "x64xf16>";
      std::string row = "tensor<" + shape + "xf32>";
      std::string text;
      llvm::raw_string_ostream ir(text);
      ir << "#id = affine_map<(b,m,k)->(b,m,k)>\n"
         << "#row = affine_map<(b,m,k)->(b,m)>\n"
         << "module { wafer.tile.module card_id = 0 tile_id = 0 { "
         << "func.func @entry(%arg: " << input << ", %row: " << row << ") { "
         << "%r = wafer.tile.region(%arg, %row : " << input << ", " << row
         << ") -> (" << input << ") { ^bb0(%a: " << input << ", %b: " << row
         << "): %z = arith.constant 0.0 : f32 "
         << "%e = tensor.empty() : " << input
         << " %v = linalg.generic {indexing_maps = [#id,#row,#id], "
            "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
         << "ins(%a, %b : " << input << ", " << row << ") outs(%e : " << input
         << ") { ^bb1(%x: f16, %y: f32, %old: f16): "
         << "%scalar_exp = math.exp %z : f32 "
         << "%c = arith.cmpf ogt, %y, %z : f32 "
         << "%wide = arith.extf %x : f16 to f32 "
         << "%sub = arith.subf %wide, %y : f32 "
         << "%exp = math.exp %sub : f32 "
         << "%masked = arith.select %c, %exp, %scalar_exp : f32 "
         << "%narrow = arith.truncf %masked : f32 to f16 "
         << "linalg.yield %narrow : f16 } -> " << input
         << " wafer.tile.yield %v : " << input << " } return } } }";
      for (size_t at = 0; (at = text.find("f16", at)) != std::string::npos;) {
        text.replace(at, 3, element);
        at += std::char_traits<char>::length(element);
      }
      SCOPED_TRACE(element);
      auto module = parse(text);
      ASSERT_TRUE(module);
      auto relations = outputRelation(*module);
      auto unprepared = queryCurrentLayoutAssignment(*module);
      EXPECT_FALSE(unprepared.query);
      auto prepared = prepareCurrentLayoutInput(*module, relations);
      ASSERT_TRUE(prepared.succeeded()) << prepared.detail;
      unsigned steps = 0, predicates = 0, scalarComputes = 0;
      module->walk([&](mlir::linalg::GenericOp op) {
        ++steps;
        EXPECT_EQ(op.getBody()->getOperations().size(), 2u);
        auto type =
            mlir::cast<mlir::RankedTensorType>(op.getResult(0).getType());
        if (type.getRank() == 0) {
          ++scalarComputes;
          EXPECT_TRUE(mlir::isa<mlir::math::ExpOp>(op.getBody()->front()));
        }
        if (type.getElementType().isInteger(1)) {
          ++predicates;
          EXPECT_EQ(type.getShape(), (llvm::ArrayRef<int64_t>{2, extent}));
        }
      });
      EXPECT_EQ(steps, 7u);
      EXPECT_EQ(predicates, 1u);
      EXPECT_EQ(scalarComputes, 1u);
      EXPECT_FALSE(hasCPUScalarAlternative(*module));
      auto query = queryCurrentLayoutAssignment(*module);
      ASSERT_TRUE(query.query) << query.outcome.detail;
      auto assignment = query.query->solve(1048576);
      auto layout = query.query->apply(*module, relations, assignment);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(countOps<ComputeConvertOp>(*module), 2u);
      EXPECT_EQ(countOps<ComputeElementwiseOp>(*module), 5u);
      EXPECT_EQ(countOps<LayoutMaterializeOp>(*module),
                layout.statistics.layoutMaterializationsAfter);
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      ASSERT_EQ(standalone->size(), 1u);
      auto &tile = standalone->front();
      TileRegionToInstrLoweringSession session(*context);
      llvm::SmallVector<TileRegionOp> regions;
      tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
      for (auto region : regions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      EXPECT_EQ(countOps<InstrConvertOp>(*tile.module), 2u);
      EXPECT_GT(countOps<InstrBit2FpOp>(*tile.module), 0u);
      TileMemoryPlanningFailure failure;
      auto planned = planTileMemory(std::move(tile.module), &failure);
      ASSERT_TRUE(mlir::succeeded(planned));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
    }
}

TEST_F(LayoutOptimizationTest, CPUScalarAlternativeKeepsActualCostAndStorage) {
  for (auto element : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(element);
      SCOPED_TRACE(extent);
      std::string type =
          "tensor<2x" + std::to_string(extent) + "x64x" + element + ">";
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "#id = affine_map<(b,m,k)->(b,m,k)>\n"
          << "module { wafer.tile.module card_id = 0 tile_id = 0 { "
          << "func.func @entry(%arg: " << type << ", %scale: f32) { "
          << "%r = wafer.tile.region(%arg, %scale : " << type << ", f32) -> ("
          << type << ") { ^bb0(%a: " << type
          << ", %s: f32): %empty = tensor.empty() : " << type
          << " %r = linalg.generic {indexing_maps = [#id,#id], "
             "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
          << "ins(%a : " << type << ") outs(%empty : " << type
          << ") { ^bb1(%x: " << element << ", %old: " << element << "): "
          << "%product = arith.mulf %s, %s : f32 ";
      if (llvm::StringRef(element) != "f32")
        out << "%wide = arith.extf %x : " << element << " to f32 ";
      out << "%added = arith.addf %"
          << (llvm::StringRef(element) == "f32" ? "x" : "wide")
          << ", %product : f32 ";
      if (llvm::StringRef(element) != "f32")
        out << "%narrow = arith.truncf %added : f32 to " << element << " ";
      out << "linalg.yield %"
          << (llvm::StringRef(element) == "f32" ? "added" : "narrow") << " : "
          << element << " } -> " << type << " wafer.tile.yield %r : " << type
          << " } return } } }";
      std::array<mlir::OwningOpRef<mlir::ModuleOp>, 2> actual;
      for (bool cpu : {false, true}) {
        auto module = parse(text);
        ASSERT_TRUE(module) << text;
        auto relations = outputRelation(*module);
        auto prepared = prepareCurrentLayoutInput(*module, relations);
        ASSERT_TRUE(prepared.succeeded()) << prepared.detail;
        ASSERT_TRUE(hasCPUScalarAlternative(*module));
        if (cpu) {
          auto materialized =
              materializeCPUScalarAlternative(*module, relations);
          ASSERT_TRUE(mlir::succeeded(materialized));
          EXPECT_EQ(*materialized, 1u);
          EXPECT_FALSE(hasCPUScalarAlternative(*module));
        }
        auto query = queryCurrentLayoutAssignment(*module);
        ASSERT_TRUE(query.query) << query.outcome.detail;
        auto assignment = query.query->solve(1048576);
        auto layout = query.query->apply(*module, relations, assignment);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        auto lowered = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
        EXPECT_EQ(countOps<ComputeElementwiseOp>(*module), cpu ? 1u : 2u);
        auto movement = materializeTileBoundaryMovement(*module, relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        std::string detail;
        auto standalone =
            createStandaloneTileModules(std::move(module), &detail, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
        auto &tile = standalone->front();
        TileRegionToInstrLoweringSession session(*context);
        llvm::SmallVector<TileRegionOp> regions;
        tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
        for (auto region : regions)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*tile.module, session)));
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
        EXPECT_EQ(countOps<mlir::memref::LoadOp>(*tile.module), 0u);
        TileMemoryPlanningFailure failure;
        auto planned = planTileMemory(std::move(tile.module), &failure);
        ASSERT_TRUE(mlir::succeeded(planned));
        actual[cpu] = std::move(*planned);
      }
      for (uint64_t cpuPrior : {uint64_t{1}, uint64_t{1'000'000'000}}) {
        analysis::SearchCostPolicy policy;
        policy.cpuScalarOperationPicosecondsEstimate = cpuPrior;
        auto cohort = *analysis::SearchCostCohort::create(policy);
        std::array<analysis::SearchObjective, 2> costs{
            analysis::UnknownSearchObjective{},
            analysis::UnknownSearchObjective{}};
        for (bool cpu : {false, true}) {
          llvm::SmallVector<analysis::TileInstructionProgram> programs{
              {TileId(0), *actual[cpu]}};
          auto cost = analysis::analyzeInstructionProgramAggregateCost(
              programs, getTargetMemoryPolicy());
          ASSERT_TRUE(
              cost.aggregateWork.cpuScalarOperations.exactExecutions.isKnown());
          EXPECT_EQ(
              cost.aggregateWork.cpuScalarOperations.exactExecutions.value,
              cpu ? 1u : 0u);
          ASSERT_TRUE(cost.aggregateCompute.vectorOtherLogicalOps.isKnown());
          EXPECT_EQ(cost.aggregateCompute.vectorOtherLogicalOps.value, 0u);
          costs[cpu] = analysis::deriveSearchObjective(cost, cohort, programs);
        }
        // F32 CT work ties under the low CPU prior; widening the narrow
        // alternatives retains a cost. The high prior reverses the choice.
        EXPECT_EQ(analysis::compareSearchObjectives(costs[1], costs[0]),
                  cpuPrior != 1
                      ? analysis::SearchObjectiveComparison::Worse
                      : llvm::StringRef(element) == "f32"
                            ? analysis::SearchObjectiveComparison::Equivalent
                            : analysis::SearchObjectiveComparison::Better);
      }
    }
}

TEST_F(LayoutOptimizationTest,
       CPUScalarAlternativeRejectsStorageAndUnsupportedMath) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned variant : {0, 1, 2, 3, 4, 5, 6, 7, 8}) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << variant);
      std::string type = "tensor<2x" + std::to_string(extent) + "x64xf32>";
      std::string text;
      llvm::raw_string_ostream out(text);
      out << "#id = affine_map<(b,m,k)->(b,m,k)>\n"
             "#scalar = affine_map<(b,m,k)->()>\n"
             "#zero = affine_map<()->()>\nmodule { "
             "wafer.tile.module card_id = 0 tile_id = 0 { "
             "func.func private @unknown(f32) -> f32\n"
             "func.func @entry(%arg: "
          << type << ", %scale: f32, %memory: tensor<f32>) { "
          << "%result = wafer.tile.region(%arg, %scale, %memory : " << type
          << ", f32, tensor<f32>) -> (" << type << ") { ^bb0(%a: " << type
          << ", %s: f32, %mem: tensor<f32>): ";
      if (variant == 1)
        out << "%seed = tensor.extract %mem[] : tensor<f32> ";
      else if (variant == 2)
        out << "%seed = func.call @unknown(%s) : (f32) -> f32 ";
      out << "%empty = tensor.empty() : tensor<f32> "
             "%scalar = linalg.generic {indexing_maps = [#zero], "
             "iterator_types = []} outs(%"
          << (variant == 5 ? "mem" : "empty")
          << " : tensor<f32>) { ^bb1(%old: f32): %value = ";
      if (variant == 3)
        out << "math.rsqrt %s : f32 ";
      else
        out << "arith.addf %" << (variant == 1 || variant == 2 ? "seed" : "s")
            << ", %s : f32 ";
      out << "linalg.yield %" << (variant == 8 ? "s" : "value")
          << " : f32 } -> tensor<f32> ";
      if (variant == 4)
        out << "%escape = tensor.extract %scalar[] : tensor<f32> ";
      out << "%destination = tensor.empty() : " << type
          << " %result = linalg.generic {indexing_maps = [#id,#scalar,#id], "
             "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
             "ins(%a, %scalar : "
          << type << ", tensor<f32>) outs(%destination : " << type
          << ") { ^bb2(%x: f32, %rhs: f32, %old: f32): "
             "%sum = arith.addf %x, %rhs : f32 linalg.yield %sum : f32 } -> "
          << type << " wafer.tile.yield %result : " << type
          << " } return } } }";
      if (variant == 6 || variant == 7) {
        llvm::StringRef dtype = variant == 6 ? "f16" : "bf16";
        for (size_t offset = 0;
             (offset = text.find("f32", offset)) != std::string::npos;) {
          text.replace(offset, 3, dtype.str());
          offset += dtype.size();
        }
      }
      auto module = parse(text);
      ASSERT_TRUE(module) << text;
      auto relations = outputRelation(*module);
      EXPECT_EQ(hasCPUScalarAlternative(*module), variant == 0);
      std::string before, after;
      llvm::raw_string_ostream beforeStream(before), afterStream(after);
      module->print(beforeStream);
      auto materialized = materializeCPUScalarAlternative(*module, relations);
      ASSERT_TRUE(mlir::succeeded(materialized));
      EXPECT_EQ(*materialized, variant == 0 ? 1u : 0u);
      module->print(afterStream);
      if (variant) {
        EXPECT_EQ(before, after);
      }
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}

TEST_F(LayoutOptimizationTest, CPUScalarChainsPreserveLoopScopeAndSharedUses) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    const unsigned trips = (extent + 31) / 32;
    std::string type = "tensor<2x" + std::to_string(extent) + "x16xf32>";
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "#id = affine_map<(b,m,k)->(b,m,k)>\n"
           "#scalar = affine_map<(b,m,k)->()>\n#zero = affine_map<()->()>\n"
           "module { wafer.tile.module card_id = 0 tile_id = 0 { "
           "func.func @entry(%arg: "
        << type << ") { %result = wafer.tile.region(%arg : " << type << ") -> ("
        << type << ") { ^bb0(%a: " << type
        << "): %c0 = arith.constant 0 : index "
           "%c1 = arith.constant 1 : index %end = arith.constant "
        << trips << " : index %loop = scf.for %i = %c0 to %end step %c1 "
        << "iter_args(%state = %a) -> " << type
        << " { "
           "%integer = arith.index_cast %i : index to i64 "
           "%s = arith.sitofp %integer : i64 to f32 "
           "%empty0 = tensor.empty() : tensor<f32> "
           "%first = linalg.generic {indexing_maps = [#zero], "
           "iterator_types = []} outs(%empty0 : tensor<f32>) { "
           "^bb1(%old: f32): %product = arith.mulf %s, %s : f32 "
           "linalg.yield %product : f32 } -> tensor<f32> "
           "%empty1 = tensor.empty() : tensor<f32> "
           "%second = linalg.generic {indexing_maps = [#zero,#zero], "
           "iterator_types = []} ins(%first : tensor<f32>) "
           "outs(%empty1 : tensor<f32>) { ^bb2(%v: f32, %old: f32): "
           "%difference = arith.subf %v, %s : f32 "
           "linalg.yield %difference : f32 } -> tensor<f32> ";
    for (unsigned index : {0, 1})
      out << "%empty" << index + 2 << " = tensor.empty() : " << type
          << " %value" << index
          << " = linalg.generic {indexing_maps = [#id,#scalar,#id], "
             "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
             "ins(%"
          << (index ? "value0" : "state") << ", %second : " << type
          << ", tensor<f32>) outs(%empty" << index + 2 << " : " << type
          << ") { ^bb3(%x: f32, %rhs: f32, %old: f32): "
             "%next = arith.addf %x, %rhs : f32 linalg.yield %next : f32 } -> "
          << type << " ";
    out << "scf.yield %value1 : " << type
        << " } wafer.tile.yield %loop : " << type << " } return } } }";
    auto module = parse(text);
    ASSERT_TRUE(module) << text;
    auto relations = outputRelation(*module);
    auto materialized = materializeCPUScalarAlternative(*module, relations);
    ASSERT_TRUE(mlir::succeeded(materialized));
    EXPECT_EQ(*materialized, 2u);
    unsigned arithmetic = 0;
    mlir::Value shared;
    module->walk([&](mlir::arith::MulFOp op) {
      EXPECT_TRUE(op->getParentOfType<mlir::scf::ForOp>());
      EXPECT_FALSE(op->getParentOfType<mlir::linalg::GenericOp>());
      ++arithmetic;
    });
    module->walk([&](mlir::arith::SubFOp op) {
      EXPECT_TRUE(op->getParentOfType<mlir::scf::ForOp>());
      EXPECT_FALSE(op->getParentOfType<mlir::linalg::GenericOp>());
      shared = op.getResult();
      ++arithmetic;
    });
    EXPECT_EQ(arithmetic, 2u);
    ASSERT_TRUE(shared);
    EXPECT_EQ(std::distance(shared.use_begin(), shared.use_end()), 2);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(*module), 1u);
    EXPECT_EQ(countOps<mlir::linalg::GenericOp>(*module), 2u);
    auto layout =
        resolveCurrentLayoutsAndBufferize(*module, relations, 1048576);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    auto lowered = lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    EXPECT_EQ(countOps<ComputeElementwiseOp>(*module), 2u);
    EXPECT_EQ(countOps<mlir::memref::LoadOp>(*module), 0u);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    auto movement = materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    std::string detail;
    auto standalone =
        createStandaloneTileModules(std::move(module), &detail, &relations);
    ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
    auto &tile = standalone->front();
    TileRegionToInstrLoweringSession session(*context);
    llvm::SmallVector<TileRegionOp> regions;
    tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
    for (auto region : regions)
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
    ASSERT_TRUE(mlir::succeeded(
        convertBufferizationCopiesToInstr(*tile.module, session)));
    ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
    auto work = analysis::analyzeInstructionProgramCost(
        *tile.module, getTargetMemoryPolicy());
    EXPECT_TRUE(work.work.cpuScalarOperations.exactExecutions.isKnown());
    EXPECT_EQ(work.work.cpuScalarOperations.exactExecutions.value, 4u * trips);
    TileMemoryPlanningFailure failure;
    auto planned = planTileMemory(std::move(tile.module), &failure);
    ASSERT_TRUE(mlir::succeeded(planned));
    EXPECT_EQ(countOps<mlir::memref::LoadOp>(**planned), 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
  }
}

TEST_F(LayoutOptimizationTest, BlockedPredicateUsesSupportedMappedTraversal) {
  for (auto element : {"f16", "bf16", "f32"})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(element);
      SCOPED_TRACE(extent);
      std::string type =
          "tensor<1x" + std::to_string(extent) + "x128x" + element + ">";
      std::string text;
      llvm::raw_string_ostream ir(text);
      ir << "#id = affine_map<(b,m,k)->(b,m,k)>\n"
         << "module { wafer.tile.module card_id = 0 tile_id = 0 { "
         << "func.func @entry(%arg: " << type << ") { "
         << "%r = wafer.tile.region(%arg : " << type << ") -> (" << type
         << ") { ^bb0(%a: " << type << "): "
         << "%zero = arith.constant 0.0 : " << element
         << " %input = bufferization.alloc_tensor() copy(%a) "
            "{memory_space = #wafer.memory<spm, ncx>} : "
         << type
         << " %e = bufferization.alloc_tensor() "
            "{memory_space = #wafer.memory<spm, ncx>} : "
         << type
         << " %v = linalg.generic {indexing_maps = [#id,#id], "
            "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
         << "ins(%input : " << type << ") outs(%e : " << type
         << ") { ^bb1(%x: " << element << ", %old: " << element << "): "
         << "%predicate = arith.cmpf ogt, %x, %zero : " << element
         << " %masked = arith.select %predicate, %zero, %x : " << element
         << " linalg.yield %masked : " << element << " } -> " << type
         << " wafer.tile.yield %v : " << type << " } return } } }";
      auto module = parse(text);
      ASSERT_TRUE(module);
      auto relations = outputRelation(*module);
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(countOps<ComputeElementwiseOp>(*module), 2u);
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      ASSERT_EQ(standalone->size(), 1u);
      auto &tile = standalone->front();
      TileRegionToInstrLoweringSession session(*context);
      llvm::SmallVector<TileRegionOp> regions;
      tile.module->walk([&](TileRegionOp op) { regions.push_back(op); });
      for (auto region : regions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      EXPECT_EQ(countOps<InstrBit2FpOp>(*tile.module), 1u);
      EXPECT_EQ(countOps<InstrMaskMoveOp>(*tile.module), 1u);
      tile.module->walk([&](InstrBit2FpOp op) {
        auto source = mlir::cast<mlir::MemRefType>(op.getSource().getType());
        auto destination = mlir::cast<mlir::MemRefType>(op.getDest().getType());
        auto identity = analysis::IndexRelation::identity(source.getShape());
        ASSERT_TRUE(identity.isExact());
        EXPECT_TRUE(mlir::succeeded(
            analysis::TransferRealizability::provePhysicalTraversal(
                source, destination, source.getShape(), *identity.get(),
                *identity.get())));
      });
      TileMemoryPlanningFailure failure;
      auto planned = planTileMemory(std::move(tile.module), &failure);
      ASSERT_TRUE(mlir::succeeded(planned));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
    }
}

TEST_F(LayoutOptimizationTest, CastSeparatesCoordinatesAndFixedPublication) {
  for (int64_t extent : {1024, 1031})
    for (bool transpose : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(transpose);
      std::string input = "tensor<2x" +
                          (transpose ? "64x" + std::to_string(extent)
                                     : std::to_string(extent) + "x64") +
                          "xf16>";
      std::string output = "tensor<2x" + std::to_string(extent) + "x64xf32>";
      std::string text;
      llvm::raw_string_ostream ir(text);
      ir << "module { wafer.tile.module card_id = 0 tile_id = 0 { "
         << "func.func @entry(%arg: " << input << ") { "
         << "%r = wafer.tile.region(%arg : " << input << ") -> (" << output
         << ") { ^bb0(%a: " << input << "): "
         << "%e = bufferization.alloc_tensor() {memory_space = "
            "#wafer.memory<spm, ncx>} : "
         << output
         << " %v = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,"
         << (transpose ? "k,m" : "m,k")
         << ")>,affine_map<(b,m,k)->(b,m,k)>], "
            "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
         << "ins(%a : " << input << ") outs(%e : " << output
         << ") { ^bb1(%x: f16, %old: f32): "
         << "%wide = arith.extf %x : f16 to f32 "
         << "linalg.yield %wide : f32 } -> " << output
         << " wafer.tile.yield %v : " << output << " } return } } }";
      auto module = parse(text);
      ASSERT_TRUE(module);
      auto relations = outputRelation(*module);
      auto prepared = prepareCurrentLayoutInput(*module, relations);
      ASSERT_TRUE(prepared.succeeded()) << prepared.detail;
      EXPECT_EQ(countOps<mlir::linalg::GenericOp>(*module), 2u);
      auto query = queryCurrentLayoutAssignment(*module);
      ASSERT_TRUE(query.query) << query.outcome.detail;
      auto assignment = query.query->solve(1048576);
      auto layout = query.query->apply(*module, relations, assignment);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(countOps<ComputeConvertOp>(*module), 1u);
      module->walk([&](ComputeConvertOp op) {
        auto source = mlir::cast<mlir::MemRefType>(op.getSource().getType());
        auto result = mlir::cast<mlir::MemRefType>(op.getResult().getType());
        EXPECT_EQ(source.getShape(), result.getShape());
        auto identity = analysis::IndexRelation::identity(result.getShape());
        ASSERT_TRUE(identity.isExact());
        EXPECT_TRUE(mlir::succeeded(
            analysis::TransferRealizability::provePhysicalTraversal(
                source, result, result.getShape(), *identity.get(),
                *identity.get())));
      });
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      TileRegionToInstrLoweringSession session(*context);
      llvm::SmallVector<TileRegionOp> regions;
      module->walk([&](TileRegionOp region) { regions.push_back(region); });
      for (auto region : regions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      EXPECT_EQ(countOps<InstrConvertOp>(*module), 1u);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}

} // namespace
