//===- LayoutOptimizationTest.cpp --------------------------------------===//

#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/LoopSubsetState.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

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

TEST_F(LayoutOptimizationTest, SegmentedLoopsCarryOnlyTheirUpdatedOutputSubset) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    const std::string type =
        "tensor<1x" + std::to_string(extent) + "x64xf16>";
    constexpr unsigned segments = 32;
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
        << "func.func @entry(%input: " << type << ") {\n"
        << "%result = wafer.tile.region(%input : " << type << ") -> ("
        << type << ") { ^bb0(%local: " << type << "):\n"
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
          << initial << ") -> " << type << " {\n"
          << "%slice = tensor.extract_slice %tile[0, %i, 0] [1, 32, 64] "
          << "[1, 1, 1] : " << type << " to tensor<1x32x64xf16>\n"
          << "%sum = linalg.add ins(%slice, %slice : tensor<1x32x64xf16>, "
          << "tensor<1x32x64xf16>) outs(%slice : tensor<1x32x64xf16>) "
          << "-> tensor<1x32x64xf16>\n"
          << "%updated = tensor.insert_slice %sum into %tile[0, %i, 0] "
          << "[1, 32, 64] [1, 1, 1] : tensor<1x32x64xf16> into " << type
          << "\nscf.yield %updated : " << type << "\n}\n";
    }
    out << "scf.yield %loop" << segments - 1 << " : " << type << "\n}\n"
        << "wafer.tile.yield %outer : " << type << "\n}\nreturn\n}}}\n";
    auto module = parse(text);
    ASSERT_TRUE(module);
    auto relations = outputRelation(*module);
    ASSERT_TRUE(mlir::succeeded(normalizeLoopSubsetState(*module, relations)));
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
    EXPECT_EQ(localStates, segments);
    EXPECT_EQ(fullStates, 1u);
    // Adjacent segment handoffs fold to one extraction/writeback per output
    // tile. The final 1/7 rows remain the original state, not an empty tensor.
    EXPECT_EQ(countOps<mlir::tensor::ExtractSliceOp>(*module), 1u);
    EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(*module), 1u);
    mlir::tensor::InsertSliceOp writeback;
    module->walk([&](mlir::tensor::InsertSliceOp op) { writeback = op; });
    ASSERT_TRUE(writeback);
    auto outer = writeback->getParentOfType<mlir::scf::ForOp>();
    ASSERT_TRUE(outer);
    EXPECT_EQ(writeback.getDest(), outer.getRegionIterArgs().front());
    EXPECT_EQ(writeback.getStaticSizes(), (llvm::ArrayRef<int64_t>{1, 32, 64}));
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

TEST_F(LayoutOptimizationTest, SubsetPromotionPreservesUnprovedLoopState) {
  for (llvm::StringRef boundary :
       {"empty", "dynamic", "variant", "observed", "fork", "nested"}) {
    SCOPED_TRACE(boundary.str());
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
        << "func.func @entry(%input: tensor<1x1031x64xf16>, %limit: index) {\n"
        << "%result = wafer.tile.region(%input, %limit : tensor<1x1031x64xf16>, index) "
        << "-> (tensor<1x1031x64xf16>) { ^bb0(%local: tensor<1x1031x64xf16>, %bound: index):\n"
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
    auto result = resolveCurrentLayoutsAndBufferize(*module, relations, 0);
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
  EXPECT_GT(firstResult.statistics.dominatedLayoutStatesPruned, 0u);
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
  RecordProperty(
      "layout_dominated_states_pruned",
      static_cast<int>(firstResult.statistics.dominatedLayoutStatesPruned));
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
  EXPECT_GT(firstResult.statistics.dominatedLayoutStatesPruned, 0u);
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
       NonPieceOutputFailsBeforeMutationRegardlessOfSolverBudget) {
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
  EXPECT_EQ(noBudget.status, ExactPBQPStatus::NoSolution);
  LayoutOptimizationResult unsupported =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  EXPECT_EQ(unsupported.status, ExactPBQPStatus::NoSolution);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
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

} // namespace
