//===- StructuredToTileTest.cpp ---------------------------------------===//

#include "Wafer/Transforms/Tile/StructuredToTile.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Linalg/CommunicationRegionClosure.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
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

class StructuredToTileTest : public ::testing::Test {
protected:
  StructuredToTileTest() {
    registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        text, mlir::ParserConfig(context.get()));
  }

  static std::string makeSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id3 = affine_map<(b, m, n) -> (b, m, n)>
#row = affine_map<(b, m, n) -> (b, m)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%lhs: tensor<2x)mlir"
           << extent
           << R"mlir(x64xf16>, %rhs: tensor<2x64x32xf16>) -> (tensor<2x)mlir"
           << extent << R"mlir(x32xf16>, tensor<2x)mlir" << extent
           << R"mlir(xf16>) {
      %result, %row_result = wafer.tile.region(
          %lhs, %rhs : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>, tensor<2x64x32xf16>)
          -> (tensor<2x)mlir"
           << extent << R"mlir(x32xf16>, tensor<2x)mlir" << extent
           << R"mlir(xf16>) {
      ^bb0(%local_lhs: tensor<2x)mlir"
           << extent << R"mlir(x64xf16>, %local_rhs: tensor<2x64x32xf16>):
        %zero = arith.constant 0.000000e+00 : f16
        %mm_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>
        %mm_init = linalg.fill ins(%zero : f16)
            outs(%mm_empty : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>) -> tensor<2x)mlir" << extent
           << R"mlir(x32xf16>
        %mm = linalg.batch_matmul
            ins(%local_lhs, %local_rhs : tensor<2x)mlir"
           << extent << R"mlir(x64xf16>, tensor<2x64x32xf16>)
            outs(%mm_init : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>) -> tensor<2x)mlir" << extent
           << R"mlir(x32xf16>
        %mapped_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id3, #id3, #id3],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%mm, %mm : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>, tensor<2x)mlir" << extent
           << R"mlir(x32xf16>)
            outs(%mapped_empty : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>) {
          ^bb1(%lhs_value: f16, %rhs_value: f16, %old: f16):
            %difference = arith.subf %lhs_value, %rhs_value : f16
            %value = math.exp %difference : f16
            linalg.yield %value : f16
        } -> tensor<2x)mlir"
           << extent << R"mlir(x32xf16>
        %row_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(xf16>
        %row_init = linalg.fill ins(%zero : f16)
            outs(%row_empty : tensor<2x)mlir"
           << extent << R"mlir(xf16>) -> tensor<2x)mlir" << extent
           << R"mlir(xf16>
        %reduced = linalg.generic {
            indexing_maps = [#id3, #row],
            iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%mapped : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>)
            outs(%row_init : tensor<2x)mlir"
           << extent << R"mlir(xf16>) {
          ^bb1(%value: f16, %acc: f16):
            %next = arith.addf %value, %acc : f16
            linalg.yield %next : f16
        } -> tensor<2x)mlir"
           << extent << R"mlir(xf16>
        wafer.tile.yield %mapped, %reduced
            : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>, tensor<2x)mlir" << extent
           << R"mlir(xf16>
      }
      return %result, %row_result : tensor<2x)mlir"
           << extent << R"mlir(x32xf16>, tensor<2x)mlir" << extent
           << R"mlir(xf16>
    }
  }
}
)mlir";
    return text;
  }

  static std::string makeConvSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream
        << R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<1x)mlir"
        << extent + 2
        << R"mlir(x66x16xf16>, %weight: tensor<3x3x16x32xf16>) -> tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16> {
      %result = wafer.tile.region(
          %input, %weight : tensor<1x)mlir"
        << extent + 2 << R"mlir(x66x16xf16>, tensor<3x3x16x32xf16>)
          -> (tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>) {
      ^bb0(%local_input: tensor<1x)mlir"
        << extent + 2
        << R"mlir(x66x16xf16>, %local_weight: tensor<3x3x16x32xf16>):
        %zero = arith.constant 0.000000e+00 : f16
        %empty = tensor.empty() : tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>
        %init = linalg.fill ins(%zero : f16)
            outs(%empty : tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>) -> tensor<1x)mlir" << extent
        << R"mlir(x64x32xf16>
        %conv = linalg.conv_2d_nhwc_hwcf
            ins(%local_input, %local_weight : tensor<1x)mlir"
        << extent + 2 << R"mlir(x66x16xf16>, tensor<3x3x16x32xf16>)
            outs(%init : tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>) -> tensor<1x)mlir" << extent
        << R"mlir(x64x32xf16>
        wafer.tile.yield %conv : tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>
      }
      return %result : tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>
    }
  }
}
)mlir";
    return text;
  }

  static std::string makeTwoRegionSource(int64_t extent, bool crossTile) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @source(%input: tensor<1x)mlir"
           << extent << "x64xf16>)";
    if (!crossTile)
      stream << " -> tensor<1x" << extent << "x64xf16>";
    stream << R"mlir( {
      %first = wafer.tile.region(
          %input : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
           << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
           << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            linalg.yield %value : f16
        } -> tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
      }
)mlir";
    if (!crossTile) {
      stream << R"mlir(      %second = wafer.tile.region(
          %first : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
             << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.mulf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
      }
)mlir";
    }
    if (crossTile)
      stream << R"mlir(      return
)mlir";
    else
      stream << "      return %second : tensor<1x" << extent << "x64xf16>\n";
    stream << R"mlir(    }
  }
)mlir";
    if (crossTile) {
      stream << R"mlir(  wafer.tile.module card_id = 0 tile_id = 1 {
    func.func @destination(%placeholder: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> tensor<1x)mlir" << extent
             << R"mlir(x64xf16> {
      %second = wafer.tile.region(
          %placeholder : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
             << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.mulf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
      }
      return %second : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
    }
  }
)mlir";
    }
    stream << "}\n";
    return text;
  }

  static std::string makeFanoutSource(int64_t extent,
                                      llvm::ArrayRef<int64_t> destinations) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64: 0, 0, 1, 2>}
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @source(%input: tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) {
      %result = wafer.tile.region(
          %input : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
           << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
           << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
      }
      return
    }
  }
)mlir";
    for (int64_t tile : destinations) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << R"mlir( {
    func.func @destination(%placeholder: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> tensor<1x)mlir" << extent
             << R"mlir(x64xf16> {
      %result = wafer.tile.region(
          %placeholder : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
             << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.mulf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
      }
      return %result : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
    }
  }
)mlir";
    }
    stream << "}\n";
    return text;
  }

  static std::string makeInterferingFanoutsSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) {
      %first, %second = wafer.tile.region(
          %input : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
           << R"mlir(x64xf16>, tensor<1x)mlir" << extent << R"mlir(x64xf16>) {
      ^bb0(%local: tensor<1x)mlir"
           << extent << R"mlir(x64xf16>):
        %empty0 = tensor.empty() : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        %empty1 = tensor.empty() : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        %mapped0 = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>)
            outs(%empty0 : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        %mapped1 = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>)
            outs(%empty1 : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %twice = arith.addf %value, %value : f16
            linalg.yield %twice : f16
        } -> tensor<1x)mlir"
           << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped0, %mapped1 : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>, tensor<1x)mlir" << extent
           << R"mlir(x64xf16>
      }
      return
    }
  }
)mlir";
    for (int64_t tile : {1, 2})
      stream
          << "  wafer.tile.module card_id = 0 tile_id = " << tile << R"mlir( {
    func.func @entry(%left: tensor<1x)mlir"
          << extent << R"mlir(x64xf16> {wafer.cross_tile_placeholder},
                     %right: tensor<1x)mlir"
          << extent
          << R"mlir(x64xf16> {wafer.cross_tile_placeholder}) -> tensor<1x)mlir"
          << extent << R"mlir(x64xf16> {
      %result = wafer.tile.region(
          %left, %right : tensor<1x)mlir"
          << extent << R"mlir(x64xf16>, tensor<1x)mlir" << extent
          << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
          << R"mlir(x64xf16>) {
      ^bb0(%local_left: tensor<1x)mlir"
          << extent << R"mlir(x64xf16>, %local_right: tensor<1x)mlir" << extent
          << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
          << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local_left, %local_right : tensor<1x)mlir"
          << extent << R"mlir(x64xf16>, tensor<1x)mlir" << extent
          << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
          << extent << R"mlir(x64xf16>) {
          ^bb1(%lhs: f16, %rhs: f16, %old: f16):
            %sum = arith.addf %lhs, %rhs : f16
            linalg.yield %sum : f16
        } -> tensor<1x)mlir"
          << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
          << extent << R"mlir(x64xf16>
      }
      return %result : tensor<1x)mlir"
          << extent << R"mlir(x64xf16>
    }
  }
)mlir";
    stream << "}\n";
    return text;
  }

  static std::string makeCompleteExchangeSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < 4; ++tile) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << R"mlir( {
    func.func @entry(%local: tensor<1x)mlir"
             << extent << "x64xf16>";
      for (int64_t source = 0; source < 4; ++source) {
        if (source == tile)
          continue;
        stream << ", %from" << source << ": tensor<1x" << extent
               << "x64xf16> {wafer.cross_tile_placeholder}";
      }
      stream << ") -> tensor<1x" << extent << R"mlir(x64xf16> {
      %source, %output = wafer.tile.region(%local)mlir";
      for (int64_t source = 0; source < 4; ++source)
        if (source != tile)
          stream << ", %from" << source;
      stream << " : tensor<1x" << extent << "x64xf16>";
      for (int64_t source = 0; source < 4; ++source)
        if (source != tile)
          stream << ", tensor<1x" << extent << "x64xf16>";
      stream << ") -> (tensor<1x" << extent << "x64xf16>, tensor<1x" << extent
             << R"mlir(x64xf16>) {
      ^bb0(%local_arg: tensor<1x)mlir"
             << extent << "x64xf16>";
      for (int64_t source = 0; source < 4; ++source)
        if (source != tile)
          stream << ", %from" << source << "_arg: tensor<1x" << extent
                 << "x64xf16>";
      stream << R"mlir():
        %source_empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %local_source = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local_arg : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%source_empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            linalg.yield %value : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %output_empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %combined = linalg.generic {
            indexing_maps = [#id, #id, #id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins()mlir";
      bool first = true;
      for (int64_t source = 0; source < 4; ++source) {
        if (source == tile)
          continue;
        stream << (first ? "" : ", ") << "%from" << source << "_arg";
        first = false;
      }
      stream << " : tensor<1x" << extent << "x64xf16>, tensor<1x" << extent
             << "x64xf16>, tensor<1x" << extent << R"mlir(x64xf16>)
            outs(%output_empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%a: f16, %b: f16, %c: f16, %old: f16):
            %ab = arith.addf %a, %b : f16
            %abc = arith.addf %ab, %c : f16
            linalg.yield %abc : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %local_source, %combined
            : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>, tensor<1x)mlir" << extent
             << R"mlir(x64xf16>
      }
      return %output : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
    }
  }
)mlir";
    }
    stream << "}\n";
    return text;
  }

  static std::string makeSplitTwoTileExchangeSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < 2; ++tile) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << R"mlir( {
    func.func @entry(%local: tensor<1x)mlir"
             << extent << "x64xf16>, %remote: tensor<1x" << extent
             << R"mlir(x64xf16> {wafer.cross_tile_placeholder})
        -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16> {
      %source = wafer.tile.region(
          %local : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
             << R"mlir(x64xf16>) {
      ^bb0(%local_arg: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local_arg : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            linalg.yield %value : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
      }
      %output = wafer.tile.region(
          %remote : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
             << R"mlir(x64xf16>) {
      ^bb0(%remote_arg: tensor<1x)mlir"
             << extent << R"mlir(x64xf16>):
        %empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%remote_arg : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.mulf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %mapped : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
      }
      return %output : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
    }
  }
)mlir";
    }
    stream << "}\n";
    return text;
  }

  static std::string makeCausalTwoTileExchangeSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < 2; ++tile) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << R"mlir( {
    func.func @entry(%local: tensor<1x)mlir"
             << extent << "x64xf16>, %remote: tensor<1x" << extent
             << R"mlir(x64xf16> {wafer.cross_tile_placeholder})
        -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16> {
      %source, %output = wafer.tile.region(
          %local, %remote : tensor<1x)mlir"
             << extent << "x64xf16>, tensor<1x" << extent
             << R"mlir(x64xf16>) -> (tensor<1x)mlir" << extent
             << "x64xf16>, tensor<1x" << extent << R"mlir(x64xf16>) {
      ^bb0(%local_arg: tensor<1x)mlir"
             << extent << "x64xf16>, %remote_arg: tensor<1x" << extent
             << R"mlir(x64xf16>):
        %remote_empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %remote_value = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%remote_arg : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%remote_empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.mulf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %local_empty = tensor.empty() : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        %local_value = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local_arg : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>)
            outs(%local_empty : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.addf %value, %value : f16
            linalg.yield %next : f16
        } -> tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
        wafer.tile.yield %local_value, %remote_value
            : tensor<1x)mlir"
             << extent << "x64xf16>, tensor<1x" << extent << R"mlir(x64xf16>
      }
      return %output : tensor<1x)mlir"
             << extent << R"mlir(x64xf16>
    }
  }
)mlir";
    }
    stream << "}\n";
    return text;
  }

  static std::string makeSquareSource(int64_t extent, int64_t exponent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<1x)mlir"
           << extent << R"mlir(x64xf32>) -> tensor<1x)mlir" << extent
           << R"mlir(x64xf32> {
      %result = wafer.tile.region(
          %input : tensor<1x)mlir"
           << extent << R"mlir(x64xf32>) -> (tensor<1x)mlir" << extent
           << R"mlir(x64xf32>) {
      ^bb0(%local: tensor<1x)mlir"
           << extent << R"mlir(x64xf32>):
        %empty = tensor.empty() : tensor<1x)mlir"
           << extent << R"mlir(x64xf32>
        %mapped = linalg.generic {
            indexing_maps = [#id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<1x)mlir"
           << extent << R"mlir(x64xf32>)
            outs(%empty : tensor<1x)mlir"
           << extent << R"mlir(x64xf32>) {
          ^bb1(%value: f32, %old: f32):
            %exponent = arith.constant )mlir"
           << exponent << R"mlir(.000000e+00 : f32
            %powered = math.powf %value, %exponent : f32
            linalg.yield %powered : f32
        } -> tensor<1x)mlir"
           << extent << R"mlir(x64xf32>
        wafer.tile.yield %mapped : tensor<1x)mlir"
           << extent << R"mlir(x64xf32>
      }
      return %result : tensor<1x)mlir"
           << extent << R"mlir(x64xf32>
    }
  }
}
)mlir";
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredToTileTest,
       LowersContractionExpressionAndReductionAtRealisticScale) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeSource(extent));
    ASSERT_TRUE(module);
    TileRegionOp region;
    module->walk([&](TileRegionOp current) { region = current; });
    ASSERT_TRUE(region);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    relations.structuralOutputs.push_back({1, region.getResult(1)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;

    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    EXPECT_EQ(lowered.statistics.contractions, 1u);
    EXPECT_EQ(lowered.statistics.reductions, 1u);
    EXPECT_EQ(lowered.statistics.elementwiseExpressions, 1u);
    EXPECT_GE(lowered.statistics.elementwiseOperations, 2u);
    EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(module->getOperation()), 0u);
    EXPECT_EQ(countOps<ComputeGemmOp>(module->getOperation()), 1u);
    EXPECT_EQ(countOps<ComputeReduceOp>(module->getOperation()), 1u);
    EXPECT_GE(countOps<ComputeElementwiseOp>(module->getOperation()), 2u);
    EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        module->getOperation(), relations)));
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ddrLoads, 2u);
    EXPECT_EQ(movement.statistics.ddrStores, 2u);
    EXPECT_EQ(movement.statistics.outputCopiesRemoved, 2u);
    EXPECT_TRUE(relations.structuralOutputs.empty());
    EXPECT_TRUE(relations.boundaryRelations.empty());
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        module->getOperation(), relations)));
  }
}

TEST_F(StructuredToTileTest,
       LowersOrdinaryConvolutionFromCurrentMapsWithoutShapeNames) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeConvSource(extent));
    ASSERT_TRUE(module);
    TileRegionOp region;
    module->walk([&](TileRegionOp current) { region = current; });
    ASSERT_TRUE(region);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    EXPECT_EQ(lowered.statistics.convolutions, 1u);
    EXPECT_EQ(countOps<ComputeConvOp>(module->getOperation()), 1u);
    EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(module->getOperation()), 0u);
    EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ddrLoads, 2u);
    EXPECT_EQ(movement.statistics.ddrStores, 1u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
  }
}

TEST_F(StructuredToTileTest, SameTileRegionsUseOneExplicitDDRStage) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeTwoRegionSource(extent, /*crossTile=*/false));
    ASSERT_TRUE(module);
    llvm::SmallVector<TileRegionOp, 2> regions;
    module->walk([&](TileRegionOp region) { regions.push_back(region); });
    ASSERT_EQ(regions.size(), 2u);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, regions[1].getResult(0)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.interRegionDDRStages, 1u);
    EXPECT_EQ(movement.statistics.ddrLoads, 2u);
    EXPECT_EQ(movement.statistics.ddrStores, 2u);
    EXPECT_EQ(movement.statistics.peerSends, 0u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
  }
}

TEST_F(StructuredToTileTest, CrossTileRelationBecomesOneMatchedPeerTransfer) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeTwoRegionSource(extent, /*crossTile=*/true));
    ASSERT_TRUE(module);
    llvm::SmallVector<TileRegionOp, 2> regions;
    module->walk([&](TileRegionOp region) { regions.push_back(region); });
    ASSERT_EQ(regions.size(), 2u);
    StructuredMaterializationRelations relations;
    relations.boundaryRelations.push_back(
        {regions[0].getResult(0), regions[1].getBody().getArgument(0)});
    relations.structuralOutputs.push_back({0, regions[1].getResult(0)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ddrLoads, 1u);
    EXPECT_EQ(movement.statistics.ddrStores, 1u);
    EXPECT_EQ(movement.statistics.peerSends, 1u);
    EXPECT_EQ(movement.statistics.peerReceives, 1u);
    EXPECT_EQ(countOps<CommPeerSendOp>(module->getOperation()), 1u);
    EXPECT_EQ(countOps<CommPeerRecvOp>(module->getOperation()), 1u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));

    std::string standaloneFailure;
    auto standalone = createStandaloneTileModules(
        std::move(module), &standaloneFailure, &relations);
    ASSERT_TRUE(mlir::succeeded(standalone)) << standaloneFailure;
    llvm::SmallVector<mlir::ModuleOp, 2> instructionModules;
    for (StandaloneTileModule &tile : *standalone) {
      llvm::SmallVector<TileRegionOp, 2> tileRegions;
      tile.module->walk(
          [&](TileRegionOp region) { tileRegions.push_back(region); });
      TileRegionToInstrLoweringSession session(*tile.module->getContext());
      for (TileRegionOp region : tileRegions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      instructionModules.push_back(*tile.module);
    }
    DirectDTECompletionResult completion =
        rebuildRequiredDirectDTEWaits(instructionModules);
    ASSERT_TRUE(completion.succeeded()) << completion.detail;
    instructionModules.clear();
    for (StandaloneTileModule &tile : *standalone) {
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      TileMemoryPlanningFailure memoryFailure;
      auto planned = planTileMemory(std::move(tile.module), &memoryFailure);
      ASSERT_TRUE(mlir::succeeded(planned));
      tile.module = std::move(*planned);
      instructionModules.push_back(*tile.module);
    }
    EXPECT_TRUE(
        mlir::succeeded(verifyDirectDTETransportSchedule(instructionModules)));
  }
}

TEST_F(StructuredToTileTest,
       SamePayloadFanoutBecomesTopologyAwareActualRelayTree) {
  constexpr int64_t destinations[] = {1, 2, 3, 4, 5};
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeFanoutSource(extent, destinations));
    ASSERT_TRUE(module);
    llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 8> regions;
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) {
        regions.push_back({tile.getTileIdAttr().getInt(), region});
      });
    llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    ASSERT_EQ(regions.size(), 6u);

    StructuredMaterializationRelations relations;
    for (unsigned index = 1; index < regions.size(); ++index) {
      relations.boundaryRelations.push_back(
          {regions.front().second.getResult(0),
           regions[index].second.getBody().getArgument(0)});
      relations.structuralOutputs.push_back(
          {0, regions[index].second.getResult(0)});
    }
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;

    EXPECT_EQ(movement.statistics.peerSends, 5u);
    EXPECT_EQ(movement.statistics.peerReceives, 5u);
    EXPECT_GT(movement.statistics.peerRelaySends, 0u);
    EXPECT_EQ(movement.statistics.topologyFanoutGroups, 1u);
    EXPECT_EQ(movement.statistics.topologyFanoutRounds, 3u);
    EXPECT_EQ(countOps<CommPeerSendOp>(module->getOperation()), 5u);
    EXPECT_EQ(countOps<CommPeerRecvOp>(module->getOperation()), 5u);
    EXPECT_EQ(countOps<mlir::async::AwaitOp>(module->getOperation()), 0u);

    unsigned sourceSends = 0;
    unsigned relaysWithActualReceiveStorage = 0;
    for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
      unsigned receives = 0;
      tile.walk([&](CommPeerRecvOp) { ++receives; });
      tile.walk([&](CommPeerSendOp send) {
        if (tile.getTileIdAttr().getInt() == 0) {
          ++sourceSends;
          return;
        }
        bool sharesReceiveStorage = false;
        tile.walk([&](CommPeerRecvOp receive) {
          sharesReceiveStorage |= receive.getBuffer() == send.getBuffer();
        });
        relaysWithActualReceiveStorage += sharesReceiveStorage;
      });
      if (tile.getTileIdAttr().getInt() != 0) {
        EXPECT_EQ(receives, 1u);
      }
    }
    EXPECT_LT(sourceSends, 5u);
    EXPECT_EQ(relaysWithActualReceiveStorage,
              movement.statistics.peerRelaySends);
    EXPECT_EQ(sourceSends + relaysWithActualReceiveStorage, 5u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
  }
}

TEST_F(StructuredToTileTest, ExactPowerTwoLowersToUnarySquareInstruction) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeSquareSource(extent, 2));
    ASSERT_TRUE(module);
    TileRegionOp region;
    module->walk([&](TileRegionOp current) { region = current; });
    ASSERT_TRUE(region);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    unsigned tileSquares = 0;
    module->walk([&](ComputeElementwiseOp operation) {
      if (operation.getKind() == ComputeElementwiseKind::Square) {
        ++tileSquares;
        EXPECT_EQ(operation.getInputs().size(), 1u);
      }
    });
    EXPECT_EQ(tileSquares, 1u);
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    llvm::SmallVector<TileRegionOp, 2> regions;
    module->walk([&](TileRegionOp current) { regions.push_back(current); });
    TileRegionToInstrLoweringSession session(*module->getContext());
    for (TileRegionOp current : regions)
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(current, session)));
    unsigned instructionSquares = 0;
    module->walk([&](InstrElementwiseOp operation) {
      instructionSquares += operation.getKind() == InstrElementwiseKind::Square;
    });
    EXPECT_EQ(instructionSquares, 1u);

    auto unsupported = parse(makeSquareSource(extent, 3));
    ASSERT_TRUE(unsupported);
    TileRegionOp unsupportedRegion;
    unsupported->walk(
        [&](TileRegionOp current) { unsupportedRegion = current; });
    StructuredMaterializationRelations unsupportedRelations;
    unsupportedRelations.structuralOutputs.push_back(
        {0, unsupportedRegion.getResult(0)});
    LayoutOptimizationResult unsupportedLayout =
        resolveCurrentLayoutsAndBufferize(*unsupported, unsupportedRelations);
    ASSERT_TRUE(unsupportedLayout.succeeded()) << unsupportedLayout.detail;
    StructuredToTileResult rejected =
        lowerStructuredComputeToTile(*unsupported, unsupportedRelations);
    EXPECT_EQ(rejected.failure, StructuredToTileFailureKind::Unsupported);
    EXPECT_EQ(countOps<mlir::math::PowFOp>(unsupported->getOperation()), 1u);
  }
}

TEST_F(StructuredToTileTest,
       PermutedElementwiseResultIsCanonicalizedToResultCoordinates) {
  constexpr llvm::StringLiteral text = R"mlir(
#swap = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x64x1025xf16>) {
      %result = wafer.tile.region(
          %input : tensor<2x64x1025xf16>)
          -> (tensor<2x64x1025xf16>) {
      ^bb0(%local: tensor<2x64x1025xf16>):
        %empty = tensor.empty() : tensor<2x64x1025xf16>
        %mapped = linalg.generic {
            indexing_maps = [#swap, #swap],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%local : tensor<2x64x1025xf16>)
            outs(%empty : tensor<2x64x1025xf16>) {
          ^bb1(%value: f16, %old: f16):
            %next = arith.negf %value : f16
            linalg.yield %next : f16
        } -> tensor<2x64x1025xf16>
        wafer.tile.yield %mapped : tensor<2x64x1025xf16>
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
  LayoutOptimizationResult layout =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(layout.succeeded()) << layout.detail;
  StructuredToTileResult lowered =
      lowerStructuredComputeToTile(*module, relations);
  ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
  ComputeElementwiseOp elementwise;
  module->walk(
      [&](ComputeElementwiseOp operation) { elementwise = operation; });
  ASSERT_TRUE(elementwise);
  mlir::ArrayAttr maps = elementwise.getIndexingMapsAttr();
  ASSERT_TRUE(maps);
  ASSERT_EQ(maps.size(), 2u);
  for (mlir::Attribute attribute : maps) {
    auto map = mlir::cast<mlir::AffineMapAttr>(attribute).getValue();
    EXPECT_TRUE(map.isIdentity());
  }
  EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(module->getOperation()), 0u);
  EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
}

TEST_F(StructuredToTileTest,
       InterferingFanoutsBecomeActualSparseRoundsWithoutDDR) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeInterferingFanoutsSource(extent));
    ASSERT_TRUE(module);
    llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 4> regions;
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) {
        regions.push_back({tile.getTileIdAttr().getInt(), region});
      });
    llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    ASSERT_EQ(regions.size(), 3u);
    StructuredMaterializationRelations relations;
    for (unsigned destination = 1; destination < regions.size();
         ++destination) {
      relations.boundaryRelations.push_back(
          {regions[0].second.getResult(0),
           regions[destination].second.getBody().getArgument(0)});
      relations.boundaryRelations.push_back(
          {regions[0].second.getResult(1),
           regions[destination].second.getBody().getArgument(1)});
      relations.structuralOutputs.push_back(
          {0, regions[destination].second.getResult(0)});
    }
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.peerSends, 4u);
    EXPECT_EQ(movement.statistics.peerReceives, 4u);
    EXPECT_EQ(movement.statistics.sparseRoundComponents, 1u);
    EXPECT_EQ(movement.statistics.sparseRounds, 4u);
    EXPECT_EQ(movement.statistics.crossTileDDRStages, 0u);
    EXPECT_EQ(countOps<CommPeerSendOp>(module->getOperation()), 4u);
    EXPECT_EQ(countOps<CommPeerRecvOp>(module->getOperation()), 4u);

    unsigned resources = 0;
    for (mlir::memref::GlobalOp global :
         module->getOps<mlir::memref::GlobalOp>())
      resources += static_cast<bool>(
          global->getAttrOfType<DDRResourceAttr>(kWaferDDRResourceAttrName));
    EXPECT_EQ(resources, 0u);
    for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
      mlir::func::FuncOp entry = *tile.getOps<mlir::func::FuncOp>().begin();
      unsigned bindings = 0;
      for (unsigned index = 0; index < entry.getNumArguments(); ++index)
        bindings += static_cast<bool>(entry.getArgAttrOfType<DDRBindingAttr>(
            index, kWaferDDRBindingAttrName));
      EXPECT_EQ(bindings, 0u);
      EXPECT_EQ(entry.getNumArguments(),
                tile.getTileIdAttr().getInt() == 0 ? 1u : 0u);
    }
  }
}

TEST_F(StructuredToTileTest, CompleteExchangeBecomesThreeTopologyRingRounds) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeCompleteExchangeSource(extent));
    ASSERT_TRUE(module);
    llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 4> regions;
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) {
        regions.push_back({tile.getTileIdAttr().getInt(), region});
      });
    llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    ASSERT_EQ(regions.size(), 4u);
    StructuredMaterializationRelations relations;
    for (unsigned source = 0; source < regions.size(); ++source)
      for (unsigned destination = 0; destination < regions.size();
           ++destination) {
        if (source == destination)
          continue;
        unsigned argument = 1;
        for (unsigned candidate = 0; candidate < source; ++candidate)
          argument += candidate != destination;
        relations.boundaryRelations.push_back(
            {regions[source].second.getResult(0),
             regions[destination].second.getBody().getArgument(argument)});
      }
    for (auto &[tile, region] : regions) {
      (void)tile;
      relations.structuralOutputs.push_back({0, region.getResult(1)});
    }
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ringComponents, 1u);
    EXPECT_EQ(movement.statistics.ringRounds, 3u);
    EXPECT_EQ(movement.statistics.sparseRoundComponents, 0u);
    EXPECT_EQ(movement.statistics.peerSends, 12u);
    EXPECT_EQ(movement.statistics.peerReceives, 12u);
    EXPECT_EQ(movement.statistics.crossTileDDRStages, 0u);
    for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
      llvm::SmallVector<unsigned, 3> sends(3, 0);
      llvm::SmallVector<unsigned, 3> receives(3, 0);
      tile.walk([&](CommPeerSendOp send) {
        ++sends[send.getMessageAttr().getRound()];
      });
      tile.walk([&](CommPeerRecvOp receive) {
        ++receives[receive.getMessageAttr().getRound()];
      });
      EXPECT_EQ(sends, (llvm::SmallVector<unsigned, 3>{1, 1, 1}));
      EXPECT_EQ(receives, (llvm::SmallVector<unsigned, 3>{1, 1, 1}));
    }

    std::string standaloneFailure;
    auto standalone = createStandaloneTileModules(
        std::move(module), &standaloneFailure, &relations);
    ASSERT_TRUE(mlir::succeeded(standalone)) << standaloneFailure;
    llvm::SmallVector<mlir::ModuleOp, 4> instructionModules;
    for (StandaloneTileModule &tile : *standalone) {
      llvm::SmallVector<TileRegionOp, 2> tileRegions;
      tile.module->walk(
          [&](TileRegionOp region) { tileRegions.push_back(region); });
      TileRegionToInstrLoweringSession session(*tile.module->getContext());
      for (TileRegionOp region : tileRegions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      instructionModules.push_back(*tile.module);
    }
    DirectDTECompletionResult completion =
        rebuildRequiredDirectDTEWaits(instructionModules);
    ASSERT_TRUE(completion.succeeded()) << completion.detail;
    instructionModules.clear();
    for (StandaloneTileModule &tile : *standalone) {
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      TileMemoryPlanningFailure memoryFailure;
      auto planned = planTileMemory(std::move(tile.module), &memoryFailure);
      ASSERT_TRUE(mlir::succeeded(planned));
      tile.module = std::move(*planned);
      instructionModules.push_back(*tile.module);
    }
    EXPECT_TRUE(
        mlir::succeeded(verifyDirectDTETransportSchedule(instructionModules)));
  }
}

TEST_F(StructuredToTileTest, SplitExchangeRegionsCloseBeforeOneRingRound) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeSplitTwoTileExchangeSource(extent));
    ASSERT_TRUE(module);
    llvm::SmallVector<llvm::SmallVector<TileRegionOp, 2>, 2> regions(2);
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) {
        regions[tile.getTileIdAttr().getInt()].push_back(region);
      });
    ASSERT_EQ(regions[0].size(), 2u);
    ASSERT_EQ(regions[1].size(), 2u);
    StructuredMaterializationRelations relations;
    relations.boundaryRelations.push_back(
        {regions[0][0].getResult(0), regions[1][1].getBody().getArgument(0)});
    relations.boundaryRelations.push_back(
        {regions[1][0].getResult(0), regions[0][1].getBody().getArgument(0)});
    relations.structuralOutputs.push_back({0, regions[0][1].getResult(0)});
    relations.structuralOutputs.push_back({0, regions[1][1].getResult(0)});
    CommunicationRegionClosureStatistics closure;
    SpatialRegionMaterializationFailure closureFailure;
    ASSERT_TRUE(mlir::succeeded(closeCrossTileCommunicationRegions(
        *module, relations, &closure, &closureFailure)))
        << closureFailure.detail;
    EXPECT_EQ(closure.closedExchangeComponents, 1u);
    EXPECT_EQ(closure.mergedTileScopes, 2u);
    EXPECT_EQ(closure.mergedRegions, 4u);
    EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()), 2u);
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ringComponents, 1u);
    EXPECT_EQ(movement.statistics.ringRounds, 1u);
    EXPECT_EQ(movement.statistics.peerSends, 2u);
    EXPECT_EQ(movement.statistics.peerReceives, 2u);
    EXPECT_EQ(movement.statistics.crossTileDDRStages, 0u);
  }
}

TEST_F(StructuredToTileTest,
       CrossComponentRegionOrderCycleUsesOnePreflightDDRBoundary) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeSplitTwoTileExchangeSource(extent));
    ASSERT_TRUE(module);
    llvm::SmallVector<llvm::SmallVector<TileRegionOp, 2>, 2> regions(2);
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) {
        regions[tile.getTileIdAttr().getInt()].push_back(region);
      });
    ASSERT_EQ(regions[0].size(), 2u);
    ASSERT_EQ(regions[1].size(), 2u);
    StructuredMaterializationRelations relations;
    relations.boundaryRelations.push_back(
        {regions[0][0].getResult(0), regions[1][1].getBody().getArgument(0)});
    relations.boundaryRelations.push_back(
        {regions[1][0].getResult(0), regions[0][1].getBody().getArgument(0)});
    relations.structuralOutputs.push_back({0, regions[0][1].getResult(0)});
    relations.structuralOutputs.push_back({0, regions[1][1].getResult(0)});

    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.noCutDDRComponents, 1u);
    EXPECT_EQ(movement.statistics.peerSends, 1u);
    EXPECT_EQ(movement.statistics.peerReceives, 1u);
    EXPECT_EQ(movement.statistics.crossTileDDRStages, 1u);
    EXPECT_EQ(countOps<CommPeerSendOp>(module->getOperation()), 1u);
    EXPECT_EQ(countOps<CommPeerRecvOp>(module->getOperation()), 1u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));

    std::string standaloneFailure;
    auto standalone = createStandaloneTileModules(
        std::move(module), &standaloneFailure, &relations);
    ASSERT_TRUE(mlir::succeeded(standalone)) << standaloneFailure;
    llvm::SmallVector<mlir::ModuleOp, 2> instructionModules;
    for (StandaloneTileModule &tile : *standalone) {
      llvm::SmallVector<TileRegionOp, 2> tileRegions;
      tile.module->walk(
          [&](TileRegionOp region) { tileRegions.push_back(region); });
      TileRegionToInstrLoweringSession session(*tile.module->getContext());
      for (TileRegionOp region : tileRegions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      instructionModules.push_back(*tile.module);
    }
    DirectDTECompletionResult completion =
        rebuildRequiredDirectDTEWaits(instructionModules);
    ASSERT_TRUE(completion.succeeded()) << completion.detail;
    instructionModules.clear();
    for (StandaloneTileModule &tile : *standalone) {
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      TileMemoryPlanningFailure memoryFailure;
      auto planned = planTileMemory(std::move(tile.module), &memoryFailure);
      ASSERT_TRUE(mlir::succeeded(planned));
      tile.module = std::move(*planned);
      instructionModules.push_back(*tile.module);
    }
    EXPECT_TRUE(
        mlir::succeeded(verifyDirectDTETransportSchedule(instructionModules)));
  }
}

TEST_F(StructuredToTileTest,
       BidirectionalNoCutUsesOneExplicitCausalDDRBoundary) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeCausalTwoTileExchangeSource(extent));
    ASSERT_TRUE(module);
    llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 2> regions;
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) {
        regions.push_back({tile.getTileIdAttr().getInt(), region});
      });
    llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    ASSERT_EQ(regions.size(), 2u);
    StructuredMaterializationRelations relations;
    relations.boundaryRelations.push_back(
        {regions[0].second.getResult(0),
         regions[1].second.getBody().getArgument(1)});
    relations.boundaryRelations.push_back(
        {regions[1].second.getResult(0),
         regions[0].second.getBody().getArgument(1)});
    relations.structuralOutputs.push_back({0, regions[0].second.getResult(1)});
    relations.structuralOutputs.push_back({0, regions[1].second.getResult(1)});

    CommunicationRegionClosureStatistics closure;
    SpatialRegionMaterializationFailure closureFailure;
    ASSERT_TRUE(mlir::succeeded(closeCrossTileCommunicationRegions(
        *module, relations, &closure, &closureFailure)))
        << closureFailure.detail;
    EXPECT_EQ(closure.closedExchangeComponents, 0u);
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    StructuredToTileResult lowered =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.noCutDDRComponents, 1u);
    EXPECT_EQ(movement.statistics.ringComponents, 0u);
    EXPECT_EQ(movement.statistics.peerSends, 0u);
    EXPECT_EQ(movement.statistics.peerReceives, 0u);
    EXPECT_EQ(movement.statistics.crossTileDDRStages, 2u);
  }
}

} // namespace
