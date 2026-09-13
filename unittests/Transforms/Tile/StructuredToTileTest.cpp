//===- StructuredToTileTest.cpp ---------------------------------------===//

#include "Wafer/Transforms/Tile/StructuredToTile.h"
#include "TestSupport/Driver/CompilerTesting.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/StandaloneTileModules/StandaloneTileModules.h"
#include "Wafer/Transforms/Instr/DirectDTETransport.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Instr/NativeDirectDTEMultiSend.h"
#include "Wafer/Transforms/Instr/SharedDDRCompletion.h"
#include "Wafer/Transforms/Instr/TileMemoryPlanning.h"
#include "Wafer/Transforms/Linalg/CommunicationRegionClosure.h"
#include "Wafer/Transforms/Linalg/ContractionAccumulation.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/DistributedCollectiveMovement.h"
#include "Wafer/Transforms/Tile/GemmFinalization.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/TiledOutputStores.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/Topology/TargetTopology.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Planning/PhysicalDataflow/CollectiveAlgorithms.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>

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
        << extent + 1
        << R"mlir(x66x16xf16>, %weight: tensor<2x3x16x32xf16>) -> tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16> {
      %result = wafer.tile.region(
          %input, %weight : tensor<1x)mlir"
        << extent + 1 << R"mlir(x66x16xf16>, tensor<2x3x16x32xf16>)
          -> (tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>) {
      ^bb0(%local_input: tensor<1x)mlir"
        << extent + 1
        << R"mlir(x66x16xf16>, %local_weight: tensor<2x3x16x32xf16>):
        %zero = arith.constant 0.000000e+00 : f16
        %empty = tensor.empty() : tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>
        %init = linalg.fill ins(%zero : f16)
            outs(%empty : tensor<1x)mlir"
        << extent << R"mlir(x64x32xf16>) -> tensor<1x)mlir" << extent
        << R"mlir(x64x32xf16>
        %conv = linalg.conv_2d_nhwc_hwcf
            ins(%local_input, %local_weight : tensor<1x)mlir"
        << extent + 1 << R"mlir(x66x16xf16>, tensor<2x3x16x32xf16>)
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
                                      llvm::ArrayRef<int64_t> destinations,
                                      int64_t sourceTile = 0,
                                      bool hasUnavailableTile = true) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64)mlir";
    if (hasUnavailableTile)
      stream << ": 0, 0, 1, 2";
    stream << R"mlir(>}
  wafer.tile.module card_id = 0 tile_id = )mlir"
           << sourceTile << R"mlir( {
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
          << extent << R"mlir(x64xf16> {wafer.cross_tile_boundary_input},
                     %right: tensor<1x)mlir"
          << extent
          << R"mlir(x64xf16> {wafer.cross_tile_boundary_input}) -> tensor<1x)mlir"
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
               << "x64xf16> {wafer.cross_tile_boundary_input}";
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

  static std::string makeDenseCompleteExchangeSource(int64_t extent,
                                                     int64_t tileCount,
                                                     bool bf16 = false) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < tileCount; ++tile) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << " {\n    func.func @entry(%local: tensor<1x" << extent
             << "x1xf16>";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << ", %from" << source << ": tensor<1x" << extent
                 << "x1xf16> {wafer.cross_tile_boundary_input}";
      stream << ") -> tensor<1x" << extent
             << "x1xf16> {\n      %source, %output = "
                "wafer.tile.region(%local";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << ", %from" << source;
      stream << " : tensor<1x" << extent << "x1xf16>";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << ", tensor<1x" << extent << "x1xf16>";
      stream << ") -> (tensor<1x" << extent << "x1xf16>, tensor<1x" << extent
             << "x1xf16>) {\n      ^bb0(%local_arg: tensor<1x" << extent
             << "x1xf16>";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << ", %from" << source << "_arg: tensor<1x" << extent
                 << "x1xf16>";
      stream << "):\n"
                "        %source_empty = tensor.empty() : tensor<1x"
             << extent << "x1xf16>\n"
             << "        %local_source = linalg.generic {indexing_maps = "
                "[#id, #id], iterator_types = [\"parallel\", \"parallel\", "
                "\"parallel\"]} ins(%local_arg : tensor<1x"
             << extent << "x1xf16>) outs(%source_empty : tensor<1x" << extent
             << "x1xf16>) {\n"
                "        ^bb1(%value: f16, %old: f16):\n"
                "          linalg.yield %value : f16\n"
                "        } -> tensor<1x"
             << extent
             << "x1xf16>\n"
                "        %output_empty = tensor.empty() : tensor<1x"
             << extent
             << "x1xf16>\n"
                "        %combined = linalg.generic {indexing_maps = [";
      for (int64_t index = 0; index < tileCount; ++index)
        stream << (index == 0 ? "" : ", ") << "#id";
      stream << "], iterator_types = [\"parallel\", \"parallel\", "
                "\"parallel\"]} ins(";
      bool first = true;
      for (int64_t source = 0; source < tileCount; ++source) {
        if (source == tile)
          continue;
        stream << (first ? "" : ", ") << "%from" << source << "_arg";
        first = false;
      }
      stream << " : ";
      for (int64_t index = 0; index < tileCount - 1; ++index)
        stream << (index == 0 ? "" : ", ") << "tensor<1x" << extent
               << "x1xf16>";
      stream << ") outs(%output_empty : tensor<1x" << extent
             << "x1xf16>) {\n        ^bb1(";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << "%value" << source << ": f16, ";
      stream << "%old: f16):\n";
      llvm::SmallVector<int64_t, 15> remoteSources;
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          remoteSources.push_back(source);
      int64_t firstSource = remoteSources[0];
      int64_t secondSource = remoteSources[1];
      stream << "          %sum0 = arith.addf %value" << firstSource
             << ", %value" << secondSource << " : f16\n";
      int64_t sumIndex = 0;
      for (int64_t source = 0; source < tileCount; ++source) {
        if (source == tile || source == firstSource || source == secondSource)
          continue;
        stream << "          %sum" << sumIndex + 1 << " = arith.addf %sum"
               << sumIndex << ", %value" << source << " : f16\n";
        ++sumIndex;
      }
      stream << "          linalg.yield %sum" << sumIndex
             << " : f16\n"
                "        } -> tensor<1x"
             << extent
             << "x1xf16>\n"
                "        wafer.tile.yield %local_source, %combined : "
                "tensor<1x"
             << extent << "x1xf16>, tensor<1x" << extent
             << "x1xf16>\n"
                "      }\n"
                "      return %output : tensor<1x"
             << extent << "x1xf16>\n    }\n  }\n";
    }
    stream << "}\n";
    if (bf16)
      for (size_t offset = 0;
           (offset = text.find("f16", offset)) != std::string::npos;
           offset += 4)
        text.replace(offset, 3, "bf16");
    return text;
  }

  static std::string makeThreeTileAllToAllSource() {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < 3; ++tile) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << " {\n    func.func @entry(%local: tensor<1x4x64xf16>";
      for (int64_t source = 0; source < 3; ++source)
        if (source != tile)
          stream << ", %from" << source
                 << ": tensor<1x2x64xf16> "
                    "{wafer.cross_tile_boundary_input}";
      stream << ") -> tensor<1x2x64xf16> {\n"
                "      %first, %second, %output = wafer.tile.region(%local";
      for (int64_t source = 0; source < 3; ++source)
        if (source != tile)
          stream << ", %from" << source;
      stream << " : tensor<1x4x64xf16>";
      for (int64_t source = 0; source < 3; ++source)
        if (source != tile)
          stream << ", tensor<1x2x64xf16>";
      stream << R"mlir() -> (tensor<1x2x64xf16>, tensor<1x2x64xf16>,
                              tensor<1x2x64xf16>) {
      ^bb0(%local_arg: tensor<1x4x64xf16>)mlir";
      for (int64_t source = 0; source < 3; ++source)
        if (source != tile)
          stream << ", %from" << source << "_arg: tensor<1x2x64xf16>";
      stream << R"mlir():
        %piece0 = tensor.extract_slice %local_arg[0, 0, 0] [1, 2, 64]
            [1, 1, 1] : tensor<1x4x64xf16> to tensor<1x2x64xf16>
        %piece1 = tensor.extract_slice %local_arg[0, 2, 0] [1, 2, 64]
            [1, 1, 1] : tensor<1x4x64xf16> to tensor<1x2x64xf16>
        %empty = tensor.empty() : tensor<1x2x64xf16>
        %combined = linalg.generic {
            indexing_maps = [#id, #id, #id],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins()mlir";
      bool first = true;
      for (int64_t source = 0; source < 3; ++source) {
        if (source == tile)
          continue;
        stream << (first ? "" : ", ") << "%from" << source << "_arg";
        first = false;
      }
      stream << R"mlir( : tensor<1x2x64xf16>, tensor<1x2x64xf16>)
            outs(%empty : tensor<1x2x64xf16>) {
          ^bb1(%lhs: f16, %rhs: f16, %old: f16):
            %sum = arith.addf %lhs, %rhs : f16
            linalg.yield %sum : f16
        } -> tensor<1x2x64xf16>
        wafer.tile.yield %piece0, %piece1, %combined
            : tensor<1x2x64xf16>, tensor<1x2x64xf16>, tensor<1x2x64xf16>
      }
      return %output : tensor<1x2x64xf16>
    }
  }
)mlir";
    }
    stream << "}\n";
    return text;
  }

  static std::string
  makeDenseAllToAllSource(int64_t extent, int64_t tileCount, bool bf16 = false,
                          bool reduceScatter = false,
                          llvm::StringRef reductionOp = "arith.addf") {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: )mlir"
           << (tileCount == 4 ? 2 : 4) << ", " << (tileCount == 4 ? 2 : 4)
           << R"mlir(>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < tileCount; ++tile) {
      const int64_t localPieceCount = reduceScatter ? tileCount : tileCount - 1;
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << " {\n    func.func @entry(%local: tensor<1x"
             << localPieceCount * extent << "x1xf16>";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << ", %from" << source << ": tensor<1x" << extent
                 << "x1xf16> {wafer.cross_tile_boundary_input}";
      stream << ") -> tensor<1x" << extent << "x1xf16> {\n      ";
      for (int64_t destination = 0; destination < tileCount; ++destination)
        if (destination != tile)
          stream << "%piece" << destination << ", ";
      stream << "%output = wafer.tile.region(%local";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << ", %from" << source;
      stream << " : tensor<1x" << localPieceCount * extent << "x1xf16>";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << ", tensor<1x" << extent << "x1xf16>";
      stream << ") -> (";
      for (int64_t result = 0; result < tileCount; ++result)
        stream << (result == 0 ? "" : ", ") << "tensor<1x" << extent
               << "x1xf16>";
      stream << ") {\n      ^bb0(%local_arg: tensor<1x"
             << localPieceCount * extent << "x1xf16>";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile)
          stream << ", %from" << source << "_arg: tensor<1x" << extent
                 << "x1xf16>";
      stream << "):\n";
      int64_t pieceIndex = 0;
      for (int64_t destination = 0; destination < tileCount; ++destination) {
        if (destination == tile && !reduceScatter)
          continue;
        stream << "        %piece" << destination
               << "_value = tensor.extract_slice %local_arg[0, "
               << pieceIndex * extent << ", 0] [1, " << extent
               << ", 1] [1, 1, 1] : tensor<1x" << localPieceCount * extent
               << "x1xf16> to tensor<1x" << extent << "x1xf16>\n";
        ++pieceIndex;
      }
      stream << "        %empty = tensor.empty() : tensor<1x" << extent
             << "x1xf16>\n"
                "        %combined = linalg.generic {indexing_maps = [";
      for (int64_t index = 0;
           index < tileCount + static_cast<int64_t>(reduceScatter); ++index)
        stream << (index == 0 ? "" : ", ") << "#id";
      stream << "], iterator_types = [\"parallel\", \"parallel\", "
                "\"parallel\"]} ins(";
      bool first = true;
      for (int64_t source = 0; source < tileCount; ++source) {
        if (source == tile && !reduceScatter)
          continue;
        stream << (first ? "" : ", ")
               << (source == tile ? "%piece" + std::to_string(tile) + "_value"
                                  : "%from" + std::to_string(source) + "_arg");
        first = false;
      }
      stream << " : ";
      for (int64_t index = 0;
           index < tileCount - 1 + static_cast<int64_t>(reduceScatter); ++index)
        stream << (index == 0 ? "" : ", ") << "tensor<1x" << extent
               << "x1xf16>";
      stream << ") outs(%empty : tensor<1x" << extent
             << "x1xf16>) {\n        ^bb1(";
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile || reduceScatter)
          stream << "%value" << source << ": f16, ";
      stream << "%old: f16):\n";
      llvm::SmallVector<int64_t, 15> sources;
      for (int64_t source = 0; source < tileCount; ++source)
        if (source != tile || reduceScatter)
          sources.push_back(source);
      stream << "          %sum0 = " << reductionOp << " %value" << sources[0]
             << ", %value" << sources[1] << " : f16\n";
      for (int64_t index = 2; index < static_cast<int64_t>(sources.size());
           ++index)
        stream << "          %sum" << index - 1 << " = " << reductionOp
               << " %sum" << index - 2 << ", %value" << sources[index]
               << " : f16\n";
      stream << "          linalg.yield %sum" << sources.size() - 2
             << " : f16\n        } -> tensor<1x" << extent
             << "x1xf16>\n        wafer.tile.yield ";
      first = true;
      for (int64_t destination = 0; destination < tileCount; ++destination) {
        if (destination == tile)
          continue;
        stream << (first ? "" : ", ") << "%piece" << destination << "_value";
        first = false;
      }
      stream << ", %combined : ";
      for (int64_t result = 0; result < tileCount; ++result)
        stream << (result == 0 ? "" : ", ") << "tensor<1x" << extent
               << "x1xf16>";
      stream << "\n      }\n      return %output : tensor<1x" << extent
             << "x1xf16>\n    }\n  }\n";
    }
    stream << "}\n";
    if (bf16)
      for (size_t offset = 0;
           (offset = text.find("f16", offset)) != std::string::npos;
           offset += 4)
        text.replace(offset, 3, "bf16");
    return text;
  }

  static std::string
  makeDenseAllReduceSource(int64_t extent, int64_t tileCount,
                           llvm::StringRef reductionOp = "arith.addf",
                           bool bf16 = false) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: )mlir"
           << (tileCount == 4 ? 2 : 4) << ", " << (tileCount == 4 ? 2 : 4)
           << R"mlir(>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < tileCount; ++tile) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << " {\n    func.func @entry(%local: tensor<" << extent
             << "x1x1xf16>";
      if (tile == 0) {
        for (int64_t source = 1; source < tileCount; ++source)
          stream << ", %from" << source << ": tensor<" << extent
                 << "x1x1xf16> {wafer.cross_tile_boundary_input}";
      } else {
        stream << ", %global: tensor<" << extent
               << "x1x1xf16> {wafer.cross_tile_boundary_input}";
      }
      stream << ") -> tensor<" << extent << "x1x1xf16> {\n";
      if (tile == 0) {
        stream << "      %contribution, %merged = wafer.tile.region(%local";
        for (int64_t source = 1; source < tileCount; ++source)
          stream << ", %from" << source;
        stream << " : tensor<" << extent << "x1x1xf16>";
        for (int64_t source = 1; source < tileCount; ++source)
          stream << ", tensor<" << extent << "x1x1xf16>";
        stream << ") -> (tensor<" << extent << "x1x1xf16>, tensor<" << extent
               << "x1x1xf16>) {\n      ^bb0(%local_arg: tensor<" << extent
               << "x1x1xf16>";
        for (int64_t source = 1; source < tileCount; ++source)
          stream << ", %from" << source << "_arg: tensor<" << extent
                 << "x1x1xf16>";
        stream
            << "):\n        %local_empty = tensor.empty() : tensor<" << extent
            << "x1x1xf16>\n        %local_value = linalg.generic "
               "{indexing_maps = [#id, #id], iterator_types = "
               "[\"parallel\", \"parallel\", \"parallel\"]} "
               "ins(%local_arg : tensor<"
            << extent << "x1x1xf16>) outs(%local_empty : tensor<" << extent
            << "x1x1xf16>) {\n        ^bb1(%value: f16, %old: f16):\n"
               "          linalg.yield %value : f16\n        } -> tensor<"
            << extent
            << "x1x1xf16>\n        %merge_empty = tensor.empty() : tensor<"
            << extent
            << "x1x1xf16>\n        %merge = linalg.generic {indexing_maps = [";
        for (int64_t index = 0; index <= tileCount; ++index)
          stream << (index == 0 ? "" : ", ") << "#id";
        stream << "], iterator_types = [\"parallel\", \"parallel\", "
                  "\"parallel\"]} ins(%local_value";
        for (int64_t source = 1; source < tileCount; ++source)
          stream << ", %from" << source << "_arg";
        stream << " : ";
        for (int64_t index = 0; index < tileCount; ++index)
          stream << (index == 0 ? "" : ", ") << "tensor<" << extent
                 << "x1x1xf16>";
        stream << ") outs(%merge_empty : tensor<" << extent
               << "x1x1xf16>) {\n        ^bb1(";
        for (int64_t source = 0; source < tileCount; ++source)
          stream << "%value" << source << ": f16, ";
        stream << "%old: f16):\n          %sum0 = " << reductionOp
               << " %value0, "
                  "%value1 : f16\n";
        for (int64_t source = 2; source < tileCount; ++source)
          stream << "          %sum" << source - 1 << " = " << reductionOp
                 << " %sum" << source - 2 << ", %value" << source << " : f16\n";
        stream << "          linalg.yield %sum" << tileCount - 2
               << " : f16\n        } -> tensor<" << extent
               << "x1x1xf16>\n        wafer.tile.yield %local_value, %merge "
                  ": tensor<"
               << extent << "x1x1xf16>, tensor<" << extent
               << "x1x1xf16>\n      }\n      return %merged : tensor<" << extent
               << "x1x1xf16>\n";
      } else {
        stream << "      %contribution, %output = wafer.tile.region(%local, "
                  "%global : tensor<"
               << extent << "x1x1xf16>, tensor<" << extent
               << "x1x1xf16>) -> (tensor<" << extent << "x1x1xf16>, tensor<"
               << extent << "x1x1xf16>) {\n      ^bb0(%local_arg: tensor<"
               << extent << "x1x1xf16>, %global_arg: tensor<" << extent
               << "x1x1xf16>):\n";
        for (llvm::StringRef name : {"local", "global"})
          stream << "        %" << name << "_empty = tensor.empty() : tensor<"
                 << extent << "x1x1xf16>\n        %" << name
                 << "_value = linalg.generic {indexing_maps = [#id, #id], "
                    "iterator_types = [\"parallel\", \"parallel\", "
                    "\"parallel\"]} ins(%"
                 << name << "_arg : tensor<" << extent << "x1x1xf16>) outs(%"
                 << name << "_empty : tensor<" << extent
                 << "x1x1xf16>) {\n        ^bb1(%value: f16, %old: f16):\n"
                    "          linalg.yield %value : f16\n        } -> tensor<"
                 << extent << "x1x1xf16>\n";
        stream << "        wafer.tile.yield %local_value, %global_value : "
                  "tensor<"
               << extent << "x1x1xf16>, tensor<" << extent
               << "x1x1xf16>\n      }\n      return %output : tensor<" << extent
               << "x1x1xf16>\n";
      }
      stream << "    }\n  }\n";
    }
    stream << "}\n";
    if (bf16)
      for (size_t offset = 0;
           (offset = text.find("f16", offset)) != std::string::npos;
           offset += 4)
        text.replace(offset, 3, "bf16");
    return text;
  }

  static std::string makeSplitExchangeSource(int64_t extent,
                                             int64_t tileCount = 2) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < tileCount; ++tile) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << R"mlir( {
    func.func @entry(%local: tensor<1x)mlir"
             << extent << "x64xf16>, %remote: tensor<1x" << extent
             << R"mlir(x64xf16> {wafer.cross_tile_boundary_input})
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

  static std::string makeSequentialExchangeSource(int64_t extent,
                                                  int64_t tileCount) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    const std::string type = "tensor<1x" + std::to_string(extent) + "x1xf16>";
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
)mlir";
    for (int64_t tile = 0; tile < tileCount; ++tile) {
      stream << "  wafer.tile.module card_id = 0 tile_id = " << tile
             << " {\n    func.func @entry(%local: " << type;
      for (int phase = 0; phase < 2; ++phase)
        for (int64_t source = 0; source < tileCount; ++source)
          if (source != tile)
            stream << ", %remote" << phase << "_" << source << ": " << type
                   << " {wafer.cross_tile_boundary_input}";
      stream << ") -> " << type << " {\n";
      for (int phase = 0; phase < 3; ++phase) {
        const int64_t inputs = phase ? tileCount : 1;
        stream << "      %result" << phase << " = wafer.tile.region("
               << (phase ? "%result" + std::to_string(phase - 1) : "%local");
        if (phase)
          for (int64_t source = 0; source < tileCount; ++source)
            if (source != tile)
              stream << ", %remote" << phase - 1 << "_" << source;
        stream << " : ";
        for (int64_t input = 0; input < inputs; ++input)
          stream << (input ? ", " : "") << type;
        stream << ") -> (" << type << ") {\n      ^bb0(";
        for (int64_t input = 0; input < inputs; ++input)
          stream << (input ? ", " : "") << "%arg" << input << ": " << type;
        stream << "):\n        %empty = tensor.empty() : " << type
               << "\n        %value = linalg.generic {indexing_maps = [";
        for (int64_t input = 0; input <= inputs; ++input)
          stream << (input ? ", " : "") << "#id";
        stream << "], iterator_types = [\"parallel\", \"parallel\", "
                  "\"parallel\"]} ins(";
        for (int64_t input = 0; input < inputs; ++input)
          stream << (input ? ", " : "") << "%arg" << input;
        stream << " : ";
        for (int64_t input = 0; input < inputs; ++input)
          stream << (input ? ", " : "") << type;
        stream << ") outs(%empty : " << type << ") {\n        ^bb1(";
        for (int64_t input = 0; input < inputs; ++input)
          stream << "%v" << input << ": f16, ";
        stream << "%old: f16):\n";
        for (int64_t input = 1; input < inputs; ++input)
          stream << "          %sum" << input << " = arith.addf "
                 << (input == 1 ? "%v0" : "%sum" + std::to_string(input - 1))
                 << ", %v" << input << " : f16\n";
        stream << "          linalg.yield "
               << (phase ? "%sum" + std::to_string(inputs - 1) : "%v0")
               << " : f16\n        } -> " << type
               << "\n        wafer.tile.yield %value : " << type
               << "\n      }\n";
      }
      stream << "      return %result2 : " << type << "\n    }\n  }\n";
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
             << R"mlir(x64xf16> {wafer.cross_tile_boundary_input})
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

  static std::string makeAliasedResultSource(int64_t extent) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<1x)mlir"
           << extent << R"mlir(x64xf16>)
        -> (tensor<1x)mlir"
           << extent << R"mlir(x64xf16>, tensor<1x)mlir" << extent
           << R"mlir(x64xf16>) {
      %first, %second = wafer.tile.region(
          %input : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>)
          -> (tensor<1x)mlir"
           << extent << R"mlir(x64xf16>, tensor<1x)mlir" << extent
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
        wafer.tile.yield %mapped, %mapped
            : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>, tensor<1x)mlir" << extent
           << R"mlir(x64xf16>
      }
      return %first, %second : tensor<1x)mlir"
           << extent << R"mlir(x64xf16>, tensor<1x)mlir" << extent
           << R"mlir(x64xf16>
    }
  }
}
)mlir";
    return text;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(StructuredToTileTest, GenericMinimumHopRingUsesAbstractDistanceOracle) {
  const llvm::SmallVector<uint64_t, 4> participants{0, 1, 2, 3};
  auto ring = buildMinimumHopRing(
      participants, /*maximumParticipants=*/4,
      [](uint64_t lhs, uint64_t rhs) -> std::optional<uint64_t> {
        return lhs > rhs ? lhs - rhs : rhs - lhs;
      });
  ASSERT_TRUE(mlir::succeeded(ring));
  EXPECT_EQ(*ring, (llvm::SmallVector<uint64_t, 4>{0, 1, 2, 3}));
}

TEST_F(StructuredToTileTest,
       GenericMinimumHopRingPreservesDirectedCostAndOpaqueParticipantIds) {
  const uint64_t a = std::numeric_limits<uint64_t>::max() - 2;
  const uint64_t b = std::numeric_limits<uint64_t>::max() - 1;
  const uint64_t c = std::numeric_limits<uint64_t>::max();
  auto ring = buildMinimumHopRing(
      {c, a, b}, /*maximumParticipants=*/4,
      [=](uint64_t lhs, uint64_t rhs) -> std::optional<uint64_t> {
        if ((lhs == a && rhs == b) || (lhs == b && rhs == c) ||
            (lhs == c && rhs == a))
          return 1;
        return 100;
      });
  ASSERT_TRUE(mlir::succeeded(ring));
  EXPECT_EQ(*ring, (llvm::SmallVector<uint64_t, 4>{a, b, c}));
}

TEST_F(StructuredToTileTest,
       GenericDimensionOrderedAllToAllUsesLogicalMeshOnly) {
  const llvm::SmallVector<uint64_t, 4> mesh{101, 203, 307, 401};
  auto steps = buildDimensionOrderedAllToAll(mesh, /*rows=*/2, /*columns=*/2);
  ASSERT_TRUE(mlir::succeeded(steps));
  EXPECT_EQ(steps->size(), 16u);
  for (const DimensionOrderedAllToAllStep &step : *steps) {
    EXPECT_NE(step.source, step.destination);
    EXPECT_TRUE(llvm::is_contained(mesh, step.source));
    EXPECT_TRUE(llvm::is_contained(mesh, step.destination));
    EXPECT_TRUE(llvm::is_contained(mesh, step.relay));
    EXPECT_LE(step.dimension, 1u);
  }
  auto oneDimensional = buildDimensionOrderedAllToAll(mesh, /*rows=*/1,
                                                      /*columns=*/4);
  ASSERT_TRUE(mlir::succeeded(oneDimensional));
  EXPECT_EQ(oneDimensional->size(), 12u);
}

TEST_F(StructuredToTileTest,
       InitializationProofStopsAtWritesAndLayoutSnapshots) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (unsigned mode : {0u, 1u, 2u}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(mode);
      const bool snapshot = mode == 1;
      std::string row = "memref<1x1x" + std::to_string(extent) +
                        "xf32, #wafer.memory<spm, " +
                        (snapshot ? "tensor" : "ncx") + ">>";
      std::string destination = "memref<1x1x" + std::to_string(extent) +
                                "xf32, #wafer.memory<spm, ncx>>";
      std::string input = "memref<1x1x" + std::to_string(extent) +
                          "x64xf32, #wafer.memory<spm, ncx>>";
      std::string text;
      llvm::raw_string_ostream stream(text);
      stream
          << R"mlir(
#id = affine_map<(b, h, m, k) -> (b, h, m, k)>
#row = affine_map<(b, h, m, k) -> (b, h, m)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %token = arith.constant false
      %unused = wafer.tile.region(%token : i1) -> (i1) {
      ^bb0(%tile_token: i1):
        %zero = arith.constant 0.0 : f32
        %five = arith.constant 5.0 : f32
        %input = memref.alloc() : )mlir"
          << input << R"mlir(
        %buffer = memref.alloc() : )mlir"
          << row << R"mlir(
        wafer.tile.fill %buffer, )mlir"
          << (snapshot ? "%five" : "%zero")
          << (snapshot
                  ? ""
                  : " {fill_domain = #wafer.fill_domain<physical_footprint>}")
          << " : " << row << ", f32\n";
      if (snapshot) {
        stream
            << "        %destination = wafer.tile.materialize_layout %buffer : "
            << row << " -> " << destination << "\n"
            << "        wafer.tile.fill %buffer, %zero : " << row << ", f32\n";
      } else {
        stream << "        %destination = memref.cast %buffer : " << row
               << " to " << destination << "\n";
        if (mode == 2)
          stream << "        wafer.tile.fill %destination, %five {fill_domain "
                    "= #wafer.fill_domain<physical_footprint>} : "
                 << destination << ", f32\n";
      }
      // Sequential reductions must read the preceding result. The alias case
      // targets the original buffer so the intervening view write is relevant.
      std::string output = snapshot ? "%destination" : "%buffer";
      for (unsigned index = 0; index < (mode == 0 ? 2u : 1u); ++index)
        stream << R"mlir(
        linalg.generic {indexing_maps = [#id, #row],
            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%input : )mlir"
               << input << ") outs(" << output << " : " << destination
               << R"mlir() {
        ^bb1(%value: f32, %accumulator: f32):
          %sum = arith.addf %value, %accumulator : f32
          linalg.yield %sum : f32
        }
)mlir";
      stream << R"mlir(
        wafer.tile.yield %tile_token : i1
      }
      return
    }
  }
})mlir";
      auto module = parse(stream.str());
      ASSERT_TRUE(module);
      StructuredMaterializationRelations relations;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(countOps<ComputeReduceOp>(module->getOperation()),
                mode == 0 ? 2u : 1u);
      EXPECT_EQ(countOps<ComputeElementwiseOp>(module->getOperation()), 1u);
      module->walk([&](ComputeReduceOp reduce) {
        auto init = reduce.getInit().getDefiningOp<mlir::arith::ConstantOp>();
        ASSERT_TRUE(init);
        EXPECT_EQ(
            mlir::cast<mlir::FloatAttr>(init.getValue()).getValueAsDouble(),
            0.0);
      });
      EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
    }
  }
}

TEST_F(StructuredToTileTest, PreservesLoopDestinationAndPreexistingView) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::string type = "memref<2x" + std::to_string(extent) +
                       "x1xf16, #wafer.memory<spm, tensor>>";
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry() {
      %token = arith.constant false
      %unused = wafer.tile.region(%token : i1) -> (i1) {
      ^bb0(%tile_token: i1):
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %sixteen = arith.constant 16 : index
        %buffer = memref.alloc() : )mlir"
           << type << R"mlir(
        %result = scf.for %iv = %zero to %sixteen step %one
            iter_args(%state = %buffer) -> ()mlir"
           << type << R"mlir() {
          %view = memref.cast %state : )mlir"
           << type << " to " << type << R"mlir(
          linalg.generic {indexing_maps = [#id, #id],
              iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%state : )mlir"
           << type << ") outs(%state : " << type << R"mlir() {
          ^bb0(%input: f16, %old: f16):
            %twice = arith.addf %input, %input : f16
            linalg.yield %twice : f16
          }
          %observed = wafer.tile.elementwise <add> %view, %view : ()mlir"
           << type << ", " << type << ") -> " << type << R"mlir(
          scf.yield %state : )mlir"
           << type << R"mlir(
        }
        wafer.tile.yield %tile_token : i1
      }
      return
    }
  }
})mlir";
    auto module = parse(stream.str());
    ASSERT_TRUE(module);
    mlir::scf::ForOp loop;
    mlir::memref::CastOp view;
    module->walk([&](mlir::scf::ForOp op) { loop = op; });
    module->walk([&](mlir::memref::CastOp op) { view = op; });
    ASSERT_TRUE(loop);
    ASSERT_TRUE(view);
    auto state = loop.getRegionIterArgs().front();
    StructuredMaterializationRelations relations;
    auto lowered = lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    EXPECT_EQ(yield.getOperand(0), state);
    EXPECT_EQ(view.getSource(), state);
    unsigned writes = 0;
    module->walk([&](MoveCopyIntoOp copy) {
      if (copy.getDest() == state) {
        ++writes;
        EXPECT_TRUE(copy.getSource().getDefiningOp<ComputeElementwiseOp>());
        EXPECT_TRUE(view->isBeforeInBlock(copy));
        EXPECT_TRUE(copy->isBeforeInBlock(yield));
      }
    });
    EXPECT_EQ(writes, 1u);
    EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
  }
}

TEST_F(StructuredToTileTest,
       NativeFinalOutputPreservesLastKBlockAndWideObservers) {
  for (auto dtype : {"f16", "bf16"})
    for (int64_t k : {1024, 1025, 1031})
      for (unsigned mode : {0u, 1u, 2u}) {
        const bool observed = mode == 1;
        const bool external = mode == 2;
        const bool preserved = observed || external;
        SCOPED_TRACE(k);
        SCOPED_TRACE(dtype);
        SCOPED_TRACE(mode);
        const std::string narrow = std::string("memref<2x16x32x") + dtype +
                                   ", #wafer.memory<spm, ncx>>";
        const std::string wide = "memref<2x16x32xf32, #wafer.memory<spm, ncx>>";
        const std::string lhs = "memref<2x16x" + std::to_string(k) + "x" +
                                dtype + ", #wafer.memory<spm, tensor>>";
        const std::string rhs = "memref<2x" + std::to_string(k) + "x32x" +
                                dtype + ", #wafer.memory<spm, tensor>>";
        std::string source = "module { func.func @test(%a: " + lhs +
                             ", %b: " + rhs + ", %init: " + wide + ") -> (" +
                             narrow;
        if (observed)
          source += ", " + wide;
        source += ") { ";
        if (!external)
          source += "%initial = memref.alloc() : " + wide +
                    " wafer.tile.copy_into %init into %initial : " + wide +
                    " into " + wide;
        source += " %c0 = arith.constant 0 : index %c128 = arith.constant 128 "
                  ": index %end = arith.constant " +
                  std::to_string((k / 128) * 128) + " : index ";
        auto block = [&](int64_t size, const std::string &offset,
                         const std::string &psum, const std::string &tag) {
          const auto a = "memref<2x16x" + std::to_string(size) + "x" + dtype;
          const auto b = "memref<2x" + std::to_string(size) + "x32x" + dtype;
          const auto av = a + ", strided<[" + std::to_string(16 * k) + ", " +
                          std::to_string(k) +
                          ", 1], offset: ?>, #wafer.memory<spm, tensor>>";
          const auto bv = b + ", strided<[" + std::to_string(32 * k) +
                          ", 32, 1], offset: ?>, #wafer.memory<spm, tensor>>";
          const auto ap = a + ", #wafer.memory<spm, ncx>>";
          const auto bp = b + ", #wafer.memory<spm, ncx>>";
          return "%av" + tag + " = memref.subview %a[0, 0, " + offset +
                 "] [2, 16, " + std::to_string(size) + "] [1,1,1] : " + lhs +
                 " to " + av + " %bv" + tag + " = memref.subview %b[0, " +
                 offset + ", 0] [2, " + std::to_string(size) +
                 ", 32] [1,1,1] : " + rhs + " to " + bv + " %ap" + tag +
                 " = wafer.tile.materialize_layout %av" + tag + " : " + av +
                 " -> " + ap + " %bp" + tag +
                 " = wafer.tile.materialize_layout %bv" + tag + " : " + bv +
                 " -> " + bp + " %g" + tag + " = wafer.tile.gemm %ap" + tag +
                 ", %bp" + tag + " psum(" + psum + " : " + wide +
                 ") {batch_count = 2 : i64, lhs_batch_dims = array<i64: 0>, "
                 "lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64, "
                 "rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : "
                 "i64, rhs_n_dim = 2 : i64, result_batch_dims = array<i64: 0>, "
                 "result_m_dim = 1 : i64, result_n_dim = 2 : i64} : (" +
                 ap + ", " + bp + ") -> " + wide + " wafer.tile.copy_into %g" +
                 tag + " into " + psum + " : " + wide + " into " + wide + " ";
        };
        source +=
            "%state = scf.for %k = %c0 to %end step %c128 iter_args(%sum = " +
            std::string(external ? "%init" : "%initial") + ") -> (" + wide +
            ") {" + block(128, "%k", "%sum", "main") +
            " scf.yield %sum : " + wide + " } ";
        if (k % 128)
          source += block(k % 128, "%end", "%state", "tail");
        source += R"mlir(
          %tensor = wafer.tile.materialize_layout %state
              : memref<2x16x32xf32, #wafer.memory<spm, ncx>>
              -> memref<2x16x32xf32, #wafer.memory<spm, tensor>>
          %arena = memref.alloc() : memref<2x16x64xf32, #wafer.memory<spm, tensor>>
          %write = memref.subview %arena[0,0,0] [2,16,32] [1,1,1]
              : memref<2x16x64xf32, #wafer.memory<spm, tensor>>
              to memref<2x16x32xf32, strided<[1024,64,1]>, #wafer.memory<spm, tensor>>
          memref.copy %tensor, %write : memref<2x16x32xf32, #wafer.memory<spm, tensor>>
              to memref<2x16x32xf32, strided<[1024,64,1]>, #wafer.memory<spm, tensor>>
          %read = memref.subview %arena[0,0,0] [2,16,32] [1,1,1]
              : memref<2x16x64xf32, #wafer.memory<spm, tensor>>
              to memref<2x16x32xf32, strided<[1024,64,1]>, #wafer.memory<spm, tensor>>
          %copied = wafer.tile.copy %read
              : memref<2x16x32xf32, strided<[1024,64,1]>, #wafer.memory<spm, tensor>>
              -> memref<2x16x32xf32, #wafer.memory<spm, tensor>>
          %final = wafer.tile.materialize_layout %copied
              : memref<2x16x32xf32, #wafer.memory<spm, tensor>>
              -> memref<2x16x32xf32, #wafer.memory<spm, ncx>>
          %dead = memref.cast %final : memref<2x16x32xf32, #wafer.memory<spm, ncx>>
              to memref<2x16x32xf32, #wafer.memory<spm, ncx>>
        )mlir";
        source += "%out = wafer.tile.compute.convert %final : " + wide +
                  " to " + narrow + " return %out";
        if (observed)
          source += ", %state";
        source += " : " + narrow;
        if (observed)
          source += ", " + wide;
        source += " } }";
        auto module = parse(source);
        ASSERT_TRUE(module);
        ASSERT_TRUE(mlir::succeeded(foldGemmOutputConversions(*module)));
        unsigned nativeOutputs = 0;
        module->walk([&](ComputeGemmOp gemm) {
          ASSERT_TRUE(gemm.getPsum());
          EXPECT_TRUE(mlir::cast<mlir::MemRefType>(gemm.getPsum().getType())
                          .getElementType()
                          .isF32());
          nativeOutputs +=
              !mlir::cast<mlir::MemRefType>(gemm.getResult().getType())
                   .getElementType()
                   .isF32();
        });
        EXPECT_EQ(nativeOutputs, preserved ? 0u : 1u);
        EXPECT_EQ(countOps<ComputeConvertOp>(module->getOperation()),
                  preserved ? 1u : 0u);
        if (!preserved)
          module->walk([&](mlir::scf::ForOp loop) {
            EXPECT_EQ(mlir::getConstantIntValue(loop.getUpperBound()),
                      k % 128 ? (k / 128) * 128 : k - 128);
            EXPECT_EQ(mlir::getConstantIntValue(loop.getStep()), 128);
          });
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      }
}

TEST_F(StructuredToTileTest, NativeOutputCrossesOnlyPrivateDDRTransport) {
  for (auto dtype : {"f16", "bf16"})
    for (int64_t k : {1024, 1025, 1031})
      for (bool loop : {false, true})
        // Unique transport, extra F32 observer, external DDR, second writer.
        for (unsigned mode : {0u, 1u, 2u, 3u}) {
          SCOPED_TRACE(dtype);
          SCOPED_TRACE(k);
          SCOPED_TRACE(mode);
          SCOPED_TRACE(loop);
          auto type = [](llvm::StringRef shape, llvm::StringRef element,
                         llvm::StringRef space, llvm::StringRef layout) {
            return "memref<" + shape.str() + "x" + element.str() +
                   ", #wafer.memory<" + space.str() + ", " + layout.str() +
                   ">>";
          };
          auto a = type("2x16x" + std::to_string(k), dtype, "ddr", "tensor");
          auto b =
              type("2x" + std::to_string(k) + "x32", dtype, "ddr", "tensor");
          auto ap = type("2x16x" + std::to_string(k), dtype, "spm", "ncx");
          auto bp = type("2x" + std::to_string(k) + "x32", dtype, "spm", "ncx");
          auto wide = type("2x16x32", "f32", "spm", "ncx");
          auto ddr = type("2x16x32", "f32", "ddr", "tensor");
          auto tensor = type("2x16x32", "f32", "spm", "tensor");
          auto out = type("2x16x32", dtype, "ddr", "tensor");
          auto narrow = type("2x16x32", dtype, "spm", "tensor");
          std::string source = "module { func.func @test(%a: " + a +
                               ", %b: " + b + ", %initial: " + ddr;
          if (mode == 2)
            source += ", %boundary: " + ddr;
          source += ") -> (" + out;
          if (mode == 1)
            source += ", " + ddr;
          source += ") { ";
          if (mode != 2)
            source += "%boundary = memref.alloc() : " + ddr;
          source +=
              " %out = memref.alloc() : " + out +
              " \"wafer.tile.region\"(%a, %b, %initial, %boundary) ({ "
              "^bb0(%aa: " +
              a + ", %bb: " + b + ", %ii: " + ddr + ", %dd: " + ddr +
              "): " + "%ap = memref.alloc() : " + ap +
              " %bp = memref.alloc() : " + bp +
              " %sum = memref.alloc() : " + wide +
              " wafer.tile.load %aa into %ap : " + a + " into " + ap +
              " wafer.tile.load %bb into %bp : " + b + " into " + bp +
              " wafer.tile.load %ii into %sum : " + ddr + " into " + wide +
              " %g = wafer.tile.gemm %ap, %bp psum(%sum : " + wide +
              ") {batch_count = 2 : i64, lhs_batch_dims = array<i64: 0>, "
              "lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64, "
              "rhs_batch_dims = array<i64: 0>, rhs_contracting_dim = 1 : i64, "
              "rhs_n_dim = 2 : i64, result_batch_dims = array<i64: 0>, "
              "result_m_dim = 1 : i64, result_n_dim = 2 : i64} : (" +
              ap + ", " + bp + ") -> " + wide +
              " wafer.tile.copy_into %g into %sum : " + wide + " into " + wide +
              " wafer.tile.store %sum, %dd : " + wide + " -> " + ddr;
          if (mode == 3)
            source += " wafer.tile.store %sum, %dd : " + wide + " -> " + ddr;
          source +=
              " \"wafer.tile.yield\"() : () -> () }) : (" + a + ", " + b +
              ", " + ddr + ", " + ddr +
              ") -> () "
              " \"wafer.tile.region\"(%boundary, %out) ({ ^bb0(%dd: " +
              ddr + ", %oo: " + out +
              "): %loaded = memref.alloc() : " + tensor +
              " wafer.tile.load %dd into %loaded : " + ddr + " into " + tensor +
              " %final = wafer.tile.compute.convert %loaded : " + tensor +
              " to " + narrow + " wafer.tile.store %final, %oo : " + narrow +
              " -> " + out + " \"wafer.tile.yield\"() : () -> () }) : (" + ddr +
              ", " + out + ") -> () return %out";
          if (mode == 1)
            source += ", %boundary";
          source += " : " + out;
          if (mode == 1)
            source += ", " + ddr;
          source += " } }";
          if (loop) {
            // Two actual accumulation blocks, with an arbitrary incoming F32
            // state; the final block is reached only after the first completes.
            const auto begin = source.find(" %g = wafer.tile.gemm");
            const auto end = source.find(" wafer.tile.store %sum", begin);
            auto body = source.substr(begin, end - begin);
            for (size_t at = 0;
                 (at = body.find("%sum", at)) != std::string::npos; at += 8)
              body.replace(at, 4, "%partial");
            const auto loopBody = " %c0 = arith.constant 0 : index %c1 = "
                                  "arith.constant 1 : index "
                                  "%c2 = arith.constant 2 : index "
                                  "%state = scf.for %iv = %c0 to %c2 step %c1 "
                                  "iter_args(%partial = %sum) -> (" +
                                  wide + ") { " + body +
                                  " scf.yield %partial : " + wide + " } ";
            source.replace(begin, end - begin, loopBody);
            for (size_t at = begin + loopBody.size();
                 (at = source.find("wafer.tile.store %sum", at)) !=
                 std::string::npos;
                 at += 22)
              source.replace(at,
                             llvm::StringRef("wafer.tile.store %sum").size(),
                             "wafer.tile.store %state");
          }
          auto module = parse(source);
          ASSERT_TRUE(module) << source;
          ASSERT_TRUE(mlir::succeeded(foldGemmOutputConversions(*module)));
          EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()), 2u);
          module->walk([&](TileRegionOp region) {
            if (countOps<ComputeGemmOp>(region) == 0) {
              EXPECT_EQ(countOps<mlir::memref::AllocOp>(region), 1u);
            }
          });
          EXPECT_EQ(countOps<ComputeConvertOp>(module->getOperation()),
                    mode ? 1u : 0u);
          unsigned nativeOutputs = 0, narrowTransports = 0;
          module->walk([&](ComputeGemmOp gemm) {
            EXPECT_TRUE(mlir::cast<mlir::MemRefType>(gemm.getPsum().getType())
                            .getElementType()
                            .isF32());
            nativeOutputs +=
                !mlir::cast<mlir::MemRefType>(gemm.getResult().getType())
                     .getElementType()
                     .isF32();
          });
          module->walk([&](StorageLoadOp load) {
            auto argument =
                mlir::dyn_cast<mlir::BlockArgument>(load.getSource());
            if (!argument)
              return;
            auto region =
                mlir::cast<TileRegionOp>(argument.getOwner()->getParentOp());
            if (!region.getInputs()[argument.getArgNumber()]
                     .getDefiningOp<mlir::memref::AllocOp>())
              return;
            auto sourceType =
                mlir::cast<mlir::MemRefType>(load.getSource().getType());
            EXPECT_EQ(sourceType.getElementType(),
                      mlir::cast<mlir::MemRefType>(load.getDest().getType())
                          .getElementType());
            narrowTransports += !sourceType.getElementType().isF32();
          });
          EXPECT_EQ(nativeOutputs, mode ? 0u : 1u);
          if (loop) {
            EXPECT_EQ(countOps<mlir::scf::ForOp>(module->getOperation()), 1u);
            module->walk([&](mlir::scf::ForOp op) {
              EXPECT_EQ(mlir::getConstantIntValue(op.getUpperBound()),
                        mode ? 2 : 1);
              EXPECT_EQ(countOps<ComputeGemmOp>(op), 1u);
              op.walk([&](ComputeGemmOp gemm) {
                EXPECT_TRUE(
                    mlir::cast<mlir::MemRefType>(gemm.getResult().getType())
                        .getElementType()
                        .isF32());
              });
            });
          }
          EXPECT_EQ(narrowTransports, mode ? 0u : 1u);
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        }
}

TEST_F(StructuredToTileTest,
       PreservesWideStateAndNativeOutputThroughLayoutAndBufferization) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool bf16 : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(bf16);
      auto source = makeSource(extent);
      if (bf16)
        for (size_t at = 0; (at = source.find("f16", at)) != std::string::npos;
             at += 4)
          source.replace(at, 3, "bf16");
      auto module = parse(source);
      ASSERT_TRUE(module);
      module->walk([&](mlir::func::FuncOp function) {
        ASSERT_TRUE(mlir::succeeded(promoteContractionAccumulation(function)));
      });
      TileRegionOp region;
      module->walk([&](TileRegionOp op) { region = op; });
      ASSERT_TRUE(region);
      StructuredMaterializationRelations relations;
      relations.structuralOutputs.push_back({0, region.getResult(0)});
      relations.structuralOutputs.push_back({1, region.getResult(1)});
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(lowered.statistics.contractions, 1u);
      EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      module->walk([&](ComputeGemmOp gemm) {
        auto lhs = mlir::cast<mlir::MemRefType>(gemm.getLhs().getType());
        auto rhs = mlir::cast<mlir::MemRefType>(gemm.getRhs().getType());
        auto output = mlir::cast<mlir::MemRefType>(gemm.getResult().getType());
        EXPECT_EQ(lhs.getElementType(), rhs.getElementType());
        EXPECT_TRUE(bf16 ? lhs.getElementType().isBF16()
                         : lhs.getElementType().isF16());
        EXPECT_EQ(output.getElementType(), lhs.getElementType());
        if (gemm.getPsum()) {
          EXPECT_TRUE(mlir::cast<mlir::MemRefType>(gemm.getPsum().getType())
                          .getElementType()
                          .isF32());
        }
        EXPECT_EQ(output.getShape(), llvm::ArrayRef<int64_t>({2, extent, 32}));
      });
      EXPECT_EQ(countOps<ComputeGemmOp>(module->getOperation()), 1u);
      EXPECT_EQ(countOps<ComputeConvertOp>(module->getOperation()), 0u);
      EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
      EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
    }
}

struct ContractionExample {
  llvm::SmallVector<int64_t> extents;
  llvm::SmallVector<unsigned> lhs, rhs, output, batch, m, n;
  unsigned k;
};

static ContractionExample multiAxisExample(int64_t extent, unsigned variant) {
  switch (variant) {
  case 0:
    return {{2, 1, extent, 8, 8},
            {1, 0, 2, 3},
            {0, 3, 4},
            {0, 1, 2, 4},
            {0},
            {1, 2},
            {4},
            3};
  case 1:
    return {{2, 3, extent, 8, 8},
            {1, 0, 2, 3},
            {0, 3, 4},
            {4, 0, 2, 1},
            {0},
            {1, 2},
            {4},
            3};
  case 2:
    return {{2, 8, 8, 3, extent},
            {1, 0, 2},
            {4, 2, 0, 3},
            {3, 0, 4, 1},
            {0},
            {1},
            {3, 4},
            2};
  case 3:
    return {{2, extent, 8, 2, 8},
            {2, 1, 0},
            {4, 2, 3},
            {4, 0, 3, 1},
            {},
            {0, 1},
            {3, 4},
            2};
  case 4:
    return {
        {3, extent, 8, 8}, {2, 0, 1}, {3, 2}, {3, 0, 1}, {}, {0, 1}, {3}, 2};
  case 5:
    return {
        {8, 8, 2, extent}, {1, 0}, {2, 1, 3}, {3, 0, 2}, {}, {0}, {2, 3}, 1};
  default:
    return {{2, 2, 2, extent, 8, 2, 8},
            {3, 0, 4, 2, 1},
            {6, 1, 4, 0, 5},
            {5, 3, 0, 6, 2, 1},
            {0, 1},
            {2, 3},
            {5, 6},
            4};
  }
}

static std::string multiAxisSource(const ContractionExample &example,
                                   llvm::StringRef element) {
  auto type = [&](llvm::ArrayRef<unsigned> dimensions, llvm::StringRef dtype) {
    std::string result = "tensor<";
    for (unsigned dim : dimensions)
      result += std::to_string(example.extents[dim]) + "x";
    return result + dtype.str() + ">";
  };
  auto map = [&](llvm::ArrayRef<unsigned> dimensions) {
    std::string result = "affine_map<(";
    for (unsigned i = 0; i < example.extents.size(); ++i)
      result += (i ? "," : "") + std::string("d") + std::to_string(i);
    result += ")->(";
    for (auto [i, dim] : llvm::enumerate(dimensions))
      result += (i ? "," : "") + std::string("d") + std::to_string(dim);
    return result + ")>";
  };
  std::string a = type(example.lhs, element), b = type(example.rhs, element);
  std::string c = type(example.output, "f32");
  std::string iterators;
  for (unsigned i = 0; i < example.extents.size(); ++i)
    iterators += (i ? "," : "") +
                 std::string(i == example.k ? "\"reduction\"" : "\"parallel\"");
  return "module { wafer.tile.module card_id = 0 tile_id = 0 { "
         "func.func @entry(%a: " +
         a + ", %b: " + b + ", %initial: " + c + ") -> " + c +
         " { %result = wafer.tile.region(%a, %b, %initial : " + a + ", " + b +
         ", " + c + ") -> (" + c + ") { ^bb0(%aa: " + a + ", %bb: " + b +
         ", %cc: " + c +
         "): "
         "%r = linalg.generic {indexing_maps = [" +
         map(example.lhs) + "," + map(example.rhs) + "," + map(example.output) +
         "], iterator_types = [" + iterators + "]} ins(%aa, %bb : " + a + ", " +
         b + ") outs(%cc : " + c + ") { ^bb0(%x: " + element.str() +
         ", %y: " + element.str() +
         ", %z: f32): "
         "%xx = arith.extf %x : " +
         element.str() +
         " to f32 "
         "%yy = arith.extf %y : " +
         element.str() +
         " to f32 "
         "%p = arith.mulf %xx, %yy : f32 %s = arith.addf %p, %z : f32 "
         "linalg.yield %s : f32 } -> " +
         c + " wafer.tile.yield %r : " + c + " } return %result : " + c +
         " } } }";
}

// Interpret only the actual movement chain, independently of the descriptor
// construction. Linear indices are logical row-major, including ragged axes.
static mlir::FailureOr<int64_t>
traceContractionElement(mlir::Value value, mlir::Value anchor, int64_t index) {
  if (value == anchor)
    return index;
  if (auto op = value.getDefiningOp<MoveTransposeOp>()) {
    auto out = mlir::cast<mlir::MemRefType>(value.getType());
    auto in = mlir::cast<mlir::MemRefType>(op.getSource().getType());
    llvm::SmallVector<int64_t> coordinates(in.getRank(), 0);
    for (int64_t dim = out.getRank() - 1; dim >= 0; --dim) {
      coordinates[op.getPermutation()[dim]] = index % out.getDimSize(dim);
      index /= out.getDimSize(dim);
    }
    int64_t sourceIndex = 0;
    for (int64_t dim = 0; dim < in.getRank(); ++dim)
      sourceIndex = sourceIndex * in.getDimSize(dim) + coordinates[dim];
    return traceContractionElement(op.getSource(), anchor, sourceIndex);
  }
  if (auto op = value.getDefiningOp<ViewReshapeOp>())
    return traceContractionElement(op.getSource(), anchor, index);
  if (auto op = value.getDefiningOp<MoveReshapeOp>())
    return traceContractionElement(op.getSource(), anchor, index);
  if (auto op = value.getDefiningOp<LayoutMaterializeOp>())
    return traceContractionElement(op.getSource(), anchor, index);
  return mlir::failure();
}

TEST_F(StructuredToTileTest,
       MultiAxisContractionsPreserveEveryLogicalCoordinate) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned variant = 0; variant < 7; ++variant)
      for (llvm::StringRef element : {"f16", "bf16"}) {
        SCOPED_TRACE(::testing::Message()
                     << extent << "/" << variant << "/" << element.str());
        auto example = multiAxisExample(extent, variant);
        auto module = parse(multiAxisSource(example, element));
        ASSERT_TRUE(module);
        TileRegionOp region;
        module->walk([&](TileRegionOp op) { region = op; });
        StructuredMaterializationRelations relations;
        relations.structuralOutputs.push_back({0, region.getResult(0)});
        auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        mlir::linalg::GenericOp source;
        module->walk([&](mlir::linalg::GenericOp op) { source = op; });
        ASSERT_TRUE(source);
        mlir::Value lhs = source.getDpsInputs()[0];
        mlir::Value rhs = source.getDpsInputs()[1];
        mlir::Value destination = source.getDpsInits()[0];
        auto lowered = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
        EXPECT_EQ(lowered.statistics.contractions, 1u);
        EXPECT_EQ(lowered.statistics.converts, 0u);
        EXPECT_EQ(countOps<ComputeConvertOp>(*module), 0u);
        ComputeGemmOp gemm;
        mlir::Value published;
        module->walk([&](ComputeGemmOp op) { gemm = op; });
        module->walk([&](MoveCopyIntoOp op) {
          if (op.getDest() == destination)
            published = op.getSource();
        });
        ASSERT_TRUE(gemm);
        ASSERT_TRUE(published);
        ASSERT_TRUE(gemm.getPsum());
        EXPECT_TRUE(mlir::cast<mlir::MemRefType>(gemm.getResult().getType())
                        .getElementType()
                        .isF32());
        EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
        auto unpack = [&](int64_t index, llvm::ArrayRef<unsigned> dims,
                          llvm::SmallVectorImpl<int64_t> &coordinates) {
          for (unsigned dim : llvm::reverse(dims)) {
            coordinates[dim] = index % example.extents[dim];
            index /= example.extents[dim];
          }
        };
        auto project = [&](llvm::ArrayRef<unsigned> dims,
                           llvm::ArrayRef<int64_t> coordinates) {
          int64_t index = 0;
          for (unsigned dim : dims)
            index = index * example.extents[dim] + coordinates[dim];
          return index;
        };
        auto product = [&](llvm::ArrayRef<unsigned> dims) {
          int64_t count = 1;
          for (unsigned dim : dims)
            count *= example.extents[dim];
          return count;
        };
        int64_t batch = product(example.batch), m = product(example.m);
        int64_t n = product(example.n), k = example.extents[example.k];
        EXPECT_EQ(
            mlir::cast<mlir::MemRefType>(gemm.getResult().getType()).getShape(),
            (llvm::ArrayRef<int64_t>{batch, m, n}));
        for (unsigned operand = 0; operand < 3; ++operand) {
          mlir::Value value = operand == 0   ? gemm.getLhs()
                              : operand == 1 ? gemm.getRhs()
                                             : gemm.getPsum();
          mlir::Value anchor = operand == 0   ? lhs
                               : operand == 1 ? rhs
                                              : destination;
          auto dims = operand == 0   ? example.lhs
                      : operand == 1 ? example.rhs
                                     : example.output;
          auto shape = mlir::cast<mlir::MemRefType>(value.getType());
          int64_t rows = operand == 1 ? k : m;
          int64_t cols = operand == 0 ? k : n;
          ASSERT_EQ(shape.getShape(),
                    (llvm::ArrayRef<int64_t>{batch, rows, cols}));
          for (int64_t i = 0; i < batch * rows * cols; ++i) {
            llvm::SmallVector<int64_t> coordinates(example.extents.size(), 0);
            unpack(i / (rows * cols), example.batch, coordinates);
            if (operand == 1)
              coordinates[example.k] = (i / cols) % rows;
            else
              unpack((i / cols) % rows, example.m, coordinates);
            if (operand == 0)
              coordinates[example.k] = i % cols;
            else
              unpack(i % cols, example.n, coordinates);
            int64_t expected = project(dims, coordinates);
            auto actual = traceContractionElement(value, anchor, i);
            ASSERT_TRUE(mlir::succeeded(actual));
            ASSERT_EQ(*actual, expected);
            if (operand == 2) {
              auto restored = traceContractionElement(
                  published, gemm.getResult(), expected);
              ASSERT_TRUE(mlir::succeeded(restored));
              ASSERT_EQ(*restored, i);
            }
          }
        }
        EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
            module->getOperation(), relations)));
        auto movement = materializeTileBoundaryMovement(*module, relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
      }
}

TEST_F(StructuredToTileTest, MultiAxisContractionsReachInstrWithTemporalTails) {
  // Four independently owned Tile lanes; each lane has a realistic M traversal.
  // The temporal choice is applied by the production transformation.
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(multiAxisSource(multiAxisExample(extent, 1), "f16"));
    ASSERT_TRUE(module);
    auto original = *module->getOps<TileModuleOp>().begin();
    for (int64_t id = 1; id < 4; ++id) {
      mlir::IRMapping mapping;
      auto clone = mlir::cast<TileModuleOp>(original->clone(mapping));
      clone.setTileIdAttr(mlir::IntegerAttr::get(
          mlir::IntegerType::get(context.get(), 64), id));
      module->getBody()->push_back(clone);
    }
    StructuredMaterializationRelations relations;
    llvm::SmallVector<TileRegionOp> regions;
    module->walk([&](TileRegionOp region) {
      regions.push_back(region);
      relations.structuralOutputs.push_back({0, region.getResult(0)});
    });
    for (TileRegionOp region : regions) {
      auto domain = buildTemporalDomain(region);
      ASSERT_TRUE(domain.succeeded());
      auto first = domain.domain->getFirstChoice();
      auto choice = *first.getChoice();
      ASSERT_EQ(choice.scopes.size(), 1u);
      choice.scopes[0].iteratorTileSizes[2] = 128;
      choice.scopes[0].loopOrder = {2};
      ASSERT_TRUE(domain.domain->contains(choice));
      ASSERT_TRUE(mlir::succeeded(
          applyTemporalTiling(*domain.domain, choice, relations)));
    }
    auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    auto lowered = lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    unsigned mainOps = 0, tailOps = 0;
    module->walk([&](ComputeGemmOp op) {
      auto shape = mlir::cast<mlir::MemRefType>(op.getResult().getType());
      EXPECT_EQ(shape.getDimSize(0), 2);
      EXPECT_EQ(shape.getDimSize(2), 8);
      if (shape.getDimSize(1) == 3 * 128) {
        ++mainOps;
        auto loop = op->getParentOfType<mlir::scf::ForOp>();
        ASSERT_TRUE(loop);
        auto lower = mlir::getConstantIntValue(loop.getLowerBound());
        auto upper = mlir::getConstantIntValue(loop.getUpperBound());
        auto step = mlir::getConstantIntValue(loop.getStep());
        ASSERT_TRUE(lower && upper && step);
        EXPECT_EQ(*lower, 0);
        EXPECT_EQ(*upper, 1024);
        EXPECT_EQ(*step, 128);
      } else {
        ++tailOps;
        EXPECT_EQ(shape.getDimSize(1), 3 * (extent % 128));
        EXPECT_FALSE(op->getParentOfType<mlir::scf::ForOp>());
      }
    });
    EXPECT_EQ(mainOps, 4u);
    EXPECT_EQ(tailOps, extent % 128 ? 4u : 0u);
    auto movement = materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    ASSERT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
    std::string detail;
    auto standalone =
        createStandaloneTileModules(std::move(module), &detail, &relations);
    ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
    ASSERT_EQ(standalone->size(), 4u);
    for (auto &tile : *standalone) {
      llvm::SmallVector<TileRegionOp> currentRegions;
      tile.module->walk(
          [&](TileRegionOp region) { currentRegions.push_back(region); });
      TileRegionToInstrLoweringSession session(*context);
      for (TileRegionOp region : currentRegions)
        ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(
          convertBufferizationCopiesToInstr(*tile.module, session)));
      EXPECT_EQ(countOps<ComputeGemmOp>(*tile.module), 0u);
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
      TileMemoryPlanningFailure failure;
      auto planned = planTileMemory(std::move(tile.module), &failure);
      ASSERT_TRUE(mlir::succeeded(planned))
          << static_cast<unsigned>(failure.kind);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
    }
  }
}

TEST_F(StructuredToTileTest,
       MultiAxisContractionsRejectIncompleteMapsAtomically) {
  for (unsigned variant = 0; variant < 3; ++variant) {
    SCOPED_TRACE(variant);
    // These verifier negatives isolate unsupported classification and int64
    // product overflow. No physical allocation or execution is requested.
    std::string source = R"mlir(module {
      wafer.tile.module card_id = 0 tile_id = 0 {
        func.func @entry() {
          wafer.tile.region() -> () {
            %a = memref.alloc() : memref<2x1024x8x8xf16, #wafer.memory<spm, ncx>>
            %b = memref.alloc() : memref<2x8x8x4xf16, #wafer.memory<spm, ncx>>
            %c = memref.alloc() : memref<2x1024x4xf16, #wafer.memory<spm, ncx>>
            linalg.generic {indexing_maps = [
              affine_map<(b,m,k0,k1,n)->(b,m,k0,k1)>,
              affine_map<(b,m,k0,k1,n)->(b,k0,k1,n)>,
              affine_map<(b,m,k0,k1,n)->(b,m,n)>],
              iterator_types = ["parallel","parallel","reduction","reduction","parallel"]}
              ins(%a, %b : memref<2x1024x8x8xf16, #wafer.memory<spm, ncx>>,
                            memref<2x8x8x4xf16, #wafer.memory<spm, ncx>>)
              outs(%c : memref<2x1024x4xf16, #wafer.memory<spm, ncx>>) {
              ^bb0(%x: f16, %y: f16, %z: f16):
                %p = arith.mulf %x, %y : f16
                %s = arith.addf %p, %z : f16
                linalg.yield %s : f16
            }
            wafer.tile.yield
          }
          return
        }
      }
    })mlir";
    auto replaceAll = [&](llvm::StringRef from, llvm::StringRef to) {
      for (size_t at = 0;
           (at = source.find(from.str(), at)) != std::string::npos;
           at += to.size())
        source.replace(at, from.size(), to.str());
    };
    if (variant != 0) {
      // k0 becomes a lhs-only parallel axis (variant 1) or a second M
      // dimension with overflowing static element count (variant 2).
      replaceAll("(b,k0,k1,n)", "(b,k1,n)");
      replaceAll("2x8x8x4xf16", "2x8x4xf16");
      replaceAll("\"reduction\",\"reduction\"", "\"parallel\",\"reduction\"");
    }
    if (variant == 2) {
      replaceAll("->(b,m,n)", "->(b,m,k0,n)");
      replaceAll("2x1024x4xf16", "2x1024x8x4xf16");
      replaceAll("1024", "4611686018427387903");
    }
    auto module = parse(source);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    StructuredMaterializationRelations relations;
    std::string before;
    llvm::raw_string_ostream stream(before);
    module->print(stream);
    auto lowered = lowerStructuredComputeToTile(*module, relations);
    EXPECT_EQ(lowered.failure, StructuredToTileFailureKind::Unsupported);
    std::string after;
    llvm::raw_string_ostream afterStream(after);
    module->print(afterStream);
    EXPECT_EQ(before, after);
    EXPECT_EQ(countOps<ComputeGemmOp>(*module), 0u);
    EXPECT_TRUE(relations.buffers.empty());
  }
}

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
       BoundaryMovementErasesSharedResultBridgeExactlyOnce) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeAliasedResultSource(extent));
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
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.outputCopiesRemoved, 2u);
    EXPECT_EQ(countOps<mlir::bufferization::ToTensorOp>(module->getOperation()),
              0u);
    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        module->getOperation(), relations)));
  }
}

TEST_F(StructuredToTileTest,
       LowersOrdinaryConvolutionFromCurrentMapsWithoutShapeNames) {
  for (int64_t extent : {1024, 1025, 1031}) {
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
    module->walk([&](ComputeConvOp convolution) {
      auto weight =
          mlir::cast<mlir::MemRefType>(convolution.getWeight().getType());
      EXPECT_EQ(weight.getShape(), (llvm::ArrayRef<int64_t>{2, 3, 32, 16}));
      EXPECT_EQ(getWaferMemoryAttr(weight).getLayout(), MemLayout::Cx);
      auto transpose = convolution.getWeight().getDefiningOp<MoveTransposeOp>();
      ASSERT_TRUE(transpose);
      EXPECT_EQ(transpose.getPermutation(),
                (llvm::ArrayRef<int64_t>{0, 1, 3, 2}));
    });
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
  for (int64_t extent : {1024, 1025, 1031}) {
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
    // Model an imported shard whose final tensor read was eliminated. Keep
    // the original live input as a witness that relation cleanup is exact.
    ASSERT_EQ(relations.boundaryRelations.size(), 1u);
    auto liveRelation = relations.boundaryRelations.front();
    auto liveArgument =
        mlir::cast<mlir::BlockArgument>(liveRelation.destinationEndpoint);
    auto destination =
        mlir::cast<TileRegionOp>(liveArgument.getOwner()->getParentOp());
    destination.getInputsMutable().append(destination.getInputs().front());
    auto unused = destination.getBody().front().addArgument(
        destination.getInputs().front().getType(), destination.getLoc());
    relations.boundaryRelations.push_back(
        {liveRelation.sourceEndpoint, unused});
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    retainCurrentStructuredBufferRelations(*module, relations);
    ASSERT_EQ(relations.boundaryRelations.size(), 1u);
    EXPECT_EQ(relations.boundaryRelations.front().sourceEndpoint,
              liveRelation.sourceEndpoint);
    EXPECT_EQ(relations.boundaryRelations.front().destinationEndpoint,
              liveRelation.destinationEndpoint);
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
       PhysicalCopyPlacementPreservesAliasAndActualMemoryContracts) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (unsigned variant = 0; variant < 9; ++variant) {
      SCOPED_TRACE(::testing::Message() << extent << "/" << variant);
      std::string sourceType = "memref<2x1x" + std::to_string(extent) +
                               "x64xf16, #wafer.memory<spm, tensor>>";
      std::string resultType = "memref<2x" + std::to_string(extent) +
                               "x64xf16, #wafer.memory<spm, tensor>>";
      std::string text;
      llvm::raw_string_ostream ir(text);
      ir << "module {\n";
      for (unsigned tile = 0; tile < 4; ++tile) {
        ir << "wafer.tile.module card_id = 0 tile_id = " << tile
           << " { func.func private @unknown_effect()\n"
              "func.func @entry() { wafer.tile.region() -> () {\n"
           << "%source = memref.alloc() : " << sourceType << "\n"
           << "%destination = memref.alloc() : " << resultType << "\n"
           << "%other = memref.alloc() : " << sourceType << "\n"
           << "%another = memref.alloc() : " << sourceType << "\n"
           << "%observable = memref.alloc() : memref<1x1024x1024xf16, "
              "#wafer.memory<ddr, tensor>>\n"
           << "%zero = arith.constant 0 : index\n"
           << "%step = arith.constant 128 : index\n"
           << "%end = arith.constant " << (variant == 3 ? 0 : extent)
           << " : index\n%value = arith.constant 0.0 : f16\n";
        if (variant == 7)
          ir << "%escaped = ";
        ir << "scf.for %iv = %zero to %end step %step";
        if (variant == 7)
          ir << " iter_args(%carried = %destination) -> (" << resultType << ")";
        ir << " {\n";
        if (variant == 6)
          ir << "%temporary = memref.alloc() : memref<1x1024x1024xf16, "
                "#wafer.memory<spm, tensor>>\n"
                "wafer.tile.fill %temporary, %value : "
                "memref<1x1024x1024xf16, #wafer.memory<spm, tensor>>, f16\n"
                "wafer.tile.store %temporary, %observable : "
                "memref<1x1024x1024xf16, #wafer.memory<spm, tensor>> -> "
                "memref<1x1024x1024xf16, #wafer.memory<ddr, tensor>>\n"
             << "wafer.tile.copy_into %source into %other : " << sourceType
             << " into " << sourceType << "\n";
        ir << "%copy = wafer.tile.reshape_copy %source : " << sourceType
           << " -> " << resultType << "\n";
        if (variant == 1)
          ir << "%alias = memref.cast %source : " << sourceType << " to "
             << sourceType
             << "\nmemref.store %value, %alias[%zero, %zero, "
                "%zero, %zero] : "
             << sourceType << "\n";
        if (variant == 2)
          ir << "wafer.tile.fill %copy, %value : " << resultType << ", f16\n";
        if (variant == 8)
          ir << "func.call @unknown_effect() : () -> ()\n";
        if (variant == 4 || variant == 5)
          ir << "%condition = arith.cmpi eq, %iv, %zero : index\n"
             << "%selected = arith.select %condition, %other, %"
             << (variant == 4 ? "another" : "source") << " : " << sourceType
             << "\nwafer.tile.fill %selected, %value : " << sourceType
             << ", f16\n";
        ir << "wafer.tile.copy_into %copy into %destination : " << resultType
           << " into " << resultType << "\n";
        if (variant == 7)
          ir << "scf.yield %copy : " << resultType << "\n";
        ir << "}\nwafer.tile.yield\n}\nreturn\n}}\n";
      }
      ir << "}\n";
      for (auto placement : {LayoutMaterializationPlacement::FirstUse,
                             LayoutMaterializationPlacement::LoopInvariant}) {
        auto module = parse(text);
        ASSERT_TRUE(module) << text;
        StructuredMaterializationRelations relations;
        const bool hoisted =
            (variant == 0 || variant == 4 || variant == 6) &&
            placement == LayoutMaterializationPlacement::LoopInvariant;
        auto moved =
            optimizePhysicalMovementPlacement(*module, relations, placement);
        ASSERT_TRUE(mlir::succeeded(moved));
        EXPECT_EQ(*moved, hoisted ? 4u : 0u);
        module->walk([&](MoveReshapeOp copy) {
          EXPECT_EQ(bool(copy->getParentOfType<mlir::scf::ForOp>()), !hoisted);
        });
        if (variant != 0 && variant != 4 && variant != 6)
          continue;
        std::string failure;
        auto standalone = createStandaloneTileModules(std::move(module),
                                                      &failure, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << failure;
        ASSERT_EQ(standalone->size(), 4u);
        for (auto &tile : *standalone) {
          TileRegionToInstrLoweringSession session(*tile.module->getContext());
          llvm::SmallVector<TileRegionOp> regions;
          tile.module->walk(
              [&](TileRegionOp region) { regions.push_back(region); });
          for (auto region : regions)
            ASSERT_TRUE(
                mlir::succeeded(convertTileRegionToInstr(region, session)));
          unsigned outside = 0, inside = 0;
          tile.module->walk([&](InstrGatherScatterOp copy) {
            if (copy->getParentOfType<mlir::scf::ForOp>())
              ++inside;
            else
              ++outside;
          });
          EXPECT_EQ(outside, hoisted ? 1u : 0u);
          EXPECT_EQ(inside, (variant == 6 ? 3u : 2u) - outside);
          const uint64_t waves = (extent + 127) / 128;
          EXPECT_EQ(outside + inside * waves, (variant == 6 ? 3u : 2u) * waves -
                                                  (hoisted ? waves - 1 : 0));
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
          TileMemoryPlanningFailure memoryFailure;
          auto planned = planTileMemory(std::move(tile.module), &memoryFailure);
          if (variant == 6 && hoisted) {
            std::string actual;
            if (mlir::succeeded(planned)) {
              llvm::raw_string_ostream stream(actual);
              (*planned)->print(stream);
            }
            ASSERT_TRUE(mlir::failed(planned)) << actual;
            EXPECT_TRUE(memoryFailure.spmCapacityOverflow);
          } else {
            ASSERT_TRUE(mlir::succeeded(planned));
            EXPECT_TRUE(mlir::succeeded(mlir::verify(**planned)));
          }
        }
      }
    }
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

TEST_F(StructuredToTileTest,
       FullFanoutFromHighTileUsesRegionRanksAndMinimumRounds) {
  constexpr int64_t destinations[] = {0, 1, 2,  3,  4,  5,  6, 7,
                                      8, 9, 10, 11, 12, 13, 14};
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeFanoutSource(extent, destinations,
                                         /*sourceTile=*/15,
                                         /*hasUnavailableTile=*/false));
    ASSERT_TRUE(module);
    llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 16> regions;
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) {
        regions.push_back({tile.getTileIdAttr().getInt(), region});
      });
    llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    ASSERT_EQ(regions.size(), 16u);

    StructuredMaterializationRelations relations;
    TileRegionOp source = regions.back().second;
    for (unsigned index = 0; index + 1 < regions.size(); ++index) {
      relations.boundaryRelations.push_back(
          {source.getResult(0),
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
    EXPECT_EQ(movement.statistics.peerSends, 15u);
    EXPECT_EQ(movement.statistics.peerReceives, 15u);
    EXPECT_EQ(movement.statistics.topologyFanoutGroups, 1u);
    EXPECT_EQ(movement.statistics.topologyFanoutRounds, 4u);

    EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
  }
}

TEST_F(StructuredToTileTest,
       QualifiedExactPayloadReachesNativeBroadcastInstrGrouping) {
  // The 256-byte payload is the exact calibrated native DTE contract. The
  // rank-3 1025-scale source/range coverage lives in DirectDTETransportTest;
  // this fixture isolates the Tile-to-Instr production handoff.
  constexpr int64_t destinations[] = {1, 2};
  auto module = parse(makeFanoutSource(/*extent=*/2, destinations));
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
  BoundaryMovementResult movement = materializeTileBoundaryMovement(
      *module, relations,
      BoundaryMovementOptions{CompleteAllGatherAlgorithm::RecursiveDoubling,
                              CompleteAllToAllAlgorithm::Direct});
  ASSERT_TRUE(movement.succeeded()) << movement.detail;
  EXPECT_EQ(movement.statistics.nativeBroadcastGroups, 1u);
  EXPECT_EQ(movement.statistics.topologyFanoutGroups, 0u);
  EXPECT_EQ(movement.statistics.peerSends, 2u);

  std::string failure;
  auto standalone =
      createStandaloneTileModules(std::move(module), &failure, &relations);
  ASSERT_TRUE(mlir::succeeded(standalone)) << failure;
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
  auto grouped = materializeNativeDirectDTEMultiSends(instructionModules);
  ASSERT_TRUE(grouped.succeeded()) << grouped.detail;
  EXPECT_EQ(grouped.statistics.broadcastOperations, 1u);
  EXPECT_EQ(grouped.statistics.unicastSendOperationsRemoved, 2u);
}

TEST_F(StructuredToTileTest,
       SeparateSourceAllocationsDoNotFabricateNativeScatter) {
  auto module = parse(makeThreeTileAllToAllSource());
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
  for (int64_t source = 0; source < 3; ++source) {
    unsigned sourceResult = 0;
    for (int64_t destination = 0; destination < 3; ++destination) {
      if (destination == source)
        continue;
      unsigned destinationArgument = 1;
      for (int64_t candidate = 0; candidate < source; ++candidate)
        destinationArgument += candidate != destination;
      relations.boundaryRelations.push_back(
          {regions[source].second.getResult(sourceResult++),
           regions[destination].second.getBody().getArgument(
               destinationArgument)});
    }
  }
  for (auto &[tile, region] : regions) {
    (void)tile;
    relations.structuralOutputs.push_back({0, region.getResult(2)});
  }
  LayoutOptimizationResult layout =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  ASSERT_TRUE(layout.succeeded()) << layout.detail;
  StructuredToTileResult lowered =
      lowerStructuredComputeToTile(*module, relations);
  ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
  BoundaryMovementResult movement = materializeTileBoundaryMovement(
      *module, relations,
      BoundaryMovementOptions{CompleteAllGatherAlgorithm::Ring,
                              CompleteAllToAllAlgorithm::DimensionOrdered,
                              DistributedReductionAlgorithm::Centralized});
  ASSERT_TRUE(movement.succeeded()) << movement.detail;
  EXPECT_EQ(movement.statistics.nativeScatterGroups, 0u);
  EXPECT_EQ(movement.statistics.dimensionOrderedAllToAllComponents, 0u);
  EXPECT_EQ(movement.statistics.nativeScatterRounds, 0u);
  EXPECT_EQ(movement.statistics.sparseRoundComponents, 1u);
  EXPECT_EQ(movement.statistics.sparseRounds, 2u);
  EXPECT_EQ(movement.statistics.peerSends, 6u);

  std::string failure;
  auto standalone =
      createStandaloneTileModules(std::move(module), &failure, &relations);
  ASSERT_TRUE(mlir::succeeded(standalone)) << failure;
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
  auto grouped = materializeNativeDirectDTEMultiSends(instructionModules);
  ASSERT_TRUE(grouped.succeeded()) << grouped.detail;
  EXPECT_EQ(grouped.statistics.scatterOperations, 0u);
  EXPECT_EQ(grouped.statistics.unicastSendOperationsRemoved, 0u);
}

TEST_F(StructuredToTileTest,
       CompleteAllToAllUsesActualDimensionOrderedMeshAggregates) {
  for (int64_t tileCount : {4, 16}) {
    for (auto [extent, bf16] : {std::pair<int64_t, bool>{1024, false},
                                std::pair<int64_t, bool>{1025, false},
                                std::pair<int64_t, bool>{1031, false},
                                std::pair<int64_t, bool>{1025, true}}) {
      SCOPED_TRACE((llvm::Twine("tiles=") + llvm::Twine(tileCount) +
                    ", extent=" + llvm::Twine(extent) +
                    ", dtype=" + (bf16 ? "bf16" : "f16"))
                       .str());
      auto module = parse(makeDenseAllToAllSource(extent, tileCount, bf16));
      ASSERT_TRUE(module);
      llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 16> regions;
      for (TileModuleOp tile : module->getOps<TileModuleOp>())
        tile.walk([&](TileRegionOp region) {
          regions.push_back({tile.getTileIdAttr().getInt(), region});
        });
      llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
      });
      ASSERT_EQ(regions.size(), static_cast<size_t>(tileCount));

      StructuredMaterializationRelations relations;
      for (unsigned source = 0; source < regions.size(); ++source)
        for (unsigned destination = 0; destination < regions.size();
             ++destination) {
          if (source == destination)
            continue;
          unsigned sourceResult =
              destination < source ? destination : destination - 1;
          unsigned destinationArgument =
              1 + (source < destination ? source : source - 1);
          relations.boundaryRelations.push_back(
              {regions[source].second.getResult(sourceResult),
               regions[destination].second.getBody().getArgument(
                   destinationArgument)});
        }
      for (auto &[tile, region] : regions) {
        (void)tile;
        relations.structuralOutputs.push_back(
            {0, region.getResult(tileCount - 1)});
      }
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      StructuredToTileResult lowered =
          lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      DistributedMovementAvailability availability =
          analyzeDistributedMovementAvailability(*module, relations);
      EXPECT_FALSE(availability.brokenContract) << availability.detail;
      EXPECT_TRUE(availability.dimensionOrderedAllToAll);
      BoundaryMovementResult movement = materializeTileBoundaryMovement(
          *module, relations,
          BoundaryMovementOptions{CompleteAllGatherAlgorithm::Ring,
                                  CompleteAllToAllAlgorithm::DimensionOrdered,
                                  DistributedReductionAlgorithm::Ring});
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.dimensionOrderedAllToAllComponents, 1u);
      EXPECT_EQ(movement.statistics.ringReduceScatterComponents, 0u);
      EXPECT_EQ(movement.statistics.ringAllReduceComponents, 0u);
      unsigned side = tileCount == 4 ? 2 : 4;
      uint64_t messages = tileCount * (2 * side - 2);
      EXPECT_EQ(movement.statistics.peerSends, messages);
      EXPECT_EQ(movement.statistics.peerReceives, messages);
      EXPECT_EQ(movement.statistics.dimensionOrderedAllToAllPackCopies,
                static_cast<uint64_t>(2 * tileCount * (tileCount - 1)));
      EXPECT_EQ(countOps<CommPeerSendOp>(module->getOperation()), messages);
      EXPECT_EQ(countOps<CommPeerRecvOp>(module->getOperation()), messages);
      EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));

      for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
        unsigned sends = 0;
        unsigned receives = 0;
        tile.walk([&](CommPeerSendOp) { ++sends; });
        tile.walk([&](CommPeerRecvOp) { ++receives; });
        EXPECT_EQ(sends, 2 * side - 2);
        EXPECT_EQ(receives, 2 * side - 2);
      }

      std::string standaloneFailure;
      auto standalone = createStandaloneTileModules(
          std::move(module), &standaloneFailure, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << standaloneFailure;
      llvm::SmallVector<mlir::ModuleOp, 16> instructionModules;
      for (StandaloneTileModule &tile : *standalone) {
        llvm::SmallVector<TileRegionOp, 2> tileRegions;
        tile.module->walk(
            [&](TileRegionOp region) { tileRegions.push_back(region); });
        TileRegionToInstrLoweringSession session(*tile.module->getContext());
        for (TileRegionOp region : tileRegions)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
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
      EXPECT_TRUE(mlir::succeeded(
          verifyDirectDTETransportSchedule(instructionModules)));
      EXPECT_TRUE(mlir::succeeded(bindDirectDTETransport(instructionModules)));
      if (tileCount == 16) {
        llvm::SmallVector<TileId, 16> tileIds;
        for (int64_t tile = 0; tile < tileCount; ++tile)
          tileIds.push_back(TileId(tile));
        auto executionConfig =
            wafer::compiler::ExecutionConfig::createForSingleCard(1);
        ASSERT_TRUE(static_cast<bool>(executionConfig))
            << llvm::toString(executionConfig.takeError());
        auto cost = wafer::compiler::testing::verifyProgramResources(
            instructionModules, tileIds, *executionConfig);
        ASSERT_TRUE(mlir::succeeded(cost));
        ASSERT_TRUE(cost->maximumTileNoCTransmitMessageCount.isKnown());
        ASSERT_TRUE(cost->maximumTileNoCReceiveMessageCount.isKnown());
        EXPECT_EQ(cost->maximumTileNoCTransmitMessageCount.value, 2 * side - 2);
        EXPECT_EQ(cost->maximumTileNoCReceiveMessageCount.value, 2 * side - 2);
      }
    }
  }
}

TEST_F(StructuredToTileTest,
       CompleteContributionMatrixUsesActualRingReduceScatter) {
  for (int64_t tileCount : {4, 16}) {
    for (auto [extent, bf16] : {std::pair<int64_t, bool>{1024, false},
                                std::pair<int64_t, bool>{1025, false},
                                std::pair<int64_t, bool>{1031, false},
                                std::pair<int64_t, bool>{1025, true}}) {
      SCOPED_TRACE((llvm::Twine("tiles=") + llvm::Twine(tileCount) +
                    ", extent=" + llvm::Twine(extent) +
                    ", dtype=" + (bf16 ? "bf16" : "f16"))
                       .str());
      llvm::StringRef reductionOp = extent == 1024   ? "arith.addf"
                                    : extent == 1025 ? "arith.maximumf"
                                                     : "arith.minimumf";
      auto module = parse(makeDenseAllToAllSource(
          extent, tileCount, bf16, /*reduceScatter=*/true, reductionOp));
      ASSERT_TRUE(module);
      llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 16> regions;
      for (TileModuleOp tile : module->getOps<TileModuleOp>())
        tile.walk([&](TileRegionOp region) {
          regions.push_back({tile.getTileIdAttr().getInt(), region});
        });
      llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
      });
      ASSERT_EQ(regions.size(), static_cast<size_t>(tileCount));

      StructuredMaterializationRelations relations;
      for (unsigned source = 0; source < regions.size(); ++source)
        for (unsigned destination = 0; destination < regions.size();
             ++destination) {
          if (source == destination)
            continue;
          unsigned sourceResult =
              destination < source ? destination : destination - 1;
          unsigned destinationArgument =
              1 + (source < destination ? source : source - 1);
          relations.boundaryRelations.push_back(
              {regions[source].second.getResult(sourceResult),
               regions[destination].second.getBody().getArgument(
                   destinationArgument)});
        }
      for (auto &[tile, region] : regions) {
        (void)tile;
        relations.structuralOutputs.push_back(
            {0, region.getResult(tileCount - 1)});
      }
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      StructuredToTileResult lowered =
          lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      DistributedMovementAvailability availability =
          analyzeDistributedMovementAvailability(*module, relations);
      EXPECT_FALSE(availability.brokenContract) << availability.detail;
      EXPECT_TRUE(availability.distributedReduction);
      BoundaryMovementResult movement = materializeTileBoundaryMovement(
          *module, relations,
          BoundaryMovementOptions{CompleteAllGatherAlgorithm::Ring,
                                  CompleteAllToAllAlgorithm::Direct,
                                  DistributedReductionAlgorithm::Ring});
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.ringReduceScatterComponents, 1u);
      const uint64_t peerOperations = tileCount * (tileCount - 1);
      EXPECT_EQ(movement.statistics.peerSends, peerOperations);
      EXPECT_EQ(movement.statistics.peerReceives, peerOperations);
      EXPECT_EQ(movement.statistics.distributedReductionCombines,
                peerOperations);
      EXPECT_EQ(countOps<ComputeElementwiseOp>(module->getOperation()),
                peerOperations);
      EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));

      for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
        llvm::SmallVector<unsigned, 16> sends(tileCount - 1);
        llvm::SmallVector<unsigned, 16> receives(tileCount - 1);
        tile.walk([&](CommPeerSendOp send) {
          ++sends[send.getMessageAttr().getRound()];
        });
        tile.walk([&](CommPeerRecvOp receive) {
          ++receives[receive.getMessageAttr().getRound()];
        });
        EXPECT_TRUE(
            llvm::all_of(sends, [](unsigned count) { return count == 1; }));
        EXPECT_TRUE(
            llvm::all_of(receives, [](unsigned count) { return count == 1; }));
      }

      std::string standaloneFailure;
      auto standalone = createStandaloneTileModules(
          std::move(module), &standaloneFailure, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << standaloneFailure;
      llvm::SmallVector<mlir::ModuleOp, 16> instructionModules;
      for (StandaloneTileModule &tile : *standalone) {
        llvm::SmallVector<TileRegionOp, 2> tileRegions;
        tile.module->walk(
            [&](TileRegionOp region) { tileRegions.push_back(region); });
        TileRegionToInstrLoweringSession session(*tile.module->getContext());
        for (TileRegionOp region : tileRegions)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
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
      EXPECT_TRUE(mlir::succeeded(
          verifyDirectDTETransportSchedule(instructionModules)));
      EXPECT_TRUE(mlir::succeeded(bindDirectDTETransport(instructionModules)));
      if (tileCount == 16) {
        llvm::SmallVector<TileId, 16> tileIds;
        for (int64_t tile = 0; tile < tileCount; ++tile)
          tileIds.push_back(TileId(tile));
        auto executionConfig =
            wafer::compiler::ExecutionConfig::createForSingleCard(1);
        ASSERT_TRUE(static_cast<bool>(executionConfig))
            << llvm::toString(executionConfig.takeError());
        auto cost = wafer::compiler::testing::verifyProgramResources(
            instructionModules, tileIds, *executionConfig);
        ASSERT_TRUE(mlir::succeeded(cost));
        ASSERT_TRUE(cost->maximumTileNoCTransmitMessageCount.isKnown());
        ASSERT_TRUE(cost->maximumTileNoCReceiveMessageCount.isKnown());
        EXPECT_EQ(cost->maximumTileNoCTransmitMessageCount.value,
                  tileCount - 1);
        EXPECT_EQ(cost->maximumTileNoCReceiveMessageCount.value, tileCount - 1);
      }
    }
  }
}

TEST_F(StructuredToTileTest, AllReduceRejectsClobberedPublication) {
  for (unsigned clobberPosition : {0u, 1u, 2u}) {
    SCOPED_TRACE(clobberPosition);
    auto module = parse(makeDenseAllReduceSource(1025, 4, "arith.addf", false));
    ASSERT_TRUE(module);
    llvm::SmallVector<TileRegionOp> regions;
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) { regions.push_back(region); });
    ASSERT_EQ(regions.size(), 4u);
    StructuredMaterializationRelations relations;
    for (unsigned source = 1; source < regions.size(); ++source) {
      relations.boundaryRelations.push_back(
          {regions[source].getResult(0),
           regions.front().getBody().getArgument(source)});
      relations.boundaryRelations.push_back(
          {regions.front().getResult(1),
           regions[source].getBody().getArgument(1)});
    }
    for (TileRegionOp region : regions)
      relations.structuralOutputs.push_back({0, region.getResult(1)});
    auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    auto lowered = lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
    auto movement = materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_EQ(movement.statistics.ringAllReduceComponents, 0u);

    MoveCopyIntoOp publication;
    module->walk([&](MoveCopyIntoOp copy) {
      if (copy.getSource().getDefiningOp<ComputeElementwiseOp>())
        publication = copy;
    });
    ASSERT_TRUE(publication);
    mlir::OpBuilder builder(publication);
    if (clobberPosition != 2)
      builder.setInsertionPointAfter(publication);
    auto clobbered =
        clobberPosition == 0 ? publication.getDest() : publication.getSource();
    auto replacement = builder.create<mlir::memref::AllocOp>(
        publication.getLoc(),
        mlir::cast<mlir::MemRefType>(clobbered.getType()));
    // A typed write between publication and fanout destroys the exact value
    // proof, even when all shapes, participants and message pairs still match.
    builder.create<MoveCopyIntoOp>(publication.getLoc(), replacement,
                                   clobbered);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    const auto sends = countOps<CommPeerSendOp>(module->getOperation());
    const auto combines =
        countOps<ComputeElementwiseOp>(module->getOperation());
    auto result = materializeRingAllReduce(*module);
    ASSERT_TRUE(result.succeeded()) << result.detail;
    EXPECT_EQ(result.allReduceComponents, 0u);
    EXPECT_EQ(countOps<CommPeerSendOp>(module->getOperation()), sends);
    EXPECT_EQ(countOps<ComputeElementwiseOp>(module->getOperation()), combines);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(StructuredToTileTest,
       FullContributionMergeAndFanoutUseActualRingAllReduce) {
  for (int64_t tileCount : {4, 16}) {
    for (auto [extent, bf16] : {std::pair<int64_t, bool>{1024, false},
                                std::pair<int64_t, bool>{1025, false},
                                std::pair<int64_t, bool>{1031, false},
                                std::pair<int64_t, bool>{1025, true}}) {
      SCOPED_TRACE((llvm::Twine("tiles=") + llvm::Twine(tileCount) +
                    ", extent=" + llvm::Twine(extent) +
                    ", dtype=" + (bf16 ? "bf16" : "f16"))
                       .str());
      llvm::StringRef reductionOp = extent == 1024   ? "arith.addf"
                                    : extent == 1025 ? "arith.maximumf"
                                                     : "arith.minimumf";
      auto module =
          parse(makeDenseAllReduceSource(extent, tileCount, reductionOp, bf16));
      ASSERT_TRUE(module);
      llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 16> regions;
      for (TileModuleOp tile : module->getOps<TileModuleOp>())
        tile.walk([&](TileRegionOp region) {
          regions.push_back({tile.getTileIdAttr().getInt(), region});
        });
      llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
      });
      ASSERT_EQ(regions.size(), static_cast<size_t>(tileCount));

      StructuredMaterializationRelations relations;
      for (unsigned source = 1; source < regions.size(); ++source) {
        relations.boundaryRelations.push_back(
            {regions[source].second.getResult(0),
             regions.front().second.getBody().getArgument(source)});
        relations.boundaryRelations.push_back(
            {regions.front().second.getResult(1),
             regions[source].second.getBody().getArgument(1)});
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
      DistributedMovementAvailability availability =
          analyzeDistributedMovementAvailability(*module, relations);
      EXPECT_FALSE(availability.brokenContract) << availability.detail;
      EXPECT_TRUE(availability.distributedReduction);
      BoundaryMovementResult movement = materializeTileBoundaryMovement(
          *module, relations,
          BoundaryMovementOptions{CompleteAllGatherAlgorithm::Ring,
                                  CompleteAllToAllAlgorithm::Direct,
                                  DistributedReductionAlgorithm::Ring});
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.ringAllReduceComponents, 1u);
      EXPECT_EQ(movement.statistics.ringReduceScatterComponents, 0u);
      const uint64_t perPhase = tileCount * (tileCount - 1);
      EXPECT_EQ(movement.statistics.peerSends, 2 * perPhase);
      EXPECT_EQ(movement.statistics.peerReceives, 2 * perPhase);
      EXPECT_EQ(movement.statistics.distributedReductionCombines, perPhase);
      EXPECT_EQ(movement.statistics.distributedReductionResultCopies,
                static_cast<uint64_t>(tileCount));
      EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));

      for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
        llvm::SmallVector<unsigned, 32> sends(2 * (tileCount - 1));
        llvm::SmallVector<unsigned, 32> receives(2 * (tileCount - 1));
        tile.walk([&](CommPeerSendOp send) {
          ++sends[send.getMessageAttr().getRound()];
        });
        tile.walk([&](CommPeerRecvOp receive) {
          ++receives[receive.getMessageAttr().getRound()];
        });
        EXPECT_TRUE(
            llvm::all_of(sends, [](unsigned count) { return count == 1; }));
        EXPECT_TRUE(
            llvm::all_of(receives, [](unsigned count) { return count == 1; }));
      }

      std::string standaloneFailure;
      auto standalone = createStandaloneTileModules(
          std::move(module), &standaloneFailure, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << standaloneFailure;
      llvm::SmallVector<mlir::ModuleOp, 16> instructionModules;
      for (StandaloneTileModule &tile : *standalone) {
        llvm::SmallVector<TileRegionOp, 2> tileRegions;
        tile.module->walk(
            [&](TileRegionOp region) { tileRegions.push_back(region); });
        TileRegionToInstrLoweringSession session(*tile.module->getContext());
        for (TileRegionOp region : tileRegions)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
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
      EXPECT_TRUE(mlir::succeeded(
          verifyDirectDTETransportSchedule(instructionModules)));
      EXPECT_TRUE(mlir::succeeded(bindDirectDTETransport(instructionModules)));
      if (tileCount == 16) {
        llvm::SmallVector<TileId, 16> tileIds;
        for (int64_t tile = 0; tile < tileCount; ++tile)
          tileIds.push_back(TileId(tile));
        auto executionConfig =
            wafer::compiler::ExecutionConfig::createForSingleCard(1);
        ASSERT_TRUE(static_cast<bool>(executionConfig))
            << llvm::toString(executionConfig.takeError());
        auto cost = wafer::compiler::testing::verifyProgramResources(
            instructionModules, tileIds, *executionConfig);
        ASSERT_TRUE(mlir::succeeded(cost));
        ASSERT_TRUE(cost->maximumTileNoCTransmitMessageCount.isKnown());
        ASSERT_TRUE(cost->maximumTileNoCReceiveMessageCount.isKnown());
        EXPECT_EQ(cost->maximumTileNoCTransmitMessageCount.value,
                  2 * (tileCount - 1));
        EXPECT_EQ(cost->maximumTileNoCReceiveMessageCount.value,
                  2 * (tileCount - 1));
      }
    }
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
       ConstantUnitInputAxesKeepTheirStorageAndProjectedMaps) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned form : {0u, 1u, 2u, 3u})
      for (bool strided : {false, true}) {
        SCOPED_TRACE(std::to_string(extent) + ":" + std::to_string(form) + ":" +
                     std::to_string(strided));
        llvm::SmallVector<int64_t, 5> shape;
        llvm::SmallVector<mlir::AffineExpr, 5> expressions;
        auto d0 = mlir::getAffineDimExpr(0, context.get());
        auto d1 = mlir::getAffineDimExpr(1, context.get());
        auto d2 = mlir::getAffineDimExpr(2, context.get());
        auto zero = mlir::getAffineConstantExpr(0, context.get());
        if (form == 0) {
          shape = {1, extent, 2, 8};
          expressions = {zero, d1, d0, d2};
        } else if (form == 1) {
          shape = {extent, 1, 2, 8};
          expressions = {d1, zero, d0, d2};
        } else if (form == 2) {
          shape = {1, extent, 1, 2, 8};
          expressions = {zero, d1, zero, d0, d2};
        } else {
          shape = {1, extent, 8};
          expressions = {zero, d1, d2};
        }
        auto fullShape = shape;
        if (strided)
          fullShape.back() *= 2;
        auto fullType = mlir::RankedTensorType::get(
            fullShape, mlir::Float16Type::get(context.get()));
        auto inputType = mlir::RankedTensorType::get(
            shape, mlir::Float16Type::get(context.get()));
        auto outputType = mlir::RankedTensorType::get(
            {2, extent, 8}, mlir::Float16Type::get(context.get()));
        auto map = mlir::AffineMap::get(3, 0, expressions, context.get());
        std::string text;
        llvm::raw_string_ostream out(text);
        out << "module { wafer.tile.module card_id = 0 tile_id = 0 {\n"
            << "func.func @entry(%input: " << fullType << ") {\n"
            << "%result = wafer.tile.region(%input : " << fullType << ") -> ("
            << outputType << ") {\n^bb0(%local: " << fullType << "):\n";
        if (strided) {
          out << "%slice = tensor.extract_slice %local[";
          llvm::SmallVector<int64_t> offsets(shape.size(), 0);
          offsets.back() = 3;
          llvm::interleaveComma(offsets, out);
          out << "] [";
          llvm::interleaveComma(shape, out);
          out << "] [";
          llvm::interleaveComma(llvm::SmallVector<int64_t>(shape.size(), 1),
                                out);
          out << "] : " << fullType << " to " << inputType << "\n";
        }
        out << "%empty = tensor.empty() : " << outputType << "\n"
            << "%mapped = linalg.generic {indexing_maps = [affine_map<" << map
            << ">, affine_map<(d0,d1,d2)->(d0,d1,d2)>], "
               "iterator_types = [\"parallel\",\"parallel\",\"parallel\"]} "
            << "ins(" << (strided ? "%slice" : "%local") << " : " << inputType
            << ") outs(%empty : " << outputType << ") {\n"
            << "^bb1(%x: f16, %old: f16):\n";
        if (form == 3)
          out << "%neg = arith.negf %x : f16\nlinalg.yield %neg : f16\n";
        else
          out << "linalg.yield %x : f16\n";
        out << "} -> " << outputType
            << "\nwafer.tile.yield %mapped : " << outputType
            << "\n}\nreturn\n}}}\n";
        auto module = parse(text);
        ASSERT_TRUE(module);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        TileRegionOp region;
        module->walk([&](TileRegionOp current) { region = current; });
        StructuredMaterializationRelations relations;
        relations.structuralOutputs.push_back({0, region.getResult(0)});
        auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        mlir::Value input;
        module->walk([&](mlir::linalg::GenericOp generic) {
          input = generic.getDpsInputOperand(0)->get();
        });
        ASSERT_TRUE(input);
        auto sourceType = mlir::cast<mlir::MemRefType>(input.getType());
        llvm::SmallVector<int64_t> sourceStrides;
        int64_t sourceOffset = 0;
        ASSERT_TRUE(mlir::succeeded(mlir::getStridesAndOffset(
            sourceType, sourceStrides, sourceOffset)));
        auto lowered = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
        unsigned reducedViews = 0;
        module->walk([&](mlir::memref::SubViewOp view) {
          if (view.getSource() != input ||
              view.getType().getRank() == sourceType.getRank())
            return;
          ++reducedViews;
          llvm::SmallVector<int64_t> expectedStrides;
          for (auto [axis, expression] : llvm::enumerate(expressions))
            if (!mlir::isa<mlir::AffineConstantExpr>(expression))
              expectedStrides.push_back(sourceStrides[axis]);
          llvm::SmallVector<int64_t> strides;
          int64_t offset = 0;
          ASSERT_TRUE(mlir::succeeded(
              mlir::getStridesAndOffset(view.getType(), strides, offset)));
          EXPECT_EQ(strides, expectedStrides);
          EXPECT_EQ(offset, sourceOffset);
          EXPECT_EQ(view.getType().getMemorySpace(),
                    sourceType.getMemorySpace());
        });
        EXPECT_EQ(reducedViews, 1u);
        module->walk([&](ComputeElementwiseOp op) {
          for (mlir::Attribute attr : op.getIndexingMapsAttr())
            EXPECT_TRUE(mlir::cast<mlir::AffineMapAttr>(attr)
                            .getValue()
                            .isProjectedPermutation());
        });
        EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(module->getOperation()), 0u);
        auto movement = materializeTileBoundaryMovement(*module, relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        TileRegionToInstrLoweringSession session(*module->getContext());
        llvm::SmallVector<TileRegionOp> regions;
        module->walk([&](TileRegionOp current) { regions.push_back(current); });
        for (auto current : regions)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(current, session)));
        EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
        EXPECT_EQ(countOps<ComputeElementwiseOp>(module->getOperation()), 0u);
      }
}

TEST_F(StructuredToTileTest,
       NonUnitConstantInputCoordinatesRemainTypedUnsupported) {
  for (int64_t coordinate : {0, 1}) {
    std::string text;
    llvm::raw_string_ostream out(text);
    out << R"mlir(module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<2x1025x2x8xf16>) {
      %result = wafer.tile.region(%input : tensor<2x1025x2x8xf16>)
          -> (tensor<2x1025x8xf16>) {
      ^bb0(%local: tensor<2x1025x2x8xf16>):
        %empty = tensor.empty() : tensor<2x1025x8xf16>
        %mapped = linalg.generic {
          indexing_maps = [affine_map<(d0,d1,d2)->()mlir"
        << coordinate << R"mlir(,d1,d0,d2)>,
                           affine_map<(d0,d1,d2)->(d0,d1,d2)>],
          iterator_types = ["parallel","parallel","parallel"]}
          ins(%local : tensor<2x1025x2x8xf16>)
          outs(%empty : tensor<2x1025x8xf16>) {
        ^bb1(%x: f16, %old: f16):
          linalg.yield %x : f16
        } -> tensor<2x1025x8xf16>
        wafer.tile.yield %mapped : tensor<2x1025x8xf16>
      }
      return
    }
  }
})mlir";
    auto module = parse(text);
    ASSERT_TRUE(module);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    TileRegionOp region;
    module->walk([&](TileRegionOp current) { region = current; });
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    std::string before;
    llvm::raw_string_ostream beforeStream(before);
    module->print(beforeStream);
    auto lowered = lowerStructuredComputeToTile(*module, relations);
    EXPECT_EQ(lowered.failure, StructuredToTileFailureKind::Unsupported);
    std::string after;
    llvm::raw_string_ostream afterStream(after);
    module->print(afterStream);
    EXPECT_EQ(before, after);
  }
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

TEST_F(StructuredToTileTest,
       OneWayReceiveRoundsFollowUsesWithoutReorderingMessages) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (bool reverseUses : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(reverseUses);
      auto module = parse(makeInterferingFanoutsSource(extent));
      ASSERT_TRUE(module);
      llvm::SmallVector<TileRegionOp, 3> regions;
      for (TileModuleOp tile : module->getOps<TileModuleOp>())
        tile.walk([&](TileRegionOp region) { regions.push_back(region); });
      ASSERT_EQ(regions.size(), 3u);
      StructuredMaterializationRelations relations;
      mlir::IRRewriter rewriter(context.get());
      for (unsigned destination = 1; destination < regions.size();
           ++destination) {
        TileRegionOp region = regions[destination];
        for (unsigned input = 0; input < 2; ++input)
          relations.boundaryRelations.push_back(
              {regions[0].getResult(input),
               region.getBody().getArgument(input)});
        relations.structuralOutputs.push_back({0, region.getResult(0)});
        auto combine =
            *region.getBody().front().getOps<mlir::linalg::GenericOp>().begin();
        rewriter.setInsertionPoint(combine);
        mlir::IRMapping mapping;
        auto independent = mlir::cast<mlir::linalg::GenericOp>(
            rewriter.clone(*combine, mapping));
        unsigned first = reverseUses ? 1 : 0;
        mlir::Value input = combine.getDpsInputs()[first];
        independent.setOperand(0, input);
        independent.setOperand(1, input);
        combine.setOperand(first, independent.getResult(0));
      }
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.sparseRoundComponents, 1u);
      EXPECT_EQ(movement.statistics.peerReceives, 4u);
      EXPECT_EQ(movement.statistics.crossTileDDRStages, 0u);
      for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
        if (tile.getTileIdAttr().getInt() == 0)
          continue;
        llvm::SmallVector<CommPeerRecvOp, 2> receives;
        llvm::SmallVector<ComputeElementwiseOp, 2> computations;
        tile.walk([&](CommPeerRecvOp op) { receives.push_back(op); });
        tile.walk([&](ComputeElementwiseOp op) { computations.push_back(op); });
        ASSERT_EQ(receives.size(), 2u);
        ASSERT_EQ(computations.size(), 2u);
        EXPECT_LT(receives[0].getMessageAttr().getRound(),
                  receives[1].getMessageAttr().getRound());
        EXPECT_TRUE(receives[0]->isBeforeInBlock(computations[0]));
        EXPECT_TRUE(receives[1]->isBeforeInBlock(computations[1]));
        EXPECT_EQ(receives[1]->isBeforeInBlock(computations[0]), reverseUses);
      }
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      llvm::SmallVector<mlir::ModuleOp, 3> instructionModules;
      for (StandaloneTileModule &tile : *standalone) {
        llvm::SmallVector<TileRegionOp, 2> tileRegions;
        tile.module->walk([&](TileRegionOp op) { tileRegions.push_back(op); });
        TileRegionToInstrLoweringSession session(*context);
        for (TileRegionOp region : tileRegions)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*tile.module, session)));
        instructionModules.push_back(*tile.module);
      }
      auto completion = rebuildRequiredDirectDTEWaits(instructionModules);
      ASSERT_TRUE(completion.succeeded()) << completion.detail;
      instructionModules.clear();
      for (StandaloneTileModule &tile : *standalone) {
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
        TileMemoryPlanningFailure failure;
        auto planned = planTileMemory(std::move(tile.module), &failure);
        ASSERT_TRUE(mlir::succeeded(planned));
        tile.module = std::move(*planned);
        instructionModules.push_back(*tile.module);
      }
      EXPECT_TRUE(mlir::succeeded(
          verifyDirectDTETransportSchedule(instructionModules)));
    }
  }
}

TEST_F(StructuredToTileTest, RingRejectsDuplicateAndOutOfRangeParticipants) {
  auto module = parse(R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 2, 2>, unavailable_tiles = array<i64>}
}
)mlir");
  ASSERT_TRUE(module);
  auto topology = TargetTopology::create(*module);
  ASSERT_TRUE(mlir::succeeded(topology));

  llvm::SmallVector<uint64_t, 4> validParticipants{0, 1, 2, 3};
  EXPECT_TRUE(
      mlir::succeeded(buildMinimumHopTileRing(*topology, validParticipants)));

  llvm::SmallVector<uint64_t, 4> duplicateParticipants{0, 1, 1, 2};
  EXPECT_TRUE(
      mlir::failed(buildMinimumHopTileRing(*topology, duplicateParticipants)));

  llvm::SmallVector<uint64_t, 4> outOfRangeParticipants{
      0, 1, 2, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1};
  EXPECT_TRUE(
      mlir::failed(buildMinimumHopTileRing(*topology, outOfRangeParticipants)));
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

TEST_F(StructuredToTileTest,
       CompleteExchangeRecursiveDoublingUsesActualAggregateBuffers) {
  for (int64_t tileCount : {4, 16}) {
    for (int64_t extent : {1024, 1025, 1031}) {
      for (bool bf16 : {false, true}) {
        if (bf16 && extent != 1025)
          continue;
        SCOPED_TRACE((llvm::Twine("tiles=") + llvm::Twine(tileCount) +
                      ", extent=" + llvm::Twine(extent) +
                      ", dtype=" + (bf16 ? "bf16" : "f16"))
                         .str());
        int64_t channels = !bf16 && tileCount == 4 ? 64 : 1;
        auto module = parse(
            !bf16 && tileCount == 4
                ? makeCompleteExchangeSource(extent)
                : makeDenseCompleteExchangeSource(extent, tileCount, bf16));
        ASSERT_TRUE(module);
        llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 4> regions;
        for (TileModuleOp tile : module->getOps<TileModuleOp>())
          tile.walk([&](TileRegionOp region) {
            regions.push_back({tile.getTileIdAttr().getInt(), region});
          });
        llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
          return lhs.first < rhs.first;
        });
        ASSERT_EQ(regions.size(), static_cast<size_t>(tileCount));
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
        RecursiveDoublingAvailability availability =
            analyzeRecursiveDoublingAvailability(*module, relations);
        ASSERT_TRUE(availability.isAvailable()) << availability.detail;
        BoundaryMovementResult movement = materializeTileBoundaryMovement(
            *module, relations,
            BoundaryMovementOptions{
                CompleteAllGatherAlgorithm::RecursiveDoubling,
                CompleteAllToAllAlgorithm::Direct});
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        EXPECT_EQ(movement.statistics.recursiveDoublingComponents, 1u);
        unsigned rounds = tileCount == 4 ? 2 : 4;
        EXPECT_EQ(movement.statistics.recursiveDoublingRounds, rounds);
        EXPECT_EQ(movement.statistics.recursiveDoublingSeedCopies, 0u);
        EXPECT_EQ(movement.statistics.recursiveDoublingSeedDonations,
                  static_cast<uint64_t>(tileCount));
        EXPECT_EQ(movement.statistics.ringComponents, 0u);
        uint64_t peerOperations = tileCount * (tileCount - 1);
        EXPECT_EQ(movement.statistics.peerSends, peerOperations);
        EXPECT_EQ(movement.statistics.peerReceives, peerOperations);
        EXPECT_EQ(countOps<mlir::memref::CopyOp>(module->getOperation()), 0u);

        for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
          unsigned aggregateAllocations = 0;
          tile.walk([&](mlir::memref::AllocOp allocation) {
            auto type = mlir::cast<mlir::MemRefType>(allocation.getType());
            if (type.getRank() == 3 && type.getDimSize(0) == tileCount &&
                type.getDimSize(1) == extent && type.getDimSize(2) == channels)
              ++aggregateAllocations;
          });
          EXPECT_EQ(aggregateAllocations, 1u);
        }

        std::string standaloneFailure;
        auto standalone = createStandaloneTileModules(
            std::move(module), &standaloneFailure, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << standaloneFailure;
        llvm::SmallVector<mlir::ModuleOp, 16> instructionModules;
        for (StandaloneTileModule &tile : *standalone) {
          llvm::SmallVector<TileRegionOp, 2> tileRegions;
          tile.module->walk(
              [&](TileRegionOp region) { tileRegions.push_back(region); });
          TileRegionToInstrLoweringSession session(*tile.module->getContext());
          for (TileRegionOp region : tileRegions)
            ASSERT_TRUE(
                mlir::succeeded(convertTileRegionToInstr(region, session)));
          ASSERT_TRUE(mlir::succeeded(
              convertBufferizationCopiesToInstr(*tile.module, session)));
          instructionModules.push_back(*tile.module);
        }
        auto coalesced = coalesceExactDirectDTETransfers(instructionModules);
        ASSERT_TRUE(coalesced.succeeded()) << coalesced.detail;
        EXPECT_EQ(coalesced.statistics.coalescedP2PTransfers,
                  static_cast<uint64_t>(tileCount * (rounds - 1)));
        unsigned sends = 0;
        unsigned receives = 0;
        for (mlir::ModuleOp tile : instructionModules) {
          tile.walk([&](InstrDTESendOp) { ++sends; });
          tile.walk([&](InstrDTERecvOp) { ++receives; });
        }
        EXPECT_EQ(sends, static_cast<unsigned>(tileCount * rounds));
        EXPECT_EQ(receives, static_cast<unsigned>(tileCount * rounds));

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
        EXPECT_TRUE(mlir::succeeded(
            verifyDirectDTETransportSchedule(instructionModules)));
        EXPECT_TRUE(
            mlir::succeeded(bindDirectDTETransport(instructionModules)));
        if (tileCount == 16) {
          llvm::SmallVector<TileId, 16> tileIds;
          for (int64_t tile = 0; tile < tileCount; ++tile)
            tileIds.push_back(TileId(tile));
          auto executionConfig =
              wafer::compiler::ExecutionConfig::createForSingleCard(1);
          ASSERT_TRUE(static_cast<bool>(executionConfig))
              << llvm::toString(executionConfig.takeError());
          auto cost = wafer::compiler::testing::verifyProgramResources(
              instructionModules, tileIds, *executionConfig);
          ASSERT_TRUE(mlir::succeeded(cost));
          ASSERT_TRUE(cost->maximumTileNoCTransmitMessageCount.isKnown());
          ASSERT_TRUE(cost->maximumTileNoCReceiveMessageCount.isKnown());
          EXPECT_EQ(cost->maximumTileNoCTransmitMessageCount.value, rounds);
          EXPECT_EQ(cost->maximumTileNoCReceiveMessageCount.value, rounds);
        }
      }
    }
  }
}

TEST_F(StructuredToTileTest,
       NonPowerOfTwoCompleteExchangeKeepsRingRealization) {
  constexpr int64_t tileCount = 3;
  constexpr int64_t extent = 1025;
  auto module = parse(makeDenseCompleteExchangeSource(extent, tileCount));
  ASSERT_TRUE(module);
  llvm::SmallVector<std::pair<int64_t, TileRegionOp>, 3> regions;
  for (TileModuleOp tile : module->getOps<TileModuleOp>())
    tile.walk([&](TileRegionOp region) {
      regions.push_back({tile.getTileIdAttr().getInt(), region});
    });
  llvm::sort(regions, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  ASSERT_EQ(regions.size(), static_cast<size_t>(tileCount));
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
  RecursiveDoublingAvailability availability =
      analyzeRecursiveDoublingAvailability(*module, relations);
  EXPECT_EQ(availability.kind, RecursiveDoublingAvailabilityKind::Unavailable);
  BoundaryMovementResult movement = materializeTileBoundaryMovement(
      *module, relations,
      BoundaryMovementOptions{CompleteAllGatherAlgorithm::RecursiveDoubling,
                              CompleteAllToAllAlgorithm::Direct});
  ASSERT_TRUE(movement.succeeded()) << movement.detail;
  EXPECT_EQ(movement.statistics.recursiveDoublingComponents, 0u);
  EXPECT_EQ(movement.statistics.recursiveDoublingSeedCopies, 0u);
  EXPECT_EQ(movement.statistics.ringComponents, 1u);
  EXPECT_EQ(movement.statistics.ringRounds, 2u);
  EXPECT_EQ(movement.statistics.peerSends, 6u);
  EXPECT_EQ(movement.statistics.peerReceives, 6u);
}

TEST_F(StructuredToTileTest, SplitExchangeRegionsCloseBeforeOneRingRound) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeSplitExchangeSource(extent));
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

TEST_F(StructuredToTileTest, RegionClosureCoalescesIdenticalImportedArguments) {
  for (int64_t tileCount : {4, 16}) {
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(tileCount);
      SCOPED_TRACE(extent);
      auto module = parse(makeSequentialExchangeSource(extent, tileCount));
      ASSERT_TRUE(module);
      llvm::SmallVector<llvm::SmallVector<TileRegionOp, 3>, 16> regions(
          tileCount);
      for (TileModuleOp tile : module->getOps<TileModuleOp>())
        tile.walk([&](TileRegionOp region) {
          regions[tile.getTileId()].push_back(region);
        });
      StructuredMaterializationRelations relations;
      for (unsigned source = 0; source < regions.size(); ++source)
        for (unsigned destination = 0; destination < regions.size();
             ++destination) {
          if (source == destination)
            continue;
          unsigned argument = 1 + source - (source > destination);
          // Both consumers import the same external SSA tensor. Their
          // distinct block arguments become one argument after merging.
          regions[destination][2]->setOperand(
              argument, regions[destination][1].getInputs()[argument]);
          for (unsigned phase : {1u, 2u})
            relations.boundaryRelations.push_back(
                {regions[source][0].getResult(0),
                 regions[destination][phase].getBody().getArgument(argument)});
        }
      for (auto &tileRegions : regions)
        relations.structuralOutputs.push_back(
            {0, tileRegions.back().getResult(0)});
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      ASSERT_TRUE(mlir::succeeded(
          checkStructuredBufferRelationsCurrent(*module, relations)));
      SpatialRegionMaterializationFailure failure;
      ASSERT_TRUE(mlir::succeeded(closeCrossTileCommunicationRegions(
          *module, relations, nullptr, &failure)))
          << failure.detail;
      EXPECT_EQ(relations.boundaryRelations.size(),
                tileCount * (tileCount - 1));
      EXPECT_EQ(countOps<TileRegionOp>(*module), tileCount);
      ASSERT_TRUE(mlir::succeeded(
          checkStructuredBufferRelationsCurrent(*module, relations)));
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto compute = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(compute.succeeded()) << compute.detail;
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.peerSends, tileCount * (tileCount - 1));
      EXPECT_EQ(movement.statistics.peerReceives, tileCount * (tileCount - 1));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
  }
}

TEST_F(StructuredToTileTest,
       SequentialExchangesKeepSeparateCutsInOneActualRegion) {
  for (int64_t tileCount : {4, 16}) {
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(tileCount);
      SCOPED_TRACE(extent);
      auto module = parse(makeSequentialExchangeSource(extent, tileCount));
      ASSERT_TRUE(module);
      llvm::SmallVector<llvm::SmallVector<TileRegionOp, 3>, 16> regions(
          tileCount);
      for (TileModuleOp tile : module->getOps<TileModuleOp>())
        tile.walk([&](TileRegionOp region) {
          regions[tile.getTileIdAttr().getInt()].push_back(region);
        });
      StructuredMaterializationRelations relations;
      // Interleave the two exchanges so relation iteration order cannot
      // substitute for current producer/consumer dependencies.
      for (unsigned source = 0; source < regions.size(); ++source)
        for (unsigned phase = 0; phase < 2; ++phase)
          for (unsigned destination = 0; destination < regions.size();
               ++destination) {
            if (source == destination)
              continue;
            unsigned argument = 1 + source - (source > destination);
            relations.boundaryRelations.push_back(
                {regions[source][phase].getResult(0),
                 regions[destination][phase + 1].getBody().getArgument(
                     argument)});
          }
      for (auto &tileRegions : regions)
        relations.structuralOutputs.push_back(
            {0, tileRegions.back().getResult(0)});
      auto print = [&]() {
        std::string text;
        llvm::raw_string_ostream stream(text);
        module->print(stream);
        return text;
      };
      const std::string before = print();
      auto available = analyzeCommunicationRegionClosure(*module, relations);
      ASSERT_TRUE(mlir::succeeded(available));
      EXPECT_EQ(*available, CommunicationRegionClosureAvailability::Available);
      EXPECT_EQ(print(), before);
      CommunicationRegionClosureStatistics closure;
      ASSERT_TRUE(mlir::succeeded(
          closeCrossTileCommunicationRegions(*module, relations, &closure)));
      EXPECT_EQ(closure.mergedTileScopes, static_cast<uint64_t>(tileCount));
      EXPECT_EQ(closure.mergedRegions, static_cast<uint64_t>(3 * tileCount));
      EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()), tileCount);
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto compute = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(compute.succeeded()) << compute.detail;
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.ringComponents, 2u);
      EXPECT_EQ(movement.statistics.ringRounds, 2 * (tileCount - 1));
      EXPECT_EQ(movement.statistics.peerSends, 2 * tileCount * (tileCount - 1));
      EXPECT_EQ(movement.statistics.peerReceives,
                2 * tileCount * (tileCount - 1));
      EXPECT_EQ(movement.statistics.crossTileDDRStages, 0u);
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      llvm::SmallVector<mlir::ModuleOp, 16> instructionModules;
      for (StandaloneTileModule &tile : *standalone) {
        llvm::SmallVector<TileRegionOp, 1> tileRegions;
        tile.module->walk(
            [&](TileRegionOp region) { tileRegions.push_back(region); });
        ASSERT_EQ(tileRegions.size(), 1u);
        TileRegionToInstrLoweringSession session(*tile.module->getContext());
        ASSERT_TRUE(mlir::succeeded(
            convertTileRegionToInstr(tileRegions.front(), session)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*tile.module, session)));
        instructionModules.push_back(*tile.module);
      }
      auto completion = rebuildRequiredDirectDTEWaits(instructionModules);
      ASSERT_TRUE(completion.succeeded()) << completion.detail;
      instructionModules.clear();
      for (StandaloneTileModule &tile : *standalone) {
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
        TileMemoryPlanningFailure failure;
        auto planned = planTileMemory(std::move(tile.module), &failure);
        ASSERT_TRUE(mlir::succeeded(planned));
        tile.module = std::move(*planned);
        instructionModules.push_back(*tile.module);
      }
      EXPECT_TRUE(mlir::succeeded(
          verifyDirectDTETransportSchedule(instructionModules)));
      EXPECT_TRUE(mlir::succeeded(bindDirectDTETransport(instructionModules)));
    }
  }
}

TEST_F(StructuredToTileTest,
       CommunicationComponentsChooseTransportIndependently) {
  for (int64_t tileCount : {4, 16}) {
    for (int64_t extent : {1024, 1025, 1031}) {
      for (unsigned mask = 0; mask < 4; ++mask) {
        SCOPED_TRACE(tileCount);
        SCOPED_TRACE(extent);
        SCOPED_TRACE(mask);
        auto module = parse(makeSequentialExchangeSource(extent, tileCount));
        ASSERT_TRUE(module);
        llvm::SmallVector<llvm::SmallVector<TileRegionOp, 3>, 16> regions(
            tileCount);
        for (TileModuleOp tile : module->getOps<TileModuleOp>())
          tile.walk([&](TileRegionOp region) {
            regions[tile.getTileIdAttr().getInt()].push_back(region);
          });
        StructuredMaterializationRelations relations;
        for (unsigned source = 0; source < regions.size(); ++source)
          for (unsigned phase = 0; phase < 2; ++phase)
            for (unsigned dest = 0; dest < regions.size(); ++dest)
              if (source != dest)
                relations.boundaryRelations.push_back(
                    {regions[source][phase].getResult(0),
                     regions[dest][phase + 1].getBody().getArgument(
                         1 + source - (source > dest))});
        for (auto &tile : regions)
          relations.structuralOutputs.push_back({0, tile.back().getResult(0)});
        ASSERT_TRUE(mlir::succeeded(
            closeCrossTileCommunicationRegions(*module, relations)));
        ASSERT_TRUE(
            resolveCurrentLayoutsAndBufferize(*module, relations).succeeded());
        ASSERT_TRUE(
            lowerStructuredComputeToTile(*module, relations).succeeded());
        auto components = queryBoundaryMovementComponents(*module, relations);
        ASSERT_TRUE(components.succeeded()) << components.detail;
        ASSERT_EQ(components.anchors.size(), 2u);
        BoundaryMovementOptions options;
        for (unsigned component = 0; component < 2; ++component)
          options.components.push_back(
              {components.anchors[component],
               mask & (1u << component) ? BoundaryMovementTransport::SharedDDR
                                        : BoundaryMovementTransport::Peer});
        auto movement =
            materializeTileBoundaryMovement(*module, relations, options);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        const unsigned ddr = (mask & 1) + ((mask >> 1) & 1);
        EXPECT_EQ(movement.statistics.crossTileDDRStages,
                  ddr * tileCount * (tileCount - 1));
        EXPECT_EQ(movement.statistics.peerSends,
                  (2 - ddr) * tileCount * (tileCount - 1));
        EXPECT_EQ(movement.statistics.peerReceives,
                  movement.statistics.peerSends);
        std::string detail;
        auto standalone =
            createStandaloneTileModules(std::move(module), &detail, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
        llvm::SmallVector<mlir::ModuleOp, 16> modules;
        llvm::SmallVector<TileId, 16> tileIds;
        for (auto &tile : *standalone) {
          llvm::SmallVector<TileRegionOp, 4> scopes;
          tile.module->walk(
              [&](TileRegionOp region) { scopes.push_back(region); });
          TileRegionToInstrLoweringSession session(*context);
          for (auto scope : scopes)
            ASSERT_TRUE(
                mlir::succeeded(convertTileRegionToInstr(scope, session)));
          ASSERT_TRUE(mlir::succeeded(
              convertBufferizationCopiesToInstr(*tile.module, session)));
          modules.push_back(*tile.module);
          tileIds.push_back(tile.tileId);
        }
        auto waits = rebuildRequiredDirectDTEWaits(modules);
        ASSERT_TRUE(waits.succeeded()) << waits.detail;
        auto shared = materializeSharedDDRCompletion(modules, tileIds);
        ASSERT_TRUE(shared.succeeded()) << shared.detail;
        waits = rebuildRequiredDirectDTEWaits(modules);
        ASSERT_TRUE(waits.succeeded()) << waits.detail;
        for (auto &tile : *standalone) {
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
          TileMemoryPlanningFailure failure;
          auto planned = planTileMemory(std::move(tile.module), &failure);
          ASSERT_TRUE(mlir::succeeded(planned));
        }
      }
    }
  }
}

TEST_F(StructuredToTileTest,
       CrossComponentRegionOrderCycleUsesOnePreflightDDRBoundary) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(makeSplitExchangeSource(extent));
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

TEST_F(StructuredToTileTest, RegionClosureIsOptionalAndDDRHasItsOwnOrderProof) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (bool merge : {false, true}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(merge);
      auto module = parse(makeSplitExchangeSource(extent));
      ASSERT_TRUE(module);
      llvm::SmallVector<llvm::SmallVector<TileRegionOp, 2>, 2> regions(2);
      for (TileModuleOp tile : module->getOps<TileModuleOp>())
        tile.walk([&](TileRegionOp region) {
          regions[tile.getTileIdAttr().getInt()].push_back(region);
        });
      StructuredMaterializationRelations relations;
      for (unsigned tile = 0; tile < 2; ++tile) {
        relations.boundaryRelations.push_back(
            {regions[tile][0].getResult(0),
             regions[1 - tile][1].getBody().getArgument(0)});
        relations.structuralOutputs.push_back(
            {0, regions[tile][1].getResult(0)});
      }
      auto text = [&]() {
        std::string result;
        llvm::raw_string_ostream stream(result);
        module->print(stream);
        return result;
      };
      std::string before = text();
      auto availability = analyzeCommunicationRegionClosure(*module, relations);
      ASSERT_TRUE(mlir::succeeded(availability));
      EXPECT_EQ(*availability,
                CommunicationRegionClosureAvailability::Available);
      EXPECT_EQ(text(), before);
      EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()), 4u);
      if (merge) {
        ASSERT_TRUE(mlir::succeeded(
            closeCrossTileCommunicationRegions(*module, relations)));
      }
      EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()),
                merge ? 2u : 4u);
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto compute = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(compute.succeeded()) << compute.detail;
      EXPECT_TRUE(
          analyzeDistributedMovementAvailability(*module, relations).sharedDDR);
      BoundaryMovementOptions options;
      options.transport = BoundaryMovementTransport::SharedDDR;
      auto movement =
          materializeTileBoundaryMovement(*module, relations, options);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.crossTileDDRStages, 2u);
      EXPECT_EQ(movement.statistics.peerSends, 0u);
      EXPECT_EQ(movement.statistics.peerReceives, 0u);
      EXPECT_EQ(countOps<mlir::memref::GlobalOp>(module->getOperation()), 2u);
      ASSERT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      llvm::SmallVector<mlir::ModuleOp> modules;
      llvm::SmallVector<TileId> tileIds;
      for (StandaloneTileModule &tile : *standalone) {
        llvm::SmallVector<TileRegionOp, 2> scopes;
        tile.module->walk(
            [&](TileRegionOp region) { scopes.push_back(region); });
        TileRegionToInstrLoweringSession session(*tile.module->getContext());
        for (TileRegionOp region : scopes)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*tile.module, session)));
        modules.push_back(*tile.module);
        tileIds.push_back(tile.tileId);
      }
      auto completion = materializeSharedDDRCompletion(modules, tileIds);
      ASSERT_TRUE(completion.succeeded()) << completion.detail;
      for (StandaloneTileModule &tile : *standalone) {
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
        TileMemoryPlanningFailure failure;
        auto planned = planTileMemory(std::move(tile.module), &failure);
        ASSERT_TRUE(mlir::succeeded(planned));
        EXPECT_EQ(countOps<InstrDTESendOp>((*planned)->getOperation()), 0u);
      }
    }
  }
}

TEST_F(StructuredToTileTest, SparseSharedDDRFanoutOmitsUnrelatedTileArguments) {
  for (int64_t tileCount : {4, 16})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(tileCount);
      SCOPED_TRACE(extent);
      auto source = makeFanoutSource(extent, {1, 2}, 0, false);
      source.erase(source.rfind('}'));
      for (int64_t tile = 3; tile < tileCount; ++tile)
        source +=
            "wafer.tile.module card_id = 0 tile_id = " + std::to_string(tile) +
            " { func.func @unused() { return } }\n";
      source += "}";
      auto module = parse(source);
      ASSERT_TRUE(module);
      llvm::SmallVector<TileRegionOp> regions;
      module->walk([&](TileRegionOp region) { regions.push_back(region); });
      ASSERT_EQ(regions.size(), 3u);
      StructuredMaterializationRelations relations;
      for (unsigned index = 1; index < regions.size(); ++index) {
        relations.boundaryRelations.push_back(
            {regions[0].getResult(0), regions[index].getBody().getArgument(0)});
        relations.structuralOutputs.push_back({0, regions[index].getResult(0)});
      }
      ASSERT_TRUE(
          resolveCurrentLayoutsAndBufferize(*module, relations).succeeded());
      ASSERT_TRUE(lowerStructuredComputeToTile(*module, relations).succeeded());
      BoundaryMovementOptions options;
      options.transport = BoundaryMovementTransport::SharedDDR;
      auto movement =
          materializeTileBoundaryMovement(*module, relations, options);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.crossTileDDRStages, 2u);
      EXPECT_EQ(countOps<mlir::memref::GlobalOp>(module->getOperation()), 1u);
      unsigned dataBindings = 0;
      module->walk([&](mlir::func::FuncOp entry) {
        auto tile = entry->getParentOfType<TileModuleOp>();
        unsigned bindings = 0;
        for (unsigned index = 0; index < entry.getNumArguments(); ++index)
          if (auto binding = entry.getArgAttrOfType<DDRBindingAttr>(
                  index, kWaferDDRBindingAttrName)) {
            ++bindings;
            EXPECT_NE(binding.getAccess(), DDRAccess::None);
            EXPECT_FALSE(entry.getArgument(index).use_empty());
          }
        EXPECT_EQ(bindings, tile.getTileIdAttr().getInt() < 3 ? 1u : 0u);
        dataBindings += bindings;
      });
      EXPECT_EQ(dataBindings, 3u);
      std::string detail;
      auto standalone =
          createStandaloneTileModules(std::move(module), &detail, &relations);
      ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
      llvm::SmallVector<mlir::ModuleOp> modules;
      llvm::SmallVector<TileId> tileIds;
      for (auto &tile : *standalone) {
        llvm::SmallVector<TileRegionOp> scopes;
        tile.module->walk(
            [&](TileRegionOp region) { scopes.push_back(region); });
        TileRegionToInstrLoweringSession session(*context);
        for (auto region : scopes)
          ASSERT_TRUE(
              mlir::succeeded(convertTileRegionToInstr(region, session)));
        ASSERT_TRUE(mlir::succeeded(
            convertBufferizationCopiesToInstr(*tile.module, session)));
        modules.push_back(*tile.module);
        tileIds.push_back(tile.tileId);
      }
      auto completion = materializeSharedDDRCompletion(modules, tileIds);
      ASSERT_TRUE(completion.succeeded()) << completion.detail;
      // A notification binding on a nonparticipant must not be silently
      // tolerated after compact materialization.
      auto unused = modules.back();
      auto unusedEntry = *unused.getOps<mlir::func::FuncOp>().begin();
      mlir::memref::GlobalOp readyDeclaration;
      for (auto global : modules.front().getOps<mlir::memref::GlobalOp>())
        if (auto resource = global->getAttrOfType<DDRResourceAttr>(kWaferDDRResourceAttrName))
          if (resource.getResourceId() == 1)
            readyDeclaration = global;
      ASSERT_TRUE(readyDeclaration);
      mlir::OpBuilder builder(unused.getBodyRegion());
      builder.setInsertionPointToStart(unused.getBody());
      mlir::IRMapping mapping;
      auto *extraDeclaration = builder.clone(*readyDeclaration, mapping);
      auto binding = DDRBindingAttr::get(context.get(),
          mlir::FlatSymbolRefAttr::get(readyDeclaration.getSymNameAttr()),
          1, DDRAccess::None);
      unsigned extraArgument = unusedEntry.getNumArguments();
      unusedEntry.insertArgument(extraArgument, readyDeclaration.getType(),
          builder.getDictionaryAttr({builder.getNamedAttr(kWaferDDRBindingAttrName, binding)}),
          unusedEntry.getLoc());
      EXPECT_EQ(verifySharedDDRCompletion(modules, tileIds).failure,
                SharedDDRCompletionFailure::Contract);
      unusedEntry.eraseArgument(extraArgument);
      extraDeclaration->erase();
      EXPECT_TRUE(verifySharedDDRCompletion(modules, tileIds).succeeded());
      unsigned publications = 0, acquisitions = 0, readyBindings = 0;
      for (auto &tile : *standalone) {
        publications += countOps<SyncDDRPublishOp>(tile.module->getOperation());
        acquisitions += countOps<SyncDDRAcquireOp>(tile.module->getOperation());
        tile.module->walk([&](mlir::func::FuncOp entry) {
          for (unsigned index = 0; index < entry.getNumArguments(); ++index)
            if (auto binding = entry.getArgAttrOfType<DDRBindingAttr>(
                    index, kWaferDDRBindingAttrName)) {
              EXPECT_NE(binding.getAccess(), DDRAccess::None);
              readyBindings += binding.getResourceId() == 1;
            }
        });
        ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
        TileMemoryPlanningFailure failure;
        auto planned = planTileMemory(std::move(tile.module), &failure);
        ASSERT_TRUE(mlir::succeeded(planned));
      }
      EXPECT_EQ(publications, 1u);
      EXPECT_EQ(acquisitions, 2u);
      EXPECT_EQ(readyBindings, 3u);
    }
}

TEST_F(StructuredToTileTest, SharedDDRLoadsEachCurrentConsumerSubview) {
  for (int64_t tileCount : {4, 16})
    for (int64_t extent : {1024, 1025, 1031})
      for (unsigned mode : {0u, 1u, 2u}) {
        bool tiled = mode != 0;
        SCOPED_TRACE(tileCount);
        SCOPED_TRACE(extent);
        SCOPED_TRACE(mode);
        auto module = parse(makeSplitExchangeSource(extent, tileCount));
        ASSERT_TRUE(module);
        llvm::SmallVector<llvm::SmallVector<TileRegionOp, 2>, 16> scopes(
            tileCount);
        for (auto tile : module->getOps<TileModuleOp>())
          tile.walk([&](TileRegionOp region) {
            scopes[tile.getTileId()].push_back(region);
          });
        StructuredMaterializationRelations relations;
        for (int64_t tile = 0; tile < tileCount; ++tile) {
          relations.boundaryRelations.push_back(
              {scopes[tile][0].getResult(0),
               scopes[(tile + 1) % tileCount][1].getBody().getArgument(0)});
          relations.structuralOutputs.push_back(
              {0, scopes[tile][1].getResult(0)});
        }
        if (tiled)
          for (auto &tile : scopes) {
            auto domain = buildTemporalDomain(tile[1]);
            ASSERT_TRUE(domain.succeeded());
            auto first = domain.domain->getFirstChoice();
            auto choice = *first.getChoice();
            ASSERT_EQ(choice.scopes.size(), 1u);
            choice.scopes[0].iteratorTileSizes[1] = 128;
            choice.scopes[0].loopOrder = {1};
            ASSERT_TRUE(domain.domain->contains(choice));
            ASSERT_TRUE(mlir::succeeded(
                applyTemporalTiling(*domain.domain, choice, relations)));
          }
        auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        auto compute = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(compute.succeeded()) << compute.detail;
        if (mode == 2) {
          // Model a spatial selection followed by temporal main/tail views.
          // Both intermediate views have no data use and must stay in DDR.
          llvm::SmallVector<TileRegionOp> consumers;
          module->walk([&](TileRegionOp region) {
            if (countOps<ComputeElementwiseOp>(region))
              consumers.push_back(region);
          });
          ASSERT_EQ(consumers.size(), tileCount);
          for (auto consumer : consumers) {
            llvm::SmallVector<mlir::bufferization::ToMemrefOp> bridges;
            consumer.walk([&](mlir::bufferization::ToMemrefOp op) {
              bridges.push_back(op);
            });
            ASSERT_EQ(bridges.size(), extent % 128 ? 2u : 1u);
            for (auto bridge : bridges) {
              llvm::SmallVector<mlir::OpOperand *> uses;
              for (auto &use : bridge.getMemref().getUses())
                uses.push_back(&use);
              mlir::OpBuilder builder(bridge);
              builder.setInsertionPointAfter(bridge);
              auto outer = builder.create<mlir::memref::SubViewOp>(
                  bridge.getLoc(), bridge.getMemref(),
                  llvm::ArrayRef<int64_t>{0, 0, 0},
                  llvm::ArrayRef<int64_t>{1, extent, 64},
                  llvm::ArrayRef<int64_t>{1, 1, 1});
              auto inner = builder.create<mlir::memref::SubViewOp>(
                  bridge.getLoc(), outer.getResult(),
                  llvm::ArrayRef<int64_t>{0, 0, 0},
                  llvm::ArrayRef<int64_t>{1, extent, 64},
                  llvm::ArrayRef<int64_t>{1, 1, 1});
              for (auto *use : uses)
                use->set(inner);
            }
          }
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        }
        BoundaryMovementOptions options;
        options.transport = BoundaryMovementTransport::SharedDDR;
        auto movement =
            materializeTileBoundaryMovement(*module, relations, options);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        EXPECT_EQ(movement.statistics.crossTileDDRStages, tileCount);
        EXPECT_EQ(movement.statistics.peerReceives, 0u);
        module->walk([&](mlir::func::FuncOp entry) {
          for (unsigned argument = 0; argument < entry.getNumArguments();
               ++argument)
            if (auto binding = entry.getArgAttrOfType<DDRBindingAttr>(
                    argument, kWaferDDRBindingAttrName)) {
              EXPECT_NE(binding.getAccess(), DDRAccess::None);
              EXPECT_FALSE(entry.getArgument(argument).use_empty());
            }
        });
        unsigned consumers = 0;
        module->walk([&](TileRegionOp region) {
          if (!countOps<ComputeElementwiseOp>(region))
            return;
          ++consumers;
          std::vector<unsigned> coverage(extent, 0);
          unsigned loads = 0;
          region.walk([&](StorageLoadOp load) {
            ++loads;
            auto type = mlir::cast<mlir::MemRefType>(load.getDest().getType());
            EXPECT_EQ(type.getDimSize(0), 1);
            EXPECT_EQ(type.getDimSize(2), 64);
            if (!tiled) {
              EXPECT_EQ(type.getDimSize(1), extent);
              std::fill(coverage.begin(), coverage.end(), 1);
              return;
            }
            EXPECT_LE(type.getDimSize(1), 128);
            auto view =
                load.getSource().getDefiningOp<mlir::memref::SubViewOp>();
            ASSERT_TRUE(view);
            EXPECT_EQ(view.getSourceType().getShape(),
                      (llvm::ArrayRef<int64_t>{1, extent, 64}));
            if (mode == 2) {
              auto source = view.getSource();
              for (unsigned level = 0; level < 1; ++level) {
                auto parent = source.getDefiningOp<mlir::memref::SubViewOp>();
                ASSERT_TRUE(parent);
                EXPECT_EQ(parent.getStaticOffsets(),
                          (llvm::ArrayRef<int64_t>{0, 0, 0}));
                EXPECT_EQ(parent.getStaticSizes(),
                          (llvm::ArrayRef<int64_t>{1, extent, 64}));
                EXPECT_EQ(parent.getStaticStrides(),
                          (llvm::ArrayRef<int64_t>{1, 1, 1}));
                source = parent.getSource();
              }
              EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(source));
            }
            auto offsets = view.getMixedOffsets();
            EXPECT_EQ(mlir::getConstantIntValue(offsets[0]), 0);
            EXPECT_EQ(mlir::getConstantIntValue(offsets[2]), 0);
            llvm::SmallVector<int64_t, 8> starts;
            if (auto constant = mlir::getConstantIntValue(offsets[1]))
              starts.push_back(*constant);
            else {
              auto loop = load->getParentOfType<mlir::scf::ForOp>();
              ASSERT_TRUE(loop);
              EXPECT_EQ(mlir::cast<mlir::Value>(offsets[1]),
                        loop.getInductionVar());
              auto lower = mlir::getConstantIntValue(loop.getLowerBound());
              auto upper = mlir::getConstantIntValue(loop.getUpperBound());
              auto step = mlir::getConstantIntValue(loop.getStep());
              ASSERT_TRUE(lower && upper && step);
              for (int64_t i = *lower; i < *upper; i += *step)
                starts.push_back(i);
            }
            for (int64_t start : starts)
              for (int64_t row = start; row < start + type.getDimSize(1);
                   ++row) {
                ASSERT_GE(row, 0);
                ASSERT_LT(row, extent);
                ++coverage[row];
              }
          });
          EXPECT_EQ(loads, tiled && extent != 1024 ? 2u : 1u);
          EXPECT_TRUE(llvm::all_of(coverage,
                                   [](unsigned count) { return count == 1; }));
        });
        EXPECT_EQ(consumers, tileCount);
        ASSERT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
        std::string detail;
        auto standalone =
            createStandaloneTileModules(std::move(module), &detail, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
        for (auto &tile : *standalone) {
          llvm::SmallVector<TileRegionOp, 2> regions;
          tile.module->walk(
              [&](TileRegionOp region) { regions.push_back(region); });
          TileRegionToInstrLoweringSession session(*tile.module->getContext());
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
}

TEST_F(StructuredToTileTest, DDRSubviewLoadsPreserveDynamicExtentSSA) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool rankReduced : {false, true}) {
      const std::string shape =
          (rankReduced ? "1x2x" : "2x") + std::to_string(extent) + "x64xf16";
      const std::string tensor = "tensor<" + shape + ">";
      const std::string ddr =
          "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
      const std::string spm =
          "memref<" + shape + ", #wafer.memory<spm, tensor>>";
      const std::string window = "memref<2x?x64xf16, strided<[" +
                                 std::to_string(extent * 64) +
                                 ", 64, 1]>, #wafer.memory<spm, tensor>>";
      const std::string owned =
          "memref<2x?x64xf16, #wafer.memory<spm, tensor>>";
      const std::string offsets = rankReduced ? "0, 0, 0, 0" : "0, 0, 0";
      const std::string sizes =
          rankReduced ? "1, 2, %size, 64" : "2, %size, 64";
      const std::string strides = rankReduced ? "1, 1, 1, 1" : "1, 1, 1";
      auto module =
          parse("module { wafer.tile.module card_id = 0 tile_id = 0 { "
                "func.func @entry(%arg: " +
                ddr +
                ", %count: index) { "
                "%tensor = bufferization.to_tensor %arg restrict : " +
                ddr + " wafer.tile.region(%tensor, %count : " + tensor +
                ", index) -> () { "
                "^bb0(%input: " +
                tensor +
                ", %size: index): "
                "%spm = bufferization.to_memref %input : " +
                spm + " %view = memref.subview %spm[" + offsets + "] [" +
                sizes + "] [" + strides + "] : " + spm + " to " + window +
                " %dst = memref.alloc(%size) : " + owned +
                " wafer.tile.copy_into %view into %dst : " + window + " into " +
                owned + " wafer.tile.yield } return } } }");
      ASSERT_TRUE(module);
      StructuredMaterializationRelations relations;
      rebuildCurrentBufferOwnerRelations(*module, relations);
      ASSERT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_EQ(movement.statistics.ddrLoads, 1u);
      EXPECT_EQ(countOps<StorageLoadOp>(module->getOperation()), 1u);
      module->walk([&](StorageLoadOp load) {
        auto allocation = load.getDest().getDefiningOp<mlir::memref::AllocOp>();
        ASSERT_TRUE(allocation);
        EXPECT_EQ(allocation.getType().getShape(),
                  (llvm::ArrayRef<int64_t>{2, mlir::ShapedType::kDynamic, 64}));
        ASSERT_EQ(allocation.getDynamicSizes().size(), 1u);
        auto region = load->getParentOfType<TileRegionOp>();
        EXPECT_EQ(allocation.getDynamicSizes().front(),
                  region.getBody().getArgument(1));
        auto view = load.getSource().getDefiningOp<mlir::memref::SubViewOp>();
        ASSERT_TRUE(view);
        EXPECT_EQ(view.getType().getShape(), allocation.getType().getShape());
        EXPECT_EQ(view.getSourceType().getRank(), rankReduced ? 4 : 3);
        EXPECT_TRUE(llvm::any_of(relations.buffers, [&](const auto &owner) {
          return owner.buffer == allocation.getResult() && owner.owner;
        }));
      });
      ASSERT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
      // Dynamic allocation/load IR is valid. The existing descriptor consumer
      // still rejects an unknown extent; this does not certify SPM capacity.
      auto tile = *module->getOps<TileModuleOp>().begin();
      auto entry = *tile.getOps<mlir::func::FuncOp>().begin();
      auto region = *entry.getOps<TileRegionOp>().begin();
      TileRegionToInstrLoweringSession session(*context);
      EXPECT_TRUE(mlir::failed(convertTileRegionToInstr(region, session)));
    }
}

TEST_F(StructuredToTileTest, RegionClosureIncludesOnlyPureLocalDependencies) {
  for (bool effect : {false, true}) {
    auto module = parse(makeSplitExchangeSource(1031));
    ASSERT_TRUE(module);
    llvm::SmallVector<llvm::SmallVector<TileRegionOp, 2>, 2> regions(2);
    for (TileModuleOp tile : module->getOps<TileModuleOp>())
      tile.walk([&](TileRegionOp region) {
        regions[tile.getTileIdAttr().getInt()].push_back(region);
      });
    StructuredMaterializationRelations relations;
    for (unsigned tile = 0; tile < 2; ++tile) {
      auto consumer = regions[tile][1];
      auto empty =
          *consumer.getBody().front().getOps<mlir::tensor::EmptyOp>().begin();
      mlir::OpBuilder builder(consumer);
      auto init = builder.create<TileRegionOp>(consumer.getLoc(),
                                               mlir::TypeRange{empty.getType()},
                                               mlir::ValueRange{});
      init.getBody().push_back(new mlir::Block());
      auto argument = consumer.getBody().front().addArgument(empty.getType(),
                                                             empty.getLoc());
      consumer.getInputsMutable().append(init.getResult(0));
      empty.getResult().replaceAllUsesWith(argument);
      empty->moveBefore(&init.getBody().front(), init.getBody().front().end());
      builder.setInsertionPointToEnd(&init.getBody().front());
      builder.create<TileYieldOp>(init.getLoc(), empty.getResult());
      if (effect) {
        auto function = consumer->getParentOfType<mlir::func::FuncOp>();
        builder.setInsertionPoint(function);
        auto external = builder.create<mlir::func::FuncOp>(
            function.getLoc(), "observe", builder.getFunctionType({}, {}));
        external.setPrivate();
        builder.setInsertionPoint(consumer);
        builder.create<mlir::func::CallOp>(consumer.getLoc(), "observe",
                                           mlir::TypeRange{},
                                           mlir::ValueRange{});
      }
      relations.boundaryRelations.push_back(
          {regions[tile][0].getResult(0),
           regions[1 - tile][1].getBody().getArgument(0)});
      relations.structuralOutputs.push_back({0, consumer.getResult(0)});
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    auto availability = analyzeCommunicationRegionClosure(*module, relations);
    ASSERT_TRUE(mlir::succeeded(availability));
    EXPECT_EQ(*availability,
              effect ? CommunicationRegionClosureAvailability::Unavailable
                     : CommunicationRegionClosureAvailability::Available);
    EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()), 6u);
    ASSERT_TRUE(mlir::succeeded(
        closeCrossTileCommunicationRegions(*module, relations)));
    EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()), effect ? 6u : 2u);
    EXPECT_EQ(countOps<mlir::func::CallOp>(module->getOperation()),
              effect ? 2u : 0u);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

TEST_F(StructuredToTileTest, SharedLocalInitializationIsMergedOnlyOnce) {
  for (int64_t tileCount : {4, 16}) {
    for (int64_t extent : {1024, 1025, 1031}) {
      for (bool effect : {false, true}) {
        SCOPED_TRACE(effect);
        SCOPED_TRACE(tileCount);
        SCOPED_TRACE(extent);
        auto module = parse(makeSequentialExchangeSource(extent, tileCount));
        ASSERT_TRUE(module);
        llvm::SmallVector<llvm::SmallVector<TileRegionOp, 4>, 16> scopes(
            tileCount);
        for (TileModuleOp tile : module->getOps<TileModuleOp>()) {
          llvm::SmallVector<TileRegionOp, 3> original;
          tile.walk([&](TileRegionOp region) { original.push_back(region); });
          mlir::OpBuilder builder(original[1]);
          auto secondProducer =
              mlir::cast<TileRegionOp>(builder.clone(*original[0]));
          original[2].getInputsMutable()[0].set(secondProducer.getResult(0));
          scopes[tile.getTileIdAttr().getInt()] = {original[0], original[1],
                                                   secondProducer, original[2]};
          // Both producers precede this shared initialization; both consumers
          // depend on it. Neither communication component initially owns it.
          auto firstEmpty = *original[1]
                                 .getBody()
                                 .front()
                                 .getOps<mlir::tensor::EmptyOp>()
                                 .begin();
          auto init = builder.create<TileRegionOp>(
              original[1].getLoc(), mlir::TypeRange{firstEmpty.getType()},
              mlir::ValueRange{});
          init.getBody().push_back(new mlir::Block());
          for (TileRegionOp consumer : {original[1], original[2]}) {
            auto empty = *consumer.getBody()
                              .front()
                              .getOps<mlir::tensor::EmptyOp>()
                              .begin();
            auto argument = consumer.getBody().front().addArgument(
                empty.getType(), empty.getLoc());
            consumer.getInputsMutable().append(init.getResult(0));
            empty.getResult().replaceAllUsesWith(argument);
            if (empty == firstEmpty)
              empty->moveBefore(&init.getBody().front(),
                                init.getBody().front().end());
            else
              empty.erase();
          }
          builder.setInsertionPointToEnd(&init.getBody().front());
          builder.create<TileYieldOp>(init.getLoc(), firstEmpty.getResult());
          auto function = original[0]->getParentOfType<mlir::func::FuncOp>();
          function.getBody().front().getTerminator()->setOperands(
              mlir::ValueRange{original[1].getResult(0),
                               original[2].getResult(0)});
          function.setFunctionType(builder.getFunctionType(
              function.getArgumentTypes(),
              mlir::TypeRange{firstEmpty.getType(), firstEmpty.getType()}));
          if (effect && tile.getTileIdAttr().getInt() == 0) {
            builder.setInsertionPoint(function);
            auto observer = builder.create<mlir::func::FuncOp>(
                function.getLoc(), "observe", builder.getFunctionType({}, {}));
            observer.setPrivate();
            builder.setInsertionPoint(original[1]);
            builder.create<mlir::func::CallOp>(original[1].getLoc(), "observe",
                                               mlir::TypeRange{},
                                               mlir::ValueRange{});
          }
        }
        StructuredMaterializationRelations relations;
        for (unsigned source = 0; source < scopes.size(); ++source)
          for (unsigned phase = 0; phase < 2; ++phase)
            for (unsigned destination = 0; destination < scopes.size();
                 ++destination) {
              if (source == destination)
                continue;
              unsigned argument = 1 + source - (source > destination);
              relations.boundaryRelations.push_back(
                  {scopes[source][2 * phase].getResult(0),
                   scopes[destination][2 * phase + 1].getBody().getArgument(
                       argument)});
            }
        for (auto &tile : scopes)
          for (unsigned phase : {1, 3})
            relations.structuralOutputs.push_back(
                {phase / 2, tile[phase].getResult(0)});
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        auto availability =
            analyzeCommunicationRegionClosure(*module, relations);
        ASSERT_TRUE(mlir::succeeded(availability));
        EXPECT_EQ(*availability,
                  effect ? CommunicationRegionClosureAvailability::Unavailable
                         : CommunicationRegionClosureAvailability::Available);
        CommunicationRegionClosureStatistics closure;
        ASSERT_TRUE(mlir::succeeded(
            closeCrossTileCommunicationRegions(*module, relations, &closure)));
        if (effect) {
          EXPECT_EQ(closure.closedExchangeComponents, 0u);
          EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()),
                    5 * tileCount);
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          continue;
        }
        EXPECT_EQ(closure.closedExchangeComponents, 2u);
        EXPECT_EQ(closure.mergedRegions, 5 * tileCount);
        EXPECT_EQ(closure.mergedTileScopes, tileCount);
        EXPECT_EQ(countOps<TileRegionOp>(module->getOperation()), tileCount);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        auto compute = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(compute.succeeded()) << compute.detail;
        auto movement = materializeTileBoundaryMovement(*module, relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        EXPECT_EQ(movement.statistics.peerSends,
                  2 * tileCount * (tileCount - 1));
        EXPECT_EQ(movement.statistics.peerReceives,
                  2 * tileCount * (tileCount - 1));
        EXPECT_EQ(movement.statistics.crossTileDDRStages, 0u);
        std::string detail;
        auto standalone =
            createStandaloneTileModules(std::move(module), &detail, &relations);
        ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
        llvm::SmallVector<mlir::ModuleOp, 16> instructionModules;
        for (StandaloneTileModule &tile : *standalone) {
          llvm::SmallVector<TileRegionOp, 1> tileRegions;
          tile.module->walk(
              [&](TileRegionOp region) { tileRegions.push_back(region); });
          ASSERT_EQ(tileRegions.size(), 1u);
          TileRegionToInstrLoweringSession session(*tile.module->getContext());
          ASSERT_TRUE(mlir::succeeded(
              convertTileRegionToInstr(tileRegions.front(), session)));
          ASSERT_TRUE(mlir::succeeded(
              convertBufferizationCopiesToInstr(*tile.module, session)));
          instructionModules.push_back(*tile.module);
        }
        auto completion = rebuildRequiredDirectDTEWaits(instructionModules);
        ASSERT_TRUE(completion.succeeded()) << completion.detail;
        instructionModules.clear();
        for (StandaloneTileModule &tile : *standalone) {
          ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
          TileMemoryPlanningFailure failure;
          auto planned = planTileMemory(std::move(tile.module), &failure);
          ASSERT_TRUE(mlir::succeeded(planned));
          tile.module = std::move(*planned);
          instructionModules.push_back(*tile.module);
        }
        EXPECT_TRUE(mlir::succeeded(
            verifyDirectDTETransportSchedule(instructionModules)));
        EXPECT_TRUE(
            mlir::succeeded(bindDirectDTETransport(instructionModules)));
      }
    }
  }
}

TEST_F(StructuredToTileTest,
       TiledOutputStoresStreamAllDestinationsInWriteOrder) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned sinks : {1u, 2u, 3u})
      for (bool shared : {false, true})
        for (bool window : {false, true}) {
          SCOPED_TRACE(extent);
          SCOPED_TRACE(sinks);
          SCOPED_TRACE(shared);
          SCOPED_TRACE(window);
          std::string shape = "2x" + std::to_string(extent) + "x64xf16";
          std::string spm = "memref<" + shape + ", #wafer.memory<spm, tensor>>";
          int64_t rows = extent + (window ? 6 : 0);
          int64_t batches = window ? 4 : 2;
          std::string ddr = "memref<" + std::to_string(batches) + "x" +
                            std::to_string(rows) +
                            "x64xf16, #wafer.memory<ddr, tensor>>";
          std::string destinationType =
              window
                  ? "memref<" + shape + ", strided<[" +
                        std::to_string(rows * 64) +
                        ", 64, 1], offset: " + std::to_string((rows + 3) * 64) +
                        ">, #wafer.memory<ddr, tensor>>"
                  : ddr;
          std::string text;
          llvm::raw_string_ostream ir(text);
          ir << "module {\n";
          if (shared)
            ir << "memref.global \"private\" @shared : " << ddr
               << " {wafer.ddr_resource = #wafer.ddr_resource<0>}\n";
          ir << "wafer.tile.module card_id = 0 tile_id = 0 { func.func @entry(";
          if (shared)
            ir << "%out0: " << ddr
               << " {wafer.ddr_binding = #wafer.ddr_binding<@shared, id = 0, "
                  "write>}";
          ir << ") {\n";
          for (unsigned sink = shared ? 1 : 0; sink < sinks; ++sink)
            ir << "%out" << sink << " = memref.alloc() : " << ddr << "\n";
          ir << "\"wafer.tile.region\"(";
          for (unsigned sink = 0; sink < sinks; ++sink)
            ir << (sink ? ", " : "") << "%out" << sink;
          ir << ") ({ ^bb0(";
          for (unsigned sink = 0; sink < sinks; ++sink)
            ir << (sink ? ", " : "") << "%dst" << sink << ": " << ddr;
          ir << "):\n%c0 = arith.constant 0 : index\n%c1 = arith.constant 1 : "
                "index\n%c2 = arith.constant 2 : index\n"
             << "%step = arith.constant 128 : index\n%end = arith.constant "
             << extent / 128 * 128 << " : index\n"
             << "%one = arith.constant 1.25 : f16\n%two = arith.constant 2.5 : "
                "f16\n"
             << "%full = memref.alloc() : " << spm << "\n"
             << "%result:2 = scf.for %batch = %c0 to %c2 step %c1 "
                "iter_args(%outer "
                "= %full, %count = %c0) -> ("
             << spm << ", index) {\n"
             << "%inner_result = scf.for %row = %c0 to %end step %step "
                "iter_args(%inner = %outer) -> ("
             << spm << ") {\n";
          auto write = [&](int64_t size, llvm::StringRef row,
                           llvm::StringRef carrier) {
            std::string piece = "memref<1x" + std::to_string(size) +
                                "x64xf16, #wafer.memory<spm, tensor>>";
            std::string view =
                "memref<1x" + std::to_string(size) + "x64xf16, strided<[" +
                std::to_string(extent * 64) +
                ", 64, 1], offset: ?>, #wafer.memory<spm, tensor>>";
            ir << "%view = memref.subview " << carrier << "[%batch, " << row
               << ", 0] [1, " << size << ", 64] [1, 1, 1] : " << spm << " to "
               << view << "\n"
               << "%a = memref.alloc() : " << piece
               << "\n%b = memref.alloc() : " << piece << "\n"
               << "wafer.tile.fill %a, %one : " << piece
               << ", f16\nwafer.tile.fill %b, %two : " << piece << ", f16\n"
               << "wafer.tile.copy_into %a into %view : " << piece << " into "
               << view << "\n"
               << "memref.copy %b, %view : " << piece << " to " << view << "\n"
               << "memref.copy %view, %view : " << view << " to " << view
               << "\n";
          };
          write(128, "%row", "%inner");
          ir << "scf.yield %inner : " << spm << "\n}\n";
          if (extent % 128)
            write(extent % 128, "%end", "%outer");
          ir << "%next_count = arith.addi %count, %c1 : index\nscf.yield "
                "%inner_result, %next_count : "
             << spm << ", index\n}\n";
          if (window)
            ir << "%collected = memref.alloc() : " << spm
               << "\nmemref.copy %result#0, %collected : " << spm << " to "
               << spm << "\n";
          for (unsigned sink = 0; sink < sinks; ++sink) {
            if (window)
              ir << "%window" << sink << " = memref.subview %dst" << sink
                 << "[1, 3, 0] [2, " << extent << ", 64] [1, 1, 1] : " << ddr
                 << " to " << destinationType << "\n";
            ir << "wafer.tile.store %" << (window ? "collected" : "result#0")
               << ", %" << (window ? "window" : "dst") << sink << " : " << spm
               << " -> " << destinationType << "\n";
          }
          ir << "wafer.tile.yield\n}) : (";
          for (unsigned sink = 0; sink < sinks; ++sink)
            ir << (sink ? ", " : "") << ddr;
          ir << ") -> ()\nreturn\n} } }";
          auto module = parse(text);
          ASSERT_TRUE(module);
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          BoundaryMovementStatistics statistics;
          materializeTiledOutputStores(*module, statistics);
          ASSERT_EQ(statistics.streamedOutputCarriers, window ? 2u : 1u);
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
          EXPECT_EQ(countOps<MoveCopyIntoOp>(*module), 0u);
          module->walk([&](mlir::scf::ForOp loop) {
            bool outer = !loop->getParentOfType<mlir::scf::ForOp>();
            ASSERT_EQ(loop.getNumRegionIterArgs(), outer ? 1u : 0u);
            if (outer) {
              EXPECT_TRUE(loop.getRegionIterArg(0).getType().isIndex());
              auto yield = mlir::cast<mlir::scf::YieldOp>(
                  loop.getBody()->getTerminator());
              auto next =
                  yield.getOperand(0).getDefiningOp<mlir::arith::AddIOp>();
              ASSERT_TRUE(next);
              EXPECT_EQ(next.getLhs(), loop.getRegionIterArg(0));
              EXPECT_EQ(mlir::getConstantIntValue(next.getRhs()), 1);
            }
          });
          EXPECT_EQ(countOps<mlir::memref::CopyOp>(*module), 0u);
          EXPECT_EQ(countOps<StorageStoreOp>(*module),
                    sinks * 2 * (extent % 128 ? 2 : 1));
          module->walk([&](mlir::memref::AllocOp allocation) {
            if (isWaferSPMMemRefType(allocation.getType())) {
              EXPECT_LE(allocation.getType().getDimSize(1), 128);
            }
          });
          TileRegionOp region;
          module->walk([&](TileRegionOp op) { region = op; });
          // Execute only control flow and coordinates, independently of the
          // transformation. Every output element receives 1.25 then 2.5.
          std::vector<std::vector<unsigned>> coverage(
              sinks, std::vector<unsigned>(batches * rows * 64));
          llvm::DenseMap<mlir::Value, int64_t> indices;
          auto index = [&](mlir::OpFoldResult value) -> int64_t {
            if (auto constant = mlir::getConstantIntValue(value))
              return *constant;
            return indices.lookup(mlir::cast<mlir::Value>(value));
          };
          auto destination =
              [&](auto &&self, mlir::Value value,
                  llvm::SmallVectorImpl<int64_t> &offsets) -> unsigned {
            if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>()) {
              for (unsigned axis = 0; axis < 3; ++axis)
                offsets[axis] += index(view.getMixedOffsets()[axis]);
              return self(self, view.getSource(), offsets);
            }
            if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
              auto loop = mlir::cast<mlir::scf::ForOp>(result.getOwner());
              return self(self, loop.getInitArgs()[result.getResultNumber()],
                          offsets);
            }
            auto argument = mlir::cast<mlir::BlockArgument>(value);
            if (argument.getOwner() == &region.getBody().front())
              return argument.getArgNumber();
            auto loop = mlir::cast<mlir::scf::ForOp>(
                argument.getOwner()->getParentOp());
            return self(self, loop.getInitArgs()[argument.getArgNumber() - 1],
                        offsets);
          };
          llvm::DenseMap<mlir::Value, unsigned> sourcePhases;
          region.walk([&](ComputeFillOp fill) {
            auto constant =
                fill.getValue().getDefiningOp<mlir::arith::ConstantOp>();
            sourcePhases[fill.getDest()] =
                mlir::cast<mlir::FloatAttr>(constant.getValue())
                            .getValueAsDouble() == 1.25
                    ? 0
                    : 1;
          });
          auto execute = [&](auto &&self, mlir::Block &block) -> void {
            for (auto &operation : block) {
              if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
                for (int64_t i = index(loop.getLowerBound());
                     i < index(loop.getUpperBound());
                     i += index(loop.getStep())) {
                  indices[loop.getInductionVar()] = i;
                  self(self, *loop.getBody());
                }
              } else if (auto store =
                             mlir::dyn_cast<StorageStoreOp>(operation)) {
                llvm::SmallVector<int64_t> offsets(3, 0);
                unsigned sink =
                    destination(destination, store.getDest(), offsets);
                ASSERT_LT(sink, sinks);
                auto type =
                    mlir::cast<mlir::MemRefType>(store.getSource().getType());
                ASSERT_EQ(type.getDimSize(0), 1);
                ASSERT_EQ(type.getDimSize(2), 64);
                ASSERT_TRUE(sourcePhases.contains(store.getSource()));
                for (int64_t row = 0; row < type.getDimSize(1); ++row)
                  for (int64_t col = 0; col < 64; ++col) {
                    int64_t address =
                        ((offsets[0] * rows + offsets[1] + row) * 64) +
                        offsets[2] + col;
                    ASSERT_GE(address, 0);
                    ASSERT_LT(address,
                              static_cast<int64_t>(coverage[sink].size()));
                    ASSERT_EQ(coverage[sink][address]++,
                              sourcePhases.lookup(store.getSource()));
                  }
              }
            }
          };
          execute(execute, region.getBody().front());
          for (auto &output : coverage)
            for (int64_t batch = 0; batch < batches; ++batch)
              for (int64_t row = 0; row < rows; ++row)
                for (int64_t col = 0; col < 64; ++col) {
                  bool inside = !window || (batch >= 1 && batch < 3 &&
                                            row >= 3 && row < extent + 3);
                  ASSERT_EQ(output[(batch * rows + row) * 64 + col],
                            inside ? 2u : 0u);
                }
          StructuredMaterializationRelations relations;
          std::string detail;
          auto standalone = createStandaloneTileModules(std::move(module),
                                                        &detail, &relations);
          ASSERT_TRUE(mlir::succeeded(standalone)) << detail;
          for (auto &tile : *standalone) {
            TileRegionToInstrLoweringSession session(*context);
            auto current = *tile.module->getOps<mlir::func::FuncOp>().begin();
            llvm::SmallVector<TileRegionOp> regions;
            current.walk([&](TileRegionOp op) { regions.push_back(op); });
            for (auto op : regions)
              ASSERT_TRUE(
                  mlir::succeeded(convertTileRegionToInstr(op, session)));
            ASSERT_TRUE(mlir::succeeded(
                convertBufferizationCopiesToInstr(*tile.module, session)));
            ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*tile.module)));
            TileMemoryPlanningFailure failure;
            auto planned = planTileMemory(std::move(tile.module), &failure);
            ASSERT_TRUE(mlir::succeeded(planned));
          }
        }
}

TEST_F(StructuredToTileTest, StridedSPMEndpointsPreserveEveryDMAAddress) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (llvm::StringRef element : {"f16", "bf16"}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(element.str());
      const std::string shape =
          "2x" + std::to_string(extent) + "x16x" + element.str();
      const std::string ddr =
          "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
      const std::string carrier = "memref<2x" + std::to_string(extent) +
                                  "x64x" + element.str() +
                                  ", #wafer.memory<spm, tensor>>";
      const std::string view =
          "memref<" + shape + ", strided<[" + std::to_string(extent * 64) +
          ", 64, 1], offset: ?>, #wafer.memory<spm, tensor>>";
      std::string text;
      llvm::raw_string_ostream os(text);
      os << "module { func.func @entry(%input: " << ddr << ", %output: " << ddr
         << ") {\n"
         << "\"wafer.tile.region\"(%input, %output) ({\n"
         << "^bb0(%in: " << ddr << ", %out: " << ddr << "):\n"
         << "%storage = memref.alloc() : " << carrier << "\n"
         << "%c0 = arith.constant 0 : index\n%c16 = arith.constant 16 : "
            "index\n%c64 = arith.constant 64 : index\n"
         << "scf.for %shift = %c0 to %c64 step %c16 {\n"
         << "%view = memref.subview %storage[0, 0, %shift] [2, " << extent
         << ", 16] [1, 1, 1] : " << carrier << " to " << view << "\n"
         << "wafer.tile.load %in into %view : " << ddr << " into " << view
         << "\n"
         << "wafer.tile.store %view, %out : " << view << " -> " << ddr
         << "\n}\n"
         << "wafer.tile.yield\n}) : (" << ddr << ", " << ddr
         << ") -> ()\nreturn\n}}";
      auto module = parse(text);
      ASSERT_TRUE(module);
      TileRegionOp region;
      module->walk([&](TileRegionOp op) { region = op; });
      TileRegionToInstrLoweringSession session(*context);
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      // Each row contains 32 bytes. On SPM adjacent rows are 128 bytes
      // apart; the dynamic view supplies the same column shift to both
      // engines in each of the four iterations.
      std::set<std::pair<int64_t, int64_t>> reads, writes, expected;
      for (int64_t row = 0; row < 2 * extent; ++row)
        for (int64_t byte = 0; byte < 32; ++byte)
          expected.emplace(row * 32 + byte, row * 128 + byte);
      module->walk([&](InstrRDMAOp op) {
        ASSERT_TRUE(op.getSrcOffset());
        ASSERT_TRUE(op.getDstOffset());
        EXPECT_EQ(op.getByteCount(), 32);
        EXPECT_EQ(op.getInnerBytes(), 32);
        EXPECT_TRUE(op.getDest().getDefiningOp<mlir::memref::SubViewOp>());
        for (int64_t byte = 0; byte < static_cast<int64_t>(op.getByteCount()); ++byte)
          EXPECT_TRUE(
              reads
                  .emplace(*op.getSrcOffset() + byte, *op.getDstOffset() + byte)
                  .second);
      });
      module->walk([&](InstrWDMAOp op) {
        ASSERT_TRUE(op.getSrcOffset());
        ASSERT_TRUE(op.getDstOffset());
        EXPECT_EQ(op.getByteCount(), 32);
        EXPECT_EQ(op.getInnerBytes(), 32);
        EXPECT_TRUE(op.getSource().getDefiningOp<mlir::memref::SubViewOp>());
        for (int64_t byte = 0; byte < static_cast<int64_t>(op.getByteCount()); ++byte)
          EXPECT_TRUE(
              writes
                  .emplace(*op.getDstOffset() + byte, *op.getSrcOffset() + byte)
                  .second);
      });
      EXPECT_EQ(reads, expected);
      EXPECT_EQ(writes, expected);
    }
  }
}

TEST_F(StructuredToTileTest, NTensorBoundaryUsesExactMappedDMA) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (llvm::StringRef element : {"f16", "bf16"}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(element.str());
      std::string text;
      llvm::raw_string_ostream os(text);
      const std::string shape =
          "2x" + std::to_string(extent) + "x3x" + element.str();
      const std::string ddr =
          "memref<" + shape + ", #wafer.memory<ddr, tensor>>";
      const std::string spm =
          "memref<" + shape + ", #wafer.memory<spm, ntensor>>";
      os << "module { func.func @entry(%input: " << ddr << ", %output: " << ddr
         << ") {\n"
         << "\"wafer.tile.region\"(%input, %output) ({ ^bb0(%in: " << ddr
         << ", %out: " << ddr << "):\n"
         << "%storage = memref.alloc() : " << spm << "\n"
         << "wafer.tile.load %in into %storage : " << ddr << " into " << spm
         << "\nwafer.tile.store %storage, %out : " << spm << " -> " << ddr
         << "\nwafer.tile.yield\n}) : (" << ddr << ", " << ddr
         << ") -> ()\nreturn\n}}";
      auto module = parse(text);
      ASSERT_TRUE(module);
      TileRegionOp region;
      module->walk([&](TileRegionOp op) { region = op; });
      TileRegionToInstrLoweringSession session(*context);
      ASSERT_TRUE(mlir::succeeded(convertTileRegionToInstr(region, session)));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      EXPECT_EQ(countOps<InstrRDMAOp>(*module), 1u);
      EXPECT_EQ(countOps<InstrWDMAOp>(*module), 1u);
      EXPECT_EQ(countOps<InstrGatherScatterOp>(*module), 0u);
      const int64_t bytes = 2 * extent * 3 * 2;
      module->walk([&](InstrRDMAOp load) {
        EXPECT_EQ(load.getByteCount(), bytes);
        EXPECT_EQ(load.getInnerBytes(), bytes);
        ASSERT_TRUE(load.getSrcOffset());
        ASSERT_TRUE(load.getDstOffset());
        EXPECT_EQ(*load.getSrcOffset(), 0);
        EXPECT_EQ(*load.getDstOffset(), 0);
        EXPECT_EQ(getWaferMemoryAttr(load.getDest().getType()).getLayout(),
                  MemLayout::NTensor);
      });
      module->walk([&](InstrWDMAOp store) {
        EXPECT_EQ(store.getByteCount(), bytes);
        EXPECT_EQ(store.getInnerBytes(), bytes);
        ASSERT_TRUE(store.getSrcOffset());
        ASSERT_TRUE(store.getDstOffset());
        EXPECT_EQ(*store.getSrcOffset(), 0);
        EXPECT_EQ(*store.getDstOffset(), 0);
        EXPECT_EQ(getWaferMemoryAttr(store.getSource().getType()).getLayout(),
                  MemLayout::NTensor);
      });
      ASSERT_TRUE(mlir::succeeded(rebuildRequiredNCCJoins(*module)));
      TileMemoryPlanningFailure failure;
      auto planned = planTileMemory(std::move(module), &failure);
      ASSERT_TRUE(mlir::succeeded(planned));
    }
  }
}

TEST_F(StructuredToTileTest, TiledOutputStoresPreserveReadsAndUnknownAliases) {
  for (llvm::StringRef variant :
       {"read", "yield", "alias", "destination-alias", "late-destination",
        "write-after-store", "ddr-copy-source"}) {
    SCOPED_TRACE(variant.str());
    auto module = parse(R"mlir(
      module {
        wafer.tile.module card_id = 0 tile_id = 0 {
          func.func @entry() -> memref<2x1024x128xf16, #wafer.memory<ddr, tensor>> {
            %output = memref.alloc() : memref<2x1024x128xf16, #wafer.memory<ddr, tensor>>
            "wafer.tile.region"(%output, %output) ({
            ^bb0(%out: memref<2x1024x128xf16, #wafer.memory<ddr, tensor>>, %alias: memref<2x1024x128xf16, #wafer.memory<ddr, tensor>>):
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %c1024 = arith.constant 1024 : index
              %full = memref.alloc() : memref<2x1024x128xf16, #wafer.memory<spm, tensor>>
              %tile = memref.alloc() : memref<2x128x128xf16, #wafer.memory<spm, tensor>>
              %result = scf.for %i = %c0 to %c1024 step %c128 iter_args(%carrier = %full) -> (memref<2x1024x128xf16, #wafer.memory<spm, tensor>>) {
                %view = memref.subview %carrier[0, %i, 0] [2, 128, 128] [1, 1, 1] : memref<2x1024x128xf16, #wafer.memory<spm, tensor>> to memref<2x128x128xf16, strided<[131072, 128, 1], offset: ?>, #wafer.memory<spm, tensor>>
                wafer.tile.copy_into %tile into %view : memref<2x128x128xf16, #wafer.memory<spm, tensor>> into memref<2x128x128xf16, strided<[131072, 128, 1], offset: ?>, #wafer.memory<spm, tensor>>
                scf.yield %carrier : memref<2x1024x128xf16, #wafer.memory<spm, tensor>>
              }
              wafer.tile.store %result, %out : memref<2x1024x128xf16, #wafer.memory<spm, tensor>> -> memref<2x1024x128xf16, #wafer.memory<ddr, tensor>>
              wafer.tile.yield
            }) : (memref<2x1024x128xf16, #wafer.memory<ddr, tensor>>, memref<2x1024x128xf16, #wafer.memory<ddr, tensor>>) -> ()
            return %output : memref<2x1024x128xf16, #wafer.memory<ddr, tensor>>
          }
        }
      })mlir");
    ASSERT_TRUE(module);
    mlir::scf::ForOp loop;
    StorageStoreOp store;
    module->walk([&](mlir::scf::ForOp op) { loop = op; });
    module->walk([&](StorageStoreOp op) { store = op; });
    mlir::OpBuilder builder(loop);
    mlir::Value full = loop.getInitArgs()[0];
    auto region = loop->getParentOfType<TileRegionOp>();
    if (variant == "read") {
      auto zero =
          builder.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 0);
      builder.create<mlir::memref::LoadOp>(loop.getLoc(), full,
                                           mlir::ValueRange{zero, zero, zero});
    } else if (variant == "yield") {
      auto other = builder.create<mlir::memref::AllocOp>(
          loop.getLoc(), mlir::cast<mlir::MemRefType>(full.getType()));
      loop.getBody()->getTerminator()->setOperand(0, other.getResult());
    } else if (variant == "alias") {
      auto type = mlir::cast<mlir::MemRefType>(full.getType());
      auto dynamic = mlir::MemRefType::get(
          {2, mlir::ShapedType::kDynamic, 128}, type.getElementType(),
          type.getLayout(), type.getMemorySpace());
      builder.create<mlir::memref::CastOp>(loop.getLoc(), dynamic, full);
    } else if (variant == "destination-alias") {
      auto other = builder.create<mlir::memref::AllocOp>(
          loop.getLoc(), mlir::cast<mlir::MemRefType>(full.getType()));
      builder.create<StorageLoadOp>(
          loop.getLoc(), region.getBody().front().getArgument(1), other);
    } else if (variant == "ddr-copy-source") {
      auto other = builder.create<mlir::memref::AllocOp>(
          loop.getLoc(),
          mlir::cast<mlir::MemRefType>(store.getDest().getType()));
      builder.create<mlir::memref::CopyOp>(loop.getLoc(), other, full);
    } else if (variant == "write-after-store") {
      builder.setInsertionPointAfter(store);
      auto other = builder.create<mlir::memref::AllocOp>(
          store.getLoc(), mlir::cast<mlir::MemRefType>(full.getType()));
      builder.create<MoveCopyIntoOp>(store.getLoc(), other, full);
    } else {
      builder.setInsertionPoint(store);
      auto zero =
          builder.create<mlir::arith::ConstantIndexOp>(store.getLoc(), 0);
      auto lateOffset =
          builder.create<mlir::arith::AddIOp>(store.getLoc(), zero, zero);
      auto view = builder.create<mlir::memref::SubViewOp>(
          store.getLoc(), store.getDest(),
          llvm::ArrayRef<mlir::OpFoldResult>{lateOffset.getResult(),
                                             builder.getIndexAttr(0),
                                             builder.getIndexAttr(0)},
          mlir::getAsIndexOpFoldResult(builder.getContext(), {2, 1024, 128}),
          mlir::getAsIndexOpFoldResult(builder.getContext(), {1, 1, 1}));
      store->setOperand(1, view);
    }
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    std::string before;
    llvm::raw_string_ostream beforeStream(before);
    module->print(beforeStream);
    BoundaryMovementStatistics statistics;
    materializeTiledOutputStores(*module, statistics);
    std::string after;
    llvm::raw_string_ostream afterStream(after);
    module->print(afterStream);
    EXPECT_EQ(before, after);
    EXPECT_EQ(statistics.streamedOutputCarriers, 0u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  }
}

} // namespace
