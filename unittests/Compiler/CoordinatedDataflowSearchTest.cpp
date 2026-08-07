//===- CoordinatedDataflowSearchTest.cpp ---------------------------------===//

#include "../../lib/Wafer/Compiler/CoordinatedDataflowSearch.h"
#include "../../lib/Wafer/Compiler/AttentionImplementationAlternative.h"
#include "../../lib/Wafer/Compiler/BoundedRankExecutor.h"
#include "../../lib/Wafer/Compiler/CompilationInternal.h"
#include "../../lib/Wafer/Compiler/StructuredImplementationAlternative.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/CompleteRankMaterialization.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <optional>

namespace {

class CountingImplementationPoint final
    : public wafer::compiler::detail::StructuredImplementationAlternativePoint {
public:
  explicit CountingImplementationPoint(
      std::shared_ptr<uint64_t> materializationCount)
      : StructuredImplementationAlternativePoint(
            {/*stableKey=*/"test.single-point", /*stableOrdinal=*/0},
            {/*outputShape=*/{8}, /*reductionShape=*/{}},
            {/*outputTileSizes=*/{4}, /*reductionTileSizes=*/{},
             /*parallelPartitionCount=*/1},
            {/*logicalOutputElements=*/8,
             /*logicalReductionElements=*/1,
             /*outputTileCount=*/2,
             /*reductionTileCount=*/1,
             /*parallelPartitionCount=*/1,
             /*structuredWorkUnitUpperBound=*/2}),
        materializationCount(std::move(materializationCount)) {}

  mlir::LogicalResult
  materialize(mlir::ModuleOp isolatedStructuredModule,
              std::string *failureReason = nullptr,
              wafer::compiler::detail::
                  StructuredImplementationAlternativeMaterialization * =
                      nullptr) const final {
    if (!isolatedStructuredModule) {
      if (failureReason)
        *failureReason = "counting point requires an isolated module";
      return mlir::failure();
    }
    ++*materializationCount;
    return mlir::success();
  }

private:
  std::shared_ptr<uint64_t> materializationCount;
};

class CountingImplementationProvider final
    : public wafer::compiler::detail::
          StructuredImplementationAlternativeProvider {
public:
  explicit CountingImplementationProvider(
      std::shared_ptr<uint64_t> materializationCount)
      : materializationCount(std::move(materializationCount)) {}

  llvm::StringRef getStableKey() const final {
    return "test.counting-provider";
  }

  mlir::LogicalResult query(
      mlir::ModuleOp currentStructuredModule,
      const wafer::compiler::detail::StructuredImplementationAlternativeQuery &,
      wafer::compiler::detail::StructuredImplementationAlternativePoints
          &points,
      std::string *failureReason = nullptr) const final {
    if (!currentStructuredModule) {
      if (failureReason)
        *failureReason = "counting provider requires a module";
      return mlir::failure();
    }
    points.push_back(
        std::make_unique<CountingImplementationPoint>(materializationCount));
    return mlir::success();
  }

private:
  std::shared_ptr<uint64_t> materializationCount;
};

struct ManyPointMaterializationTrace {
  void record(uint64_t pointOrdinal) {
    std::lock_guard<std::mutex> lock(mutex);
    pointOrdinals.push_back(pointOrdinal);
  }

  std::mutex mutex;
  std::vector<uint64_t> pointOrdinals;
};

class ManyPointImplementationPoint final
    : public wafer::compiler::detail::StructuredImplementationAlternativePoint {
public:
  ManyPointImplementationPoint(
      uint64_t pointOrdinal,
      std::shared_ptr<ManyPointMaterializationTrace> trace)
      : StructuredImplementationAlternativePoint(
            {/*stableKey=*/
             (llvm::Twine("test.many-point.") +
              llvm::Twine(1000 + pointOrdinal))
                 .str(),
             /*stableOrdinal=*/pointOrdinal},
            {/*outputShape=*/{64}, /*reductionShape=*/{}},
            {/*outputTileSizes=*/
             {static_cast<int64_t>(pointOrdinal + 1)},
             /*reductionTileSizes=*/{},
             /*parallelPartitionCount=*/1},
            {/*logicalOutputElements=*/64,
             /*logicalReductionElements=*/1,
             /*outputTileCount=*/
             (64 + pointOrdinal) / (pointOrdinal + 1),
             /*reductionTileCount=*/1,
             /*parallelPartitionCount=*/1,
             /*structuredWorkUnitUpperBound=*/
             (64 + pointOrdinal) / (pointOrdinal + 1)}),
        pointOrdinal(pointOrdinal), trace(std::move(trace)) {}

  mlir::LogicalResult
  materialize(mlir::ModuleOp isolatedStructuredModule,
              std::string *failureReason = nullptr,
              wafer::compiler::detail::
                  StructuredImplementationAlternativeMaterialization * =
                      nullptr) const final {
    if (!isolatedStructuredModule) {
      if (failureReason)
        *failureReason =
            "many-point implementation requires an isolated module";
      return mlir::failure();
    }
    trace->record(pointOrdinal);
    return mlir::success();
  }

private:
  uint64_t pointOrdinal;
  std::shared_ptr<ManyPointMaterializationTrace> trace;
};

class ManyPointImplementationProvider final
    : public wafer::compiler::detail::
          StructuredImplementationAlternativeProvider {
public:
  ManyPointImplementationProvider(
      bool reverse, std::shared_ptr<ManyPointMaterializationTrace> trace)
      : reverse(reverse), trace(std::move(trace)) {}

  llvm::StringRef getStableKey() const final {
    return "test.many-point-provider";
  }

  mlir::LogicalResult query(
      mlir::ModuleOp currentStructuredModule,
      const wafer::compiler::detail::StructuredImplementationAlternativeQuery &,
      wafer::compiler::detail::StructuredImplementationAlternativePoints
          &points,
      std::string *failureReason = nullptr) const final {
    if (!currentStructuredModule) {
      if (failureReason)
        *failureReason = "many-point provider requires a module";
      return mlir::failure();
    }
    constexpr uint64_t pointCount = 48;
    for (uint64_t step = 0; step < pointCount; ++step) {
      uint64_t pointOrdinal = reverse ? pointCount - step - 1 : step;
      points.push_back(
          std::make_unique<ManyPointImplementationPoint>(pointOrdinal, trace));
    }
    return mlir::success();
  }

private:
  bool reverse;
  std::shared_ptr<ManyPointMaterializationTrace> trace;
};

struct ParallelismCoverageMaterializationTrace {
  void record(int64_t partitionCount) {
    std::lock_guard<std::mutex> lock(mutex);
    partitionCounts.push_back(partitionCount);
  }

  std::mutex mutex;
  std::vector<int64_t> partitionCounts;
};

class ParallelismCoverageImplementationPoint final
    : public wafer::compiler::detail::StructuredImplementationAlternativePoint {
public:
  ParallelismCoverageImplementationPoint(
      uint64_t pointOrdinal, int64_t partitionCount,
      std::shared_ptr<ParallelismCoverageMaterializationTrace> trace)
      : StructuredImplementationAlternativePoint(
            {/*stableKey=*/
             (llvm::Twine("test.parallelism-coverage.") +
              llvm::Twine(pointOrdinal))
                 .str(),
             /*stableOrdinal=*/pointOrdinal},
            {/*outputShape=*/{8}, /*reductionShape=*/{}},
            {/*outputTileSizes=*/{4}, /*reductionTileSizes=*/{},
             /*parallelPartitionCount=*/partitionCount},
            {/*logicalOutputElements=*/8,
             /*logicalReductionElements=*/1,
             /*outputTileCount=*/2,
             /*reductionTileCount=*/1,
             /*parallelPartitionCount=*/
             static_cast<uint64_t>(partitionCount),
             /*structuredWorkUnitUpperBound=*/2}),
        partitionCount(partitionCount), trace(std::move(trace)) {}

  mlir::LogicalResult
  materialize(mlir::ModuleOp isolatedStructuredModule,
              std::string *failureReason = nullptr,
              wafer::compiler::detail::
                  StructuredImplementationAlternativeMaterialization * =
                      nullptr) const final {
    if (!isolatedStructuredModule) {
      if (failureReason)
        *failureReason =
            "parallelism coverage point requires an isolated module";
      return mlir::failure();
    }
    trace->record(partitionCount);
    return mlir::success();
  }

private:
  int64_t partitionCount;
  std::shared_ptr<ParallelismCoverageMaterializationTrace> trace;
};

class ParallelismCoverageImplementationProvider final
    : public wafer::compiler::detail::
          StructuredImplementationAlternativeProvider {
public:
  explicit ParallelismCoverageImplementationProvider(
      std::shared_ptr<ParallelismCoverageMaterializationTrace> trace)
      : trace(std::move(trace)) {}

  llvm::StringRef getStableKey() const final {
    return "test.parallelism-coverage-provider";
  }

  mlir::LogicalResult query(
      mlir::ModuleOp currentStructuredModule,
      const wafer::compiler::detail::StructuredImplementationAlternativeQuery &,
      wafer::compiler::detail::StructuredImplementationAlternativePoints
          &points,
      std::string *failureReason = nullptr) const final {
    if (!currentStructuredModule) {
      if (failureReason)
        *failureReason = "parallelism coverage provider requires a module";
      return mlir::failure();
    }
    constexpr std::array<int64_t, 5> partitionCounts = {1, 2, 4, 8, 16};
    for (auto [ordinal, partitionCount] : llvm::enumerate(partitionCounts))
      points.push_back(std::make_unique<ParallelismCoverageImplementationPoint>(
          ordinal, partitionCount, trace));
    return mlir::success();
  }

private:
  std::shared_ptr<ParallelismCoverageMaterializationTrace> trace;
};

struct TileAxisCoverageMaterializationTrace {
  void record(int64_t outputTileSize, int64_t reductionTileSize) {
    std::lock_guard<std::mutex> lock(mutex);
    tileSizePairs.push_back({outputTileSize, reductionTileSize});
  }

  std::mutex mutex;
  std::vector<std::array<int64_t, 2>> tileSizePairs;
};

class TileAxisCoverageImplementationPoint final
    : public wafer::compiler::detail::StructuredImplementationAlternativePoint {
public:
  TileAxisCoverageImplementationPoint(
      uint64_t pointOrdinal, int64_t outputTileSize, int64_t reductionTileSize,
      std::shared_ptr<TileAxisCoverageMaterializationTrace> trace,
      std::optional<uint64_t> estimatedPeakLiveBytes = std::nullopt,
      std::optional<uint64_t> estimatedComputeScalarOps = std::nullopt)
      : StructuredImplementationAlternativePoint(
            {/*stableKey=*/
             (llvm::Twine("test.tile-axis-coverage.") +
              llvm::Twine(pointOrdinal))
                 .str(),
             /*stableOrdinal=*/pointOrdinal},
            {/*outputShape=*/{256}, /*reductionShape=*/{256}},
            {/*outputTileSizes=*/{outputTileSize},
             /*reductionTileSizes=*/{reductionTileSize},
             /*parallelPartitionCount=*/1},
            {/*logicalOutputElements=*/256,
             /*logicalReductionElements=*/256,
             /*outputTileCount=*/
             static_cast<uint64_t>(256 / outputTileSize),
             /*reductionTileCount=*/
             static_cast<uint64_t>(256 / reductionTileSize),
             /*parallelPartitionCount=*/1,
             /*structuredWorkUnitUpperBound=*/
             static_cast<uint64_t>(256 / outputTileSize) *
                 static_cast<uint64_t>(256 / reductionTileSize),
             /*estimatedPeakLiveBytes=*/estimatedPeakLiveBytes,
             /*estimatedComputeScalarOps=*/estimatedComputeScalarOps}),
        outputTileSize(outputTileSize), reductionTileSize(reductionTileSize),
        trace(std::move(trace)) {}

  mlir::LogicalResult
  materialize(mlir::ModuleOp isolatedStructuredModule,
              std::string *failureReason = nullptr,
              wafer::compiler::detail::
                  StructuredImplementationAlternativeMaterialization * =
                      nullptr) const final {
    if (!isolatedStructuredModule) {
      if (failureReason)
        *failureReason = "tile-axis coverage point requires an isolated module";
      return mlir::failure();
    }
    trace->record(outputTileSize, reductionTileSize);
    return mlir::success();
  }

private:
  int64_t outputTileSize;
  int64_t reductionTileSize;
  std::shared_ptr<TileAxisCoverageMaterializationTrace> trace;
};

class TileAxisCoverageImplementationProvider final
    : public wafer::compiler::detail::
          StructuredImplementationAlternativeProvider {
public:
  explicit TileAxisCoverageImplementationProvider(
      std::shared_ptr<TileAxisCoverageMaterializationTrace> trace)
      : trace(std::move(trace)) {}

  llvm::StringRef getStableKey() const final {
    return "test.tile-axis-coverage-provider";
  }

  mlir::LogicalResult query(
      mlir::ModuleOp currentStructuredModule,
      const wafer::compiler::detail::StructuredImplementationAlternativeQuery &,
      wafer::compiler::detail::StructuredImplementationAlternativePoints
          &points,
      std::string *failureReason = nullptr) const final {
    if (!currentStructuredModule) {
      if (failureReason)
        *failureReason = "tile-axis coverage provider requires a module";
      return mlir::failure();
    }
    constexpr std::array<int64_t, 8> commonTileSizes = {2,  4,  8,   16,
                                                        32, 64, 128, 256};
    uint64_t pointOrdinal = 0;
    for (int64_t outputTileSize : commonTileSizes)
      for (int64_t reductionTileSize : commonTileSizes)
        points.push_back(std::make_unique<TileAxisCoverageImplementationPoint>(
            pointOrdinal++, outputTileSize, reductionTileSize, trace));

    // The 64 Cartesian points above all have strictly less structural work
    // than these two points. A work-only 64-point cap therefore drops both:
    // one is the sole output-tile=1 representative and the other is the sole
    // reduction-tile=1 representative.
    points.push_back(std::make_unique<TileAxisCoverageImplementationPoint>(
        pointOrdinal++, /*outputTileSize=*/1, /*reductionTileSize=*/2, trace));
    points.push_back(std::make_unique<TileAxisCoverageImplementationPoint>(
        pointOrdinal++, /*outputTileSize=*/2, /*reductionTileSize=*/1, trace));
    return mlir::success();
  }

private:
  std::shared_ptr<TileAxisCoverageMaterializationTrace> trace;
};

class ResourceHeadroomImplementationProvider final
    : public wafer::compiler::detail::
          StructuredImplementationAlternativeProvider {
public:
  explicit ResourceHeadroomImplementationProvider(
      std::shared_ptr<TileAxisCoverageMaterializationTrace> trace)
      : trace(std::move(trace)) {}

  llvm::StringRef getStableKey() const final {
    return "test.resource-headroom-provider";
  }

  mlir::LogicalResult query(
      mlir::ModuleOp currentStructuredModule,
      const wafer::compiler::detail::StructuredImplementationAlternativeQuery &,
      wafer::compiler::detail::StructuredImplementationAlternativePoints
          &points,
      std::string *failureReason = nullptr) const final {
    if (!currentStructuredModule) {
      if (failureReason)
        *failureReason = "resource headroom provider requires a module";
      return mlir::failure();
    }
    // The first point is cheaper but consumes more than half the default SPM
    // window. The second leaves complete-candidate headroom and must lead each
    // residency family without deleting the first point from backfill.
    points.push_back(std::make_unique<TileAxisCoverageImplementationPoint>(
        /*pointOrdinal=*/0, /*outputTileSize=*/2,
        /*reductionTileSize=*/2, trace,
        /*estimatedPeakLiveBytes=*/2'000'000,
        /*estimatedComputeScalarOps=*/1));
    points.push_back(std::make_unique<TileAxisCoverageImplementationPoint>(
        /*pointOrdinal=*/1, /*outputTileSize=*/1,
        /*reductionTileSize=*/1, trace,
        /*estimatedPeakLiveBytes=*/1'000'000,
        /*estimatedComputeScalarOps=*/2));
    return mlir::success();
  }

private:
  std::shared_ptr<TileAxisCoverageMaterializationTrace> trace;
};

class CoordinatedDataflowSearchTest : public ::testing::Test {
protected:
  CoordinatedDataflowSearchTest() {
    wafer::compiler::detail::registerCompilationDialects(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out = tensor.empty() : tensor<8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>) outs(%out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    } -> tensor<8xf16>
    return %result : tensor<8xf16>
  }

}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeDecodeAttentionProgram() {
    std::string path = std::string(WAFER_TEST_SOURCE_DIR) +
                       "/test/Transforms/materialize-flash-decoding.mlir";
    return mlir::parseSourceFile<mlir::ModuleOp>(
        path, mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeChainProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<8xf16>) -> tensor<8xf16> {
    %producer_out = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input : tensor<8xf16>)
        outs(%producer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<8xf16>
    %consumer_out = tensor.empty() : tensor<8xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%producer : tensor<8xf16>)
        outs(%consumer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %squared = arith.mulf %value, %value : f16
      linalg.yield %squared : f16
    } -> tensor<8xf16>
    return %consumer : tensor<8xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeReshapedOutputChainProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<1x128xf16>) -> tensor<1x1x1x128xf16> {
    %producer_out = tensor.empty() : tensor<1x128xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%input : tensor<1x128xf16>)
        outs(%producer_out : tensor<1x128xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<1x128xf16>
    %consumer_out = tensor.empty() : tensor<1x128xf16>
    %consumer = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%producer : tensor<1x128xf16>)
        outs(%consumer_out : tensor<1x128xf16>) {
    ^bb0(%value: f16, %old: f16):
      %squared = arith.mulf %value, %value : f16
      linalg.yield %squared : f16
    } -> tensor<1x128xf16>
    %result = tensor.expand_shape %consumer [[0, 1, 2], [3]]
        output_shape [1, 1, 1, 128]
        : tensor<1x128xf16> into tensor<1x1x1x128xf16>
    return %result : tensor<1x1x1x128xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeMixedShapeFanoutProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<8xf16>)
      -> (tensor<8xf16>, tensor<4xf16>, tensor<4xf16>) {
    %producer_out = tensor.empty() : tensor<8xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%input : tensor<8xf16>) outs(%producer_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<8xf16>
    %whole_out = tensor.empty() : tensor<8xf16>
    %whole = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<8xf16>) outs(%whole_out : tensor<8xf16>) {
    ^bb0(%value: f16, %old: f16):
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    } -> tensor<8xf16>
    %low_slice = tensor.extract_slice %producer[0] [4] [1]
        : tensor<8xf16> to tensor<4xf16>
    %low_out = tensor.empty() : tensor<4xf16>
    %low = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%low_slice : tensor<4xf16>) outs(%low_out : tensor<4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %square = arith.mulf %value, %value : f16
      linalg.yield %square : f16
    } -> tensor<4xf16>
    %high_slice = tensor.extract_slice %producer[4] [4] [1]
        : tensor<8xf16> to tensor<4xf16>
    %high_out = tensor.empty() : tensor<4xf16>
    %high = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%high_slice : tensor<4xf16>) outs(%high_out : tensor<4xf16>) {
    ^bb0(%value: f16, %old: f16):
      %difference = arith.subf %value, %value : f16
      linalg.yield %difference : f16
    } -> tensor<4xf16>
    return %whole, %low, %high
        : tensor<8xf16>, tensor<4xf16>, tensor<4xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeMatmulProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<2x3xf16>, %rhs: tensor<3x2xf16>)
      -> tensor<2x2xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<2x2xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<2x2xf16>) -> tensor<2x2xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<2x3xf16>, tensor<3x2xf16>)
        outs(%init : tensor<2x2xf16>) -> tensor<2x2xf16>
    return %result : tensor<2x2xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeDiamondFaninTailProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%left: tensor<10xf16>, %right: tensor<10xf16>)
      -> tensor<10xf16> {
    %producer_out = tensor.empty() : tensor<10xf16>
    %producer = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%left : tensor<10xf16>) outs(%producer_out : tensor<10xf16>) {
    ^bb0(%value: f16, %old: f16):
      %negated = arith.negf %value : f16
      linalg.yield %negated : f16
    } -> tensor<10xf16>
    %upper_out = tensor.empty() : tensor<10xf16>
    %upper = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer : tensor<10xf16>) outs(%upper_out : tensor<10xf16>) {
    ^bb0(%value: f16, %old: f16):
      %square = arith.mulf %value, %value : f16
      linalg.yield %square : f16
    } -> tensor<10xf16>
    %lower_out = tensor.empty() : tensor<10xf16>
    %lower = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%producer, %right : tensor<10xf16>, tensor<10xf16>)
      outs(%lower_out : tensor<10xf16>) {
    ^bb0(%first: f16, %second: f16, %old: f16):
      %sum = arith.addf %first, %second : f16
      linalg.yield %sum : f16
    } -> tensor<10xf16>
    %result_out = tensor.empty() : tensor<10xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
      ins(%upper, %lower : tensor<10xf16>, tensor<10xf16>)
      outs(%result_out : tensor<10xf16>) {
    ^bb0(%first: f16, %second: f16, %old: f16):
      %difference = arith.subf %first, %second : f16
      linalg.yield %difference : f16
    } -> tensor<10xf16>
    return %result : tensor<10xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeSharedInputContractionProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<4x8xf16>, %first_rhs: tensor<8x6xf16>,
                  %second_rhs: tensor<8x6xf16>)
      -> (tensor<4x6xf16>, tensor<4x6xf16>) {
    %zero = arith.constant 0.0 : f16
    %first_out = tensor.empty() : tensor<4x6xf16>
    %first_init = linalg.fill ins(%zero : f16)
        outs(%first_out : tensor<4x6xf16>) -> tensor<4x6xf16>
    %first = linalg.matmul
        ins(%lhs, %first_rhs : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%first_init : tensor<4x6xf16>) -> tensor<4x6xf16>
    %second_out = tensor.empty() : tensor<4x6xf16>
    %second_init = linalg.fill ins(%zero : f16)
        outs(%second_out : tensor<4x6xf16>) -> tensor<4x6xf16>
    %second = linalg.matmul
        ins(%lhs, %second_rhs : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%second_init : tensor<4x6xf16>) -> tensor<4x6xf16>
    return %first, %second : tensor<4x6xf16>, tensor<4x6xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeTransposedWeightContractionProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%activation: tensor<4x8xf16>,
                  %weight: tensor<6x8xf16>) -> tensor<4x6xf16> {
    %transpose_out = tensor.empty() : tensor<8x6xf16>
    %transpose = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%weight : tensor<6x8xf16>)
      outs(%transpose_out : tensor<8x6xf16>) {
    ^bb0(%value: f16, %old: f16):
      linalg.yield %value : f16
    } -> tensor<8x6xf16>
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<4x6xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<4x6xf16>) -> tensor<4x6xf16>
    %result = linalg.matmul
        ins(%activation, %transpose : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%init : tensor<4x6xf16>) -> tensor<4x6xf16>
    return %result : tensor<4x6xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeInvariantMatmulProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%lhs: tensor<4x8xf16>, %rhs: tensor<8x6xf16>)
      -> tensor<4x6xf16> {
    %zero = arith.constant 0.0 : f16
    %out = tensor.empty() : tensor<4x6xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%out : tensor<4x6xf16>) -> tensor<4x6xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x6xf16>)
        outs(%init : tensor<4x6xf16>) -> tensor<4x6xf16>
    return %result : tensor<4x6xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp>
  makeRepeatedProducerOperandContractionProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: tensor<4x4xf16>, %first_weight: tensor<4x4xf16>,
                  %second_weight: tensor<4x4xf16>) -> tensor<4x4xf16> {
    %zero = arith.constant 0.0 : f16
    %first_out = tensor.empty() : tensor<4x4xf16>
    %first_init = linalg.fill ins(%zero : f16)
        outs(%first_out : tensor<4x4xf16>) -> tensor<4x4xf16>
    %first = linalg.matmul
        ins(%input, %first_weight : tensor<4x4xf16>, tensor<4x4xf16>)
        outs(%first_init : tensor<4x4xf16>) -> tensor<4x4xf16>
    %pointwise_out = tensor.empty() : tensor<4x4xf16>
    %pointwise = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%first, %first : tensor<4x4xf16>, tensor<4x4xf16>)
      outs(%pointwise_out : tensor<4x4xf16>) {
    ^bb0(%left: f16, %right: f16, %old: f16):
      %sum = arith.addf %left, %right : f16
      linalg.yield %sum : f16
    } -> tensor<4x4xf16>
    %second_out = tensor.empty() : tensor<4x4xf16>
    %second_init = linalg.fill ins(%zero : f16)
        outs(%second_out : tensor<4x4xf16>) -> tensor<4x4xf16>
    %result = linalg.matmul
        ins(%pointwise, %second_weight : tensor<4x4xf16>, tensor<4x4xf16>)
        outs(%second_init : tensor<4x4xf16>) -> tensor<4x4xf16>
    return %result : tensor<4x4xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeAllReduceProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%input: tensor<8xf16>) -> tensor<8xf16> {
    %out = tensor.empty() : tensor<8xf16>
    %result = wafer.linalg_ext.collective.all_reduce
        ins(%input : tensor<8xf16>) outs(%out : tensor<8xf16>) {
    ^bb0(%left: f16, %right: f16):
      %sum = arith.addf %left, %right : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 17 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<8xf16>
    return %result : tensor<8xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeLocalReductionAllReduceProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(%input: tensor<4x8xf16>) -> tensor<4xf16> {
    %zero = arith.constant 0.0 : f16
    %local_out = tensor.empty() : tensor<4xf16>
    %local_init = linalg.fill ins(%zero : f16)
        outs(%local_out : tensor<4xf16>) -> tensor<4xf16>
    %local = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
      ins(%input : tensor<4x8xf16>) outs(%local_init : tensor<4xf16>) {
    ^bb0(%value: f16, %accumulator: f16):
      %sum = arith.addf %value, %accumulator : f16
      linalg.yield %sum : f16
    } -> tensor<4xf16>
    %global_out = tensor.empty() : tensor<4xf16>
    %global = wafer.linalg_ext.collective.all_reduce
        ins(%local : tensor<4xf16>) outs(%global_out : tensor<4xf16>) {
    ^bb0(%left: f16, %right: f16):
      %sum = arith.addf %left, %right : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {channel_id = 29 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<4xf16>
    return %global : tensor<4xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeStructuredConcatProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%first: tensor<1x2x2x4xf16>,
                  %second: tensor<1x2x2x4xf16>)
      -> tensor<1x2x2x8xf16> {
    %c4 = arith.constant 4 : index
    %out = tensor.empty() : tensor<1x2x2x8xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3)
                                     -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } outs(%out : tensor<1x2x2x8xf16>) {
    ^bb0(%old: f16):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %i2 = linalg.index 2 : index
      %i3 = linalg.index 3 : index
      %in_first = arith.cmpi ult, %i3, %c4 : index
      %value = scf.if %in_first -> f16 {
        %first_value = tensor.extract %first[%i0, %i1, %i2, %i3]
            : tensor<1x2x2x4xf16>
        scf.yield %first_value : f16
      } else {
        %second_i3 = arith.subi %i3, %c4 : index
        %second_value = tensor.extract %second[%i0, %i1, %i2, %second_i3]
            : tensor<1x2x2x4xf16>
        scf.yield %second_value : f16
      }
      linalg.yield %value : f16
    } -> tensor<1x2x2x8xf16>
    return %result : tensor<1x2x2x8xf16>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::OwningOpRef<mlir::ModuleOp> makeConstantConditionalTileProgram() {
    return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(%input: memref<4xf16, #wafer.memory<ddr, tensor>>)
      -> memref<4xf16, #wafer.memory<ddr, tensor>> {
    %output = memref.alloc()
        : memref<4xf16, #wafer.memory<ddr, tensor>>
    %result = wafer.tile.region(%input, %output
        : memref<4xf16, #wafer.memory<ddr, tensor>>,
          memref<4xf16, #wafer.memory<ddr, tensor>>) ->
        (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%in: memref<4xf16, #wafer.memory<ddr, tensor>>,
         %out: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %true = arith.constant true
      scf.if %true {
        %live = memref.alloc()
            : memref<4xf16, #wafer.memory<spm, tensor>>
        wafer.tile.load %in into %live
            : memref<4xf16, #wafer.memory<ddr, tensor>>
              into memref<4xf16, #wafer.memory<spm, tensor>>
      } else {
        %dead0 = memref.alloc()
            : memref<4xf16, #wafer.memory<spm, tensor>>
        wafer.tile.load %in into %dead0
            : memref<4xf16, #wafer.memory<ddr, tensor>>
              into memref<4xf16, #wafer.memory<spm, tensor>>
        %dead1 = memref.alloc()
            : memref<4xf16, #wafer.memory<spm, tensor>>
        wafer.tile.load %in into %dead1
            : memref<4xf16, #wafer.memory<ddr, tensor>>
              into memref<4xf16, #wafer.memory<spm, tensor>>
      }
      wafer.tile.yield %out
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return %result : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
}
)mlir",
                                                   context.get());
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CoordinatedDataflowSearchTest,
       ReservesGenerationFinalizationAndRepairBeforeWork) {
  auto finalization = wafer::compiler::detail::CoordinatedWorkLedger::
      getExecutableFinalizationUpperBound(/*rankCount=*/4);
  auto action = wafer::compiler::detail::CoordinatedWorkLedger::
      getExecutableScheduleAttemptUpperBound(/*rankCount=*/4);
  ASSERT_TRUE(mlir::succeeded(finalization));
  ASSERT_TRUE(mlir::succeeded(action));
  ASSERT_TRUE(finalization->getTotal());
  constexpr uint64_t rankCount = 4;
  static_assert(
      wafer::compiler::detail::kMaximumCoordinatedExactScheduleActions == 8,
      "the exact-success frontier must remain independently bounded");
  EXPECT_EQ(
      finalization->get(
          wafer::compiler::detail::CoordinatedWorkKind::TileToInstrLowering),
      rankCount);
  EXPECT_EQ(finalization->get(wafer::compiler::detail::CoordinatedWorkKind::
                                  ExecutableScheduleAction),
            rankCount);
  EXPECT_EQ(
      finalization->get(
          wafer::compiler::detail::CoordinatedWorkKind::SPMAllocationProblem),
      rankCount);
  EXPECT_EQ(
      finalization->get(
          wafer::compiler::detail::CoordinatedWorkKind::DDRAllocationDomain),
      rankCount);
  EXPECT_EQ(
      finalization->get(
          wafer::compiler::detail::CoordinatedWorkKind::TransportValidation),
      1u);
  EXPECT_EQ(finalization->get(
                wafer::compiler::detail::CoordinatedWorkKind::ABIValidation),
            rankCount);
  EXPECT_EQ(
      action->get(
          wafer::compiler::detail::CoordinatedWorkKind::TileToInstrLowering),
      0u);
  for (auto kind :
       {wafer::compiler::detail::CoordinatedWorkKind::ExecutableScheduleAction,
        wafer::compiler::detail::CoordinatedWorkKind::SPMAllocationProblem,
        wafer::compiler::detail::CoordinatedWorkKind::DDRAllocationDomain,
        wafer::compiler::detail::CoordinatedWorkKind::ABIValidation})
    EXPECT_EQ(action->get(kind), rankCount);
  EXPECT_EQ(
      action->get(
          wafer::compiler::detail::CoordinatedWorkKind::TransportValidation),
      1u);
  constexpr uint64_t repair = 5;
  constexpr uint64_t mandatoryGeneration = 8;
  uint64_t capacity =
      mandatoryGeneration + *finalization->getTotal() + repair + 2;
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      /*rankCount=*/4, capacity, repair);
  ASSERT_TRUE(mlir::succeeded(ledger));
  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.mandatoryGenerationReserved, mandatoryGeneration);
  EXPECT_EQ(snapshot.finalizationReserved, *finalization->getTotal());
  EXPECT_EQ(snapshot.scheduleAttemptCapacity, 16u);
  EXPECT_EQ(snapshot.scheduleAttemptsReserved, 1u);
  EXPECT_EQ(snapshot.scheduleAttemptsConsumed, 0u);
  EXPECT_EQ(snapshot.repairReserved, repair);
  EXPECT_EQ(snapshot.unreserved, 2u);
  EXPECT_TRUE(ledger->tryConsumeGeneration(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 2));
  EXPECT_FALSE(ledger->tryConsumeGeneration(
      wafer::compiler::detail::CoordinatedWorkKind::StructuredExpansion, 1));
  snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.finalizationReserved, *finalization->getTotal());
  EXPECT_EQ(snapshot.repairReserved, repair);
}

TEST_F(CoordinatedDataflowSearchTest,
       InvocationLedgerSharesSixteenAttemptsAcrossSeedAndExpansion) {
  using wafer::compiler::detail::CoordinatedWorkKind;
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedWorkEstimate generation;
  generation.set(CoordinatedWorkKind::StructuredExpansion, 1);
  generation.set(CoordinatedWorkKind::ActualTileClone, 1);
  ASSERT_TRUE(
      mlir::succeeded(ledger->completeMandatoryBaselineGeneration(generation)));

  auto completeSeed =
      [&](wafer::compiler::detail::ExecutableFinalizationReservation
              reservation) {
        wafer::compiler::detail::CoordinatedWorkEstimate actual;
        actual.set(CoordinatedWorkKind::TileToInstrLowering, 1);
        actual.set(CoordinatedWorkKind::ExecutableScheduleAction, 1);
        return ledger->completeExecutableFinalization(reservation, actual);
      };
  auto completeExpansion =
      [&](wafer::compiler::detail::ExecutableFinalizationReservation
              reservation) {
        // A recoverable action rejection still consumes its one global attempt.
        wafer::compiler::detail::CoordinatedWorkEstimate actual;
        actual.set(CoordinatedWorkKind::ExecutableScheduleAction, 1);
        return ledger->completeExecutableFinalization(reservation, actual);
      };

  ASSERT_TRUE(
      mlir::succeeded(completeSeed(ledger->getMandatoryBaselineReservation())));
  auto seedUpperBound = wafer::compiler::detail::CoordinatedWorkLedger::
      getExecutableFinalizationUpperBound(1);
  ASSERT_TRUE(mlir::succeeded(seedUpperBound));
  for (unsigned index = 0; index < 7; ++index) {
    auto reservation =
        ledger->tryReserveExecutableFinalization(*seedUpperBound);
    ASSERT_TRUE(reservation);
    ASSERT_TRUE(mlir::succeeded(completeSeed(*reservation)));
  }
  for (unsigned index = 0; index < 8; ++index) {
    auto reservation = ledger->tryReserveExecutableScheduleAttempt();
    ASSERT_TRUE(reservation);
    ASSERT_TRUE(mlir::succeeded(completeExpansion(*reservation)));
  }

  const auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.scheduleAttemptCapacity, 16u);
  EXPECT_EQ(snapshot.scheduleAttemptsConsumed, 16u);
  EXPECT_EQ(snapshot.scheduleAttemptsReserved, 0u);
  EXPECT_EQ(snapshot.finalizationReserved, 0u);
  EXPECT_FALSE(ledger->tryReserveExecutableScheduleAttempt());
  EXPECT_FALSE(ledger->tryReserveExecutableFinalization(*seedUpperBound));
}

TEST_F(CoordinatedDataflowSearchTest,
       OwnsAllRanksAsActualTileClonesUnderOneReservation) {
  auto source = makeProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      /*rankCount=*/4);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 4;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);
  const auto &baseline = frontier->front();
  EXPECT_TRUE(baseline.reservedBaseline);
  EXPECT_EQ(baseline.finalizationReservation.id,
            ledger->getMandatoryBaselineReservation().id);
  llvm::SmallSet<uint64_t, 8> reservations;
  unsigned baselineCount = 0;
  for (const auto &variant : *frontier) {
    baselineCount += variant.reservedBaseline;
    EXPECT_TRUE(reservations.insert(variant.finalizationReservation.id).second);
    ASSERT_EQ(variant.ranks.size(), 4u);
    for (auto [logicalRank, rank] : llvm::enumerate(variant.ranks)) {
      EXPECT_EQ(rank.logicalRank, static_cast<int64_t>(logicalRank));
      EXPECT_TRUE(wafer::containsTileDataflowOperations(
          rank.module.get().getOperation()));
      EXPECT_FALSE(rank.selectedTileIR);
      bool hasInstruction = false;
      rank.module.get().walk([&](mlir::Operation *operation) {
        hasInstruction |=
            mlir::isa<wafer::WaferInstructionOpInterface, wafer::SyncNCCJoinOp>(
                operation);
      });
      EXPECT_FALSE(hasInstruction);
    }
    EXPECT_NE(variant.ranks[0].module.get(), variant.ranks[1].module.get());
    EXPECT_EQ(variant.frontierDigest,
              wafer::compiler::detail::computeCoordinatedTileFrontierDigest(
                  *frontier));
  }
  EXPECT_EQ(baselineCount, 1u);
  auto snapshot = ledger->getSnapshot();
  EXPECT_EQ(snapshot.mandatoryGenerationReserved, 0u);
  EXPECT_GT(snapshot.finalizationReserved, 0u);
}

TEST_F(CoordinatedDataflowSearchTest,
       SessionPullsOnlyBaselineInReservedModeAndLeavesSourceImmutable) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  std::string sourceBefore;
  llvm::raw_string_ostream(sourceBefore) << *source;

  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      /*rankCount=*/2);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 2;
  config.reservedBaselineOnly = true;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto session =
      wafer::compiler::detail::CoordinatedDataflowSearchSession::create(
          *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(session));
  EXPECT_EQ(stats.structuralProposalsDerived, 1u);

  auto baseline = (*session)->admitNextActualCandidate();
  ASSERT_TRUE(mlir::succeeded(baseline));
  ASSERT_TRUE(*baseline);
  EXPECT_TRUE((*baseline)->reservedBaseline);
  EXPECT_EQ((*baseline)->ranks.size(), 2u);
  EXPECT_FALSE((*session)->getActualAdmissionDigest().empty());
  EXPECT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
      (*baseline)->stableSemanticOrdinal,
      wafer::compiler::detail::CoordinatedActualCandidateDisposition::
          RetainedForCompatibility)));

  auto exhausted = (*session)->admitNextActualCandidate();
  ASSERT_TRUE(mlir::succeeded(exhausted));
  EXPECT_FALSE(*exhausted);
  EXPECT_TRUE((*session)->exhausted());
  EXPECT_EQ(stats.actualCandidateAttempts, 1u);
  EXPECT_EQ(stats.actualCandidateAdmissions, 1u);
  EXPECT_EQ(stats.successfulActualCandidates, 1u);
  EXPECT_EQ(stats.peakLiveActualCandidates, 1u);
  EXPECT_EQ(stats.structuralPendingAtStop, 0u);

  std::string sourceAfter;
  llvm::raw_string_ostream(sourceAfter) << *source;
  EXPECT_EQ(sourceBefore, sourceAfter);
}

TEST_F(CoordinatedDataflowSearchTest,
       SessionBackfillsStableNextActualCandidateAfterExactFailure) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      /*rankCount=*/1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  config.maximumSuccessfulActualCandidates = 2;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto session =
      wafer::compiler::detail::CoordinatedDataflowSearchSession::create(
          *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(session));

  auto closeExactWork = [&](const auto &candidate) {
    wafer::compiler::detail::CoordinatedWorkEstimate actual;
    actual.set(
        wafer::compiler::detail::CoordinatedWorkKind::TileToInstrLowering, 1);
    actual.set(
        wafer::compiler::detail::CoordinatedWorkKind::ExecutableScheduleAction,
        1);
    return ledger->completeExecutableFinalization(
        candidate.finalizationReservation, actual);
  };

  auto baseline = (*session)->admitNextActualCandidate();
  ASSERT_TRUE(mlir::succeeded(baseline));
  ASSERT_TRUE(*baseline);
  ASSERT_TRUE(mlir::succeeded(closeExactWork(**baseline)));
  ASSERT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
      (*baseline)->stableSemanticOrdinal,
      wafer::compiler::detail::CoordinatedActualCandidateDisposition::
          ExactAccepted)));

  auto rejected = (*session)->admitNextActualCandidate();
  ASSERT_TRUE(mlir::succeeded(rejected));
  ASSERT_TRUE(*rejected);
  ASSERT_FALSE((*rejected)->reservedBaseline);
  const int64_t rejectedOrdinal = (*rejected)->stableSemanticOrdinal;
  ASSERT_TRUE(mlir::succeeded(closeExactWork(**rejected)));
  ASSERT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
      rejectedOrdinal,
      wafer::compiler::detail::CoordinatedActualCandidateDisposition::
          ExactRejected,
      "test-exact-gate")));

  auto backfill = (*session)->admitNextActualCandidate();
  ASSERT_TRUE(mlir::succeeded(backfill));
  ASSERT_TRUE(*backfill);
  EXPECT_GT((*backfill)->stableSemanticOrdinal, rejectedOrdinal);
  EXPECT_EQ(stats.exactFailureBackfills, 1u);
  EXPECT_EQ(stats.peakLiveActualCandidates, 1u);
  EXPECT_LE(stats.successfulActualCandidates,
            config.maximumSuccessfulActualCandidates);
  ASSERT_TRUE(mlir::succeeded(closeExactWork(**backfill)));
  EXPECT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
      (*backfill)->stableSemanticOrdinal,
      wafer::compiler::detail::CoordinatedActualCandidateDisposition::
          ExactAccepted)));
  EXPECT_EQ(ledger->getSnapshot().scheduleAttemptsConsumed, 3u);
}

TEST_F(CoordinatedDataflowSearchTest,
       OneProviderPointCompetesThroughProductionPhysicalActionsOnDemand) {
  auto source = makeProgram();
  ASSERT_TRUE(source);
  std::string sourceBefore;
  {
    llvm::raw_string_ostream stream(sourceBefore);
    source->print(stream);
  }

  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      /*rankCount=*/1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  auto materializationCount = std::make_shared<uint64_t>(0);
  CountingImplementationProvider provider(materializationCount);
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  config.optimizations = wafer::OptimizationConfig::production();
  config.implementationAlternativeProviders.push_back(&provider);
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto session =
      wafer::compiler::detail::CoordinatedDataflowSearchSession::create(
          *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(session));
  EXPECT_EQ(stats.implementationAlternativeQueries, 1u);
  EXPECT_EQ(stats.implementationAlternativeProposals, 3u);
  EXPECT_EQ(stats.implementationAlternativeMaterializations, 0u);
  EXPECT_EQ(*materializationCount, 0u);
  EXPECT_EQ(stats.actualCandidateAttempts, 0u);

  while (*materializationCount < 2) {
    auto next = (*session)->admitNextActualCandidate();
    ASSERT_TRUE(mlir::succeeded(next));
    ASSERT_TRUE(*next);
    ASSERT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
        (*next)->stableSemanticOrdinal,
        wafer::compiler::detail::CoordinatedActualCandidateDisposition::
            RetainedForCompatibility)));
  }
  EXPECT_GE(stats.implementationAlternativeMaterializations, 2u);
  EXPECT_EQ(*materializationCount, 2u);
  EXPECT_LE(stats.peakLiveActualCandidates, 1u);
  EXPECT_LE(stats.successfulActualCandidates,
            config.maximumSuccessfulActualCandidates);

  std::string sourceAfter;
  {
    llvm::raw_string_ostream stream(sourceAfter);
    source->print(stream);
  }
  EXPECT_EQ(sourceBefore, sourceAfter);
}

TEST_F(CoordinatedDataflowSearchTest,
       ManyProviderPointsCoverIdentitiesBeforePhysicalSiblings) {
  struct Result {
    std::vector<uint64_t> materializedPointOrdinals;
    wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
    std::string structuralDigest;
  };
  auto run = [&](bool reverseProviderOrder,
                 int64_t candidateParallelism) -> std::optional<Result> {
    auto source = makeProgram();
    if (!source)
      return std::nullopt;
    auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
    if (mlir::failed(ledger))
      return std::nullopt;
    auto trace = std::make_shared<ManyPointMaterializationTrace>();
    ManyPointImplementationProvider provider(reverseProviderOrder, trace);
    wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
    config.rankCount = 1;
    config.candidateParallelism = candidateParallelism;
    config.implementationAlternativeProviders.push_back(&provider);
    wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
    auto session =
        wafer::compiler::detail::CoordinatedDataflowSearchSession::create(
            *source, config, *ledger, &stats);
    if (mlir::failed(session))
      return std::nullopt;
    std::string structuralDigest =
        (*session)->getStructuralFrontierDigest().str();
    while (true) {
      auto next = (*session)->admitNextActualCandidate();
      if (mlir::failed(next))
        return std::nullopt;
      if (!*next)
        break;
      const bool baseline = (*next)->reservedBaseline;
      if (mlir::failed((*session)->completeActiveCandidate(
              (*next)->stableSemanticOrdinal,
              baseline
                  ? wafer::compiler::detail::
                        CoordinatedActualCandidateDisposition::
                            RetainedForCompatibility
                  : wafer::compiler::detail::
                        CoordinatedActualCandidateDisposition::ExactRejected)))
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(trace->mutex);
    return Result{trace->pointOrdinals, stats, std::move(structuralDigest)};
  };

  std::optional<Result> forwardSerial =
      run(/*reverseProviderOrder=*/false, /*candidateParallelism=*/1);
  std::optional<Result> reverseSerial =
      run(/*reverseProviderOrder=*/true, /*candidateParallelism=*/1);
  std::optional<Result> forwardParallel =
      run(/*reverseProviderOrder=*/false, /*candidateParallelism=*/8);
  ASSERT_TRUE(forwardSerial);
  ASSERT_TRUE(reverseSerial);
  ASSERT_TRUE(forwardParallel);

  auto verifyCoverage = [&](const Result &result) {
    EXPECT_EQ(result.stats.implementationAlternativeProposals, 48u * 3u);
    EXPECT_EQ(result.stats.implementationAlternativeMaterializations,
              result.materializedPointOrdinals.size());
    EXPECT_GT(result.stats.structuralCoverageBeamPruned, 0u);
    llvm::SmallSet<uint64_t, 32> uniquePoints;
    std::array<uint64_t, 48> materializationsPerPoint = {};
    for (uint64_t pointOrdinal : result.materializedPointOrdinals) {
      ASSERT_LT(pointOrdinal, materializationsPerPoint.size());
      uniquePoints.insert(pointOrdinal);
      ++materializationsPerPoint[pointOrdinal];
    }
    // Identity actions for distinct parameter points consume the bulk of the
    // bounded provider share; repeated residency siblings cannot reduce the
    // beam to one third as many distinct points.
    EXPECT_GT(uniquePoints.size() * 2, result.materializedPointOrdinals.size());
    // Coverage selection still keeps all three typed residency actions for at
    // least one representative point.
    EXPECT_EQ(*llvm::max_element(materializationsPerPoint), 3u);
  };
  verifyCoverage(*forwardSerial);
  verifyCoverage(*reverseSerial);
  verifyCoverage(*forwardParallel);
  EXPECT_EQ(forwardSerial->materializedPointOrdinals,
            reverseSerial->materializedPointOrdinals);
  EXPECT_EQ(forwardSerial->materializedPointOrdinals,
            forwardParallel->materializedPointOrdinals);
  EXPECT_EQ(forwardSerial->structuralDigest, reverseSerial->structuralDigest);
  EXPECT_EQ(forwardSerial->structuralDigest, forwardParallel->structuralDigest);
}

TEST_F(CoordinatedDataflowSearchTest,
       ProviderAdmissionCoversResidencyBeforeParallelPartitionDetails) {
  auto source = makeProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  auto trace = std::make_shared<ParallelismCoverageMaterializationTrace>();
  ParallelismCoverageImplementationProvider provider(trace);
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  config.implementationAlternativeProviders.push_back(&provider);
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto session =
      wafer::compiler::detail::CoordinatedDataflowSearchSession::create(
          *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(session));

  while (true) {
    auto next = (*session)->admitNextActualCandidate();
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    const bool baseline = (*next)->reservedBaseline;
    ASSERT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
        (*next)->stableSemanticOrdinal,
        baseline
            ? wafer::compiler::detail::CoordinatedActualCandidateDisposition::
                  RetainedForCompatibility
            : wafer::compiler::detail::CoordinatedActualCandidateDisposition::
                  ExactRejected)));
  }

  std::vector<int64_t> partitionCounts;
  {
    std::lock_guard<std::mutex> lock(trace->mutex);
    partitionCounts = trace->partitionCounts;
  }
  EXPECT_EQ(stats.implementationAlternativeProposals, 5u * 3u);
  EXPECT_EQ(stats.implementationAlternativeMaterializations,
            partitionCounts.size());
  ASSERT_GE(partitionCounts.size(), 8u);
  const std::array<int64_t, 8> expectedFirstEight = {1, 2, 1, 2, 1, 2, 4, 8};
  EXPECT_TRUE(llvm::equal(llvm::ArrayRef(partitionCounts).take_front(8),
                          expectedFirstEight));
}

TEST_F(CoordinatedDataflowSearchTest,
       ProviderAdmissionLeavesHeadroomForCompleteActualCandidate) {
  auto source = makeProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  auto trace = std::make_shared<TileAxisCoverageMaterializationTrace>();
  ResourceHeadroomImplementationProvider provider(trace);
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  config.implementationAlternativeProviders.push_back(&provider);
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto session =
      wafer::compiler::detail::CoordinatedDataflowSearchSession::create(
          *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(session));

  while (true) {
    auto next = (*session)->admitNextActualCandidate();
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    const bool baseline = (*next)->reservedBaseline;
    ASSERT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
        (*next)->stableSemanticOrdinal,
        baseline
            ? wafer::compiler::detail::CoordinatedActualCandidateDisposition::
                  RetainedForCompatibility
            : wafer::compiler::detail::CoordinatedActualCandidateDisposition::
                  ExactRejected)));
  }

  std::vector<std::array<int64_t, 2>> materialized;
  {
    std::lock_guard<std::mutex> lock(trace->mutex);
    materialized = trace->tileSizePairs;
  }
  EXPECT_EQ(stats.implementationAlternativeProposals, 2u * 3u);
  ASSERT_GE(materialized.size(), 3u);
  for (const std::array<int64_t, 2> &point :
       llvm::ArrayRef(materialized).take_front(3))
    EXPECT_EQ(point, (std::array<int64_t, 2>{1, 1}));
  EXPECT_TRUE(llvm::is_contained(materialized, std::array<int64_t, 2>{2, 2}));
}

TEST_F(CoordinatedDataflowSearchTest,
       ProviderPointCapPreservesOutputAndReductionTileAxisCoverage) {
  auto source = makeProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  auto trace = std::make_shared<TileAxisCoverageMaterializationTrace>();
  TileAxisCoverageImplementationProvider provider(trace);
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  config.implementationAlternativeProviders.push_back(&provider);
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto session =
      wafer::compiler::detail::CoordinatedDataflowSearchSession::create(
          *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(session));

  while (true) {
    auto next = (*session)->admitNextActualCandidate();
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    const bool baseline = (*next)->reservedBaseline;
    ASSERT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
        (*next)->stableSemanticOrdinal,
        baseline
            ? wafer::compiler::detail::CoordinatedActualCandidateDisposition::
                  RetainedForCompatibility
            : wafer::compiler::detail::CoordinatedActualCandidateDisposition::
                  ExactRejected)));
  }

  EXPECT_EQ(
      stats.implementationAlternativeProposals,
      wafer::compiler::detail::kMaximumCoordinatedStructuralFrontierStates *
          3u);
  EXPECT_GT(stats.structuralCoverageBeamPruned, 0u);
  std::vector<std::array<int64_t, 2>> materializedTileSizePairs;
  {
    std::lock_guard<std::mutex> lock(trace->mutex);
    materializedTileSizePairs = trace->tileSizePairs;
  }
  EXPECT_TRUE(llvm::is_contained(materializedTileSizePairs,
                                 std::array<int64_t, 2>{1, 2}))
      << "the sole output-tile=1 representative was starved";
  EXPECT_TRUE(llvm::is_contained(materializedTileSizePairs,
                                 std::array<int64_t, 2>{2, 1}))
      << "the sole reduction-tile=1 representative was starved";
}

TEST_F(CoordinatedDataflowSearchTest,
       MultiResultDecodeProviderUsesDomainFirstSeedsAndMaterializesOnPull) {
  auto source = makeDecodeAttentionProgram();
  ASSERT_TRUE(source);
  mlir::func::FuncOp decode =
      source->lookupSymbol<mlir::func::FuncOp>("functional_decode");
  ASSERT_TRUE(decode);
  ASSERT_EQ(decode.getNumArguments(), 8u);
  mlir::OpBuilder builder(decode.getBody());
  builder.setInsertionPointToStart(&decode.front());
  for (unsigned argumentIndex = 5; argumentIndex < 8; ++argumentIndex) {
    auto type = mlir::cast<mlir::RankedTensorType>(
        decode.getArgument(argumentIndex).getType());
    mlir::Value destination = builder.create<mlir::tensor::EmptyOp>(
        decode.getLoc(), type.getShape(), type.getElementType());
    decode.getArgument(argumentIndex).replaceAllUsesWith(destination);
  }
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      /*rankCount=*/1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::AttentionImplementationAlternativeProvider provider;
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  config.optimizations = wafer::OptimizationConfig::production();
  config.implementationAlternativeProviders.push_back(&provider);
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto session =
      wafer::compiler::detail::CoordinatedDataflowSearchSession::create(
          *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(session));
  EXPECT_EQ(stats.implementationAlternativeQueries, 1u);
  EXPECT_GT(stats.implementationAlternativeProposals, 0u);
  EXPECT_EQ(stats.implementationAlternativeMaterializations, 0u);

  while (stats.implementationAlternativeMaterializations == 0) {
    auto next = (*session)->admitNextActualCandidate();
    ASSERT_TRUE(mlir::succeeded(next));
    if (!*next)
      break;
    ASSERT_TRUE(mlir::succeeded((*session)->completeActiveCandidate(
        (*next)->stableSemanticOrdinal,
        wafer::compiler::detail::CoordinatedActualCandidateDisposition::
            RetainedForCompatibility)));
  }
  EXPECT_GT(stats.implementationAlternativeMaterializations, 0u);
  EXPECT_GT(stats.retainedImplementationAlternativeCandidates, 0u)
      << " attempts=" << stats.actualCandidateAttempts
      << " materialization_failures=" << stats.candidateMaterializationFailures
      << " equivalent=" << stats.equivalentCandidatesRejected
      << " dominated=" << stats.dominatedCandidatesRejected
      << " promotion=" << stats.promotionIneligibleCandidatesRejected;
  EXPECT_LE(stats.peakLiveActualCandidates, 1u);
}

TEST_F(CoordinatedDataflowSearchTest,
       CoupledTraversalRemovesTheIntermediateDDRRoundTrip) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);

  auto countTransfers = [](mlir::ModuleOp module) {
    unsigned count = 0;
    module.walk([&](mlir::Operation *operation) {
      count +=
          mlir::isa<wafer::StorageLoadOp, wafer::StorageStoreOp>(operation);
    });
    return count;
  };
  const unsigned baselineTransfers =
      countTransfers(*frontier->front().ranks.front().module);
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        return countTransfers(*variant.ranks.front().module) <
               baselineTransfers;
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       ActualResidencyActionsCoverFusedSeparatedSplitAndSelectiveSpill) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  auto materialize = [&](wafer::CompleteRankTraversalComposition composition,
                         wafer::CandidateTileResidencyAction residency) {
    std::string failureReason;
    auto candidate = wafer::materializeCompleteRankCandidateTileProgram(
        *source, /*logicalRank=*/0, /*candidateTileSizes=*/{8},
        /*candidateReductionTileSizes=*/{},
        wafer::CandidateTileTraversalKind::ResultDriven, composition, residency,
        wafer::CandidateBoundaryMovementAction::Staged,
        wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidate)) << failureReason;
    return candidate;
  };

  auto fused =
      materialize(wafer::CompleteRankTraversalComposition::Coupled,
                  wafer::CandidateTileResidencyAction::KeepSingleRegion);
  auto separated =
      materialize(wafer::CompleteRankTraversalComposition::Separated,
                  wafer::CandidateTileResidencyAction::KeepSingleRegion);
  auto split = materialize(
      wafer::CompleteRankTraversalComposition::Separated,
      wafer::CandidateTileResidencyAction::SplitAtExplicitDDRBoundary);
  auto selective =
      materialize(wafer::CompleteRankTraversalComposition::Coupled,
                  wafer::CandidateTileResidencyAction::SelectiveSpill);
  ASSERT_TRUE(mlir::succeeded(fused));
  ASSERT_TRUE(mlir::succeeded(separated));
  ASSERT_TRUE(mlir::succeeded(split));
  ASSERT_TRUE(mlir::succeeded(selective));

  auto count = [](mlir::ModuleOp module, auto tag) {
    unsigned result = 0;
    module.walk([&](decltype(tag)) { ++result; });
    return result;
  };
  EXPECT_EQ(count(**fused, wafer::TileRegionOp{}), 1u);
  EXPECT_EQ(count(**separated, wafer::TileRegionOp{}), 1u);
  EXPECT_EQ(count(**split, wafer::TileRegionOp{}), 2u);
  EXPECT_EQ(count(**selective, wafer::TileRegionOp{}), 1u);
  const auto transferCount = [&](mlir::ModuleOp module) {
    return count(module, wafer::StorageLoadOp{}) +
           count(module, wafer::StorageStoreOp{});
  };
  EXPECT_LT(transferCount(**fused), transferCount(**separated));
  EXPECT_GT(transferCount(**selective), transferCount(**fused));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**fused)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**separated)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**split)));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**selective)));

  (*split)->walk([&](wafer::TileRegionOp region) {
    for (mlir::Type type : region.getOperandTypes())
      if (mlir::isa<mlir::ShapedType>(type))
        EXPECT_TRUE(wafer::isWaferDDRMemRefType(type));
    for (mlir::Type type : region.getResultTypes())
      if (mlir::isa<mlir::ShapedType>(type))
        EXPECT_TRUE(wafer::isWaferDDRMemRefType(type));
  });

  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(llvm::any_of(*frontier, [&](const auto &variant) {
    return count(*variant.ranks.front().module, wafer::TileRegionOp{}) == 2;
  }));
  EXPECT_TRUE(llvm::any_of(*frontier, [&](const auto &variant) {
    mlir::ModuleOp module = *variant.ranks.front().module;
    return count(module, wafer::TileRegionOp{}) == 1 &&
           transferCount(module) > transferCount(**fused);
  }));
}

TEST_F(CoordinatedDataflowSearchTest,
       ConnectionActionsMaterializeActualChainAndMixedFanoutStates) {
  using Action = wafer::CandidateTraversalConnectionAction;
  auto chain = makeChainProgram();
  ASSERT_TRUE(chain);
  std::string failureReason;
  auto connectionCount =
      wafer::getCompleteRankCandidateConnectionCount(*chain, &failureReason);
  ASSERT_TRUE(mlir::succeeded(connectionCount)) << failureReason;
  ASSERT_EQ(*connectionCount, 1u);

  auto materializeChain = [&](Action action) {
    failureReason.clear();
    auto candidate = wafer::materializeCompleteRankConnectionTileProgram(
        *chain, /*logicalRank=*/0, /*candidateTileSizes=*/{8},
        llvm::ArrayRef<Action>{action},
        wafer::CandidateBoundaryMovementAction::Staged,
        wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidate)) << failureReason;
    return candidate;
  };
  auto coupled = materializeChain(Action::CoupledResident);
  auto resident = materializeChain(Action::SeparatedResident);
  auto staged = materializeChain(Action::SeparatedDDR);
  auto cut = materializeChain(Action::CrossRegion);
  auto selective = materializeChain(Action::SelectiveSpill);
  ASSERT_TRUE(mlir::succeeded(coupled));
  ASSERT_TRUE(mlir::succeeded(resident));
  ASSERT_TRUE(mlir::succeeded(staged));
  ASSERT_TRUE(mlir::succeeded(cut));
  ASSERT_TRUE(mlir::succeeded(selective));

  auto count = [](mlir::ModuleOp module, auto tag) {
    unsigned result = 0;
    module.walk([&](decltype(tag)) { ++result; });
    return result;
  };
  EXPECT_EQ(count(**coupled, wafer::TileRegionOp{}), 1u);
  EXPECT_EQ(count(**resident, wafer::TileRegionOp{}), 1u);
  EXPECT_EQ(count(**staged, wafer::TileRegionOp{}), 1u);
  EXPECT_EQ(count(**cut, wafer::TileRegionOp{}), 2u);
  EXPECT_EQ(count(**selective, wafer::TileRegionOp{}), 1u);
  EXPECT_GT(count(**staged, wafer::StorageStoreOp{}), 0u);
  EXPECT_GT(count(**staged, wafer::StorageLoadOp{}), 0u);
  EXPECT_GT(count(**selective, wafer::StorageStoreOp{}), 0u);
  EXPECT_GT(count(**selective, wafer::StorageLoadOp{}), 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**resident)));

  auto diamond = makeDiamondFaninTailProgram();
  ASSERT_TRUE(diamond);
  auto diamondConnections =
      wafer::getCompleteRankCandidateConnectionCount(*diamond, &failureReason);
  ASSERT_TRUE(mlir::succeeded(diamondConnections)) << failureReason;
  ASSERT_EQ(*diamondConnections, 4u);
  llvm::SmallVector<Action, 4> mixedActions = {
      Action::CoupledResident, Action::SeparatedResident,
      Action::CoupledResident, Action::CoupledResident};
  auto mixed = wafer::materializeCompleteRankConnectionTileProgram(
      *diamond, /*logicalRank=*/0, /*candidateTileSizes=*/{10}, mixedActions,
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
  ASSERT_TRUE(mlir::succeeded(mixed)) << failureReason;
  EXPECT_EQ(count(**mixed, wafer::TileRegionOp{}), 1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**mixed)));
}

TEST_F(CoordinatedDataflowSearchTest,
       ConnectionChoiceMaterializesIndependentProducerAndConsumerTiles) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  std::string failureReason;
  auto topology = wafer::getCompleteRankCandidateConnectionTopology(
      *source, &failureReason);
  ASSERT_TRUE(mlir::succeeded(topology)) << failureReason;
  ASSERT_EQ(topology->domains.size(), 1u);
  EXPECT_EQ(topology->domains.front().producerResultShape,
            (llvm::SmallVector<int64_t, 4>{8}));
  EXPECT_EQ(topology->domains.front().consumerResultShape,
            (llvm::SmallVector<int64_t, 4>{8}));
  EXPECT_EQ(topology->domains.front().producerResultElementBytes, 2u);
  EXPECT_EQ(topology->domains.front().consumerOperandElementBytes, 2u);
  EXPECT_EQ(topology->domains.front().consumerResultElementBytes, 2u);

  wafer::CandidateTraversalConnectionChoice choice;
  choice.action = wafer::CandidateTraversalConnectionAction::SeparatedDDR;
  choice.producerTileSizes = {4};
  choice.consumerTileSizes = {2};
  auto candidate = wafer::materializeCompleteRankConnectionChoicesTileProgram(
      *source, /*logicalRank=*/0,
      llvm::ArrayRef<wafer::CandidateTraversalConnectionChoice>{choice},
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;

  llvm::SmallVector<int64_t, 4> steps;
  (*candidate)->walk([&](mlir::scf::ForOp loop) {
    std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
    ASSERT_TRUE(step);
    steps.push_back(*step);
  });
  EXPECT_TRUE(llvm::is_contained(steps, 4));
  EXPECT_TRUE(llvm::is_contained(steps, 2));
  unsigned stores = 0;
  unsigned loads = 0;
  (*candidate)->walk([&](wafer::StorageStoreOp) { ++stores; });
  (*candidate)->walk([&](wafer::StorageLoadOp) { ++loads; });
  EXPECT_GT(stores, 0u);
  EXPECT_GT(loads, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**candidate)));

  choice.action = wafer::CandidateTraversalConnectionAction::CrossRegion;
  auto crossRegion = wafer::materializeCompleteRankConnectionChoicesTileProgram(
      *source, /*logicalRank=*/0,
      llvm::ArrayRef<wafer::CandidateTraversalConnectionChoice>{choice},
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
  ASSERT_TRUE(mlir::succeeded(crossRegion)) << failureReason;
  unsigned regions = 0;
  stores = 0;
  loads = 0;
  (*crossRegion)->walk([&](wafer::TileRegionOp) { ++regions; });
  (*crossRegion)->walk([&](wafer::StorageStoreOp) { ++stores; });
  (*crossRegion)->walk([&](wafer::StorageLoadOp) { ++loads; });
  EXPECT_EQ(regions, 2u);
  EXPECT_GT(stores, 0u);
  EXPECT_GT(loads, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**crossRegion)));
}

TEST_F(CoordinatedDataflowSearchTest,
       CoupledChoiceDerivesProducerDemandFromHeldOutTilingInterface) {
  auto source = makeTransposedWeightContractionProgram();
  ASSERT_TRUE(source);
  std::string failureReason;
  auto topology = wafer::getCompleteRankCandidateConnectionTopology(
      *source, &failureReason);
  ASSERT_TRUE(mlir::succeeded(topology)) << failureReason;
  ASSERT_EQ(topology->domains.size(), 1u);
  EXPECT_EQ(topology->domains.front().producerResultShape,
            (llvm::SmallVector<int64_t, 4>{8, 6}));
  EXPECT_EQ(topology->domains.front().consumerResultShape,
            (llvm::SmallVector<int64_t, 4>{4, 6}));

  wafer::CandidateTraversalConnectionChoice choice;
  choice.action = wafer::CandidateTraversalConnectionAction::CoupledResident;
  choice.consumerTileSizes = {2, 3};
  ASSERT_TRUE(choice.producerTileSizes.empty());
  auto candidate = wafer::materializeCompleteRankConnectionChoicesTileProgram(
      *source, /*logicalRank=*/0,
      llvm::ArrayRef<wafer::CandidateTraversalConnectionChoice>{choice},
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;

  bool sawConsumerTile = false;
  bool sawDifferentProducerDemand = false;
  (*candidate)->walk([&](mlir::Operation *operation) {
    auto inspect = [&](mlir::Type type) {
      auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
      if (!memref || !memref.hasStaticShape())
        return;
      sawConsumerTile |= memref.getRank() == 2 && memref.getDimSize(0) == 2 &&
                         memref.getDimSize(1) == 3;
      sawDifferentProducerDemand |= memref.getRank() == 2 &&
                                    memref.getDimSize(0) == 8 &&
                                    memref.getDimSize(1) == 3;
    };
    llvm::for_each(operation->getOperandTypes(), inspect);
    llvm::for_each(operation->getResultTypes(), inspect);
  });
  EXPECT_TRUE(sawConsumerTile);
  EXPECT_TRUE(sawDifferentProducerDemand);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**candidate)));
}

TEST_F(CoordinatedDataflowSearchTest,
       JointTraversalMapsPublicDestinationThroughExactOutputReshape) {
  using Action = wafer::CandidateTraversalConnectionAction;
  auto source = makeReshapedOutputChainProgram();
  ASSERT_TRUE(source);
  std::string failureReason;
  auto candidate = wafer::materializeCompleteRankConnectionTileProgram(
      *source, /*logicalRank=*/0, /*candidateTileSizes=*/{1, 64},
      llvm::ArrayRef<Action>{Action::CoupledResident},
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**candidate)));
}

TEST_F(CoordinatedDataflowSearchTest,
       RepairClonesTheLiveAllRankTileParentWithoutSourceReplay) {
  auto source = makeChainProgram();
  ASSERT_TRUE(source);
  std::string materializationFailure;
  auto rankZero = wafer::materializeCompleteRankCandidateTileProgram(
      *source, /*logicalRank=*/0, /*candidateTileSizes=*/{8},
      /*candidateReductionTileSizes=*/{},
      wafer::CandidateTileTraversalKind::ResultDriven,
      wafer::CompleteRankTraversalComposition::Coupled,
      wafer::CandidateTileResidencyAction::KeepSingleRegion,
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed,
      &materializationFailure);
  ASSERT_TRUE(mlir::succeeded(rankZero)) << materializationFailure;

  auto finalization = wafer::compiler::detail::CoordinatedWorkLedger::
      getExecutableFinalizationUpperBound(/*rankCount=*/2);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(2);
  ASSERT_TRUE(mlir::succeeded(finalization));
  ASSERT_TRUE(mlir::succeeded(ledger));
  auto parentReservation =
      ledger->tryReserveExecutableFinalization(*finalization);
  ASSERT_TRUE(parentReservation);

  wafer::compiler::detail::CoordinatedTileVariant parent;
  parent.stableSemanticOrdinal = 7;
  parent.finalizationReservation = *parentReservation;
  std::string selectedText;
  llvm::raw_string_ostream selectedStream(selectedText);
  (*rankZero)->print(selectedStream);
  selectedStream.flush();
  auto selected = std::make_shared<const std::string>(std::move(selectedText));
  auto rankOne = mlir::cast<mlir::ModuleOp>((*rankZero)->clone());
  parent.ranks.emplace_back(/*logicalRank=*/0, std::move(*rankZero), selected);
  parent.ranks.emplace_back(/*logicalRank=*/1, std::move(rankOne), selected);
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::verifyCoordinatedTileVariant(parent, 2)));
  const std::string parentDigest =
      wafer::compiler::detail::computeCoordinatedTileVariantContentDigest(
          parent);

  wafer::compiler::detail::CoordinatedWorkEstimate failedAttempt;
  failedAttempt.set(
      wafer::compiler::detail::CoordinatedWorkKind::TileToInstrLowering, 2);
  ASSERT_TRUE(mlir::succeeded(ledger->completeExecutableFinalization(
      parent.finalizationReservation, failedAttempt)));
  const auto beforeRepair = ledger->getSnapshot();

  std::string repairFailure;
  auto repaired = wafer::compiler::detail::materializeCoordinatedTileRepair(
      parent,
      wafer::compiler::detail::CoordinatedTileRepairAction::SelectiveSpill,
      /*stableSemanticOrdinal=*/8, *ledger, &repairFailure);
  ASSERT_TRUE(mlir::succeeded(repaired)) << repairFailure;
  EXPECT_EQ(repaired->repairDepth, 1u);
  EXPECT_FALSE(repaired->reservedBaseline);
  EXPECT_NE(repaired->finalizationReservation.id,
            parent.finalizationReservation.id);
  ASSERT_EQ(repaired->ranks.size(), 2u);
  EXPECT_EQ(wafer::compiler::detail::computeCoordinatedTileVariantContentDigest(
                parent),
            parentDigest);
  EXPECT_NE(wafer::compiler::detail::computeCoordinatedTileVariantContentDigest(
                *repaired),
            parentDigest);
  for (const auto &rank : repaired->ranks) {
    unsigned stores = 0;
    unsigned loads = 0;
    rank.module.get().walk([&](wafer::StorageStoreOp) { ++stores; });
    rank.module.get().walk([&](wafer::StorageLoadOp) { ++loads; });
    EXPECT_GT(stores, 0u);
    EXPECT_GT(loads, 0u);
  }
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::verifyCoordinatedTileVariant(*repaired, 2)));

  const auto afterRepair = ledger->getSnapshot();
  EXPECT_EQ(afterRepair.repairReserved + 3, beforeRepair.repairReserved);
  EXPECT_EQ(
      afterRepair.consumedByKind[static_cast<size_t>(
          wafer::compiler::detail::CoordinatedWorkKind::RepairExpansion)],
      beforeRepair.consumedByKind[static_cast<size_t>(
          wafer::compiler::detail::CoordinatedWorkKind::RepairExpansion)] +
          1);
  EXPECT_EQ(
      afterRepair.consumedByKind[static_cast<size_t>(
          wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone)],
      beforeRepair.consumedByKind[static_cast<size_t>(
          wafer::compiler::detail::CoordinatedWorkKind::ActualTileClone)] +
          2);

  auto nestedRepair = wafer::compiler::detail::materializeCoordinatedTileRepair(
      *repaired,
      wafer::compiler::detail::CoordinatedTileRepairAction::SelectiveSpill,
      /*stableSemanticOrdinal=*/9, *ledger, &repairFailure);
  EXPECT_TRUE(mlir::failed(nestedRepair));
  EXPECT_EQ(ledger->getSnapshot().repairReserved, afterRepair.repairReserved);
}

TEST_F(CoordinatedDataflowSearchTest,
       MixedShapeThreeConsumerFanoutMaterializesActualBoundedVersions) {
  auto source = makeMixedShapeFanoutProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);
  EXPECT_TRUE(stats.usedGeneralDAGBeam);
  EXPECT_EQ(stats.structuredConnections, 3u);
  EXPECT_GT(stats.connectionLocalChoicesPruned, 0u);
  EXPECT_GT(stats.connectionIncompatibleSharedVersionsPruned, 0u);
  EXPECT_LE(
      stats.maximumConnectionLocalChoices,
      wafer::compiler::detail::kMaximumCoordinatedConnectionExpansionsPerStep);
  EXPECT_LE(
      stats.maximumConnectionExpansionsPerParent,
      wafer::compiler::detail::kMaximumCoordinatedConnectionExpansionsPerStep);
  EXPECT_LE(
      stats.maximumConnectionStates,
      wafer::compiler::detail::kMaximumCoordinatedConnectionDAGBeamStates);

  auto countTransfers = [](mlir::ModuleOp module) {
    unsigned count = 0;
    module.walk([&](mlir::Operation *operation) {
      count +=
          mlir::isa<wafer::StorageLoadOp, wafer::StorageStoreOp>(operation);
    });
    return count;
  };
  const unsigned baselineTransfers =
      countTransfers(*frontier->front().ranks.front().module);
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        mlir::ModuleOp module = *variant.ranks.front().module;
        auto function = *module.getOps<mlir::func::FuncOp>().begin();
        return function.getNumResults() == 3 &&
               countTransfers(module) < baselineTransfers &&
               mlir::succeeded(mlir::verify(module));
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       ExactDirectMappedBoundaryMovementIsAnActualFrontierSibling) {
  auto source = makeMatmulProgram();
  ASSERT_TRUE(source);
  auto materialize = [&](wafer::CandidateBoundaryMovementAction movement) {
    std::string failureReason;
    auto candidate = wafer::materializeCompleteRankCandidateTileProgram(
        *source, /*logicalRank=*/0, /*candidateTileSizes=*/{2, 2},
        /*candidateReductionTileSizes=*/{},
        wafer::CandidateTileTraversalKind::ResultDriven,
        wafer::CompleteRankTraversalComposition::Coupled,
        wafer::CandidateTileResidencyAction::KeepSingleRegion, movement,
        wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidate)) << failureReason;
    return candidate;
  };
  auto staged = materialize(wafer::CandidateBoundaryMovementAction::Staged);
  auto direct =
      materialize(wafer::CandidateBoundaryMovementAction::ExactDirectMapped);
  ASSERT_TRUE(mlir::succeeded(staged));
  ASSERT_TRUE(mlir::succeeded(direct));

  auto countLayoutMaterializations = [](mlir::ModuleOp module) {
    unsigned count = 0;
    module.walk([&](wafer::LayoutMaterializeOp) { ++count; });
    return count;
  };
  auto hasDirectMappedLoad = [](mlir::ModuleOp module) {
    bool found = false;
    module.walk([&](wafer::StorageLoadOp load) {
      auto destination =
          mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
      found |=
          destination && wafer::getWaferMemoryAttr(destination).getLayout() !=
                             wafer::MemLayout::Tensor;
    });
    return found;
  };
  EXPECT_LT(countLayoutMaterializations(**direct),
            countLayoutMaterializations(**staged));
  EXPECT_FALSE(hasDirectMappedLoad(**staged));
  EXPECT_TRUE(hasDirectMappedLoad(**direct));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**direct)));

  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        return hasDirectMappedLoad(*variant.ranks.front().module);
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       DiamondFaninUsesOneCoupledTraversalWithAnExactTail) {
  auto source = makeDiamondFaninTailProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_GT(frontier->size(), 1u);

  auto countTransfers = [](mlir::ModuleOp module) {
    unsigned count = 0;
    module.walk([&](mlir::Operation *operation) {
      count +=
          mlir::isa<wafer::StorageLoadOp, wafer::StorageStoreOp>(operation);
    });
    return count;
  };
  const unsigned baselineTransfers =
      countTransfers(*frontier->front().ranks.front().module);
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        mlir::ModuleOp module = *variant.ranks.front().module;
        llvm::SmallSet<int64_t, 4> oneDimensionalSPMExtents;
        module.walk([&](mlir::memref::AllocOp alloc) {
          mlir::MemRefType type = alloc.getType();
          if (wafer::isWaferSPMMemRefType(type) && type.getRank() == 1)
            oneDimensionalSPMExtents.insert(type.getDimSize(0));
        });
        return countTransfers(module) < baselineTransfers &&
               oneDimensionalSPMExtents.contains(8) &&
               oneDimensionalSPMExtents.contains(2) &&
               mlir::succeeded(mlir::verify(module));
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       SharedInputContractionsReuseTheActualBoundaryTile) {
  auto source = makeSharedInputContractionProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        mlir::ModuleOp module = *variant.ranks.front().module;
        unsigned gemms = 0;
        unsigned loads = 0;
        module.walk([&](wafer::ComputeGemmOp) { ++gemms; });
        module.walk([&](wafer::StorageLoadOp) { ++loads; });
        return gemms == 2 && loads == 3 &&
               mlir::succeeded(mlir::verify(module));
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       TransposedWeightIsFusedWithoutAFullRuntimeTransposeRoundTrip) {
  auto source = makeTransposedWeightContractionProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        mlir::ModuleOp module = *variant.ranks.front().module;
        unsigned gemms = 0;
        bool hasFullTransposeBuffer = false;
        bool storesFullTranspose = false;
        bool hasDirectMappedLoad = false;
        module.walk([&](wafer::ComputeGemmOp) { ++gemms; });
        module.walk([&](mlir::memref::AllocOp alloc) {
          mlir::MemRefType type = alloc.getType();
          hasFullTransposeBuffer |= type.getRank() == 2 &&
                                    type.getDimSize(0) == 8 &&
                                    type.getDimSize(1) == 6;
        });
        module.walk([&](wafer::StorageStoreOp store) {
          auto type =
              mlir::dyn_cast<mlir::MemRefType>(store.getSource().getType());
          storesFullTranspose |= type && type.getRank() == 2 &&
                                 type.getDimSize(0) == 8 &&
                                 type.getDimSize(1) == 6;
        });
        module.walk([&](wafer::StorageLoadOp load) {
          auto type =
              mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
          hasDirectMappedLoad |=
              type && wafer::getWaferMemoryAttr(type).getLayout() !=
                          wafer::MemLayout::Tensor;
        });
        return gemms > 0 && !hasFullTransposeBuffer && !storesFullTranspose &&
               hasDirectMappedLoad && mlir::succeeded(mlir::verify(module));
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       ReadOnlyInvariantLoadMovesOutsideTheConsumerNLoop) {
  auto source = makeInvariantMatmulProgram();
  ASSERT_TRUE(source);
  auto materialize = [&](wafer::CandidateLoopMovementAction loopMovement) {
    std::string failureReason;
    auto candidate = wafer::materializeCompleteRankCandidateTileProgram(
        *source, /*logicalRank=*/0, /*candidateTileSizes=*/{4, 4},
        /*candidateReductionTileSizes=*/{},
        wafer::CandidateTileTraversalKind::ResultDriven,
        wafer::CompleteRankTraversalComposition::Coupled,
        wafer::CandidateTileResidencyAction::KeepSingleRegion,
        wafer::CandidateBoundaryMovementAction::ExactDirectMapped, loopMovement,
        &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidate)) << failureReason;
    return candidate;
  };
  auto constructed =
      materialize(wafer::CandidateLoopMovementAction::AsConstructed);
  auto hoisted = materialize(
      wafer::CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary);
  ASSERT_TRUE(mlir::succeeded(constructed));
  ASSERT_TRUE(mlir::succeeded(hoisted));

  auto maximumLoadLoopDepth = [](mlir::ModuleOp module, int64_t firstExtent,
                                 int64_t secondExtent) {
    unsigned maximum = 0;
    module.walk([&](wafer::StorageLoadOp load) {
      auto type = mlir::dyn_cast<mlir::MemRefType>(load.getDest().getType());
      if (!type || type.getRank() != 2 || type.getDimSize(0) != firstExtent ||
          type.getDimSize(1) != secondExtent)
        return;
      unsigned depth = 0;
      for (mlir::scf::ForOp loop = load->getParentOfType<mlir::scf::ForOp>();
           loop; loop = loop->getParentOfType<mlir::scf::ForOp>())
        ++depth;
      maximum = std::max(maximum, depth);
    });
    return maximum;
  };
  EXPECT_EQ(maximumLoadLoopDepth(**constructed, 4, 8), 2u);
  EXPECT_EQ(maximumLoadLoopDepth(**hoisted, 4, 8), 1u);
  EXPECT_EQ(maximumLoadLoopDepth(**hoisted, 8, 4), 2u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**hoisted)));

  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 1;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  EXPECT_TRUE(
      llvm::any_of(llvm::drop_begin(*frontier), [&](const auto &variant) {
        return maximumLoadLoopDepth(*variant.ranks.front().module, 4, 8) == 1 &&
               maximumLoadLoopDepth(*variant.ranks.front().module, 8, 4) == 2;
      }));
}

TEST_F(CoordinatedDataflowSearchTest,
       RepeatedConsumerOperandFusesOneSharedPhysicalProducerTile) {
  auto source = makeRepeatedProducerOperandContractionProgram();
  ASSERT_TRUE(source);
  std::string failureReason;
  auto candidate = wafer::materializeCompleteRankCandidateTileProgram(
      *source, /*logicalRank=*/0, /*candidateTileSizes=*/{4, 4},
      /*candidateReductionTileSizes=*/{},
      wafer::CandidateTileTraversalKind::ResultDriven,
      wafer::CompleteRankTraversalComposition::Coupled,
      wafer::CandidateTileResidencyAction::KeepSingleRegion,
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
  ASSERT_TRUE(mlir::succeeded(candidate)) << failureReason;

  unsigned gemms = 0;
  unsigned pointwise = 0;
  (*candidate)->walk([&](wafer::ComputeGemmOp) { ++gemms; });
  (*candidate)->walk([&](wafer::ComputeElementwiseOp) { ++pointwise; });
  EXPECT_EQ(gemms, 2u);
  EXPECT_EQ(pointwise, 1u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(**candidate)));
}

TEST_F(CoordinatedDataflowSearchTest,
       StructuredControlFlowKeepsBaselineWhenWholeTileIsUnsupported) {
  auto source = makeStructuredConcatProgram();
  ASSERT_TRUE(source);
  std::string materializationFailure;
  auto direct = wafer::materializeCompleteRankCandidateTileProgram(
      *source, /*logicalRank=*/0, /*candidateTileSizes=*/{1, 1, 1, 1},
      /*candidateReductionTileSizes=*/{},
      wafer::CandidateTileTraversalKind::ResultDriven,
      wafer::CompleteRankTraversalComposition::Coupled,
      wafer::CandidateTileResidencyAction::KeepSingleRegion,
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed,
      &materializationFailure);
  ASSERT_TRUE(mlir::succeeded(direct)) << materializationFailure;
  auto full = wafer::materializeCompleteRankCandidateTileProgram(
      *source, /*logicalRank=*/0, /*candidateTileSizes=*/{1, 2, 2, 8},
      /*candidateReductionTileSizes=*/{},
      wafer::CandidateTileTraversalKind::ResultDriven,
      wafer::CompleteRankTraversalComposition::Coupled,
      wafer::CandidateTileResidencyAction::KeepSingleRegion,
      wafer::CandidateBoundaryMovementAction::Staged,
      wafer::CandidateLoopMovementAction::AsConstructed,
      &materializationFailure);
  EXPECT_TRUE(mlir::failed(full));
  EXPECT_NE(materializationFailure.find(
                "unsupported linalg.generic body op linalg.index"),
            std::string::npos)
      << materializationFailure;
  unsigned directBranches = 0;
  (*direct)->walk([&](mlir::scf::IfOp) { ++directBranches; });
  EXPECT_GT(directBranches, 0u);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(2);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 2;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_EQ(frontier->size(), 1u);
  EXPECT_TRUE(frontier->front().reservedBaseline);
  ASSERT_EQ(frontier->front().ranks.size(), 2u);
  for (const auto &rank : frontier->front().ranks) {
    unsigned branches = 0;
    unsigned loads = 0;
    rank.module.get().walk([&](mlir::scf::IfOp) { ++branches; });
    rank.module.get().walk([&](wafer::StorageLoadOp) { ++loads; });
    EXPECT_GT(branches, 0u);
    EXPECT_GE(loads, 2u);
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*rank.module)));
  }
}

TEST_F(CoordinatedDataflowSearchTest,
       NonePolicyKeepsOnlyTheMandatoryConservativeVariant) {
  auto source = makeProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(2);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 2;
  config.optimizations = wafer::OptimizationConfig::none();
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_EQ(frontier->size(), 1u);
  EXPECT_TRUE(frontier->front().reservedBaseline);
}

TEST_F(CoordinatedDataflowSearchTest,
       FrontierDigestIsIndependentOfGenerationParallelism) {
  auto firstSource = makeChainProgram();
  auto secondSource = makeChainProgram();
  ASSERT_TRUE(firstSource);
  ASSERT_TRUE(secondSource);
  auto firstLedger = wafer::compiler::detail::CoordinatedWorkLedger::create(4);
  auto secondLedger = wafer::compiler::detail::CoordinatedWorkLedger::create(4);
  ASSERT_TRUE(mlir::succeeded(firstLedger));
  ASSERT_TRUE(mlir::succeeded(secondLedger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig serial;
  serial.rankCount = 4;
  serial.candidateParallelism = 1;
  auto parallel = serial;
  parallel.candidateParallelism = 8;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics
      serialStatistics;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics
      parallelStatistics;
  auto first = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *firstSource, serial, *firstLedger, &serialStatistics);
  auto second = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *secondSource, parallel, *secondLedger, &parallelStatistics);
  ASSERT_TRUE(mlir::succeeded(first));
  ASSERT_TRUE(mlir::succeeded(second));
  EXPECT_EQ(serialStatistics.connectionWorkerCount, 1u);
  // Rank-invariant actual proposals are materialized once and cloned to the
  // complete rank domain; candidateParallelism cannot change admission order
  // or introduce competing reservations.
  EXPECT_EQ(parallelStatistics.connectionWorkerCount, 1u);
  EXPECT_EQ(
      wafer::compiler::detail::computeCoordinatedTileFrontierDigest(*first),
      wafer::compiler::detail::computeCoordinatedTileFrontierDigest(*second));
}

TEST_F(CoordinatedDataflowSearchTest,
       ConnectionSearchUsesChainDPAndGeneralDAGBoundedBeam) {
  auto run = [&](mlir::OwningOpRef<mlir::ModuleOp> source,
                 unsigned expectedConnections, bool expectedDAG) {
    auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
    EXPECT_TRUE(mlir::succeeded(ledger));
    wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
    config.rankCount = 1;
    wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
    auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
        *source, config, *ledger, &stats);
    EXPECT_TRUE(mlir::succeeded(frontier));
    if (mlir::failed(frontier))
      return stats;
    EXPECT_EQ(stats.structuredConnections, expectedConnections);
    EXPECT_EQ(stats.usedGeneralDAGBeam, expectedDAG);
    EXPECT_GT(stats.connectionProposalExpansions, 0u);
    EXPECT_GE(stats.connectionProposalExpansions + 1,
              stats.connectionActionMaterializations);
    EXPECT_GT(stats.connectionActionMaterializations, 0u);
    EXPECT_LE(
        stats.connectionActionMaterializations,
        wafer::compiler::detail::kMaximumCoordinatedConnectionDAGBeamStates);
    EXPECT_GT(stats.retainedConnectionCandidates, 0u);
    EXPECT_GT(stats.connectionLocalChoicesPruned, 0u);
    // The production raw tile domain is larger (and therefore pruned), but
    // exact coverage dedup may retain only one representative for each of the
    // five action families when those representatives already cover every
    // producer/consumer tile feature.
    EXPECT_GE(stats.maximumConnectionLocalChoices, 5u);
    EXPECT_LE(stats.maximumConnectionLocalChoices,
              wafer::compiler::detail::
                  kMaximumCoordinatedConnectionExpansionsPerStep);
    EXPECT_GT(stats.maximumConnectionExpansionsPerParent, 0u);
    EXPECT_LE(stats.maximumConnectionExpansionsPerParent,
              wafer::compiler::detail::
                  kMaximumCoordinatedConnectionExpansionsPerStep);
    EXPECT_LE(stats.retainedStates, 1u + stats.materializedCandidates);
    auto finalization = wafer::compiler::detail::CoordinatedWorkLedger::
        getExecutableFinalizationUpperBound(1);
    EXPECT_TRUE(mlir::succeeded(finalization));
    EXPECT_TRUE(finalization->getTotal());
    EXPECT_EQ(ledger->getSnapshot().finalizationReserved,
              frontier->size() * *finalization->getTotal());
    return stats;
  };

  auto chain = run(makeChainProgram(), /*expectedConnections=*/1,
                   /*expectedDAG=*/false);
  EXPECT_GT(chain.maximumConnectionStates, 0u);
  EXPECT_LE(chain.maximumConnectionStates,
            wafer::compiler::detail::kMaximumCoordinatedConnectionDPStates);
  auto diamond = run(makeDiamondFaninTailProgram(),
                     /*expectedConnections=*/4, /*expectedDAG=*/true);
  EXPECT_GT(diamond.connectionProposalExpansions,
            diamond.connectionActionMaterializations);
  EXPECT_GT(diamond.maximumConnectionStates, 0u);
  EXPECT_LE(
      diamond.maximumConnectionStates,
      wafer::compiler::detail::kMaximumCoordinatedConnectionDAGBeamStates);
}

TEST_F(CoordinatedDataflowSearchTest,
       BudgetExhaustionPreservesBaselineReservationAndDigest) {
  auto firstSource = makeProgram();
  auto secondSource = makeProgram();
  ASSERT_TRUE(firstSource);
  ASSERT_TRUE(secondSource);
  constexpr int64_t rankCount = 1;
  auto finalization = wafer::compiler::detail::CoordinatedWorkLedger::
      getExecutableFinalizationUpperBound(rankCount);
  ASSERT_TRUE(mlir::succeeded(finalization));
  ASSERT_TRUE(finalization->getTotal());
  const uint64_t capacity = 2 + *finalization->getTotal();
  auto firstLedger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      rankCount, capacity, /*repairReserve=*/0);
  auto secondLedger = wafer::compiler::detail::CoordinatedWorkLedger::create(
      rankCount, capacity, /*repairReserve=*/0);
  ASSERT_TRUE(mlir::succeeded(firstLedger));
  ASSERT_TRUE(mlir::succeeded(secondLedger));

  wafer::compiler::detail::CoordinatedDataflowSearchConfig serial;
  serial.rankCount = rankCount;
  serial.candidateParallelism = 1;
  auto parallel = serial;
  parallel.candidateParallelism = 8;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics firstStats;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics secondStats;
  auto first = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *firstSource, serial, *firstLedger, &firstStats);
  auto second = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *secondSource, parallel, *secondLedger, &secondStats);
  ASSERT_TRUE(mlir::succeeded(first));
  ASSERT_TRUE(mlir::succeeded(second));
  ASSERT_EQ(first->size(), 1u);
  ASSERT_EQ(second->size(), 1u);
  EXPECT_TRUE(first->front().reservedBaseline);
  EXPECT_TRUE(second->front().reservedBaseline);
  EXPECT_EQ(
      wafer::compiler::detail::computeCoordinatedTileFrontierDigest(*first),
      wafer::compiler::detail::computeCoordinatedTileFrontierDigest(*second));
  EXPECT_EQ(firstStats.finalizationAdmissionDenied, 0u);
  EXPECT_EQ(secondStats.finalizationAdmissionDenied, 0u);
  EXPECT_EQ(firstStats.generationAdmissionDenied, 0u);
  EXPECT_EQ(secondStats.generationAdmissionDenied, 0u);
  EXPECT_TRUE(firstStats.structuralBudgetExhausted);
  EXPECT_TRUE(secondStats.structuralBudgetExhausted);
  for (const auto *ledger : {&*firstLedger, &*secondLedger}) {
    auto snapshot = ledger->getSnapshot();
    EXPECT_EQ(snapshot.finalizationReserved, *finalization->getTotal());
    EXPECT_EQ(snapshot.repairReserved, 0u);
    EXPECT_EQ(snapshot.unreserved, 0u);
  }
}

TEST_F(CoordinatedDataflowSearchTest,
       StructuredFactsRequireTheCompleteComparableSelectionVector) {
  using Facts = wafer::compiler::detail::CoordinatedStructuredFrontierFacts;
  using Dimension = wafer::compiler::detail::CoordinatedStructuredDimension;
  using Knowledge = wafer::compiler::detail::CoordinatedStructuredFactKnowledge;
  using Order = wafer::compiler::detail::CoordinatedStructuredFrontierOrder;
  auto metric = [](Facts &facts, Dimension dimension) -> auto & {
    return facts.selection[static_cast<size_t>(dimension)];
  };
  auto makeFacts = [&]() {
    Facts facts;
    facts.futureLiveInterface = "future";
    facts.physicalVersions = "versions";
    facts.loopReuseAndEffects = "effects";
    facts.collectiveAndPeerInterface = "peers";
    return facts;
  };

  Facts left = makeFacts();
  Facts right = makeFacts();
  metric(left, Dimension::DDRAggregateReadBytes).value = 8;
  metric(right, Dimension::DDRAggregateReadBytes).value = 16;
  EXPECT_EQ(wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
                left, right),
            Order::LeftDominates);

  Facts residentVersion = left;
  Facts separatedVersion = right;
  residentVersion.physicalVersions =
      "producer-result:storage=resident:tile=8:region-cut=0";
  separatedVersion.physicalVersions =
      "producer-result:storage=ddr:tile=4:region-cut=1";
  EXPECT_EQ(wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
                residentVersion, separatedVersion),
            Order::Incomparable);

  metric(left, Dimension::CompletionMaximumRankWaitWork).knowledge =
      Knowledge::Estimated;
  metric(right, Dimension::CompletionMaximumRankWaitWork).knowledge =
      Knowledge::Estimated;
  metric(left, Dimension::CompletionMaximumRankWaitWork).value = 4;
  metric(right, Dimension::CompletionMaximumRankWaitWork).value = 8;
  metric(left, Dimension::CompletionMaximumRankWaitWork).disposition =
      "same-estimate-model";
  metric(right, Dimension::CompletionMaximumRankWaitWork).disposition =
      "same-estimate-model";
  EXPECT_EQ(wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
                left, right),
            Order::LeftDominates);

  metric(right, Dimension::CompletionMaximumRankWaitWork).disposition =
      "different-estimate-model";
  EXPECT_EQ(wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
                left, right),
            Order::Incomparable);

  metric(left, Dimension::CompletionMaximumRankWaitWork).knowledge =
      Knowledge::Unknown;
  metric(right, Dimension::CompletionMaximumRankWaitWork).knowledge =
      Knowledge::Unknown;
  metric(left, Dimension::CompletionMaximumRankWaitWork).disposition =
      "same-current-effects";
  metric(right, Dimension::CompletionMaximumRankWaitWork).disposition =
      "same-current-effects";
  metric(left, Dimension::CompletionMaximumRankWaitWork).value = 0;
  metric(right, Dimension::CompletionMaximumRankWaitWork).value = 0;
  EXPECT_EQ(wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
                left, right),
            Order::LeftDominates);

  metric(right, Dimension::CompletionMaximumRankWaitWork).disposition =
      "different-current-effects";
  EXPECT_EQ(wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
                left, right),
            Order::Incomparable);

  metric(right, Dimension::CompletionMaximumRankWaitWork).disposition =
      "same-current-effects";
  metric(left, Dimension::LocalAggregateMovementBytes).value = 32;
  metric(right, Dimension::LocalAggregateMovementBytes).value = 16;
  EXPECT_EQ(wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
                left, right),
            Order::Incomparable);
}

TEST_F(CoordinatedDataflowSearchTest,
       StructuredPromotionRejectsOnlyProvenInvariantMovementRegression) {
  using Facts = wafer::compiler::detail::CoordinatedStructuredFrontierFacts;
  using Dimension = wafer::compiler::detail::CoordinatedStructuredDimension;
  using Knowledge = wafer::compiler::detail::CoordinatedStructuredFactKnowledge;
  auto metric = [](Facts &facts, Dimension dimension) -> auto & {
    return facts.selection[static_cast<size_t>(dimension)];
  };

  Facts baseline;
  Facts candidate;
  metric(baseline, Dimension::LocalAggregateMovementBytes).value = 16;
  metric(candidate, Dimension::LocalAggregateMovementBytes).value = 32;
  EXPECT_FALSE(
      wafer::compiler::detail::canStillSatisfyCoordinatedProductionPromotion(
          candidate, baseline));

  metric(candidate, Dimension::LocalAggregateMovementBytes).knowledge =
      Knowledge::Unknown;
  metric(candidate, Dimension::LocalAggregateMovementBytes).disposition =
      "dynamic-local-movement";
  EXPECT_TRUE(
      wafer::compiler::detail::canStillSatisfyCoordinatedProductionPromotion(
          candidate, baseline));

  metric(candidate, Dimension::LocalAggregateMovementBytes).knowledge =
      Knowledge::Known;
  metric(candidate, Dimension::LocalAggregateMovementBytes).value = 16;
  metric(candidate, Dimension::ComputeAggregateWork).value = 1024;
  EXPECT_TRUE(
      wafer::compiler::detail::canStillSatisfyCoordinatedProductionPromotion(
          candidate, baseline));
}

TEST_F(CoordinatedDataflowSearchTest,
       ConstantConditionalCountsOnlyTheReachableStructuredBranch) {
  auto module = makeConstantConditionalTileProgram();
  ASSERT_TRUE(module);
  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.ranks.emplace_back(/*logicalRank=*/0, std::move(module), nullptr);
  auto facts =
      wafer::compiler::detail::deriveCoordinatedStructuredFrontierFacts(
          variant);
  ASSERT_TRUE(mlir::succeeded(facts));
  using Dimension = wafer::compiler::detail::CoordinatedStructuredDimension;
  using Knowledge = wafer::compiler::detail::CoordinatedStructuredFactKnowledge;
  const auto &reads =
      facts->selection[static_cast<size_t>(Dimension::DDRAggregateReadBytes)];
  EXPECT_EQ(reads.knowledge, Knowledge::Known);
  EXPECT_EQ(reads.value, 8u);
}

TEST_F(CoordinatedDataflowSearchTest,
       StructuredInsertionIsOrderIndependentAndRetainsBaseline) {
  using Facts = wafer::compiler::detail::CoordinatedStructuredFrontierFacts;
  using Dimension = wafer::compiler::detail::CoordinatedStructuredDimension;
  using View = wafer::compiler::detail::CoordinatedStructuredFrontierView;
  auto makeFacts = [](uint64_t ddr, uint64_t local) {
    Facts facts;
    facts.futureLiveInterface = "future";
    facts.physicalVersions = "versions";
    facts.loopReuseAndEffects = "effects";
    facts.collectiveAndPeerInterface = "peers";
    facts.selection[static_cast<size_t>(Dimension::DDRAggregateReadBytes)]
        .value = ddr;
    facts.selection[static_cast<size_t>(Dimension::LocalAggregateMovementBytes)]
        .value = local;
    return facts;
  };

  // Ordinal 0 is the mandatory baseline. Ordinals 1/2/3 are mutually
  // incomparable Pareto representatives, 4 is dominated, and 5 is a later
  // exact equivalent of 1.
  std::array<Facts, 6> facts = {makeFacts(100, 100), makeFacts(80, 100),
                                makeFacts(100, 80),  makeFacts(90, 90),
                                makeFacts(120, 120), makeFacts(80, 100)};
  auto run = [&](std::array<size_t, 5> order) {
    llvm::SmallVector<size_t, 8> retained = {0};
    for (size_t candidateIndex : order) {
      llvm::SmallVector<View, 8> existing;
      for (size_t retainedIndex : retained)
        existing.push_back({static_cast<int64_t>(retainedIndex),
                            retainedIndex == 0, &facts[retainedIndex]});
      auto plan =
          wafer::compiler::detail::planCoordinatedStructuredFrontierInsertion(
              {static_cast<int64_t>(candidateIndex), false,
               &facts[candidateIndex]},
              existing);
      EXPECT_TRUE(mlir::succeeded(plan));
      if (mlir::failed(plan) || !plan->retainCandidate)
        continue;
      for (size_t eraseIndex : llvm::reverse(plan->eraseIndices))
        retained.erase(retained.begin() + eraseIndex);
      retained.push_back(candidateIndex);
    }
    llvm::sort(retained);
    return retained;
  };

  std::array<size_t, 5> order = {1, 2, 3, 4, 5};
  do {
    EXPECT_EQ(run(order), (llvm::SmallVector<size_t, 8>{0, 1, 2, 3}));
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST_F(CoordinatedDataflowSearchTest,
       CurrentIRFactsIgnoreSymbolSpellingButDistinguishPhysicalActions) {
  auto source = makeMatmulProgram();
  ASSERT_TRUE(source);
  auto materialize = [&](wafer::CandidateBoundaryMovementAction movement) {
    std::string failureReason;
    auto candidate = wafer::materializeCompleteRankCandidateTileProgram(
        *source, /*logicalRank=*/0, /*candidateTileSizes=*/{2, 2},
        /*candidateReductionTileSizes=*/{},
        wafer::CandidateTileTraversalKind::ResultDriven,
        wafer::CompleteRankTraversalComposition::Coupled,
        wafer::CandidateTileResidencyAction::KeepSingleRegion, movement,
        wafer::CandidateLoopMovementAction::AsConstructed, &failureReason);
    EXPECT_TRUE(mlir::succeeded(candidate)) << failureReason;
    return candidate;
  };
  auto staged = materialize(wafer::CandidateBoundaryMovementAction::Staged);
  auto direct =
      materialize(wafer::CandidateBoundaryMovementAction::ExactDirectMapped);
  ASSERT_TRUE(mlir::succeeded(staged));
  ASSERT_TRUE(mlir::succeeded(direct));
  auto renamed = mlir::cast<mlir::ModuleOp>((*staged)->clone());
  auto renamedFunction = *renamed.getOps<mlir::func::FuncOp>().begin();
  renamedFunction.setSymName("different_user_spelling");

  auto makeVariant = [](mlir::OwningOpRef<mlir::ModuleOp> module) {
    wafer::compiler::detail::CoordinatedTileVariant variant;
    variant.stableSemanticOrdinal = 1;
    variant.finalizationReservation.id = 1;
    variant.ranks.emplace_back(/*logicalRank=*/0, std::move(module), nullptr);
    return variant;
  };
  auto stagedVariant = makeVariant(std::move(*staged));
  auto renamedVariant = makeVariant(std::move(renamed));
  auto directVariant = makeVariant(std::move(*direct));
  auto stagedFacts =
      wafer::compiler::detail::deriveCoordinatedStructuredFrontierFacts(
          stagedVariant);
  auto renamedFacts =
      wafer::compiler::detail::deriveCoordinatedStructuredFrontierFacts(
          renamedVariant);
  auto directFacts =
      wafer::compiler::detail::deriveCoordinatedStructuredFrontierFacts(
          directVariant);
  ASSERT_TRUE(mlir::succeeded(stagedFacts));
  ASSERT_TRUE(mlir::succeeded(renamedFacts));
  ASSERT_TRUE(mlir::succeeded(directFacts));
  using Dimension = wafer::compiler::detail::CoordinatedStructuredDimension;
  using Knowledge = wafer::compiler::detail::CoordinatedStructuredFactKnowledge;
  const auto &compute =
      stagedFacts
          ->selection[static_cast<size_t>(Dimension::ComputeAggregateWork)];
  EXPECT_EQ(compute.knowledge, Knowledge::Known);
  EXPECT_GT(compute.value, 0u);
  const auto &criticalPath =
      stagedFacts
          ->selection[static_cast<size_t>(Dimension::CriticalPathLowerBound)];
  EXPECT_EQ(criticalPath.knowledge, Knowledge::Known);
  EXPECT_GT(criticalPath.value, 0u);
  const auto &underutilization =
      stagedFacts
          ->selection[static_cast<size_t>(Dimension::TileUnderutilization)];
  EXPECT_EQ(underutilization.knowledge, Knowledge::Known);
  for (Dimension dimension :
       {Dimension::DDRMaximumRankIssueWork,
        Dimension::CompletionMaximumRankWaitWork, Dimension::InstrAggregateWork,
        Dimension::DescriptorPressure, Dimension::ResourcePressure}) {
    const auto &estimate =
        stagedFacts->selection[static_cast<size_t>(dimension)];
    EXPECT_EQ(estimate.knowledge, Knowledge::Estimated);
    EXPECT_FALSE(estimate.disposition.empty());
  }
  const auto &recompute =
      stagedFacts
          ->selection[static_cast<size_t>(Dimension::RecomputeAggregateWork)];
  EXPECT_EQ(recompute.knowledge, Knowledge::Estimated);
  EXPECT_EQ(recompute.value, 0u);
  EXPECT_FALSE(recompute.disposition.empty());
  EXPECT_EQ(
      wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
          *stagedFacts, *renamedFacts),
      wafer::compiler::detail::CoordinatedStructuredFrontierOrder::Equivalent);
  EXPECT_EQ(wafer::compiler::detail::compareCoordinatedStructuredFrontierFacts(
                *stagedFacts, *directFacts),
            wafer::compiler::detail::CoordinatedStructuredFrontierOrder::
                Incomparable);
}

TEST_F(CoordinatedDataflowSearchTest,
       BatchedGemmWorkUsesVerifiedLogicalDimensions) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %lhs: memref<6x5x8xf16, #wafer.memory<spm, ncx>>,
      %rhs: memref<6x8x7xf16, #wafer.memory<spm, ncx>>) {
    %result = wafer.tile.gemm %lhs, %rhs
        {batch_count = 6 : i64,
         lhs_batch_dims = array<i64: 0>,
         lhs_contracting_dim = 2 : i64,
         lhs_m_dim = 1 : i64,
         result_batch_dims = array<i64: 0>,
         result_m_dim = 1 : i64,
         result_n_dim = 2 : i64,
         rhs_batch_dims = array<i64: 0>,
         rhs_contracting_dim = 1 : i64,
         rhs_n_dim = 2 : i64}
        : (memref<6x5x8xf16, #wafer.memory<spm, ncx>>,
           memref<6x8x7xf16, #wafer.memory<spm, ncx>>)
       -> memref<6x5x7xf16, #wafer.memory<spm, ncx>>
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.stableSemanticOrdinal = 1;
  variant.finalizationReservation.id = 1;
  variant.ranks.emplace_back(/*logicalRank=*/0, std::move(module), nullptr);
  auto facts =
      wafer::compiler::detail::deriveCoordinatedStructuredFrontierFacts(
          variant);
  ASSERT_TRUE(mlir::succeeded(facts));
  const auto &compute = facts->selection[static_cast<size_t>(
      wafer::compiler::detail::CoordinatedStructuredDimension::
          ComputeAggregateWork)];
  EXPECT_EQ(compute.knowledge,
            wafer::compiler::detail::CoordinatedStructuredFactKnowledge::Known);
  EXPECT_EQ(compute.value, 2u * 6u * 5u * 8u * 7u);
  const auto &underutilization = facts->selection[static_cast<size_t>(
      wafer::compiler::detail::CoordinatedStructuredDimension::
          TileUnderutilization)];
  EXPECT_EQ(underutilization.knowledge,
            wafer::compiler::detail::CoordinatedStructuredFactKnowledge::Known);
  EXPECT_GT(underutilization.value, 0u);
}

TEST_F(CoordinatedDataflowSearchTest,
       StructuredFactsPreserveStaticExecutionOverflowDisposition) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  func.func @main(
      %boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %boundary : memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cmax = arith.constant 9223372036854775807 : index
      %spm = memref.alloc()
          : memref<4xf16, #wafer.memory<spm, tensor>>
      scf.for %outer = %c0 to %cmax step %c1 {
        scf.for %inner = %c0 to %cmax step %c1 {
          %copy = wafer.tile.copy %spm
              : memref<4xf16, #wafer.memory<spm, tensor>>
             -> memref<4xf16, #wafer.memory<spm, tensor>>
        }
      }
      wafer.tile.yield %arg0
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));

  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.stableSemanticOrdinal = 1;
  variant.finalizationReservation.id = 1;
  variant.ranks.emplace_back(/*logicalRank=*/0, std::move(module), nullptr);
  auto facts =
      wafer::compiler::detail::deriveCoordinatedStructuredFrontierFacts(
          variant);
  ASSERT_TRUE(mlir::succeeded(facts));
  const auto &movement = facts->selection[static_cast<size_t>(
      wafer::compiler::detail::CoordinatedStructuredDimension::
          LocalAggregateMovementBytes)];
  EXPECT_EQ(
      movement.knowledge,
      wafer::compiler::detail::CoordinatedStructuredFactKnowledge::Overflow);
  EXPECT_FALSE(movement.disposition.empty());
}

TEST_F(CoordinatedDataflowSearchTest,
       ProductionReductionKeepsExactFinalizationReservations) {
  auto run = [&](mlir::OwningOpRef<mlir::ModuleOp> source) {
    auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(1);
    EXPECT_TRUE(mlir::succeeded(ledger));
    wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
    config.rankCount = 1;
    wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
    auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
        *source, config, *ledger, &stats);
    EXPECT_TRUE(mlir::succeeded(frontier));
    if (mlir::failed(frontier))
      return uint64_t{0};
    EXPECT_EQ(stats.retainedStates, frontier->size());
    EXPECT_GT(stats.materializedCandidates, 0u);
    auto finalization = wafer::compiler::detail::CoordinatedWorkLedger::
        getExecutableFinalizationUpperBound(1);
    EXPECT_TRUE(mlir::succeeded(finalization));
    EXPECT_TRUE(finalization->getTotal());
    EXPECT_EQ(ledger->getSnapshot().finalizationReserved,
              frontier->size() * *finalization->getTotal());
    return stats.retainedStates;
  };

  auto chain = run(makeChainProgram());
  auto fanout = run(makeMixedShapeFanoutProgram());
  EXPECT_GT(chain, 1u);
  EXPECT_GT(fanout, 1u);
}

TEST_F(CoordinatedDataflowSearchTest,
       AllRankCollectiveParametersAreReprovedFromActualTileIR) {
  auto source = makeAllReduceProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(2);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 2;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_FALSE(frontier->empty());
  llvm::SmallSet<int64_t, 4> collectiveTileBytes;
  for (const auto &variant : *frontier) {
    EXPECT_TRUE(mlir::succeeded(
        wafer::compiler::detail::verifyCoordinatedStructuredCommunication(
            variant, 2)));
    llvm::SmallVector<int64_t, 16> referenceBytes;
    for (auto [rankIndex, rank] : llvm::enumerate(variant.ranks)) {
      llvm::SmallVector<int64_t, 16> rankBytes;
      rank.module.get().walk([&](wafer::CommAllReduceOp collective) {
        rankBytes.push_back(collective.getBytes());
        if (rankIndex == 0)
          collectiveTileBytes.insert(collective.getBytes());
      });
      if (rankIndex == 0)
        referenceBytes = std::move(rankBytes);
      else
        EXPECT_EQ(rankBytes, referenceBytes);
    }
  }
  // The baseline and a larger-tile sibling are separate actual all-rank
  // states. Shared collective granularity is therefore expanded before
  // executable finalization, while target ring/tree selection remains a
  // Tile-to-Instr responsibility.
  EXPECT_TRUE(collectiveTileBytes.contains(2));
  EXPECT_TRUE(collectiveTileBytes.contains(16));

  wafer::CommAllReduceOp rankOneCollective;
  frontier->front().ranks[1].module.get().walk(
      [&](wafer::CommAllReduceOp collective) {
        if (!rankOneCollective)
          rankOneCollective = collective;
      });
  ASSERT_TRUE(rankOneCollective);
  auto facts =
      wafer::compiler::detail::deriveCoordinatedStructuredFrontierFacts(
          frontier->front());
  ASSERT_TRUE(mlir::succeeded(facts));
  const auto &payload = facts->selection[static_cast<size_t>(
      wafer::compiler::detail::CoordinatedStructuredDimension::
          NoCAggregatePayloadBytes)];
  EXPECT_EQ(payload.knowledge,
            wafer::compiler::detail::CoordinatedStructuredFactKnowledge::Known);
  // Each rank issues eight 2-byte tile requests for the complete 16-byte
  // logical tensor. The all-rank aggregate therefore expands to 32 bytes.
  EXPECT_EQ(payload.value, 32u);
  rankOneCollective->setAttr(
      "communication_id",
      mlir::IntegerAttr::get(mlir::IntegerType::get(context.get(), 64), 18));
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::verifyCoordinatedStructuredCommunication(
          frontier->front(), 2)));
}

TEST_F(CoordinatedDataflowSearchTest,
       ReturnedReductionAddsAnActualAllRankPartialSibling) {
  auto source = makeLocalReductionAllReduceProgram();
  ASSERT_TRUE(source);
  auto ledger = wafer::compiler::detail::CoordinatedWorkLedger::create(2);
  ASSERT_TRUE(mlir::succeeded(ledger));
  wafer::compiler::detail::CoordinatedDataflowSearchConfig config;
  config.rankCount = 2;
  wafer::compiler::detail::CoordinatedStructuredFrontierStatistics stats;
  auto frontier = wafer::compiler::detail::buildCoordinatedTileFrontier(
      *source, config, *ledger, &stats);
  ASSERT_TRUE(mlir::succeeded(frontier));
  ASSERT_FALSE(frontier->empty());

  bool foundPartial = false;
  for (const auto &variant : *frontier) {
    ASSERT_EQ(variant.ranks.size(), 2u);
    llvm::SmallVector<unsigned, 2> rankReductionCounts;
    for (const auto &rank : variant.ranks) {
      unsigned reductions = 0;
      unsigned merges = 0;
      unsigned collectives = 0;
      rank.module.get().walk([&](wafer::ComputeReduceOp) { ++reductions; });
      rank.module.get().walk([&](wafer::ComputeElementwiseOp) { ++merges; });
      rank.module.get().walk([&](wafer::CommAllReduceOp) { ++collectives; });
      EXPECT_EQ(collectives, 1u);
      rankReductionCounts.push_back(reductions);
      if (reductions > 1 && merges > 0)
        foundPartial = true;
    }
    EXPECT_EQ(rankReductionCounts.front(), rankReductionCounts.back());
    EXPECT_TRUE(mlir::succeeded(
        wafer::compiler::detail::verifyCoordinatedTileVariant(variant, 2)));
  }
  EXPECT_TRUE(foundPartial)
      << "structural=" << stats.structuralProposalsDerived
      << " attempts=" << stats.actualCandidateAttempts
      << " failures=" << stats.candidateMaterializationFailures
      << " equivalent=" << stats.equivalentCandidatesRejected
      << " dominated=" << stats.dominatedCandidatesRejected
      << " promotion=" << stats.promotionIneligibleCandidatesRejected;
}

TEST_F(CoordinatedDataflowSearchTest,
       PeerMessagesRequireAllAndOnlyTypedEndpointsBeforeFinalization) {
  auto makePeerModule = [&](bool send, int64_t peer,
                            llvm::StringRef elementType) {
    std::string text =
        (llvm::Twine(R"mlir(
module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {topology = @default, axes = ["rank"], shape = array<i64: 2>,
       policy = "all_available", endpoints = array<i64>}
  func.func @main(
      %boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %boundary : memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %buffer = memref.alloc() : memref<4x)mlir") +
         elementType + R"mlir(, #wafer.memory<spm, tensor>>
      %token = wafer.tile.peer_)mlir" +
         (send ? "send" : "recv") + " %buffer {peer = " + llvm::Twine(peer) +
         R"mlir( : i64, bytes = 8 : i64,
        message = #wafer.dte_message<communication = 29, phase = peer_dataflow, round = 0, slice = 0>}
        : memref<4x)mlir" +
         elementType + R"mlir(, #wafer.memory<spm, tensor>> -> !async.token
      async.await %token : !async.token
      wafer.tile.yield %arg0
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir")
            .str();
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  };
  auto makeVariant = [&](llvm::StringRef recvElementType) {
    auto sender = makePeerModule(/*send=*/true, /*peer=*/1, "f16");
    auto receiver = makePeerModule(/*send=*/false, /*peer=*/0, recvElementType);
    EXPECT_TRUE(sender);
    EXPECT_TRUE(receiver);
    wafer::compiler::detail::CoordinatedTileVariant variant;
    variant.stableSemanticOrdinal = 1;
    variant.finalizationReservation.id = 1;
    variant.ranks.emplace_back(0, std::move(sender), nullptr);
    variant.ranks.emplace_back(1, std::move(receiver), nullptr);
    return variant;
  };

  auto matched = makeVariant("f16");
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::verifyCoordinatedStructuredCommunication(matched,
                                                                        2)));
  auto mismatched = makeVariant("i16");
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::verifyCoordinatedStructuredCommunication(
          mismatched, 2)));
}

TEST_F(CoordinatedDataflowSearchTest,
       PeerMessagesOnDifferentStructuredBranchesDoNotMatch) {
  auto parseEndpoint =
      [&](bool send, bool inThenBranch, bool condition) {
        llvm::StringRef prefix = send ? "peer_send" : "peer_recv";
        int64_t peer = send ? 1 : 0;
        std::string thenBody;
        std::string elseBody;
        std::string endpoint =
            (llvm::Twine("%buffer = memref.alloc() : memref<4xf16, "
                         "#wafer.memory<spm, tensor>>\n") +
             "        %token = wafer.tile." + prefix +
             " %buffer {peer = " + llvm::Twine(peer) +
             " : i64, bytes = 8 : i64, message = "
             "#wafer.dte_message<communication = 31, phase = peer_dataflow, "
             "round = 0, slice = 0>} : memref<4xf16, "
             "#wafer.memory<spm, tensor>> -> !async.token\n"
             "        async.await %token : !async.token\n")
                .str();
        if (inThenBranch)
          thenBody = endpoint;
        else
          elseBody = endpoint;
        std::string text = (llvm::Twine(R"mlir(
module {
  func.func @main(
      %boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %boundary : memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %condition = arith.constant )mlir") +
                            (condition ? "true" : "false") + R"mlir(
      scf.if %condition {
        )mlir" + thenBody + R"mlir(      } else {
        )mlir" + elseBody + R"mlir(      }
      wafer.tile.yield %arg0
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir")
                               .str();
        return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
      };

  auto sender = parseEndpoint(/*send=*/true, /*inThenBranch=*/true,
                              /*condition=*/true);
  auto receiver = parseEndpoint(/*send=*/false, /*inThenBranch=*/false,
                                /*condition=*/true);
  ASSERT_TRUE(sender);
  ASSERT_TRUE(receiver);
  wafer::compiler::detail::CoordinatedTileVariant variant;
  variant.stableSemanticOrdinal = 1;
  variant.finalizationReservation.id = 1;
  variant.ranks.emplace_back(0, std::move(sender), nullptr);
  variant.ranks.emplace_back(1, std::move(receiver), nullptr);
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::verifyCoordinatedStructuredCommunication(variant,
                                                                        2)));

  auto conditionSender = parseEndpoint(/*send=*/true, /*inThenBranch=*/true,
                                       /*condition=*/true);
  auto conditionReceiver = parseEndpoint(/*send=*/false, /*inThenBranch=*/true,
                                         /*condition=*/false);
  ASSERT_TRUE(conditionSender);
  ASSERT_TRUE(conditionReceiver);
  wafer::compiler::detail::CoordinatedTileVariant conditionVariant;
  conditionVariant.stableSemanticOrdinal = 2;
  conditionVariant.finalizationReservation.id = 2;
  conditionVariant.ranks.emplace_back(0, std::move(conditionSender), nullptr);
  conditionVariant.ranks.emplace_back(1, std::move(conditionReceiver), nullptr);
  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::verifyCoordinatedStructuredCommunication(
          conditionVariant, 2)));
}

TEST_F(CoordinatedDataflowSearchTest,
       SymbolicPeerLoopBoundsProveEndpointsAndUseStaticSiteEstimate) {
  auto parseEndpoint = [&](bool send, unsigned boundArgument) {
    llvm::StringRef operation = send ? "peer_send" : "peer_recv";
    int64_t peer = send ? 1 : 0;
    std::string text =
        (llvm::Twine(R"mlir(
module {
  func.func @main(
      %extent0: index, %extent1: index,
      %boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
    %region = wafer.tile.region(
        %extent0, %extent1, %boundary
        : index, index, memref<4xf16, #wafer.memory<ddr, tensor>>)
        -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
    ^bb0(%region_extent0: index, %region_extent1: index,
         %arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %region_extent)mlir") +
         llvm::Twine(boundArgument) + R"mlir( step %c1 {
        %buffer = memref.alloc()
            : memref<4xf16, #wafer.memory<spm, tensor>>
        %token = wafer.tile.)mlir" +
         operation + " %buffer {peer = " + llvm::Twine(peer) + R"mlir( : i64,
          bytes = 8 : i64,
          message = #wafer.dte_message<communication = 37, phase = peer_dataflow, round = 0, slice = 0>}
          : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
        async.await %token : !async.token
      }
      wafer.tile.yield %arg0
          : memref<4xf16, #wafer.memory<ddr, tensor>>
    }
    return
  }
}
)mlir")
            .str();
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  };
  auto makeVariant = [&](unsigned sendBound, unsigned receiveBound) {
    auto sender = parseEndpoint(/*send=*/true, sendBound);
    auto receiver = parseEndpoint(/*send=*/false, receiveBound);
    EXPECT_TRUE(sender);
    EXPECT_TRUE(receiver);
    wafer::compiler::detail::CoordinatedTileVariant variant;
    variant.stableSemanticOrdinal = 1;
    variant.finalizationReservation.id = 1;
    variant.ranks.emplace_back(0, std::move(sender), nullptr);
    variant.ranks.emplace_back(1, std::move(receiver), nullptr);
    return variant;
  };

  auto matched = makeVariant(/*sendBound=*/0, /*receiveBound=*/0);
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::verifyCoordinatedStructuredCommunication(matched,
                                                                        2)));
  auto facts =
      wafer::compiler::detail::deriveCoordinatedStructuredFrontierFacts(
          matched);
  ASSERT_TRUE(mlir::succeeded(facts));
  const auto &payload = facts->selection[static_cast<size_t>(
      wafer::compiler::detail::CoordinatedStructuredDimension::
          NoCAggregatePayloadBytes)];
  EXPECT_EQ(
      payload.knowledge,
      wafer::compiler::detail::CoordinatedStructuredFactKnowledge::Estimated);
  EXPECT_EQ(payload.value, 16u);
  EXPECT_FALSE(payload.disposition.empty());

  auto mismatched = makeVariant(/*sendBound=*/0, /*receiveBound=*/1);
  mlir::ScopedDiagnosticHandler handler(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  EXPECT_TRUE(mlir::failed(
      wafer::compiler::detail::verifyCoordinatedStructuredCommunication(
          mismatched, 2)));
}

} // namespace
