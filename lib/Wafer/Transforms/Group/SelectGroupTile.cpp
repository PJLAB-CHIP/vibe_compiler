//===- SelectGroupTile.cpp - R3.2h group tile selection ------------------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>

namespace wafer {
#define GEN_PASS_DEF_SELECTGROUPTILEPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

enum class TileSearchMode { FirstLegal, MinEstimatedTime };

struct CandidateSpec {
  llvm::SmallVector<int64_t, 4> tileSizes;
  llvm::SmallVector<int64_t, 2> reductionSplitSizes;
};

struct TileInstance {
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
};

struct CandidateStats {
  int64_t computeOps = 0;
  int64_t ddrBytes = 0;
  int64_t spmBytes = 0;
  int64_t instrCount = 0;
};

struct CandidateEvaluation {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  CandidateStats stats;
  std::string failureReason;
};

struct SelectedCandidate {
  std::string label;
  std::string symbolBase;
  unsigned ordinal = 0;
  CandidateSpec spec;
  CandidateStats stats;
  int64_t estimatedCycles = 0;
  int64_t candidateCount = 0;
  int64_t rejectedCount = 0;
  int64_t representativeCount = 0;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

static std::string getNearestSymbolName(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto name = parent->getAttrOfType<mlir::StringAttr>("sym_name"))
      return ("@" + name.getValue()).str();
  }
  return "@unknown";
}

static std::string sanitizeSymbolBase(llvm::StringRef label) {
  llvm::StringRef base = label;
  if (base.consume_front("@")) {
  }
  size_t hash = base.find('#');
  if (hash != llvm::StringRef::npos)
    base = base.take_front(hash);

  std::string result;
  for (char c : base) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') {
      result.push_back(c);
    } else {
      result.push_back('_');
    }
  }
  if (result.empty())
    return "group";
  return result;
}

static void printI64List(llvm::ArrayRef<int64_t> values,
                         llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0)
      os << ",";
    os << value;
  }
  os << "]";
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
parseI64List(llvm::StringRef text, llvm::StringRef optionName,
             mlir::Operation *anchor) {
  llvm::SmallVector<int64_t, 8> values;
  llvm::SmallVector<llvm::StringRef, 8> parts;
  llvm::SplitString(text, parts, ",");
  for (llvm::StringRef part : parts) {
    part = part.trim();
    if (part.empty())
      continue;
    int64_t value = 0;
    if (part.getAsInteger(10, value) || value <= 0) {
      anchor->emitError() << "invalid positive integer in " << optionName
                          << ": " << part;
      return mlir::failure();
    }
    values.push_back(value);
  }
  return values;
}

static mlir::FailureOr<TileSearchMode>
parseTileSearchMode(llvm::StringRef text, mlir::Operation *anchor) {
  if (text == "first-legal")
    return TileSearchMode::FirstLegal;
  if (text == "min-estimated-time")
    return TileSearchMode::MinEstimatedTime;
  anchor->emitError()
      << "invalid_tile_search: expected first-legal or min-estimated-time";
  return mlir::failure();
}

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static int64_t saturatingAdd(int64_t lhs, int64_t rhs) {
  int64_t result = 0;
  if (!checkedAdd(lhs, rhs, result))
    return std::numeric_limits<int64_t>::max();
  return result;
}

static int64_t saturatingMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return std::numeric_limits<int64_t>::max();
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return std::numeric_limits<int64_t>::max();
  return lhs * rhs;
}

static int64_t ceilDiv(int64_t numerator, int64_t denominator) {
  if (numerator <= 0)
    return 0;
  if (denominator <= 0)
    return std::numeric_limits<int64_t>::max();
  return (numerator + denominator - 1) / denominator;
}

static int64_t getStaticMemRefElementCount(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape())
    return 0;
  int64_t count = 1;
  for (int64_t dim : type.getShape())
    count = saturatingMul(count, dim);
  return count;
}

static int64_t getOptionalBatchCount(InstrGemmOp op) {
  if (auto attr = op.getBatchCountAttr())
    return attr.getInt();
  return 1;
}

static CandidateStats estimateStats(mlir::ModuleOp module) {
  CandidateStats stats;
  module.walk([&](mlir::Operation *op) {
    if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(op)) {
      stats.instrCount = saturatingAdd(stats.instrCount, 1);
      stats.ddrBytes =
          saturatingAdd(stats.ddrBytes, rdma.getByteCountAttr().getInt());
      return;
    }
    if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(op)) {
      stats.instrCount = saturatingAdd(stats.instrCount, 1);
      stats.ddrBytes =
          saturatingAdd(stats.ddrBytes, wdma.getByteCountAttr().getInt());
      return;
    }
    if (auto gs = mlir::dyn_cast<InstrGatherScatterOp>(op)) {
      stats.instrCount = saturatingAdd(stats.instrCount, 1);
      stats.spmBytes =
          saturatingAdd(stats.spmBytes, gs.getByteCountAttr().getInt());
      return;
    }
    if (auto fill = mlir::dyn_cast<InstrFillOp>(op)) {
      stats.instrCount = saturatingAdd(stats.instrCount, 1);
      stats.computeOps = saturatingAdd(
          stats.computeOps, getStaticMemRefElementCount(fill.getDest()));
      return;
    }
    if (auto elementwise = mlir::dyn_cast<InstrElementwiseOp>(op)) {
      stats.instrCount = saturatingAdd(stats.instrCount, 1);
      int64_t elements = getStaticMemRefElementCount(elementwise.getDest());
      int64_t inputCount = std::max<int64_t>(1, elementwise.getInputs().size());
      stats.computeOps =
          saturatingAdd(stats.computeOps, saturatingMul(elements, inputCount));
      return;
    }
    if (auto reduce = mlir::dyn_cast<InstrReduceOp>(op)) {
      stats.instrCount = saturatingAdd(stats.instrCount, 1);
      stats.computeOps = saturatingAdd(
          stats.computeOps, getStaticMemRefElementCount(reduce.getInput()));
      return;
    }
    if (auto convert = mlir::dyn_cast<InstrConvertOp>(op)) {
      stats.instrCount = saturatingAdd(stats.instrCount, 1);
      stats.computeOps = saturatingAdd(
          stats.computeOps, getStaticMemRefElementCount(convert.getDest()));
      return;
    }
    if (auto gemm = mlir::dyn_cast<InstrGemmOp>(op)) {
      stats.instrCount = saturatingAdd(stats.instrCount, 1);
      int64_t ops =
          saturatingMul(gemm.getMAttr().getInt(), gemm.getNAttr().getInt());
      ops = saturatingMul(ops, gemm.getKAttr().getInt());
      ops = saturatingMul(ops, getOptionalBatchCount(gemm));
      ops = saturatingMul(ops, 2);
      stats.computeOps = saturatingAdd(stats.computeOps, ops);
      return;
    }
  });
  return stats;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
getStaticTraversalShape(GroupOp group) {
  if (group.getNumResults() != 1)
    return mlir::failure();
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(group.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> shape(resultType.getShape().begin(),
                                      resultType.getShape().end());
  return shape;
}

static std::optional<int64_t> getElementByteWidth(mlir::Type type) {
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    type = shaped.getElementType();
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return std::max<int64_t>(1, floatType.getWidth() / 8);
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return std::max<int64_t>(1, (intType.getWidth() + 7) / 8);
  return std::nullopt;
}

static mlir::linalg::LinalgOp getYieldedRootLinalgOp(GroupOp group) {
  auto yield =
      mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return nullptr;
  return mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
      yield.getValues().front().getDefiningOp());
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 2>>
getStaticMatmulReductionRanges(GroupOp group) {
  mlir::linalg::LinalgOp root = getYieldedRootLinalgOp(group);
  if (!root || !mlir::isa<mlir::linalg::MatmulOp>(root.getOperation()))
    return llvm::SmallVector<int64_t, 2>{};

  llvm::SmallVector<int64_t, 4> loopRanges = root.getStaticLoopRanges();
  if (llvm::any_of(loopRanges, [](int64_t range) {
        return mlir::ShapedType::isDynamic(range);
      }))
    return mlir::failure();

  llvm::SmallVector<int64_t, 2> reductionRanges;
  for (auto [index, iteratorType] :
       llvm::enumerate(root.getIteratorTypesArray())) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      reductionRanges.push_back(loopRanges[index]);
  }
  return reductionRanges;
}

static std::optional<int64_t>
estimateCompactSPMBytes(GroupOp group, const CandidateSpec &candidate) {
  mlir::linalg::LinalgOp root = getYieldedRootLinalgOp(group);
  if (!root)
    return std::nullopt;

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return std::nullopt;
  std::optional<int64_t> elementBytes = getElementByteWidth(resultType);
  if (!elementBytes)
    return std::nullopt;

  if (mlir::isa<mlir::linalg::MatmulOp>(root.getOperation()) &&
      candidate.tileSizes.size() == 2 && root.getNumDpsInputs() == 2) {
    auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(
        root.getDpsInputOperand(0)->get().getType());
    auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(
        root.getDpsInputOperand(1)->get().getType());
    if (!lhsType || !rhsType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape())
      return std::nullopt;
    int64_t m = candidate.tileSizes[0];
    int64_t n = candidate.tileSizes[1];
    int64_t k = lhsType.getDimSize(1);
    if (!candidate.reductionSplitSizes.empty())
      k = candidate.reductionSplitSizes.front();
    int64_t elements = saturatingAdd(saturatingMul(m, k), saturatingMul(k, n));
    elements = saturatingAdd(elements, saturatingMul(m, n));
    // GEMM lowering currently materializes tensor and aligned-family SPM
    // buffers. This bound is intentionally conservative; full legality still
    // comes from R3.2e-g.
    return saturatingMul(saturatingMul(elements, *elementBytes), 4);
  }

  int64_t tileElements = 1;
  for (int64_t size : candidate.tileSizes)
    tileElements = saturatingMul(tileElements, size);
  int64_t bufferCount =
      std::max<int64_t>(1, root.getNumDpsInputs() + root.getNumDpsInits());
  return saturatingMul(saturatingMul(tileElements, *elementBytes), bufferCount);
}

static bool failsCheapSPMBound(GroupOp group, const CandidateSpec &candidate,
                               int64_t spmBase, int64_t spmLimit) {
  std::optional<int64_t> estimatedBytes =
      estimateCompactSPMBytes(group, candidate);
  if (!estimatedBytes)
    return false;
  if (spmLimit <= spmBase)
    return false;
  return *estimatedBytes > (spmLimit - spmBase);
}

static void addUnique(llvm::SmallVectorImpl<int64_t> &values, int64_t value,
                      int64_t dim) {
  if (value <= 0 || value > dim)
    return;
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

static llvm::SmallVector<int64_t, 8>
buildDimTileSizes(int64_t dim, llvm::ArrayRef<int64_t> preferred,
                  int64_t maxCandidatesPerDim) {
  llvm::SmallVector<int64_t, 8> values;
  addUnique(values, dim, dim);

  for (int64_t divisor = 2; divisor * divisor <= dim; ++divisor) {
    if (dim % divisor != 0)
      continue;
    addUnique(values, dim / divisor, dim);
    addUnique(values, divisor, dim);
  }

  for (int64_t value : preferred)
    addUnique(values, value, dim);

  llvm::sort(values, std::greater<int64_t>());
  if (maxCandidatesPerDim > 0 &&
      static_cast<int64_t>(values.size()) > maxCandidatesPerDim)
    values.resize(static_cast<size_t>(maxCandidatesPerDim));
  return values;
}

static void buildTileSizeProducts(
    llvm::ArrayRef<llvm::SmallVector<int64_t, 8>> perDimSizes, unsigned dim,
    llvm::SmallVectorImpl<int64_t> &current,
    llvm::SmallVectorImpl<llvm::SmallVector<int64_t, 4>> &tileProducts) {
  if (dim == perDimSizes.size()) {
    tileProducts.push_back(
        llvm::SmallVector<int64_t, 4>(current.begin(), current.end()));
    return;
  }
  for (int64_t size : perDimSizes[dim]) {
    current.push_back(size);
    buildTileSizeProducts(perDimSizes, dim + 1, current, tileProducts);
    current.pop_back();
  }
}

static void buildReductionSplitProducts(
    llvm::ArrayRef<llvm::SmallVector<int64_t, 8>> perDimSizes,
    llvm::ArrayRef<int64_t> ranges, unsigned dim,
    llvm::SmallVectorImpl<int64_t> &current,
    llvm::SmallVectorImpl<llvm::SmallVector<int64_t, 2>> &splits) {
  if (dim == perDimSizes.size()) {
    bool isFullRange = true;
    for (auto [range, size] : llvm::zip(ranges, current)) {
      if (range != size) {
        isFullRange = false;
        break;
      }
    }
    if (!isFullRange)
      splits.push_back(
          llvm::SmallVector<int64_t, 2>(current.begin(), current.end()));
    return;
  }
  for (int64_t size : perDimSizes[dim]) {
    current.push_back(size);
    buildReductionSplitProducts(perDimSizes, ranges, dim + 1, current, splits);
    current.pop_back();
  }
}

static llvm::SmallVector<llvm::SmallVector<int64_t, 2>, 8>
buildReductionSplitSpecs(llvm::ArrayRef<int64_t> reductionRanges,
                         llvm::ArrayRef<int64_t> preferred,
                         int64_t maxCandidatesPerDim) {
  llvm::SmallVector<llvm::SmallVector<int64_t, 2>, 8> splits;
  splits.push_back({});
  if (reductionRanges.empty())
    return splits;

  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 2> perDimSizes;
  for (int64_t range : reductionRanges)
    perDimSizes.push_back(
        buildDimTileSizes(range, preferred, maxCandidatesPerDim));

  llvm::SmallVector<int64_t, 2> current;
  buildReductionSplitProducts(perDimSizes, reductionRanges, /*dim=*/0, current,
                              splits);
  return splits;
}

static llvm::SmallVector<CandidateSpec, 32>
buildCandidateSpecs(llvm::ArrayRef<int64_t> traversalShape,
                    llvm::ArrayRef<int64_t> reductionRanges,
                    llvm::ArrayRef<int64_t> preferred,
                    int64_t maxCandidatesPerDim) {
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 4> perDimSizes;
  for (int64_t dim : traversalShape)
    perDimSizes.push_back(
        buildDimTileSizes(dim, preferred, maxCandidatesPerDim));

  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 32> tileProducts;
  llvm::SmallVector<int64_t, 4> currentTile;
  buildTileSizeProducts(perDimSizes, 0, currentTile, tileProducts);

  llvm::SmallVector<llvm::SmallVector<int64_t, 2>, 8> splitSpecs =
      buildReductionSplitSpecs(reductionRanges, preferred, maxCandidatesPerDim);

  llvm::SmallVector<CandidateSpec, 32> candidates;
  for (llvm::ArrayRef<int64_t> tileSizes : tileProducts) {
    CandidateSpec candidate;
    candidate.tileSizes.assign(tileSizes.begin(), tileSizes.end());
    candidates.push_back(std::move(candidate));
  }
  for (llvm::ArrayRef<int64_t> splitSizes : llvm::drop_begin(splitSpecs)) {
    for (llvm::ArrayRef<int64_t> tileSizes : tileProducts) {
      CandidateSpec candidate;
      candidate.tileSizes.assign(tileSizes.begin(), tileSizes.end());
      candidate.reductionSplitSizes.assign(splitSizes.begin(),
                                           splitSizes.end());
      candidates.push_back(std::move(candidate));
    }
  }
  return candidates;
}

static bool isFullFirstTile(llvm::ArrayRef<int64_t> traversalShape,
                            const TileInstance &tile) {
  if (tile.offsets.size() != traversalShape.size() ||
      tile.sizes.size() != traversalShape.size())
    return false;
  for (auto [shapeDim, values] :
       llvm::zip(traversalShape, llvm::zip(tile.offsets, tile.sizes))) {
    if (std::get<0>(values) != 0 || std::get<1>(values) != shapeDim)
      return false;
  }
  return true;
}

static llvm::SmallVector<TileInstance, 8>
buildRepresentativeTiles(llvm::ArrayRef<int64_t> traversalShape,
                         llvm::ArrayRef<int64_t> tileSizes) {
  llvm::SmallVector<llvm::SmallVector<std::pair<int64_t, int64_t>, 2>, 4>
      perDim;
  for (auto [dim, tileSize] : llvm::zip(traversalShape, tileSizes)) {
    llvm::SmallVector<std::pair<int64_t, int64_t>, 2> reps;
    reps.push_back({0, std::min(dim, tileSize)});
    if (dim > tileSize) {
      int64_t tail = dim % tileSize;
      if (tail == 0)
        reps.push_back({dim - tileSize, tileSize});
      else
        reps.push_back({dim - tail, tail});
    }
    perDim.push_back(reps);
  }

  llvm::SmallVector<TileInstance, 8> tiles;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> sizes;
  std::function<void(unsigned)> build = [&](unsigned dim) {
    if (dim == perDim.size()) {
      for (const TileInstance &tile : tiles)
        if (tile.offsets == offsets && tile.sizes == sizes)
          return;
      tiles.push_back(TileInstance{llvm::SmallVector<int64_t, 4>(offsets),
                                   llvm::SmallVector<int64_t, 4>(sizes)});
      return;
    }
    for (auto [offset, size] : perDim[dim]) {
      offsets.push_back(offset);
      sizes.push_back(size);
      build(dim + 1);
      offsets.pop_back();
      sizes.pop_back();
    }
  };
  build(0);
  return tiles;
}

static std::string
takeDiagnostics(mlir::MLIRContext *context,
                llvm::function_ref<mlir::LogicalResult()> callback,
                mlir::LogicalResult &result) {
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context, [&](mlir::Diagnostic &diagnostic) {
        llvm::raw_string_ostream os(diagnostics);
        diagnostic.print(os);
        os << "\n";
        return mlir::success();
      });
  result = callback();
  return diagnostics;
}

static std::string joinFailure(llvm::StringRef gate, llvm::StringRef reason,
                               llvm::StringRef diagnostics) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << gate << ": ";
  if (!reason.empty())
    os << reason;
  else if (!diagnostics.empty())
    os << diagnostics.trim();
  else
    os << "failed";
  return os.str();
}

static CandidateEvaluation evaluateTileInstance(
    GroupOp group, llvm::ArrayRef<int64_t> traversalShape,
    const CandidateSpec &candidate, const TileInstance &tile, int64_t spmBase,
    int64_t spmLimit, int64_t spmAlignment, int64_t ddrCapacityBytes,
    int64_t ddrLargestContiguousBytes, int64_t ddrBandwidthLimitBytes,
    int64_t ddrAlignmentBytes) {
  CandidateEvaluation evaluation;
  mlir::MLIRContext *context = group.getContext();

  std::string failureReason;
  mlir::LogicalResult result = mlir::success();
  std::string diagnostics = takeDiagnostics(
      context,
      [&]() {
        return lowerCandidateGroupToTileRegionModule(
            group, tile.offsets, tile.sizes, candidate.reductionSplitSizes,
            evaluation.module, &failureReason);
      },
      result);

  if (mlir::failed(result) && isFullFirstTile(traversalShape, tile)) {
    failureReason.clear();
    diagnostics = takeDiagnostics(
        context,
        [&]() {
          return lowerGroupToTileRegionModule(group, evaluation.module,
                                              &failureReason);
        },
        result);
  }

  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("R3.2e", failureReason, diagnostics);
    return evaluation;
  }

  failureReason.clear();
  diagnostics = takeDiagnostics(
      context,
      [&]() {
        return convertTileRegionToInstrModule(*evaluation.module,
                                              &failureReason);
      },
      result);
  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("R3.2d", failureReason, diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context,
      [&]() {
        return planSPMMemoryModule(*evaluation.module, spmBase, spmLimit,
                                   spmAlignment);
      },
      result);
  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("R3.2f", "", diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context,
      [&]() {
        return planDDRMemoryModule(*evaluation.module, ddrAlignmentBytes,
                                   ddrCapacityBytes, ddrLargestContiguousBytes,
                                   ddrBandwidthLimitBytes);
      },
      result);
  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("R3.2g", "", diagnostics);
    return evaluation;
  }

  diagnostics = takeDiagnostics(
      context, [&]() { return mlir::verify(*evaluation.module); }, result);
  if (mlir::failed(result)) {
    evaluation.failureReason = joinFailure("verifier", "", diagnostics);
    return evaluation;
  }

  evaluation.stats = estimateStats(*evaluation.module);
  return evaluation;
}

static int64_t computeTileCount(llvm::ArrayRef<int64_t> traversalShape,
                                llvm::ArrayRef<int64_t> tileSizes) {
  int64_t count = 1;
  for (auto [dim, tile] : llvm::zip(traversalShape, tileSizes))
    count = saturatingMul(count, ceilDiv(dim, tile));
  return count;
}

static int64_t estimateCycles(const CandidateStats &stats, int64_t tileCount,
                              int64_t computeOpsPerCycle,
                              int64_t ddrBytesPerCycle,
                              int64_t spmBytesPerCycle,
                              int64_t instrIssueCycles,
                              bool assumeDdrComputeOverlap) {
  int64_t computeCycles = ceilDiv(stats.computeOps, computeOpsPerCycle);
  int64_t ddrCycles = ceilDiv(stats.ddrBytes, ddrBytesPerCycle);
  int64_t spmCycles = ceilDiv(stats.spmBytes, spmBytesPerCycle);
  int64_t issueCycles = saturatingMul(stats.instrCount, instrIssueCycles);
  int64_t localCycles = 0;
  if (assumeDdrComputeOverlap)
    localCycles = std::max(computeCycles, ddrCycles);
  else
    localCycles = saturatingAdd(computeCycles, ddrCycles);
  localCycles = saturatingAdd(localCycles, spmCycles);
  localCycles = saturatingAdd(localCycles, issueCycles);
  return saturatingMul(localCycles, tileCount);
}

struct SelectionConfig {
  TileSearchMode mode = TileSearchMode::FirstLegal;
  llvm::SmallVector<int64_t, 8> preferredTileSizes;
  int64_t maxCandidatesPerDim = 8;
  int64_t spmBase = 65536;
  int64_t spmLimit = 3080192;
  int64_t spmAlignment = 256;
  int64_t ddrCapacityBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrLargestContiguousBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrBandwidthLimitBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrAlignmentBytes = 256;
  int64_t computeOpsPerCycle = 1024;
  int64_t ddrBytesPerCycle = 256;
  int64_t spmBytesPerCycle = 1024;
  int64_t instrIssueCycles = 1;
  bool assumeDdrComputeOverlap = false;
};

static bool isBetterCandidate(const SelectedCandidate &candidate,
                              const SelectedCandidate *best) {
  if (!best)
    return true;
  if (candidate.estimatedCycles != best->estimatedCycles)
    return candidate.estimatedCycles < best->estimatedCycles;
  if (candidate.spec.tileSizes != best->spec.tileSizes)
    return candidate.spec.tileSizes > best->spec.tileSizes;
  if (candidate.spec.reductionSplitSizes.empty() !=
      best->spec.reductionSplitSizes.empty())
    return candidate.spec.reductionSplitSizes.empty();
  return candidate.spec.reductionSplitSizes > best->spec.reductionSplitSizes;
}

static mlir::FailureOr<SelectedCandidate>
selectCandidateForGroup(GroupOp group, llvm::StringRef label, unsigned ordinal,
                        const SelectionConfig &config) {
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> shape =
      getStaticTraversalShape(group);
  if (mlir::failed(shape)) {
    group.emitError()
        << "no_candidate: R3.2h requires one static ranked group result";
    return mlir::failure();
  }

  mlir::FailureOr<llvm::SmallVector<int64_t, 2>> reductionRanges =
      getStaticMatmulReductionRanges(group);
  if (mlir::failed(reductionRanges)) {
    group.emitError()
        << "no_candidate: R3.2h requires static matmul reduction ranges";
    return mlir::failure();
  }

  llvm::SmallVector<CandidateSpec, 32> candidates =
      buildCandidateSpecs(*shape, *reductionRanges, config.preferredTileSizes,
                          config.maxCandidatesPerDim);
  if (candidates.empty()) {
    group.emitError() << "no_candidate: empty candidate search space";
    return mlir::failure();
  }

  std::optional<SelectedCandidate> best;
  int64_t rejectedCount = 0;
  std::string lastFailure;
  int64_t visitedCount = 0;

  for (const CandidateSpec &candidate : candidates) {
    ++visitedCount;
    if (failsCheapSPMBound(group, candidate, config.spmBase, config.spmLimit)) {
      ++rejectedCount;
      lastFailure = "cheap_bound: estimated SPM bytes exceed planning window";
      continue;
    }
    llvm::SmallVector<TileInstance, 8> reps =
        buildRepresentativeTiles(*shape, candidate.tileSizes);

    CandidateEvaluation firstEvaluation;
    bool passed = true;
    for (auto [repIndex, rep] : llvm::enumerate(reps)) {
      CandidateEvaluation evaluation = evaluateTileInstance(
          group, *shape, candidate, rep, config.spmBase, config.spmLimit,
          config.spmAlignment, config.ddrCapacityBytes,
          config.ddrLargestContiguousBytes, config.ddrBandwidthLimitBytes,
          config.ddrAlignmentBytes);
      if (!evaluation.failureReason.empty()) {
        passed = false;
        ++rejectedCount;
        lastFailure = evaluation.failureReason;
        break;
      }
      if (repIndex == 0)
        firstEvaluation = std::move(evaluation);
    }

    if (!passed)
      continue;

    SelectedCandidate selected;
    selected.label = label.str();
    selected.symbolBase = sanitizeSymbolBase(label);
    selected.ordinal = ordinal;
    selected.spec = candidate;
    selected.stats = firstEvaluation.stats;
    selected.estimatedCycles = estimateCycles(
        selected.stats, computeTileCount(*shape, candidate.tileSizes),
        config.computeOpsPerCycle, config.ddrBytesPerCycle,
        config.spmBytesPerCycle, config.instrIssueCycles,
        config.assumeDdrComputeOverlap);
    selected.candidateCount = visitedCount;
    selected.rejectedCount = rejectedCount;
    selected.representativeCount = static_cast<int64_t>(reps.size());
    selected.module = std::move(firstEvaluation.module);

    if (config.mode == TileSearchMode::FirstLegal)
      return selected;

    if (isBetterCandidate(selected, best ? &*best : nullptr))
      best = std::move(selected);
  }

  if (best) {
    best->candidateCount = visitedCount;
    best->rejectedCount = rejectedCount;
    return std::move(*best);
  }

  group.emitError() << "no_candidate: R3.2h found no passing candidate"
                    << (lastFailure.empty() ? "" : "; last failure: ")
                    << lastFailure;
  return mlir::failure();
}

static void printSelectedSummary(const SelectedCandidate &selected,
                                 TileSearchMode mode) {
  llvm::errs() << "wafer.r3_2h selected group " << selected.label << " mode="
               << (mode == TileSearchMode::FirstLegal ? "first-legal"
                                                      : "min-estimated-time")
               << " tile=";
  printI64List(selected.spec.tileSizes, llvm::errs());
  llvm::errs() << " split=";
  printI64List(selected.spec.reductionSplitSizes, llvm::errs());
  llvm::errs() << " estimated_cycles=" << selected.estimatedCycles
               << " candidates=" << selected.candidateCount
               << " rejected=" << selected.rejectedCount
               << " representatives=" << selected.representativeCount << "\n";
}

static void moveSelectedModulesInto(
    mlir::ModuleOp target,
    llvm::MutableArrayRef<SelectedCandidate> selectedCandidates) {
  mlir::Block *body = target.getBody();
  body->getOperations().clear();

  for (SelectedCandidate &selected : selectedCandidates) {
    unsigned funcOrdinal = 0;
    for (mlir::Operation &op : llvm::make_early_inc_range(
             selected.module->getBody()->getOperations())) {
      op.remove();
      if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(&op)) {
        std::string name;
        llvm::raw_string_ostream os(name);
        os << selected.symbolBase << "_selected_group_" << selected.ordinal;
        if (funcOrdinal != 0)
          os << "_" << funcOrdinal;
        func.setName(os.str());
        ++funcOrdinal;
      }
      body->push_back(&op);
    }
  }
}

struct SelectGroupTilePass
    : public impl::SelectGroupTilePassBase<SelectGroupTilePass> {
  using impl::SelectGroupTilePassBase<
      SelectGroupTilePass>::SelectGroupTilePassBase;

  void runOnOperation() final {
    mlir::FailureOr<TileSearchMode> parsedMode =
        parseTileSearchMode(tileSearch, getOperation());
    mlir::FailureOr<llvm::SmallVector<int64_t, 8>> parsedPreferred =
        parseI64List(preferredTileSizes, "preferred-tile-sizes",
                     getOperation());
    if (mlir::failed(parsedMode) || mlir::failed(parsedPreferred)) {
      signalPassFailure();
      return;
    }
    if (maxCandidatesPerDim <= 0 || computeOpsPerCycle <= 0 ||
        ddrBytesPerCycle <= 0 || spmBytesPerCycle <= 0 ||
        instrIssueCycles < 0) {
      getOperation()->emitError()
          << "invalid_tile_search_config: candidate and timing limits must be "
             "positive, with non-negative instr issue cycles";
      signalPassFailure();
      return;
    }

    SelectionConfig config;
    config.mode = *parsedMode;
    config.preferredTileSizes = *parsedPreferred;
    config.maxCandidatesPerDim = maxCandidatesPerDim;
    config.spmBase = spmBase;
    config.spmLimit = spmLimit;
    config.spmAlignment = spmAlignment;
    config.ddrCapacityBytes = ddrCapacityBytes;
    config.ddrLargestContiguousBytes = ddrLargestContiguousBytes;
    config.ddrBandwidthLimitBytes = ddrBandwidthLimitBytes;
    config.ddrAlignmentBytes = ddrAlignmentBytes;
    config.computeOpsPerCycle = computeOpsPerCycle;
    config.ddrBytesPerCycle = ddrBytesPerCycle;
    config.spmBytesPerCycle = spmBytesPerCycle;
    config.instrIssueCycles = instrIssueCycles;
    config.assumeDdrComputeOverlap = assumeDdrComputeOverlap;

    llvm::SmallVector<GroupOp, 8> groups;
    getOperation().walk([&](GroupOp group) { groups.push_back(group); });
    if (groups.empty()) {
      markAllAnalysesPreserved();
      return;
    }

    llvm::DenseMap<mlir::Operation *, unsigned> groupOrdinals;
    llvm::SmallVector<SelectedCandidate, 8> selectedCandidates;
    for (GroupOp group : groups) {
      std::string symbolName = getNearestSymbolName(group.getOperation());
      unsigned ordinal = groupOrdinals[group->getParentOp()]++;
      std::string label;
      llvm::raw_string_ostream labelOs(label);
      labelOs << symbolName << "#" << ordinal;

      mlir::FailureOr<SelectedCandidate> selected =
          selectCandidateForGroup(group, labelOs.str(), ordinal, config);
      if (mlir::failed(selected)) {
        signalPassFailure();
        return;
      }
      if (printCandidateSummary)
        printSelectedSummary(*selected, config.mode);
      selectedCandidates.push_back(std::move(*selected));
    }

    moveSelectedModulesInto(getOperation(), selectedCandidates);
  }
};

} // namespace

} // namespace wafer
