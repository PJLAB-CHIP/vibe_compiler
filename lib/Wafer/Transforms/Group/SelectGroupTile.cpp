//===- SelectGroupTile.cpp - Closed-loop group tile selection -------------===//

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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
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
  GroupOp group;
  CandidateSpec spec;
  CandidateStats stats;
  int64_t estimatedCycles = 0;
  int64_t candidateCount = 0;
  int64_t rejectedCount = 0;
  int64_t representativeCount = 0;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

struct CandidateWorkItem {
  CandidateSpec spec;
};

static std::string getNearestSymbolName(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto name = parent->getAttrOfType<mlir::StringAttr>("sym_name"))
      return ("@" + name.getValue()).str();
  }
  return "@unknown";
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
  if (group.getNumResults() == 0)
    return mlir::failure();

  auto firstType =
      mlir::dyn_cast<mlir::RankedTensorType>(group.getResult(0).getType());
  if (!firstType || !firstType.hasStaticShape())
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> shape(firstType.getShape().begin(),
                                      firstType.getShape().end());

  for (mlir::Value result : group.getResults().drop_front()) {
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!resultType || !resultType.hasStaticShape())
      return mlir::failure();
    if (!std::equal(shape.begin(), shape.end(), resultType.getShape().begin(),
                    resultType.getShape().end()))
      return mlir::failure();
  }
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

static std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
getYieldedRootLinalgOps(GroupOp group) {
  auto yield =
      mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
  if (!yield || yield.getValues().empty())
    return std::nullopt;

  llvm::SmallVector<mlir::linalg::LinalgOp, 4> roots;
  for (mlir::Value value : yield.getValues()) {
    auto root = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(
        value.getDefiningOp());
    if (!root)
      return std::nullopt;
    roots.push_back(root);
  }
  return roots;
}

static bool hasReductionIterator(mlir::linalg::LinalgOp op) {
  for (mlir::utils::IteratorType iteratorType : op.getIteratorTypesArray()) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      return true;
  }
  return false;
}

static mlir::FailureOr<llvm::SmallVector<int64_t, 2>>
getStaticRootReductionRanges(GroupOp group) {
  std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      getYieldedRootLinalgOps(group);
  if (!roots)
    return llvm::SmallVector<int64_t, 2>{};

  std::optional<llvm::SmallVector<int64_t, 2>> commonReductionRanges;
  for (mlir::linalg::LinalgOp root : *roots) {
    if (!mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::GenericOp>(
            root.getOperation()) ||
        !hasReductionIterator(root))
      continue;

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
    if (reductionRanges.empty())
      continue;
    if (!commonReductionRanges) {
      commonReductionRanges = reductionRanges;
      continue;
    }
    if (*commonReductionRanges != reductionRanges)
      return llvm::SmallVector<int64_t, 2>{};
  }
  if (!commonReductionRanges)
    return llvm::SmallVector<int64_t, 2>{};
  return *commonReductionRanges;
}

static std::optional<int64_t> estimateRootMinimumSPMBytes(
    mlir::linalg::LinalgOp root, const CandidateSpec &candidate) {
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
    return saturatingMul(elements, *elementBytes);
  }

  int64_t tileElements = 1;
  for (int64_t size : candidate.tileSizes)
    tileElements = saturatingMul(tileElements, size);
  return saturatingMul(tileElements, *elementBytes);
}

static std::optional<int64_t>
estimateRequiredSPMLowerBoundBytes(GroupOp group,
                                   const CandidateSpec &candidate) {
  std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      getYieldedRootLinalgOps(group);
  if (!roots)
    return std::nullopt;

  int64_t required = 0;
  for (mlir::linalg::LinalgOp root : *roots) {
    std::optional<int64_t> rootBytes =
        estimateRootMinimumSPMBytes(root, candidate);
    if (!rootBytes)
      return std::nullopt;
    required = std::max(required, *rootBytes);
  }
  return required;
}

static bool failsCheapSPMBound(GroupOp group, const CandidateSpec &candidate,
                               int64_t spmBase, int64_t spmLimit) {
  std::optional<int64_t> lowerBoundBytes =
      estimateRequiredSPMLowerBoundBytes(group, candidate);
  if (!lowerBoundBytes)
    return false;
  if (spmLimit <= spmBase)
    return false;
  return *lowerBoundBytes > (spmLimit - spmBase);
}

static void addUnique(llvm::SmallVectorImpl<int64_t> &values, int64_t value,
                      int64_t dim) {
  if (value <= 0 || value > dim)
    return;
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

static int64_t absDiff(int64_t lhs, int64_t rhs) {
  return lhs > rhs ? lhs - rhs : rhs - lhs;
}

static int64_t chooseNextRefinementSize(int64_t dim, int64_t current,
                                        llvm::ArrayRef<int64_t> preferred) {
  if (current <= 1)
    return current;

  int64_t target = std::max<int64_t>(1, ceilDiv(current, 2));
  llvm::SmallVector<int64_t, 16> candidates;
  addUnique(candidates, target, dim);

  for (int64_t divisor = 2; divisor * divisor <= dim; ++divisor) {
    if (dim % divisor != 0)
      continue;
    addUnique(candidates, dim / divisor, dim);
    addUnique(candidates, divisor, dim);
  }

  for (int64_t value : preferred)
    addUnique(candidates, value, dim);
  addUnique(candidates, 1, dim);

  int64_t best = current;
  bool found = false;
  for (int64_t candidate : candidates) {
    if (candidate >= current)
      continue;
    if (!found) {
      best = candidate;
      found = true;
      continue;
    }

    int64_t candidateScore = absDiff(candidate, target);
    int64_t bestScore = absDiff(best, target);
    bool candidateDividesDim = dim % candidate == 0;
    bool bestDividesDim = dim % best == 0;
    if (candidateScore < bestScore ||
        (candidateScore == bestScore && candidateDividesDim &&
         !bestDividesDim) ||
        (candidateScore == bestScore && candidateDividesDim == bestDividesDim &&
         candidate > best))
      best = candidate;
  }

  return found ? best : current;
}

static llvm::SmallVector<int64_t, 8>
buildDimTileSizes(int64_t dim, llvm::ArrayRef<int64_t> preferred,
                  int64_t maxCandidatesPerDim) {
  llvm::SmallVector<int64_t, 8> values;
  addUnique(values, dim, dim);

  int64_t current = dim;
  while (maxCandidatesPerDim <= 0 ||
         static_cast<int64_t>(values.size()) < maxCandidatesPerDim) {
    int64_t next = chooseNextRefinementSize(dim, current, preferred);
    if (next >= current)
      break;
    addUnique(values, next, dim);
    current = next;
  }
  return values;
}

struct SearchLadders {
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 4> traversal;
  llvm::SmallVector<llvm::SmallVector<int64_t, 8>, 2> reduction;
};

static SearchLadders buildSearchLadders(llvm::ArrayRef<int64_t> traversalShape,
                                        llvm::ArrayRef<int64_t> reductionRanges,
                                        llvm::ArrayRef<int64_t> preferred,
                                        int64_t maxCandidatesPerDim) {
  SearchLadders ladders;
  for (int64_t dim : traversalShape)
    ladders.traversal.push_back(
        buildDimTileSizes(dim, preferred, maxCandidatesPerDim));

  for (int64_t range : reductionRanges)
    ladders.reduction.push_back(
        buildDimTileSizes(range, preferred, maxCandidatesPerDim));
  return ladders;
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

  if (mlir::failed(result) && candidate.reductionSplitSizes.empty() &&
      isFullFirstTile(traversalShape, tile)) {
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
    evaluation.failureReason =
        joinFailure("tile-region", failureReason, diagnostics);
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
    evaluation.failureReason =
        joinFailure("instr-lowering", failureReason, diagnostics);
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
    evaluation.failureReason = joinFailure("spm-offsets", "", diagnostics);
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
    evaluation.failureReason = joinFailure("ddr-offsets", "", diagnostics);
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

static std::string getCandidateKey(const CandidateSpec &candidate) {
  std::string key;
  llvm::raw_string_ostream os(key);
  printI64List(candidate.tileSizes, os);
  os << "|";
  printI64List(candidate.reductionSplitSizes, os);
  return os.str();
}

static std::optional<size_t> findSizeIndex(llvm::ArrayRef<int64_t> sizes,
                                           int64_t value) {
  for (auto [index, size] : llvm::enumerate(sizes))
    if (size == value)
      return static_cast<size_t>(index);
  return std::nullopt;
}

struct RefinementDim {
  bool isReduction = false;
  unsigned index = 0;
  int64_t pressure = 0;
};

static int64_t getCandidateReductionSize(const CandidateSpec &candidate,
                                         llvm::ArrayRef<int64_t> ranges,
                                         unsigned index) {
  if (candidate.reductionSplitSizes.empty())
    return ranges[index];
  return candidate.reductionSplitSizes[index];
}

static void addPressure(llvm::SmallVectorImpl<int64_t> &pressure,
                        unsigned index, int64_t bytes) {
  if (index >= pressure.size())
    return;
  pressure[index] = saturatingAdd(pressure[index], bytes);
}

static void addMatmulPressure(mlir::linalg::LinalgOp root,
                              const CandidateSpec &candidate,
                              llvm::SmallVectorImpl<int64_t> &traversal,
                              llvm::SmallVectorImpl<int64_t> &reduction) {
  if (candidate.tileSizes.size() != 2 || root.getNumDpsInputs() != 2)
    return;
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType)
    return;
  std::optional<int64_t> elementBytes = getElementByteWidth(resultType);
  if (!elementBytes)
    return;

  int64_t m = candidate.tileSizes[0];
  int64_t n = candidate.tileSizes[1];
  int64_t k = 1;
  if (!candidate.reductionSplitSizes.empty())
    k = candidate.reductionSplitSizes.front();
  else {
    auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(
        root.getDpsInputOperand(0)->get().getType());
    if (!lhsType || lhsType.getRank() != 2 || !lhsType.hasStaticShape())
      return;
    k = lhsType.getDimSize(1);
  }

  int64_t lhsBytes = saturatingMul(saturatingMul(m, k), *elementBytes);
  int64_t rhsBytes = saturatingMul(saturatingMul(k, n), *elementBytes);
  int64_t outBytes = saturatingMul(saturatingMul(m, n), *elementBytes);
  addPressure(traversal, 0, saturatingAdd(lhsBytes, outBytes));
  addPressure(traversal, 1, saturatingAdd(rhsBytes, outBytes));
  addPressure(reduction, 0, saturatingAdd(lhsBytes, rhsBytes));
}

static void addGenericRootPressure(mlir::linalg::LinalgOp root,
                                   const CandidateSpec &candidate,
                                   llvm::SmallVectorImpl<int64_t> &traversal) {
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
  if (!resultType)
    return;
  std::optional<int64_t> elementBytes = getElementByteWidth(resultType);
  if (!elementBytes)
    return;

  int64_t tileElements = 1;
  for (int64_t size : candidate.tileSizes)
    tileElements = saturatingMul(tileElements, size);
  int64_t bufferCount =
      std::max<int64_t>(1, root.getNumDpsInputs() + root.getNumDpsInits());
  int64_t bytes =
      saturatingMul(saturatingMul(tileElements, *elementBytes), bufferCount);
  for (unsigned dim = 0; dim < traversal.size(); ++dim)
    addPressure(traversal, dim, bytes);
}

static llvm::SmallVector<RefinementDim, 6> rankRefinementDims(
    GroupOp group, const CandidateSpec &candidate,
    llvm::ArrayRef<int64_t> reductionRanges) {
  llvm::SmallVector<int64_t, 4> traversalPressure(candidate.tileSizes.size(),
                                                  1);
  llvm::SmallVector<int64_t, 2> reductionPressure(reductionRanges.size(), 1);

  std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      getYieldedRootLinalgOps(group);
  if (roots) {
    for (mlir::linalg::LinalgOp root : *roots) {
      if (mlir::isa<mlir::linalg::MatmulOp>(root.getOperation())) {
        addMatmulPressure(root, candidate, traversalPressure,
                          reductionPressure);
        continue;
      }
      addGenericRootPressure(root, candidate, traversalPressure);
    }
  }

  llvm::SmallVector<RefinementDim, 6> dims;
  for (auto [index, pressure] : llvm::enumerate(traversalPressure))
    dims.push_back(RefinementDim{/*isReduction=*/false,
                                 static_cast<unsigned>(index), pressure});
  for (auto [index, pressure] : llvm::enumerate(reductionPressure))
    dims.push_back(RefinementDim{/*isReduction=*/true,
                                 static_cast<unsigned>(index), pressure});

  llvm::stable_sort(dims, [](const RefinementDim &lhs,
                             const RefinementDim &rhs) {
    if (lhs.pressure != rhs.pressure)
      return lhs.pressure > rhs.pressure;
    if (lhs.isReduction != rhs.isReduction)
      return lhs.isReduction;
    return lhs.index < rhs.index;
  });
  return dims;
}

static std::optional<CandidateSpec>
refineCandidateDim(const CandidateSpec &candidate,
                   llvm::ArrayRef<int64_t> reductionRanges,
                   const SearchLadders &ladders, const RefinementDim &dim) {
  CandidateSpec refined = candidate;
  if (!dim.isReduction) {
    if (dim.index >= refined.tileSizes.size() ||
        dim.index >= ladders.traversal.size())
      return std::nullopt;
    std::optional<size_t> index = findSizeIndex(
        ladders.traversal[dim.index], refined.tileSizes[dim.index]);
    if (!index || *index + 1 >= ladders.traversal[dim.index].size())
      return std::nullopt;
    refined.tileSizes[dim.index] = ladders.traversal[dim.index][*index + 1];
    return refined;
  }

  if (dim.index >= reductionRanges.size() ||
      dim.index >= ladders.reduction.size())
    return std::nullopt;
  if (refined.reductionSplitSizes.empty())
    refined.reductionSplitSizes.assign(reductionRanges.begin(),
                                       reductionRanges.end());

  std::optional<size_t> index =
      findSizeIndex(ladders.reduction[dim.index],
                    refined.reductionSplitSizes[dim.index]);
  if (!index || *index + 1 >= ladders.reduction[dim.index].size())
    return std::nullopt;
  refined.reductionSplitSizes[dim.index] =
      ladders.reduction[dim.index][*index + 1];
  return refined;
}

static void enqueueCandidate(
    const CandidateSpec &candidate, llvm::StringSet<> &seen,
    llvm::SmallVectorImpl<CandidateWorkItem> &queue) {
  std::string key = getCandidateKey(candidate);
  if (!seen.insert(key).second)
    return;
  queue.push_back(CandidateWorkItem{candidate});
}

static void enqueueRefinements(
    GroupOp group, const CandidateSpec &candidate,
    llvm::ArrayRef<int64_t> reductionRanges, const SearchLadders &ladders,
    llvm::StringSet<> &seen, llvm::SmallVectorImpl<CandidateWorkItem> &queue) {
  llvm::SmallVector<RefinementDim, 6> dims =
      rankRefinementDims(group, candidate, reductionRanges);

  llvm::SmallVector<CandidateSpec, 6> oneStep;
  for (const RefinementDim &dim : dims) {
    std::optional<CandidateSpec> refined =
        refineCandidateDim(candidate, reductionRanges, ladders, dim);
    if (!refined)
      continue;
    oneStep.push_back(*refined);
    enqueueCandidate(*refined, seen, queue);
  }

  // A single split can be insufficient when pressure comes from a product of
  // two dimensions.  Add a small combined-neighbor frontier without expanding
  // the full Cartesian product upfront.
  for (unsigned i = 0; i < std::min<size_t>(oneStep.size(), 3); ++i) {
    for (unsigned j = i + 1; j < std::min<size_t>(oneStep.size(), 3); ++j) {
      CandidateSpec combined = oneStep[i];
      for (auto [dimIndex, value] : llvm::enumerate(oneStep[j].tileSizes))
        if (value != candidate.tileSizes[dimIndex])
          combined.tileSizes[dimIndex] = value;
      if (!oneStep[j].reductionSplitSizes.empty()) {
        if (combined.reductionSplitSizes.empty())
          combined.reductionSplitSizes.assign(reductionRanges.begin(),
                                              reductionRanges.end());
        for (auto [dimIndex, value] :
             llvm::enumerate(oneStep[j].reductionSplitSizes))
          if (value != getCandidateReductionSize(candidate, reductionRanges,
                                                 static_cast<unsigned>(dimIndex)))
            combined.reductionSplitSizes[dimIndex] = value;
      }
      enqueueCandidate(combined, seen, queue);
    }
  }
}

static mlir::FailureOr<SelectedCandidate>
selectCandidateForGroup(GroupOp group, llvm::StringRef label,
                        const SelectionConfig &config) {
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> shape =
      getStaticTraversalShape(group);
  if (mlir::failed(shape)) {
    group.emitError() << "no_candidate: tile selection requires static ranked "
                         "group "
                         "results with one traversal shape";
    return mlir::failure();
  }

  mlir::FailureOr<llvm::SmallVector<int64_t, 2>> reductionRanges =
      getStaticRootReductionRanges(group);
  if (mlir::failed(reductionRanges)) {
    group.emitError()
        << "no_candidate: tile selection requires static reduction ranges";
    return mlir::failure();
  }

  SearchLadders ladders = buildSearchLadders(
      *shape, *reductionRanges, config.preferredTileSizes,
      config.maxCandidatesPerDim);

  std::optional<SelectedCandidate> best;
  int64_t rejectedCount = 0;
  std::string lastFailure;
  int64_t visitedCount = 0;
  llvm::StringSet<> seen;
  llvm::SmallVector<CandidateWorkItem, 32> queue;

  CandidateSpec initial;
  initial.tileSizes.assign(shape->begin(), shape->end());
  enqueueCandidate(initial, seen, queue);

  for (size_t queueIndex = 0; queueIndex < queue.size(); ++queueIndex) {
    const CandidateSpec &candidate = queue[queueIndex].spec;
    ++visitedCount;
    if (failsCheapSPMBound(group, candidate, config.spmBase, config.spmLimit)) {
      ++rejectedCount;
      lastFailure =
          "cheap_bound: minimum SPM bytes exceed planning window";
      enqueueRefinements(group, candidate, *reductionRanges, ladders, seen,
                         queue);
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

    if (!passed) {
      if (llvm::StringRef(lastFailure).contains("spm-offsets") ||
          llvm::StringRef(lastFailure).contains("capacity_overflow") ||
          llvm::StringRef(lastFailure).contains("tile-region") ||
          llvm::StringRef(lastFailure).contains("cheap_bound"))
        enqueueRefinements(group, candidate, *reductionRanges, ladders, seen,
                           queue);
      continue;
    }

    SelectedCandidate selected;
    selected.label = label.str();
    selected.group = group;
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
    enqueueRefinements(group, candidate, *reductionRanges, ladders, seen,
                       queue);
  }

  if (best) {
    best->candidateCount = visitedCount;
    best->rejectedCount = rejectedCount;
    return std::move(*best);
  }

  group.emitError() << "no_candidate: tile selection found no passing candidate"
                    << (lastFailure.empty() ? "" : "; last failure: ")
                    << lastFailure;
  return mlir::failure();
}

static void printSelectedSummary(const SelectedCandidate &selected,
                                 TileSearchMode mode) {
  llvm::errs() << "wafer.select_group_tile selected group " << selected.label
               << " mode="
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

static mlir::FailureOr<mlir::func::FuncOp>
getStandaloneSelectedFunction(GroupOp group, mlir::ModuleOp selectedModule) {
  mlir::func::FuncOp selectedFunc;
  for (auto func : selectedModule.getOps<mlir::func::FuncOp>()) {
    if (selectedFunc) {
      group.emitError()
          << "selected candidate commit expected one lowered function";
      return mlir::failure();
    }
    selectedFunc = func;
  }
  if (!selectedFunc) {
    group.emitError() << "selected candidate commit found no lowered function";
    return mlir::failure();
  }
  return selectedFunc;
}

static mlir::LogicalResult commitSelectedCandidate(SelectedCandidate &selected) {
  GroupOp group = selected.group;
  if (!group || !selected.module) {
    if (group)
      group.emitError() << "selected candidate commit missing lowered module";
    return mlir::failure();
  }

  mlir::FailureOr<mlir::func::FuncOp> selectedFunc =
      getStandaloneSelectedFunction(group, *selected.module);
  if (mlir::failed(selectedFunc))
    return mlir::failure();
  if (!selectedFunc->getBody().hasOneBlock()) {
    group.emitError() << "selected candidate commit requires one-block "
                         "lowered function";
    return mlir::failure();
  }

  mlir::Block &entry = selectedFunc->getBody().front();
  mlir::Operation *terminator = entry.getTerminator();
  if (!terminator) {
    group.emitError()
        << "selected candidate commit requires function terminator";
    return mlir::failure();
  }
  auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(terminator);
  if (!returnOp) {
    group.emitError()
        << "selected candidate commit requires func.return terminator";
    return mlir::failure();
  }

  unsigned expectedArgCount =
      static_cast<unsigned>(group.getInputs().size() + group.getOuts().size());
  if (entry.getNumArguments() != expectedArgCount) {
    group.emitError() << "selected candidate commit argument count mismatch";
    return mlir::failure();
  }
  if (returnOp.getNumOperands() != group->getNumResults()) {
    group.emitError() << "selected candidate commit result count mismatch";
    return mlir::failure();
  }

  mlir::IRMapping mapping;
  unsigned argumentIndex = 0;
  for (mlir::Value input : group.getInputs())
    mapping.map(entry.getArgument(argumentIndex++), input);
  for (mlir::Value output : group.getOuts())
    mapping.map(entry.getArgument(argumentIndex++), output);

  mlir::OpBuilder builder(group.getOperation());
  for (mlir::Operation &op : entry.getOperations()) {
    if (&op == terminator)
      break;
    builder.clone(op, mapping);
  }

  llvm::SmallVector<mlir::Value, 2> replacements;
  replacements.reserve(returnOp.getNumOperands());
  for (mlir::Value returned : returnOp.getOperands()) {
    mlir::Value mapped = mapping.lookupOrNull(returned);
    if (!mapped) {
      group.emitError()
          << "selected candidate commit could not map returned value";
      return mlir::failure();
    }
    replacements.push_back(mapped);
  }

  group->replaceAllUsesWith(replacements);
  group->erase();
  return mlir::success();
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
          selectCandidateForGroup(group, labelOs.str(), config);
      if (mlir::failed(selected)) {
        signalPassFailure();
        return;
      }
      if (printCandidateSummary)
        printSelectedSummary(*selected, config.mode);
      selectedCandidates.push_back(std::move(*selected));
    }

    for (SelectedCandidate &selected : selectedCandidates) {
      if (mlir::failed(commitSelectedCandidate(selected))) {
        signalPassFailure();
        return;
      }
    }

    if (mlir::failed(mlir::verify(getOperation()))) {
      getOperation()->emitError()
          << "selected candidate commit produced invalid IR";
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace wafer
