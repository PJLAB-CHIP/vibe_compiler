//===- CandidateAnalysis.cpp - Tile candidate implementation
//-----------------===//

#include "Group/SelectGroupTileInternal.h"

namespace wafer::group_tile_selection {

std::string getNearestSymbolName(mlir::Operation *op) {
  for (mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto name = parent->getAttrOfType<mlir::StringAttr>("sym_name"))
      return ("@" + name.getValue()).str();
  }
  return "@unknown";
}

void printI64List(llvm::ArrayRef<int64_t> values, llvm::raw_ostream &os) {
  os << "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index != 0)
      os << ",";
    os << value;
  }
  os << "]";
}

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
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

mlir::FailureOr<TileSearchMode> parseTileSearchMode(llvm::StringRef text,
                                                    mlir::Operation *anchor) {
  if (text == "first-legal")
    return TileSearchMode::FirstLegal;
  if (text == "min-estimated-time")
    return TileSearchMode::MinEstimatedTime;
  anchor->emitError()
      << "invalid_tile_search: expected first-legal or min-estimated-time";
  return mlir::failure();
}

mlir::FailureOr<TileSearchEffort>
parseTileSearchEffort(llvm::StringRef text, mlir::Operation *anchor) {
  if (text == "quick")
    return TileSearchEffort::Quick;
  if (text == "default")
    return TileSearchEffort::Default;
  if (text == "deep")
    return TileSearchEffort::Deep;
  anchor->emitError()
      << "invalid_tile_search_effort: expected quick, default or deep";
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

int64_t saturatingAdd(int64_t lhs, int64_t rhs) {
  int64_t result = 0;
  if (!checkedAdd(lhs, rhs, result))
    return std::numeric_limits<int64_t>::max();
  return result;
}

int64_t saturatingMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return std::numeric_limits<int64_t>::max();
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return std::numeric_limits<int64_t>::max();
  return lhs * rhs;
}

int64_t ceilDiv(int64_t numerator, int64_t denominator) {
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

CandidateStats estimateStats(mlir::ModuleOp module) {
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

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
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

std::optional<int64_t> getElementByteWidth(mlir::Type type) {
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    type = shaped.getElementType();
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type))
    return std::max<int64_t>(1, floatType.getWidth() / 8);
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return std::max<int64_t>(1, (intType.getWidth() + 7) / 8);
  return std::nullopt;
}

std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>>
getYieldedRootLinalgOps(GroupOp group) {
  auto yield =
      mlir::dyn_cast<GroupYieldOp>(group.getBody().front().getTerminator());
  if (!yield || yield.getValues().empty())
    return std::nullopt;

  llvm::SmallVector<mlir::linalg::LinalgOp, 4> roots;
  for (mlir::Value value : yield.getValues()) {
    auto root =
        mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(value.getDefiningOp());
    if (!root)
      return std::nullopt;
    roots.push_back(root);
  }
  return roots;
}

std::string getStandaloneGroupModuleText(GroupOp group) {
  mlir::OwningOpRef<mlir::ModuleOp> standaloneModule =
      detail::cloneGroupToStandaloneModule(group);
  std::string text;
  llvm::raw_string_ostream os(text);
  standaloneModule->print(os);
  return os.str();
}

GroupOp findSingleSelectionGroup(mlir::ModuleOp module) {
  GroupOp found;
  module.walk([&](GroupOp group) {
    if (!found)
      found = group;
  });
  return found;
}

static bool hasReductionIterator(mlir::linalg::LinalgOp op) {
  for (mlir::utils::IteratorType iteratorType : op.getIteratorTypesArray()) {
    if (iteratorType == mlir::utils::IteratorType::reduction)
      return true;
  }
  return false;
}

mlir::FailureOr<llvm::SmallVector<int64_t, 2>>
getStaticRootReductionRanges(GroupOp group) {
  std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      getYieldedRootLinalgOps(group);
  if (!roots)
    return llvm::SmallVector<int64_t, 2>{};

  std::optional<llvm::SmallVector<int64_t, 2>> commonReductionRanges;
  for (mlir::linalg::LinalgOp root : *roots) {
    if (!mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp,
                   mlir::linalg::GenericOp>(root.getOperation()) ||
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

static std::optional<int64_t>
estimateRootMinimumSPMBytes(mlir::linalg::LinalgOp root,
                            const CandidateSpec &candidate) {
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

bool failsCheapSPMBound(GroupOp group, const CandidateSpec &candidate,
                        int64_t spmBase, int64_t spmLimit) {
  std::optional<int64_t> lowerBoundBytes =
      estimateRequiredSPMLowerBoundBytes(group, candidate);
  if (!lowerBoundBytes)
    return false;
  if (spmLimit <= spmBase)
    return false;
  return *lowerBoundBytes > (spmLimit - spmBase);
}

std::optional<std::string>
getCheapTargetGeometryFailure(GroupOp group, const CandidateSpec &candidate,
                              llvm::ArrayRef<int64_t> reductionRanges) {
  std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots =
      getYieldedRootLinalgOps(group);
  if (!roots)
    return std::nullopt;

  constexpr int64_t maxTargetDimension = std::numeric_limits<uint16_t>::max();
  auto exceedsTargetDimension = [&](llvm::ArrayRef<int64_t> dimensions) {
    return llvm::any_of(dimensions, [&](int64_t dimension) {
      return dimension > maxTargetDimension;
    });
  };

  for (mlir::linalg::LinalgOp root : *roots) {
    bool isGemm =
        mlir::isa<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp>(
            root.getOperation());
    bool isReduction = hasReductionIterator(root);
    if (!isGemm && !isReduction)
      continue;

    if (exceedsTargetDimension(candidate.tileSizes))
      return "target_abi_narrowing: candidate traversal dimension must fit "
             "uint16_t";

    llvm::ArrayRef<int64_t> reductionSizes =
        candidate.reductionSplitSizes.empty()
            ? reductionRanges
            : llvm::ArrayRef<int64_t>(candidate.reductionSplitSizes);
    if (exceedsTargetDimension(reductionSizes))
      return "target_abi_narrowing: candidate reduction dimension must fit "
             "uint16_t";
  }
  return std::nullopt;
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

TileSizeOptions buildTileSizeOptions(llvm::ArrayRef<int64_t> traversalShape,
                                     llvm::ArrayRef<int64_t> reductionRanges,
                                     llvm::ArrayRef<int64_t> preferred,
                                     int64_t maxCandidatesPerDim) {
  TileSizeOptions options;
  for (int64_t dim : traversalShape)
    options.traversal.push_back(
        buildDimTileSizes(dim, preferred, maxCandidatesPerDim));
  (void)reductionRanges;
  return options;
}

bool isFullFirstTile(llvm::ArrayRef<int64_t> traversalShape,
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

llvm::SmallVector<TileInstance, 8>
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

} // namespace wafer::group_tile_selection
