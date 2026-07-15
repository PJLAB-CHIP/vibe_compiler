//===- CandidateSelection.cpp - Tile candidate implementation
//-----------------===//

#include "Group/SelectGroupTileInternal.h"

namespace wafer::group_tile_selection {

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

static llvm::SmallVector<RefinementDim, 6>
rankRefinementDims(GroupOp group, const CandidateSpec &candidate,
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

  llvm::stable_sort(dims,
                    [](const RefinementDim &lhs, const RefinementDim &rhs) {
                      if (lhs.pressure != rhs.pressure)
                        return lhs.pressure > rhs.pressure;
                      if (lhs.isReduction != rhs.isReduction)
                        return lhs.isReduction;
                      return lhs.index < rhs.index;
                    });
  return dims;
}

static std::optional<CandidateSpec> refineCandidateDim(
    const CandidateSpec &candidate, llvm::ArrayRef<int64_t> reductionRanges,
    const TileSizeOptions &tileSizeOptions, const RefinementDim &dim) {
  CandidateSpec refined = candidate;
  if (!dim.isReduction) {
    if (dim.index >= refined.tileSizes.size() ||
        dim.index >= tileSizeOptions.traversal.size())
      return std::nullopt;
    std::optional<size_t> index = findSizeIndex(
        tileSizeOptions.traversal[dim.index], refined.tileSizes[dim.index]);
    if (!index || *index + 1 >= tileSizeOptions.traversal[dim.index].size())
      return std::nullopt;
    refined.tileSizes[dim.index] =
        tileSizeOptions.traversal[dim.index][*index + 1];
    return refined;
  }

  (void)reductionRanges;
  return std::nullopt;
}

static bool enqueueCandidate(const CandidateSpec &candidate,
                             llvm::StringSet<> &seen,
                             llvm::SmallVectorImpl<CandidateWorkItem> &queue) {
  std::string key = getCandidateKey(candidate);
  if (!seen.insert(key).second)
    return false;
  queue.push_back(CandidateWorkItem{candidate});
  return true;
}

static void enqueueRefinements(GroupOp group, const CandidateSpec &candidate,
                               llvm::ArrayRef<int64_t> reductionRanges,
                               const TileSizeOptions &tileSizeOptions,
                               llvm::StringSet<> &seen,
                               llvm::SmallVectorImpl<CandidateWorkItem> &queue,
                               int64_t beamWidth) {
  llvm::SmallVector<RefinementDim, 6> dims =
      rankRefinementDims(group, candidate, reductionRanges);

  llvm::SmallVector<CandidateSpec, 6> oneStep;
  llvm::SmallVector<CandidateSpec, 8> neighbors;
  for (const RefinementDim &dim : dims) {
    std::optional<CandidateSpec> refined =
        refineCandidateDim(candidate, reductionRanges, tileSizeOptions, dim);
    if (!refined)
      continue;
    oneStep.push_back(*refined);
    neighbors.push_back(*refined);
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
      neighbors.push_back(combined);
    }
  }

  int64_t enqueued = 0;
  for (const CandidateSpec &neighbor : neighbors) {
    if (beamWidth > 0 && enqueued >= beamWidth)
      break;
    if (enqueueCandidate(neighbor, seen, queue))
      ++enqueued;
  }
}

mlir::FailureOr<SelectedCandidate>
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

  TileSizeOptions tileSizeOptions =
      buildTileSizeOptions(*shape, *reductionRanges, config.preferredTileSizes,
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

  auto buildSelected = [&](CandidateCheckResult &check,
                           int64_t currentVisited) {
    SelectedCandidate selected;
    selected.label = label.str();
    selected.group = group;
    selected.spec = check.spec;
    selected.stats = check.stats;
    selected.estimatedCycles = estimateCycles(
        selected.stats, computeTileCount(*shape, check.spec.tileSizes),
        config.computeOpsPerCycle, config.ddrBytesPerCycle,
        config.spmBytesPerCycle, config.instrIssueCycles,
        config.assumeDdrComputeOverlap);
    selected.candidateCount = currentVisited;
    selected.rejectedCount = rejectedCount;
    selected.representativeCount = check.representativeCount;
    selected.module = std::move(check.module);
    selected.artifactSource = check.artifactSource;
    return selected;
  };

  auto isRetryableFailure = [](llvm::StringRef failure) {
    return failure.starts_with("cheap_bound:") ||
           failure.contains("capacity_overflow") ||
           failure.starts_with("tile-bounds:") ||
           failure.starts_with("tile-demand:") ||
           failure.starts_with("target_abi_narrowing:");
  };

  auto rejectCandidate = [&](const CandidateSpec &candidate,
                             llvm::StringRef failure) {
    ++rejectedCount;
    lastFailure = failure.str();
    if (isRetryableFailure(lastFailure))
      enqueueRefinements(group, candidate, *reductionRanges, tileSizeOptions,
                         seen, queue, best ? config.searchBeamWidth : 0);
  };

  auto processCheckResult = [&](CandidateCheckResult &check) {
    if (!check.failureReason.empty()) {
      rejectCandidate(check.spec, check.failureReason);
      return;
    }
    if (!isCompleteArtifactSource(check.artifactSource)) {
      rejectCandidate(
          check.spec,
          "complete-artifact: candidate passed without complete provenance");
      return;
    }

    SelectedCandidate selected = buildSelected(check, visitedCount);
    if (isBetterCandidate(selected, best ? &*best : nullptr)) {
      if (!selected.module) {
        CandidateEvaluation accepted =
            evaluateCompleteCandidate(group, *shape, selected.spec, config);
        if (!accepted.failureReason.empty()) {
          rejectCandidate(selected.spec, accepted.failureReason);
          return;
        }
        selected.module = std::move(accepted.module);
        selected.artifactSource = accepted.artifactSource;
      }
      best = std::move(selected);
    }
    enqueueRefinements(group, check.spec, *reductionRanges, tileSizeOptions,
                       seen, queue, config.searchBeamWidth);
  };

  size_t queueIndex = 0;
  std::string standaloneGroupModuleText;
  if (config.mode == TileSearchMode::MinEstimatedTime &&
      config.candidateParallelism > 1)
    standaloneGroupModuleText = getStandaloneGroupModuleText(group);

  while (queueIndex < queue.size()) {
    if (best && visitedCount >= config.maxSearchCandidates)
      break;

    if (config.mode == TileSearchMode::MinEstimatedTime &&
        config.candidateParallelism > 1) {
      size_t remainingBudget =
          best ? static_cast<size_t>(config.maxSearchCandidates - visitedCount)
               : std::numeric_limits<size_t>::max();
      size_t batchSize =
          std::min<size_t>({static_cast<size_t>(config.candidateParallelism),
                            queue.size() - queueIndex, remainingBudget});
      llvm::SmallVector<CandidateCheckResult, 8> results;
      results.resize(batchSize);
      llvm::SmallVector<unsigned, 8> futureSlots;
      std::vector<std::future<CandidateCheckResult>> futures;

      for (size_t batchOffset = 0; batchOffset < batchSize; ++batchOffset) {
        const CandidateSpec candidate = queue[queueIndex + batchOffset].spec;
        ++visitedCount;
        if (std::optional<std::string> failure = getCheapTargetGeometryFailure(
                group, candidate, *reductionRanges)) {
          CandidateCheckResult result;
          result.spec = candidate;
          result.failureReason = std::move(*failure);
          results[batchOffset] = std::move(result);
          continue;
        }
        if (failsCheapSPMBound(group, candidate, config.spmBase,
                               config.spmLimit)) {
          CandidateCheckResult result;
          result.spec = candidate;
          result.failureReason =
              "cheap_bound: minimum SPM bytes exceed planning window";
          results[batchOffset] = std::move(result);
          continue;
        }

        futureSlots.push_back(static_cast<unsigned>(batchOffset));
        futures.push_back(std::async(
            std::launch::async,
            [&standaloneGroupModuleText, shape = *shape, candidate, config]() {
              return evaluateCandidateOnStandaloneText(
                  standaloneGroupModuleText, shape, candidate, config);
            }));
      }

      for (auto [futureIndex, slot] : llvm::enumerate(futureSlots))
        results[slot] = futures[futureIndex].get();

      queueIndex += batchSize;
      for (CandidateCheckResult &result : results)
        processCheckResult(result);
      continue;
    }

    const CandidateSpec &candidate = queue[queueIndex].spec;
    ++queueIndex;
    ++visitedCount;
    if (std::optional<std::string> failure =
            getCheapTargetGeometryFailure(group, candidate, *reductionRanges)) {
      rejectCandidate(candidate, *failure);
      continue;
    }
    if (failsCheapSPMBound(group, candidate, config.spmBase, config.spmLimit)) {
      rejectCandidate(candidate,
                      "cheap_bound: minimum SPM bytes exceed planning window");
      continue;
    }
    CandidateCheckResult check =
        evaluateCandidateOnOriginalGroup(group, *shape, candidate, config);
    if (!check.failureReason.empty()) {
      rejectCandidate(candidate, check.failureReason);
      continue;
    }
    if (!isCompleteArtifactSource(check.artifactSource) || !check.module) {
      rejectCandidate(
          candidate,
          "complete-artifact: candidate passed without accepted module");
      continue;
    }

    SelectedCandidate selected = buildSelected(check, visitedCount);

    if (config.mode == TileSearchMode::FirstLegal)
      return selected;

    if (isBetterCandidate(selected, best ? &*best : nullptr))
      best = std::move(selected);
    enqueueRefinements(group, candidate, *reductionRanges, tileSizeOptions,
                       seen, queue, config.searchBeamWidth);
  }

  if (best) {
    best->candidateCount = visitedCount;
    best->rejectedCount = rejectedCount;
    if (!best->module || !isCompleteArtifactSource(best->artifactSource)) {
      CandidateEvaluation accepted =
          evaluateCompleteCandidate(group, *shape, best->spec, config);
      if (!accepted.failureReason.empty()) {
        group.emitError() << "selected candidate complete artifact failed: "
                          << accepted.failureReason;
        return mlir::failure();
      }
      best->module = std::move(accepted.module);
      best->artifactSource = accepted.artifactSource;
    }
    return std::move(*best);
  }

  group.emitError() << "no_candidate: tile selection found no passing candidate"
                    << (lastFailure.empty() ? "" : "; last failure: ")
                    << lastFailure;
  return mlir::failure();
}

} // namespace wafer::group_tile_selection
