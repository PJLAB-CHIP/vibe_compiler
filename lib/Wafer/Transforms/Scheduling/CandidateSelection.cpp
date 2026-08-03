//===- CandidateSelection.cpp - Tile candidate implementation
//-----------------===//

#include "Scheduling/ScheduleTensorProgramInternal.h"

namespace wafer::tensor_program_scheduling {

static int64_t computeTileCount(llvm::ArrayRef<int64_t> traversalShape,
                                llvm::ArrayRef<int64_t> tileSizes) {
  int64_t count = 1;
  for (auto [dim, tile] : llvm::zip(traversalShape, tileSizes))
    count = saturatingMul(count, ceilDiv(dim, tile));
  return count;
}

std::optional<std::string> getRankingCostFailure(const CandidateStats &stats) {
  const analysis::InstructionProgramCost &cost = stats.program;
  struct NamedMetric {
    llvm::StringLiteral name;
    const analysis::ScheduleCostMetric *metric;
  };
  const NamedMetric required[] = {
      {"NPU f16/bf16 logical ops", &cost.compute.npuF16Bf16LogicalOps},
      {"NPU other logical ops", &cost.compute.npuOtherLogicalOps},
      {"vector f16/bf16 logical ops", &cost.compute.vectorF16Bf16LogicalOps},
      {"vector f32 logical ops", &cost.compute.vectorF32LogicalOps},
      {"vector other logical ops", &cost.compute.vectorOtherLogicalOps},
      {"DDR read bytes", &cost.ddrReadBytes},
      {"DDR write bytes", &cost.ddrWriteBytes},
      {"SPM movement bytes", &cost.spmMovementBytes},
      {"NoC transmit bytes", &cost.noc.aggregateTransmitBytes},
      {"NoC receive bytes", &cost.noc.aggregateReceiveBytes},
      {"instruction count", &cost.instructionCount},
      {"event count", &cost.eventCount},
      {"NCC join count", &cost.nccJoinCount},
      {"steady-state NCC join count", &cost.steadyStateNCCJoinCount},
      {"non-terminal NCC join count", &cost.nonTerminalNCCJoinCount},
      {"NCC participant wait count", &cost.nccParticipantWaitCount},
      {"steady-state NCC participant wait count",
       &cost.steadyStateNCCParticipantWaitCount},
      {"non-terminal NCC participant wait count",
       &cost.nonTerminalNCCParticipantWaitCount},
      {"intrinsic NCC drain count", &cost.intrinsicNCCDrainCount},
  };
  for (const NamedMetric &entry : required) {
    if (entry.metric->isKnown())
      continue;
    std::string failure;
    llvm::raw_string_ostream os(failure);
    os << "ranking-cost: " << entry.name << " is "
       << analysis::stringifyScheduleCostKnowledge(entry.metric->knowledge)
       << " (" << analysis::stringifyScheduleCostReason(entry.metric->reason)
       << ")";
    return failure;
  }
  return std::nullopt;
}

static std::string getCandidateKey(const CandidateSpec &candidate) {
  std::string key;
  llvm::raw_string_ostream os(key);
  printI64List(candidate.tileSizes, os);
  os << "|";
  printI64List(candidate.reductionSplitSizes, os);
  os << "|";
  if (candidate.selectedImplementationAlternative)
    os << stringifyTargetImplementationKind(
        *candidate.selectedImplementationAlternative);
  else
    os << "baseline";
  os << "|" << static_cast<unsigned>(candidate.traversalKind);
  return os.str();
}

static llvm::SmallVector<TargetImplementationKind, 2>
collectImplementationAlternatives(mlir::func::FuncOp task) {
  llvm::SmallVector<TargetImplementationKind, 2> alternatives;
  task.walk([&](WaferTargetImplementationOpInterface interface) {
    llvm::SmallVector<TargetImplementationCandidate, 2> candidates;
    interface.collectTargetImplementationCandidates(WaferTargetCapabilities{},
                                                    candidates);
    for (const TargetImplementationCandidate &candidate :
         llvm::drop_begin(candidates)) {
      if (!llvm::is_contained(alternatives, candidate.kind))
        alternatives.push_back(candidate.kind);
    }
  });
  return alternatives;
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

static llvm::SmallVector<RefinementDim, 6> rankRefinementDimsImpl(
    std::optional<llvm::SmallVector<mlir::linalg::LinalgOp, 4>> roots,
    const CandidateSpec &candidate, llvm::ArrayRef<int64_t> reductionRanges) {
  llvm::SmallVector<int64_t, 4> traversalPressure(candidate.tileSizes.size(),
                                                  1);
  llvm::SmallVector<int64_t, 2> reductionPressure(reductionRanges.size(), 1);

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
  if (reductionRanges.size() == 1)
    dims.push_back(RefinementDim{/*isReduction=*/true, /*index=*/0,
                                 reductionPressure.front()});

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

static llvm::SmallVector<RefinementDim, 6>
rankRefinementDims(mlir::func::FuncOp task, const CandidateSpec &candidate,
                   llvm::ArrayRef<int64_t> reductionRanges) {
  return rankRefinementDimsImpl(getTraversalComputeRootLinalgOps(task),
                                candidate, reductionRanges);
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

  if (reductionRanges.size() != 1 || dim.index != 0 ||
      tileSizeOptions.reduction.size() != 1)
    return std::nullopt;
  int64_t current = candidate.reductionSplitSizes.empty()
                        ? reductionRanges.front()
                        : candidate.reductionSplitSizes.front();
  std::optional<size_t> index =
      findSizeIndex(tileSizeOptions.reduction.front(), current);
  if (!index || *index + 1 >= tileSizeOptions.reduction.front().size())
    return std::nullopt;
  refined.reductionSplitSizes.assign(
      1, tileSizeOptions.reduction.front()[*index + 1]);
  return refined;
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

static void enqueueRefinements(mlir::func::FuncOp task,
                               const CandidateSpec &candidate,
                               llvm::ArrayRef<int64_t> reductionRanges,
                               const TileSizeOptions &tileSizeOptions,
                               llvm::StringSet<> &seen,
                               llvm::SmallVectorImpl<CandidateWorkItem> &queue,
                               int64_t beamWidth) {
  llvm::SmallVector<RefinementDim, 6> dims =
      rankRefinementDims(task, candidate, reductionRanges);
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
  for (unsigned i = 0; i < std::min<size_t>(oneStep.size(), 3); ++i) {
    for (unsigned j = i + 1; j < std::min<size_t>(oneStep.size(), 3); ++j) {
      CandidateSpec combined = oneStep[i];
      for (auto [dimIndex, value] : llvm::enumerate(oneStep[j].tileSizes))
        if (value != candidate.tileSizes[dimIndex])
          combined.tileSizes[dimIndex] = value;
      if (oneStep[j].reductionSplitSizes != candidate.reductionSplitSizes)
        combined.reductionSplitSizes = oneStep[j].reductionSplitSizes;
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

static std::optional<CandidateSpec>
findCapacityDirectedSeed(mlir::func::FuncOp task, const CandidateSpec &initial,
                         llvm::ArrayRef<int64_t> reductionRanges,
                         const TileSizeOptions &tileSizeOptions,
                         const SelectionConfig &config) {
  if (config.spmLimit <= config.spmBase)
    return std::nullopt;
  int64_t capacityBytes = config.spmLimit - config.spmBase;
  if (config.spmWorkingSetMultiplicity == 0)
    return std::nullopt;
  capacityBytes /= static_cast<int64_t>(config.spmWorkingSetMultiplicity);
  if (capacityBytes <= 0)
    return std::nullopt;
  std::optional<int64_t> initialBytes =
      estimateSearchSPMWorkingSetBytes(task, initial, config.spmAlignment);
  if (!initialBytes || *initialBytes <= capacityBytes)
    return std::nullopt;

  CandidateSpec current = initial;
  while (true) {
    llvm::SmallVector<RefinementDim, 6> dims =
        rankRefinementDims(task, current, reductionRanges);
    std::optional<CandidateSpec> refined;
    for (const RefinementDim &dim : dims) {
      // Capacity seeding changes traversal shape only. Reduction candidates
      // remain exclusively controlled by the source numeric-legality menu and
      // the ordinary bounded search.
      if (dim.isReduction)
        continue;
      refined =
          refineCandidateDim(current, reductionRanges, tileSizeOptions, dim);
      if (refined)
        break;
    }
    if (!refined)
      return std::nullopt;

    std::optional<int64_t> refinedBytes =
        estimateSearchSPMWorkingSetBytes(task, *refined, config.spmAlignment);
    if (!refinedBytes)
      return std::nullopt;
    if (*refinedBytes <= capacityBytes)
      return refined;
    current = std::move(*refined);
  }
}

static std::optional<std::string>
getSearchSPMHeadroomFailure(mlir::func::FuncOp task,
                            const CandidateSpec &candidate,
                            const SelectionConfig &config) {
  if (config.spmWorkingSetMultiplicity <= 1 ||
      config.spmLimit <= config.spmBase)
    return std::nullopt;
  int64_t capacityBytes = config.spmLimit - config.spmBase;
  capacityBytes /= static_cast<int64_t>(config.spmWorkingSetMultiplicity);
  std::optional<int64_t> workingSet =
      estimateSearchSPMWorkingSetBytes(task, candidate, config.spmAlignment);
  if (!workingSet || *workingSet <= capacityBytes)
    return std::nullopt;
  return "search_spm_headroom: modeled working set exceeds the selected "
         "concurrent-residency budget";
}

static mlir::FailureOr<SelectedCandidate> selectCandidateForTask(
    const structured_scheduler::StructuredSchedulingScope &scope,
    mlir::func::FuncOp task, llvm::StringRef label,
    const SelectionConfig &config) {
  mlir::Operation *anchor = scope.insertionPoint;
  if (!anchor || !task) {
    if (anchor)
      anchor->emitError("no_candidate: scheduling task boundary is invalid");
    return mlir::failure();
  }
  mlir::FailureOr<llvm::SmallVector<int64_t, 4>> shape =
      getStaticTraversalShape(task);
  if (mlir::failed(shape)) {
    anchor->emitError()
        << "no_candidate: tile selection requires static ranked task results "
           "with one traversal shape";
    return mlir::failure();
  }
  CandidateTraversalRootCapability traversalRootCapability =
      getTaskTraversalRootCapability(task);
  if (traversalRootCapability ==
      CandidateTraversalRootCapability::Unsupported) {
    anchor->emitError()
        << "no_candidate: candidate traversal root is unsupported";
    return mlir::failure();
  }
  bool supportsTiledTraversal =
      traversalRootCapability == CandidateTraversalRootCapability::Tiled;
  mlir::FailureOr<llvm::SmallVector<int64_t, 2>> reductionRanges =
      getStaticRootReductionRanges(task, config.traversalKind);
  if (mlir::failed(reductionRanges)) {
    anchor->emitError()
        << "no_candidate: tile selection requires static reduction ranges";
    return mlir::failure();
  }
  std::optional<std::string> reductionSplitLegalityFailure =
      getReductionSplitLegalityFailure(task, config.traversalKind);
  llvm::SmallVector<TargetImplementationKind, 2> implementationAlternatives =
      collectImplementationAlternatives(task);
  bool forcedImplementationApplies =
      config.forcedImplementationAlternative &&
      llvm::is_contained(implementationAlternatives,
                         *config.forcedImplementationAlternative);
  TileSizeOptions tileSizeOptions;
  if (supportsTiledTraversal)
    tileSizeOptions = buildTileSizeOptions(
        *shape, *reductionRanges, config.preferredTileSizes,
        config.maxCandidatesPerDim, !reductionSplitLegalityFailure);

  llvm::SmallVector<SelectedCandidate, 8> candidateFrontier;
  int64_t rejectedCount = 0;
  std::string lastFailure;
  int64_t visitedCount = 0;
  int64_t completeEvaluationCount = 0;
  llvm::StringSet<> seen;
  llvm::SmallVector<CandidateWorkItem, 32> queue;
  CandidateSpec initial;
  initial.tileSizes.assign(shape->begin(), shape->end());
  initial.traversalKind = config.traversalKind;
  if (config.traversalKind == CandidateTileTraversalKind::PartialReduction &&
      !reductionRanges->empty() && !reductionSplitLegalityFailure)
    initial.reductionSplitSizes.assign(reductionRanges->begin(),
                                       reductionRanges->end());
  if (forcedImplementationApplies)
    initial.selectedImplementationAlternative =
        config.forcedImplementationAlternative;
  std::optional<CandidateSpec> capacitySeed;
  if (supportsTiledTraversal) {
    capacitySeed = findCapacityDirectedSeed(task, initial, *reductionRanges,
                                            tileSizeOptions, config);
  }
  // The target-physical estimator is available only for a closed direct or
  // exact passthrough-transpose named matmul. Visit its first capacity-fitting
  // menu entry before spending a hard-cap slot on the full candidate. The
  // fused form is a conservative seven-root inventory, so keep the full
  // candidate in the queue for exhaustive ranking modes; accepted artifacts
  // still pass every complete gate.
  if (capacitySeed)
    enqueueCandidate(*capacitySeed, seen, queue);
  enqueueCandidate(initial, seen, queue);
  if (config.allowAutomaticImplementationAlternatives &&
      !config.forcedImplementationAlternative)
    for (TargetImplementationKind implementation : implementationAlternatives) {
      CandidateSpec alternative = initial;
      alternative.selectedImplementationAlternative = implementation;
      enqueueCandidate(alternative, seen, queue);
    }

  auto buildSelected = [&](CandidateCheckResult &check,
                           int64_t currentVisited) {
    SelectedCandidate selected;
    selected.label = label.str();
    selected.scope = scope;
    selected.sourceTask = task;
    selected.spec = check.spec;
    selected.stats = check.stats;
    selected.candidateCount = currentVisited;
    selected.completeEvaluationCount = completeEvaluationCount;
    selected.rejectedCount = rejectedCount;
    selected.representativeCount = check.representativeCount;
    selected.module = std::move(check.module);
    selected.artifactSource = check.artifactSource;
    return selected;
  };
  unsigned frontierLimit = static_cast<unsigned>(std::max<int64_t>(
      {1, config.searchBeamWidth,
       static_cast<int64_t>(config.taskAlternativeOrdinal) + 1}));
  const size_t requiredPassingCandidateCount =
      static_cast<size_t>(config.taskAlternativeOrdinal) + 1;
  auto hasRequestedAlternative = [&]() {
    return candidateFrontier.size() >= requiredPassingCandidateCount;
  };
  auto insertCandidate = [&](SelectedCandidate selected) {
    if (candidateFrontier.size() < frontierLimit)
      candidateFrontier.push_back(std::move(selected));
  };
  auto isRetryableFailure = [](llvm::StringRef failure) {
    return failure.starts_with("cheap_bound:") ||
           failure.starts_with("search_spm_headroom:") ||
           failure.starts_with("target_spm_bound:") ||
           failure.contains("capacity_overflow") ||
           failure.contains("static_terminal_budget_exceeded:") ||
           failure.starts_with("tile-bounds:") ||
           failure.starts_with("tile-demand:") ||
           failure.starts_with("target_abi_narrowing:");
  };
  auto rejectCandidate = [&](const CandidateSpec &candidate,
                             llvm::StringRef failure) {
    ++rejectedCount;
    lastFailure = failure.str();
    if (config.printCandidateSummary) {
      llvm::errs() << "wafer.schedule_tensor_program rejected task " << label
                   << " tile=";
      printI64List(candidate.tileSizes, llvm::errs());
      llvm::errs() << " split=";
      printI64List(candidate.reductionSplitSizes, llvm::errs());
      llvm::errs() << " reason=" << failure << "\n";
    }
    if (supportsTiledTraversal && isRetryableFailure(lastFailure))
      enqueueRefinements(
          task, candidate, *reductionRanges, tileSizeOptions, seen, queue,
          candidateFrontier.empty() ? 0 : config.searchBeamWidth);
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
    if (!check.module &&
        mlir::failed(importAcceptedCandidateModule(
            check, *task.getContext(), config.scheduleCostPolicy))) {
      rejectCandidate(check.spec, check.failureReason);
      return;
    }
    if (!check.module) {
      rejectCandidate(
          check.spec,
          "complete-artifact: candidate passed without accepted module");
      return;
    }
    if (std::optional<std::string> failure =
            getRankingCostFailure(check.stats)) {
      rejectCandidate(check.spec, *failure);
      return;
    }
    SelectedCandidate selected = buildSelected(check, visitedCount);
    if (candidateFrontier.size() < frontierLimit)
      insertCandidate(std::move(selected));
    if (hasRequestedAlternative())
      return;
    if (supportsTiledTraversal)
      enqueueRefinements(task, check.spec, *reductionRanges, tileSizeOptions,
                         seen, queue, config.searchBeamWidth);
  };

  size_t queueIndex = 0;
  std::shared_ptr<const std::string> standaloneTaskModuleText;
  std::unique_ptr<CandidateEvaluationExecutor> fallbackEvaluationExecutor;
  CandidateEvaluationExecutor *evaluationExecutor = config.evaluationExecutor;
  if (config.candidateParallelism > 1) {
    standaloneTaskModuleText =
        std::make_shared<const std::string>(getStandaloneTaskModuleText(task));
    if (!evaluationExecutor) {
      fallbackEvaluationExecutor =
          std::make_unique<CandidateEvaluationExecutor>(
              static_cast<unsigned>(config.candidateParallelism));
      evaluationExecutor = fallbackEvaluationExecutor.get();
    }
  }

  while (queueIndex < queue.size()) {
    if (hasRequestedAlternative())
      break;
    if (config.maxSearchCandidates > 0 &&
        visitedCount >= config.maxSearchCandidates)
      break;
    if (config.candidateParallelism > 1) {
      size_t remainingBudget =
          config.maxSearchCandidates > 0
              ? static_cast<size_t>(config.maxSearchCandidates - visitedCount)
              : std::numeric_limits<size_t>::max();
      size_t batchSize =
          std::min<size_t>({static_cast<size_t>(config.candidateParallelism),
                            queue.size() - queueIndex, remainingBudget});
      llvm::SmallVector<CandidateCheckResult, 8> results(batchSize);
      llvm::SmallVector<unsigned, 8> futureSlots;
      std::vector<std::future<CandidateCheckResult>> futures;
      for (size_t batchOffset = 0; batchOffset < batchSize; ++batchOffset) {
        CandidateSpec candidate = queue[queueIndex + batchOffset].spec;
        ++visitedCount;
        if (std::optional<std::string> failure = getCheapTargetGeometryFailure(
                task, candidate, *reductionRanges, config.targetProfile)) {
          results[batchOffset].spec = candidate;
          results[batchOffset].failureReason = std::move(*failure);
          continue;
        }
        if (std::optional<std::string> failure =
                getSearchSPMHeadroomFailure(task, candidate, config)) {
          results[batchOffset].spec = candidate;
          results[batchOffset].failureReason = std::move(*failure);
          continue;
        }
        if (std::optional<int64_t> required =
                estimateTargetSPMRequiredLiveBytes(task, candidate,
                                                   config.spmAlignment);
            required && config.spmLimit > config.spmBase &&
            *required > config.spmLimit - config.spmBase) {
          results[batchOffset].spec = candidate;
          results[batchOffset].failureReason =
              "target_spm_bound: required modeled live roots exceed planning "
              "window";
          continue;
        }
        if (failsCheapSPMBound(task, candidate, config.spmBase,
                               config.spmLimit)) {
          results[batchOffset].spec = candidate;
          results[batchOffset].failureReason =
              "cheap_bound: minimum SPM bytes exceed planning window";
          continue;
        }
        futureSlots.push_back(static_cast<unsigned>(batchOffset));
        ++completeEvaluationCount;
        futures.push_back(evaluationExecutor->submit(
            standaloneTaskModuleText, *shape, candidate, config));
      }
      for (auto [futureIndex, slot] : llvm::enumerate(futureSlots))
        results[slot] = futures[futureIndex].get();
      queueIndex += batchSize;
      for (CandidateCheckResult &result : results) {
        processCheckResult(result);
        if (hasRequestedAlternative())
          break;
      }
      continue;
    }

    CandidateSpec candidate = queue[queueIndex++].spec;
    ++visitedCount;
    if (std::optional<std::string> failure = getCheapTargetGeometryFailure(
            task, candidate, *reductionRanges, config.targetProfile)) {
      rejectCandidate(candidate, *failure);
      continue;
    }
    if (std::optional<std::string> failure =
            getSearchSPMHeadroomFailure(task, candidate, config)) {
      rejectCandidate(candidate, *failure);
      continue;
    }
    if (std::optional<int64_t> required = estimateTargetSPMRequiredLiveBytes(
            task, candidate, config.spmAlignment);
        required && config.spmLimit > config.spmBase &&
        *required > config.spmLimit - config.spmBase) {
      rejectCandidate(candidate,
                      "target_spm_bound: required modeled live roots exceed "
                      "planning window");
      continue;
    }
    if (failsCheapSPMBound(task, candidate, config.spmBase, config.spmLimit)) {
      rejectCandidate(candidate,
                      "cheap_bound: minimum SPM bytes exceed planning window");
      continue;
    }
    ++completeEvaluationCount;
    CandidateCheckResult check =
        evaluateCandidateOnOriginalTask(task, *shape, candidate, config);
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
    if (std::optional<std::string> failure =
            getRankingCostFailure(check.stats)) {
      rejectCandidate(candidate, *failure);
      continue;
    }
    SelectedCandidate selected = buildSelected(check, visitedCount);
    insertCandidate(std::move(selected));
    if (hasRequestedAlternative())
      break;
    if (supportsTiledTraversal)
      enqueueRefinements(task, candidate, *reductionRanges, tileSizeOptions,
                         seen, queue, config.searchBeamWidth);
  }

  if (config.taskAlternativeOrdinal < candidateFrontier.size()) {
    SelectedCandidate selected =
        std::move(candidateFrontier[config.taskAlternativeOrdinal]);
    selected.candidateCount = visitedCount;
    selected.completeEvaluationCount = completeEvaluationCount;
    selected.rejectedCount = rejectedCount;
    if (!selected.module ||
        !isCompleteArtifactSource(selected.artifactSource)) {
      anchor->emitError()
          << "selected task candidate has no imported complete artifact";
      return mlir::failure();
    }
    return selected;
  }
  mlir::InFlightDiagnostic diagnostic = anchor->emitError();
  diagnostic
      << "no_candidate: tile selection found no passing task candidate after "
      << visitedCount << " candidates";
  if (!candidateFrontier.empty())
    diagnostic << "; requested bounded alternative ordinal "
               << config.taskAlternativeOrdinal << " but only "
               << candidateFrontier.size() << " candidate(s) survived";
  if (!lastFailure.empty())
    diagnostic << "; last failure: " << lastFailure;
  if (!supportsTiledTraversal)
    diagnostic << "; traversal root supports full traversal only";
  if (reductionSplitLegalityFailure && !reductionRanges->empty())
    diagnostic << "; reduction refinement unavailable: "
               << *reductionSplitLegalityFailure;
  return mlir::failure();
}

mlir::FailureOr<SelectedCandidate> selectCandidateForScope(
    const structured_scheduler::StructuredSchedulingScope &scope,
    mlir::func::FuncOp task, llvm::StringRef label,
    const SelectionConfig &config) {
  return selectCandidateForTask(scope, task, label, config);
}

} // namespace wafer::tensor_program_scheduling
