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

static int64_t estimateMetricTimePs(const analysis::ScheduleCostMetric &metric,
                                    uint64_t unitsPerSecond) {
  if (metric.isKnown() && metric.value == 0)
    return 0;
  if (!metric.isKnown() || unitsPerSecond == 0)
    return std::numeric_limits<int64_t>::max();
  constexpr uint64_t picosecondsPerSecond = 1'000'000'000'000ULL;
  __uint128_t numerator =
      static_cast<__uint128_t>(metric.value) * picosecondsPerSecond;
  __uint128_t value = (numerator + unitsPerSecond - 1) / unitsPerSecond;
  if (value > static_cast<__uint128_t>(std::numeric_limits<int64_t>::max()))
    return std::numeric_limits<int64_t>::max();
  return static_cast<int64_t>(value);
}

static uint64_t deriveCoarseRate(uint64_t referenceUnitsPerSecond,
                                 int64_t unitsPerCycle,
                                 int64_t referenceUnitsPerCycle) {
  if (unitsPerCycle <= 0 || referenceUnitsPerCycle <= 0)
    return 0;
  __uint128_t rate = static_cast<__uint128_t>(referenceUnitsPerSecond) *
                     static_cast<uint64_t>(unitsPerCycle) /
                     static_cast<uint64_t>(referenceUnitsPerCycle);
  return rate > static_cast<__uint128_t>(std::numeric_limits<uint64_t>::max())
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(rate);
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

int64_t estimateCandidateTimePs(const CandidateStats &stats) {
  const analysis::InstructionProgramCost &cost = stats.program;
  analysis::TargetScheduleCostPolicy policy =
      analysis::getTargetScheduleCostPolicy(
          TargetProfileId::waferTx81SingleCardKernelV1());
  const TargetTimingPolicy coarseTiming;

  int64_t compute = 0;
  compute = saturatingAdd(
      compute,
      estimateMetricTimePs(cost.compute.npuF16Bf16LogicalOps,
                           policy.f16Bf16NPULogicalOpsPerSecondPerTile));
  compute = saturatingAdd(
      compute,
      estimateMetricTimePs(cost.compute.vectorF16Bf16LogicalOps,
                           policy.f16Bf16VectorLogicalOpsPerSecondPerTile));
  compute = saturatingAdd(
      compute,
      estimateMetricTimePs(cost.compute.vectorF32LogicalOps,
                           policy.f32VectorLogicalOpsPerSecondPerTile));
  if (cost.compute.npuOtherLogicalOps.value != 0 ||
      cost.compute.vectorOtherLogicalOps.value != 0)
    compute = std::numeric_limits<int64_t>::max();

  analysis::ScheduleCostMetric ddr = cost.ddrReadBytes;
  if (ddr.isKnown() && cost.ddrWriteBytes.isKnown()) {
    if (cost.ddrWriteBytes.value >
        std::numeric_limits<uint64_t>::max() - ddr.value) {
      ddr.knowledge = analysis::ScheduleCostKnowledge::Overflow;
      ddr.reason = analysis::ScheduleCostReason::ArithmeticOverflow;
      ddr.value = 0;
    } else {
      ddr.value += cost.ddrWriteBytes.value;
    }
  } else if (!cost.ddrWriteBytes.isKnown()) {
    ddr = cost.ddrWriteBytes;
  }
  int64_t movement = estimateMetricTimePs(ddr, policy.cardDDRBytesPerSecond);
  movement = saturatingAdd(
      movement, estimateMetricTimePs(cost.noc.aggregateTransmitBytes,
                                     policy.directionalNoCBytesPerSecond));

  // The target contract establishes no calibrated SPM or issue clock. Keep
  // these dimensions in the ranking by anchoring the compiler-private
  // bytes/cycle and issue-cycle ratios to the established DDR rate. This is a
  // coarse ordering policy, not a board-time claim. Every term is serialized;
  // no compute/movement or event overlap is assumed.
  uint64_t spmBytesPerSecond = deriveCoarseRate(policy.cardDDRBytesPerSecond,
                                                coarseTiming.spmBytesPerCycle,
                                                coarseTiming.ddrBytesPerCycle);
  int64_t issueReferenceUnitsPerCycle = saturatingMul(
      coarseTiming.ddrBytesPerCycle, coarseTiming.instrIssueCycles);
  uint64_t issuesPerSecond =
      deriveCoarseRate(policy.cardDDRBytesPerSecond, /*unitsPerCycle=*/1,
                       issueReferenceUnitsPerCycle);
  movement = saturatingAdd(
      movement, estimateMetricTimePs(cost.spmMovementBytes, spmBytesPerSecond));
  int64_t setup = estimateMetricTimePs(cost.instructionCount, issuesPerSecond);
  setup = saturatingAdd(setup,
                        estimateMetricTimePs(cost.eventCount, issuesPerSecond));

  return saturatingAdd(saturatingAdd(compute, movement), setup);
}

bool hasStrictExecutionCostDominance(const CandidateStats &candidate,
                                     const CandidateStats &baseline) {
  const analysis::InstructionProgramCost &candidateCost = candidate.program;
  const analysis::InstructionProgramCost &baselineCost = baseline.program;
  struct MetricPair {
    const analysis::ScheduleCostMetric *candidate;
    const analysis::ScheduleCostMetric *baseline;
  };
  const MetricPair dimensions[] = {
      {&candidateCost.compute.npuF16Bf16LogicalOps,
       &baselineCost.compute.npuF16Bf16LogicalOps},
      {&candidateCost.compute.npuOtherLogicalOps,
       &baselineCost.compute.npuOtherLogicalOps},
      {&candidateCost.compute.vectorF16Bf16LogicalOps,
       &baselineCost.compute.vectorF16Bf16LogicalOps},
      {&candidateCost.compute.vectorF32LogicalOps,
       &baselineCost.compute.vectorF32LogicalOps},
      {&candidateCost.compute.vectorOtherLogicalOps,
       &baselineCost.compute.vectorOtherLogicalOps},
      {&candidateCost.ddrReadBytes, &baselineCost.ddrReadBytes},
      {&candidateCost.ddrWriteBytes, &baselineCost.ddrWriteBytes},
      {&candidateCost.spmMovementBytes, &baselineCost.spmMovementBytes},
      {&candidateCost.noc.aggregateTransmitBytes,
       &baselineCost.noc.aggregateTransmitBytes},
      {&candidateCost.noc.aggregateReceiveBytes,
       &baselineCost.noc.aggregateReceiveBytes},
      {&candidateCost.instructionCount, &baselineCost.instructionCount},
      {&candidateCost.eventCount, &baselineCost.eventCount},
  };

  bool strictlyLower = false;
  for (const MetricPair &dimension : dimensions) {
    if (!dimension.candidate->isKnown() || !dimension.baseline->isKnown() ||
        dimension.candidate->value > dimension.baseline->value)
      return false;
    strictlyLower |= dimension.candidate->value < dimension.baseline->value;
  }
  return strictlyLower;
}

static bool isBetterCandidate(const SelectedCandidate &candidate,
                              const SelectedCandidate *best) {
  if (!best)
    return true;
  if (candidate.estimatedTimePs != best->estimatedTimePs)
    return candidate.estimatedTimePs < best->estimatedTimePs;
  if (candidate.spec.tileSizes != best->spec.tileSizes)
    return candidate.spec.tileSizes > best->spec.tileSizes;
  if (candidate.spec.reductionSplitSizes.empty() !=
      best->spec.reductionSplitSizes.empty())
    return candidate.spec.reductionSplitSizes.empty();
  if (candidate.spec.reductionSplitSizes != best->spec.reductionSplitSizes)
    return candidate.spec.reductionSplitSizes > best->spec.reductionSplitSizes;
  return !candidate.spec.selectedImplementationAlternative &&
         best->spec.selectedImplementationAlternative.has_value();
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
  return rankRefinementDimsImpl(getYieldedRootLinalgOps(task), candidate,
                                reductionRanges);
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
  std::optional<int64_t> initialBytes =
      estimateTargetSPMWorkingSetBytes(task, initial, config.spmAlignment);
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
        estimateTargetSPMWorkingSetBytes(task, *refined, config.spmAlignment);
    if (!refinedBytes)
      return std::nullopt;
    if (*refinedBytes <= capacityBytes)
      return refined;
    current = std::move(*refined);
  }
}

mlir::FailureOr<SelectedCandidate> selectCandidateForScope(
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
      getStaticRootReductionRanges(task);
  if (mlir::failed(reductionRanges)) {
    anchor->emitError()
        << "no_candidate: tile selection requires static reduction ranges";
    return mlir::failure();
  }
  std::optional<std::string> reductionSplitLegalityFailure =
      getReductionSplitLegalityFailure(task);
  TileSizeOptions tileSizeOptions;
  if (supportsTiledTraversal)
    tileSizeOptions = buildTileSizeOptions(
        *shape, *reductionRanges, config.preferredTileSizes,
        config.maxCandidatesPerDim, !reductionSplitLegalityFailure);

  std::optional<SelectedCandidate> best;
  int64_t rejectedCount = 0;
  std::string lastFailure;
  int64_t visitedCount = 0;
  llvm::StringSet<> seen;
  llvm::SmallVector<CandidateWorkItem, 32> queue;
  CandidateSpec initial;
  initial.tileSizes.assign(shape->begin(), shape->end());
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
  for (TargetImplementationKind implementation :
       collectImplementationAlternatives(task)) {
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
    selected.estimatedTimePs = estimateCandidateTimePs(selected.stats);
    selected.candidateCount = currentVisited;
    selected.rejectedCount = rejectedCount;
    selected.representativeCount = check.representativeCount;
    selected.module = std::move(check.module);
    selected.artifactSource = check.artifactSource;
    return selected;
  };
  auto isRetryableFailure = [](llvm::StringRef failure) {
    return failure.starts_with("cheap_bound:") ||
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
      enqueueRefinements(task, candidate, *reductionRanges, tileSizeOptions,
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
    if (config.mode == TileSearchMode::MinEstimatedTime) {
      if (std::optional<std::string> failure =
              getRankingCostFailure(check.stats)) {
        rejectCandidate(check.spec, *failure);
        return;
      }
    }
    SelectedCandidate selected = buildSelected(check, visitedCount);
    if (isBetterCandidate(selected, best ? &*best : nullptr)) {
      if (!selected.module) {
        CandidateEvaluation accepted =
            evaluateCompleteCandidate(task, *shape, selected.spec, config);
        if (!accepted.failureReason.empty()) {
          rejectCandidate(selected.spec, accepted.failureReason);
          return;
        }
        selected.module = std::move(accepted.module);
        selected.artifactSource = accepted.artifactSource;
      }
      best = std::move(selected);
    }
    if (supportsTiledTraversal)
      enqueueRefinements(task, check.spec, *reductionRanges, tileSizeOptions,
                         seen, queue, config.searchBeamWidth);
  };

  size_t queueIndex = 0;
  std::string standaloneTaskModuleText;
  if (config.mode == TileSearchMode::MinEstimatedTime &&
      config.candidateParallelism > 1)
    standaloneTaskModuleText = getStandaloneTaskModuleText(task);

  while (queueIndex < queue.size()) {
    if (config.maxSearchCandidates > 0 &&
        visitedCount >= config.maxSearchCandidates)
      break;
    if (config.mode == TileSearchMode::MinEstimatedTime &&
        config.candidateParallelism > 1) {
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
                task, candidate, *reductionRanges)) {
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
        futures.push_back(std::async(
            std::launch::async,
            [&standaloneTaskModuleText, shape = *shape, candidate, config]() {
              return evaluateCandidateOnStandaloneTaskText(
                  standaloneTaskModuleText, shape, candidate, config);
            }));
      }
      for (auto [futureIndex, slot] : llvm::enumerate(futureSlots))
        results[slot] = futures[futureIndex].get();
      queueIndex += batchSize;
      for (CandidateCheckResult &result : results)
        processCheckResult(result);
      continue;
    }

    CandidateSpec candidate = queue[queueIndex++].spec;
    ++visitedCount;
    if (std::optional<std::string> failure =
            getCheapTargetGeometryFailure(task, candidate, *reductionRanges)) {
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
    if (config.mode == TileSearchMode::MinEstimatedTime) {
      if (std::optional<std::string> failure =
              getRankingCostFailure(check.stats)) {
        rejectCandidate(candidate, *failure);
        continue;
      }
    }
    SelectedCandidate selected = buildSelected(check, visitedCount);
    if (config.mode == TileSearchMode::FirstLegal)
      return selected;
    if (isBetterCandidate(selected, best ? &*best : nullptr))
      best = std::move(selected);
    if (supportsTiledTraversal)
      enqueueRefinements(task, candidate, *reductionRanges, tileSizeOptions,
                         seen, queue, config.searchBeamWidth);
  }

  if (best) {
    best->candidateCount = visitedCount;
    best->rejectedCount = rejectedCount;
    if (!best->module || !isCompleteArtifactSource(best->artifactSource)) {
      CandidateEvaluation accepted =
          evaluateCompleteCandidate(task, *shape, best->spec, config);
      if (!accepted.failureReason.empty()) {
        anchor->emitError()
            << "selected task candidate complete artifact failed: "
            << accepted.failureReason;
        return mlir::failure();
      }
      best->module = std::move(accepted.module);
      best->artifactSource = accepted.artifactSource;
    }
    return std::move(*best);
  }
  mlir::InFlightDiagnostic diagnostic = anchor->emitError();
  diagnostic
      << "no_candidate: tile selection found no passing task candidate after "
      << visitedCount << " candidates";
  if (!lastFailure.empty())
    diagnostic << "; last failure: " << lastFailure;
  if (!supportsTiledTraversal)
    diagnostic << "; traversal root supports full traversal only";
  if (reductionSplitLegalityFailure && !reductionRanges->empty())
    diagnostic << "; reduction refinement unavailable: "
               << *reductionSplitLegalityFailure;
  return mlir::failure();
}

} // namespace wafer::tensor_program_scheduling
