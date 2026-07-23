//===- WholeVariantCoordinator.cpp - All-rank candidate commit ----------===//

#include "WholeVariantCoordinator.h"

#include "AcceptedCallClosure.h"
#include "DirectDTETransport.h"
#include "ExecutableBundleInternal.h"
#include "TargetArtifactInternal.h"
#include "WholeVariantResourceAcceptance.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

constexpr size_t kWholeVariantVisitLimit = 64;
constexpr size_t kCoordinatedPolicyVisitLimit = 64;
constexpr size_t kWholeVariantParetoLimit = 16;
constexpr size_t kReportedAttemptLimit = 8;

static mlir::LogicalResult
verifyAcceptedRankModule(mlir::ModuleOp module, const ExecutionConfig &config,
                         int64_t logicalRank, TransportContract transport) {
  if (logicalRank < 0 || logicalRank >= config.getRankCount())
    return module.emitOpError("logical rank is outside ExecutionConfig");
  if (mlir::failed(verifyExactExecutionConfig(module, config)) ||
      mlir::failed(mlir::verify(module)))
    return mlir::failure();

  mlir::Operation *illegal = nullptr;
  module.walk([&](mlir::Operation *operation) {
    llvm::StringRef dialect = operation->getName().getDialectNamespace();
    llvm::StringRef name = operation->getName().getStringRef();
    bool allowed = dialect == "builtin" || dialect == "func" ||
                   dialect == "arith" || dialect == "math" ||
                   dialect == "memref" || dialect == "scf" || dialect == "cf";
    if (dialect == "wafer")
      allowed = mlir::isa<TargetTopologyOp, ExecutionMeshOp, TileRegionOp,
                          TileYieldOp>(operation) ||
                name.starts_with("wafer.instr.");
    if (!allowed) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation)) {
      if (transport != TransportContract::DirectDTE || !send.getBinding()) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    }
    if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation)) {
      if (transport != TransportContract::DirectDTE || !recv.getBinding()) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    }
    if (mlir::isa<InstrDTEWaitOp>(operation) &&
        transport != TransportContract::DirectDTE) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }

    auto verifyType = [](mlir::Type type) {
      return !mlir::isa<mlir::BaseMemRefType>(type) || isWaferMemRefType(type);
    };
    if (!llvm::all_of(operation->getOperandTypes(), verifyType) ||
        !llvm::all_of(operation->getResultTypes(), verifyType)) {
      illegal = operation;
      return mlir::WalkResult::interrupt();
    }
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        if (!llvm::all_of(block.getArgumentTypes(), verifyType)) {
          illegal = operation;
          return mlir::WalkResult::interrupt();
        }

    if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
      auto memory = getWaferMemoryAttr(alloc.getType());
      if (!memory ||
          (memory.getSpace() == MemorySpace::SPM &&
           !alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName)) ||
          (memory.getSpace() == MemorySpace::DDR &&
           !alloc->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName))) {
        illegal = operation;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  if (!illegal)
    return mlir::success();
  if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(illegal))
    return illegal->emitOpError(
        "does not satisfy the accepted executable transport contract");
  return illegal->emitOpError(
      "is not legal in an accepted static-rank executable");
}

static std::optional<frontend::ProgramRankSlice>
findRankSlice(llvm::ArrayRef<frontend::ProgramRankSlice> slices,
              int64_t logicalRank) {
  const frontend::ProgramRankSlice *match = nullptr;
  for (const frontend::ProgramRankSlice &slice : slices) {
    if (slice.logicalRank != logicalRank)
      continue;
    if (match)
      return std::nullopt;
    match = &slice;
  }
  if (!match)
    return std::nullopt;
  return *match;
}

static mlir::FailureOr<std::vector<RankProgramBinding>>
buildRankProgramBindings(
    const frontend::FrontendProgramVerificationResult &program,
    int64_t logicalRank, mlir::ModuleOp diagnosticAnchor) {
  std::vector<RankProgramBinding> bindings;
  auto appendBoundary = [&](const frontend::ProgramBoundaryBinding &binding,
                            ProgramResourceRole role) -> mlir::LogicalResult {
    std::optional<frontend::ProgramRankSlice> slice =
        findRankSlice(binding.rankSlices, logicalRank);
    if (!slice)
      return diagnosticAnchor.emitOpError(
          "typed program boundary does not contain exactly one rank slice");
    bindings.push_back({role,
                        binding.index,
                        binding.programIndex,
                        {},
                        binding.dtype,
                        binding.distribution,
                        binding.globalShape,
                        binding.localShape,
                        std::move(*slice)});
    return mlir::success();
  };
  for (const frontend::ProgramBoundaryBinding &binding :
       program.distributedInputs)
    if (mlir::failed(appendBoundary(binding, ProgramResourceRole::UserInput)))
      return mlir::failure();
  for (const frontend::ProgramParameterBinding &parameter :
       program.parameters) {
    std::optional<frontend::ProgramRankSlice> slice =
        findRankSlice(parameter.rankSlices, logicalRank);
    if (!slice) {
      diagnosticAnchor.emitOpError(
          "typed parameter metadata does not contain exactly one rank slice");
      return mlir::failure();
    }
    bindings.push_back({ProgramResourceRole::Parameter, parameter.argumentIndex,
                        -1, parameter.name, parameter.dtype,
                        parameter.distribution, parameter.globalShape,
                        parameter.localShape, std::move(*slice)});
  }
  for (const frontend::ProgramConstantBinding &constant : program.constants) {
    frontend::ProgramRankSlice slice;
    slice.logicalRank = logicalRank;
    slice.replicaId = logicalRank;
    slice.offsets.assign(constant.shape.size(), 0);
    slice.sizes = constant.shape;
    slice.strides.assign(constant.shape.size(), 1);
    slice.payloadPath = constant.payloadPath;
    bindings.push_back({ProgramResourceRole::Constant,
                        constant.argumentIndex,
                        constant.position,
                        {},
                        constant.dtype,
                        frontend::ProgramDistributionKind::Replicated,
                        constant.shape,
                        constant.shape,
                        std::move(slice)});
  }
  for (const frontend::ProgramBoundaryBinding &binding :
       program.distributedOutputs)
    if (mlir::failed(appendBoundary(binding, ProgramResourceRole::Output)))
      return mlir::failure();
  return bindings;
}

struct Combination {
  std::vector<size_t> positions;
};

struct WorseCombination {
  bool operator()(const Combination &lhs, const Combination &rhs) const {
    return lhs.positions > rhs.positions;
  }
};

using CandidateOrder = std::vector<std::vector<size_t>>;

static mlir::FailureOr<std::vector<size_t>>
getReservedBaselineIndices(const std::vector<RankVariantFrontier> &frontiers) {
  std::vector<size_t> indices;
  indices.reserve(frontiers.size());
  for (const RankVariantFrontier &frontier : frontiers) {
    std::optional<size_t> baseline;
    for (auto [index, candidate] : llvm::enumerate(frontier)) {
      if (!candidate.reservedBaseline)
        continue;
      if (baseline)
        return mlir::failure();
      baseline = index;
    }
    if (!baseline)
      return mlir::failure();
    indices.push_back(*baseline);
  }
  return indices;
}

static mlir::FailureOr<CandidateOrder>
buildCandidateOrder(const std::vector<RankVariantFrontier> &frontiers,
                    const ExecutionConfig &executionConfig) {
  if (frontiers.size() != static_cast<size_t>(executionConfig.getRankCount()))
    return mlir::failure();
  CandidateOrder order(frontiers.size());
  for (auto [rank, frontier] : llvm::enumerate(frontiers)) {
    if (frontier.empty())
      return mlir::failure();
    order[rank].resize(frontier.size());
    for (size_t index = 0; index < frontier.size(); ++index) {
      if (!frontier[index].module || frontier[index].stableOrdinal < 0)
        return mlir::failure();
      order[rank][index] = index;
    }
    llvm::sort(order[rank], [&](size_t lhs, size_t rhs) {
      const RankVariantCandidate &left = frontier[lhs];
      const RankVariantCandidate &right = frontier[rhs];
      return std::tie(left.stableOrdinal, left.artifactKind, lhs) <
             std::tie(right.stableOrdinal, right.artifactKind, rhs);
    });
  }
  return order;
}

static std::vector<size_t> getCandidateIndices(llvm::ArrayRef<size_t> positions,
                                               const CandidateOrder &order) {
  std::vector<size_t> indices;
  indices.reserve(positions.size());
  for (size_t rank = 0; rank < positions.size(); ++rank)
    indices.push_back(order[rank][positions[rank]]);
  return indices;
}

static mlir::FailureOr<AcceptedWholeVariant>
tryCombination(llvm::ArrayRef<size_t> candidateIndices,
               const std::vector<RankVariantFrontier> &frontiers,
               const frontend::FrontendProgramVerificationResult &program,
               const ExecutionConfig &executionConfig,
               std::string &failureGate) {
  if (candidateIndices.size() != frontiers.size() || candidateIndices.empty()) {
    failureGate = "rank-candidate-correspondence";
    return mlir::failure();
  }
  std::optional<int64_t> stableOrdinal;
  std::optional<wafer::RankArtifactKind> artifactKind;
  for (auto [rank, candidateIndex] : llvm::enumerate(candidateIndices)) {
    if (candidateIndex >= frontiers[rank].size()) {
      failureGate = "rank-candidate-correspondence";
      return mlir::failure();
    }
    int64_t current = frontiers[rank][candidateIndex].stableOrdinal;
    wafer::RankArtifactKind currentArtifactKind =
        frontiers[rank][candidateIndex].artifactKind;
    if ((stableOrdinal && current != *stableOrdinal) ||
        (artifactKind && currentArtifactKind != *artifactKind)) {
      failureGate = "rank-candidate-correspondence";
      return mlir::failure();
    }
    stableOrdinal = current;
    artifactKind = currentArtifactKind;
  }

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
  llvm::SmallVector<mlir::ModuleOp, 16> moduleViews;
  modules.reserve(candidateIndices.size());
  moduleViews.reserve(candidateIndices.size());
  for (auto [rank, candidateIndex] : llvm::enumerate(candidateIndices)) {
    const RankVariantCandidate &candidate = frontiers[rank][candidateIndex];
    modules.push_back(
        mlir::cast<mlir::ModuleOp>(candidate.module.get()->clone()));
    mlir::ModuleOp module = *modules.back();
    if (mlir::failed(verifyExactExecutionConfig(module, executionConfig)) ||
        mlir::failed(mlir::verify(module))) {
      failureGate = "rank-verifier";
      return mlir::failure();
    }
    moduleViews.push_back(module);
  }

  // DDR placement is intentionally absent from rank-frontier entries. Apply
  // it only to this disposable complete tuple so a failed late gate cannot
  // leak offsets into another combination or back into candidate generation.
  const TargetMemoryPolicy memory =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default).memory;
  for (mlir::ModuleOp module : moduleViews) {
    if (mlir::failed(planDDRMemoryModule(
            module, memory.ddrAlignmentBytes, memory.ddrCapacityBytes,
            memory.ddrLargestContiguousBytes, memory.ddrBandwidthLimitBytes)) ||
        mlir::failed(mlir::verify(module))) {
      failureGate = "whole-variant-ddr";
      return mlir::failure();
    }
  }

  mlir::FailureOr<TransportContract> transport =
      acceptDirectDTETransport(moduleViews);
  if (mlir::failed(transport)) {
    failureGate = "direct-dte";
    return mlir::failure();
  }
  mlir::FailureOr<analysis::WholeCardInstructionProgramCost> resourceCost =
      acceptWholeVariantResources(moduleViews, executionConfig);
  if (mlir::failed(resourceCost)) {
    failureGate = "whole-card-resources";
    return mlir::failure();
  }

  std::vector<RankExecutable> ranks;
  ranks.reserve(modules.size());
  for (size_t rank = 0; rank < modules.size(); ++rank) {
    mlir::ModuleOp module = *modules[rank];
    if (mlir::failed(verifyAcceptedRankModule(
            module, executionConfig, static_cast<int64_t>(rank), *transport))) {
      failureGate = "accepted-rank-verifier";
      return mlir::failure();
    }
    llvm::Expected<AcceptedCallClosure> closure =
        analyzeAcceptedCallClosure(module);
    if (!closure) {
      llvm::consumeError(closure.takeError());
      failureGate = "accepted-call-closure";
      return mlir::failure();
    }
    mlir::FailureOr<std::vector<RankProgramBinding>> bindings =
        buildRankProgramBindings(program, static_cast<int64_t>(rank), module);
    if (mlir::failed(bindings)) {
      failureGate = "rank-resource-projection";
      return mlir::failure();
    }
    std::string entrySymbol = closure->entry.getSymName().str();
    ranks.push_back(ExecutableBundleBuilder::makeRank(
        static_cast<int64_t>(rank), std::move(modules[rank]), entrySymbol,
        std::move(*bindings), *transport));
  }

  // Target ABI and target-call legality are candidate gates, not a later
  // opportunity to replace one rank after the remaining domain was accepted.
  // Lower on owned clones and discard the results; the target-artifact stage
  // will translate the exact committed rank modules once more for publication.
  for (RankExecutable &rank : ranks) {
    mlir::FailureOr<PreparedTargetRank> prepared =
        prepareTargetABI(rank, executionConfig);
    if (mlir::failed(prepared)) {
      failureGate = "target-abi-preparation";
      return mlir::failure();
    }
    if (mlir::failed(lowerToTargetLLVM(*prepared)) ||
        mlir::failed(
            verifyLoweredKernelABI(*prepared, rank.getEntrySymbol()))) {
      failureGate = "target-abi-lowering";
      return mlir::failure();
    }
  }

  AcceptedWholeVariant accepted;
  accepted.ranks = std::move(ranks);
  accepted.resourceCost = std::move(*resourceCost);
  accepted.selectedStableOrdinals.reserve(candidateIndices.size());
  accepted.selectedArtifactKinds.reserve(candidateIndices.size());
  accepted.selectedReservedBaselines.reserve(candidateIndices.size());
  for (auto [rank, candidateIndex] : llvm::enumerate(candidateIndices)) {
    accepted.selectedStableOrdinals.push_back(
        frontiers[rank][candidateIndex].stableOrdinal);
    accepted.selectedArtifactKinds.push_back(
        frontiers[rank][candidateIndex].artifactKind);
    accepted.selectedReservedBaselines.push_back(
        frontiers[rank][candidateIndex].reservedBaseline);
  }
  return accepted;
}

enum class ParetoOrder {
  Equivalent,
  LeftDominates,
  RightDominates,
  Incomparable,
  Unknown,
};

struct MetricPair {
  const analysis::ScheduleCostMetric *left;
  const analysis::ScheduleCostMetric *right;
};

static ParetoOrder compareKnownMetrics(llvm::ArrayRef<MetricPair> dimensions) {
  bool leftLower = false;
  bool rightLower = false;
  for (const MetricPair &dimension : dimensions) {
    if (!dimension.left->isKnown() || !dimension.right->isKnown())
      return ParetoOrder::Unknown;
    leftLower |= dimension.left->value < dimension.right->value;
    rightLower |= dimension.right->value < dimension.left->value;
  }
  if (!leftLower && !rightLower)
    return ParetoOrder::Equivalent;
  if (leftLower && !rightLower)
    return ParetoOrder::LeftDominates;
  if (!leftLower && rightLower)
    return ParetoOrder::RightDominates;
  return ParetoOrder::Incomparable;
}

static ParetoOrder compareExactWholeVariantCost(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right) {
  const MetricPair dimensions[] = {
      {&left.aggregateCompute.npuF16Bf16LogicalOps,
       &right.aggregateCompute.npuF16Bf16LogicalOps},
      {&left.aggregateCompute.npuOtherLogicalOps,
       &right.aggregateCompute.npuOtherLogicalOps},
      {&left.aggregateCompute.vectorF16Bf16LogicalOps,
       &right.aggregateCompute.vectorF16Bf16LogicalOps},
      {&left.aggregateCompute.vectorF32LogicalOps,
       &right.aggregateCompute.vectorF32LogicalOps},
      {&left.aggregateCompute.vectorOtherLogicalOps,
       &right.aggregateCompute.vectorOtherLogicalOps},
      {&left.aggregateDDRReadBytes, &right.aggregateDDRReadBytes},
      {&left.aggregateDDRWriteBytes, &right.aggregateDDRWriteBytes},
      {&left.aggregateSPMMovementBytes, &right.aggregateSPMMovementBytes},
      {&left.aggregateNoC.aggregateTransmitBytes,
       &right.aggregateNoC.aggregateTransmitBytes},
      {&left.aggregateNoC.aggregateReceiveBytes,
       &right.aggregateNoC.aggregateReceiveBytes},
      {&left.minimumHopLinkByteDemand, &right.minimumHopLinkByteDemand},
      {&left.aggregateNoC.collectiveTransmitBytes[0],
       &right.aggregateNoC.collectiveTransmitBytes[0]},
      {&left.aggregateNoC.collectiveTransmitBytes[1],
       &right.aggregateNoC.collectiveTransmitBytes[1]},
      {&left.aggregateNoC.collectiveTransmitBytes[2],
       &right.aggregateNoC.collectiveTransmitBytes[2]},
      {&left.aggregateNoC.collectiveTransmitBytes[3],
       &right.aggregateNoC.collectiveTransmitBytes[3]},
      {&left.aggregateNoC.collectiveTransmitBytes[4],
       &right.aggregateNoC.collectiveTransmitBytes[4]},
      {&left.aggregateInstructionCount, &right.aggregateInstructionCount},
      {&left.aggregateEventCount, &right.aggregateEventCount},
      {&left.maximumRankSPMHighWaterBytes, &right.maximumRankSPMHighWaterBytes},
      {&left.summedRankSPMHighWaterBytes, &right.summedRankSPMHighWaterBytes},
  };

  return compareKnownMetrics(dimensions);
}

static ParetoOrder compareExactExecutionResources(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right) {
  analysis::WholeCardInstructionProgramCost leftExecution = left;
  analysis::WholeCardInstructionProgramCost rightExecution = right;
  leftExecution.maximumRankSPMHighWaterBytes = {};
  leftExecution.summedRankSPMHighWaterBytes = {};
  rightExecution.maximumRankSPMHighWaterBytes = {};
  rightExecution.summedRankSPMHighWaterBytes = {};
  return compareExactWholeVariantCost(leftExecution, rightExecution);
}

static ParetoOrder compareStaticDataflowPolicy(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right) {
  const analysis::ScheduleCostMetric &leftDepth =
      left.maximumRankDataDependencyDepth;
  const analysis::ScheduleCostMetric &rightDepth =
      right.maximumRankDataDependencyDepth;
  if (!leftDepth.isKnown() || !rightDepth.isKnown())
    return ParetoOrder::Unknown;
  if (leftDepth.value < rightDepth.value)
    return ParetoOrder::LeftDominates;
  if (leftDepth.value > rightDepth.value)
    return ParetoOrder::RightDominates;
  const analysis::ScheduleCostMetric &leftInversions =
      left.aggregateReadyOrderPriorityInversions;
  const analysis::ScheduleCostMetric &rightInversions =
      right.aggregateReadyOrderPriorityInversions;
  if (!leftInversions.isKnown() || !rightInversions.isKnown())
    return ParetoOrder::Unknown;
  if (leftInversions.value < rightInversions.value)
    return ParetoOrder::LeftDominates;
  if (leftInversions.value > rightInversions.value)
    return ParetoOrder::RightDominates;
  return ParetoOrder::Equivalent;
}

static ParetoOrder compareTargetStaticTradeoff(
    const analysis::WholeCardInstructionProgramCost &left,
    const analysis::WholeCardInstructionProgramCost &right,
    const TargetStaticSelectionPolicy &policy) {
  if (policy.tradeoff == TargetStaticTradeoffPolicy::Conservative)
    return ParetoOrder::Equivalent;

  const llvm::SmallVector<llvm::SmallVector<MetricPair, 8>, 6> priorityClasses =
      {
          {{&left.aggregateDDRReadBytes, &right.aggregateDDRReadBytes},
           {&left.aggregateDDRWriteBytes, &right.aggregateDDRWriteBytes}},
          {{&left.aggregateNoC.aggregateTransmitBytes,
            &right.aggregateNoC.aggregateTransmitBytes},
           {&left.aggregateNoC.aggregateReceiveBytes,
            &right.aggregateNoC.aggregateReceiveBytes},
           {&left.minimumHopLinkByteDemand, &right.minimumHopLinkByteDemand},
           {&left.aggregateNoC.collectiveTransmitBytes[0],
            &right.aggregateNoC.collectiveTransmitBytes[0]},
           {&left.aggregateNoC.collectiveTransmitBytes[1],
            &right.aggregateNoC.collectiveTransmitBytes[1]},
           {&left.aggregateNoC.collectiveTransmitBytes[2],
            &right.aggregateNoC.collectiveTransmitBytes[2]},
           {&left.aggregateNoC.collectiveTransmitBytes[3],
            &right.aggregateNoC.collectiveTransmitBytes[3]},
           {&left.aggregateNoC.collectiveTransmitBytes[4],
            &right.aggregateNoC.collectiveTransmitBytes[4]}},
          {{&left.aggregateSPMMovementBytes, &right.aggregateSPMMovementBytes}},
          {{&left.aggregateInstructionCount, &right.aggregateInstructionCount},
           {&left.aggregateEventCount, &right.aggregateEventCount}},
          {{&left.aggregateCompute.npuF16Bf16LogicalOps,
            &right.aggregateCompute.npuF16Bf16LogicalOps},
           {&left.aggregateCompute.npuOtherLogicalOps,
            &right.aggregateCompute.npuOtherLogicalOps},
           {&left.aggregateCompute.vectorF16Bf16LogicalOps,
            &right.aggregateCompute.vectorF16Bf16LogicalOps},
           {&left.aggregateCompute.vectorF32LogicalOps,
            &right.aggregateCompute.vectorF32LogicalOps},
           {&left.aggregateCompute.vectorOtherLogicalOps,
            &right.aggregateCompute.vectorOtherLogicalOps}},
          {{&left.maximumRankDataDependencyDepth,
            &right.maximumRankDataDependencyDepth},
           {&left.aggregateReadyOrderPriorityInversions,
            &right.aggregateReadyOrderPriorityInversions}},
      };
  for (const llvm::SmallVector<MetricPair, 8> &priorityClass :
       priorityClasses) {
    ParetoOrder order = compareKnownMetrics(priorityClass);
    if (order != ParetoOrder::Equivalent)
      return order;
  }
  return ParetoOrder::Equivalent;
}

static bool isPreferredOver(const AcceptedWholeVariant &candidate,
                            const AcceptedWholeVariant &baseline,
                            const TargetStaticSelectionPolicy &policy) {
  ParetoOrder order = compareExactWholeVariantCost(candidate.resourceCost,
                                                   baseline.resourceCost);
  if (order == ParetoOrder::LeftDominates)
    return true;
  if (order == ParetoOrder::Equivalent)
    return compareStaticDataflowPolicy(candidate.resourceCost,
                                       baseline.resourceCost) ==
           ParetoOrder::LeftDominates;
  if (order != ParetoOrder::Incomparable)
    return false;
  // Accepted high-water is a capacity fact, not a calibrated performance
  // quantity. When all exact execution-resource dimensions are no worse and
  // at least one is lower, the static target policy accepts the known
  // high-water tradeoff instead of requiring an unavailable timing model.
  if (compareExactExecutionResources(candidate.resourceCost,
                                     baseline.resourceCost) ==
      ParetoOrder::LeftDominates)
    return true;
  return compareTargetStaticTradeoff(candidate.resourceCost,
                                     baseline.resourceCost,
                                     policy) == ParetoOrder::LeftDominates;
}

static bool hasEarlierStaticPolicyOrder(const AcceptedWholeVariant &lhs,
                                        const AcceptedWholeVariant &rhs) {
  return std::tie(lhs.selectedStableOrdinals, lhs.selectedArtifactKinds,
                  lhs.selectedReservedBaselines) <
         std::tie(rhs.selectedStableOrdinals, rhs.selectedArtifactKinds,
                  rhs.selectedReservedBaselines);
}

static void
insertParetoCandidate(AcceptedWholeVariant candidate,
                      llvm::SmallVectorImpl<AcceptedWholeVariant> &frontier) {
  llvm::SmallVector<unsigned, 8> dominatedIndices;
  for (auto [index, existing] : llvm::enumerate(frontier)) {
    switch (compareExactWholeVariantCost(candidate.resourceCost,
                                         existing.resourceCost)) {
    case ParetoOrder::Unknown:
    case ParetoOrder::RightDominates:
      return;
    case ParetoOrder::Equivalent:
      switch (compareStaticDataflowPolicy(candidate.resourceCost,
                                          existing.resourceCost)) {
      case ParetoOrder::LeftDominates:
        dominatedIndices.push_back(index);
        continue;
      case ParetoOrder::RightDominates:
        return;
      case ParetoOrder::Equivalent:
      case ParetoOrder::Incomparable:
      case ParetoOrder::Unknown:
        break;
      }
      if (!hasEarlierStaticPolicyOrder(candidate, existing))
        return;
      dominatedIndices.push_back(index);
      break;
    case ParetoOrder::LeftDominates:
      dominatedIndices.push_back(index);
      break;
    case ParetoOrder::Incomparable:
      break;
    }
  }
  for (unsigned index : llvm::reverse(dominatedIndices))
    frontier.erase(frontier.begin() + index);
  frontier.push_back(std::move(candidate));
  llvm::sort(frontier, hasEarlierStaticPolicyOrder);
  if (frontier.size() > kWholeVariantParetoLimit)
    frontier.pop_back();
}

static std::string summarizeAttemptFailure(llvm::ArrayRef<size_t> indices,
                                           llvm::StringRef gate,
                                           llvm::StringRef diagnostics) {
  std::string summary;
  llvm::raw_string_ostream os(summary);
  os << "candidates=[";
  for (auto [rank, index] : llvm::enumerate(indices)) {
    if (rank)
      os << ",";
    os << index;
  }
  os << "] gate=" << gate;
  diagnostics = diagnostics.trim();
  if (!diagnostics.empty()) {
    constexpr size_t maxDiagnosticBytes = 512;
    os << " diagnostic=" << diagnostics.take_front(maxDiagnosticBytes);
    if (diagnostics.size() > maxDiagnosticBytes)
      os << "...";
  }
  return summary;
}

} // namespace

mlir::FailureOr<AcceptedWholeVariant> selectAcceptedWholeVariant(
    const std::vector<RankVariantFrontier> &frontiers,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig, llvm::raw_ostream &diagnostics) {
  if (program.logicalRankCount != executionConfig.getRankCount()) {
    diagnostics << "wafer-compile: typed program rank domain does not match "
                   "whole-variant ExecutionConfig\n";
    return mlir::failure();
  }
  mlir::FailureOr<CandidateOrder> candidateOrder =
      buildCandidateOrder(frontiers, executionConfig);
  if (mlir::failed(candidateOrder)) {
    diagnostics << "wafer-compile: rank scheduling frontiers do not form the "
                   "complete canonical rank domain\n";
    return mlir::failure();
  }

  mlir::MLIRContext *context =
      frontiers.front().front().module.get().getContext();
  for (const RankVariantFrontier &frontier : frontiers)
    for (const RankVariantCandidate &candidate : frontier)
      if (candidate.module.get().getContext() != context) {
        diagnostics << "wafer-compile: rank scheduling frontiers do not share "
                       "the executable-bundle owner context\n";
        return mlir::failure();
      }

  mlir::FailureOr<std::vector<size_t>> reservedBaselineIndices =
      getReservedBaselineIndices(frontiers);
  if (mlir::failed(reservedBaselineIndices)) {
    diagnostics << "wafer-compile: every rank frontier must contain exactly "
                   "one reserved baseline candidate\n";
    return mlir::failure();
  }

  std::set<std::vector<size_t>> enqueuedPositions;
  std::set<std::vector<size_t>> attemptedCandidateIndices;
  std::priority_queue<Combination, std::vector<Combination>, WorseCombination>
      queue;
  std::vector<size_t> initial(frontiers.size(), 0);
  queue.push({initial});
  enqueuedPositions.insert(initial);

  llvm::SmallVector<std::string, kReportedAttemptLimit> failures;
  auto attempt = [&](const std::vector<size_t> &candidateIndices)
      -> mlir::FailureOr<AcceptedWholeVariant> {
    if (!attemptedCandidateIndices.insert(candidateIndices).second)
      return mlir::failure();
    std::string capturedDiagnostics;
    std::string failureGate = "unknown";
    mlir::FailureOr<AcceptedWholeVariant> result = mlir::failure();
    {
      mlir::ScopedDiagnosticHandler handler(
          context, [&](mlir::Diagnostic &diagnostic) {
            llvm::raw_string_ostream os(capturedDiagnostics);
            diagnostic.print(os);
            os << "\n";
            return mlir::success();
          });
      result = tryCombination(candidateIndices, frontiers, program,
                              executionConfig, failureGate);
    }
    if (mlir::succeeded(result))
      return result;
    std::string summary = summarizeAttemptFailure(candidateIndices, failureGate,
                                                  capturedDiagnostics);
    if (failures.size() < kReportedAttemptLimit)
      failures.push_back(std::move(summary));
    else
      failures.back() = std::move(summary);
    return mlir::failure();
  };

  // The reserved baseline has its own allowance and must pass every late gate
  // before any optimization budget is consumed. Keep the accepted baseline as
  // the conservative fallback while alternatives are evaluated.
  mlir::FailureOr<AcceptedWholeVariant> baseline =
      attempt(*reservedBaselineIndices);
  if (mlir::failed(baseline)) {
    diagnostics << "wafer-compile: reserved all-baseline variant failed "
                   "whole-variant acceptance\n";
    for (const std::string &failure : failures)
      diagnostics << "  - " << failure << "\n";
    return mlir::failure();
  }
  AcceptedWholeVariant baselineAccepted = std::move(*baseline);
  llvm::SmallVector<AcceptedWholeVariant, kWholeVariantParetoLimit>
      paretoFrontier;
  auto retainAccepted = [&](mlir::FailureOr<AcceptedWholeVariant> accepted) {
    if (mlir::succeeded(accepted))
      insertParetoCandidate(std::move(*accepted), paretoFrontier);
  };

  size_t visited = 0;
  while (!queue.empty() && visited < kWholeVariantVisitLimit) {
    Combination combination = queue.top();
    queue.pop();
    ++visited;
    std::vector<size_t> candidateIndices =
        getCandidateIndices(combination.positions, *candidateOrder);
    retainAccepted(attempt(candidateIndices));

    for (size_t rank = 0; rank < combination.positions.size(); ++rank) {
      std::vector<size_t> neighbor = combination.positions;
      if (++neighbor[rank] >= (*candidateOrder)[rank].size())
        continue;
      if (!enqueuedPositions.insert(neighbor).second)
        continue;
      queue.push({neighbor});
    }
  }

  // A same-generation tuple can sit far from the canonical Cartesian corner
  // of a high-dimensional product. Try each complete correspondence ordinal
  // after the bounded product walk so cross-rank actual clones generated from
  // one semantic recipe are evaluated together without reconstructing state.
  using CorrespondenceKey = std::pair<int64_t, wafer::RankArtifactKind>;
  std::set<CorrespondenceKey> correspondenceKeys;
  for (const RankVariantCandidate &candidate : frontiers.front())
    correspondenceKeys.insert(
        {candidate.stableOrdinal, candidate.artifactKind});
  size_t coordinatedVisited = 0;
  for (CorrespondenceKey key : correspondenceKeys) {
    if (coordinatedVisited >= kCoordinatedPolicyVisitLimit)
      break;
    std::vector<size_t> candidateIndices;
    candidateIndices.reserve(frontiers.size());
    bool complete = true;
    for (const RankVariantFrontier &frontier : frontiers) {
      std::optional<size_t> match;
      for (auto [index, candidate] : llvm::enumerate(frontier)) {
        if (candidate.stableOrdinal != key.first ||
            candidate.artifactKind != key.second)
          continue;
        if (!match || index < *match)
          match = index;
      }
      if (!match) {
        complete = false;
        break;
      }
      candidateIndices.push_back(*match);
    }
    if (!complete || attemptedCandidateIndices.count(candidateIndices))
      continue;
    ++coordinatedVisited;
    retainAccepted(attempt(candidateIndices));
  }

  const TargetStaticSelectionPolicy selectionPolicy =
      getDefaultWaferTargetPolicy(TileSearchEffort::Default).staticSelection;
  AcceptedWholeVariant selected = std::move(baselineAccepted);
  for (AcceptedWholeVariant &candidate : paretoFrontier) {
    if (isPreferredOver(candidate, selected, selectionPolicy))
      selected = std::move(candidate);
  }
  return selected;
}

} // namespace wafer::compiler::detail
