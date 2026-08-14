//===- CoordinatedExecutableFinalization.cpp - Exact all-rank gate
//-----------===//

#include "CoordinatedExecutableFinalization.h"

#include "BoundedRankExecutor.h"
#include "CoordinatedVariantSelection.h"
#include "ScheduledRankFinalization.h"

#include "Wafer/Analysis/ScheduleCostAnalysis.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/CompileTiming.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetSchedulingCapability.h"
#include "Wafer/Analysis/TargetSchedulingAnalysis.h"
#include "Wafer/Transforms/PhysicalDataflow.h"
#include "Wafer/Transforms/SoftwarePipelining.h"
#include "Wafer/Transforms/WorkerPlacement.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>

namespace wafer::compiler::detail {

std::string computeAdmittedCoordinatedExecutableFrontierDigest(
    llvm::ArrayRef<AdmittedCoordinatedExecutable> frontier) {
  llvm::SHA256 hasher;
  for (const AdmittedCoordinatedExecutable &executable : frontier) {
    std::string header =
        (llvm::Twine("semantic:") +
         llvm::Twine(executable.stableSemanticOrdinal) + ":action:" +
         llvm::Twine(executable.scheduleActionOrdinal) + ":ready-order:" +
         llvm::Twine(
             static_cast<unsigned>(executable.actionIdentity.readyOrderKind)) +
         ":buffering:" +
         llvm::Twine(
             static_cast<unsigned>(executable.actionIdentity.bufferingKind)) +
         ":buffering-plan:" +
         llvm::Twine(executable.actionIdentity.bufferingPlanOrdinal) +
         ":worker-placement:" +
         llvm::Twine(static_cast<unsigned>(
             executable.actionIdentity.workerPlacementKind)) +
         ":worker-placement-plan:" +
         llvm::Twine(executable.actionIdentity.workerPlacementPlanOrdinal) +
         ":serialization:" +
         llvm::Twine(static_cast<unsigned>(
             executable.actionIdentity.serializationKind)) +
         ":baseline:" + llvm::Twine(executable.reservedBaseline ? 1 : 0) +
         ":communication-point:")
            .str();
    if (executable.actionIdentity.communicationPointIdentity) {
      header +=
          executable.actionIdentity.communicationPointIdentity->providerKey;
      header += ':';
      header += llvm::utostr(
          executable.actionIdentity.communicationPointIdentity->stableOrdinal);
    } else {
      header += "none";
    }
    header += '\n';
    hasher.update(header);
    for (const RankExecutable &rank : executable.variant.ranks) {
      std::string moduleText;
      llvm::raw_string_ostream stream(moduleText);
      stream << "rank:" << rank.getLogicalRank() << '\n';
      rank.getModule().print(stream);
      stream.flush();
      hasher.update(moduleText);
    }
  }
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

namespace {

static llvm::SmallVector<mlir::ModuleOp, 16>
getModuleViews(llvm::ArrayRef<mlir::OwningOpRef<mlir::ModuleOp>> modules) {
  llvm::SmallVector<mlir::ModuleOp, 16> views;
  views.reserve(modules.size());
  for (const mlir::OwningOpRef<mlir::ModuleOp> &module : modules)
    views.push_back(*module);
  return views;
}

static std::shared_ptr<const std::string>
captureSelectedTileIR(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return std::make_shared<const std::string>(std::move(text));
}

static void appendUniqueRankDiagnostics(llvm::raw_ostream &diagnostics,
                                        llvm::StringRef captured) {
  llvm::SmallVector<llvm::StringRef, 8> lines;
  captured.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  llvm::StringSet<> seen;
  for (llvm::StringRef line : lines)
    if (seen.insert(line).second)
      diagnostics << line << '\n';
}

struct OversizedSPMAllocation {
  mlir::MemRefType type;
  int64_t physicalBytes = 0;
};

static std::optional<OversizedSPMAllocation>
findOversizedSPMAllocation(mlir::ModuleOp module, int64_t capacityBytes) {
  std::optional<OversizedSPMAllocation> oversized;
  module.walk([&](mlir::memref::AllocOp allocation) {
    if (oversized || !isWaferSPMMemRefType(allocation.getType()))
      return;
    std::optional<WaferPhysicalTensorInfo> physical =
        computeWaferPhysicalTensorInfo(allocation.getType());
    if (physical && physical->physicalBytes > capacityBytes)
      oversized =
          OversizedSPMAllocation{allocation.getType(), physical->physicalBytes};
  });
  return oversized;
}

static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
cloneCanonicalParents(llvm::ArrayRef<mlir::ModuleOp> parents,
                      std::string *failureReason = nullptr) {
  wafer::support::ScopedCompileTimingSpan timing("transformation",
                                                 "coordinated-schedule-action",
                                                 "canonical-parent-clone");
  if (failureReason)
    failureReason->clear();
  if (parents.empty()) {
    if (failureReason)
      *failureReason = "empty canonical parent domain";
    return mlir::failure();
  }
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> clones;
  clones.reserve(parents.size());
  for (size_t rankIndex = 0; rankIndex < parents.size(); ++rankIndex) {
    mlir::ModuleOp parent = parents[rankIndex];
    if (!parent || containsTileDataflowOperations(parent.getOperation())) {
      if (failureReason)
        *failureReason =
            (llvm::Twine("canonical parent is unavailable at rank ") +
             llvm::Twine(rankIndex))
                .str();
      return mlir::failure();
    }
    clones.push_back(mlir::cast<mlir::ModuleOp>(parent->clone()));
    clearRankCandidatePhysicalFacts(*clones.back());
    if (mlir::failed(mlir::verify(*clones.back()))) {
      if (failureReason)
        *failureReason =
            (llvm::Twine("canonical parent clone verification failed at rank ") +
             llvm::Twine(rankIndex))
                .str();
      return mlir::failure();
    }
  }
  return clones;
}

static bool isSupportedSchedulingAction(
    mlir::ModuleOp module, TargetSchedulingMechanism mechanism,
    const TargetSchedulingCapabilityRegistry &registry) {
  llvm::Expected<TargetSchedulingWindowQuery> query =
      analysis::analyzeTargetSchedulingWindow(module, mechanism);
  if (!query) {
    llvm::consumeError(query.takeError());
    return false;
  }
  llvm::Expected<TargetSchedulingWindowDecision> decision =
      registry.query(*query);
  if (!decision) {
    llvm::consumeError(decision.takeError());
    return false;
  }
  return decision->legality == TargetSchedulingCapabilityState::Supported;
}

static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
deriveReadyOrderAction(llvm::ArrayRef<mlir::ModuleOp> parents) {
  wafer::support::ScopedCompileTimingSpan timing(
      "optimization", "coordinated-schedule-action", "ready-order");
  mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>> clones =
      cloneCanonicalParents(parents);
  if (mlir::failed(clones))
    return mlir::failure();
  bool anyMoved = false;
  for (mlir::OwningOpRef<mlir::ModuleOp> &clone : *clones) {
    anyMoved |=
        scheduleIndependentInstructionsByReadyOrder(clone->getOperation()) != 0;
    if (mlir::failed(placeRequiredNCCJoins(*clone)) ||
        mlir::failed(mlir::verify(*clone)))
      return mlir::failure();
  }
  if (!anyMoved)
    return mlir::failure();
  return clones;
}

struct FixedSlotLoopIdentity {
  std::vector<uint32_t> structuredPath;
  int64_t lowerBound = 0;
  int64_t upperBound = 0;
  int64_t step = 0;

  friend bool operator==(const FixedSlotLoopIdentity &lhs,
                         const FixedSlotLoopIdentity &rhs) {
    return lhs.structuredPath == rhs.structuredPath &&
           lhs.lowerBound == rhs.lowerBound &&
           lhs.upperBound == rhs.upperBound && lhs.step == rhs.step;
  }

  friend bool operator<(const FixedSlotLoopIdentity &lhs,
                        const FixedSlotLoopIdentity &rhs) {
    return std::tie(lhs.structuredPath, lhs.lowerBound, lhs.upperBound,
                    lhs.step) < std::tie(rhs.structuredPath, rhs.lowerBound,
                                         rhs.upperBound, rhs.step);
  }
};

static std::optional<std::vector<uint32_t>>
getStructuredOperationPath(mlir::Operation *operation,
                           mlir::ModuleOp containingModule) {
  if (!operation || !containingModule ||
      !containingModule->isAncestor(operation))
    return std::nullopt;

  llvm::SmallVector<std::array<uint32_t, 3>, 8> reversed;
  while (operation != containingModule.getOperation()) {
    mlir::Block *block = operation->getBlock();
    mlir::Region *region = block ? block->getParent() : nullptr;
    mlir::Operation *parent = region ? region->getParentOp() : nullptr;
    if (!block || !region || !parent)
      return std::nullopt;

    uint64_t regionOrdinal = 0;
    for (mlir::Region &candidate : parent->getRegions()) {
      if (&candidate == region)
        break;
      ++regionOrdinal;
    }
    uint64_t blockOrdinal = 0;
    for (mlir::Block &candidate : *region) {
      if (&candidate == block)
        break;
      ++blockOrdinal;
    }
    uint64_t operationOrdinal = 0;
    for (mlir::Operation &candidate : *block) {
      if (&candidate == operation)
        break;
      ++operationOrdinal;
    }
    if (regionOrdinal > std::numeric_limits<uint32_t>::max() ||
        blockOrdinal > std::numeric_limits<uint32_t>::max() ||
        operationOrdinal > std::numeric_limits<uint32_t>::max())
      return std::nullopt;
    reversed.push_back({static_cast<uint32_t>(regionOrdinal),
                        static_cast<uint32_t>(blockOrdinal),
                        static_cast<uint32_t>(operationOrdinal)});
    operation = parent;
  }

  std::vector<uint32_t> path;
  path.reserve(reversed.size() * 3);
  for (auto segment = reversed.rbegin(); segment != reversed.rend(); ++segment)
    path.insert(path.end(), segment->begin(), segment->end());
  return path;
}

static std::optional<FixedSlotLoopIdentity>
getFixedSlotLoopIdentity(mlir::scf::ForOp loop,
                         mlir::ModuleOp containingModule) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  std::optional<std::vector<uint32_t>> path =
      getStructuredOperationPath(loop.getOperation(), containingModule);
  if (!path || !lower || !upper || !step || *step <= 0 || *lower >= *upper)
    return std::nullopt;
  return FixedSlotLoopIdentity{std::move(*path), *lower, *upper, *step};
}

static llvm::SmallVector<FixedSlotLoopIdentity, 8>
getCommonFixedSlotLoopIdentities(llvm::ArrayRef<mlir::ModuleOp> parents) {
  llvm::SmallVector<FixedSlotLoopIdentity, 8> common;
  if (parents.empty())
    return common;
  mlir::ModuleOp firstParent = parents.front();
  firstParent.walk([&](mlir::scf::ForOp loop) {
    std::optional<FixedSlotLoopIdentity> identity =
        getFixedSlotLoopIdentity(loop, firstParent);
    if (identity)
      common.push_back(std::move(*identity));
  });
  llvm::sort(common);
  common.erase(std::unique(common.begin(), common.end()), common.end());

  for (mlir::ModuleOp parent : parents.drop_front()) {
    llvm::SmallVector<FixedSlotLoopIdentity, 8> rankIdentities;
    parent.walk([&](mlir::scf::ForOp loop) {
      std::optional<FixedSlotLoopIdentity> identity =
          getFixedSlotLoopIdentity(loop, parent);
      if (identity)
        rankIdentities.push_back(std::move(*identity));
    });
    llvm::sort(rankIdentities);
    rankIdentities.erase(
        std::unique(rankIdentities.begin(), rankIdentities.end()),
        rankIdentities.end());
    llvm::erase_if(common, [&](const FixedSlotLoopIdentity &identity) {
      return !llvm::is_contained(rankIdentities, identity);
    });
  }
  if (common.size() > kMaximumCoordinatedFixedSlotActions)
    common.resize(kMaximumCoordinatedFixedSlotActions);
  return common;
}

static mlir::scf::ForOp
findFixedSlotLoop(mlir::ModuleOp module,
                  const FixedSlotLoopIdentity &identity) {
  mlir::scf::ForOp match;
  bool ambiguous = false;
  module.walk([&](mlir::scf::ForOp loop) {
    std::optional<FixedSlotLoopIdentity> candidate =
        getFixedSlotLoopIdentity(loop, module);
    if (!candidate || !(*candidate == identity))
      return;
    if (match)
      ambiguous = true;
    else
      match = loop;
  });
  return ambiguous ? mlir::scf::ForOp{} : match;
}

static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
deriveFixedSlotAction(llvm::ArrayRef<mlir::ModuleOp> parents,
                      const FixedSlotLoopIdentity &loopIdentity,
                      const TargetSchedulingCapabilityRegistry &registry) {
  wafer::support::ScopedCompileTimingSpan timing(
      "optimization", "coordinated-schedule-action", "fixed-slot");
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> siblings;
  siblings.reserve(parents.size());
  for (mlir::ModuleOp parent : parents) {
    mlir::scf::ForOp loop = findFixedSlotLoop(parent, loopIdentity);
    if (!loop)
      return mlir::failure();

    std::string failureReason;
    mlir::FailureOr<StaticFixedSlotPipelineCandidate> candidate =
        mlir::failure();
    {
      mlir::ScopedDiagnosticHandler handler(
          parent.getContext(),
          [](mlir::Diagnostic &) { return mlir::success(); });
      candidate =
          deriveStaticFixedSlotPipelineCandidate(parent, loop, &failureReason);
    }
    if (mlir::failed(candidate) || candidate->stageCount < 2 ||
        candidate->slotAllocationCount < 2 ||
        !isSupportedSchedulingAction(*candidate->module,
                                     TargetSchedulingMechanism::StaticFixedSlot,
                                     registry) ||
        mlir::failed(mlir::verify(*candidate->module)))
      return mlir::failure();
    siblings.push_back(std::move(candidate->module));
  }
  return siblings;
}

static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
deriveWorkerAction(llvm::ArrayRef<mlir::ModuleOp> parents,
                   CoordinatedBufferingKind bufferingKind,
                   const TargetSchedulingCapabilityRegistry &registry) {
  wafer::support::ScopedCompileTimingSpan timing("optimization",
                                                 "coordinated-schedule-action",
                                                 "disjoint-worker-placement");
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> siblings;
  siblings.reserve(parents.size());
  for (mlir::ModuleOp parent : parents) {
    std::string failureReason;
    mlir::FailureOr<NCCWorkerPlacementCandidate> candidate = mlir::failure();
    {
      mlir::ScopedDiagnosticHandler handler(
          parent.getContext(),
          [](mlir::Diagnostic &) { return mlir::success(); });
      candidate =
          deriveDisjointNCCWorkerPlacementCandidate(parent, &failureReason);
    }
    if (mlir::failed(candidate) ||
        llvm::popcount(candidate->participantMask) < 2)
      return mlir::failure();

    bool hasDirectDTE = false;
    candidate->module->walk([&](WaferInstructionOpInterface instruction) {
      hasDirectDTE |= instruction.getInstructionFamily() == InstrFamily::DTE;
    });
    TargetSchedulingMechanism mechanism =
        bufferingKind == CoordinatedBufferingKind::StaticFixedSlot
            ? TargetSchedulingMechanism::StaticFixedSlot
        : hasDirectDTE ? TargetSchedulingMechanism::DirectDTEOverlap
                       : TargetSchedulingMechanism::WorkerPlacement;
    if (!isSupportedSchedulingAction(*candidate->module, mechanism, registry) ||
        mlir::failed(mlir::verify(*candidate->module)))
      return mlir::failure();
    siblings.push_back(std::move(candidate->module));
  }
  return siblings;
}

/// Derive a disposable serialized sibling from already-derived fixed-slot
/// parents. A sibling exists only when every rank contains at least one exact
/// same-block Direct-DTE issue -> CT/NE -> single-use wait window and moving
/// those compute operations after their wait converges. The ordinary action
/// evaluator subsequently rebuilds completion and reruns every physical and
/// target gate from this rewritten current IR.
static mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>>
deriveSerializedDirectDTEComputeAction(llvm::ArrayRef<mlir::ModuleOp> parents) {
  wafer::support::ScopedCompileTimingSpan timing(
      "optimization", "coordinated-schedule-action",
      "serialized-direct-dte-compute");
  mlir::FailureOr<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>> clones =
      cloneCanonicalParents(parents);
  if (mlir::failed(clones))
    return mlir::failure();

  for (mlir::OwningOpRef<mlir::ModuleOp> &clone : *clones) {
    unsigned operationCount = 0;
    clone->walk([&](mlir::Operation *) { ++operationCount; });
    bool foundWindow = false;
    bool converged = false;
    for (unsigned iteration = 0; iteration <= operationCount; ++iteration) {
      llvm::SmallVector<
          std::pair<InstrDTEWaitOp, llvm::SmallVector<mlir::Operation *, 4>>, 8>
          windows;
      clone->walk([&](mlir::Operation *issue) {
        mlir::Value token;
        if (auto send = mlir::dyn_cast<InstrDTESendOp>(issue))
          token = send.getToken();
        else if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(issue))
          token = recv.getToken();
        else
          return;
        if (!token.hasOneUse())
          return;
        auto wait = mlir::dyn_cast<InstrDTEWaitOp>(*token.getUsers().begin());
        if (!wait || wait->getBlock() != issue->getBlock() ||
            !issue->isBeforeInBlock(wait))
          return;

        llvm::SmallVector<mlir::Operation *, 4> computeOperations;
        for (mlir::Operation *between = issue->getNextNode();
             between && between != wait.getOperation();
             between = between->getNextNode()) {
          auto instruction =
              mlir::dyn_cast<WaferInstructionOpInterface>(between);
          if (instruction &&
              (instruction.getInstructionFamily() == InstrFamily::CT ||
               instruction.getInstructionFamily() == InstrFamily::NE))
            computeOperations.push_back(between);
        }
        if (!computeOperations.empty())
          windows.emplace_back(wait, std::move(computeOperations));
      });
      if (windows.empty()) {
        converged = true;
        break;
      }
      foundWindow = true;
      bool moved = false;
      for (auto &[wait, computeOperations] : windows) {
        mlir::Operation *insertionAnchor = wait.getOperation();
        for (mlir::Operation *compute : computeOperations) {
          if (compute->getBlock() != wait->getBlock() ||
              !compute->isBeforeInBlock(wait))
            continue;
          compute->moveAfter(insertionAnchor);
          insertionAnchor = compute;
          moved = true;
        }
      }
      if (!moved)
        break;
    }
    if (!foundWindow || !converged || mlir::failed(mlir::verify(*clone)))
      return mlir::failure();
  }
  return clones;
}

static bool chargeActionMaterialization(uint64_t rankCount, uint64_t &work) {
  if (rankCount > std::numeric_limits<uint64_t>::max() - work)
    return false;
  work += rankCount;
  return true;
}

struct ExecutableFinalizationWork {
  uint64_t tileLowerings = 0;
  uint64_t actionMaterializations = 0;
  uint64_t spmProblems = 0;
  uint64_t exactActionAttempts = 0;
};

struct CoordinatedScheduleRecipe {
  uint32_t stableOrdinal = 0;
  CoordinatedScheduleActionIdentity actionIdentity;
  /// Query-local point owned by walkCoordinatedScheduleActions. It contains no
  /// pointer into the canonical IR and is never persisted or reconstructed
  /// from its opaque identity.
  const CoordinatedCommunicationActionPoint *communicationPoint = nullptr;
  std::optional<FixedSlotLoopIdentity> fixedSlotLoopIdentity;
  uint64_t fixedSlotTripCount = 0;
  uint64_t fixedSlotOperationSites = 0;
};

static std::string
getFixedSlotCoverageIdentity(const FixedSlotLoopIdentity &identity) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "path";
  for (uint32_t component : identity.structuredPath)
    stream << '.' << component;
  stream << ":bounds=" << identity.lowerBound << ':' << identity.upperBound
         << ':' << identity.step;
  return text;
}

static std::optional<uint64_t>
getFixedSlotTripCount(const FixedSlotLoopIdentity &identity) {
  if (identity.step <= 0 || identity.lowerBound >= identity.upperBound)
    return std::nullopt;
  __int128 span = static_cast<__int128>(identity.upperBound) -
                  static_cast<__int128>(identity.lowerBound);
  __int128 count = (span + static_cast<__int128>(identity.step) - 1) /
                   static_cast<__int128>(identity.step);
  if (count <= 0 ||
      count > static_cast<__int128>(std::numeric_limits<uint64_t>::max()))
    return std::nullopt;
  return static_cast<uint64_t>(count);
}

static uint64_t
countFixedSlotOperationSites(llvm::ArrayRef<mlir::ModuleOp> parents,
                             const FixedSlotLoopIdentity &identity) {
  uint64_t aggregate = 0;
  for (mlir::ModuleOp parent : parents) {
    mlir::scf::ForOp loop = findFixedSlotLoop(parent, identity);
    if (!loop)
      return 0;
    uint64_t rankSites = 0;
    loop.getBody()->walk([&](mlir::Operation *) { ++rankSites; });
    if (rankSites > std::numeric_limits<uint64_t>::max() - aggregate)
      return 0;
    aggregate += rankSites;
  }
  return aggregate;
}

static std::optional<uint64_t>
knownValue(const analysis::ScheduleCostMetric &metric) {
  if (!metric.isKnown())
    return std::nullopt;
  return metric.value;
}

static CoordinatedScheduleActionEstimate deriveScheduleRecipeEstimate(
    const CoordinatedScheduleRecipe &recipe,
    const analysis::WholeCardInstructionProgramCost &canonicalCost) {
  CoordinatedScheduleActionEstimate estimate;
  estimate.stableOrdinal = recipe.stableOrdinal;
  estimate.readyOrderKind = recipe.actionIdentity.readyOrderKind;
  estimate.bufferingKind = recipe.actionIdentity.bufferingKind;
  estimate.workerPlacementKind = recipe.actionIdentity.workerPlacementKind;
  estimate.serializationKind = recipe.actionIdentity.serializationKind;
  estimate.communicationPointIdentity =
      recipe.actionIdentity.communicationPointIdentity;
  if (recipe.fixedSlotLoopIdentity)
    estimate.fixedSlotCoverageIdentity =
        getFixedSlotCoverageIdentity(*recipe.fixedSlotLoopIdentity);
  auto set = [&](CoordinatedScheduleActionEstimateDimension dimension,
                 const analysis::ScheduleCostMetric &metric) {
    estimate.values[static_cast<size_t>(dimension)] = knownValue(metric);
  };
  set(CoordinatedScheduleActionEstimateDimension::QualifiedOverlapWindows,
      canonicalCost.aggregateQualifiedOverlapWindowCount);
  set(CoordinatedScheduleActionEstimateDimension::DirectDTEOverlapWindows,
      canonicalCost.aggregateDirectDTEComputeOverlapWindowCount);
  set(CoordinatedScheduleActionEstimateDimension::SteadyStateParticipantWaits,
      canonicalCost.aggregateSteadyStateNCCParticipantWaitCount);
  // Static sites remain useful when an enclosing dynamic trip count prevents
  // exact execution multiplicity. The model identity explicitly classifies
  // them as a coverage estimate rather than exact final cost.
  set(CoordinatedScheduleActionEstimateDimension::InstructionSites,
      canonicalCost.aggregateWork.instructions.staticSites);
  set(CoordinatedScheduleActionEstimateDimension::EventSites,
      canonicalCost.aggregateWork.asynchronousEvents.staticSites);
  set(CoordinatedScheduleActionEstimateDimension::MaximumDependencyDepth,
      canonicalCost.maximumRankDataDependencyDepth);
  set(CoordinatedScheduleActionEstimateDimension::ReadyOrderInversions,
      canonicalCost.aggregateReadyOrderPriorityInversions);
  set(CoordinatedScheduleActionEstimateDimension::SPMMovementBytes,
      canonicalCost.aggregateSPMMovementBytes);
  // Fixed-slot recipes have not been cloned or transformed. Their exact
  // static trip count and current loop-body site count are nevertheless
  // read-only structural coverage evidence, so use them only to rank multiple
  // loop recipes within the same typed schedule shape.
  if (recipe.fixedSlotLoopIdentity) {
    estimate.values[static_cast<size_t>(
        CoordinatedScheduleActionEstimateDimension::QualifiedOverlapWindows)] =
        recipe.fixedSlotTripCount;
    estimate.values[static_cast<size_t>(
        CoordinatedScheduleActionEstimateDimension::InstructionSites)] =
        recipe.fixedSlotOperationSites;
  }
  return estimate;
}

struct CoordinatedScheduleActionShape {
  uint8_t readyOrder = 0;
  uint8_t buffering = 0;
  uint8_t workerPlacement = 0;
  uint8_t serialization = 0;
  std::optional<CoordinatedCommunicationActionPointIdentity>
      communicationPointIdentity;

  friend bool operator<(const CoordinatedScheduleActionShape &lhs,
                        const CoordinatedScheduleActionShape &rhs) {
    auto lhsAxes = std::tie(lhs.readyOrder, lhs.buffering, lhs.workerPlacement,
                            lhs.serialization);
    auto rhsAxes = std::tie(rhs.readyOrder, rhs.buffering, rhs.workerPlacement,
                            rhs.serialization);
    if (lhsAxes != rhsAxes)
      return lhsAxes < rhsAxes;
    if (lhs.communicationPointIdentity.has_value() !=
        rhs.communicationPointIdentity.has_value())
      return lhs.communicationPointIdentity.has_value() <
             rhs.communicationPointIdentity.has_value();
    if (!lhs.communicationPointIdentity)
      return false;
    return std::tie(lhs.communicationPointIdentity->providerKey,
                    lhs.communicationPointIdentity->stableOrdinal) <
           std::tie(rhs.communicationPointIdentity->providerKey,
                    rhs.communicationPointIdentity->stableOrdinal);
  }
};

static std::optional<CoordinatedScheduleActionShape>
getScheduleActionShape(const CoordinatedScheduleActionEstimate &estimate) {
  std::optional<uint8_t> readyOrder;
  if (estimate.readyOrderKind == CoordinatedReadyOrderKind::Canonical)
    readyOrder = 0;
  else if (estimate.readyOrderKind == CoordinatedReadyOrderKind::ReadyOrder)
    readyOrder = 1;

  std::optional<uint8_t> buffering;
  if (estimate.bufferingKind == CoordinatedBufferingKind::Single)
    buffering = 0;
  else if (estimate.bufferingKind == CoordinatedBufferingKind::StaticFixedSlot)
    buffering = 1;

  std::optional<uint8_t> placement;
  if (estimate.workerPlacementKind == CoordinatedWorkerPlacementKind::Unplaced)
    placement = 0;
  else if (estimate.workerPlacementKind ==
           CoordinatedWorkerPlacementKind::DisjointComponents)
    placement = 1;

  std::optional<uint8_t> serialization;
  if (estimate.serializationKind ==
      CoordinatedScheduleSerializationKind::Unchanged)
    serialization = 0;
  else if (estimate.serializationKind ==
           CoordinatedScheduleSerializationKind::DirectDTEComputeWindows)
    serialization = 1;

  if (!readyOrder || !buffering || !placement || !serialization ||
      (estimate.communicationPointIdentity &&
       estimate.communicationPointIdentity->providerKey.empty()))
    return std::nullopt;
  return CoordinatedScheduleActionShape{*readyOrder, *buffering, *placement,
                                        *serialization,
                                        estimate.communicationPointIdentity};
}

static bool isMaximizedEstimateDimension(
    CoordinatedScheduleActionEstimateDimension dimension) {
  return dimension == CoordinatedScheduleActionEstimateDimension::
                          QualifiedOverlapWindows ||
         dimension == CoordinatedScheduleActionEstimateDimension::
                          DirectDTEOverlapWindows;
}

static bool
isBetterScheduleActionEstimate(const CoordinatedScheduleActionEstimate &left,
                               const CoordinatedScheduleActionEstimate &right) {
  for (size_t index = 0;
       index < kCoordinatedScheduleActionEstimateDimensionCount; ++index) {
    const std::optional<uint64_t> &leftValue = left.values[index];
    const std::optional<uint64_t> &rightValue = right.values[index];
    if (leftValue.has_value() != rightValue.has_value())
      return leftValue.has_value();
    if (!leftValue || *leftValue == *rightValue)
      continue;
    auto dimension =
        static_cast<CoordinatedScheduleActionEstimateDimension>(index);
    return isMaximizedEstimateDimension(dimension) ? *leftValue > *rightValue
                                                   : *leftValue < *rightValue;
  }
  if (left.fixedSlotCoverageIdentity != right.fixedSlotCoverageIdentity)
    return left.fixedSlotCoverageIdentity < right.fixedSlotCoverageIdentity;
  return left.stableOrdinal < right.stableOrdinal;
}

static CoordinatedWorkEstimate
getActualExecutableFinalizationWork(const ExecutableFinalizationWork &work,
                                    uint64_t rankCount) {
  CoordinatedWorkEstimate actual;
  actual.set(CoordinatedWorkKind::TileToInstrLowering, work.tileLowerings);
  actual.set(CoordinatedWorkKind::ExecutableScheduleAction,
             work.actionMaterializations);
  actual.set(CoordinatedWorkKind::SPMAllocationProblem, work.spmProblems);
  actual.set(CoordinatedWorkKind::DDRAllocationDomain,
             work.exactActionAttempts * rankCount);
  actual.set(CoordinatedWorkKind::TransportValidation,
             work.exactActionAttempts);
  actual.set(CoordinatedWorkKind::ABIValidation,
             work.exactActionAttempts * rankCount);
  return actual;
}

} // namespace

mlir::LogicalResult CoordinatedExecutableFinalizationLaneCoordinator::
    recordMandatoryBaselineAccepted() {
  if (baselineAccepted || attemptCount != 0 || exactAcceptedCount != 0)
    return mlir::failure();
  baselineAccepted = true;
  preferSeed = true;
  attemptCount = 1;
  exactAcceptedCount = 1;
  return mlir::success();
}

std::optional<CoordinatedExecutableFinalizationLane>
CoordinatedExecutableFinalizationLaneCoordinator::chooseNextLane(
    bool seedAvailable, bool expansionAvailable) const {
  if (!canAttempt() || (!seedAvailable && !expansionAvailable))
    return std::nullopt;
  if (seedAvailable && expansionAvailable)
    return preferSeed ? CoordinatedExecutableFinalizationLane::Seed
                      : CoordinatedExecutableFinalizationLane::Expansion;
  return seedAvailable ? CoordinatedExecutableFinalizationLane::Seed
                       : CoordinatedExecutableFinalizationLane::Expansion;
}

mlir::LogicalResult
CoordinatedExecutableFinalizationLaneCoordinator::recordAttempt(
    CoordinatedExecutableFinalizationLane lane, bool exactAccepted) {
  if (!baselineAccepted || !canAttempt())
    return mlir::failure();
  ++attemptCount;
  if (exactAccepted)
    ++exactAcceptedCount;
  preferSeed = lane == CoordinatedExecutableFinalizationLane::Expansion;
  return exactAcceptedCount <= kMaximumCoordinatedExactScheduleActions
             ? mlir::success()
             : mlir::failure();
}

bool CoordinatedExecutableFinalizationLaneCoordinator::canAttempt() const {
  return baselineAccepted &&
         attemptCount <
             kMaximumCoordinatedScheduleRecipeMaterializationAttempts &&
         exactAcceptedCount < kMaximumCoordinatedExactScheduleActions;
}

bool isRecoverableCoordinatedExecutableSetupFailure(
    const CoordinatedExecutableAdmissionFailure &failure) {
  return failure.kind ==
             CoordinatedExecutableAdmissionFailureKind::RankFinalization ||
         failure.kind ==
             CoordinatedExecutableAdmissionFailureKind::SPMAllocation;
}

mlir::FailureOr<llvm::SmallVector<uint32_t, 8>>
selectCoordinatedScheduleActionEstimateBeam(
    llvm::ArrayRef<CoordinatedScheduleActionEstimate> estimates) {
  if (estimates.empty())
    return mlir::failure();

  llvm::SmallSet<uint32_t, 32> seenOrdinals;
  std::optional<size_t> baselineIndex;
  std::map<CoordinatedScheduleActionShape, size_t> shapeRepresentatives;
  for (auto [index, estimate] : llvm::enumerate(estimates)) {
    if (!seenOrdinals.insert(estimate.stableOrdinal).second)
      return mlir::failure();
    std::optional<CoordinatedScheduleActionShape> shape =
        getScheduleActionShape(estimate);
    if (!shape)
      return mlir::failure();
    if (estimate.stableOrdinal == 0)
      baselineIndex = index;
    auto representative = shapeRepresentatives.find(*shape);
    if (representative == shapeRepresentatives.end() ||
        isBetterScheduleActionEstimate(estimate,
                                       estimates[representative->second]))
      shapeRepresentatives[*shape] = index;
  }
  if (!baselineIndex ||
      estimates[*baselineIndex].readyOrderKind !=
          CoordinatedReadyOrderKind::Canonical ||
      estimates[*baselineIndex].bufferingKind !=
          CoordinatedBufferingKind::Single ||
      estimates[*baselineIndex].workerPlacementKind !=
          CoordinatedWorkerPlacementKind::Unplaced ||
      estimates[*baselineIndex].serializationKind !=
          CoordinatedScheduleSerializationKind::Unchanged ||
      estimates[*baselineIndex].communicationPointIdentity)
    return mlir::failure();

  // Ordinal zero is a semantic fallback, not merely the best estimate in its
  // shape. Force its shape representative to the baseline before adding the
  // other typed scheduling shapes.
  std::optional<CoordinatedScheduleActionShape> baselineShape =
      getScheduleActionShape(estimates[*baselineIndex]);
  shapeRepresentatives[*baselineShape] = *baselineIndex;

  llvm::SmallSet<uint32_t, 8> selected;
  selected.insert(0);
  for (const auto &[shape, representative] : shapeRepresentatives) {
    if (selected.size() >= kMaximumCoordinatedExactScheduleActions)
      break;
    selected.insert(estimates[representative].stableOrdinal);
  }

  llvm::SmallVector<size_t, 36> modelOrder;
  modelOrder.reserve(estimates.size());
  for (size_t index = 0; index < estimates.size(); ++index)
    modelOrder.push_back(index);
  llvm::sort(modelOrder, [&](size_t left, size_t right) {
    return isBetterScheduleActionEstimate(estimates[left], estimates[right]);
  });
  for (size_t index : modelOrder) {
    if (selected.size() >= kMaximumCoordinatedExactScheduleActions)
      break;
    selected.insert(estimates[index].stableOrdinal);
  }

  llvm::SmallVector<uint32_t, 8> result(selected.begin(), selected.end());
  llvm::sort(result);
  if (result.empty() || result.front() != 0 ||
      result.size() > kMaximumCoordinatedExactScheduleActions)
    return mlir::failure();
  return result;
}

static mlir::FailureOr<std::vector<CoordinatedScheduleRecipe>>
enumerateCoordinatedScheduleRecipes(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const OptimizationConfig &optimizations, bool baselineOnly,
    CoordinatedScheduleActionFamily family,
    bool schedulingCapabilitiesAvailable,
    llvm::ArrayRef<const CoordinatedCommunicationActionPoint *>
        communicationPoints) {
  if (canonicalInstrParents.empty())
    return mlir::failure();

  const bool productionPolicy =
      !baselineOnly && optimizations.isProduction();
  const bool productionFamily =
      family == CoordinatedScheduleActionFamily::Production ||
      family == CoordinatedScheduleActionFamily::
                    ProductionWithSerializedDirectDTECompute;
  const bool includeReadyOrder =
      productionPolicy &&
      (productionFamily ||
       family == CoordinatedScheduleActionFamily::ReadyOrderQualification);
  const bool includeStaticFixedSlot =
      productionPolicy &&
      (productionFamily ||
       family ==
           CoordinatedScheduleActionFamily::StaticFixedSlotQualification);
  const bool includeDisjointWorkerPlacement =
      productionPolicy &&
      (productionFamily ||
       family == CoordinatedScheduleActionFamily::
                     DisjointWorkerPlacementQualification);
  const bool includeSerializedDirectDTECompute =
      productionPolicy &&
      family == CoordinatedScheduleActionFamily::
                    ProductionWithSerializedDirectDTECompute;

  std::vector<CoordinatedScheduleRecipe> recipes;
  recipes.reserve(kMaximumCoordinatedScheduleActions);
  auto addRecipe =
      [&](CoordinatedReadyOrderKind readyOrderKind,
          CoordinatedBufferingKind bufferingKind, uint32_t bufferingPlanOrdinal,
          CoordinatedWorkerPlacementKind workerPlacementKind,
          uint32_t workerPlacementPlanOrdinal,
          CoordinatedScheduleSerializationKind serializationKind,
          std::optional<FixedSlotLoopIdentity> fixedSlotLoopIdentity =
              std::nullopt) -> mlir::LogicalResult {
    if (recipes.size() >= kMaximumCoordinatedScheduleActions)
      return mlir::failure();
    CoordinatedScheduleRecipe recipe;
    recipe.stableOrdinal = static_cast<uint32_t>(recipes.size());
    recipe.actionIdentity.readyOrderKind = readyOrderKind;
    recipe.actionIdentity.bufferingKind = bufferingKind;
    recipe.actionIdentity.bufferingPlanOrdinal = bufferingPlanOrdinal;
    recipe.actionIdentity.workerPlacementKind = workerPlacementKind;
    recipe.actionIdentity.workerPlacementPlanOrdinal =
        workerPlacementPlanOrdinal;
    recipe.actionIdentity.serializationKind = serializationKind;
    recipe.fixedSlotLoopIdentity = std::move(fixedSlotLoopIdentity);
    if (recipe.fixedSlotLoopIdentity) {
      std::optional<uint64_t> tripCount =
          getFixedSlotTripCount(*recipe.fixedSlotLoopIdentity);
      if (!tripCount)
        return mlir::failure();
      recipe.fixedSlotTripCount = *tripCount;
      recipe.fixedSlotOperationSites = countFixedSlotOperationSites(
          canonicalInstrParents, *recipe.fixedSlotLoopIdentity);
      if (recipe.fixedSlotOperationSites == 0)
        return mlir::failure();
    }
    recipes.push_back(std::move(recipe));
    return mlir::success();
  };

  llvm::SmallVector<FixedSlotLoopIdentity, 8> fixedSlotLoops;
  if (includeStaticFixedSlot && schedulingCapabilitiesAvailable)
    fixedSlotLoops = getCommonFixedSlotLoopIdentities(canonicalInstrParents);

  auto appendReadyOrderDomain =
      [&](CoordinatedReadyOrderKind readyOrderKind) -> mlir::LogicalResult {
    if (mlir::failed(
            addRecipe(readyOrderKind, CoordinatedBufferingKind::Single, 0,
                      CoordinatedWorkerPlacementKind::Unplaced, 0,
                      CoordinatedScheduleSerializationKind::Unchanged)))
      return mlir::failure();
    if (includeDisjointWorkerPlacement && schedulingCapabilitiesAvailable &&
        mlir::failed(
            addRecipe(readyOrderKind, CoordinatedBufferingKind::Single, 0,
                      CoordinatedWorkerPlacementKind::DisjointComponents, 1,
                      CoordinatedScheduleSerializationKind::Unchanged)))
      return mlir::failure();

    for (auto [loopIndex, loopIdentity] : llvm::enumerate(fixedSlotLoops)) {
      const uint32_t planOrdinal = static_cast<uint32_t>(loopIndex) + 1;
      if (mlir::failed(addRecipe(
              readyOrderKind, CoordinatedBufferingKind::StaticFixedSlot,
              planOrdinal, CoordinatedWorkerPlacementKind::Unplaced, 0,
              CoordinatedScheduleSerializationKind::Unchanged, loopIdentity)))
        return mlir::failure();
      if (includeDisjointWorkerPlacement &&
          mlir::failed(addRecipe(
              readyOrderKind, CoordinatedBufferingKind::StaticFixedSlot,
              planOrdinal, CoordinatedWorkerPlacementKind::DisjointComponents,
              1, CoordinatedScheduleSerializationKind::Unchanged,
              loopIdentity)))
        return mlir::failure();
      if (includeSerializedDirectDTECompute &&
          mlir::failed(addRecipe(
              readyOrderKind, CoordinatedBufferingKind::StaticFixedSlot,
              planOrdinal, CoordinatedWorkerPlacementKind::Unplaced, 0,
              CoordinatedScheduleSerializationKind::DirectDTEComputeWindows,
              loopIdentity)))
        return mlir::failure();
    }
    return mlir::success();
  };

  // Recipe zero is always the immutable canonical fallback.
  if (mlir::failed(
          appendReadyOrderDomain(CoordinatedReadyOrderKind::Canonical)))
    return mlir::failure();
  if (includeReadyOrder &&
      mlir::failed(
          appendReadyOrderDomain(CoordinatedReadyOrderKind::ReadyOrder)))
    return mlir::failure();
  if (recipes.empty() || recipes.front().stableOrdinal != 0 ||
      recipes.front().actionIdentity.readyOrderKind !=
          CoordinatedReadyOrderKind::Canonical ||
      recipes.front().actionIdentity.bufferingKind !=
          CoordinatedBufferingKind::Single ||
      recipes.front().actionIdentity.workerPlacementKind !=
          CoordinatedWorkerPlacementKind::Unplaced ||
      recipes.front().communicationPoint ||
      recipes.front().actionIdentity.communicationPointIdentity)
    return mlir::failure();

  // Communication providers add one orthogonal structural domain. Keep the
  // canonical schedule domain first so ordinal zero remains the mandatory
  // fallback, then copy only cheap recipes. No IR is cloned here, and common
  // coordination carries the point object itself rather than interpreting its
  // provider key or ordinal.
  const size_t scheduleRecipeCount = recipes.size();
  if (communicationPoints.size() >
          kMaximumCoordinatedCommunicationActionPoints ||
      scheduleRecipeCount > std::numeric_limits<uint32_t>::max() /
                                (communicationPoints.size() + 1))
    return mlir::failure();
  recipes.reserve(scheduleRecipeCount * (communicationPoints.size() + 1));
  for (const CoordinatedCommunicationActionPoint *point : communicationPoints) {
    if (!point || point->getIdentity().providerKey.empty())
      return mlir::failure();
    for (size_t recipeIndex = 0; recipeIndex < scheduleRecipeCount;
         ++recipeIndex) {
      CoordinatedScheduleRecipe recipe = recipes[recipeIndex];
      recipe.stableOrdinal = static_cast<uint32_t>(recipes.size());
      recipe.communicationPoint = point;
      recipe.actionIdentity.communicationPointIdentity = point->getIdentity();
      recipes.push_back(std::move(recipe));
    }
  }
  return recipes;
}

static mlir::FailureOr<CoordinatedScheduleAction>
materializeCoordinatedScheduleRecipe(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const CoordinatedScheduleRecipe &recipe,
    const TargetSchedulingCapabilityRegistry *registry,
    const frontend::FrontendProgramVerificationResult *program,
    std::string *failureReason = nullptr) {
  if (failureReason)
    failureReason->clear();
  std::optional<std::vector<mlir::OwningOpRef<mlir::ModuleOp>>> current;
  llvm::SmallVector<mlir::ModuleOp, 16> currentViews(
      canonicalInstrParents.begin(), canonicalInstrParents.end());
  auto replaceCurrent =
      [&](std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules) {
        current.emplace(std::move(modules));
        currentViews = getModuleViews(*current);
      };

  const bool hasCommunicationIdentity =
      recipe.actionIdentity.communicationPointIdentity.has_value();
  if ((recipe.communicationPoint != nullptr) != hasCommunicationIdentity)
    return mlir::failure();
  if (recipe.communicationPoint) {
    if (!program || !(recipe.communicationPoint->getIdentity() ==
                      *recipe.actionIdentity.communicationPointIdentity))
      return mlir::failure();
    std::string failureReason;
    mlir::FailureOr<CoordinatedCommunicationAction> communication =
        materializeCoordinatedCommunicationAction(
            canonicalInstrParents, *program, *recipe.communicationPoint,
            &failureReason);
    if (mlir::failed(communication) ||
        !(communication->identity ==
          *recipe.actionIdentity.communicationPointIdentity))
      return mlir::failure();
    replaceCurrent(std::move(communication->rankModules));
  }

  if (recipe.actionIdentity.readyOrderKind ==
      CoordinatedReadyOrderKind::ReadyOrder) {
    auto ready = deriveReadyOrderAction(currentViews);
    if (mlir::failed(ready))
      return mlir::failure();
    replaceCurrent(std::move(*ready));
  }
  if (recipe.fixedSlotLoopIdentity) {
    if (!registry)
      return mlir::failure();
    auto fixed = deriveFixedSlotAction(
        currentViews, *recipe.fixedSlotLoopIdentity, *registry);
    if (mlir::failed(fixed))
      return mlir::failure();
    replaceCurrent(std::move(*fixed));
  }
  if (recipe.actionIdentity.workerPlacementKind ==
      CoordinatedWorkerPlacementKind::DisjointComponents) {
    if (!registry)
      return mlir::failure();
    auto placed = deriveWorkerAction(
        currentViews, recipe.actionIdentity.bufferingKind, *registry);
    if (mlir::failed(placed))
      return mlir::failure();
    replaceCurrent(std::move(*placed));
  }
  if (recipe.actionIdentity.serializationKind ==
      CoordinatedScheduleSerializationKind::DirectDTEComputeWindows) {
    auto serialized = deriveSerializedDirectDTEComputeAction(currentViews);
    if (mlir::failed(serialized))
      return mlir::failure();
    replaceCurrent(std::move(*serialized));
  }
  if (!current) {
    auto baseline =
        cloneCanonicalParents(canonicalInstrParents, failureReason);
    if (mlir::failed(baseline))
      return mlir::failure();
    replaceCurrent(std::move(*baseline));
  }

  CoordinatedScheduleAction action;
  action.stableOrdinal = recipe.stableOrdinal;
  action.readyOrderKind = recipe.actionIdentity.readyOrderKind;
  action.bufferingKind = recipe.actionIdentity.bufferingKind;
  action.bufferingPlanOrdinal = recipe.actionIdentity.bufferingPlanOrdinal;
  action.workerPlacementKind = recipe.actionIdentity.workerPlacementKind;
  action.workerPlacementPlanOrdinal =
      recipe.actionIdentity.workerPlacementPlanOrdinal;
  action.serializationKind = recipe.actionIdentity.serializationKind;
  action.communicationPointIdentity =
      recipe.actionIdentity.communicationPointIdentity;
  action.rankModules = std::move(*current);
  return action;
}

mlir::FailureOr<std::vector<CoordinatedScheduleAction>>
deriveCoordinatedScheduleActions(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const OptimizationConfig &optimizations, uint64_t *materializationWork,
    CoordinatedScheduleActionFamily family,
    CoordinatedScheduleRecipeStatistics *statistics,
    const frontend::FrontendProgramVerificationResult *program,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders) {
  std::vector<CoordinatedScheduleAction> actions;
  actions.reserve(kMaximumCoordinatedExactScheduleActions);
  mlir::LogicalResult result = walkCoordinatedScheduleActions(
      canonicalInstrParents, optimizations,
      [&](CoordinatedScheduleAction &&action) {
        actions.push_back(std::move(action));
        return CoordinatedScheduleActionConsumption::Accepted;
      },
      materializationWork, /*baselineOnly=*/false, family, statistics, program,
      communicationProviders);
  if (mlir::failed(result))
    return mlir::failure();
  return actions;
}

namespace {

struct CoordinatedScheduleRecipePlan {
  std::optional<TargetSchedulingCapabilityRegistry> registry;
  CoordinatedCommunicationActionPoints communicationPointOwners;
  std::vector<CoordinatedScheduleRecipe> recipes;
  llvm::SmallSet<uint32_t, 8> initiallySelected;
  llvm::SmallVector<uint32_t, 16> materializationOrder;
  size_t nextRecipe = 0;
};

static mlir::FailureOr<CoordinatedScheduleRecipePlan>
buildCoordinatedScheduleRecipePlan(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const OptimizationConfig &optimizations, bool baselineOnly,
    CoordinatedScheduleActionFamily family,
    const frontend::FrontendProgramVerificationResult *program,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders,
    CoordinatedScheduleRecipeStatistics *statistics) {
  if (canonicalInstrParents.empty())
    return mlir::failure();

  CoordinatedScheduleRecipePlan plan;
  llvm::Expected<TargetSchedulingCapabilityRegistry> registryResult =
      getTargetSchedulingCapabilityRegistry();
  if (registryResult)
    plan.registry.emplace(std::move(*registryResult));
  else
    llvm::consumeError(registryResult.takeError());

  llvm::SmallVector<const CoordinatedCommunicationActionPoint *, 8>
      communicationPoints;
  if (!baselineOnly && optimizations.isProduction() &&
      !communicationProviders.empty()) {
    if (!program || communicationProviders.size() >
                        kMaximumCoordinatedCommunicationActionPoints)
      return mlir::failure();

    llvm::SmallVector<const CoordinatedCommunicationActionProvider *, 8>
        orderedProviders(communicationProviders.begin(),
                         communicationProviders.end());
    if (llvm::any_of(orderedProviders,
                     [](const auto *provider) { return provider == nullptr; }))
      return mlir::failure();
    llvm::sort(orderedProviders, [](const auto *lhs, const auto *rhs) {
      return lhs->getStableKey() < rhs->getStableKey();
    });

    std::set<std::pair<std::string, uint32_t>> seenPointIdentities;
    llvm::StringRef previousProviderKey;
    for (const CoordinatedCommunicationActionProvider *provider :
         orderedProviders) {
      const llvm::StringRef providerKey = provider->getStableKey();
      if (providerKey.empty() || providerKey == previousProviderKey)
        return mlir::failure();
      previousProviderKey = providerKey;

      CoordinatedCommunicationActionPoints providerPoints;
      std::string failureReason;
      if (mlir::failed(provider->query(canonicalInstrParents, *program,
                                       providerPoints, &failureReason)))
        return mlir::failure();
      if (plan.communicationPointOwners.size() + providerPoints.size() >
          kMaximumCoordinatedCommunicationActionPoints)
        return mlir::failure();
      if (llvm::any_of(providerPoints,
                       [](const auto &point) { return !point; }))
        return mlir::failure();
      llvm::sort(providerPoints, [](const auto &lhs, const auto &rhs) {
        return lhs->getIdentity().stableOrdinal <
               rhs->getIdentity().stableOrdinal;
      });
      for (std::unique_ptr<CoordinatedCommunicationActionPoint> &point :
           providerPoints) {
        const CoordinatedCommunicationActionPointIdentity &identity =
            point->getIdentity();
        if (identity.providerKey != providerKey ||
            !seenPointIdentities
                 .insert({identity.providerKey, identity.stableOrdinal})
                 .second)
          return mlir::failure();
        communicationPoints.push_back(point.get());
        plan.communicationPointOwners.push_back(std::move(point));
      }
    }
  }

  auto recipes = enumerateCoordinatedScheduleRecipes(
      canonicalInstrParents, optimizations, baselineOnly, family,
      plan.registry.has_value(), communicationPoints);
  if (mlir::failed(recipes))
    return mlir::failure();
  plan.recipes = std::move(*recipes);
  if (statistics)
    statistics->enumeratedRecipes = plan.recipes.size();

  wafer::support::ScopedCompileTimingSpan estimateTiming(
      "search", "coordinated-schedule-action", "online-estimate-beam");
  llvm::SmallVector<mlir::Operation *, 16> roots;
  roots.reserve(canonicalInstrParents.size());
  for (mlir::ModuleOp parent : canonicalInstrParents)
    roots.push_back(parent.getOperation());
  analysis::WholeCardInstructionProgramCost canonicalCost =
      analysis::analyzeWholeCardInstructionProgramCost(
          roots, analysis::getTargetScheduleCostPolicy());
  std::vector<CoordinatedScheduleActionEstimate> estimates;
  estimates.reserve(plan.recipes.size());
  for (const CoordinatedScheduleRecipe &recipe : plan.recipes)
    estimates.push_back(deriveScheduleRecipeEstimate(recipe, canonicalCost));
  auto selected = selectCoordinatedScheduleActionEstimateBeam(estimates);
  if (mlir::failed(selected) || selected->empty())
    return mlir::failure();
  if (statistics)
    statistics->coverageRetainedRecipes = selected->size();

  for (uint32_t ordinal : *selected) {
    plan.initiallySelected.insert(ordinal);
    plan.materializationOrder.push_back(ordinal);
  }
  llvm::SmallVector<size_t, 52> modelOrder;
  for (size_t index = 0; index < estimates.size(); ++index)
    modelOrder.push_back(index);
  llvm::sort(modelOrder, [&](size_t left, size_t right) {
    return isBetterScheduleActionEstimate(estimates[left], estimates[right]);
  });
  for (size_t index : modelOrder)
    if (!plan.initiallySelected.contains(estimates[index].stableOrdinal))
      plan.materializationOrder.push_back(estimates[index].stableOrdinal);
  if (plan.materializationOrder.empty() ||
      plan.materializationOrder.front() != 0)
    return mlir::failure();
  return plan;
}

} // namespace

mlir::LogicalResult walkCoordinatedScheduleActions(
    llvm::ArrayRef<mlir::ModuleOp> canonicalInstrParents,
    const OptimizationConfig &optimizations,
    llvm::function_ref<mlir::FailureOr<CoordinatedScheduleActionConsumption>(
        CoordinatedScheduleAction &&)>
        consume,
    uint64_t *materializationWork, bool baselineOnly,
    CoordinatedScheduleActionFamily family,
    CoordinatedScheduleRecipeStatistics *statistics,
    const frontend::FrontendProgramVerificationResult *program,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders) {
  if (materializationWork)
    *materializationWork = 0;
  if (statistics)
    *statistics = {};
  if (canonicalInstrParents.empty())
    return mlir::failure();

  const uint64_t rankCount = canonicalInstrParents.size();
  uint64_t work = 0;
  auto finish = [&](mlir::LogicalResult result) {
    if (materializationWork)
      *materializationWork = work;
    return result;
  };
  mlir::FailureOr<CoordinatedScheduleRecipePlan> plan =
      buildCoordinatedScheduleRecipePlan(
          canonicalInstrParents, optimizations, baselineOnly, family, program,
          communicationProviders, statistics);
  if (mlir::failed(plan))
    return finish(mlir::failure());
  const size_t targetAcceptedActions = plan->initiallySelected.size();
  size_t acceptedActions = 0;
  size_t attempts = 0;
  for (uint32_t ordinal : plan->materializationOrder) {
    if (acceptedActions >= targetAcceptedActions ||
        attempts >= kMaximumCoordinatedScheduleRecipeMaterializationAttempts)
      break;
    if (ordinal >= plan->recipes.size() ||
        !chargeActionMaterialization(rankCount, work))
      return finish(mlir::failure());
    ++attempts;
    if (statistics) {
      ++statistics->materializationAttempts;
      statistics->actualRankClones += rankCount;
      if (!plan->initiallySelected.contains(ordinal))
        ++statistics->materializationBackfills;
    }
    auto action = materializeCoordinatedScheduleRecipe(
        canonicalInstrParents, plan->recipes[ordinal],
        plan->registry ? &*plan->registry : nullptr, program);
    if (mlir::failed(action)) {
      if (statistics)
        ++statistics->materializationFailures;
      if (ordinal == 0)
        return finish(mlir::failure());
      continue;
    }
    mlir::FailureOr<CoordinatedScheduleActionConsumption> consumption =
        consume(std::move(*action));
    if (mlir::failed(consumption))
      return finish(mlir::failure());
    if (*consumption ==
        CoordinatedScheduleActionConsumption::RecoverableRejection)
      continue;
    ++acceptedActions;
    if (statistics)
      ++statistics->successfulActionClones;
  }
  if (acceptedActions == 0 ||
      acceptedActions > kMaximumCoordinatedExactScheduleActions ||
      attempts > kMaximumCoordinatedScheduleRecipeMaterializationAttempts)
    return finish(mlir::failure());
  return finish(mlir::success());
}

struct CoordinatedExecutableFinalizationCursor::Impl {
  int64_t stableSemanticOrdinal = 0;
  bool reservedBaseline = false;
  bool implementationAlternativeOrigin = false;
  int64_t rankCount = 0;
  unsigned requestedRankWorkers = 1;
  const frontend::FrontendProgramVerificationResult *program = nullptr;
  const ExecutionConfig *executionConfig = nullptr;
  CoordinatedWorkLedger *ledger = nullptr;
  llvm::raw_ostream *diagnostics = nullptr;
  WholeVariantSelectionStatistics *statistics = nullptr;
  WholeVariantSelectionMode selectionMode = WholeVariantSelectionMode::Production;
  ExecutableFinalizationReservation seedReservation;
  bool seedReservationClosed = false;
  ExecutableFinalizationWork setupWork;
  bool didSeedAttempt = false;
  bool seedExactAccepted = false;
  mlir::MLIRContext *rankContext = nullptr;
  std::vector<int64_t> logicalRanks;
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> canonicalOwners;
  llvm::SmallVector<mlir::ModuleOp, 16> canonicalParents;
  std::vector<std::shared_ptr<const std::string>> selectedTileIR;
  CoordinatedScheduleRecipePlan recipePlan;
};

CoordinatedExecutableFinalizationCursor::
    CoordinatedExecutableFinalizationCursor(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}
CoordinatedExecutableFinalizationCursor::~CoordinatedExecutableFinalizationCursor() {
  if (impl && !impl->seedReservationClosed && !impl->reservedBaseline)
    (void)impl->ledger->releaseExecutableFinalization(impl->seedReservation);
}
CoordinatedExecutableFinalizationCursor::
    CoordinatedExecutableFinalizationCursor(
        CoordinatedExecutableFinalizationCursor &&) noexcept = default;
CoordinatedExecutableFinalizationCursor &
CoordinatedExecutableFinalizationCursor::operator=(
    CoordinatedExecutableFinalizationCursor &&) noexcept = default;

int64_t CoordinatedExecutableFinalizationCursor::getStableSemanticOrdinal()
    const {
  return impl ? impl->stableSemanticOrdinal : -1;
}
bool CoordinatedExecutableFinalizationCursor::isReservedBaseline() const {
  return impl && impl->reservedBaseline;
}
size_t CoordinatedExecutableFinalizationCursor::getCanonicalParentCount() const {
  return impl ? impl->canonicalParents.size() : 0;
}
bool CoordinatedExecutableFinalizationCursor::seedAttempted() const {
  return impl && impl->didSeedAttempt;
}
bool CoordinatedExecutableFinalizationCursor::exactSeedAccepted() const {
  return impl && impl->seedExactAccepted;
}
bool CoordinatedExecutableFinalizationCursor::exhausted() const {
  return !impl ||
         (impl->didSeedAttempt && !impl->seedExactAccepted) ||
         impl->recipePlan.nextRecipe >=
             impl->recipePlan.materializationOrder.size();
}

mlir::FailureOr<std::unique_ptr<CoordinatedExecutableFinalizationCursor>>
beginCoordinatedExecutableFinalization(
    const CoordinatedTileVariant &tileVariant,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    const OptimizationConfig &optimizations, CoordinatedWorkLedger &ledger,
    llvm::raw_ostream &diagnostics,
    CoordinatedExecutableAdmissionFailure &failure,
    WholeVariantSelectionStatistics *statistics,
    WholeVariantSelectionMode selectionMode, unsigned rankPipelineParallelism,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders) {
  std::string variantTimingDetail =
      (llvm::Twine("semantic-ordinal=") +
       llvm::Twine(tileVariant.stableSemanticOrdinal))
          .str();
  wafer::support::ScopedCompileTimingSpan variantTiming(
      "finalization", "coordinated-executable-finalization", "tile-seed",
      variantTimingDetail);
  failure = {};
  const int64_t rankCount = executionConfig.getRankCount();
  if (ledger.getRankCount() != rankCount ||
      mlir::failed(verifyCoordinatedTileVariant(tileVariant, rankCount))) {
    failure.kind = CoordinatedExecutableAdmissionFailureKind::RankDomain;
    failure.gate = "rank-domain";
    if (!tileVariant.reservedBaseline &&
        mlir::failed(ledger.releaseExecutableFinalization(
            tileVariant.finalizationReservation))) {
      failure.kind = CoordinatedExecutableAdmissionFailureKind::WorkLedger;
      failure.gate = "work-ledger";
    }
    return mlir::failure();
  }

  auto state = std::make_unique<CoordinatedExecutableFinalizationCursor::Impl>();
  state->stableSemanticOrdinal = tileVariant.stableSemanticOrdinal;
  state->reservedBaseline = tileVariant.reservedBaseline;
  state->implementationAlternativeOrigin =
      tileVariant.implementationAlternativeOrigin;
  state->rankCount = rankCount;
  state->requestedRankWorkers =
      rankPipelineParallelism == 0 ? kMaximumBoundedRankPipelineWorkers
                                   : rankPipelineParallelism;
  state->program = &program;
  state->executionConfig = &executionConfig;
  state->ledger = &ledger;
  state->diagnostics = &diagnostics;
  state->statistics = statistics;
  state->selectionMode = selectionMode;
  state->seedReservation = tileVariant.finalizationReservation;
  state->rankContext = tileVariant.ranks.front().module.get().getContext();
  state->logicalRanks.reserve(tileVariant.ranks.size());
  if (llvm::any_of(tileVariant.ranks, [&](const auto &rank) {
        return rank.module.get().getContext() != state->rankContext;
      })) {
    failure.kind = CoordinatedExecutableAdmissionFailureKind::RankDomain;
    failure.gate = "rank-context-domain";
    if (!tileVariant.reservedBaseline &&
        mlir::failed(ledger.releaseExecutableFinalization(
            tileVariant.finalizationReservation))) {
      failure.kind = CoordinatedExecutableAdmissionFailureKind::WorkLedger;
      failure.gate = "work-ledger";
    }
    return mlir::failure();
  }

  const TargetMemoryPolicy memoryPolicy;
  const int64_t spmCapacityBytes = memoryPolicy.spmLimit - memoryPolicy.spmBase;
  struct CanonicalRankResult {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::string loweringFailure;
    std::optional<OversizedSPMAllocation> oversizedBeforeLowering;
    bool loweredTile = false;
    bool conversionFailed = false;
    bool verificationFailed = false;
  };
  std::vector<CanonicalRankResult> canonicalResults(tileVariant.ranks.size());
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "lowering", "coordinated-executable-finalization",
        "canonical-tile-to-instr");
    const unsigned workers = runBoundedRankPipelines(
        state->rankContext, tileVariant.ranks.size(),
        [&](size_t rankIndex) {
          CanonicalRankResult &result = canonicalResults[rankIndex];
          result.oversizedBeforeLowering = findOversizedSPMAllocation(
              tileVariant.ranks[rankIndex].module.get(), spmCapacityBytes);
          if (result.oversizedBeforeLowering)
            return;
          result.module = mlir::cast<mlir::ModuleOp>(
              tileVariant.ranks[rankIndex].module.get()->clone());
          result.loweredTile =
              containsTileDataflowOperations(result.module->getOperation());
          if (mlir::failed(convertTileRegionToInstrModule(
                  *result.module, &result.loweringFailure))) {
            result.conversionFailed = true;
            return;
          }
          clearRankCandidatePhysicalFacts(*result.module);
          result.verificationFailed =
              containsTileDataflowOperations(result.module->getOperation()) ||
              mlir::failed(mlir::verify(*result.module));
        },
        state->requestedRankWorkers);
    if (statistics)
      statistics->maximumRankPipelineWorkers =
          std::max<uint64_t>(statistics->maximumRankPipelineWorkers, workers);
  }
  for (const CanonicalRankResult &result : canonicalResults) {
    state->setupWork.tileLowerings += result.loweredTile;
    state->setupWork.spmProblems += result.oversizedBeforeLowering.has_value();
  }

  auto closeUnavailableSeed = [&]() {
    CoordinatedWorkEstimate actual = getActualExecutableFinalizationWork(
        state->setupWork, static_cast<uint64_t>(rankCount));
    mlir::LogicalResult result = mlir::failure();
    if (actual.getTotal() && *actual.getTotal() != 0)
      result =
          ledger.completeExecutableFinalization(state->seedReservation, actual);
    else if (!tileVariant.reservedBaseline)
      result = ledger.releaseExecutableFinalization(state->seedReservation);
    if (mlir::succeeded(result))
      state->seedReservationClosed = true;
    return result;
  };
  for (auto [rankIndex, result] : llvm::enumerate(canonicalResults)) {
    state->logicalRanks.push_back(tileVariant.ranks[rankIndex].logicalRank);
    if (result.oversizedBeforeLowering) {
      failure.kind = CoordinatedExecutableAdmissionFailureKind::SPMAllocation;
      failure.logicalRank = tileVariant.ranks[rankIndex].logicalRank;
      failure.gate = "individual-spm-capacity";
      diagnostics << "target_spm_bound: exact allocation of "
                  << result.oversizedBeforeLowering->physicalBytes
                  << " bytes exceeds target SPM window of " << spmCapacityBytes
                  << " bytes; type=" << result.oversizedBeforeLowering->type
                  << '\n';
      if (mlir::failed(closeUnavailableSeed())) {
        failure.kind = CoordinatedExecutableAdmissionFailureKind::WorkLedger;
        failure.gate = "work-ledger";
      }
      return mlir::failure();
    }
    if (result.conversionFailed || result.verificationFailed) {
      failure.kind =
          CoordinatedExecutableAdmissionFailureKind::RankFinalization;
      failure.logicalRank = tileVariant.ranks[rankIndex].logicalRank;
      failure.gate =
          result.conversionFailed ? "tile-to-instr" : "canonical-instr-parent";
      if (!result.loweringFailure.empty())
        diagnostics << result.loweringFailure << '\n';
      if (mlir::failed(closeUnavailableSeed())) {
        failure.kind = CoordinatedExecutableAdmissionFailureKind::WorkLedger;
        failure.gate = "work-ledger";
      }
      return mlir::failure();
    }
    std::optional<OversizedSPMAllocation> oversized =
        findOversizedSPMAllocation(*result.module, spmCapacityBytes);
    if (oversized) {
      failure.kind = CoordinatedExecutableAdmissionFailureKind::SPMAllocation;
      failure.logicalRank = tileVariant.ranks[rankIndex].logicalRank;
      failure.gate = "individual-spm-capacity";
      diagnostics << "target_spm_bound: exact allocation of "
                  << oversized->physicalBytes
                  << " bytes exceeds target SPM window of " << spmCapacityBytes
                  << " bytes; type=" << oversized->type << '\n';
      if (mlir::failed(closeUnavailableSeed())) {
        failure.kind = CoordinatedExecutableAdmissionFailureKind::WorkLedger;
        failure.gate = "work-ledger";
      }
      return mlir::failure();
    }
    state->canonicalOwners.push_back(std::move(result.module));
    state->canonicalParents.push_back(*state->canonicalOwners.back());
  }

  state->selectedTileIR.reserve(tileVariant.ranks.size());
  {
    wafer::support::ScopedCompileTimingSpan timing(
        "serialization", "coordinated-executable-finalization",
        "selected-tile-evidence");
    for (const CoordinatedRankTileProgram &rank : tileVariant.ranks)
      state->selectedTileIR.push_back(
          rank.selectedTileIR ? rank.selectedTileIR
                              : captureSelectedTileIR(*rank.module));
  }

  CoordinatedScheduleRecipeStatistics recipeStatistics;
  const CoordinatedScheduleActionFamily actionFamily =
      selectionMode == WholeVariantSelectionMode::
                           SelectSerializedDirectDTEComputeBaseline
          ? CoordinatedScheduleActionFamily::
                ProductionWithSerializedDirectDTECompute
          : CoordinatedScheduleActionFamily::Production;
  mlir::FailureOr<CoordinatedScheduleRecipePlan> plan =
      buildCoordinatedScheduleRecipePlan(
          state->canonicalParents, optimizations,
          selectionMode == WholeVariantSelectionMode::ReservedBaseline,
          actionFamily, &program, communicationProviders, &recipeStatistics);
  if (mlir::failed(plan)) {
    failure.kind = CoordinatedExecutableAdmissionFailureKind::RankFinalization;
    failure.gate = "schedule-recipe-plan";
    if (mlir::failed(closeUnavailableSeed())) {
      failure.kind = CoordinatedExecutableAdmissionFailureKind::WorkLedger;
      failure.gate = "work-ledger";
    }
    return mlir::failure();
  }
  state->recipePlan = std::move(*plan);
  if (statistics) {
    statistics->scheduleEstimatedActions += recipeStatistics.enumeratedRecipes;
    statistics->scheduleCoverageRetainedActions +=
        recipeStatistics.coverageRetainedRecipes;
  }
  return std::unique_ptr<CoordinatedExecutableFinalizationCursor>(
      new CoordinatedExecutableFinalizationCursor(std::move(state)));
}

mlir::FailureOr<CoordinatedExecutableFinalizationStep>
advanceCoordinatedExecutableFinalization(
    CoordinatedExecutableFinalizationCursor &cursor,
    ExecutableFinalizationReservation reservation) {
  if (!cursor.impl)
    return mlir::failure();
  CoordinatedExecutableFinalizationCursor::Impl &state = *cursor.impl;
  CoordinatedExecutableFinalizationStep step;
  if (state.recipePlan.nextRecipe >=
      state.recipePlan.materializationOrder.size()) {
    step.kind = CoordinatedExecutableFinalizationStepKind::Exhausted;
    return step;
  }
  if (!state.didSeedAttempt && reservation.id != state.seedReservation.id)
    return mlir::failure();
  if (state.didSeedAttempt && reservation.id == state.seedReservation.id)
    return mlir::failure();

  const bool seed = !state.didSeedAttempt;
  const uint32_t ordinal =
      state.recipePlan.materializationOrder[state.recipePlan.nextRecipe++];
  if ((seed && ordinal != 0) || ordinal >= state.recipePlan.recipes.size())
    return mlir::failure();
  state.didSeedAttempt = true;

  ExecutableFinalizationWork work;
  if (seed)
    work = state.setupWork;
  if (!chargeActionMaterialization(static_cast<uint64_t>(state.rankCount),
                                   work.actionMaterializations))
    return mlir::failure();
  if (state.statistics) {
    ++state.statistics->scheduleMaterializationAttempts;
    state.statistics->scheduleActualRankClones += state.rankCount;
    state.statistics->peakLiveScheduleActionClones =
        std::max<uint64_t>(state.statistics->peakLiveScheduleActionClones, 1);
    if (seed)
      ++state.statistics->scheduleSeedAttempts;
    else
      ++state.statistics->scheduleExpansionAttempts;
    if (!state.recipePlan.initiallySelected.contains(ordinal))
      ++state.statistics->scheduleMaterializationBackfills;
  }
  auto closeAttempt = [&]() {
    CoordinatedWorkEstimate actual = getActualExecutableFinalizationWork(
        work, static_cast<uint64_t>(state.rankCount));
    mlir::LogicalResult result =
        state.ledger->completeExecutableFinalization(reservation, actual);
    if (seed && mlir::succeeded(result))
      state.seedReservationClosed = true;
    return result;
  };
  auto reject = [&](CoordinatedExecutableAdmissionFailure failure)
      -> mlir::FailureOr<CoordinatedExecutableFinalizationStep> {
    if (mlir::failed(closeAttempt()))
      return mlir::failure();
    if (seed)
      state.seedExactAccepted = false;
    step.kind =
        CoordinatedExecutableFinalizationStepKind::RecoverableRejected;
    step.failure = std::move(failure);
    return std::move(step);
  };

  std::string actionFailureReason;
  auto action = materializeCoordinatedScheduleRecipe(
      state.canonicalParents, state.recipePlan.recipes[ordinal],
      state.recipePlan.registry ? &*state.recipePlan.registry : nullptr,
      state.program, &actionFailureReason);
  if (mlir::failed(action)) {
    if (state.statistics)
      ++state.statistics->scheduleMaterializationFailures;
    CoordinatedExecutableAdmissionFailure failure;
    failure.kind = CoordinatedExecutableAdmissionFailureKind::RankFinalization;
    failure.gate = "schedule-action-materialization";
    *state.diagnostics
        << "wafer-compile: schedule action unavailable"
        << " ordinal=" << ordinal
        << " ready_order="
        << static_cast<unsigned>(
               state.recipePlan.recipes[ordinal].actionIdentity.readyOrderKind)
        << " buffering="
        << static_cast<unsigned>(
               state.recipePlan.recipes[ordinal].actionIdentity.bufferingKind)
        << " worker_placement="
        << static_cast<unsigned>(state.recipePlan.recipes[ordinal]
                                     .actionIdentity.workerPlacementKind);
    if (!actionFailureReason.empty())
      *state.diagnostics << " reason=" << actionFailureReason;
    *state.diagnostics << '\n';
    return reject(std::move(failure));
  }

  std::string timingDetail =
      (llvm::Twine("action-ordinal=") + llvm::Twine(action->stableOrdinal))
          .str();
  wafer::support::ScopedCompileTimingSpan timing(
      "finalization", "coordinated-schedule-action", "exact-gates",
      timingDetail);
  const bool reservedBaseline =
      state.reservedBaseline && action->stableOrdinal == 0;
  const CoordinatedScheduleActionIdentity actionIdentity{
      action->readyOrderKind,
      action->bufferingKind,
      action->bufferingPlanOrdinal,
      action->workerPlacementKind,
      action->workerPlacementPlanOrdinal,
      action->serializationKind,
      action->communicationPointIdentity};
  struct FinalizedRankResult {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    RankFinalizationFailure failure;
  };
  std::vector<FinalizedRankResult> rankResults(action->rankModules.size());
  std::string capturedDiagnostics;
  unsigned workers = 1;
  {
    // DiagnosticEngineImpl::emit serializes registered handler callbacks under
    // its mutex. runBoundedRankPipelines joins every worker before this string
    // is read, so this shared capture has neither a writer race nor a lifetime
    // race.
    mlir::ScopedDiagnosticHandler handler(
        state.rankContext, [&](mlir::Diagnostic &diagnostic) {
          llvm::raw_string_ostream stream(capturedDiagnostics);
          diagnostic.print(stream);
          stream << '\n';
          return mlir::success();
        });
    workers = runBoundedRankPipelines(
        state.rankContext, action->rankModules.size(),
        [&](size_t rankIndex) {
          mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> finalized =
              finalizeCoordinatedRankModule(
                  std::move(action->rankModules[rankIndex]),
                  &rankResults[rankIndex].failure);
          if (mlir::succeeded(finalized))
            rankResults[rankIndex].module = std::move(*finalized);
        },
        state.requestedRankWorkers);
  }
  work.spmProblems += action->rankModules.size();
  if (state.statistics)
    state.statistics->maximumRankPipelineWorkers = std::max<uint64_t>(
        state.statistics->maximumRankPipelineWorkers, workers);

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> finalizedRanks;
  finalizedRanks.reserve(action->rankModules.size());
  for (auto [rankIndex, result] : llvm::enumerate(rankResults)) {
    if (!result.module) {
      appendUniqueRankDiagnostics(*state.diagnostics, capturedDiagnostics);
      CoordinatedExecutableAdmissionFailure failure;
      failure.kind =
          result.failure.kind == RankFinalizationFailureKind::SPMAllocation
              ? CoordinatedExecutableAdmissionFailureKind::SPMAllocation
              : CoordinatedExecutableAdmissionFailureKind::RankFinalization;
      failure.logicalRank = state.logicalRanks[rankIndex];
      failure.gate = "rank-finalization";
      return reject(std::move(failure));
    }
    finalizedRanks.push_back(std::move(result.module));
  }

  ++work.exactActionAttempts;
  std::string exactGate;
  mlir::FailureOr<AcceptedWholeVariant> accepted = admitCoordinatedExecutable(
      std::move(finalizedRanks), state.selectedTileIR, actionIdentity,
      *state.program, *state.executionConfig, *state.diagnostics, &exactGate,
      state.statistics, state.requestedRankWorkers);
  if (mlir::failed(accepted)) {
    CoordinatedExecutableAdmissionFailure failure;
    failure.kind =
        CoordinatedExecutableAdmissionFailureKind::WholeVariantExactGate;
    failure.gate = std::move(exactGate);
    return reject(std::move(failure));
  }

  if (mlir::failed(closeAttempt()))
    return mlir::failure();
  if (state.statistics) {
    ++state.statistics->scheduleSuccessfulActionClones;
    ++state.statistics->scheduleExactActions;
  }
  if (seed)
    state.seedExactAccepted = true;
  step.kind = CoordinatedExecutableFinalizationStepKind::Accepted;
  step.admitted.emplace(
      state.stableSemanticOrdinal, reservedBaseline, std::move(*accepted),
      action->stableOrdinal, actionIdentity,
      state.implementationAlternativeOrigin);
  return step;
}

mlir::FailureOr<std::vector<AdmittedCoordinatedExecutable>>
finalizeCoordinatedTileVariant(
    const CoordinatedTileVariant &tileVariant,
    const frontend::FrontendProgramVerificationResult &program,
    const ExecutionConfig &executionConfig,
    const OptimizationConfig &optimizations, CoordinatedWorkLedger &ledger,
    llvm::raw_ostream &diagnostics,
    CoordinatedExecutableAdmissionFailure &failure,
    WholeVariantSelectionStatistics *statistics,
    WholeVariantSelectionMode selectionMode,
    const analysis::WholeCardInstructionProgramCost *productionBaselineCost,
    unsigned rankPipelineParallelism,
    llvm::ArrayRef<const CoordinatedCommunicationActionProvider *>
        communicationProviders) {
  auto cursor = beginCoordinatedExecutableFinalization(
      tileVariant, program, executionConfig, optimizations, ledger, diagnostics,
      failure, statistics, selectionMode, rankPipelineParallelism,
      communicationProviders);
  if (mlir::failed(cursor))
    return mlir::failure();

  std::vector<AdmittedCoordinatedExecutable> admitted;
  uint32_t exactAccepted = 0;
  ExecutableFinalizationReservation reservation =
      tileVariant.finalizationReservation;
  while (!(*cursor)->exhausted() &&
         exactAccepted < kMaximumCoordinatedExactScheduleActions) {
    const bool seedAttempt = !(*cursor)->seedAttempted();
    auto step = advanceCoordinatedExecutableFinalization(
        **cursor, reservation);
    if (mlir::failed(step))
      return mlir::failure();
    if (step->kind ==
        CoordinatedExecutableFinalizationStepKind::RecoverableRejected) {
      failure = step->failure;
      if (tileVariant.reservedBaseline && seedAttempt) {
        diagnostics << "wafer-compile: mandatory executable seed rejected"
                    << " semantic_ordinal="
                    << tileVariant.stableSemanticOrdinal
                    << " logical_rank=" << failure.logicalRank
                    << " gate=" << failure.gate << '\n';
        return mlir::failure();
      }
    } else if (step->kind ==
               CoordinatedExecutableFinalizationStepKind::Accepted) {
      ++exactAccepted;
      admitted.push_back(std::move(*step->admitted));
      if (mlir::failed(reduceAdmittedExecutableFrontier(
              admitted, selectionMode, productionBaselineCost)))
        return mlir::failure();
    }
    if ((*cursor)->exhausted() ||
        exactAccepted >= kMaximumCoordinatedExactScheduleActions)
      break;
    std::optional<ExecutableFinalizationReservation> next =
        ledger.tryReserveExecutableScheduleAttempt();
    if (!next)
      break;
    reservation = *next;
  }
  if (admitted.empty()) {
    failure.kind =
        CoordinatedExecutableAdmissionFailureKind::WholeVariantExactGate;
    failure.gate = "no-admitted-executable";
    return mlir::failure();
  }
  failure = {};
  return admitted;
}

} // namespace wafer::compiler::detail
