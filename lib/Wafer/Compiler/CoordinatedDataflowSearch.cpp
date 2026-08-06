//===- CoordinatedDataflowSearch.cpp - All-rank Tile frontier ------------===//

#include "CoordinatedDataflowSearch.h"

#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Scheduling/RankCandidateFrontier.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace wafer::compiler::detail {
namespace {

static std::optional<uint64_t> checkedAdd(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left)
    return std::nullopt;
  return left + right;
}

static std::optional<uint64_t> checkedMultiply(uint64_t left,
                                               uint64_t right) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
    return std::nullopt;
  return left * right;
}

static bool isWithinEstimate(const CoordinatedWorkEstimate &actual,
                             const CoordinatedWorkEstimate &upperBound) {
  for (size_t index = 0; index < kCoordinatedWorkKindCount; ++index)
    if (actual.counts[index] > upperBound.counts[index])
      return false;
  return true;
}

static bool containsInstructionSemantics(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *operation) {
    if (!mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(operation))
      return mlir::WalkResult::advance();
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

struct StructuredTraversalProposal {
  int64_t stableSemanticOrdinal = 0;
  CompleteRankTraversalComposition composition =
      CompleteRankTraversalComposition::Coupled;
  CandidateTileTraversalKind traversalKind =
      CandidateTileTraversalKind::ResultDriven;
  CandidateTileResidencyAction residencyAction =
      CandidateTileResidencyAction::KeepSingleRegion;
  CandidateBoundaryMovementAction boundaryMovementAction =
      CandidateBoundaryMovementAction::Staged;
  CandidateLoopMovementAction loopMovementAction =
      CandidateLoopMovementAction::AsConstructed;
  llvm::SmallVector<int64_t, 4> tileSizes;
};

static std::optional<llvm::SmallVector<int64_t, 4>>
getCommonStaticResultShape(mlir::ModuleOp module) {
  mlir::func::FuncOp function;
  for (mlir::func::FuncOp candidate : module.getOps<mlir::func::FuncOp>()) {
    if (candidate.isExternal())
      continue;
    if (function)
      return std::nullopt;
    function = candidate;
  }
  if (!function || function.getNumResults() == 0)
    return std::nullopt;

  std::optional<llvm::SmallVector<int64_t, 4>> commonShape;
  for (mlir::Type type : function.getResultTypes()) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!ranked || !ranked.hasStaticShape() || ranked.getRank() == 0 ||
        llvm::any_of(ranked.getShape(),
                     [](int64_t extent) { return extent <= 0; }))
      return std::nullopt;
    llvm::SmallVector<int64_t, 4> shape(ranked.getShape().begin(),
                                        ranked.getShape().end());
    if (!commonShape) {
      commonShape = std::move(shape);
      continue;
    }
    if (commonShape->size() != shape.size())
      return std::nullopt;
    for (size_t index = 0; index < shape.size(); ++index)
      (*commonShape)[index] = std::min((*commonShape)[index], shape[index]);
  }
  return commonShape;
}

static std::vector<StructuredTraversalProposal>
buildStructuredTraversalProposals(mlir::ModuleOp sourceModule,
                                  const OptimizationConfig &optimizations) {
  if (optimizations == OptimizationConfig::none())
    return {};
  std::optional<llvm::SmallVector<int64_t, 4>> shape =
      getCommonStaticResultShape(sourceModule);
  if (!shape)
    return {};

  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 3> tileVectors;
  auto appendUnique = [&](llvm::SmallVector<int64_t, 4> tiles) {
    if (!llvm::is_contained(tileVectors, tiles))
      tileVectors.push_back(std::move(tiles));
  };
  appendUnique(llvm::SmallVector<int64_t, 4>(shape->size(), 1));

  llvm::SmallVector<int64_t, 4> full(shape->begin(), shape->end());
  if (optimizations.isEnabled(OptimizationKind::TileSearchAlternatives)) {
    const WaferTargetPolicy policy =
        getDefaultWaferTargetPolicy(TileSearchEffort::Default);
    for (int64_t preferred : policy.tileSearch.preferredTileSizes) {
      llvm::SmallVector<int64_t, 4> tiles;
      tiles.reserve(shape->size());
      for (int64_t extent : *shape)
        tiles.push_back(std::min(extent, preferred));
      if (tiles != full &&
          tiles != llvm::SmallVector<int64_t, 4>(shape->size(), 1)) {
        appendUnique(std::move(tiles));
        break;
      }
    }
  }
  appendUnique(full);

  std::vector<StructuredTraversalProposal> proposals;
  int64_t nextOrdinal = 1;
  if (optimizations.isEnabled(OptimizationKind::ScopeComposition)) {
    for (const llvm::SmallVector<int64_t, 4> &tiles : tileVectors)
      proposals.push_back({nextOrdinal++,
                           CompleteRankTraversalComposition::Coupled,
                           CandidateTileTraversalKind::ResultDriven,
                           CandidateTileResidencyAction::KeepSingleRegion,
                           CandidateBoundaryMovementAction::Staged,
                           CandidateLoopMovementAction::AsConstructed, tiles});
  }
  if (optimizations.isEnabled(OptimizationKind::ScopeComposition) &&
      optimizations.isEnabled(OptimizationKind::DirectMappedBoundaryTransfer)) {
    for (const llvm::SmallVector<int64_t, 4> &tiles : tileVectors)
      proposals.push_back({nextOrdinal++,
                           CompleteRankTraversalComposition::Coupled,
                           CandidateTileTraversalKind::ResultDriven,
                           CandidateTileResidencyAction::KeepSingleRegion,
                           CandidateBoundaryMovementAction::ExactDirectMapped,
                           CandidateLoopMovementAction::AsConstructed, tiles});
  }
  if (optimizations.isEnabled(OptimizationKind::ScopeComposition) &&
      optimizations.isEnabled(OptimizationKind::LoopInvariantCodeMotion)) {
    const llvm::SmallVector<int64_t, 4> &tiles =
        tileVectors.size() > 1 ? tileVectors[1] : tileVectors.front();
    if (tiles != full) {
      proposals.push_back(
          {nextOrdinal++, CompleteRankTraversalComposition::Coupled,
           CandidateTileTraversalKind::ResultDriven,
           CandidateTileResidencyAction::KeepSingleRegion,
           CandidateBoundaryMovementAction::Staged,
           CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary, tiles});
      if (optimizations.isEnabled(
              OptimizationKind::DirectMappedBoundaryTransfer))
        proposals.push_back(
            {nextOrdinal++, CompleteRankTraversalComposition::Coupled,
             CandidateTileTraversalKind::ResultDriven,
             CandidateTileResidencyAction::KeepSingleRegion,
             CandidateBoundaryMovementAction::ExactDirectMapped,
             CandidateLoopMovementAction::HoistInvariantReadOnlyBoundary,
             tiles});
    }
  }
  if (optimizations.isEnabled(OptimizationKind::TileSearchAlternatives)) {
    for (const llvm::SmallVector<int64_t, 4> &tiles :
         llvm::drop_begin(tileVectors))
      proposals.push_back({nextOrdinal++,
                           CompleteRankTraversalComposition::Separated,
                           CandidateTileTraversalKind::ResultDriven,
                           CandidateTileResidencyAction::KeepSingleRegion,
                           CandidateBoundaryMovementAction::Staged,
                           CandidateLoopMovementAction::AsConstructed, tiles});
  }
  // These are bounded complete-state actions, not another independent axis:
  // one large separated traversal gets a real DDR-clean region cut, and one
  // large coupled traversal gets one selective root spill. Failed actions are
  // simply absent from the actual frontier.
  if (optimizations.isEnabled(OptimizationKind::ScopeComposition) &&
      optimizations.isEnabled(OptimizationKind::TileSearchAlternatives))
    proposals.push_back(
        {nextOrdinal++, CompleteRankTraversalComposition::Separated,
         CandidateTileTraversalKind::ResultDriven,
         CandidateTileResidencyAction::SplitAtExplicitDDRBoundary,
         CandidateBoundaryMovementAction::Staged,
         CandidateLoopMovementAction::AsConstructed, full});
  if (optimizations.isEnabled(OptimizationKind::ScopeComposition) &&
      optimizations.isEnabled(OptimizationKind::ConcurrentWorkingSetSelection))
    proposals.push_back({nextOrdinal++,
                         CompleteRankTraversalComposition::Coupled,
                         CandidateTileTraversalKind::ResultDriven,
                         CandidateTileResidencyAction::SelectiveSpill,
                         CandidateBoundaryMovementAction::Staged,
                         CandidateLoopMovementAction::AsConstructed, full});
  return proposals;
}

static std::shared_ptr<const std::string>
captureSelectedTileIR(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return std::make_shared<const std::string>(std::move(text));
}

static std::string computeCoordinatedTileVariantContentDigestImpl(
    const CoordinatedTileVariant &variant) {
  llvm::SHA256 hasher;
  for (const CoordinatedRankTileProgram &rank : variant.ranks) {
    std::string moduleText;
    llvm::raw_string_ostream stream(moduleText);
    stream << "rank:" << rank.logicalRank << '\n';
    rank.module.get().print(stream);
    stream.flush();
    hasher.update(moduleText);
  }
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

} // namespace

std::string computeCoordinatedTileVariantContentDigest(
    const CoordinatedTileVariant &variant) {
  return computeCoordinatedTileVariantContentDigestImpl(variant);
}

std::optional<uint64_t> CoordinatedWorkEstimate::getTotal() const {
  uint64_t total = 0;
  for (uint64_t count : counts) {
    std::optional<uint64_t> next = checkedAdd(total, count);
    if (!next)
      return std::nullopt;
    total = *next;
  }
  return total;
}

mlir::FailureOr<CoordinatedWorkEstimate>
CoordinatedWorkLedger::getMandatoryGenerationUpperBound(int64_t rankCount) {
  if (rankCount <= 0)
    return mlir::failure();
  CoordinatedWorkEstimate estimate;
  uint64_t ranks = static_cast<uint64_t>(rankCount);
  estimate.set(CoordinatedWorkKind::StructuredExpansion, ranks);
  estimate.set(CoordinatedWorkKind::ActualTileClone, ranks);
  return estimate;
}

mlir::FailureOr<CoordinatedWorkEstimate>
CoordinatedWorkLedger::getTerminalActionUpperBound(int64_t rankCount) {
  if (rankCount <= 0)
    return mlir::failure();
  uint64_t ranks = static_cast<uint64_t>(rankCount);
  uint64_t actions = kMaximumCoordinatedTerminalActions;
  std::optional<uint64_t> rankActions = checkedMultiply(ranks, actions);
  if (!rankActions)
    return mlir::failure();
  CoordinatedWorkEstimate estimate;
  estimate.set(CoordinatedWorkKind::TileToInstrLowering, ranks);
  estimate.set(CoordinatedWorkKind::TerminalScheduleAction, *rankActions);
  estimate.set(CoordinatedWorkKind::SPMAllocationProblem, *rankActions);
  estimate.set(CoordinatedWorkKind::DDRAllocationDomain, *rankActions);
  estimate.set(CoordinatedWorkKind::TransportValidation, actions);
  estimate.set(CoordinatedWorkKind::ABIValidation, *rankActions);
  return estimate;
}

CoordinatedWorkLedger::CoordinatedWorkLedger(
    int64_t rankCount, uint64_t capacity, uint64_t repairReserve,
    CoordinatedWorkEstimate generationUpperBound,
    CoordinatedWorkEstimate terminalUpperBound)
    : rankCount(rankCount), capacity(capacity),
      mandatoryGenerationReserved(*generationUpperBound.getTotal()),
      repairReserved(repairReserve),
      mandatoryGenerationUpperBound(std::move(generationUpperBound)) {
  mandatoryBaselineReservation.id = nextReservationId++;
  terminalReservations.push_back({mandatoryBaselineReservation.id,
                                  /*mandatoryBaseline=*/true,
                                  std::move(terminalUpperBound)});
}

mlir::FailureOr<CoordinatedWorkLedger>
CoordinatedWorkLedger::create(int64_t rankCount, uint64_t capacity,
                              uint64_t repairReserve) {
  mlir::FailureOr<CoordinatedWorkEstimate> generation =
      getMandatoryGenerationUpperBound(rankCount);
  mlir::FailureOr<CoordinatedWorkEstimate> terminal =
      getTerminalActionUpperBound(rankCount);
  if (mlir::failed(generation) || mlir::failed(terminal))
    return mlir::failure();
  std::optional<uint64_t> generationTotal = generation->getTotal();
  std::optional<uint64_t> terminalTotal = terminal->getTotal();
  if (!generationTotal || !terminalTotal)
    return mlir::failure();
  std::optional<uint64_t> mandatory =
      checkedAdd(*generationTotal, *terminalTotal);
  if (!mandatory)
    return mlir::failure();
  mandatory = checkedAdd(*mandatory, repairReserve);
  if (!mandatory || *mandatory > capacity)
    return mlir::failure();
  return CoordinatedWorkLedger(rankCount, capacity, repairReserve,
                               std::move(*generation), std::move(*terminal));
}

CoordinatedWorkLedger::TerminalReservationRecord *
CoordinatedWorkLedger::findReservation(uint64_t id) {
  auto found = llvm::find_if(
      terminalReservations,
      [&](const TerminalReservationRecord &record) { return record.id == id; });
  return found == terminalReservations.end() ? nullptr : &*found;
}

const CoordinatedWorkLedger::TerminalReservationRecord *
CoordinatedWorkLedger::findReservation(uint64_t id) const {
  auto found = llvm::find_if(
      terminalReservations,
      [&](const TerminalReservationRecord &record) { return record.id == id; });
  return found == terminalReservations.end() ? nullptr : &*found;
}

bool CoordinatedWorkLedger::canReserve(uint64_t credits) const {
  CoordinatedWorkLedgerSnapshot snapshot = getSnapshot();
  return credits <= snapshot.unreserved;
}

void CoordinatedWorkLedger::addConsumed(const CoordinatedWorkEstimate &actual) {
  consumed += *actual.getTotal();
  for (size_t index = 0; index < kCoordinatedWorkKindCount; ++index)
    consumedByKind[index] += actual.counts[index];
}

mlir::LogicalResult CoordinatedWorkLedger::completeMandatoryBaselineGeneration(
    const CoordinatedWorkEstimate &actual) {
  if (mandatoryGenerationReserved == 0 ||
      !isWithinEstimate(actual, mandatoryGenerationUpperBound) ||
      !actual.getTotal())
    return mlir::failure();
  addConsumed(actual);
  mandatoryGenerationReserved = 0;
  mandatoryGenerationUpperBound = {};
  return mlir::success();
}

bool CoordinatedWorkLedger::tryConsumeGeneration(CoordinatedWorkKind kind,
                                                 uint64_t credits) {
  if (kind != CoordinatedWorkKind::StructuredExpansion &&
      kind != CoordinatedWorkKind::ActualTileClone)
    return false;
  CoordinatedWorkEstimate actual;
  actual.set(kind, credits);
  return tryConsumeGeneration(actual);
}

bool CoordinatedWorkLedger::tryConsumeGeneration(
    const CoordinatedWorkEstimate &actual) {
  std::optional<uint64_t> credits = actual.getTotal();
  if (!credits || *credits == 0 ||
      actual.get(CoordinatedWorkKind::RepairExpansion) != 0 ||
      actual.get(CoordinatedWorkKind::TileToInstrLowering) != 0 ||
      actual.get(CoordinatedWorkKind::TerminalScheduleAction) != 0 ||
      actual.get(CoordinatedWorkKind::SPMAllocationProblem) != 0 ||
      actual.get(CoordinatedWorkKind::DDRAllocationDomain) != 0 ||
      actual.get(CoordinatedWorkKind::TransportValidation) != 0 ||
      actual.get(CoordinatedWorkKind::ABIValidation) != 0 ||
      !canReserve(*credits))
    return false;
  addConsumed(actual);
  return true;
}

std::optional<TerminalActionReservation>
CoordinatedWorkLedger::tryReserveTerminalAction(
    const CoordinatedWorkEstimate &upperBound) {
  std::optional<uint64_t> credits = upperBound.getTotal();
  if (!credits || *credits == 0 || !canReserve(*credits))
    return std::nullopt;
  TerminalActionReservation reservation{nextReservationId++};
  terminalReservations.push_back(
      {reservation.id, /*mandatoryBaseline=*/false, upperBound});
  return reservation;
}

mlir::LogicalResult CoordinatedWorkLedger::completeTerminalAction(
    TerminalActionReservation reservation,
    const CoordinatedWorkEstimate &actual) {
  TerminalReservationRecord *record = findReservation(reservation.id);
  if (!record || !isWithinEstimate(actual, record->upperBound) ||
      !actual.getTotal())
    return mlir::failure();
  addConsumed(actual);
  llvm::erase_if(terminalReservations,
                 [&](const TerminalReservationRecord &candidate) {
                   return candidate.id == reservation.id;
                 });
  return mlir::success();
}

mlir::LogicalResult CoordinatedWorkLedger::releaseTerminalAction(
    TerminalActionReservation reservation) {
  TerminalReservationRecord *record = findReservation(reservation.id);
  if (!record || record->mandatoryBaseline)
    return mlir::failure();
  llvm::erase_if(terminalReservations,
                 [&](const TerminalReservationRecord &candidate) {
                   return candidate.id == reservation.id;
                 });
  return mlir::success();
}

bool CoordinatedWorkLedger::tryConsumeRepair(uint64_t credits) {
  CoordinatedWorkEstimate actual;
  actual.set(CoordinatedWorkKind::RepairExpansion, credits);
  return tryConsumeRepair(actual);
}

bool CoordinatedWorkLedger::tryConsumeRepair(
    const CoordinatedWorkEstimate &actual) {
  std::optional<uint64_t> credits = actual.getTotal();
  if (!credits || *credits == 0 ||
      actual.get(CoordinatedWorkKind::RepairExpansion) == 0 ||
      actual.get(CoordinatedWorkKind::StructuredExpansion) != 0 ||
      actual.get(CoordinatedWorkKind::TileToInstrLowering) != 0 ||
      actual.get(CoordinatedWorkKind::TerminalScheduleAction) != 0 ||
      actual.get(CoordinatedWorkKind::SPMAllocationProblem) != 0 ||
      actual.get(CoordinatedWorkKind::DDRAllocationDomain) != 0 ||
      actual.get(CoordinatedWorkKind::TransportValidation) != 0 ||
      actual.get(CoordinatedWorkKind::ABIValidation) != 0 ||
      *credits > repairReserved)
    return false;
  repairReserved -= *credits;
  addConsumed(actual);
  return true;
}

CoordinatedWorkLedgerSnapshot CoordinatedWorkLedger::getSnapshot() const {
  CoordinatedWorkLedgerSnapshot snapshot;
  snapshot.capacity = capacity;
  snapshot.consumed = consumed;
  snapshot.mandatoryGenerationReserved = mandatoryGenerationReserved;
  snapshot.repairReserved = repairReserved;
  snapshot.consumedByKind = consumedByKind;
  for (const TerminalReservationRecord &record : terminalReservations)
    snapshot.terminalReserved += *record.upperBound.getTotal();
  uint64_t committed = consumed + mandatoryGenerationReserved +
                       snapshot.terminalReserved + repairReserved;
  snapshot.unreserved = committed <= capacity ? capacity - committed : 0;
  return snapshot;
}

mlir::LogicalResult
verifyCoordinatedTileVariant(const CoordinatedTileVariant &variant,
                             int64_t expectedRankCount) {
  if (expectedRankCount <= 0 ||
      variant.ranks.size() != static_cast<size_t>(expectedRankCount) ||
      variant.terminalReservation.id == 0 ||
      variant.repairDepth > kMaximumCoordinatedRepairDepth ||
      (variant.reservedBaseline && variant.repairDepth != 0))
    return mlir::failure();
  for (auto [expectedRank, rank] : llvm::enumerate(variant.ranks)) {
    if (rank.logicalRank != static_cast<int64_t>(expectedRank) || !rank.module)
      return mlir::failure();
    if (containsInstructionSemantics(*rank.module)) {
      rank.module.get().emitError()
          << "coordinated Tile variant contains instruction semantics at "
             "logical rank "
          << expectedRank;
      return mlir::failure();
    }
    bool hasPhysicalFacts = false;
    rank.module.get().walk([&](mlir::Operation *operation) {
      if (auto allocation = mlir::dyn_cast<mlir::memref::AllocOp>(operation))
        hasPhysicalFacts |= allocation->hasAttr(kWaferSPMOffsetAttrName) ||
                            allocation->hasAttr(kWaferDDROffsetAttrName);
      if (auto send = mlir::dyn_cast<InstrDTESendOp>(operation))
        hasPhysicalFacts |= send.getBinding().has_value();
      if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(operation))
        hasPhysicalFacts |= recv.getBinding().has_value();
    });
    if (hasPhysicalFacts) {
      rank.module.get().emitError()
          << "coordinated Tile variant contains terminal physical facts at "
             "logical rank "
          << expectedRank;
      return mlir::failure();
    }
    if (mlir::failed(mlir::verify(*rank.module)))
      return mlir::failure();
    if (containsTileDataflowOperations(rank.module.get().getOperation()) &&
        !rank.selectedTileIR) {
      rank.module.get().emitError()
          << "coordinated Tile variant is missing actual-clone debug evidence "
             "at logical rank "
          << expectedRank;
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::FailureOr<CoordinatedTileVariant> materializeCoordinatedTileRepair(
    const CoordinatedTileVariant &parent, CoordinatedTileRepairAction action,
    int64_t stableSemanticOrdinal, CoordinatedWorkLedger &ledger,
    std::string *failureReason) {
  auto fail = [&](llvm::StringRef reason)
      -> mlir::FailureOr<CoordinatedTileVariant> {
    if (failureReason)
      *failureReason = reason.str();
    return mlir::failure();
  };
  if (parent.reservedBaseline ||
      parent.repairDepth >= kMaximumCoordinatedRepairDepth ||
      stableSemanticOrdinal <= parent.stableSemanticOrdinal ||
      ledger.getRankCount() <= 0 ||
      mlir::failed(
          verifyCoordinatedTileVariant(parent, ledger.getRankCount())))
    return fail("invalid coordinated Tile repair parent or ordinal");

  mlir::FailureOr<CoordinatedWorkEstimate> terminalUpperBound =
      CoordinatedWorkLedger::getTerminalActionUpperBound(
          ledger.getRankCount());
  if (mlir::failed(terminalUpperBound))
    return fail("cannot derive repair terminal-action upper bound");
  std::optional<TerminalActionReservation> reservation =
      ledger.tryReserveTerminalAction(*terminalUpperBound);
  if (!reservation)
    return fail("coordinated repair has no terminal reservation");

  CoordinatedWorkEstimate repairWork;
  repairWork.set(CoordinatedWorkKind::RepairExpansion, 1);
  repairWork.set(CoordinatedWorkKind::ActualTileClone, parent.ranks.size());
  if (!ledger.tryConsumeRepair(repairWork)) {
    (void)ledger.releaseTerminalAction(*reservation);
    return fail("coordinated repair reserve is exhausted");
  }

  CandidateTileResidencyAction residencyAction =
      action == CoordinatedTileRepairAction::SelectiveSpill
          ? CandidateTileResidencyAction::SelectiveSpill
          : CandidateTileResidencyAction::SplitAtExplicitDDRBoundary;
  CoordinatedTileVariant sibling;
  sibling.stableSemanticOrdinal = stableSemanticOrdinal;
  sibling.repairDepth = parent.repairDepth + 1;
  sibling.terminalReservation = *reservation;
  sibling.ranks.reserve(parent.ranks.size());
  for (const CoordinatedRankTileProgram &rank : parent.ranks) {
    std::string rankFailure;
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> rankSibling =
        materializeCompleteRankTileResidencySibling(
            *rank.module, residencyAction, &rankFailure);
    if (mlir::failed(rankSibling)) {
      (void)ledger.releaseTerminalAction(*reservation);
      if (failureReason)
        *failureReason =
            (llvm::Twine("logical rank ") + llvm::Twine(rank.logicalRank) +
             " repair failed" +
             (rankFailure.empty() ? llvm::Twine()
                                  : llvm::Twine(": ") + rankFailure))
                .str();
      return mlir::failure();
    }
    std::shared_ptr<const std::string> selectedTileIR =
        captureSelectedTileIR(**rankSibling);
    sibling.ranks.emplace_back(rank.logicalRank, std::move(*rankSibling),
                               std::move(selectedTileIR));
  }
  if (mlir::failed(
          verifyCoordinatedTileVariant(sibling, ledger.getRankCount()))) {
    (void)ledger.releaseTerminalAction(*reservation);
    return fail("coordinated repair produced invalid actual Tile IR");
  }
  if (failureReason)
    failureReason->clear();
  return sibling;
}

std::string
computeCoordinatedTileFrontierDigest(const CoordinatedTileFrontier &frontier) {
  llvm::SHA256 hasher;
  for (const CoordinatedTileVariant &variant : frontier) {
    std::string header =
        (llvm::Twine("variant:") + llvm::Twine(variant.stableSemanticOrdinal) +
         ":" + llvm::Twine(variant.reservedBaseline ? 1 : 0) + "\n")
            .str();
    hasher.update(header);
    for (const CoordinatedRankTileProgram &rank : variant.ranks) {
      std::string moduleText;
      llvm::raw_string_ostream stream(moduleText);
      stream << "rank:" << rank.logicalRank << '\n';
      rank.module.get().print(stream);
      stream.flush();
      hasher.update(moduleText);
    }
  }
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

mlir::FailureOr<CoordinatedTileFrontier>
buildCoordinatedTileFrontier(mlir::ModuleOp sourceModule,
                             const CoordinatedDataflowSearchConfig &config,
                             CoordinatedWorkLedger &ledger) {
  if (!sourceModule || config.rankCount <= 0 ||
      config.rankCount != ledger.getRankCount() ||
      config.candidateParallelism <= 0) {
    if (sourceModule)
      sourceModule.emitError("invalid coordinated dataflow search contract");
    return mlir::failure();
  }

  CoordinatedTileVariant baseline;
  baseline.stableSemanticOrdinal = 0;
  baseline.reservedBaseline = true;
  baseline.terminalReservation = ledger.getMandatoryBaselineReservation();
  baseline.ranks.reserve(static_cast<size_t>(config.rankCount));

  const bool rankInvariant =
      isTensorProgramSchedulingRankInvariant(sourceModule);
  CoordinatedWorkEstimate actualGeneration;
  actualGeneration.set(CoordinatedWorkKind::StructuredExpansion,
                       rankInvariant ? 1
                                     : static_cast<uint64_t>(config.rankCount));
  actualGeneration.set(CoordinatedWorkKind::ActualTileClone,
                       static_cast<uint64_t>(config.rankCount));

  auto generate = [&](int64_t logicalRank)
      -> mlir::FailureOr<wafer::ScheduledRankCandidate> {
    TensorProgramSchedulingConfig schedulingConfig;
    schedulingConfig.logicalRank = logicalRank;
    schedulingConfig.candidateParallelism = config.candidateParallelism;
    schedulingConfig.requestShardIndex = 0;
    schedulingConfig.requestShardCount = 1;
    schedulingConfig.optimizations = config.optimizations;
    mlir::FailureOr<std::vector<ScheduledRankCandidate>> generated =
        buildScheduledRankCandidateFrontier(sourceModule, schedulingConfig);
    if (mlir::failed(generated) || generated->size() != 1 ||
        !generated->front().reservedBaseline ||
        generated->front().stableOrdinal != 0 ||
        generated->front().artifactKind != RankArtifactKind::Spill ||
        generated->front().bufferingKind != RankBufferingKind::Single ||
        generated->front().workerPlacementKind !=
            RankWorkerPlacementKind::Unplaced) {
      sourceModule.emitError()
          << "coordinated Tile generation did not return exactly one "
             "conservative actual baseline for logical rank "
          << logicalRank;
      return mlir::failure();
    }
    return std::move(generated->front());
  };

  if (rankInvariant) {
    mlir::FailureOr<ScheduledRankCandidate> generated = generate(0);
    if (mlir::failed(generated))
      return mlir::failure();
    for (int64_t logicalRank = 0; logicalRank < config.rankCount;
         ++logicalRank) {
      mlir::OwningOpRef<mlir::ModuleOp> module =
          logicalRank == 0 ? std::move(generated->module)
                           : mlir::cast<mlir::ModuleOp>(
                                 baseline.ranks.front().module->clone());
      baseline.ranks.emplace_back(logicalRank, std::move(module),
                                  generated->selectedTileIR);
    }
  } else {
    for (int64_t logicalRank = 0; logicalRank < config.rankCount;
         ++logicalRank) {
      mlir::FailureOr<ScheduledRankCandidate> generated = generate(logicalRank);
      if (mlir::failed(generated))
        return mlir::failure();
      baseline.ranks.emplace_back(logicalRank, std::move(generated->module),
                                  std::move(generated->selectedTileIR));
    }
  }

  if (mlir::failed(verifyCoordinatedTileVariant(baseline, config.rankCount)) ||
      mlir::failed(
          ledger.completeMandatoryBaselineGeneration(actualGeneration))) {
    sourceModule.emitError(
        "coordinated Tile baseline failed rank-domain or work-ledger closure");
    return mlir::failure();
  }

  CoordinatedTileFrontier frontier;
  frontier.push_back(std::move(baseline));

  llvm::StringSet<> seenVariantDigests;
  seenVariantDigests.insert(
      computeCoordinatedTileVariantContentDigest(frontier.front()));
  mlir::FailureOr<CoordinatedWorkEstimate> terminalUpperBound =
      CoordinatedWorkLedger::getTerminalActionUpperBound(config.rankCount);
  if (mlir::failed(terminalUpperBound)) {
    sourceModule.emitError(
        "cannot derive coordinated terminal-action work upper bound");
    return mlir::failure();
  }

  for (const StructuredTraversalProposal &proposal :
       buildStructuredTraversalProposals(sourceModule, config.optimizations)) {
    std::optional<TerminalActionReservation> terminalReservation =
        ledger.tryReserveTerminalAction(*terminalUpperBound);
    if (!terminalReservation)
      break;

    CoordinatedWorkEstimate generationWork;
    generationWork.set(CoordinatedWorkKind::StructuredExpansion,
                       rankInvariant ? 1
                                     : static_cast<uint64_t>(config.rankCount));
    generationWork.set(CoordinatedWorkKind::ActualTileClone,
                       static_cast<uint64_t>(config.rankCount));
    if (!ledger.tryConsumeGeneration(generationWork)) {
      if (mlir::failed(ledger.releaseTerminalAction(*terminalReservation))) {
        sourceModule.emitError(
            "cannot release unmaterialized terminal-action reservation");
        return mlir::failure();
      }
      break;
    }

    CoordinatedTileVariant candidate;
    candidate.stableSemanticOrdinal = proposal.stableSemanticOrdinal;
    candidate.terminalReservation = *terminalReservation;
    candidate.ranks.reserve(static_cast<size_t>(config.rankCount));
    auto generateCandidate = [&](int64_t logicalRank)
        -> mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> {
      std::string failureReason;
      return materializeCompleteRankCandidateTileProgram(
          sourceModule, logicalRank, proposal.tileSizes,
          /*candidateReductionTileSizes=*/{}, proposal.traversalKind,
          proposal.composition, proposal.residencyAction,
          proposal.boundaryMovementAction, proposal.loopMovementAction,
          &failureReason);
    };

    bool materialized = true;
    if (rankInvariant) {
      mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> generated =
          generateCandidate(/*logicalRank=*/0);
      if (mlir::failed(generated)) {
        materialized = false;
      } else {
        std::shared_ptr<const std::string> selectedTileIR =
            captureSelectedTileIR(**generated);
        for (int64_t logicalRank = 0; logicalRank < config.rankCount;
             ++logicalRank) {
          mlir::OwningOpRef<mlir::ModuleOp> module =
              logicalRank == 0 ? std::move(*generated)
                               : mlir::cast<mlir::ModuleOp>(
                                     candidate.ranks.front().module->clone());
          candidate.ranks.emplace_back(logicalRank, std::move(module),
                                       selectedTileIR);
        }
      }
    } else {
      for (int64_t logicalRank = 0; logicalRank < config.rankCount;
           ++logicalRank) {
        mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> generated =
            generateCandidate(logicalRank);
        if (mlir::failed(generated)) {
          materialized = false;
          break;
        }
        std::shared_ptr<const std::string> selectedTileIR =
            captureSelectedTileIR(**generated);
        candidate.ranks.emplace_back(logicalRank, std::move(*generated),
                                     std::move(selectedTileIR));
      }
    }

    if (!materialized) {
      if (mlir::failed(ledger.releaseTerminalAction(*terminalReservation))) {
        sourceModule.emitError(
            "cannot release rejected terminal-action reservation");
        return mlir::failure();
      }
      continue;
    }
    if (mlir::failed(
            verifyCoordinatedTileVariant(candidate, config.rankCount))) {
      (void)ledger.releaseTerminalAction(*terminalReservation);
      sourceModule.emitError(
          "materialized coordinated Tile candidate failed verification");
      return mlir::failure();
    }

    std::string contentDigest =
        computeCoordinatedTileVariantContentDigest(candidate);
    if (!seenVariantDigests.insert(contentDigest).second) {
      if (mlir::failed(ledger.releaseTerminalAction(*terminalReservation))) {
        sourceModule.emitError(
            "cannot release duplicate terminal-action reservation");
        return mlir::failure();
      }
      continue;
    }
    frontier.push_back(std::move(candidate));
  }

  std::string frontierDigest = computeCoordinatedTileFrontierDigest(frontier);
  for (CoordinatedTileVariant &variant : frontier)
    variant.frontierDigest = frontierDigest;
  return frontier;
}

} // namespace wafer::compiler::detail
