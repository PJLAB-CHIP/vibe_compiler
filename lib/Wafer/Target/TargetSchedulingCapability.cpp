//===- TargetSchedulingCapability.cpp - Static scheduling contract -----===//

#include "Wafer/Target/TargetSchedulingCapability.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace wafer {
namespace {

static llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

static unsigned engineCount(TargetSchedulingEngineMask engines) {
  return llvm::popcount(engines);
}

static bool validEnvelope(const TargetSchedulingGeometryEnvelope &envelope) {
  return envelope.minimumPayloadBytes <= envelope.maximumPayloadBytes;
}

static bool envelopesOverlap(const TargetSchedulingGeometryEnvelope &lhs,
                             const TargetSchedulingGeometryEnvelope &rhs) {
  return lhs.minimumPayloadBytes <= rhs.maximumPayloadBytes &&
         rhs.minimumPayloadBytes <= lhs.maximumPayloadBytes;
}

static bool envelopeContains(const TargetSchedulingGeometryEnvelope &outer,
                             const TargetSchedulingGeometryEnvelope &inner) {
  return outer.minimumPayloadBytes <= inner.minimumPayloadBytes &&
         outer.maximumPayloadBytes >= inner.maximumPayloadBytes &&
         outer.maximumVisibleBuffers >= inner.maximumVisibleBuffers &&
         outer.maximumVisibleTokens >= inner.maximumVisibleTokens;
}

static bool
sameCategoricalPredicate(const TargetSchedulingWindowPredicate &lhs,
                         const TargetSchedulingWindowPredicate &rhs) {
  return lhs.mechanism == rhs.mechanism && lhs.engines == rhs.engines &&
         lhs.workerRelation == rhs.workerRelation &&
         lhs.completion == rhs.completion && lhs.irRelation == rhs.irRelation &&
         lhs.observer == rhs.observer;
}

static llvm::Error
validatePredicate(const TargetSchedulingWindowPredicate &predicate) {
  if (predicate.engines == 0 ||
      (predicate.engines & ~kAllTargetSchedulingEngineMask) != 0)
    return invalid("target scheduling predicate has an invalid engine set");
  if (!validEnvelope(predicate.geometry))
    return invalid("target scheduling predicate has an invalid payload "
                   "geometry interval");

  const bool hasDTE = (predicate.engines & kTargetSchedulingDTEEngineMask) != 0;
  const bool hasNCC = (predicate.engines & kTargetSchedulingNCCEngineMask) != 0;
  switch (predicate.workerRelation) {
  case TargetSchedulingWorkerRelation::SameNCCWorker:
    if (!hasNCC || hasDTE ||
        predicate.completion !=
            TargetSchedulingCompletionKind::SameWorkerIssueOrder)
      return invalid("same-worker scheduling predicate has mismatched "
                     "engine/completion fields");
    break;
  case TargetSchedulingWorkerRelation::CrossNCCWorkers:
    if (!hasNCC || hasDTE ||
        predicate.completion != TargetSchedulingCompletionKind::ParticipantJoin)
      return invalid("cross-worker scheduling predicate has mismatched "
                     "engine/completion fields");
    break;
  case TargetSchedulingWorkerRelation::ExactDTEEvent:
    if (!hasDTE || hasNCC ||
        predicate.completion != TargetSchedulingCompletionKind::ExactEvent)
      return invalid("DTE-event scheduling predicate has mismatched "
                     "engine/completion fields");
    break;
  case TargetSchedulingWorkerRelation::MixedNCCAndDTE:
    if (!hasDTE || !hasNCC ||
        predicate.completion !=
            TargetSchedulingCompletionKind::ParticipantJoinAndExactEvent)
      return invalid("mixed NCC/DTE scheduling predicate has mismatched "
                     "engine/completion fields");
    break;
  }
  return llvm::Error::success();
}

template <typename RowT>
static llvm::Error validateNoOverlappingPredicates(llvm::ArrayRef<RowT> rows,
                                                   llvm::StringRef kind) {
  for (auto [index, row] : llvm::enumerate(rows)) {
    if (llvm::Error error = validatePredicate(row.predicate))
      return error;
    for (size_t prior = 0; prior < index; ++prior) {
      if (!sameCategoricalPredicate(rows[prior].predicate, row.predicate) ||
          !envelopesOverlap(rows[prior].predicate.geometry,
                            row.predicate.geometry))
        continue;
      return invalid(
          (kind + " contains duplicate or overlapping predicates").str());
    }
  }
  return llvm::Error::success();
}

static bool matches(const TargetSchedulingWindowPredicate &predicate,
                    const TargetSchedulingWindowQuery &query) {
  if (!query.geometryKnown)
    return false;
  const TargetSchedulingGeometryEnvelope &geometry = predicate.geometry;
  return predicate.mechanism == query.mechanism &&
         predicate.engines == query.engines &&
         predicate.workerRelation == query.workerRelation &&
         predicate.completion == query.completion &&
         predicate.irRelation == query.irRelation &&
         predicate.observer == query.observer &&
         query.maximumPayloadBytes >= geometry.minimumPayloadBytes &&
         query.maximumPayloadBytes <= geometry.maximumPayloadBytes &&
         query.visibleBufferCount <= geometry.maximumVisibleBuffers &&
         query.visibleTokenCount <= geometry.maximumVisibleTokens;
}

static TargetSchedulingWindowPredicate
makePredicate(TargetSchedulingMechanism mechanism,
              TargetSchedulingEngineMask engines,
              TargetSchedulingWorkerRelation workerRelation,
              TargetSchedulingCompletionKind completion) {
  TargetSchedulingWindowPredicate predicate{mechanism, engines, workerRelation,
                                            completion};
  predicate.geometry.maximumVisibleBuffers = 4096;
  predicate.geometry.maximumVisibleTokens = 4096;
  return predicate;
}

static void
appendNCCLegalityRows(std::vector<TargetSchedulingLegalityRow> &rows,
                      TargetSchedulingMechanism mechanism,
                      bool allowCrossWorker) {
  for (TargetSchedulingEngineMask engines = 1;
       engines <= kTargetSchedulingNCCEngineMask; ++engines) {
    rows.push_back(
        {makePredicate(mechanism, engines,
                       TargetSchedulingWorkerRelation::SameNCCWorker,
                       TargetSchedulingCompletionKind::SameWorkerIssueOrder),
         TargetSchedulingCapabilityState::Supported});
    if (allowCrossWorker)
      rows.push_back(
          {makePredicate(mechanism, engines,
                         TargetSchedulingWorkerRelation::CrossNCCWorkers,
                         TargetSchedulingCompletionKind::ParticipantJoin),
           TargetSchedulingCapabilityState::Supported});
  }
}

static void
appendMixedDTEAndNCCLegalityRows(std::vector<TargetSchedulingLegalityRow> &rows,
                                 TargetSchedulingMechanism mechanism) {
  for (TargetSchedulingEngineMask nccEngines = 1;
       nccEngines <= kTargetSchedulingNCCEngineMask; ++nccEngines)
    rows.push_back(
        {makePredicate(
             mechanism, nccEngines | kTargetSchedulingDTEEngineMask,
             TargetSchedulingWorkerRelation::MixedNCCAndDTE,
             TargetSchedulingCompletionKind::ParticipantJoinAndExactEvent),
         TargetSchedulingCapabilityState::Supported});
}

static void appendFixedSlotProfitabilityRows(
    std::vector<TargetSchedulingProfitabilityRow> &rows) {
  for (TargetSchedulingEngineMask engines = 1;
       engines <= kTargetSchedulingNCCEngineMask; ++engines) {
    const unsigned count = engineCount(engines);
    if (count < 2)
      continue;
    rows.push_back(
        {makePredicate(TargetSchedulingMechanism::StaticFixedSlot, engines,
                       TargetSchedulingWorkerRelation::SameNCCWorker,
                       TargetSchedulingCompletionKind::SameWorkerIssueOrder),
         count == 2 ? TargetSchedulingProfitabilityScope::ExactPair
                    : TargetSchedulingProfitabilityScope::ExactGroup,
         count == 2 ? TargetSchedulingOverlapEvidence::QualifiedOverlap
                    : TargetSchedulingOverlapEvidence::Unknown,
         TargetSchedulingDrainEvidence::QualifiedDrainElision});
  }
}

static std::optional<TargetSchedulingEngine> convertEngine(InstrFamily family) {
  switch (family) {
  case InstrFamily::CT:
    return TargetSchedulingEngine::CT;
  case InstrFamily::NE:
    return TargetSchedulingEngine::NE;
  case InstrFamily::RDMA:
    return TargetSchedulingEngine::RDMA;
  case InstrFamily::WDMA:
    return TargetSchedulingEngine::WDMA;
  case InstrFamily::TDMA:
    return TargetSchedulingEngine::TDMA;
  case InstrFamily::DTE:
    return TargetSchedulingEngine::DTE;
  }
  return std::nullopt;
}

static void recordMemrefGeometry(mlir::Value value,
                                 llvm::DenseSet<mlir::Value> &visibleBuffers,
                                 TargetSchedulingWindowQuery &query) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !isWaferMemRefType(type))
    return;
  if (isWaferSPMMemRefType(type))
    visibleBuffers.insert(value);
  std::optional<WaferPhysicalTensorInfo> physical =
      computeWaferPhysicalTensorInfo(type);
  if (!physical || physical->physicalBytes < 0) {
    query.geometryKnown = false;
    return;
  }
  query.maximumPayloadBytes =
      std::max(query.maximumPayloadBytes,
               static_cast<uint64_t>(physical->physicalBytes));
}

} // namespace

llvm::Expected<TargetSchedulingCapabilityRegistry>
TargetSchedulingCapabilityRegistry::create(
    TargetProfileId targetProfile, uint32_t contractVersion,
    llvm::ArrayRef<TargetSchedulingLegalityRow> legalityRows,
    llvm::ArrayRef<TargetSchedulingProfitabilityRow> profitabilityRows) {
  if (contractVersion == 0)
    return invalid("target scheduling contract version must be nonzero");
  if (legalityRows.empty())
    return invalid("target scheduling legality registry must not be empty");
  if (llvm::Error error =
          validateNoOverlappingPredicates(legalityRows, "legality registry"))
    return std::move(error);
  if (llvm::Error error = validateNoOverlappingPredicates(
          profitabilityRows, "profitability registry"))
    return std::move(error);

  for (const TargetSchedulingProfitabilityRow &row : profitabilityRows) {
    const unsigned count = engineCount(row.predicate.engines);
    if ((row.scope == TargetSchedulingProfitabilityScope::ExactPair &&
         count != 2) ||
        (row.scope == TargetSchedulingProfitabilityScope::ExactGroup &&
         count < 3))
      return invalid("target scheduling profitability scope does not match "
                     "its exact engine set");
    if (row.overlap == TargetSchedulingOverlapEvidence::Unknown &&
        row.drain == TargetSchedulingDrainEvidence::Unknown)
      return invalid("target scheduling profitability row contains no "
                     "qualified evidence axis");

    const TargetSchedulingLegalityRow *coveringLegality = nullptr;
    for (const TargetSchedulingLegalityRow &legality : legalityRows) {
      if (!sameCategoricalPredicate(legality.predicate, row.predicate) ||
          !envelopeContains(legality.predicate.geometry,
                            row.predicate.geometry))
        continue;
      if (coveringLegality)
        return invalid("target scheduling profitability predicate has "
                       "ambiguous legality coverage");
      coveringLegality = &legality;
    }
    if (!coveringLegality ||
        coveringLegality->state != TargetSchedulingCapabilityState::Supported)
      return invalid("target scheduling profitability predicate lacks one "
                     "supported covering legality row");
  }

  return TargetSchedulingCapabilityRegistry(
      targetProfile, contractVersion,
      std::vector<TargetSchedulingLegalityRow>(legalityRows.begin(),
                                               legalityRows.end()),
      std::vector<TargetSchedulingProfitabilityRow>(profitabilityRows.begin(),
                                                    profitabilityRows.end()));
}

llvm::Expected<TargetSchedulingWindowDecision>
TargetSchedulingCapabilityRegistry::query(
    const TargetSchedulingWindowQuery &query) const {
  if (query.targetProfile != targetProfile)
    return invalid("target scheduling query/profile mismatch");
  if (query.engines == 0 ||
      (query.engines & ~kAllTargetSchedulingEngineMask) != 0)
    return invalid("target scheduling query has an invalid engine set");
  TargetSchedulingWindowPredicate queryPredicate{
      query.mechanism, query.engines, query.workerRelation, query.completion};
  queryPredicate.irRelation = query.irRelation;
  queryPredicate.observer = query.observer;
  if (llvm::Error error = validatePredicate(queryPredicate))
    return std::move(error);

  const TargetSchedulingLegalityRow *legality = nullptr;
  for (const TargetSchedulingLegalityRow &row : legalityRows) {
    if (!matches(row.predicate, query))
      continue;
    if (legality)
      return invalid("target scheduling legality query is ambiguous");
    legality = &row;
  }
  TargetSchedulingWindowDecision decision;
  if (!legality)
    return decision;
  decision.legality = legality->state;
  if (decision.legality != TargetSchedulingCapabilityState::Supported)
    return decision;

  const TargetSchedulingProfitabilityRow *profitability = nullptr;
  for (const TargetSchedulingProfitabilityRow &row : profitabilityRows) {
    if (!matches(row.predicate, query))
      continue;
    if (profitability)
      return invalid("target scheduling profitability query is ambiguous");
    profitability = &row;
  }
  if (profitability) {
    decision.profitability.overlap = profitability->overlap;
    decision.profitability.drain = profitability->drain;
  }
  return decision;
}

llvm::Expected<TargetSchedulingCapabilityRegistry>
getTargetSchedulingCapabilityRegistry(TargetProfileId targetProfile) {
  std::vector<TargetSchedulingLegalityRow> legality;
  std::vector<TargetSchedulingProfitabilityRow> profitability;

  const bool nonzeroWorkers =
      targetProfile == TargetProfileId::waferTx81SingleCardKernelV3();
  appendNCCLegalityRows(legality, TargetSchedulingMechanism::StaticFixedSlot,
                        nonzeroWorkers);
  if (nonzeroWorkers) {
    appendNCCLegalityRows(legality, TargetSchedulingMechanism::WorkerPlacement,
                          /*allowCrossWorker=*/true);
    appendMixedDTEAndNCCLegalityRows(
        legality, TargetSchedulingMechanism::StaticFixedSlot);
    appendMixedDTEAndNCCLegalityRows(
        legality, TargetSchedulingMechanism::DirectDTEOverlap);
  }
  appendFixedSlotProfitabilityRows(profitability);
  if (nonzeroWorkers) {
    const TargetSchedulingEngineMask ctRdma =
        targetSchedulingEngineBit(TargetSchedulingEngine::CT) |
        targetSchedulingEngineBit(TargetSchedulingEngine::RDMA);
    profitability.push_back(
        {makePredicate(TargetSchedulingMechanism::WorkerPlacement, ctRdma,
                       TargetSchedulingWorkerRelation::CrossNCCWorkers,
                       TargetSchedulingCompletionKind::ParticipantJoin),
         TargetSchedulingProfitabilityScope::ExactPair,
         TargetSchedulingOverlapEvidence::QualifiedOverlap,
         TargetSchedulingDrainEvidence::Unknown});
  }

  return TargetSchedulingCapabilityRegistry::create(
      targetProfile, nonzeroWorkers ? 3u : 1u, legality, profitability);
}

llvm::Expected<TargetSchedulingWindowQuery>
analyzeTargetSchedulingWindow(mlir::ModuleOp module,
                              TargetProfileId targetProfile,
                              TargetSchedulingMechanism mechanism) {
  if (!module)
    return invalid("target scheduling analysis requires a module");

  TargetSchedulingWindowQuery query(targetProfile, mechanism);
  llvm::DenseSet<mlir::Value> visibleBuffers;
  bool hasNCC = false;
  bool hasDTE = false;
  bool countOverflow = false;

  module.walk([&](mlir::Operation *operation) {
    auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(operation);
    if (instruction) {
      std::optional<TargetSchedulingEngine> engine =
          convertEngine(instruction.getInstructionFamily());
      if (!engine) {
        query.geometryKnown = false;
      } else {
        query.engines |= targetSchedulingEngineBit(*engine);
        hasDTE |= *engine == TargetSchedulingEngine::DTE;
        hasNCC |= *engine != TargetSchedulingEngine::DTE;
      }
      if (instruction.getInstructionFamily() == InstrFamily::DTE &&
          operation->getNumResults() != 0) {
        if (query.visibleTokenCount == std::numeric_limits<uint32_t>::max())
          countOverflow = true;
        else
          ++query.visibleTokenCount;
      }
    }

    NCCCompletionContract completion = getNCCCompletionContract(operation);
    if (completion.behavior == LocalInstructionCompletion::SynchronousWriteback)
      query.observer = TargetSchedulingObserverKind::SynchronousHostWriteback;

    for (mlir::Value operand : operation->getOperands())
      recordMemrefGeometry(operand, visibleBuffers, query);
    for (mlir::Value result : operation->getResults())
      recordMemrefGeometry(result, visibleBuffers, query);
  });

  if (countOverflow ||
      visibleBuffers.size() > std::numeric_limits<uint32_t>::max())
    return invalid("target scheduling analysis structural bound overflows");
  query.visibleBufferCount = static_cast<uint32_t>(visibleBuffers.size());
  if (query.engines == 0)
    return invalid("target scheduling analysis found no typed instruction "
                   "engine");
  NCCWorkerWindowSummary workerWindows = analyzeNCCWorkerWindows(module);

  if (hasDTE && hasNCC) {
    query.workerRelation = TargetSchedulingWorkerRelation::MixedNCCAndDTE;
    query.completion =
        TargetSchedulingCompletionKind::ParticipantJoinAndExactEvent;
  } else if (hasDTE) {
    query.workerRelation = TargetSchedulingWorkerRelation::ExactDTEEvent;
    query.completion = TargetSchedulingCompletionKind::ExactEvent;
  } else if (workerWindows.hasCrossWorkerWindow) {
    query.workerRelation = TargetSchedulingWorkerRelation::CrossNCCWorkers;
    query.completion = TargetSchedulingCompletionKind::ParticipantJoin;
  } else {
    query.workerRelation = TargetSchedulingWorkerRelation::SameNCCWorker;
    query.completion = TargetSchedulingCompletionKind::SameWorkerIssueOrder;
  }
  return query;
}

} // namespace wafer
