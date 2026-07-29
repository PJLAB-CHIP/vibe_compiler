//===- TargetSchedulingCapability.h - Static scheduling contract -*- C++
//-*-===//

#ifndef WAFER_TARGET_TARGETSCHEDULINGCAPABILITY_H
#define WAFER_TARGET_TARGETSCHEDULINGCAPABILITY_H

#include "Wafer/Target/TargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace wafer {

/// Stable target-engine identities used only by the compiler-shipped static
/// scheduling contract. They are derived from WaferInstructionOpInterface,
/// never from operation, function, fixture, or workload names.
enum class TargetSchedulingEngine : uint8_t {
  CT = 0,
  NE = 1,
  RDMA = 2,
  WDMA = 3,
  TDMA = 4,
  DTE = 5,
};

using TargetSchedulingEngineMask = uint32_t;

constexpr TargetSchedulingEngineMask
targetSchedulingEngineBit(TargetSchedulingEngine engine) {
  return TargetSchedulingEngineMask{1}
         << static_cast<TargetSchedulingEngineMask>(engine);
}

inline constexpr TargetSchedulingEngineMask kTargetSchedulingNCCEngineMask =
    targetSchedulingEngineBit(TargetSchedulingEngine::CT) |
    targetSchedulingEngineBit(TargetSchedulingEngine::NE) |
    targetSchedulingEngineBit(TargetSchedulingEngine::RDMA) |
    targetSchedulingEngineBit(TargetSchedulingEngine::WDMA) |
    targetSchedulingEngineBit(TargetSchedulingEngine::TDMA);
inline constexpr TargetSchedulingEngineMask kTargetSchedulingDTEEngineMask =
    targetSchedulingEngineBit(TargetSchedulingEngine::DTE);
inline constexpr TargetSchedulingEngineMask kAllTargetSchedulingEngineMask =
    kTargetSchedulingNCCEngineMask | kTargetSchedulingDTEEngineMask;

/// Compiler transformation family being admitted. The accepted IR remains the
/// sole schedule representation; this enum is an invocation-local query key.
enum class TargetSchedulingMechanism : uint8_t {
  StaticFixedSlot,
  WorkerPlacement,
  DirectDTEOverlap,
};

enum class TargetSchedulingWorkerRelation : uint8_t {
  SameNCCWorker,
  CrossNCCWorkers,
  ExactDTEEvent,
  MixedNCCAndDTE,
};

enum class TargetSchedulingCompletionKind : uint8_t {
  SameWorkerIssueOrder,
  ParticipantJoin,
  ExactEvent,
  ParticipantJoinAndExactEvent,
};

/// The registry never replaces dependency, lifetime, address, or completion
/// verification. A supported row means that those exact IR late gates are the
/// required proof mechanism for this target contract.
enum class TargetSchedulingIRRelation : uint8_t {
  RequiresExactLateGate,
};

enum class TargetSchedulingObserverKind : uint8_t {
  None,
  SynchronousHostWriteback,
};

enum class TargetSchedulingCapabilityState : uint8_t {
  Supported,
  Unsupported,
  Unknown,
};

enum class TargetSchedulingProfitabilityScope : uint8_t {
  ExactPair,
  ExactGroup,
};

enum class TargetSchedulingOverlapEvidence : uint8_t {
  Unknown,
  QualifiedOverlap,
};

enum class TargetSchedulingDrainEvidence : uint8_t {
  Unknown,
  QualifiedDrainElision,
};

/// Closed geometry envelope for a static predicate. The byte interval is
/// inclusive. Buffer/token bounds are finite compiler-visible structural
/// bounds; exact capacity and lifetime acceptance remain downstream gates.
struct TargetSchedulingGeometryEnvelope {
  uint64_t minimumPayloadBytes = 0;
  uint64_t maximumPayloadBytes = std::numeric_limits<uint64_t>::max();
  uint32_t maximumVisibleBuffers = std::numeric_limits<uint32_t>::max();
  uint32_t maximumVisibleTokens = std::numeric_limits<uint32_t>::max();
};

struct TargetSchedulingWindowPredicate {
  TargetSchedulingMechanism mechanism;
  TargetSchedulingEngineMask engines;
  TargetSchedulingWorkerRelation workerRelation;
  TargetSchedulingCompletionKind completion;
  TargetSchedulingIRRelation irRelation =
      TargetSchedulingIRRelation::RequiresExactLateGate;
  TargetSchedulingObserverKind observer = TargetSchedulingObserverKind::None;
  TargetSchedulingGeometryEnvelope geometry;
};

struct TargetSchedulingLegalityRow {
  TargetSchedulingWindowPredicate predicate;
  TargetSchedulingCapabilityState state;
};

struct TargetSchedulingProfitabilityRow {
  TargetSchedulingWindowPredicate predicate;
  TargetSchedulingProfitabilityScope scope;
  TargetSchedulingOverlapEvidence overlap =
      TargetSchedulingOverlapEvidence::Unknown;
  TargetSchedulingDrainEvidence drain = TargetSchedulingDrainEvidence::Unknown;
};

/// Recomputable query derived from one actual instruction module.
struct TargetSchedulingWindowQuery {
  TargetSchedulingWindowQuery(TargetProfileId targetProfile,
                              TargetSchedulingMechanism mechanism)
      : targetProfile(targetProfile), mechanism(mechanism) {}

  TargetProfileId targetProfile;
  TargetSchedulingMechanism mechanism;
  TargetSchedulingEngineMask engines = 0;
  TargetSchedulingWorkerRelation workerRelation =
      TargetSchedulingWorkerRelation::SameNCCWorker;
  TargetSchedulingCompletionKind completion =
      TargetSchedulingCompletionKind::SameWorkerIssueOrder;
  TargetSchedulingIRRelation irRelation =
      TargetSchedulingIRRelation::RequiresExactLateGate;
  TargetSchedulingObserverKind observer = TargetSchedulingObserverKind::None;
  uint64_t maximumPayloadBytes = 0;
  uint32_t visibleBufferCount = 0;
  uint32_t visibleTokenCount = 0;
  bool geometryKnown = true;
};

struct TargetSchedulingProfitabilityEvidence {
  TargetSchedulingOverlapEvidence overlap =
      TargetSchedulingOverlapEvidence::Unknown;
  TargetSchedulingDrainEvidence drain = TargetSchedulingDrainEvidence::Unknown;

  bool isEntirelyUnknown() const {
    return overlap == TargetSchedulingOverlapEvidence::Unknown &&
           drain == TargetSchedulingDrainEvidence::Unknown;
  }
};

struct TargetSchedulingWindowDecision {
  TargetSchedulingCapabilityState legality =
      TargetSchedulingCapabilityState::Unknown;
  TargetSchedulingProfitabilityEvidence profitability;
};

/// Immutable, versioned, compiler-owned scheduling registry. Construction
/// validates the complete row set and rejects malformed, duplicate, or
/// overlapping predicates before any query can be consumed.
class TargetSchedulingCapabilityRegistry {
public:
  TargetSchedulingCapabilityRegistry() = delete;

  static llvm::Expected<TargetSchedulingCapabilityRegistry>
  create(TargetProfileId targetProfile, uint32_t contractVersion,
         llvm::ArrayRef<TargetSchedulingLegalityRow> legalityRows,
         llvm::ArrayRef<TargetSchedulingProfitabilityRow> profitabilityRows);

  TargetProfileId getTargetProfile() const { return targetProfile; }
  uint32_t getContractVersion() const { return contractVersion; }

  llvm::ArrayRef<TargetSchedulingLegalityRow> getLegalityRows() const {
    return legalityRows;
  }
  llvm::ArrayRef<TargetSchedulingProfitabilityRow>
  getProfitabilityRows() const {
    return profitabilityRows;
  }

  llvm::Expected<TargetSchedulingWindowDecision>
  query(const TargetSchedulingWindowQuery &query) const;

private:
  TargetSchedulingCapabilityRegistry(
      TargetProfileId targetProfile, uint32_t contractVersion,
      std::vector<TargetSchedulingLegalityRow> legalityRows,
      std::vector<TargetSchedulingProfitabilityRow> profitabilityRows)
      : targetProfile(targetProfile), contractVersion(contractVersion),
        legalityRows(std::move(legalityRows)),
        profitabilityRows(std::move(profitabilityRows)) {}

  TargetProfileId targetProfile;
  uint32_t contractVersion;
  std::vector<TargetSchedulingLegalityRow> legalityRows;
  std::vector<TargetSchedulingProfitabilityRow> profitabilityRows;
};

/// Returns the validated compiler-shipped registry for the exact target
/// profile. No card state, runtime profile, PMU sample, or local cache is read.
llvm::Expected<TargetSchedulingCapabilityRegistry>
getTargetSchedulingCapabilityRegistry(TargetProfileId targetProfile);

/// Derives an exact categorical scheduling query from typed instruction IR.
/// Dynamic/unknown physical geometry is represented as `geometryKnown=false`
/// and therefore cannot match a supported static row.
llvm::Expected<TargetSchedulingWindowQuery>
analyzeTargetSchedulingWindow(mlir::ModuleOp module,
                              TargetProfileId targetProfile,
                              TargetSchedulingMechanism mechanism);

} // namespace wafer

#endif // WAFER_TARGET_TARGETSCHEDULINGCAPABILITY_H
