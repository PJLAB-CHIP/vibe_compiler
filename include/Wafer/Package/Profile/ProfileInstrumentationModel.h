//===- ProfileInstrumentationModel.h - Profiler instrumentation model -----===//
//
// The typed profile instrumentation contract shared by the compiler-side
// instrumentation writer and the runtime-side strict reader: site kinds,
// static cost model, canonical spelling and JSON emission. The runtime
// collection/loading machinery lives in
// Wafer/Runtime/Profile/ProfileInstrumentation.h and consumes this owner
// through the neutral package support library.
//
//===----------------------------------------------------------------------===//

#ifndef WAFER_PACKAGE_PROFILEINSTRUMENTATIONMODEL_H
#define WAFER_PACKAGE_PROFILEINSTRUMENTATIONMODEL_H

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Package/Manifest/PackageManifest.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm::json {
class OStream;
} // namespace llvm::json

namespace wafer {

struct TargetCallDescriptor;

namespace runtime {

inline constexpr llvm::StringLiteral kProfileInstrumentationActivationFileName =
    "activation.json";
inline constexpr llvm::StringLiteral kProfileInstrumentationPlanFileName =
    "plan.json";
inline constexpr llvm::StringLiteral kProfileInstrumentationSiteMapFileName =
    "site-map.json";
inline constexpr int64_t kProfileInstrumentationCardCount = 1;
inline constexpr int64_t kProfileInstrumentationTileCount = 16;
inline constexpr llvm::StringLiteral kProfileSiteCorrelationBasis =
    "target-call-ordinal-ssa-position";
inline constexpr llvm::StringLiteral kProfileSiteKeyContract =
    "tile-local-site-id-and-correlation-key";
inline constexpr llvm::StringLiteral kProfileRecordABI =
    WAFER_TX81_PROFILER_RECORD_ABI;
inline constexpr llvm::StringLiteral kProfileStaticCostModelName =
    "tx81-static-throughput-lower-bound";
inline constexpr llvm::StringLiteral kProfileStaticCostModelScope =
    "complete-final-instruction-program-per-physical-tile";

enum class ProfileCaptureKind {
  Count,
  Trace,
};

enum class ProfileTSMEngine {
  CT,
  NE,
  RDMA,
  WDMA,
  TDMA,
  DirectDTE,
};

enum class ProfileTargetSiteKind {
  NCCCommand,
  NCCCompletion,
  DirectDTEControl,
  DirectDTEIssue,
  DirectDTEWait,
};

/// One exact, final-IR-derived static work dimension. Known uint64 values use a
/// decimal string on the JSON wire so the complete unsigned range survives
/// round-trip through every consumer. Non-known dimensions carry null.
struct ProfileStaticCostMetric {
  std::string knowledge;
  std::optional<uint64_t> value;
  std::string reason;
};

struct ProfileStaticCostRates {
  uint64_t cardDDRBytesPerSecond = 0;
  uint64_t directionalNoCBytesPerSecond = 0;
  uint64_t f16Bf16NPULogicalOpsPerSecondPerTile = 0;
  uint64_t f16Bf16VectorLogicalOpsPerSecondPerTile = 0;
  uint64_t f32VectorLogicalOpsPerSecondPerTile = 0;
  /// The current target has no calibrated SPM rate.
  std::optional<uint64_t> spmMovementBytesPerSecond;
};

/// Reporting reference rates written into a profile package. They are not
/// legality limits or search estimates.
ProfileStaticCostRates getTargetProfileStaticCostRates();

struct ProfileStaticDirectionalNoCWork {
  ProfileStaticCostMetric north;
  ProfileStaticCostMetric east;
  ProfileStaticCostMetric south;
  ProfileStaticCostMetric west;
};

struct ProfileStaticTileWork {
  ProfileStaticCostMetric npuF16Bf16LogicalOps;
  ProfileStaticCostMetric npuOtherLogicalOps;
  ProfileStaticCostMetric vectorF16Bf16LogicalOps;
  ProfileStaticCostMetric vectorF32LogicalOps;
  ProfileStaticCostMetric vectorOtherLogicalOps;
  ProfileStaticCostMetric ddrReadBytes;
  ProfileStaticCostMetric ddrWriteBytes;
  ProfileStaticCostMetric spmMovementBytes;
  ProfileStaticCostMetric nocTransmitBytes;
  ProfileStaticCostMetric nocReceiveBytes;
  ProfileStaticDirectionalNoCWork directionalNoCTransmitBytes;
};

struct ProfileStaticTileCost {
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlot;
  ProfileStaticTileWork work;
};

/// Static target rates plus exact work derived from the accepted final
/// instruction program. It is evidence for lower-bound comparison, not an
/// issue-latency, overlap, contention, or wall-time prediction.
struct ProfileStaticCostModel {
  std::string model;
  std::string scope;
  ProfileStaticCostRates rates;
  std::vector<ProfileStaticTileCost> tiles;
};

struct ProfileTargetCallSite {
  uint64_t siteId = 0;
  uint64_t targetCallOrdinal = 0;
  std::string targetCallSymbol;
  ProfileTargetSiteKind siteKind = ProfileTargetSiteKind::NCCCommand;
  /// Present only for NCCCommand (one of the five NCC engines) and Direct DTE
  /// issue/wait observation sites (DirectDTE). Completion/control sites have
  /// no engine.
  std::optional<ProfileTSMEngine> engine;
  std::string correlationKey;
  std::optional<uint64_t> functionOrdinal;
  std::optional<uint64_t> blockOrdinal;
  std::optional<uint64_t> instructionOrdinal;
};

struct ProfileTileSiteMap {
  CardId cardId{0};
  TileId tileId{0};
  LaunchSlotId launchSlot;
  std::vector<ProfileTargetCallSite> sites;
};

llvm::StringRef stringifyProfileCaptureKind(ProfileCaptureKind capture);
llvm::StringRef stringifyProfileTSMEngine(ProfileTSMEngine engine);
llvm::StringRef stringifyProfileTargetSiteKind(ProfileTargetSiteKind kind);

/// Emits the canonical strict JSON representation used in the profile plan
/// and top-level profile evidence.
void writeProfileStaticCostModel(llvm::json::OStream &json,
                                 const ProfileStaticCostModel &model);

/// Returns the profiler site semantic owned by one closed target-call
/// descriptor. Every descriptor in the public target-call registry maps to
/// exactly one site kind; this function never recovers semantics from symbols.
ProfileTargetSiteKind
getProfileTargetSiteKind(const TargetCallDescriptor &descriptor);

} // namespace runtime
} // namespace wafer

#endif // WAFER_PACKAGE_PROFILEINSTRUMENTATIONMODEL_H
