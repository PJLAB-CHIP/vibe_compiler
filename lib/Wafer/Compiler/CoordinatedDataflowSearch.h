//===- CoordinatedDataflowSearch.h - All-rank Tile frontier ----*- C++ -*-===//

#ifndef WAFER_COMPILER_COORDINATEDDATAFLOWSEARCH_H
#define WAFER_COMPILER_COORDINATEDDATAFLOWSEARCH_H

#include "Wafer/Support/OptimizationConfig.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

/// The terminal action family is finite: original/ready order, the canonical
/// bounded fixed-slot loop identities, and original/disjoint worker placement.
inline constexpr uint32_t kMaximumCoordinatedFixedSlotActions = 8;
inline constexpr uint32_t kMaximumCoordinatedTerminalActions =
    2 * (1 + kMaximumCoordinatedFixedSlotActions) * 2;

/// Stable semantic work classes charged to the one invocation-local search
/// ledger. They are compiler scheduling policy, not IR or artifact metadata.
enum class CoordinatedWorkKind : uint8_t {
  StructuredExpansion,
  ActualTileClone,
  TileToInstrLowering,
  TerminalScheduleAction,
  SPMAllocationProblem,
  DDRAllocationDomain,
  TransportValidation,
  ABIValidation,
  RepairExpansion,
  Count,
};

inline constexpr size_t kCoordinatedWorkKindCount =
    static_cast<size_t>(CoordinatedWorkKind::Count);

struct CoordinatedWorkEstimate {
  std::array<uint64_t, kCoordinatedWorkKindCount> counts = {};

  uint64_t get(CoordinatedWorkKind kind) const {
    return counts[static_cast<size_t>(kind)];
  }
  void set(CoordinatedWorkKind kind, uint64_t value) {
    counts[static_cast<size_t>(kind)] = value;
  }
  std::optional<uint64_t> getTotal() const;
};

struct TerminalActionReservation {
  uint64_t id = 0;
};

struct CoordinatedWorkLedgerSnapshot {
  uint64_t capacity = 0;
  uint64_t consumed = 0;
  uint64_t mandatoryGenerationReserved = 0;
  uint64_t terminalReserved = 0;
  uint64_t repairReserved = 0;
  uint64_t unreserved = 0;
  std::array<uint64_t, kCoordinatedWorkKindCount> consumedByKind = {};
};

/// One deterministic invocation-level work ledger shared by structured
/// generation, terminal evaluation, and repair. Mandatory baseline generation
/// and terminal evaluation plus a finite repair allowance are reserved in the
/// constructor, before any candidate is generated. Optional generation cannot
/// borrow those credits.
class CoordinatedWorkLedger {
public:
  static constexpr uint64_t kDefaultCapacity = 262144;
  static constexpr uint64_t kDefaultRepairReserve = 16384;

  static mlir::FailureOr<CoordinatedWorkLedger>
  create(int64_t rankCount, uint64_t capacity = kDefaultCapacity,
         uint64_t repairReserve = kDefaultRepairReserve);

  int64_t getRankCount() const { return rankCount; }
  TerminalActionReservation getMandatoryBaselineReservation() const {
    return mandatoryBaselineReservation;
  }

  /// Closes the pre-reserved baseline generation allowance with actual work.
  /// Every actual count must be no greater than the deterministic upper bound
  /// reserved at construction.
  mlir::LogicalResult
  completeMandatoryBaselineGeneration(const CoordinatedWorkEstimate &actual);

  /// Consumes unreserved optional generation credits. This never consumes a
  /// terminal or repair reservation.
  bool tryConsumeGeneration(const CoordinatedWorkEstimate &actual);
  bool tryConsumeGeneration(CoordinatedWorkKind kind, uint64_t credits = 1);

  /// Reserves one complete all-rank terminal action before admitting its
  /// actual Tile clone to the coordinated frontier.
  std::optional<TerminalActionReservation>
  tryReserveTerminalAction(const CoordinatedWorkEstimate &upperBound);

  /// Completes an admitted terminal action atomically. Unused upper-bound
  /// credits are released; an actual count beyond any reserved class fails.
  mlir::LogicalResult
  completeTerminalAction(TerminalActionReservation reservation,
                         const CoordinatedWorkEstimate &actual);

  /// Releases a rejected non-mandatory action that never reached a terminal
  /// gate. The mandatory baseline reservation cannot be released.
  mlir::LogicalResult
  releaseTerminalAction(TerminalActionReservation reservation);

  /// Consumes the pre-reserved repair allowance. It cannot borrow mandatory
  /// baseline or admitted terminal credits.
  bool tryConsumeRepair(uint64_t credits = 1);
  bool tryConsumeRepair(const CoordinatedWorkEstimate &actual);

  CoordinatedWorkLedgerSnapshot getSnapshot() const;

  static mlir::FailureOr<CoordinatedWorkEstimate>
  getTerminalActionUpperBound(int64_t rankCount);

private:
  struct TerminalReservationRecord {
    uint64_t id = 0;
    bool mandatoryBaseline = false;
    CoordinatedWorkEstimate upperBound;
  };

  CoordinatedWorkLedger(int64_t rankCount, uint64_t capacity,
                        uint64_t repairReserve,
                        CoordinatedWorkEstimate generationUpperBound,
                        CoordinatedWorkEstimate terminalUpperBound);

  static mlir::FailureOr<CoordinatedWorkEstimate>
  getMandatoryGenerationUpperBound(int64_t rankCount);
  TerminalReservationRecord *findReservation(uint64_t id);
  const TerminalReservationRecord *findReservation(uint64_t id) const;
  bool canReserve(uint64_t credits) const;
  void addConsumed(const CoordinatedWorkEstimate &actual);

  int64_t rankCount = 0;
  uint64_t capacity = 0;
  uint64_t consumed = 0;
  uint64_t mandatoryGenerationReserved = 0;
  uint64_t repairReserved = 0;
  uint64_t nextReservationId = 1;
  CoordinatedWorkEstimate mandatoryGenerationUpperBound;
  std::array<uint64_t, kCoordinatedWorkKindCount> consumedByKind = {};
  std::vector<TerminalReservationRecord> terminalReservations;
  TerminalActionReservation mandatoryBaselineReservation;
};

/// One real rank entry in an all-rank Tile candidate. The module is the only
/// semantic artifact; selectedTileIR is immutable debug evidence captured
/// from this actual clone before terminal lowering.
struct CoordinatedRankTileProgram {
  CoordinatedRankTileProgram(int64_t logicalRank,
                             mlir::OwningOpRef<mlir::ModuleOp> module,
                             std::shared_ptr<const std::string> selectedTileIR)
      : logicalRank(logicalRank), module(std::move(module)),
        selectedTileIR(std::move(selectedTileIR)) {}

  CoordinatedRankTileProgram(CoordinatedRankTileProgram &&) = default;
  CoordinatedRankTileProgram &
  operator=(CoordinatedRankTileProgram &&) = default;
  CoordinatedRankTileProgram(const CoordinatedRankTileProgram &) = delete;
  CoordinatedRankTileProgram &
  operator=(const CoordinatedRankTileProgram &) = delete;

  int64_t logicalRank = 0;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::shared_ptr<const std::string> selectedTileIR;
};

struct CoordinatedTileVariant {
  int64_t stableSemanticOrdinal = 0;
  bool reservedBaseline = false;
  uint8_t repairDepth = 0;
  TerminalActionReservation terminalReservation;
  std::vector<CoordinatedRankTileProgram> ranks;
  std::string frontierDigest;
};

using CoordinatedTileFrontier = std::vector<CoordinatedTileVariant>;

enum class CoordinatedTileRepairAction : uint8_t {
  SelectiveSpill,
  SplitAtExplicitDDRBoundary,
};

inline constexpr uint8_t kMaximumCoordinatedRepairDepth = 1;

struct CoordinatedDataflowSearchConfig {
  int64_t rankCount = 0;
  int64_t candidateParallelism = 1;
  OptimizationConfig optimizations = OptimizationConfig::production();
};

/// Builds a coordinated frontier of actual complete-rank Tile clones. The
/// current production seam materializes the mandatory conservative variant;
/// every returned member nevertheless already owns all-and-only rank entries
/// under one terminal reservation. No Tile-to-Instr, completion, SPM, DDR,
/// transport, ABI, cost selection, or winner commit occurs here.
mlir::FailureOr<CoordinatedTileFrontier>
buildCoordinatedTileFrontier(mlir::ModuleOp sourceModule,
                             const CoordinatedDataflowSearchConfig &config,
                             CoordinatedWorkLedger &ledger);

mlir::LogicalResult
verifyCoordinatedTileVariant(const CoordinatedTileVariant &variant,
                             int64_t expectedRankCount);

std::string
computeCoordinatedTileFrontierDigest(const CoordinatedTileFrontier &frontier);

std::string
computeCoordinatedTileVariantContentDigest(const CoordinatedTileVariant &variant);

/// Materializes one all-rank repair sibling directly from a still-live,
/// unplaced actual Tile parent. Terminal reservation is admitted before repair
/// work is charged; no source replay or failed Instr/offset state is accepted.
mlir::FailureOr<CoordinatedTileVariant> materializeCoordinatedTileRepair(
    const CoordinatedTileVariant &parent, CoordinatedTileRepairAction action,
    int64_t stableSemanticOrdinal, CoordinatedWorkLedger &ledger,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COORDINATEDDATAFLOWSEARCH_H
