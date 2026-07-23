//===- ExecutionTopologyAnalysis.h - Execution topology facts -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_EXECUTIONTOPOLOGYANALYSIS_H
#define WAFER_ANALYSIS_EXECUTIONTOPOLOGYANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>

namespace wafer::analysis {

/// A typed physical endpoint in the regular topology coordinate system.
struct ExecutionEndpoint {
  int64_t cardY = 0;
  int64_t cardX = 0;
  int64_t tileY = 0;
  int64_t tileX = 0;

  bool operator==(const ExecutionEndpoint &other) const {
    return cardY == other.cardY && cardX == other.cardX &&
           tileY == other.tileY && tileX == other.tileX;
  }
};

/// Recomputable, read-only facts derived from the unique
/// wafer.target.topology / wafer.execution.mesh pair in a module.
///
/// The analysis models only adjacency established by the typed topology IR.
/// Distances are unweighted shortest-hop lower bounds. They are not routes,
/// directional link loads, congestion estimates, or execution times.
class ExecutionTopologyAnalysis {
public:
  static mlir::FailureOr<ExecutionTopologyAnalysis>
  create(mlir::ModuleOp module);

  int64_t getRankCount() const {
    return static_cast<int64_t>(rankEndpoints.size());
  }
  llvm::ArrayRef<int64_t> getCardGrid() const { return cardGrid; }
  llvm::ArrayRef<int64_t> getTileGrid() const { return tileGrid; }
  llvm::StringRef getCardInterconnect() const {
    return llvm::StringRef(cardInterconnect.data(), cardInterconnect.size());
  }
  llvm::ArrayRef<ExecutionEndpoint> getRankEndpoints() const {
    return rankEndpoints;
  }

  std::optional<ExecutionEndpoint> getRankEndpoint(int64_t logicalRank) const;

  /// Returns the minimum number of available topology links between two
  /// logical ranks, or std::nullopt for an out-of-domain/unreachable query.
  std::optional<uint64_t> getShortestHopDistance(int64_t sourceRank,
                                                 int64_t destinationRank) const;

  /// Returns a row-major shortest-hop matrix for the requested logical ranks.
  /// Repeated ranks are allowed and preserve caller order.
  mlir::FailureOr<llvm::SmallVector<uint64_t, 16>>
  getShortestHopMatrix(llvm::ArrayRef<int64_t> logicalRanks) const;

  /// Compares the complete derived topology/placement fact set. This is used
  /// to fail closed when independently cloned rank modules disagree.
  bool isEquivalentTo(const ExecutionTopologyAnalysis &other) const;

private:
  std::array<int64_t, 2> cardGrid{};
  std::array<int64_t, 2> tileGrid{};
  llvm::SmallVector<char, 8> cardInterconnect;
  llvm::SmallVector<unsigned char, 64> endpointAvailability;
  llvm::SmallVector<ExecutionEndpoint, 16> rankEndpoints;
  llvm::SmallVector<uint64_t, 256> shortestHopDistances;
};

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_EXECUTIONTOPOLOGYANALYSIS_H
