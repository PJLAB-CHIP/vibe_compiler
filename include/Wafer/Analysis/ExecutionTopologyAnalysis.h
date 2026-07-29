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

/// One directed adjacency edge in the typed physical topology.
struct ExecutionDirectedLink {
  ExecutionEndpoint source;
  ExecutionEndpoint destination;

  bool operator==(const ExecutionDirectedLink &other) const {
    return source == other.source && destination == other.destination;
  }
};

/// Recomputable, read-only facts derived from the unique
/// wafer.target.topology / wafer.execution.mesh pair in a module.
///
/// The analysis models only adjacency established by the typed topology IR.
/// Distances are unweighted shortest-hop lower bounds. The canonical path API
/// below is a deterministic modeling mechanism, not a claim about hardware
/// routing, directional link loads, congestion, or execution times.
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
  /// Number of directed adjacency links in the complete available topology
  /// graph. This is a topology fact, not a bandwidth or routing guarantee.
  uint64_t getDirectedLinkCount() const { return directedLinkCount; }

  std::optional<ExecutionEndpoint> getRankEndpoint(int64_t logicalRank) const;

  /// Returns the minimum number of available topology links between two
  /// logical ranks, or std::nullopt for an out-of-domain/unreachable query.
  std::optional<uint64_t> getShortestHopDistance(int64_t sourceRank,
                                                 int64_t destinationRank) const;

  /// Returns one deterministic shortest path as directed typed-topology
  /// adjacency links. Equal-length alternatives are resolved by examining
  /// every endpoint's neighbors in ascending row-major order during BFS. The
  /// result is therefore stable for a fixed typed topology, including meshes,
  /// card toruses and unavailable endpoints, but is only a canonical route for
  /// static modeling.
  mlir::FailureOr<llvm::SmallVector<ExecutionDirectedLink, 8>>
  getCanonicalShortestPath(int64_t sourceRank, int64_t destinationRank) const;

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
  uint64_t directedLinkCount = 0;
};

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_EXECUTIONTOPOLOGYANALYSIS_H
