//===- MovementTransferBuilder.h - Selected peer transfers -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTTRANSFERBUILDER_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTTRANSFERBUILDER_H

#include "Wafer/Planning/PhysicalDataflow/MovementDomain.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace wafer::compiler::detail {

struct MovementEndpointBinding {
  TileId tile{0};
  mlir::Value buffer;
  mlir::OpBuilder *builder = nullptr;
};

struct PreparedPeerTransfer {
  std::vector<MovementActionId> actions;
  std::vector<MovementHop> hops;
  uint64_t physicalBytes = 0;
};

struct EmittedPeerHop {
  MovementHop hop;
  mlir::Value sendToken;
  mlir::Value receiveToken;
};

struct EmittedPeerTransfer {
  llvm::SmallVector<mlir::Value, 4> sendTokens;
  llvm::SmallVector<mlir::Value, 4> receiveTokens;
  llvm::SmallVector<EmittedPeerHop, 4> hops;
};

/// Candidate-local bindings for one selected payload graph. `actions` is the
/// graph's semantic identity and must equal the realization's all-and-only
/// action set. Endpoint values remain owned by the caller's current IR.
struct SelectedPeerGraphBinding {
  std::vector<MovementActionId> actions;
  std::vector<TileId> terminals;
  std::vector<MovementEndpointBinding> endpoints;
  /// Required only for ExternalLoadFanout. The builder emits one explicit DDR
  /// load into the selected graph root before producing peer tokens.
  mlir::Value ddrSource;
  uint32_t payloadSlice = 0;
  llvm::SmallVector<int64_t, 4> logicalOffsets;
  llvm::SmallVector<int64_t, 4> logicalSizes;
};

struct PreparedSelectedPeerGraph {
  PreparedPeerTransfer transfer;
  std::vector<MovementEndpointBinding> endpoints;
  int64_t communication = 0;
  int64_t payload = 0;
  mlir::Value ddrSource;
  std::optional<TileId> ddrRoot;
};

struct EmittedSelectedPeerGraph {
  std::vector<MovementActionId> actions;
  EmittedPeerTransfer transfer;
  mlir::Operation *ddrRootLoad = nullptr;
};

mlir::FailureOr<PreparedPeerTransfer>
preparePeerTransfer(const MovementActionId &action,
                    const PeerTransferGraphPlan &realization,
                    llvm::ArrayRef<MovementEndpointBinding> endpoints,
                    std::string *failureReason = nullptr);

/// Emits matching target-routed send/recv operations for each explicit
/// compiler hop. Message ordinals are candidate-local lowering carriers and
/// never participate in MovementPlan identity.
mlir::FailureOr<EmittedPeerTransfer>
emitPreparedPeerTransfer(const PreparedPeerTransfer &prepared,
                         llvm::ArrayRef<MovementEndpointBinding> endpoints,
                         int64_t communication, int64_t payload,
                         std::string *failureReason = nullptr);

/// Preflights all-and-only selected non-DDR payload graphs without modifying
/// IR. Every member action and every exact finite payload piece is covered
/// once, shared fanout is prepared once per piece, and no fallback
/// realization is selected.
mlir::FailureOr<std::vector<PreparedSelectedPeerGraph>>
prepareSelectedPeerGraphs(const MovementPlan &plan,
                          llvm::ArrayRef<MovementResourceDescription> resources,
                          llvm::ArrayRef<SelectedPeerGraphBinding> bindings,
                          std::string *failureReason = nullptr);

/// Emits a preflighted selected plan. The caller owns the enclosing Card
/// transaction and discards it on failure.
mlir::FailureOr<std::vector<EmittedSelectedPeerGraph>>
emitPreparedSelectedPeerGraphs(
    llvm::ArrayRef<PreparedSelectedPeerGraph> prepared,
    std::string *failureReason = nullptr);

/// H-stage verifier for the token-only result. It intentionally rejects an
/// immediate await; Q50.J later consumes the exact token relations and places
/// waits at first-read/last-release/resource boundaries.
mlir::LogicalResult verifyTokenOnlySelectedPeerGraphs(
    llvm::ArrayRef<PreparedSelectedPeerGraph> prepared,
    llvm::ArrayRef<EmittedSelectedPeerGraph> emitted,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTTRANSFERBUILDER_H
