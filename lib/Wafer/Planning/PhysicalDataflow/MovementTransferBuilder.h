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
#include <string>
#include <vector>

namespace wafer::compiler::detail {

struct MovementEndpointBinding {
  TileId tile{0};
  mlir::Value buffer;
  mlir::OpBuilder *builder = nullptr;
};

struct PreparedPeerTransfer {
  MovementActionId action;
  std::vector<MovementHop> hops;
  uint64_t physicalBytes = 0;
};

struct EmittedPeerTransfer {
  llvm::SmallVector<mlir::Value, 4> sendTokens;
  llvm::SmallVector<mlir::Value, 4> receiveTokens;
};

mlir::FailureOr<PreparedPeerTransfer>
preparePeerTransfer(const MovementActionId &action,
                    const MovementRealization &realization,
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

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_MOVEMENTTRANSFERBUILDER_H
