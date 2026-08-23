//===- MovementTransferBuilder.cpp - Selected peer transfers ---------===//

#include "Wafer/Planning/PhysicalDataflow/MovementTransferBuilder.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

void setFailure(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
}

mlir::LogicalResult fail(std::string *failureReason, llvm::StringRef detail) {
  setFailure(failureReason, detail);
  return mlir::failure();
}

const MovementEndpointBinding *
findEndpoint(llvm::ArrayRef<MovementEndpointBinding> endpoints, TileId tile) {
  auto found = llvm::find_if(
      endpoints, [&](const auto &endpoint) { return endpoint.tile == tile; });
  return found == endpoints.end() ? nullptr : &*found;
}

} // namespace

mlir::FailureOr<PreparedPeerTransfer>
preparePeerTransfer(const MovementActionId &action,
                    const MovementRealization &realization,
                    llvm::ArrayRef<MovementEndpointBinding> endpoints,
                    std::string *failureReason) {
  if (realization.kind == MovementRealizationKind::DDRStage ||
      realization.hops.empty()) {
    setFailure(failureReason, "peer transfer requires an explicit non-DDR hop");
    return mlir::failure();
  }
  if (realization.kind == MovementRealizationKind::TargetRoutedPeer &&
      realization.hops.size() != 1) {
    setFailure(failureReason,
               "target-routed transfer must contain one endpoint hop");
    return mlir::failure();
  }
  std::map<int64_t, const MovementEndpointBinding *> uniqueEndpoints;
  for (const MovementEndpointBinding &endpoint : endpoints)
    if (!endpoint.buffer || !endpoint.builder ||
        !uniqueEndpoints.try_emplace(endpoint.tile.getValue(), &endpoint)
             .second) {
      setFailure(failureReason,
                 "peer transfer endpoints are missing or duplicated");
      return mlir::failure();
    }

  std::optional<int64_t> physicalBytes;
  TileId current = realization.hops.front().source;
  std::set<int64_t> visited{current.getValue()};
  for (const MovementHop &hop : realization.hops) {
    if (hop.source != current || hop.source == hop.destination ||
        !visited.insert(hop.destination.getValue()).second) {
      setFailure(failureReason,
                 "peer transfer hop graph is cyclic or disconnected");
      return mlir::failure();
    }
    const MovementEndpointBinding *source = findEndpoint(endpoints, hop.source);
    const MovementEndpointBinding *destination =
        findEndpoint(endpoints, hop.destination);
    auto sourceType =
        source ? mlir::dyn_cast<mlir::MemRefType>(source->buffer.getType())
               : mlir::MemRefType{};
    auto destinationType =
        destination
            ? mlir::dyn_cast<mlir::MemRefType>(destination->buffer.getType())
            : mlir::MemRefType{};
    std::optional<WaferPhysicalTensorInfo> sourceInfo =
        sourceType ? computeWaferPhysicalTensorInfo(sourceType) : std::nullopt;
    std::optional<WaferPhysicalTensorInfo> destinationInfo =
        destinationType ? computeWaferPhysicalTensorInfo(destinationType)
                        : std::nullopt;
    if (!source || !destination || !sourceInfo || !destinationInfo ||
        sourceInfo->physicalBytes <= 0 ||
        sourceInfo->physicalBytes != destinationInfo->physicalBytes ||
        getWaferMemoryAttr(sourceType) != getWaferMemoryAttr(destinationType)) {
      setFailure(failureReason,
                 "peer transfer endpoints have incompatible physical buffers");
      return mlir::failure();
    }
    if (physicalBytes && *physicalBytes != sourceInfo->physicalBytes) {
      setFailure(failureReason, "peer relay changes physical payload size");
      return mlir::failure();
    }
    physicalBytes = sourceInfo->physicalBytes;
    current = hop.destination;
  }
  if (!physicalBytes || static_cast<uint64_t>(*physicalBytes) >
                            std::numeric_limits<uint32_t>::max()) {
    setFailure(failureReason,
               "peer transfer payload is outside target byte range");
    return mlir::failure();
  }
  return PreparedPeerTransfer{action, realization.hops,
                              static_cast<uint64_t>(*physicalBytes)};
}

mlir::FailureOr<EmittedPeerTransfer>
emitPreparedPeerTransfer(const PreparedPeerTransfer &prepared,
                         llvm::ArrayRef<MovementEndpointBinding> endpoints,
                         int64_t communication, int64_t payload,
                         std::string *failureReason) {
  if (communication < 0 || payload < 0 || prepared.hops.empty() ||
      prepared.physicalBytes == 0 ||
      prepared.physicalBytes > std::numeric_limits<uint32_t>::max())
    return fail(failureReason, "prepared peer transfer metadata is invalid");
  EmittedPeerTransfer emitted;
  for (auto [round, hop] : llvm::enumerate(prepared.hops)) {
    const MovementEndpointBinding *source = findEndpoint(endpoints, hop.source);
    const MovementEndpointBinding *destination =
        findEndpoint(endpoints, hop.destination);
    if (!source || !destination || !source->builder || !destination->builder)
      return fail(failureReason, "prepared peer transfer endpoint disappeared");
    auto message =
        DTEMessageAttr::get(source->builder->getContext(), communication,
                            static_cast<int64_t>(round), payload);
    auto send = source->builder->create<CommPeerSendOp>(
        source->buffer.getLoc(),
        mlir::async::TokenType::get(source->builder->getContext()),
        source->buffer,
        source->builder->getI64IntegerAttr(hop.destination.getValue()),
        source->builder->getI64IntegerAttr(
            static_cast<int64_t>(prepared.physicalBytes)),
        message);
    emitted.sendTokens.push_back(send.getToken());
    auto receive = destination->builder->create<CommPeerRecvOp>(
        destination->buffer.getLoc(),
        mlir::async::TokenType::get(destination->builder->getContext()),
        destination->buffer,
        destination->builder->getI64IntegerAttr(hop.source.getValue()),
        destination->builder->getI64IntegerAttr(
            static_cast<int64_t>(prepared.physicalBytes)),
        message);
    emitted.receiveTokens.push_back(receive.getToken());
  }
  return emitted;
}

} // namespace wafer::compiler::detail
