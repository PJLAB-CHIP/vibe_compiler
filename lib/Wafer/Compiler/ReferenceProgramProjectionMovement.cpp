//===- ReferenceProgramProjectionMovement.cpp - Movement projection -----===//

#include "ReferenceProgramProjectionInternal.h"

#include <limits>

namespace wafer::compiler::reference_detail {
namespace {

llvm::Error validateMovementByteCount(uint64_t byteCount, uint64_t innerBytes,
                                      llvm::ArrayRef<int64_t> iterations) {
  if (iterations.size() != 3 || innerBytes == 0)
    return invalid("movement descriptor has invalid iteration rank");
  uint64_t chunks = 1;
  for (int64_t iteration : iterations) {
    if (iteration < 0 || static_cast<uint64_t>(iteration) >
                             std::numeric_limits<uint64_t>::max() / chunks)
      return invalid("movement descriptor iteration count overflows");
    chunks *= static_cast<uint64_t>(iteration);
  }
  if (chunks > std::numeric_limits<uint64_t>::max() / innerBytes ||
      byteCount != chunks * innerBytes)
    return invalid("movement byte_count disagrees with its descriptor");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<bool>
ProgramProjector::projectMovement(mlir::Operation &operation,
                                  Command &command) {
  if (auto rdma = mlir::dyn_cast<wafer::InstrRDMAOp>(operation)) {
    if (llvm::Error error = validateMovementByteCount(
            rdma.getByteCount(), rdma.getInnerBytes(), rdma.getSrcIterations()))
      return error;
    command.kind = CommandKind::RDMA;
    if (llvm::Error error =
            projectMovementOperands(rdma.getSource(), rdma.getDest(), command))
      return error;
    command.sourceStrides.assign(rdma.getSrcStrides().begin(),
                                 rdma.getSrcStrides().end());
    command.sourceIterations.assign(rdma.getSrcIterations().begin(),
                                    rdma.getSrcIterations().end());
    command.byteCount = rdma.getByteCount();
    command.innerBytes = rdma.getInnerBytes();
  } else if (auto wdma = mlir::dyn_cast<wafer::InstrWDMAOp>(operation)) {
    if (llvm::Error error = validateMovementByteCount(
            wdma.getByteCount(), wdma.getInnerBytes(), wdma.getDstIterations()))
      return error;
    command.kind = CommandKind::WDMA;
    if (llvm::Error error =
            projectMovementOperands(wdma.getSource(), wdma.getDest(), command))
      return error;
    command.destStrides.assign(wdma.getDstStrides().begin(),
                               wdma.getDstStrides().end());
    command.destIterations.assign(wdma.getDstIterations().begin(),
                                  wdma.getDstIterations().end());
    command.byteCount = wdma.getByteCount();
    command.innerBytes = wdma.getInnerBytes();
  } else if (auto movement =
                 mlir::dyn_cast<wafer::InstrGatherScatterOp>(operation)) {
    if (llvm::Error error = validateMovementByteCount(
            movement.getByteCount(), movement.getInnerBytes(),
            movement.getSrcIterations()))
      return error;
    command.kind = CommandKind::GatherScatter;
    if (llvm::Error error = projectMovementOperands(
            movement.getSource(), movement.getDest(), command))
      return error;
    command.sourceStrides.assign(movement.getSrcStrides().begin(),
                                 movement.getSrcStrides().end());
    command.sourceIterations.assign(movement.getSrcIterations().begin(),
                                    movement.getSrcIterations().end());
    command.destStrides.assign(movement.getDstStrides().begin(),
                               movement.getDstStrides().end());
    command.destIterations.assign(movement.getDstIterations().begin(),
                                  movement.getDstIterations().end());
    command.sourceOffset = movement.getSrcOffset();
    command.destOffset = movement.getDstOffset();
    command.byteCount = movement.getByteCount();
    command.innerBytes = movement.getInnerBytes();

  } else {
    return false;
  }
  return true;
}

llvm::Error ProgramProjector::projectMovementOperands(mlir::Value sourceValue,
                                                      mlir::Value destValue,
                                                      Command &command) {
  auto source = use(sourceValue);
  auto dest = use(destValue);
  if (!source)
    return source.takeError();
  if (!dest)
    return dest.takeError();
  command.source = *source;
  command.dest = *dest;
  return llvm::Error::success();
}

} // namespace wafer::compiler::reference_detail
