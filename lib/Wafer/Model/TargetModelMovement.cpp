//===- TargetModelMovement.cpp - Movement transaction kernels --------===//

#include "TargetModelKernelInternal.h"

#include <array>
#include <utility>
#include <vector>

namespace wafer::model::kernel_detail {
namespace {

llvm::Error
checkMovementSegmentBudget(const std::array<uint32_t, 3> &iterations,
                           TargetModelKernelBudget budget) {
  llvm::Expected<uint64_t> count = getDescriptorSegmentCount(iterations);
  if (!count)
    return count.takeError();
  if (*count > budget.getMaximumMovementSegments())
    return kernelError(TargetModelKernelErrorCode::WorkBudgetExceeded,
                       "movement segment count exceeds the explicit budget");
  return llvm::Error::success();
}

TargetModelStridedByteLayout
makeLayout(uint32_t innerBytes, const std::array<uint32_t, 3> &strides,
           const std::array<uint32_t, 3> &iterations) {
  return TargetModelStridedByteLayout{innerBytes, strides, iterations};
}

TargetModelCommandEffect
withReads(TargetModelCommandEffect effect,
          std::vector<TargetModelByteRead> pendingReads) {
  effect.pendingReads = std::move(pendingReads);
  return effect;
}

} // namespace

llvm::Expected<std::vector<uint8_t>>
readSnapshot(const InvocationMemoryRegistry &memory, int64_t launchSlot,
             TargetModelAddressSpace space, uint64_t address, uint64_t bytes,
             uint64_t alignment) {
  llvm::Expected<std::vector<uint8_t>> result =
      memory.readSnapshot(launchSlot, space, address, bytes, alignment);
  if (!result)
    return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                       llvm::toString(result.takeError()));
  return result;
}

llvm::Expected<TargetModelCommandEffect>
executeMovement(const compiler::TargetTransaction &transaction,
                const compiler::TargetStridedDMATransaction &value,
                const InvocationMemoryRegistry &memory,
                TargetModelKernelBudget budget) {
  if (value.byteCount > budget.getMaximumMovementBytes())
    return kernelError(TargetModelKernelErrorCode::WorkBudgetExceeded,
                       "DMA byte_count exceeds the explicit budget");
  if (llvm::Error error = checkMovementSegmentBudget(value.iterations, budget))
    return std::move(error);
  TargetModelStridedByteLayout layout =
      makeLayout(value.innerBytes, value.strides, value.iterations);

  if (value.direction == compiler::TargetDMADirection::Read) {
    llvm::Expected<std::vector<uint8_t>> payload = memory.readStridedSnapshot(
        transaction.launchSlotId.getValue(), TargetModelAddressSpace::CardDDR,
        value.source, layout, 1);
    if (!payload)
      return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                         llvm::toString(payload.takeError()));
    return withReads(
        TargetModelCommandEffect{
            {TargetModelByteWrite{transaction.launchSlotId.getValue(),
                                  TargetModelAddressSpace::TileSPM,
                                  value.destination, 1, std::move(*payload)}},
            {},
            TargetModelControlAction::None},
        {TargetModelByteRead{transaction.launchSlotId.getValue(),
                             TargetModelAddressSpace::CardDDR, value.source,
                             value.byteCount, layout}});
  }

  llvm::Expected<std::vector<uint8_t>> payload = readSnapshot(
      memory, transaction.launchSlotId.getValue(),
      TargetModelAddressSpace::TileSPM, value.source, value.byteCount);
  if (!payload)
    return payload.takeError();
  return withReads(
      TargetModelCommandEffect{
          {TargetModelByteWrite{transaction.launchSlotId.getValue(),
                                TargetModelAddressSpace::CardDDR,
                                value.destination, 1, std::move(*payload),
                                layout}},
          {},
          TargetModelControlAction::None},
      {TargetModelByteRead{transaction.launchSlotId.getValue(),
                           TargetModelAddressSpace::TileSPM, value.source,
                           value.byteCount, std::nullopt}});
}

llvm::Expected<TargetModelCommandEffect>
executeGatherScatter(const compiler::TargetTransaction &transaction,
                     const compiler::TargetGatherScatterTransaction &value,
                     const InvocationMemoryRegistry &memory,
                     TargetModelKernelBudget budget) {
  if (value.byteCount > budget.getMaximumMovementBytes())
    return kernelError(TargetModelKernelErrorCode::WorkBudgetExceeded,
                       "gather/scatter byte_count exceeds explicit budget");
  if (llvm::Error error =
          checkMovementSegmentBudget(value.sourceIterations, budget))
    return std::move(error);
  if (llvm::Error error =
          checkMovementSegmentBudget(value.destinationIterations, budget))
    return std::move(error);
  llvm::Expected<uint64_t> sourceSegments =
      getDescriptorSegmentCount(value.sourceIterations);
  if (!sourceSegments)
    return sourceSegments.takeError();
  llvm::Expected<uint64_t> destinationSegments =
      getDescriptorSegmentCount(value.destinationIterations);
  if (!destinationSegments)
    return destinationSegments.takeError();
  if (*sourceSegments != *destinationSegments)
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "gather/scatter source and destination segment counts "
                       "differ");
  TargetModelStridedByteLayout sourceLayout =
      makeLayout(value.innerBytes, value.sourceStrides, value.sourceIterations);
  TargetModelStridedByteLayout destinationLayout = makeLayout(
      value.innerBytes, value.destinationStrides, value.destinationIterations);
  llvm::Expected<std::vector<uint8_t>> snapshot = memory.readStridedSnapshot(
      transaction.launchSlotId.getValue(), TargetModelAddressSpace::TileSPM,
      value.source, sourceLayout, 1);
  if (!snapshot)
    return kernelError(TargetModelKernelErrorCode::MemoryReadFailure,
                       llvm::toString(snapshot.takeError()));
  return withReads(
      TargetModelCommandEffect{
          {TargetModelByteWrite{transaction.launchSlotId.getValue(),
                                TargetModelAddressSpace::TileSPM,
                                value.destination, 1, std::move(*snapshot),
                                destinationLayout}},
          {},
          TargetModelControlAction::None},
      {TargetModelByteRead{transaction.launchSlotId.getValue(),
                           TargetModelAddressSpace::TileSPM, value.source,
                           value.byteCount, sourceLayout}});
}

} // namespace wafer::model::kernel_detail
