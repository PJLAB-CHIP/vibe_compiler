//===- TargetModelMovement.cpp - Movement transaction kernels --------===//

#include "TargetModelKernelInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <limits>
#include <utility>
#include <vector>

namespace wafer::model::kernel_detail {
namespace {

llvm::Expected<std::vector<uint64_t>> getSegmentAddresses(
    uint64_t base, uint32_t innerBytes, const std::array<uint32_t, 3> &strides,
    const std::array<uint32_t, 3> &iterations, TargetModelKernelBudget budget) {
  llvm::Expected<uint64_t> count = getDescriptorSegmentCount(iterations);
  if (!count)
    return count.takeError();
  if (*count > budget.getMaximumMovementSegments())
    return kernelError(TargetModelKernelErrorCode::WorkBudgetExceeded,
                       "movement segment count exceeds the explicit budget");
  std::vector<uint64_t> addresses;
  addresses.reserve(static_cast<size_t>(*count));
  for (uint64_t outer = 0; outer < iterations[2]; ++outer)
    for (uint64_t middle = 0; middle < iterations[1]; ++middle)
      for (uint64_t inner = 0; inner < iterations[0]; ++inner) {
        uint64_t address = base;
        for (auto [index, coordinate] :
             llvm::enumerate(std::array<uint64_t, 3>{inner, middle, outer})) {
          uint64_t delta = 0;
          if (!checkedMultiply(coordinate, strides[index], delta) ||
              !checkedAdd(address, delta, address) ||
              !checkedAdd(address, innerBytes, delta))
            return kernelError(
                TargetModelKernelErrorCode::InvalidTransactionField,
                "movement segment address overflows");
        }
        addresses.push_back(address);
      }
  return addresses;
}

} // namespace

llvm::Expected<std::vector<uint8_t>>
readSnapshot(const InvocationMemoryRegistry &memory, int64_t rank,
             TargetModelAddressSpace space, uint64_t address, uint64_t bytes,
             uint64_t alignment) {
  llvm::Expected<std::vector<uint8_t>> result =
      memory.readSnapshot(rank, space, address, bytes, alignment);
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
  llvm::Expected<std::vector<uint64_t>> addresses = getSegmentAddresses(
      value.direction == compiler::TargetDMADirection::Read ? value.source
                                                            : value.destination,
      value.innerBytes, value.strides, value.iterations, budget);
  if (!addresses)
    return addresses.takeError();

  if (value.direction == compiler::TargetDMADirection::Read) {
    std::vector<uint8_t> payload;
    payload.reserve(value.byteCount);
    for (uint64_t address : *addresses) {
      llvm::Expected<std::vector<uint8_t>> segment = readSnapshot(
          memory, transaction.logicalRank, TargetModelAddressSpace::CardDDR,
          address, value.innerBytes);
      if (!segment)
        return segment.takeError();
      payload.insert(payload.end(), segment->begin(), segment->end());
    }
    return TargetModelCommandEffect{
        {TargetModelByteWrite{transaction.logicalRank,
                              TargetModelAddressSpace::RankSPM,
                              value.destination, 1, std::move(payload)}},
        {},
        TargetModelControlAction::None};
  }

  llvm::Expected<std::vector<uint8_t>> payload = readSnapshot(
      memory, transaction.logicalRank, TargetModelAddressSpace::RankSPM,
      value.source, value.byteCount);
  if (!payload)
    return payload.takeError();
  std::vector<TargetModelByteWrite> writes;
  writes.reserve(addresses->size());
  for (auto [index, address] : llvm::enumerate(*addresses)) {
    const size_t begin = index * value.innerBytes;
    writes.push_back(TargetModelByteWrite{
        transaction.logicalRank, TargetModelAddressSpace::CardDDR, address, 1,
        std::vector<uint8_t>(payload->begin() + begin,
                             payload->begin() + begin + value.innerBytes)});
  }
  return TargetModelCommandEffect{
      std::move(writes), {}, TargetModelControlAction::None};
}

llvm::Expected<TargetModelCommandEffect>
executeGatherScatter(const compiler::TargetTransaction &transaction,
                     const compiler::TargetGatherScatterTransaction &value,
                     const InvocationMemoryRegistry &memory,
                     TargetModelKernelBudget budget) {
  if (value.byteCount > budget.getMaximumMovementBytes())
    return kernelError(TargetModelKernelErrorCode::WorkBudgetExceeded,
                       "gather/scatter byte_count exceeds explicit budget");
  llvm::Expected<std::vector<uint64_t>> sourceAddresses =
      getSegmentAddresses(value.source, value.innerBytes, value.sourceStrides,
                          value.sourceIterations, budget);
  if (!sourceAddresses)
    return sourceAddresses.takeError();
  llvm::Expected<std::vector<uint64_t>> destinationAddresses =
      getSegmentAddresses(value.destination, value.innerBytes,
                          value.destinationStrides, value.destinationIterations,
                          budget);
  if (!destinationAddresses)
    return destinationAddresses.takeError();
  if (sourceAddresses->size() != destinationAddresses->size())
    return kernelError(TargetModelKernelErrorCode::InvalidTransactionField,
                       "gather/scatter source and destination segment counts "
                       "differ");
  std::vector<std::vector<uint8_t>> snapshots;
  snapshots.reserve(sourceAddresses->size());
  for (uint64_t address : *sourceAddresses) {
    llvm::Expected<std::vector<uint8_t>> segment = readSnapshot(
        memory, transaction.logicalRank, TargetModelAddressSpace::RankSPM,
        address, value.innerBytes);
    if (!segment)
      return segment.takeError();
    snapshots.push_back(std::move(*segment));
  }
  std::vector<TargetModelByteWrite> writes;
  writes.reserve(destinationAddresses->size());
  for (auto [index, address] : llvm::enumerate(*destinationAddresses))
    writes.push_back({transaction.logicalRank, TargetModelAddressSpace::RankSPM,
                      address, 1, std::move(snapshots[index])});
  return TargetModelCommandEffect{
      std::move(writes), {}, TargetModelControlAction::None};
}

} // namespace wafer::model::kernel_detail
