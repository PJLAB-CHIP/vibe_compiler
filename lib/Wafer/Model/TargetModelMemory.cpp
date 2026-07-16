//===- TargetModelMemory.cpp - Private target model memory --------------===//

#include "Wafer/Model/TargetModelMemory.h"

#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetProfile.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

namespace wafer::model {
namespace {

llvm::Error memoryError(TargetModelMemoryErrorCode code,
                        const llvm::Twine &detail) {
  return llvm::make_error<TargetModelMemoryError>(code, detail.str());
}

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool isPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool isInputRole(compiler::KernelABISlotRole role) {
  switch (role) {
  case compiler::KernelABISlotRole::UserInput:
  case compiler::KernelABISlotRole::Parameter:
  case compiler::KernelABISlotRole::Constant:
    return true;
  case compiler::KernelABISlotRole::Output:
  case compiler::KernelABISlotRole::Workspace:
  case compiler::KernelABISlotRole::TransportStatus:
    return false;
  }
  llvm_unreachable("unknown kernel ABI slot role");
}

bool permitsRead(compiler::KernelABISlotRole) { return true; }

bool permitsWrite(compiler::KernelABISlotRole role) {
  switch (role) {
  case compiler::KernelABISlotRole::UserInput:
  case compiler::KernelABISlotRole::Parameter:
  case compiler::KernelABISlotRole::Constant:
    return false;
  case compiler::KernelABISlotRole::Output:
  case compiler::KernelABISlotRole::Workspace:
  case compiler::KernelABISlotRole::TransportStatus:
    return true;
  }
  llvm_unreachable("unknown kernel ABI slot role");
}

bool permitsAccess(compiler::KernelABISlotRole role, TargetModelAccess access) {
  switch (access) {
  case TargetModelAccess::Read:
    return permitsRead(role);
  case TargetModelAccess::Write:
    return permitsWrite(role);
  case TargetModelAccess::ReadWrite:
    return permitsRead(role) && permitsWrite(role);
  }
  llvm_unreachable("unknown target model access");
}

std::string rankSlot(int64_t logicalRank, int64_t slotOrdinal) {
  return (llvm::Twine("rank ") + llvm::Twine(logicalRank) + " slot " +
          llvm::Twine(slotOrdinal))
      .str();
}

} // namespace

llvm::StringRef
stringifyTargetModelMemoryErrorCode(TargetModelMemoryErrorCode code) {
  switch (code) {
  case TargetModelMemoryErrorCode::InvalidInvocation:
    return "invalid-invocation";
  case TargetModelMemoryErrorCode::InvalidSlot:
    return "invalid-slot";
  case TargetModelMemoryErrorCode::InvalidInputBinding:
    return "invalid-input-binding";
  case TargetModelMemoryErrorCode::AddressOverflow:
    return "address-overflow";
  case TargetModelMemoryErrorCode::AddressMisaligned:
    return "address-misaligned";
  case TargetModelMemoryErrorCode::ReservedSPM:
    return "reserved-spm";
  case TargetModelMemoryErrorCode::UnknownResource:
    return "unknown-resource";
  case TargetModelMemoryErrorCode::CrossResource:
    return "cross-resource";
  case TargetModelMemoryErrorCode::AccessDenied:
    return "access-denied";
  case TargetModelMemoryErrorCode::InvalidEffect:
    return "invalid-effect";
  }
  llvm_unreachable("unknown target model memory error code");
}

char TargetModelMemoryError::ID;

void TargetModelMemoryError::log(llvm::raw_ostream &stream) const {
  stream << "target model memory " << stringifyTargetModelMemoryErrorCode(code)
         << ": " << detail;
}

std::error_code TargetModelMemoryError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<InvocationAddressPlan> InvocationAddressPlan::create(
    const compiler::TargetCallInvocationDescriptor &invocation,
    llvm::ArrayRef<TargetModelInputBinding> inputBindings) {
  if (invocation.ranks.empty())
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "invocation has no logical ranks");

  const TargetProfileRecord &profile =
      getTargetProfileRecord(invocation.targetProfile);
  const size_t rankCount = invocation.ranks.size();
  std::vector<const compiler::TargetCallRankDescriptor *> ranks(rankCount,
                                                                nullptr);
  for (const compiler::TargetCallRankDescriptor &rank : invocation.ranks) {
    if (rank.logicalRank < 0 ||
        static_cast<uint64_t>(rank.logicalRank) >= rankCount)
      return memoryError(
          TargetModelMemoryErrorCode::InvalidInvocation,
          llvm::Twine("logical rank domain must be exactly [0, ") +
              llvm::Twine(rankCount) + ")");
    if (ranks[static_cast<size_t>(rank.logicalRank)])
      return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                         llvm::Twine("duplicate logical rank ") +
                             llvm::Twine(rank.logicalRank));
    if (rank.targetIdentity != profile.targetIdentity ||
        rank.kernelRuntimeABI != profile.kernelRuntimeABI)
      return memoryError(
          TargetModelMemoryErrorCode::InvalidInvocation,
          llvm::Twine("rank ") + llvm::Twine(rank.logicalRank) +
              " identity or kernel runtime ABI differs from target profile");
    ranks[static_cast<size_t>(rank.logicalRank)] = &rank;
  }
  if (llvm::is_contained(ranks, nullptr))
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "logical rank domain is incomplete");

  std::set<std::pair<int64_t, int64_t>> boundInputs;
  for (const TargetModelInputBinding &binding : inputBindings) {
    if (binding.logicalRank < 0 ||
        static_cast<uint64_t>(binding.logicalRank) >= rankCount)
      return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                         "input binding names an unknown logical rank");
    const compiler::TargetCallRankDescriptor &rank =
        *ranks[static_cast<size_t>(binding.logicalRank)];
    if (binding.slotOrdinal < 0 || static_cast<uint64_t>(binding.slotOrdinal) >=
                                       rank.kernelABISlots.size())
      return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                         "input binding names an unknown slot ordinal");
    if (!boundInputs.emplace(binding.logicalRank, binding.slotOrdinal).second)
      return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                         "duplicate input binding identity");
  }

  const TargetMemoryPolicy memoryPolicy = getDefaultWaferTargetPolicy().memory;
  if (memoryPolicy.spmBase < 0 || memoryPolicy.spmLimit <= memoryPolicy.spmBase)
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "target policy has an invalid SPM planned window");
  const uint64_t spmBase = static_cast<uint64_t>(memoryPolicy.spmBase);
  const uint64_t spmLimit = static_cast<uint64_t>(memoryPolicy.spmLimit);

  std::vector<int64_t> logicalRanks;
  std::vector<TargetModelPlannedSlot> slots;
  std::vector<InitialSlotStorage> initialStorage;
  logicalRanks.reserve(rankCount);
  for (size_t rankIndex = 0; rankIndex < rankCount; ++rankIndex) {
    const compiler::TargetCallRankDescriptor &rank = *ranks[rankIndex];
    logicalRanks.push_back(rank.logicalRank);
    if (rank.kernelABISlots.size() != rank.slotValues.size())
      return memoryError(
          TargetModelMemoryErrorCode::InvalidInvocation,
          llvm::Twine("rank ") + llvm::Twine(rank.logicalRank) +
              " ABI slot metadata and values have different lengths");
    for (size_t slotIndex = 0; slotIndex < rank.kernelABISlots.size();
         ++slotIndex) {
      const compiler::KernelABISlot &slot = rank.kernelABISlots[slotIndex];
      if (slot.ordinal != static_cast<int64_t>(slotIndex))
        return memoryError(TargetModelMemoryErrorCode::InvalidSlot,
                           rankSlot(rank.logicalRank, slot.ordinal) +
                               " is not in ordinal order");
      if (slot.byteSize <= 0 || slot.alignment <= 0 ||
          !isPowerOfTwo(static_cast<uint64_t>(slot.alignment)))
        return memoryError(TargetModelMemoryErrorCode::InvalidSlot,
                           rankSlot(rank.logicalRank, slot.ordinal) +
                               " has invalid size or alignment");
      const uint64_t base = rank.slotValues[slotIndex];
      const uint64_t byteSize = static_cast<uint64_t>(slot.byteSize);
      const uint64_t alignment = static_cast<uint64_t>(slot.alignment);
      uint64_t end = 0;
      if (!checkedAdd(base, byteSize, end))
        return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                           rankSlot(rank.logicalRank, slot.ordinal) +
                               " address range overflows");
      if (base % alignment != 0)
        return memoryError(TargetModelMemoryErrorCode::AddressMisaligned,
                           rankSlot(rank.logicalRank, slot.ordinal) +
                               " base does not satisfy ABI alignment");

      slots.push_back({rank.logicalRank, slot.ordinal, slot.role,
                       slot.resourceIndex, base, byteSize, alignment});

      const TargetModelInputBinding *input = nullptr;
      for (const TargetModelInputBinding &candidate : inputBindings)
        if (candidate.logicalRank == rank.logicalRank &&
            candidate.slotOrdinal == slot.ordinal) {
          input = &candidate;
          break;
        }
      if (isInputRole(slot.role)) {
        if (!input)
          return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                             rankSlot(rank.logicalRank, slot.ordinal) +
                                 " requires exact initial bytes");
        if (input->bytes.size() != byteSize)
          return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                             rankSlot(rank.logicalRank, slot.ordinal) +
                                 " initial byte count differs from ABI size");
        initialStorage.push_back(
            {rank.logicalRank, slot.ordinal, input->bytes});
      } else {
        if (input)
          return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                             rankSlot(rank.logicalRank, slot.ordinal) +
                                 " is model-owned and cannot be prebound");
        initialStorage.push_back(
            {rank.logicalRank, slot.ordinal,
             std::vector<uint8_t>(static_cast<size_t>(byteSize), 0)});
      }
    }
  }

  std::vector<const TargetModelPlannedSlot *> byBase;
  byBase.reserve(slots.size());
  for (const TargetModelPlannedSlot &slot : slots)
    byBase.push_back(&slot);
  llvm::sort(byBase, [](const TargetModelPlannedSlot *lhs,
                        const TargetModelPlannedSlot *rhs) {
    return std::tie(lhs->base, lhs->logicalRank, lhs->slotOrdinal) <
           std::tie(rhs->base, rhs->logicalRank, rhs->slotOrdinal);
  });
  for (size_t index = 1; index < byBase.size(); ++index) {
    uint64_t previousEnd = 0;
    (void)checkedAdd(byBase[index - 1]->base, byBase[index - 1]->byteSize,
                     previousEnd);
    if (byBase[index]->base < previousEnd)
      return memoryError(
          TargetModelMemoryErrorCode::InvalidSlot,
          rankSlot(byBase[index - 1]->logicalRank,
                   byBase[index - 1]->slotOrdinal) +
              " overlaps " +
              rankSlot(byBase[index]->logicalRank, byBase[index]->slotOrdinal));
  }

  return InvocationAddressPlan(invocation.targetProfile,
                               std::move(logicalRanks), std::move(slots),
                               std::move(initialStorage), spmBase, spmLimit);
}

const TargetModelPlannedSlot *
InvocationAddressPlan::findSlot(int64_t logicalRank,
                                int64_t slotOrdinal) const {
  for (const TargetModelPlannedSlot &slot : slots)
    if (slot.logicalRank == logicalRank && slot.slotOrdinal == slotOrdinal)
      return &slot;
  return nullptr;
}

const InvocationAddressPlan::InitialSlotStorage *
InvocationAddressPlan::findInitialStorage(int64_t logicalRank,
                                          int64_t slotOrdinal) const {
  for (const InitialSlotStorage &storage : initialStorage)
    if (storage.logicalRank == logicalRank &&
        storage.slotOrdinal == slotOrdinal)
      return &storage;
  return nullptr;
}

llvm::Expected<TargetModelResolvedRange> InvocationAddressPlan::resolve(
    int64_t logicalRank, TargetModelAddressSpace addressSpace,
    TargetModelAccess access, uint64_t address, uint64_t byteCount,
    uint64_t requiredAlignment) const {
  if (!llvm::is_contained(logicalRanks, logicalRank))
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       llvm::Twine("unknown logical rank ") +
                           llvm::Twine(logicalRank));
  if (byteCount == 0)
    return memoryError(TargetModelMemoryErrorCode::UnknownResource,
                       "zero-byte ranges are not addressable resources");
  if (!isPowerOfTwo(requiredAlignment) || address % requiredAlignment != 0)
    return memoryError(TargetModelMemoryErrorCode::AddressMisaligned,
                       "address does not satisfy the requested alignment");
  uint64_t end = 0;
  if (!checkedAdd(address, byteCount, end))
    return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                       "address plus byte count overflows");

  if (addressSpace == TargetModelAddressSpace::RankSPM) {
    if (address < spmBase || end > spmLimit)
      return memoryError(
          TargetModelMemoryErrorCode::ReservedSPM,
          llvm::Twine("SPM range is outside the planned tensor window [") +
              llvm::Twine(spmBase) + ", " + llvm::Twine(spmLimit) + ")");
    return TargetModelResolvedRange{logicalRank, addressSpace, std::nullopt,
                                    address - spmBase, byteCount};
  }

  const TargetModelPlannedSlot *containingStart = nullptr;
  for (const TargetModelPlannedSlot &slot : slots) {
    uint64_t slotEnd = 0;
    (void)checkedAdd(slot.base, slot.byteSize, slotEnd);
    if (address >= slot.base && address < slotEnd) {
      containingStart = &slot;
      break;
    }
  }
  if (!containingStart)
    return memoryError(TargetModelMemoryErrorCode::UnknownResource,
                       "DDR address does not begin in an ABI slot resource");
  uint64_t slotEnd = 0;
  (void)checkedAdd(containingStart->base, containingStart->byteSize, slotEnd);
  if (end > slotEnd)
    return memoryError(TargetModelMemoryErrorCode::CrossResource,
                       "DDR range crosses its ABI slot resource boundary");
  if (!permitsAccess(containingStart->role, access))
    return memoryError(
        TargetModelMemoryErrorCode::AccessDenied,
        rankSlot(containingStart->logicalRank, containingStart->slotOrdinal) +
            " does not permit the requested access");
  return TargetModelResolvedRange{containingStart->logicalRank, addressSpace,
                                  containingStart->slotOrdinal,
                                  address - containingStart->base, byteCount};
}

llvm::Expected<InvocationMemoryRegistry>
InvocationMemoryRegistry::create(InvocationAddressPlan plan) {
  const uint64_t spmBytes = plan.spmLimit - plan.spmBase;
  if (spmBytes > std::numeric_limits<size_t>::max())
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "SPM planned window exceeds host size_t");
  std::vector<RankSPMStorage> spmStorage;
  spmStorage.reserve(plan.logicalRanks.size());
  for (int64_t logicalRank : plan.logicalRanks)
    spmStorage.push_back(
        {logicalRank, std::vector<uint8_t>(static_cast<size_t>(spmBytes), 0)});

  std::vector<SlotStorage> slotStorage;
  slotStorage.reserve(plan.initialStorage.size());
  for (const InvocationAddressPlan::InitialSlotStorage &initial :
       plan.initialStorage)
    slotStorage.push_back(
        {initial.logicalRank, initial.slotOrdinal, initial.bytes});
  return InvocationMemoryRegistry(std::move(plan), std::move(spmStorage),
                                  std::move(slotStorage));
}

InvocationMemoryRegistry::RankSPMStorage *
InvocationMemoryRegistry::findSPM(int64_t logicalRank) {
  for (RankSPMStorage &storage : spmStorage)
    if (storage.logicalRank == logicalRank)
      return &storage;
  return nullptr;
}

const InvocationMemoryRegistry::RankSPMStorage *
InvocationMemoryRegistry::findSPM(int64_t logicalRank) const {
  for (const RankSPMStorage &storage : spmStorage)
    if (storage.logicalRank == logicalRank)
      return &storage;
  return nullptr;
}

InvocationMemoryRegistry::SlotStorage *
InvocationMemoryRegistry::findSlot(int64_t logicalRank, int64_t slotOrdinal) {
  for (SlotStorage &storage : slotStorage)
    if (storage.logicalRank == logicalRank &&
        storage.slotOrdinal == slotOrdinal)
      return &storage;
  return nullptr;
}

const InvocationMemoryRegistry::SlotStorage *
InvocationMemoryRegistry::findSlot(int64_t logicalRank,
                                   int64_t slotOrdinal) const {
  for (const SlotStorage &storage : slotStorage)
    if (storage.logicalRank == logicalRank &&
        storage.slotOrdinal == slotOrdinal)
      return &storage;
  return nullptr;
}

llvm::Expected<std::vector<uint8_t>> InvocationMemoryRegistry::readSnapshot(
    int64_t logicalRank, TargetModelAddressSpace addressSpace, uint64_t address,
    uint64_t byteCount, uint64_t requiredAlignment) const {
  llvm::Expected<TargetModelResolvedRange> resolved =
      plan.resolve(logicalRank, addressSpace, TargetModelAccess::Read, address,
                   byteCount, requiredAlignment);
  if (!resolved)
    return resolved.takeError();
  if (resolved->regionOffset > std::numeric_limits<size_t>::max() ||
      resolved->byteCount > std::numeric_limits<size_t>::max())
    return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                       "resolved read range exceeds host size_t");
  const std::vector<uint8_t> *bytes = nullptr;
  if (addressSpace == TargetModelAddressSpace::RankSPM) {
    const RankSPMStorage *storage = findSPM(logicalRank);
    if (storage)
      bytes = &storage->bytes;
  } else if (resolved->slotOrdinal) {
    const SlotStorage *storage =
        findSlot(resolved->logicalRank, *resolved->slotOrdinal);
    if (storage)
      bytes = &storage->bytes;
  }
  if (!bytes)
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "resolved resource has no private backing storage");
  const size_t begin = static_cast<size_t>(resolved->regionOffset);
  const size_t size = static_cast<size_t>(resolved->byteCount);
  return std::vector<uint8_t>(bytes->begin() + begin,
                              bytes->begin() + begin + size);
}

llvm::Expected<std::vector<uint8_t>>
InvocationMemoryRegistry::readSlotSnapshot(int64_t logicalRank,
                                           int64_t slotOrdinal) const {
  const TargetModelPlannedSlot *planned =
      plan.findSlot(logicalRank, slotOrdinal);
  const SlotStorage *storage = findSlot(logicalRank, slotOrdinal);
  if (!planned || !storage)
    return memoryError(TargetModelMemoryErrorCode::UnknownResource,
                       "unknown ABI slot resource");
  return storage->bytes;
}

} // namespace wafer::model
