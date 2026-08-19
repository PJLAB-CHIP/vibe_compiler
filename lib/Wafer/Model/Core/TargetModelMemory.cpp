//===- TargetModelMemory.cpp - Private target model memory --------------===//

#include "Wafer/Model/Core/TargetModelMemory.h"

#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/Core/TargetIdentity.h"

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

bool isInputRole(compiler::TileEntryArgumentKind kind) {
  switch (kind) {
  case compiler::TileEntryArgumentKind::ExternalInput:
  case compiler::TileEntryArgumentKind::TargetTensor:
    return true;
  case compiler::TileEntryArgumentKind::ExternalOutput:
  case compiler::TileEntryArgumentKind::Workspace:
  case compiler::TileEntryArgumentKind::ProfileRecord:
  case compiler::TileEntryArgumentKind::TransportStatus:
    return false;
  }
  llvm_unreachable("unknown tile entry argument kind");
}

bool permitsRead(compiler::TileEntryArgumentKind) { return true; }

bool permitsWrite(compiler::TileEntryArgumentKind kind) {
  switch (kind) {
  case compiler::TileEntryArgumentKind::ExternalInput:
  case compiler::TileEntryArgumentKind::TargetTensor:
    return false;
  case compiler::TileEntryArgumentKind::ExternalOutput:
  case compiler::TileEntryArgumentKind::Workspace:
  case compiler::TileEntryArgumentKind::ProfileRecord:
  case compiler::TileEntryArgumentKind::TransportStatus:
    return true;
  }
  llvm_unreachable("unknown tile entry argument kind");
}

bool permitsAccess(compiler::TileEntryArgumentKind kind, TargetModelAccess access) {
  switch (access) {
  case TargetModelAccess::Read:
    return permitsRead(kind);
  case TargetModelAccess::Write:
    return permitsWrite(kind);
  case TargetModelAccess::ReadWrite:
    return permitsRead(kind) && permitsWrite(kind);
  }
  llvm_unreachable("unknown target model access");
}

std::string tileSlot(int64_t launchSlot, int64_t slotOrdinal) {
  return (llvm::Twine("launch slot ") + llvm::Twine(launchSlot) +
          " entry argument " + llvm::Twine(slotOrdinal))
      .str();
}

bool haveSameResourceGeometry(const compiler::TileEntryArgument &lhs,
                              const compiler::TileEntryArgument &rhs) {
  return lhs.kind == rhs.kind && lhs.resourceIndex == rhs.resourceIndex &&
         lhs.dtype == rhs.dtype && lhs.layout == rhs.layout &&
         lhs.shape == rhs.shape && lhs.byteSize == rhs.byteSize &&
         lhs.alignment == rhs.alignment;
}

} // namespace

TargetModelResourceId getTargetModelResourceId(CardId cardId,
                                               TileId tileId,
                                               compiler::TileEntryArgumentKind kind,
                                               int64_t resourceIndex) {
  std::optional<TileId> ownerTile;
  if (kind == compiler::TileEntryArgumentKind::Workspace ||
      kind == compiler::TileEntryArgumentKind::ProfileRecord ||
      kind == compiler::TileEntryArgumentKind::TransportStatus)
    ownerTile = tileId;
  return {cardId, ownerTile, kind, resourceIndex};
}

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
  if (invocation.tiles.empty())
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "invocation has no Tiles");

  if (invocation.targetIdentity != TargetIdentityId::waferTx81SingleCard())
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "invocation target identity is unsupported");
  const size_t tileCount = invocation.tiles.size();
  std::vector<const compiler::TargetCallTileDescriptor *> tiles(tileCount,
                                                                nullptr);
  std::set<std::pair<int64_t, int64_t>> tileEndpoints;
  for (const compiler::TargetCallTileDescriptor &tile : invocation.tiles) {
    const int64_t launchSlot = tile.launchSlotId.getValue();
    if (launchSlot < 0 || static_cast<uint64_t>(launchSlot) >= tileCount)
      return memoryError(
          TargetModelMemoryErrorCode::InvalidInvocation,
          llvm::Twine("launch-slot domain must be exactly [0, ") +
              llvm::Twine(tileCount) + ")");
    if (tiles[static_cast<size_t>(launchSlot)])
      return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                         llvm::Twine("duplicate launch slot ") +
                             llvm::Twine(launchSlot));
    if (tile.cardId != CardId(0) ||
        tile.tileId.getValue() < 0 ||
        !tileEndpoints
             .emplace(tile.cardId.getValue(),
                      tile.tileId.getValue())
             .second)
      return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                         "Tile identity is invalid or duplicate");
    if (tile.targetIdentity != invocation.targetIdentity ||
        tile.kernelRuntimeABI != KernelRuntimeABIId::waferTx81Kernel())
      return memoryError(
          TargetModelMemoryErrorCode::InvalidInvocation,
          llvm::Twine("launch slot ") + llvm::Twine(launchSlot) +
              " Tile identity or kernel runtime ABI is inconsistent");
    tiles[static_cast<size_t>(launchSlot)] = &tile;
  }
  if (llvm::is_contained(tiles, nullptr))
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "launch-slot domain is incomplete");

  for (auto [index, binding] : llvm::enumerate(inputBindings))
    for (size_t previous = 0; previous < index; ++previous)
      if (inputBindings[previous].resource == binding.resource)
        return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                           "duplicate input resource identity");

  const TargetMemoryPolicy memoryPolicy = getDefaultWaferTargetPolicy().memory;
  if (memoryPolicy.spmBase < 0 || memoryPolicy.spmLimit <= memoryPolicy.spmBase)
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "target policy has an invalid SPM planned window");
  const uint64_t spmBase = static_cast<uint64_t>(memoryPolicy.spmBase);
  const uint64_t spmLimit = static_cast<uint64_t>(memoryPolicy.spmLimit);

  std::vector<int64_t> launchSlots;
  std::vector<TargetModelPlannedSlot> slots;
  std::vector<InitialResourceStorage> initialStorage;
  struct ResourceFacts {
    TargetModelResourceId id;
    compiler::TileEntryArgument slot;
    uint64_t base = 0;
    uint64_t byteSize = 0;
    uint64_t alignment = 0;
    std::vector<int64_t> launchSlots;
  };
  std::vector<ResourceFacts> resources;
  launchSlots.reserve(tileCount);
  for (size_t launchSlotIndex = 0; launchSlotIndex < tileCount;
       ++launchSlotIndex) {
    const compiler::TargetCallTileDescriptor &tile = *tiles[launchSlotIndex];
    const int64_t launchSlot = tile.launchSlotId.getValue();
    launchSlots.push_back(launchSlot);
    if (tile.tileEntryArguments.size() != tile.slotValues.size())
      return memoryError(
          TargetModelMemoryErrorCode::InvalidInvocation,
          llvm::Twine("launch slot ") + llvm::Twine(launchSlot) +
              " ABI slot metadata and values have different lengths");
    for (size_t slotIndex = 0; slotIndex < tile.tileEntryArguments.size();
         ++slotIndex) {
      const compiler::TileEntryArgument &slot = tile.tileEntryArguments[slotIndex];
      if (slot.ordinal != static_cast<int64_t>(slotIndex) ||
          slot.resourceIndex < 0)
        return memoryError(TargetModelMemoryErrorCode::InvalidSlot,
                           tileSlot(launchSlot, slot.ordinal) +
                               " has invalid ordinal or resource index");
      if (slot.byteSize <= 0 || slot.alignment <= 0 ||
          !isPowerOfTwo(static_cast<uint64_t>(slot.alignment)))
        return memoryError(TargetModelMemoryErrorCode::InvalidSlot,
                           tileSlot(launchSlot, slot.ordinal) +
                               " has invalid size or alignment");
      const uint64_t base = tile.slotValues[slotIndex];
      const uint64_t byteSize = static_cast<uint64_t>(slot.byteSize);
      const uint64_t alignment = static_cast<uint64_t>(slot.alignment);
      uint64_t end = 0;
      if (!checkedAdd(base, byteSize, end))
        return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                           tileSlot(launchSlot, slot.ordinal) +
                               " address range overflows");
      if (base % alignment != 0)
        return memoryError(TargetModelMemoryErrorCode::AddressMisaligned,
                           tileSlot(launchSlot, slot.ordinal) +
                               " base does not satisfy ABI alignment");

      const TargetModelResourceId resource =
          getTargetModelResourceId(tile.cardId, tile.tileId,
                                   slot.kind, slot.resourceIndex);
      ResourceFacts *facts = nullptr;
      for (ResourceFacts &candidate : resources)
        if (candidate.id == resource) {
          facts = &candidate;
          break;
        }
      if (facts) {
        if (llvm::is_contained(facts->launchSlots, launchSlot))
          return memoryError(TargetModelMemoryErrorCode::InvalidSlot,
                             tileSlot(launchSlot, slot.ordinal) +
                                 " duplicates one resource in a Tile ABI");
        if (!haveSameResourceGeometry(facts->slot, slot) ||
            facts->base != base || facts->byteSize != byteSize ||
            facts->alignment != alignment)
          return memoryError(
              TargetModelMemoryErrorCode::InvalidSlot,
              tileSlot(launchSlot, slot.ordinal) +
                  " disagrees with another slot for the same resource");
        facts->launchSlots.push_back(launchSlot);
      } else {
        const TargetModelInputBinding *input = nullptr;
        for (const TargetModelInputBinding &candidate : inputBindings)
          if (candidate.resource == resource) {
            input = &candidate;
            break;
          }
        if (isInputRole(slot.kind)) {
          if (!input)
            return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                               tileSlot(launchSlot, slot.ordinal) +
                                   " requires exact initial bytes");
          if (input->bytes.size() != byteSize)
            return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                               tileSlot(launchSlot, slot.ordinal) +
                                   " initial byte count differs from ABI size");
          initialStorage.push_back({resource, input->bytes});
        } else {
          if (input)
            return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                               tileSlot(launchSlot, slot.ordinal) +
                                   " is model-owned and cannot be prebound");
          initialStorage.push_back(
              {resource,
               std::vector<uint8_t>(static_cast<size_t>(byteSize), 0)});
        }
        resources.push_back(
            {resource, slot, base, byteSize, alignment, {launchSlot}});
      }
      slots.push_back({launchSlot, slot.ordinal, slot.kind, slot.resourceIndex,
                       resource, base, byteSize, alignment});
    }
  }

  for (const TargetModelInputBinding &binding : inputBindings) {
    const auto resource = llvm::find_if(resources, [&](const ResourceFacts &r) {
      return r.id == binding.resource;
    });
    if (resource == resources.end())
      return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                         "input binding names an unknown resource");
    if (!isInputRole(resource->slot.kind))
      return memoryError(TargetModelMemoryErrorCode::InvalidInputBinding,
                         "model-owned resource cannot be prebound");
  }
  for (const ResourceFacts &resource : resources) {
    const bool cardOwned = !resource.id.tileId.has_value();
    if ((cardOwned && resource.launchSlots.size() != tileCount) ||
        (!cardOwned && resource.launchSlots.size() != 1))
      return memoryError(
          TargetModelMemoryErrorCode::InvalidSlot,
          "resource owner does not match its Tile reference domain");
  }

  std::vector<const ResourceFacts *> byBase;
  byBase.reserve(resources.size());
  for (const ResourceFacts &resource : resources)
    byBase.push_back(&resource);
  llvm::sort(byBase, [](const ResourceFacts *lhs, const ResourceFacts *rhs) {
    return std::tie(lhs->base, lhs->slot.kind, lhs->slot.resourceIndex) <
           std::tie(rhs->base, rhs->slot.kind, rhs->slot.resourceIndex);
  });
  for (size_t index = 1; index < byBase.size(); ++index) {
    uint64_t previousEnd = 0;
    (void)checkedAdd(byBase[index - 1]->base, byBase[index - 1]->byteSize,
                     previousEnd);
    if (byBase[index]->base < previousEnd)
      return memoryError(
          TargetModelMemoryErrorCode::InvalidSlot,
          "distinct target model resources have overlapping addresses");
  }

  return InvocationAddressPlan(invocation.targetIdentity,
                               std::move(launchSlots), std::move(slots),
                               std::move(initialStorage), spmBase, spmLimit);
}

const TargetModelPlannedSlot *
InvocationAddressPlan::findSlot(int64_t launchSlot, int64_t slotOrdinal) const {
  for (const TargetModelPlannedSlot &slot : slots)
    if (slot.launchSlot == launchSlot && slot.slotOrdinal == slotOrdinal)
      return &slot;
  return nullptr;
}

llvm::Expected<TargetModelResolvedRange> InvocationAddressPlan::resolve(
    int64_t launchSlot, TargetModelAddressSpace addressSpace,
    TargetModelAccess access, uint64_t address, uint64_t byteCount,
    uint64_t requiredAlignment) const {
  if (!llvm::is_contained(launchSlots, launchSlot))
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       llvm::Twine("unknown launch slot ") +
                           llvm::Twine(launchSlot));
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

  if (addressSpace == TargetModelAddressSpace::TileSPM) {
    if (address < spmBase || end > spmLimit)
      return memoryError(
          TargetModelMemoryErrorCode::ReservedSPM,
          llvm::Twine("SPM range is outside the planned tensor window [") +
              llvm::Twine(spmBase) + ", " + llvm::Twine(spmLimit) + ")");
    return TargetModelResolvedRange{launchSlot,        addressSpace,
                                    std::nullopt,      std::nullopt,
                                    address - spmBase, byteCount};
  }

  const TargetModelPlannedSlot *containingStart = nullptr;
  for (const TargetModelPlannedSlot &slot : slots) {
    if (slot.launchSlot != launchSlot)
      continue;
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
  if (!permitsAccess(containingStart->kind, access))
    return memoryError(
        TargetModelMemoryErrorCode::AccessDenied,
        tileSlot(containingStart->launchSlot, containingStart->slotOrdinal) +
            " does not permit the requested access");
  return TargetModelResolvedRange{
      containingStart->launchSlot,     addressSpace,
      containingStart->slotOrdinal,    containingStart->resource,
      address - containingStart->base, byteCount};
}

llvm::Expected<InvocationMemoryRegistry>
InvocationMemoryRegistry::create(InvocationAddressPlan plan) {
  const uint64_t spmBytes = plan.spmLimit - plan.spmBase;
  if (spmBytes > std::numeric_limits<size_t>::max())
    return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                       "SPM planned window exceeds host size_t");
  std::vector<TileSPMStorage> spmStorage;
  spmStorage.reserve(plan.launchSlots.size());
  for (int64_t launchSlot : plan.launchSlots)
    spmStorage.push_back(
        {launchSlot, std::vector<uint8_t>(static_cast<size_t>(spmBytes), 0)});

  std::vector<ResourceStorage> resourceStorage;
  resourceStorage.reserve(plan.initialStorage.size());
  for (const InvocationAddressPlan::InitialResourceStorage &initial :
       plan.initialStorage)
    resourceStorage.push_back({initial.resource, initial.bytes});
  return InvocationMemoryRegistry(std::move(plan), std::move(spmStorage),
                                  std::move(resourceStorage));
}

InvocationMemoryRegistry::TileSPMStorage *
InvocationMemoryRegistry::findSPM(int64_t launchSlot) {
  for (TileSPMStorage &storage : spmStorage)
    if (storage.launchSlot == launchSlot)
      return &storage;
  return nullptr;
}

const InvocationMemoryRegistry::TileSPMStorage *
InvocationMemoryRegistry::findSPM(int64_t launchSlot) const {
  for (const TileSPMStorage &storage : spmStorage)
    if (storage.launchSlot == launchSlot)
      return &storage;
  return nullptr;
}

InvocationMemoryRegistry::ResourceStorage *
InvocationMemoryRegistry::findResource(const TargetModelResourceId &resource) {
  for (ResourceStorage &storage : resourceStorage)
    if (storage.resource == resource)
      return &storage;
  return nullptr;
}

const InvocationMemoryRegistry::ResourceStorage *
InvocationMemoryRegistry::findResource(
    const TargetModelResourceId &resource) const {
  for (const ResourceStorage &storage : resourceStorage)
    if (storage.resource == resource)
      return &storage;
  return nullptr;
}

llvm::Expected<std::vector<uint8_t>> InvocationMemoryRegistry::readSnapshot(
    int64_t launchSlot, TargetModelAddressSpace addressSpace, uint64_t address,
    uint64_t byteCount, uint64_t requiredAlignment) const {
  llvm::Expected<TargetModelResolvedRange> resolved =
      plan.resolve(launchSlot, addressSpace, TargetModelAccess::Read, address,
                   byteCount, requiredAlignment);
  if (!resolved)
    return resolved.takeError();
  if (resolved->regionOffset > std::numeric_limits<size_t>::max() ||
      resolved->byteCount > std::numeric_limits<size_t>::max())
    return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                       "resolved read range exceeds host size_t");
  const std::vector<uint8_t> *bytes = nullptr;
  if (addressSpace == TargetModelAddressSpace::TileSPM) {
    const TileSPMStorage *storage = findSPM(launchSlot);
    if (storage)
      bytes = &storage->bytes;
  } else if (resolved->resource) {
    const ResourceStorage *storage = findResource(*resolved->resource);
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
InvocationMemoryRegistry::readSlotSnapshot(int64_t launchSlot,
                                           int64_t slotOrdinal) const {
  const TargetModelPlannedSlot *planned =
      plan.findSlot(launchSlot, slotOrdinal);
  const ResourceStorage *storage =
      planned ? findResource(planned->resource) : nullptr;
  if (!planned || !storage)
    return memoryError(TargetModelMemoryErrorCode::UnknownResource,
                       "unknown ABI slot resource");
  return storage->bytes;
}

} // namespace wafer::model
