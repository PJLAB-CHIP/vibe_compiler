//===- TargetModelMemory.h - Private target model memory ------*- C++ -*-===//

#ifndef WAFER_MODEL_TARGETMODELMEMORY_H
#define WAFER_MODEL_TARGETMODELMEMORY_H

#include "Wafer/Compiler/TargetCallFrontend.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer::model {

enum class TargetModelAddressSpace : uint8_t { TileSPM, CardDDR };
enum class TargetModelAccess : uint8_t { Read, Write, ReadWrite };

enum class TargetModelMemoryErrorCode : uint8_t {
  InvalidInvocation,
  InvalidSlot,
  InvalidInputBinding,
  AddressOverflow,
  AddressMisaligned,
  ReservedSPM,
  UnknownResource,
  CrossResource,
  AccessDenied,
  InvalidEffect,
};

llvm::StringRef
stringifyTargetModelMemoryErrorCode(TargetModelMemoryErrorCode code);

class TargetModelMemoryError final
    : public llvm::ErrorInfo<TargetModelMemoryError> {
public:
  static char ID;

  TargetModelMemoryError(TargetModelMemoryErrorCode code, std::string detail)
      : code(code), detail(std::move(detail)) {}

  TargetModelMemoryErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  TargetModelMemoryErrorCode code;
  std::string detail;
};

/// Invocation-local physical allocation identity. Program-boundary resources
/// are owned by the card and therefore omit tileId. Compiler-managed
/// workspace, profile records and transport status are owned by one Tile.
/// Kind and resourceIndex are the typed entry-argument identity; names never
/// participate.
struct TargetModelResourceId {
  CardId cardId{0};
  std::optional<TileId> tileId;
  compiler::TileEntryArgumentKind kind =
      compiler::TileEntryArgumentKind::ExternalInput;
  int64_t resourceIndex = -1;

  friend bool operator==(const TargetModelResourceId &lhs,
                         const TargetModelResourceId &rhs) {
    return lhs.cardId == rhs.cardId &&
           lhs.tileId == rhs.tileId && lhs.kind == rhs.kind &&
           lhs.resourceIndex == rhs.resourceIndex;
  }
  friend bool operator!=(const TargetModelResourceId &lhs,
                         const TargetModelResourceId &rhs) {
    return !(lhs == rhs);
  }
};

/// Derives allocation identity from typed physical ownership and tile entry
/// argument facts. External input, TargetTensor and external output arguments
/// are card-owned; workspace, profile record and transport status arguments
/// are Tile-owned.
TargetModelResourceId getTargetModelResourceId(CardId cardId,
                                               TileId tileId,
                                               compiler::TileEntryArgumentKind kind,
                                               int64_t resourceIndex);

/// Initial contents for one read-only model allocation. There is exactly one
/// binding for a card-shared input resource, regardless of how many Tile ABI
/// slots reference it. Bytes never alias caller storage.
struct TargetModelInputBinding {
  TargetModelResourceId resource;
  std::vector<uint8_t> bytes;
};

struct TargetModelPlannedSlot {
  int64_t launchSlot = -1;
  int64_t slotOrdinal = -1;
  compiler::TileEntryArgumentKind kind =
      compiler::TileEntryArgumentKind::ExternalInput;
  int64_t resourceIndex = -1;
  TargetModelResourceId resource;
  uint64_t base = 0;
  uint64_t byteSize = 0;
  uint64_t alignment = 0;
};

/// A checked range identity. regionOffset is relative either to the private
/// per-Tile SPM planned window or to one exact DDR ABI slot.
struct TargetModelResolvedRange {
  int64_t launchSlot = -1;
  TargetModelAddressSpace addressSpace = TargetModelAddressSpace::TileSPM;
  std::optional<int64_t> slotOrdinal;
  std::optional<TargetModelResourceId> resource;
  uint64_t regionOffset = 0;
  uint64_t byteCount = 0;
};

/// Complete immutable address/resource plan for one target-call invocation.
/// The plan owns initial bytes but never aliases caller storage.
class InvocationAddressPlan {
public:
  InvocationAddressPlan(InvocationAddressPlan &&) = default;
  InvocationAddressPlan &operator=(InvocationAddressPlan &&) = default;
  InvocationAddressPlan(const InvocationAddressPlan &) = delete;
  InvocationAddressPlan &operator=(const InvocationAddressPlan &) = delete;

  static llvm::Expected<InvocationAddressPlan>
  create(const compiler::TargetCallInvocationDescriptor &invocation,
         llvm::ArrayRef<TargetModelInputBinding> inputBindings);

  TargetIdentityId getTargetIdentity() const { return targetIdentity; }
  llvm::ArrayRef<int64_t> getLaunchSlots() const { return launchSlots; }
  llvm::ArrayRef<TargetModelPlannedSlot> getSlots() const { return slots; }
  uint64_t getSPMBase() const { return spmBase; }
  uint64_t getSPMLimit() const { return spmLimit; }

  llvm::Expected<TargetModelResolvedRange>
  resolve(int64_t launchSlot, TargetModelAddressSpace addressSpace,
          TargetModelAccess access, uint64_t address, uint64_t byteCount,
          uint64_t requiredAlignment) const;

private:
  struct InitialResourceStorage {
    TargetModelResourceId resource;
    std::vector<uint8_t> bytes;
  };

  InvocationAddressPlan(TargetIdentityId targetIdentity,
                        std::vector<int64_t> launchSlots,
                        std::vector<TargetModelPlannedSlot> slots,
                        std::vector<InitialResourceStorage> initialStorage,
                        uint64_t spmBase, uint64_t spmLimit)
      : targetIdentity(targetIdentity), launchSlots(std::move(launchSlots)),
        slots(std::move(slots)), initialStorage(std::move(initialStorage)),
        spmBase(spmBase), spmLimit(spmLimit) {}

  const TargetModelPlannedSlot *findSlot(int64_t launchSlot,
                                         int64_t slotOrdinal) const;

  TargetIdentityId targetIdentity;
  std::vector<int64_t> launchSlots;
  std::vector<TargetModelPlannedSlot> slots;
  std::vector<InitialResourceStorage> initialStorage;
  uint64_t spmBase;
  uint64_t spmLimit;

  friend class InvocationMemoryRegistry;
};

/// One compact byte payload produced by a command kernel. A strided
/// layout maps consecutive payload segments to destination addresses without
/// expanding the command effect. Address space and alignment remain explicit.
struct TargetModelStridedByteLayout {
  uint32_t innerBytes = 0;
  std::array<uint32_t, 3> strides{};
  std::array<uint32_t, 3> iterations{};
};

/// Exact bytes observed by one plain target command before its asynchronous
/// NCC completion becomes visible. A strided layout describes the addressed
/// segments; byteCount is the compact payload size.
struct TargetModelByteRead {
  int64_t launchSlot = -1;
  TargetModelAddressSpace addressSpace = TargetModelAddressSpace::TileSPM;
  uint64_t address = 0;
  uint64_t byteCount = 0;
  std::optional<TargetModelStridedByteLayout> stridedLayout;
};

struct TargetModelByteWrite {
  int64_t launchSlot = -1;
  TargetModelAddressSpace addressSpace = TargetModelAddressSpace::TileSPM;
  uint64_t address = 0;
  uint64_t requiredAlignment = 1;
  std::vector<uint8_t> bytes;
  std::optional<TargetModelStridedByteLayout> stridedLayout;
};

/// Invocation-private bytes. No mutation API exposes a backing pointer;
/// applyAtomically validates the complete pending effect before any write.
class InvocationMemoryRegistry {
public:
  InvocationMemoryRegistry(InvocationMemoryRegistry &&) = default;
  InvocationMemoryRegistry &operator=(InvocationMemoryRegistry &&) = default;
  InvocationMemoryRegistry(const InvocationMemoryRegistry &) = delete;
  InvocationMemoryRegistry &
  operator=(const InvocationMemoryRegistry &) = delete;

  static llvm::Expected<InvocationMemoryRegistry>
  create(InvocationAddressPlan plan);

  const InvocationAddressPlan &getAddressPlan() const { return plan; }

  llvm::Expected<std::vector<uint8_t>>
  readSnapshot(int64_t launchSlot, TargetModelAddressSpace addressSpace,
               uint64_t address, uint64_t byteCount,
               uint64_t requiredAlignment) const;

  /// Returns one compact payload in descriptor iteration order. Repeated or
  /// overlapping source segments are legal and are snapshotted independently.
  llvm::Expected<std::vector<uint8_t>>
  readStridedSnapshot(int64_t launchSlot, TargetModelAddressSpace addressSpace,
                      uint64_t address,
                      const TargetModelStridedByteLayout &layout,
                      uint64_t requiredAlignment) const;

  llvm::Error
  applyAtomically(llvm::ArrayRef<TargetModelByteWrite> pendingWrites);

  llvm::Expected<std::vector<uint8_t>>
  readSlotSnapshot(int64_t launchSlot, int64_t slotOrdinal) const;

private:
  struct TileSPMStorage {
    int64_t launchSlot = -1;
    std::vector<uint8_t> bytes;
  };
  struct ResourceStorage {
    TargetModelResourceId resource;
    std::vector<uint8_t> bytes;
  };

  InvocationMemoryRegistry(InvocationAddressPlan plan,
                           std::vector<TileSPMStorage> spmStorage,
                           std::vector<ResourceStorage> resourceStorage)
      : plan(std::move(plan)), spmStorage(std::move(spmStorage)),
        resourceStorage(std::move(resourceStorage)) {}

  TileSPMStorage *findSPM(int64_t launchSlot);
  const TileSPMStorage *findSPM(int64_t launchSlot) const;
  ResourceStorage *findResource(const TargetModelResourceId &resource);
  const ResourceStorage *
  findResource(const TargetModelResourceId &resource) const;

  InvocationAddressPlan plan;
  std::vector<TileSPMStorage> spmStorage;
  std::vector<ResourceStorage> resourceStorage;
};

} // namespace wafer::model

#endif // WAFER_MODEL_TARGETMODELMEMORY_H
