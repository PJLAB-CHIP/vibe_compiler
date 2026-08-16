//===- PhysicalTensorCodec.h - Wafer physical tensor codec ----*- C++ -*-===//

#ifndef WAFER_TARGET_PHYSICALTENSORCODEC_H
#define WAFER_TARGET_PHYSICALTENSORCODEC_H

#include "Wafer/Target/NumericSemantics.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace wafer {

enum class PhysicalTensorCodecErrorCode : uint8_t {
  InvalidLayout,
  InvalidStorageSize,
  InvalidLogicalValueCount,
  InvalidLogicalEncoding,
};

llvm::StringRef
stringifyPhysicalTensorCodecErrorCode(PhysicalTensorCodecErrorCode code);

class PhysicalTensorCodecError final
    : public llvm::ErrorInfo<PhysicalTensorCodecError> {
public:
  static char ID;

  PhysicalTensorCodecError(PhysicalTensorCodecErrorCode code,
                           std::string detail)
      : code(code), detail(std::move(detail)) {}

  PhysicalTensorCodecErrorCode getCode() const { return code; }
  llvm::StringRef getDetail() const { return detail; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  PhysicalTensorCodecErrorCode code;
  std::string detail;
};

/// Uses the single Wafer physical-layout calculator. No target-call consumer
/// may reproduce Cx/NCx/tail/BOOL geometry from a format name.
llvm::Expected<uint64_t>
getPhysicalTensorStorageBytes(const NumericTensorKey &key);

/// Converts target-physical storage to logical row-major values using the
/// selected model profile's scalar decode policy.
llvm::Expected<std::vector<RawLogicalValue>>
unpackPhysicalTensorLogicalValues(const NumericTensorKey &key,
                                  llvm::ArrayRef<uint8_t> storage);

/// Writes logical row-major values into an exact physical template. Bytes not
/// owned by a logical element are preserved from the template.
llvm::Expected<std::vector<uint8_t>>
packPhysicalTensorLogicalValues(const NumericTensorKey &key,
                                llvm::ArrayRef<RawLogicalValue> values,
                                llvm::ArrayRef<uint8_t> storageTemplate);

/// Convenience for a newly allocated physical destination with one explicit
/// padding fill byte.
llvm::Expected<std::vector<uint8_t>>
packPhysicalTensorLogicalValues(const NumericTensorKey &key,
                                llvm::ArrayRef<RawLogicalValue> values,
                                uint8_t paddingFill);

/// Bounded physical-order write plan over one validated tensor key. Each
/// returned window is a disjoint contiguous byte span and names the logical
/// row-major values that belong in that span. This inverse view of the shared
/// physical geometry lets callers fetch arbitrary logical source ranges
/// without assuming that a logical C row is contiguous in Cx/NCx storage.
/// Both output bytes and element mappings are bounded by caller budgets;
/// bit-packed BOOL uses byte-aligned windows.
class PhysicalTensorWindowPlan {
public:
  struct ElementWrite {
    /// Row-major logical element index in the tensor key.
    uint64_t logicalIndex = 0;
    /// Destination bit offset relative to `WriteWindow::bytes`.
    uint64_t windowBitOffset = 0;
  };

  /// One disjoint physical span. `bytes` starts with `paddingFill`; callers
  /// overwrite only the positions named by `elements` through NumericCodec.
  struct WriteWindow {
    uint64_t physicalOffset = 0;
    std::vector<uint8_t> bytes;
    std::vector<ElementWrite> elements;
  };

  static llvm::Expected<PhysicalTensorWindowPlan>
  create(const NumericTensorKey &key);

  PhysicalTensorWindowPlan();
  PhysicalTensorWindowPlan(PhysicalTensorWindowPlan &&) noexcept;
  PhysicalTensorWindowPlan(const PhysicalTensorWindowPlan &) = delete;
  PhysicalTensorWindowPlan &
  operator=(const PhysicalTensorWindowPlan &) = delete;
  PhysicalTensorWindowPlan &operator=(PhysicalTensorWindowPlan &&) noexcept;
  ~PhysicalTensorWindowPlan();

  uint64_t getStorageBytes() const;
  uint64_t getElementCount() const;
  uint64_t getPlannedValueCount() const;
  bool done() const;

  /// Plans the next physical window. Both budgets must be positive. `maxBytes`
  /// bounds `bytes`; `maxLogicalValues` bounds the logical mappings. A
  /// non-final BOOL window needs room for at least eight physical elements so
  /// the next window remains byte-aligned. Returns an error on an
  /// exhausted/moved-from plan or invalid budget, leaving state intact.
  llvm::Expected<WriteWindow>
  takeNext(uint64_t maxBytes, uint64_t maxLogicalValues, uint8_t paddingFill);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace wafer

#endif // WAFER_TARGET_PHYSICALTENSORCODEC_H
