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

/// Bounded-window physical packer over one validated tensor key. Logical
/// values are consumed in row-major order and packed into contiguous physical
/// byte spans using the same shared layout calculator as the full-buffer
/// codec; padding bytes are written as `paddingFill`. Memory never scales with
/// the full tensor: spans are capped by the caller's byte budget, with one
/// single-channel-row fallback for blocked layouts whose row exceeds the
/// budget. Bit-packed elements (single-bit storage) are rejected.
class PhysicalTensorWindowPacker {
public:
  /// One contiguous physical span produced from a leading window of the
  /// supplied logical values. `physicalOffset` is a byte offset relative to
  /// the start of the physical storage; it is always byte-aligned.
  struct WriteWindow {
    uint64_t physicalOffset = 0;
    /// Number of leading input values this window consumed.
    uint64_t valueCount = 0;
    /// The packed span, zero-padded, of size <= the requested byte budget
    /// (except the single-row fallback described above).
    std::vector<uint8_t> bytes;
  };

  static llvm::Expected<PhysicalTensorWindowPacker>
  create(const NumericTensorKey &key);

  PhysicalTensorWindowPacker();
  PhysicalTensorWindowPacker(PhysicalTensorWindowPacker &&) noexcept;
  PhysicalTensorWindowPacker(const PhysicalTensorWindowPacker &) = delete;
  PhysicalTensorWindowPacker &
  operator=(const PhysicalTensorWindowPacker &) = delete;
  PhysicalTensorWindowPacker &
  operator=(PhysicalTensorWindowPacker &&) noexcept;
  ~PhysicalTensorWindowPacker();

  uint64_t getStorageBytes() const;
  uint64_t getElementCount() const;
  /// Number of logical values already packed, in row-major order.
  uint64_t getPackedValueCount() const;

  /// Packs the next window. `values` are row-major logical values continuing
  /// after the already packed prefix; every consumed value must be in the
  /// key's format. Returns an error on format mismatch, an exhausted packer,
  /// or geometry failure; the packer state is unchanged on error.
  llvm::Expected<WriteWindow>
  packNext(llvm::ArrayRef<RawLogicalValue> values, uint64_t budgetBytes,
           uint8_t paddingFill);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace wafer

#endif // WAFER_TARGET_PHYSICALTENSORCODEC_H
