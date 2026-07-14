//===- NumericCodec.h - Target-independent logical scalar codec -*- C++ -*-===//

#ifndef WAFER_TARGET_NUMERICCODEC_H
#define WAFER_TARGET_NUMERICCODEC_H

#include "Wafer/Target/TargetFormat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace wafer {

/// Policy for storage bits that are outside a format's semantic encoding.
/// Callers must choose explicitly; target numeric execution uses Reject while
/// explicit normalization utilities may use ClearUnusedBits.
enum class NonCanonicalEncodingPolicy : uint8_t {
  Reject,
  ClearUnusedBits,
};

enum class LogicalByteOrder : uint8_t {
  LittleEndian,
  BigEndian,
};

enum class LogicalBitOrder : uint8_t {
  LeastSignificantBitFirstWithinByte,
  MostSignificantBitFirstWithinByte,
};

/// Complete scalar storage policy selected by a model profile. This is kept
/// outside LogicalFormatDescriptor because byte order, BOOL bit order, and
/// noncanonical-input handling are model/layout policy rather than properties
/// of the abstract numeric format.
struct LogicalScalarCodecPolicy {
  LogicalByteOrder byteOrder;
  LogicalBitOrder bitOrder;
  NonCanonicalEncodingPolicy nonCanonicalEncoding;

  friend constexpr bool operator==(LogicalScalarCodecPolicy lhs,
                                   LogicalScalarCodecPolicy rhs) {
    return lhs.byteOrder == rhs.byteOrder && lhs.bitOrder == rhs.bitOrder &&
           lhs.nonCanonicalEncoding == rhs.nonCanonicalEncoding;
  }
  friend constexpr bool operator!=(LogicalScalarCodecPolicy lhs,
                                   LogicalScalarCodecPolicy rhs) {
    return !(lhs == rhs);
  }
};

/// A scalar logical value carried as raw storage bits. This object does not
/// imply a host native representation and does not carry physical tensor
/// layout. Use makeRawLogicalValue before publishing an instance.
struct RawLogicalValue {
  LogicalFormat format;
  uint64_t bits;
};

enum class LogicalValueClass : uint8_t {
  SignedInteger,
  UnsignedInteger,
  Boolean,
  Zero,
  Subnormal,
  Normal,
  Infinity,
  QuietNaN,
  SignalingNaN,
};

struct LogicalValueClassification {
  LogicalValueClass valueClass;
  bool negative;
};

/// Validates raw bits and applies the explicitly selected canonicalization
/// policy. In particular, numeric TF32 rejects or clears nonzero low 13 bits.
llvm::Expected<RawLogicalValue>
makeRawLogicalValue(LogicalFormat format, uint64_t bits,
                    NonCanonicalEncodingPolicy policy);

/// Reads/writes one scalar using the caller-selected byte and bit order. Only
/// BOOL is bit-addressable; every other format requires byte-aligned bitOffset.
/// Physical Cx/NCx/tail geometry remains owned by the layout helper.
llvm::Expected<RawLogicalValue>
readRawLogicalValue(LogicalFormat format, llvm::ArrayRef<uint8_t> storage,
                    uint64_t bitOffset, LogicalScalarCodecPolicy policy);
llvm::Error writeRawLogicalValue(RawLogicalValue value,
                                 llvm::MutableArrayRef<uint8_t> storage,
                                 uint64_t bitOffset,
                                 LogicalScalarCodecPolicy policy);

llvm::Expected<LogicalValueClassification>
classifyRawLogicalValue(RawLogicalValue value,
                        NonCanonicalEncodingPolicy policy);

} // namespace wafer

#endif // WAFER_TARGET_NUMERICCODEC_H
