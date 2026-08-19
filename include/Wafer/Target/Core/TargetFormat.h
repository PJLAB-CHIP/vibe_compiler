//===- TargetFormat.h - Closed target format registry -----------*- C++ -*-===//

#ifndef WAFER_TARGET_TARGETFORMAT_H
#define WAFER_TARGET_TARGETFORMAT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace wafer {

/// Closed logical storage domain known to the compiler. Enum ordinals have no
/// target ABI meaning; target code evidence is owned by this registry.
enum class LogicalFormat : uint8_t {
  I8,
  I16,
  F16,
  BF16,
  I32,
  F32,
  TF32,
  Bool,
  U8,
  U16,
  U32,
  I64,
  U64,
};

/// Numeric category of a logical storage format. Floating-point sign bits are
/// described by the floating category rather than by integer signedness.
enum class LogicalFormatCategory : uint8_t {
  SignedInteger,
  UnsignedInteger,
  BinaryFloatingPoint,
  Boolean,
};

/// Target-independent storage facts shared by target legality and future
/// numeric consumers. Public Data_Format codes belong to the current target's
/// TargetDataFormatCodeRecord table; permission to emit a format-bearing
/// command belongs to an engine-specific TargetFormatEncodingRecord. Neither
/// target fact belongs to this logical descriptor.
struct LogicalFormatDescriptor {
  LogicalFormat format;
  llvm::StringLiteral canonicalSpelling;
  uint8_t storageBits;
  uint8_t semanticBits;
  LogicalFormatCategory category;
  /// IEEE exponent width for floating-point formats, otherwise zero.
  uint8_t exponentBits;
  /// IEEE precision including the implicit leading bit for normal values,
  /// otherwise zero.
  uint8_t precisionBits;
  /// Bits that may be nonzero in a canonical scalar storage encoding.
  uint64_t canonicalMask;
  bool bitpacked;
  bool hasInfinity;
  bool hasNaN;
  bool hasSubnormal;
};

llvm::ArrayRef<LogicalFormatDescriptor> getLogicalFormatDescriptors();
const LogicalFormatDescriptor *
findLogicalFormatDescriptor(LogicalFormat format);
llvm::Expected<LogicalFormat>
parseLogicalFormat(llvm::StringRef canonicalSpelling);
llvm::StringRef stringifyLogicalFormat(LogicalFormat format);

/// Current-target evidence for the public target Data_Format enum. This table
/// is complete even when no engine is legal to emit a format. Consumers must
/// not treat this record as engine legality; target lowering obtains a usable
/// code only from a supported TargetFormatEncodingRecord below.
struct TargetDataFormatCodeRecord {
  LogicalFormat format;
  uint8_t dataFormatCode;
};

llvm::ArrayRef<TargetDataFormatCodeRecord> getTargetDataFormatCodeRecords();
const TargetDataFormatCodeRecord *
findTargetDataFormatCode(LogicalFormat format);

/// Engines whose format-bearing commands use Data_Format. Byte-counted raw
/// movement and DTE transport are outside this enum.
enum class TargetFormatEngine : uint8_t { RDMA, WDMA, TDMA, CT, NE };

/// Operand orientation encoded by the target GEMM call ABI. MLIR lowering
/// maps its dialect enum to this target protocol type explicitly.
enum class TargetGemmOrientation : uint8_t { Normal = 0, Transpose = 1 };
llvm::StringRef stringifyTargetGemmOrientation(TargetGemmOrientation value);

llvm::ArrayRef<TargetFormatEngine> getTargetFormatEngines();
llvm::Expected<TargetFormatEngine>
parseTargetFormatEngine(llvm::StringRef canonicalSpelling);
llvm::StringRef stringifyTargetFormatEngine(TargetFormatEngine engine);

/// Additional op/layout gates required after an engine-format row is selected.
enum class TargetFormatConstraint : uint8_t {
  None,
  BitpackedLayout,
  BitpackedDMA,
};

struct TargetFormatEncodingRecord {
  TargetFormatEngine engine;
  LogicalFormat format;
  TargetFormatConstraint constraint;
  /// ABI/register code for this exact engine x logical-format row.
  uint8_t dataFormatCode;
};

/// Enumerates every current engine x logical-format row.
llvm::ArrayRef<TargetFormatEncodingRecord> getTargetFormatEncodingRecords();
const TargetFormatEncodingRecord *
findTargetFormatEncoding(TargetFormatEngine engine, LogicalFormat format);
/// Decodes one engine-specific ABI format field. Enum codes that are not
/// uniquely admitted by the registry fail closed.
llvm::Expected<LogicalFormat> decodeTargetFormat(TargetFormatEngine engine,
                                                 uint32_t dataFormatCode);
llvm::StringRef
stringifyTargetFormatConstraint(TargetFormatConstraint constraint);

/// Parameter shape of an opcode-specific CT convert wrapper. This remains a
/// target-library type so the common registry does not depend on Wafer IR.
enum class TargetConvertParameterKind : uint8_t {
  None,
  RoundingMode,
  ZeroPoint,
};

/// One exact CT convert command. These routes are independent of the generic
/// CT x format matrix: a route does not open either endpoint as a generic CT
/// command format.
struct TargetConvertRoute {
  uint16_t opcode;
  llvm::StringLiteral canonicalSpelling;
  LogicalFormat source;
  LogicalFormat destination;
  TargetConvertParameterKind parameterKind;
};

llvm::ArrayRef<TargetConvertRoute> getTargetConvertRoutes();
const TargetConvertRoute *findTargetConvertRoute(uint16_t opcode);
const TargetConvertRoute *findTargetConvertRoute(LogicalFormat source,
                                                 LogicalFormat destination);
const TargetConvertRoute *
findTargetConvertRoute(llvm::StringRef canonicalSpelling);
llvm::StringRef
stringifyTargetConvertParameterKind(TargetConvertParameterKind kind);

} // namespace wafer

#endif // WAFER_TARGET_TARGETFORMAT_H
