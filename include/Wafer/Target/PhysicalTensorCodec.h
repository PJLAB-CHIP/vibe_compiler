//===- PhysicalTensorCodec.h - Wafer physical tensor codec ----*- C++ -*-===//

#ifndef WAFER_TARGET_PHYSICALTENSORCODEC_H
#define WAFER_TARGET_PHYSICALTENSORCODEC_H

#include "Wafer/Target/NumericSemantics.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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

} // namespace wafer

#endif // WAFER_TARGET_PHYSICALTENSORCODEC_H
