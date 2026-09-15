//===- ProgramElementTypeConversion.cpp - Program/target type bridge ----===//

#include "Wafer/CodeGen/ProgramElementTypeConversion.h"
#include "Wafer/Target/PhysicalTensor/TargetTensorMaterialization.h"
#include <limits>

#include "llvm/Support/ErrorHandling.h"

namespace wafer {

llvm::Expected<RawLogicalValue>
readProgramElement(ProgramElementType type, llvm::ArrayRef<uint8_t> bytes,
                   uint64_t index) {
  auto width = getProgramElementByteCount(type);
  auto format = getTargetLogicalFormat(type);
  if (!width || *width <= 0 || !format || index >= bytes.size() / *width ||
      index > std::numeric_limits<uint64_t>::max() / (uint64_t(*width) * 8))
    return llvm::createStringError(
        "program element is outside its source byte span");
  if (type == ProgramElementType::Bool) {
    if (bytes[index] > 1)
      return llvm::createStringError(
          "program Boolean source must contain canonical 0/1 bytes");
    return RawLogicalValue{LogicalFormat::Bool, bytes[index]};
  }
  return readRawLogicalValue(*format, bytes, index * *width * 8,
                             getTargetTensorScalarCodecPolicy());
}

std::optional<LogicalFormat> getTargetLogicalFormat(ProgramElementType type) {
  switch (type) {
  case ProgramElementType::Bool:
    return LogicalFormat::Bool;
  case ProgramElementType::I8:
    return LogicalFormat::I8;
  case ProgramElementType::U8:
    return LogicalFormat::U8;
  case ProgramElementType::I16:
    return LogicalFormat::I16;
  case ProgramElementType::U16:
    return LogicalFormat::U16;
  case ProgramElementType::I32:
    return LogicalFormat::I32;
  case ProgramElementType::U32:
    return LogicalFormat::U32;
  case ProgramElementType::I64:
    return LogicalFormat::I64;
  case ProgramElementType::U64:
    return LogicalFormat::U64;
  case ProgramElementType::F16:
    return LogicalFormat::F16;
  case ProgramElementType::BF16:
    return LogicalFormat::BF16;
  case ProgramElementType::F32:
    return LogicalFormat::F32;
  case ProgramElementType::TF32:
    return LogicalFormat::TF32;
  case ProgramElementType::F64:
    return std::nullopt;
  }
  llvm_unreachable("unknown program element type");
}

ProgramElementType getProgramElementType(LogicalFormat format) {
  switch (format) {
  case LogicalFormat::Bool:
    return ProgramElementType::Bool;
  case LogicalFormat::I8:
    return ProgramElementType::I8;
  case LogicalFormat::U8:
    return ProgramElementType::U8;
  case LogicalFormat::I16:
    return ProgramElementType::I16;
  case LogicalFormat::U16:
    return ProgramElementType::U16;
  case LogicalFormat::I32:
    return ProgramElementType::I32;
  case LogicalFormat::U32:
    return ProgramElementType::U32;
  case LogicalFormat::I64:
    return ProgramElementType::I64;
  case LogicalFormat::U64:
    return ProgramElementType::U64;
  case LogicalFormat::F16:
    return ProgramElementType::F16;
  case LogicalFormat::BF16:
    return ProgramElementType::BF16;
  case LogicalFormat::F32:
    return ProgramElementType::F32;
  case LogicalFormat::TF32:
    return ProgramElementType::TF32;
  }
  llvm_unreachable("unknown target logical format");
}

} // namespace wafer
