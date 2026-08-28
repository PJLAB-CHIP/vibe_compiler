//===- ProgramElementType.h - Closed program element types ----*- C++ -*-===//

#ifndef WAFER_FRONTEND_PROGRAMELEMENTTYPE_H
#define WAFER_FRONTEND_PROGRAMELEMENTTYPE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>

namespace wafer {

/// Logical element type after an external program format has been parsed.
/// This type has no target ABI ordinal; target storage uses LogicalFormat.
enum class ProgramElementType : uint8_t {
  Bool,
  I8,
  U8,
  I16,
  U16,
  I32,
  U32,
  I64,
  U64,
  F16,
  BF16,
  F32,
  TF32,
  F64,
};

inline llvm::StringRef stringifyProgramElementType(ProgramElementType type) {
  switch (type) {
  case ProgramElementType::Bool:
    return "bool";
  case ProgramElementType::I8:
    return "i8";
  case ProgramElementType::U8:
    return "u8";
  case ProgramElementType::I16:
    return "i16";
  case ProgramElementType::U16:
    return "u16";
  case ProgramElementType::I32:
    return "i32";
  case ProgramElementType::U32:
    return "u32";
  case ProgramElementType::I64:
    return "i64";
  case ProgramElementType::U64:
    return "u64";
  case ProgramElementType::F16:
    return "f16";
  case ProgramElementType::BF16:
    return "bf16";
  case ProgramElementType::F32:
    return "f32";
  case ProgramElementType::TF32:
    return "tf32";
  case ProgramElementType::F64:
    return "f64";
  }
  llvm_unreachable("unknown program element type");
}

inline llvm::Expected<ProgramElementType>
parseProgramElementType(llvm::StringRef spelling) {
  if (spelling == "bool" || spelling == "i1")
    return ProgramElementType::Bool;
  if (spelling == "i8")
    return ProgramElementType::I8;
  if (spelling == "u8" || spelling == "ui8")
    return ProgramElementType::U8;
  if (spelling == "i16")
    return ProgramElementType::I16;
  if (spelling == "u16" || spelling == "ui16")
    return ProgramElementType::U16;
  if (spelling == "i32")
    return ProgramElementType::I32;
  if (spelling == "u32" || spelling == "ui32")
    return ProgramElementType::U32;
  if (spelling == "i64")
    return ProgramElementType::I64;
  if (spelling == "u64" || spelling == "ui64")
    return ProgramElementType::U64;
  if (spelling == "f16")
    return ProgramElementType::F16;
  if (spelling == "bf16")
    return ProgramElementType::BF16;
  if (spelling == "f32")
    return ProgramElementType::F32;
  if (spelling == "tf32")
    return ProgramElementType::TF32;
  if (spelling == "f64")
    return ProgramElementType::F64;
  return llvm::createStringError("unknown program element type '%s'",
                                 spelling.str().c_str());
}

inline std::optional<int64_t>
getProgramElementByteCount(ProgramElementType type) {
  switch (type) {
  case ProgramElementType::Bool:
  case ProgramElementType::I8:
  case ProgramElementType::U8:
    return 1;
  case ProgramElementType::I16:
  case ProgramElementType::U16:
  case ProgramElementType::F16:
  case ProgramElementType::BF16:
    return 2;
  case ProgramElementType::I32:
  case ProgramElementType::U32:
  case ProgramElementType::F32:
  case ProgramElementType::TF32:
    return 4;
  case ProgramElementType::I64:
  case ProgramElementType::U64:
  case ProgramElementType::F64:
    return 8;
  }
  llvm_unreachable("unknown program element type");
}

inline bool isFloatingProgramElementType(ProgramElementType type) {
  return type == ProgramElementType::F16 || type == ProgramElementType::BF16 ||
         type == ProgramElementType::F32 || type == ProgramElementType::TF32 ||
         type == ProgramElementType::F64;
}

} // namespace wafer

#endif // WAFER_FRONTEND_PROGRAMELEMENTTYPE_H
