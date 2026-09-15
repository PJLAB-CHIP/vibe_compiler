//===- ProgramElementTypeConversion.h - Program/target type bridge -*- C++ -*-===//

#ifndef WAFER_CODEGEN_PROGRAMELEMENTTYPECONVERSION_H
#define WAFER_CODEGEN_PROGRAMELEMENTTYPECONVERSION_H

#include "Wafer/Frontend/ProgramElementType.h"
#include "Wafer/Target/PhysicalTensor/NumericCodec.h"
#include "Wafer/Target/TargetFormat.h"

#include <optional>

namespace wafer {

/// Maps the frontend program element schema to the target logical format
/// schema at the code-generation boundary.
std::optional<LogicalFormat> getTargetLogicalFormat(ProgramElementType type);
ProgramElementType getProgramElementType(LogicalFormat format);

/// Program tensors use whole bytes per logical element (including Bool).
/// Target tensor packing is performed separately by the physical codec.
llvm::Expected<RawLogicalValue>
readProgramElement(ProgramElementType type, llvm::ArrayRef<uint8_t> bytes,
                   uint64_t index);

} // namespace wafer

#endif // WAFER_CODEGEN_PROGRAMELEMENTTYPECONVERSION_H
