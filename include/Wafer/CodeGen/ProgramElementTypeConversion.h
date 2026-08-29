//===- ProgramElementTypeConversion.h - Program/target type bridge -*- C++ -*-===//

#ifndef WAFER_CODEGEN_PROGRAMELEMENTTYPECONVERSION_H
#define WAFER_CODEGEN_PROGRAMELEMENTTYPECONVERSION_H

#include "Wafer/Frontend/ProgramElementType.h"
#include "Wafer/Target/TargetFormat.h"

#include <optional>

namespace wafer {

/// Maps the frontend program element schema to the target logical format
/// schema at the code-generation boundary.
std::optional<LogicalFormat> getTargetLogicalFormat(ProgramElementType type);
ProgramElementType getProgramElementType(LogicalFormat format);

} // namespace wafer

#endif // WAFER_CODEGEN_PROGRAMELEMENTTYPECONVERSION_H
