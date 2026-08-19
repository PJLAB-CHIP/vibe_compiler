//===- NumericSemanticsInternal.h - Numeric internals ----------*- C++ -*-===//

#ifndef WAFER_LIB_TARGET_NUMERICSEMANTICSINTERNAL_H
#define WAFER_LIB_TARGET_NUMERICSEMANTICSINTERNAL_H

#include "Wafer/Target/Numeric/NumericSemantics.h"

namespace wafer::numeric_semantics_internal {

std::string digestCanonical(llvm::StringRef canonical);
bool isValidDigest(llvm::StringRef digest);

bool isMPFRElementwiseOperation(NumericElementwiseOperation operation);
std::optional<NumericModelImplementationReason>
getElementwiseUnsupportedReason(NumericElementwiseOperation operation,
                                LogicalFormat inputFormat);
LogicalFormat
getElementwiseDestinationFormat(NumericElementwiseOperation operation,
                                LogicalFormat inputFormat);

bool isFloating(LogicalFormat format);
bool isInteger(LogicalFormat format);

llvm::ArrayRef<LogicalFormat>
getCompilerNumericFormats(TargetFormatEngine engine);

} // namespace wafer::numeric_semantics_internal

#endif // WAFER_LIB_TARGET_NUMERICSEMANTICSINTERNAL_H
