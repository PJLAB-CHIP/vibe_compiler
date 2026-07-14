//===- BulkTensorNumericInternal.h - Qualification-only bulk seam --------===//

#ifndef WAFER_TARGET_BULKTENSORNUMERICINTERNAL_H
#define WAFER_TARGET_BULKTENSORNUMERICINTERNAL_H

#include "Wafer/Target/BulkTensorNumeric.h"

namespace wafer::detail {

struct UnqualifiedBulkExecutionResult {
  BulkTensorStorage destination;
  BulkDispatchEvidence evidence;
};

llvm::Expected<UnqualifiedBulkExecutionResult>
executeBulkTensorForQualification(const BulkExecutionEnvironment &environment,
                                  const ResolvedNumericCommand &command,
                                  llvm::ArrayRef<BulkTensorStorage> inputs,
                                  const BulkTensorStorage &destinationTemplate,
                                  BulkNumericWorkBudget budget);

} // namespace wafer::detail

#endif // WAFER_TARGET_BULKTENSORNUMERICINTERNAL_H
