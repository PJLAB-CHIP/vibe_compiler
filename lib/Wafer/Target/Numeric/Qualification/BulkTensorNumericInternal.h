//===- BulkTensorNumericInternal.h - Qualification-only bulk seam -*- C++
//-*-===//

#ifndef WAFER_TARGET_BULKTENSORNUMERICINTERNAL_H
#define WAFER_TARGET_BULKTENSORNUMERICINTERNAL_H

#include "Wafer/Target/Numeric/Qualification/BulkTensorNumeric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

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

namespace wafer::bulk_detail {

llvm::Error bulkError(BulkTensorNumericErrorCode code,
                      const llvm::Twine &detail);
std::string sha256(llvm::StringRef payload);
std::string sha256(llvm::ArrayRef<uint8_t> payload);
void appendField(llvm::raw_ostream &stream, llvm::StringRef name,
                 llvm::StringRef value);
void appendField(llvm::raw_ostream &stream, llvm::StringRef name,
                 uint64_t value);
uint32_t readMXCSR();

llvm::Expected<BulkTensorStorage>
packIntoTemplate(const NumericTensorKey &key,
                 llvm::ArrayRef<RawLogicalValue> values,
                 std::vector<uint8_t> storage);

llvm::Expected<detail::UnqualifiedBulkExecutionResult>
executeOneDNN(const BulkExecutionEnvironment &environment,
              const ResolvedNumericCommand &command,
              llvm::ArrayRef<BulkTensorStorage> inputs,
              const BulkTensorStorage &destinationTemplate,
              BulkNumericWorkBudget budget);

} // namespace wafer::bulk_detail

#endif // WAFER_TARGET_BULKTENSORNUMERICINTERNAL_H
