//===- OneDNNTensorNumericInternal.h - Qualification-only onednn seam -*- C++
//-*-===//

#ifndef WAFER_TARGET_ONEDNNTENSORNUMERICINTERNAL_H
#define WAFER_TARGET_ONEDNNTENSORNUMERICINTERNAL_H

#include "Wafer/Target/Numeric/Qualification/OneDNNTensorNumeric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace wafer::detail {

struct UnqualifiedOneDNNExecutionResult {
  OneDNNTensorStorage destination;
  OneDNNDispatchEvidence evidence;
};

llvm::Expected<UnqualifiedOneDNNExecutionResult>
executeOneDNNTensorForQualification(
    const OneDNNExecutionEnvironment &environment,
    const FormalGemmOperation &operation,
    llvm::ArrayRef<OneDNNTensorStorage> inputs,
    const OneDNNTensorStorage &destinationTemplate,
    OneDNNNumericWorkBudget budget);

} // namespace wafer::detail

namespace wafer::onednn_detail {

llvm::Error onednnError(OneDNNTensorNumericErrorCode code,
                        const llvm::Twine &detail);
std::string sha256(llvm::StringRef payload);
std::string sha256(llvm::ArrayRef<uint8_t> payload);
void appendField(llvm::raw_ostream &stream, llvm::StringRef name,
                 llvm::StringRef value);
void appendField(llvm::raw_ostream &stream, llvm::StringRef name,
                 uint64_t value);
uint32_t readMXCSR();

llvm::Expected<OneDNNTensorStorage>
packIntoTemplate(const PhysicalTensorDescriptor &key,
                 llvm::ArrayRef<RawLogicalValue> values,
                 std::vector<uint8_t> storage);

llvm::Expected<detail::UnqualifiedOneDNNExecutionResult>
executeOneDNN(const OneDNNExecutionEnvironment &environment,
              const FormalGemmOperation &operation,
              llvm::ArrayRef<OneDNNTensorStorage> inputs,
              const OneDNNTensorStorage &destinationTemplate,
              OneDNNNumericWorkBudget budget);

} // namespace wafer::onednn_detail

#endif // WAFER_TARGET_ONEDNNTENSORNUMERICINTERNAL_H
