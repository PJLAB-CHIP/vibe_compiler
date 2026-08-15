//===- BulkTensorNumericSupport.cpp - Bulk numeric shared support ----===//

#include "BulkTensorNumericInternal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace wafer::bulk_detail {

llvm::Error bulkError(BulkTensorNumericErrorCode code,
                      const llvm::Twine &detail) {
  return llvm::make_error<BulkTensorNumericError>(code, detail.str());
}

std::string sha256(llvm::StringRef payload) {
  llvm::SHA256 hasher;
  hasher.update(payload);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

std::string sha256(llvm::ArrayRef<uint8_t> payload) {
  llvm::SHA256 hasher;
  hasher.update(payload);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

void appendField(llvm::raw_ostream &stream, llvm::StringRef name,
                 llvm::StringRef value) {
  stream << name << '=' << value.size() << ':' << value << '\n';
}

void appendField(llvm::raw_ostream &stream, llvm::StringRef name,
                 uint64_t value) {
  stream << name << '=' << value << '\n';
}

} // namespace wafer::bulk_detail

namespace wafer {

using namespace bulk_detail;

llvm::StringRef stringifyBulkQualificationKind(BulkQualificationKind kind) {
  switch (kind) {
  case BulkQualificationKind::BitExact:
    return "bit-exact";
  case BulkQualificationKind::ProfileBounded:
    return "profile-bounded";
  }
  llvm_unreachable("bulk qualification kind is not registered");
}

llvm::StringRef
stringifyBulkTensorNumericErrorCode(BulkTensorNumericErrorCode code) {
  switch (code) {
  case BulkTensorNumericErrorCode::UnsupportedResolvedCommand:
    return "unsupported-resolved-command";
  case BulkTensorNumericErrorCode::UnsupportedFormat:
    return "unsupported-format";
  case BulkTensorNumericErrorCode::InvalidPhysicalLayout:
    return "invalid-physical-layout";
  case BulkTensorNumericErrorCode::InvalidPhysicalStorage:
    return "invalid-physical-storage";
  case BulkTensorNumericErrorCode::InvalidInputEncoding:
    return "invalid-input-encoding";
  case BulkTensorNumericErrorCode::InputArityMismatch:
    return "input-arity-mismatch";
  case BulkTensorNumericErrorCode::QualificationMismatch:
    return "qualification-mismatch";
  case BulkTensorNumericErrorCode::EnvironmentMismatch:
    return "environment-mismatch";
  case BulkTensorNumericErrorCode::WorkCountOverflow:
    return "work-count-overflow";
  case BulkTensorNumericErrorCode::TotalByteBudgetExceeded:
    return "total-byte-budget-exceeded";
  case BulkTensorNumericErrorCode::ScratchpadBudgetExceeded:
    return "scratchpad-budget-exceeded";
  case BulkTensorNumericErrorCode::ReorderBudgetExceeded:
    return "reorder-budget-exceeded";
  case BulkTensorNumericErrorCode::BackendConfigurationFailure:
    return "backend-configuration-failure";
  case BulkTensorNumericErrorCode::BackendDescriptorFailure:
    return "backend-descriptor-failure";
  case BulkTensorNumericErrorCode::BackendExecutionFailure:
    return "backend-execution-failure";
  case BulkTensorNumericErrorCode::BackendOutputMismatch:
    return "backend-output-mismatch";
  }
  llvm_unreachable("bulk tensor numeric error code is not registered");
}

char BulkTensorNumericError::ID;

void BulkTensorNumericError::log(llvm::raw_ostream &stream) const {
  stream << "bulk tensor numeric " << stringifyBulkTensorNumericErrorCode(code)
         << ": " << detail;
}

std::error_code BulkTensorNumericError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::StringRef getBulkAdapterContractDigest() {
  static const std::string digest = [] {
    llvm::SmallString<512> payload;
    llvm::raw_svector_ostream stream(payload);
    appendField(stream, "schema", "wafer-bulk-adapter-v1");
    appendField(stream, "input", "target-owned-physical-codec-layout");
    appendField(stream, "backend_dense_format", "f32");
    appendField(stream, "backend_primitive",
                "one-matmul-optional-weight-reorder");
    appendField(stream, "destination", "formal-gemm-finalize-and-target-pack");
    appendField(stream, "result_write", "copy-after-success");
    return sha256(payload);
  }();
  return digest;
}

} // namespace wafer
