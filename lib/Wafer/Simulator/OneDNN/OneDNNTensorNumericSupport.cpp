//===- OneDNNTensorNumericSupport.cpp - OneDNN numeric shared support ----===//

#include "OneDNNTensorNumericInternal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace wafer::onednn_detail {

llvm::Error onednnError(OneDNNTensorNumericErrorCode code,
                        const llvm::Twine &detail) {
  return llvm::make_error<OneDNNTensorNumericError>(code, detail.str());
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

} // namespace wafer::onednn_detail

namespace wafer {

using namespace onednn_detail;

llvm::StringRef stringifyOneDNNQualificationKind(OneDNNQualificationKind kind) {
  switch (kind) {
  case OneDNNQualificationKind::BitExact:
    return "bit-exact";
  case OneDNNQualificationKind::ProfileBounded:
    return "profile-bounded";
  }
  llvm_unreachable("onednn qualification kind is not registered");
}

llvm::StringRef
stringifyOneDNNTensorNumericErrorCode(OneDNNTensorNumericErrorCode code) {
  switch (code) {
  case OneDNNTensorNumericErrorCode::UnsupportedOperation:
    return "unsupported-operation";
  case OneDNNTensorNumericErrorCode::UnsupportedFormat:
    return "unsupported-format";
  case OneDNNTensorNumericErrorCode::InvalidPhysicalLayout:
    return "invalid-physical-layout";
  case OneDNNTensorNumericErrorCode::InvalidPhysicalStorage:
    return "invalid-physical-storage";
  case OneDNNTensorNumericErrorCode::InvalidInputEncoding:
    return "invalid-input-encoding";
  case OneDNNTensorNumericErrorCode::InputArityMismatch:
    return "input-arity-mismatch";
  case OneDNNTensorNumericErrorCode::QualificationMismatch:
    return "qualification-mismatch";
  case OneDNNTensorNumericErrorCode::EnvironmentMismatch:
    return "environment-mismatch";
  case OneDNNTensorNumericErrorCode::WorkCountOverflow:
    return "work-count-overflow";
  case OneDNNTensorNumericErrorCode::TotalByteBudgetExceeded:
    return "total-byte-budget-exceeded";
  case OneDNNTensorNumericErrorCode::ScratchpadBudgetExceeded:
    return "scratchpad-budget-exceeded";
  case OneDNNTensorNumericErrorCode::ReorderBudgetExceeded:
    return "reorder-budget-exceeded";
  case OneDNNTensorNumericErrorCode::BackendConfigurationFailure:
    return "backend-configuration-failure";
  case OneDNNTensorNumericErrorCode::BackendDescriptorFailure:
    return "backend-descriptor-failure";
  case OneDNNTensorNumericErrorCode::BackendExecutionFailure:
    return "backend-execution-failure";
  case OneDNNTensorNumericErrorCode::BackendOutputMismatch:
    return "backend-output-mismatch";
  }
  llvm_unreachable("onednn tensor numeric error code is not registered");
}

char OneDNNTensorNumericError::ID;

void OneDNNTensorNumericError::log(llvm::raw_ostream &stream) const {
  stream << "onednn tensor numeric "
         << stringifyOneDNNTensorNumericErrorCode(code) << ": " << detail;
}

std::error_code OneDNNTensorNumericError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::StringRef getOneDNNAdapterContractDigest() {
  static const std::string digest = [] {
    llvm::SmallString<512> payload;
    llvm::raw_svector_ostream stream(payload);
    appendField(stream, "schema", "wafer-onednn-adapter");
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
