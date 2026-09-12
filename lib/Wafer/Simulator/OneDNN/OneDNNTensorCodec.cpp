//===- OneDNNTensorCodec.cpp - OneDNN physical codec and digest ----------===//

#include "OneDNNTensorNumericInternal.h"

#include "Wafer/Target/PhysicalTensor/PhysicalTensorCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <utility>
#include <vector>

namespace wafer::onednn_detail {

std::string descriptorDigest(const PhysicalTensorDescriptor &descriptor) {
  llvm::SmallString<256> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "format", stringifyLogicalFormat(descriptor.getFormat()));
  appendField(stream, "layout",
              stringifyPhysicalTensorLayout(descriptor.getLayout()));
  appendField(stream, "rank", descriptor.getShape().size());
  for (auto [index, extent] : llvm::enumerate(descriptor.getShape()))
    appendField(stream, (llvm::Twine("dim_") + llvm::Twine(index)).str(),
                extent);
  return sha256(payload);
}

llvm::Expected<OneDNNTensorStorage>
packIntoTemplate(const PhysicalTensorDescriptor &key,
                 llvm::ArrayRef<RawLogicalValue> values,
                 std::vector<uint8_t> storage) {
  llvm::Expected<std::vector<uint8_t>> packed =
      packPhysicalTensorLogicalValues(key, values, storage);
  if (!packed)
    return onednnError(OneDNNTensorNumericErrorCode::InvalidInputEncoding,
                       llvm::toString(packed.takeError()));
  return OneDNNTensorStorage::create(key, std::move(*packed));
}

} // namespace wafer::onednn_detail

namespace wafer {

using namespace onednn_detail;

llvm::Expected<OneDNNTensorStorage>
OneDNNTensorStorage::create(PhysicalTensorDescriptor key,
                            std::vector<uint8_t> storage) {
  llvm::Expected<uint64_t> bytes = getOneDNNTensorPhysicalBytes(key);
  if (!bytes)
    return bytes.takeError();
  if (*bytes != storage.size())
    return onednnError(OneDNNTensorNumericErrorCode::InvalidPhysicalStorage,
                       llvm::Twine("tensor requires ") + llvm::Twine(*bytes) +
                           " physical bytes, got " +
                           llvm::Twine(storage.size()));
  return OneDNNTensorStorage(std::move(key), std::move(storage));
}

llvm::Expected<uint64_t>
getOneDNNTensorPhysicalBytes(const PhysicalTensorDescriptor &key) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return onednnError(OneDNNTensorNumericErrorCode::InvalidPhysicalLayout,
                       llvm::toString(bytes.takeError()));
  return *bytes;
}

llvm::Expected<std::vector<RawLogicalValue>>
unpackOneDNNTensorLogicalValues(const OneDNNTensorStorage &tensor) {
  llvm::Expected<std::vector<RawLogicalValue>> values =
      unpackPhysicalTensorLogicalValues(tensor.getKey(), tensor.getStorage());
  if (!values)
    return onednnError(OneDNNTensorNumericErrorCode::InvalidInputEncoding,
                       llvm::toString(values.takeError()));
  return values;
}

llvm::Expected<OneDNNTensorStorage>
packOneDNNTensorLogicalValues(const PhysicalTensorDescriptor &key,
                              llvm::ArrayRef<RawLogicalValue> values,
                              uint8_t paddingFill) {
  llvm::Expected<uint64_t> bytes = getOneDNNTensorPhysicalBytes(key);
  if (!bytes)
    return bytes.takeError();
  if (*bytes > std::numeric_limits<size_t>::max())
    return onednnError(OneDNNTensorNumericErrorCode::WorkCountOverflow,
                       "physical tensor footprint exceeds host size_t");
  return packIntoTemplate(
      key, values,
      std::vector<uint8_t>(static_cast<size_t>(*bytes), paddingFill));
}

std::string
computeOneDNNTensorPayloadDigest(llvm::ArrayRef<OneDNNTensorStorage> tensors) {
  llvm::SmallString<1024> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-onednn-payload");
  appendField(stream, "tensor_count", tensors.size());
  for (auto [index, tensor] : llvm::enumerate(tensors)) {
    appendField(stream,
                (llvm::Twine("tensor_") + llvm::Twine(index) + "_key").str(),
                descriptorDigest(tensor.getKey()));
    appendField(stream,
                (llvm::Twine("tensor_") + llvm::Twine(index) + "_bytes").str(),
                sha256(tensor.getStorage()));
  }
  return sha256(payload);
}

std::string
computeOneDNNTensorStorageDigest(const OneDNNTensorStorage &tensor) {
  llvm::SmallString<512> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-onednn-storage");
  appendField(stream, "key", descriptorDigest(tensor.getKey()));
  appendField(stream, "bytes", sha256(tensor.getStorage()));
  return sha256(payload);
}

llvm::Expected<std::string>
computeOneDNNGemmProblemDigest(const FormalGemmOperation &operation) {
  if (operation.psum)
    return onednnError(OneDNNTensorNumericErrorCode::UnsupportedOperation,
                       "oneDNN GEMM identity has no psum qualification");
  const FormalGemmOperation *gemm = &operation;

  llvm::SmallString<1024> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-onednn-gemm-problem");
  appendField(stream, "lhs", descriptorDigest(gemm->lhs));
  appendField(stream, "rhs", descriptorDigest(gemm->rhs));
  appendField(stream, "destination", descriptorDigest(gemm->destination));
  appendField(stream, "m", gemm->m);
  appendField(stream, "k", gemm->k);
  appendField(stream, "n", gemm->n);
  appendField(stream, "batch", gemm->batchCount);
  appendField(stream, "lhs_orientation",
              stringifyTargetGemmOrientation(gemm->lhsOrientation));
  appendField(stream, "rhs_orientation",
              stringifyTargetGemmOrientation(gemm->rhsOrientation));
  auto appendDimensions = [&](llvm::StringRef name,
                              llvm::ArrayRef<uint64_t> dimensions) {
    appendField(stream, (llvm::Twine(name) + "_count").str(),
                dimensions.size());
    for (auto [index, dimension] : llvm::enumerate(dimensions))
      appendField(stream, (llvm::Twine(name) + "_" + llvm::Twine(index)).str(),
                  dimension);
  };
  appendDimensions("lhs_batch", gemm->axes.lhsBatchDimensions);
  appendField(stream, "lhs_m", gemm->axes.lhsMDimension);
  appendField(stream, "lhs_k", gemm->axes.lhsContractingDimension);
  appendDimensions("rhs_batch", gemm->axes.rhsBatchDimensions);
  appendField(stream, "rhs_k", gemm->axes.rhsContractingDimension);
  appendField(stream, "rhs_n", gemm->axes.rhsNDimension);
  appendDimensions("destination_batch", gemm->axes.destinationBatchDimensions);
  appendField(stream, "destination_m", gemm->axes.destinationMDimension);
  appendField(stream, "destination_n", gemm->axes.destinationNDimension);
  return sha256(payload);
}

} // namespace wafer
