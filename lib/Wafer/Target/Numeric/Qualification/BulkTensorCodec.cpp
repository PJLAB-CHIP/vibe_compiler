//===- BulkTensorCodec.cpp - Bulk physical codec and digest ----------===//

#include "BulkTensorNumericInternal.h"

#include "Wafer/Target/Layout/PhysicalTensorCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <utility>
#include <vector>

namespace wafer::bulk_detail {

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

llvm::Expected<BulkTensorStorage>
packIntoTemplate(const PhysicalTensorDescriptor &key,
                 llvm::ArrayRef<RawLogicalValue> values,
                 std::vector<uint8_t> storage) {
  llvm::Expected<std::vector<uint8_t>> packed =
      packPhysicalTensorLogicalValues(key, values, storage);
  if (!packed)
    return bulkError(BulkTensorNumericErrorCode::InvalidInputEncoding,
                     llvm::toString(packed.takeError()));
  return BulkTensorStorage::create(key, std::move(*packed));
}

} // namespace wafer::bulk_detail

namespace wafer {

using namespace bulk_detail;

llvm::Expected<BulkTensorStorage>
BulkTensorStorage::create(PhysicalTensorDescriptor key,
                          std::vector<uint8_t> storage) {
  llvm::Expected<uint64_t> bytes = getBulkTensorPhysicalBytes(key);
  if (!bytes)
    return bytes.takeError();
  if (*bytes != storage.size())
    return bulkError(BulkTensorNumericErrorCode::InvalidPhysicalStorage,
                     llvm::Twine("tensor requires ") + llvm::Twine(*bytes) +
                         " physical bytes, got " + llvm::Twine(storage.size()));
  return BulkTensorStorage(std::move(key), std::move(storage));
}

llvm::Expected<uint64_t>
getBulkTensorPhysicalBytes(const PhysicalTensorDescriptor &key) {
  llvm::Expected<uint64_t> bytes = getPhysicalTensorStorageBytes(key);
  if (!bytes)
    return bulkError(BulkTensorNumericErrorCode::InvalidPhysicalLayout,
                     llvm::toString(bytes.takeError()));
  return *bytes;
}

llvm::Expected<std::vector<RawLogicalValue>>
unpackBulkTensorLogicalValues(const BulkTensorStorage &tensor) {
  llvm::Expected<std::vector<RawLogicalValue>> values =
      unpackPhysicalTensorLogicalValues(tensor.getKey(), tensor.getStorage());
  if (!values)
    return bulkError(BulkTensorNumericErrorCode::InvalidInputEncoding,
                     llvm::toString(values.takeError()));
  return values;
}

llvm::Expected<BulkTensorStorage>
packBulkTensorLogicalValues(const PhysicalTensorDescriptor &key,
                            llvm::ArrayRef<RawLogicalValue> values,
                            uint8_t paddingFill) {
  llvm::Expected<uint64_t> bytes = getBulkTensorPhysicalBytes(key);
  if (!bytes)
    return bytes.takeError();
  if (*bytes > std::numeric_limits<size_t>::max())
    return bulkError(BulkTensorNumericErrorCode::WorkCountOverflow,
                     "physical tensor footprint exceeds host size_t");
  return packIntoTemplate(
      key, values,
      std::vector<uint8_t>(static_cast<size_t>(*bytes), paddingFill));
}

std::string
computeBulkTensorPayloadDigest(llvm::ArrayRef<BulkTensorStorage> tensors) {
  llvm::SmallString<1024> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-bulk-payload");
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

std::string computeBulkTensorStorageDigest(const BulkTensorStorage &tensor) {
  llvm::SmallString<512> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-bulk-storage");
  appendField(stream, "key", descriptorDigest(tensor.getKey()));
  appendField(stream, "bytes", sha256(tensor.getStorage()));
  return sha256(payload);
}

llvm::Expected<std::string>
computeBulkGemmProblemDigest(const FormalGemmOperation &operation) {
  const FormalGemmOperation *gemm = &operation;

  llvm::SmallString<1024> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-bulk-gemm-problem");
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
