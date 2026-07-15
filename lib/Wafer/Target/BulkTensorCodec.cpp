//===- BulkTensorCodec.cpp - Bulk physical codec and digest ----------===//

#include "BulkTensorNumericInternal.h"

#include "Wafer/Target/PhysicalTensorCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <utility>
#include <vector>

namespace wafer::bulk_detail {

llvm::Expected<BulkTensorStorage>
packIntoTemplate(const NumericTensorKey &key,
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
BulkTensorStorage::create(NumericTensorKey key, std::vector<uint8_t> storage) {
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
getBulkTensorPhysicalBytes(const NumericTensorKey &key) {
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
packBulkTensorLogicalValues(const NumericTensorKey &key,
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
  appendField(stream, "schema", "wafer-bulk-payload-v1");
  appendField(stream, "tensor_count", tensors.size());
  for (auto [index, tensor] : llvm::enumerate(tensors)) {
    appendField(stream,
                (llvm::Twine("tensor_") + llvm::Twine(index) + "_key").str(),
                tensor.getKey().getDigest());
    appendField(stream,
                (llvm::Twine("tensor_") + llvm::Twine(index) + "_bytes").str(),
                sha256(tensor.getStorage()));
  }
  return sha256(payload);
}

std::string computeBulkTensorStorageDigest(const BulkTensorStorage &tensor) {
  llvm::SmallString<512> payload;
  llvm::raw_svector_ostream stream(payload);
  appendField(stream, "schema", "wafer-bulk-storage-v1");
  appendField(stream, "key", tensor.getKey().getDigest());
  appendField(stream, "bytes", sha256(tensor.getStorage()));
  return sha256(payload);
}

} // namespace wafer
