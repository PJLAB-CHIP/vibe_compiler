//===- XlaSpmdPayload.cpp - NPY codec and parameter shard payloads ---===//

#include "XlaSpmdPartitionerInternal.h"

#include "absl/strings/str_cat.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <fstream>
#include <iterator>
#include <utility>

namespace wafer::xla_spmd_helper {

namespace {

struct TensorPayload {
  std::vector<uint8_t> bytes;
  size_t dataOffset = 0;
};

} // namespace

static int64_t elementSize(std::string_view dtype) {
  if (dtype == "float32" || dtype == "int32")
    return 4;
  if (dtype == "float16" || dtype == "bfloat16")
    return 2;
  if (dtype == "int64")
    return 8;
  if (dtype == "bool")
    return 1;
  return 0;
}

static int64_t elementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape)
    count *= dim;
  return count;
}

static absl::StatusOr<std::string> npyDescrForDtype(std::string_view dtype) {
  if (dtype == "float32")
    return "<f4";
  if (dtype == "float64")
    return "<f8";
  if (dtype == "float16")
    return "<f2";
  if (dtype == "bfloat16")
    return "|V2";
  if (dtype == "int8")
    return "|i1";
  if (dtype == "uint8")
    return "|u1";
  if (dtype == "int16")
    return "<i2";
  if (dtype == "int32")
    return "<i4";
  if (dtype == "int64")
    return "<i8";
  if (dtype == "bool")
    return "|b1";
  return absl::InvalidArgumentError(
      absl::StrCat("unsupported parameter dtype for npy payload: ", dtype));
}

static std::string npyShapeTuple(llvm::ArrayRef<int64_t> shape) {
  std::string result = "(";
  for (auto [index, dim] : llvm::enumerate(shape)) {
    if (index)
      result += ", ";
    result += std::to_string(dim);
  }
  if (shape.size() == 1)
    result += ",";
  result += ")";
  return result;
}

static absl::StatusOr<std::vector<uint8_t>>
makeNpyPayload(llvm::ArrayRef<uint8_t> rawBytes, std::string_view dtype,
               llvm::ArrayRef<int64_t> shape) {
  TF_ASSIGN_OR_RETURN(std::string descr, npyDescrForDtype(dtype));
  std::string header = absl::StrCat(
      "{'descr': '", descr,
      "', 'fortran_order': False, 'shape': ", npyShapeTuple(shape), ", }");
  constexpr size_t kPreambleSize = 10;
  size_t padding = (16 - ((kPreambleSize + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > UINT16_MAX)
    return absl::InvalidArgumentError("npy header is too large for v1.0");

  std::vector<uint8_t> payload;
  payload.reserve(kPreambleSize + header.size() + rawBytes.size());
  const char magic[] = "\x93NUMPY";
  payload.insert(payload.end(), magic, magic + 6);
  payload.push_back(1);
  payload.push_back(0);
  uint16_t headerLen = static_cast<uint16_t>(header.size());
  payload.push_back(static_cast<uint8_t>(headerLen & 0xff));
  payload.push_back(static_cast<uint8_t>((headerLen >> 8) & 0xff));
  payload.insert(payload.end(), header.begin(), header.end());
  payload.insert(payload.end(), rawBytes.begin(), rawBytes.end());
  return payload;
}

static absl::StatusOr<TensorPayload> readTensorPayload(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return absl::NotFoundError(
        absl::StrCat("missing parameter data file: ", path.string()));
  TensorPayload payload;
  payload.bytes.assign(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  if (payload.bytes.size() >= 10 &&
      std::string_view(reinterpret_cast<const char *>(payload.bytes.data()),
                       6) == std::string_view("\x93NUMPY", 6)) {
    uint8_t major = payload.bytes[6];
    size_t headerLenOffset = 8;
    uint32_t headerLen = 0;
    if (major == 1) {
      headerLen = payload.bytes[8] | (payload.bytes[9] << 8);
      headerLenOffset = 10;
    } else if (major == 2) {
      if (payload.bytes.size() < 12)
        return absl::InvalidArgumentError("truncated npy header");
      headerLen = payload.bytes[8] | (payload.bytes[9] << 8) |
                  (payload.bytes[10] << 16) | (payload.bytes[11] << 24);
      headerLenOffset = 12;
    } else {
      return absl::InvalidArgumentError("unsupported npy version");
    }
    if (payload.bytes.size() < headerLenOffset + headerLen)
      return absl::InvalidArgumentError("truncated npy payload");
    std::string header(
        reinterpret_cast<const char *>(payload.bytes.data() + headerLenOffset),
        headerLen);
    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("\"fortran_order\": false") == std::string::npos)
      return absl::InvalidArgumentError(
          "only row-major npy payloads are supported");
    payload.dataOffset = headerLenOffset + headerLen;
  }
  return payload;
}

static std::vector<int64_t> rowMajorStrides(llvm::ArrayRef<int64_t> shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

static void copySliceRecursive(const uint8_t *input,
                               llvm::ArrayRef<int64_t> strides,
                               llvm::ArrayRef<int64_t> offsets,
                               llvm::ArrayRef<int64_t> sizes, int64_t dim,
                               int64_t inputBaseElement, int64_t elementBytes,
                               std::vector<uint8_t> &output) {
  if (dim == static_cast<int64_t>(sizes.size()) - 1) {
    const uint8_t *source =
        input + (inputBaseElement + offsets[dim] * strides[dim]) * elementBytes;
    size_t bytes = static_cast<size_t>(sizes[dim] * elementBytes);
    output.insert(output.end(), source, source + bytes);
    return;
  }
  for (int64_t i = 0; i < sizes[dim]; ++i) {
    copySliceRecursive(input, strides, offsets, sizes, dim + 1,
                       inputBaseElement + (offsets[dim] + i) * strides[dim],
                       elementBytes, output);
  }
}

static absl::StatusOr<std::vector<uint8_t>>
sliceRowMajorPayload(const TensorPayload &payload,
                     llvm::ArrayRef<int64_t> globalShape,
                     llvm::ArrayRef<int64_t> offsets,
                     llvm::ArrayRef<int64_t> sizes, int64_t elementBytes) {
  if (globalShape.size() != offsets.size() || offsets.size() != sizes.size())
    return absl::InvalidArgumentError("slice rank does not match tensor rank");
  int64_t requiredElements = elementCount(globalShape);
  size_t requiredBytes =
      payload.dataOffset + static_cast<size_t>(requiredElements * elementBytes);
  if (payload.bytes.size() < requiredBytes)
    return absl::InvalidArgumentError(
        "parameter payload is smaller than tensor");
  std::vector<uint8_t> output;
  output.reserve(static_cast<size_t>(elementCount(sizes) * elementBytes));
  if (sizes.empty()) {
    output.insert(output.end(), payload.bytes.begin() + payload.dataOffset,
                  payload.bytes.begin() + payload.dataOffset + elementBytes);
    return output;
  }
  std::vector<int64_t> strides = rowMajorStrides(globalShape);
  copySliceRecursive(payload.bytes.data() + payload.dataOffset, strides,
                     offsets, sizes, 0, 0, elementBytes, output);
  return output;
}

static ParameterShard shardForPartition(const xla::Shape &globalShape,
                                        const xla::HloSharding *sharding,
                                        int64_t partitionId,
                                        std::string file) {
  ParameterShard shard;
  shard.partitionId = partitionId;
  bool replicated = !sharding || sharding->IsReplicated();
  shard.replicaId = replicated ? partitionId : 0;
  shard.file = std::move(file);
  shard.strides = ones(globalShape.rank());

  if (replicated) {
    shard.offsets = zeros(globalShape.rank());
    shard.sizes = shapeDims(globalShape);
    return shard;
  }

  shard.offsets = sharding->TileOffsetForDevice(globalShape, partitionId);
  std::vector<int64_t> limits =
      sharding->TileLimitForDevice(globalShape, partitionId);
  for (auto [offset, limit] : llvm::zip(shard.offsets, limits))
    shard.sizes.push_back(limit - offset);
  return shard;
}

absl::Status
materializeParameterShards(const Options &options, const ProgramMetadata &meta,
                           const xla::HloModule &prePartitionModule,
                           const xla::HloModule &partitionedModule,
                           std::vector<ParameterBinding> &bindings) {
  const xla::HloComputation *preEntry = prePartitionModule.entry_computation();
  const xla::HloComputation *postEntry = partitionedModule.entry_computation();
  if (!preEntry || !postEntry)
    return absl::InvalidArgumentError(
        "HLO module is missing entry computation");

  for (size_t index = 0; index < meta.inputLocations.size(); ++index) {
    const InputLocation &location = meta.inputLocations[index];
    if (location.type != "parameter")
      continue;
    if (location.name.empty())
      return absl::InvalidArgumentError("parameter input has empty name");

    const xla::HloInstruction *preParam =
        preEntry->parameter_instruction(static_cast<int64_t>(index));
    const xla::HloInstruction *postParam =
        postEntry->parameter_instruction(static_cast<int64_t>(index));
    if (!preParam || !postParam)
      return absl::InvalidArgumentError("parameter instruction is missing");

    ParameterBinding binding;
    binding.argumentIndex = static_cast<int64_t>(index);
    binding.name = location.name;
    binding.dtype = meta.inputSignatures[index].dtype;
    binding.globalShape = shapeDims(preParam->shape());
    binding.localShape = shapeDims(postParam->shape());

    const xla::HloSharding *sharding =
        preParam->has_sharding() ? &preParam->sharding() : nullptr;
    if (sharding && sharding->HasPartialReplication())
      return absl::InvalidArgumentError(absl::StrCat(
          "parameter '", binding.name,
          "' uses partial replication, which current parameter shard metadata does "
          "not encode"));
    if (sharding && sharding->IsTileMaximal() && !sharding->IsReplicated())
      return absl::InvalidArgumentError(absl::StrCat(
          "parameter '", binding.name,
          "' uses single-device sharding, which current parameter shard metadata "
          "does not encode"));
    binding.distribution =
        (!sharding || sharding->IsReplicated()) ? "replicated" : "partitioned";
    TF_ASSIGN_OR_RETURN(
        TensorPayload payload,
        readTensorPayload(options.inputProgramDir / "data" / location.name));
    int64_t bytesPerElement = elementSize(binding.dtype);
    if (bytesPerElement <= 0)
      return absl::InvalidArgumentError(
          absl::StrCat("unsupported parameter dtype: ", binding.dtype));

    for (int64_t partitionId = 0; partitionId < options.numPartitions;
         ++partitionId) {
      std::string relative =
          absl::StrCat("parameter_shards/", binding.name, "/partition_",
                       llvm::formatv("{0:05}", partitionId).str(), ".npy");
      ParameterShard shard =
          shardForPartition(preParam->shape(), sharding, partitionId,
                            relative);
      TF_ASSIGN_OR_RETURN(std::vector<uint8_t> rawShard,
                          sliceRowMajorPayload(payload, binding.globalShape,
                                               shard.offsets, shard.sizes,
                                               bytesPerElement));
      TF_ASSIGN_OR_RETURN(std::vector<uint8_t> npyShard,
                          makeNpyPayload(rawShard, binding.dtype, shard.sizes));
      TF_RETURN_IF_ERROR(
          writeBytes(options.outputProgramDir / relative, npyShard));
      binding.shards.push_back(std::move(shard));
    }

    bindings.push_back(std::move(binding));
  }
  return absl::OkStatus();
}

} // namespace wafer::xla_spmd_helper
