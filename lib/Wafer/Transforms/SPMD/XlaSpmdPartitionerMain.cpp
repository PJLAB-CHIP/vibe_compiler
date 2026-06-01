//===- XlaSpmdPartitionerMain.cpp - Wafer XLA SPMD stage helper ----------===//

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "tsl/platform/statusor.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/spmd_prepare.h"
#include "xla/shape.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  fs::path inputBundle;
  fs::path outputBundle;
  std::string entryFunction = "forward";
  int64_t logicalRankCount = 0;
};

struct InputLocation {
  std::string type;
  std::string name;
};

struct TensorSignature {
  std::vector<int64_t> shape;
  std::string dtype;
};

struct BundleMeta {
  llvm::json::Object root;
  std::vector<InputLocation> inputLocations;
  std::vector<TensorSignature> inputSignatures;
  std::vector<TensorSignature> outputSignatures;
};

struct ParameterShard {
  int64_t rank = 0;
  int64_t replicaId = 0;
  std::string file;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct ParameterBinding {
  int64_t argumentIndex = 0;
  std::string name;
  std::string dtype;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::vector<ParameterShard> shards;
};

struct NpyPayload {
  std::vector<uint8_t> bytes;
  size_t dataOffset = 0;
  std::string descr;
};

void printUsage() {
  std::cerr << "wafer_xla_spmd_partitioner "
               "--input-bundle <dir> --output-bundle <dir> "
               "--entry-function forward --logical-rank-count <n>\n";
}

absl::StatusOr<Options> parseOptions(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto requireValue = [&](std::string_view name) -> absl::StatusOr<char *> {
      if (i + 1 >= argc)
        return absl::InvalidArgumentError(
            absl::StrCat("missing value for ", name));
      return argv[++i];
    };

    if (arg == "--input-bundle") {
      TF_ASSIGN_OR_RETURN(char *value, requireValue(arg));
      options.inputBundle = value;
    } else if (arg == "--output-bundle") {
      TF_ASSIGN_OR_RETURN(char *value, requireValue(arg));
      options.outputBundle = value;
    } else if (arg == "--entry-function") {
      TF_ASSIGN_OR_RETURN(char *value, requireValue(arg));
      options.entryFunction = value;
    } else if (arg == "--logical-rank-count") {
      TF_ASSIGN_OR_RETURN(char *value, requireValue(arg));
      try {
        options.logicalRankCount = std::stoll(value);
      } catch (...) {
        return absl::InvalidArgumentError(
            "--logical-rank-count must be an integer");
      }
    } else if (arg == "--help") {
      printUsage();
      std::exit(0);
    } else {
      return absl::InvalidArgumentError(absl::StrCat("unknown option: ", arg));
    }
  }

  if (options.inputBundle.empty())
    return absl::InvalidArgumentError("missing --input-bundle");
  if (options.outputBundle.empty())
    return absl::InvalidArgumentError("missing --output-bundle");
  if (options.entryFunction.empty())
    return absl::InvalidArgumentError("missing --entry-function");
  if (options.logicalRankCount <= 0)
    return absl::InvalidArgumentError("--logical-rank-count must be positive");
  return options;
}

absl::StatusOr<std::string> readFile(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return absl::NotFoundError(absl::StrCat("failed to open ", path.string()));
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

absl::Status writeFile(const fs::path &path, std::string_view content) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output)
    return absl::InternalError(absl::StrCat("failed to write ", path.string()));
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  return absl::OkStatus();
}

absl::Status writeBytes(const fs::path &path,
                        const std::vector<uint8_t> &content) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output)
    return absl::InternalError(absl::StrCat("failed to write ", path.string()));
  output.write(reinterpret_cast<const char *>(content.data()),
               static_cast<std::streamsize>(content.size()));
  return absl::OkStatus();
}

absl::StatusOr<std::string> jsonString(const llvm::json::Object &object,
                                       llvm::StringRef field) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return absl::InvalidArgumentError(
        absl::StrCat("missing string field ", field.str()));
  return value->str();
}

absl::StatusOr<std::vector<int64_t>> jsonShape(const llvm::json::Object &object,
                                               llvm::StringRef field) {
  const llvm::json::Array *array = object.getArray(field);
  if (!array)
    return absl::InvalidArgumentError(
        absl::StrCat("missing array field ", field.str()));
  std::vector<int64_t> shape;
  for (const llvm::json::Value &value : *array) {
    std::optional<int64_t> dim = value.getAsInteger();
    if (!dim)
      return absl::InvalidArgumentError(
          absl::StrCat("non-integer shape dim in ", field.str()));
    shape.push_back(*dim);
  }
  return shape;
}

absl::StatusOr<std::vector<TensorSignature>>
parseSignatures(const llvm::json::Object &root, llvm::StringRef field) {
  const llvm::json::Array *array = root.getArray(field);
  if (!array)
    return absl::InvalidArgumentError(
        absl::StrCat("missing array field ", field.str()));
  std::vector<TensorSignature> signatures;
  for (const llvm::json::Value &value : *array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return absl::InvalidArgumentError(
          absl::StrCat("signature entries must be objects in ", field.str()));
    TensorSignature signature;
    TF_ASSIGN_OR_RETURN(signature.shape, jsonShape(*object, "shape"));
    TF_ASSIGN_OR_RETURN(signature.dtype, jsonString(*object, "dtype"));
    signatures.push_back(std::move(signature));
  }
  return signatures;
}

absl::StatusOr<BundleMeta> parseMeta(const fs::path &path) {
  TF_ASSIGN_OR_RETURN(std::string metaText, readFile(path));
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(metaText);
  if (!parsed)
    return absl::InvalidArgumentError(llvm::toString(parsed.takeError()));
  llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return absl::InvalidArgumentError("bundle meta root must be an object");

  BundleMeta meta;
  meta.root = std::move(*root);
  TF_ASSIGN_OR_RETURN(meta.inputSignatures,
                      parseSignatures(meta.root, "input_signature"));
  TF_ASSIGN_OR_RETURN(meta.outputSignatures,
                      parseSignatures(meta.root, "output_signature"));

  const llvm::json::Array *locations = meta.root.getArray("input_locations");
  if (!locations)
    return absl::InvalidArgumentError("missing input_locations");
  for (const llvm::json::Value &value : *locations) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return absl::InvalidArgumentError(
          "input_locations entries must be objects");
    InputLocation location;
    TF_ASSIGN_OR_RETURN(location.type, jsonString(*object, "type_"));
    if (std::optional<llvm::StringRef> name = object->getString("name"))
      location.name = name->str();
    meta.inputLocations.push_back(std::move(location));
  }
  return meta;
}

llvm::json::Array shapeToJson(llvm::ArrayRef<int64_t> shape) {
  llvm::json::Array array;
  for (int64_t dim : shape)
    array.push_back(dim);
  return array;
}

llvm::json::Array
signaturesToJson(const std::vector<TensorSignature> &signatures) {
  llvm::json::Array array;
  for (const TensorSignature &signature : signatures) {
    array.push_back(llvm::json::Object{
        {"shape", shapeToJson(signature.shape)},
        {"dtype", signature.dtype},
        {"dynamic_dims", llvm::json::Array{}},
    });
  }
  return array;
}

std::string metaToString(BundleMeta meta,
                         std::vector<TensorSignature> inputSignatures,
                         std::vector<TensorSignature> outputSignatures) {
  meta.root["input_signature"] = signaturesToJson(inputSignatures);
  meta.root["output_signature"] = signaturesToJson(outputSignatures);
  return llvm::formatv("{0:2}", llvm::json::Value(std::move(meta.root))).str() +
         "\n";
}

std::string dtypeFromElementType(mlir::Type type,
                                 std::string fallback = "float32") {
  if (type.isF32())
    return "float32";
  if (type.isF16())
    return "float16";
  if (type.isBF16())
    return "bfloat16";
  if (type.isInteger(64))
    return "int64";
  if (type.isInteger(32))
    return "int32";
  if (type.isInteger(1))
    return "bool";
  return fallback;
}

absl::StatusOr<TensorSignature> signatureFromType(mlir::Type type,
                                                  std::string fallbackDtype) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return absl::InvalidArgumentError(
        "post-SPMD function boundary must use static ranked tensors");
  TensorSignature signature;
  signature.shape.assign(tensor.getShape().begin(), tensor.getShape().end());
  signature.dtype =
      dtypeFromElementType(tensor.getElementType(), fallbackDtype);
  return signature;
}

absl::StatusOr<mlir::func::FuncOp> findMain(mlir::ModuleOp module) {
  if (auto named = module.lookupSymbol<mlir::func::FuncOp>("main"))
    return named;
  mlir::func::FuncOp result;
  module.walk([&](mlir::func::FuncOp func) {
    if (!result && !func.isPrivate()) {
      result = func;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!result)
    return absl::InvalidArgumentError("post-SPMD module has no public func");
  return result;
}

absl::StatusOr<std::vector<TensorSignature>>
inputSignaturesFromFunc(mlir::func::FuncOp func, const BundleMeta &meta) {
  std::vector<TensorSignature> signatures;
  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != meta.inputSignatures.size())
    return absl::InvalidArgumentError(
        "post-SPMD argument count differs from bundle metadata");
  for (auto [index, input] : llvm::enumerate(type.getInputs())) {
    TF_ASSIGN_OR_RETURN(
        TensorSignature signature,
        signatureFromType(input, meta.inputSignatures[index].dtype));
    signatures.push_back(std::move(signature));
  }
  return signatures;
}

absl::StatusOr<std::vector<TensorSignature>>
outputSignaturesFromFunc(mlir::func::FuncOp func, const BundleMeta &meta) {
  std::vector<TensorSignature> signatures;
  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumResults() != meta.outputSignatures.size())
    return absl::InvalidArgumentError(
        "post-SPMD result count differs from bundle metadata");
  for (auto [index, result] : llvm::enumerate(type.getResults())) {
    TF_ASSIGN_OR_RETURN(
        TensorSignature signature,
        signatureFromType(result, meta.outputSignatures[index].dtype));
    signatures.push_back(std::move(signature));
  }
  return signatures;
}

std::vector<int64_t> shapeDims(const xla::Shape &shape) {
  return std::vector<int64_t>(shape.dimensions().begin(),
                              shape.dimensions().end());
}

int64_t elementSize(std::string_view dtype) {
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

std::string npyDescr(std::string_view dtype) {
  if (dtype == "float32")
    return "<f4";
  if (dtype == "float16")
    return "<f2";
  if (dtype == "int32")
    return "<i4";
  if (dtype == "int64")
    return "<i8";
  if (dtype == "bool")
    return "|b1";
  return "<f4";
}

int64_t elementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape)
    count *= dim;
  return count;
}

absl::StatusOr<NpyPayload> readTensorPayload(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return absl::NotFoundError(
        absl::StrCat("missing parameter data file: ", path.string()));
  NpyPayload payload;
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
    size_t descr = header.find("'descr': '");
    if (descr != std::string::npos) {
      size_t start = descr + 10;
      size_t end = header.find("'", start);
      if (end != std::string::npos)
        payload.descr = header.substr(start, end - start);
    }
  }
  return payload;
}

std::vector<int64_t> rowMajorStrides(llvm::ArrayRef<int64_t> shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

void copySliceRecursive(const uint8_t *input, llvm::ArrayRef<int64_t> strides,
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

absl::StatusOr<std::vector<uint8_t>>
sliceRowMajorPayload(const NpyPayload &payload,
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

std::vector<uint8_t> withNpyHeader(std::vector<uint8_t> payload,
                                   llvm::ArrayRef<int64_t> shape,
                                   std::string descr) {
  std::string shapeText = "(" + absl::StrJoin(shape, ", ");
  if (shape.size() == 1)
    shapeText += ",";
  shapeText += ")";
  std::string header = "{'descr': '" + descr +
                       "', 'fortran_order': False, 'shape': " + shapeText +
                       ", }";
  size_t prefix = 10;
  size_t padding = 16 - ((prefix + header.size() + 1) % 16);
  if (padding == 16)
    padding = 0;
  uint16_t headerLen = static_cast<uint16_t>(header.size() + padding + 1);

  std::vector<uint8_t> output;
  const char magic[] = "\x93NUMPY";
  output.insert(output.end(), magic, magic + 6);
  output.push_back(1);
  output.push_back(0);
  output.push_back(static_cast<uint8_t>(headerLen & 0xff));
  output.push_back(static_cast<uint8_t>((headerLen >> 8) & 0xff));
  output.insert(output.end(), header.begin(), header.end());
  output.insert(output.end(), padding, ' ');
  output.push_back('\n');
  output.insert(output.end(), payload.begin(), payload.end());
  return output;
}

std::vector<int64_t> zeros(size_t size) {
  return std::vector<int64_t>(size, 0);
}

std::vector<int64_t> ones(size_t size) { return std::vector<int64_t>(size, 1); }

ParameterShard shardForRank(const xla::Shape &globalShape,
                            const xla::HloSharding *sharding, int64_t rank,
                            std::string file) {
  ParameterShard shard;
  shard.rank = rank;
  shard.replicaId = 0;
  shard.file = std::move(file);
  shard.strides = ones(globalShape.rank());

  if (!sharding || sharding->IsReplicated()) {
    shard.offsets = zeros(globalShape.rank());
    shard.sizes = shapeDims(globalShape);
    return shard;
  }

  shard.offsets = sharding->TileOffsetForDevice(globalShape, rank);
  std::vector<int64_t> limits = sharding->TileLimitForDevice(globalShape, rank);
  for (auto [offset, limit] : llvm::zip(shard.offsets, limits))
    shard.sizes.push_back(limit - offset);
  return shard;
}

llvm::json::Object shardToJson(const ParameterShard &shard) {
  return llvm::json::Object{
      {"rank", shard.rank},
      {"replica_id", shard.replicaId},
      {"file", shard.file},
      {"offsets", shapeToJson(shard.offsets)},
      {"sizes", shapeToJson(shard.sizes)},
      {"strides", shapeToJson(shard.strides)},
  };
}

llvm::json::Object bindingToJson(const ParameterBinding &binding) {
  llvm::json::Array shards;
  for (const ParameterShard &shard : binding.shards)
    shards.push_back(shardToJson(shard));
  return llvm::json::Object{
      {"argument_index", binding.argumentIndex},
      {"name", binding.name},
      {"global_shape", shapeToJson(binding.globalShape)},
      {"local_shape", shapeToJson(binding.localShape)},
      {"dtype", binding.dtype},
      {"shards", std::move(shards)},
  };
}

std::string bindingsToJson(const std::string &functionName,
                           int64_t logicalRankCount,
                           const std::vector<ParameterBinding> &bindings) {
  llvm::json::Array parameters;
  for (const ParameterBinding &binding : bindings)
    parameters.push_back(bindingToJson(binding));
  llvm::json::Object root{
      {"parameter_shards_version", 2},
      {"function", functionName},
      {"logical_rank_count", logicalRankCount},
      {"parameters", std::move(parameters)},
  };
  return llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str() +
         "\n";
}

absl::Status
materializeParameterShards(const Options &options, const BundleMeta &meta,
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
    TF_ASSIGN_OR_RETURN(
        NpyPayload payload,
        readTensorPayload(options.inputBundle / "data" / location.name));
    int64_t bytesPerElement = elementSize(binding.dtype);
    if (bytesPerElement <= 0)
      return absl::InvalidArgumentError(
          absl::StrCat("unsupported parameter dtype: ", binding.dtype));

    std::string descr =
        payload.descr.empty() ? npyDescr(binding.dtype) : payload.descr;
    for (int64_t rank = 0; rank < options.logicalRankCount; ++rank) {
      std::string relative =
          absl::StrCat("parameter_shards/", binding.name, "/rank_",
                       llvm::formatv("{0:05}", rank).str(), ".npy");
      ParameterShard shard =
          shardForRank(preParam->shape(), sharding, rank, relative);
      TF_ASSIGN_OR_RETURN(std::vector<uint8_t> rawShard,
                          sliceRowMajorPayload(payload, binding.globalShape,
                                               shard.offsets, shard.sizes,
                                               bytesPerElement));
      std::vector<uint8_t> fileBytes =
          withNpyHeader(std::move(rawShard), shard.sizes, descr);
      TF_RETURN_IF_ERROR(
          writeBytes(options.outputBundle / relative, fileBytes));
      binding.shards.push_back(std::move(shard));
    }

    bindings.push_back(std::move(binding));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<xla::HloModule>>
stablehloToHloModule(mlir::ModuleOp module, int64_t logicalRankCount) {
  xla::XlaComputation computation;
  TF_RETURN_IF_ERROR(xla::MlirToXlaComputation(
      module, computation, /*use_tuple_args=*/false, /*return_tuple=*/false,
      /*use_shardy=*/true));

  xla::HloModuleConfig config;
  xla::ProgramShape programShape(computation.proto().host_program_shape());
  config.SetDefaultComputationLayout(programShape);
  config.set_use_spmd_partitioning(true);
  config.set_num_partitions(logicalRankCount);
  config.set_replica_count(1);
  absl::InlinedVector<bool, 8> allowParams(programShape.parameters_size(),
                                           false);
  config.set_allow_spmd_sharding_propagation_to_parameters(allowParams);
  absl::InlinedVector<bool, 8> allowOutputs(
      programShape.result().IsTuple()
          ? programShape.result().tuple_shapes_size()
          : 1,
      true);
  config.set_allow_spmd_sharding_propagation_to_output(allowOutputs);

  return xla::HloModule::CreateFromProto(computation.proto(), config);
}

absl::Status runSpmdPartitioner(xla::HloModule *module,
                                int64_t logicalRankCount) {
  xla::spmd::SpmdPartitionerOptions options;
  options.allow_module_signature_change = true;
  auto collectiveOpsCreator =
      xla::spmd::GetDefaultCollectiveOpsCreator(logicalRankCount,
                                                /*num_replicas=*/1);

  xla::HloPassPipeline pipeline("wafer-spmd-partitioning");
  pipeline.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                     /*allow_mixed_precision=*/false);
  pipeline.AddPass<xla::spmd::SpmdPrepare>();
  pipeline.AddPass<xla::spmd::SpmdPartitioner>(
      logicalRankCount, /*num_replicas=*/1, options, collectiveOpsCreator);
  pipeline.AddPass<xla::HloVerifier>(/*layout_sensitive=*/false,
                                     /*allow_mixed_precision=*/false);
  TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
hloModuleToStablehlo(mlir::MLIRContext &context, xla::HloModule *module,
                     const std::vector<std::string> &parameterShardings) {
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlirModule,
      xla::ConvertHloToMlirHlo(context, module,
                               /*import_all_computations=*/false,
                               /*flatten_computation_args_result=*/true));

  mlir::PassManager pm(&context);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (mlir::failed(pm.run(*mlirModule)))
    return absl::InternalError(
        "failed to legalize partitioned HLO to StableHLO");

  std::vector<mlir::Attribute> shardingAttrs;
  shardingAttrs.reserve(parameterShardings.size());
  for (const std::string &sharding : parameterShardings)
    shardingAttrs.push_back(mlir::StringAttr::get(&context, sharding));
  (*mlirModule)
      ->setAttr("mhlo.spmd_parameters_shardings",
                mlir::ArrayAttr::get(&context, shardingAttrs));
  (*mlirModule)
      ->setAttr("mhlo.use_auto_spmd_partitioning",
                mlir::BoolAttr::get(&context, false));
  return mlirModule;
}

std::string moduleToString(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  os << "\n";
  return text;
}

absl::Status run(const Options &options) {
  fs::remove_all(options.outputBundle);
  fs::create_directories(options.outputBundle / "functions");

  TF_ASSIGN_OR_RETURN(BundleMeta meta, parseMeta(options.inputBundle /
                                                 "functions" / "forward.meta"));
  TF_ASSIGN_OR_RETURN(
      std::string mlirText,
      readFile(options.inputBundle / "functions" / "forward.mlir"));

  mlir::MLIRContext inputContext;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> inputModule,
                      xla::ParseMlirModuleString(mlirText, inputContext));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::HloModule> prePartitionModule,
      stablehloToHloModule(*inputModule, options.logicalRankCount));

  std::vector<std::string> parameterShardings;
  const xla::HloComputation *preEntry = prePartitionModule->entry_computation();
  for (int64_t index = 0; index < preEntry->num_parameters(); ++index) {
    const xla::HloInstruction *parameter =
        preEntry->parameter_instruction(index);
    parameterShardings.push_back(parameter->has_sharding()
                                     ? parameter->sharding().ToString()
                                     : std::string("{replicated}"));
  }

  std::unique_ptr<xla::HloModule> partitionedModule =
      prePartitionModule->Clone();
  TF_RETURN_IF_ERROR(
      runSpmdPartitioner(partitionedModule.get(), options.logicalRankCount));
  mlir::MLIRContext outputContext;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> outputModule,
                      hloModuleToStablehlo(outputContext,
                                           partitionedModule.get(),
                                           parameterShardings));
  TF_ASSIGN_OR_RETURN(mlir::func::FuncOp mainFunc, findMain(*outputModule));
  TF_ASSIGN_OR_RETURN(std::vector<TensorSignature> inputSignatures,
                      inputSignaturesFromFunc(mainFunc, meta));
  TF_ASSIGN_OR_RETURN(std::vector<TensorSignature> outputSignatures,
                      outputSignaturesFromFunc(mainFunc, meta));

  std::vector<ParameterBinding> bindings;
  TF_RETURN_IF_ERROR(materializeParameterShards(
      options, meta, *prePartitionModule, *partitionedModule, bindings));

  TF_RETURN_IF_ERROR(
      writeFile(options.outputBundle / "functions" / "forward.mlir",
                moduleToString(*outputModule)));
  TF_RETURN_IF_ERROR(
      writeFile(options.outputBundle / "functions" / "forward.meta",
                metaToString(std::move(meta), std::move(inputSignatures),
                             std::move(outputSignatures))));
  TF_RETURN_IF_ERROR(writeFile(
      options.outputBundle / "functions" / "forward.parameter_shards.json",
      bindingsToJson(options.entryFunction, options.logicalRankCount,
                     bindings)));
  return absl::OkStatus();
}

} // namespace

int main(int argc, char **argv) {
  absl::StatusOr<Options> options = parseOptions(argc, argv);
  if (!options.ok()) {
    std::cerr << "wafer_xla_spmd_partitioner: " << options.status() << "\n";
    printUsage();
    return 1;
  }
  absl::Status status = run(*options);
  if (!status.ok()) {
    std::cerr << "wafer_xla_spmd_partitioner: " << status << "\n";
    return 1;
  }
  return 0;
}
