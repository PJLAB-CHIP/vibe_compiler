//===- XlaSpmdMetadata.cpp - Program and shard metadata codec --------===//

#include "XlaSpmdPartitionerInternal.h"

#include "absl/strings/str_cat.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tsl/platform/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <optional>
#include <utility>

namespace wafer::xla_spmd_helper {

static absl::StatusOr<std::string> jsonString(const llvm::json::Object &object,
                                              llvm::StringRef field) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return absl::InvalidArgumentError(
        absl::StrCat("missing string field ", field.str()));
  return value->str();
}

static absl::StatusOr<std::vector<int64_t>>
jsonShape(const llvm::json::Object &object, llvm::StringRef field) {
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

static absl::StatusOr<std::vector<TensorSignature>>
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

absl::StatusOr<ProgramMetadata> parseMeta(const fs::path &path) {
  TF_ASSIGN_OR_RETURN(std::string metaText, readFile(path));
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(metaText);
  if (!parsed)
    return absl::InvalidArgumentError(llvm::toString(parsed.takeError()));
  llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return absl::InvalidArgumentError(
        "program metadata root must be an object");

  ProgramMetadata meta;
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
    if (std::optional<int64_t> position = object->getInteger("position"))
      location.position = *position;
    meta.inputLocations.push_back(std::move(location));
  }
  return meta;
}

static llvm::json::Array shapeToJson(llvm::ArrayRef<int64_t> shape) {
  llvm::json::Array array;
  for (int64_t dim : shape)
    array.push_back(dim);
  return array;
}

static llvm::json::Array
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

static llvm::json::Object
distributedPartitionToJson(const DistributedPartition &partition) {
  return llvm::json::Object{
      {"partition_id", partition.partitionId},
      {"replica_id", partition.replicaId},
      {"offsets", shapeToJson(partition.offsets)},
      {"sizes", shapeToJson(partition.sizes)},
      {"strides", shapeToJson(partition.strides)},
  };
}

static llvm::json::Object
distributedBindingToJson(const DistributedBoundaryBinding &binding,
                         llvm::StringRef indexField) {
  llvm::json::Array partitions;
  for (const DistributedPartition &partition : binding.partitions)
    partitions.push_back(distributedPartitionToJson(partition));
  return llvm::json::Object{
      {indexField, binding.index},
      {"distribution", binding.distribution},
      {"global_shape", shapeToJson(binding.globalShape)},
      {"local_shape", shapeToJson(binding.localShape)},
      {"dtype", binding.dtype},
      {"partitions", std::move(partitions)},
  };
}

static llvm::json::Object
distributedBoundaryToJson(const DistributedBoundary &boundary) {
  llvm::json::Array inputs;
  for (const DistributedBoundaryBinding &binding : boundary.inputs)
    inputs.push_back(distributedBindingToJson(binding, "argument_index"));
  llvm::json::Array outputs;
  for (const DistributedBoundaryBinding &binding : boundary.outputs)
    outputs.push_back(distributedBindingToJson(binding, "result_index"));
  return llvm::json::Object{
      {"num_partitions", boundary.numPartitions},
      {"inputs", std::move(inputs)},
      {"outputs", std::move(outputs)},
  };
}

std::string metaToString(ProgramMetadata meta,
                         std::vector<TensorSignature> inputSignatures,
                         std::vector<TensorSignature> outputSignatures,
                         DistributedBoundary boundary) {
  meta.root["input_signature"] = signaturesToJson(inputSignatures);
  meta.root["output_signature"] = signaturesToJson(outputSignatures);
  meta.root["distributed_boundary"] = distributedBoundaryToJson(boundary);
  return llvm::formatv("{0:2}", llvm::json::Value(std::move(meta.root))).str() +
         "\n";
}

static std::string dtypeFromElementType(mlir::Type type,
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

static absl::StatusOr<TensorSignature>
signatureFromType(mlir::Type type, std::string fallbackDtype) {
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
inputSignaturesFromFunc(mlir::func::FuncOp func, const ProgramMetadata &meta) {
  std::vector<TensorSignature> signatures;
  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != meta.inputSignatures.size())
    return absl::InvalidArgumentError(
        "post-SPMD argument count differs from program directory metadata");
  for (auto [index, input] : llvm::enumerate(type.getInputs())) {
    TF_ASSIGN_OR_RETURN(
        TensorSignature signature,
        signatureFromType(input, meta.inputSignatures[index].dtype));
    signatures.push_back(std::move(signature));
  }
  return signatures;
}

absl::StatusOr<std::vector<TensorSignature>>
outputSignaturesFromFunc(mlir::func::FuncOp func, const ProgramMetadata &meta) {
  std::vector<TensorSignature> signatures;
  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumResults() != meta.outputSignatures.size())
    return absl::InvalidArgumentError(
        "post-SPMD result count differs from program directory metadata");
  for (auto [index, result] : llvm::enumerate(type.getResults())) {
    TF_ASSIGN_OR_RETURN(
        TensorSignature signature,
        signatureFromType(result, meta.outputSignatures[index].dtype));
    signatures.push_back(std::move(signature));
  }
  return signatures;
}

static llvm::json::Object shardToJson(const ParameterShard &shard) {
  return llvm::json::Object{
      {"partition_id", shard.partitionId},
      {"replica_id", shard.replicaId},
      {"file", shard.file},
      {"offsets", shapeToJson(shard.offsets)},
      {"sizes", shapeToJson(shard.sizes)},
      {"strides", shapeToJson(shard.strides)},
  };
}

static llvm::json::Object bindingToJson(const ParameterBinding &binding) {
  llvm::json::Array shards;
  for (const ParameterShard &shard : binding.shards)
    shards.push_back(shardToJson(shard));
  return llvm::json::Object{
      {"argument_index", binding.argumentIndex},
      {"name", binding.name},
      {"distribution", binding.distribution},
      {"global_shape", shapeToJson(binding.globalShape)},
      {"local_shape", shapeToJson(binding.localShape)},
      {"dtype", binding.dtype},
      {"shards", std::move(shards)},
  };
}

std::string bindingsToJson(const std::string &functionName,
                           int64_t numPartitions,
                           const std::vector<ParameterBinding> &bindings) {
  llvm::json::Array parameters;
  for (const ParameterBinding &binding : bindings)
    parameters.push_back(bindingToJson(binding));
  llvm::json::Object root{
      {"function", functionName},
      {"num_partitions", numPartitions},
      {"parameters", std::move(parameters)},
  };
  return llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str() +
         "\n";
}

} // namespace wafer::xla_spmd_helper
