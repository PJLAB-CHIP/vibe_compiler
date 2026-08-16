//===- ProgramMetadata.cpp - Frontend program metadata ------------------===//

#include "ProgramInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace wafer::frontend::program_detail {

bool readStringField(const llvm::json::Object &object, llvm::StringRef field,
                     std::string &out, llvm::raw_ostream &diagnostics) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return rejectProgramDirectory("expected string field '" + field.str() + "'",
                                  diagnostics);
  out = value->str();
  return false;
}

bool readIntegerField(const llvm::json::Object &object, llvm::StringRef field,
                      int64_t &out, llvm::raw_ostream &diagnostics) {
  std::optional<int64_t> value = object.getInteger(field);
  if (!value)
    return rejectProgramDirectory(
        "expected integer field '" + field.str() + "'", diagnostics);
  out = *value;
  return false;
}

namespace {

bool readShapeField(const llvm::json::Object &object,
                    std::vector<int64_t> &shape,
                    llvm::raw_ostream &diagnostics) {
  const llvm::json::Array *array = object.getArray("shape");
  if (!array)
    return rejectProgramDirectory("expected array field 'shape'", diagnostics);

  for (const llvm::json::Value &value : *array) {
    std::optional<int64_t> dim = value.getAsInteger();
    if (!dim || *dim < 0)
      return rejectProgramDirectory("expected non-negative shape dimension",
                                    diagnostics);
    shape.push_back(*dim);
  }
  return false;
}

} // namespace

bool readIntegerArrayField(const llvm::json::Object &object,
                           llvm::StringRef field, std::vector<int64_t> &values,
                           llvm::raw_ostream &diagnostics,
                           bool requirePositive) {
  const llvm::json::Array *array = object.getArray(field);
  if (!array)
    return rejectProgramDirectory("expected array field '" + field.str() + "'",
                                  diagnostics);

  for (const llvm::json::Value &value : *array) {
    std::optional<int64_t> entry = value.getAsInteger();
    if (!entry || *entry < 0 || (requirePositive && *entry == 0))
      return rejectProgramDirectory(
          "invalid integer array field '" + field.str() + "'", diagnostics);
    values.push_back(*entry);
  }
  return false;
}

namespace {

FailureOr<ProgramSignature> parseSignature(const llvm::json::Value &value,
                                           llvm::raw_ostream &diagnostics) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object) {
    rejectProgramDirectory("signature entries must be objects", diagnostics);
    return failure();
  }

  ProgramSignature signature;
  if (readShapeField(*object, signature.shape, diagnostics) ||
      readStringField(*object, "dtype", signature.dtype, diagnostics))
    return failure();
  return signature;
}

FailureOr<std::vector<ProgramSignature>>
parseSignatures(const llvm::json::Object &root, llvm::StringRef field,
                llvm::raw_ostream &diagnostics) {
  const llvm::json::Array *array = root.getArray(field);
  if (!array) {
    rejectProgramDirectory("expected array field '" + field.str() + "'",
                           diagnostics);
    return failure();
  }

  std::vector<ProgramSignature> signatures;
  signatures.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    FailureOr<ProgramSignature> signature = parseSignature(value, diagnostics);
    if (failed(signature))
      return failure();
    signatures.push_back(std::move(*signature));
  }
  return signatures;
}

FailureOr<std::vector<ProgramInputLocation>>
parseInputLocations(const llvm::json::Object &root,
                    llvm::raw_ostream &diagnostics) {
  const llvm::json::Array *array = root.getArray("input_locations");
  if (!array) {
    rejectProgramDirectory("expected array field 'input_locations'",
                           diagnostics);
    return failure();
  }

  std::vector<ProgramInputLocation> locations;
  locations.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object) {
      rejectProgramDirectory("input_locations entries must be objects",
                             diagnostics);
      return failure();
    }

    ProgramInputLocation location;
    if (readStringField(*object, "type_", location.type, diagnostics) ||
        readIntegerField(*object, "position", location.position, diagnostics) ||
        readStringField(*object, "name", location.name, diagnostics))
      return failure();
    locations.push_back(std::move(location));
  }
  return locations;
}

FailureOr<DistributedBoundaryPartition>
parseDistributedBoundaryPartition(const llvm::json::Value &value,
                                  llvm::raw_ostream &diagnostics) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object) {
    rejectProgramDirectory(
        "distributed boundary partition entries must be objects", diagnostics);
    return failure();
  }

  DistributedBoundaryPartition partition;
  if (readIntegerField(*object, "partition_id", partition.partitionId,
                       diagnostics) ||
      readIntegerField(*object, "replica_id", partition.replicaId,
                       diagnostics) ||
      readIntegerArrayField(*object, "offsets", partition.offsets,
                            diagnostics) ||
      readIntegerArrayField(*object, "sizes", partition.sizes, diagnostics) ||
      readIntegerArrayField(*object, "strides", partition.strides, diagnostics,
                            /*requirePositive=*/true))
    return failure();
  return partition;
}

FailureOr<DistributedBoundaryBinding>
parseDistributedBoundaryBinding(const llvm::json::Value &value,
                                llvm::StringRef indexField,
                                llvm::raw_ostream &diagnostics) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object) {
    rejectProgramDirectory("distributed boundary entries must be objects",
                           diagnostics);
    return failure();
  }

  DistributedBoundaryBinding binding;
  if (readIntegerField(*object, indexField, binding.index, diagnostics) ||
      readStringField(*object, "distribution", binding.distribution,
                      diagnostics) ||
      readIntegerArrayField(*object, "global_shape", binding.globalShape,
                            diagnostics) ||
      readIntegerArrayField(*object, "local_shape", binding.localShape,
                            diagnostics) ||
      readStringField(*object, "dtype", binding.dtype, diagnostics))
    return failure();

  const llvm::json::Array *partitions = object->getArray("partitions");
  if (!partitions) {
    rejectProgramDirectory(
        "expected array field 'partitions' in distributed boundary",
        diagnostics);
    return failure();
  }
  binding.partitions.reserve(partitions->size());
  for (const llvm::json::Value &partitionValue : *partitions) {
    FailureOr<DistributedBoundaryPartition> partition =
        parseDistributedBoundaryPartition(partitionValue, diagnostics);
    if (failed(partition))
      return failure();
    binding.partitions.push_back(std::move(*partition));
  }
  return binding;
}

FailureOr<std::vector<DistributedBoundaryBinding>>
parseDistributedBoundaryBindings(const llvm::json::Object &object,
                                 llvm::StringRef field,
                                 llvm::StringRef indexField,
                                 llvm::raw_ostream &diagnostics) {
  const llvm::json::Array *bindings = object.getArray(field);
  if (!bindings) {
    rejectProgramDirectory("expected array field '" + field.str() +
                               "' in distributed boundary",
                           diagnostics);
    return failure();
  }

  std::vector<DistributedBoundaryBinding> result;
  result.reserve(bindings->size());
  for (const llvm::json::Value &value : *bindings) {
    FailureOr<DistributedBoundaryBinding> binding =
        parseDistributedBoundaryBinding(value, indexField, diagnostics);
    if (failed(binding))
      return failure();
    result.push_back(std::move(*binding));
  }
  return result;
}

FailureOr<DistributedBoundary>
parseDistributedBoundary(const llvm::json::Object &object,
                         llvm::raw_ostream &diagnostics) {
  constexpr std::array<llvm::StringLiteral, 3> expectedFields = {
      "num_partitions", "inputs", "outputs"};
  for (const auto &field : object) {
    llvm::StringRef fieldName = field.first;
    if (!llvm::is_contained(expectedFields, fieldName)) {
      rejectProgramDirectory(
          "unexpected distributed_boundary field '" + fieldName.str() + "'",
          diagnostics);
      return failure();
    }
  }
  DistributedBoundary boundary;
  if (readIntegerField(object, "num_partitions", boundary.numPartitions,
                       diagnostics))
    return failure();
  FailureOr<std::vector<DistributedBoundaryBinding>> inputs =
      parseDistributedBoundaryBindings(object, "inputs", "argument_index",
                                       diagnostics);
  FailureOr<std::vector<DistributedBoundaryBinding>> outputs =
      parseDistributedBoundaryBindings(object, "outputs", "result_index",
                                       diagnostics);
  if (failed(inputs) || failed(outputs))
    return failure();
  boundary.inputs = std::move(*inputs);
  boundary.outputs = std::move(*outputs);
  return boundary;
}

} // namespace

FailureOr<ProgramMetadata>
parseProgramMetadata(llvm::StringRef metaPath, llvm::raw_ostream &diagnostics) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(metaPath);
  if (!bufferOrError) {
    rejectProgramDirectory("failed to read '" + metaPath.str() + "'",
                           diagnostics);
    return failure();
  }

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*bufferOrError)->getBuffer());
  if (!parsed) {
    std::string message = llvm::toString(parsed.takeError());
    rejectProgramDirectory("invalid JSON: " + message, diagnostics);
    return failure();
  }

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    rejectProgramDirectory("root must be an object", diagnostics);
    return failure();
  }

  ProgramMetadata meta;
  if (readStringField(*root, "name", meta.name, diagnostics))
    return failure();

  FailureOr<std::vector<ProgramSignature>> inputs =
      parseSignatures(*root, "input_signature", diagnostics);
  FailureOr<std::vector<ProgramSignature>> outputs =
      parseSignatures(*root, "output_signature", diagnostics);
  FailureOr<std::vector<ProgramInputLocation>> locations =
      parseInputLocations(*root, diagnostics);
  if (failed(inputs) || failed(outputs) || failed(locations))
    return failure();

  meta.inputSignatures = std::move(*inputs);
  meta.outputSignatures = std::move(*outputs);
  meta.inputLocations = std::move(*locations);
  if (const llvm::json::Value *boundaryValue =
          root->get("distributed_boundary")) {
    const llvm::json::Object *boundaryObject = boundaryValue->getAsObject();
    if (!boundaryObject) {
      rejectProgramDirectory("distributed_boundary must be an object",
                             diagnostics);
      return failure();
    }
    FailureOr<DistributedBoundary> boundary =
        parseDistributedBoundary(*boundaryObject, diagnostics);
    if (failed(boundary))
      return failure();
    meta.distributedBoundary = std::move(*boundary);
  }
  return meta;
}

namespace {

bool verifySignature(Type type, const ProgramSignature &signature,
                     llvm::StringRef kind, unsigned index,
                     llvm::raw_ostream &diagnostics) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape())
    return rejectProgramDirectory((kind + " " + Twine(index) +
                                   " must be a statically shaped ranked tensor")
                                      .str(),
                                  diagnostics);

  if (!llvm::equal(tensorType.getShape(), signature.shape))
    return rejectProgramDirectory(
        ("shape mismatch for " + kind + " " + Twine(index)).str(), diagnostics);

  std::string expectedDtype = dtypeString(tensorType.getElementType());
  std::string actualDtype = normalizeProgramDtype(signature.dtype);
  if (expectedDtype.empty() || expectedDtype != actualDtype)
    return rejectProgramDirectory(
        ("dtype mismatch for " + kind + " " + Twine(index)).str(), diagnostics);

  return false;
}

} // namespace

FailureOr<func::FuncOp> findSingleFunction(ModuleOp module,
                                           llvm::raw_ostream &diagnostics) {
  SmallVector<func::FuncOp> functions;
  SmallVector<func::FuncOp> publicFunctions;
  module.walk([&](func::FuncOp func) { functions.push_back(func); });
  for (func::FuncOp func : functions) {
    if (!func.isPrivate())
      publicFunctions.push_back(func);
  }

  if (publicFunctions.size() == 1)
    return publicFunctions.front();
  if (publicFunctions.empty() && functions.size() == 1)
    return functions.front();

  rejectProgramDirectory(
      "expected exactly one public entry func.func in StableHLO "
      "program directory MLIR",
      diagnostics);
  return failure();
}

namespace {

bool verifyParameterDataFile(llvm::StringRef programDir,
                             const ProgramInputLocation &location,
                             RankedTensorType tensorType,
                             llvm::raw_ostream &diagnostics,
                             const ProgramPayloadResolver *resolver) {
  if (location.name.empty())
    return rejectProgramDirectory("parameter location name must be non-empty",
                                  diagnostics);
  llvm::StringRef name(location.name);
  if (llvm::sys::path::is_absolute(name) || name == "." || name == ".." ||
      name.contains('/') || name.contains('\\'))
    return rejectProgramDirectory(
        "parameter location name must be a safe single path component",
        diagnostics);

  std::string relativePath = (Twine("data/") + Twine(location.name)).str();
  if (resolver) {
    const ProgramPayloadSource *source = resolver->resolve(relativePath);
    if (!source)
      return rejectProgramDirectory(
          "parameter payload is unavailable in the transaction: " +
              relativePath,
          diagnostics);
    return verifyNpyTensorPayloadFromSource(
        *source, relativePath, tensorType.getShape(),
        tensorType.getElementType(), diagnostics);
  }

  std::string path = programPath(programDir, {relativePath});
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return rejectProgramDirectory(
        "parameter data file is missing: " + location.name, diagnostics);
  if (!llvm::sys::fs::is_regular_file(status))
    return rejectProgramDirectory(
        "parameter data path is not a regular file: " + location.name,
        diagnostics);

  return verifyNpyTensorPayloadFile(
      path, relativePath, tensorType.getShape(), tensorType.getElementType(),
      diagnostics);
}

bool verifyConstantDataFile(llvm::StringRef programDir,
                            const ProgramInputLocation &location,
                            RankedTensorType tensorType,
                            llvm::raw_ostream &diagnostics,
                            const ProgramPayloadResolver *resolver) {
  if (location.position < 0)
    return rejectProgramDirectory("constant location has negative position",
                                  diagnostics);

  std::string relativePath =
      (Twine("constants/") + Twine(location.position)).str();
  if (resolver) {
    const ProgramPayloadSource *source = resolver->resolve(relativePath);
    if (!source)
      return rejectProgramDirectory(
          "constant payload is unavailable in the transaction: " +
              relativePath,
          diagnostics);
    return verifyNpyTensorPayloadFromSource(
        *source, relativePath, tensorType.getShape(),
        tensorType.getElementType(), diagnostics);
  }

  std::string path = programPath(programDir, {relativePath});
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return rejectProgramDirectory(
        "constant data file is missing: " + relativePath, diagnostics);
  if (!llvm::sys::fs::is_regular_file(status))
    return rejectProgramDirectory("constant data path is not a regular file: " +
                                      relativePath,
                                  diagnostics);

  return verifyNpyTensorPayloadFile(path, relativePath, tensorType.getShape(),
                                    tensorType.getElementType(), diagnostics);
}

} // namespace

bool verifyProgramMetadata(
    ModuleOp module, llvm::StringRef programDir, const ProgramMetadata &meta,
    llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result,
    const ProgramPayloadResolver *resolver) {
  if (meta.name != "forward")
    return rejectProgramDirectory("program metadata name must be 'forward'",
                                  diagnostics);
  FailureOr<func::FuncOp> func = findSingleFunction(module, diagnostics);
  if (failed(func))
    return true;

  FunctionType functionType = func->getFunctionType();
  if (meta.inputSignatures.size() != functionType.getNumInputs())
    return rejectProgramDirectory(
        "input_signature length does not match func.func inputs", diagnostics);
  if (meta.inputLocations.size() != functionType.getNumInputs())
    return rejectProgramDirectory(
        "input_locations length does not match func.func inputs", diagnostics);
  if (meta.outputSignatures.size() != functionType.getNumResults())
    return rejectProgramDirectory(
        "output_signature length does not match func.func results",
        diagnostics);

  unsigned parameterCount = 0;
  unsigned userInputCount = 0;
  unsigned constantCount = 0;
  llvm::SmallVector<int64_t, 8> userInputPositions;
  bool rejected = false;
  bool partitioned =
      hasSpmdParameterShardings(module) ||
      fileExists(programPath(programDir,
                             {"functions", "forward.parameter_shards.json"}));
  for (auto [index, signature] : llvm::enumerate(meta.inputSignatures)) {
    Type inputType = functionType.getInput(index);
    rejected |= verifySignature(inputType, signature, "function argument",
                                index, diagnostics);

    const ProgramInputLocation &location = meta.inputLocations[index];
    if (location.type == "parameter") {
      ++parameterCount;
      if (!partitioned) {
        if (auto tensorType = dyn_cast<RankedTensorType>(inputType))
          rejected |= verifyParameterDataFile(programDir, location, tensorType,
                                              diagnostics, resolver);
      }
    } else if (location.type == "constant") {
      ++constantCount;
      if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
        rejected |= verifyConstantDataFile(programDir, location, tensorType,
                                           diagnostics, resolver);
        if (result) {
          wafer::frontend::ProgramConstantBinding constant;
          constant.argumentIndex = index;
          constant.position = location.position;
          constant.shape.assign(tensorType.getShape().begin(),
                                tensorType.getShape().end());
          constant.dtype = dtypeString(tensorType.getElementType());
          constant.payloadPath =
              (Twine("constants/") + Twine(location.position)).str();
          result->constants.push_back(std::move(constant));
        }
      }
    } else if (location.type == "input_arg") {
      ++userInputCount;
      if (location.position < 0)
        rejected |= rejectProgramDirectory(
            ("input_arg location has negative position at function argument " +
             Twine(index))
                .str(),
            diagnostics);
      else
        userInputPositions.push_back(location.position);
    } else {
      rejected |= rejectProgramDirectory(
          ("unsupported input location type '" + location.type +
           "' at function argument " + Twine(index))
              .str(),
          diagnostics);
    }
  }

  for (auto [index, signature] : llvm::enumerate(meta.outputSignatures))
    rejected |= verifySignature(functionType.getResult(index), signature,
                                "function result", index, diagnostics);

  llvm::sort(userInputPositions);
  for (auto [expected, position] : llvm::enumerate(userInputPositions)) {
    if (position != static_cast<int64_t>(expected)) {
      rejected |= rejectProgramDirectory(
          "input_arg positions must be unique and contiguous from zero",
          diagnostics);
      break;
    }
  }

  if (!rejected && result) {
    result->programParameterCount = parameterCount;
    result->programUserInputCount = userInputCount;
    result->programConstantCount = constantCount;
  }
  return rejected;
}

} // namespace wafer::frontend::program_detail

namespace wafer::frontend {

llvm::Expected<std::vector<ProgramInputLocator>>
readProgramInputLocators(llvm::StringRef metaPath) {
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  FailureOr<program_detail::ProgramMetadata> meta =
      program_detail::parseProgramMetadata(metaPath, diagnostics);
  if (failed(meta))
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   diagnostics.str().c_str());
  std::vector<ProgramInputLocator> locators;
  locators.reserve(meta->inputLocations.size());
  for (const program_detail::ProgramInputLocation &location :
       meta->inputLocations)
    locators.push_back(
        {location.type, location.position, location.name});
  return locators;
}

} // namespace wafer::frontend
