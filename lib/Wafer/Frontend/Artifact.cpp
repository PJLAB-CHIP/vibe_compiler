//===- Artifact.cpp - Wafer frontend artifact verifier -------------------===//

#include "Wafer/Frontend/Artifact.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kDynamicBoundsAttr =
    "wafer.frontend.dynamic_bounds";

struct BundleSignature {
  std::vector<int64_t> shape;
  std::string dtype;
};

struct BundleInputLocation {
  std::string type;
  int64_t position = -1;
  std::string name;
};

struct BundleMeta {
  std::string name;
  std::vector<BundleSignature> inputSignatures;
  std::vector<BundleSignature> outputSignatures;
  std::vector<BundleInputLocation> inputLocations;
};

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-import-model: " << message << "\n";
  return true;
}

bool rejectBundle(llvm::StringRef reason, llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-import-model: StableHLO bundle metadata rejected: "
              << reason << "\n";
  return true;
}

bool rejectImportMarker(ModuleOp module, llvm::StringRef attrName,
                        llvm::StringRef message,
                        llvm::raw_ostream &diagnostics) {
  Attribute marker = module->getAttr(attrName);
  if (!marker)
    return false;

  if (auto boolMarker = dyn_cast<BoolAttr>(marker);
      boolMarker && !boolMarker.getValue())
    return false;

  diagnostics << "wafer-import-model: " << message << " rejected";
  if (auto stringMarker = dyn_cast<StringAttr>(marker))
    diagnostics << ": " << stringMarker.getValue();
  else
    diagnostics << ": " << marker;
  diagnostics << "\n";
  return true;
}

DictionaryAttr getFunctionArgAttrs(func::FuncOp func, unsigned index) {
  auto function = cast<FunctionOpInterface>(func.getOperation());
  return function_interface_impl::getArgAttrDict(function, index);
}

DictionaryAttr getFunctionResultAttrs(func::FuncOp func, unsigned index) {
  auto function = cast<FunctionOpInterface>(func.getOperation());
  return function_interface_impl::getResultAttrDict(function, index);
}

bool rejectUnboundedDynamicShape(func::FuncOp func, llvm::StringRef kind,
                                 unsigned index, Type type,
                                 llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-import-model: unbounded dynamic shape rejected "
              << "in func.func @" << func.getSymName();
  if (!kind.empty())
    diagnostics << " " << kind << " " << index;
  diagnostics << ": " << type << "\n";
  return true;
}

bool rejectInvalidDynamicBound(func::FuncOp func, llvm::StringRef kind,
                               unsigned index, llvm::StringRef reason,
                               llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-import-model: invalid dynamic bound rejected "
              << "in func.func @" << func.getSymName() << " " << kind << " "
              << index << ": " << reason << "\n";
  return true;
}

bool verifyDynamicBounds(func::FuncOp func, llvm::StringRef kind,
                         unsigned index, Type type, DictionaryAttr attrs,
                         llvm::raw_ostream &diagnostics) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return false;

  if (shapedType.hasStaticShape())
    return false;

  if (!shapedType.hasRank())
    return rejectUnboundedDynamicShape(func, kind, index, type, diagnostics);

  Attribute boundsAttr = attrs ? attrs.get(kDynamicBoundsAttr) : Attribute();
  auto bounds = dyn_cast_or_null<ArrayAttr>(boundsAttr);
  if (!bounds)
    return rejectUnboundedDynamicShape(func, kind, index, type, diagnostics);

  ArrayRef<int64_t> shape = shapedType.getShape();
  if (bounds.size() != shape.size())
    return rejectInvalidDynamicBound(func, kind, index,
                                     "rank does not match tensor type",
                                     diagnostics);

  for (auto [dim, boundAttr] : llvm::enumerate(bounds)) {
    auto bound = dyn_cast<IntegerAttr>(boundAttr);
    if (!bound)
      return rejectInvalidDynamicBound(func, kind, index,
                                       "bound entry is not an integer",
                                       diagnostics);

    int64_t boundValue = bound.getInt();
    int64_t dimValue = shape[dim];
    if (ShapedType::isDynamic(dimValue)) {
      if (boundValue <= 0)
        return rejectInvalidDynamicBound(func, kind, index,
                                         "dynamic dimension bound must be "
                                         "positive",
                                         diagnostics);
      continue;
    }

    if (boundValue != dimValue)
      return rejectInvalidDynamicBound(func, kind, index,
                                       "static dimension bound must match "
                                       "the static tensor dimension",
                                       diagnostics);
  }

  return false;
}

bool verifyFunctionBoundary(ModuleOp module, llvm::raw_ostream &diagnostics) {
  bool rejected = false;
  module.walk([&](func::FuncOp func) {
    for (auto [index, type] :
         llvm::enumerate(func.getFunctionType().getInputs())) {
      rejected |= verifyDynamicBounds(func, "argument", index, type,
                                      getFunctionArgAttrs(func, index),
                                      diagnostics);
    }
    for (auto [index, type] :
         llvm::enumerate(func.getFunctionType().getResults())) {
      rejected |= verifyDynamicBounds(func, "result", index, type,
                                      getFunctionResultAttrs(func, index),
                                      diagnostics);
    }
  });
  return rejected;
}

bool readStringField(const llvm::json::Object &object, llvm::StringRef field,
                     std::string &out, llvm::raw_ostream &diagnostics) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return rejectBundle("expected string field '" + field.str() + "'",
                        diagnostics);
  out = value->str();
  return false;
}

bool readIntegerField(const llvm::json::Object &object, llvm::StringRef field,
                      int64_t &out, llvm::raw_ostream &diagnostics) {
  std::optional<int64_t> value = object.getInteger(field);
  if (!value)
    return rejectBundle("expected integer field '" + field.str() + "'",
                        diagnostics);
  out = *value;
  return false;
}

bool readShapeField(const llvm::json::Object &object,
                    std::vector<int64_t> &shape,
                    llvm::raw_ostream &diagnostics) {
  const llvm::json::Array *array = object.getArray("shape");
  if (!array)
    return rejectBundle("expected array field 'shape'", diagnostics);

  for (const llvm::json::Value &value : *array) {
    std::optional<int64_t> dim = value.getAsInteger();
    if (!dim || *dim < 0)
      return rejectBundle("expected non-negative shape dimension", diagnostics);
    shape.push_back(*dim);
  }
  return false;
}

FailureOr<BundleSignature>
parseSignature(const llvm::json::Value &value,
               llvm::raw_ostream &diagnostics) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object) {
    rejectBundle("signature entries must be objects", diagnostics);
    return failure();
  }

  BundleSignature signature;
  if (readShapeField(*object, signature.shape, diagnostics) ||
      readStringField(*object, "dtype", signature.dtype, diagnostics))
    return failure();
  return signature;
}

FailureOr<std::vector<BundleSignature>>
parseSignatures(const llvm::json::Object &root, llvm::StringRef field,
                llvm::raw_ostream &diagnostics) {
  const llvm::json::Array *array = root.getArray(field);
  if (!array) {
    rejectBundle("expected array field '" + field.str() + "'", diagnostics);
    return failure();
  }

  std::vector<BundleSignature> signatures;
  signatures.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    FailureOr<BundleSignature> signature = parseSignature(value, diagnostics);
    if (failed(signature))
      return failure();
    signatures.push_back(std::move(*signature));
  }
  return signatures;
}

FailureOr<std::vector<BundleInputLocation>>
parseInputLocations(const llvm::json::Object &root,
                    llvm::raw_ostream &diagnostics) {
  const llvm::json::Array *array = root.getArray("input_locations");
  if (!array) {
    rejectBundle("expected array field 'input_locations'", diagnostics);
    return failure();
  }

  std::vector<BundleInputLocation> locations;
  locations.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object) {
      rejectBundle("input_locations entries must be objects", diagnostics);
      return failure();
    }

    BundleInputLocation location;
    if (readStringField(*object, "type_", location.type, diagnostics) ||
        readIntegerField(*object, "position", location.position,
                         diagnostics) ||
        readStringField(*object, "name", location.name, diagnostics))
      return failure();
    locations.push_back(std::move(location));
  }
  return locations;
}

FailureOr<BundleMeta> parseBundleMeta(llvm::StringRef metaPath,
                                      llvm::raw_ostream &diagnostics) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(metaPath);
  if (!bufferOrError) {
    rejectBundle("failed to read '" + metaPath.str() + "'", diagnostics);
    return failure();
  }

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*bufferOrError)->getBuffer());
  if (!parsed) {
    std::string message = llvm::toString(parsed.takeError());
    rejectBundle("invalid JSON: " + message, diagnostics);
    return failure();
  }

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    rejectBundle("root must be an object", diagnostics);
    return failure();
  }

  BundleMeta meta;
  if (readStringField(*root, "name", meta.name, diagnostics))
    return failure();

  FailureOr<std::vector<BundleSignature>> inputs =
      parseSignatures(*root, "input_signature", diagnostics);
  FailureOr<std::vector<BundleSignature>> outputs =
      parseSignatures(*root, "output_signature", diagnostics);
  FailureOr<std::vector<BundleInputLocation>> locations =
      parseInputLocations(*root, diagnostics);
  if (failed(inputs) || failed(outputs) || failed(locations))
    return failure();

  meta.inputSignatures = std::move(*inputs);
  meta.outputSignatures = std::move(*outputs);
  meta.inputLocations = std::move(*locations);
  return meta;
}

std::string dtypeString(Type elementType) {
  if (isa<Float32Type>(elementType))
    return "f32";
  if (isa<Float16Type>(elementType))
    return "f16";
  if (isa<BFloat16Type>(elementType))
    return "bf16";
  if (isa<Float64Type>(elementType))
    return "f64";
  if (auto integer = dyn_cast<IntegerType>(elementType))
    return ("i" + Twine(integer.getWidth())).str();
  return "";
}

std::string normalizeBundleDtype(llvm::StringRef dtype) {
  if (dtype == "float32")
    return "f32";
  if (dtype == "float16")
    return "f16";
  if (dtype == "bfloat16")
    return "bf16";
  if (dtype == "float64")
    return "f64";
  if (dtype == "int8")
    return "i8";
  if (dtype == "int16")
    return "i16";
  if (dtype == "int32")
    return "i32";
  if (dtype == "int64")
    return "i64";
  return dtype.str();
}

uint64_t rawByteSize(RankedTensorType type) {
  unsigned bitWidth = type.getElementTypeBitWidth();
  if (bitWidth % 8 != 0)
    return 0;
  return static_cast<uint64_t>(type.getNumElements()) * (bitWidth / 8);
}

bool verifySignature(Type type, const BundleSignature &signature,
                     llvm::StringRef kind, unsigned index,
                     llvm::raw_ostream &diagnostics) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape())
    return rejectBundle((kind + " " + Twine(index) +
                         " must be a statically shaped ranked tensor")
                            .str(),
                        diagnostics);

  if (!llvm::equal(tensorType.getShape(), signature.shape))
    return rejectBundle(("shape mismatch for " + kind + " " + Twine(index))
                            .str(),
                        diagnostics);

  std::string expectedDtype = dtypeString(tensorType.getElementType());
  std::string actualDtype = normalizeBundleDtype(signature.dtype);
  if (expectedDtype.empty() || expectedDtype != actualDtype)
    return rejectBundle(("dtype mismatch for " + kind + " " + Twine(index))
                            .str(),
                        diagnostics);

  return false;
}

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

  rejectBundle("expected exactly one public entry func.func in StableHLO "
               "bundle MLIR",
               diagnostics);
  return failure();
}

std::string bundlePath(llvm::StringRef bundleDir,
                       llvm::ArrayRef<llvm::StringRef> components) {
  llvm::SmallString<256> path(bundleDir);
  for (llvm::StringRef component : components)
    llvm::sys::path::append(path, component);
  return path.str().str();
}

bool verifyParameterDataFile(llvm::StringRef bundleDir,
                             const BundleInputLocation &location,
                             RankedTensorType tensorType,
                             llvm::raw_ostream &diagnostics) {
  if (location.name.empty())
    return rejectBundle("parameter location name must be non-empty",
                        diagnostics);

  std::string path = bundlePath(bundleDir, {"data", location.name});
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return rejectBundle("parameter data file is missing: " + location.name,
                        diagnostics);
  if (!llvm::sys::fs::is_regular_file(status))
    return rejectBundle("parameter data path is not a regular file: " +
                            location.name,
                        diagnostics);

  uint64_t expectedRawBytes = rawByteSize(tensorType);
  if (expectedRawBytes == 0)
    return rejectBundle("parameter tensor byte size is not representable: " +
                            location.name,
                        diagnostics);
  if (status.getSize() < expectedRawBytes)
    return rejectBundle("parameter data file is smaller than tensor payload: " +
                            location.name,
                        diagnostics);
  return false;
}

bool verifyBundleMeta(ModuleOp module, llvm::StringRef bundleDir,
                      const BundleMeta &meta, llvm::raw_ostream &diagnostics,
                      wafer::frontend::ArtifactVerificationResult *result) {
  FailureOr<func::FuncOp> func = findSingleFunction(module, diagnostics);
  if (failed(func))
    return true;

  FunctionType functionType = func->getFunctionType();
  if (meta.inputSignatures.size() != functionType.getNumInputs())
    return rejectBundle("input_signature length does not match func.func inputs",
                        diagnostics);
  if (meta.inputLocations.size() != functionType.getNumInputs())
    return rejectBundle("input_locations length does not match func.func inputs",
                        diagnostics);
  if (meta.outputSignatures.size() != functionType.getNumResults())
    return rejectBundle(
        "output_signature length does not match func.func results", diagnostics);

  unsigned parameterCount = 0;
  unsigned userInputCount = 0;
  bool rejected = false;
  for (auto [index, signature] : llvm::enumerate(meta.inputSignatures)) {
    Type inputType = functionType.getInput(index);
    rejected |= verifySignature(inputType, signature, "function argument",
                                index, diagnostics);

    const BundleInputLocation &location = meta.inputLocations[index];
    if (location.type == "parameter") {
      ++parameterCount;
      if (auto tensorType = dyn_cast<RankedTensorType>(inputType))
        rejected |=
            verifyParameterDataFile(bundleDir, location, tensorType, diagnostics);
    } else if (location.type == "input_arg") {
      ++userInputCount;
      if (location.position < 0)
        rejected |= rejectBundle(
            ("input_arg location has negative position at function argument " +
             Twine(index))
                .str(),
            diagnostics);
    } else {
      rejected |= rejectBundle(
          ("unsupported input location type '" + location.type +
           "' at function argument " + Twine(index))
              .str(),
          diagnostics);
    }
  }

  for (auto [index, signature] : llvm::enumerate(meta.outputSignatures))
    rejected |= verifySignature(functionType.getResult(index), signature,
                                "function result", index, diagnostics);

  if (!rejected && result) {
    result->bundleParameterCount = parameterCount;
    result->bundleUserInputCount = userInputCount;
  }
  return rejected;
}

} // namespace

namespace wafer::frontend {

LogicalResult verifyFrontendArtifact(ModuleOp module,
                                     llvm::raw_ostream &diagnostics,
                                     ArtifactVerificationResult *result) {
  if (result)
    *result = ArtifactVerificationResult{};

  bool rejected = false;
  rejected |= rejectImportMarker(module, "wafer.import.graph_break",
                                 "graph break", diagnostics);
  rejected |= rejectImportMarker(module, "wafer.import.eager_fallback",
                                 "eager fallback", diagnostics);
  rejected |= verifyFunctionBoundary(module, diagnostics);

  return rejected ? failure() : success();
}

LogicalResult verifyStableHLOBundle(ModuleOp module, llvm::StringRef bundlePath,
                                    llvm::raw_ostream &diagnostics,
                                    ArtifactVerificationResult *result) {
  if (result)
    *result = ArtifactVerificationResult{};

  if (failed(verifyFrontendArtifact(module, diagnostics, result)))
    return failure();

  std::string metaPath =
      ::bundlePath(bundlePath, {"functions", "forward.meta"});
  FailureOr<BundleMeta> meta = parseBundleMeta(metaPath, diagnostics);
  if (failed(meta))
    return failure();

  bool rejected = verifyBundleMeta(module, bundlePath, *meta, diagnostics, result);
  return rejected ? failure() : success();
}

} // namespace wafer::frontend
