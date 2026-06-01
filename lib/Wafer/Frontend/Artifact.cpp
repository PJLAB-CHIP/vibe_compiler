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
#include <optional>
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

struct NpyPayloadMetadata {
  uint64_t fileSize = 0;
  uint64_t dataOffset = 0;
  std::string descr;
  bool fortranOrder = false;
  std::vector<int64_t> shape;
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
    return rejectInvalidDynamicBound(
        func, kind, index, "rank does not match tensor type", diagnostics);

  for (auto [dim, boundAttr] : llvm::enumerate(bounds)) {
    auto bound = dyn_cast<IntegerAttr>(boundAttr);
    if (!bound)
      return rejectInvalidDynamicBound(
          func, kind, index, "bound entry is not an integer", diagnostics);

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
      rejected |=
          verifyDynamicBounds(func, "argument", index, type,
                              getFunctionArgAttrs(func, index), diagnostics);
    }
    for (auto [index, type] :
         llvm::enumerate(func.getFunctionType().getResults())) {
      rejected |=
          verifyDynamicBounds(func, "result", index, type,
                              getFunctionResultAttrs(func, index), diagnostics);
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

bool readIntegerArrayField(const llvm::json::Object &object,
                           llvm::StringRef field, std::vector<int64_t> &values,
                           llvm::raw_ostream &diagnostics,
                           bool requirePositive = false) {
  const llvm::json::Array *array = object.getArray(field);
  if (!array)
    return rejectBundle("expected array field '" + field.str() + "'",
                        diagnostics);

  for (const llvm::json::Value &value : *array) {
    std::optional<int64_t> entry = value.getAsInteger();
    if (!entry || *entry < 0 || (requirePositive && *entry == 0))
      return rejectBundle("invalid integer array field '" + field.str() + "'",
                          diagnostics);
    values.push_back(*entry);
  }
  return false;
}

FailureOr<BundleSignature> parseSignature(const llvm::json::Value &value,
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
        readIntegerField(*object, "position", location.position, diagnostics) ||
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

uint64_t rawByteSize(llvm::ArrayRef<int64_t> shape, Type elementType) {
  unsigned bitWidth =
      mlir::getElementTypeOrSelf(elementType).getIntOrFloatBitWidth();
  if (bitWidth % 8 != 0)
    return 0;
  uint64_t elements = 1;
  for (int64_t dim : shape) {
    if (dim < 0)
      return 0;
    elements *= static_cast<uint64_t>(dim);
  }
  return elements * (bitWidth / 8);
}

std::optional<llvm::StringRef> findNpyFieldValue(llvm::StringRef header,
                                                 llvm::StringRef field) {
  std::string singleQuoted = (Twine("'") + field + "'").str();
  std::string doubleQuoted = (Twine("\"") + field + "\"").str();
  size_t key = header.find(singleQuoted);
  if (key == llvm::StringRef::npos)
    key = header.find(doubleQuoted);
  if (key == llvm::StringRef::npos)
    return std::nullopt;

  size_t colon = header.find(':', key);
  if (colon == llvm::StringRef::npos)
    return std::nullopt;
  return header.drop_front(colon + 1).ltrim();
}

std::optional<std::string> parseNpyStringField(llvm::StringRef header,
                                               llvm::StringRef field) {
  std::optional<llvm::StringRef> value = findNpyFieldValue(header, field);
  if (!value || value->empty())
    return std::nullopt;
  char quote = value->front();
  if (quote != '\'' && quote != '"')
    return std::nullopt;
  llvm::StringRef rest = value->drop_front();
  size_t end = rest.find(quote);
  if (end == llvm::StringRef::npos)
    return std::nullopt;
  return rest.take_front(end).str();
}

std::optional<bool> parseNpyBoolField(llvm::StringRef header,
                                      llvm::StringRef field) {
  std::optional<llvm::StringRef> value = findNpyFieldValue(header, field);
  if (!value)
    return std::nullopt;
  if (value->starts_with("False") || value->starts_with("false"))
    return false;
  if (value->starts_with("True") || value->starts_with("true"))
    return true;
  return std::nullopt;
}

std::optional<std::vector<int64_t>> parseNpyShapeField(llvm::StringRef header) {
  std::optional<llvm::StringRef> value = findNpyFieldValue(header, "shape");
  if (!value)
    return std::nullopt;

  size_t open = value->find('(');
  size_t close = value->find(')');
  if (open == llvm::StringRef::npos || close == llvm::StringRef::npos ||
      close < open)
    return std::nullopt;

  std::vector<int64_t> shape;
  llvm::StringRef body = value->slice(open + 1, close);
  while (!body.empty()) {
    auto split = body.split(',');
    llvm::StringRef token = split.first.trim();
    if (!token.empty()) {
      int64_t dim = 0;
      if (token.getAsInteger(10, dim) || dim < 0)
        return std::nullopt;
      shape.push_back(dim);
    }
    body = split.second;
  }
  return shape;
}

FailureOr<NpyPayloadMetadata>
readNpyPayloadMetadata(llvm::StringRef path, llvm::StringRef displayName,
                       llvm::raw_ostream &diagnostics) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError) {
    rejectBundle(("failed to read npy payload file: " + displayName).str(),
                 diagnostics);
    return failure();
  }

  llvm::StringRef bytes = (*bufferOrError)->getBuffer();
  if (bytes.size() < 10 ||
      bytes.take_front(6) != llvm::StringRef("\x93NUMPY", 6)) {
    rejectBundle(("npy payload is missing magic: " + displayName).str(),
                 diagnostics);
    return failure();
  }

  auto byte = [&](size_t index) -> uint64_t {
    return static_cast<unsigned char>(bytes[index]);
  };

  uint64_t major = byte(6);
  uint64_t headerLen = 0;
  uint64_t headerOffset = 0;
  if (major == 1) {
    headerOffset = 10;
    headerLen = byte(8) | (byte(9) << 8);
  } else if (major == 2) {
    if (bytes.size() < 12) {
      rejectBundle(("truncated npy v2 header: " + displayName).str(),
                   diagnostics);
      return failure();
    }
    headerOffset = 12;
    headerLen = byte(8) | (byte(9) << 8) | (byte(10) << 16) |
                (byte(11) << 24);
  } else {
    rejectBundle(("unsupported npy payload version: " + displayName).str(),
                 diagnostics);
    return failure();
  }

  if (bytes.size() < headerOffset + headerLen) {
    rejectBundle(("truncated npy payload header: " + displayName).str(),
                 diagnostics);
    return failure();
  }

  llvm::StringRef header = bytes.slice(headerOffset, headerOffset + headerLen);
  std::optional<std::string> descr = parseNpyStringField(header, "descr");
  std::optional<bool> fortranOrder =
      parseNpyBoolField(header, "fortran_order");
  std::optional<std::vector<int64_t>> shape = parseNpyShapeField(header);
  if (!descr || !fortranOrder || !shape) {
    rejectBundle(("invalid npy payload header: " + displayName).str(),
                 diagnostics);
    return failure();
  }

  NpyPayloadMetadata metadata;
  metadata.fileSize = bytes.size();
  metadata.dataOffset = headerOffset + headerLen;
  metadata.descr = std::move(*descr);
  metadata.fortranOrder = *fortranOrder;
  metadata.shape = std::move(*shape);
  return metadata;
}

bool npyDescrMatchesDtype(llvm::StringRef descr, Type elementType) {
  std::string dtype = dtypeString(elementType);
  if (dtype == "f32")
    return descr == "<f4" || descr == "=f4";
  if (dtype == "f64")
    return descr == "<f8" || descr == "=f8";
  if (dtype == "f16")
    return descr == "<f2" || descr == "=f2";
  if (dtype == "bf16")
    return descr == "|V2";
  if (dtype == "i1")
    return descr == "|b1" || descr == "|i1";
  if (dtype == "i8")
    return descr == "|i1";
  if (dtype == "i16")
    return descr == "<i2" || descr == "=i2";
  if (dtype == "i32")
    return descr == "<i4" || descr == "=i4";
  if (dtype == "i64")
    return descr == "<i8" || descr == "=i8";
  return false;
}

bool verifyNpyTensorPayloadFile(llvm::StringRef path,
                                llvm::StringRef displayName,
                                llvm::ArrayRef<int64_t> expectedShape,
                                Type elementType,
                                llvm::raw_ostream &diagnostics) {
  FailureOr<NpyPayloadMetadata> metadata =
      readNpyPayloadMetadata(path, displayName, diagnostics);
  if (failed(metadata))
    return true;

  if (metadata->fortranOrder)
    return rejectBundle(("npy payload must be row-major: " + displayName).str(),
                        diagnostics);
  if (!llvm::equal(metadata->shape, expectedShape))
    return rejectBundle(
        ("npy payload shape does not match tensor: " + displayName).str(),
        diagnostics);
  if (!npyDescrMatchesDtype(metadata->descr, elementType))
    return rejectBundle(
        ("npy payload dtype does not match tensor: " + displayName).str(),
        diagnostics);

  uint64_t expectedRawBytes = rawByteSize(expectedShape, elementType);
  if (expectedRawBytes == 0)
    return rejectBundle(
        ("npy tensor byte size is not representable: " + displayName).str(),
        diagnostics);
  if (metadata->dataOffset > metadata->fileSize ||
      expectedRawBytes > metadata->fileSize - metadata->dataOffset)
    return rejectBundle(
        ("npy payload file is smaller than tensor payload: " + displayName)
            .str(),
        diagnostics);

  return false;
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
    return rejectBundle(
        ("shape mismatch for " + kind + " " + Twine(index)).str(), diagnostics);

  std::string expectedDtype = dtypeString(tensorType.getElementType());
  std::string actualDtype = normalizeBundleDtype(signature.dtype);
  if (expectedDtype.empty() || expectedDtype != actualDtype)
    return rejectBundle(
        ("dtype mismatch for " + kind + " " + Twine(index)).str(), diagnostics);

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

  return verifyNpyTensorPayloadFile(
      path, ("data/" + Twine(location.name)).str(), tensorType.getShape(),
      tensorType.getElementType(), diagnostics);
}

bool hasSpmdParameterShardings(ModuleOp module);
bool fileExists(llvm::StringRef path);

bool verifyBundleMeta(ModuleOp module, llvm::StringRef bundleDir,
                      const BundleMeta &meta, llvm::raw_ostream &diagnostics,
                      wafer::frontend::ArtifactVerificationResult *result) {
  FailureOr<func::FuncOp> func = findSingleFunction(module, diagnostics);
  if (failed(func))
    return true;

  FunctionType functionType = func->getFunctionType();
  if (meta.inputSignatures.size() != functionType.getNumInputs())
    return rejectBundle(
        "input_signature length does not match func.func inputs", diagnostics);
  if (meta.inputLocations.size() != functionType.getNumInputs())
    return rejectBundle(
        "input_locations length does not match func.func inputs", diagnostics);
  if (meta.outputSignatures.size() != functionType.getNumResults())
    return rejectBundle(
        "output_signature length does not match func.func results",
        diagnostics);

  unsigned parameterCount = 0;
  unsigned userInputCount = 0;
  bool rejected = false;
  bool partitioned =
      hasSpmdParameterShardings(module) ||
      fileExists(bundlePath(bundleDir,
                            {"functions", "forward.parameter_shards.json"}));
  for (auto [index, signature] : llvm::enumerate(meta.inputSignatures)) {
    Type inputType = functionType.getInput(index);
    rejected |= verifySignature(inputType, signature, "function argument",
                                index, diagnostics);

    const BundleInputLocation &location = meta.inputLocations[index];
    if (location.type == "parameter") {
      ++parameterCount;
      if (!partitioned) {
        if (auto tensorType = dyn_cast<RankedTensorType>(inputType))
          rejected |= verifyParameterDataFile(bundleDir, location, tensorType,
                                              diagnostics);
      }
    } else if (location.type == "input_arg") {
      ++userInputCount;
      if (location.position < 0)
        rejected |= rejectBundle(
            ("input_arg location has negative position at function argument " +
             Twine(index))
                .str(),
            diagnostics);
    } else {
      rejected |=
          rejectBundle(("unsupported input location type '" + location.type +
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

bool hasSpmdParameterShardings(ModuleOp module) {
  if (module->getAttr("mhlo.spmd_parameters_shardings"))
    return true;

  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getAttr("mhlo.spmd_parameters_shardings")) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

bool fileExists(llvm::StringRef path) {
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return false;
  return llvm::sys::fs::is_regular_file(status);
}

FailureOr<llvm::json::Value> parseJsonFile(llvm::StringRef path,
                                           llvm::raw_ostream &diagnostics) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError) {
    rejectBundle("failed to read '" + path.str() + "'", diagnostics);
    return failure();
  }

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*bufferOrError)->getBuffer());
  if (!parsed) {
    std::string message = llvm::toString(parsed.takeError());
    rejectBundle("invalid JSON: " + message, diagnostics);
    return failure();
  }
  return std::move(*parsed);
}

bool isSafeRelativePath(llvm::StringRef path) {
  return !path.empty() && !llvm::sys::path::is_absolute(path) &&
         !path.contains("..");
}

bool verifyShardEntry(const llvm::json::Object &object,
                      int64_t logicalRankCount,
                      llvm::ArrayRef<int64_t> globalShape,
                      llvm::ArrayRef<int64_t> localShape, Type elementType,
                      llvm::StringRef parameterName, llvm::StringRef bundleDir,
                      std::vector<bool> &seenRanks,
                      llvm::raw_ostream &diagnostics) {
  int64_t rank = -1;
  int64_t replicaId = -1;
  std::string file;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  if (readIntegerField(object, "rank", rank, diagnostics) ||
      readIntegerField(object, "replica_id", replicaId, diagnostics) ||
      readStringField(object, "file", file, diagnostics) ||
      readIntegerArrayField(object, "offsets", offsets, diagnostics) ||
      readIntegerArrayField(object, "sizes", sizes, diagnostics) ||
      readIntegerArrayField(object, "strides", strides, diagnostics,
                            /*requirePositive=*/true))
    return true;

  if (rank < 0 || rank >= logicalRankCount)
    return rejectBundle("parameter shard rank is out of range", diagnostics);
  if (seenRanks[rank])
    return rejectBundle("duplicate parameter shard rank", diagnostics);
  seenRanks[rank] = true;

  if (replicaId < 0)
    return rejectBundle("parameter shard replica_id must be non-negative",
                        diagnostics);

  size_t rankSize = globalShape.size();
  if (offsets.size() != rankSize || sizes.size() != rankSize ||
      strides.size() != rankSize)
    return rejectBundle("parameter shard rank does not match global shape",
                        diagnostics);

  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    int64_t size = sizes[dim];
    if (offset > globalShape[dim] || size > globalShape[dim] - offset)
      return rejectBundle("parameter shard slice exceeds global shape",
                          diagnostics);
    if (dim < localShape.size() && size > localShape[dim])
      return rejectBundle("parameter shard size exceeds local tensor shape",
                          diagnostics);
  }

  if (!isSafeRelativePath(file))
    return rejectBundle("parameter shard file must be a safe relative path",
                        diagnostics);
  std::string expectedPrefix =
      (Twine("parameter_shards/") + parameterName + "/").str();
  if (!llvm::StringRef(file).starts_with(expectedPrefix))
    return rejectBundle("parameter shard file path does not match parameter",
                        diagnostics);

  std::string path = bundlePath(bundleDir, {file});
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return rejectBundle("parameter shard file is missing: " + file,
                        diagnostics);
  if (!llvm::sys::fs::is_regular_file(status))
    return rejectBundle("parameter shard path is not a regular file: " + file,
                        diagnostics);

  return verifyNpyTensorPayloadFile(path, file, sizes, elementType, diagnostics);
}

bool verifyParameterShardBinding(const llvm::json::Object &object,
                                 const BundleMeta &meta,
                                 FunctionType functionType,
                                 int64_t logicalRankCount,
                                 std::vector<bool> &seenParameterArgs,
                                 llvm::StringRef bundleDir,
                                 llvm::raw_ostream &diagnostics) {
  int64_t argumentIndex = -1;
  std::string name;
  std::string dtype;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;

  if (readIntegerField(object, "argument_index", argumentIndex, diagnostics) ||
      readStringField(object, "name", name, diagnostics) ||
      readStringField(object, "dtype", dtype, diagnostics) ||
      readIntegerArrayField(object, "global_shape", globalShape, diagnostics) ||
      readIntegerArrayField(object, "local_shape", localShape, diagnostics))
    return true;

  if (argumentIndex < 0 ||
      argumentIndex >= static_cast<int64_t>(functionType.getNumInputs()))
    return rejectBundle("parameter shard argument_index is out of range",
                        diagnostics);

  const BundleInputLocation &location = meta.inputLocations[argumentIndex];
  if (location.type != "parameter")
    return rejectBundle("parameter shard argument_index does not refer to a "
                        "parameter input",
                        diagnostics);
  if (name != location.name)
    return rejectBundle("parameter shard name does not match input location",
                        diagnostics);

  Type inputType = functionType.getInput(argumentIndex);
  auto tensorType = dyn_cast<RankedTensorType>(inputType);
  if (!tensorType || !tensorType.hasStaticShape())
    return rejectBundle(
        "parameter shard argument must be a static ranked tensor", diagnostics);
  if (!llvm::equal(tensorType.getShape(), localShape))
    return rejectBundle("parameter shard local_shape does not match func.func "
                        "argument type",
                        diagnostics);
  if (globalShape.size() != static_cast<size_t>(tensorType.getRank()))
    return rejectBundle("parameter shard global_shape rank does not match "
                        "func.func argument type",
                        diagnostics);
  if (dtypeString(tensorType.getElementType()) != normalizeBundleDtype(dtype))
    return rejectBundle("parameter shard dtype does not match func.func "
                        "argument type",
                        diagnostics);

  uint64_t expectedGlobalBytes =
      rawByteSize(globalShape, tensorType.getElementType());
  if (expectedGlobalBytes == 0)
    return rejectBundle("parameter shard global tensor byte size is not "
                        "representable",
                        diagnostics);

  const llvm::json::Array *shards = object.getArray("shards");
  if (!shards)
    return rejectBundle("expected array field 'shards'", diagnostics);
  if (shards->size() != static_cast<size_t>(logicalRankCount))
    return rejectBundle("parameter shard count does not match "
                        "logical_rank_count",
                        diagnostics);

  std::vector<bool> seenRanks(logicalRankCount, false);
  for (const llvm::json::Value &value : *shards) {
    const llvm::json::Object *shardObject = value.getAsObject();
    if (!shardObject)
      return rejectBundle("parameter shard entries must be objects",
                          diagnostics);
    if (verifyShardEntry(*shardObject, logicalRankCount, globalShape,
                         localShape, tensorType.getElementType(), name,
                         bundleDir, seenRanks, diagnostics))
      return true;
  }

  seenParameterArgs[argumentIndex] = true;
  return false;
}

bool verifyParameterShardBindings(
    ModuleOp module, llvm::StringRef bundleDir, const BundleMeta &meta,
    FunctionType functionType, llvm::raw_ostream &diagnostics,
    wafer::frontend::ArtifactVerificationResult *result) {
  std::string path =
      bundlePath(bundleDir, {"functions", "forward.parameter_shards.json"});
  if (!fileExists(path)) {
    if (hasSpmdParameterShardings(module))
      return rejectBundle("partitioned StableHLO bundle is missing parameter "
                          "shard bindings",
                          diagnostics);
    return false;
  }

  FailureOr<llvm::json::Value> parsed = parseJsonFile(path, diagnostics);
  if (failed(parsed))
    return true;

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return rejectBundle("parameter shard binding root must be an object",
                        diagnostics);

  int64_t version = 0;
  std::string function;
  int64_t logicalRankCount = 0;
  if (readIntegerField(*root, "parameter_shards_version", version,
                       diagnostics) ||
      readStringField(*root, "function", function, diagnostics) ||
      readIntegerField(*root, "logical_rank_count", logicalRankCount,
                       diagnostics))
    return true;
  if (version != 2)
    return rejectBundle("unsupported parameter shard binding version",
                        diagnostics);
  if (function != meta.name)
    return rejectBundle("parameter shard function does not match bundle meta",
                        diagnostics);
  if (logicalRankCount <= 0)
    return rejectBundle("logical_rank_count must be positive", diagnostics);

  const llvm::json::Array *parameters = root->getArray("parameters");
  if (!parameters)
    return rejectBundle("expected array field 'parameters'", diagnostics);

  std::vector<bool> seenParameterArgs(functionType.getNumInputs(), false);
  unsigned bindingCount = 0;
  for (const llvm::json::Value &value : *parameters) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return rejectBundle("parameter shard binding entries must be objects",
                          diagnostics);
    if (verifyParameterShardBinding(*object, meta, functionType,
                                    logicalRankCount, seenParameterArgs,
                                    bundleDir, diagnostics))
      return true;
    ++bindingCount;
  }

  for (auto [index, location] : llvm::enumerate(meta.inputLocations)) {
    if (location.type == "parameter" && !seenParameterArgs[index])
      return rejectBundle("parameter input is missing shard binding: " +
                              location.name,
                          diagnostics);
  }

  if (result)
    result->bundleParameterShardBindingCount = bindingCount;
  return false;
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

  bool rejected =
      verifyBundleMeta(module, bundlePath, *meta, diagnostics, result);
  if (!rejected) {
    FailureOr<func::FuncOp> func = findSingleFunction(module, diagnostics);
    if (failed(func))
      return failure();
    rejected |= verifyParameterShardBindings(module, bundlePath, *meta,
                                             func->getFunctionType(),
                                             diagnostics, result);
  }
  return rejected ? failure() : success();
}

} // namespace wafer::frontend
