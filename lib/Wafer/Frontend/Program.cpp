//===- Program.cpp - Wafer frontend program verifier ---------------------===//

#include "Wafer/Frontend/Program.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kDynamicBoundsAttr =
    "wafer.frontend.dynamic_bounds";

struct ProgramSignature {
  std::vector<int64_t> shape;
  std::string dtype;
};

struct ProgramInputLocation {
  std::string type;
  int64_t position = -1;
  std::string name;
};

struct ProgramMetadata {
  std::string name;
  std::vector<ProgramSignature> inputSignatures;
  std::vector<ProgramSignature> outputSignatures;
  std::vector<ProgramInputLocation> inputLocations;
};

struct VerifiedParameterShard {
  int64_t rank = -1;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct VerifiedParameterBoundaryShard {
  int64_t argumentIndex = -1;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::vector<VerifiedParameterShard> shards;
};

struct NpyPayloadMetadata {
  uint64_t fileSize = 0;
  uint64_t dataOffset = 0;
  std::string descr;
  bool fortranOrder = false;
  std::vector<int64_t> shape;
};

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-compile-stablehlo: " << message << "\n";
  return true;
}

bool rejectProgramDirectory(llvm::StringRef reason,
                            llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-compile-stablehlo: StableHLO program directory "
                 "metadata rejected: "
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

  diagnostics << "wafer-compile-stablehlo: " << message << " rejected";
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
  diagnostics << "wafer-compile-stablehlo: unbounded dynamic shape rejected "
              << "in func.func @" << func.getSymName();
  if (!kind.empty())
    diagnostics << " " << kind << " " << index;
  diagnostics << ": " << type << "\n";
  return true;
}

bool rejectInvalidDynamicBound(func::FuncOp func, llvm::StringRef kind,
                               unsigned index, llvm::StringRef reason,
                               llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-compile-stablehlo: invalid dynamic bound rejected "
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

bool readIntegerArrayField(const llvm::json::Object &object,
                           llvm::StringRef field, std::vector<int64_t> &values,
                           llvm::raw_ostream &diagnostics,
                           bool requirePositive = false) {
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

std::string normalizeProgramDtype(llvm::StringRef dtype) {
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

bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
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
    rejectProgramDirectory(
        ("failed to read npy payload file: " + displayName).str(), diagnostics);
    return failure();
  }

  llvm::StringRef bytes = (*bufferOrError)->getBuffer();
  if (bytes.size() < 10 ||
      bytes.take_front(6) != llvm::StringRef("\x93NUMPY", 6)) {
    rejectProgramDirectory(
        ("npy payload is missing magic: " + displayName).str(), diagnostics);
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
      rejectProgramDirectory(("truncated npy v2 header: " + displayName).str(),
                             diagnostics);
      return failure();
    }
    headerOffset = 12;
    headerLen = byte(8) | (byte(9) << 8) | (byte(10) << 16) | (byte(11) << 24);
  } else {
    rejectProgramDirectory(
        ("unsupported npy payload version: " + displayName).str(), diagnostics);
    return failure();
  }

  if (bytes.size() < headerOffset + headerLen) {
    rejectProgramDirectory(
        ("truncated npy payload header: " + displayName).str(), diagnostics);
    return failure();
  }

  llvm::StringRef header = bytes.slice(headerOffset, headerOffset + headerLen);
  std::optional<std::string> descr = parseNpyStringField(header, "descr");
  std::optional<bool> fortranOrder = parseNpyBoolField(header, "fortran_order");
  std::optional<std::vector<int64_t>> shape = parseNpyShapeField(header);
  if (!descr || !fortranOrder || !shape) {
    rejectProgramDirectory(("invalid npy payload header: " + displayName).str(),
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
    return rejectProgramDirectory(
        ("npy payload must be row-major: " + displayName).str(), diagnostics);
  if (!llvm::equal(metadata->shape, expectedShape))
    return rejectProgramDirectory(
        ("npy payload shape does not match tensor: " + displayName).str(),
        diagnostics);
  if (!npyDescrMatchesDtype(metadata->descr, elementType))
    return rejectProgramDirectory(
        ("npy payload dtype does not match tensor: " + displayName).str(),
        diagnostics);

  uint64_t expectedRawBytes = rawByteSize(expectedShape, elementType);
  if (expectedRawBytes == 0)
    return rejectProgramDirectory(
        ("npy tensor byte size is not representable: " + displayName).str(),
        diagnostics);
  if (metadata->dataOffset > metadata->fileSize ||
      expectedRawBytes > metadata->fileSize - metadata->dataOffset)
    return rejectProgramDirectory(
        ("npy payload file is smaller than tensor payload: " + displayName)
            .str(),
        diagnostics);

  return false;
}

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

std::string programPath(llvm::StringRef programDir,
                        llvm::ArrayRef<llvm::StringRef> components) {
  llvm::SmallString<256> path(programDir);
  for (llvm::StringRef component : components)
    llvm::sys::path::append(path, component);
  return path.str().str();
}

bool verifyParameterDataFile(llvm::StringRef programDir,
                             const ProgramInputLocation &location,
                             RankedTensorType tensorType,
                             llvm::raw_ostream &diagnostics) {
  if (location.name.empty())
    return rejectProgramDirectory("parameter location name must be non-empty",
                                  diagnostics);

  std::string path = programPath(programDir, {"data", location.name});
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return rejectProgramDirectory(
        "parameter data file is missing: " + location.name, diagnostics);
  if (!llvm::sys::fs::is_regular_file(status))
    return rejectProgramDirectory(
        "parameter data path is not a regular file: " + location.name,
        diagnostics);

  return verifyNpyTensorPayloadFile(
      path, ("data/" + Twine(location.name)).str(), tensorType.getShape(),
      tensorType.getElementType(), diagnostics);
}

bool hasSpmdParameterShardings(ModuleOp module);
bool fileExists(llvm::StringRef path);

bool verifyProgramMetadata(
    ModuleOp module, llvm::StringRef programDir, const ProgramMetadata &meta,
    llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result) {
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
                                              diagnostics);
      }
    } else if (location.type == "input_arg") {
      ++userInputCount;
      if (location.position < 0)
        rejected |= rejectProgramDirectory(
            ("input_arg location has negative position at function argument " +
             Twine(index))
                .str(),
            diagnostics);
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

  if (!rejected && result) {
    result->programParameterCount = parameterCount;
    result->programUserInputCount = userInputCount;
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
    rejectProgramDirectory("failed to read '" + path.str() + "'", diagnostics);
    return failure();
  }

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*bufferOrError)->getBuffer());
  if (!parsed) {
    std::string message = llvm::toString(parsed.takeError());
    rejectProgramDirectory("invalid JSON: " + message, diagnostics);
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
                      llvm::StringRef parameterName, llvm::StringRef programDir,
                      std::vector<bool> &seenRanks,
                      llvm::raw_ostream &diagnostics,
                      VerifiedParameterShard *verifiedShard) {
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
    return rejectProgramDirectory("parameter shard rank is out of range",
                                  diagnostics);
  if (seenRanks[rank])
    return rejectProgramDirectory("duplicate parameter shard rank",
                                  diagnostics);
  seenRanks[rank] = true;

  if (replicaId < 0)
    return rejectProgramDirectory(
        "parameter shard replica_id must be non-negative", diagnostics);

  size_t rankSize = globalShape.size();
  if (offsets.size() != rankSize || sizes.size() != rankSize ||
      strides.size() != rankSize)
    return rejectProgramDirectory(
        "parameter shard rank does not match global shape", diagnostics);

  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    int64_t size = sizes[dim];
    if (offset > globalShape[dim] || size > globalShape[dim] - offset)
      return rejectProgramDirectory(
          "parameter shard slice exceeds global shape", diagnostics);
    if (dim < localShape.size() && size > localShape[dim])
      return rejectProgramDirectory(
          "parameter shard size exceeds local tensor shape", diagnostics);
  }

  if (!isSafeRelativePath(file))
    return rejectProgramDirectory(
        "parameter shard file must be a safe relative path", diagnostics);
  std::string expectedPrefix =
      (Twine("parameter_shards/") + parameterName + "/").str();
  if (!llvm::StringRef(file).starts_with(expectedPrefix))
    return rejectProgramDirectory(
        "parameter shard file path does not match parameter", diagnostics);

  std::string path = programPath(programDir, {file});
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(path, status))
    return rejectProgramDirectory("parameter shard file is missing: " + file,
                                  diagnostics);
  if (!llvm::sys::fs::is_regular_file(status))
    return rejectProgramDirectory(
        "parameter shard path is not a regular file: " + file, diagnostics);

  if (verifyNpyTensorPayloadFile(path, file, sizes, elementType, diagnostics))
    return true;

  if (verifiedShard) {
    verifiedShard->rank = rank;
    verifiedShard->offsets = std::move(offsets);
    verifiedShard->sizes = std::move(sizes);
    verifiedShard->strides = std::move(strides);
  }
  return false;
}

bool verifyParameterBoundaryShard(
    const llvm::json::Object &object, const ProgramMetadata &meta,
    FunctionType functionType, int64_t logicalRankCount,
    std::vector<bool> &seenParameterArgs, llvm::StringRef programDir,
    llvm::raw_ostream &diagnostics, VerifiedParameterBoundaryShard *verified) {
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
    return rejectProgramDirectory(
        "parameter shard argument_index is out of range", diagnostics);

  const ProgramInputLocation &location = meta.inputLocations[argumentIndex];
  if (location.type != "parameter")
    return rejectProgramDirectory(
        "parameter shard argument_index does not refer to a "
        "parameter input",
        diagnostics);
  if (name != location.name)
    return rejectProgramDirectory(
        "parameter shard name does not match input location", diagnostics);

  Type inputType = functionType.getInput(argumentIndex);
  auto tensorType = dyn_cast<RankedTensorType>(inputType);
  if (!tensorType || !tensorType.hasStaticShape())
    return rejectProgramDirectory(
        "parameter shard argument must be a static ranked tensor", diagnostics);
  if (!llvm::equal(tensorType.getShape(), localShape))
    return rejectProgramDirectory(
        "parameter shard local_shape does not match func.func "
        "argument type",
        diagnostics);
  if (globalShape.size() != static_cast<size_t>(tensorType.getRank()))
    return rejectProgramDirectory(
        "parameter shard global_shape rank does not match "
        "func.func argument type",
        diagnostics);
  if (dtypeString(tensorType.getElementType()) != normalizeProgramDtype(dtype))
    return rejectProgramDirectory(
        "parameter shard dtype does not match func.func "
        "argument type",
        diagnostics);

  uint64_t expectedGlobalBytes =
      rawByteSize(globalShape, tensorType.getElementType());
  if (expectedGlobalBytes == 0)
    return rejectProgramDirectory(
        "parameter shard global tensor byte size is not "
        "representable",
        diagnostics);

  const llvm::json::Array *shards = object.getArray("shards");
  if (!shards)
    return rejectProgramDirectory("expected array field 'shards'", diagnostics);
  if (shards->size() != static_cast<size_t>(logicalRankCount))
    return rejectProgramDirectory("parameter shard count does not match "
                                  "logical_rank_count",
                                  diagnostics);

  std::vector<bool> seenRanks(logicalRankCount, false);
  std::vector<VerifiedParameterShard> verifiedShards;
  verifiedShards.reserve(shards->size());
  for (const llvm::json::Value &value : *shards) {
    const llvm::json::Object *shardObject = value.getAsObject();
    if (!shardObject)
      return rejectProgramDirectory("parameter shard entries must be objects",
                                    diagnostics);
    VerifiedParameterShard verifiedShard;
    if (verifyShardEntry(*shardObject, logicalRankCount, globalShape,
                         localShape, tensorType.getElementType(), name,
                         programDir, seenRanks, diagnostics, &verifiedShard))
      return true;
    verifiedShards.push_back(std::move(verifiedShard));
  }

  seenParameterArgs[argumentIndex] = true;
  if (verified) {
    verified->argumentIndex = argumentIndex;
    verified->globalShape = std::move(globalShape);
    verified->localShape = std::move(localShape);
    llvm::sort(verifiedShards, [](const VerifiedParameterShard &lhs,
                                  const VerifiedParameterShard &rhs) {
      return lhs.rank < rhs.rank;
    });
    verified->shards = std::move(verifiedShards);
  }
  return false;
}

bool denseArrayEquals(DenseI64ArrayAttr attr, llvm::ArrayRef<int64_t> values) {
  return llvm::equal(attr.asArrayRef(), values);
}

void flattenShardSlices(llvm::ArrayRef<VerifiedParameterShard> shards,
                        std::vector<int64_t> &ranks,
                        std::vector<int64_t> &offsets,
                        std::vector<int64_t> &sizes,
                        std::vector<int64_t> &strides) {
  for (const VerifiedParameterShard &shard : shards) {
    ranks.push_back(shard.rank);
    offsets.insert(offsets.end(), shard.offsets.begin(), shard.offsets.end());
    sizes.insert(sizes.end(), shard.sizes.begin(), shard.sizes.end());
    strides.insert(strides.end(), shard.strides.begin(), shard.strides.end());
  }
}

bool boundaryShardsMatch(wafer::BoundaryShardsOp op,
                         FlatSymbolRefAttr executionMesh,
                         const VerifiedParameterBoundaryShard &boundaryShard,
                         llvm::ArrayRef<int64_t> ranks,
                         llvm::ArrayRef<int64_t> offsets,
                         llvm::ArrayRef<int64_t> sizes,
                         llvm::ArrayRef<int64_t> strides) {
  return op.getExecutionMeshAttr() == executionMesh &&
         denseArrayEquals(op.getGlobalShapeAttr(), boundaryShard.globalShape) &&
         denseArrayEquals(op.getLocalShapeAttr(), boundaryShard.localShape) &&
         denseArrayEquals(op.getShardRanksAttr(), ranks) &&
         denseArrayEquals(op.getShardOffsetsAttr(), offsets) &&
         denseArrayEquals(op.getShardSizesAttr(), sizes) &&
         denseArrayEquals(op.getShardStridesAttr(), strides);
}

bool hasBoundaryShardFacts(ModuleOp module) {
  bool found = false;
  module.walk([&](wafer::BoundaryShardsOp) {
    found = true;
    return WalkResult::interrupt();
  });
  if (found)
    return true;

  module.walk([&](func::FuncOp funcOp) {
    auto function = cast<FunctionOpInterface>(funcOp.getOperation());
    for (unsigned index = 0, e = funcOp.getFunctionType().getNumInputs();
         index < e; ++index) {
      if (function.getArgAttr(index, wafer::kWaferBoundaryShardsAttrName)) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    for (unsigned index = 0, e = funcOp.getFunctionType().getNumResults();
         index < e; ++index) {
      if (function.getResultAttr(index, wafer::kWaferBoundaryShardsAttrName)) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

FailureOr<wafer::ExecutionMeshOp>
findBoundaryShardExecutionMesh(ModuleOp module,
                               llvm::raw_ostream &diagnostics) {
  if (auto defaultMesh =
          module.lookupSymbol<wafer::ExecutionMeshOp>("default_mesh"))
    return defaultMesh;

  wafer::ExecutionMeshOp foundMesh;
  bool multipleMeshes = false;
  module.walk([&](wafer::ExecutionMeshOp meshOp) {
    if (!foundMesh) {
      foundMesh = meshOp;
      return;
    }
    multipleMeshes = true;
  });
  if (multipleMeshes) {
    rejectProgramDirectory(
        "multiple execution meshes require @default_mesh for boundary shards",
        diagnostics);
    return failure();
  }
  if (!foundMesh) {
    rejectProgramDirectory(
        "parameter shard metadata requires wafer.execution.mesh", diagnostics);
    return failure();
  }
  return foundMesh;
}

FailureOr<int64_t> getExecutionMeshRankCount(wafer::ExecutionMeshOp meshOp,
                                             llvm::raw_ostream &diagnostics) {
  int64_t rankCount = 1;
  for (int64_t dim : meshOp.getShapeAttr().asArrayRef()) {
    int64_t next = 0;
    if (dim <= 0 || !checkedMul(rankCount, dim, next)) {
      rejectProgramDirectory("execution mesh rank count is invalid",
                             diagnostics);
      return failure();
    }
    rankCount = next;
  }
  return rankCount;
}

std::string makeBoundaryShardsSymbolName(ModuleOp module, StringRef funcName,
                                         int64_t argumentIndex) {
  std::string base =
      (funcName + "_arg" + llvm::Twine(argumentIndex) + "_shards").str();
  std::string candidate = base;
  unsigned suffix = 0;
  while (module.lookupSymbol(candidate)) {
    candidate = (base + "_" + llvm::Twine(++suffix)).str();
  }
  return candidate;
}

bool materializeParameterBoundaryShards(
    ModuleOp module, func::FuncOp func, int64_t logicalRankCount,
    llvm::ArrayRef<VerifiedParameterBoundaryShard> boundaryShards,
    llvm::raw_ostream &diagnostics, bool createMissingBoundaryShards) {
  FailureOr<wafer::ExecutionMeshOp> meshOp =
      findBoundaryShardExecutionMesh(module, diagnostics);
  if (failed(meshOp))
    return true;
  FailureOr<int64_t> meshRankCount =
      getExecutionMeshRankCount(*meshOp, diagnostics);
  if (failed(meshRankCount))
    return true;
  if (*meshRankCount != logicalRankCount)
    return rejectProgramDirectory(
        "parameter shard logical_rank_count does not match execution mesh "
        "rank count",
        diagnostics);

  OpBuilder builder(module.getContext());
  builder.setInsertionPointAfter(func);
  auto executionMeshAttr =
      FlatSymbolRefAttr::get(builder.getContext(), meshOp->getSymName());
  auto function = cast<FunctionOpInterface>(func.getOperation());

  for (const VerifiedParameterBoundaryShard &boundaryShard : boundaryShards) {
    std::vector<int64_t> ranks;
    std::vector<int64_t> offsets;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    flattenShardSlices(boundaryShard.shards, ranks, offsets, sizes, strides);

    auto existingAttr = dyn_cast_or_null<FlatSymbolRefAttr>(function.getArgAttr(
        boundaryShard.argumentIndex, wafer::kWaferBoundaryShardsAttrName));
    if (existingAttr) {
      auto existingShards =
          module.lookupSymbol<wafer::BoundaryShardsOp>(existingAttr.getValue());
      if (!existingShards)
        return rejectProgramDirectory(
            "existing boundary shard attribute references missing symbol",
            diagnostics);
      if (!boundaryShardsMatch(existingShards, executionMeshAttr, boundaryShard,
                               ranks, offsets, sizes, strides))
        return rejectProgramDirectory(
            "existing boundary shards do not match parameter shard metadata",
            diagnostics);
      continue;
    }

    if (!createMissingBoundaryShards)
      continue;

    std::string symbolName = makeBoundaryShardsSymbolName(
        module, func.getSymName(), boundaryShard.argumentIndex);
    function.setArgAttr(
        boundaryShard.argumentIndex, wafer::kWaferBoundaryShardsAttrName,
        FlatSymbolRefAttr::get(builder.getContext(), symbolName));

    builder.create<wafer::BoundaryShardsOp>(
        module.getLoc(), builder.getStringAttr(symbolName), executionMeshAttr,
        DenseI64ArrayAttr::get(builder.getContext(), boundaryShard.globalShape),
        DenseI64ArrayAttr::get(builder.getContext(), boundaryShard.localShape),
        DenseI64ArrayAttr::get(builder.getContext(), ranks),
        DenseI64ArrayAttr::get(builder.getContext(), offsets),
        DenseI64ArrayAttr::get(builder.getContext(), sizes),
        DenseI64ArrayAttr::get(builder.getContext(), strides));
  }

  return false;
}

bool verifyParameterBoundaryShards(
    ModuleOp module, llvm::StringRef programDir, const ProgramMetadata &meta,
    func::FuncOp func, llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result,
    bool materializeBoundaryShards) {
  std::string path =
      programPath(programDir, {"functions", "forward.parameter_shards.json"});
  if (!fileExists(path)) {
    if (hasSpmdParameterShardings(module))
      return rejectProgramDirectory(
          "partitioned StableHLO program directory is missing parameter "
          "boundary shards",
          diagnostics);
    return false;
  }

  FailureOr<llvm::json::Value> parsed = parseJsonFile(path, diagnostics);
  if (failed(parsed))
    return true;

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return rejectProgramDirectory(
        "parameter shard metadata root must be an object", diagnostics);

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
    return rejectProgramDirectory(
        "unsupported parameter shard metadata version", diagnostics);
  if (function != meta.name)
    return rejectProgramDirectory(
        "parameter shard function does not match program directory meta",
        diagnostics);
  if (logicalRankCount <= 0)
    return rejectProgramDirectory("logical_rank_count must be positive",
                                  diagnostics);

  const llvm::json::Array *parameters = root->getArray("parameters");
  if (!parameters)
    return rejectProgramDirectory("expected array field 'parameters'",
                                  diagnostics);

  FunctionType functionType = func.getFunctionType();
  std::vector<bool> seenParameterArgs(functionType.getNumInputs(), false);
  std::vector<VerifiedParameterBoundaryShard> verifiedBoundaryShards;
  unsigned boundaryShardCount = 0;
  for (const llvm::json::Value &value : *parameters) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return rejectProgramDirectory(
          "parameter shard metadata entries must be objects", diagnostics);
    VerifiedParameterBoundaryShard verifiedBoundaryShard;
    if (verifyParameterBoundaryShard(
            *object, meta, functionType, logicalRankCount, seenParameterArgs,
            programDir, diagnostics, &verifiedBoundaryShard))
      return true;
    verifiedBoundaryShards.push_back(std::move(verifiedBoundaryShard));
    ++boundaryShardCount;
  }

  for (auto [index, location] : llvm::enumerate(meta.inputLocations)) {
    if (location.type == "parameter" && !seenParameterArgs[index])
      return rejectProgramDirectory(
          "parameter input is missing boundary shard metadata: " +
              location.name,
          diagnostics);
  }

  if (result)
    result->programParameterBoundaryShardCount = boundaryShardCount;
  if ((materializeBoundaryShards || hasBoundaryShardFacts(module)) &&
      materializeParameterBoundaryShards(
          module, func, logicalRankCount, verifiedBoundaryShards, diagnostics,
          /*createMissingBoundaryShards=*/materializeBoundaryShards))
    return true;
  return false;
}

} // namespace

namespace wafer::frontend {

LogicalResult verifyFrontendProgram(ModuleOp module,
                                    llvm::raw_ostream &diagnostics,
                                    FrontendProgramVerificationResult *result) {
  if (result)
    *result = FrontendProgramVerificationResult{};

  bool rejected = false;
  rejected |= rejectImportMarker(module, "wafer.import.graph_break",
                                 "graph break", diagnostics);
  rejected |= rejectImportMarker(module, "wafer.import.eager_fallback",
                                 "eager fallback", diagnostics);
  rejected |= verifyFunctionBoundary(module, diagnostics);

  return rejected ? failure() : success();
}

static LogicalResult
verifyStableHLOProgramDirImpl(ModuleOp module, llvm::StringRef programPath,
                              llvm::raw_ostream &diagnostics,
                              FrontendProgramVerificationResult *result,
                              bool materializeBoundaryShards) {
  if (result)
    *result = FrontendProgramVerificationResult{};

  if (failed(verifyFrontendProgram(module, diagnostics, result)))
    return failure();

  std::string metaPath =
      ::programPath(programPath, {"functions", "forward.meta"});
  FailureOr<ProgramMetadata> meta = parseProgramMetadata(metaPath, diagnostics);
  if (failed(meta))
    return failure();

  bool rejected =
      verifyProgramMetadata(module, programPath, *meta, diagnostics, result);
  if (!rejected) {
    FailureOr<func::FuncOp> func = findSingleFunction(module, diagnostics);
    if (failed(func))
      return failure();
    rejected |= verifyParameterBoundaryShards(module, programPath, *meta, *func,
                                              diagnostics, result,
                                              materializeBoundaryShards);
  }
  if (!rejected && materializeBoundaryShards && failed(mlir::verify(module)))
    return failure();
  return rejected ? failure() : success();
}

LogicalResult
verifyStableHLOProgramDir(ModuleOp module, llvm::StringRef programPath,
                          llvm::raw_ostream &diagnostics,
                          FrontendProgramVerificationResult *result) {
  return verifyStableHLOProgramDirImpl(module, programPath, diagnostics, result,
                                       /*materializeBoundaryShards=*/false);
}

LogicalResult verifyAndMaterializeStableHLOProgramDir(
    ModuleOp module, llvm::StringRef programPath,
    llvm::raw_ostream &diagnostics, FrontendProgramVerificationResult *result) {
  return verifyStableHLOProgramDirImpl(module, programPath, diagnostics, result,
                                       /*materializeBoundaryShards=*/true);
}

} // namespace wafer::frontend
