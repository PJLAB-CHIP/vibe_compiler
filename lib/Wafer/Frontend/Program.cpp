//===- Program.cpp - Wafer frontend program verifier ---------------------===//

#include "Wafer/Frontend/Program.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <memory>
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

struct DistributedBoundaryRank {
  int64_t rank = -1;
  int64_t replicaId = -1;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct DistributedBoundaryBinding {
  int64_t index = -1;
  std::string distribution;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;
  std::string dtype;
  std::vector<DistributedBoundaryRank> ranks;
};

struct DistributedBoundary {
  int64_t version = 0;
  int64_t logicalRankCount = 0;
  std::vector<DistributedBoundaryBinding> inputs;
  std::vector<DistributedBoundaryBinding> outputs;
};

struct ProgramMetadata {
  std::string name;
  std::vector<ProgramSignature> inputSignatures;
  std::vector<ProgramSignature> outputSignatures;
  std::vector<ProgramInputLocation> inputLocations;
  std::optional<DistributedBoundary> distributedBoundary;
};

struct NpyPayloadMetadata {
  uint64_t fileSize = 0;
  uint64_t dataOffset = 0;
  std::string descr;
  bool fortranOrder = false;
  std::vector<int64_t> shape;
};

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << message << "\n";
  return true;
}

bool rejectProgramDirectory(llvm::StringRef reason,
                            llvm::raw_ostream &diagnostics) {
  diagnostics << "program directory metadata rejected: " << reason << "\n";
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

  diagnostics << message << " rejected";
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
  diagnostics << "unbounded dynamic shape rejected "
              << "in func.func @" << func.getSymName();
  if (!kind.empty())
    diagnostics << " " << kind << " " << index;
  diagnostics << ": " << type << "\n";
  return true;
}

bool rejectInvalidDynamicBound(func::FuncOp func, llvm::StringRef kind,
                               unsigned index, llvm::StringRef reason,
                               llvm::raw_ostream &diagnostics) {
  diagnostics << "invalid dynamic bound rejected "
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

FailureOr<DistributedBoundaryRank>
parseDistributedBoundaryRank(const llvm::json::Value &value,
                             llvm::raw_ostream &diagnostics) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object) {
    rejectProgramDirectory("distributed boundary rank entries must be objects",
                           diagnostics);
    return failure();
  }

  DistributedBoundaryRank rank;
  if (readIntegerField(*object, "rank", rank.rank, diagnostics) ||
      readIntegerField(*object, "replica_id", rank.replicaId, diagnostics) ||
      readIntegerArrayField(*object, "offsets", rank.offsets, diagnostics) ||
      readIntegerArrayField(*object, "sizes", rank.sizes, diagnostics) ||
      readIntegerArrayField(*object, "strides", rank.strides, diagnostics,
                            /*requirePositive=*/true))
    return failure();
  return rank;
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

  const llvm::json::Array *ranks = object->getArray("ranks");
  if (!ranks) {
    rejectProgramDirectory(
        "expected array field 'ranks' in distributed boundary", diagnostics);
    return failure();
  }
  binding.ranks.reserve(ranks->size());
  for (const llvm::json::Value &rankValue : *ranks) {
    FailureOr<DistributedBoundaryRank> rank =
        parseDistributedBoundaryRank(rankValue, diagnostics);
    if (failed(rank))
      return failure();
    binding.ranks.push_back(std::move(*rank));
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
  DistributedBoundary boundary;
  if (readIntegerField(object, "version", boundary.version, diagnostics) ||
      readIntegerField(object, "logical_rank_count", boundary.logicalRankCount,
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

bool checkedMulUint64(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::optional<uint64_t> checkedRawByteSize(llvm::ArrayRef<int64_t> shape,
                                           Type elementType) {
  elementType = mlir::getElementTypeOrSelf(elementType);
  if (dtypeString(elementType).empty())
    return std::nullopt;

  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;

  // A rank-zero tensor is a scalar and therefore has one element. A static
  // zero dimension makes the tensor empty; zero is a representable byte size,
  // not an overflow sentinel.
  uint64_t elements = 1;
  for (int64_t dim : shape) {
    uint64_t next = 0;
    if (dim < 0 ||
        !checkedMulUint64(elements, static_cast<uint64_t>(dim), next))
      return std::nullopt;
    elements = next;
  }

  uint64_t bytes = 0;
  if (!checkedMulUint64(elements, bitWidth / 8, bytes))
    return std::nullopt;
  return bytes;
}

bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
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

std::optional<std::pair<llvm::StringRef, uint64_t>>
decodeNpyDescr(llvm::StringRef descr) {
  if (descr == "<f4" || descr == "=f4")
    return std::pair<llvm::StringRef, uint64_t>{"f32", 4};
  if (descr == "<f8" || descr == "=f8")
    return std::pair<llvm::StringRef, uint64_t>{"f64", 8};
  if (descr == "<f2" || descr == "=f2")
    return std::pair<llvm::StringRef, uint64_t>{"f16", 2};
  if (descr == "|V2")
    return std::pair<llvm::StringRef, uint64_t>{"bf16", 2};
  if (descr == "|b1")
    return std::pair<llvm::StringRef, uint64_t>{"i1", 1};
  if (descr == "|i1")
    return std::pair<llvm::StringRef, uint64_t>{"i8", 1};
  if (descr == "<i2" || descr == "=i2")
    return std::pair<llvm::StringRef, uint64_t>{"i16", 2};
  if (descr == "<i4" || descr == "=i4")
    return std::pair<llvm::StringRef, uint64_t>{"i32", 4};
  if (descr == "<i8" || descr == "=i8")
    return std::pair<llvm::StringRef, uint64_t>{"i64", 8};
  return std::nullopt;
}

bool npyDescrMatchesDtype(llvm::StringRef descr, Type elementType) {
  // NumPy's one-byte signed-integer descriptor was historically accepted for
  // i1 program payloads as well as i8. Preserve that verifier compatibility;
  // the generic payload loader decodes the unambiguous storage dtype as i8.
  if (elementType.isInteger(1) && descr == "|i1")
    return true;
  auto decoded = decodeNpyDescr(descr);
  return decoded && decoded->first == dtypeString(elementType);
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

  std::optional<uint64_t> expectedRawBytes =
      checkedRawByteSize(expectedShape, elementType);
  if (!expectedRawBytes)
    return rejectProgramDirectory(
        ("npy tensor byte size is not representable: " + displayName).str(),
        diagnostics);
  if (metadata->dataOffset > metadata->fileSize ||
      *expectedRawBytes > metadata->fileSize - metadata->dataOffset)
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
  llvm::StringRef name(location.name);
  if (llvm::sys::path::is_absolute(name) || name == "." || name == ".." ||
      name.contains('/') || name.contains('\\'))
    return rejectProgramDirectory(
        "parameter location name must be a safe single path component",
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

bool verifyConstantDataFile(llvm::StringRef programDir,
                            const ProgramInputLocation &location,
                            RankedTensorType tensorType,
                            llvm::raw_ostream &diagnostics) {
  if (location.position < 0)
    return rejectProgramDirectory("constant location has negative position",
                                  diagnostics);

  std::string relativePath =
      (Twine("constants/") + Twine(location.position)).str();
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

bool hasSpmdParameterShardings(ModuleOp module);
bool fileExists(llvm::StringRef path);

bool verifyProgramMetadata(
    ModuleOp module, llvm::StringRef programDir, const ProgramMetadata &meta,
    llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result) {
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
                                              diagnostics);
      }
    } else if (location.type == "constant") {
      ++constantCount;
      if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
        rejected |= verifyConstantDataFile(programDir, location, tensorType,
                                           diagnostics);
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

struct ParameterShardSlice {
  int64_t rank = -1;
  int64_t replicaId = -1;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::string path;
  std::string relativePath;
};

wafer::frontend::ProgramDistributionKind
getVerifiedDistributionKind(llvm::StringRef distribution) {
  return distribution == "partitioned"
             ? wafer::frontend::ProgramDistributionKind::Partitioned
             : wafer::frontend::ProgramDistributionKind::Replicated;
}

wafer::frontend::ProgramRankSlice
getVerifiedRankSlice(const DistributedBoundaryRank &rank) {
  return {rank.rank,  rank.replicaId, rank.offsets,
          rank.sizes, rank.strides,   {}};
}

wafer::frontend::ProgramRankSlice
getVerifiedRankSlice(const ParameterShardSlice &slice) {
  return {slice.rank,  slice.replicaId, slice.offsets,
          slice.sizes, slice.strides,   slice.relativePath};
}

bool verifyShardEntry(const llvm::json::Object &object,
                      int64_t logicalRankCount,
                      llvm::ArrayRef<int64_t> globalShape,
                      llvm::ArrayRef<int64_t> localShape, Type elementType,
                      llvm::StringRef parameterName, llvm::StringRef programDir,
                      llvm::StringRef distribution,
                      std::vector<bool> &seenRanks,
                      std::vector<bool> &seenReplicaIds,
                      std::vector<ParameterShardSlice> &verifiedSlices,
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
    return rejectProgramDirectory("parameter shard rank is out of range",
                                  diagnostics);
  if (seenRanks[rank])
    return rejectProgramDirectory("duplicate parameter shard rank",
                                  diagnostics);
  seenRanks[rank] = true;

  if (replicaId < 0)
    return rejectProgramDirectory(
        "parameter shard replica_id must be non-negative", diagnostics);
  if (distribution == "partitioned" && replicaId != 0)
    return rejectProgramDirectory(
        "partitioned parameter shard replica_id must be 0", diagnostics);
  if (distribution == "replicated") {
    if (replicaId >= logicalRankCount)
      return rejectProgramDirectory(
          "replicated parameter shard replica_id is out of range", diagnostics);
    if (seenReplicaIds[replicaId])
      return rejectProgramDirectory(
          "duplicate replicated parameter shard replica_id", diagnostics);
    seenReplicaIds[replicaId] = true;
  }

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
    if (strides[dim] != 1)
      return rejectProgramDirectory(
          "parameter shard stride must be 1 for coverage verification",
          diagnostics);
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
  verifiedSlices.push_back(ParameterShardSlice{
      rank, replicaId, std::move(offsets), std::move(sizes), std::move(strides),
      std::move(path), std::move(file)});
  return false;
}

bool verifyParameterShardCoverage(llvm::ArrayRef<ParameterShardSlice> slices,
                                  llvm::ArrayRef<int64_t> globalShape,
                                  llvm::StringRef distribution,
                                  llvm::raw_ostream &diagnostics) {
  if (distribution == "replicated") {
    std::unique_ptr<llvm::MemoryBuffer> canonicalPayload;
    for (const ParameterShardSlice &slice : slices) {
      if (!llvm::all_of(slice.offsets,
                        [](int64_t offset) { return offset == 0; }) ||
          !llvm::equal(slice.sizes, globalShape))
        return rejectProgramDirectory(
            "replicated parameter shard must contain the full global tensor",
            diagnostics);

      auto payload = llvm::MemoryBuffer::getFile(slice.path);
      if (!payload)
        return rejectProgramDirectory(
            "failed to reread replicated parameter shard payload", diagnostics);
      if (!canonicalPayload) {
        canonicalPayload = std::move(*payload);
      } else if (canonicalPayload->getBuffer() != (*payload)->getBuffer()) {
        return rejectProgramDirectory(
            "replicated parameter shard payloads must be byte-identical",
            diagnostics);
      }
    }
    return false;
  }

  if (distribution != "partitioned")
    return rejectProgramDirectory(
        "parameter shard distribution must be 'replicated' or 'partitioned'",
        diagnostics);

  int64_t globalElements = 1;
  for (int64_t dim : globalShape) {
    if (!checkedMul(globalElements, dim, globalElements))
      return rejectProgramDirectory(
          "parameter shard global coverage size overflows int64", diagnostics);
  }

  int64_t coveredElements = 0;
  for (const ParameterShardSlice &slice : slices) {
    int64_t sliceElements = 1;
    for (int64_t size : slice.sizes) {
      if (!checkedMul(sliceElements, size, sliceElements))
        return rejectProgramDirectory(
            "parameter shard coverage size overflows int64", diagnostics);
    }
    if (!checkedAdd(coveredElements, sliceElements, coveredElements))
      return rejectProgramDirectory(
          "parameter shard coverage size overflows int64", diagnostics);
  }

  for (size_t lhsIndex = 0; lhsIndex < slices.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1; rhsIndex < slices.size(); ++rhsIndex) {
      const ParameterShardSlice &lhs = slices[lhsIndex];
      const ParameterShardSlice &rhs = slices[rhsIndex];
      bool overlaps = true;
      for (size_t dim = 0; dim < globalShape.size(); ++dim) {
        int64_t lhsEnd = lhs.offsets[dim] + lhs.sizes[dim];
        int64_t rhsEnd = rhs.offsets[dim] + rhs.sizes[dim];
        if (lhsEnd <= rhs.offsets[dim] || rhsEnd <= lhs.offsets[dim]) {
          overlaps = false;
          break;
        }
      }
      if (overlaps)
        return rejectProgramDirectory("parameter shard slices overlap",
                                      diagnostics);
    }
  }

  if (coveredElements != globalElements)
    return rejectProgramDirectory(
        "parameter shard slices do not cover the global tensor", diagnostics);
  return false;
}

bool verifyParameterShardMetadata(
    const llvm::json::Object &object, const ProgramMetadata &meta,
    FunctionType functionType, int64_t logicalRankCount,
    std::vector<bool> &seenParameterArgs, llvm::StringRef programDir,
    llvm::raw_ostream &diagnostics,
    wafer::frontend::ProgramParameterBinding *verifiedBinding) {
  int64_t argumentIndex = -1;
  std::string name;
  std::string dtype;
  std::string distribution;
  std::vector<int64_t> globalShape;
  std::vector<int64_t> localShape;

  if (readIntegerField(object, "argument_index", argumentIndex, diagnostics) ||
      readStringField(object, "name", name, diagnostics) ||
      readStringField(object, "dtype", dtype, diagnostics) ||
      readStringField(object, "distribution", distribution, diagnostics) ||
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
  if (seenParameterArgs[argumentIndex])
    return rejectProgramDirectory(
        "duplicate parameter shard metadata record for parameter argument",
        diagnostics);

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

  std::optional<uint64_t> expectedGlobalBytes =
      checkedRawByteSize(globalShape, tensorType.getElementType());
  if (!expectedGlobalBytes)
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
  std::vector<bool> seenReplicaIds(logicalRankCount, false);
  std::vector<ParameterShardSlice> verifiedSlices;
  verifiedSlices.reserve(shards->size());
  for (const llvm::json::Value &value : *shards) {
    const llvm::json::Object *shardObject = value.getAsObject();
    if (!shardObject)
      return rejectProgramDirectory("parameter shard entries must be objects",
                                    diagnostics);
    if (verifyShardEntry(*shardObject, logicalRankCount, globalShape,
                         localShape, tensorType.getElementType(), name,
                         programDir, distribution, seenRanks, seenReplicaIds,
                         verifiedSlices, diagnostics))
      return true;
  }
  if (distribution == "replicated" &&
      llvm::any_of(seenReplicaIds, [](bool seen) { return !seen; }))
    return rejectProgramDirectory(
        "replicated parameter shard replica_id domain is incomplete",
        diagnostics);
  if (verifyParameterShardCoverage(verifiedSlices, globalShape, distribution,
                                   diagnostics))
    return true;

  seenParameterArgs[argumentIndex] = true;
  if (verifiedBinding) {
    verifiedBinding->argumentIndex = argumentIndex;
    verifiedBinding->name = std::move(name);
    verifiedBinding->distribution = getVerifiedDistributionKind(distribution);
    verifiedBinding->globalShape = std::move(globalShape);
    verifiedBinding->localShape = std::move(localShape);
    verifiedBinding->dtype = normalizeProgramDtype(dtype);
    verifiedBinding->rankSlices.reserve(verifiedSlices.size());
    for (const ParameterShardSlice &slice : verifiedSlices)
      verifiedBinding->rankSlices.push_back(getVerifiedRankSlice(slice));
  }
  return false;
}

FailureOr<wafer::ExecutionMeshOp>
findParameterShardExecutionMesh(ModuleOp module,
                                llvm::raw_ostream &diagnostics) {
  SmallVector<wafer::ExecutionMeshOp, 2> meshes;
  for (wafer::ExecutionMeshOp mesh : module.getOps<wafer::ExecutionMeshOp>())
    meshes.push_back(mesh);
  bool nestedMesh = false;
  module.walk([&](wafer::ExecutionMeshOp mesh) {
    if (mesh->getParentOp() != module.getOperation()) {
      nestedMesh = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (nestedMesh) {
    rejectProgramDirectory(
        "post-SPMD execution mesh must be a direct module member", diagnostics);
    return failure();
  }
  if (meshes.size() != 1) {
    rejectProgramDirectory(
        "post-SPMD metadata requires exactly one wafer.execution.mesh",
        diagnostics);
    return failure();
  }
  return meshes.front();
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

bool verifyDistributedBoundaryCoverage(
    llvm::ArrayRef<DistributedBoundaryRank> ranks,
    llvm::ArrayRef<int64_t> globalShape, llvm::StringRef distribution,
    llvm::raw_ostream &diagnostics) {
  if (distribution == "replicated") {
    for (const DistributedBoundaryRank &rank : ranks) {
      if (!llvm::all_of(rank.offsets,
                        [](int64_t offset) { return offset == 0; }) ||
          !llvm::equal(rank.sizes, globalShape))
        return rejectProgramDirectory(
            "replicated distributed boundary rank must cover the full global "
            "tensor",
            diagnostics);
    }
    return false;
  }
  if (distribution != "partitioned")
    return rejectProgramDirectory(
        "distributed boundary distribution must be 'replicated' or "
        "'partitioned'",
        diagnostics);

  uint64_t globalElements = 1;
  for (int64_t dim : globalShape) {
    uint64_t next = 0;
    if (!checkedMulUint64(globalElements, static_cast<uint64_t>(dim), next))
      return rejectProgramDirectory(
          "distributed boundary global coverage size overflows uint64",
          diagnostics);
    globalElements = next;
  }

  uint64_t coveredElements = 0;
  for (const DistributedBoundaryRank &rank : ranks) {
    uint64_t rankElements = 1;
    for (int64_t size : rank.sizes) {
      uint64_t next = 0;
      if (!checkedMulUint64(rankElements, static_cast<uint64_t>(size), next))
        return rejectProgramDirectory(
            "distributed boundary rank coverage size overflows uint64",
            diagnostics);
      rankElements = next;
    }
    if (rankElements > std::numeric_limits<uint64_t>::max() - coveredElements)
      return rejectProgramDirectory(
          "distributed boundary coverage size overflows uint64", diagnostics);
    coveredElements += rankElements;
  }

  for (size_t lhsIndex = 0; lhsIndex < ranks.size(); ++lhsIndex) {
    for (size_t rhsIndex = lhsIndex + 1; rhsIndex < ranks.size(); ++rhsIndex) {
      const DistributedBoundaryRank &lhs = ranks[lhsIndex];
      const DistributedBoundaryRank &rhs = ranks[rhsIndex];
      bool overlaps = true;
      for (size_t dim = 0; dim < globalShape.size(); ++dim) {
        int64_t lhsEnd = lhs.offsets[dim] + lhs.sizes[dim];
        int64_t rhsEnd = rhs.offsets[dim] + rhs.sizes[dim];
        if (lhsEnd <= rhs.offsets[dim] || rhsEnd <= lhs.offsets[dim]) {
          overlaps = false;
          break;
        }
      }
      if (overlaps)
        return rejectProgramDirectory(
            "distributed boundary rank slices overlap", diagnostics);
    }
  }
  if (coveredElements != globalElements)
    return rejectProgramDirectory(
        "distributed boundary rank slices do not cover the global tensor",
        diagnostics);
  return false;
}

bool verifyDistributedBoundaryBinding(const DistributedBoundaryBinding &binding,
                                      RankedTensorType tensorType,
                                      const ProgramSignature &signature,
                                      int64_t logicalRankCount,
                                      llvm::raw_ostream &diagnostics) {
  if (!tensorType.hasStaticShape())
    return rejectProgramDirectory(
        "distributed boundary must refer to a static ranked tensor",
        diagnostics);
  if (!llvm::equal(binding.localShape, tensorType.getShape()) ||
      !llvm::equal(binding.localShape, signature.shape))
    return rejectProgramDirectory(
        "distributed boundary local_shape does not match module and metadata",
        diagnostics);
  if (binding.globalShape.size() != static_cast<size_t>(tensorType.getRank()))
    return rejectProgramDirectory(
        "distributed boundary global_shape rank does not match local tensor",
        diagnostics);

  std::string elementDtype = dtypeString(tensorType.getElementType());
  if (elementDtype.empty() ||
      normalizeProgramDtype(binding.dtype) != elementDtype ||
      normalizeProgramDtype(signature.dtype) != elementDtype)
    return rejectProgramDirectory(
        "distributed boundary dtype does not match module and metadata",
        diagnostics);
  if (binding.distribution != "replicated" &&
      binding.distribution != "partitioned")
    return rejectProgramDirectory(
        "distributed boundary distribution must be 'replicated' or "
        "'partitioned'",
        diagnostics);
  if (binding.ranks.size() != static_cast<size_t>(logicalRankCount))
    return rejectProgramDirectory(
        "distributed boundary rank count does not match logical_rank_count",
        diagnostics);

  std::vector<bool> seenRanks(logicalRankCount, false);
  std::vector<bool> seenReplicaIds(logicalRankCount, false);
  std::vector<int64_t> maximumSizes(binding.globalShape.size(), 0);
  for (const DistributedBoundaryRank &rank : binding.ranks) {
    if (rank.rank < 0 || rank.rank >= logicalRankCount)
      return rejectProgramDirectory("distributed boundary rank is out of range",
                                    diagnostics);
    if (seenRanks[rank.rank])
      return rejectProgramDirectory("duplicate distributed boundary rank",
                                    diagnostics);
    seenRanks[rank.rank] = true;

    if (binding.distribution == "partitioned") {
      if (rank.replicaId != 0)
        return rejectProgramDirectory(
            "partitioned distributed boundary replica_id must be 0",
            diagnostics);
    } else {
      if (rank.replicaId < 0 || rank.replicaId >= logicalRankCount)
        return rejectProgramDirectory(
            "replicated distributed boundary replica_id is out of range",
            diagnostics);
      if (seenReplicaIds[rank.replicaId])
        return rejectProgramDirectory(
            "duplicate replicated distributed boundary replica_id",
            diagnostics);
      seenReplicaIds[rank.replicaId] = true;
    }

    size_t tensorRank = binding.globalShape.size();
    if (rank.offsets.size() != tensorRank || rank.sizes.size() != tensorRank ||
        rank.strides.size() != tensorRank)
      return rejectProgramDirectory(
          "distributed boundary slice rank does not match tensor rank",
          diagnostics);
    for (auto [dim, offset] : llvm::enumerate(rank.offsets)) {
      int64_t size = rank.sizes[dim];
      if (offset > binding.globalShape[dim] ||
          size > binding.globalShape[dim] - offset)
        return rejectProgramDirectory(
            "distributed boundary slice exceeds global_shape", diagnostics);
      if (size > binding.localShape[dim])
        return rejectProgramDirectory(
            "distributed boundary slice exceeds local_shape", diagnostics);
      if (rank.strides[dim] != 1)
        return rejectProgramDirectory("distributed boundary stride must be 1",
                                      diagnostics);
      maximumSizes[dim] = std::max(maximumSizes[dim], size);
    }
  }

  if (llvm::any_of(seenRanks, [](bool seen) { return !seen; }))
    return rejectProgramDirectory(
        "distributed boundary logical rank domain is incomplete", diagnostics);
  if (binding.distribution == "replicated") {
    if (binding.globalShape != binding.localShape)
      return rejectProgramDirectory(
          "replicated distributed boundary global_shape and local_shape must "
          "match",
          diagnostics);
    if (llvm::any_of(seenReplicaIds, [](bool seen) { return !seen; }))
      return rejectProgramDirectory(
          "replicated distributed boundary replica_id domain is incomplete",
          diagnostics);
  } else if (maximumSizes != binding.localShape) {
    return rejectProgramDirectory(
        "partitioned distributed boundary slices do not explain local_shape",
        diagnostics);
  }
  return verifyDistributedBoundaryCoverage(binding.ranks, binding.globalShape,
                                           binding.distribution, diagnostics);
}

bool verifyDistributedBoundary(
    ModuleOp module, const ProgramMetadata &meta, func::FuncOp func,
    bool postSpmdMarker, llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result) {
  if (!postSpmdMarker) {
    if (meta.distributedBoundary)
      return rejectProgramDirectory(
          "distributed_boundary requires a post-SPMD marker", diagnostics);
    return false;
  }
  if (!meta.distributedBoundary)
    return rejectProgramDirectory(
        "post-SPMD program directory is missing distributed_boundary",
        diagnostics);

  const DistributedBoundary &boundary = *meta.distributedBoundary;
  if (boundary.version != 1)
    return rejectProgramDirectory("unsupported distributed_boundary version",
                                  diagnostics);
  if (boundary.logicalRankCount != 1 && boundary.logicalRankCount != 16)
    return rejectProgramDirectory(
        "distributed_boundary logical_rank_count must be 1 or 16", diagnostics);

  FailureOr<wafer::ExecutionMeshOp> mesh =
      findParameterShardExecutionMesh(module, diagnostics);
  if (failed(mesh))
    return true;
  FailureOr<int64_t> meshRankCount =
      getExecutionMeshRankCount(*mesh, diagnostics);
  if (failed(meshRankCount))
    return true;
  if (*meshRankCount != boundary.logicalRankCount)
    return rejectProgramDirectory(
        "distributed_boundary logical_rank_count does not match execution "
        "mesh rank count",
        diagnostics);

  FunctionType functionType = func.getFunctionType();
  size_t expectedInputCount = llvm::count_if(
      meta.inputLocations, [](const ProgramInputLocation &location) {
        return location.type == "input_arg";
      });
  if (boundary.inputs.size() != expectedInputCount)
    return rejectProgramDirectory(
        "distributed_boundary inputs do not exactly cover input_arg "
        "locations",
        diagnostics);
  if (boundary.outputs.size() != functionType.getNumResults())
    return rejectProgramDirectory(
        "distributed_boundary outputs do not exactly cover function results",
        diagnostics);

  std::vector<bool> seenInputs(functionType.getNumInputs(), false);
  for (const DistributedBoundaryBinding &binding : boundary.inputs) {
    if (binding.index < 0 ||
        binding.index >= static_cast<int64_t>(functionType.getNumInputs()))
      return rejectProgramDirectory(
          "distributed boundary argument_index is out of range", diagnostics);
    if (meta.inputLocations[binding.index].type != "input_arg")
      return rejectProgramDirectory(
          "distributed boundary argument_index does not refer to input_arg",
          diagnostics);
    if (seenInputs[binding.index])
      return rejectProgramDirectory(
          "duplicate distributed boundary argument_index", diagnostics);
    seenInputs[binding.index] = true;
    auto tensorType =
        dyn_cast<RankedTensorType>(functionType.getInput(binding.index));
    if (!tensorType ||
        verifyDistributedBoundaryBinding(
            binding, tensorType, meta.inputSignatures[binding.index],
            boundary.logicalRankCount, diagnostics))
      return true;
  }
  for (auto [index, location] : llvm::enumerate(meta.inputLocations)) {
    if (location.type == "input_arg" && !seenInputs[index])
      return rejectProgramDirectory(
          "input_arg is missing distributed boundary metadata", diagnostics);
  }

  std::vector<bool> seenOutputs(functionType.getNumResults(), false);
  for (const DistributedBoundaryBinding &binding : boundary.outputs) {
    if (binding.index < 0 ||
        binding.index >= static_cast<int64_t>(functionType.getNumResults()))
      return rejectProgramDirectory(
          "distributed boundary result_index is out of range", diagnostics);
    if (seenOutputs[binding.index])
      return rejectProgramDirectory(
          "duplicate distributed boundary result_index", diagnostics);
    seenOutputs[binding.index] = true;
    auto tensorType =
        dyn_cast<RankedTensorType>(functionType.getResult(binding.index));
    if (!tensorType ||
        verifyDistributedBoundaryBinding(
            binding, tensorType, meta.outputSignatures[binding.index],
            boundary.logicalRankCount, diagnostics))
      return true;
  }
  if (llvm::any_of(seenOutputs, [](bool seen) { return !seen; }))
    return rejectProgramDirectory(
        "function result is missing distributed boundary metadata",
        diagnostics);
  if (result) {
    result->logicalRankCount = boundary.logicalRankCount;
    auto copyBindings =
        [](llvm::ArrayRef<DistributedBoundaryBinding> source,
           std::vector<wafer::frontend::ProgramBoundaryBinding> &destination) {
          destination.reserve(source.size());
          for (const DistributedBoundaryBinding &binding : source) {
            wafer::frontend::ProgramBoundaryBinding typed;
            typed.index = binding.index;
            typed.distribution =
                getVerifiedDistributionKind(binding.distribution);
            typed.globalShape = binding.globalShape;
            typed.localShape = binding.localShape;
            typed.dtype = normalizeProgramDtype(binding.dtype);
            typed.rankSlices.reserve(binding.ranks.size());
            for (const DistributedBoundaryRank &rank : binding.ranks)
              typed.rankSlices.push_back(getVerifiedRankSlice(rank));
            destination.push_back(std::move(typed));
          }
        };
    copyBindings(boundary.inputs, result->distributedInputs);
    copyBindings(boundary.outputs, result->distributedOutputs);
  }
  return false;
}

bool verifyParameterShards(
    ModuleOp module, llvm::StringRef programDir, const ProgramMetadata &meta,
    func::FuncOp func, llvm::raw_ostream &diagnostics,
    wafer::frontend::FrontendProgramVerificationResult *result) {
  std::string path =
      programPath(programDir, {"functions", "forward.parameter_shards.json"});
  if (!fileExists(path)) {
    if (hasSpmdParameterShardings(module))
      return rejectProgramDirectory(
          "partitioned StableHLO program directory is missing parameter "
          "shards",
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
  if (version != 3)
    return rejectProgramDirectory(
        "unsupported parameter shard metadata version", diagnostics);
  if (function != meta.name)
    return rejectProgramDirectory(
        "parameter shard function does not match program directory meta",
        diagnostics);
  if (logicalRankCount <= 0)
    return rejectProgramDirectory("logical_rank_count must be positive",
                                  diagnostics);

  FailureOr<wafer::ExecutionMeshOp> meshOp =
      findParameterShardExecutionMesh(module, diagnostics);
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

  const llvm::json::Array *parameters = root->getArray("parameters");
  if (!parameters)
    return rejectProgramDirectory("expected array field 'parameters'",
                                  diagnostics);

  FunctionType functionType = func.getFunctionType();
  std::vector<bool> seenParameterArgs(functionType.getNumInputs(), false);
  unsigned parameterShardCount = 0;
  for (const llvm::json::Value &value : *parameters) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return rejectProgramDirectory(
          "parameter shard metadata entries must be objects", diagnostics);
    wafer::frontend::ProgramParameterBinding verifiedBinding;
    if (verifyParameterShardMetadata(
            *object, meta, functionType, logicalRankCount, seenParameterArgs,
            programDir, diagnostics, result ? &verifiedBinding : nullptr))
      return true;
    if (result)
      result->parameters.push_back(std::move(verifiedBinding));
    ++parameterShardCount;
  }

  for (auto [index, location] : llvm::enumerate(meta.inputLocations)) {
    if (location.type == "parameter" && !seenParameterArgs[index])
      return rejectProgramDirectory(
          "parameter input is missing shard metadata: " + location.name,
          diagnostics);
  }

  if (result)
    result->programParameterShardCount = parameterShardCount;
  return false;
}

} // namespace

namespace wafer::frontend {

llvm::Expected<NpyTensorPayload> loadNpyTensorPayload(llvm::StringRef path) {
  std::string diagnosticsText;
  llvm::raw_string_ostream diagnostics(diagnosticsText);
  FailureOr<NpyPayloadMetadata> metadata =
      readNpyPayloadMetadata(path, path, diagnostics);
  if (failed(metadata))
    return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                   diagnostics.str().c_str());
  if (metadata->fortranOrder)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "npy payload must be row-major: %s",
                                   path.str().c_str());
  auto decoded = decodeNpyDescr(metadata->descr);
  if (!decoded)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "npy payload has unsupported dtype: %s",
                                   path.str().c_str());

  uint64_t rawBytes = decoded->second;
  for (int64_t dim : metadata->shape) {
    uint64_t next = 0;
    if (dim < 0 ||
        !checkedMulUint64(rawBytes, static_cast<uint64_t>(dim), next))
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "npy payload byte count is not representable: %s",
          path.str().c_str());
    rawBytes = next;
  }
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to reopen npy payload: %s",
                                   path.str().c_str());
  llvm::StringRef fileBytes = (*buffer)->getBuffer();
  if (metadata->dataOffset > fileBytes.size() ||
      rawBytes > fileBytes.size() - metadata->dataOffset)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "npy tensor payload is truncated: %s",
                                   path.str().c_str());
  llvm::ArrayRef<uint8_t> payload(
      reinterpret_cast<const uint8_t *>(fileBytes.data()) +
          metadata->dataOffset,
      static_cast<size_t>(rawBytes));
  return NpyTensorPayload{decoded->first.str(), metadata->shape,
                          std::vector<uint8_t>(payload.begin(), payload.end())};
}

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
verifyProgramDirectoryImpl(ModuleOp module, llvm::StringRef programPath,
                           llvm::raw_ostream &diagnostics,
                           FrontendProgramVerificationResult *result) {
  FrontendProgramVerificationResult verified;

  if (failed(verifyFrontendProgram(module, diagnostics, &verified)))
    return failure();

  std::string metaPath =
      ::programPath(programPath, {"functions", "forward.meta"});
  FailureOr<ProgramMetadata> meta = parseProgramMetadata(metaPath, diagnostics);
  if (failed(meta))
    return failure();

  bool rejected =
      verifyProgramMetadata(module, programPath, *meta, diagnostics, &verified);
  if (!rejected) {
    FailureOr<func::FuncOp> func = findSingleFunction(module, diagnostics);
    if (failed(func))
      return failure();
    bool postSpmdMarker =
        hasSpmdParameterShardings(module) ||
        fileExists(::programPath(
            programPath, {"functions", "forward.parameter_shards.json"}));
    rejected |= verifyDistributedBoundary(module, *meta, *func, postSpmdMarker,
                                          diagnostics, &verified);
    if (!rejected)
      rejected |= verifyParameterShards(module, programPath, *meta, *func,
                                        diagnostics, &verified);
  }
  if (rejected)
    return failure();
  if (result)
    *result = std::move(verified);
  return success();
}

LogicalResult
verifyProgramDirectory(ModuleOp module, llvm::StringRef programPath,
                       llvm::raw_ostream &diagnostics,
                       FrontendProgramVerificationResult *result) {
  return verifyProgramDirectoryImpl(module, programPath, diagnostics, result);
}

} // namespace wafer::frontend
