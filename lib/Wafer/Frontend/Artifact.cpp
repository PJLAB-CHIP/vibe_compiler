//===- Artifact.cpp - Wafer frontend artifact verifier -------------------===//

#include "Wafer/Frontend/Artifact.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kDynamicBoundsAttr =
    "wafer.frontend.dynamic_bounds";
constexpr llvm::StringLiteral kConstantAttr = "wafer.frontend.constant";

struct SidecarConstant {
  std::string function;
  int64_t arg = -1;
  std::string resourceKey;
  std::vector<int64_t> shape;
  std::string dtype;
  uint64_t byteSize = 0;
  std::string checksum;
};

bool reject(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-import-model: " << message << "\n";
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
    return reject(diagnostics, "sidecar rejected: expected string field '" +
                                   field.str() + "'");
  out = value->str();
  return false;
}

bool readIntegerField(const llvm::json::Object &object, llvm::StringRef field,
                      int64_t &out, llvm::raw_ostream &diagnostics) {
  std::optional<int64_t> value = object.getInteger(field);
  if (!value)
    return reject(diagnostics, "sidecar rejected: expected integer field '" +
                                   field.str() + "'");
  out = *value;
  return false;
}

bool readUInt64Field(const llvm::json::Object &object, llvm::StringRef field,
                     uint64_t &out, llvm::raw_ostream &diagnostics) {
  int64_t value = 0;
  if (readIntegerField(object, field, value, diagnostics))
    return true;
  if (value < 0)
    return reject(diagnostics, "sidecar rejected: expected non-negative field '" +
                                   field.str() + "'");
  out = static_cast<uint64_t>(value);
  return false;
}

bool readShapeField(const llvm::json::Object &object,
                    std::vector<int64_t> &shape,
                    llvm::raw_ostream &diagnostics) {
  const llvm::json::Array *array = object.getArray("shape");
  if (!array)
    return reject(diagnostics,
                  "sidecar rejected: expected array field 'shape'");

  for (const llvm::json::Value &value : *array) {
    std::optional<int64_t> dim = value.getAsInteger();
    if (!dim || *dim < 0)
      return reject(diagnostics,
                    "sidecar rejected: expected non-negative shape dimension");
    shape.push_back(*dim);
  }
  return false;
}

FailureOr<std::vector<SidecarConstant>>
parseSidecarConstants(llvm::StringRef sidecarPath,
                      llvm::raw_ostream &diagnostics) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(sidecarPath);
  if (!bufferOrError) {
    reject(diagnostics, "sidecar rejected: failed to read '" + sidecarPath.str() +
                            "'");
    return failure();
  }

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*bufferOrError)->getBuffer());
  if (!parsed) {
    std::string message = llvm::toString(parsed.takeError());
    reject(diagnostics, "sidecar rejected: invalid JSON: " + message);
    return failure();
  }

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    reject(diagnostics, "sidecar rejected: root must be an object");
    return failure();
  }

  std::optional<int64_t> version = root->getInteger("version");
  if (!version || *version != 0) {
    reject(diagnostics, "sidecar rejected: expected version 0");
    return failure();
  }

  const llvm::json::Array *constants = root->getArray("constants");
  if (!constants) {
    reject(diagnostics,
           "sidecar rejected: expected array field 'constants'");
    return failure();
  }

  std::vector<SidecarConstant> result;
  result.reserve(constants->size());
  for (const llvm::json::Value &value : *constants) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object) {
      reject(diagnostics,
             "sidecar rejected: constants entries must be objects");
      return failure();
    }

    SidecarConstant constant;
    if (readStringField(*object, "function", constant.function, diagnostics) ||
        readIntegerField(*object, "arg", constant.arg, diagnostics) ||
        readStringField(*object, "resource_key", constant.resourceKey,
                        diagnostics) ||
        readShapeField(*object, constant.shape, diagnostics) ||
        readStringField(*object, "dtype", constant.dtype, diagnostics) ||
        readUInt64Field(*object, "byte_size", constant.byteSize,
                        diagnostics))
      return failure();

    if (std::optional<llvm::StringRef> checksum =
            object->getString("checksum"))
      constant.checksum = checksum->str();

    if (constant.arg < 0) {
      reject(diagnostics,
             "sidecar rejected: constant argument index must be non-negative");
      return failure();
    }
    if (constant.resourceKey.empty()) {
      reject(diagnostics,
             "sidecar rejected: constant resource_key must be non-empty");
      return failure();
    }

    result.push_back(std::move(constant));
  }

  return result;
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

bool rejectSidecarConstant(const SidecarConstant &constant,
                           llvm::StringRef reason,
                           llvm::raw_ostream &diagnostics) {
  diagnostics << "wafer-import-model: sidecar constant rejected: " << reason
              << "\n";
  return true;
}

bool verifySidecarConstant(ModuleOp module, const SidecarConstant &constant,
                           llvm::raw_ostream &diagnostics) {
  auto func = module.lookupSymbol<func::FuncOp>(constant.function);
  if (!func)
    return rejectSidecarConstant(
        constant, "missing func.func @" + constant.function, diagnostics);

  if (static_cast<uint64_t>(constant.arg) >=
      func.getFunctionType().getNumInputs()) {
    return rejectSidecarConstant(
        constant,
        ("func.func @" + constant.function + " argument " +
         Twine(constant.arg) + " is out of range")
            .str(),
        diagnostics);
  }

  Type argType = func.getFunctionType().getInput(constant.arg);
  auto tensorType = dyn_cast<RankedTensorType>(argType);
  if (!tensorType || !tensorType.hasStaticShape()) {
    return rejectSidecarConstant(
        constant,
        ("func.func @" + constant.function + " argument " +
         Twine(constant.arg) + " must be a statically shaped ranked tensor")
            .str(),
        diagnostics);
  }

  DictionaryAttr attrs = getFunctionArgAttrs(func, constant.arg);
  Attribute constantAttr = attrs ? attrs.get(kConstantAttr) : Attribute();
  auto resourceKey = dyn_cast_or_null<StringAttr>(constantAttr);
  if (!resourceKey) {
    return rejectSidecarConstant(
        constant,
        ("func.func @" + constant.function + " argument " +
         Twine(constant.arg) + " is missing wafer.frontend.constant")
            .str(),
        diagnostics);
  }
  if (resourceKey.getValue() != constant.resourceKey) {
    return rejectSidecarConstant(
        constant,
        ("func.func @" + constant.function + " argument " +
         Twine(constant.arg) + " resource key mismatch")
            .str(),
        diagnostics);
  }

  if (!llvm::equal(tensorType.getShape(), constant.shape)) {
    return rejectSidecarConstant(
        constant,
        ("shape mismatch for func.func @" + constant.function + " argument " +
         Twine(constant.arg))
            .str(),
        diagnostics);
  }

  std::string expectedDtype = dtypeString(tensorType.getElementType());
  if (expectedDtype.empty() || expectedDtype != constant.dtype) {
    return rejectSidecarConstant(
        constant,
        ("dtype mismatch for func.func @" + constant.function + " argument " +
         Twine(constant.arg))
            .str(),
        diagnostics);
  }

  unsigned bitWidth = tensorType.getElementTypeBitWidth();
  if (bitWidth % 8 != 0)
    return rejectSidecarConstant(
        constant,
        ("byte size is not representable for func.func @" +
         constant.function + " argument " + Twine(constant.arg))
            .str(),
        diagnostics);

  uint64_t expectedByteSize =
      static_cast<uint64_t>(tensorType.getNumElements()) * (bitWidth / 8);
  if (expectedByteSize != constant.byteSize) {
    return rejectSidecarConstant(
        constant,
        ("byte size mismatch for func.func @" + constant.function +
         " argument " + Twine(constant.arg))
            .str(),
        diagnostics);
  }

  return false;
}

bool verifySidecar(ModuleOp module, llvm::StringRef sidecarPath,
                   llvm::raw_ostream &diagnostics,
                   wafer::frontend::ArtifactVerificationResult *result) {
  if (sidecarPath.empty())
    return false;

  FailureOr<std::vector<SidecarConstant>> constants =
      parseSidecarConstants(sidecarPath, diagnostics);
  if (failed(constants))
    return true;

  bool rejected = false;
  for (const SidecarConstant &constant : *constants)
    rejected |= verifySidecarConstant(module, constant, diagnostics);

  if (!rejected && result)
    result->sidecarConstantCount = constants->size();
  return rejected;
}

} // namespace

namespace wafer::frontend {

LogicalResult verifyFrontendArtifact(ModuleOp module, llvm::StringRef sidecarPath,
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
  rejected |= verifySidecar(module, sidecarPath, diagnostics, result);

  return rejected ? failure() : success();
}

} // namespace wafer::frontend
