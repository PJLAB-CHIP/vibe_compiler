//===- CanonicalIRSnapshot.cpp - Structural transaction snapshot --------===//

#include "Wafer/Support/CanonicalIRSnapshot.h"

#include "Wafer/IR/WaferDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace wafer {
namespace {

static bool appendU32(std::vector<uint8_t> &bytes, uint64_t value) {
  if (value > std::numeric_limits<uint32_t>::max())
    return false;
  uint32_t narrowed = static_cast<uint32_t>(value);
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(narrowed >> shift));
  return true;
}

static void appendU16(std::vector<uint8_t> &bytes, uint16_t value) {
  bytes.push_back(static_cast<uint8_t>(value >> 8));
  bytes.push_back(static_cast<uint8_t>(value));
}

static void appendU64(std::vector<uint8_t> &bytes, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

static void appendI64(std::vector<uint8_t> &bytes, int64_t value) {
  appendU64(bytes, static_cast<uint64_t>(value));
}

static bool appendString(std::vector<uint8_t> &bytes, llvm::StringRef value) {
  if (!appendU32(bytes, value.size()))
    return false;
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
  return true;
}

static AdoptionDigest sha256(llvm::ArrayRef<uint8_t> bytes) {
  llvm::SHA256 hash;
  hash.update(bytes);
  return hash.final();
}

static void appendDigest(std::vector<uint8_t> &bytes,
                         const AdoptionDigest &digest) {
  bytes.insert(bytes.end(), digest.begin(), digest.end());
}

class Encoder {
public:
  Encoder(CanonicalIRSnapshotMode mode, std::vector<uint8_t> &bytes,
          std::string *diagnostic)
      : mode(mode), bytes(bytes), diagnostic(diagnostic) {}

  bool encode(mlir::Operation *root) {
    constexpr char domain[] = "wafer.canonical-ir-snapshot";
    bytes.insert(bytes.end(), domain, domain + sizeof(domain));
    appendU16(bytes, 1);
    bytes.push_back(static_cast<uint8_t>(mode));

    // Both revisions are semantic header fields.  Adding/changing a codec or
    // changing which dialect identities may occur requires changing the
    // corresponding digest, even when an individual snapshot does not use the
    // newly added codec.
    constexpr llvm::StringLiteral codecRevision =
        "wafer.canonical-ir-codec-registry-v3";
    appendDigest(bytes, sha256(llvm::ArrayRef<uint8_t>(
                            reinterpret_cast<const uint8_t *>(
                                codecRevision.data()),
                            codecRevision.size())));
    std::vector<std::string> dialects;
    for (mlir::Dialect *dialect : root->getContext()->getLoadedDialects())
      dialects.push_back(dialect->getNamespace().str());
    llvm::sort(dialects);
    dialects.erase(std::unique(dialects.begin(), dialects.end()),
                   dialects.end());
    std::vector<uint8_t> dialectRegistry;
    if (!appendU32(dialectRegistry, dialects.size()))
      return fail("dialect registry exceeds snapshot limit");
    for (const std::string &dialect : dialects)
      if (!appendString(dialectRegistry, dialect))
        return fail("dialect identity exceeds snapshot limit");
    appendDigest(bytes, sha256(dialectRegistry));
    return encodeOperation(root);
  }

private:
  bool fail(llvm::StringRef message) {
    if (diagnostic)
      *diagnostic = message.str();
    return false;
  }

  bool tag(llvm::StringRef codec) {
    return appendString(bytes, codec);
  }

  bool encodeOptionalAttribute(mlir::Attribute attribute) {
    bytes.push_back(attribute ? 1 : 0);
    return !attribute || encodeAttribute(attribute);
  }

  bool encodeTypeSequence(mlir::TypeRange types) {
    if (!appendU32(bytes, types.size()))
      return fail("type sequence exceeds snapshot limit");
    for (mlir::Type type : types)
      if (!encodeType(type))
        return false;
    return true;
  }

  bool encodeShape(llvm::ArrayRef<int64_t> shape) {
    if (!appendU32(bytes, shape.size()))
      return fail("shape rank exceeds snapshot limit");
    for (int64_t dimension : shape)
      appendI64(bytes, dimension);
    return true;
  }

  bool encodeFloatType(mlir::FloatType type) {
    llvm::StringRef identity;
    if (mlir::isa<mlir::Float8E5M2Type>(type))
      identity = "builtin.float8e5m2-type-v1";
    else if (mlir::isa<mlir::Float8E4M3Type>(type))
      identity = "builtin.float8e4m3-type-v1";
    else if (mlir::isa<mlir::Float8E4M3FNType>(type))
      identity = "builtin.float8e4m3fn-type-v1";
    else if (mlir::isa<mlir::Float8E5M2FNUZType>(type))
      identity = "builtin.float8e5m2fnuz-type-v1";
    else if (mlir::isa<mlir::Float8E4M3FNUZType>(type))
      identity = "builtin.float8e4m3fnuz-type-v1";
    else if (mlir::isa<mlir::Float8E4M3B11FNUZType>(type))
      identity = "builtin.float8e4m3b11fnuz-type-v1";
    else if (mlir::isa<mlir::Float8E3M4Type>(type))
      identity = "builtin.float8e3m4-type-v1";
    else if (mlir::isa<mlir::Float6E3M2FNType>(type))
      identity = "builtin.float6e3m2fn-type-v1";
    else if (mlir::isa<mlir::BFloat16Type>(type))
      identity = "builtin.bfloat16-type-v1";
    else if (mlir::isa<mlir::Float16Type>(type))
      identity = "builtin.float16-type-v1";
    else if (mlir::isa<mlir::FloatTF32Type>(type))
      identity = "builtin.floattf32-type-v1";
    else if (mlir::isa<mlir::Float32Type>(type))
      identity = "builtin.float32-type-v1";
    else if (mlir::isa<mlir::Float64Type>(type))
      identity = "builtin.float64-type-v1";
    else if (mlir::isa<mlir::Float80Type>(type))
      identity = "builtin.float80-type-v1";
    else if (mlir::isa<mlir::Float128Type>(type))
      identity = "builtin.float128-type-v1";
    else
      return fail("float type has no registered canonical codec");
    return tag(identity);
  }

  bool encodeType(mlir::Type type) {
    if (!type)
      return fail("null type has no registered canonical codec");
    if (mlir::isa<mlir::IndexType>(type))
      return tag("builtin.index-type-v1");
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
      if (!tag("builtin.integer-type-v1") ||
          !appendU32(bytes, integer.getWidth()))
        return false;
      bytes.push_back(static_cast<uint8_t>(integer.getSignedness()));
      return true;
    }
    if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
      return encodeFloatType(floating);
    if (auto complex = mlir::dyn_cast<mlir::ComplexType>(type))
      return tag("builtin.complex-type-v1") &&
             encodeType(complex.getElementType());
    if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type))
      return tag("builtin.ranked-tensor-type-v1") &&
             encodeShape(tensor.getShape()) &&
             encodeType(tensor.getElementType()) &&
             encodeOptionalAttribute(tensor.getEncoding());
    if (auto tensor = mlir::dyn_cast<mlir::UnrankedTensorType>(type))
      return tag("builtin.unranked-tensor-type-v1") &&
             encodeType(tensor.getElementType());
    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type))
      return tag("builtin.memref-type-v1") && encodeShape(memref.getShape()) &&
             encodeType(memref.getElementType()) &&
             encodeOptionalAttribute(memref.getLayout()) &&
             encodeOptionalAttribute(memref.getMemorySpace());
    if (auto memref = mlir::dyn_cast<mlir::UnrankedMemRefType>(type))
      return tag("builtin.unranked-memref-type-v1") &&
             encodeType(memref.getElementType()) &&
             encodeOptionalAttribute(memref.getMemorySpace());
    if (auto function = mlir::dyn_cast<mlir::FunctionType>(type))
      return tag("builtin.function-type-v1") &&
             encodeTypeSequence(function.getInputs()) &&
             encodeTypeSequence(function.getResults());
    if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type))
      return tag("builtin.tuple-type-v1") &&
             encodeTypeSequence(tuple.getTypes());
    if (mlir::isa<mlir::NoneType>(type))
      return tag("builtin.none-type-v1");
    if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
      if (!tag("builtin.vector-type-v1") || !encodeShape(vector.getShape()) ||
          !appendU32(bytes, vector.getScalableDims().size()))
        return false;
      for (bool scalable : vector.getScalableDims())
        bytes.push_back(scalable ? 1 : 0);
      return encodeType(vector.getElementType());
    }
    if (mlir::isa<mlir::async::TokenType>(type))
      return tag("async.token-type-v1");
    if (mlir::isa<mlir::async::GroupType>(type))
      return tag("async.group-type-v1");
    if (auto value = mlir::dyn_cast<mlir::async::ValueType>(type))
      return tag("async.value-type-v1") &&
             encodeType(value.getValueType());
    return fail("type has no registered canonical codec");
  }

  bool encodeAPInt(const llvm::APInt &value) {
    unsigned width = value.getBitWidth();
    if (!appendU32(bytes, width))
      return false;
    unsigned byteCount = (width + 7) / 8;
    if (!appendU32(bytes, byteCount))
      return false;
    for (unsigned index = byteCount; index > 0; --index) {
      unsigned offset = (index - 1) * 8;
      unsigned count = std::min(8u, width - offset);
      bytes.push_back(static_cast<uint8_t>(
          value.extractBitsAsZExtValue(count, offset)));
    }
    return true;
  }

  bool encodeAffineExpr(mlir::AffineExpr expression) {
    if (auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression))
      return tag("affine.dim-expr-v1") &&
             appendU32(bytes, dimension.getPosition());
    if (auto symbol = mlir::dyn_cast<mlir::AffineSymbolExpr>(expression))
      return tag("affine.symbol-expr-v1") &&
             appendU32(bytes, symbol.getPosition());
    if (auto constant = mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
      if (!tag("affine.constant-expr-v1"))
        return false;
      appendI64(bytes, constant.getValue());
      return true;
    }
    if (auto binary = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression)) {
      if (!tag("affine.binary-expr-v1"))
        return false;
      bytes.push_back(static_cast<uint8_t>(binary.getKind()));
      return encodeAffineExpr(binary.getLHS()) &&
             encodeAffineExpr(binary.getRHS());
    }
    return fail("affine expression has no registered canonical codec");
  }

  bool encodeAffineMap(mlir::AffineMap map) {
    if (!appendU32(bytes, map.getNumDims()) ||
        !appendU32(bytes, map.getNumSymbols()) ||
        !appendU32(bytes, map.getNumResults()))
      return false;
    for (mlir::AffineExpr result : map.getResults())
      if (!encodeAffineExpr(result))
        return false;
    return true;
  }

  bool encodeDictionary(mlir::DictionaryAttr dictionary) {
    std::vector<mlir::NamedAttribute> attributes(dictionary.begin(),
                                                  dictionary.end());
    llvm::sort(attributes,
               [](mlir::NamedAttribute lhs, mlir::NamedAttribute rhs) {
                 return lhs.getName().strref() < rhs.getName().strref();
               });
    if (!appendU32(bytes, attributes.size()))
      return fail("attribute dictionary exceeds snapshot limit");
    for (mlir::NamedAttribute attribute : attributes)
      if (!appendString(bytes, attribute.getName().strref()) ||
          !encodeAttribute(attribute.getValue()))
        return false;
    return true;
  }

  template <typename EnumAttr>
  bool encodeEnumAttribute(mlir::Attribute attribute, llvm::StringRef codec) {
    if (auto value = mlir::dyn_cast<EnumAttr>(attribute))
      return tag(codec) &&
             appendU32(bytes, static_cast<uint32_t>(value.getValue()));
    return false;
  }

  bool encodeArithAttribute(mlir::Attribute attribute) {
#define WAFER_ARITH_ENUM_CODEC(ATTR, CODEC)                                 \
  if (mlir::isa<mlir::arith::ATTR>(attribute))                              \
    return encodeEnumAttribute<mlir::arith::ATTR>(attribute, CODEC)
    WAFER_ARITH_ENUM_CODEC(CmpFPredicateAttr,
                           "arith.cmpf-predicate-attr-v1");
    WAFER_ARITH_ENUM_CODEC(CmpIPredicateAttr,
                           "arith.cmpi-predicate-attr-v1");
    WAFER_ARITH_ENUM_CODEC(AtomicRMWKindAttr,
                           "arith.atomic-rmw-kind-attr-v1");
    WAFER_ARITH_ENUM_CODEC(FastMathFlagsAttr,
                           "arith.fast-math-flags-attr-v1");
    WAFER_ARITH_ENUM_CODEC(IntegerOverflowFlagsAttr,
                           "arith.integer-overflow-flags-attr-v1");
    WAFER_ARITH_ENUM_CODEC(RoundingModeAttr,
                           "arith.rounding-mode-attr-v1");
#undef WAFER_ARITH_ENUM_CODEC
    return false;
  }

  bool encodeLinalgAttribute(mlir::Attribute attribute) {
#define WAFER_LINALG_ENUM_CODEC(ATTR, CODEC)                                \
  if (mlir::isa<mlir::linalg::ATTR>(attribute))                             \
    return encodeEnumAttribute<mlir::linalg::ATTR>(attribute, CODEC)
    WAFER_LINALG_ENUM_CODEC(UnaryFnAttr, "linalg.unary-fn-attr-v1");
    WAFER_LINALG_ENUM_CODEC(BinaryFnAttr, "linalg.binary-fn-attr-v1");
    WAFER_LINALG_ENUM_CODEC(TernaryFnAttr, "linalg.ternary-fn-attr-v1");
    WAFER_LINALG_ENUM_CODEC(TypeFnAttr, "linalg.type-fn-attr-v1");
    WAFER_LINALG_ENUM_CODEC(IteratorTypeAttr,
                            "linalg.iterator-type-attr-v1");
#undef WAFER_LINALG_ENUM_CODEC
    return false;
  }

  bool encodeStablehloAttribute(mlir::Attribute attribute) {
    constexpr std::array<llvm::StringLiteral, 15> registeredNames = {
        "stablehlo.channel_handle",
        "stablehlo.comparison_direction",
        "stablehlo.comparison_type",
        "stablehlo.conv",
        "stablehlo.dot_algorithm",
        "stablehlo.dot",
        "stablehlo.fft_type",
        "stablehlo.gather",
        "stablehlo.output_operand_alias",
        "stablehlo.precision",
        "stablehlo.rng_algorithm",
        "stablehlo.rng_distribution",
        "stablehlo.scatter",
        "stablehlo.transpose",
        "stablehlo.type_extensions",
    };
    llvm::StringRef name = attribute.getAbstractAttribute().getName();
    if (!llvm::is_contained(registeredNames, name))
      return false;
    std::string spelling;
    llvm::raw_string_ostream stream(spelling);
    attribute.print(stream);
    stream.flush();
    return tag("stablehlo.registered-attribute-spelling-v1") &&
           appendString(bytes, name) && appendString(bytes, spelling);
  }

  bool encodeWaferAttribute(mlir::Attribute attribute) {
    if (auto memory = mlir::dyn_cast<MemoryAttr>(attribute))
      return tag("wafer.memory-attr-v1") &&
             appendU32(bytes, static_cast<uint32_t>(memory.getSpace())) &&
             appendU32(bytes, static_cast<uint32_t>(memory.getLayout()));
    if (auto offset = mlir::dyn_cast<SPMOffsetAttr>(attribute)) {
      if (!tag("wafer.spm-offset-attr-v1"))
        return false;
      appendI64(bytes, offset.getOffset());
      return true;
    }
    if (auto offset = mlir::dyn_cast<DDROffsetAttr>(attribute)) {
      if (!tag("wafer.ddr-offset-attr-v1"))
        return false;
      appendI64(bytes, offset.getOffset());
      return true;
    }
    if (auto message = mlir::dyn_cast<DTEMessageAttr>(attribute)) {
      if (!tag("wafer.dte-message-attr-v1"))
        return false;
      appendI64(bytes, message.getCommunicationId());
      appendU32(bytes, static_cast<uint32_t>(message.getPhase()));
      appendI64(bytes, message.getRound());
      appendI64(bytes, message.getPayloadSlice());
      return true;
    }
    if (auto binding = mlir::dyn_cast<DirectDTEBindingAttr>(attribute)) {
      if (!tag("wafer.direct-dte-binding-attr-v1"))
        return false;
      appendU32(bytes,
                static_cast<uint32_t>(binding.getAllocationProfile()));
      appendI64(bytes, binding.getReceiverFsmId());
      appendI64(bytes, binding.getRemoteReceiverOffset());
      appendU32(bytes,
                static_cast<uint32_t>(binding.getCompletionProfile()));
      return true;
    }
#define WAFER_ENUM_CODEC(ATTR, CODEC)                                        \
  if (mlir::isa<ATTR>(attribute))                                           \
    return encodeEnumAttribute<ATTR>(attribute, CODEC)
    WAFER_ENUM_CODEC(TargetAttr, "wafer.target-attr-v1");
    WAFER_ENUM_CODEC(ComputeElementwiseKindAttr,
                     "wafer.compute-elementwise-kind-attr-v1");
    WAFER_ENUM_CODEC(ComputeReduceKindAttr,
                     "wafer.compute-reduce-kind-attr-v1");
    WAFER_ENUM_CODEC(InstrElementwiseKindAttr,
                     "wafer.instr-elementwise-kind-attr-v1");
    WAFER_ENUM_CODEC(InstrReduceKindAttr,
                     "wafer.instr-reduce-kind-attr-v1");
    WAFER_ENUM_CODEC(InstrConvertKindAttr,
                     "wafer.instr-convert-kind-attr-v1");
    WAFER_ENUM_CODEC(InstrConvKindAttr, "wafer.instr-conv-kind-attr-v1");
    WAFER_ENUM_CODEC(InstrPoolKindAttr, "wafer.instr-pool-kind-attr-v1");
    WAFER_ENUM_CODEC(InstrUnpoolKindAttr,
                     "wafer.instr-unpool-kind-attr-v1");
    WAFER_ENUM_CODEC(InstrDataMoveKindAttr,
                     "wafer.instr-data-move-kind-attr-v1");
    WAFER_ENUM_CODEC(InstrPeripheralKindAttr,
                     "wafer.instr-peripheral-kind-attr-v1");
#undef WAFER_ENUM_CODEC
    return false;
  }

  bool encodeLocation(mlir::LocationAttr location) {
    if (mlir::isa<mlir::UnknownLoc>(location))
      return tag("builtin.unknown-location-v1");
    if (auto file = mlir::dyn_cast<mlir::FileLineColLoc>(location))
      return tag("builtin.file-line-column-location-v1") &&
             appendString(bytes, file.getFilename().getValue()) &&
             appendU32(bytes, file.getLine()) &&
             appendU32(bytes, file.getColumn());
    if (auto name = mlir::dyn_cast<mlir::NameLoc>(location))
      return tag("builtin.name-location-v1") &&
             appendString(bytes, name.getName().getValue()) &&
             encodeLocation(name.getChildLoc());
    if (auto call = mlir::dyn_cast<mlir::CallSiteLoc>(location))
      return tag("builtin.call-site-location-v1") &&
             encodeLocation(call.getCallee()) &&
             encodeLocation(call.getCaller());
    if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(location)) {
      if (!tag("builtin.fused-location-v1") ||
          !appendU32(bytes, fused.getLocations().size()))
        return false;
      for (mlir::Location child : fused.getLocations())
        if (!encodeLocation(child))
          return false;
      return encodeOptionalAttribute(fused.getMetadata());
    }
    // OpaqueLoc contains a process pointer and TypeID, neither of which is a
    // canonical semantic payload.  Its fallback cannot make that payload safe.
    return fail("location has no registered canonical codec");
  }

  bool encodeDenseArray(mlir::DenseArrayAttr array) {
    if (!tag("builtin.dense-array-attr-v1") ||
        !encodeType(array.getElementType()) ||
        !appendU32(bytes, array.size()))
      return false;
    auto appendBits = [&](auto values) {
      for (auto value : values) {
        using T = decltype(value);
        if constexpr (std::is_floating_point_v<T>) {
          std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t> bits = 0;
          std::memcpy(&bits, &value, sizeof(T));
          appendU64(bytes, bits);
        } else {
          appendU64(bytes, static_cast<uint64_t>(value));
        }
      }
      return true;
    };
    if (auto values = mlir::dyn_cast<mlir::DenseBoolArrayAttr>(array))
      return appendBits(values.asArrayRef());
    if (auto values = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(array))
      return appendBits(values.asArrayRef());
    if (auto values = mlir::dyn_cast<mlir::DenseI16ArrayAttr>(array))
      return appendBits(values.asArrayRef());
    if (auto values = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(array))
      return appendBits(values.asArrayRef());
    if (auto values = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(array))
      return appendBits(values.asArrayRef());
    if (auto values = mlir::dyn_cast<mlir::DenseF32ArrayAttr>(array))
      return appendBits(values.asArrayRef());
    if (auto values = mlir::dyn_cast<mlir::DenseF64ArrayAttr>(array))
      return appendBits(values.asArrayRef());
    return fail("dense array element type has no registered canonical codec");
  }

  bool encodeAttribute(mlir::Attribute attribute) {
    if (!attribute)
      return fail("null attribute requires an explicit optional codec");
    if (auto location = mlir::dyn_cast<mlir::LocationAttr>(attribute))
      return encodeLocation(location);
    if (mlir::isa<mlir::UnitAttr>(attribute))
      return tag("builtin.unit-attr-v1");
    if (auto boolean = mlir::dyn_cast<mlir::BoolAttr>(attribute)) {
      if (!tag("builtin.bool-attr-v1"))
        return false;
      bytes.push_back(boolean.getValue() ? 1 : 0);
      return true;
    }
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute))
      return tag("builtin.integer-attr-v1") &&
             encodeType(integer.getType()) && encodeAPInt(integer.getValue());
    if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(attribute))
      return tag("builtin.float-attr-v1") &&
             encodeType(floating.getType()) &&
             encodeAPInt(floating.getValue().bitcastToAPInt());
    if (auto string = mlir::dyn_cast<mlir::StringAttr>(attribute))
      return tag("builtin.string-attr-v1") &&
             encodeType(string.getType()) &&
             appendString(bytes, string.getValue());
    if (auto type = mlir::dyn_cast<mlir::TypeAttr>(attribute))
      return tag("builtin.type-attr-v1") && encodeType(type.getValue());
    if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attribute)) {
      if (!tag("builtin.array-attr-v1") || !appendU32(bytes, array.size()))
        return false;
      for (mlir::Attribute element : array)
        if (!encodeAttribute(element))
          return false;
      return true;
    }
    if (auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attribute))
      return tag("builtin.dictionary-attr-v1") &&
             encodeDictionary(dictionary);
    if (auto symbol = mlir::dyn_cast<mlir::SymbolRefAttr>(attribute)) {
      if (!tag("builtin.symbol-ref-attr-v1") ||
          !appendString(bytes, symbol.getRootReference().getValue()) ||
          !appendU32(bytes, symbol.getNestedReferences().size()))
        return false;
      for (mlir::FlatSymbolRefAttr nested : symbol.getNestedReferences())
        if (!appendString(bytes, nested.getValue()))
          return false;
      return true;
    }
    if (auto affine = mlir::dyn_cast<mlir::AffineMapAttr>(attribute))
      return tag("builtin.affine-map-attr-v1") &&
             encodeAffineMap(affine.getValue());
    if (auto integerSet = mlir::dyn_cast<mlir::IntegerSetAttr>(attribute)) {
      mlir::IntegerSet set = integerSet.getValue();
      if (!tag("builtin.integer-set-attr-v1") ||
          !appendU32(bytes, set.getNumDims()) ||
          !appendU32(bytes, set.getNumSymbols()) ||
          !appendU32(bytes, set.getNumConstraints()))
        return false;
      for (auto [constraint, equality] :
           llvm::zip(set.getConstraints(), set.getEqFlags())) {
        bytes.push_back(equality ? 1 : 0);
        if (!encodeAffineExpr(constraint))
          return false;
      }
      return true;
    }
    if (auto layout = mlir::dyn_cast<mlir::StridedLayoutAttr>(attribute)) {
      if (!tag("builtin.strided-layout-attr-v1"))
        return false;
      appendI64(bytes, layout.getOffset());
      if (!appendU32(bytes, layout.getStrides().size()))
        return false;
      for (int64_t stride : layout.getStrides())
        appendI64(bytes, stride);
      return true;
    }
    if (auto denseArray = mlir::dyn_cast<mlir::DenseArrayAttr>(attribute))
      return encodeDenseArray(denseArray);
    if (auto dense =
            mlir::dyn_cast<mlir::DenseIntOrFPElementsAttr>(attribute)) {
      if (!tag("builtin.dense-int-or-fp-elements-attr-v1") ||
          !encodeType(dense.getType()) ||
          !appendU32(bytes, dense.getNumElements()))
        return false;
      for (mlir::Attribute element : dense.getValues<mlir::Attribute>())
        if (!encodeAttribute(element))
          return false;
      return true;
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseStringElementsAttr>(attribute)) {
      if (!tag("builtin.dense-string-elements-attr-v1") ||
          !encodeType(dense.getType()) ||
          !appendU32(bytes, dense.getNumElements()))
        return false;
      for (llvm::StringRef element : dense.getValues<llvm::StringRef>())
        if (!appendString(bytes, element))
          return false;
      return true;
    }
    if (auto resource =
            mlir::dyn_cast<mlir::DenseResourceElementsAttr>(attribute)) {
      const mlir::AsmResourceBlob *blob = resource.getRawHandle().getBlob();
      if (!blob)
        return fail("dense resource attribute has no available canonical blob");
      if (!tag("builtin.dense-resource-elements-attr-v1") ||
          !encodeType(resource.getType()) ||
          !appendU32(bytes, blob->getDataAlignment()) ||
          !appendU32(bytes, blob->getData().size()))
        return false;
      bytes.insert(bytes.end(), blob->getData().begin(),
                   blob->getData().end());
      return true;
    }
    if (auto sparse = mlir::dyn_cast<mlir::SparseElementsAttr>(attribute))
      return tag("builtin.sparse-elements-attr-v1") &&
             encodeType(sparse.getType()) &&
             encodeAttribute(sparse.getIndices()) &&
             encodeAttribute(sparse.getValues());
    if (encodeArithAttribute(attribute))
      return true;
    if (encodeLinalgAttribute(attribute))
      return true;
    if (encodeStablehloAttribute(attribute))
      return true;
    if (encodeWaferAttribute(attribute))
      return true;
    return fail(("attribute or property '" +
                 attribute.getAbstractAttribute().getName() +
                 "' has no registered canonical codec")
                    .str());
  }

  bool encodeOperation(mlir::Operation *operation) {
    if (!operation->getRegisteredInfo())
      return fail("unknown operation has no registered canonical codec");
    if (!appendString(bytes, operation->getName().getStringRef()) ||
        !appendU32(bytes, operation->getNumResults()))
      return fail("operation header exceeds snapshot limit");
    for (mlir::OpResult result : operation->getResults()) {
      if (!encodeType(result.getType()))
        return false;
      values[result.getAsOpaquePointer()] = nextValue++;
    }

    // Registered inherent properties and discardable attrs are separate
    // channels in MLIR and remain separate in the snapshot.
    if (!encodeOptionalAttribute(operation->getPropertiesAsAttribute()) ||
        !encodeDictionary(operation->getDiscardableAttrDictionary()))
      return false;
    if (mode == CanonicalIRSnapshotMode::MutationGuard &&
        !encodeLocation(operation->getLoc()))
      return false;

    if (!appendU32(bytes, operation->getNumOperands()))
      return fail("operand list exceeds snapshot limit");
    for (mlir::Value operand : operation->getOperands()) {
      auto found = values.find(operand.getAsOpaquePointer());
      if (found == values.end())
        return fail("operand definition is outside canonical traversal");
      appendU64(bytes, found->second);
    }

    if (!appendU32(bytes, operation->getNumSuccessors()))
      return fail("successor list exceeds snapshot limit");
    for (mlir::Block *successor : operation->getSuccessors()) {
      auto found = blocks.find(successor);
      if (found == blocks.end())
        return fail("successor block is outside canonical region traversal");
      appendU64(bytes, found->second);
    }

    if (!appendU32(bytes, operation->getNumRegions()))
      return fail("region list exceeds snapshot limit");
    for (mlir::Region &region : operation->getRegions()) {
      if (!appendU32(bytes, region.getBlocks().size()))
        return fail("block list exceeds snapshot limit");
      for (mlir::Block &block : region)
        blocks[&block] = nextBlock++;
      for (mlir::Block &block : region) {
        appendU64(bytes, blocks.lookup(&block));
        if (!appendU32(bytes, block.getNumArguments()))
          return fail("block argument list exceeds snapshot limit");
        for (mlir::BlockArgument argument : block.getArguments()) {
          if (!encodeType(argument.getType()))
            return false;
          values[argument.getAsOpaquePointer()] = nextValue++;
          if (mode == CanonicalIRSnapshotMode::MutationGuard &&
              !encodeLocation(argument.getLoc()))
            return false;
        }
        if (!appendU32(bytes, static_cast<uint64_t>(
                                  std::distance(block.begin(), block.end()))))
          return fail("operation list exceeds snapshot limit");
        for (mlir::Operation &nested : block)
          if (!encodeOperation(&nested))
            return false;
      }
    }
    return true;
  }

  CanonicalIRSnapshotMode mode;
  std::vector<uint8_t> &bytes;
  std::string *diagnostic;
  llvm::DenseMap<void *, uint64_t> values;
  llvm::DenseMap<mlir::Block *, uint64_t> blocks;
  uint64_t nextValue = 0;
  uint64_t nextBlock = 0;
};

} // namespace

bool createCanonicalIRSnapshotV1(mlir::Operation *root,
                                 CanonicalIRSnapshotMode mode,
                                 CanonicalIRSnapshotV1 &snapshot,
                                 std::string *diagnostic) {
  if (!root) {
    if (diagnostic)
      *diagnostic = "snapshot root is null";
    return false;
  }
  snapshot = {};
  snapshot.mode = mode;
  snapshot.rootOperationKind = root->getName().getStringRef().str();
  Encoder encoder(mode, snapshot.structuralBytes, diagnostic);
  if (!encoder.encode(root)) {
    snapshot = {};
    return false;
  }
  snapshot.sha256Digest = sha256(snapshot.structuralBytes);
  return true;
}

} // namespace wafer
