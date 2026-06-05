//===- WaferDialect.cpp - Wafer dialect implementation -------------------===//

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>

using namespace wafer;

#include "Wafer/IR/WaferEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Wafer/IR/WaferAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Wafer/IR/WaferTypes.cpp.inc"

#include "Wafer/IR/WaferOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Wafer/IR/WaferOps.cpp.inc"

namespace {

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static std::optional<int64_t> getElementBitWidth(mlir::Type elementType) {
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
    return floatType.getWidth();
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(elementType))
    return integerType.getWidth();
  if (mlir::isa<mlir::IndexType>(elementType))
    return 64;
  if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elementType)) {
    std::optional<int64_t> elementBits =
        getElementBitWidth(complexType.getElementType());
    if (!elementBits)
      return std::nullopt;
    int64_t complexBits = 0;
    if (!checkedMul(*elementBits, 2, complexBits))
      return std::nullopt;
    return complexBits;
  }
  return std::nullopt;
}

static int64_t ceilDivToBytes(int64_t bits) {
  return bits / 8 + (bits % 8 == 0 ? 0 : 1);
}

static std::optional<int64_t>
getStaticElementCount(llvm::ArrayRef<int64_t> shape) {
  int64_t elements = 1;
  for (int64_t dim : shape) {
    if (dim == mlir::ShapedType::kDynamic)
      return std::nullopt;
    int64_t next = 0;
    if (!checkedMul(elements, dim, next))
      return std::nullopt;
    elements = next;
  }
  return elements;
}

static std::optional<int64_t>
getStaticPhysicalElementCount(llvm::ArrayRef<int64_t> shape,
                              MemLayout layout,
                              WaferPhysicalTensorInfo &info) {
  if (shape.empty())
    return getStaticElementCount(shape);
  if (layout != MemLayout::Cx && layout != MemLayout::NCx)
    return getStaticElementCount(shape);

  int64_t c = shape.back();
  if (c == mlir::ShapedType::kDynamic)
    return std::nullopt;

  constexpr int64_t cBlock = 64;
  int64_t blocks = (c + cBlock - 1) / cBlock;
  int64_t alignedC = blocks * cBlock;
  int64_t outerElements = 1;
  for (int64_t dim : shape.drop_back()) {
    if (dim == mlir::ShapedType::kDynamic)
      return std::nullopt;
    int64_t next = 0;
    if (!checkedMul(outerElements, dim, next))
      return std::nullopt;
    outerElements = next;
  }

  int64_t physicalElements = 0;
  if (!checkedMul(outerElements, alignedC, physicalElements))
    return std::nullopt;

  info.cBlock = cBlock;
  info.alignedC = alignedC;
  info.tailC = c % cBlock;
  return physicalElements;
}

static std::optional<int64_t>
getStaticByteSize(mlir::Type elementType, int64_t elementCount) {
  std::optional<int64_t> elementBits = getElementBitWidth(elementType);
  if (!elementBits || *elementBits <= 0)
    return std::nullopt;
  int64_t totalBits = 0;
  if (!checkedMul(elementCount, *elementBits, totalBits))
    return std::nullopt;
  return ceilDivToBytes(totalBits);
}

} // namespace

MemoryAttr wafer::getWaferMemoryAttr(mlir::MemRefType type) {
  return mlir::dyn_cast_or_null<MemoryAttr>(type.getMemorySpace());
}

bool wafer::isWaferMemRefType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  return memrefType && static_cast<bool>(getWaferMemoryAttr(memrefType));
}

bool wafer::isWaferSPMMemRefType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return false;
  MemoryAttr memory = getWaferMemoryAttr(memrefType);
  return memory && memory.getSpace() == MemorySpace::SPM;
}

bool wafer::isWaferDDRMemRefType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return false;
  MemoryAttr memory = getWaferMemoryAttr(memrefType);
  return memory && memory.getSpace() == MemorySpace::DDR;
}

std::optional<WaferPhysicalTensorInfo>
wafer::computeWaferPhysicalTensorInfo(mlir::MemRefType type) {
  MemoryAttr memory = getWaferMemoryAttr(type);
  if (!memory)
    return std::nullopt;

  WaferPhysicalTensorInfo info;
  info.logicalTensorType =
      mlir::RankedTensorType::get(type.getShape(), type.getElementType());
  info.memorySpace = memory.getSpace();
  info.layout = memory.getLayout();
  if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type.getElementType()))
    info.bitPackedElement = integerType.getWidth() == 1;

  std::optional<int64_t> compactElements = getStaticElementCount(type.getShape());
  if (compactElements) {
    if (std::optional<int64_t> bytes =
            getStaticByteSize(type.getElementType(), *compactElements))
      info.compactBytes = *bytes;
  }

  std::optional<int64_t> physicalElements =
      getStaticPhysicalElementCount(type.getShape(), info.layout, info);
  if (physicalElements) {
    if (std::optional<int64_t> bytes =
            getStaticByteSize(type.getElementType(), *physicalElements))
      info.physicalBytes = *bytes;
  }

  return info;
}

void WaferDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Wafer/IR/WaferAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Wafer/IR/WaferTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Wafer/IR/WaferOps.cpp.inc"
      >();
}
