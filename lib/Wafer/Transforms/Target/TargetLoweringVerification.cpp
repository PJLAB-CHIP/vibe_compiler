//===- Target LLVM lowering implementation -------------------------------===//

#include "MemoryPlanning/StaticIndexRange.h"
#include "Target/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"
#include "Wafer/Analysis/ControlFlow/SingleExecutionRegionFlow.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/Core/TargetCall.h"
#include "Wafer/Target/Core/TargetFormat.h"
#include "Wafer/Transforms/TargetConversion.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace wafer::target_llvm_detail {

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool isWaferInstruction(mlir::Operation *op) {
  return mlir::isa<WaferInstructionOpInterface, SyncNCCJoinOp>(op);
}

mlir::Value resolveTileRegionBoundaryValue(mlir::Value value) {
  while (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Value entry =
        analysis::getSingleExecutionRegionEntryOperand(blockArg);
    if (!entry)
      return value;
    value = entry;
  }
  return value;
}

mlir::Value getRootViewSource(mlir::Value value) {
  value = resolveTileRegionBoundaryValue(value);
  while (mlir::Operation *def = value.getDefiningOp()) {
    auto viewLike = mlir::dyn_cast<mlir::ViewLikeOpInterface>(def);
    if (!viewLike)
      return value;
    mlir::Value source =
        resolveTileRegionBoundaryValue(viewLike.getViewSource());
    if (source == value)
      return value;
    value = source;
  }
  return value;
}

mlir::Value resolveReturnedMemRefRoot(mlir::Value value) {
  while (true) {
    value = resolveTileRegionBoundaryValue(value);

    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      if (mlir::Value exit =
              analysis::getSingleExecutionRegionExitOperand(result)) {
        if (exit == value)
          return value;
        value = exit;
        continue;
      }
    }

    mlir::Value root = getRootViewSource(value);
    if (root == value)
      return value;
    value = root;
  }
}

mlir::FailureOr<int64_t> getStaticElementCount(mlir::Operation *op,
                                               mlir::MemRefType type,
                                               llvm::StringRef role) {
  if (!type.hasStaticShape())
    return op->emitError()
           << "unsupported_target_shape: " << role
           << " memref must have static shape for target LLVM lowering";

  int64_t elements = 1;
  for (int64_t dim : type.getShape()) {
    int64_t next = 0;
    if (!checkedMul(elements, dim, next))
      return op->emitError() << "target_shape_overflow: " << role
                             << " element count overflows int64";
    elements = next;
  }
  return elements;
}

mlir::FailureOr<int64_t>
getPhysicalTraversalElementCount(mlir::Operation *op, mlir::MemRefType type,
                                 llvm::StringRef role) {
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(type);
  if (!info || info->physicalElements <= 0)
    return op->emitError()
           << "unsupported_target_geometry: " << role
           << " requires a static positive physical traversal count";
  if (static_cast<uint64_t>(info->physicalElements) >
      std::numeric_limits<uint32_t>::max())
    return op->emitError() << "target_abi_narrowing: " << role
                           << " physical traversal count must fit uint32_t";
  return info->physicalElements;
}

mlir::FailureOr<int64_t> getStaticViewOffsetBytes(mlir::Operation *op,
                                                  mlir::MemRefType viewType,
                                                  llvm::StringRef role) {
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offsetElements = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(viewType, strides, offsetElements)) ||
      offsetElements == mlir::ShapedType::kDynamic)
    return op->emitError() << "unsupported_target_address: " << role
                           << " memref view must have a static layout offset";
  if (offsetElements < 0)
    return op->emitError() << "unsupported_target_address: " << role
                           << " memref view layout offset must be non-negative";

  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(viewType);
  if (!info)
    return op->emitError() << "unsupported_target_address: " << role
                           << " operand must be a Wafer memref";

  if (info->layout == MemLayout::Cx || info->layout == MemLayout::NCx) {
    if (offsetElements != 0)
      return op->emitError()
             << "unsupported_target_address: non-zero cx/ncx view offset is "
                "not supported by target LLVM lowering yet";
    return 0;
  }

  if (info->bitPackedElement) {
    if (offsetElements != 0)
      return op->emitError()
             << "unsupported_target_address: bitpacked view offset must be "
                "zero for target LLVM lowering";
    return 0;
  }

  if (info->elementBytes <= 0)
    return op->emitError()
           << "unsupported_target_address: cannot compute element byte size "
              "for "
           << role << " memref";

  int64_t offsetBytes = 0;
  if (!checkedMul(offsetElements, info->elementBytes, offsetBytes))
    return op->emitError() << "target_address_overflow: " << role
                           << " byte offset overflows int64";
  return offsetBytes;
}

mlir::FailureOr<DynamicSubviewAddressPlan>
analyzeDynamicTensorSubviewAddressing(mlir::memref::SubViewOp subviewOp) {
  mlir::MemRefType sourceType = subviewOp.getSourceType();
  mlir::MemRefType resultType = subviewOp.getType();
  MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
  MemoryAttr resultMemory = getWaferMemoryAttr(resultType);
  if (!sourceMemory || !resultMemory ||
      sourceMemory.getSpace() != resultMemory.getSpace() ||
      sourceMemory.getLayout() != MemLayout::Tensor ||
      resultMemory.getLayout() != MemLayout::Tensor)
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic subview requires matching "
              "tensor-layout source and result memory spaces";
  if (sourceType.getElementType() != resultType.getElementType())
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview must "
              "preserve the element type";
  if (!sourceType.hasStaticShape() || !resultType.hasStaticShape())
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview "
              "requires static source and result shapes";

  llvm::ArrayRef<int64_t> staticOffsets = subviewOp.getStaticOffsets();
  llvm::ArrayRef<int64_t> staticSizes = subviewOp.getStaticSizes();
  llvm::ArrayRef<int64_t> staticStrides = subviewOp.getStaticStrides();
  if (staticOffsets.size() != static_cast<size_t>(sourceType.getRank()) ||
      staticSizes.size() != staticOffsets.size() ||
      staticStrides.size() != staticOffsets.size())
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview rank "
              "does not match its offset/size/stride lists";
  if (llvm::any_of(staticSizes,
                   [](int64_t value) {
                     return mlir::ShapedType::isDynamic(value) || value < 0;
                   }) ||
      llvm::any_of(staticStrides, [](int64_t value) {
        return mlir::ShapedType::isDynamic(value) || value <= 0;
      }))
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview "
              "requires static non-negative sizes and positive strides";

  llvm::SmallVector<int64_t, 4> sourceStrides;
  int64_t sourceOffset = 0;
  if (mlir::failed(
          mlir::getStridesAndOffset(sourceType, sourceStrides, sourceOffset)) ||
      sourceStrides.size() != staticOffsets.size() ||
      llvm::any_of(sourceStrides, [](int64_t value) {
        return mlir::ShapedType::isDynamic(value) || value < 0;
      }))
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview source "
              "requires static non-negative memref strides";

  std::optional<WaferPhysicalTensorInfo> sourceInfo =
      computeWaferPhysicalTensorInfo(sourceType);
  if (!sourceInfo || sourceInfo->bitPackedElement ||
      sourceInfo->elementBytes <= 0)
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview "
              "requires a byte-addressable element type";

  DynamicSubviewAddressPlan plan;
  for (auto [offset, sourceStride] :
       llvm::zip_equal(staticOffsets, sourceStrides)) {
    int64_t byteStride = 0;
    if (!checkedMul(sourceStride, sourceInfo->elementBytes, byteStride))
      return subviewOp.emitError()
             << "target_address_overflow: dynamic tensor subview byte "
                "stride overflows int64";
    if (mlir::ShapedType::isDynamic(offset)) {
      plan.dynamicByteStrides.push_back(byteStride);
      continue;
    }
    int64_t byteOffset = 0;
    if (!checkedMul(offset, byteStride, byteOffset) ||
        !checkedAdd(plan.staticByteOffset, byteOffset, plan.staticByteOffset))
      return subviewOp.emitError()
             << "target_address_overflow: dynamic tensor subview static "
                "byte offset overflows int64";
  }
  if (plan.dynamicByteStrides.empty())
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview "
              "addressing requires at least one dynamic offset";
  if (plan.dynamicByteStrides.size() != subviewOp.getOffsets().size())
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview offset "
              "operand count does not match its layout";
  return plan;
}

namespace {
using StaticIndexRange = memory_planning::detail::StaticIndexRange;

static mlir::FailureOr<StaticIndexRange>
getStaticIndexRange(mlir::memref::SubViewOp subviewOp,
                    mlir::Value dynamicOffset, unsigned dynamicIndex) {
  memory_planning::detail::StaticIndexRangeResult result =
      memory_planning::detail::evaluateNonNegativeStaticIndexRange(
          dynamicOffset, subviewOp);
  using Failure = memory_planning::detail::StaticIndexRangeFailureKind;
  switch (result.failure) {
  case Failure::None:
    return result.range;
  case Failure::DynamicLoopBounds:
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview "
              "offset #"
           << dynamicIndex
           << " requires constant non-negative scf.for bounds and a positive "
              "constant step";
  case Failure::InvalidLoopBounds:
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview "
              "offset #"
           << dynamicIndex
           << " requires non-negative scf.for bounds and a positive constant "
              "step";
  case Failure::NonSingletonMultiplication:
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview "
              "offset #"
           << dynamicIndex
           << " multiplication requires one statically bounded singleton "
              "operand";
  case Failure::InvalidUnsignedDivision:
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview "
              "offset #"
           << dynamicIndex
           << " unsigned division requires non-negative static operands and "
              "a positive divisor";
  case Failure::ArithmeticOverflow:
    return subviewOp.emitError()
           << "target_address_overflow: dynamic tensor subview offset #"
           << dynamicIndex << " expression overflows int64";
  case Failure::NegativeRange:
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview offset #"
           << dynamicIndex << " range must be non-negative";
  case Failure::UnsupportedExpression:
    return subviewOp.emitError()
           << "unsupported_target_address: dynamic tensor subview offset #"
           << dynamicIndex
           << " must be a supported statically bounded index expression";
  }
  llvm_unreachable("unhandled static index range failure");
}

static mlir::LogicalResult
verifyDynamicTensorSubviewBounds(mlir::memref::SubViewOp subviewOp) {
  mlir::FailureOr<DynamicSubviewAddressPlan> plan =
      analyzeDynamicTensorSubviewAddressing(subviewOp);
  if (mlir::failed(plan))
    return mlir::failure();

  llvm::ArrayRef<int64_t> sourceShape = subviewOp.getSourceType().getShape();
  llvm::ArrayRef<int64_t> offsets = subviewOp.getStaticOffsets();
  llvm::ArrayRef<int64_t> sizes = subviewOp.getStaticSizes();
  llvm::ArrayRef<int64_t> strides = subviewOp.getStaticStrides();
  mlir::ValueRange dynamicOffsets = subviewOp.getOffsets();
  llvm::SmallVector<StaticIndexRange, 4> dynamicRanges;
  dynamicRanges.reserve(dynamicOffsets.size());
  bool unreachable = false;
  for (auto [index, offset] : llvm::enumerate(dynamicOffsets)) {
    mlir::FailureOr<StaticIndexRange> range =
        getStaticIndexRange(subviewOp, offset, index);
    if (mlir::failed(range))
      return mlir::failure();
    unreachable |= range->empty;
    dynamicRanges.push_back(*range);
  }
  if (unreachable)
    return mlir::success();

  unsigned dynamicIndex = 0;
  int64_t maximumDynamicByteOffset = plan->staticByteOffset;
  for (unsigned dim = 0; dim < offsets.size(); ++dim) {
    int64_t minimum = offsets[dim];
    int64_t maximum = offsets[dim];
    if (mlir::ShapedType::isDynamic(offsets[dim])) {
      minimum = dynamicRanges[dynamicIndex].min;
      maximum = dynamicRanges[dynamicIndex].max;
      int64_t dynamicByteOffset = 0;
      if (!checkedMul(maximum, plan->dynamicByteStrides[dynamicIndex],
                      dynamicByteOffset) ||
          !checkedAdd(maximumDynamicByteOffset, dynamicByteOffset,
                      maximumDynamicByteOffset))
        return subviewOp.emitError()
               << "target_address_overflow: dynamic tensor subview "
                  "maximum byte offset overflows int64";
      ++dynamicIndex;
    }

    if (sizes[dim] == 0)
      continue;
    int64_t span = 0;
    int64_t last = 0;
    if (!checkedMul(sizes[dim] - 1, strides[dim], span) ||
        !checkedAdd(maximum, span, last))
      return subviewOp.emitError()
             << "target_address_overflow: dynamic tensor subview source "
                "coordinate overflows int64";
    if (minimum < 0 || last >= sourceShape[dim])
      return subviewOp.emitError()
             << "target_geometry_mismatch: dynamic tensor subview "
                "dimension #"
             << dim << " may access source coordinate " << last
             << " outside static extent " << sourceShape[dim];
  }
  (void)maximumDynamicByteOffset;
  return mlir::success();
}
} // namespace

mlir::FailureOr<int64_t> getStaticUInt32SPMAddress(mlir::Operation *op,
                                                   mlir::Value value,
                                                   llvm::StringRef role) {
  auto viewType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!viewType || !isWaferSPMMemRefType(viewType))
    return op->emitError() << "target_abi_narrowing: " << role
                           << " address must be a Wafer SPM memref";

  mlir::FailureOr<int64_t> viewOffset =
      getStaticViewOffsetBytes(op, viewType, role);
  if (mlir::failed(viewOffset))
    return mlir::failure();

  mlir::Value root = getRootViewSource(value);
  while (auto castOp = root.getDefiningOp<mlir::memref::CastOp>())
    root = getRootViewSource(castOp.getSource());
  auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
  auto alloc = root.getDefiningOp<mlir::memref::AllocOp>();
  if (!rootType || !isWaferSPMMemRefType(rootType) || !alloc)
    return op->emitError() << "target_abi_narrowing: " << role
                           << " address must be statically rooted in a "
                              "planned SPM allocation";
  auto offset = alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
  if (!offset)
    return op->emitError() << "target_llvm_missing_spm_offset: " << role
                           << " SPM allocation is missing accepted "
                              "wafer.spm.offset";

  std::optional<WaferPhysicalTensorInfo> viewInfo =
      computeWaferPhysicalTensorInfo(viewType);
  std::optional<WaferPhysicalTensorInfo> rootInfo =
      computeWaferPhysicalTensorInfo(rootType);
  if (!viewInfo || !rootInfo || viewInfo->physicalBytes < 0 ||
      rootInfo->physicalBytes < 0)
    return op->emitError() << "target_abi_narrowing: " << role
                           << " physical address range must be statically "
                              "known";

  int64_t start = 0;
  if (!checkedAdd(offset.getOffset(), *viewOffset, start))
    return op->emitError() << "target_range_overflow: " << role
                           << " physical start address overflows int64";
  int64_t end = start;
  if (viewInfo->physicalBytes > 0 &&
      !checkedAdd(start, viewInfo->physicalBytes - 1, end))
    return op->emitError() << "target_range_overflow: " << role
                           << " physical end address overflows int64";

  int64_t rootEnd = offset.getOffset();
  if (rootInfo->physicalBytes > 0 &&
      !checkedAdd(offset.getOffset(), rootInfo->physicalBytes - 1, rootEnd))
    return op->emitError() << "target_range_overflow: " << role
                           << " root physical range overflows int64";
  if (end > rootEnd)
    return op->emitError() << "target_geometry_mismatch: " << role
                           << " view physical range exceeds its planned SPM "
                              "allocation";

  constexpr int64_t maxUInt32 =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  if (start > maxUInt32 || end > maxUInt32)
    return op->emitError() << "target_abi_narrowing: " << role
                           << " physical address range [" << start << ", "
                           << end << "] must fit uint32_t";
  return start;
}

mlir::FailureOr<int64_t> getStaticSPMAddress(mlir::Operation *op,
                                             mlir::Value value,
                                             llvm::StringRef role) {
  auto viewType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!viewType || !isWaferSPMMemRefType(viewType))
    return op->emitError() << "unsupported_target_address: " << role
                           << " must be a Wafer SPM memref";
  mlir::FailureOr<int64_t> viewOffset =
      getStaticViewOffsetBytes(op, viewType, role);
  if (mlir::failed(viewOffset))
    return mlir::failure();
  mlir::Value root = getRootViewSource(value);
  auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
  if (!allocation)
    return op->emitError() << "unsupported_target_address: " << role
                           << " must have a planned SPM allocation root";
  auto offset =
      allocation->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
  int64_t address = 0;
  if (!offset || !checkedAdd(offset.getOffset(), *viewOffset, address))
    return op->emitError() << "unsupported_target_address: " << role
                           << " has no representable accepted SPM offset";
  return address;
}

static mlir::FailureOr<LogicalFormat> getLogicalFormat(mlir::Operation *op,
                                                       mlir::Type elementType,
                                                       llvm::StringRef role) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 1:
      return LogicalFormat::Bool;
    case 8:
      return intType.isUnsigned() ? LogicalFormat::U8 : LogicalFormat::I8;
    case 16:
      return intType.isUnsigned() ? LogicalFormat::U16 : LogicalFormat::I16;
    case 32:
      return intType.isUnsigned() ? LogicalFormat::U32 : LogicalFormat::I32;
    case 64:
      return intType.isUnsigned() ? LogicalFormat::U64 : LogicalFormat::I64;
    default:
      break;
    }
  }
  if (mlir::isa<mlir::Float16Type>(elementType))
    return LogicalFormat::F16;
  if (mlir::isa<mlir::BFloat16Type>(elementType))
    return LogicalFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(elementType))
    return LogicalFormat::F32;
  if (mlir::isa<mlir::FloatTF32Type>(elementType))
    return LogicalFormat::TF32;

  return op->emitError() << "unsupported_target_dtype: " << role
                         << " element type " << elementType
                         << " has no logical target-format descriptor";
}

static mlir::FailureOr<TargetFormatEngine>
getTargetFormatEngine(mlir::Operation *op) {
  auto instruction = mlir::dyn_cast<WaferInstructionOpInterface>(op);
  if (!instruction)
    return op->emitError()
           << "unsupported_target_format: format-bearing operation does not "
              "implement WaferInstructionOpInterface";
  switch (instruction.getInstructionFamily()) {
  case InstrFamily::RDMA:
    return TargetFormatEngine::RDMA;
  case InstrFamily::WDMA:
    return TargetFormatEngine::WDMA;
  case InstrFamily::TDMA:
    return TargetFormatEngine::TDMA;
  case InstrFamily::CT:
    return TargetFormatEngine::CT;
  case InstrFamily::NE:
    return TargetFormatEngine::NE;
  case InstrFamily::DTE:
    return op->emitError()
           << "unsupported_target_format: byte-counted DTE does not carry a "
              "target data-format field";
  }
  llvm_unreachable("unknown instruction family");
}

static mlir::FailureOr<LogicalFormat>
getLogicalFormat(mlir::Operation *op, mlir::Value value, llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memrefType)
    return op->emitError() << "unsupported_target_dtype: " << role
                           << " operand must be a memref";
  return getLogicalFormat(op, memrefType.getElementType(), role);
}

int64_t getIntegerAttrValue(mlir::IntegerAttr attr) {
  return attr.getValue().getSExtValue();
}

int64_t getOptionalIntegerAttrValue(mlir::IntegerAttr attr, int64_t fallback) {
  if (!attr)
    return fallback;
  return getIntegerAttrValue(attr);
}

mlir::FailureOr<int64_t> getConstantScalarValue(mlir::Operation *op,
                                                mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return op->emitError()
           << "unsupported_target_scalar: target LLVM lowering requires "
              "scalar operands to be arith.constant";

  mlir::Attribute attr = constant.getValue();
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    const llvm::APInt &bits = intAttr.getValue();
    if (bits.getBitWidth() > 32)
      return op->emitError()
             << "unsupported_target_scalar: integer constant is wider than "
                "the raw 32-bit target scalar field";
    return bits.getZExtValue();
  }
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    llvm::APInt bits = floatAttr.getValue().bitcastToAPInt();
    if (bits.getBitWidth() > 64)
      return op->emitError()
             << "unsupported_target_scalar: floating constant is wider than "
                "64 bits";
    return bits.getZExtValue();
  }

  return op->emitError()
         << "unsupported_target_scalar: unsupported scalar constant attr "
         << attr;
}

bool isTargetRelationElementwiseKind(InstrElementwiseKind kind) {
  switch (kind) {
  case InstrElementwiseKind::Eq:
  case InstrElementwiseKind::Ne:
  case InstrElementwiseKind::Ge:
  case InstrElementwiseKind::Gt:
  case InstrElementwiseKind::Le:
  case InstrElementwiseKind::Lt:
    return true;
  default:
    return false;
  }
}

static mlir::LogicalResult verifyBitpackedFormatValue(mlir::Operation *op,
                                                      mlir::Value value,
                                                      llvm::StringRef role) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  std::optional<WaferPhysicalTensorInfo> info =
      type ? computeWaferPhysicalTensorInfo(type) : std::nullopt;
  if (!type || !type.getElementType().isInteger(1) || !info ||
      !info->bitPackedElement)
    return op->emitError() << "unsupported_target_format: " << role
                           << " must use the accepted bitpacked BOOL layout";
  mlir::FailureOr<int64_t> elements = getStaticElementCount(op, type, role);
  if (mlir::failed(elements))
    return mlir::failure();
  if (*elements < 0 ||
      static_cast<uint64_t>(*elements) > std::numeric_limits<uint32_t>::max())
    return op->emitError()
           << "target_abi_narrowing: " << role
           << " bitpacked BOOL logical element count must fit uint32_t";
  return mlir::success();
}

static mlir::LogicalResult
verifyTargetFormatConstraint(mlir::Operation *op, mlir::Value value,
                             llvm::StringRef role,
                             const TargetFormatEncodingRecord &record) {
  switch (record.constraint) {
  case TargetFormatConstraint::None:
    return mlir::success();
  case TargetFormatConstraint::BitpackedLayout:
    return verifyBitpackedFormatValue(op, value, role);
  case TargetFormatConstraint::BitpackedDMA: {
    if (mlir::failed(verifyBitpackedFormatValue(op, value, role)))
      return mlir::failure();
    mlir::IntegerAttr innerBytes;
    if (auto rdma = mlir::dyn_cast<InstrRDMAOp>(op))
      innerBytes = rdma.getInnerBytesAttr();
    else if (auto wdma = mlir::dyn_cast<InstrWDMAOp>(op))
      innerBytes = wdma.getInnerBytesAttr();
    else
      return op->emitError()
             << "unsupported_target_format: bitpacked DMA format constraint "
                "is attached to a non-RDMA/WDMA instruction";
    int64_t logicalElements = 0;
    if (!checkedMul(innerBytes.getInt(), 8, logicalElements) ||
        static_cast<uint64_t>(logicalElements) >
            std::numeric_limits<uint32_t>::max())
      return op->emitError()
             << "target_abi_narrowing: bitpacked BOOL inner_bytes * 8 must "
                "fit uint32_t logical element count";
    return mlir::success();
  }
  }
  llvm_unreachable("unknown target format constraint");
}

mlir::FailureOr<int64_t> getDataFormatCode(mlir::Operation *op,
                                           mlir::Value value,
                                           llvm::StringRef role) {
  mlir::FailureOr<LogicalFormat> format = getLogicalFormat(op, value, role);
  mlir::FailureOr<TargetFormatEngine> engine = getTargetFormatEngine(op);
  if (mlir::failed(format) || mlir::failed(engine))
    return mlir::failure();

  const LogicalFormatDescriptor *descriptor =
      findLogicalFormatDescriptor(*format);
  const TargetFormatEncodingRecord *record =
      findTargetFormatEncoding(*engine, *format);
  if (!descriptor || !record)
    return op->emitError() << "unsupported_target_format: engine '"
                           << stringifyTargetFormatEngine(*engine)
                           << "', format '" << stringifyLogicalFormat(*format)
                           << "' has no closed target-format registry row";
  if (mlir::failed(verifyTargetFormatConstraint(op, value, role, *record)))
    return mlir::failure();
  return record->dataFormatCode;
}

static TargetConvertParameterKind
toTargetConvertParameterKind(InstrConvertParameterKind kind) {
  switch (kind) {
  case InstrConvertParameterKind::None:
    return TargetConvertParameterKind::None;
  case InstrConvertParameterKind::RoundingMode:
    return TargetConvertParameterKind::RoundingMode;
  case InstrConvertParameterKind::ZeroPoint:
    return TargetConvertParameterKind::ZeroPoint;
  }
  llvm_unreachable("unknown instruction convert parameter kind");
}

static mlir::LogicalResult verifyTargetConvertRoute(InstrConvertOp op) {
  auto instruction = mlir::cast<WaferInstructionOpInterface>(op.getOperation());
  if (instruction.getInstructionFamily() != InstrFamily::CT)
    return op.emitError()
           << "unsupported_target_convert: convert must report CT family";

  uint16_t opcode = static_cast<uint16_t>(op.getKind());
  const TargetConvertRoute *route = findTargetConvertRoute(opcode);
  if (!route)
    return op.emitError() << "unsupported_target_convert: opcode " << opcode
                          << " is not registered for the current target";

  mlir::FailureOr<LogicalFormat> source =
      getLogicalFormat(op, op.getSource(), "convert source");
  mlir::FailureOr<LogicalFormat> destination =
      getLogicalFormat(op, op.getDest(), "convert destination");
  if (mlir::failed(source) || mlir::failed(destination))
    return mlir::failure();
  auto [canonicalSourceType, canonicalDestinationType] =
      getInstrConvertTypePair(op.getContext(), op.getKind());
  auto sourceType =
      mlir::cast<mlir::MemRefType>(op.getSource().getType()).getElementType();
  auto destinationType =
      mlir::cast<mlir::MemRefType>(op.getDest().getType()).getElementType();
  TargetConvertParameterKind parameterKind =
      toTargetConvertParameterKind(getInstrConvertParameterKind(op.getKind()));
  if (route->canonicalSpelling != stringifyEnum(op.getKind()) ||
      route->source != *source || route->destination != *destination ||
      route->parameterKind != parameterKind ||
      sourceType != canonicalSourceType ||
      destinationType != canonicalDestinationType)
    return op.emitError()
           << "unsupported_target_convert: kind, opcode, source/destination "
              "type and registry route do not conform";
  const TargetConvertRoute *typedRoute =
      findTargetConvertRoute(*source, *destination);
  if (typedRoute != route)
    return op.emitError()
           << "unsupported_target_convert: opcode and typed route registry "
              "lookups disagree";

  bool hasZeroPoint = static_cast<bool>(op.getZeroPointAttr());
  bool hasRoundingMode = static_cast<bool>(op.getRoundingModeAttr());
  switch (route->parameterKind) {
  case TargetConvertParameterKind::None:
    if (hasZeroPoint || hasRoundingMode)
      return op.emitError()
             << "unsupported_target_convert: parameterless route carries a "
                "zero-point or rounding-mode parameter";
    break;
  case TargetConvertParameterKind::RoundingMode:
    if (hasZeroPoint || !hasRoundingMode)
      return op.emitError()
             << "unsupported_target_convert: route requires exactly one "
                "rounding-mode parameter";
    break;
  case TargetConvertParameterKind::ZeroPoint:
    if (!hasZeroPoint || hasRoundingMode)
      return op.emitError()
             << "unsupported_target_convert: route requires exactly one "
                "zero-point parameter";
    break;
  }
  return mlir::success();
}

static mlir::LogicalResult
verifyTargetPhysicalTraversal(mlir::Operation *op, mlir::Value source,
                              mlir::Value dest, llvm::StringRef family) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto destType = mlir::dyn_cast<mlir::MemRefType>(dest.getType());
  if (!sourceType || !destType || sourceType.getShape() != destType.getShape())
    return op->emitError() << "unsupported_target_physical_traversal: "
                           << family
                           << " requires equal static logical memref shapes";
  analysis::IndexRelationResult identity =
      analysis::IndexRelation::identity(destType.getShape());
  if (!identity.isExact() ||
      mlir::failed(analysis::TransferRealizability::provePhysicalTraversal(
          sourceType, destType, destType.getShape(), *identity.get(),
          *identity.get())))
    return op->emitError()
           << "unsupported_target_physical_traversal: " << family
           << " source does not cover the compatible destination physical "
              "element traversal";
  return mlir::success();
}

static mlir::LogicalResult
verifyTargetCTPhysicalTraversal(mlir::Operation *op) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
      .Case<InstrElementwiseOp>([&](auto typedOp) {
        for (mlir::Value input : typedOp.getInputs())
          if (mlir::failed(verifyTargetPhysicalTraversal(
                  op, input, typedOp.getDest(), "elementwise")))
            return mlir::failure();
        auto destType =
            mlir::cast<mlir::MemRefType>(typedOp.getDest().getType());
        return mlir::succeeded(getPhysicalTraversalElementCount(
                   op, destType, "elementwise dest"))
                   ? mlir::success()
                   : mlir::failure();
      })
      .Case<InstrBit2FpOp>([&](auto typedOp) {
        if (mlir::failed(verifyTargetPhysicalTraversal(
                op, typedOp.getSource(), typedOp.getDest(), "bit2fp")))
          return mlir::failure();
        auto destType =
            mlir::cast<mlir::MemRefType>(typedOp.getDest().getType());
        return mlir::succeeded(getPhysicalTraversalElementCount(op, destType,
                                                                "bit2fp dest"))
                   ? mlir::success()
                   : mlir::failure();
      })
      .Case<InstrMaskMoveOp>([&](auto typedOp) {
        if (mlir::failed(verifyTargetPhysicalTraversal(
                op, typedOp.getSource(), typedOp.getDest(), "mask_move")) ||
            mlir::failed(verifyTargetPhysicalTraversal(
                op, typedOp.getMask(), typedOp.getDest(), "mask_move")))
          return mlir::failure();
        auto destType =
            mlir::cast<mlir::MemRefType>(typedOp.getDest().getType());
        return mlir::succeeded(getPhysicalTraversalElementCount(
                   op, destType, "mask_move dest"))
                   ? mlir::success()
                   : mlir::failure();
      })
      .Case<InstrConvertOp>([&](auto typedOp) {
        if (mlir::failed(verifyTargetPhysicalTraversal(
                op, typedOp.getSource(), typedOp.getDest(), "convert")))
          return mlir::failure();
        auto destType =
            mlir::cast<mlir::MemRefType>(typedOp.getDest().getType());
        return mlir::succeeded(getPhysicalTraversalElementCount(op, destType,
                                                                "convert dest"))
                   ? mlir::success()
                   : mlir::failure();
      })
      .Default([](mlir::Operation *) { return mlir::success(); });
}

static mlir::LogicalResult verifyTargetInstructionFormat(mlir::Operation *op) {
  if (mlir::failed(verifyTargetCTPhysicalTraversal(op)))
    return mlir::failure();
  auto verify = [&](mlir::Value value,
                    llvm::StringRef role) -> mlir::LogicalResult {
    return mlir::succeeded(getDataFormatCode(op, value, role))
               ? mlir::success()
               : mlir::failure();
  };

  return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
      .Case<InstrRDMAOp>(
          [&](auto typedOp) { return verify(typedOp.getDest(), "rdma dest"); })
      .Case<InstrWDMAOp>([&](auto typedOp) {
        return verify(typedOp.getSource(), "wdma source");
      })
      .Case<InstrGatherScatterOp, InstrDTESendOp, InstrDTERecvOp,
            InstrDTEWaitOp, SyncNCCJoinOp>(
          [&](auto) { return mlir::success(); })
      .Case<InstrFillOp>(
          [&](auto typedOp) { return verify(typedOp.getDest(), "fill dest"); })
      .Case<InstrElementwiseOp>([&](auto typedOp) -> mlir::LogicalResult {
        mlir::Value input = typedOp.getInputs().front();
        mlir::Value encoded = isTargetRelationElementwiseKind(typedOp.getKind())
                                  ? input
                                  : typedOp.getDest();
        if (mlir::failed(verify(encoded, "elementwise format")))
          return mlir::failure();
        if (typedOp.getDest() != encoded &&
            mlir::cast<mlir::MemRefType>(typedOp.getDest().getType())
                .getElementType()
                .isInteger(1))
          return verify(typedOp.getDest(), "elementwise BOOL destination");
        return mlir::success();
      })
      .Case<InstrBit2FpOp>([&](auto typedOp) {
        return verify(typedOp.getDest(), "bit2fp dest");
      })
      .Case<InstrMaskMoveOp>([&](auto typedOp) {
        return verify(typedOp.getDest(), "mask_move dest");
      })
      .Case<InstrReduceOp>([&](auto typedOp) -> mlir::LogicalResult {
        return verify(typedOp.getInput(), "reduce input");
      })
      .Case<InstrConvertOp>(
          [&](auto typedOp) { return verifyTargetConvertRoute(typedOp); })
      .Case<InstrGemmOp>([&](auto typedOp) -> mlir::LogicalResult {
        mlir::FailureOr<LogicalFormat> format =
            getLogicalFormat(op, typedOp.getDest(), "gemm destination");
        if (mlir::failed(format))
          return mlir::failure();
        if (*format == LogicalFormat::F32)
          return typedOp.emitError()
                 << "unsupported_target_instr: GEMM does not support f32";
        return verify(typedOp.getDest(), "gemm dest");
      })
      .Case<InstrConvOp>(
          [&](auto typedOp) { return verify(typedOp.getDest(), "conv dest"); })
      .Case<InstrPoolOp>([&](auto typedOp) {
        return verify(typedOp.getInput(), "pool input");
      })
      .Case<InstrUnpoolOp>([&](auto typedOp) {
        return verify(typedOp.getInput(), "unpool input");
      })
      .Case<InstrTDMADataMoveOp>([&](auto typedOp) {
        return verify(typedOp.getDest(), "tdma_data_move dest");
      })
      .Case<InstrPeripheralOp>([&](auto typedOp) {
        return verify(typedOp.getInputs().front(), "peripheral input");
      })
      .Default([&](mlir::Operation *unknown) {
        return unknown->emitError()
               << "unsupported_target_instr: instruction family has no "
                  "target format verification";
      });
}

mlir::LogicalResult verifyTargetInstructionFormats(mlir::ModuleOp moduleOp) {
  bool failed = false;
  moduleOp.walk([&](mlir::Operation *op) {
    if (!isWaferInstruction(op))
      return mlir::WalkResult::advance();
    if (mlir::failed(verifyNoSchemaFreeSemanticAttributes(op)) ||
        mlir::failed(verifyTargetInstructionFormat(op))) {
      failed = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return failed ? mlir::failure() : mlir::success();
}

mlir::LogicalResult verifyTargetSubviewAddresses(mlir::ModuleOp moduleOp) {
  mlir::WalkResult result =
      moduleOp.walk([&](mlir::memref::SubViewOp subviewOp) {
        if (subviewOp.getOffsets().empty())
          return mlir::WalkResult::advance();
        if (mlir::failed(verifyDynamicTensorSubviewBounds(subviewOp)))
          return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

} // namespace wafer::target_llvm_detail
