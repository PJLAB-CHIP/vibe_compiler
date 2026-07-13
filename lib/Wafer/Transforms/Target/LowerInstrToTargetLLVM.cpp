//===- LowerInstrToTargetLLVM.cpp - Lower instr IR to target LLVM ---------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

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

namespace wafer {
#define GEN_PASS_DEF_LOWERINSTRTOTARGETLLVMPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

struct AddressValue {
  mlir::Value dynamicBase;
  int64_t staticOffset = 0;
};

struct CalleeSignature {
  mlir::LLVM::LLVMFunctionType type;
};

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool isWaferTargetMetadata(mlir::Operation *op) {
  return mlir::isa<TargetTopologyOp, ExecutionMeshOp>(op);
}

static bool isWaferInstruction(mlir::Operation *op) {
  return mlir::isa<WaferInstructionOpInterface, SyncLocalFenceOp>(op);
}

static mlir::Value resolveTileRegionBoundaryValue(mlir::Value value) {
  while (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = blockArg.getOwner();
    if (!owner)
      return value;
    auto tileRegion =
        mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp());
    if (!tileRegion || tileRegion.getBody().empty() ||
        owner != &tileRegion.getBody().front())
      return value;
    if (blockArg.getArgNumber() >= tileRegion.getInputs().size())
      return value;
    value = tileRegion.getInputs()[blockArg.getArgNumber()];
  }
  return value;
}

static mlir::Value getRootViewSource(mlir::Value value) {
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

static mlir::Value resolveReturnedMemRefRoot(mlir::Value value) {
  while (true) {
    value = resolveTileRegionBoundaryValue(value);

    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(result.getOwner())) {
        if (tileRegion.getBody().empty())
          return value;
        auto yield = mlir::dyn_cast<TileYieldOp>(
            tileRegion.getBody().front().getTerminator());
        if (!yield || result.getResultNumber() >= yield.getValues().size())
          return value;
        mlir::Value yielded = yield.getValues()[result.getResultNumber()];
        if (yielded == value)
          return value;
        value = yielded;
        continue;
      }
    }

    mlir::Value root = getRootViewSource(value);
    if (root == value)
      return value;
    value = root;
  }
}

static mlir::FailureOr<int64_t> getStaticElementCount(mlir::Operation *op,
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

static mlir::FailureOr<int64_t>
getStaticViewOffsetBytes(mlir::Operation *op, mlir::MemRefType viewType,
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

static mlir::FailureOr<int64_t> getStaticUInt32MaskAddress(mlir::Operation *op,
                                                           mlir::Value value) {
  auto viewType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!viewType || !isWaferSPMMemRefType(viewType))
    return op->emitError()
           << "target_abi_narrowing: mask address must be a Wafer SPM memref";

  mlir::FailureOr<int64_t> viewOffset =
      getStaticViewOffsetBytes(op, viewType, "mask");
  if (mlir::failed(viewOffset))
    return mlir::failure();

  mlir::Value root = getRootViewSource(value);
  while (auto castOp = root.getDefiningOp<mlir::memref::CastOp>())
    root = getRootViewSource(castOp.getSource());
  auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
  auto alloc = root.getDefiningOp<mlir::memref::AllocOp>();
  if (!rootType || !isWaferSPMMemRefType(rootType) || !alloc)
    return op->emitError()
           << "target_abi_narrowing: mask address must be statically rooted "
              "in a planned SPM allocation";
  auto offset = alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
  if (!offset)
    return op->emitError()
           << "target_llvm_missing_spm_offset: mask SPM allocation is missing "
              "accepted wafer.spm.offset";

  std::optional<WaferPhysicalTensorInfo> viewInfo =
      computeWaferPhysicalTensorInfo(viewType);
  std::optional<WaferPhysicalTensorInfo> rootInfo =
      computeWaferPhysicalTensorInfo(rootType);
  if (!viewInfo || !rootInfo || viewInfo->physicalBytes < 0 ||
      rootInfo->physicalBytes < 0)
    return op->emitError()
           << "target_abi_narrowing: mask physical address range must be "
              "statically known";

  int64_t start = 0;
  if (!checkedAdd(offset.getOffset(), *viewOffset, start))
    return op->emitError()
           << "target_range_overflow: mask physical start address overflows "
              "int64";
  int64_t end = start;
  if (viewInfo->physicalBytes > 0 &&
      !checkedAdd(start, viewInfo->physicalBytes - 1, end))
    return op->emitError()
           << "target_range_overflow: mask physical end address overflows "
              "int64";

  int64_t rootEnd = offset.getOffset();
  if (rootInfo->physicalBytes > 0 &&
      !checkedAdd(offset.getOffset(), rootInfo->physicalBytes - 1, rootEnd))
    return op->emitError()
           << "target_range_overflow: mask root physical range overflows "
              "int64";
  if (end > rootEnd)
    return op->emitError()
           << "target_geometry_mismatch: mask view physical range exceeds "
              "its planned SPM allocation";

  constexpr int64_t maxUInt32 =
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  if (start > maxUInt32 || end > maxUInt32)
    return op->emitError()
           << "target_abi_narrowing: mask physical address range [" << start
           << ", " << end << "] must fit uint32_t";
  return start;
}

static mlir::FailureOr<int64_t> getDataFormatCode(mlir::Operation *op,
                                                  mlir::Type elementType,
                                                  llvm::StringRef role) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    switch (intType.getWidth()) {
    case 1:
      return 7; // BOOL
    case 8:
      return intType.isUnsigned() ? 8 : 0; // UINT8 / INT8
    case 16:
      return intType.isUnsigned() ? 9 : 1; // UINT16 / INT16
    case 32:
      return intType.isUnsigned() ? 10 : 4; // UINT32 / INT32
    case 64:
      return intType.isUnsigned() ? 12 : 11; // UINT64 / INT64
    default:
      break;
    }
  }
  if (mlir::isa<mlir::Float16Type>(elementType))
    return 2; // FP16
  if (mlir::isa<mlir::BFloat16Type>(elementType))
    return 3; // BF16
  if (mlir::isa<mlir::Float32Type>(elementType))
    return 5; // FP32

  return op->emitError() << "unsupported_target_dtype: cannot encode " << role
                         << " element type " << elementType;
}

static mlir::FailureOr<int64_t> getDataFormatCode(mlir::Operation *op,
                                                  mlir::Value value,
                                                  llvm::StringRef role) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memrefType)
    return op->emitError() << "unsupported_target_dtype: " << role
                           << " operand must be a memref";
  return getDataFormatCode(op, memrefType.getElementType(), role);
}

static int64_t getIntegerAttrValue(mlir::IntegerAttr attr) {
  return attr.getValue().getSExtValue();
}

static int64_t getOptionalIntegerAttrValue(mlir::IntegerAttr attr,
                                           int64_t fallback) {
  if (!attr)
    return fallback;
  return getIntegerAttrValue(attr);
}

static mlir::FailureOr<int64_t> getConstantScalarValue(mlir::Operation *op,
                                                       mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return op->emitError()
           << "unsupported_target_scalar: target LLVM lowering requires "
              "scalar operands to be arith.constant";

  mlir::Attribute attr = constant.getValue();
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return getIntegerAttrValue(intAttr);
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

static llvm::SmallString<64> makeTargetSymbol(llvm::StringRef base) {
  llvm::SmallString<64> result("wafer_tx81_");
  result += base;
  return result;
}

template <typename EnumT>
static llvm::SmallString<64> makeTargetSymbol(llvm::StringRef base,
                                              EnumT kind) {
  llvm::SmallString<64> result = makeTargetSymbol(base);
  result += "_";
  result += stringifyEnum(kind);
  return result;
}

static llvm::SmallString<64> makeConvSymbol(InstrConvKind kind) {
  switch (kind) {
  case InstrConvKind::Conv:
    return makeTargetSymbol("conv");
  case InstrConvKind::Depthwise:
    return makeTargetSymbol("depthwise_conv");
  case InstrConvKind::BackwardConv:
    return makeTargetSymbol("backward_conv");
  }
  llvm_unreachable("unknown InstrConvKind");
}

static bool isTargetRelationElementwiseKind(InstrElementwiseKind kind) {
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

struct FunctionLowering {
  mlir::OpBuilder &builder;
  mlir::MLIRContext *context;
  mlir::Type i64Type;
  mlir::Type i32Type;
  mlir::Type voidType;
  llvm::DenseMap<mlir::Value, mlir::Value> convertedValues;
  llvm::StringMap<CalleeSignature> &usedCallees;

  FunctionLowering(mlir::MLIRContext *context, mlir::OpBuilder &builder,
                   llvm::StringMap<CalleeSignature> &used)
      : builder(builder), context(context),
        i64Type(mlir::IntegerType::get(context, 64)),
        i32Type(mlir::IntegerType::get(context, 32)),
        voidType(mlir::LLVM::LLVMVoidType::get(context)), usedCallees(used) {}

  mlir::Value constantI64(mlir::Location loc, int64_t value) {
    return builder.create<mlir::LLVM::ConstantOp>(loc, i64Type, value);
  }

  mlir::Value constantI32(mlir::Location loc, int64_t value) {
    return builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, value);
  }

  void appendI32(mlir::Location loc, llvm::SmallVectorImpl<mlir::Value> &out,
                 int64_t value) {
    out.push_back(constantI32(loc, value));
  }

  void appendArrayI32(mlir::Location loc,
                      llvm::SmallVectorImpl<mlir::Value> &out,
                      llvm::ArrayRef<int64_t> values) {
    for (int64_t value : values)
      appendI32(loc, out, value);
  }

  mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
  getNHWCShape(mlir::Operation *op, mlir::MemRefType type,
               llvm::StringRef role) {
    if (!type.hasStaticShape())
      return op->emitError()
             << "unsupported_target_shape: " << role
             << " memref must have static shape for target LLVM lowering";
    if (type.getRank() > 4)
      return op->emitError()
             << "unsupported_target_shape: " << role
             << " memref rank must be <= 4 for fixed target CRT shape ABI";

    llvm::SmallVector<int64_t, 4> shape(4, 1);
    int64_t offset = 4 - type.getRank();
    for (auto [index, dim] : llvm::enumerate(type.getShape()))
      shape[offset + index] = dim;
    return shape;
  }

  mlir::FailureOr<AddressValue>
  addStaticOffset(mlir::Operation *op, AddressValue address, int64_t offset) {
    int64_t combined = 0;
    if (!checkedAdd(address.staticOffset, offset, combined))
      return op->emitError()
             << "target_address_overflow: byte offset overflows int64";
    address.staticOffset = combined;
    return address;
  }

  mlir::Value materializeAddress(mlir::Location loc, AddressValue address) {
    if (!address.dynamicBase)
      return constantI64(loc, address.staticOffset);
    if (address.staticOffset == 0)
      return address.dynamicBase;
    mlir::Value offset = constantI64(loc, address.staticOffset);
    return builder.create<mlir::LLVM::AddOp>(loc, address.dynamicBase, offset);
  }

  mlir::FailureOr<AddressValue>
  resolveAddress(mlir::Operation *op, mlir::Value value, llvm::StringRef role) {
    auto viewType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!viewType || !isWaferMemRefType(viewType))
      return op->emitError() << "unsupported_target_address: " << role
                             << " operand must be a Wafer memref";

    if (auto it = convertedValues.find(value); it != convertedValues.end())
      return AddressValue{it->second, 0};

    mlir::FailureOr<int64_t> viewOffset =
        getStaticViewOffsetBytes(op, viewType, role);
    if (mlir::failed(viewOffset))
      return mlir::failure();

    mlir::Value root = getRootViewSource(value);
    root = resolveTileRegionBoundaryValue(root);
    auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
    if (!rootType || !isWaferMemRefType(rootType))
      return op->emitError() << "unsupported_target_address: " << role
                             << " root must be a Wafer memref";

    AddressValue address;
    if (isWaferSPMMemRefType(rootType)) {
      auto alloc = root.getDefiningOp<mlir::memref::AllocOp>();
      if (!alloc)
        return op->emitError()
               << "target_llvm_missing_spm_offset: " << role
               << " SPM root must be a memref.alloc with accepted "
                  "wafer.spm.offset";
      auto offset =
          alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
      if (!offset)
        return op->emitError()
               << "target_llvm_missing_spm_offset: " << role
               << " SPM root is missing accepted wafer.spm.offset";
      address.staticOffset = offset.getOffset();
    } else if (isWaferDDRMemRefType(rootType)) {
      if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(root)) {
        auto it = convertedValues.find(arg);
        if (it == convertedValues.end())
          return op->emitError()
                 << "unsupported_target_address: " << role
                 << " DDR block argument is not a target LLVM function "
                    "parameter";
        address.dynamicBase = it->second;
      } else {
        auto alloc = root.getDefiningOp<mlir::memref::AllocOp>();
        if (!alloc)
          return op->emitError()
                 << "unsupported_target_address: " << role
                 << " DDR root must be either a function argument or "
                    "compiler-managed memref.alloc";
        auto offset =
            alloc->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
        if (!offset)
          return op->emitError()
                 << "target_llvm_missing_ddr_offset: " << role
                 << " compiler-managed DDR root is missing accepted "
                    "wafer.ddr.offset";
        address.staticOffset = offset.getOffset();
      }
    } else {
      return op->emitError()
             << "unsupported_target_address: unknown Wafer memory space";
    }

    return addStaticOffset(op, address, *viewOffset);
  }

  mlir::FailureOr<mlir::Value> materializeAddress(mlir::Operation *op,
                                                  mlir::Value value,
                                                  llvm::StringRef role) {
    mlir::FailureOr<AddressValue> address = resolveAddress(op, value, role);
    if (mlir::failed(address))
      return mlir::failure();
    return materializeAddress(op->getLoc(), *address);
  }

  void emitCall(mlir::Location loc, llvm::StringRef symbol,
                mlir::ValueRange args) {
    llvm::SmallVector<mlir::Type, 16> argTypes;
    for (mlir::Value arg : args)
      argTypes.push_back(arg.getType());
    mlir::LLVM::LLVMFunctionType functionType =
        mlir::LLVM::LLVMFunctionType::get(voidType, argTypes,
                                          /*isVarArg=*/false);
    auto [it, inserted] =
        usedCallees.try_emplace(symbol, CalleeSignature{functionType});
    if (!inserted && it->second.type != functionType)
      llvm_unreachable(
          "same target CRT symbol emitted with incompatible signature");
    builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange(), mlir::FlatSymbolRefAttr::get(context, symbol),
        args);
  }

  mlir::LogicalResult lowerRDMA(InstrRDMAOp op) {
    llvm::SmallVector<mlir::Value, 12> args;
    mlir::FailureOr<mlir::Value> source =
        materializeAddress(op, op.getSource(), "rdma source");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "rdma dest");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, op.getDest(), "rdma dest");
    if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*source);
    args.push_back(*dest);
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getByteCountAttr()));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getInnerBytesAttr()));
    appendArrayI32(op.getLoc(), args, op.getSrcStrides());
    appendArrayI32(op.getLoc(), args, op.getSrcIterations());
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("rdma"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerWDMA(InstrWDMAOp op) {
    llvm::SmallVector<mlir::Value, 12> args;
    mlir::FailureOr<mlir::Value> source =
        materializeAddress(op, op.getSource(), "wdma source");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "wdma dest");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, op.getSource(), "wdma source");
    if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*source);
    args.push_back(*dest);
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getByteCountAttr()));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getInnerBytesAttr()));
    appendArrayI32(op.getLoc(), args, op.getDstStrides());
    appendArrayI32(op.getLoc(), args, op.getDstIterations());
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("wdma"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerGatherScatter(InstrGatherScatterOp op) {
    mlir::FailureOr<AddressValue> source =
        resolveAddress(op, op.getSource(), "gather_scatter source");
    mlir::FailureOr<AddressValue> dest =
        resolveAddress(op, op.getDest(), "gather_scatter dest");
    if (mlir::failed(source) || mlir::failed(dest))
      return mlir::failure();
    source = addStaticOffset(
        op, *source, getOptionalIntegerAttrValue(op.getSrcOffsetAttr(), 0));
    dest = addStaticOffset(
        op, *dest, getOptionalIntegerAttrValue(op.getDstOffsetAttr(), 0));
    if (mlir::failed(source) || mlir::failed(dest))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 20> args;
    args.push_back(materializeAddress(op.getLoc(), *source));
    args.push_back(materializeAddress(op.getLoc(), *dest));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getByteCountAttr()));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getInnerBytesAttr()));
    appendArrayI32(op.getLoc(), args, op.getSrcStrides());
    appendArrayI32(op.getLoc(), args, op.getSrcIterations());
    appendArrayI32(op.getLoc(), args, op.getDstStrides());
    appendArrayI32(op.getLoc(), args, op.getDstIterations());
    emitCall(op.getLoc(), makeTargetSymbol("gather_scatter"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerFill(InstrFillOp op) {
    llvm::SmallVector<mlir::Value, 5> args;
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "fill dest");
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    mlir::FailureOr<int64_t> elements =
        getStaticElementCount(op, destType, "fill dest");
    mlir::FailureOr<int64_t> scalar = getConstantScalarValue(op, op.getValue());
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, destType.getElementType(), "fill dest");
    if (mlir::failed(dest) || mlir::failed(elements) || mlir::failed(scalar) ||
        mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*dest);
    appendI32(op.getLoc(), args, *scalar);
    appendI32(op.getLoc(), args, *elements);
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("memset"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerElementwise(InstrElementwiseOp op) {
    llvm::SmallVector<mlir::Value, 8> args;
    for (mlir::Value input : op.getInputs()) {
      mlir::FailureOr<mlir::Value> address =
          materializeAddress(op, input, "elementwise input");
      if (mlir::failed(address))
        return mlir::failure();
      args.push_back(*address);
    }
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "elementwise dest");
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    mlir::FailureOr<int64_t> elements =
        getStaticElementCount(op, destType, "elementwise dest");
    mlir::Value formatValue = isTargetRelationElementwiseKind(op.getKind())
                                  ? op.getInputs().front()
                                  : op.getDest();
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, formatValue, "elementwise format");
    if (mlir::failed(dest) || mlir::failed(elements) || mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*dest);
    appendI32(op.getLoc(), args, *elements);
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("elementwise", op.getKind()), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerBit2Fp(InstrBit2FpOp op) {
    llvm::SmallVector<mlir::Value, 5> args;
    mlir::FailureOr<mlir::Value> source =
        materializeAddress(op, op.getSource(), "bit2fp source");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "bit2fp dest");
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    mlir::FailureOr<int64_t> elements =
        getStaticElementCount(op, destType, "bit2fp dest");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, destType.getElementType(), "bit2fp dest");
    if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(elements) ||
        mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*source);
    args.push_back(*dest);
    appendI32(op.getLoc(), args, *elements);
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("bit2fp"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerMaskMove(InstrMaskMoveOp op) {
    llvm::SmallVector<mlir::Value, 6> args;
    mlir::FailureOr<mlir::Value> source =
        materializeAddress(op, op.getSource(), "mask_move source");
    mlir::FailureOr<int64_t> mask =
        getStaticUInt32MaskAddress(op, op.getMask());
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "mask_move dest");
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    mlir::FailureOr<int64_t> elements =
        getStaticElementCount(op, destType, "mask_move dest");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, destType.getElementType(), "mask_move dest");
    if (mlir::failed(source) || mlir::failed(mask) || mlir::failed(dest) ||
        mlir::failed(elements) || mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*source);
    args.push_back(constantI32(op.getLoc(), *mask));
    args.push_back(*dest);
    appendI32(op.getLoc(), args, *elements);
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("mask_move"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerReduce(InstrReduceOp op) {
    llvm::SmallVector<mlir::Value, 10> args;
    mlir::FailureOr<mlir::Value> input =
        materializeAddress(op, op.getInput(), "reduce input");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "reduce dest");
    auto inputType = mlir::cast<mlir::MemRefType>(op.getInput().getType());
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, inputType.getElementType(), "reduce input");
    mlir::FailureOr<llvm::SmallVector<int64_t, 4>> shape =
        getNHWCShape(op, inputType, "reduce input");
    if (mlir::failed(input) || mlir::failed(dest) || mlir::failed(fmt) ||
        mlir::failed(shape))
      return mlir::failure();
    args.push_back(*input);
    args.push_back(*dest);
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getDimAttr()));
    appendArrayI32(op.getLoc(), args, *shape);
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("reduce", op.getKind()), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerConvert(InstrConvertOp op) {
    llvm::SmallVector<mlir::Value, 8> args;
    mlir::FailureOr<mlir::Value> source =
        materializeAddress(op, op.getSource(), "convert source");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "convert dest");
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    mlir::FailureOr<int64_t> elements =
        getStaticElementCount(op, destType, "convert dest");
    if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(elements))
      return mlir::failure();
    args.push_back(*source);
    args.push_back(*dest);
    appendI32(op.getLoc(), args, *elements);
    appendI32(op.getLoc(), args,
              getOptionalIntegerAttrValue(op.getZeroPointAttr(), -1));
    appendI32(op.getLoc(), args,
              getOptionalIntegerAttrValue(op.getRoundingModeAttr(), -1));
    emitCall(op.getLoc(), makeTargetSymbol("convert", op.getKind()), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerGemm(InstrGemmOp op) {
    llvm::SmallVector<mlir::Value, 12> args;
    mlir::FailureOr<mlir::Value> lhs =
        materializeAddress(op, op.getLhs(), "gemm lhs");
    mlir::FailureOr<mlir::Value> rhs =
        materializeAddress(op, op.getRhs(), "gemm rhs");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "gemm dest");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, op.getDest(), "gemm dest");
    if (mlir::failed(lhs) || mlir::failed(rhs) || mlir::failed(dest) ||
        mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*lhs);
    args.push_back(*rhs);
    args.push_back(*dest);
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getMAttr()));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getKAttr()));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getNAttr()));
    appendI32(op.getLoc(), args,
              getOptionalIntegerAttrValue(op.getBatchCountAttr(), 1));
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("gemm"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerConv(InstrConvOp op) {
    llvm::SmallVector<mlir::Value, 32> args;
    mlir::FailureOr<mlir::Value> input =
        materializeAddress(op, op.getInput(), "conv input");
    mlir::FailureOr<mlir::Value> weight =
        materializeAddress(op, op.getWeight(), "conv weight");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "conv dest");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, op.getDest(), "conv dest");
    if (mlir::failed(input) || mlir::failed(weight) || mlir::failed(dest) ||
        mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*input);
    args.push_back(*weight);
    args.push_back(*dest);
    appendI32(op.getLoc(), args, static_cast<int64_t>(op.getKind()));
    appendArrayI32(op.getLoc(), args, op.getInputShape());
    appendArrayI32(op.getLoc(), args, op.getWeightShape());
    appendArrayI32(op.getLoc(), args, op.getOutputShape());
    appendArrayI32(op.getLoc(), args, op.getPads());
    appendArrayI32(op.getLoc(), args, op.getUnpads());
    appendArrayI32(op.getLoc(), args, op.getKernelStrides());
    appendArrayI32(op.getLoc(), args, op.getDilations());
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeConvSymbol(op.getKind()), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerPool(InstrPoolOp op) {
    llvm::SmallVector<mlir::Value, 24> args;
    mlir::FailureOr<mlir::Value> input =
        materializeAddress(op, op.getInput(), "pool input");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, op.getInput(), "pool input");
    if (mlir::failed(input) || mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*input);
    for (mlir::Value destValue : op.getDests()) {
      mlir::FailureOr<mlir::Value> dest =
          materializeAddress(op, destValue, "pool dest");
      if (mlir::failed(dest))
        return mlir::failure();
      args.push_back(*dest);
    }
    appendI32(op.getLoc(), args, static_cast<int64_t>(op.getKind()));
    appendArrayI32(op.getLoc(), args, op.getSourceShape());
    appendArrayI32(op.getLoc(), args, op.getDestShape());
    appendArrayI32(op.getLoc(), args, op.getPads());
    appendArrayI32(op.getLoc(), args, op.getKernelStrides());
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("pool", op.getKind()), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerUnpool(InstrUnpoolOp op) {
    llvm::SmallVector<mlir::Value, 20> args;
    mlir::FailureOr<mlir::Value> input =
        materializeAddress(op, op.getInput(), "unpool input");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "unpool dest");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, op.getInput(), "unpool input");
    if (mlir::failed(input) || mlir::failed(dest) || mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*input);
    args.push_back(*dest);
    appendI32(op.getLoc(), args, static_cast<int64_t>(op.getKind()));
    appendI32(op.getLoc(), args,
              getOptionalIntegerAttrValue(op.getIndexAttr(), -1));
    appendArrayI32(op.getLoc(), args, op.getSourceShape());
    appendArrayI32(op.getLoc(), args, op.getDestShape());
    appendArrayI32(op.getLoc(), args, op.getKernelStrides());
    appendI32(op.getLoc(), args, *fmt);
    emitCall(op.getLoc(), makeTargetSymbol("unpool", op.getKind()), args);
    return mlir::success();
  }

  bool isTransformLikeTDMA(InstrDataMoveKind kind) {
    switch (kind) {
    case InstrDataMoveKind::Pad:
    case InstrDataMoveKind::Img2Col:
      return false;
    case InstrDataMoveKind::Mirror:
    case InstrDataMoveKind::Transpose:
    case InstrDataMoveKind::Rotate90:
    case InstrDataMoveKind::Rotate180:
    case InstrDataMoveKind::Rotate270:
    case InstrDataMoveKind::Nchw2Nhwc:
    case InstrDataMoveKind::Nhwc2Nchw:
    case InstrDataMoveKind::TensorNom:
      return true;
    }
    llvm_unreachable("unknown InstrDataMoveKind");
  }

  mlir::LogicalResult lowerTDMADataMove(InstrTDMADataMoveOp op) {
    if (isTransformLikeTDMA(op.getKind()))
      return op.emitError()
             << "unsupported_target_instr: transform-like tdma_data_move kind "
                "reached target LLVM lowering";

    llvm::SmallVector<mlir::Value, 24> args;
    mlir::FailureOr<mlir::Value> source =
        materializeAddress(op, op.getSource(), "tdma_data_move source");
    mlir::FailureOr<mlir::Value> dest =
        materializeAddress(op, op.getDest(), "tdma_data_move dest");
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, op.getDest(), "tdma_data_move dest");
    if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(fmt))
      return mlir::failure();
    args.push_back(*source);
    args.push_back(*dest);
    appendArrayI32(op.getLoc(), args, op.getSourceShape());
    appendArrayI32(op.getLoc(), args, op.getDestShape());
    if (op.getPads())
      appendArrayI32(op.getLoc(), args, *op.getPads());
    if (op.getKernelStrides())
      appendArrayI32(op.getLoc(), args, *op.getKernelStrides());
    appendI32(op.getLoc(), args, *fmt);

    if (op.getKind() == InstrDataMoveKind::Pad) {
      emitCall(op.getLoc(), makeTargetSymbol("tdma_pad"), args);
      return mlir::success();
    }
    if (op.getKind() == InstrDataMoveKind::Img2Col) {
      emitCall(op.getLoc(), makeTargetSymbol("tdma_img2col"), args);
      return mlir::success();
    }
    llvm_unreachable("transform-like TDMA kinds handled above");
  }

  mlir::LogicalResult lowerPeripheral(InstrPeripheralOp op) {
    llvm::SmallVector<mlir::Value, 24> args;
    for (mlir::Value input : op.getInputs()) {
      mlir::FailureOr<mlir::Value> address =
          materializeAddress(op, input, "peripheral input");
      if (mlir::failed(address))
        return mlir::failure();
      args.push_back(*address);
    }
    for (mlir::Value destValue : op.getDests()) {
      mlir::FailureOr<mlir::Value> address =
          materializeAddress(op, destValue, "peripheral dest");
      if (mlir::failed(address))
        return mlir::failure();
      args.push_back(*address);
    }
    mlir::FailureOr<int64_t> fmt =
        getDataFormatCode(op, op.getInputs().front(), "peripheral input");
    if (mlir::failed(fmt))
      return mlir::failure();
    appendI32(op.getLoc(), args, static_cast<int64_t>(op.getKind()));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getElemCountAttr()));
    appendI32(op.getLoc(), args, *fmt);
    if (op.getSourceShape())
      appendArrayI32(op.getLoc(), args, *op.getSourceShape());
    if (op.getDestShape())
      appendArrayI32(op.getLoc(), args, *op.getDestShape());
    appendI32(op.getLoc(), args,
              getOptionalIntegerAttrValue(op.getLutElemCountAttr(), -1));
    appendI32(op.getLoc(), args,
              getOptionalIntegerAttrValue(op.getScaleAttr(), -1));
    appendI32(op.getLoc(), args,
              getOptionalIntegerAttrValue(op.getProbabilityAttr(), -1));
    appendI32(op.getLoc(), args,
              getOptionalIntegerAttrValue(op.getRoundingModeAttr(), -1));

    emitCall(op.getLoc(), makeTargetSymbol("peripheral", op.getKind()), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerLocalFence(SyncLocalFenceOp op) {
    emitCall(op.getLoc(), makeTargetSymbol("local_fence"), {});
    return mlir::success();
  }

  mlir::LogicalResult lowerInstruction(mlir::Operation *op) {
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
        .Case<InstrRDMAOp>([&](auto typedOp) { return lowerRDMA(typedOp); })
        .Case<InstrWDMAOp>([&](auto typedOp) { return lowerWDMA(typedOp); })
        .Case<InstrGatherScatterOp>(
            [&](auto typedOp) { return lowerGatherScatter(typedOp); })
        .Case<InstrFillOp>([&](auto typedOp) { return lowerFill(typedOp); })
        .Case<InstrElementwiseOp>(
            [&](auto typedOp) { return lowerElementwise(typedOp); })
        .Case<InstrBit2FpOp>([&](auto typedOp) { return lowerBit2Fp(typedOp); })
        .Case<InstrMaskMoveOp>(
            [&](auto typedOp) { return lowerMaskMove(typedOp); })
        .Case<InstrReduceOp>([&](auto typedOp) { return lowerReduce(typedOp); })
        .Case<InstrConvertOp>(
            [&](auto typedOp) { return lowerConvert(typedOp); })
        .Case<InstrGemmOp>([&](auto typedOp) { return lowerGemm(typedOp); })
        .Case<InstrConvOp>([&](auto typedOp) { return lowerConv(typedOp); })
        .Case<InstrPoolOp>([&](auto typedOp) { return lowerPool(typedOp); })
        .Case<InstrUnpoolOp>([&](auto typedOp) { return lowerUnpool(typedOp); })
        .Case<InstrTDMADataMoveOp>(
            [&](auto typedOp) { return lowerTDMADataMove(typedOp); })
        .Case<InstrPeripheralOp>(
            [&](auto typedOp) { return lowerPeripheral(typedOp); })
        .Case<SyncLocalFenceOp>(
            [&](auto typedOp) { return lowerLocalFence(typedOp); })
        .Default([&](mlir::Operation *unknown) {
          return unknown->emitError()
                 << "unsupported_target_instr: Wafer instruction op is not "
                    "handled by target LLVM lowering";
        });
  }
};

static mlir::LogicalResult flattenTileRegions(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<TileRegionOp, 8> tileRegions;
  moduleOp.walk(
      [&](TileRegionOp tileRegion) { tileRegions.push_back(tileRegion); });

  mlir::IRRewriter rewriter(moduleOp.getContext());
  for (TileRegionOp tileRegion : llvm::reverse(tileRegions)) {
    if (!tileRegion.getBody().hasOneBlock())
      return tileRegion.emitError()
             << "unsupported_target_structure: wafer.tile.region must have "
                "exactly one block";
    mlir::Block &body = tileRegion.getBody().front();
    auto yield = mlir::dyn_cast<TileYieldOp>(body.getTerminator());
    if (!yield)
      return tileRegion.emitError()
             << "unsupported_target_structure: wafer.tile.region must end "
                "with wafer.tile.yield";

    llvm::SmallVector<mlir::Value, 4> yieldedValues;
    for (mlir::Value value : yield.getValues()) {
      auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
      if (blockArg && blockArg.getOwner() == &body) {
        if (blockArg.getArgNumber() >= tileRegion.getInputs().size())
          return tileRegion.emitError()
                 << "unsupported_target_structure: tile yield block argument "
                    "has no matching boundary input";
        value = tileRegion.getInputs()[blockArg.getArgNumber()];
      }
      yieldedValues.push_back(value);
    }
    if (yieldedValues.size() != tileRegion.getNumResults())
      return tileRegion.emitError()
             << "unsupported_target_structure: tile yield/result count "
                "mismatch";

    rewriter.inlineBlockBefore(&body, tileRegion.getOperation(),
                               tileRegion.getInputs());
    rewriter.eraseOp(yield);
    for (auto [result, replacement] :
         llvm::zip_equal(tileRegion.getResults(), yieldedValues))
      result.replaceAllUsesWith(replacement);
    rewriter.eraseOp(tileRegion);
  }
  return mlir::success();
}

using AliasSummary = llvm::SmallVector<unsigned, 4>;

struct DirectCallGraph {
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::func::CallOp, 4>>
      calls;
  llvm::DenseSet<mlir::Operation *> calledFunctions;
  llvm::SmallVector<mlir::func::FuncOp, 8> calleeFirstOrder;
};

static mlir::LogicalResult
analyzeDirectCallGraph(mlir::ModuleOp moduleOp, DirectCallGraph &graph,
                       int64_t defaultDDRArenaArgumentIndex) {
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  for (mlir::func::FuncOp funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
    if (funcOp.isDeclaration())
      return funcOp.emitError()
             << "unsupported_target_call: external func.func declarations "
                "are not accepted by target LLVM lowering";
    functions.push_back(funcOp);

    for (auto [index, type] :
         llvm::enumerate(funcOp.getFunctionType().getInputs()))
      if (!isWaferDDRMemRefType(type) &&
          !(static_cast<int64_t>(index) == defaultDDRArenaArgumentIndex &&
            type.isInteger(64)))
        return funcOp.emitError()
               << "unsupported_target_function: argument #" << index
               << " must be a Wafer DDR memref target binding";

    mlir::WalkResult result =
        funcOp.walk([&](mlir::Operation *op) -> mlir::WalkResult {
          if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            mlir::func::FuncOp callee =
                moduleOp.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
            if (!callee) {
              call.emitError()
                  << "unsupported_target_call: unresolved direct callee @"
                  << call.getCallee();
              return mlir::WalkResult::interrupt();
            }
            graph.calls[funcOp.getOperation()].push_back(call);
            graph.calledFunctions.insert(callee.getOperation());
            return mlir::WalkResult::advance();
          }
          if (mlir::isa<mlir::CallOpInterface>(op)) {
            op->emitError()
                << "unsupported_target_call: indirect or unknown call-like "
                   "operation '"
                << op->getName() << "' is not supported";
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted())
      return mlir::failure();
  }

  llvm::DenseMap<mlir::Operation *, unsigned> state;
  auto visit = [&](auto &&self,
                   mlir::func::FuncOp funcOp) -> mlir::LogicalResult {
    unsigned &currentState = state[funcOp.getOperation()];
    if (currentState == 2)
      return mlir::success();
    if (currentState == 1)
      return funcOp.emitError()
             << "unsupported_target_call: recursive direct call graph is not "
                "supported";
    currentState = 1;
    for (mlir::func::CallOp call : graph.calls[funcOp.getOperation()]) {
      mlir::func::FuncOp callee =
          moduleOp.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (state[callee.getOperation()] == 1)
        return call.emitError()
               << "unsupported_target_call: recursive call to @"
               << call.getCallee() << " is not supported";
      if (mlir::failed(self(self, callee)))
        return mlir::failure();
    }
    currentState = 2;
    graph.calleeFirstOrder.push_back(funcOp);
    return mlir::success();
  };

  for (mlir::func::FuncOp funcOp : functions)
    if (mlir::failed(visit(visit, funcOp)))
      return mlir::failure();
  return mlir::success();
}

static mlir::FailureOr<unsigned> resolveDDRAliasToFunctionArgument(
    mlir::Value value, mlir::func::FuncOp funcOp, mlir::ModuleOp moduleOp,
    const llvm::DenseMap<mlir::Operation *, AliasSummary> &summaries,
    llvm::DenseSet<mlir::Value> &visiting) {
  while (true) {
    mlir::Value root = resolveReturnedMemRefRoot(value);
    if (root != value) {
      value = root;
      continue;
    }
    if (auto castOp = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = castOp.getSource();
      continue;
    }
    break;
  }

  if (!visiting.insert(value).second)
    return mlir::failure();
  auto eraseVisiting = llvm::make_scope_exit([&] { visiting.erase(value); });

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = blockArg.getOwner();
    if (owner == &funcOp.getBody().front()) {
      if (!isWaferDDRMemRefType(blockArg.getType()))
        return mlir::failure();
      return blockArg.getArgNumber();
    }

    std::optional<unsigned> commonAlias;
    bool sawPredecessor = false;
    for (mlir::Block *predecessor : owner->getPredecessors()) {
      auto branch =
          mlir::dyn_cast<mlir::BranchOpInterface>(predecessor->getTerminator());
      if (!branch)
        return mlir::failure();
      for (unsigned successorIndex = 0,
                    end = predecessor->getTerminator()->getNumSuccessors();
           successorIndex < end; ++successorIndex) {
        if (predecessor->getTerminator()->getSuccessor(successorIndex) != owner)
          continue;
        mlir::SuccessorOperands successorOperands =
            branch.getSuccessorOperands(successorIndex);
        if (blockArg.getArgNumber() >= successorOperands.size())
          return mlir::failure();
        mlir::Value incoming = successorOperands[blockArg.getArgNumber()];
        if (!incoming)
          return mlir::failure();
        if (incoming == value) {
          sawPredecessor = true;
          continue;
        }
        llvm::DenseSet<mlir::Value> branchVisiting = visiting;
        mlir::FailureOr<unsigned> alias = resolveDDRAliasToFunctionArgument(
            incoming, funcOp, moduleOp, summaries, branchVisiting);
        if (mlir::failed(alias) || (commonAlias && *commonAlias != *alias))
          return mlir::failure();
        commonAlias = *alias;
        sawPredecessor = true;
      }
    }
    if (sawPredecessor && commonAlias)
      return *commonAlias;
    return mlir::failure();
  }

  auto result = mlir::dyn_cast<mlir::OpResult>(value);
  if (!result)
    return mlir::failure();
  auto call = mlir::dyn_cast<mlir::func::CallOp>(result.getOwner());
  if (!call)
    return mlir::failure();
  mlir::func::FuncOp callee =
      moduleOp.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return mlir::failure();
  auto summaryIt = summaries.find(callee.getOperation());
  if (summaryIt == summaries.end() ||
      result.getResultNumber() >= summaryIt->second.size())
    return mlir::failure();
  unsigned calleeArg = summaryIt->second[result.getResultNumber()];
  if (calleeArg >= call.getNumOperands())
    return mlir::failure();
  return resolveDDRAliasToFunctionArgument(call.getOperand(calleeArg), funcOp,
                                           moduleOp, summaries, visiting);
}

static mlir::LogicalResult analyzeDDRAliasContracts(
    mlir::ModuleOp moduleOp, const DirectCallGraph &graph,
    llvm::DenseMap<mlir::Operation *, AliasSummary> &summaries) {
  for (mlir::func::FuncOp funcOp : graph.calleeFirstOrder) {
    unsigned resultCount = funcOp.getFunctionType().getNumResults();
    AliasSummary summary(resultCount);
    if (resultCount == 0) {
      summaries[funcOp.getOperation()] = std::move(summary);
      continue;
    }

    for (auto [index, type] :
         llvm::enumerate(funcOp.getFunctionType().getResults()))
      if (!isWaferDDRMemRefType(type))
        return funcOp.emitError()
               << "unsupported_target_alias: function result #" << index
               << " must be a Wafer DDR memref";

    llvm::SmallVector<std::optional<unsigned>, 4> aliases(resultCount);
    unsigned returnCount = 0;
    mlir::LogicalResult valid = mlir::success();
    funcOp.walk([&](mlir::func::ReturnOp returnOp) {
      if (mlir::failed(valid) ||
          returnOp->getParentOfType<mlir::func::FuncOp>() != funcOp)
        return;
      ++returnCount;
      if (returnOp.getNumOperands() != resultCount) {
        valid = returnOp.emitError()
                << "unsupported_target_alias: return operand count does not "
                   "match function result count";
        return;
      }
      for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
        llvm::DenseSet<mlir::Value> visiting;
        mlir::FailureOr<unsigned> alias = resolveDDRAliasToFunctionArgument(
            operand, funcOp, moduleOp, summaries, visiting);
        if (mlir::failed(alias)) {
          valid = returnOp.emitError()
                  << "unsupported_target_alias: DDR result #" << index
                  << " must provably alias one DDR function argument through "
                     "views, CFG forwarding, or direct-call alias results";
          return;
        }
        if (aliases[index] && *aliases[index] != *alias) {
          valid = returnOp.emitError()
                  << "unsupported_target_alias: DDR result #" << index
                  << " aliases different function arguments on different "
                     "return paths";
          return;
        }
        aliases[index] = *alias;
      }
    });
    if (mlir::failed(valid))
      return mlir::failure();
    if (returnCount == 0)
      return funcOp.emitError()
             << "unsupported_target_alias: result-bearing function has no "
                "func.return";
    for (unsigned index = 0; index < resultCount; ++index) {
      if (!aliases[index])
        return funcOp.emitError()
               << "unsupported_target_alias: could not prove result #" << index
               << " alias";
      summary[index] = *aliases[index];
    }
    summaries[funcOp.getOperation()] = std::move(summary);
  }
  return mlir::success();
}

static void dropRootAliasResults(mlir::ModuleOp moduleOp,
                                 const DirectCallGraph &graph) {
  for (mlir::func::FuncOp funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
    if (graph.calledFunctions.contains(funcOp.getOperation()) ||
        funcOp.getFunctionType().getNumResults() == 0)
      continue;
    funcOp.setFunctionType(mlir::FunctionType::get(
        moduleOp.getContext(), funcOp.getFunctionType().getInputs(), {}));
    funcOp.removeResAttrsAttr();
    funcOp.walk([&](mlir::func::ReturnOp returnOp) {
      if (returnOp->getParentOfType<mlir::func::FuncOp>() == funcOp)
        returnOp->setOperands({});
    });
  }
}

static void eraseTargetMetadata(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::Operation *, 4> toErase;
  moduleOp.walk([&](mlir::Operation *op) {
    if (op != moduleOp && isWaferTargetMetadata(op))
      toErase.push_back(op);
  });
  for (mlir::Operation *op : toErase)
    op->erase();
}

static mlir::FailureOr<int64_t>
getStaticViewDeltaBytes(mlir::Operation *op, mlir::MemRefType sourceType,
                        mlir::MemRefType resultType) {
  mlir::FailureOr<int64_t> sourceOffset =
      getStaticViewOffsetBytes(op, sourceType, "view source");
  mlir::FailureOr<int64_t> resultOffset =
      getStaticViewOffsetBytes(op, resultType, "view result");
  if (mlir::failed(sourceOffset) || mlir::failed(resultOffset))
    return mlir::failure();
  if (*resultOffset >= *sourceOffset)
    return *resultOffset - *sourceOffset;
  return -(*sourceOffset - *resultOffset);
}

static mlir::Value
applyStaticAddressDelta(mlir::ConversionPatternRewriter &rewriter,
                        mlir::Location loc, mlir::Value source, int64_t delta) {
  if (delta == 0)
    return source;
  assert(delta != std::numeric_limits<int64_t>::min() &&
         "static view offsets are non-negative int64 values");
  mlir::Type i64Type = rewriter.getI64Type();
  int64_t magnitude = delta < 0 ? -delta : delta;
  mlir::Value constant =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Type, magnitude);
  if (delta < 0)
    return rewriter.create<mlir::LLVM::SubOp>(loc, source, constant);
  return rewriter.create<mlir::LLVM::AddOp>(loc, source, constant);
}

struct TargetFuncOpLowering
    : public mlir::OpConversionPattern<mlir::func::FuncOp> {
  using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const auto *converter = getTypeConverter<mlir::LLVMTypeConverter>();
    llvm::SmallVector<mlir::Type, 8> argumentTypes;
    mlir::TypeConverter::SignatureConversion signature(
        funcOp.getNumArguments());
    for (auto [index, type] :
         llvm::enumerate(funcOp.getFunctionType().getInputs())) {
      mlir::Type converted = converter->convertType(type);
      if (!converted || !mlir::LLVM::isCompatibleType(converted))
        return funcOp.emitError()
               << "unsupported_target_function: cannot convert argument #"
               << index << " type " << type;
      argumentTypes.push_back(converted);
      signature.addInputs(index, converted);
    }

    mlir::Type resultType = mlir::LLVM::LLVMVoidType::get(funcOp.getContext());
    if (funcOp.getFunctionType().getNumResults() != 0) {
      resultType =
          converter->packFunctionResults(funcOp.getFunctionType().getResults());
      if (!resultType || !mlir::LLVM::isCompatibleType(resultType))
        return funcOp.emitError()
               << "unsupported_target_function: cannot convert function "
                  "results";
    }
    auto llvmType = mlir::LLVM::LLVMFunctionType::get(resultType, argumentTypes,
                                                      /*isVarArg=*/false);
    auto newFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getSymName(), llvmType);
    newFunc.setVisibility(funcOp.getVisibility());

    rewriter.inlineRegionBefore(funcOp.getBody(), newFunc.getBody(),
                                newFunc.end());
    if (mlir::failed(rewriter.convertRegionTypes(&newFunc.getBody(), *converter,
                                                 &signature)))
      return rewriter.notifyMatchFailure(funcOp,
                                         "failed to convert function blocks");
    rewriter.eraseOp(funcOp);
    return mlir::success();
  }
};

struct TargetCallOpLowering
    : public mlir::OpConversionPattern<mlir::func::CallOp> {
  using mlir::OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp callOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const auto *converter = getTypeConverter<mlir::LLVMTypeConverter>();
    mlir::Type packedResult;
    if (callOp.getNumResults() != 0) {
      packedResult = converter->packFunctionResults(callOp.getResultTypes());
      if (!packedResult)
        return callOp.emitError()
               << "unsupported_target_call: cannot convert direct-call "
                  "results";
    }

    auto llvmCall = rewriter.create<mlir::LLVM::CallOp>(
        callOp.getLoc(),
        packedResult ? mlir::TypeRange(packedResult) : mlir::TypeRange(),
        callOp.getCalleeAttr(), adaptor.getOperands());
    if (callOp.getNumResults() <= 1) {
      rewriter.replaceOp(callOp, llvmCall.getResults());
      return mlir::success();
    }

    llvm::SmallVector<mlir::Value, 4> results;
    for (unsigned index = 0; index < callOp.getNumResults(); ++index)
      results.push_back(rewriter.create<mlir::LLVM::ExtractValueOp>(
          callOp.getLoc(), llvmCall->getResult(0), index));
    rewriter.replaceOp(callOp, results);
    return mlir::success();
  }
};

struct TargetReturnOpLowering
    : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using mlir::OpConversionPattern<mlir::func::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp returnOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (returnOp.getNumOperands() <= 1) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(
          returnOp, mlir::TypeRange(), adaptor.getOperands());
      return mlir::success();
    }

    const auto *converter = getTypeConverter<mlir::LLVMTypeConverter>();
    mlir::Type packedType =
        converter->packFunctionResults(returnOp.getOperandTypes());
    if (!packedType)
      return returnOp.emitError()
             << "unsupported_target_function: cannot pack return operands";
    mlir::Value packed =
        rewriter.create<mlir::LLVM::UndefOp>(returnOp.getLoc(), packedType);
    for (auto [index, operand] : llvm::enumerate(adaptor.getOperands()))
      packed = rewriter.create<mlir::LLVM::InsertValueOp>(
          returnOp.getLoc(), packed, operand, index);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(
        returnOp, mlir::TypeRange(), packed);
    return mlir::success();
  }
};

struct TargetAllocOpLowering
    : public mlir::OpConversionPattern<mlir::memref::AllocOp> {
  TargetAllocOpLowering(mlir::LLVMTypeConverter &converter,
                        mlir::MLIRContext *context,
                        int64_t defaultDDRArenaArgumentIndex)
      : mlir::OpConversionPattern<mlir::memref::AllocOp>(converter, context),
        defaultDDRArenaArgumentIndex(defaultDDRArenaArgumentIndex) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocOp allocOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType type = allocOp.getType();
    if (!isWaferMemRefType(type))
      return allocOp.emitError()
             << "unsupported_target_address: memref.alloc must produce a "
                "Wafer memref";
    if (!adaptor.getOperands().empty())
      return allocOp.emitError()
             << "unsupported_target_address: dynamic or symbolic Wafer "
                "allocations are not supported";

    int64_t offset = 0;
    if (isWaferSPMMemRefType(type)) {
      auto attr =
          allocOp->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
      if (!attr)
        return allocOp.emitError()
               << "target_llvm_missing_spm_offset: SPM allocation is missing "
                  "accepted wafer.spm.offset";
      offset = attr.getOffset();
    } else if (isWaferDDRMemRefType(type)) {
      auto attr =
          allocOp->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
      if (!attr)
        return allocOp.emitError()
               << "unsupported_target_address: DDR allocation is missing "
                  "accepted wafer.ddr.offset";
      offset = attr.getOffset();
      if (defaultDDRArenaArgumentIndex < 0)
        return allocOp.emitError()
               << "unsupported_target_address: compiler-managed DDR "
                  "allocation requires an explicit arena base binding; "
                  "wafer.ddr.offset is arena-relative and cannot be lowered "
                  "as an absolute address";
      mlir::Operation *function = allocOp->getParentOp();
      while (function &&
             !mlir::isa<mlir::func::FuncOp, mlir::LLVM::LLVMFuncOp>(function))
        function = function->getParentOp();
      if (!function || function->getNumRegions() != 1 ||
          function->getRegion(0).empty() ||
          defaultDDRArenaArgumentIndex >=
              static_cast<int64_t>(
                  function->getRegion(0).front().getNumArguments()))
        return allocOp.emitError()
               << "unsupported_target_address: default DDR arena argument "
                  "index is outside the containing entry signature";
      mlir::Value arenaBase = function->getRegion(0).front().getArgument(
          defaultDDRArenaArgumentIndex);
      if (!arenaBase.getType().isInteger(64))
        return allocOp.emitError()
               << "unsupported_target_address: default DDR arena base must "
                  "lower to an i64 address";
      mlir::Value offsetValue = rewriter.create<mlir::LLVM::ConstantOp>(
          allocOp.getLoc(), rewriter.getI64Type(), offset);
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(allocOp, arenaBase,
                                                     offsetValue);
      return mlir::success();
    } else {
      return allocOp.emitError()
             << "unsupported_target_address: unknown Wafer memory space";
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        allocOp, rewriter.getI64Type(), offset);
    return mlir::success();
  }

private:
  int64_t defaultDDRArenaArgumentIndex;
};

struct TargetSubViewOpLowering
    : public mlir::OpConversionPattern<mlir::memref::SubViewOp> {
  using mlir::OpConversionPattern<mlir::memref::SubViewOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp subviewOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(subviewOp.getSource().getType());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(subviewOp.getType());
    if (!sourceType || !resultType || !isWaferMemRefType(sourceType) ||
        !isWaferMemRefType(resultType) ||
        !mlir::isa<mlir::IntegerType>(adaptor.getSource().getType()) ||
        mlir::cast<mlir::IntegerType>(adaptor.getSource().getType())
                .getWidth() != 64)
      return subviewOp.emitError()
             << "unsupported_target_address: subview must preserve a Wafer "
                "i64 address";
    mlir::FailureOr<int64_t> delta =
        getStaticViewDeltaBytes(subviewOp, sourceType, resultType);
    if (mlir::failed(delta))
      return mlir::failure();
    rewriter.replaceOp(subviewOp,
                       applyStaticAddressDelta(rewriter, subviewOp.getLoc(),
                                               adaptor.getSource(), *delta));
    return mlir::success();
  }
};

struct TargetReinterpretCastOpLowering
    : public mlir::OpConversionPattern<mlir::memref::ReinterpretCastOp> {
  using mlir::OpConversionPattern<
      mlir::memref::ReinterpretCastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(castOp.getSource().getType());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(castOp.getType());
    if (!sourceType || !resultType || !isWaferMemRefType(sourceType) ||
        !isWaferMemRefType(resultType))
      return castOp.emitError()
             << "unsupported_target_address: reinterpret_cast must preserve "
                "Wafer memory";
    mlir::FailureOr<int64_t> delta =
        getStaticViewDeltaBytes(castOp, sourceType, resultType);
    if (mlir::failed(delta))
      return mlir::failure();
    rewriter.replaceOp(castOp,
                       applyStaticAddressDelta(rewriter, castOp.getLoc(),
                                               adaptor.getSource(), *delta));
    return mlir::success();
  }
};

struct TargetMemRefCastOpLowering
    : public mlir::OpConversionPattern<mlir::memref::CastOp> {
  using mlir::OpConversionPattern<mlir::memref::CastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CastOp castOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!isWaferMemRefType(castOp.getSource().getType()) ||
        !isWaferMemRefType(castOp.getType()))
      return castOp.emitError()
             << "unsupported_target_address: memref.cast must preserve Wafer "
                "memory";
    rewriter.replaceOp(castOp, adaptor.getSource());
    return mlir::success();
  }
};

struct TargetDeallocOpLowering
    : public mlir::OpConversionPattern<mlir::memref::DeallocOp> {
  using mlir::OpConversionPattern<mlir::memref::DeallocOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DeallocOp deallocOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!isWaferMemRefType(deallocOp.getMemref().getType()))
      return mlir::failure();
    rewriter.eraseOp(deallocOp);
    return mlir::success();
  }
};

struct TargetInstructionOpLowering : public mlir::ConversionPattern {
  TargetInstructionOpLowering(mlir::LLVMTypeConverter &converter,
                              llvm::StringMap<CalleeSignature> &usedCallees)
      : mlir::ConversionPattern(converter, mlir::Pattern::MatchAnyOpTypeTag(),
                                /*benefit=*/10, &converter.getContext()),
        usedCallees(usedCallees) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!isWaferInstruction(op))
      return mlir::failure();
    if (mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(op))
      return op->emitError()
             << "unsupported_target_transport: DTE instruction requires a "
                "physical transport/endpoint binding and target CRT support";
    if (auto peripheral = mlir::dyn_cast<InstrPeripheralOp>(op);
        peripheral && peripheral.getKind() == InstrPeripheralKind::Factorize)
      return op->emitError()
             << "unsupported_target_operation: peripheral factorize lacks a "
                "proven production target semantic profile";
    if (operands.size() != op->getNumOperands())
      return op->emitError()
             << "target_llvm_lowering_failure: converted operand count "
                "mismatch";

    const auto *converter = getTypeConverter<mlir::LLVMTypeConverter>();
    for (mlir::Type resultType : op->getResultTypes())
      if (!mlir::isa<mlir::async::TokenType>(resultType) ||
          !converter->convertType(resultType))
        return op->emitError()
               << "unsupported_target_instr: instruction result type is not "
                  "a target async token";

    FunctionLowering lowering(op->getContext(), rewriter, usedCallees);
    for (auto [source, converted] :
         llvm::zip_equal(op->getOperands(), operands))
      lowering.convertedValues[source] = converted;
    rewriter.setInsertionPoint(op);
    if (mlir::failed(lowering.lowerInstruction(op)))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 2> replacements;
    for (mlir::Type resultType : op->getResultTypes()) {
      mlir::Type convertedType = converter->convertType(resultType);
      replacements.push_back(rewriter.create<mlir::LLVM::ConstantOp>(
          op->getLoc(), convertedType, 0));
    }
    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }

  llvm::StringMap<CalleeSignature> &usedCallees;
};

static mlir::LogicalResult
declareCallees(mlir::ModuleOp moduleOp,
               const llvm::StringMap<CalleeSignature> &callees) {
  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());

  llvm::SmallVector<llvm::StringRef, 32> sortedNames;
  for (const auto &entry : callees)
    sortedNames.push_back(entry.getKey());
  llvm::sort(sortedNames);

  for (llvm::StringRef name : sortedNames) {
    if (mlir::Operation *existing = moduleOp.lookupSymbol(name)) {
      auto llvmFunc = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(existing);
      auto found = callees.find(name);
      if (!llvmFunc || found == callees.end() ||
          llvmFunc.getFunctionType() != found->second.type)
        return existing->emitError()
               << "target_llvm_symbol_collision: existing @" << name
               << " does not match the target CRT declaration";
      continue;
    }
    auto found = callees.find(name);
    assert(found != callees.end() && "callee name must have a signature");
    const CalleeSignature &signature = found->second;
    builder.create<mlir::LLVM::LLVMFuncOp>(moduleOp.getLoc(), name,
                                           signature.type);
  }
  return mlir::success();
}

static mlir::LogicalResult lowerSCFToControlFlow(mlir::ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(moduleOp.getContext());
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::ConversionTarget target(*moduleOp.getContext());
  target.addIllegalDialect<mlir::scf::SCFDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  if (mlir::failed(
          mlir::applyPartialConversion(moduleOp, target, std::move(patterns))))
    return moduleOp.emitError()
           << "unsupported_target_structure: failed to lower SCF control "
              "flow to CFG";
  return mlir::success();
}

static mlir::LogicalResult
lowerModuleInPlace(mlir::ModuleOp moduleOp,
                   int64_t defaultDDRArenaArgumentIndex) {
  if (mlir::failed(flattenTileRegions(moduleOp)))
    return mlir::failure();

  DirectCallGraph callGraph;
  if (mlir::failed(analyzeDirectCallGraph(moduleOp, callGraph,
                                          defaultDDRArenaArgumentIndex)))
    return mlir::failure();
  if (mlir::failed(lowerSCFToControlFlow(moduleOp)))
    return mlir::failure();
  llvm::DenseMap<mlir::Operation *, AliasSummary> aliasSummaries;
  if (mlir::failed(
          analyzeDDRAliasContracts(moduleOp, callGraph, aliasSummaries)))
    return mlir::failure();
  dropRootAliasResults(moduleOp, callGraph);
  eraseTargetMetadata(moduleOp);

  mlir::LLVMTypeConverter converter(moduleOp.getContext());
  converter.addConversion(
      [&](mlir::MemRefType type) -> std::optional<mlir::Type> {
        if (!isWaferMemRefType(type))
          return mlir::Type();
        return mlir::IntegerType::get(moduleOp.getContext(), 64);
      });
  converter.addConversion([&](mlir::async::TokenType) -> mlir::Type {
    return mlir::IntegerType::get(moduleOp.getContext(), 1);
  });

  llvm::StringMap<CalleeSignature> usedCallees;
  mlir::RewritePatternSet patterns(moduleOp.getContext());
  patterns
      .add<TargetFuncOpLowering, TargetCallOpLowering, TargetReturnOpLowering,
           TargetSubViewOpLowering, TargetReinterpretCastOpLowering,
           TargetMemRefCastOpLowering, TargetDeallocOpLowering>(
          converter, moduleOp.getContext());
  patterns.add<TargetAllocOpLowering>(converter, moduleOp.getContext(),
                                      defaultDDRArenaArgumentIndex);
  patterns.add<TargetInstructionOpLowering>(converter, usedCallees);
  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);

  mlir::ConversionTarget target(*moduleOp.getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                           mlir::cf::ControlFlowDialect,
                           mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                           WaferDialect>();
  if (mlir::failed(
          mlir::applyFullConversion(moduleOp, target, std::move(patterns))))
    return moduleOp.emitError()
           << "target_llvm_lowering_failure: full target LLVM conversion "
              "failed";

  if (mlir::failed(declareCallees(moduleOp, usedCallees)))
    return mlir::failure();

  mlir::Operation *remainingNonLLVMOp = nullptr;
  moduleOp.walk([&](mlir::Operation *op) {
    if (op == moduleOp || mlir::isa<mlir::LLVM::LLVMFuncOp>(op) ||
        (op->getDialect() &&
         op->getDialect()->getNamespace() ==
             mlir::LLVM::LLVMDialect::getDialectNamespace()))
      return mlir::WalkResult::advance();
    remainingNonLLVMOp = op;
    return mlir::WalkResult::interrupt();
  });
  if (!remainingNonLLVMOp)
    return mlir::success();

  remainingNonLLVMOp->emitError()
      << "target_llvm_lowering_failure: non-LLVM operation remains after "
         "lowering";
  return mlir::failure();
}

struct LowerInstrToTargetLLVMPass
    : public impl::LowerInstrToTargetLLVMPassBase<LowerInstrToTargetLLVMPass> {
  using impl::LowerInstrToTargetLLVMPassBase<
      LowerInstrToTargetLLVMPass>::LowerInstrToTargetLLVMPassBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::OwningOpRef<mlir::ModuleOp> loweredModule = moduleOp.clone();
    if (mlir::failed(
            lowerModuleInPlace(*loweredModule, defaultDDRArenaArgumentIndex))) {
      signalPassFailure();
      return;
    }

    moduleOp->setAttrs((*loweredModule)->getAttrs());
    moduleOp.getBodyRegion().takeBody(loweredModule->getBodyRegion());
  }
};

} // namespace
} // namespace wafer
