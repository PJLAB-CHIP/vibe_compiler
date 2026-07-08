//===- LowerInstrToTargetLLVM.cpp - Lower instr IR to target LLVM ---------===//

#include "Wafer/Transforms/Passes.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace wafer {
#define GEN_PASS_DEF_LOWERINSTRTOTARGETLLVMPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

struct AddressValue {
  mlir::Value dynamicBase;
  int64_t staticOffset = 0;
};

struct LoweredFunction {
  mlir::func::FuncOp source;
  mlir::LLVM::LLVMFuncOp lowered;
  std::string symbolName;
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

static bool isWaferDialectOp(mlir::Operation *op) {
  mlir::Dialect *dialect = op->getDialect();
  return dialect &&
         dialect->getNamespace() == WaferDialect::getDialectNamespace();
}

static bool isWaferTargetMetadata(mlir::Operation *op) {
  return mlir::isa<TargetTopologyOp, ExecutionMeshOp>(op);
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

static bool isFunctionArgumentOf(mlir::Value value, mlir::func::FuncOp funcOp) {
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
  return blockArg && blockArg.getOwner() == &funcOp.getBody().front();
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
  mlir::ModuleOp moduleOp;
  mlir::OpBuilder builder;
  mlir::MLIRContext *context;
  mlir::Type i64Type;
  mlir::Type i32Type;
  mlir::Type voidType;
  llvm::DenseMap<mlir::Value, mlir::Value> ddrArguments;
  llvm::StringMap<CalleeSignature> &usedCallees;

  FunctionLowering(mlir::ModuleOp moduleOp,
                   llvm::StringMap<CalleeSignature> &used)
      : moduleOp(moduleOp), builder(moduleOp.getContext()),
        context(moduleOp.getContext()),
        i64Type(mlir::IntegerType::get(moduleOp.getContext(), 64)),
        i32Type(mlir::IntegerType::get(moduleOp.getContext(), 32)),
        voidType(mlir::LLVM::LLVMVoidType::get(moduleOp.getContext())),
        usedCallees(used) {}

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
        auto it = ddrArguments.find(arg);
        if (it == ddrArguments.end())
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
      llvm_unreachable("same target CRT symbol emitted with incompatible signature");
    builder.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange(), mlir::FlatSymbolRefAttr::get(context, symbol),
        args);
  }

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
  lowerFunctionSkeleton(mlir::func::FuncOp funcOp) {
    for (mlir::Type resultType : funcOp.getFunctionType().getResults()) {
      if (!isWaferDDRMemRefType(resultType))
        return funcOp.emitError()
               << "unsupported_target_function: target LLVM lowering can "
                  "only drop Wafer DDR memref function results";
    }

    mlir::LogicalResult validReturns = mlir::success();
    funcOp.walk([&](mlir::func::ReturnOp returnOp) {
      if (mlir::failed(validReturns))
        return;
      if (returnOp.getNumOperands() !=
          funcOp.getFunctionType().getNumResults()) {
        validReturns = returnOp.emitError()
                       << "unsupported_target_function: func.return operand "
                          "count must match function results";
        return;
      }
      for (mlir::Value operand : returnOp.getOperands()) {
        if (!isWaferDDRMemRefType(operand.getType())) {
          validReturns = returnOp.emitError()
                         << "unsupported_target_function: target LLVM "
                            "lowering can only drop Wafer DDR memref return "
                            "operands";
          return;
        }
        mlir::Value root = resolveReturnedMemRefRoot(operand);
        if (!isFunctionArgumentOf(root, funcOp) ||
            !isWaferDDRMemRefType(root.getType())) {
          validReturns = returnOp.emitError()
                         << "unsupported_target_function: dropped DDR memref "
                            "return must alias a target LLVM function "
                            "argument";
          return;
        }
      }
    });
    if (mlir::failed(validReturns))
      return mlir::failure();

    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (mlir::BlockArgument arg : funcOp.getArguments()) {
      if (!isWaferDDRMemRefType(arg.getType()))
        return funcOp.emitError()
               << "unsupported_target_function: only Wafer DDR memref "
                  "function arguments lower to target LLVM kernel arguments";
      argTypes.push_back(i64Type);
    }

    mlir::LLVM::LLVMFunctionType functionType =
        mlir::LLVM::LLVMFunctionType::get(voidType, argTypes,
                                          /*isVarArg=*/false);
    builder.setInsertionPoint(funcOp);
    llvm::SmallString<64> tempName("__wafer_lowered_");
    tempName += funcOp.getSymName();
    mlir::LLVM::LLVMFuncOp llvmFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
        funcOp.getLoc(), tempName, functionType);
    mlir::Block *entry = llvmFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entry);

    for (auto [sourceArg, loweredArg] :
         llvm::zip_equal(funcOp.getArguments(), entry->getArguments()))
      ddrArguments[sourceArg] = loweredArg;

    return llvmFunc;
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
    appendI32(op.getLoc(), args,
                   getIntegerAttrValue(op.getByteCountAttr()));
    appendI32(op.getLoc(), args,
                   getIntegerAttrValue(op.getInnerBytesAttr()));
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
    appendI32(op.getLoc(), args,
                   getIntegerAttrValue(op.getByteCountAttr()));
    appendI32(op.getLoc(), args,
                   getIntegerAttrValue(op.getInnerBytesAttr()));
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
    appendI32(op.getLoc(), args,
                   getIntegerAttrValue(op.getByteCountAttr()));
    appendI32(op.getLoc(), args,
                   getIntegerAttrValue(op.getInnerBytesAttr()));
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
    mlir::Value formatValue =
        isTargetRelationElementwiseKind(op.getKind()) ? op.getInputs().front()
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
    mlir::FailureOr<mlir::Value> mask =
        materializeAddress(op, op.getMask(), "mask_move mask");
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
    args.push_back(*mask);
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
    appendI32(op.getLoc(), args,
                   getIntegerAttrValue(op.getElemCountAttr()));
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

  mlir::LogicalResult lowerDTESend(InstrDTESendOp op) {
    llvm::SmallVector<mlir::Value, 4> args;
    mlir::FailureOr<mlir::Value> buffer =
        materializeAddress(op, op.getBuffer(), "dte_send buffer");
    if (mlir::failed(buffer))
      return mlir::failure();
    args.push_back(*buffer);
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getPeerAttr()));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getBytesAttr()));
    emitCall(op.getLoc(), makeTargetSymbol("dte_send"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerDTERecv(InstrDTERecvOp op) {
    llvm::SmallVector<mlir::Value, 4> args;
    mlir::FailureOr<mlir::Value> buffer =
        materializeAddress(op, op.getBuffer(), "dte_recv buffer");
    if (mlir::failed(buffer))
      return mlir::failure();
    args.push_back(*buffer);
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getPeerAttr()));
    appendI32(op.getLoc(), args, getIntegerAttrValue(op.getBytesAttr()));
    emitCall(op.getLoc(), makeTargetSymbol("dte_recv"), args);
    return mlir::success();
  }

  mlir::LogicalResult lowerDTEWait(InstrDTEWaitOp op) {
    llvm::SmallVector<mlir::Value, 1> args;
    appendI32(op.getLoc(), args, op.getTokens().size());
    emitCall(op.getLoc(), makeTargetSymbol("dte_wait"), args);
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
        .Case<InstrDTESendOp>(
            [&](auto typedOp) { return lowerDTESend(typedOp); })
        .Case<InstrDTERecvOp>(
            [&](auto typedOp) { return lowerDTERecv(typedOp); })
        .Case<InstrDTEWaitOp>(
            [&](auto typedOp) { return lowerDTEWait(typedOp); })
        .Case<SyncLocalFenceOp>(
            [&](auto typedOp) { return lowerLocalFence(typedOp); })
        .Default([&](mlir::Operation *unknown) {
          return unknown->emitError()
                 << "unsupported_target_instr: Wafer instruction op is not "
                    "handled by target LLVM lowering";
        });
  }

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
  lowerFunction(mlir::func::FuncOp funcOp) {
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> llvmFunc =
        lowerFunctionSkeleton(funcOp);
    if (mlir::failed(llvmFunc))
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *, 32> instrOps;
    funcOp.walk([&](mlir::Operation *op) {
      if (mlir::isa<InstrRDMAOp, InstrWDMAOp, InstrGatherScatterOp, InstrFillOp,
                    InstrElementwiseOp, InstrBit2FpOp, InstrMaskMoveOp,
                    InstrReduceOp, InstrConvertOp, InstrGemmOp, InstrConvOp,
                    InstrPoolOp, InstrUnpoolOp, InstrTDMADataMoveOp,
                    InstrPeripheralOp, InstrDTESendOp, InstrDTERecvOp,
                    InstrDTEWaitOp, SyncLocalFenceOp>(op))
        instrOps.push_back(op);
    });

    for (mlir::Operation *op : instrOps)
      if (mlir::failed(lowerInstruction(op)))
        return mlir::failure();

    builder.create<mlir::LLVM::ReturnOp>(funcOp.getLoc(), mlir::ValueRange());
    return *llvmFunc;
  }
};

static void eraseTargetMetadata(mlir::ModuleOp moduleOp) {
  llvm::SmallVector<mlir::Operation *, 4> toErase;
  moduleOp.walk([&](mlir::Operation *op) {
    if (op != moduleOp && isWaferTargetMetadata(op))
      toErase.push_back(op);
  });
  for (mlir::Operation *op : toErase)
    op->erase();
}

static void declareCallees(mlir::ModuleOp moduleOp,
                           const llvm::StringMap<CalleeSignature> &callees) {
  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());

  llvm::SmallVector<llvm::StringRef, 32> sortedNames;
  for (const auto &entry : callees)
    sortedNames.push_back(entry.getKey());
  llvm::sort(sortedNames);

  for (llvm::StringRef name : sortedNames) {
    if (moduleOp.lookupSymbol(name))
      continue;
    auto found = callees.find(name);
    assert(found != callees.end() && "callee name must have a signature");
    const CalleeSignature &signature = found->second;
    builder.create<mlir::LLVM::LLVMFuncOp>(moduleOp.getLoc(), name,
                                           signature.type);
  }
}

struct LowerInstrToTargetLLVMPass
    : public impl::LowerInstrToTargetLLVMPassBase<LowerInstrToTargetLLVMPass> {
  using impl::LowerInstrToTargetLLVMPassBase<
      LowerInstrToTargetLLVMPass>::LowerInstrToTargetLLVMPassBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    llvm::SmallVector<mlir::func::FuncOp, 4> functions;
    moduleOp.walk(
        [&](mlir::func::FuncOp funcOp) { functions.push_back(funcOp); });

    llvm::StringMap<CalleeSignature> usedCallees;
    llvm::SmallVector<LoweredFunction, 4> loweredFunctions;
    for (mlir::func::FuncOp funcOp : functions) {
      FunctionLowering lowering(moduleOp, usedCallees);
      mlir::FailureOr<mlir::LLVM::LLVMFuncOp> lowered =
          lowering.lowerFunction(funcOp);
      if (mlir::failed(lowered)) {
        signalPassFailure();
        return;
      }
      loweredFunctions.push_back(
          LoweredFunction{funcOp, *lowered, funcOp.getSymName().str()});
    }

    for (LoweredFunction &lowered : loweredFunctions)
      lowered.source.erase();
    for (LoweredFunction &lowered : loweredFunctions)
      lowered.lowered.setSymName(lowered.symbolName);

    eraseTargetMetadata(moduleOp);
    declareCallees(moduleOp, usedCallees);

    mlir::Operation *remainingWaferOp = nullptr;
    moduleOp.walk([&](mlir::Operation *op) {
      if (op == moduleOp || !isWaferDialectOp(op))
        return mlir::WalkResult::advance();
      remainingWaferOp = op;
      return mlir::WalkResult::interrupt();
    });
    if (remainingWaferOp) {
      remainingWaferOp->emitError()
          << "target_llvm_lowering_failure: Wafer op remains after lowering";
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace wafer
