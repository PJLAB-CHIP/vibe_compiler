//===- Target LLVM lowering implementation -------------------------------===//

#include "Target/LowerInstrToTargetLLVMInternal.h"
#include "Target/TargetCallIRAdapter.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetFormat.h"
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

namespace {

mlir::FailureOr<int64_t> getFillElementCount(InstrFillOp op) {
  auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
  if (op.getFillDomain().value_or(FillDomain::LogicalValid) ==
      FillDomain::LogicalValid)
    return getStaticElementCount(op, destType, "fill dest");

  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(destType);
  if (!info || info->physicalBytes <= 0 || info->physicalElements <= 0)
    return op.emitError()
           << "unsupported_target_geometry: physical_footprint fill requires "
              "a static positive physical destination footprint";
  if (!info->bitPackedElement)
    return info->physicalElements;
  if (static_cast<uint64_t>(info->physicalBytes) >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / UINT64_C(8))
    return op.emitError()
           << "target_range_overflow: bitpacked physical fill element count "
              "overflows int64";
  return info->physicalBytes * INT64_C(8);
}

} // namespace

mlir::LogicalResult FunctionLowering::lowerFill(InstrFillOp op) {
  llvm::SmallVector<mlir::Value, 5> args;
  mlir::FailureOr<mlir::Value> dest =
      materializeAddress(op, op.getDest(), "fill dest");
  mlir::FailureOr<int64_t> elements = getFillElementCount(op);
  mlir::FailureOr<int64_t> scalar = getConstantScalarValue(op, op.getValue());
  mlir::FailureOr<int64_t> fmt =
      getDataFormatCode(op, op.getDest(), "fill dest");
  if (mlir::failed(dest) || mlir::failed(elements) || mlir::failed(scalar) ||
      mlir::failed(fmt))
    return mlir::failure();
  args.push_back(*dest);
  appendI32(op.getLoc(), args, *scalar);
  appendI32(op.getLoc(), args, *elements);
  appendI32(op.getLoc(), args, *fmt);
  emitNCCCall(op.getLoc(),
              getTargetCallDescriptor(TargetCallBuiltin::Memset),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerElementwise(InstrElementwiseOp op) {
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
      getPhysicalTraversalElementCount(op, destType, "elementwise dest");
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
  emitNCCCall(op.getLoc(), getTargetCallDescriptor(op.getKind()),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerBit2Fp(InstrBit2FpOp op) {
  llvm::SmallVector<mlir::Value, 5> args;
  mlir::FailureOr<mlir::Value> source =
      materializeAddress(op, op.getSource(), "bit2fp source");
  mlir::FailureOr<mlir::Value> dest =
      materializeAddress(op, op.getDest(), "bit2fp dest");
  auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
  mlir::FailureOr<int64_t> elements =
      getPhysicalTraversalElementCount(op, destType, "bit2fp dest");
  mlir::FailureOr<int64_t> fmt =
      getDataFormatCode(op, op.getDest(), "bit2fp dest");
  if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(elements) ||
      mlir::failed(fmt))
    return mlir::failure();
  args.push_back(*source);
  args.push_back(*dest);
  appendI32(op.getLoc(), args, *elements);
  appendI32(op.getLoc(), args, *fmt);
  emitNCCCall(op.getLoc(),
              getTargetCallDescriptor(TargetCallBuiltin::Bit2FP),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerMaskMove(InstrMaskMoveOp op) {
  llvm::SmallVector<mlir::Value, 6> args;
  mlir::FailureOr<mlir::Value> source =
      materializeAddress(op, op.getSource(), "mask_move source");
  mlir::FailureOr<int64_t> mask =
      getStaticUInt32SPMAddress(op, op.getMask(), "mask");
  mlir::FailureOr<mlir::Value> dest =
      materializeAddress(op, op.getDest(), "mask_move dest");
  auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
  mlir::FailureOr<int64_t> elements =
      getPhysicalTraversalElementCount(op, destType, "mask_move dest");
  mlir::FailureOr<int64_t> fmt =
      getDataFormatCode(op, op.getDest(), "mask_move dest");
  if (mlir::failed(source) || mlir::failed(mask) || mlir::failed(dest) ||
      mlir::failed(elements) || mlir::failed(fmt))
    return mlir::failure();
  args.push_back(*source);
  args.push_back(constantI32(op.getLoc(), *mask));
  args.push_back(*dest);
  appendI32(op.getLoc(), args, *elements);
  appendI32(op.getLoc(), args, *fmt);
  emitNCCCall(
      op.getLoc(),
      getTargetCallDescriptor(TargetCallBuiltin::MaskMove), args,
      op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerReduce(InstrReduceOp op) {
  llvm::SmallVector<mlir::Value, 10> args;
  mlir::FailureOr<mlir::Value> input =
      materializeAddress(op, op.getInput(), "reduce input");
  mlir::FailureOr<mlir::Value> dest =
      materializeAddress(op, op.getDest(), "reduce dest");
  auto inputType = mlir::cast<mlir::MemRefType>(op.getInput().getType());
  mlir::FailureOr<int64_t> fmt =
      getDataFormatCode(op, op.getInput(), "reduce input");
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
  emitNCCCall(op.getLoc(), getTargetCallDescriptor(op.getKind()),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerConvert(InstrConvertOp op) {
  llvm::SmallVector<mlir::Value, 8> args;
  mlir::FailureOr<mlir::Value> source =
      materializeAddress(op, op.getSource(), "convert source");
  mlir::FailureOr<mlir::Value> dest =
      materializeAddress(op, op.getDest(), "convert dest");
  auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
  mlir::FailureOr<int64_t> elements =
      getPhysicalTraversalElementCount(op, destType, "convert dest");
  if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(elements))
    return mlir::failure();
  args.push_back(*source);
  args.push_back(*dest);
  appendI32(op.getLoc(), args, *elements);
  appendI32(op.getLoc(), args,
            getOptionalIntegerAttrValue(op.getZeroPointAttr(), -1));
  appendI32(op.getLoc(), args,
            getOptionalIntegerAttrValue(op.getRoundingModeAttr(), -1));
  emitNCCCall(op.getLoc(), getTargetCallDescriptor(op.getKind()),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerGemm(InstrGemmOp op) {
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
  if (op.getLhsOrientationAttr()) {
    appendI32(op.getLoc(), args, static_cast<int64_t>(*op.getLhsOrientation()));
    appendI32(op.getLoc(), args, static_cast<int64_t>(*op.getRhsOrientation()));
    emitNCCCall(op.getLoc(),
                getTargetCallDescriptor(TargetCallBuiltin::GemmOriented),
                args, op.getWorker());
  } else {
    emitNCCCall(op.getLoc(),
                getTargetCallDescriptor(TargetCallBuiltin::Gemm),
                args, op.getWorker());
  }
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerConv(InstrConvOp op) {
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
  emitNCCCall(op.getLoc(), getTargetCallDescriptor(op.getKind()),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerPool(InstrPoolOp op) {
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
  emitNCCCall(op.getLoc(), getTargetCallDescriptor(op.getKind()),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerUnpool(InstrUnpoolOp op) {
  llvm::SmallVector<mlir::Value, 20> args;
  mlir::FailureOr<mlir::Value> input =
      materializeAddress(op, op.getInput(), "unpool input");
  mlir::FailureOr<mlir::Value> dest =
      materializeAddress(op, op.getDest(), "unpool dest");
  mlir::FailureOr<int64_t> fmt =
      getDataFormatCode(op, op.getInput(), "unpool input");
  if (mlir::failed(input) || mlir::failed(dest) || mlir::failed(fmt))
    return mlir::failure();
  int64_t indexAddress = 0;
  if (mlir::Value index = op.getIndex()) {
    mlir::FailureOr<int64_t> resolvedIndex =
        getStaticUInt32SPMAddress(op, index, "unpool index");
    if (mlir::failed(resolvedIndex))
      return mlir::failure();
    indexAddress = *resolvedIndex;
  }
  args.push_back(*input);
  args.push_back(*dest);
  appendI32(op.getLoc(), args, static_cast<int64_t>(op.getKind()));
  appendI32(op.getLoc(), args, indexAddress);
  appendArrayI32(op.getLoc(), args, op.getSourceShape());
  appendArrayI32(op.getLoc(), args, op.getDestShape());
  appendArrayI32(op.getLoc(), args, op.getKernelStrides());
  appendI32(op.getLoc(), args, *fmt);
  emitNCCCall(op.getLoc(), getTargetCallDescriptor(op.getKind()),
              args, op.getWorker());
  return mlir::success();
}

} // namespace wafer::target_llvm_detail
