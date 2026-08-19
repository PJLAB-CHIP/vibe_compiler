//===- Target LLVM lowering implementation -------------------------------===//

#include "Target/LowerInstrToTargetLLVMInternal.h"
#include "Target/TargetCallIRAdapter.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/Core/TargetCall.h"
#include "Wafer/Target/Core/TargetFormat.h"
#include "Wafer/Target/Core/TargetMemory.h"
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

mlir::LogicalResult FunctionLowering::lowerPeripheral(InstrPeripheralOp op) {
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

  emitNCCCall(op.getLoc(), getTargetCallDescriptor(op.getKind()), args,
              op.getWorker());
  return mlir::success();
}

} // namespace wafer::target_llvm_detail
