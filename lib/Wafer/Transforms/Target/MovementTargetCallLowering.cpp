//===- Target LLVM lowering implementation -------------------------------===//

#include "Target/LowerInstrToTargetLLVMInternal.h"
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

mlir::LogicalResult FunctionLowering::lowerRDMA(InstrRDMAOp op) {
  llvm::SmallVector<mlir::Value, 12> args;
  mlir::FailureOr<AddressValue> source =
      resolveAddress(op, op.getSource(), "rdma source");
  mlir::FailureOr<AddressValue> dest =
      resolveAddress(op, op.getDest(), "rdma dest");
  if (mlir::failed(source) || mlir::failed(dest))
    return mlir::failure();
  source = addStaticOffset(
      op, *source, getOptionalIntegerAttrValue(op.getSrcOffsetAttr(), 0));
  dest = addStaticOffset(op, *dest,
                         getOptionalIntegerAttrValue(op.getDstOffsetAttr(), 0));
  mlir::FailureOr<int64_t> fmt =
      getDataFormatCode(op, op.getDest(), "rdma dest");
  if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(fmt))
    return mlir::failure();
  args.push_back(materializeAddress(op.getLoc(), *source));
  args.push_back(materializeAddress(op.getLoc(), *dest));
  appendI32(op.getLoc(), args, getIntegerAttrValue(op.getByteCountAttr()));
  appendI32(op.getLoc(), args, getIntegerAttrValue(op.getInnerBytesAttr()));
  appendArrayI32(op.getLoc(), args, op.getSrcStrides());
  appendArrayI32(op.getLoc(), args, op.getSrcIterations());
  appendI32(op.getLoc(), args, *fmt);
  emitNCCCall(op.getLoc(),
              getTargetCallDescriptor(TargetCallBuiltin::RDMA),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerWDMA(InstrWDMAOp op) {
  llvm::SmallVector<mlir::Value, 12> args;
  mlir::FailureOr<AddressValue> source =
      resolveAddress(op, op.getSource(), "wdma source");
  mlir::FailureOr<AddressValue> dest =
      resolveAddress(op, op.getDest(), "wdma dest");
  if (mlir::failed(source) || mlir::failed(dest))
    return mlir::failure();
  source = addStaticOffset(
      op, *source, getOptionalIntegerAttrValue(op.getSrcOffsetAttr(), 0));
  dest = addStaticOffset(op, *dest,
                         getOptionalIntegerAttrValue(op.getDstOffsetAttr(), 0));
  mlir::FailureOr<int64_t> fmt =
      getDataFormatCode(op, op.getSource(), "wdma source");
  if (mlir::failed(source) || mlir::failed(dest) || mlir::failed(fmt))
    return mlir::failure();
  args.push_back(materializeAddress(op.getLoc(), *source));
  args.push_back(materializeAddress(op.getLoc(), *dest));
  appendI32(op.getLoc(), args, getIntegerAttrValue(op.getByteCountAttr()));
  appendI32(op.getLoc(), args, getIntegerAttrValue(op.getInnerBytesAttr()));
  appendArrayI32(op.getLoc(), args, op.getDstStrides());
  appendArrayI32(op.getLoc(), args, op.getDstIterations());
  appendI32(op.getLoc(), args, *fmt);
  emitNCCCall(op.getLoc(),
              getTargetCallDescriptor(TargetCallBuiltin::WDMA),
              args, op.getWorker());
  return mlir::success();
}

mlir::LogicalResult
FunctionLowering::lowerGatherScatter(InstrGatherScatterOp op) {
  mlir::FailureOr<AddressValue> source =
      resolveAddress(op, op.getSource(), "gather_scatter source");
  mlir::FailureOr<AddressValue> dest =
      resolveAddress(op, op.getDest(), "gather_scatter dest");
  if (mlir::failed(source) || mlir::failed(dest))
    return mlir::failure();
  source = addStaticOffset(
      op, *source, getOptionalIntegerAttrValue(op.getSrcOffsetAttr(), 0));
  dest = addStaticOffset(op, *dest,
                         getOptionalIntegerAttrValue(op.getDstOffsetAttr(), 0));
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
  emitNCCCall(
      op.getLoc(),
      getTargetCallDescriptor(TargetCallBuiltin::GatherScatter),
      args, op.getWorker());
  return mlir::success();
}

bool FunctionLowering::isTransformLikeTDMA(InstrDataMoveKind kind) {
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

mlir::LogicalResult
FunctionLowering::lowerTDMADataMove(InstrTDMADataMoveOp op) {
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
    emitNCCCall(
        op.getLoc(),
        getTargetCallDescriptor(TargetCallBuiltin::TDMAPad),
        args, op.getWorker());
    return mlir::success();
  }
  if (op.getKind() == InstrDataMoveKind::Img2Col) {
    emitNCCCall(
        op.getLoc(),
        getTargetCallDescriptor(TargetCallBuiltin::TDMAImg2Col),
        args, op.getWorker());
    return mlir::success();
  }
  llvm_unreachable("transform-like TDMA kinds handled above");
}

} // namespace wafer::target_llvm_detail
