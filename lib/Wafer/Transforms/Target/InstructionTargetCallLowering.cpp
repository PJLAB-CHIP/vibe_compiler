//===- Target LLVM lowering implementation -------------------------------===//

#include "Target/LowerInstrToTargetLLVMInternal.h"
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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace wafer::target_llvm_detail {

mlir::LogicalResult FunctionLowering::lowerInstruction(mlir::Operation *op) {
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
      .Case<InstrConvertOp>([&](auto typedOp) { return lowerConvert(typedOp); })
      .Case<InstrGemmOp>([&](auto typedOp) { return lowerGemm(typedOp); })
      .Case<InstrConvOp>([&](auto typedOp) { return lowerConv(typedOp); })
      .Case<InstrPoolOp>([&](auto typedOp) { return lowerPool(typedOp); })
      .Case<InstrUnpoolOp>([&](auto typedOp) { return lowerUnpool(typedOp); })
      .Case<InstrTDMADataMoveOp>(
          [&](auto typedOp) { return lowerTDMADataMove(typedOp); })
      .Case<InstrPeripheralOp>(
          [&](auto typedOp) { return lowerPeripheral(typedOp); })
      .Case<SyncNCCJoinOp>([&](auto typedOp) { return lowerNCCJoin(typedOp); })
      .Default([&](mlir::Operation *unknown) {
        return unknown->emitError()
               << "unsupported_target_instr: Wafer instruction op is not "
                  "handled by target LLVM lowering";
      });
}

namespace {
static mlir::LogicalResult
lowerTargetInstruction(mlir::Operation *op, mlir::ValueRange operands,
                       mlir::ConversionPatternRewriter &rewriter,
                       const mlir::LLVMTypeConverter *converter,
                       const DirectDTEEndpointDomain *dteDomain) {
  bool isDTE = mlir::isa<InstrDTESendOp, InstrDTERecvOp, InstrDTEWaitOp>(op);
  if (isDTE && !dteDomain)
    return op->emitError()
           << "unsupported_target_transport: DTE instruction requires an "
              "accepted endpoint domain and launch status ABI";
  if (auto peripheral = mlir::dyn_cast<InstrPeripheralOp>(op);
      peripheral && peripheral.getKind() == InstrPeripheralKind::Factorize)
    return op->emitError()
           << "unsupported_target_operation: peripheral factorize lacks a "
              "proven production target semantic implementation";
  if (operands.size() != op->getNumOperands())
    return op->emitError()
           << "target_llvm_lowering_failure: converted operand count "
              "mismatch";

  for (mlir::Type resultType : op->getResultTypes())
    if (!mlir::isa<mlir::async::TokenType>(resultType) ||
        !converter->convertType(resultType))
      return op->emitError()
             << "unsupported_target_instr: instruction result type is not "
                "a target async token";

  FunctionLowering lowering(op->getContext(), rewriter);
  for (auto [source, converted] : llvm::zip_equal(op->getOperands(), operands))
    lowering.convertedValues[source] = converted;
  rewriter.setInsertionPoint(op);

  if (auto send = mlir::dyn_cast<InstrDTESendOp>(op)) {
    mlir::FailureOr<mlir::Value> event =
        lowering.lowerDTESend(send, *dteDomain);
    if (mlir::failed(event))
      return mlir::failure();
    rewriter.replaceOp(op, *event);
    return mlir::success();
  }
  if (auto recv = mlir::dyn_cast<InstrDTERecvOp>(op)) {
    mlir::FailureOr<mlir::Value> event =
        lowering.lowerDTERecv(recv, *dteDomain);
    if (mlir::failed(event))
      return mlir::failure();
    rewriter.replaceOp(op, *event);
    return mlir::success();
  }
  if (auto wait = mlir::dyn_cast<InstrDTEWaitOp>(op)) {
    llvm::SmallVector<mlir::Value, 4> events(operands.begin(), operands.end());
    if (mlir::failed(lowering.lowerDTEWait(wait, events)))
      return mlir::failure();
    rewriter.eraseOp(op);
    return mlir::success();
  }

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

struct TargetInstructionOpLowering
    : public mlir::OpInterfaceConversionPattern<WaferInstructionOpInterface> {
  TargetInstructionOpLowering(mlir::LLVMTypeConverter &converter,
                              const DirectDTEEndpointDomain *dteDomain)
      : mlir::OpInterfaceConversionPattern<WaferInstructionOpInterface>(
            converter, &converter.getContext()),
        dteDomain(dteDomain) {}

  mlir::LogicalResult
  matchAndRewrite(WaferInstructionOpInterface op,
                  llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return lowerTargetInstruction(op.getOperation(), operands, rewriter,
                                  getTypeConverter<mlir::LLVMTypeConverter>(),
                                  dteDomain);
  }

  const DirectDTEEndpointDomain *dteDomain;
};

struct TargetNCCJoinOpLowering
    : public mlir::OpConversionPattern<SyncNCCJoinOp> {
  TargetNCCJoinOpLowering(mlir::LLVMTypeConverter &converter,
                          const DirectDTEEndpointDomain *dteDomain)
      : mlir::OpConversionPattern<SyncNCCJoinOp>(converter,
                                                 &converter.getContext()),
        dteDomain(dteDomain) {}

  mlir::LogicalResult
  matchAndRewrite(SyncNCCJoinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return lowerTargetInstruction(
        op.getOperation(), adaptor.getOperands(), rewriter,
        getTypeConverter<mlir::LLVMTypeConverter>(), dteDomain);
  }

  const DirectDTEEndpointDomain *dteDomain;
};
} // namespace

void populateTargetInstructionConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    const DirectDTEEndpointDomain *dteDomain) {
  patterns.add<TargetInstructionOpLowering, TargetNCCJoinOpLowering>(converter,
                                                                     dteDomain);
}

} // namespace wafer::target_llvm_detail
