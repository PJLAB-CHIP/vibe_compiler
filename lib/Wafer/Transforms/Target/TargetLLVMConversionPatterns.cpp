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

namespace {
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
    if (subviewOp.getOffsets().empty()) {
      mlir::FailureOr<int64_t> delta =
          getStaticViewDeltaBytes(subviewOp, sourceType, resultType);
      if (mlir::failed(delta))
        return mlir::failure();
      rewriter.replaceOp(subviewOp,
                         applyStaticAddressDelta(rewriter, subviewOp.getLoc(),
                                                 adaptor.getSource(), *delta));
      return mlir::success();
    }

    mlir::FailureOr<DynamicSubviewAddressPlan> plan =
        analyzeDynamicDDRSubviewAddressing(subviewOp);
    if (mlir::failed(plan))
      return mlir::failure();
    if (adaptor.getOffsets().size() != plan->dynamicByteStrides.size())
      return subviewOp.emitError()
             << "unsupported_target_address: converted dynamic DDR tensor "
                "subview offset count changed during lowering";

    mlir::Value address =
        applyStaticAddressDelta(rewriter, subviewOp.getLoc(),
                                adaptor.getSource(), plan->staticByteOffset);
    mlir::Type i64Type = rewriter.getI64Type();
    for (auto [offset, byteStride] :
         llvm::zip_equal(adaptor.getOffsets(), plan->dynamicByteStrides)) {
      auto offsetType = mlir::dyn_cast<mlir::IntegerType>(offset.getType());
      if (!offsetType || offsetType.getWidth() != 64)
        return subviewOp.emitError()
               << "unsupported_target_address: dynamic DDR tensor subview "
                  "offset must lower to i64";
      mlir::Value dynamicBytes = offset;
      if (byteStride != 1) {
        mlir::Value stride = rewriter.create<mlir::LLVM::ConstantOp>(
            subviewOp.getLoc(), i64Type, byteStride);
        dynamicBytes = rewriter.create<mlir::LLVM::MulOp>(subviewOp.getLoc(),
                                                          offset, stride);
      }
      address = rewriter.create<mlir::LLVM::AddOp>(subviewOp.getLoc(), address,
                                                   dynamicBytes);
    }
    rewriter.replaceOp(subviewOp, address);
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

struct TargetCollapseShapeOpLowering
    : public mlir::OpConversionPattern<mlir::memref::CollapseShapeOp> {
  using mlir::OpConversionPattern<
      mlir::memref::CollapseShapeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CollapseShapeOp collapseOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(collapseOp.getSrc().getType());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(collapseOp.getType());
    if (!sourceType || !resultType || !isWaferMemRefType(sourceType) ||
        !isWaferMemRefType(resultType))
      return collapseOp.emitError()
             << "unsupported_target_address: collapse_shape must preserve "
                "Wafer memory";

    MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
    MemoryAttr resultMemory = getWaferMemoryAttr(resultType);
    if (sourceMemory.getSpace() != resultMemory.getSpace() ||
        sourceMemory.getLayout() != MemLayout::Tensor ||
        resultMemory.getLayout() != MemLayout::Tensor ||
        sourceType.getElementType() != resultType.getElementType())
      return collapseOp.emitError()
             << "unsupported_target_address: collapse_shape requires "
                "matching Wafer tensor-layout memory and element types";

    mlir::FailureOr<int64_t> sourceElements =
        getStaticElementCount(collapseOp, sourceType, "collapse source");
    mlir::FailureOr<int64_t> resultElements =
        getStaticElementCount(collapseOp, resultType, "collapse result");
    std::optional<WaferPhysicalTensorInfo> sourceInfo =
        computeWaferPhysicalTensorInfo(sourceType);
    std::optional<WaferPhysicalTensorInfo> resultInfo =
        computeWaferPhysicalTensorInfo(resultType);
    if (mlir::failed(sourceElements) || mlir::failed(resultElements))
      return mlir::failure();
    if (*sourceElements != *resultElements || !sourceInfo || !resultInfo ||
        sourceInfo->compactBytes != resultInfo->compactBytes ||
        sourceInfo->physicalBytes != resultInfo->physicalBytes)
      return collapseOp.emitError()
             << "unsupported_target_address: collapse_shape must preserve "
                "element count and physical footprint";

    mlir::FailureOr<int64_t> delta =
        getStaticViewDeltaBytes(collapseOp, sourceType, resultType);
    if (mlir::failed(delta))
      return mlir::failure();
    rewriter.replaceOp(collapseOp,
                       applyStaticAddressDelta(rewriter, collapseOp.getLoc(),
                                               adaptor.getSrc(), *delta));
    return mlir::success();
  }
};

struct TargetExpandShapeOpLowering
    : public mlir::OpConversionPattern<mlir::memref::ExpandShapeOp> {
  using mlir::OpConversionPattern<
      mlir::memref::ExpandShapeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExpandShapeOp expandOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(expandOp.getSrc().getType());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(expandOp.getType());
    if (!sourceType || !resultType || !isWaferMemRefType(sourceType) ||
        !isWaferMemRefType(resultType))
      return expandOp.emitError()
             << "unsupported_target_address: expand_shape must preserve "
                "Wafer memory";

    MemoryAttr sourceMemory = getWaferMemoryAttr(sourceType);
    MemoryAttr resultMemory = getWaferMemoryAttr(resultType);
    if (sourceMemory.getSpace() != resultMemory.getSpace() ||
        sourceMemory.getLayout() != MemLayout::Tensor ||
        resultMemory.getLayout() != MemLayout::Tensor ||
        sourceType.getElementType() != resultType.getElementType())
      return expandOp.emitError()
             << "unsupported_target_address: expand_shape requires matching "
                "Wafer tensor-layout memory and element types";

    mlir::FailureOr<int64_t> sourceElements =
        getStaticElementCount(expandOp, sourceType, "expand source");
    mlir::FailureOr<int64_t> resultElements =
        getStaticElementCount(expandOp, resultType, "expand result");
    std::optional<WaferPhysicalTensorInfo> sourceInfo =
        computeWaferPhysicalTensorInfo(sourceType);
    std::optional<WaferPhysicalTensorInfo> resultInfo =
        computeWaferPhysicalTensorInfo(resultType);
    if (mlir::failed(sourceElements) || mlir::failed(resultElements))
      return mlir::failure();
    if (*sourceElements != *resultElements || !sourceInfo || !resultInfo ||
        sourceInfo->compactBytes != resultInfo->compactBytes ||
        sourceInfo->physicalBytes != resultInfo->physicalBytes)
      return expandOp.emitError()
             << "unsupported_target_address: expand_shape must preserve "
                "element count and physical footprint";

    mlir::FailureOr<int64_t> delta =
        getStaticViewDeltaBytes(expandOp, sourceType, resultType);
    if (mlir::failed(delta))
      return mlir::failure();
    rewriter.replaceOp(expandOp,
                       applyStaticAddressDelta(rewriter, expandOp.getLoc(),
                                               adaptor.getSrc(), *delta));
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
} // namespace

void populateTargetLLVMStructureConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    int64_t defaultDDRArenaArgumentIndex) {
  patterns.add<TargetFuncOpLowering, TargetCallOpLowering,
               TargetReturnOpLowering, TargetSubViewOpLowering,
               TargetReinterpretCastOpLowering, TargetCollapseShapeOpLowering,
               TargetExpandShapeOpLowering,
               TargetMemRefCastOpLowering, TargetDeallocOpLowering>(
      converter, &converter.getContext());
  patterns.add<TargetAllocOpLowering>(converter, &converter.getContext(),
                                      defaultDDRArenaArgumentIndex);
}

} // namespace wafer::target_llvm_detail
