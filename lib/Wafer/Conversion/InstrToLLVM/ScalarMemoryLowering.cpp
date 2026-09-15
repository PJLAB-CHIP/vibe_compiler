//===- ScalarMemoryLowering.cpp - Bounded mapped scalar accesses --------===//

#include "Wafer/Conversion/InstrToLLVM/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Analysis/Instr/StaticIndexRange.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Target/TargetCall.h"
#include "Wafer/Target/TargetMemory.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

namespace wafer::target_llvm_detail {
namespace {
struct ScalarAddressing {
  unsigned byteWidth;
  llvm::SmallVector<int64_t> byteStrides;
  int64_t byteSpan = 0;
};

mlir::FailureOr<ScalarAddressing> analyzeScalarAddressing(
    mlir::Operation *op, mlir::MemRefType type) {
  auto memory = getWaferMemoryAttr(type);
  auto element = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  bool supportedInteger =
      element && element.isSignless() &&
      (element.getWidth() == 32 || element.getWidth() == 64);
  if (!memory || memory.getLayout() != MemLayout::Tensor ||
      !type.hasStaticShape() ||
      (!supportedInteger && !type.getElementType().isF32()))
    return op->emitError("unsupported_target_scalar_access: requires static "
                         "Tensor-layout i32/i64/f32 memory");
  ScalarAddressing result{type.getElementType().getIntOrFloatBitWidth() / 8,
                          {}};
  int64_t ignoredOffset;
  if (mlir::failed(mlir::getStridesAndOffset(type, result.byteStrides,
                                              ignoredOffset)))
    return op->emitError("unsupported_target_scalar_access: requires static strides");
  int64_t lastByte = result.byteWidth - 1;
  for (auto [extent, stride] : llvm::zip_equal(type.getShape(), result.byteStrides)) {
    int64_t bytes;
    if (stride < 0 || extent <= 0 ||
        llvm::MulOverflow(stride, int64_t(result.byteWidth), stride) ||
        llvm::MulOverflow(extent - 1, stride, bytes) ||
        llvm::AddOverflow(lastByte, bytes, lastByte))
      return op->emitError("unsupported_target_scalar_access: byte range overflows");
  }
  if (llvm::AddOverflow(lastByte, int64_t{1}, result.byteSpan))
    return op->emitError("unsupported_target_scalar_access: byte span overflows");
  if (memory.getSpace() == MemorySpace::DDR &&
      result.byteSpan > std::numeric_limits<int32_t>::max())
    return op->emitError("unsupported_target_scalar_access: DDR mapping exceeds int32 range");
  return result;
}

struct OriginalInput {
  mlir::Value source;
  mlir::BlockArgument address;
};

static std::optional<OriginalInput> findOriginalInput(mlir::Value value, mlir::Value adapted,
                                      mlir::ConversionPatternRewriter &rewriter) {
  while (auto view = mlir::dyn_cast_or_null<mlir::ViewLikeOpInterface>(value.getDefiningOp()))
    value = view.getViewSource();
  mlir::Value mapped = rewriter.getRemappedValue(value);
  if (mlir::isa<mlir::BlockArgument>(adapted))
    mapped = adapted;
  while (auto cast = mapped ? mapped.getDefiningOp<mlir::UnrealizedConversionCastOp>()
                            : mlir::UnrealizedConversionCastOp{}) {
    if (cast.getInputs().size() != 1)
      break;
    mapped = cast.getInputs().front();
  }
  if (!mapped)
    return {};
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(mapped);
  if (!argument)
    return {};
  auto function = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(argument.getOwner()->getParentOp());
  if (!function || function.getBody().empty() ||
      argument.getOwner() != &function.getBody().front() ||
      !function.getArgAttrOfType<ProgramArgumentAttr>(argument.getArgNumber(),
                                                    kWaferProgramArgumentAttrName))
    return {};
  return OriginalInput{value, argument};
}

template <typename OpTy>
struct ScalarAccessLowering : mlir::OpConversionPattern<OpTy> {
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    auto addressing = analyzeScalarAddressing(op, op.getMemRefType());
    if (mlir::failed(addressing))
      return mlir::failure();
    auto i64 = rewriter.getI64Type();
    if (adaptor.getMemref().getType() != i64 ||
        llvm::any_of(adaptor.getIndices(),
                     [&](mlir::Value index) { return index.getType() != i64; }))
      return rewriter.notifyMatchFailure(op, "scalar address must lower to i64");
    bool ddr = getWaferMemoryAttr(op.getMemRefType()).getSpace() == MemorySpace::DDR;
    if constexpr (std::is_same_v<OpTy, mlir::memref::StoreOp>)
      if (ddr)
        return rewriter.notifyMatchFailure(op, "DDR scalar stores require a publication contract");
    mlir::Value mappingSource = adaptor.getMemref();
    int64_t mappingBytes = addressing->byteSpan;
    auto original = ddr ? findOriginalInput(op.getMemref(), adaptor.getMemref(), rewriter)
                        : std::optional<OriginalInput>{};
    if (original) {
      auto originalAddressing = analyzeScalarAddressing(op, mlir::cast<mlir::MemRefType>(original->source.getType()));
      if (mlir::failed(originalAddressing))
        return mlir::failure();
      mappingSource = original->address;
      mappingBytes = originalAddressing->byteSpan;
    }
    mlir::Value address = rewriter.create<mlir::LLVM::SubOp>(
        op.getLoc(), adaptor.getMemref(), mappingSource);
    auto loc = op.getLoc();
    for (auto [index, stride] : llvm::zip_equal(adaptor.getIndices(),
                                               addressing->byteStrides)) {
      mlir::Value factor = rewriter.create<mlir::LLVM::ConstantOp>(loc, i64, stride);
      mlir::Value offset = rewriter.create<mlir::LLVM::MulOp>(loc, index, factor);
      address = rewriter.create<mlir::LLVM::AddOp>(loc, address, offset);
    }
    auto pointerType = mlir::LLVM::LLVMPointerType::get(op.getContext());
    auto &mappingDescriptor = getTargetCallDescriptor(
        ddr ? TargetCallBuiltin::DDRReadMapping : TargetCallBuiltin::SPMMapping);
    mlir::Value mappedPointer;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      if (original) {
        auto argument = mlir::cast<mlir::BlockArgument>(mappingSource);
        for (auto call : argument.getOwner()->getOps<mlir::LLVM::CallOp>())
          if (call.getCallee() == mappingDescriptor.symbol &&
              call.getNumOperands() == 2 && call.getOperand(0) == mappingSource) {
            mappedPointer = call.getResult();
            break;
          }
        rewriter.setInsertionPointToStart(argument.getOwner());
      }
      if (!mappedPointer) {
        llvm::SmallVector<mlir::Value> arguments{mappingSource};
        if (ddr)
          arguments.push_back(rewriter.create<mlir::LLVM::ConstantOp>(
              loc, rewriter.getI32Type(), mappingBytes));
        mappedPointer = rewriter.create<mlir::LLVM::CallOp>(
            loc, mlir::TypeRange{pointerType}, mappingDescriptor.symbol,
            arguments).getResult();
      }
    }
    auto pointer = rewriter.create<mlir::LLVM::GEPOp>(
        loc, pointerType, rewriter.getI8Type(), mappedPointer,
        mlir::ValueRange{address});
    if constexpr (std::is_same_v<OpTy, mlir::memref::LoadOp>) {
      mlir::Type bitsType = rewriter.getIntegerType(addressing->byteWidth * 8);
      mlir::Value loaded = rewriter.create<mlir::LLVM::LoadOp>(
          loc, bitsType, pointer, addressing->byteWidth, /*isVolatile=*/true);
      if (op.getType() != bitsType)
        loaded =
            rewriter.create<mlir::LLVM::BitcastOp>(loc, op.getType(), loaded);
      rewriter.replaceOp(op, loaded);
    } else {
      mlir::Value stored = adaptor.getValue();
      mlir::Type bitsType = rewriter.getIntegerType(addressing->byteWidth * 8);
      if (stored.getType() != bitsType)
        stored = rewriter.create<mlir::LLVM::BitcastOp>(loc, bitsType, stored);
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(
          op, stored, pointer, addressing->byteWidth, /*isVolatile=*/true);
    }
    return mlir::success();
  }
};
struct KcoreReleaseLowering : mlir::OpConversionPattern<mlir::LLVM::FenceOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(mlir::LLVM::FenceOp op, OpAdaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    if (op.getSyncscope() != kTargetKcoreReleaseScope ||
        op.getOrdering() != mlir::LLVM::AtomicOrdering::release)
      return rewriter.notifyMatchFailure(op, "not a Kcore device release");
    rewriter.replaceOpWithNewOp<mlir::LLVM::InlineAsmOp>(
        op, mlir::Type{}, mlir::ValueRange{}, kTargetKcoreReleaseAssembly,
        "~{memory}", true, false, mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
    return mlir::success();
  }
};
} // namespace

mlir::LogicalResult verifyTargetScalarMemoryAccesses(mlir::ModuleOp module) {
  auto result = module.walk([&](mlir::Operation *op) {
    mlir::Value buffer;
    mlir::ValueRange indices;
    if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      buffer = load.getMemref();
      indices = load.getIndices();
    } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
      buffer = store.getMemref();
      indices = store.getIndices();
    } else {
      return mlir::WalkResult::advance();
    }
    auto type = mlir::cast<mlir::MemRefType>(buffer.getType());
    if (mlir::failed(analyzeScalarAddressing(op, type)))
      return mlir::WalkResult::interrupt();
    for (auto [dimension, index] : llvm::enumerate(indices)) {
      auto range = memory_planning::detail::evaluateNonNegativeStaticIndexRange(index, op);
      if (!range.succeeded() || (!range.range.empty &&
                                 range.range.max >= type.getDimSize(dimension))) {
        op->emitError("unsupported_target_scalar_access: index is unbounded or out of bounds at dimension ")
            << dimension;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  return mlir::success(!result.wasInterrupted());
}

void populateTargetScalarMemoryConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<ScalarAccessLowering<mlir::memref::LoadOp>,
               ScalarAccessLowering<mlir::memref::StoreOp>, KcoreReleaseLowering>(
      converter, &converter.getContext());
}
} // namespace wafer::target_llvm_detail
