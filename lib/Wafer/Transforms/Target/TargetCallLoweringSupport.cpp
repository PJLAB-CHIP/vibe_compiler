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

FunctionLowering::FunctionLowering(mlir::MLIRContext *context,
                                   mlir::OpBuilder &builder,
                                   llvm::StringMap<CalleeSignature> &used)
    : builder(builder), context(context),
      i64Type(mlir::IntegerType::get(context, 64)),
      i32Type(mlir::IntegerType::get(context, 32)),
      voidType(mlir::LLVM::LLVMVoidType::get(context)), usedCallees(used) {}

mlir::Value FunctionLowering::constantI64(mlir::Location loc, int64_t value) {
  return builder.create<mlir::LLVM::ConstantOp>(loc, i64Type, value);
}

mlir::Value FunctionLowering::constantI32(mlir::Location loc, int64_t value) {
  return builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, value);
}

void FunctionLowering::appendI32(mlir::Location loc,
                                 llvm::SmallVectorImpl<mlir::Value> &out,
                                 int64_t value) {
  out.push_back(constantI32(loc, value));
}

void FunctionLowering::appendArrayI32(mlir::Location loc,
                                      llvm::SmallVectorImpl<mlir::Value> &out,
                                      llvm::ArrayRef<int64_t> values) {
  for (int64_t value : values)
    appendI32(loc, out, value);
}

mlir::FailureOr<llvm::SmallVector<int64_t, 4>>
FunctionLowering::getNHWCShape(mlir::Operation *op, mlir::MemRefType type,
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
FunctionLowering::addStaticOffset(mlir::Operation *op, AddressValue address,
                                  int64_t offset) {
  int64_t combined = 0;
  if (!checkedAdd(address.staticOffset, offset, combined))
    return op->emitError()
           << "target_address_overflow: byte offset overflows int64";
  address.staticOffset = combined;
  return address;
}

mlir::Value FunctionLowering::materializeAddress(mlir::Location loc,
                                                 AddressValue address) {
  if (!address.dynamicBase)
    return constantI64(loc, address.staticOffset);
  if (address.staticOffset == 0)
    return address.dynamicBase;
  mlir::Value offset = constantI64(loc, address.staticOffset);
  return builder.create<mlir::LLVM::AddOp>(loc, address.dynamicBase, offset);
}

mlir::FailureOr<AddressValue>
FunctionLowering::resolveAddress(mlir::Operation *op, mlir::Value value,
                                 llvm::StringRef role) {
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
    auto offset = alloc->getAttrOfType<SPMOffsetAttr>(kWaferSPMOffsetAttrName);
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

mlir::FailureOr<mlir::Value>
FunctionLowering::materializeAddress(mlir::Operation *op, mlir::Value value,
                                     llvm::StringRef role) {
  mlir::FailureOr<AddressValue> address = resolveAddress(op, value, role);
  if (mlir::failed(address))
    return mlir::failure();
  return materializeAddress(op->getLoc(), *address);
}

void FunctionLowering::verifyCallSignature(
    const TargetCallDescriptor &descriptor, mlir::ValueRange args,
    TargetCallResultType result) const {
  if (descriptor.result != result || descriptor.arguments.size() != args.size())
    llvm_unreachable("target call does not match its registered signature");
  for (auto [argument, scalar] : llvm::zip_equal(args, descriptor.arguments)) {
    unsigned width = scalar == TargetCallScalarType::I64 ? 64 : 32;
    if (!argument.getType().isInteger(width))
      llvm_unreachable("target call argument width is not registered");
  }
}

void FunctionLowering::emitCall(mlir::Location loc,
                                const TargetCallDescriptor &descriptor,
                                mlir::ValueRange args) {
  verifyCallSignature(descriptor, args, TargetCallResultType::Void);
  llvm::SmallVector<mlir::Type, 16> argTypes;
  for (mlir::Value arg : args)
    argTypes.push_back(arg.getType());
  mlir::LLVM::LLVMFunctionType functionType =
      mlir::LLVM::LLVMFunctionType::get(voidType, argTypes,
                                        /*isVarArg=*/false);
  auto [it, inserted] =
      usedCallees.try_emplace(descriptor.symbol, CalleeSignature{functionType});
  if (!inserted && it->second.type != functionType)
    llvm_unreachable(
        "same target CRT symbol emitted with incompatible signature");
  builder.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange(),
      mlir::FlatSymbolRefAttr::get(context, descriptor.symbol), args);
}

void FunctionLowering::emitNCCCall(mlir::Location loc,
                                   const TargetCallDescriptor &descriptor,
                                   mlir::ValueRange args, NCCWorker worker) {
  if (!descriptor.issueDomain)
    llvm_unreachable("NCC target call must register an issue domain");
  if (!descriptor.issueDomain->nccWorkerArgument ||
      *descriptor.issueDomain->nccWorkerArgument != args.size())
    llvm_unreachable("worker-aware target call must register one trailing "
                     "worker argument");
  llvm::SmallVector<mlir::Value, 32> workerArguments(args.begin(), args.end());
  workerArguments.push_back(constantI32(loc, static_cast<uint32_t>(worker)));
  emitCall(loc, descriptor, workerArguments);
}

mlir::Value
FunctionLowering::emitI64Call(mlir::Location loc,
                              const TargetCallDescriptor &descriptor,
                              mlir::ValueRange args) {
  verifyCallSignature(descriptor, args, TargetCallResultType::I64);
  llvm::SmallVector<mlir::Type, 16> argTypes;
  for (mlir::Value arg : args)
    argTypes.push_back(arg.getType());
  mlir::LLVM::LLVMFunctionType functionType =
      mlir::LLVM::LLVMFunctionType::get(i64Type, argTypes,
                                        /*isVarArg=*/false);
  auto [it, inserted] =
      usedCallees.try_emplace(descriptor.symbol, CalleeSignature{functionType});
  if (!inserted && it->second.type != functionType)
    llvm_unreachable(
        "same target CRT symbol emitted with incompatible signature");
  auto call = builder.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{i64Type},
      mlir::FlatSymbolRefAttr::get(context, descriptor.symbol), args);
  return call.getResult();
}

} // namespace wafer::target_llvm_detail
