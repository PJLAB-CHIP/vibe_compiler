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

mlir::LogicalResult FunctionLowering::lowerLocalFence(SyncLocalFenceOp op) {
  emitCall(
      op.getLoc(),
      getTargetCallDescriptor(TargetCallBuiltin::LocalFence, targetProfile),
      {});
  return mlir::success();
}

mlir::LogicalResult FunctionLowering::lowerNCCJoin(SyncNCCJoinOp op) {
  uint32_t participantMask = 0;
  for (int64_t worker : op.getParticipants()) {
    if (worker < 0 || worker >= static_cast<int64_t>(kNCCWorkerCount))
      return op.emitError() << "target_completion_failure: NCC participant "
                            << worker << " is outside the target worker domain";
    participantMask |= uint32_t{1} << static_cast<uint32_t>(worker);
  }
  if (participantMask == 0)
    return op.emitError()
           << "target_completion_failure: NCC participant set is empty";
  emitCall(op.getLoc(),
           getTargetCallDescriptor(TargetCallBuiltin::NCCJoin, targetProfile),
           {constantI32(op.getLoc(), participantMask)});
  return mlir::success();
}

} // namespace wafer::target_llvm_detail
