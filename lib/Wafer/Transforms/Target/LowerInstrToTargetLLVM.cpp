//===- LowerInstrToTargetLLVM.cpp - Lower instr IR to target LLVM ---------===//

#include "Wafer/Transforms/Passes.h"

#include "Target/LowerInstrToTargetLLVMInternal.h"
#include "Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.h"
#include "Wafer/Transforms/TargetConversion.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace wafer {
#define GEN_PASS_DEF_LOWERINSTRTOTARGETLLVMPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

struct LowerInstrToTargetLLVMPass
    : public impl::LowerInstrToTargetLLVMPassBase<LowerInstrToTargetLLVMPass> {
  using impl::LowerInstrToTargetLLVMPassBase<
      LowerInstrToTargetLLVMPass>::LowerInstrToTargetLLVMPassBase;

  explicit LowerInstrToTargetLLVMPass(const TargetConversionRequest &request) {
    defaultDDRArenaArgumentIndex = request.defaultDDRArenaArgumentIndex;
    cardId = request.cardId;
    tileId = request.tileId;
    transportStatusArgumentIndex = request.transportStatusArgumentIndex;
    transportPreparedBeforeEntry = request.transportPreparedBeforeEntry;
    profileRecordArgumentIndex = request.profileRecordArgumentIndex;
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    if (mlir::failed(
            target_llvm_detail::verifyTargetSubviewAddresses(moduleOp)) ||
        mlir::failed(
            target_llvm_detail::verifyTargetInstructionFormats(moduleOp))) {
      signalPassFailure();
      return;
    }

    uint64_t executableOperationCount = 0;
    switch (detail::countStaticExecutableOperations(moduleOp.getOperation(),
                                                    executableOperationCount)) {
    case detail::StaticExecutableOperationCountStatus::Counted:
      break;
    case detail::StaticExecutableOperationCountStatus::CountOverflow:
      moduleOp.emitError()
          << "static_executable_operation_count_overflow: final target "
             "module operation count overflowed uint64_t";
      signalPassFailure();
      return;
    }
    if (mlir::failed(target_llvm_detail::lowerModuleInPlace(
            moduleOp, transportPreparedBeforeEntry,
            defaultDDRArenaArgumentIndex, cardId, tileId,
            transportStatusArgumentIndex, profileRecordArgumentIndex))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
createLowerInstrToTargetLLVMPass(const TargetConversionRequest &request) {
  return std::make_unique<LowerInstrToTargetLLVMPass>(request);
}

} // namespace wafer
