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

  explicit LowerInstrToTargetLLVMPass(const TargetConversionRequest &request)
      : typedTargetProfile(request.targetProfile) {
    defaultDDRArenaArgumentIndex = request.defaultDDRArenaArgumentIndex;
    logicalRank = request.logicalRank;
    transportStatusArgumentIndex = request.transportStatusArgumentIndex;
    transportPreparedBeforeEntry = request.transportPreparedBeforeEntry;
    profileRecordArgumentIndex = request.profileRecordArgumentIndex;
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    std::optional<TargetProfileId> resolvedProfile = typedTargetProfile;
    if (!resolvedProfile) {
      if (targetProfile.getValue().empty()) {
        moduleOp.emitError()
            << "missing required target-profile for target LLVM conversion";
        signalPassFailure();
        return;
      }
      llvm::Expected<TargetProfileId> parsed =
          parseTargetProfileId(targetProfile.getValue());
      if (!parsed) {
        moduleOp.emitError()
            << "invalid target-profile for target LLVM conversion: "
            << llvm::toString(parsed.takeError());
        signalPassFailure();
        return;
      }
      resolvedProfile = *parsed;
    }
    if (mlir::failed(target_llvm_detail::preflightTargetAddresses(moduleOp)) ||
        mlir::failed(target_llvm_detail::preflightTargetFormats(
            moduleOp, *resolvedProfile))) {
      signalPassFailure();
      return;
    }

    mlir::OwningOpRef<mlir::ModuleOp> loweredModule = moduleOp.clone();
    uint64_t terminalOperationCount = 0;
    switch (detail::checkStaticTerminalOperationBudget(
        loweredModule->getOperation(), terminalOperationCount)) {
    case detail::StaticTerminalOperationBudgetStatus::WithinBudget:
      break;
    case detail::StaticTerminalOperationBudgetStatus::CountOverflow:
      moduleOp.emitError()
          << "static_terminal_budget_exceeded: final target module terminal "
             "operation count overflowed uint64_t";
      signalPassFailure();
      return;
    case detail::StaticTerminalOperationBudgetStatus::BudgetExceeded:
      moduleOp.emitError()
          << "static_terminal_budget_exceeded: final target module contains "
          << terminalOperationCount
          << " terminal instruction issues/completions; maximum is "
          << detail::kStaticTerminalOperationBudget;
      signalPassFailure();
      return;
    }
    if (mlir::failed(target_llvm_detail::lowerModuleInPlace(
            *loweredModule, *resolvedProfile, transportPreparedBeforeEntry,
            defaultDDRArenaArgumentIndex, logicalRank,
            transportStatusArgumentIndex, profileRecordArgumentIndex))) {
      signalPassFailure();
      return;
    }

    moduleOp->setAttrs((*loweredModule)->getAttrs());
    moduleOp.getBodyRegion().takeBody(loweredModule->getBodyRegion());
  }

  std::optional<TargetProfileId> typedTargetProfile;
  int64_t profileRecordArgumentIndex = -1;
};

} // namespace

std::unique_ptr<mlir::Pass>
createLowerInstrToTargetLLVMPass(const TargetConversionRequest &request) {
  return std::make_unique<LowerInstrToTargetLLVMPass>(request);
}

} // namespace wafer
