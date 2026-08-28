//===- FrontendVerification.cpp - Imported frontend IR verification ------===//

#include "Wafer/Transforms/StableHLO/FrontendVerification.h"

#include "Wafer/Frontend/Program/Program.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/raw_ostream.h"

namespace wafer {

#define GEN_PASS_DEF_VERIFYFRONTENDPROGRAMPASS
#include "Wafer/Transforms/WaferPasses.h.inc"

namespace {

struct VerifyFrontendProgramPass final
    : public impl::VerifyFrontendProgramPassBase<VerifyFrontendProgramPass> {
  void runOnOperation() override {
    std::string detail;
    llvm::raw_string_ostream diagnostics(detail);
    if (mlir::succeeded(
            frontend::verifyFrontendProgram(getOperation(), diagnostics)))
      return;
    diagnostics.flush();
    getOperation()->emitError("frontend verification failed: ") << detail;
    signalPassFailure();
  }
};

} // namespace

void buildFrontendVerificationPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createVerifyFrontendProgramPass());
}

} // namespace wafer
