//===- FrontendVerification.h - Frontend verification pipeline -*- C++ -*-===//

#pragma once

namespace mlir {
class OpPassManager;
}

namespace wafer {

void buildFrontendVerificationPipeline(mlir::OpPassManager &pm);

} // namespace wafer
