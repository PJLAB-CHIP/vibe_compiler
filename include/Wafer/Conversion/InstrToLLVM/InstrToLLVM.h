//===- TargetConversion.h - Typed target conversion request ----*- C++ -*-===//

#ifndef WAFER_CONVERSION_INSTRTOLLVM_H
#define WAFER_CONVERSION_INSTRTOLLVM_H

#include "Wafer/Target/TargetIdentity.h"

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

/// Complete, typed request for production instruction-to-target conversion.
/// The production pipeline targets the one current Wafer backend.
struct TargetConversionRequest {
  int64_t defaultDDRArenaArgumentIndex = -1;
  int64_t cardId = -1;
  int64_t tileId = -1;
  int64_t transportStatusArgumentIndex = -1;
  bool transportPreparedBeforeEntry = false;
  int64_t profileRecordArgumentIndex = -1;
};

std::unique_ptr<mlir::Pass>
createLowerInstrToTargetLLVMPass(const TargetConversionRequest &request);

} // namespace wafer

#endif // WAFER_CONVERSION_INSTRTOLLVM_H
