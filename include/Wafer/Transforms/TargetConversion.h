//===- TargetConversion.h - Typed target conversion request ----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_TARGETCONVERSION_H
#define WAFER_TRANSFORMS_TARGETCONVERSION_H

#include "Wafer/Target/TargetLaunchABI.h"
#include "Wafer/Target/TargetProfile.h"

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

/// Complete, typed request for production instruction-to-target conversion.
/// The target profile has no default: production callers must forward the
/// profile already carried by their accepted executable artifact.
struct TargetConversionRequest {
  TargetProfileId targetProfile;
  int64_t defaultDDRArenaArgumentIndex = -1;
  int64_t logicalRank = -1;
  int64_t transportStatusArgumentIndex = -1;
  TargetLaunchABIId launchABI = TargetLaunchABIId::perRankPointerBlockV1();
  int64_t profileRecordArgumentIndex = -1;
};

std::unique_ptr<mlir::Pass>
createLowerInstrToTargetLLVMPass(const TargetConversionRequest &request);

} // namespace wafer

#endif // WAFER_TRANSFORMS_TARGETCONVERSION_H
