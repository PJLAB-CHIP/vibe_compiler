//===- EquivalentInputVariantInternal.h - Qualification input variants -*- C++
//-*-===//

#ifndef WAFER_PIPELINES_EQUIVALENTINPUTVARIANTINTERNAL_H
#define WAFER_PIPELINES_EQUIVALENTINPUTVARIANTINTERNAL_H

#include "Wafer/Support/OptimizationInvocation.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer::qualification_internal {

/// Creates the qualification-only metamorphic input constructor. Production
/// never calls this factory. The produced full tensor slices must be consumed
/// by required tensor normalization before the structured tensor artifact is
/// published.
std::unique_ptr<mlir::Pass>
createEquivalentInputVariantPass(EquivalentInputVariantV1 variant);

} // namespace wafer::qualification_internal

#endif // WAFER_PIPELINES_EQUIVALENTINPUTVARIANTINTERNAL_H
