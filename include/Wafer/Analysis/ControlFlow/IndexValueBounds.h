//===- IndexValueBounds.h - Index range interface models --------*- C++ -*-===//

#ifndef WAFER_ANALYSIS_CONTROLFLOW_INDEXVALUEBOUNDS_H
#define WAFER_ANALYSIS_CONTROLFLOW_INDEXVALUEBOUNDS_H

namespace mlir {
class DialectRegistry;
}

namespace wafer::analysis {

/// Add index extrema and constant-positive signed ceil-div bounds missing
/// from the pinned arith interface models.
void registerIndexValueBoundsModels(mlir::DialectRegistry &registry);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_CONTROLFLOW_INDEXVALUEBOUNDS_H
