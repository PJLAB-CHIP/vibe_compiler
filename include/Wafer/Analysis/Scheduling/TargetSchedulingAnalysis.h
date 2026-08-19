//===- TargetSchedulingAnalysis.h - Scheduling query from Wafer IR -*- C++
//-*-===//

#ifndef WAFER_ANALYSIS_TARGETSCHEDULINGANALYSIS_H
#define WAFER_ANALYSIS_TARGETSCHEDULINGANALYSIS_H

#include "Wafer/Target/Core/TargetSchedulingCapability.h"

#include "mlir/IR/BuiltinOps.h"

namespace wafer::analysis {

/// Derives an exact categorical target-scheduling query from typed instruction
/// IR. Dynamic or unknown physical geometry is represented by
/// `geometryKnown=false` and cannot match a supported static capability row.
llvm::Expected<TargetSchedulingWindowQuery>
analyzeTargetSchedulingWindow(mlir::ModuleOp module,
                              TargetSchedulingMechanism mechanism);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_TARGETSCHEDULINGANALYSIS_H
