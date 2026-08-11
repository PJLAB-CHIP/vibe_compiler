//===- PhysicalDataflow.h - MLIR-native physical dataflow ------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PHYSICALDATAFLOW_H
#define WAFER_TRANSFORMS_PHYSICALDATAFLOW_H

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace wafer {

/// Remove derived SPM/DDR placement and Direct-DTE binding facts from a
/// complete whole-card physical-Tile candidate before cloning another
/// physical realization.
/// The candidate's ordinary SSA/control-flow/instruction IR remains the sole
/// semantic input; every derived clone must rerun placement and verification.
void clearPhysicalTileCandidateFacts(mlir::ModuleOp module);

} // namespace wafer

#endif // WAFER_TRANSFORMS_PHYSICALDATAFLOW_H
