//===- PhysicalDataflow.h - MLIR-native physical dataflow ------*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_PHYSICALDATAFLOW_H
#define WAFER_TRANSFORMS_PHYSICALDATAFLOW_H

namespace mlir {
class DialectRegistry;
}

namespace wafer {

/// Register Wafer-owned target implementation external models for supported
/// upstream structured operations.
void registerTargetImplementationExternalModels(
    mlir::DialectRegistry &registry);

} // namespace wafer

#endif // WAFER_TRANSFORMS_PHYSICALDATAFLOW_H
