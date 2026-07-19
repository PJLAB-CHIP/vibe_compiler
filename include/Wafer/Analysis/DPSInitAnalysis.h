//===- DPSInitAnalysis.h - Recomputable destination-init facts -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_DPSINITANALYSIS_H
#define WAFER_ANALYSIS_DPSINITANALYSIS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"

#include <cstdint>

namespace mlir::linalg {
class LinalgOp;
} // namespace mlir::linalg

namespace wafer {

enum class InitReadState : uint8_t { Unread, Read };
enum class InitOrigin : uint8_t { Undefined, ExactSplat, ExistingValue };

/// Invocation-local facts recovered exclusively from the current DPS tie and
/// SSA/view chain.  They are never persisted in IR or a side table.
struct DPSInitFacts {
  InitReadState readState = InitReadState::Read;
  InitOrigin origin = InitOrigin::ExistingValue;
  mlir::Value splatScalar;
  mlir::Value sourceRoot;
  mlir::Attribute exactSplatValue;
  uint64_t inspectedNodes = 0;
};

DPSInitFacts analyzeDPSInit(mlir::linalg::LinalgOp operation,
                            unsigned initIndex);

} // namespace wafer

#endif // WAFER_ANALYSIS_DPSINITANALYSIS_H
