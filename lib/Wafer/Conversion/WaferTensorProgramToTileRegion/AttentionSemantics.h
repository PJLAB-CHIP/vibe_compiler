//===- AttentionSemantics.h - Query-local attention semantics -*- C++ -*-===//
#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace wafer::tensor_program_to_tile_region {

/// Transient facts proven from current structured SSA.  The object is valid
/// only until that IR is mutated and is never attached to an operation or
/// persisted with a candidate.
struct AttentionSemantics {
  mlir::linalg::LinalgOp output;
  mlir::linalg::GenericOp probability;
  mlir::linalg::GenericOp sumBroadcast;
  mlir::linalg::GenericOp rowSum;
  mlir::linalg::GenericOp exponential;
  mlir::linalg::GenericOp shifted;
  mlir::linalg::GenericOp maxBroadcast;
  mlir::linalg::GenericOp rowMax;
  mlir::Value observableOutput;
  mlir::Value probabilityStorage;
  mlir::Value physicalValues;
  mlir::Value scores;
  /// Logical value tensor before layout-only reassociation.
  mlir::Value values;
  mlir::Value maxInit;
  mlir::Value sumInit;
  mlir::Value outputInit;
  mlir::RankedTensorType scoreType;
  mlir::RankedTensorType valueType;
  mlir::RankedTensorType outputType;
  mlir::RankedTensorType physicalOutputType;
  int64_t reductionExtent = 0;
  unsigned outputIndex = 0;
};

/// A functional decode boundary proven entirely from current SSA.  Each cache
/// update is an exact prefix append into a fresh result tensor; the updated K
/// and V values are both consumed by attention and returned to the caller for
/// the next invocation.  No field is persisted in IR.
struct DecodeAttentionSemantics {
  AttentionSemantics attention;
  mlir::Value pastKey;
  mlir::Value newKey;
  mlir::Value updatedKey;
  mlir::Value pastValue;
  mlir::Value newValue;
  mlir::Value updatedValue;
  unsigned keyOutputIndex = 0;
  unsigned valueOutputIndex = 0;
  int64_t cacheDimension = -1;
};

mlir::FailureOr<AttentionSemantics>
analyzeAttentionSemantics(mlir::Operation *root,
                          std::string *failureReason = nullptr);

mlir::FailureOr<AttentionSemantics>
analyzeAttentionSemantics(mlir::ModuleOp module,
                          std::string *failureReason = nullptr);

mlir::FailureOr<DecodeAttentionSemantics>
analyzeDecodeAttentionSemantics(mlir::ModuleOp module,
                                std::string *failureReason = nullptr);

} // namespace wafer::tensor_program_to_tile_region
