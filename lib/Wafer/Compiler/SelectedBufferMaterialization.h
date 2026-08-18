//===- SelectedBufferMaterialization.h - Joint buffer actualization -*- C++
//-*-===//

#pragma once

#include "Wafer/Conversion/WaferTensorProgramToCardModule/WaferTensorProgramToCardModule.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace wafer::compiler::detail {

enum class SelectedBufferMessageDirection : uint8_t { Send, Receive };

/// One exact Direct-DTE endpoint belonging to a selected logical edge on one
/// Tile. Message identity is already part of typed Instr IR; this
/// query-local record only binds that IR fact back to the joint-search edge.
struct SelectedBufferMessage {
  SelectedBufferMessageDirection direction =
      SelectedBufferMessageDirection::Send;
  int64_t communicationId = 0;
  int64_t payloadSlice = 0;
};

/// Query-local exact-buffer request for one selected logical SSA edge on one
/// Tile. DAG node ids are interpreted only against the current
/// invocation's materialized-buffer relations; they are not written into IR.
struct SelectedBufferRequest {
  std::optional<uint32_t> producerNode;
  std::optional<uint32_t> consumerNode;
  uint8_t bufferCount = 1;
  bool requireLocalDataflow = false;
  llvm::SmallVector<SelectedBufferMessage, 2> messages;
};

enum class SelectedBufferMaterializationFailureKind : uint8_t {
  None,
  InvalidRequest,
  NoExactLoop,
  EdgeNotWitnessed,
  NestedRegion,
  NoCrossEngineStage,
  TripCountTooSmall,
  MultiplicityMismatch,
  UnsupportedStructure,
};

struct SelectedBufferMaterializationFailure {
  SelectedBufferMaterializationFailureKind kind =
      SelectedBufferMaterializationFailureKind::None;
  size_t requestIndex = std::numeric_limits<size_t>::max();
  std::optional<uint32_t> producerNode;
  std::optional<uint32_t> consumerNode;
  uint8_t bufferCount = 1;
  std::string detail;
};

/// Successful actualization of one caller-owned Tile module. Module ownership
/// and every current-IR relation move together so a failed in-place rewrite
/// cannot leave the caller with relations into destroyed IR.
struct SelectedBufferingResult {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  StructuredMaterializationRelations materializationRelations;
  unsigned slotAllocationCount = 0;
};

/// Materializes the buffer multiplicity already selected by the structured-DAG
/// candidate into ordinary allocation, SSA recurrence and scf.for IR.  This
/// is an exact actualization gate, not a second candidate owner: it either
/// rewrites one (possibly nested) loop whose derived rotating-slot family has
/// exactly `requestedBufferCount`, or fails. Ownership is consumed so a failed
/// in-place transformation cannot expose partially rewritten IR; this
/// low-level mechanism entry does not clone the whole module to manufacture
/// rollback. Search policy must use the exact logical-edge overload below.
mlir::FailureOr<SelectedBufferingResult> materializeSelectedBuffering(
    mlir::OwningOpRef<mlir::ModuleOp> module, uint8_t requestedBufferCount,
    std::string *failureReason = nullptr, bool permitNoOpportunity = false);

/// Search-policy exact gate. Unlike the low-level mechanism entry above, this
/// overload must prove that the materialized stage dependency belongs to all
/// selected logical-edge requests on the Tile. A loop for an unrelated edge
/// is not an admissible witness. Ownership is consumed and this overload
/// applies once in place, returning the same owned module only on success.
mlir::FailureOr<SelectedBufferingResult> materializeSelectedBuffering(
    mlir::OwningOpRef<mlir::ModuleOp> module,
    llvm::ArrayRef<SelectedBufferRequest> requests,
    StructuredMaterializationRelations materializationRelations,
    SelectedBufferMaterializationFailure *failure = nullptr);

} // namespace wafer::compiler::detail
