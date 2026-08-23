//===- TensorOps.cpp - Selected tensor-stage operations ----------------===//

#include "Wafer/IR/WaferDialect.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace wafer;

mlir::LogicalResult TensorCompletionOp::verify() {
  if (getStates().empty() || getResults().size() != getStates().size())
    return emitOpError(
        "requires a nonempty state list with one forwarded result per input");
  for (auto [state, result] : llvm::zip_equal(getStates(), getResults())) {
    auto stateType = mlir::dyn_cast<mlir::RankedTensorType>(state.getType());
    if (!stateType || !stateType.hasStaticShape() ||
        result.getType() != state.getType())
      return emitOpError(
          "requires matching static ranked tensor state/result types");
  }
  llvm::ArrayRef<int64_t> participants = getParticipants();
  if (participants.empty() || !llvm::is_sorted(participants) ||
      std::adjacent_find(participants.begin(), participants.end()) !=
          participants.end() ||
      llvm::any_of(participants, [](int64_t participant) {
        return participant < 0 ||
               participant >= static_cast<int64_t>(kNCCWorkerCount);
      }))
    return emitOpError(
        "requires a sorted unique nonempty NCC worker participant set");
  return mlir::success();
}

mlir::FailureOr<TensorIndexingDescription>
TensorCompletionOp::getTensorIndexingDescription(unsigned result) {
  if (result >= getResults().size() || result >= getStates().size())
    return mlir::failure();
  TensorIndexingDescription description;
  description.kind = TensorIndexingTransformKind::Cast;
  description.result = result;
  description.operands.push_back(TensorIndexingOperandDescription{
      static_cast<uint32_t>(result), TensorIndexingOperandRole::Source, {}, {}});
  return description;
}
