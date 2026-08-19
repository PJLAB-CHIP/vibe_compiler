//===- TensorProgramAlternative.cpp -----------------------------------===//

#include "Wafer/Planning/Search/TensorProgramAlternative.h"

#include "Wafer/Planning/Search/AttentionAlternative.h"
#include "Wafer/Planning/Search/AttentionAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Verifier.h"

#include <cassert>
#include <limits>
#include <utility>

namespace wafer::compiler::detail {
namespace {

using tensor_program_alternatives::AttentionProgramAlgorithm;
using tensor_program_alternatives::AttentionSemantics;

static bool hasFusibleScoreComputation(AttentionSemantics semantics) {
  auto scoreResult = mlir::dyn_cast<mlir::OpResult>(semantics.scores);
  mlir::Operation *producer = scoreResult ? scoreResult.getOwner() : nullptr;
  if (!producer || !mlir::isa<mlir::TilingInterface>(producer) ||
      producer->getBlock() != semantics.rowMax->getBlock())
    return false;

  unsigned uses = 0;
  bool sawRowMaximum = false;
  bool sawShift = false;
  for (mlir::OpOperand &use : scoreResult.getUses()) {
    ++uses;
    sawRowMaximum |= use.getOwner() == semantics.rowMax.getOperation();
    sawShift |= use.getOwner() == semantics.shifted.getOperation();
    if (use.getOwner() != semantics.rowMax.getOperation() &&
        use.getOwner() != semantics.shifted.getOperation())
      return false;
  }
  return uses == 2 && sawRowMaximum && sawShift;
}

static int64_t maximumBlockSize(int64_t reductionExtent,
                                int64_t partitionCount) {
  return reductionExtent / partitionCount +
         (reductionExtent % partitionCount != 0);
}

static mlir::func::FuncOp findSingleTensorProgram(mlir::ModuleOp module) {
  mlir::func::FuncOp found;
  bool multiple = false;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.isExternal())
      return;
    if (found) {
      multiple = true;
      return;
    }
    found = function;
  });
  return multiple ? mlir::func::FuncOp{} : found;
}

} // namespace

TensorProgramAlternativeAssignment
TensorProgramAlternativeAssignment::original() {
  return {};
}

TensorProgramAlternativeAssignment
TensorProgramAlternativeAssignment::onlineAttention(int64_t blockSize) {
  return {TensorProgramAlternativeKind::OnlineAttention, blockSize, 1};
}

TensorProgramAlternativeAssignment
TensorProgramAlternativeAssignment::splitKeyValueAttention(
    int64_t blockSize, int64_t partitionCount) {
  return {TensorProgramAlternativeKind::SplitKeyValueAttention, blockSize,
          partitionCount};
}

TensorProgramAlternativeDomain
TensorProgramAlternativeDomain::originalOnly() {
  return TensorProgramAlternativeDomain(/*reductionExtent=*/0,
                                        /*splitKeyValue=*/false);
}

TensorProgramAlternativeDomain
TensorProgramAlternativeDomain::attention(int64_t reductionExtent,
                                          bool supportsSplit) {
  assert(reductionExtent > 0 && "attention domain must be nonempty");
  return TensorProgramAlternativeDomain(reductionExtent, supportsSplit);
}

TensorProgramAlternativeAssignment
TensorProgramAlternativeDomain::getFirstAssignment() const {
  return TensorProgramAlternativeAssignment::original();
}

bool TensorProgramAlternativeDomain::contains(
    const TensorProgramAlternativeAssignment &assignment) const {
  switch (assignment.kind) {
  case TensorProgramAlternativeKind::Original:
    return assignment.keyValueBlockSize == 0 &&
           assignment.keyValuePartitionCount == 1;
  case TensorProgramAlternativeKind::OnlineAttention:
    return reductionExtent > 0 && assignment.keyValueBlockSize > 0 &&
           assignment.keyValueBlockSize <= reductionExtent &&
           assignment.keyValuePartitionCount == 1;
  case TensorProgramAlternativeKind::SplitKeyValueAttention:
    return splitKeyValue && assignment.keyValuePartitionCount > 1 &&
           assignment.keyValuePartitionCount <= reductionExtent &&
           assignment.keyValueBlockSize > 0 &&
           assignment.keyValueBlockSize <=
               maximumBlockSize(reductionExtent,
                                assignment.keyValuePartitionCount);
  }
  return false;
}

mlir::FailureOr<std::optional<TensorProgramAlternativeAssignment>>
TensorProgramAlternativeDomain::getNextAssignment(
    const TensorProgramAlternativeAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  switch (assignment.kind) {
  case TensorProgramAlternativeKind::Original:
    if (reductionExtent == 0)
      return std::optional<TensorProgramAlternativeAssignment>{};
    return std::optional<TensorProgramAlternativeAssignment>(
        TensorProgramAlternativeAssignment::onlineAttention(1));
  case TensorProgramAlternativeKind::OnlineAttention:
    if (assignment.keyValueBlockSize < reductionExtent)
      return std::optional<TensorProgramAlternativeAssignment>(
          TensorProgramAlternativeAssignment::onlineAttention(
              assignment.keyValueBlockSize + 1));
    if (!splitKeyValue || reductionExtent < 2)
      return std::optional<TensorProgramAlternativeAssignment>{};
    return std::optional<TensorProgramAlternativeAssignment>(
        TensorProgramAlternativeAssignment::splitKeyValueAttention(1, 2));
  case TensorProgramAlternativeKind::SplitKeyValueAttention:
    if (assignment.keyValueBlockSize <
        maximumBlockSize(reductionExtent,
                         assignment.keyValuePartitionCount))
      return std::optional<TensorProgramAlternativeAssignment>(
          TensorProgramAlternativeAssignment::splitKeyValueAttention(
              assignment.keyValueBlockSize + 1,
              assignment.keyValuePartitionCount));
    if (assignment.keyValuePartitionCount < reductionExtent)
      return std::optional<TensorProgramAlternativeAssignment>(
          TensorProgramAlternativeAssignment::splitKeyValueAttention(
              1, assignment.keyValuePartitionCount + 1));
    return std::optional<TensorProgramAlternativeAssignment>{};
  }
  return mlir::failure();
}

mlir::FailureOr<uint64_t>
TensorProgramAlternativeDomain::getAssignmentCount() const {
  uint64_t count = 1;
  if (reductionExtent == 0)
    return count;
  count += static_cast<uint64_t>(reductionExtent);
  if (!splitKeyValue)
    return count;
  for (int64_t partitions = 2; partitions <= reductionExtent; ++partitions) {
    uint64_t blocks =
        static_cast<uint64_t>(maximumBlockSize(reductionExtent, partitions));
    if (blocks > std::numeric_limits<uint64_t>::max() - count)
      return mlir::failure();
    count += blocks;
  }
  return count;
}

mlir::FailureOr<TensorProgramAlternativeDomain>
getTensorProgramAlternativeDomain(mlir::ModuleOp source) {
  if (!source || !findSingleTensorProgram(source))
    return mlir::failure();

  mlir::FailureOr<tensor_program_alternatives::DecodeAttentionSemantics>
      decode = tensor_program_alternatives::analyzeDecodeAttentionSemantics(
          source);
  if (mlir::succeeded(decode) &&
      hasFusibleScoreComputation(decode->attention))
    return TensorProgramAlternativeDomain::attention(
        decode->attention.reductionExtent, /*supportsSplit=*/true);

  mlir::FailureOr<AttentionSemantics> attention =
      tensor_program_alternatives::analyzeAttentionSemantics(source);
  if (mlir::succeeded(attention) && hasFusibleScoreComputation(*attention))
    return TensorProgramAlternativeDomain::attention(
        attention->reductionExtent, /*supportsSplit=*/false);
  return TensorProgramAlternativeDomain::originalOnly();
}

TensorProgramAlternativeMaterialization
TensorProgramAlternativeMaterialization::materialized(
    mlir::OwningOpRef<mlir::ModuleOp> module) {
  return {TensorProgramAlternativeMaterializationKind::Materialized,
          std::move(module), {}};
}

TensorProgramAlternativeMaterialization
TensorProgramAlternativeMaterialization::invalidAssignment(
    std::string diagnostic) {
  return {TensorProgramAlternativeMaterializationKind::InvalidAssignment,
          {}, std::move(diagnostic)};
}

TensorProgramAlternativeMaterialization
TensorProgramAlternativeMaterialization::indeterminate(
    std::string diagnostic) {
  return {TensorProgramAlternativeMaterializationKind::Indeterminate, {},
          std::move(diagnostic)};
}

mlir::OwningOpRef<mlir::ModuleOp>
TensorProgramAlternativeMaterialization::takeModule() {
  assert(kind == TensorProgramAlternativeMaterializationKind::Materialized &&
         module && "only a materialized result owns a module");
  return std::move(module);
}

TensorProgramAlternativeMaterialization materializeTensorProgramAlternative(
    mlir::ModuleOp source,
    const TensorProgramAlternativeAssignment &assignment) {
  mlir::FailureOr<TensorProgramAlternativeDomain> domain =
      getTensorProgramAlternativeDomain(source);
  if (mlir::failed(domain))
    return TensorProgramAlternativeMaterialization::indeterminate(
        "TensorProgram alternative query requires one standalone program");
  if (!domain->contains(assignment))
    return TensorProgramAlternativeMaterialization::invalidAssignment(
        "TensorProgram alternative is outside the current typed domain");

  mlir::OwningOpRef<mlir::ModuleOp> alternative =
      mlir::cast<mlir::ModuleOp>(source->clone());
  if (assignment.kind == TensorProgramAlternativeKind::Original)
    return TensorProgramAlternativeMaterialization::materialized(
        std::move(alternative));

  mlir::func::FuncOp function = findSingleTensorProgram(*alternative);
  AttentionProgramAlgorithm algorithm =
      assignment.kind == TensorProgramAlternativeKind::OnlineAttention
          ? AttentionProgramAlgorithm::Online
          : AttentionProgramAlgorithm::SplitKeyValue;
  std::string diagnostic;
  if (mlir::failed(tensor_program_alternatives::
                       materializeAttentionProgramAlternative(
                           function, algorithm, assignment.keyValueBlockSize,
                           assignment.keyValuePartitionCount, &diagnostic)) ||
      mlir::failed(mlir::verify(*alternative)))
    return TensorProgramAlternativeMaterialization::indeterminate(
        diagnostic.empty() ? "TensorProgram alternative produced invalid IR"
                           : std::move(diagnostic));
  return TensorProgramAlternativeMaterialization::materialized(
      std::move(alternative));
}

} // namespace wafer::compiler::detail
