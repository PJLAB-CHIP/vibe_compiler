//===- SingleExecutionRegionFlow.h - Exact region forwarding ---*- C++ -*-===//

#ifndef WAFER_ANALYSIS_SINGLEEXECUTIONREGIONFLOW_H
#define WAFER_ANALYSIS_SINGLEEXECUTIONREGIONFLOW_H

#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace wafer::analysis {

/// Standard-interface view of an operation that invokes one single-block
/// region exactly once and then returns directly to its parent operation.
///
/// The view deliberately excludes conditionals, loops, and region graphs with
/// multiple successors. Those operations require path- or recurrence-aware
/// analysis instead of transparent value forwarding.
struct SingleExecutionRegionFlow {
  mlir::Region *region = nullptr;
  mlir::ValueRange entryOperands;
  mlir::ValueRange entryArguments;
  mlir::ValueRange exitOperands;
  mlir::ValueRange results;
};

inline std::optional<SingleExecutionRegionFlow>
getSingleExecutionRegionFlow(mlir::Operation *operation) {
  auto branch = mlir::dyn_cast_or_null<mlir::RegionBranchOpInterface>(operation);
  if (!branch || operation->getNumRegions() != 1 || branch.hasLoop())
    return std::nullopt;

  llvm::SmallVector<mlir::Attribute, 4> constants(operation->getNumOperands());
  for (auto [index, operand] : llvm::enumerate(operation->getOperands()))
    (void)mlir::matchPattern(operand, mlir::m_Constant(&constants[index]));

  llvm::SmallVector<mlir::InvocationBounds, 1> bounds;
  branch.getRegionInvocationBounds(constants, bounds);
  if (bounds.size() != 1 || bounds.front().getLowerBound() != 1 ||
      bounds.front().getUpperBound() != std::optional<unsigned>(1))
    return std::nullopt;

  mlir::Region &region = operation->getRegion(0);
  if (!region.hasOneBlock())
    return std::nullopt;

  llvm::SmallVector<mlir::RegionSuccessor, 1> entrySuccessors;
  branch.getSuccessorRegions(mlir::RegionBranchPoint::parent(),
                             entrySuccessors);
  if (entrySuccessors.size() != 1 ||
      entrySuccessors.front().getSuccessor() != &region)
    return std::nullopt;
  mlir::ValueRange entryArguments =
      entrySuccessors.front().getSuccessorInputs();
  mlir::OperandRange entryOperands =
      branch.getEntrySuccessorOperands(entrySuccessors.front());
  if (entryOperands.size() != entryArguments.size())
    return std::nullopt;

  auto terminator = mlir::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(
      region.front().getTerminator());
  if (!terminator)
    return std::nullopt;

  llvm::SmallVector<mlir::RegionSuccessor, 1> exitSuccessors;
  branch.getSuccessorRegions(region, exitSuccessors);
  if (exitSuccessors.size() != 1 || !exitSuccessors.front().isParent())
    return std::nullopt;
  mlir::ValueRange results = exitSuccessors.front().getSuccessorInputs();
  mlir::OperandRange exitOperands =
      terminator.getSuccessorOperands(exitSuccessors.front());
  if (exitOperands.size() != results.size())
    return std::nullopt;

  return SingleExecutionRegionFlow{&region, entryOperands, entryArguments,
                                   exitOperands, results};
}

inline mlir::Value
getSingleExecutionRegionEntryOperand(mlir::BlockArgument argument) {
  mlir::Operation *parent =
      argument.getOwner() ? argument.getOwner()->getParentOp() : nullptr;
  std::optional<SingleExecutionRegionFlow> flow =
      getSingleExecutionRegionFlow(parent);
  if (!flow || argument.getOwner() != &flow->region->front())
    return {};
  for (auto [entryOperand, entryArgument] :
       llvm::zip_equal(flow->entryOperands, flow->entryArguments))
    if (entryArgument == argument)
      return entryOperand;
  return {};
}

inline mlir::Value getSingleExecutionRegionExitOperand(mlir::OpResult result) {
  std::optional<SingleExecutionRegionFlow> flow =
      getSingleExecutionRegionFlow(result.getOwner());
  if (!flow)
    return {};
  for (auto [exitOperand, operationResult] :
       llvm::zip_equal(flow->exitOperands, flow->results))
    if (operationResult == result)
      return exitOperand;
  return {};
}

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_SINGLEEXECUTIONREGIONFLOW_H
