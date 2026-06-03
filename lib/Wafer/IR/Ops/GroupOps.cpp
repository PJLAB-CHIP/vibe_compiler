//===- GroupOps.cpp - Wafer Group verifier implementation ----------===//

#include "Wafer/IR/WaferDialect.h"

#include "OpVerifierUtils.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/ADT/STLExtras.h"

using namespace wafer;
using namespace wafer::detail;

namespace {

bool isRawStableHLOOp(mlir::Operation *op) {
  llvm::StringRef dialect = op->getName().getDialectNamespace();
  return dialect == "stablehlo" || dialect == "mhlo";
}

bool isAllowedWaferGroupBodyOp(mlir::Operation *op) {
  if (mlir::isa<GroupYieldOp>(op))
    return true;
  return op->getName().getStringRef().starts_with("wafer.tensor_collective.");
}

bool isLowerLevelWaferOp(mlir::Operation *op) {
  return op->getName().getDialectNamespace() == "wafer" &&
         !isAllowedWaferGroupBodyOp(op);
}

bool isAllowedTensorLevelDialect(mlir::Operation *op) {
  llvm::StringRef dialect = op->getName().getDialectNamespace();
  return dialect == "arith" || dialect == "linalg" || dialect == "math" ||
         dialect == "scf" || dialect == "tensor";
}

bool isForbiddenGroupBoundaryType(mlir::Type type) {
  return isSPMMemRef(type) || isSPMTileBuffer(type) ||
         mlir::isa<mlir::async::TokenType>(type);
}

} // namespace

mlir::LogicalResult GroupOp::verify() {
  if (getNumResults() != getOuts().size())
    return emitOpError("expected result count to match outs count, got ")
           << getNumResults() << " results and " << getOuts().size() << " outs";

  for (auto [index, resultAndOut] :
       llvm::enumerate(llvm::zip(getResults(), getOuts()))) {
    mlir::Type resultType = std::get<0>(resultAndOut).getType();
    mlir::Type outType = std::get<1>(resultAndOut).getType();
    if (resultType != outType)
      return emitOpError("result type ")
             << resultType << " does not match outs type " << outType
             << " at index " << index;
  }

  for (auto input : getInputs()) {
    if (isSPMMemRef(input.getType()))
      return emitOpError("does not accept SPM memref inputs");
    if (isSPMTileBuffer(input.getType()))
      return emitOpError("does not accept SPM tile_buffer inputs");
    if (mlir::isa<mlir::async::TokenType>(input.getType()))
      return emitOpError("does not accept async token inputs");
  }

  if (getBody().empty())
    return emitOpError("expected non-empty body region");

  mlir::Block &block = getBody().front();
  size_t expectedBlockArgs = getInputs().size() + getOuts().size();
  if (block.getNumArguments() != expectedBlockArgs)
    return emitOpError("expected ")
           << expectedBlockArgs
           << " body block arguments matching group ins plus outs, got "
           << block.getNumArguments();

  unsigned blockArgIndex = 0;
  for (auto input : getInputs()) {
    mlir::Type blockArgType = block.getArgument(blockArgIndex).getType();
    if (blockArgType != input.getType())
      return emitOpError("body block argument type ")
             << blockArgType << " does not match input type " << input.getType()
             << " at index " << blockArgIndex;
    if (isSPMMemRef(blockArgType))
      return emitOpError("does not accept SPM memref body arguments");
    if (isSPMTileBuffer(blockArgType))
      return emitOpError("does not accept SPM tile_buffer body arguments");
    if (mlir::isa<mlir::async::TokenType>(blockArgType))
      return emitOpError("does not accept async token body arguments");
    ++blockArgIndex;
  }
  for (auto out : getOuts()) {
    mlir::Type blockArgType = block.getArgument(blockArgIndex).getType();
    if (blockArgType != out.getType())
      return emitOpError("body block argument type ")
             << blockArgType << " does not match outs type " << out.getType()
             << " at index " << blockArgIndex;
    if (isForbiddenGroupBoundaryType(blockArgType))
      return emitOpError("does not accept lower-level body arguments");
    ++blockArgIndex;
  }

  auto yield = mlir::dyn_cast<GroupYieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError("expected wafer.group_yield terminator");

  if (yield.getValues().size() != getNumResults())
    return emitOpError(
               "expected group_yield value count to match result count, got ")
           << yield.getValues().size() << " values and " << getNumResults()
           << " results";

  for (auto [index, yieldedAndResult] :
       llvm::enumerate(llvm::zip(yield.getValues(), getResults()))) {
    mlir::Type yieldedType = std::get<0>(yieldedAndResult).getType();
    mlir::Type resultType = std::get<1>(yieldedAndResult).getType();
    if (yieldedType != resultType)
      return emitOpError("group yield type ")
             << yieldedType << " does not match result type " << resultType
             << " at index " << index;
  }

  mlir::WalkResult bodyLegality =
      getBody().walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        if (isRawStableHLOOp(op)) {
          emitOpError("body cannot contain raw StableHLO op '")
              << op->getName() << "'";
          return mlir::WalkResult::interrupt();
        }

        if (isLowerLevelWaferOp(op)) {
          emitOpError("body cannot contain lower-level op '")
              << op->getName() << "'";
          return mlir::WalkResult::interrupt();
        }

        if (!isAllowedWaferGroupBodyOp(op) &&
            !isAllowedTensorLevelDialect(op)) {
          emitOpError("body cannot contain unsupported op '")
              << op->getName() << "'";
          return mlir::WalkResult::interrupt();
        }

        for (mlir::Value operand : op->getOperands()) {
          if (isForbiddenGroupBoundaryType(operand.getType())) {
            emitOpError("body cannot contain lower-level operand type ")
                << operand.getType() << " on op '" << op->getName() << "'";
            return mlir::WalkResult::interrupt();
          }
        }
        for (mlir::Value result : op->getResults()) {
          if (isForbiddenGroupBoundaryType(result.getType())) {
            emitOpError("body cannot contain lower-level result type ")
                << result.getType() << " on op '" << op->getName() << "'";
            return mlir::WalkResult::interrupt();
          }
        }

        return mlir::WalkResult::advance();
      });
  if (bodyLegality.wasInterrupted())
    return mlir::failure();

  for (mlir::NamedAttribute attr : getOperation()->getAttrs()) {
    if (attr.getName() != getOperandSegmentSizesAttrName())
      return emitOpError("does not accept semantic attributes");
  }

  return mlir::success();
}

void GroupOp::collectWaferTilingDemand(
    llvm::SmallVectorImpl<WaferTilingDemand> &demands) {
  for (auto [index, input] : llvm::enumerate(getInputs()))
    demands.push_back({WaferTilingDemandKind::Input,
                       static_cast<unsigned>(index), input.getType()});
  for (auto [index, out] : llvm::enumerate(getOuts()))
    demands.push_back({WaferTilingDemandKind::Output,
                       static_cast<unsigned>(index), out.getType()});
  for (auto [index, result] : llvm::enumerate(getResults()))
    demands.push_back({WaferTilingDemandKind::Result,
                       static_cast<unsigned>(index), result.getType()});
}

mlir::LogicalResult GroupOp::verifyWaferTilingContract() {
  llvm::SmallVector<WaferTilingDemand, 8> demands;
  collectWaferTilingDemand(demands);
  if (demands.empty())
    return emitOpError("tiling interface must expose group boundary values");
  for (const WaferTilingDemand &demand : demands) {
    if (!demand.type)
      return emitOpError("tiling interface returned a demand without type");
    if (isSPMMemRef(demand.type))
      return emitOpError(
          "tiling interface must not expose SPM memref at group level");
  }
  return mlir::success();
}
