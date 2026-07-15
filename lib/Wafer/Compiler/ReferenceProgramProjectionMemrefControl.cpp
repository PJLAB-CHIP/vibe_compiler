//===- ReferenceProgramProjectionMemrefControl.cpp -----------------------===//

#include "ReferenceProgramProjectionInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <memory>

namespace wafer::compiler::reference_detail {
namespace {

llvm::Expected<int64_t> getStaticViewOffsetBytes(mlir::MemRefType type) {
  llvm::SmallVector<int64_t> strides;
  int64_t offsetElements = 0;
  if (mlir::failed(mlir::getStridesAndOffset(type, strides, offsetElements)) ||
      offsetElements == mlir::ShapedType::kDynamic || offsetElements < 0)
    return unsupported("view requires a non-negative static layout offset");
  auto info = wafer::computeWaferPhysicalTensorInfo(type);
  if (!info || info->elementBytes <= 0 || info->bitPackedElement)
    return unsupported("view requires byte-addressable physical geometry");
  if (info->layout == wafer::MemLayout::Cx ||
      info->layout == wafer::MemLayout::NCx) {
    if (offsetElements != 0)
      return unsupported("Cx/NCx view requires zero layout offset");
    return 0;
  }
  if (offsetElements > std::numeric_limits<int64_t>::max() / info->elementBytes)
    return unsupported("view byte offset overflows int64");
  return offsetElements * info->elementBytes;
}

llvm::Expected<int64_t> getStaticViewDeltaBytes(mlir::MemRefType sourceType,
                                                mlir::MemRefType resultType) {
  auto source = getStaticViewOffsetBytes(sourceType);
  auto result = getStaticViewOffsetBytes(resultType);
  if (!source)
    return source.takeError();
  if (!result)
    return result.takeError();
  if (*result >= *source)
    return *result - *source;
  return -(*source - *result);
}

} // namespace

llvm::Expected<bool>
ProgramProjector::projectMemrefAndStructuredControl(mlir::Operation &operation,
                                                    Command &command) {
  if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
    if (!alloc.getDynamicSizes().empty() || !alloc.getSymbolOperands().empty())
      return unsupported("dynamic memref allocation");
    auto info = wafer::computeWaferPhysicalTensorInfo(alloc.getType());
    if (!info || info->physicalBytes < 0)
      return unsupported("allocation has no static physical extent");
    command.kind = CommandKind::Alloc;
    command.type = alloc.getType();
    command.physicalBytes = info->physicalBytes;
    if (wafer::isWaferSPMMemRefType(alloc.getType())) {
      auto offset = alloc->getAttrOfType<wafer::SPMOffsetAttr>(
          wafer::kWaferSPMOffsetAttrName);
      if (!offset)
        return invalid("SPM allocation is missing accepted offset");
      command.acceptedOffset = offset.getOffset();
      command.spmAllocation = true;
    } else if (wafer::isWaferDDRMemRefType(alloc.getType())) {
      auto offset = alloc->getAttrOfType<wafer::DDROffsetAttr>(
          wafer::kWaferDDROffsetAttrName);
      if (!offset)
        return invalid("DDR allocation is missing accepted offset");
      command.acceptedOffset = offset.getOffset();
    } else {
      return unsupported("allocation is outside Wafer DDR/SPM memory");
    }
    auto result = define(alloc.getResult());
    if (!result)
      return result.takeError();
    command.result = *result;
  } else if (auto dealloc =
                 mlir::dyn_cast<mlir::memref::DeallocOp>(operation)) {
    auto source = use(dealloc.getMemref());
    if (!source)
      return source.takeError();
    command.kind = CommandKind::Dealloc;
    command.source = *source;
  } else if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(operation)) {
    auto source = use(cast.getSource());
    if (!source)
      return source.takeError();
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(cast.getSource().getType());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(cast.getType());
    auto sourceInfo = sourceType
                          ? wafer::computeWaferPhysicalTensorInfo(sourceType)
                          : std::nullopt;
    auto resultInfo = resultType
                          ? wafer::computeWaferPhysicalTensorInfo(resultType)
                          : std::nullopt;
    if (!sourceInfo || !resultInfo || resultInfo->physicalBytes < 0 ||
        resultInfo->physicalBytes > sourceInfo->physicalBytes)
      return unsupported("memref.cast changes accepted physical extent");
    auto result = define(cast.getResult());
    if (!result)
      return result.takeError();
    command.kind = CommandKind::Cast;
    command.source = *source;
    command.result = *result;
    command.type = resultType;
    command.physicalBytes = resultInfo->physicalBytes;
  } else if (auto subview =
                 mlir::dyn_cast<mlir::memref::SubViewOp>(operation)) {
    if (llvm::Error error =
            projectStaticView(subview.getSource(), subview.getResult(),
                              subview.getType(), command))
      return error;
  } else if (auto reinterpret =
                 mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(operation)) {
    if (llvm::Error error =
            projectStaticView(reinterpret.getSource(), reinterpret.getResult(),
                              reinterpret.getType(), command))
      return error;
  } else if (auto collapse =
                 mlir::dyn_cast<mlir::memref::CollapseShapeOp>(operation)) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(collapse.getSrc().getType());
    auto resultType = mlir::dyn_cast<mlir::MemRefType>(collapse.getType());
    if (!sourceType || !resultType || !wafer::isWaferMemRefType(sourceType) ||
        !wafer::isWaferMemRefType(resultType))
      return unsupported("collapse_shape must preserve Wafer memory");
    wafer::MemoryAttr sourceMemory = wafer::getWaferMemoryAttr(sourceType);
    wafer::MemoryAttr resultMemory = wafer::getWaferMemoryAttr(resultType);
    auto sourceInfo = wafer::computeWaferPhysicalTensorInfo(sourceType);
    auto resultInfo = wafer::computeWaferPhysicalTensorInfo(resultType);
    if (!sourceType.hasStaticShape() || !resultType.hasStaticShape() ||
        sourceMemory.getSpace() != resultMemory.getSpace() ||
        sourceMemory.getLayout() != wafer::MemLayout::Tensor ||
        resultMemory.getLayout() != wafer::MemLayout::Tensor ||
        sourceType.getElementType() != resultType.getElementType() ||
        sourceType.getNumElements() != resultType.getNumElements() ||
        !sourceInfo || !resultInfo ||
        sourceInfo->compactBytes != resultInfo->compactBytes ||
        sourceInfo->physicalBytes != resultInfo->physicalBytes)
      return unsupported(
          "collapse_shape changes accepted tensor physical geometry");
    if (llvm::Error error = projectStaticView(
            collapse.getSrc(), collapse.getResult(), resultType, command))
      return error;
  } else if (auto constant =
                 mlir::dyn_cast<mlir::arith::ConstantOp>(operation)) {
    auto result = define(constant.getResult());
    if (!result)
      return result.takeError();
    command.kind = CommandKind::Constant;
    command.result = *result;
    if (auto value = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue()))
      command.scalarValue.floating = value.getValue();
    else if (auto value =
                 mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())) {
      command.scalarValue.integer = value.getValue();
      auto type = mlir::dyn_cast<mlir::IntegerType>(constant.getType());
      command.scalarValue.integerIsUnsigned = type && type.isUnsigned();
    } else {
      return unsupported("non-scalar arith.constant");
    }
  } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
    if (!ifOp.getThenRegion().hasOneBlock() ||
        (!ifOp.getElseRegion().empty() && !ifOp.getElseRegion().hasOneBlock()))
      return unsupported("scf.if requires single-block regions");
    if (!ifOp.getCondition().getType().isInteger(1))
      return unsupported("scf.if condition is not i1");
    auto condition = use(ifOp.getCondition());
    if (!condition)
      return condition.takeError();
    command.kind = CommandKind::If;
    command.condition = *condition;
    command.body = std::make_shared<ReferenceProgram::Impl::BlockProgram>();
    if (llvm::Error error =
            projectBlock(ifOp.getThenRegion().front(), *command.body))
      return error;
    if (command.body->commands.empty() ||
        command.body->commands.back().kind != CommandKind::Return ||
        command.body->commands.back().inputs.size() != ifOp.getNumResults())
      return unsupported("scf.if then yield arity is not projectable");
    if (!ifOp.getElseRegion().empty()) {
      command.elseBody =
          std::make_shared<ReferenceProgram::Impl::BlockProgram>();
      if (llvm::Error error =
              projectBlock(ifOp.getElseRegion().front(), *command.elseBody))
        return error;
      if (command.elseBody->commands.empty() ||
          command.elseBody->commands.back().kind != CommandKind::Return ||
          command.elseBody->commands.back().inputs.size() !=
              ifOp.getNumResults())
        return unsupported("scf.if else yield arity is not projectable");
    } else if (ifOp.getNumResults() != 0) {
      return unsupported("result-producing scf.if has no else region");
    }
    for (mlir::Value resultValue : ifOp.getResults()) {
      auto result = define(resultValue);
      if (!result)
        return result.takeError();
      command.results.push_back(*result);
    }
  } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
    if (!forOp.getRegion().hasOneBlock())
      return unsupported("scf.for requires a single-block body");
    auto lower = use(forOp.getLowerBound());
    auto upper = use(forOp.getUpperBound());
    auto step = use(forOp.getStep());
    if (!lower)
      return lower.takeError();
    if (!upper)
      return upper.takeError();
    if (!step)
      return step.takeError();
    command.kind = CommandKind::For;
    command.lowerBound = *lower;
    command.upperBound = *upper;
    command.step = *step;
    for (mlir::Value init : forOp.getInitArgs()) {
      auto value = use(init);
      if (!value)
        return value.takeError();
      command.iterInputs.push_back(*value);
    }
    mlir::Block &body = forOp.getRegion().front();
    if (body.getNumArguments() != 1 + forOp.getInitArgs().size())
      return unsupported("scf.for body argument arity is not projectable");
    auto induction = define(body.getArgument(0));
    if (!induction)
      return induction.takeError();
    command.inductionArgument = *induction;
    for (mlir::BlockArgument argument : body.getArguments().drop_front()) {
      auto value = define(argument);
      if (!value)
        return value.takeError();
      command.iterArguments.push_back(*value);
    }
    command.body = std::make_shared<ReferenceProgram::Impl::BlockProgram>();
    if (llvm::Error error = projectBlock(body, *command.body))
      return error;
    if (command.body->commands.empty() ||
        command.body->commands.back().kind != CommandKind::Return ||
        command.body->commands.back().inputs.size() != forOp.getNumResults())
      return unsupported("scf.for yield arity is not projectable");
    for (mlir::Value resultValue : forOp.getResults()) {
      auto result = define(resultValue);
      if (!result)
        return result.takeError();
      command.results.push_back(*result);
    }
  } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(operation)) {
    mlir::ModuleOp module = operation.getParentOfType<mlir::ModuleOp>();
    mlir::func::FuncOp callee =
        module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
    auto function =
        callee ? functionIds.find(callee.getOperation()) : functionIds.end();
    if (function == functionIds.end())
      return unsupported("func.call callee is outside projected closure");
    if (call.getOperandTypes() != callee.getFunctionType().getInputs() ||
        call.getResultTypes() != callee.getFunctionType().getResults())
      return unsupported("func.call signature disagrees with its callee");
    command.kind = CommandKind::Call;
    command.callee = function->second;
    for (mlir::Value operand : call.getOperands()) {
      auto input = use(operand);
      if (!input)
        return input.takeError();
      command.inputs.push_back(*input);
    }
    for (mlir::Value resultValue : call.getResults()) {
      auto result = define(resultValue);
      if (!result)
        return result.takeError();
      command.results.push_back(*result);
    }
  } else if (auto tile = mlir::dyn_cast<wafer::TileRegionOp>(operation)) {
    if (!tile.getBody().hasOneBlock())
      return unsupported("tile region requires unsupported multi-block body");
    mlir::Block &body = tile.getBody().front();
    if (body.getNumArguments() != tile.getInputs().size())
      return invalid("tile region argument arity mismatch");
    command.kind = CommandKind::TileRegion;
    for (mlir::Value input : tile.getInputs()) {
      auto id = use(input);
      if (!id)
        return id.takeError();
      command.inputs.push_back(*id);
    }
    for (mlir::BlockArgument argument : body.getArguments()) {
      auto id = define(argument);
      if (!id)
        return id.takeError();
      command.blockArguments.push_back(*id);
    }
    command.body = std::make_shared<ReferenceProgram::Impl::BlockProgram>();
    if (llvm::Error error = projectBlock(body, *command.body))
      return error;
    if (command.body->commands.empty() ||
        command.body->commands.back().kind != CommandKind::Return ||
        command.body->commands.back().inputs.size() != tile.getNumResults())
      return unsupported("tile region yield arity is not projectable");
    for (mlir::Value resultValue : tile.getResults()) {
      auto id = define(resultValue);
      if (!id)
        return id.takeError();
      command.results.push_back(*id);
    }

  } else {
    return false;
  }
  return true;
}

llvm::Expected<bool>
ProgramProjector::projectCFGAndReturn(mlir::Operation &operation,
                                      Command &command, bool &sawTerminator) {
  if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(operation)) {
    auto target = blocks.find(branch.getDest());
    if (target == blocks.end())
      return unsupported("cf.br targets a block outside entry CFG");
    if (branch.getDestOperands().size() !=
            branch.getDest()->getNumArguments() ||
        !llvm::equal(branch.getDestOperands().getTypes(),
                     branch.getDest()->getArgumentTypes()))
      return unsupported("cf.br successor operands disagree with block");
    command.kind = CommandKind::Branch;
    command.successor = target->second;
    for (mlir::Value operand : branch.getDestOperands()) {
      auto id = use(operand);
      if (!id)
        return id.takeError();
      command.inputs.push_back(*id);
    }
    sawTerminator = true;
  } else if (auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(operation)) {
    auto trueTarget = blocks.find(branch.getTrueDest());
    auto falseTarget = blocks.find(branch.getFalseDest());
    if (trueTarget == blocks.end() || falseTarget == blocks.end())
      return unsupported("cf.cond_br targets a block outside entry CFG");
    if (!branch.getCondition().getType().isInteger(1))
      return unsupported("cf.cond_br condition is not i1");
    if (branch.getTrueDestOperands().size() !=
            branch.getTrueDest()->getNumArguments() ||
        branch.getFalseDestOperands().size() !=
            branch.getFalseDest()->getNumArguments() ||
        !llvm::equal(branch.getTrueDestOperands().getTypes(),
                     branch.getTrueDest()->getArgumentTypes()) ||
        !llvm::equal(branch.getFalseDestOperands().getTypes(),
                     branch.getFalseDest()->getArgumentTypes()))
      return unsupported("cf.cond_br successor operands disagree with block");
    auto condition = use(branch.getCondition());
    if (!condition)
      return condition.takeError();
    command.kind = CommandKind::CondBranch;
    command.condition = *condition;
    command.trueSuccessor = trueTarget->second;
    command.falseSuccessor = falseTarget->second;
    for (mlir::Value operand : branch.getTrueDestOperands()) {
      auto id = use(operand);
      if (!id)
        return id.takeError();
      command.trueInputs.push_back(*id);
    }
    for (mlir::Value operand : branch.getFalseDestOperands()) {
      auto id = use(operand);
      if (!id)
        return id.takeError();
      command.falseInputs.push_back(*id);
    }
    sawTerminator = true;
  } else if (mlir::isa<wafer::TileYieldOp, mlir::scf::YieldOp,
                       mlir::func::ReturnOp>(operation)) {
    command.kind = CommandKind::Return;
    for (mlir::Value operand : operation.getOperands()) {
      auto id = use(operand);
      if (!id)
        return id.takeError();
      command.inputs.push_back(*id);
    }
    if (mlir::isa<mlir::func::ReturnOp>(operation) &&
        command.inputs.size() !=
            operation.getParentOfType<mlir::func::FuncOp>().getNumResults())
      return unsupported("function return arity disagrees with its type");
    sawTerminator = true;

  } else {
    return false;
  }
  return true;
}

llvm::Error ProgramProjector::projectStaticView(mlir::Value sourceValue,
                                                mlir::Value resultValue,
                                                mlir::MemRefType resultType,
                                                Command &command) {
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(sourceValue.getType());
  auto sourceInfo = sourceType
                        ? wafer::computeWaferPhysicalTensorInfo(sourceType)
                        : std::nullopt;
  auto resultInfo = wafer::computeWaferPhysicalTensorInfo(resultType);
  if (!sourceInfo || !resultInfo || resultInfo->physicalBytes < 0)
    return unsupported("static view has no accepted physical geometry");
  auto delta = getStaticViewDeltaBytes(sourceType, resultType);
  if (!delta)
    return delta.takeError();
  auto source = use(sourceValue);
  auto result = define(resultValue);
  if (!source)
    return source.takeError();
  if (!result)
    return result.takeError();
  command.kind = CommandKind::Cast;
  command.source = *source;
  command.result = *result;
  command.type = resultType;
  command.physicalBytes = resultInfo->physicalBytes;
  command.viewDelta = *delta;
  return llvm::Error::success();
}

} // namespace wafer::compiler::reference_detail
