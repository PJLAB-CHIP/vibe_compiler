//===- ReferenceProgramProjection.cpp - Accepted IR projection ----------===//

#include "ReferenceExecutorInternal.h"

#include "AcceptedCallClosure.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

llvm::Error validateMovementByteCount(uint64_t byteCount, uint64_t innerBytes,
                                      llvm::ArrayRef<int64_t> iterations) {
  if (iterations.size() != 3 || innerBytes == 0)
    return invalid("movement descriptor has invalid iteration rank");
  uint64_t chunks = 1;
  for (int64_t iteration : iterations) {
    if (iteration < 0 || static_cast<uint64_t>(iteration) >
                             std::numeric_limits<uint64_t>::max() / chunks)
      return invalid("movement descriptor iteration count overflows");
    chunks *= static_cast<uint64_t>(iteration);
  }
  if (chunks > std::numeric_limits<uint64_t>::max() / innerBytes ||
      byteCount != chunks * innerBytes)
    return invalid("movement byte_count disagrees with its descriptor");
  return llvm::Error::success();
}

bool isSupportedElementwiseKind(wafer::InstrElementwiseKind kind,
                                size_t &arity) {
  switch (kind) {
  case wafer::InstrElementwiseKind::Abs:
  case wafer::InstrElementwiseKind::Recip:
  case wafer::InstrElementwiseKind::Square:
  case wafer::InstrElementwiseKind::Sqrt:
  case wafer::InstrElementwiseKind::Rsqrt:
  case wafer::InstrElementwiseKind::Neg:
  case wafer::InstrElementwiseKind::Log2:
  case wafer::InstrElementwiseKind::Ln:
  case wafer::InstrElementwiseKind::Pow2:
  case wafer::InstrElementwiseKind::Exp:
  case wafer::InstrElementwiseKind::ExpLp:
  case wafer::InstrElementwiseKind::Sin:
  case wafer::InstrElementwiseKind::Cos:
  case wafer::InstrElementwiseKind::Tanh:
  case wafer::InstrElementwiseKind::Sigmoid:
  case wafer::InstrElementwiseKind::Relu:
  case wafer::InstrElementwiseKind::SatRelu:
  case wafer::InstrElementwiseKind::Softplus:
    arity = 1;
    return true;
  case wafer::InstrElementwiseKind::Max:
  case wafer::InstrElementwiseKind::Min:
  case wafer::InstrElementwiseKind::Add:
  case wafer::InstrElementwiseKind::Sub:
  case wafer::InstrElementwiseKind::Mul:
  case wafer::InstrElementwiseKind::Div:
    arity = 2;
    return true;
  default:
    return false;
  }
}

struct ConvertSpec {
  NumericFormat source;
  NumericFormat dest;
  wafer::InstrConvertParameterKind parameter;
};

std::optional<ConvertSpec> getConvertSpec(mlir::MLIRContext *context,
                                          wafer::InstrConvertKind kind) {
  auto [sourceType, destType] = wafer::getInstrConvertTypePair(context, kind);
  std::optional<NumericFormat> source = getNumericFormat(sourceType);
  std::optional<NumericFormat> dest = getNumericFormat(destType);
  if (!source || !dest)
    return std::nullopt;
  return ConvertSpec{*source, *dest, wafer::getInstrConvertParameterKind(kind)};
}

llvm::Expected<llvm::APFloat::roundingMode>
projectRoundingMode(uint64_t value) {
  switch (value) {
  case 0:
    return llvm::APFloat::rmNearestTiesToEven;
  case 1:
    return llvm::APFloat::rmTowardZero;
  case 2:
    return llvm::APFloat::rmTowardPositive;
  case 3:
    return llvm::APFloat::rmTowardNegative;
  case 4:
    return llvm::APFloat::rmNearestTiesToEven;
  default:
    return unsupported("convert rounding mode is outside RND_MODE");
  }
}

class ProgramProjector {
public:
  explicit ProgramProjector(ReferenceProgram::Impl &program)
      : program(program) {}

  llvm::Error project(detail::AcceptedCallClosure &closure) {
    if (closure.functions.size() > std::numeric_limits<uint32_t>::max())
      return unsupported("function id space is exhausted");
    program.functions.resize(closure.functions.size());

    for (auto [functionIndex, function] : llvm::enumerate(closure.functions)) {
      if (function.getBody().empty())
        return unsupported(
            ("function @" + function.getSymName() + " has no body").str());
      if (function.getBody().getBlocks().size() >
          std::numeric_limits<uint32_t>::max())
        return unsupported(
            ("function @" + function.getSymName() + " exhausts CFG block ids")
                .str());
      functionIds[function.getOperation()] =
          static_cast<uint32_t>(functionIndex);
      auto &projected = program.functions[functionIndex];
      projected.body.blocks.resize(function.getBody().getBlocks().size());
      for (auto [blockIndex, block] : llvm::enumerate(function.getBody())) {
        blocks[&block] = static_cast<uint32_t>(blockIndex);
        auto &projectedBlock = projected.body.blocks[blockIndex];
        for (mlir::BlockArgument argument : block.getArguments()) {
          auto value = define(argument);
          if (!value)
            return value.takeError();
          projectedBlock.arguments.push_back(*value);
        }
      }
      for (mlir::Block &block : function.getBody())
        for (mlir::Operation &operation : block)
          for (mlir::Value result : operation.getResults())
            if (llvm::Error error = predeclare(result))
              return error;

      for (mlir::BlockArgument argument :
           function.getBody().front().getArguments()) {
        auto type = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
        if (!type)
          return unsupported(("function @" + function.getSymName() +
                              " argument is not a memref")
                                 .str());
        projected.arguments.push_back(values.lookup(argument));
        projected.argumentTypes.push_back(type);
      }
    }

    auto entry = functionIds.find(closure.entry.getOperation());
    if (entry == functionIds.end())
      return unsupported("entry is outside the projected call closure");
    program.entryFunction = entry->second;

    for (auto [functionIndex, function] : llvm::enumerate(closure.functions)) {
      auto &projected = program.functions[functionIndex];
      for (auto [blockIndex, block] : llvm::enumerate(function.getBody()))
        if (llvm::Error error =
                projectBlock(block, projected.body.blocks[blockIndex]))
          return error;
      if (llvm::Error error = validateCFG(projected.body))
        return error;
    }
    if (!pendingDefinitions.empty())
      return unsupported("call closure contains an unprojected SSA definition");
    return validateProgramBindings(closure.entry);
  }

private:
  using Command = ReferenceProgram::Impl::Command;
  using CommandKind = ReferenceProgram::Impl::CommandKind;
  using ValueId = ReferenceProgram::Impl::ValueId;

  llvm::Expected<ValueId> define(mlir::Value value) {
    auto existing = values.find(value);
    if (existing != values.end()) {
      if (!pendingDefinitions.erase(value))
        return unsupported("SSA value is projected more than once");
      return existing->second;
    }
    if (nextValue == std::numeric_limits<ValueId>::max())
      return unsupported("reference value id space is exhausted");
    ValueId id = nextValue++;
    values[value] = id;
    return id;
  }

  llvm::Error predeclare(mlir::Value value) {
    if (values.count(value))
      return unsupported("SSA value is declared more than once");
    if (nextValue == std::numeric_limits<ValueId>::max())
      return unsupported("reference value id space is exhausted");
    values[value] = nextValue++;
    pendingDefinitions.insert(value);
    return llvm::Error::success();
  }

  llvm::Expected<ValueId> use(mlir::Value value) const {
    auto found = values.find(value);
    if (found == values.end())
      return unsupported("operand has no projected SSA definition");
    return found->second;
  }

  llvm::Expected<mlir::MemRefType> requireF32Buffer(mlir::Value value,
                                                    llvm::StringRef purpose) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!type || !type.getElementType().isF32() || !type.hasStaticShape())
      return unsupported((purpose + " requires a static f32 memref").str());
    if (!wafer::computeWaferPhysicalTensorInfo(type))
      return unsupported((purpose + " has no accepted physical layout").str());
    return type;
  }

  llvm::Error projectBlock(mlir::Block &block,
                           ReferenceProgram::Impl::BlockProgram &output) {
    bool sawTerminator = false;
    for (mlir::Operation &operation : block) {
      if (sawTerminator)
        return unsupported("operation follows a projected terminator");
      Command command;
      if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(operation)) {
        if (!alloc.getDynamicSizes().empty() ||
            !alloc.getSymbolOperands().empty())
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
        auto sourceInfo =
            sourceType ? wafer::computeWaferPhysicalTensorInfo(sourceType)
                       : std::nullopt;
        auto resultInfo =
            resultType ? wafer::computeWaferPhysicalTensorInfo(resultType)
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
                     mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(
                         operation)) {
        if (llvm::Error error = projectStaticView(
                reinterpret.getSource(), reinterpret.getResult(),
                reinterpret.getType(), command))
          return error;
      } else if (auto collapse =
                     mlir::dyn_cast<mlir::memref::CollapseShapeOp>(operation)) {
        auto sourceType =
            mlir::dyn_cast<mlir::MemRefType>(collapse.getSrc().getType());
        auto resultType = mlir::dyn_cast<mlir::MemRefType>(collapse.getType());
        if (!sourceType || !resultType ||
            !wafer::isWaferMemRefType(sourceType) ||
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
            (!ifOp.getElseRegion().empty() &&
             !ifOp.getElseRegion().hasOneBlock()))
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
            command.body->commands.back().inputs.size() !=
                forOp.getNumResults())
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
        auto function = callee ? functionIds.find(callee.getOperation())
                               : functionIds.end();
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
          return unsupported(
              "tile region requires unsupported multi-block body");
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
      } else if (auto rdma = mlir::dyn_cast<wafer::InstrRDMAOp>(operation)) {
        if (llvm::Error error = validateMovementByteCount(
                rdma.getByteCount(), rdma.getInnerBytes(),
                rdma.getSrcIterations()))
          return error;
        command.kind = CommandKind::RDMA;
        if (llvm::Error error = projectMovementOperands(
                rdma.getSource(), rdma.getDest(), command))
          return error;
        command.sourceStrides.assign(rdma.getSrcStrides().begin(),
                                     rdma.getSrcStrides().end());
        command.sourceIterations.assign(rdma.getSrcIterations().begin(),
                                        rdma.getSrcIterations().end());
        command.byteCount = rdma.getByteCount();
        command.innerBytes = rdma.getInnerBytes();
      } else if (auto wdma = mlir::dyn_cast<wafer::InstrWDMAOp>(operation)) {
        if (llvm::Error error = validateMovementByteCount(
                wdma.getByteCount(), wdma.getInnerBytes(),
                wdma.getDstIterations()))
          return error;
        command.kind = CommandKind::WDMA;
        if (llvm::Error error = projectMovementOperands(
                wdma.getSource(), wdma.getDest(), command))
          return error;
        command.destStrides.assign(wdma.getDstStrides().begin(),
                                   wdma.getDstStrides().end());
        command.destIterations.assign(wdma.getDstIterations().begin(),
                                      wdma.getDstIterations().end());
        command.byteCount = wdma.getByteCount();
        command.innerBytes = wdma.getInnerBytes();
      } else if (auto movement =
                     mlir::dyn_cast<wafer::InstrGatherScatterOp>(operation)) {
        if (llvm::Error error = validateMovementByteCount(
                movement.getByteCount(), movement.getInnerBytes(),
                movement.getSrcIterations()))
          return error;
        command.kind = CommandKind::GatherScatter;
        if (llvm::Error error = projectMovementOperands(
                movement.getSource(), movement.getDest(), command))
          return error;
        command.sourceStrides.assign(movement.getSrcStrides().begin(),
                                     movement.getSrcStrides().end());
        command.sourceIterations.assign(movement.getSrcIterations().begin(),
                                        movement.getSrcIterations().end());
        command.destStrides.assign(movement.getDstStrides().begin(),
                                   movement.getDstStrides().end());
        command.destIterations.assign(movement.getDstIterations().begin(),
                                      movement.getDstIterations().end());
        command.sourceOffset = movement.getSrcOffset();
        command.destOffset = movement.getDstOffset();
        command.byteCount = movement.getByteCount();
        command.innerBytes = movement.getInnerBytes();
      } else if (auto convert =
                     mlir::dyn_cast<wafer::InstrConvertOp>(operation)) {
        std::optional<ConvertSpec> spec =
            getConvertSpec(convert.getContext(), convert.getKind());
        if (!spec)
          return unsupported("unknown instruction convert kind");
        auto sourceType = convert.getSource().getType();
        auto destType = convert.getDest().getType();
        if (!sourceType.hasStaticShape() || !destType.hasStaticShape() ||
            !wafer::isWaferSPMMemRefType(sourceType) ||
            !wafer::isWaferSPMMemRefType(destType))
          return unsupported("convert requires static Wafer SPM memrefs");
        if (!matchesNumericFormat(sourceType.getElementType(), spec->source) ||
            !matchesNumericFormat(destType.getElementType(), spec->dest))
          return unsupported("convert kind disagrees with memref element type");
        if (sourceType.getNumElements() != destType.getNumElements())
          return unsupported(
              "convert source and destination element counts disagree");
        if (!wafer::computeWaferPhysicalTensorInfo(sourceType) ||
            !wafer::computeWaferPhysicalTensorInfo(destType))
          return unsupported("convert has no accepted physical layout");

        std::optional<uint64_t> zeroPoint = convert.getZeroPoint();
        std::optional<uint64_t> roundingMode = convert.getRoundingMode();
        switch (spec->parameter) {
        case wafer::InstrConvertParameterKind::None:
          if (zeroPoint || roundingMode)
            return unsupported("plain convert has unexpected parameters");
          break;
        case wafer::InstrConvertParameterKind::RoundingMode: {
          if (zeroPoint || !roundingMode)
            return unsupported(
                "rounding convert has invalid parameter combination");
          auto projectedRounding = projectRoundingMode(*roundingMode);
          if (!projectedRounding)
            return projectedRounding.takeError();
          command.roundingMode = *projectedRounding;
          if (*roundingMode == 4) {
            command.stochasticRounding = true;
            program.usesStochasticRounding = true;
          }
          break;
        }
        case wafer::InstrConvertParameterKind::ZeroPoint:
          if (!zeroPoint || roundingMode)
            return unsupported(
                "zero-point convert has invalid parameter combination");
          return unsupported(
              "INT8 zero-point convert formula is not evidenced");
        }

        auto source = use(convert.getSource());
        auto dest = use(convert.getDest());
        if (!source)
          return source.takeError();
        if (!dest)
          return dest.takeError();
        command.kind = CommandKind::Convert;
        command.source = *source;
        command.dest = *dest;
        command.sourceFormat = spec->source;
        command.destFormat = spec->dest;
      } else if (auto reduce =
                     mlir::dyn_cast<wafer::InstrReduceOp>(operation)) {
        (void)reduce;
        return unsupported(
            "target-native reduce has no compiler-owned source equivalence "
            "proof");
      } else if (auto gemm = mlir::dyn_cast<wafer::InstrGemmOp>(operation)) {
        auto lhsType = requireF32Buffer(gemm.getLhs(), "GEMM lhs");
        if (!lhsType)
          return lhsType.takeError();
        auto rhsType = requireF32Buffer(gemm.getRhs(), "GEMM rhs");
        if (!rhsType)
          return rhsType.takeError();
        auto destType = requireF32Buffer(gemm.getDest(), "GEMM dest");
        if (!destType)
          return destType.takeError();
        int64_t m = static_cast<int64_t>(gemm.getM());
        int64_t n = static_cast<int64_t>(gemm.getN());
        int64_t k = static_cast<int64_t>(gemm.getK());
        if (lhsType->getRank() != rhsType->getRank() ||
            lhsType->getRank() != destType->getRank() || lhsType->getRank() < 2)
          return invalid("GEMM accepted buffer ranks disagree");
        int64_t rank = lhsType->getRank();
        if (lhsType->getDimSize(rank - 2) != m ||
            lhsType->getDimSize(rank - 1) != k ||
            rhsType->getDimSize(rank - 2) != k ||
            rhsType->getDimSize(rank - 1) != n ||
            destType->getDimSize(rank - 2) != m ||
            destType->getDimSize(rank - 1) != n)
          return invalid("GEMM dimensions disagree with accepted buffers");
        command.batchShape.assign(destType->getShape().begin(),
                                  destType->getShape().end() - 2);
        if (lhsType->getShape().drop_back(2) !=
                llvm::ArrayRef<int64_t>(command.batchShape) ||
            rhsType->getShape().drop_back(2) !=
                llvm::ArrayRef<int64_t>(command.batchShape))
          return invalid("GEMM accepted batch shapes disagree");
        if (rank > 2) {
          llvm::SmallVector<int64_t> expectedBatchDims;
          for (int64_t dimension = 0; dimension < rank - 2; ++dimension)
            expectedBatchDims.push_back(dimension);
          auto hasBatchDims = [&](llvm::StringRef name) {
            auto attr = gemm->getAttrOfType<mlir::DenseI64ArrayAttr>(name);
            return attr && attr.asArrayRef() ==
                               llvm::ArrayRef<int64_t>(expectedBatchDims);
          };
          auto hasDim = [&](llvm::StringRef name, int64_t expected) {
            auto attr = gemm->getAttrOfType<mlir::IntegerAttr>(name);
            return attr && attr.getInt() == expected;
          };
          int64_t batchCount = 1;
          for (int64_t size : command.batchShape) {
            if (size <= 0 ||
                batchCount > std::numeric_limits<int64_t>::max() / size)
              return invalid("GEMM accepted batch count overflows");
            batchCount *= size;
          }
          if (!hasBatchDims("lhs_batch_dims") ||
              !hasBatchDims("rhs_batch_dims") ||
              !hasBatchDims("result_batch_dims") ||
              !hasDim("lhs_m_dim", rank - 2) ||
              !hasDim("lhs_contracting_dim", rank - 1) ||
              !hasDim("rhs_contracting_dim", rank - 2) ||
              !hasDim("rhs_n_dim", rank - 1) ||
              !hasDim("result_m_dim", rank - 2) ||
              !hasDim("result_n_dim", rank - 1) ||
              !hasDim("batch_count", batchCount))
            return invalid("GEMM batched dimension attrs are not canonical");
        } else if (gemm.getBatchCount()) {
          return invalid("rank-2 GEMM unexpectedly carries batch attrs");
        }
        command.kind = CommandKind::Gemm;
        auto lhs = use(gemm.getLhs());
        auto rhs = use(gemm.getRhs());
        auto dest = use(gemm.getDest());
        if (!lhs)
          return lhs.takeError();
        if (!rhs)
          return rhs.takeError();
        if (!dest)
          return dest.takeError();
        command.lhs = *lhs;
        command.rhs = *rhs;
        command.dest = *dest;
        command.m = m;
        command.n = n;
        command.k = k;
      } else if (auto elementwise =
                     mlir::dyn_cast<wafer::InstrElementwiseOp>(operation)) {
        size_t arity = 0;
        if (!isSupportedElementwiseKind(elementwise.getKind(), arity) ||
            elementwise.getInputs().size() != arity)
          return unsupported("elementwise kind or arity");
        auto destType =
            requireF32Buffer(elementwise.getDest(), "elementwise dest");
        if (!destType)
          return destType.takeError();
        command.kind = CommandKind::Elementwise;
        command.elementwiseKind = elementwise.getKind();
        auto dest = use(elementwise.getDest());
        if (!dest)
          return dest.takeError();
        command.dest = *dest;
        for (mlir::Value value : elementwise.getInputs()) {
          auto inputType = requireF32Buffer(value, "elementwise input");
          if (!inputType)
            return inputType.takeError();
          if (inputType->getShape() != destType->getShape())
            return invalid(
                "elementwise input shape disagrees with destination");
          auto input = use(value);
          if (!input)
            return input.takeError();
          command.inputs.push_back(*input);
        }
      } else if (auto bit2fp =
                     mlir::dyn_cast<wafer::InstrBit2FpOp>(operation)) {
        auto sourceType =
            mlir::dyn_cast<mlir::MemRefType>(bit2fp.getSource().getType());
        auto destType = requireF32Buffer(bit2fp.getDest(), "bit2fp dest");
        auto sourceInfo =
            sourceType ? wafer::computeWaferPhysicalTensorInfo(sourceType)
                       : std::nullopt;
        if (!sourceType || !sourceType.hasStaticShape() ||
            !sourceType.getElementType().isInteger(1) ||
            !wafer::isWaferSPMMemRefType(sourceType) || !sourceInfo ||
            sourceInfo->layout != wafer::MemLayout::Tensor || !destType)
          return unsupported(
              "bit2fp requires static tensor-layout i1 and f32 SPM memrefs");
        if (sourceType.getShape() != destType->getShape())
          return invalid("bit2fp source and destination shapes disagree");
        auto source = use(bit2fp.getSource());
        auto dest = use(bit2fp.getDest());
        if (!source)
          return source.takeError();
        if (!dest)
          return dest.takeError();
        command.kind = CommandKind::Bit2Fp;
        command.source = *source;
        command.dest = *dest;
      } else if (auto maskMove =
                     mlir::dyn_cast<wafer::InstrMaskMoveOp>(operation)) {
        auto sourceType =
            requireF32Buffer(maskMove.getSource(), "mask_move source");
        auto maskType = requireF32Buffer(maskMove.getMask(), "mask_move mask");
        auto destType = requireF32Buffer(maskMove.getDest(), "mask_move dest");
        if (!sourceType)
          return sourceType.takeError();
        if (!maskType)
          return maskType.takeError();
        if (!destType)
          return destType.takeError();
        if (sourceType->getShape() != destType->getShape() ||
            maskType->getShape() != destType->getShape())
          return invalid("mask_move buffer shapes disagree");
        auto source = use(maskMove.getSource());
        auto mask = use(maskMove.getMask());
        auto dest = use(maskMove.getDest());
        if (!source)
          return source.takeError();
        if (!mask)
          return mask.takeError();
        if (!dest)
          return dest.takeError();
        command.kind = CommandKind::MaskMove;
        command.source = *source;
        command.mask = *mask;
        command.dest = *dest;
      } else if (auto fill = mlir::dyn_cast<wafer::InstrFillOp>(operation)) {
        auto destType =
            mlir::dyn_cast<mlir::MemRefType>(fill.getDest().getType());
        auto destInfo = destType
                            ? wafer::computeWaferPhysicalTensorInfo(destType)
                            : std::nullopt;
        if (!destType || !destType.hasStaticShape() ||
            !wafer::isWaferSPMMemRefType(destType) || !destInfo ||
            (!destType.getElementType().isF32() &&
             !destType.getElementType().isInteger(1)))
          return unsupported("fill requires a static f32 or i1 SPM memref");
        if (destType.getElementType().isInteger(1) &&
            destInfo->layout != wafer::MemLayout::Tensor)
          return unsupported("i1 fill requires tensor-layout storage");
        auto dest = use(fill.getDest());
        auto scalar = use(fill.getValue());
        if (!dest)
          return dest.takeError();
        if (!scalar)
          return scalar.takeError();
        command.kind = CommandKind::Fill;
        command.dest = *dest;
        command.scalar = *scalar;
      } else if (auto send = mlir::dyn_cast<wafer::InstrDTESendOp>(operation)) {
        if (llvm::Error error = projectDTEIssue(
                send.getBuffer(), send.getToken(), send.getPeer(),
                send.getBytes(), send.getMessage(), send.getBinding(),
                /*isSend=*/true, command))
          return error;
      } else if (auto recv = mlir::dyn_cast<wafer::InstrDTERecvOp>(operation)) {
        if (llvm::Error error = projectDTEIssue(
                recv.getBuffer(), recv.getToken(), recv.getPeer(),
                recv.getBytes(), recv.getMessage(), recv.getBinding(),
                /*isSend=*/false, command))
          return error;
      } else if (auto wait = mlir::dyn_cast<wafer::InstrDTEWaitOp>(operation)) {
        if (wait.getTokens().empty())
          return unsupported("Direct DTE wait has no tokens");
        command.kind = CommandKind::DTEWait;
        for (mlir::Value token : wait.getTokens()) {
          auto id = use(token);
          if (!id)
            return id.takeError();
          command.inputs.push_back(*id);
        }
      } else if (mlir::isa<wafer::SyncLocalFenceOp>(operation)) {
        command.kind = CommandKind::LocalFence;
      } else if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(operation)) {
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
      } else if (auto branch =
                     mlir::dyn_cast<mlir::cf::CondBranchOp>(operation)) {
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
          return unsupported(
              "cf.cond_br successor operands disagree with block");
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
        return unsupported(
            ("unsupported operation " + operation.getName().getStringRef())
                .str());
      }
      ++program.projectedOperationCount;
      output.commands.push_back(std::move(command));
    }
    if (!sawTerminator)
      return unsupported("block has no return-like terminator");
    return llvm::Error::success();
  }

  llvm::Error projectMovementOperands(mlir::Value sourceValue,
                                      mlir::Value destValue, Command &command) {
    auto source = use(sourceValue);
    auto dest = use(destValue);
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    command.source = *source;
    command.dest = *dest;
    return llvm::Error::success();
  }

  llvm::Error
  projectDTEIssue(mlir::Value bufferValue, mlir::Value tokenValue, int64_t peer,
                  int64_t bytes, wafer::DTEMessageAttr message,
                  std::optional<wafer::DirectDTEBindingAttr> binding,
                  bool isSend, Command &command) {
    auto bufferType = mlir::dyn_cast<mlir::MemRefType>(bufferValue.getType());
    auto info = bufferType ? wafer::computeWaferPhysicalTensorInfo(bufferType)
                           : std::nullopt;
    if (!bufferType || !wafer::isWaferSPMMemRefType(bufferType) || !info ||
        info->physicalBytes < 0)
      return unsupported("Direct DTE issue requires static Wafer SPM memref");
    if (peer < 0 || bytes <= 0 || bytes > info->physicalBytes)
      return unsupported("Direct DTE peer or byte range is invalid");
    if (!binding ||
        binding->getAllocationProfile() !=
            wafer::DTEAllocationProfile::Normal ||
        binding->getCompletionProfile() !=
            wafer::DTECompletionProfile::SenderWaitReceiverFSM ||
        binding->getReceiverFsmId() < 0 ||
        binding->getRemoteReceiverOffset() < 0)
      return unsupported("Direct DTE issue has no supported accepted binding");
    auto buffer = use(bufferValue);
    auto token = define(tokenValue);
    if (!buffer)
      return buffer.takeError();
    if (!token)
      return token.takeError();
    command.kind = isSend ? CommandKind::DTESend : CommandKind::DTERecv;
    command.source = *buffer;
    command.result = *token;
    command.peer = peer;
    command.byteCount = static_cast<uint64_t>(bytes);
    command.messageCommunication = message.getCommunicationId();
    command.messagePhase = message.getPhase();
    command.messageRound = message.getRound();
    command.messagePayloadSlice = message.getPayloadSlice();
    command.remoteReceiverOffset = binding->getRemoteReceiverOffset();
    return llvm::Error::success();
  }

  llvm::Error projectStaticView(mlir::Value sourceValue,
                                mlir::Value resultValue,
                                mlir::MemRefType resultType, Command &command) {
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

  llvm::Error validateProgramBindings(mlir::func::FuncOp entry) {
    llvm::SmallVector<bool> argumentUsed(entry.getNumArguments(), false);
    llvm::SmallVector<bool> resultUsed(entry.getNumResults(), false);
    llvm::SmallVector<int64_t> userInputIndices;
    llvm::SmallVector<int64_t> outputIndices;
    for (const RankProgramBinding &binding : program.programBindings) {
      llvm::SmallVectorImpl<bool> &domain =
          binding.role == ProgramResourceRole::Output ? resultUsed
                                                      : argumentUsed;
      if (binding.index < 0 ||
          binding.index >= static_cast<int64_t>(domain.size()))
        return unsupported("program binding index is outside entry signature");
      if (domain[binding.index])
        return unsupported("duplicate program binding index");
      domain[binding.index] = true;
      mlir::Type type = binding.role == ProgramResourceRole::Output
                            ? entry.getResultTypes()[binding.index]
                            : entry.getArgument(binding.index).getType();
      auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
      if (!memref || elementDType(memref.getElementType()) != binding.dtype ||
          memref.getShape() != llvm::ArrayRef<int64_t>(binding.localShape))
        return unsupported(
            "program binding disagrees with entry memref signature");
      if (binding.role == ProgramResourceRole::UserInput)
        userInputIndices.push_back(binding.programIndex);
      else if (binding.role == ProgramResourceRole::Output)
        outputIndices.push_back(binding.programIndex);
    }
    if (llvm::is_contained(argumentUsed, false) ||
        llvm::is_contained(resultUsed, false))
      return unsupported("program bindings do not cover entry signature");
    auto validateProgramDomain = [](llvm::SmallVectorImpl<int64_t> &indices,
                                    llvm::StringRef message) -> llvm::Error {
      llvm::sort(indices);
      for (auto [expected, index] : llvm::enumerate(indices))
        if (index != static_cast<int64_t>(expected))
          return unsupported(message);
      return llvm::Error::success();
    };
    if (llvm::Error error = validateProgramDomain(
            userInputIndices,
            "program input indices are not unique and contiguous"))
      return error;
    if (llvm::Error error = validateProgramDomain(
            outputIndices,
            "program output indices are not unique and contiguous"))
      return error;
    return llvm::Error::success();
  }

  llvm::Error
  validateCFG(const ReferenceProgram::Impl::ControlFlowProgram &controlFlow) {
    llvm::SmallVector<uint8_t> state(controlFlow.blocks.size(), 0);
    auto visit = [&](auto &&self, uint32_t index) -> llvm::Error {
      if (index >= controlFlow.blocks.size())
        return unsupported("CFG successor is outside projected block graph");
      if (state[index] == 1)
        return unsupported("cyclic CFG is unsupported; use structured scf.for");
      if (state[index] == 2)
        return llvm::Error::success();
      state[index] = 1;
      const auto &block = controlFlow.blocks[index];
      if (block.commands.empty())
        return unsupported("CFG block has no projected terminator");
      const Command &terminator = block.commands.back();
      auto visitSuccessor = [&](uint32_t successor,
                                llvm::ArrayRef<ValueId> inputs) -> llvm::Error {
        if (successor >= controlFlow.blocks.size())
          return unsupported("CFG successor is outside projected block graph");
        if (inputs.size() != controlFlow.blocks[successor].arguments.size())
          return unsupported("CFG successor operand arity mismatch");
        return self(self, successor);
      };
      if (terminator.kind == CommandKind::Branch) {
        if (llvm::Error error =
                visitSuccessor(terminator.successor, terminator.inputs))
          return error;
      } else if (terminator.kind == CommandKind::CondBranch) {
        if (llvm::Error error =
                visitSuccessor(terminator.trueSuccessor, terminator.trueInputs))
          return error;
        if (llvm::Error error = visitSuccessor(terminator.falseSuccessor,
                                               terminator.falseInputs))
          return error;
      } else if (terminator.kind != CommandKind::Return) {
        return unsupported("CFG block has no branch or return terminator");
      }
      state[index] = 2;
      return llvm::Error::success();
    };
    for (uint32_t index = 0; index < controlFlow.blocks.size(); ++index)
      if (llvm::Error error = visit(visit, index))
        return error;
    return llvm::Error::success();
  }

  ReferenceProgram::Impl &program;
  llvm::DenseMap<mlir::Value, ValueId> values;
  llvm::DenseSet<mlir::Value> pendingDefinitions;
  llvm::DenseMap<mlir::Block *, uint32_t> blocks;
  llvm::DenseMap<mlir::Operation *, uint32_t> functionIds;
  ValueId nextValue = 0;
};

} // namespace

llvm::Error projectReferenceProgram(ReferenceProgram::Impl &program,
                                    mlir::ModuleOp module,
                                    llvm::StringRef entrySymbol) {
  llvm::Expected<detail::AcceptedCallClosure> closure =
      detail::analyzeAcceptedCallClosure(module, entrySymbol);
  if (!closure) {
    std::string message = "accepted call closure is invalid: " +
                          llvm::toString(closure.takeError());
    return unsupported(message);
  }
  return ProgramProjector(program).project(*closure);
}

} // namespace wafer::compiler::reference_detail
