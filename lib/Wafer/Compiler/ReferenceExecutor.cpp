//===- ReferenceExecutor.cpp - Accepted-rank semantic execution ----------===//

#include "Wafer/Compiler/ReferenceExecutor.h"

#include "AcceptedCallClosure.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wafer::compiler {

struct ReferenceExecutionResultBuilder {
  static ReferenceExecutionResult
  make(int64_t logicalRank, std::vector<ReferenceOutputBinding> outputs) {
    return ReferenceExecutionResult(logicalRank, std::move(outputs));
  }
};

struct ReferenceProgram::Impl {
  using ValueId = uint32_t;

  enum class CommandKind {
    Alloc,
    Dealloc,
    Cast,
    Constant,
    TileRegion,
    If,
    For,
    Call,
    RDMA,
    WDMA,
    GatherScatter,
    Convert,
    Gemm,
    Elementwise,
    Fill,
    LocalFence,
    Branch,
    CondBranch,
    Return,
  };

  enum class NumericFormat {
    Int8,
    Int16,
    Int32,
    BFloat16,
    Float16,
    Float32,
    TF32
  };

  struct Scalar {
    std::optional<llvm::APInt> integer;
    std::optional<llvm::APFloat> floating;
    bool integerIsUnsigned = false;
  };

  struct BlockProgram;

  struct ControlFlowProgram;

  struct Command {
    CommandKind kind = CommandKind::LocalFence;
    ValueId result = 0;
    ValueId source = 0;
    ValueId dest = 0;
    ValueId lhs = 0;
    ValueId rhs = 0;
    ValueId scalar = 0;
    std::vector<ValueId> inputs;
    std::vector<ValueId> results;
    std::vector<ValueId> blockArguments;
    std::shared_ptr<BlockProgram> body;
    std::shared_ptr<BlockProgram> elseBody;
    ValueId condition = 0;
    ValueId lowerBound = 0;
    ValueId upperBound = 0;
    ValueId step = 0;
    ValueId inductionArgument = 0;
    std::vector<ValueId> iterInputs;
    std::vector<ValueId> iterArguments;
    std::vector<ValueId> trueInputs;
    std::vector<ValueId> falseInputs;
    uint32_t successor = 0;
    uint32_t trueSuccessor = 0;
    uint32_t falseSuccessor = 0;
    uint32_t callee = 0;
    mlir::MemRefType type;
    int64_t physicalBytes = 0;
    int64_t viewDelta = 0;
    int64_t acceptedOffset = 0;
    bool spmAllocation = false;
    Scalar scalarValue;
    std::vector<int64_t> sourceStrides;
    std::vector<int64_t> sourceIterations;
    std::vector<int64_t> destStrides;
    std::vector<int64_t> destIterations;
    std::optional<int64_t> sourceOffset;
    std::optional<int64_t> destOffset;
    uint64_t byteCount = 0;
    uint64_t innerBytes = 0;
    NumericFormat sourceFormat = NumericFormat::Float32;
    NumericFormat destFormat = NumericFormat::Float32;
    llvm::APFloat::roundingMode roundingMode =
        llvm::APFloat::rmNearestTiesToEven;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    wafer::InstrElementwiseKind elementwiseKind =
        wafer::InstrElementwiseKind::Abs;
  };

  struct BlockProgram {
    std::vector<ValueId> arguments;
    std::vector<Command> commands;
  };

  struct ControlFlowProgram {
    std::vector<BlockProgram> blocks;
  };

  struct FunctionProgram {
    std::vector<ValueId> arguments;
    std::vector<mlir::MemRefType> argumentTypes;
    ControlFlowProgram body;
  };

  std::shared_ptr<mlir::MLIRContext> contextOwner;
  int64_t logicalRank = 0;
  std::vector<RankProgramBinding> programBindings;
  std::vector<FunctionProgram> functions;
  uint32_t entryFunction = 0;
  size_t projectedOperationCount = 0;
};

namespace {

llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

llvm::Error unsupported(llvm::StringRef message) {
  return invalid(("reference capability preflight failed: " + message).str());
}

std::optional<int64_t> getDTypeByteWidth(llvm::StringRef dtype) {
  if (dtype == "i1")
    return std::nullopt;
  if (dtype == "i8" || dtype == "ui8")
    return 1;
  if (dtype == "i16" || dtype == "ui16" || dtype == "f16" || dtype == "bf16")
    return 2;
  if (dtype == "i32" || dtype == "ui32" || dtype == "f32" || dtype == "tf32")
    return 4;
  if (dtype == "i64" || dtype == "ui64" || dtype == "f64")
    return 8;
  return std::nullopt;
}

std::optional<int64_t> getCompactByteCount(llvm::StringRef dtype,
                                           llvm::ArrayRef<int64_t> shape) {
  std::optional<int64_t> elementBytes = getDTypeByteWidth(dtype);
  if (!elementBytes)
    return std::nullopt;
  int64_t bytes = *elementBytes;
  for (int64_t dim : shape) {
    if (dim < 0 ||
        (dim != 0 && bytes > std::numeric_limits<int64_t>::max() / dim))
      return std::nullopt;
    bytes *= dim;
  }
  return bytes;
}

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

std::string elementDType(mlir::Type type) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return (integer.isUnsigned() ? "ui" : "i") +
           std::to_string(integer.getWidth());
  return {};
}

template <typename Callback>
llvm::Error forEachLogicalIndex(llvm::ArrayRef<int64_t> shape,
                                Callback callback) {
  for (int64_t dim : shape)
    if (dim < 0)
      return invalid("reference executor requires static tensor shapes");
  if (llvm::is_contained(shape, int64_t{0}))
    return llvm::Error::success();
  llvm::SmallVector<int64_t> index(shape.size(), 0);
  if (shape.empty())
    return callback(index);
  while (true) {
    if (llvm::Error error = callback(index))
      return error;
    int64_t dim = static_cast<int64_t>(shape.size()) - 1;
    for (; dim >= 0; --dim) {
      if (++index[dim] < shape[dim])
        break;
      index[dim] = 0;
    }
    if (dim < 0)
      break;
  }
  return llvm::Error::success();
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

using NumericFormat = ReferenceProgram::Impl::NumericFormat;

struct ConvertSpec {
  NumericFormat source;
  NumericFormat dest;
  wafer::InstrConvertParameterKind parameter;
};

std::optional<NumericFormat> getNumericFormat(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (!integer.isSignless())
      return std::nullopt;
    if (integer.getWidth() == 8)
      return NumericFormat::Int8;
    if (integer.getWidth() == 16)
      return NumericFormat::Int16;
    if (integer.getWidth() == 32)
      return NumericFormat::Int32;
    return std::nullopt;
  }
  if (type.isBF16())
    return NumericFormat::BFloat16;
  if (type.isF16())
    return NumericFormat::Float16;
  if (mlir::isa<mlir::FloatTF32Type>(type))
    return NumericFormat::TF32;
  if (type.isF32())
    return NumericFormat::Float32;
  return std::nullopt;
}

std::optional<ConvertSpec> getConvertSpec(mlir::MLIRContext *context,
                                          wafer::InstrConvertKind kind) {
  auto [sourceType, destType] = wafer::getInstrConvertTypePair(context, kind);
  std::optional<NumericFormat> source = getNumericFormat(sourceType);
  std::optional<NumericFormat> dest = getNumericFormat(destType);
  if (!source || !dest)
    return std::nullopt;
  return ConvertSpec{*source, *dest, wafer::getInstrConvertParameterKind(kind)};
}

bool matchesNumericFormat(mlir::Type type, NumericFormat format) {
  return getNumericFormat(type) == format;
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
    return unsupported(
        "stochastic convert rounding has no deterministic reference policy");
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
      } else if (auto gemm = mlir::dyn_cast<wafer::InstrGemmOp>(operation)) {
        if (gemm.getBatchCount().value_or(1) != 1)
          return unsupported("batched GEMM");
        auto lhsType = requireF32Buffer(gemm.getLhs(), "GEMM lhs");
        auto rhsType = requireF32Buffer(gemm.getRhs(), "GEMM rhs");
        auto destType = requireF32Buffer(gemm.getDest(), "GEMM dest");
        if (!lhsType)
          return lhsType.takeError();
        if (!rhsType)
          return rhsType.takeError();
        if (!destType)
          return destType.takeError();
        int64_t m = static_cast<int64_t>(gemm.getM());
        int64_t n = static_cast<int64_t>(gemm.getN());
        int64_t k = static_cast<int64_t>(gemm.getK());
        if (lhsType->getShape() != llvm::ArrayRef<int64_t>({m, k}) ||
            rhsType->getShape() != llvm::ArrayRef<int64_t>({k, n}) ||
            destType->getShape() != llvm::ArrayRef<int64_t>({m, n}))
          return invalid(
              "GEMM dimensions disagree with accepted buffer shapes");
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
      } else if (auto fill = mlir::dyn_cast<wafer::InstrFillOp>(operation)) {
        auto destType = requireF32Buffer(fill.getDest(), "fill dest");
        if (!destType)
          return destType.takeError();
        auto dest = use(fill.getDest());
        auto scalar = use(fill.getValue());
        if (!dest)
          return dest.takeError();
        if (!scalar)
          return scalar.takeError();
        command.kind = CommandKind::Fill;
        command.dest = *dest;
        command.scalar = *scalar;
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
    }
    if (llvm::is_contained(argumentUsed, false) ||
        llvm::is_contained(resultUsed, false))
      return unsupported("program bindings do not cover entry signature");
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

struct Storage {
  std::vector<uint8_t> bytes;
};

struct BufferView {
  std::shared_ptr<Storage> storage;
  int64_t base = 0;
  int64_t viewOffset = 0;
  int64_t physicalBytes = 0;
  mlir::MemRefType type;
};

struct NumericValue {
  std::optional<llvm::APInt> integer;
  std::optional<llvm::APFloat> floating;
};

unsigned getNumericBitWidth(NumericFormat format) {
  switch (format) {
  case NumericFormat::Int8:
    return 8;
  case NumericFormat::Int16:
  case NumericFormat::BFloat16:
  case NumericFormat::Float16:
    return 16;
  case NumericFormat::Int32:
  case NumericFormat::Float32:
  case NumericFormat::TF32:
    return 32;
  }
  llvm_unreachable("unknown reference numeric format");
}

const llvm::fltSemantics *getFloatSemantics(NumericFormat format) {
  switch (format) {
  case NumericFormat::BFloat16:
    return &llvm::APFloat::BFloat();
  case NumericFormat::Float16:
    return &llvm::APFloat::IEEEhalf();
  case NumericFormat::Float32:
    return &llvm::APFloat::IEEEsingle();
  case NumericFormat::TF32:
    return &llvm::APFloat::FloatTF32();
  case NumericFormat::Int8:
  case NumericFormat::Int16:
  case NumericFormat::Int32:
    return nullptr;
  }
  llvm_unreachable("unknown reference numeric format");
}

llvm::Expected<int64_t> getAbsoluteOffset(const BufferView &buffer,
                                          llvm::ArrayRef<int64_t> indices,
                                          int64_t byteWidth) {
  std::optional<int64_t> relative =
      wafer::computeWaferPhysicalElementByteOffset(buffer.type, indices);
  if (!relative)
    return invalid("cannot map logical index to accepted physical layout");
  if (*relative < 0 || byteWidth < 0 ||
      *relative > buffer.physicalBytes - byteWidth)
    return invalid("logical access exceeds accepted buffer extent");
  int64_t absolute = buffer.base + buffer.viewOffset + *relative;
  if (absolute < 0 ||
      absolute > static_cast<int64_t>(buffer.storage->bytes.size()) - byteWidth)
    return invalid("logical access exceeds reference storage arena");
  return absolute;
}

llvm::Expected<NumericValue> readNumeric(const BufferView &buffer,
                                         llvm::ArrayRef<int64_t> indices,
                                         NumericFormat format) {
  if (!matchesNumericFormat(buffer.type.getElementType(), format))
    return invalid("projected convert source format disagrees with buffer");
  unsigned bitWidth = getNumericBitWidth(format);
  int64_t byteWidth = bitWidth / 8;
  auto offset = getAbsoluteOffset(buffer, indices, byteWidth);
  if (!offset)
    return offset.takeError();
  llvm::APInt bits(bitWidth, 0);
  for (int64_t byte = 0; byte < byteWidth; ++byte)
    bits |= llvm::APInt(bitWidth, buffer.storage->bytes[*offset + byte])
            << (byte * 8);
  if (const llvm::fltSemantics *semantics = getFloatSemantics(format)) {
    if (format == NumericFormat::TF32) {
      uint64_t storage = bits.getZExtValue();
      uint64_t compact = ((storage >> 31) << 18) |
                         (((storage >> 23) & 0xff) << 10) |
                         ((storage >> 13) & 0x3ff);
      bits = llvm::APInt(/*numBits=*/19, compact);
    }
    return NumericValue{std::nullopt, llvm::APFloat(*semantics, bits)};
  }
  return NumericValue{std::move(bits), std::nullopt};
}

llvm::Expected<llvm::APInt>
convertNumeric(const NumericValue &source, NumericFormat destFormat,
               llvm::APFloat::roundingMode roundingMode) {
  const llvm::fltSemantics *destSemantics = getFloatSemantics(destFormat);
  if (source.integer && destSemantics) {
    llvm::APFloat result = llvm::APFloat::getZero(*destSemantics);
    result.convertFromAPInt(*source.integer, /*IsSigned=*/true, roundingMode);
    return result.bitcastToAPInt();
  }
  if (source.floating && destSemantics) {
    llvm::APFloat result = *source.floating;
    bool losesInfo = false;
    result.convert(*destSemantics, roundingMode, &losesInfo);
    return result.bitcastToAPInt();
  }
  if (source.floating && !destSemantics) {
    llvm::APSInt result(getNumericBitWidth(destFormat), /*isUnsigned=*/false);
    bool isExact = false;
    llvm::APFloat::opStatus status =
        source.floating->convertToInteger(result, roundingMode, &isExact);
    if ((status & llvm::APFloat::opInvalidOp) != 0)
      return invalid(
          "floating-to-integer convert input is NaN, Inf, or out of range");
    return llvm::APInt(result);
  }
  return invalid("projected convert has unsupported integer-to-integer pair");
}

llvm::Error writeNumericBits(const BufferView &buffer,
                             llvm::ArrayRef<int64_t> indices,
                             NumericFormat format, const llvm::APInt &bits) {
  if (!matchesNumericFormat(buffer.type.getElementType(), format))
    return invalid(
        "projected convert destination format disagrees with buffer");
  llvm::APInt storageBits = bits;
  if (format == NumericFormat::TF32) {
    if (bits.getBitWidth() != 19)
      return invalid(
          "projected TF32 convert produced a non-TF32 semantic width");
    uint64_t compact = bits.getZExtValue();
    uint64_t storage = ((compact >> 18) << 31) |
                       (((compact >> 10) & 0xff) << 23) |
                       ((compact & 0x3ff) << 13);
    storageBits = llvm::APInt(/*numBits=*/32, storage);
  } else if (bits.getBitWidth() != getNumericBitWidth(format)) {
    return invalid(
        "projected convert destination format disagrees with buffer");
  }
  int64_t byteWidth = storageBits.getBitWidth() / 8;
  auto offset = getAbsoluteOffset(buffer, indices, byteWidth);
  if (!offset)
    return offset.takeError();
  for (int64_t byte = 0; byte < byteWidth; ++byte)
    buffer.storage->bytes[*offset + byte] =
        static_cast<uint8_t>(storageBits.extractBitsAsZExtValue(8, byte * 8));
  return llvm::Error::success();
}

llvm::Expected<float> readF32(const BufferView &buffer,
                              llvm::ArrayRef<int64_t> indices) {
  if (!buffer.type.getElementType().isF32())
    return invalid("reference floating compute currently requires f32");
  auto offset = getAbsoluteOffset(buffer, indices, sizeof(float));
  if (!offset)
    return offset.takeError();
  float value;
  std::memcpy(&value, buffer.storage->bytes.data() + *offset, sizeof(value));
  return value;
}

llvm::Error writeF32(const BufferView &buffer, llvm::ArrayRef<int64_t> indices,
                     float value) {
  if (!buffer.type.getElementType().isF32())
    return invalid("reference floating compute currently requires f32");
  auto offset = getAbsoluteOffset(buffer, indices, sizeof(float));
  if (!offset)
    return offset.takeError();
  std::memcpy(buffer.storage->bytes.data() + *offset, &value, sizeof(value));
  return llvm::Error::success();
}

class ProgramInterpreter {
public:
  explicit ProgramInterpreter(const ReferenceProgram::Impl &program)
      : program(program), spmArena(std::make_shared<Storage>()),
        ddrArena(std::make_shared<Storage>()) {}

  llvm::Expected<ReferenceExecutionResult>
  run(llvm::ArrayRef<ReferenceInputBinding> inputs) {
    if (program.entryFunction >= program.functions.size())
      return invalid("projected entry function is outside the function graph");
    if (llvm::Error error = bindEntryArguments(inputs))
      return std::move(error);
    auto returned =
        executeControlFlow(program.functions[program.entryFunction].body);
    if (!returned)
      return returned.takeError();

    std::vector<ReferenceOutputBinding> outputs;
    for (const RankProgramBinding &binding : program.programBindings) {
      if (binding.role != ProgramResourceRole::Output)
        continue;
      if (binding.index < 0 ||
          binding.index >= static_cast<int64_t>(returned->size()))
        return invalid("output binding index is outside entry results");
      auto buffer = lookup((*returned)[binding.index]);
      if (!buffer)
        return buffer.takeError();
      auto tensor = exportTensor(*buffer, binding.dtype, binding.localShape);
      if (!tensor)
        return tensor.takeError();
      outputs.push_back({binding.index, std::move(*tensor)});
    }
    return ReferenceExecutionResultBuilder::make(program.logicalRank,
                                                 std::move(outputs));
  }

private:
  using Command = ReferenceProgram::Impl::Command;
  using CommandKind = ReferenceProgram::Impl::CommandKind;
  using Scalar = ReferenceProgram::Impl::Scalar;
  using ValueId = ReferenceProgram::Impl::ValueId;

  struct RuntimeValue {
    std::optional<BufferView> buffer;
    std::optional<Scalar> scalar;
  };

  enum class TransferKind { Return, Branch };

  struct ControlTransfer {
    TransferKind kind = TransferKind::Return;
    uint32_t successor = 0;
    std::vector<ValueId> inputs;
  };

  llvm::Error bindEntryArguments(llvm::ArrayRef<ReferenceInputBinding> inputs) {
    const auto &entry = program.functions[program.entryFunction];
    llvm::SmallVector<bool> used(inputs.size(), false);
    for (const RankProgramBinding &binding : program.programBindings) {
      if (binding.role == ProgramResourceRole::Output)
        continue;
      int64_t match = -1;
      for (auto [position, input] : llvm::enumerate(inputs)) {
        if (input.role == binding.role && input.index == binding.index) {
          if (match >= 0)
            return invalid("duplicate reference input binding");
          match = static_cast<int64_t>(position);
        }
      }
      if (match < 0)
        return invalid("missing reference input binding");
      used[match] = true;
      if (inputs[match].tensor.getDType() != binding.dtype ||
          inputs[match].tensor.getShape() !=
              llvm::ArrayRef<int64_t>(binding.localShape))
        return invalid(
            "reference input tensor disagrees with typed rank binding");
      auto imported = importTensor(inputs[match].tensor,
                                   entry.argumentTypes[binding.index]);
      if (!imported)
        return imported.takeError();
      buffers[entry.arguments[binding.index]] = std::move(*imported);
    }
    if (llvm::is_contained(used, false))
      return invalid("unexpected reference input binding");
    return llvm::Error::success();
  }

  llvm::Expected<BufferView> importTensor(const ReferenceTensor &tensor,
                                          mlir::MemRefType type) {
    auto info = wafer::computeWaferPhysicalTensorInfo(type);
    if (!info || info->physicalBytes < 0 || info->elementBytes <= 0)
      return invalid("input has no static accepted physical layout");
    if (elementDType(type.getElementType()) != tensor.getDType() ||
        type.getShape() != tensor.getShape())
      return invalid("input tensor type disagrees with entry memref");
    BufferView view{std::make_shared<Storage>(), 0, 0, info->physicalBytes,
                    type};
    view.storage->bytes.resize(info->physicalBytes, 0);
    int64_t linearByte = 0;
    if (llvm::Error error = forEachLogicalIndex(
            type.getShape(), [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
              auto offset = getAbsoluteOffset(view, index, info->elementBytes);
              if (!offset)
                return offset.takeError();
              std::memcpy(view.storage->bytes.data() + *offset,
                          tensor.getBytes().data() + linearByte,
                          info->elementBytes);
              linearByte += info->elementBytes;
              return llvm::Error::success();
            }))
      return std::move(error);
    return view;
  }

  llvm::Expected<ReferenceTensor> exportTensor(const BufferView &view,
                                               llvm::StringRef dtype,
                                               llvm::ArrayRef<int64_t> shape) {
    auto bytes = getCompactByteCount(dtype, shape);
    if (!bytes)
      return invalid("output tensor has unsupported dtype or shape");
    if (elementDType(view.type.getElementType()) != dtype ||
        view.type.getShape() != shape)
      return invalid("entry result disagrees with typed output binding");
    std::vector<uint8_t> compact(*bytes);
    int64_t linearByte = 0;
    auto info = wafer::computeWaferPhysicalTensorInfo(view.type);
    if (!info || info->elementBytes <= 0)
      return invalid("output has no static accepted physical layout");
    if (llvm::Error error = forEachLogicalIndex(
            shape, [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
              auto offset = getAbsoluteOffset(view, index, info->elementBytes);
              if (!offset)
                return offset.takeError();
              std::memcpy(compact.data() + linearByte,
                          view.storage->bytes.data() + *offset,
                          info->elementBytes);
              linearByte += info->elementBytes;
              return llvm::Error::success();
            }))
      return std::move(error);
    return ReferenceTensor::create(dtype, shape, compact);
  }

  llvm::Expected<std::vector<ValueId>> executeControlFlow(
      const ReferenceProgram::Impl::ControlFlowProgram &controlFlow) {
    if (controlFlow.blocks.empty())
      return invalid("projected entry CFG has no blocks");
    uint32_t current = 0;
    while (true) {
      if (current >= controlFlow.blocks.size())
        return invalid("projected CFG successor is outside block graph");
      auto transfer = executeBlock(controlFlow.blocks[current]);
      if (!transfer)
        return transfer.takeError();
      if (transfer->kind == TransferKind::Return)
        return std::move(transfer->inputs);
      if (transfer->successor >= controlFlow.blocks.size())
        return invalid("projected branch successor is outside block graph");
      const auto &arguments = controlFlow.blocks[transfer->successor].arguments;
      if (llvm::Error error = assignValues(arguments, transfer->inputs))
        return std::move(error);
      current = transfer->successor;
    }
  }

  llvm::Expected<std::vector<ValueId>>
  executeStructuredBlock(const ReferenceProgram::Impl::BlockProgram &block) {
    auto transfer = executeBlock(block);
    if (!transfer)
      return transfer.takeError();
    if (transfer->kind != TransferKind::Return)
      return invalid("structured region produced a CFG branch");
    return std::move(transfer->inputs);
  }

  llvm::Expected<ControlTransfer>
  executeBlock(const ReferenceProgram::Impl::BlockProgram &block) {
    for (const Command &command : block.commands) {
      switch (command.kind) {
      case CommandKind::Alloc: {
        auto allocated = allocate(command);
        if (!allocated)
          return allocated.takeError();
        buffers[command.result] = std::move(*allocated);
        break;
      }
      case CommandKind::Dealloc:
        break;
      case CommandKind::Cast: {
        auto source = lookup(command.source);
        if (!source)
          return source.takeError();
        if ((command.viewDelta > 0 &&
             source->viewOffset >
                 std::numeric_limits<int64_t>::max() - command.viewDelta) ||
            (command.viewDelta < 0 && source->viewOffset < -command.viewDelta))
          return invalid("projected view offset exceeds its root storage");
        source->viewOffset += command.viewDelta;
        source->type = command.type;
        source->physicalBytes = command.physicalBytes;
        buffers[command.result] = *source;
        break;
      }
      case CommandKind::Constant:
        scalars[command.result] = command.scalarValue;
        break;
      case CommandKind::TileRegion: {
        if (command.inputs.size() != command.blockArguments.size())
          return invalid("projected tile region argument arity mismatch");
        if (llvm::Error error =
                assignValues(command.blockArguments, command.inputs))
          return std::move(error);
        auto yielded = executeStructuredBlock(*command.body);
        if (!yielded)
          return yielded.takeError();
        if (yielded->size() != command.results.size())
          return invalid("projected tile region result arity mismatch");
        if (llvm::Error error = assignValues(command.results, *yielded))
          return std::move(error);
        break;
      }
      case CommandKind::If: {
        auto condition = lookupBoolean(command.condition);
        if (!condition)
          return condition.takeError();
        std::vector<ValueId> yielded;
        if (*condition) {
          auto values = executeStructuredBlock(*command.body);
          if (!values)
            return values.takeError();
          yielded = std::move(*values);
        } else if (command.elseBody) {
          auto values = executeStructuredBlock(*command.elseBody);
          if (!values)
            return values.takeError();
          yielded = std::move(*values);
        }
        if (yielded.size() != command.results.size())
          return invalid("projected scf.if result arity mismatch");
        if (llvm::Error error = assignValues(command.results, yielded))
          return std::move(error);
        break;
      }
      case CommandKind::For: {
        auto lower = lookupSignedInteger(command.lowerBound);
        auto upper = lookupSignedInteger(command.upperBound);
        auto step = lookupSignedInteger(command.step);
        if (!lower)
          return lower.takeError();
        if (!upper)
          return upper.takeError();
        if (!step)
          return step.takeError();
        if (*step <= 0)
          return invalid("projected scf.for requires a positive step");
        auto initial = readValues(command.iterInputs);
        if (!initial)
          return initial.takeError();
        std::vector<RuntimeValue> carried = std::move(*initial);
        for (int64_t induction = *lower; induction < *upper;) {
          auto boundScalar = scalars.find(command.lowerBound);
          if (boundScalar == scalars.end() || !boundScalar->second.integer)
            return invalid("projected scf.for bound scalar is unavailable");
          unsigned width = boundScalar->second.integer->getBitWidth();
          Scalar inductionValue;
          inductionValue.integer = llvm::APInt(
              width, static_cast<uint64_t>(induction), /*isSigned=*/true);
          scalars[command.inductionArgument] = std::move(inductionValue);
          if (llvm::Error error = writeValues(command.iterArguments, carried))
            return std::move(error);
          auto yielded = executeStructuredBlock(*command.body);
          if (!yielded)
            return yielded.takeError();
          auto next = readValues(*yielded);
          if (!next)
            return next.takeError();
          carried = std::move(*next);
          if (induction > std::numeric_limits<int64_t>::max() - *step)
            return invalid("projected scf.for induction overflows int64");
          induction += *step;
        }
        if (llvm::Error error = writeValues(command.results, carried))
          return std::move(error);
        break;
      }
      case CommandKind::Call: {
        if (command.callee >= program.functions.size())
          return invalid("projected callee is outside the function graph");
        const auto &callee = program.functions[command.callee];
        auto arguments = readValues(command.inputs);
        if (!arguments)
          return arguments.takeError();
        if (llvm::Error error = writeValues(callee.arguments, *arguments))
          return std::move(error);
        auto returned = executeControlFlow(callee.body);
        if (!returned)
          return returned.takeError();
        auto returnValues = readValues(*returned);
        if (!returnValues)
          return returnValues.takeError();
        if (llvm::Error error = writeValues(command.results, *returnValues))
          return std::move(error);
        break;
      }
      case CommandKind::RDMA:
        if (llvm::Error error = executeRDMA(command))
          return std::move(error);
        break;
      case CommandKind::WDMA:
        if (llvm::Error error = executeWDMA(command))
          return std::move(error);
        break;
      case CommandKind::GatherScatter:
        if (llvm::Error error = executeGatherScatter(command))
          return std::move(error);
        break;
      case CommandKind::Convert:
        if (llvm::Error error = executeConvert(command))
          return std::move(error);
        break;
      case CommandKind::Gemm:
        if (llvm::Error error = executeGemm(command))
          return std::move(error);
        break;
      case CommandKind::Elementwise:
        if (llvm::Error error = executeElementwise(command))
          return std::move(error);
        break;
      case CommandKind::Fill:
        if (llvm::Error error = executeFill(command))
          return std::move(error);
        break;
      case CommandKind::LocalFence:
        break;
      case CommandKind::Branch:
        return ControlTransfer{TransferKind::Branch, command.successor,
                               command.inputs};
      case CommandKind::CondBranch: {
        auto condition = lookupBoolean(command.condition);
        if (!condition)
          return condition.takeError();
        return *condition
                   ? ControlTransfer{TransferKind::Branch,
                                     command.trueSuccessor, command.trueInputs}
                   : ControlTransfer{TransferKind::Branch,
                                     command.falseSuccessor,
                                     command.falseInputs};
      }
      case CommandKind::Return:
        return ControlTransfer{TransferKind::Return, 0, command.inputs};
      }
    }
    return invalid("projected block has no terminator command");
  }

  llvm::Expected<RuntimeValue> readValue(ValueId value) const {
    auto buffer = buffers.find(value);
    auto scalar = scalars.find(value);
    if (buffer != buffers.end() && scalar != scalars.end())
      return invalid("projected value is both buffer and scalar");
    if (buffer != buffers.end())
      return RuntimeValue{buffer->second, std::nullopt};
    if (scalar != scalars.end())
      return RuntimeValue{std::nullopt, scalar->second};
    return invalid("projected runtime value is unavailable");
  }

  llvm::Expected<std::vector<RuntimeValue>>
  readValues(llvm::ArrayRef<ValueId> values) const {
    std::vector<RuntimeValue> result;
    result.reserve(values.size());
    for (ValueId value : values) {
      auto runtimeValue = readValue(value);
      if (!runtimeValue)
        return runtimeValue.takeError();
      result.push_back(std::move(*runtimeValue));
    }
    return result;
  }

  llvm::Error writeValues(llvm::ArrayRef<ValueId> destinations,
                          llvm::ArrayRef<RuntimeValue> values) {
    if (destinations.size() != values.size())
      return invalid("projected runtime value arity mismatch");
    for (auto [destination, value] : llvm::zip_equal(destinations, values)) {
      buffers.erase(destination);
      scalars.erase(destination);
      if (value.buffer)
        buffers[destination] = *value.buffer;
      else if (value.scalar)
        scalars[destination] = *value.scalar;
      else
        return invalid("projected runtime value has no representation");
    }
    return llvm::Error::success();
  }

  llvm::Error assignValues(llvm::ArrayRef<ValueId> destinations,
                           llvm::ArrayRef<ValueId> sources) {
    auto values = readValues(sources);
    if (!values)
      return values.takeError();
    return writeValues(destinations, *values);
  }

  llvm::Expected<bool> lookupBoolean(ValueId value) const {
    auto found = scalars.find(value);
    if (found == scalars.end() || !found->second.integer ||
        found->second.integer->getBitWidth() != 1)
      return invalid("projected branch condition is not an available i1");
    return !found->second.integer->isZero();
  }

  llvm::Expected<int64_t> lookupSignedInteger(ValueId value) const {
    auto found = scalars.find(value);
    if (found == scalars.end() || !found->second.integer ||
        found->second.integer->getBitWidth() > 64)
      return invalid("projected loop bound is not an available integer");
    return found->second.integer->getSExtValue();
  }

  llvm::Expected<BufferView> allocate(const Command &command) {
    if (command.acceptedOffset < 0 || command.physicalBytes < 0 ||
        command.physicalBytes >
            std::numeric_limits<int64_t>::max() - command.acceptedOffset)
      return invalid("accepted allocation range overflows");
    std::shared_ptr<Storage> arena =
        command.spmAllocation ? spmArena : ddrArena;
    int64_t end = command.acceptedOffset + command.physicalBytes;
    if (end > static_cast<int64_t>(arena->bytes.size()))
      arena->bytes.resize(end, 0);
    return BufferView{std::move(arena), command.acceptedOffset, 0,
                      command.physicalBytes, command.type};
  }

  llvm::Expected<BufferView> lookup(ValueId value) const {
    auto found = buffers.find(value);
    if (found == buffers.end())
      return invalid("instruction operand buffer is unavailable");
    return found->second;
  }

  llvm::Error copyChunks(const BufferView &source, const BufferView &dest,
                         llvm::ArrayRef<int64_t> sourceOffsets,
                         llvm::ArrayRef<int64_t> destOffsets,
                         int64_t innerBytes) {
    if (sourceOffsets.size() != destOffsets.size() || innerBytes < 0)
      return invalid("movement descriptor has inconsistent chunk counts");
    std::vector<uint8_t> payload(sourceOffsets.size() * innerBytes);
    for (auto [position, offset] : llvm::enumerate(sourceOffsets)) {
      if (offset < 0 || offset > source.physicalBytes - innerBytes)
        return invalid("movement source descriptor exceeds buffer extent");
      int64_t absolute = source.base + source.viewOffset + offset;
      if (absolute < 0 ||
          absolute >
              static_cast<int64_t>(source.storage->bytes.size()) - innerBytes)
        return invalid("movement source exceeds storage arena");
      std::memcpy(payload.data() + position * innerBytes,
                  source.storage->bytes.data() + absolute, innerBytes);
    }
    for (auto [position, offset] : llvm::enumerate(destOffsets)) {
      if (offset < 0 || offset > dest.physicalBytes - innerBytes)
        return invalid("movement destination descriptor exceeds buffer extent");
      int64_t absolute = dest.base + dest.viewOffset + offset;
      if (absolute < 0 ||
          absolute >
              static_cast<int64_t>(dest.storage->bytes.size()) - innerBytes)
        return invalid("movement destination exceeds storage arena");
      std::memcpy(dest.storage->bytes.data() + absolute,
                  payload.data() + position * innerBytes, innerBytes);
    }
    return llvm::Error::success();
  }

  llvm::Expected<std::vector<int64_t>>
  descriptorOffsets(llvm::ArrayRef<int64_t> strides,
                    llvm::ArrayRef<int64_t> iterations, int64_t base = 0) {
    if (strides.size() != 3 || iterations.size() != 3 || base < 0)
      return invalid("movement descriptor must have three non-negative levels");
    std::vector<int64_t> offsets;
    for (int64_t i0 = 0; i0 < iterations[0]; ++i0)
      for (int64_t i1 = 0; i1 < iterations[1]; ++i1)
        for (int64_t i2 = 0; i2 < iterations[2]; ++i2) {
          if (strides[0] < 0 || strides[1] < 0 || strides[2] < 0)
            return invalid("movement descriptor contains a negative field");
          offsets.push_back(base + i0 * strides[0] + i1 * strides[1] +
                            i2 * strides[2]);
        }
    return offsets;
  }

  llvm::Error executeRDMA(const Command &command) {
    auto source = lookup(command.source);
    auto dest = lookup(command.dest);
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    auto sourceOffsets =
        descriptorOffsets(command.sourceStrides, command.sourceIterations);
    if (!sourceOffsets)
      return sourceOffsets.takeError();
    std::vector<int64_t> destOffsets(sourceOffsets->size());
    for (auto [index, offset] : llvm::enumerate(destOffsets))
      offset = index * command.innerBytes;
    return copyChunks(*source, *dest, *sourceOffsets, destOffsets,
                      command.innerBytes);
  }

  llvm::Error executeWDMA(const Command &command) {
    auto source = lookup(command.source);
    auto dest = lookup(command.dest);
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    auto destOffsets =
        descriptorOffsets(command.destStrides, command.destIterations);
    if (!destOffsets)
      return destOffsets.takeError();
    std::vector<int64_t> sourceOffsets(destOffsets->size());
    for (auto [index, offset] : llvm::enumerate(sourceOffsets))
      offset = index * command.innerBytes;
    return copyChunks(*source, *dest, sourceOffsets, *destOffsets,
                      command.innerBytes);
  }

  llvm::Error executeGatherScatter(const Command &command) {
    auto source = lookup(command.source);
    auto dest = lookup(command.dest);
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    auto sourceOffsets =
        descriptorOffsets(command.sourceStrides, command.sourceIterations,
                          command.sourceOffset.value_or(0));
    auto destOffsets =
        descriptorOffsets(command.destStrides, command.destIterations,
                          command.destOffset.value_or(0));
    if (!sourceOffsets)
      return sourceOffsets.takeError();
    if (!destOffsets)
      return destOffsets.takeError();
    if (sourceOffsets->size() != destOffsets->size())
      return invalid("movement source/destination descriptors disagree");
    return copyChunks(*source, *dest, *sourceOffsets, *destOffsets,
                      command.innerBytes);
  }

  llvm::Error executeConvert(const Command &command) {
    auto source = lookup(command.source);
    auto dest = lookup(command.dest);
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();

    std::vector<llvm::APInt> converted;
    converted.reserve(source->type.getNumElements());
    if (llvm::Error error = forEachLogicalIndex(
            source->type.getShape(),
            [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
              auto value = readNumeric(*source, index, command.sourceFormat);
              if (!value)
                return value.takeError();
              auto result = convertNumeric(*value, command.destFormat,
                                           command.roundingMode);
              if (!result)
                return result.takeError();
              converted.push_back(std::move(*result));
              return llvm::Error::success();
            }))
      return error;

    size_t position = 0;
    if (llvm::Error error = forEachLogicalIndex(
            dest->type.getShape(),
            [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
              if (position >= converted.size())
                return invalid(
                    "projected convert destination has excess elements");
              return writeNumericBits(*dest, index, command.destFormat,
                                      converted[position++]);
            }))
      return error;
    if (position != converted.size())
      return invalid("projected convert destination has too few elements");
    return llvm::Error::success();
  }

  llvm::Error executeGemm(const Command &command) {
    auto lhs = lookup(command.lhs);
    auto rhs = lookup(command.rhs);
    auto dest = lookup(command.dest);
    if (!lhs)
      return lhs.takeError();
    if (!rhs)
      return rhs.takeError();
    if (!dest)
      return dest.takeError();
    for (int64_t m = 0; m < command.m; ++m)
      for (int64_t n = 0; n < command.n; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < command.k; ++k) {
          auto lhsValue = readF32(*lhs, {m, k});
          auto rhsValue = readF32(*rhs, {k, n});
          if (!lhsValue)
            return lhsValue.takeError();
          if (!rhsValue)
            return rhsValue.takeError();
          sum += *lhsValue * *rhsValue;
        }
        if (llvm::Error error = writeF32(*dest, {m, n}, sum))
          return error;
      }
    return llvm::Error::success();
  }

  llvm::Error executeElementwise(const Command &command) {
    auto dest = lookup(command.dest);
    if (!dest)
      return dest.takeError();
    std::vector<BufferView> inputs;
    for (ValueId value : command.inputs) {
      auto input = lookup(value);
      if (!input)
        return input.takeError();
      inputs.push_back(std::move(*input));
    }
    return forEachLogicalIndex(
        dest->type.getShape(), [&](llvm::ArrayRef<int64_t> index) {
          llvm::SmallVector<float> values;
          for (const BufferView &input : inputs) {
            auto value = readF32(input, index);
            if (!value)
              return value.takeError();
            values.push_back(*value);
          }
          float result = 0.0f;
          switch (command.elementwiseKind) {
          case wafer::InstrElementwiseKind::Abs:
            result = std::fabs(values[0]);
            break;
          case wafer::InstrElementwiseKind::Recip:
            result = 1.0f / values[0];
            break;
          case wafer::InstrElementwiseKind::Square:
            result = values[0] * values[0];
            break;
          case wafer::InstrElementwiseKind::Sqrt:
            result = std::sqrt(values[0]);
            break;
          case wafer::InstrElementwiseKind::Rsqrt:
            result = 1.0f / std::sqrt(values[0]);
            break;
          case wafer::InstrElementwiseKind::Neg:
            result = -values[0];
            break;
          case wafer::InstrElementwiseKind::Max:
            result = std::max(values[0], values[1]);
            break;
          case wafer::InstrElementwiseKind::Min:
            result = std::min(values[0], values[1]);
            break;
          case wafer::InstrElementwiseKind::Add:
            result = values[0] + values[1];
            break;
          case wafer::InstrElementwiseKind::Sub:
            result = values[0] - values[1];
            break;
          case wafer::InstrElementwiseKind::Mul:
            result = values[0] * values[1];
            break;
          case wafer::InstrElementwiseKind::Div:
            result = values[0] / values[1];
            break;
          case wafer::InstrElementwiseKind::Log2:
            result = std::log2(values[0]);
            break;
          case wafer::InstrElementwiseKind::Ln:
            result = std::log(values[0]);
            break;
          case wafer::InstrElementwiseKind::Pow2:
            result = std::exp2(values[0]);
            break;
          case wafer::InstrElementwiseKind::Exp:
          case wafer::InstrElementwiseKind::ExpLp:
            result = std::exp(values[0]);
            break;
          case wafer::InstrElementwiseKind::Sin:
            result = std::sin(values[0]);
            break;
          case wafer::InstrElementwiseKind::Cos:
            result = std::cos(values[0]);
            break;
          case wafer::InstrElementwiseKind::Tanh:
            result = std::tanh(values[0]);
            break;
          case wafer::InstrElementwiseKind::Sigmoid:
            result = 1.0f / (1.0f + std::exp(-values[0]));
            break;
          case wafer::InstrElementwiseKind::Relu:
          case wafer::InstrElementwiseKind::SatRelu:
            result = std::max(0.0f, values[0]);
            break;
          case wafer::InstrElementwiseKind::Softplus:
            result = std::log1p(std::exp(values[0]));
            break;
          default:
            return invalid("unsupported projected elementwise kind");
          }
          return writeF32(*dest, index, result);
        });
  }

  llvm::Expected<float> convertScalarToF32(const Scalar &scalar) {
    if (scalar.floating) {
      llvm::APFloat value = *scalar.floating;
      bool losesInfo = false;
      value.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      return value.convertToFloat();
    }
    if (scalar.integer) {
      if (scalar.integer->getBitWidth() > 64)
        return unsupported("fill integer wider than 64 bits");
      if (scalar.integerIsUnsigned)
        return static_cast<float>(scalar.integer->getZExtValue());
      return static_cast<float>(scalar.integer->getSExtValue());
    }
    return invalid("projected scalar has no value");
  }

  llvm::Error executeFill(const Command &command) {
    auto dest = lookup(command.dest);
    if (!dest)
      return dest.takeError();
    auto scalar = scalars.find(command.scalar);
    if (scalar == scalars.end())
      return invalid("fill scalar is unavailable");
    auto value = convertScalarToF32(scalar->second);
    if (!value)
      return value.takeError();
    return forEachLogicalIndex(dest->type.getShape(),
                               [&](llvm::ArrayRef<int64_t> index) {
                                 return writeF32(*dest, index, *value);
                               });
  }

  const ReferenceProgram::Impl &program;
  std::shared_ptr<Storage> spmArena;
  std::shared_ptr<Storage> ddrArena;
  llvm::DenseMap<ValueId, BufferView> buffers;
  llvm::DenseMap<ValueId, Scalar> scalars;
};

} // namespace

struct ReferenceProgramBuilder {
  static llvm::Expected<ReferenceProgram>
  prepare(const ExecutableBundle &bundle, int64_t logicalRank) {
    if (logicalRank < 0 ||
        logicalRank >= bundle.getExecutionConfig().getRankCount())
      return invalid("reference logical rank is outside executable bundle");
    const RankExecutable *match = nullptr;
    for (const RankExecutable &rank : bundle.getRankExecutables()) {
      if (rank.getLogicalRank() != logicalRank)
        continue;
      if (match)
        return invalid("executable bundle contains a duplicate logical rank");
      match = &rank;
    }
    if (!match)
      return invalid("executable bundle is missing requested logical rank");
    if (match->getTransportContract() != TransportContract::None)
      return unsupported("single-rank executor does not accept transport");

    llvm::Expected<detail::AcceptedCallClosure> closure =
        detail::analyzeAcceptedCallClosure(match->getModule(),
                                           match->getEntrySymbol());
    if (!closure) {
      std::string message = "accepted call closure is invalid: " +
                            llvm::toString(closure.takeError());
      return unsupported(message);
    }

    auto impl = std::make_unique<ReferenceProgram::Impl>();
    impl->contextOwner = bundle.context;
    impl->logicalRank = logicalRank;
    impl->programBindings = match->getProgramBindings();
    if (llvm::Error error = ProgramProjector(*impl).project(*closure))
      return std::move(error);
    return ReferenceProgram(std::move(impl));
  }
};

ReferenceProgram::ReferenceProgram(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

ReferenceProgram::ReferenceProgram(ReferenceProgram &&) noexcept = default;
ReferenceProgram &
ReferenceProgram::operator=(ReferenceProgram &&) noexcept = default;
ReferenceProgram::~ReferenceProgram() = default;

int64_t ReferenceProgram::getLogicalRank() const {
  return impl ? impl->logicalRank : -1;
}

size_t ReferenceProgram::getProjectedOperationCount() const {
  return impl ? impl->projectedOperationCount : 0;
}

llvm::Expected<ReferenceTensor>
ReferenceTensor::create(llvm::StringRef dtype, llvm::ArrayRef<int64_t> shape,
                        llvm::ArrayRef<uint8_t> bytes) {
  std::optional<int64_t> expected = getCompactByteCount(dtype, shape);
  if (!expected)
    return invalid("reference tensor has unsupported dtype or shape");
  if (*expected != static_cast<int64_t>(bytes.size()))
    return invalid(
        "reference tensor byte count disagrees with dtype and shape");
  return ReferenceTensor(dtype.str(), std::vector<int64_t>(shape),
                         std::vector<uint8_t>(bytes));
}

llvm::Expected<ReferenceProgram>
prepareReferenceRank(const ExecutableBundle &bundle, int64_t logicalRank) {
  return ReferenceProgramBuilder::prepare(bundle, logicalRank);
}

llvm::Expected<ReferenceExecutionResult>
executeReferenceProgram(const ReferenceProgram &program,
                        llvm::ArrayRef<ReferenceInputBinding> inputs) {
  if (!program.impl)
    return invalid("reference program is moved-from");
  return ProgramInterpreter(*program.impl).run(inputs);
}

llvm::Expected<ReferenceExecutionResult>
executeReferenceRank(const ExecutableBundle &bundle, int64_t logicalRank,
                     llvm::ArrayRef<ReferenceInputBinding> inputs) {
  auto program = prepareReferenceRank(bundle, logicalRank);
  if (!program)
    return program.takeError();
  return executeReferenceProgram(*program, inputs);
}

} // namespace wafer::compiler
