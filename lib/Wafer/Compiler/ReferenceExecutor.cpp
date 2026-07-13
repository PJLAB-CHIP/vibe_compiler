//===- ReferenceExecutor.cpp - Accepted-rank semantic execution ----------===//

#include "Wafer/Compiler/ReferenceExecutor.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
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
    RDMA,
    WDMA,
    GatherScatter,
    Convert,
    Gemm,
    Elementwise,
    Fill,
    LocalFence,
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
    std::vector<Command> commands;
  };

  std::shared_ptr<mlir::MLIRContext> contextOwner;
  int64_t logicalRank = 0;
  std::vector<RankProgramBinding> programBindings;
  std::vector<ValueId> entryArguments;
  std::vector<mlir::MemRefType> entryArgumentTypes;
  BlockProgram entry;
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

enum class ConvertParameterKind { None, Rounding, ZeroPoint };

struct ConvertSpec {
  NumericFormat source;
  NumericFormat dest;
  ConvertParameterKind parameter;
};

std::optional<ConvertSpec> getConvertSpec(wafer::InstrConvertKind kind) {
  using Kind = wafer::InstrConvertKind;
  using Parameter = ConvertParameterKind;
  using Format = NumericFormat;
  switch (kind) {
  case Kind::Int8Fp16:
    return ConvertSpec{Format::Int8, Format::Float16, Parameter::ZeroPoint};
  case Kind::Int8Bf16:
    return ConvertSpec{Format::Int8, Format::BFloat16, Parameter::ZeroPoint};
  case Kind::Int8Fp32:
    return ConvertSpec{Format::Int8, Format::Float32, Parameter::ZeroPoint};
  case Kind::Int8Tf32:
    return ConvertSpec{Format::Int8, Format::TF32, Parameter::ZeroPoint};
  case Kind::Int16Fp16:
    return ConvertSpec{Format::Int16, Format::Float16, Parameter::None};
  case Kind::Int16Bf16:
    return ConvertSpec{Format::Int16, Format::BFloat16, Parameter::Rounding};
  case Kind::Int16Fp32:
    return ConvertSpec{Format::Int16, Format::Float32, Parameter::Rounding};
  case Kind::Int16Tf32:
    return ConvertSpec{Format::Int16, Format::TF32, Parameter::Rounding};
  case Kind::Int32Fp16:
    return ConvertSpec{Format::Int32, Format::Float16, Parameter::Rounding};
  case Kind::Int32Bf16:
    return ConvertSpec{Format::Int32, Format::BFloat16, Parameter::Rounding};
  case Kind::Int32Fp32:
    return ConvertSpec{Format::Int32, Format::Float32, Parameter::Rounding};
  case Kind::Int32Tf32:
    return ConvertSpec{Format::Int32, Format::TF32, Parameter::Rounding};
  case Kind::Bf16Int8:
    return ConvertSpec{Format::BFloat16, Format::Int8, Parameter::None};
  case Kind::Bf16Int16:
    return ConvertSpec{Format::BFloat16, Format::Int16, Parameter::Rounding};
  case Kind::Bf16Int32:
    return ConvertSpec{Format::BFloat16, Format::Int32, Parameter::Rounding};
  case Kind::Bf16Fp16:
    return ConvertSpec{Format::BFloat16, Format::Float16, Parameter::None};
  case Kind::Bf16Fp32:
    return ConvertSpec{Format::BFloat16, Format::Float32, Parameter::None};
  case Kind::Bf16Tf32:
    return ConvertSpec{Format::BFloat16, Format::TF32, Parameter::None};
  case Kind::Fp16Int8:
    return ConvertSpec{Format::Float16, Format::Int8, Parameter::Rounding};
  case Kind::Fp16Int16:
    return ConvertSpec{Format::Float16, Format::Int16, Parameter::Rounding};
  case Kind::Fp16Int32:
    return ConvertSpec{Format::Float16, Format::Int32, Parameter::Rounding};
  case Kind::Fp16Bf16:
    return ConvertSpec{Format::Float16, Format::BFloat16, Parameter::Rounding};
  case Kind::Fp16Fp32:
    return ConvertSpec{Format::Float16, Format::Float32, Parameter::None};
  case Kind::Fp16Tf32:
    return ConvertSpec{Format::Float16, Format::TF32, Parameter::None};
  case Kind::Fp32Int8:
    return ConvertSpec{Format::Float32, Format::Int8, Parameter::Rounding};
  case Kind::Fp32Int16:
    return ConvertSpec{Format::Float32, Format::Int16, Parameter::Rounding};
  case Kind::Fp32Int32:
    return ConvertSpec{Format::Float32, Format::Int32, Parameter::Rounding};
  case Kind::Fp32Fp16:
    return ConvertSpec{Format::Float32, Format::Float16, Parameter::Rounding};
  case Kind::Fp32Bf16:
    return ConvertSpec{Format::Float32, Format::BFloat16, Parameter::Rounding};
  case Kind::Fp32Tf32:
    return ConvertSpec{Format::Float32, Format::TF32, Parameter::Rounding};
  case Kind::Tf32Int8:
    return ConvertSpec{Format::TF32, Format::Int8, Parameter::Rounding};
  case Kind::Tf32Int16:
    return ConvertSpec{Format::TF32, Format::Int16, Parameter::Rounding};
  case Kind::Tf32Int32:
    return ConvertSpec{Format::TF32, Format::Int32, Parameter::Rounding};
  case Kind::Tf32Fp16:
    return ConvertSpec{Format::TF32, Format::Float16, Parameter::None};
  case Kind::Tf32Bf16:
    return ConvertSpec{Format::TF32, Format::BFloat16, Parameter::Rounding};
  case Kind::Tf32Fp32:
    return ConvertSpec{Format::TF32, Format::Float32, Parameter::None};
  }
  return std::nullopt;
}

bool matchesNumericFormat(mlir::Type type, NumericFormat format) {
  switch (format) {
  case NumericFormat::Int8:
  case NumericFormat::Int16:
  case NumericFormat::Int32: {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
    unsigned expectedWidth = format == NumericFormat::Int8    ? 8
                             : format == NumericFormat::Int16 ? 16
                                                              : 32;
    return integer && integer.isSignless() &&
           integer.getWidth() == expectedWidth;
  }
  case NumericFormat::BFloat16:
    return type.isBF16();
  case NumericFormat::Float16:
    return type.isF16();
  case NumericFormat::Float32:
    return type.isF32();
  case NumericFormat::TF32:
    return mlir::isa<mlir::FloatTF32Type>(type);
  }
  return false;
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

  llvm::Error project(mlir::func::FuncOp entry) {
    if (!entry.getBody().hasOneBlock())
      return unsupported("entry requires unsupported multi-block control flow");
    mlir::Block &block = entry.getBody().front();
    for (mlir::BlockArgument argument : block.getArguments()) {
      auto type = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
      if (!type)
        return unsupported("entry argument is not a memref");
      auto value = define(argument);
      if (!value)
        return value.takeError();
      program.entryArguments.push_back(*value);
      program.entryArgumentTypes.push_back(type);
    }
    if (llvm::Error error = projectBlock(block, program.entry))
      return error;
    if (program.entry.commands.empty() ||
        program.entry.commands.back().kind !=
            ReferenceProgram::Impl::CommandKind::Return)
      return unsupported("entry has no projected return terminator");
    if (program.entry.commands.back().inputs.size() != entry.getNumResults())
      return unsupported("entry return arity disagrees with function type");
    return validateProgramBindings(entry);
  }

private:
  using Command = ReferenceProgram::Impl::Command;
  using CommandKind = ReferenceProgram::Impl::CommandKind;
  using ValueId = ReferenceProgram::Impl::ValueId;

  llvm::Expected<ValueId> define(mlir::Value value) {
    if (values.count(value))
      return unsupported("SSA value is projected more than once");
    if (nextValue == std::numeric_limits<ValueId>::max())
      return unsupported("reference value id space is exhausted");
    ValueId id = nextValue++;
    values[value] = id;
    return id;
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
        std::optional<ConvertSpec> spec = getConvertSpec(convert.getKind());
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
        case ConvertParameterKind::None:
          if (zeroPoint || roundingMode)
            return unsupported("plain convert has unexpected parameters");
          break;
        case ConvertParameterKind::Rounding: {
          if (zeroPoint || !roundingMode)
            return unsupported(
                "rounding convert has invalid parameter combination");
          auto projectedRounding = projectRoundingMode(*roundingMode);
          if (!projectedRounding)
            return projectedRounding.takeError();
          command.roundingMode = *projectedRounding;
          break;
        }
        case ConvertParameterKind::ZeroPoint:
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
      } else if (mlir::isa<wafer::TileYieldOp, mlir::func::ReturnOp>(
                     operation)) {
        command.kind = CommandKind::Return;
        for (mlir::Value operand : operation.getOperands()) {
          auto id = use(operand);
          if (!id)
            return id.takeError();
          command.inputs.push_back(*id);
        }
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

  ReferenceProgram::Impl &program;
  llvm::DenseMap<mlir::Value, ValueId> values;
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
  if (const llvm::fltSemantics *semantics = getFloatSemantics(format))
    return NumericValue{std::nullopt, llvm::APFloat(*semantics, bits)};
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
  if (!matchesNumericFormat(buffer.type.getElementType(), format) ||
      bits.getBitWidth() != getNumericBitWidth(format))
    return invalid(
        "projected convert destination format disagrees with buffer");
  int64_t byteWidth = bits.getBitWidth() / 8;
  auto offset = getAbsoluteOffset(buffer, indices, byteWidth);
  if (!offset)
    return offset.takeError();
  for (int64_t byte = 0; byte < byteWidth; ++byte)
    buffer.storage->bytes[*offset + byte] =
        static_cast<uint8_t>(bits.extractBitsAsZExtValue(8, byte * 8));
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
    if (llvm::Error error = bindEntryArguments(inputs))
      return std::move(error);
    auto returned = executeBlock(program.entry);
    if (!returned)
      return returned.takeError();

    std::vector<ReferenceOutputBinding> outputs;
    for (const RankProgramBinding &binding : program.programBindings) {
      if (binding.role != ProgramResourceRole::Output)
        continue;
      if (binding.index < 0 ||
          binding.index >= static_cast<int64_t>(returned->size()))
        return invalid("output binding index is outside entry results");
      auto tensor = exportTensor((*returned)[binding.index], binding.dtype,
                                 binding.localShape);
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

  llvm::Error bindEntryArguments(llvm::ArrayRef<ReferenceInputBinding> inputs) {
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
                                   program.entryArgumentTypes[binding.index]);
      if (!imported)
        return imported.takeError();
      buffers[program.entryArguments[binding.index]] = std::move(*imported);
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

  llvm::Expected<std::vector<BufferView>>
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
        for (auto [argument, input] :
             llvm::zip_equal(command.blockArguments, command.inputs)) {
          auto found = lookup(input);
          if (!found)
            return found.takeError();
          buffers[argument] = *found;
        }
        auto yielded = executeBlock(*command.body);
        if (!yielded)
          return yielded.takeError();
        if (yielded->size() != command.results.size())
          return invalid("projected tile region result arity mismatch");
        for (auto [result, value] : llvm::zip_equal(command.results, *yielded))
          buffers[result] = value;
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
      case CommandKind::Return: {
        std::vector<BufferView> result;
        for (ValueId value : command.inputs) {
          auto found = lookup(value);
          if (!found)
            return found.takeError();
          result.push_back(*found);
        }
        return result;
      }
      }
    }
    return invalid("projected block has no return command");
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

    mlir::func::FuncOp entry =
        match->getModule().lookupSymbol<mlir::func::FuncOp>(
            match->getEntrySymbol());
    if (!entry)
      return invalid("accepted rank entry function is missing");

    auto impl = std::make_unique<ReferenceProgram::Impl>();
    impl->contextOwner = bundle.context;
    impl->logicalRank = logicalRank;
    impl->programBindings = match->getProgramBindings();
    if (llvm::Error error = ProgramProjector(*impl).project(entry))
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
