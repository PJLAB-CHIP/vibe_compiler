//===- ReferenceProgramInterpreter.cpp - Immutable graph interpreter ----===//

#include "ReferenceExecutorInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler::reference_detail {
namespace {

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

class ProgramInterpreter {
public:
  ProgramInterpreter(const ReferenceProgram::Impl &program,
                     ReferenceExecutionOptions options)
      : program(program), spmArena(std::make_shared<Storage>()),
        ddrArena(std::make_shared<Storage>()), options(options),
        stochasticState(options.stochasticSeed.value_or(0)) {}

  llvm::Expected<ReferenceExecutionResult>
  run(llvm::ArrayRef<ReferenceInputBinding> inputs) {
    if (program.entryFunction >= program.functions.size())
      return invalid("projected entry function is outside the function graph");
    if (program.usesStochasticRounding && !options.stochasticSeed)
      return invalid(
          "stochastic reference rounding requires an explicit execution seed");
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
              auto result =
                  command.stochasticRounding
                      ? convertNumericStochastic(*value, command.destFormat,
                                                 nextStochasticBits())
                      : convertNumeric(*value, command.destFormat,
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

  uint64_t nextStochasticBits() {
    uint64_t value = (stochasticState += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
  }

  const ReferenceProgram::Impl &program;
  std::shared_ptr<Storage> spmArena;
  std::shared_ptr<Storage> ddrArena;
  ReferenceExecutionOptions options;
  uint64_t stochasticState;
  llvm::DenseMap<ValueId, BufferView> buffers;
  llvm::DenseMap<ValueId, Scalar> scalars;
};

} // namespace

llvm::Expected<ReferenceExecutionResult>
interpretReferenceProgram(const ReferenceProgram::Impl &program,
                          llvm::ArrayRef<ReferenceInputBinding> inputs,
                          ReferenceExecutionOptions options) {
  return ProgramInterpreter(program, options).run(inputs);
}

} // namespace wafer::compiler::reference_detail
