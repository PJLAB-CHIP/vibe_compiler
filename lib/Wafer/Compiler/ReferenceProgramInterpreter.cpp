//===- ReferenceProgramInterpreter.cpp - Immutable graph interpreter ----===//

#include "ReferenceExecutorInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
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

static float maximumF32(float lhs, float rhs) {
  if (std::isnan(lhs) || std::isnan(rhs))
    return std::numeric_limits<float>::quiet_NaN();
  if (lhs == 0.0f && rhs == 0.0f)
    return std::signbit(lhs) && std::signbit(rhs) ? -0.0f : 0.0f;
  return std::max(lhs, rhs);
}

static float minimumF32(float lhs, float rhs) {
  if (std::isnan(lhs) || std::isnan(rhs))
    return std::numeric_limits<float>::quiet_NaN();
  if (lhs == 0.0f && rhs == 0.0f)
    return std::signbit(lhs) || std::signbit(rhs) ? -0.0f : 0.0f;
  return std::min(lhs, rhs);
}

struct TransportEventKey {
  int64_t source = -1;
  int64_t destination = -1;
  int64_t communication = -1;
  uint32_t phase = 0;
  int64_t round = -1;
  int64_t payloadSlice = -1;
  std::vector<int64_t> controlInstance;

  bool operator<(const TransportEventKey &other) const {
    return std::tie(source, destination, communication, phase, round,
                    payloadSlice, controlInstance) <
           std::tie(other.source, other.destination, other.communication,
                    other.phase, other.round, other.payloadSlice,
                    other.controlInstance);
  }
};

class TransportCoordinator {
public:
  virtual ~TransportCoordinator() = default;
  virtual llvm::Error issue(const TransportEventKey &key, bool isSend,
                            uint64_t byteCount, int64_t remoteReceiverOffset,
                            const BufferView &buffer) = 0;
  virtual bool isComplete(const TransportEventKey &key) const = 0;
};

struct TransportToken {
  TransportEventKey key;
  bool isSend = false;
};

class TransportBlockedError : public llvm::ErrorInfo<TransportBlockedError> {
public:
  static char ID;
  explicit TransportBlockedError(std::string message)
      : message(std::move(message)) {}
  void log(llvm::raw_ostream &stream) const override { stream << message; }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
  std::string message;
};

char TransportBlockedError::ID;

struct ProgramRunAttempt {
  std::optional<ReferenceExecutionResult> result;
  std::string blockedReason;
};

class ProgramInterpreter {
public:
  ProgramInterpreter(const ReferenceProgram::Impl &program,
                     ReferenceExecutionOptions options,
                     TransportCoordinator *transport = nullptr)
      : program(program), spmArena(std::make_shared<Storage>()),
        ddrArena(std::make_shared<Storage>()), options(options),
        stochasticState(options.stochasticSeed.value_or(0)),
        transport(transport) {}

  llvm::Expected<ReferenceExecutionResult>
  run(llvm::ArrayRef<ReferenceInputBinding> inputs) {
    auto attempt = runUntilBlocked(inputs);
    if (!attempt)
      return attempt.takeError();
    if (!attempt->result)
      return invalid("single-rank reference execution blocked on transport");
    return std::move(*attempt->result);
  }

  llvm::Expected<ProgramRunAttempt>
  runUntilBlocked(llvm::ArrayRef<ReferenceInputBinding> inputs) {
    if (program.entryFunction >= program.functions.size())
      return invalid("projected entry function is outside the function graph");
    if (program.usesStochasticRounding && !options.stochasticSeed)
      return invalid(
          "stochastic reference rounding requires an explicit execution seed");
    if (llvm::Error error = bindEntryArguments(inputs))
      return std::move(error);
    auto returned =
        executeControlFlow(program.functions[program.entryFunction].body);
    if (!returned) {
      llvm::Error error = returned.takeError();
      ProgramRunAttempt attempt;
      error = llvm::handleErrors(std::move(error),
                                 [&](const TransportBlockedError &blocked) {
                                   attempt.blockedReason = blocked.message;
                                 });
      if (error)
        return std::move(error);
      return attempt;
    }

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
      outputs.push_back({binding.programIndex, std::move(*tensor)});
    }
    ProgramRunAttempt attempt;
    attempt.result = ReferenceExecutionResultBuilder::make(program.logicalRank,
                                                           std::move(outputs));
    return attempt;
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
        controlInstance.push_back(kIfControlMarker);
        controlInstance.push_back(*condition ? 1 : 0);
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
        controlInstance.resize(controlInstance.size() - 2);
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
          controlInstance.push_back(kForControlMarker);
          controlInstance.push_back(induction);
          if (llvm::Error error = writeValues(command.iterArguments, carried))
            return std::move(error);
          auto yielded = executeStructuredBlock(*command.body);
          if (!yielded)
            return yielded.takeError();
          controlInstance.resize(controlInstance.size() - 2);
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
      case CommandKind::Reduce:
        if (llvm::Error error = executeReduce(command))
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
      case CommandKind::Bit2Fp:
        if (llvm::Error error = executeBit2Fp(command))
          return std::move(error);
        break;
      case CommandKind::MaskMove:
        if (llvm::Error error = executeMaskMove(command))
          return std::move(error);
        break;
      case CommandKind::LocalFence:
        break;
      case CommandKind::DTESend:
      case CommandKind::DTERecv: {
        if (!transport)
          return unsupported("Direct DTE requires multi-rank execution");
        auto buffer = lookup(command.source);
        if (!buffer)
          return buffer.takeError();
        const bool isSend = command.kind == CommandKind::DTESend;
        TransportEventKey key{isSend ? program.logicalRank : command.peer,
                              isSend ? command.peer : program.logicalRank,
                              command.messageCommunication,
                              static_cast<uint32_t>(command.messagePhase),
                              command.messageRound,
                              command.messagePayloadSlice,
                              controlInstance};
        if (llvm::Error error =
                transport->issue(key, isSend, command.byteCount,
                                 command.remoteReceiverOffset, *buffer))
          return std::move(error);
        tokens[command.result] = TransportToken{std::move(key), isSend};
        break;
      }
      case CommandKind::DTEWait: {
        std::vector<TransportToken> blocked;
        for (ValueId tokenId : command.inputs) {
          auto token = tokens.find(tokenId);
          if (token == tokens.end())
            return invalid("Direct DTE wait token is unavailable");
          if (!transport || !transport->isComplete(token->second.key))
            blocked.push_back(token->second);
        }
        if (!blocked.empty()) {
          std::string reason;
          llvm::raw_string_ostream stream(reason);
          stream << "rank " << program.logicalRank << " waits for";
          for (const TransportToken &token : blocked) {
            const TransportEventKey &key = token.key;
            stream << " [" << key.source << "->" << key.destination
                   << " token=" << (token.isSend ? "send" : "recv")
                   << " communication=" << key.communication
                   << " phase=" << key.phase << " round=" << key.round
                   << " slice=" << key.payloadSlice << " control=[";
            llvm::interleaveComma(key.controlInstance, stream);
            stream << "]]";
          }
          return llvm::make_error<TransportBlockedError>(stream.str());
        }
        break;
      }
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
    return forEachLogicalIndex(
        command.batchShape,
        [&](llvm::ArrayRef<int64_t> batchIndex) -> llvm::Error {
          for (int64_t m = 0; m < command.m; ++m)
            for (int64_t n = 0; n < command.n; ++n) {
              float sum = 0.0f;
              for (int64_t k = 0; k < command.k; ++k) {
                llvm::SmallVector<int64_t> lhsIndex(batchIndex.begin(),
                                                    batchIndex.end());
                llvm::SmallVector<int64_t> rhsIndex(batchIndex.begin(),
                                                    batchIndex.end());
                lhsIndex.append({m, k});
                rhsIndex.append({k, n});
                auto lhsValue = readF32(*lhs, lhsIndex);
                auto rhsValue = readF32(*rhs, rhsIndex);
                if (!lhsValue)
                  return lhsValue.takeError();
                if (!rhsValue)
                  return rhsValue.takeError();
                sum += *lhsValue * *rhsValue;
              }
              llvm::SmallVector<int64_t> destIndex(batchIndex.begin(),
                                                   batchIndex.end());
              destIndex.append({m, n});
              if (llvm::Error error = writeF32(*dest, destIndex, sum))
                return error;
            }
          return llvm::Error::success();
        });
  }

  llvm::Error executeReduce(const Command &command) {
    auto input = lookup(command.source);
    auto dest = lookup(command.dest);
    if (!input)
      return input.takeError();
    if (!dest)
      return dest.takeError();

    const Scalar *init = &command.scalarValue;
    if (command.reduceInit) {
      auto found = scalars.find(*command.reduceInit);
      if (found == scalars.end())
        return invalid("reduce scalar initialization is unavailable");
      init = &found->second;
    }
    auto initValue = convertScalarToF32(*init);
    if (!initValue)
      return initValue.takeError();
    if (llvm::Error error = forEachLogicalIndex(
            dest->type.getShape(), [&](llvm::ArrayRef<int64_t> index) {
              return writeF32(*dest, index, *initValue);
            }))
      return error;

    return forEachLogicalIndex(
        input->type.getShape(), [&](llvm::ArrayRef<int64_t> inputIndex) {
          llvm::SmallVector<int64_t> destIndex;
          for (auto [dimension, value] : llvm::enumerate(inputIndex))
            if (!llvm::is_contained(command.reduceDimensions,
                                    static_cast<int64_t>(dimension)))
              destIndex.push_back(value);
          auto value = readF32(*input, inputIndex);
          auto accumulator = readF32(*dest, destIndex);
          if (!value)
            return value.takeError();
          if (!accumulator)
            return accumulator.takeError();

          float result = 0.0f;
          switch (command.reduceKind) {
          case wafer::InstrReduceKind::Sum:
            result = *accumulator + *value;
            break;
          case wafer::InstrReduceKind::Max:
            result = maximumF32(*accumulator, *value);
            break;
          case wafer::InstrReduceKind::Min:
            result = minimumF32(*accumulator, *value);
            break;
          case wafer::InstrReduceKind::Avg:
            return invalid("unsupported projected reduce kind");
          }
          return writeF32(*dest, destIndex, result);
        });
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
            result = maximumF32(values[0], values[1]);
            break;
          case wafer::InstrElementwiseKind::Min:
            result = minimumF32(values[0], values[1]);
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
    if (dest->type.getElementType().isInteger(1)) {
      if (!scalar->second.integer || scalar->second.integer->getBitWidth() != 1)
        return invalid("i1 fill requires an i1 scalar");
      bool value = !scalar->second.integer->isZero();
      return forEachLogicalIndex(dest->type.getShape(),
                                 [&](llvm::ArrayRef<int64_t> index) {
                                   return writeI1(*dest, index, value);
                                 });
    }
    auto value = convertScalarToF32(scalar->second);
    if (!value)
      return value.takeError();
    return forEachLogicalIndex(dest->type.getShape(),
                               [&](llvm::ArrayRef<int64_t> index) {
                                 return writeF32(*dest, index, *value);
                               });
  }

  llvm::Error executeBit2Fp(const Command &command) {
    auto source = lookup(command.source);
    auto dest = lookup(command.dest);
    if (!source)
      return source.takeError();
    if (!dest)
      return dest.takeError();
    return forEachLogicalIndex(
        dest->type.getShape(), [&](llvm::ArrayRef<int64_t> index) {
          auto value = readI1(*source, index);
          if (!value)
            return value.takeError();
          return writeF32(*dest, index, *value ? 1.0f : 0.0f);
        });
  }

  llvm::Error executeMaskMove(const Command &command) {
    auto source = lookup(command.source);
    auto mask = lookup(command.mask);
    auto dest = lookup(command.dest);
    if (!source)
      return source.takeError();
    if (!mask)
      return mask.takeError();
    if (!dest)
      return dest.takeError();
    return forEachLogicalIndex(
        dest->type.getShape(),
        [&](llvm::ArrayRef<int64_t> index) -> llvm::Error {
          auto maskValue = readF32(*mask, index);
          if (!maskValue)
            return maskValue.takeError();
          if (*maskValue == 0.0f)
            return llvm::Error::success();
          auto sourceValue = readF32(*source, index);
          if (!sourceValue)
            return sourceValue.takeError();
          return writeF32(*dest, index, *sourceValue);
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
  TransportCoordinator *transport = nullptr;
  llvm::DenseMap<ValueId, BufferView> buffers;
  llvm::DenseMap<ValueId, Scalar> scalars;
  llvm::DenseMap<ValueId, TransportToken> tokens;
  std::vector<int64_t> controlInstance;

  static constexpr int64_t kIfControlMarker =
      std::numeric_limits<int64_t>::min();
  static constexpr int64_t kForControlMarker =
      std::numeric_limits<int64_t>::min() + 1;
};

class DeterministicTransportCoordinator final : public TransportCoordinator {
public:
  void beginReplay(int64_t logicalRank) {
    currentRank = logicalRank;
    replaySends.clear();
    replayRecvs.clear();
  }

  llvm::Error issue(const TransportEventKey &key, bool isSend,
                    uint64_t byteCount, int64_t remoteReceiverOffset,
                    const BufferView &buffer) override {
    if ((isSend ? key.source : key.destination) != currentRank)
      return invalid("Direct DTE issue rank disagrees with event endpoint");
    std::set<TransportEventKey> &seen = isSend ? replaySends : replayRecvs;
    if (!seen.insert(key).second)
      return invalid(isSend ? "duplicate Direct DTE send instance"
                            : "duplicate Direct DTE recv instance");
    if (byteCount == 0 ||
        byteCount > static_cast<uint64_t>(buffer.physicalBytes))
      return invalid("Direct DTE byte count exceeds projected buffer");
    if (buffer.base < 0 || buffer.viewOffset < 0 ||
        buffer.base > std::numeric_limits<int64_t>::max() - buffer.viewOffset)
      return invalid("Direct DTE buffer address overflows");
    const int64_t absolute = buffer.base + buffer.viewOffset;
    if (absolute > static_cast<int64_t>(buffer.storage->bytes.size()) ||
        byteCount >
            buffer.storage->bytes.size() - static_cast<size_t>(absolute))
      return invalid("Direct DTE buffer exceeds rank-local storage");

    EventState &state = events[key];
    if (isSend) {
      std::vector<uint8_t> payload(byteCount);
      std::memcpy(payload.data(), buffer.storage->bytes.data() + absolute,
                  byteCount);
      if (!state.sendIssued) {
        state.sendIssued = true;
        state.sendBytes = byteCount;
        state.senderRemoteOffset = remoteReceiverOffset;
        state.payload = std::move(payload);
        madeProgress = true;
      } else if (state.sendBytes != byteCount ||
                 state.senderRemoteOffset != remoteReceiverOffset ||
                 state.payload != payload) {
        return invalid("replayed Direct DTE send is not deterministic");
      }
      return llvm::Error::success();
    }

    if (absolute != remoteReceiverOffset)
      return invalid(
          "Direct DTE recv address disagrees with accepted remote offset");
    if (!state.recvIssued) {
      state.recvIssued = true;
      state.recvBytes = byteCount;
      state.receiverOffset = absolute;
      madeProgress = true;
    } else if (state.recvBytes != byteCount ||
               state.receiverOffset != absolute) {
      return invalid("replayed Direct DTE recv is not deterministic");
    }
    if (state.transferred) {
      if (state.payload.size() != byteCount)
        return invalid("completed Direct DTE payload size is inconsistent");
      std::memcpy(buffer.storage->bytes.data() + absolute, state.payload.data(),
                  byteCount);
    }
    return llvm::Error::success();
  }

  bool isComplete(const TransportEventKey &key) const override {
    auto found = events.find(key);
    return found != events.end() && found->second.transferred;
  }

  llvm::Error matchReadyEvents() {
    for (auto &[key, state] : events) {
      if (state.transferred || !state.sendIssued || !state.recvIssued)
        continue;
      if (state.sendBytes != state.recvBytes)
        return invalid("matched Direct DTE endpoints disagree on byte count");
      if (state.senderRemoteOffset != state.receiverOffset)
        return invalid(
            "matched Direct DTE endpoints disagree on receiver address");
      state.transferred = true;
      madeProgress = true;
    }
    return llvm::Error::success();
  }

  bool takeProgress() {
    bool result = madeProgress;
    madeProgress = false;
    return result;
  }

  void describePending(llvm::raw_ostream &stream) const {
    for (const auto &[key, state] : events) {
      if (state.transferred)
        continue;
      stream << " [" << key.source << "->" << key.destination
             << " communication=" << key.communication << " phase=" << key.phase
             << " round=" << key.round << " slice=" << key.payloadSlice
             << " control=[";
      llvm::interleaveComma(key.controlInstance, stream);
      stream << "]"
             << " send=" << (state.sendIssued ? "issued" : "missing")
             << " recv=" << (state.recvIssued ? "issued" : "missing") << "]";
    }
  }

private:
  struct EventState {
    bool sendIssued = false;
    bool recvIssued = false;
    bool transferred = false;
    uint64_t sendBytes = 0;
    uint64_t recvBytes = 0;
    int64_t senderRemoteOffset = -1;
    int64_t receiverOffset = -1;
    std::vector<uint8_t> payload;
  };

  int64_t currentRank = -1;
  bool madeProgress = false;
  std::set<TransportEventKey> replaySends;
  std::set<TransportEventKey> replayRecvs;
  std::map<TransportEventKey, EventState> events;
};

llvm::Expected<std::vector<ReferenceGlobalOutputBinding>>
reassembleOutputs(llvm::ArrayRef<const ReferenceProgram::Impl *> programs,
                  llvm::ArrayRef<ReferenceExecutionResult> rankResults) {
  if (programs.empty() || programs.size() != rankResults.size())
    return invalid("multi-rank output domain is incomplete");

  std::vector<const RankProgramBinding *> firstOutputs;
  for (const RankProgramBinding &binding : programs.front()->programBindings)
    if (binding.role == ProgramResourceRole::Output)
      firstOutputs.push_back(&binding);
  llvm::sort(firstOutputs, [](const RankProgramBinding *left,
                              const RankProgramBinding *right) {
    return left->programIndex < right->programIndex;
  });
  for (const ReferenceProgram::Impl *program : programs) {
    size_t outputCount = llvm::count_if(
        program->programBindings, [](const RankProgramBinding &binding) {
          return binding.role == ProgramResourceRole::Output;
        });
    if (outputCount != firstOutputs.size())
      return invalid("rank output binding domain disagrees across bundle");
  }

  std::vector<ReferenceGlobalOutputBinding> globals;
  for (const RankProgramBinding *first : firstOutputs) {
    std::vector<const RankProgramBinding *> bindings;
    std::vector<const ReferenceTensor *> tensors;
    for (size_t rank = 0; rank < programs.size(); ++rank) {
      const RankProgramBinding *binding = nullptr;
      for (const RankProgramBinding &candidate :
           programs[rank]->programBindings)
        if (candidate.role == ProgramResourceRole::Output &&
            candidate.programIndex == first->programIndex) {
          if (binding)
            return invalid("rank contains duplicate output binding index");
          binding = &candidate;
        }
      if (!binding || binding->name != first->name ||
          binding->dtype != first->dtype ||
          binding->globalShape != first->globalShape ||
          binding->distribution != first->distribution)
        return invalid("rank output metadata disagrees across bundle");
      const ReferenceTensor *tensor = nullptr;
      for (const ReferenceOutputBinding &output :
           rankResults[rank].getOutputs())
        if (output.index == first->programIndex) {
          if (tensor)
            return invalid("rank result contains duplicate output index");
          tensor = &output.tensor;
        }
      if (!tensor || tensor->getDType() != binding->dtype ||
          tensor->getShape() != llvm::ArrayRef<int64_t>(binding->localShape))
        return invalid("rank result disagrees with typed output slice");
      bindings.push_back(binding);
      tensors.push_back(tensor);
    }

    if (first->distribution == frontend::ProgramDistributionKind::Replicated) {
      if (first->localShape != first->globalShape)
        return invalid("replicated output does not cover global shape");
      bool requireExactReplicas =
          programs.front()->transportContract == TransportContract::None ||
          first->dtype != "f32";
      for (size_t rankIndex = 1;
           requireExactReplicas && rankIndex < tensors.size(); ++rankIndex) {
        llvm::ArrayRef<uint8_t> canonical = tensors.front()->getBytes();
        llvm::ArrayRef<uint8_t> candidate = tensors[rankIndex]->getBytes();
        if (candidate == canonical)
          continue;
        auto mismatch = std::mismatch(canonical.begin(), canonical.end(),
                                      candidate.begin(), candidate.end());
        size_t byteOffset = static_cast<size_t>(
            std::distance(canonical.begin(), mismatch.first));
        std::string message;
        llvm::raw_string_ostream stream(message);
        stream << "replicated output index " << first->programIndex << " rank "
               << programs[rankIndex]->logicalRank
               << " differs from rank 0 at byte " << byteOffset;
        if (first->dtype == "f32" && byteOffset / 4 < canonical.size() / 4) {
          size_t element = byteOffset / 4;
          float rankZero = 0.0f;
          float rankValue = 0.0f;
          std::memcpy(&rankZero, canonical.data() + element * 4, 4);
          std::memcpy(&rankValue, candidate.data() + element * 4, 4);
          stream << " (element " << element << ": rank0=" << rankZero
                 << ", rank=" << rankValue << ")";
        }
        stream.flush();
        return invalid(message);
      }
      auto global = ReferenceTensor::create(first->dtype, first->globalShape,
                                            tensors.front()->getBytes());
      if (!global)
        return global.takeError();
      globals.push_back({first->programIndex, first->name, std::move(*global)});
      continue;
    }

    auto elementBytes = getCompactByteCount(first->dtype, {1});
    auto globalBytes = getCompactByteCount(first->dtype, first->globalShape);
    if (!elementBytes || !globalBytes || *elementBytes <= 0)
      return invalid("partitioned output has unsupported tensor type");
    const int64_t globalElements = *globalBytes / *elementBytes;
    std::vector<uint8_t> bytes(*globalBytes, 0);
    std::vector<uint8_t> coverage(globalElements, 0);
    for (size_t rankIndex = 0; rankIndex < bindings.size(); ++rankIndex) {
      const RankProgramBinding *binding = bindings[rankIndex];
      const ReferenceTensor *tensor = tensors[rankIndex];
      const auto &slice = binding->slice;
      const size_t rank = first->globalShape.size();
      if (slice.logicalRank != programs[rankIndex]->logicalRank ||
          binding->localShape != slice.sizes || slice.offsets.size() != rank ||
          slice.sizes.size() != rank || slice.strides.size() != rank)
        return invalid("partitioned output slice has inconsistent rank");
      int64_t localLinear = 0;
      if (llvm::Error error = forEachLogicalIndex(
              binding->localShape,
              [&](llvm::ArrayRef<int64_t> local) -> llvm::Error {
                int64_t globalLinear = 0;
                for (size_t dim = 0; dim < rank; ++dim) {
                  if (slice.offsets[dim] < 0 || slice.strides[dim] <= 0 ||
                      local[dim] < 0 ||
                      local[dim] > (std::numeric_limits<int64_t>::max() -
                                    slice.offsets[dim]) /
                                       slice.strides[dim])
                    return invalid("partitioned output slice overflows");
                  int64_t coordinate =
                      slice.offsets[dim] + local[dim] * slice.strides[dim];
                  if (coordinate < 0 || coordinate >= first->globalShape[dim])
                    return invalid("partitioned output slice is out of bounds");
                  globalLinear =
                      globalLinear * first->globalShape[dim] + coordinate;
                }
                if (coverage[globalLinear]++)
                  return invalid("partitioned output slices overlap");
                std::memcpy(bytes.data() + globalLinear * *elementBytes,
                            tensor->getBytes().data() +
                                localLinear * *elementBytes,
                            *elementBytes);
                ++localLinear;
                return llvm::Error::success();
              }))
        return std::move(error);
    }
    if (llvm::is_contained(coverage, uint8_t{0}))
      return invalid("partitioned output slices do not cover global tensor");
    auto global =
        ReferenceTensor::create(first->dtype, first->globalShape, bytes);
    if (!global)
      return global.takeError();
    globals.push_back({first->programIndex, first->name, std::move(*global)});
  }
  return globals;
}

} // namespace

llvm::Expected<ReferenceExecutionResult>
interpretReferenceProgram(const ReferenceProgram::Impl &program,
                          llvm::ArrayRef<ReferenceInputBinding> inputs,
                          ReferenceExecutionOptions options) {
  return ProgramInterpreter(program, options).run(inputs);
}

llvm::Expected<ReferenceMultiRankExecutionResult> interpretReferencePrograms(
    llvm::ArrayRef<const ReferenceProgram::Impl *> programs,
    llvm::ArrayRef<ReferenceRankInvocation> invocations,
    ReferenceExecutionOptions options) {
  if (programs.size() <= 1 || programs.size() != invocations.size())
    return invalid("multi-rank invocation domain is incomplete");
  std::vector<const ReferenceRankInvocation *> byRank(programs.size(), nullptr);
  for (const ReferenceRankInvocation &invocation : invocations) {
    if (invocation.logicalRank < 0 ||
        invocation.logicalRank >= static_cast<int64_t>(programs.size()) ||
        byRank[invocation.logicalRank])
      return invalid("multi-rank invocation domain is not all-and-only");
    byRank[invocation.logicalRank] = &invocation;
  }
  for (auto [rank, program] : llvm::enumerate(programs)) {
    if (!program || program->logicalRank != static_cast<int64_t>(rank) ||
        program->transportContract != programs.front()->transportContract)
      return invalid("projected multi-rank domain is not canonical");
  }

  DeterministicTransportCoordinator coordinator;
  std::vector<std::optional<ReferenceExecutionResult>> completed(
      programs.size());
  std::vector<std::string> blocked(programs.size());
  size_t remaining = programs.size();
  while (remaining != 0) {
    coordinator.takeProgress();
    bool roundProgress = false;
    for (size_t rank = 0; rank < programs.size(); ++rank) {
      if (completed[rank])
        continue;
      coordinator.beginReplay(static_cast<int64_t>(rank));
      ProgramInterpreter interpreter(*programs[rank], options, &coordinator);
      auto attempt = interpreter.runUntilBlocked(byRank[rank]->inputs);
      if (!attempt)
        return attempt.takeError();
      blocked[rank] = attempt->blockedReason;
      if (attempt->result) {
        completed[rank] = std::move(*attempt->result);
        --remaining;
        roundProgress = true;
      }
    }
    if (llvm::Error error = coordinator.matchReadyEvents())
      return std::move(error);
    roundProgress |= coordinator.takeProgress();
    if (remaining != 0 && !roundProgress) {
      std::string message;
      llvm::raw_string_ostream stream(message);
      stream << "deterministic Direct DTE no-progress/deadlock:";
      for (size_t rank = 0; rank < blocked.size(); ++rank)
        if (!completed[rank])
          stream << " {rank=" << rank << " reason=" << blocked[rank] << "}";
      stream << " pending=";
      coordinator.describePending(stream);
      return invalid(stream.str());
    }
  }

  std::vector<ReferenceExecutionResult> rankResults;
  rankResults.reserve(completed.size());
  for (std::optional<ReferenceExecutionResult> &result : completed)
    rankResults.push_back(std::move(*result));
  auto globals = reassembleOutputs(programs, rankResults);
  if (!globals)
    return globals.takeError();
  return ReferenceMultiRankExecutionResultBuilder::make(std::move(rankResults),
                                                        std::move(*globals));
}

} // namespace wafer::compiler::reference_detail
