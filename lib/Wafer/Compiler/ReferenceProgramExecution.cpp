//===- ReferenceProgramExecution.cpp - Interpreter coordination ------===//

#include "ReferenceProgramInterpreterInternal.h"

#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace wafer::compiler::reference_detail {

ProgramInterpreter::ProgramInterpreter(const ReferenceProgram::Impl &program,
                                       ReferenceExecutionOptions options,
                                       TransportCoordinator *transport)
    : program(program), spmArena(std::make_shared<Storage>()),
      ddrArena(std::make_shared<Storage>()), options(options),
      stochasticState(options.stochasticSeed.value_or(0)),
      transport(transport) {}

llvm::Expected<ReferenceExecutionResult>
ProgramInterpreter::run(llvm::ArrayRef<ReferenceInputBinding> inputs) {
  auto attempt = runUntilBlocked(inputs);
  if (!attempt)
    return attempt.takeError();
  if (!attempt->result)
    return invalid("single-rank reference execution blocked on transport");
  return std::move(*attempt->result);
}

llvm::Expected<ProgramRunAttempt> ProgramInterpreter::runUntilBlocked(
    llvm::ArrayRef<ReferenceInputBinding> inputs) {
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

llvm::Expected<std::vector<ValueId>> ProgramInterpreter::executeControlFlow(
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

llvm::Expected<std::vector<ValueId>> ProgramInterpreter::executeStructuredBlock(
    const ReferenceProgram::Impl::BlockProgram &block) {
  auto transfer = executeBlock(block);
  if (!transfer)
    return transfer.takeError();
  if (transfer->kind != TransferKind::Return)
    return invalid("structured region produced a CFG branch");
  return std::move(transfer->inputs);
}

llvm::Expected<ControlTransfer> ProgramInterpreter::executeBlock(
    const ReferenceProgram::Impl::BlockProgram &block) {
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
                 ? ControlTransfer{TransferKind::Branch, command.trueSuccessor,
                                   command.trueInputs}
                 : ControlTransfer{TransferKind::Branch, command.falseSuccessor,
                                   command.falseInputs};
    }
    case CommandKind::Return:
      return ControlTransfer{TransferKind::Return, 0, command.inputs};
    }
  }
  return invalid("projected block has no terminator command");
}

} // namespace wafer::compiler::reference_detail
