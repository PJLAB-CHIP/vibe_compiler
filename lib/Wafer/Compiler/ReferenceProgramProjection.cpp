//===- ReferenceProgramProjection.cpp - Accepted IR projection ----------===//

#include "ReferenceProgramProjectionInternal.h"

#include "AcceptedCallClosure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <string>
#include <utility>

namespace wafer::compiler::reference_detail {

llvm::Error ProgramProjector::project(detail::AcceptedCallClosure &closure) {
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
    functionIds[function.getOperation()] = static_cast<uint32_t>(functionIndex);
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
        return unsupported(
            ("function @" + function.getSymName() + " argument is not a memref")
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

llvm::Expected<ProgramProjector::ValueId>
ProgramProjector::define(mlir::Value value) {
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

llvm::Error ProgramProjector::predeclare(mlir::Value value) {
  if (values.count(value))
    return unsupported("SSA value is declared more than once");
  if (nextValue == std::numeric_limits<ValueId>::max())
    return unsupported("reference value id space is exhausted");
  values[value] = nextValue++;
  pendingDefinitions.insert(value);
  return llvm::Error::success();
}

llvm::Expected<ProgramProjector::ValueId>
ProgramProjector::use(mlir::Value value) const {
  auto found = values.find(value);
  if (found == values.end())
    return unsupported("operand has no projected SSA definition");
  return found->second;
}

llvm::Error
ProgramProjector::projectBlock(mlir::Block &block,
                               ReferenceProgram::Impl::BlockProgram &output) {
  bool sawTerminator = false;
  for (mlir::Operation &operation : block) {
    if (sawTerminator)
      return unsupported("operation follows a projected terminator");

    Command command;
    auto memrefControl = projectMemrefAndStructuredControl(operation, command);
    if (!memrefControl)
      return memrefControl.takeError();
    bool projected = *memrefControl;

    if (!projected) {
      auto movement = projectMovement(operation, command);
      if (!movement)
        return movement.takeError();
      projected = *movement;
    }
    if (!projected) {
      auto numeric = projectNumeric(operation, command);
      if (!numeric)
        return numeric.takeError();
      projected = *numeric;
    }
    if (!projected) {
      auto dte = projectDTEAndSync(operation, command);
      if (!dte)
        return dte.takeError();
      projected = *dte;
    }
    if (!projected) {
      auto cfg = projectCFGAndReturn(operation, command, sawTerminator);
      if (!cfg)
        return cfg.takeError();
      projected = *cfg;
    }
    if (!projected)
      return unsupported(
          ("unsupported operation " + operation.getName().getStringRef())
              .str());

    ++program.projectedOperationCount;
    output.commands.push_back(std::move(command));
  }
  if (!sawTerminator)
    return unsupported("block has no return-like terminator");
  return llvm::Error::success();
}

llvm::Error
ProgramProjector::validateProgramBindings(mlir::func::FuncOp entry) {
  llvm::SmallVector<bool> argumentUsed(entry.getNumArguments(), false);
  llvm::SmallVector<bool> resultUsed(entry.getNumResults(), false);
  llvm::SmallVector<int64_t> userInputIndices;
  llvm::SmallVector<int64_t> outputIndices;
  for (const RankProgramBinding &binding : program.programBindings) {
    llvm::SmallVectorImpl<bool> &domain =
        binding.role == ProgramResourceRole::Output ? resultUsed : argumentUsed;
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

llvm::Error ProgramProjector::validateCFG(
    const ReferenceProgram::Impl::ControlFlowProgram &controlFlow) {
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
      if (llvm::Error error =
              visitSuccessor(terminator.falseSuccessor, terminator.falseInputs))
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
