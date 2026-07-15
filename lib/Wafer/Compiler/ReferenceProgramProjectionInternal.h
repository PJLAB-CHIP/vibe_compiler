//===- ReferenceProgramProjectionInternal.h - Projection internals -*- C++
//-*-===//

#ifndef WAFER_COMPILER_REFERENCEPROGRAMPROJECTIONINTERNAL_H
#define WAFER_COMPILER_REFERENCEPROGRAMPROJECTIONINTERNAL_H

#include "ReferenceExecutorInternal.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace wafer::compiler::detail {
struct AcceptedCallClosure;
} // namespace wafer::compiler::detail

namespace wafer::compiler::reference_detail {

/// Projects accepted MLIR into the immutable reference-execution artifact.
/// This object is the sole owner of projection-local SSA and CFG identity.
class ProgramProjector {
public:
  explicit ProgramProjector(ReferenceProgram::Impl &program)
      : program(program) {}

  llvm::Error project(detail::AcceptedCallClosure &closure);

private:
  using Command = ReferenceProgram::Impl::Command;
  using CommandKind = ReferenceProgram::Impl::CommandKind;
  using ValueId = ReferenceProgram::Impl::ValueId;

  llvm::Expected<ValueId> define(mlir::Value value);
  llvm::Error predeclare(mlir::Value value);
  llvm::Expected<ValueId> use(mlir::Value value) const;

  llvm::Error projectBlock(mlir::Block &block,
                           ReferenceProgram::Impl::BlockProgram &output);
  llvm::Expected<bool>
  projectMemrefAndStructuredControl(mlir::Operation &operation,
                                    Command &command);
  llvm::Expected<bool> projectMovement(mlir::Operation &operation,
                                       Command &command);
  llvm::Expected<bool> projectNumeric(mlir::Operation &operation,
                                      Command &command);
  llvm::Expected<bool> projectDTEAndSync(mlir::Operation &operation,
                                         Command &command);
  llvm::Expected<bool> projectCFGAndReturn(mlir::Operation &operation,
                                           Command &command,
                                           bool &sawTerminator);

  llvm::Expected<mlir::MemRefType> requireF32Buffer(mlir::Value value,
                                                    llvm::StringRef purpose);
  llvm::Error projectMovementOperands(mlir::Value sourceValue,
                                      mlir::Value destValue, Command &command);
  llvm::Error
  projectDTEIssue(mlir::Value bufferValue, mlir::Value tokenValue, int64_t peer,
                  int64_t bytes, wafer::DTEMessageAttr message,
                  std::optional<wafer::DirectDTEBindingAttr> binding,
                  bool isSend, Command &command);
  llvm::Error projectStaticView(mlir::Value sourceValue,
                                mlir::Value resultValue,
                                mlir::MemRefType resultType, Command &command);

  llvm::Error validateProgramBindings(mlir::func::FuncOp entry);
  llvm::Error
  validateCFG(const ReferenceProgram::Impl::ControlFlowProgram &controlFlow);

  ReferenceProgram::Impl &program;
  llvm::DenseMap<mlir::Value, ValueId> values;
  llvm::DenseSet<mlir::Value> pendingDefinitions;
  llvm::DenseMap<mlir::Block *, uint32_t> blocks;
  llvm::DenseMap<mlir::Operation *, uint32_t> functionIds;
  ValueId nextValue = 0;
};

} // namespace wafer::compiler::reference_detail

#endif // WAFER_COMPILER_REFERENCEPROGRAMPROJECTIONINTERNAL_H
