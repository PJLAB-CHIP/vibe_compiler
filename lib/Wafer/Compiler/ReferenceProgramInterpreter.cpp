//===- ReferenceProgramInterpreter.cpp - Reference execution facade ---===//

#include "ReferenceProgramInterpreterInternal.h"

namespace wafer::compiler::reference_detail {

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
  return interpretReferenceProgramsWithDTE(programs, invocations, options);
}

} // namespace wafer::compiler::reference_detail
