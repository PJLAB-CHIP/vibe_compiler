//===- CompiledProgram.h - Retained compiler inspection product -*- C++ -*-===//

#ifndef WAFER_DRIVER_COMPILEDPROGRAM_H
#define WAFER_DRIVER_COMPILEDPROGRAM_H

#include "Wafer/CodeGen/TargetCodeGen.h"
#include "Wafer/Driver/Compilation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <utility>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

/// Owner-backed result retained by same-invocation inspection consumers.
class CompiledProgram {
public:
  CompiledProgram(CompiledProgram &&) = default;
  CompiledProgram &operator=(CompiledProgram &&) = default;
  CompiledProgram(const CompiledProgram &) = delete;
  CompiledProgram &operator=(const CompiledProgram &) = delete;

  const DeviceExecutable &getDeviceExecutable() const {
    return deviceExecutable;
  }
  const TargetLLVMModules &getTargetLLVMModules() const {
    return targetLLVMModules;
  }
  const CompilationIRTrace &getIRTrace() const { return irTrace; }

private:
  friend llvm::Expected<CompiledProgram> compileProgramWithTargetLLVMModules(
      CompilationRequest request, llvm::StringRef outputDirectory,
      llvm::StringRef xlaSpmdPartitionerHelper,
      const TargetToolchain &targetToolchain, CompilationOptions options,
      llvm::raw_ostream &diagnostics);

  CompiledProgram(DeviceExecutable deviceExecutable,
                  TargetLLVMModules targetLLVMModules,
                  CompilationIRTrace irTrace)
      : deviceExecutable(std::move(deviceExecutable)),
        targetLLVMModules(std::move(targetLLVMModules)),
        irTrace(std::move(irTrace)) {}

  DeviceExecutable deviceExecutable;
  TargetLLVMModules targetLLVMModules;
  CompilationIRTrace irTrace;
};

/// Runs the production transaction while retaining its accepted CodeGen
/// products for same-invocation qualification and IR inspection.
llvm::Expected<CompiledProgram> compileProgramWithTargetLLVMModules(
    CompilationRequest request, llvm::StringRef outputDirectory,
    llvm::StringRef xlaSpmdPartitionerHelper,
    const TargetToolchain &targetToolchain, CompilationOptions options,
    llvm::raw_ostream &diagnostics);

} // namespace wafer::compiler

#endif // WAFER_DRIVER_COMPILEDPROGRAM_H
