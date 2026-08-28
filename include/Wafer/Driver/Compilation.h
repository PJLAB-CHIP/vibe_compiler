//===- Compilation.h - Typed Wafer compiler request ------------*- C++ -*-===//

#ifndef WAFER_DRIVER_COMPILATION_H
#define WAFER_DRIVER_COMPILATION_H

#include "Wafer/CodeGen/DeviceExecutable.h"
#include "Wafer/Support/OptimizationConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace wafer::compiler {

class TargetToolchain;

/// Invocation-local diagnostic policy. Detailed timing never changes source
/// semantics, candidate verification, selection, or written outputs.
enum class CompilationTimingMode { Disabled, Detailed };

/// Move-only semantic input to the compiler driver.
class CompilationRequest {
public:
  static llvm::Expected<CompilationRequest>
  create(llvm::StringRef sourceProgramDirectory,
         ExecutionConfig executionConfig);

  CompilationRequest(CompilationRequest &&) = default;
  CompilationRequest &operator=(CompilationRequest &&) = default;
  CompilationRequest(const CompilationRequest &) = delete;
  CompilationRequest &operator=(const CompilationRequest &) = delete;

  llvm::StringRef getSourceProgramDirectory() const {
    return sourceProgramDirectory;
  }
  const ExecutionConfig &getExecutionConfig() const { return executionConfig; }

private:
  CompilationRequest(llvm::StringRef sourceProgramDirectory,
                     ExecutionConfig executionConfig)
      : sourceProgramDirectory(sourceProgramDirectory.str()),
        executionConfig(executionConfig) {}

  std::string sourceProgramDirectory;
  ExecutionConfig executionConfig;
};

/// Typed orchestration intent for one production compilation transaction.
class CompilationOptions {
public:
  static CompilationOptions
  standard(OptimizationConfig optimizations = OptimizationConfig::none(),
           CompilationTimingMode timing = CompilationTimingMode::Disabled) {
    return CompilationOptions(/*profileInstrumentation=*/false, optimizations,
                              timing);
  }

  static llvm::Expected<CompilationOptions>
  profile(const ExecutionConfig &executionConfig,
          OptimizationConfig optimizations = OptimizationConfig::none(),
          CompilationTimingMode timing = CompilationTimingMode::Disabled);

  bool shouldProduceProfileInstrumentation() const {
    return profileInstrumentation;
  }
  OptimizationConfig getOptimizationConfig() const { return optimizations; }
  bool shouldReportDetailedTiming() const {
    return timing == CompilationTimingMode::Detailed;
  }

private:
  explicit CompilationOptions(bool profileInstrumentation,
                              OptimizationConfig optimizations,
                              CompilationTimingMode timing)
      : profileInstrumentation(profileInstrumentation),
        optimizations(optimizations), timing(timing) {}

  bool profileInstrumentation;
  OptimizationConfig optimizations;
  CompilationTimingMode timing;
};

/// Host-boundary classification of one compilation transaction failure.
enum class CompilationStage {
  SourceVerification,
  SpmdPartitioning,
  TensorProgramPreparation,
  ExecutableCompilation,
  TargetCodeGeneration,
  PackageAssembly,
  PackageCommit,
};

llvm::StringRef stringifyCompilationStage(CompilationStage stage);

class CompilationFailure final : public llvm::ErrorInfo<CompilationFailure> {
public:
  static char ID;

  explicit CompilationFailure(CompilationStage stage) : stage(stage) {}

  CompilationStage getStage() const { return stage; }

  void log(llvm::raw_ostream &stream) const override {
    stream << "compilation failed at the " << stringifyCompilationStage(stage)
           << " stage";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::errc::operation_not_permitted;
  }

private:
  CompilationStage stage;
};

} // namespace wafer::compiler

#endif // WAFER_DRIVER_COMPILATION_H
