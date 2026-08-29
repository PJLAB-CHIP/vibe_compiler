//===- CompilationResultInternal.h - Driver result construction -*- C++ -*-===//

#ifndef WAFER_DRIVER_COMPILATIONRESULTINTERNAL_H
#define WAFER_DRIVER_COMPILATIONRESULTINTERNAL_H

#include "Wafer/Driver/CompilationResult.h"

#include <optional>
#include <utility>

namespace wafer::compiler {

struct CompilationResultBuilder {
  static CompilationResult
  make(ExecutionConfig executionConfig, ExecutablePackage package,
       std::optional<ProfileInstrumentationProduct> profileInstrumentation) {
    return CompilationResult(executionConfig, std::move(package),
                             std::move(profileInstrumentation));
  }
};

} // namespace wafer::compiler

#endif // WAFER_DRIVER_COMPILATIONRESULTINTERNAL_H
