//===- TxBoardRuntime.h - TX public-runtime board provider -----*- C++ -*-===//

#ifndef WAFER_RUNTIME_TXBOARDRUNTIME_H
#define WAFER_RUNTIME_TXBOARDRUNTIME_H

#include "Wafer/Runtime/Board/BoardRuntime.h"

#include <memory>

namespace wafer::runtime {

/// Creates the provider backed by the public TX runtime API selected by the
/// board-enabled build. Availability is still checked at invocation time.
llvm::Expected<std::unique_ptr<BoardRuntimeDriver>>
createTxBoardRuntimeDriver(llvm::StringRef expectedRuntimeLibraryDigest);

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_TXBOARDRUNTIME_H
