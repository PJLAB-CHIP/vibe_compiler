//===- WaferRunBoardIO.h - Board invocation file bindings ------*- C++ -*-===//

#ifndef WAFER_TOOLS_WAFER_RUN_BOARD_IO_H
#define WAFER_TOOLS_WAFER_RUN_BOARD_IO_H

#include "Wafer/Runtime/BoardRuntime.h"
#include "Wafer/Runtime/PackageManifest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace wafer::runtime::cli {

struct ResourceFile {
  uint64_t resourceId = std::numeric_limits<uint64_t>::max();
  std::string path;
};

/// Validated file-backed invocation state. The request is ready for the board
/// executor; the remaining fields validate and publish its typed result.
struct BoardInvocationFilePlan {
  BoardRuntimeInvocationRequest request;
  llvm::DenseMap<uint64_t, std::vector<uint8_t>> expectedBytes;
  llvm::DenseMap<uint64_t, std::string> outputPaths;
  llvm::DenseMap<uint64_t, uint64_t> writableResourceBytes;
};

llvm::Expected<BoardInvocationFilePlan>
prepareBoardInvocationFiles(const PackageManifest &manifest,
                            BoardRuntimeInvocationRequest request,
                            llvm::ArrayRef<ResourceFile> resourceFiles,
                            llvm::ArrayRef<ResourceFile> expectedFiles,
                            llvm::ArrayRef<ResourceFile> outputFiles);

/// Rebinds one already prepared user invocation to another verified package
/// using the stable `(logical_rank, role, role_index)` resource identity.
/// Every host-visible resource contract must match exactly; internal
/// workspaces are intentionally outside this user-I/O projection.
llvm::Expected<BoardInvocationFilePlan>
remapBoardInvocationFilePlan(const BoardInvocationFilePlan &sourcePlan,
                             const PackageManifest &sourceManifest,
                             const PackageManifest &targetManifest);

/// Validates all-and-only writable outputs, exact byte counts, and every
/// supplied expected tensor without publishing any --output file.
llvm::Error validateBoardOutputs(llvm::ArrayRef<BoardRuntimeOutput> outputs,
                                 const BoardInvocationFilePlan &plan);

/// Validate the complete provider result before staging any captures. Each
/// capture is written to an adjacent temporary file, and publication begins
/// only after every capture has been staged successfully. Each target rename
/// is atomic; publication across multiple independent target paths is not a
/// group transaction, so a later rename failure can leave earlier targets
/// published.
llvm::Error
validateAndPublishBoardOutputs(llvm::ArrayRef<BoardRuntimeOutput> outputs,
                               const BoardInvocationFilePlan &plan);

} // namespace wafer::runtime::cli

#endif // WAFER_TOOLS_WAFER_RUN_BOARD_IO_H
