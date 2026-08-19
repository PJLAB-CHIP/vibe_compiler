//===- WaferRunBoardIO.h - Board invocation file bindings ------*- C++ -*-===//

#ifndef WAFER_TOOLS_WAFER_RUN_BOARD_IO_H
#define WAFER_TOOLS_WAFER_RUN_BOARD_IO_H

#include "Wafer/Runtime/Board/BoardRuntime.h"
#include "Wafer/Package/Manifest/PackageManifest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace wafer::runtime::cli {

struct PortFile {
  uint64_t portId = std::numeric_limits<uint64_t>::max();
  std::string path;
};

enum class BoardOutputComparisonKind {
  Exact,
  RelaxedF16,
};

inline constexpr double kRelaxedF16AbsoluteTolerance = 0.0009765625;
inline constexpr double kRelaxedF16RelativeTolerance = 0.001;
inline constexpr uint32_t kRelaxedF16MaximumUlp = 1;

/// Validated file-backed invocation state. The request is ready for the board
/// executor; the remaining fields validate and return its typed result.
struct BoardInvocationFilePlan {
  BoardRuntimeInvocationRequest request;
  llvm::DenseMap<uint64_t, std::vector<uint8_t>> expectedBytes;
  llvm::DenseMap<uint64_t, BoardOutputComparisonKind> expectedComparisons;
  llvm::DenseMap<uint64_t, std::string> outputPaths;
  llvm::DenseMap<uint64_t, uint64_t> outputBytes;
};

llvm::Expected<BoardInvocationFilePlan> prepareBoardInvocationFiles(
    const PackageManifest &manifest, BoardRuntimeInvocationRequest request,
    llvm::ArrayRef<PortFile> inputFiles, llvm::ArrayRef<PortFile> expectedFiles,
    llvm::ArrayRef<PortFile> outputFiles,
    llvm::ArrayRef<PortFile> relaxedF16ExpectedFiles = {});

/// Rebinds one already prepared user invocation to another verified package
/// using the stable external port identity (input/output role index).
/// Every external port contract must match exactly; entry-local requirements
/// are intentionally outside this user-I/O projection.
llvm::Expected<BoardInvocationFilePlan>
remapBoardInvocationFilePlan(const BoardInvocationFilePlan &sourcePlan,
                             const PackageManifest &sourceManifest,
                             const PackageManifest &targetManifest);

/// Validates all-and-only outputs, exact byte counts, and every supplied
/// expected tensor without writing any --output file.
llvm::Error validateBoardOutputs(llvm::ArrayRef<BoardRuntimeOutput> outputs,
                                 const BoardInvocationFilePlan &plan);

/// Validate the complete provider result before staging any captures. Each
/// capture is written to an adjacent temporary file, and destination
/// replacement begins only after every capture has been staged successfully.
/// Each rename is atomic; replacement across independent target paths is not
/// atomic as a group, so a later rename failure can leave earlier files
/// replaced.
llvm::Error
validateAndWriteBoardOutputs(llvm::ArrayRef<BoardRuntimeOutput> outputs,
                             const BoardInvocationFilePlan &plan);

} // namespace wafer::runtime::cli

#endif // WAFER_TOOLS_WAFER_RUN_BOARD_IO_H
