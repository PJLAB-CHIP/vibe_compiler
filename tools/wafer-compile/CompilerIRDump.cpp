//===- CompilerIRDump.cpp - Compiler IR inspection output ----------------===//

#include "DriverInternal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace wafer::compile_driver {
namespace {

bool createDirectory(llvm::StringRef path, llvm::raw_ostream &diagnostics) {
  if (std::error_code error = llvm::sys::fs::create_directories(path)) {
    diagnostics << "wafer-compile: failed to create compiler IR dump "
                   "directory '"
                << path << "': " << error.message() << "\n";
    return false;
  }
  return true;
}

llvm::SmallString<256> tilePath(llvm::StringRef directory, TileId tileId,
                                llvm::StringRef extension) {
  llvm::SmallString<32> filename;
  llvm::raw_svector_ostream stream(filename);
  stream << "tile_" << llvm::formatv("{0:05}", tileId.getValue()) << extension;
  llvm::SmallString<256> path(directory);
  llvm::sys::path::append(path, filename);
  return path;
}

template <typename Print>
bool writeIRFile(llvm::StringRef path, Print print,
                 llvm::raw_ostream &diagnostics) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    diagnostics << "wafer-compile: failed to create compiler IR dump '" << path
                << "': " << error.message() << "\n";
    return false;
  }
  print(output);
  output << "\n";
  output.close();
  if (output.has_error()) {
    diagnostics << "wafer-compile: failed to write compiler IR dump '" << path
                << "'\n";
    return false;
  }
  return true;
}

} // namespace

bool dumpCompilerIR(llvm::StringRef destination,
                    const wafer::compiler::CompiledProgram &compiledProgram,
                    llvm::raw_ostream &diagnostics) {
  if (destination.empty()) {
    diagnostics << "wafer-compile: --dump-compiler-ir must not be empty\n";
    return false;
  }
  if (llvm::sys::fs::exists(destination)) {
    diagnostics << "wafer-compile: refusing to replace existing compiler IR "
                   "dump directory: '"
                << destination << "'\n";
    return false;
  }

  llvm::SmallString<256> instructionDirectory(destination);
  llvm::sys::path::append(instructionDirectory, "instruction");
  llvm::SmallString<256> tileDataflowDirectory(destination);
  llvm::sys::path::append(tileDataflowDirectory, "tile-dataflow");
  llvm::SmallString<256> targetLLVMDirectory(destination);
  llvm::sys::path::append(targetLLVMDirectory, "target-llvm");
  if (!createDirectory(tileDataflowDirectory, diagnostics) ||
      !createDirectory(instructionDirectory, diagnostics) ||
      !createDirectory(targetLLVMDirectory, diagnostics))
    return false;

  const auto &tiles =
      compiledProgram.getDeviceExecutable().getTileExecutables();
  const auto &targetModules =
      compiledProgram.getTargetLLVMModules().getModules();
  const auto &irTrace = compiledProgram.getIRTrace().tiles;
  if (tiles.size() != targetModules.size() || tiles.size() != irTrace.size()) {
    diagnostics << "wafer-compile: compiler IR dump Tile domains differ\n";
    return false;
  }
  for (size_t index = 0; index < tiles.size(); ++index) {
    const wafer::compiler::TileExecutable &tile = tiles[index];
    const wafer::compiler::TargetLLVMModule &targetModule =
        targetModules[index];
    const wafer::compiler::TileIRTrace &tileTrace = irTrace[index];
    if (tile.getCardId() != targetModule.getCardId() ||
        tile.getTileId() != targetModule.getTileId() ||
        tile.getLaunchSlotId() != targetModule.getLaunchSlotId() ||
        tile.getCardId() != tileTrace.cardId ||
        tile.getTileId() != tileTrace.tileId ||
        tile.getLaunchSlotId() != tileTrace.launchSlotId) {
      diagnostics << "wafer-compile: compiler IR dump Tile order differs\n";
      return false;
    }
    llvm::SmallString<256> tileDataflowPath =
        tilePath(tileDataflowDirectory, tile.getTileId(), ".mlir");
    if (!writeIRFile(
            tileDataflowPath,
            [&](llvm::raw_ostream &output) {
              output << tileTrace.tileDataflowIR;
            },
            diagnostics))
      return false;
    llvm::SmallString<256> instructionPath =
        tilePath(instructionDirectory, tile.getTileId(), ".mlir");
    if (!writeIRFile(
            instructionPath,
            [&](llvm::raw_ostream &output) { tile.getModule().print(output); },
            diagnostics))
      return false;

    llvm::SmallString<256> targetPath =
        tilePath(targetLLVMDirectory, tile.getTileId(), ".ll");
    if (!writeIRFile(
            targetPath,
            [&](llvm::raw_ostream &output) {
              targetModule.getModule().print(output, nullptr);
            },
            diagnostics))
      return false;
  }
  return true;
}

} // namespace wafer::compile_driver
