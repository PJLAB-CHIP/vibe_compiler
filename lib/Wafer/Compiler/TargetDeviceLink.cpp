//===- TargetDeviceLink.cpp - Target LLVM emission and device link -------===//

#include "TargetArtifactInternal.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

namespace wafer::compiler::detail {

llvm::Error writeLLVMIR(const llvm::Module &module, llvm::StringRef path) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to open target LLVM IR");
  module.print(output, nullptr);
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to write target LLVM IR");
  return llvm::Error::success();
}

llvm::Error runDeviceLink(const TargetToolchain &toolchain,
                          llvm::StringRef llvmIR, llvm::StringRef module,
                          llvm::StringRef object, llvm::StringRef crtObject) {
  std::string python = toolchain.getPythonExecutable().str();
  std::string script = toolchain.getDeviceLinkerScript().str();
  std::string llvmIRStorage = llvmIR.str();
  std::string moduleStorage = module.str();
  std::string objectStorage = object.str();
  std::string crtObjectStorage = crtObject.str();
  llvm::SmallVector<llvm::StringRef, 11> arguments = {python,
                                                      script,
                                                      "--llvm-ir",
                                                      llvmIRStorage,
                                                      "--output",
                                                      moduleStorage,
                                                      "--object-output",
                                                      objectStorage,
                                                      "--crt-object-output",
                                                      crtObjectStorage};
  int exitCode = llvm::sys::ExecuteAndWait(python, arguments);
  if (exitCode == 0)
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::errc::io_error, "device link failed with exit code %d", exitCode);
}

} // namespace wafer::compiler::detail
