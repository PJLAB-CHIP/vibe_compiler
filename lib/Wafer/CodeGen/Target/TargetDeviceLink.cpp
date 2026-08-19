//===- TargetDeviceLink.cpp - Target LLVM emission and device link -------===//

#include "Wafer/CodeGen/Target/TargetCodeGenInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

namespace wafer::compiler::detail {

namespace {

} // namespace

llvm::Error writeTargetLLVMIR(const llvm::Module &module,
                              llvm::StringRef path) {
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
                          llvm::StringRef object, llvm::StringRef crtObject,
                          const RuntimeLaunchContract &runtimeLaunchContract,
                          ProfileCaptureKind profileCapture) {
  std::string python = toolchain.getPythonExecutable().str();
  std::string script = toolchain.getDeviceLinkerScript().str();
  std::string clangXX = toolchain.getLLVMClangXX().str();
  std::string tx8DepsRoot = toolchain.getTx8DepsRoot().str();
  std::string waferIncludeDir = toolchain.getWaferIncludeDir().str();
  std::string waferCrtSource = toolchain.getWaferCrtSource().str();
  std::string waferCrtIncludeDir = toolchain.getWaferCrtIncludeDir().str();
  std::string llvmIRStorage = llvmIR.str();
  std::string moduleStorage = module.str();
  std::string objectStorage = object.str();
  std::string crtObjectStorage = crtObject.str();
  std::string loaderABI = "tx8-kcore-loader";
  const KernelRuntimeLaunchContract &kernel = runtimeLaunchContract.getKernel();
  if (kernel.form == KernelLaunchForm::Grid)
    loaderABI = "tx8-kcore-loader-grid";
  else if (kernel.form == KernelLaunchForm::Cluster)
    loaderABI = "tx8-kcore-loader-cluster";
  llvm::SmallVector<llvm::StringRef, 26> arguments = {
      python,
      script,
      "--llvm-ir",
      llvmIRStorage,
      "--llvm-clangxx",
      clangXX,
      "--tx8-deps-root",
      tx8DepsRoot,
      "--wafer-crt-source",
      waferCrtSource,
      "--wafer-crt-include-dir",
      waferCrtIncludeDir,
      "--wafer-include-dir",
      waferIncludeDir,
      "--output",
      moduleStorage,
      "--object-output",
      objectStorage,
      "--crt-object-output",
      crtObjectStorage,
      "--loader-abi",
      loaderABI};
  std::string captureStorage;
  if (profileCapture != ProfileCaptureKind::None) {
    captureStorage = stringifyProfileCaptureKind(profileCapture).str();
    arguments.push_back("--profile-capture");
    arguments.push_back(captureStorage);
  }
  int exitCode = llvm::sys::ExecuteAndWait(python, arguments);
  if (exitCode == 0)
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::errc::io_error, "device link failed with exit code %d", exitCode);
}

} // namespace wafer::compiler::detail
