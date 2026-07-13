//===- TargetArtifact.cpp - Typed all-rank target artifacts --------------===//

#include "TargetArtifactInternal.h"

#include "AcceptedCallClosure.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Support/TargetPolicy.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace wafer::compiler {

struct TargetArtifactBundleBuilder {
  static VerifiedTargetModule
  makeModule(int64_t logicalRank, llvm::StringRef entrySymbol,
             llvm::StringRef relativePath, llvm::StringRef contentDigest,
             std::vector<KernelABISlot> kernelABISlots) {
    return VerifiedTargetModule(logicalRank, entrySymbol, relativePath,
                                contentDigest, std::move(kernelABISlots));
  }

  static TargetArtifactBundle
  makeBundle(llvm::StringRef rootDirectory, ExecutionConfig executionConfig,
             std::vector<VerifiedTargetModule> modules) {
    return TargetArtifactBundle(rootDirectory, executionConfig,
                                std::move(modules));
  }
};

llvm::Expected<TargetToolchain>
TargetToolchain::create(llvm::StringRef pythonExecutable,
                        llvm::StringRef deviceLinkerScript) {
  if (pythonExecutable.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Python executable must not be empty");
  if (deviceLinkerScript.empty())
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "device linker script must not be empty");
  return TargetToolchain(pythonExecutable, deviceLinkerScript);
}

namespace {

struct PreparedTargetRank {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<KernelABISlot> slots;
  int64_t defaultDDRArenaArgumentIndex = -1;
};

llvm::Error fail(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-compile: " << message << "\n";
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

bool pathEntryExists(llvm::StringRef path) {
  llvm::sys::fs::file_type type =
      llvm::sys::fs::get_file_type(path, /*Follow=*/false);
  return type != llvm::sys::fs::file_type::file_not_found &&
         type != llvm::sys::fs::file_type::status_error;
}

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

bool isExecutableFile(llvm::StringRef path) {
  return isRegularFile(path) &&
         !llvm::sys::fs::access(path, llvm::sys::fs::AccessMode::Execute);
}

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

KernelABISlotRole getKernelRole(ProgramResourceRole role) {
  switch (role) {
  case ProgramResourceRole::UserInput:
    return KernelABISlotRole::UserInput;
  case ProgramResourceRole::Parameter:
    return KernelABISlotRole::Parameter;
  case ProgramResourceRole::Constant:
    return KernelABISlotRole::Constant;
  case ProgramResourceRole::Output:
    return KernelABISlotRole::Output;
  }
  llvm_unreachable("unknown program resource role");
}

mlir::FailureOr<int64_t> getPhysicalBytes(mlir::Type type,
                                          mlir::Operation *anchor) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || !isWaferDDRMemRefType(memrefType)) {
    anchor->emitError()
        << "target_abi_mismatch: kernel resource is not a Wafer DDR memref";
    return mlir::failure();
  }
  std::optional<WaferPhysicalTensorInfo> info =
      computeWaferPhysicalTensorInfo(memrefType);
  if (!info || info->physicalBytes < 0) {
    anchor->emitError()
        << "target_abi_mismatch: cannot derive static resource byte size";
    return mlir::failure();
  }
  return info->physicalBytes;
}

mlir::Value resolveOutputAllocation(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (value) {
    if (auto blockArgument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Block *owner = blockArgument.getOwner();
      auto tileRegion =
          owner ? mlir::dyn_cast_or_null<TileRegionOp>(owner->getParentOp())
                : TileRegionOp{};
      if (!tileRegion || owner != &tileRegion.getBody().front() ||
          blockArgument.getArgNumber() >= tileRegion.getInputs().size())
        return value;
      value = tileRegion.getInputs()[blockArgument.getArgNumber()];
      continue;
    }

    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return value;
    if (mlir::isa<mlir::memref::AllocOp>(definition))
      return value;
    if (auto tileRegion = mlir::dyn_cast<TileRegionOp>(definition)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      auto yield = mlir::dyn_cast<TileYieldOp>(
          tileRegion.getBody().front().getTerminator());
      if (!result || !yield ||
          result.getResultNumber() >= yield.getNumOperands())
        return value;
      value = yield.getOperand(result.getResultNumber());
      continue;
    }
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(definition)) {
      value = view.getViewSource();
      continue;
    }
    return value;
  }
  return value;
}

mlir::FailureOr<PreparedTargetRank>
prepareTargetABI(const RankExecutable &rankExecutable) {
  PreparedTargetRank prepared;
  prepared.module = rankExecutable.getModule().clone();
  const int64_t defaultDDRAlignment =
      getDefaultWaferTargetPolicy().memory.ddrAlignmentBytes;

  llvm::Expected<detail::AcceptedCallClosure> closure =
      detail::analyzeAcceptedCallClosure(*prepared.module,
                                         rankExecutable.getEntrySymbol());
  if (!closure) {
    prepared.module->emitError()
        << "target_abi_mismatch: " << llvm::toString(closure.takeError());
    return mlir::failure();
  }
  mlir::func::FuncOp function = closure->entry;

  unsigned originalArgumentCount = function.getNumArguments();
  unsigned resultCount = function.getFunctionType().getNumResults();
  std::vector<const RankProgramBinding *> argumentBindings(
      originalArgumentCount, nullptr);
  std::vector<const RankProgramBinding *> outputBindings(resultCount, nullptr);
  for (const RankProgramBinding &binding :
       rankExecutable.getProgramBindings()) {
    std::vector<const RankProgramBinding *> &domain =
        binding.role == ProgramResourceRole::Output ? outputBindings
                                                    : argumentBindings;
    if (binding.index < 0 ||
        binding.index >= static_cast<int64_t>(domain.size()) ||
        domain[binding.index]) {
      function.emitError()
          << "target_abi_mismatch: resource bindings do not form an exact "
             "function boundary";
      return mlir::failure();
    }
    domain[binding.index] = &binding;
  }
  if (llvm::is_contained(argumentBindings, nullptr) ||
      llvm::is_contained(outputBindings, nullptr)) {
    function.emitError()
        << "target_abi_mismatch: resource bindings do not cover every "
           "argument and result";
    return mlir::failure();
  }

  prepared.slots.reserve(originalArgumentCount + resultCount + 1);
  auto appendSlot = [&](const RankProgramBinding &binding, mlir::Type type,
                        KernelABISlotRole role) -> mlir::LogicalResult {
    mlir::FailureOr<int64_t> byteSize = getPhysicalBytes(type, function);
    if (mlir::failed(byteSize))
      return mlir::failure();
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()), role,
                              binding.index, binding.name, binding.dtype,
                              binding.localShape, *byteSize,
                              defaultDDRAlignment});
    return mlir::success();
  };

  for (unsigned index = 0; index < originalArgumentCount; ++index)
    if (mlir::failed(appendSlot(*argumentBindings[index],
                                function.getArgument(index).getType(),
                                getKernelRole(argumentBindings[index]->role))))
      return mlir::failure();

  llvm::SmallVector<mlir::func::ReturnOp, 2> returns;
  function.walk([&](mlir::func::ReturnOp returnOp) {
    if (returnOp->getParentOfType<mlir::func::FuncOp>() == function)
      returns.push_back(returnOp);
  });
  if (returns.size() != 1 || returns.front().getNumOperands() != resultCount) {
    function.emitError()
        << "target_abi_mismatch: current output binding requires exactly one "
           "complete entry return";
    return mlir::failure();
  }

  for (unsigned index = 0; index < resultCount; ++index) {
    mlir::Value root =
        resolveOutputAllocation(returns.front().getOperand(index));
    auto allocation = root.getDefiningOp<mlir::memref::AllocOp>();
    if (!allocation || !isWaferDDRMemRefType(allocation.getType()) ||
        !allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName)) {
      returns.front().emitError()
          << "target_abi_mismatch: output #" << index
          << " is not backed by one accepted compiler-managed DDR root";
      return mlir::failure();
    }
    mlir::Type resultType = function.getFunctionType().getResult(index);
    if (allocation.getType() != resultType) {
      allocation.emitError()
          << "target_abi_mismatch: output root type differs from result type";
      return mlir::failure();
    }
    unsigned outputArgumentIndex = function.getNumArguments();
    function.insertArgument(outputArgumentIndex, resultType,
                            mlir::DictionaryAttr{}, function.getLoc());
    mlir::BlockArgument outputArgument =
        function.getArgument(outputArgumentIndex);
    allocation.getResult().replaceAllUsesWith(outputArgument);
    allocation.erase();
    if (mlir::failed(appendSlot(*outputBindings[index], resultType,
                                KernelABISlotRole::Output)))
      return mlir::failure();
  }

  int64_t arenaBytes = 0;
  int64_t arenaAlignment = defaultDDRAlignment;
  mlir::LogicalResult arenaValid = mlir::success();
  function.walk([&](mlir::memref::AllocOp allocation) {
    if (mlir::failed(arenaValid) || !isWaferDDRMemRefType(allocation.getType()))
      return;
    auto offset =
        allocation->getAttrOfType<DDROffsetAttr>(kWaferDDROffsetAttrName);
    std::optional<WaferPhysicalTensorInfo> info =
        computeWaferPhysicalTensorInfo(allocation.getType());
    int64_t end = 0;
    if (!offset || !info || info->physicalBytes < 0 ||
        !checkedAdd(offset.getOffset(), info->physicalBytes, end)) {
      arenaValid = allocation.emitError()
                   << "target_abi_mismatch: cannot derive default DDR arena "
                      "high-water bytes";
      return;
    }
    arenaBytes = std::max(arenaBytes, end);
    if (auto alignment = allocation.getAlignmentAttr())
      arenaAlignment = std::max(arenaAlignment, alignment.getInt());
  });
  if (mlir::failed(arenaValid))
    return mlir::failure();

  if (arenaBytes > 0) {
    prepared.defaultDDRArenaArgumentIndex = function.getNumArguments();
    function.insertArgument(prepared.defaultDDRArenaArgumentIndex,
                            mlir::IntegerType::get(function.getContext(), 64),
                            mlir::DictionaryAttr{}, function.getLoc());
    prepared.slots.push_back({static_cast<int64_t>(prepared.slots.size()),
                              KernelABISlotRole::Workspace,
                              0,
                              "default_ddr_arena",
                              "u8",
                              {arenaBytes},
                              arenaBytes,
                              arenaAlignment});
  }

  if (mlir::failed(mlir::verify(*prepared.module)))
    return mlir::failure();
  return prepared;
}

mlir::LogicalResult lowerToTargetLLVM(PreparedTargetRank &prepared) {
  LowerInstrToTargetLLVMPassOptions options;
  options.defaultDDRArenaArgumentIndex = prepared.defaultDDRArenaArgumentIndex;
  mlir::PassManager manager(prepared.module->getContext());
  manager.addPass(createLowerInstrToTargetLLVMPass(options));
  return manager.run(*prepared.module);
}

mlir::LogicalResult verifyLoweredKernelABI(PreparedTargetRank &prepared,
                                           llvm::StringRef entrySymbol) {
  auto entry =
      prepared.module->lookupSymbol<mlir::LLVM::LLVMFuncOp>(entrySymbol);
  if (!entry || entry.isDeclaration())
    return prepared.module->emitError()
           << "target_abi_mismatch: lowered entry is missing or external";
  mlir::LLVM::LLVMFunctionType type = entry.getFunctionType();
  if (type.isVarArg() ||
      !mlir::isa<mlir::LLVM::LLVMVoidType>(type.getReturnType()) ||
      type.getNumParams() != prepared.slots.size() ||
      !llvm::all_of(type.getParams(), [](mlir::Type parameter) {
        return parameter.isInteger(64);
      }))
    return entry.emitError()
           << "target_abi_mismatch: lowered entry must be fixed void(i64...) "
              "with one parameter per typed Kernel ABI slot";
  return mlir::success();
}

llvm::Error writeLLVMIR(mlir::ModuleOp module, llvm::StringRef path) {
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext, "wafer_rank");
  if (!llvmModule)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target LLVM IR translation failed");
  llvmModule->setTargetTriple("riscv64-unknown-unknown-elf");
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to open target LLVM IR");
  llvmModule->print(output, nullptr);
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

llvm::Expected<std::string> verifyTargetModule(llvm::StringRef path,
                                               llvm::StringRef entrySymbol) {
  if (!isRegularFile(path))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target module is not a regular file");
  auto buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                            /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read target module");
  llvm::StringRef bytes = (*buffer)->getBuffer();
  if (bytes.size() < 4 || !bytes.starts_with("\x7f"
                                             "ELF"))
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target module is not ELF");

  auto object = llvm::object::ObjectFile::createObjectFile(path);
  if (!object)
    return object.takeError();
  if ((*object).getBinary()->makeTriple().getArch() != llvm::Triple::riscv64)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target module is not RISC-V 64-bit ELF");
  bool foundEntry = false;
  for (llvm::object::SymbolRef symbol : (*object).getBinary()->symbols()) {
    llvm::Expected<llvm::StringRef> name = symbol.getName();
    llvm::Expected<uint32_t> flags = symbol.getFlags();
    if (!name || !flags) {
      if (!name)
        llvm::consumeError(name.takeError());
      if (!flags)
        llvm::consumeError(flags.takeError());
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "target module symbol readback failed");
    }
    if (*name == entrySymbol &&
        !(*flags & llvm::object::BasicSymbolRef::SF_Undefined)) {
      foundEntry = true;
      break;
    }
  }
  if (!foundEntry)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "target entry symbol is not defined");

  llvm::SHA256 hasher;
  hasher.update(bytes);
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

bool publishDirectoryNoReplace(llvm::StringRef source,
                               llvm::StringRef destination,
                               llvm::raw_ostream &diagnostics) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return false;
  int errorNumber = errno;
  diagnostics << "wafer-compile: target_publication_failed: "
              << std::error_code(errorNumber, std::generic_category()).message()
              << "\n";
  return true;
#else
  (void)source;
  (void)destination;
  diagnostics << "wafer-compile: target_publication_failed: atomic no-replace "
                 "directory publication is unsupported on this host\n";
  return true;
#endif
}

} // namespace

mlir::LogicalResult
detail::lowerTargetABIForTesting(const RankExecutable &rankExecutable) {
  mlir::FailureOr<PreparedTargetRank> prepared =
      prepareTargetABI(rankExecutable);
  if (mlir::failed(prepared) || mlir::failed(lowerToTargetLLVM(*prepared)))
    return mlir::failure();
  return verifyLoweredKernelABI(*prepared, rankExecutable.getEntrySymbol());
}

llvm::Expected<TargetArtifactBundle>
detail::compileExecutableBundleToTargetArtifactsImpl(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics,
    std::optional<int64_t> failAfterLogicalRank) {
  if (outputDirectory.empty())
    return fail(diagnostics, "target artifact directory must not be empty");
  if (pathEntryExists(outputDirectory))
    return fail(diagnostics,
                "refusing to replace existing target artifact directory");
  if (!isExecutableFile(toolchain.getPythonExecutable()))
    return fail(diagnostics, "configured Python executable is not executable");
  if (!isRegularFile(toolchain.getDeviceLinkerScript()))
    return fail(diagnostics,
                "configured device linker script is not a regular file");

  llvm::SmallString<256> outputParent(outputDirectory);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";
  if (std::error_code error = llvm::sys::fs::create_directories(outputParent))
    return fail(diagnostics,
                "failed to create target artifact parent: " + error.message());
  llvm::SmallString<256> stagingPrefix(outputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-target-artifacts-staging");
  llvm::SmallString<256> stagingRoot;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingPrefix, stagingRoot))
    return fail(diagnostics,
                "failed to create target artifact staging: " + error.message());
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(stagingRoot); });

  llvm::SmallString<256> modulesDirectory(stagingRoot);
  llvm::sys::path::append(modulesDirectory, "modules");
  llvm::SmallString<256> workDirectory(stagingRoot);
  llvm::sys::path::append(workDirectory, "work");
  if (std::error_code error =
          llvm::sys::fs::create_directories(modulesDirectory))
    return fail(diagnostics,
                "failed to create staged module directory: " + error.message());
  if (std::error_code error = llvm::sys::fs::create_directories(workDirectory))
    return fail(diagnostics,
                "failed to create target work directory: " + error.message());

  const std::vector<RankExecutable> &ranks =
      executableBundle.getRankExecutables();
  if (ranks.size() !=
      static_cast<size_t>(executableBundle.getExecutionConfig().getRankCount()))
    return fail(diagnostics, "target rank domain is incomplete");

  std::vector<VerifiedTargetModule> modules;
  modules.reserve(ranks.size());
  for (auto [expectedRank, rank] : llvm::enumerate(ranks)) {
    if (rank.getLogicalRank() != static_cast<int64_t>(expectedRank))
      return fail(diagnostics, "target rank domain is not canonical");
    mlir::FailureOr<PreparedTargetRank> prepared = prepareTargetABI(rank);
    if (mlir::failed(prepared))
      return fail(diagnostics,
                  "target ABI preparation failed for logical rank " +
                      std::to_string(expectedRank));
    if (mlir::failed(lowerToTargetLLVM(*prepared)))
      return fail(diagnostics, "target lowering failed for logical rank " +
                                   std::to_string(expectedRank));
    if (mlir::failed(verifyLoweredKernelABI(*prepared, rank.getEntrySymbol())))
      return fail(diagnostics,
                  "target ABI verification failed for logical rank " +
                      std::to_string(expectedRank));

    std::string stem = (llvm::formatv("rank_{0:D5}", expectedRank)).str();
    llvm::SmallString<256> llvmIRPath(workDirectory);
    llvm::sys::path::append(llvmIRPath, stem + ".ll");
    llvm::SmallString<256> objectPath(workDirectory);
    llvm::sys::path::append(objectPath, stem + ".o");
    llvm::SmallString<256> crtObjectPath(workDirectory);
    llvm::sys::path::append(crtObjectPath, stem + ".wafer_crt.o");
    llvm::SmallString<256> modulePath(modulesDirectory);
    llvm::sys::path::append(modulePath, stem + ".so");
    if (llvm::Error error = writeLLVMIR(*prepared->module, llvmIRPath))
      return fail(diagnostics, "target_module_verification_failed: " +
                                   llvm::toString(std::move(error)));
    if (llvm::Error error = runDeviceLink(toolchain, llvmIRPath, modulePath,
                                          objectPath, crtObjectPath))
      return fail(diagnostics, "target_module_verification_failed: " +
                                   llvm::toString(std::move(error)));
    llvm::Expected<std::string> digest =
        verifyTargetModule(modulePath, rank.getEntrySymbol());
    if (!digest)
      return fail(diagnostics, "target_module_verification_failed: " +
                                   llvm::toString(digest.takeError()));
    if (failAfterLogicalRank &&
        static_cast<int64_t>(expectedRank) == *failAfterLogicalRank)
      return fail(diagnostics,
                  "test-only injected target failure after logical rank " +
                      std::to_string(expectedRank));

    std::string relativePath = (llvm::Twine("modules/") + stem + ".so").str();
    modules.push_back(TargetArtifactBundleBuilder::makeModule(
        expectedRank, rank.getEntrySymbol(), relativePath, *digest,
        std::move(prepared->slots)));
  }

  if (std::error_code error = llvm::sys::fs::remove_directories(workDirectory))
    return fail(diagnostics,
                "failed to remove target work directory: " + error.message());
  if (publishDirectoryNoReplace(stagingRoot, outputDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::io_error,
                                   "target artifact publication failed");
  cleanup.release();
  return TargetArtifactBundleBuilder::makeBundle(
      outputDirectory, executableBundle.getExecutionConfig(),
      std::move(modules));
}

llvm::Expected<TargetArtifactBundle> compileExecutableBundleToTargetArtifacts(
    const ExecutableBundle &executableBundle, llvm::StringRef outputDirectory,
    const TargetToolchain &toolchain, llvm::raw_ostream &diagnostics) {
  return detail::compileExecutableBundleToTargetArtifactsImpl(
      executableBundle, outputDirectory, toolchain, diagnostics, std::nullopt);
}

} // namespace wafer::compiler
